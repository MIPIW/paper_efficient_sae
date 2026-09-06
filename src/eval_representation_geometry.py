"""Representation geometry analysis for Kron checkpoints (no retraining).

This script compares two frozen checkpoints on pooled branch representations
using probe metrics (linear / kNN / MLP / HSIC) and geometry diagnostics
(participation ratio, discriminative-energy PR, Fisher ratio, direction overlap).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch as t
from nnsight import LanguageModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from data.amazon_reviews import load_prepared_datasets
from dictionary_learning.dictionary_kron import KronAutoEncoderTopK
from eval_joint_probe import (
    balanced_sentiment_subset_indices,
    balanced_topic_subset_indices,
    collect_doc_representations,
    label_count_dict,
    stratified_split_indices,
    train_linear_probe,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


def balanced_subset_indices(labels: t.Tensor, label_name: str, seed: int) -> t.Tensor:
    if label_name == "sentiment":
        return balanced_sentiment_subset_indices(labels, seed=seed)
    if label_name == "topic":
        return balanced_topic_subset_indices(labels, seed=seed)
    raise ValueError(f"Unsupported label_name={label_name}")


def stratified_cap_indices(labels: t.Tensor, max_n: int, seed: int) -> t.Tensor:
    n = labels.shape[0]
    if n <= max_n:
        return t.arange(n, dtype=t.long)

    g = t.Generator()
    g.manual_seed(seed)

    per_class: Dict[int, t.Tensor] = {}
    classes = t.unique(labels).tolist()
    for c in classes:
        idx = (labels == int(c)).nonzero(as_tuple=False).squeeze(-1)
        idx = idx[t.randperm(idx.numel(), generator=g)]
        per_class[int(c)] = idx

    n_classes = len(classes)
    base = max_n // n_classes
    rem = max_n % n_classes

    selected = []
    leftovers = []
    for i, c in enumerate(sorted(per_class.keys())):
        take = min(base + (1 if i < rem else 0), per_class[c].numel())
        selected.append(per_class[c][:take])
        leftovers.append(per_class[c][take:])

    cur = sum(x.numel() for x in selected)
    if cur < max_n:
        extra_need = max_n - cur
        extra_pool = t.cat([x for x in leftovers if x.numel() > 0], dim=0)
        if extra_pool.numel() > 0:
            extra_take = min(extra_need, extra_pool.numel())
            selected.append(extra_pool[:extra_take])

    return t.cat(selected, dim=0)


def knn_accuracy(
    x_train: t.Tensor,
    y_train: t.Tensor,
    x_test: t.Tensor,
    y_test: t.Tensor,
    num_classes: int,
    k: int = 15,
    metric: str = "cosine",
    chunk: int = 512,
) -> float:
    if metric not in {"cosine", "euclidean"}:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    x_train = x_train.float()
    x_test = x_test.float()
    y_train = y_train.long()
    y_test = y_test.long()
    k_eff = max(1, min(k, x_train.shape[0]))

    if metric == "cosine":
        x_train_n = t.nn.functional.normalize(x_train, dim=-1)
        x_test_n = t.nn.functional.normalize(x_test, dim=-1)
    else:
        x_train_n = x_train
        x_test_n = x_test

    preds = []
    for i in range(0, x_test_n.shape[0], chunk):
        xb = x_test_n[i : i + chunk]
        if metric == "cosine":
            score = xb @ x_train_n.T
            nn_idx = score.topk(k_eff, dim=1, largest=True).indices
        else:
            d2 = (
                (xb**2).sum(dim=1, keepdim=True)
                + (x_train_n**2).sum(dim=1, keepdim=True).T
                - 2.0 * (xb @ x_train_n.T)
            )
            nn_idx = (-d2).topk(k_eff, dim=1, largest=True).indices

        nn_y = y_train[nn_idx]
        votes = t.zeros((nn_y.shape[0], num_classes), device=nn_y.device, dtype=t.long)
        votes.scatter_add_(1, nn_y, t.ones_like(nn_y, dtype=t.long))
        preds.append(votes.argmax(dim=1))

    y_pred = t.cat(preds, dim=0)
    return float((y_pred == y_test).float().mean().item())


class MLPProbe(t.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(in_dim, hidden_dim),
            t.nn.ReLU(),
            t.nn.Dropout(dropout),
            t.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x)


def mlp_probe_accuracy(
    x_train: t.Tensor,
    y_train: t.Tensor,
    x_test: t.Tensor,
    y_test: t.Tensor,
    num_classes: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str,
) -> float:
    set_seed(seed)
    model = MLPProbe(x_train.shape[1], hidden_dim, num_classes, dropout).to(device)
    opt = t.optim.Adam(model.parameters(), lr=lr)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    for _ in range(epochs):
        perm = t.randperm(x_train.shape[0], device=device)
        for i in range(0, x_train.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            logits = model(x_train[idx])
            loss = t.nn.functional.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with t.no_grad():
        pred = model(x_test).argmax(dim=-1)
        return float((pred == y_test).float().mean().item())


def _center_kernel(k: t.Tensor) -> t.Tensor:
    row_mean = k.mean(dim=1, keepdim=True)
    col_mean = k.mean(dim=0, keepdim=True)
    grand_mean = k.mean()
    return k - row_mean - col_mean + grand_mean


def hsic_rbf_delta(
    x: t.Tensor,
    y: t.Tensor,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    x = x.float()
    y = y.long()
    n = x.shape[0]
    if n < 4:
        return 0.0, 0.0, 0.0

    xx = (x * x).sum(dim=1, keepdim=True)
    d2 = (xx + xx.T - 2.0 * (x @ x.T)).clamp_min(0.0)

    mask = ~t.eye(n, dtype=t.bool, device=x.device)
    med = t.median(d2[mask])
    sigma2 = float(med.item()) if med.numel() > 0 else 1.0
    if not math.isfinite(sigma2) or sigma2 <= 0.0:
        sigma2 = float(d2[mask].mean().item()) if mask.any() else 1.0
        sigma2 = max(sigma2, 1e-6)

    kx = t.exp(-d2 / (2.0 * sigma2))
    ky = (y[:, None] == y[None, :]).float()

    kxc = _center_kernel(kx)
    kyc = _center_kernel(ky)

    denom = float((n - 1) ** 2)
    hsic_xy = float((kxc * kyc).sum().item() / denom)
    hsic_xx = float((kxc * kxc).sum().item() / denom)
    hsic_yy = float((kyc * kyc).sum().item() / denom)
    hsic_norm = hsic_xy / math.sqrt(max(hsic_xx * hsic_yy, eps))
    return hsic_xy, hsic_norm, sigma2


def participation_ratio_from_cov(x: t.Tensor, eps: float = 1e-12) -> float:
    x = x.float()
    if x.shape[0] < 2:
        return 0.0
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    eig = t.linalg.eigvalsh(cov).clamp_min(0.0)
    num = eig.sum() ** 2
    den = (eig**2).sum() + eps
    return float((num / den).item())


def discriminative_energy_vector(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    x = x.float()
    y = y.long()
    mu = x.mean(dim=0, keepdim=True)
    e = t.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    for cls in t.unique(y):
        idx = (y == cls).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        mu_c = x[idx].mean(dim=0, keepdim=True)
        diff = (mu_c - mu).squeeze(0)
        e = e + diff * diff
    return e


def participation_ratio_from_energy(e: t.Tensor, eps: float = 1e-12) -> float:
    e = e.float().clamp_min(0.0)
    num = e.sum() ** 2
    den = (e * e).sum() + eps
    return float((num / den).item())


def fisher_trace_ratio(x: t.Tensor, y: t.Tensor, eps: float = 1e-12) -> float:
    x = x.float()
    y = y.long()
    mu = x.mean(dim=0, keepdim=True)
    sb = 0.0
    sw = 0.0
    for cls in t.unique(y):
        idx = (y == cls).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        xc = x[idx]
        mu_c = xc.mean(dim=0, keepdim=True)
        sb += float(idx.numel()) * float(((mu_c - mu) ** 2).sum().item())
        sw += float(((xc - mu_c) ** 2).sum().item())
    return sb / max(sw, eps)


def sentiment_direction(x: t.Tensor, y_sent: t.Tensor) -> t.Tensor:
    x = x.float()
    y_sent = y_sent.long()
    idx0 = (y_sent == 0).nonzero(as_tuple=False).squeeze(-1)
    idx1 = (y_sent == 1).nonzero(as_tuple=False).squeeze(-1)
    if idx0.numel() == 0 or idx1.numel() == 0:
        return t.zeros(x.shape[1], dtype=x.dtype)
    d = x[idx1].mean(dim=0) - x[idx0].mean(dim=0)
    n = d.norm() + 1e-12
    return d / n


def topic_direction(x: t.Tensor, y_topic: t.Tensor) -> t.Tensor:
    x = x.float()
    y_topic = y_topic.long()
    classes = t.unique(y_topic)
    if classes.numel() < 2:
        return t.zeros(x.shape[1], dtype=x.dtype)
    mu = x.mean(dim=0, keepdim=True)
    rows = []
    for cls in classes:
        idx = (y_topic == cls).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        rows.append((x[idx].mean(dim=0, keepdim=True) - mu).squeeze(0))
    if not rows:
        return t.zeros(x.shape[1], dtype=x.dtype)
    m = t.stack(rows, dim=0)
    _, _, vh = t.linalg.svd(m, full_matrices=False)
    d = vh[0]
    n = d.norm() + 1e-12
    return d / n


def abs_cosine(a: t.Tensor, b: t.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = (a.norm() * b.norm()).item()
    if denom <= 0:
        return 0.0
    return float(abs(t.dot(a, b).item() / denom))


@dataclass
class CheckpointBundle:
    name: str
    ckpt_path: Path
    trainer_config: dict
    run_metadata: dict
    run_root: Path
    trainer_name: str
    valid_labels: Set[str]


def load_checkpoint_bundle(ckpt_path: str, name: str | None) -> CheckpointBundle:
    ckpt = Path(ckpt_path).resolve()
    trainer_dir = ckpt.parent
    cfg_path = trainer_dir / "trainer_config.json"
    run_root = trainer_dir.parent.parent
    meta_path = run_root / "run_metadata.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing trainer_config.json: {cfg_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run_metadata.json: {meta_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if cfg.get("dict_class") != "KronAutoEncoderTopK":
        raise ValueError(f"Only Kron checkpoints supported, got dict_class={cfg.get('dict_class')}")

    trainer_name = trainer_dir.name
    bundle_name = name or f"{run_root.name}:{trainer_name}"
    valid_labels = infer_valid_labels_from_trainer_name(trainer_name)
    return CheckpointBundle(
        name=bundle_name,
        ckpt_path=ckpt,
        trainer_config=cfg,
        run_metadata=meta,
        run_root=run_root,
        trainer_name=trainer_name,
        valid_labels=valid_labels,
    )


def infer_valid_labels_from_trainer_name(trainer_name: str) -> Set[str]:
    if trainer_name == "kron_pilot_sentiment":
        return {"sentiment"}
    if trainer_name == "kron_pilot_topic":
        return {"topic"}
    if trainer_name == "kron_joint":
        return {"sentiment", "topic"}
    return {"sentiment", "topic"}


def load_saved_linear_context(bundle: CheckpointBundle) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}

    if bundle.run_root.name == "pilot_balanced_v2_420k":
        probe_path = bundle.run_root / "pilot_probe_results.json"
        if not probe_path.exists():
            return out
        obj = json.loads(probe_path.read_text(encoding="utf-8"))
        row = None
        for r in obj.get("results", []):
            if r.get("run_name") == bundle.trainer_name:
                row = r
                break
        if row is None:
            return out
        label_type = row.get("label_type")
        if label_type == "sentiment":
            out["sentiment"] = {
                "p": float(row.get("p_probe_accuracy", float("nan"))),
                "q": float(row.get("q_probe_accuracy", float("nan"))),
            }
        elif label_type == "topic":
            out["topic"] = {
                "p": float(row.get("p_probe_accuracy", float("nan"))),
                "q": float(row.get("q_probe_accuracy", float("nan"))),
            }
        return out

    if bundle.run_root.name == "joint_dpo_cross_full420k":
        probe_path = bundle.run_root / "joint_probe_results.json"
        if not probe_path.exists():
            return out
        obj = json.loads(probe_path.read_text(encoding="utf-8"))
        row = None
        for r in obj.get("results", []):
            if r.get("run_name") == bundle.trainer_name:
                row = r
                break
        if row is None:
            return out
        sent = row.get("sentiment_eval", {}).get("branch_accuracy", {})
        topic = row.get("topic_eval", {}).get("branch_accuracy", {})
        out["sentiment"] = {"p": float(sent.get("p", float("nan"))), "q": float(sent.get("q", float("nan")))}
        out["topic"] = {"p": float(topic.get("p", float("nan"))), "q": float(topic.get("q", float("nan")))}
        return out

    return out


def build_kron_from_config(cfg: dict, ckpt_path: Path, device: str) -> KronAutoEncoderTopK:
    ae = KronAutoEncoderTopK(
        activation_dim=cfg["activation_dim"],
        h=cfg["h"],
        m=cfg["m"],
        n=cfg["n"],
        k=cfg["k"],
        combine_rule=cfg["combine_rule"],
    )
    state = t.load(ckpt_path, map_location="cpu")
    ae.load_state_dict(state)
    ae.to(device)
    ae.eval()
    return ae


def evaluate_label_metrics(
    x_branch: t.Tensor,
    labels: t.Tensor,
    label_name: str,
    seed: int,
    test_fraction: float,
    knn_k: int,
    mlp_hidden: int,
    mlp_dropout: float,
    mlp_epochs: int,
    mlp_lr: float,
    probe_batch_size: int,
    device: str,
    hsic_max_samples: int,
    linear_override: Optional[float] = None,
    compute_linear_probe: bool = False,
) -> dict:
    subset_idx = balanced_subset_indices(labels, label_name, seed=seed)
    x_bal = x_branch[subset_idx]
    y_bal = labels[subset_idx]

    train_idx, test_idx = stratified_split_indices(y_bal, test_fraction=test_fraction, seed=seed + 11)
    if train_idx.numel() == 0 or test_idx.numel() == 0:
        raise RuntimeError(f"Empty train/test split for label={label_name}")

    x_train = x_bal[train_idx]
    y_train = y_bal[train_idx]
    x_test = x_bal[test_idx]
    y_test = y_bal[test_idx]

    num_classes = int(y_bal.max().item()) + 1
    linear_acc: float
    if linear_override is not None and not compute_linear_probe:
        linear_acc = float(linear_override)
    else:
        linear_acc = train_linear_probe(
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            epochs=30,
            lr=1e-2,
            batch_size=256,
            seed=seed + 1,
            device=device,
        )
    knn_acc = knn_accuracy(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        k=knn_k,
        metric="cosine",
    )
    mlp_acc = mlp_probe_accuracy(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        hidden_dim=mlp_hidden,
        dropout=mlp_dropout,
        epochs=mlp_epochs,
        lr=mlp_lr,
        batch_size=probe_batch_size,
        seed=seed + 2,
        device=device,
    )

    hsic_idx_local = stratified_cap_indices(y_bal, max_n=hsic_max_samples, seed=seed + 17)
    x_hsic = x_bal[hsic_idx_local].to(device)
    y_hsic = y_bal[hsic_idx_local].to(device)
    hsic_raw, hsic_norm, sigma2 = hsic_rbf_delta(x_hsic, y_hsic)

    return {
        "num_classes": num_classes,
        "balanced_subset_counts": label_count_dict(y_bal),
        "probe_train_counts": label_count_dict(y_train),
        "probe_test_counts": label_count_dict(y_test),
        "balanced_subset_n": int(y_bal.shape[0]),
        "train_n": int(y_train.shape[0]),
        "test_n": int(y_test.shape[0]),
        "hsic_n": int(y_hsic.shape[0]),
        "linear_probe_acc": float(linear_acc),
        "linear_probe_source": "saved_probe_results" if (linear_override is not None and not compute_linear_probe) else "computed_now",
        "knn_probe_acc": float(knn_acc),
        "mlp_probe_acc": float(mlp_acc),
        "hsic_rbf_delta": float(hsic_raw),
        "hsic_normalized_cka": float(hsic_norm),
        "hsic_rbf_sigma2": float(sigma2),
    }


def evaluate_branch_geometry(
    x_branch: t.Tensor,
    topic_labels: t.Tensor,
    sentiment_labels: t.Tensor,
    seed: int,
    valid_labels: Set[str],
) -> dict:
    out = {
        "overall_n": int(x_branch.shape[0]),
        "feature_dim": int(x_branch.shape[1]),
        "overall_participation_ratio": participation_ratio_from_cov(x_branch),
    }

    label_geom = {}
    for label_name, labels, seed_off in (
        ("sentiment", sentiment_labels, 101),
        ("topic", topic_labels, 202),
    ):
        if label_name not in valid_labels:
            continue
        idx = balanced_subset_indices(labels, label_name, seed=seed + seed_off)
        xb = x_branch[idx]
        yb = labels[idx]
        e = discriminative_energy_vector(xb, yb)
        label_geom[label_name] = {
            "n": int(yb.shape[0]),
            "counts": label_count_dict(yb),
            "discriminative_energy_pr": participation_ratio_from_energy(e),
            "fisher_trace_ratio": fisher_trace_ratio(xb, yb),
        }
    out["label_geometry"] = label_geom

    if {"sentiment", "topic"}.issubset(valid_labels):
        sent_idx = balanced_subset_indices(sentiment_labels, "sentiment", seed=seed + 301)
        topic_idx = balanced_subset_indices(topic_labels, "topic", seed=seed + 302)
        w_sent = sentiment_direction(x_branch[sent_idx], sentiment_labels[sent_idx])
        w_topic = topic_direction(x_branch[topic_idx], topic_labels[topic_idx])
        out["sentiment_topic_direction_abs_cosine"] = abs_cosine(w_sent, w_topic)
    else:
        out["sentiment_topic_direction_abs_cosine"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Representation geometry analysis on frozen Kron checkpoints.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            str(ROOT / "runs" / "pilot_balanced_v2_420k" / "checkpoints" / "kron_pilot_sentiment" / "ae_step_420000.pt"),
            str(ROOT / "runs" / "pilot_balanced_v2_420k" / "checkpoints" / "kron_pilot_topic" / "ae_step_420000.pt"),
            str(ROOT / "runs" / "joint_dpo_cross_full420k" / "checkpoints" / "kron_joint" / "ae_step_420000.pt"),
        ],
    )
    parser.add_argument(
        "--checkpoint_names",
        nargs="*",
        default=["pilot_sentiment_control", "pilot_topic_control", "joint_failure_case"],
    )
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--max_probe_docs", type=int, default=12000)
    parser.add_argument("--doc_batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--knn_k", type=int, default=15)
    parser.add_argument("--mlp_hidden", type=int, default=128)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_epochs", type=int, default=30)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--hsic_max_samples", type=int, default=4000)
    parser.add_argument(
        "--compute_linear_probe",
        action="store_true",
        help="If set, recompute linear probe now; default uses saved established probe JSON when available.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(ROOT / "runs" / "representation_geometry_results.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda:0" if t.cuda.is_available() else "cpu")

    if args.checkpoint_names and len(args.checkpoint_names) not in {0, len(args.checkpoints)}:
        raise ValueError("--checkpoint_names must be empty or match --checkpoints length")

    bundles = []
    for i, ckpt in enumerate(args.checkpoints):
        nm = args.checkpoint_names[i] if args.checkpoint_names and i < len(args.checkpoint_names) else None
        bundles.append(load_checkpoint_bundle(ckpt, nm))

    # Shared model/dataset assumptions.
    train_args0 = bundles[0].run_metadata["args"]
    model_name = train_args0["model_name"]
    layer = train_args0["layer"]
    ctx_len = train_args0["ctx_len"]
    for b in bundles[1:]:
        ta = b.run_metadata["args"]
        if ta["model_name"] != model_name or ta["layer"] != layer or ta["ctx_len"] != ctx_len:
            raise ValueError("Checkpoints must share model_name/layer/ctx_len for this comparison script")

    dataset_cache_dir = args.dataset_cache_dir or train_args0["dataset_cache_dir"]
    _, eval_ds, _, _ = load_prepared_datasets(dataset_cache_dir)

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    submodule = model.gpt_neox.layers[layer]

    output = {
        "settings": {
            "device": device,
            "dataset_cache_dir": str(dataset_cache_dir),
            "model_name": model_name,
            "layer": layer,
            "ctx_len": ctx_len,
            "max_probe_docs": args.max_probe_docs,
            "doc_batch_size": args.doc_batch_size,
            "num_workers": args.num_workers,
            "test_fraction": args.test_fraction,
            "knn_k": args.knn_k,
            "mlp_hidden": args.mlp_hidden,
            "mlp_dropout": args.mlp_dropout,
            "mlp_epochs": args.mlp_epochs,
            "mlp_lr": args.mlp_lr,
            "probe_batch_size": args.probe_batch_size,
            "hsic_max_samples": args.hsic_max_samples,
            "hsic_kernel_x": "rbf (median heuristic)",
            "hsic_kernel_y": "delta(label match)",
            "hsic_normalization": "centered kernel alignment (CKA-like)",
            "linear_probe_mode": "compute_now" if args.compute_linear_probe else "reuse_saved_when_available",
        },
        "checkpoints": [],
    }

    for b_idx, bundle in enumerate(bundles):
        ae = build_kron_from_config(bundle.trainer_config, bundle.ckpt_path, device=device)
        reps, topic_labels, sentiment_labels = collect_doc_representations(
            ae=ae,
            dict_class="KronAutoEncoderTopK",
            model=model,
            submodule=submodule,
            eval_ds=eval_ds,
            ctx_len=ctx_len,
            doc_batch_size=args.doc_batch_size,
            num_workers=args.num_workers,
            max_docs=args.max_probe_docs,
            device=device,
        )

        ckpt_result = {
            "name": bundle.name,
            "checkpoint_path": str(bundle.ckpt_path),
            "run_root": str(bundle.run_root),
            "dict_size": int(bundle.trainer_config["dict_size"]),
            "h": int(bundle.trainer_config["h"]),
            "m": int(bundle.trainer_config["m"]),
            "n": int(bundle.trainer_config["n"]),
            "trainer_name": bundle.trainer_name,
            "valid_labels": sorted(bundle.valid_labels),
            "num_docs_collected": int(topic_labels.shape[0]),
            "branch_results": {},
        }
        linear_ctx = load_saved_linear_context(bundle)

        for branch in ("p", "q"):
            xb = reps[branch]
            branch_out = {
                "geometry": evaluate_branch_geometry(
                    x_branch=xb,
                    topic_labels=topic_labels,
                    sentiment_labels=sentiment_labels,
                    seed=args.seed + 1000 * b_idx + (0 if branch == "p" else 100),
                    valid_labels=bundle.valid_labels,
                ),
                "labels": {},
            }

            label_specs = []
            if "sentiment" in bundle.valid_labels:
                label_specs.append(("sentiment", sentiment_labels, args.seed + 10 + 1000 * b_idx + (0 if branch == "p" else 100)))
            if "topic" in bundle.valid_labels:
                label_specs.append(("topic", topic_labels, args.seed + 20 + 1000 * b_idx + (0 if branch == "p" else 100)))
            for label_name, labels, seed_lab in label_specs:
                linear_override = None
                if label_name in linear_ctx:
                    linear_override = linear_ctx[label_name].get(branch)
                branch_out["labels"][label_name] = evaluate_label_metrics(
                    x_branch=xb,
                    labels=labels,
                    label_name=label_name,
                    seed=seed_lab,
                    test_fraction=args.test_fraction,
                    knn_k=args.knn_k,
                    mlp_hidden=args.mlp_hidden,
                    mlp_dropout=args.mlp_dropout,
                    mlp_epochs=args.mlp_epochs,
                    mlp_lr=args.mlp_lr,
                    probe_batch_size=args.probe_batch_size,
                    device=device,
                    hsic_max_samples=args.hsic_max_samples,
                    linear_override=linear_override,
                    compute_linear_probe=args.compute_linear_probe,
                )

            ckpt_result["branch_results"][branch] = branch_out

        output["checkpoints"].append(ckpt_result)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("Saved representation geometry analysis to:", out_path)
    for ck in output["checkpoints"]:
        print(f"\n[{ck['name']}] {ck['checkpoint_path']}")
        for branch in ("p", "q"):
            br = ck["branch_results"][branch]
            g = br["geometry"]
            overlap = g["sentiment_topic_direction_abs_cosine"]
            overlap_str = "n/a" if overlap is None else f"{overlap:.4f}"
            print(
                f"  branch={branch} | overall_PR={g['overall_participation_ratio']:.2f} | "
                f"dir_overlap_abs_cos={overlap_str}"
            )
            for label_name in ("sentiment", "topic"):
                if label_name not in br["labels"]:
                    continue
                m = br["labels"][label_name]
                lg = g["label_geometry"][label_name]
                print(
                    f"    {label_name}: linear={m['linear_probe_acc']:.4f}, knn={m['knn_probe_acc']:.4f}, "
                    f"mlp={m['mlp_probe_acc']:.4f}, hsic={m['hsic_rbf_delta']:.6f}, hsic_norm={m['hsic_normalized_cka']:.4f}, "
                    f"discPR={lg['discriminative_energy_pr']:.2f}, fisher={lg['fisher_trace_ratio']:.6f}, "
                    f"n_bal={m['balanced_subset_n']}, n_test={m['test_n']}, n_hsic={m['hsic_n']}"
                )


if __name__ == "__main__":
    main()
