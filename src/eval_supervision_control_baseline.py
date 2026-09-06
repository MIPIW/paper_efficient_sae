"""Supervision control experiment: what do the probes read WITHOUT supervision?

Three arms, evaluated with one identical pipeline (same cached held-out token
activations, same doc mean-pooling, same balanced subsets, same stratified
splits, same probe configuration):

  Arm A  raw    -- pythia-410m layer-12 residual activations, doc mean-pooled.
                   No SAE at all. Plus a matched-dimension control
                   (Gaussian random projection 1024 -> 2048).
  Arm B  rand   -- freshly initialized, never-trained KronSAE (h=128,m=8,n=16,k=24).
                   Controls for "any random projection preserves linear info".
  Arm C  trained-- the existing 420k checkpoints, re-probed identically.

For every (representation, label) we report, across `--n_seeds` data seeds
(each seed re-draws the balanced subset, the train/test split, and the probe
init):
  * std_logreg : sklearn LogisticRegression(max_iter=5000, C=1.0) on
                 train-z-scored features  <- the CORRECTED methodology
  * std_mlp    : MLPProbe on train-z-scored features
  * raw_adam   : the ORIGINAL (broken) fixed-LR Adam linear probe on
                 unstandardized features, shown so the size of the
                 methodological artifact is visible per arm.

Dimensionality controls (a linear probe gets more capacity from more dims
regardless of content):
  * raw_rp2048   : raw 1024-d projected UP to 2048 by a random Gaussian matrix
  * <q>_pca1024  : any 2048-d Q branch projected DOWN to 1024 by PCA fit on the
                   TRAIN split only (no label leakage)
  * <q>_sub1024  : same Q branch cut to a fixed random 1024-dim coordinate subset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch as t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from dictionary_learning.dictionary_kron import KronAutoEncoderTopK
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc
from eval_joint_probe import stratified_split_indices, train_linear_probe
from eval_representation_geometry import (
    balanced_subset_indices,
    build_kron_from_config,
    mlp_probe_accuracy,
    set_seed,
)

PILOT = ROOT / "runs" / "pilot_balanced_v2_420k" / "checkpoints"
JOINT = ROOT / "runs" / "joint_dpo_cross_full420k" / "checkpoints"


# ---------------------------------------------------------------------------
# representation builders (each returns a (n_docs, d) float32 CPU tensor)
# ---------------------------------------------------------------------------
def pooled_raw(batches, device) -> t.Tensor:
    out = []
    with t.no_grad():
        for b in batches:
            x = b["x"].to(device=device, dtype=t.float32)
            ids = b["token_doc_ids"].to(device)
            out.append(_mean_pool_by_doc(x, ids, b["n_docs"]).cpu())
    return t.cat(out)


def pooled_kron_branches(ae, batches, device) -> Dict[str, t.Tensor]:
    p_all, q_all = [], []
    with t.no_grad():
        for b in batches:
            x = b["x"].to(device=device, dtype=t.float32)
            _, _, p_pos, q_pos = ae._prelatents(x)
            ids = b["token_doc_ids"].to(device)
            p_all.append(_mean_pool_by_doc(p_pos.reshape(p_pos.shape[0], -1), ids, b["n_docs"]).cpu())
            q_all.append(_mean_pool_by_doc(q_pos.reshape(q_pos.shape[0], -1), ids, b["n_docs"]).cpu())
    return {"p": t.cat(p_all), "q": t.cat(q_all)}


def pooled_flat(ae, batches, device) -> t.Tensor:
    """Matches eval_joint_probe.collect_doc_representations for flat SAEs:
    relu(encoder(x - b_dec)), pre-top-k, doc mean-pooled."""
    out = []
    with t.no_grad():
        for b in batches:
            x = b["x"].to(device=device, dtype=t.float32)
            post_relu = t.nn.functional.relu(ae.encoder(x - ae.b_dec))
            ids = b["token_doc_ids"].to(device)
            out.append(_mean_pool_by_doc(post_relu, ids, b["n_docs"]).cpu())
    return t.cat(out)


def build_flat_from_config(cfg: dict, ckpt_path: Path, device: str) -> AutoEncoderTopK:
    ae = AutoEncoderTopK(activation_dim=cfg["activation_dim"], dict_size=cfg["dict_size"], k=cfg["k"])
    ae.load_state_dict(t.load(ckpt_path, map_location="cpu"))
    ae.to(device)
    ae.eval()
    return ae


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------
def probe_rep(
    x_rep: t.Tensor,
    sent_labels: t.Tensor,
    topic_labels: t.Tensor,
    args,
    device: str,
    reduce_to: int | None = None,
    reduce_mode: str = "pca",
) -> Dict[str, dict]:
    """Run the full probe battery for both labels over `args.n_seeds` seeds.

    `reduce_to` (with reduce_mode='pca') fits PCA on the TRAIN split only and
    projects both splits, so the dimensionality control carries no label or
    test-set leakage.
    """
    from sklearn.linear_model import LogisticRegression

    results: Dict[str, dict] = {}
    for label_name, labels in (("sentiment", sent_labels), ("topic", topic_labels)):
        per_seed = []
        for s in range(args.n_seeds):
            seed = args.seed + 1000 * s
            idx = balanced_subset_indices(labels, label_name, seed=seed)
            x_bal, y_bal = x_rep[idx].float(), labels[idx]
            tr, te = stratified_split_indices(y_bal, test_fraction=args.test_fraction, seed=seed + 11)
            x_tr_raw, x_te_raw = x_bal[tr], x_bal[te]
            y_tr, y_te = y_bal[tr], y_bal[te]
            nc = int(y_bal.max().item()) + 1

            # --- old / broken probe: unstandardized features, fixed-LR Adam ---
            raw_acc = train_linear_probe(
                X_train=x_tr_raw, y_train=y_tr, X_test=x_te_raw, y_test=y_te,
                num_classes=nc, epochs=30, lr=1e-2, batch_size=256,
                seed=seed + 1, device=device,
            )

            # --- corrected probes: train-z-score (+ optional PCA on train) ---
            mu = x_tr_raw.mean(0, keepdim=True)
            sd = x_tr_raw.std(0, keepdim=True).clamp_min(1e-6)
            x_tr = (x_tr_raw - mu) / sd
            x_te = (x_te_raw - mu) / sd

            if reduce_to is not None and reduce_to < x_tr.shape[1] and reduce_mode == "pca":
                g_tr, g_te = x_tr.to(device), x_te.to(device)
                cen = g_tr.mean(0, keepdim=True)
                _, _, v = t.pca_lowrank(g_tr - cen, q=min(reduce_to, min(g_tr.shape) - 1), niter=4)
                x_tr = ((g_tr - cen) @ v).cpu()
                x_te = ((g_te - cen) @ v).cpu()
                del g_tr, g_te
                t.cuda.empty_cache()
                mu2 = x_tr.mean(0, keepdim=True)
                sd2 = x_tr.std(0, keepdim=True).clamp_min(1e-6)
                x_tr, x_te = (x_tr - mu2) / sd2, (x_te - mu2) / sd2

            clf = LogisticRegression(max_iter=5000, C=1.0)
            clf.fit(x_tr.numpy(), y_tr.numpy())
            logreg_acc = float((clf.predict(x_te.numpy()) == y_te.numpy()).mean())
            converged = bool(int(np.max(clf.n_iter_)) < 5000)

            mlp_acc = mlp_probe_accuracy(
                x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te,
                num_classes=nc, hidden_dim=args.mlp_hidden, dropout=args.mlp_dropout,
                epochs=args.mlp_epochs, lr=args.mlp_lr, batch_size=args.probe_batch_size,
                seed=seed + 2, device=device,
            )
            per_seed.append(
                {
                    "seed": seed,
                    "n_balanced": int(y_bal.shape[0]),
                    "n_train": int(y_tr.shape[0]),
                    "n_test": int(y_te.shape[0]),
                    "probe_dim": int(x_tr.shape[1]),
                    "raw_adam_acc": float(raw_acc),
                    "std_logreg_acc": logreg_acc,
                    "std_logreg_converged": converged,
                    "std_mlp_acc": float(mlp_acc),
                }
            )

        def agg(key: str) -> dict:
            v = np.array([r[key] for r in per_seed], dtype=float)
            return {
                "mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "max": float(v.max()),
            }

        results[label_name] = {
            "feature_dim": int(x_rep.shape[1]),
            "probe_dim": per_seed[0]["probe_dim"],
            "n_balanced": per_seed[0]["n_balanced"],
            "n_train": per_seed[0]["n_train"],
            "n_test": per_seed[0]["n_test"],
            "chance": 0.5 if label_name == "sentiment" else 1.0 / 6.0,
            "std_logreg": agg("std_logreg_acc"),
            "std_mlp": agg("std_mlp_acc"),
            "raw_adam": agg("raw_adam_acc"),
            "all_logreg_converged": all(r["std_logreg_converged"] for r in per_seed),
            "per_seed": per_seed,
        }
    return results


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation_cache", type=str, required=True)
    ap.add_argument("--max_probe_docs", type=int, default=12000)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--test_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init_seed", type=int, default=42, help="seed for random-init SAE and random projections")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_dropout", type=float, default=0.2)
    ap.add_argument("--mlp_epochs", type=int, default=30)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--probe_batch_size", type=int, default=256)
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "Cflat"])
    ap.add_argument("--output_json", type=str, default=str(ROOT / "runs" / "supervision_control_baseline.json"))
    args = ap.parse_args()

    device = args.device
    set_seed(args.seed)
    batches = t.load(args.activation_cache, map_location="cpu")
    n_docs_total = sum(b["n_docs"] for b in batches)
    n_tokens = sum(b["x"].shape[0] for b in batches)
    print(f"cache: {len(batches)} batches | {n_docs_total} docs | {n_tokens} tokens", flush=True)

    md = args.max_probe_docs
    topic = t.cat([b["topic"] for b in batches])[:md]
    sent = t.cat([b["sentiment"] for b in batches])[:md]

    out = {
        "settings": {
            "device": device,
            "activation_cache": args.activation_cache,
            "max_probe_docs": md,
            "n_seeds": args.n_seeds,
            "seed": args.seed,
            "init_seed": args.init_seed,
            "test_fraction": args.test_fraction,
            "n_docs_cached": n_docs_total,
            "n_tokens_cached": n_tokens,
            "corrected_probe": "train-z-scored features; sklearn LogisticRegression(max_iter=5000, C=1.0); "
            "MLPProbe(hidden=128, dropout=0.2, 30 epochs, Adam lr=1e-3, bs=256) on the same z-scored features",
            "broken_probe": "unstandardized features; torch Linear + Adam lr=1e-2, 30 epochs, bs=256 (identical to §6/§10)",
            "seeds_vary": "balanced subset draw, stratified train/test split, and probe initialization",
            "chance": {"sentiment": 0.5, "topic": 1.0 / 6.0},
        },
        "arms": {},
    }

    def record(arm: str, key: str, x_rep: t.Tensor, note: str, reduce_to=None):
        t0 = time.time()
        res = probe_rep(x_rep, sent, topic, args, device, reduce_to=reduce_to)
        out["arms"].setdefault(arm, {})[key] = {"note": note, "labels": res}
        for ln in ("sentiment", "topic"):
            r = res[ln]
            print(
                f"  [{arm}] {key:34s} {ln:9s} d={r['probe_dim']:5d} | "
                f"logreg {r['std_logreg']['mean']:.4f}+-{r['std_logreg']['std']:.4f} "
                f"[{r['std_logreg']['min']:.3f},{r['std_logreg']['max']:.3f}] | "
                f"mlp {r['std_mlp']['mean']:.4f}+-{r['std_mlp']['std']:.4f} | "
                f"rawAdam {r['raw_adam']['mean']:.4f}+-{r['raw_adam']['std']:.4f} "
                f"[{r['raw_adam']['min']:.3f},{r['raw_adam']['max']:.3f}] "
                f"conv={r['all_logreg_converged']}",
                flush=True,
            )
        print(f"    ({time.time()-t0:.0f}s)", flush=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------------- Arm A: raw LM activations ----------------
    if "A" in args.arms:
        print("\n=== Arm A: raw pythia-410m layer-12 activations ===", flush=True)
        raw = pooled_raw(batches, device)[:md]
        record("A_raw", "raw_act_1024", raw, "doc mean-pooled residual stream, layer 12, no SAE")
        g = t.Generator().manual_seed(args.init_seed)
        proj = t.randn(raw.shape[1], 2048, generator=g) / np.sqrt(raw.shape[1])
        record("A_raw", "raw_act_randproj2048", raw @ proj,
               "DIM CONTROL: raw activations Gaussian-random-projected 1024 -> 2048 (matches Q's dim)")
        del raw

    # ---------------- Arm B: random-init, untrained KronSAE ----------------
    if "B" in args.arms:
        print("\n=== Arm B: randomly-initialized, untrained KronSAE (h=128,m=8,n=16,k=24) ===", flush=True)
        set_seed(args.init_seed)
        ae = KronAutoEncoderTopK(activation_dim=1024, h=128, m=8, n=16, k=24, combine_rule="mand").to(device).eval()
        reps = pooled_kron_branches(ae, batches, device)
        del ae
        t.cuda.empty_cache()
        record("B_randinit_kron", "P_1024", reps["p"][:md], "untrained KronSAE P branch (h*m=1024)")
        record("B_randinit_kron", "Q_2048", reps["q"][:md], "untrained KronSAE Q branch (h*n=2048)")
        record("B_randinit_kron", "Q_pca1024", reps["q"][:md],
               "DIM CONTROL: Q reduced to 1024 dims by PCA fit on the train split", reduce_to=1024)
        del reps
        set_seed(args.seed)

    # ---------------- Arm C: trained Kron checkpoints ----------------
    kron_ckpts: List[Tuple[str, Path]] = [
        ("kron_pilot_sentiment", PILOT / "kron_pilot_sentiment" / "ae_step_420000.pt"),
        ("kron_pilot_topic", PILOT / "kron_pilot_topic" / "ae_step_420000.pt"),
        ("kron_joint_dpo_cross", JOINT / "kron_joint" / "ae_step_420000.pt"),
    ]
    if "C" in args.arms:
        print("\n=== Arm C: trained KronSAE checkpoints (420k steps) ===", flush=True)
        for name, ck in kron_ckpts:
            cfg = json.loads((ck.parent / "trainer_config.json").read_text())
            ae = build_kron_from_config(cfg, ck, device=device)
            reps = pooled_kron_branches(ae, batches, device)
            del ae
            t.cuda.empty_cache()
            record("C_trained", f"{name}.P_1024", reps["p"][:md], f"{name} P branch")
            record("C_trained", f"{name}.Q_2048", reps["q"][:md], f"{name} Q branch")
            record("C_trained", f"{name}.Q_pca1024", reps["q"][:md],
                   f"DIM CONTROL: {name} Q reduced to 1024 dims by PCA fit on the train split", reduce_to=1024)
            del reps

    # ---------------- Arm C (flat) ----------------
    if "Cflat" in args.arms:
        print("\n=== Arm C: trained flat SAE pilots (dict_size=16384) ===", flush=True)
        for name in ("flat_pilot_sentiment", "flat_pilot_topic"):
            ck = PILOT / name / "ae_step_420000.pt"
            cfg = json.loads((ck.parent / "trainer_config.json").read_text())
            ae = build_flat_from_config(cfg, ck, device=device)
            rep = pooled_flat(ae, batches, device)[:md]
            del ae
            t.cuda.empty_cache()
            record("C_trained_flat", f"{name}.full_16384", rep, f"{name} full pre-top-k feature vector")
            record("C_trained_flat", f"{name}.full_pca1024", rep,
                   f"DIM CONTROL: {name} reduced to 1024 dims by PCA fit on the train split", reduce_to=1024)
            del rep

    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved:", args.output_json, flush=True)


if __name__ == "__main__":
    main()
