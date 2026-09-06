"""Q-branch liveness check for the pilot Kron checkpoints (no retraining).

Decides between two explanations of `kron_pilot_sentiment`'s Q-sentiment == 0.500:
  (a) Q is a live representation that specifically lacks sentiment.
  (b) Q is a collapsed/dead branch that lacks essentially everything.

Measures, on the same frozen checkpoints and with the same data/pooling/probe
conventions as `eval_representation_geometry.py` (which produced REPORT.md §10):
  1. Linear / kNN / MLP probes for BOTH labels on BOTH branches (§10 restricted
     each pilot checkpoint to the label it was trained on -- that restriction is
     exactly why Q-topic in kron_pilot_sentiment was never measured).
  2. Reconstruction ablation of each branch (zero-ablation and mean-ablation).
  3. Dead/inactive unit statistics at token level for each branch.
  4. Participation ratio + top-K normalized covariance eigenvalue spectrum.

The LM forward pass is run ONCE and the token activations are cached, then
re-used for every checkpoint / branch / ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch as t
from nnsight import LanguageModel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from data.amazon_reviews import load_prepared_datasets, collate_document_batch
from dictionary_learning.labeled_buffer import LabeledActivationBuffer
from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc
from eval_joint_probe import label_count_dict
from eval_representation_geometry import (  # noqa: E402
    MLPProbe,
    balanced_subset_indices,
    build_kron_from_config,
    evaluate_label_metrics,
    load_checkpoint_bundle,
    mlp_probe_accuracy,
    participation_ratio_from_cov,
    set_seed,
)
from eval_joint_probe import stratified_split_indices, train_linear_probe  # noqa: E402


# --------------------------------------------------------------------------
# well-conditioned ("standardized") probes
# --------------------------------------------------------------------------
def robust_label_metrics(
    x_branch: t.Tensor,
    labels: t.Tensor,
    label_name: str,
    seed: int,
    test_fraction: float,
    mlp_hidden: int,
    mlp_dropout: float,
    mlp_epochs: int,
    mlp_lr: float,
    probe_batch_size: int,
    device: str,
) -> dict:
    """Same balanced subset / stratified split as evaluate_label_metrics (so the
    numbers are directly comparable), but the features are z-scored using TRAIN
    statistics before probing. Needed because the raw Q representations have
    mean-L2 ~1e3 with per-dim std ~1e1, which makes the fixed-LR Adam probes in
    the §10 pipeline fail to leave the constant-predictor solution."""
    subset_idx = balanced_subset_indices(labels, label_name, seed=seed)
    x_bal = x_branch[subset_idx].float()
    y_bal = labels[subset_idx]
    train_idx, test_idx = stratified_split_indices(y_bal, test_fraction=test_fraction, seed=seed + 11)

    x_train_raw, x_test_raw = x_bal[train_idx], x_bal[test_idx]
    y_train, y_test = y_bal[train_idx], y_bal[test_idx]
    num_classes = int(y_bal.max().item()) + 1

    mu = x_train_raw.mean(dim=0, keepdim=True)
    sd = x_train_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train_raw - mu) / sd
    x_test = (x_test_raw - mu) / sd

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=5000, C=1.0, n_jobs=-1)
    clf.fit(x_train.numpy(), y_train.numpy())
    logreg_acc = float((clf.predict(x_test.numpy()) == y_test.numpy()).mean())

    torch_linear_std = train_linear_probe(
        X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test,
        num_classes=num_classes, epochs=30, lr=1e-2, batch_size=256,
        seed=seed + 1, device=device,
    )
    mlp_std = mlp_probe_accuracy(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        num_classes=num_classes, hidden_dim=mlp_hidden, dropout=mlp_dropout,
        epochs=mlp_epochs, lr=mlp_lr, batch_size=probe_batch_size,
        seed=seed + 2, device=device,
    )
    # majority-class (constant predictor) accuracy on the test split
    maj = float(t.bincount(y_test, minlength=num_classes).max().item()) / float(y_test.shape[0])

    return {
        "num_classes": num_classes,
        "train_n": int(y_train.shape[0]),
        "test_n": int(y_test.shape[0]),
        "majority_class_acc_test": maj,
        "sklearn_logreg_std_acc": logreg_acc,
        "torch_linear_std_acc": float(torch_linear_std),
        "mlp_std_acc": float(mlp_std),
        "logreg_n_iter": int(np.max(clf.n_iter_)),
    }


# --------------------------------------------------------------------------
# activation caching (single LM pass, shared across checkpoints)
# --------------------------------------------------------------------------
def cache_token_activations(
    model: LanguageModel,
    submodule,
    eval_ds,
    activation_dim: int,
    ctx_len: int,
    doc_batch_size: int,
    num_workers: int,
    max_docs: int,
    device: str,
) -> List[dict]:
    """Replicates eval_joint_probe.collect_doc_representations' data path exactly,
    but caches raw token activations instead of AE outputs."""
    loader = DataLoader(
        eval_ds,
        batch_size=doc_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=t.cuda.is_available(),
        collate_fn=collate_document_batch,
        drop_last=False,
    )
    buffer = LabeledActivationBuffer(
        data=iter(loader),
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        io="out",
        ctx_len=ctx_len,
        device=device,
        remove_bos=False,
        add_special_tokens=True,
        max_activation_norm_multiple=None,
    )

    batches: List[dict] = []
    collected_docs = 0
    with t.no_grad():
        while collected_docs < max_docs:
            try:
                batch = next(buffer)
            except StopIteration:
                break
            n_docs = int(batch.topic_labels.shape[0])
            batches.append(
                {
                    "x": batch.activations.to(dtype=t.float16).cpu(),
                    "token_doc_ids": batch.token_doc_ids.long().cpu(),
                    "topic": batch.topic_labels.long().cpu(),
                    "sentiment": batch.sentiment_labels.long().cpu(),
                    "n_docs": n_docs,
                }
            )
            collected_docs += n_docs
    if not batches:
        raise RuntimeError("No activations collected")
    return batches


# --------------------------------------------------------------------------
# ablated encode
# --------------------------------------------------------------------------
def encode_with_override(ae, x, p_override=None, q_override=None):
    """Reproduces KronAutoEncoderTopK.encode (use_threshold=False) but lets us
    replace the P and/or Q post-ReLU branch activations before the mAND combine.

    p_override / q_override are either None (no change), the string 'zero'
    (branch activations set to 0), or a tensor of shape (h,m)/(h,n) that is
    broadcast over the batch (mean-ablation).
    """
    u_pre, v_pre, p_pos, q_pos = ae._prelatents(x)
    if p_override is not None:
        p_pos = t.zeros_like(p_pos) if isinstance(p_override, str) else p_override.unsqueeze(0).expand_as(p_pos)
    if q_override is not None:
        q_pos = t.zeros_like(q_pos) if isinstance(q_override, str) else q_override.unsqueeze(0).expand_as(q_pos)
    dense_BF = ae._combine(u_pre, v_pre, p_pos, q_pos)
    post_topk = dense_BF.topk(int(ae.k.item()), sorted=False, dim=-1)
    encoded = t.zeros_like(dense_BF).scatter_(dim=-1, index=post_topk.indices, src=post_topk.values)
    return encoded


# --------------------------------------------------------------------------
# per-checkpoint analysis
# --------------------------------------------------------------------------
def analyze_checkpoint(bundle, batches, device, args) -> dict:
    ae = build_kron_from_config(bundle.trainer_config, bundle.ckpt_path, device=device)
    h, m, n = int(bundle.trainer_config["h"]), int(bundle.trainer_config["m"]), int(bundle.trainer_config["n"])

    # ---- pass 1: branch activations, doc pooling, token stats, x moments ----
    all_p_docs, all_q_docs, all_topic, all_sent = [], [], [], []
    p_active = t.zeros(h * m, dtype=t.float64, device=device)
    q_active = t.zeros(h * n, dtype=t.float64, device=device)
    p_sum = t.zeros(h * m, dtype=t.float64, device=device)
    q_sum = t.zeros(h * n, dtype=t.float64, device=device)
    p_sum_active = t.zeros(h * m, dtype=t.float64, device=device)
    q_sum_active = t.zeros(h * n, dtype=t.float64, device=device)
    n_tokens = 0
    x_sum = t.zeros(ae.activation_dim, dtype=t.float64, device=device)
    x_sq_sum = t.zeros((), dtype=t.float64, device=device)

    with t.no_grad():
        for b in batches:
            x = b["x"].to(device=device, dtype=t.float32)
            _, _, p_pos, q_pos = ae._prelatents(x)
            p_tok = p_pos.reshape(p_pos.shape[0], -1)
            q_tok = q_pos.reshape(q_pos.shape[0], -1)
            ids = b["token_doc_ids"].to(device)
            all_p_docs.append(_mean_pool_by_doc(p_tok, ids, b["n_docs"]).cpu())
            all_q_docs.append(_mean_pool_by_doc(q_tok, ids, b["n_docs"]).cpu())
            all_topic.append(b["topic"])
            all_sent.append(b["sentiment"])

            p_active += (p_tok > 0).sum(dim=0).double()
            q_active += (q_tok > 0).sum(dim=0).double()
            p_sum += p_tok.sum(dim=0).double()
            q_sum += q_tok.sum(dim=0).double()
            p_sum_active += t.where(p_tok > 0, p_tok, t.zeros_like(p_tok)).sum(dim=0).double()
            q_sum_active += t.where(q_tok > 0, q_tok, t.zeros_like(q_tok)).sum(dim=0).double()
            n_tokens += x.shape[0]
            x_sum += x.sum(dim=0).double()
            x_sq_sum += (x.double() ** 2).sum()

    max_docs = args.max_probe_docs
    p_docs = t.cat(all_p_docs, dim=0)[:max_docs]
    q_docs = t.cat(all_q_docs, dim=0)[:max_docs]
    topic_labels = t.cat(all_topic, dim=0)[:max_docs]
    sentiment_labels = t.cat(all_sent, dim=0)[:max_docs]
    reps = {"p": p_docs, "q": q_docs}

    p_mean_unit = (p_sum / n_tokens).float().reshape(h, m)
    q_mean_unit = (q_sum / n_tokens).float().reshape(h, n)

    # ---- pass 2: reconstruction ablations (AE-only, cached activations) ----
    x_mean = (x_sum / n_tokens).float()
    conditions = {
        "baseline": (None, None),
        "q_zero": (None, "zero"),
        "p_zero": ("zero", None),
        "q_mean": (None, q_mean_unit.to(device)),
        "p_mean": (p_mean_unit.to(device), None),
    }
    sse = {k: t.zeros((), dtype=t.float64, device=device) for k in conditions}
    total_var_num = t.zeros((), dtype=t.float64, device=device)
    with t.no_grad():
        for b in batches:
            x = b["x"].to(device=device, dtype=t.float32)
            total_var_num += ((x - x_mean).double() ** 2).sum()
            for name, (po, qo) in conditions.items():
                enc = encode_with_override(ae, x, p_override=po, q_override=qo)
                x_hat = ae.decode(enc)
                sse[name] += ((x - x_hat).double() ** 2).sum()

    total_var = float(total_var_num.item())
    recon = {
        name: {
            "mse_per_token": float(v.item()) / n_tokens,
            "fvu": float(v.item()) / total_var,
            "frac_variance_explained": 1.0 - float(v.item()) / total_var,
        }
        for name, v in sse.items()
    }

    # ---- geometry + dead-unit stats + probes ----
    out = {
        "name": bundle.name,
        "trainer_name": bundle.trainer_name,
        "checkpoint_path": str(bundle.ckpt_path),
        "h": h,
        "m": m,
        "n": n,
        "k": int(ae.k.item()),
        "combine_rule": bundle.trainer_config["combine_rule"],
        "num_docs": int(topic_labels.shape[0]),
        "num_tokens": int(n_tokens),
        "reconstruction": recon,
        "branch_results": {},
    }

    unit_stats = {
        "p": (p_active, p_sum, p_sum_active),
        "q": (q_active, q_sum, q_sum_active),
    }

    for branch in ("p", "q"):
        xb = reps[branch]
        active, ssum, ssum_active = unit_stats[branch]
        act_rate = (active / n_tokens).cpu().numpy()
        mean_act = (ssum / n_tokens).cpu().numpy()
        mean_act_when_active = (ssum_active / active.clamp_min(1.0)).cpu().numpy()

        # covariance eigen spectrum on doc-pooled reps (same object §10's PR used)
        xf = xb.float()
        xc = xf - xf.mean(dim=0, keepdim=True)
        cov = (xc.T @ xc) / max(1, xf.shape[0] - 1)
        eig = t.linalg.eigvalsh(cov).clamp_min(0.0)
        eig_sorted = t.sort(eig, descending=True).values
        eig_total = float(eig_sorted.sum().item())
        top_eig = (eig_sorted[: args.top_eigs] / max(eig_total, 1e-30)).tolist()

        branch_out = {
            "feature_dim": int(xb.shape[1]),
            "overall_participation_ratio": participation_ratio_from_cov(xb),
            "cov_eigenvalue_total": eig_total,
            "cov_eigenvalue_normalized_top": top_eig,
            "cov_eigenvalue_top1_frac": top_eig[0] if top_eig else None,
            "cov_eigenvalue_top5_frac": float(sum(top_eig[:5])) if top_eig else None,
            "doc_rep_mean_l2": float(xf.norm(dim=1).mean().item()),
            "doc_rep_std_mean": float(xf.std(dim=0).mean().item()),
            "unit_stats": {
                "num_units": int(act_rate.shape[0]),
                "frac_units_never_active": float((act_rate == 0.0).mean()),
                "frac_units_active_lt_1e-4": float((act_rate < 1e-4).mean()),
                "frac_units_active_lt_1e-3": float((act_rate < 1e-3).mean()),
                "frac_units_active_lt_1e-2": float((act_rate < 1e-2).mean()),
                "mean_activation_rate": float(act_rate.mean()),
                "median_activation_rate": float(np.median(act_rate)),
                "activation_rate_percentiles": {
                    str(p): float(np.percentile(act_rate, p)) for p in (1, 5, 25, 50, 75, 95, 99)
                },
                "mean_activation_magnitude": float(mean_act.mean()),
                "mean_activation_magnitude_percentiles": {
                    str(p): float(np.percentile(mean_act, p)) for p in (1, 5, 25, 50, 75, 95, 99)
                },
                "mean_magnitude_when_active_percentiles": {
                    str(p): float(np.percentile(mean_act_when_active, p)) for p in (1, 5, 25, 50, 75, 95, 99)
                },
            },
            "labels": {},
        }

        for li, (label_name, labels) in enumerate((("sentiment", sentiment_labels), ("topic", topic_labels))):
            branch_out["labels"][label_name] = evaluate_label_metrics(
                x_branch=xb,
                labels=labels,
                label_name=label_name,
                seed=args.seed + 10 * (li + 1) + (0 if branch == "p" else 100),
                test_fraction=args.test_fraction,
                knn_k=args.knn_k,
                mlp_hidden=args.mlp_hidden,
                mlp_dropout=args.mlp_dropout,
                mlp_epochs=args.mlp_epochs,
                mlp_lr=args.mlp_lr,
                probe_batch_size=args.probe_batch_size,
                device=device,
                hsic_max_samples=args.hsic_max_samples,
                linear_override=None,
                compute_linear_probe=True,
            )
            branch_out["labels"][label_name]["standardized"] = robust_label_metrics(
                x_branch=xb,
                labels=labels,
                label_name=label_name,
                seed=args.seed + 10 * (li + 1) + (0 if branch == "p" else 100),
                test_fraction=args.test_fraction,
                mlp_hidden=args.mlp_hidden,
                mlp_dropout=args.mlp_dropout,
                mlp_epochs=args.mlp_epochs,
                mlp_lr=args.mlp_lr,
                probe_batch_size=args.probe_batch_size,
                device=device,
            )

        out["branch_results"][branch] = branch_out

    out["label_counts"] = {
        "topic": label_count_dict(topic_labels),
        "sentiment": label_count_dict(sentiment_labels),
    }
    del ae
    t.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Q-branch liveness check on frozen pilot Kron checkpoints.")
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            str(ROOT / "runs" / "pilot_balanced_v2_420k" / "checkpoints" / "kron_pilot_sentiment" / "ae_step_420000.pt"),
            str(ROOT / "runs" / "pilot_balanced_v2_420k" / "checkpoints" / "kron_pilot_topic" / "ae_step_420000.pt"),
        ],
    )
    ap.add_argument("--checkpoint_names", nargs="*", default=["kron_pilot_sentiment", "kron_pilot_topic"])
    ap.add_argument("--dataset_cache_dir", type=str, default=None)
    ap.add_argument("--max_probe_docs", type=int, default=12000)
    ap.add_argument("--doc_batch_size", type=int, default=24)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--test_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--knn_k", type=int, default=15)
    ap.add_argument("--mlp_hidden", type=int, default=128)
    ap.add_argument("--mlp_dropout", type=float, default=0.2)
    ap.add_argument("--mlp_epochs", type=int, default=30)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--probe_batch_size", type=int, default=256)
    ap.add_argument("--hsic_max_samples", type=int, default=4000)
    ap.add_argument("--top_eigs", type=int, default=20)
    ap.add_argument("--activation_cache", type=str, default=None)
    ap.add_argument("--output_json", type=str, default=str(ROOT / "runs" / "pilot_q_liveness_check.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda:0" if t.cuda.is_available() else "cpu")

    bundles = [
        load_checkpoint_bundle(c, args.checkpoint_names[i] if i < len(args.checkpoint_names) else None)
        for i, c in enumerate(args.checkpoints)
    ]
    ta0 = bundles[0].run_metadata["args"]
    model_name, layer, ctx_len = ta0["model_name"], ta0["layer"], ta0["ctx_len"]
    dataset_cache_dir = args.dataset_cache_dir or ta0["dataset_cache_dir"]
    _, eval_ds, _, _ = load_prepared_datasets(dataset_cache_dir)

    cache_path = Path(args.activation_cache) if args.activation_cache else None
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached token activations from {cache_path} (no LM pass)", flush=True)
        batches = t.load(cache_path, map_location="cpu")
    else:
        model = LanguageModel(model_name, dispatch=True, device_map=device)
        submodule = model.gpt_neox.layers[layer]
        print("Caching token activations (single LM pass)...", flush=True)
        batches = cache_token_activations(
            model=model,
            submodule=submodule,
            eval_ds=eval_ds,
            activation_dim=int(bundles[0].trainer_config["activation_dim"]),
            ctx_len=ctx_len,
            doc_batch_size=args.doc_batch_size,
            num_workers=args.num_workers,
            max_docs=args.max_probe_docs,
            device=device,
        )
        del model
        t.cuda.empty_cache()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            t.save(batches, cache_path)

    total_tokens = sum(b["x"].shape[0] for b in batches)
    total_docs = sum(b["n_docs"] for b in batches)
    print(f"cached {len(batches)} batches | {total_docs} docs | {total_tokens} tokens", flush=True)

    output = {
        "settings": {
            "device": device,
            "model_name": model_name,
            "layer": layer,
            "ctx_len": ctx_len,
            "dataset_cache_dir": str(dataset_cache_dir),
            "max_probe_docs": args.max_probe_docs,
            "doc_batch_size": args.doc_batch_size,
            "num_workers": args.num_workers,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "knn_k": args.knn_k,
            "mlp_hidden": args.mlp_hidden,
            "mlp_dropout": args.mlp_dropout,
            "mlp_epochs": args.mlp_epochs,
            "mlp_lr": args.mlp_lr,
            "probe_batch_size": args.probe_batch_size,
            "hsic_max_samples": args.hsic_max_samples,
            "linear_probe_mode": "compute_now (all labels, both branches)",
            "ablation_definition": (
                "encode() reproduced with use_threshold=False; the named branch's post-ReLU "
                "activations (p_pos / q_pos) are replaced BEFORE the mAND combine "
                "z_ij = sqrt(p_i*q_j), then top-k (k from ckpt) and decode proceed normally. "
                "'zero' sets them to 0; 'mean' sets them to the per-unit mean over all "
                "held-out tokens (broadcast), removing input-dependent information while "
                "preserving scale."
            ),
            "chance": {"sentiment": 0.5, "topic": 1.0 / 6.0},
        },
        "checkpoints": [],
    }

    for bundle in bundles:
        print(f"\n=== {bundle.name} ===", flush=True)
        res = analyze_checkpoint(bundle, batches, device, args)
        output["checkpoints"].append(res)

        r = res["reconstruction"]
        print("  reconstruction FVU: " + ", ".join(f"{k}={v['fvu']:.4f}" for k, v in r.items()), flush=True)
        for br in ("p", "q"):
            b = res["branch_results"][br]
            print(
                f"  branch={br} dim={b['feature_dim']} PR={b['overall_participation_ratio']:.4f} "
                f"top1_eig={b['cov_eigenvalue_top1_frac']:.4f} "
                f"dead(never)={b['unit_stats']['frac_units_never_active']:.4f} "
                f"act_rate_mean={b['unit_stats']['mean_activation_rate']:.4f}",
                flush=True,
            )
            for ln in ("sentiment", "topic"):
                mm = b["labels"][ln]
                s = mm["standardized"]
                print(
                    f"    {ln}: [raw/§10-style] linear={mm['linear_probe_acc']:.4f} mlp={mm['mlp_probe_acc']:.4f} "
                    f"knn={mm['knn_probe_acc']:.4f} hsic_norm={mm['hsic_normalized_cka']:.4f} "
                    f"n_bal={mm['balanced_subset_n']} n_test={mm['test_n']}",
                    flush=True,
                )
                print(
                    f"      [standardized] logreg={s['sklearn_logreg_std_acc']:.4f} "
                    f"linear={s['torch_linear_std_acc']:.4f} mlp={s['mlp_std_acc']:.4f} "
                    f"majority={s['majority_class_acc_test']:.4f}",
                    flush=True,
                )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
