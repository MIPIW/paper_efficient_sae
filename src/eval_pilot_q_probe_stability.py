"""Seed-stability of the §6/§10-style raw-feature linear probe on the pilot branches.

Uses the cached token activations produced by eval_pilot_q_liveness.py (no LM pass),
and re-runs the exact §6 probe (Adam, lr=1e-2, 30 epochs, bs=256, raw unstandardized
features) across many probe seeds, alongside the standardized logistic regression,
to show how much of the reported accuracy is probe-optimization noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc
from eval_joint_probe import stratified_split_indices, train_linear_probe
from eval_representation_geometry import (
    balanced_subset_indices,
    build_kron_from_config,
    load_checkpoint_bundle,
    set_seed,
)


def branch_reps(bundle, batches, device, max_docs):
    ae = build_kron_from_config(bundle.trainer_config, bundle.ckpt_path, device=device)
    p_all, q_all, topic, sent = [], [], [], []
    with t.no_grad():
        for b in batches:
            x = b["x"].to(device=device, dtype=t.float32)
            _, _, p_pos, q_pos = ae._prelatents(x)
            ids = b["token_doc_ids"].to(device)
            p_all.append(_mean_pool_by_doc(p_pos.reshape(p_pos.shape[0], -1), ids, b["n_docs"]).cpu())
            q_all.append(_mean_pool_by_doc(q_pos.reshape(q_pos.shape[0], -1), ids, b["n_docs"]).cpu())
            topic.append(b["topic"])
            sent.append(b["sentiment"])
    del ae
    t.cuda.empty_cache()
    return (
        {"p": t.cat(p_all)[:max_docs], "q": t.cat(q_all)[:max_docs]},
        t.cat(topic)[:max_docs],
        t.cat(sent)[:max_docs],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation_cache", type=str, required=True)
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            str(ROOT / "runs" / "pilot_balanced_v2_420k" / "checkpoints" / "kron_pilot_sentiment" / "ae_step_420000.pt"),
            str(ROOT / "runs" / "pilot_balanced_v2_420k" / "checkpoints" / "kron_pilot_topic" / "ae_step_420000.pt"),
        ],
    )
    ap.add_argument("--names", nargs="*", default=["kron_pilot_sentiment", "kron_pilot_topic"])
    ap.add_argument("--max_probe_docs", type=int, default=12000)
    # §6 used 4000 docs; §10 used 12000. Sweep both to expose sample-size sensitivity.
    ap.add_argument("--doc_subsets", nargs="+", type=int, default=[4000, 12000])
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--test_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--output_json", type=str, default=str(ROOT / "runs" / "pilot_q_probe_stability.json"))
    args = ap.parse_args()

    device = args.device or ("cuda:0" if t.cuda.is_available() else "cpu")
    set_seed(args.seed)
    batches = t.load(args.activation_cache, map_location="cpu")

    from sklearn.linear_model import LogisticRegression

    out = {
        "settings": {
            "device": device,
            "activation_cache": args.activation_cache,
            "n_seeds": args.n_seeds,
            "doc_subsets": args.doc_subsets,
            "raw_probe": "Adam lr=1e-2, 30 epochs, bs=256, unstandardized features (identical to §6/§10)",
            "std_probe": "sklearn LogisticRegression(max_iter=5000, C=1.0) on train-z-scored features",
            "chance": {"sentiment": 0.5, "topic": 1 / 6},
        },
        "results": [],
    }

    for i, ck in enumerate(args.checkpoints):
        bundle = load_checkpoint_bundle(ck, args.names[i] if i < len(args.names) else None)
        reps, topic_labels, sent_labels = branch_reps(bundle, batches, device, args.max_probe_docs)

        for n_docs in args.doc_subsets:
            for branch in ("p", "q"):
                for label_name, labels in (("sentiment", sent_labels), ("topic", topic_labels)):
                    xb = reps[branch][:n_docs]
                    yb_full = labels[:n_docs]
                    idx = balanced_subset_indices(yb_full, label_name, seed=args.seed)
                    x_bal, y_bal = xb[idx].float(), yb_full[idx]
                    tr, te = stratified_split_indices(y_bal, test_fraction=args.test_fraction, seed=args.seed + 11)
                    x_tr, x_te, y_tr, y_te = x_bal[tr], x_bal[te], y_bal[tr], y_bal[te]
                    nc = int(y_bal.max().item()) + 1

                    raw_accs = [
                        train_linear_probe(
                            X_train=x_tr, y_train=y_tr, X_test=x_te, y_test=y_te,
                            num_classes=nc, epochs=30, lr=1e-2, batch_size=256,
                            seed=args.seed + 100 * s, device=device,
                        )
                        for s in range(args.n_seeds)
                    ]
                    mu = x_tr.mean(0, keepdim=True)
                    sd = x_tr.std(0, keepdim=True).clamp_min(1e-6)
                    clf = LogisticRegression(max_iter=5000, C=1.0, n_jobs=-1)
                    clf.fit(((x_tr - mu) / sd).numpy(), y_tr.numpy())
                    std_acc = float((clf.predict(((x_te - mu) / sd).numpy()) == y_te.numpy()).mean())

                    row = {
                        "checkpoint": bundle.name,
                        "branch": branch,
                        "label": label_name,
                        "n_docs": n_docs,
                        "n_balanced": int(y_bal.shape[0]),
                        "n_test": int(y_te.shape[0]),
                        "raw_linear_accs": [float(a) for a in raw_accs],
                        "raw_linear_mean": float(np.mean(raw_accs)),
                        "raw_linear_std": float(np.std(raw_accs)),
                        "raw_linear_min": float(np.min(raw_accs)),
                        "raw_linear_max": float(np.max(raw_accs)),
                        "std_logreg_acc": std_acc,
                    }
                    out["results"].append(row)
                    print(
                        f"{bundle.name:22s} {branch} {label_name:9s} n={n_docs:5d} | "
                        f"raw {row['raw_linear_mean']:.4f} +-{row['raw_linear_std']:.4f} "
                        f"[{row['raw_linear_min']:.4f},{row['raw_linear_max']:.4f}] | "
                        f"std_logreg {std_acc:.4f}",
                        flush=True,
                    )

    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved:", args.output_json)


if __name__ == "__main__":
    main()
