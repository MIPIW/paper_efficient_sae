"""Measure per-loss-term gradient norms on the P/Q branches at a frozen checkpoint.

Measurement-only: builds the exact §8 joint_dpo_cross trainer, loads a checkpoint,
runs real Amazon-Reviews batches through `loss()`, and decomposes the gradient on
`p_proj/p_bias` and `q_proj/q_bias` into reconstruction / collect / contrast parts
via `KronTopKTrainer.probe_term_grad_norms()`. No optimizer step is ever taken and
`param.grad` is never written, so training dynamics are untouched.
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

from data.amazon_reviews import (  # noqa: E402
    infinite_dataloader_batches,
    load_prepared_datasets,
)
from dictionary_learning.labeled_buffer import LabeledActivationBuffer  # noqa: E402
from train import (  # noqa: E402
    DistEnv,
    build_joint_balanced_dataloader,
    build_model,
    create_joint_trainers,
    set_seed,
)

PROBE_KEYS = [
    "q_recon_grad_norm",
    "q_collect_grad_norm",
    "q_contrast_grad_norm",
    "p_recon_grad_norm",
    "p_collect_grad_norm",
    "p_contrast_grad_norm",
]

ANCHOR_KEYS = [
    "p_collect_sentiment_valid_anchor_frac",
    "p_contrast_topic_valid_anchor_frac",
    "q_collect_topic_valid_anchor_frac",
    "q_contrast_sentiment_valid_anchor_frac",
]


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--measure_step", type=int, default=420000)
    parser.add_argument("--output", type=str, required=True)
    # Config knobs mirroring the §8 run.
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--io", type=str, default="out")
    parser.add_argument("--activation_dim", type=int, default=1024)
    parser.add_argument("--joint_h", type=int, default=128)
    parser.add_argument("--joint_m", type=int, default=8)
    parser.add_argument("--joint_n", type=int, default=16)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--total_steps", type=int, default=420000)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--decay_start_frac", type=float, default=0.8)
    parser.add_argument("--threshold_start_frac", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--decay_start", type=int, default=None)
    parser.add_argument("--threshold_start_step", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--auxk_alpha", type=float, default=1.0 / 32.0)
    parser.add_argument("--lambda_sup", type=float, default=0.5)
    parser.add_argument("--lambda_sup_warmup_frac", type=float, default=0.15)
    parser.add_argument("--supcon_temperature", type=float, default=0.1)
    parser.add_argument("--dpo_beta", type=float, default=2.0)
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--doc_batch_size", type=int, default=16)
    parser.add_argument("--dataset_cache_dir", type=str, required=True)
    parser.add_argument("--total_cpu_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flat_dict_size", type=int, default=16384)
    parser.add_argument("--flat_sup_grad_scale", type=float, default=0.3)
    parser.add_argument("--remove_bos", action="store_true")
    parser.add_argument("--max_activation_norm_multiple", type=int, default=None)
    args = parser.parse_args()

    # Defaults required by create_joint_trainers / build_joint_balanced_dataloader.
    args.joint_mode = "dpo_cross"
    for name in (
        "dpo_beta_p_collect_sentiment",
        "dpo_beta_p_contrast_topic",
        "dpo_beta_q_collect_topic",
        "dpo_beta_q_contrast_sentiment",
    ):
        setattr(args, name, None)
    for name in (
        "dpo_w_p_collect_sentiment",
        "dpo_w_p_contrast_topic",
        "dpo_w_q_collect_topic",
        "dpo_w_q_contrast_sentiment",
    ):
        setattr(args, name, 1.0)
    args.contrast_start_frac = 0.0
    args.q_gradient_surgery = False
    args.q_gradient_surgery_include_recon = False
    args.q_contrast_grad_norm_match = False
    args.q_contrast_grad_norm_match_cap = 0.0
    args.q_separate_grad_clip = False
    args.grad_norm_probe_every = 0
    args.lambda_orth = 0.0

    set_seed(args.seed)
    dist_env = DistEnv(enabled=False, rank=0, local_rank=0, world_size=1)
    device = "cuda:0" if t.cuda.is_available() else "cpu"

    train_ds, _, _, _ = load_prepared_datasets(args.dataset_cache_dir)
    model, submodule = build_model(args, dist_env, device)
    loader, sampler, _ = build_joint_balanced_dataloader(train_ds, args, dist_env)
    buffer = LabeledActivationBuffer(
        data=infinite_dataloader_batches(loader),
        model=model,
        submodule=submodule,
        d_submodule=args.activation_dim,
        io=args.io,
        ctx_len=args.ctx_len,
        device=device,
        remove_bos=args.remove_bos,
        add_special_tokens=True,
        max_activation_norm_multiple=args.max_activation_norm_multiple,
    )

    trainers = dict(create_joint_trainers(args, device=device))
    trainer = trainers["kron_joint"]
    state = t.load(args.checkpoint, map_location=device)
    trainer.ae.load_state_dict(state)
    trainer.ae.to(device)
    print(f"Loaded {args.checkpoint}", flush=True)

    records: list[dict[str, float]] = []
    for i in range(args.num_batches):
        batch = next(buffer)
        batch.activations = batch.activations.to(dtype=t.float32)
        loss = trainer.loss(batch, step=args.measure_step)
        norms = trainer.probe_term_grad_norms()
        rec = dict(norms)
        rec["loss"] = float(loss.detach().cpu())
        rec["num_tokens"] = int(batch.activations.shape[0])
        rec["effective_lambda_sup"] = float(trainer.effective_lambda_sup)
        rec["contrast_scale"] = float(trainer.contrast_scale)
        for key in ANCHOR_KEYS:
            rec[key] = float(getattr(trainer, key))
        records.append(rec)
        # Drop the graph before the next iteration.
        del loss
        trainer.ae.zero_grad(set_to_none=True)
        if (i + 1) % 10 == 0:
            print(f"batch {i + 1}/{args.num_batches}", flush=True)

    summary = {
        key: summarize([r[key] for r in records])
        for key in PROBE_KEYS + ANCHOR_KEYS + ["loss", "num_tokens"]
    }
    ratios = {}
    for branch in ("q", "p"):
        for term in ("collect", "contrast"):
            per_batch = [
                r[f"{branch}_recon_grad_norm"] / max(r[f"{branch}_{term}_grad_norm"], 1e-12)
                for r in records
            ]
            ratios[f"{branch}_recon_over_{term}"] = summarize(per_batch)

    payload = {
        "checkpoint": args.checkpoint,
        "measure_step": args.measure_step,
        "num_batches": args.num_batches,
        "config": trainer.config,
        "args": vars(args),
        "summary": summary,
        "ratios": ratios,
        "per_batch": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"summary": summary, "ratios": ratios}, indent=2))
    print(f"Wrote {out_path}", flush=True)

    del sampler
    if hasattr(loader, "_iterator") and loader._iterator is not None:
        loader._iterator._shutdown_workers()


if __name__ == "__main__":
    main()
