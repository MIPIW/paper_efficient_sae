"""Consolidate the checkpoint and trajectory gradient-norm measurements into one JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# §12 synthetic reference (runs/synth_fixA_recon3way/synthetic_diagnostic_results.json,
# lambda_sup=1.0, lambda_sup_warmup_frac=0.0, doc_batch_size=48, CPU).
SYNTHETIC = {
    "source": "runs/synth_fixA_recon3way/synthetic_diagnostic_results.json",
    "lambda_sup": 1.0,
    "q_recon_grad_norm": 18.776460821151733,
    "q_collect_grad_norm": 0.3545453258752823,
    "q_contrast_grad_norm": 0.4130852895617485,
}


def summarize(values) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
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
    ckpt = json.loads((ROOT / "runs" / "real_data_grad_norms_ckpt420k.json").read_text())
    traj_rows = [
        json.loads(line)
        for line in (ROOT / "runs" / "grad_norm_trajectory_3k" / "debug_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]

    lambda_warmup_end = int(0.15 * 3000)
    keys = [
        "q_recon_grad_norm",
        "q_collect_grad_norm",
        "q_contrast_grad_norm",
        "p_recon_grad_norm",
        "p_collect_grad_norm",
        "p_contrast_grad_norm",
    ]
    traj_records = [
        {"step": r["step"], **{k: float(r[k]) for k in keys},
         "q_collect_topic_valid_anchor_frac": float(r["q_collect_topic_valid_anchor_frac"]),
         "q_contrast_sentiment_valid_anchor_frac": float(r["q_contrast_sentiment_valid_anchor_frac"]),
         "p_collect_sentiment_valid_anchor_frac": float(r["p_collect_sentiment_valid_anchor_frac"]),
         "p_contrast_topic_valid_anchor_frac": float(r["p_contrast_topic_valid_anchor_frac"]),
         "loss": float(r["loss"])}
        for r in traj_rows
    ]
    post_warmup = [r for r in traj_records if r["step"] >= lambda_warmup_end]

    traj_summary = {
        "lambda_sup_warmup_end_step": lambda_warmup_end,
        "post_warmup_norms": {k: summarize(r[k] for r in post_warmup) for k in keys},
        "post_warmup_ratios": {
            f"{b}_recon_over_{term}": summarize(
                r[f"{b}_recon_grad_norm"] / max(r[f"{b}_{term}_grad_norm"], 1e-12) for r in post_warmup
            )
            for b in ("q", "p")
            for term in ("collect", "contrast")
        },
    }

    # lambda-normalised ratios: divide out lambda_sup so the real (0.5) and synthetic (1.0)
    # settings are compared on the same supervision scale.
    lam_real = 0.5
    ckpt_lambda_norm = {}
    for b in ("q", "p"):
        for term in ("collect", "contrast"):
            per_batch = [
                r[f"{b}_recon_grad_norm"] / max(r[f"{b}_{term}_grad_norm"] / lam_real, 1e-12)
                for r in ckpt["per_batch"]
            ]
            ckpt_lambda_norm[f"{b}_recon_over_{term}"] = summarize(per_batch)

    synth_ratios = {
        "q_recon_over_collect": SYNTHETIC["q_recon_grad_norm"] / SYNTHETIC["q_collect_grad_norm"],
        "q_recon_over_contrast": SYNTHETIC["q_recon_grad_norm"] / SYNTHETIC["q_contrast_grad_norm"],
    }

    payload = {
        "description": (
            "Per-loss-term gradient norms on KronSAE P/Q branch parameters, measured on real "
            "Amazon-Reviews activations (pythia-410m layer 12). Measurement-only: computed via "
            "KronTopKTrainer.probe_term_grad_norms(), which never writes param.grad."
        ),
        "synthetic_reference_section12": {**SYNTHETIC, "ratios": synth_ratios},
        "checkpoint_420k": {
            "checkpoint": ckpt["checkpoint"],
            "num_batches": ckpt["num_batches"],
            "lambda_sup": lam_real,
            "summary": ckpt["summary"],
            "ratios_as_trained": ckpt["ratios"],
            "ratios_lambda_normalised": ckpt_lambda_norm,
            "per_batch": ckpt["per_batch"],
        },
        "trajectory_3k": {
            "run_dir": "runs/grad_norm_trajectory_3k",
            **traj_summary,
            "per_probe_step": traj_records,
        },
    }
    out = ROOT / "runs" / "real_data_grad_norms.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(
        {
            "ckpt_ratios_as_trained": {k: round(v["median"], 1) for k, v in ckpt["ratios"].items()},
            "ckpt_ratios_lambda_norm": {k: round(v["median"], 1) for k, v in ckpt_lambda_norm.items()},
            "traj_post_warmup_ratios": {k: round(v["median"], 1) for k, v in traj_summary["post_warmup_ratios"].items()},
            "synthetic": {k: round(v, 1) for k, v in synth_ratios.items()},
        },
        indent=2,
    ))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
