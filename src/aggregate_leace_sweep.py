"""Aggregate the LEACE-inspired cross-covariance suppression sweep into one JSON.

Directly comparable to runs/aggregate_adversarial_sweep.py's output (same synthetic-diagnostic
setting, same kron_joint/h/m/n/dpo_beta/seed, same random-init controls). Written incrementally:
re-run after each config completes; configs that have not finished yet are simply absent.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "runs" / "leace_suppress"
CONTROL_PATH = ROOT / "runs" / "gradnorm_sweep" / "randinit_control_check" / "synthetic_diagnostic_results.json"
ADV_RESULTS_PATH = ROOT / "runs" / "adversarial_suppress_results.json"
OUT_PATH = ROOT / "runs" / "leace_suppress_results.json"

LAMBDAS = [
    ("0.01", "0p01"), ("0.03", "0p03"), ("0.1", "0p1"), ("0.3", "0p3"), ("1.0", "1p0"), ("3.0", "3p0"),
]
KEYS = [
    "p_sentiment_acc",
    "q_sentiment_acc",
    "p_topic_acc",
    "q_topic_acc",
    "recon_l2_loss",
    "recon_fvu",
    "leace_lambda",
    "leace_ref_lr",
    "leace_ref_hidden",
    "leace_cross_cov_final_loss",
    "leace_ref_disc_final_loss",
    "leace_ref_disc_final_acc",
    "leace_history",
]


def load(path: Path) -> dict:
    with path.open() as f:
        payload = json.load(f)
    res = payload["results"]["kron_joint"]
    out = {k: res[k] for k in KEYS if k in res}
    out["total_steps"] = payload["args"]["total_steps"]
    out["skip_training"] = payload["args"].get("synthetic_skip_training", False)
    out["leace_suppress"] = payload["args"].get("leace_suppress", False)
    return out


def main() -> None:
    trained = {}
    for label, tag in LAMBDAS:
        p = SWEEP_DIR / f"lam_{tag}" / "synthetic_diagnostic_results.json"
        if p.exists():
            trained[label] = load(p)

    control = load(CONTROL_PATH) if CONTROL_PATH.exists() else None

    adversarial_for_comparison = None
    if ADV_RESULTS_PATH.exists():
        with ADV_RESULTS_PATH.open() as f:
            adv_payload = json.load(f)
        adversarial_for_comparison = {
            lam: {
                k: v[k]
                for k in (
                    "p_sentiment_acc", "q_sentiment_acc", "p_topic_acc", "q_topic_acc",
                    "recon_fvu", "adv_ref_disc_final_acc",
                )
                if k in v
            }
            for lam, v in adv_payload.get("adversarial", {}).items()
        }

    payload = {
        "description": (
            "LEACE-inspired (Belrose et al. 2023) cross-covariance suppression on the synthetic "
            "diagnostic (joint_dpo_cross, kron_joint, h=128/m=8/n=16, dpo_beta=2.0, 5000 steps, "
            "seed=42), same setting as runs/adversarial_suppress_results.json for direct "
            "comparison. A differentiable penalty (leace_lambda * ||Cov(Q, onehot(sentiment))||^2, "
            "batch-level) is added directly to the total loss -- no adversary, no gradient-reversal "
            "layer. This REPLACES Q's DPO contrast-sentiment term, exactly as --adv_suppress does. "
            "leace_ref_disc_* is a measurement-only discriminator trained on the DETACHED "
            "representation (never influences the model), mirroring adv_ref_disc_*."
        ),
        "chance": {"sentiment": 0.5, "topic": 1.0 / 6.0},
        "random_init_control_same_device_reused": control,
        "no_intervention_baseline_report_9_2": {
            "p_sentiment_acc": 0.941,
            "q_sentiment_acc": 0.972,
            "p_topic_acc": 0.786,
            "q_topic_acc": 0.838,
            "recon_fvu": 0.953,
        },
        "adversarial_grl_for_comparison": adversarial_for_comparison,
        "leace": trained,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(trained)} leace configs)")


if __name__ == "__main__":
    main()
