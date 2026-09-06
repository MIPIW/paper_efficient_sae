"""Aggregate the lambda_recon sweep + random-init control into one JSON."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "runs" / "lrsweep"
OUT_PATH = ROOT / "runs" / "lambda_recon_sweep_results.json"

LAMBDAS = [("1.0", "1p0"), ("0.3", "0p3"), ("0.1", "0p1"), ("0.03", "0p03"), ("0.01", "0p01"), ("0.0", "0p0")]
KEYS = [
    "p_sentiment_acc",
    "q_sentiment_acc",
    "p_topic_acc",
    "q_topic_acc",
    "recon_l2_loss",
    "recon_fvu",
]


def load(run_dir: Path) -> dict:
    with (run_dir / "synthetic_diagnostic_results.json").open() as f:
        payload = json.load(f)
    res = payload["results"]["kron_joint"]
    out = {k: res[k] for k in KEYS if k in res}
    out["lambda_recon_arg"] = payload["args"]["lambda_recon"]
    out["total_steps"] = payload["args"]["total_steps"]
    out["skip_training"] = payload["args"].get("synthetic_skip_training", False)
    return out


def main() -> None:
    control = load(SWEEP_DIR / "randinit_control")
    control_extra_seeds = {
        d.name.split("seed")[-1]: load(d)
        for d in sorted(SWEEP_DIR.glob("randinit_control_seed*"))
        if (d / "synthetic_diagnostic_results.json").exists()
    }
    trained = {}
    for label, tag in LAMBDAS:
        d = SWEEP_DIR / f"trained_lrecon_{tag}"
        if (d / "synthetic_diagnostic_results.json").exists():
            trained[label] = load(d)
    payload = {
        "description": (
            "lambda_recon sweep on the synthetic diagnostic (joint_dpo_cross, kron_joint, "
            "h=128/m=8/n=16, dpo_beta=2.0, lambda_sup=1.0, 5000 steps, seed=42, CPU). "
            "Random-init control = identical construction, zero optimizer steps, identical "
            "eval activation distribution (seed+1001) and probe seed (seed+2001)."
        ),
        "chance": {"sentiment": 0.5, "topic": 1.0 / 6.0},
        "random_init_control": control,
        "random_init_control_other_seeds": control_extra_seeds,
        "trained": trained,
        "reference_9_2_baseline": {
            "source": "runs/synth_fix_baseline/synthetic_diagnostic_results.json",
            "p_sentiment_acc": 0.9409722089767456,
            "q_sentiment_acc": 0.9722222089767456,
            "p_topic_acc": 0.7864583134651184,
            "q_topic_acc": 0.8376736044883728,
        },
    }
    with OUT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUT_PATH}")
    hdr = f"{'lambda_recon':>13} {'P-sent':>7} {'Q-sent':>7} {'P-top':>7} {'Q-top':>7} {'FVU':>8} {'l2':>10}"
    print(hdr)
    c = control
    print(f"{'rand-init':>13} {c['p_sentiment_acc']:7.3f} {c['q_sentiment_acc']:7.3f} "
          f"{c['p_topic_acc']:7.3f} {c['q_topic_acc']:7.3f} {c['recon_fvu']:8.4f} {c['recon_l2_loss']:10.2f}")
    for label, _ in LAMBDAS:
        if label not in trained:
            print(f"{label:>13} (missing)")
            continue
        r = trained[label]
        print(f"{label:>13} {r['p_sentiment_acc']:7.3f} {r['q_sentiment_acc']:7.3f} "
              f"{r['p_topic_acc']:7.3f} {r['q_topic_acc']:7.3f} {r['recon_fvu']:8.4f} {r['recon_l2_loss']:10.2f}")


if __name__ == "__main__":
    main()
