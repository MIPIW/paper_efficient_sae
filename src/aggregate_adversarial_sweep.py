"""Aggregate the adversarial (gradient-reversal) suppression sweep into one JSON.

Written incrementally: re-run after each config completes; configs that have not finished
yet are simply absent from the output.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "runs" / "adversarial_suppress"
CONTROL_PATH = ROOT / "runs" / "gradnorm_sweep" / "randinit_control_check" / "synthetic_diagnostic_results.json"
OUT_PATH = ROOT / "runs" / "adversarial_suppress_results.json"

LAMBDAS = [("0.3", "0p3"), ("1.0", "1p0"), ("3.0", "3p0"), ("10.0", "10p0")]
KEYS = [
    "p_sentiment_acc",
    "q_sentiment_acc",
    "p_topic_acc",
    "q_topic_acc",
    "recon_l2_loss",
    "recon_fvu",
    "adv_grl_lambda",
    "adv_lr",
    "adv_hidden",
    "adv_disc_final_loss",
    "adv_disc_final_acc",
    "adv_ref_disc_final_loss",
    "adv_ref_disc_final_acc",
    "adv_disc_history",
]


def load(path: Path) -> dict:
    with path.open() as f:
        payload = json.load(f)
    res = payload["results"]["kron_joint"]
    out = {k: res[k] for k in KEYS if k in res}
    out["total_steps"] = payload["args"]["total_steps"]
    out["skip_training"] = payload["args"].get("synthetic_skip_training", False)
    out["adv_suppress"] = payload["args"].get("adv_suppress", False)
    return out


def main() -> None:
    trained = {}
    for label, tag in LAMBDAS:
        p = SWEEP_DIR / f"grl_{tag}" / "synthetic_diagnostic_results.json"
        if p.exists():
            trained[label] = load(p)

    control = load(CONTROL_PATH) if CONTROL_PATH.exists() else None
    gpu_control_path = SWEEP_DIR / "randinit_control_gpu" / "synthetic_diagnostic_results.json"
    gpu_control = load(gpu_control_path) if gpu_control_path.exists() else None

    payload = {
        "description": (
            "Adversarial suppression via gradient reversal (Ganin & Lempitsky, 2015) on the "
            "synthetic diagnostic (joint_dpo_cross, kron_joint, h=128/m=8/n=16, dpo_beta=2.0, "
            "5000 steps, seed=42, CPU). A 2-layer MLP discriminator (hidden=128, own Adam at "
            "--adv_lr) predicts sentiment from Q's doc-pooled representation; a gradient-reversal "
            "layer between Q and the discriminator trains Q to fool it with strength "
            "--adv_grl_lambda. This REPLACES Q's DPO contrast-sentiment term (dropped from both "
            "numerator and denominator of the supervision loss); P's terms, Q's collect-topic "
            "term and reconstruction are unchanged. adv_ref_disc_* is a measurement-only "
            "discriminator of identical architecture trained on the DETACHED representation "
            "(never influences the model): it reports how separable Q actually is at that point "
            "in training, distinguishing 'the adversarial discriminator never learned' from "
            "'it learned and then got fooled'."
        ),
        "chance": {"sentiment": 0.5, "topic": 1.0 / 6.0},
        "random_init_control_cpu_reused": control,
        "random_init_control_same_device": gpu_control,
        "no_intervention_baseline_report_9_2": {
            "p_sentiment_acc": 0.941,
            "q_sentiment_acc": 0.972,
            "p_topic_acc": 0.786,
            "q_topic_acc": 0.838,
            "recon_fvu": 0.953,
        },
        "adversarial": trained,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(trained)} adversarial configs)")


if __name__ == "__main__":
    main()
