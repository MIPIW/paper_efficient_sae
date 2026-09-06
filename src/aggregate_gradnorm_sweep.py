"""Aggregate the GradNorm alpha sweep + reused random-init control into one JSON."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "runs" / "gradnorm_sweep"
LRSWEEP_PATH = ROOT / "runs" / "lambda_recon_sweep_results.json"
OUT_PATH = ROOT / "runs" / "gradnorm_sweep_results.json"

ALPHAS = [("0.12", "0p12"), ("0.5", "0p5"), ("1.5", "1p5")]
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
    out["total_steps"] = payload["args"]["total_steps"]
    out["skip_training"] = payload["args"].get("synthetic_skip_training", False)
    out["gradnorm"] = payload["args"].get("gradnorm", False)
    out["gradnorm_alpha"] = payload["args"].get("gradnorm_alpha")
    out["gradnorm_lr"] = payload["args"].get("gradnorm_lr")
    if "gradnorm_task_names" in res:
        out["gradnorm_task_names"] = res["gradnorm_task_names"]
        out["gradnorm_final_w"] = res["gradnorm_final_w"]
        out["gradnorm_weight_history"] = res["gradnorm_weight_history"]
    return out


def main() -> None:
    with LRSWEEP_PATH.open() as f:
        lr = json.load(f)

    trained = {}
    for label, tag in ALPHAS:
        d = SWEEP_DIR / f"alpha_{tag}"
        if (d / "synthetic_diagnostic_results.json").exists():
            trained[label] = load(d)

    control_check = None
    ccdir = SWEEP_DIR / "randinit_control_check"
    if (ccdir / "synthetic_diagnostic_results.json").exists():
        control_check = load(ccdir)

    payload = {
        "description": (
            "GradNorm (Chen et al., 2018) alpha sweep on the synthetic diagnostic "
            "(joint_dpo_cross, kron_joint, h=128/m=8/n=16, dpo_beta=2.0, 5000 steps, seed=42, "
            "CPU). GradNorm learns one weight per task over "
            "{recon, p_collect_sentiment, p_contrast_topic, q_collect_topic, q_contrast_sentiment} "
            "with W = P and Q encoder projections; the learned w_i fully supersede lambda_recon, "
            "lambda_sup and dpo_w_*. Random-init control reused verbatim from "
            "runs/lambda_recon_sweep_results.json (identical construction/eval/probe seeds; "
            "re-verified by runs/gradnorm_sweep/randinit_control_check)."
        ),
        "chance": {"sentiment": 0.5, "topic": 1.0 / 6.0},
        "random_init_control": lr["random_init_control"],
        "random_init_control_other_seeds": lr["random_init_control_other_seeds"],
        "random_init_control_recheck_with_gradnorm_flag": control_check,
        "reference_9_2_baseline": lr["reference_9_2_baseline"],
        "reference_9_5_lambda_recon_sweep": lr["trained"],
        "gradnorm_trained": trained,
    }
    with OUT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUT_PATH}")

    hdr = f"{'config':>26} {'P-sent':>7} {'Q-sent':>7} {'P-top':>7} {'Q-top':>7} {'FVU':>8}"
    print(hdr)

    def row(name, r):
        print(
            f"{name:>26} {r['p_sentiment_acc']:7.3f} {r['q_sentiment_acc']:7.3f} "
            f"{r['p_topic_acc']:7.3f} {r['q_topic_acc']:7.3f} {r.get('recon_fvu', float('nan')):8.4f}"
        )

    row("random-init control", lr["random_init_control"])
    if control_check:
        row("rand-init recheck", control_check)
    row("9.2 baseline (lrecon 1.0)", lr["trained"]["1.0"])
    for label, _ in ALPHAS:
        if label in trained:
            row(f"gradnorm alpha={label}", trained[label])
        else:
            print(f"{'gradnorm alpha=' + label:>26} (missing)")

    for label, _ in ALPHAS:
        if label not in trained:
            continue
        r = trained[label]
        names = r["gradnorm_task_names"]
        print(f"\nalpha={label} learned w_i ({', '.join(names)}):")
        hist = r["gradnorm_weight_history"]
        for h in hist[:: max(1, len(hist) // 10)] + [hist[-1]]:
            print(f"  step {h['step']:>5}: " + " ".join(f"{v:6.3f}" for v in h["w"]))


if __name__ == "__main__":
    main()
