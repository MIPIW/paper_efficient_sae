"""Aggregate the CAGrad synthetic-diagnostic sweep into runs/cagrad_sweep_results.json.

Safe to call repeatedly / after each config: it re-reads whatever result files exist so
partial sweeps are already durable on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "runs" / "cagrad_sweep"
OUT = ROOT / "runs" / "cagrad_sweep_results.json"

REFERENCES = {
    "randinit_control_cpu (runs/gradnorm_sweep/randinit_control_check)": {
        "p_sentiment_acc": 0.967,
        "q_sentiment_acc": 0.997,
        "p_topic_acc": 0.828,
        "q_topic_acc": 0.973,
        "recon_fvu": 1.010,
        "source": "REPORT.md 9.5 random-init control (0 training steps)",
    },
    "no_intervention_baseline (REPORT.md 9.2)": {
        "p_sentiment_acc": 0.941,
        "q_sentiment_acc": 0.972,
        "p_topic_acc": 0.786,
        "q_topic_acc": 0.838,
        "recon_fvu": 0.953,
        "source": "REPORT.md 9.2 / 9.5 baseline row",
    },
}

KEYS = ["p_sentiment_acc", "q_sentiment_acc", "p_topic_acc", "q_topic_acc"]


def fvu(metrics: dict) -> float | None:
    for k in ("recon_fvu", "fvu", "reconstruction_fvu", "synthetic_fvu"):
        if k in metrics:
            return metrics[k]
    return None


def main() -> None:
    payload: dict = {"references": REFERENCES, "runs": {}}
    check = SWEEP_DIR / "cagrad_correctness_check.json"
    if check.exists():
        text = check.read_text()
        payload["correctness_check"] = {
            "script": "src/check_cagrad.py",
            "verdict": "PASS" if "CAGRAD CHECK: PASS" in text else "FAIL",
            "cases": json.loads(text.split("\nCAGRAD CHECK:")[0]),
        }
    for path in sorted(SWEEP_DIR.glob("*/synthetic_diagnostic_results.json")):
        tag = path.parent.name
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # partial write while a run is in flight
            payload["runs"][tag] = {"error": str(exc)}
            continue
        res = data.get("results", {}).get("kron_joint")
        if res is None:
            continue
        args = data.get("args", {})
        hist = res.get("cagrad_history", [])
        row = {k: res.get(k) for k in KEYS}
        row["recon_fvu"] = fvu(res)
        row["cagrad"] = args.get("cagrad", False)
        row["cagrad_c"] = args.get("cagrad_c")
        row["total_steps"] = args.get("total_steps")
        row["skip_training"] = args.get("synthetic_skip_training", False)
        row["seed"] = args.get("seed")
        if hist:
            n = max(1, len(hist) // 10)
            tail = hist[-n:]
            row["cagrad_final_w"] = {
                f"w_{name}": sum(h[f"w_{name}"] for h in tail) / len(tail)
                for name in res.get("cagrad_task_names", [])
            }
            row["cagrad_min_improve_d_mean"] = sum(h["min_improve_d"] for h in hist) / len(hist)
            row["cagrad_min_improve_sum_mean"] = sum(h["min_improve_sum"] for h in hist) / len(hist)
            row["cagrad_frac_steps_d_beats_sum"] = sum(
                1 for h in hist if h["min_improve_d"] >= h["min_improve_sum"]
            ) / len(hist)
            row["cagrad_solvers"] = sorted({h["solver"] for h in hist})
        payload["runs"][tag] = row

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT} ({len(payload['runs'])} runs)")


if __name__ == "__main__":
    main()
