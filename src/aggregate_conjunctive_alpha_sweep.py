"""Aggregate the conjunctive-alpha sweep (Part A) + its random-init controls into one JSON.

Consumes the `synthetic_diagnostic_results.json` files written by
`src/train.py --synthetic_diagnostic --synthetic_conjunctive`, one run directory
per (arm, alpha) cell, and produces the single table the Part A research question
is decided on:

    for each alpha:  acc_kron(alpha) - acc_flat(alpha),
                     each first corrected against its OWN random-init control.

The correction is not optional. REPORT.md §9.4 established that a random linear
map into a comparable-or-larger dimension already preserves whatever is linearly
decodable in its input, so an absolute probe accuracy carries no evidence on its
own; only the trained-minus-random-init delta does. The prediction under test is
that `kron_minus_flat_delta` is ~0 at alpha=0 (linear ground truth: the bilinear
encoder buys nothing) and grows with alpha (conjunctive ground truth: the
bilinear encoder can express the product term a flat encoder cannot).

Expected layout (created by the sweep driver, see `--sweep_dir`)::

    runs/<sweep_dir>/trained_alpha_0p00/synthetic_diagnostic_results.json
    runs/<sweep_dir>/control_alpha_0p00/synthetic_diagnostic_results.json
    runs/<sweep_dir>/trained_alpha_0p25/...

Run:  python src/aggregate_conjunctive_alpha_sweep.py --sweep_dir runs/conj_alpha
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ALPHAS = ["0.0", "0.25", "0.5", "0.75", "1.0"]
TRAINER_NAMES = ["flat_joint", "kron_joint"]
ARM_KEYS = ["dense", "topk", "raw", "p", "q"]


def alpha_tag(alpha: str) -> str:
    """`0.25` -> `0p25`, matching this project's existing sweep directory naming."""
    return alpha.replace(".", "p")


def load_cell(run_dir: Path) -> Optional[dict[str, Any]]:
    """Load one run's conjunctive block, or None if the run has not been produced yet."""
    path = run_dir / "synthetic_diagnostic_results.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    conj = payload.get("conjunctive")
    if not conj:
        return None
    out: dict[str, Any] = {
        "conjunctive_alpha": payload["args"].get("conjunctive_alpha"),
        "conjunctive_balance": payload["args"].get("conjunctive_balance"),
        "total_steps": payload["args"].get("total_steps"),
        "skip_training": payload["args"].get("synthetic_skip_training", False),
        "combine_rule": payload["args"].get("combine_rule"),
        "k": payload["args"].get("k"),
        "flat_dict_size": payload["args"].get("flat_dict_size"),
        "trainers": {},
    }
    for name, metrics in conj.items():
        row = {f"{key}_acc": metrics.get(f"{key}_conjunctive_acc") for key in ARM_KEYS}
        row["all_converged"] = all(
            metrics.get(f"{key}_conjunctive_converged", True) for key in ARM_KEYS
        )
        row["dict_size"] = metrics.get("dict_size")
        row["k"] = metrics.get("k")
        row["eval_marginal_pos_frac"] = metrics.get("eval_marginal_pos_frac")
        out["trainers"][name] = row
    first = next(iter(conj.values()))
    out["buffer_diagnostics"] = first.get("buffer_diagnostics", {})
    return out


def _delta(trained: Optional[float], control: Optional[float]) -> Optional[float]:
    if trained is None or control is None:
        return None
    return trained - control


def build_alpha_row(trained: Optional[dict], control: Optional[dict], probe_arm: str) -> dict[str, Any]:
    """One alpha's summary: per-architecture control-corrected accuracy and the Kron-minus-flat gap.

    `probe_arm` selects which latent readout to summarize ('dense' = pre-top-k
    code, 'topk' = post-top-k sparse code, 'raw' = no SAE reference).
    """
    row: dict[str, Any] = {"probe_arm": probe_arm}
    for name in TRAINER_NAMES:
        tr = (trained or {}).get("trainers", {}).get(name, {})
        ct = (control or {}).get("trainers", {}).get(name, {})
        row[f"{name}_trained"] = tr.get(f"{probe_arm}_acc")
        row[f"{name}_control"] = ct.get(f"{probe_arm}_acc")
        row[f"{name}_delta"] = _delta(row[f"{name}_trained"], row[f"{name}_control"])
    row["kron_minus_flat_trained"] = _delta(row["kron_joint_trained"], row["flat_joint_trained"])
    row["kron_minus_flat_delta"] = _delta(row["kron_joint_delta"], row["flat_joint_delta"])
    row["marginal_pos_frac"] = (trained or control or {}).get("buffer_diagnostics", {}).get(
        "marginal_pos_frac"
    )
    row["linear_rule_acc"] = (trained or control or {}).get("buffer_diagnostics", {}).get(
        "linear_rule_acc"
    )
    row["conjunctive_rule_acc"] = (trained or control or {}).get("buffer_diagnostics", {}).get(
        "conjunctive_rule_acc"
    )
    return row


def print_table(rows: dict[str, dict[str, Any]], probe_arm: str) -> None:
    print(f"\nprobe arm = {probe_arm}  (all accuracies; delta = trained - random-init control)")
    hdr = (
        f"{'alpha':>6} {'P(y=1)':>7} {'linrule':>8} {'prodrule':>9} "
        f"{'flat_tr':>8} {'flat_ct':>8} {'flat_d':>8} "
        f"{'kron_tr':>8} {'kron_ct':>8} {'kron_d':>8} {'K-F_d':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for alpha, r in rows.items():
        def f(key: str) -> str:
            v = r.get(key)
            return "     --" if v is None else f"{v:8.3f}"

        print(
            f"{alpha:>6} "
            f"{(r.get('marginal_pos_frac') or float('nan')):7.3f} "
            f"{(r.get('linear_rule_acc') or float('nan')):8.3f} "
            f"{(r.get('conjunctive_rule_acc') or float('nan')):9.3f} "
            f"{f('flat_joint_trained')}{f('flat_joint_control')}{f('flat_joint_delta')}"
            f"{f('kron_joint_trained')}{f('kron_joint_control')}{f('kron_joint_delta')}"
            f"{f('kron_minus_flat_delta')}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep_dir", type=str, default=str(ROOT / "runs" / "conj_alpha"))
    ap.add_argument("--alphas", nargs="*", default=DEFAULT_ALPHAS)
    ap.add_argument("--trained_prefix", type=str, default="trained_alpha_")
    ap.add_argument("--control_prefix", type=str, default="control_alpha_")
    ap.add_argument("--probe_arms", nargs="*", default=["dense", "topk", "raw"])
    ap.add_argument("--output_json", type=str, default=None)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_path = Path(args.output_json) if args.output_json else sweep_dir / "conjunctive_alpha_sweep_results.json"

    cells: dict[str, dict[str, Any]] = {}
    for alpha in args.alphas:
        tag = alpha_tag(alpha)
        cells[alpha] = {
            "trained": load_cell(sweep_dir / f"{args.trained_prefix}{tag}"),
            "control": load_cell(sweep_dir / f"{args.control_prefix}{tag}"),
        }

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for probe_arm in args.probe_arms:
        summary[probe_arm] = {
            alpha: build_alpha_row(c["trained"], c["control"], probe_arm)
            for alpha, c in cells.items()
        }

    payload = {
        "description": (
            "Conjunctive-alpha sweep (Part A). y_alpha = 1[(1-alpha)*p_A + alpha*(p_A*p_B/s) > offset], "
            "with p_A/p_B read off two orthonormal primitive directions disjoint from the "
            "sentiment/topic blocks, s fit empirically so the product term matches p_A's scale, "
            "and offset holding P(y=1) at ~0.5 at every alpha. y_alpha is never supervised. "
            "Every accuracy is a standardized LogisticRegression(max_iter=5000, C=1.0) on "
            "train-z-scored, doc-mean-pooled latents (REPORT.md §9.3). The control arm is the "
            "identical run with --synthetic_skip_training (zero optimizer steps), which is the "
            "baseline REPORT.md §9.4 requires results to be read against instead of chance."
        ),
        "chance": 0.5,
        "sweep_dir": str(sweep_dir),
        "alphas": args.alphas,
        "summary": summary,
        "cells": cells,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for probe_arm in args.probe_arms:
        print_table(summary[probe_arm], probe_arm)


def _self_test() -> int:
    """CPU-only self-test: build synthetic sweep payloads on disk, aggregate, assert the shape."""
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        sweep = Path(tmp) / "conj_alpha"
        for alpha, kron_tr, flat_tr in [("0.0", 0.90, 0.90), ("1.0", 0.80, 0.55)]:
            for prefix, kron, flat in [("trained_alpha_", kron_tr, flat_tr), ("control_alpha_", 0.50, 0.50)]:
                d = sweep / f"{prefix}{alpha_tag(alpha)}"
                d.mkdir(parents=True)
                payload = {
                    "args": {"conjunctive_alpha": float(alpha), "total_steps": 10, "k": 8},
                    "conjunctive": {
                        "flat_joint": {
                            "flat": 0,
                            "dense_conjunctive_acc": flat,
                            "topk_conjunctive_acc": flat,
                            "raw_conjunctive_acc": 0.5,
                            "dense_conjunctive_converged": True,
                            "buffer_diagnostics": {"marginal_pos_frac": 0.5, "linear_rule_acc": 0.99,
                                                   "conjunctive_rule_acc": 0.49},
                        },
                        "kron_joint": {
                            "dense_conjunctive_acc": kron,
                            "topk_conjunctive_acc": kron,
                            "raw_conjunctive_acc": 0.5,
                            "dense_conjunctive_converged": True,
                            "buffer_diagnostics": {"marginal_pos_frac": 0.5, "linear_rule_acc": 0.99,
                                                   "conjunctive_rule_acc": 0.49},
                        },
                    },
                }
                (d / "synthetic_diagnostic_results.json").write_text(json.dumps(payload), encoding="utf-8")

        rows = {
            a: build_alpha_row(
                load_cell(sweep / f"trained_alpha_{alpha_tag(a)}"),
                load_cell(sweep / f"control_alpha_{alpha_tag(a)}"),
                "dense",
            )
            for a in ("0.0", "1.0")
        }
        print_table(rows, "dense")

        checks = {
            "alpha0_no_kron_advantage": abs(rows["0.0"]["kron_minus_flat_delta"]) < 1e-9,
            "alpha1_kron_advantage_positive": rows["1.0"]["kron_minus_flat_delta"] > 0.0,
            "delta_is_control_corrected": abs(rows["1.0"]["kron_joint_delta"] - 0.30) < 1e-9,
            "missing_cell_is_none": build_alpha_row(None, None, "dense")["kron_minus_flat_delta"] is None,
        }
        for name, passed in checks.items():
            print(f"  {name}: {'PASS' if passed else 'FAIL'}")
            ok = ok and passed
    print("\nAGGREGATE_CONJUNCTIVE_ALPHA_SWEEP SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--self_test" in sys.argv:
        raise SystemExit(_self_test())
    main()
