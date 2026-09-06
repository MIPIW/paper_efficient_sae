"""Sweep driver for the conjunctive-structure experiment: one command per cell.

Two sweep modes, one entry point, because both halves of the investigation are
indexed by the same kind of grid:

  --mode alpha       Part A. One cell per (combine_rule, alpha, arm), launched as
                     `src/train.py --synthetic_diagnostic --synthetic_conjunctive`.
                     The `control` arm is the same cell with
                     `--synthetic_skip_training`, which REPORT.md §9.4 makes
                     mandatory: absolute probe accuracies carry no evidence, only
                     the trained-minus-random-init delta does. Aggregate with
                     `src/aggregate_conjunctive_alpha_sweep.py`.

  --mode absorption  Part B. One cell per (checkpoint, model_name), launched as
                     `src/eval_absorption_hedging.py`. Each cell writes its own
                     result JSON; `--aggregate_only` merges them and runs the
                     cross-architecture tercile comparison below.

The comparison this whole flow turns on lives in `compare_cells_by_tercile`: for
each interaction-structure tercile, how much better (or worse) is the Kron cell
than the flat cell? The conjunctive explanation predicts the gap is concentrated
in the top tercile and absent in the bottom. A flat gap across all three
terciles falsifies it.

Run:
    python src/run_absorption_benchmark.py --mode alpha --dry_run
    python src/run_absorption_benchmark.py --mode absorption --dry_run
    python src/run_absorption_benchmark.py --mode absorption --aggregate_only
    python src/run_absorption_benchmark.py --self_test
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_ALPHAS = ["0.0", "0.25", "0.5", "0.75", "1.0"]
DEFAULT_COMBINE_RULES = ["mand", "concat", "mor", "mnand", "mnor"]
TERCILE_METRIC_KEYS = ("absorption_fraction", "full_absorption_rate", "hedging_score", "fvu")


def alpha_tag(alpha: str) -> str:
    """`0.25` -> `0p25`, matching this project's existing sweep directory naming."""
    return alpha.replace(".", "p")


# ---------------------------------------------------------------------------
# cell construction
# ---------------------------------------------------------------------------
def build_alpha_cells(args) -> List[Dict[str, Any]]:
    """One Part A cell per (combine_rule, alpha, arm). Arms are `trained` and `control`.

    The two arms are byte-identical except for `--synthetic_skip_training`, and
    share the eval-activation seed and probe-split seed by construction inside
    `train.py`, so the delta between them isolates what training did.
    """
    cells: List[Dict[str, Any]] = []
    for rule in args.combine_rules:
        for alpha in args.alphas:
            for arm in ("trained", "control"):
                name = f"{arm}_alpha_{alpha_tag(alpha)}"
                out_dir = Path(args.sweep_dir) / (rule if len(args.combine_rules) > 1 else "") / name
                cmd = [
                    sys.executable, str(SRC / "train.py"),
                    "--synthetic_diagnostic", "--joint",
                    "--joint_mode", "dpo_cross",
                    "--synthetic_conjunctive",
                    "--conjunctive_alpha", str(alpha),
                    "--conjunctive_balance", args.conjunctive_balance,
                    "--combine_rule", rule,
                    "--k", str(args.k),
                    "--flat_dict_size", str(args.flat_dict_size),
                    "--joint_h", str(args.joint_h),
                    "--joint_m", str(args.joint_m),
                    "--joint_n", str(args.joint_n),
                    "--total_steps", str(args.total_steps),
                    "--seed", str(args.seed),
                    "--output_dir", str(out_dir),
                ]
                if arm == "control":
                    cmd.append("--synthetic_skip_training")
                cells.append({
                    "mode": "alpha", "arm": arm, "combine_rule": rule, "alpha": alpha,
                    "name": name, "output_dir": str(out_dir), "cmd": cmd,
                    "result_json": str(out_dir / "synthetic_diagnostic_results.json"),
                })
    return cells


def build_absorption_cells(args) -> List[Dict[str, Any]]:
    """One Part B cell per (checkpoint, model_name).

    `model_name` is swept rather than fixed so the cross-model-family question
    (does the same interaction-structure -> absorption relationship hold on
    gemma-2-2b as on pythia-410m?) is one more axis of the same grid, not a
    separate script.
    """
    cells: List[Dict[str, Any]] = []
    for model_name in args.model_names:
        for checkpoint in args.checkpoint_names:
            name = f"{model_name.replace('/', '_')}__{checkpoint}"
            out_dir = Path(args.sweep_dir)
            cmd = [
                sys.executable, str(SRC / "eval_absorption_hedging.py"),
                "--run_dir", args.run_dir,
                "--checkpoint_step", str(args.checkpoint_step),
                "--checkpoint_names", checkpoint,
                "--model_name", model_name,
                "--layer", str(args.layer),
                "--device", args.device,
                "--llm_dtype", args.llm_dtype,
                "--llm_batch_size", str(args.llm_batch_size),
                "--max_activation_words", str(args.max_activation_words),
                "--n_probe_splits", str(args.n_probe_splits),
                "--n_control_permutations", str(args.n_control_permutations),
                "--seed", str(args.seed),
                "--output_dir", str(out_dir),
                "--output_name", f"{name}.json",
            ]
            cells.append({
                "mode": "absorption", "model_name": model_name, "checkpoint": checkpoint,
                "name": name, "output_dir": str(out_dir), "cmd": cmd,
                "result_json": str(out_dir / f"{name}.json"),
            })
    return cells


# ---------------------------------------------------------------------------
# cross-cell analysis
# ---------------------------------------------------------------------------
def load_absorption_cells(sweep_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Merge every per-cell absorption/hedging JSON in `sweep_dir` into one dict keyed by SAE name."""
    merged: Dict[str, Dict[str, Any]] = {}
    for path in sorted(sweep_dir.glob("*.json")):
        if path.name.endswith("_merged.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for name, cell in payload.get("cells", {}).items():
            merged[f"{path.stem}:{name}"] = {**cell, "settings": payload.get("settings", {})}
    return merged


def compare_cells_by_tercile(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
    metric_keys: Sequence[str] = TERCILE_METRIC_KEYS,
) -> Dict[str, Any]:
    """Per-tercile `candidate - reference` differences for each metric.

    `reference` is normally the flat `AutoEncoderTopK` cell and `candidate` a
    `KronAutoEncoderTopK` cell. Both dicts are the `cells[...]` entries produced
    by `src/eval_absorption_hedging.py`.

    The output's `top_minus_bottom_of_diff` field is the single number the
    conjunctive hypothesis lives or dies on: it is the extent to which Kron's
    advantage over flat is *larger in the top interaction tercile than in the
    bottom*. Positive-and-large supports "the bilinear encoder helps because the
    structure is conjunctive"; ~0 says the advantage, if any, is generic.

    Sign convention is left to the caller: absorption and hedging are
    pathologies (lower is better) while a raw metric difference is reported
    as-is, so downstream reporting must state the direction explicitly rather
    than assume it.
    """
    out: Dict[str, Any] = {
        "reference_sae": reference.get("sae_name") or reference.get("combine_rule"),
        "candidate_sae": candidate.get("sae_name") or candidate.get("combine_rule"),
        "reference_combine_rule": reference.get("combine_rule"),
        "candidate_combine_rule": candidate.get("combine_rule"),
        "metric_keys": list(metric_keys),
    }
    ref_t = reference.get("summary", {}).get("by_interaction_tercile", {}).get("terciles")
    cand_t = candidate.get("summary", {}).get("by_interaction_tercile", {}).get("terciles")
    if not ref_t or not cand_t:
        out["error"] = "one or both cells lack a tercile summary (too few scored cases?)"
        return out

    diffs: Dict[str, Dict[str, float]] = {}
    for tercile in ("bottom", "middle", "top"):
        diffs[tercile] = {
            key: float(cand_t[tercile][key]["mean"] - ref_t[tercile][key]["mean"]) for key in metric_keys
        }
        diffs[tercile]["n_reference"] = float(ref_t[tercile]["n"])
        diffs[tercile]["n_candidate"] = float(cand_t[tercile]["n"])
    out["diff_by_tercile"] = diffs
    out["top_minus_bottom_of_diff"] = {
        key: diffs["top"][key] - diffs["bottom"][key] for key in metric_keys
    }
    return out


def print_comparison(cmp: Dict[str, Any]) -> None:
    if "error" in cmp:
        print(f"  {cmp['error']}")
        return
    keys = cmp["metric_keys"]
    print(f"\n{cmp['candidate_sae']} minus {cmp['reference_sae']} (candidate - reference)")
    hdr = f"  {'tercile':>10} " + " ".join(f"{k:>20}" for k in keys)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tercile in ("bottom", "middle", "top"):
        row = cmp["diff_by_tercile"][tercile]
        print(f"  {tercile:>10} " + " ".join(f"{row[k]:20.4f}" for k in keys))
    print(f"  {'top-bottom':>10} " + " ".join(f"{cmp['top_minus_bottom_of_diff'][k]:20.4f}" for k in keys))


# ---------------------------------------------------------------------------
def run_cells(cells: Sequence[Dict[str, Any]], dry_run: bool, continue_on_error: bool) -> Dict[str, Any]:
    """Execute (or just print) each cell's command, returning a per-cell status log."""
    log: Dict[str, Any] = {}
    for cell in cells:
        printable = " ".join(shlex.quote(part) for part in cell["cmd"])
        if dry_run:
            print(printable)
            log[cell["name"]] = {"status": "dry_run", "cmd": printable}
            continue
        Path(cell["output_dir"]).mkdir(parents=True, exist_ok=True)
        print(f"\n=== {cell['name']} ===\n{printable}", flush=True)
        proc = subprocess.run(cell["cmd"])
        status = "ok" if proc.returncode == 0 else f"failed(returncode={proc.returncode})"
        log[cell["name"]] = {"status": status, "cmd": printable, "result_json": cell["result_json"]}
        if proc.returncode != 0 and not continue_on_error:
            print(f"Cell {cell['name']} failed; stopping (pass --continue_on_error to keep going).")
            break
    return log


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", type=str, default="absorption", choices=["alpha", "absorption"])
    ap.add_argument("--sweep_dir", type=str, default=None)
    ap.add_argument("--dry_run", action="store_true", help="Print the per-cell commands without running them.")
    ap.add_argument("--aggregate_only", action="store_true", help="Skip execution; merge and compare existing cell JSONs.")
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # --- alpha (Part A) ---
    ap.add_argument("--alphas", nargs="*", default=DEFAULT_ALPHAS)
    ap.add_argument("--combine_rules", nargs="*", default=["mand"])
    ap.add_argument("--conjunctive_balance", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--total_steps", type=int, default=5000)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--flat_dict_size", type=int, default=16384)
    ap.add_argument("--joint_h", type=int, default=128)
    ap.add_argument("--joint_m", type=int, default=8)
    ap.add_argument("--joint_n", type=int, default=16)

    # --- absorption (Part B) ---
    ap.add_argument("--run_dir", type=str, default=str(ROOT / "runs" / "pilot_balanced_v2_420k"))
    ap.add_argument("--checkpoint_step", type=int, default=420000)
    ap.add_argument("--checkpoint_names", nargs="*", default=["flat_pilot_sentiment", "kron_pilot_sentiment"])
    ap.add_argument("--model_names", nargs="*", default=["pythia-410m"],
                    help="Add gemma-2-2b here for the cross-model-family axis.")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--llm_dtype", type=str, default="float32")
    ap.add_argument("--llm_batch_size", type=int, default=32)
    ap.add_argument("--max_activation_words", type=int, default=2000)
    ap.add_argument("--n_probe_splits", type=int, default=5)
    ap.add_argument("--n_control_permutations", type=int, default=3)

    # --- comparison ---
    ap.add_argument("--reference_cell", type=str, default=None,
                    help="Substring matching the flat baseline cell key for compare_cells_by_tercile.")
    ap.add_argument("--candidate_cells", nargs="*", default=None,
                    help="Substrings matching the Kron cell keys to compare against the reference.")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.self_test:
        return _self_test()

    if args.sweep_dir is None:
        args.sweep_dir = str(ROOT / "runs" / ("conj_alpha" if args.mode == "alpha" else "absorption_hedging"))
    sweep_dir = Path(args.sweep_dir)

    cells = build_alpha_cells(args) if args.mode == "alpha" else build_absorption_cells(args)
    print(f"{len(cells)} cell(s) in mode={args.mode}, sweep_dir={sweep_dir}")

    log: Dict[str, Any] = {}
    if not args.aggregate_only:
        log = run_cells(cells, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
    if args.dry_run:
        return 0

    if args.mode == "alpha":
        print(
            "\nAlpha cells complete. Aggregate with:\n"
            f"  {sys.executable} {SRC / 'aggregate_conjunctive_alpha_sweep.py'} "
            f"--sweep_dir {sweep_dir} --alphas {' '.join(args.alphas)}"
        )
        return 0

    merged = load_absorption_cells(sweep_dir)
    comparisons: List[Dict[str, Any]] = []
    if args.reference_cell and args.candidate_cells:
        ref_key = next((k for k in merged if args.reference_cell in k), None)
        if ref_key is None:
            print(f"No cell key matching --reference_cell={args.reference_cell!r}; skipping comparison.")
        else:
            for pattern in args.candidate_cells:
                for key in (k for k in merged if pattern in k and k != ref_key):
                    cmp = compare_cells_by_tercile(merged[ref_key], merged[key])
                    cmp["reference_key"], cmp["candidate_key"] = ref_key, key
                    comparisons.append(cmp)
                    print_comparison(cmp)

    out_path = sweep_dir / "absorption_hedging_merged.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"run_log": log, "cells": merged, "comparisons": comparisons}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
def _self_test() -> int:
    """CPU-only self-test: cell construction, merging, and the tercile comparison.

    Builds a synthetic pair of cells where the Kron candidate's advantage is
    deliberately concentrated in the TOP interaction tercile, and asserts the
    comparison recovers that concentration.
    """
    import tempfile

    ok = True
    checks: Dict[str, bool] = {}
    ap = build_arg_parser()

    # --- alpha cells ---
    a = ap.parse_args(["--mode", "alpha", "--alphas", "0.0", "1.0", "--sweep_dir", "/tmp/x"])
    alpha_cells = build_alpha_cells(a)
    checks["alpha_cell_count"] = len(alpha_cells) == 4  # 2 alphas x {trained, control}
    checks["alpha_control_has_skip_training"] = all(
        ("--synthetic_skip_training" in c["cmd"]) == (c["arm"] == "control") for c in alpha_cells
    )
    checks["alpha_cells_carry_alpha_flag"] = all("--conjunctive_alpha" in c["cmd"] for c in alpha_cells)
    checks["alpha_dir_tags_use_p_naming"] = any("alpha_0p0" in c["output_dir"] for c in alpha_cells)

    # --- absorption cells ---
    b = ap.parse_args([
        "--mode", "absorption", "--model_names", "pythia-410m", "gemma-2-2b",
        "--checkpoint_names", "flat_x", "kron_x", "--sweep_dir", "/tmp/y",
    ])
    abs_cells = build_absorption_cells(b)
    checks["absorption_cell_count"] = len(abs_cells) == 4  # 2 models x 2 checkpoints
    checks["absorption_cells_carry_model_name"] = all("--model_name" in c["cmd"] for c in abs_cells)
    checks["cross_model_family_supported"] = any("gemma-2-2b" in c["cmd"] for c in abs_cells)

    # --- comparison: advantage concentrated in the top tercile ---
    def make_cell(name: str, rule: Optional[str], top_bonus: float) -> Dict[str, Any]:
        from eval_absorption_hedging import summarize_records

        records = []
        for i, s in enumerate(np.linspace(0.0, 0.8, 9)):
            in_top = i >= 6
            records.append({
                "interaction_selectivity": float(s),
                "absorption_fraction": 0.6 - (top_bonus if in_top else 0.0),
                "full_absorption_rate": 0.3,
                "hedging_score": 0.2,
                "fvu": 0.25,
            })
        return {"sae_name": name, "combine_rule": rule, "summary": summarize_records(records)}

    flat = make_cell("flat", None, 0.0)
    kron = make_cell("kron", "mand", 0.3)  # only the top tercile improves
    cmp = compare_cells_by_tercile(flat, kron)
    print_comparison(cmp)
    checks["comparison_has_all_terciles"] = set(cmp["diff_by_tercile"]) == {"bottom", "middle", "top"}
    checks["bottom_tercile_no_advantage"] = abs(cmp["diff_by_tercile"]["bottom"]["absorption_fraction"]) < 1e-9
    checks["top_tercile_advantage"] = cmp["diff_by_tercile"]["top"]["absorption_fraction"] < -0.25
    checks["top_minus_bottom_recovers_concentration"] = (
        cmp["top_minus_bottom_of_diff"]["absorption_fraction"] < -0.25
    )
    checks["missing_summary_gives_error"] = "error" in compare_cells_by_tercile({}, {})

    # --- merging from disk ---
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "cellA.json").write_text(json.dumps({"settings": {"model_name": "pythia-410m"},
                                                  "cells": {"flat_x": flat}}), encoding="utf-8")
        (d / "cellB.json").write_text(json.dumps({"settings": {"model_name": "pythia-410m"},
                                                  "cells": {"kron_x": kron}}), encoding="utf-8")
        (d / "notjson.txt").write_text("ignore me", encoding="utf-8")
        merged = load_absorption_cells(d)
        checks["merge_finds_both_cells"] = len(merged) == 2
        checks["merge_keys_are_namespaced"] = all(":" in k for k in merged)

    # --- dry-run execution path must not launch anything ---
    log = run_cells(alpha_cells[:1], dry_run=True, continue_on_error=False)
    checks["dry_run_does_not_execute"] = all(v["status"] == "dry_run" for v in log.values())

    print()
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
        ok = ok and bool(passed)
    print("\nRUN_ABSORPTION_BENCHMARK SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
