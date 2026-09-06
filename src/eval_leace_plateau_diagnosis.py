"""Diagnose WHY LEACE plateaus well above chance on real pythia-410m activations
(sentiment 0.947->0.84, topic 0.583->0.28) instead of reaching exact chance the
way it does on synthetic data (both drop to exact chance).

Two competing explanations:
  (A) Finite-sample generalization gap. LEACE's guardedness theorem is a
      POPULATION-statistics guarantee (true Sigma_xx, Sigma_xz). We estimate
      these from x_fit (~10000 docs, d=1024) and apply the resulting fixed
      projection to a DISJOINT x_probe split. If Sigma_xx/Sigma_xz estimated
      from x_fit don't fully match x_probe's true statistics (high-dimensional
      covariance estimation is sample-hungry: d=1024 means Sigma_xx has
      ~524K free parameters), the erasure that is EXACT for x_fit becomes only
      approximate for x_probe -- a generalization gap, not a failure of the
      method itself.
  (B) Genuine residual (nonlinear or theorem-violating) signal. If even
      evaluating on the SAME data the projection was fit on (in-sample, no
      train/apply split at all) still leaves a logistic-regression probe able
      to beat chance, that is NOT a generalization-gap story -- it would mean
      either an implementation bug, or that real activations violate an
      assumption the guardedness theorem needs (e.g. that the specific
      finite-sample point estimates of Sigma_xx/Sigma_xz are the relevant
      "true" statistics evaluated against are inconsistent, or that a convex
      loss family the proof covers still admits above-chance ACCURACY, even
      at the population optimum, on non-Gaussian real data as opposed to the
      loss-value equivalence the theorem actually proves).

This script isolates the two by comparing, for the SAME fitted LEACE
projection:
  1. in-sample:  fit + apply LEACE on x_fit; probe (train+test split BOTH
     drawn from the erased x_fit) -- population-vs-finite-sample effects
     from a SECOND split are removed here since fit and apply share the exact
     same erasure statistics.
  2. out-of-sample: fit LEACE on x_fit; apply to x_probe (the disjoint split);
     probe there (this is exactly what eval_concept_erasure_pythia.py already
     reports).
  3. A fitting-sample-size sweep (n_fit in {2000, 5000, 10000, all}) to see
     whether the out-of-sample plateau shrinks toward the in-sample number as
     n_fit grows -- direct evidence for hypothesis (A) if so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from eval_pilot_q_liveness import set_seed  # noqa: E402
from eval_concept_erasure_control import leace_fit, leace_erase, probe_split  # noqa: E402
from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc  # noqa: E402


def load_pooled(cache_path: Path, max_docs: int) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    batches = t.load(cache_path, map_location="cpu")
    x_list, sent_list, top_list = [], [], []
    for b in batches:
        x = b["x"].to(dtype=t.float32)
        ids = b["token_doc_ids"]
        pooled = _mean_pool_by_doc(x, ids, b["n_docs"]).cpu().float()
        x_list.append(pooled)
        sent_list.append(b["sentiment"])
        top_list.append(b["topic"])
    x_all = t.cat(x_list)[:max_docs]
    sent_all = t.cat(sent_list)[:max_docs]
    top_all = t.cat(top_list)[:max_docs]
    return x_all, sent_all, top_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--activation_cache", type=str,
                     default=str(ROOT.parent / "runs" / "pythia_erasure_activation_cache.pt"))
    ap.add_argument("--max_probe_docs", type=int, default=20000)
    ap.add_argument("--n_fit_sweep", type=int, nargs="+", default=[2000, 5000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", type=str,
                     default=str(ROOT.parent / "runs" / "leace_plateau_diagnosis.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    x_all, sent_all, top_all = load_pooled(Path(args.activation_cache), args.max_probe_docs)
    n_sentiments = int(t.unique(sent_all).numel())
    n_topics = int(t.unique(top_all).numel())
    print(f"n_docs={x_all.shape[0]} n_sentiments={n_sentiments} n_topics={n_topics}", flush=True)

    n_fit_full = int(0.5 * x_all.shape[0])
    n_probe = x_all.shape[0] - n_fit_full
    n_probe_train = int(0.65 * n_probe)
    x_probe, sent_probe, top_probe = x_all[n_fit_full:], sent_all[n_fit_full:], top_all[n_fit_full:]

    results: dict = {
        "chance": {"sentiment": 1.0 / n_sentiments, "topic": 1.0 / n_topics},
        "n_docs": int(x_all.shape[0]),
        "n_fit_full": n_fit_full,
        "n_probe": n_probe,
    }

    def cross_cov_magnitude(x: t.Tensor, y: t.Tensor, n_classes: int) -> tuple[float, float]:
        """Cross-covariance norm between (centered) x and (centered, one-hot
        if multiclass) y -- an evaluation metric immune to the probe-
        overfitting artifact found for sentiment (where raw C=1.0 accuracy of
        0.84 was shown, via a C-regularization sweep, to be pure noise-fit
        variance rather than real residual signal; cross-covariance directly
        measures what LEACE's own guarantee is actually about)."""
        if n_classes <= 2:
            z = y.float().unsqueeze(-1)
        else:
            z = t.nn.functional.one_hot(y.long(), num_classes=n_classes).float()
        z = z - z.mean(dim=0, keepdim=True)
        xc = x - x.mean(dim=0, keepdim=True)
        cc = (xc.T @ z) / (x.shape[0] - 1)
        return cc.abs().max().item(), cc.norm().item()

    for target_name, y_all_full in (("sentiment", sent_all), ("topic", top_all)):
        print(f"\n=== target: {target_name} ===", flush=True)
        target_results: dict = {}
        n_classes = int(t.unique(y_all_full).numel())
        raw_max_abs, raw_norm = cross_cov_magnitude(x_probe, sent_probe if target_name == "sentiment" else top_probe, n_classes)
        print(f"  raw cross-cov on x_probe (pre-erasure): max_abs={raw_max_abs:.4f} norm={raw_norm:.4f}", flush=True)
        target_results["raw_cross_cov"] = {"max_abs": raw_max_abs, "norm": raw_norm}

        # --- (1) in-sample: fit and evaluate on the SAME data ---
        x_fit_full, y_fit_full = x_all[:n_fit_full], y_all_full[:n_fit_full]
        fit = leace_fit(x_fit_full, y_fit_full)
        x_fit_erased = leace_erase(x_fit_full, fit)
        n_split = int(0.8 * x_fit_erased.shape[0])
        in_sample_acc, in_sample_conv = probe_split(x_fit_erased, y_fit_full, n_split)
        print(f"  in-sample (fit==apply, rank={fit['rank']}): acc={in_sample_acc:.4f} converged={in_sample_conv}",
              flush=True)
        target_results["in_sample"] = {"rank": fit["rank"], "acc": in_sample_acc, "converged": in_sample_conv}

        # --- (2) out-of-sample: fit on x_fit_full, apply+evaluate on x_probe ---
        y_probe = sent_probe if target_name == "sentiment" else top_probe
        x_probe_erased = leace_erase(x_probe, fit)
        out_acc, out_conv = probe_split(x_probe_erased, y_probe, n_probe_train)
        print(f"  out-of-sample (fit on n={n_fit_full}, apply to disjoint n={n_probe}): "
              f"acc={out_acc:.4f} converged={out_conv}", flush=True)
        target_results["out_of_sample_full_fit"] = {"rank": fit["rank"], "acc": out_acc, "converged": out_conv}

        # --- (3) fitting-sample-size sweep: does the out-of-sample plateau
        # shrink toward the in-sample number as n_fit grows? ---
        sweep = {}
        for n_fit in (*args.n_fit_sweep, n_fit_full):
            if n_fit > n_fit_full:
                continue
            x_fit_n, y_fit_n = x_all[:n_fit], y_all_full[:n_fit]
            fit_n = leace_fit(x_fit_n, y_fit_n)
            x_probe_erased_n = leace_erase(x_probe, fit_n)
            acc_n, conv_n = probe_split(x_probe_erased_n, y_probe, n_probe_train)
            cc_max_n, cc_norm_n = cross_cov_magnitude(x_probe_erased_n, y_probe, n_classes)
            surviving_frac = cc_norm_n / max(raw_norm, 1e-12)
            print(f"  n_fit={n_fit:6d} -> out-of-sample acc={acc_n:.4f} (rank={fit_n['rank']}) converged={conv_n} | "
                  f"cross-cov norm={cc_norm_n:.4f} ({surviving_frac*100:.2f}% of raw survives)", flush=True)
            sweep[str(n_fit)] = {
                "rank": fit_n["rank"], "acc": acc_n, "converged": conv_n,
                "cross_cov_max_abs": cc_max_n, "cross_cov_norm": cc_norm_n,
                "surviving_frac_of_raw": surviving_frac,
            }
        target_results["n_fit_sweep"] = sweep

        results[target_name] = target_results

    print(json.dumps(results, indent=2))
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
