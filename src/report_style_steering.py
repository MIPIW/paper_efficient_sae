"""Turn judged steering generations into the 2x2 selectivity matrix, under the pilot's reporting rules.

This is a separate script from `eval_style_steering.py` on purpose. The generation run is expensive
and the aggregation is where the previous pilot went wrong, so the aggregation needs to be re-runnable
and inspectable without regenerating anything. Four rules are enforced here rather than left to
whoever reads the JSON:

1. FLUENCY GATING. Every concept metric appears twice: `all` over all generations, and `fluent` over
   the judge-fluent subset, with `n_fluent` beside it. When no generation in a condition is fluent,
   the concept cell is emitted as null and printed as "--", never as a number. This is not caution
   for its own sake: in the previous pilot the layer-12 probe reported a confident P(positive)=0.161
   on text whose self-perplexity was 122,068. The probe is a linear function of a mean-pooled
   activation; it has no way to abstain on garbage, so the abstention has to be imposed from outside.

2. DELTAS AGAINST alpha=0, NOT THE NO-HOOK BASELINE. Routing activations through the dictionary
   changes sentiment, MATTR and perplexity before any steering happens. The alpha=0 condition
   measures precisely that reconstruction cost, so it is the zero point for every Delta. The no-hook
   baseline is still reported, as the size of the reconstruction cost itself -- which is worth
   knowing, but is not the thing steering should be credited or blamed for.

3. SIGNS REPORTED SEPARATELY. Negative and positive alphas are never averaged into one "steering
   effect". In the previous pilot negative pushes worked and positive pushes failed catastrophically;
   a signed mean would have shown a mild success and concealed both halves of what happened.

4. THE OFF-DIAGONAL IS THE RESULT. Pushing P should move sentiment and leave MATTR alone; pushing Q
   the reverse. A large diagonal with an equally large off-diagonal is not partial success, it is the
   null result this project has repeatedly obtained -- the branches moving the shared downstream
   representation rather than two separable concepts.

The success criterion is evaluated exactly as it was fixed in advance, and its three clauses are
reported individually so a near-miss is legible rather than collapsing to a bare False:
pushing P shifts judged sentiment by >= 1.0 Likert point, while changing continuation MATTR by less
than the alpha=0 reconstruction noise, at >= 70% fluency retention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SENTIMENT_DELTA_THRESHOLD = 1.0   # Likert points, on the judge's 1-5 scale
FLUENCY_RETENTION_THRESHOLD = 0.70


def _mean(xs) -> float:
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def summarize(cond: dict) -> dict:
    """Per-condition metrics, computed over all entries and over the fluent subset separately."""
    entries = cond["entries"]
    fluent = [e for e in entries if e.get("judge_fluent")]

    def block(subset):
        return {
            "n": len(subset),
            "judge_sentiment_1_5": _mean(e.get("judge_rating", float("nan")) for e in subset),
            "probe_sentiment_p_positive": _mean(e["probe_sentiment_p_positive"] for e in subset),
            "mattr": _mean(e["mattr"] for e in subset),
            "n_scored_for_mattr": sum(1 for e in subset if e["mattr"] == e["mattr"]),
            "self_perplexity": _mean(e["self_perplexity"] for e in subset),
        }

    return {
        "label": cond["label"],
        "branch": cond.get("branch"),
        "alpha": cond.get("alpha"),
        "n_total": len(entries),
        "n_fluent": len(fluent),
        "fluent_rate": len(fluent) / max(len(entries), 1),
        "n_too_short_for_mattr": cond.get("n_too_short_for_mattr"),
        "all": block(entries),
        # Emitted as None, not NaN and not 0.0, when nothing was fluent: a concept metric over an
        # empty fluent subset is undefined and must print as "--" downstream.
        "fluent": block(fluent) if fluent else None,
    }


def delta(cell: dict | None, ref: dict | None, key: str) -> float | None:
    if cell is None or ref is None:
        return None
    a, b = cell[key], ref[key]
    if a != a or b != b:
        return None
    return float(a - b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default=str(ROOT / "runs" / "style_steering_judged.json"))
    ap.add_argument("--out_path", type=str, default=str(ROOT / "runs" / "style_steering_report.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.in_path).read_text())
    summaries = [summarize(c) for c in data["conditions"]]
    by_label = {s["label"]: s for s in summaries}

    baseline = by_label.get("baseline_no_hook")
    zero = {b: next((s for s in summaries if s["branch"] == b and s["alpha"] == 0.0), None)
            for b in ("P", "Q")}

    # Reconstruction cost: what alpha=0 alone does relative to no hook. This is the noise floor that
    # the success criterion's MATTR clause is measured against.
    recon_cost = {}
    for b in ("P", "Q"):
        if zero[b] is None or baseline is None:
            continue
        recon_cost[b] = {
            "d_judge_sentiment": delta(zero[b]["fluent"], baseline["fluent"], "judge_sentiment_1_5"),
            "d_mattr": delta(zero[b]["fluent"], baseline["fluent"], "mattr"),
            "d_mattr_all": delta(zero[b]["all"], baseline["all"], "mattr"),
            "d_self_perplexity": delta(zero[b]["fluent"], baseline["fluent"], "self_perplexity"),
            "fluent_rate_alpha0": zero[b]["fluent_rate"],
            "fluent_rate_no_hook": baseline["fluent_rate"],
        }
    # The MATTR noise floor. Both branches' alpha=0 conditions decode through the same dictionary, so
    # they are two independent estimates of the same quantity; take the larger in magnitude, which is
    # the conservative choice (it makes the criterion's MATTR clause harder to fail spuriously and
    # easier to fail genuinely).
    mattr_floor_candidates = [abs(v["d_mattr"]) for v in recon_cost.values()
                              if v.get("d_mattr") is not None]
    mattr_noise_floor = max(mattr_floor_candidates) if mattr_floor_candidates else None

    matrix = []
    for s in summaries:
        if s["branch"] is None or s["alpha"] == 0.0:
            continue
        ref = zero[s["branch"]]
        row = {
            "branch_pushed": s["branch"],
            "alpha": s["alpha"],
            "sign": "negative" if s["alpha"] < 0 else "positive",
            "n_fluent": s["n_fluent"],
            "n_total": s["n_total"],
            "fluent_rate": s["fluent_rate"],
            "fluency_retention_vs_alpha0": (
                s["fluent_rate"] / ref["fluent_rate"] if ref and ref["fluent_rate"] > 0 else None
            ),
            "n_too_short_for_mattr": s["n_too_short_for_mattr"],
            # Concept deltas over the fluent subset -- the ones that count.
            "d_judge_sentiment_fluent": delta(s["fluent"], ref["fluent"] if ref else None,
                                              "judge_sentiment_1_5"),
            "d_mattr_fluent": delta(s["fluent"], ref["fluent"] if ref else None, "mattr"),
            "d_probe_sentiment_fluent": delta(s["fluent"], ref["fluent"] if ref else None,
                                              "probe_sentiment_p_positive"),
            "d_self_perplexity_fluent": delta(s["fluent"], ref["fluent"] if ref else None,
                                              "self_perplexity"),
            # And over everything, so the gap between the two is visible rather than hidden.
            "d_judge_sentiment_all": delta(s["all"], ref["all"] if ref else None, "judge_sentiment_1_5"),
            "d_mattr_all": delta(s["all"], ref["all"] if ref else None, "mattr"),
            "d_self_perplexity_all": delta(s["all"], ref["all"] if ref else None, "self_perplexity"),
        }
        matrix.append(row)

    # --- success criterion, clause by clause, on the P pushes only ---
    criterion = {
        "definition": (
            "pushing P shifts judged sentiment by >=1.0 Likert point (fluent subset, vs P's own "
            "alpha=0) while |Delta MATTR| < the alpha=0 reconstruction noise floor, at >=70% "
            "fluency retention relative to alpha=0"),
        "mattr_noise_floor": mattr_noise_floor,
        "per_alpha": [],
        "met": False,
    }
    for row in matrix:
        if row["branch_pushed"] != "P":
            continue
        ds, dm, ret = (row["d_judge_sentiment_fluent"], row["d_mattr_fluent"],
                       row["fluency_retention_vs_alpha0"])
        clauses = {
            "sentiment_shift": (abs(ds) >= SENTIMENT_DELTA_THRESHOLD) if ds is not None else None,
            "mattr_within_noise": (abs(dm) < mattr_noise_floor)
                                  if (dm is not None and mattr_noise_floor is not None) else None,
            "fluency_retained": (ret >= FLUENCY_RETENTION_THRESHOLD) if ret is not None else None,
        }
        criterion["per_alpha"].append({
            "alpha": row["alpha"], "d_judge_sentiment": ds, "d_mattr": dm,
            "fluency_retention": ret, "clauses": clauses,
            "all_clauses_met": all(v is True for v in clauses.values()),
        })
    criterion["met"] = any(a["all_clauses_met"] for a in criterion["per_alpha"])

    out = {
        "source": args.in_path,
        "baseline_no_hook": baseline,
        "reconstruction_cost_alpha0_vs_no_hook": recon_cost,
        "conditions": summaries,
        "selectivity_matrix": matrix,
        "success_criterion": criterion,
    }
    Path(args.out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # --- console rendering ---
    fmt = lambda v, p=3: "--" if v is None or v != v else f"{v:+.{p}f}"  # noqa: E731
    print(f"\n{'condition':<20} {'n_flu/n':>9} {'flu%':>6} {'judge1-5':>9} {'MATTR':>8} "
          f"{'probe':>7} {'ppl':>10} {'short':>6}")
    for s in summaries:
        blk = s["fluent"] or s["all"]
        tag = "" if s["fluent"] else "  (NO FLUENT)"
        j = "--" if not s["fluent"] else f"{blk['judge_sentiment_1_5']:.2f}"
        m = "--" if not s["fluent"] else f"{blk['mattr']:.4f}"
        p = "--" if not s["fluent"] else f"{blk['probe_sentiment_p_positive']:.3f}"
        print(f"{s['label']:<20} {s['n_fluent']:>4}/{s['n_total']:<4} {100*s['fluent_rate']:>5.1f}% "
              f"{j:>9} {m:>8} {p:>7} {blk['self_perplexity']:>10.1f} "
              f"{str(s['n_too_short_for_mattr']):>6}{tag}")

    print(f"\nReconstruction cost (alpha=0 vs no hook), the zero point for every Delta below:")
    for b, v in recon_cost.items():
        print(f"  branch {b}: d_judge={fmt(v['d_judge_sentiment'],2)} d_MATTR={fmt(v['d_mattr'],4)} "
              f"d_ppl={fmt(v['d_self_perplexity'],1)} fluent {100*v['fluent_rate_no_hook']:.0f}%"
              f"->{100*v['fluent_rate_alpha0']:.0f}%")
    print(f"  MATTR noise floor used by the criterion: "
          f"{'--' if mattr_noise_floor is None else f'{mattr_noise_floor:.4f}'}")

    print(f"\n2x2 selectivity (fluent subset, Delta vs each branch's own alpha=0):")
    print(f"{'push':>5} {'alpha':>6} {'n_flu':>6} {'flu_ret':>8} {'D judge sent':>13} {'D MATTR':>10} {'D ppl':>10}")
    for row in matrix:
        print(f"{row['branch_pushed']:>5} {row['alpha']:>6.0f} {row['n_fluent']:>6} "
              f"{fmt(row['fluency_retention_vs_alpha0'],2):>8} "
              f"{fmt(row['d_judge_sentiment_fluent'],2):>13} {fmt(row['d_mattr_fluent'],4):>10} "
              f"{fmt(row['d_self_perplexity_fluent'],1):>10}")

    print(f"\nSuccess criterion: {'MET' if criterion['met'] else 'NOT MET'}")
    for a in criterion["per_alpha"]:
        c = a["clauses"]
        print(f"  P alpha={a['alpha']:>4.0f}: sentiment_shift={c['sentiment_shift']} "
              f"mattr_within_noise={c['mattr_within_noise']} fluency_retained={c['fluency_retained']}")
    print(f"\nSaved {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
