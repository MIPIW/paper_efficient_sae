"""Interaction-structure probe: is an SAE latent's firing conjunctive in two primitives?

This is the real-data counterpart of the synthetic alpha knob in
`src/train.py --synthetic_conjunctive`. There, ground-truth conjunctivity is set
by fiat; here it has to be *measured* on a latent whose ground truth is unknown.

Given a target latent's activation (or firing indicator) `y` and two primitive
probe features `a`, `b`, we cross-validate two logistic regressions:

    additive model     y ~ [a, b]
    interaction model  y ~ [a, b, a*b]

and define

    interaction-structure score  Delta-AUC = AUC(interaction) - AUC(additive)

A latent that is genuinely a conjunction of `a` and `b` (the feature-absorption
case: child concept = parent AND differentiator) can only be separated by the
product term, so Delta-AUC is large. A latent that is a linear mixture of the
primitives is already fit by the additive model, so Delta-AUC is ~0.

The obvious objection is that the interaction model simply has one more
parameter and would score higher on *any* target. That is exactly the confound
Hewitt & Liang (2019) introduced control tasks for, so this module always
computes one: the same three-column design matrix, but with the `a*b` column
randomly permuted across samples. The permuted column has an identical marginal
distribution and grants identical added expressivity, while carrying no real
relationship to `y`. The corrected score is

    selectivity = Delta-AUC - Delta-AUC(control)

and `selectivity`, not raw Delta-AUC, is what should be reported and stratified
on. Every probe uses the standardized methodology REPORT.md §9.3 made mandatory
for this project -- train-fold z-scoring plus
`LogisticRegression(max_iter=5000, C=1.0)` -- and never the older unstandardized
fixed-LR probe, whose failure to converge on differently-scaled features
manufactured the spurious results §9.3 retracted.

Self-test:  python src/eval_interaction_structure.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_ITER = 5000
PROBE_C = 1.0


@dataclass
class InteractionScore:
    """Result of one interaction-structure probe on one latent."""

    auc_additive: float
    auc_interaction: float
    auc_control: float
    delta_auc: float
    delta_auc_control: float
    selectivity: float
    n_samples: int
    n_positive: int
    positive_rate: float
    n_splits: int
    all_converged: bool
    degenerate: bool = False
    note: str = ""
    per_fold: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _nan_result(n: int, n_pos: int, n_splits: int, note: str) -> InteractionScore:
    return InteractionScore(
        auc_additive=float("nan"),
        auc_interaction=float("nan"),
        auc_control=float("nan"),
        delta_auc=float("nan"),
        delta_auc_control=float("nan"),
        selectivity=float("nan"),
        n_samples=int(n),
        n_positive=int(n_pos),
        positive_rate=float(n_pos) / n if n else float("nan"),
        n_splits=int(n_splits),
        all_converged=False,
        degenerate=True,
        note=note,
    )


def binarize_latent(latent: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Turn a latent's activations into a firing indicator (`activation > threshold`).

    Top-K SAE latents are exactly zero whenever they are not in the top-k, so
    `threshold=0.0` recovers "this latent fired on this input", which is the
    event the absorption literature reasons about.
    """
    return (np.asarray(latent, dtype=np.float64).ravel() > float(threshold)).astype(np.int64)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score using TRAIN-fold statistics only (no test leakage)."""
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def _cv_out_of_fold_scores(
    design: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, List[float], bool]:
    """Stratified k-fold out-of-fold decision scores for one design matrix.

    Returns (out_of_fold_scores, per_fold_aucs, all_converged). Pooling the
    out-of-fold scores into a single AUC is more stable than averaging per-fold
    AUCs when the positive class is rare, which it usually is for a single SAE
    latent; the per-fold AUCs are returned too so the spread stays visible.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    oof = np.zeros(y.shape[0], dtype=np.float64)
    per_fold: List[float] = []
    converged = True
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(design, y):
        x_tr, x_te = _standardize(design[train_idx], design[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        clf = LogisticRegression(max_iter=MAX_ITER, C=PROBE_C)
        clf.fit(x_tr, y_tr)
        scores = clf.decision_function(x_te)
        oof[test_idx] = scores
        converged = converged and bool(int(np.max(clf.n_iter_)) < MAX_ITER)
        if np.unique(y_te).size == 2:
            per_fold.append(float(roc_auc_score(y_te, scores)))
    return oof, per_fold, converged


def interaction_structure_score(
    latent: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    binarize_threshold: Optional[float] = 0.0,
    n_control_permutations: int = 3,
    min_per_class: int = 5,
) -> InteractionScore:
    """Compute the Delta-AUC interaction-structure score for one latent.

    Args:
        latent: (n,) latent activations, or an already-binary firing indicator
            if `binarize_threshold` is None.
        a, b: (n,) primitive probe features (e.g. two probe-direction
            projections of the residual stream -- a parent concept and a
            differentiator).
        n_splits: folds for stratified cross-validation.
        seed: controls the CV shuffle and the control-task permutations.
        binarize_threshold: threshold turning `latent` into a firing indicator;
            pass None if `latent` is already 0/1.
        n_control_permutations: number of Hewitt & Liang (2019) control-task
            repetitions; their mean Delta-AUC is the expressivity baseline that
            gets subtracted off to form `selectivity`.
        min_per_class: below this many samples in either class the probe is not
            trustworthy and a degenerate (NaN) result is returned rather than a
            number that would be read as evidence.

    Returns:
        InteractionScore. The field to report and to stratify on is
        `selectivity`; `delta_auc` alone is confounded by the interaction
        model's extra degree of freedom.

    Serves the RQ of whether KronSAE's advantage tracks *conjunctive* ground
    truth: latents scoring high here are the ones a bilinear encoder should be
    able to represent with a single atom while a flat encoder cannot.
    """
    from sklearn.metrics import roc_auc_score

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    y_raw = np.asarray(latent, dtype=np.float64).ravel()
    if not (a.shape == b.shape == y_raw.shape):
        raise ValueError(f"latent/a/b must be 1-D and the same length, got {y_raw.shape}, {a.shape}, {b.shape}")

    y = binarize_latent(y_raw, binarize_threshold) if binarize_threshold is not None else y_raw.astype(np.int64)
    n = int(y.shape[0])
    n_pos = int(y.sum())
    n_neg = n - n_pos

    if min(n_pos, n_neg) < max(min_per_class, n_splits):
        return _nan_result(n, n_pos, n_splits, note=f"too few samples in a class (pos={n_pos}, neg={n_neg})")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return _nan_result(n, n_pos, n_splits, note="non-finite values in a or b")

    ab = a * b
    additive = np.stack([a, b], axis=1)
    interaction = np.stack([a, b, ab], axis=1)

    oof_add, folds_add, conv_add = _cv_out_of_fold_scores(additive, y, n_splits, seed)
    oof_int, folds_int, conv_int = _cv_out_of_fold_scores(interaction, y, n_splits, seed)
    auc_add = float(roc_auc_score(y, oof_add))
    auc_int = float(roc_auc_score(y, oof_int))

    rng = np.random.default_rng(seed + 977)
    control_aucs: List[float] = []
    conv_ctl = True
    for _ in range(max(1, n_control_permutations)):
        # Hewitt & Liang (2019) control task: same design, same added capacity,
        # relationship to y destroyed by permutation.
        ctl = np.stack([a, b, rng.permutation(ab)], axis=1)
        oof_ctl, _, c = _cv_out_of_fold_scores(ctl, y, n_splits, seed)
        control_aucs.append(float(roc_auc_score(y, oof_ctl)))
        conv_ctl = conv_ctl and c
    auc_ctl = float(np.mean(control_aucs))

    delta = auc_int - auc_add
    delta_ctl = auc_ctl - auc_add
    return InteractionScore(
        auc_additive=auc_add,
        auc_interaction=auc_int,
        auc_control=auc_ctl,
        delta_auc=delta,
        delta_auc_control=delta_ctl,
        selectivity=delta - delta_ctl,
        n_samples=n,
        n_positive=n_pos,
        positive_rate=float(n_pos) / n,
        n_splits=int(n_splits),
        all_converged=bool(conv_add and conv_int and conv_ctl),
        per_fold={
            "additive": folds_add,
            "interaction": folds_int,
            "control_aucs": control_aucs,
        },
    )


def batch_interaction_scores(
    latents: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    latent_ids: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Dict[int, InteractionScore]:
    """Run `interaction_structure_score` over the columns of a (n_samples, n_latents) matrix."""
    latents = np.asarray(latents, dtype=np.float64)
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2-D (n_samples, n_latents), got {latents.shape}")
    ids = list(range(latents.shape[1])) if latent_ids is None else list(latent_ids)
    if len(ids) != latents.shape[1]:
        raise ValueError("latent_ids length must match latents.shape[1]")
    return {int(fid): interaction_structure_score(latents[:, j], a, b, **kwargs) for j, fid in enumerate(ids)}


def stratify_by_interaction_tercile(
    records: Sequence[Dict[str, Any]],
    score_key: str = "interaction_selectivity",
    metric_keys: Sequence[str] = ("absorption_fraction", "hedging_score", "fvu"),
) -> Dict[str, Any]:
    """Split records into interaction-score terciles and summarize each metric per tercile.

    This is the core analysis the project's flow depends on, kept as a standalone
    reusable function rather than inlined in the benchmark harness so it can be
    applied to any (architecture, combine_rule, model) result set and unit-tested
    on its own.

    The reasoning it supports: if KronSAE's bilinear encoder helps *because of*
    conjunctive ground-truth structure, then its advantage on absorption/hedging
    should be concentrated in the TOP interaction-score tercile and absent in the
    BOTTOM one. A uniform improvement across all three terciles would instead
    point at some generic capacity or optimization difference and would falsify
    the conjunctive explanation.

    Records whose score is NaN (degenerate probes) are excluded from the terciles
    and counted separately, so an unusable probe never silently lands in a bucket.

    Returns a dict with `cutoffs`, per-tercile summaries (`bottom`/`middle`/`top`),
    and `top_minus_bottom` differences for each metric.
    """
    rows = list(records)
    scores = np.array([r.get(score_key, np.nan) for r in rows], dtype=np.float64)
    valid = np.isfinite(scores)
    n_dropped = int((~valid).sum())

    out: Dict[str, Any] = {
        "score_key": score_key,
        "metric_keys": list(metric_keys),
        "n_records": len(rows),
        "n_valid": int(valid.sum()),
        "n_dropped_nan_score": n_dropped,
    }
    if valid.sum() < 3:
        out["error"] = "fewer than 3 records with a finite interaction score; terciles undefined"
        return out

    valid_scores = scores[valid]
    lo, hi = np.quantile(valid_scores, [1.0 / 3.0, 2.0 / 3.0])
    out["cutoffs"] = {"lower_tercile": float(lo), "upper_tercile": float(hi)}

    buckets: Dict[str, List[int]] = {"bottom": [], "middle": [], "top": []}
    for i, (row_valid, s) in enumerate(zip(valid, scores)):
        if not row_valid:
            continue
        if s <= lo:
            buckets["bottom"].append(i)
        elif s >= hi:
            buckets["top"].append(i)
        else:
            buckets["middle"].append(i)

    summaries: Dict[str, Dict[str, Any]] = {}
    for name, idxs in buckets.items():
        summary: Dict[str, Any] = {"n": len(idxs)}
        sc = scores[idxs] if idxs else np.array([])
        summary["interaction_score_mean"] = float(sc.mean()) if sc.size else float("nan")
        for key in metric_keys:
            vals = np.array([rows[i].get(key, np.nan) for i in idxs], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            summary[key] = {
                "n": int(vals.size),
                "mean": float(vals.mean()) if vals.size else float("nan"),
                "std": float(vals.std()) if vals.size else float("nan"),
                "median": float(np.median(vals)) if vals.size else float("nan"),
            }
        summaries[name] = summary
    out["terciles"] = summaries
    out["top_minus_bottom"] = {
        key: summaries["top"][key]["mean"] - summaries["bottom"][key]["mean"] for key in metric_keys
    }
    return out


# ---------------------------------------------------------------------------
# CPU-only self-test
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Tiny synthetic check that the probe is well-formed. NOT a scientific validation.

    Constructs four latents with known structure and asserts only that the
    metric behaves the way its definition requires:
      * a purely non-additive (XOR) latent has selectivity clearly > 0, because
        no linear function of (a, b) separates it at all
      * an AND latent has selectivity > 0 but *modest*: the AND quadrant is
        already partly cut off by a linear boundary, so the additive model gets
        most of the way there on its own. This is the honest expectation for a
        real absorption case and is asserted as such, not inflated.
      * a purely additive latent has selectivity ~0
      * a pure-noise latent has selectivity ~0 and AUCs ~0.5
      * the control task's Delta-AUC is small (the third column buys little on its own)
    """
    rng = np.random.default_rng(0)
    n = 800
    a = rng.normal(size=n)
    b = rng.normal(size=n)

    latents = {
        "xor_nonadditive": ((a > 0) ^ (b > 0)).astype(float),
        "conjunctive_and": ((a > 0) & (b > 0)).astype(float),
        "additive_linear": (a + 0.5 * b > 0).astype(float),
        "pure_noise": (rng.normal(size=n) > 0).astype(float),
    }

    ok = True
    results: Dict[str, Dict[str, Any]] = {}
    for name, lat in latents.items():
        r = interaction_structure_score(lat, a, b, n_splits=5, seed=7, n_control_permutations=3)
        results[name] = r.to_dict()
        finite = all(
            np.isfinite(v) for v in (r.auc_additive, r.auc_interaction, r.auc_control, r.selectivity)
        )
        in_range = all(0.0 <= v <= 1.0 for v in (r.auc_additive, r.auc_interaction, r.auc_control))
        small_control = abs(r.delta_auc_control) < 0.05
        checks = {
            "finite": finite,
            "aucs_in_[0,1]": in_range,
            "selectivity_in_[-1,1]": -1.0 <= r.selectivity <= 1.0,
            "control_delta_small": small_control,
            "converged": r.all_converged,
        }
        for cname, passed in checks.items():
            ok = ok and passed
            if not passed:
                print(f"  FAIL {name}/{cname}")
        print(
            f"  {name:16s} auc_add={r.auc_additive:.3f} auc_int={r.auc_interaction:.3f} "
            f"auc_ctl={r.auc_control:.3f} dAUC={r.delta_auc:+.3f} dAUC_ctl={r.delta_auc_control:+.3f} "
            f"selectivity={r.selectivity:+.3f}"
        )

    xor = results["xor_nonadditive"]["selectivity"]
    conj = results["conjunctive_and"]["selectivity"]
    add = results["additive_linear"]["selectivity"]
    noise = results["pure_noise"]["selectivity"]
    ordering = {
        "xor_selectivity_large": xor > 0.20,
        "xor_additive_model_at_chance": abs(results["xor_nonadditive"]["auc_additive"] - 0.5) < 0.10,
        "and_selectivity_positive": conj > 0.01,
        "additive_selectivity_small": abs(add) < 0.05,
        "noise_selectivity_small": abs(noise) < 0.05,
        "xor_beats_and_beats_additive": xor > conj > add,
    }
    for cname, passed in ordering.items():
        print(f"  {cname}: {'PASS' if passed else 'FAIL'}")
        ok = ok and passed

    # degenerate input must return NaN, not a number
    deg = interaction_structure_score(np.zeros(200), a[:200], b[:200])
    deg_ok = deg.degenerate and not np.isfinite(deg.selectivity)
    print(f"  degenerate_returns_nan: {'PASS' if deg_ok else 'FAIL'}")
    ok = ok and deg_ok

    # tercile stratification
    recs = [
        {"interaction_selectivity": s, "absorption_fraction": 1.0 - s, "hedging_score": s, "fvu": 0.5}
        for s in np.linspace(0.0, 0.9, 9)
    ] + [{"interaction_selectivity": float("nan"), "absorption_fraction": 0.0, "hedging_score": 0.0, "fvu": 0.5}]
    strat = stratify_by_interaction_tercile(recs)
    strat_ok = (
        strat["n_dropped_nan_score"] == 1
        and strat["terciles"]["top"]["n"] == 3
        and strat["terciles"]["bottom"]["n"] == 3
        and strat["terciles"]["middle"]["n"] == 3
        and strat["top_minus_bottom"]["absorption_fraction"] < 0
        and strat["top_minus_bottom"]["hedging_score"] > 0
    )
    print(f"  tercile_stratification: {'PASS' if strat_ok else 'FAIL'} -> {json.dumps(strat['top_minus_bottom'])}")
    ok = ok and strat_ok

    print("\nINTERACTION_STRUCTURE SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
