"""Existence-proof control #2: linear concept erasure (no SAE, no adversarial training).

Motivation: `eval_no_sae_control.py`'s GRL-based adversarial suppression got
stuck at a suppress-accuracy excess of +0.08 to +0.14 over random-init
regardless of GRL strength (lambda 1 -> 50) -- consistent with the well-known
limitation that adversarial training (Ganin & Lempitsky 2016) only guarantees a
representation fools the SPECIFIC discriminator trained alongside it, not that
the concept is genuinely, information-theoretically absent (this motivated
closed-form concept-erasure methods: INLP, Ravfogel et al. 2020; LEACE,
Belrose et al. 2023).

This script tests suppression via the simplest principled alternative: fit a
linear probe for the label to be suppressed, then project the representation
onto the orthogonal complement of that probe's decision SUBSPACE (one
iteration of INLP; for a binary label the subspace is a single direction, for
an n-class label it is the up-to-(n-1)-dimensional subspace spanned by the
multinomial probe's per-class weight vectors). No adversarial optimization, no
SAE, no reconstruction -- just linear algebra.

This version erases EACH label independently and checks BOTH directions of
the 2x2 cross (does erasing sentiment leave topic intact? does erasing topic
leave sentiment intact?), each against a rank-matched random-subspace-erasure
control (erasing a random subspace of the SAME dimensionality, to confirm any
suppression is attributable to targeting the real concept direction and not
just to removing capacity/dimensionality generically).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch as t

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import SyntheticJointActivationBuffer, standardized_logreg_accuracy  # noqa: E402


def _collect_batch(buf: SyntheticJointActivationBuffer, n_docs: int):
    xs, sents, tops = [], [], []
    got = 0
    while got < n_docs:
        b = next(buf)
        xs.append(b.activations.detach().cpu())
        sents.append(b.sentiment_labels.detach().cpu())
        tops.append(b.topic_labels.detach().cpu())
        got += b.activations.shape[0]
    return t.cat(xs)[:n_docs], t.cat(sents)[:n_docs], t.cat(tops)[:n_docs]


def fit_probe_subspace(x_fit: t.Tensor, y_fit: t.Tensor) -> tuple[t.Tensor, float]:
    """Fit a standardized logistic-regression probe for `y_fit` on `x_fit`
    (binary or multiclass) and return (a) an orthonormal basis (rows) for the
    raw-x-space subspace spanned by its per-class decision directions -- one
    iteration of INLP (Ravfogel et al. 2020), and (b) that SAME fit's own
    training accuracy, so a caller doing repeated rounds does not need a
    second, redundant LogisticRegression fit just to check a stopping
    criterion (this roughly halves per-round cost). Binary labels give a
    rank-1 subspace; an n-class multinomial probe gives up to an n-row
    subspace (sklearn's `coef_` has one row per class for
    `multi_class='multinomial'`, collapsed to rank via QR since the rows need
    not be independent)."""
    from sklearn.linear_model import LogisticRegression

    mu = x_fit.mean(dim=0, keepdim=True)
    sd = x_fit.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_std = ((x_fit - mu) / sd).numpy()
    y_np = y_fit.numpy()
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(x_std, y_np)
    train_acc = float(clf.score(x_std, y_np))
    # coef_ shape: (1, d) for binary, (n_classes, d) for multiclass.
    w_std = t.tensor(np.atleast_2d(clf.coef_), dtype=t.float32)
    # Map each standardized-space row back into raw x-space: standardized
    # score = w . ((x-mu)/sd) = (w/sd) . x - (w/sd).mu, so the raw-space
    # direction for each row is w/sd (translation term is irrelevant to a
    # projection).
    w_raw = w_std / sd  # broadcasts (1,d) against (n_classes,d)
    # Orthonormal basis for the row space via QR of the transpose.
    q, r = t.linalg.qr(w_raw.T)  # q: (d, k), columns orthonormal
    rank = int((r.diagonal().abs() > 1e-5 * r.diagonal().abs().max()).sum().item())
    return q[:, :rank].T.contiguous(), train_acc  # (rank, d), rows orthonormal


def project_out_subspace(x: t.Tensor, basis_rows: t.Tensor) -> t.Tensor:
    """Remove the component of each row of `x` lying in the subspace spanned
    by `basis_rows` (rows assumed orthonormal): x - x @ B^T @ B."""
    return x - (x @ basis_rows.T) @ basis_rows


def random_subspace(activation_dim: int, rank: int, generator: t.Generator) -> t.Tensor:
    """A random rank-matched orthonormal subspace, for the "removing any
    generic capacity would look like this" control."""
    q, _ = t.linalg.qr(t.randn(activation_dim, rank, generator=generator))
    return q.T.contiguous()


def probe_split(x: t.Tensor, y: t.Tensor, n_split: int) -> tuple[float, bool]:
    return standardized_logreg_accuracy(x[:n_split], y[:n_split], x[n_split:], y[n_split:])


def leace_fit(
    x_fit: t.Tensor, y_fit: t.Tensor, eps: float = 1e-4,
) -> dict:
    """LEACE (Belrose et al. 2023, arXiv:2306.03819): closed-form, ONE-SHOT
    linear concept erasure, computed from the full cross-covariance structure
    rather than iteratively hunting for one discriminative direction at a
    time (INLP's approach). Motivated by the finding that INLP converges very
    slowly on real pythia-410m activations (25 rounds only moved sentiment
    accuracy 0.947->0.909, extrapolating to needing several hundred erased
    dimensions to reach chance) -- LEACE's guardedness theorem guarantees
    EVERY linear classifier is driven to exactly the base rate once the
    (whitened) cross-covariance between features and label is zeroed, in a
    single computation, regardless of how many dimensions that structure is
    spread across.

    Returns a dict of the fitted transform's pieces so `leace_erase` can be
    applied to any other tensor (e.g. a held-out probe split) without
    re-fitting: {mu_x, W, W_inv, proj} where `proj` is the projector onto the
    whitened cross-covariance subspace (Q Q^T for an orthonormal basis Q of
    that subspace).
    """
    n_classes = int(t.unique(y_fit).numel())
    if n_classes <= 2:
        z = y_fit.float().unsqueeze(-1)
    else:
        z = t.nn.functional.one_hot(y_fit.long(), num_classes=n_classes).float()
    z = z - z.mean(dim=0, keepdim=True)

    mu_x = x_fit.mean(dim=0, keepdim=True)
    xc = x_fit - mu_x
    n = xc.shape[0]
    sigma_xx = (xc.T @ xc) / (n - 1) + eps * t.eye(xc.shape[1])
    sigma_xz = (xc.T @ z) / (n - 1)  # (d, n_classes_or_1)

    # W = Sigma_xx^{-1/2} via eigendecomposition (symmetric PSD).
    eigvals, eigvecs = t.linalg.eigh(sigma_xx)
    eigvals = eigvals.clamp_min(eps)
    w = eigvecs @ t.diag(eigvals.rsqrt()) @ eigvecs.T
    w_inv = eigvecs @ t.diag(eigvals.sqrt()) @ eigvecs.T

    w_sigma_xz = w @ sigma_xz  # (d, n_classes_or_1), the whitened cross-covariance
    q, r = t.linalg.qr(w_sigma_xz)
    rank = int((r.diagonal().abs() > 1e-6 * r.diagonal().abs().max().clamp_min(1e-12)).sum().item())
    q = q[:, :rank]
    proj = q @ q.T
    return {"mu_x": mu_x, "w": w, "w_inv": w_inv, "proj": proj, "rank": rank}


def leace_erase(x: t.Tensor, fit: dict) -> t.Tensor:
    """Apply a `leace_fit` transform to `x` (fit on a DISJOINT split, applied
    here, exactly like `iterative_nullspace_projection`'s train/apply split).

    r(x) = x - (x - mu_x) @ (W^T P W^+)  [row-batched form; W, W^+ symmetric]
    equivalent to the per-sample column-vector form x - W^+ P W (x - mu_x).
    """
    xc = x - fit["mu_x"]
    correction_matrix = fit["w"] @ fit["proj"] @ fit["w_inv"]  # (d, d)
    return x - xc @ correction_matrix


def iterative_leace(
    x_fit: t.Tensor, y_fit: t.Tensor, x_apply: t.Tensor, max_rounds: int = 5,
    eps: float = 1e-4, verbose: bool = False,
) -> tuple[t.Tensor, int, int]:
    """Repeat LEACE's one-shot closed-form erasure across several rounds,
    refitting on the just-erased `x_fit` each time -- the LEACE analogue of
    `iterative_nullspace_projection`'s INLP loop, but each round removes a
    whole cross-covariance subspace via one linear-algebra computation
    instead of one probe direction via iterative logistic-regression
    optimization. Motivated by finding that a SINGLE LEACE application
    already suppresses real pythia-410m activations far more per erased
    dimension than 25 rounds of INLP (rank 1 LEACE: -0.10 sentiment vs. rank
    25 INLP: -0.04) but does not by itself reach exact chance -- repeating the
    same efficient step should converge much faster than INLP's slow,
    one-direction-at-a-time search. Returns (erased_x_apply, total_rank_removed, rounds_run)."""
    x_fit_cur = x_fit.clone()
    x_apply_cur = x_apply.clone()
    n_classes = int(t.unique(y_fit).numel())
    chance = 1.0 / n_classes
    total_rank = 0
    rounds_run = 0
    for round_idx in range(max_rounds):
        fit = leace_fit(x_fit_cur, y_fit, eps=eps)
        if fit["rank"] == 0:
            break
        x_fit_next = leace_erase(x_fit_cur, fit)
        # Stopping check reuses a cheap standardized-logreg fit on the JUST
        # erased fitting split (same convention as INLP's stopping rule).
        acc, _ = probe_split(x_fit_next, y_fit, int(0.8 * x_fit_next.shape[0]))
        if verbose:
            print(f"    [LEACE round {round_idx}] rank_this_round={fit['rank']} "
                  f"post-round held-out-within-fit acc={acc:.4f} (chance={chance:.4f})", flush=True)
        x_fit_cur = x_fit_next
        x_apply_cur = leace_erase(x_apply_cur, fit)
        total_rank += fit["rank"]
        rounds_run += 1
        if acc < chance + 0.03:
            break
    return x_apply_cur, total_rank, rounds_run


def iterative_nullspace_projection(
    x_fit: t.Tensor, y_fit: t.Tensor, x_apply: t.Tensor, max_rounds: int = 3, verbose: bool = False,
) -> tuple[t.Tensor, t.Tensor, int]:
    """Full Iterative Nullspace Projection (Ravfogel et al. 2020): repeatedly
    (a) fit a probe for `y_fit` on the CURRENTLY-erased `x_fit`, (b) project
    out its subspace from both `x_fit` and `x_apply`, until the refit probe's
    own training accuracy is at chance (it can no longer find any linear
    signal at all) or `max_rounds` is hit. A single round (as tried first
    here) can leave residual signal for a noisy/multi-class concept, because
    one multinomial fit does not always recover the FULL discriminative
    subspace in one shot -- refitting on the already-partially-erased
    activations exposes whatever subspace survived the previous round.
    Returns (erased_x_apply, cumulative_basis, rounds_run)."""
    x_fit_cur = x_fit.clone()
    x_apply_cur = x_apply.clone()
    n_classes = int(t.unique(y_fit).numel())
    chance = 1.0 / n_classes
    cumulative_basis: Optional[t.Tensor] = None
    rounds_run = 0
    for round_idx in range(max_rounds):
        # `train_acc` here reflects the signal STILL PRESENT in x_fit_cur after
        # all PREVIOUS rounds' erasure (this round hasn't erased anything yet)
        # -- reusing this one fit both to build this round's erasure basis AND
        # to decide whether erasing further is still worthwhile avoids a
        # second, redundant LogisticRegression fit per round (roughly halves
        # wall-clock on real, slow-to-converge activations).
        basis, train_acc = fit_probe_subspace(x_fit_cur, y_fit)
        if verbose:
            print(f"    [INLP round {round_idx}] pre-round train_acc={train_acc:.4f} (chance={chance:.4f})",
                  flush=True)
        if train_acc < chance + 0.03:
            break  # negligible signal left; do not spend a round erasing noise
        cumulative_basis = basis if cumulative_basis is None else t.cat([cumulative_basis, basis], dim=0)
        x_fit_cur = project_out_subspace(x_fit_cur, basis)
        x_apply_cur = project_out_subspace(x_apply_cur, basis)
        rounds_run += 1
    if cumulative_basis is None:
        # Even round 0's probe was already at chance -- nothing to erase.
        cumulative_basis = t.zeros(0, x_fit.shape[1])
    return x_apply_cur, cumulative_basis, rounds_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--activation_dim", type=int, default=1024)
    ap.add_argument("--num_topics", type=int, default=6)
    ap.add_argument("--num_sentiments", type=int, default=2)
    ap.add_argument("--doc_batch_size", type=int, default=64)
    ap.add_argument("--n_docs", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", type=str, default="inlp", choices=["inlp", "leace", "iterative_leace"],
                     help="inlp = iterative Nullspace Projection (Ravfogel et al. 2020, rank-1-per-round); "
                          "leace = closed-form one-shot cross-covariance erasure (Belrose et al. 2023); "
                          "iterative_leace = repeat LEACE's closed-form step across several rounds.")
    ap.add_argument("--output", type=str, default=str(ROOT.parent / "runs" / "concept_erasure_control_results.json"))
    args = ap.parse_args()

    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = t.Generator().manual_seed(args.seed + 777)

    buf = SyntheticJointActivationBuffer(
        activation_dim=args.activation_dim, doc_batch_size=args.doc_batch_size,
        num_topics=args.num_topics, num_sentiments=args.num_sentiments,
        topic_signal_scale=1.0, sentiment_signal_scale=1.0, noise_std=1.0,
        seed=args.seed, device="cpu",
    )
    x_all, sent_all, top_all = _collect_batch(buf, args.n_docs)
    n_fit = int(0.5 * x_all.shape[0])
    n_probe = x_all.shape[0] - n_fit
    n_probe_train = int(0.65 * n_probe)

    x_fit, sent_fit, top_fit = x_all[:n_fit], sent_all[:n_fit], top_all[:n_fit]
    x_probe, sent_probe, top_probe = x_all[n_fit:], sent_all[n_fit:], top_all[n_fit:]

    def two_probes(x: t.Tensor) -> tuple[float, float]:
        s_acc, s_conv = probe_split(x, sent_probe, n_probe_train)
        t_acc, t_conv = probe_split(x, top_probe, n_probe_train)
        if not (s_conv and t_conv):
            print("WARNING: a probe did not converge", file=sys.stderr)
        return s_acc, t_acc

    # --- Raw baseline ---
    sent_acc_raw, top_acc_raw = two_probes(x_probe)

    # --- Fit each concept's erasure subspace on the DISJOINT x_fit split,
    # iterating (full INLP) until the refit probe can no longer beat chance
    # on its own fitting split, not just a single round ---
    results: dict[str, dict] = {
        "chance": {"sentiment": 1.0 / args.num_sentiments, "topic": 1.0 / args.num_topics},
        "raw": {"sentiment_acc": sent_acc_raw, "topic_acc": top_acc_raw},
    }

    for target_name, y_fit in (("sentiment", sent_fit), ("topic", top_fit)):
        if args.method == "leace":
            fit = leace_fit(x_fit, y_fit)
            x_erased = leace_erase(x_probe, fit)
            rank = fit["rank"]
            rounds_run = 1
        elif args.method == "iterative_leace":
            x_erased, rank, rounds_run = iterative_leace(x_fit, y_fit, x_probe)
        else:
            x_erased, basis, rounds_run = iterative_nullspace_projection(x_fit, y_fit, x_probe)
            rank = basis.shape[0]
        sent_acc_e, top_acc_e = two_probes(x_erased)

        rand_basis = random_subspace(args.activation_dim, rank, generator)
        x_rand_erased = project_out_subspace(x_probe, rand_basis)
        sent_acc_r, top_acc_r = two_probes(x_rand_erased)

        results[f"erase_{target_name}"] = {
            "method": args.method,
            "erasure_rank": rank,
            "inlp_rounds": rounds_run,
            "targeted": {
                "sentiment_acc": sent_acc_e, "topic_acc": top_acc_e,
                "sentiment_delta_vs_raw": sent_acc_e - sent_acc_raw,
                "topic_delta_vs_raw": top_acc_e - top_acc_raw,
            },
            "random_subspace_control": {
                "sentiment_acc": sent_acc_r, "topic_acc": top_acc_r,
                "sentiment_delta_vs_raw": sent_acc_r - sent_acc_raw,
                "topic_delta_vs_raw": top_acc_r - top_acc_raw,
            },
        }

    results["args"] = vars(args)
    print(json.dumps(results, indent=2))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    print("\n=== Summary table (iterative-INLP-targeted erasure vs. raw baseline) ===")
    print(f"{'erase target':>14} {'rank':>5} {'rounds':>7} {'sent_acc':>9} {'sent_Δ':>8} {'top_acc':>9} {'top_Δ':>8}")
    print(f"{'(raw)':>14} {'':>5} {'':>7} {sent_acc_raw:9.4f} {'':>8} {top_acc_raw:9.4f} {'':>8}")
    for target_name in ("sentiment", "topic"):
        r = results[f"erase_{target_name}"]["targeted"]
        rank = results[f"erase_{target_name}"]["erasure_rank"]
        rounds = results[f"erase_{target_name}"]["inlp_rounds"]
        print(f"{target_name:>14} {rank:5d} {rounds:7d} {r['sentiment_acc']:9.4f} {r['sentiment_delta_vs_raw']:+8.4f} "
              f"{r['topic_acc']:9.4f} {r['topic_delta_vs_raw']:+8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
