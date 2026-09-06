"""Two-branch steering selectivity: push P, push Q, and see which concept actually moves.

`eval_style_probe_matrix.py` asks what is *readable* from each branch. Readability is necessary but
not sufficient: a probe can find a concept in a branch that has no causal grip on the model's
behaviour. This script asks the causal question. It patches the trained dictionary into pythia-410m
at layer 12, pushes ONE branch along ONE direction, generates text, and measures BOTH concepts in
the output. The claim under test is that the 2x2 is DIAGONAL:

                     Delta judged sentiment   Delta continuation MATTR
    push P (content)        large                     ~zero
    push Q (style)          ~zero                     large

Differences from `eval_neutral_generation.py`, which this is otherwise a close relative of:

1. It steers P as well as Q. The existing `SteeringHook` only has a Q path, because until this pilot
   the P branch was never the thing under test. The P path here pushes `p_pos` along its own signed
   direction, recombines through `ae._combine`, re-applies top-k and decodes -- structurally the
   mirror of the Q path, so any asymmetry in the results is a property of the model rather than of
   the intervention.
2. Each branch is steered along the direction of the concept IT was supervised on: P along sentiment,
   Q along style. Pushing a branch along a concept it was never trained to carry would test nothing.
3. Style is measured in the generated text, not just probed. `mattr_and_length` is imported directly
   from `build_style_dataset.py` -- the exact function that produced the training label -- so the
   generation-side metric and the supervision signal cannot drift apart through a reimplementation.
   Continuations are ~40 tokens, far below the 25-word minimum used when labelling the corpus, so
   `--mattr_min_words` lowers that floor; the count of continuations still too short to score is
   recorded per condition, because a condition whose MATTR is computed on half its samples is not
   comparable to one computed on all of them.

Three reporting rules are baked into the output structure rather than left to the analysis step,
because the previous pilot in this project produced misleading numbers by not observing them:

* Every concept metric is emitted twice: over all generations, and over the judge-fluent subset only
  (populated by `judge_generations_qwen.py` downstream), with n_fluent carried alongside. A linear
  probe returns a confident sentiment for token salad -- this actually happened here, the probe
  reporting P(positive)=0.161 on text with self-perplexity 122,068. A concept number on non-fluent
  text is not a weak result, it is not a result.
* Deltas are referenced to each model's OWN alpha=0 condition, never to the no-hook baseline.
  Passing activations through the dictionary at all perturbs both metrics; the alpha=0 row measures
  exactly that perturbation and is the only honest zero point. The no-hook baseline is still
  generated, but as a separate reconstruction-cost reference.
* Positive and negative alphas are kept separate throughout. In the previous pilot negative steering
  worked while positive steering failed catastrophically; a signed average would have reported a
  moderate success and hidden both facts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as t

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "dictionary_learning", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train import _signed_binary_direction, _stratified_split_indices  # noqa: E402
from eval_real_intervention import load_kron_checkpoint  # noqa: E402
from eval_patch_generate import generate_with_hook, score_perplexity, score_sentiment  # noqa: E402
from eval_neutral_generation import NEUTRAL_PROMPTS  # noqa: E402
from build_style_dataset import mattr_and_length  # noqa: E402
from eval_style_probe_matrix import build_activation_cache, doc_pool_branches, doc_pool_raw  # noqa: E402


class BranchSteeringHook:
    """Forward hook on a GPT-NeoX block: encode -> push ONE branch -> recombine -> re-topk -> decode.

    The Q path reproduces `eval_patch_generate.SteeringHook`'s kron behaviour exactly. The P path is
    its mirror: `p_pos` is pushed while `q_pos` passes through untouched. Both then go back through
    `ae._combine` and the top-k, so the patched activation is a genuine point in the dictionary's
    output space rather than an arbitrary edit of the residual stream -- the intervention stays on
    the manifold the dictionary can express, which is the whole reason for steering in feature space.

    The push magnitude is `alpha * proj_std`, where `proj_std` is the standard deviation of the FIT
    split's document projections onto the unit direction. So alpha is in units of "document-level
    standard deviations of this concept", and is comparable across the two branches even though P
    (h*m = 1024 dims) and Q (h*n = 2048 dims) have different scales.
    """

    def __init__(self, ae, branch: str, w_hat: t.Tensor, proj_std: float, alpha: float):
        assert branch in ("P", "Q")
        self.ae = ae
        self.branch = branch
        self.w_hat = w_hat
        self.proj_std = proj_std
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        orig_dtype = hidden.dtype
        b, seq, d = hidden.shape
        x = hidden.reshape(b * seq, d).float()

        with t.no_grad():
            _, p_pos, q_pos = self.ae.encode(x, return_branches=True)
            shift = self.alpha * self.proj_std * self.w_hat.unsqueeze(0)
            if self.branch == "P":
                p_new = (p_pos.reshape(p_pos.shape[0], -1) + shift).clamp_min(0.0).reshape(p_pos.shape)
                q_new = q_pos
            else:
                q_new = (q_pos.reshape(q_pos.shape[0], -1) + shift).clamp_min(0.0).reshape(q_pos.shape)
                p_new = p_pos
            dense = self.ae._combine(p_new, q_new, p_new, q_new)
            k = int(self.ae.k.item())
            top = dense.topk(k, sorted=False, dim=-1)
            f_patched = t.zeros_like(dense).scatter_(dim=-1, index=top.indices, src=top.values)
            x_hat = self.ae.decode(f_patched)

        x_hat = x_hat.reshape(b, seq, d).to(orig_dtype)
        return (x_hat,) + output[1:] if is_tuple else x_hat


def fit_direction(docs_fit: t.Tensor, labels_fit: t.Tensor, device: str):
    """Signed binary direction + the projection sd that sets alpha's units.

    `docs_fit` is already document-pooled and therefore small (n_docs x branch_dim), so unlike the
    token-level case there is nothing to stream here -- the streaming happens upstream in
    `doc_pool_branches`, which never materializes [n_tokens, branch_dim].
    """
    w = _signed_binary_direction(docs_fit, labels_fit)
    w_hat = (w / w.norm().clamp_min(1e-12)).to(device)
    proj_std = float((docs_fit.to(device) @ w_hat).std().clamp_min(1e-12))
    return w_hat, proj_std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--run_dir", type=str, default=str(ROOT / "runs" / "style_content_pilot"))
    ap.add_argument("--ckpt_step", type=int, default=420000)
    ap.add_argument("--alphas", type=str, default="-8,-4,0,4,8")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--n_docs", type=int, default=12000)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--mattr_window", type=int, default=15)
    ap.add_argument("--mattr_min_words", type=int, default=15,
                    help="Lower than the corpus label's 25 (continuations are only ~40 tokens), but "
                         "NOT below --mattr_window; see the note in main().")
    ap.add_argument("--cache_path", type=str,
                    default=str(ROOT / "runs" / "style_eval_activation_cache.pt"))
    ap.add_argument("--out_path", type=str, default=str(ROOT / "runs" / "style_steering_results.json"))
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    device = "cuda:0" if t.cuda.is_available() else "cpu"

    # `mattr_and_length` divides the accumulated type count by `n_windows * window`. When a document
    # is shorter than `window` there is exactly one (partial) window, but the denominator is still
    # the full `window` -- so a 10-word continuation with 10 distinct words scores 10/15 = 0.667
    # rather than 1.0, and every short generation is pushed toward "low diversity" for reasons that
    # have nothing to do with its diversity. Since the whole point of this measurement is that it
    # uses the identical function to the training label, the fix is the floor rather than a patched
    # copy of the function: refuse to score anything shorter than one full window, and count it.
    if args.mattr_min_words < args.mattr_window:
        raise SystemExit(
            f"--mattr_min_words ({args.mattr_min_words}) must be >= --mattr_window "
            f"({args.mattr_window}); below the window MATTR is systematically underestimated."
        )

    # --- activations, directions, and the sentiment reference probe, all on the same documents ---
    blob = build_activation_cache(Path(args.cache_path), args.n_docs, args.ctx, args.layer,
                                  args.model_name, device)
    n_docs = blob["n_docs"]
    # Split on sentiment so the reference probe's fit/test split is stratified on the label it reads;
    # the same fit indices then fit both push directions, keeping every fitted object on one split.
    doc_fit, _ = _stratified_split_indices(blob["sentiment"], test_fraction=0.5, seed=42)

    ckpt_dir = Path(args.run_dir) / "checkpoints" / "kron_joint"
    ae = load_kron_checkpoint(ckpt_dir / "trainer_config.json",
                              ckpt_dir / f"ae_step_{args.ckpt_step}.pt", device)
    print(f"kron h={ae.h} m={ae.m} n={ae.n} k={int(ae.k.item())}", flush=True)

    p_docs, q_docs = doc_pool_branches(ae, blob["x"], blob["doc_ids"], n_docs, device)
    p_w, p_std = fit_direction(p_docs[doc_fit], blob["sentiment"][doc_fit], device)   # P <- sentiment
    q_w, q_std = fit_direction(q_docs[doc_fit], blob["style"][doc_fit], device)       # Q <- style
    print(f"P(sentiment) proj_std={p_std:.4f} | Q(style) proj_std={q_std:.4f}", flush=True)
    del p_docs, q_docs
    t.cuda.empty_cache()

    from sklearn.linear_model import LogisticRegression

    raw_docs = doc_pool_raw(blob["x"], blob["doc_ids"], n_docs, device)
    x_fit = raw_docs[doc_fit]
    mu_x = x_fit.mean(dim=0, keepdim=True)
    sd_x = x_fit.std(dim=0, keepdim=True).clamp_min(1e-6)
    clf_sent = LogisticRegression(max_iter=5000, C=1.0).fit(
        ((x_fit - mu_x) / sd_x).numpy(), blob["sentiment"][doc_fit].numpy()
    )
    del raw_docs, x_fit, blob
    t.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    layer_module = model.gpt_neox.layers[args.layer]

    results = {
        "config": vars(args), "prompts": NEUTRAL_PROMPTS, "alphas": alphas,
        "proj_std": {"P_sentiment": p_std, "Q_style": q_std},
        "conditions": [],
    }

    def run_condition(label: str, branch: str | None, alpha: float | None, hook) -> None:
        entries = []
        for prompt in NEUTRAL_PROMPTS:
            for sample_idx in range(args.num_samples):
                seed = 1000 * sample_idx + (abs(hash(prompt)) % 1000)
                text = generate_with_hook(
                    model, tokenizer, layer_module, hook, prompt, args.max_new_tokens, device,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p, seed=seed,
                )
                mattr, n_words = mattr_and_length(text, args.mattr_window, args.mattr_min_words)
                entries.append({
                    "prompt": prompt, "sample_idx": sample_idx, "continuation": text,
                    "probe_sentiment_p_positive": score_sentiment(
                        model, tokenizer, clf_sent, mu_x, sd_x, text, device),
                    "mattr": float("nan") if mattr is None else float(mattr),
                    "n_words": int(n_words),
                    "self_perplexity": score_perplexity(model, tokenizer, prompt, text, device),
                })
        n_short = sum(1 for e in entries if e["mattr"] != e["mattr"])
        results["conditions"].append({
            "label": label, "branch": branch, "alpha": alpha,
            "n_too_short_for_mattr": n_short, "entries": entries,
        })
        finite = lambda key: [e[key] for e in entries if e[key] == e[key]]  # noqa: E731
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
        print(f"{label:<20} n={len(entries):>3} probe_sent={mean(finite('probe_sentiment_p_positive')):.3f} "
              f"mattr={mean(finite('mattr')):.4f} (n_short={n_short}) "
              f"ppl={mean(finite('self_perplexity')):.2f}", flush=True)

    run_condition("baseline_no_hook", None, None, None)
    for branch, w_hat, proj_std in (("P", p_w, p_std), ("Q", q_w, q_std)):
        for alpha in alphas:
            run_condition(f"push{branch}_alpha_{alpha:g}", branch, alpha,
                          BranchSteeringHook(ae, branch, w_hat, proj_std, alpha))

    Path(args.out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
