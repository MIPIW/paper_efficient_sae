"""Neutral-prefix patch-and-generate steering eval.

Difference from `eval_patch_generate.py` (sec.19.11-19.12): that script seeded generation with
truncated *real* Amazon-review openers, which already carry their own sentiment -- so a steered
continuation's sentiment is confounded with the prompt's. Here every prompt is a deliberately
sentiment-NEUTRAL opener ("This movie is", "I saw the movie, and the movie is", ...), so any
sentiment in the continuation has to have come from the steering intervention itself.

Second difference: judging. sec.19.11-19.12 scored the generated text with the project's own
`clf_sent` logistic probe on layer-12 activations, which is the same family of object as the
steering direction -- an internal metric grading itself. Here the generations are written to disk
and judged separately by an external LLM (`judge_generations_qwen.py`, Qwen3-14B), so the
sentiment verdict comes from a model that knows nothing about this project's dictionary.

Self-perplexity under the clean unpatched model is still recorded as the coherence/degeneracy
control, since a "successfully steered" generation that is fluent garbage should not count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch as t

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "dictionary_learning", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_real_intervention import load_kron_checkpoint  # noqa: E402
from eval_flat_intervention import load_flat_checkpoint  # noqa: E402
from train import _signed_binary_direction  # noqa: E402
from eval_patch_generate import (  # noqa: E402
    SteeringHook,
    build_reference_classifiers,
    generate_with_hook,
    score_perplexity,
    score_sentiment,
)


@t.no_grad()
def fit_push_direction_streaming(encode_fn, feat_dim: int, x_full: t.Tensor, doc_ids_full: t.Tensor,
                                 sentiment_full: t.Tensor, doc_fit, n_docs_total: int, device: str,
                                 chunk: int = 20000):
    """Doc-mean-pool the branch features by streaming accumulation instead of materializing every
    token's features first.

    `eval_patch_generate.fit_push_direction` takes an already-computed [n_tokens, feat] tensor,
    which for the flat SAE is 421k x 16384 floats (~27 GB) held in CPU RAM -- enough to get the
    process OOM-killed once anything else is resident. Only the per-document mean is ever needed,
    so accumulate sums/counts per doc (9600 x feat, ~630 MB on GPU) and never keep the token-level
    tensor at all. Mathematically identical to `_mean_pool_by_doc` over the fit split.
    """
    sums = t.zeros(n_docs_total, feat_dim, device=device)
    counts = t.zeros(n_docs_total, device=device)
    is_fit_doc = t.zeros(n_docs_total, dtype=t.bool)
    is_fit_doc[doc_fit] = True
    is_fit_doc = is_fit_doc.to(device)

    for start in range(0, x_full.shape[0], chunk):
        end = min(start + chunk, x_full.shape[0])
        ids = doc_ids_full[start:end].to(device)
        mask = is_fit_doc[ids]
        if not bool(mask.any()):
            continue
        xb = x_full[start:end].to(device)[mask]
        feats = encode_fn(xb).reshape(xb.shape[0], -1)
        ids_m = ids[mask]
        sums.index_add_(0, ids_m, feats.float())
        counts.index_add_(0, ids_m, t.ones(ids_m.shape[0], device=device))
        del xb, feats

    docs_fit = (sums / counts.clamp_min(1.0).unsqueeze(1))[doc_fit].cpu()
    del sums, counts
    w = _signed_binary_direction(docs_fit, sentiment_full[doc_fit])
    w_hat = (w / w.norm().clamp_min(1e-12)).to(device)
    proj_std = float((docs_fit.to(device) @ w_hat).std().clamp_min(1e-12))
    return w_hat, proj_std

# 24 sentiment-neutral openers. Deliberately spread across the review domains this project's
# topic labels cover (film, book, music, electronics, kitchen, apparel) plus a few
# domain-agnostic ones, so the steering result is not a single-domain artifact.
NEUTRAL_PROMPTS = [
    "This movie is",
    "I saw the movie, and the movie is",
    "After watching the film, my impression is that it is",
    "The movie I watched last night was",
    "I finished the book, and it is",
    "This book is",
    "Having read the whole thing, I would say the writing is",
    "I listened to the album, and it is",
    "This album is",
    "The songs on this record are",
    "I bought this product, and it is",
    "This product is",
    "After using it for a week, I can say the product is",
    "The device I ordered turned out to be",
    "This headphone set is",
    "The laptop I purchased is",
    "I tried the kitchen gadget, and it is",
    "This blender is",
    "The knife set arrived, and it is",
    "I wore the jacket, and it is",
    "This shirt is",
    "The shoes I ordered are",
    "Overall, my experience with it was",
    "If I had to describe it in one word, it is",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--alphas", type=str, default="-8,-4,0,4,8")
    ap.add_argument("--kron_run_dir", type=str, default=str(ROOT / "runs" / "joint_dpo_cross_full420k"))
    ap.add_argument("--ckpt_step", type=int, default=420000)
    ap.add_argument("--out_path", type=str, default=str(ROOT / "runs" / "neutral_generation_results.json"))
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    prompts = NEUTRAL_PROMPTS
    print(f"{len(prompts)} neutral prompts, alphas={alphas}, {args.num_samples} samples each", flush=True)

    device = "cuda:0" if t.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()
    layer_module = model.gpt_neox.layers[args.layer]

    kron_dir = Path(args.kron_run_dir) / "checkpoints" / "kron_joint"
    flat_dir = Path(args.kron_run_dir) / "checkpoints" / "flat_joint"
    kron_ae = load_kron_checkpoint(kron_dir / "trainer_config.json", kron_dir / f"ae_step_{args.ckpt_step}.pt", device)
    flat_ae = load_flat_checkpoint(flat_dir / "trainer_config.json", flat_dir / f"ae_step_{args.ckpt_step}.pt", device)
    print(f"kron h={kron_ae.h} m={kron_ae.m} n={kron_ae.n} | flat dict_size={flat_ae.dict_size}", flush=True)

    clf_sent, mu_x, sd_x, x_full, doc_ids_full, topic_full, sentiment_full, n_docs_total, doc_fit = \
        build_reference_classifiers(device)

    kron_shape = (kron_ae.h, kron_ae.n)
    kron_w_hat, kron_proj_std = fit_push_direction_streaming(
        lambda xb: kron_ae.encode(xb, return_branches=True)[2],
        kron_ae.h * kron_ae.n, x_full, doc_ids_full, sentiment_full, doc_fit, n_docs_total, device,
    )
    flat_w_hat, flat_proj_std = fit_push_direction_streaming(
        lambda xb: flat_ae.encode(xb, return_topk=True)[3],
        flat_ae.dict_size, x_full, doc_ids_full, sentiment_full, doc_fit, n_docs_total, device,
    )
    print(f"kron proj_std={kron_proj_std:.4f} flat proj_std={flat_proj_std:.4f}", flush=True)

    results = {
        "prompts": prompts, "alphas": alphas, "max_new_tokens": args.max_new_tokens,
        "num_samples": args.num_samples, "temperature": args.temperature, "top_p": args.top_p,
        "conditions": [],
    }

    def run_condition(label: str, hook) -> None:
        print(f"\n=== {label} ===", flush=True)
        entries = []
        for prompt in prompts:
            for sample_idx in range(args.num_samples):
                seed = 1000 * sample_idx + (abs(hash(prompt)) % 1000)
                text = generate_with_hook(
                    model, tokenizer, layer_module, hook, prompt, args.max_new_tokens, device,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p, seed=seed,
                )
                entries.append({
                    "prompt": prompt, "sample_idx": sample_idx, "continuation": text,
                    "probe_sentiment_p_positive": score_sentiment(model, tokenizer, clf_sent, mu_x, sd_x, text, device),
                    "self_perplexity": score_perplexity(model, tokenizer, prompt, text, device),
                })
        results["conditions"].append({"label": label, "entries": entries})
        finite = [e["probe_sentiment_p_positive"] for e in entries if e["probe_sentiment_p_positive"] == e["probe_sentiment_p_positive"]]
        print(f"  n={len(entries)} mean_probe_sent={sum(finite)/max(len(finite),1):.3f}", flush=True)

    run_condition("baseline_no_hook", None)
    for arch, ae, w_hat, proj_std, shape in (
        ("kron", kron_ae, kron_w_hat, kron_proj_std, kron_shape),
        ("flat", flat_ae, flat_w_hat, flat_proj_std, None),
    ):
        for alpha in alphas:
            run_condition(f"{arch}_alpha_{alpha:g}",
                          SteeringHook(arch, ae, w_hat, proj_std, alpha, branch_shape=shape))

    Path(args.out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
