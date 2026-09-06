"""Patch-and-GENERATE causal steering eval (REPORT_FULL.md sec.19.11) -- the experiment
sec.19.3-19.10 never actually did despite the "patch-and-generate" name used throughout: every
prior steering result stopped at scoring the SAE's *reconstruction* with an external classifier.
It never fed the steered reconstruction back into the model and generated actual text.

Motivating question (raised directly by the user): does rising reconstruction FVU under a
steering push mean the edit broke the representation (implausible, off-manifold, would generate
garbage) -- or does it just mean a real, larger semantic shift necessarily reconstructs further
from the original x? FVU alone can't distinguish these. Actually patching the steered
reconstruction into the live model and generating text can: if flat SAE's high-FVU pushes
generate incoherent/degenerate text while Kron's low-FVU pushes stay fluent, that validates FVU
as a coherence proxy. If flat SAE's high-FVU text is still fluent and correctly sentiment-shifted,
FVU was misleading as an interpretation of "broken" vs. "genuinely more strongly steered".

Method: a plain forward hook on pythia-410m's layer-12 GPT-NeoX block (matching this project's
io='out' convention) intercepts the block's hidden-state output at *every* forward call --
prompt processing and every subsequent cached single-token generation step alike -- runs it
through the trained SAE (Kron control's Q-branch, or the flat SAE), applies the alpha push along
the already-established signed sentiment direction, re-applies top-k, decodes, and substitutes
the result back in place of the true activation. The rest of the model computes on top of that
patched value, generation proceeds token-by-token as normal.

Judging, entirely self-contained (no new external model, consistent with this project's existing
methodology): (a) sentiment -- re-encode the GENERATED CONTINUATION text through the clean
(unpatched) model to get its own genuine layer-12 activation, score with the same
`clf_sent`/`clf_top` classifiers fit on real activations used throughout sec.19; (b) coherence --
self-perplexity: average per-token cross-entropy of the continuation under the clean model
conditioned on the original prompt, a standard degenerate-text detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch as t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dictionary_learning.dictionary_kron import KronAutoEncoderTopK  # noqa: E402
from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402
from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc  # noqa: E402
from train import _stratified_split_indices, _signed_binary_direction  # noqa: E402
from eval_real_intervention import load_kron_checkpoint, load_cache_batches  # noqa: E402
from eval_flat_intervention import load_flat_checkpoint  # noqa: E402

def build_prompts(n_per_topic_sentiment: int = 2, n_words: int = 8, seed: int = 42) -> list[str]:
    """Balanced real-review-opener prompts: `n_per_topic_sentiment` examples per (topic,
    sentiment) cell, truncated to their first `n_words` words -- so the model has to generate
    genuinely new content, not echo a memorized continuation, while the prompt set itself covers
    every topic x sentiment combination this project's labels distinguish."""
    import random

    rng = random.Random(seed)
    rows = []
    with open(ROOT / "data" / "amazon_reviews_full500k" / "eval.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    topics = sorted(set(r["topic"] for r in rows))
    selected = []
    for topic in topics:
        for sent in (0, 1):
            cands = [r for r in rows if r["topic"] == topic and r["sentiment_label"] == sent and len(r["text"].split()) > 15]
            rng.shuffle(cands)
            selected.extend(cands[:n_per_topic_sentiment])
    prompts = []
    for r in selected:
        text = r["text"].replace("\n\n", " ").strip()
        prompts.append(" ".join(text.split()[:n_words]))
    return prompts


def build_reference_classifiers(device: str):
    from sklearn.linear_model import LogisticRegression

    cache_path = ROOT / "runs" / "pythia_erasure_activation_cache.pt"
    batches, n_docs = load_cache_batches(cache_path, max_batches=400)
    x_full = t.cat([b[0] for b in batches], dim=0)
    doc_ids_full = t.cat([b[1] for b in batches], dim=0)
    topic_full = t.cat([b[2] for b in batches], dim=0)
    sentiment_full = t.cat([b[3] for b in batches], dim=0)
    n_docs_total = topic_full.shape[0]

    seed = 42
    doc_fit, _ = _stratified_split_indices(sentiment_full, test_fraction=0.5, seed=seed)
    is_fit_doc = t.zeros(n_docs_total, dtype=t.bool)
    is_fit_doc[doc_fit] = True
    token_is_fit = is_fit_doc[doc_ids_full]

    with t.no_grad():
        x_docs_fit = _mean_pool_by_doc(
            x_full[token_is_fit].to(device), doc_ids_full[token_is_fit].to(device), n_docs_total
        )[doc_fit].cpu()
    mu_x, sd_x = x_docs_fit.mean(dim=0, keepdim=True), x_docs_fit.std(dim=0, keepdim=True).clamp_min(1e-6)
    clf_sent = LogisticRegression(max_iter=5000, C=1.0).fit(
        ((x_docs_fit - mu_x) / sd_x).numpy(), sentiment_full[doc_fit].numpy()
    )
    return clf_sent, mu_x, sd_x, x_full, doc_ids_full, topic_full, sentiment_full, n_docs_total, doc_fit


def fit_push_direction(target_pos_full: t.Tensor, doc_ids_full: t.Tensor, sentiment_full: t.Tensor,
                        doc_fit, n_docs_total: int, device: str):
    """Reuses eval_real_intervention.py's exact direction-fitting method."""
    target_shape = target_pos_full.shape
    target_flat_full = target_pos_full.reshape(target_shape[0], -1)
    is_fit_doc = t.zeros(n_docs_total, dtype=t.bool)
    is_fit_doc[doc_fit] = True
    token_is_fit = is_fit_doc[doc_ids_full]
    with t.no_grad():
        target_docs_fit = _mean_pool_by_doc(
            target_flat_full[token_is_fit].to(device), doc_ids_full[token_is_fit].to(device), n_docs_total
        )[doc_fit].cpu()
    w = _signed_binary_direction(target_docs_fit, sentiment_full[doc_fit])
    w_hat = (w / w.norm().clamp_min(1e-12)).to(device)
    proj = target_docs_fit.to(device) @ w_hat
    proj_std = float(proj.std().clamp_min(1e-12))
    return w_hat, proj_std, (target_shape[1], target_shape[2]) if len(target_shape) == 3 else None


class SteeringHook:
    """Forward hook for a GPT-NeoX decoder block: encode -> push -> re-topk -> decode, replacing
    the block's hidden-state output in place. `mode="off"` makes this a no-op passthrough (used
    for the true unpatched baseline without needing to remove/reattach the hook)."""

    def __init__(self, arch: str, ae, w_hat: t.Tensor, proj_std: float, alpha: float, branch_shape=None):
        self.arch = arch  # "kron" or "flat"
        self.ae = ae
        self.w_hat = w_hat
        self.proj_std = proj_std
        self.alpha = alpha
        self.branch_shape = branch_shape  # (h, n) for kron's Q branch
        self.enabled = True

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        orig_dtype = hidden.dtype
        b, seq, d = hidden.shape
        x = hidden.reshape(b * seq, d).float()

        with t.no_grad():
            if self.arch == "kron":
                _, p_pos, q_pos = self.ae.encode(x, return_branches=True)
                h, n = self.branch_shape
                q_flat = q_pos.reshape(q_pos.shape[0], -1)
                q_pushed = (q_flat + self.alpha * self.proj_std * self.w_hat.unsqueeze(0)).clamp_min(0.0)
                q_pushed = q_pushed.reshape(q_pos.shape[0], h, n)
                dense = self.ae._combine(p_pos, q_pushed, p_pos, q_pushed)
                k = int(self.ae.k.item())
                post_topk = dense.topk(k, sorted=False, dim=-1)
                f_patched = t.zeros_like(dense).scatter_(dim=-1, index=post_topk.indices, src=post_topk.values)
                x_hat = self.ae.decode(f_patched)
            else:  # flat
                _, _, _, post_relu = self.ae.encode(x, return_topk=True)
                pushed = (post_relu + self.alpha * self.proj_std * self.w_hat.unsqueeze(0)).clamp_min(0.0)
                k = int(self.ae.k.item())
                post_topk = pushed.topk(k, sorted=False, dim=-1)
                f_patched = t.zeros_like(pushed).scatter_(dim=-1, index=post_topk.indices, src=post_topk.values)
                x_hat = self.ae.decode(f_patched)

        x_hat = x_hat.reshape(b, seq, d).to(orig_dtype)
        if is_tuple:
            return (x_hat,) + output[1:]
        return x_hat


@t.no_grad()
def generate_with_hook(model, tokenizer, layer_module, hook: "SteeringHook | None", prompt: str,
                        max_new_tokens: int, device: str, do_sample: bool = False,
                        temperature: float = 1.0, top_p: float = 1.0, seed: Optional[int] = None) -> str:
    handle = None
    if hook is not None:
        handle = layer_module.register_forward_hook(hook)
    try:
        if seed is not None:
            t.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)
        out = model.generate(**inputs, **gen_kwargs)
        continuation_ids = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(continuation_ids, skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()


@t.no_grad()
def score_sentiment(model, tokenizer, clf_sent, mu_x, sd_x, text: str, device: str) -> float:
    if not text.strip():
        return float("nan")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    # hidden_states[0] = embeddings, hidden_states[i] = output of block i-1 (0-indexed blocks);
    # block 12's output is hidden_states[13]. Verified against this project's layer=12/io='out'
    # convention (the block list index matches args.layer directly).
    act = hidden_states[13][0].mean(dim=0, keepdim=True).float().cpu()
    x_std = ((act - mu_x) / sd_x).numpy()
    return float(clf_sent.predict_proba(x_std)[:, 1].item())


@t.no_grad()
def score_perplexity(model, tokenizer, prompt: str, continuation: str, device: str) -> float:
    if not continuation.strip():
        return float("nan")
    full = tokenizer(prompt + continuation, return_tensors="pt").to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    input_ids = full["input_ids"]
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    # Only score the continuation span (predicting continuation tokens from context).
    cont_start = max(prompt_len - 1, 0)
    logp = t.nn.functional.log_softmax(logits[:, cont_start:, :].float(), dim=-1)
    tok_targets = targets[:, cont_start:]
    if tok_targets.shape[1] == 0:
        return float("nan")
    nll = -logp.gather(-1, tok_targets.unsqueeze(-1)).squeeze(-1).mean()
    return float(t.exp(nll).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--alphas", type=str, default="-6,0,6")
    ap.add_argument(
        "--kron_run_dir", type=str, default=str(ROOT / "runs" / "joint_dpo_cross_full420k"),
    )
    ap.add_argument("--ckpt_step", type=int, default=420000)
    ap.add_argument("--out_path", type=str, default=str(ROOT / "runs" / "patch_generate_results.json"))
    ap.add_argument("--n_per_topic_sentiment", type=int, default=2, help="Prompts per (topic, sentiment) cell.")
    ap.add_argument("--prompt_words", type=int, default=8)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--num_samples", type=int, default=1, help="Repeated generations per (prompt, condition), different seeds.")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    prompts = build_prompts(args.n_per_topic_sentiment, args.prompt_words, seed=42)
    print(f"Built {len(prompts)} prompts from real review openers.", flush=True)

    device = "cuda:0" if t.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading pythia-410m (plain transformers, for hook-based generation)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()
    layer_module = model.gpt_neox.layers[args.layer]

    kron_dir = Path(args.kron_run_dir) / "checkpoints" / "kron_joint"
    flat_dir = Path(args.kron_run_dir) / "checkpoints" / "flat_joint"
    kron_ae = load_kron_checkpoint(kron_dir / "trainer_config.json", kron_dir / f"ae_step_{args.ckpt_step}.pt", device)
    flat_ae = load_flat_checkpoint(flat_dir / "trainer_config.json", flat_dir / f"ae_step_{args.ckpt_step}.pt", device)
    print(f"Loaded kron (h={kron_ae.h},m={kron_ae.m},n={kron_ae.n}) and flat (dict_size={flat_ae.dict_size}) checkpoints.", flush=True)

    print("Fitting reference classifiers + push directions on cached real activations...", flush=True)
    clf_sent, mu_x, sd_x, x_full, doc_ids_full, topic_full, sentiment_full, n_docs_total, doc_fit = \
        build_reference_classifiers(device)

    with t.no_grad():
        CHUNK = 20000
        p_chunks, q_chunks, flat_chunks = [], [], []
        for start in range(0, x_full.shape[0], CHUNK):
            end = min(start + CHUNK, x_full.shape[0])
            _, p_c, q_c = kron_ae.encode(x_full[start:end].to(device), return_branches=True)
            p_chunks.append(p_c.cpu())
            q_chunks.append(q_c.cpu())
            _, _, _, flat_c = flat_ae.encode(x_full[start:end].to(device), return_topk=True)
            flat_chunks.append(flat_c.cpu())
        q_pos_full = t.cat(q_chunks, dim=0)
        flat_pos_full = t.cat(flat_chunks, dim=0)

    kron_w_hat, kron_proj_std, kron_shape = fit_push_direction(
        q_pos_full, doc_ids_full, sentiment_full, doc_fit, n_docs_total, device
    )
    flat_w_hat, flat_proj_std, _ = fit_push_direction(
        flat_pos_full, doc_ids_full, sentiment_full, doc_fit, n_docs_total, device
    )
    print(f"kron Q direction: proj_std={kron_proj_std:.4f} | flat direction: proj_std={flat_proj_std:.4f}", flush=True)

    results = {
        "prompts": prompts, "max_new_tokens": args.max_new_tokens, "do_sample": args.do_sample,
        "temperature": args.temperature, "top_p": args.top_p, "num_samples": args.num_samples,
        "conditions": [],
    }

    def run_condition(label: str, hook):
        print(f"\n=== condition: {label} ===", flush=True)
        entries = []
        for prompt in prompts:
            for sample_idx in range(args.num_samples):
                seed = 1000 * sample_idx + (hash(prompt) % 1000)
                text = generate_with_hook(
                    model, tokenizer, layer_module, hook, prompt, args.max_new_tokens, device,
                    do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
                    seed=seed if args.do_sample else None,
                )
                sent = score_sentiment(model, tokenizer, clf_sent, mu_x, sd_x, text, device)
                ppl = score_perplexity(model, tokenizer, prompt, text, device)
                entries.append({
                    "prompt": prompt, "sample_idx": sample_idx, "continuation": text,
                    "sentiment_p_positive": sent, "self_perplexity": ppl,
                })
                print(f"  [{prompt!r} #{sample_idx}] -> {text!r}\n    sent={sent:.3f} ppl={ppl:.2f}", flush=True)
        results["conditions"].append({"label": label, "entries": entries})

    # True unpatched baseline.
    run_condition("baseline_no_hook", None)

    for arch, ae, w_hat, proj_std, shape in (
        ("kron", kron_ae, kron_w_hat, kron_proj_std, kron_shape),
        ("flat", flat_ae, flat_w_hat, flat_proj_std, None),
    ):
        for alpha in alphas:
            hook = SteeringHook(arch, ae, w_hat, proj_std, alpha, branch_shape=shape)
            run_condition(f"{arch}_alpha_{alpha:g}", hook)

    out_path = Path(args.out_path)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
