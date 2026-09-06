"""Probe access matrix for the content/style pilot: which concept is readable from which branch?

The pilot trains one KronSAE whose two branches are supervised on two *uncorrelated* concepts --
branch P on sentiment (content) and branch Q on lexical diversity / MATTR (style). The question the
project has never been able to answer affirmatively is not whether a supervised branch *collects*
its own concept (it always has), but whether it *excludes* the other one. So the object of interest
here is a 2x2 matrix, and specifically its OFF-DIAGONAL:

              probe: sentiment   probe: style
    branch P     [diagonal]      [off-diagonal]   <- P should NOT know style
    branch Q   [off-diagonal]     [diagonal]      <- Q should NOT know sentiment

A high diagonal alone means nothing: every previous run in this project got that for free, because
the residual stream carries both concepts and a branch that simply copies the residual stream reads
high on everything. Factorization is the off-diagonal DROPPING relative to what the raw residual
stream already affords. That is why every accuracy below is reported next to two references:

  * the RAW layer-12 ceiling, refit here on the very same documents and the very same split, so the
    comparison is not contaminated by a different doc sample (runs/style_ceiling_probe.json measured
    style 0.7458 / sentiment 0.9255 on an independent draw; the numbers this script prints under
    `raw` should land near those, and if they do not, that itself is worth knowing);
  * the MAJORITY-CLASS baseline. This is not decoration. Sentiment in this corpus is ~84% positive,
    so a branch that has learned nothing at all still scores 0.84 on the sentiment probe. Reading a
    raw 0.86 as "Q knows sentiment" would be a straightforward error, and reporting accuracy without
    the baseline beside it invites exactly that error.

A branch scoring ABOVE raw is concentrating the concept (the dictionary has made it more linearly
accessible than the residual stream did); BELOW raw is losing it. Only the second is evidence of
factorization, and only when it happens off-diagonal while the diagonal holds.

The control (`runs/joint_dpo_cross_full420k`) is the same recipe with Q supervised on 6-way TOPIC
instead of style. It is probed on the same documents for all three concepts -- sentiment, style, and
the original topic label (preserved in the style corpus as `orig_topic_label`) -- so that "P leaks
style in the style run" can be compared against "P leaks topic in the control run" rather than
against nothing.

Participation ratio per branch is reported alongside. A branch whose effective dimensionality has
collapsed can score high on a probe simply by being a low-rank near-copy of a single strong
direction; PR is the cheap check on whether the accuracy reflects structure or degeneracy.

Activations: Pythia-410m layer 12, io='out' (i.e. `hidden_states[13]`, the block's output), matching
this project's convention throughout. There is no pre-existing activation cache carrying style
labels -- `runs/pythia_erasure_activation_cache.pt` predates the style corpus -- so this script runs
the forward pass itself and caches the result to disk for reuse across checkpoints.
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

from train import (  # noqa: E402
    _participation_ratio_from_cov,
    _stratified_split_indices,
    standardized_logreg_accuracy,
)
from eval_real_intervention import load_kron_checkpoint  # noqa: E402

# Reference ceilings from runs/style_ceiling_probe.json, measured on an independent draw of
# documents. Carried here so every printed table is self-describing.
RAW_REFERENCE = {"style": 0.7458, "sentiment": 0.9255, "mattr_regression_r2": 0.4210}


@t.no_grad()
def build_activation_cache(cache_path: Path, n_docs: int, ctx: int, layer: int,
                           model_name: str, device: str, batch_size: int = 32):
    """Layer-`layer` token activations over style-labelled eval documents, cached to disk.

    Stored fp16 to keep ~1M x 1024 floats at ~2GB rather than ~4GB; the probes standardize their
    inputs anyway, so fp16 storage costs nothing that a logistic regression can detect. Padding
    tokens are dropped at build time rather than masked later, so `doc_ids` indexes real tokens only
    and mean-pooling needs no mask.
    """
    if cache_path.exists():
        blob = t.load(cache_path, map_location="cpu", weights_only=False)
        print(f"loaded activation cache {cache_path} "
              f"({blob['x'].shape[0]} tokens, {blob['n_docs']} docs)", flush=True)
        return blob

    rows = []
    with open(ROOT / "data" / "amazon_reviews_style500k" / "eval.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n_docs:
                break
    print(f"forward pass over {len(rows)} docs (ctx={ctx}, layer={layer})", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    xs, ids = [], []
    for i in range(0, len(rows), batch_size):
        batch = [r["text"] for r in rows[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=ctx).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1]
        mask = enc["attention_mask"].bool()
        b, seq = mask.shape
        doc_index = (t.arange(b, device=device).unsqueeze(1).expand(b, seq) + i)
        xs.append(hs[mask].half().cpu())
        ids.append(doc_index[mask].cpu())
        if i % (batch_size * 50) == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    del model
    t.cuda.empty_cache()

    blob = {
        "x": t.cat(xs),
        "doc_ids": t.cat(ids).long(),
        "n_docs": len(rows),
        "style": t.tensor([r["topic_label"] for r in rows], dtype=t.long),
        "sentiment": t.tensor([r["sentiment_label"] for r in rows], dtype=t.long),
        "orig_topic": t.tensor([r["orig_topic_label"] for r in rows], dtype=t.long),
        "mattr": t.tensor([r["mattr"] for r in rows], dtype=t.float),
        "layer": layer, "ctx": ctx, "model_name": model_name,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t.save(blob, cache_path)
    print(f"saved activation cache -> {cache_path} ({blob['x'].shape[0]} tokens)", flush=True)
    return blob


@t.no_grad()
def doc_pool_branches(ae, x: t.Tensor, doc_ids: t.Tensor, n_docs: int, device: str,
                      chunk: int = 20000):
    """Doc-mean-pooled P and Q branch activations, by streaming accumulation.

    Never materializes a [n_tokens, branch_dim] tensor: at ~1.1M tokens, Q alone (128*16 = 2048
    dims) would be ~9GB in float32 and P another ~4.5GB, which is how this project has previously
    got processes OOM-killed. Only per-document sums are ever held (n_docs x dim), and the token
    chunk is freed each iteration. Identical in value to `_mean_pool_by_doc` over all tokens.
    """
    p_dim, q_dim = ae.h * ae.m, ae.h * ae.n
    p_sum = t.zeros(n_docs, p_dim, device=device)
    q_sum = t.zeros(n_docs, q_dim, device=device)
    counts = t.zeros(n_docs, device=device)

    for start in range(0, x.shape[0], chunk):
        end = min(start + chunk, x.shape[0])
        xb = x[start:end].to(device).float()
        idb = doc_ids[start:end].to(device)
        _, p_pos, q_pos = ae.encode(xb, return_branches=True)
        p_sum.index_add_(0, idb, p_pos.reshape(xb.shape[0], -1).float())
        q_sum.index_add_(0, idb, q_pos.reshape(xb.shape[0], -1).float())
        counts.index_add_(0, idb, t.ones(idb.shape[0], device=device))
        del xb, p_pos, q_pos

    denom = counts.clamp_min(1.0).unsqueeze(1)
    return (p_sum / denom).cpu(), (q_sum / denom).cpu()


@t.no_grad()
def doc_pool_raw(x: t.Tensor, doc_ids: t.Tensor, n_docs: int, device: str, chunk: int = 50000):
    """Doc-mean-pooled raw residual-stream activations -- the ceiling reference, same docs, same split."""
    acc = t.zeros(n_docs, x.shape[1], device=device)
    counts = t.zeros(n_docs, device=device)
    for start in range(0, x.shape[0], chunk):
        end = min(start + chunk, x.shape[0])
        idb = doc_ids[start:end].to(device)
        acc.index_add_(0, idb, x[start:end].to(device).float())
        counts.index_add_(0, idb, t.ones(idb.shape[0], device=device))
    return (acc / counts.clamp_min(1.0).unsqueeze(1)).cpu()


def probe_cell(feats: t.Tensor, labels: t.Tensor, seed: int = 42) -> dict:
    """One cell of the access matrix: probe accuracy + the majority baseline it must be read against."""
    fit_idx, test_idx = _stratified_split_indices(labels, test_fraction=0.5, seed=seed)
    acc, converged = standardized_logreg_accuracy(
        feats[fit_idx], labels[fit_idx], feats[test_idx], labels[test_idx]
    )
    y_te = labels[test_idx].numpy()
    _, cnt = np.unique(y_te, return_counts=True)
    majority = float(cnt.max() / cnt.sum())
    return {
        "acc": float(acc),
        "majority_baseline": majority,
        "above_majority": float(acc) - majority,
        "converged": bool(converged),
        "n_test": int(len(test_idx)),
    }


def evaluate_model(name: str, run_dir: Path, steps: list[int], blob, concepts: dict,
                   device: str, out: dict) -> None:
    cfg_path = run_dir / "checkpoints" / "kron_joint" / "trainer_config.json"
    for step in steps:
        ckpt = run_dir / "checkpoints" / "kron_joint" / f"ae_step_{step}.pt"
        if not ckpt.exists():
            print(f"[{name}] step {step}: checkpoint missing, skipping", flush=True)
            continue
        ae = load_kron_checkpoint(cfg_path, ckpt, device)
        p_docs, q_docs = doc_pool_branches(ae, blob["x"], blob["doc_ids"], blob["n_docs"], device)
        del ae
        t.cuda.empty_cache()

        entry = {"branches": {}}
        for branch_name, feats in (("P", p_docs), ("Q", q_docs)):
            cells = {c: probe_cell(feats, y) for c, y in concepts.items()}
            cells["participation_ratio"] = _participation_ratio_from_cov(feats.to(device))
            cells["dim"] = int(feats.shape[1])
            entry["branches"][branch_name] = cells
            desc = "  ".join(
                f"{c}={cells[c]['acc']:.4f}(maj {cells[c]['majority_baseline']:.4f})"
                for c in concepts
            )
            print(f"[{name}] step {step:>7} branch {branch_name}: {desc}  "
                  f"PR={cells['participation_ratio']:.2f}", flush=True)
        out.setdefault(name, {})[str(step)] = entry
        del p_docs, q_docs
        t.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=12000)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    ap.add_argument("--style_run_dir", type=str, default=str(ROOT / "runs" / "style_content_pilot"))
    ap.add_argument("--style_steps", type=str, default="105000,210000,315000,420000")
    ap.add_argument("--control_run_dir", type=str, default=str(ROOT / "runs" / "joint_dpo_cross_full420k"))
    ap.add_argument("--control_steps", type=str, default="420000")
    ap.add_argument("--cache_path", type=str,
                    default=str(ROOT / "runs" / "style_eval_activation_cache.pt"))
    ap.add_argument("--out_path", type=str, default=str(ROOT / "runs" / "style_probe_matrix.json"))
    args = ap.parse_args()

    device = "cuda:0" if t.cuda.is_available() else "cpu"
    blob = build_activation_cache(Path(args.cache_path), args.n_docs, args.ctx, args.layer,
                                  args.model_name, device)

    concepts = {
        "sentiment": blob["sentiment"],
        "style": blob["style"],
        "orig_topic": blob["orig_topic"],
    }

    out = {
        "config": vars(args),
        "raw_reference_independent_draw": RAW_REFERENCE,
        "n_docs": int(blob["n_docs"]),
        "n_tokens": int(blob["x"].shape[0]),
    }

    # Raw ceiling on THESE documents and THIS split -- the only fair comparator for the cells below.
    raw_docs = doc_pool_raw(blob["x"], blob["doc_ids"], blob["n_docs"], device)
    out["raw"] = {c: probe_cell(raw_docs, y) for c, y in concepts.items()}
    out["raw"]["participation_ratio"] = _participation_ratio_from_cov(raw_docs.to(device))
    print("[raw] " + "  ".join(
        f"{c}={out['raw'][c]['acc']:.4f}(maj {out['raw'][c]['majority_baseline']:.4f})"
        for c in concepts) + f"  PR={out['raw']['participation_ratio']:.2f}", flush=True)
    del raw_docs
    t.cuda.empty_cache()

    evaluate_model("style_pilot", Path(args.style_run_dir),
                   [int(s) for s in args.style_steps.split(",")], blob, concepts, device, out)
    evaluate_model("control_topic", Path(args.control_run_dir),
                   [int(s) for s in args.control_steps.split(",")], blob, concepts, device, out)

    Path(args.out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
