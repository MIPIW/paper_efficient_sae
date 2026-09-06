"""Concept erasure existence-proof control, on REAL pythia-410m activations.

Extends `eval_concept_erasure_control.py`'s synthetic-data result (iterative
INLP achieves near-perfect, near-chance, symmetric collect+suppress erasure
of sentiment/topic with ~zero collateral damage) to real pythia-410m layer-12
residual-stream activations on this project's Amazon Reviews sentiment x topic
corpus -- the same activations every SAE experiment in this project (§6-§9,
REPORT1.md) was trained and evaluated on. If iterative INLP succeeds here too,
that rules out "the synthetic buffer's orthogonal-by-construction directions
made this too easy" as an objection: real LM activations have no such
guarantee (§9.4 already found sentiment/topic probes are NOT perfectly
orthogonal or perfectly linearly separable on real activations).

Reuses `cache_token_activations` (from `eval_pilot_q_liveness.py`) for the
single-LM-pass extraction, exactly matching every other real-data evaluation
in this project (same dataset, same layer, same buffer), then applies
`iterative_nullspace_projection`/`project_out_subspace`/`random_subspace` from
`eval_concept_erasure_control.py` unchanged.
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

from data.amazon_reviews import load_prepared_datasets  # noqa: E402
from eval_pilot_q_liveness import cache_token_activations, set_seed  # noqa: E402
from eval_concept_erasure_control import (  # noqa: E402
    iterative_nullspace_projection,
    project_out_subspace,
    random_subspace,
    probe_split,
    leace_fit,
    leace_erase,
    iterative_leace,
)
from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc  # noqa: E402


def two_probes(x: t.Tensor, sent: t.Tensor, top: t.Tensor, n_train: int) -> tuple[float, float]:
    s_acc, s_conv = probe_split(x, sent, n_train)
    t_acc, t_conv = probe_split(x, top, n_train)
    if not (s_conv and t_conv):
        print("WARNING: a probe did not converge", file=sys.stderr)
    return s_acc, t_acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--ctx_len", type=int, default=128)
    ap.add_argument("--activation_dim", type=int, default=1024)
    ap.add_argument("--dataset_cache_dir", type=str,
                     default=str(ROOT.parent / "data" / "amazon_reviews_full500k"))
    ap.add_argument("--doc_batch_size", type=int, default=24)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_probe_docs", type=int, default=20000)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--method", type=str, default="inlp", choices=["inlp", "leace", "iterative_leace"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--activation_cache", type=str, default=str(ROOT.parent / "runs" / "pythia_erasure_activation_cache.pt"))
    ap.add_argument("--output_json", type=str, default=str(ROOT.parent / "runs" / "concept_erasure_pythia_results.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda:0" if t.cuda.is_available() else "cpu")

    cache_path = Path(args.activation_cache)
    if cache_path.exists():
        print(f"Loading cached token activations from {cache_path} (no LM pass)", flush=True)
        batches = t.load(cache_path, map_location="cpu")
    else:
        from nnsight import LanguageModel

        _, eval_ds, _, _ = load_prepared_datasets(args.dataset_cache_dir)
        model = LanguageModel(args.model_name, dispatch=True, device_map=device)
        layer_list_paths = ("gpt_neox.layers", "transformer.h", "model.layers")
        submodule = None
        for path in layer_list_paths:
            obj = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                submodule = obj[args.layer]
                break
            except AttributeError:
                continue
        if submodule is None:
            raise AttributeError(f"Could not find a transformer block list on {args.model_name}")
        print("Caching token activations (single LM pass)...", flush=True)
        batches = cache_token_activations(
            model=model, submodule=submodule, eval_ds=eval_ds,
            activation_dim=args.activation_dim, ctx_len=args.ctx_len,
            doc_batch_size=args.doc_batch_size, num_workers=args.num_workers,
            max_docs=args.max_probe_docs, device=device,
        )
        del model
        t.cuda.empty_cache()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        t.save(batches, cache_path)

    print("Doc-mean-pooling cached token activations...", flush=True)
    x_all_list, sent_all, top_all = [], [], []
    for b in batches:
        x = b["x"].to(device=device, dtype=t.float32)
        ids = b["token_doc_ids"].to(device)
        pooled = _mean_pool_by_doc(x, ids, b["n_docs"]).cpu().float()
        x_all_list.append(pooled)
        sent_all.append(b["sentiment"])
        top_all.append(b["topic"])
    x_all = t.cat(x_all_list)[: args.max_probe_docs]
    sent_all = t.cat(sent_all)[: args.max_probe_docs]
    top_all = t.cat(top_all)[: args.max_probe_docs]
    n_sentiments = int(t.unique(sent_all).numel())
    n_topics = int(t.unique(top_all).numel())
    print(f"n_docs={x_all.shape[0]} n_sentiments={n_sentiments} n_topics={n_topics}", flush=True)

    n_fit = int(0.5 * x_all.shape[0])
    n_probe = x_all.shape[0] - n_fit
    n_probe_train = int(0.65 * n_probe)
    x_fit, sent_fit, top_fit = x_all[:n_fit], sent_all[:n_fit], top_all[:n_fit]
    x_probe, sent_probe, top_probe = x_all[n_fit:], sent_all[n_fit:], top_all[n_fit:]

    def probes(x: t.Tensor) -> tuple[float, float]:
        return two_probes(x, sent_probe, top_probe, n_probe_train)

    sent_acc_raw, top_acc_raw = probes(x_probe)

    results: dict = {
        "chance": {"sentiment": 1.0 / n_sentiments, "topic": 1.0 / n_topics},
        "n_docs": int(x_all.shape[0]),
        "raw": {"sentiment_acc": sent_acc_raw, "topic_acc": top_acc_raw},
    }
    generator = t.Generator().manual_seed(args.seed + 777)

    for target_name, y_fit in (("sentiment", sent_fit), ("topic", top_fit)):
        print(f"--- {args.method.upper()} erasing '{target_name}' (max_rounds={args.max_rounds}) ---", flush=True)
        if args.method == "leace":
            fit = leace_fit(x_fit, y_fit)
            x_erased = leace_erase(x_probe, fit)
            rank = fit["rank"]
            rounds_run = 1
        elif args.method == "iterative_leace":
            x_erased, rank, rounds_run = iterative_leace(
                x_fit, y_fit, x_probe, max_rounds=args.max_rounds, verbose=True,
            )
        else:
            x_erased, basis, rounds_run = iterative_nullspace_projection(
                x_fit, y_fit, x_probe, max_rounds=args.max_rounds, verbose=True,
            )
            rank = basis.shape[0]
        sent_acc_e, top_acc_e = probes(x_erased)

        rand_basis = random_subspace(args.activation_dim, rank, generator)
        x_rand_erased = project_out_subspace(x_probe, rand_basis)
        sent_acc_r, top_acc_r = probes(x_rand_erased)

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
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    print("\n=== Summary table (real pythia-410m activations, iterative-INLP erasure vs. raw) ===")
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
