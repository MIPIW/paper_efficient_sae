"""Real-pythia-410m version of the patch-and-generate causal steering eval
(REPORT_FULL.md sec.19.4, follow-up to sec.19.3's synthetic-only result).

Reuses the already-trained real checkpoint (`runs/joint_dpo_cross_full420k`, 420,000 steps,
joint_dpo_cross, h=128/m=8/n=16) and the already-cached real pythia-410m layer-12 activations
(`runs/pythia_erasure_activation_cache.pt`, from the LEACE experiments, sec.13-15) -- no
retraining, no fresh LM forward pass needed.

Method mirrors `evaluate_synthetic_intervention` in `src/train.py`: fit two reference
classifiers on genuine raw activations (never touching the SAE), fit Q's SIGNED
sentiment direction (`mean(positive) - mean(negative)`, not the unsigned energy
construction used for sec.19.1's interference score) on a disjoint FIT split, then on an
EVAL split push Q along that direction by `alpha` std-devs, recombine with the untouched P,
decode, and score the reconstruction with the FIT-split reference classifiers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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
from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc  # noqa: E402
from train import _stratified_split_indices, _signed_binary_direction, standardized_logreg_accuracy  # noqa: E402


def load_kron_checkpoint(trainer_config_path: Path, ckpt_path: Path, device: str) -> KronAutoEncoderTopK:
    cfg = json.loads(trainer_config_path.read_text(encoding="utf-8"))
    ae = KronAutoEncoderTopK(
        activation_dim=cfg["activation_dim"],
        h=cfg["h"],
        m=cfg["m"],
        n=cfg["n"],
        k=cfg["k"],
        combine_rule=cfg.get("combine_rule", "mand"),
    )
    state = t.load(ckpt_path, map_location="cpu")
    ae.load_state_dict(state)
    ae = ae.to(device)
    ae.eval()
    return ae


def load_cache_batches(cache_path: Path, max_batches: int, min_doc_id: int = 0):
    """Yields (x, token_doc_ids [offset to be globally unique], topic, sentiment, n_docs)."""
    raw = t.load(cache_path, map_location="cpu", weights_only=False)
    n_docs_seen = min_doc_id
    out = []
    for i, batch in enumerate(raw[:max_batches]):
        x = batch["x"].float()
        doc_ids = batch["token_doc_ids"].long() + n_docs_seen
        topic = batch["topic"].long()
        sentiment = batch["sentiment"].long()
        out.append((x, doc_ids, topic, sentiment))
        n_docs_seen += batch["n_docs"]
    return out, n_docs_seen


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir", type=str, default=str(ROOT / "runs" / "joint_dpo_cross_full420k"),
        help="Run directory containing checkpoints/kron_joint/{trainer_config.json,ae_step_N.pt}.",
    )
    ap.add_argument("--ckpt_step", type=int, default=420000)
    ap.add_argument(
        "--out_path", type=str, default=str(ROOT / "runs" / "real_intervention_results.json"),
    )
    args = ap.parse_args()

    device = "cuda:0" if t.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir) / "checkpoints" / "kron_joint"
    ae = load_kron_checkpoint(run_dir / "trainer_config.json", run_dir / f"ae_step_{args.ckpt_step}.pt", device)
    print(f"Loaded real checkpoint: activation_dim={ae.activation_dim}, h={ae.h}, m={ae.m}, n={ae.n}", flush=True)

    cache_path = ROOT / "runs" / "pythia_erasure_activation_cache.pt"
    batches, n_docs = load_cache_batches(cache_path, max_batches=400)
    print(f"Loaded {len(batches)} cached real batches, {n_docs} docs total", flush=True)

    x_full = t.cat([b[0] for b in batches], dim=0)
    doc_ids_full = t.cat([b[1] for b in batches], dim=0)
    topic_full = t.cat([b[2] for b in batches], dim=0)
    sentiment_full = t.cat([b[3] for b in batches], dim=0)
    # topic/sentiment tensors in the cache are per-batch (length = that batch's n_docs) and
    # already concatenated in doc order matching doc_ids_full's unique values.
    n_docs_total = topic_full.shape[0]
    assert n_docs_total == n_docs, (n_docs_total, n_docs)

    seed = 42
    doc_fit, doc_eval = _stratified_split_indices(sentiment_full, test_fraction=0.5, seed=seed)
    is_fit_doc = t.zeros(n_docs_total, dtype=t.bool)
    is_fit_doc[doc_fit] = True
    token_is_fit = is_fit_doc[doc_ids_full]

    from sklearn.linear_model import LogisticRegression

    with t.no_grad():
        x_docs_fit = _mean_pool_by_doc(
            x_full[token_is_fit].to(device), doc_ids_full[token_is_fit].to(device), n_docs_total
        )[doc_fit].cpu()
    mu_x, sd_x = x_docs_fit.mean(dim=0, keepdim=True), x_docs_fit.std(dim=0, keepdim=True).clamp_min(1e-6)
    clf_sent = LogisticRegression(max_iter=5000, C=1.0).fit(
        ((x_docs_fit - mu_x) / sd_x).numpy(), sentiment_full[doc_fit].numpy()
    )
    clf_top = LogisticRegression(max_iter=5000, C=1.0).fit(
        ((x_docs_fit - mu_x) / sd_x).numpy(), topic_full[doc_fit].numpy()
    )
    print("Reference classifiers fit on genuine raw pythia activations.", flush=True)

    # `ae._combine`/`ae.encode` materializes a dense (n_tokens, dict_size=16384) tensor
    # internally -- for the full 9600-doc set (~380k tokens) that alone is ~25GB and OOMs a
    # single 96GB GPU shared with other jobs. Chunk by token count to bound peak memory;
    # p_pos/q_pos (pre-combine, much smaller: h*m / h*n per token) are kept on CPU across
    # chunks and moved back to GPU only for the per-alpha combine/decode below, chunked again.
    CHUNK_TOKENS = 20000
    n_tokens_total = x_full.shape[0]
    p_pos_chunks, q_pos_chunks = [], []
    with t.no_grad():
        for start in range(0, n_tokens_total, CHUNK_TOKENS):
            end = min(start + CHUNK_TOKENS, n_tokens_total)
            _, p_pos_c, q_pos_c = ae.encode(x_full[start:end].to(device), return_branches=True)
            p_pos_chunks.append(p_pos_c.cpu())
            q_pos_chunks.append(q_pos_c.cpu())
    p_pos_full = t.cat(p_pos_chunks, dim=0)
    q_pos_full = t.cat(q_pos_chunks, dim=0)
    del p_pos_chunks, q_pos_chunks
    print(f"Encoded {n_tokens_total} tokens in {len(range(0, n_tokens_total, CHUNK_TOKENS))} chunks.", flush=True)

    def run_intervention(branch: str, other_pos_full, target_pos_full):
        """`target_pos_full` is the branch being pushed; `other_pos_full` stays untouched.
        Both CPU tensors of shape (n_tokens, h, m) for p or (n_tokens, h, n) for q."""
        with t.no_grad():
            target_shape = target_pos_full.shape
            target_flat_full = target_pos_full.reshape(target_shape[0], -1)
            target_docs_fit = _mean_pool_by_doc(
                target_flat_full[token_is_fit].to(device), doc_ids_full[token_is_fit].to(device), n_docs_total
            )[doc_fit].cpu()
            w_sent = _signed_binary_direction(target_docs_fit, sentiment_full[doc_fit])
            w_sent_hat = (w_sent / w_sent.norm().clamp_min(1e-12)).to(device)
            proj_fit = target_docs_fit.to(device) @ w_sent_hat
            proj_std = float(proj_fit.std().clamp_min(1e-12))
            print(f"[{branch}] sentiment direction fit, ||w||={float(w_sent.norm()):.4f}, proj_std={proj_std:.4f}", flush=True)

            eval_mask = ~token_is_fit  # CPU bool, matches CPU p_pos_full/q_pos_full/x_full
            x_eval_cpu = x_full[eval_mask]
            doc_ids_eval_raw = doc_ids_full[eval_mask]
            uniq_eval_docs, doc_ids_eval_cpu = t.unique(doc_ids_eval_raw, return_inverse=True)
            n_eval_docs = uniq_eval_docs.shape[0]
            topic_eval = topic_full[doc_eval].numpy()

            other_eval = other_pos_full[eval_mask]  # CPU, untouched branch
            target_eval = target_pos_full[eval_mask]  # CPU, branch being pushed
            direction_shaped_cpu = w_sent_hat.view(target_shape[1], target_shape[2]).cpu()

            x_eval_docs = _mean_pool_by_doc(x_eval_cpu.to(device), doc_ids_eval_cpu.to(device), n_eval_docs).cpu()
            x_eval_docs_var = float((x_eval_docs - x_eval_docs.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12))
            x_eval_std = ((x_eval_docs - mu_x) / sd_x).numpy()

            n_eval_tokens = target_eval.shape[0]
            alphas = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
            results = {"branch": branch, "alphas": alphas, "recon_fvu": [], "p_sentiment_positive": [], "topic_acc": []}
            for alpha in alphas:
                x_hat_chunks = []
                for start in range(0, n_eval_tokens, CHUNK_TOKENS):
                    end = min(start + CHUNK_TOKENS, n_eval_tokens)
                    other_c = other_eval[start:end].to(device)
                    target_c = target_eval[start:end].to(device) + alpha * proj_std * direction_shaped_cpu.to(device).unsqueeze(0)
                    target_c = target_c.clamp_min(0.0)
                    p_c, q_c = (target_c, other_c) if branch == "p" else (other_c, target_c)
                    dense = ae._combine(p_c, q_c, p_c, q_c)
                    post_topk = dense.topk(int(ae.k.item()), sorted=False, dim=-1)
                    f_patched = t.zeros_like(dense).scatter_(dim=-1, index=post_topk.indices, src=post_topk.values)
                    x_hat_chunks.append(ae.decode(f_patched).cpu())
                x_hat_patched = t.cat(x_hat_chunks, dim=0)
                x_hat_docs = _mean_pool_by_doc(x_hat_patched.to(device), doc_ids_eval_cpu.to(device), n_eval_docs).cpu()

                recon_fvu = float((x_hat_docs - x_eval_docs).pow(2).sum()) / x_eval_docs_var
                x_hat_std = ((x_hat_docs - mu_x) / sd_x).numpy()
                p_positive = clf_sent.predict_proba(x_hat_std)[:, 1].mean()
                top_acc = float((clf_top.predict(x_hat_std) == topic_eval).mean())

                results["recon_fvu"].append(recon_fvu)
                results["p_sentiment_positive"].append(float(p_positive))
                results["topic_acc"].append(top_acc)
                print(f"  [{branch}] alpha={alpha:6.1f}  fvu={recon_fvu:.4f}  P(sent+)={p_positive:.4f}  topic_acc={top_acc:.4f}", flush=True)

            results["raw_topic_acc_no_intervention"] = float((clf_top.predict(x_eval_std) == topic_eval).mean())
            results["raw_p_sentiment_positive_no_intervention"] = float(clf_sent.predict_proba(x_eval_std)[:, 1].mean())
            results["n_fit_docs"] = int(len(doc_fit))
            results["n_eval_docs"] = int(n_eval_docs)
            return results

    results_q = run_intervention("q", p_pos_full, q_pos_full)
    results_p = run_intervention("p", q_pos_full, p_pos_full)

    out_path = Path(args.out_path)
    out_path.write_text(json.dumps({"q": results_q, "p": results_p}, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
