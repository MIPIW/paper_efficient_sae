"""Full-width flat SAE version of the patch-and-generate causal steering eval
(REPORT_FULL.md sec.19.8, follow-up to sec.19.3-19.7's Kron-branch results).

Question: does a plain, undivided flat AutoEncoderTopK (dict_size=16384, no P/Q split at all
-- the single "branch" is 8-16x wider than any individual Kron branch tested so far) support
*more* steering control than Kron's branches, given sec.19.6-19.7 showed steering power tracks
capacity? Or does the lack of any compositional/two-factor architectural bias mean a flat SAE's
directions are just as good, worse, or differently-behaved?

Reuses the already-trained real flat_joint checkpoint (`runs/joint_dpo_cross_full420k`,
420,000 steps, joint_dpo_cross, dict_size=16384 -- trained alongside kron_joint in the same
run, no retraining needed) and the same cached real pythia-410m layer-12 activations used by
`eval_real_intervention.py` (`runs/pythia_erasure_activation_cache.pt`).

Method mirrors `eval_real_intervention.py` exactly, with one architectural adaptation: there is
no P/Q branch split to push independently, so the push targets the SAE's single *pre-topk*
dense ReLU activation (`post_relu_feat_acts_BF`, the same architectural point Kron's
`_combine`'s dense pre-topk output corresponds to), then re-applies top-k selection before
decoding -- an intervention that bypasses top-k entirely would let it bleed into features
never actually used for reconstruction, which is not the property being tested.
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

from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402
from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc  # noqa: E402
from train import _stratified_split_indices, _signed_binary_direction, standardized_logreg_accuracy  # noqa: E402

from eval_real_intervention import load_cache_batches  # noqa: E402 (reuse cache loader verbatim)


def load_flat_checkpoint(trainer_config_path: Path, ckpt_path: Path, device: str) -> AutoEncoderTopK:
    cfg = json.loads(trainer_config_path.read_text(encoding="utf-8"))
    ae = AutoEncoderTopK(
        activation_dim=cfg["activation_dim"],
        dict_size=cfg["dict_size"],
        k=cfg["k"],
    )
    state = t.load(ckpt_path, map_location="cpu")
    ae.load_state_dict(state)
    ae = ae.to(device)
    ae.eval()
    return ae


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=str(ROOT / "runs" / "joint_dpo_cross_full420k"))
    ap.add_argument("--ckpt_step", type=int, default=420000)
    ap.add_argument("--out_path", type=str, default=str(ROOT / "runs" / "real_intervention_results_flat.json"))
    args = ap.parse_args()

    device = "cuda:0" if t.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir) / "checkpoints" / "flat_joint"
    ae = load_flat_checkpoint(run_dir / "trainer_config.json", run_dir / f"ae_step_{args.ckpt_step}.pt", device)
    print(f"Loaded flat checkpoint: activation_dim={ae.activation_dim}, dict_size={ae.dict_size}, k={int(ae.k.item())}", flush=True)

    cache_path = ROOT / "runs" / "pythia_erasure_activation_cache.pt"
    batches, n_docs = load_cache_batches(cache_path, max_batches=400)
    print(f"Loaded {len(batches)} cached real batches, {n_docs} docs total", flush=True)

    x_full = t.cat([b[0] for b in batches], dim=0)
    doc_ids_full = t.cat([b[1] for b in batches], dim=0)
    topic_full = t.cat([b[2] for b in batches], dim=0)
    sentiment_full = t.cat([b[3] for b in batches], dim=0)
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

    # Same OOM concern as eval_real_intervention.py: encode() materializes a dense
    # (n_tokens, dict_size=16384) tensor -- for ~421k tokens that alone is ~25GB. Chunk by
    # token count. Store the PRE-TOPK dense ReLU activations (post_relu_feat_acts_BF), the
    # architectural analogue of Kron's pre-combine p_pos/q_pos -- pushing here and then
    # re-applying top-k mirrors exactly what the Kron eval does at its combine/topk boundary.
    CHUNK_TOKENS = 20000
    n_tokens_total = x_full.shape[0]
    pre_topk_chunks = []
    k = int(ae.k.item())
    with t.no_grad():
        for start in range(0, n_tokens_total, CHUNK_TOKENS):
            end = min(start + CHUNK_TOKENS, n_tokens_total)
            _, _, _, post_relu_BF = ae.encode(x_full[start:end].to(device), return_topk=True)
            pre_topk_chunks.append(post_relu_BF.cpu())
    pre_topk_full = t.cat(pre_topk_chunks, dim=0)
    del pre_topk_chunks
    print(f"Encoded {n_tokens_total} tokens in {len(range(0, n_tokens_total, CHUNK_TOKENS))} chunks.", flush=True)

    with t.no_grad():
        pre_topk_docs_fit = _mean_pool_by_doc(
            pre_topk_full[token_is_fit].to(device), doc_ids_full[token_is_fit].to(device), n_docs_total
        )[doc_fit].cpu()
    w_sent = _signed_binary_direction(pre_topk_docs_fit, sentiment_full[doc_fit])
    w_sent_hat = (w_sent / w_sent.norm().clamp_min(1e-12)).to(device)
    proj_fit = pre_topk_docs_fit.to(device) @ w_sent_hat
    proj_std = float(proj_fit.std().clamp_min(1e-12))
    print(f"sentiment direction fit, ||w||={float(w_sent.norm()):.4f}, proj_std={proj_std:.4f}", flush=True)

    eval_mask = ~token_is_fit
    x_eval_cpu = x_full[eval_mask]
    doc_ids_eval_raw = doc_ids_full[eval_mask]
    uniq_eval_docs, doc_ids_eval_cpu = t.unique(doc_ids_eval_raw, return_inverse=True)
    n_eval_docs = uniq_eval_docs.shape[0]
    topic_eval = topic_full[doc_eval].numpy()

    pre_topk_eval = pre_topk_full[eval_mask]

    x_eval_docs = _mean_pool_by_doc(x_eval_cpu.to(device), doc_ids_eval_cpu.to(device), n_eval_docs).cpu()
    x_eval_docs_var = float((x_eval_docs - x_eval_docs.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12))
    x_eval_std = ((x_eval_docs - mu_x) / sd_x).numpy()

    n_eval_tokens = pre_topk_eval.shape[0]
    alphas = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
    results = {"branch": "flat_full16384", "alphas": alphas, "recon_fvu": [], "p_sentiment_positive": [], "topic_acc": []}
    for alpha in alphas:
        with t.no_grad():
            x_hat_chunks = []
            for start in range(0, n_eval_tokens, CHUNK_TOKENS):
                end = min(start + CHUNK_TOKENS, n_eval_tokens)
                pushed_c = pre_topk_eval[start:end].to(device) + alpha * proj_std * w_sent_hat.unsqueeze(0)
                pushed_c = pushed_c.clamp_min(0.0)
                post_topk = pushed_c.topk(k, sorted=False, dim=-1)
                f_patched = t.zeros_like(pushed_c).scatter_(dim=-1, index=post_topk.indices, src=post_topk.values)
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
        print(f"  alpha={alpha:6.1f}  fvu={recon_fvu:.4f}  P(sent+)={p_positive:.4f}  topic_acc={top_acc:.4f}", flush=True)

    results["raw_topic_acc_no_intervention"] = float((clf_top.predict(x_eval_std) == topic_eval).mean())
    results["raw_p_sentiment_positive_no_intervention"] = float(clf_sent.predict_proba(x_eval_std)[:, 1].mean())
    results["n_fit_docs"] = int(len(doc_fit))
    results["n_eval_docs"] = int(n_eval_docs)

    out_path = Path(args.out_path)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
