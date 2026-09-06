"""Synthetic-data version of the flat-SAE patch-and-generate causal steering eval
(REPORT_FULL.md sec.19.9, the synthetic-data counterpart to sec.19.8's real-data result).

Question: sec.19.8 showed the flat SAE achieves a nominally wider steering range on real
pythia-410m data, but only by exploding reconstruction FVU (up to 35x at alpha=+6) -- does the
same "wider range, blown-up FVU" pattern hold on the synthetic diagnostic buffer, the same
apples-to-apples setting used for every Kron synthetic result in sec.19.3-19.6?

Trains a flat AutoEncoderTopK (dict_size=16384) fresh via `FlatSupervisedTopKTrainer` on the
synthetic joint buffer (identical protocol to every other synthetic result: 5000 steps, seed=42,
dpo_beta=2.0, joint_dpo_cross), then runs the same push-pre-topk/re-topk/decode method as
`eval_flat_intervention.py`, scored by external reference classifiers fit on genuine synthetic
`x`, matching `evaluate_synthetic_intervention`'s Kron-branch method exactly except for the
flat/undivided architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc  # noqa: E402
from train import (  # noqa: E402
    FlatSupervisedTopKTrainer,
    SyntheticJointActivationBuffer,
    _signed_binary_direction,
    _stratified_split_indices,
)


def _collect_synthetic(
    buffer: SyntheticJointActivationBuffer, n_batches: int, doc_offset: int = 0
):
    """Mirrors the collection pattern used elsewhere in this project's synthetic evals: pull
    `n_batches` batches, concatenating activations/doc_ids (offset to stay globally unique)/
    topic/sentiment labels."""
    xs, doc_ids, topics, sentiments = [], [], [], []
    n_docs_seen = doc_offset
    for _ in range(n_batches):
        batch = next(buffer)
        xs.append(batch.activations.float())
        doc_ids.append(batch.token_doc_ids.long() + n_docs_seen)
        topics.append(batch.topic_labels.long())
        sentiments.append(batch.sentiment_labels.long())
        n_docs_seen += int(batch.token_doc_ids.max().item()) + 1
    return (
        t.cat(xs, dim=0),
        t.cat(doc_ids, dim=0),
        t.cat(topics, dim=0),
        t.cat(sentiments, dim=0),
        n_docs_seen,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=str(ROOT / "runs" / "idea_flat_synthetic"))
    ap.add_argument("--ckpt_step", type=int, default=5000)
    ap.add_argument("--total_steps", type=int, default=5000, help="Training steps if the checkpoint doesn't exist yet.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpo_beta", type=float, default=2.0)
    ap.add_argument("--flat_dict_size", type=int, default=16384)
    ap.add_argument("--activation_dim", type=int, default=1024)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--out_path", type=str, default=str(ROOT / "runs" / "synthetic_intervention_results_flat.json"))
    args = ap.parse_args()

    device = "cuda:0" if t.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir) / "checkpoints" / "flat_joint"
    cfg_path = run_dir / "trainer_config.json"
    ckpt_path = run_dir / f"ae_step_{args.ckpt_step}.pt"
    if not (cfg_path.exists() and ckpt_path.exists()):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} -- train it first via: "
            f"python src/train.py --joint --joint_mode dpo_cross --synthetic_diagnostic "
            f"--only_trainers flat_joint --dpo_beta {args.dpo_beta} --seed {args.seed} "
            f"--total_steps {args.total_steps} --flat_dict_size {args.flat_dict_size} "
            f"--output_dir {args.run_dir}"
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    from dictionary_learning.trainers.top_k import AutoEncoderTopK

    ae = AutoEncoderTopK(activation_dim=cfg["activation_dim"], dict_size=cfg["dict_size"], k=cfg["k"])
    ae.load_state_dict(t.load(ckpt_path, map_location="cpu"))
    ae = ae.to(device)
    ae.eval()
    print(f"Loaded flat synthetic checkpoint: activation_dim={ae.activation_dim}, dict_size={ae.dict_size}, k={int(ae.k.item())}", flush=True)

    # Two disjoint synthetic streams (FIT / EVAL), matching evaluate_synthetic_intervention's
    # use of a separate seed offset per role.
    fit_buffer = SyntheticJointActivationBuffer(
        activation_dim=args.activation_dim, doc_batch_size=16, num_topics=6, num_sentiments=2,
        topic_signal_scale=1.0, sentiment_signal_scale=1.0, noise_std=1.0,
        seed=args.seed + 1001, device=device,
    )
    eval_buffer = SyntheticJointActivationBuffer(
        activation_dim=args.activation_dim, doc_batch_size=16, num_topics=6, num_sentiments=2,
        topic_signal_scale=1.0, sentiment_signal_scale=1.0, noise_std=1.0,
        seed=args.seed + 4001, device=device,
    )
    NUM_BATCHES = 64
    x_fit, doc_ids_fit, topic_fit, sentiment_fit, n_fit_docs = _collect_synthetic(fit_buffer, NUM_BATCHES)
    x_eval, doc_ids_eval, topic_eval_t, sentiment_eval_t, n_eval_docs = _collect_synthetic(eval_buffer, NUM_BATCHES)

    from sklearn.linear_model import LogisticRegression

    with t.no_grad():
        x_docs_fit = _mean_pool_by_doc(x_fit.to(device), doc_ids_fit.to(device), n_fit_docs).cpu()
    mu_x, sd_x = x_docs_fit.mean(dim=0, keepdim=True), x_docs_fit.std(dim=0, keepdim=True).clamp_min(1e-6)
    clf_sent = LogisticRegression(max_iter=5000, C=1.0).fit(
        ((x_docs_fit - mu_x) / sd_x).numpy(), _per_doc_labels(sentiment_fit, doc_ids_fit, n_fit_docs)
    )
    clf_top = LogisticRegression(max_iter=5000, C=1.0).fit(
        ((x_docs_fit - mu_x) / sd_x).numpy(), _per_doc_labels(topic_fit, doc_ids_fit, n_fit_docs)
    )
    print("Reference classifiers fit on genuine synthetic raw activations.", flush=True)

    k = int(ae.k.item())
    with t.no_grad():
        _, _, _, post_relu_fit_BF = ae.encode(x_fit.to(device), return_topk=True)
        post_relu_fit_BF = post_relu_fit_BF.cpu()
    post_relu_docs_fit = _mean_pool_by_doc(post_relu_fit_BF.to(device), doc_ids_fit.to(device), n_fit_docs).cpu()
    sentiment_fit_docs = t.tensor(_per_doc_labels(sentiment_fit, doc_ids_fit, n_fit_docs))
    w_sent = _signed_binary_direction(post_relu_docs_fit, sentiment_fit_docs)
    w_sent_hat = (w_sent / w_sent.norm().clamp_min(1e-12)).to(device)
    proj_fit = post_relu_docs_fit.to(device) @ w_sent_hat
    proj_std = float(proj_fit.std().clamp_min(1e-12))
    print(f"sentiment direction fit, ||w||={float(w_sent.norm()):.4f}, proj_std={proj_std:.4f}", flush=True)

    with t.no_grad():
        _, _, _, post_relu_eval_BF = ae.encode(x_eval.to(device), return_topk=True)
        post_relu_eval_BF = post_relu_eval_BF.cpu()
    topic_eval_docs = _per_doc_labels(topic_eval_t, doc_ids_eval, n_eval_docs)

    x_eval_docs = _mean_pool_by_doc(x_eval.to(device), doc_ids_eval.to(device), n_eval_docs).cpu()
    x_eval_docs_var = float((x_eval_docs - x_eval_docs.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12))
    x_eval_std = ((x_eval_docs - mu_x) / sd_x).numpy()

    alphas = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
    results = {"branch": "flat_full16384_synthetic", "alphas": alphas, "recon_fvu": [], "p_sentiment_positive": [], "topic_acc": []}
    for alpha in alphas:
        with t.no_grad():
            pushed = post_relu_eval_BF.to(device) + alpha * proj_std * w_sent_hat.unsqueeze(0)
            pushed = pushed.clamp_min(0.0)
            post_topk = pushed.topk(k, sorted=False, dim=-1)
            f_patched = t.zeros_like(pushed).scatter_(dim=-1, index=post_topk.indices, src=post_topk.values)
            x_hat = ae.decode(f_patched)
            x_hat_docs = _mean_pool_by_doc(x_hat, doc_ids_eval.to(device), n_eval_docs).cpu()

        recon_fvu = float((x_hat_docs - x_eval_docs).pow(2).sum()) / x_eval_docs_var
        x_hat_std = ((x_hat_docs - mu_x) / sd_x).numpy()
        p_positive = clf_sent.predict_proba(x_hat_std)[:, 1].mean()
        top_acc = float((clf_top.predict(x_hat_std) == topic_eval_docs).mean())

        results["recon_fvu"].append(recon_fvu)
        results["p_sentiment_positive"].append(float(p_positive))
        results["topic_acc"].append(top_acc)
        print(f"  alpha={alpha:6.1f}  fvu={recon_fvu:.4f}  P(sent+)={p_positive:.4f}  topic_acc={top_acc:.4f}", flush=True)

    results["raw_topic_acc_no_intervention"] = float((clf_top.predict(x_eval_std) == topic_eval_docs).mean())
    results["raw_p_sentiment_positive_no_intervention"] = float(clf_sent.predict_proba(x_eval_std)[:, 1].mean())
    results["n_fit_docs"] = int(n_fit_docs)
    results["n_eval_docs"] = int(n_eval_docs)

    out_path = Path(args.out_path)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}", flush=True)


def _per_doc_labels(token_labels: t.Tensor, doc_ids: t.Tensor, n_docs: int):
    """Per-token labels -> per-doc labels (every token of a doc shares one label; take the
    first token's label per doc via a scatter to avoid assuming token order)."""
    import numpy as np

    out = -np.ones(n_docs, dtype=np.int64)
    doc_ids_np = doc_ids.detach().cpu().numpy()
    labels_np = token_labels.detach().cpu().numpy()
    out[doc_ids_np] = labels_np
    assert (out >= 0).all(), "some doc has no token / label unset"
    return out


if __name__ == "__main__":
    main()
