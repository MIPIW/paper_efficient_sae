"""Joint probe evaluation for DPO-supervised flat/kron runs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch as t
from nnsight import LanguageModel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from data.amazon_reviews import collate_document_batch, load_prepared_datasets
from dictionary_learning.dictionary_kron import KronAutoEncoderTopK
from dictionary_learning.labeled_buffer import LabeledActivationBuffer
from dictionary_learning.trainers.kron_top_k import _mean_pool_by_doc
from dictionary_learning.trainers.top_k import AutoEncoderTopK


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


def stratified_split_indices(labels: t.Tensor, test_fraction: float, seed: int) -> tuple[t.Tensor, t.Tensor]:
    g = t.Generator()
    g.manual_seed(seed)

    train_idx = []
    test_idx = []
    for cls in t.unique(labels):
        idx = (labels == cls).nonzero(as_tuple=False).squeeze(-1)
        perm = idx[t.randperm(idx.numel(), generator=g)]
        n_test = max(1, int(round(idx.numel() * test_fraction))) if idx.numel() > 1 else 0
        n_test = min(n_test, max(0, idx.numel() - 1))
        test_idx.append(perm[:n_test])
        train_idx.append(perm[n_test:])

    train_idx = t.cat(train_idx) if train_idx else t.empty(0, dtype=t.long)
    test_idx = t.cat(test_idx) if test_idx else t.empty(0, dtype=t.long)
    return train_idx, test_idx


def label_count_dict(labels: t.Tensor) -> dict[str, int]:
    out: dict[str, int] = {}
    for cls in t.unique(labels).tolist():
        key = str(int(cls))
        out[key] = int((labels == int(cls)).sum().item())
    return out


def balanced_sentiment_subset_indices(labels: t.Tensor, seed: int) -> t.Tensor:
    neg_idx = (labels == 0).nonzero(as_tuple=False).squeeze(-1)
    pos_idx = (labels == 1).nonzero(as_tuple=False).squeeze(-1)
    n = min(neg_idx.numel(), pos_idx.numel())
    if n <= 0:
        raise RuntimeError("Cannot build balanced sentiment subset: one class has zero examples")

    g = t.Generator()
    g.manual_seed(seed)
    neg_sel = neg_idx[t.randperm(neg_idx.numel(), generator=g)[:n]]
    pos_sel = pos_idx[t.randperm(pos_idx.numel(), generator=g)[:n]]
    return t.cat([neg_sel, pos_sel], dim=0)


def balanced_topic_subset_indices(labels: t.Tensor, seed: int) -> t.Tensor:
    classes = t.unique(labels)
    if classes.numel() < 2:
        raise RuntimeError("Need >=2 topic classes for balanced topic probe")
    class_indices = []
    for cls in classes:
        idx = (labels == cls).nonzero(as_tuple=False).squeeze(-1)
        class_indices.append(idx)
    n = min(idx.numel() for idx in class_indices)
    if n <= 0:
        raise RuntimeError("Cannot build balanced topic subset: one class has zero examples")

    g = t.Generator()
    g.manual_seed(seed)
    chosen = []
    for idx in class_indices:
        sel = idx[t.randperm(idx.numel(), generator=g)[:n]]
        chosen.append(sel)
    return t.cat(chosen, dim=0)


def train_linear_probe(
    X_train: t.Tensor,
    y_train: t.Tensor,
    X_test: t.Tensor,
    y_test: t.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str,
) -> float:
    set_seed(seed)
    model = t.nn.Linear(X_train.shape[1], num_classes).to(device)
    opt = t.optim.Adam(model.parameters(), lr=lr)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    for _ in range(epochs):
        perm = t.randperm(X_train.shape[0], device=device)
        for i in range(0, X_train.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            logits = model(X_train[idx])
            loss = t.nn.functional.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with t.no_grad():
        pred = model(X_test).argmax(dim=-1)
        acc = (pred == y_test).float().mean().item()
    return acc


def collect_doc_representations(
    ae,
    dict_class: str,
    model: LanguageModel,
    submodule,
    eval_ds,
    ctx_len: int,
    doc_batch_size: int,
    num_workers: int,
    max_docs: int,
    device: str,
):
    loader = DataLoader(
        eval_ds,
        batch_size=doc_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=t.cuda.is_available(),
        collate_fn=collate_document_batch,
        drop_last=False,
    )

    buffer = LabeledActivationBuffer(
        data=iter(loader),
        model=model,
        submodule=submodule,
        d_submodule=ae.activation_dim,
        io="out",
        ctx_len=ctx_len,
        device=device,
        remove_bos=False,
        add_special_tokens=True,
        max_activation_norm_multiple=None,
    )

    all_p = []
    all_q = []
    all_full = []
    all_topic = []
    all_sentiment = []
    collected_docs = 0

    ae.eval()
    with t.no_grad():
        while collected_docs < max_docs:
            try:
                batch = next(buffer)
            except StopIteration:
                break

            x = batch.activations.to(device=device, dtype=t.float32)
            token_doc_ids = batch.token_doc_ids.long().to(device)
            topic_labels = batch.topic_labels.long().to(device)
            sentiment_labels = batch.sentiment_labels.long().to(device)
            n_docs = topic_labels.shape[0]

            if dict_class == "KronAutoEncoderTopK":
                _, p_pos, q_pos = ae.encode(x, return_branches=True)
                p_token = p_pos.reshape(p_pos.shape[0], -1)
                q_token = q_pos.reshape(q_pos.shape[0], -1)
                p_docs = _mean_pool_by_doc(p_token, token_doc_ids, n_docs)
                q_docs = _mean_pool_by_doc(q_token, token_doc_ids, n_docs)
                all_p.append(p_docs.detach().cpu())
                all_q.append(q_docs.detach().cpu())
            else:
                post_relu = t.nn.functional.relu(ae.encoder(x - ae.b_dec))
                full_docs = _mean_pool_by_doc(post_relu, token_doc_ids, n_docs)
                all_full.append(full_docs.detach().cpu())

            all_topic.append(topic_labels.detach().cpu())
            all_sentiment.append(sentiment_labels.detach().cpu())
            collected_docs += n_docs

    if not all_topic:
        raise RuntimeError("No document representations collected for probe evaluation")

    topic = t.cat(all_topic, dim=0)[:max_docs]
    sentiment = t.cat(all_sentiment, dim=0)[:max_docs]

    if dict_class == "KronAutoEncoderTopK":
        p = t.cat(all_p, dim=0)[:max_docs]
        q = t.cat(all_q, dim=0)[:max_docs]
        return {"p": p, "q": q}, topic, sentiment

    full = t.cat(all_full, dim=0)[:max_docs]
    return {"full": full}, topic, sentiment


def subset_representations(reps: dict[str, t.Tensor], idx: t.Tensor) -> dict[str, t.Tensor]:
    return {k: v[idx] for k, v in reps.items()}


def evaluate_label(
    reps: dict[str, t.Tensor],
    labels: t.Tensor,
    label_name: str,
    test_fraction: float,
    seed: int,
    probe_epochs: int,
    probe_lr: float,
    probe_batch_size: int,
    device: str,
) -> dict[str, object]:
    if label_name == "sentiment":
        subset_idx = balanced_sentiment_subset_indices(labels, seed=seed)
    elif label_name == "topic":
        subset_idx = balanced_topic_subset_indices(labels, seed=seed)
    else:
        raise ValueError(f"Unsupported label_name={label_name}")

    reps_bal = subset_representations(reps, subset_idx)
    y_bal = labels[subset_idx]

    train_idx, test_idx = stratified_split_indices(y_bal, test_fraction=test_fraction, seed=seed)
    if train_idx.numel() == 0 or test_idx.numel() == 0:
        raise RuntimeError(f"Probe split failed for label={label_name}; not enough data per class")

    y_train = y_bal[train_idx]
    y_test = y_bal[test_idx]
    num_classes = int(y_bal.max().item()) + 1
    uniform_chance = 1.0 / num_classes
    test_counts = t.bincount(y_test)
    majority_chance = test_counts.max().item() / test_counts.sum().item()

    branch_acc: dict[str, float] = {}
    for i, (branch, feats) in enumerate(reps_bal.items()):
        acc = train_linear_probe(
            X_train=feats[train_idx],
            y_train=y_train,
            X_test=feats[test_idx],
            y_test=y_test,
            num_classes=num_classes,
            epochs=probe_epochs,
            lr=probe_lr,
            batch_size=probe_batch_size,
            seed=seed + i,
            device=device,
        )
        branch_acc[branch] = acc

    return {
        "label_name": label_name,
        "num_classes": num_classes,
        "balanced_subset_counts": label_count_dict(y_bal),
        "probe_train_counts": label_count_dict(y_train),
        "probe_test_counts": label_count_dict(y_test),
        "chance_uniform": uniform_chance,
        "chance_majority": majority_chance,
        "branch_accuracy": branch_acc,
        "num_docs_used": int(y_bal.shape[0]),
    }


def evaluate_run(
    run_name: str,
    run_dir: Path,
    model: LanguageModel,
    submodule,
    eval_ds,
    args,
    device: str,
):
    cfg_path = run_dir / "trainer_config.json"
    if not cfg_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ckpt = run_dir / f"ae_step_{args.checkpoint_step}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    dict_class = cfg["dict_class"]
    if dict_class == "KronAutoEncoderTopK":
        ae = KronAutoEncoderTopK(
            activation_dim=cfg["activation_dim"],
            h=cfg["h"],
            m=cfg["m"],
            n=cfg["n"],
            k=cfg["k"],
            combine_rule=cfg["combine_rule"],
        )
    else:
        ae = AutoEncoderTopK(
            activation_dim=cfg["activation_dim"],
            dict_size=cfg["dict_size"],
            k=cfg["k"],
        )

    state = t.load(ckpt, map_location="cpu")
    ae.load_state_dict(state)
    ae.to(device)

    reps, topic_labels, sentiment_labels = collect_doc_representations(
        ae=ae,
        dict_class=dict_class,
        model=model,
        submodule=submodule,
        eval_ds=eval_ds,
        ctx_len=args.ctx_len,
        doc_batch_size=args.doc_batch_size,
        num_workers=args.num_workers,
        max_docs=args.max_probe_docs,
        device=device,
    )

    branch_sanity: dict[str, object] | None = None
    if dict_class == "KronAutoEncoderTopK":
        p = reps["p"]
        q = reps["q"]
        if p.data_ptr() == q.data_ptr():
            raise RuntimeError("Probe bug: P and Q tensors share storage unexpectedly")
        compare_docs = min(8, p.shape[0], q.shape[0])
        compare_dim = min(8, p.shape[1], q.shape[1])
        same_prefix = bool(t.allclose(p[:compare_docs, :compare_dim], q[:compare_docs, :compare_dim]))
        if same_prefix:
            raise RuntimeError("Probe bug: P and Q feature prefixes are unexpectedly identical")
        branch_sanity = {
            "p_shape": [int(p.shape[0]), int(p.shape[1])],
            "q_shape": [int(q.shape[0]), int(q.shape[1])],
            "p_doc0_first5": [float(v) for v in p[0, :5].tolist()] if p.shape[0] > 0 else [],
            "q_doc0_first5": [float(v) for v in q[0, :5].tolist()] if q.shape[0] > 0 else [],
            "p_q_prefix_equal_check": same_prefix,
        }

    sentiment_eval = evaluate_label(
        reps=reps,
        labels=sentiment_labels,
        label_name="sentiment",
        test_fraction=args.test_fraction,
        seed=args.seed,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        probe_batch_size=args.probe_batch_size,
        device=device,
    )
    topic_eval = evaluate_label(
        reps=reps,
        labels=topic_labels,
        label_name="topic",
        test_fraction=args.test_fraction,
        seed=args.seed + 7,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        probe_batch_size=args.probe_batch_size,
        device=device,
    )

    return {
        "run_name": run_name,
        "dict_class": dict_class,
        "feature_dims": {k: int(v.shape[1]) for k, v in reps.items()},
        "branch_sanity": branch_sanity,
        "sentiment_eval": sentiment_eval,
        "topic_eval": topic_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint probe evaluation for DPO-supervised runs")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=None)
    parser.add_argument("--ctx_len", type=int, default=None)
    parser.add_argument("--doc_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_probe_docs", type=int, default=12000)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--probe_epochs", type=int, default=30)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")

    run_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_args = run_meta["args"]

    dataset_cache_dir = args.dataset_cache_dir or train_args["dataset_cache_dir"]
    checkpoint_step = args.checkpoint_step if args.checkpoint_step is not None else train_args["total_steps"]
    ctx_len = args.ctx_len if args.ctx_len is not None else train_args["ctx_len"]
    device = args.device or ("cuda:0" if t.cuda.is_available() else "cpu")

    _, eval_ds, _, _ = load_prepared_datasets(dataset_cache_dir)

    model = LanguageModel(
        train_args["model_name"],
        dispatch=True,
        device_map=device,
    )
    submodule = model.gpt_neox.layers[train_args["layer"]]

    ckpt_root = run_dir / "checkpoints"
    run_dirs = sorted([p for p in ckpt_root.iterdir() if p.is_dir()])

    results = []
    for one_run_dir in run_dirs:
        result = evaluate_run(
            run_name=one_run_dir.name,
            run_dir=one_run_dir,
            model=model,
            submodule=submodule,
            eval_ds=eval_ds,
            args=argparse.Namespace(
                checkpoint_step=checkpoint_step,
                ctx_len=ctx_len,
                doc_batch_size=args.doc_batch_size,
                num_workers=args.num_workers,
                max_probe_docs=args.max_probe_docs,
                test_fraction=args.test_fraction,
                probe_epochs=args.probe_epochs,
                probe_lr=args.probe_lr,
                probe_batch_size=args.probe_batch_size,
                seed=args.seed,
            ),
            device=device,
        )
        if result is not None:
            results.append(result)

    out = {
        "run_dir": str(run_dir),
        "dataset_cache_dir": str(dataset_cache_dir),
        "checkpoint_step": checkpoint_step,
        "results": results,
    }
    out_path = run_dir / "joint_probe_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for r in results:
        if r["dict_class"] == "KronAutoEncoderTopK":
            p_sent = r["sentiment_eval"]["branch_accuracy"]["p"]
            p_topic = r["topic_eval"]["branch_accuracy"]["p"]
            q_sent = r["sentiment_eval"]["branch_accuracy"]["q"]
            q_topic = r["topic_eval"]["branch_accuracy"]["q"]
            print(
                f"{r['run_name']} | arch=kron | "
                f"P->sent={p_sent:.4f}, P->topic={p_topic:.4f}, "
                f"Q->sent={q_sent:.4f}, Q->topic={q_topic:.4f} | "
                f"Pshape={r['branch_sanity']['p_shape']}, Qshape={r['branch_sanity']['q_shape']} | "
                f"sent_test_counts={r['sentiment_eval']['probe_test_counts']} | "
                f"sent_chance={r['sentiment_eval']['chance_majority']:.4f} | "
                f"topic_test_counts={r['topic_eval']['probe_test_counts']} | "
                f"topic_chance={r['topic_eval']['chance_majority']:.4f}",
                flush=True,
            )
        else:
            full_sent = r["sentiment_eval"]["branch_accuracy"]["full"]
            full_topic = r["topic_eval"]["branch_accuracy"]["full"]
            print(
                f"{r['run_name']} | arch=flat | full->sent={full_sent:.4f}, full->topic={full_topic:.4f} | "
                f"sent_test_counts={r['sentiment_eval']['probe_test_counts']} | "
                f"sent_chance={r['sentiment_eval']['chance_majority']:.4f} | "
                f"topic_test_counts={r['topic_eval']['probe_test_counts']} | "
                f"topic_chance={r['topic_eval']['chance_majority']:.4f}",
                flush=True,
            )
    print(f"Saved joint probe results to: {out_path}")


if __name__ == "__main__":
    main()
