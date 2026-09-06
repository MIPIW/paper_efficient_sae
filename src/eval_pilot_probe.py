"""Pilot diagnostic probe: flat full-vector probe and Kron P/Q branch probes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    """Return indices for an exactly 50:50 subset over sentiment labels {0,1}."""
    if labels.numel() == 0:
        return t.empty(0, dtype=t.long)

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
    label_type: str,
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
    all_y = []
    collected_docs = 0

    ae.eval()
    with t.no_grad():
        while True:
            if collected_docs >= max_docs:
                break
            try:
                batch = next(buffer)
            except StopIteration:
                break

            x = batch.activations.to(device=device, dtype=t.float32)
            token_doc_ids = batch.token_doc_ids.long().to(device)
            topic_labels = batch.topic_labels.long().to(device)
            sentiment_labels = batch.sentiment_labels.long().to(device)
            labels = topic_labels if label_type == "topic" else sentiment_labels
            n_docs = labels.shape[0]

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
            all_y.append(labels.detach().cpu())
            collected_docs += n_docs

    if not all_y:
        raise RuntimeError("No document representations collected for probe evaluation")

    Y = t.cat(all_y, dim=0)[:max_docs]
    if dict_class == "KronAutoEncoderTopK":
        P = t.cat(all_p, dim=0)[:max_docs]
        Q = t.cat(all_q, dim=0)[:max_docs]
        return {"p": P, "q": Q}, Y

    FULL = t.cat(all_full, dim=0)[:max_docs]
    return {"full": FULL}, Y


def subset_representations(reps: dict[str, t.Tensor], idx: t.Tensor) -> dict[str, t.Tensor]:
    return {k: v[idx] for k, v in reps.items()}


def inspect_eval_label_counts(eval_ds) -> dict[str, dict[str, int]]:
    topic_counts: dict[str, int] = defaultdict(int)
    sentiment_counts: dict[str, int] = defaultdict(int)
    for row in eval_ds.rows:
        topic_counts[str(int(row["topic_label"]))] += 1
        sentiment_counts[str(int(row["sentiment_label"]))] += 1
    return {
        "topic_counts": dict(sorted(topic_counts.items(), key=lambda kv: int(kv[0]))),
        "sentiment_counts": dict(sorted(sentiment_counts.items(), key=lambda kv: int(kv[0]))),
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
    label_type = cfg.get("label_type")
    if label_type not in {"topic", "sentiment"}:
        return None

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

    reps, Y = collect_doc_representations(
        ae=ae,
        dict_class=dict_class,
        label_type=label_type,
        model=model,
        submodule=submodule,
        eval_ds=eval_ds,
        ctx_len=args.ctx_len,
        doc_batch_size=args.doc_batch_size,
        num_workers=args.num_workers,
        max_docs=args.max_probe_docs,
        device=device,
    )

    full_counts = label_count_dict(Y)
    balanced_eval_applied = False
    balanced_subset_counts: dict[str, int] | None = None
    if label_type == "sentiment":
        subset_idx = balanced_sentiment_subset_indices(Y, seed=args.seed)
        reps = subset_representations(reps, subset_idx)
        Y = Y[subset_idx]
        balanced_eval_applied = True
        balanced_subset_counts = label_count_dict(Y)

    train_idx, test_idx = stratified_split_indices(Y, args.test_fraction, args.seed)
    if train_idx.numel() == 0 or test_idx.numel() == 0:
        raise RuntimeError(f"Probe split failed for run={run_name}; not enough data per class")

    y_train = Y[train_idx]
    y_test = Y[test_idx]

    num_classes = int(Y.max().item()) + 1
    test_counts = t.bincount(y_test)
    majority_chance = (test_counts.max().item() / test_counts.sum().item()) if test_counts.numel() > 0 else float("nan")
    uniform_chance = 1.0 / num_classes

    result = {
        "run_name": run_name,
        "label_type": label_type,
        "dict_class": dict_class,
        "num_docs_used": int(Y.shape[0]),
        "num_classes": num_classes,
        "all_collected_label_counts": full_counts,
        "balanced_eval_applied": balanced_eval_applied,
        "balanced_subset_label_counts": balanced_subset_counts,
        "probe_train_label_counts": label_count_dict(y_train),
        "probe_test_label_counts": label_count_dict(y_test),
        "probe_train_size": int(y_train.shape[0]),
        "probe_test_size": int(y_test.shape[0]),
        "chance_uniform": uniform_chance,
        "chance_majority": majority_chance,
    }

    if dict_class == "KronAutoEncoderTopK":
        p_train = reps["p"][train_idx]
        p_test = reps["p"][test_idx]
        q_train = reps["q"][train_idx]
        q_test = reps["q"][test_idx]

        p_acc = train_linear_probe(
            X_train=p_train,
            y_train=y_train,
            X_test=p_test,
            y_test=y_test,
            num_classes=num_classes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            batch_size=args.probe_batch_size,
            seed=args.seed,
            device=device,
        )
        q_acc = train_linear_probe(
            X_train=q_train,
            y_train=y_train,
            X_test=q_test,
            y_test=y_test,
            num_classes=num_classes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
            batch_size=args.probe_batch_size,
            seed=args.seed + 1,
            device=device,
        )
        result["p_probe_accuracy"] = p_acc
        result["q_probe_accuracy"] = q_acc
        result["p_feature_dim"] = int(reps["p"].shape[1])
        result["q_feature_dim"] = int(reps["q"].shape[1])
        return result

    full_train = reps["full"][train_idx]
    full_test = reps["full"][test_idx]
    full_acc = train_linear_probe(
        X_train=full_train,
        y_train=y_train,
        X_test=full_test,
        y_test=y_test,
        num_classes=num_classes,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        batch_size=args.probe_batch_size,
        seed=args.seed,
        device=device,
    )
    result["full_probe_accuracy"] = full_acc
    result["full_feature_dim"] = int(reps["full"].shape[1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot probe evaluation for collect-vs-contrast supervision")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=None)
    parser.add_argument("--ctx_len", type=int, default=None)
    parser.add_argument("--doc_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_probe_docs", type=int, default=4000)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--probe_epochs", type=int, default=30)
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--inspect_eval_balance_only",
        action="store_true",
        help="Data-only mode: inspect eval-label balance logic without loading checkpoints/models.",
    )
    parser.add_argument(
        "--inspect_label_type",
        type=str,
        choices=["topic", "sentiment"],
        default="sentiment",
        help="Label type to inspect when --inspect_eval_balance_only is set.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.inspect_eval_balance_only:
        if args.dataset_cache_dir is None:
            raise ValueError("--dataset_cache_dir is required with --inspect_eval_balance_only")

        _, eval_ds, _, _ = load_prepared_datasets(args.dataset_cache_dir)
        counts = inspect_eval_label_counts(eval_ds)
        print(f"Inspect mode on eval split: {args.dataset_cache_dir}")
        print(f"topic_counts={counts['topic_counts']}")
        print(f"sentiment_counts={counts['sentiment_counts']}")

        if args.inspect_label_type == "sentiment":
            labels = t.tensor([int(row["sentiment_label"]) for row in eval_ds.rows], dtype=t.long)
            balanced_idx = balanced_sentiment_subset_indices(labels, seed=args.seed)
            y_bal = labels[balanced_idx]
            bal_counts = label_count_dict(y_bal)
            num_classes = int(y_bal.max().item()) + 1
            chance_uniform = 1.0 / num_classes
            bal_bincount = t.bincount(y_bal)
            chance_majority = bal_bincount.max().item() / bal_bincount.sum().item()
            _, test_idx = stratified_split_indices(y_bal, test_fraction=args.test_fraction, seed=args.seed)
            y_test = y_bal[test_idx]
            test_counts = label_count_dict(y_test)
            test_bincount = t.bincount(y_test)
            test_chance_majority = test_bincount.max().item() / test_bincount.sum().item()
            print(f"balanced_sentiment_counts={bal_counts}")
            print(
                f"balanced_sentiment_size={int(y_bal.shape[0])} | "
                f"chance_uniform={chance_uniform:.4f} | chance_majority={chance_majority:.4f}"
            )
            print(
                f"probe_test_counts_if_evaluated={test_counts} | "
                f"probe_test_size={int(y_test.shape[0])} | "
                f"test_chance_uniform={chance_uniform:.4f} | "
                f"test_chance_majority={test_chance_majority:.4f}"
            )
        else:
            labels = t.tensor([int(row["topic_label"]) for row in eval_ds.rows], dtype=t.long)
            topic_counts = label_count_dict(labels)
            vals = list(topic_counts.values())
            spread = max(vals) - min(vals) if vals else 0
            print(f"topic_count_spread_max_minus_min={spread}")
        return

    if args.run_dir is None:
        raise ValueError("--run_dir is required unless --inspect_eval_balance_only is set")

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
    out_path = run_dir / "pilot_probe_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for r in results:
        if r["dict_class"] == "KronAutoEncoderTopK":
            print(
                f"{r['run_name']} | arch=kron | label={r['label_type']} | "
                f"P-acc={r['p_probe_accuracy']:.4f} | Q-acc={r['q_probe_accuracy']:.4f} | "
                f"test_counts={r['probe_test_label_counts']} | "
                f"chance_uniform={r['chance_uniform']:.4f} | chance_majority={r['chance_majority']:.4f} | "
                f"balanced_eval_applied={r['balanced_eval_applied']}",
                flush=True,
            )
        else:
            print(
                f"{r['run_name']} | arch=flat | label={r['label_type']} | "
                f"full-acc={r['full_probe_accuracy']:.4f} | "
                f"test_counts={r['probe_test_label_counts']} | "
                f"chance_uniform={r['chance_uniform']:.4f} | chance_majority={r['chance_majority']:.4f} | "
                f"balanced_eval_applied={r['balanced_eval_applied']}",
                flush=True,
            )
    print(f"Saved probe results to: {out_path}")


if __name__ == "__main__":
    main()
