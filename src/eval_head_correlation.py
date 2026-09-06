"""Section 5.2-style within-head correlation analysis for KronSAE checkpoints.

This script:
1) Collects residual-stream activations on held-out texts via the repo's
   LabeledActivationBuffer + nnsight pipeline.
2) Computes Kron feature activations per document.
3) Computes per-feature mean Pearson correlation within-head vs across-head.
4) Compares trained checkpoints to random-uniform KronSAEs of identical shape.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
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

from data.amazon_reviews import collate_document_batch, load_prepared_datasets  # noqa: E402
from dictionary_learning.dictionary_kron import KronAutoEncoderTopK  # noqa: E402
from dictionary_learning.labeled_buffer import LabeledActivationBuffer  # noqa: E402


@dataclass
class CorrelationStats:
    mean_within_head: float
    mean_across_head: float
    mean_delta_within_minus_across: float
    std_within_head: float
    std_across_head: float
    valid_feature_count: int
    total_feature_count: int


@dataclass
class EvalResult:
    checkpoint_name: str
    mode: str
    stats: CorrelationStats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def collect_token_activations(
    model: LanguageModel,
    submodule,
    eval_ds,
    ctx_len: int,
    doc_batch_size: int,
    num_workers: int,
    max_docs: int,
    device: str,
) -> tuple[t.Tensor, t.Tensor, int]:
    """Collect token activations and global doc ids for the first `max_docs` eval docs.

    Returns:
      token_acts: Float16 tensor [num_tokens, d_model] on CPU
      token_doc_ids: Int64 tensor [num_tokens] on CPU in [0, max_docs)
      num_docs_collected: int
    """

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
        d_submodule=1024,
        io="out",
        ctx_len=ctx_len,
        device=device,
        remove_bos=False,
        add_special_tokens=True,
        max_activation_norm_multiple=None,
    )

    token_chunks: list[t.Tensor] = []
    doc_id_chunks: list[t.Tensor] = []
    docs_collected = 0

    while docs_collected < max_docs:
        try:
            batch = next(buffer)
        except StopIteration:
            break

        n_docs_batch = int(batch.topic_labels.shape[0])
        if n_docs_batch <= 0:
            continue

        docs_needed = max_docs - docs_collected
        take_docs = min(n_docs_batch, docs_needed)

        if take_docs < n_docs_batch:
            keep_mask = batch.token_doc_ids < take_docs
            acts = batch.activations[keep_mask]
            local_doc_ids = batch.token_doc_ids[keep_mask]
        else:
            acts = batch.activations
            local_doc_ids = batch.token_doc_ids

        global_doc_ids = local_doc_ids.to(dtype=t.long) + docs_collected

        token_chunks.append(acts.detach().cpu().to(dtype=t.float16))
        doc_id_chunks.append(global_doc_ids.detach().cpu().to(dtype=t.long))

        docs_collected += take_docs
        print(
            f"[collect] docs={docs_collected}/{max_docs} | tokens_so_far={sum(x.shape[0] for x in token_chunks)}",
            flush=True,
        )

    if docs_collected == 0:
        raise RuntimeError("No activations collected from eval split")

    token_acts = t.cat(token_chunks, dim=0)
    token_doc_ids = t.cat(doc_id_chunks, dim=0)
    return token_acts, token_doc_ids, docs_collected


def init_kron_uniform_(ae: KronAutoEncoderTopK, seed: int) -> None:
    """Uniform random init baseline, matching paper's random-uniform comparison intent."""

    g = t.Generator(device="cpu")
    g.manual_seed(seed)

    with t.no_grad():
        bound_proj = 1.0 / math.sqrt(ae.activation_dim)
        ae.p_proj.uniform_(-bound_proj, bound_proj, generator=g)
        ae.q_proj.uniform_(-bound_proj, bound_proj, generator=g)
        ae.p_bias.uniform_(-bound_proj, bound_proj, generator=g)
        ae.q_bias.uniform_(-bound_proj, bound_proj, generator=g)

        # Decoder is not used for correlations, but initialize uniformly for completeness.
        bound_dec = 1.0 / math.sqrt(ae.dict_size)
        ae.decoder.weight.uniform_(-bound_dec, bound_dec, generator=g)
        ae.b_dec.zero_()
        ae.threshold.fill_(-1.0)


def encode_docs_dense_features(
    ae: KronAutoEncoderTopK,
    token_acts_cpu: t.Tensor,
    token_doc_ids_cpu: t.Tensor,
    num_docs: int,
    token_batch_size: int,
    device: str,
) -> t.Tensor:
    """Encode token activations and mean-pool features per document.

    Uses pre-TopK dense feature activations by setting `use_threshold=True` with
    threshold=-1 (default), so no TopK competition is applied.
    """

    ae = ae.to(device)
    ae.eval()

    doc_sums = t.zeros((num_docs, ae.dict_size), dtype=t.float32, device=device)
    doc_counts = t.zeros((num_docs,), dtype=t.float32, device=device)

    n_tokens = token_acts_cpu.shape[0]

    with t.no_grad():
        for start in range(0, n_tokens, token_batch_size):
            end = min(start + token_batch_size, n_tokens)
            x = token_acts_cpu[start:end].to(device=device, dtype=t.float32, non_blocking=True)
            doc_ids = token_doc_ids_cpu[start:end].to(device=device, dtype=t.long, non_blocking=True)

            # Pre-TopK dense features.
            feats = ae.encode(x, use_threshold=True)

            doc_sums.index_add_(0, doc_ids, feats)
            doc_counts.index_add_(0, doc_ids, t.ones_like(doc_ids, dtype=t.float32))

            if (start // token_batch_size) % 20 == 0:
                print(f"[encode] tokens={end}/{n_tokens}", flush=True)

    doc_features = doc_sums / doc_counts.unsqueeze(-1).clamp_min(1.0)
    return doc_features.detach().cpu()


def compute_within_across_correlations(
    doc_features_cpu: t.Tensor,
    h: int,
    m: int,
    n: int,
    device: str,
    eps: float = 1e-8,
    allowed_feature_mask_cpu: t.Tensor | None = None,
) -> CorrelationStats:
    """Compute per-feature mean Pearson correlation within vs across heads."""

    x = doc_features_cpu.to(device=device, dtype=t.float32)
    num_docs, num_features = x.shape

    group_size = m * n
    if h * group_size != num_features:
        raise ValueError(
            f"Feature shape mismatch: h*m*n={h * group_size} but got F={num_features}"
        )

    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=True)
    valid = std > eps
    if allowed_feature_mask_cpu is not None:
        allowed = allowed_feature_mask_cpu.to(device=device, dtype=t.bool)
        if allowed.numel() != num_features:
            raise ValueError(
                f"allowed_feature_mask size mismatch: {allowed.numel()} vs num_features={num_features}"
            )
        valid = valid & allowed

    x = x / std.clamp_min(eps).unsqueeze(0)
    x[:, ~valid] = 0.0

    corr = (x.T @ x) / max(1, num_docs - 1)
    corr = corr.clamp(min=-1.0, max=1.0)

    # Ensure valid feature diagonals are 1, invalid are 0.
    diag = corr.diagonal()
    diag.copy_(t.where(valid, t.ones_like(diag), t.zeros_like(diag)))

    valid_hg = valid.view(h, group_size)
    corr_hghg = corr.view(h, group_size, h, group_size)

    within_means_all: list[t.Tensor] = []
    across_means_all: list[t.Tensor] = []

    for src_h in range(h):
        src_valid = valid_hg[src_h]  # [g]
        if not bool(src_valid.any()):
            continue

        # [g, g] correlations from source-head features to source-head features.
        within_block = corr_hghg[src_h, :, src_h, :]

        target_valid_within = src_valid.to(dtype=t.float32)
        within_weighted_sum = (within_block * target_valid_within.unsqueeze(0)).sum(dim=1)

        self_corr = within_block.diagonal()
        self_valid = src_valid.to(dtype=t.float32)
        within_weighted_sum = within_weighted_sum - (self_corr * self_valid)

        within_denom = target_valid_within.sum() - self_valid
        within_mean = within_weighted_sum / within_denom.clamp_min(1.0)

        # Across-head means use valid targets from all other heads.
        across_target_mask = valid_hg.clone()
        across_target_mask[src_h, :] = False
        across_target_mask_f = across_target_mask.to(dtype=t.float32)

        # [g, h, g]
        src_to_all = corr_hghg[src_h]
        across_weighted_sum = (src_to_all * across_target_mask_f.unsqueeze(0)).sum(dim=(1, 2))
        across_denom = across_target_mask_f.sum()
        across_mean = across_weighted_sum / across_denom.clamp_min(1.0)

        within_means_all.append(within_mean[src_valid])
        across_means_all.append(across_mean[src_valid])

    if not within_means_all:
        raise RuntimeError("No valid features with non-zero variance to compute correlations")

    within_all = t.cat(within_means_all)
    across_all = t.cat(across_means_all)

    mean_within = float(within_all.mean().item())
    mean_across = float(across_all.mean().item())

    return CorrelationStats(
        mean_within_head=mean_within,
        mean_across_head=mean_across,
        mean_delta_within_minus_across=mean_within - mean_across,
        std_within_head=float(within_all.std(unbiased=True).item()),
        std_across_head=float(across_all.std(unbiased=True).item()),
        valid_feature_count=int(within_all.numel()),
        total_feature_count=int(num_features),
    )


def compute_valid_feature_mask(
    doc_features_cpu: t.Tensor,
    eps: float = 1e-8,
) -> t.Tensor:
    std = doc_features_cpu.to(dtype=t.float32).std(dim=0, unbiased=True)
    return std > eps


def sample_head_stratified_features(
    valid_mask_cpu: t.Tensor,
    h: int,
    m: int,
    n: int,
    target_count: int,
    rng: random.Random,
) -> tuple[t.Tensor, list[int], list[int]]:
    """Uniformly sample valid features with approximate head-proportional allocation."""
    group_size = m * n
    if valid_mask_cpu.numel() != h * group_size:
        raise ValueError(
            f"valid_mask size mismatch: {valid_mask_cpu.numel()} vs h*m*n={h * group_size}"
        )

    valid_hg = valid_mask_cpu.view(h, group_size)
    avail_per_head = [int(valid_hg[hh].sum().item()) for hh in range(h)]
    total_avail = int(sum(avail_per_head))
    if target_count > total_avail:
        raise ValueError(
            f"Cannot sample {target_count} features from only {total_avail} valid random features"
        )
    if target_count <= 0:
        raise ValueError(f"target_count must be > 0, got {target_count}")

    raw_targets = [target_count * (a / max(1, total_avail)) for a in avail_per_head]
    base_targets = [min(a, int(math.floor(rt))) for a, rt in zip(avail_per_head, raw_targets)]
    assigned = int(sum(base_targets))
    rem = int(target_count - assigned)

    # Distribute remaining slots by largest fractional remainder, respecting head capacity.
    frac_order = sorted(
        range(h),
        key=lambda hh: (raw_targets[hh] - math.floor(raw_targets[hh])),
        reverse=True,
    )
    while rem > 0:
        progressed = False
        for hh in frac_order:
            if base_targets[hh] < avail_per_head[hh]:
                base_targets[hh] += 1
                rem -= 1
                progressed = True
                if rem == 0:
                    break
        if not progressed:
            raise RuntimeError("Failed to assign remainder in head-stratified sampling")

    selected_mask = t.zeros((h * group_size,), dtype=t.bool)
    for hh in range(h):
        take = base_targets[hh]
        if take <= 0:
            continue
        local_valid = t.nonzero(valid_hg[hh], as_tuple=False).view(-1).tolist()
        chosen_local = rng.sample(local_valid, take)
        for j in chosen_local:
            selected_mask[hh * group_size + int(j)] = True

    if int(selected_mask.sum().item()) != target_count:
        raise RuntimeError(
            f"Selected feature count mismatch: {int(selected_mask.sum().item())} vs target={target_count}"
        )
    return selected_mask, avail_per_head, base_targets


def load_trained_kron(run_dir: Path, checkpoint_name: str, checkpoint_step: int, device: str) -> tuple[KronAutoEncoderTopK, dict[str, Any]]:
    cfg_path = run_dir / "checkpoints" / checkpoint_name / "trainer_config.json"
    ckpt_path = run_dir / "checkpoints" / checkpoint_name / f"ae_step_{checkpoint_step}.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing trainer config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("dict_class") != "KronAutoEncoderTopK":
        raise ValueError(f"Checkpoint {checkpoint_name} is not KronAutoEncoderTopK")

    ae = KronAutoEncoderTopK(
        activation_dim=int(cfg["activation_dim"]),
        h=int(cfg["h"]),
        m=int(cfg["m"]),
        n=int(cfg["n"]),
        k=int(cfg["k"]),
        combine_rule=str(cfg.get("combine_rule", "mand")),
    )
    state = t.load(ckpt_path, map_location="cpu")
    ae.load_state_dict(state)
    ae.to(device)
    ae.eval()
    return ae, cfg


def print_results_table(results: list[EvalResult]) -> None:
    print(
        "checkpoint | mode | mean_within_head | mean_across_head | delta(within-across) | valid_features",
        flush=True,
    )
    print("-" * 116, flush=True)
    for r in results:
        s = r.stats
        print(
            f"{r.checkpoint_name} | {r.mode} | {s.mean_within_head:.6f} | {s.mean_across_head:.6f} | "
            f"{s.mean_delta_within_minus_across:.6f} | {s.valid_feature_count}/{s.total_feature_count}",
            flush=True,
        )


def asdict_stats(s: CorrelationStats) -> dict[str, Any]:
    return {
        "mean_within_head": s.mean_within_head,
        "mean_across_head": s.mean_across_head,
        "mean_delta_within_minus_across": s.mean_delta_within_minus_across,
        "std_within_head": s.std_within_head,
        "std_across_head": s.std_across_head,
        "valid_feature_count": s.valid_feature_count,
        "total_feature_count": s.total_feature_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KronSAE within-head correlation analysis")
    parser.add_argument("--run_dir", type=str, default=str(ROOT / "runs" / "pilot_balanced_v2_420k"))
    parser.add_argument("--dataset_cache_dir", type=str, default=str(ROOT / "data" / "amazon_reviews_full500k"))
    parser.add_argument("--checkpoint_step", type=int, default=420000)
    parser.add_argument(
        "--checkpoint_names",
        nargs="+",
        default=["kron_pilot_topic", "kron_pilot_sentiment"],
    )
    parser.add_argument("--max_docs", type=int, default=5000)
    parser.add_argument("--ctx_len", type=int, default=None)
    parser.add_argument("--doc_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--token_batch_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corr_device", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_meta = load_run_metadata(run_dir)
    train_args = run_meta["args"]

    dataset_cache_dir = Path(args.dataset_cache_dir)
    _, eval_ds, _, _ = load_prepared_datasets(dataset_cache_dir)

    device = args.device or ("cuda:0" if t.cuda.is_available() else "cpu")
    corr_device = args.corr_device or device

    ctx_len = args.ctx_len if args.ctx_len is not None else int(train_args["ctx_len"])

    model = LanguageModel(
        train_args["model_name"],
        dispatch=True,
        device_map=device,
    )
    submodule = model.gpt_neox.layers[int(train_args["layer"])]

    token_acts_cpu, token_doc_ids_cpu, docs_collected = collect_token_activations(
        model=model,
        submodule=submodule,
        eval_ds=eval_ds,
        ctx_len=ctx_len,
        doc_batch_size=args.doc_batch_size,
        num_workers=args.num_workers,
        max_docs=args.max_docs,
        device=device,
    )

    print(
        f"Collected token activations: docs={docs_collected}, tokens={token_acts_cpu.shape[0]}, "
        f"activation_dim={token_acts_cpu.shape[1]}",
        flush=True,
    )

    results: list[EvalResult] = []
    checkpoint_payloads: list[dict[str, Any]] = []

    for idx, checkpoint_name in enumerate(args.checkpoint_names):
        print(f"\n[checkpoint] {checkpoint_name}", flush=True)
        trained_ae, cfg = load_trained_kron(
            run_dir=run_dir,
            checkpoint_name=checkpoint_name,
            checkpoint_step=args.checkpoint_step,
            device=device,
        )

        trained_doc_features = encode_docs_dense_features(
            ae=trained_ae,
            token_acts_cpu=token_acts_cpu,
            token_doc_ids_cpu=token_doc_ids_cpu,
            num_docs=docs_collected,
            token_batch_size=args.token_batch_size,
            device=device,
        )
        trained_stats = compute_within_across_correlations(
            trained_doc_features,
            h=int(cfg["h"]),
            m=int(cfg["m"]),
            n=int(cfg["n"]),
            device=corr_device,
        )
        results.append(EvalResult(checkpoint_name=checkpoint_name, mode="trained", stats=trained_stats))

        random_ae = KronAutoEncoderTopK(
            activation_dim=int(cfg["activation_dim"]),
            h=int(cfg["h"]),
            m=int(cfg["m"]),
            n=int(cfg["n"]),
            k=int(cfg["k"]),
            combine_rule=str(cfg.get("combine_rule", "mand")),
        )
        init_kron_uniform_(random_ae, seed=args.seed + idx)

        random_doc_features = encode_docs_dense_features(
            ae=random_ae,
            token_acts_cpu=token_acts_cpu,
            token_doc_ids_cpu=token_doc_ids_cpu,
            num_docs=docs_collected,
            token_batch_size=args.token_batch_size,
            device=device,
        )
        random_stats = compute_within_across_correlations(
            random_doc_features,
            h=int(cfg["h"]),
            m=int(cfg["m"]),
            n=int(cfg["n"]),
            device=corr_device,
        )
        results.append(EvalResult(checkpoint_name=checkpoint_name, mode="random_uniform", stats=random_stats))

        random_valid_mask = compute_valid_feature_mask(random_doc_features)
        random_valid_count = int(random_valid_mask.sum().item())
        trained_valid_count = int(trained_stats.valid_feature_count)

        rng_match = random.Random(args.seed + 10_000 + idx)
        matched_mask, random_valid_per_head, random_match_per_head = sample_head_stratified_features(
            valid_mask_cpu=random_valid_mask,
            h=int(cfg["h"]),
            m=int(cfg["m"]),
            n=int(cfg["n"]),
            target_count=min(trained_valid_count, random_valid_count),
            rng=rng_match,
        )
        random_matched_stats = compute_within_across_correlations(
            random_doc_features,
            h=int(cfg["h"]),
            m=int(cfg["m"]),
            n=int(cfg["n"]),
            device=corr_device,
            allowed_feature_mask_cpu=matched_mask,
        )
        results.append(
            EvalResult(
                checkpoint_name=checkpoint_name,
                mode="random_uniform_size_matched",
                stats=random_matched_stats,
            )
        )

        checkpoint_payloads.append(
            {
                "checkpoint_name": checkpoint_name,
                "config": {
                    "h": int(cfg["h"]),
                    "m": int(cfg["m"]),
                    "n": int(cfg["n"]),
                    "k": int(cfg["k"]),
                    "activation_dim": int(cfg["activation_dim"]),
                    "combine_rule": str(cfg.get("combine_rule", "mand")),
                },
                "trained": asdict_stats(trained_stats),
                "random_uniform": asdict_stats(random_stats),
                "random_uniform_size_matched": asdict_stats(random_matched_stats),
                "random_valid_feature_count": random_valid_count,
                "random_valid_feature_fraction": float(random_valid_count / max(1, random_stats.total_feature_count)),
                "random_size_match": {
                    "target_feature_count": trained_valid_count,
                    "matched_feature_count": int(random_matched_stats.valid_feature_count),
                    "sampling": "uniform random over valid random features with head-stratified proportional quotas",
                    "valid_features_per_head": random_valid_per_head,
                    "selected_features_per_head": random_match_per_head,
                    "rng_seed": int(args.seed + 10_000 + idx),
                },
                "delta_trained_minus_random": {
                    "mean_within_head": trained_stats.mean_within_head - random_stats.mean_within_head,
                    "mean_across_head": trained_stats.mean_across_head - random_stats.mean_across_head,
                    "mean_delta_within_minus_across": (
                        trained_stats.mean_delta_within_minus_across
                        - random_stats.mean_delta_within_minus_across
                    ),
                },
                "delta_trained_minus_random_size_matched": {
                    "mean_within_head": trained_stats.mean_within_head - random_matched_stats.mean_within_head,
                    "mean_across_head": trained_stats.mean_across_head - random_matched_stats.mean_across_head,
                    "mean_delta_within_minus_across": (
                        trained_stats.mean_delta_within_minus_across
                        - random_matched_stats.mean_delta_within_minus_across
                    ),
                },
            }
        )

    print("", flush=True)
    print_results_table(results)

    payload = {
        "run_dir": str(run_dir),
        "dataset_cache_dir": str(dataset_cache_dir),
        "checkpoint_step": args.checkpoint_step,
        "num_docs": docs_collected,
        "num_tokens": int(token_acts_cpu.shape[0]),
        "feature_mode": "dense_pre_topk_via_use_threshold",
        "random_baseline": "KronAutoEncoderTopK initialized with uniform weights/biases",
        "random_size_matching": "head-stratified proportional subsample from valid random features to trained valid_feature_count",
        "results": checkpoint_payloads,
    }

    out_path = Path(args.output_json) if args.output_json else (run_dir / "head_correlation_results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results JSON: {out_path}", flush=True)


if __name__ == "__main__":
    main()
