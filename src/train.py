"""DDP training entrypoint for flat SAE and KronSAE compositional experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch as t
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add local packages to path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from data.amazon_reviews import (  # noqa: E402
    BalancedJointLabelBatchSampler,
    BalancedLabelBatchSampler,
    PREFERRED_CATEGORIES,
    collate_document_batch,
    format_count_table,
    infinite_dataloader_batches,
    load_prepared_datasets,
    prepare_amazon_reviews,
)
from dictionary_learning.trainers.kron_top_k import (  # noqa: E402
    KronTopKTrainer,
    _mean_pool_by_doc,
    dpo_pairwise_preference_loss,
    supervised_contrastive_loss,
    valid_anchor_fraction_for_label,
)
from dictionary_learning.trainers.top_k import TopKTrainer, geometric_median  # noqa: E402
from dictionary_learning.trainers.trainer import (  # noqa: E402
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)

if TYPE_CHECKING:
    from nnsight import LanguageModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


@dataclass
class SyntheticActivationBatch:
    activations: t.Tensor
    token_doc_ids: t.Tensor
    topic_labels: t.Tensor
    sentiment_labels: t.Tensor
    doc_token_counts: t.Tensor
    doc_start_positions: t.Tensor
    # Optional third, alpha-parameterized conjunctive target (see
    # SyntheticJointActivationBuffer's `conjunctive_alpha`). Left as None unless
    # --synthetic_conjunctive is on, so every pre-existing code path is unchanged.
    conjunctive_labels: Optional[t.Tensor] = None
    conjunctive_p_a: Optional[t.Tensor] = None
    conjunctive_p_b: Optional[t.Tensor] = None


class FlatSupervisedTopKTrainer(TopKTrainer):
    """TopKTrainer + explicit supervision, without changing baseline implementation files.

    Default supervision scheme is `split_half`:
    - first half of flat features gets topic SupCon
    - second half gets sentiment SupCon
    """

    def __init__(
        self,
        *args,
        use_supervision: bool = False,
        lambda_sup: float = 0.5,
        lambda_sup_warmup_frac: float = 0.15,
        supcon_temperature: float = 0.1,
        supervision_mode: Literal["supcon", "joint_dpo", "joint_collect_only", "joint_dpo_cross"] = "supcon",
        dpo_beta: float = 5.0,
        flat_sup_mode: str = "split_half",
        label_type: Optional[Literal["topic", "sentiment"]] = None,
        p_region_dim: Optional[int] = None,
        sup_grad_scale: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_supervision = use_supervision
        self.lambda_sup = lambda_sup
        self.lambda_sup_warmup_frac = lambda_sup_warmup_frac
        self.supcon_temperature = supcon_temperature
        self.supervision_mode = supervision_mode
        self.dpo_beta = float(dpo_beta)
        self.flat_sup_mode = flat_sup_mode
        self.label_type = label_type
        self.p_region_dim = p_region_dim
        self.sup_grad_scale = sup_grad_scale
        if self.supervision_mode not in {"supcon", "joint_dpo", "joint_collect_only", "joint_dpo_cross"}:
            raise ValueError(
                "supervision_mode must be one of: 'supcon', 'joint_dpo', 'joint_collect_only', 'joint_dpo_cross'"
            )
        if self.dpo_beta <= 0:
            raise ValueError("dpo_beta must be > 0")
        if self.label_type not in {None, "topic", "sentiment"}:
            raise ValueError("label_type must be one of: None, 'topic', 'sentiment'")
        if not (0.0 < self.sup_grad_scale <= 1.0):
            raise ValueError("sup_grad_scale must be in (0, 1]")

        self.logging_parameters.extend(
            [
                "sup_loss",
                "topic_supcon_loss",
                "sentiment_supcon_loss",
                "collect_supcon_loss",
                "contrast_supcon_loss",
                "effective_lambda_sup",
                "topic_dpo_loss",
                "sentiment_dpo_loss",
                "topic_valid_anchor_frac",
                "sentiment_valid_anchor_frac",
                "flat_collect_sentiment_dpo_loss",
                "flat_collect_topic_dpo_loss",
            ]
        )
        self.sup_loss = 0.0
        self.topic_supcon_loss = 0.0
        self.sentiment_supcon_loss = 0.0
        self.collect_supcon_loss = 0.0
        self.contrast_supcon_loss = 0.0
        self.effective_lambda_sup = 0.0
        self.topic_dpo_loss = 0.0
        self.sentiment_dpo_loss = 0.0
        self.topic_valid_anchor_frac = 0.0
        self.sentiment_valid_anchor_frac = 0.0
        self.flat_collect_sentiment_dpo_loss = 0.0
        self.flat_collect_topic_dpo_loss = 0.0

    def _unpack_batch(self, batch: Any):
        if isinstance(batch, t.Tensor):
            return batch, None, None, None
        if hasattr(batch, "activations"):
            return batch.activations, batch.token_doc_ids, batch.topic_labels, batch.sentiment_labels
        if isinstance(batch, dict):
            return (
                batch["activations"],
                batch.get("token_doc_ids"),
                batch.get("topic_labels"),
                batch.get("sentiment_labels"),
            )
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _mean_pool_by_doc(self, token_features_BF: t.Tensor, doc_ids_B: t.Tensor, num_docs: int) -> t.Tensor:
        pooled_DF = t.zeros(
            (num_docs, token_features_BF.shape[-1]),
            device=token_features_BF.device,
            dtype=token_features_BF.dtype,
        )
        pooled_DF.index_add_(0, doc_ids_B, token_features_BF)

        counts_D = t.bincount(doc_ids_B, minlength=num_docs)
        counts_D = counts_D.to(dtype=token_features_BF.dtype).unsqueeze(-1)
        return pooled_DF / counts_D.clamp_min(1.0)

    def _supervision_loss(
        self,
        post_relu_acts_BF: t.Tensor,
        token_doc_ids_B: Optional[t.Tensor],
        topic_labels_D: Optional[t.Tensor],
        sentiment_labels_D: Optional[t.Tensor],
    ) -> t.Tensor:
        def _reset_supervision_stats() -> None:
            self.sup_loss = 0.0
            self.topic_supcon_loss = 0.0
            self.sentiment_supcon_loss = 0.0
            self.collect_supcon_loss = 0.0
            self.contrast_supcon_loss = 0.0
            self.topic_dpo_loss = 0.0
            self.sentiment_dpo_loss = 0.0
            self.topic_valid_anchor_frac = 0.0
            self.sentiment_valid_anchor_frac = 0.0
            self.flat_collect_sentiment_dpo_loss = 0.0
            self.flat_collect_topic_dpo_loss = 0.0

        if not self.use_supervision:
            _reset_supervision_stats()
            return t.zeros((), device=post_relu_acts_BF.device, dtype=post_relu_acts_BF.dtype)

        # Flat stabilization: keep SupCon values from post-ReLU features, but scale only
        # the supervision gradient path to reduce abrupt interference with TopK competition.
        sup_features = post_relu_acts_BF.detach() + self.sup_grad_scale * (
            post_relu_acts_BF - post_relu_acts_BF.detach()
        )

        if token_doc_ids_B is None or topic_labels_D is None or sentiment_labels_D is None:
            _reset_supervision_stats()
            return t.zeros((), device=post_relu_acts_BF.device, dtype=post_relu_acts_BF.dtype)

        token_doc_ids_B = token_doc_ids_B.long()
        topic_labels_D = topic_labels_D.long()
        sentiment_labels_D = sentiment_labels_D.long()
        num_docs = topic_labels_D.shape[0]

        if num_docs < 2:
            _reset_supervision_stats()
            return t.zeros((), device=post_relu_acts_BF.device, dtype=post_relu_acts_BF.dtype)

        if self.supervision_mode in {"joint_collect_only", "joint_dpo_cross"}:
            # For flat, both joint experiments use the same non-contradictory objective:
            # collect sentiment + collect topic on the full undivided feature vector.
            doc_repr = self._mean_pool_by_doc(sup_features, token_doc_ids_B, num_docs)
            sentiment_collect = supervised_contrastive_loss(
                doc_repr,
                sentiment_labels_D,
                temperature=self.supcon_temperature,
                invert=False,
            )
            topic_collect = supervised_contrastive_loss(
                doc_repr,
                topic_labels_D,
                temperature=self.supcon_temperature,
                invert=False,
            )
            sup = 0.5 * (sentiment_collect + topic_collect)
            self.sentiment_supcon_loss = float(sentiment_collect.detach().cpu())
            self.topic_supcon_loss = float(topic_collect.detach().cpu())
            self.collect_supcon_loss = float((0.5 * (sentiment_collect + topic_collect)).detach().cpu())
            self.contrast_supcon_loss = 0.0
            self.sentiment_valid_anchor_frac = valid_anchor_fraction_for_label(sentiment_labels_D, invert=False)
            self.topic_valid_anchor_frac = valid_anchor_fraction_for_label(topic_labels_D, invert=False)
            self.sentiment_dpo_loss = 0.0
            self.topic_dpo_loss = 0.0
            self.flat_collect_sentiment_dpo_loss = self.sentiment_supcon_loss
            self.flat_collect_topic_dpo_loss = self.topic_supcon_loss
            self.sup_loss = float(sup.detach().cpu())
            return sup
        elif self.supervision_mode == "joint_dpo":
            doc_repr = self._mean_pool_by_doc(sup_features, token_doc_ids_B, num_docs)
            sentiment_loss, sentiment_valid = dpo_pairwise_preference_loss(
                doc_repr,
                sentiment_labels_D,
                beta=self.dpo_beta,
                invert=False,
            )
            topic_loss, topic_valid = dpo_pairwise_preference_loss(
                doc_repr,
                topic_labels_D,
                beta=self.dpo_beta,
                invert=False,
            )
            sup = 0.5 * (sentiment_loss + topic_loss)
            self.sentiment_dpo_loss = float(sentiment_loss.detach().cpu())
            self.topic_dpo_loss = float(topic_loss.detach().cpu())
            self.sentiment_valid_anchor_frac = float(sentiment_valid)
            self.topic_valid_anchor_frac = float(topic_valid)
            self.sentiment_supcon_loss = self.sentiment_dpo_loss
            self.topic_supcon_loss = self.topic_dpo_loss
            self.collect_supcon_loss = self.sentiment_dpo_loss
            self.contrast_supcon_loss = self.topic_dpo_loss
            self.flat_collect_sentiment_dpo_loss = 0.0
            self.flat_collect_topic_dpo_loss = 0.0
            self.sup_loss = float(sup.detach().cpu())
            return sup

        self.topic_dpo_loss = 0.0
        self.sentiment_dpo_loss = 0.0
        self.topic_valid_anchor_frac = 0.0
        self.sentiment_valid_anchor_frac = 0.0
        self.flat_collect_sentiment_dpo_loss = 0.0
        self.flat_collect_topic_dpo_loss = 0.0

        if self.label_type is not None and self.flat_sup_mode == "pilot_collect_full":
            label = topic_labels_D if self.label_type == "topic" else sentiment_labels_D
            doc_repr = self._mean_pool_by_doc(sup_features, token_doc_ids_B, num_docs)
            collect_loss = supervised_contrastive_loss(
                doc_repr,
                label,
                temperature=self.supcon_temperature,
                invert=False,
            )
            self.collect_supcon_loss = float(collect_loss.detach().cpu())
            self.contrast_supcon_loss = 0.0
            self.topic_supcon_loss = self.collect_supcon_loss if self.label_type == "topic" else 0.0
            self.sentiment_supcon_loss = self.collect_supcon_loss if self.label_type == "sentiment" else 0.0
            self.topic_valid_anchor_frac = valid_anchor_fraction_for_label(label, invert=False)
            self.sentiment_valid_anchor_frac = 0.0
            self.sup_loss = float(collect_loss.detach().cpu())
            return collect_loss

        if self.p_region_dim is None:
            split_idx = sup_features.shape[-1] // 2
        else:
            split_idx = int(self.p_region_dim)
            split_idx = max(1, min(split_idx, sup_features.shape[-1] - 1))

        p_role_feats = sup_features[:, :split_idx]
        q_role_feats = sup_features[:, split_idx:]
        p_docs = self._mean_pool_by_doc(p_role_feats, token_doc_ids_B, num_docs)
        q_docs = self._mean_pool_by_doc(q_role_feats, token_doc_ids_B, num_docs)

        if self.label_type is not None:
            label = topic_labels_D if self.label_type == "topic" else sentiment_labels_D
            collect_loss = supervised_contrastive_loss(
                p_docs,
                label,
                temperature=self.supcon_temperature,
                invert=False,
            )
            contrast_loss = supervised_contrastive_loss(
                q_docs,
                label,
                temperature=self.supcon_temperature,
                invert=True,
            )
            sup = 0.5 * (collect_loss + contrast_loss)
            self.collect_supcon_loss = float(collect_loss.detach().cpu())
            self.contrast_supcon_loss = float(contrast_loss.detach().cpu())
            self.topic_supcon_loss = self.collect_supcon_loss
            self.sentiment_supcon_loss = self.contrast_supcon_loss
            self.topic_valid_anchor_frac = valid_anchor_fraction_for_label(label, invert=False)
            self.sentiment_valid_anchor_frac = valid_anchor_fraction_for_label(label, invert=True)
            self.sup_loss = float(sup.detach().cpu())
            return sup

        if self.flat_sup_mode == "joint":
            doc_repr = self._mean_pool_by_doc(sup_features, token_doc_ids_B, num_docs)
            joint_labels = topic_labels_D * 2 + sentiment_labels_D
            sup = supervised_contrastive_loss(
                doc_repr,
                joint_labels,
                temperature=self.supcon_temperature,
                invert=False,
            )
            self.topic_supcon_loss = float(sup.detach().cpu())
            self.sentiment_supcon_loss = float(sup.detach().cpu())
            self.collect_supcon_loss = self.topic_supcon_loss
            self.contrast_supcon_loss = self.sentiment_supcon_loss
            self.topic_valid_anchor_frac = valid_anchor_fraction_for_label(joint_labels, invert=False)
            self.sentiment_valid_anchor_frac = 0.0
            self.sup_loss = float(sup.detach().cpu())
            return sup

        # Default mode: split feature space in half for mirrored topic/sentiment supervision.
        topic_loss = supervised_contrastive_loss(
            p_docs,
            topic_labels_D,
            temperature=self.supcon_temperature,
            invert=False,
        )
        sentiment_loss = supervised_contrastive_loss(
            q_docs,
            sentiment_labels_D,
            temperature=self.supcon_temperature,
            invert=False,
        )
        sup = 0.5 * (topic_loss + sentiment_loss)

        self.topic_supcon_loss = float(topic_loss.detach().cpu())
        self.sentiment_supcon_loss = float(sentiment_loss.detach().cpu())
        self.collect_supcon_loss = self.topic_supcon_loss
        self.contrast_supcon_loss = self.sentiment_supcon_loss
        self.topic_valid_anchor_frac = valid_anchor_fraction_for_label(topic_labels_D, invert=False)
        self.sentiment_valid_anchor_frac = valid_anchor_fraction_for_label(sentiment_labels_D, invert=False)
        self.sup_loss = float(sup.detach().cpu())
        return sup

    def _get_effective_lambda_sup(self, step: Optional[int]) -> float:
        if not self.use_supervision:
            return 0.0
        if step is None:
            return self.lambda_sup
        warmup_steps = int(self.lambda_sup_warmup_frac * self.steps)
        if warmup_steps <= 0:
            return self.lambda_sup
        scale = min(1.0, float(step + 1) / float(warmup_steps))
        return self.lambda_sup * scale

    def loss(self, batch: Any, step: Optional[int] = None, logging: bool = False):
        x, token_doc_ids_B, topic_labels_D, sentiment_labels_D = self._unpack_batch(batch)
        x = x.to(self.device)

        f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
            x,
            return_topk=True,
            use_threshold=False,
        )

        if step is not None and step > self.threshold_start_step:
            self.update_threshold(top_acts_BK)

        x_hat = self.ae.decode(f)
        residual = x - x_hat

        self.effective_l0 = top_acts_BK.size(1)

        num_tokens_in_step = x.shape[0]
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = residual.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(residual.detach(), post_relu_acts_BF) if self.auxk_alpha > 0 else 0
        sae_loss = l2_loss + self.auxk_alpha * auxk_loss

        sup_loss = self._supervision_loss(
            post_relu_acts_BF,
            token_doc_ids_B.to(self.device) if isinstance(token_doc_ids_B, t.Tensor) else None,
            topic_labels_D.to(self.device) if isinstance(topic_labels_D, t.Tensor) else None,
            sentiment_labels_D.to(self.device) if isinstance(sentiment_labels_D, t.Tensor) else None,
        )

        effective_lambda_sup = self._get_effective_lambda_sup(step)
        self.effective_lambda_sup = effective_lambda_sup
        total_loss = sae_loss + (effective_lambda_sup * sup_loss)

        if not logging:
            return total_loss

        return {
            "l2_loss": float(l2_loss.detach().cpu()),
            "auxk_loss": float(auxk_loss.detach().cpu()) if isinstance(auxk_loss, t.Tensor) else float(auxk_loss),
            "sae_loss": float(sae_loss.detach().cpu()),
            "sup_loss": float(sup_loss.detach().cpu()),
            "loss": float(total_loss.detach().cpu()),
        }

    def update(self, step: int, batch: Any):
        x, _, _, _ = self._unpack_batch(batch)
        x = x.to(self.device)

        if step == 0:
            median = geometric_median(x)
            self.ae.b_dec.data = median.to(self.ae.b_dec.dtype)

        loss = self.loss(batch, step=step)
        loss.backward()

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)

        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight,
            self.ae.activation_dim,
            self.ae.dict_size,
        )

        return float(loss.detach().cpu())

    @property
    def config(self):
        cfg = super().config
        cfg.update(
            {
                "trainer_class": "FlatSupervisedTopKTrainer",
                "use_supervision": self.use_supervision,
                "lambda_sup": self.lambda_sup,
                "lambda_sup_warmup_frac": self.lambda_sup_warmup_frac,
                "supcon_temperature": self.supcon_temperature,
                "supervision_mode": self.supervision_mode,
                "dpo_beta": self.dpo_beta,
                "flat_sup_mode": self.flat_sup_mode,
                "label_type": self.label_type,
                "p_region_dim": self.p_region_dim,
                "sup_grad_scale": self.sup_grad_scale,
            }
        )
        return cfg


class DistEnv:
    def __init__(self, enabled: bool, rank: int, local_rank: int, world_size: int):
        self.enabled = enabled
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(args) -> DistEnv:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if t.cuda.is_available():
            t.cuda.set_device(local_rank)
            try:
                dist.init_process_group(
                    backend="nccl",
                    device_id=t.device(f"cuda:{local_rank}"),
                )
            except TypeError:
                dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")
        return DistEnv(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)

    return DistEnv(enabled=False, rank=0, local_rank=0, world_size=1)


def dist_barrier(dist_env: DistEnv) -> None:
    if not dist_env.enabled:
        return
    if t.cuda.is_available():
        dist.barrier(device_ids=[dist_env.local_rank])
    else:
        dist.barrier()


def cleanup_distributed(dist_env: DistEnv) -> None:
    if not dist_env.enabled:
        return
    dist_barrier(dist_env)
    dist.destroy_process_group()


class DDPProxy(t.nn.Module):
    """Expose wrapped module attributes while delegating forward/backward to DDP."""

    def __init__(self, module: t.nn.Module, **ddp_kwargs):
        super().__init__()
        self.ddp = DDP(module, **ddp_kwargs)

    def forward(self, *args, **kwargs):
        return self.ddp(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.ddp.module, name)


def unwrap_model(module):
    if isinstance(module, DDPProxy):
        return module.ddp.module
    if isinstance(module, DDP):
        return module.module
    return module


def rank_print(dist_env: DistEnv, msg: str) -> None:
    if dist_env.is_main:
        print(msg, flush=True)


def build_dataloader(train_ds, args, dist_env: DistEnv):
    workers_per_rank = max(1, args.total_cpu_workers // dist_env.world_size)

    sampler = None
    if dist_env.enabled:
        sampler = DistributedSampler(
            train_ds,
            num_replicas=dist_env.world_size,
            rank=dist_env.rank,
            shuffle=True,
            drop_last=False,
        )

    loader = DataLoader(
        train_ds,
        batch_size=args.doc_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=workers_per_rank,
        pin_memory=t.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=4 if workers_per_rank > 0 else None,
        collate_fn=collate_document_batch,
        drop_last=False,
    )

    return loader, sampler, workers_per_rank


def build_pilot_balanced_dataloaders(train_ds, args, dist_env: DistEnv):
    workers_per_rank_total = max(1, args.total_cpu_workers // dist_env.world_size)
    topic_workers = max(1, (workers_per_rank_total + 1) // 2)
    sentiment_workers = max(1, workers_per_rank_total - topic_workers)

    topic_sampler = BalancedLabelBatchSampler(
        dataset=train_ds,
        label_type="topic",
        batch_size=args.doc_batch_size,
        seed=args.seed,
        rank=dist_env.rank,
        world_size=dist_env.world_size,
        drop_last=False,
    )
    sentiment_sampler = BalancedLabelBatchSampler(
        dataset=train_ds,
        label_type="sentiment",
        batch_size=args.doc_batch_size,
        seed=args.seed + 17,
        rank=dist_env.rank,
        world_size=dist_env.world_size,
        drop_last=False,
    )

    topic_loader = DataLoader(
        train_ds,
        batch_sampler=topic_sampler,
        num_workers=topic_workers,
        pin_memory=t.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=4 if topic_workers > 0 else None,
        collate_fn=collate_document_batch,
    )
    sentiment_loader = DataLoader(
        train_ds,
        batch_sampler=sentiment_sampler,
        num_workers=sentiment_workers,
        pin_memory=t.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=4 if sentiment_workers > 0 else None,
        collate_fn=collate_document_batch,
    )

    return (
        {"topic": topic_loader, "sentiment": sentiment_loader},
        {"topic": topic_sampler, "sentiment": sentiment_sampler},
        {"topic": topic_workers, "sentiment": sentiment_workers},
    )


def build_joint_balanced_dataloader(train_ds, args, dist_env: DistEnv):
    workers_per_rank = max(1, args.total_cpu_workers // dist_env.world_size)
    sampler = BalancedJointLabelBatchSampler(
        dataset=train_ds,
        batch_size=args.doc_batch_size,
        seed=args.seed,
        rank=dist_env.rank,
        world_size=dist_env.world_size,
        drop_last=False,
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=workers_per_rank,
        pin_memory=t.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=4 if workers_per_rank > 0 else None,
        collate_fn=collate_document_batch,
    )
    return loader, sampler, workers_per_rank


def batch_label_histogram(batch: Any) -> dict[str, dict[str, int]]:
    topic = batch.topic_labels.detach().cpu().tolist()
    sentiment = batch.sentiment_labels.detach().cpu().tolist()
    topic_hist: dict[str, int] = {}
    sentiment_hist: dict[str, int] = {"0": 0, "1": 0}
    for v in topic:
        key = str(int(v))
        topic_hist[key] = topic_hist.get(key, 0) + 1
    for v in sentiment:
        key = str(int(v))
        sentiment_hist[key] = sentiment_hist.get(key, 0) + 1
    return {"topic": topic_hist, "sentiment": sentiment_hist}


class SyntheticJointActivationBuffer:
    """Infinite synthetic batch stream with orthogonal sentiment/topic subspaces.

    Optionally (``conjunctive_alpha is not None``) also emits a third,
    alpha-parameterized binary target ``y_alpha`` built from two *new*
    orthonormal primitive directions ``d_A``/``d_B`` carved out of the same QR
    decomposition, disjoint from the sentiment and topic blocks::

        p_A = d_A . x,  p_B = d_B . x
        score  = (1 - alpha) * p_A  +  alpha * (p_A * p_B / s)
        y_alpha = 1[score - offset > 0]

    ``s`` rescales the product term so that it carries the *same* scale as the
    purely additive term (``std(p_A p_B) / std(p_A)``), measured empirically on a
    calibration sample so alpha is the only thing that changes across the sweep
    and not the effective signal magnitude. ``offset`` (enabled by
    ``conjunctive_balance``) is the calibration-sample median of ``score``, which
    holds the marginal frequency of ``y_alpha`` at ~0.5 at every alpha — the
    confound control for "did the label just get rarer/easier?".

    Why this exists (Part A of the conjunctive-structure RQ): alpha interpolates
    the ground-truth target from purely *linear* in the primitives (alpha=0,
    ``y = sign(p_A)``) to purely *conjunctive*/XOR-like (alpha=1,
    ``y = sign(p_A) XOR sign(p_B)``, which no linear function of (p_A, p_B) can
    beat chance on). It is the causal knob for testing whether KronSAE's
    bilinear encoder beats a flat SAE *because of* conjunctive ground-truth
    structure rather than because of anything else about the data.

    The sentiment/topic blocks are always generated exactly as before, so they
    remain present as background signal and every existing run reproduces
    bit-for-bit when ``conjunctive_alpha is None``.
    """

    def __init__(
        self,
        activation_dim: int,
        doc_batch_size: int,
        num_topics: int,
        num_sentiments: int,
        topic_signal_scale: float,
        sentiment_signal_scale: float,
        noise_std: float,
        seed: int,
        device: str,
        conjunctive_alpha: Optional[float] = None,
        conjunctive_signal_scale: float = 1.0,
        conjunctive_balance: bool = True,
        conjunctive_calib_batches: int = 16,
    ):
        if activation_dim <= (num_topics + num_sentiments):
            raise ValueError("activation_dim must exceed num_topics + num_sentiments")
        if doc_batch_size < (num_topics * num_sentiments):
            raise ValueError("doc_batch_size must be at least num_topics * num_sentiments for balanced synthetic batches")
        if num_sentiments < 2:
            raise ValueError("num_sentiments must be >=2")
        self.activation_dim = activation_dim
        self.doc_batch_size = doc_batch_size
        self.num_topics = num_topics
        self.num_sentiments = num_sentiments
        self.topic_signal_scale = float(topic_signal_scale)
        self.sentiment_signal_scale = float(sentiment_signal_scale)
        self.noise_std = float(noise_std)
        self.device = device
        self.generator = t.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.conjunctive_alpha = None if conjunctive_alpha is None else float(conjunctive_alpha)
        self.conjunctive_signal_scale = float(conjunctive_signal_scale)
        self.conjunctive_balance = bool(conjunctive_balance)
        self.conjunctive_calib_batches = int(conjunctive_calib_batches)

        q, _ = t.linalg.qr(t.randn(activation_dim, activation_dim, generator=self.generator))
        self.sentiment_dirs = q[:, :num_sentiments].T.contiguous()
        self.topic_dirs = q[:, num_sentiments : num_sentiments + num_topics].T.contiguous()
        block_end = num_sentiments + num_topics
        if self.conjunctive_alpha is None:
            self.conjunctive_dirs: Optional[t.Tensor] = None
            self.noise_basis = q[:, block_end:].contiguous()
        else:
            if not (0.0 <= self.conjunctive_alpha <= 1.0):
                raise ValueError("conjunctive_alpha must lie in [0, 1]")
            if activation_dim <= (block_end + 2):
                raise ValueError("activation_dim must exceed num_topics + num_sentiments + 2 for the conjunctive block")
            # Two more reserved columns of the SAME orthonormal basis, so d_A/d_B
            # are exactly orthogonal to every sentiment/topic direction and to the
            # noise basis; p_A/p_B therefore read out the injected coefficients
            # exactly, with no contamination from the background blocks.
            self.conjunctive_dirs = q[:, block_end : block_end + 2].T.contiguous()
            self.noise_basis = q[:, block_end + 2 :].contiguous()
        self._conjunctive_product_scale = 1.0
        self._conjunctive_offset = 0.0
        self._conjunctive_calibration: dict[str, float] = {}
        if self.conjunctive_dirs is not None:
            self._calibrate_conjunctive(seed=seed + 7919)

        combos = []
        for topic in range(num_topics):
            for sentiment in range(num_sentiments):
                combos.append((topic, sentiment))
        self.combos = t.tensor(combos, dtype=t.long)

    def direction_diagnostics(self) -> dict[str, float]:
        sent_topic_cos = self.sentiment_dirs @ self.topic_dirs.T
        sent_intra = self.sentiment_dirs @ self.sentiment_dirs.T
        topic_intra = self.topic_dirs @ self.topic_dirs.T
        sent_eye = t.eye(self.num_sentiments, dtype=sent_intra.dtype)
        topic_eye = t.eye(self.num_topics, dtype=topic_intra.dtype)
        return {
            "max_abs_sent_topic_cos": float(sent_topic_cos.abs().max().item()),
            "max_abs_sent_self_offdiag": float((sent_intra - sent_eye).abs().max().item()),
            "max_abs_topic_self_offdiag": float((topic_intra - topic_eye).abs().max().item()),
        }

    def _draw_conjunctive_coeffs(self, n: int, generator: t.Generator) -> t.Tensor:
        """Draw (n, 2) i.i.d. N(0, conjunctive_signal_scale^2) coefficients for d_A/d_B.

        Because d_A/d_B are orthonormal and disjoint from every other block, the
        drawn coefficients ARE p_A/p_B once the signal is added to x, so the
        label can be defined on the true primitives with no estimation error.
        """
        return t.randn(n, 2, generator=generator) * self.conjunctive_signal_scale

    def _conjunctive_score(self, p_a: t.Tensor, p_b: t.Tensor) -> t.Tensor:
        """score = (1-alpha) * p_A + alpha * (p_A * p_B / s), pre-threshold."""
        alpha = float(self.conjunctive_alpha)
        return (1.0 - alpha) * p_a + alpha * (p_a * p_b / self._conjunctive_product_scale)

    def _calibrate_conjunctive(self, seed: int) -> None:
        """Fit `s` (product rescale) and `offset` (marginal balance) on a sample batch.

        `s = std(p_A p_B) / std(p_A)` puts the product term on the same scale as
        the additive term, so sweeping alpha changes the *structure* of the
        target and not its effective signal-to-noise. `offset` is the median of
        the resulting score, which is what holds P(y=1) at ~0.5 across alpha.

        Uses a dedicated generator so calibration never perturbs the main
        activation stream (the stream stays reproducible from `seed` alone).
        """
        g = t.Generator(device="cpu")
        g.manual_seed(seed)
        n = max(1, self.conjunctive_calib_batches) * self.doc_batch_size
        coeffs = self._draw_conjunctive_coeffs(n, g)
        p_a, p_b = coeffs[:, 0], coeffs[:, 1]

        prod_std = float((p_a * p_b).std().item())
        a_std = float(p_a.std().item())
        self._conjunctive_product_scale = (prod_std / a_std) if a_std > 0 and prod_std > 0 else 1.0

        raw_score = self._conjunctive_score(p_a, p_b)
        self._conjunctive_offset = float(raw_score.median().item()) if self.conjunctive_balance else 0.0

        y = (raw_score - self._conjunctive_offset > 0).long()
        y_sign = y.float() * 2.0 - 1.0
        # Two reference "ceilings" that make the alpha knob interpretable:
        #   linear_rule_acc      -- accuracy of the best purely additive rule sign(p_A)
        #   conjunctive_rule_acc -- accuracy of the pure product rule sign(p_A * p_B)
        # At alpha=0 the first is 1.0 and the second ~0.5; at alpha=1 they swap.
        self._conjunctive_calibration = {
            "n_calibration_samples": int(n),
            "product_scale_s": self._conjunctive_product_scale,
            "score_offset": self._conjunctive_offset,
            "marginal_pos_frac": float(y.float().mean().item()),
            "linear_rule_acc": float(((p_a > 0).long() == y).float().mean().item()),
            "conjunctive_rule_acc": float((((p_a * p_b) > 0).long() == y).float().mean().item()),
            "corr_p_a_y": float(t.corrcoef(t.stack([p_a, y_sign]))[0, 1].item()),
            "corr_p_a_p_b_y": float(t.corrcoef(t.stack([p_a * p_b, y_sign]))[0, 1].item()),
        }

    def conjunctive_diagnostics(self) -> dict[str, float]:
        """Report the conjunctive block's calibration + orthogonality to the background blocks.

        Serves the Part A confound controls: `marginal_pos_frac` shows the label
        stayed balanced across the alpha sweep, and the cosine terms show d_A/d_B
        did not accidentally re-encode sentiment or topic.
        """
        if self.conjunctive_dirs is None:
            return {}
        sent_cos = self.conjunctive_dirs @ self.sentiment_dirs.T
        topic_cos = self.conjunctive_dirs @ self.topic_dirs.T
        ab_cos = self.conjunctive_dirs[0] @ self.conjunctive_dirs[1]
        out: dict[str, float] = {
            "conjunctive_alpha": float(self.conjunctive_alpha),
            "conjunctive_signal_scale": self.conjunctive_signal_scale,
            "conjunctive_balance": float(self.conjunctive_balance),
            "max_abs_conj_sent_cos": float(sent_cos.abs().max().item()),
            "max_abs_conj_topic_cos": float(topic_cos.abs().max().item()),
            "abs_cos_d_a_d_b": float(ab_cos.abs().item()),
        }
        out.update(self._conjunctive_calibration)
        return out

    def __iter__(self):
        return self

    def __next__(self) -> SyntheticActivationBatch:
        reps = int(np.ceil(self.doc_batch_size / self.combos.shape[0]))
        combo_idx = self.combos.repeat((reps, 1))[: self.doc_batch_size]
        perm = t.randperm(combo_idx.shape[0], generator=self.generator)
        combo_idx = combo_idx[perm]
        topic_labels = combo_idx[:, 0]
        sentiment_labels = combo_idx[:, 1]

        topic_signal = self.topic_dirs[topic_labels] * self.topic_signal_scale
        sentiment_signal = self.sentiment_dirs[sentiment_labels] * self.sentiment_signal_scale

        noise_coeff = t.randn(
            self.doc_batch_size,
            self.noise_basis.shape[1],
            generator=self.generator,
        ) * self.noise_std
        noise = noise_coeff @ self.noise_basis.T

        activations = topic_signal + sentiment_signal + noise

        conj_labels: Optional[t.Tensor] = None
        conj_p_a: Optional[t.Tensor] = None
        conj_p_b: Optional[t.Tensor] = None
        if self.conjunctive_dirs is not None:
            coeffs = self._draw_conjunctive_coeffs(self.doc_batch_size, self.generator)
            conj_p_a, conj_p_b = coeffs[:, 0].contiguous(), coeffs[:, 1].contiguous()
            activations = activations + coeffs @ self.conjunctive_dirs
            score = self._conjunctive_score(conj_p_a, conj_p_b)
            conj_labels = (score - self._conjunctive_offset > 0).long()

        token_doc_ids = t.arange(self.doc_batch_size, dtype=t.long)
        doc_token_counts = t.ones(self.doc_batch_size, dtype=t.long)
        doc_start_positions = t.arange(self.doc_batch_size, dtype=t.long)
        return SyntheticActivationBatch(
            activations=activations.to(self.device),
            token_doc_ids=token_doc_ids.to(self.device),
            topic_labels=topic_labels.to(self.device),
            sentiment_labels=sentiment_labels.to(self.device),
            doc_token_counts=doc_token_counts.to(self.device),
            doc_start_positions=doc_start_positions.to(self.device),
            conjunctive_labels=None if conj_labels is None else conj_labels.to(self.device),
            conjunctive_p_a=None if conj_p_a is None else conj_p_a.to(self.device),
            conjunctive_p_b=None if conj_p_b is None else conj_p_b.to(self.device),
        )


def _stratified_split_indices(labels: t.Tensor, test_fraction: float, seed: int) -> tuple[t.Tensor, t.Tensor]:
    g = t.Generator(device="cpu")
    g.manual_seed(seed)
    train_idx = []
    test_idx = []
    for cls in t.unique(labels, sorted=True):
        idx = (labels == cls).nonzero(as_tuple=False).squeeze(-1).cpu()
        perm = idx[t.randperm(idx.numel(), generator=g)]
        n_test = max(1, int(round(idx.numel() * test_fraction))) if idx.numel() > 1 else 0
        n_test = min(n_test, max(0, idx.numel() - 1))
        test_idx.append(perm[:n_test])
        train_idx.append(perm[n_test:])
    return t.cat(train_idx), t.cat(test_idx)


def _train_linear_probe(
    X_train: t.Tensor,
    y_train: t.Tensor,
    X_test: t.Tensor,
    y_test: t.Tensor,
    num_classes: int,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
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
        return float((pred == y_test).float().mean().item())


def evaluate_synthetic_joint_probe(
    trainer: KronTopKTrainer,
    batch_source: SyntheticJointActivationBuffer,
    num_batches: int,
    seed: int,
    probe_epochs: int,
    probe_lr: float,
    probe_batch_size: int,
    device: str,
) -> dict[str, float]:
    ae = unwrap_model(trainer.ae)
    was_training = ae.training
    ae.eval()

    p_docs_all = []
    q_docs_all = []
    topic_all = []
    sentiment_all = []

    with t.no_grad():
        for _ in range(num_batches):
            batch = next(batch_source)
            x = batch.activations.to(device=device, dtype=t.float32)
            token_doc_ids = batch.token_doc_ids.long().to(device)
            topic_labels = batch.topic_labels.long().cpu()
            sentiment_labels = batch.sentiment_labels.long().cpu()

            _, p_pos, q_pos = ae.encode(x, return_branches=True)
            # `--q_bottleneck_rank` (REPORT1.md idea 4) is a capacity constraint on Q, so the
            # probe has to read the post-bottleneck representation -- the pre-bottleneck
            # `q_pos` is the unrestricted encoder output and would not measure the mechanism.
            # A no-op when the bottleneck is disabled.
            q_pos = trainer.apply_q_bottleneck(q_pos)
            p_token = p_pos.reshape(p_pos.shape[0], -1)
            q_token = q_pos.reshape(q_pos.shape[0], -1)
            p_docs = _mean_pool_by_doc(p_token, token_doc_ids, topic_labels.shape[0]).detach().cpu()
            q_docs = _mean_pool_by_doc(q_token, token_doc_ids, topic_labels.shape[0]).detach().cpu()

            p_docs_all.append(p_docs)
            q_docs_all.append(q_docs)
            topic_all.append(topic_labels)
            sentiment_all.append(sentiment_labels)

    if was_training:
        ae.train()

    p = t.cat(p_docs_all, dim=0)
    q = t.cat(q_docs_all, dim=0)
    topic = t.cat(topic_all, dim=0)
    sentiment = t.cat(sentiment_all, dim=0)

    sent_train, sent_test = _stratified_split_indices(sentiment, test_fraction=0.3, seed=seed)
    topic_train, topic_test = _stratified_split_indices(topic, test_fraction=0.3, seed=seed + 1)

    def _probe(
        feats: t.Tensor,
        labels: t.Tensor,
        train_idx: t.Tensor,
        test_idx: t.Tensor,
        classes: int,
        probe_seed: int,
    ) -> float:
        return _train_linear_probe(
            X_train=feats[train_idx],
            y_train=labels[train_idx],
            X_test=feats[test_idx],
            y_test=labels[test_idx],
            num_classes=classes,
            seed=probe_seed,
            epochs=probe_epochs,
            lr=probe_lr,
            batch_size=probe_batch_size,
            device=device,
        )

    return {
        "p_sentiment_acc": _probe(p, sentiment, sent_train, sent_test, 2, seed + 10),
        "q_sentiment_acc": _probe(q, sentiment, sent_train, sent_test, 2, seed + 11),
        "p_topic_acc": _probe(p, topic, topic_train, topic_test, int(topic.max().item()) + 1, seed + 12),
        "q_topic_acc": _probe(q, topic, topic_train, topic_test, int(topic.max().item()) + 1, seed + 13),
        "sentiment_chance": 0.5,
        "topic_chance": 1.0 / float(int(topic.max().item()) + 1),
    }


def standardized_logreg_accuracy(
    X_train: t.Tensor,
    y_train: t.Tensor,
    X_test: t.Tensor,
    y_test: t.Tensor,
) -> tuple[float, bool]:
    """Train-z-scored `LogisticRegression(max_iter=5000, C=1.0)` accuracy.

    This is the corrected probe standard mandated by REPORT.md §9.3: the older
    unstandardized fixed-LR Adam probe never leaves the constant-predictor
    solution on branches with large per-dim scale, which manufactured the
    spurious "exactly chance" readings that §9.3 retracted. Standardization uses
    TRAIN-split statistics only, so no test information leaks.

    Returns (accuracy, converged) where `converged` is False if the solver hit
    max_iter -- a run with converged=False must not be read as a real result.
    """
    from sklearn.linear_model import LogisticRegression

    mu = X_train.mean(dim=0, keepdim=True)
    sd = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_tr = ((X_train - mu) / sd).cpu().numpy()
    x_te = ((X_test - mu) / sd).cpu().numpy()
    y_tr = y_train.cpu().numpy()
    y_te = y_test.cpu().numpy()

    if np.unique(y_tr).size < 2:
        return float((y_te == y_tr[0]).mean()), True

    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(x_tr, y_tr)
    return float((clf.predict(x_te) == y_te).mean()), bool(int(np.max(clf.n_iter_)) < 5000)


def _participation_ratio_from_cov(x: t.Tensor, eps: float = 1e-12) -> float:
    """Effective dimensionality of `x`'s covariance: (sum eig)^2 / sum(eig^2). Identical
    formula to `eval_representation_geometry.participation_ratio_from_cov` (Part I sec.9.1),
    reimplemented here to avoid importing that real-data-oriented module's dependencies."""
    x = x.float()
    if x.shape[0] < 2:
        return 0.0
    xc = x - x.mean(dim=0, keepdim=True)
    # Double precision + a small diagonal jitter: `eigvalsh` on a near-singular or
    # many-repeated-eigenvalue covariance (common for a longer-trained, more specialized
    # dictionary, where many features are dead or highly correlated) can fail to converge in
    # float32 (torch._C._LinAlgError). Both are standard, harmless fixes for this exact failure
    # mode -- jitter shifts eigenvalues by a fixed, tiny amount (negligible next to the participation
    # ratio's scale), double precision gives the iterative solver more room to converge.
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    cov = cov.double()
    jitter = eps * t.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    try:
        eig = t.linalg.eigvalsh(cov + jitter).clamp_min(0.0)
    except t._C._LinAlgError:
        eig = t.linalg.eigvalsh((cov + jitter).cpu()).clamp_min(0.0).to(cov.device)
    num = eig.sum() ** 2
    den = (eig**2).sum() + eps
    return float((num / den).item())


def _discriminative_direction(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """Per-dimension squared between-class mean deviation, summed over classes -- a label's
    "direction" in `x`'s coordinate frame without fitting a classifier. Identical construction
    to `eval_representation_geometry.discriminative_energy_vector` (Part I sec.9.1's
    `cos(sentiment-dir, topic-dir)` diagnostic used exactly this)."""
    x = x.float()
    y = y.long()
    mu = x.mean(dim=0, keepdim=True)
    e = t.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    for cls in t.unique(y):
        idx = (y == cls).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        diff = (x[idx].mean(dim=0, keepdim=True) - mu).squeeze(0)
        e = e + diff * diff
    return e


def evaluate_synthetic_representation_geometry(
    trainer: KronTopKTrainer,
    batch_source: SyntheticJointActivationBuffer,
    num_batches: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Cross-check + geometry diagnostics for the REPORT1.md sec.19 idea sweep, addressing two
    gaps flagged when the user asked about participation ratio and the "pairwise interference"
    score from Elhage et al. 2022 (Toy Models of Superposition, `sum_{j!=i} (What_i . W_j)^2`):

    1. `evaluate_synthetic_joint_probe`'s reported accuracies come from an UNSTANDARDIZED
       `torch.nn.Linear` trained by Adam -- not `standardized_logreg_accuracy`, the function
       this file documents as "the corrected probe standard mandated by REPORT.md sec.9.3."
       This computes the standardized-logreg number on the same representations so the two can
       be compared directly.
    2. Participation ratio (`_participation_ratio_from_cov`) on P's and Q's raw pooled
       representations, and a pairwise interference score between the sentiment- and
       topic-discriminative directions within each branch: `(What_sent . What_topic)^2` after
       normalizing both directions to unit norm (0 = orthogonal, i.e. a spare direction exists
       to separate the two labels; >0 = the same direction partly carries both signals, an
       accessible superposition in Elhage et al.'s sense, not necessarily a probe artifact).
    """
    ae = unwrap_model(trainer.ae)
    was_training = ae.training
    ae.eval()

    p_docs_all, q_docs_all, topic_all, sentiment_all = [], [], [], []
    with t.no_grad():
        for _ in range(num_batches):
            batch = next(batch_source)
            x = batch.activations.to(device=device, dtype=t.float32)
            token_doc_ids = batch.token_doc_ids.long().to(device)
            topic_labels = batch.topic_labels.long().cpu()
            sentiment_labels = batch.sentiment_labels.long().cpu()

            _, p_pos, q_pos = ae.encode(x, return_branches=True)
            q_pos = trainer.apply_q_bottleneck(q_pos)
            p_docs = _mean_pool_by_doc(p_pos.reshape(p_pos.shape[0], -1), token_doc_ids, topic_labels.shape[0]).detach().cpu()
            q_docs = _mean_pool_by_doc(q_pos.reshape(q_pos.shape[0], -1), token_doc_ids, topic_labels.shape[0]).detach().cpu()
            p_docs_all.append(p_docs)
            q_docs_all.append(q_docs)
            topic_all.append(topic_labels)
            sentiment_all.append(sentiment_labels)
    if was_training:
        ae.train()

    p, q = t.cat(p_docs_all, dim=0), t.cat(q_docs_all, dim=0)
    topic, sentiment = t.cat(topic_all, dim=0), t.cat(sentiment_all, dim=0)
    sent_train, sent_test = _stratified_split_indices(sentiment, test_fraction=0.3, seed=seed)
    topic_train, topic_test = _stratified_split_indices(topic, test_fraction=0.3, seed=seed + 1)

    out: dict[str, Any] = {}
    for branch_name, feats in (("p", p), ("q", q)):
        sent_acc, sent_converged = standardized_logreg_accuracy(
            feats[sent_train], sentiment[sent_train], feats[sent_test], sentiment[sent_test]
        )
        top_acc, top_converged = standardized_logreg_accuracy(
            feats[topic_train], topic[topic_train], feats[topic_test], topic[topic_test]
        )
        out[f"{branch_name}_sentiment_acc_std"] = sent_acc
        out[f"{branch_name}_sentiment_acc_std_converged"] = sent_converged
        out[f"{branch_name}_topic_acc_std"] = top_acc
        out[f"{branch_name}_topic_acc_std_converged"] = top_converged
        out[f"{branch_name}_participation_ratio"] = _participation_ratio_from_cov(feats)

        w_sent = _discriminative_direction(feats, sentiment)
        w_topic = _discriminative_direction(feats, topic)
        w_sent_hat = w_sent / w_sent.norm().clamp_min(1e-12)
        w_topic_hat = w_topic / w_topic.norm().clamp_min(1e-12)
        out[f"{branch_name}_sent_topic_interference"] = float((w_sent_hat @ w_topic_hat) ** 2)
        out[f"{branch_name}_sent_dir_norm"] = float(w_sent.norm())
        out[f"{branch_name}_topic_dir_norm"] = float(w_topic.norm())
    return out


def _signed_binary_direction(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """mean(class 1) - mean(class 0): the correctly-SIGNED direction for a 2-class label.

    `_discriminative_direction`'s squared/summed construction is built to generalize to
    topic's 6 classes for the interference/PR diagnostics (sec.19.1), where no single signed
    direction exists -- but for sentiment specifically it discards sign (a per-dimension
    squared deviation is always >= 0), so it is NOT guaranteed to point toward "more positive"
    consistently across dimensions. A causal steering intervention needs an oriented push, so
    this binary-only, explicitly-signed direction is what `evaluate_synthetic_intervention`
    actually steers along -- the unsigned energy vector stays reserved for interference/PR.
    """
    x = x.float()
    y = y.long()
    return x[y == 1].mean(dim=0) - x[y == 0].mean(dim=0)


def evaluate_synthetic_intervention(
    trainer: KronTopKTrainer,
    batch_source: SyntheticJointActivationBuffer,
    num_batches: int,
    seed: int,
    device: str,
    alphas: list[float],
    branch: str = "q",
) -> dict[str, Any]:
    """Patch-and-generate causal intervention eval (REPORT_FULL.md sec.19.3-19.5), testing
    steerability rather than erasure -- the property probe accuracy / participation ratio /
    direction interference (sec.19-19.2) cannot speak to. Fits the target branch's sentiment
    direction on a FIT split (signed `_signed_binary_direction`, sec.19.3's correction over
    sec.19.1's unsigned energy construction), then on a disjoint EVAL split: pushes each doc's
    per-token TARGET-branch representation by `alpha` standard deviations (of the FIT split's
    projection onto that direction) along the direction, recombines with the UNTOUCHED other
    branch, decodes, and scores the resulting RECONSTRUCTION -- not the branch itself -- with
    two classifiers trained on genuine raw activations from a disjoint reference split (never
    touched by the SAE at all). This is the causal test: does pushing the branch's axis
    actually move what the model outputs, read by an honest external judge, while leaving
    topic (read by the same kind of external judge) alone.

    `branch`: "q" (default, sec.19.3-19.4 -- Q is trained to SUPPRESS sentiment) or "p"
    (sec.19.5 -- P is trained to COLLECT sentiment; comparing the two tells you whether
    steerability is a property of "the branch actually assigned this label" specifically, or
    present throughout the representation regardless of which branch nominally owns it).
    """
    if branch not in ("p", "q"):
        raise ValueError(f"branch must be 'p' or 'q', got {branch!r}")
    ae = unwrap_model(trainer.ae)
    was_training = ae.training
    ae.eval()

    x_all, doc_ids_all, topic_all, sentiment_all = [], [], [], []
    n_docs_seen = 0
    with t.no_grad():
        for _ in range(num_batches):
            batch = next(batch_source)
            x = batch.activations.to(device=device, dtype=t.float32)
            doc_ids = batch.token_doc_ids.long().to(device) + n_docs_seen
            topic_labels = batch.topic_labels.long().cpu()
            sentiment_labels = batch.sentiment_labels.long().cpu()
            x_all.append(x.cpu())
            doc_ids_all.append(doc_ids.cpu())
            topic_all.append(topic_labels)
            sentiment_all.append(sentiment_labels)
            n_docs_seen += topic_labels.shape[0]

    x_full = t.cat(x_all, dim=0)
    doc_ids_full = t.cat(doc_ids_all, dim=0)
    topic_full = t.cat(topic_all, dim=0)
    sentiment_full = t.cat(sentiment_all, dim=0)
    n_docs = topic_full.shape[0]

    doc_fit, doc_eval = _stratified_split_indices(sentiment_full, test_fraction=0.5, seed=seed)
    is_fit_doc = t.zeros(n_docs, dtype=t.bool)
    is_fit_doc[doc_fit] = True
    token_is_fit = is_fit_doc[doc_ids_full.cpu()]

    from sklearn.linear_model import LogisticRegression

    # Reference classifiers: fit on genuine raw activations, doc-pooled, FIT split only --
    # these never see the SAE at all, so they are an honest external judge of "does this
    # reconstructed vector look like it carries sentiment/topic X" rather than a probe the SAE
    # could have been shaped to fool.
    x_docs_fit = _mean_pool_by_doc(x_full[token_is_fit].to(device), doc_ids_full[token_is_fit].to(device), n_docs)[doc_fit].cpu()
    mu_x, sd_x = x_docs_fit.mean(dim=0, keepdim=True), x_docs_fit.std(dim=0, keepdim=True).clamp_min(1e-6)
    clf_sent = LogisticRegression(max_iter=5000, C=1.0).fit(
        ((x_docs_fit - mu_x) / sd_x).numpy(), sentiment_full[doc_fit].numpy()
    )
    clf_top = LogisticRegression(max_iter=5000, C=1.0).fit(
        ((x_docs_fit - mu_x) / sd_x).numpy(), topic_full[doc_fit].numpy()
    )

    with t.no_grad():
        _, p_pos_full, q_pos_full = ae.encode(x_full.to(device), return_branches=True)
        q_pos_full = trainer.apply_q_bottleneck(q_pos_full)
        target_full = p_pos_full if branch == "p" else q_pos_full
        target_shape = target_full.shape  # (n_tokens, h, m) for p, (n_tokens, h, n) for q
        target_flat_full = target_full.reshape(target_shape[0], -1)
        target_docs_fit = _mean_pool_by_doc(
            target_flat_full[token_is_fit], doc_ids_full[token_is_fit].to(device), n_docs
        )[doc_fit].cpu()
        w_sent = _signed_binary_direction(target_docs_fit, sentiment_full[doc_fit])
        w_sent_hat = (w_sent / w_sent.norm().clamp_min(1e-12)).to(device)
        proj_fit = target_docs_fit.to(device) @ w_sent_hat
        proj_std = float(proj_fit.std().clamp_min(1e-12))

        eval_mask = ~token_is_fit
        x_eval = x_full[eval_mask].to(device)
        doc_ids_eval_raw = doc_ids_full[eval_mask].to(device)
        uniq_eval_docs, doc_ids_eval = t.unique(doc_ids_eval_raw, return_inverse=True)
        n_eval_docs = uniq_eval_docs.shape[0]
        sentiment_eval = sentiment_full[doc_eval].numpy()
        topic_eval = topic_full[doc_eval].numpy()

        p_pos_eval = p_pos_full[eval_mask]
        q_pos_eval = q_pos_full[eval_mask]
        direction_shaped = w_sent_hat.view(target_shape[1], target_shape[2])

        x_eval_docs = _mean_pool_by_doc(x_eval, doc_ids_eval, n_eval_docs).cpu()
        x_eval_docs_var = float((x_eval_docs - x_eval_docs.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12))
        x_eval_std = ((x_eval_docs - mu_x) / sd_x).numpy()

        results: dict[str, Any] = {"alphas": [], "recon_fvu": [], "p_sentiment_positive": [], "topic_acc": [], "branch": branch}
        for alpha in alphas:
            if branch == "q":
                p_patched = p_pos_eval
                q_patched = (q_pos_eval + alpha * proj_std * direction_shaped.unsqueeze(0)).clamp_min(0.0)
            else:
                p_patched = (p_pos_eval + alpha * proj_std * direction_shaped.unsqueeze(0)).clamp_min(0.0)
                q_patched = q_pos_eval
            dense = ae._combine(p_patched, q_patched, p_patched, q_patched)
            post_topk = dense.topk(int(ae.k.item()), sorted=False, dim=-1)
            f_patched = t.zeros_like(dense).scatter_(dim=-1, index=post_topk.indices, src=post_topk.values)
            x_hat_patched = ae.decode(f_patched)
            x_hat_docs = _mean_pool_by_doc(x_hat_patched, doc_ids_eval, n_eval_docs).cpu()

            # FVU against the doc's OWN (unpatched) target -- how much intervention distorts
            # reconstruction relative to genuine per-doc variance in the eval split.
            recon_fvu = float((x_hat_docs - x_eval_docs).pow(2).sum()) / x_eval_docs_var
            x_hat_std = ((x_hat_docs - mu_x) / sd_x).numpy()
            p_positive = clf_sent.predict_proba(x_hat_std)[:, 1].mean()
            top_acc = float((clf_top.predict(x_hat_std) == topic_eval).mean())

            results["alphas"].append(alpha)
            results["recon_fvu"].append(recon_fvu)
            results["p_sentiment_positive"].append(float(p_positive))
            results["topic_acc"].append(top_acc)

        results["raw_topic_acc_no_intervention"] = float((clf_top.predict(x_eval_std) == topic_eval).mean())
        results["raw_p_sentiment_positive_no_intervention"] = float(clf_sent.predict_proba(x_eval_std)[:, 1].mean())

    if was_training:
        ae.train()
    return results


def _pooled_sae_representations(ae, x: t.Tensor, token_doc_ids: t.Tensor, n_docs: int) -> dict[str, t.Tensor]:
    """Doc-mean-pooled representations for a flat or Kron SAE, keyed by arm name.

    Emits both the pre-top-k dense code (`dense`) and the post-top-k sparse code
    (`topk`) so the Part A flat-vs-Kron comparison is not an artifact of where in
    the encoder the probe reads from; for Kron the individual `p`/`q` branches
    are emitted too.
    """
    from dictionary_learning.dictionary_kron import KronAutoEncoderTopK

    out: dict[str, t.Tensor] = {}
    if isinstance(ae, KronAutoEncoderTopK):
        dense, p_pos, q_pos = ae._dense_features(x)
        topk = ae.encode(x)
        out["p"] = p_pos.reshape(p_pos.shape[0], -1)
        out["q"] = q_pos.reshape(q_pos.shape[0], -1)
    else:
        dense = t.nn.functional.relu(ae.encoder(x - ae.b_dec))
        topk = ae.encode(x)
    out["dense"] = dense
    out["topk"] = topk
    return {key: _mean_pool_by_doc(val, token_doc_ids, n_docs).detach().cpu() for key, val in out.items()}


def evaluate_synthetic_conjunctive_probe(
    trainer: Any,
    batch_source: SyntheticJointActivationBuffer,
    num_batches: int,
    seed: int,
    device: str,
    test_fraction: float = 0.3,
) -> dict[str, Any]:
    """Decode the alpha-parameterized conjunctive target y_alpha from an SAE's latents.

    This is the readout half of Part A's causal test. `y_alpha` is never
    supervised by any training loss -- it is a held-out target -- so probe
    accuracy measures only whether the learned dictionary happens to expose the
    ground-truth structure. The RQ is whether KronSAE's bilinear (mAND) encoder
    pulls ahead of a flat SAE *as alpha increases*, i.e. specifically as the
    ground truth becomes conjunctive rather than linear in the primitives.

    Every accuracy uses the REPORT.md §9.3 standardized-probe standard, and is
    reported alongside two references that make it readable:
      * `raw_acc`      -- the same probe on the raw activation, no SAE at all
                          (an SAE that beats this is adding something)
      * `linear_rule_acc` / `conjunctive_rule_acc` from the buffer's calibration
                          (the additive and product ceilings on the primitives)
    Per REPORT.md §9.4 the number that actually matters is the gap against the
    matching `--synthetic_skip_training` random-init control run, NOT the gap
    against chance.
    """
    if batch_source.conjunctive_dirs is None:
        raise ValueError("batch_source was not constructed with a conjunctive_alpha")

    ae = unwrap_model(trainer.ae)
    was_training = ae.training
    ae.eval()

    reps_all: dict[str, list[t.Tensor]] = {}
    raw_all: list[t.Tensor] = []
    label_all: list[t.Tensor] = []

    with t.no_grad():
        for _ in range(num_batches):
            batch = next(batch_source)
            x = batch.activations.to(device=device, dtype=t.float32)
            token_doc_ids = batch.token_doc_ids.long().to(device)
            labels = batch.conjunctive_labels.long().cpu()
            n_docs = labels.shape[0]

            raw_all.append(_mean_pool_by_doc(x, token_doc_ids, n_docs).detach().cpu())
            for key, val in _pooled_sae_representations(ae, x, token_doc_ids, n_docs).items():
                reps_all.setdefault(key, []).append(val)
            label_all.append(labels)

    if was_training:
        ae.train()

    labels = t.cat(label_all, dim=0)
    train_idx, test_idx = _stratified_split_indices(labels, test_fraction=test_fraction, seed=seed)

    arms: dict[str, t.Tensor] = {"raw": t.cat(raw_all, dim=0)}
    arms.update({key: t.cat(vals, dim=0) for key, vals in reps_all.items()})

    metrics: dict[str, Any] = {
        "conjunctive_chance": 0.5,
        "n_docs": int(labels.shape[0]),
        "n_train": int(train_idx.shape[0]),
        "n_test": int(test_idx.shape[0]),
        "eval_marginal_pos_frac": float(labels.float().mean().item()),
    }
    for key, feats in arms.items():
        acc, converged = standardized_logreg_accuracy(
            X_train=feats[train_idx],
            y_train=labels[train_idx],
            X_test=feats[test_idx],
            y_test=labels[test_idx],
        )
        metrics[f"{key}_conjunctive_acc"] = acc
        metrics[f"{key}_conjunctive_converged"] = converged
        metrics[f"{key}_probe_dim"] = int(feats.shape[1])
    metrics["buffer_diagnostics"] = batch_source.conjunctive_diagnostics()
    return metrics


def build_model(args, dist_env: DistEnv, device: str) -> tuple[LanguageModel, Any]:
    # nnsight/transformers load onto rank-local device.
    from nnsight import LanguageModel

    model = LanguageModel(
        args.model_name,
        dispatch=True,
        device_map=device,
    )
    # Different HF model families expose their transformer block list under a
    # different attribute path (pythia/GPT-NeoX: model.gpt_neox.layers; BLOOM:
    # model.transformer.h; GPT-2 style: model.transformer.h as well). Try the
    # known paths in order rather than hardcoding pythia's, so --model_name can
    # be swapped (e.g. bigscience/bloom-560m) without touching this function.
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
        raise AttributeError(
            f"Could not find a transformer block list on {args.model_name} "
            f"(tried {layer_list_paths}); add its attribute path to build_model()."
        )
    return model, submodule


def resolve_schedule_steps(args) -> tuple[int, Optional[int], int]:
    total = args.total_steps
    if total <= 1:
        return 0, None, 0

    if not (0.0 <= args.warmup_frac < 1.0):
        raise ValueError("--warmup_frac must be in [0,1)")
    if not (0.0 <= args.decay_start_frac <= 1.0):
        raise ValueError("--decay_start_frac must be in [0,1]")
    if not (0.0 <= args.threshold_start_frac <= 1.0):
        raise ValueError("--threshold_start_frac must be in [0,1]")

    if args.warmup_steps is None:
        warmup_steps = int(total * args.warmup_frac)
    else:
        warmup_steps = int(args.warmup_steps)
    warmup_steps = max(0, min(warmup_steps, max(0, total - 1)))

    if args.threshold_start_step is None:
        threshold_start_step = int(total * args.threshold_start_frac)
    else:
        threshold_start_step = int(args.threshold_start_step)
    threshold_start_step = max(0, min(threshold_start_step, max(0, total - 1)))

    if args.decay_start is None:
        decay_start = int(total * args.decay_start_frac)
    else:
        decay_start = int(args.decay_start)
    decay_start = max(0, min(decay_start, max(0, total - 1)))
    if decay_start <= warmup_steps:
        decay_start = warmup_steps + 1
    if decay_start >= total:
        decay_start = None

    return warmup_steps, decay_start, threshold_start_step


def create_main_trainers(args, device: str):
    warmup_steps, decay_start, threshold_start_step = resolve_schedule_steps(args)
    shared = {
        "steps": args.total_steps,
        "activation_dim": args.activation_dim,
        "layer": args.layer,
        "lm_name": args.model_name,
        "k": args.k,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "threshold_start_step": threshold_start_step,
        "seed": args.seed,
        "device": device,
        "lr": args.lr,
    }

    trainers = []

    trainers.append(
        (
            "flat_unsup",
            TopKTrainer(
                dict_size=args.flat_dict_size,
                auxk_alpha=args.auxk_alpha,
                wandb_name="flat_unsup",
                **shared,
            ),
        )
    )

    trainers.append(
        (
            "flat_sup",
            FlatSupervisedTopKTrainer(
                dict_size=args.flat_dict_size,
                auxk_alpha=args.auxk_alpha,
                use_supervision=True,
                lambda_sup=args.lambda_sup,
                lambda_sup_warmup_frac=args.lambda_sup_warmup_frac,
                supcon_temperature=args.supcon_temperature,
                flat_sup_mode=args.flat_sup_mode,
                sup_grad_scale=args.flat_sup_grad_scale,
                wandb_name="flat_sup",
                **shared,
            ),
        )
    )

    trainers.append(
        (
            "kron_unsup_mand",
            KronTopKTrainer(
                h=args.kron_h,
                m=args.kron_m,
                n=args.kron_n,
                combine_rule="mand",
                use_supervision=False,
                lambda_sup=args.lambda_sup,
                auxk_alpha=args.auxk_alpha,
                wandb_name="kron_unsup_mand",
                **shared,
            ),
        )
    )

    trainers.append(
        (
            "kron_sup_mand",
            KronTopKTrainer(
                h=args.kron_h,
                m=args.kron_m,
                n=args.kron_n,
                combine_rule="mand",
                use_supervision=True,
                lambda_sup=args.lambda_sup,
                lambda_sup_warmup_frac=args.lambda_sup_warmup_frac,
                auxk_alpha=args.auxk_alpha,
                supcon_temperature=args.supcon_temperature,
                wandb_name="kron_sup_mand",
                **shared,
            ),
        )
    )

    trainers.append(
        (
            "kron_sup_mor",
            KronTopKTrainer(
                h=args.kron_h,
                m=args.kron_m,
                n=args.kron_n,
                combine_rule="mor",
                use_supervision=True,
                lambda_sup=args.lambda_sup,
                lambda_sup_warmup_frac=args.lambda_sup_warmup_frac,
                auxk_alpha=args.auxk_alpha,
                supcon_temperature=args.supcon_temperature,
                wandb_name="kron_sup_mor",
                **shared,
            ),
        )
    )

    trainers.append(
        (
            "kron_sup_concat",
            KronTopKTrainer(
                h=args.kron_h,
                m=args.kron_m,
                n=args.kron_n,
                combine_rule="concat",
                use_supervision=True,
                lambda_sup=args.lambda_sup,
                lambda_sup_warmup_frac=args.lambda_sup_warmup_frac,
                auxk_alpha=args.auxk_alpha,
                supcon_temperature=args.supcon_temperature,
                wandb_name="kron_sup_concat",
                **shared,
            ),
        )
    )

    return trainers


def create_pilot_trainers(args, device: str):
    warmup_steps, decay_start, threshold_start_step = resolve_schedule_steps(args)

    shared = {
        "steps": args.total_steps,
        "activation_dim": args.activation_dim,
        "layer": args.layer,
        "lm_name": args.model_name,
        "k": args.k,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "threshold_start_step": threshold_start_step,
        "seed": args.seed,
        "device": device,
        "lr": args.lr,
    }

    trainers = []
    for label_type in ("topic", "sentiment"):
        trainers.append(
            (
                f"flat_pilot_{label_type}",
                FlatSupervisedTopKTrainer(
                    dict_size=args.flat_dict_size,
                    auxk_alpha=args.auxk_alpha,
                    use_supervision=True,
                    lambda_sup=args.lambda_sup,
                    lambda_sup_warmup_frac=args.lambda_sup_warmup_frac,
                    supcon_temperature=args.supcon_temperature,
                    flat_sup_mode="pilot_collect_full",
                    label_type=label_type,
                    sup_grad_scale=args.flat_sup_grad_scale,
                    wandb_name=f"flat_pilot_{label_type}",
                    **shared,
                ),
            )
        )

        trainers.append(
            (
                f"kron_pilot_{label_type}",
                KronTopKTrainer(
                    h=args.pilot_h,
                    m=args.pilot_m,
                    n=args.pilot_n,
                    combine_rule="mand",
                    use_supervision=True,
                    lambda_sup=args.lambda_sup,
                    lambda_sup_warmup_frac=args.lambda_sup_warmup_frac,
                    auxk_alpha=args.auxk_alpha,
                    supcon_temperature=args.supcon_temperature,
                    label_type=label_type,
                    wandb_name=f"kron_pilot_{label_type}",
                    **shared,
                ),
            )
        )

    return trainers


def create_joint_trainers(args, device: str):
    warmup_steps, decay_start, threshold_start_step = resolve_schedule_steps(args)

    shared = {
        "steps": args.total_steps,
        "activation_dim": args.activation_dim,
        "layer": args.layer,
        "lm_name": args.model_name,
        "k": args.k,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "threshold_start_step": threshold_start_step,
        "seed": args.seed,
        "device": device,
        "lr": args.lr,
    }

    if args.joint_mode == "collect_only":
        flat_mode = "joint_collect_only"
        kron_mode = "joint_collect_only"
    elif args.joint_mode == "dpo_cross":
        flat_mode = "joint_dpo_cross"
        kron_mode = "joint_dpo_cross"
    else:
        raise ValueError(f"Unsupported --joint_mode={args.joint_mode}")

    return [
        (
            "flat_joint",
            FlatSupervisedTopKTrainer(
                dict_size=args.flat_dict_size,
                auxk_alpha=args.auxk_alpha,
                use_supervision=True,
                lambda_sup=args.lambda_sup,
                lambda_sup_warmup_frac=args.lambda_sup_warmup_frac,
                supervision_mode=flat_mode,
                dpo_beta=args.dpo_beta,
                flat_sup_mode="joint_dpo_full",
                sup_grad_scale=args.flat_sup_grad_scale,
                wandb_name=f"flat_joint_{args.joint_mode}",
                **shared,
            ),
        ),
        (
            "kron_joint",
            KronTopKTrainer(
                h=args.joint_h,
                m=args.joint_m,
                n=args.joint_n,
                combine_rule=args.combine_rule,
                use_supervision=True,
                lambda_sup=args.lambda_sup,
                lambda_sup_warmup_frac=args.lambda_sup_warmup_frac,
                auxk_alpha=args.auxk_alpha,
                supervision_mode=kron_mode,
                dpo_beta=args.dpo_beta,
                dpo_beta_p_collect_sentiment=args.dpo_beta_p_collect_sentiment,
                dpo_beta_p_contrast_topic=args.dpo_beta_p_contrast_topic,
                dpo_beta_q_collect_topic=args.dpo_beta_q_collect_topic,
                dpo_beta_q_contrast_sentiment=args.dpo_beta_q_contrast_sentiment,
                dpo_weight_p_collect_sentiment=args.dpo_w_p_collect_sentiment,
                dpo_weight_p_contrast_topic=args.dpo_w_p_contrast_topic,
                dpo_weight_q_collect_topic=args.dpo_w_q_collect_topic,
                dpo_weight_q_contrast_sentiment=args.dpo_w_q_contrast_sentiment,
                contrast_start_frac=args.contrast_start_frac,
                q_gradient_surgery=args.q_gradient_surgery,
                q_gradient_surgery_include_recon=args.q_gradient_surgery_include_recon,
                q_contrast_grad_norm_match=args.q_contrast_grad_norm_match,
                q_contrast_grad_norm_match_cap=args.q_contrast_grad_norm_match_cap,
                q_separate_grad_clip=args.q_separate_grad_clip,
                grad_norm_probe_every=args.grad_norm_probe_every,
                lambda_orth=args.lambda_orth,
                lambda_recon=args.lambda_recon,
                gradnorm=args.gradnorm,
                gradnorm_alpha=args.gradnorm_alpha,
                gradnorm_lr=args.gradnorm_lr,
                gradnorm_log_every=args.gradnorm_log_every,
                cagrad=args.cagrad,
                cagrad_c=args.cagrad_c,
                cagrad_log_every=args.cagrad_log_every,
                decouple_recon_mode=args.decouple_recon_mode,
                decouple_stopgrad_strength=args.decouple_stopgrad_strength,
                decouple_stopgrad_branches=args.decouple_stopgrad_branches,
                decouple_dir_ema=args.decouple_dir_ema,
                q_recon_grad_scale=args.q_recon_grad_scale,
                adv_suppress=args.adv_suppress,
                adv_grl_lambda=args.adv_grl_lambda,
                adv_lr=args.adv_lr,
                adv_hidden=args.adv_hidden,
                leace_suppress=args.leace_suppress,
                leace_lambda=args.leace_lambda,
                leace_ref_lr=args.leace_ref_lr,
                leace_ref_hidden=args.leace_ref_hidden,
                leace_subspace=args.leace_subspace,
                leace_subspace_refit_every=args.leace_subspace_refit_every,
                leace_subspace_ema_decay=args.leace_subspace_ema_decay,
                leace_subspace_eps=args.leace_subspace_eps,
                bcd_alternate=args.bcd_alternate,
                bcd_phase_steps=args.bcd_phase_steps,
                cond_recon=args.cond_recon,
                swap_recon=args.swap_recon,
                swap_recon_lambda=args.swap_recon_lambda,
                q_bottleneck_rank=args.q_bottleneck_rank,
                rl_suppress=args.rl_suppress,
                rl_sigma=args.rl_sigma,
                rl_lambda=args.rl_lambda,
                rl_ref_lr=args.rl_ref_lr,
                rl_ref_hidden=args.rl_ref_hidden,
                rl_baseline_decay=args.rl_baseline_decay,
                bcd_rl_combo=args.bcd_rl_combo,
                grad_accum_steps=args.grad_accum_steps,
                wandb_name=f"kron_joint_{args.joint_mode}",
                **shared,
            ),
        ),
    ]


def maybe_wrap_ddp(trainers, dist_env: DistEnv):
    if not dist_env.enabled:
        return

    for _, trainer in trainers:
        if t.cuda.is_available():
            trainer.ae = DDPProxy(
                trainer.ae,
                device_ids=[dist_env.local_rank],
                output_device=dist_env.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
            )
        else:
            trainer.ae = DDPProxy(trainer.ae)


def filter_trainers(trainers, only_names: Optional[list[str]]):
    if not only_names:
        return trainers
    only = set(only_names)
    filtered = [(n, t) for n, t in trainers if n in only]
    if not filtered:
        raise ValueError(f"--only_trainers did not match any trainer names. Provided={sorted(only)}")
    return filtered


def _topk_jaccard(prev_idx: Optional[t.Tensor], curr_idx: t.Tensor) -> float:
    if prev_idx is None:
        return float("nan")
    prev_u = t.unique(prev_idx.reshape(-1))
    curr_u = t.unique(curr_idx.reshape(-1))
    inter = t.isin(prev_u, curr_u).sum().item()
    union = prev_u.numel() + curr_u.numel() - inter
    return float(inter / union) if union > 0 else float("nan")


def collect_debug_stats(trainer, batch, prev_topk_idx: Optional[t.Tensor]) -> tuple[dict[str, float], Optional[t.Tensor]]:
    ae = unwrap_model(trainer.ae)
    x = batch.activations.to(trainer.device, dtype=t.float32)
    out: dict[str, float] = {}

    with t.no_grad():
        if isinstance(trainer, FlatSupervisedTopKTrainer):
            _, top_acts_BK, top_idx_BK, post_relu_BF = ae.encode(
                x,
                return_topk=True,
                use_threshold=False,
            )
            out.update(
                {
                    "post_relu_max": float(post_relu_BF.max().item()),
                    "post_relu_mean": float(post_relu_BF.mean().item()),
                    "topk_act_max": float(top_acts_BK.max().item()),
                    "topk_act_mean": float(top_acts_BK.mean().item()),
                    "topk_feat_jaccard": _topk_jaccard(prev_topk_idx, top_idx_BK.detach().cpu()),
                }
            )
            if trainer.flat_sup_mode == "split_half":
                split_idx = trainer.p_region_dim
                if split_idx is None:
                    split_idx = post_relu_BF.shape[-1] // 2
                split_idx = max(1, min(int(split_idx), post_relu_BF.shape[-1] - 1))
                p = post_relu_BF[:, :split_idx]
                q = post_relu_BF[:, split_idx:]
                out.update(
                    {
                        "p_max": float(p.max().item()),
                        "q_max": float(q.max().item()),
                        "p_mean": float(p.mean().item()),
                        "q_mean": float(q.mean().item()),
                    }
                )
            return out, top_idx_BK.detach().cpu()

        if isinstance(trainer, KronTopKTrainer):
            _, top_acts_BK, top_idx_BK, dense_BF, p_pos_BHM, q_pos_BHN = ae.encode(
                x,
                return_topk=True,
                use_threshold=False,
                return_branches=True,
            )
            out.update(
                {
                    "dense_max": float(dense_BF.max().item()),
                    "dense_mean": float(dense_BF.mean().item()),
                    "p_max": float(p_pos_BHM.max().item()),
                    "q_max": float(q_pos_BHN.max().item()),
                    "p_mean": float(p_pos_BHM.mean().item()),
                    "q_mean": float(q_pos_BHN.mean().item()),
                    "topk_act_max": float(top_acts_BK.max().item()),
                    "topk_act_mean": float(top_acts_BK.mean().item()),
                    "topk_feat_jaccard": _topk_jaccard(prev_topk_idx, top_idx_BK.detach().cpu()),
                }
            )
            return out, top_idx_BK.detach().cpu()

    return out, prev_topk_idx


def save_checkpoints(trainers, save_dir: Path, step: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, trainer in trainers:
        run_dir = save_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        ae = unwrap_model(trainer.ae)
        state = {k: v.detach().cpu() for k, v in ae.state_dict().items()}
        t.save(state, run_dir / f"ae_step_{step}.pt")
        with (run_dir / "trainer_config.json").open("w", encoding="utf-8") as f:
            json.dump(trainer.config, f, indent=2)


def measure_synthetic_reconstruction(
    trainer: KronTopKTrainer,
    batch_source: SyntheticJointActivationBuffer,
    num_batches: int,
) -> dict[str, float]:
    """Measurement-only reconstruction quality (mean per-token l2_loss and FVU)."""
    l2_sum = 0.0
    sq_err_sum = 0.0
    xs = []
    with t.no_grad():
        for _ in range(num_batches):
            batch = next(batch_source)
            batch.activations = batch.activations.to(dtype=t.float32)
            log = trainer.loss(batch, step=None, logging=True)
            l2_sum += log.losses["l2_loss"]
            sq_err_sum += float((log.x - log.x_hat).pow(2).sum().item())
            xs.append(log.x.detach().cpu())
    X = t.cat(xs, dim=0)
    total_var = float((X - X.mean(dim=0, keepdim=True)).pow(2).sum().item())
    return {
        "recon_l2_loss": l2_sum / max(1, num_batches),
        "recon_fvu": sq_err_sum / total_var if total_var > 0 else float("nan"),
    }


def conjunctive_buffer_kwargs(args) -> dict[str, Any]:
    """Translate the --synthetic_conjunctive CLI flags into buffer kwargs.

    Returns `{}` when --synthetic_conjunctive is off, so every pre-existing
    synthetic run constructs its buffer exactly as before.
    """
    if not getattr(args, "synthetic_conjunctive", False):
        return {}
    return {
        "conjunctive_alpha": args.conjunctive_alpha,
        "conjunctive_signal_scale": args.conjunctive_signal_scale,
        "conjunctive_balance": args.conjunctive_balance == "on",
        "conjunctive_calib_batches": args.conjunctive_calib_batches,
    }


def train_synthetic_diagnostic(args, dist_env: DistEnv, device: str) -> None:
    if args.joint_mode != "dpo_cross":
        raise ValueError("--synthetic_diagnostic currently supports only --joint_mode dpo_cross")

    trainers = create_joint_trainers(args, device=device)
    trainers = filter_trainers(trainers, args.only_trainers)
    maybe_wrap_ddp(trainers, dist_env)

    batch_source = SyntheticJointActivationBuffer(
        activation_dim=args.activation_dim,
        doc_batch_size=args.doc_batch_size,
        num_topics=args.synthetic_num_topics,
        num_sentiments=args.synthetic_num_sentiments,
        topic_signal_scale=args.synthetic_topic_signal_scale,
        sentiment_signal_scale=args.synthetic_sentiment_signal_scale,
        noise_std=args.synthetic_noise_std,
        seed=args.seed + (17 * dist_env.rank),
        device=device,
        **conjunctive_buffer_kwargs(args),
    )

    if dist_env.is_main:
        diag = batch_source.direction_diagnostics()
        rank_print(
            dist_env,
            "Synthetic diagnostics: "
            f"max|cos(sent,topic)|={diag['max_abs_sent_topic_cos']:.4e}, "
            f"max|sent-orth-error|={diag['max_abs_sent_self_offdiag']:.4e}, "
            f"max|topic-orth-error|={diag['max_abs_topic_self_offdiag']:.4e}",
        )
        if batch_source.conjunctive_dirs is not None:
            cdiag = batch_source.conjunctive_diagnostics()
            rank_print(
                dist_env,
                "Conjunctive target: "
                f"alpha={cdiag['conjunctive_alpha']:.3f}, s={cdiag['product_scale_s']:.4f}, "
                f"offset={cdiag['score_offset']:.4e}, P(y=1)={cdiag['marginal_pos_frac']:.4f}, "
                f"linear-rule={cdiag['linear_rule_acc']:.4f}, product-rule={cdiag['conjunctive_rule_acc']:.4f}, "
                f"max|cos(conj,sent)|={cdiag['max_abs_conj_sent_cos']:.3e}, "
                f"max|cos(conj,topic)|={cdiag['max_abs_conj_topic_cos']:.3e}",
            )
        rank_print(dist_env, f"Training synthetic mode for {args.total_steps} steps on {device}")

    q_grad_stat_keys = [
        "q_pcgrad_collect_grad_norm",
        "q_pcgrad_contrast_grad_norm",
        "q_pcgrad_recon_grad_norm",
        "q_pcgrad_collect_proj_grad_norm",
        "q_pcgrad_contrast_proj_grad_norm",
        "q_pcgrad_recon_proj_grad_norm",
        "q_pcgrad_conflict_dot_collect_contrast",
        "q_pcgrad_conflict_dot_collect_recon",
        "q_pcgrad_conflict_dot_contrast_recon",
        "q_pcgrad_conflict_active_collect_contrast",
        "q_pcgrad_conflict_active_collect_recon",
        "q_pcgrad_conflict_active_contrast_recon",
        "q_contrast_raw_norm",
        "q_contrast_scaled_norm",
        "q_other_norm",
    ]
    q_grad_stat_sums: dict[str, dict[str, float]] = {
        name: {key: 0.0 for key in q_grad_stat_keys}
        for name, trainer in trainers
        if isinstance(trainer, KronTopKTrainer)
    }
    q_grad_stat_count = 0

    save_dir = Path(args.output_dir) / "checkpoints"
    save_interval = args.save_every if args.save_every > 0 else None
    num_train_steps = 0 if args.synthetic_skip_training else args.total_steps
    probe_every = getattr(args, "synthetic_probe_every", 0)
    probe_trajectories: dict[str, list[dict[str, Any]]] = {
        name: [] for name, trainer in trainers if isinstance(trainer, KronTopKTrainer)
    }

    def _run_probe_checkpoint(current_step: int) -> None:
        """Mid-training probe snapshot (REPORT1.md sec.19 follow-up): the point is to see
        whether a mechanism's numbers are still moving at the final step or have already
        plateaued, since a flat final-step comparison can't distinguish "5000 steps wasn't
        enough" from "this mechanism doesn't do anything." Defaults to a smaller eval batch
        count than the end-of-run probe to keep the added cost small relative to training;
        pass --synthetic_probe_snapshot_batches to match the final read's batch count exactly
        (REPORT1.md sec.19.2: needed to tell a real training-step effect apart from the
        snapshot simply having less eval data than the final read)."""
        if not (dist_env.is_main and probe_every > 0):
            return
        override = getattr(args, "synthetic_probe_snapshot_batches", 0)
        small_num_batches = override if override > 0 else max(4, args.synthetic_eval_batches // 4)
        for probe_name, probe_trainer in trainers:
            if not isinstance(probe_trainer, KronTopKTrainer):
                continue
            probe_source = SyntheticJointActivationBuffer(
                activation_dim=args.activation_dim,
                doc_batch_size=args.doc_batch_size,
                num_topics=args.synthetic_num_topics,
                num_sentiments=args.synthetic_num_sentiments,
                topic_signal_scale=args.synthetic_topic_signal_scale,
                sentiment_signal_scale=args.synthetic_sentiment_signal_scale,
                noise_std=args.synthetic_noise_std,
                seed=args.seed + 1001 + current_step,
                device=device,
                **conjunctive_buffer_kwargs(args),
            )
            snap = evaluate_synthetic_joint_probe(
                trainer=probe_trainer,
                batch_source=probe_source,
                num_batches=small_num_batches,
                seed=args.seed + 2001 + current_step,
                probe_epochs=args.synthetic_probe_epochs,
                probe_lr=args.synthetic_probe_lr,
                probe_batch_size=args.synthetic_probe_batch_size,
                device=device,
            )
            snap["step"] = current_step
            probe_trajectories[probe_name].append(snap)
            rank_print(
                dist_env,
                f"  [probe @ step={current_step}] [{probe_name}] "
                f"P(sent)={snap['p_sentiment_acc']:.3f} Q(sent)={snap['q_sentiment_acc']:.3f} "
                f"P(top)={snap['p_topic_acc']:.3f} Q(top)={snap['q_topic_acc']:.3f}",
            )

    for step in range(num_train_steps):
        batch = next(batch_source)
        batch.activations = batch.activations.to(dtype=t.float32)
        losses = {}
        for name, trainer in trainers:
            if isinstance(trainer, TopKTrainer) and not isinstance(trainer, FlatSupervisedTopKTrainer):
                loss_val = trainer.update(step, batch.activations)
            else:
                loss_val = trainer.update(step, batch)
            losses[name] = loss_val
            if name in q_grad_stat_sums:
                for key in q_grad_stat_keys:
                    q_grad_stat_sums[name][key] += float(getattr(trainer, key, 0.0))
        q_grad_stat_count += 1
        if dist_env.is_main and step % args.log_every == 0:
            loss_parts = " | ".join(f"{k}:{v:.4f}" for k, v in losses.items())
            print(f"step={step} | {loss_parts}", flush=True)
        if probe_every > 0 and step > 0 and step % probe_every == 0:
            _run_probe_checkpoint(step)
        if save_interval is not None and step > 0 and step % save_interval == 0:
            if dist_env.enabled:
                dist_barrier(dist_env)
            if dist_env.is_main:
                save_checkpoints(trainers, save_dir, step)
            if dist_env.enabled:
                dist_barrier(dist_env)

    if dist_env.enabled:
        dist_barrier(dist_env)
    if dist_env.is_main:
        save_checkpoints(trainers, save_dir, args.total_steps)
        result_payload: dict[str, Any] = {
            "mode": "synthetic_diagnostic",
            "args": vars(args),
            "results": {},
        }
        for name, trainer in trainers:
            if isinstance(trainer, KronTopKTrainer):
                eval_source = SyntheticJointActivationBuffer(
                    activation_dim=args.activation_dim,
                    doc_batch_size=args.doc_batch_size,
                    num_topics=args.synthetic_num_topics,
                    num_sentiments=args.synthetic_num_sentiments,
                    topic_signal_scale=args.synthetic_topic_signal_scale,
                    sentiment_signal_scale=args.synthetic_sentiment_signal_scale,
                    noise_std=args.synthetic_noise_std,
                    seed=args.seed + 1001,
                    device=device,
                    **conjunctive_buffer_kwargs(args),
                )
                metrics = evaluate_synthetic_joint_probe(
                    trainer=trainer,
                    batch_source=eval_source,
                    num_batches=args.synthetic_eval_batches,
                    seed=args.seed + 2001,
                    probe_epochs=args.synthetic_probe_epochs,
                    probe_lr=args.synthetic_probe_lr,
                    probe_batch_size=args.synthetic_probe_batch_size,
                    device=device,
                )
                metrics = dict(metrics)
                metrics["probe_trajectory"] = probe_trajectories.get(name, [])
                geometry_source = SyntheticJointActivationBuffer(
                    activation_dim=args.activation_dim,
                    doc_batch_size=args.doc_batch_size,
                    num_topics=args.synthetic_num_topics,
                    num_sentiments=args.synthetic_num_sentiments,
                    topic_signal_scale=args.synthetic_topic_signal_scale,
                    sentiment_signal_scale=args.synthetic_sentiment_signal_scale,
                    noise_std=args.synthetic_noise_std,
                    seed=args.seed + 1001,
                    device=device,
                    **conjunctive_buffer_kwargs(args),
                )
                metrics.update(
                    evaluate_synthetic_representation_geometry(
                        trainer=trainer,
                        batch_source=geometry_source,
                        num_batches=args.synthetic_eval_batches,
                        seed=args.seed + 3001,
                        device=device,
                    )
                )
                intervention_alphas_str = getattr(args, "synthetic_intervention_alphas", "") or ""
                if intervention_alphas_str.strip():
                    intervention_alphas = [float(a) for a in intervention_alphas_str.split(",")]
                    for branch_name in ("q", "p"):
                        intervention_source = SyntheticJointActivationBuffer(
                            activation_dim=args.activation_dim,
                            doc_batch_size=args.doc_batch_size,
                            num_topics=args.synthetic_num_topics,
                            num_sentiments=args.synthetic_num_sentiments,
                            topic_signal_scale=args.synthetic_topic_signal_scale,
                            sentiment_signal_scale=args.synthetic_sentiment_signal_scale,
                            noise_std=args.synthetic_noise_std,
                            seed=args.seed + 1001,
                            device=device,
                            **conjunctive_buffer_kwargs(args),
                        )
                        metrics[f"intervention_{branch_name}"] = evaluate_synthetic_intervention(
                            trainer=trainer,
                            batch_source=intervention_source,
                            num_batches=args.synthetic_eval_batches,
                            seed=args.seed + 4001,
                            device=device,
                            alphas=intervention_alphas,
                            branch=branch_name,
                        )
                    metrics["intervention"] = metrics["intervention_q"]  # back-compat key
                recon_source = SyntheticJointActivationBuffer(
                    activation_dim=args.activation_dim,
                    doc_batch_size=args.doc_batch_size,
                    num_topics=args.synthetic_num_topics,
                    num_sentiments=args.synthetic_num_sentiments,
                    topic_signal_scale=args.synthetic_topic_signal_scale,
                    sentiment_signal_scale=args.synthetic_sentiment_signal_scale,
                    noise_std=args.synthetic_noise_std,
                    seed=args.seed + 1001,
                    device=device,
                    **conjunctive_buffer_kwargs(args),
                )
                metrics = dict(metrics)
                metrics.update(
                    measure_synthetic_reconstruction(
                        trainer=trainer,
                        batch_source=recon_source,
                        num_batches=args.synthetic_eval_batches,
                    )
                )
                if getattr(trainer, "gradnorm", False):
                    metrics = dict(metrics)
                    metrics["gradnorm_task_names"] = trainer.gradnorm_task_names
                    metrics["gradnorm_alpha"] = trainer.gradnorm_alpha
                    metrics["gradnorm_lr"] = trainer.gradnorm_lr
                    metrics["gradnorm_weight_history"] = trainer.gradnorm_weight_history
                    metrics["gradnorm_final_w"] = (
                        [float(v) for v in trainer.gradnorm_weights.detach().cpu()]
                        if trainer.gradnorm_weights is not None
                        else None
                    )
                if getattr(trainer, "cagrad", False):
                    metrics = dict(metrics)
                    metrics["cagrad_c"] = trainer.cagrad_c
                    metrics["cagrad_task_names"] = trainer.cagrad_task_names
                    metrics["cagrad_history"] = trainer.cagrad_history
                if getattr(trainer, "adv_suppress", False):
                    metrics = dict(metrics)
                    metrics["adv_grl_lambda"] = trainer.adv_grl_lambda
                    metrics["adv_lr"] = trainer.adv_lr
                    metrics["adv_hidden"] = trainer.adv_hidden
                    metrics["adv_disc_final_loss"] = trainer.adv_disc_loss
                    metrics["adv_disc_final_acc"] = trainer.adv_disc_acc
                    metrics["adv_ref_disc_final_loss"] = trainer.adv_ref_disc_loss
                    metrics["adv_ref_disc_final_acc"] = trainer.adv_ref_disc_acc
                    metrics["adv_disc_history"] = trainer.adv_history
                if getattr(trainer, "leace_suppress", False):
                    metrics = dict(metrics)
                    metrics["leace_lambda"] = trainer.leace_lambda
                    metrics["leace_ref_lr"] = trainer.leace_ref_lr
                    metrics["leace_ref_hidden"] = trainer.leace_ref_hidden
                    metrics["leace_cross_cov_final_loss"] = trainer.leace_cross_cov_loss
                    metrics["leace_ref_disc_final_loss"] = trainer.leace_ref_disc_loss
                    metrics["leace_ref_disc_final_acc"] = trainer.leace_ref_disc_acc
                    metrics["leace_history"] = trainer.leace_history
                    metrics["leace_subspace"] = trainer.leace_subspace
                    if trainer.leace_subspace:
                        metrics["leace_subspace_refit_every"] = trainer.leace_subspace_refit_every
                        metrics["leace_subspace_ema_decay"] = trainer.leace_subspace_ema_decay
                        metrics["leace_subspace_eps"] = trainer.leace_subspace_eps
                        metrics["leace_subspace_last_refit_step"] = trainer.leace_subspace_last_refit_step
                if getattr(trainer, "bcd_alternate", False):
                    metrics = dict(metrics)
                    metrics["bcd_alternate"] = trainer.bcd_alternate
                    metrics["bcd_phase_steps"] = trainer.bcd_phase_steps
                    metrics["bcd_final_phase"] = trainer.bcd_current_phase
                    metrics["bcd_phase_counts"] = dict(trainer.bcd_phase_counts)
                    metrics["bcd_history"] = trainer.bcd_history
                if getattr(trainer, "cond_recon", False):
                    metrics = dict(metrics)
                    metrics["cond_recon"] = trainer.cond_recon
                    metrics["cond_recon_bias_norm"] = trainer.cond_recon_bias_norm
                if getattr(trainer, "swap_recon", False):
                    metrics = dict(metrics)
                    metrics["swap_recon"] = trainer.swap_recon
                    metrics["swap_recon_lambda"] = trainer.swap_recon_lambda
                    metrics["swap_recon_final_loss"] = trainer.swap_recon_loss
                    metrics["swap_recon_final_valid_frac"] = trainer.swap_recon_valid_frac
                if getattr(trainer, "rl_suppress", False):
                    metrics = dict(metrics)
                    metrics["rl_suppress"] = trainer.rl_suppress
                    metrics["rl_sigma"] = trainer.rl_sigma
                    metrics["rl_lambda"] = trainer.rl_lambda
                    metrics["rl_ref_lr"] = trainer.rl_ref_lr
                    metrics["rl_ref_hidden"] = trainer.rl_ref_hidden
                    metrics["rl_baseline_decay"] = trainer.rl_baseline_decay
                    metrics["rl_ref_disc_final_loss"] = trainer.rl_ref_disc_loss
                    metrics["rl_ref_disc_final_acc"] = trainer.rl_ref_disc_acc
                    metrics["rl_final_reward_baseline"] = trainer.rl_reward_baseline
                    metrics["rl_history"] = trainer.rl_history
                if getattr(trainer, "bcd_rl_combo", False):
                    # Mirrors both parents' blocks: the BCD schedule fields (idea 1) and the RL
                    # fields (idea 5), since this mode runs both halves.
                    metrics = dict(metrics)
                    metrics["bcd_rl_combo"] = trainer.bcd_rl_combo
                    metrics["bcd_phase_steps"] = trainer.bcd_phase_steps
                    metrics["bcd_final_phase"] = trainer.bcd_current_phase
                    metrics["bcd_phase_counts"] = dict(trainer.bcd_phase_counts)
                    metrics["bcd_history"] = trainer.bcd_history
                    metrics["rl_sigma"] = trainer.rl_sigma
                    metrics["rl_lambda"] = trainer.rl_lambda
                    metrics["rl_ref_lr"] = trainer.rl_ref_lr
                    metrics["rl_ref_hidden"] = trainer.rl_ref_hidden
                    metrics["rl_baseline_decay"] = trainer.rl_baseline_decay
                    metrics["rl_ref_disc_final_loss"] = trainer.rl_ref_disc_loss
                    metrics["rl_ref_disc_final_acc"] = trainer.rl_ref_disc_acc
                    metrics["rl_final_reward_baseline"] = trainer.rl_reward_baseline
                    metrics["rl_history"] = trainer.rl_history
                if getattr(trainer, "q_bottleneck_rank", 0) > 0:
                    metrics = dict(metrics)
                    metrics["q_bottleneck_rank"] = trainer.q_bottleneck_rank
                    metrics["q_bottleneck_recon_error"] = trainer.q_bottleneck_recon_error
                if name in q_grad_stat_sums and q_grad_stat_count > 0:
                    metrics = dict(metrics)
                    metrics["q_grad_stats_mean"] = {
                        key: val / q_grad_stat_count for key, val in q_grad_stat_sums[name].items()
                    }
                result_payload["results"][name] = metrics
                rank_print(
                    dist_env,
                    f"Synthetic probe [{name}] "
                    f"P(sent)={metrics['p_sentiment_acc']:.3f}, Q(sent)={metrics['q_sentiment_acc']:.3f}, "
                    f"P(topic)={metrics['p_topic_acc']:.3f}, Q(topic)={metrics['q_topic_acc']:.3f}; "
                    f"chance(sent)={metrics['sentiment_chance']:.3f}, chance(topic)={metrics['topic_chance']:.3f}",
                )

        if args.synthetic_conjunctive:
            # Part A readout. Runs over EVERY trainer (flat and Kron alike), since
            # the whole point is the flat-vs-Kron contrast as alpha varies. With
            # --synthetic_skip_training this is the random-init control arm at the
            # same alpha: identical construction, identical eval-activation seed
            # (seed+1001) and identical probe split seed (seed+3001), zero
            # optimizer steps. Per REPORT.md §9.4 that control -- not chance -- is
            # what every trained number must be read against.
            result_payload["conjunctive"] = {}
            for name, trainer in trainers:
                conj_source = SyntheticJointActivationBuffer(
                    activation_dim=args.activation_dim,
                    doc_batch_size=args.doc_batch_size,
                    num_topics=args.synthetic_num_topics,
                    num_sentiments=args.synthetic_num_sentiments,
                    topic_signal_scale=args.synthetic_topic_signal_scale,
                    sentiment_signal_scale=args.synthetic_sentiment_signal_scale,
                    noise_std=args.synthetic_noise_std,
                    seed=args.seed + 1001,
                    device=device,
                    **conjunctive_buffer_kwargs(args),
                )
                conj_metrics = evaluate_synthetic_conjunctive_probe(
                    trainer=trainer,
                    batch_source=conj_source,
                    num_batches=args.synthetic_eval_batches,
                    seed=args.seed + 3001,
                    device=device,
                    test_fraction=args.conjunctive_probe_test_fraction,
                )
                conj_metrics["is_random_init_control"] = bool(args.synthetic_skip_training)
                conj_metrics["dict_size"] = int(unwrap_model(trainer.ae).dict_size)
                conj_metrics["k"] = int(unwrap_model(trainer.ae).k.item())
                result_payload["conjunctive"][name] = conj_metrics
                rank_print(
                    dist_env,
                    f"Conjunctive probe [{name}] alpha={args.conjunctive_alpha:.3f} "
                    f"dense={conj_metrics['dense_conjunctive_acc']:.3f}, "
                    f"topk={conj_metrics['topk_conjunctive_acc']:.3f}, "
                    f"raw={conj_metrics['raw_conjunctive_acc']:.3f}; "
                    f"chance=0.500, random_init_control={conj_metrics['is_random_init_control']}",
                )

        out_path = Path(args.output_dir) / "synthetic_diagnostic_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)
        rank_print(dist_env, f"Saved synthetic diagnostics to {out_path}")


def train(args) -> None:
    if args.pilot and args.joint:
        raise ValueError("--pilot and --joint are mutually exclusive; choose one mode.")
    if args.synthetic_diagnostic and args.pilot:
        raise ValueError("--synthetic_diagnostic and --pilot are mutually exclusive.")

    dist_env = setup_distributed(args)
    try:
        workers_per_rank_target = max(1, args.total_cpu_workers // dist_env.world_size)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["RAYON_NUM_THREADS"] = str(workers_per_rank_target)
        t.set_num_threads(workers_per_rank_target)

        set_seed(args.seed + dist_env.rank)

        if t.cuda.is_available():
            device = f"cuda:{dist_env.local_rank}"
        else:
            device = "cpu"

        if args.synthetic_diagnostic:
            train_synthetic_diagnostic(args=args, dist_env=dist_env, device=device)
            return

        output_dir = Path(args.output_dir)

        categories = args.categories if args.categories else PREFERRED_CATEGORIES[: args.num_categories]

        # Build dataset metadata once, then all ranks read prepared files.
        if dist_env.is_main:
            metadata = prepare_amazon_reviews(
                output_dir=args.dataset_cache_dir,
                categories=categories,
                num_categories=args.num_categories,
                sample_per_category=args.sample_per_category,
                eval_fraction=args.eval_fraction,
                seed=args.seed,
                force_rebuild=args.force_rebuild_data,
            )
            rank_print(dist_env, format_count_table(metadata))

        if dist_env.enabled:
            dist_barrier(dist_env)

        train_ds, eval_ds, withheld_ds, metadata = load_prepared_datasets(args.dataset_cache_dir)

        model, submodule = build_model(args, dist_env, device)
        from dictionary_learning.labeled_buffer import LabeledActivationBuffer

        mode_info: dict[str, Any] = {}
        single_loader = None
        single_sampler = None
        single_buffer = None
        joint_loader = None
        joint_sampler = None
        joint_buffer = None
        pilot_loaders = None
        pilot_samplers = None
        pilot_buffers = None
        workers_per_rank = 0
        workers_per_rank_by_label: dict[str, int] = {}

        if args.pilot:
            trainers = create_pilot_trainers(args, device=device)
            pilot_loaders, pilot_samplers, workers_per_rank_by_label = build_pilot_balanced_dataloaders(
                train_ds=train_ds,
                args=args,
                dist_env=dist_env,
            )
            workers_per_rank = sum(workers_per_rank_by_label.values())

            pilot_buffers = {}
            for label_type in ("topic", "sentiment"):
                pilot_buffers[label_type] = LabeledActivationBuffer(
                    data=infinite_dataloader_batches(pilot_loaders[label_type]),
                    model=model,
                    submodule=submodule,
                    d_submodule=args.activation_dim,
                    io=args.io,
                    ctx_len=args.ctx_len,
                    device=device,
                    remove_bos=args.remove_bos,
                    add_special_tokens=True,
                    max_activation_norm_multiple=args.max_activation_norm_multiple,
                )

            mode_info = {
                "pilot": True,
                "pilot_h": args.pilot_h,
                "pilot_m": args.pilot_m,
                "pilot_n": args.pilot_n,
                "flat_pilot_supervision": "collect_only_full_vector",
                "flat_vector_dim": args.flat_dict_size,
                "balanced_train_batches": True,
                "balanced_batch_label_types": ["topic", "sentiment"],
                "pilot_label_types": ["topic", "sentiment"],
            }
            rank_print(
                dist_env,
                f"Pilot mode enabled: h={args.pilot_h}, m={args.pilot_m}, n={args.pilot_n}, "
                f"flat supervision=collect-only on full {args.flat_dict_size}-dim vector; "
                f"balanced doc batching active (workers/rank topic={workers_per_rank_by_label['topic']}, "
                f"sentiment={workers_per_rank_by_label['sentiment']})",
            )
        elif args.joint:
            trainers = create_joint_trainers(args, device=device)
            joint_loader, joint_sampler, workers_per_rank = build_joint_balanced_dataloader(
                train_ds=train_ds,
                args=args,
                dist_env=dist_env,
            )
            joint_buffer = LabeledActivationBuffer(
                data=infinite_dataloader_batches(joint_loader),
                model=model,
                submodule=submodule,
                d_submodule=args.activation_dim,
                io=args.io,
                ctx_len=args.ctx_len,
                device=device,
                remove_bos=args.remove_bos,
                add_special_tokens=True,
                max_activation_norm_multiple=args.max_activation_norm_multiple,
            )
            mode_info = {
                "joint": True,
                "joint_mode": args.joint_mode,
                "joint_h": args.joint_h,
                "joint_m": args.joint_m,
                "joint_n": args.joint_n,
                "joint_combine_rule": args.combine_rule,
                "joint_supervision": "supcon_collect_only"
                if args.joint_mode == "collect_only"
                else "dpo_cross_for_kron_collect_only_for_flat",
                "dpo_beta": args.dpo_beta,
                "dpo_beta_p_collect_sentiment": args.dpo_beta_p_collect_sentiment,
                "dpo_beta_p_contrast_topic": args.dpo_beta_p_contrast_topic,
                "dpo_beta_q_collect_topic": args.dpo_beta_q_collect_topic,
                "dpo_beta_q_contrast_sentiment": args.dpo_beta_q_contrast_sentiment,
                "dpo_weight_p_collect_sentiment": args.dpo_w_p_collect_sentiment,
                "dpo_weight_p_contrast_topic": args.dpo_w_p_contrast_topic,
                "dpo_weight_q_collect_topic": args.dpo_w_q_collect_topic,
                "dpo_weight_q_contrast_sentiment": args.dpo_w_q_contrast_sentiment,
                "contrast_start_frac": args.contrast_start_frac,
                "q_gradient_surgery": args.q_gradient_surgery,
                "q_gradient_surgery_include_recon": args.q_gradient_surgery_include_recon,
                "q_contrast_grad_norm_match": args.q_contrast_grad_norm_match,
                "q_contrast_grad_norm_match_cap": args.q_contrast_grad_norm_match_cap,
                "q_separate_grad_clip": args.q_separate_grad_clip,
                "lambda_orth": args.lambda_orth,
                "flat_joint_supervision": (
                    "sentiment_collect + topic_collect on full vector"
                    if args.joint_mode in {"collect_only", "dpo_cross"}
                    else "n/a"
                ),
                "balanced_train_batches": True,
                "balanced_batch_strategy": "topic_x_sentiment_cross_product",
            }
            rank_print(
                dist_env,
                f"Joint mode enabled: h={args.joint_h}, m={args.joint_m}, n={args.joint_n}, "
                f"joint_mode={args.joint_mode}, dpo_beta={args.dpo_beta}, "
                f"dpo_w=[p_cs={args.dpo_w_p_collect_sentiment}, p_ct={args.dpo_w_p_contrast_topic}, "
                f"q_ct={args.dpo_w_q_collect_topic}, q_cs={args.dpo_w_q_contrast_sentiment}], "
                f"contrast_start_frac={args.contrast_start_frac}, "
                f"q_gradient_surgery={args.q_gradient_surgery}, "
                f"q_gradient_surgery_include_recon={args.q_gradient_surgery_include_recon}, "
                f"q_contrast_grad_norm_match={args.q_contrast_grad_norm_match}, "
                f"q_contrast_grad_norm_match_cap={args.q_contrast_grad_norm_match_cap}, "
                f"q_separate_grad_clip={args.q_separate_grad_clip}, "
                f"lambda_orth={args.lambda_orth}, "
                f"combine_rule={args.combine_rule}, "
                f"balanced topic×sentiment batching active "
                f"(workers/rank={workers_per_rank})",
            )
        else:
            trainers = create_main_trainers(args, device=device)
            single_loader, single_sampler, workers_per_rank = build_dataloader(train_ds, args, dist_env)
            single_buffer = LabeledActivationBuffer(
                data=infinite_dataloader_batches(single_loader),
                model=model,
                submodule=submodule,
                d_submodule=args.activation_dim,
                io=args.io,
                ctx_len=args.ctx_len,
                device=device,
                remove_bos=args.remove_bos,
                add_special_tokens=True,
                max_activation_norm_multiple=args.max_activation_norm_multiple,
            )

        rank_print(
            dist_env,
            f"Rank setup: world_size={dist_env.world_size}, workers_per_rank={workers_per_rank}, "
            f"rayon_threads={os.environ['RAYON_NUM_THREADS']}",
        )

        trainers = filter_trainers(trainers, args.only_trainers)
        maybe_wrap_ddp(trainers, dist_env)

        rank_print(dist_env, f"Training {len(trainers)} configs: {[name for name, _ in trainers]}")

        save_dir = output_dir / "checkpoints"
        save_interval = args.save_every if args.save_every > 0 else None
        prev_topk_indices: dict[str, Optional[t.Tensor]] = {name: None for name, _ in trainers}
        debug_stats_file = None
        if args.debug_stats:
            output_dir.mkdir(parents=True, exist_ok=True)
            debug_stats_file = (output_dir / "debug_stats.jsonl").open("w", encoding="utf-8")

        for step in range(args.total_steps):
            if args.pilot:
                assert pilot_samplers is not None and pilot_loaders is not None and pilot_buffers is not None
                for label_type in ("topic", "sentiment"):
                    sampler = pilot_samplers[label_type]
                    loader = pilot_loaders[label_type]
                    if step % max(1, len(loader)) == 0:
                        sampler.set_epoch(step // max(1, len(loader)))
                batch_topic = next(pilot_buffers["topic"])
                batch_sentiment = next(pilot_buffers["sentiment"])
                batch_topic.activations = batch_topic.activations.to(dtype=t.float32)
                batch_sentiment.activations = batch_sentiment.activations.to(dtype=t.float32)
                batch_by_label = {"topic": batch_topic, "sentiment": batch_sentiment}
            elif args.joint:
                if joint_sampler is not None and step % max(1, len(joint_loader)) == 0:
                    joint_sampler.set_epoch(step // max(1, len(joint_loader)))
                assert joint_buffer is not None
                if args.grad_accum_steps > 1:
                    # REPORT1.md sec.19.7: N independently-sampled microbatches per step,
                    # each getting its own DPO-pairwise pairing (see kron_top_k.py's
                    # _update_grad_accum docstring for why this differs from one bigger batch).
                    accum_batches = [next(joint_buffer) for _ in range(args.grad_accum_steps)]
                    for b in accum_batches:
                        b.activations = b.activations.to(dtype=t.float32)
                    batch_by_label = {"default": accum_batches}
                else:
                    batch = next(joint_buffer)
                    batch.activations = batch.activations.to(dtype=t.float32)
                    batch_by_label = {"default": batch}
            else:
                if single_sampler is not None and step % max(1, len(single_loader)) == 0:
                    single_sampler.set_epoch(step // max(1, len(single_loader)))
                assert single_buffer is not None
                batch = next(single_buffer)
                batch.activations = batch.activations.to(dtype=t.float32)
                batch_by_label = {"default": batch}

            losses = {}
            batches_for_trainer: dict[str, Any] = {}
            for name, trainer in trainers:
                if args.pilot and hasattr(trainer, "label_type") and getattr(trainer, "label_type") in {"topic", "sentiment"}:
                    batch_for_trainer = batch_by_label[getattr(trainer, "label_type")]
                else:
                    batch_for_trainer = batch_by_label["default"]
                batches_for_trainer[name] = batch_for_trainer

                if isinstance(trainer, TopKTrainer) and not isinstance(trainer, FlatSupervisedTopKTrainer):
                    if isinstance(batch_for_trainer, list):
                        raise NotImplementedError(
                            f"--grad_accum_steps > 1 is only implemented for KronTopKTrainer "
                            f"(kron_joint), not '{name}' (a flat TopKTrainer, which extracts "
                            ".activations only). Restrict --only_trainers to kron_joint."
                        )
                    loss_val = trainer.update(step, batch_for_trainer.activations)
                else:
                    loss_val = trainer.update(step, batch_for_trainer)
                losses[name] = loss_val

            if dist_env.is_main and step % args.log_every == 0:
                loss_parts = " | ".join(f"{k}:{v:.4f}" for k, v in losses.items())
                print(f"step={step} | {loss_parts}", flush=True)
                if args.debug_stats and debug_stats_file is not None:
                    for name, trainer in trainers:
                        this_batch = batches_for_trainer[name]
                        stats, next_prev = collect_debug_stats(trainer, this_batch, prev_topk_indices[name])
                        prev_topk_indices[name] = next_prev
                        payload = {
                            "step": step,
                            "trainer": name,
                            "loss": losses[name],
                            "stats": stats,
                            "batch_label_hist": batch_label_histogram(this_batch),
                            "sup_loss": float(getattr(trainer, "sup_loss", float("nan"))),
                            "collect_supcon_loss": float(getattr(trainer, "collect_supcon_loss", float("nan"))),
                            "contrast_supcon_loss": float(getattr(trainer, "contrast_supcon_loss", float("nan"))),
                            "topic_dpo_loss": float(getattr(trainer, "topic_dpo_loss", float("nan"))),
                            "sentiment_dpo_loss": float(getattr(trainer, "sentiment_dpo_loss", float("nan"))),
                            "topic_valid_anchor_frac": float(getattr(trainer, "topic_valid_anchor_frac", float("nan"))),
                            "sentiment_valid_anchor_frac": float(
                                getattr(trainer, "sentiment_valid_anchor_frac", float("nan"))
                            ),
                            "p_collect_sentiment_dpo_loss": float(
                                getattr(trainer, "p_collect_sentiment_dpo_loss", float("nan"))
                            ),
                            "p_contrast_topic_dpo_loss": float(
                                getattr(trainer, "p_contrast_topic_dpo_loss", float("nan"))
                            ),
                            "q_collect_topic_dpo_loss": float(
                                getattr(trainer, "q_collect_topic_dpo_loss", float("nan"))
                            ),
                            "q_contrast_sentiment_dpo_loss": float(
                                getattr(trainer, "q_contrast_sentiment_dpo_loss", float("nan"))
                            ),
                            "p_collect_sentiment_valid_anchor_frac": float(
                                getattr(trainer, "p_collect_sentiment_valid_anchor_frac", float("nan"))
                            ),
                            "p_contrast_topic_valid_anchor_frac": float(
                                getattr(trainer, "p_contrast_topic_valid_anchor_frac", float("nan"))
                            ),
                            "q_collect_topic_valid_anchor_frac": float(
                                getattr(trainer, "q_collect_topic_valid_anchor_frac", float("nan"))
                            ),
                            "q_contrast_sentiment_valid_anchor_frac": float(
                                getattr(trainer, "q_contrast_sentiment_valid_anchor_frac", float("nan"))
                            ),
                            "contrast_scale": float(getattr(trainer, "contrast_scale", float("nan"))),
                            "p_orthogonality_penalty": float(
                                getattr(trainer, "p_orthogonality_penalty", float("nan"))
                            ),
                            "p_orthogonality_abs_cos": float(
                                getattr(trainer, "p_orthogonality_abs_cos", float("nan"))
                            ),
                            "q_pcgrad_conflict_dot": float(getattr(trainer, "q_pcgrad_conflict_dot", float("nan"))),
                            "q_pcgrad_conflict_active": float(
                                getattr(trainer, "q_pcgrad_conflict_active", float("nan"))
                            ),
                            "q_pcgrad_collect_grad_norm": float(
                                getattr(trainer, "q_pcgrad_collect_grad_norm", float("nan"))
                            ),
                            "q_pcgrad_contrast_grad_norm": float(
                                getattr(trainer, "q_pcgrad_contrast_grad_norm", float("nan"))
                            ),
                            "q_pcgrad_collect_proj_grad_norm": float(
                                getattr(trainer, "q_pcgrad_collect_proj_grad_norm", float("nan"))
                            ),
                            "q_pcgrad_contrast_proj_grad_norm": float(
                                getattr(trainer, "q_pcgrad_contrast_proj_grad_norm", float("nan"))
                            ),
                            "flat_collect_sentiment_dpo_loss": float(
                                getattr(trainer, "flat_collect_sentiment_dpo_loss", float("nan"))
                            ),
                            "flat_collect_topic_dpo_loss": float(
                                getattr(trainer, "flat_collect_topic_dpo_loss", float("nan"))
                            ),
                        }
                        payload.update(getattr(trainer, "probe_grad_norms", {}) or {})
                        debug_stats_file.write(json.dumps(payload) + "\n")
                    debug_stats_file.flush()

            if save_interval is not None and step > 0 and step % save_interval == 0:
                if dist_env.enabled:
                    dist_barrier(dist_env)
                if dist_env.is_main:
                    save_checkpoints(trainers, save_dir, step)
                if dist_env.enabled:
                    dist_barrier(dist_env)

        if dist_env.enabled:
            dist_barrier(dist_env)
        if dist_env.is_main:
            warmup_steps, decay_start, threshold_start_step = resolve_schedule_steps(args)
            save_checkpoints(trainers, save_dir, args.total_steps)
            with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "args": vars(args),
                        "resolved_schedule": {
                            "warmup_steps": warmup_steps,
                            "decay_start": decay_start,
                            "threshold_start_step": threshold_start_step,
                        },
                        "dataset_metadata": metadata,
                        "num_train_docs": len(train_ds),
                        "num_eval_docs": len(eval_ds),
                        "num_withheld_docs": len(withheld_ds),
                        "world_size": dist_env.world_size,
                        "doc_batch_size_per_rank": args.doc_batch_size,
                        "global_doc_batch_size": args.doc_batch_size * dist_env.world_size,
                        "workers_per_rank": workers_per_rank,
                        "workers_per_rank_by_label": workers_per_rank_by_label if args.pilot else None,
                        "total_dataloader_workers": workers_per_rank * dist_env.world_size,
                        "device": device,
                        "pilot_info": mode_info if args.pilot else None,
                        "joint_info": mode_info if args.joint else None,
                        "mode_info": mode_info,
                    },
                    f,
                    indent=2,
                )

        if single_loader is not None and hasattr(single_loader, "_iterator") and single_loader._iterator is not None:
            single_loader._iterator._shutdown_workers()
        if pilot_loaders is not None:
            for one_loader in pilot_loaders.values():
                if hasattr(one_loader, "_iterator") and one_loader._iterator is not None:
                    one_loader._iterator._shutdown_workers()
        if joint_loader is not None and hasattr(joint_loader, "_iterator") and joint_loader._iterator is not None:
            joint_loader._iterator._shutdown_workers()
        if debug_stats_file is not None:
            debug_stats_file.close()

    finally:
        cleanup_distributed(dist_env)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train flat SAE + KronSAE with DDP")

    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--io", type=str, default="out", choices=["in", "out"])
    parser.add_argument("--activation_dim", type=int, default=1024)

    parser.add_argument("--flat_dict_size", type=int, default=16384)
    parser.add_argument("--kron_h", type=int, default=256)
    parser.add_argument("--kron_m", type=int, default=8)
    parser.add_argument("--kron_n", type=int, default=8)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--pilot", action="store_true", help="Run pilot 4-config single-label collect-vs-contrast grid.")
    parser.add_argument("--joint", action="store_true", help="Run joint 2-config grid (flat_joint, kron_joint).")
    parser.add_argument(
        "--joint_mode",
        type=str,
        default="collect_only",
        choices=["collect_only", "dpo_cross"],
        help="Joint experiment mode: collect_only first, then dpo_cross.",
    )
    parser.add_argument("--pilot_h", type=int, default=128)
    parser.add_argument("--pilot_m", type=int, default=8)
    parser.add_argument("--pilot_n", type=int, default=16)
    parser.add_argument("--joint_h", type=int, default=128)
    parser.add_argument("--joint_m", type=int, default=8)
    parser.add_argument("--joint_n", type=int, default=16)

    parser.add_argument("--total_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--decay_start", type=int, default=None)
    parser.add_argument("--decay_start_frac", type=float, default=0.8)
    parser.add_argument("--threshold_start_step", type=int, default=None)
    parser.add_argument("--threshold_start_frac", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--auxk_alpha", type=float, default=1.0 / 32.0)

    parser.add_argument("--use_supervision", action="store_true", help="Unused switch: grid always includes supervised configs")
    parser.add_argument("--lambda_sup", type=float, default=0.5)
    parser.add_argument("--lambda_sup_warmup_frac", type=float, default=0.15)
    parser.add_argument("--supcon_temperature", type=float, default=0.1)
    parser.add_argument("--dpo_beta", type=float, default=5.0)
    parser.add_argument("--dpo_beta_p_collect_sentiment", type=float, default=None)
    parser.add_argument("--dpo_beta_p_contrast_topic", type=float, default=None)
    parser.add_argument("--dpo_beta_q_collect_topic", type=float, default=None)
    parser.add_argument("--dpo_beta_q_contrast_sentiment", type=float, default=None)
    parser.add_argument("--dpo_w_p_collect_sentiment", type=float, default=1.0)
    parser.add_argument("--dpo_w_p_contrast_topic", type=float, default=1.0)
    parser.add_argument("--dpo_w_q_collect_topic", type=float, default=1.0)
    parser.add_argument("--dpo_w_q_contrast_sentiment", type=float, default=1.0)
    parser.add_argument(
        "--contrast_start_frac",
        type=float,
        default=0.0,
        help="For kron joint_dpo_cross: step fraction before contrast starts; after start, contrast ramps with lambda_sup_warmup_frac.",
    )
    parser.add_argument(
        "--q_gradient_surgery",
        action="store_true",
        help="Enable PCGrad-style conflict projection between Q collect-topic and Q contrast-sentiment gradients.",
    )
    parser.add_argument(
        "--q_gradient_surgery_include_recon",
        action="store_true",
        help="Extend --q_gradient_surgery to a 3-way PCGrad over Q's collect, contrast and reconstruction gradients.",
    )
    parser.add_argument(
        "--q_contrast_grad_norm_match",
        action="store_true",
        help="Rescale Q's contrast-sentiment gradient to match the norm of (reconstruction + collect) gradient on Q params.",
    )
    parser.add_argument(
        "--q_contrast_grad_norm_match_cap",
        type=float,
        default=0.0,
        help="Cap the multiplicative boost applied by --q_contrast_grad_norm_match to this factor of contrast's raw norm (0 = uncapped full match).",
    )
    parser.add_argument(
        "--q_separate_grad_clip",
        action="store_true",
        help="Clip Q branch params (q_proj, q_bias) with their own norm-1.0 budget, excluded from the global clip.",
    )
    parser.add_argument(
        "--grad_norm_probe_every",
        type=int,
        default=0,
        help=(
            "If >0, every N steps measure (measurement-only, does not alter param.grad) the "
            "per-loss-term gradient norms on the P and Q branch parameters."
        ),
    )
    parser.add_argument(
        "--lambda_orth",
        type=float,
        default=0.0,
        help="Weight for P-branch orthogonality penalty |cos(sent_dir, topic_dir)|^2.",
    )
    parser.add_argument(
        "--lambda_recon",
        type=float,
        default=1.0,
        help="Weight on the reconstruction (SAE) loss term in the kron joint total loss.",
    )
    parser.add_argument(
        "--gradnorm",
        action="store_true",
        help=(
            "Enable GradNorm (Chen et al., 2018) adaptive per-task loss balancing on the kron "
            "joint_dpo_cross trainer. Learns one weight per task over {recon, p_collect_sentiment, "
            "p_contrast_topic, q_collect_topic, q_contrast_sentiment}, superseding --lambda_recon, "
            "--lambda_sup, --lambda_sup_warmup_frac and --dpo_w_* for those terms."
        ),
    )
    parser.add_argument("--gradnorm_alpha", type=float, default=1.5, help="GradNorm asymmetry exponent alpha.")
    parser.add_argument("--gradnorm_lr", type=float, default=0.025, help="Adam lr for the GradNorm task weights.")
    parser.add_argument(
        "--gradnorm_log_every",
        type=int,
        default=50,
        help="Record the GradNorm weight trajectory every N steps (0 disables).",
    )
    parser.add_argument(
        "--cagrad",
        action="store_true",
        help=(
            "Enable CAGrad (Liu et al., NeurIPS 2021, arXiv:2110.14048) conflict-averse gradient "
            "combination on the kron joint_dpo_cross trainer's Q branch (q_proj/q_bias). The "
            "CAGrad direction over {recon, q_collect_topic, q_contrast_sentiment} replaces the raw "
            "sum of those three (weighted) task gradients on Q's encoder; P's terms, the decoder "
            "and all loss weights are unchanged. Mutually exclusive with --gradnorm, "
            "--q_gradient_surgery, --q_contrast_grad_norm_match and --adv_suppress."
        ),
    )
    parser.add_argument(
        "--cagrad_c",
        type=float,
        default=0.5,
        help=(
            "CAGrad trust-region coefficient c in [0,1): the update stays within "
            "||d - g_0|| <= c*||g_0|| of the average gradient. Paper's typical range 0.1-0.5; "
            "c=0 reduces exactly to the unmodified summed gradient."
        ),
    )
    parser.add_argument(
        "--cagrad_log_every",
        type=int,
        default=50,
        help="Record the CAGrad weight/worst-case-improvement trajectory every N steps (0 disables).",
    )
    parser.add_argument(
        "--q_recon_grad_scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to the gradient that the reconstruction loss sends back into "
            "Q's encoder (q_proj/q_bias), via a value-preserving gradient-scaling op on the "
            "decode path only. 1.0 (default) = unchanged behavior; 0.0 = Q's encoder fully "
            "shielded from reconstruction gradient. P, the decoder, and Q's own supervision "
            "terms are unaffected."
        ),
    )
    parser.add_argument(
        "--adv_suppress",
        action="store_true",
        help=(
            "Enable adversarial (gradient-reversal, Ganin & Lempitsky 2015) suppression of "
            "sentiment in Q's doc-pooled representation. A small MLP discriminator predicts "
            "sentiment from Q and is trained by its own Adam optimizer; a gradient-reversal "
            "layer between Q and the discriminator trains Q to fool it. Requires "
            "--joint_mode dpo_cross; REPLACES Q's DPO contrast-sentiment term."
        ),
    )
    parser.add_argument(
        "--adv_grl_lambda",
        type=float,
        default=1.0,
        help="Gradient-reversal scale: the adversarial gradient into Q is multiplied by -lambda.",
    )
    parser.add_argument("--adv_lr", type=float, default=1e-3, help="Adam lr for the adversarial discriminator.")
    parser.add_argument("--adv_hidden", type=int, default=128, help="Hidden width of the discriminator MLP.")
    parser.add_argument(
        "--leace_suppress",
        action="store_true",
        help=(
            "Enable LEACE-inspired (Belrose et al. 2023) suppression of sentiment in Q's "
            "doc-pooled representation: a differentiable penalty on the squared batch "
            "cross-covariance between Q and one-hot(sentiment), minimized directly via "
            "ordinary gradient descent on the encoder -- no adversary, no gradient-reversal "
            "layer. Replaces --adv_suppress's GRL minimax mechanism, which this project found "
            "plateaus 0.08-0.14 above chance regardless of tuning (REPORT1.md sec.10-11). "
            "Requires --joint_mode dpo_cross; REPLACES Q's DPO contrast-sentiment term; "
            "mutually exclusive with --adv_suppress, --gradnorm, --cagrad."
        ),
    )
    parser.add_argument(
        "--leace_lambda",
        type=float,
        default=1.0,
        help="Weight on the LEACE-inspired cross-covariance penalty term.",
    )
    parser.add_argument(
        "--leace_ref_lr", type=float, default=1e-3,
        help="Adam lr for the measurement-only reference probe (never touched by the suppression loss).",
    )
    parser.add_argument(
        "--leace_ref_hidden", type=int, default=128,
        help="Hidden width of the measurement-only reference probe MLP.",
    )
    parser.add_argument(
        "--leace_subspace",
        action="store_true",
        help=(
            "Subspace-restricted variant of --leace_suppress (REPORT1.md sec.17 follow-up 1): "
            "instead of penalizing the raw, unwhitened batch cross-covariance, periodically "
            "refit a closed-form whitening matrix from an EMA of the population covariance and "
            "penalize the whitened cross-covariance -- the differentiable analogue of the "
            "closed-form LEACE fit in eval_concept_erasure_control.py. Requires --leace_suppress."
        ),
    )
    parser.add_argument(
        "--leace_subspace_refit_every", type=int, default=200,
        help="Steps between refitting the EMA-based whitening matrix.",
    )
    parser.add_argument(
        "--leace_subspace_ema_decay", type=float, default=0.99,
        help="EMA decay for the running population-covariance estimate used to fit whitening.",
    )
    parser.add_argument(
        "--leace_subspace_eps", type=float, default=1e-4,
        help="Eigenvalue floor for the whitening matrix's inverse-sqrt (numerical stability).",
    )
    parser.add_argument(
        "--bcd_alternate",
        action="store_true",
        help=(
            "REPORT1.md sec.19 idea 1: three-way block-coordinate/ALS-style alternation over "
            "P/Q/decoder instead of updating all three simultaneously. Cycles "
            "P -> Q -> decoder every --bcd_phase_steps steps from step 0 (no joint warmup), "
            "freezing the other two blocks' parameters in each phase. The losses are unchanged "
            "from the joint_dpo_cross baseline: only the active block's own DPO terms are added "
            "(the decoder phase uses reconstruction alone). Requires "
            "--joint_mode dpo_cross; mutually exclusive with gradnorm/cagrad/adv/leace."
        ),
    )
    parser.add_argument(
        "--bcd_phase_steps", type=int, default=200,
        help="Number of steps each BCD phase (P, Q, decoder) holds before cycling to the next.",
    )
    parser.add_argument(
        "--bcd_rl_combo",
        action="store_true",
        help=(
            "REPORT1.md idea 6, composes idea 1's P/Q/decoder alternation with idea 5's "
            "REINFORCE suppression during Q's phase. Self-sufficient mode switch: do NOT also "
            "pass --bcd_alternate/--rl_suppress (it is mutually exclusive with both). Runs the "
            "same P -> Q -> decoder cycle as --bcd_alternate (--bcd_phase_steps), but in Q's "
            "phase the sentiment-suppression signal comes from the REINFORCE term "
            "(--rl_sigma/--rl_lambda/--rl_ref_lr/--rl_ref_hidden/--rl_baseline_decay) instead of "
            "Q's DPO contrast-sentiment term; the RL term is skipped in the P and decoder phases, "
            "where Q is frozen. Requires --joint_mode dpo_cross; mutually exclusive with "
            "bcd_alternate/rl_suppress/adv_suppress/leace_suppress/swap_recon/gradnorm/cagrad."
        ),
    )
    parser.add_argument(
        "--grad_accum_steps", type=int, default=1,
        help=(
            "REPORT1.md sec.19.7: accumulate gradients over N independently-sampled "
            "microbatches (each of size --doc_batch_size) before one optimizer step, to "
            "replicate the effective per-step data volume of an N-GPU DDP run (e.g. this "
            "project's original torchrun --nproc_per_node=4 runs) on a single GPU. Each "
            "microbatch forms its own DPO-pairwise/contrastive pairs independently -- matching "
            "DDP's per-rank-local-pairing semantics -- rather than pooling into one larger "
            "batch, which would change the pairing statistics. 1 (default) = off, ordinary "
            "single-batch-per-step training. Only supported for kron_joint with "
            "joint_mode=dpo_cross and none of q_gradient_surgery/q_contrast_grad_norm_match/"
            "gradnorm/cagrad/adv_suppress/grad_norm_probe_every active (raises otherwise)."
        ),
    )
    parser.add_argument(
        "--cond_recon",
        action="store_true",
        help=(
            "REPORT1.md idea 2, label-conditioned decoder (conditional reconstruction). Adds a "
            "learned bias-free linear head one_hot(sentiment_label) -> activation_dim whose "
            "output is added to the decoder's x_hat before the reconstruction loss, so the "
            "decoder gets the ground-truth sentiment label directly and reconstruction no longer "
            "*needs* Q to encode sentiment on its own. Purely additive on top of the "
            "joint_dpo_cross DPO losses (nothing is replaced), and applied only to the x_hat used "
            "by the loss -- the pooled p_docs/q_docs fed to the probes and DPO terms are "
            "untouched, and no label is used at eval time. Requires --joint_mode dpo_cross."
        ),
    )
    parser.add_argument(
        "--swap_recon",
        action="store_true",
        help=(
            "REPORT1.md idea 3, swap-and-reconstruct self-supervision, replaces Q's DPO "
            "contrast-sentiment term. Pairs each in-batch doc i with a doc j of the same topic "
            "but different sentiment, rebuilds i's reconstruction from P_i together with Q_j "
            "through the unchanged combine rule / top-k / decoder, and penalizes "
            "||x_i - x_hat_swapped_i||^2. If Q leaks sentiment the swap corrupts the "
            "reconstruction, so minimizing this pressures Q to drop it -- a purely "
            "self-supervised consistency signal with no repulsion or adversarial term. Zeroes "
            "the q_contrast_sentiment DPO weight exactly like --adv_suppress / --leace_suppress "
            "and adds its own loss term instead; P's terms, Q's collect-topic term and "
            "reconstruction are unchanged. Requires --joint_mode dpo_cross; mutually exclusive "
            "with adv_suppress/leace_suppress/gradnorm/cagrad."
        ),
    )
    parser.add_argument(
        "--swap_recon_lambda", type=float, default=1.0,
        help="Coefficient on the --swap_recon loss term.",
    )
    parser.add_argument(
        "--q_bottleneck_rank", type=int, default=0,
        help=(
            "REPORT1.md idea 4, trainable rank-restricted bottleneck on Q's full representation. "
            "0 disables it. When > 0, Q's flattened h*n activations are pushed through a "
            "trainable rank-r linear autoencoder (relu(up(down(q))), both layers bias-free) "
            "before every downstream use -- the combine/decode path, the pooled q_docs behind "
            "the DPO terms, and the evaluation probes. A hard architectural capacity limit "
            "instead of a soft loss-based push, testing whether restricting how much Q *can* "
            "represent reduces sentiment leakage where LEACE/GRL/DPO-contrast did not. Purely "
            "additive: every joint_dpo_cross DPO term is kept exactly as-is. Unlike the other "
            "additive mechanisms this is NOT a no-op at step 0 (an identity map is unreachable "
            "below full rank). Requires --joint_mode dpo_cross; untested in combination with "
            "swap_recon/leace_suppress/adv_suppress/gradnorm/cagrad."
        ),
    )
    parser.add_argument(
        "--rl_suppress",
        action="store_true",
        help=(
            "REPORT1.md idea 5, REINFORCE-based suppression via a reference discriminator, "
            "replacing Q's DPO contrast-sentiment term. Treats Q's doc-pooled representation as "
            "the mean of a fixed-variance Gaussian policy: Gaussian exploration noise "
            "(--rl_sigma) is added to form an action, a reference discriminator is trained on "
            "the noised action to decode sentiment, and its cross-entropy is used as the reward "
            "(high loss = sentiment hard to read = good suppression). Q's encoder is then "
            "updated by a score-function (REINFORCE) estimator with an EMA reward baseline, so "
            "no gradient flows through the discriminator at all -- unlike --adv_suppress "
            "(gradient reversal through the discriminator) or --leace_suppress (gradient through "
            "a closed-form covariance statistic). The noise is confined to this term: the decode "
            "path, the other DPO terms and the evaluation probes all see the deterministic Q. "
            "Requires --joint_mode dpo_cross; mutually exclusive with "
            "adv_suppress/leace_suppress/swap_recon/gradnorm/cagrad."
        ),
    )
    parser.add_argument(
        "--rl_sigma", type=float, default=0.1,
        help=(
            "Std of the Gaussian exploration noise added to Q's doc-pooled representation to "
            "form the policy's action. Also sets the score-function scale (grad log pi = "
            "eps/sigma^2), so smaller values mean larger policy gradients."
        ),
    )
    parser.add_argument(
        "--rl_lambda", type=float, default=1.0,
        help="Coefficient on the REINFORCE suppression term.",
    )
    parser.add_argument(
        "--rl_ref_lr", type=float, default=1e-3,
        help="Adam lr for the reward reference discriminator (trained on the detached action).",
    )
    parser.add_argument(
        "--rl_ref_hidden", type=int, default=128,
        help="Hidden width of the reward reference discriminator MLP.",
    )
    parser.add_argument(
        "--rl_baseline_decay", type=float, default=0.99,
        help=(
            "EMA decay for the scalar reward baseline subtracted from the reward to form the "
            "advantage (REINFORCE variance reduction)."
        ),
    )
    parser.add_argument(
        "--decouple_recon_mode",
        type=str,
        default="none",
        choices=["none", "stopgrad"],
        help=(
            "Decouple reconstruction from the branch that must suppress a label. "
            "'none' (default) = unchanged behavior. 'stopgrad' = value-preserving "
            "stop-gradient on each branch's suppressed-label class-mean subspace along "
            "the decode path only, so reconstruction cannot pull that subspace back."
        ),
    )
    parser.add_argument(
        "--decouple_stopgrad_strength",
        type=float,
        default=1.0,
        help="Fraction of the suppressed-label subspace gradient removed from the decode path (0=off, 1=full).",
    )
    parser.add_argument(
        "--decouple_stopgrad_branches",
        type=str,
        default="q",
        choices=["q", "p", "both"],
        help="Which branch(es) get the decode-path stop-gradient (Q suppresses sentiment, P suppresses topic).",
    )
    parser.add_argument(
        "--decouple_dir_ema",
        type=float,
        default=0.9,
        help="EMA momentum for the per-class means defining the suppressed-label subspace.",
    )
    parser.add_argument(
        "--flat_sup_mode",
        type=str,
        default="split_half",
        choices=["split_half", "joint", "pilot_collect_full", "joint_dpo_full"],
    )
    parser.add_argument("--flat_sup_grad_scale", type=float, default=0.3)

    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument(
        "--doc_batch_size",
        type=int,
        default=32,
        help="Document batch size per rank/process (global batch = doc_batch_size * world_size).",
    )
    parser.add_argument("--remove_bos", action="store_true")
    parser.add_argument("--max_activation_norm_multiple", type=int, default=None)

    parser.add_argument("--dataset_cache_dir", type=str, default=str(ROOT / "data" / "amazon_reviews"))
    parser.add_argument("--num_categories", type=int, default=6)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--sample_per_category", type=int, default=50_000)
    parser.add_argument("--eval_fraction", type=float, default=0.02)
    parser.add_argument("--force_rebuild_data", action="store_true")

    parser.add_argument(
        "--total_cpu_workers",
        type=int,
        default=20,
        help="Total dataloader workers across all ranks (split roughly evenly per rank).",
    )

    parser.add_argument("--output_dir", type=str, default=str(ROOT / "runs" / "kron_sae"))
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--debug_stats", action="store_true")
    parser.add_argument("--only_trainers", nargs="*", default=None)

    parser.add_argument(
        "--combine_rule",
        type=str,
        default="mand",
        choices=["mand", "mor", "concat", "mnand", "mnor"],
        help="Combine rule for the kron_joint trainer's KronSAE (default mand = prior behavior).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic_skip_training",
        action="store_true",
        help=(
            "Synthetic diagnostic random-init control: build the trainers exactly as for a "
            "full run but perform zero optimizer steps, then run the identical eval path."
        ),
    )
    parser.add_argument(
        "--synthetic_diagnostic",
        action="store_true",
        help="Bypass LM/data pipeline and train joint_dpo_cross on synthetic orthogonal-label activations.",
    )
    parser.add_argument("--synthetic_num_topics", type=int, default=6)
    parser.add_argument("--synthetic_num_sentiments", type=int, default=2)
    parser.add_argument("--synthetic_topic_signal_scale", type=float, default=3.0)
    parser.add_argument("--synthetic_sentiment_signal_scale", type=float, default=3.0)
    parser.add_argument("--synthetic_noise_std", type=float, default=1.0)
    parser.add_argument("--synthetic_eval_batches", type=int, default=64)
    parser.add_argument("--synthetic_probe_epochs", type=int, default=80)
    parser.add_argument("--synthetic_probe_lr", type=float, default=1e-2)
    parser.add_argument("--synthetic_probe_batch_size", type=int, default=128)
    parser.add_argument(
        "--synthetic_probe_snapshot_batches", type=int, default=0,
        help=(
            "Eval-batch count for --synthetic_probe_every's mid-training snapshots. 0 (default) "
            "= use a quarter of --synthetic_eval_batches (cost-bounded); set equal to "
            "--synthetic_eval_batches to match the final-step read exactly, isolating a real "
            "training-step effect from the snapshot simply having less eval data."
        ),
    )
    parser.add_argument(
        "--synthetic_intervention_alphas", type=str, default="",
        help=(
            "Comma-separated intervention strengths (in std-devs of Q's sentiment-direction "
            "projection) for the patch-and-generate causal steering eval (REPORT1.md sec.19.3). "
            "Empty (default) = skip. Example: '-4,-2,-1,0,1,2,4'. Tests whether pushing Q's "
            "sentiment axis causally moves the DECODED reconstruction, as read by an external "
            "classifier trained on genuine raw activations -- a different property than probe "
            "accuracy on Q itself (erasure) tests."
        ),
    )
    parser.add_argument(
        "--synthetic_probe_every", type=int, default=0,
        help=(
            "Run a mid-training probe snapshot every N steps (0 = disabled, only the final-step "
            "probe runs). REPORT1.md sec.19 follow-up: distinguishes 'still moving, needs more "
            "steps' from 'plateaued, this mechanism doesn't do anything' -- a single final-step "
            "read can't tell the two apart. Uses a smaller eval batch count than the end-of-run "
            "probe to bound the added cost."
        ),
    )

    # --- Part A: alpha-parameterized conjunctive target (additive; off by default) ---
    parser.add_argument(
        "--synthetic_conjunctive",
        action="store_true",
        help=(
            "Add a third, alpha-parameterized held-out target y_alpha built from two new "
            "orthonormal primitive directions, and probe every trainer's latents for it. "
            "Sentiment/topic remain as background signal. Combine with --synthetic_skip_training "
            "for the random-init control arm at the same alpha."
        ),
    )
    parser.add_argument(
        "--conjunctive_alpha",
        type=float,
        default=0.5,
        help=(
            "Interpolation between a purely linear target (0.0: y=sign(p_A)) and a purely "
            "conjunctive/XOR-like one (1.0: y=sign(p_A*p_B)). One value per run; sweep it "
            "across runs and aggregate with src/aggregate_conjunctive_alpha_sweep.py."
        ),
    )
    parser.add_argument(
        "--conjunctive_signal_scale",
        type=float,
        default=1.0,
        help="Std of the p_A/p_B coefficients injected along d_A/d_B.",
    )
    parser.add_argument(
        "--conjunctive_balance",
        type=str,
        default="on",
        choices=["on", "off"],
        help=(
            "Confound control: hold the marginal frequency of y_alpha at ~0.5 at every alpha "
            "by subtracting the calibration-sample median of the score before thresholding."
        ),
    )
    parser.add_argument(
        "--conjunctive_calib_batches",
        type=int,
        default=16,
        help="Batches used to empirically fit the product rescale s and the balance offset.",
    )
    parser.add_argument(
        "--conjunctive_probe_test_fraction",
        type=float,
        default=0.3,
        help="Held-out fraction for the standardized y_alpha probe.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
