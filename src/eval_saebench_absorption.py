"""Run SAEBench absorption on local flat/Kron checkpoints via custom SAE adapters."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from dictionary_learning.dictionary_kron import KronAutoEncoderTopK  # noqa: E402
from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402

import sae_bench.custom_saes.base_sae as base_sae  # noqa: E402
import sae_bench.evals.absorption.main as absorption_main  # noqa: E402
from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig  # noqa: E402
from sae_bench.sae_bench_utils.general_utils import str_to_dtype  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class LocalCheckpointSpec:
    name: str
    dict_class: str
    cfg: dict[str, Any]
    ckpt_path: Path


class FlatTopKSAEAdapter(base_sae.BaseSAE):
    """Adapter from local AutoEncoderTopK checkpoints to SAEBench BaseSAE."""

    def __init__(
        self,
        inner: AutoEncoderTopK,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__(
            d_in=inner.activation_dim,
            d_sae=inner.dict_size,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
            hook_name=f"blocks.{hook_layer}.hook_resid_post",
        )
        self.inner = inner.to(device=device, dtype=dtype).eval()
        self._sync_from_inner()
        self.cfg.architecture = "topk"

    @torch.no_grad()
    def _sync_from_inner(self) -> None:
        self.W_enc.data.copy_(self.inner.encoder.weight.data.T.to(self.W_enc.dtype))
        self.W_dec.data.copy_(self.inner.decoder.weight.data.T.to(self.W_dec.dtype))
        self.b_enc.data.copy_(self.inner.encoder.bias.data.to(self.b_enc.dtype))
        self.b_dec.data.copy_(self.inner.b_dec.data.to(self.b_dec.dtype))

        # SAEBench expects unit-norm decoder rows.
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor):
        orig_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        acts = self.inner.encode(flat)
        return acts.reshape(*orig_shape, acts.shape[-1])

    def decode(self, feature_acts: torch.Tensor):
        orig_shape = feature_acts.shape[:-1]
        flat = feature_acts.reshape(-1, feature_acts.shape[-1])
        recon = self.inner.decode(flat)
        return recon.reshape(*orig_shape, recon.shape[-1])

    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))


class KronTopKSAEAdapter(base_sae.BaseSAE):
    """Adapter from local KronAutoEncoderTopK checkpoints to SAEBench BaseSAE."""

    def __init__(
        self,
        inner: KronAutoEncoderTopK,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__(
            d_in=inner.activation_dim,
            d_sae=inner.dict_size,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
            hook_name=f"blocks.{hook_layer}.hook_resid_post",
        )
        self.inner = inner.to(device=device, dtype=dtype).eval()
        self._sync_from_inner()
        self.cfg.architecture = "kron_topk"

    @torch.no_grad()
    def _sync_from_inner(self) -> None:
        # Decoder rows correspond to SAE features.
        self.W_dec.data.copy_(self.inner.decoder.weight.data.T.to(self.W_dec.dtype))

        # Kron has no single linear encoder matrix; use decoder transpose as a surrogate
        # metadata matrix for SAEBench's optional reporting fields.
        self.W_enc.data.copy_(self.W_dec.data.T)

        self.b_enc.data.zero_()
        self.b_dec.data.copy_(self.inner.b_dec.data.to(self.b_dec.dtype))

        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor):
        orig_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        acts = self.inner.encode(flat)
        return acts.reshape(*orig_shape, acts.shape[-1])

    def decode(self, feature_acts: torch.Tensor):
        orig_shape = feature_acts.shape[:-1]
        flat = feature_acts.reshape(-1, feature_acts.shape[-1])
        recon = self.inner.decode(flat)
        return recon.reshape(*orig_shape, recon.shape[-1])

    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))


def load_checkpoint_spec(run_dir: Path, checkpoint_name: str, checkpoint_step: int) -> LocalCheckpointSpec:
    cfg_path = run_dir / "checkpoints" / checkpoint_name / "trainer_config.json"
    ckpt_path = run_dir / "checkpoints" / checkpoint_name / f"ae_step_{checkpoint_step}.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing trainer config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return LocalCheckpointSpec(
        name=checkpoint_name,
        dict_class=str(cfg["dict_class"]),
        cfg=cfg,
        ckpt_path=ckpt_path,
    )


def build_adapter(
    spec: LocalCheckpointSpec,
    model_name: str,
    hook_layer: int,
    device: torch.device,
    dtype: torch.dtype,
) -> base_sae.BaseSAE:
    state = torch.load(spec.ckpt_path, map_location="cpu")

    if spec.dict_class == "AutoEncoderTopK":
        inner = AutoEncoderTopK(
            activation_dim=int(spec.cfg["activation_dim"]),
            dict_size=int(spec.cfg["dict_size"]),
            k=int(spec.cfg["k"]),
        )
        inner.load_state_dict(state)
        return FlatTopKSAEAdapter(
            inner=inner,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
        )

    if spec.dict_class == "KronAutoEncoderTopK":
        inner = KronAutoEncoderTopK(
            activation_dim=int(spec.cfg["activation_dim"]),
            h=int(spec.cfg["h"]),
            m=int(spec.cfg["m"]),
            n=int(spec.cfg["n"]),
            k=int(spec.cfg["k"]),
            combine_rule=str(spec.cfg.get("combine_rule", "mand")),
        )
        inner.load_state_dict(state)
        return KronTopKSAEAdapter(
            inner=inner,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
        )

    raise ValueError(f"Unsupported dict_class={spec.dict_class} for {spec.name}")


def summarize_absorption_results(results_dict: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for full_name, payload in results_dict.items():
        mean_metrics = payload["eval_result_metrics"]["mean"]
        cfg = payload["eval_config"]
        rows.append(
            {
                "sae_result_key": full_name,
                "mean_absorption_fraction_score": float(mean_metrics["mean_absorption_fraction_score"]),
                "mean_full_absorption_score": float(mean_metrics["mean_full_absorption_score"]),
                "mean_num_split_features": float(mean_metrics["mean_num_split_features"]),
                "std_dev_absorption_fraction_score": float(mean_metrics["std_dev_absorption_fraction_score"]),
                "std_dev_full_absorption_score": float(mean_metrics["std_dev_full_absorption_score"]),
                "std_dev_num_split_features": float(mean_metrics["std_dev_num_split_features"]),
                "prompt_template": cfg["prompt_template"],
                "prompt_token_pos": cfg["prompt_token_pos"],
            }
        )

    rows.sort(key=lambda r: r["sae_result_key"])
    return rows


def load_rows_from_existing_eval_jsons(output_dir: Path) -> list[dict[str, Any]]:
    """Read cached SAEBench eval JSONs and build summary rows."""
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*_eval_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mean_metrics = payload["eval_result_metrics"]["mean"]
        cfg = payload["eval_config"]
        rows.append(
            {
                "sae_result_key": path.name.replace("_eval_results.json", ""),
                "mean_absorption_fraction_score": float(mean_metrics["mean_absorption_fraction_score"]),
                "mean_full_absorption_score": float(mean_metrics["mean_full_absorption_score"]),
                "mean_num_split_features": float(mean_metrics["mean_num_split_features"]),
                "std_dev_absorption_fraction_score": float(mean_metrics["std_dev_absorption_fraction_score"]),
                "std_dev_full_absorption_score": float(mean_metrics["std_dev_full_absorption_score"]),
                "std_dev_num_split_features": float(mean_metrics["std_dev_num_split_features"]),
                "prompt_template": cfg["prompt_template"],
                "prompt_token_pos": cfg["prompt_token_pos"],
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print(
        "sae | mean_absorption_fraction | mean_full_absorption | mean_num_split_features",
        flush=True,
    )
    print("-" * 96, flush=True)
    for r in rows:
        print(
            f"{r['sae_result_key']} | {r['mean_absorption_fraction_score']:.6f} | "
            f"{r['mean_full_absorption_score']:.6f} | {r['mean_num_split_features']:.6f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAEBench absorption on local checkpoints")
    parser.add_argument("--run_dir", type=str, default=str(ROOT / "runs" / "pilot_balanced_v2_420k"))
    parser.add_argument("--checkpoint_step", type=int, default=420000)
    parser.add_argument(
        "--checkpoint_names",
        nargs="+",
        default=[
            "flat_pilot_topic",
            "flat_pilot_sentiment",
            "kron_pilot_topic",
            "kron_pilot_sentiment",
        ],
    )
    parser.add_argument("--model_name", type=str, default="pythia-410m")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm_batch_size", type=int, default=128)
    parser.add_argument("--llm_dtype", type=str, default="float32")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "runs" / "pilot_balanced_v2_420k" / "saebench_absorption"),
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=str(ROOT / "runs" / "pilot_balanced_v2_420k" / "saebench_absorption_summary.json"),
    )
    args = parser.parse_args()

    set_seed(args.random_seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = str_to_dtype(args.llm_dtype)
    device_t = torch.device(args.device)

    selected_saes: list[tuple[str, base_sae.BaseSAE]] = []
    specs: list[LocalCheckpointSpec] = []

    for checkpoint_name in args.checkpoint_names:
        spec = load_checkpoint_spec(run_dir, checkpoint_name, args.checkpoint_step)
        specs.append(spec)
        adapter = build_adapter(
            spec=spec,
            model_name=args.model_name,
            hook_layer=args.layer,
            device=device_t,
            dtype=dtype,
        )
        selected_saes.append((checkpoint_name, adapter))

    config = AbsorptionEvalConfig(
        model_name=args.model_name,
        random_seed=args.random_seed,
        llm_batch_size=args.llm_batch_size,
        llm_dtype=args.llm_dtype,
    )

    results_dict = absorption_main.run_eval(
        config=config,
        selected_saes=selected_saes,
        device=args.device,
        output_path=str(output_dir),
        force_rerun=args.force_rerun,
    )

    rows = summarize_absorption_results(results_dict)
    if not rows:
        rows = load_rows_from_existing_eval_jsons(output_dir)
    print_table(rows)

    payload = {
        "run_dir": str(run_dir),
        "checkpoint_step": args.checkpoint_step,
        "model_name": args.model_name,
        "layer": args.layer,
        "llm_batch_size": args.llm_batch_size,
        "llm_dtype": args.llm_dtype,
        "eval": "saebench_absorption_first_letter",
        "results": rows,
        "checkpoints": [
            {
                "name": s.name,
                "dict_class": s.dict_class,
                "ckpt_path": str(s.ckpt_path),
            }
            for s in specs
        ],
    }
    summary_path = Path(args.summary_json)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved SAEBench absorption summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
