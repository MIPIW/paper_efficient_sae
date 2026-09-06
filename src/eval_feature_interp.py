"""Section 5.3-style feature interpretability pipeline for pilot SAE checkpoints.

Pipeline stages implemented from Kurochkin et al. (2025), Appendix D:
1) Activation/statistics collection with fixed-size buffers per feature.
2) LLM-based feature interpretation from 16 above-median activation examples.
3) Detection/Fuzzing evaluation with balanced examples and batch-discard on parse failures.

This script is intentionally resumable by persisting intermediate JSON artifacts.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch as t
from nnsight import LanguageModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from data.amazon_reviews import collate_document_batch, load_prepared_datasets  # noqa: E402
from dictionary_learning.dictionary_kron import KronAutoEncoderTopK  # noqa: E402
from dictionary_learning.labeled_buffer import LabeledActivationBuffer  # noqa: E402
from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402


INTERP_SYSTEM_PROMPT = (
    "You are an expert in mechanistic interpretability. You will receive activating text snippets for one SAE "
    "feature. Infer the shared trigger pattern and return ONE concise sentence."
)

DETECTION_SYSTEM_PROMPT = (
    "You are evaluating whether text examples match a feature description. "
    "Return only a Python list of 0/1 integers with EXACTLY the same length as the input examples."
)

FUZZING_SYSTEM_PROMPT = (
    "You are verifying whether highlighted tokens are the true feature-triggering tokens. "
    "Return only a Python list of 0/1 integers with EXACTLY the same length as the input examples, "
    "where 1 means the highlighting is correct and 0 means incorrect."
)


@dataclass
class FeatureKey:
    run_name: str
    branch: str
    feature_id: int

    @property
    def key(self) -> str:
        return f"{self.run_name}|{self.branch}|{self.feature_id}"


class CachingIterator:
    """Iterator wrapper that keeps the most recently yielded batch."""

    def __init__(self, iterable: Iterable[dict[str, Any]]):
        self._it = iter(iterable)
        self.last_batch: dict[str, Any] | None = None

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        self.last_batch = next(self._it)
        return self.last_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest_checkpoint_step(run_dir: Path, checkpoint_name: str) -> int:
    ckpt_dir = run_dir / "checkpoints" / checkpoint_name
    steps: list[int] = []
    for p in ckpt_dir.glob("ae_step_*.pt"):
        m = re.match(r"ae_step_(\d+)\.pt", p.name)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(steps)


def load_checkpoint_config(run_dir: Path, checkpoint_name: str) -> dict[str, Any]:
    cfg_path = run_dir / "checkpoints" / checkpoint_name / "trainer_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing trainer config: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def load_autoencoder(
    run_dir: Path,
    checkpoint_name: str,
    checkpoint_step: int,
    device: str,
):
    cfg = load_checkpoint_config(run_dir, checkpoint_name)
    ckpt = run_dir / "checkpoints" / checkpoint_name / f"ae_step_{checkpoint_step}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    if cfg.get("dict_class") == "KronAutoEncoderTopK":
        ae = KronAutoEncoderTopK(
            activation_dim=int(cfg["activation_dim"]),
            h=int(cfg["h"]),
            m=int(cfg["m"]),
            n=int(cfg["n"]),
            k=int(cfg["k"]),
            combine_rule=str(cfg.get("combine_rule", "mand")),
        )
    else:
        ae = AutoEncoderTopK(
            activation_dim=int(cfg["activation_dim"]),
            dict_size=int(cfg["dict_size"]),
            k=int(cfg["k"]),
        )
    state = t.load(ckpt, map_location="cpu")
    ae.load_state_dict(state)
    ae.to(device)
    ae.eval()
    return ae, cfg


def sample_feature_ids(
    dim: int,
    count: int,
    rng: random.Random,
) -> list[int]:
    if count >= dim:
        return list(range(dim))
    return sorted(rng.sample(list(range(dim)), count))


def init_feature_state(feature: FeatureKey) -> dict[str, Any]:
    return {
        "run_name": feature.run_name,
        "branch": feature.branch,
        "feature_id": feature.feature_id,
        "activation_count": 0,
        "activation_sum": 0.0,
        "activation_min": None,
        "activation_max": None,
        "token_counts": {},
        "multitoken_sum": 0.0,
        "multitoken_count": 0,
        "pos_seen": 0,
        "neg_seen": 0,
        "pos_examples": [],
        "neg_examples": [],
    }


def reservoir_add(
    example_list: list[dict[str, Any]],
    seen_count: int,
    item: dict[str, Any],
    cap: int,
    rng: random.Random,
) -> int:
    seen_count += 1
    if len(example_list) < cap:
        example_list.append(item)
        return seen_count
    j = rng.randint(0, seen_count - 1)
    if j < cap:
        example_list[j] = item
    return seen_count


def flatten_masked_token_ids(
    tokenizer,
    texts: list[str],
    ctx_len: int,
    add_special_tokens: bool,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        max_length=ctx_len,
        padding=True,
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    input_ids = tokenized["input_ids"]  # [B, L]
    attn = tokenized["attention_mask"] != 0

    doc_token_ids: list[list[int]] = []
    for i in range(input_ids.shape[0]):
        ids = input_ids[i][attn[i]].tolist()
        doc_token_ids.append(ids)

    flat_token_ids = input_ids[attn].cpu().numpy().astype(np.int64)
    flat_doc_ids = (
        t.arange(attn.shape[0], dtype=t.long).unsqueeze(1).expand(-1, attn.shape[1])[attn].cpu().numpy().astype(np.int64)
    )
    return doc_token_ids, flat_token_ids, flat_doc_ids


def decode_window_plain(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def decode_window_highlight(tokenizer, token_ids: list[int], highlight_positions: list[int]) -> str:
    hi = set(highlight_positions)
    parts: list[str] = []
    for i, tid in enumerate(token_ids):
        piece = tokenizer.decode(
            [tid],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if i in hi:
            parts.append(f"<<{piece}>>")
        else:
            parts.append(piece)
    return "".join(parts)


def collect_feature_data(args: argparse.Namespace) -> Path:
    start_time = time.time()
    run_dir = Path(args.run_dir)
    run_meta = load_run_metadata(run_dir)
    train_args = run_meta["args"]

    dataset_cache_dir = Path(args.dataset_cache_dir or train_args["dataset_cache_dir"])
    _, eval_ds, _, _ = load_prepared_datasets(dataset_cache_dir)

    checkpoint_names = list(args.checkpoint_names)
    if args.include_flat:
        for name in ("flat_pilot_topic", "flat_pilot_sentiment"):
            if name not in checkpoint_names:
                checkpoint_names.append(name)

    device = args.device or ("cuda:0" if t.cuda.is_available() else "cpu")
    checkpoint_step = args.checkpoint_step
    if checkpoint_step is None:
        # all pilot checkpoints use the same steps; derive from first checkpoint by default.
        checkpoint_step = find_latest_checkpoint_step(run_dir, checkpoint_names[0])

    # Build model used for activation extraction (same as training/eval scripts).
    model = LanguageModel(
        train_args["model_name"],
        dispatch=True,
        device_map=device,
    )
    submodule = model.gpt_neox.layers[int(train_args["layer"])]
    activation_dim = int(train_args["activation_dim"])

    # Load all requested autoencoders.
    aes: dict[str, Any] = {}
    cfgs: dict[str, dict[str, Any]] = {}
    for ckpt_name in checkpoint_names:
        ae, cfg = load_autoencoder(
            run_dir=run_dir,
            checkpoint_name=ckpt_name,
            checkpoint_step=checkpoint_step,
            device=device,
        )
        aes[ckpt_name] = ae
        cfgs[ckpt_name] = cfg

    # Feature sampling plan.
    rng = random.Random(args.seed)
    run_branch_features: dict[tuple[str, str], list[int]] = {}
    for ckpt_name in checkpoint_names:
        cfg = cfgs[ckpt_name]
        if cfg.get("dict_class") == "KronAutoEncoderTopK":
            p_dim = int(cfg["h"]) * int(cfg["m"])
            q_dim = int(cfg["h"]) * int(cfg["n"])
            post_dim = int(cfg["dict_size"])
            run_branch_features[(ckpt_name, "p")] = sample_feature_ids(p_dim, args.kron_p_features, rng)
            run_branch_features[(ckpt_name, "q")] = sample_feature_ids(q_dim, args.kron_q_features, rng)
            run_branch_features[(ckpt_name, "post")] = sample_feature_ids(post_dim, args.kron_post_features, rng)
        else:
            f_dim = int(cfg["dict_size"])
            run_branch_features[(ckpt_name, "full")] = sample_feature_ids(f_dim, args.flat_features, rng)

    feature_states: dict[str, dict[str, Any]] = {}
    for (run_name, branch), feat_ids in run_branch_features.items():
        for fid in feat_ids:
            fk = FeatureKey(run_name=run_name, branch=branch, feature_id=int(fid))
            feature_states[fk.key] = init_feature_state(fk)

    branch_total_tokens: dict[str, int] = defaultdict(int)

    # Build data loader and labeled activation buffer.
    loader = DataLoader(
        eval_ds,
        batch_size=args.doc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=t.cuda.is_available(),
        collate_fn=collate_document_batch,
        drop_last=False,
    )
    caching_it = CachingIterator(loader)
    buffer = LabeledActivationBuffer(
        data=caching_it,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        io="out",
        ctx_len=args.ctx_len,
        device=device,
        remove_bos=False,
        add_special_tokens=True,
        max_activation_norm_multiple=None,
    )

    docs_collected = 0
    tokens_collected = 0
    per_feature_rng = random.Random(args.seed + 17)

    # Prebuild index tensors for selected features per run/branch.
    branch_idx_tensors: dict[tuple[str, str], t.Tensor] = {
        (run_name, branch): t.tensor(feat_ids, device=device, dtype=t.long)
        for (run_name, branch), feat_ids in run_branch_features.items()
    }

    with t.no_grad():
        while docs_collected < args.max_docs:
            try:
                batch = next(buffer)
            except StopIteration:
                break

            raw_batch = caching_it.last_batch
            if raw_batch is None:
                raise RuntimeError("Caching iterator lost last batch unexpectedly")
            texts: list[str] = list(raw_batch["text"])

            # Reconstruct token ids from the same tokenizer settings to build windows.
            doc_token_ids_all, flat_token_ids, flat_doc_ids = flatten_masked_token_ids(
                tokenizer=model.tokenizer,
                texts=texts,
                ctx_len=args.ctx_len,
                add_special_tokens=True,
            )

            x = batch.activations.to(device=device, dtype=t.float32)
            token_doc_ids = batch.token_doc_ids.detach().cpu().numpy().astype(np.int64)
            doc_token_counts = batch.doc_token_counts.detach().cpu().numpy().astype(np.int64)
            n_docs = int(batch.topic_labels.shape[0])

            # Handle potential doc filtering (rare with current settings).
            if n_docs != len(doc_token_ids_all):
                # Keep only docs with at least one token.
                filtered = [ids for ids in doc_token_ids_all if len(ids) > 0]
                if len(filtered) != n_docs:
                    raise RuntimeError(
                        f"Doc alignment mismatch: buffer docs={n_docs}, reconstructed docs={len(doc_token_ids_all)}, "
                        f"non-empty docs={len(filtered)}"
                    )
                doc_token_ids = filtered
            else:
                doc_token_ids = doc_token_ids_all

            if x.shape[0] != flat_token_ids.shape[0]:
                raise RuntimeError(
                    f"Token count mismatch between activations and reconstructed tokenization: "
                    f"{x.shape[0]} vs {flat_token_ids.shape[0]}"
                )
            if x.shape[0] != token_doc_ids.shape[0]:
                raise RuntimeError(
                    f"Token doc-id mismatch: activations={x.shape[0]} token_doc_ids={token_doc_ids.shape[0]}"
                )
            if flat_doc_ids.shape[0] != token_doc_ids.shape[0]:
                raise RuntimeError(
                    f"Flattened token doc-ids mismatch: reconstructed={flat_doc_ids.shape[0]} buffer={token_doc_ids.shape[0]}"
                )

            # Build per-document row slices for token-level tensors.
            doc_starts = np.zeros(n_docs, dtype=np.int64)
            if n_docs > 1:
                doc_starts[1:] = np.cumsum(doc_token_counts[:-1])
            doc_ends = doc_starts + doc_token_counts

            # Compute selected branch activations for each requested checkpoint.
            selected_branch_acts_cpu: dict[tuple[str, str], np.ndarray] = {}
            for run_name, ae in aes.items():
                cfg = cfgs[run_name]
                if cfg.get("dict_class") == "KronAutoEncoderTopK":
                    _, _, _, dense_post, p_pos, q_pos = ae.encode(
                        x,
                        return_topk=True,
                        return_branches=True,
                    )
                    p_flat = p_pos.reshape(p_pos.shape[0], -1)
                    q_flat = q_pos.reshape(q_pos.shape[0], -1)
                    for branch, tensor in (("p", p_flat), ("q", q_flat), ("post", dense_post)):
                        key = (run_name, branch)
                        if key not in branch_idx_tensors:
                            continue
                        idx = branch_idx_tensors[key]
                        selected = t.index_select(tensor, dim=1, index=idx)
                        selected_branch_acts_cpu[key] = selected.detach().cpu().numpy()
                        branch_total_tokens[f"{run_name}|{branch}"] += int(selected.shape[0])
                else:
                    encoded = ae.encode(x)  # sparse post-TopK latents
                    key = (run_name, "full")
                    if key in branch_idx_tensors:
                        idx = branch_idx_tensors[key]
                        selected = t.index_select(encoded, dim=1, index=idx)
                        selected_branch_acts_cpu[key] = selected.detach().cpu().numpy()
                        branch_total_tokens[f"{run_name}|full"] += int(selected.shape[0])

            # Update feature states per selected branch.
            for (run_name, branch), acts_sel in selected_branch_acts_cpu.items():
                feat_ids = run_branch_features[(run_name, branch)]
                if acts_sel.shape[1] != len(feat_ids):
                    raise RuntimeError(
                        f"Selected activation shape mismatch for {run_name}/{branch}: "
                        f"{acts_sel.shape[1]} vs {len(feat_ids)}"
                    )

                for doc_i in range(n_docs):
                    rs = int(doc_starts[doc_i])
                    re = int(doc_ends[doc_i])
                    if re <= rs:
                        continue
                    doc_ids = doc_token_ids[doc_i]
                    doc_len = len(doc_ids)
                    if doc_len <= 0:
                        continue

                    block = acts_sel[rs:re, :]  # [doc_len, S]
                    if block.shape[0] != doc_len:
                        # Fallback for rare alignment mismatch.
                        doc_len = min(block.shape[0], doc_len)
                        block = block[:doc_len]
                        doc_ids = doc_ids[:doc_len]
                        if doc_len <= 0:
                            continue

                    global_doc_id = docs_collected + doc_i

                    for j, fid in enumerate(feat_ids):
                        vals = block[:, j]
                        fk = FeatureKey(run_name=run_name, branch=branch, feature_id=int(fid)).key
                        state = feature_states[fk]

                        active = np.nonzero(vals > 0.0)[0]
                        if active.size > 0:
                            act_vals = vals[active]
                            state["activation_count"] += int(active.size)
                            state["activation_sum"] += float(act_vals.sum())

                            a_min = float(act_vals.min())
                            a_max = float(act_vals.max())
                            if state["activation_min"] is None or a_min < state["activation_min"]:
                                state["activation_min"] = a_min
                            if state["activation_max"] is None or a_max > state["activation_max"]:
                                state["activation_max"] = a_max

                            token_counts: dict[str, int] = state["token_counts"]
                            grouped_windows: dict[tuple[int, int], dict[str, Any]] = {}
                            for pos, aval in zip(active.tolist(), act_vals.tolist()):
                                tok_id = int(doc_ids[pos])
                                tk = str(tok_id)
                                token_counts[tk] = token_counts.get(tk, 0) + 1

                                w_start = max(0, pos - args.window_radius)
                                w_end = min(doc_len, pos + args.window_radius + 1)
                                key = (w_start, w_end)
                                if key not in grouped_windows:
                                    grouped_windows[key] = {
                                        "start": w_start,
                                        "end": w_end,
                                        "active_positions": [],
                                        "activation": float(aval),
                                    }
                                grouped_windows[key]["active_positions"].append(int(pos - w_start))
                                if aval > grouped_windows[key]["activation"]:
                                    grouped_windows[key]["activation"] = float(aval)

                            state["multitoken_sum"] += float(active.size / max(1, doc_len))
                            state["multitoken_count"] += 1

                            for w in grouped_windows.values():
                                item = {
                                    "doc_id": int(global_doc_id),
                                    "window_token_ids": [int(x) for x in doc_ids[w["start"] : w["end"]]],
                                    "active_positions": sorted(set(int(p) for p in w["active_positions"])),
                                    "activation": float(w["activation"]),
                                }
                                state["pos_seen"] = reservoir_add(
                                    example_list=state["pos_examples"],
                                    seen_count=int(state["pos_seen"]),
                                    item=item,
                                    cap=args.buffer_size,
                                    rng=per_feature_rng,
                                )

                        inactive = np.nonzero(vals <= 0.0)[0]
                        if inactive.size > 0:
                            neg_pos = int(inactive[per_feature_rng.randrange(inactive.size)])
                            w_start = max(0, neg_pos - args.window_radius)
                            w_end = min(doc_len, neg_pos + args.window_radius + 1)
                            neg_item = {
                                "doc_id": int(global_doc_id),
                                "window_token_ids": [int(x) for x in doc_ids[w_start:w_end]],
                                "active_positions": [],
                                "activation": 0.0,
                            }
                            state["neg_seen"] = reservoir_add(
                                example_list=state["neg_examples"],
                                seen_count=int(state["neg_seen"]),
                                item=neg_item,
                                cap=args.buffer_size,
                                rng=per_feature_rng,
                            )

            docs_collected += n_docs
            tokens_collected += int(x.shape[0])
            if args.log_every_docs > 0 and docs_collected % args.log_every_docs < n_docs:
                elapsed = time.time() - start_time
                print(
                    f"[collect] docs={docs_collected}/{args.max_docs} tokens={tokens_collected:,} "
                    f"elapsed={elapsed/60:.1f}m",
                    flush=True,
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "feature_interp_collection.json"

    payload = {
        "metadata": {
            "run_dir": str(run_dir),
            "dataset_cache_dir": str(dataset_cache_dir),
            "checkpoint_names": checkpoint_names,
            "checkpoint_step": int(checkpoint_step),
            "model_name": str(train_args["model_name"]),
            "layer": int(train_args["layer"]),
            "ctx_len": int(args.ctx_len),
            "window_size": int(2 * args.window_radius + 1),
            "buffer_size": int(args.buffer_size),
            "max_docs": int(args.max_docs),
            "docs_collected": int(docs_collected),
            "tokens_collected": int(tokens_collected),
            "collection_time_sec": float(time.time() - start_time),
            "kron_p_features": int(args.kron_p_features),
            "kron_q_features": int(args.kron_q_features),
            "kron_post_features": int(args.kron_post_features),
            "flat_features": int(args.flat_features),
            "seed": int(args.seed),
        },
        "checkpoint_configs": cfgs,
        "feature_plan": {
            f"{run}|{branch}": feat_ids for (run, branch), feat_ids in run_branch_features.items()
        },
        "branch_total_tokens": dict(branch_total_tokens),
        "feature_states": feature_states,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Saved collection artifact: {out_path}", flush=True)
    return out_path


def _sanitize_eval_output(raw_text: str) -> str:
    text = raw_text.strip()
    # Normalize common Qwen chat scaffolding to make parsing robust.
    text = text.translate(str.maketrans("０１", "01"))
    text = re.sub(r"<\|im_start\|>assistant", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<\|im_end\|>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return text


def extract_list_labels(raw_text: str, expected_len: int) -> list[int] | None:
    text = _sanitize_eval_output(raw_text)

    # Try direct parse first.
    candidates = [text]
    # Try fenced code block content.
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    # Try first bracketed list.
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            obj = ast.literal_eval(cand)
        except Exception:
            continue
        if not isinstance(obj, (list, tuple)):
            continue
        out: list[int] = []
        valid = True
        for x in obj:
            if isinstance(x, bool):
                out.append(int(x))
            elif isinstance(x, (int, np.integer)):
                if int(x) not in (0, 1):
                    valid = False
                    break
                out.append(int(x))
            elif isinstance(x, str) and x.strip() in {"0", "1"}:
                out.append(int(x.strip()))
            else:
                valid = False
                break
        if not valid:
            continue
        if len(out) < expected_len:
            return None
        return out[:expected_len]

    # Fallback: pull standalone binary tokens from free-form text.
    low = text.lower()
    token_matches = re.findall(r"\b(?:0|1|true|false|yes|no)\b", low)
    if len(token_matches) >= expected_len:
        out: list[int] = []
        for tok in token_matches[:expected_len]:
            if tok in {"1", "true", "yes"}:
                out.append(1)
            else:
                out.append(0)
        return out
    return None


def infer_qwen_snapshot_path(model_hint: str) -> Path:
    p = Path(model_hint)
    if p.exists():
        # If users pass the model directory root, resolve first snapshot.
        snapshots = p / "snapshots"
        if snapshots.exists():
            cands = sorted([x for x in snapshots.iterdir() if x.is_dir()])
            if not cands:
                raise FileNotFoundError(f"No snapshots under {snapshots}")
            return cands[0]
        return p
    raise FileNotFoundError(f"Qwen model path does not exist: {model_hint}")


def format_interp_prompt(
    source_tokenizer,
    feature_key: str,
    examples: list[dict[str, Any]],
) -> str:
    lines = [
        f"Feature: {feature_key}",
        "Examples (activated tokens are wrapped with << >>):",
    ]
    for i, ex in enumerate(examples, start=1):
        text = decode_window_highlight(
            source_tokenizer,
            token_ids=[int(x) for x in ex["window_token_ids"]],
            highlight_positions=[int(x) for x in ex["active_positions"]],
        )
        lines.append(f"{i}. {text}")
    lines.append("Return exactly one concise sentence describing what this feature detects.")
    return "\n".join(lines)


def format_detection_prompt(
    feature_desc: str,
    example_texts: list[str],
) -> str:
    lines = [
        f"Feature description: {feature_desc}",
        "Predict activation for each example (1=activates, 0=does not activate).",
        f"Return ONLY a Python list with exactly {len(example_texts)} integers in order, e.g. "
        f"[1,0,1,...] with length {len(example_texts)}.",
        "Examples:",
    ]
    for i, text in enumerate(example_texts, start=1):
        lines.append(f"{i}. {text}")
    return "\n".join(lines)


def format_fuzzing_prompt(
    feature_desc: str,
    highlighted_texts: list[str],
) -> str:
    lines = [
        f"Feature description: {feature_desc}",
        "For each example, decide whether the <<highlighted>> tokens are correctly labeled as triggering this feature.",
        f"Return ONLY a Python list with exactly {len(highlighted_texts)} integers in order "
        "(1=correct highlight, 0=incorrect).",
        "Examples:",
    ]
    for i, text in enumerate(highlighted_texts, start=1):
        lines.append(f"{i}. {text}")
    return "\n".join(lines)


def apply_chat_template(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    *,
    thinking_enabled: bool | None,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: dict[str, Any] = {}
    if thinking_enabled is not None:
        # Qwen chat template supports toggling reasoning mode via this field.
        # In practice we set False for machine-parseable deterministic label lists.
        kwargs["enable_thinking"] = thinking_enabled
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def batched_generate(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    chunk_size: int,
) -> list[str]:
    outputs: list[str] = []
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        gen = llm.generate(chunk, sampling_params=sampling_params)
        for out in gen:
            if out.outputs:
                outputs.append(out.outputs[0].text)
            else:
                outputs.append("")
    return outputs


def make_eval_sets(
    state: dict[str, Any],
    max_pos_neg: int,
    interp_examples: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    pos_examples = state["pos_examples"]
    neg_examples = state["neg_examples"]
    if not pos_examples or not neg_examples:
        return None

    acts = np.array([float(ex["activation"]) for ex in pos_examples], dtype=np.float64)
    if acts.size == 0:
        return None
    med = float(np.median(acts))
    high = [ex for ex in pos_examples if float(ex["activation"]) > med]
    if len(high) < interp_examples:
        # Fallback only when strict > median leaves too few examples.
        high = [ex for ex in pos_examples if float(ex["activation"]) >= med]
    if len(high) < interp_examples:
        return None

    interp_pool = rng.sample(high, interp_examples)
    n = min(max_pos_neg, len(high), len(neg_examples))
    if n <= 0:
        return None
    pos_eval = rng.sample(high, n)
    neg_eval = rng.sample(neg_examples, n)

    return {
        "median_activation": med,
        "high_pos_pool_size": len(high),
        "interp_examples": interp_pool,
        "pos_eval": pos_eval,
        "neg_eval": neg_eval,
    }


def build_fuzzing_examples(
    source_tokenizer,
    feature_desc: str,
    pos_pool: list[dict[str, Any]],
    neg_pool: list[dict[str, Any]],
    max_total: int,
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    # 50% correct positives, 25% mislabeled positives, 25% random negatives.
    cap_from_pos = (4 * len(pos_pool)) // 3
    cap_from_neg = 4 * len(neg_pool)
    n_total = min(max_total, cap_from_pos, cap_from_neg)
    n_total = n_total - (n_total % 4)
    if n_total < 4:
        return [], []

    n_correct = n_total // 2
    n_mislabeled = n_total // 4
    n_random_neg = n_total - n_correct - n_mislabeled

    correct_pos = rng.sample(pos_pool, n_correct)
    mislabeled_pos = rng.sample(pos_pool, n_mislabeled)
    random_neg = rng.sample(neg_pool, n_random_neg)

    examples: list[str] = []
    labels: list[int] = []

    for ex in correct_pos:
        txt = decode_window_highlight(
            source_tokenizer,
            token_ids=[int(x) for x in ex["window_token_ids"]],
            highlight_positions=[int(x) for x in ex["active_positions"]],
        )
        examples.append(txt)
        labels.append(1)

    for ex in mislabeled_pos:
        ids = [int(x) for x in ex["window_token_ids"]]
        active = set(int(x) for x in ex["active_positions"])
        candidates = [i for i in range(len(ids)) if i not in active]
        if not candidates:
            candidates = list(range(len(ids)))
        wrong_pos = [rng.choice(candidates)]
        txt = decode_window_highlight(
            source_tokenizer,
            token_ids=ids,
            highlight_positions=wrong_pos,
        )
        examples.append(txt)
        labels.append(0)

    for ex in random_neg:
        ids = [int(x) for x in ex["window_token_ids"]]
        if not ids:
            continue
        rand_pos = [rng.randrange(len(ids))]
        txt = decode_window_highlight(
            source_tokenizer,
            token_ids=ids,
            highlight_positions=rand_pos,
        )
        examples.append(txt)
        labels.append(0)

    if not examples:
        return [], []
    order = list(range(len(examples)))
    rng.shuffle(order)
    examples = [examples[i] for i in order]
    labels = [labels[i] for i in order]
    return examples, labels


def balanced_accuracy_from_pairs(true_labels: list[int], pred_labels: list[int]) -> float | None:
    if not true_labels or not pred_labels or len(true_labels) != len(pred_labels):
        return None
    tp = tn = fp = fn = 0
    for y, p in zip(true_labels, pred_labels):
        if y == 1 and p == 1:
            tp += 1
        elif y == 1 and p == 0:
            fn += 1
        elif y == 0 and p == 0:
            tn += 1
        elif y == 0 and p == 1:
            fp += 1
    pos_total = tp + fn
    neg_total = tn + fp
    if pos_total == 0 or neg_total == 0:
        return None
    tpr = tp / pos_total
    tnr = tn / neg_total
    return 0.5 * (tpr + tnr)


def run_llm_eval(args: argparse.Namespace) -> Path:
    collection_path = Path(args.collection_path)
    if not collection_path.exists():
        raise FileNotFoundError(f"Missing collection artifact: {collection_path}")
    payload = json.loads(collection_path.read_text(encoding="utf-8"))

    feature_states: dict[str, dict[str, Any]] = payload["feature_states"]
    branch_total_tokens: dict[str, int] = payload.get("branch_total_tokens", {})
    source_model_name = str(payload["metadata"]["model_name"])

    # Load source tokenizer for reconstructing windows exactly from source-model tokens.
    source_tokenizer = AutoTokenizer.from_pretrained(
        source_model_name,
        local_files_only=True,
        trust_remote_code=True,
    )

    qwen_path = infer_qwen_snapshot_path(args.qwen_model_path)
    llm_tokenizer = AutoTokenizer.from_pretrained(
        qwen_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    # Prepare evaluation records first.
    rng = random.Random(args.seed + 101)
    eval_records: dict[str, dict[str, Any]] = {}
    for fk, state in feature_states.items():
        eval_set = make_eval_sets(
            state=state,
            max_pos_neg=args.max_pos_neg_examples,
            interp_examples=args.interp_examples,
            rng=rng,
        )
        if eval_set is None:
            continue

        token_total_key = f"{state['run_name']}|{state['branch']}"
        denom = max(1, int(branch_total_tokens.get(token_total_key, 1)))
        act_count = int(state["activation_count"])
        mean_act = float(state["activation_sum"] / act_count) if act_count > 0 else float("nan")
        token_counts = {int(k): int(v) for k, v in state["token_counts"].items()}
        total_token_hits = sum(token_counts.values())
        entropy = float("nan")
        if total_token_hits > 0:
            probs = np.array([v / total_token_hits for v in token_counts.values()], dtype=np.float64)
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        multitoken_ratio = (
            float(state["multitoken_sum"] / state["multitoken_count"])
            if int(state["multitoken_count"]) > 0
            else float("nan")
        )

        eval_records[fk] = {
            "feature_key": fk,
            "run_name": state["run_name"],
            "branch": state["branch"],
            "feature_id": int(state["feature_id"]),
            "activation_count": act_count,
            "activation_min": state["activation_min"],
            "activation_max": state["activation_max"],
            "activation_mean": mean_act,
            "activation_frequency": float(act_count / denom),
            "token_entropy": entropy,
            "multitoken_ratio": multitoken_ratio,
            "median_activation": float(eval_set["median_activation"]),
            "high_pos_pool_size": int(eval_set["high_pos_pool_size"]),
            "interp_examples": eval_set["interp_examples"],
            "pos_eval": eval_set["pos_eval"],
            "neg_eval": eval_set["neg_eval"],
        }

    if not eval_records:
        raise RuntimeError("No valid features available after collection filtering.")

    # Load LLM once for interpretation + scoring.
    # Avoid CUDA re-init failures when parent process has touched CUDA.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    llm = LLM(
        model=str(qwen_path),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype=args.llm_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.llm_enforce_eager,
    )

    interp_sampling = SamplingParams(
        temperature=args.interp_temperature,
        top_p=0.95,
        max_tokens=args.interp_max_tokens,
    )
    eval_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.eval_max_tokens,
    )

    # Stage 1: feature interpretation.
    interp_feature_keys = sorted(eval_records.keys())
    interp_prompts: list[str] = []
    for fk in interp_feature_keys:
        rec = eval_records[fk]
        user_prompt = format_interp_prompt(
            source_tokenizer=source_tokenizer,
            feature_key=fk,
            examples=rec["interp_examples"],
        )
        prompt = apply_chat_template(
            tokenizer=llm_tokenizer,
            system_prompt=INTERP_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            thinking_enabled=False,  # Qwen template: False inserts <think> section.
        )
        interp_prompts.append(prompt)

    interp_outputs = batched_generate(
        llm=llm,
        prompts=interp_prompts,
        sampling_params=interp_sampling,
        chunk_size=args.llm_prompt_chunk_size,
    )

    for fk, out in zip(interp_feature_keys, interp_outputs):
        cleaned = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL | re.IGNORECASE).strip()
        eval_records[fk]["description"] = cleaned
        eval_records[fk]["raw_description_response"] = out

    # Stage 2: detection scoring batches.
    det_tasks: list[dict[str, Any]] = []
    for fk in interp_feature_keys:
        rec = eval_records[fk]
        desc = rec["description"].strip()
        if not desc:
            continue
        pos_eval = rec["pos_eval"]
        neg_eval = rec["neg_eval"]
        det_examples: list[tuple[str, int]] = []
        for ex in pos_eval:
            det_examples.append(
                (
                    decode_window_plain(source_tokenizer, [int(x) for x in ex["window_token_ids"]]),
                    1,
                )
            )
        for ex in neg_eval:
            det_examples.append(
                (
                    decode_window_plain(source_tokenizer, [int(x) for x in ex["window_token_ids"]]),
                    0,
                )
            )
        rng.shuffle(det_examples)

        for i in range(0, len(det_examples), args.eval_batch_examples):
            batch = det_examples[i : i + args.eval_batch_examples]
            texts = [x[0] for x in batch]
            labels = [int(x[1]) for x in batch]
            user_prompt = format_detection_prompt(feature_desc=desc, example_texts=texts)
            prompt = apply_chat_template(
                tokenizer=llm_tokenizer,
                system_prompt=DETECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                # Disable reasoning mode here; with Qwen3 this yields compact
                # machine-parseable label lists for deterministic scoring.
                thinking_enabled=False,
            )
            det_tasks.append(
                {
                    "feature_key": fk,
                    "true_labels": labels,
                    "prompt": prompt,
                }
            )

    det_outputs = batched_generate(
        llm=llm,
        prompts=[t["prompt"] for t in det_tasks],
        sampling_params=eval_sampling,
        chunk_size=args.llm_prompt_chunk_size,
    )

    det_true: dict[str, list[int]] = defaultdict(list)
    det_pred: dict[str, list[int]] = defaultdict(list)
    det_discarded_batches: dict[str, int] = defaultdict(int)
    parse_fail_rows: list[dict[str, Any]] = []

    for task, out in zip(det_tasks, det_outputs):
        labels = task["true_labels"]
        parsed = extract_list_labels(out, expected_len=len(labels))
        fk = task["feature_key"]
        if parsed is None:
            det_discarded_batches[fk] += 1
            parse_fail_rows.append(
                {
                    "stage": "detection",
                    "feature_key": fk,
                    "expected_len": int(len(labels)),
                    "raw_output": out,
                    "sanitized_output": _sanitize_eval_output(out),
                }
            )
            continue
        det_true[fk].extend(labels)
        det_pred[fk].extend(parsed)

    # Stage 3: fuzzing scoring batches.
    fuzz_tasks: list[dict[str, Any]] = []
    for fk in interp_feature_keys:
        rec = eval_records[fk]
        desc = rec["description"].strip()
        if not desc:
            continue
        examples, labels = build_fuzzing_examples(
            source_tokenizer=source_tokenizer,
            feature_desc=desc,
            pos_pool=rec["pos_eval"],
            neg_pool=rec["neg_eval"],
            max_total=args.max_fuzz_examples,
            rng=rng,
        )
        rec["fuzz_example_count"] = len(labels)
        if not examples:
            continue
        for i in range(0, len(examples), args.eval_batch_examples):
            b_texts = examples[i : i + args.eval_batch_examples]
            b_labels = labels[i : i + args.eval_batch_examples]
            user_prompt = format_fuzzing_prompt(feature_desc=desc, highlighted_texts=b_texts)
            prompt = apply_chat_template(
                tokenizer=llm_tokenizer,
                system_prompt=FUZZING_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                thinking_enabled=False,
            )
            fuzz_tasks.append(
                {
                    "feature_key": fk,
                    "true_labels": b_labels,
                    "prompt": prompt,
                }
            )

    fuzz_outputs = batched_generate(
        llm=llm,
        prompts=[t["prompt"] for t in fuzz_tasks],
        sampling_params=eval_sampling,
        chunk_size=args.llm_prompt_chunk_size,
    )

    fuzz_true: dict[str, list[int]] = defaultdict(list)
    fuzz_pred: dict[str, list[int]] = defaultdict(list)
    fuzz_discarded_batches: dict[str, int] = defaultdict(int)
    for task, out in zip(fuzz_tasks, fuzz_outputs):
        labels = task["true_labels"]
        parsed = extract_list_labels(out, expected_len=len(labels))
        fk = task["feature_key"]
        if parsed is None:
            fuzz_discarded_batches[fk] += 1
            parse_fail_rows.append(
                {
                    "stage": "fuzzing",
                    "feature_key": fk,
                    "expected_len": int(len(labels)),
                    "raw_output": out,
                    "sanitized_output": _sanitize_eval_output(out),
                }
            )
            continue
        fuzz_true[fk].extend(labels)
        fuzz_pred[fk].extend(parsed)

    # Finalize per-feature records.
    feature_results: list[dict[str, Any]] = []
    for fk in interp_feature_keys:
        rec = eval_records[fk]
        det_score = balanced_accuracy_from_pairs(det_true[fk], det_pred[fk])
        fuzz_score = balanced_accuracy_from_pairs(fuzz_true[fk], fuzz_pred[fk])
        row = {
            "feature_key": fk,
            "run_name": rec["run_name"],
            "branch": rec["branch"],
            "feature_id": rec["feature_id"],
            "description": rec.get("description", ""),
            "detection_score": det_score,
            "fuzzing_score": fuzz_score,
            "detection_pairs_used": len(det_true[fk]),
            "fuzzing_pairs_used": len(fuzz_true[fk]),
            "detection_discarded_batches": int(det_discarded_batches[fk]),
            "fuzzing_discarded_batches": int(fuzz_discarded_batches[fk]),
            "activation_count": rec["activation_count"],
            "activation_frequency": rec["activation_frequency"],
            "activation_mean": rec["activation_mean"],
            "activation_min": rec["activation_min"],
            "activation_max": rec["activation_max"],
            "token_entropy": rec["token_entropy"],
            "multitoken_ratio": rec["multitoken_ratio"],
            "median_activation": rec["median_activation"],
            "high_pos_pool_size": rec["high_pos_pool_size"],
            "num_pos_eval": len(rec["pos_eval"]),
            "num_neg_eval": len(rec["neg_eval"]),
        }
        feature_results.append(row)

    # Aggregate by run/branch.
    agg: dict[str, dict[str, Any]] = {}
    for row in feature_results:
        key = f"{row['run_name']}|{row['branch']}"
        if key not in agg:
            agg[key] = {
                "run_name": row["run_name"],
                "branch": row["branch"],
                "num_features": 0,
                "det_scores": [],
                "fuzz_scores": [],
            }
        agg[key]["num_features"] += 1
        if row["detection_score"] is not None:
            agg[key]["det_scores"].append(float(row["detection_score"]))
        if row["fuzzing_score"] is not None:
            agg[key]["fuzz_scores"].append(float(row["fuzzing_score"]))

    aggregate_rows: list[dict[str, Any]] = []
    for k in sorted(agg.keys()):
        a = agg[k]
        det_scores = a["det_scores"]
        fuzz_scores = a["fuzz_scores"]
        aggregate_rows.append(
            {
                "run_name": a["run_name"],
                "branch": a["branch"],
                "num_features": int(a["num_features"]),
                "num_features_with_detection": int(len(det_scores)),
                "num_features_with_fuzzing": int(len(fuzz_scores)),
                "mean_detection_score": float(np.mean(det_scores)) if det_scores else None,
                "mean_fuzzing_score": float(np.mean(fuzz_scores)) if fuzz_scores else None,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "feature_interp_results.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in feature_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "collection_path": str(collection_path),
        "llm_model_path": str(qwen_path),
        "llm_dtype": args.llm_dtype,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "num_feature_results": len(feature_results),
        "aggregate": aggregate_rows,
    }
    summary_path = out_dir / "feature_interp_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_parse_failures and parse_fail_rows:
        parse_fail_path = out_dir / "feature_interp_parse_failures.jsonl"
        with parse_fail_path.open("w", encoding="utf-8") as f:
            for row in parse_fail_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved parse-failure debug rows: {parse_fail_path}", flush=True)

    print(f"Saved per-feature results: {out_jsonl}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 5.3-style feature interpretability pipeline")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "collect", "llm"])

    parser.add_argument("--run_dir", type=str, default=str(ROOT / "runs" / "pilot_balanced_v2_420k"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "runs" / "pilot_balanced_v2_420k" / "feature_interp"))
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=420000)
    parser.add_argument(
        "--checkpoint_names",
        nargs="+",
        default=["kron_pilot_topic", "kron_pilot_sentiment"],
    )
    parser.add_argument("--include_flat", action="store_true")

    # Collection settings.
    parser.add_argument("--ctx_len", type=int, default=256)
    parser.add_argument("--max_docs", type=int, default=12000)
    parser.add_argument("--doc_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--window_radius", type=int, default=15)
    parser.add_argument("--buffer_size", type=int, default=384)
    parser.add_argument("--kron_p_features", type=int, default=64)
    parser.add_argument("--kron_q_features", type=int, default=64)
    parser.add_argument("--kron_post_features", type=int, default=56)
    parser.add_argument("--flat_features", type=int, default=64)
    parser.add_argument("--log_every_docs", type=int, default=1000)

    # LLM settings.
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default="/huggingface_cache/hub/models--Qwen--Qwen3-14B",
    )
    parser.add_argument("--llm_dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--llm_enforce_eager", action="store_true")
    parser.add_argument("--llm_prompt_chunk_size", type=int, default=96)
    parser.add_argument("--interp_examples", type=int, default=16)
    parser.add_argument("--max_pos_neg_examples", type=int, default=64)
    parser.add_argument("--max_fuzz_examples", type=int, default=128)
    parser.add_argument("--eval_batch_examples", type=int, default=8)
    parser.add_argument("--interp_temperature", type=float, default=0.2)
    parser.add_argument("--interp_max_tokens", type=int, default=96)
    parser.add_argument("--eval_max_tokens", type=int, default=64)
    parser.add_argument("--save_parse_failures", action="store_true")

    parser.add_argument("--collection_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    # Avoid accidental environment leakage from global PYTHONPATH during tool execution.
    if os.environ.get("PYTHONPATH"):
        print(f"Warning: PYTHONPATH is set to {os.environ['PYTHONPATH']!r}; consider unsetting for reproducibility.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collection_path: Path | None = None
    if args.phase in {"all", "collect"}:
        collection_path = collect_feature_data(args)
    else:
        if args.collection_path is None:
            collection_path = output_dir / "feature_interp_collection.json"
        else:
            collection_path = Path(args.collection_path)

    if args.phase in {"all", "llm"}:
        if collection_path is None:
            raise RuntimeError("Collection artifact path is not available for LLM phase")
        args.collection_path = str(collection_path)
        run_llm_eval(args)


if __name__ == "__main__":
    main()
