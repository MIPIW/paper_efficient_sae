"""Absorption + hedging benchmark harness, stratified by interaction-structure tercile.

Part B of the conjunctive-structure investigation. The synthetic test
(`src/train.py --synthetic_conjunctive`) establishes *causally* that a bilinear
encoder should pull ahead of a flat one exactly when the ground-truth target is
conjunctive in two primitives. This harness asks whether that mechanism explains
a real, independently-documented SAE pathology.

Feature absorption (Chanin et al., 2024) is the canonical instance: a child
concept is `parent AND differentiator`, i.e. a conjunction. When an SAE cannot
represent that conjunction as one atom, a specific-token latent swallows the
parent direction and the parent's own latent stops firing. Feature hedging
(Chanin et al., 2025) is the mirror failure: the parent latent's decoder mixes
in a fraction of its children's directions instead of staying pure.

For one SAE -- either a flat `AutoEncoderTopK` or a `KronAutoEncoderTopK` with
`combine_rule in {mand, concat, mor, mnand, mnor}` -- this harness computes, on
the SAEBench first-letter-classification task:

  * absorption score  -- SAEBench's `absorption_fraction` and full-absorption
                         rate, per letter and per identified absorbing latent
  * hedging score     -- see `hedging_score_from_decoders`
  * FVU               -- fraction of variance unexplained, on the same
                         activations, so a "good" absorption number produced by
                         a model that barely reconstructs is visible as such
                         (REPORT1.md §8 shows exactly that confound biting)
  * interaction score -- `src/eval_interaction_structure.py`'s control-corrected
                         Delta-AUC selectivity, per identified case

and then stratifies every score by interaction-score tercile via
`stratify_by_interaction_tercile`. That stratification is the actual test: if
the bilinear encoder helps *because* the structure is conjunctive, its advantage
must be concentrated in the top tercile and absent in the bottom one. A uniform
gain across terciles falsifies the conjunctive explanation and points at a
generic capacity difference instead.

Nothing here is model-family-specific: `--model_name` defaults to this project's
pythia-410m pipeline but `gemma-2-2b` (or any transformer_lens model) is a
straight substitution, since d_model, layer and hook name are all derived from
the loaded model and the CLI rather than hardcoded.

Self-test:  python src/eval_absorption_hedging.py --self_test
Single cell: python src/eval_absorption_hedging.py --run_dir runs/... --checkpoint_names kron_joint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DL_ROOT = ROOT / "dictionary_learning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from eval_interaction_structure import (  # noqa: E402
    interaction_structure_score,
    stratify_by_interaction_tercile,
)

# Cosine-similarity floor for calling a latent a plausible absorber of a letter.
# Matches SAEBench's own `absorption_fraction_probe_cos_sim_threshold` so the two
# metrics identify the same population of latents.
DEFAULT_ABSORBER_COS_THRESHOLD = 0.1

TERCILE_METRIC_KEYS = (
    "absorption_fraction",
    "full_absorption_rate",
    "hedging_score",
    "fvu",
)


# ---------------------------------------------------------------------------
# pure metric functions (no model, no SAEBench -- these are what the self-test covers)
# ---------------------------------------------------------------------------
def compute_fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained: ||x - x_hat||^2 / ||x - mean(x)||^2.

    Identical definition to `train.py:measure_synthetic_reconstruction`, so FVU
    numbers are comparable across this project's synthetic and real arms. FVU is
    reported alongside every absorption/hedging number because REPORT1.md §8
    found rules (mnand/mnor) that posted attractive leakage numbers while
    reconstructing no better than an untrained network -- a result that must be
    read as confounded, not as a success.
    """
    x = x.detach().to(dtype=torch.float64)
    x_hat = x_hat.detach().to(dtype=torch.float64)
    sq_err = float((x - x_hat).pow(2).sum().item())
    total_var = float((x - x.mean(dim=0, keepdim=True)).pow(2).sum().item())
    return sq_err / total_var if total_var > 0 else float("nan")


def child_specific_direction(decoder_row: torch.Tensor, probe_direction: torch.Tensor) -> torch.Tensor:
    """The part of a latent's decoder direction that is NOT the parent concept.

    `d_child = normalize(W_dec[c] - (W_dec[c] . u) u)` for unit parent direction
    `u`. Removing the parent component is what makes the resulting direction a
    *differentiator*: it is the "AND what else" half of `child = parent AND
    differentiator`, and it is the natural choice for the second primitive `b`
    fed to the interaction-structure probe.

    Returns a zero vector if the decoder row lies entirely along the parent
    direction (no differentiator exists), which callers must treat as a
    degenerate case rather than as a direction.
    """
    u = probe_direction.detach().to(dtype=torch.float32).ravel()
    u = u / u.norm().clamp_min(1e-12)
    d = decoder_row.detach().to(dtype=torch.float32).ravel()
    residual = d - (d @ u) * u
    norm = residual.norm()
    if float(norm) < 1e-8:
        return torch.zeros_like(residual)
    return residual / norm


def hedging_score_from_decoders(
    w_dec: torch.Tensor,
    main_feature_ids: Sequence[int],
    child_feature_ids: Sequence[int],
    probe_direction: torch.Tensor,
) -> Dict[str, Any]:
    """Hedging score: how much of the CHILDREN's specific directions sit inside the PARENT latent.

    Operationalizes Chanin et al. (2025)'s feature hedging on the artifacts the
    first-letter absorption pipeline already produces. For a letter with unit
    probe (parent) direction `u`, parent latents `M` and child/absorbing latents
    `C`, we project `u` out of every decoder row -- leaving only the
    child-specific component of each -- and measure how much the parent latents'
    residual directions still point along the children's::

        d_i        = normalize(W_dec[i] - (W_dec[i] . u) u)
        hedging(L) = mean_{m in M} mean_{c in C} max(0, cos(d_m, d_c))

    A parent latent that cleanly encodes only the parent concept has a residual
    direction unrelated to any child's, giving ~0. A parent latent that has
    hedged -- absorbed a slice of each child to shave reconstruction error --
    points partway at every child, giving a positive score. Projecting `u` out
    first is essential: without it parent and child align trivially because both
    contain the parent direction, and the metric would measure absorption's
    precondition rather than hedging.

    Directionality note: this is a decoder-geometry operationalization built from
    the quantities SAEBench exposes, not a line-for-line port of the reference
    implementation from the paper. It is reported as such, and should be
    cross-checked against the authors' implementation before publication.

    Returns the aggregate score plus the full parent x child cosine matrix, so
    the aggregate can be audited rather than taken on faith.
    """
    main_ids = [int(i) for i in main_feature_ids]
    child_ids = [int(i) for i in child_feature_ids if int(i) not in set(int(j) for j in main_feature_ids)]
    out: Dict[str, Any] = {
        "main_feature_ids": main_ids,
        "child_feature_ids": child_ids,
        "n_main": len(main_ids),
        "n_child": len(child_ids),
    }
    if not main_ids or not child_ids:
        out["hedging_score"] = float("nan")
        out["note"] = "no main and/or child latents identified for this letter"
        return out

    main_dirs = torch.stack([child_specific_direction(w_dec[i], probe_direction) for i in main_ids])
    child_dirs = torch.stack([child_specific_direction(w_dec[i], probe_direction) for i in child_ids])
    cos = main_dirs @ child_dirs.T  # both sides are unit-norm (or exactly zero)
    cos_pos = cos.clamp_min(0.0)
    out["hedging_score"] = float(cos_pos.mean().item())
    out["hedging_score_max"] = float(cos_pos.max().item())
    out["parent_child_cos_matrix"] = cos.tolist()
    return out


def build_case_records(
    per_case: Sequence[Dict[str, Any]],
    hedging_by_letter: Dict[str, Dict[str, Any]],
    fvu: float,
    sae_name: str,
    combine_rule: Optional[str],
) -> List[Dict[str, Any]]:
    """Flatten per-(letter, absorbing-latent) results into the record schema the tercile analysis consumes.

    One record = one identified absorption case. Every record carries all four
    metrics plus the interaction score it will be stratified on, so the
    stratification function stays generic and this is the only place the schema
    is defined.
    """
    records: List[Dict[str, Any]] = []
    for case in per_case:
        letter = case["letter"]
        hedge = hedging_by_letter.get(letter, {})
        records.append(
            {
                "sae_name": sae_name,
                "combine_rule": combine_rule,
                "letter": letter,
                "absorbing_latent": case.get("absorbing_latent"),
                "main_feature_ids": case.get("main_feature_ids", []),
                "n_words": case.get("n_words"),
                "absorption_fraction": case.get("absorption_fraction", float("nan")),
                "full_absorption_rate": case.get("full_absorption_rate", float("nan")),
                "hedging_score": hedge.get("hedging_score", float("nan")),
                "fvu": fvu,
                "interaction_delta_auc": case.get("interaction_delta_auc", float("nan")),
                "interaction_delta_auc_control": case.get("interaction_delta_auc_control", float("nan")),
                "interaction_selectivity": case.get("interaction_selectivity", float("nan")),
                "interaction_auc_additive": case.get("interaction_auc_additive", float("nan")),
                "interaction_auc_interaction": case.get("interaction_auc_interaction", float("nan")),
                "interaction_n_positive": case.get("interaction_n_positive"),
                "interaction_degenerate": case.get("interaction_degenerate", True),
            }
        )
    return records


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Overall means plus the interaction-tercile stratification -- the harness's headline output."""
    def _mean(key: str) -> float:
        vals = np.array([r.get(key, np.nan) for r in records], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    return {
        "n_cases": len(records),
        "overall": {key: _mean(key) for key in (*TERCILE_METRIC_KEYS, "interaction_selectivity")},
        "by_interaction_tercile": stratify_by_interaction_tercile(
            records,
            score_key="interaction_selectivity",
            metric_keys=TERCILE_METRIC_KEYS,
        ),
    }


# ---------------------------------------------------------------------------
# model-dependent pipeline (imports are lazy so this file loads on a CPU box
# without transformer_lens / a downloaded model)
# ---------------------------------------------------------------------------
def build_sae_adapter(
    run_dir: Path,
    checkpoint_name: str,
    checkpoint_step: int,
    model_name: str,
    layer: int,
    device: str,
    llm_dtype: str,
) -> Tuple[Any, Optional[str]]:
    """Load a local flat/Kron checkpoint as a SAEBench-compatible SAE.

    Reuses `src/eval_saebench_absorption.py`'s adapters verbatim rather than
    reimplementing them, so both absorption entry points see byte-identical SAEs.
    Returns (adapter, combine_rule) where combine_rule is None for flat SAEs.
    """
    from sae_bench.sae_bench_utils.general_utils import str_to_dtype

    from eval_saebench_absorption import build_adapter, load_checkpoint_spec

    spec = load_checkpoint_spec(run_dir, checkpoint_name, checkpoint_step)
    adapter = build_adapter(
        spec=spec,
        model_name=model_name,
        hook_layer=layer,
        device=torch.device(device),
        dtype=str_to_dtype(llm_dtype),
    )
    combine_rule = spec.cfg.get("combine_rule") if spec.dict_class == "KronAutoEncoderTopK" else None
    return adapter, combine_rule


def collect_word_activations(
    model: Any,
    sae: Any,
    words: Sequence[str],
    layer: int,
    icl_word_list: Sequence[str],
    batch_size: int = 32,
    base_template: str = "{word} has the first letter:",
    word_token_pos: int = -6,
    max_icl_examples: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Residual activations and SAE latent activations at the word token, for a word list.

    Uses the same ICL prompt construction as SAEBench's
    `FeatureAbsorptionCalculator`, so the activations the interaction probe sees
    are the same distribution the absorption metric was computed on.

    Prompts are grouped by exact token length before batching. That is not an
    optimization -- `word_token_pos` is a NEGATIVE index, so mixing token lengths
    in one padded batch would silently read the wrong position for every prompt
    but the longest, and the resulting activations would be garbage. SAEBench's
    own calculator hard-errors on mixed lengths for the same reason; here we
    group instead of erroring so no word has to be dropped.

    Returns (resid (n, d_model), sae_acts (n, d_sae), words_in_order). The word
    order is returned because grouping reorders the inputs, and callers must
    align labels against the returned order rather than the input order.
    """
    from sae_bench.evals.absorption.prompting import create_icl_prompt, first_letter_formatter

    prompts = [
        create_icl_prompt(
            word,
            examples=list(icl_word_list),
            base_template=base_template,
            answer_formatter=first_letter_formatter(),
            max_icl_examples=max_icl_examples,
            shuffle_examples=True,
            # Matches FeatureAbsorptionCalculator's own default, so prompts here
            # are constructed identically to the ones absorption was scored on.
            prepend_separator_to_first_example=True,
        )
        for word in words
    ]

    by_length: Dict[int, List[Any]] = {}
    for prompt in prompts:
        n_tok = int(model.to_tokens(prompt.base).shape[1])
        by_length.setdefault(n_tok, []).append(prompt)

    hook_point = f"blocks.{layer}.hook_resid_post"
    resid_chunks: List[torch.Tensor] = []
    act_chunks: List[torch.Tensor] = []
    ordered_words: List[str] = []
    with torch.inference_mode():
        for _, group in sorted(by_length.items()):
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                resid = model.run_with_cache(
                    [p.base for p in batch], names_filter=[hook_point]
                )[1][hook_point][:, word_token_pos, :]
                acts = sae.encode(resid)
                resid_chunks.append(resid.detach().float().cpu())
                act_chunks.append(acts.detach().float().cpu())
                ordered_words.extend(p.word for p in batch)
    return torch.cat(resid_chunks), torch.cat(act_chunks), ordered_words


def score_cases_with_interaction_probe(
    resid: torch.Tensor,
    sae_acts: torch.Tensor,
    w_dec: torch.Tensor,
    cases: Sequence[Dict[str, Any]],
    probe_directions: Dict[str, torch.Tensor],
    n_splits: int = 5,
    seed: int = 42,
    n_control_permutations: int = 3,
) -> List[Dict[str, Any]]:
    """Attach an interaction-structure score to each identified absorption case.

    For a case (letter L, absorbing latent c) the two primitives are:
        a = x . u_L                      -- the parent (first-letter) direction
        b = x . d_c                      -- the child-specific differentiator,
                                            i.e. latent c's decoder direction
                                            with the parent component removed
    and the target is `sae_acts[:, c] > 0`, i.e. "did the absorbing latent fire".

    The hypothesis under test is that absorbing latents are *conjunctions* --
    they fire on `parent AND differentiator` -- which shows up as a positive,
    control-corrected Delta-AUC. Latents that are merely linear mixtures score
    ~0 and land in the bottom tercile.
    """
    out: List[Dict[str, Any]] = []
    for case in cases:
        case = dict(case)
        letter = case["letter"]
        latent = int(case["absorbing_latent"])
        u = probe_directions.get(letter)
        if u is None:
            case["interaction_degenerate"] = True
            case["interaction_note"] = f"no probe direction for letter {letter}"
            out.append(case)
            continue

        u = u.detach().float().ravel()
        u = u / u.norm().clamp_min(1e-12)
        d_child = child_specific_direction(w_dec[latent], u)
        if float(d_child.norm()) < 1e-8:
            case["interaction_degenerate"] = True
            case["interaction_note"] = "absorbing latent has no child-specific direction"
            out.append(case)
            continue

        a = (resid @ u).numpy()
        b = (resid @ d_child).numpy()
        score = interaction_structure_score(
            latent=sae_acts[:, latent].numpy(),
            a=a,
            b=b,
            n_splits=n_splits,
            seed=seed,
            binarize_threshold=0.0,
            n_control_permutations=n_control_permutations,
        )
        case.update(
            {
                "interaction_delta_auc": score.delta_auc,
                "interaction_delta_auc_control": score.delta_auc_control,
                "interaction_selectivity": score.selectivity,
                "interaction_auc_additive": score.auc_additive,
                "interaction_auc_interaction": score.auc_interaction,
                "interaction_n_positive": score.n_positive,
                "interaction_degenerate": score.degenerate,
                "interaction_note": score.note,
            }
        )
        out.append(case)
    return out


def cases_from_absorption_df(
    raw_df: Any,
    min_absorption_fraction: float = 0.0,
) -> List[Dict[str, Any]]:
    """Collapse SAEBench's per-token absorption rows into one case per (letter, absorbing latent).

    SAEBench emits one row per probe-true-positive token; the unit this analysis
    reasons about is the *latent* that did the absorbing, so rows are grouped by
    (`letter`, `top_projection_feat`) and their absorption statistics averaged.
    """
    cases: List[Dict[str, Any]] = []
    grouped = raw_df[raw_df["absorption_fraction"] > min_absorption_fraction].groupby(
        ["letter", "top_projection_feat"]
    )
    for (letter, latent), grp in grouped:
        cases.append(
            {
                "letter": str(letter),
                "absorbing_latent": int(latent),
                "n_words": int(grp.shape[0]),
                "absorption_fraction": float(grp["absorption_fraction"].mean()),
                "full_absorption_rate": float(grp["is_full_absorption"].mean()),
                "main_feature_ids": [int(i) for i in grp["split_feats"].iloc[0]],
            }
        )
    return cases


def run_absorption_hedging_eval(args) -> Dict[str, Any]:
    """Full pipeline for one (checkpoint, combine_rule) cell. Requires a GPU-class box.

    Order of operations: load SAE -> SAEBench k-sparse probing + absorption ->
    group into cases -> collect activations -> hedging -> FVU -> interaction
    score -> tercile stratification.
    """
    from sae_bench.evals.absorption.common import PROBES_DIR, load_or_train_probe
    from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
    from sae_bench.evals.absorption.feature_absorption import run_feature_absortion_experiment
    from sae_bench.evals.absorption.k_sparse_probing import run_k_sparse_probing_experiment
    from sae_bench.evals.absorption.vocab import LETTERS, get_alpha_tokens
    from sae_bench.sae_bench_utils.general_utils import str_to_dtype
    from transformer_lens import HookedTransformer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SAEBench's own config supplies the task constants (prompt template, token
    # position, k-sparse probing hyperparameters, f1 thresholds). Taking them from
    # here rather than re-declaring them keeps this harness's absorption numbers
    # directly comparable to src/eval_saebench_absorption.py's.
    cfg = AbsorptionEvalConfig(
        model_name=args.model_name,
        random_seed=args.seed,
        llm_batch_size=args.llm_batch_size,
        llm_dtype=args.llm_dtype,
    )

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name, device=args.device, dtype=str_to_dtype(args.llm_dtype)
    )
    # Auto-trains and caches the ground-truth first-letter linear probe under
    # PROBES_DIR the first time a given (model_name, layer) pair is evaluated --
    # SAEBench ships this probe only for the model/layer combos its own paper
    # covers (pythia-410m/layer_12 in this repo's cache), so any new model
    # (e.g. bigscience/bloom-560m for this project's cross-model-family RQ)
    # needs the probe trained once, after which load_probe alone would suffice.
    probe = load_or_train_probe(
        model=model,
        base_template=cfg.prompt_template,
        pos_idx=cfg.prompt_token_pos,
        layer=args.layer,
        probes_dir=PROBES_DIR,
        device=args.device
    )
    probe_directions = {letter: probe.weights[i].detach().cpu() for i, letter in enumerate(LETTERS)}

    results: Dict[str, Any] = {
        "settings": {
            "model_name": args.model_name,
            "layer": args.layer,
            "run_dir": str(run_dir),
            "checkpoint_step": args.checkpoint_step,
            "llm_dtype": args.llm_dtype,
            "seed": args.seed,
            "n_probe_splits": args.n_probe_splits,
            "n_control_permutations": args.n_control_permutations,
            "max_activation_words": args.max_activation_words,
            "prompt_template": cfg.prompt_template,
            "prompt_token_pos": cfg.prompt_token_pos,
            "max_k_value": cfg.max_k_value,
            "min_GT_probe_f1": cfg.min_GT_probe_f1,
        },
        "cells": {},
    }

    # The full alpha-token vocabulary is the ICL example POOL (matching
    # FeatureAbsorptionCalculator); `max_icl_examples` draws a subset per prompt,
    # and create_icl_prompt's contamination check keeps the target word out of
    # its own examples. `probe_words` is the sample the interaction probe is
    # fitted on -- the whole vocabulary, not just the absorbed tokens, so the
    # probe sees genuine negatives as well as positives.
    all_words = get_alpha_tokens(model.tokenizer)
    icl_words = all_words
    probe_words = all_words[: args.max_activation_words]

    for checkpoint_name in args.checkpoint_names:
        sae, combine_rule = build_sae_adapter(
            run_dir=run_dir,
            checkpoint_name=checkpoint_name,
            checkpoint_step=args.checkpoint_step,
            model_name=args.model_name,
            layer=args.layer,
            device=args.device,
            llm_dtype=args.llm_dtype,
        )

        # k-sparse probing MUST run first: run_feature_absortion_experiment reads
        # its on-disk output to decide each letter's main ("split") latents and
        # its probe-true-positive token list. Calling absorption alone silently
        # depends on a stale or missing artifact.
        k_sparse_df = run_k_sparse_probing_experiment(
            model=model,
            sae=sae,
            layer=args.layer,
            sae_name=checkpoint_name,
            max_k_value=cfg.max_k_value,
            prompt_template=cfg.prompt_template,
            prompt_token_pos=cfg.prompt_token_pos,
            device=args.device,
            force=args.force_rerun,
            f1_jump_threshold=cfg.f1_jump_threshold,
            k_sparse_probe_l1_decay=cfg.k_sparse_probe_l1_decay,
            k_sparse_probe_batch_size=cfg.k_sparse_probe_batch_size,
            k_sparse_probe_num_epochs=cfg.k_sparse_probe_num_epochs,
            eval_batch_size=cfg.eval_k_sparse_probe_batch_size,
        )
        n_good_letters = int((k_sparse_df["f1_probe"] > cfg.min_GT_probe_f1).sum())
        if n_good_letters < cfg.min_feats_for_eval:
            # Same guard SAEBench's own runner applies; without it the absorption
            # numbers below would be computed over letters whose ground-truth
            # probe does not work, which is not a result.
            results["cells"][checkpoint_name] = {
                "error": (
                    f"only {n_good_letters} letters clear min_GT_probe_f1={cfg.min_GT_probe_f1} "
                    f"(need {cfg.min_feats_for_eval}); absorption not evaluable for this model"
                ),
            }
            continue

        raw_df = run_feature_absortion_experiment(
            model=model,
            sae=sae,
            layer=args.layer,
            sae_name=checkpoint_name,
            max_k_value=cfg.max_k_value,
            prompt_template=cfg.prompt_template,
            prompt_token_pos=cfg.prompt_token_pos,
            device=args.device,
            force=args.force_rerun,
            feature_split_f1_jump_threshold=cfg.f1_jump_threshold,
            batch_size=args.llm_batch_size,
        )
        cases = cases_from_absorption_df(raw_df, min_absorption_fraction=args.min_absorption_fraction)

        resid, sae_acts, _ordered_words = collect_word_activations(
            model=model,
            sae=sae,
            words=probe_words,
            layer=args.layer,
            icl_word_list=icl_words,
            batch_size=args.llm_batch_size,
            # Same template and read position the absorption metric used, so the
            # interaction probe reasons about the same activations.
            base_template=cfg.prompt_template,
            word_token_pos=cfg.prompt_token_pos,
            max_icl_examples=args.n_icl_examples,
        )
        with torch.inference_mode():
            x_hat = sae.decode(sae_acts.to(sae.device, dtype=sae.dtype)).float().cpu()
        fvu = compute_fvu(resid, x_hat)

        w_dec = sae.W_dec.detach().float().cpu()
        hedging_by_letter: Dict[str, Dict[str, Any]] = {}
        for letter in sorted({c["letter"] for c in cases}):
            letter_cases = [c for c in cases if c["letter"] == letter]
            hedging_by_letter[letter] = hedging_score_from_decoders(
                w_dec=w_dec,
                main_feature_ids=letter_cases[0]["main_feature_ids"],
                child_feature_ids=[c["absorbing_latent"] for c in letter_cases],
                probe_direction=probe_directions[letter],
            )

        scored = score_cases_with_interaction_probe(
            resid=resid,
            sae_acts=sae_acts,
            w_dec=w_dec,
            cases=cases,
            probe_directions=probe_directions,
            n_splits=args.n_probe_splits,
            seed=args.seed,
            n_control_permutations=args.n_control_permutations,
        )
        records = build_case_records(
            per_case=scored,
            hedging_by_letter=hedging_by_letter,
            fvu=fvu,
            sae_name=checkpoint_name,
            combine_rule=combine_rule,
        )
        results["cells"][checkpoint_name] = {
            "combine_rule": combine_rule,
            "dict_size": int(sae.cfg.d_sae),
            "fvu": fvu,
            "hedging_by_letter": hedging_by_letter,
            "records": records,
            "summary": summarize_records(records),
        }

    out_path = output_dir / args.output_name
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved absorption/hedging results: {out_path}", flush=True)
    return results


def print_summary(results: Dict[str, Any]) -> None:
    for name, cell in results.get("cells", {}).items():
        s = cell["summary"]
        strat = s["by_interaction_tercile"]
        print(f"\n{name} (combine_rule={cell['combine_rule']}, FVU={cell['fvu']:.4f}, n_cases={s['n_cases']})")
        if "terciles" not in strat:
            print(f"  {strat.get('error', 'no tercile summary')}")
            continue
        hdr = f"  {'tercile':>8} {'n':>4} {'intscore':>9} " + " ".join(f"{k:>18}" for k in TERCILE_METRIC_KEYS)
        print(hdr)
        for tercile in ("bottom", "middle", "top"):
            row = strat["terciles"][tercile]
            vals = " ".join(f"{row[k]['mean']:18.4f}" for k in TERCILE_METRIC_KEYS)
            print(f"  {tercile:>8} {row['n']:4d} {row['interaction_score_mean']:9.4f} {vals}")
        tmb = " ".join(f"{strat['top_minus_bottom'][k]:18.4f}" for k in TERCILE_METRIC_KEYS)
        print(f"  {'top-bot':>8} {'':4} {'':>9} {tmb}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Absorption + hedging benchmark, stratified by interaction tercile")
    ap.add_argument("--run_dir", type=str, default=str(ROOT / "runs" / "pilot_balanced_v2_420k"))
    ap.add_argument("--checkpoint_step", type=int, default=420000)
    ap.add_argument("--checkpoint_names", nargs="+", default=["kron_pilot_sentiment", "flat_pilot_sentiment"])
    # pythia-410m matches this project's existing pipeline (src/train.py --model_name,
    # src/eval_saebench_absorption.py); gemma-2-2b is a drop-in substitution for the
    # cross-model-family question -- nothing below assumes a model family.
    ap.add_argument("--model_name", type=str, default="pythia-410m")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--llm_dtype", type=str, default="float32")
    ap.add_argument("--llm_batch_size", type=int, default=32)
    ap.add_argument("--n_icl_examples", type=int, default=10)
    ap.add_argument("--max_activation_words", type=int, default=2000,
                    help="Words used to build the interaction probe's sample; more words = tighter Delta-AUC.")
    ap.add_argument("--min_absorption_fraction", type=float, default=0.0,
                    help="Only tokens above this absorption_fraction contribute to a case.")
    ap.add_argument("--n_probe_splits", type=int, default=5)
    ap.add_argument("--n_control_permutations", type=int, default=3,
                    help="Hewitt & Liang control-task repetitions per interaction probe.")
    ap.add_argument("--force_rerun", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default=str(ROOT / "runs" / "absorption_hedging"))
    ap.add_argument("--output_name", type=str, default="absorption_hedging_results.json")
    ap.add_argument("--self_test", action="store_true", help="Run the CPU-only self-test and exit.")
    return ap


# ---------------------------------------------------------------------------
# CPU-only self-test
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Tiny synthetic check of every model-free component. NOT a scientific validation.

    Exercises FVU, the child-specific direction, the hedging score, case
    construction, interaction scoring on a hand-built conjunctive latent, record
    building and tercile stratification -- asserting only shapes, ranges and
    the sign relations the definitions require.
    """
    ok = True
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # --- FVU ---
    x = torch.randn(200, 16)
    fvu_perfect = compute_fvu(x, x)
    fvu_mean = compute_fvu(x, x.mean(dim=0, keepdim=True).expand_as(x))
    checks = {
        "fvu_perfect_is_zero": abs(fvu_perfect) < 1e-9,
        "fvu_mean_predictor_is_one": abs(fvu_mean - 1.0) < 1e-6,
    }

    # --- child-specific direction ---
    d_model = 32
    u = torch.zeros(d_model)
    u[0] = 1.0
    pure_parent = torch.zeros(d_model)
    pure_parent[0] = 3.0
    mixed = torch.zeros(d_model)
    mixed[0], mixed[5] = 2.0, 1.0
    checks["pure_parent_has_no_differentiator"] = float(child_specific_direction(pure_parent, u).norm()) < 1e-6
    diff = child_specific_direction(mixed, u)
    checks["differentiator_is_unit_norm"] = abs(float(diff.norm()) - 1.0) < 1e-5
    checks["differentiator_orthogonal_to_parent"] = abs(float(diff @ u)) < 1e-6

    # --- hedging: a hedged parent points at its children; a clean parent does not ---
    w_dec = torch.zeros(6, d_model)
    w_dec[0, 0] = 1.0                      # clean parent: parent direction only
    w_dec[1, 0], w_dec[1, 5] = 1.0, 1.0    # hedged parent: parent + child-1 direction
    w_dec[2, 0], w_dec[2, 5] = 0.5, 1.0    # child 1
    w_dec[3, 0], w_dec[3, 6] = 0.5, 1.0    # child 2
    clean = hedging_score_from_decoders(w_dec, [0], [2, 3], u)
    hedged = hedging_score_from_decoders(w_dec, [1], [2, 3], u)
    empty = hedging_score_from_decoders(w_dec, [], [2, 3], u)
    checks["clean_parent_hedging_is_nan_or_zero"] = (
        not np.isfinite(clean["hedging_score"]) or clean["hedging_score"] < 1e-6
    )
    checks["hedged_parent_hedging_positive"] = hedged["hedging_score"] > 0.1
    checks["hedging_in_[0,1]"] = 0.0 <= hedged["hedging_score"] <= 1.0
    checks["hedging_empty_is_nan"] = not np.isfinite(empty["hedging_score"])

    # --- interaction scoring on a synthetic conjunctive latent ---
    n = 600
    resid = torch.randn(n, d_model)
    latent_idx = 2
    a_vals = resid @ u
    b_dir = child_specific_direction(w_dec[latent_idx], u)
    b_vals = resid @ b_dir
    sae_acts = torch.zeros(n, 6)
    sae_acts[:, latent_idx] = ((a_vals > 0) & (b_vals > 0)).float()      # conjunctive
    sae_acts[:, 3] = (resid @ u > 0).float()                             # purely additive
    cases = [
        {"letter": "a", "absorbing_latent": 2, "n_words": 50, "absorption_fraction": 0.7,
         "full_absorption_rate": 0.4, "main_feature_ids": [1]},
        {"letter": "a", "absorbing_latent": 3, "n_words": 40, "absorption_fraction": 0.2,
         "full_absorption_rate": 0.1, "main_feature_ids": [1]},
    ]
    scored = score_cases_with_interaction_probe(
        resid=resid, sae_acts=sae_acts, w_dec=w_dec, cases=cases,
        probe_directions={"a": u}, n_splits=5, seed=3, n_control_permutations=2,
    )
    conj_sel = scored[0]["interaction_selectivity"]
    add_sel = scored[1]["interaction_selectivity"]
    checks["conjunctive_case_scored"] = np.isfinite(conj_sel)
    checks["conjunctive_selectivity_positive"] = conj_sel > 0.005
    checks["conjunctive_beats_additive_case"] = conj_sel > add_sel
    checks["no_nan_in_aucs"] = np.isfinite(scored[0]["interaction_auc_interaction"])

    # --- records + tercile stratification ---
    records = build_case_records(
        per_case=scored,
        hedging_by_letter={"a": hedged},
        fvu=0.25,
        sae_name="selftest_sae",
        combine_rule="mand",
    )
    checks["records_have_all_metric_keys"] = all(
        all(k in r for k in TERCILE_METRIC_KEYS) for r in records
    )
    synthetic_records = [
        {"interaction_selectivity": s, "absorption_fraction": 1.0 - s, "full_absorption_rate": 0.5 - s / 2,
         "hedging_score": s, "fvu": 0.3}
        for s in np.linspace(0.0, 0.8, 9)
    ]
    summary = summarize_records(synthetic_records)
    strat = summary["by_interaction_tercile"]
    checks["tercile_buckets_are_balanced"] = (
        strat["terciles"]["bottom"]["n"] == strat["terciles"]["top"]["n"] == 3
    )
    checks["tercile_top_minus_bottom_signs"] = (
        strat["top_minus_bottom"]["absorption_fraction"] < 0
        and strat["top_minus_bottom"]["hedging_score"] > 0
    )
    checks["overall_means_finite"] = all(np.isfinite(v) for v in summary["overall"].values())

    # --- printing path must not crash on a real-shaped payload ---
    try:
        print_summary({"cells": {"selftest_sae": {
            "combine_rule": "mand", "fvu": 0.25, "records": synthetic_records,
            "summary": summarize_records(synthetic_records)}}})
        checks["print_summary_runs"] = True
    except Exception as exc:  # pragma: no cover
        print(f"  print_summary raised: {exc}")
        checks["print_summary_runs"] = False

    print()
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
        ok = ok and bool(passed)
    print(
        f"\n  (reference values: fvu_perfect={fvu_perfect:.2e} fvu_mean={fvu_mean:.4f} "
        f"hedged={hedged['hedging_score']:.4f} conj_sel={conj_sel:+.4f} add_sel={add_sel:+.4f})"
    )
    print("\nABSORPTION_HEDGING SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.self_test:
        return _self_test()
    results = run_absorption_hedging_eval(args)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
