# Paper-Scale Experimental Design

Rebuilds the exploratory sweep documented in `REPORT_FULL.md` (§1-19.13) into a
paper-shaped experimental section: fixed research questions, a controlled
architecture/scale grid, one primary evaluation axis instead of an
open-ended search, and a compute budget. Every design choice below is
justified by citing which prior exploratory result motivated it — this is a
rebuild, not a fresh start.

---

## 0. What changes vs. the exploratory phase

| Axis | Exploratory phase (§1-19.13) | Paper-scale rebuild |
|---|---|---|
| Research question | "Does *some* mechanism make KronSAE disentangle?" (open search over 10+ mechanisms) | Two fixed RQs (§1 below), no further mechanism search |
| Base LM | Pythia-410M only | Pythia-410M (kept, for continuity with all prior results) + one larger/different-family model to test generality |
| Concept pair | Amazon Reviews sentiment × topic only | Sentiment × topic (kept as primary) + one independent concept pair on a second dataset |
| Capacity grid | 3 points (P≫Q, swapped, equal) — §19.9 showed 3 points already breaks a monotonic story | 5-6 capacity ratios, matched compute at every point, both branches and both datasets |
| Steering eval | 6 prompts (§19.11) → 22 prompts/44 samples (§19.12), single layer, one model | ~150-200 prompts × multiple sampling seeds, primary layer + 1 robustness layer, both models |
| Statistics | Point estimates, occasional noise flagged post hoc (§19.12's ranking-reversal) | Every headline number gets seed-level error bars / bootstrap CIs before being reported |
| Erasure axis | Extensive (§3-18), already exhausted | Frozen: keep LEACE-vs-KronSAE result as a single confirmatory table, no new mechanism runs |
| Human/LLM judgment of generated text | None (only classifier-probe + perplexity proxies) | Add an LLM-judge coherence/fluency score alongside self-perplexity, since §19.11-12 showed perplexity alone can look ambiguous |

---

## 1. Research questions (fixed, no further search)

- **RQ1 (erasure, confirmatory only).** Does any KronSAE training-time
  mechanism close the gap to LEACE's closed-form suppression? *(Already
  answered no, ten times over, in §3-18; paper-scale work here is limited to
  one clean confirmatory table at final scale — not new mechanism search.)*
- **RQ2 (steering, primary contribution).** Does compositional (branch-factored)
  SAE structure produce *more coherent* causal steering than an equal-capacity
  undivided SAE, when measured at the generated-text level rather than the
  reconstruction level — and is this a stable finding across model scale,
  dataset, and concept pair, or an artifact of the single Pythia-410M /
  Amazon-Reviews setting it was found in?
- **RQ3 (mechanism, secondary).** Is steering dominance explained by branch
  capacity alone (§19.6-19.9's finding: partially — capacity matters but a
  5x asymmetry survives at equal capacity), and does that residual asymmetry
  replicate at paper scale or shrink with more capacity points / more data?

---

## 2. Architecture grid

| Architecture | Config | Role |
|---|---|---|
| KronSAE, capacity-asymmetric | P:Q ∈ {8:1, 4:1, 2:1, 1:1, 1:2, 1:4} at fixed total width `h·(m+n)` | Primary capacity sweep (RQ3), replaces the 3-point grid |
| KronSAE, matched to flat total width | `h·(m+n) = 16384` (same total as flat baseline) | Primary architecture-vs-architecture comparison (RQ2) |
| Flat TopK SAE (Gao et al. 2024 arch.) | `dict_size = 16384`, same `k` | Steering ceiling / coherence-cost baseline |
| LEACE (closed-form, no dictionary) | rank-1 (binary concept) or rank-(C-1) (multiclass) | Erasure-axis ceiling (RQ1) — not a steering baseline, kept separate |
| (Optional, if budget allows) Matryoshka SAE or JumpReLU SAE | matched total width | Generalizes RQ2 beyond "Kron vs. flat" to "any hierarchical/compositional SAE vs. flat" |

All non-LEACE architectures trained with identical: optimizer, LR schedule,
`k`, total steps, effective batch size (via `grad_accum_steps`, per the DDP-parity
fix already implemented), and random seed set (≥3 seeds per config for
variance estimates — the exploratory phase ran single-seed almost throughout,
which is the main statistical gap to close).

---

## 3. Models and datasets

| Component | Exploratory phase | Paper-scale addition | Why |
|---|---|---|---|
| Base LM | Pythia-410M, layer 12 | + **Pythia-1.4B (confirmed)**, layer scaled to the equivalent relative depth (Pythia-410M has 24 layers, layer 12 = 50% depth; Pythia-1.4B also has 24 layers, so layer 12 stays the matched depth — verify exact layer count before wiring the hook) | Same-family scale-up, user-confirmed |
| Dataset | Amazon Reviews (sentiment × topic, 6 topics × 2 sentiments) | **Kept as the only dataset** (user decision: no second dataset/concept pair) — full-scale (not the 500k-doc cache subset used for pilots), identical across both model-scale cells | Continuity with all prior sections; isolates model scale as the only varying factor |

**Scope decision (user-confirmed):** the grid is now a single dataset ×
2 models (410M, 1.4B) × architecture/capacity grid — the earlier "second
dataset generality cell" (§3, prior draft) is dropped. This means RQ2's
"is the coherence finding Amazon-Reviews-specific" question is out of scope
for this round; what the paper-scale rebuild tests instead is purely
whether the finding holds as *model scale* changes, with dataset held fixed.
Every other design element (capacity sweep, matched-capacity comparison,
evaluation protocol, ablations) still runs at both model scales, on the
same Amazon Reviews cache.

### 3.1 Dataset-volume parity across the 410M / 1.4B comparison (user-mandated)

Same principle as the DDP-parity fix already applied earlier in this project
(equal effective batch × equal step count, via `grad_accum_steps`, regardless
of per-GPU microbatch size): the 410M and 1.4B cells must each train the SAE
on the **same total number of activation-cache tokens/docs**, not more data
for the bigger model just because it "can afford" more compute. Concretely:

- Fix `total_docs_seen` (or equivalently `total_tokens_seen`) as a constant
  across both model-scale cells — same value already used for the 410M
  pipeline (the 420,000-step / effective-batch-64 real-data runs), not a
  larger number for 1.4B.
- The SAE-side compute (dictionary training) is essentially model-size-agnostic
  in FLOPs — it operates on cached `layer-12` activations, not on the LM
  itself — so the part that scales with model size is only the one-time
  **activation-caching forward pass** (extracting Pythia-1.4B's layer-12
  hidden states over the same document set), not the SAE training loop.
  That forward pass is ~3.4x the 410M model's parameter count, so budget
  proportionally more wall-clock for caching, but the SAE training step count,
  batch size, and total docs/tokens stay identical to the 410M cell.
- If 1.4B's activation dimensionality differs from 410M's (Pythia-1.4B's
  hidden size is 2048 vs. 410M's 1024), all SAE-side widths (`h`, `m`, `n`,
  flat `dict_size`) scale with `activation_dim` per the existing convention
  in `train.py`, but **total docs/tokens processed does not scale** — the
  two model-scale cells must be directly comparable on "how much data did
  the SAE see," only differing in "which model generated the activations."
- Concretely: reuse `--grad_accum_steps` (already implemented) to hit the
  same effective batch size on one GPU for both models, adjusting only the
  per-step microbatch size downward for 1.4B if its larger activation cache
  and larger one-time forward pass leave less headroom in the shared 128GB
  RAM / single-GPU budget — never by processing fewer total docs.

---

## 4. Evaluation protocol

### 4.1 Erasure axis (RQ1) — confirmatory, not exploratory
One final table: standardized-probe accuracy (regularized, matched to
random-init baseline per §9.4) for {LEACE, best-of-ten KronSAE mechanisms}
on both branches, both concept labels, at final paper-scale training length.
No new mechanism variants introduced at this stage.

### 4.2 Steering axis (RQ2/RQ3) — primary
Two-layer evaluation, replacing the single reconstruction-then-decode check:
1. **Reconstruction-level** (fast, cheap, run at every capacity/alpha point):
   FVU, probe-based `P(concept)`, matching the existing `eval_real_intervention.py` /
   `eval_synthetic_flat_intervention.py` methodology — kept as a *screening*
   metric, not the final claim, per the user's original critique that FVU
   alone conflates "broken" with "successfully steered."
2. **Generation-level** (expensive, run only at the alpha/capacity points
   the screening pass flags as interesting — not the full grid):
   - patch-and-generate into real text (existing `eval_patch_generate.py`
     machinery, scaled from 22 to ~150-200 balanced prompts, ≥3 sampled
     continuations each, both greedy and temperature sampling)
   - sentiment/topic re-scored via the external classifier (as now)
   - self-perplexity (as now) **plus** an LLM-judge fluency/coherence score
     (0-5 scale, blind to architecture identity, batched prompts) — added
     because §19.12 showed perplexity alone can leave "which architecture is
     more broken" ambiguous or noisy at small n
   - bootstrap CIs over prompts × seeds for every headline number

### 4.3 Statistical protocol (new — largely absent in the exploratory phase)
- ≥3 training seeds per architecture/capacity config.
- Bootstrap or seed-level CIs on every table cell reported in the paper.
- Any "architecture A beats architecture B" claim requires non-overlapping
  CIs, not just a point-estimate gap — directly motivated by §19.12's own
  correction of a §19.11 ranking that turned out to be noise.

---

## 5. Ablations (run once at the primary Pythia-410M × Amazon-Reviews cell)

- Combine rule: mAND vs. mOR vs. concat (already implemented, §8 — rerun only
  at final scale for the steering axis, not full mechanism sweep).
- `k` (top-k sparsity level) sweep, since steering range plausibly interacts
  with sparsity independent of capacity.
- Layer choice: primary layer (matching existing layer-12 cache) + one other
  layer, to check the coherence-advantage finding isn't layer-specific.
- Alpha range: extend beyond the current `{-6...6}` grid only if the
  paper-scale generation-level screen shows the interesting regime sits
  outside it.

---

## 6. Staging (mirrors the pilot→mid→full pattern already used in this project)

1. **Stage A — synthetic, cheap.** Full capacity grid (6 points) + both
   architectures, **both model scales** (410M and 1.4B activation dims,
   synthetic buffer just needs `activation_dim` set to match each), synthetic
   diagnostic buffer only. Validates the design and catches bugs before
   spending real-data compute. (~hours, single GPU.)
2. **Stage B — real data, Pythia-410M × Amazon Reviews.** Full capacity grid,
   reconstruction-level screening for all points, generation-level eval only
   at flagged points. (~1-2 days, matches the existing `idea_size_*_pipeline.sh`
   scale — this is a rerun/extension of work already largely done in
   §19.6-19.12, now with ≥3 seeds instead of 1.)
3. **Stage C — real data, Pythia-1.4B × Amazon Reviews (same dataset, new
   model).** Same capacity grid and evaluation protocol as Stage B, run on
   Pythia-1.4B's layer-12 (matched relative depth) activations over the
   *same* Amazon Reviews document set and *same* total docs/tokens (§3.1) —
   this is the whole point of the model-scale comparison, so it gets the
   full grid, not a reduced single-config check like the dropped
   second-dataset plan.
4. **Stage D — final tables.** Aggregate seeds, compute CIs across both
   model scales, finalize the erasure-axis confirmatory table (kept at
   410M only — RQ1 is frozen/confirmatory, not part of the scale comparison),
   assemble paper figures/tables.

---

## 7. Rough compute budget (single-GPU, `CUDA_VISIBLE_DEVICES=0` constraint kept)

| Stage | Runs | Approx. cost each | Approx. total |
|---|---|---|---|
| A (synthetic, both models) | 6 capacity points × 2 arch × 3 seeds × 2 models = 72 | minutes | < 1 GPU-day |
| B (real, Pythia-410M) | 6 capacity points × 2 arch × 3 seeds = 36 full-scale (420k-step) runs | ~4 days each per §19.7's real-data run cost | **dominant cost — see de-scoping below** |
| C (real, Pythia-1.4B, same data volume) | 6 capacity points × 2 arch × 3 seeds = 36 full-scale runs | ~4 days × ~3.4x forward-pass overhead for the larger model's one-time activation caching (SAE training step count/data volume unchanged per §3.1, so per-step cost is close to Stage B's) | also dominant |
| D | eval-only, no training | hours | < 1 GPU-day |

**Note:** Stage B+C at literal full scale (36+36 four-ish-day runs on one
GPU) is still well over 100 GPU-days serialized — not realistic within a
normal paper timeline on one GPU. Recommended de-scoping, in priority order:
1. Reduce seeds-per-config from 3 to 2 for the capacity sweep specifically
   (keep 3 seeds only for the headline matched-capacity comparison, at both
   model scales).
2. Shorten the "full-scale" step count for capacity-sweep points that Stage
   A's synthetic screen shows are not near the interesting regime (i.e. only
   the 2-3 capacity points closest to the equal-capacity/swap boundary get
   the full 420k steps; the rest get the mid-scale 17.5k-step budget already
   used for probe-only checks in this project) — applied identically at both
   model scales so the comparison stays apples-to-apples.
3. If Stage C still doesn't fit budget even after (1)-(2), reduce Stage C to
   the matched-capacity config only (Kron vs. flat at equal total width) plus
   the 1-2 capacity points nearest the equal-capacity/swap boundary that
   Stage B/§19.9 already flagged as the interesting region, rather than the
   full 6-point grid — narrower than originally planned, but still answers
   the primary RQ2 model-scale question, just not the full RQ3 capacity
   curve at 1.4B.

This still needs the user's sign-off on total wall-clock budget before Stage
B/C are launched at any step-count larger than what's already been run.

---

## 8. Open decisions needing your input before build-out

Resolved: second base model is **Pythia-1.4B** (§2/§3), dataset stays
**Amazon Reviews only, no second dataset** (§3, scope decision) — model
scale is the sole new axis this round.

Still open:
- Whether LLM-judge scoring is worth the added engineering (batched prompt
  calls to a judge model) vs. staying with perplexity + classifier probes
  and just adding more samples/seeds to shrink the CI on those.
- Final compute ceiling (in GPU-days) you're willing to commit, which
  determines how aggressively Stage B/C get de-scoped per §7 (default plan:
  reduce seeds 3→2 off the capacity sweep, then shorten step count for
  capacity points away from the interesting region, then — only if still
  over budget — narrow Stage C's grid to the matched-capacity config plus
  the boundary points nearest §19.9's equal-capacity result).
