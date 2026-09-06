# Paper Plan — Branch-Structured SAEs as Control Interfaces

Status: draft plan, written 2026-09-02, superseding `PAPER_SCALE_EXPERIMENT_DESIGN.md`'s RQ framing
(that document's compute/parity mechanics in §3.1, §6, §7 remain valid and are referenced here rather
than repeated). Grounded only in pilot findings that survived verification; the abandoned
collect-and-suppress RQ is retained solely as a scoped negative result (§C1), not as a thesis.

---

## 0. Title

**Primary:** *Addressable, Not Separable: What Branch-Structured Sparse Autoencoders Buy You for Steering*

Alternates:
- *Collection Without Dispersion: Supervised Kronecker SAEs Route Concepts but Do Not Disentangle Them*
- *Steering Needs an Address, Erasure Needs a Projection: Scoping Structured SAEs Against LEACE*

The title has to do two jobs at once, because the pilots produced one strong positive and one strong
negative and the negative is the more surprising half. "Addressable" is the positive (you know before
training which branch a concept lands in, and you can push it at generation time). "Not separable" is
the negative (the other concept is still in there; LEACE removes it and we cannot). The colon
separates the capability from its boundary.

---

## 1. Thesis

A branch-structured (Kronecker-factored) SAE trained with concept supervision gives you an
**architecturally addressable subspace**: a named branch that reliably carries a designated concept,
which you can intervene on at generation time without first having to search the dictionary for the
right feature. It does **not** give you a disentangled subspace: the non-designated concept remains
decodable from the same branch, and no training-time suppression mechanism we tested changes that,
while a closed-form projection (LEACE) achieves it trivially and without a dictionary at all.

The paper's job is therefore to (a) establish addressability as a real, measurable, architecture-
attributable property, (b) show what it is worth at the only place control actually matters —
generated text — and (c) draw the boundary honestly against the erasure literature.

---

## 2. Research questions

| RQ | Question | Maps to your control variable | Primary evidence |
|----|----------|-------------------------------|------------------|
| **RQ1** | Does concept supervision reliably route a designated concept to a designated branch, and is routing *collection-only*? | (1) label segregation, (2) DPO merges but does not disperse | Branch probe accuracy (collection) vs. off-concept probe accuracy on the same branch (suppression); 16-mechanism suppression sweep as the negative |
| **RQ2** | Does branch structure change the *geometry* of the learned dictionary relative to a flat SAE of equal capacity — and relative to no SAE at all? | (3) feature validation | Participation ratio, subspace overlap, probe accuracy, **dictionary-collapse suite** (Gini / dead features / top-k concentration) |
| **RQ3** | Does branch addressability produce a better **control frontier** at generation time — sentiment shift per unit of collateral damage? | (4) steering + perplexity + sub-SAE size ablation | Neutral-prompt patch-and-generate, external LLM judge, selectivity + fluency frontier |
| **RQ4** | Is the target concept incidental — does the same structure hold when the steered concept is *topic* rather than *sentiment*, and when branch roles are swapped? | (5) topic steering | Label-swap experiment (**new**) + topic steering |
| **RQ5** | Do RQ1–RQ4 hold across model scale? | (6) Pythia suite | Pythia-410M → 1.4B, identical data volume |

RQ4 is deliberately placed before RQ5: a scale study on a result that turns out to be
sentiment-specific is wasted compute.

---

## 3. Contributions (claimable now, from pilot evidence)

**C1 — Routing is collection-only, and that scopes the whole structured-SAE-for-disentanglement
program.** Supervision reliably concentrates a concept in a designated branch (high branch probe
accuracy, stable across capacities and seeds). No mechanism we tested removes the *other* concept
from that branch: 16 mechanisms across four families — objective design (conditional reconstruction,
swap-and-reconstruct, adversarial/GRL, RL-REINFORCE), capacity restriction, optimization schedule
(BCD/ALS alternation, gradient surgery), and representation constraints — either failed or damaged
collection. LEACE, a closed-form whitened projection with no dictionary, succeeds. **Claim:**
structured SAEs should be evaluated as control interfaces, not as erasure operators.

**C2 — Branch geometry is set by supervised role, not by capacity.** Swapping P and Q widths
(1024↔2048) leaves the asymmetry intact: the sentiment-supervised branch is always the entangled,
low-FVU-under-steering one (direction overlap 0.9457 control / 0.8751 swapped), the topic-supervised
branch always the separated, high-FVU one (0.0949 / 0.1870). Three independent measurements —
direction overlap, FVU under steering, steering span — agree. *Needs the label-swap experiment (§5.4)
to be airtight; without it "role" and "concept" are confounded.*

**C3 — Flat TopK SAEs undergo dictionary collapse on real data at matched budget; the factored one
does not.** On real Pythia-410M activations at 420k steps, the flat SAE (dict 16384, k=24) puts
35.10% of all token firings on a single feature, has Gini 0.7676 over firing frequency, and a
document-level participation ratio of **1.02 out of a possible 16384** — while its sentiment and topic
discriminative directions become effectively collinear (overlap **0.9995**). Kron, on identical data
and identical steps, does not. We contribute the diagnostic suite (Gini / dead-feature count / top-k
concentration / PR) alongside the finding, plus synthetic-data controls at 5k/50k/200k steps showing
this is not an undertraining artifact. *This is the most novel single result and is currently
under-reported in the SAE literature.*

**C4 — Branch-addressed steering has a better shift/coherence frontier than flat-SAE steering in
actual generation, in the direction where steering works at all.** Measured on 24 sentiment-neutral
prompts, judged by Qwen3-14B with no relationship to the dictionary or the probe. At matched fluency
retention (72%), Kron delivers ~1.35 judge points of sentiment shift against flat's interpolated
~1.19 — a real but modest edge, and a frontier rather than a dominance (flat is better at small
shifts, Kron at large ones). Scope: this holds for *negative* steering only; see C6.

**C6 — Probe-scored steering evaluations are invalid above the coherence threshold, and the field
should stop using them unguarded.** Our own layer-12 logistic probe scored a set of generations at
P(positive)=0.161 — a confident "strongly negative" — when the text was token salad with self-
perplexity 122,068 and the external judge marked 0/72 fluent. A linear probe on a mean-pooled
activation always returns a sentiment; it cannot abstain. Any steering result that reports concept
shift without a fluency gate is therefore unfalsifiable in the large-intervention regime, which is
exactly the regime steering papers report. We contribute the failure demonstration and the fix
(fluency-gated reporting, external judge, self-perplexity control).

**C7 — Steering is strongly direction-asymmetric, and positive-direction pushes fail for both
architectures.** Flat drops to 0% fluent at both tested positive α; Kron degrades to 29% then 1.4%
while still emitting on-target sentiment lexis ("excellent", "very good") with broken syntax. Two
candidate causes we can separate: baseline headroom (the unsteered model already reads positive) and
the asymmetry of post-push `clamp_min(0)`, which zeroes near-zero features under a negative push but
inflates every positively-loaded feature at once under a positive push.

**C5 — The above holds across Pythia scale (410M → 1.4B) at fixed data volume.**

---

## 4. Methodological improvements to adopt before the full run

Three of these come directly from your own critique; the fourth I am adding.

### 4.1 Replace 1-D direction overlap with a subspace metric

Current metric: `w_sent = Σ_c (μ_c − μ)²` per label, normalized, then `(ŵ_sent · ŵ_top)²`. For binary
sentiment this is defensible. For **6-way topic it is not** — it crushes a genuinely 5-dimensional
between-class structure into one non-negative energy vector, which is why topic overlap numbers are
hard to interpret and why the flat SAE's 0.9995 is suspiciously extreme.

**Proposed:** compute the between-class scatter subspace per concept — `S_b = Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ`
— take its top-(C−1) eigenvectors (1-D for sentiment, 5-D for topic), and report the **normalized
projection overlap**

```
overlap(U_s, U_t) = ‖U_sᵀ U_t‖_F² / min(d_s, d_t)      ∈ [0, 1]
```

equivalently the mean squared cosine of the principal angles. This is invariant to within-subspace
rotation, has a defined [0,1] range regardless of dimensionality mismatch, and reduces to the current
metric when both concepts are binary. Report principal angles individually too — a single scalar hides
whether one topic axis is entangled and four are clean.

### 4.2 Measure geometry in *decoder-output* space, not only feature space

You are right that element-wise comparison across architectures is only meaningful after a common
linear map. Feature-space PR and overlap are not comparable between a 16384-dim flat dictionary and a
(h×m)+(h×n) Kron factorization — the ambient dimensionality differs, and PR's range is [1, d].

**Protocol:** report every geometry metric **twice**: (a) in feature space, with PR normalized as
`PR/d` so it is comparable across widths; (b) in residual-stream space after `decode(·)`, where all
architectures — and the no-SAE baseline — live in the same 1024-dim (or 2048-dim) space. Claims about
"the SAE changed the geometry" should rest on (b); claims about "the dictionary collapsed" rest on (a).

### 4.3 Add the raw-residual (no-SAE) reference to *every* table

You already named this. It is also the most dangerous baseline in the paper: Xie (2025,
arXiv:2510.01246) reports activation-difference steering matching or beating SAE steering, and if a
plain mean-difference push in the residual stream steers sentiment as well as our branch push, the SAE
earns nothing on raw steering strength. The defensible position is **selectivity**: raw-residual
steering has no mechanism for moving one concept while holding another fixed. So —

### 4.4 Make the headline metric selectivity, not shift

Report the control frontier, not a single number:

- **Selectivity** = Δ(target concept) / Δ(off-target concept), swept over α.
- **Coherence cost** = Δ(target concept) / Δ(self-perplexity), swept over α.
- **Fluency retention** = % of generations the external judge marks fluent.

A method that shifts sentiment 0.4 while dragging topic 0.3 and doubling perplexity is worse than one
that shifts 0.25 cleanly, and a scalar Δsentiment cannot say so.

---

## 5. Experiment grid

### 5.1 Architecture × supervision (the core 2×2, plus references)

The pilots compare Kron+DPO against flat-unsupervised, which confounds architecture with objective.
Fix it:

| Cell | Architecture | Supervision | Purpose |
|------|--------------|-------------|---------|
| A | Kron (h,m,n) | DPO-cross | The proposed method |
| B | Kron (h,m,n) | none (plain TopK) | Isolates architecture from supervision |
| C | Flat TopK, matched dict + matched k | DPO-cross | Isolates supervision from architecture |
| D | Flat TopK, matched dict + matched k | none | Standard SAE baseline |
| R1 | — (raw residual) | — | No-SAE reference; ActAdd-style mean-difference steering |
| R2 | — (raw residual) | — | LEACE / whitened steering reference |

C is the cell that decides whether the paper is about *architecture* or about *the objective*. It does
not exist yet and is mandatory.

### 5.2 Capacity ablation (your "sub-SAE size")

Within cell A, sweep the (m, n) allocation at fixed total capacity: P<Q, P=Q, P>Q — the §19.6 grid,
extended to ≥3 seeds. Report all RQ2 geometry metrics and the RQ3 control frontier per allocation.

### 5.3 Concept ablation: steer topic, not sentiment

Run the entire RQ3 protocol with topic as the target concept and sentiment as the off-target. For a
6-way concept, "steering" means pushing toward a designated class centroid along the class-specific
signed direction; the judge prompt becomes a topic-classification prompt. This tests whether the
control frontier is a property of the architecture or of sentiment being an unusually linear concept.

### 5.4 Label swap (**new, highest value**)

Train cell A with the branch assignments reversed: **P←topic, Q←sentiment**. The width-swap experiment
(§19.6/19.7) showed geometry follows role rather than width, but it never varied *which concept* each
role carries — so "the sentiment branch is entangled" and "the P branch is entangled" are still
confounded. This single run separates them, and it is cheap. If the asymmetry follows the concept, C2
is a statement about sentiment; if it follows the branch, C2 is a statement about the loss geometry.
**Run this before committing to the scale study.**

### 5.5 Scale

Pythia-410M → Pythia-1.4B, identical dataset and identical token volume (mechanics in
`PAPER_SCALE_EXPERIMENT_DESIGN.md` §3.1). Full grid at 410M; at 1.4B, the same grid, de-scoped in the
order given in that document's §7 if the budget binds.

---

## 5.6 Revisions forced by the neutral-prompt pilot (2026-09-02)

The pilot in `runs/neutral_generation_judged.json` invalidated four assumptions the earlier design
rested on. These are not refinements; without them the full run produces uninterpretable tables.

### 5.6.1 Every concept metric must be fluency-gated, and fluency is a primary result

Report concept shift **twice**: over all generations, and over the judge-fluent subset only, with
`n_fluent` alongside. Where they diverge, the all-generations number is an artifact. A condition with
0% fluency has no concept value at all and must appear as `—`, never as a number — the probe's 0.161
on token salad is exactly the error this prevents. Fluency retention becomes a headline axis, not a
sanity check, because the architectures separate on it far more than on shift.

### 5.6.2 α must be calibrated per architecture, not shared

A shared α grid compares different intervention magnitudes: at α=+4 flat is already fully degenerate
(ppl 122k) while Kron is at 29% fluency, so "flat is worse at α=+4" partly restates that α=4 means
something different in each dictionary. Fix: **calibrate α per architecture to a fixed fluency
budget** (e.g. the α achieving 75%, 50%, 25% fluency retention, found by bisection on a held-out
prompt set), then compare concept shift at matched budget. The current `alpha × arch` grid should be
replaced by `fluency_budget × arch`. This is the single most important protocol change.

### 5.6.3 Δ is measured against each architecture's own α=0, and α=0 is a reported result

Reconstruction alone moves the metrics (flat α=0: judge 3.53, fluency 97.2%; Kron α=0: 3.33, 91.7%;
no-hook: 3.29, 100%). Flat reconstructs better, so Kron enters every sweep already behind on
coherence. All Δs take the architecture's own α=0 as origin, and the α=0 row is reported as the
reconstruction-fidelity result it is — it is also where the flat SAE's extra capacity legitimately
wins, which the paper should state plainly rather than bury.

### 5.6.4 Both push directions, reported separately — never averaged

Negative and positive steering behave qualitatively differently (negative: works, graceful; positive:
fails for both, catastrophically for flat). Averaging over sign would hide the entire effect. Add the
two mechanistic controls that separate the causes:
- **Headroom control:** run the same sweep from prompts whose unsteered continuations read *negative*,
  so the positive push has room to move. If positive steering works there, the asymmetry is headroom.
- **Clamp control:** ablate the post-push `clamp_min(0)` (allow signed features, or push in the
  pre-ReLU space). If positive steering works without the clamp, the asymmetry is the nonlinearity.

These two runs are cheap and they decide whether C7 is a fact about sentiment in Pythia or a fact
about our intervention operator — a reviewer will ask, and right now we cannot answer.

### 5.6.5 Judge validity must itself be reported

Report judge–probe correlation per condition, judge self-consistency (re-judge a 10% subsample at a
different temperature or with permuted scale wording), and the fluency-flag agreement against
self-perplexity. The judge is now load-bearing for the paper's headline claims, so its reliability is
a result, not an implementation detail.

---

## 6. Evaluation protocol

| Family | Metric | Space | Notes |
|--------|--------|-------|-------|
| Reconstruction | FVU, explained variance | residual | Standard SAE health |
| Sparsity/health | dead features, Gini of firing freq, top-1 / top-1% / top-10% firing share | feature | The C3 suite |
| Geometry | participation ratio (raw and PR/d) | feature + residual | §4.2 |
| Geometry | subspace overlap + principal angles | feature + residual | §4.1 |
| Concept access | linear probe accuracy per concept per branch | feature | Collection metric |
| Concept access | off-concept probe accuracy on the designated branch | feature | Suppression metric (the C1 negative) |
| Control | **fluency retention** vs α | generation | External judge boolean; *primary* axis (§5.6.1) |
| Control | Δ target concept @ matched fluency budget | generation | Judge, fluent-subset only (§5.6.2) |
| Control | Δ off-target concept @ matched fluency budget | generation | Selectivity denominator |
| Control | self-perplexity vs α | generation | Coherence, clean unpatched model; also the degeneracy detector |
| Control | α=0 reconstruction fidelity | generation | Per-architecture origin for all Δ (§5.6.3) |
| Validity | judge–probe correlation, judge self-consistency | — | §5.6.5 |

Judging: Qwen3-14B, 1–5 Likert + fluency boolean, temperature 0, strict JSON. Report judge–probe
agreement (correlation) as a validity check — where they diverge is itself interesting, and divergence
in favor of the probe is exactly the self-grading artifact the external judge exists to catch.

Seeds: ≥3 everywhere, with dispersion reported. Prompts: ≥24 neutral openers spread across the review
domains the topic labels cover (implemented in `src/eval_neutral_generation.py`).

---

## 7. Paper structure

1. Introduction — steering needs an address; SAEs are usually searched, not addressed.
2. Background — TopK SAEs; Kronecker factorization; concept erasure (LEACE) vs. activation steering.
3. Method — architecture, DPO-cross supervision, the branch-addressing interface.
4. RQ1: routing works, suppression does not (C1; the 16-mechanism sweep as a table, not a narrative).
5. RQ2: dictionary geometry and the flat-SAE collapse (C3, C2).
6. RQ3: the control frontier in generation (C4).
7. RQ4: concept ablation and label swap (C2 airtight).
8. RQ5: scale.
9. Related work — split cleanly into erasure (§11 survey) and steering (§19.13 survey).
10. Limitations — two concepts, one dataset, one layer, English reviews.

---

## 7.1 Honest assessment of where the thesis now stands

After the neutral-prompt pilot, the paper has **two strong claims and one modest one**, and the
framing should reflect that ordering rather than the original one:

- **Strong, novel, low-risk:** C3 (flat-SAE dictionary collapse on real but not synthetic
  activations) and C6 (probe-scored steering evaluations are invalid unguarded). Neither depends on
  Kron beating flat. C6 in particular is a methods contribution that stands even if every
  architectural comparison came out neutral.
- **Strong and already established:** C1 (collection without dispersion) and C2 (geometry follows
  role, pending the label swap).
- **Modest:** C4. Kron's generation-time steering edge is ~13% more shift at matched fluency, in one
  direction only. That is a result, not a headline. If the full grid does not strengthen it, the
  paper should lead with C3+C6 and present C4 as a supporting finding — which argues for the title's
  emphasis staying on *scoping* structured SAEs rather than on advocating them.

The risk this creates: a paper whose most novel results are a *negative* (collapse) and a
*methodological correction* (probe invalidity) is a good paper, but it is a different paper from
"KronSAE steers better." Worth deciding deliberately rather than by drift.

## 8. Open questions for you

1. **Topic direction.** You flagged that the topic direction construction needs to be better. §4.1's
   between-class subspace is my proposal. The alternative is to drop 6-way topic for a *second binary*
   concept (e.g. formality, or product-vs-service), which would make every metric symmetric and every
   comparison cleaner at the cost of one axis of generality. Which do you prefer?
2. **Venue and length.** Workshop-length (8pp) means RQ1–RQ3 only, with RQ4/RQ5 as appendix. Full
   conference means all five. That decision changes what we run, not just what we write.
3. **Judge budget.** Qwen3-14B at temperature 0 on the full grid (6 cells × 3 allocations × 5 α ×
   24 prompts × 3 samples × 2 concepts × 3 seeds) is ~78k judgments. Cheap per item but not free —
   confirm we judge the full grid, or only the headline cells with the probe covering the rest.
4. **The `unsupervised Kron` cell (B).** It has never been trained. Confirm it is in scope; without it
   we cannot separate "Kronecker factorization prevents collapse" from "supervision prevents collapse",
   and C3 is the paper's most novel claim.
5. **Framing decision (new, after the pilot).** Lead with C3+C6 (collapse + probe invalidity, both
   architecture-agnostic and both strong), or lead with C4 (Kron steers better, currently modest)?
   This determines the title, the abstract, and which cells of §5.1 get the seeds. My recommendation
   is C3+C6 first — they do not depend on the comparison going our way.
6. **Positive-direction diagnosis (new).** The headroom and clamp controls in §5.6.4 are cheap and
   they decide whether C7 is a claim about Pythia's sentiment geometry or about our own intervention
   operator. Confirm I should run them next; they are the highest information-per-GPU-hour item on
   the list right now.
