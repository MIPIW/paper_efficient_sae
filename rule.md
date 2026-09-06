# Research Blueprint Workflow

A reusable process for turning a raw research idea into a set of complete, experiment-ready paper blueprints — before any code is written. Written for both a human researcher and an AI agent executing the workflow.

---

## What This Workflow Produces

One `flow_[id].md` file per research storyline. Each file is a self-contained complete blueprint covering: why the research question matters, what the hypothesis predicts, how to run the experiments, and which 40–50 papers support and contextualize the work. After the files are done, the researcher goes directly to implementation.

---

## What the Researcher Must Provide

| Input | What it is | Why it matters |
|-------|-----------|----------------|
| **Core idea** | The central technical contribution or observation | Defines the pivot point for all flows |
| **Base papers** | 1–3 key papers (with links) that define the method and domain | Must be read before ideation |
| **System constraints** | Architecture, scale, compute, or resource limits | Shapes feasibility of experiments |
| **Goals** | What the researcher wants to demonstrate or discover | Prevents flows from drifting off-topic |
| **Domain of interest** | The application area or analysis target | Grounds the flows in a specific context |
| **Caveats / preferences** | What to explore, what to avoid, what would be ideal | Prevents wasted flows |
| **Number of flows** | How many distinct storylines to produce (≥ 5 recommended) | Sets the scope of the search |

If any input is missing or ambiguous, ask before proceeding.

---

## The Process (Two Phases + Iteration Loop)

```
Phase 1 (Bottom-up Survey)
        ↓
   Gap Map built
        ↓
Phase 2 (Top-down Flow Construction)
        ↓
   RQ novelty check
        ↓
  Already answered? ──Yes──→ Mark closed → back to Phase 1 with narrower or another diverese clue
        ↓ No
   Write flow_[id].md
        ↓
  Repeat until target number of novel flows reached
```

---

### Phase 1 — Bottom-up Survey (Literature Gap Mapping)

**Goal:** Discover what is already answered, what is partially answered, and what is genuinely open — before constructing any research question.

**Starting point:** The researcher provides a small clue (an observation, a method property, an analogy, a limitation noticed in a paper). Do not start from a research question. Start from the clue.

**Process:**

1. **Extract atomic observations from the clue.** Break the core idea into the smallest meaningful claims. Each claim becomes a survey seed.
   - Example clue: "Kronecker product in SAE encoder is structurally similar to query-key attention."
   - Atomic observations: (a) Kronecker factorization reduces encoder cost, (b) the score function is a bilinear inner product, (c) bilinear products appear in attention, (d) attention heads specialize — do Kronecker groups also specialize?

2. **For each atomic observation, find papers that already study it.** Search broadly. Read abstracts and conclusions, not just titles. Extract:
   - What did they find?
   - What did they leave open?
   - What did they assume but not test?

3. **Build a gap map.** Organize findings into three bins:
   - **Closed:** The question is answered. A paper already reports the result. Do not build a flow here.
   - **Partial:** The question is studied but only in a limited setting (different model, different domain, different scale). A flow here replicates and extends with a novel axis.
   - **Open:** No paper directly addresses this. A flow here is genuinely novel.

4. **Prioritize open and partial gaps** as candidates for Phase 2.

---

### Phase 2 — Top-down Flow Construction

**Goal:** For each open or partial gap identified in Phase 1, construct a complete research flow with a falsifiable hypothesis and a concrete experimental plan.

**Process:**

1. **Frame the central research question** from the gap. Be specific: name the phenomenon, the model type, the measurement axis.

2. **Design the story arc:** problem → why the gap matters → proposed approach → what a positive result looks like.

3. **Identify the contribution angle** (use as a starting taxonomy, not an exhaustive list):
   - Efficiency — making the method tractable at scale
   - Structural interpretation — what the method's inductive bias reveals about representations
   - Domain application — applying the method to a specific target
   - Compositional analysis — whether the structure encodes hierarchical or compositional relationships
   - Intervention and control — whether structured features enable better steering, editing, or probing
   - Theoretical grounding — what the method implies about representation geometry

4. **Write the blueprint skeleton** before searching for supporting papers:
   - Overview (3 paragraphs)
   - Research Questions (6–8)
   - Hypothesis (falsifiable, with predicted direction or magnitude)
   - Experimental Plan (specific models, baselines, metrics, ablations)

5. **Search for 40–50 supporting papers** across 7 categories (see Paper Search section below).

---

### Iteration Loop — RQ Novelty Check

After Phase 2 produces a candidate flow, run a novelty check before finalizing it.

**Check:** Search specifically for papers that directly answer the flow's primary research question. Use targeted queries combining the method name, the phenomenon studied, and the measurement approach.

**Decision:**
- **Already answered (Closed):** A paper reports essentially the same finding under the same conditions. Mark this flow as closed. Do not write the flow file. Return to Phase 1 with a narrower or shifted clue derived from the gap map.
- **Partial match:** A paper answers a related question but in a different setting (different model family, different scale, different language). Keep the flow but reframe the RQ around the novel axis — explicitly position against the existing work.
- **Open:** No direct answer exists. Finalize and write the flow file.

Iterate Phase 1 → Phase 2 → novelty check until the target number of novel flows is reached.

---

### Paper Search (Per Flow)

For each finalized flow, identify **7 paper categories** that together cover:

1. Foundational background — why the problem exists
2. Direct baseline or predecessor methods
3. The proposed method itself
4. Related or competing approaches
5. Evaluation tools and benchmarks
6. Theoretical grounding
7. Downstream application or validation use cases

For each category, run multiple web searches with different keyword phrasings. Find **5–7 papers per category**, targeting **40–50 papers per flow total**.

Papers found during Phase 1 that ended up in the "closed" bin are still useful — they belong in category 4 (related/competing approaches) and their open questions can seed new Phase 1 clues.

---

### Agent Strategy (Brief)

Running all flows simultaneously exhausts session token limits before agents finish writing. Recommended pattern: complete Phase 1 fully before spawning any Phase 2 agents. Then run one flow to completion at a time, or in small batches of 2–3. Within a single flow, all 7 category searches can run in parallel.

---

## Output Format

One file per flow: `flow_1.md`, `flow_2.md`, ..., `flow_N.md`

```markdown
# Flow [N]: [Descriptive Title]

## Overview
[Paragraph 1: What problem does this flow address, and why does it matter?]
[Paragraph 2: How does the proposed method tackle this problem?]
[Paragraph 3: What does the experimental validation look like, and what constitutes success?]

## Research Questions
1. **[Short label].** [Full question — specific and measurable.]
2. ...
(6–8 total)

## Hypothesis
[2–3 sentences. State the predicted result, the mechanism, and the falsification condition.]

## Experimental Plan
### Models
### Baselines
### Metrics
### Ablations

---

# [Category Name]

## [Paper Title]
**Authors & Year:** [Last name et al., YYYY]
**ArXiv/Link:** https://arxiv.org/abs/XXXX.XXXXX
**Summary:** [1–2 complete sentences. Main finding or contribution.]
**Implication:** [1–2 complete sentences. What this paper contributes specifically to this flow.]

[Repeat for each paper]
```

---

## Format Constraints

| Field | Constraint |
|-------|-----------|
| Summary | 1–2 complete sentences, no bullet fragments |
| Implication | 1–2 complete sentences, specific to this flow's RQs — not a generic statement |
| Overview | 3 paragraphs, each 3–6 sentences |
| Research Questions | 6–8, each falsifiable and measurable |
| Hypothesis | Must include a predicted direction or magnitude; must state what would falsify it |
| Experimental Plan | Must name specific models, not placeholders like "a large LLM" |
| Papers per flow | Minimum 40, target 45–50 |
| Categories per flow | 7 |
| Papers per category | 5–7 |

---

## Quality Checks

Before declaring a flow complete:
- [ ] Novelty check passed — no paper directly answers the primary RQ
- [ ] Every paper has a valid link
- [ ] No two papers in a file are duplicates
- [ ] Every Implication names how the paper connects to a specific RQ or experimental axis in this flow
- [ ] The experimental plan includes at least one ablation per major design choice
- [ ] The hypothesis is falsifiable by the listed experiments

---

## What This Workflow Does Not Do

- Does not write code or run experiments
- Does not guarantee arxiv IDs are correct — verify before citing in a submission
- Does not rank flows by feasibility — the researcher decides which flow to pursue
- Does not produce a paper draft — it produces a blueprint that guides implementation
