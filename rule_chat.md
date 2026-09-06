# Research Blueprint Workflow — Chat Input Variant

A reusable process for turning raw AI chat logs into a set of complete, experiment-ready paper blueprints — before any code is written. Written for both a human researcher and an AI agent executing the workflow.

The only difference from the standard workflow is the input: the researcher provides a copy-pasted AI conversation instead of a structured brief. A parsing phase extracts the necessary structure from the chat before ideation begins. Everything after that is identical.

---

## What the Researcher Provides

A copy-pasted conversation between the researcher and an AI assistant (ChatGPT, Claude, Gemini, or similar). The conversation may be:
- A brainstorming session about a research idea
- A Q&A about a paper the researcher just read
- A back-and-forth debugging of a hypothesis
- A mix of technical discussion and off-topic remarks

No structure is guaranteed. The researcher does not need to clean or format the chat before handing it over.

---

## What This Workflow Produces

One `flow_[id].md` file per research storyline. Each file is a self-contained complete blueprint covering: why the research question matters, what the hypothesis predicts, how to run the experiments, and which 40–50 papers support and contextualize the work. After the files are done, the researcher goes directly to implementation.

---

## The Process (Phase 0 + Two Phases + Iteration Loop)

```
Phase 0 (Chat Parsing)
        ↓
   Structured inputs extracted
        ↓
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

## Phase 0 — Chat Parsing (Extract Structured Inputs)

Before any ideation, read the full chat and reconstruct the structured inputs listed below. Show the extracted summary to the researcher before proceeding. If any required field is missing or ambiguous, ask. Do not invent values for missing fields.

### What to Extract

| Input | What it is | Where to look in the chat |
|-------|-----------|--------------------------|
| **Core idea** | The central technical contribution or observation | Statements of intent, described methods, noticed limitations |
| **Base papers** | Key papers that define the method and domain | Arxiv IDs, paper titles, method names (e.g. "LAPE", "LoRA") |
| **System constraints** | Architecture, scale, compute, or resource limits | Model names, hardware mentions, parameter counts |
| **Goals** | What the researcher wants to demonstrate or discover | "I want to show", "the point is", "ultimately" |
| **Domain of interest** | The application area or analysis target | Named tasks, phenomena, model families |
| **Caveats / preferences** | What to explore, what to avoid, what would be ideal | "I don't want to", "ideally", "it doesn't have to be" |
| **Number of flows** | How many distinct storylines to produce | Explicit mention, otherwise default to 5 |

### Extraction Output Format

```
Core idea: [one sentence]
Base papers: [list with links]
System constraints: [list]
Goals (primary): [one sentence]
Goals (secondary): [list]
Domain: [named]
Caveats: [list]
Number of flows: [N]
Unresolved items: [anything mentioned but unclear — flag for researcher]
```

### Handling Ambiguity in Chat Logs

- **Researcher overrides AI:** If the researcher corrects or rejects something the AI said, treat the AI's suggestion as discarded.
- **Later statements override earlier ones:** If the researcher changes direction mid-chat, use the later version.
- **Questions are not assertions:** "Could we do X?" is a candidate to evaluate, not a fixed goal.
- **Off-topic passages:** Ignore exchanges about unrelated topics, logistics, or casual conversation.

---

## Phase 1 — Bottom-up Survey (Literature Gap Mapping)

**Goal:** Discover what is already answered, what is partially answered, and what is genuinely open — before constructing any research question.

**Starting point:** The atomic observations extracted from the core idea in Phase 0. Do not start from a research question. Start from the smallest meaningful claims within the idea.

**Process:**

1. **Extract atomic observations from the core idea.** Break it into the smallest meaningful claims. Each becomes a survey seed.

2. **For each atomic observation, find papers that already study it.** Search broadly. Read abstracts and conclusions. Extract:
   - What did they find?
   - What did they leave open?
   - What did they assume but not test?

3. **Build a gap map.** Organize findings into three bins:
   - **Closed:** The question is answered. A paper already reports the result. Do not build a flow here.
   - **Partial:** The question is studied but only in a limited setting (different model, domain, or scale). A flow here replicates and extends with a novel axis.
   - **Open:** No paper directly addresses this. A flow here is genuinely novel.

4. **Prioritize open and partial gaps** as candidates for Phase 2.

---

## Phase 2 — Top-down Flow Construction

**Goal:** For each open or partial gap identified in Phase 1, construct a complete research flow with a falsifiable hypothesis and a concrete experimental plan.

**Process:**

1. **Frame the central research question** from the gap. Name the phenomenon, the model type, and the measurement axis specifically.

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

## Iteration Loop — RQ Novelty Check

After Phase 2 produces a candidate flow, run a novelty check before finalizing it.

**Check:** Search specifically for papers that directly answer the flow's primary research question. Use targeted queries combining the method name, the phenomenon studied, and the measurement approach.

**Decision:**
- **Already answered (Closed):** A paper reports essentially the same finding under the same conditions. Mark this flow as closed. Do not write the flow file. Return to Phase 1 with a narrower or shifted clue derived from the gap map.
- **Partial match:** A paper answers a related question but in a different setting. Keep the flow but reframe the RQ around the novel axis — explicitly position against the existing work.
- **Open:** No direct answer exists. Finalize and write the flow file.

Iterate Phase 1 → Phase 2 → novelty check until the target number of novel flows is reached.

---

## Paper Search (Per Flow)

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

## Agent Strategy (Brief)

Complete Phase 0 and Phase 1 fully before spawning any Phase 2 agents. Then run one flow to completion at a time, or in small batches of 2–3. Within a single flow, all 7 category searches can run in parallel. Do not run all flows simultaneously if total flows ≥ 4 — session token limits will cut agents before they finish writing.

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
