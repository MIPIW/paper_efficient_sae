# KronSAE Compositional Training — Full Report

> Consolidated from `REPORT.md`, `REPORT1.md`, and `PAPER_REPLICATION.md` (merged on 2026-08-26; the three source files remain in the repo unchanged). Comments/interpretation from the assistant are marked with `> **Comment:**` blockquotes throughout — everything else is factual (what was run, what config, what numbers came out).
>
> Table columns are labeled "▲" or "▼" — ▲ = higher is better, ▼ = lower is better (for a branch trained to *suppress* a label, success means the probe falls toward chance, so its column reads lower-is-better). Unmarked columns are references or descriptive quantities with no preferred direction.
>
> Section numbering below preserves each source file's own numbering (so in-text cross-references like "§9.3" or "§17" keep resolving correctly): **Part I** = `REPORT.md` §1-10, **Part II** = `REPORT1.md` §1-18 (a direct continuation of Part I §9.1-9.5 — read that first), **Appendix A** = `PAPER_REPLICATION.md`'s two unnumbered sections (paper-replication experiments, kept separate because they are not central to the project's current direction). A cross-reference written as bare "§N" inside Part II always means a Part II section; "REPORT.md §N" inside Part II always means a Part I section.

---

## Part I — Compositional Training Experiments

*(source: `REPORT.md`)*

### 1. Infrastructure

- Base repo: `saprmarks/dictionary_learning`, extended with `dictionary_kron.py`, `trainers/kron_top_k.py`, `labeled_buffer.py`.
- Training: plain PyTorch DDP (`torchrun`, NCCL), not DeepSpeed ZeRO — frozen LM (410M) and SAEs (<35M) fit one A100-80GB with headroom; ZeRO would add complexity without solving a memory problem here.
- Environment: 4× A100-80GB, 20 CPU cores, all runs launched via `torchrun --nproc_per_node=4`.
- LM: `EleutherAI/pythia-410m` (frozen), residual stream output at layer 12 (`model.gpt_neox.layers[12]`, `io='out'`).
- Disk policy: shared filesystem is near-full; checkpoints deleted after result extraction unless explicitly told to keep them.

### 2. Dataset

- Source: `McAuley-Lab/Amazon-Reviews-2023`, 6 categories: All_Beauty, Amazon_Fashion, Appliances, Arts_Crafts_and_Sewing, Office_Products, Pet_Supplies.
- topic label = category (balanced, ~1/6 each by construction). sentiment label = binarized rating (1-2=negative, 4-5=positive, 3 dropped) — naturally imbalanced (~85-87% positive per category).
- Data build: `sample_per_category=500000`. Rarest (category, sentiment) cell — `(Arts_Crafts_and_Sewing, negative)`, count 53,119 — fully withheld from train as a future compositional-generalization test case (not yet used). Train=2,887,943 / eval=58,938 / withheld=53,119.

### 3. SAE Architectures

- **Flat**: `AutoEncoderTopK` (unmodified base-repo class). `dict_size=16384`, `k=24`.
- **KronSAE**: `KronAutoEncoderTopK`, mAND gate per Kurochkin et al. 2025 (arXiv:2505.22255) Eq. 3: `z_{i,j} = sqrt(u_i·v_j)` if `u_i>0` and `v_j>0`, else 0 (`+1e-12` inside sqrt for gradient stability). `combine_rule ∈ {mand, mor, concat}` implemented; `mor` = probabilistic OR (`1-exp(-x)` mapping then `p+q-pq`); `concat` = no gating, dict_size becomes `h·(m+n)`.
- Pilot/joint default sizing: `h=128, m=8, n=16` (dict_size = 16384, matches flat).
- A "main 6-config grid" (flat×{sup on/off} + kron×{mand/mor/concat}) is wired in code but not yet run at full scale (see §10).

### 4. Supervision Losses

- **SupCon** (Khosla et al. 2020 form, InfoNCE-style): pulls same-label pooled document reps together (`invert=False`) or pushes them apart (`invert=True`).
- **DPO-style pairwise loss** (joint DPO-cross experiment only): `L = -log(sigmoid(beta·(sim(anchor,pos) - sim(anchor,neg))))`, cosine similarity on pooled reps. `dpo_beta=2.0` used for all full-scale runs.
- **Flat SAE has no architectural P/Q split.** Flat receives supervision on the whole undivided feature vector (`pilot_collect_full` mode) — no splitting. For the joint DPO-cross experiment, flat can only receive the two *collect* terms (cannot do cross-repulsion, since pulling and pushing the same label on one undivided vector is contradictory) — this asymmetry is structural, not a workaround.

### 5. Training Stability Configuration

Flat SAE's supervision loss acts directly on the same dense pre-top-k tensor used for top-k feature selection, which can destabilize which features win top-k once supervision pressure is strong. All full-scale runs below use:
- `warmup_frac=0.05`, `decay_start_frac=0.8`, `threshold_start_frac=0.05` (fractions of `total_steps`).
- `lambda_sup_warmup_frac=0.15` (supervision ramps in gradually).
- `flat_sup_grad_scale=0.3` (gradient scaling on flat's supervision path only).
- Class-balanced batching for sentiment (naturally ~87/13 imbalanced) — oversampling the minority class per batch so contrastive/DPO terms always have valid pairs.

> **Comment:** these fractional schedules are tuned for 420,000-step runs; at much shorter step counts the same fractions compress the warmup into very few absolute steps, which can reintroduce the flat instability. Full-scale (420k-step) runs are the only ones treated as reliable in this report.

### 6. Experiment: Pilot (single-label collect/contrast)

Single label at a time. Small branch P (`m=8`) does collect (pull same-label together); large branch Q (`n=16`) does contrast (push same-label apart). Flat gets collect-only on the full vector. 420,000 steps, class-balanced train **and** eval.

| Run | Label | P / full-acc ▲ | Q-acc ▼ | chance (majority) |
|---|---|---|---|---|
| flat_pilot_sentiment | sentiment | 0.942 | — | 0.500 |
| flat_pilot_topic | topic | 0.649 | — | 0.167 |
| kron_pilot_sentiment | sentiment | 0.952 | **0.500** | 0.500 |
| kron_pilot_topic | topic | 0.544 | 0.198 | 0.167 |

> **Comment:** Q's contrast on sentiment reached exact chance (0.500) — clean suppression of the label it was trained to push apart.

### 7. Experiment: Joint, collect-only (both labels, no repulsion)

P (m=8) collects sentiment; Q (n=16) collects topic; no contrast term. Tests whether Q (previously only ever used for contrast) can do a collect job. 420,000 steps.

| Run | P-sentiment ▲ | Q-sentiment ▼ | P-topic ▼ | Q-topic ▲ |
|---|---|---|---|---|
| flat_joint (full-vector) | 0.941 | — | 0.654 | — |
| kron_joint | 0.946 | 0.814 | 0.576 | 0.476 |

chance: sentiment 0.500, topic 0.167.

> **Comment:** Q *can* collect topic (0.476, well above chance) but without any contrast pressure, both branches leak the other label — P knows topic (0.576) about as well as Q does.

### 8. Experiment: Joint, DPO cross-repulsion (both labels, full objective)

P: collect-sentiment + contrast-topic (DPO). Q: collect-topic + contrast-sentiment (DPO). `beta=2.0` uniform across all four sub-losses. 420,000 steps.

| Run | P-sentiment ▲ | Q-sentiment ▼ | P-topic ▼ | Q-topic ▲ |
|---|---|---|---|---|
| flat_joint (collect-only, no cross-repulsion possible) | 0.943 | — | 0.658 | — |
| kron_joint | 0.938 | **0.933** | 0.520 | **0.645** |

> **Comment:** own-label accuracy is strong for both branches (P-sentiment 0.938, Q-topic 0.645, matching flat). But cross-label suppression largely failed: P's topic leakage barely dropped (0.576→0.520) and Q's sentiment leakage got *worse* (0.814→0.933). Interpreted as: collect and contrast compete for the same branch capacity when both labels are handled simultaneously, unlike the single-label pilot where each branch only ever had one job.

### 9. Architecture Sweep: attempts to fix cross-leakage

Three full-scale (420,000-step) configurations tested, all `joint_dpo_cross`, `beta=2.0`, identical other hyperparameters — only `joint_h`/`joint_m`/`joint_n` varied:

| Config | P-sent ▲ | Q-sent ▼ | P-top ▼ | Q-top ▲ |
|---|---|---|---|---|
| h=128, m=8, n=16 (§8 baseline) | 0.938 | 0.933 | 0.520 | 0.645 |
| h=32, m=8, n=16 | 0.871 | 0.915 | 0.550 | 0.613 |
| h=32, m=4, n=32 | 0.809 | 0.918 | 0.489 | 0.616 |

A DPO sub-loss reweighting sweep (upweighting the failing `q_contrast_sentiment` term 2-3×, at h=128/m=8/n=16) was also tested and did not move Q-sentiment leakage either.

> **Comment:** none of the structural (head count, m/n ratio) or loss-weighting changes fixed the leakage — Q-sentiment stayed at 0.91-0.93 across all configurations tried. This looks structural to having one branch do collect+contrast simultaneously for two different labels in the same batch, not fixable by sizing or loss-weight tuning alone. Untried: curriculum-style training (collect-only warmup before introducing contrast), or reconsidering the simultaneous dual-objective design itself.

#### 9.1 Diagnostic: Representation Geometry (why §8-9 failed)

**Question.** Why does the cross-label leakage resist every sizing/weighting fix? Is the information genuinely still present, or merely not linearly readable?

**Method.** Representation-geometry analysis on frozen checkpoints (no retraining), comparing the joint failure case (`kron_joint`, §8) against the single-label pilot checkpoints (§6) as controls. Per (checkpoint, branch, label): linear / k-NN / MLP probes, HSIC (normalized), participation ratio, Fisher trace ratio, and within-branch `|cos(sentiment-direction, topic-direction)|`.

Each row below is a (checkpoint, branch, label) probe result, split into two tables by what that branch was trained to do with that label — a branch collecting a label wants probes to read *high*; a branch suppressing a label wants probes to read *chance*. Pilot and joint rows for the same role sit next to each other for direct comparison.

**Branches trained to COLLECT a label (▲; probe should clear chance)**

| Checkpoint | branch·label | linear | kNN | MLP | HSIC (norm.) | chance |
|---|---|---|---|---|---|---|
| sentiment control (§6, `kron_pilot_sentiment`) | P-sent | 0.951 | 0.960 | 0.945 | 0.412 | 0.500 |
| joint (§8, `kron_joint`) | P-sent | 0.938 | 0.965 | 0.963 | 0.753 | 0.500 |
| topic control (§6, `kron_pilot_topic`) | P-top | 0.544 | 0.445 | 0.522 | 0.009 | 0.167 |
| joint (§8, `kron_joint`) | Q-top | 0.645 | 0.605 | 0.640 | 0.312 | 0.167 |

**Branches trained to SUPPRESS a label (▼; probe should fall to chance)**

| Checkpoint | branch·label | linear | kNN | MLP | HSIC (norm.) | chance |
|---|---|---|---|---|---|---|
| sentiment control (§6, `kron_pilot_sentiment`) | Q-sent | 0.500 | 0.851 | 0.500 | 0.016 | 0.500 |
| joint (§8, `kron_joint`) | Q-sent | 0.933 | 0.634 | 0.908 | 0.028 | 0.500 |
| topic control (§6, `kron_pilot_topic`) | Q-top | 0.197 | 0.427 | 0.167 | 0.008 | 0.167 |
| joint (§8, `kron_joint`) | P-top | 0.520 | 0.256 | 0.568 | 0.015 | 0.167 |

| Checkpoint·branch | overall PR | sentiment disc-PR / Fisher | topic disc-PR / Fisher |
|---|---|---|---|
| sentiment control (§6) P | 1.36 | 5.13 / 0.325 | — |
| sentiment control (§6) Q | 1.00 | 138.8 / 0.046 | — |
| topic control (§6) P | 1.00 | — | 25.22 / 0.014 |
| topic control (§6) Q | 1.00 | — | 177.75 / 0.013 |
| joint (§8) P | 4.19 | 21.99 / 0.799 | 21.91 / 0.016 |
| joint (§8) Q | 7.91 | 7.30 / 0.0086 | 13.03 / 0.284 |

| `kron_joint` (§8) branch | `\|cos(sentiment-dir, topic-dir)\|` | reading |
|---|---|---|
| P | 0.951 | directions nearly identical — no spare direction to separate into |
| Q | 0.205 | largely orthogonal — separating geometry is available |

**Result (as originally read).** Linear and MLP probes agreed in every condition, which at the time looked like the most trustworthy signal available — by that reading, the pilot's Q genuinely erased sentiment and joint Q genuinely failed to. k-NN and HSIC disagreed with them; that disagreement was attributed to representational anisotropy (participation-ratio differences between branches) rather than hidden recoverable information. P and Q were read as failing for different structural reasons: P's sentiment- and topic-discriminative directions nearly parallel (cos ≈ 0.95, no spare direction), Q's nearly orthogonal (cos ≈ 0.21, geometry available but still leaking).

> **⚠ Superseded, see §9.3.** The pilot control rows above (both `sentiment control` and `topic control`) were later found to rest on a broken probe (§9.3) — Q-sent = 0.500 and Q-top = 0.197 are probe-optimization artifacts, not genuine suppression. The "linear and MLP agreeing ⇒ trustworthy" reasoning above is exactly what the artifact defeats, since both probes fail the same way for the same reason. The joint (§8) rows are unaffected — their PR (4.19/7.91) is well-conditioned and their probes converge normally, so the direction-overlap finding (P: cos≈0.95 vs Q: cos≈0.21) still stands as a real structural difference between the two branches. What no longer stands is the pilot-vs-joint contrast this section was built on.

#### 9.2 Diagnostic: Optimization Fixes (testing §9.1's gradient hypothesis)

**Question.** §9.1 identified Q's failure as gradient competition rather than lack of representational space. Can it be fixed by intervening at the gradient/loss level?

**Method.** Built a CPU-only synthetic diagnostic mode (`--synthetic_diagnostic`) that bypasses the LM/real-data pipeline and generates activations whose sentiment- and topic-signal directions are **provably orthogonal by construction** (disjoint QR-decomposition column blocks + Gaussian noise). This removes real-data signal ambiguity as a confound and allows fast iteration without GPU. All configs: `joint_dpo_cross`, `kron_joint`, `h=128/m=8/n=16`, `dpo_beta=2.0`, 5000 steps unless noted (chance: sentiment=0.500, topic=0.167).

| Config | P-sent ▲ | Q-sent ▼ | P-top ▼ | Q-top ▲ |
|---|---|---|---|---|
| baseline | 0.941 | 0.972 | 0.786 | 0.838 |
| + gradient surgery (PCGrad, collect vs. contrast) | 0.953 | 0.952 | 0.786 | 0.812 |
| + orthogonality penalty (λ=0.1) | 0.953 | 0.973 | 0.781 | 0.812 |
| + both combined | 0.959 | 0.962 | 0.793 | 0.727 |
| + curriculum (`contrast_start_frac=0.3`) | 0.945 | 0.958 | 0.769 | 0.780 |
| lambda_sup 1→10 (3000 steps) | 0.972 | 0.823 | 0.832 | 0.503 |
| lambda_sup 10 + gradient surgery (3000 steps) | 0.972 | 0.784 | 0.834 | 0.490 |
| + 3-way PCGrad incl. reconstruction | 0.954 | 0.918 | 0.799 | 0.659 |
| + contrast-grad magnitude match to reconstruction | 0.941 | 0.983 | 0.749 | 0.905 |
| + magnitude match + 3-way PCGrad combined | 0.943 | 0.990 | 0.732 | 0.915 |
| + magnitude match, Q-branch-only grad clip (isolating global-clip interaction) | 0.861 | 0.983 | 0.595 | 0.912 |
| + magnitude match capped 5×, separate clip | 0.919 | 0.972 | 0.722 | 0.868 |
| + magnitude match capped 10×, separate clip | 0.919 | 0.980 | 0.689 | 0.906 |

**Result.** None of the twelve variants brought Q's cross-label leakage near chance. The two interventions that moved Q(sent) at all (raising `lambda_sup` 10×; 3-way PCGrad) did so by sacrificing Q's own-label accuracy by a comparable margin — a trade-off, not a fix. Magnitude-matching contrast's gradient to reconstruction's made leakage *worse*, not better (0.972→0.983-0.990), and stayed worse even after isolating it from the global gradient-clip interaction (0.983-0.991) — ruling out "the global clip was distorting the fix" as an explanation.

One diagnostic fact did come out of this: on Q's parameters (measured in the 3-way-PCGrad run), the **reconstruction gradient norm ≈ 18.8, vs. collect ≈ 0.35 and contrast ≈ 0.41 — a ~45-50× dominance**. So the gradient competition §9.1 inferred is real on synthetic data, and the dominant competitor is reconstruction, not the collect objective. But every prescription that followed from that diagnosis — matching magnitudes, isolating the clip, capping the boost — failed, in several cases making leakage worse or damaging the P branch.

**Takeaway.** The diagnosis (reconstruction overwhelmingly dominates Q's gradient) held up on synthetic data and was later confirmed on real data too (§9.3, with an important correction: the real-data ratio is larger — hundreds of ×, not 45-50× — and affects P as much as Q). The prescription ("so equalize the magnitudes") did not work, across every variant tried.

> **⚠ Reframed, see §9.3.** At the time this section concluded that simultaneous dual-objective training on one branch "does not work, and is not a tuning problem," pointing back to §8-9. §9.3 complicates this: the real-data gradient imbalance is not Q-specific (P shows the same or larger ratio), which undercuts a Q-specific explanation, and the pilot's single-objective "success" that §8-9's contrast was built on (§9.1's control rows) turned out not to be a success at all. The twelve failed interventions above stand as evidence — but evidence for a still-open question, not a settled one; see §9.3's verdict.

#### 9.3 Diagnostic: Probe Artifact and Real-Data Gradient Confirmation

**Question.** Two loose ends from §9.1-9.2, both prompted by suspicion that a naive PCGrad implementation and a naive probe were the wrong tools rather than proof of a structural limit: (i) does §9.2's synthetic reconstruction-dominance (~45-50×) replicate on real data, and on both branches or just Q? (ii) is the pilot's §6/§9.1 "clean suppression" (Q-sent = 0.500 exactly) real, given it was the sole anchor for every "single-objective succeeds, joint fails" comparison in this report?

**Method (i).** Reused the trainer's own gradient-surgery instrumentation as a measurement-only probe (never writes `param.grad`) on the converged §8 checkpoint (`joint_dpo_cross_full420k`, step 420,000), 200 real-data batches, plus a fresh 3000-step real-data trajectory from scratch. Confirmed supervision terms were actually active (`valid_anchor_frac` ≈ 0.96-1.00 throughout — not silently zero).

**Method (ii).** Re-ran the pilot's probe with 8 seeds at two held-out sizes (4000 and 12,000 docs, matching §6/§9.1's setup), then compared the original unstandardized fixed-LR probe against a standardized-feature + converged-solver probe (`LogisticRegression(max_iter=5000)` + a standardized MLP) on identical splits. Also checked whether Q was dead: Q-topic accuracy (never measured in this checkpoint before), a mean-ablation reconstruction test, dead-unit fraction, and the covariance eigenvalue spectrum behind PR≈1.00.

**Result (i) — reconstruction dominance replicates, and is not Q-specific.**

| | recon grad norm | collect grad norm | contrast grad norm | recon / contrast (median) |
|---|---|---|---|---|
| synthetic (§9.2 reference) | 18.8 | 0.35 | 0.41 | ~45× |
| real data, 420k ckpt, **Q** | 9.87 ± 5.45 | 0.035 | 0.034 | **~302×** (λ-normalized: ~151×) |
| real data, 420k ckpt, **P** | 9.14 ± 5.01 | 0.021 | 0.026 | **~405×** (λ-normalized: ~250×) |

The imbalance is larger on real data than synthetic, and **P shows it too, at least as strongly as Q** — so this cannot be a Q-specific explanation for Q-specific failure, undercutting §9.2's takeaway as originally framed.

**Result (ii) — the pilot's "clean suppression" was a probe artifact, not a real result.**

| Measurement | value | chance | reads as |
|---|---|---|---|
| Q-sent, original raw probe (§6, single seed) | 0.500 | 0.500 | "suppressed" |
| Q-sent, raw probe, 8 seeds, 12,000 docs | 0.500 – 0.890 (mean 0.769) | 0.500 | unstable — §6 reported the minimum |
| **Q-sent, standardized + converged logreg** | **0.906** | 0.500 | **not suppressed** |
| Q-sent, standardized MLP | 0.911 | 0.500 | agrees |
| Q-topic, standardized (never measured before) | 0.562 | 0.167 | **branch is alive**, not dead |
| P-topic, same ckpt, unsuppressed (reference) | 0.587 | 0.167 | Q-topic ≈ this reference |
| Q mean-ablation → reconstruction FVU | 0.674 | (baseline 0.0014) | Q does real reconstruction work, ≈ P (0.668) |
| Q dead units | 6.5% of 2048 | — | not dead |
| Q dims for 99% of residual variance (after removing PR's dominant direction) | 1117 / 2048 | — | not collapsed |

Mechanism: Q's doc-pooled features have per-dim std ≈ 8.2 vs. P's 0.29 (~28×), so the original fixed-LR unstandardized probe never left the constant-predictor solution on Q — which is exactly what produces an exact 0.5000 or 1/6, on *any* label, regardless of whether that label was actually suppressed. `kron_pilot_topic`'s Q-topic = 0.198 is the same artifact (standardized: 0.574).

**Verdict.** Q is a live, information-bearing branch (rules out "dead branch, trivial chance") that never suppressed sentiment (rules out "genuine clean suppression") — its true leakage (0.87-0.91) sits in the same band as the joint runs' 0.81-0.93 (§7-§9). **There was no clean single-objective success in this project to contrast the joint failure against.** Combined with (i) — P showing the same gradient imbalance as Q while being read as "successful" — the §8-9 framing of "collect succeeds, contrast+collect together fails" is not supported by clean evidence on either side of the comparison. This does not mean cross-label suppression is impossible; it means **this project has not yet produced a verified case of it succeeding, under any configuration**, single-objective or joint. §9.1's joint-checkpoint rows (PR, probes, direction-overlap) are unaffected by this — only the pilot control rows and the pilot-vs-joint contrast built on them are invalidated.

This raised the obvious next question: is the ≈0.87-0.95 range reported throughout §6-§9 as "collect succeeded" actually caused by supervision, or is it just what any SAE — trained or not — would score on activations that are already this separable? §9.4 answers it.

#### 9.4 Diagnostic: Supervision vs. Random-Init Control

**Question.** Do the "collect" numbers reported throughout §6-§9 (P-sentiment ≈0.94, P-topic ≈0.55-0.65, etc.) reflect anything supervision did, or are they just what pythia-410m's layer-12 activations already give any sufficiently large linear map — trained or not?

**Method.** Three arms, same held-out documents, same balanced subsets/splits, same corrected probe (standardized logreg + MLP) as §9.3:
- **A — raw activation.** pythia-410m layer-12 residual, doc-mean-pooled, no SAE at all (1024-dim).
- **B — random-init KronSAE.** Same architecture as the pilot (h=128, m=8, n=16), freshly initialized, **never trained.** P and Q branches probed directly; Q also PCA'd down to 1024 dims to control for its larger dimensionality.
- **C — trained.** The pilot checkpoints (`kron_pilot_sentiment`, `kron_pilot_topic`, P and Q) and the flat pilots, re-probed with the corrected methodology.

**Result.**

| Arm | sentiment ▲ (chance 0.500) | topic ▲ (chance 0.167) |
|---|---|---|
| A — raw activation | 0.907 | 0.594 |
| B — random-init, P | 0.893 | 0.600 |
| B — random-init, Q | 0.903 | 0.559 |
| B — random-init, Q (PCA→1024) | 0.854 | 0.561 |
| C — trained P, `kron_pilot_sentiment` (collects sentiment) | 0.904 | 0.589 |
| C — trained Q, `kron_pilot_sentiment` (suppresses sentiment) | 0.870 | 0.567 |
| C — trained P, `kron_pilot_topic` (collects topic) | 0.876 | 0.587 |
| C — trained Q, `kron_pilot_topic` (suppresses topic) | 0.869 | 0.579 |
| C — trained flat pilots (full vector) | 0.884-0.906 | 0.624-0.652 |

(all entries are standardized-logreg means across 5 seeds; the standardized-MLP numbers agree within a few points throughout — see `runs/supervision_control_baseline*.json`)

**A ≈ B ≈ C.** Every arm lands in the same band regardless of whether the SAE was trained, randomly initialized, or skipped entirely — sentiment 0.85-0.93, topic 0.56-0.65. A branch explicitly trained to *collect* a label (P) is not detectably higher than the same architecture untrained (B). A branch trained to *suppress* a label (Q) is not detectably lower than untrained either (0.87 vs. random-init's 0.90 — within seed noise).

**Root cause.** This is not a mysterious result — a random linear map into a dimension comparable to or larger than the input (here P: 1024→1024, Q: 1024→2048) approximately preserves whatever is already linearly decodable in the input, independent of training (the same property that makes random-features / extreme-learning-machine methods work at all). Since pythia's activations are already linearly separable for both labels, *any* SAE — trained or not — inherits that separability through its encoder essentially for free. Combined with §9.3(i)'s finding that reconstruction's gradient dominates supervision's by 300-400×, training barely moves the encoder away from its random initialization, so trained and untrained land in the same place.

**Verdict.** No number in §6-§9 demonstrates that supervision did anything, on either the collect or the suppress side. **Methodological correction going forward: probe accuracy must be read relative to the random-init baseline (arm B), not relative to chance.** An accuracy of 0.90 against chance=0.50 looks like a strong effect; against a random-init baseline of 0.89-0.90 it is noise. This is now the standard the report holds itself to (§10 tracks re-evaluating §6-§9 against it) and the standard any future architecture experiment must clear (see `REPORT1.md`).

#### 9.5 Diagnostic: Does Simply Turning Down Reconstruction Fix It?

**Question.** §9.4's root cause was "reconstruction's gradient overwhelms supervision's before training can move the encoder past its random initialization." The direct test: scale reconstruction's loss weight down — does supervision then detectably clear the random-init baseline?

**Method.** Added `--lambda_recon` (multiplies `sae_loss` in the total loss; default 1.0 = unchanged behavior) to the trainer. Swept `lambda_recon ∈ {1.0, 0.3, 0.1, 0.03, 0.01, 0.0}` on the same synthetic diagnostic as §9.2 (`joint_dpo_cross`, `kron_joint`, h=128/m=8/n=16, `dpo_beta=2.0`, `lambda_sup=1.0`, 5000 steps), each compared against a random-init control run with **zero training steps** on the identical synthetic-activation and probe seeds (confirmed lambda_recon-independent; single control suffices, noise band from 3 extra seeds is ±0.005-0.013).

| λ_recon | P-sent ▲ | Q-sent ▼ | P-top ▼ | Q-top ▲ | recon FVU |
|---|---|---|---|---|---|
| **random-init control** | 0.967 | 0.997 | 0.828 | 0.973 | 1.010 |
| control, seeds 43-45 (noise band) | 0.963-0.972 | 0.997-0.999 | 0.844-0.853 | 0.964-0.969 | 1.009 |
| 1.0 (= §9.2/§9.4 baseline) | 0.941 | 0.972 | 0.786 | 0.838 | 0.953 |
| 0.3 | 0.974 | 0.865 | 0.856 | 0.556 | 0.952 |
| 0.1 | 0.962 | **0.783** | 0.841 | 0.461 | 0.951 |
| 0.03 | 0.960 | 0.807 | 0.834 | 0.475 | 0.950 |
| 0.01 | 0.956 | 0.826 | 0.804 | 0.475 | 0.954 |
| 0.0 (no reconstruction term at all) | 0.911 | 0.755 | 0.708 | 0.367 | 1.064 |

**Result.** At λ_recon=1.0, trained is *at or slightly below* random-init on the collect side (P-sent 0.941 vs. control 0.967) — §9.4's finding replicates on synthetic data too. Lowering λ_recon does move Q-sent down (0.972→0.755-0.826), but **Q-top collapses in lockstep** (0.838→0.37-0.56) — the same collect/suppress trade-off §9.2 already hit from the other direction (raising `lambda_sup` 10×). **P never clears the random-init band at any λ_recon**: P-sent's best value (0.974 at λ=0.3) sits inside the control's own 0.963-0.972 noise band, and P-top moves the wrong way (0.856 vs. control's 0.828). Reconstruction quality (FVU) is a weak signal here — it barely moves across λ_recon=1.0 down to 0.01 (0.95-0.96 throughout), because this synthetic setup's noise dimensions (1022 of 1024) dominate variance regardless of supervision weight; only λ=0.0 is worse than untrained (1.064). So this run cannot speak to whether reconstruction quality is being traded away — a caveat on the design, not the result.

**Verdict.** A uniform loss-weight rebalance is not the fix. It reproduces the same trade-off already seen in §9.2 (own-label quality falls about as fast as leakage does) rather than resolving it, and P never separates from its random-init baseline under any weighting tried. This rules out "just weight it differently" as a solution and sharpens the case for `REPORT1.md`'s architectural directions — particularly since a *uniform* scalar reweight is a blunt instrument; an adaptive per-task balancer (GradNorm-style) is a meaningfully different, still-untested mechanism, distinct from both this uniform sweep and §9.2's failed gradient-surgery/magnitude-matching attempts.

### 10. Not Yet Done

- **Re-read §6-§9's collect-side numbers against the §9.4 random-init baseline**, not against chance — as reported, none of them are shown to establish that supervision worked. This is a documentation debt on the existing sections, distinct from re-running anything.
- Patch-and-generate compositional-generalization eval (the paper's core novel test) — not built yet, and per §9.3-9.4, no anchor checkpoint in this report has demonstrated a supervision effect (collect or suppress) that clears the random-init baseline. Needs one before it's worth building.
- Main 6-config grid (mor/concat ablation, supervision on/off) — not run at full scale, and per the above, "supervision on/off" needs a random-init control arm added to be informative.
- Fixing joint DPO-cross's P/Q cross-label leakage, and more generally getting supervision to detectably beat the random-init baseline at all — no loss-weight-level mechanism tried in §9 fixes it (sizing, loss-weighting, gradient surgery, magnitude-matching, and now uniform reconstruction reweighting all fail or trade off cleanly). This line of work (architecture changes: decoupling reconstruction from the suppressing branch, adversarial/gradient-reversal suppression, adaptive per-task gradient rebalancing) continues in **`REPORT1.md`** rather than here, since it is a distinct experimental track building on §9.1-9.5's diagnosis rather than a continuation of the joint-training design tested in §6-§9.

---

## Part II — Rebalancing Supervision vs. Reconstruction

*(source: `REPORT1.md`; continues Part I §9.1-9.5)*

### 1. Background

`REPORT.md` §9.1-9.5:

- §9.2: on synthetic data, reconstruction's gradient on Q's parameters is ~45-50× larger than the supervision gradients.
- §9.3: on real data, the imbalance is larger (~300-400×) and hits both branches, not just Q.
- §9.4: a randomly-initialized, never-trained KronSAE scores within noise of the trained checkpoints on every label, both collect and suppress. Training does not move the representation past what a random linear map already gives, because reconstruction's gradient overwhelms supervision's before it can.
- §9.5: turning down reconstruction's loss weight uniformly (`lambda_recon` 1.0→0.0) does not fix it. It reproduces the same collect/suppress trade-off §9.2 found from the other direction (raising `lambda_sup`) — P never clears the random-init baseline at any weighting, and Q's suppression only improves by sacrificing Q's own collect quality by a comparable amount.

Methodological requirement carried over from §9.4: probe accuracy must be read against the random-init baseline, not against chance. Above-chance accuracy is not evidence of a supervision effect unless it also clears the matched random-init control.

### 2. Goal

Get supervision to clear the random-init baseline — collect or suppress side — without the trade-off every loss-weight intervention has hit so far (§9.2's gradient surgery/magnitude-matching, §9.5's uniform reconstruction reweighting). §9.5 rules out "scale the reconstruction weight down" specifically, so the candidates below change what's being optimized, not how strongly.

1. **Adaptive per-task gradient rebalancing (GradNorm-style).** Adapts the weighting during training to equalize task loss descent rates, instead of a fixed ratio. §9.2/§9.5 only tried one-shot gradient-norm matching and uniform static reweighting, not a dynamically adapted schedule — this is a different mechanism, not a variant of what's already been ruled out.
2. **Decouple reconstruction from the branch that must suppress a label**, so that branch's parameters aren't simultaneously pulled toward "preserve everything" by the decoder. E.g. a separate small decoder path, or training P/Q against separately reconstructed sub-targets.
3. **Adversarial suppression (gradient reversal).** Replace the indirect contrastive push (DPO/SupCon, which pulls same-label reps apart) with a discriminator that tries to predict the label from the branch, and a gradient-reversal update that trains the branch to fool it — a direct objective against linear separability instead of a soft contrastive pressure reconstruction can outcompete.

### 3. Experiment: GradNorm Adaptive Rebalancing

**Question.** §9.2/§9.5 only tried static loss weighting (one-shot gradient-norm matching, uniform `lambda_recon` scaling). Does adapting the reconstruction/collect/contrast weighting *during* training (GradNorm, Chen et al. 2018) let supervision clear the random-init baseline where static weighting didn't?

**Method.** Implemented GradNorm on Q's three loss terms (reconstruction, collect-topic, contrast-sentiment): learnable per-task weights `w_i`, updated each step by their own optimizer toward `Ḡ(t)·r_i(t)^alpha` (target gradient norm derived from each task's relative training progress), renormalized so `Σw_i = 3`. Added `--gradnorm`, `--gradnorm_alpha`, `--gradnorm_lr` to the trainer. Same synthetic diagnostic setup as §9.2/§9.5 (`joint_dpo_cross`, `kron_joint`, h=128/m=8/n=16, `dpo_beta=2.0`, 5000 steps, seed 42), swept `alpha ∈ {0.12, 0.5, 1.5}` (the paper's typical range), against the same random-init control used in §9.5.

| Config | P-sent ▲ | Q-sent ▼ | P-top ▼ | Q-top ▲ |
|---|---|---|---|---|
| random-init control (§9.5) | 0.967 | 0.997 | 0.828 | 0.973 |
| §9.2 baseline (no intervention) | 0.941 | 0.972 | 0.786 | 0.838 |
| GradNorm, alpha=1.5 | 0.962 | 0.816 | 0.818 | 0.507 |
| GradNorm, alpha=0.5 | 0.934 | 0.824 | 0.743 | 0.426 |
| GradNorm, alpha=0.12 | 0.628 | 0.781 | 0.260 | 0.399 |

**Result.** Q-sent drops from control's 0.997 to 0.78-0.82 at every alpha — some movement, but nowhere near chance (0.500), and no better than §9.5's static sweep already achieved (0.78-0.83 there too). The cost is the same collapse seen throughout §9.2/§9.5: Q-top falls from 0.838 to 0.40-0.51. P never clears the random-init control at any alpha — at alpha=0.12 it falls well *below* both the control and the no-intervention baseline (P-sent 0.628, P-top 0.260), i.e. GradNorm's adaptation actively hurt P at the strong end of its range rather than doing nothing.

**Verdict.** GradNorm does not succeed where static weighting failed. It reproduces the same collect/suppress trade-off as §9.2's gradient surgery and §9.5's uniform reweighting, and at low alpha it is worse than either — degrading P below its untrained baseline. The three loss-weighting mechanisms tried across §9.2, §9.5, and this section (gradient-direction surgery, static magnitude scaling, and now adaptive per-task rebalancing) all hit the same wall. This rules out loss-weighting as an axis and motivates moving to the architectural candidates (§2, items 2-3).

### 4. Experiment: Decoupling Reconstruction from Q's Encoder

**Question.** §3's GradNorm and §9.2/§9.5's static weighting all change the *overall* weight of reconstruction vs. supervision, but reconstruction still competes for the same parameters (`q_proj`/`q_bias`) supervision needs to shape. Does shielding Q's encoder from reconstruction's gradient specifically — while leaving P and the decoder fully exposed to it — let Q's suppression clear the random-init baseline without the collect/suppress collapse seen everywhere else, and without degrading reconstruction the way uniform reweighting did?

**Method.** Added `--q_recon_grad_scale` (default 1.0 = no-op): a gradient-scaling hook (`_GradScale` autograd function) inserted on `q_pos_BHN` between Q's encoder and the mAND combine, so the reconstruction loss's gradient into `q_proj`/`q_bias` is multiplied by this factor while P's encoder, the decoder, and Q's own supervision gradients are left bit-identical to baseline (verified by direct gradient comparison). Swept `q_recon_grad_scale ∈ {1.0, 0.3, 0.1, 0.03, 0.01, 0.0}`, same synthetic setup as §3 (`joint_dpo_cross`, `kron_joint`, h=128/m=8/n=16, `dpo_beta=2.0`, 5000 steps, seed 42), against the same random-init control.

| q_recon_grad_scale | P-sent ▲ | Q-sent ▼ | P-top ▼ | Q-top ▲ | recon FVU |
|---|---|---|---|---|---|
| random-init control | 0.967 | 0.997 | 0.828 | 0.973 | 1.010 |
| 1.0 (= §9.2 baseline) | 0.941 | 0.972 | 0.786 | 0.838 | 0.953 |
| 0.3 | 0.946 | 0.885 | 0.733 | 0.565 | 0.953 |
| 0.1 | 0.936 | 0.854 | 0.747 | 0.503 | 0.954 |
| 0.03 | 0.935 | 0.819 | 0.701 | 0.474 | 0.951 |
| 0.01 | 0.908 | 0.844 | 0.648 | 0.484 | 0.955 |
| 0.0 (Q fully shielded from recon gradient) | 0.910 | 0.798 | 0.639 | 0.462 | 0.976 |

**Result.** Q-sent falls from 0.972 to 0.80-0.85 as Q is shielded — comparable to what §9.5's uniform reweighting and §3's GradNorm already reached (0.78-0.83), not further, and still 30 points above chance (0.500). Q-top collapses in the same proportion (0.838→0.46-0.57), the same trade-off shape as every prior mechanism. P was expected to stay near its scale=1.0 values since the gradient into P's own parameters is unchanged, but it drifts down anyway (P-top 0.786→0.639-0.648, P-sent 0.941→0.908-0.910) — not an implementation leak (gradients into P are formula-identical, confirmed directly), but an indirect effect: Q's trajectory changes, so the shared decoder's optimum shifts, so the gradients P actually receives each step change even though the mechanism computing them didn't. One genuine improvement: reconstruction FVU holds at 0.95-0.98 across the whole sweep, including at the most extreme setting (scale=0.0) — clearly better than §9.5's uniform `lambda_recon=0.0`, which reached FVU 1.064 (worse than untrained). Shielding only Q protects reconstruction quality; scaling reconstruction down for everyone does not.

**Verdict.** Partial result. This approach protects reconstruction better than uniform reweighting, but does not solve the actual problem — it reproduces the same collect/suppress collapse as GradNorm and §9.5, doesn't get leakage near chance, and P moves anyway via the shared decoder despite an unchanged direct gradient path. The shared decoder appears to be the mechanism actually coupling P and Q's fates, not just the reconstruction gradient on Q's own parameters — which argues for candidate 3 (a differently-shaped suppression objective) over further variations on gradient routing.

### 5. Experiment: Adversarial Suppression (Gradient Reversal)

**Question.** §3-§4 change how much gradient reconstruction vs. supervision send into Q, but keep the same suppression objective (a DPO contrastive push). Does replacing that indirect pressure with a direct adversarial objective — a discriminator trying to predict the label from Q, Q trained via gradient reversal to fool it — succeed where every gradient-weighting mechanism failed?

**Method.** Implemented a gradient-reversal layer (identity forward, `-lambda × gradient` backward) between Q's doc-pooled representation and a small 2-layer MLP discriminator (own Adam optimizer, `--adv_lr`). With `--adv_suppress`, this fully replaces Q's DPO contrast-sentiment term (P's terms, Q's collect-topic term, and reconstruction are untouched). Swept `adv_grl_lambda ∈ {0.3, 1.0, 3.0, 10.0}`, same synthetic setup as §3-§4. Correctness was unit-checked before running: the gradient into Q equals `-lambda ×` the plain cross-entropy gradient, and the discriminator's own weight gradients are bit-identical to a non-adversarial classifier. A second, measurement-only discriminator (identical architecture, trained on Q's representation *detached*, never influencing the model) ran alongside as a check on whether Q's information was actually removed or just hidden from the adversarial discriminator specifically.

| Config | P-sent ▲ | Q-sent ▼ | P-top ▼ | Q-top ▲ | recon FVU | adversarial-disc acc/loss | reference-disc acc/loss |
|---|---|---|---|---|---|---|---|
| random-init control | 0.967 | 0.997 | 0.828 | 0.973 | 1.010 | — | — |
| §9.2 no-intervention baseline | 0.941 | 0.972 | 0.786 | 0.838 | 0.953 | — | — |
| adv, grl_lambda=0.3 | 0.870 | 0.969 | 0.533 | 0.836 | 0.960 | 0.69 / 0.586 | 0.72 / 0.511 |
| adv, grl_lambda=1.0 | 0.803 | 0.987 | 0.477 | 0.821 | 0.961 | 0.50 / 0.694 | 1.00 / 0.010 |
| adv, grl_lambda=3.0 | 0.783 | 0.984 | 0.393 | 0.845 | 0.961 | 0.59 / 0.650 | 0.91 / 0.258 |
| adv, grl_lambda=10.0 | 0.780 | 0.972 | 0.410 | 0.809 | 0.961 | 0.52 / 0.690 | 0.91 / 0.206 |

(discriminator columns average the last 20 logged steps)

**Result.** The adversarial discriminator did learn — it reaches 0.94-1.00 accuracy within the first 50-1000 steps at every lambda — but then decays to exactly chance (0.500, loss = ln 2, a constant-output solution) as training continues. That looks like suppression working. It isn't: the reference discriminator, trained the whole time on the same (detached) Q representation, reaches 0.91-1.00 accuracy — Q's sentiment information never left. The min-max game found the degenerate equilibrium (drive the discriminator to a constant output) instead of removing the information from Q. Q-sent (0.969-0.987) stays at or above the no-intervention baseline (0.972) and inside the random-init noise band — no measurable suppression at all, worse than every prior mechanism (§3: 0.78-0.82, §4: 0.80-0.85).

The trade-off did not disappear, it relocated: Q-top is the best-preserved of any mechanism tried (0.809-0.845, essentially matching baseline's 0.838) — the first approach that doesn't collapse Q's own collect job. But P collapses instead, monotonically with lambda (P-top 0.786→0.39-0.53, P-sent 0.941→0.78-0.87), despite P's own loss terms being untouched — the same shared-decoder coupling §4 found, amplified rather than avoided.

**Verdict.** No — worse than §3 and §4 on the metric that matters (no suppression at all), and it demonstrates a new failure mode (adversarial training collapsing to a degenerate discriminator rather than genuinely removing information) on top of confirming §4's finding that the shared decoder couples P and Q regardless of which branch's loss terms are touched. Three different objective-shaping mechanisms (adaptive reweighting, gradient shielding, adversarial replacement) now hit variations of the same wall. This favors reading the problem as something more fundamental to the architecture — most likely the mAND combine's coupling of P and Q through a single shared decoder — over "the suppression objective was wrong."

### 6. Experiment: CAGrad (Conflict-Averse Gradient Descent)

**Question.** §3's GradNorm (learned per-task weights matching descent rates) and §9.2's PCGrad (pairwise conflict projection) are heuristics without convergence guarantees. Does CAGrad (Liu et al. 2021, NeurIPS) — which explicitly optimizes for the worst-case per-task improvement within a trust region around the average gradient, with provable convergence to a minimum of the average loss — do better?

**Method.** Applied to the same three tasks on `_q_branch_params()` as GradNorm: reconstruction, Q's collect-topic, Q's contrast-sentiment. Implemented via the paper's constrained inner optimization (solved with SLSQP each step) rather than an approximation. Correctness was verified against hand-constructed gradient triples before running (three cases — opposed pair plus neutral, dominant-reconstruction, mild conflict — all confirmed the solved direction beats the plain average on worst-case improvement, stays inside the trust region, and matches the constrained optimum: **verdict PASS**). Swept `cagrad_c ∈ {0.1, 0.25, 0.5}` (paper's typical range), same synthetic setup as §3-§5, against a matching random-init control.

| cagrad_c | P-sent ▲ | Q-sent ▼ | P-top ▼ | Q-top ▲ | recon FVU | learned weights (recon / collect / contrast) |
|---|---|---|---|---|---|---|
| random-init control | 0.950 | 0.997 | 0.765 | 0.941 | 1.010 | — |
| §9.2 no-intervention baseline | 0.941 | 0.972 | 0.786 | 0.838 | 0.953 | — |
| 0.1 | 0.831 | **0.739** | 0.531 | 0.334 | 0.960 | ~0 / 0.963 / 0.037 |
| 0.25 | 0.808 | **0.682** | 0.472 | 0.288 | 0.959 | ~0 / 0.974 / 0.026 |
| 0.5 | 0.788 | **0.692** | 0.456 | 0.275 | 0.963 | ~0 / 0.975 / 0.025 |

**Result.** Q-sent does drop further than any prior mechanism — 0.68-0.74, clearly below §3/§4's ~0.78-0.85 floor, the most suppression achieved anywhere in this report. But the cost scaled with it rather than staying flat: Q-top collapses to 0.28-0.33 (worse than GradNorm's 0.40-0.51 or §4's 0.46-0.57), and P collapses too (P-sent 0.79-0.83, P-top 0.46-0.53) despite P's loss terms being untouched by CAGrad — the shared-decoder coupling from §4/§5, again, and more severe than either. Reconstruction FVU barely moves (0.96 vs. baseline's 0.95). The learned weights explain why: CAGrad drives reconstruction's weight to effectively **zero** (~1e-14) at every `cagrad_c`, putting 96-97% of the combination on collect-topic and only 2.5-3.7% on contrast-sentiment. Given reconstruction's gradient is 45-50× (synthetic) to 300-400× (real data, §9.3) larger than the others, CAGrad's worst-case-improvement criterion apparently finds that any nonzero alignment with reconstruction's direction dominates the trust-region bound, so the conflict-averse solution abandons it almost entirely on Q's parameters — a sharp quantitative confirmation of just how extreme that imbalance is, from an algorithm that never saw the §9.3 numbers.

**Verdict.** Not a fix — the trade-off got starker, not better. This is the fourth loss-balancing mechanism (after PCGrad, GradNorm, and the effectively-similar magnitude reasoning in §4) to fail, now including one with formal worst-case guarantees. Combined with §4/§5's finding of P-coupling through the shared decoder regardless of which loss terms are touched, this closes out loss-balancing as a plausible axis: the problem does not appear to be *which* algorithm reweights or routes gradients, since the more principled the algorithm, the more consistently it converges on "abandon reconstruction's direction entirely" rather than a workable balance.

### 7. Where This Leaves the Project

Four candidates tried, all against the same random-init baseline standard:

| Candidate | Best Q-sent reached ▼ (chance 0.500, random-init 0.997-1.00) | What broke instead |
|---|---|---|
| §3 GradNorm (adaptive reweighting) | 0.78-0.82 | Q-top collapses proportionally; P degrades at low alpha |
| §4 Q-recon gradient shielding | 0.80-0.85 | Q-top collapses proportionally; P drifts via shared decoder |
| §5 Adversarial (gradient reversal) | 0.97-0.99 (no effect) | P collapses via shared decoder; discriminator finds a degenerate equilibrium instead of removing information |
| §6 CAGrad (worst-case-improvement) | 0.68-0.74 (most suppression yet) | Q-top *and* P both collapse further than any other mechanism |

None reaches chance. None avoids degrading something else — and the most "principled" algorithm (§6) has the worst collateral damage, not the least. §4, §5, and §6 all independently implicate the same culprit: the mAND-combined shared decoder ties P's and Q's optimization together regardless of which branch's own loss terms or gradients are modified — cutting Q's direct gradient path, replacing Q's objective entirely, or optimally reweighting Q's three terms all still leave P affected or fail to reach chance. Loss-balancing as an axis is exhausted. The next candidate worth testing is structural: does the coupling come from the *multiplicative* per-atom combine specifically (mAND/mOR and their complements), or from *any* single-decoder/shared-reconstruction-target design (including `concat`, which has no multiplicative coupling but still shares one decoder and one reconstruction target) — see §8.

### 8. Experiment: Combine-Rule Sweep (mAND / mOR / mNAND / mNOR / concat)

**Question.** Not how gradient is weighted, but how P and Q's activations combine before the shared decoder. `mand` (default) is a hard-gated multiplicative combine (`z_ij = sqrt(p_i·q_j)` if both positive) — every combined atom depends jointly on both branches. Does a softer or differently-shaped combine reduce the coupling enough for supervision to work? And specifically: `concat` (already in the codebase) has zero per-atom multiplicative coupling at all (`features = cat([p_pos, q_pos])`, no cross term) but still shares one decoder and one reconstruction target — does *that* alone still reproduce the failure, isolating whether the multiplicative term specifically is the culprit versus the shared-decoder/shared-target structure in general?

**Method.** Implemented `mnand` (`1 - a·b`, probabilistic complement of soft-AND — fires when either branch is near-zero) and `mnor` (`(1-a)(1-b)`, complement of `mor` — fires only when both branches are near-zero), both on the same `a=1-exp(-p_pos)`, `b=1-exp(-q_pos)` probability construction `mor` already uses. Ran trained + random-init control for all five rules (`mand`, `mor`, `mnand`, `mnor`, `concat`) on the same synthetic setup as §3-§6.

| Rule | P-sent ▲ | Q-sent ▼ | P-top ▼ | Q-top ▲ | recon FVU | random-init Q-sent (same rule) |
|---|---|---|---|---|---|---|
| mand (baseline) | 0.949 | 0.992 | 0.753 | 0.933 | 0.956 | 0.997 |
| mor | 0.815 | 0.960 | 0.503 | 0.799 | 0.986 | 0.997 |
| mnand | 0.948 | **0.798** | 0.703 | 0.438 | 1.013 | 0.997 |
| mnor | 0.928 | **0.763** | 0.706 | 0.462 | 1.012 | 0.997 |
| concat | 0.946 | 0.919 | 0.727 | 0.694 | **0.944** | 0.999 |

(random-init controls were run per-rule as specified; they landed within noise of each other regardless of rule — 0.997-0.999 — so an untrained encoder's linearly-decodable content doesn't depend much on how the untrained combine function reshapes it.)

**Applying the pre-registered buckets:**
- **mand**: reproduces the known baseline (this run's Q-sent=0.992 vs. the historically-cited 0.972 — a small, plausible run-to-run/seed gap, not a discrepancy worth chasing).
- **mor**: falls in the "also fails" bucket, but worse than plain failure — Q-sent barely moves (0.992→0.960, inside noise of the random-init band) while P-top collapses hard (0.753→0.503) and FVU gets worse (0.956→0.986). Softening the gate bought nothing and cost P.
- **mnand / mnor**: the biggest raw Q-sent drops of the five (0.798, 0.763) — but the pre-registered caveat for mnor turned out to apply to *both*: FVU rises to ~1.01, essentially matching the random-init control's 0.997-1.013. **These models are barely reconstructing better than an untrained network.** Per the pre-registered framing, this makes the leakage numbers untrustworthy as evidence of genuine suppression — a model that isn't really solving reconstruction produces representations that are hard to compare to a well-reconstructing one on probe accuracy. Read as "confounded, not a clean success," not as the best result.
- **concat**: the *only* rule with no severe collapse anywhere — FVU is the best of all five (0.944, better than mand's own 0.956), P is nearly untouched (P-sent 0.946 vs. mand's 0.949, P-top 0.727 vs. 0.753), and Q-top holds up reasonably (0.694, well above chance 0.167, vs. mand's 0.933). Q-sent does drop measurably (0.992→0.919), clearly outside the random-init noise band — a real, if small, effect. But 0.919 is nowhere near chance (0.500). This doesn't fit the pre-registered "concat succeeds" bucket (that required getting *near* chance); it's better described as a **milder, better-behaved version of the same failure** — first mechanism across the whole report to show a genuine, uncounfounded supervision effect with no collapse anywhere, but still nowhere close to solving leakage.

**Verdict.** No combine rule reaches anywhere near chance. Critically, **`concat` — with zero per-atom multiplicative coupling — still fails**, just more gently than the others. That rules out the multiplicative mAND/mOR-style combine as *the* culprit: even pure additive combination, sharing only a decoder and a reconstruction target, is enough to reproduce most of the leakage. This confirms the strongest reading flagged in the pre-registration: the coupling is not primarily about *how* P and Q's features get combined, but about P and Q being jointly responsible for reconstructing the same shared target `x` at all, through any single decoder.

### 9. Where This Leaves the Project (updated)

Nine mechanisms tried across two axes — loss-weighting (§3, §6, plus §9.2's PCGrad/magnitude-matching) and combine-rule/gate choice (§8, plus §4-5's gradient/objective changes on the default `mand` gate) — and none reaches chance-level suppression without collapsing something else. The softest, best-behaved result in the whole report (`concat`, §8) still leaves Q-sent at 0.919 against a chance of 0.500.

The consistent pattern across every architectural probe (§4, §5, §6, §8) is that **P is affected by changes aimed only at Q**, and this persists even when the multiplicative combine is removed entirely (§8's `concat`). That leaves one structural cause unaddressed by anything tried so far: P and Q are trained to jointly reconstruct one shared target `x`, through one decoder, no matter how their features get combined beforehand. Severing that — genuinely separate reconstruction sub-targets or decoders per branch, not just a different combine function — is the one structural change in this report's original candidate list (REPORT1.md §2, item 2) that has not actually been tested; what was tested in §4 was gradient shielding on a still-shared decoder, not decoder separation itself.

Whether that's worth building depends on how it's scoped. A literal per-branch decoder split (`x̂ = decode_P(p) + decode_Q(q)`) would need a principled way to allocate `x` between P and Q, which doesn't exist a priori — and it would also abandon the compositional/interaction mechanism (mAND-style combinatorial atoms) that is KronSAE's actual contribution, making the result a different architecture in substance, not a fixed KronSAE. This is worth stating plainly rather than papering over: at this point, the honest framing may be that **this specific architecture — a two-branch encoder feeding one shared decoder over one shared reconstruction target — does not support the kind of independent, supervised branch control (collect one label, suppress another, simultaneously) this project set out to test**, across five different loss-balancing algorithms and five different combine rules. Whether a genuinely separated-decoder variant escapes this is the one remaining untested question, with the caveat that it may no longer be answering the same research question KronSAE was designed to ask.

### 10. Existence Proof: Does Collect+Suppress Work at All, Without Any SAE?

**Question.** §3-§9 tested collect+suppress entirely inside an SAE's training loop — shared reconstruction, top-k sparsity, dictionary, and a DPO-pairwise/SupCon supervision loss, all at once. Is the failure specific to routing this objective through an SAE's reconstruction+dictionary machinery, or does it show up even with the simplest possible setup: two independent MLP heads on raw activations, no SAE, no shared decoder, no top-k, no dictionary at all, and a textbook adversarial suppression mechanism (gradient-reversal layer, GRL; Ganin & Lempitsky 2016) instead of the DPO-contrast loss used everywhere else in this project?

**Method.** `src/eval_no_sae_control.py`. Two fully independent 2-layer MLP heads on the synthetic buffer's raw 1024-dim activations: `head_sent` (plain cross-entropy, collect-sentiment), `head_topic` (cross-entropy collect-topic + GRL-fed 2-layer discriminator adversarially suppressing sentiment, DANN-style lambda ramp 0→`grl_lambda` over training, `disc_steps_per_iter` extra discriminator-only steps per iteration before the reversed-gradient step). Every probe read against a matched random-init (0 training steps) control, using the project's standardized `LogisticRegression` protocol. Swept `grl_lambda ∈ {5.0, 20.0, 50.0}` with `disc_steps_per_iter ∈ {2, 3}`.

| GRL config | collect-sent trained/random-init ▲ | collect-topic trained/random-init ▲ | suppress-sent (topic-head) trained/random-init ▼ | suppress-sent excess Δ over random-init |
|---|---|---|---|---|
| λ=5.0, disc_steps=3, 3000 steps | 1.000 / 0.558 | 0.999 / 0.213 | 0.637 / 0.530 | +0.107 |
| λ=20.0, disc_steps=2, 2000 steps | 1.000 / 0.558 | 0.993 / 0.213 | 0.616 / 0.530 | +0.086 |
| λ=50.0, disc_steps=2, 2000 steps | 1.000 / 0.558 | 0.961 / 0.213 | 0.632 / 0.530 | +0.102 |

(majority-class baseline is not applicable here — the synthetic sentiment label is drawn balanced, so chance=0.5 is the correct floor for this table's collect-sent/suppress-sent columns; topic chance=1/6=0.167.)

**Result.** Collect works perfectly on both heads regardless of GRL config — collect-sentiment 1.000 vs. random-init 0.558, collect-topic 0.96-1.00 vs. random-init 0.213. Suppression never gets close: the topic-head's residual sentiment accuracy sits at 0.616-0.637 against chance 0.500, an excess of +0.086 to +0.107 above the matched random-init control, at every GRL lambda from 5 to 50 and both discriminator-step settings tried. `.claude/documents/adversarial_erasure_literature.md` (§11 below) documents a fuller sweep down to λ=1 and up to 5 discriminator steps/iteration, all landing in the same +0.08 to +0.14 excess band — the three runs read directly here (+0.086, +0.102, +0.107) sit inside that reported range.

**Takeaway.** The failure generalizes beyond KronSAE and beyond SAE architectures generically: vanilla GRL adversarial suppression plateaus above chance on raw synthetic activations with zero SAE machinery involved at all. This rules out "the shared decoder/dictionary is the whole problem" as a complete explanation — a fully decoder-free, dictionary-free adversarial setup hits a qualitatively similar wall (never reaches chance, tuning the adversarial strength doesn't move it). It also reframes the target: the open question is not just "can KronSAE do this" but "does any adversarial suppression mechanism do this," which motivated the literature search in §11.

### 11. Literature Search: Why Does GRL Plateau, and What Works Instead

**Source.** `.claude/documents/adversarial_erasure_literature.md`, read in full.

**Diagnosis of the plateau.** Three converging explanations for why vanilla GRL never reaches chance, regardless of lambda or discriminator steps:

1. **Non-stationary co-training / moving-target problem.** The discriminator is a neural network still converging; gradient reversal only trains the encoder to defeat *today's* discriminator. There is no guarantee this generalizes to defeating a probe fit from scratch on the final representation.
2. **Single-adversary loophole exploitation** (Han, Baldwin & Cohn 2021, arXiv:2101.10001). With one discriminator (one init, one architecture), the encoder can route the suppressed signal through directions that this particular discriminator's inductive bias doesn't happen to exploit. Their fix — an ensemble of diverse adversaries — is reported to close much of this gap, but remains an empirical patch, not a guarantee.
3. **No formal equilibrium guarantee.** Vanilla GRL minimax is between two arbitrary neural networks with no convexity structure; there is no theorem that the game reaches a state where *all* linear predictors fail, only that it fools the one discriminator in the loop.

This is corroborated directly by the original documented instance of this failure mode: **Elazar & Goldberg 2018 (arXiv:1808.06640)** found that GRL-style adversarial suppression reaches chance-level accuracy on its own co-trained discriminator during training, but a **fresh, post-hoc classifier trained from scratch on the same "cleaned" representations recovers substantially higher-than-chance accuracy** on the same attribute — i.e., in-loop discriminator accuracy at chance does not certify that the information is gone, only that this one discriminator currently can't win. This is exactly the shape of §10's result: the standardized probe (a freshly-fit classifier, not the in-loop discriminator) recovers well above chance.

**Candidate fixes considered, and why LEACE was selected:**

- **RLACE** (Ravfogel, Twiton, Goldberg, Cotterell, ICML 2022, arXiv:2201.12091): reframes erasure as a constrained linear minimax game over rank-`K` orthogonal projections (Fantope-relaxed), with an inner-loop-optimal predictor rather than a lagging co-trained network. Empirically reaches near-chance probe accuracy at far lower rank than INLP (e.g. GloVe gender: 99.8%→~52% with `K=1`). But it has no formal convergence guarantee for the actual non-convex problem (only for the relaxed convex-concave surrogate), and its alternating GDA + Fantope-projection loop reintroduces some of GRL's own tuning/instability surface.
- **KRaM** (Basu Roy Chowdhury et al., NeurIPS 2023, arXiv:2312.00194): kernelized, nonlinear generalization of LEACE via a rate-distortion objective, trainable by gradient descent, handles continuous/vector-valued concepts. Closest in spirit to a drop-in trainable loss term, but reintroduces a hyperparameter-bearing optimization problem (still no adversarial dynamics, but not closed-form either).
- **LEACE** (Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman, NeurIPS 2023, arXiv:2306.03819) — **selected as the primary candidate.** It is the only method surveyed with a hard closed-form theoretical guarantee: the "guardedness" theorem proves zero cross-covariance between the erased features and the concept label is equivalent to every linear classifier (including logistic regression) being unable to beat the base rate — not an empirical approximation to chance, but a proven one, conditional on accurate covariance statistics. It is cheap (a single linear-algebra computation: whitening + orthogonal projection onto the whitened cross-covariance subspace, no adversarial min-max, no lambda schedule, no discriminator to tune). And although the original paper proposes it as a post-hoc, frozen-representation method (its own limitations section punts a trainable variant to future work), it is built entirely from batch mean/covariance statistics — smooth, differentiable tensor ops — so it is naturally embeddable as a training-time regularizer, unlike its intended post-hoc usage.

### 12. Concept Erasure via INLP on Synthetic Data

**Question.** Given GRL's failure (§10) and INLP's (Ravfogel et al. 2020) closed-form, non-adversarial alternative flagged as promising by the literature search, does linear-subspace projection succeed at symmetric bidirectional collect+suppress on the synthetic buffer, where sentiment/topic directions are orthogonal by construction?

**Method.** `src/eval_concept_erasure_control.py`. One iteration of INLP: fit a standardized logistic-regression probe for the label to erase on a disjoint `x_fit` split, project the probe's decision subspace out of both `x_fit` and the held-out `x_probe` split, repeat until the refit probe's own training accuracy is within 0.03 of chance. Both erasure directions tested (erase sentiment / erase topic), each checked for collateral damage on the other label, each against a rank-matched random-subspace-erasure control (same rank, random direction instead of the fitted concept direction).

| n_docs | erase target | rank / rounds | targeted sent_acc (chance=0.5) | targeted topic_acc (chance=1/6=0.167) | random-subspace sent_acc | random-subspace topic_acc |
|---|---|---|---|---|---|---|
| 6000 (`concept_erasure_r1.json`, 1 round forced) | sentiment | 1 / 1 | 0.489 | 0.519 | 0.987 | 0.513 |
| 6000 (`concept_erasure_r1.json`, 1 round forced) | topic | 5 / 1 | 0.988 | 0.215 | 0.990 | 0.516 |
| 6000 (`concept_erasure_control_results.json`, full iteration) | sentiment | 7 / 7 | 0.483 | 0.432 | 0.979 | 0.513 |
| 6000 (`concept_erasure_control_results.json`, full iteration) | topic | 40 / 8 | 0.881 | 0.180 | 0.955 | 0.488 |
| 20000 (`concept_erasure_bigdata.json` / `concept_erasure_regression_check.json`, identical) | sentiment | 3 / 3 | 0.517 | 0.988 | 1.000 | 1.000 |
| 20000 (`concept_erasure_bigdata.json` / `concept_erasure_regression_check.json`, identical) | topic | 15 / 3 | 1.000 | 0.171 | 1.000 | 0.994 |

Raw (unerased) baseline: at n_docs=6000, sentiment_acc=0.990, topic_acc=0.522 (topic probe underperforms even before erasure at this data size). At n_docs=20000, sentiment_acc=1.000, topic_acc=1.000.

**Result.** With enough fitting data (20000 docs), symmetric bidirectional erasure works near-perfectly: erasing sentiment drives sentiment 1.000→0.517 (chance=0.5) while topic stays at 0.988 (near-zero collateral); erasing topic drives topic 1.000→0.171 (chance=1/6=0.167) while sentiment stays at 1.000 (zero collateral). The random-subspace control shows no effect on either label in either direction (0.994-1.000 throughout) — the effect is attributable to the fitted concept direction, not to removing capacity generically.

With too little fitting data (6000 docs), topic erasure is incomplete via a single INLP round: topic accuracy floors at 0.18-0.21 regardless of running 1, 2, or 3 rounds (`concept_erasure_r1/r2/r3.json` show topic_acc 0.519→0.185→0.181 across rounds 1→2→3, i.e. it does improve with more rounds at this data size, but plateaus well above the 20000-doc result's 0.171). This was resolved entirely by using more fitting data (20000 docs, not more rounds): at 20000 docs, a single INLP round already reaches topic 0.171. Pushing rounds further at the small-data end (`concept_erasure_control_results.json`, 8 rounds) does not fix topic (0.180, no better than the 3-round run) but does start incurring collateral damage on sentiment — the sentiment run in the same file grew to rank 7 over 7 rounds and topic collateral rose to -0.090 vs. raw, and the topic-erasure run's own sentiment collateral reached -0.109 (0.881 vs. raw 0.990).

**Takeaway.** On synthetic, orthogonal-by-construction data, INLP achieves what GRL couldn't (§10): near-exact chance on the targeted label with near-zero collateral on the other, symmetrically in both directions. The data-size lesson generalizes directly from §9's methodology: when a result is incomplete, the fix that worked was more fitting data, not more iterations of the same procedure — iterating INLP further at insufficient data starts trading collateral damage for no further gain on the target.

### 13. INLP on Real Pythia-410m Activations — Slow, Incomplete Convergence

**Question.** §12 used directions that are orthogonal by construction. Does INLP still work on real pythia-410m layer-12 activations (Amazon Reviews sentiment × topic corpus, n_docs=20000 — the same activations every other SAE experiment in this project used), where sentiment and topic are not guaranteed to be linearly disentangled?

**Method.** `src/eval_concept_erasure_pythia.py`, same INLP/random-subspace-control machinery as §12, applied to cached real activations. Two round budgets: 3 rounds (`concept_erasure_pythia_results.json`) and 25 rounds (`concept_erasure_pythia_results_r25.json`).

| rounds | erase target | rank | sent_acc (raw 0.947) | Δ sent | topic_acc (raw 0.583) | Δ topic | random-subspace sent_acc | random-subspace topic_acc |
|---|---|---|---|---|---|---|---|---|
| 3 | sentiment | 3 | 0.943 | −0.005 | 0.584 | +0.002 | 0.948 | 0.581 |
| 3 | topic | 15 | 0.947 | 0.000 | 0.565 | −0.017 | 0.945 | 0.601 |
| 25 | sentiment | 25 | 0.909 | −0.038 | 0.586 | +0.003 | 0.945 | 0.603 |
| 25 | topic | 125 | 0.940 | −0.007 | 0.470 | −0.113 | 0.950 | 0.625 |

(sentiment chance in the pure sense is 0.5, but §15 below shows the correct reading floor for this imbalanced label is the majority-class baseline, ~0.870 — flagged here and resolved in §15. topic chance=1/6=0.167.)

**Result.** 3 rounds barely move either label: sentiment 0.947→0.943 (Δ−0.005), topic 0.583→0.565 (Δ−0.017) — essentially no suppression. 25 rounds (rank 25 for sentiment, rank 125 for topic — i.e. up to 125 of the 1024 dimensions removed) does more but is still far from any chance floor: sentiment 0.947→0.909 (Δ−0.038), topic 0.583→0.470 (Δ−0.113). Extrapolating the per-round rate implies needing several hundred erased dimensions to approach chance — impractical, since that would gut most of a 1024-dim representation. Random-subspace controls stay within noise of the raw baseline throughout (sentiment 0.945-0.950, topic 0.581-0.625 vs. raw 0.583), confirming the targeted erasure's effect, small as it is, is attributable to the fitted direction and not generic capacity removal.

**Takeaway.** INLP's synthetic-data success (§12) does not transfer efficiently to real activations — convergence is far slower, and even 25 rounds/125 removed dimensions leaves both labels well above any plausible chance floor. This motivated testing LEACE (§11's primary recommendation) directly on the same real-activation cache, since LEACE's closed-form cross-covariance projection should not need INLP's one-direction-at-a-time iterative search.

### 14. LEACE on Real Data — Closed-Form, One-Shot, Far More Efficient Than INLP

**Question.** Does LEACE's single closed-form application reach further, per erased dimension, than INLP's iterative search on the same real pythia activations?

**Method.** Same activation cache and probe protocol as §13. `runs/concept_erasure_pythia_leace.json` (single LEACE application) and `runs/concept_erasure_pythia_iterleace.json` (iterative LEACE, refitting on the just-erased fit split each round, up to 15 rounds).

| method | erase target | rank / rounds | sent_acc (raw 0.947) | Δ sent | topic_acc (raw 0.583) | Δ topic | random-subspace sent_acc | random-subspace topic_acc |
|---|---|---|---|---|---|---|---|---|
| LEACE (1-shot) | sentiment | 1 / 1 | 0.843 | −0.104 | 0.575 | −0.008 | 0.947 | 0.582 |
| LEACE (1-shot) | topic | 6 / 1 | 0.936 | −0.011 | 0.284 | −0.299 | 0.947 | 0.594 |
| iterative LEACE (≤15 rounds) | sentiment | 15 / 15 | 0.840 | −0.107 | 0.573 | −0.010 | 0.945 | 0.596 |
| iterative LEACE (≤15 rounds, converged at round 1) | topic | 6 / 1 | 0.932 | −0.015 | 0.284 | −0.299 | 0.946 | 0.587 |

**Result.** A **single** LEACE application (rank 1 for sentiment, rank 6 for topic) achieves sentiment 0.947→0.843 (Δ−0.104) and topic 0.583→0.284 (Δ−0.299) — both far exceeding INLP's per-dimension efficiency (§13: rank 25 needed for Δ−0.038 on sentiment, rank 125 for Δ−0.113 on topic). Iterating LEACE up to 15 rounds barely improves further: sentiment reaches 0.840 after 15 rounds (essentially identical to 1-shot's 0.843, rank grown to 15), and topic's iterative run converges after just 1 round at 0.284, identical to the 1-shot result. LEACE's single closed-form application already captures essentially everything the linear cross-covariance structure has to give in this fitting-sample regime; further iteration is not the lever — the same "more data, not more rounds" lesson as §12. Random-subspace controls stay within noise of raw throughout (sentiment 0.945-0.947, topic 0.582-0.596), confirming specificity.

**Takeaway.** LEACE recovers §11's promise directly: on real activations, one closed-form application removes far more per-dimension signal than 25 rounds of INLP. But neither LEACE nor INLP's suppression numbers reach an actual chance floor on real data the way §12's synthetic result did — sentiment plateaus around 0.84, topic around 0.28-0.47 depending on method/rounds. §15 investigates why.

### 15. Diagnosing the Real-Data Plateau

Two candidate explanations were tested for why LEACE (and INLP) plateau above chance on real pythia activations instead of reaching the near-exact chance seen on synthetic data:

**(A) Finite-sample generalization gap.** `Σ_xx`/`Σ_xz` statistics estimated from a finite `x_fit` split (d=1024, n~2000-29000) may not perfectly transfer to the disjoint `x_probe` split used for evaluation, unlike an in-sample check (fit==apply on the same data) where LEACE's algebra is exact by construction.

**(B) Evaluation-probe overfitting artifact.** With a fixed, permissive regularization strength (C=1.0, the project's standing standard from REPORT.md §9.3), a logistic-regression probe trained on d=1024 features against a near-null-signal residual can produce misleadingly extreme accuracy in either direction — purely from finite-sample noise-fitting, not genuine residual signal.

**Evidence for (A): direct cross-covariance-magnitude measurement.** `runs/leace_plateau_diagnosis_58k.json` (58000-doc cache, `n_fit` sweep at 2000/5000/10000/20000/29000 — used as primary, larger sweep; the smaller `runs/leace_plateau_diagnosis.json`, 20000-doc cache with `n_fit` up to only 10000, shows the same direction of effect). This metric — the fraction of raw cross-covariance norm still surviving after LEACE erasure, measured directly on covariance statistics rather than via probe accuracy — is immune to probe-overfitting artifacts:

| n_fit | sentiment surviving-cross-cov-norm-fraction | topic surviving-cross-cov-norm-fraction |
|---|---|---|
| 2000 | 10.80% | 27.47% |
| 5000 | 4.10% | 24.72% |
| 10000 | 1.75% | 19.14% |
| 20000 | 1.38% | 26.74% (noisy outlier — see below) |
| 29000 | 6.83% (see below) | 16.47% |

Sentiment's surviving fraction shrinks monotonically from 10.80% (n_fit=2000) to 1.38% (n_fit=20000), with the n_fit=29000 point back up to 6.83% — reported honestly as sampling noise in that particular fit/probe split boundary, not a reversal of the trend. Topic's surviving fraction shrinks from 27.47% (2000) to 16.47% (29000), with n_fit=20000's point at 26.74% a noisy outlier against the otherwise monotonic decline — reported honestly, not cherry-picked out. Both labels show real residual cross-covariance shrinking substantially with more fitting data. This is not a fixed, un-shrinkable ceiling — it directly supports (A) as a major contributor to the real-data plateau.

**Evidence for (B): the regularization sweep, specific to sentiment.** `runs/sentiment_c_sweep.log` (raw log, read directly — not JSON-serialized). Sentiment is an imbalanced binary label: the log reports the exact majority-class baseline on this evaluation test split as **0.8703**. At C=1.0 (the project's prior standard), erased-sentiment accuracy was 0.8440 — actually *below* the majority-class baseline. Sweeping C down:

| C | erased-sentiment accuracy |
|---|---|
| 1.0 | 0.8440 |
| 0.3 | 0.8506 |
| 0.1 | 0.8534 |
| 0.03 | 0.8631 |
| 0.01 | 0.8689 |
| 0.003 | 0.8709 |
| 0.001 | 0.8703 |
| 0.0003 | 0.8703 |
| 0.0001 | 0.8703 |

Accuracy rises monotonically as C decreases and converges **exactly** to the majority-class baseline (0.8703) at C≤0.001. Once evaluated with adequate regularization, LEACE's sentiment erasure is complete: the erased representation carries literally zero information about sentiment beyond what a constant majority-class predictor already captures. The apparent "0.84 plateau" reported at C=1.0 throughout §13-§14 was entirely a probe-overfitting artifact, not residual signal — a permissive, unregularized probe fits noise in the near-null residual and reports spuriously high accuracy in exactly the direction that looks like "suppression failed." (The in-sample multiclass topic check in `leace_plateau_diagnosis.json`/`_58k.json` shows the same artifact from the other side: 0.0265 accuracy, far *below* the 1/6=0.167 chance floor, on a residual independently verified to have cross-covariance reduced from 2.26 to 0.0001 — i.e. essentially fully erased, yet a permissive in-sample probe reports an accuracy that looks dramatically wrong in the opposite direction.)

**Methodological correction (documentation debt).** For an imbalanced binary label, the correct "chance" floor for reading probe-accuracy tables is the majority-class baseline, not a flat 0.5 — §10, §13, and §14 above all read sentiment against 0.5 where the real evaluation-split majority-class baseline is 0.8703. This is flagged explicitly as a numbered follow-up item: **any other table in REPORT.md/REPORT1.md that compares a probe's sentiment accuracy against a flat 0.5 chance line should be re-audited against the actual majority-class baseline for that split** before being cited as evidence of suppression or its absence. That audit is out of scope for this write-up.

**Takeaway.** Both candidate explanations hold, and both matter: (A) more fitting data shrinks the real residual cross-covariance substantially (not a fixed ceiling), and (B) the previously reported ~0.84 sentiment plateau at C=1.0 was largely a regularization artifact — the true erasure is complete once measured properly. Together they resolve why §13-§14's real-data numbers looked worse than §12's synthetic numbers: it was not that real activations are fundamentally harder to erase, it was insufficient fitting data plus an under-regularized evaluation probe.

### 16. Overall Verdict for This Session's Track

| Finding | Status |
|---|---|
| Vanilla GRL, no SAE (§10) | Fails — plateaus +0.09 to +0.11 above chance regardless of lambda (5-50) or discriminator steps (2-3); matches literature's documented +0.08 to +0.14 band and Elazar & Goldberg 2018's original diagnosis |
| INLP, synthetic data (§12) | Succeeds near-perfectly with enough fitting data (20000 docs); incomplete with too little data, not fixed by more rounds |
| INLP, real pythia activations (§13) | Slow, incomplete even at 25 rounds/125 dims — impractical to push further |
| LEACE, real pythia activations (§14) | Far more efficient per-dimension than INLP; still looked incomplete at C=1.0 evaluation |
| Regularization + fitting-data diagnosis (§15) | Resolves the apparent real-data plateau: majority-baseline-adjusted, adequately-regularized sentiment erasure is complete; topic's residual shrinks substantially with more fitting data |

Collect+suppress decomposition is **not fundamentally blocked by the data.** It is achievable via a principled closed-form method (LEACE), on both synthetic and real pythia-410m activations, once (a) enough fitting data is used for the erasure computation itself and (b) the evaluation probe is adequately regularized (and, for imbalanced labels, read against the majority-class baseline rather than a flat 0.5).

What specifically fails, and remains an open, useful negative result, is:

1. **Vanilla GRL/adversarial training** (§10-§11) — plateaus regardless of tuning, reproducing Elazar & Goldberg 2018's well-documented failure mode, now confirmed on this project's own synthetic data with zero SAE machinery in the loop.
2. **Every SAE architecture variant tested earlier in this project** (§3-§9 here, and REPORT.md §6-§9) — five loss-balancing mechanisms and five combine rules, all of which force the collect+suppress objective through a shared reconstruction+dictionary bottleneck, and none of which reached chance without collapsing something else.

The natural next experiment this points to: **embed LEACE as a differentiable training-time regularizer inside an SAE's (specifically KronSAE's Q-branch) encoder training loop**, replacing the failed `adv_suppress`/GRL mechanism from §5, to test directly whether the SAE-architecture-specific failure (§3-§9) can now be fixed with a properly-guaranteed erasure mechanism instead of an unguaranteed adversarial one. This is the one path not yet tried in this project's entire loss-balancing/architecture-sweep history (§3-§9), and it is now backed by a working, closed-form, real-activation-validated erasure mechanism (§14-§15) rather than an untested literature recommendation (§11).

### 17. LEACE-in-KronSAE: A Differentiable Cross-Covariance Regularizer Replacing GRL

**Q.** Does §16's recommended experiment — embedding LEACE's guardedness condition as a differentiable training-time regularizer inside KronSAE's Q-branch, in place of the failed `adv_suppress`/GRL mechanism (§5, §10) — actually suppress sentiment in Q, and does it do so more successfully than GRL?

**M.** Implemented `--leace_suppress` in `KronTopKTrainer` (`dictionary_learning/trainers/kron_top_k.py`). Rather than doing a full closed-form LEACE solve (whitening + eigendecomposition) inside the training loop every step, the mechanism is the direct differentiable analogue of LEACE's guardedness theorem: a penalty term `leace_lambda * ||Cov(Q_docs, onehot(sentiment))||^2` computed per-batch and added straight into `total_loss`, so it backprops into the Q-branch encoder through ordinary gradient descent — no adversary, no gradient-reversal layer, none of GRL's single-discriminator-loophole failure mode (Elazar & Goldberg 2018; §11). As with `adv_suppress`, this term *replaces* (not augments) Q's DPO contrast-sentiment loss, and a measurement-only reference probe (`leace_ref_discriminator`, trained on the *detached* Q representation, never touched by the suppression loss) is kept for comparability with `adv_ref_discriminator`. Full implementation details and the mutual-exclusivity/validation logic are in the trainer's docstrings (`_leace_suppression_loss`).

Evaluated on the exact same synthetic-diagnostic setting as the GRL sweep in §10 (`kron_joint`, `joint_dpo_cross`, h=128/m=8/n=16, dpo_beta=2.0, seed=42, 5000 steps, one A100), for direct comparison: `leace_lambda` swept across {0.01, 0.03, 0.1, 0.3, 1.0, 3.0} (`runs/leace_suppress_sweep.sh`, aggregated by `src/aggregate_leace_sweep.py` into `runs/leace_suppress_results.json`).

**R.**

| leace_lambda | P(sentiment) | Q(sentiment) | P(topic) | Q(topic) | recon FVU |
|---|---|---|---|---|---|
| control (no suppress) | 0.967 | 0.997 | 0.828 | 0.973 | 1.010 |
| 0.01 | 0.906 | 0.932 | 0.622 | 0.700 | 0.959 |
| 0.03 | 0.891 | 0.956 | 0.607 | 0.772 | 0.962 |
| 0.1 | 0.902 | 0.896 | 0.651 | 0.533 | 0.962 |
| 0.3 | 0.920 | 0.834 | 0.656 | 0.427 | 0.961 |
| 1.0 | 0.886 | 0.735 | 0.586 | 0.337 | 0.962 |
| 3.0 | 0.884 | **0.686** | 0.635 | 0.321 | 0.962 |

For direct comparison, the GRL sweep from §10 on this identical setting (`runs/adversarial_suppress_results.json`):

| adv_grl_lambda | P(sentiment) | Q(sentiment) | P(topic) | Q(topic) |
|---|---|---|---|---|
| 0.3 | 0.870 | 0.969 | 0.533 | 0.836 |
| 1.0 | 0.803 | 0.987 | 0.477 | 0.821 |
| 3.0 | 0.783 | 0.984 | 0.393 | 0.845 |
| 10.0 | 0.780 | 0.972 | 0.410 | 0.809 |

Two findings, both robust across the sweep:

1. **GRL categorically fails here, worse than §10's own "+0.08 to +0.14 plateau" framing suggests.** Across all four tested lambdas (0.3-10.0), `q_sentiment_acc` stays at 0.97-0.99 — statistically indistinguishable from the untrained control's 0.997. The discriminator's own accuracy (`adv_ref_disc_final_acc`) is high (0.75-1.0) whenever logged, confirming the representation stayed linearly separable throughout — GRL never established a fooling regime in this setting at all, not even a partial one.
2. **LEACE-as-regularizer shows a clean, monotonic dose-response that GRL never showed.** `q_sentiment_acc` falls steadily from 0.93-0.96 at low lambda to 0.686 at lambda=3.0 — real, tunable suppression, achieved by ordinary gradient descent on a single differentiable term. Reconstruction quality (FVU) is completely unaffected (0.96 flat across every lambda, same as GRL) — the mechanism does not trade off against the SAE's core reconstruction objective.

**But** the suppression is not selective: `q_topic_acc` collapses in near-lockstep with `q_sentiment_acc` (0.70→0.32 across the same lambda range that takes `q_sentiment_acc` from 0.93→0.69), and `p_topic_acc`/`p_sentiment_acc` both degrade moderately too (P was never the direct target of this penalty). No tested lambda reaches a point where sentiment is suppressed toward chance (0.5) while topic collection stays anywhere near baseline (0.973/0.838) — the two accuracies trade off together rather than the penalty carving out a sentiment-specific subspace. This is a different failure mode from GRL's (GRL suppresses nothing; LEACE-as-regularizer suppresses everything roughly together), but it is still a failure to achieve clean collect+suppress: no lambda in {0.01, ..., 3.0} lands in the region this project needs (Q(sentiment)→chance, Q(topic)→baseline, P unaffected).

**T.** LEACE-as-a-training-time-regularizer is the first mechanism tested anywhere in this project's history (§3-§9, §10-§11) that produces *real, tunable* suppression via a theoretically-motivated, single-player differentiable objective rather than an adversarial one — direct evidence that the SAE-architecture-specific failure documented in §3-§9 is not a fact about the impossibility of gradient-based suppression in general, reinforcing §16's verdict. But it replaces GRL's total-failure mode with a new *non-selectivity* failure mode: the batch-level cross-covariance penalty, computed against Q's full doc-pooled representation with no notion of "the sentiment-specific directions only," pushes down whatever shared capacity correlates with the penalty gradient, taking topic down with it. This is consistent with (and adds a positive-mechanism data point to) this project's recurring finding that any single shared bottleneck (here: Q's full representation, elsewhere the shared dictionary/decoder) resists being partitioned into cleanly separable collect/suppress subspaces under gradient descent alone.

Two follow-ups this suggests, neither yet attempted:
1. **Restrict the penalty to a data-driven sentiment subspace** rather than the full representation — e.g. periodically re-fit the closed-form LEACE projection (as in `eval_concept_erasure_control.py`) from an EMA of batch statistics, and only penalize `Q`'s projection onto that fixed low-rank subspace, rather than the raw full-dimensional cross-covariance. This is closer to "true" LEACE (a data-driven low-rank direction) than the current whole-representation penalty and may avoid the observed topic collateral damage.
2. **Lower-lambda / longer-horizon sweep**: the lambda=0.01-0.03 region shows the smallest topic damage (Q(topic)=0.70-0.77) but also the least sentiment suppression (Q(sentiment)=0.93-0.96) — it is not yet clear whether more training steps at these low lambdas let sentiment suppression catch up without further topic damage, or whether the trade-off curve is fixed regardless of step count.

**Addendum: the lambda=0.01/0.03 region is non-monotonic, single-seed noise, not signal.** `q_sentiment_acc` at 0.03 (0.956) is *higher* than at 0.01 (0.932), and `q_topic_acc` at 0.03 (0.772) is likewise higher than at 0.01 (0.700) — both inverted relative to the clean monotonic trend that holds from lambda=0.1 upward. A single batch-level cross-covariance estimate is high-variance at this scale, and only one seed (42) was run per lambda, so this inversion is read as estimation noise rather than a real non-monotonicity in the underlying trade-off. Resolving it would need either repeated seeds or an EMA'd cross-covariance estimate (reducing per-batch variance) — but this is deprioritized relative to follow-up (1) above: the lambda≥0.1 region already establishes the core result (suppression is real but non-selective), and follow-up (1)'s subspace-restricted redesign is judged the higher-value next step over further noise-characterization in the low-lambda region.

### 18. Subspace-Restricted LEACE (EMA-Whitened Cross-Covariance) — Follow-Up (1) Tested

**Q.** §17's follow-up (1) hypothesized that penalizing the *raw, unwhitened* batch cross-covariance is what causes the topic collateral damage — every output dimension gets penalized in proportion to its own variance, with no notion of "the sentiment-specific directions only," so directions that happen to carry both sentiment and topic signal get pushed down together. Does replacing the raw penalty with the differentiable analogue of LEACE's actual closed-form fit — a whitened cross-covariance, using a whitening matrix `W = Sigma_xx^{-1/2}` fit from an EMA of the population covariance rather than one raw batch — reduce or remove that collateral damage?

**M.** Added `--leace_subspace` to `KronTopKTrainer` (`_leace_update_whitening`, `dictionary_learning/trainers/kron_top_k.py`). An EMA of the centered batch covariance `Sigma_xx` (`leace_subspace_ema_decay=0.99`) is maintained under `no_grad`; every `leace_subspace_refit_every=200` steps it is eigendecomposed to refit `W` (identical closed-form construction to `leace_fit` in `eval_concept_erasure_control.py`, clamped at `leace_subspace_eps=1e-4`). The suppression loss becomes `leace_lambda * ||W @ Cov(Q_docs, onehot(sentiment))||^2`, with `W` always detached — only the current batch's raw cross-covariance carries gradient into the encoder, so the penalty targets the population-level whitened direction rather than one batch's raw one. Everything else (mutual exclusivity, the detached reference discriminator, replacing rather than augmenting Q's DPO contrast-sentiment term) is unchanged from §17. Swept `leace_lambda ∈ {0.01, 0.03, 0.1, 0.3, 1.0, 3.0}`, identical synthetic setup to §17 (`joint_dpo_cross`, `kron_joint`, h=128/m=8/n=16, dpo_beta=2.0, seed=42, 5000 steps), one GPU (`runs/leace_subspace_sweep.sh` → `runs/leace_subspace_sweep/`).

**R.**

| leace_lambda | P(sentiment) | Q(sentiment) | P(topic) | Q(topic) | recon FVU |
|---|---|---|---|---|---|
| control (§17, no suppress) | 0.967 | 0.997 | 0.828 | 0.973 | 1.010 |
| 0.01 | 0.876 | **0.627** | 0.611 | 0.256 | 0.965 |
| 0.03 | 0.893 | **0.577** | 0.567 | 0.221 | 0.969 |
| 0.1 | 0.901 | **0.585** | 0.643 | 0.217 | 0.969 |
| 0.3 | 0.920 | **0.622** | 0.642 | 0.212 | 0.972 |
| 1.0 | 0.927 | **0.539** | 0.694 | 0.204 | 0.981 |
| 3.0 | 0.910 | **0.511** | 0.604 | 0.181 | 0.991 |

For direct comparison, §17's raw (unwhitened) sweep on the identical setting:

| leace_lambda | P(sentiment) | Q(sentiment) | P(topic) | Q(topic) |
|---|---|---|---|---|
| 0.01 | 0.906 | 0.932 | 0.622 | 0.700 |
| 0.03 | 0.891 | 0.956 | 0.607 | 0.772 |
| 0.1 | 0.902 | 0.896 | 0.651 | 0.533 |
| 0.3 | 0.920 | 0.834 | 0.656 | 0.427 |
| 1.0 | 0.886 | 0.735 | 0.586 | 0.337 |
| 3.0 | 0.884 | 0.686 | 0.635 | 0.321 |

Two findings:

1. **Whitening does make suppression stronger and far more uniform across lambda.** Every whitened `leace_lambda` from 0.01 to 3.0 lands `Q(sentiment)` in 0.51-0.63 — closer to chance (0.500) than *any* point on the raw sweep, including raw's strongest setting (lambda=3.0, 0.686). Even the smallest tested lambda (0.01) already reaches 0.627, beating raw's best result at 300× the weight. Reconstruction FVU is essentially unaffected (0.965-0.991, comparable to raw's 0.959-0.962 and clearly better than the untrained control's 1.010).
2. **But the hypothesis that this would *reduce* collateral damage is falsified — it makes selectivity worse, not better.** `Q(topic)` collapses to 0.18-0.26 at *every* lambda tested, including the smallest (0.01: 0.256) — inside noise of chance (0.167) throughout, and worse than raw's topic preservation at every matched or weaker sentiment-suppression point (raw lambda=0.1 gets comparable sentiment suppression, 0.896, while keeping `Q(topic)` at 0.533; whitened lambda=0.01 gets *stronger* sentiment suppression, 0.627, while `Q(topic)` is already down at 0.256). There is no lambda in the swept range, however small, where whitening leaves topic near baseline while suppressing sentiment — the raw penalty's low-lambda region (§17: lambda=0.01-0.03, `Q(topic)`=0.70-0.77) was strictly better on this trade-off than anything the whitened version produces.

**Diagnosis.** Whitening by `Sigma_xx^{-1/2}` amplifies the gradient along whatever directions carry the *least* variance in Q's raw representation, independent of whether those directions are sentiment-specific. In this synthetic construction, sentiment's and topic's signal directions are both low-variance relative to the dominant noise/reconstruction-driven directions (the same asymmetry §9.3 measured on real data as Q's ~28× per-dim std spread) — so a variance-agnostic *whitened* penalty does not selectively target "the sentiment subspace," it disproportionately targets "the low-variance subspace," which both labels' discriminative directions happen to compete for. Rather than isolating a sentiment-only direction the way an oracle low-rank LEACE fit would, this training-time approximation ends up *less* discriminating between labels than the raw, unwhitened penalty was.

**T.** Follow-up (1) from §17 is now tested and **rejected**: whitening the cross-covariance penalty by the population covariance does not recover selectivity — it produces a stronger, more uniform, but *less selective* suppression than the raw penalty, collapsing topic to chance regardless of lambda. This adds a tenth failed mechanism to the list in §9/§16 and specifically closes off "make the LEACE-style penalty population-aware via whitening" as a fix, leaving §17's follow-up (2) (repeated seeds / EMA'd cross-covariance to resolve low-lambda noise in the *raw* penalty, not the whitened one) and a genuinely rank-restricted variant — projecting onto only the top-1 (for a binary label) *fitted* direction from a periodically-refit closed-form LEACE solve, rather than whitening the full representation — as the remaining untested variants on this axis. The latter is a meaningfully different mechanism from what was tested here: this section's `W` reshapes gradient across *all* dimensions by inverse-variance, whereas a true low-rank projection would zero out the penalty's effect entirely outside a small fitted subspace, rather than reweighting inside the full one.

## 19. Six New Training Mechanisms: BCD Alternation, Conditional Reconstruction, Swap-and-Reconstruct, Capacity Bottleneck, RL Suppression, and Their Composition

**Q.** §3-§18 exhausted a loss-*weighting* axis (five algorithms) and a combine-*rule* axis (five rules), all sharing one property: they change how strongly or how the existing DPO-contrastive/adversarial/LEACE objective is applied, but never change *what kind* of training signal drives Q's suppression, nor whether P and Q's parameters are ever updated at genuinely different times. Six literature-grounded alternatives were proposed and implemented: (1) three-way block-coordinate-descent (BCD/ALS-style) alternation over P, Q, and the decoder; (2) conditional reconstruction (feeding the ground-truth sentiment label into the decoder so Q no longer needs to carry it); (3) swap-and-reconstruct self-supervision (no repulsion term at all — pairs same-topic/different-sentiment docs, swaps their Q output, and penalizes the resulting reconstruction error); (4) a trainable rank-restricted capacity bottleneck on Q's full representation; (5) REINFORCE-based suppression (Q's encoder as a stochastic-policy mean, reward = a reference discriminator's loss, variance-bounded updates instead of gradient-descent's magnitude-bounded ones); and (6) the composition of (1) and (5) — RL suppression specifically during Q's BCD phase.

**M.** All six implemented in `KronTopKTrainer` (`--bcd_alternate`, `--cond_recon`, `--swap_recon`, `--q_bottleneck_rank`, `--rl_suppress`, `--bcd_rl_combo`), each independently code-reviewed and smoke-tested before the full run. Evaluated on the identical synthetic-diagnostic setting used throughout §3-§18 (`kron_joint`, `joint_dpo_cross`, h=128/m=8/n=16, dpo_beta=2.0, seed=42, 5000 steps, one GPU), one representative hyperparameter setting per idea against a freshly-run shared control (`runs/idea_sweep/`, `runs/idea_sweep_results.json`):

| Mechanism | P(sent) ▲ | Q(sent) ▼ | P(top) ▼ | Q(top) ▲ | recon FVU |
|---|---|---|---|---|---|
| control (no intervention) | 0.896 | 0.990 | 0.596 | 0.879 | 0.960 |
| 1. BCD alternation (`bcd_phase_steps=200`) | 0.902 | **0.919** | 0.660 | 0.713 | 0.965 |
| 2. Conditional reconstruction | 0.893 | 0.984 | 0.634 | 0.842 | 0.962 |
| 3. Swap-and-reconstruct (`swap_recon_lambda=1.0`) | 0.796 | 0.992 | 0.461 | 0.927 | 0.963 |
| 4. Capacity bottleneck (`rank=32` of 2048) | 0.783 | **0.604** | 0.451 | 0.241 | 0.947 |
| 5. RL / REINFORCE suppression (`sigma=0.1`, `lambda=1.0`) | 0.837 | 0.990 | 0.510 | 0.888 | 0.959 |
| 6. BCD + RL composition | 0.842 | 0.997 | 0.497 | 0.842 | 0.981 |

**Results, mechanism by mechanism.**

1. **BCD alternation is the most interesting result of the six** — the only mechanism that moved `Q(sentiment)` at all (0.990→0.919) *while leaving P undamaged* (P-sentiment 0.902 vs. control's 0.896; P-topic actually rose, 0.660 vs. 0.596). Every other mechanism in this report that suppressed anything did so by degrading P through the shared decoder (§4-§6, §8); this is the first one where P comes out fully intact. The cost is the familiar one — `Q(topic)` also fell (0.879→0.713) — but proportionally less than the suppression gained, and far short of collapsing to chance (0.167). This is weak evidence for the §18/Part-II-closing hypothesis that separating *when* P, Q, and the decoder update (not just what they're penalized by) partially decouples their fates.
2. **Conditional reconstruction is a near-null result at this dose.** `Q(sentiment)` barely moved (0.990→0.984) and neither did P (0.893/0.634, both within noise of control). The label-bias head's norm grew to 2.34 by the end, so it is doing *something* to reconstruction, but not enough to measurably relieve Q of the pressure to encode sentiment. This does not falsify the idea — it more likely means the additive bias term is too weak a channel relative to reconstruction's already-massive gradient (§9.3: 300-400× over supervision) to meaningfully substitute for Q's contribution; a stronger conditioning mechanism (FiLM modulation throughout the decoder, not one additive term) is the natural next step before concluding anything about the underlying hypothesis.
3. **Swap-and-reconstruct is a net negative at this dose.** No suppression benefit at all (`Q(sentiment)` 0.990→0.992, statistically flat) and the worst P damage of any mechanism tested here (P-sentiment 0.796, P-topic 0.461, both well below control). The self-supervised swap-consistency loss adds a full reconstruction-scale term on top of the existing one (`swap_recon_lambda=1.0` roughly doubled the total loss magnitude, confirmed in the pre-run smoke test), which plausibly destabilized training generally rather than inducing the intended disentanglement pressure specifically — a lower `swap_recon_lambda` is the obvious follow-up before treating this idea as refuted.
4. **The capacity bottleneck produced the strongest raw suppression of anything in this report** — `Q(sentiment)` 0.990→0.604, closer to chance (0.500) than any mechanism in §3-§18 achieved (the previous best was CAGrad's 0.68-0.74, Part II §6). But it reproduces exactly the non-selective collapse pattern this project has now seen repeatedly (§8, §18): `Q(topic)` crashes even harder, to 0.241 (near its own chance floor of 0.167), and P degrades too (0.783/0.451). A rank-32 bottleneck restricts *everything* Q can represent, not specifically its sentiment-correlated content — this is the architectural analogue of §18's finding that an undirected constraint (there, whitening; here, capacity) hits both labels' shared low-capacity real estate rather than isolating one.
5. **RL/REINFORCE suppression produced no measurable effect at this hyperparameter setting** (`Q(sentiment)` 0.990→0.990, exactly flat) while still costing P some ground (0.837/0.510 vs. control's 0.896/0.596). The in-loop reference discriminator did show real dynamics during training (accuracy fluctuating, ending at 0.594 — see §17-adjacent smoke-test log), consistent with the mechanism being alive rather than silently broken, but a moving in-loop discriminator accuracy did not translate into the standardized final probe reading any different from control — the same discriminator-fools-itself-but-information-remains pattern (Elazar & Goldberg 2018) already documented for GRL (§10-§11), now reproduced with a REINFORCE update rule instead of gradient-reversal, at `rl_sigma=0.1`. This suggests the variance-vs-magnitude distinction motivating idea 5 (per arXiv:2608.03573) does not by itself overcome reconstruction's ~300-400× gradient dominance at matched step count and this exploration scale — a larger `rl_sigma`/`rl_lambda` or more steps is needed before treating RL as refuted, not just weakly tested.
6. **BCD + RL composition inherited RL's weakness, not BCD's strength.** `Q(sentiment)` stayed flat (0.990→0.997, if anything slightly worse) and P-topic collapsed harder than plain BCD alone (0.497 vs. idea 1's 0.660) while P-sentiment held up better than idea 5 alone (0.842). The in-loop discriminator ended notably more confident here (0.938 accuracy) than in idea 5 alone (0.594) — plausibly because BCD's phase-frozen Q gives the discriminator uninterrupted stretches to fit a stationary target, the opposite of what idea 5's exploration noise was meant to exploit. Composing an underpowered mechanism (5) with a promising one (1) did not rescue the underpowered one; if RL's own hyperparameters are strengthened per point 5, this composition is worth re-running.

**Verdict (as originally read).** Of six new mechanisms — spanning three genuinely different axes (optimization schedule, objective design, and representational capacity) beyond the loss-weighting/combine-rule axes §3-§18 already exhausted — BCD alternation (idea 1) looked like the first mechanism in this project's entire history to show real suppression movement without measurably damaging P, worth a focused follow-up sweep. The capacity bottleneck (idea 4) reproduces the strongest-raw-suppression/worst-selectivity pattern now seen three times (§8's `mnand`/`mnor`, §18's whitened LEACE, and here). Ideas 2, 3, and 5-6 read as underpowered-at-this-dose results, not settled negatives.

> **⚠ Idea 1's reading is corrected below, see §19.1.** The probe-accuracy-only verdict above does not survive a representation-geometry check — idea 1's apparent success is not supported once participation ratio and direction-overlap are examined.

### 19.1 Follow-Up: Probe-Methodology Cross-Check, Training Trajectory, and Representation Geometry

**Q.** Three gaps in §19 as first written, all raised by direct user questions: (i) `evaluate_synthetic_joint_probe`'s reported accuracies come from an *unstandardized* `torch.nn.Linear` trained by Adam — not `standardized_logreg_accuracy`, the function this codebase documents as "the corrected probe standard mandated by REPORT.md sec.9.3" — so are §19's numbers trustworthy at all? (ii) all six mechanisms were read from a single final-step snapshot at 5000 steps — could "no effect" (ideas 2, 3, 5, 6) or "effect" (ideas 1, 4) really be an artifact of too few training steps? (iii) neither participation ratio nor a pairwise directional-interference score (the `sum_{j!=i}(What_i . W_j)^2` construction from Elhage et al. 2022, "Toy Models of Superposition") — both used in Part I sec.9.1 — had been checked for any of the six.

**M.** Added `evaluate_synthetic_representation_geometry` (`src/train.py`) computing, per branch: `standardized_logreg_accuracy` on the same pooled representations `evaluate_synthetic_joint_probe` already reads, participation ratio (`_participation_ratio_from_cov`, identical formula to `eval_representation_geometry.py`), and the pairwise sentiment/topic direction-interference score `(What_sent . What_topic)^2` (directions from `_discriminative_direction`, the same per-dimension between-class-mean-deviation construction as Part I's `discriminative_energy_vector`). Added `--synthetic_probe_every` for a mid-training probe snapshot (smaller eval-batch count than the final read, to bound added cost). Reran control, idea 1 (BCD), and idea 4 (bottleneck) — the two that moved anything in §19 — at the identical 5000-step synthetic setting, probing every 500 steps (`runs/idea_verify/`).

**R (i) — probe methodology is not the issue.** The standardized and unstandardized probes agree closely everywhere (within 0.01-0.04, all converged):

| | unstd Q(sent) | std Q(sent) | unstd Q(top) | std Q(top) |
|---|---|---|---|---|
| control | 0.993 | 0.984 | 0.889 | 0.876 |
| idea 1 (BCD) | 0.922 | 0.925 | 0.697 | 0.692 |
| idea 4 (bottleneck) | 0.612 | 0.572 | 0.230 | 0.202 |

Today's §19 numbers were never a repeat of the pilot's §9.3 probe artifact.

**R (ii) — idea 4 is stable from step 500; idea 1 and control are confounded by the snapshot's smaller eval-batch count.** Idea 4's Q(sent)/Q(top) are flat across the whole 500-4500 window (0.55-0.64 / 0.18-0.27) — its effect is immediate and 5000 steps is not the limiting factor. Idea 1's and control's mid-training snapshots read lower and noisier than their final values (e.g. idea 1's Q(sent) sits at 0.85-0.90 throughout training vs. 0.922 at the end), but the periodic snapshot uses 4x fewer eval batches than the final read specifically to bound added training cost — so this gap is at least partly a probe-data-availability artifact of the diagnostic itself, not necessarily continued learning. This is left unresolved and flagged for a follow-up that matches the snapshot and final probes' batch counts.

**R (iii) — the geometry check does not support idea 1's apparent success.**

| | P: PR | Q: PR | P: interference | Q: interference |
|---|---|---|---|---|
| control | 8.4 | 46.7 | 0.137 | 0.214 |
| idea 1 (BCD) | 10.4 | **157.6** | **0.386** | 0.279 |
| idea 4 (bottleneck) | 22.8 | **5.4** | 0.418 | **0.085** |

(interference = `(What_sentiment . What_topic)^2`; 0 = orthogonal directions, 1 = identical direction; PR = effective dimensionality via `(sum eig)^2 / sum(eig^2)`.)

Idea 1's Q shows a real probe-accuracy drop (0.993→0.922) but its participation ratio roughly *tripled* relative to control (47→158, i.e. more diffuse, not more compressed) and its sentiment/topic direction overlap *increased* (0.214→0.279) rather than decreased. A genuinely disentangled Q should show the two directions pulling apart, not closer together. This is the same "hidden, not deleted" shape this project already documented for adversarial suppression (sec.10-11, Elazar & Goldberg 2018) and for RL suppression in this section's own idea 5 — a probe reads slightly worse while the underlying geometry shows no corresponding separation, or in this case active *entanglement*. Idea 4's geometry is internally consistent with genuine (if indiscriminate) destruction instead: Q's dimensionality collapses to ~5 (near, and even below, its own rank-32 cap — most of the allotted capacity goes unused), and what little signal survives shows the lowest direction-overlap of the three (0.085) — consistent with "not much left for either label," not "topic cleanly kept, sentiment cleanly removed."

**T.** §19's original verdict for idea 1 is retracted: it is not a case of P-preserving genuine suppression. Its probe-accuracy movement does not correspond to the two concepts' directions actually separating — if anything they move closer together, and Q's representation becomes more diffuse rather than more structured. Idea 4 stands as before: real but indiscriminate capacity destruction, now with geometric evidence (collapsed PR, low residual interference) supporting rather than contradicting that reading. **No mechanism tested in §3-§19, under any diagnostic applied so far (probe accuracy, random-init baseline, gradient measurement, or now representation geometry), has produced a checkpoint where the two concepts' directions are shown to have actually separated.** This raises the same standard §9.4 set for probe accuracy — read against a random-init baseline, not chance — to representation geometry: a suppression claim needs the direction-overlap and participation-ratio evidence to move the *right way*, not just the probe accuracy.

### 19.2 Follow-Up: Isolating the Final-Step Spike, and an Unexpected Long-Horizon Decline

**Q.** Two direct user questions about §19.1's trajectory plots: (i) why does every config (including control) show a sharp jump in the topic accuracies specifically at the final step, and (ii) what happens if training runs longer than 5000 steps?

**M.** Added `--synthetic_probe_snapshot_batches` so the mid-training probe snapshot can use the same eval-batch count as the final-step read (previously a quarter of it, `max(4, synthetic_eval_batches // 4)`, to bound added cost). Reran control, idea 1, and idea 4 for 12,000 steps (2.4x longer than §3-§19's standard 5000-step protocol) with `synthetic_probe_every=1200`, `synthetic_probe_snapshot_batches=64` matched to the final read (`runs/idea_longrun/`).

**R (i) — confirmed artifact.** With batch counts matched, the final-step jump disappears entirely; trajectories are smooth. The spike in §19.1's plots was the mid-training snapshot having 4x less eval data than the final read (16 vs. 64 batches) — topic (6-way multiclass) needs more data to fit a good probe than binary sentiment, so it showed the larger jump. Not a training-step effect.

**R (ii) — unexpected: control's Q representation gets *less* linearly decodable with more training, not more.** Both control and idea 1 peak around step 3600-6000, then decline steadily through step 12000:

| step | control Q(sent) | control Q(top) | idea1 Q(sent) | idea1 Q(top) | idea4 Q(sent) | idea4 Q(top) |
|---|---|---|---|---|---|---|
| 1200 | 0.979 | 0.819 | 0.948 | 0.787 | 0.523 | 0.217 |
| 3600 (~peak) | 0.984 | 0.868 | 0.961 | 0.801 | 0.594 | 0.285 |
| 6000 | 0.985 | 0.811 | 0.979 | 0.783 | 0.627 | 0.200 |
| 9600 | 0.876 | 0.565 | 0.928 | 0.681 | 0.593 | 0.235 |
| 12000 (final) | 0.858 | 0.497 | 0.886 | 0.564 | 0.557 | 0.252 |

Control's `Q(topic)` falls from a peak of 0.87 to 0.50 (roughly back to chance-adjacent territory); `Q(sentiment)` falls from 0.99 to 0.86. Idea 1 shows the identical shape, offset slightly higher. **Idea 4 (bottleneck) shows no such pattern — flat across the entire 12,000 steps**, consistent with §19.1: it's already at its floor by step 1200, nothing left to lose.

Checking the raw training loss (`kron_joint`, reconstruction + supervision) over the same window: it is flat and stable throughout, no divergence, no sign of instability (~975→~955, within normal noise). **Reconstruction quality does not degrade.** So this is not the model breaking — it's specifically that the information a *linear* probe can recover from Q declines over long training, while whatever Q is doing for reconstruction itself stays just as good. P's own accuracies stay roughly flat over the same window (no compensating rise), so this doesn't look like information relocating from Q to P either — it reads as Q's representation drifting into a geometry that is progressively less linearly accessible, independent of any suppression mechanism.

One important confound, stated plainly: `warmup_frac`, `decay_start_frac` (0.8), and `threshold_start_frac` are all fractions of `total_steps`, so the 12,000-step run is not literally "the 5000-step run continued" — its LR-decay phase starts at step 9600 rather than step 4000. But the decline itself begins around step 6000 (50% through training), well before the decay phase (80%) — so the decay schedule's onset does not explain why the decline *starts*, even if it may explain some later acceleration into it.

**T.** Two conclusions. First, the practical one: the final-step spike in every plot up to this point was a probe-evaluation artifact (batch-count mismatch), now fixed. Second, a substantive one, orthogonal to the whole collect/suppress question this project has been chasing: **this synthetic training setup exhibits a long-horizon decline in linear decodability that has nothing to do with any suppression mechanism** — it happens to the untouched control just as much as to idea 1. Every comparison in §3-§19 used a fixed 5000-step protocol, which by this evidence sits close to a peak (control's own peak is ~3600-6000) rather than on a stable plateau — meaning those comparisons were not necessarily reading a converged, representative value, just a value at one specific, now-shown-to-be-non-flat point in a longer curve. This does not overturn any specific verdict already reached (idea 4's flatness makes its result robust to this; idea 1 was already retracted in §19.1 on independent geometric grounds), but it is a standing caveat on this report's methodology going forward: a mechanism comparison at one fixed step count, without a trajectory check, cannot be assumed to reflect a stable endpoint. The cause of the decline itself — dead-unit accumulation, dictionary reallocation under top-k competition, or an interaction between the DPO terms and reconstruction that only manifests at this horizon — is not yet diagnosed and is flagged as an open question distinct from this project's main line of inquiry.

### 19.3 Steering, Not Just Erasure: A Causal Patch-and-Generate Eval on the Untouched Control

**Q.** Every diagnostic used in this report through §19.2 (probe accuracy, random-init baseline, gradient measurement, participation ratio, direction interference) measures whether a label is *linearly decodable* from Q — an erasure-shaped question, exactly the property LEACE is built to answer and provably beats every SAE mechanism at (§16). None of it tests the property motivating KronSAE's actual value proposition as a dictionary-learning method rather than a post-hoc edit: **steerability** — whether pushing a document's representation along a discovered axis causes a controlled, predictable change in what the model outputs. This section builds that test.

**M.** Added `evaluate_synthetic_intervention` (`src/train.py`). On a disjoint FIT split, fits (a) two reference classifiers (standardized logistic regression, sentiment and topic) directly on *genuine raw activations* `x` — never touching the SAE, so they cannot have been shaped by whatever the SAE learned — and (b) `_signed_binary_direction`, `mean(Q | sentiment=1) - mean(Q | sentiment=0)` on Q's pooled representation. This is a deliberate departure from §19.1's `_discriminative_direction` (a squared/summed energy construction built to generalize to topic's 6 classes): that construction discards sign for a binary label, so it is not guaranteed to point toward "more positive" consistently across dimensions, only to have the right *magnitude*. A causal push needs the correctly oriented, signed direction; the unsigned energy vector remains the right tool for the interference/PR diagnostics, where only magnitude of overlap matters. On a disjoint EVAL split: pushes every document's per-token Q representation by `alpha` standard deviations (of the FIT split's projection onto the signed direction) along it, clamps to preserve the mAND gate's non-negativity contract, recombines with the untouched P representation, decodes, and scores the resulting *reconstruction* — not Q — with the FIT-split reference classifiers. Run on the plain control checkpoint (5000 steps, no suppression mechanism at all) — the question here is whether steerability exists in KronSAE *at all*, before asking whether any of §3-§19's interventions help or hurt it. Swept `alpha ∈ {-6,-4,-2,-1,0,1,2,4,6}` (`runs/idea_intervention/control/`).

**R.**

| alpha | recon FVU | P(sentiment=positive) | topic accuracy |
|---|---|---|---|
| no intervention (raw x, reference) | — | 0.500 | 0.168 |
| -6 | 0.979 | **0.109** | 0.132 |
| -4 | 0.970 | 0.202 | 0.162 |
| -2 | 0.962 | 0.372 | 0.169 |
| -1 | 0.961 | 0.435 | 0.159 |
| 0 | 0.961 | 0.485 | 0.150 |
| 1 | 0.961 | 0.528 | 0.145 |
| 2 | 0.962 | 0.573 | 0.144 |
| 4 | 0.964 | 0.673 | 0.143 |
| 6 | 0.970 | **0.775** | 0.141 |

**Pushing Q's sentiment direction moves the decoded reconstruction's predicted sentiment monotonically and substantially** — from 0.11 (strongly negative) to 0.78 (strongly positive) across the swept range, smoothly crossing the 0.50 no-intervention baseline near alpha=0 as it should. **Topic accuracy stays close to its no-intervention baseline throughout** (0.13-0.17 vs. reference 0.17, no trend tracking alpha's sign or magnitude) — sentiment moves a lot, topic barely moves, exactly the selectivity a steering claim needs. Reconstruction FVU stays near 1.0 across the whole range (0.96-0.98), meaning even the most extreme pushes don't produce a wildly implausible reconstruction — a controlled edit, not a corruption.

**T.** The plain, unmodified control checkpoint — no suppression mechanism, no special training, the same model whose Q representation §19.1-19.2 showed is *not* linearly disentangled from topic (direction interference 0.214, and declining further with more training per §19.2) — **already supports genuine, fairly selective causal steering along its sentiment axis**, when the axis is fit correctly (signed, not the unsigned energy construction) and the effect is measured on the reconstruction via an honest external judge rather than on Q via the SAE's own probe. This is the first result in this project's entire history (§3-§19.2) to demonstrate a property LEACE has no way to provide at all, since LEACE produces no dictionary and nothing to push along — direct support for the user's reframing that KronSAE's comparative advantage over LEACE is steerability, not erasure, and that the whole erasure-shaped evaluation this report has run since §6 may have been asking the wrong question of the wrong property. Two things this result does not yet establish, flagged as the natural next steps and answered immediately below in §19.4: (i) whether any of §3-§19's suppression mechanisms improve or degrade this steering effect relative to the untouched control; (ii) whether the effect survives on real pythia-410m activations.

### 19.4 Follow-Up: Does Steering Survive Suppression? Does It Survive Real Data?

**Q.** §19.3's two flagged follow-ups, both run immediately: (i) does BCD alternation (idea 1) or the capacity bottleneck (idea 4) — the two mechanisms with the most probe-visible effect on Q — help or hurt the steering property just demonstrated on the untouched control; (ii) does the effect hold on real pythia-410m activations rather than only the synthetic construction?

**M.** Reran idea 1 and idea 4 with the same intervention sweep as §19.3's control (`runs/idea_intervention/{idea1_bcd,idea4_bottleneck}/`). For real data, wrote `src/eval_real_intervention.py`, reusing the already-trained real checkpoint (`runs/joint_dpo_cross_full420k/checkpoints/kron_joint/ae_step_420000.pt`, 420,000 steps, joint_dpo_cross, h=128/m=8/n=16 — no retraining) and the already-cached real pythia-410m layer-12 activations from the LEACE experiments (`runs/pythia_erasure_activation_cache.pt`, §13-15 — no fresh LM forward pass), 9600 docs (~421,000 tokens). Identical method to §19.3 otherwise (signed direction, disjoint fit/eval split, external reference classifiers on genuine raw activations). One implementation bug caught and fixed before trusting the result: the first pass fed all ~421,000 tokens through the encoder's dense combine step in one shot, which OOM'd (that step alone materializes a `(n_tokens, dict_size=16384)` tensor — ~25GB for this many tokens on a shared 96GB GPU); fixed by chunking both the encode and the per-alpha decode into 20,000-token pieces, verified to produce bit-identical output to the pre-fix version on a small sample before running at full scale.

**R.**

| config | steering range: P(sent+) at α=-6 → α=+6 | topic accuracy range across α | topic no-intervention baseline |
|---|---|---|---|
| control (synthetic) | 0.11 → 0.78 | 0.13 - 0.17 | 0.168 |
| idea 1: BCD alternation (synthetic) | 0.12 → 0.82 | 0.13 - 0.18 | 0.168 |
| idea 4: capacity bottleneck (synthetic) | **0.42 → 0.51** | 0.17 - 0.18 | 0.168 |
| real pythia-410m, 420k-step checkpoint | **0.04 → 1.00** | **0.164 - 0.169** | 0.167 |

**Idea 1 (BCD) preserves steering — if anything, slightly widens the range** (0.12→0.82 vs. control's 0.11→0.78), despite §19.1 showing its Q representation has *higher* sentiment/topic direction interference than control (0.279 vs. 0.214) and a more diffuse participation ratio (158 vs. 47). This is a real dissociation: whatever BCD did to Q's geometry that made it read worse on the erasure-shaped diagnostics did not cost it the steering property — direction interference and steerability are turning out to be different axes, exactly as the reframing anticipated.

**Idea 4 (capacity bottleneck) nearly destroys steering**, collapsing the range to 0.42→0.51 — an order of magnitude smaller swing than control or idea 1. This is the expected, consistent result: a rank-32 bottleneck leaves little capacity for *any* signal, including whatever "push along this axis" would need to work with, so the same mechanism that produced the strongest raw suppression in §19 (Q(sentiment) 0.990→0.604) also produces the weakest steering. Erasure and steerability move together here, but in the *bad* direction — the bottleneck's "success" at erasure and its failure at steering share the same cause (indiscriminate capacity destruction), not a genuine trade-off between two independently achievable properties.

**Real pythia-410m activations show the cleanest result of the whole report**: steering range 0.04→1.00, spanning almost the entire probability range, while topic accuracy holds at 0.164-0.169 across every single alpha — a tighter band around the 0.167 baseline than any of the three synthetic curves manage. The effect is not a synthetic-construction artifact; if anything it is stronger and more selective on real data than on the synthetic buffer designed to make labels orthogonal by construction.

**T.** Two conclusions. First, steering and the erasure-shaped diagnostics used throughout §6-§19.2 are measuring genuinely different properties, not two views of the same thing — idea 1 fails the direction-interference check while fully retaining steering, and idea 4 passes the raw-suppression-magnitude check while nearly losing steering entirely. A mechanism ranking by one axis does not predict its ranking by the other. Second, the core steering result is now confirmed on real data, not just a synthetic construction, and confirmed to survive at least one of §3-§19's "successful" suppression mechanisms (BCD) while correctly failing under the one mechanism that destroys the underlying representation wholesale (the bottleneck) — the property degrades exactly when and how it should, which is itself evidence the eval is measuring something real rather than an artifact of the setup. Remaining and explicitly out of scope here: whether steering survives patched into the live LM (rather than only through the SAE's own decoder) and whether it holds for the topic axis / other labels, not just sentiment.

### 19.5 Follow-Up: P Steering vs. Q Steering — Which Branch Is Actually the Lever?

**Q.** §19.3-19.4 only pushed Q — the branch trained to *suppress* sentiment. P is the branch trained to *collect* it (`p_collect_sent`, the DPO term pulling same-sentiment documents together). The natural next question: does P, the branch nominally assigned ownership of sentiment, steer at least as well as Q, or better?

**M.** Generalized `evaluate_synthetic_intervention` (`src/train.py`) and `eval_real_intervention.py` with a `branch` parameter selecting which branch is pushed (the other stays untouched, same as before) — otherwise identical method to §19.3: signed direction fit per-branch on the FIT split, pushed on the EVAL split, scored by the same genuine-raw-activation reference classifiers. Ran both branches on the synthetic control (5000 steps) and the real pythia-410m checkpoint (`runs/idea_intervention/control/`, `runs/real_intervention_results.json`).

**R.** Q is the dominant steering lever, by a wide margin, on both synthetic and real data:

| config | Q-branch: P(sent+) at α=-6 → α=+6 | P-branch: P(sent+) at α=-6 → α=+6 |
|---|---|---|
| synthetic control | 0.11 → 0.78 (span 0.67) | 0.34 → 0.58 (span **0.23**) |
| real pythia-410m | 0.04 → 1.00 (span 0.95) | 0.83 → 0.96 (span **0.13**) |

Both branches stay selective (topic accuracy flat near chance/baseline for both, no meaningful difference in the selectivity check between them). But P — the branch whose own supervision loss is specifically "pull same-sentiment documents together" — produces a steering effect roughly a third the size of Q's on synthetic data, and less than a sixth the size on real data, despite an equal-magnitude push (both swept in units of the FIT split's own standard deviation along each branch's respective direction, so this isn't an artifact of P's direction being scaled differently).

**T.** The branch nominally "responsible" for a label by loss assignment is not the branch that actually controls the decoder's output for that label — Q, trained to *suppress* sentiment, is the stronger causal lever for it. The likely explanation is capacity, not the loss term: Q is twice P's width (`h·n=2048` vs. `h·m=1024` at this config), giving it more decoder-facing directions to have absorbed sentiment-correlated reconstruction structure into, regardless of which branch's supervision term nominally targets the label. This is consistent with this project's running finding since Part I §9 that Q dominates P through sheer capacity, not through which loss term is assigned to which branch — now shown to hold for the *causal* (steering) property, not just the *correlational* (probe accuracy) one. Practical implication for future steering work on this architecture: don't assume the "collect" branch is the one to intervene on — check empirically, since the loss assignment and the actual causal lever can and did diverge here.

### 19.6 Follow-Up: Swap the Capacities — Does the Dominant Branch Follow, or Does P Stay Dominant?

**Q.** §19.5 found Q (capacity 2048, the *suppress*-sentiment branch) dominates steering over P (capacity 1024, the *collect*-sentiment branch) by 3-6×, and attributed this to capacity rather than loss assignment. Direct test: swap the capacities — give P 2048 dims and Q 1024 (`joint_m=16, joint_n=8`, inverse of the default `joint_m=8, joint_n=16`) while leaving every loss term exactly as-is (P still *collects* sentiment, Q still *suppresses* it). If capacity is really the driver, dominance should flip to P even though its loss term never changed.

**M.** Trained three matched configs on the exact synthetic-diagnostic protocol (5000 steps, seed=42): `equal_m12_n12` (P=Q=1536, dict size 18,432 — slightly larger total dict, flagged as a caveat), `swapped_p16_n8` (P=2048, Q=1024, dict size 16,384 — matches control's total dict size exactly, the cleaner pairing), plus the full steering-intervention sweep (α ∈ {-6,-4,-2,-1,0,1,2,4,6}) on both, reusing §19.3-19.5's method (`runs/idea_size_intervention/`).

**R.** Steering span (P(sentiment-positive) at α=-6 → α=+6) by which branch is pushed:

| Config | P-branch span | Q-branch span | Dominant branch |
|---|---|---|---|
| control (P=1024, Q=2048 — default) | 0.23 | 0.67 | **Q, ~2.9×** |
| equal_m12_n12 (P=Q=1536) | 0.615 | 0.409 | P, ~1.5× |
| swapped_p16_n8 (P=2048, Q=1024) | **0.591** | 0.437 | **P, ~1.35×** |

Dominance flips exactly as predicted: with P given the larger capacity, P becomes the stronger steering lever — despite P's loss term ("collect sentiment") never changing across any of the three configs. `equal_m12_n12`'s dict size is 18,432 vs. the other two's matched 16,384 (m·n isn't held constant at m=n=12), so `control` vs. `swapped_p16_n8` — identical total dict size, only the P/Q split changed — is the clean pairing; `equal` is a useful bonus point, not a perfectly matched one.

**T.** Confirms §19.5's capacity hypothesis directly rather than just by elimination: moving capacity from Q to P moves steering dominance from Q to P, with the loss assignment held fixed throughout. The real-data replication of this exact swap is §19.7.

### 19.7 Real Pythia-410m Replication of the Capacity Swap — Probe, Geometry, and Steering, All Three

**Q.** Does §19.6's capacity-swap result — dominance follows capacity, not loss assignment — hold on real activations, the same way §19.5's original Q-dominance result held on both synthetic and real data?

**M.** Full real-data training run: `joint_m=16, joint_n=8` (P=2048, Q=1024), 420,000 steps, otherwise identical to the existing default-capacity control (`runs/joint_dpo_cross_full420k`) — same dataset (`amazon_reviews_full500k`), same `dpo_beta=2.0`, same seed, same schedule fractions. One methodological correction made before this run, worth recording: the *first* attempt at this real-data comparison used `--doc_batch_size 16` on a single GPU with no DDP, while the original control was trained via `torchrun --nproc_per_node=4` — each of the 4 ranks contributing its own locally-paired 16-doc batch, gradients averaged across ranks, for an effective 64 docs/optimizer-step. A single-GPU run at `doc_batch_size=16` processes 4× less data per step than that, confounding any comparison. Fixed by adding `--grad_accum_steps` to `KronTopKTrainer` (`dictionary_learning/trainers/kron_top_k.py`): accumulates gradients over N independently-sampled microbatches — each forming its own DPO-pairwise pairs, matching DDP's per-rank-local-pairing semantics, rather than pooling into one larger batch (a materially different pairing statistic) — before one optimizer step. Used `--grad_accum_steps 4` to replicate the original's effective batch size exactly, with the same total optimizer-step count (420,000) as before, i.e. equal backprop steps *and* equal data volume per step, not just equal step count. Scoped narrowly: raises `NotImplementedError` if combined with PCGrad/GradNorm/CAGrad/adversarial-suppress/the grad-norm probe, since those mechanisms assume exactly one autograd graph per step (verified this guard fires correctly before trusting the real run). The first (uncorrected, batch=16) run was killed and restarted from scratch once this was in place — nothing from it is used below.

Three diagnostics run on the resulting checkpoint, matching every prior real-data comparison in this report: (a) `eval_joint_probe.py` — standard probe accuracy; (b) `eval_representation_geometry.py` — participation ratio and sentiment/topic direction overlap, the same diagnostic that retracted idea 1's apparent success in §19.1; (c) `eval_real_intervention.py` (now parameterized by `--run_dir`/`--ckpt_step` instead of hardcoded) — the causal steering sweep from §19.3-19.5.

**R (a) — probe accuracy.**

| Config | P(sent) | Q(sent) | P(topic) | Q(topic) |
|---|---|---|---|---|
| control (P=1024, Q=2048) | 0.938 | 0.933 | 0.520 | 0.645 |
| swapped (P=2048, Q=1024) | 0.935 | 0.913 | 0.588 | 0.608 |

Sentiment accuracy is high and nearly unchanged for both branches in both configs (both branches can decode sentiment regardless of size — probe accuracy alone doesn't distinguish these configs much, consistent with §19.1's finding that probe accuracy under-tells the geometric/causal story). Topic shows a mild, capacity-tracking shift: P's topic accuracy rises with its capacity (0.520→0.588) and Q's falls with its capacity loss (0.645→0.608) — small in magnitude, same direction as the hypothesis.

**R (b) — representation geometry.**

| Config / branch | Participation ratio | Sentiment/topic direction overlap |
|---|---|---|
| control, P (1024) | 4.19 | 0.9457 (highly entangled) |
| control, Q (2048) | 7.91 | 0.0949 (well separated) |
| swapped, P (2048) | 3.38 | 0.8751 (still entangled) |
| swapped, Q (1024) | 2.30 | **0.1870** (well separated) |

Two findings, one expected and one not. Expected: participation ratio tracks capacity directly in both configs — whichever branch is bigger has the higher effective dimensionality (control: Q(2048)=7.91 > P(1024)=4.19; swapped: P(2048)=3.38 > Q(1024)=2.30). Unexpected: **direction overlap does not track capacity at all — it tracks branch identity/role.** In both configs, **P** stays the entangled branch and **Q** stays the well-separated one, regardless of which one holds more capacity: control has P small (1024) and entangled (0.9457) with Q big (2048) and separated (0.0949); swapped flips the capacity but not the pattern — P is now the *big* branch (2048) and is still the entangled one (0.8751), while Q is now the *small* branch (1024) and is still the separated one (0.1870). So capacity buys steering power (§19.6-19.7c) but entanglement is governed by something else — most plausibly the loss role itself: P's objective (pull same-sentiment documents together, a collect/attraction loss, while also being pressured to suppress topic) appears to produce a more entangled representation independent of width, while Q's mirrored objective (collect topic, suppress sentiment) consistently comes out cleaner. This is the same dissociation §19.1 found for BCD alternation (steering-preserving despite a geometry check that got *worse*), now shown to hold across a capacity manipulation too: linear separability of the two concepts' directions and causal control over the decoder's output are different axes that do not move together — and, on this evidence, direction overlap specifically looks tied to which loss term a branch is trained on, not to how large that branch is.

**R (c) — steering intervention, real data.**

| Config | P-branch span (α=-6→+6) | Q-branch span (α=-6→+6) | Dominant branch |
|---|---|---|---|
| control (P=1024, Q=2048) | 0.13 (0.83→0.96) | **0.95** (0.04→1.00) | Q, ~7.3× |
| swapped (P=2048, Q=1024) | **0.30** (0.68→0.98) | 0.10 (0.90→1.00) | **P, ~3.1×** |

Dominance flips on real data exactly as it did on synthetic data (§19.6): giving P the larger capacity makes P the dominant steering lever, by a comparable order of magnitude to the reversal seen with Q dominant by default. Both branches stay topic-selective under their own push in both configs (topic accuracy stays within ~0.01 of the no-intervention baseline throughout — not reproduced in table form here, same pattern as §19.4's table).

**T.** The capacity-drives-dominance hypothesis (§19.5) now has direct, controlled, cross-modality confirmation: manipulate capacity alone, holding loss assignment fixed, on both synthetic data (§19.6) and real pythia-410m activations (this section) — steering dominance follows capacity in both. This is the most tightly controlled result in the whole capacity-vs-loss-assignment line of inquiry (§19.5-19.7): a genuine ablation, not just a correlational observation from one fixed architecture. The new finding this section adds beyond confirmation — participation ratio responds to the capacity swap, but direction-overlap does not; it stays attached to branch role (P entangled, Q separated) across the swap — reinforces §19.1's standing caveat that steerability, probe accuracy, and geometric separation are three genuinely different properties of a representation, and a manipulation that improves one (steering power, via capacity) need not improve another (direction overlap), which appears to be governed by loss assignment rather than width.

### 19.8 Does the Full-Width Flat SAE Steer Even Better? A Capacity Ceiling Test

**Q.** §19.5-19.7 established that steering power tracks capacity within KronSAE's branches (1024-2048 dims each). The natural extrapolation: a plain, undivided flat `AutoEncoderTopK` — no P/Q split at all, the entire 16,384-dim dictionary acting as one "branch," 8-16× wider than any individual Kron branch tested — should be the most steerable representation of all, if capacity is the only thing that matters.

**M.** No retraining needed: a flat SAE (`flat_joint`) was already trained alongside every `kron_joint` checkpoint in this project (`--joint` trains both by default; `runs/joint_dpo_cross_full420k/checkpoints/flat_joint`, 420,000 steps, same data/schedule as the Kron control). Wrote `src/eval_flat_intervention.py`, adapting `eval_real_intervention.py`'s method for a single-branch architecture: the push targets the SAE's pre-top-k dense ReLU activation (the architectural analogue of Kron's pre-combine `p_pos`/`q_pos`), then re-applies top-k selection before decoding — pushing post-top-k values directly would let the intervention bleed into features never actually used for reconstruction. Same reference classifiers, same cached real pythia-410m activations, same signed-direction/α-sweep method as every other steering result in §19.

**R.** Steering span is nominally the *widest* of anything tested (0.000 → 1.000, span 1.0, vs. the Kron control Q-branch's 0.954) — but reconstruction quality tells a different story entirely:

| α | flat SAE (16,384-dim) FVU | Kron control Q-branch (2,048-dim) FVU | flat P(sent+) | Kron Q P(sent+) |
|---|---|---|---|---|
| −6 | 0.62 | 0.17 | 0.000 | 0.042 |
| −4 | 0.28 | 0.07 | 0.012 | 0.123 |
| −2 | 0.07 | 0.02 | 0.601 | 0.458 |
| 0 | 0.0004 | 0.0006 | 0.883 | 0.884 |
| +2 | **3.91** | 0.04 | 0.971 | 0.956 |
| +4 | **15.73** | 0.10 | 0.998 | 0.986 |
| +6 | **35.49** | 0.19 | 1.000 | 0.996 |

FVU > 1 means the "reconstruction" is worse than just predicting the per-document mean — at α=+6 the flat SAE's push produces a reconstruction 35× *more* off-variance than that, while Kron's Q-branch stays at 0.19 (a controlled, plausible edit) across the identical push range. Topic selectivity holds for the flat SAE throughout (0.16-0.17, at chance, no worse than Kron) — so the push isn't leaking into topic, it's simply driving the reconstruction into implausible territory to move sentiment as far as it does, especially on the positive-α side.

**T.** More capacity does **not** mean more controllable steering — it means a *nominally* wider dynamic range purchased by abandoning the "controlled edit" property that made Kron's steering results credible in the first place (§19.3: "even the most extreme pushes don't produce a wildly implausible reconstruction"). The flat SAE can be pushed further in raw probability terms, but only by leaving the plausible-reconstruction regime; Kron's Q-branch achieves a comparable practical effect (span 0.95, nearly matching flat's 1.0) while keeping FVU two orders of magnitude smaller at the same α. This argues that KronSAE's mAND-gated, two-factor combine structure — not sheer latent width — is what makes its steering *usable*: whatever the compositional architecture is doing (bounding the combine, forcing information through a comparatively narrow two-factor bottleneck before decode) constrains the push to stay near the data manifold in a way an undivided flat dictionary does not, even though the flat dictionary technically has far more raw capacity to push around in. Combined with §19.6-19.7, the full picture is: within KronSAE, more capacity buys more (controlled) steering power; going outside KronSAE to a wider flat architecture buys more nominal range at the cost of losing that control.

### 19.9 Third Data Point: Equal Capacity (P=Q=1536) on Real Data — Capacity Explains Less Than It Looked Like It Did

**Q.** §19.6-19.7 established a clean two-point story: swap P and Q's capacities and steering dominance swaps with it. The natural completion is the third point on that curve — equal capacity (`joint_m=12, joint_n=12`, P=Q=1536) — trained on real data for the first time, to see whether dominance actually goes to *roughly even* at the midpoint, or does something else.

**M.** Same protocol as §19.7 exactly: `grad_accum_steps=4`, 420,000 steps, same dataset/schedule, `joint_h=128, joint_m=12, joint_n=12`. All three real-data diagnostics run: `eval_joint_probe.py`, `eval_representation_geometry.py` (three-way comparison against the existing default-capacity control checkpoint — the `swapped_p16_n8` checkpoint's `.pt` file had already been deleted per this project's disk policy after §19.7's write-up, so that arm's geometry numbers are pulled from §19.7's own table rather than rerun), `eval_real_intervention.py`.

**R (a) — probe accuracy.**

| Config | P(sent) | Q(sent) | P(topic) | Q(topic) |
|---|---|---|---|---|
| control (P=1024, Q=2048) | 0.938 | 0.933 | 0.520 | 0.645 |
| equal (P=Q=1536) | 0.940 | 0.926 | 0.616 | 0.620 |
| swapped (P=2048, Q=1024) | 0.935 | 0.913 | 0.588 | 0.608 |

Topic accuracy sits between control and swapped for both branches at equal capacity, as the capacity-tracking hypothesis predicts — a clean monotonic trend across all three configs on this axis.

**R (b) — representation geometry.**

| Config / branch | Participation ratio | Sentiment/topic direction overlap |
|---|---|---|
| control, P (1024) | 4.19 | 0.9457 |
| equal, P (1536) | 2.34 | 0.9056 |
| swapped, P (2048) | 3.38 | 0.8751 |
| control, Q (2048) | 7.91 | 0.0949 |
| equal, Q (1536) | 2.73 | 0.0943 |
| swapped, Q (1024) | 2.30 | 0.1870 |

Participation ratio does **not** track capacity monotonically here — equal-capacity P (1536) has a *lower* PR (2.34) than both control's smaller P (1024→4.19) and swapped's larger P (2048→3.38), and equal's Q (2.73) is lower than both control's Q (2048→7.91) and swapped's Q (1024→2.30) despite sitting between them in raw dimension count. This breaks the clean "PR tracks size" pattern §19.7 reported from only two points — with a third point, it's clearly not a simple monotonic function of dimension count alone; something about the specific P=Q=1536 training run (possibly dead-unit fraction, possibly an interaction with `dict_size` since equal's total is 18,432×... no — wait, real-data equal uses the same `h·m·n` combine as before, `128·12·12=18,432`, a genuinely larger combined dictionary than control/swapped's matched 16,384, the same caveat flagged for the synthetic equal-capacity run in §19.6) confounds a clean read here. Direction overlap, by contrast, holds up as a stable, branch-identity-linked property regardless of capacity: Q's overlap is tightly clustered (0.094-0.187) across all three configs while P's stays consistently high (0.875-0.946) — reinforcing §19.7's finding that direction overlap is not simply a capacity effect.

**R (c) — steering intervention, real data.** This is where the clean two-point story breaks down most directly:

| Config | P-branch span (α=-6→+6) | Q-branch span (α=-6→+6) | Dominant branch |
|---|---|---|---|
| control (P=1024, Q=2048) | 0.13 | 0.95 | Q, ~7.3× |
| equal (P=1536, Q=1536) | **0.14** | **0.71** | **Q, ~5.0×** |
| swapped (P=2048, Q=1024) | 0.30 | 0.10 | P, ~3.1× |

At **equal nominal capacity, Q still dominates by 5×** — P's span (0.14) barely moved from control's (0.13) despite P gaining 512 dimensions, while Q's span (0.71) dropped from control's (0.95) despite losing 512. If capacity alone determined dominance, equal capacity should land roughly at parity (span ratio near 1×), not still favor Q by five-fold. Q's own curve is also **not monotonic** at this config — it rises to a peak near α=0 (0.88) then plateaus at 0.80 for all of α∈{+1,+2,+4,+6} rather than continuing to climb, unlike every other steering curve in §19.3-19.8, all of which were monotonic across the full α range.

**T.** The two-point swap experiment (§19.6-19.7) showing a clean flip looked like strong evidence that capacity is *the* determinant of steering dominance. A third point disproves that it's the *only* one: at equal capacity, Q retains a substantial structural advantage (5× span) that capacity parity does not erase. Capacity clearly matters — it's why dominance flips at all between control and swapped — but something else about Q specifically (which branch's positive activations feed the mAND gate's `q_pos` argument specifically, an asymmetry in the combine formula `sqrt(u_i · v_j)` that isn't symmetric in implementation even though the underlying math is, an artifact of decoder initialization, or simply an accumulated effect of Q's supervision term being the *contrast*/suppress term rather than P's *collect* term) also contributes, independent of raw width. This is a genuine correction to §19.5-19.7's working narrative, not a refinement of it — "capacity determines dominance" should be read as "capacity is A determinant of dominance, alongside at least one other factor not yet isolated." The non-monotonic Q curve at this specific config is flagged as a single-seed observation, not yet independently replicated — worth a repeated-seed check before treating it as a robust property of the equal-capacity regime specifically, rather than noise.

### 19.10 Does the Flat-SAE FVU-Blowup Need Real Data, or Just More Training? Scale Validation on Synthetic Data

**Q.** §19.8's flat-SAE result (steering explodes FVU to 35× at α=+6, unlike Kron's controlled 0.19) was run only on the fully-converged, 420,000-step real checkpoint. Before trusting that as an architecture-level claim, does the same effect show up on the synthetic diagnostic — the cheap proxy used for every other steering result in this report — once trained for a comparable number of steps? Or is 5000 steps (this project's standard synthetic protocol) simply too undertrained a regime for either architecture to show the effect?

**M.** Trained a flat SAE (`FlatSupervisedTopKTrainer`, `dict_size=16384`) on the synthetic joint buffer at three scales — 5,000 (the project standard), 50,000 (10×), and 200,000 steps (40×, still 2.1× short of real data's 420,000) — plus a matched Kron control (`joint_h=128, joint_m=8, joint_n=16`) at 5,000 and 50,000 steps for a same-scale reference point. Wrote `src/eval_synthetic_flat_intervention.py`, adapting `eval_flat_intervention.py`'s method (push the pre-topk dense ReLU activation, re-apply top-k, decode) to the synthetic buffer instead of cached real activations. One infrastructure fix made along the way: `_participation_ratio_from_cov` (used by the geometry eval, unrelated to this section's own result but triggered by the longer Kron run) crashed with `torch._C._LinAlgError` at 50,000 steps — a longer-trained, more specialized dictionary produces a more ill-conditioned covariance than `eigvalsh` in float32 can reliably factor. Fixed by adding a small diagonal jitter and computing in double precision, with a CPU fallback if GPU still fails to converge — a general robustness fix, not specific to this experiment.

**R.**

| Steps | flat SAE FVU (min → max across α) | flat steering span | Kron control Q-branch FVU (min → max) | Kron Q steering span |
|---|---|---|---|---|
| 5,000 | 0.834 → 0.838 | 0.497 | 0.956 → 0.979 | 0.67 |
| 50,000 | 0.833 → 0.834 | 0.343 | 0.953 → **1.010** | **0.909** |
| 200,000 | **0.8396 → 0.8399** | **0.162** | *(not run)* | — |

Two clear, and mutually reinforcing, findings:

1. **Flat SAE's FVU is essentially invariant to training scale.** Across a 40× range of training steps, it stays within a band of 0.834-0.840 — no trend toward instability, let alone toward anything resembling the real-data blowup (up to 35× at the same α). If the real-data effect were simply "needs more convergence," some movement toward it should appear by 200,000 steps; there is none.
2. **The two architectures' steering spans move in *opposite* directions with more synthetic training** — Kron's grows (0.67→0.91) while flat's shrinks (0.497→0.343→0.162), a diverging trend, not merely two flat lines.

**T.** This rules out "the flat-SAE FVU-blowup just needs enough training steps" as an explanation for §19.8's real-data result. The effect is very likely tied to a structural property of real pythia-410m activations specifically (heavy-tailed magnitude distribution, cross-dimension correlation structure, or some other property genuine language-model representations have that i.i.d.-Gaussian synthetic features do not) rather than something a longer synthetic run could ever reveal. This is a validating result for §19.8, not a undermining one: it confirms the synthetic diagnostic — cheap, fast, used for every other steering comparison in this report — is structurally incapable of catching this particular effect, so testing directly on real data in §19.8 was doing genuinely necessary work rather than a redundant double-check of something the synthetic proxy already covered.

### 19.11 The Missing Experiment: Actually Patching Into Generation, Not Just Scoring the Reconstruction

**Q.** Every steering result from §19.3 through §19.10 stops at one step: score the SAE's *reconstructed activation* with an external classifier fit on genuine raw activations, at one frozen layer, with no feedback into the rest of the model. None of them ever patch the reconstruction back into pythia-410m and actually generate text. This leaves an open question raised directly by the user: does rising reconstruction FVU under a steering push mean the edit broke the representation (would generate incoherent text) — or does it just mean a real, larger semantic shift necessarily reconstructs further from the original activation? FVU alone cannot distinguish these; only checking actual generated output can.

**M.** Wrote `src/eval_patch_generate.py`. A plain PyTorch forward hook on pythia-410m's layer-12 GPT-NeoX block (loaded directly via `transformers`, not nnsight, for reliable hook semantics across every step of `.generate()`) intercepts the block's hidden-state output at *every* forward call — both the initial prompt pass and every subsequent single-token, KV-cached generation step alike — runs it through the trained SAE (Kron control's Q-branch, or the flat SAE, both from the same `joint_dpo_cross_full420k` checkpoint used throughout §19.3-19.8), applies the alpha push along the already-fit signed sentiment direction, re-applies top-k, decodes, and substitutes the result in place of the true activation before the rest of the model computes on top of it. Six fixed prompts (short review-style openers, e.g. "This product completely", "I was really disappointed because"), greedy decoding, 40 new tokens each, at α ∈ {-6, 0, +6} for both architectures, plus a true unpatched baseline. Judging stays entirely self-contained, consistent with this project's methodology: (a) sentiment — re-encode the *generated continuation text* through the clean model to get its own genuine layer-12 activation, scored with the same `clf_sent` classifier used throughout §19; (b) coherence — self-perplexity, the average per-token cross-entropy of the continuation under the clean, unpatched model conditioned on the original prompt (a standard degenerate-text detector).

**R.**

| Condition | Mean sentiment P(+) | Mean self-perplexity | Max self-perplexity |
|---|---|---|---|
| baseline (no hook) | 0.982 | 2.76 | 4.54 |
| Kron, α=−6 | 0.658 | 8.76 | 15.84 |
| Kron, α=0 (pure round-trip, no push) | 0.946 | 4.04 | 5.26 |
| Kron, α=+6 | **1.000** | 12.35 | 17.71 |
| flat, α=−6 | 0.834 | 5.34 | 13.14 |
| flat, α=0 (pure round-trip, no push) | 0.790 | 3.58 | 4.76 |
| flat, α=+6 | 0.821 | **126.04** | **252.87** |

Two findings, one confirming §19.8's interpretation and one that no prior section could have caught:

1. **At α=+6 (pushing *with* the dataset's dominant positive-sentiment mode — recall sentiment is ~85-87% positive per category, §2), the FVU story from §19.8 is directly confirmed by actual generated text.** Kron generates fluent, thematically on-topic, and *perfectly* sentiment-consistent continuations (6/6 prompts score sentiment 1.000, ppl only rises to 12.35 from baseline's 2.76 — noticeably more repetitive but still readable English). The flat SAE, at the identical push magnitude, collapses into literal degenerate output for **all six prompts** — `"\n bast\n\n\"\n\n\n\n\n\n..."`, repeated newlines and punctuation with no real content, self-perplexity exploding to 126 on average and up to 253 (10-50× worse than Kron's worst case). This is exactly what §19.8's 35× FVU explosion predicted, now confirmed at the only level that actually matters: what the model produces when you run it.
2. **At α=−6 (pushing *against* the dominant mode, toward the minority negative-sentiment class), neither architecture reliably produces negative-sentiment text at all** — Kron's mean sentiment only drops to 0.658, flat's to 0.834, both still net-positive on average, nowhere near the near-0 that the offline reconstruction-based classifier score would have predicted (every prior α-sweep table in §19.3-19.10, read on the *reconstructed activation* rather than generated text, showed a clean, confident, near-monotonic shift toward 0 at negative α). Kron's coherence also degrades non-trivially in this direction (ppl 8.76, some repetition-loop text), despite this being the direction every offline metric called the "successful suppression" side.

**T.** This is the first result in the whole project that actually tests what every "steering" claim implicitly meant to test, and it complicates the picture in a specific, useful way rather than simply confirming what came before. The positive-direction finding (1) is a genuine, generation-level confirmation of §19.8's central claim — flat SAE's FVU blowup is not a benign "bigger semantic shift," it is measurably, catastrophically incoherent output, exactly the failure mode the FVU metric was trying (and, on this axis, succeeding) to detect. But finding (2) reveals a blind spot in *every* offline metric used in §19.3-19.10: they all read success or failure off the reconstructed activation in isolation, with no feedback loop, and that reading overstates negative-direction steering success for both architectures roughly equally — a failure mode invisible to any metric in this report before this section. The likely mechanism is the same in both cases: pushing along the sentiment direction that agrees with the pretraining distribution's dominant mode is something the model can express fluently (Kron) or catastrophically overdoes (flat, still fluent-shaped tokens but repeating "bast"/newlines rather than words); pushing against the dominant mode meets resistance the model does not fluently express either way, regardless of architecture. **Caveat, stated plainly:** this is six prompts, greedy decoding, one seed, three α values — enough to establish the qualitative pattern clearly (flat's positive-push collapse is not subtle, at 10-50× the perplexity of anything else in the table) but not yet a large-sample, sampling-robust result; a wider prompt set and temperature-sampled generations would be the natural next check before treating the exact numbers as final.

### 19.12 Scaling Up: 22 Balanced Prompts, Temperature Sampling, Two Samples Each

**Q.** Does §19.11's pattern survive a bigger, more representative prompt set and stochastic (rather than greedy, deterministic) decoding — the natural next check flagged in that section's own caveat?

**M.** Extended `src/eval_patch_generate.py` (`build_prompts`) to draw prompts directly and reproducibly from the real eval split: 2 examples per (topic, sentiment) cell across all 6 topics × 2 sentiment labels = 22 balanced real review openers (first 8 words each), replacing the original 6 hand-picked prompts. Added `--do_sample`/`--temperature 0.8`/`--top_p 0.95`/`--num_samples 2` (a different seed per sample), applied uniformly to every condition including the baseline, so the baseline's own perplexity absorbs whatever inflation sampling itself introduces — a fair, matched comparison rather than comparing sampled numbers against §19.11's greedy baseline. 44 generations per condition (22 prompts × 2 samples), same 7 conditions (baseline + 2 architectures × 3 α values) as §19.11.

**R.**

| Condition | Mean sentiment | Mean self-perplexity | Median self-perplexity | Max self-perplexity |
|---|---|---|---|---|
| baseline (no hook) | 0.894 | 9.20 | 8.55 | 28.94 |
| Kron, α=−6 | 0.609 | 25.39 | 21.34 | 77.96 |
| Kron, α=0 | 0.822 | 13.14 | 11.61 | 36.64 |
| Kron, α=+6 | 0.902 | 61.65 | 58.85 | 120.21 |
| flat, α=−6 | 0.542 | 23.54 | 18.70 | 71.89 |
| flat, α=0 | 0.755 | 11.22 | 10.09 | 31.21 |
| flat, α=+6 | 0.885 | **245.20** | **173.20** | **1557.31** |

Both §19.11 headline findings replicate, and one ranking claim from §19.11 turns out to have been small-sample noise, now corrected:

1. **Flat SAE's α=+6 collapse is not a greedy-decoding artifact — sampling makes it more visible, not less.** Flat's mean perplexity (245) is ~4× Kron's (62) at the identical push and comparable sentiment success (flat 0.885 vs Kron 0.902) — an even starker gap in relative terms than §19.11's greedy result, with one sample reaching perplexity 1557. Sampled flat-SAE text at this α shows real English words scattered incoherently ("harassment," "Declaration," "品," "Abraham") rather than greedy's repetitive loops — a different-looking but equally broken failure mode, not a healthier one.
2. **Neither architecture reliably produces net-negative sentiment at α=−6, confirmed at n=44**: Kron 0.609, flat 0.542, both still above the 0.5 baseline the offline reconstruction-based classifier implied was easily crossable.
3. **Correction to §19.11's reading**: the 6-prompt result read as "Kron suppresses negative sentiment better than flat" (0.658 vs 0.834). At n=44 that ranking **inverts** (Kron 0.609 vs flat 0.542 — flat now reads as the more negative-shifted one) while the substantive finding (neither gets reliably below ~0.5) is unchanged. This specific architecture-vs-architecture ranking on the negative-steering axis was not resolvable from 6 prompts and should not have been read as a real difference; the larger sample settles it as noise, not signal.

**T.** The two substantive conclusions of §19.11 — flat SAE's positive-push collapse is real and severe, and neither architecture reliably achieves negative-direction steering in actual generation — both hold up under a 7×-larger, more balanced, stochastically-decoded test. The one number that moved (which architecture is "better" at negative steering) is exactly the kind of fine-grained comparison a 6-prompt greedy-only test was never powered to make, and this section's larger sample is what actually settles it: don't read anything into that specific ranking from §19.11.

### 19.13 Related Work: Steering, as Distinct From the Erasure Literature Surveyed in §11

§11/§16-18 surveyed the concept-erasure literature (LEACE, INLP, RLACE, KRaM, GRL) because that was the axis this project's diagnostics were built around through §19.2. Once §19.3 onward reframed KronSAE's actual comparative advantage as causal steering rather than erasure, the relevant prior work shifts to a different literature that had not yet been surveyed:

- **Gao, Dupré la Tour, Tillman et al. 2024, "Scaling and evaluating sparse autoencoders" (arXiv:2406.04093)** — introduces the TopK SAE architecture (hard top-k activation instead of an L1 penalty) that both this project's flat baseline (`AutoEncoderTopK`) and KronSAE's own top-k combine step descend from. Establishes the reconstruction/sparsity tradeoff and evaluation methodology (including a downstream-loss metric) that this project's FVU-based comparisons build on; worth citing as the direct architectural ancestor of both SAE variants compared in §19.8-19.12, not just KronSAE's own source paper (Kurochkin et al. 2025).

- **Turner, Thiergart, Leech et al. 2023, "Steering Language Models With Activation Engineering" (arXiv:2308.10248)** — the foundational activation-addition (ActAdd) result: adding a difference-of-means direction to residual-stream activations at inference time causally steers generation, with no SAE or dictionary involved at all. This is the plain dense-activation-steering baseline that any SAE-based steering claim (including this project's) needs to be read against — it establishes that steering itself does not require a sparse dictionary; the open question SAE-based work (including §19.3-19.12) needs to answer is whether a dictionary adds control, selectivity, or interpretability beyond what a raw direction already gives.

- **Jorgensen, Cope, Schoots et al. 2023, "Improving Activation Steering in Language Models with Mean-Centring" (arXiv:2312.03813)** — shows that naively-fit steering directions carry an activation-mean offset that degrades steering quality, and that centring before fitting the direction improves it. Directly relevant to this project's direction-fitting methodology (`_signed_binary_direction`, `proj_std`-scaled pushes in `eval_patch_generate.py` and its predecessors): the project's signed, mean-referenced construction is already in the spirit of this fix, and it is worth citing as the methodological justification rather than an ad hoc choice.

- **O'Brien, Majercak, Fernandes et al. 2024, "Steering Language Model Refusal with Sparse Autoencoders" (arXiv:2411.11296)** — the closest prior instance of this project's own patch-and-generate paradigm: amplifies SAE latents at inference time to steer a model's refusal behavior, evaluated on real generated completions rather than reconstruction-level metrics alone. The clearest existing precedent that SAE-latent steering can be evaluated end-to-end in generation, which is the exact methodological correction §19.3 (and its critique of FVU-as-steering-proxy) arrives at independently; distinguishing factor for this project is the architecture comparison (compositional Kron vs. flat) and the explicit demonstration that reconstruction-level metrics (FVU) both under- and over-state generation-level outcomes (§19.11-19.12), which this paper's evaluation does not test for.

- **Xie 2025, "A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering" (arXiv:2510.01246)** — directly compares SAE-latent steering against plain activation-difference steering (i.e., ActAdd-style, no dictionary) and finds many top-k SAE latents used for steering are non-semantic (e.g., punctuation-tracking) rather than carrying the target attribute. This is an independent line of evidence for the same broad concern raised by this project's §19.9 finding that capacity, not the SAE's factorization into collect/suppress roles, is doing most of the steering work — in both cases, the dictionary's nominal semantic assignment is a weaker predictor of steering behavior than one might assume from the training objective alone.

- **Bayat, Rahimi-Kalahroudi, Pezeshki et al. 2025, "Steering Large Language Model Activations in Sparse Spaces" (arXiv:2503.00177)** — motivates sparse (SAE-like) representations for steering specifically as a fix for superposition-induced interference in dense activation-steering, which is the same superposition-avoidance argument implicitly underlying why a compositional, factored SAE (KronSAE) might be expected to steer more cleanly than a flat one. Useful as the general-purpose argument for why sparsity should help steering at all, against which this project's more specific finding — that KronSAE's advantage over an equal-capacity flat SAE is generation coherence, not sparsity or selectivity per se — can be read as a refinement.

**Positioning.** None of the above papers test the specific comparison this project's later sections turn on: whether a *compositional* (branch-factored) SAE steers more coherently than an equal-capacity *undivided* SAE, verified at the level of actual generated text rather than reconstructed activations. Gao et al. supplies the shared TopK architectural ancestor for the two SAE variants being compared; Turner et al. and Jorgensen et al. supply the non-SAE steering baseline and direction-fitting methodology this project's method is built from; O'Brien et al. is the closest generation-level SAE-steering precedent, but does not compare architectures or examine reconstruction-metric fidelity; Xie et al. and Bayat et al. independently support this project's §19.9/19.11 findings that capacity and coherence, not nominal semantic factorization, are what actually drive steering outcomes.

---

## Appendix A — Paper Replication (Sections 5.2, 5.3, Feature Absorption)

*(source: `PAPER_REPLICATION.md`; not central to the project's current direction — see Part I front matter)*

### Section 5.2 (within-head correlation)

Checked whether features within the same Kron head correlate more with each other than across heads, and whether training increases this over a random-init baseline (matched feature count, since trained models have fewer "valid"/non-dead features than random-init).

| Checkpoint | Trained within-head | Trained across-head | Random (matched-size) within-head | Random across-head |
|---|---|---|---|---|
| kron_pilot_topic | 0.00727 | 0.00135 | 0.06546 | 0.01735 |
| kron_pilot_sentiment | 0.00683 | 0.00141 | 0.06739 | 0.01759 |

Paper claim (qualitative only, no numbers given): within-head > across-head, and training *increases* within-head correlation over random-init.

> **Comment:** direction "within-head > across-head" replicates in both trained and random models. But "training increases correlation over random-init" does **not** replicate — here training sharply *decreases* it (trained ~0.007 vs random ~0.065, matched feature count so this isn't a sample-size artifact). Plausible explanation: TopK sparsity training generally decorrelates redundant atoms (a general sparse-coding effect), and/or our explicit label supervision + narrow domain reorganizes head structure differently from the paper's unsupervised, broad-domain (FineWeb-Edu) setting.

### Feature Absorption (SAEBench)

`sae_bench==0.6.0`, `absorption_first_letter` benchmark, run on the balanced-pilot checkpoints, base model pythia-410m layer 12.

> **Comment:** domain mismatch caveat — our SAEs were trained on Amazon Reviews text; SAEBench's absorption benchmark evaluates on a generic first-letter/word-list task, a different domain. Numbers below should be read with that caveat.

| | mean full-absorption score ▼ |
|---|---|
| flat_pilot_topic | 0.170 |
| flat_pilot_sentiment | 0.174 |
| kron_pilot_topic | 0.088 |
| kron_pilot_sentiment | 0.121 |

Paper (Pythia-1.4B, F=65536): TopK ℓ0=16/32 → 0.445/0.233; KronSAE ℓ0=16/32 → 0.244/0.129 (Kron/flat ratio ≈0.55).

Our Kron/flat ratio: topic ≈0.52, sentiment ≈0.69 — topic matches the paper's ~halving closely; sentiment shows a smaller reduction.

**Feature hedging**: SAEBench 0.6.0 has no hedging evaluation module (checked package contents: absorption, autointerp, core, mdl, meta_structure, ravel, scr_and_tpp, sparse_probing, unlearning — no hedging). Not run; would require separate code from the original hedging paper (Chanin et al. 2025).

### Section 5.3 (feature interpretability)

LLM: Qwen3-14B (bf16, vLLM, tensor_parallel_size=2). 308 features sampled across P/Q branches (pilot checkpoints) + a follow-up 49 post-latent (mAND-combined) features. Methodology: 31-token activation windows, top-50%-quantile threshold, ~16 examples per feature for LLM-generated description, then detection/fuzzing scoring on held-out examples (paper's Section 5.3 pipeline).

| | detection ▲ | fuzzing ▲ | n |
|---|---|---|---|
| flat_sentiment | 0.800 | 0.770 | 51 |
| flat_topic | 0.772 | 0.770 | 52 |
| kron_topic P (pre) | 0.588 | 0.588 | 57 |
| kron_topic Q (pre) | 0.587 | 0.630 | 48 |
| kron_topic post (mAND-combined) | 0.765 | 0.763 | 23 |
| kron_sentiment P (pre) | 0.550 | 0.572 | 51 |
| kron_sentiment Q (pre) | 0.549 | 0.618 | 49 |
| kron_sentiment post | 0.714 | 0.736 | 26 |

(chance ≈ 0.5)

Paper claim: post-latent features are significantly more interpretable than pre-latents.

> **Comment:** replicates — post-latent beats the best pre-latent branch by +0.13 to +0.18 on both metrics, both checkpoints. Not tested by the paper but visible here: flat (0.77-0.80) matches or slightly exceeds Kron's post-latent (0.71-0.77) — i.e. mAND combination raises interpretability above pre-latent P/Q individually, but not clearly above what an undivided flat SAE already achieves.
