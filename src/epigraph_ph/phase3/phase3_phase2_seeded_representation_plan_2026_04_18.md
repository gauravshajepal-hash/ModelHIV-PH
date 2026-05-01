# Phase 2 Seeded Champions: Representation Plan

**Date:** 2026-04-18

## Scope

Role: Representation / Modeling Agent

Question:
- `A)` expand bounded scenario families on the existing kernel
- `B)` add a stricter regime/sojourn layer now
- `C)` choose a stronger prerequisite path if the current evidence supports it

## Decision

**Winner: `C`**

The next justified artifact is **not** a larger scenario menu and **not** a regime/sojourn model.

The next justified artifact is a **Phase 2 structural admissibility and stability package**:
- empirical block excursion atlas
- edge and readout stability audit under blocked history slices
- admissible perturbation envelope for seeded scenarios
- simple dwell summaries without fitting a new latent regime model yet

Only after that should the project expand bounded scenario families on the existing kernel.

## Why `C` Wins

### Direct Evidence

| Artifact | Evidence | Read |
|---|---|---|
| [future_experiment_phase2_seeded_champions_2026_04_14.md](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/future_experiment_phase2_seeded_champions_2026_04_14.md) | the intended use is structural scenario emulation, not benchmark forecasting | the modeling contract is already bounded and scenario-oriented |
| [tr_v3_phase2_seeded_champion_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/tr_v3_phase2_seeded_champion_batch_report.md) | seeded pass already works as a bounded stress-test layer | there is already one viable kernel to extend |
| [phase2_structural_payload.json](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260418-s02/phase2/phase2_structural_payload.json) | `4` blocks, `2` direct lag-1 edges, `0` hidden rows, `0` hidden modes surviving threshold | there is not enough structural richness to justify a regime engine yet |
| [tr_v3_monthly_edge_audit_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-edge-audit-20260418-s01/analysis/tr_v3_monthly_edge_audit_batch_report.md) | testing-to-care strengthened and testing-to-suppression appeared under preserved support | the monthly lane is promising, but still narrow |
| [tr_v3_monthly_phase2_lane_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260418-s02/analysis/tr_v3_monthly_phase2_lane_batch_report.md) | rebuilt lane fixed the axis defect and preserves monthly support | geometry is now usable, but representation strength is still modest |

### Contextual Evidence

| Concern | Current status |
|---|---|
| Structural width | only `2` direct edges survive |
| Hidden state evidence | absent |
| Edge support counts | only `3` for each direct edge |
| Readout data size | residual counts are small on exact (`12-13`) and moderate on dense (`33-49`) |
| Current seeded scenarios | interpretable, but partly hand-authored |
| Overfitting risk for HSMM/regime layer | high on one national sequence with no hidden rows |

## Why Not `A` Immediately

`A` is closer than `B`, but still premature as the immediate next artifact.

The current kernel is usable, but expanding scenario families immediately would mostly increase the number of hand-authored stories. That creates a scientific risk:
- more scenarios without an admissibility envelope become harder to justify
- scenario responses can be interpreted too strongly when the underlying structural support is still thin
- the project could drift into narrative inflation instead of evidence-bound stress testing

So `A` should happen **after** the admissibility/stability package, not before.

## Why Not `B` Now

A stricter regime/sojourn layer is not justified yet because the current monthly structural outputs do **not** show:
- threshold-surviving hidden rows
- multiple stable lag pathways
- robust direct support beyond `3` counts per retained edge
- enough evidence that dwell-time mis-specification is the current bottleneck

Right now, a regime/sojourn model would mostly add latent structure on top of a graph that is still sparse and only recently corrected for monthly support geometry.

That is the wrong complexity jump.

## Representation Choice

Use the existing seeded kernel as the base, but add a new representation layer around it:

### Artifact

**Phase 2 Structural Admissibility Package**

This package should encode:
- empirical quarterly block ranges
- historical shock magnitudes
- empirical plateau durations
- recovery lag distributions
- blocked-split edge and readout stability

The seeded champion layer should then consume only **admissible structural perturbations** derived from those summaries.

### Practical Form

Keep:

\[
\tilde y_t = \hat y_t^{champ} + \lambda \Delta(z_t)
\]

But constrain future `z_t` perturbations using:
- observed block excursion quantiles
- observed multi-quarter dwell windows
- edge/readout stability weights

This gives a defensible scenario engine without inventing an HSMM prematurely.

## Minimal Experiment Ladder

### 1. `EXP-P2-ADMIT-01`
Build an empirical block excursion atlas from the corrected monthly lane.

Output:
- per-block quarterly level distributions
- per-block quarterly delta distributions
- top historical excursion windows
- pairwise co-movement summaries

Keep rule:
- excursion atlas must clearly separate ordinary variation from tail events

### 2. `EXP-P2-ADMIT-02`
Estimate descriptive dwell summaries without fitting a latent regime model.

Output:
- plateau duration histogram
- recovery lag histogram
- disruption duration histogram
- block-threshold occupancy summaries

Keep rule:
- dwell summaries must be stable under blocked history slices

### 3. `EXP-P2-ADMIT-03`
Run blocked stability on the two retained edges and on the seeded readout.

Output:
- splitwise edge weights
- splitwise readout coefficients
- stability intervals
- one “safe perturbation budget” per block

Keep rule:
- perturbation budgets should not depend on a single slice or one late-era window

### 4. `EXP-P2-ADMIT-04`
Generate an admissible scenario library from the atlas rather than hand-authoring all pulses.

Examples:
- mild testing plateau
- severe testing collapse
- delayed care rebound
- coupled testing and suppression push
- mobility stress envelope

Keep rule:
- every scenario must map to a historical or quantile-bounded structural pattern

### 5. `EXP-P2-SCEN-02`
Only after the above, expand the bounded scenario families on the same kernel.

This is the first place where option `A` should proceed.

## Explicit No-Go List

Do **not** do these next:

- do not fit a large HSMM
- do not add a semi-Markov layer and call it identified
- do not create many new hand-authored scenarios before the admissibility pass
- do not let Phase 2 overwrite the champion mean path
- do not claim future-shock prediction from endogenous history alone
- do not treat the current two-edge graph as a discovered mechanistic HIV system
- do not use suppression as a live observed structural head outside its supported window

## What `A` Becomes After `C`

If the admissibility package lands, then option `A` becomes the right next move:
- expand bounded scenario families
- keep the same kernel
- attach each scenario to a historical or quantile-bounded analogue
- publish graph packs and envelope comparisons, not forecast win claims

## Autoresearch Handoff

- Variant: `evidence-to-model-loop`
- Evaluation harness:
  - corrected monthly HARP lane
  - blocked stability slices
  - seeded scenario report pack
  - admissibility envelope checks
- Mutation units:
  - block excursion atlas
  - dwell summary extraction
  - splitwise edge/readout stability
  - scenario-budget enforcement
  - bounded scenario-library expansion
- Stop rule:
  - keep only if the new artifact reduces arbitrariness and increases interpretability without adding unsupported latent structure

## Tooling Note

`ai-scientist-v2` was attempted but is not usable on this Linux workspace as configured:
- the wrapper doctor script hardcodes a Windows `D:\\` disk probe
- ideation also requires an Ollama or API backend that is not configured here

So this plan is grounded in repo artifacts plus council-style evidence synthesis, not the AI Scientist runtime.
