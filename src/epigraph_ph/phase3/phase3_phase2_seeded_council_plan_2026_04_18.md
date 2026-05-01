# Phase 3 Phase 2 Seeded Champions Council Plan

**Date:** 2026-04-18  
**Question:** after the first bounded Phase 2 seeded-champion scenario batch, should the next phase expand bounded scenario families on the same kernel, or move to a stricter regime/sojourn layer?

## Inputs

- [future_experiment_phase2_seeded_champions_2026_04_14.md](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/future_experiment_phase2_seeded_champions_2026_04_14.md)
- [tr_v3_phase2_seeded_champion_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/tr_v3_phase2_seeded_champion_batch_report.md)
- [tr_v3_monthly_edge_audit_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-edge-audit-20260418-s01/analysis/tr_v3_monthly_edge_audit_batch_report.md)
- [tr_v3_monthly_phase2_lane_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260418-s02/analysis/tr_v3_monthly_phase2_lane_batch_report.md)
- [phase3_phase2_seeded_decision_gate_2026_04_18.md](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/phase3_phase2_seeded_decision_gate_2026_04_18.md)
- [phase3_phase2_seeded_representation_plan_2026_04_18.md](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/phase3_phase2_seeded_representation_plan_2026_04_18.md)
- [phase3_phase2_seeded_champions_validity_skeptic_2026_04_18.md](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/phase3_phase2_seeded_champions_validity_skeptic_2026_04_18.md)

## AI Scientist Status

`ai-scientist-v2` was attempted but not operational on this Linux workspace:

- the required `doctor` step fails because the wrapper hardcodes a Windows `D:\` disk probe
- `ideate` also cannot run because no Ollama or API backend is configured

So the plan below is based on the council plus repo evidence, not on AI Scientist ideation output.

## Direct Evidence

| Artifact | Concrete evidence | Read |
|---|---|---|
| [tr_v3_monthly_phase2_lane_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260418-s02/analysis/tr_v3_monthly_phase2_lane_batch_report.md) | `192` contiguous months, `697` monthly national rows, `622` injected HARP rows, `60` historical-panel rows, `22` canonicals | monthly lane is now geometrically usable |
| [tr_v3_monthly_phase2_lane_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260418-s02/analysis/tr_v3_monthly_phase2_lane_batch_report.md) | Phase 15 shape `[1, 192, 4]`; Phase 2 yields `4` blocks, `2` direct lagged edges, `0` hidden rows | current structural substrate is still small and direct |
| [tr_v3_monthly_edge_audit_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-edge-audit-20260418-s01/analysis/tr_v3_monthly_edge_audit_batch_report.md) | `testing -> care @ lag1` score `0.157639 -> 0.222596`; `testing -> suppression @ lag1` newly `0.225294`; no support-collapse flag | narrow testing-centered follow-up is justified |
| [tr_v3_phase2_seeded_champion_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/tr_v3_phase2_seeded_champion_batch_report.md) | seeded layer uses `4` blocks and `2` lag-1 edges over `2010-Q1` to `2025-Q4` | current seeded scenario engine is a bounded scenario layer, not a hidden-state engine |
| [tr_v3_phase2_seeded_champion_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/tr_v3_phase2_seeded_champion_batch_report.md) | exact residual counts `12-13`; dense residual counts `33-49` | readout is data-thin, especially on exact |
| [tr_v3_phase2_seeded_champion_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/tr_v3_phase2_seeded_champion_batch_report.md) | dense `testing_plateau` terminal deltas `(-728.8, -340.6, -180.0)`; dense `disruption_recovery` `(72.5, -131.4, 31.5)` | bounded kernel already separates plateau and recovery behavior in a meaningful way |

## Contextual Evidence

| Concern | Status |
|---|---|
| hidden-driver support | absent |
| hidden modes / dwell-time evidence | not established |
| support geometry confounding | one major bug already fixed; must stay guarded |
| exact readout sample size | weak |
| seeded scenario semantics | plausible, but still partly hand-authored |
| archive alignment | current seeded batch used an older archive run and should be rerun on the merged HARP-enriched archive |
| circularity risk | real; current Phase 2 blocks include some of the same observed heads later perturbed by the seeded layer |

## Council Synthesis

All council roles agreed on these points:

- do **not** move to a regime/sojourn layer now
- do **not** claim future shock prediction
- keep the seeded layer as a **bounded scenario sandbox**
- require a falsification/stability gate before promoting more scenario work

The only live disagreement was ordering:

- `A`: expand same-kernel scenarios after minimal stability gates
- `B`: do a stronger falsification/admissibility package first, then decide
- `AB`: run one bounded falsification + admissibility batch now; if it passes, expand same-kernel scenarios next; keep regime/sojourn deferred

## Autoreason Pass

Winner: `AB`

### Strongest claim retained

The next publishable and defensible move is still on the **same bounded kernel**, not a new HSMM/semi-Markov layer.

### Strongest claim revised or removed

“Expand scenario families next” is too eager.  
It becomes defensible only **after** one archive-aligned falsification and admissibility batch.

### Why `AB` won

- preserves the strongest empirical signal already visible in the seeded batch
- answers the skeptic’s circularity and archive-mismatch objections
- avoids a premature jump to hidden-state complexity
- still keeps momentum toward a richer scenario library if the current layer survives

## Proper Plan

### Phase 1: falsification and admissibility

Run this as one bounded batch before any new scenario family work.

1. `EXP-P2-GATE-00`
   Archive-aligned rerun of the seeded batch using the merged HARP-enriched archive, not the older `harp-archive-wdi-standard-20260412-s19`.

2. `EXP-P2-GATE-01`
   Blocked edge-stability audit.
   Check whether the two retained lag-1 edges survive across blocked monthly folds with stable sign and rank order.

3. `EXP-P2-GATE-02`
   Seeded readout stability audit.
   Refit the residual readout under blocked folds and bootstrap resamples.
   Gate on coefficient sign stability, terminal-delta stability, and boundedness.

4. `EXP-P2-GATE-03`
   Boundedness and stock-flow sanity audit.
   Verify seeded outputs never violate:
   - `alive_on_art <= diagnosed_plhiv`
   - non-negative counts
   - scenario deltas staying within pre-registered caps

5. `EXP-P2-GATE-04`
   Outcome-circularity ablation.
   Remove or mute Phase 2 block contributions derived from the same direct heads later perturbed by the seeded layer, then rerun the seeded readout.

6. `EXP-P2-GATE-05`
   Placebo shock audit.
   Use edge-permuted or block-shuffled kernels under the same scenario templates.
   Keep only if the real kernel materially outperforms the placebos on structural coherence and readout consistency.

7. `EXP-P2-ADMIT-01`
   Empirical block excursion atlas.
   Build observed quarterly excursion envelopes for each block from the monthly lane.

8. `EXP-P2-ADMIT-02`
   Descriptive dwell summaries.
   Measure how long blocks historically remain elevated, suppressed, or recovering.
   This is descriptive dwell evidence, not yet a regime model.

9. `EXP-P2-ADMIT-03`
   Admissible scenario library seed.
   Convert the observed excursion atlas plus dwell summaries into allowed shock magnitudes and durations.

### Phase 2: scenario-family expansion on the same kernel

Only if Phase 1 passes.

1. `EXP-P2-SCEN-01`
   Expand scenario families using admissible, data-bounded templates:
   - testing collapse
   - care bottleneck
   - suppression push
   - testing then delayed care
   - two-wave disruption
   - slow drift up
   - slow drift down
   - mixed policy pulse

2. `EXP-P2-SCEN-02`
   Scenario-semantics audit.
   Confirm families are not just amplitude-scaled duplicates.

3. `EXP-P2-SCEN-03`
   Graph pack and narrative pack.
   Produce final structural-path plots, quarter-level effect tables, and champion-response overlays for interpretation.

### Phase 3: minimal regime feasibility only if needed

Only after Phase 1 and Phase 2 pass, and only if explicit dwell-time semantics become necessary.

1. `EXP-P2-REG-00`
   Tiny sticky regime feasibility probe with `K=2..3`.

2. Keep only if:
   - it improves interpretability over the admissible scenario library
   - regime occupancy is stable across blocked folds
   - it is not just restating amplitude categories

3. Do **not** move to larger HSMM/semi-Markov work before this tiny probe clears.

## Keep / Revert Rules

Keep the seeded path alive only if Phase 1 shows all of these:

- archive-aligned seeded results remain qualitatively similar
- retained edges stay stable across blocked folds
- readout coefficients and terminal deltas are stable enough to interpret
- placebos do not produce comparable scenario behavior
- admissible shock magnitudes and durations can be grounded in observed block excursions

Revert or pause if any of these happen:

- edge signs or ranks flip across folds
- exact readout remains too unstable to interpret
- placebo kernels produce similar scenario responses
- circularity ablation destroys the seeded effect
- scenario families are revealed to be amplitude clones only

## Avoid Next

- no regime/sojourn layer now
- no large HSMM now
- no future-shock prediction claim
- no promotion of the seeded layer into the benchmark loop
- no many new hand-authored scenarios before the admissibility batch

## Graphs To Keep Open

- [phase2_seeded_structural_scenarios.png](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/phase2_seeded_structural_scenarios.png)
- [exact_only_phase2_seeded_scenarios.png](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/exact_only_phase2_seeded_scenarios.png)
- [purged_dense_phase2_seeded_scenarios.png](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/purged_dense_phase2_seeded_scenarios.png)
- [phase2_seeded_readout_heatmap.png](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/phase2_seeded_readout_heatmap.png)
- [phase2_seeded_terminal_delta_heatmap.png](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/phase2_seeded_terminal_delta_heatmap.png)
- [testing_edge_bars.png](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-edge-audit-20260418-s01/analysis/testing_edge_bars.png)
- [edge_delta_heatmap.png](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-edge-audit-20260418-s01/analysis/edge_delta_heatmap.png)

## AutoResearch Handoff

- Variant: `evidence-to-model-loop`
- Objective: keep the Phase 2 seeded scenario line only if archive-aligned falsification and admissibility checks pass
- Evaluation harness:
  - blocked monthly edge stability
  - readout bootstrap stability
  - boundedness / stock-flow sanity
  - placebo shock comparison
  - admissible excursion/dwell summaries
- Mutation units:
  - archive alignment
  - edge stability audit
  - readout bootstrap audit
  - circularity ablation
  - placebo shock audit
  - excursion atlas
  - descriptive dwell summaries
  - admissible scenario library
- First experiments:
  - `EXP-P2-GATE-00`
  - `EXP-P2-GATE-01`
  - `EXP-P2-GATE-02`
  - `EXP-P2-GATE-03`
  - `EXP-P2-GATE-04`

