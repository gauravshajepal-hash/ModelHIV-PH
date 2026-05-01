# Phase 2 Outcome-Lite Council Plan

Date: 2026-04-19

Method:
- same-model council with four roles: evidence, validity, representation, evaluation
- bounded peer-critique round
- one `A / B / AB` autoreason arbitration pass

## Decision

Autoreason winner: `AB`

Meaning:
- do not rebuild the testing block immediately
- do not expand scenario families now
- first stabilize the cleaned 3-block substrate under archive-matched sidecar ablation and archive-alignment repair
- only then attempt a non-outcome testing rebuild, if the cleaned core survives

## Strongest Retained Claim

The outcome-lite rerun successfully removed the old direct outcome contamination and left a smaller but real monthly structural core:
- retained blocks: `care_access_continuity`, `suppression_capacity`, `mobility_exposure_pressure`
- retained direct edge: `care_access_continuity -> suppression_capacity @ lag1`
- blocked-fold edge gate: `keep`
- outcome loading share in retained blocks: `0.000`

## Strongest Revised Claim

The project does not currently have a defensible testing-centered structural substrate.

What changed:
- the old testing branch was circular and unstable
- after pruning direct outcome heads, `testing_engagement` disappeared
- `testing_pulse` and `testing_plateau` now collapse to zero

So the correct update is:
- the old testing-centered layer was mostly contaminated
- the surviving kernel is care/suppression-centered
- testing must be rebuilt from upstream, non-outcome indicators or dropped from the seeded scenario substrate

## Direct Evidence Table

| Claim | Evidence For | Evidence Against / Limit |
|---|---|---|
| Outcome contamination was materially reduced | [clean gate report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-gates-20260419-s00-outcome-lite/analysis/tr_v3_phase2_seeded_gate_batch_report.md) shows `0` outcome indicators and `0.000` outcome loading share for retained blocks | This does not prove interpretability; sidecar dependence remains |
| A small structural core survives | [monthly lane report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260419-s00-outcome-lite/analysis/tr_v3_monthly_phase2_lane_batch_report.md) shows `3` retained blocks and `1` direct edge; [clean gate report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-gates-20260419-s00-outcome-lite/analysis/tr_v3_phase2_seeded_gate_batch_report.md) keeps edge stability | The core is thin: no hidden rows, no testing block, one edge only |
| The old testing layer is not currently valid | [old gate report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-gates-20260418-s01/analysis/tr_v3_phase2_seeded_gate_batch_report.md) had outcome-heavy loadings and unstable testing-led edges; [clean aligned seeded report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-gates-20260419-s00-outcome-lite-aligned-seeded/analysis/tr_v3_phase2_seeded_champion_batch_report.md) implies testing scenarios collapse | That does not yet tell us whether a rebuilt upstream testing block can work |
| Archive alignment is the current hard blocker | [clean gate report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-gates-20260419-s00-outcome-lite/analysis/tr_v3_phase2_seeded_gate_batch_report.md): sign agreement `0.375`, mean abs delta-of-delta `132.533`, gate `revisit_archive_alignment` | Alignment is being compared across a materially changed substrate, so some degradation is expected |
| Remaining risk moved from direct outcome overlap to sidecar/time-mix dependence | [clean loading report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-loading-sanity-20260419-s01-outcome-lite/analysis/tr_v3_monthly_loading_sanity_batch_report.md): `care_access_continuity` cascade share `0.547`; `suppression_capacity` cascade share `0.939`, annual-only share `0.411` | These may still be useful measurement anchors if they survive ablation |

## Contextual Evidence Table

| Concern | Current Read |
|---|---|
| Global portability of one seeded layer | not supported for the old testing-centered substrate |
| Care-to-suppression path | provisionally supported |
| Testing control surface | absent after cleanup |
| Scenario expansion | blocked |
| Hidden-state / regime escalation | premature |
| Archive-matched validation | mandatory next |

## Candidate Plans For Autoreason

### `A`
Rebuild `testing_engagement` immediately as `testing_prevention_reach` using only upstream indicators:
- normalized `annual_hiv_tests_volume`
- `prevention_coverage` or `prevention_access`
- `hiv_knowledge_index`
- optional `prep_people_receiving` as a rate

Then rerun the seeded gates.

### `B`
Do not rebuild testing yet.

First run:
- archive-matched cascade-sidecar ablation on the current 3-block core
- archive-alignment repair on the cleaned core

Only after that decide whether testing should be rebuilt or abandoned.

### `AB`
Two-stage plan:
1. stabilize and falsify the cleaned 3-block core
2. only if it passes, rebuild testing as an upstream block and rerun semantic/non-null gates

## Why `AB` Won

`AB` beat `A` because:
- immediate testing rebuild would stack two uncertainties at once: archive instability plus representation change
- the council agreed the current hard blocker is archive/sidecar validity, not missing complexity

`AB` beat `B` because:
- `B` is too conservative if the cleaned core passes
- there is still scientific value in trying to recover a true upstream testing/prevention block later

So `AB` is the best bounded plan:
- clean the retained substrate first
- then decide whether testing can be honestly restored

## Concrete Plan

### Phase 1: Stabilize The Clean Core

#### `EXP-P2-GATE-00b`
Archive-alignment repair on the outcome-lite 3-block substrate.

Scope:
- freeze kernel
- freeze readout basis
- freeze scenario amplitudes
- vary only archive-alignment plumbing / normalization

Keep if:
- terminal-delta sign agreement `>= 0.75`
- mean abs delta-of-delta does not worsen from `132.533`

#### `EXP-P2-GATE-04c`
Archive-matched cascade-sidecar ablation.

Scope:
- prune or downweight the highest-risk retained sidecars first:
  - `tested_for_viral_load`
  - `virally_suppressed`
  - `suppression_outcomes`
  - `viral_suppression_rate`
- rerun monthly lane, loading sanity, and seeded gates on the same archive

Keep if:
- retained edge still survives blocked folds
- archive alignment improves materially
- outcome-ablation remains stable

Revert if:
- the surviving edge disappears immediately
- or alignment does not improve

### Phase 2: Semantic Repair

Decision branch after Phase 1:

#### Branch `T`
If the cleaned core survives, rebuild testing as an upstream block.

Rename:
- `testing_engagement` -> `testing_prevention_reach`

Candidate include set:
- denominator-normalized `annual_hiv_tests_volume`
- `prevention_coverage` or `prevention_access`
- `hiv_knowledge_index`
- optional denominator-normalized `prep_people_receiving`

Explicit excludes:
- `hiv_test_positivity_percent`
- `late_hiv_diagnosis_percent`
- raw counts without denominator normalization
- any direct diagnosis/enrollment head

#### Branch `C`
If the cleaned core does not support a rebuilt testing block, stop making testing-mechanism claims.

Then:
- remove or rename `testing_pulse`
- remove or rename `testing_plateau`
- keep the seeded layer as a bounded care/suppression stress-test sandbox only

### Phase 3: Admit Or Reject Scenario Expansion

Only if Phase 1 and Phase 2 pass:

#### `EXP-P2-GATE-04d`
Anti-circular response gate on the repaired semantics.

Keep if:
- ablation sign agreement `>= 0.75`
- mean abs ratio in `[0.4, 2.0]`

#### `EXP-P2-GATE-05`
Placebo shock audit.

Keep if:
- real kernel materially beats shuffled/placebo kernels

#### `EXP-P2-ADMIT-01/02/03`
Empirical excursion and dwell audit.

Keep if:
- new scenario amplitudes and durations fit observed historical envelopes

Only then:
- run a new scenario library

## Interventions Worth Trying

- archive-matched sidecar ablation
- suppression-sidecar pruning / reweighting
- denominator-normalized testing-volume features
- renamed upstream testing/prevention block
- holdout validation of rebuilt testing block against positivity / late diagnosis as validators, not block inputs

## Interventions To Avoid

- reintroducing `diagnosed_plhiv`, `new_diagnosed_cases_period`, or `alive_on_art` into the structural substrate
- expanding scenario families now
- calling the current substrate a testing mechanism
- hidden-state / HSMM escalation before the cleaned core passes
- “fixing” dead testing scenarios by amplitude inflation alone
- archive-shopping

## Stop / Continue Decision

Continue, but only with the `AB` ladder.

Do not:
- promote the current outcome-lite seeded layer as a general scenario engine
- rebuild testing and archive plumbing in the same experiment

## AutoResearch Handoff

Variant:
- `evidence-to-model-loop`

Evaluation Harness:
- [monthly outcome-lite lane report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260419-s00-outcome-lite/analysis/tr_v3_monthly_phase2_lane_batch_report.md)
- [clean loading sanity report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-loading-sanity-20260419-s01-outcome-lite/analysis/tr_v3_monthly_loading_sanity_batch_report.md)
- [clean seeded gate report](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-gates-20260419-s00-outcome-lite/analysis/tr_v3_phase2_seeded_gate_batch_report.md)

Mutation Units:
- archive-alignment plumbing only
- sidecar ablation only
- testing-block rebuild only
- scenario semantics only

First Experiments:
1. `EXP-P2-GATE-00b`
2. `EXP-P2-GATE-04c`
3. branch decision: `T` or `C`

Stop Rules:
- stop if `EXP-P2-GATE-00b` and `EXP-P2-GATE-04c` both fail
- stop if rebuilt testing still collapses to zero without outcome heads
- stop if placebo kernels perform similarly to the real kernel
