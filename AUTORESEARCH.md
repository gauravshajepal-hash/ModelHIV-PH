# AUTORESEARCH

## Variant

Selected variant: `evidence-to-model-loop`

Rationale:
- the current system mixes literature-derived contextual factors, mechanistic cascade fitting, and benchmark comparison in one over-parameterized learner
- local evidence shows the determinant path is inactive, the representation variants are non-differentiating, and the model is mostly interpolating latent monthly trajectories
- the immediate task is not to add more model; it is to build the smallest identifiable national forecast engine that directly attacks diagnosis and linkage error

## Baseline Snapshot

Reference run:
- `artifacts/runs/audit-phase0-reuse-s00-20260331/`

Current benchmark numbers:

| Metric | Value | Source |
|---|---:|---|
| Current winner MAE | `0.089551` | [representation_tournament_summary.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest_tournament/representation_tournament_summary.json) |
| Carry-forward MAE | `0.063217` | [validation_artifact.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest_clumped_baseline/validation_artifact.json) |
| Simple compartmental MAE | `0.063741` | [validation_artifact.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest_clumped_baseline/validation_artifact.json) |
| Diagnosis-flow mean absolute error | `0.016117` | [validation_artifact.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest_clumped_baseline/validation_artifact.json) |
| Diagnosis-flow max absolute error | `0.027970` | [validation_artifact.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest_clumped_baseline/validation_artifact.json) |
| Active diagnosis-flow penalty scale | `0.0` | [hiv.py:329](/D:/EpiGraph_PH/src/epigraph_ph/plugins/hiv.py#L329), [hiv.py:361](/D:/EpiGraph_PH/src/epigraph_ph/plugins/hiv.py#L361) |
| Direct monthly support fraction for core targets | about `0.005` | [mixed_frequency_observation_summary.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest_clumped_baseline/mixed_frequency_observation_summary.json) |

Current failure markers:
- determinant path rejected: [determinant_modifiers.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest_clumped_baseline/determinant_modifiers.json)
- representation variants collapsed: [phase3_hypothesis_experiments.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_experiments/phase3_hypothesis_experiments.json)
- monthly path underconstrained: [mixed_frequency_observation_summary.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest_clumped_baseline/mixed_frequency_observation_summary.json)

## Directory Convention

Every reset experiment must write to:

`artifacts/runs/<run_id>/phase3_national_reset/<experiment_id>/`

Required filenames:
- `experiment_spec.json`
- `observation_table.json`
- `fit_artifact.json`
- `evaluation.json`
- `baseline_comparison.json`
- `diagnosis_flow_evaluation.json`
- `decision.json`

If arrays are emitted, use:
- `state_estimates.npz`
- `forecast_states.npz`

If charts are emitted, use:
- `mae_summary.png`
- `diagnosis_flow_fit.png`

## Run ID Convention

Use:

`nr-<YYYYMMDD>-s<seed>-<experiment_id>`

Examples:
- `nr-20260401-s00-NR-00-observation-table`
- `nr-20260401-s00-NR-01-national-uda-baseline`

## Code Targets

New reset path:
- [src/epigraph_ph/phase3/national_reset_pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/national_reset_pipeline.py)
- [src/epigraph_ph/phase3/national_reset_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/national_reset_core.py)
- [tests/test_phase3_national_reset_pytest.py](/D:/EpiGraph_PH/tests/test_phase3_national_reset_pytest.py)

Existing code to freeze, not extend during reset:
- [src/epigraph_ph/phase3/rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py)
- [src/epigraph_ph/plugins/hiv.py](/D:/EpiGraph_PH/src/epigraph_ph/plugins/hiv.py)

Existing evidence source to reuse:
- [historical_metric_rows.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/harp_archive/historical_metric_rows.json)

## Reset Model Scope

Phase 1 model:
- states: `U`, `D`, `A`
- latent national PLHIV size
- transitions: `U -> D`, `D -> A`
- primary observations:
  - `new_diagnosed_cases_period`
  - `diagnosed_plhiv`
  - `alive_on_art`
- weak auxiliary observations:
  - `advanced_hiv_cases_period`
  - `median_cd4_at_enrollment`

Not allowed in Phase 1:
- determinant modifiers as active forecast covariates
- province archetypes
- region/province hierarchy
- metapopulation and network-family pressure terms
- representation tournament variants
- full back-half cascade detail
- subgroup microstructure beyond the reset plan

## Experiment Registry

### NR-00-observation-table

Objective:
- construct the quarterly national reset table from OCR/HARP metrics

Inputs:
- `new_diagnosed_cases_monthly`
- `new_diagnosed_cases_period`
- `diagnosed_plhiv`
- `alive_on_art`
- `advanced_hiv_cases_period`
- `median_cd4_at_enrollment`

Output path:
- `artifacts/runs/<run_id>/phase3_national_reset/NR-00-observation-table/`

Required artifacts:
- `observation_table.json`
- `experiment_spec.json`
- `decision.json`

Pass criteria:
- every quarter from `2022-Q1` through the latest available archive quarter appears exactly once
- `diagnosed_plhiv` and `alive_on_art` coverage is at least `12` quarterly points each
- `new_diagnosed_cases_period` coverage is at least `8` quarterly points
- missingness report is explicit for `advanced_hiv_cases_period` and `median_cd4_at_enrollment`
- provenance is preserved as `source_ids` and `source_labels`

Fail criteria:
- duplicate quarter rows
- silently filled values without provenance
- mixed monthly/quarterly units without normalization metadata

Decision rule:
- keep only if the table is reproducible from archive data alone with no manual patching

### NR-01-national-uda-baseline

Objective:
- fit the smallest national count-space `U/D/A` model

Output path:
- `artifacts/runs/<run_id>/phase3_national_reset/NR-01-national-uda-baseline/`

Required artifacts:
- `fit_artifact.json`
- `evaluation.json`
- `baseline_comparison.json`
- `diagnosis_flow_evaluation.json`
- `state_estimates.npz`
- `forecast_states.npz`
- `decision.json`

Primary metrics:
- holdout MAE on:
  - `new_diagnosed_cases_period`
  - `diagnosed_plhiv`
  - `alive_on_art`
- holdout SMAPE on the same panel

Pass criteria:
- mean holdout MAE is strictly less than `0.063217`
- mean holdout MAE is strictly less than `0.063741`
- diagnosis-flow mean absolute error is strictly less than `0.016117`
- diagnosis-flow mean absolute error improves by at least `25%` versus baseline
- no single holdout year has MAE worse than current baseline model MAE `0.089551`

Fail criteria:
- loses to carry-forward
- loses to simple compartmental
- improves aggregate MAE but diagnosis-flow MAE does not improve
- relies on external retrospective `estimated_plhiv` as direct training truth

Decision rule:
- revert if any pass criterion fails

### NR-02-delay-aux

Objective:
- add weak diagnosis-delay regularization using advanced disease and CD4 signals

Additions:
- `advanced_hiv_cases_period`
- `median_cd4_at_enrollment`

Output path:
- `artifacts/runs/<run_id>/phase3_national_reset/NR-02-delay-aux/`

Required artifacts:
- all `NR-01` artifacts
- `delay_aux_summary.json`

Pass criteria:
- mean holdout MAE does not regress relative to `NR-01`
- diagnosis-flow mean absolute error improves by at least `10%` relative to `NR-01`
- auxiliary penalties remain weak; they must not dominate the loss

Fail criteria:
- aggregate MAE improves only by forcing implausible `U -> D` spikes
- auxiliary terms overwhelm primary observations
- CD4 or advanced-HIV missingness is hidden rather than explicit

Decision rule:
- keep only if diagnosis-flow fit improves without degrading stock fit

### NR-03-vl-observation-process

Objective:
- add VL testing and suppression as an observation process, not as hard biological truth

Additions:
- `tested_for_viral_load`
- `virally_suppressed`

Output path:
- `artifacts/runs/<run_id>/phase3_national_reset/NR-03-vl-observation-process/`

Required artifacts:
- all `NR-02` artifacts
- `vl_observation_process.json`

Pass criteria:
- no regression on the `NR-02` front-half panel
- documented suppression and VL-tested series fit improves versus `NR-02`
- the model explicitly represents ascertainment or service-process uncertainty

Fail criteria:
- downstream fit improves while `new_diagnosed_cases_period` regresses
- VL/suppression are treated as clean latent truth with no observation-process distinction

Decision rule:
- keep only if front-half performance survives the added back-half observation process

### NR-04-coarse-subgroups

Objective:
- reintroduce only essential Philippines heterogeneity

Allowed subgroup structure:
- `MSM`, `TGW`, `other`
- `15-24`, `25+`
- `pre-COVID`, `COVID`, `recovery`

Output path:
- `artifacts/runs/<run_id>/phase3_national_reset/NR-04-coarse-subgroups/`

Required artifacts:
- all `NR-03` artifacts
- `subgroup_summary.json`

Pass criteria:
- mean holdout MAE improves relative to `NR-03`
- no subgroup branch is effectively unobserved without strong priors being disclosed
- COVID-era break improves 2020-2021 retrospective fit without degrading 2023-2025 holdout

Fail criteria:
- subgroup structure increases variance without improving holdout performance
- TGW is treated as MSM proxy with no separate behavior or care path
- age split adds parameters without measurable gain

Decision rule:
- keep only if subgroup structure produces real holdout improvement or materially better diagnosis-flow attribution

### NR-05-deferred-complexity-scan

Objective:
- test deferred complexity one unit at a time after `NR-04` passes

This experiment is a gate, not a model.

Output path:
- `artifacts/runs/<run_id>/phase3_national_reset/NR-05-deferred-complexity-scan/`

Required artifacts:
- `complexity_queue.json`
- `decision.json`

Pass criteria:
- every deferred item has a single-experiment plan, one metric target, and one revert condition

Fail criteria:
- any deferred item is reintroduced as part of a bundled change

Decision rule:
- no deferred complexity enters the mainline without a standalone win

## Deferred Complexity Queue

These are not part of the reset baseline.

| Deferred ID | Item | Earliest eligible phase | Pass rule |
|---|---|---|---|
| DX-01 | determinant modifiers as active forecast covariates | after `NR-04` | improves holdout MAE without scaffold fallback |
| DX-02 | province archetypes | after `NR-04` | improves holdout or subgroup attribution without overfitting |
| DX-03 | region/province hierarchy | after `NR-04` | improves subnational validation after national model is stable |
| DX-04 | metapopulation and network-family pressure terms | after `NR-04` | improves out-of-sample performance as a standalone mutation |
| DX-05 | representation tournament variants | after `NR-04` | variants must produce materially different scores |
| DX-06 | full back-half detail | after `NR-03` | improves without regressing front-half metrics |
| DX-07 | subgroup microstructure beyond essential splits | after `NR-04` | improves with explicit observational support |

## Global Keep-or-Revert Rule

Keep a change only if:
- it improves frozen-holdout MAE on the active target panel
- it improves or preserves diagnosis-flow fit
- it does not depend on retrospective external denominators as leaked training truth
- it does not hide missing data behind interpolation

Revert a change if:
- it only improves in-sample metrics
- it improves ratio metrics while count-space metrics worsen
- it reactivates dead complexity that the scaffold gate already showed to be useless
- it cannot be identified from the current observation table

## Immediate Operating Rules

- quarterly first, monthly second
- national first, province later
- front half first, back half later
- direct observations first, literature-derived priors second
- one mutation unit per experiment

## Execution Order

1. `NR-00-observation-table`
2. `NR-01-national-uda-baseline`
3. `NR-02-delay-aux`
4. `NR-03-vl-observation-process`
5. `NR-04-coarse-subgroups`
6. `NR-05-deferred-complexity-scan`

## Explicit Non-Goals For The Reset

- do not extend the current representation tournament
- do not turn determinant silos back on in the baseline path
- do not reintroduce province-level machinery before national success
- do not claim AEM/Spectrum superiority without a frozen-vintage comparator
