# Phase 3 R21 Diagnosis-Flow Guard Diagnostic, 2026-05-02

## Verdict
R21 is **diagnostic only**, not a promoted publication reference. It tests a guarded train-origin diagnosis-flow selector on top of the R19 joint-service cascade, but the strict v3 smoke gate worsens the R10-comparable program-route score. The active R13-006 slot has therefore been restored to R19; R21 remains only as a lower-priority flow-only diagnostic in the experiment catalog.

## Evidence
- R19 joint service: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r19-joint-service-r13-queue-20260502-final-v3/analysis/r13_priority_experiment_results.json`
- R20 service capacity: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r20-service-capacity-smoke-20260502-v3/analysis/r13_priority_experiment_results.json`
- R21 flow guard: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r21-flow-guard-smoke-20260502-v3/analysis/r13_priority_experiment_results.json`
- R21 catalog guard: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r21-catalog-guard-smoke-20260502/analysis/r13_priority_experiment_results.json`
- Dashboard: `src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r21_diagnosis_flow_guard_diagnostic_dashboard_20260502.png`

## Key Scores
| Gate | Branch | Candidate mean MAE | Carry-forward mean MAE | R10-scope candidate MAE | Decision | Blockers |
|---|---:|---:|---:|---:|---|---|
| R13-001 annual official-style | catalog guard | 0.396656 | 0.592472 | n/a | promote_for_next_wave | none |
| R13-006 program h3/h5 | R19 joint service | 0.507500 | 0.751890 | 0.164004 | keep_as_diagnostic | matched_r10_gate_failed |
| R13-006 program h3/h5 | R20 service capacity | 0.520612 | 0.751890 | 0.188142 | keep_as_diagnostic | matched_r10_gate_failed |
| R13-006 program h3/h5 | R21 flow guard v3 | 0.526970 | 0.751890 | 0.200338 | keep_as_diagnostic | matched_r10_gate_failed |
| R13-006 program h3/h5 | current catalog guard | 0.507500 | 0.751890 | 0.164004 | keep_as_diagnostic | matched_r10_gate_failed |

## Metric-Level Read
| Horizon/metric | R19 candidate error | R21 candidate error | R21 minus R19 |
|---|---:|---:|---:|
| h3 alive_on_art | 0.133504 | 0.133504 | +0.000000 |
| h5 alive_on_art | 0.158473 | 0.158473 | +0.000000 |
| h3 new_diagnosed_cases_period | 0.152491 | 0.254773 | +0.102282 |
| h5 new_diagnosed_cases_period | 0.211550 | 0.254602 | +0.043051 |
| h3 tested_for_viral_load | 0.613251 | 0.613251 | +0.000000 |
| h5 tested_for_viral_load | 1.087903 | 1.087903 | +0.000000 |
| h3 virally_suppressed | 0.678959 | 0.678959 | +0.000000 |
| h5 virally_suppressed | 1.213995 | 1.213995 | +0.000000 |

## Interpretation
R21 did what the contract allowed: it left ART, VL testing, and suppression unchanged and moved only diagnosis flow. The problem is that the moved flow was worse than R19 on both h3 and h5 program-route scoring. That falsifies the immediate hypothesis that a guarded train-origin diagnosis-flow selector closes the matched-R10 gap by itself.

The catalog guard confirms the active queue is not contaminated by this failed branch: R13-006 again runs `r19_joint_service_cascade_process` and reproduces the locked R19 score exactly. R21 should be cited only as a negative diagnostic branch.

## Next Scientific Step
Do not spend more budget on flow-only readout selection. The next useful experiment is a process-level diagnosis-and-ART coupling branch that uses observable monthly transition support, or an evaluation-contract audit of why R10 retains a large advantage on long-horizon program trajectory shape.
