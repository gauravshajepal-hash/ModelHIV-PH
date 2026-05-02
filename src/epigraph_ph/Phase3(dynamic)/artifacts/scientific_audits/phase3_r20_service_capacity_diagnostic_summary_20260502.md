# Phase 3 R20 Service-Capacity Diagnostic, 2026-05-02

## Verdict
R20 is **diagnostic only**, not a promoted publication reference. It adds a train-origin service-capacity process for program diagnosis flow, VL testing, and suppression on top of R19. The v3 selector compares against the actual R19 base and splits diagnosis-flow from back-half service selection, but the locked R13-006 smoke gate still fails matched R10.

## Evidence
- R19 reference: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r19-joint-service-r13-queue-20260502-final-v3/analysis/r13_priority_experiment_results.json`
- R20 smoke: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r20-service-capacity-smoke-20260502-v3/analysis/r13_priority_experiment_results.json`
- Dashboard: `src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r20_service_capacity_diagnostic_dashboard_20260502.png`

## Key Scores
| Gate | Branch | Candidate mean MAE | Carry-forward mean MAE | R10-scope candidate MAE | Decision | Blockers |
|---|---:|---:|---:|---:|---|---|
| R13-001 annual official-style | R20 queue | 0.396656 | 0.592472 | n/a | promote_for_next_wave | none |
| R13-006 program h3/h5 | R19 joint service | 0.507500 | 0.751890 | 0.164004 | keep_as_diagnostic | matched_r10_gate_failed |
| R13-006 program h3/h5 | R20 service capacity | 0.520612 | 0.751890 | 0.188142 | keep_as_diagnostic | matched_r10_gate_failed |

## Interpretation
R20 tests the right mechanism class, but the current service-capacity formulation is not enough. It does not produce a matched-R10-safe diagnosis-flow trajectory, so it should not replace R19 in the publication sentinel. The remaining blocker is still R10-scope trajectory shape, especially diagnosis flow, not just third-95 service capacity.

## Next Scientific Step
Build a diagnosis-flow-specific guarded process that can choose between R19 flow, carry-forward flow, and train-derived reporting/rebound flow under blocked-time non-regression. It must leave D/A stocks unchanged unless a separate stock-consistency gate passes.
