# Phase 3 R19 Lineage State-Space Diagnostic, 2026-05-02

## Verdict
The lineage-aware third-95 rate state branch is **diagnostic only**, not a promoted reference. It keeps the annual AEM/Spectrum-style challenge unchanged, but it worsens the R13-006 program-route VL/suppression count score versus the current R19 joint-service branch.

## Locked Evidence
- R19 reference run: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r19-joint-service-r13-queue-20260502-final-v3/analysis/r13_priority_experiment_results.json`
- R19 lineage diagnostic smoke run: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r19-lineage-state-smoke-20260502-v2/analysis/r13_priority_experiment_results.json`
- Dashboard: `src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r19_lineage_state_diagnostic_dashboard_20260502.png`

## Key Scores
| Gate | Branch | Candidate mean MAE | Carry-forward mean MAE | Decision | Blockers |
|---|---:|---:|---:|---|---|
| R13-001 annual official-style | R19 joint service | 0.396656 | 0.592472 | promote_for_next_wave | none |
| R13-006 program h3/h5 | R19 joint service | 0.507500 | 0.751890 | keep_as_diagnostic | matched_r10_gate_failed |
| R13-006 program h3/h5 | R19 lineage diagnostic | 0.572787 | 0.751890 | keep_as_diagnostic | matched_r10_gate_failed |
| R13-050 full sentinel | R19 joint service | 0.334614 | 0.551204 | keep_as_diagnostic | matched_r10_gate_failed |

## Scientific Interpretation
The rate-state transfer from survey/astronomy-style lineage calibration is not enough here. It improves the representation of mixed measurement lineages in principle, but the blocked count-space stress test says it pushes the VL/suppression counts away from observed program trajectories. Therefore it should not be allowed into the publication sentinel or horizon selector.

## Action
Keep `r19_joint_service_cascade_process` as the current Phase 3 research reference. Keep `r19_lineage_state_space_back_half_process` as a falsified diagnostic branch for the next council: the remaining failure is not solved by lineage-aware conditional-rate observation alone.
