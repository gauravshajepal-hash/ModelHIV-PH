# Phase 3 R25 Predictive Endpoint Readout Evaluation

Generated: 2026-05-02T11:22:37.306328+00:00

## Verdict

R25 does not yet satisfy the active goal. It is diagnostic unless all target R25 gates all promote; remaining blockers are recorded in the result table.

## Gate Summary

| Experiment | Scope | Candidate | Carry | Candidate minus carry | R10-scope | Matched R10 | Candidate minus R10 | Decision | Blockers |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `R25-002` | program | 0.643294 | 0.751890 | -0.108596 | 0.263576 | 0.122536 | 0.141041 | `keep_as_diagnostic` | `matched_r10_gate_failed` |
| `R25-003` | all | 0.473754 | 0.551204 | -0.077449 | 0.244224 | 0.113458 | 0.130766 | `keep_as_diagnostic` | `matched_r10_gate_failed` |

## Horizon Detail

| Experiment | Horizon | Candidate | Carry | R10-scope | Matched R10 | Delta R10 |
|---|---:|---:|---:|---:|---:|---:|
| `R25-002` | 3 | 0.428100 | 0.536595 | 0.181521 | 0.115651 | 0.065870 |
| `R25-002` | 5 | 0.858488 | 0.967184 | 0.345632 | 0.129421 | 0.216211 |
| `R25-003` | 1 | 0.256147 | 0.286974 | 0.129344 | 0.095303 | 0.034041 |
| `R25-003` | 3 | 0.435516 | 0.527872 | 0.246386 | 0.115651 | 0.130735 |
| `R25-003` | 5 | 0.729598 | 0.838764 | 0.356942 | 0.129421 | 0.227521 |

## Scientific Contract

- R25 is a predictive endpoint readout branch, not a mechanistic transition-process claim.
- R25 mutates only the R10-comparable endpoint scope: diagnosed stock, ART stock, and diagnosis flow.
- VL testing and suppression are regenerated from the Phase 3 conditional-rate projection after stock-cone projection.
- Matched R10 is used only as an external gate, not as a training target.
