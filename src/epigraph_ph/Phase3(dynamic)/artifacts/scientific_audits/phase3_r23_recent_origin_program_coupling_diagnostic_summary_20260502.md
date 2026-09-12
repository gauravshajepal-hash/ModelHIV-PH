# Phase 3 R23 Recent-Origin Program Coupling Diagnostic, 2026-05-02

## Verdict
R23 is **diagnostic only**. It adds a recent-origin, train-only policy on top of the R19/R16 coupling idea. This prevents the R22 h3 ART regression and slightly improves R10-scope versus R19 at both h3 and h5, but it gives up the full-service trajectory gain and still fails matched R10 by a wide margin.

## Evidence
- Result JSON: `src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r23_recent_origin_program_coupling_diagnostic_results_20260502.json`
- Dashboard: `src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r23_recent_origin_program_coupling_diagnostic_dashboard_20260502.png`
- Replay contract: exact R13-006 program rows, horizons 3 and 5, `rolling_origin_splits`, 2010-2025, min_train_years=5.

## Key Scores
| Horizon | Branch | Candidate MAE | Carry-forward MAE | R10-scope candidate MAE | R10 reference MAE |
|---:|---|---:|---:|---:|---:|
| 3 | R19 | 0.374070 | 0.536595 | 0.142997 | 0.115651 |
| 3 | R22 | 0.359755 | 0.536595 | 0.148135 | 0.115651 |
| 3 | R23 | 0.374208 | 0.536595 | 0.142604 | 0.115651 |
| 5 | R19 | 0.640929 | 0.967184 | 0.185012 | 0.129421 |
| 5 | R22 | 0.613597 | 0.967184 | 0.179700 | 0.129421 |
| 5 | R23 | 0.641067 | 0.967184 | 0.184619 | 0.129421 |

## Interpretation
R22 showed that R16 carries useful service/back-half signal, but R23 shows a lead-aware ART guard cannot convert that signal into a matched-R10 win. The current process-family search has probably exhausted the recoverable R16/R19 coupling signal. The next scientific step is a matched-R10 fairness/readout audit, or a new evidence source for ART/diagnosis trajectory shape rather than more selector variants.
