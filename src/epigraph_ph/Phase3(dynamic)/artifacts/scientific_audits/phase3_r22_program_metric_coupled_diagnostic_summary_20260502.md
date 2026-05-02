# Phase 3 R22 Program Metric Coupling Diagnostic, 2026-05-02

## Verdict
R22 is **diagnostic only**, not a promoted publication reference. It tests whether restoring the earlier R16 support-cadence program backbone on DOH program-lineage ART, VL, suppression, and diagnosis-flow metrics improves the R19 program route. It improves full program-route MAE at h3/h5 and improves h5 R10-scope error, but it worsens h3 R10-scope error and still fails matched R10.

## Evidence
- Result JSON: `src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r22_program_metric_coupled_diagnostic_results_20260502.json`
- Dashboard: `src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r22_program_metric_coupled_diagnostic_dashboard_20260502.png`
- Replay contract: exact R13-006 program rows, horizons 3 and 5, `rolling_origin_splits`, 2010-2025, min_train_years=5.

## Key Scores
| Horizon | Branch | Candidate MAE | Carry-forward MAE | R10-scope candidate MAE | R10 reference MAE |
|---:|---|---:|---:|---:|---:|
| 3 | R19 | 0.374070 | 0.536595 | 0.142997 | 0.115651 |
| 3 | R22 | 0.359755 | 0.536595 | 0.148135 | 0.115651 |
| 5 | R19 | 0.640929 | 0.967184 | 0.185012 | 0.129421 |
| 5 | R22 | 0.613597 | 0.967184 | 0.179700 | 0.129421 |

## Metric Deltas
| Horizon/metric | R19 error | R22 error | R22 minus R19 |
|---|---:|---:|---:|
| h3 alive_on_art | 0.133504 | 0.144186 | +0.010682 |
| h3 new_diagnosed_cases_period | 0.152491 | 0.152084 | -0.000407 |
| h3 tested_for_viral_load | 0.613251 | 0.581659 | -0.031593 |
| h3 virally_suppressed | 0.678959 | 0.639056 | -0.039903 |
| h5 alive_on_art | 0.158473 | 0.148257 | -0.010216 |
| h5 new_diagnosed_cases_period | 0.211550 | 0.211143 | -0.000407 |
| h5 tested_for_viral_load | 1.087903 | 1.040620 | -0.047283 |
| h5 virally_suppressed | 1.213995 | 1.157865 | -0.056130 |

## Interpretation
R22 confirms that R19 lost some useful R16 service/back-half behavior. The full program-route score improves from `0.374070` to `0.359755` at h3 and from `0.640929` to `0.613597` at h5. However, matched R10 remains far better, and h3 R10-scope regresses from `0.142997` to `0.148135` because ART error worsens.

## Next Scientific Step
The next candidate should not replace all program metrics with R16. It should be a lead-aware ART guard: preserve R19 ART at h3, allow R16-style back-half/service restoration where it improves, and test whether h5 ART can improve without h3 R10-scope regression. If that still fails, the blocker is likely the matched-R10 evaluation/readout advantage rather than recoverable program dynamics.
