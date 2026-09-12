# Phase 3 R28 Matched R10 Contract Lineage Audit

Generated: 2026-05-02T12:00:46.398912+00:00

## Verdict

R28 is an audit-only artifact. It localizes the matched-R10 blocker to the listed metric/horizon/lineage cells; no model is promoted.

## R10 Reference Rows

| Horizon | Reference | R10 MAE | Carry MAE | Splits |
|---:|---|---:|---:|---:|
| 1 | `EXP-R10-DENSE-M1-C1-H1` | 0.095303 | 0.141370 | 11 |
| 3 | `EXP-R10-DENSE-M1-C1-H1` | 0.115651 | 0.189677 | 11 |
| 5 | `EXP-R10-DENSE-M1-C1-H1` | 0.129421 | 0.229520 | 11 |

## Largest Phase3 vs R10 Metric Gaps

| Scope | Horizon | Metric | Phase3 | R10 | Delta |
|---|---:|---|---:|---:|---:|
| program | 5 | `alive_on_art` | 0.176892 | 0.081262 | 0.095631 |
| all | 5 | `alive_on_art` | 0.131426 | 0.081262 | 0.050164 |
| all | 5 | `new_diagnosed_cases_period` | 0.235106 | 0.202399 | 0.032707 |
| program | 5 | `new_diagnosed_cases_period` | 0.224672 | 0.202399 | 0.022273 |
| all | 3 | `alive_on_art` | 0.097544 | 0.077951 | 0.019592 |
| program | 3 | `alive_on_art` | 0.094132 | 0.077951 | 0.016180 |
| all | 1 | `new_diagnosed_cases_period` | 0.133174 | 0.142557 | -0.009383 |
| all | 3 | `new_diagnosed_cases_period` | 0.166907 | 0.185435 | -0.018528 |

## R10 Error By Metric And Horizon

| Horizon | Metric | Entries | R10 | Carry | R10 minus carry | R10 better share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `alive_on_art` | 43 | 0.071027 | 0.117449 | -0.046422 | 0.697674 |
| 1 | `diagnosed_plhiv` | 35 | 0.075718 | 0.052735 | 0.022982 | 0.514286 |
| 1 | `new_diagnosed_cases_period` | 38 | 0.142557 | 0.257345 | -0.114788 | 0.736842 |
| 3 | `alive_on_art` | 117 | 0.077951 | 0.155492 | -0.077541 | 0.717949 |
| 3 | `diagnosed_plhiv` | 94 | 0.103367 | 0.095156 | 0.008211 | 0.510638 |
| 3 | `new_diagnosed_cases_period` | 103 | 0.185435 | 0.351467 | -0.166032 | 0.786408 |
| 5 | `alive_on_art` | 175 | 0.081262 | 0.176585 | -0.095323 | 0.748571 |
| 5 | `diagnosed_plhiv` | 140 | 0.146546 | 0.132641 | 0.013905 | 0.378571 |
| 5 | `new_diagnosed_cases_period` | 155 | 0.202399 | 0.427960 | -0.225561 | 0.819355 |

## Dominant Target Lineages

| Horizon | Source lineage | Entries | R10 | Carry | R10 minus carry |
|---:|---|---:|---:|---:|---:|
| 5 | `unknown_source_tier|unknown_measurement_class|unknown_series_kind` | 175 | 0.163999 | 0.231771 | -0.067773 |
| 5 | `official_doh_archive|program_observed_harp|quarterly_snapshot` | 135 | 0.124994 | 0.351394 | -0.226400 |
| 3 | `unknown_source_tier|unknown_measurement_class|unknown_series_kind` | 110 | 0.127712 | 0.195790 | -0.068078 |
| 3 | `official_doh_archive|program_observed_harp|quarterly_snapshot` | 81 | 0.096668 | 0.274010 | -0.177343 |
| 5 | `official_doh_archive|program_observed_harp|monthly_snapshot` | 79 | 0.141963 | 0.185186 | -0.043223 |
| 5 | `official_user_provided_slide|program_observed_harp|annual_snapshot` | 78 | 0.114166 | 0.165400 | -0.051234 |
| 3 | `official_doh_archive|program_observed_harp|monthly_snapshot` | 72 | 0.172385 | 0.185318 | -0.012933 |
| 3 | `official_user_provided_slide|program_observed_harp|annual_snapshot` | 48 | 0.063854 | 0.122443 | -0.058589 |
| 1 | `unknown_source_tier|unknown_measurement_class|unknown_series_kind` | 37 | 0.085510 | 0.192927 | -0.107417 |
| 1 | `official_doh_archive|program_observed_harp|monthly_snapshot` | 35 | 0.160444 | 0.154397 | 0.006047 |

## Provenance Coverage

| Horizon | Entries | Unmapped entries | Unmapped share |
|---:|---:|---:|---:|
| 1 | 116 | 37 | 0.318966 |
| 3 | 314 | 110 | 0.350318 |
| 5 | 470 | 175 | 0.372340 |

## Contract

- R28 is not a model branch and cannot promote a champion.
- It reads the frozen horizon-matched R10 replay artifacts and joins scored target rows back to the active ObservationRoleLedger-derived provenance.
- Rows reported as unmapped could not be traced to an active strict-ledger metric provenance row for the same quarter and metric.
- Phase3-vs-R10 metric comparison is labeled approximate because R13 Phase3 rows and dense R10 replay rows do not have identical split scopes.
- The audit is intended to decide whether the next useful action is model dynamics, observation-lineage stratification, or benchmark-contract revision.
