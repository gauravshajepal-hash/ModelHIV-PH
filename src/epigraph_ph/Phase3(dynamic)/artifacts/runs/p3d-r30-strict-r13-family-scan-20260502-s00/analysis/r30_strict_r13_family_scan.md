# Phase 3 R30 Strict R13 Family Scan

Generated: 2026-05-02T12:10:26.205259+00:00

## Verdict

R30 found no already-run R13 family that beats strict mapped R10 on required long-horizon routes.

## Best Full-Sentinel Rows

| Scope | Experiment | Family | Horizons | Max delta | Mean delta | Status | Blockers |
|---|---|---|---|---:|---:|---|---|
| all | `R13-021` | `support_partition_calibration` | h1 | -0.012494 | -0.012494 | `fail` | `missing_required_horizons_h3_h5` |
| all | `R13-049` | `multi_horizon_weighted_process` | h1 | -0.009643 | -0.009643 | `fail` | `missing_required_horizons_h3_h5` |
| all | `R13-024` | `linkage_lag_plus_support_partition` | h1 | 0.010223 | 0.010223 | `fail` | `missing_required_horizons_h3_h5;strict_r10_gate_failed` |
| all | `R13-050` | `r19_joint_service_cascade_process` | h1,h3,h5 | 0.010243 | -0.006963 | `fail` | `strict_r10_gate_failed` |
| all | `R13-023` | `linkage_lag_kernel` | h1 | 0.034912 | 0.034912 | `fail` | `missing_required_horizons_h3_h5;strict_r10_gate_failed` |
| all | `R13-007` | `r12_da_process_split_transition` | h1,h3,h5 | 0.059204 | 0.022612 | `fail` | `strict_r10_gate_failed` |
| all | `R13-008` | `r12_da_residual_source_process` | h1,h3,h5 | 0.060914 | 0.026037 | `fail` | `strict_r10_gate_failed` |
| all | `R13-009` | `era_datv_transition_process` | h1,h3,h5 | 0.061910 | 0.023969 | `fail` | `strict_r10_gate_failed` |
| all | `R13-022` | `transition_shrinkage` | h1 | 0.082275 | 0.082275 | `fail` | `missing_required_horizons_h3_h5;strict_r10_gate_failed` |
| all | `R13-019` | `local_level_filter` | h1 | 0.091625 | 0.091625 | `fail` | `missing_required_horizons_h3_h5;strict_r10_gate_failed` |
| all | `R13-003` | `multi_horizon_weighted_process` | h1,h3,h5 | 0.105597 | 0.042632 | `fail` | `strict_r10_gate_failed` |
| all | `R13-011` | `diagnosis_lag_stock_process` | h1,h3,h5 | 0.115745 | 0.064431 | `fail` | `strict_r10_gate_failed` |

## Best Program Rows

| Scope | Experiment | Family | Horizons | Max delta | Mean delta | Status | Blockers |
|---|---|---|---|---:|---:|---|---|
| program | `R13-026` | `r14_two_factor_program_process` | h1 | -0.054823 | -0.054823 | `fail` | `missing_required_horizons_h3_h5` |
| program | `R13-002` | `r14_two_factor_program_process` | h1 | -0.032359 | -0.032359 | `fail` | `missing_required_horizons_h3_h5` |
| program | `R13-044` | `linkage_lag_kernel` | h1 | -0.002889 | -0.002889 | `fail` | `missing_required_horizons_h3_h5` |
| program | `R13-041` | `support_era_diagnosis_flow_process` | h1 | 0.020479 | 0.020479 | `fail` | `missing_required_horizons_h3_h5;strict_r10_gate_failed` |
| program | `R13-043` | `diagnosed_reporting_bias_process` | h1,h3 | 0.022833 | -0.024827 | `fail` | `missing_required_horizons_h5;strict_r10_gate_failed` |
| program | `R13-006` | `r19_joint_service_cascade_process` | h3,h5 | 0.053655 | 0.031585 | `fail` | `strict_r10_gate_failed` |
| program | `R13-040` | `art_horizon_selector_process` | h1,h3 | 0.090310 | 0.022112 | `fail` | `missing_required_horizons_h5;strict_r10_gate_failed` |
| program | `R13-042` | `stock_flow_reconciliation_process` | h1 | 0.759737 | 0.759737 | `fail` | `missing_required_horizons_h3_h5;strict_r10_gate_failed` |

## Contract

- R30 does not run a new model. It rescans already-run R13 families against the R29 strict-ledger references.
- Program rows must cover h3 and h5; full/all rows must cover h1, h3, and h5.
- A family passes only if every required horizon beats the strict mapped R10 reference.
