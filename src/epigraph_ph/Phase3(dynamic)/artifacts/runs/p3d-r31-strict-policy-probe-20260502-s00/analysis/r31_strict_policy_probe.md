# Phase 3 R31 Strict Policy Probe

Generated: 2026-05-02T12:14:25.052553+00:00

## Verdict

R31 confirms diagnosis-flow is repairable by train-only velocity, but ART remains worse than strict program R10.

## Best Train-Only Policy By Cell

| Scope | Horizon | Metric | Policy | Policy MAE | Strict R10 | Delta | Status |
|---|---:|---|---|---:|---:|---:|---|
| all_mapped | 3 | `alive_on_art` | `median_velocity` | 0.121648 | 0.071148 | 0.050500 | `fail` |
| all_mapped | 3 | `new_diagnosed_cases_period` | `positive_velocity` | 0.129255 | 0.172982 | -0.043727 | `pass` |
| all_mapped | 5 | `alive_on_art` | `median_velocity` | 0.129992 | 0.058593 | 0.071399 | `fail` |
| all_mapped | 5 | `new_diagnosed_cases_period` | `positive_velocity` | 0.161279 | 0.182487 | -0.021208 | `pass` |
| program_mapped | 3 | `alive_on_art` | `median_velocity` | 0.135068 | 0.085470 | 0.049599 | `fail` |
| program_mapped | 3 | `new_diagnosed_cases_period` | `positive_velocity` | 0.129255 | 0.172982 | -0.043727 | `pass` |
| program_mapped | 5 | `alive_on_art` | `median_velocity` | 0.121335 | 0.054822 | 0.066512 | `fail` |
| program_mapped | 5 | `new_diagnosed_cases_period` | `positive_velocity` | 0.161279 | 0.182487 | -0.021208 | `pass` |

## Contract

- R31 is a train-only policy probe, not a promoted model.
- It tests simple non-handwritten time-series policies on the same strict mapped R10 target rows.
- The purpose is to separate a diagnosis-flow repair opportunity from the harder ART trajectory blocker.
