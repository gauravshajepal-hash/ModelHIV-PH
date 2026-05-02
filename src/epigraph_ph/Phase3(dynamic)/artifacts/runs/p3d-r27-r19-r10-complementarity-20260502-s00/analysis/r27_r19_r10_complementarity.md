# Phase 3 R27 R19/R10 Complementarity Diagnostic

Generated: 2026-05-02T11:47:18.049941+00:00

## Verdict

R27 oracle complementarity does not beat matched R10 in every audited route. Even choosing the better of Phase3/R19-family and R10 per horizon cannot clear the program route or full h5 gate.

## Horizon Scores

| Scope | Horizon | Fusion | Matched R10 | Phase3/R19 | Carry-forward | Fusion minus R10 | Fusion minus Phase3 | Status | Blockers |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| full_sentinel_r19 | 1 | 0.074729 | 0.095303 | 0.074729 | 0.286974 | -0.020575 | 0.000000 | `pass` | `` |
| full_sentinel_r19 | 3 | 0.111969 | 0.115651 | 0.111969 | 0.527872 | -0.003682 | 0.000000 | `pass` | `` |
| full_sentinel_r19 | 5 | 0.129421 | 0.129421 | 0.137054 | 0.838764 | 0.000000 | -0.007633 | `fail` | `oracle_not_strictly_better_than_matched_r10` |
| program_best_phase3 | 3 | 0.115651 | 0.115651 | 0.142604 | 0.536595 | 0.000000 | -0.026954 | `fail` | `oracle_not_strictly_better_than_matched_r10` |
| program_best_phase3 | 5 | 0.129421 | 0.129421 | 0.179700 | 0.967184 | 0.000000 | -0.050279 | `fail` | `oracle_not_strictly_better_than_matched_r10` |

## Contract

- R27 fast mode reads the locked R24 fairness audit and computes an oracle upper bound over Phase3/R19-family versus matched R10.
- Because this oracle sees horizon-level outcomes, it is not a promotable model; it only decides whether a dense learned selector is worth running.
- If the oracle cannot strictly beat matched R10, a train-origin selector using less information should not be expected to beat it.
- The expensive identical-row replay remains available behind `--dense-identical-row-replay` but is not the default path.
