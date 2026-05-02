# Phase 3 R29 Strict-Ledger Matched R10 Gate

Generated: 2026-05-02T12:06:14.349887+00:00

## Verdict

R29 strict-ledger gate fails. Even after excluding unmapped R10 rows, Phase 3 does not beat strict mapped R10 on every required route.

## Candidate Gate Rows

| Candidate scope | Ref scope | Horizon | Candidate | Strict R10 | Delta | Carry | Status | Blockers |
|---|---|---:|---:|---:|---:|---:|---|---|
| full_sentinel_r19 | all_mapped | 1 | 0.074729 | 0.100728 | -0.025999 | 0.286974 | `pass` | `` |
| full_sentinel_r19 | all_mapped | 3 | 0.111969 | 0.117100 | -0.005131 | 0.527872 | `pass` | `` |
| full_sentinel_r19 | all_mapped | 5 | 0.137054 | 0.126811 | 0.010243 | 0.838764 | `fail` | `candidate_not_better_than_strict_mapped_r10` |
| program_best_phase3 | program_mapped | 3 | 0.142604 | 0.133483 | 0.009122 | 0.536595 | `fail` | `candidate_not_better_than_strict_program_r10` |
| program_best_phase3 | program_mapped | 5 | 0.179700 | 0.131356 | 0.048344 | 0.967184 | `fail` | `candidate_not_better_than_strict_program_r10` |

## Strict R10 References

| Scope | Horizon | Entries | Mapped share | Strict R10 | Carry | R10 better share |
|---|---:|---:|---:|---:|---:|---:|
| all_mapped | 1 | 79 | 0.681034 | 0.100728 | 0.120720 | 0.582278 |
| program_mapped | 1 | 63 | 0.681034 | 0.115953 | 0.133405 | 0.571429 |
| all_mapped | 3 | 204 | 0.649682 | 0.117100 | 0.204909 | 0.642157 |
| program_mapped | 3 | 156 | 0.649682 | 0.133483 | 0.230283 | 0.653846 |
| all_mapped | 5 | 295 | 0.627660 | 0.126811 | 0.255071 | 0.650847 |
| program_mapped | 5 | 217 | 0.627660 | 0.131356 | 0.287303 | 0.686636 |

## Annual Gate

- Status: `pass`
- Decision: `promote_for_next_wave`
- Candidate MAE: `0.396656`
- Carry-forward MAE: `0.592472`
- Conservation residual: `0.000000`

## Contract

- R29 is a strict-ledger rescore of the matched-R10 gate, not a new model.
- `all_mapped` excludes R10 target rows that cannot be mapped to active ObservationRoleLedger provenance.
- `program_mapped` additionally restricts the R10 reference to DOH HARP program lineages.
- Promotion requires the Phase 3 candidate to beat carry-forward, strict mapped R10, and the official annual gate.
