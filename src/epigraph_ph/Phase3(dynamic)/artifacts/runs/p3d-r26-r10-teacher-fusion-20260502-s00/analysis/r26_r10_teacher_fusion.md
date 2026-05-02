# Phase 3 R26 R10-Teacher Fusion Diagnostic

Generated: 2026-05-02T11:47:32.873935+00:00

## Verdict

R26 does not beat matched R10 on every tested horizon. Teacher fusion is diagnostic only and should not replace the R19 mechanistic reference.

## Horizon Scores

| Horizon | Fusion | Matched R10 | Carry-forward | Fusion minus R10 | Fusion minus carry | Status | Blockers |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.109426 | 0.095303 | 0.141370 | 0.014123 | -0.031943 | `fail` | `fusion_not_better_than_matched_r10` |
| 3 | 0.140489 | 0.115651 | 0.189677 | 0.024838 | -0.049188 | `fail` | `fusion_not_better_than_matched_r10` |
| 5 | 0.176493 | 0.129421 | 0.229520 | 0.047071 | -0.053027 | `fail` | `fusion_not_better_than_matched_r10` |

## R10 Replay Reproduction Check

| Horizon | Rescored R10 | Replay reference | Absolute delta |
|---:|---:|---:|---:|
| 1 | 0.095303 | 0.095303 | 0.000000 |
| 3 | 0.115651 | 0.115651 | 0.000000 |
| 5 | 0.129421 | 0.129421 | 0.000000 |

## Contract

- R26 uses frozen horizon-matched R10 replay artifacts and never reads future splits when choosing a policy for a split.
- Policies are selected from previous blocked replay rows only.
- This is a predictive teacher-fusion diagnostic, not a mechanistic cascade claim.
- If it fails to beat matched R10, the remaining blocker is the R10 endpoint frontier itself, not only R19 implementation details.
