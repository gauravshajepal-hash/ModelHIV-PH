# Phase 3 R90 Claim-Grade Gate

R90 is a publication-safety gate over the locked R86, R88, and R89 artifacts. It does not refit a model. It checks whether the current positive annual results can be cited as claim-grade evidence, and whether raw incidence/mortality mechanism claims are admissible.

## Result

| Claim family | R90 status | Scientific meaning |
| --- | --- | --- |
| R86 annual-calibrated ledger | `claim_grade_ready` | Scoped annual/readout win is publication-grade under the active gate |
| R88 guarded annual ledger | `claim_grade_ready` | Conservative annual/readout selector is publication-grade under the active gate |
| R89 incidence/mortality mechanism support | `mechanism_claim_blocked` | Raw incidence/death mechanism claims remain blocked |

Overall R90 status:

```text
claim_grade_annual_readout_ready_mechanisms_blocked
```

## Checks Enforced

R90 requires all of the following before a readout claim is allowed:

- Source artifact exists and is hashable.
- Primary source gate passed.
- Every score row is `validation_only / validation_only`.
- Annual new infections, AIDS deaths, and estimated PLHIV all have complete target support.
- Overall mean normalized error strictly beats carry-forward.
- Overall interval coverage is not worse than carry-forward.
- Every annual metric has mean, p90, and interval coverage non-regression versus carry-forward.
- Every horizon has mean, p90, and interval coverage non-regression versus carry-forward.

R90 additionally requires R89 mechanism support before incidence/death mechanism claims are allowed.

## Failed Mechanism Requirements

R89/R90 block raw incidence and mortality mechanism claims because:

| Failed requirement | Candidate | Reference |
| --- | ---: | ---: |
| direct incidence process support present | 0 | 1 required |
| reported-death bridge mean normalized error beats carry-forward | 0.5529 | 0.4764 |
| reported-death bridge interval coverage non-regression | 0.5000 | 0.6071 |

There were zero validation-role leakage rows in R90.

## Defensible Claim

The defensible current statement is:

> R86 and R88 are claim-grade scoped annual/readout wins against carry-forward on held-out annual incidence, AIDS deaths, and estimated PLHIV targets. They do not identify raw quarterly incidence or mortality mechanisms.

The non-defensible statement remains:

> The current model has solved mechanistic incidence and AIDS-death dynamics or can replace official epidemic models overall.

## Artifacts

| Artifact | Path |
| --- | --- |
| R90 report | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r90-claim-grade-gate-20260509-s00/analysis/r90_claim_grade_gate_report.json` |
| R90 markdown | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r90-claim-grade-gate-20260509-s00/analysis/r90_claim_grade_gate_report.md` |
| R90 requirement rows | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r90-claim-grade-gate-20260509-s00/analysis/r90_requirement_rows.csv` |

## Next Scientific Step

Run `R91` as mechanism-support expansion, not another readout tweak:

1. Add direct or proxy-valid incidence-process evidence if available.
2. Improve reported-death-to-AIDS-death bridge with source-family ablation and train-origin uncertainty.
3. Keep R86/R88 frozen as annual/readout claims until raw incidence/death mechanisms pass an R90-style gate.
