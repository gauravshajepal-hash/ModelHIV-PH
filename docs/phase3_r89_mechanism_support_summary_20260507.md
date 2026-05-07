# Phase 3 R89 Incidence/Mortality Mechanism Support Gate

Generated: 2026-05-07

## Result

R89 is diagnostic-only. It blocks claims that the raw incidence and AIDS-death mechanisms are identified under the current active evidence ledger.

## Evidence Counts

| Evidence stream | Count | Role |
| --- | ---: | --- |
| direct incidence process support (`incident_infections_period`) | 0 | absent |
| annual incidence (`annual_new_infections`) | 15 | validation-only |
| direct reported deaths (`deaths_reported_period`) | 20 | direct target |
| annual AIDS deaths (`annual_aids_deaths`) | 15 | validation-only |

## Mortality Bridge Test

R89 tested whether direct reported-death support can bridge to annual AIDS-death validation better than carry-forward:

| Candidate | Mean normalized error | Interval coverage |
| --- | ---: | ---: |
| reported-death bridge | 0.5529 | 0.5000 |
| carry-forward | 0.4764 | 0.6071 |

The reported-death bridge loses to carry-forward, so mortality-process promotion is blocked.

## Scientific Interpretation

R86 and R88 remain valid scoped annual/readout wins. R89 says something different: the current public/active ledger does not yet support a raw incidence/death mechanism claim.

The next high-value step is evidence expansion, not model tuning: obtain or derive stronger direct incidence proxies and more complete mortality/removal support before attempting another raw mechanism branch.
