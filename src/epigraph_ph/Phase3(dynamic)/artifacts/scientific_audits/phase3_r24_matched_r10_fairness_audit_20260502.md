# Phase 3 R24 Matched-R10 Fairness and Readout Audit

Generated: 2026-05-02

## Verdict

R19 remains the active mechanistic reference because it beats carry-forward and the conserved annual official-style gate, but it has not achieved the active goal: it still fails the matched R10 endpoint/readout benchmark on the long-horizon program route and narrowly at 5y on the full sentinel.

The audit changes the next experiment choice. More deterministic R19/R16 process selector variants are low value. The evidence says matched R10 is a direct endpoint/readout forecast family over a narrower metric scope, while R19 is a process-family cascade model. If beating matched R10 remains mandatory, the next branch must be an explicitly labeled predictive readout layer on top of the conserved state model, not another hidden process knob pretending to be mechanistic.

## Source Artifacts

- `r13_final_results`: `/home/gaurav/codex_work/ModelHIV-PH/src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r19-joint-service-r13-queue-20260502-final-v3/analysis/r13_priority_experiment_results.json`
- `matched_r10_horizon_replay`: `/home/gaurav/codex_work/ModelHIV-PH/src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r12-joint-annual-latent-monthly-20260501-s00/analysis/r10_horizon_matched_replay_report.json`
- `r22_program_diagnostic`: `/home/gaurav/codex_work/ModelHIV-PH/src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r22_program_metric_coupled_diagnostic_results_20260502.json`
- `r23_program_diagnostic`: `/home/gaurav/codex_work/ModelHIV-PH/src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r23_recent_origin_program_coupling_diagnostic_results_20260502.json`
- `r10_winner_documentation`: `/home/gaurav/codex_work/ModelHIV-PH/src/epigraph_ph/Phase3(dynamic)/R10_WINNER_DOCUMENTATION.md`

## Matched R10 Contract

| Horizon | R10 reference | Carry-forward | Reference experiment | Metric scope |
|---:|---:|---:|---|---|
| 1y | 0.095303 | 0.141370 | `EXP-R10-DENSE-M1-C1-H1` | diagnosed_plhiv, alive_on_art, new_diagnosed_cases_period |
| 3y | 0.115651 | 0.189677 | `EXP-R10-DENSE-M1-C1-H1` | diagnosed_plhiv, alive_on_art, new_diagnosed_cases_period |
| 5y | 0.129421 | 0.229520 | `EXP-R10-DENSE-M1-C1-H1` | diagnosed_plhiv, alive_on_art, new_diagnosed_cases_period |

R10 is therefore a valid predictive benchmark, but it is not evidence that the same equations identify cascade transition mechanisms. It forecasts the scored endpoints directly and scores only diagnosed stock, ART stock, and diagnosis flow.

## R19 Annual Gate

| Gate | Decision | Candidate MAE | Carry-forward MAE | Conservation residual |
|---|---|---:|---:|---:|
| official annual conserved-head gate | `promote_for_next_wave` | 0.396656 | 0.592472 | 0.000000 |

## Full Sentinel R19 vs Matched R10

| Horizon | R19 R10-scope | Matched R10 | Delta | Carry-forward | Decision |
|---:|---:|---:|---:|---:|---|
| 1y | 0.074729 | 0.095303 | -0.020575 | 0.286974 | beats R10 |
| 3y | 0.111969 | 0.115651 | -0.003682 | 0.527872 | beats R10 |
| 5y | 0.137054 | 0.129421 | 0.007633 | 0.838764 | fails R10 |

## Program Route R19/R22/R23

| Branch | Horizon | Full candidate MAE | R10-scope candidate | Matched R10 | Delta | Carry-forward |
|---|---:|---:|---:|---:|---:|---:|
| R19 | 3y | 0.374070 | 0.142997 | 0.115651 | 0.027347 | 0.536595 |
| R22 | 3y | 0.359755 | 0.148135 | 0.115651 | 0.032484 | 0.536595 |
| R23 | 3y | 0.374208 | 0.142604 | 0.115651 | 0.026954 | 0.536595 |
| R19 | 5y | 0.640929 | 0.185012 | 0.129421 | 0.055591 | 0.967184 |
| R22 | 5y | 0.613597 | 0.179700 | 0.129421 | 0.050279 | 0.967184 |
| R23 | 5y | 0.641067 | 0.184619 | 0.129421 | 0.055198 | 0.967184 |

## Scientific Interpretation

- R19 is scientifically better than R18 and passes the annual conserved-head challenge, so it remains useful as a mechanistic research reference.
- R22 proves service/back-half signal exists in the earlier support-cadence branch, but it worsens the h3 R10-scope ART path.
- R23 proves a recent-origin guard can prevent that h3 ART regression, but the improvement is too small and it gives up the full program-route service gain.
- The remaining R10 gap is not mainly a missing VL/suppression stock-flow equation; it is the advantage of an endpoint/readout family over the exact R10 scoring scope.

## Next Experiment Contract

R24 should not be promoted as a model. It is a fairness/readout audit. The next model branch, if the active goal still requires beating matched R10, should be a two-track hybrid:

- Mechanistic track: keep R19 conserved states, annual mass balance, stock cone, and conditional VL/suppression gates.
- Predictive track: add a train-origin endpoint readout head for the R10 metric scope only, labeled as predictive readout rather than transition mechanism.
- Promotion rule: the predictive head must beat carry-forward and matched R10 on the same horizon-matched gate, while the mechanistic state remains valid under annual conservation and stock-cone checks.

Ralph check: helpful. This audit stops the loop from spending more budget on low-yield selector variants and defines the only next branch that directly attacks the active blocker without mislabeling readout performance as mechanism.
