# Phase 3 R93 Open Public Incumbent Comparator

R93 formalizes the annual public incumbent comparison. It does not create official Philippines AEM/Spectrum output. Instead, it freezes the promoted R78 public-domain annual proxy as an open AEM/Spectrum-style incumbent comparator for annual incidence, AIDS deaths, and estimated PLHIV.

## Result

Overall status:

```text
public_incumbent_comparator_ready_model_blocked
```

Annual superiority status:

```text
blocked_by_open_public_incumbent
```

## What Was Tested

| Component | Role |
| --- | --- |
| `R78 public_train_selected_annual_proxy_v2` | open public annual incumbent comparator |
| `R79 matched gate` | same blocked annual splits comparing Phase 3 annual head versus R78 incumbent |
| annual targets | validation-only public annual incidence, AIDS deaths, estimated PLHIV |

The comparator covers all required annual public metrics:

| Metric | Target Years |
| --- | ---: |
| annual AIDS deaths | 15 |
| annual new infections | 15 |
| estimated PLHIV | 15 |

## Key Scores

| Comparison | Mean Normalized Error | Interval Coverage |
| --- | ---: | ---: |
| Current matched Phase 3 annual head | 0.2435 | 0.7500 |
| Open public incumbent | 0.1638 | 0.9286 |
| Delta model minus incumbent | 0.0797 | -0.1786 |

Metric-level matched comparison:

| Metric | Model Mean Error | Incumbent Mean Error | Delta |
| --- | ---: | ---: | ---: |
| annual AIDS deaths | 0.3947 | 0.3808 | 0.0139 |
| annual new infections | 0.2679 | 0.0427 | 0.2251 |
| estimated PLHIV | 0.0679 | 0.0679 | 0.0000 |

## Scientific Interpretation

R93 is an important negative/adjudication result. It means the project now has a transparent public incumbent comparator, but the current Phase 3 annual head does not beat it. The blocker is mainly annual new infections; PLHIV is tied and AIDS deaths are close, but incidence remains much better tracked by the public annual incumbent.

Allowed claim:

- The repository has an open annual incumbent comparator that can be used while official Philippines AEM/Spectrum outputs are absent.

Blocked claim:

- The current model is broadly better than AEM/Spectrum-style annual incumbents.

## Artifacts

| Artifact | Path |
| --- | --- |
| R93 report | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r93-open-public-incumbent-comparator-20260510-s00/analysis/r93_open_public_incumbent_comparator_report.json` |
| R93 markdown | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r93-open-public-incumbent-comparator-20260510-s00/analysis/r93_open_public_incumbent_comparator_report.md` |
| R93 incumbent metric rows | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r93-open-public-incumbent-comparator-20260510-s00/analysis/r93_incumbent_metric_rows.csv` |
| R93 matched metric rows | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r93-open-public-incumbent-comparator-20260510-s00/analysis/r93_matched_metric_rows.csv` |

## Next Scientific Step

Build R94 as a targeted incidence evidence scanner and incidence-readout repair gate. R93 localizes the public-incumbent failure to annual new infections, so the next pass should not tune all streams blindly. It should search for direct incidence-adjacent evidence and test whether the diagnosis-flow/process-repair signal can improve incidence without converting validation-only annual estimates into training truth.
