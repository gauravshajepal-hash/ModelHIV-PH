# Phase 3 R92 Process-Repair Experiment Queue

R92 is the next Ralph-loop pass after R91. It tests whether richer train-origin process-repair families can rescue the blocked raw incidence and AIDS-death mechanism claims without using annual validation targets as quarterly truth.

## Result

Overall status:

```text
mortality_process_signal_detected_mechanism_claim_blocked
```

R92 finds a real process signal, but it does not promote a mechanism claim.

## What Was Tested

| Bridge | Target | Proxy | Mechanism Claim Status |
| --- | --- | --- | --- |
| `incidence_proxy_diagnosis_flow_bridge` | annual new infections | quarterly diagnosed-flow support | blocked because direct incidence-process support is still absent |
| `mortality_reported_death_bridge` | annual AIDS deaths | direct reported deaths | blocked because source-family ablation is unstable |

The branch uses train-origin family selection over proxy ratio, proxy AR-ratio, and proxy log-linear target families. Target-only AR readout is retained as a readout diagnostic but excluded from mechanism-eligible selection.

## Key Scores

| Bridge | Mode | Candidate Mean Error | Carry-Forward Mean Error | Candidate Coverage | Carry Coverage | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| diagnosis-flow to annual incidence | mechanism | 0.0497 | 0.3116 | 1.0000 | 0.4286 | strong proxy signal, mechanism still blocked |
| reported deaths to annual AIDS deaths | mechanism | 0.1636 | 0.4764 | 1.0000 | 0.6071 | strong signal, source-family unstable |
| diagnosis-flow to annual incidence | readout | 0.0923 | 0.3116 | 0.9286 | 0.4286 | readout/proxy signal |
| reported deaths to annual AIDS deaths | readout | 0.2772 | 0.4764 | 0.9286 | 0.6071 | readout/proxy signal |

## Blockers

- `direct_incidence_process_support_absent`
- `mortality_process_repair_not_source_family_stable`

## Scientific Interpretation

This is a genuine improvement over R91, but not a publication-grade mechanism win. The diagnosis-flow proxy now tracks annual incidence estimates well under blocked-time scoring, and reported deaths track annual AIDS-death estimates much better than carry-forward. However, the incidence bridge still has no direct incidence-process evidence, and the mortality bridge depends too heavily on monthly DOH archive source-family support.

The correct claim boundary is therefore:

- R92 can be cited as evidence that process-repair families contain useful signal.
- R92 cannot be cited as identified incidence or mortality dynamics.
- R90 remains the controlling publication gate: R86/R88 are scoped annual/readout wins, while raw incidence/death mechanisms remain blocked.

## Experiment Queue

| Priority | Experiment | Purpose |
| ---: | --- | --- |
| 1 | `R92` mortality/incidence process repair | implemented in this pass |
| 2 | `R93` open AEM/Spectrum public comparator | build a reproducible public incumbent comparator because official files are absent |
| 3 | `R94` direct incidence evidence scanner | search for evidence that can unblock direct incidence-process support |
| 4 | `R95` subnational sparse hierarchy | improve regional split stability under national total constraints |
| 5 | `R96` Phase 2 source-stable prior retest | keep determinant knobs sensitivity-only until source-stable bundles survive |
| 6 | `R97` third-95 VL/suppression process gate | strengthen back-half process claims |

## Artifacts

| Artifact | Path |
| --- | --- |
| R92 report | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r92-process-repair-experiment-queue-20260509-s00/analysis/r92_process_repair_experiment_queue_report.json` |
| R92 markdown | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r92-process-repair-experiment-queue-20260509-s00/analysis/r92_process_repair_experiment_queue_report.md` |
| R92 mechanism scores | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r92-process-repair-experiment-queue-20260509-s00/analysis/r92_mechanism_score_rows.csv` |
| R92 source-family ablations | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r92-process-repair-experiment-queue-20260509-s00/analysis/r92_source_family_ablation_rows.csv` |

## Next Scientific Step

Run `R93`: build an open, reproducible AEM/Spectrum-style annual public comparator from public incidence, deaths, PLHIV, ART, and cascade sources. This is higher value than tuning the proxy bridge again because the project still cannot make a broad “better than official models” claim without a transparent incumbent comparator.
