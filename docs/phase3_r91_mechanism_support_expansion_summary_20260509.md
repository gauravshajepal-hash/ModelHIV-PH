# Phase 3 R91 Mechanism Support Expansion Gate

R91 extends the R89/R90 mechanism-support check. It asks whether the active evidence ledger contains any train-origin proxy bridge that can rescue raw incidence or AIDS-death mechanism claims without using validation-only annual targets as training truth.

## Result

Overall status:

```text
mechanism_support_expansion_diagnostic_only
```

R91 does not promote a mechanism claim.

## What Was Tested

| Bridge | Target | Proxy | Use |
| --- | --- | --- | --- |
| `incidence_proxy_diagnosis_flow_bridge` | annual new infections | quarterly diagnosed-flow support | proxy-only incidence diagnostic |
| `mortality_reported_death_bridge` | annual AIDS deaths | direct reported deaths | mortality mechanism-support diagnostic |

Each bridge used train-only policy selection over ratio policies, then scored 2019-2024 annual holdouts across 1y, 3y, and 5y blocked horizons. R91 also reran source-family ablation for the proxy source families.

## Key Scores

| Bridge | Candidate Mean Error | Carry-Forward Mean Error | Candidate Coverage | Carry Coverage | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| diagnosis-flow to annual incidence | 0.4859 | 0.3116 | 0.0714 | 0.4286 | worse than carry-forward |
| reported deaths to annual AIDS deaths | 0.4764 | 0.4764 | 0.6071 | 0.6071 | ties carry-forward, not a win |

## Blockers

- `direct_incidence_process_support_absent`
- `diagnosis_flow_proxy_bridge_not_better_than_carry_forward`
- `diagnosis_flow_proxy_bridge_interval_coverage_worse_than_carry_forward`
- `reported_death_bridge_not_better_than_carry_forward`
- `mortality_bridge_not_source_family_stable`

## Scientific Interpretation

This is a useful negative result. Diagnosis flow is real direct program evidence, but it is not a reliable train-origin proxy for annual incidence under the current public annual target gate. Direct reported deaths are available, but their bridge to annual AIDS-death estimates is not better than carry-forward and becomes worse under source-family ablation.

The R90 claim split therefore remains correct:

- R86/R88 can be cited as scoped annual/readout wins.
- Raw incidence and AIDS-death mechanism claims remain blocked.
- The next model-building step should not tune annual readouts again; it should improve direct process evidence or build a better mortality/reporting observation process.

## Artifacts

| Artifact | Path |
| --- | --- |
| R91 report | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r91-mechanism-support-expansion-gate-20260509-s00/analysis/r91_mechanism_support_expansion_gate_report.json` |
| R91 markdown | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r91-mechanism-support-expansion-gate-20260509-s00/analysis/r91_mechanism_support_expansion_gate_report.md` |
| R91 bridge scores | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r91-mechanism-support-expansion-gate-20260509-s00/analysis/r91_bridge_score_rows.csv` |
| R91 ablation rows | `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r91-mechanism-support-expansion-gate-20260509-s00/analysis/r91_source_family_ablation_rows.csv` |

## Next Scientific Step

Build R92 around evidence acquisition and observation-process modeling:

1. Search public DOH/HARP/HASP sources for stronger monthly/annual direct reported death, AIDS mortality, and incidence-adjacent process evidence.
2. Add a reporting-delay/death-ascertainment observation model for mortality instead of a single ratio bridge.
3. Keep annual incidence as validation-only until direct incidence support or a defensible back-calculation process clears blocked-time gates.
