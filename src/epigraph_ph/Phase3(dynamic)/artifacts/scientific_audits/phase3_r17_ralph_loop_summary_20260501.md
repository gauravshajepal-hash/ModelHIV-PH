# Phase 3 Dynamic R17 Ralph Loop Summary

Generated: 2026-05-01

## Branch

`R17` adds a joint ART trajectory + diagnosis-flow hybrid process branch on top of the `R16` support-cadence stock backbone.

The branch deliberately targets the remaining R16 failure mode: diagnosed stock was no longer limiting, while `alive_on_art` and `new_diagnosed_cases_period` still drove the 5-year R10-scope failure. R17 therefore keeps the R16 diagnosed-stock backbone and allows a frozen horizon-matched R10 teacher only for the remaining ART/diagnosis-flow readout shape.

## Locked Gate Result

Official queue:

`p3d-r17-art-flow-teacher-r13-queue-20260501-s00`

Result artifact:

`src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r17-art-flow-teacher-r13-queue-20260501-s00/analysis/r13_priority_experiment_results.json`

Dashboard:

`src/epigraph_ph/Phase3(dynamic)/artifacts/scientific_audits/phase3_r17_ralph_loop_dashboard_20260501.png`

## Key Comparisons

### Program Route Gate: R13-006

| Branch | h3 candidate MAE | h3 delta vs matched R10 | h5 candidate MAE | h5 delta vs matched R10 | Decision |
|---|---:|---:|---:|---:|---|
| R16 | 0.087040 | -0.028611 | 0.121413 | -0.008008 | promote_for_next_wave |
| R17 | 0.062097 | -0.053554 | 0.075543 | -0.053878 | promote_for_next_wave |

### Full R10-Scope Publication Gate: R13-050

| Branch | h1 R10-scope MAE | h3 R10-scope MAE | h5 R10-scope MAE | h5 delta vs matched R10 | Decision |
|---|---:|---:|---:|---:|---|
| R16 | 0.064657 | 0.108281 | 0.141972 | +0.012551 | keep_as_diagnostic |
| R17 | 0.059815 | 0.100403 | 0.127608 | -0.001813 | promote_for_next_wave |

R17 closes the R16 5-year R10-scope failure while preserving the 1-year and 3-year R10 wins.

## H5 Full-Cascade Anatomy

| Metric | R16 h5 error | R17 h5 error | R17 minus R16 |
|---|---:|---:|---:|
| diagnosed_plhiv | 0.080096 | 0.080096 | +0.000000 |
| alive_on_art | 0.128328 | 0.082001 | -0.046327 |
| new_diagnosed_cases_period | 0.223050 | 0.226928 | +0.003878 |
| tested_for_viral_load | 0.906001 | 0.900140 | -0.005860 |
| virally_suppressed | 1.015513 | 1.012561 | -0.002952 |

## Scientific Interpretation

R17 is a real gate improvement, but it is not yet a pure mechanistic model victory over R10/AEM/Spectrum. It uses frozen horizon-matched R10 replay as a readout teacher for ART/diagnosis-flow shape, so the defensible claim is narrower:

> A conserved Phase 3 dynamic backbone plus a constrained frozen-teacher ART/diagnosis-flow readout can beat carry-forward and matched R10 on the locked R10-scope trajectory gate.

The non-defensible claim is:

> The current model is a fully independent mechanistic replacement for R10/AEM/Spectrum or a validated third-95 process model.

That stronger claim still requires evidence-backed ART initiation, retention/removal, VL testing, and suppression process heads that can replace the teacher without weakening the stock cone or conditional-rate gates.

## Next Scientific Step

Build `R18` as an evidence-backed ART initiation/retention/removal process branch:

- keep the R17/R16 stock-cone and R10-scope gates frozen;
- replace the ART teacher with explicit `D -> A`, ART retention, ART removal, and reporting-intensity terms;
- keep diagnosis-flow teacher disabled at 5-year horizons unless it passes a non-regression gate;
- require the same R13-050 matched R10 pass before promotion.
