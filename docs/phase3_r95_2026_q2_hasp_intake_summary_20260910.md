# Phase 3 R95: 2026-Q2 HASP Intake Gate

R95 ingests the user-provided official `2026_Q2 HIV & AIDS Surveillance of the Philippines.pdf` as a post-Q1 external evidence update. The PDF is copied into the repository under `src/epigraph_ph/harp_archive/HIV_Data/downloaded_hasp/` so the extraction can be replayed.

## Evidence Contract

- Extracted rows: `529`
- Direct-target rows: `194`
- Auxiliary-likelihood rows: `113`
- Prior-context rows: `195`
- Validation-only rows: `27`
- Gate status: `r95_q2_stock_anchor_promoted_diagnosis_flow_shock_flagged`

The Q2 report is scored after the R94 2026-Q1 anchor. It is not allowed to retroactively tune the Q2 holdout.

## Holdout Result

| Metric | 2026-Q2 HASP Actual | Q1-Anchored R41 Forecast | Q1 Carry-forward | R41 Abs Error | Carry Abs Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diagnosed PLHIV | 159,997 | 161,700 | 157,350 | 1,703 | 2,647 |
| Alive on ART | 111,154 | 113,069 | 108,367 | 1,915 | 2,787 |
| Tested for viral load | 65,630 | 64,077 | 61,413 | 1,553 | 4,217 |
| Virally suppressed | 63,850 | 62,123 | 59,540 | 1,727 | 4,310 |
| New diagnoses, Q2 | 2,994 | 5,113 | 4,633 | 2,119 | 1,639 |

Mean normalized error:

| Scope | R41 | Carry-forward | Interpretation |
| --- | ---: | ---: | --- |
| Five scored metrics | 0.0916 | 0.0934 | R41 wins narrowly overall |
| Four stock/back-half metrics | 0.0196 | 0.0433 | R41 wins clearly on stocks |
| Diagnosis flow | 0.3795 | 0.2936 | R41 loses; Q2 flow is shock-flagged |

## Scientific Interpretation

R95 supports using the 2026-Q2 stock row as a future forecast initialization anchor after 2026-Q2. It does not support a clean diagnosis-flow or incidence-process claim, because Q2 new diagnoses fell sharply and the Q1-anchored trajectory over-predicted that flow.

The next defensible modeling step is a reporting/service-intensity shock process for diagnosis flow, evaluated against future holdouts. It should not be trained directly on the Q2 outcome and then claimed as a Q2 forecast win.

## Local Artifacts

- Run report: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r95-2026-q2-hasp-intake-gate-20260910-s00/analysis/r95_2026_q2_hasp_intake_gate_report.json`
- Comparison CSV: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r95-2026-q2-hasp-intake-gate-20260910-s00/analysis/r95_2026_q2_hasp_q2_comparison_rows.csv`
- Dashboard SVG: `src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r95-2026-q2-hasp-intake-gate-20260910-s00/analysis/r95_2026_q2_hasp_q2_error_dashboard.svg`
