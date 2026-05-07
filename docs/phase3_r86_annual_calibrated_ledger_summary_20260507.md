# Phase 3 R86 Annual-Calibrated Forecast Grid Ledger

Generated: 2026-05-07

## Claim

R86 is a scoped annual-ledger model win. It preserves the complete quarterly forecast-grid ledger introduced by R85, fits annual weak-measurement heads only on pre-holdout annual rows, and scores held-out annual incidence, AIDS deaths, and estimated PLHIV as validation-only evidence.

This is not a broad claim that raw quarterly incidence or AIDS-death emissions beat official models. It is a leakage-guarded annual public-target gate.

## Gate Result

| Metric | R86 mean normalized error | Carry-forward mean normalized error | Scored entries |
| --- | ---: | ---: | ---: |
| annual new infections | 0.2679 | 0.3116 | 28 / 28 |
| annual AIDS deaths | 0.3947 | 0.4764 | 28 / 28 |
| estimated PLHIV | 0.0633 | 0.3818 | 28 / 28 |
| all required annual heads | 0.2419 | 0.3900 | 84 / 84 |

Interval coverage also improved: R86 `0.7500` versus carry-forward `0.4881`.

## Scientific Limit

R86 wins because annual incidence, AIDS deaths, and PLHIV are calibrated through train-origin weak-measurement heads and then emitted through the complete quarterly ledger. The next required scientific step is to make the raw quarterly incidence and mortality process itself win before annual calibration.

## Local Full Artifact

The full ignored run artifact is produced at:

`src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r86-annual-calibrated-forecast-grid-ledger-20260507-s00/analysis/r86_annual_calibrated_forecast_grid_ledger_report.json`
