# Phase 3 R87/R88 Annual-Ledger Loop

Generated: 2026-05-07

## Result

R87 falsified free raw-emission process rescaling. R88 then promoted a guarded annual-ledger selector.

R88 is a scoped conservative model win: it keeps raw quarterly PLHIV stock emissions because internal train-window evidence supports them, but rejects weak raw incidence and AIDS-death emissions back to a carry-forward prior.

## Gate Result

| Metric | R88 mean normalized error | Carry-forward mean normalized error | Selected behavior |
| --- | ---: | ---: | --- |
| annual new infections | 0.3116 | 0.3116 | carry-forward prior |
| annual AIDS deaths | 0.4764 | 0.4764 | carry-forward prior |
| estimated PLHIV | 0.1048 | 0.3818 | raw quarterly process |
| all annual heads | 0.2976 | 0.3900 | guarded annual model win |

Interval coverage improved from carry-forward `0.4881` to R88 `0.6429`.

## Scientific Interpretation

The win is not coming from identified incidence or mortality dynamics. It is coming from an evidence-gated model-selection rule:

1. Use the raw quarterly process where train rolling-origin evidence beats carry-forward.
2. Fall back to a conservative carry-forward prior where the raw process is weak.
3. Score held-out annual targets as validation-only evidence.

This is defensible for publication as a conservative annual-ledger selector, not as proof that raw incidence or AIDS-death mechanisms are solved.

## Local Full Artifacts

`src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r87-train-backtested-emission-process-calibration-20260507-s00/analysis/r87_train_backtested_emission_process_calibration_report.json`

`src/epigraph_ph/Phase3(dynamic)/artifacts/runs/p3d-r88-guarded-annual-ledger-selector-20260507-s00/analysis/r88_guarded_annual_ledger_selector_report.json`
