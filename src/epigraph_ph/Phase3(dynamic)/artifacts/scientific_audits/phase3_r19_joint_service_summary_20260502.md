# Phase 3 R19 Joint Service Cascade Summary

Generated: 2026-05-02

## Goal

R19 tests the next mechanistic repair after R18:

- Keep the R18 evidence-backed ART initiation/retention/removal/reporting process.
- Add long-horizon VL testing as a stock-flow channel constrained by ART stock.
- Add viral suppression as a stock-flow channel constrained by VL-tested stock.
- Add a diagnosis-flow shape process from train-window support/reporting signatures.
- Promote only through blocked train-origin selection and the locked R13/R10 gates.

No frozen R10 teacher, holdout target update, or hand grid is used.

## Final Run

- Run: `p3d-r19-joint-service-r13-queue-20260502-final-v3`
- Queue size: 50 locked R13 experiments
- Evidence root: `/media/gaurav/New_Volume/EpiGraph_PH`
- Working tree: `/home/gaurav/codex_work/ModelHIV-PH`
- Dashboard: `phase3_r19_joint_service_dashboard_20260502.png`

## Gate Results

| Gate | Decision | Candidate | Comparator | Scientific read |
|---|---:|---:|---:|---|
| R13-001 official annual AEM/Spectrum-style challenge | promote_for_next_wave | 0.396656 | carry-forward 0.592472 | Annual conserved-head gate still passes. |
| R13-006 program service stress test | keep_as_diagnostic | 0.507500 | carry-forward 0.751890 | Beats carry-forward but fails matched R10. |
| R13-050 full publication sentinel | keep_as_diagnostic | 0.334614 | carry-forward 0.551204 | Beats carry-forward but fails matched R10 at 5y. |

## R13-050 Horizon Detail

| Horizon | R19 R10-scope MAE | Matched R10 MAE | Delta |
|---:|---:|---:|---:|
| 1y | 0.074729 | 0.095303 | -0.020575 |
| 3y | 0.111969 | 0.115651 | -0.003682 |
| 5y | 0.137054 | 0.129421 | +0.007633 |

## R19 vs R18

| Quantity | R18 | R19 | Delta |
|---|---:|---:|---:|
| Full sentinel mean MAE | 0.336848 | 0.334614 | -0.002234 |
| R10-scope mean MAE | 0.108389 | 0.107917 | -0.000472 |
| H5 tested_for_viral_load | 0.985311 | 0.973919 | -0.011392 |
| H5 virally_suppressed | 1.102189 | 1.099065 | -0.003124 |
| H5 new_diagnosed_cases_period | 0.223050 | 0.221289 | -0.001761 |

## Verdict

R19 is a real mechanistic improvement over R18, but it is not enough for a SOTA or AEM/Spectrum-superiority claim.

The result is scientifically useful because it falsifies a simple explanation: the remaining failure is not solved by adding evidence-backed VL/suppression stock-flow channels on top of R18. The 5-year matched-R10 gap remains, while VL/suppression errors are still large enough to block a full-cascade process claim.

Allowed claim:

- R19 improves the R18 back-half trajectory slightly while preserving the annual conserved-head gate.

Blocked claim:

- R19 does not beat matched R10 at 5y and therefore cannot be claimed to beat AEM/Spectrum-like or SOTA models.

## Next Scientific Step

Build R20 as an observation-lineage state-space model for VL testing and suppression rather than another deterministic stock-flow readout. The R19 failure suggests the third-95 blocker is mixed observation lineage and sparse VL support, not only missing transition dynamics.

Ralph check: helpful. R19 moved the model in the right direction and produced a negative but informative gate result; continuing with more deterministic readout tuning would be busywork.
