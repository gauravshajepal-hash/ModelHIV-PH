# Phase 3 R18 Evidence-Backed ART Process Summary

Generated: 2026-05-01

## Goal

R18 tests whether the R17 frozen R10 ART teacher can be replaced by fitted ART process evidence:

- ART initiation from diagnosis flow and diagnosed-not-ART backlog.
- ART retention/removal from train-origin ART stock dynamics.
- Support/reporting availability from monthly program support.
- Program-volume shock from monthly HARP/HASP program evidence.

R18 deliberately does not use frozen R10 replay as a teacher and does not train on holdout targets.

## Final Run

- Run: `p3d-r18-art-process-r13-queue-20260501-final`
- Queue size: 50 locked R13 experiments
- Evidence window: 2010-2025
- Minimum train years: 5
- Dashboard: `phase3_r18_art_process_dashboard_20260501.png`

## Gate Results

| Gate | Decision | Candidate | Comparator | Scientific read |
|---|---:|---:|---:|---|
| R13-001 official annual AEM/Spectrum-style challenge | promote_for_next_wave | 0.396656 | carry-forward 0.592472 | Passes conserved annual incidence/deaths/PLHIV/cascade-style gate. |
| R13-006 program route stress test | promote_for_next_wave | 0.108494 | matched R10 0.122536 | Passes program-route R10 gate over 3y/5y. |
| R13-050 full publication sentinel | keep_as_diagnostic | 0.336848 | carry-forward 0.551204 | Beats carry-forward but fails matched R10 at 5y. |

## R13-050 Horizon Detail

| Horizon | R18 R10-scope MAE | Matched R10 MAE | Delta |
|---:|---:|---:|---:|
| 1y | 0.074972 | 0.095303 | -0.020332 |
| 3y | 0.112556 | 0.115651 | -0.003095 |
| 5y | 0.137641 | 0.129421 | +0.008220 |

## H5 Failure Anatomy

| Metric | Mean normalized error |
|---|---:|
| diagnosed_plhiv | 0.080096 |
| alive_on_art | 0.114620 |
| new_diagnosed_cases_period | 0.223050 |
| tested_for_viral_load | 0.985311 |
| virally_suppressed | 1.102189 |

## Verdict

R18 is scientifically useful but is not a publication-grade champion. It proves that a fitted ART initiation/retention/removal/reporting process can replace part of the R17 ART teacher and still beat carry-forward, but the full sentinel remains blocked by 5-year matched R10 trajectory drift.

The claim should be narrow:

- Keep R18 as a mechanistic falsification/process branch.
- Do not claim SOTA or AEM/Spectrum superiority from R18.
- Keep R17 as a hybrid benchmark only, because it uses a frozen R10 teacher.
- The next model needs a stronger joint long-horizon process over ART trajectory, diagnosis flow, VL testing, and suppression rather than another readout correction.

## Ralph Check

Helpful, not busywork: this branch directly tests the core blocker from R17. The result is negative for SOTA promotion but positive for scientific diagnosis because it localizes the remaining failure to 5-year trajectory dynamics after removing the R10 ART teacher.
