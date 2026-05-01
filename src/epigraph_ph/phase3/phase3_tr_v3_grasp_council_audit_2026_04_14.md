# Phase 3 TR-V3 GRASP Council Audit

**Date:** 2026-04-14  
**Question:** scientific audit of [phase3_tr_v3_grasp_next_phase_2026_04_14.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_grasp_next_phase_2026_04_14.md) plus autoreason-backed next experiments  
**Council mode:** multi-agent review with bounded autoreason arbitration  
**Source evidence:** current Phase 3 artifacts, GRASP note, exact/dense evaluation hardening reports

---

## 1. Bottom Line

The GRASP-inspired direction is **interesting but not yet justified as a live model family**.

The council converged on a narrower conclusion:

- keep the current observation-first `R10` line as the production benchmark
- do **not** implement the full lifted-state `R10-G` stack yet
- first prove that shock-like and plateau-like structure exists beyond broad residual bias
- only then try a **minimal shared-shock overlay**

The bounded autoreason result was:

- `A`: implement the memo roughly as written
- `B`: stop and work only on calibration/null-baseline work
- `AB`: run a strict precheck package first, then at most one minimal GRASP-style shock experiment if the prechecks clear

**Winner: `AB`**

That is the scientifically defensible path.

---

## 2. What The Current Evidence Actually Says

### 2.1 Stable facts already supported by the repo

From [exp_eval_01_frozen_protocol_rebuild.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-eval-hardening-20260414-s00/analysis/exp_eval_01_frozen_protocol_rebuild.md), [exp_eval_02_endpoint_tier_era_audit.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-eval-hardening-20260414-s00/analysis/exp_eval_02_endpoint_tier_era_audit.md), and [exp_eval_03_calibration_interval_coverage.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-eval-hardening-20260414-s00/analysis/exp_eval_03_calibration_interval_coverage.md):

- exact winner `EXP-R10-M1-F1` is stable at quarterly mean MAE `0.069471`
- dense winner `EXP-R10-DENSE-M1-H1` is stable at purged-dense quarterly mean MAE `0.085418`
- mechanistic comparator `EXP-R1` is still far behind:
  - exact `0.319273`
  - purged dense `0.358783`
- exact-lane `F1` helps materially
- dense transfer of `F1` fails badly:
  - `EXP-R10-DENSE-M1-F1-H1 = 0.166045`

So the current best line is still observation-first and contract-sensitive.

### 2.2 Weak points that are directly observed, not speculative

- Dense diagnosed-stock calibration is poor.
  - `EXP-R10-DENSE-M1-H1` mean diagnosed residual: `-4924.889`
  - diagnosed interval coverage at `50/80/95`: `0.000 / 0.000 / 0.000`
- Exact-lane uncertainty is also weak on diagnosis flow.
  - `EXP-R10-M1-F1` flow mean residual: `-604.734`
  - flow coverage at `50/80/95`: `0.083 / 0.083 / 0.333`
- Pre-COVID exact evidence does not exist in the exact contract.
  - pre-COVID exact counts are `0` for all three scored metrics
- Dense pre-COVID diagnosed performance is worse than baseline.
  - `EXP-R10-DENSE-M1-H1` pre-COVID diagnosed raw MAE: `5748.088`
  - purged-dense baseline: `1041.605`

These facts matter because they point first to **bias, calibration, and bridge-era reconstruction issues**, not yet to a proven sparse-shock structure.

---

## 3. Scientific Audit Of The GRASP Memo

### 3.1 What is scientifically strong

The memo is directionally right on three points:

1. the next phase should stay on the `R10` observation-first line, not revert to the failed free-hazard branch
2. shocks and plateaus are plausible candidates for explaining residual structure
3. if GRASP is adapted, it should be adapted as a structured optimization pattern, not as a claim that we now have a mechanistic epidemic planner

These points are consistent with the current repo evidence.

### 3.2 What is overstated

The memo currently overstates the proximity between GRASP and this project.

GRASP solves a different core problem:

- learned world model
- virtual states
- action-sequence planning
- long-horizon goal-directed optimization

Current Phase 3 solves:

- aggregate quarterly forecasting over three scored heads
- mixed evidence tiers
- no action sequence
- no validated latent world model

So the honest statement is:

- GRASP is a **useful optimization analogy**
- GRASP is **not** a directly matched scientific model class for the current Phase 3 system

### 3.3 What is unsupported today

The following claims are not yet backed by direct repo evidence:

- that residual errors are concentrated in a small number of true shared shocks
- that plateaus are a better explanation than simple piecewise bias
- that lifted quarterly virtual states add anything beyond a smoother bias corrector
- that stop-gradient style protection is important in this codebase rather than just an imported idea
- that the next publishable novelty is a GRASP-inspired model rather than a benchmark/evidence paper

### 3.4 What would fail peer review if claimed too strongly

- “GRASP solves a problem very close to ours”
- “shock states are interpretable epidemiologic objects”
- “plateaus and backlog releases are already demonstrated in the current data”
- any claim that `R10-G` is a mechanistic epidemic model

---

## 4. Council Consensus

### 4.1 Evidence Agent

Main judgment:

- the live evidence supports a stable `R10` frontier
- it does **not** yet support the shock/plateau interpretation claimed by the memo

### 4.2 Representation Agent

Main judgment:

- if GRASP ideas are used, they should map to a **virtual observation path**
- not to lifted quarterly epidemic compartments
- the smallest credible structure is:
  - base `R10` path
  - one shared low-rank shock factor
  - soft consistency
  - projection back to valid observed space

### 4.3 Evaluation / Failure Agent

Main judgment:

- diagnostics-first is mandatory
- otherwise shocks will simply absorb calibration error or provenance artifacts

### 4.4 Validity Skeptic

Main judgment:

- the main current defect looks like **broad level bias**, especially dense diagnosed stock
- not yet a sparse burst structure
- therefore the full `R10-G` plan should not be trusted yet

---

## 5. What To Borrow From GRASP Right Now

Borrow now:

- sequence-level thinking rather than only one-step recursion
- soft consistency penalties
- low-rank shared residual structure
- bounded sparse shock variables
- train-only regime segmentation

Do not borrow yet:

- fully lifted free quarterly states
- generic latent optimization over the entire path
- stop-gradient rhetoric as if it were already the core technical bottleneck
- action-planning framing
- quarterly `S(t)`
- richer leakage

---

## 6. Autoreason Arbitration

### Option A

Proceed with `EXP-R10-G0-*` and `EXP-R10-G1-*` roughly as written in the memo.

Problem:

- too much of the claim is still conjectural
- too high a chance of fitting bridge-era bias

### Option B

Stop the GRASP line and do only calibration/null-baseline work.

Problem:

- too conservative
- discards the useful hypothesis that shocks and plateaus may still matter after hardening

### Option AB

Run a strict falsification package first, then allow only one minimal shock experiment if the diagnostics clear.

Why it wins:

- it preserves the creative hypothesis
- it forces direct evidence before family expansion
- it keeps the failure mode cheap and interpretable

**Autoreason winner: `AB`**

---

## 7. Frozen Next Experiment Phases

## Phase 0: Required Falsification Checks

These are not optional.

### `EXP-G-CHECK-01`
Residual-to-shock concordance audit.

Question:

- do large residual quarters cluster in a small number of time points
- and are those bursts shared across at least two scored endpoints

Minimum outputs:

- quarter-ranked residual burst table
- cross-endpoint residual concordance
- exact vs purged-dense comparison
- era breakdown

Keep only if:

- bursts are sparse
- shared across endpoints
- not confined to bridge diagnosed rows alone

### `EXP-G-CHECK-02`
Plateau census.

Question:

- do candidate plateaus exist as true low-curvature runs in `D/A/F`
- and are they more than what a simple piecewise smoother would produce

Minimum outputs:

- per-endpoint plateau run lengths
- era-specific plateau counts
- comparison against baseline and against simple piecewise model

Keep only if:

- plateaus are frequent enough to matter
- and not reproducible trivially by naive smoothing

### `EXP-G-NULL-01`
Piecewise/changepoint observation baseline.

This is the main falsification test for GRASP-style novelty.

Model family:

- piecewise linear or fused-lasso observation model on `D/A/F`
- no lifted states
- no sparse shock factors

Keep only if:

- the later GRASP-style overlay beats this null
- otherwise stop the GRASP line

### `EXP-CAL-02`
Dense diagnosed recalibration and interval repair.

Reason:

- current dense diagnosed coverage is `0%` even at 95%
- until that is repaired, “shock discovery” is scientifically weak

Keep only if:

- diagnosed coverage becomes nondegenerate
- and calibration improves without destroying ranking

## Phase 1: Smallest Possible GRASP Adaptation

Run this only if Phase 0 clears.

### `EXP-R10-G1-01-min`
Minimal shared-shock overlay.

Structure:

- start from `EXP-R10-DENSE-M1-H1` and `EXP-R10-M1-F1`
- one shared scalar shock per quarter
- no free per-endpoint shocks
- no free lifted-state path optimization
- no quarterly `S(t)`
- no richer leakage

Interpretation:

- test whether a very small shared residual factor helps after calibration/null checks

Keep only if:

- exact improves below `0.069471` or dense improves below `0.085418`
- or it ties while materially improving diagnosed raw MAE and coverage
- and lockbox remains competitive

### `EXP-R10-G1-02-min`
Plateau penalty on the shared-shock overlay.

Run only if `G1-01-min` lands.

Purpose:

- test whether explicit low-curvature structure helps beyond the shared-shock factor

### `EXP-R10-G1-03-min`
Periodic sync / alternating fit.

Run only if `G1-02-min` lands.

Purpose:

- test whether alternating between base path and shock-adjusted path helps
- not as a grand GRASP claim, but as a bounded optimizer variant

## Phase 2: Sidecars Only

Keep these outside the live benchmark loop.

### `EXP-S1-A2`
Annual susceptible/denominator sidecar.

Purpose:

- denominator sanity and annual coherence

### `EXP-L2-A2`
Late-era suppression-linked leakage residual sidecar.

Purpose:

- diagnostic residual interpretation only

Do not promote either into the live benchmark family unless support changes materially.

---

## 8. Keep / Revert Rules

The GRASP line should stop immediately if any of these happen:

- gains appear only in bridge diagnosed rows
- the piecewise/changepoint null matches the same gains
- dense diagnosed coverage remains degenerate
- the model improves MAE by worsening honesty or lockbox behavior
- the supposed shock factor behaves like a broad level-bias absorber rather than sparse shared bursts

The GRASP line can continue only if:

- residual bursts are directly observed
- the null baseline does not already explain them
- coverage improves
- and a minimal shared-shock overlay lands under the frozen contracts

---

## 9. Final Recommendation

The current memo should be interpreted as a **research hypothesis memo**, not as a direct implementation plan.

The next correct move is:

1. `EXP-G-CHECK-01`
2. `EXP-G-CHECK-02`
3. `EXP-G-NULL-01`
4. `EXP-CAL-02`
5. only if those clear: `EXP-R10-G1-01-min`

That is the highest-scientific-value path because it gives the GRASP hypothesis a real chance to survive, but only after it has passed the falsification tests that the current evidence still demands.

---

## 10. AutoResearch Handoff

- Variant: `evidence-to-model-loop`
- Current trusted candidates:
  - exact: `EXP-R10-M1-F1`
  - dense: `EXP-R10-DENSE-M1-H1`
  - mechanistic comparator: `EXP-R1`
- Evaluation harness:
  - `exact_only`
  - `legacy_dense`
  - `purged_dense`
  - retroactive `2025` lockbox
  - endpoint x tier x era audit
  - calibration and interval coverage audit
- Mutation units:
  - residual burst detection
  - plateau census
  - piecewise/changepoint null baseline
  - dense diagnosed recalibration
  - minimal shared-shock overlay
- Stop rule:
  - stop the GRASP line if null baselines or calibration repair explain the supposed gain
  - continue only if the minimal shared-shock overlay improves frozen contracts without honesty or calibration regression
