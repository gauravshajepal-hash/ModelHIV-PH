# Phase 3 TR-V3 Next Phases Council

**Date:** 2026-04-14
**Method:** same-model council with bounded autoreason-style arbitration
**Question:** What new model families or experiment phases are actually justified now, including creative tangential paths, while staying scientifically honest?

---

## 1. Current Frontier

This memo takes the current frontier as fixed:

- exact-lane primary: `EXP-R10-M1-F1`
- dense rolling-origin primary: `EXP-R10-DENSE-M1-H1`
- dense calibration-oriented companion: `EXP-R10-DENSE-M1-B1-H1`
- annual susceptible sidecar: `EXP-S1-A1` is `defer`
- late leakage sidecar: `EXP-L2-A1` is `late_window_sensitivity_only`

Direct evidence:

- [summary.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-autoresearch-r10m1f1-s1a1-l2a1-20260414-s00/analysis/summary.md)
- [tr_v3_publishability_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/tr_v3_publishability_batch_report.md)
- [phase3_tr_v3_publishability_autoreason_2026_04_13.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_publishability_autoreason_2026_04_13.md)

The council agreed on three constraints:

1. the main live frontier is still observation-first
2. richer latent epidemic structure is still only weakly identified at quarterly resolution
3. the highest-value next work is no longer broad model churn; it is targeted structure around the current frontier plus stronger publishability-grade audits

---

## 2. Council Roles

### Evidence Agent

Read: the winning performance improvements are still incremental and local to the `R10` family, but they are real. `EXP-R10-M1-F1` moves exact quarterly mean MAE from `0.071109` to `0.069471` and improves exact lockbox `2025` from `0.053306` to `0.045118`.

### Validity Skeptic

Read: the largest remaining scientific risk is not ordinary overfitting. It is representational drift between:

- exact-only forecasting
- dense rolling-origin forecasting
- unsupported quarterly latent interpretations

The next phases must harden residual structure, contract calibration, and honesty flags before expanding biological claims.

### Representation / Modeling Agent

Read: the justified new families are those that preserve the observation-first outer model while adding bounded internal structure:

- hierarchical observation structure
- residual/measurement-noise structure
- regime segmentation

The unjustified families are:

- explicit quarterly `S(t)` as a live state
- richer live leakage blocks
- broad deep sequence models without a clear evidence contract

### Evaluation / Failure Agent

Read: the next experiments should only survive if they improve the main score *and* explain where the gain came from:

- endpoint-level
- tier-level
- calibration-level
- lockbox-level

Anything that only improves the headline MAE while weakening honesty or metric breadth should revert.

---

## 3. Autoreason Arbitration

### A: Keep pushing the `R10` frontier narrowly

Focus on:

- flow correction
- dense/exact transfer
- calibration
- residual structure

### B: Start a new partially mechanistic family now

Focus on:

- regime-aware hidden states
- annual anchors
- weak leakage sidecars

### AB: Use `R10` as the production frontier, but open a bounded new phase around observation hierarchy and regime structure while keeping mechanism sidecar-only

Judgment:

- `A` is safest but risks incrementalism
- `B` is too early and likely to produce elegant under-identification again
- `AB` is strongest because it keeps the live frontier while opening a publishable next layer that is still defensible

**Winner:** `AB`

Meaning:

- continue the main paper line with the current `R10` frontier
- start the next experimental phases around structured observation modeling, benchmark hardening, and bounded regime inference
- keep explicit `S(t)` and richer leakage as sidecars, not promoted live blocks

---

## 4. Three Core Next Phases

These are the next phases for the main model line.

### Phase C1: Hierarchical Observation Model

**What it is**

Replace the current mostly independent stock/flow heads with a shared quarterly observation hierarchy:

- common latent reporting regime factor
- metric-specific observation heads for:
  - `diagnosed_plhiv`
  - `alive_on_art`
  - `new_diagnosed_cases_period`
- explicit bounded coupling:
  - `alive_on_art <= diagnosed_plhiv`
  - diagnosis flow consistent with stock increments up to reporting error

This is not a latent epidemic model. It is a hierarchical observation model.

**Why it is justified now**

The current winner already behaves like a coupled stock/flow observation model. The next honest step is to formalize that coupling rather than pretending quarterly care hazards are identified.

**Artifact changes**

- new family: `EXP-R10-H1-*`
- new report sections:
  - shared-factor diagnostics
  - stock-flow residual decomposition
  - per-metric calibration under shared reporting regime
- new plots:
  - coupled stock/flow residual curves
  - quarter-level reporting residual band

**Keep / revert gate**

Keep only if all are true:

- exact MAE improves below `0.069471` or ties within negligible tolerance with better raw diagnosed and flow MAE
- dense MAE stays at or below `0.085418` for the primary dense variant
- suppression honesty flags do not regress
- lockbox remains competitive

**Paper novelty**

- "hierarchical stock-flow forecasting under provenance-tagged surveillance evidence"
- stronger than another plain ablation because it explains *why* the current winner works

### Phase C2: Regime Segmentation and Changepoint-Aware Forecasting

**What it is**

Add train-only regime segmentation to the observation-first frontier:

- changepoints inferred only from training data
- allow piecewise parameter blocks for:
  - stock growth
  - diagnosis flow drift
  - residual variance
- no manual COVID labeling required, though one comparison can include a fixed COVID cut as a baseline

This is the observation-first analogue of what the earlier mechanistic failures were asking for, but without pretending the hidden states are known.

**Why it is justified now**

The exact and dense winners are already piecewise in behavior. Current results suggest non-stationarity matters more than extra mechanism.

**Artifact changes**

- new family: `EXP-R10-CP-*`
- new artifacts:
  - changepoint table
  - regime-weight plots
  - pre/post-regime error table
- compare:
  - no changepoint
  - one inferred changepoint
  - two inferred changepoints
  - fixed COVID cut

**Keep / revert gate**

Keep only if:

- either exact or dense primary improves materially
- the inferred changepoints are stable across rolling splits
- no heavy dependence on one late split

Revert if changepoints bounce wildly between adjacent splits or only explain the lockbox post hoc.

**Paper novelty**

- "contract-sensitive regime discovery under mixed epidemic evidence"
- publishable if the inferred regimes improve both score and interpretability

### Phase C3: Residual Structure, Calibration, and Conformal Forecast Layer

**What it is**

Stop treating uncertainty as an afterthought. Build a proper residual layer on top of the frozen winners:

- endpoint-wise residual modeling
- exact-vs-bridge stratified residual pools
- rolling-origin conformal bands
- calibration curves by endpoint and evidence tier

This is not a new predictor family. It is a new publishable forecast layer.

**Why it is justified now**

The current winners are already good enough to deserve serious forecast calibration. That is more valuable right now than another mechanism-heavy branch.

**Artifact changes**

- new family: `EXP-UQ-*` or report-only package
- add:
  - conformal interval coverage tables
  - calibration by endpoint
  - calibration by exact vs bridge support
  - residual symmetry and drift audit

**Keep / revert gate**

Keep if:

- empirical coverage is near nominal on rolling-origin splits
- interval widths are not trivially huge
- point MAE does not materially regress if any recalibration is introduced

**Paper novelty**

- "honest uncertainty under provenance-stratified epidemic forecasting"
- this materially strengthens publishability even if point scores barely move

---

## 5. Three Parallel / Tangential Phases

These are not the main line, but they could strengthen the paper or open a second paper.

### Phase T1: Archive-Noise and Reporting-Noise Modeling

**What it is**

Build an explicit archive-noise sidecar:

- OCR / extraction instability indicators
- bridge vs exact reporting noise priors
- row-level confidence or noise class
- optional robust loss reweighting by inferred archive-noise class

**Why it is justified now**

The project’s most unusual asset is the mixed-evidence archive itself. A publishable methods angle exists if you can show that archive-noise modeling changes calibration or leaderboard stability.

**Artifact changes**

- noise annotations in the panel
- new report:
  - metric error by archive-noise class
  - robust-loss vs standard-loss comparison
  - leaderboard stability under noise-aware scoring

**Keep / revert gate**

Keep if:

- rankings become more stable or calibration improves
- exact-lane performance does not regress materially

**Paper novelty**

- "archive-noise-aware epidemic forecasting"
- this is broader than HIV and could support a methods companion paper

### Phase T2: Subnational / KP Sidecar Forecasting

**What it is**

Do not rebuild the national model as a full subnational epidemic simulator. Instead, add bounded sidecars:

- region-level or KP-level auxiliary heads where support exists
- reconcile them back to the national forecast
- treat them as auxiliary tasks, not primary metrics

Possible sidecars:

- NCR vs non-NCR
- MSM/TGW sidecar if support is enough
- key-population annual share sidecars from curated anchors

**Why it is justified now**

It creates immediate scientific interest and practical relevance without demanding a fully identified KP epidemic model.

**Artifact changes**

- auxiliary datasets
- sidecar error tables
- coherence plots between national and subgroup forecasts

**Keep / revert gate**

Keep if:

- national primary metrics do not regress materially
- subgroup sidecars improve auxiliary coherence or subgroup calibration

Revert if they destabilize the main benchmark for only marginal subgroup signal.

**Paper novelty**

- "multi-resolution forecasting from mixed surveillance evidence"
- attractive to applied public-health readers

### Phase T3: Publication-Oriented Comparator Pack

**What it is**

Build a small, honest comparator set designed for reviewers:

- carry-forward baseline
- current `R10` winners
- hierarchical observation variant
- changepoint variant
- one constrained mechanistic comparator only

Possibly also:

- one classical exponential smoothing or structural time-series comparator if not already represented

**Why it is justified now**

A paper with only internal family ablations is easier to dismiss. A publication-oriented comparator pack turns the benchmark into a stronger external-facing artifact.

**Artifact changes**

- frozen comparator table
- paired significance / win-rate table
- lockbox comparator figure

**Keep / revert gate**

Keep if:

- the paper candidate still wins against the broader comparator pack
- or loses only to a trivially close classical baseline, which is still scientifically informative

**Paper novelty**

- not novelty by itself, but it sharply improves reviewer defensibility

---

## 6. Interventions To Avoid

Do not prioritize these next:

- explicit quarterly `S(t)` promotion into the live benchmark loop
- richer live leakage blocks
- large deep sequence models without a stronger contract story
- unconstrained hybrid latent-plus-neural models that let both sides explain the same quarterly signal
- broad new mechanistic family search before calibration and regime structure are hardened

These may become reasonable later, but they are not the highest-value next phases now.

---

## 7. Recommended Order

### Main line

1. Phase `C1`: hierarchical observation model
2. Phase `C3`: residual/calibration/conformal layer
3. Phase `C2`: changepoint-aware forecasting

### Parallel

1. Phase `T3`: publication-oriented comparator pack
2. Phase `T1`: archive-noise sidecar
3. Phase `T2`: bounded subgroup sidecars

Reason:

- `C1` has the best mix of score upside and scientific honesty
- `C3` has the best publishability upside
- `C2` is creative but should not come before the residual/calibration package

---

## 8. AutoResearch Handoff

**Variant:** `evidence-to-model-loop`

**Evaluation harness**

- exact-only contract
- legacy dense contract
- purged dense contract
- retroactive lockbox `2025`
- per-metric raw MAE
- per-tier scoring
- support counts
- suppression honesty flags
- calibration coverage

**Mutation units**

- shared reporting-factor observation heads
- flow/stock reconciliation weights
- train-only changepoint inference
- residual stratification by endpoint and evidence tier
- conformal calibration layer
- archive-noise sidecar features
- subgroup sidecar heads

**First experiments**

1. `EXP-R10-H1-EXACT`
2. `EXP-R10-H1-DENSE`
3. `EXP-UQ-C1`
4. `EXP-R10-CP1`
5. `EXP-COMP-PACK`
6. `EXP-NOISE-A1`
7. `EXP-KP-SIDECAR-A1`

**Stop rule**

- keep only if the new phase improves either the primary score or the paper’s defensibility without honesty regression
- if a phase only adds biological narrative without stronger benchmark evidence, revert it

