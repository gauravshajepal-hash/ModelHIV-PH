# Phase 2 Seeded Champion Decision Gate

Date: 2026-04-18
Role: Evaluation / Failure Agent
Question: What must be true before we promote more Phase 2 seeded scenarios or escalate to a regime/sojourn layer?

## Inputs

- Seeded scenario batch: [/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/tr_v3_phase2_seeded_champion_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/tr_v3_phase2_seeded_champion_batch_report.md)
- Monthly edge audit: [/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-edge-audit-20260418-s01/analysis/tr_v3_monthly_edge_audit_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-edge-audit-20260418-s01/analysis/tr_v3_monthly_edge_audit_batch_report.md)
- Monthly lane rebuild: [/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260418-s02/analysis/tr_v3_monthly_phase2_lane_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260418-s02/analysis/tr_v3_monthly_phase2_lane_batch_report.md)

## Current Read

The monthly lane is now structurally usable:

- contiguous `2010-01` to `2025-12`
- `192` monthly points, `0` axis gaps
- `697` monthly-aligned Phase 1 rows
- `622` injected HARP rows
- `22` canonicals
- `4` retained Phase 15 blocks
- `2` direct lag-1 edges
- `0` hidden temporal edges
- `0` hidden driver rows

The corrected candidate edge shift survives the support fix:

- `testing_engagement -> care_access_continuity @ lag1` strengthened
- `testing_engagement -> suppression_capacity @ lag1` appeared
- no hard support-collapse flag fired in the corrected audit

The seeded scenario batch is scientifically acceptable as a bounded stress-test layer:

- it uses the corrected monthly lane
- it keeps the exact and dense champions frozen as the mean models
- it fits a small residual readout
- it produces interpretable deltas without exploding the mean path

But the evidence does **not** justify a regime/sojourn layer yet:

- only `2` direct edges remain
- hidden structure is still absent
- exact residual counts are small: `12-13`
- the current seeded pass shows scenario sensitivity, not discovered dwell-time structure

## 1. First Audits / Checks

These checks must pass before any promotion.

### Gate A: Structural support integrity

Must hold:

- monthly axis remains contiguous
- diagnosis-flow support remains preserved
- HARP program-point support remains preserved
- no rerun-path confounders

Fail if:

- any structural candidate changes edge structure while also changing support geometry
- diagnosis-flow points or HARP program points collapse again

### Gate B: Edge stability under blocked rebuilds

Run the corrected monthly rebuild in at least `3` blocked folds:

- `2010-2017 -> 2018-2020 -> 2021-2022`
- `2010-2019 -> 2020-2022 -> 2023-2024`
- `2010-2021 -> 2022-2024 -> 2025`

Keep only if:

- `testing -> care @ lag1` appears in at least `2/3` folds
- sign is consistent in all retained folds
- score rank stays top-2 among direct edges

Fail if:

- the edge appears only in one fold
- the sign flips
- the apparent testing-centered structure disappears when late annual anchors are held out

### Gate C: Readout robustness

The seeded readout must be stable, not overfit.

Checks:

- refit the seeded readout with bootstrap or leave-one-split-out perturbations
- record sign consistency of the dominant coefficients
- measure scenario terminal deltas under coefficient perturbation

Keep only if:

- coefficient signs for dominant block-to-metric paths stay stable in at least `80%` of resamples
- terminal scenario deltas vary by less than `25%` for the main scenarios

Fail if:

- scenario ordering changes from minor resampling
- the readout depends on one split or one endpoint only

### Gate D: Boundedness and physical sanity

Every scenario layer must remain subordinate to the champions.

Checks:

- scenario deltas remain below a fixed fraction of the base forecast path
- stock constraints remain respected:
  - `alive_on_art <= diagnosed_plhiv`
  - no impossible negative flows or stocks
- scenario shocks decay or plateau according to declared semantics

Keep only if:

- no scenario breaks stock-flow sanity
- no quarter exceeds a predefined perturbation cap, for example `<= 20%` of the base endpoint level unless the scenario is explicitly labeled extreme

Fail if:

- the seeded layer starts acting like a replacement mean model
- a scenario causes implausible sign reversals or non-decaying oscillation

### Gate E: Semantic identifiability

The scenario names must match what the structure actually supports.

Keep only if:

- a "testing" scenario mainly moves outcomes through `testing_engagement` and its retained lag-1 descendants
- a "mobility" scenario stays labeled as stress-test only, not causal forecast

Fail if:

- semantic labels outrun the measured structure
- we start telling mechanistic stories not supported by the retained graph

## 2. Bounded Experiments With Keep / Revert Criteria

### EXP-P2-GATE-01
Blocked monthly edge-stability audit.

Purpose:

- test whether the testing-centered structure survives blocked rebuilds

Keep if:

- Gate B passes

Revert if:

- the testing-centered shift is fold-specific

### EXP-P2-GATE-02
Seeded readout bootstrap stability audit.

Purpose:

- test whether the seeded mapping from Phase 2 blocks to champion residuals is stable

Keep if:

- Gate C passes

Revert if:

- the readout is coefficient-unstable or endpoint-fragile

### EXP-P2-GATE-03
Scenario boundedness and monotonicity audit.

Purpose:

- enforce that the seeded layer remains a stress-test layer

Checks:

- perturbation ratio by quarter and endpoint
- stock sanity
- pulse decay versus plateau persistence behavior

Keep if:

- Gate D passes

Revert if:

- any scenario breaks the declared semantics or boundedness cap

### EXP-P2-SCEN-01
Same-kernel scenario expansion.

Add only a few new families:

- `testing_collapse`
- `care_bottleneck`
- `testing_then_care_delay`
- `two_wave_disruption`
- `slow_drift_up`
- `slow_drift_down`

Keep if:

- at least `3` new scenarios produce distinct terminal and path-level behavior
- they remain bounded
- they remain interpretable under the retained monthly graph

Revert if:

- new scenarios collapse onto existing ones
- they only differ by arbitrary amplitude, not structure

### EXP-P2-REG-00
Regime/sojourn feasibility probe only.

This is not full promotion. It is a feasibility check.

Do only after Gates A-D pass.

Minimal scope:

- `K=2` or `K=3` sticky regime model
- no large HSMM sweep
- no promoted scenario library yet

Keep if:

- it yields stable regime occupancy across blocked folds
- mean sojourns are reproducible
- scenario semantics become cleaner than the same-kernel library

Revert if:

- inferred regimes are unstable
- different folds learn different dwell-time stories
- regime labels mostly restate amplitude rather than new structure

## 3. Failure Modes / False Wins

### False win 1: support-driven graph change

The graph looks richer only because support changed, not because structure improved.

Why this matters:

- this already happened once when diagnosis-flow support collapsed to zero

Mitigation:

- never interpret edge shifts without paired support context

### False win 2: scenario richness without structural discrimination

More scenario families can look impressive while being the same kernel with renamed amplitudes.

Mitigation:

- require path distinctness and terminal distinctness tests

### False win 3: residual readout overfit

The seeded layer appears interpretable, but one unstable readout coefficient is driving the entire picture.

Mitigation:

- bootstrap sign stability and delta stability audit

### False win 4: premature hidden-state promotion

A regime/sojourn model can always create neat labels even when the underlying data only support a simple lagged graph.

Mitigation:

- demand fold-stable occupancy and sojourn estimates before treating regimes as real

### False win 5: mechanistic overclaim

Calling these outputs "future shock prediction" would overstate the evidence.

Mitigation:

- keep the seeded layer labeled as scenario emulation or structural stress testing unless exogenous shock predictors are introduced and validated separately

## 4. Recommendation

Recommendation: **same-kernel scenario expansion now, regime/sojourn later if the gates pass.**

Reason:

- the monthly rebuild is finally clean
- the corrected audit shows the testing-centered shift survives preserved support
- the seeded batch already behaves correctly as a bounded scenario layer
- there is still no hidden-driver evidence
- there are still only `2` retained direct edges
- the seeded readout is promising but not yet audited for stability

So the next phase should be:

1. run `EXP-P2-GATE-01` to `EXP-P2-GATE-03`
2. if they pass, run `EXP-P2-SCEN-01` on the same bounded kernel
3. only after that run `EXP-P2-REG-00`

Do **not** start with a full regime/sojourn layer now. That would be a complexity jump ahead of the evidence.

## AutoResearch Handoff

- Variant: `evidence-to-model-loop`
- Trusted evaluation harness:
  - corrected monthly rebuild
  - corrected monthly edge audit
  - seeded scenario boundedness audit
- Mutation units:
  - blocked structural rebuild
  - readout bootstrap stability
  - bounded scenario family additions
  - minimal regime feasibility probe
- Stop rule:
  - keep only if the structural story stays support-stable and the seeded layer remains bounded and interpretable
