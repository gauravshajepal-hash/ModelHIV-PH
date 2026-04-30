# Phase 3 Transition Frontier Implementation Plan

Date: 2026-04-05

Repo: `D:\EpiGraph_PH`

Directory scope:
- `D:\EpiGraph_PH\src\epigraph_ph\phase3`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\incidence_research`

Companion note:
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\phase3_architecture_and_frontier_2026_04_05.md`

## Executive Position

This plan converts the architecture note into an implementation program.

The central judgment is:

- `transition_research` becomes the primary Phase 3 research frontier
- `rescue_core` remains in the repo, but as broad benchmark plus infrastructure reservoir
- `incidence_research` remains downstream of a kept transition winner
- new Phase 2-aware mathematics should be introduced first inside the winning transition branch family, not inside broad `rescue_core`

This is not a documentation preference. It is a model-selection decision justified by current results:

- broad multiyear frozen-history `rescue_core` still loses to simple baselines in its kept benchmark regime
- quarter-level national `transition_research` branches are the current winners in their kept evaluation regime
- `incidence_research` currently inherits a kept transition winner rather than establishing an independent forecasting core

## Problem Statement

Phase 3 currently contains multiple mathematical programs with different strengths:

1. `rescue_core.py` is broad and infrastructure-rich, but not currently the winning predictive branch.
2. `transition_research/` is better identified and currently winning in its valid regime.
3. `incidence_research/` adds meaningful scientific structure, but currently depends on a locked transition branch.

The repo problem is not that these all exist.

The problem is that the current folder still makes it too easy to treat `rescue_core` as the default research future, when the evidence says the frontier should move to the transition family.

## Strategic Goal

Make the winning `transition_research` branch family the active Phase 3 development frontier without losing:

- broad benchmark honesty
- reusable Phase 3 infrastructure
- the ability to later reintegrate successful transition logic back into a broader Phase 3 system

## Non-Goals

This plan does **not** recommend:

- deleting `rescue_core.py`
- rewriting the entire Phase 3 folder at once
- claiming that `incidence_research` is already an independent broad winner
- pushing Phase 2 signals directly into the broad rescue-core additive modifier stack as more ordinary covariates

## Core Principle

The correct inheritance direction is:

`winning transition branch -> future broad Phase 3 rebuild`

not:

`broad rescue_core -> transition branch`

That means:

- extract infrastructure from `rescue_core`
- do not copy the whole model into `transition_research`
- do not expand the broad model first
- build the next mathematical layer on top of the best transition branch

## Current Folder Roles

This section translates the architecture note into concrete ownership.

### Shared platform modules

These should remain top-level shared modules:

- `pipeline.py`
- `evaluation_regimes.py`
- `broad_backtest_support.py`
- `mixed_frequency.py`
- `temporal_scaffold.py`

Why:
- they express Phase 3 platform concerns rather than branch-specific modeling

### Broad benchmark modules

These should remain top-level, but no longer define the default research frontier:

- `rescue_core.py`
- `national_reset_core.py`
- `national_reset_pipeline.py`

Why:
- `rescue_core.py` remains the broad benchmark and infrastructure reservoir
- `national_reset_*` remains part of the lineage and reference stack that transition research uses

### Branch-specific frontier modules

These should become the main Phase 3 research locus:

- `transition_research/`

Why:
- they host the current winning branch family and the best-identified hazard logic

### Dependent extension modules

These remain as downstream dependent branches:

- `incidence_research/`

Why:
- they inherit from a kept transition winner and should remain explicit about that dependency

### Tooling modules

These remain support tooling:

- `peak_search.py`

Why:
- useful for search and evaluation, but not itself the main mechanistic model

## Concrete Objectives

### Objective 1

Promote `transition_research` to the explicit Phase 3 frontier.

Success condition:
- the repo contains a named transition-frontier plan and branch family
- new Phase 3 v2 experiments are implemented under `transition_research`
- benchmark reporting explicitly distinguishes frontier status from broad benchmark status

### Objective 2

Extract reusable observation/evaluation infrastructure from `rescue_core`.

Success condition:
- transition experiments can reuse mixed-frequency, anchor handling, and evaluation helpers without ad hoc copy-paste

### Objective 3

Introduce Phase 2 as structured hazard priors and hidden shared shocks in the transition family.

Success condition:
- Phase 2 does not enter the new branch as generic modifier columns
- sparse direct graph structure and low-rank hidden structure are modeled separately

### Objective 4

Keep broad rescue-core as a valid multiyear benchmark, not as a deprecated branch.

Success condition:
- broad frozen-history rescue-core backtests still run cleanly
- new transition experiments can be compared against broad benchmark outputs later

### Objective 5

Preserve dependency transparency for `incidence_research`.

Success condition:
- incidence artifacts continue to record which transition branch they inherit from
- no incidence result is presented as an independent forecasting frontier unless it truly is one

## Workstreams

## Workstream A

### Name

`phase3-platform-extraction`

### Purpose

Extract shared machinery from `rescue_core` and other top-level files into reusable platform helpers so that `transition_research` can evolve without copy-paste debt.

### Target files

- `D:\EpiGraph_PH\src\epigraph_ph\phase3\mixed_frequency.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\rescue_core.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\pipeline.py`
- new helper modules to add under `D:\EpiGraph_PH\src\epigraph_ph\phase3\`

### Concrete mutations

1. Extract broad-but-generic observation support code from `rescue_core.py` into a shared helper module.
2. Extract anchor-array and reference-point preparation into a shared helper module if it is currently entangled with broad rescue-specific internals.
3. Move any genuinely generic quarter-level evaluation utilities out of branch-specific files into shared Phase 3 support modules.
4. Keep branch-specific hazard logic inside `transition_research`, not in the shared layer.

### Acceptance criteria

1. No duplicate copies of mixed-frequency or anchor preparation logic appear in `transition_research`.
2. `transition_research` can call shared observation helpers directly.
3. `rescue_core` still runs using the extracted helpers.

### Keep-or-revert rule

Keep only if:
- behavior remains unchanged for current rescue-core evaluation artifacts
- code volume in `transition_research` does not grow by copy-paste reuse

## Workstream B

### Name

`transition-frontier-baseline-lock`

### Purpose

Create an explicit immutable winning baseline for the next research generation.

### Target base branch

`AGE-01B-youth-diagnosis-modifier`

### Why `AGE-01B`

It is currently the best transition-side base because:

- it inherits the successful `PEAK-01F` detector-gated branch
- it improves over its branch parent
- it adds only a narrow, support-aware diagnosis modifier
- it has cleaner identifiability than a full broad transition refit

### Concrete mutations

1. Add a named `TR-V2-00` experiment entry in `transition_research/registry.py`.
2. Implement `TR-V2-00` as a strict locked reproduction of `AGE-01B`.
3. Emit a baseline-lock artifact that records:
   - parent branch id
   - exact inherited hazards
   - exact inherited holdout quarters
   - exact performance metrics

### Acceptance criteria

1. `TR-V2-00` reproduces `AGE-01B` metrics within machine tolerance.
2. The artifact explicitly documents the inherited branch lineage.
3. This becomes the regression baseline for later `TR-V2-*` experiments.

## Workstream C

### Name

`phase2-to-hazard-direct-effects`

### Purpose

Introduce sparse direct lagged Phase 2 structure into the transition hazard model in a mathematically clean way.

### Mathematical target

For transition `r`:

`logit h_r(t) = alpha_r(t) + base_r(t) + sum_{b,l} Gamma[r,b,l] z_b(t-l)`

where:
- `z_b` is the Phase 15 latent block state aggregated to the national quarter scale
- `Gamma` is a direct lagged effect from Phase 2 sparse temporal structure

### Important restriction

This must enter as:

- a structural prior or structured hazard term

and not as:

- another generic modifier covariate bundle

### Target files

- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\sources.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\transition_engine.py`
- new module to add, recommended:
  - `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\phase2_hazard_priors.py`

### Concrete mutations

1. Build quarter-level national latent block surfaces from Phase 15/2 outputs.
2. Read sparse direct temporal edges from the kept Phase 2 latent graph bundle.
3. Map eligible block-lag pairs to transition channels.
4. Introduce `Gamma[r,b,l]` as hazard terms or priors.
5. Weight prior strength by Phase 2 support/stability.

### Acceptance criteria

1. The implementation keeps direct Phase 2 structure separate from hidden low-rank structure.
2. The experiment emits a table of:
   - retained direct block-lag terms
   - corresponding hazard channels
   - prior mean and prior scale
3. The model can be ablated against `TR-V2-00`.

### Proposed experiment id

`TR-V2-01-phase2-direct-hazard-priors`

## Workstream D

### Name

`phase2-hidden-shock-layer`

### Purpose

Introduce the Phase 2 low-rank hidden structure as separate latent shared shock channels, not as direct effects.

### Mathematical target

`u_m(t) = rho_m u_m(t-1) + xi_m(t)`

`logit h_r(t) = ... + sum_m Lambda[r,m] u_m(t)`

### Why this is necessary

Phase 2 now decomposes temporal structure into:

- sparse direct edges
- low-rank hidden shared structure

Those cannot be collapsed back into one hazard modifier layer without losing the mathematical point of the decomposition.

### Target files

- new module to add, recommended:
  - `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\hidden_shocks.py`
- integration points in:
  - `transition_engine.py`
  - `decomposition.py`

### Concrete mutations

1. Read the low-rank hidden-driver output from Phase 2.
2. Represent hidden temporal modes at quarter scale.
3. Add hazard shock terms with separate coefficients `Lambda`.
4. Keep these terms separate in artifacts and diagnostics from direct `Gamma` terms.

### Acceptance criteria

1. Direct and hidden Phase 2 structure are logged separately.
2. The model can run with:
   - direct only
   - hidden only
   - both
3. Hidden shock magnitude can be ablated independently from direct edge magnitude.

### Proposed experiment id

`TR-V2-02-phase2-hidden-shock-hazards`

## Workstream E

### Name

`transition-frontier-ablation-suite`

### Purpose

Prevent accidental over-claiming by making the new transition frontier pass controlled ablations.

### Required comparisons

1. `TR-V2-00` locked baseline
2. direct Phase 2 priors only
3. hidden shocks only
4. direct plus hidden
5. each of the above with and without peak gating if technically compatible

### Metrics

At minimum:
- mean absolute error
- sMAPE
- diagnosis-flow mean absolute error
- peak alive-on-ART absolute error

### Required artifact

Recommended new artifact:
- `transition_frontier_ablation_summary.json`

### Acceptance criteria

1. Every new `TR-V2-*` experiment records parent lineage.
2. Every new `TR-V2-*` experiment is compared directly against `TR-V2-00`.
3. No new branch is considered "kept" unless it beats `TR-V2-00` on the agreed metric rule.

## Workstream F

### Name

`incidence-branch-dependency-explicitness`

### Purpose

Keep `incidence_research` scientifically honest as a dependent extension unless and until it becomes independently predictive.

### Target files

- `D:\EpiGraph_PH\src\epigraph_ph\phase3\incidence_research\modeling.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\incidence_research\registry.py`

### Concrete mutations

1. Standardize lineage recording for inherited transition branches.
2. Emit explicit dependency metadata in every incidence run.
3. Separate:
   - forecast inheritance quality
   - incidence compatibility quality

### Acceptance criteria

1. Every incidence artifact records the parent transition branch.
2. Incidence branch success is not conflated with independent forecast success.

## Workstream G

### Name

`broad-reintegration-readiness`

### Purpose

Prepare for a later broad rescue-core rebuild around winning transition logic, without doing that prematurely.

### Principle

This workstream is readiness only, not immediate reintegration.

### Concrete mutations

1. Define a narrow interface between transition hazards and broad rescue-core transition hooks.
2. Make sure shared observation logic is reusable from the top level.
3. Record which transition-frontier components are broadening candidates and which are too regime-specific.

### Acceptance criteria

1. The repo contains a documented interface boundary for future rescue-core reintegration.
2. No premature direct copy of winning transition logic into rescue-core occurs in this plan phase.

## Implementation Order

This is the recommended order of execution.

### Slice 1

`phase3-platform-extraction`

Reason:
- needed before the new frontier evolves without copy-paste debt

### Slice 2

`transition-frontier-baseline-lock`

Reason:
- needed before any new experiment can be judged properly

### Slice 3

`phase2-to-hazard-direct-effects`

Reason:
- direct sparse graph structure is the simplest valid Phase 2 integration layer

### Slice 4

`phase2-hidden-shock-layer`

Reason:
- hidden shared structure should only be added after the direct path is working

### Slice 5

`transition-frontier-ablation-suite`

Reason:
- needed to keep the frontier honest

### Slice 6

`incidence-branch-dependency-explicitness`

Reason:
- low implementation cost and high scientific clarity

### Slice 7

`broad-reintegration-readiness`

Reason:
- should come only after the new frontier is stable

## Concrete File-Level Plan

## Add

- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\phase2_hazard_priors.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\hidden_shocks.py`
- possibly `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\baseline_lock.py`
- possibly `D:\EpiGraph_PH\src\epigraph_ph\phase3\phase3_shared_observation_support.py`

## Edit

- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\registry.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\transition_engine.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\decomposition.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\sources.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\transition_research\artifacts.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\incidence_research\modeling.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\pipeline.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\rescue_core.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\mixed_frequency.py`

## Possibly avoid editing unless needed

- `D:\EpiGraph_PH\src\epigraph_ph\phase3\national_reset_core.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\national_reset_pipeline.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase3\peak_search.py`

Reason:
- these are lineage/tooling modules and should not be destabilized unless the frontier work actually requires it

## Experiment Registry Plan

Recommended new experiment ids:

1. `TR-V2-00-age01b-baseline-lock`
2. `TR-V2-01-phase2-direct-hazard-priors`
3. `TR-V2-02-phase2-hidden-shock-hazards`
4. `TR-V2-03-phase2-ablation-suite`
5. `TR-V2-04-region-pooled-transition-frontier`

Recommended incidence companion ids:

1. `INC-V2-00-lineage-explicitness-audit`
2. `INC-V2-01-phase2-compatible-incidence-extension`

## Required Artifacts

Every new transition-frontier experiment should emit:

- `baseline_comparison.json`
- `evaluation.json`
- `mechanistic_forecast.json`
- `state_trajectory_rows.json`
- `transition_hazard_summary.json`
- `frontier_lineage.json`

Phase 2-integrated experiments should additionally emit:

- `phase2_direct_hazard_prior_summary.json`
- `phase2_hidden_shock_summary.json`
- `transition_frontier_ablation_summary.json`

## Evaluation Rules

## Hard gates

1. `TR-V2-00` must exactly reproduce `AGE-01B`.
2. Any new `TR-V2-*` branch must be directly compared against `TR-V2-00`.
3. Any branch claiming Phase 2 benefit must log:
   - direct terms used
   - hidden terms used
   - excluded terms
   - prior scales
4. `incidence_research` branches must keep explicit parent transition lineage.

## Keep-or-revert rule

Keep a new transition-frontier mutation only if:

1. it preserves artifact completeness,
2. it improves or at least does not materially regress the winning branch metrics,
3. it does not silently blur the distinction between direct and hidden Phase 2 structure,
4. it does not rely on unsupported broadening assumptions,
5. it keeps lineage and evaluation regime explicit.

## Scientific posture rule

No new branch should be described as:

- broad winner
- final mechanistic truth
- independent incidence winner

unless it actually satisfies those broader evaluation criteria.

## Mathematical Upgrade Path

The intended mathematical progression is:

### Current winning structure

Quarter-level national hazard model with:

- anchored holdout state
- detector-gated fused hazards
- support-aware age-conditioned diagnosis modifier

### Next structure

Quarter-level national hazard model with:

- anchored holdout state
- detector-gated fused hazards
- Phase 2 sparse direct hazard priors
- Phase 2 low-rank hidden hazard shocks
- support-aware prior scales

### Later broadening

Only after that:

- region-pooled transition frontier
- then possible reintegration into broader rescue-core structures

## Why This Plan Is Correct For The Current Folder

Because the current folder is not a mess. It is a layered history of model families.

The right move is not simplification by deletion.
The right move is functional reclassification:

- shared platform at the top level
- broad benchmark in `rescue_core`
- winning frontier in `transition_research`
- dependent extension in `incidence_research`

That is the cleanest interpretation of the current `phase3` folder and the cleanest path for the next implementation cycle.

## Immediate Next Slice Recommendation

The first implementation slice after this plan should be:

1. platform extraction of shared observation/evaluation helpers
2. `TR-V2-00` baseline lock
3. `TR-V2-01` direct Phase 2 hazard priors

This is the highest-value path because it:

- respects the current winning branch
- avoids premature broad rescue-core mutation
- gives the repo a real Phase 3 frontier with explicit regression control
