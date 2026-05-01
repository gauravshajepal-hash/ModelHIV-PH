# Autoresearch Loop

Status:
- operational execution manifest only
- canonical scientific design lives in [phase3_tr_v3_autoresearch_design_2026_04_10.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_autoresearch_design_2026_04_10.md)

Document contract:
- use [phase3_tr_v3_autoresearch_design_2026_04_10.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_autoresearch_design_2026_04_10.md) for model structure, assumptions, mathematics, identifiability, and keep/revert logic
- use this file only for live execution order, experiment IDs, contract names, and implementation status
- if these two files disagree, the design memo wins and this file must be updated to match it

Selected variant: `evidence-to-model-loop`

Rules:
- frozen artifacts only
- blocked-time evaluation only
- keep or revert by benchmark result, not narrative appeal
- no experiment is kept just because it improves one late split
- no experiment is allowed to treat `rule_based_extrapolated` or `latent_imputed` holdout rows as truth

## Data Contract

Quarterly evaluation contracts:
- `exact_only`
  - train and score on exact quarterly rows only
  - use bridge and annual history only for diagnostics
- `dense_train_observed_score`
  - train on `exact_observed + bridge_observed + rule_based_extrapolated`
  - score holdout only on `exact_observed + bridge_observed`

Missing-data ladder:
- `exact_observed`
- `bridge_observed`
- `rule_based_extrapolated`
- `latent_imputed`
- `rejected_or_quarantined`

## Experiment Registry

### EXP-B0

Objective:
- benchmark-contract ablation

Contract:
- compare `exact_only` against `dense_train_observed_score`
- do not score any holdout metric from `rule_based_extrapolated` or `latent_imputed`

Required outputs:
- contract support counts by year and metric
- benchmark deltas between contracts
- graph of contract coverage and benchmark change

Current implementation status:
- partially implemented
- current suite can run both contracts and emit graphs, but does not yet report an explicit side-by-side B0 comparison artifact

### EXP-B1

Objective:
- missingness and provenance map for quarterly and annual targets

Required outputs:
- year-by-metric availability matrix
- missing-data ladder counts
- graph of exact vs bridge vs rule-based support

Current implementation status:
- implemented as diagnostic in the current suite

### EXP-B2

Objective:
- imputation-contract ablation

Contract:
- train can use imputed rows
- holdout scoring can only use observed rows

Required outputs:
- dense-panel summary
- score-eligible quarter list
- benchmark delta versus `exact_only`

Current implementation status:
- partially implemented
- dense panel exists and the second contract runs, but the dedicated B2 comparison artifact is not yet split out

### EXP-V1

Objective:
- harden the dense contract by purging split-local lookahead leakage

Contract:
- rebuild the dense quarterly panel separately inside each rolling split using only information available up to that split's train end
- rerun only:
  - `EXP-R10-EXACT-CHAMPION`
  - `EXP-R10-DENSE-CHAMPION`
  - `EXP-R1`

Required outputs:
- legacy dense leaderboard
- purged dense leaderboard
- winner stability check

Current implementation status:
- implemented
- writes a dedicated purged-dense audit artifact
- the suite dense contract now uses split-local purged reconstruction by default

### EXP-V2

Objective:
- endpoint and provenance-tier audit for the promoted winners and mechanistic anchor

Required outputs:
- per-metric raw MAE for:
  - `diagnosed_plhiv`
  - `alive_on_art`
  - `new_diagnosed_cases_period`
- per-tier scoring for `exact_observed` vs `bridge_observed`
- support counts per metric per split
- hard suppression honesty flag

Current implementation status:
- implemented
- suite rows now carry endpoint/tier audit payloads and the run emits a dedicated `EXP-V2` artifact

### EXP-05a-01

Objective:
- hazard-side ascertainment-only control

Model:
- apply `A` only to `U_to_D`

Current implementation status:
- implemented

### EXP-05a-02

Objective:
- hazard-side care-only control

Model:
- apply `C` to `D_to_A`, `A_to_V`, `A_to_L`, `L_to_A`

Current implementation status:
- implemented

### EXP-05a-04

Objective:
- joint `A + C` hazard controls

Current implementation status:
- implemented

### EXP-05a-05

Objective:
- joint `A + C + R` hazard controls

Current implementation status:
- implemented

### EXP-05a-06

Objective:
- lag ablation for hazard-side controls

Model:
- compare `0`, `1`, and `2` quarter lags under the `A + C + R` control family

Current implementation status:
- partially implemented
- lag variants exist in code, but they should be reported under one experiment family rather than three separate top-level IDs

### EXP-R1

Objective:
- strict-support hazard repair for sparse early bridge years

Model:
- fit each transition only on rows the archive actually identifies
- do not send unsupported zero hazards through `logit(eps)`
- forecast by bounded drift around carry-forward, not free logit-trend extrapolation

Required outputs:
- train observed hazard curves
- train fitted hazard curves
- explicit supported-fit points
- forecast hazard curves

Current implementation status:
- implemented
- emits a separate hazard-curve artifact so unsupported transition fits are visually obvious

### EXP-R2

Objective:
- care-transition repair under the design memo contract

Model:
- keep strict-support fitting for diagnosis and leakage transitions
- forecast ART coverage share directly from observed diagnosed and ART stocks
- forecast suppression share directly from observed ART and suppressed stocks
- derive `D_to_A` and `A_to_V` from those share targets instead of pretending sparse quarterly care hazards are directly identified

Current implementation status:
- implemented
- should be compared against `EXP-R1` on both the primary exact benchmark and the broader dense diagnostic contract

### EXP-R3

Objective:
- stock-reconciliation care repair centered on `alive_on_art`

Model:
- keep strict-support fitting for `U_to_D`, `A_to_L`, and `L_to_A`
- forecast `alive_on_art` directly as a supported stock series
- derive `D_to_A` from the ART stock target instead of free-fitting sparse quarterly care hazards
- keep `A_to_V` on carried suppression share unless the archive directly supports it

Current implementation status:
- implemented
- should be compared against `EXP-R1` as the current stock-centered care repair candidate

### EXP-R4

Objective:
- hybrid care repair that only falls back to stock reconciliation when care support is sparse

Model:
- fit all transitions with strict-support drift first
- keep strict-support `D_to_A` and `A_to_V` when the split has enough care support
- fall back to `alive_on_art` stock reconciliation for `D_to_A` only when care support is sparse
- fall back to suppression carry for `A_to_V` only when suppression support is sparse

Current implementation status:
- implemented
- should be compared against `EXP-R1` and `EXP-R3` on both exact and dense contracts

### EXP-R5

Objective:
- `D_to_A`-only hybrid with stricter suppression gating

Model:
- keep the `D_to_A` hybrid logic from `EXP-R4`
- allow `D_to_A` to switch between strict-support and `alive_on_art` stock reconciliation by split
- treat `A_to_V` as carry/frozen unless direct suppression support is genuinely strong for that split

Current implementation status:
- implemented
- should be compared against `EXP-R1` and `EXP-R4` to test whether later-period damage was caused by premature suppression-hazard fitting

### EXP-R6

Objective:
- pure `D_to_A` repair with no `A_to_V` mode switching

Model:
- keep the split-aware `D_to_A` hybrid logic
- switch `D_to_A` between strict-support and `alive_on_art` reconciliation by split
- keep `A_to_V` frozen on suppression carry for every split

Current implementation status:
- implemented
- should be compared against `EXP-R1`, `EXP-R4`, and `EXP-R5`

### EXP-R7

Objective:
- regime-aware `D_to_A` repair with partial blending

Model:
- keep `A_to_V` frozen on suppression carry
- use strict-support `D_to_A` when care support is strong
- use stock reconciliation when care support is absent
- blend the two when care support is partial and recent, instead of hard-switching

Current implementation status:
- implemented
- should be compared against `EXP-R1`, `EXP-R4`, and `EXP-R6`

### EXP-R8

Objective:
- explicit two-regime `D_to_A` repair

Model:
- identify a recent contiguous supported `D_to_A` block near the end of training
- use reconciliation in the earlier sparse regime
- use strict-support `D_to_A` only inside the recent late regime
- keep `A_to_V` frozen on suppression carry

Current implementation status:
- implemented
- should be compared against `EXP-R1`, `EXP-R6`, and `EXP-R7`

### EXP-R9

Objective:
- identify `D_to_A` from net ART-stock change rather than absolute ART stock level

Model:
- fit `U_to_D`, `A_to_L`, and `L_to_A` with strict-support drift
- forecast quarterly `alive_on_art` net change from supported stock deltas
- derive `D_to_A` from the ART delta target
- keep `A_to_V` frozen on suppression carry
- optionally blend repaired `D_to_A` with strict-support `D_to_A` when direct care support is partially present

Current implementation status:
- implemented
- included in the bounded unattended repair search as a genuinely new family beyond `R1/R6/R7`

### EXP-R10

Objective:
- build an observation-first upper-bound repair around the actually scored quarterly metrics

Model:
- forecast `diagnosed_plhiv`, `alive_on_art`, and `new_diagnosed_cases_period` directly from supported bounded-drift series
- blend those forecasts against carry-forward with explicit weights
- keep `virally_suppressed` on suppression carry from the last supported ART share
- treat derived transition hazards as diagnostics, not as the primary fitted object

Current implementation status:
- implemented
- included in the bounded unattended repair search as an explicit “best possible scored-target” comparison family

### EXP-R10-CHAMPION

Objective:
- preserve the original promoted R10 point as a legacy reference row

Model:
- fixed promotion of `SEARCH-R10-s100-f50`
- direct observation repair with:
  - diagnosed stock weight `1.0`
  - ART stock weight `1.0`
  - diagnosis-flow weight `0.5`
  - level-series forecasts for diagnosed, ART, and flow
  - suppression on carried ART share

Current implementation status:
- implemented
- legacy promoted candidate only
- superseded by the explicit exact/dense benchmark split below

### EXP-R10-M1

Objective:
- add explicit joint consistency to the winning `R10` family

Model:
- forecast diagnosed stock, ART stock, and diagnosis flow jointly
- enforce `alive_on_art <= diagnosed_plhiv`
- bound diagnosed stock-flow inconsistency by the empirical train-time reporting residual range
- keep suppression as a sidecar only

Current implementation status:
- implemented
- currently evaluated under the exact contract and targeted purged-dense runs

### EXP-R10-M2

Objective:
- add train-only residual bias correction to the winning `R10` family

Model:
- estimate mean supported train residual per metric
- apply additive correction only from train-time information
- do not use holdout residuals or post-hoc forecast leakage

Current implementation status:
- implemented
- currently evaluated under the exact contract and targeted purged-dense runs

### EXP-R10-EXACT-CHAMPION

Objective:
- freeze the exact-only predictive benchmark candidate

Model:
- fixed promotion of `SEARCH-R10E-a100-f25-delta-sup75`
- direct observation repair with:
  - diagnosed stock weight `1.0`
  - ART stock weight `1.0`
  - diagnosis-flow weight `0.25`
  - delta-mode diagnosis-flow forecast
  - suppression carry weight `0.75`

Current implementation status:
- implemented
- default predictive benchmark candidate for `exact_only`

### EXP-R10-DENSE-CHAMPION

Objective:
- freeze the dense-contract predictive benchmark candidate

Model:
- fixed promotion of `SEARCH-R10D-artdelta-ab100-f100`
- direct observation repair with:
  - ART level forecast built from delta-mode ART series
  - full diagnosis-flow weight `1.0`
  - full suppression carry weight `1.0`

Current implementation status:
- implemented
- default predictive benchmark candidate for `dense_train_observed_score`

### EXP-R11

Objective:
- add the smallest honest mechanistic overlay on top of the `R10` observation heads

Model:
- keep the direct observation heads for diagnosed stock, ART stock, and diagnosis flow
- add a constrained diagnosis/infection overlay with annual incidence anchoring
- do not add richer leakage
- do not add a quarterly mortality block
- do not make suppression claims

Current implementation status:
- implemented
- currently evaluated under the exact contract and targeted purged-dense runs

## Explicit Track Split

Predictive benchmark track:
- `exact_only` default candidate = `EXP-R10-EXACT-CHAMPION`
- `dense_train_observed_score` default candidate = `EXP-R10-DENSE-CHAMPION`

Mechanistic research track:
- anchor = `EXP-R1`
- keep using the design memo’s mechanistic interpretation and identifiability rules
- do not judge mechanistic progress only by whether it beats the predictive R10 frontier immediately

## Bounded Repair Search

Objective:
- run a bounded unattended search around the `R1`, `R6`, `R7`, `R9`, and `R10` repair families without changing the design-memo benchmark contract

Execution rule:
- evaluate a fixed local candidate set only
- use the same scorer and keep/revert semantics as the experiment suite
- score on both:
  - `exact_only`
  - `dense_train_observed_score`
- keep only candidates that enter the Pareto frontier across the chosen contracts

Current implementation status:
- implemented in [tr_v3_repair_search.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_repair_search.py)
- writes a champion report plus per-contract candidate artifacts under the standard `artifacts/runs/<run-id>/analysis` tree

### EXP-N1

Objective:
- annual inflow with equal quarterly share

Current implementation status:
- implemented

### EXP-N2

Objective:
- annual inflow with infectious-pool quarterly shares

Current implementation status:
- implemented

### EXP-N3

Objective:
- annual inflow with infectious-pool plus `A/C/R` share scores

Current implementation status:
- implemented

### EXP-L1

Objective:
- minimal richer leakage block

Scope:
- add `D_to_L`, `A_to_L`, `L_to_A`

Current implementation status:
- deferred
- not yet honest on the live archive

### EXP-M1

Objective:
- shared mortality baseline

Scope:
- annual deaths anchored, but no quarterly/state-specific death identification

Current implementation status:
- deferred

### EXP-05b-full-open-leaky-model

Objective:
- full open leaky incidence-flow model with leakage and mortality blocks

Current implementation status:
- deferred
- depends on `EXP-L1` and `EXP-M1`

## Current Execution Order

1. `EXP-B0`
2. `EXP-B1`
3. `EXP-B2`
4. `EXP-05a-01`
5. `EXP-05a-02`
6. `EXP-05a-04`
7. `EXP-05a-05`
8. `EXP-05a-06`
9. `EXP-R1`
10. `EXP-R2`
11. `EXP-N1`
12. `EXP-N2`
13. `EXP-N3`
14. `EXP-L1`
15. `EXP-M1`
16. `EXP-05b-full-open-leaky-model`

## Immediate Implementation Boundary

Executable now:
- `EXP-B1`
- `EXP-05a-01`
- `EXP-05a-02`
- `EXP-05a-04`
- `EXP-05a-05`
- `EXP-N1`
- `EXP-N2`
- `EXP-N3`

Executable with current code but needing cleaner reporting:
- `EXP-B0`
- `EXP-B2`
- `EXP-05a-06`
- `EXP-R1`
- `EXP-R2`

Still deferred:
- `EXP-L1`
- `EXP-M1`
- `EXP-05b-full-open-leaky-model`
