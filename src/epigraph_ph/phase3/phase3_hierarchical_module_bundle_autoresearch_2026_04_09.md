Date: 2026-04-09

# Phase 3 Hierarchical Module-Bundle Autoresearch Design

## Purpose

This note defines the next unattended Phase 3 loop after `AN-03A`.

It does not start another blind national search.

It uses the retained provincial autoresearch evidence to answer a concrete question:

```text
How should Phase 2 direct and hidden structure be inserted into a joint
mechanistic HIV cascade model so that national fit is preserved while the
provincial structure lost by aggregation is recovered?
```

The selected autoresearch variant is:

```text
evidence-to-model-loop
```

with a benchmark-hardening gate.

That is the correct choice because:

- the national mechanistic loop already beats both national baselines;
- the overlapping provincial evidence still loses badly to naive;
- therefore the next loop must be constrained by evidence rather than given a larger blind search surface.

## Evidence Base

This design is grounded in two kept artifacts.

### National mechanistic incumbent

Run:

```text
phase3v2int-20260409-s08
```

Key comparison artifact:

```text
artifacts/runs/phase3v2int-20260409-s08/transition_research/
PHASE3-V2-INT-explicit-incidence-autoresearch/baseline_comparison.json
```

National benchmark facts:

- model primary loss `0.081574`
- carry-forward primary loss `0.091381`
- simple compartmental primary loss `0.145822`
- model diagnosis-flow loss `0.094933`
- carry-forward diagnosis-flow loss `0.161338`
- simple compartmental diagnosis-flow loss `0.268798`

So the current national loop is worth protecting.

### Provincial evidence source

Diagnostic run:

```text
an03a-20260409-s02
```

Evidence summary artifact:

```text
artifacts/runs/an03a-20260409-s02/transition_research/
AN-03A-phase2-aggregation-loss-diagnostic/provincial_evidence_summary.json
```

Key facts:

- selected provincial evidence run:
  `bounded-v2-replay-allprovinces-harp-avfix-20260329`
- retained factor overlap count: `6`
- provincial model MAE `0.25976`
- provincial naive MAE `0.025236`
- diagnosed optimism gate failed
- hierarchy reconciliation gate failed

So the provincial loop is not yet acceptable, but it contains useful factor evidence and geography-dependent failure information.

### Aggregation-loss seed artifact

Seed artifact:

```text
artifacts/runs/an03a-20260409-s02/transition_research/
AN-03A-phase2-aggregation-loss-diagnostic/module_bundle_seed_report.json
```

This file is the search-space source for the next loop.

The loop must not invent a fresh module-factor search space from scratch.

## Problem Statement

The current national mechanistic loop uses Phase 2, but only after geographic aggregation.

That loses structure in two ways:

1. province-specific direct factors are collapsed before the hazard model sees them;
2. hidden temporal modes are estimated from the already-aggregated national matrix, not from the full province-by-time variation.

The next loop must therefore optimize:

- a province-resolved mechanistic state model,
- module-specific direct Phase 2 bundle selection,
- shared hidden latent shocks,
- hierarchical national/region/province coefficient structure,
- and a keep/reject gate that protects the current national gains while forcing provincial improvement.

## State Space

For province `p = 1, ..., P`, region `r(p)`, and quarter `t = 1, ..., T`:

```text
x_{p,t} = (U_{p,t}, D_{p,t}, A_{p,t}, V_{p,t}, L_{p,t})
```

with:

- `U`: undiagnosed PLHIV
- `D`: diagnosed, not on ART
- `A`: on ART, not documented in suppressed state
- `V`: virally suppressed on ART
- `L`: interrupted / lost from effective treatment

Province totals aggregate exactly:

```text
X_t = sum_p x_{p,t}
```

Region totals aggregate exactly:

```text
X_{g,t} = sum_{p: r(p)=g} x_{p,t}
```

National evaluation is performed on `X_t`.

## Observed Inputs

### Denominator

For each province and quarter:

```text
N_{p,t}
```

is the observed denominator / exposure pool used for incidence.

### Direct Phase 2 surfaces

For each province, quarter, and retained direct factor `j`:

```text
z_{p,j}(t)
```

comes from the province-level Phase 2 factor tensor.

These are not merged into a single national factor before the mechanistic fit.

### Hidden Phase 2 structure

Let:

```text
u_m(t)
```

be the quarterized shared hidden-mode score for mode `m`, derived from the Phase 2 hidden-mode tensor.

These are shared temporal shocks.

They are allowed to have module-specific and geography-specific loadings, but the latent time process itself remains shared.

### Observations

At province, region, or national level where available:

- `y_diag_{p,t}`
- `y_art_{p,t}`
- `y_newdiag_{p,t}`
- `y_vltest_{p,t}`
- `y_vs_{p,t}`
- `y_total_{p,t}`

Missing observations are treated as unobserved, not as zero and not as infinite-loss years.

## Candidate Model Family

The candidate model family is:

```text
HMBA = Hierarchical Module-Bundle Autoresearch
```

with family tuple:

```text
F = (
  diagnosis_family,
  care_family,
  hidden_rank,
  B_I,
  B_UD,
  B_DA,
  B_AV,
  B_AL,
  B_LA,
  geography_family,
  observation_family
)
```

where:

- `diagnosis_family in {hazard, delay}`
- `care_family in {markov, semi_markov}`
- `hidden_rank` is selected from the non-empty retained hidden modes
- `B_*` are selected direct-bundle sets for each module
- `geography_family in {national_only, national_region, national_region_province}`
- `observation_family` controls the ascertainment layer

## Mechanistic Equations

### Incidence module

For province `p`:

```text
I_{p,t} = N_{p,t} * lambda_{p,t}
```

with:

```text
log(lambda_{p,t}) =
    a_I
  + a_I^{reg}[r(p)]
  + a_I^{prov}[p]
  + f_I^{dir}(p, t; B_I)
  + f_I^{hid}(p, t)
```

### Diagnosis module

Two admissible families are allowed.

#### F3-Hazard diagnosis

```text
d_{p,t} = h_{UD,p,t} * U_{p,t}
```

with:

```text
cloglog(h_{UD,p,t}) =
    a_{UD}
  + a_{UD}^{reg}[r(p)]
  + a_{UD}^{prov}[p]
  + f_{UD}^{dir}(p, t; B_{UD})
  + f_{UD}^{hid}(p, t)
```

#### F3-Delay diagnosis

```text
E[y_{newdiag,p,t}] = sum_{k=0}^{K_t} I_{p,t-k} * pi_{p,k}(t)
```

with:

```text
pi_{p,k}(t) = g_{p,k}(t) * product_{j=0}^{k-1} (1 - g_{p,j}(t-k+j))
```

and:

```text
logit(g_{p,k}(t)) =
    a_{D,k}
  + a_{D,k}^{reg}[r(p)]
  + a_{D,k}^{prov}[p]
  + f_{UD}^{dir}(p, t-k; B_{UD})
  + f_{UD}^{hid}(p, t-k)
```

The optimizer is allowed to choose either diagnosis family.

### Downstream care modules

For transitions:

- `D -> A`
- `A -> V`
- `A -> L`
- `L -> A`

the Markov family uses:

```text
cloglog(h_{r,p,t}) =
    a_r
  + a_r^{reg}[r(p)]
  + a_r^{prov}[p]
  + f_r^{dir}(p, t; B_r)
  + f_r^{hid}(p, t)
```

The semi-Markov family adds dwell-time dependence.

For dwell time `s` in the source state:

```text
cloglog(h_{r,p,t}(s)) =
    a_r
  + a_r^{reg}[r(p)]
  + a_r^{prov}[p]
  + b_r * q_r(s)
  + f_r^{dir}(p, t; B_r)
  + f_r^{hid}(p, t)
```

where:

- `q_r(s)` is an empirical basis in dwell time,
- its basis dimension is selected by model evidence,
- not by a hand-set polynomial degree.

### State updates

For hazard diagnosis:

```text
U_{p,t+1} = U_{p,t} + I_{p,t} - d_{p,t}
D_{p,t+1} = D_{p,t} + d_{p,t} - a_{p,t}
A_{p,t+1} = A_{p,t} + a_{p,t} + r_{p,t} - v_{p,t} - l_{p,t}
V_{p,t+1} = V_{p,t} + v_{p,t}
L_{p,t+1} = L_{p,t} + l_{p,t} - r_{p,t}
```

with:

```text
a_{p,t} = h_{DA,p,t} * D_{p,t}
v_{p,t} = h_{AV,p,t} * A_{p,t}
l_{p,t} = h_{AL,p,t} * A_{p,t}
r_{p,t} = h_{LA,p,t} * L_{p,t}
```

For delay diagnosis, `d_{p,t}` is the latent diagnosed inflow implied by the delay convolution and state consistency constraints.

## Direct And Hidden Structure

The direct and hidden Phase 2 structures must remain separate.

### Direct terms

For module `m` and selected bundle set `B_m`:

```text
f_m^{dir}(p, t; B_m) =
  sum_{b in B_m} sum_{j in J_b} sum_{ell in L_{m,b}}
    beta_{m,b,j,ell,p} * z_{p,j}(t-ell)
```

with hierarchical decomposition:

```text
beta_{m,b,j,ell,p} =
    beta_{m,b,j,ell}^{nat}
  + beta_{m,b,j,ell}^{reg}[r(p)]
  + beta_{m,b,j,ell}^{prov}[p]
```

Bundle selection happens at the `b` level.

Coefficient estimation happens at the factor-and-lag level inside selected bundles.

### Hidden terms

For module `m`:

```text
f_m^{hid}(p, t) =
  sum_{q=1}^{R}
    (
        lambda_{m,q}^{nat}
      + lambda_{m,q}^{reg}[r(p)]
      + lambda_{m,q}^{prov}[p]
    ) * u_q(t)
```

with `R = hidden_rank`.

The `u_q(t)` processes are shared latent shocks from Phase 2.

The loadings may vary hierarchically, but the hidden shock trajectories are not redefined per province.

## Hierarchical Shrinkage

The model must not fit provinces independently.

Coefficient hierarchy is:

```text
coefficient = national effect + region deviation + province deviation
```

For any regional deviation `delta^{reg}` and province deviation `delta^{prov}`:

```text
delta^{reg} ~ N(0, tau_reg^2)
delta^{prov} ~ N(0, tau_prov^2)
```

The shrinkage scales are estimated by empirical Bayes from the data:

- maximize marginal likelihood under the current family,
- or equivalently choose the regularization strength that optimizes the frozen evaluation contract.

They are not hand-written constants.

## Module-Specific Phase 2 Search Space

The direct-bundle search space is seeded from `AN-03A`.

The loop must use the module bundle rankings already extracted from the provincial evidence overlap.

### Incidence seed bundles

- `epidemiology_cascade`
- `service_delivery_infrastructure`
- `logistics_access`
- `stigma_behavior_information`
- `mobility_network_mixing`

### `U -> D` seed bundles

- `epidemiology_cascade`
- `service_delivery_infrastructure`
- `logistics_access`
- `population_structure_demography`
- `stigma_behavior_information`
- `mobility_network_mixing`

### `D -> A` seed bundles

- `mobility_network_mixing`
- `population_structure_demography`
- `stigma_behavior_information`

### `A -> V` seed bundles

- `stigma_behavior_information`

### `A -> L` seed bundles

- `service_delivery_infrastructure`
- `logistics_access`
- `mobility_network_mixing`

### `L -> A` seed bundles

- `mobility_network_mixing`
- `service_delivery_infrastructure`
- `logistics_access`
- `population_structure_demography`
- `stigma_behavior_information`

The next loop may expand these seed sets only if a new aggregation-loss diagnostic on a frozen snapshot promotes additional bundles.

## Search Algorithm

The search must be structured.

It must not do a raw Cartesian product over all factors, lags, modules, provinces, and families.

### Stage 0. Freeze the contract

Freeze:

- source run ID
- historical observation snapshot
- provincial evidence run ID
- baseline comparison contracts
- module bundle seed artifact

The contract artifact must be emitted before model search starts.

### Stage 1. Module-local bundle promotion

For each module independently:

1. start with the current incumbent family and no new direct bundles for that module;
2. add one candidate bundle from the module seed list;
3. re-fit the affected module coefficients under the full hierarchical model;
4. keep the addition only if the evaluation tuple improves and no gate fails;
5. repeat until no remaining bundle improves the tuple.

This is staged forward selection with exact keep-or-revert.

No hand-picked top-`K` cutoff is used.

### Stage 2. Diagnosis family selection

Compare:

- hazard diagnosis
- delay-convolution diagnosis

under the promoted incidence and `U -> D` bundle sets.

Keep the diagnosis family that is lexicographically best under the frozen gate.

### Stage 3. Care family selection

Compare:

- Markov downstream care
- semi-Markov downstream care

under the promoted `D -> A`, `A -> V`, `A -> L`, and `L -> A` bundle sets.

### Stage 4. Hidden-rank selection

Search over all non-empty hidden ranks that are available from the retained Phase 2 hidden-mode tensor.

Keep the rank that improves the evaluation tuple after re-fitting.

This is a finite exact search, not a hand-written rank guess.

### Stage 5. Cross-module combination

After module-local promotion, define the promoted bundle sets:

```text
B_I^*
B_{UD}^*
B_{DA}^*
B_{AV}^*
B_{AL}^*
B_{LA}^*
```

Search over cross-module combinations using staged composition:

1. combine incidence with diagnosis;
2. combine front half with downstream care;
3. activate geography deviations;
4. activate observation family variants.

At each stage:

- discard dominated candidates,
- keep only candidates that pass the full gate,
- do not prune by a hand-written beam width.

### Stage 6. Geography activation

Compare:

- national-only coefficients
- national + region deviations
- national + region + province deviations

using the same promoted module bundles.

This step decides whether a candidate truly benefits from partial pooling beyond national fit.

### Stage 7. Target-seeking loop

Only after the mechanistic champion is kept:

1. freeze the mechanistic champion;
2. expose bounded multipliers on the selected direct bundle terms;
3. optimize the latent cascade objective toward `95-95-95`;
4. reject any target-seeking candidate that worsens the mechanistic historical fit gate.

The target loop is not allowed to rewrite the structural model family.

## Evaluation Contract

The keep/reject rule must evaluate both national and provincial behavior.

### Hard validity gates

Reject immediately if any of the following occurs:

- non-finite states or predictions
- negative state masses beyond numerical tolerance
- hierarchy aggregation mismatch
- impossible cascade ratios
- optimizer failure

### National benchmark gates

The candidate must beat both national baselines on the frozen window:

```text
candidate_primary_loss < carry_forward_primary_loss
candidate_primary_loss < simple_compartmental_primary_loss
candidate_diag_flow_loss < carry_forward_diag_flow_loss
candidate_diag_flow_loss < simple_compartmental_diag_flow_loss
```

### Provincial benchmark gates

The candidate must satisfy both:

```text
candidate_provincial_mae <= incumbent_provincial_mae
candidate_hierarchy_reconciliation <= incumbent_hierarchy_reconciliation
```

Until a candidate actually beats provincial naive, ranking must also minimize:

```text
provincial_gap_ratio = candidate_provincial_mae / provincial_naive_mae
```

That keeps the search honest about the provincial failure.

### Lexicographic ranking tuple

Among candidates that pass all hard gates, rank by:

```text
(
  candidate_primary_loss,
  candidate_diag_flow_loss,
  provincial_gap_ratio,
  candidate_provincial_mae,
  candidate_hierarchy_reconciliation,
  candidate_diagnosed_optimism,
  candidate_province_instability
)
```

This ordering is deliberate.

National fit is protected first because it is already stronger than baseline.

Provincial improvement is next because it is the unresolved failure mode.

## Observation Layer

The observation module remains explicit.

At minimum it must separately model:

- diagnosed counts
- ART counts
- VL testing documentation
- documented suppression

The loop may compare observation families, but the latent cascade and the documented program outputs must remain distinct objects.

## Required Artifacts

Each run of this loop must emit:

- `contract_snapshot.json`
- `module_seed_manifest.json`
- `module_promotion_frontier.json`
- `family_selection_frontier.json`
- `hidden_rank_frontier.json`
- `geography_activation_report.json`
- `national_baseline_comparison.json`
- `provincial_baseline_comparison.json`
- `hierarchy_reconciliation_report.json`
- `decision.json`

Required charts:

- `module_promotion_frontier.png`
- `national_vs_baselines.png`
- `provincial_vs_baselines.png`
- `hierarchy_reconciliation.png`
- `cascade_target_projection.png`

These charts are mandatory because the loop is supposed to be unattended and visually auditable.

## Why This Design Is The Correct Next Step

This design does three things the current loop does not.

1. It preserves the national mechanistic win instead of discarding it.
2. It uses provincial autoresearch artifacts as evidence rather than as a disconnected failed branch.
3. It makes Phase 2 structurally central to the mechanistic model without allowing an uncontrolled combinatorial explosion.

That is exactly what the current evidence demands.

## Next Implementation Slices

The correct build order is:

1. `HMBA-00`
   - freeze the contract artifact
   - materialize province/region/national evaluation tensors
2. `HMBA-01`
   - implement module-local direct-bundle forward selection
   - keep direct and hidden terms separate
3. `HMBA-02`
   - implement hierarchical coefficient decomposition with empirical-Bayes shrinkage
4. `HMBA-03`
   - implement diagnosis family selection and care family selection
5. `HMBA-04`
   - add geography activation and provincial baseline gates
6. `HMBA-05`
   - add post-fit `95-95-95` target search with non-regression protection

Only after those slices are in place should another unattended province-aware autoresearch epoch be launched.
