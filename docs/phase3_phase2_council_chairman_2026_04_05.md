# Chairman Memo: Phase 3 Math and Phase 2 Integration

Date: 2026-04-05
Repo: `D:\EpiGraph_PH`
Method: same-model council plus local code inspection plus literature review

## Bottom line

Current `Phase 3` is not one benchmark regime. The repo now contains at least three materially different Phase 3 families:

1. broad `rescue_core` frozen-history backtests under `phase3_frozen_backtest` and `phase3_frozen_backtest_tournament`
2. national quarter-level `transition_research` branches under `src/epigraph_ph/phase3/transition_research`
3. incidence-side `incidence_research` branches under `src/epigraph_ph/phase3/incidence_research`

The previous version of this memo was too coarse because it treated those as one evaluation story. That was wrong.

The corrected reading is:

- the broad `rescue_core` family still does **not** reliably beat carry-forward or simple compartmental baselines
- the newer `transition_research` family **does** beat those baselines in its own national quarter-level anchored-holdout regime
- the `incidence_research` branch currently inherits that strong diagnosis-locked transition regime, so it is supportive evidence for the branch design but not an independent broad forecast win

So the best next design is still:

1. treat `Phase 2` sparse lagged edges as priors on transition channels
2. treat `Phase 2` low-rank hidden structure as explicit shared latent innovation terms
3. treat `Phase 2` blankets as screening and shrinkage gates, not as causal truth
4. treat `Phase 2` uncertainty and multiscale support as prior precision, inclusion probability, and pooling-strength inputs

But the evaluation language must now distinguish those families explicitly.

## Current Phase 3 math

### rescue_core math

This subsection describes the broad `rescue_core` Phase 3 path in [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L1). It is the multiyear frozen-history province-to-national cascade model.

The core latent state is a multigroup HIV cascade over province, month, key population, age, sex, duration, and compartment:

`x[p,k,a,s,d,t] in {U, D, A, V, L}`

The transition families are:

`r in {U->D, D->A, A->V, A->L, L->A}`

The current transition logit is additive:

`eta[p,k,a,s,d,t,r]`
`= eta_nat[r]`
`+ eta_region[region(p), year(t), r]`
`+ eta_province[p,r]`
`+ eta_kp[k,r] + eta_age[a,r] + eta_sex[s,r]`
`+ eta_duration[d,r]`
`+ eta_scaffold[t,r] + eta_slow[t,r] + eta_medium[t,r] + eta_shock[t,r]`
`+ eta_cd4[p,k,a,s,t,r]`
`+ eta_cov[p,t,r]`

Then:

`pi[p,k,a,s,d,t,r] = sigmoid(eta[p,k,a,s,d,t,r])`

Mass is propagated through compartment flow equations. In coarse form:

`U_{t+1} = U_t - U_t * p_ud`
`D_{t+1} = D_t + U_t * p_ud - D_t * p_da`
`A_{t+1} = A_t + D_t * p_da + L_t * p_la - A_t * p_av - A_t * p_al`
`V_{t+1} = V_t + A_t * p_av`
`L_{t+1} = L_t + A_t * p_al - L_t * p_la`

The model then aggregates the subgroup state to predictions for:

- diagnosed stock
- ART stock
- documented suppression
- testing coverage
- deaths

The observation layer builds support-aware targets and penalties. Mixed-frequency information exists, but is currently used mainly through annual-to-month anchor placement and penalties, not a full multiscale observation operator.

The current loss is roughly:

`L = observation_fit`
`+ observation_anchor_penalty`
`+ official_reference_penalty`
`+ national_anchor_penalty`
`+ harp_program_penalty`
`+ diagnosis_flow_penalty`
`+ linkage_penalty`
`+ suppression_penalty`
`+ hierarchy_penalty`
`+ stock_penalty`
`+ regularization`

### transition/incidence research math

The `transition_research` and `incidence_research` branches are separate national quarter-level branch systems, not just parameter tweaks of `rescue_core`.

In `transition_research`, the state is still a reduced U/D/A/V/L cascade, but the branch logic is quarter-level and usually anchored to a kept parent branch. The core transition simulator in [transition_engine.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/transition_research/transition_engine.py#L1) applies transition hazards directly to the current quarter state:

`u_to_d = h_ud * U_t`
`d_to_a = h_da * D_t`
`l_to_a = h_la * L_t`
`a_to_v = h_av * A_t`
`a_to_l = h_al * A_t`

then updates:

`U_{t+1} = U_t - u_to_d`
`D_{t+1} = D_t + u_to_d - d_to_a`
`A_{t+1} = A_t + d_to_a + l_to_a - a_to_v - a_to_l`
`V_{t+1} = V_t + a_to_v`
`L_{t+1} = L_t + a_to_l - l_to_a`

The transition-research branches differ in how they build the hazard map `h_*`:

- `MECH-*` branches estimate or lock baseline transition hazards
- `DECOMP-*` branches decompose hazards into trend, shock, and residual channels, then refit or gate those channels
- `PEAK-*` branches detect quarter-level windows and gate hazard modifications around those windows
- `AGE-*` and `KP-*` branches add age-share or subgroup overlays on top of a kept transition branch

So the transition-research math is best summarized as:

`hazard_map = baseline_hazard + supported_overlay`
`forecast = quarter_level_simulation(hazard_map, anchored_initial_state)`

In `incidence_research`, the main kept branch is [run_inc_01d](/D:/EpiGraph_PH/src/epigraph_ph/phase3/incidence_research/modeling.py#L239). It inherits the quarter-level diagnosis path from the kept `AGE-01B` branch, then makes the pre-U inflow explicit by splitting undiagnosed mass balance into:

`latent_incidence_inflow >= 0`
`residual_undiagnosed_clearance >= 0`

subject to the undiagnosed-state accounting identity:

`U_t = U_{t-1} + incidence_inflow_t - diagnosis_flow_t - residual_clearance_t`

The key point is that `INC-01D` is diagnosis-locked:

- the diagnosis-side holdout forecast is inherited from the kept transition branch
- the incidence branch is judged by whether that explicit inflow accounting stays compatible with annual incidence targets

So the incidence-research math is not a better broad forecaster yet. It is a constrained incidence-accounting branch layered on top of a winning transition branch.

## What Phase 2 currently contributes

Again, this section is primarily about the broad `rescue_core` path. The `transition_research` and `incidence_research` branches are currently using branch-specific helper and overlay logic rather than the same generic Phase 2 injection contract.

The new `Phase 2` object is a sparse-plus-low-rank temporal graph over `Phase 15` latent innovations.

It estimates something like:

`epsilon_t = sum_l S_l z_{t-l} + sum_l L_l z_{t-l} + noise`

where:

- `S_l` is sparse direct lagged block-to-block structure
- `L_l` is low-rank hidden shared-driver structure

In principle this gives five different signal types:

1. sparse lagged direct edges
2. hidden low-rank shared-driver structure
3. target blankets
4. multiscale support across province / region / national
5. stability and uncertainty summaries

But current `Phase 3` mostly uses:

`Phase 2 selection -> covariate inclusion -> transition-logit modifier`

That is much weaker than what the new Phase 2 can support.

## Validity warning

The original version of this memo was wrong in two ways:

1. it cited the wrong artifact family
2. it collapsed several incompatible Phase 3 evaluation regimes into one statement

The repo currently has at least three different benchmark families that should not be mixed.

### A. Broad rescue-core frozen-history family

The most relevant broad backtest family is still `audit-phase0-reuse-s00-20260331`.

In its top-level frozen-history backtest:

- [phase3_frozen_backtest/frozen_history_backtest_evaluation.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest/frozen_history_backtest_evaluation.json#L1)
- [phase3_frozen_backtest/frozen_history_backtest_spec.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest/frozen_history_backtest_spec.json#L1)

the summary is:

- `model_mean_absolute_error = 0.168083`
- `carry_forward_mean_absolute_error = 0.113485`
- `simple_compartmental_mean_absolute_error = 0.078644`
- holdout years `2021` through `2025`
- forecast horizon `60`

So the broad `rescue_core` family still does **not** support a claim that Phase 3 reliably beats carry-forward or simple compartmental baselines.

The broad rescue-core representation tournament also does not rescue the model. In [representation_tournament.json](/D:/EpiGraph_PH/artifacts/runs/audit-phase0-reuse-s00-20260331/phase3_frozen_backtest_tournament/representation_tournament.json#L1), the winner is still worse than carry-forward:

- `winner_representation = clumped_baseline`
- `winner_model_mean_absolute_error = 0.089551`
- `winner_beats_carry_forward = false`

### B. Transition-research family

There is also a distinct Phase 3 research track under [transition_research](/D:/EpiGraph_PH/src/epigraph_ph/phase3/transition_research/registry.py#L1). Those experiments use a different regime:

- national quarter-level holdouts, typically `2025-Q1` through `2025-Q4`
- normalized MAE/SMAPE evaluation against national quarterly targets
- anchored-holdout branch designs, where some experiments explicitly copy the first holdout quarter from a parent branch before forecasting downstream quarters

In this family, the latest April 2026 runs do beat the simple baselines. For example:

- [MECH-01A baseline_comparison.json](/D:/EpiGraph_PH/artifacts/runs/tr-20260402-s00-MECH-01A-national-udavl-baseline/transition_research/MECH-01A-national-udavl-baseline/baseline_comparison.json#L1)
  - `model_mean_absolute_error = 0.042292`
  - `carry_forward_mean_absolute_error = 0.098051`
  - `simple_compartmental_mean_absolute_error = 0.043493`
- [PEAK-01F baseline_comparison.json](/D:/EpiGraph_PH/artifacts/runs/tr-20260402-s00-PEAK-01F-region-plus-kp-modifier-gated-fused-forecast/transition_research/PEAK-01F-region-plus-kp-modifier-gated-fused-forecast/baseline_comparison.json#L1)
  - `model_mean_absolute_error = 0.034749`
  - `carry_forward_mean_absolute_error = 0.098051`
  - `simple_compartmental_mean_absolute_error = 0.043493`
- [DECOMP-01F baseline_comparison.json](/D:/EpiGraph_PH/artifacts/runs/tr-20260402-s00-DECOMP-01F-reverse-grasp-peak-clusters/transition_research/DECOMP-01F-reverse-grasp-peak-clusters/baseline_comparison.json#L1)
  - `model_mean_absolute_error = 0.034754`
  - `carry_forward_mean_absolute_error = 0.098051`
  - `simple_compartmental_mean_absolute_error = 0.043493`

Across the `tr-20260402-s00-*` baseline-comparison artifacts, `14/17` beat carry-forward and `13/17` beat both carry-forward and simple compartmental by direct MAE comparison.

So the user's objection was correct: the repo does contain a current Phase 3 branch family that is materially outperforming those baselines.

But that does **not** make the broad rescue-core statement false by itself, because these are not the same benchmark regime.

### C. Incidence-research family

There is also a separate [incidence_research](/D:/EpiGraph_PH/src/epigraph_ph/phase3/incidence_research/registry.py#L1) branch. The strongest current result is:

- [INC-01D baseline_comparison.json](/D:/EpiGraph_PH/artifacts/runs/inc-20260402-s00-INC-01D-diagnosis-locked-incidence-branch/incidence_research/INC-01D-diagnosis-locked-incidence-branch/baseline_comparison.json#L1)
  - `model_mean_absolute_error = 0.034685`
  - `carry_forward_mean_absolute_error = 0.098051`
  - `simple_compartmental_mean_absolute_error = 0.043493`

However this branch is diagnosis-locked and explicitly reports:

- `comparison_reference_run_id = tr-20260402-s00-AGE-01B-youth-diagnosis-modifier-stable` in [INC-01D evaluation.json](/D:/EpiGraph_PH/artifacts/runs/inc-20260402-s00-INC-01D-diagnosis-locked-incidence-branch/incidence_research/INC-01D-diagnosis-locked-incidence-branch/evaluation.json#L1)
- `model_matches_locked_baseline = true` in the same baseline comparison artifact

So `INC-01D` is supportive evidence for the diagnosis-locked incidence formulation, but it is not an independent broad Phase 3 forecast win. It inherits the kept transition branch.

### Corrected conclusion

The scientifically correct reading is now:

- broad `rescue_core` Phase 3 backtests still do **not** have robust evidence of beating simple baselines across the latest audit-family frozen-history evaluations
- the separate `transition_research` branch family **does** have current evidence of beating carry-forward and simple compartmental baselines in its own national quarter-level anchored-holdout regime
- the `incidence_research` branch currently inherits that winning transition regime rather than independently establishing a broader forecasting win

So the previous memo was wrong by over-aggregating these regimes. The right fix is not to claim universal Phase 3 failure or universal Phase 3 success. It is to separate the benchmark families explicitly and reason about them on their own terms.

Also, some legacy Phase 2 interfaces remain dangerous:

- legacy blanket injection can be endpoint-adjacent
- some backtests appear able to reuse full-run Phase 2 artifacts
- determinant gains can be confounded with always-kept observation covariates

So `Phase 2` should still enter `Phase 3` as an uncertainty-aware structural prior, not as mechanistic truth. But the transition/incidence branches now deserve to be treated as real positive evidence rather than ignored.

## What the literature supports

### Sparse direct graph and hidden latent dynamics should stay separate

- You and Yu, *Sparse plus low-rank identification for dynamical latent-variable graphical AR models* (Automatica, 2024): sparse direct interactions and low-rank hidden dynamics should be represented separately, not collapsed into one effect class.

### Mixed-frequency state-space modeling should be explicit

- Li, Zhou, Pitt, *Dynamic Mortality Forecasting via Mixed-Frequency State-Space Models* (2026): annual and monthly data should be joined by explicit aggregation inside the latent model, not by crude carry-forward or year-end replacement.

### Irregular observations should be assimilated jointly with latent dynamics

- Si and Chen, *LEVDA* (2026)
- Tong, Wang, Yan, *Latent Autoencoder Ensemble Kalman Filter for Data assimilation* (2026)

These support treating graph outputs as priors in a joint latent assimilation system rather than fixed regressors.

### Graph structure can live inside the state-space model

- Alippi and Zambon, *Graph Kalman Filters* (2023)

This supports using graph structure to define coupling in the state equation, not just feature selection.

### Hierarchical inference should be joint, not fully staged

- Mancarella and Gerosa, *Sampling the full hierarchical population posterior distribution in gravitational-wave astronomy* (2025)

This supports pushing more of the province / region / national hierarchy into a joint inference step.

### Mechanistic epidemic models should treat extra latent factors as competing hypotheses, not truths

- Friston et al., *Dynamic causal modelling of COVID-19 and its mitigations* (2022)
- Prashad, *State-space modelling for infectious disease surveillance data* (2025)

These support adding graph-derived drivers as uncertain mechanistic hypotheses with model comparison and uncertainty, not as fixed discovered mechanisms.

## Best next mathematical design

### 1. Direct Phase 2 edges become priors on transition channels

Let `b` index Phase 15 latent blocks and `l` index lag.

Define:

`g_r[p,t] = sum_{b,l} Gamma[r,b,l] * z_b[p,t-l]`

Then:

`eta[...,t,r] = eta_existing[...,t,r] + g_r[p,t] + h_r[p,t]`

where:

- `Gamma[r,b,l]` is learned
- Phase 2 edge sign, stability, and support define the prior on `Gamma`

This means Phase 2 does not force the transition. It tells Phase 3 which block-lag-transition links are plausible and how strongly to shrink them.

### 2. Hidden low-rank Phase 2 structure becomes latent shock processes

Introduce shared latent shocks:

`u_m[t] = rho_m * u_m[t-1] + xi_m[t]`

and inject them through:

`h_r[p,t] = sum_m Lambda_hidden[r,m] * W_hidden[p,m] * u_m[t]`

Interpretation:

- sparse direct Phase 2 edges model directed block effects
- low-rank Phase 2 structure models common hidden drift, regime pressure, or shared shocks

These should not be merged.

### 3. Blankets become eligibility gates, not truth

For each transition family `r`, define an eligibility mask:

`I[r,b,l] in {0,1}`

from Phase 2 target blankets after filtering out endpoint-adjacent features.

Then:

`Gamma[r,b,l] ~ spike_slab(I[r,b,l], support[r,b,l], stability[r,b,l])`

In practice this can be continuous shrinkage, but the logic is:

- blanket says a block is allowed to compete
- data decides whether it survives

### 4. Phase 2 uncertainty controls prior precision

Use support counts and bootstrap stability to scale prior variance:

`Gamma[r,b,l] ~ Normal(mu_phase2[r,b,l], sigma_phase2[r,b,l]^2)`

with:

`sigma_phase2[r,b,l]^2` large when support is weak or unstable
`sigma_phase2[r,b,l]^2` small when support is strong and stable

This is the scientifically safe use of Phase 2.

### 5. Multiscale support chooses where effects live

If Phase 2 support is mostly national:

`Gamma[r,b,l] = Gamma_nat[r,b,l]`

If it is regional:

`Gamma_region[region,t,r,b,l]`

If it is province-rich:

`Gamma_province[p,r,b,l]`

with hierarchical shrinkage.

This avoids pretending province-specific effects are identified when only national support exists.

### 6. Mixed-frequency observation operator should be promoted

Instead of only snapping annual anchors to a month, define:

`y_i = H_i(x_{1:T}) + eps_i`

where `H_i` is:

- annual national aggregation
- annual regional aggregation
- monthly province observation

This is the proper mixed-frequency state-space move supported by the literature.

## Recommended priority order

1. Stop unsafe legacy Phase 2 injection in frozen-history evaluation.
2. Load `latent_temporal_graph_bundle.json` and `latent_temporal_phase3_target_blankets.json` directly.
3. Use sparse direct edges as priors on transition channels.
4. Use hidden low-rank structure as shared latent shock terms.
5. Use blankets and support as screening/shrinkage controls.
6. Upgrade mixed-frequency anchoring into a true observation operator.
7. Only then consider deeper joint inference.

## Chairman judgment

The mathematically best use of Phase 2 in Phase 3 is not “more covariates.” It is:

- structural transition priors from sparse lagged edges
- latent shared shock channels from low-rank hidden structure
- uncertainty-aware gating from blankets and support
- multiscale placement from scale support

That matches both the current repo architecture and the strongest cross-domain literature.
