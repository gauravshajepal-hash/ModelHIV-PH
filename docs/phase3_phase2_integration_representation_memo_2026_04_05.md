# Phase 3 Representation Memo: Using Phase 2 Outputs Coherently

Date: 2026-04-05  
Role: Same-model council, Representation / Modeling Agent  
Repo: `D:\EpiGraph_PH`

## Scope

This memo inspects the current `Phase 3` mathematical structure and proposes the best mathematically coherent ways to use `Phase 2` outputs:

- sparse lagged edges
- hidden-driver low-rank terms
- target blankets
- multiscale support
- uncertainty

The focus is representation quality: latent objects, dynamic state-space integration, priors, transition modifiers, observation selection, regularization, and hierarchy.

## Files And Artifacts Used

Code:

- [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L1793)
- [mixed_frequency.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/mixed_frequency.py#L32)
- [pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/pipeline.py#L380)
- [latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py#L87)

Run artifacts:

- [latent_temporal_graph_bundle.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase2/latent_temporal_graph_bundle.json#L1)
- [latent_temporal_phase3_target_blankets.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase2/latent_temporal_phase3_target_blankets.json#L1)
- [phase15_v2_fit_summary.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_fit_summary.json#L1)

## 1. Current Phase 3 Mathematical Structure

### 1.1 Latent object

The core latent object is not a Phase 2 latent graph state. It is a mechanistic HIV cascade state over province, month, subgroup, and duration:

`x[p,k,a,s,d,t] in {U,D,A,V,L}`

where:

- `p` = province
- `k` = key population group
- `a` = age band
- `s` = sex
- `d` = duration bucket
- `t` = month

This is explicit in the transition step and tensor construction in [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L1990) and the torch fit path in [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L4040).

### 1.2 Transition model

Phase 3 uses five transition hazards:

`r in {U->D, D->A, A->V, A->L, L->A}`

The current transition logits are additive:

`eta[p,k,a,s,d,t,r]`
`= eta_nat[r]`
`+ eta_region[region(p), year(t), r]`
`+ eta_province[p,r]`
`+ eta_kp[k,r] + eta_age[a,r] + eta_sex[s,r]`
`+ eta_duration[d,r]`
`+ eta_scaffold[t,r] + eta_slow[t,r] + eta_medium[t,r] + eta_shock[t,r]`
`+ eta_cd4[p,k,a,s,t,r]`
`+ eta_cov[p,t,r]`

Then

`pi[p,k,a,s,d,t,r] = sigmoid(eta[p,k,a,s,d,t,r])`

This is implemented directly in [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L4396).

Interpretation:

- the model is already a multilevel state-space hazard model
- temporal structure enters through scaffold, slow, medium, and shock terms
- Phase 2 determinants currently enter only through `eta_cov`

### 1.3 Mechanistic state evolution

Given `pi`, the state is propagated by mass-conserving compartment flows:

`U_{t+1} = U_t - U_t * p_ud`
`D_{t+1} = D_t + U_t * p_ud - D_t * p_da`
`A_{t+1} = A_t + D_t * p_da + L_t * p_la - A_t * p_av - A_t * p_al`
`V_{t+1} = V_t + A_t * p_av`
`L_{t+1} = L_t + A_t * p_al - L_t * p_la`

with duration and subgroup structure around this base flow. See [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L1990) and [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L2031).

### 1.4 Observation layer

Phase 3 builds target-specific observation support surfaces for:

- `diagnosed_stock`
- `art_stock`
- `documented_suppression`
- `testing_coverage`
- `deaths`

For each target it computes:

- `observed_mask[p,t]`
- `support_strength[p,t]`
- `latent_weight[p,t] = 1 - support_strength[p,t]`

from normalized rows or sparse tensor support. See [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L1793).

The initialization prior is then:

`mu_init[p,t,state] = blend(observed-derived state mean, default state, support_strength[p,t])`

See [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L1902).

### 1.5 Mixed-frequency anchoring

Mixed-frequency support exists, but only as anchor arrays and penalties, not as a full observation operator over latent state cells.

The current bundle resolves annual points to an effective month, usually year-end:

`y_anchor(year) -> y_anchor(month = latest month in year)`

See [mixed_frequency.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/mixed_frequency.py#L32) and [mixed_frequency.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/mixed_frequency.py#L89).

Important current limitation:

- `regional_anchor_arrays` are allocated but currently zero-filled in the bundle
- the mixed-frequency system is national-anchor-centric, not fully multiscale

### 1.6 Current use of Phase 2

There are two distinct regimes.

Old regime:

- [pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/pipeline.py#L380) builds a legacy intervention tensor from `markov_blanket.json` and `core_feature_tensor.npz`
- this is feature pooling over same-time blanket variables

Current rescue-core regime:

- Phase 2 is used to decide which mesoscopic factors are active via `retained_predictive_factor_set`, `retained_context_factor_set`, and `multiscale_phase3_target_blankets.json`
- those factors are converted into centered covariate surfaces
- those covariates are mapped to transition hook masks

See [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L2374), [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L2598), and [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L2868).

So, mathematically, Phase 2 currently influences Phase 3 as:

`Phase2 selection -> covariate inclusion -> transition-logit modifier`

It does **not** currently influence Phase 3 as:

- lagged graph priors
- hidden-driver latent terms
- multiscale uncertainty-aware hierarchy
- observation-selection logic

### 1.7 Current objective

The torch MAP objective is:

`L = observation_fit`
`+ anchor penalties`
`+ diagnosis-flow penalty`
`+ linkage penalty`
`+ suppression penalty`
`+ hierarchy penalty`
`+ stock penalty`
`+ plausibility penalty`
`+ regularization`

See [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py#L4625).

This is good news structurally: Phase 3 already has enough modularity to accept graph-informed priors and dynamic latent modifiers without rewriting the entire engine.

## 2. What Phase 2 Actually Gives Us

Phase 2 now estimates a sparse-plus-low-rank lagged graph on Phase 15 latent innovations:

`epsilon_t = S_1 z_{t-1} + ... + S_L z_{t-L} + L_1 z_{t-1} + ... + L_L z_{t-L} + noise`

where:

- `S_l` = sparse direct lagged edge matrix
- `L_l` = low-rank hidden-driver matrix

See [latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py#L87), [latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py#L200), and [latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py#L262).

The current run artifact shows:

- a block axis of four latent blocks
- lagged direct edges with support counts
- hidden-driver pairs with support counts
- merged target blankets for Phase 3

See [latent_temporal_graph_bundle.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase2/latent_temporal_graph_bundle.json#L1) and [latent_temporal_phase3_target_blankets.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase2/latent_temporal_phase3_target_blankets.json#L1).

This means Phase 2 now carries five mathematically different signal types:

1. sparse lagged directed edges
2. low-rank hidden shared drivers
3. blanket membership
4. scale support across province / region / national
5. edge stability and hidden-driver support counts

These should not all be fed into Phase 3 the same way.

## 3. Ranked Integration Options

## Option 1. Graph-Informed Hazard Priors With Separate Hidden Shock Factors

Rank: `1`

This is the best next design.

### Representation

Let `b` index Phase 15 latent blocks, and let `z_b[p,t]` be the Phase 15 v2 province-month block state.

Define graph-informed transition modifiers:

`g_r[p,t] = sum_{b,l} Gamma[r,b,l] * z_b[p,t-l]`

where:

- `r` is a Phase 3 transition
- `l` is lag
- `Gamma[r,b,l]` is a learned coefficient

Use Phase 2 sparse edges only to set the **prior structure** on `Gamma`, not the final value.

Then modify the transition logit:

`eta[...,t,r] = existing_terms[...,t,r] + g_r[p,t] + h_r[p,t]`

where `h_r[p,t]` is a hidden-driver term defined below.

### Hidden-driver integration

Do **not** treat low-rank hidden pairs as causal edges.

Instead introduce low-rank dynamic shock factors:

`u_m[t] = rho_m * u_m[t-1] + xi_m[t]`

`h_r[p,t] = sum_m Lambda_hidden[r,m] * W_hidden[p,m] * u_m[t]`

The hidden rank `M` is chosen from Phase 2 hidden-rank summaries by scale.

### Priors

For direct edges:

`Gamma[r,b,l] ~ Normal(mu_phase2[r,b,l], sigma_phase2[r,b,l]^2)`

where:

- `mu_phase2` is derived from edge sign and weight if the block belongs to the target blanket
- `sigma_phase2` shrinks with edge stability and multiscale support

For unsupported block-lag pairs:

`Gamma[r,b,l] ~ Horseshoe(very strong shrinkage)`

This is the cleanest use of Phase 2 because:

- sparse edges become priors on direct mechanistic modifiers
- hidden low-rank structure becomes dynamic latent confounding/stress structure
- blanket membership becomes a variable selection gate
- scale support becomes prior strength, not causal truth

### Why this is best

It preserves the meaning of each object:

- `S` means direct predictive dependence
- `L` means shared omitted-driver structure
- blankets mean candidate relevance
- support means certainty, not effect size

It also fits the existing Phase 3 architecture with minimal distortion because Phase 3 already works through additive transition logits.

## Option 2. Joint Graph-Augmented State-Space Model

Rank: `2`

This is mathematically elegant but heavier.

### Representation

Promote Phase 15 block states to explicit internal Phase 3 latent covariates:

`z_b[p,t] = sum_l A_b[l] z_b[p,t-l] + sum_{b' != b, l} S[b',b,l] z_{b'}[p,t-l] + sum_m R[b,m] u_m[t] + e_b[p,t]`

Then let Phase 3 transitions depend on `z_b[p,t]` directly:

`eta[...,t,r] = existing_terms[...,t,r] + sum_b Beta[r,b] * z_b[p,t]`

### Why it is not ranked first

- this duplicates part of Phase 15 inside Phase 3
- it risks inconsistent latent semantics unless Phase 15 and Phase 3 are jointly fit
- it is the right long-run model, but not the best immediate integration move

## Option 3. Blanket-Gated Modifier Selection With Graph-Regularized Coefficients

Rank: `3`

This is the conservative implementation path.

### Representation

Keep the current covariate surfaces, but change the regularization:

`eta[...,t,r] = existing_terms[...,t,r] + sum_b Beta[r,b] * z_b[p,t]`

and penalize:

`Penalty = sum_{r,b,l} omega[r,b,l] * |Beta[r,b,l]|`

where:

- low `omega` for supported Phase 2 edges
- high `omega` for unsupported edges

Use hidden-driver support only to widen posterior uncertainty, not to add structure.

### Why it is weaker

This helps selection, but it does not use the lag structure or hidden low-rank structure in a mathematically faithful way.

## Option 4. Observation-Selection Only

Rank: `4`

Use Phase 2 only to decide which block-derived covariates are observed strongly enough to enter Phase 3 and which ones should be latent-only.

This is too weak given how much dynamic graph structure Phase 2 now provides.

## 4. Best Options: Exact Math Sketches

## Best Option A: Sparse Edge Priors + Hidden Dynamic Shock Layer

### 4.1 Block-to-transition mapping

Let:

- `B` = Phase 15 blocks
- `R` = Phase 3 transitions
- `L` = maximum lag from Phase 2

Construct a transition-hook mapping matrix:

`M[r,b] in [0,1]`

This says how much block `b` is allowed to affect transition `r`.

Current hook masks already exist in Phase 3 metadata, so this extends current practice rather than replacing it.

### 4.2 Direct graph modifier

For each province and month:

`g_r[p,t] = sum_{b in blanket(r)} sum_{l=1}^L M[r,b] * Gamma[r,b,l] * z_b[p,t-l]`

with prior:

`Gamma[r,b,l] ~ Normal(mu[r,b,l], tau[r,b,l]^{-1})`

Set:

`mu[r,b,l] = c_mu * sign(w_phase2[b,l]) * |w_phase2[b,l]| * support_scale[b,l]`

`tau[r,b,l] = tau_base * (1 + c_tau * stability[b,l] * multiscale_support[b,l])`

Meaning:

- stronger, stable, multiscale-supported edges get tighter priors away from zero
- unsupported edges get strong shrinkage to zero

### 4.3 Hidden-driver term

Let `u_m[t]` be latent hidden shocks:

`u_m[t] = rho_m u_m[t-1] + xi_m[t]`

`xi_m[t] ~ Normal(0, sigma_m^2)`

Then:

`h_r[p,t] = sum_m Lambda_hidden[r,m] * Q[p,m] * u_m[t]`

where:

- `m = 1,...,M`
- `M` is informed by `estimated_hidden_rank`
- `Q[p,m]` may be province loadings, region loadings, or a hierarchy

This is the right place to use hidden-driver low-rank structure.

### 4.4 Transition equation

The full Phase 3 transition logit becomes:

`eta[...,t,r]`
`= eta_existing[...,t,r]`
`+ g_r[p,t]`
`+ h_r[p,t]`

Then:

`pi[...,t,r] = sigmoid(eta[...,t,r])`

### 4.5 Uncertainty propagation

If Phase 2 reports edge support count `s` and stability `q`, define prior scale:

`sigma_direct[r,b,l] = sigma_max / (1 + a1*s + a2*q)`

and if hidden support is weak:

`sigma_hidden[m]` larger, `rho_m` shrunk toward zero

Thus Phase 2 uncertainty becomes prior variance, not effect magnitude.

## Best Option B: Blanket-Gated Graph-Regularized Modifier Layer

This is the best lower-risk implementation if full shock augmentation is too much for the next cycle.

### 4.6 Modifier equation

Build block covariates:

`x_b[p,t,l] = z_b[p,t-l]`

Then:

`eta[...,t,r] = eta_existing[...,t,r] + sum_{b,l} Beta[r,b,l] * x_b[p,t,l]`

Estimate `Beta` by MAP under graph-aware penalty:

`Penalty(Beta)`
`= lambda1 * sum_{supported} w1[r,b,l] * Beta[r,b,l]^2`
`+ lambda2 * sum_{unsupported} w2[r,b,l] * |Beta[r,b,l]|`

with:

- low penalty on Phase 2-supported edges
- high penalty on unsupported edges
- blanket exclusion for blocks outside the target blanket

### 4.7 Hidden-driver use

Do not add hidden drivers directly here.

Instead add a variance inflation term to the observation/transition regularization:

`tau_obs^{-1} = tau_base^{-1} + c_hidden * hidden_driver_support(target)`

This is weaker than Option A, but still mathematically coherent.

## 5. How Each Phase 2 Output Should Be Used

### Sparse lagged edges

Use as:

- priors on transition modifier coefficients
- lag-specific eligibility masks

Do not use as hard deterministic edges inside the mechanistic cascade.

### Hidden-driver low-rank terms

Use as:

- latent shock channels
- uncertainty inflation channels

Do not use as direct causal hooks.

### Target blankets

Use as:

- selection gates on which block-lag modifiers can enter each target-specific transition family

This is the cleanest existing Phase 2-to-Phase 3 bridge.

### Multiscale support

Use as:

- prior precision calibration
- hierarchy level selection

Example:

- if support is mostly national, prefer national-level modifier or shock
- if support is strong at province scale, allow province-specific modifier

### Uncertainty

Use as:

- prior variance on graph-informed coefficients
- posterior widening in forecast and peak-search outputs

Current Phase 3 does not do this yet. It mostly consumes selection and covariate surfaces, not Phase 2 uncertainty.

## 6. Recommended Immediate Plan

1. Keep the current Phase 3 state-space and hazard engine intact.
2. Replace the current Phase 2-to-Phase 3 bridge from “selected factor surfaces only” to “selected factor surfaces plus graph-informed priors”.
3. Implement `Option 1` first:
   - sparse lagged edges -> priors on `Gamma[r,b,l]`
   - hidden-driver rank/support -> latent shock layer `u_m[t]`
   - blanket blocks -> inclusion mask
   - multiscale support/stability -> prior precision
4. Use `Option 3` only as a fallback if the full hidden shock layer is too expensive.

## 7. Bottom Line

The current Phase 3 model is already mathematically rich enough to absorb Phase 2 correctly. The main mismatch is not lack of machinery; it is that Phase 2 is still being used mostly as a covariate selector rather than as dynamic structural information.

The most coherent next version is:

`Phase15 latent block states`
`-> Phase2 sparse lagged graph + low-rank hidden structure`
`-> Phase3 graph-informed transition priors + hidden dynamic shocks`
`-> mechanistic semi-Markov cascade`

That preserves the semantics of every layer and avoids the main category error:

using hidden low-rank dependence as if it were a direct causal transition modifier.
