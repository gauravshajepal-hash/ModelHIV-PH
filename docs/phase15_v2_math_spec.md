# Phase 15 v2 Mathematical Specification

## Purpose

`Phase 15 v2` replaces the current `signed_weighted_national_scaffold_v1` and `province_factor_graph_scaffold_v1` heuristics with a mixed-frequency hierarchical state-space model.

The fundamental latent state is the province-month block state:

- `z_prov[p,b,t]`: province `p`, latent block `b`, month `t`

Derived latent states:

- `z_reg[r,b,t]`: region `r`, block `b`, month `t`
- `z_nat[b,t]`: national block `b`, month `t`

The central design choice is that **province states are fundamental** and region/national states are **bottom-up aggregates** of province states.

## Equations

### E1. National state dynamics

`z_nat[b,t] = a_b + phi_b * z_nat[b,t-1] + eps_nat[b,t]`

English:

- each latent block has a national monthly trajectory
- `a_b` is the block intercept
- `phi_b` is the persistence
- `eps_nat[b,t]` is national innovation noise

### E2. Region deviation dynamics

`d_reg[r,b,t] = psi_b * d_reg[r,b,t-1] + eps_reg[r,b,t]`

English:

- each region carries a deviation from the national path
- the deviation is persistent, but it is regularized through an estimated innovation variance rather than a fixed shrinkage constant

Constraint:

`sum_r Omega[r,t] * d_reg[r,b,t] = 0`

This keeps region deviations centered so they do not redefine the national mean.

### E3. Province deviation dynamics

`d_prov[p,b,t] = rho_b * d_prov[p,b,t-1] + eps_prov[p,b,t]`

English:

- each province also carries a local deviation around its region and national path
- this is the term that prevents provinces from collapsing onto NCR-dominated national behavior

Constraint:

`sum_{p in r} omega_reg[p|r,t] * d_prov[p,b,t] = 0`

This keeps province deviations centered within each region.

### E4. Province latent construction

`z_prov[p,b,t] = z_nat[b,t] + d_reg[r(p),b,t] + d_prov[p,b,t]`

English:

- a province state is the national state plus a region adjustment plus a province adjustment
- this is the core decomposition used for smoothing and hierarchy

### E5. Bottom-up aggregation

`z_reg[r,b,t] = sum_{p in r} omega_reg[p|r,t] * z_prov[p,b,t]`

`z_nat[b,t] = sum_p omega_nat[p,t] * z_prov[p,b,t]`

English:

- region and national states are weighted averages of province states
- they are not separate top-down estimates pushed downward afterward

Weights:

- `omega_reg[p|r,t] >= 0`
- `sum_{p in r} omega_reg[p|r,t] = 1`
- `omega_nat[p,t] >= 0`
- `sum_p omega_nat[p,t] = 1`

In v2 these weights remain scaffolded from explicit geography and population-proxy rules, because the current data are too sparse to estimate them freely.

### E6. Mixed-frequency observation operator

For each observed row `i`, define an operator:

`A_i(z) = sum_{p,t} H_i[p,t] * z_prov[p,b(i),t]`

English:

- `H_i[p,t]` says how much row `i` depends on province `p` and month `t`
- for a monthly province row, `H_i` puts mass on one province-month cell
- for a region-year row, `H_i` distributes mass only across province-month cells inside that region and year
- for a national annual row, `H_i` distributes mass only across the covered provinces and months

Constraints:

- `H_i[p,t] >= 0`
- `sum_{p,t} H_i[p,t] = 1`

This replaces the current annual fan-out rule.

### E7. Measurement model

`y_i = alpha_j + lambda_j * A_i(z) + eps_i`

where `j = indicator(i)`.

English:

- every observed row is a noisy measurement of the latent block through an indicator-specific intercept `alpha_j` and loading `lambda_j`
- the same equation works for province, region, national, monthly, annual, and survey-wave rows because the support is handled by `H_i`

### E8. Signed loading parameterization

For indicators with a positive or negative sign prior:

`lambda_j = s_j * softplus(eta_j)`

with `s_j in {-1, +1}`.

For neutral indicators:

`lambda_j = eta_j`

English:

- `s_j` is the sign prior from Phase 0 evidence and the latent-block spec
- `eta_j` is estimated from data
- `softplus` enforces positive magnitude while keeping optimization smooth

This replaces the current fixed `+1/-1` sign multiplier with heuristic support weights.

### E9. Estimated observation precision

`log sigma_i = kappa_0 + kappa_j + kappa_role[role_i] + kappa_source[source_i] + kappa_geo[geo_i] + kappa_time[time_i]`

and

`eps_i ~ Normal(0, sigma_i^2)`

English:

- observation noise is estimated
- direct indicators, proxy indicators, different source classes, and different temporal/geographic supports can have different noise levels

This replaces the current fixed precision constants.

### E10. Estimated smoothing scales

`tau_nat[b], tau_reg[b], tau_prov[b], tau_loading[b]`

are all estimated under hierarchical priors.

English:

- these scales govern how much national, region, and province paths are allowed to move
- they replace the current fixed values such as `4.0`, `2.0`, `0.25`, and `1.75`

## What Remains Scaffolded

These parts remain explicitly scaffolded in v2:

- indicator-to-block membership
- indicator sign priors
- province-to-region membership
- row support masks for `H_i`
- aggregation weight source strategy

Reason:

- the current evidence does not identify unrestricted block discovery or unrestricted aggregation weights reliably enough

## What Is Estimated

These quantities are estimated in v2:

- national latent trajectories `z_nat[b,t]`
- region deviations `d_reg[r,b,t]`
- province deviations `d_prov[p,b,t]`
- indicator intercepts `alpha_j`
- loading magnitudes through `eta_j`
- persistence terms `phi_b`, `psi_b`, `rho_b`
- innovation scales `tau_nat[b]`, `tau_reg[b]`, `tau_prov[b]`
- observation variance terms `kappa_*`

## Keep-Or-Revert Gates

The replacement should only be kept if:

- bottom-up coherence residual is effectively zero
- loading sign stability exceeds the configured threshold
- held-out log score improves relative to the scaffold
- calibration improves or at least does not degrade
- peak-window and turning-point metrics improve
- provinces with weak support do not collapse to a false national certainty
