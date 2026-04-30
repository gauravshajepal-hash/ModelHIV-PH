# Phase 3 TR-V3 Autoresearch Design

**Date:** 2026-04-10  
**Author:** Epigraph PH (G S + AI agent)  
**Status:** Draft
**Selected autoresearch variant:** `evidence-to-model-loop`

---

## 0. Document Contract

This memo is the canonical TR-V3 design document.

Use this file for:

- model structure
- mathematics
- identifiability assumptions
- data-contract rules
- benchmark gates
- experiment meaning

Use [AUTORESEARCH.md](/D:/EpiGraph_PH/Phase3(dynamic)/AUTORESEARCH.md) only as the operational execution manifest:

- experiment IDs
- live execution order
- current implementation status
- run-contract names

If the two files disagree:

- this design memo is correct
- the operational manifest must be updated

This split exists to prevent a common failure mode:

- the execution file slowly becomes a second design memo
- the two docs drift
- and the loop starts running experiments that no longer match the scientific spec

The current benchmark frontier adds a second explicit split:

- predictive benchmark candidates may differ by contract
- mechanistic research candidates may remain scientifically valuable even when they are not the best predictive rows

So the repo now needs three named lanes:

- `exact_only` predictive benchmark lane
- `dense_train_observed_score` predictive benchmark lane
- mechanistic research lane

---

## 1. Purpose

This memo defines the next Phase 3 frontier after the stricter blocked-time contract was accepted.

The central scientific fact is now:

- the stricter contract fixed publication-honesty issues;
- the stricter contract also exposed a real weakness;
- the current national mechanistic branch is still not strong enough to beat carry-forward on mean blocked-time evaluation.

This memo therefore does **not** propose a larger blind search over Phase 2 features.

It proposes a new model family, `TR-V3`, with a stronger time-series backbone:

- dynamic hazard baseline,
- learned observation model,
- explicit shock layer,
- then Phase 2 direct priors,
- then optional hidden-driver channels.

The goal is practical, not decorative:

- beat simple baselines under blocked-time evaluation,
- preserve explicit uncertainty,
- keep mixed-evidence honesty,
- and produce a reproducible benchmark that policymakers can inspect.

---

## 2. Executive Position

The current `TR-V2` family is still too close to:

```text
carry-forward hazard map + small structured offsets
```

That is why it can help on later stable splits while still failing badly on the `2021 -> 2022` blocked-time split.

Evidence:

- [TR-V2 rolling-origin report](/D:/EpiGraph_PH/artifacts/runs/tr-v2-rolling-origin-cli-20260409-s00/analysis/tr_v2_rolling_origin_report.md#L7)
- current train-only carry-forward baseline construction in [tr_v2.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/tr_v2.py#L191)
- current direct prior fit in [tr_v2.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/tr_v2.py#L399)
- current holdout simulation in [tr_v2.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/tr_v2.py#L616)

The next frontier should therefore **not** be:

- more broad `rescue_core`,
- more generic modifier covariates,
- or a larger Phase 2 search surface.

The next frontier should be:

```text
stronger train-only dynamic national hazard model
+ explicit shock representation
+ learned observation/readout model
+ Phase 2 as structured prior layer
```

---

## 3. Why Borrow From Other Fields

The point is not to copy astronomy, statistical physics, or mathematics as branding.

The point is to import specific motifs that solve the exact failure mode we now see.

### 3.1 Astronomy

Useful motifs:

- adaptive change-point detection in noisy time series
- latent stochastic time-series models for irregular, uncertain observations

What this contributes to HIV Phase 3:

- better regime-shift detection,
- stronger train-only forecasting under noisy observations,
- explicit uncertainty propagation from latent state to observed readout.

Representative examples:

- Bayesian Blocks style adaptive segmentation
- damped/stochastic latent-process time-series modeling

### 3.2 Statistical Physics

Useful motifs:

- mean-reverting latent deviations
- forced response plus endogenous relaxation
- shock processes superposed on smooth baseline dynamics

What this contributes:

- a mathematically clean way to represent temporary service disruptions, backlog releases, and diagnosis shocks without pretending the whole system permanently changed.

### 3.3 Mathematics / Statistics

Useful motifs:

- locally adaptive trend estimation
- trend filtering
- piecewise-smooth latent trajectories with sparse breaks

What this contributes:

- a hazard model that is smooth most of the time,
- but still allowed to bend or jump when the data genuinely require it.

---

## 4. Current TR-V2 Failure Pattern

The live `TR-V2` blocked-time summary is:

| Variant | Mean MAE | Median MAE | Wins vs no priors |
|---|---:|---:|---:|
| `no_priors` | `1.048634` | `0.175597` | `0` |
| `direct_only` | `1.570368` | `0.191663` | `2` |
| `direct_plus_multiscale` | `1.565610` | `0.191814` | `2` |
| `direct_plus_hidden` | `1.603070` | `0.191663` | `2` |
| `direct_plus_hidden_plus_multiscale` | `1.598180` | `0.191814` | `2` |

Source:

- [TR-V2 rolling-origin report](/D:/EpiGraph_PH/artifacts/runs/tr-v2-rolling-origin-cli-20260409-s00/analysis/tr_v2_rolling_origin_report.md#L7)

Worst split:

| Train End | Holdout | No Priors MAE | Direct MAE |
|---|---|---:|---:|
| `2021` | `2022` | `3.796171` | `5.858825` |

Interpretation:

- on stable later splits, structured priors can help;
- on the difficult regime-break split, the model family is still too rigid and too close to carry-forward logic.

This is a **model-family** weakness, not just a Phase 2 weakness.

---

## 5. TR-V3 Mathematical Design

The new family should be built around quarter-level national latent hazard states.

For each transition:

```text
r in {U_to_D, D_to_A, A_to_V, A_to_L, L_to_A}
```

define:

```text
eta_r(t) = logit(h_r(t))
```

where `h_r(t)` is the transition hazard.

### 5.1 Dynamic baseline

The dynamic baseline should be:

\[
\eta_r(t) = \ell_r(t) + q_r(t) + d_r(t)
\]

where:

- `\ell_r(t)` is smooth latent trend,
- `q_r(t)` is shock / regime component,
- `d_r(t)` is optional structured Phase 2 deviation term.

The initial `TR-V3-00` model uses only:

\[
\eta_r(t) = \ell_r(t)
\]

with local-linear-trend dynamics:

\[
\ell_r(t+1) = \ell_r(t) + s_r(t) + \epsilon_r(t)
\]

\[
s_r(t+1) = s_r(t) + \zeta_r(t)
\]

English:

- hazard level changes over time;
- hazard slope also changes over time;
- the model can extrapolate a trend instead of copying the last value.

This is stronger than the current last-train carry-forward baseline in [tr_v2.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/tr_v2.py#L191).

### 5.2 Shock layer

`TR-V3-02` adds an explicit shock component:

\[
q_r(t+1) = \phi_r q_r(t) + \nu_r(t)
\]

or equivalently as sparse segmented shocks:

\[
q_r(t) = \sum_k \gamma_{r,k} \mathbf{1}\{t \in B_k\}
\]

where:

- `B_k` are change-point-defined blocks,
- `\gamma_{r,k}` are transition-specific shock amplitudes.

English:

- most of the time the hazard evolves smoothly;
- in exceptional periods, it can jump or relax sharply.

This is the direct mathematical response to the `2021 -> 2022` failure.

### 5.3 Phase 2 direct priors

`TR-V3-03` adds direct Phase 2 block-lag effects:

\[
d_r(t) = \sum_{(b,l)\in \mathcal{E}_r}\Gamma_{r,b,l} z_b(t-l)
\]

with priors:

\[
\Gamma_{r,b,l} \sim \mathcal{N}(\mu^{(phase2)}_{r,b,l}, \sigma^{2\,(phase2)}_{r,b,l})
\]

where:

- `z_b(t-l)` is the quarterized national Phase 15 latent block state,
- `\mathcal{E}_r` is the admissible block-lag set for transition `r`,
- prior mean and variance come from Phase 2 support, weight, stability, and cross-scale agreement.

English:

- Phase 2 says which lagged block effects are plausible;
- the dynamic hazard model decides how strongly they survive blocked-time fitting.

This preserves the current explicit plugin map idea from [hiv.py](/D:/EpiGraph_PH/src/epigraph_ph/plugins/hiv.py#L1677), but attaches it to a better baseline family.

### 5.4 Hidden-driver channels

`TR-V3-04` adds hidden-driver channels:

\[
\eta_r(t) = \ell_r(t) + q_r(t) + \sum_{(b,l)\in \mathcal{E}_r}\Gamma_{r,b,l} z_b(t-l) + \sum_m \Lambda_{r,m} u_m(t)
\]

with:

\[
u_m(t+1) = \rho_m u_m(t) + \xi_m(t)
\]

English:

- `u_m(t)` are shared latent temporal pressures;
- they are not direct mechanistic edges;
- they are allowed only if they improve blocked-time holdout beyond direct-only.

### 5.5 State evolution

The compartment update remains flow-based:

\[
U_{t+1} = U_t - f_{UD}(t)
\]
\[
D_{t+1} = D_t + f_{UD}(t) - f_{DA}(t)
\]
\[
A_{t+1} = A_t + f_{DA}(t) + f_{LA}(t) - f_{AV}(t) - f_{AL}(t)
\]
\[
V_{t+1} = V_t + f_{AV}(t)
\]
\[
L_{t+1} = L_t + f_{AL}(t) - f_{LA}(t)
\]

where each flow is:

\[
f_r(t) = h_r(t)\cdot \text{eligible stock}_r(t)
\]

This keeps the U/D/A/V/L mechanistic interpretation intact.

### 5.6 Observation model

`TR-V3-01` replaces static readout shortcuts with learned observation equations.

Observed targets:

\[
y^{diag}_t \sim \mathcal{N}(D_t + A_t + V_t + L_t,\ \sigma^2_{diag,t})
\]

\[
y^{art}_t \sim \mathcal{N}(A_t + V_t,\ \sigma^2_{art,t})
\]

\[
y^{vs}_t \sim \mathcal{N}(V_t,\ \sigma^2_{vs,t})
\]

Testing readout:

\[
y^{test}_t \sim \text{Binomial}(A_t + V_t,\ p_{test}(t))
\]

\[
\text{logit}(p_{test}(t)) = a_{test} + b_{test} t + u_{test}(t)
\]

English:

- diagnosed, ART, suppression, and testing are not fixed deterministic fractions forever;
- each is a noisy observation of latent states, with its own dynamics.

This is the clean replacement for static testing-share logic like [tr_v2.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/tr_v2.py#L660).

---

## 6. Autoresearch Program

This is a Karpathy-style autoresearch loop in the narrow sense:

- one bounded optimization unit,
- one trusted benchmark harness,
- one keep-or-revert rule,
- repeated mutation only when the benchmark is honest.

### 6.1 Optimization unit

The optimization unit is:

```text
TR-V3 national blocked-time frontier family
```

not:

- all of Phase 3,
- not `rescue_core`,
- not a broad Phase 2 search,
- not province-level primary forecasting.

### 6.2 Trusted evaluation harness

Primary harness:

- full blocked-time rolling-origin benchmark for `2017-2025`
- [rolling-origin report](/D:/EpiGraph_PH/artifacts/runs/tr-v2-rolling-origin-cli-20260409-s00/analysis/tr_v2_rolling_origin_report.md#L7)

Secondary contextual harness:

- `2010-2016` early-history partial-observation benchmark
- [early-history report](/D:/EpiGraph_PH/artifacts/runs/tr-v2-early-history-cli-20260409-s00/analysis/tr_v2_early_history_partial_report.md)

Dashboard:

- [combined dashboard](/D:/EpiGraph_PH/artifacts/runs/tr-v2-benchmark-dashboard-cli-20260409-s00/analysis/tr_v2_benchmark_dashboard_report.md)

### 6.3 Keep-or-revert rule

Keep a mutation only if:

1. it improves mean blocked-time MAE relative to current `no_priors` baseline;
2. it does not worsen the worst split catastrophically;
3. it preserves honest contract wording;
4. it improves or preserves interpretability.

Revert if:

- the gain only appears on one late split,
- the `2021 -> 2022` split gets worse,
- or the observation model becomes less defensible.

### 6.4 Mutation order

Strict order:

1. `TR-V3-00` dynamic baseline only
2. `TR-V3-01` learned observation model
3. `TR-V3-02` shock layer
4. `TR-V3-03` direct Phase 2 priors
5. `TR-V3-04` hidden-driver channels

Do not skip ahead.

---

## 7. Experiment Ladder

### `TR-V3-00-dynamic-baseline-only`

Purpose:

- replace carried-forward hazard map with train-only dynamic hazard forecast.

Included:

- local-linear-trend hazard states
- no Phase 2
- no hidden channels
- no shock layer yet

Primary question:

- can a stronger train-only baseline beat current `no_priors`?

Expected outputs:

- `dynamic_baseline_summary.json`
- `hazard_state_posterior.json`
- `baseline_comparison.json`
- `evaluation.json`
- `mechanistic_forecast.json`

### `TR-V3-01-dynamic-baseline-plus-observation-model`

Purpose:

- reduce readout mismatch by learning the observation layer.

Included:

- `TR-V3-00`
- learned testing process
- learned readout noise / observation equations

Primary question:

- does removing static testing-share shortcuts improve blocked-time fit?

Expected outputs:

- `observation_model_summary.json`
- `observation_fit_diagnostics.json`
- standard evaluation artifacts

### `TR-V3-02-dynamic-baseline-plus-shocks`

Purpose:

- add explicit regime-shock structure.

Included:

- `TR-V3-01`
- smooth hazard trend
- shock channel / change-point component

Primary question:

- does the family stop failing catastrophically on `2021 -> 2022`?

Expected outputs:

- `shock_layer_summary.json`
- `shock_activation_table.json`
- `blocked_time_split_diagnostics.json`

### `TR-V3-03-phase2-direct-priors-on-dynamic-hazards`

Purpose:

- add Phase 2 direct priors only after the dynamic baseline is credible.

Included:

- `TR-V3-02`
- direct block-lag Phase 2 priors on hazard deviations

Primary question:

- does Phase 2 help more when attached to a stronger baseline family?

Expected outputs:

- `phase2_direct_hazard_prior_summary.json`
- `phase2_dynamic_prior_effects.json`
- standard evaluation artifacts

### `TR-V3-04-phase2-hidden-driver-channels`

Purpose:

- evaluate whether hidden channels add anything beyond direct priors.

Included:

- `TR-V3-03`
- hidden-driver channels as separate latent shock basis

Primary question:

- do hidden-driver channels improve blocked-time holdout beyond direct-only?

Expected outputs:

- `phase2_hidden_shock_summary.json`
- `hidden_channel_ablation.json`
- standard evaluation artifacts

### `TR-V3-05a-corrected-mechanism-control-layer`

Purpose:

- build on the best completed sandbox family, `TR-V3-04d`;
- add explicit hazard-side mechanism controls that the current family still lacks;
- avoid re-running the old `dynamic baseline -> observation -> shock -> priors -> hidden` ladder under a new name.

Correction to the earlier decomposition idea:

- `trend + seasonal + residual` is not the right default framing for HIV here;
- most of that proposal overlaps with work already completed in `TR-V3-00` through `TR-V3-04d`;
- the genuine missing piece is a compact mechanism-control layer acting on hazards, not another generic seasonal or residual term.

What is already present in the completed family:

- slow transition drift via the dynamic AR-trend backbone;
- explicit shock logic;
- learned observation/readout calibration;
- Phase 2 direct priors;
- selective hidden-driver channels.

What is still missing:

- explicit low-dimensional hazard-side controls for:
  - ascertainment pressure;
  - care-system pressure;
  - reporting / registry artifact pressure;
  - a stronger infection / network-pressure story, which should enter the overall family through incidence generation rather than as a generic hazard tweak.

Literature-informed correction:

- `N(t)` is not a decorative add-on;
- the HIV literature reviewed for this repo points in the same direction:
  - diagnosis and care do not create the epidemic;
  - upstream infection pressure does;
  - treatment and suppression must feed back into future infection generation;
- therefore `N(t)` is a first-class latent object in the overall `TR-V3` program, even if it is deferred out of the first `05a` hazard-control ablation for identifiability reasons.

Rejected default:

- no unconditional sinusoidal or seasonal harmonic term;
- no claim of biological HIV seasonality unless a separate ablation supports it;
- if a periodic pattern appears, treat it first as reporting or operations artifact, not transmission biology.

Corrected mathematical form:

\[
\eta_r(t) = \operatorname{logit} h_r(t)
\]

For `TR-V3-05a`, the default should be:

\[
\eta_r(t)
=
\eta^{(04d)}_r(t)
+
\beta^A_r A(t)
+
\beta^C_r C(t)
+
\beta^R_r R(t)
\]

with `N(t)` deliberately deferred out of the hazard equation in `05a`.

Reason:

- `05a` is the hazard-side `A/C/R` control experiment;
- `N(t)` should primarily enter the full family through the incidence source term in `TR-V3-05b`;
- that keeps the mechanistic decomposition cleaner and avoids turning `N(t)` into a generic catch-all covariate.

Channel definitions:

- `A(t)` = ascertainment / diagnosis pressure
- `C(t)` = care-system performance pressure
- `R(t)` = reporting / registry artifact pressure
- `N(t)` = latent infection / network pressure for the overall program, but not a default hazard-side control in `05a`

Transition-specific admissibility:

- `U -> D` may use `A(t)` and `R(t)`
- `D -> A` may use `C(t)` and `R(t)`
- `A -> V` may use `C(t)` and `R(t)`
- `A -> L` may use `-C(t)` and `R(t)` and existing shock terms
- `L -> A` may use `C(t)` and `R(t)` and existing shock terms

`N(t)` is reserved for the incidence-generation block in `05b` and later full-family variants.

First control definitions:

\[
A(t) = w_{A,1}\,\widetilde{\Delta \log(\text{diagnosed\_plhiv})}(t)
+ w_{A,2}\,\widetilde{\text{new\_diagnosed\_cases\_period}}(t)
+ w_{A,3}\,\widetilde{\text{testing-related share}}(t)
\]

\[
C(t) = w_{C,1}\,\widetilde{\text{alive\_on\_art share}}(t)
+ w_{C,2}\,\widetilde{\text{viral-load-tested share}}(t)
+ w_{C,3}\,\widetilde{\text{virally-suppressed share}}(t)
\]

\[
R(t) = w_{R,1}\,\widetilde{\text{quarter-end imbalance}}(t)
+ w_{R,2}\,\widetilde{\text{stock-flow discrepancy}}(t)
+ w_{R,3}\,\widetilde{\text{reporting dummy}}(t)
\]

Discipline:

- fit all control construction on training data only;
- standardize on training statistics only;
- keep the control basis small and strongly shrunk;
- inject controls into hazards before simulation, not only into the readout layer.

Primary question:

- does explicit hazard-side `A/C/R` control improve blocked-time performance beyond `TR-V3-04d` without reintroducing late-tail overreaction?

Expected outputs:

- `mechanism_control_summary.json`
- `hazard_control_coefficients.json`
- `control_channel_diagnostics.json`
- standard evaluation artifacts

### `TR-V3-05b-open-leaky-incidence-flow-model`

Purpose:

- add the missing mechanistic incidence source term and leakage structure;
- separate infection generation from diagnosis pressure;
- keep the system identifiable by using an open infected-population model first, rather than forcing a full susceptible compartment immediately.

Literature-informed justification for `N(t)`:

- HIV infection generation is upstream of diagnosis;
- network structure, suppression coverage, and risky-contact structure change future infections even when diagnosis pressure is unchanged;
- youth-focused and social-structural HIV models support treating infection pressure as a distinct mechanism, not as a synonym for testing or reporting;
- therefore the open incidence-flow model is the first place where `N(t)` becomes non-optional in the `TR-V3` roadmap.

Important scope decision:

- do **not** add a fully explicit `S(t)` compartment yet;
- add `S(t)` only after stronger denominator data exist;
- until then, use an open incidence inflow into `U(t)` anchored by incidence and prevalence signals.

State vector:

- `U(t)` = undiagnosed infected
- `D(t)` = diagnosed
- `A(t)` = on ART and not suppressed
- `V(t)` = virally suppressed
- `L(t)` = lost to follow-up
- `M_{cum}(t)` = cumulative HIV deaths, optional accounting state

State equations:

\[
U_{t+1}=U_t+\iota(t)-f_{UD}(t)-m_U(t)U_t
\]

\[
D_{t+1}=D_t+f_{UD}(t)-f_{DA}(t)-f_{DL}(t)-m_D(t)D_t
\]

\[
A_{t+1}=A_t+f_{DA}(t)+f_{LA}(t)-f_{AV}(t)-f_{AL}(t)-m_A(t)A_t
\]

\[
V_{t+1}=V_t+f_{AV}(t)-f_{VL}(t)-m_V(t)V_t
\]

\[
L_{t+1}=L_t+f_{DL}(t)+f_{AL}(t)+f_{VL}(t)-f_{LA}(t)-m_L(t)L_t
\]

\[
M_{cum,t+1}=M_{cum,t}+m_U(t)U_t+m_D(t)D_t+m_A(t)A_t+m_V(t)V_t+m_L(t)L_t
\]

where

\[
f_r(t)=h_r(t)\cdot X_r(t)
\]

and `X_r(t)` is the eligible stock for transition `r`.

Incidence inflow:

\[
\iota(t)=\operatorname{softplus}\Big(\alpha_I+\beta_N N(t)+\beta_I\log(I_{\mathrm{eff}}(t)+\varepsilon)+\beta_P\log(P(t)+\varepsilon)\Big)
\]

\[
I_{\mathrm{eff}}(t)=w_UU(t)+w_DD(t)+w_AA(t)+w_VV(t)+w_LL(t)
\]

Interpretation:

- `N(t)` = latent infection / network pressure
- `I_{\mathrm{eff}}(t)` = effective infectious pool
- `P(t)` = population or exposure-pool scale
- `\iota(t)` enters `U(t)` and is strictly nonnegative

Key separation:

- `N(t)` creates infections;
- `A_c(t)` and `R_c(t)` govern diagnosis and reporting;
- `C_c(t)` governs treatment, suppression, retention, and re-entry.

This separation is mandatory.

If the fitted model only works by letting diagnosis or reporting absorb what should be infection pressure, the fit should be treated as non-identifiable and rejected.

Hazard parameterization:

\[
h_r(t)=h_{r,\min}+\big(h_{r,\max}-h_{r,\min}\big)\sigma(\eta_r(t))
\]

\[
\eta_r(t)=\eta_r^{(04d)}(t)+\beta_r^A A_c(t)+\beta_r^C C_c(t)+\beta_r^R R_c(t)
\]

Default transition set for `05b`:

- `U -> D`
- `D -> A`
- `D -> L`
- `A -> V`
- `A -> L`
- `V -> L`
- `L -> A`

Mortality parameterization:

\[
m_s(t)=\kappa_s\,m_0(t)
\]

\[
m_0(t)=\operatorname{softplus}\big(\alpha_M+\beta_M^R R_c(t)\big)
\]

with monotone severity constraint:

\[
0 \le \kappa_V \le \kappa_A \le \kappa_D \le \kappa_U \le \kappa_L
\]

This avoids trying to estimate five unrelated death hazards freely.

Parameter constraints for first implementation:

- fix `w_U = 1`
- strongly constrain `w_D, w_L` near `0.8-1.0`
- constrain `w_A \in [0.2, 0.7]`
- constrain `w_V \in [0, 0.05]`
- enforce `|\rho_N| < 1`
- standardize `A_c, C_c, R_c` on train only
- use strong shrinkage on all control coefficients
- add explicit `S(t)` only after stronger denominator data are available

Observation equations:

\[
y^{diag}_t \sim \mathcal{N}(D_t+A_t+V_t+L_t,\sigma^2_{diag})
\]

\[
y^{art}_t \sim \mathcal{N}(A_t+V_t,\sigma^2_{art})
\]

\[
y^{vs}_t \sim \mathcal{N}(V_t,\sigma^2_{vs})
\]

\[
y^{diagflow}_t \sim \mathcal{N}(f_{UD}(t),\sigma^2_{diagflow})
\]

\[
y^{death}_t \sim \mathcal{N}\Big(\sum_s m_s(t)X_s(t),\sigma^2_{death}\Big)
\]

\[
y^{inc,annual}_y \sim \mathcal{N}\Big(\sum_{t \in y}\iota(t),\sigma^2_{inc}\Big)
\]

This is the first TR-V3 family that gives infection generation, leakage, and mortality their own explicit roles.

Expected outputs:

- `incidence_inflow_summary.json`
- `leakage_flow_summary.json`
- `mortality_block_summary.json`
- `identifiability_ledger.json`
- standard evaluation artifacts

### `TR-V3-05b-identifiability-ledger`

Purpose:

- document which parameter blocks are primarily identified by which data families;
- prevent the model from fitting infection, diagnosis, leakage, and reporting from the same signal;
- make calibration scientifically auditable rather than relying on fit quality alone.

Core rule:

- each parameter block must have a primary data family, explicit constraints, and declared confounders.

Ledger blocks:

| Block | Parameters / latent objects | Primary data family | Secondary checks | Constraints / priors | Main confounders |
|---|---|---|---|---|---|
| Infection | `N(t)`, incidence intercept, infectivity weights | annual new infections, prevalence trends, youth incidence proxies | PLHIV trend consistency | strong priors on `w_*`, fixed `w_U`, near-zero `w_V`, smooth `N(t)` | diagnosis pressure, reporting artifacts |
| Diagnosis | `h_{UD}(t)`, coefficients on `A_c(t)` and `R_c(t)` | new diagnoses, diagnosed stock growth | testing-related signals | bounded hazards, shrinkage on `A_c/R_c` effects | infection inflow, reporting delay |
| Care | `h_{DA}(t)`, `h_{AV}(t)`, coefficients on `C_c(t)` | ART coverage, VL testing, suppression rates | alive-on-ART stock trajectory | bounded hazards, monotone care-flow priors | reporting artifact, leakage |
| Leakage | `h_{DL}(t)`, `h_{AL}(t)`, `h_{VL}(t)`, `h_{LA}(t)` | retention / loss-to-follow-up statistics where available | longer-horizon stock drift | strong shrinkage, sparse transition set | mortality, care quality, reporting gaps |
| Mortality | `m_0(t)`, `\kappa_s` | HIV mortality and death anchors | long-run prevalence balance | ordered `\kappa_s`, shared baseline mortality | leakage, under-reporting of deaths |
| Reporting | `R_c(t)` and reporting observation effects | quarter-end anomalies, stock-flow discrepancy, known reporting irregularities | residual diagnostic plots | strong shrinkage, no biological interpretation | diagnosis pressure, care pressure |

Calibration sequence:

1. initialize incidence / infection block from annual incidence and prevalence anchors
2. fit diagnosis block to diagnosis flow and diagnosed stock
3. fit care block to ART and suppression targets
4. fit leakage and mortality under strong priors
5. perform constrained joint refinement

Identifiability diagnostics required:

- profile-likelihood or one-at-a-time perturbation for major parameter blocks
- parameter correlation matrix or local Hessian diagnostics
- posterior / ensemble spread if a Bayesian or multi-start stage is used
- ablation showing that removal of a data family destabilizes the intended parameter block

---

### Calibration

Purpose:

- calibrate `TR-V3` in a way that respects identifiability, provenance, and the blocked-time forecasting goal;
- avoid the common failure mode of fitting too many biological and reporting mechanisms from the same late-period signals;
- keep calibration tied to the actual Phase 3 objects rather than generic epidemic-model parameter names.

Calibration principles:

- no ad hoc extra weight on recent years in the primary calibration objective;
- no parameter block is allowed to borrow its main evidence from the same signal as another block without being declared as a confounder;
- no quarterly transition is treated as identified just because a hidden state reconstruction produces a number for it;
- bridge and extrapolated rows may widen training support, but they do not become silent substitutes for exact targets.

Calibration blocks for `TR-V3`:

| Block | What is being calibrated | Primary data | What is fixed or tightly constrained |
|---|---|---|---|
| Infection | `N(t)`, incidence inflow `\iota(t)`, infectivity weights | annual new infections, prevalence trends, youth/incidence proxies | `w_U` fixed, `w_V` near zero, smooth `N(t)` |
| Diagnosis | `U_to_D`, ascertainment effects `A(t)`, reporting effects `R(t)` | diagnosis flow, diagnosed stock growth | bounded hazards, strong shrinkage |
| Care | `D_to_A`, `A_to_V`, care effects `C(t)` | alive-on-ART stock, ART coverage, suppression observations where supported | bounded hazards or direct share reconciliation when quarterly support is weak |
| Leakage | `A_to_L`, `L_to_A`, later richer leakage if support improves | net ART stock drift plus diagnosis-flow support | sparse transition set, strong shrinkage |
| Mortality | shared mortality block only after support improves | annual AIDS deaths | shared baseline, ordered state multipliers |
| Reporting | observation calibration and `R(t)` | stock-flow discrepancy, release timing anomalies | reporting is not allowed to masquerade as biology |

Operational calibration sequence:

1. initialize infection / annual incidence block from annual anchors
2. fit diagnosis block to diagnosis flow and diagnosed stock
3. fit care block using the strongest supported representation available:
   - direct quarterly hazards only when support exists
   - otherwise stock/share reconciliation rather than free quarterly hazard fitting
4. fit leakage only under an honest identification contract
5. defer mortality-rich calibration until the archive supports it
6. run constrained joint refinement only after each block has a valid primary signal

Practical implication:

- the current code should not be described as calibrating classical ODE parameters such as `\beta_H` or `\theta_{ART}` directly;
- it is calibrating transition hazards, observation mappings, and later infection-pressure processes under mixed-evidence support.

Current repair-path conclusion:

- keep `EXP-R1` as the better repair baseline because it fixes the unsupported-zero hazard explosion;
- do not keep `EXP-R2` as a mainline repair because it improves some sparse early care geometry but damages later exact-window performance;
- the next repair should target `alive_on_art` directly through stock reconciliation instead of forcing quarterly suppression-share drift where support remains sparse.

---

### Validation

Purpose:

- validate forecast honesty, not only in-sample fit;
- prevent collapse into tail-only interpretation;
- force the model to survive both the strongest exact benchmark and the broader-history diagnostic benchmark.

Validation rules:

- blocked-time validation is mandatory;
- no holdout tuning;
- no use of `rule_based_extrapolated` or `latent_imputed` rows as headline holdout truth;
- pre-COVID and post-2020 windows must both be reported whenever the contract supports them.

Validation layers:

#### Primary validation

Use the exact-only quarterly blocked-time benchmark.

Contract:

- train on earlier exact rows only
- score holdout only on `exact_observed`
- use the strongest comparable recent quarterly regime

Reason:

- this remains the most honest high-trust benchmark in the current archive

Limitation:

- it is late-window dominated by data availability
- therefore it is necessary but not sufficient as the only interpretation layer

#### Secondary validation

Use the dense quarterly contract as a broader-history diagnostic benchmark.

Contract:

- train on `exact_observed + bridge_observed + rule_based_extrapolated`
- score holdout only on `exact_observed + bridge_observed`

Reason:

- this is the correct way to widen temporal support without pretending extrapolated rows are truth

Interpretation rule:

- dense-contract wins can support diagnosis of model-family behavior across pre-COVID years
- but they do not overrule the exact-only primary gate by themselves

#### Annual auxiliary validation

Use annual incidence and deaths anchors as supporting diagnostics.

Reason:

- these help identify infection-pressure and mortality blocks
- but they are not allowed to substitute for failed quarterly national transition forecasting

Retrospective validation style:

- rolling-origin blocked-time evaluation, not a single train/test split
- report at least:
  - overall primary-window performance
  - pre-COVID performance where the dense contract supports it
  - post-2020 performance
  - split-by-split tables for the years that dominate acceptance decisions

Current interpretation discipline:

- if a model improves annual incidence but still loses the quarterly blocked-time benchmark, it is a revert;
- if a model looks better only in the late tail and worse on broader-history diagnostics, it is not considered robust;
- if a model looks better only on dense training support but not on exact primary holdout, it is not a keep.

---

### Uncertainty

Purpose:

- quantify how much of the result is stable versus contract-sensitive;
- avoid reporting a single deterministic curve when the archive itself is mixed and partially observed.

Uncertainty sources:

1. parameter uncertainty
2. data-contract uncertainty
3. support / identifiability uncertainty
4. structural model uncertainty

For `TR-V3`, uncertainty should be summarized through:

- multi-start or constrained ensemble variation over admissible parameter blocks
- variation across blocked-time splits
- variation across data contracts:
  - exact-only
  - dense-train-observed-score
- provenance-aware reporting of which rows actually constrain each block

Important restriction:

- naive IID bootstrap is not the default uncertainty method here because it breaks the time structure

Preferred uncertainty procedures:

- blocked bootstrap or split-resampling over rolling-origin windows
- parameter perturbation around fitted blocks
- support-aware ablations
- ensemble spread across honest calibration runs

Required uncertainty outputs:

- holdout error distribution by split
- provenance summary by train/holdout window
- support summary by transition
- parameter or local-fit sensitivity summaries for the blocks that are actually estimated

Plain-English interpretation:

- uncertainty here is not only “how much the parameter might move”
- it is also “how much the answer changes when the evidence contract changes”

That is especially important for the current care and leakage blocks.

---

### Scenario Engine Boundary

Purpose:

- keep policy-scenario work from corrupting the forecast benchmark;
- preserve a clean separation between a forecaster and a future counterfactual simulator.

Core rule:

- the core forecaster must beat its blocked-time benchmark on its own terms before any richer policy scenario layer is allowed to stand in for it

The core forecaster is for:

- national blocked-time forecasting
- benchmark comparison
- transition/hazard identification
- observation honesty under mixed evidence

The future scenario engine is for:

- policy counterfactuals
- intervention package comparisons
- subgroup or KP-targeted intervention logic
- richer social/behavioral simulation

What belongs in the scenario engine, not the current forecaster:

- comprehensive policy bundles
- social-intervention counterfactuals
- explicit PrEP strategy packages
- rich network or microsimulation layers
- long-horizon intervention-return analyses to 2030 and beyond

Those are scientifically useful, but they must remain downstream of a stable benchmark forecaster.

This means:

- scenario design should not drive current keep/revert decisions
- scenario complexity is not allowed to rescue a weak forecaster
- richer policy simulation can be added later as a separate module once the core forecast contract is stable

---

### Missing-data ladder

Purpose:

- make mixed-evidence handling explicit rather than ad hoc;
- allow imputation without pretending imputed values are exact observations;
- keep blocked-time evaluation honest when sources have different temporal density and trustworthiness.

Data tiers:

1. `exact_observed`
   - direct source value
   - exact metric
   - exact period

2. `bridge_observed`
   - derived from adjacent monthly or quarterly releases
   - or from tightly neighboring source documents
   - with explicit provenance

3. `rule_based_extrapolated`
   - extrapolated from a stronger source or location using a transparent rule
   - source tier and rule must be recorded

4. `latent_imputed`
   - generated by the model under explicit uncertainty
   - never silently promoted to canonical ground truth

5. `rejected_or_quarantined`
   - fails scale, unit, or plausibility checks
   - excluded from calibration until resolved

Rules:

- benchmark targets should use `exact_observed` first;
- `bridge_observed` may be used to widen coverage, but must remain labeled as bridge evidence;
- `rule_based_extrapolated` values may support calibration priors and auxiliary diagnostics, but should not silently replace exact targets in headline benchmark tables;
- `latent_imputed` values may stabilize state inference, but they are model outputs, not input truth;
- every emitted artifact should report row counts by provenance tier.

What this means in plain English:

- imputation is allowed;
- silent imputation is not;
- mixed evidence is acceptable only if the tiers remain visible.

### Core forecaster vs future scenario engine

The literature review also supports a hard boundary between two different model roles.

#### Core forecaster

This is the `TR-V3` family described in this memo.

Its job is:

- national blocked-time forecasting
- explicit uncertainty
- mixed-frequency observation handling
- honest comparison against simple baselines

It should stay:

- low-dimensional enough to benchmark reliably
- identifiable enough to audit
- simple enough to keep or revert based on the live gate

#### Future scenario engine

This is a later, separate layer that may eventually use:

- agent-based network simulation
- social-structural microsimulation
- richer policy counterfactual logic
- subgroup targeting or KP-specific intervention analysis

It is informed by papers like:

- youth-focused TasP network models
- MicroCOSM

but it is **not** the same object as the core forecaster.

Contract:

- the scenario engine does not replace the core benchmark forecaster by default;
- the forecaster must win on blocked-time gates on its own terms;
- scenario models are allowed later for counterfactual policy analysis, not as a substitute for forecast honesty.

This split avoids a common failure mode:

- importing a rich social simulator too early,
- then losing identifiability, benchmark clarity, and keep-or-revert discipline.

---

## 8. Files To Edit

### New files

- `src/epigraph_ph/phase3/frontier/tr_v3.py`
  - core TR-V3 family
  - dynamic hazard baseline
  - observation model
  - shock layer
  - direct and hidden priors

- `src/epigraph_ph/phase3/frontier/tr_v3_observation.py`
  - optional split-out of learned observation layer if `tr_v3.py` becomes too large

- `src/epigraph_ph/phase3/frontier/tr_v3_shocks.py`
  - optional split-out of shock/state process logic

- `Phase3(dynamic)/src/phase3_dynamic/controls.py`
  - train-only hazard-side `A/C/R` control construction for `TR-V3-05a`

- `Phase3(dynamic)/src/phase3_dynamic/decompose.py`
  - optional helper utilities for train-only control decomposition

- `Phase3(dynamic)/src/phase3_dynamic/incidence.py`
  - incidence inflow construction and `N(t)` block helpers for `TR-V3-05b`

### Files to modify

- `src/epigraph_ph/phase3/frontier/registry.py`
  - register `TR-V3-00` through `TR-V3-04`

- `src/epigraph_ph/phase3/frontier/cli.py`
  - dispatch new TR-V3 runs

- `src/epigraph_ph/plugins/hiv.py`
  - add `phase3.frontier.tr_v3` settings:
    - dynamic-baseline priors
    - shock-process settings
    - observation-process priors
    - Phase 2 direct-prior map reuse or refinement

- `src/epigraph_ph/phase3/frontier/tr_v2.py`
  - reuse benchmark/report helpers where appropriate
  - do **not** overwrite `TR-V2` logic in place

- `Phase3(dynamic)/src/phase3_dynamic/data.py`
  - derive train-only `A/C/R` control inputs from national observation rows
  - expose annual incidence / prevalence anchors for the open incidence-flow model
  - preserve provenance tiers for `exact_observed`, `bridge_observed`, `rule_based_extrapolated`, and `latent_imputed`

- `Phase3(dynamic)/src/phase3_dynamic/model.py`
  - inject mechanism controls into hazards on top of the `TR-V3-04d` backbone
  - add explicit incidence inflow, leakage, and mortality blocks for `TR-V3-05b`
  - reject or quarantine impossible scale/unit combinations before calibration

- `Phase3(dynamic)/src/phase3_dynamic/loop.py`
  - add `TR-V3-05a`
  - add `TR-V3-05b`

- `Phase3(dynamic)/src/phase3_dynamic/cli.py`
  - expose the `TR-V3-05a` loop
  - expose the `TR-V3-05b` loop

- `Phase3(dynamic)/tests/test_data.py`
  - leakage checks for train-only control construction
  - incidence-anchor loading checks

- `Phase3(dynamic)/tests/test_model.py`
  - mechanism-control regression tests
  - nonnegative incidence-flow and leakage-flow tests

- `tests/test_phase3_transition_research_pytest.py`
  - parser coverage for new CLI names

- `tests/test_phase3_frontier_integration_pytest.py`
  - slow real-artifact regression for TR-V3 variants

- `tests/test_phase3_structural_frontier_pytest.py`
  - new fast synthetic regression tests for dynamic baseline / shock logic

---

## 9. Benchmark Gates

### Primary benchmark

Blocked-time rolling-origin benchmark:

- train on earlier years only
- forecast next holdout year
- use `2017-2025` regime

Primary acceptance numbers are relative to current live benchmark:

- current `no_priors` mean MAE: `1.048634`
- current worst-split `2021 -> 2022` MAE: `3.796171`

Source:

- [rolling-origin report](/D:/EpiGraph_PH/artifacts/runs/tr-v2-rolling-origin-cli-20260409-s00/analysis/tr_v2_rolling_origin_report.md#L7)

### Gate for `TR-V3-00`

Must satisfy:

- mean blocked-time MAE `< 1.048634`
- no split worse than `4.0`
- median MAE not worse than current `no_priors`

### Gate for `TR-V3-01`

Must satisfy:

- non-inferior to `TR-V3-00` on mean MAE
- improved readout consistency diagnostics
- no degradation in observation honesty

### Gate for `TR-V3-02`

Must satisfy:

- `2021 -> 2022` MAE materially improved relative to `3.796171`
- mean MAE not worse than `TR-V3-01`
- shock activations sparse and interpretable

### Gate for `TR-V3-03`

Must satisfy:

- beats `TR-V3-02` or clearly improves at least `3/4` late blocked-time splits
- Phase 2 direct priors remain explicitly auditable

### Gate for `TR-V3-04`

Must satisfy:

- hidden-driver channels improve blocked-time holdout beyond direct-only
- hidden-only branch remains a losing control or is clearly justified otherwise

If `TR-V3-04` does not beat `TR-V3-03`, hidden channels are not kept as part of the main family.

### Gate for `TR-V3-05a`

Must satisfy:

- mean MAE `< 0.195570`
- `2023` MAE `<= 0.169219`
- `2024-2025` tail mean MAE `<= 0.216455`
- `2022` MAE `<= 0.207171`
- no generic seasonal term unless a separate ablation proves it is necessary

If `TR-V3-05a` wins only by adding observation/reporting periodicity, that periodic component must be disclosed as `R(t)` rather than described as biological HIV seasonality.

### Gate for `TR-V3-05b`

Must satisfy:

- mean MAE `< 0.195570`
- `2023` MAE `<= 0.169219`
- `2024-2025` tail mean MAE `<= 0.216455`
- `2022` MAE `<= 0.207171`
- annualized incidence implied by `\iota(t)` is non-inferior to a quarterized incidence-anchor baseline
- all state trajectories remain nonnegative
- no hazard or mortality block sits on its upper bound persistently without strong evidence

If `TR-V3-05b` wins only by shifting mass between infection, diagnosis, and reporting blocks without stable identifiability diagnostics, it must be rejected.

---

## 10. Contract Rules

These are mandatory and inherited from the stricter paper contract.

### Must remain true

- blocked-time validation/holdout is used
- provincial hierarchy is auxiliary only
- Phase 2 insertion is weaker than “structured mechanistic discovery”
- the third 95 is not a primary validated target
- evidence sources are mixed and disclosed

### Must not happen

- no reversion to broad `rescue_core` as the main research path
- no tuning on holdout years
- no province/region evaluation presented as equal to national validation
- no silent heuristic readout shortcuts

---

## 11. Suggested Implementation Order

### Slice A: Scaffolding

1. add `TR-V3-*` registry entries
2. add CLI names
3. create `tr_v3.py`
4. factor out benchmark helpers from `tr_v2.py` if needed

### Slice B: Dynamic baseline

1. implement latent hazard local-linear-trend model
2. reuse existing holdout simulation shell
3. add `TR-V3-00`
4. benchmark against current `no_priors`

### Slice C: Observation model

1. replace fixed testing-share readout
2. add dynamic observation equations
3. add `TR-V3-01`
4. compare to `TR-V3-00`

### Slice D: Shock layer

1. implement sparse or AR shock component
2. add `TR-V3-02`
3. evaluate specifically on `2021 -> 2022`

### Slice E: Phase 2 direct priors

1. reuse explicit target-based admissibility map
2. attach direct priors to dynamic hazards
3. add `TR-V3-03`

### Slice F: Hidden channels

1. add hidden-driver channels only after direct priors are stable
2. add `TR-V3-04`
3. keep only if blocked-time gain is real

### Slice G: Corrected mechanism controls

1. freeze `TR-V3-04d` as the backbone
2. derive train-only `A/C/R` controls from national observation rows with explicit provenance tiers
3. inject `A/C/R` controls into hazards, not just the readout layer
4. add `TR-V3-05a`
5. benchmark against the explicit `TR-V3-04d` gate above
6. do not force `N(t)` into the hazard equation here; reserve it for the incidence block unless a later ablation proves otherwise

### Slice H: Open leaky incidence-flow model

1. keep `TR-V3-04d` plus the `05a` control layer as the hazard backbone
2. add explicit incidence inflow `\iota(t)` into `U`, with `N(t)` as a first-class infection-pressure driver
3. add leakage transitions and constrained mortality
4. calibrate with an explicit identifiability ledger
5. add `TR-V3-05b`
6. defer explicit `S(t)` until stronger denominator data are available

### Slice I: Boundary discipline

1. keep the blocked-time forecaster and any future scenario engine separate
2. allow bridge and extrapolated data to widen calibration support, but keep provenance explicit
3. do not allow scenario-simulation complexity to bypass the forecaster benchmark gates

---

## 12. Why This Is Still Worth Doing

There is already substantial HIV modeling literature.

So the value of this repo is **not**:

- “nobody has modeled HIV transitions before”

The value would be:

- a Philippines-specific, open, mixed-evidence, blocked-time-validated national forecasting benchmark;
- transparent negative and positive results under a stricter contract;
- and a model family that can beat simple baselines honestly, not just look mechanistic on paper.

Secondarily, if the core forecaster becomes strong enough, the repo may later support a separate scenario engine for policy counterfactuals. That is a later layer, not the primary validation object.

If `TR-V3` succeeds, the contribution is practical and scientific.

If it fails, the contribution is still useful:

- it shows that stricter contract honesty changes the conclusions,
- and it documents which mechanistic ingredients are insufficient.

---

## 13. Final Judgment

The next move should be:

```text
improve the national mechanistic time-series backbone first,
then let Phase 2 constrain it.
```

That means:

- `TR-V3-00`: stronger dynamic baseline
- `TR-V3-01`: learned observation model
- `TR-V3-02`: shock layer
- `TR-V3-03`: direct Phase 2 priors
- `TR-V3-04`: hidden-driver channels only if they win
- `TR-V3-05a`: hazard-side `A/C/R` mechanism controls on top of `TR-V3-04d`, with no default seasonality
- `TR-V3-05b`: open leaky incidence-flow model with explicit `N(t)`-driven infection inflow, leakage, mortality, identifiability ledger, and provenance-aware missing-data handling

Once stronger denominator data exist, the next extension should be:

- add the explicit `S(t)` susceptible compartment
- upgrade the open infected-population system into a fuller susceptible-infection-flow system
- do this only after the incidence and denominator blocks become identifiable enough to support it

The rich network and social-structural literature should inform a later scenario layer, but it should not displace the core blocked-time forecaster unless it can clear the same benchmark gates.

This is the scientifically correct next autoresearch program under the new blocked-time contract.
