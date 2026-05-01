Date: 2026-04-09

# Phase 3 Mathematical Specification And Autoresearch Program

## Purpose

This note turns the recent literature scan into a concrete Phase 3 mathematical specification.

It has four jobs:

1. define the exact mathematical objects for the next Phase 3 frontier;
2. rank the candidate model families rather than treating them as equally good;
3. define the correct optimization order:
   - fit the mechanistic hazard model first;
   - only then optimize toward `95-95-95`;
4. define an unattended autoresearch loop that can run continuously without manual intervention while still remaining scientifically defensible.

This note is deliberately strict. "Continuous optimization until the best possible model" is not a mathematically well-posed instruction. A valid autoresearch loop needs:

- a frozen data snapshot,
- a frozen evaluation family,
- a constrained mutation surface,
- a keep-or-revert rule,
- a promotion gate,
- and explicit stop or plateau conditions.
- a line graph to show the pporgress of the loop.

Without those rules, the loop will eventually overfit the holdout and reward-hack the weakly observed back half of the cascade.

## Executive Judgment

There are two different things that must not be conflated:

1. the target mathematical architecture for Phase 3;
2. the staged implementation order for unattended autoresearch.

The correct target architecture after the literature review is:

1. explicit incidence module with observed denominator `N_t` and latent incidence hazard `lambda_t`;
2. diagnosis module using either:
   - `U -> D` hazard dynamics, or
   - a diagnosis-delay / back-calculation convolution;
3. downstream care cascade using multistate semi-Markov hazards;
4. Phase 2 direct temporal surfaces entering as structured priors on transition intensities;
5. hidden modes entering as shared latent shock channels;
6. explicit observation / ascertainment module for diagnosis, ART, VL testing, and documented suppression;
7. only after calibration passes, a target-seeking loop toward `95-95-95`.

The correct implementation order for autoresearch is still staged:

1. lock the current winning frontier branch as the regression baseline;
2. fit the mechanistic transition and observation stack on observable behavior;
3. make incidence explicit in the state equations and front-half likelihood;
4. then open the full integrated incidence-diagnosis-care architecture;
5. only then run target-seeking toward `95-95-95`.

The correct first-class object is not "population -> undiagnosed".

The correct core object is:

```text
state_t = (U_t, D_t, A_t, V_t, L_t)
```

with transition hazards:

```text
U_to_D, D_to_A, A_to_V, A_to_L, L_to_A
```

with explicit incidence inflow:

```text
I_t = N_t * lambda_t
```

where `N_t` is an observed denominator and `lambda_t` is an incidence hazard.

The current repo already supports the first half of this program:

- `transition_engine.py` implements the mechanistic `U/D/A/V/L` frontier;
- `tr_v2.py` already fits Phase 2 direct and hidden adjustments on top of a locked `AGE-01B` baseline;
- `phase2_structural_inputs.py` already quarterizes the Phase 2 direct and hidden structural payload for the frontier.

What is still missing in code is the full integrated explicit-incidence architecture. What is still missing in design was a single explicit mathematical spec that says which family is the end-state target, which families are staging families, and how the unattended autoresearch supervisor is allowed to search.

## Repo Alignment

This spec is aligned to:

- [phase3_architecture_and_frontier_2026_04_05.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_architecture_and_frontier_2026_04_05.md)
- [phase3_tr_v2_implementation_plan_2026_04_06.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v2_implementation_plan_2026_04_06.md)
- [AUTORESEARCH.md](/D:/EpiGraph_PH/AUTORESEARCH.md)
- [AUTORESEARCH_TRANSITIONS.md](/D:/EpiGraph_PH/AUTORESEARCH_TRANSITIONS.md)
- [registry.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/registry.py)
- [tr_v2.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/tr_v2.py)
- [phase2_structural_inputs.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/phase2_structural_inputs.py)

The selected autoresearch variant remains:

```text
evidence-to-model-loop
```

not a pure benchmark-chasing loop.

That is still correct because the third `95` remains partly observation-process limited. The optimizer must not be allowed to learn that service documentation is the same thing as latent virologic truth.

## Notation

Time:

- `t = 1, ..., T` indexes quarters.

Core states:

- `U_t`: undiagnosed PLHIV
- `D_t`: diagnosed PLHIV not currently on ART
- `A_t`: alive on ART and not yet counted in suppressed state
- `V_t`: virally suppressed on ART
- `L_t`: previously on ART but lost / interrupted

Optional incidence terms:

- `N_t`: observed denominator / exposure pool
- `lambda_t`: incidence hazard
- `I_t = N_t * lambda_t`: incidence inflow to `U`

Observed series:

- `y_diag_t`: diagnosed PLHIV
- `y_art_t`: alive on ART
- `y_newdiag_t`: new diagnosed cases in period
- `y_vltest_t`: tested for viral load
- `y_vs_t`: documented virally suppressed

Phase 2 covariates:

- `z_b(t)`: quarterized direct temporal block value for block `b`
- `u_m(t)`: quarterized hidden mode score for latent mode `m`
- `X_hid(t)`: hidden-mode design row built from the retained `u_m(t)` coordinates at quarter `t`

Transition hazards:

- `h_r(t)` for `r in {U_to_D, D_to_A, A_to_V, A_to_L, L_to_A}`

Baseline frontier hazard:

- `hbar_r(t)`: locked inherited hazard from the current kept champion
- `eta_anchor_r(t)`: the corresponding locked linear predictor, using the same link as the active hazard family

Linear predictors:

- `eta_r(t)`: hazard linear predictor for transition `r`

## Cascade Definitions

The latent cascade targets are:

```text
C1_t = (D_t + A_t + V_t + L_t) / max(U_t + D_t + A_t + V_t + L_t, eps)
C2_t = (A_t + V_t) / max(D_t + A_t + V_t + L_t, eps)
C3_t = V_t / max(A_t + V_t, eps)
```

These are the latent biological or program-mechanistic quantities.

They are not the same as the documented program observations:

```text
y_vltest_t
y_vs_t
```

The target-seeking loop should optimize the latent cascade, while the calibration loop should score both the latent fit and the observation process.

## Selected Literature-Backed Target Architecture

This is the architecture selected from the literature review.

It is the intended Phase 3 end-state model, even if the implementation reaches it in stages.

### Module A: Incidence

Observed denominator:

```text
N_t
```

Latent incidence hazard:

```text
lambda_t > 0
```

Baseline log-hazard form:

```text
log(lambda_t) =
    alpha_I
  + beta_I^T * w_t
  + sum_{(b,l) in S_I} Gamma_{I,b,l} * z_b(t-l)
  + sum_{m=1}^{M_I} Psi_m * u_m(t)
  + xi_t
```

Incidence inflow:

```text
I_t = N_t * lambda_t
```

Optional renewal form:

```text
I_t = N_t * exp(mu_t + sum_{k=1}^K omega_k * I_{t-k} / max(N_{t-k}, eps))
```

where `mu_t` may itself be modeled by direct Phase 2 surfaces and hidden shocks.

### Module B: Diagnosis

There are two admissible front-half parameterizations.

#### B1. Explicit `U -> D` hazard

```text
d_t = h_U_to_D(t) * U_t
```

with:

```text
h_U_to_D(t) = 1 - exp(-exp(eta_U_to_D(t)))
```

and:

```text
eta_U_to_D(t) =
    alpha_D
  + q_D(t)
  + sum_{(b,l) in S_D} Gamma_{D,b,l} * z_b(t-l)
  + sum_{m=1}^{M_D} Lambda_{D,m} * u_m(t)
```

#### B2. Diagnosis-delay / back-calculation convolution

```text
E[y_newdiag_t] = sum_{k=0}^K I_{t-k} * pi_k(t)
```

where the diagnosis-delay kernel is:

```text
pi_k(t) = g_k(t) * product_{j=0}^{k-1} (1 - g_j(t-k+j))
```

with `g_k(t)` the quarter-`k` conditional diagnosis probability.

Rule:

- the autoresearch loop may compare `B1` and `B2`;
- it may not mix them carelessly inside one candidate without a declared combined likelihood.

### Module C: Downstream Care Cascade

The downstream care system is semi-Markov.

For each downstream transition `r in {D_to_A, A_to_V, A_to_L, L_to_A}`:

```text
h_r(t, a) = 1 - exp(-exp(eta_r(t, a)))
```

with:

```text
eta_r(t, a) =
    alpha_r
  + f_r(a)
  + q_r(t)
  + sum_{(b,l) in S_r} Gamma_{r,b,l} * z_b(t-l)
  + sum_{m=1}^{M_r} Lambda_{r,m} * u_m(t)
```

where:

- `a` is time since entry into the source state;
- `f_r(a)` is a dwell-time effect;
- `q_r(t)` is an optional era or disruption intercept.

### Module D: Shared Hidden Structure

The hidden modes evolve dynamically:

```text
u_t = A * u_{t-1} + epsilon_t
epsilon_t ~ N(0, Sigma_u)
```

These hidden modes may affect:

- incidence;
- diagnosis;
- downstream care transitions;
- observation / ascertainment rates.

They must not be merged with the direct sparse Phase 2 surface.

### Module E: State Evolution

With explicit incidence and diagnosis hazard:

```text
U_{t+1} = U_t + I_t - d_t
D_{t+1} = D_t + d_t - a_t
A_{t+1} = A_t + a_t + r_t - v_t - l_t
V_{t+1} = V_t + v_t
L_{t+1} = L_t + l_t - r_t
```

where:

```text
a_t = h_D_to_A(t, a_D) * D_t
v_t = h_A_to_V(t, a_A) * A_t
l_t = h_A_to_L(t, a_A) * A_t
r_t = h_L_to_A(t, a_L) * L_t
```

If the diagnosis-delay convolution is active instead of explicit `U -> D` hazard, the latent undiagnosed accounting must still obey:

```text
U_{t+1} = U_t + I_t - d_t
```

but `d_t` is implied through the delay kernel rather than direct hazard multiplication.

### Module F: Observation / Ascertainment

Diagnosis and care observations:

```text
E[y_diag_t] = D_t + A_t + V_t + L_t
E[y_art_t] = A_t + V_t
E[y_newdiag_t] = d_t
```

VL testing and documented suppression:

```text
E[y_vltest_t] = (A_t + V_t) * pi_test_t
E[y_vs_t] = V_t * pi_test_t * pi_doc_t
```

with:

```text
logit(pi_test_t) = alpha_test + theta_test^T * s_t + sum_m Kappa_test,m * u_m(t)
logit(pi_doc_t) = alpha_doc + theta_doc^T * s_t + sum_m Kappa_doc,m * u_m(t)
```

This is the minimum acceptable observation layer.

### Module G: Where Phase 2 Enters

Direct temporal surface:

- enters as structured priors or covariates on:
  - `lambda_t`,
  - diagnosis hazard or delay kernel,
  - downstream care hazards.

Hidden mode scores:

- enter as shared latent shocks across the same modules.

Constraint:

- incidence, diagnosis, and downstream care must use separate coefficients even when the same Phase 2 factor appears in all three places.

## Candidate Model Families

The candidate families are not equal. The ranking matters.

### Family F0: Locked Anchored Frontier Baseline

Role:

- regression baseline only
- not the final scientific target

Definition:

```text
eta_r(t) = logit(hbar_r(t))
h_r(t) = hbar_r(t)
```

with state updates inherited from the kept champion.

This is the current `TR-V2-00` lock role.

### Family F1: Anchored Mechanistic Hazard Model With Phase 2 Direct And Hidden Adjustments

Role:

- recommended first structural overlay inside the Phase 3 frontier
- immediate constrained structural family for unattended autoresearch

This family is the mathematically correct refinement of current `TR-V2`.

Exact form:

```text
eta_r(t) = logit(hbar_r(t)) + Delta_dir_r(t) + Delta_hid_r(t)
h_r(t) = inv_logit(eta_r(t))
```

where:

```text
Delta_dir_r(t) = sum_{(b,l) in S_r} Gamma_{r,b,l} * z_b(t-l)
Delta_hid_r(t) = sum_{m=1}^{M_r} Lambda_{r,m} * u_m(t)
```

with `S_r` the support-filtered direct edge set for transition `r`.

This is the closest family to the current code in [tr_v2.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/tr_v2.py).

Current fitting objective for the direct block is a support-weighted Gaussian MAP regression on the locked hazard logits:

```text
min_{alpha_r, Gamma_r}
sum_t (logit(hbar_r(t)) - alpha_r - X_dir_r(t) * Gamma_r)^2
+ sum_j Gamma_{r,j}^2 / tau_{r,j}^2
```

with:

```text
tau_{r,j} = prior_scale_{r,j} * (1 + support_count_{r,j}) * max(stability_{r,j}, 0.05)
```

The hidden block is then fit on the residualized logit hazard:

```text
min_{alpha_r, Lambda_r}
sum_t (logit(hbar_r(t)) - Delta_dir_r(t) - alpha_r - X_hid(t) * Lambda_r)^2
+ ||Lambda_r||_2^2
```

This is scientifically acceptable as the first structural overlay because:

- it preserves the kept frontier as anchor,
- it constrains search to supported Phase 2 surfaces,
- it separates direct sparse structure from hidden low-rank structure,
- and it does not reopen the full broad Phase 3 space.

### Family F2: Joint Discrete-Time Mechanistic Hazard Model

Role:

- recommended first family to fit directly against observations
- should become the Stage 1 calibration target

This family is the mathematically correct generalization of `F1` once the frontier is allowed to fit the hazards directly rather than only residualizing the locked champion.

Use the complementary log-log link so discrete-time hazards remain consistent with an underlying continuous-time intensity:

```text
h_r(t) = 1 - exp(-exp(eta_r(t)))
```

with:

```text
eta_r(t) =
    alpha_r
  + q_r(t)
  + rho_r * eta_r(t-1)
  + sum_{(b,l) in S_r} Gamma_{r,b,l} * z_b(t-l)
  + sum_{m=1}^{M_r} Lambda_{r,m} * u_m(t)
```

where:

- `q_r(t)` is an optional piecewise constant quarter-era or shock-era intercept;
- `rho_r` is a persistence term;
- `Gamma` and `Lambda` remain transition-specific.

Quarterly flow definitions:

```text
d_t = h_U_to_D(t) * U_t
a_t = h_D_to_A(t) * D_t
v_t = h_A_to_V(t) * A_t
l_t = h_A_to_L(t) * A_t
r_t = h_L_to_A(t) * L_t
```

The clean simultaneous state update is:

```text
U_{t+1} = U_t + I_t - d_t
D_{t+1} = D_t + d_t - a_t
A_{t+1} = A_t + a_t + r_t - v_t - l_t
V_{t+1} = V_t + v_t
L_{t+1} = L_t + l_t - r_t
```

The current code uses a stricter ordered-within-quarter clipping scheme to keep all flows feasible:

```text
u_to_d <= U_t
d_to_a <= D_t + u_to_d
a_to_v <= A_t + d_to_a
a_to_l <= A_t + d_to_a - a_to_v
l_to_a <= L_t + a_to_l
```

Both are valid as long as the implementation fixes one convention and scores it consistently.

Recommendation:

- keep the current ordered clipping in code for numerical safety;
- write all model comparison artifacts against the same convention.

### Family F3: Back-Calculation / Diagnosis-Delay Front Half + Mechanistic Downstream Care

Role:

- strongest literature-grounded front-half alternative
- useful if direct `U_to_D` hazard fitting remains weakly identified

Incidence:

```text
I_t = N_t * lambda_t
log(lambda_t) = alpha_I + beta_I^T * w_t + xi_t
```

Diagnosis delay kernel:

```text
E[y_newdiag_t] = sum_{k=0}^K I_{t-k} * pi_k(t)
```

where `pi_k(t)` is the probability that an infection occurring `k` quarters earlier is diagnosed in quarter `t`.

A hazard-based parameterization of the delay kernel is:

```text
pi_k(t) = g_k(t) * product_{j=0}^{k-1} (1 - g_j(t-k+j))
```

where `g_k` is the discrete probability of diagnosis at lag `k` conditional on still being undiagnosed.

The downstream care system remains mechanistic:

```text
D_to_A, A_to_V, A_to_L, L_to_A
```

using either `F1` or `F2`.

This family is attractive because it matches HIV back-calculation literature, but it should not be the first unattended frontier because it adds another large source of front-half non-identifiability.

### Family F4: Incidence-Augmented State Model

Role:

- correct full extension after the transition family is stable
- not the first frontier branch

Definition:

```text
I_t = N_t * lambda_t
log(lambda_t) =
    alpha_I
  + beta_I^T * w_t
  + sum_{m=1}^{M_I} Psi_m * u_m(t)
  + xi_t
```

with the transition model:

```text
h_r(t) = 1 - exp(-exp(eta_r(t)))
```

or, if the anchored family is retained,

```text
eta_r(t) = cloglog(hbar_r(t)) + Delta_dir_r(t) + Delta_hid_r(t)
```

The full state update becomes:

```text
U_{t+1} = U_t + I_t - d_t
D_{t+1} = D_t + d_t - a_t
A_{t+1} = A_t + a_t + r_t - v_t - l_t
V_{t+1} = V_t + v_t
L_{t+1} = L_t + l_t - r_t
```

The same Phase 2 surfaces may affect both incidence and diagnosis, but they must use different coefficients:

```text
beta_I != beta_U_to_D
Gamma_I != Gamma_U_to_D
```

This separation is non-negotiable.

### Family F5: Semi-Markov Downstream Cascade

Role:

- best mathematical candidate if downstream dwell time matters materially
- second-wave family after `F2`

The downstream transitions depend on elapsed dwell time in state `s`:

```text
h_r(t, a) = 1 - exp(-exp(eta_r(t, a)))
```

with:

```text
eta_r(t, a) =
    alpha_r
  + f_r(a)
  + sum_{(b,l) in S_r} Gamma_{r,b,l} * z_b(t-l)
  + sum_m Lambda_{r,m} * u_m(t)
```

where `a` is time-since-entry and `f_r(a)` is a spline, histogram, or piecewise constant dwell-time effect.

This family is particularly relevant for:

- `D_to_A`,
- `A_to_L`,
- `L_to_A`.

It is a strong scientific family, but it expands the state representation and should not be the first unattended frontier.

### Family F6: Sojourn-Time And Jump-Probability Parameterization

Role:

- interpretable alternative if raw hazard estimation becomes unstable

For each state `s`, define expected dwell time `mu_s(t)` and next-jump probabilities `p_{s->j}(t)`.

```text
log(mu_s(t)) = kappa_s + c_s^T * x_t
p_{s->j}(t) = exp(omega_{s->j}(t)) / sum_{j' in out(s)} exp(omega_{s->j'}(t))
```

Then implied transition intensities are:

```text
q_{s->j}(t) = p_{s->j}(t) / max(mu_s(t), eps)
```

and quarterly hazards may be approximated by:

```text
h_{s->j}(t) = 1 - exp(-q_{s->j}(t))
```

This family is useful if you want a more interpretable downstream care model, but it is not the first Phase 3 frontier family.

### Family F7: Latent Self-Exciting Or Renewal Incidence / Reporting Model

Role:

- optional research branch only
- not part of the primary frontier

Example form:

```text
lambda_t = mu_t + sum_{k=1}^K phi_k * y_newdiag_{t-k}
log(mu_t) = alpha + beta^T * w_t + xi_t
```

This may help capture reporting bursts or backlog release, but it is too easy to confound with service disruptions. It should remain a side branch, not the main Phase 3 program.

### Family F8: Integrated Explicit-Incidence Literature Architecture

Role:

- selected target architecture for mature Phase 3
- integrates the literature-backed modules into one state-space system

This family combines:

- explicit incidence with observed denominator `N_t`;
- either diagnosis hazard or diagnosis-delay convolution;
- semi-Markov downstream care hazards;
- Phase 2 direct surfaces as structured priors;
- hidden modes as shared latent shocks;
- explicit observation / ascertainment for diagnosis, ART, VL testing, and documented suppression.

Canonical form:

```text
I_t = N_t * lambda_t
log(lambda_t) =
    alpha_I
  + beta_I^T * w_t
  + sum_{(b,l) in S_I} Gamma_{I,b,l} * z_b(t-l)
  + sum_m Psi_m * u_m(t)
  + xi_t
```

Front half:

```text
d_t = h_U_to_D(t) * U_t
```

or:

```text
E[y_newdiag_t] = sum_{k=0}^K I_{t-k} * pi_k(t)
```

Downstream care:

```text
h_r(t, a) = 1 - exp(-exp(eta_r(t, a)))
```

State update:

```text
U_{t+1} = U_t + I_t - d_t
D_{t+1} = D_t + d_t - a_t
A_{t+1} = A_t + a_t + r_t - v_t - l_t
V_{t+1} = V_t + v_t
L_{t+1} = L_t + l_t - r_t
```

This is the mathematically strongest family in the document.

It is not the first implementation slice only because it is also the hardest family to identify and stabilize in one jump.

## Recommended Family Order

There are again two orderings:

1. target-architecture priority;
2. implementation-staging priority.

### Target-Architecture Priority

The mathematically strongest target family is:

1. `F8` integrated explicit-incidence literature architecture
2. `F5` semi-Markov downstream cascade
3. `F4` incidence-augmented state model
4. `F3` delay-kernel front half
5. `F2` joint mechanistic hazard model
6. `F1` anchored structural overlay
7. `F0` baseline lock
8. `F6` and `F7` as side branches only

### Implementation-Staging Priority

The correct staged autoresearch order is:

1. `F0` baseline lock
2. `F2` joint mechanistic hazard model
3. `F1` anchored Phase 2 direct and hidden overlays as constrained structural regularization
4. `F4` explicit incidence augmentation
5. `F3` diagnosis-delay / back-calculation front half if the direct diagnosis hazard remains weakly identified
6. `F5` semi-Markov downstream variant
7. `F8` full integrated architecture
8. `F6` and `F7` only as side branches

The practical interpretation is:

- `F8` is the intended destination;
- `F2 -> F4 -> F3/F5 -> F8` is the safer path for unattended autoresearch;
- `F1` remains the constrained structural overlay that ties the active frontier to Phase 2 evidence while the larger architecture is being stabilized.

## Observation Model

The observation model must distinguish state truth from service documentation.

### Front-Half Observation Equations

```text
y_diag_t = D_t + A_t + V_t + L_t + e_diag_t
y_art_t = A_t + V_t + e_art_t
y_newdiag_t = d_t + e_newdiag_t
```

where `e_*` may be Gaussian, Laplace, or count-family residuals depending on the chosen fitting engine.

### Viral Load And Suppression Observation Process

Let:

```text
n_art_t = A_t + V_t
pi_test_t = P(VL tested | on ART at t)
pi_doc_t = P(documented suppression | tested and suppressed at t)
```

Then:

```text
E[y_vltest_t] = n_art_t * pi_test_t
E[y_vs_t] = V_t * pi_test_t * pi_doc_t
```

with possible link functions:

```text
logit(pi_test_t) = alpha_test + theta_test^T * s_t
logit(pi_doc_t) = alpha_doc + theta_doc^T * s_t
```

This separation is required. The current repo already signals that treating documented suppression as clean latent truth is unsafe.

## Calibration Objective

Stage 1 calibration should minimize:

```text
L_cal =
    w1 * MAE_norm(y_newdiag, yhat_newdiag)
  + w2 * MAE_norm(y_diag, yhat_diag)
  + w3 * MAE_norm(y_art, yhat_art)
  + w4 * MAE_norm(y_vltest, yhat_vltest)
  + w5 * MAE_norm(y_vs, yhat_vs)
  + lambda_anchor * P_anchor
  + lambda_smooth * P_smooth
  + lambda_sparse * P_sparse
  + lambda_implaus * P_implaus
```

with normalized MAE:

```text
MAE_norm(y, yhat) = mean_t |y_t - yhat_t| / max(scale_y, eps)
```

Recommended weighting:

- primary weights on `y_newdiag`, `y_diag`, `y_art`;
- weak secondary weights on `y_vltest`, `y_vs`;
- no candidate may win by improving the third `95` while degrading the front half.

Penalty definitions:

```text
P_anchor = sum_{r,t} (eta_r(t) - eta_anchor_r(t))^2 / max(sigmar^2, eps)
P_smooth = sum_{r,t} (eta_r(t) - eta_r(t-1))^2
P_sparse = sum_r (||Gamma_r||_1 + ||Lambda_r||_2^2)
P_implaus = positivity_violations + mass_balance_violations + hazard_bound_violations
```

If a family does not use an anchor, set `lambda_anchor = 0` and rely on rolling backtest rejection instead.

## Target-Seeking Objective Toward 95-95-95

Stage 2 should not refit the whole model from scratch. It should optimize bounded perturbations or policy levers on top of a kept Stage 1 calibrated champion.

Let `a` denote the policy or intervention parameter vector. Then:

```text
eta_r_star(t; a) = eta_r(t) + B_r(t) * a
```

or, for surface-scaling control:

```text
eta_r_star(t; a) =
    eta_r(t)
  + sum_{(b,l) in S_r} a_{r,b,l} * Gamma_{r,b,l} * z_b(t-l)
```

with hard bounds:

```text
a_{r,b,l} in [amin_{r,b,l}, amax_{r,b,l}]
```

Define cascade shortfall at the target horizon `H`:

```text
Shortfall(a) =
    wc1 * max(0, 0.95 - C1_H(a))^2
  + wc2 * max(0, 0.95 - C2_H(a))^2
  + wc3 * max(0, 0.95 - C3_H(a))^2
```

Then optimize:

```text
J_target(a) =
    E[Shortfall(a)]
  + lambda_dev * sum_{r,t} (eta_r_star(t; a) - eta_r(t))^2
  + lambda_cost * Cost(a)
  + lambda_robust * Var(Shortfall(a))
```

The expectation and variance should be taken over:

- posterior draws if a Bayesian engine is used,
- otherwise bootstrap or champion-ensemble draws across accepted calibrated models.

This is the correct place to optimize toward `95-95-95`.

It is not the correct place to calibrate the epidemic model.

## Autoresearch Program

### Stage Separation

The unattended supervisor must operate in two distinct stages.

### Stage 1: Mechanistic Calibration Loop

Goal:

- minimize `L_cal`
- produce the best calibrated champion under a frozen data snapshot

Allowed mutations:

- candidate family selection within the approved family set;
- link choice: `cloglog` vs `logit` only where explicitly allowed;
- lag depth by transition;
- support thresholds for Phase 2 direct edges;
- hidden rank cap;
- persistence terms `rho_r`;
- observation model class for VL testing and documented suppression;
- penalty weights within pre-declared numeric ranges;
- sparse vs dense prior family if explicitly registered.

Forbidden mutations:

- changing the holdout split;
- changing metric scales after the run starts;
- changing the keep gate;
- editing benchmark artifacts;
- introducing unsupported transition-factor assignments;
- directly optimizing the Stage 1 score for `95-95-95`.

Stage 1 keep gate:

```text
accept candidate k over champion c only if:

L_cal(k) <= L_cal(c) - delta_cal
and MAE_front(k) <= MAE_front(c)
and DiagFlowMAE(k) <= DiagFlowMAE(c)
and no hard plausibility gate fails
```

where:

- `MAE_front` is the aggregated front-half score on `y_newdiag`, `y_diag`, `y_art`;
- `DiagFlowMAE` is a mandatory guard;
- `delta_cal` is a pre-declared minimum improvement margin.

### Stage 2: Target-Seeking Loop

Goal:

- search bounded interventions or bounded Phase 2-surface perturbations on top of the accepted Stage 1 champion

Allowed mutations:

- bounded lever vector `a`;
- horizon `H` chosen from a pre-declared set;
- robust optimization settings;
- uncertainty-ensemble size.

Forbidden mutations:

- changing the calibrated structural family during Stage 2;
- refitting observation equations to chase `95-95-95`;
- promoting a target-seeking candidate that materially degrades Stage 1 calibration.

Stage 2 keep gate:

```text
accept target policy a_k only if:

J_target(a_k) <= J_target(a_c) - delta_target
and L_cal_under_policy(a_k) <= L_cal(champion) + eps_cal
and no plausibility or cost gate fails
```

This means the target loop is subordinate to the calibration loop.

## Continuous Operation

The correct unattended continuous loop is not:

```text
for ever:
    mutate everything
    keep whatever improves the visible score
```

The correct unattended continuous loop is:

```text
while true:
    if new data snapshot exists:
        freeze snapshot S_k
        freeze evaluation contract E_k
        reset search budget

    run Stage 1 search on S_k until:
        - plateau budget exhausted, or
        - no promotion for N_stage1 iterations, or
        - hard failure rate exceeds threshold

    freeze champion C_k

    run Stage 2 search on top of C_k until:
        - plateau budget exhausted, or
        - target shortfall no longer improves, or
        - policy deviation limit reached

    archive:
        - champion model
        - challenger table
        - failed mutation reasons
        - calibration frontier
        - target-seeking frontier

    sleep until:
        - new data arrive, or
        - scheduled replay on the same snapshot is due
```

That is a valid unattended loop.

"Best possible model" must mean:

```text
best accepted champion under the frozen snapshot and frozen evaluation contract
```

not a metaphysical global optimum.

## Promotion And Overfitting Control

Continuous optimization on one visible holdout will eventually overfit.

Therefore the supervisor must use:

1. blocked rolling training windows;
2. a promotion validation set used for frequent challenger comparison;
3. a shadow audit set evaluated only on promotion candidates;
4. periodic epoch reset when a new data snapshot arrives.

Recommended evaluation layers:

- inner train: used for fitting;
- validation: used for mutation acceptance;
- audit: used only for promotion confirmation;
- historical champion table: used for non-regression.

No candidate is allowed to see the audit result unless it survives the validation gate.

## Mutation Surface

The mutation unit should be configuration-first, not arbitrary code-first.

Allowed mutation bundles:

- `family_id`
- `transition_link`
- `lag_set_by_transition`
- `phase2_support_threshold`
- `phase2_prior_scale_family`
- `hidden_rank_cap`
- `persistence_family`
- `observation_process_family`
- `penalty_weight_bundle`
- `policy_lever_bounds`

Only after the config surface is exhausted should the loop mutate implementation details.

Reason:

- arbitrary code mutation before the evaluation harness is mature will mostly discover exploit paths, not real models.

## Required Artifacts Per Epoch

Each autoresearch epoch should emit:

- `epoch_spec.json`
- `data_snapshot_manifest.json`
- `evaluation_contract.json`
- `champion_table.json`
- `challenger_log.json`
- `rejected_mutations.json`
- `stage1_calibration_frontier.json`
- `stage2_target_frontier.json`
- `promotion_audit.json`
- `numeric_justification.json`

Each promoted champion should emit:

- `mechanistic_spec.json`
- `state_trajectory_rows.json`
- `transition_hazard_summary.json`
- `observation_process_summary.json`
- `cascade_target_projection.json`

## Mapping To The Existing Repo

The current repo already contains pieces of this program.

### What Already Exists

- `TR-V2-00`: baseline lock
- `TR-V2-01`: direct Phase 2 hazard priors
- `TR-V2-02`: hidden shock hazards
- `TR-V2-03`: ablation suite

### What Should Happen Next

The next concrete experiment order should be:

1. `TR-V2-00` keep as immutable baseline lock
2. `TR-V2-01A` fit `F2` with direct observation-calibrated hazards and no Phase 2 surfaces
3. `TR-V2-01B` add `F1` direct Phase 2 priors on top of `TR-V2-01A`
4. `TR-V2-02A` add hidden shocks
5. `TR-V2-02B` add explicit VL testing and documented suppression observation process
6. `INC-V2-01` add `F4` explicit incidence augmentation with observed denominator `N_t`
7. `INC-V2-02` compare diagnosis hazard versus diagnosis-delay convolution
8. `CARE-V2-01` promote downstream hazards to semi-Markov form
9. `PHASE3-V2-INT` assemble the full `F8` integrated architecture
10. `TARGET-V2-01` run Stage 2 target-seeking toward `95-95-95`

This is the correct order.

The wrong order is:

1. optimize directly for `95-95-95`
2. hope the fitted model is scientific later

## Final Recommendation

Yes, build the unattended autoresearch loop.

Yes, make incidence explicit in the target Phase 3 architecture.

No, do not let the loop optimize the whole space forever against a visible `95-95-95` score.

The correct program is:

1. explicit incidence module with observed denominator `N_t` and latent `lambda_t`;
2. diagnosis module using either direct hazard or diagnosis-delay convolution;
3. semi-Markov downstream care hazards;
4. Phase 2 direct structure as structured priors and hidden modes as shared shocks;
5. explicit observation / ascertainment module;
6. frozen champion-challenger calibration loop first;
7. bounded target-seeking toward `95-95-95` second.

The correct implementation strategy is still staged:

1. fit the mechanistic transition and observation core;
2. make incidence explicit;
3. compare diagnosis hazard vs delay-kernel front half;
4. promote downstream care to semi-Markov form;
5. integrate the full architecture;
6. only then open the target-seeking loop.

That is the mathematically defensible version of the literature-backed design.
