# Phase 3 TR-V3 Autoresearch Design

**Date:** 2026-04-10  
**Author:** Epigraph PH (G S + AI agent)  
**Status:** Draft  
**Selected autoresearch variant:** `evidence-to-model-loop`

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

---

## 12. Why This Is Still Worth Doing

There is already substantial HIV modeling literature.

So the value of this repo is **not**:

- “nobody has modeled HIV transitions before”

The value would be:

- a Philippines-specific, open, mixed-evidence, blocked-time-validated national forecasting benchmark;
- transparent negative and positive results under a stricter contract;
- and a model family that can beat simple baselines honestly, not just look mechanistic on paper.

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

This is the scientifically correct next autoresearch program under the new blocked-time contract.
