# Phase 3 TR-V3 Representation Audit

**Date:** 2026-04-12  
**Role:** Representation / Modeling Agent  
**Scope:** What the current winner actually models, what mechanistic content still survives honestly, and which next model family has the highest probability of success under the current benchmark and data contract.

---

## 1. Bottom Line

The current winning family is not a latent epidemic model.

It is an **observation-first quarterly stock/flow forecaster** that:

- forecasts `diagnosed_plhiv` directly,
- forecasts `alive_on_art` directly,
- forecasts `new_diagnosed_cases_period` directly,
- carries or lightly blends a suppression share,
- then back-computes diagnostic `D_to_A` and `A_to_V` hazards from those forecasted observations.

This is not a criticism. It is simply the honest mathematical description of the winner.

The mechanistic `TR-V3` branch still contains scientifically useful structure, but under the current archive and benchmark it does **not** survive as the primary predictive model. The predictive and mechanistic goals have already split in practice.

Evidence:

- winner registry and contract split in [phase3_track_registry.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-experiment-suite-design-split-20260412-s43/analysis/phase3_track_registry.md)
- canonical benchmark gap in [canonical_14_vs_benchmark.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-experiment-suite-design-split-20260412-s43/analysis/canonical_14_vs_benchmark.md)
- search frontier collapse to `R10` variants in [tr_v3_repair_search_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-repair-search-20260412-s05/analysis/tr_v3_repair_search_report.md)

---

## 2. What The Winner Is Really Modeling Mathematically

The current exact-contract winner is `EXP-R10-EXACT-CHAMPION` in [tr_v3_experiment_suite.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_experiment_suite.py).

Its implementation is in [_fit_direct_observation_repair_candidate](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_experiment_suite.py#L2202).

The critical fact is that this path explicitly discards the dynamic hazard and observation-model objects:

- `del dynamic_cfg`
- `del observation_cfg`

So the winner is not using the `TR-V3-05a/05b` dynamic hazard machinery as the core predictive object.

### 2.1 Forecasted quantities

Let:

- `D_t` = diagnosed stock
- `A_t` = ART stock
- `F_t` = new diagnosed cases in the quarter
- `V_t` = virally suppressed

The winner fits supported-series forecasters separately for:

- `D_t`
- `A_t`
- `F_t`

using only `exact_observed` and `bridge_observed` support points.

The exact winner settings are:

- `diagnosed_weight = 1.0`
- `art_weight = 1.0`
- `flow_weight = 0.25`
- `diagnosed_series_model = level`
- `art_series_model = level`
- `flow_series_model = delta`
- `suppression_carry_weight = 0.75`

### 2.2 Level-series forecaster

For a supported level series `y_t`, the model fits a bounded delta recurrence:

\[
\Delta_t \approx \beta_0 + \beta_1 \, dt_t + \beta_2 \, \Delta_{t-1}
\]

then reconstructs the level forward:

\[
\hat y_t = \hat y_{t-1} + \hat \Delta_t
\]

with clipping to observed support-based bounds.

In plain English:

- estimate how much the stock tends to change,
- keep that change bounded by observed behavior,
- roll the stock forward quarter by quarter.

This is used directly for:

- diagnosed stock in the exact winner
- ART stock in the exact winner

### 2.3 Delta-series forecaster

For diagnosis flow, the exact winner uses a delta model:

\[
\delta_t \approx \alpha + s \, p_t + \rho \, \delta_{t-1}
\]

with bounded drift and then blends the forecast with the last observed flow delta.

In plain English:

- forecast how the quarterly flow is changing,
- not just its absolute level,
- then damp that forecast back toward recent support.

### 2.4 Suppression term

Suppression is not freely forecast.

The model computes a carried suppression share:

\[
\pi_V = \text{last supported } \frac{V}{A}
\]

then uses:

\[
V_t \approx 0.75 \, (\pi_V A_t) + 0.25 \, V_{t-1}
\]

if suppression support exists, otherwise it simply carries the last available suppression level.

### 2.5 Diagnostic hazards

After forecasting `D_t`, `A_t`, `F_t`, and `V_t`, the model back-computes:

\[
h_{D \to A,t} \approx \frac{\max(A_t - A_{t-1}, 0)}{\max(D_{t-1}, \varepsilon)}
\]

and

\[
h_{A \to V,t} \approx \frac{\max(V_t - \pi_V A_{t-1}, 0)}{\max(A_{t-1}(1-\pi_V), \varepsilon)}
\]

while setting:

\[
h_{U \to D,t}=0,\quad h_{A \to L,t}=0,\quad h_{L \to A,t}=0
\]

So the hazard curves emitted for `R10` are **diagnostic reconstructions**, not the real driver of the forecast.

### 2.6 Representational interpretation

The winner is therefore best described as:

\[
\text{direct multivariate quarterly stock/flow forecasting with bounded drift and carried suppression share}
\]

not:

\[
\text{latent mechanistic epidemic inference}
\]

That is why it wins. It forecasts the objects that are actually scored.

---

## 3. What Mechanistic Content Survives Honestly

The best mechanistic research anchor is still `EXP-R1` in [tr_v3_experiment_suite.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_experiment_suite.py#L184).

This is the last branch where the model still means roughly what the design memo intended:

- explicit transitions,
- support-aware hazard fitting,
- bounded drift instead of exploding logit extrapolation,
- explicit `U/D/A/V/L` state evolution.

What survives honestly:

### 3.1 Support-aware transition fitting

The `strict_support_drift` logic is a real methodological gain. It prevents unsupported zero-flow quarters from being interpreted as strongly identified hazards.

This is scientifically important and should remain in any future mechanistic branch.

### 3.2 Flow-based state semantics

The `U/D/A/V/L` graph is still meaningful as a **reconciliation scaffold**, especially for:

- diagnosis pressure,
- care initiation,
- leakage hypotheses,
- interpretation of stock-flow mismatches.

### 3.3 Regime awareness belongs in the mechanistic branch too

The design memo is correct that non-stationarity must be explicit. The predictive winner proves this indirectly.

What does **not** survive honestly under current support:

### 3.4 Free quarterly `D_to_A` and `A_to_V` hazards

These are the weak point. The repair family from `R2` through `R9` shows that:

- `D_to_A` can be partially helped by stock reconciliation,
- `A_to_V` is mostly not identified and falls back to carry logic,
- richer care/leakage structure quickly hurts broader-history performance.

### 3.5 Rich leakage network

The data do not yet justify a fully free:

- `D -> L`
- `A -> L`
- `V -> L`
- `L -> A`
- `L -> D`

network at quarterly national level.

### 3.6 Quarterly mortality block

The current archive supports an annual deaths anchor, not an honest quarterly compartment-specific mortality system.

So the mechanistic content that survives honestly is:

- support-aware transition estimation,
- bounded-drift transition forecasting,
- a reduced `U/D/A/V/L` reconciliation skeleton,
- annual incidence and annual deaths as external anchors,
- regime-aware structure,

but **not** a fully free quarterly latent epidemic system.

---

## 4. What The Search Frontier Is Telling Us

The search report is decisive:

- the Pareto frontier is entirely `R10`-family variants
- no mechanistic family survives the current exact+dense tradeoff

See [tr_v3_repair_search_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-repair-search-20260412-s05/analysis/tr_v3_repair_search_report.md).

This means three things.

### 4.1 The benchmark mostly rewards observed-target forecasting

That is not a flaw in the benchmark. It is a statement about where information actually exists in the data.

### 4.2 The mechanistic family is still paying an identifiability tax

The mechanistic variants spend model capacity on latent hazards that the archive does not identify well enough.

### 4.3 The next successful model family should not try to out-mechanize `R10`

The next family with highest success probability is not:

- bigger leakage
- richer mortality
- explicit `S(t)`
- more free hazard blocks

It is:

- keep the observation-first predictive lane,
- add a thin constrained latent reconciliation layer behind it,
- and only estimate latent structure where the benchmark and support both justify it.

---

## 5. Next Model Family With Highest Success Probability

The next model family should be:

## `R11 = observation-first state-space reconciliation`

This should sit between pure `R10` and mechanistic `R1`.

### 5.1 Core idea

Keep direct forecasting of the scored observables:

- `diagnosed_plhiv`
- `alive_on_art`
- `new_diagnosed_cases_period`

but add a constrained latent reconciliation layer that enforces:

- nonnegative states,
- stock-flow consistency,
- care-stock consistency,
- annual incidence anchor consistency,
- optional annual deaths anchor consistency.

### 5.2 What changes mathematically

Instead of forecasting diagnostic hazards first, forecast observed targets first:

\[
\hat D_t,\quad \hat A_t,\quad \hat F_t
\]

then solve a constrained reconciliation problem for latent states and flows:

\[
\min_{x_t, f_t} \; 
\lambda_D \|\mathrm{diag}(x_t)-\hat D_t\|^2
+ \lambda_A \|\mathrm{art}(x_t)-\hat A_t\|^2
+ \lambda_F \|f_{UD,t}-\hat F_t\|^2
+ \lambda_S \|\Delta x_t - G(x_{t-1}, f_t)\|^2
+ \lambda_R \,\text{regime smoothness penalty}
\]

subject to:

\[
x_t \ge 0,\qquad f_t \ge 0
\]

and optional annual constraints:

\[
\sum_{q \in y} \iota_q \approx y^{inc}_y
\]

\[
\sum_{q \in y} m_q \approx y^{death}_y
\]

In plain English:

- let the predictive lane keep driving the quarterly forecast,
- then find the most plausible latent care/diagnosis story consistent with those forecasts,
- instead of asking weak latent hazards to generate the forecast from scratch.

### 5.3 Why this family has the highest success probability

Because it preserves what already works:

- direct forecasting of supported quarterly observables

while reintroducing mechanistic content only where it can be constrained:

- latent stock-flow consistency
- annual anchor consistency
- regime-smoothness constraints

This is the most plausible path to recovering mechanistic interpretability without giving away the predictive gains.

---

## 6. Three Concrete Bounded Experiments

These are the three highest-probability next experiments.

### Experiment 1: `EXP-R11a-observation-first-reconciliation`

**Goal:** Add a latent reconciliation layer behind `R10`, with no annual mortality yet.

**Keep fixed:**

- direct forecast of `diagnosed_plhiv`
- direct forecast of `alive_on_art`
- direct forecast of `new_diagnosed_cases_period`
- carried suppression share

**Add:**

- latent `U/D/A/V/L` state solve after forecasting
- nonnegative states and flows
- consistency penalty between latent states and forecast observables

**Why it has high success probability:**

- minimal disturbance to the winning predictive structure
- adds only a constrained latent layer, not a new free hazard family

**Success criterion:**

- preserve or nearly preserve `R10` exact and dense MAE
- improve latent consistency diagnostics over `R10`

### Experiment 2: `EXP-R11b-regime-aware-reconciliation`

**Goal:** Make the reconciliation layer regime-aware without changing the direct forecast family.

**Add:**

- train-only segmentation or support-aware regime labels
- different reconciliation penalties by regime

For example:

- sparse early regime: stronger stock-reconciliation weight
- supported late regime: stronger flow-consistency weight

**Why it has high success probability:**

- `R7` already showed that regime-aware care handling helps early years
- the mistake in `R7` was letting that regime logic drive the whole forecast
- in `R11b`, regime structure only governs reconciliation, not the top-level forecast

**Success criterion:**

- dense-history MAE stays close to dense champion
- early bridge-year latent inconsistency decreases

### Experiment 3: `EXP-R11c-annual-anchor-reconciliation`

**Goal:** Add annual incidence and annual deaths anchors as soft constraints only.

**Add:**

- annual incidence penalty
- annual deaths penalty

Do **not** add:

- quarterly mortality hazards
- explicit `S(t)`
- richer leakage network

**Why it has high success probability:**

- the design memo already treats annual anchors as the honest support level for infection pressure and mortality
- soft annual constraints can improve scientific plausibility without forcing weak quarterly identification

**Success criterion:**

- quarterly MAE remains near `R10`
- annual incidence/deaths consistency improves relative to `R10`

---

## 7. Experiments To Avoid Next

These have lower expected value under the current evidence.

- full `TR-V3-05b` revival with open inflow, leakage, and mortality all at once
- explicit `S(t)` now
- richer quarterly mortality block
- free `A_to_V` hazard estimation
- larger search over old `R1/R6/R7` knobs

These are not impossible forever. They are just not the highest-probability next move under the current archive and benchmark.

---

## 8. Final Representation Judgment

The winner is a **bounded multivariate quarterly program forecaster**, not an epidemic generator.

That means the next successful model should not try to reverse that fact. It should build on it.

So the honest roadmap is:

1. keep `R10` as predictive champion,
2. add constrained latent reconciliation behind it,
3. make that reconciliation regime-aware,
4. add annual anchors before adding richer epidemic compartments.

That is the highest-probability path to a model that is both useful and scientifically defensible.
