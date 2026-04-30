# Phase 3 TR-V3 Shock-Probabilistic Representation Agent

**Date:** 2026-04-14  
**Role:** Representation / Modeling Agent  
**Question:** What are the best next model-side experiments for a publishable probabilistic shock-aware extension after the current deterministic winners?

---

## 1. Bottom Line

The right abstraction is **not** an endogenous shock predictor as the next live model.

The right abstraction is:

```text
calibrated residual distribution
+ optional rare-jump sidecar
+ only later an exogenous or shared-jump structure if it proves incremental
```

So:

- **yes** to a residual bootstrap / residual-jump sidecar as a first probabilistic extension
- **no** to treating jump prediction from the current endogenous quarterly heads as established
- **no** to making a shared shock factor the next promoted live model without stronger evidence

---

## 2. Direct Repo Evidence

### 2.1 Deterministic winners are stable

From [exp_eval_01_frozen_protocol_rebuild.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-eval-hardening-20260414-s00/analysis/exp_eval_01_frozen_protocol_rebuild.md):

- exact winner: `EXP-R10-M1-F1 = 0.069471` vs baseline `0.228036`
- dense winner: `EXP-R10-DENSE-M1-H1 = 0.085418` vs baseline `0.144010`

This means the next model pass should be an **uncertainty / probabilistic extension on top of a frozen deterministic backbone**, not another full deterministic family search.

### 2.2 Calibration is the main weakness

From [exp_eval_03_calibration_interval_coverage.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-eval-hardening-20260414-s00/analysis/exp_eval_03_calibration_interval_coverage.md):

- `EXP-R10-M1-F1` exact flow mean residual: `-604.734`
- `EXP-R10-M1-F1` exact flow 95 coverage: `0.333`
- `EXP-R10-DENSE-M1-H1` dense diagnosed mean residual: `-4924.889`
- `EXP-R10-DENSE-M1-H1` dense diagnosed coverage is `0.000` at 50, 80, and 95

So the next high-value model work is calibration-aware residual modeling.

### 2.3 Endogenous shock prediction is not supported yet

From [tr_v3_shock_plateau_predictability_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-shock-plateau-predictability-20260414-s00/analysis/tr_v3_shock_plateau_predictability_batch_report.md):

- exact shock probe: `insufficient_events` with event count `1`
- dense shock probe: `no_incremental_predictive_signal`
- dense candidate Brier: `0.130340`
- dense prevalence Brier: `0.121756`
- dense persistence Brier: `0.156863`
- dense candidate recall@K: `0.286`
- dense persistence recall@K: `0.429`

This blocks a claim that shock timing is currently predictable from the internal series.

### 2.4 Plateau signal exists, but plateau prediction is not proven

From [exp_g_check_02_plateau_census.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-grasp-falsification-20260414-s00/analysis/exp_g_check_02_plateau_census.md):

- exact observed plateau runs: `3`
- dense observed plateau runs: `12`

But from [tr_v3_shock_plateau_predictability_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-shock-plateau-predictability-20260414-s00/analysis/tr_v3_shock_plateau_predictability_batch_report.md):

- dense diagnosed plateau prediction: `no_incremental_plateau_signal`
- dense flow plateau prediction: `no_incremental_plateau_signal`

So plateau structure exists descriptively, but not yet as a predictive endogenous sidecar.

### 2.5 Simple piecewise null is not enough

From [exp_g_null_01_piecewise_baseline.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-grasp-falsification-20260414-s00/analysis/exp_g_null_01_piecewise_baseline.md):

- exact piecewise null: `0.075109`
- dense piecewise null: `0.128025`

This means piecewise segmentation alone does not replace the live winners.

### 2.6 Shared shock evidence is weak

From [exp_g_check_01_exact.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-grasp-falsification-20260414-s00/analysis/exp_g_check_01_exact.md):

- exact shared burst quarters: `1`

From [exp_g_check_01_dense.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-grasp-falsification-20260414-s00/analysis/exp_g_check_01_dense.md):

- dense shared burst quarters: `16`
- but pairwise correlations are tiny:
  - diagnosed vs ART: `-0.0187`
  - diagnosed vs flow: `0.0477`
  - ART vs flow: `0.0800`

So a shared-jump latent factor is not yet justified as the first promoted probabilistic model.

---

## 3. Is Residual Bootstrap Plus Jump Sidecar The Right Abstraction?

### 3.1 Yes, but only in a specific form

It is the right next abstraction if it means:

1. freeze the deterministic winner
2. model residuals train-only
3. allow a rare heavy-tail or jump component in residual space
4. do not claim that jump timing is currently predictable from the endogenous heads

That is identifiable enough because it only uses:

- observed residuals
- endpoint-specific calibration error
- optionally cross-endpoint residual co-occurrence

### 3.2 No, if it means "predict shocks directly"

It is the wrong abstraction if it means:

- train a shock classifier on the current quarterly heads and expect incremental signal
- promote a shared shock factor as if it were already empirically justified
- interpret residual jumps as biological epidemic shocks

The current evidence does not support that.

---

## 4. Alternative Model Paths

### A. Conformal residual layer

**Strength:** highest  
**Why:** directly answers the current coverage failure with minimal structural risk  
**Weakness:** honest intervals, but not a rich generative story

### B. Mixture residual model

**Strength:** high  
**Why:** captures heavy tails and rare jumps without claiming jump predictability  
**Weakness:** needs careful shrinkage to avoid overfitting sparse exact events

### C. Hierarchical shared-jump factor

**Strength:** medium  
**Why:** can share rare jump mass across `D`, `A`, `F`  
**Weakness:** weak current evidence for truly shared shocks

### D. Covariate-conditioned jump hazard

**Strength:** medium later, low now  
**Why:** could become publishable if exogenous covariates explain residual jumps  
**Weakness:** current endogenous predictability probes already failed; needs new external signals

### E. State-space random-effects residual layer

**Strength:** medium-high  
**Why:** good fit to persistent dense diagnosed underprediction  
**Weakness:** more machinery than conformal / mixture without a clear first-paper advantage

### F. Changepoint nulls

**Strength:** useful comparator  
**Why:** interpretable baseline  
**Weakness:** already materially weaker than live winners

---

## 5. Ranked Experiment Ladder

## 5.1 `EXP-UQ-01` Split-conformal residual layer

**Rank:** 1  
**Why first:** coverage is the clearest live weakness

**Implementation shape**

- base models:
  - `EXP-R10-M1-F1`
  - `EXP-R10-DENSE-M1-H1`
- compute train-only rolling residual scores by:
  - endpoint
  - contract
  - era
  - provenance tier
- produce 50/80/95 intervals with split-conformal or cross-split empirical quantiles

**Keep if**

- dense diagnosed 95 coverage rises materially above `0.000`
- exact flow 95 coverage rises materially above `0.333`
- point MAE does not regress materially

**Why identifiable enough**

- no latent shock timing
- only uses observed residual distributions

## 5.2 `EXP-UQ-02` Mixture residual / jump sidecar

**Rank:** 2  
**Why second:** this is the cleanest "shock-aware" extension that does not overclaim predictability

**Implementation shape**

- residual model per endpoint:
  \[
  e_t \sim (1-\pi) \mathcal{N}(0,\sigma_0^2) + \pi \mathcal{N}(0,\sigma_1^2)
  \]
  or Student-t base plus rare jump component
- fit on train-only residuals
- separate exact and dense contracts
- optionally allow era-specific mixture weights

**Keep if**

- interval coverage improves over `EXP-UQ-01`, especially on dense diagnosed
- point MAE stays flat
- jump mass remains sparse and stable across rolling splits

**Why identifiable enough**

- uses residual tails directly
- does not require jump timing to be predicted

## 5.3 `EXP-UQ-03` Hierarchical shared-jump factor

**Rank:** 3  
**Why third:** only after the marginal jump sidecar works

**Implementation shape**

- latent jump indicator \(J_t \in \{0,1\}\)
- endpoint residuals:
  \[
  e_{m,t} = \alpha_m J_t + \epsilon_{m,t}
  \]
- shrink loadings heavily
- start with exact and dense trained separately
- no endogenous predictors yet

**Keep if**

- beats marginal mixture residuals on dense diagnosed calibration
- inferred jump quarters align across endpoints often enough to be stable

**Revert if**

- it recreates the weak shared-signal problem already seen in `EXP-G-CHECK-01`

**Why identifiable enough**

- only if fitted as a residual-cooccurrence model
- not if treated as a general hidden shock engine

## 5.4 `EXP-UQ-04` Covariate-conditioned jump hazard

**Rank:** 4  
**Why fourth:** only after a jump sidecar is real

**Implementation shape**

- exogenous covariates only:
  - reporting calendar features
  - known COVID policy windows
  - bulletin delay indicators
  - maybe national testing/service program covariates if available
- logistic jump hazard:
  \[
  \Pr(J_t = 1 \mid x_t) = \operatorname{logit}^{-1}(\beta^\top x_t)
  \]

**Keep if**

- Brier beats both prevalence and persistence baselines
- recall@K beats persistence

**Why identifiable enough**

- only with genuinely external covariates
- not with the current endogenous heads alone

## 5.5 `EXP-UQ-05` Residual local-level / random-effects state-space layer

**Rank:** 5  
**Why fifth:** best answer to persistent dense diagnosed bias if simpler methods still fail

**Implementation shape**

- keep deterministic base path
- add residual local level:
  \[
  r_{t+1} = r_t + \eta_t
  \]
- or mean-reverting version:
  \[
  r_{t+1} = \phi r_t + \eta_t
  \]
- endpoint-specific, with stronger shrinkage on exact lane

**Keep if**

- diagnosed calibration improves without widening intervals excessively

**Why identifiable enough**

- residual persistence is visible in calibration artifacts
- but this is a heavier model than the first two steps

## 5.6 `EXP-UQ-06` Changepoint null as a standing comparator

**Rank:** comparator only  
**Why:** paper-grade benchmark, not mainline development

**Implementation shape**

- retain `EXP-G-NULL-01`
- maybe add one penalized changepoint residual model
- no promotion unless it unexpectedly overtakes the live winners

---

## 6. Bounded Autoreason Decision

I would frame the next probabilistic phase as:

- `A`: conformal / calibration-first
- `B`: shock-aware residual-jump sidecar
- `AB`: calibration-first plus rare-jump residual sidecar

Winner: `AB`, with sequencing:

1. conformal residual layer first
2. then mixture residual / jump sidecar
3. only then shared-jump or exogenous jump hazard

This wins because:

- `A` alone is the safest technical step, but not the most interesting publishable extension
- `B` alone is too under-supported by the current shock predictability evidence
- `AB` keeps the identifiable calibration gains while still opening a defensible shock-aware paper angle

---

## 7. Practical Recommendation

Do next:

1. `EXP-UQ-01` split-conformal intervals on `EXP-R10-M1-F1` and `EXP-R10-DENSE-M1-H1`
2. `EXP-UQ-02` train-only mixture residual / jump sidecar
3. `EXP-UQ-03` only if residual jump co-occurrence is stable

Do not do next:

- endogenous jump prediction from current quarterly heads
- direct shared-shock promotion into the point forecaster
- quarterly explicit `S(t)`
- richer leakage as a live probabilistic block

The publishable probabilistic extension is most likely:

```text
frozen deterministic observation-first forecast
+ provenance-aware conformal calibration
+ rare-jump residual sidecar
```

That is scientifically honest, repo-aligned, and identifiable enough to execute now.
