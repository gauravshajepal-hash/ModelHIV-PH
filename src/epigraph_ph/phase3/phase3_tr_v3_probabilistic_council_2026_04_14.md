# Phase 3 TR-V3 Probabilistic Next Phases Council

**Date:** 2026-04-14  
**Question:** next experiments after the promoted exact and dense candidates, with special attention to Monte Carlo residual-plus-jump models, exogenous covariates, calibration, and publishable shock-aware extensions  
**Mode:** same-model council with bounded autoreason arbitration  
**Winning deterministic models:** `EXP-R10-M1-F1-C1` on `exact_only`, `EXP-R10-DENSE-M1-C1-H1` on `purged_dense`

---

## 1. Bottom Line

The next publishable line is:

```text
calibration-first probabilistic forecasting
+ bounded rare-jump residual sidecar
+ strictly lagged exogenous covariates only if they beat persistence/prevalence baselines
```

The council rejected the idea of promoting an endogenous shock model now.

The current evidence supports:

- strong deterministic point forecasts
- weak uncertainty calibration
- no incremental endogenous shock predictability
- no incremental plateau predictability

So the correct next phase is a **probabilistic sidecar** on top of the frozen winners, not a new shock-driven mean model.

---

## 2. Current Frontier

### 2.1 Exact lane

From [tr_v3_exact_calibration_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-exact-calibration-20260414-s00/analysis/tr_v3_exact_calibration_batch_report.md):

- `EXP-R10-M1-F1-C1`: quarterly mean MAE `0.068651`
- lockbox `2025`: `0.041018`

### 2.2 Dense lane

From [tr_v3_dense_calibration_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-dense-calibration-20260414-s02/analysis/tr_v3_dense_calibration_batch_report.md):

- `EXP-R10-DENSE-M1-C1-H1`: purged-dense quarterly mean MAE `0.081361`
- lockbox `2025`: `0.057008`

### 2.3 Shock / plateau probe

From [tr_v3_shock_plateau_predictability_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-shock-plateau-predictability-20260414-s00/analysis/tr_v3_shock_plateau_predictability_batch_report.md):

- exact shock events: `1`
- dense shock status: `no_incremental_predictive_signal`
- dense shock candidate Brier: `0.130340`
- dense shock prevalence baseline Brier: `0.121756`
- dense plateau probes: no metric beat both prevalence and persistence baselines

This is the decisive evidence against promoting an endogenous shock model now.

---

## 3. Direct Evidence Table

| Claim | Evidence For | Evidence Against | Council Read |
|---|---|---|---|
| deterministic `R10` line is mature enough to freeze | exact and dense winners improved again after `C1` calibration | none materially stronger than current winners | `true` |
| calibration is the main remaining weakness | earlier coverage audit showed severe undercoverage; dense diagnosed bias was large before `C1` | point MAE is already strong | `true` |
| endogenous shocks are forecastable from quarterly history alone | some dense shared-burst quarters exist | no incremental Brier vs prevalence; exact has only one event | `false for now` |
| plateaus are forecastable from quarterly history alone | observed plateau runs exist | no incremental plateau predictability | `false for now` |
| piecewise null fully explains the winners | changepoint null beats carry-forward | still materially weaker than promoted winners | `false` |
| a probabilistic residual sidecar is justified | deterministic winners are strong; calibration is weak; uncertainty claims are missing | residual pools can leak tiers/eras if handled badly | `true if provenance-aware` |
| exogenous covariates might still help | shock prediction probably requires outside information | no evidence yet that the needed covariates are forecast-time clean or strong enough | `possible but unproven` |

---

## 4. Contextual Evidence Table

| Topic | Context |
|---|---|
| GRASP-style shock logic | useful as an optimization analogy, not as proof that latent epidemic shocks are currently identifiable |
| residual bootstrap | plausible if tier-aware and train-only; invalid if iid-pooled across eras and tiers |
| jump process | should be modeled as rare residual volatility or event-risk, not as a new mean-state epidemic driver |
| exogenous covariates | only acceptable if strictly lagged and forecast-origin available |
| publishability | strongest paper path is deterministic benchmark plus probabilistic calibration / risk extension |

---

## 5. Council Positions

### 5.1 Validity Skeptic

Main conclusion:

- the shock-aware idea is only defensible as an uncertainty-sidecar
- a direct endogenous jump process would overclaim
- placebo and persistence baselines must be beaten before any shock claim is made

### 5.2 Evaluation / Failure Agent

Main conclusion:

- do not let shocks touch the mean forecast
- first rebuild calibration and interval coverage for the promoted winners
- then test residual-risk and variance conditioning

### 5.3 Representation / Modeling Agent

Main conclusion:

- residual bootstrap plus a rare-jump sidecar is the right abstraction
- not a latent shock engine
- not a quarterly `S(t)` or richer leakage restart

---

## 6. Bounded Autoreason

The bounded arbitration compared:

- `A`: calibration-first probabilistic layer
- `B`: jump-sidecar now
- `AB`: calibration-first plus a bounded rare-jump residual sidecar later

**Winner: `AB`**

Reason:

- `A` is the safest scientifically
- `B` overclaims after the failed endogenous predictability probe
- `AB` preserves the publishable probabilistic extension without pretending shocks are already forecastable

---

## 7. Ranked Experiment Ladder

### 1. `EXP-UQ-01`
**Provenance-aware conformal / empirical interval layer**

Build train-only interval calibration on top of:

- exact: `EXP-R10-M1-F1-C1`
- dense: `EXP-R10-DENSE-M1-C1-H1`

Use separate residual pools by:

- contract
- endpoint
- tier

Primary metrics:

- coverage `50/80/95`
- WIS
- interval width

Keep if:

- coverage improves materially without trivial width inflation
- WIS improves on both exact and dense

Revert if:

- coverage only improves by exploding width
- dense diagnosed coverage remains poor

Risk:

- residual pools are too small if oversplit

### 2. `EXP-UQ-02`
**Tier-aware moving-block residual bootstrap**

Prototype Monte Carlo draws from empirical residual blocks rather than iid residuals.

Do not mix:

- exact with bridge
- pre-COVID with later eras unless justified

Primary metrics:

- WIS
- coverage
- lockbox calibration

Keep if:

- it beats the simpler interval layer on WIS or coverage at comparable width

Revert if:

- gains disappear on the `2025` lockbox
- pooled residual variants outperform the provenance-aware version

Risk:

- small effective sample size after tier / era separation

### 3. `EXP-RISK-01`
**High-error event prediction**

Do not predict “epidemic shocks.” Predict **high-error quarters** instead.

Targets:

- `abs_residual > q80`
- `abs_residual > q90`

Use only forecast-origin features.

Primary metrics:

- Brier
- prevalence baseline Brier
- persistence baseline Brier

Keep if:

- it beats both prevalence and persistence on dense
- and shows at least nontrivial signal on exact

Revert if:

- it only beats one baseline
- or only works on bridge diagnosed rows

Risk:

- event counts are still small

### 4. `EXP-UQ-03`
**Rare-jump residual sidecar**

Add a residual mixture:

\[
\epsilon_t \sim (1-p_t)\,F_{\mathrm{base}} + p_t\,F_{\mathrm{jump}}
\]

At quarterly resolution, use a **Bernoulli jump indicator** first, not a Poisson count process.

This should alter:

- predictive intervals
- tail-risk probabilities

Not:

- deterministic mean forecasts

Primary metrics:

- WIS
- conditional coverage on high-risk quarters
- Brier for high-error event occurrence

Keep if:

- it improves WIS or conditional coverage over `EXP-UQ-02`
- without degrading lockbox calibration

Revert if:

- it does not beat the no-jump probabilistic baseline

Risk:

- confounding with tier artifacts and backlog release

### 5. `EXP-UQ-04`
**Strictly lagged exogenous jump-hazard model**

Only after `EXP-RISK-01` and `EXP-UQ-03`.

Candidate covariates:

- COVID phase indicators
- reporting calendar effects
- testing pulse indicators
- any other forecast-time-clean disruption signals

Model only the jump probability or interval scale:

\[
\mathrm{logit}(p_t)=\alpha + x_t^\top \beta
\]

Primary metrics:

- Brier improvement over residual-only risk model
- WIS improvement

Keep if:

- it beats both prevalence and persistence baselines
- and improves the residual-only probabilistic model

Revert if:

- gains are era-specific only
- covariates mostly proxy provenance changes

Risk:

- forecast-time leakage

### 6. `EXP-UQ-05`
**Shared volatility-regime sidecar**

One shared high-volatility quarter-level regime across:

- `diagnosed_plhiv`
- `alive_on_art`
- `new_diagnosed_cases_period`

This is a variance model, not a mean model.

Primary metrics:

- WIS
- CRPS if a full predictive mixture is emitted
- coverage on high-risk quarters

Keep if:

- it materially improves probabilistic scores beyond `EXP-UQ-03`

Revert if:

- the shared regime just replays persistence

Risk:

- weak cross-metric coherence may make this too correlated with noise

### 7. `EXP-PLAT-01`
**Plateau-conditioned uncertainty sidecar**

Only if the residual-risk experiments show plateau-linked undercoverage.

Do not promote as a mean model.

Primary metrics:

- conditional coverage during plateau-like runs
- WIS

Keep if:

- plateau-conditioned undercoverage improves materially

Revert if:

- no incremental gain over the generic probabilistic sidecar

Risk:

- event scarcity

---

## 8. Interventions To Avoid

- Do not promote an endogenous shock model now.
- Do not use a Poisson jump count process as the first quarterly jump model.
- Do not let the jump layer mutate the deterministic mean forecast.
- Do not mix exact and bridge residual pools blindly.
- Do not use contemporaneous or revised exogenous covariates.
- Do not reopen quarterly `S(t)` or richer leakage as live benchmark blocks.

---

## 9. Recommended First Batch

Run in this order:

1. `EXP-UQ-01`
2. `EXP-UQ-02`
3. `EXP-RISK-01`
4. `EXP-UQ-03`
5. `EXP-UQ-04` only if `EXP-RISK-01` and `EXP-UQ-03` pass

Stop early if:

- coverage becomes acceptable but event-risk prediction does not beat persistence/prevalence

That would still leave a publishable probabilistic calibration paper without a shock-prediction claim.

---

## 10. AutoResearch Handoff

- **Variant:** `evidence-to-model-loop`
- **Evaluation harness:** frozen exact and dense winners, plus retroactive `2025` lockbox
- **Mutation units:** one probabilistic sidecar change at a time
- **Stop rule:** keep only if the trusted probabilistic score improves without invalidating point-forecast stability or forecast-time honesty

### First Experiments

1. `EXP-UQ-01`
2. `EXP-UQ-02`
3. `EXP-RISK-01`

