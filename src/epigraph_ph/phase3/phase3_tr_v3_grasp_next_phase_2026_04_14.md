# Phase 3 TR-V3 GRASP-Inspired Next Phase

**Date:** 2026-04-14  
**Source prompt:** adapt [GRASP](https://www.michaelpsenka.io/grasp/) and [arXiv:2602.00475](https://arxiv.org/abs/2602.00475) to the current Phase 3 quarterly HIV forecasting line  
**Current live winners:** `EXP-R10-M1-F1` on `exact_only`, `EXP-R10-DENSE-M1-H1` on `purged_dense`  
**Current scientific constraint:** the benchmark is now stable, but calibration and dense diagnosed-stock residual structure are still weak

---

## 1. Bottom Line

GRASP is useful here, but not literally.

We should **not** copy:

- world-model planning over image latents
- action-sequence control as the main problem
- direct state optimization with no epidemiologic structure

We **should** copy:

- lifted intermediate states
- soft consistency instead of fully serial fitting
- sparse stochastic exploration around shocks
- stop-gradient style protection against brittle state cheating
- periodic sync back to the true recursive rollout

The right next phase is therefore:

```text
R10-G = lifted observation-state forecasting with sparse shock and plateau structure
```

This should be treated as a structured extension of the current `R10` observation-first family, not as a return to the failed fully mechanistic quarterly hazard line.

---

## 2. Why GRASP Matters For Us

GRASP solves a problem that is very close to ours in shape, even though the application is different.

Their setting:

- planning over long horizons is hard
- pure serial optimization is brittle
- gradients through latent states can exploit unrealistic directions
- non-greedy solutions need temporary detours and shocks

Our setting:

- fitting quarter-by-quarter recursive models is brittle
- the best model so far is observation-first, not mechanistic
- abrupt reporting and service shocks matter
- plateaus and backlog releases are real
- forcing everything through a one-step recursive fit can create unstable residual structure

So the GRASP lesson for Phase 3 is:

- fit a whole quarter sequence jointly,
- allow soft consistency rather than exact one-step recursion,
- add explicit sparse shock structure,
- and periodically sync back to the true recursive forecast objective.

That is directly relevant.

---

## 3. What To Borrow And What To Reject

### 3.1 Borrow

1. **Lifted states**
   Instead of fitting only the recursive one-step path, optimize intermediate quarterly latent targets over the whole train/holdout split.

2. **Soft dynamics consistency**
   Penalize inconsistency between adjacent quarters, but do not require exact recursive equality during the inner optimization.

3. **Noise / multi-start exploration**
   Use bounded stochasticity on the shock layer, not on the final reported outputs.

4. **Stop-gradient protection**
   Do not let the optimizer exploit fragile gradients through both sides of the same latent consistency map.

5. **Periodic sync**
   After optimizing the lifted path, project back to the actual recursive rollout and refine there.

### 3.2 Reject

1. Full latent epidemic state optimization.
2. Image-like latent-state freedom.
3. Direct quarterly `S(t)` promotion.
4. Richer leakage promotion.
5. Any claim that this becomes a mechanistic epidemic model.

---

## 4. Proposed Model Family

Name:

```text
TR-V3 R10-G
```

Interpretation:

- `R10` observation-first outer model stays the base class.
- `G` adds a GRASP-inspired lifted optimization layer for shocks and plateaus.

Primary scored heads:

- `D_t = diagnosed_plhiv`
- `A_t = alive_on_art`
- `F_t = new_diagnosed_cases_period`

Sidecar only:

- `V_t = virally_suppressed`

Suppression honesty remains:

- if support is absent, leave it `unsupported_or_unclaimed`

---

## 5. Mathematical Translation

Let

\[
y_t = \begin{bmatrix} D_t \\ A_t \\ F_t \end{bmatrix}
\]

be the observed quarterly stock/flow vector.

Let

\[
\tilde y_t
\]

be a **virtual quarterly state** for the observation model.

Let

\[
z_t
\]

be a low-dimensional shared shock/regime variable.

Let the baseline one-step observation transition be:

\[
\hat y_{t+1} = G_\theta(\tilde y_t, z_t)
\]

where `G_\theta` is a structured observation-first transition map, not a free neural network.

### 5.1 Lifted consistency objective

Instead of only fitting a serial rollout, optimize:

\[
\mathcal{L}_{\mathrm{dyn}}
=
\sum_{t=0}^{T-1}
\left\|
G_\theta(\bar{\tilde y}_t, z_t) - \tilde y_{t+1}
\right\|_{W_t}^2
\]

where:

- \(\bar{\tilde y}_t\) is a stop-gradient copy of \(\tilde y_t\)
- \(W_t\) is a metric/tier weight matrix

This is the direct GRASP adaptation:

- keep gradients through the transition map into the shock/control side
- avoid letting the optimizer exploit both ends of the same latent state relation at once

### 5.2 Observation anchoring

Only observed data should anchor the lifted states:

\[
\mathcal{L}_{\mathrm{obs}}
=
\sum_{t=0}^{T}
\left\|
M_t(\tilde y_t - y_t)
\right\|^2
\]

where \(M_t\) masks missing values and can downweight bridge observations relative to exact observations.

### 5.3 Plateau penalty

Plateaus should be encouraged through a sparse curvature penalty:

\[
\mathcal{L}_{\mathrm{plateau}}
=
\lambda_{\mathrm{plateau}}
\sum_{t=1}^{T-1}
\left\|
\tilde y_{t+1} - 2\tilde y_t + \tilde y_{t-1}
\right\|_1
\]

Plain English:

- if the curve is smooth or flat, this penalty stays small
- if the curve bends sharply every quarter, the penalty grows
- this encourages long plateaus and a small number of structural bends

### 5.4 Sparse shared shocks

Add a shared sparse shock factor:

\[
\mathcal{L}_{\mathrm{shock}}
=
\lambda_{\mathrm{shock}}
\sum_{t=0}^{T-1}
\|z_t\|_1
\]

and optionally a fused penalty:

\[
\mathcal{L}_{\mathrm{shock\_tv}}
=
\lambda_{\mathrm{tv}}
\sum_{t=1}^{T-1}
\|z_t - z_{t-1}\|_1
\]

Plain English:

- most quarters should have no shock
- when shocks appear, they should be few and interpretable
- the fused term makes shocks persist briefly instead of jittering randomly every quarter

### 5.5 Stock-flow consistency

The exact-lane `F1` lesson should stay:

\[
\mathcal{L}_{\mathrm{flow}}
=
\lambda_{\mathrm{flow}}
\sum_t
\left(
F_t - \Psi(D_t, D_{t-1})
\right)^2
\]

where \(\Psi\) is the train-only stock-flow reporting relation already used in `EXP-R10-M1-F1`.

### 5.6 Final inner objective

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{dyn}}
\mathcal{L}_{\mathrm{obs}}
\mathcal{L}_{\mathrm{plateau}}
\mathcal{L}_{\mathrm{shock}}
\mathcal{L}_{\mathrm{shock\_tv}}
\mathcal{L}_{\mathrm{flow}}
\]

subject to:

\[
A_t \le D_t,\qquad D_t \ge 0,\qquad A_t \ge 0,\qquad F_t \ge 0
\]

and suppression sidecar honesty rules unchanged.

---

## 6. Optimization Strategy

This is where the GRASP adaptation matters most.

### 6.1 Inner lifted phase

Optimize over:

- virtual states \(\tilde y_t\)
- shared shock states \(z_t\)

Use:

- deterministic gradient steps on \(\tilde y_t\)
- small stochastic perturbations on \(z_t\) only
- multiple short restarts rather than one huge free run

Reason:

- stochasticity belongs on the shock layer, not on the public outputs

### 6.2 Sync phase

Every `K_sync` inner iterations:

1. discard the lifted path
2. roll forward recursively from the real train endpoint using the current shock path
3. take a few small refinement steps on the true recursive blocked-time loss

This is the exact analogue of the GRASP sync step.

It prevents the lifted objective from drifting too far away from the real benchmark objective.

### 6.3 Keep the stop-gradient analogue

When computing the consistency term, use a detached predecessor state in the transition map.

That means:

- the optimizer can still improve the transition through the shock/control channel
- but it cannot win by making both sides of the latent path move together in unrealistic ways

This is likely important for our dense diagnosed-stock bias problem.

---

## 7. Proposed Experiments

### Phase G0: Diagnostics First

#### `EXP-R10-G0-01`
Residual-to-shock audit.

Purpose:

- identify where the current winners show shock-like residual bursts
- separate true shocks from slow plateaus

Artifact:

- residual burst table by quarter and endpoint

Keep/revert:

- diagnostic only

#### `EXP-R10-G0-02`
Plateau census.

Purpose:

- detect long runs of near-zero delta in `D`, `A`, and `F`
- quantify how often the winner is already implicitly plateauing

Artifact:

- plateau duration histogram

Keep/revert:

- diagnostic only

### Phase G1: Minimal GRASP-style lifted observation model

#### `EXP-R10-G1-01`
Lifted states + soft consistency + stock-flow consistency.

No shock factor yet.

Purpose:

- test whether lifted fitting alone helps smooth residual structure without worsening MAE

Keep if:

- exact MAE improves below `0.069471`, or
- exact ties within negligible tolerance while diagnosed and flow raw MAE both improve

#### `EXP-R10-G1-02`
Add plateau penalty.

Purpose:

- explicitly model the flat segments the user asked for

Keep if:

- exact improves or ties while reducing split-to-split residual curvature

#### `EXP-R10-G1-03`
Add sparse shared shock factor.

Purpose:

- allow synchronized quarterly disturbances across `D`, `A`, and `F`

Keep if:

- dense diagnosed raw MAE improves materially
- no suppression honesty regression

#### `EXP-R10-G1-04`
Add periodic sync.

Purpose:

- ensure the lifted solution still respects the real recursive benchmark

Keep if:

- MAE stays at least as good as the best preceding `G1` variant
- lockbox does not deteriorate

### Phase G2: Stochastic exploration

#### `EXP-R10-G2-01`
Shock-noise multi-start.

Purpose:

- add bounded noise to `z_t` only
- compare multiple random seeds

Keep if:

- the best candidate improves and the median candidate is still stable

#### `EXP-R10-G2-02`
Shock budget ablation.

Purpose:

- vary shock sparsity penalty
- check whether improvements are from a few real shocks or from over-flexibility

Keep if:

- the winning solution uses a small number of nonzero shocks

---

## 8. What This Phase Should Not Do

This phase should **not**:

- claim a mechanistic epidemic breakthrough
- promote quarterly explicit `S(t)`
- promote richer leakage
- revive unsupported suppression modeling
- replace the exact or dense benchmark candidates before it clears their gates

This is a structured observation-model phase, not a latent epidemic restart.

---

## 9. Why This Is Publishable If It Works

If the GRASP-style adaptation works, the publishable claim is:

```text
Lifted, provenance-aware stock/flow forecasting with sparse shock and plateau structure
improves mixed-evidence quarterly epidemic forecasting while preserving contract honesty.
```

That is stronger than:

- another micro-tuned `R10` variant
- another failed mechanistic quarterly hazard branch

It is also directly aligned with the current paper direction:

- benchmark contracts
- provenance awareness
- observation-first forecasting
- honest uncertainty and stress testing

---

## 10. Recommended Next Technical Pass

The next actual implementation sequence should be:

1. `EXP-R10-G0-01` residual-to-shock audit
2. `EXP-R10-G0-02` plateau census
3. `EXP-R10-G1-01` lifted state model without shocks
4. `EXP-R10-G1-02` add plateau penalty
5. `EXP-R10-G1-03` add sparse shared shock factor
6. `EXP-R10-G1-04` add periodic sync

This is better than jumping straight to a fully shock-aware model because:

- we still need to separate plateau benefit from shock benefit
- otherwise the model can hide everything inside one flexible disturbance term

---

## 11. Practical Recommendation

Do **not** go back to explicit `S(t)` or richer leakage as the next live benchmark pass.

Do:

- stay on the stable `R10` line
- open a new GRASP-inspired lifted observation phase
- use shocks and plateaus as structured residual objects
- keep the sync-to-rollout step so the lifted fit cannot drift away from the real blocked-time evaluation target

That is the highest-value next phase from this paper.
