# Phase 3 TR-V3 GRASP Representation Agent

**Date:** 2026-04-14  
**Role:** Representation / Modeling Agent  
**Question:** What can actually be encoded from the current evidence if GRASP-style ideas are adapted into the `R10` line?

---

## 1. GRASP Ingredients That Map Cleanly

These ingredients fit the current repo and evidence contract without pretending we have a new mechanistic epidemic model.

### 1.1 Lifted intermediate observation states

This maps cleanly if the lifted state is the scored quarterly observation vector:

\[
y_t = \begin{bmatrix}
D_t \\
A_t \\
F_t
\end{bmatrix}
\]

where:

- \(D_t\) = `diagnosed_plhiv`
- \(A_t\) = `alive_on_art`
- \(F_t\) = `new_diagnosed_cases_period`

The current code already treats these as the true live forecast heads in [_fit_direct_observation_repair_candidate](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_experiment_suite.py#L2642). So a GRASP-style lifted state can be encoded as a **virtual observation path**, not as hidden epidemic compartments.

### 1.2 Soft consistency instead of hard one-step recursion

This maps cleanly because the repo already has:

- stock-flow reconciliation via `use_joint_consistency`
- train-only flow anchoring via `flow_consistency_weight`
- residual bias correction via `use_bias_correction`

in [_fit_direct_observation_repair_candidate](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_experiment_suite.py#L2642).

So the GRASP-style consistency term should be a **soft penalty between the lifted path and the recursive base path**, not a full replacement of the base dynamics.

### 1.3 Sparse shared shock structure

This maps cleanly as a low-dimensional additive layer on the observation heads:

\[
\tilde y_t = b_t + \Lambda z_t
\]

where:

- \(b_t\) = base `R10` path from supported series fits
- \(z_t \in \mathbb{R}^k\) = small shock/regime state
- \(\Lambda\) = metric loadings

This is justified by the current evidence:

- exact lane is strongest in the `recovery_tail`
- dense lane still has large pre-COVID diagnosed-stock error while ART and flow are good
- global exact-lane `F1` transfer to dense already failed badly in [exp_eval_01_frozen_protocol_rebuild.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-eval-hardening-20260414-s00/analysis/exp_eval_01_frozen_protocol_rebuild.md)

So if a new latent structure is added, it should be **shared residual shock structure**, not a new global flow law.

### 1.4 Plateau / curvature regularization

This maps cleanly as a second-difference penalty on the lifted observation path:

\[
\sum_t \lVert \tilde y_{t+1} - 2\tilde y_t + \tilde y_{t-1} \rVert_1
\]

That is implementable with the current forecast heads and does not require autodiff or a new state engine.

### 1.5 Periodic sync back to the recursive rollout

This maps cleanly if "sync" means:

1. build a train-only lifted adjustment layer on top of the base `R10` forecast
2. project it back into valid recursive forecast rows
3. score only the final projected rows

That is consistent with the current experiment harness, which only scores final `prediction_rows`.

---

## 2. Ingredients That Should Be Rejected Or Simplified

### 2.1 Reject: lifted hidden epidemic states

Do **not** lift `U/D/A/V/L` quarterly latent states inside `R10-G`.

Reason:

- the repo already showed those latent quarterly transitions are weakly identified
- `R1` and related repair families are still far behind
- current evidence supports observation-first structure, not hidden-state freedom

### 2.2 Reject: full GRASP-style free latent optimization

The current code path is NumPy time-series fitting, not a differentiable latent optimizer. A literal GRASP translation would require a new optimization engine and new failure modes.

So "stop-gradient" should be simplified to:

- alternating block updates
- frozen-base plus residual-layer fitting
- train-only lifted adjustments

not full end-to-end latent gradient games.

### 2.3 Reject: direct quarterly `S(t)` inside the live line

This is already blocked by `EXP-S1-A1`.

The current evidence only supports a weak annual denominator sidecar. So `S(t)` should stay:

- annual sidecar only
- not a quarterly live state

### 2.4 Reject: richer live leakage

This is already blocked by `EXP-L2-A1`.

Late suppression shortfall is only an upper-bound proxy. It does not identify:

- `D -> L`
- `A -> L`
- `V -> L`
- mortality-coupled leakage

So leakage can appear only as a diagnostic sidecar, not as a GRASP shock state with biological interpretation.

### 2.5 Simplify: no high-capacity neural block

Do not add a neural sequence model here.

The repo evidence does not justify a high-capacity residual model yet. The right representation is:

- low-rank factor
- sparse shocks
- piecewise regimes
- bounded convex or coordinate-descent fitting

not a generic deep learner.

---

## 3. Minimal `R10-G` Representation Spec

The smallest defensible GRASP-style representation is:

### 3.1 Base path

Start from the current base winner:

- exact lane: `EXP-R10-M1-F1`
- dense lane: `EXP-R10-DENSE-M1-H1`

Let the base train/forecast path be:

\[
b_t =
\begin{bmatrix}
b^D_t \\
b^A_t \\
b^F_t
\end{bmatrix}
\]

generated exactly as the current direct observation candidate does.

### 3.2 Lifted observation state

Define:

\[
\tilde y_t = b_t + \Lambda z_t + r_t
\]

where:

- \(z_t \in \mathbb{R}^k\) is a low-rank shared shock state, with \(k = 1\) or \(2\)
- \(r_t\) is a small metric-specific residual correction
- \(\Lambda\) is a bounded loading matrix

Practical simplification:

- start with \(k=1\)
- allow one shared shock factor only

### 3.3 Constraints

Project every quarter back into valid observed space:

\[
\tilde D_t \ge 0
\]

\[
0 \le \tilde A_t \le \tilde D_t
\]

\[
\tilde F_t \ge 0
\]

Suppression stays sidecar only:

- exact supported: carried share if truly supported
- otherwise: `unsupported_or_unclaimed`

### 3.4 Objective

Fit the lifted layer on train-supported rows only:

\[
\mathcal{L}
=
\lambda_{\mathrm{obs}} \mathcal{L}_{\mathrm{obs}}
+ \lambda_{\mathrm{dyn}} \mathcal{L}_{\mathrm{base}}
+ \lambda_{\mathrm{flow}} \mathcal{L}_{\mathrm{flow}}
+ \lambda_{\mathrm{shock}} \mathcal{L}_{1}(z)
+ \lambda_{\mathrm{tv}} \mathcal{L}_{\mathrm{tv}}(z)
+ \lambda_{\mathrm{curv}} \mathcal{L}_{\mathrm{curv}}(\tilde y)
\]

with:

\[
\mathcal{L}_{\mathrm{obs}}
=
\sum_t \lVert M_t(\tilde y_t - y_t) \rVert^2
\]

\[
\mathcal{L}_{\mathrm{base}}
=
\sum_t \lVert \tilde y_t - b_t \rVert^2
\]

\[
\mathcal{L}_{\mathrm{flow}}
=
\sum_t \left(\tilde F_t - \Psi(\tilde D_t,\tilde D_{t-1})\right)^2
\]

\[
\mathcal{L}_{1}(z) = \sum_t \lVert z_t \rVert_1
\]

\[
\mathcal{L}_{\mathrm{tv}}(z) = \sum_t \lVert z_t - z_{t-1} \rVert_1
\]

\[
\mathcal{L}_{\mathrm{curv}}(\tilde y)
=
\sum_t \lVert \tilde y_{t+1} - 2\tilde y_t + \tilde y_{t-1} \rVert_1
\]

Interpretation:

- `obs`: fit the supported rows
- `base`: do not drift too far from the already-winning `R10` path
- `flow`: preserve exact-lane `F1` logic where it is supported
- `L1 + TV`: few shocks, persistent when present
- `curv`: encourage plateaus instead of jitter

### 3.5 Representation caution

This is an **observation-state model**.

It is not:

- a mechanistic HIV simulator
- a quarterly susceptible model
- a leakage model

That distinction must stay explicit.

---

## 4. Smallest Implementable Experiment Ladder

### Step G1: Exact-lane shared shock factor

**Experiment:** `EXP-R10-G1-EXACT`

Representation:

- start from `EXP-R10-M1-F1`
- add one shared shock factor \(z_t\)
- no metric-specific residuals yet
- fit on exact-supported train rows only

Purpose:

- test whether a sparse shared reporting/service shock improves exact residual structure without broadening the model class too much

### Step G2: Dense-lane bridge-aware shared shock factor

**Experiment:** `EXP-R10-G1-DENSE`

Representation:

- start from `EXP-R10-DENSE-M1-H1`
- same shared shock factor
- bridge rows downweighted relative to exact rows
- no flow-consistency transfer unless gated by tier/era

Purpose:

- address dense diagnosed-stock error without repeating the global `F1` failure

### Step G3: Tier-aware flow gate

**Experiment:** `EXP-R10-G2-DENSE`

Representation:

- same as `G1-DENSE`
- add flow-consistency only on:
  - exact-supported segments, or
  - recovery-tail segments where train support is good

Purpose:

- preserve the dense winner’s flow accuracy while preventing the global `F1` collapse

### Step G4: Regime-aware shock segmentation

**Experiment:** `EXP-R10-G3-CP`

Representation:

- same as `G1/G2`
- replace free quarterly shocks with 1 or 2 train-only regime segments

Purpose:

- compress the shock representation into a more interpretable piecewise form
- create a better paper story if it survives

This is the full ladder. Do not start with anything larger.

---

## 5. Explicit Keep / Revert Gates

### Gate for `EXP-R10-G1-EXACT`

Keep only if all are true:

- exact quarterly mean MAE improves below `0.069471`
- exact raw diagnosed MAE improves below `2097.987`
- exact raw flow MAE does not worsen above `795.799`
- exact lockbox stays at or below `0.045118`
- suppression honesty does not regress

Otherwise revert.

### Gate for `EXP-R10-G1-DENSE`

Keep only if all are true:

- purged dense MAE improves below `0.085418`, or ties within negligible tolerance with materially better diagnosed raw MAE
- diagnosed raw MAE improves below `4111.096`
- flow raw MAE does not worsen materially above `588.591`
- suppression flags remain `unsupported_or_unclaimed`
- dense lockbox remains competitive with `0.063772`

Otherwise revert.

### Gate for `EXP-R10-G2-DENSE`

Keep only if:

- it beats `G1-DENSE` on the dense primary metric, or
- it ties while improving diagnosed raw MAE and not regressing flow MAE

Hard revert if it repeats the `EXP-R10-DENSE-M1-F1-H1` failure pattern:

- dense MAE near `0.166`
- flow raw MAE near or above `1435`

### Gate for `EXP-R10-G3-CP`

Keep only if:

- changepoints are train-stable across rolling splits
- performance improves in at least one primary lane without collapsing the other
- the regime story is interpretable enough to support a paper figure

Revert if changepoints drift unpredictably or only explain one tail split after the fact.

---

## 6. Representation Conclusion

The justified GRASP adaptation is:

- lifted observation states
- sparse shared shocks
- soft stock-flow consistency
- plateau regularization
- train-only regime segmentation

The unjustified GRASP adaptation is:

- lifted latent epidemic states
- quarterly explicit `S(t)`
- richer leakage
- high-capacity neural control blocks

So the correct next representation family is:

\[
\textbf{R10-G = base R10 forecast + low-rank sparse shock layer + soft consistency + projection back to valid observed space}
\]

That is the largest scientifically honest step supported by the repo right now.

