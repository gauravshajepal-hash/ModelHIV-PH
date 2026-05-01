# TR-V3 Failure Analysis

Date: `2026-04-12`

Primary artifacts:
- [fresh archive `s19`](/D:/EpiGraph_PH/artifacts/runs/harp-archive-wdi-standard-20260412-s19/harp_archive/harp_archive_manifest.json)
- [dense panel `s19`](/D:/EpiGraph_PH/artifacts/runs/dense-quarterly-panel-20260412-s19/analysis/dense_quarterly_panel.md)
- [full-history suite `s20`](/D:/EpiGraph_PH/artifacts/runs/tr-v3-experiment-suite-dense-gapaware-l1-20260412-s20/analysis/tr_v3_experiment_suite_report.md)

## Executive Summary

The current `TR-V3-05a/05b` family is failing for structural reasons, not because of a single bad late split.

The strongest evidence is:
- the model is worse on pre-COVID holdouts than on post-2020 holdouts,
- the early failures are dominated by bridge-observed diagnosis flow and ART stock,
- the hazard extrapolator turns many early zero-flow quarters into extreme negative logits and then linearly trends those logits into impossible positive hazards,
- and the hidden state reconstruction is too synthetic in the early bridge years to support a free hazard model.

This means the current family is trying to learn quarter-to-quarter mechanistic dynamics from data that are not quarter-to-quarter mechanistic observations.

In plain English:
- the model is not learning the disease system,
- it is learning artifacts of the reconstructed quarterly panel,
- then amplifying those artifacts with an unstable logit-trend extrapolator.

## First Correction

The earlier `0.884500` figure was only the late-window quarterly mean MAE for `EXP-05a-02` over `2024-2025`.

That number was too narrow for interpretation.

Using the full score-eligible window from the dense contract, the same experiment is:

| Window | Candidate MAE | Carry-forward MAE |
|---|---:|---:|
| `2013-2025` overall | `0.803380` | `0.135885` |
| `2013-2019` pre-COVID | `0.940024` | `0.124777` |
| `2020-2025` COVID/post-COVID | `0.643962` | `0.148845` |

So the model is not only bad on the tail.
It is actually worse in the pre-COVID bridge years.

## Full-History Result Table

Best experiments by full-history quarterly mean MAE from the `s20` run:

| Experiment | Candidate mean MAE | Carry-forward mean MAE | Pre-COVID mean | Post-2020 mean |
|---|---:|---:|---:|---:|
| `EXP-05a-05` | `0.798085` | `0.135885` | `0.900568` | `0.678522` |
| `EXP-05a-06` | `0.798085` | `0.135885` | `0.900568` | `0.678522` |
| `EXP-L1` | `0.798092` | `0.135885` | `0.900569` | `0.678535` |
| `EXP-N1` | `0.799800` | `0.135885` | `0.892671` | `0.691450` |
| `EXP-N2` | `0.799826` | `0.135885` | `0.892671` | `0.691507` |
| `EXP-N3` | `0.799832` | `0.135885` | `0.892684` | `0.691504` |
| `EXP-05a-02` | `0.803380` | `0.135885` | `0.940024` | `0.643962` |

This is enough to reject the idea that the model mainly fails because of the COVID period.

## What Is Actually Being Scored In The Early Years

The early bridge years are not fully observed quarterly state snapshots.

For example, in the dense panel:

| Quarter | Diagnosed tier | ART tier | New diagnoses tier | Scored metrics |
|---|---|---|---|---|
| `2013-Q1` | `bridge_observed` | `rule_based_extrapolated` | `bridge_observed` | diagnosed, diagnosis flow |
| `2013-Q2` | `rule_based_extrapolated` | `bridge_observed` | `bridge_observed` | ART, diagnosis flow |
| `2013-Q3` | `rule_based_extrapolated` | `bridge_observed` | `bridge_observed` | ART, diagnosis flow |
| `2013-Q4` | `rule_based_extrapolated` | `bridge_observed` | `bridge_observed` | ART, diagnosis flow |
| `2014-Q1` | `rule_based_extrapolated` | `bridge_observed` | `bridge_observed` | ART, diagnosis flow |

So the early benchmark is mostly asking:
- can you match bridge-observed ART stock,
- and can you match bridge-observed diagnosis flow,
- while diagnosed stock is often not even scored.

This matters because the model is still fitting a full hidden state system underneath those sparse observations.

## Root Cause 1: State Reconstruction Is Too Synthetic

The quarterly state builder does this:

- `U` = estimated PLHIV minus diagnosed
- `D` = diagnosed but not on ART and not lost
- `A` = on ART but not virally suppressed
- `V` = virally suppressed
- `L` = lost to follow-up

But in the early bridge years:
- `estimated_plhiv` is often annual and then rule-based within-year,
- `virally_suppressed` is mostly rule-based,
- `tested_for_viral_load` is mostly rule-based,
- `L` is effectively forced to zero because the `lost_gap_share` heuristic has too little evidence.

That means the early hidden state is not a strongly observed state.
It is a constructed state.

In plain English:
- the model is treating an imputed hidden state as if it were measured,
- then fitting hazards to transitions between those synthetic states.

That is a classic identifiability failure.

## Root Cause 2: Zero Flows Become Extreme Logits

The hazard fitter uses:

\[
\eta = \operatorname{logit}(h)
\]

with a tiny epsilon floor for zero hazards.

When the observed hazard is zero, this becomes roughly:

\[
\operatorname{logit}(\varepsilon) \approx -15.94
\]

for the current float epsilon.

That is an enormous negative value.

In the early training window for `2013`, the fitted `U \to D` hazard series is:

| Quarter | Train hazard |
|---|---:|
| `2010-Q2` | `0.0000` |
| `2010-Q3` | `0.0000` |
| `2010-Q4` | `0.0285` |
| `2011-Q1` | `0.0000` |
| `2011-Q2` | `0.0378` |
| `2011-Q3` | `0.0440` |
| `2011-Q4` | `0.0439` |
| `2012-Q1` | `0.0497` |
| `2012-Q2` | `0.0557` |
| `2012-Q3` | `0.0486` |
| `2012-Q4` | `0.0479` |

After `logit`, that becomes:

| Quarter | Logit hazard |
|---|---:|
| zero-flow quarters | about `-15.942` |
| nonzero quarters | about `-3.53` to `-2.83` |

This is not a mild transformation.
It creates a huge artificial gap between zero-flow and nonzero-flow quarters.

## Root Cause 3: The AR-Trend Extrapolator Is Unstable

The hazard forecast model is:

\[
\eta_t = \alpha + \beta_{\text{time}} t + \rho \eta_{t-1} + \text{controls}
\]

This is reasonable only if the fitted `\eta_t` series is itself stable.

It is not stable in the early bridge regime.

For the `2013` holdout in `EXP-05a-02`, the fitted `U \to D` coefficients are:

| Parameter | Value |
|---|---:|
| intercept | `-17.9347` |
| time slope | `+1.7324` |
| AR term | `-0.3958` |

That produces forecast hazards:

| Quarter | Forecast `U_to_D` hazard |
|---|---:|
| `2013-Q1` | `0.9093` |
| `2013-Q2` | `0.8746` |
| `2013-Q3` | `0.9785` |
| `2013-Q4` | `0.9919` |

This is mathematically absurd relative to the training series, which lived around `0.03-0.06`.

The same problem appears in `A \to V`.

Training hazards:

| Quarter | Train `A_to_V` hazard |
|---|---:|
| most pre-2012 quarters | `0.0000` |
| `2012-Q2` | `0.0660` |
| `2012-Q3` | `0.0399` |
| `2012-Q4` | `0.0333` |

Forecast hazards:

| Quarter | Forecast `A_to_V` hazard |
|---|---:|
| `2013-Q1` | `0.9988` |
| `2013-Q2` | `0.9998` |
| `2013-Q3` | `1.0000` |
| `2013-Q4` | `1.0000` |

This is the strongest failure signature in the whole analysis.

The model is not gently drifting.
It is numerically exploding.

## Root Cause 4: The Explosion Comes Before Observation Calibration

In the `2013` holdout for `EXP-05a-02`, the reconstructed predicted states already look wrong before readout:

`2013-Q1` predicted state:

| State | Value |
|---|---:|
| `U` | `1710.49` |
| `D` | `24093.60` |
| `A` | `950.77` |
| `V` | `3445.13` |
| `L` | `0.00` |

The model has already moved almost the entire undiagnosed stock into diagnosis in one quarter.

The readout layer does not create this pathology.
It can only distort or shrink it afterward.

## Root Cause 5: Early MAE Is Dominated By Diagnosis-Flow Blowups

For the `2013` holdout, candidate normalized errors are:

| Quarter | Metric | Candidate error |
|---|---|---:|
| `2013-Q1` | diagnosed stock | `1.0037` |
| `2013-Q1` | diagnosis flow | `12.7595` |
| `2013-Q2` | ART stock | `0.9793` |
| `2013-Q2` | diagnosis flow | `0.2082` |
| `2013-Q3` | ART stock | `2.0330` |
| `2013-Q3` | diagnosis flow | `0.8331` |
| `2013-Q4` | ART stock | `2.9429` |
| `2013-Q4` | diagnosis flow | `0.9765` |

Mean candidate MAE for `2013`: `2.7170`

Baseline carry-forward errors for the same split:

| Quarter | Metric | Carry-forward error |
|---|---|---:|
| `2013-Q1` | diagnosed stock | `0.0231` |
| `2013-Q1` | diagnosis flow | `0.1476` |
| `2013-Q2` | ART stock | `0.0132` |
| `2013-Q2` | diagnosis flow | `0.2973` |
| `2013-Q3` | ART stock | `0.0094` |
| `2013-Q3` | diagnosis flow | `0.3491` |
| `2013-Q4` | ART stock | `0.0986` |
| `2013-Q4` | diagnosis flow | `0.3604` |

Mean carry-forward MAE for `2013`: `0.1624`

So the early failure is not spread evenly.
It is mostly:
- diagnosis-flow overshoot,
- plus ART-stock overshoot after the state has already been destabilized.

## Why Carry-Forward Wins

Carry-forward wins for a boring reason:
- it does not try to infer dynamic quarter-to-quarter hazard shifts from sparse bridge-imputed hidden states.

This does not prove carry-forward is mechanistically correct.
It proves the current dynamic model is trying to estimate too much from too little.

In plain English:
- the baseline is conservative,
- the candidate is unstable,
- so the conservative model wins.

## The Deep Diagnosis

The current failure is the combination of four design choices:

1. **Synthetic state reconstruction**
   - early `U/D/A/V/L` states are only weakly anchored

2. **Zero hazard encoding**
   - zero flows become `logit(eps)`, which is extremely negative

3. **Free logit trend extrapolation**
   - the AR-trend model treats those extreme logits as valid regression targets

4. **Quarterly support mismatch**
   - early benchmark support is mostly bridge ART stock and bridge diagnosis flow, not full state observation

This is why the model looks worst in the early bridge years.
The early years contain the most synthetic hidden-state content and the least exact quarter-end support.

## What This Means For Interpretation

The correct interpretation is not:
- "the HIV dynamics changed in some strange pre-COVID way"

The correct interpretation is:
- "the model is numerically unstable when it tries to fit dynamic hazards on sparse bridge-era quarterly reconstructions"

That is a model-design failure.
Not a scientific discovery.

## What Should Change Before The Next Model Run

### 1. Stop treating zero-flow quarters as exact zero hazards

Do not send missing-or-censored zero flows through `logit(eps)`.

They should be handled as one of:
- missing transition evidence,
- censored low-support evidence,
- or weak prior anchors around the last stable hazard.

### 2. Make transition fitting support-aware by transition

Each transition should have its own evidence contract.

Examples:
- `U -> D` should require real diagnosis-flow support
- `A -> V` should not be fit from rule-based suppression shares
- `A -> L` and `L -> A` should stay off unless there is explicit support

### 3. Do not fit `A -> V` on mostly synthetic `V`

In the early bridge years, `V` is largely constructed from imputed suppression share.

That means `A -> V` is not a directly observed quarterly transition.

It should either:
- be fixed to a conservative prior,
- be pooled at annual scale,
- or be excluded entirely from early-period dynamic fitting.

### 4. Replace free logit-trend extrapolation with bounded drift around carry-forward

The current hazard forecast form is too permissive for low-support eras.

The next safe version should be:
- start from carry-forward hazard,
- allow only bounded local deviation,
- and scale that deviation by support quality.

In plain English:
- the model should earn the right to move away from carry-forward,
- not assume it automatically can.

### 5. Separate “bridge reconstruction” from “hazard-learning regime”

The dense quarterly panel is useful for support widening.
It is not automatically valid as a free dynamic training target.

We need a regime flag such as:
- `exact_dynamic_fit`
- `bridge_level_fit_only`
- `diagnosis_flow_only`

so that early bridge years can contribute where they are honest without forcing the full transition engine to pretend they are exact state snapshots.

## Recommended Immediate Experiments

The next experiments should be diagnostic and restrictive:

1. **Zero-hazard censoring ablation**
   - remove `logit(eps)` treatment for zero-flow quarters
   - compare against current early-year explosion

2. **Support-aware transition mask**
   - disable `A -> V` dynamic fitting in years where `V` is mostly rule-based
   - disable `U -> D` dynamic fitting when diagnosis flow is not bridge-or-exact

3. **Bounded-drift hazard model**
   - hazard forecast should be carry-forward plus bounded deviation
   - no free linear logit slope in low-support eras

4. **Metric-local benchmarking**
   - report separate quarterly error for:
     - diagnosis flow
     - ART stock
     - diagnosed stock
   - do not hide the failure inside one mean

## Final Conclusion

The model does not currently fail because HIV is uniquely hard in the COVID era.

It fails because:
- early quarterly state reconstruction is too synthetic,
- zero-flow handling is mathematically toxic,
- and the hazard extrapolator is too free for the support regime.

That is why the pre-COVID bridge years look worst.

The next correct step is not another bigger autoresearch loop.
It is to redesign the transition-fitting contract so that dynamic hazards are only learned where the quarterly evidence can actually identify them.
