# AUTORESEARCH_INCIDENCE.md

## Summary

Create a new parallel plan file at [AUTORESEARCH_INCIDENCE.md](/D:/EpiGraph_PH/AUTORESEARCH_INCIDENCE.md). Do not replace [AUTORESEARCH.md](/D:/EpiGraph_PH/AUTORESEARCH.md) or [AUTORESEARCH_TRANSITIONS.md](/D:/EpiGraph_PH/AUTORESEARCH_TRANSITIONS.md).

This plan is the incidence-before-`U` research track:
- evidence first: audit what can support a pre-`U` layer without target leakage
- identifiability second: separate incidence pressure from diagnosis bottlenecks
- mechanistic third: introduce a latent inflow before `U` while keeping the kept transition branch stable
- decomposition fourth: only after a stable pre-`U` branch exists, decompose incidence pressure into interpretable channels

The scientific goal is narrow:
- do not add a naive `total_population -> U` node
- instead test whether an observed population denominator plus a latent incidence inflow can improve forecasting and interpretation without destabilizing `U_to_D`

The core distinction is:
- population is an observed denominator or exposure constraint
- incidence is the latent inflow entering `U`
- diagnosis remains its own transition bottleneck

The plan must treat evidence in tiers:
- hard stock anchors: full HARP program-count years only
- flow and auxiliary evidence: diagnosis-flow, advanced-HIV, CD4, age, and later downstream service observations
- external validation only: annual new infections, annual AIDS deaths, or any AEM/Spectrum-style quantities that are not independent observations for training

The current kept lineage remains the reference until an incidence branch beats it:
- [AGE-01B](/D:/EpiGraph_PH/artifacts/runs/tr-20260402-s00-AGE-01B-youth-diagnosis-modifier-stable/transition_research/AGE-01B-youth-diagnosis-modifier)
- [PEAK-01F](/D:/EpiGraph_PH/artifacts/runs/tr-20260402-s00-PEAK-01F-region-plus-kp-modifier-gated-fused-forecast/transition_research/PEAK-01F-region-plus-kp-modifier-gated-fused-forecast)
- [DECOMP-01E](/D:/EpiGraph_PH/artifacts/runs/tr-20260402-s00-DECOMP-01E-loo-gated-fused-forecast/transition_research/DECOMP-01E-loo-gated-fused-forecast)

## Numerical Policy

This section is intentionally the same policy as [AUTORESEARCH_TRANSITIONS.md](/D:/EpiGraph_PH/AUTORESEARCH_TRANSITIONS.md). It is binding for every `INC-*` experiment.

- No handwritten numbers anywhere in the analysis or model path.
- Every numeric value introduced by an experiment must be justified by one of:
  - data-estimated quantity
  - semi-Markov dwell-time estimate
  - Bayesian prior/posterior
  - physical or probabilistic constraint
  - numerical-stability guard
- No silent constants for thresholds, penalties, weights, horizons, smoothing strengths, clipping levels, or fallback benchmark values.
- Every non-trivial number must be recorded with provenance in a required artifact: `numeric_justification.json`
- Each entry in `numeric_justification.json` must include:
  - `name`
  - `value`
  - `role`
  - `source_type` (`estimated`, `semi_markov`, `bayesian_prior`, `bayesian_posterior`, `physical_constraint`, `numerical_guard`)
  - `estimation_data`
  - `estimation_method`
  - `uncertainty` or `sensitivity_range`
  - `why_needed`
- Allowed exceptions without estimation:
  - simplex / probability bounds
  - machine epsilon style guards
  - array indexing or calendar mapping constants
- Even allowed exceptions must still be declared in `numeric_justification.json`.
- Any experiment that introduces an undocumented constant automatically fails review.

Default policy for model numbers:
- dwell-like transition timing: estimate with semi-Markov style dwell-time summaries where possible
- regularization / pooling / uncertainty weights: estimate or calibrate through Bayesian or empirical Bayes procedures
- helper-factor shrinkage: use sparse Bayesian shrinkage or equivalent defensible estimator
- fallback to fixed literals is prohibited

## Implementation Changes

- Add a new plan file [AUTORESEARCH_INCIDENCE.md](/D:/EpiGraph_PH/AUTORESEARCH_INCIDENCE.md).
- Add a new experiment folder for source code under:
  - [src/epigraph_ph/phase3/incidence_research/](/D:/EpiGraph_PH/src/epigraph_ph/phase3/incidence_research/)
- Keep the current transition-research branch reusable. The incidence track must be able to load:
  - kept transition forecasts
  - kept state estimates
  - existing age, peak, and decomposition artifacts
- Reuse existing sources rather than inventing new paths:
  - `harp_archive` for observation support
  - `phase15` and `phase2` for mesoscopic factors
  - `transition_research` for baseline comparisons and locked diagnosis paths
- Keep graph topology construction on CPU in v1.
- Prefer GPU only for:
  - repeated incidence-state simulation
  - batched holdout scoring
  - Bayesian posterior sampling or variational inference if used

## Public Interfaces And Artifact Contract

- Run ID format: `inc-<YYYYMMDD>-s<seed>-<experiment_id>`
- Artifact root for every experiment: `artifacts/runs/<run_id>/incidence_research/<experiment_id>/`
- Every experiment must emit:
  - `experiment_spec.json`
  - `decision.json`
  - `coverage_summary.json`
  - `numeric_justification.json`

Common model artifacts:
- `fit_artifact.json`
- `evaluation.json`
- `state_estimates.npz`
- `forecast_states.npz`

Incidence-specific row/types to standardize:
- `incidence_support_row`
- `incidence_driver_split_row`
- `incidence_flow_row`
- `reservoir_mass_row`

## Experiment Registry

**INC-00A-incidence-evidence-audit**
- Audit all evidence that could support a pre-`U` inflow.
- Scope:
  - diagnosis-flow observations
  - annual new infections if present
  - annual AIDS deaths if present
  - any archive indicators that could proxy incident infection rather than diagnosis backlog
  - current `U_to_D` observation support and timing gaps
- Emit:
  - `incidence_evidence_audit.json`
  - `incidence_support_heatmap.png`
  - `incidence_signal_coverage_summary.json`
- Pass if:
  - evidence is explicitly split into direct inflow support vs indirect proxy vs validation-only
  - the audit states whether a latent incidence layer is justified on current evidence
  - target-leaky quantities are flagged rather than promoted

**INC-00B-population-denominator-audit**
- Audit observed population denominators that can be used before `U`.
- Scope:
  - total population
  - adult population when available
  - any observed denominator that is stable enough to normalize incidence pressure
- Rule:
  - do not create a latent population state
  - population is observed input only in this stage
- Emit:
  - `population_denominator_audit.json`
  - `population_alignment_summary.json`
- Pass if:
  - the selected denominator is justified
  - unsupported denominator options are explicitly rejected

**INC-00C-incidence-identifiability-audit**
- Quantify whether the current observation set can separate:
  - higher incidence
  - worse diagnosis
  - changing backlog size
- Emit:
  - `incidence_identifiability_audit.json`
  - `identifiability_stress_table.json`
- Pass if:
  - the audit gives a decision on whether pre-`U` modeling is currently identifiable
  - weakly identified components are listed with the exact missing evidence

**SHOCK-00A-covid-shock-subparameter-audit**
- Audit Phase 0 and Phase 1 normalized subparameters that could explain shock behavior without conflating:
  - true incidence change
  - diagnosis-service disruption
  - backlog release after disruption
  - downstream care disruption
- Scope:
  - typed Phase 1 normalized subparameters only
  - no direct forcing of hazards in this stage
  - use the same numerical policy as every other incidence experiment
- Families:
  - `diagnosis_service_shock`
  - `backlog_release_shock`
  - `incidence_side_shock`
  - `downstream_care_shock`
  - `general_context_only`
- Emit:
  - `shock_subparameter_audit.json`
  - `shock_signal_coverage_summary.json`
  - `shock_factor_family_heatmap.png`
- Pass if:
  - retained shock-relevant rows are typed into one family with evidence provenance
  - family coverage by year is emitted without silently inventing unsupported years
  - the audit gives a decision on whether a shock-aware diagnosis-release split is justified next

**SHOCK-00B-quarterly-shock-factor-surfaces**
- Aggregate the retained `SHOCK-00A` subparameters into quarterly national and region-level factor surfaces.
- Rule:
  - no province expansion yet
  - no direct use in mechanistic fitting yet
- Emit:
  - `shock_factor_surfaces_national.json`
  - `shock_factor_surfaces_region.json`
  - `shock_surface_alignment_summary.json`
- Pass if:
  - surfaces are typed by shock family
  - quarterly alignment is explicit
  - unsupported region-quarter cells are flagged instead of imputed

**INC-01E-covid-shock-diagnosis-release-split**
- Add a shock-aware split around the pre-`U` branch:
  - one latent incidence inflow channel
  - one diagnosis-service shock channel
  - one backlog-release channel into `U_to_D`
- Keep:
  - diagnosis baseline stable unless the shock terms earn non-zero out-of-sample support
- Emit:
  - `shock_split_summary.json`
  - `incidence_flow_summary.json`
  - `evaluation.json`
- Pass if:
  - the post-disruption incompatibility seen in `INC-01B` shrinks
  - the branch does not improve only by destabilizing diagnosis

**INC-01F-shock-gated-inflow-validation**
- Use the retained shock factor families only as gates on when inflow is allowed to move materially.
- Rule:
  - shock factors may validate or gate incidence changes
  - shock factors may not directly overwrite the kept baseline without skill
- Emit:
  - `shock_gated_inflow_summary.json`
  - `annual_incidence_validation.png`
- Pass if:
  - shock-aware gating improves compatibility with validation-only annual incidence
  - unsupported shock families stay inactive

**INC-01A-latent-incidence-inflow-baseline**
- Add the smallest defensible pre-`U` layer:
  - observed population denominator
  - latent incidence inflow `I_t`
  - state update into `U`
- Keep:
  - diagnosis hazard path anchored to the kept transition baseline
  - downstream transition logic inherited from the kept branch
- Emit:
  - `incidence_flow_summary.json`
  - `fit_artifact.json`
  - `evaluation.json`
  - `state_estimates.npz`
- Pass if:
  - overall MAE improves or a front-half bottleneck metric improves without MAE regression
  - `U_to_D` remains stable
  - incidence inflow remains finite and physically plausible

**INC-01B-backlog-vs-incidence-swap-stress-test**
- Stress-test whether the new inflow layer is just relabeling diagnosis backlog.
- Procedure:
  - perturb inflow and diagnosis components under matched observation fit
  - measure how much the solution swaps mass between `I_t` and `U_to_D`
- Emit:
  - `swap_stress_summary.json`
  - `identifiability_margin.json`
- Pass if:
  - the incidence branch shows non-trivial resistance to diagnosis-backlog swapping
  - or explicitly fails with a documented identifiability warning

**INC-01C-incidence-smoothness-family-comparison**
- Compare defensible temporal priors for `I_t`:
  - semi-Markov dwell-style persistence
  - Bayesian random walk
  - shock-aware smooth trend
- Emit:
  - `incidence_prior_comparison.json`
  - `incidence_trajectory_overlay.png`
- Pass if:
  - the winning prior family is chosen on holdout behavior, not in-sample fit
  - every smoothness or persistence number is justified in `numeric_justification.json`

**INC-01D-diagnosis-locked-incidence-branch**
- Repeat `INC-01A` with `U_to_D` explicitly locked to the kept branch.
- Purpose:
  - isolate whether the pre-`U` layer improves the forecast without touching diagnosis
- Emit:
  - `diagnosis_locked_incidence_summary.json`
  - `forecast_vs_locked_baseline.png`
- Pass if:
  - any gain comes from incidence/backlog separation rather than diagnosis distortion

**INC-02A-effective-risk-reservoir-audit**
- Test whether a raw population denominator is too crude and whether a smaller effective at-risk reservoir is needed.
- Scope:
  - observed denominator only
  - no latent reservoir state yet
- Emit:
  - `effective_risk_audit.json`
  - `reservoir_candidate_comparison.json`
- Pass if:
  - one reservoir definition is empirically better justified than raw population
  - unsupported reservoir definitions are explicitly rejected

**INC-02B-observed-reservoir-plus-latent-inflow**
- Replace raw population scaling with the selected observed risk reservoir from `INC-02A`.
- Keep:
  - no latent population state
  - no province split
- Emit:
  - `reservoir_inflow_summary.json`
  - `evaluation.json`
- Pass if:
  - it improves on `INC-01A` or `INC-01D`
  - or it improves interpretability without harming holdout performance

**INC-02C-incidence-vs-diagnosis-driver-split**
- Split mesoscopic factors into two typed driver families:
  - incidence-side drivers affecting inflow before `U`
  - diagnosis-side drivers affecting `U_to_D`
- Rule:
  - no shared undifferentiated factor pool
- Emit:
  - `incidence_diagnosis_driver_split.json`
  - `driver_split_heatmap.png`
- Pass if:
  - every retained factor is typed with evidence provenance
  - no factor is assigned to both sides without explicit justification

**INC-02D-incidence-helper-locked-diagnosis**
- Use only the incidence-side helper family from `INC-02C` while keeping diagnosis-side hazards locked.
- Emit:
  - `incidence_helper_summary.json`
  - `forecast_vs_incidence_baselines.png`
- Pass if:
  - incidence helpers improve the fit without changing diagnosis behavior

**INC-03A-external-validation-only-annual-incidence**
- Use annual new infections and annual AIDS deaths only as validation, not training.
- Emit:
  - `external_validation_summary.json`
  - `annual_incidence_validation.png`
- Pass if:
  - the incidence branch is directionally compatible with external annual validation
  - validation-only quantities are not silently reused as training targets

**INC-03B-shock-aware-incidence-decomposition**
- Decompose the latent incidence inflow into:
  - structural trend
  - disruption shock
  - residual
- Purpose:
  - determine whether shocks belong in incidence pressure or later diagnosis/service transitions
- Emit:
  - `incidence_channel_decomposition.npz`
  - `incidence_channel_summary.json`
- Pass if:
  - decomposition is numerically stable
  - reconstruction error is bounded and reported

**INC-03C-incidence-channel-driver-coupling**
- Couple mesoscopic drivers to incidence channels only after `INC-03B`.
- Rule:
  - no diagnosis driver may enter the incidence channel set unless justified by `INC-02C`
- Emit:
  - `incidence_channel_driver_weights.json`
  - `incidence_channel_driver_heatmap.png`
- Pass if:
  - supported driver-channel pairs are sparse and interpretable
  - unsupported pairs remain explicitly unsupported

**INC-03D-fused-incidence-plus-transition-forecast**
- Fuse the kept incidence branch with the kept transition branch.
- Keep:
  - diagnosis locked unless an earlier incidence experiment proves otherwise
  - unsupported incidence channels on baseline
- Emit:
  - `fused_incidence_transition_forecast.json`
  - `forecast_vs_transition_baselines.png`
- Pass if:
  - it beats the kept non-incidence branch on holdout MAE or peak-aware metrics
  - the gain is not coming from a destabilized `U_to_D`

**INC-04A-region-only-incidence-pressure**
- After a kept national incidence branch exists, allow incidence pressure to vary by region only.
- Forbidden in this stage:
  - province-level incidence modeling
  - KP-conditioned incidence modeling as a co-equal branch
- Emit:
  - `regional_incidence_pressure.json`
  - `regional_incidence_support_summary.json`
- Pass if:
  - region-level incidence variation shows non-zero out-of-sample support
  - unsupported regions are flagged, not imputed

**INC-04B-region-to-national-incidence-gating**
- Use region-level incidence pressure only as a gate or modifier on the national incidence branch.
- Purpose:
  - mirror the successful region-first peak logic without prematurely moving to province level
- Emit:
  - `regional_incidence_gate_summary.json`
  - `forecast_vs_national_incidence.png`
- Pass if:
  - region-only incidence gating changes at least one supported channel and improves held-out behavior

**INC-04C-kp-as-incidence-modifier-only**
- Only after a kept region-only incidence branch exists, reintroduce KP as a modifier on supported incidence channels.
- Rule:
  - KP may not be a co-equal detector source here
  - KP support must remain low-confidence where evidence is sparse
- Emit:
  - `kp_incidence_modifier_summary.json`
  - `kp_incidence_effects.json`
- Pass if:
  - KP adds signal on top of a supported region-only incidence branch
  - otherwise it is rejected and remains deferred

## Keep-Or-Revert Rule

Keep a new incidence experiment only if:
- it improves holdout MAE or peak-aware forecasting without destabilizing `U_to_D`
- it improves interpretability by separating incidence pressure from diagnosis pressure in a way that survives stress tests
- it obeys the numerical policy exactly
- it does not convert validation-only series into training targets

Revert if:
- the new inflow merely relabels diagnosis backlog
- gains appear only in-sample
- `U_to_D` becomes numerically unstable
- the experiment relies on province or KP structure that the current evidence does not support

## Test Plan

- Unit tests for incidence evidence-tier assignment and validation-only labeling
- Unit tests for population denominator loading with no latent population state
- Unit tests for pre-`U` state updates and mass conservation
- Unit tests for swap-stress diagnostics and incidence/diagnosis separation metrics
- Unit tests for incidence prior families with explicit `numeric_justification.json` entries
- Unit tests for incidence driver splitting to ensure factors cannot silently enter both incidence and diagnosis pools
- Unit tests for incidence decomposition reconstruction error
- Integration tests on the kept transition branch:
  - `INC-00*` must run from existing archive and transition artifacts
  - `INC-01*` must keep diagnosis locked unless the experiment explicitly tests otherwise
  - `INC-04*` must stay region-only before any province branch is even considered
- Regression tests to ensure no incidence experiment silently introduces handwritten constants

## Assumptions And Defaults

- This is a separate plan track, not a replacement for the transition-research plan.
- The first incidence experiments are national only.
- Population is observed input, not a latent state, unless a later audit proves otherwise.
- The first purpose of the pre-`U` layer is identifiability, not complexity.
- Region may enter before province, but only after a kept national incidence branch exists.
- KP may enter incidence only as a modifier on already supported region/national incidence structure.
- Existing evidence that direct helper injection destabilizes diagnosis remains binding.
