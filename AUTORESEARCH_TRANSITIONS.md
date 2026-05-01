# AUTORESEARCH_TRANSITIONS.md

## Summary

Create a new parallel plan file at [AUTORESEARCH_TRANSITIONS.md](/D:/EpiGraph_PH/AUTORESEARCH_TRANSITIONS.md). Do not replace [AUTORESEARCH.md](/D:/EpiGraph_PH/AUTORESEARCH.md).

This new plan is the transition-centric research track:
- analytics first: yearly mesoscopic signal evolution and KP/transition relevance
- mechanistic second: national `U/D/A/V/L` predictive engine
- decomposition third: HIV-specific transition-hazard decomposition coupled to the mechanistic engine

The plan must treat data in tiers, not as one homogeneous panel:
- hard stock anchors: full HARP program-count years only
- flow and auxiliary evidence: diagnosis-flow, advanced-HIV, CD4, and subgroup evidence from all available years
- no experiment may silently treat sparse early years as equivalent to fully observed later stock-anchor years

Use user-facing terminology `Phase 1.5`, but keep internal Python package/module names as `phase15`. Add an external alias; do not rename imports.

## Numerical Policy

Add a non-negotiable rule to the plan:

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

- Add a new plan file [AUTORESEARCH_TRANSITIONS.md](/D:/EpiGraph_PH/AUTORESEARCH_TRANSITIONS.md) with its own registry, run-ID convention, artifact contract, and pass/fail gates.
- Add a CLI alias `phase1_5 build` that dispatches to the existing `phase15 build` implementation. Keep artifact directories on disk under `phase15/`.
- Add a new Phase 3 package folder for this line:
  - [src/epigraph_ph/phase3/transition_research/](/D:/EpiGraph_PH/src/epigraph_ph/phase3/transition_research/)
- Reuse existing sources rather than inventing new data paths:
  - `phase15` multiscale tensors and factor catalogs
  - `phase2` retained predictive/context factor sets and multiscale blankets
  - `phase3` subgroup/KP machinery in [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py)
  - `temporal_scaffold` in [temporal_scaffold.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/temporal_scaffold.py)
- Keep graph topology construction on CPU in v1. Only move batched yearly scoring, transition simulation, and decomposition fitting to GPU when available. Do not port NOTears/topology logic first.

## Public Interfaces And Artifact Contract

- Run ID format: `tr-<YYYYMMDD>-s<seed>-<experiment_id>`
- Artifact root for every experiment: `artifacts/runs/<run_id>/transition_research/<experiment_id>/`
- Every experiment must emit:
  - `experiment_spec.json`
  - `decision.json`
  - `coverage_summary.json`
  - `numeric_justification.json`

Analytics row types to standardize:
- `yearly_window_score_row`
- `regional_window_score_row`
- `factor_transition_relevance_row`
- `factor_kp_relevance_row`

Model row/types to standardize:
- `transition_engine_spec`
- `transition_hazard_row`
- `state_mass_row`

Common model artifacts:
- `fit_artifact.json`
- `evaluation.json`
- `state_estimates.npz`
- `forecast_states.npz`

## Experiment Registry

**AN-01A-national-yearly-factor-evolution**
- Build yearly national windowed importance scores for all retained Phase `1.5`/2 mesoscopic factors across `2010-2025`.
- Score by year using only evidence available inside that year; no future leakage.
- Emit `yearly_factor_scores_national.json`, `national_factor_heatmap.png`, `national_factor_trajectories.png`, `ranked_yearly_tables.json`.
- Pass if every year `2010-2025` is present with explicit coverage/confidence and scores are not copied from global survival rank.

**AN-01B-regional-yearly-factor-evolution**
- Repeat `AN-01A` at region level using `phase15` regional multiscale tensors.
- Emit `yearly_factor_scores_region.json`, `regional_factor_heatmap.png`, `regional_factor_trajectories.png`.
- Pass if every region in `multiscale_factor_axes.region` is scored for every supported year and unsupported year-region cells are marked low-confidence rather than imputed silently.

**AN-01C-factor-importance-drift-report**
- Produce a drift summary showing which factors rise, fall, or switch targets over time nationally and regionally.
- Emit `factor_drift_report.json` and `factor_drift_summary.png`.
- Pass if the report contains top risers, top decliners, and stability bands for each target family.

**AN-02A-factor-to-transition-map**
- Map retained mesoscopic factors to `U_to_D`, `D_to_A`, `A_to_V`, `A_to_L`, and `L_to_A`.
- Use current transition hooks, empirical transition sensitivity, and target alignment. Do not infer from names alone.
- Emit `factor_transition_relevance.json` and `factor_transition_heatmap.png`.
- Pass if every retained factor has a relevance vector and confidence score, with at least one non-zero transition assignment justified by hooks or empirical signal.

**AN-02B-factor-to-kp-map**
- Map retained mesoscopic factors to KP relevance for `MSM`, `TGW`, `other`, using subgroup priors, subgroup-feature interactions, anchor pack evidence, and network-signal adjustments already in Phase 3.
- Emit `factor_kp_relevance.json` and `factor_kp_heatmap.png`.
- Pass if all KP relevance rows include evidence provenance and low-confidence flags where subgroup anchors are sparse.

**AN-02C-kp-transition-relevance-drift**
- Combine `AN-02A` and `AN-02B` into yearly KP-to-transition relevance maps.
- Emit `kp_transition_relevance.json` and `kp_transition_drift.png`.
- Pass if the output shows, by year, which factors matter most for each KP-transition pair and clearly marks unsupported cells.

**AGE-01A-age-evidence-audit**
- Audit all age-related evidence available to the transition-research track before introducing age-conditioned forecasting logic.
- Scope:
  - OCR archive age signals such as `youth_cases_15_24_period`, `art_median_age`, and any age-language extracted from surveillance summaries
  - existing age priors and age distributions already present in [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py)
  - any age-coded subgroup summaries already recoverable from prior Phase 3 artifacts
- Required rule:
  - do not silently create age bins from convenience
  - every age grouping used downstream must be justified by this audit
- Initial target grouping:
  - `15-24`
  - `25+`
- Deferred unless explicit evidence coverage is sufficient:
  - `0-14`
- Emit:
  - `age_evidence_audit.json`
  - `age_evidence_heatmap.png`
  - `age_signal_coverage_summary.json`
- Pass if:
  - age-support coverage is quantified by year for `2010-2025`
  - age evidence is split into direct observation vs auxiliary proxy vs prior-only support
  - the audit explicitly recommends either `15-24 vs 25+` or a narrower fallback and records why

**AGE-01B-youth-diagnosis-modifier**
- Add a youth modifier only on `U_to_D`, using the age grouping justified by `AGE-01A`.
- This is a modifier experiment, not a state-space explosion.
- Baseline reference:
  - the kept `PEAK-01F` branch
- Allowed evidence:
  - direct youth surveillance signal from the OCR archive
  - age priors already present in the old Phase 3 stack
  - mesoscopic helpers only if assigned to `U_to_D`
- Emit:
  - `age_transition_modifier_summary.json`
  - `age_diagnosis_modifier_effects.json`
  - `forecast_vs_peak01f.png`
- Pass if:
  - normalized MAE improves versus `PEAK-01F` or peak error improves without MAE regression
  - diagnosis-flow MAE does not worsen
  - the modifier is active on `U_to_D` and explicitly inactive on all other transitions

**AGE-01C-youth-downstream-modifier**
- Extend age modulation from `U_to_D` into downstream transitions only after `AGE-01B`.
- Allowed downstream transitions:
  - `D_to_A`
  - `A_to_V`
- Forbidden in this stage:
  - direct age modulation on `A_to_L` or `L_to_A` unless `AGE-01A` shows direct evidence rather than only proxy support
- Auxiliary evidence may include:
  - `art_median_age`
  - age-sensitive service proxies already present in the archive
- Emit:
  - `age_downstream_modifier_summary.json`
  - `age_transition_effect_heatmap.png`
  - `forecast_vs_age01b.png`
- Pass if:
  - it beats `AGE-01B` on MAE or peak error
  - it does not degrade diagnosis-flow MAE
  - any active downstream age effect is supported by explicit audit evidence, not only by unrestricted fitting

**AGE-02A-two-band-mechanistic-split**
- Promote age from modifier-only status into a two-band national mechanistic split only if `AGE-01B` or `AGE-01C` is kept.
- Initial split:
  - `15-24`
  - `25+`
- Do not introduce `0-14` here unless `AGE-01A` shows direct age evidence and enough support to avoid a prior-only pediatric branch.
- States remain national and transition-centric, but duplicated across the two age bands:
  - `U`
  - `D`
  - `A`
  - `V`
  - `L`
- Transition set remains:
  - `U_to_D`
  - `D_to_A`
  - `A_to_V`
  - `A_to_L`
  - `L_to_A`
- Emit:
  - `age_band_state_estimates.npz`
  - `age_band_transition_hazards.json`
  - `age_band_forecast.json`
  - `forecast_vs_age01c.png`
- Pass if:
  - the two-band model improves on the kept age-modifier baseline
  - the younger band carries non-trivial signal rather than collapsing into the prior
  - mass is conserved within and across age bands
  - any cross-band coupling numbers are justified in `numeric_justification.json`

**MECH-01A-national-udavl-baseline**
- Build the first national mechanistic engine with states `U`, `D`, `A`, `V`, `L`.
- Predict the five transitions directly: `U_to_D`, `D_to_A`, `A_to_V`, `A_to_L`, `L_to_A`.
- Use mesoscopic factors disabled in this baseline; only scaffold, stock anchors, and flow/aux observations drive the fit.
- Emit `fit_artifact.json`, `evaluation.json`, `transition_hazard_summary.json`, `state_estimates.npz`, `forecast_states.npz`.
- Pass if front-half performance is no worse than the current national-reset winner by more than `5%`, and all transition hazards are finite and identifiable.

**MECH-01B-mesoscopic-transition-helpers**
- Add mesoscopic helper covariates to the transition engine with explicit per-transition assignments from `AN-02A`.
- Covariates may only enter the transitions they are assigned to.
- Emit `mesoscopic_transition_helper_summary.json`.
- Pass if holdout MAE improves versus `MECH-01A` and diagnosis-flow MAE does not regress.
- Fail immediately if ungated mesoscopic helpers worsen the holdout, repeating the current `phase3_frozen_backtest_ungated_mesoscopic` outcome.

**KP-01A-national-kp-lite-overlay**
- Add a national KP overlay on top of `MECH-01B`.
- KP enters as hazard modulation on the five transitions, not as a full province x KP x age x sex x state explosion.
- Reuse existing subgroup/KP machinery from [rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/rescue_core.py), but collapse to national KP weights first.
- Emit `kp_overlay_summary.json` and `kp_transition_hazards.json`.
- Pass if KP modulation improves fit or interpretability on diagnosis/linkage transitions without degrading aggregate holdout MAE by more than `2%`.

**DECOMP-01A-transition-channel-decomposition**
- Decompose each of the five transition hazards into HIV-specific temporal channels:
  - structural trend
  - service/disruption shock
  - short-horizon residual
- Use [temporal_scaffold.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/temporal_scaffold.py) as initializer, but decompose hazards, not raw epidemic outputs.
- Emit `channel_decomposition.npz` and `channel_summary.json`.
- Pass if each transition gets a valid three-channel decomposition and reconstruction error is bounded and reported.

**DECOMP-01B-channel-driver-coupling**
- Assign mesoscopic drivers to each transition-channel pair.
- Each channel gets its own sparse driver matrix; no shared undifferentiated covariate pool.
- Emit `channel_driver_weights.json` and `channel_driver_heatmap.png`.
- Pass if every transition-channel pair either has sparse drivers with confidence or is explicitly marked unsupported.

**DECOMP-01C-fused-mechanistic-forecast**
- Fuse the decomposed transition channels back into the mechanistic `U/D/A/V/L` simulator to produce forecasts.
- Emit `hazard_reconstruction.json`, `mechanistic_forecast.json`, and `forecast_vs_baselines.png`.
- Pass if it beats `MECH-01B` on holdout MAE or diagnosis-flow MAE without introducing implausible hazard spikes.
- Fail if decomposition only improves in-sample fit.

## Test Plan

- Unit tests for yearly window construction, year-level coverage flags, and no-leakage enforcement.
- Unit tests for region-level aggregation using `phase15` multiscale region tensors.
- Unit tests for factor-to-transition mapping to ensure only allowed transitions receive each factor.
- Unit tests for KP relevance mapping to ensure low-confidence is emitted when subgroup anchor support is missing.
- Unit tests for `U/D/A/V/L` transition simulator mass conservation and valid hazard bounds.
- Unit tests for decomposition reconstruction error and channel sparsity.
- Integration tests on `audit-phase0-reuse-s00-20260331`:
  - `AN-01*` must emit yearly outputs for `2010-2025`
  - `AN-02*` must emit transition and KP relevance maps for the retained factor set
  - `AGE-01*` must use age evidence tiers explicitly and must not invent unsupported age bands
  - `AGE-02A` must run only after an age-modifier branch is kept
  - `MECH-01A/B` and `KP-01A` must run on train `2010-2020`, holdout `2021-2025`, while using typed evidence tiers
  - `DECOMP-01*` must run at national level only
- Regression tests to ensure the scaffold gate does not silently delete all helpers in `MECH-01B`; the experiment must record both pre-gate and post-gate factor counts.
- Regression tests to ensure age branches do not silently expand from `15-24 vs 25+` into additional bands without an `AGE-01A` evidence decision artifact.
- Performance profiling test:
  - confirm graph construction remains CPU
  - confirm batched yearly scoring and transition simulation can use GPU when available
  - fail only if GPU code path changes numerical results beyond tolerance

## Assumptions And Defaults

- This is a separate plan file, not a replacement for the existing reset plan.
- User-facing naming is `Phase 1.5`; internal package/module/disk path stays `phase15`.
- "Use all data" means typed use of all available evidence, not treating all years as equally observed stock-anchor years.
- Early years may contribute diagnosis-flow and auxiliary evidence even when they do not contribute full stock-anchor loss.
- The first mechanistic model is national only.
- KP enters first as a national hazard overlay, not as a full high-dimensional latent grid.
- Age enters before province, but first as a narrow national modifier or two-band national split, not as a province x age expansion.
- GPU is used first for batched scoring and transition simulation, not for graph topology construction.
- Existing `phase3_frozen_backtest_ungated_mesoscopic` evidence is binding: mesoscopic helpers must be treated as helper controls, not baseline drivers, unless a later experiment proves otherwise.
