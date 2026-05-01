# Methods Subsection: Evaluation Contracts And Honesty Flags

**Date:** 2026-04-13  
**Implementation sources:** [tr_v3_experiment_suite.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_experiment_suite.py), [tr_v3_publishability_batch.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_publishability_batch.py)

---

## Methods: Evaluation Contracts, Lockbox, And Suppression Honesty

To avoid mixing heterogeneous evidence without explicit rules, we evaluated Phase 3 candidates under named quarterly contracts. Each contract defines which rows may be used for training, which rows may be used for holdout scoring, and how mixed-evidence quarterly panels are constructed. This contract language is part of the method, not a reporting convenience.

### Exact-only contract

The `exact_only` contract is the primary benchmark for model promotion. Under this contract, the quarterly modeling loop uses only rows tagged `exact_observed` as quarterly observation rows. In practice, this means:

- training uses exact quarterly rows only;
- holdout scoring uses exact quarterly rows only;
- bridge rows and annual rows may still be used for availability summaries or auxiliary diagnostics, but they do not enter the quarterly scoring contract.

This contract is therefore the most conservative quarterly benchmark. It answers the question: how well does the candidate forecast the exact quarterly observations without support from bridge or rule-based quarterly reconstruction?

### Legacy dense contract

The `legacy_dense` contract is a mixed-evidence reference contract used for comparison and lockbox reporting. It begins from a dense quarterly panel constructed across the full archive range. That dense panel may contain rows from multiple provenance tiers, including:

- `exact_observed`
- `bridge_observed`
- `rule_based_extrapolated`

Under the legacy dense contract:

- training may use exact, bridge, and rule-based dense rows;
- holdout scoring is restricted to `exact_observed` and `bridge_observed` rows;
- `rule_based_extrapolated` rows are never scored as truth.

The limitation of the legacy dense contract is that the dense quarterly panel is built once across the full archive window and then reused across rolling splits. That makes it useful as a mixed-evidence reference, but weaker than the purged dense contract as a strict publication benchmark.

### Purged dense contract

The `purged_dense` contract is the hardened mixed-evidence benchmark. It keeps the same scoring rule as the legacy dense contract, but changes how the dense panel is constructed inside each rolling split.

For each split:

- the training portion of the dense panel is rebuilt using only information available up to the end of the training period;
- the rebuilding step is truncated at the training end quarter;
- holdout scoring remains restricted to `exact_observed` and `bridge_observed` rows.

In practical terms, the purged dense contract allows mixed-evidence training while blocking post-training information from leaking backward through dense-panel reconstruction. This is why the purged dense contract is the dense-lane benchmark used for model promotion, while the legacy dense contract is retained only as a reference comparison.

### Retroactive lockbox

In addition to rolling-origin evaluation, we used a fixed holdout lockbox with `2025` as the held-out year. This lockbox was frozen only after the main model development cycle, so it should be described as a **retroactive lockbox** rather than a pristine untouched prospective split.

The retroactive lockbox has three defining rules:

- the holdout year is fixed in advance for the frozen comparison package as `2025`;
- evaluation is performed under `exact_only`, `legacy_dense`, and `purged_dense`;
- each experiment is evaluated using the already-frozen best configuration selected from prior rolling-origin runs, rather than re-optimizing inside the lockbox.

This lockbox therefore serves as a final stability check for promoted candidates. It is useful for showing that the frozen models still beat carry-forward on a held-out year, but it is not the primary model-selection device and should not be described as an untouched external test set.

### Suppression honesty flags

Quarterly viral suppression support is incomplete in the current archive. For that reason, we attached a suppression honesty flag to each evaluated model result rather than treating all suppression forecasts as equally supported.

The suppression honesty flag is assigned per split from two pieces of information:

- whether the training and holdout rows contain scorable `virally_suppressed` support in the allowed scoring tiers;
- how the model handled suppression when direct support was absent.

The flag categories are:

- `scored_direct_support`: holdout suppression is supported directly by scored rows and the training window also contains suppression support.
- `share_carry_from_supported_train`: the holdout split has no direct suppression row to score, but the model carries a suppression share estimated from supported training data.
- `unsupported_level_carry`: no qualifying suppression support is available, and the model still carries a suppression level forward as if it were substantively forecastable.
- `unsupported_share_carry`: no qualifying suppression support is available, and the model carries suppression through an unsupported share-based fallback.
- `unsupported_or_unclaimed`: no qualifying suppression support is available, and the model does not present the suppression output as a supported substantive claim.

These flags are not cosmetic. They are part of the interpretation contract. For example, `EXP-R10-DENSE-H1` was retained as an honesty-improved dense companion because it preserved dense-lane accuracy while replacing unsupported suppression carry claims with `unsupported_or_unclaimed`.

### Recommended manuscript wording

The methods section can summarize the contract system as follows:

> We evaluated quarterly forecasting models under two promotion contracts. The conservative `exact_only` contract trained and scored only on exact quarterly observations. The hardened `purged_dense` contract trained on split-local dense quarterly panels containing exact, bridge, and rule-based rows, but scored holdout quarters only on exact and bridge observations. A `legacy_dense` contract was retained as a reference comparison only. We also used a retroactive `2025` lockbox for final stability checks, evaluating only frozen model configurations rather than re-optimizing within the holdout year. Because quarterly viral suppression support remained incomplete, each result also carried a suppression honesty flag indicating whether suppression was directly supported, carried from supported training data, unsupported but still carried, or explicitly left unclaimed.

### Interpretation rule

The intended interpretation is:

- `exact_only` decides the exact-lane paper candidate;
- `purged_dense` decides the dense-lane paper candidate;
- `legacy_dense` provides reference continuity with earlier experiments;
- the retroactive lockbox checks stability of already-frozen candidates;
- suppression honesty flags prevent unsupported suppression handling from being mistaken for validated epidemiologic signal.

---

## Addendum: Probabilistic Sidecars (2026-04-14)

**Implementation sources:** [tr_v3_probabilistic_batch.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_probabilistic_batch.py), [tr_v3_probabilistic_extension_batch.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_probabilistic_extension_batch.py)

After freezing the deterministic mean models, we evaluated uncertainty sidecars under the same contract structure. These sidecars do not alter the deterministic quarterly mean path. They operate only on residual distributions around the already-frozen mean forecasts.

### Dense-lane uncertainty sidecar

For the `purged_dense` contract, the retained uncertainty sidecar is `EXP-UQ-02`, a tier-aware moving-block residual bootstrap. This sidecar samples quarterly residual blocks from train-only calibration pools defined by endpoint and scoring tier. It was retained because it improved probabilistic performance over conformal intervals and remained better than richer dense extensions involving residual jumps, exogenous jump covariates, shared quarter-level volatility regimes, or plateau-conditioned residual pools.

### Exact-lane uncertainty sidecar

For the `exact_only` contract, the retained uncertainty sidecar is `EXP-UQ-01Z-exact`. The exact lane remained materially under-covered under the plain bootstrap sidecar, so the final exact interval method adds two train-only adjustments per endpoint:

- **median residual recentering**, which shifts the interval center by the train-only median of `target - prediction`;
- **asymmetric lower and upper inflation**, which scales the lower and upper residual quantiles separately rather than applying one symmetric multiplier.

Formally, for endpoint `m` and nominal level `1-\alpha`, the exact interval is:

\[
\hat y_{m,t}^{\,center} = \hat y_{m,t} + \operatorname{median}\{y_{m,\tau} - \hat y_{m,\tau}\}_{\tau \in \mathcal{T}_{train}}
\]

\[
L_{m,t}^{(\alpha)} = \hat y_{m,t}^{\,center} + s_{m}^{lower}\,Q_{\alpha/2}(\varepsilon_{m,\tau}^{centered})
\]

\[
U_{m,t}^{(\alpha)} = \hat y_{m,t}^{\,center} + s_{m}^{upper}\,Q_{1-\alpha/2}(\varepsilon_{m,\tau}^{centered})
\]

where:

\[
\varepsilon_{m,\tau}^{centered} = y_{m,\tau} - \hat y_{m,\tau} - \operatorname{median}\{y_{m,\tau} - \hat y_{m,\tau}\}
\]

and the endpoint-specific `s_m^{lower}` and `s_m^{upper}` values are selected by bounded train-only search to minimize normalized interval score plus a coverage-gap penalty.

### Endpoint-by-tier coverage reporting

For the exact-lane uncertainty sidecar, the frozen package now includes endpoint-by-tier coverage plots. These plots report empirical `80%` and `95%` coverage for:

- `diagnosed_plhiv`
- `alive_on_art`
- `new_diagnosed_cases_period`

across:

- `overall`
- `exact_observed`
- `bridge_observed`

Although the exact contract is scored on exact rows only, the bridge tier is shown in the plot when available so that unsupported or empty cells are explicit rather than hidden.

### Recommended manuscript wording

> Probabilistic evaluation was performed with contract-specific uncertainty sidecars layered on top of frozen deterministic mean forecasts. The dense contract used a tier-aware moving-block residual bootstrap (`EXP-UQ-02`). The exact contract required a stronger repair: a bias-aware asymmetric interval method (`EXP-UQ-01Z-exact`) that recentered each endpoint by the train-only median residual and scaled lower and upper residual quantiles separately. These uncertainty methods did not alter the deterministic mean path; they only transformed the residual distribution used for interval estimation.
