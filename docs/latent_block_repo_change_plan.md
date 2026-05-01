# Latent Block Repo Change Plan

## Selected Loop

Selected autoresearch variant: `evidence-to-model-loop`.

Reason:

- The direct measurement layer is sparse.
- Most Phase 0 literature rows are contextual rather than promotable observations.
- The correct first mutation unit is the evidence-to-latent contract, not a new downstream optimizer.

## Scope of This Implementation

This slice implements the foundational contract for a constrained latent-block pipeline:

1. Phase 0 evidence-role and sign-prior artifacts.
2. Registry propagation of latent metadata.
3. Phase 1 observability audit and direct-vs-contextual split.
4. Phase 1.5 national signed weighted latent-block scaffold.
5. Phase 1.5 sparse latent measurement surface and province factor graph scaffold.

This slice does **not** yet implement a full mixed-frequency Bayesian estimator.

## Phase 15 v2 Contract

The repo now also carries an explicit `Phase 15 v2` mathematical contract in:

- `src/epigraph_ph/phase15/v2_spec.py`
- `docs/phase15_v2_math_spec.md`

This v2 contract replaces the current heuristic logic conceptually with:

1. explicit mixed-frequency observation operators
2. bottom-up national aggregation from province latents
3. sign-constrained learned loadings
4. estimated observation and shrinkage precisions
5. dynamic temporal smoothing through hierarchical latent dynamics

The current code still executes the scaffold estimator, but the pipeline now emits a machine-readable `phase15_v2_model_spec.json` artifact so the target estimator is explicit and auditable.

## New / Updated Functions

### Shared latent semantics

File: `src/epigraph_ph/latent_blocks.py`

```python
def latent_block_specs(plugin_id: str) -> list[dict[str, Any]]
def latent_indicator_lookup(plugin_id: str) -> dict[str, dict[str, Any]]
def classify_measurement_role(row: Mapping[str, Any]) -> tuple[str, str]
def infer_candidate_block(canonical_name: str, plugin_id: str) -> tuple[str, str]
def infer_expected_sign(canonical_name: str, plugin_id: str) -> tuple[str, str]
def infer_observation_operator(row: Mapping[str, Any], measurement_role: str) -> str
def annotate_latent_indicator_fields(row: Mapping[str, Any], plugin_id: str) -> dict[str, Any]
```

### Phase 0 evidence artifacts

File: `src/epigraph_ph/phase0/evidence_artifacts.py`

```python
def build_candidate_evidence_rows(*, validated_candidates: list[dict[str, Any]], plugin_id: str) -> list[dict[str, Any]]
def build_literature_context_rows(*, literature_review_payload: dict[str, Any], plugin_id: str) -> list[dict[str, Any]]
def build_block_sign_priors(*, evidence_rows: list[dict[str, Any]], plugin_id: str) -> dict[str, Any]
def build_measurement_manifest(*, evidence_rows: list[dict[str, Any]], plugin_id: str) -> dict[str, Any]
```

### Phase 0 structured numeric collectors

File: `src/epigraph_ph/phase0/structured_numeric_sources.py`

```python
def build_structured_numeric_candidates(*, raw_dir: Path, source_rows: Mapping[str, dict[str, Any]], plugin_id: str) -> dict[str, Any]
```

This collector currently lifts promotable direct indicators from:

- World Bank WDI national annual series for `poverty_rate`, `education`, and `cash_instability`
- Google Community Mobility Reports monthly national and region rollups for `mobility_network_mixing` and `congestion_travel_time`
- local `YAFS5` regional profile PDFs for region-year `education` and `economic_access_constraint`
- PhilHealth Annual Report 2024 for region-year administrative proxies:
  - `health_system_reach` as each region's share of PhilHealth leave-benefit payable across regional offices
  - `policy_implementation_weakness` as each region's casual-share proxy from the same table

Current limitation:

- Region-level structural collectors are still blocked on obtaining machine-readable official tables from PSA or equivalent local survey/report downloads. The repo can now consume those files once placed locally, but it does not yet have a robust online collector for them.

### Phase 1 observability

File: `src/epigraph_ph/phase1/latent_observability.py`

```python
def build_latent_observability_audit(*, normalized_rows: list[dict[str, Any]], parameter_catalog: list[dict[str, Any]], plugin_id: str) -> dict[str, Any]
def build_direct_contextual_split(*, normalized_rows: list[dict[str, Any]], plugin_id: str) -> dict[str, Any]
```

### Phase 1.5 national scaffold

File: `src/epigraph_ph/phase15/national_factor_model.py`

```python
def build_national_measurement_spec(*, normalized_rows: list[dict[str, Any]], observability_audit: dict[str, Any], month_axis: list[str], plugin_id: str) -> dict[str, Any]
def fit_national_factor_model_scaffold(*, standardized_tensor: np.ndarray, axis_catalogs: Mapping[str, list[str]], normalized_rows: list[dict[str, Any]], measurement_spec: dict[str, Any], plugin_id: str) -> dict[str, Any]
```

### Phase 1.5 sparse latent measurements

File: `src/epigraph_ph/phase15/latent_measurements.py`

```python
def build_sparse_indicator_cube(*, normalized_rows: list[dict[str, Any]], province_axis: list[str], month_axis: list[str], canonical_names: list[str], region_labels: list[str] | None = None, include_national_rows: bool = True, observation_weight_floor: float = 0.25) -> dict[str, Any]
```

### Phase 1.5 province factor graph scaffold

File: `src/epigraph_ph/phase15/province_factor_graph.py`

```python
def build_province_factor_graph_scaffold(*, standardized_tensor: np.ndarray, axis_catalogs: Mapping[str, list[str]], normalized_rows: list[dict[str, Any]], national_scaffold: Mapping[str, Any], region_labels: list[str], plugin_id: str) -> dict[str, Any]
```

## Artifact Schemas

### Phase 0

`phase0/extracted/evidence_indicator_rows.json`

Each row contains:

- `evidence_indicator_id`
- `source_stage`
- `source_bank`
- `source_id`
- `candidate_id`
- `canonical_name`
- `candidate_block`
- `candidate_block_display_name`
- `block_source`
- `expected_sign`
- `sign_source`
- `measurement_role`
- `role_reason`
- `observation_operator`
- `geo_resolution`
- `time_resolution`
- `is_numeric`
- `is_direct_measurement`
- `is_anchor_eligible`
- `confidence`
- `evidence_weight_hint`
- `literature_ref_count`
- `literature_basis`
- `top_document_titles`
- optional literature-context fields:
  - `supporting_silo_id`
  - `promotion_track`
  - `structured_adapter_ids`
  - `query_examples`

`phase0/extracted/block_sign_priors.json`

- `plugin_id`
- `row_count`
- `block_count`
- `by_block`
- `rows`

Each row contains:

- `block_id`
- `canonical_name`
- `expected_sign`
- `evidence_row_count`
- `direct_indicator_count`
- `proxy_indicator_count`
- `context_only_count`
- `literature_ref_count_total`
- `literature_basis`
- `supporting_silos`
- `top_document_titles`

`phase0/extracted/measurement_manifest.json`

- `plugin_id`
- `row_count`
- `measurement_role_counts`
- `candidate_block_counts`
- `expected_sign_counts`
- `observation_operator_counts`
- `source_stage_counts`
- `blocks`

`phase0/extracted/structured_numeric_candidate_summary.json`

- `source_bank`
- `candidate_count`
- `numeric_observation_count`
- `cache_dir`
- `collectors`

Each collector row contains:

- `collector`
- optional `canonical_name`
- `status`
- `row_count`
- `cache_used`
- optional `error`

Collector notes:

- the Google mobility cache now stores both national and region rows, preserving `geo`, `region`, `province`, `time`, and `value`
- the local `YAFS5` collector reads profile PDFs from `docs/Pdf/` and emits region-year structural indicators for all available regional profiles
- the PhilHealth collector reuses a cached local `ar2024.pdf` when available, otherwise fetches the official report once into the Phase 0 structured cache and extracts `2023` and `2024` region tables from the leave-benefit note

### Phase 1

`phase1/latent_observability_audit.json`

- `plugin_id`
- `row_count`
- `summary`
- `rows`

Each row contains:

- `canonical_name`
- `candidate_block`
- `expected_sign`
- `row_count`
- `numeric_row_count`
- `direct_indicator_count`
- `proxy_indicator_count`
- `context_only_count`
- `anchor_count`
- `national_support_count`
- `regional_support_count`
- `province_support_count`
- `monthly_support_count`
- `annual_support_count`
- `source_bank_count`
- `literature_basis`
- `eligible_for_national_likelihood`
- `eligible_for_province_graph`
- `parameter_catalog_row_count`

`phase1/direct_vs_contextual_split.json`

- `plugin_id`
- `summary`
- `rows`
- `blocks`

`summary` contains row and canonical counts by `direct_indicator`, `proxy_indicator`, and `context_only`.

### Phase 1.5

`phase15/national_block_measurement_spec.json`

- `method`
- `plugin_id`
- `source_row_count`
- `month_axis`
- `retained_block_count`
- `dropped_block_count`
- `retained_blocks`
- `dropped_blocks`

Each retained or dropped block contains:

- `block_id`
- `display_name`
- `description`
- `minimum_direct_indicators`
- `minimum_indicator_count`
- `literature_basis`
- `indicator_rows`
- `direct_indicator_count`
- `eligible_indicator_count`

Each indicator row contains:

- `canonical_name`
- `expected_sign`
- `direct_indicator_count`
- `proxy_indicator_count`
- `context_only_count`
- `national_support_count`
- `regional_support_count`
- `province_support_count`
- `monthly_support_count`
- `annual_support_count`
- `eligible_for_national_likelihood`
- `literature_basis`
- `loading_weight_hint`

`phase15/national_block_loadings.json`

- `method`
- `rows`

Each row contains:

- `block_id`
- `display_name`
- `canonical_name`
- `expected_sign`
- `loading`
- `direct_indicator_count`
- `proxy_indicator_count`
- `literature_basis`

`phase15/national_block_states.json`

- `method`
- `month_axis`
- `rows`

Each row contains:

- `block_id`
- `display_name`
- `state_values`

`phase15/national_block_ppc.json`

- `method`
- `rows`

Each row contains:

- `block_id`
- `canonical_name`
- `expected_sign`
- `correlation_with_state`
- `mean_absolute_error`

`phase15/national_block_identification_report.json`

- `method`
- `plugin_id`
- `is_scaffold`
- `national_geo_label`
- `month_axis`
- `retained_block_count`
- `requested_block_count`
- `retained_blocks`
- `dropped_blocks`
- `notes`

`phase15/province_block_states.json`

- `method`
- `province_axis`
- `month_axis`
- `block_axis`
- `rows`

Each row contains:

- `block_id`
- `display_name`
- `province`
- `region`
- `state_values`

`phase15/province_block_uncertainty.json`

- `method`
- `province_axis`
- `month_axis`
- `block_axis`
- `rows`

Each row contains:

- `block_id`
- `display_name`
- `province`
- `region`
- `posterior_std_values`
- `local_precision_values`
- `region_precision_values`
- `local_support_mass_values`
- `regional_support_mass_values`
- `observed_indicator_count_values`
- `regional_indicator_count_values`

`phase15/province_loading_deviations.json`

- `method`
- `rows`

Each row contains:

- `block_id`
- `display_name`
- `province`
- `region`
- `canonical_name`
- `national_loading`
- `effective_loading`
- `loading_deviation`
- `scale_multiplier`
- `observed_month_count`
- `support_mass`
- `correlation_with_block_state`

`phase15/province_block_state_tensor.npz`

- axis names: `["province", "month", "latent_block"]`

`phase15/region_block_state_tensor.npz`

- axis names: `["region", "month", "latent_block"]`

## Plugin Configuration Contract

Added under `plugins/hiv.py -> HIV_CONSTRAINT_SETTINGS["phase15"]["latent_blocks"]`:

- `enabled`
- `minimum_retained_blocks`
- `province_factor_graph`
- `blocks`

`province_factor_graph` contains:

- `enabled`
- `national_precision`
- `region_precision`
- `loading_prior_precision`
- `loading_scale_min`
- `loading_scale_max`
- `observation_weight_floor`
- `posterior_precision_eps`

Each block contains:

- `block_id`
- `display_name`
- `description`
- `minimum_direct_indicators`
- `minimum_indicator_count`
- `literature_basis`
- `indicators`

Each indicator contains:

- `expected_sign`
- optional `literature_basis`

## Literature Integration Rule

The implementation uses literature in two places:

1. Local block and sign rationale in plugin config.
2. Phase 0 literature review outputs as contextual prior evidence written into `evidence_indicator_rows.json`.

Literature-review aggregates remain `context_only`. They do not become fake numeric observations.

## Test Cases Added / Updated

### Phase 0 tests

- confirm `evidence_indicator_rows.json`, `block_sign_priors.json`, and `measurement_manifest.json` exist
- confirm evidence rows include `measurement_role`, `candidate_block`, and `expected_sign`
- confirm measurement manifest reports nonzero role counts after a build

### Registry tests

- confirm registry rows include:
  - `candidate_block`
  - `expected_sign`
  - `measurement_role`
  - `observation_operator`
- confirm `measurement_role` is one of `direct_indicator`, `proxy_indicator`, `context_only`

### Phase 1 tests

- confirm normalized rows include latent fields
- confirm `latent_observability_audit.json` and `direct_vs_contextual_split.json` exist
- confirm audit rows report boolean eligibility flags and coherent role counts

### Phase 1.5 tests

- confirm plugin latent block config exists
- confirm national scaffold artifacts exist
- confirm national block states align with the month axis length
- confirm identification report declares scaffold mode
- confirm province factor graph artifacts exist
- confirm province latent tensor shape matches province and month axes
- confirm province loading deviations report observed support where local evidence exists

## Next Implementation Step

After this slice stabilizes:

1. replace the signed weighted national scaffold with a mixed-frequency constrained factor estimator
2. replace the current scaffold graph with a richer hierarchical province estimator that separates region-level and province-level uncertainty more cleanly
3. score the latent path against frozen backtests and long-horizon peak behavior

## Phase 15 v2 Spec Layer

The repo now also carries a `Phase 15 v2` mathematical-spec layer implemented in:

- `src/epigraph_ph/phase15/v2_spec.py`
- `phase15/phase15_v2_model_spec.json`
- `phase15/phase15_v2_observation_support.json`
- `phase15/phase15_v2_mathematical_spec.md`

This layer does not yet fit the full estimator. It formalizes the scientifically serious replacement target:

1. sparse mixed-frequency observation operators on the province-month grid
2. bottom-up region and national aggregation from province latent states
3. sign-constrained learned loadings
4. estimated observation and shrinkage precisions
5. temporal smoothing over the full latent path rather than annual fan-out
