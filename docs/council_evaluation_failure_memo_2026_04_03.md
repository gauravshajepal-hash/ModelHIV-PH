# Evaluation / Failure Memo: Phases 0, 1, 15, and 2

Date: 2026-04-03  
Repo: `D:\EpiGraph_PH`  
Run inspected: `smoke-latent-blocks`

## Scope

This memo audits what is actually validated in Phases 0, 1, 15, and 2, what can still fail silently, what would count as a false win, and what benchmark or audit structure should gate future changes.

The emphasis here is evaluation, not architecture.

## Files And Artifacts Inspected

Code and tests:

- `D:\EpiGraph_PH\tests\test_phase0_pipeline_pytest.py`
- `D:\EpiGraph_PH\tests\test_phase1_pipeline_pytest.py`
- `D:\EpiGraph_PH\tests\test_phase15_pipeline_pytest.py`
- `D:\EpiGraph_PH\tests\test_phase15_v2_engine_pytest.py`
- `D:\EpiGraph_PH\tests\test_phase2_pipeline_pytest.py`
- `D:\EpiGraph_PH\tests\test_phase2_latent_temporal_graph_pytest.py`
- `D:\EpiGraph_PH\tests\test_extraction_quality_audit_pytest.py`

Run manifests and QC outputs:

- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase0\phase0_manifest.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase1\phase1_manifest.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase15\phase15_manifest.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase2\phase2_manifest.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\analysis\extraction_quality_audit.md`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\analysis\harp_source_adjudication_table.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase15\phase15_v2_fit_summary.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase2\latent_temporal_graph_bundle.json`

## What Is Actually Validated Today

There is meaningful validation, but it is concentrated on contracts, invariants, and deterministic parity rather than scientific correctness.

### Phase 0

Validated:

- Structured extraction adapters have deterministic parity against cached or local sources, according to `extraction_quality_audit.md`.
- HARP annual panel selection now respects official-seed precedence and emits explicit adjudication categories, according to `harp_source_adjudication_table.json`.
- Candidate-shape and semantic helper tests are fairly extensive in `test_phase0_pipeline_pytest.py`.

Not yet validated:

- Absolute correctness of free-text extraction. The audit explicitly states there is no literal gold standard yet for unstructured literature extraction.
- End-to-end semantic precision of canonicalization, geography binding, and time binding for narrative or OCR-derived rows.

### Phase 1

Validated:

- Normalization helper behavior, reliability classes, and tensor contracts are tested in `test_phase1_pipeline_pytest.py`.
- The run emits `latent_observability_audit.json`, `direct_vs_contextual_split.json`, standardized tensors, denominator tensors, missing masks, and quality weights, and these are shape-checked.

Not yet validated:

- Whether the evidence weights and bias penalties are empirically calibrated.
- Whether observability eligibility corresponds to real identifiability rather than just artifact presence and counts.

### Phase 15

Validated:

- The old scaffold path is artifact-complete and contract-tested in `test_phase15_pipeline_pytest.py`.
- The v2 engine has targeted synthetic tests for sign-constrained loadings, bottom-up national aggregation, burden-weight learning, and unidentified-weight fallback in `test_phase15_v2_engine_pytest.py`.
- The live run shows the v2 engine activated on `2379` measurement rows with `4` retained blocks in `phase15_v2_fit_summary.json`.

Not yet validated:

- Calibration of posterior uncertainty.
- Recovery under realistic mixed-frequency sparse observation regimes.
- Sensitivity to wrong sign priors, wrong burden supervision, or weakly identified blocks.

### Phase 2

Validated:

- The latent temporal graph has a synthetic recovery test for a simple lagged chain plus hidden shared structure in `test_phase2_latent_temporal_graph_pytest.py`.
- The pipeline emits latent temporal graph artifacts and multiscale blankets and checks contract presence in `test_phase2_pipeline_pytest.py`.
- The live run produces completed province, region, and national latent temporal graph bundles in `latent_temporal_graph_bundle.json`.

Not yet validated:

- Recovery quality of the sparse-plus-low-rank split under realistic Phase 15 state noise and extraction perturbations.
- Whether direct sparse edges remain stable when upstream latent states shift modestly.
- Whether hidden-driver rows represent real latent confounding or decomposition slack.

## 1. Strongest Evaluation Gaps

### A. No gold-standard benchmark for unstructured Phase 0 extraction

This is the largest single gap.

The extraction audit says the structured-source layer passes parity, but it also states:

`Unstructured literature extraction does not have a literal gold standard in this repo yet.`

That means the repo can prove:

- internal consistency
- deterministic replay
- conflict surfacing

but it still cannot prove:

- correct canonical name
- correct geography
- correct time
- correct measurement role

for free-text or OCR-heavy rows.

### B. Manifests are not trustworthy as evaluation gates

The live Phase 0 manifest reports:

- `canonical_candidate_count = 0`
- `numeric_observation_count = 0`

in `phase0_manifest.json`, even though downstream Phase 1 reports `canonical_candidate_count = 11708`, Phase 1 has a populated tensor build, and `extraction_quality_audit.md` reports `11354` structured rows exactly matched on regeneration.

This is a serious evaluation problem. A stage summary can claim completion while its own counts are not faithful to what was actually emitted.

### C. Most tests validate contracts and invariants, not scientific truth

Across Phases 1, 15, and 2, the tests strongly validate:

- shape
- finiteness
- presence of artifacts
- simple synthetic recovery
- whitelist and threshold behavior

They do not yet strongly validate:

- identifiability
- calibration
- robustness to adversarial or realistic misspecification
- correctness of scientific interpretation

### D. Phase 15 v2 remains under-calibrated even though it is structurally improved

The v2 engine is materially better than the scaffold. But the current live fit still shows:

- all retained blocks at `regional_precision = 25.0`

in `phase15_v2_fit_summary.json`.

That ceiling-hitting pattern is a warning sign. It may mean the regional pooling term is effectively saturating rather than being comfortably learned.

### E. Phase 2 can produce a coherent graph bundle without a strong downstream truth test

`latent_temporal_graph_bundle.json` is internally coherent and the synthetic test is reasonable. But there is still no benchmark that says:

- this direct sparse edge set is recoverable under realistic upstream Phase 15 uncertainty
- this hidden-driver split is stable enough to trust scientifically

That means Phase 2 can win on internal consistency while still being scientifically weak.

## 2. Failure Modes Likely To Be Missed

### A. False win: perfect structured parity hides incorrect semantic promotion

Phase 0 can regenerate structured rows perfectly and still mis-promote a narrative row from `context_only` to `proxy_indicator` or `direct_indicator`.

Current audits would mostly miss that unless it causes a downstream obvious inconsistency.

### B. False win: completed manifests with misleading summary counts

A run can appear complete because `stage_status` is `completed`, while the manifest-level summary fields are stale or not representative. `phase0_manifest.json` is the clearest current example.

### C. False win: Phase 15 learns a mathematically consistent latent state that is not well identified

The current v2 engine can now say `learned_burden_softmax` and emit smooth bottom-up states. That does not by itself imply:

- calibrated uncertainty
- stable causal interpretation
- valid block magnitudes

If the supervision is narrow or the priors dominate, the fit can still look elegant and be scientifically weak.

### D. False win: Phase 2 direct edges are stable only because upstream states are over-regularized

If Phase 15 states are overly smoothed or regionally pooled, Phase 2 may find stable sparse edges that are artifacts of regularization rather than true temporal dependence.

### E. Hidden common drivers can be mistaken for meaningful scientific latent mechanisms

The low-rank component in Phase 2 is a mathematically useful decomposition. It is not yet a validated scientific mechanism model. Without perturbation tests and benchmark recovery, it could easily be overinterpreted.

### F. Evaluation can miss NCR-dominance or reporting-dominance effects

The burden-weight model is now much better than before, but the repo still needs explicit checks that national supervision is not just a proxy for reporting concentration, especially around NCR and large urban areas.

## 3. Must-Run Audits Before Trusting Results

These should become mandatory gates before treating results as scientifically trustworthy.

### A. Human-labeled Phase 0 benchmark

Build a stratified labeled set of extracted rows covering:

- structured tables
- OCR text
- free-text literature
- HARP-like reports
- PhilHealth / PSA / survey sources

For each row, label:

- canonical name
- geo scope and geo id
- time
- measurement role
- value
- unit

Primary metrics should be exact-match and tolerant-match accuracy, not only parser success.

### B. Phase 15 synthetic identifiability suite

Create synthetic province-month latent states, then observe them through:

- monthly province rows
- region-year rows
- national annual rows
- missingness patterns that resemble the real repo

Then test whether the v2 engine recovers:

- signs
- loading magnitudes
- burden weights
- uncertainty width
- persistence `phi`

under progressively weaker supervision.

### C. Phase 15 uncertainty calibration audit

The v2 engine emits posterior standard deviations, but there is no calibration audit yet.

Needed checks:

- empirical coverage on held-out synthetic truth
- interval-width vs error relationship
- uncertainty widening under information ablation

### D. Phase 2 perturbation and stability audit

Take the same run and perturb:

- a small fraction of Phase 0 rows
- sign priors
- burden-weight supervision rows
- Phase 15 hyperparameters

Then measure how much Phase 2 direct edges and hidden-driver rows change.

If edge support is brittle under small upstream changes, the graph is not ready for strong interpretation.

### E. Cross-phase keep-or-revert benchmark

A future change should not be accepted because it emits more artifacts or a smoother latent surface.

It should pass a fixed evaluation panel:

- Phase 0 labeled extraction accuracy
- Phase 15 synthetic recovery and calibration
- Phase 2 graph stability under perturbation
- downstream Phase 3 forecast or peak-behavior delta

## 4. Top 5 Evaluation Improvements

### 1. Add a real labeled benchmark for unstructured extraction

This is the highest-value addition because it closes the main gap acknowledged by the current audit itself.

### 2. Make manifests auditable against emitted artifacts

Every phase manifest should include counters that are recomputed from artifacts before writeout. A manifest that says `completed` while reporting stale zeros should fail validation.

### 3. Add identifiability and calibration as first-class tests for Phase 15

The current synthetic tests are good contract tests, but the phase now needs:

- recovery benchmarks
- uncertainty calibration
- prior sensitivity checks

### 4. Add upstream-to-downstream perturbation audits

The repo currently validates many stages in isolation. It now needs pipeline sensitivity tests:

small extraction change -> Phase 1 eligibility change -> Phase 15 state shift -> Phase 2 edge shift

That is the only way to detect fragile scientific wins.

### 5. Separate “artifact complete” from “scientifically trusted”

The repo should expose two statuses:

- `artifact_status`
- `trust_status`

Right now, several stages can be artifact-complete while still scientifically provisional. That distinction should be explicit in manifests and reports.

## Recommended Gate Structure For Future Changes

Every future change to Phases 0, 1, 15, or 2 should pass all of the following:

1. Contract gate  
   Artifact paths exist, shapes are correct, tensors finite, schemas valid.

2. Determinism gate  
   Structured-source replay and HARP adjudication remain stable.

3. Accuracy gate  
   Human-labeled Phase 0 benchmark does not regress.

4. Identifiability gate  
   Phase 15 synthetic recovery and calibration do not regress.

5. Stability gate  
   Phase 2 sparse and low-rank structure remains stable under perturbations and bootstrap.

6. End-to-end gate  
   Downstream Phase 3 or later task metrics improve, or at minimum do not regress materially.

## Bottom Line

The repo is no longer in the dangerous early state where “everything is heuristic and nothing is checked.” There is now substantial validation infrastructure.

But the current validation is still strongest on:

- deterministic parity
- internal consistency
- artifact contracts
- selected synthetic checks

and still weaker on:

- semantic extraction truth
- identifiability
- uncertainty calibration
- cross-phase perturbation sensitivity

So the correct scientific stance is:

- Phase 0 structured extraction and HARP adjudication are close to trustworthy.
- Phase 1 normalization is contractually solid but not yet empirically calibrated.
- Phase 15 v2 is a serious model, but still not fully benchmarked as an inferential engine.
- Phase 2 latent temporal graphing is promising, but still vulnerable to false wins from upstream regularization and weak decomposition identification.

The next investment should go into evaluation, not more feature growth.
