# Pipeline Ground Truth Contract

This document defines how **every module in the pipeline gets a ground truth reference, anchor, or falsification test**.

The key point is:

> not every module has direct observed truth,
> but every module must have a way to be checked against something external, internal, or counterfactual.

That is the only way to keep the pipeline scientific under Philippines-style sparse, biased, incomplete HIV data.

This contract is informed by:

- the current `phase0` to `phase4` architecture
- the rescue-core documents
- the mesoscopic factor engine design
- the Philippines surveillance and estimates decks
- the older `epigraph-ph` repo audit scripts, especially:
  - `audit_official_vs_local.py`
  - `systematic_hiv_audit.py`

Those older scripts are useful not because the old model is strong, but because they already identified the right discipline:

- compare local outputs against official references
- detect denominator mismatches
- separate headline values from annex values
- perform internal consistency checks before claiming anything

## 1. Ground Truth Is a Ladder, Not a Binary

For this pipeline, truth is not one thing.

We need a formal ladder:

1. `anchor_truth`
2. `benchmark_truth`
3. `proxy_truth`
4. `prior_truth`
5. `synthetic_truth`
6. `null_or_invariance_test`

## 1.1 Anchor truth

Observed or institutionally authoritative quantities that the module is not allowed to ignore.

Examples:

- official HARP diagnosis and ART counts
- official SHIP or DOH surveillance headline cascade values
- official UNAIDS Philippines all-ages annual series
- lab-confirmed counts where definitions are explicit

## 1.2 Benchmark truth

Reference values that are not perfect truth, but are authoritative enough to benchmark against.

Examples:

- UNAIDS estimated first 95
- AEM/Spectrum national PLHIV estimates
- WHO/UNAIDS country updates
- validated quarterly surveillance tables

## 1.3 Proxy truth

Observed or derived quantities that measure the target indirectly.

Examples:

- viral-load-tested denominator as a proxy for suppression process
- bottles dispensed as proxy for months on PrEP
- treatment refill recency as proxy for continuity of treatment

## 1.4 Prior truth

Scientifically plausible constraints that are not directly observed, but must shape the model.

Examples:

- nonnegativity
- cascade monotonicity in stock relationships
- reasonable retention dynamics
- plausible subgroup proportions

## 1.5 Synthetic truth

Controlled toy data or generated fixtures used only for numerical or structural validation.

Examples:

- mass-conserving state transition fixtures
- sparse mobility graph fixtures
- factor extraction recovery fixtures

## 1.6 Null or invariance tests

When direct truth is impossible, the module must still pass falsification.

Examples:

- source-dropout stability
- sign stability
- permutation null gap
- environment invariance
- reconstruction error limits

## 2. Global Rule

Every module must emit a **ground truth package**.

Minimum required files per module:

- `ground_truth_manifest.json`
- `ground_truth_checks.json`
- `ground_truth_summary.json`

Optional but recommended:

- `ground_truth_failures.csv`
- `ground_truth_reference_inventory.json`
- `ground_truth_visual_report.md`

Every check row should contain:

- `module`
- `artifact`
- `quantity`
- `truth_class`
- `reference_source`
- `reference_value`
- `candidate_value`
- `comparison_type`
- `tolerance`
- `status`
- `notes`

## 3. Phase 0 Ground Truth

Phase 0 does not estimate epidemic states.
Its truth problem is:

- did we ingest what we think we ingested?
- did we preserve provenance?
- did we extract the right thing from the right source?

## 3.1 Ground truth objects

Phase 0 truth classes:

- `anchor_truth`: source URL, filename, page, table, paragraph, provenance metadata
- `proxy_truth`: OCR confidence, parse confidence, extraction confidence
- `synthetic_truth`: extraction test fixtures

## 3.2 Required checks

### A. Source truth

Check that each harvested source has:

- stable source identity
- document type
- country relevance
- date or period
- basic provenance path

Failure mode:

- orphaned extracted values with no source lineage

### B. Extraction truth

For a stratified sample of documents, check extracted quantities against manual review or gold snippets:

- values
- units
- metric type
- geography
- period

This should be a rolling audit, not only one-time testing.

### C. Duplicate and conflict truth

When multiple documents report the same quantity:

- detect duplicates
- detect conflicting values
- record source precedence

### D. Geography and time truth

The extracted geography and period must be checked against:

- explicit text in source
- known Philippines location aliases
- known reporting periods

### E. Coverage truth

Phase 0 should measure whether the intended source universe was actually covered:

- which official reports were found
- which expected reports were missing
- which periods are missing

## 3.3 Phase 0 artifacts to add or standardize

- `source_inventory.json`
- `source_coverage_report.json`
- `extraction_gold_audit.json`
- `duplicate_conflict_report.json`
- `geo_time_resolution_audit.json`

## 4. Registry Ground Truth

The registry is the first place where the system decides what exists.

That means its ground truth is:

- provenance correctness
- de-duplication correctness
- candidate identity correctness

## 4.1 Required checks

- one canonical candidate id maps back to known supporting rows
- no candidate is promoted without at least one verified provenance chain
- conflicts are explicit, not silently merged
- truth class is attached to each candidate:
  - anchor-backed
  - benchmark-backed
  - proxy-only
  - literature-only

## 4.2 Registry artifact requirements

- `candidate_truth_index.json`
- `candidate_provenance_graph.json`
- `candidate_conflict_matrix.csv`
- `candidate_truth_class_summary.json`

## 5. Phase 1 Ground Truth

Phase 1 aligns data.
Its truth problem is:

- did we preserve the meaning of the variable while harmonizing unit, geography, and time?

## 5.1 Ground truth objects

- `anchor_truth`: official denominators, province and region mappings, time labels
- `proxy_truth`: interpolation or carry-forward metadata
- `prior_truth`: monotonicity, feasible ranges, nonnegative counts

## 5.2 Required checks

### A. Unit truth

After conversion, verify that:

- percentages remain percentages
- counts are counts
- rates match their denominators
- no impossible unit conversions occurred

### B. Geography truth

Province, region, and national mappings must reconcile:

- province to region
- region to national
- no duplicate alias-induced rows

### C. Time truth

Check that:

- monthly alignment is explicit
- quarterly-to-month conversion flags exist
- carried-forward or interpolated values are labeled

### D. Denominator truth

Every density or coverage quantity must record its denominator source and denominator class.

### E. Alignment distortion audit

Record how much the aligned value differs from the raw source when interpolation, smoothing, or harmonization was applied.

## 5.3 Phase 1 artifacts to add or standardize

- `unit_conversion_audit.json`
- `denominator_registry.json`
- `geo_reconciliation_report.json`
- `time_alignment_audit.json`
- `alignment_distortion_report.json`

## 6. Phase 1.5 Ground Truth

This is the mesoscopic factor engine.
It cannot have direct truth in the same way an observed count does.

Its truth must come from:

- reconstruction quality
- interpretability
- stability
- invariance
- source robustness

## 6.1 Ground truth objects

- `proxy_truth`: reconstruction of block structure and co-behavior
- `prior_truth`: interpretable signed loadings, reasonable block membership
- `null_or_invariance_test`: source-dropout, sign stability, environment stability
- `synthetic_truth`: recover known latent factors from controlled fixtures

## 6.2 Required checks

### A. Factor reconstruction truth

For each factor, report:

- weighted explained variance
- reconstruction error
- missing-data sensitivity
- loading concentration

### B. Interpretability truth

A factor cannot become promotable unless:

- it has a coherent member set
- signs are interpretable
- the block label is plausible

### C. Source robustness truth

Test each factor under:

- removal of literature-only signals
- removal of proxy-only signals
- official-only subset where possible
- mixed-source subset

### D. Environment stability truth

Test each factor across:

- Luzon / Visayas / Mindanao
- pre-pandemic / pandemic / post-pandemic
- high-data / sparse-data provinces

### E. Synthetic recovery truth

On generated fixtures with known latent factors:

- recover the factors within tolerance
- keep deterministic results under fixed seeds

## 6.3 Network-derived factor truth

Reaction-diffusion, percolation, and information-propagation features must not be accepted just because they are mathematically sophisticated.

Each family needs its own truth tests.

### A. Reaction-diffusion / mobility features

Truth tests:

- graph validity
- mobility matrix normalization
- diffusion operator stability
- monotone response to added or removed mobility edges

### B. Percolation / fragility features

Truth tests:

- path inflation under node removal
- reachability loss under service disruption
- redundancy metrics behave correctly on known graph fixtures

### C. Information propagation features

Truth tests:

- boundedness
- monotone behavior with increased connectivity
- correct behavior on simple metapopulation fixtures

## 6.4 Phase 1.5 artifacts to add or standardize

- `factor_reconstruction_report.json`
- `factor_interpretability_report.json`
- `factor_source_dropout_report.json`
- `factor_environment_stability_report.json`
- `network_operator_audit.json`
- `synthetic_factor_recovery_report.json`

## 7. Phase 2 Ground Truth

Phase 2 is promotion, not truth creation.
Its job is to decide what survives.

Its truth problem is:

- did we promote the right factors for the right reasons?

## 7.1 Ground truth objects

- `benchmark_truth`: holdout gain against official or validated targets
- `null_or_invariance_test`: permutation null, source dropout, sign stability
- `prior_truth`: mechanistic admissibility

## 7.2 Required checks

### A. Promotion truth

A promoted factor must have:

- positive predictive gain against a frozen baseline
- stability across environments
- no catastrophic guardrail violations

### B. No-leakage truth

Phase 2 must explicitly detect and reject:

- target synonyms
- downstream leakage variables
- direct restatements of the outcome

### C. Budget truth

Promotion budgets must be enforced and audited:

- number of main predictive factors
- number of support factors
- number of direct transition modifiers
- number of subgroup-specific modifiers

### D. Sign and reference plausibility

If a factor improves fit but has wildly unstable sign or contradicts known anchors without explanation, it should not be promoted to main predictive.

## 7.3 Phase 2 artifacts to add or standardize

- `promotion_truth_report.json`
- `leakage_screen_report.json`
- `promotion_budget_report.json`
- `promotion_reason_codes.json`
- `factor_holdout_comparison.csv`

## 8. Phase 3 Ground Truth

Phase 3 is the core scientific model.
This is where anchor truth matters most.

## 8.1 Ground truth objects

- `anchor_truth`: HARP and official surveillance cascade surfaces
- `benchmark_truth`: UNAIDS and WHO national estimates
- `proxy_truth`: viral-load-tested quantities, proxy retention surfaces
- `prior_truth`: mechanistic constraints
- `null_or_invariance_test`: perturbation and dropout tests

## 8.2 Required checks

### A. Observation ladder truth

Every observation used by the model must be labeled:

- direct observed
- bounded observed
- proxy observed
- prior only

The model must never present proxy truth as anchor truth.

### B. Official-reference truth

Every fit run must compare against:

- official national UNAIDS series where available
- official surveillance headline values
- WHO/DOH checkpoints when relevant

### C. Internal consistency truth

This is where the old repo already gave us the correct pattern.

We must always check:

- first 95 denominator correctness
- second 95 denominator correctness
- third 95 denominator correctness
- suppression among tested vs suppression among on-ART separation
- regional annex sums vs national headline counts
- treatment table sums vs headline counts

No run should be considered credible without these checks.

### D. Hierarchy truth

Province, region, and national outputs must reconcile.

### E. Mechanistic truth

Check:

- nonnegative state occupancy
- mass conservation
- valid transition probabilities
- reasonable subgroup occupancy
- valid CD4 simplex

### F. Counterfactual sanity truth

If factor perturbations produce effects:

- signs should be reasonable
- magnitude should not be absurd
- policy-like shocks should not break core epidemiologic constraints

## 8.3 Phase 3 artifacts to add or standardize

- `official_reference_inventory.json`
- `reference_check_official.json`
- `internal_consistency_audit.json`
- `cascade_denominator_audit.json`
- `hierarchy_reconciliation_audit.json`
- `mechanistic_guardrails.json`
- `counterfactual_sanity_report.json`

## 9. Phase 4 Ground Truth

Phase 4 has the weakest direct truth.
That is exactly why it must remain blocked until Phase 3 is strong.

## 9.1 Ground truth objects

- `benchmark_truth`: retrospective scenario checks where possible
- `synthetic_truth`: simulator fixtures
- `null_or_invariance_test`: policy sensitivity, stability, no-explosion tests

## 9.2 Required checks before Phase 4 is active

- Phase 3 benchmark gates pass repeatedly
- promoted factors are stable
- official-reference gaps are within tolerance
- no unresolved denominator or reconciliation issues

## 9.3 If Phase 4 is enabled later

Then it must report:

- simulator validity assumptions
- action-to-transition mapping
- retrospective policy sanity checks
- uncertainty range
- failure modes

## 10. Ground Truth for the Competing Incumbent

The incumbent Philippines stack is useful here because it already suggests what must count as anchor truth.

From the surveillance and estimates decks, the minimum national anchor set is:

- HARP diagnosis counts
- HARP ART counts
- official SHIP / DOH care-cascade headline values
- IHBSS prevalence / behavior / KP coverage inputs
- KP size estimate rounds
- UNAIDS all-ages annual series

The old repo also already surfaced one crucial lesson:

> denominator choice is itself a ground-truth problem.

For example:

- suppression among those tested is not the same thing as third 95
- 30-day vs 90-day ART definition materially changes PLHIV and incidence estimates

So a ground-truth contract must include **definition truth**, not just numeric truth.

## 11. Implementation Recommendation

The practical way to implement this is to give every pipeline stage a common truth schema.

Recommended shared enums:

- `truth_class`
  - `anchor_truth`
  - `benchmark_truth`
  - `proxy_truth`
  - `prior_truth`
  - `synthetic_truth`
  - `null_test`
- `status`
  - `pass`
  - `warn`
  - `fail`
  - `not_applicable`
- `comparison_type`
  - `exact_match`
  - `within_tolerance`
  - `rank_order`
  - `reconstruction`
  - `invariance`
  - `consistency`
  - `synthetic_recovery`

Recommended shared writer:

- `write_ground_truth_package(run_dir, module_name, checks, summary, references)`

Recommended hard rule:

> no module artifact is complete until its ground-truth package exists.

## 12. Bottom Line

The case for module-level ground truth is strong because this pipeline has too many places where error can become invisible:

- extraction errors
- geography errors
- denominator errors
- stale survey assumptions
- proxy slippage
- spurious factor promotion
- mechanistic inconsistency

The solution is not to pretend every stage has perfect truth.
The solution is:

- define the truth class for every output
- define the check type for every output
- require a truth package from every module
- make failure explicit before downstream promotion

That is how the pipeline becomes scientifically defensible.
