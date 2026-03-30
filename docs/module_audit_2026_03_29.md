# Module Audit 2026-03-29

## Scope

This audit covers the live bounded replay:

- `bounded-v2-replay-allprovinces-anchorpack2-20260329`

It also compares against the prior replay:

- `bounded-v2-replay-allprovinces-harp-avfix-20260329`

The audit is tied to code behavior and emitted artifacts, not just static code reading.

## Executive Verdict

The pipeline is now more honest than before:

- Phase 0 and Phase 1 now ingest real official anchor rows from local Philippines HIV PDFs.
- `anchor_eligible_count` moved from `0` to `137`.
- The regional outputs are no longer flat because of the earlier Phase 3 missing-cell bug.
- The truth package is emitted across the pipeline and the full test suite passes.

But the pipeline is still not publication-ready as a calibrated scientific model.

The two highest-risk validity issues are:

1. Time extraction is still polluted by slide/chart years, which contaminates the month axis.
2. Geography resolution is still mixed, so province axes contain region-level labels.

The current model remains substantially over-optimistic on diagnosis and still misses the second 95.

## Evidence Snapshot

Current bounded replay:

- Phase 1 ground truth summary:
  - `anchor_eligible_count = 137`
  - `province_count = 108`
  - `month_count = 76`
- Phase 3 HARP check:
  - modeled diagnosed stock `0.910468`
  - reference diagnosed stock `0.622527`
  - diagnosed error `0.287941`
- Phase 3 official check:
  - 2022 first95 error `0.260562`
  - 2025 first95 error `0.360468`

## Findings

### P0. Time-axis pollution remains a critical scientific defect

Files:

- `src/epigraph_ph/phase0/pipeline.py`
- `src/epigraph_ph/phase1/pipeline.py`

The current extraction path still treats many slide/chart years as real measurement timestamps. In the current replay, the month axis contains implausible entries such as:

- `1913-01`
- `1954-01`
- `1972-01`
- `2050-01`

These do not represent valid HIV observation months. They arise because:

- `_detect_time_for_span()` extracts the first year-like token it sees.
- `_unit_and_values()` extracts numeric spans from whole slide pages, including chart axes and presentation furniture.

This contaminates:

- Phase 1 tensor construction
- Phase 1.5 factor stability environments
- Phase 2 time-feature summaries
- Phase 3 anchor interpolation and losses

This must be fixed before publication claims.

### P0. Geography resolution is still mixed inside the province axis

Files:

- `src/epigraph_ph/geography.py`
- `src/epigraph_ph/phase1/pipeline.py`

The current `province` axis includes true provinces/cities and region-level labels together:

- `Abra`
- `Cebu City`
- `CALABARZON`
- `Central Luzon`
- `National Capital Region`
- `Philippines`
- `unknown`

The root issue is that `geo_resolution_label()` collapses everything that is not national/global into `subnational`. That is too coarse. Region, province, city, and synthetic buckets should not share one resolution class.

This breaks:

- province aggregation
- regional rollups
- province-vs-region truth checks
- interpretation of subnational MAE

### P1. Phase 3 is still over-diagnosing badly

Files:

- `src/epigraph_ph/phase3/rescue_core.py`

Even after the HARP-first and local-anchor changes, the current replay still overshoots diagnosis:

- HARP diagnosed stock reference: `0.622527`
- Model diagnosed stock: `0.910468`

This is not a cosmetic miss. It means the latent core still permits a much larger diagnosed mass than the operational program counts justify.

The observation system is therefore not yet dominating the latent state sufficiently.

### P1. Phase 3 hierarchy reconciliation is still too weak

Files:

- `src/epigraph_ph/phase3/rescue_core.py`

Current Phase 4 block report shows:

- `hierarchy_reconciliation_nearly_exact = false`
- reconciliation value `0.27238`

This is too large for a system that wants to claim credible subnational structure. The hierarchy exists, but it is not constraining the state estimates strongly enough.

### P1. Phase 3 subnational MAE is still worse than the naive baseline

Files:

- `src/epigraph_ph/phase3/rescue_core.py`
- `src/epigraph_ph/phase4/pipeline.py`

Phase 4 blocking report shows:

- model subnational MAE: `0.233653`
- naive MAE: `0.07321`

That is a direct signal that the current promoted factor/core combination is not yet extracting usable subnational predictive structure.

### P2. Phase 1.5 and Phase 2 are still mostly diagnostic, not yet predictive

Files:

- `src/epigraph_ph/phase15/pipeline.py`
- `src/epigraph_ph/phase2/pipeline.py`

Current Phase 2 ground truth summary:

- `main_predictive_count = 0`
- `supporting_context_count = 12`

This is acceptable as a conservative safety posture, but it means the mesoscopic factor engine is not yet delivering the key scientific promise: promoted predictive structure that improves the HIV core.

### P2. Phase 1.5 depends on Phase 3 observation logic

Files:

- `src/epigraph_ph/phase15/pipeline.py`
- `src/epigraph_ph/phase3/rescue_core.py`

Phase 1.5 imports `build_observation_ladder()` from Phase 3. That creates an undesirable coupling:

- upstream factor scoring depends on downstream observation logic
- module truth boundaries are less clean
- future observation changes can silently alter factor stability selection

This should be inverted into a shared observation contract module, not a Phase 3 import.

### P2. Phase 2 base DAG still destroys geography too early

Files:

- `src/epigraph_ph/phase2/pipeline.py`

`_lagged_feature_matrix()` reduces the tensor by averaging over provinces before the base DAG step. For the Philippines use case, this loses exactly the subnational heterogeneity the pipeline is supposed to preserve.

The current DAG is therefore still mostly a national-level diagnostic scaffold, not a real subnational promotion engine.

### P3. Registry is operationally sound but still thin scientifically

Files:

- `src/epigraph_ph/registry/sources.py`
- `src/epigraph_ph/registry/subparameters.py`

The registry layer is useful and audit-friendly, but it mostly republishes upstream fields. It does not yet compute:

- source contradictions
- source freshness conflicts
- definition mismatches
- duplicate numeric anchor conflicts

That is acceptable for engineering, but weak for publishable evidence synthesis.

### P3. Phase 4 gating is correct and should remain strict

Files:

- `src/epigraph_ph/phase4/pipeline.py`

Phase 4 is behaving correctly. In the current replay it stays blocked because Phase 3 failed critical gates. This is good. The pipeline is not pretending the policy layer is ready.

## Module-by-Module Audit

### Runtime

Strengths:

- `EPIGRAPH_FORCE_CPU` works and made bounded replay safe.
- `write_ground_truth_package()` now enforces a truth contract across phases.

Weaknesses:

- Backend warnings from third-party packages remain noisy.
- There is no built-in memory budget logger or per-phase RAM snapshot.

### Geography

Strengths:

- national alias deduplication is working
- Philippines alias coverage is strong

Weaknesses:

- `geo_resolution_label()` is too coarse
- region labels leak into province-level modeling
- city/province/region are not separated formally

Required correction:

- replace `subnational` with explicit `region`, `province`, `city`, `unknown`

### Phase 0 Harvest

Strengths:

- broad corpus design is working
- local official anchor PDFs are now wired in
- source provenance is preserved

Weaknesses:

- official web seeds are mostly placeholders
- no dedicated official numeric ingestion adapters yet

Required correction:

- separate metadata seeds from numeric anchor feeds

### Phase 0 Parse

Strengths:

- local anchor PDFs now enter as real documents
- page-specific parsing for official anchor pages works

Weaknesses:

- generic PDF/text parsing still mixes slide furniture with real data
- no chart-axis suppression

Required correction:

- page- and layout-aware exclusion rules for presentation decks

### Phase 0 Extract

Strengths:

- official anchor pack extraction now emits real numeric rows
- extraction can produce national and subnational official anchor observations

Weaknesses:

- generic regex extraction still captures too many irrelevant numbers
- time extraction is still too naive

Required correction:

- stricter numeric context windows
- table-aware extraction
- date parsing tied to measurement statements, not page-wide first-year wins

### Registry

Strengths:

- source and subparameter registries are simple and reproducible

Weaknesses:

- little conflict detection
- little definition auditing

Required correction:

- add registry-level contradiction and definition checks

### Phase 1

Strengths:

- bias/reliability fields are useful
- truth package catches duplicate national labels

Weaknesses:

- mixed-resolution geography survives normalization
- time axis is polluted

Required correction:

- resolution-stratified axes
- time-axis cleaning before tensorization

### Phase 1.5

Strengths:

- mesoscopic factor engine is real
- network feature families are explicit

Weaknesses:

- factors still inherit polluted time/geography upstream
- factor engine is not yet generating main predictive winners

Required correction:

- only build factors on cleaned time/geography slices
- add stronger factor interpretability and exclusion rules

### Phase 2

Strengths:

- promotion budgets are enforced
- conservative no-promotion behavior is safer than fake promotion

Weaknesses:

- base DAG step is still too nationalized
- promotion remains diagnostic-heavy

Required correction:

- subnational-aware tournament scoring
- remove dependence on province-averaged lag matrix as the main scaffold

### Phase 3

Strengths:

- observation ladder is now explicit
- HARP and official checks are emitted
- phase4 remains blocked when calibration fails

Weaknesses:

- over-diagnosis remains severe
- hierarchy reconciliation remains weak
- many regional differences still look too smooth

Required correction:

- stronger diagnosed stock penalty
- stronger province-to-region reconciliation
- real subnational anchor pack beyond the current PDF-derived seeds

### Phase 4

Strengths:

- correctly blocked

Weaknesses:

- none urgent; it should stay blocked

## What Changed in This Audit Pass

Implemented:

- local official anchor PDF ingestion in Phase 0
- page-targeted PDF parsing for official anchor docs
- structured anchor extraction from the WHO core team and surveillance decks
- Phase 0 alignment scaling fix to avoid overflow from large count inputs
- updated tests so rescue-v2 does not falsely require non-empty main predictive promotions

Validated:

- full test suite passes: `41 passed`
- bounded replay completed safely on CPU with single-threaded math libraries

## Immediate Next Corrections

1. Fix time extraction so slide/chart years do not populate the month axis.
2. Split geography resolution into region/province/city and rebuild Phase 1 axes.
3. Rebuild Phase 1.5 and Phase 2 on the cleaned axes.
4. Strengthen Phase 3 diagnosed stock and hierarchy penalties again after the upstream cleanup.
