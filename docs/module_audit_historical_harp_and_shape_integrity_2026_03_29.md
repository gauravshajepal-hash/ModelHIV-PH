# Module Audit: Historical HARP, Shape Integrity, and Regional Allocation

Date: 2026-03-29

## Scope

This audit focused on four things:

1. Expanding the historical HARP/PNAC archive so frozen backtests are scientifically usable.
2. Stress-testing matrix and tensor shape integrity, especially small-feature and JAX paths.
3. Verifying that Phase 4 uses a real regional allocation seam instead of leaking pseudo-regions.
4. Tightening test coverage around the exact failure modes likely to be silent in research code.

## What Changed

### HARP archive

Files:
- `src/epigraph_ph/harp_archive/pipeline.py`
- `src/epigraph_ph/harp_archive/seeds/historical_harp_panel_curated.csv`

Audit result:
- Before this pass, the archive was infrastructure-ready but not backtest-ready because only one true program-observed year was assembled.
- After this pass, the archive now includes curated year-end program-observed rows for 2017-2024 and is backtest-ready.

Current artifact:
- `artifacts/runs/manual-harp-archive-20260329/harp_archive/backtest_assessment.json`

Current status:
- `backtest_ready = true`
- `program_observed_year_count = 8`
- `observed_program_years = 2017..2024`

Current frozen-history artifact:
- `artifacts/runs/manual-harp-archive-20260329/harp_archive/frozen_backtest_spec.json`
- `artifacts/runs/manual-harp-archive-20260329/harp_archive/frozen_backtest_summary.json`

Important caveat:
- The 2017-2022 rows are curated from a public mirror of the 2022 DOH annual report overview.
- The 2023 row is curated from an accessible public mirror preview.
- The 2024 row is curated from the public Q4 2024 HARP surveillance PDF and is intentionally preferred over later summary-style sources for the program panel.

### Phase 2 matrix integrity

Files:
- `src/epigraph_ph/phase2/pipeline.py`

Bug found:
- Correlation-based skeleton building and NOTEARS fallback could collapse shape semantics when the feature count was 1 or the feature vector was constant.
- That could produce scalar correlation output or malformed adjacency behavior instead of a stable `(1, 1)` matrix.

Fix:
- Added `_safe_feature_corr`.
- Forced square matrix preservation for single-feature cases.
- Forced NOTEARS single-feature fallback to emit a stable zero `(1, 1)` adjacency.

Why it matters:
- This is exactly the kind of silent matrix-shape corruption that can turn into bogus DAG artifacts or downstream axis mismatch.

### Phase 3 JAX / archetype integrity

Files:
- `src/epigraph_ph/phase3/rescue_core.py`

Audit result:
- The JAX path now has explicit shape-preservation coverage for province operators.
- The test checks that province rows are not dropped, row count is preserved, and row simplex structure is preserved after mixing.

Relevant output artifacts still present:
- `artifacts/runs/pytest-rescue-v2-full/phase3/province_archetype_mixture.json`
- `artifacts/runs/pytest-rescue-v2-full/phase3/synthetic_pretraining_summary.json`

Observed operational note:
- One isolated targeted test run briefly hit a filesystem write error while building a rescue fixture.
- The failure did not reproduce after rerun and the isolated Phase 3 suite and full suite both passed cleanly afterward.
- I am classifying that as an operational flake rather than an unresolved logic bug.

### Phase 4 regional allocation seam

Files:
- `src/epigraph_ph/phase4/pipeline.py`

Bug found:
- The real regional allocator was still allowing a pseudo-region called `national` to receive budget shares.

Fix:
- `national` is now excluded from allocatable regional budgets.
- The node graph continues to act on the regional seam, but only real regions receive allocations.

Current artifact:
- `artifacts/runs/pytest-legacy-full/phase4/regional_selected_policy.json`

Audit result:
- `national` no longer appears in the regional allocation output.

## New Tests Added

### HARP archive
- `tests/test_harp_archive_pipeline_pytest.py::test_packaged_curated_seed_makes_archive_backtest_ready`
- `tests/test_harp_archive_pipeline_pytest.py::test_packaged_seed_survives_manual_seed_dir_override`

### Phase 2
- `tests/test_phase2_pipeline_pytest.py::test_phase2_safe_feature_corr_preserves_square_shape_for_single_feature`
- `tests/test_phase2_pipeline_pytest.py::test_phase2_notears_single_feature_never_drops_matrix_shape`

### Phase 3
- `tests/test_phase3_pipeline_pytest.py::test_phase3_jax_operator_preserves_row_count_when_available`

### Phase 4
- `tests/test_phase4_pipeline_pytest.py::test_regional_allocation_surface_excludes_national_pseudo_region`

## Test Results

Targeted runs:
- `python -m pytest tests/test_harp_archive_pipeline_pytest.py -q`
- `python -m pytest tests/test_phase2_pipeline_pytest.py -q`
- `python -m pytest tests/test_phase3_pipeline_pytest.py -q`
- `python -m pytest tests/test_phase4_pipeline_pytest.py -q`

Full suite:
- `python -m pytest tests -q`
- result: `65 passed, 4 warnings`

Warnings were third-party environment warnings, not repo test failures:
- `requests` dependency warning
- `Swig` deprecation warnings

## Module-by-Module Judgment

### Phase 0
- No new blocking bug found in this pass.
- Existing retrieval/evidence stack remains usable.
- Main remaining risk is scientific coverage quality, not code breakage.

### Registry / adapters / validate
- No new structural bug found in this pass.
- The archive expansion now gives these layers better truth-bearing historical inputs.

### Phase 1
- No new shape bug found in this pass.
- Still inherits any upstream source-definition limitations.

### Phase 1.5
- No new implementation defect found in this pass.
- Still scientifically dependent on the cleanliness of the aligned evidence layer.

### Phase 2
- Real bug fixed: tiny-feature correlation/adjacency shape collapse.

### Phase 3
- JAX row-preservation now explicitly tested.
- No current evidence of silent province-row dropping after the fix/test pass.

### Phase 4
- Real bug fixed: pseudo-region leakage into allocation.
- The allocator is now closer to a publishable regional decision seam.

## Remaining Scientific Caveats

1. The historical panel is now usable, but not all years are sourced from the same access path.
   - 2017-2022: curated mirror table
   - 2023: curated public preview
   - 2024: public HARP PDF

2. The frozen backtest is still a carry-forward baseline proof, not the full Phase 3 frozen model backtest.

3. The next research-grade step is not more archive plumbing.
   - It is wiring a true Phase 3 frozen-history fit/evaluate loop against the new archive.

## Recommended Next Step

Build the actual frozen-history Phase 3 backtest runner on top of the now-ready archive:
- fit on 2017-2023
- lock parameters/history
- predict 2024
- compare against the observed HARP holdout

That is now a real scientific task, not blocked infrastructure.
