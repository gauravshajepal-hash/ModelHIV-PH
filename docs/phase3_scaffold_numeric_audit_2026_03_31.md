# Phase 3 Scaffold Numerical Audit

Date: 2026-03-31

## Scope

This audit covered the new Phase 3 scaffold path:

- `src/epigraph_ph/phase3/temporal_scaffold.py`
- `src/epigraph_ph/phase3/mixed_frequency.py`
- `src/epigraph_ph/phase3/peak_search.py`
- `src/epigraph_ph/phase3/rescue_core.py`

It focused on:

- Torch/JAX consistency
- edge-case shape stability
- removal of hidden implementation constants
- code bloat in the new scaffold path

## Findings Fixed

1. Death prediction scaling was duplicated and partially hardcoded.
   - Fixed by introducing shared Phase 3 death helpers in `rescue_core.py`.
   - Death scale and ceiling now come from plugin config instead of repeated literals.

2. JAX penalty scales diverged from the documented/configured model.
   - Fixed by adding `phase3.jax_svi_loss_scales` to the HIV plugin contract.
   - JAX now reads observation and penalty scales from config rather than inline numbers.

3. Peak-search search behavior contained hidden assumptions.
   - Initial search standard deviations, adaptive std bounds, peak quantiles, and transition-group templates now live in `phase3.peak_search` config.
   - Peak-search no longer embeds those values directly in code.

4. Temporal scaffold fallback behavior used numeric config defaults inline.
   - Fixed by threading required temporal and evolution settings into the scaffold bundle.
   - New scaffold code no longer depends on numeric `.get(..., default)` fallbacks for its main settings.

5. Peak-search mass penalty was declared but not actually used.
   - Fixed by tracking state-mass violation during simulated future rollout and adding it to the objective.

6. Single-column / single-regime shape regressions were not explicitly tested.
   - Fixed by adding tests for:
     - single-month temporal basis
     - single-regime shock basis
     - single-month Torch anchor override handling
     - peak-search rollout mass preservation

## Defensible Numbers

The remaining numerical values in the audited Phase 3 scaffold code now fall into two categories only:

1. Plugin-configured modeling assumptions
   - temporal basis spacing
   - penalty scales
   - transition group templates
   - search dispersion settings
   - death scale / ceiling
   - suppression/testing margins

2. Numerical safeguards
   - `SAFE_DIVISION_EPS` in `phase3/numerics.py`
   - derived from `np.finfo(np.float32).eps`
   - used only to avoid singular normalization / division / logit failures

## Tests Run

- `pytest tests/test_phase3_temporal_peak_pytest.py -q`
- `pytest tests/test_phase3_pipeline_pytest.py -q`
- `pytest tests -q`

Result:

- `157 passed, 24 warnings`

## Current Bloat Hotspots

Line counts after this pass:

- `src/epigraph_ph/phase0/pipeline.py`: `4603`
- `src/epigraph_ph/phase2/pipeline.py`: `1544`
- `src/epigraph_ph/phase3/pipeline.py`: `1640`
- `src/epigraph_ph/phase3/rescue_core.py`: `6121`
- `src/epigraph_ph/plugins/hiv.py`: `1875`

Phase 3 local files created for de-bloating:

- `src/epigraph_ph/phase3/peak_search.py`: `480`
- `src/epigraph_ph/phase3/temporal_scaffold.py`: `326`
- `src/epigraph_ph/phase3/mixed_frequency.py`: `169`
- `src/epigraph_ph/phase3/numerics.py`: `9`

## Recommended Next Refactors

1. Split Phase 3 penalty construction from `rescue_core.py`.
   - national/official/HARP/linkage/suppression penalty builders should live in a dedicated module.

2. Split Phase 3 artifact emission from `rescue_core.py`.
   - manifest assembly and artifact writing are large and mechanically separate from inference.

3. Split HIV plugin config into phase-specific config builders.
   - `plugins/hiv.py` is now primarily a configuration monolith.

4. Split Phase 0 parsing/extraction orchestration.
   - `phase0/pipeline.py` remains the largest operational bloat source in the repo.
