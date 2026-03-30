# Numeric Contract And Shape Audit

Date: 2026-03-30

## Scope

This audit focused on:

- leftover handwritten numeric behavior in `src/epigraph_ph`
- shape/compatibility risks such as row-dropping, singleton-matrix collapse, and regional allocation inconsistencies

## What was fixed

### Plugin-owned behavior moved out of runtime modules

The highest-risk behavior tables are now plugin-owned in:

- `src/epigraph_ph/plugins/hiv.py`

This cleanup covered:

- province archetype definitions and synthetic-generator parameters
- node-graph confidence mixes and fallback settings
- Phase 4 counterfactual scoring settings
- Phase 4 stochastic-control settings
- regional allocation epsilon and node-graph decision multipliers

### Runtime modules cleaned up

The following modules were refactored to consume plugin-owned settings instead of embedding larger numeric tables:

- `src/epigraph_ph/core/node_graph.py`
- `src/epigraph_ph/core/province_archetypes.py`
- `src/epigraph_ph/phase4/pipeline.py`

### New shape/integrity tests

Added or extended checks for:

- one node-state per enabled node per region
- exact per-channel regional budget conservation
- no allocation to vetoed regions
- no allocation to the `national` pseudo-region
- no inline numeric lookup tables in the high-risk control/archetype/node-graph modules

Relevant tests:

- `tests/test_phase4_pipeline_pytest.py`
- `tests/test_numeric_contract_pytest.py`
- `tests/test_phase2_pipeline_pytest.py`
- `tests/test_phase3_pipeline_pytest.py`

## Verification

Full suite:

- `82 passed, 4 warnings`

Warnings are third-party environment warnings:

- `requests` dependency version mismatch
- SWIG deprecation warnings

## Remaining hotspots

A repo-wide AST scan still shows many numeric literals outside the plugin contract. The largest remaining hotspots are:

1. `src/epigraph_ph/phase3/rescue_core.py`
2. `src/epigraph_ph/phase3/pipeline.py`
3. `src/epigraph_ph/phase0/pipeline.py`
4. `src/epigraph_ph/phase2/pipeline.py`
5. `src/epigraph_ph/phase15/pipeline.py`

These are not all equally bad.

### Remaining acceptable categories

Many remaining literals are still acceptable because they are:

- numerical stabilizers
- structural dimensions
- explicit year or month bounds
- test tolerances
- rounding precision

### Remaining suspicious categories

The main suspicious remaining categories are:

- backtest/tuning search coefficients in `phase3/pipeline.py`
- fallback and ranking heuristics in `phase0/pipeline.py`
- threshold-style discovery heuristics in `phase2/pipeline.py`
- some factor-clustering defaults in `phase15/pipeline.py`
- residual calibration weights and helper priors in `phase3/rescue_core.py`

## Honest status

The repo is in a better state, but not numerically “finished.”

What is true now:

- the most control-critical handwritten lookup tables were moved into the HIV plugin contract
- the node graph, province archetypes, and Phase 4 are under stronger contract testing
- matrix/row preservation and allocation integrity checks are in place

What is not yet true:

- every non-plugin numeric behavior coefficient in the repo has been migrated
- `phase3/rescue_core.py` is not yet numerically minimal
- `phase0`, `phase2`, and `phase15` still contain heuristic numeric defaults that need another pass

## Next cleanup targets

1. migrate `phase3/pipeline.py` backtest/tuning coefficients into the plugin or a declared calibration spec
2. classify remaining `phase3/rescue_core.py` literals into:
   - prior hyperparameter
   - numerical stabilizer
   - learned parameter initializer
   - remove
3. move `phase2` and `phase15` threshold defaults into declared discovery contracts
4. add one repo-wide audit test for suspicious high-density numeric files, with an allowlist for stabilizers and year bounds
