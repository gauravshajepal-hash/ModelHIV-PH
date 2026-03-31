# Phase 3 Numeric Classification Audit

## Rule

For `src/epigraph_ph/phase3/rescue_core.py`, model-facing coefficients are allowed only when they are one of:

- prior hyperparameters
- numerical stabilizers
- explicit constraint settings
- observed reference data

Anonymous fallback coefficients inside the core are not allowed.

## What Moved Out Of `rescue_core.py`

The following classes now come from the HIV plugin contract instead of inline defaults in the core:

- Phase 3 prior vectors and templates
  - subgroup priors
  - transition priors
  - duration template
  - observation fallback curves
  - latent observation hyperparameters
  - subgroup/CD4/archetype hyperpriors
  - Torch MAP loss scales
  - frozen-backtest hyperparameters
- Phase 3 constraint settings
  - anchor-curve decay floors and linkage blend settings
  - subgroup row-weight mixing
  - subgroup anchor-strength rules
  - network-signal mixing
  - feature-support scoring
  - target-match scoring and top-k limits
  - matched-support and normalized-row support weighting
  - Torch initialization settings
- Phase 3 observed reference data
  - official UNAIDS/WHO reference points
  - HARP program reference points

## What Is Still Allowed In `rescue_core.py`

After this pass, the remaining literals in `rescue_core.py` are intended to fall into only these categories:

- ontology/cardinality/indexing
  - state positions
  - duration/month arithmetic
  - tensor axis counts
- direct data defaults
  - missing row fields defaulting to `0.0`
  - empty summaries defaulting to zero-sized arrays
- numerical algebra around named contract values
  - `0.0` and `1.0` in clipping/simplex expressions
  - shape-safe normalization logic

## New Guardrail

`tests/test_phase3_pipeline_pytest.py` now checks that `phase3/rescue_core.py` no longer uses numeric fallback values on plugin/config reads such as:

- `_phase3_prior(..., 0.18)`
- `loss_scale_cfg.get(..., 4.0)`
- `latent_cfg.get(..., 0.22)`

That pattern is now treated as a contract violation.

## Remaining Work

This pass removed anonymous plugin/config fallbacks, but it did not yet eliminate every inline literal in the core. The next strict cleanup target is:

- derived-feature hook masks and blending formulas
- JAX-side internal regularization weights that are still literal math constants
- residual metric/report constants that should either become named stabilizers or be explicitly documented as ontology/report formatting

The standard is now:

- no anonymous coefficient fallbacks in the core
- named contract ownership for priors, stabilizers, constraints, and observed references
