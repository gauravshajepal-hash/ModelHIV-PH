# Probabilistic Module Standard

## Modeling Standard

- Phase 0: uncertain evidence retrieval and measurement alignment
- Phase 1: probabilistic measurement and normalization model
- Phase 2: constrained causal structure model
- Phase 3: semi-Markov latent epidemic and cascade model
- Phase 4: stochastic decision and control model
- node graph: runtime assurance and constraint layer

## Numeric Contract

Every hardcoded number must belong to one of these buckets:

1. estimated parameter
2. prior hyperparameter
3. numerical stabilizer
4. explicit policy or constraint setting

If a number does not fit one of those buckets, it should not exist in the codebase.

## Core Implications

- Semi-Markov structure is reserved for latent cascade dynamics where dwell time matters.
- Retrieval, alignment, structure learning, and policy control do not need to be semi-Markov, but they do need to be probabilistic, calibrated, and interpretable.
- Direct-observation anchors must be allowed to pull targets downward as well as upward.
- Missing observation repair must use anchor-informed priors rather than arbitrary fallback shares.
- State initialization must use calibrated priors rather than manual splits.
- Policy effects may remain heuristic only if they are explicitly declared as policy settings outside the epidemic core.

## Current Implementation Notes

- HIV plugin-owned prior hyperparameters, numerical stabilizers, constraint settings, and policy settings now live in:
  - `src/epigraph_ph/plugins/hiv.py`
- Phase 3 rescue core consumes plugin-owned priors and constraints for:
  - subgroup priors
  - observation repair
  - cascade ordering
  - state initialization
  - CD4 overlay shaping
- Node graph and Phase 4 consume plugin-owned constraint and policy settings rather than hidden module-level tables.

## Remaining Governance Rule

When new numbers are introduced:

- epidemic-core numbers belong in plugin prior hyperparameters or numerical stabilizers
- node-graph veto and scoring numbers belong in plugin constraint settings
- Phase 4 allocation and action-response numbers belong in plugin policy settings
- inline anonymous constants should be treated as a bug
