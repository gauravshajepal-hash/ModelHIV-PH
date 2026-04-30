# Future Experiment: Phase 2 Seeded Champions

**Date:** 2026-04-14

## Idea

Use the Phase 2 outputs as structural seeds for the frozen Phase 3 champion models, then propagate those structural states into the future to emulate shock-like and plateau-like behavior.

This is **not** a benchmark-replacement forecasting proposal. It is a structural scenario and stress-test layer on top of the current champions.

## Working intuition

- Phase 2 contains lagged graph structure and hidden/mesoscopic state information.
- Phase 3 champions already provide the strongest short-horizon readout for observed quarterly targets.
- Therefore:
  - Phase 2 should provide the **structural seed** and state evolution path.
  - Phase 3 champions should provide the **observable consequences**.

## Proposed coupling

Let:

\[
z_t = \text{Phase 2 structural state/features at time } t
\]

\[
\hat y_t^{champ} = \text{frozen Phase 3 champion forecast}
\]

Then define a bounded structural readout:

\[
\tilde y_t = \hat y_t^{champ} + \lambda \Delta(z_t)
\]

where:

- `\lambda` is bounded and regularized
- `\Delta(z_t)` is a small structural correction derived from Phase 2
- the structural layer is not allowed to freely overwrite the champion

## Intended use

- plateau emulation
- disruption pulse emulation
- delayed recovery scenarios
- regime persistence scenarios
- structural drift stress tests

## What this should not be called

- not a new benchmark forecaster
- not a true future-shock predictor
- not a fully identified mechanistic epidemic model

## Experiment ladder

1. Build a quarter-aligned or month-aligned Phase 2 structural state table from retained Phase 2 outputs.
2. Fit a bounded regime-persistence or semi-Markov evolution rule to those structural states.
3. Learn a small readout from propagated structural state to champion residual deltas.
4. Generate scenario families:
   - persistence
   - shock pulse
   - plateau
   - recovery
5. Report envelopes and qualitative structural behaviors rather than claiming a single correct future path.

## Current scientific boundary

This idea is only defensible if:

- the structural layer remains bounded
- the champion mean path remains the base forecast
- the output is reported as structural scenario emulation or stress testing
- any future "shock prediction" claim is deferred until exogenous lead indicators are added

## Immediate follow-up

Diagnose the current Phase 15 month axis and rebuild a monthly structural lane from HARP `2010+` before attempting a larger hidden-state or HSMM scenario engine.
