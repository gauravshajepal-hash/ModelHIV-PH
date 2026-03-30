# Probabilistic Defensibility Audit

Date: 2026-03-29

## Bottom line

The repository is **not** currently "all semi-Markov and probability-based".

What is true:
- The Phase 3 rescue core contains a real semi-Markov state transition kernel over `U, D, A, V, L`.
- The core state update conserves probability mass and uses transition probabilities.

What is also true:
- The same Phase 3 path still contains multiple **hand-authored numeric heuristics** for priors, subgroup composition, CD4 overlays, observation repair, initialization, and loss weighting.
- Province archetypes are currently **hand-authored prototypes**, not learned latent classes.
- The node graph and Phase 4 policy engine are **heuristic decision systems**, not semi-Markov or fully probabilistic control layers.

So the correct scientific claim today is:

- **Phase 3 contains a semi-Markov epidemic core**
- but the full pipeline is still surrounded by **heuristic support and policy layers**

## Module classification

### Acceptable probabilistic core

- `src/epigraph_ph/phase3/rescue_core.py`
  - semi-Markov state transitions:
    - `_semi_markov_step_numpy`
    - `_semi_markov_step_torch`
    - `_semi_markov_model`
  - hierarchical transition parameters in JAX/NumPyro and Torch MAP

### Mixed: probabilistic core plus heuristics

- `src/epigraph_ph/phase3/rescue_core.py`
  - hand-authored priors and fallback rules still affect inference
- `src/epigraph_ph/core/province_archetypes.py`
  - archetype mixtures are inferred, but archetype templates and prior shifts are manually specified

### Heuristic, not semi-Markov

- `src/epigraph_ph/core/node_graph.py`
- `src/epigraph_ph/phase4/pipeline.py`
- parts of `src/epigraph_ph/phase2/pipeline.py`
- parts of `src/epigraph_ph/phase15/pipeline.py`

These are not necessarily wrong, but they must not be described in papers as if they were probabilistic epidemic state models.

## High-severity issues

### 1. Hand-authored subgroup priors still shape the latent state

Files:
- `src/epigraph_ph/phase3/rescue_core.py`

Examples:
- `DEFAULT_KP_PRIOR`, `DEFAULT_AGE_PRIOR`, `DEFAULT_SEX_PRIOR`: lines 97-100
- subgroup shrinkage strengths and pseudocounts: lines 739-754
- network-to-subgroup adjustments:
  - urbanity/stress synthesis: lines 612-613
  - subgroup shifts: lines 665-685

Why this matters:
- subgroup composition is not inferred from a generative population model
- it is still partially imposed by hard-coded shifts
- this directly affects province/KP/age/sex state mass allocation

### 2. Observation repair still injects arbitrary cascade values

Files:
- `src/epigraph_ph/phase3/rescue_core.py`

Examples:
- cascade ordering clamps: lines 1136-1143
- fallback repairs when targets are empty:
  - diagnosed fallback: line 1159
  - ART fallback: line 1161
  - suppression fallback: line 1163
  - testing fallback: line 1165

Why this matters:
- these are not posterior draws
- they are deterministic rescue rules
- they can dominate sparse years or sparse provinces

### 3. Initial latent state decomposition still uses manual fractions

Files:
- `src/epigraph_ph/phase3/rescue_core.py`

Examples:
- province observed state target construction: lines 1296-1306
- especially:
  - lost-share formula: line 1301

Why this matters:
- the split from observed cascade shares into `D`, `A`, `V`, `L` still depends on fixed fractions
- this is not learned from data
- it affects the initial condition of the semi-Markov chain

### 4. CD4 overlay is still hand-authored

Files:
- `src/epigraph_ph/phase3/rescue_core.py`

Examples:
- `DEFAULT_CD4_PRIOR`: line 100
- `_build_cd4_overlay`: starts line 1474

Why this matters:
- CD4 layering is not estimated through a coherent observation model
- the current overlay is a rule-based decoration on top of the state space

## Medium-severity issues

### 5. Loss function weights are still manually tuned

Files:
- `src/epigraph_ph/phase3/rescue_core.py`

Examples:
- calibration override defaults: lines 2301-2307
- optimizer learning rate: line 2458
- regularization coefficients: lines 2596-2608
- total loss coefficients: lines 2609-2619

Why this matters:
- this is standard engineering practice for now
- but these coefficients are still hand-set, not inferred or calibrated through a principled hyperprior layer

### 6. Forecast output stack still contains fallback constants

Files:
- `src/epigraph_ph/phase3/rescue_core.py`

Examples:
- forecast helper starts line 2772
- fallback VL share: line 2780
- fallback documented-given-test share: line 2781
- death proxy: line 2796

Why this matters:
- these affect exported forecast summaries and backtest comparisons
- they are not all directly tied to estimated stochastic processes

### 7. Archetypes are still prototype-driven, not learned from first principles

Files:
- `src/epigraph_ph/core/province_archetypes.py`

Examples:
- archetype library starts line 40
- manual reporting/prior shifts: lines 57-158
- synthetic library generation: starts line 243

Why this matters:
- archetypes are useful, but current archetype definitions are curated by hand
- they are closer to a structured prior library than to a learned latent class model

## Low-severity but important boundary issues

### 8. Node graph is explicitly heuristic

Files:
- `src/epigraph_ph/core/node_graph.py`

Examples:
- fixed node priors: line 13
- Bayesian-lite scoring: line 101
- raw signal formulas: lines 241-269
- veto/bonus/penalty caps: lines 320-327

Interpretation:
- this is a runtime assurance / decision-scoring layer
- it should never be presented as probabilistic epidemic inference

### 9. Phase 4 policy engine is a heuristic controller, not a defensible stochastic control model yet

Files:
- `src/epigraph_ph/phase4/pipeline.py`

Examples:
- `CHANNEL_EFFECTS`: line 40
- `DEFAULT_THRESHOLDS`: line 52
- `DEFAULT_SATURATION`: line 53
- simulated control transition perturbation: lines 144-185
- regional scoring formulas: lines 348-354
- scalar objective weights: line 474

Interpretation:
- Phase 4 remains a heuristic research scaffold
- it is not yet defensible as a formal RL/MPC policy engine for publication claims

## What is acceptable today

These are acceptable if they are described honestly:
- Phase 0 retrieval and evidence synthesis
- Phase 1 robust scaling and alignment
- Phase 2 constrained graph screening
- Phase 3 semi-Markov HIV core with explicit practical priors
- node graph as runtime assurance / decision support
- Phase 4 as blocked or exploratory

## What should change before strong publication claims

1. Replace hand-authored subgroup priors with a proper generative subgroup prior model or data-calibrated hyperprior layer.
2. Replace deterministic observation repair with explicit missing-observation latent variables.
3. Replace manual initial-state decomposition with a probabilistic initialization model.
4. Replace manual CD4 overlay rules with a fitted CD4 observation/emission model.
5. Move tuned loss coefficients into documented hyperparameter calibration experiments.
6. Keep node graph and Phase 4 clearly labeled as decision heuristics unless rebuilt as a true stochastic control layer.

## Safe paper language right now

Acceptable:
- "hierarchical observation-first semi-Markov HIV state-space core"
- "heuristic decision-support sidecar"
- "runtime-assurance node graph"
- "exploratory policy layer"

Not acceptable:
- "fully probabilistic end-to-end platform"
- "entire pipeline is semi-Markov"
- "policy optimizer is a calibrated stochastic control system"
