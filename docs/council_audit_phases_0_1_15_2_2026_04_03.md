# Council Audit: Phases 0, 1, 15, and 2

Date: 2026-04-03
Repo: `D:\EpiGraph_PH`
Council mode: same-model structured synthesis
Selected autoresearch variant: `evidence-to-model-loop`

## Scope

This note audits the current mathematical and scientific design of:

- `phase0`: extraction, evidence typing, and source adjudication
- `phase1`: normalization, tensor construction, and observability
- `phase15`: latent block inference and mixed-frequency state-space approximation
- `phase2`: graph learning and target blanket construction

The standard here is not “works as software.” The standard is “scientifically defensible under sparse, heterogeneous HIV evidence.”

## Executive View

The repo is now substantially stronger than a heuristic feature-engineering stack. The important shift is real:

- `phase0` now distinguishes direct indicators from contextual evidence instead of collapsing them.
- `phase1` now exposes observability explicitly.
- `phase15` now uses a mixed-frequency latent-state model rather than a pure heuristic score builder.
- `phase2` now has a latent temporal graph path over `phase15` states and separates direct sparse edges from shared hidden-driver structure.

That said, the system is still not a fully joint probabilistic model. It remains a staged approximation. The main scientific risk is not “bad code.” The main scientific risk is over-interpreting intermediate artifacts as if they were uniquely identified truths.

## Direct Evidence Table

| Phase | Direct Evidence Object | Current Strength | Main Risk |
| --- | --- | --- | --- |
| Phase 0 | structured numeric indicators, official anchors, HARP archive rows, PSA/PhilHealth portal rows | strong and improving | canonicalization and geo-time binding errors still dominate failure if they slip through |
| Phase 1 | normalized numeric rows, aligned tensor, denominator-aware transforms, observability counts | strong enough for evidence bookkeeping | noise model is still partly heuristic |
| Phase 15 | latent measurement rows and mixed-frequency support operators | medium-to-strong | block states are still MAP approximations, not full joint posterior inference |
| Phase 2 | latent temporal graph over `phase15` states | medium | graph structure is penalty-sensitive and not yet externally validated |

## Contextual Evidence Table

| Phase | Contextual Evidence Object | Correct Current Use | Misuse To Avoid |
| --- | --- | --- | --- |
| Phase 0 | literature seeds, ontology tags, sign priors | priors on block membership, sign, and search coverage | treating narrative support as measurement |
| Phase 1 | evidence weights, bias penalties, preprocessing choices | row weighting and observability diagnostics | pretending these are learned uncertainties |
| Phase 15 | sign priors, shrinkage priors, aggregation priors | stabilizing latent inference under sparse data | interpreting prior-driven state detail as recovered truth |
| Phase 2 | target block list, sparsity penalties, hidden-rank thresholds | regularization and edge screening | interpreting threshold-selected edges as robust biology or policy pathways |

## Current Run Facts

The current cached run `artifacts/runs/smoke-latent-blocks` materially supports some claims:

- `phase1/latent_observability_audit.json` shows real support for several structural and mobility indicators:
  - `poverty_rate`: `9805` direct rows, `5384` province-support rows
  - `economic_access_constraint`: `199` direct rows, `165` province-support rows
  - `mobility_network_mixing`: `561` direct rows, `495` regional-support rows
  - `policy_implementation_weakness`: `53` direct rows
  - `health_system_reach`: `49` direct rows

- `phase15/phase15_v2_fit_summary.json` shows:
  - `block_count = 4`
  - `measurement_row_count = 2379`
  - `weight_source = learned_burden_softmax`
  - aggregation supervision count `= 2792`
  - but `weight_dispersion = 0.0121`, which means the burden-weight learner is active yet still weakly spatially differentiated

- `phase2/latent_temporal_graph_bundle.json` now shows both direct and hidden structure:
  - province: `5` direct edges, `2` hidden-driver pairs, hidden rank `2`
  - region: `4` direct edges, `1` hidden-driver pair, hidden rank `1`
  - national: `6` direct edges, `2` hidden-driver pairs, hidden rank `3`

This is scientifically better than before. It is also a warning: shared latent temporal structure remains substantial even after `phase15`.

## Phase 0 Opinion

### What is strong

- The repo now treats `direct_indicator`, `proxy_indicator`, and `context_only` as genuinely different objects.
- Structured-source extraction from HARP, PhilHealth, PSA, WDI, Google mobility, and local PDFs materially improved promotable indicators.
- The extraction audit and adjudication table moved `phase0` from “retrieval plus hopeful parsing” toward a real evidence registry.

### What remains weak

- Canonicalization is still too fragile to be considered mathematically closed. It is still largely deterministic string-and-rule collapse.
- Geo-time binding remains one of the highest-risk failure modes, especially when annual or regional values are expanded or aliased.
- The current validation stack can prove internal consistency better than semantic correctness. That is necessary but not sufficient.

### Highest-value improvements

1. Replace deterministic canonical collapse with a probabilistic conflict graph over `(canonical_name, geo, time, unit, source)` claims.
2. Introduce a small hand-labeled extraction benchmark with exactness labels at row level.
3. Attach explicit observation operators already in `phase0`, not only later in `phase15`.
4. Add row-level duplicate/conflict posterior scoring instead of rule precedence alone.
5. Version extraction semantics aggressively; any ontology or canonicalization change should invalidate downstream caches.

## Phase 1 Opinion

### What is strong

- `phase1` now does what it should do in a sparse scientific pipeline: normalize, weight, align, and report observability.
- The direct-vs-contextual split survives into the audit layer.
- The denominator tensor and quality-weight machinery are better than naive standardization.

### What remains weak

- Evidence weights and bias penalties are still expert-set rather than inferred from error behavior.
- The aligned tensor remains a convenience representation, not a literal observation model.
- Some preprocessing choices still compress uncertainty too early.

### Highest-value improvements

1. Replace rule-based evidence weights with an estimated observation-noise model.
2. Carry denominator uncertainty explicitly rather than only dividing and clipping.
3. Make missingness handling more operator-aware and less tensor-cell centric.
4. Separate “support exists” from “support is comparable” more cleanly in the tensor audit.
5. Add calibration checks for the evidence-weight/bias-penalty scheme against audited source subsets.

## Phase 15 Opinion

### What is strong

- This is now the mathematical center of the repo.
- The mixed-frequency latent-state design is the correct place for observation operators, sign constraints, aggregation, and temporal smoothing.
- Bottom-up national aggregation from provinces is the right direction.
- Learned signed loadings and burden-weight learning are materially better than the earlier heuristic scaffold.

### What remains weak

- It is still an alternating MAP smoother, not full joint posterior inference.
- Regional precision still saturates at the configured ceiling in the current run, which is a warning sign of over-pooling.
- Burden weights are now learned but still weakly dispersed. That means the machinery is right, but supervision remains thin.
- The model still relies on several structural approximations that are better described as regularized engineering than final statistical inference.

### Highest-value improvements

1. Move from alternating MAP updates to joint variational or EM-style inference with uncertainty carried across states, loadings, and precisions.
2. Use external burden priors or stronger supervision for province aggregation weights.
3. Relax the regional precision ceiling and estimate pooling hyperpriors more flexibly.
4. Add posterior sensitivity analysis for sign priors and burden priors.
5. Treat province-level detail as posterior uncertainty objects, not only point state values.

## Phase 2 Opinion

### What is strong

- The new latent temporal graph is the correct replacement path.
- Operating on `phase15` innovations is mathematically much closer to what modern state-space and dynamic graph methods would do.
- Separating direct sparse edges from low-rank hidden-driver structure is a major improvement over a single dense temporal graph.
- Multiscale support aggregation across province, region, and national levels is the right idea.

### What remains weak

- The graph is still a penalized estimator without external causal validation.
- Penalty balance strongly affects whether structure appears direct, hidden, or absent.
- The current temporal graph uses one-lag structure and point estimates from `phase15`; it does not yet consume state uncertainty.
- The old same-time observable DAG is still present, but mathematically it should now be treated as secondary diagnostics, not primary structural truth.

### Highest-value improvements

1. Propagate `phase15` state uncertainty into `phase2` edge confidence.
2. Move from one-lag to multi-lag or lag-selection with stability control.
3. Add synthetic recovery benchmarks where the true sparse and hidden temporal structure is known.
4. Evaluate edge reproducibility under time-window shifts and source-ablation tests.
5. Prevent downstream consumers from treating hidden-driver pairs as direct intervention targets.

## Strongest Scientific Threats

1. **Construct drift across phases**
   The object called a “factor,” “latent block,” or “edge” changes meaning across phases. This is manageable, but only if documentation and downstream usage remain explicit.

2. **Over-interpretation of province detail**
   Province-month outputs can look more precise than they truly are. Sparse-data smoothing can produce visually convincing but weakly identified local structure.

3. **Conflation of causal and predictive structure**
   Even the improved `phase2` graph is still a regularized temporal dependence model, not a proven causal DAG.

4. **Prior-dominant stability**
   Several results may be stable because priors and penalties are strong, not because evidence is rich.

5. **Silent success metrics**
   Internal consistency and artifact completeness can look excellent even when the scientific interpretation is too strong.

## What Should Be Done Next

### Interventions worth trying

1. Build a synthetic end-to-end benchmark where:
   - extraction errors are injected,
   - observation operators are mixed-frequency,
   - latent blocks have known sparse and hidden temporal structure,
   - and downstream recovery is scored phase by phase.

2. Add posterior-sensitivity reports for:
   - sign priors,
   - aggregation priors,
   - sparsity penalties,
   - hidden-rank penalties.

3. Treat `phase2` direct edges and hidden-driver rows as separate downstream objects with different semantics.

4. Add a “trust region” report for province detail:
   which outputs are data-driven, shrinkage-dominant, or prior-dominant.

5. Unify the repo around one observation-operator formalism so `phase0`, `phase1`, `phase15`, and `phase2` all refer to the same measurement semantics.

### Interventions to avoid

1. Do not push more complexity back into the old observable NOTEARS DAG and call it causality.
2. Do not collapse contextual literature evidence into the likelihood.
3. Do not interpret hidden-driver pairs as direct intervention levers.
4. Do not treat high-resolution province-month trajectories as equally trustworthy across geographies.
5. Do not optimize downstream forecast score without preserving extraction and identifiability audits.

## Direct Evidence vs Contextual Evidence Judgment

The repo is now strongest when it is honest about this split:

- direct evidence should drive likelihoods,
- contextual evidence should drive priors and admissibility,
- sparse-data smoothing should produce uncertainty-aware latent structure,
- and graph outputs should be interpreted as regularized temporal dependence unless externally validated.

This is the correct scientific posture for the current codebase.

## AutoResearch Handoff

Variant: `evidence-to-model-loop`

### Trusted evaluation gates

1. extraction audit stays passing
2. synthetic recovery benchmark for latent blocks and temporal graph
3. posterior sensitivity summaries for priors and penalties
4. stability of `phase2` sparse edges under resampling and time-window shift
5. downstream long-horizon performance improves without widening scientific overclaim

### Mutation units

1. `phase0`: conflict-graph row adjudication
2. `phase1`: inferred observation-noise model
3. `phase15`: joint posterior inference or stronger uncertainty propagation
4. `phase2`: uncertainty-aware multi-lag sparse-plus-low-rank temporal graph

### Keep-or-revert rule

Keep only if:

- synthetic recovery improves,
- edge stability improves,
- prior sensitivity narrows or becomes more interpretable,
- and downstream behavior improves without making the outputs scientifically less honest.

## Bottom Line

This repo is now close to a serious scientific modeling stack.

It is no longer mainly a heuristic feature-engineering pipeline.
But it is also not yet a fully identified, fully joint probabilistic scientific model.

The highest-value next work is not “add another model.”
It is:

- better uncertainty accounting,
- better identifiability diagnostics,
- better synthetic recovery evaluation,
- and sharper separation between direct structure and hidden shared structure.
