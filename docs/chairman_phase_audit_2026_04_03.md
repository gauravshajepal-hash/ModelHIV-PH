# Chairman Audit: Phases 0, 1, 15, and 2

Date: 2026-04-03
Repo: `D:\EpiGraph_PH`
Mode: same-model council, chairman synthesis
Selected autoresearch variant: `evidence-to-model-loop`

## Executive Judgment

The repo has crossed an important threshold. It is no longer mainly a heuristic feature-engineering system. It is now a staged scientific inference stack:

- `Phase 0` builds an evidence registry with explicit direct/proxy/context roles and source adjudication.
- `Phase 1` normalizes that evidence into a modeling representation and exposes observability.
- `Phase 15` performs mixed-frequency latent-state inference with bottom-up aggregation.
- `Phase 2` learns temporal dependency structure over latent innovations and now separates direct sparse edges from low-rank hidden shared structure.

That is real progress.

The corresponding warning is equally important: the stack is now mathematically sophisticated enough to produce outputs that look more certain, more structural, and more mechanistic than the evidence can yet support.

The correct global interpretation today is:

- the repo is `artifact-complete` for these phases in a meaningful engineering sense,
- but it is not yet `scientifically trusted` in the full sense,
- because semantic gold standards, identifiability benchmarks, uncertainty calibration, and adversarial falsification are still incomplete.

The strongest recurring council conclusion was:

`Phase 0` and `Phase 1` are the strongest parts operationally.

`Phase 15` and `Phase 2` are now directionally correct mathematically, but they are still partially identified and must be interpreted as latent predictive structure under regularization, not discovered mechanism.

## What Is Already Strong

### Phase 0

Strengths:

- Direct vs contextual evidence is now explicit rather than implicit.
- Structured-source extraction is reproducible on cached/local sources.
- HARP archive selection now surfaces and adjudicates source conflicts rather than flattening them.
- Measurement manifests and sign-prior artifacts create a workable evidence boundary.

Why this matters:

- The repo now has a real evidence layer instead of a mixed bag of rows with unclear semantic status.

### Phase 1

Strengths:

- Normalization is more disciplined than naive tensor building.
- The denominator-aware and quality-weighted tensor build is defensible.
- Observability gating is a major improvement. It blocks some variables from entering later likelihoods when there is no real support.

Why this matters:

- The repo now distinguishes “interesting concept” from “usable measurement.”

### Phase 15

Strengths:

- The mixed-frequency latent-state design is the correct mathematical center of the stack.
- Bottom-up national aggregation from provinces is the right direction.
- Burden-weight learning, sign-constrained loadings, and explicit observation operators are major improvements over the old scaffold.

Why this matters:

- This phase is now trying to solve the right problem, not the wrong problem more cleverly.

### Phase 2

Strengths:

- The new latent temporal graph is the right replacement path.
- Working on latent innovations is much better than learning same-time structure over blended observables.
- The sparse-plus-low-rank split is scientifically more honest than forcing all dependency into one graph.
- Province, region, and national multiscale support are now exposed separately.

Why this matters:

- Phase 2 now reflects the actual latent-state math of the repo rather than an earlier pre-latent design.

## Strongest Scientific Risks

### 1. Construct Mismatch Under Extreme Missingness

This is still the largest scientific risk.

Even with many direct rows, the aligned province-month information regime remains extremely sparse. In the current run, the aligned tensor missingness remains near-total in practice. That means later phases depend heavily on:

- support geometry
- pooling
- priors
- operator design
- regularization

This does not make the model invalid. It means the outputs are partially identified and should be treated that way.

### 2. Semantic Correctness Is Still Behind Internal Consistency

Structured parity and HARP adjudication are strong.

But the repo still lacks a literal gold-standard benchmark for unstructured free-text extraction. So the system can prove:

- deterministic replay
- conflict surfacing
- internal consistency

without yet proving:

- correct canonical variable assignment
- correct geography binding
- correct time binding
- correct measurement-role assignment

for all narrative and OCR-heavy rows.

### 3. Latent Block Semantics Are Not Yet Cleanly Identified

Some `Phase 15` blocks are well motivated, but several remain:

- proxy-dominated
- unevenly supported
- heavily pooled
- only weakly identified at the subnational level

A mathematically coherent latent summary is not automatically a scientifically clean construct.

### 4. Regional Pooling Is Likely Too Strong

In the current live fit, regional precision saturates at the configured ceiling across retained blocks. That is a warning sign that:

- provincial heterogeneity may be getting washed out
- regional center bias can propagate downward
- province-month outputs can look smoother and more resolved than the evidence warrants

### 5. Phase 2 Edges Are Easy To Over-Interpret

The new Phase 2 edges are much better than the old observable DAG.

But they are still:

- lagged dependency estimates on learned latent innovations
- regularization-dependent
- upstream-model-dependent
- not causally identified intervention effects

They should be read as temporal hypotheses, not mechanistic arrows.

## Direct Evidence Table

| Phase | Direct evidence object | Present status | Main limitation |
| --- | --- | --- | --- |
| Phase 0 | structured numeric indicators, official anchor rows, HARP rows, PhilHealth/PSA/WDI/mobility records | strong and reproducible | semantic row correctness for free-text/OCR is not yet gold-standard validated |
| Phase 1 | normalized rows, aligned tensor cells, denominator-aware values, observability counts | operationally strong | support counts are not the same as information content or identifiability |
| Phase 15 | measurement rows through explicit support operators over latent states | mathematically much improved | annual/aggregate support is still only approximately handled and some blocks are weakly identified |
| Phase 2 | lagged latent innovation structure | now scientifically legible | edge meaning is predictive/structural, not causal/mechanistic |

## Contextual Evidence Table

| Phase | Contextual evidence object | Correct use | Misuse to avoid |
| --- | --- | --- | --- |
| Phase 0 | literature seeds, ontology tags, sign priors | priors on sign, block membership, search coverage | treating narrative support as direct measurement |
| Phase 1 | evidence weights, bias penalties, preprocessing choices | stabilization and observability diagnostics | pretending these are empirically learned noise models |
| Phase 15 | sign priors, shrinkage priors, burden priors | regularization under sparse data | reading prior-driven local detail as recovered truth |
| Phase 2 | sparsity penalties, low-rank penalties, target block definitions | edge screening and structural regularization | reading retained edges as validated mechanisms |

## Phase-By-Phase Opinion

### Phase 0

Current opinion:

- Strong as an evidence-registry and adjudication system.
- Not yet strong enough to support “gold standard extraction” language for unstructured text.

Main improvements:

1. Add a hand-labeled extraction benchmark spanning structured tables, OCR text, and free-text literature.
2. Move canonicalization toward a probabilistic claim graph over `(canonical_name, geo, time, unit, source)` rather than deterministic collapse alone.
3. Version extraction semantics aggressively so ontology or canonicalization changes invalidate downstream caches.
4. Attach explicit observation-operator metadata earlier, at row creation time, not only downstream.
5. Add row-level semantic confidence and conflict posterior scoring.

### Phase 1

Current opinion:

- Strong as evidence normalization and observability accounting.
- Still too heuristic to be treated as a learned noise model.

Main improvements:

1. Replace expert-set evidence weights and bias penalties with an estimated observation-noise model wherever possible.
2. Separate “support exists” from “support is informative enough for this downstream use.”
3. Carry denominator and transformation uncertainty more explicitly.
4. Add calibration checks for the weighting system against audited source subsets.
5. Make missingness geometry a first-class output, not just a scalar missingness rate.

### Phase 15

Current opinion:

- The best mathematical direction in the repo.
- Still a strong approximation rather than final inference.

Main improvements:

1. Move from alternating MAP smoothing toward joint variational or EM-style hierarchical inference.
2. Run weaker-pooling sensitivity fits and treat pooling saturation as a warning state.
3. Strengthen burden-weight supervision with more burden-specific evidence, not only proxy families.
4. Add block-level construct-validity gates:
   - no mostly-floor or mostly-ceiling loading patterns
   - no single-source-family domination
   - no unstable sign behavior under resampling
5. Treat province-level detail as uncertainty objects first, point trajectories second.

### Phase 2

Current opinion:

- The new latent temporal graph is the correct path.
- The old same-time observable DAG should now be considered a secondary diagnostic path.

Main improvements:

1. Propagate `Phase 15` state uncertainty into `Phase 2`, rather than using only point latent trajectories.
2. Extend from one-lag structure to multi-lag or lag-selection with stability control.
3. Add synthetic recovery benchmarks where the true sparse direct structure and hidden shared structure are known.
4. Add source-ablation, time-window shift, and adversarial falsification tests for edges.
5. Keep direct sparse edges and hidden-driver rows semantically separate everywhere downstream.

## What Must Not Be Over-Claimed

The final synthesis must be explicit about these:

1. `Phase 15` block states are latent predictive summaries under partial identification, not directly observed determinants.
2. Learned burden weights are not yet gold-standard province burden shares.
3. `Phase 15` loading magnitudes are measurement parameters, not causal effect sizes.
4. `Phase 2` direct sparse edges are stable lagged dependencies, not intervention-valid causal arrows.
5. `Phase 2` hidden-driver rows are residual shared structure, not named latent mechanisms.

## Immediate Checks The Council Agrees Must Happen First

1. Build a literal gold-standard benchmark for unstructured Phase 0 extraction.
2. Add end-to-end synthetic recovery for:
   - latent blocks
   - aggregation weights
   - direct-vs-hidden temporal structure
3. Add uncertainty calibration for `Phase 15` posterior outputs.
4. Add weaker-pooling sensitivity fits and report when conclusions move materially.
5. Add adversarial falsification tests for `Phase 2` edges.
6. Fix trust in manifest summaries so stage-complete metadata cannot disagree with populated downstream artifacts.

## Interventions Worth Trying

1. Probabilistic conflict/adjudication in Phase 0.
2. Inferred observation-noise learning in Phase 1.
3. Joint hierarchical inference in Phase 15.
4. Uncertainty-aware multi-lag sparse-plus-low-rank temporal graphing in Phase 2.
5. Trust-region reporting for province-month outputs:
   - data-driven
   - shrinkage-dominant
   - prior-dominant

## Interventions To Avoid

1. Do not push more complexity into the old observable DAG and call it causality.
2. Do not collapse contextual literature evidence into the likelihood.
3. Do not treat hidden-driver rows as intervention targets.
4. Do not present all province-month outputs as equally trustworthy.
5. Do not optimize downstream prediction while letting evidence or identifiability gates weaken.

## AutoResearch Handoff

Variant: `evidence-to-model-loop`

### Mutation Units

1. `phase0`: probabilistic row adjudication and semantic benchmark
2. `phase1`: learned observation-noise and coverage-aware eligibility
3. `phase15`: joint posterior inference and pooling-sensitivity analysis
4. `phase2`: uncertainty-aware multi-lag latent graph with falsification tests

### Evaluation Harness

The trusted harness should contain:

1. structured parity and HARP adjudication
2. hand-labeled free-text extraction benchmark
3. synthetic mixed-frequency latent recovery benchmark
4. uncertainty calibration benchmark
5. temporal-edge recovery and falsification benchmark

### Keep-Or-Revert Rule

Keep a change only if:

- extraction audit does not regress,
- semantic benchmark does not regress,
- synthetic recovery improves,
- uncertainty calibration does not worsen,
- edge stability improves under perturbation,
- and downstream behavior improves without making interpretation less honest.

## Bottom Line

The repo is now close to a serious scientific modeling stack.

But the mathematically sophisticated parts are ahead of the evidential guarantees.

So the right statement today is:

`The pipeline is scientifically promising and structurally much improved, but it still produces partially identified latent summaries and temporal hypotheses rather than validated mechanistic truth.`

That is not a criticism. It is the correct scientific posture for the current codebase.
