# Validity Skeptic Memo on Phases 0, 1, 15, and 2

Date: 2026-04-03  
Repo: `D:\EpiGraph_PH`  
Run audited: `smoke-latent-blocks`

## Scope

This memo audits the scientific validity of the current Phase 0, 1, 15, and 2 stack, with emphasis on:

- confounding
- endpoint mismatch
- construct mismatch
- missingness
- transportability
- identifiability
- places where the math can produce scientifically misleading conclusions

Primary code inspected:

- [phase0/boundary_models.py](/D:/EpiGraph_PH/src/epigraph_ph/phase0/boundary_models.py#L597)
- [phase0/evidence_artifacts.py](/D:/EpiGraph_PH/src/epigraph_ph/phase0/evidence_artifacts.py#L34)
- [phase1/pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase1/pipeline.py#L219)
- [phase1/latent_observability.py](/D:/EpiGraph_PH/src/epigraph_ph/phase1/latent_observability.py#L13)
- [phase15/v2_engine.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/v2_engine.py#L536)
- [phase15/v2_engine.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/v2_engine.py#L1238)
- [phase15/v2_engine.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/v2_engine.py#L1478)
- [phase2/latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py#L87)
- [phase2/latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py#L185)
- [phase2/latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py#L229)

Primary run artifacts inspected:

- [structured_numeric_candidate_summary.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase0/extracted/structured_numeric_candidate_summary.json)
- [philhealth_portal_metric_summary.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase0/extracted/philhealth_portal_metric_summary.json)
- [normalization_report.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase1/normalization_report.json)
- [latent_observability_audit.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase1/latent_observability_audit.json)
- [phase15_v2_fit_summary.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_fit_summary.json)
- [phase15_v2_indicator_parameters.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_indicator_parameters.json)
- [phase15_v2_aggregation_weights.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_aggregation_weights.json)
- [latent_temporal_graph_bundle.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase2/latent_temporal_graph_bundle.json)
- [latent_temporal_phase3_target_blankets.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase2/latent_temporal_phase3_target_blankets.json)
- [extraction_quality_audit.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/analysis/extraction_quality_audit.json)

## Executive View

The pipeline is materially better engineered than a naive evidence-to-forecast stack. It now has:

- explicit evidence-role separation
- extraction audit and HARP adjudication
- observability gating
- mixed-frequency latent-state inference
- bottom-up aggregation direction
- a temporal graph that operates on latent innovations instead of same-time observables

However, the main scientific problem has not disappeared. It has shifted:

- Phase 0 and 1 create a very large quantity of direct numeric rows, but they are extremely uneven in construct meaning and support geometry.
- Phase 15 converts those rows into a small number of latent blocks, but several blocks remain only weakly identified and are dominated by proxy families rather than direct block-specific measures.
- Phase 2 then learns temporal dependencies on those latent blocks, so any construct mismatch or unmodeled source bias in Phase 15 propagates directly into the graph.

The current stack is best interpreted as a **measurement-aware latent dependency model under sparse heterogeneous evidence**, not as a discovered mechanistic causal model of HIV dynamics.

## 1. Strongest Validity Threats

### 1. Construct mismatch is still the central threat

Phase 0 is now good at turning structured sources into rows, but the resulting evidence bank is heavily dominated by a few source families. In the audited run, Phase 1 reports `11387` direct indicators, but the domain mix is extremely imbalanced: `economics = 10097`, `mobility = 1122`, `philhealth = 119`, while many HIV-proximal constructs remain thin or indirect in [normalization_report.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase1/normalization_report.json).

This matters because Phase 15 block semantics are assigned upstream in [phase0/evidence_artifacts.py](/D:/EpiGraph_PH/src/epigraph_ph/phase0/evidence_artifacts.py#L34), then used downstream as if the mapped indicators actually instantiate the block. That is defensible for some indicators, but weak for others. For example, `structural_barrier_pressure` currently absorbs poverty, expenditure burden, cash instability, education, and PhilHealth-related access proxies. Those may all co-vary with a barrier construct, but they are not interchangeable measurements of a single scientifically clean latent.

Risk:

- the model may estimate a stable latent score that is mathematically coherent but scientifically heterogeneous
- downstream edges can then look like mechanisms when they are partly artifacts of block construction

### 2. Missingness remains extreme, despite the large row count

Phase 1 reports `missing_mask_fraction = 0.999921` in [normalization_report.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase1/normalization_report.json). This is the single most important reality check.

The system has many rows, but almost no dense province-month-by-indicator support. That means the model is still operating in a regime where:

- geometry of observation support matters more than raw row count
- pooling and priors do a large fraction of the work
- latent trajectories can become smoother and more stable than the raw information actually warrants

The Phase 15 v2 observation model is better than annual fan-out, but it still depends on the support-cell operator plus strong temporal and regional regularization in [phase15/v2_engine.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/v2_engine.py#L1238). In sparse settings, this can create overconfident-looking latent structure even when information is weak.

### 3. Identifiability of several Phase 15 blocks is weak

The fit summary in [phase15_v2_fit_summary.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_fit_summary.json) shows only `4` retained blocks and strong differences in effective support:

- `care_access_continuity`: `row_count = 25`
- `suppression_capacity`: `row_count = 130`
- `mobility_exposure_pressure`: `row_count = 1122`
- `structural_barrier_pressure`: `row_count = 1102`

This is not balanced identification.

The indicator parameter artifact confirms the issue. In [phase15_v2_indicator_parameters.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_indicator_parameters.json), many `suppression_capacity` indicators are pinned at the loading floor `0.05`, while mobility and structural indicators hit the ceiling `4.0`. That means:

- some indicators are effectively uninformative under the current fit
- some blocks are driven by very strong proxy families
- block comparability is limited

This is not a software failure. It is a scientific identifiability warning.

### 4. Regional pooling is likely too strong and can hide subnational heterogeneity

All four retained blocks report `regional_precision = 25.0` in [phase15_v2_fit_summary.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_fit_summary.json), which is the configured ceiling behavior implied by the posterior precision update in [phase15/v2_engine.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/v2_engine.py#L1312).

That is a red flag. It means the current data + prior structure repeatedly wants maximal regional pooling. In a country where reporting intensity, urban concentration, and service architecture are highly uneven, maximal pooling can easily:

- wash out real provincial divergence
- propagate regional center bias
- make inferred province-month states look more transportable than they are

### 5. Burden-weight learning is now active, but still weakly interpretable

The aggregation model in [phase15/v2_engine.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/v2_engine.py#L536) now uses learned softmax burden weights, and [phase15_v2_aggregation_weights.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_aggregation_weights.json) shows real optimization gain with `2792` supervision samples. That is a meaningful advance.

But the learned national weights still have extremely small dispersion, and the supervision set is proxy-heavy: poverty, economic access, mobility, prevention coverage, policy weakness, health-system reach. Those are not direct burden observations. So the aggregation model is learning a **burden proxy weighting**, not a validated epidemiologic burden distribution.

Risk:

- the bottom-up aggregation direction is correct
- the actual province weights are still only weakly transportable and not yet a gold-standard burden surface

### 6. Phase 2 direct edges are not uniquely identified causal effects

Phase 2 now operates on latent innovations in [phase2/latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py#L185), then decomposes cross-block temporal structure into:

- a sparse direct component
- a low-rank hidden shared component

using [phase2/latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py#L87).

This is mathematically much better than a same-time observable DAG. But it still does **not** identify causal direction in a scientific sense. The sparse-vs-low-rank split is regularization-dependent. The “direct” edges are conditional dependence claims after:

- Phase 15 block construction
- Phase 15 pooling
- persistence subtraction using estimated `phi`
- sparse-plus-low-rank penalization
- bootstrap thresholding

So a retained edge such as `mobility_exposure_pressure -> structural_barrier_pressure` in [latent_temporal_graph_bundle.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase2/latent_temporal_graph_bundle.json) should be treated as a stable lagged dependency in the learned latent system, not as established mechanism.

### 7. Endpoint mismatch is still serious

Many extracted variables are upstream determinants or service proxies, while the HIV anchors and most evaluation targets are cascade-oriented. That means the system is still trying to explain or forecast cascade outcomes using determinants that are only partially aligned to those endpoints.

The risk is strongest when:

- Phase 15 block names sound mechanistic
- Phase 2 edges sound causal
- Phase 3 later uses those blankets to influence transition hazards

At that point, the stack can sound more mechanistic than the evidence really supports.

## 2. Outputs That Should Not Be Interpreted Causally or Mechanistically

The following should **not** be presented as discovered causal mechanisms without much stronger validation:

1. Phase 15 latent block levels as “true underlying determinants”  
   The block states are model-based summaries of heterogeneous indicators, not direct measurements.

2. Phase 15 burden weights as “true province HIV burden shares”  
   They are learned from burden-related proxies, not from gold-standard province burden labels, in [phase15_v2_aggregation_weights.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_aggregation_weights.json).

3. Phase 15 loading magnitudes as mechanistic effect sizes  
   The loadings are measurement parameters under regularization, not intervention elasticities. Floor and ceiling behavior in [phase15_v2_indicator_parameters.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/phase15/phase15_v2_indicator_parameters.json) is a warning.

4. Phase 2 sparse temporal edges as causal arrows  
   They are stable lagged dependencies in latent innovations after decomposition, not intervention effects.

5. Phase 2 hidden-driver pairs as identified hidden mechanisms  
   They are residual low-rank shared structure, not uniquely named latent causes.

6. Phase 2 Phase-3 target blankets as mechanistic sufficient sets  
   They are candidate dependency neighborhoods for downstream modeling, not scientifically sufficient causal adjustment sets.

## 3. Defensible Assumptions vs Weak Assumptions

### Defensible

1. Explicit evidence-role separation is a major improvement  
   The distinction between `direct_indicator`, `proxy_indicator`, and `context_only` in [phase0/evidence_artifacts.py](/D:/EpiGraph_PH/src/epigraph_ph/phase0/evidence_artifacts.py#L34) and [phase1/latent_observability.py](/D:/EpiGraph_PH/src/epigraph_ph/phase1/latent_observability.py#L47) is scientifically defensible and necessary.

2. HARP source adjudication is strong and should be preserved  
   The extraction audit shows the current run passes HARP consistency checks and surfaces disagreements explicitly in [extraction_quality_audit.json](/D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks/analysis/extraction_quality_audit.json). That is good scientific hygiene.

3. Bottom-up aggregation is a better structural direction than top-down forcing  
   The choice to aggregate province states upward is mathematically better than imposing national truth downward.

4. Innovation-based Phase 2 is better than same-time causal graphing  
   Removing own-block persistence before graph learning is a real conceptual improvement.

### Weak

1. “Direct indicator” does not automatically imply valid measurement of the intended latent construct  
   Directness of extraction is not the same as construct validity.

2. Support counts are not the same as information content  
   A variable with many repeated rows from one source family can dominate a block without being broadly informative.

3. Current block labels may overstate measurement coherence  
   Some blocks are semantically too broad for their current evidence mix.

4. Learned burden weights are still weakly transportable  
   The supervision surface is proxy-heavy and geographically uneven.

5. Sparse-plus-low-rank decomposition does not uniquely identify direct vs hidden causes  
   The split is helpful, but not ontologically unique.

## 4. Top 5 Safeguards or Redesigns

### 1. Add block-level construct validity gates

A block should only remain active if:

- at least one core indicator family is HIV-proximal rather than purely socioeconomic
- loadings are not mostly pinned at floor or ceiling
- sign conflict stays low across resamples
- removal of one dominant source family does not collapse the block

Without this, Phase 15 can produce stable but construct-misaligned latents.

### 2. Model missingness and observation geometry explicitly in evaluation

The current fit already uses support cells, but evaluation should also report:

- effective support entropy by block
- support concentration by source family
- province-month posterior uncertainty calibrated against support density
- leave-one-source-family-out degradation

This is necessary because the core data regime is still `0.999921` missing at the aligned tensor level.

### 3. Replace one-number regional pooling with adaptive, block-specific heterogeneity checks

The current repeated `regional_precision = 25.0` should trigger an automatic warning state. The model should:

- flag pooling saturation
- run a weaker-pooling sensitivity fit
- compare province rank stability and edge stability under that change

If conclusions change materially, the original fit should not be trusted mechanistically.

### 4. Introduce negative controls and adversarial falsification tests

For each retained block and Phase 2 edge, test:

- placebo indicators that should not belong to the block
- time-permuted indicators
- source-family-restricted fits
- holdout geographies

If the same graph or block emerges under these perturbations, confidence improves. If not, the current interpretation is too strong.

### 5. Separate predictive utility from scientific interpretation in all outputs

The repo should explicitly label outputs as one of:

- `measurement summary`
- `predictive latent`
- `dependency hypothesis`
- `causal/mechanistic claim`

At present, Phases 15 and 2 support the first three categories. They do not yet support the fourth.

## Bottom Line

The strongest scientific reading is:

- Phase 0 is now a credible evidence assembly and adjudication layer.
- Phase 1 is a reasonable normalization and observability layer, but the effective tensor support remains extremely sparse.
- Phase 15 is a serious latent-state inference engine, but several blocks are still only weakly identified and proxy-dominated.
- Phase 2 is now a serious temporal dependency model on latent innovations, but its direct edges are not mechanistic truth and should not be narrated that way.

The main thing that should change next is not more graph sophistication. It is stronger **construct validity discipline** and more explicit **falsification / robustness testing** against source-family dominance, pooling saturation, and proxy-driven transport failure.
