# Evidence Agent Memo: Phases 0, 1, 15, 2

Scope: evidence audit of extraction, normalization, latent-state inference, and latent temporal graph construction.

Repo examined:
- `D:\EpiGraph_PH\src\epigraph_ph\phase0\pipeline.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase0\evidence_artifacts.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase1\pipeline.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase1\latent_observability.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase1\normalization_helpers.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase15\v2_engine.py`
- `D:\EpiGraph_PH\src\epigraph_ph\phase2\latent_temporal_graph.py`

Run artifacts examined:
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\analysis\extraction_quality_audit.md`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase0\extracted\measurement_manifest.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase0\extracted\block_sign_priors.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase1\latent_observability_audit.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase15\phase15_v2_measurement_rows.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase15\phase15_v2_aggregation_weights.json`
- `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase2\latent_temporal_graph_bundle.json`

## 1. Strongest Evidence Strengths

1. Phase 0 now preserves the direct-vs-contextual distinction as an explicit artifact, instead of collapsing everything into one evidence pool. Candidate rows are converted into evidence rows, sign-prior rows, and a measurement manifest in `D:\EpiGraph_PH\src\epigraph_ph\phase0\evidence_artifacts.py:34`, `D:\EpiGraph_PH\src\epigraph_ph\phase0\evidence_artifacts.py:136`, and `D:\EpiGraph_PH\src\epigraph_ph\phase0\evidence_artifacts.py:191`, and those artifacts are written in `D:\EpiGraph_PH\src\epigraph_ph\phase0\pipeline.py` near the Phase 0 extract bundle. This is the right scientific boundary.

2. Structured-source extraction is internally reproducible. The current run-level audit reports exact parity for structured sources: emitted `11354` vs regenerated `11354`, with zero missing and zero extra rows, and PhilHealth portal summary match is `True`. See `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\analysis\extraction_quality_audit.md`.

3. HARP extraction is no longer silently flattening source conflicts. The same audit reports zero panel-vs-seed discrepancies, zero time mismatches, and explicit adjudication of alternative rows into `quarterly_snapshot`, `official_override`, and `model_update`. This is materially stronger than pretending there is one uncontested truth. See `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\analysis\extraction_quality_audit.md`.

4. Phase 1 observability is honest enough to stop some unsupported variables from entering later likelihoods. The eligibility logic in `D:\EpiGraph_PH\src\epigraph_ph\phase1\latent_observability.py:13` only marks variables as likelihood-eligible if they have direct indicators and actual national/annual/monthly support, or graph-eligible if they have direct or proxy rows and some geo support. The artifact confirms this behavior: `collective_risk_behavior` and `biological_progression_modifier` remain ineligible, while `congestion_travel_time`, `poverty_rate`, and `health_system_reach` are eligible. See `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase1\latent_observability_audit.json`.

5. Phase 15 v2 is evidence-aware in a way the earlier scaffold was not. Measurement rows are constructed from explicit support operators and carry `operator_kind`, `support_scope`, `time_support_mode`, `support_cells`, `measurement_role`, and source metadata in `D:\EpiGraph_PH\src\epigraph_ph\phase15\v2_engine.py:395`. This makes the latent-state likelihood legible. It is a major improvement over annual fan-out hidden inside preprocessing.

6. Phase 2 now separates direct temporal edges from hidden shared structure instead of overloading one graph with both jobs. The sparse-plus-low-rank decomposition is explicit in `D:\EpiGraph_PH\src\epigraph_ph\phase2\latent_temporal_graph.py:87`, the innovation construction is explicit in `D:\EpiGraph_PH\src\epigraph_ph\phase2\latent_temporal_graph.py:185`, and hidden-driver rows are surfaced separately in `D:\EpiGraph_PH\src\epigraph_ph\phase2\latent_temporal_graph.py:376`. This is scientifically much more defensible than a same-time observable DAG.

## 2. Strongest Evidence Weaknesses

1. Phase 0 is now dominated by structured numeric evidence, but that dominance is highly unbalanced across blocks. In the current manifest, `direct_indicator = 11387` and `context_only = 8`, which sounds strong, but `structural_barrier_pressure` alone contributes `10113` rows while `care_access_continuity` has only `5` and `testing_engagement` only `10`. See `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase0\extracted\measurement_manifest.json`. So the evidence base is wide, but not balanced.

2. The contextual layer is too small and too noisy to support strong semantic claims by itself. `block_sign_priors.json` has only `36` rows total, and some “top document titles” are clearly OCR-contaminated text chunks rather than clean provenance labels. See `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase0\extracted\block_sign_priors.json`. The sign-prior layer is useful, but it is not yet a gold-standard curated knowledge base.

3. Phase 1 normalization is mathematically coherent, but still aggressive relative to the underlying evidence sparsity. The pipeline applies density scaling, power transform, winsorization, median imputation, robust scaling, and then multiplies by a quality-weight tensor in `D:\EpiGraph_PH\src\epigraph_ph\phase1\pipeline.py:357`. This is reasonable engineering, but it means the tensor entering later phases is no longer “the data”; it is a heavily stabilized representation. That is acceptable only if later phases avoid overclaiming measurement precision.

4. Phase 1 eligibility gates are still permissive in one important sense: any direct indicator with annual or national support can become eligible for the national likelihood, and any direct indicator with any national/regional/province support can become eligible for the province graph. See `D:\EpiGraph_PH\src\epigraph_ph\phase1\latent_observability.py:79-82`. This is useful for recall, but it still lets very thin variables such as `cash_instability` enter later phases with little true subnational support.

5. Phase 15 v2 has explicit operators, but the current observation semantics are still coarse for annual anchors. The emitted rows show annual national anchor-like measurements entering as `operator_kind = "monthly_snapshot"` with a single `support_cells` entry and `time_support_mode = "month_snapshot"`. See `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase15\phase15_v2_measurement_rows.json`. This is better than annual fan-out, but it is still not a full mixed-frequency temporal integral operator.

6. Phase 15 burden-weight learning is now structurally present, but the actual learned national weights remain almost uniform. The artifact exposes the correct feature set, but the first weights are all around `3.57e-05`. See `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase15\phase15_v2_aggregation_weights.json`. So the model has the right mechanism, but the present evidence does not yet strongly identify burden heterogeneity.

7. Phase 2 can now avoid overclaiming direct causation by separating hidden-driver rows, but the retained graph still rests on only four latent blocks in this run. The national scale reports `6` direct edges and `2` hidden-driver pairs over a `4`-block axis. See `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\phase2\latent_temporal_graph_bundle.json`. This is acceptable as a hypothesis layer, not yet as a causal discovery result.

## 3. Direct-vs-Contextual Evidence Table

| Phase | What enters | Direct evidence use | Contextual evidence use | Main evidence risk | Verdict |
|---|---|---|---|---|---|
| Phase 0 | Extracted candidate rows, then evidence rows and sign priors | Direct indicators are kept as explicit evidence rows with geo/time/operator metadata in `D:\EpiGraph_PH\src\epigraph_ph\phase0\evidence_artifacts.py:34` | Context is mostly converted into sign/block priors in `D:\EpiGraph_PH\src\epigraph_ph\phase0\evidence_artifacts.py:136` | Provenance text quality is still uneven; OCR leakage remains in some sign-prior titles | Scientifically acceptable, but provenance curation still needs hardening |
| Phase 1 | Normalized rows plus observability audit | Direct rows dominate eligibility; direct support counts drive likelihood entry in `D:\EpiGraph_PH\src\epigraph_ph\phase1\latent_observability.py:79-82` | Context-only rows are tracked and counted, but ineligible variables are correctly blocked | Heavy preprocessing can make weak evidence look smoother and more comparable than it really is | Good gatekeeper, but downstream claims must remember this is a stabilized representation |
| Phase 15 | Aggregated measurement rows over latent support cells | Numeric direct rows become likelihood terms via explicit support cells in `D:\EpiGraph_PH\src\epigraph_ph\phase15\v2_engine.py:395` | Context-only rows are not supposed to become numeric likelihood rows; they mainly survive through sign/block structure | Annual national evidence is still mapped coarsely; burden weights are not strongly identified | Strong improvement; still should be described as a partially identified latent model |
| Phase 2 | Latent innovation graph on Phase 15 states | Direct evidence enters only indirectly, through Phase 15 latent states | Contextual influence is only as far as it changed Phase 15 block definitions or priors | Easy to overread sparse latent graph edges as evidence-supported causal links | Should be framed as temporal hypothesis structure, not direct empirical causation |

## 4. Top 5 Improvements

1. Replace annual `month_snapshot` handling in Phase 15 with true interval-support operators. A yearly national row should constrain a weighted integral or average over all province-month cells in that year, not a single month cell. The operator system is already explicit in `D:\EpiGraph_PH\src\epigraph_ph\phase15\v2_engine.py:395`; this is the highest-value mathematical fix left in the evidence path.

2. Introduce a human-labeled benchmark for free-text extraction. The extraction audit is strong on structured sources and HARP, but it explicitly states that unstructured literature still lacks a literal gold standard. See `D:\EpiGraph_PH\artifacts\runs\smoke-latent-blocks\analysis\extraction_quality_audit.md`. Until that benchmark exists, literature-derived semantics should be treated as curated priors, not validated observations.

3. Tighten Phase 1 eligibility to distinguish “nationally usable” from “subnationally informative.” The current gate in `D:\EpiGraph_PH\src\epigraph_ph\phase1\latent_observability.py:79-82` is still broad enough that thin annual national variables can flow into the province graph. Add a stricter subnational coverage threshold for province-graph eligibility.

4. Add provenance-quality scoring for sign-prior rows. `block_sign_priors.json` is useful, but the current top-document titles mix clean portal/report titles with OCR fragments. A small provenance-cleanliness score or curated-title canonicalizer would make sign priors much easier to trust and review.

5. In Phase 2 reporting, keep direct edges and hidden-driver rows visually and semantically separate everywhere downstream. The code already exposes both in `D:\EpiGraph_PH\src\epigraph_ph\phase2\latent_temporal_graph.py:376-449`; the main remaining risk is interpretive drift. Direct sparse edges should feed mechanistic hypotheses, while low-rank rows should feed uncertainty/stress-test narratives rather than causal statements.

## Bottom Line

The strongest scientific claim currently supported by the repo is:

- structured extraction is now internally reproducible and conflict-aware,
- Phase 1 is a defensible evidence-normalization and observability gate,
- Phase 15 is a real latent mixed-evidence inference layer with explicit support operators,
- Phase 2 is now a temporal hypothesis layer over latent innovations rather than a naive same-time DAG.

The strongest scientific claim not yet supported is:

- that the full pipeline has gold-standard validity for unstructured literature extraction,
- that annual national anchors are already handled with a fully faithful mixed-frequency operator,
- or that Phase 2 edges can be read as confirmed causal structure rather than evidence-conditioned hypotheses.
