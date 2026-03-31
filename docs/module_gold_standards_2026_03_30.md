# Module Gold Standards

Date checked: 2026-03-30

## Principle

Each module needs one of the following:

- an authoritative external standard
- an accepted benchmark family
- an explicit statement that no true gold standard exists, in which case we must use benchmark ladders and stress tests instead of claiming external truth

The goal is to stop blind engineering. A module should never be evaluated only against its own outputs.

## Cross-Cutting Standards

These apply across the whole repo:

- Provenance and metadata:
  - W3C PROV-O
  - FAIR Guiding Principles
- Measurement uncertainty:
  - JCGM 100:2008 Guide to the Expression of Uncertainty in Measurement
- Prediction-model reporting and bias:
  - TRIPOD+AI
  - PROBAST / PROBAST+AI
- Decision-model transparency and validation:
  - ISPOR-SMDM Modeling Good Research Practices
  - AHRQ Guidance for the Conduct and Reporting of Modeling and Simulation Studies

## Phase 0

### Gold standards

- Systematic evidence retrieval and reporting:
  - PRISMA 2020
- RHIS and health facility data quality:
  - WHO Data Quality Assurance / Data Quality Review toolkit
- HIV monitoring minimum datasets and indicators:
  - WHO Consolidated HIV strategic information guidelines (2020)
  - WHO person-centred HIV strategic information guidelines (2022)
- Provenance and machine-actionable traceability:
  - W3C PROV-O
  - FAIR
- Measurement uncertainty:
  - JCGM 100:2008

### What this phase must be judged against

- PRISMA-style search traceability
- explicit provenance for every extracted numeric row
- WHO DQA dimensions:
  - completeness
  - timeliness
  - internal consistency
- uncertainty carried forward instead of hidden

### What is not a gold standard here

- cosine similarity
- FAISS retrieval
- OCR model choice

Those are implementation choices, not standards.

## Phase 1

### Gold standards

- Measurement uncertainty and propagation:
  - JCGM 100:2008
- RHIS consistency and quality review:
  - WHO DQA / DQR
- Robust location and scale estimation:
  - NIST robust summary guidance for median / robust scale estimators

### What this phase must be judged against

- no unit/denominator ambiguity
- uncertainty and quality weights remain attached
- robust scaling justified and documented
- no silent distortion of low-volume provinces

### Important note

There is no HIV-specific official standard that says “use median/IQR scaling.” That part is a robust-statistics choice, so it must be documented as a method choice grounded in NIST-style robust estimation, not presented as an HIV program standard.

## Phase 2

### Gold standards

There is no official public-health gold standard for high-dimensional causal discovery on this problem.

Use the nearest accepted benchmark family:

- exact DAG validity:
  - acyclicity must be satisfied exactly
- benchmark graph recovery:
  - Sachs et al. 2005 causal protein-signaling network
- time-series causal discovery benchmark family:
  - Runge et al. 2019 / PCMCI-style causal discovery for autocorrelated nonlinear time series
- uncertainty over discovered edges:
  - bootstrap / bagged causal discovery confidence

### What this phase must be judged against

- zero cycle violations
- respect for tier and lag masks
- recovery on known benchmark graphs
- edge stability under bootstrap or source dropout
- negative controls and permutation-null tests

### Important note

Phase 2 should never be described as having a single external “truth set” for the Philippines HIV determinant graph. It does not. It has benchmark families and mechanistic admissibility rules.

## Phase 3

### Gold standards

- HIV cascade denominator definitions:
  - UNAIDS 95-95-95 indicator definitions
- strategic information minimum dataset and monitoring framework:
  - WHO Consolidated HIV strategic information guidelines (2020)
  - WHO person-centred HIV strategic information guidelines (2022)
- viral suppression semantics:
  - WHO viral suppression policy brief
- incumbent epidemic estimation comparator:
  - Spectrum / AIDS Impact Model (AIM) manual
- operational HIV program indicator comparator:
  - PEPFAR MER indicator reference guides
- Philippines-specific observed anchors:
  - HARP / PNAC / DOH operational program data

### What this phase must be judged against

- exact denominator semantics for first 95 / second 95 / third 95 / cascade view
- WHO/UNAIDS suppression definitions
- historical frozen backtests against observed HARP panel
- comparison against naive carry-forward
- comparison against incumbent Spectrum/AIM outputs where available
- hierarchy consistency:
  - province to region to national

### This is the strongest ground-truth phase

If a module in this repo gets to claim “gold standard,” this is the one that comes closest, because it can be evaluated against official HIV indicator definitions and program-observed reference series.

## Phase 4

### Gold standards

There is no single HIV-specific official gold standard for stochastic control or allocation optimization.

Use the accepted decision-modeling standards:

- ISPOR-SMDM good research practices
- AHRQ modeling guidance
- CHEERS 2022 for reporting, if policy/economic evaluation claims are made

### What this phase must be judged against

- decision logic must preserve uncertainty, not just optimize point estimates
- policy comparisons must be externally auditable
- validation must include:
  - face validity
  - predictive validity where possible
  - cross-model validity
  - sensitivity and stability analysis
- allocations must beat:
  - naive allocation
  - current-policy baseline
  - simple heuristic baselines

### Important note

Phase 4 does not have external “truth” in the same sense as Phase 3. It has good-practice standards plus comparator policies.

## Node Graph

### Gold standards

There is no formal public-health gold standard for a runtime-assurance node graph.

Use:

- provenance standard:
  - W3C PROV-O
- decision-model validation standards:
  - ISPOR-SMDM / AHRQ
- explicit local contract:
  - node graph must not rewrite the epidemic core

### What this layer must be judged against

- provenance of every node signal
- reproducible evidence scoring
- no change to structural forecast when node graph is disabled
- only decision-layer effects are allowed

## Practical Benchmark Ladder For This Repo

### Phase 0

- PRISMA 2020 checklist
- WHO DQA metrics
- provenance completeness

### Phase 1

- JCGM-style uncertainty carry-through
- WHO DQA consistency checks
- robust scaling diagnostics

### Phase 2

- Sachs benchmark
- synthetic DAG recovery
- bootstrap edge stability

### Phase 3

- HARP frozen-history backtests
- UNAIDS / WHO indicator definitions
- Spectrum/AIM comparison
- carry-forward baseline

### Phase 4

- naive allocation baseline
- current-policy baseline
- uncertainty-aware policy robustness checks

## What Should Be Wired Into Code Next

Every phase should eventually emit:

- `gold_standard_manifest.json`
- `gold_standard_checks.json`
- `gold_standard_summary.json`

Minimal expected contents:

- source standard or benchmark used
- version/date
- what exact metric or rule was checked
- pass/fail
- residual gaps

## Sources

- PRISMA 2020: <https://www.bmj.com/content/372/bmj.n71>
- WHO DQA / DQR: <https://www.who.int/data/data-collection-tools/health-service-data/data-quality-assurance-dqa>
- WHO Consolidated HIV strategic information guidelines (2020): <https://www.who.int/publications-detail-redirect/9789240000735>
- WHO person-centred HIV strategic information guidelines (2022): <https://www.who.int/publications-detail-redirect/9789240055315>
- UNAIDS 95-95-95 definitions (2024): <https://www.unaids.org/en/resources/documents/2024/progress-towards-95-95-95>
- WHO viral suppression policy brief: <https://www.who.int/publications/i/item/9789240055179>
- Spectrum manual: <https://www.avenirhealth.org/Download/Spectrum/Manuals/SpectrumManualE.pdf>
- PEPFAR MER reference guides: <https://help.datim.org/hc/en-us/articles/360000084446-MER-Indicator-Reference-Guides>
- JCGM 100:2008: <https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf>
- W3C PROV-O: <https://www.w3.org/TR/prov-o/>
- FAIR Guiding Principles: <https://www.nature.com/articles/sdata201618>
- Sachs benchmark paper: <https://pubmed.ncbi.nlm.nih.gov/15845847/>
- PCMCI / time-series causal discovery benchmark family: <https://www.science.org/doi/10.1126/sciadv.aau4996>
- PROBAST: <https://www.probast.org/wp-content/uploads/2020/02/PROBAST_20190515.pdf>
- TRIPOD+AI: <https://www.bmj.com/content/385/bmj-2023-078378>
- ISPOR-SMDM conceptualizing a model: <https://www.ispor.org/docs/default-source/resources/outcomes-research-guidelines-index/conceptualizing_a_model-2.pdf>
- ISPOR-SMDM model transparency and validation: <https://www.sciencedirect.com/science/article/pii/S1098301512016567>
- AHRQ modeling guidance: <https://effectivehealthcare.ahrq.gov/products/decision-models-guidance/methods>
- CHEERS 2022: <https://www.bmj.com/content/376/bmj-2021-067975>
