# Codex Memory Context: Full Pipeline, Phase 0 to Phase 4

This document is a reconstruction of the pipeline context **from our Codex conversations**, not a fresh repository audit.

That distinction matters:

- **High-confidence sections**: things we designed, implemented, tested, or discussed in detail in the conversation.
- **Medium-confidence sections**: things we repeatedly implied architecturally, but did not fully re-specify line by line.
- **Low-confidence sections**: downstream phase boundaries that were discussed more as intended system structure than as fully closed implementation detail.

The goal of this document is to give us one coherent picture of the full stack we built and the scientific rules we attached to it.

---

## 1. Global Goal

The full system is a **scientific, local-first, production-shaped pipeline** for:

1. harvesting evidence and data from many domains,
2. extracting candidate subparameters and measurements,
3. aligning them into tensor-ready model inputs,
4. fitting a mechanistic HIV model with strong integrity controls,
5. evaluating, benchmarking, and only then promoting downstream production candidates.

The pipeline is intentionally wider than HIV-only literature.

We explicitly decided that the stack must discover:

- direct HIV mechanisms,
- indirect upstream determinants,
- system frictions,
- population structure,
- social and cultural constraints,
- biology and clinical severity,
- logistics and mobility,
- economics and affordability,
- subgroup and interaction effects.

The long-term idea is:

- **collect very wide evidence first**,
- **derive candidate linkages later**,
- **promote only what survives scientific gates**.

The best memory-consistent Phase 0 cold-start rewrite is:

- start with broad domain categories
- harvest widely
- extract softly
- preserve provenance
- let cheap filters and scoring narrow the field
- then use ablation, benchmark, and scientific gates to decide what survives

One related later clarification was the Phase 0 embedding strategy:

- use a **general embedding model** for broad discovery
- organize the corpus into explicit **domain silos**
- keep retrieval collections split and coherent
- reserve narrow biomedical models for targeted refinement, not as the universal reader

One later portability correction also became important:

- the framework can be shared across diseases,
- but the disease model cannot be reduced to a country config alone,
- so cross-disease portability requires a **disease plugin**, not just a pack swap.

One more later correction also mattered:

- multi-disease use should default to **federated disease engines**
- disease coupling should happen through exchanged summaries and explicit diagnostics
- not through one giant merged disease tensor by default

---

## 2. Core Design Principles

These principles came up repeatedly across the conversation.

### 2.1 Local-first and low-cost

The default system should work:

- locally,
- without cloud APIs,
- under limited VRAM,
- with hosted backends optional rather than required.

### 2.2 Scientific integrity over convenience

We repeatedly hardened the stack against:

- fabricated numbers,
- fabricated references,
- fabricated statistical claims,
- silent fallback behavior,
- pretending an upstream module already ran,
- treating prior-driven or synthetic outputs as direct truth.

One later extension of this principle became explicit:

- every module in the pipeline should emit a **ground-truth package**
- not every module has direct observed truth
- but every module must have an external reference, internal consistency check, synthetic recovery test, or invariance test
- no module should be considered complete without an explicit truth class and truth audit

### 2.3 Separation of candidate discovery from promotion

We explicitly separated:

- `candidate_universe`
- `curated_registry`
- `active_inferential_set`

This is critical.

The system is allowed to discover widely.
It is **not** allowed to automatically fit everything it discovers.

### 2.4 Dual evidence lanes

One of the most important design corrections was this:

- **HIV-direct lane**
- **upstream determinant lane**

This avoided a bad failure mode:

- if we keep only HIV-specific papers, we miss upstream drivers like poverty, transport friction, cultural stigma, migration, remoteness, education, governance, and collective behavior
- if we drop the HIV filter entirely, we drown in irrelevant broad literature

So we ended with a two-lane literature/discovery model.

### 2.5 Soft ontology first, hard promotion later

We explicitly chose to keep the ontology **soft** in the widening stages.

That means:

- preserve broad tags,
- preserve soft subparameter hints,
- do not over-collapse too early,
- allow later linkage and ablation to decide what matters.

### 2.6 Selective HPC adoption, not full rewrite

We explicitly converged on a selective high-performance-computing roadmap rather than a blanket GPU-first rewrite.

The agreed direction was:

- use **PyTorch tensors more** in:
  - Phase 1 alignment math
  - Phase 2 scoring / selection math
  - Phase 3 transition computations
- consider **GPyTorch** only for specific alignment subproblems where GP interpolation is the real bottleneck
- consider **NumPyro / JAX** only after:
  - deciding to migrate the Phase 3 probabilistic core, and
  - getting JAX GPU working on the target machine
- consider **Ray** only when Phase 4 is mature enough that distributed rollout infrastructure is justified

This was an important correction.

We did **not** agree to:

- make every phase tensor-only,
- keep all data in VRAM all the time,
- replace the scientific storage and gating backbone with a pure GPU stack,
- or force JAX/Ray into the system before the scientific core is stable.

---

## 3. End-to-End Architecture

At the highest level, the full pipeline now looks like this:

```text
Phase 0
  Broad evidence and data intake
  -> source manifests
  -> document parsing
  -> numeric and textual extraction
  -> canonical candidates
  -> vector retrieval sidecars
  -> dual literature registries

Phase 1
  Alignment and tensor preparation
  -> normalized geo/time/unit harmonization
  -> tensor-ready evidence surfaces
  -> aligned model-facing inputs

Phase 2
  Candidate block structuring and inferential packaging
  -> candidate universe
  -> curated registry
  -> blockwise organization
  -> admissibility / linkage / selection surfaces

Phase 3
  Scientific model fitting and benchmarking
  -> mechanistic HIV cascade model
  -> subnational hierarchy
  -> KP-by-age-by-sex decomposition
  -> coarse CD4 severity overlay
  -> fit artifact / validation artifact / benchmark artifact

Phase 4
  Downstream production candidate and decision layer
  -> controlled ablations
  -> simulation / scenario use
  -> production candidate gating
  -> docs / audit / release discipline
```

---

## 4. Phase 0

## 4.1 Role

Phase 0 is the **evidence and data intake layer**.

We explicitly reframed it as:

> ETL + NLP + schema + validation

not “just AI.”

Its job is to move from:

- papers,
- reports,
- portals,
- datasets,
- tables,
- numeric snippets,
- broad scientific literature

to:

- verifiable source manifests,
- parsed documents,
- extracted numbers,
- canonical parameter candidates,
- searchable retrieval surfaces,
- wide literature registries.

## 4.2 Source scope

By the end of the conversation, the intended harvest scope included:

### Official anchor sources

- WHO
- UNAIDS
- UN agencies
- DOH Philippines
- Government of the Philippines
- official Philippines registries and portals

### Scientific literature

- PubMed
- Crossref
- arXiv
- bioRxiv
- OpenAlex
- Semantic Scholar
- manual seeds
- targeted scraping
- optional OpenResearcher-style discovery

### Structured repositories and auxiliary data

- Kaggle
- World Bank style structured repositories
- other public covariate datasets

The source policy was explicitly tiered:

- `Tier 1`: official anchors
- `Tier 2`: scientific literature
- `Tier 3`: structured repositories and covariates
- `Tier 4`: targeted scrape-only sources

## 4.3 What Phase 0 had to support

We explicitly defined the required tooling layers:

- search / harvesting
- PDF / HTML / CSV ingestion
- layout-aware parsing
- table extraction
- chart/image extraction when necessary
- entity and number extraction
- canonicalization
- strict schema validation
- structured storage
- embeddings
- vector retrieval

## 4.3A Phase 0 cold-start rule

One important later clarification was that the operator should not begin by trying to guess the perfect variable list manually.

The practical rule became:

- start with broad domain categories
- define source maps
- define initial query banks
- widen the evidence universe
- preserve provenance and candidate status
- let cheap filters and scoring narrow the field
- only later use ablation, benchmark, and scientific gates to decide what survives

This was the practical answer to the “infinite variable universe” problem.

## 4.4 Phase 0 production stack we settled on

By the end of the conversation, the best answer was a **hybrid architecture**:

- keep the current scientific core,
- graft in the strongest parts of the proposed retrieval stack,
- do **not** replace the canonical storage and scientific-gating layers.

So the intended production stack was:

- **search/harvest**
  - manual seeds
  - PubMed
  - Crossref
  - WHO / UNAIDS / DOH / GovPH / UN scraping
  - Kaggle
  - OpenAlex
  - Semantic Scholar
  - optional OpenResearcher as a discovery adapter, not the default harvester

- **parse**
  - Docling used more aggressively where safe
  - GROBID as academic metadata sidecar
  - PyMuPDF / pdfplumber / HTML / CSV fallback paths
  - no commitment to one parser as the sole path

- **table extraction**
  - Docling table detection when useful
  - Camelot
  - pdfplumber
  - `pandas.read_html` for HTML

- **chart path**
  - local VLM only when needed
  - chart extraction always lower priority than CSV / table / registry

- **entity and number extraction**
  - regex and rules first
  - lightweight NLP
  - local LLM normalization second
  - Pydantic-style schema validation always

- **embeddings**
  - lighter local embeddings remain the operational default
  - BGE-M3 is treated as a strong optional upgrade, not a required default

- **storage**
  - DuckDB + Parquet as the canonical storage plane
  - JSON only for manifests and summaries

- **retrieval**
  - FAISS as the production vector index for exact or controlled candidate retrieval
  - Chroma as a sidecar for:
    - document / chunk retrieval
    - lane-specific exploration
    - metadata + `where_document` retrieval over curated subsets

This was the key correction:

- the proposed OpenResearcher / Docling / BGE-M3 / Chroma stack is good for discovery and exploratory retrieval,
- the current scientific stack remains better as the production backbone,
- so the final architecture is intentionally hybrid rather than replacement-only.

## 4.5 Phase 0 source and evidence rules

We established a number of non-negotiable scientific rules.

### Reference verification

Phase 0 references must be verifiable.

At least one of the following must exist:

- `url`
- `doi`
- `pmid`
- `openalex_id`

Free-text-only references are not allowed into the scientific registry path.

### Direct truth vs broad evidence

We explicitly separated:

- `anchor_eligible`
- `direct_truth_eligible`
- prior-only
- scrape-only
- chart-derived

Not everything that is useful for discovery is valid as scientific truth.

### Chart-derived values

Chart/image-derived values must be:

- marked as chart-derived,
- excluded from direct-truth scoring by default,
- manually promoted if ever used as something stronger.

## 4.6 Dual literature lanes

This was one of the key architectural decisions.

### Lane A inside Phase 0: HIV-direct literature

This lane is for literature explicitly about:

- HIV
- AIDS
- cascade
- key populations
- testing
- ART
- suppression
- CD4
- stigma in HIV context
- HIV biology / therapeutic development

This lane has stronger scientific relevance for direct mechanistic priors.

### Lane B inside Phase 0: upstream determinants

This lane intentionally includes non-HIV papers that may still matter causally upstream, for example:

- poverty
- transport friction
- migration
- remoteness
- education
- cultural stigma
- trust and disclosure
- social norms
- governance
- service delivery
- diagnostics logistics
- mobility networks
- collective behavior
- statistical physics of contagion or human movement
- immunology or drug papers with broader biological relevance

These are not treated as direct HIV truth.
They are treated as candidate determinant evidence.

## 4.7 Linkage scoring

We added a linkage scorer on top of the upstream lane so the system could estimate which determinants are plausibly relevant to HIV pathways.

The linkage targets we discussed included:

- prevention access
- testing uptake
- linkage to care
- retention / adherence
- suppression outcomes
- mobility / network mixing
- health system reach
- biological progression

This was a major improvement because it let the upstream corpus stay broad without becoming undirected noise.

## 4.8 Soft ontology

We explicitly chose a soft ontology policy in Phase 0.

That means we preserve:

- `soft_ontology_tags`
- `soft_subparameter_hints`
- broad domain buckets
- lane membership
- linkage targets

The point is to avoid prematurely forcing every paper or extracted value into a brittle hard ontology.

## 4.9 Phase 0 quality controls

By the end of the conversation, we had several quality layers on top of Phase 0:

- domain-quality scoring
- HIV-direct scorer
- upstream determinant scorer
- linkage scorer
- soft ontology tags
- no-fabrication scientific gate
- fallback-reliance sanity checks
- choice-overload sanity checks

## 4.10 Phase 0 retrieval architecture

The retrieval architecture stabilized into this:

### Production default

- `DuckDB + Parquet + FAISS`

This remained the default because it performed better and was more faithful for exact candidate retrieval.

### Chroma sidecar

Chroma was retained, but only for the modes where it actually made sense:

- document / chunk retrieval
- lane-specific exploration
- metadata + `where_document` searches over curated subsets

OpenResearcher was retained in the same spirit:

- useful as an optional discovery and citation-chasing adapter,
- not trusted as the canonical evidence engine by itself.

We explicitly learned that Chroma underperformed when fed the polluted raw candidate dump.

That led to the correction:

- never use one raw mixed collection as the default surface
- split collections into curated candidates, raw candidates, document chunks, HIV-direct lane, and upstream lane

## 4.10A Phase 0 embedding-and-silo strategy

We later made the embedding strategy more explicit:

- default to a general local embedder for broad discovery
- treat BGE-M3 as a strong optional upgrade when compute permits
- keep lighter local models as the practical default when stability and local cost matter most
- partition the corpus into domain silos
- partition retrieval into coherent collections instead of one mixed dump
- keep domain-specific biomedical models for focused biomedical refinement rather than universal discovery

## 4.11 Phase 0 large-corpus widening

We spent a large part of the conversation widening Phase 0 aggressively.

The important additions were:

- broad query banks
- Philippines-focused queries
- economics, logistics, geography, culture, dynamics, statistical-physics style queries
- biology, CD4, immunology, antivirals, therapeutic development
- OpenAlex
- Semantic Scholar

The corpus goal was explicitly to widen the subparameter field as much as possible, not just collect HIV-title-matching papers.

## 4.11A Phase 0 cold-start playbook

We later made the operational cold-start pattern explicit as a dedicated playbook covering:

- category list
- Philippine source map
- initial query bank
- evidence-quality rules
- candidate-only vs promotion-eligible split

## 4.12 Phase 0 current conversation status

From the conversation, the latest broad harvest state was:

- a much larger metadata-first corpus than the earlier 16.9k ceiling
- large-scale harvest with OpenAlex and Semantic Scholar added
- a larger scored sweep than before
- more real Philippines-focused downloads
- stronger dual-lane separation

We also learned some operational truths:

- OpenAlex materially increases corpus scale
- Semantic Scholar bulk search is useful, but less dominant than OpenAlex at current settings
- large-corpus parse needs careful heavy-document budgeting
- some “PDF” links are actually HTML or landing pages and must be sniffed before parse

---

## 5. Phase 1

This phase was less explicitly developed in the conversation than Phase 0 or Phase 3, so this section is more architectural synthesis than line-by-line closure.

## 5.1 Role

Phase 1 is the **alignment and tensorization layer**.

Its job is to convert the wide and messy Phase 0 evidence plane into model-facing aligned tensors and normalized surfaces.

## 5.2 What Phase 1 must do

Based on the conversation, Phase 1 should handle:

- unit normalization
- geo normalization
  - province
  - region
  - national
- time normalization
  - monthly / quarterly / annual where appropriate
- subgroup normalization
  - KP
  - age
  - sex
- provenance retention
- confidence retention
- direct vs prior labels
- tensor-ready output contracts

## 5.3 Expected outputs

Conceptually, Phase 1 should emit:

- aligned measurement rows
- normalized axes
- tensor-ready arrays
- alignment summaries
- reconciliation summaries

This is the phase that turns Phase 0 from “interesting extracted evidence” into “fit-ready model inputs.”

### 5.4 HPC direction for Phase 1

Phase 1 is one of the best places to increase tensor-native computation safely.

The agreed upgrade path is:

- move more alignment math into PyTorch tensor operations,
- keep canonical storage and manifests in DuckDB / Parquet,
- use GPyTorch only for the alignment subproblems where Gaussian-process interpolation is actually the bottleneck,
- avoid forcing every alignment job through a GP if simpler normalization or weighted reconciliation is sufficient.

## 5.5 Why it matters

Without Phase 1, the wide corpus is scientifically hard to use because:

- units differ,
- geographies differ,
- time scales differ,
- evidence strength differs,
- direct truth and upstream priors get mixed.

Phase 1 is the first place where the evidence becomes structurally consumable by the model.

---

## 6. Phase 2

Like Phase 1, this phase was more architectural than fully re-derived in the conversation, but we discussed its logical contents repeatedly through the registry, curation, and promotion design.

## 6.1 Role

Phase 2 is the **candidate packaging, block organization, and admissibility layer**.

It takes aligned evidence and turns it into structured model candidate blocks.

## 6.2 Core objects

The three key registry layers were explicit in the conversation:

- `candidate_universe`
- `curated_registry`
- `active_inferential_set`

This is effectively the heart of Phase 2.

## 6.2 HPC direction for Phase 2

Phase 2 is another appropriate place for more tensor-native computation.

The agreed direction is:

- move more scoring, ranking, and selection math into PyTorch tensors,
- keep the registry and provenance layer structurally explicit rather than trying to hide it inside a pure tensor program,
- do not treat NOTEARS or one specific causal-discovery implementation as the mandatory Phase 2 backbone unless we explicitly choose that route later.

## 6.3 Domain blocks

We repeatedly organized the candidate field into broad blocks:

- population
- behavior
- biology / clinical
- logistics / health systems
- mobility / network
- economics / access
- policy / implementation
- environment / disruption
- subgroup-specific mechanisms
- interaction-level mechanisms

These are not all promoted at once.

They are:

- discovered,
- tagged,
- deduplicated,
- linked,
- curated,
- then moved into ablation and promotion.

## 6.4 Selection and integrity rules

The following rules belong naturally to Phase 2:

- no proxy-target leakage in feature screening
- no target-synonym contamination
- no free-text-only references
- no snippetless promoted candidates
- provenance required
- ontology mapping required
- evidence strength required
- promotion must be artifact-backed

## 6.5 Controlled promotion

We also repeatedly discussed a controlled ablation matrix.

That logically sits at the Phase 2 / Phase 3 boundary:

- core only
- core + population
- core + behavior
- core + biology / clinical
- core + logistics
- core + mobility / network
- core + economics / access
- full promoted stack with shrinkage

Phase 2 decides what is eligible to enter that matrix.

Phase 3 decides what survives it scientifically.

---

## 7. Phase 3

This was the other major focus area after Phase 0.

Phase 3 is the **scientific model layer**.

It is where we moved from wide evidence collection to mechanistic HIV modeling, fitting, validation, and benchmark discipline.

## 7.1 Role

Phase 3 is where the system becomes a real HIV inference engine rather than just a literature and ETL stack.

Its duties include:

- fitting the cascade model,
- maintaining mechanistic honesty,
- handling region / province structure,
- integrating promoted subparameters,
- producing fit / validation / benchmark artifacts.

## 7.1 HPC direction for Phase 3

Phase 3 is the main place where heavier numerical acceleration may eventually matter.

The agreed roadmap is:

- use PyTorch tensors more for transition and state-update computations,
- consider NumPyro / JAX only if we intentionally migrate the probabilistic core,
- require JAX GPU to be working before treating that migration as operational,
- do not assume the current machine is ready for a JAX-first probabilistic backend by default.

### Possible Phase 3 inference choices

We did **not** fully lock one inference family for Phase 3 in the conversation.

The realistic option set is:

- **MCMC**
  - strongest reference-style Bayesian option when fidelity matters most
  - likely slower and more operationally expensive
- **SVI**
  - faster approximate inference if and only if we explicitly migrate the probabilistic core into a variational-friendly backend
  - should be treated as an approximation strategy, not as automatic scientific truth
- **Laplace / variational hybrids**
  - compromise family for cases where full MCMC is too slow but pure variational inference is too brittle

### What is committed vs optional in Phase 3

Committed from the conversation:

- mechanistic HIV state structure
- hierarchy and shrinkage
- benchmark and validation discipline
- no-fabrication scientific gates
- more PyTorch tensor usage in transition and state-update math

Still optional:

- exact posterior family
- MCMC as the required default
- SVI as the required default
- Laplace / variational hybrids as the required default
- NumPyro / JAX as the operational backend

## 7.2 Core cascade model

We repeatedly used and refined a state structure around:

- `U`: undiagnosed
- `D`: diagnosed, not effectively on ART
- `A`: on ART, unsuppressed or treatment-active care state
- `V`: virally suppressed
- `L`: loss / lapse / disengagement style treatment-linked state

We also explicitly preserved a public 95-95-95 interpretation layer:

- latent `first_95`
- latent `second_95`
- latent `true_third_95`
- observed testing coverage
- observed documented suppression

This distinction was extremely important.

## 7.3 Lane A: core scientific correction

We spent significant effort on what we called **Lane A**.

Lane A aimed to fix:

- cascade mass optimism
- latent vs observed third-95 confusion
- transition plausibility
- province clipping / saturation
- weak regional behavior
- proxy-target leakage

### Lane A workstreams

- cascade mass correction
- transition correction
- hierarchy correction
- selection and graph correction
- subnational correction

### Lane A artifacts

We discussed or implemented artifacts such as:

- `CascadeMassAuditArtifact`
- `TransitionPlausibilityArtifact`
- `SubnationalStabilityArtifact`
- `GraphSelectionAuditArtifact`
- updated fit / validation / benchmark outputs

### Lane A benchmark intent

The model needed to improve:

- hidden suppression reservoir behavior
- documented lower-bound consistency
- diagnosed stock overimputation
- regional MAE
- province boundary-hit behavior

## 7.4 KP-by-age-by-sex extension

Later in the conversation, we explicitly widened Phase 3 structurally.

We decided to add:

- explicit key-population decomposition
- explicit age
- explicit sex
- coarse CD4 severity overlay

### KP groups

We discussed a decomposition including groups like:

- remaining population
- MSM
- TGW
- FSW
- clients of sex workers
- PWID
- non-KP partners

### Age bands

At minimum:

- `15-24`
- `25-34`
- `35-49`
- `50+`

### Sex

Explicit sex axis in v1, with TGW represented through the KP axis rather than a separate biological sex expansion.

### CD4

We decided:

- yes to CD4
- but not as a full first-pass Cartesian explosion

So CD4 became a **coarse auxiliary severity overlay**, useful mainly for:

- progression
- treatment initiation severity
- attrition / mortality-like severity structure

not a massive state-space multiplier on day one.

## 7.5 State-space control

We explicitly rejected using Johnson-Lindenstrauss on the core latent-state tensor.

Instead, we chose:

- structured low-rank factorization
- hierarchical shrinkage
- sparse interactions
- pooling / archetype logic

Random projection was only allowed, if at all, on feature-side screening.

## 7.6 Validation and benchmark discipline

Phase 3 is where the main scientific evaluation lives.

We repeatedly referenced:

- fit artifact
- validation artifact
- benchmark report
- cascade decomposition
- regional and province tables
- calibration checks
- integrity reports

## 7.7 No-fabrication scientific contract

One of the strongest cross-cutting upgrades landed here.

We hardened the scientific gate so that:

- fallback numeric placeholders are not allowed in scientific claims
- missing upstream artifacts should fail scientific validation
- prior-driven / synthetic outputs must be separated from scientific-claim-eligible outputs
- benchmark claim scope must be reduced when comparator coverage is partial

This is crucial Phase 3 behavior.

## 7.8 Lane B interface with Phase 3

We also defined **Lane B** as the subparameter expansion lane.

Its relationship to Phase 3 was:

- Phase 0/2 discover and curate
- Phase 3 ablates and tests
- nothing enters the production model until:
  - Lane A is green
  - backend benchmark is green
  - promotion gate is green
  - integrity gates are green

## 7.9 Current conversation status for Phase 3

From our discussion, the Phase 3 situation was:

- mechanistic honesty improved,
- third-95 behavior improved materially,
- subnational behavior improved materially,
- diagnosed stock optimism still remained a key open issue,
- Lane A was improved but not fully closed,
- the KP-by-age-by-sex + coarse CD4 extension was added as the right structural direction.

So Phase 3 is advanced, but still scientifically active rather than “finished.”

---

## 7.10 HIV-first rescue direction

One important later synthesis was that the architecture should enter an explicit HIV-first rescue mode:

- freeze a narrow HIV scientific core
- strengthen the observation ladder first
- keep determinants outside the core until they win block tournaments
- use hard promotion budgets
- stress-test every promoted block
- target one narrow benchmark win before expanding claims or disease scope

## 8. Phase 4

Phase 4 was the least explicitly specified in the conversation.

So this section should be read as the **intended downstream layer** rather than the most fully detailed one.

## 8.1 Role

Phase 4 is the **downstream operational layer** after scientific fitting.

It should handle:

- controlled scenario use,
- production-candidate packaging,
- downstream decision support,
- operational release discipline.

## 8.2 What belongs in Phase 4

From the conversation, the following clearly belong here:

- controlled ablation comparison surfaces
- first production release candidate
- benchmark regeneration
- documentation refresh
- audit-log refresh
- final claim scope checks
- safe downstream simulation or intervention use only after Phase 3 gates

### Possible Phase 4 control choices

We did **not** commit to one policy/control family either.

The practical option set is:

- **heuristic frontier**
  - lowest-risk downstream choice
  - closest to the current conversation state
  - suitable when we want ranked trade-offs without heavy control infrastructure
- **MPC**
  - reasonable if we want receding-horizon planning on top of a stable fitted model
  - should come only after Phase 3 is scientifically stable
- **PPO-like RL**
  - possible only much later
  - requires much stronger environment definition, reward discipline, and rollout infrastructure than we currently fixed

### Where KL divergence does and does not belong

KL divergence belongs:

- in **Phase 3** if we deliberately adopt SVI or another variational inference family
- in **Phase 4** if we deliberately adopt PPO-like policy optimization and want controlled policy drift

KL divergence does **not** belong as:

- a guarantee that the model has learned reality
- a replacement for scientific validation or benchmark gates
- proof that a policy is safe just because it changes smoothly
- a currently required objective in the baseline pipeline

## 8.3 What Phase 4 is not allowed to do

It is **not** allowed to:

- bypass Phase 3 scientific gates,
- turn broad candidate discoveries directly into policy claims,
- use uncertified extraction backends as production truth,
- hide fallback or missing-comparator limitations.

## 8.4 Current confidence

Compared with Phases 0 and 3, Phase 4 is less deeply specified in the conversation.

But the intent is clear:

Phase 4 exists to make the scientifically-gated model operational without weakening the scientific contract upstream.

### What is committed vs optional in Phase 4

Committed from the conversation:

- controlled scenario surfaces only after Phase 3 gates
- documentation and audit discipline
- release and claim-scope discipline
- keeping the downstream layer from weakening upstream scientific controls

Still optional:

- heuristic frontier as the long-run default
- MPC as the long-run default
- PPO-like RL as the long-run default
- KL-controlled policy optimization
- Ray-based distributed rollout infrastructure

## 8.5 HPC direction for Phase 4

Phase 4 is the only place where a distributed rollout stack like Ray is likely to make sense.

But we explicitly agreed:

- do not introduce Ray early,
- do not build a large distributed control layer before the scientific model is stable,
- only consider Ray when Phase 4 is mature enough that many-rollout simulation or MPC infrastructure is truly required.

## 8.6 Disease portability clarification

We later made the portability model more explicit.

What is shared across diseases:

- framework phases
- artifact and provenance discipline
- scientific gates
- storage / retrieval backbone
- pack architecture

What must remain disease-specific:

- state topology
- observation model
- subgroup structure
- intervention semantics
- benchmark targets

So the correct portability object is not just:

- `global config + country pack`

It is:

- `global config + disease plugin + country pack + source adapter pack + benchmark pack + policy pack`

This is why HIV, dengue, and TB can live inside one broader framework without pretending they are the same model.

## 8.7 Syndemic coupling clarification

We later clarified that multi-disease interaction should follow a federated syndemic pattern.

That means:

- keep disease engines separate by default
- exchange low-dimensional exogenous summaries where justified
- start with cheap coupling diagnostics
- only later consider heavier nonlinear coupling methods

We also clarified that coupling evidence itself should be graded:

- weak evidence for exploration
- medium evidence for scientific retention and structured testing
- strong evidence for cautious scientific or operational claims

This keeps the system tractable on the target machine and avoids pretending that all diseases belong in one shared latent state space.

---

## 9. Cross-Cutting Sanity Checks and Gates

These protections apply across phases.

## 9.1 Fallback-problem sanity

We explicitly added stack sanity checks so fallback behavior is not hidden.

That includes:

- extraction fallback
- benchmark fallback
- graph fallback
- projection fallback

Fallback must be surfaced as:

- `pass`
- `warn`
- `fail`

depending on severity and scientific context.

## 9.2 Choice-overload sanity

We also explicitly added choice-overload checks.

This means the stack should warn when there are:

- too many candidates,
- weak separation between top options,
- weak winner margins,
- underdetermined graph selections,
- too many admissible options with poor distinction.

## 9.3 Reference verification

Every scientific reference path must be verifiable.

## 9.4 Missing-module guard

The system should not silently behave as if an upstream module already ran when it did not.

## 9.5 Research-only vs scientific-claim-eligible

We explicitly separated:

- research-only outputs
- prior-driven outputs
- synthetic outputs
- scientific-claim-eligible outputs

This is one of the most important integrity boundaries in the whole stack.

---

## 10. What We Built Most Strongly

From the conversation, the strongest and most developed parts are:

### Very strong

- Phase 0 widening architecture
- dual-lane literature/discovery model
- source tiering and provenance
- reference verification rules
- fallback / choice-overload sanity checks
- Phase 3 mechanistic honesty corrections
- KP-by-age-by-sex + coarse CD4 model direction

### Medium strength

- alignment / tensorization layer as a conceptual bridge
- candidate packaging / curation / promotion layer
- Phase 4 operational release framing

### Open or still active

- final diagnosed stock correction in the model
- hosted backend certification for extraction
- broader empirical promotion of expanded subparameter blocks
- full closure of release-candidate gates

---

## 11. Plain-English Architecture Summary

If I had to explain the whole system simply:

### First, we cast a very wide net.

We gather material from:

- official HIV sources,
- scientific papers,
- economics,
- transport,
- geography,
- population studies,
- culture and stigma work,
- mobility and network science,
- biology and immunology,
- structured repositories like Kaggle,
- broad scholarly indexes like OpenAlex and Semantic Scholar.

### Second, we do not trust raw intake.

We parse, extract, normalize, validate, and attach provenance.

Nothing important should exist without:

- source identity,
- evidence text,
- units,
- geo/time context,
- and a verifiable reference path.

### Third, we separate broad discovery from scientific promotion.

The system is allowed to discover a huge number of possible determinants.
It is not allowed to treat all of them as model truth.

That is why we have:

- candidate universe,
- curated registry,
- active inferential set,
- ablation and promotion gates.

### Fourth, we fit a mechanistic HIV model, not just a correlational black box.

The model tries to preserve:

- real cascade mass,
- real transitions,
- real subnational structure,
- and now a richer decomposition by key population, age, sex, and coarse CD4 severity.

### Fifth, we keep a hard scientific contract.

No fabricated numbers.
No fabricated references.
No silent fallback claims.
No pretending a module ran when it did not.
No prior-driven output masquerading as observed truth.

### Sixth, only after that do we treat the system as operational.

That is what the downstream phase is for:

- scenario use,
- production candidates,
- release discipline,
- and eventually decision support.

---

## 12. Final Operational Picture

The final picture from our conversation is:

- **Phase 0**: wide, local-first scientific intake and discovery
- **Phase 1**: alignment into tensor-ready evidence
- **Phase 2**: curation, block organization, promotion staging
- **Phase 3**: mechanistic HIV inference and scientific validation
- **Phase 4**: downstream production candidate and operational use

The most important meta-point is this:

> The system is designed to be wide in discovery, strict in promotion, mechanistic in inference, and explicit about uncertainty, provenance, and limits.
