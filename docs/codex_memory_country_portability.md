# Codex Memory Context: Country Portability and Pack Structure

This document is a **conversation-memory-based portability specification**, not a fresh repository audit.

Its purpose is to describe how the pipeline should generalize across countries without pretending that every country is a zero-effort restart.

It should be read together with the disease-portability view:

- country packs localize the environment
- disease plugins localize the scientific model

The core principle is:

> The framework can be shared globally, but the evidence packs, ontology packs, hierarchy packs, benchmark packs, and policy packs are local.

---

## 1. Portability Principle

We repeatedly converged on a distinction between:

- a **shared software framework**
- a **localized scientific instantiation**

That means the following can be reused:

- Phase 0 harvest / parse / extract / canonicalize / retrieve workflow
- Phase 1 alignment and tensorization logic
- Phase 2 candidate-universe / curated-registry / promotion workflow
- Phase 3 mechanistic-model discipline and scientific gates
- Phase 4 downstream scenario / release / operational packaging discipline

But the following are country-specific:

- source systems
- geography hierarchy
- language and alias dictionaries
- KPI definitions and measurement conventions
- benchmark targets and holdout design
- legal and operational policy constraints
- subgroup taxonomy
- intervention channels

So a new country is not a pure “press run” restart.

It is better described as a **country port** inside a common framework.

---

## 2. Five-Portability-Pack Architecture

The agreed portability design is best represented as five pack types:

1. `global_config`
2. `country_pack`
3. `source_adapter_pack`
4. `benchmark_pack`
5. `policy_pack`

Together they define a country instance:

\[
\mathfrak{I}_{country}
=
\left(
\text{global\_config},
\text{country\_pack},
\text{source\_adapter\_pack},
\text{benchmark\_pack},
\text{policy\_pack}
\right)
\]

The framework stays fixed.
The instance changes.

---

## 3. Global Config

The `global_config` defines what is common across deployments.

It should include:

- runtime mode
  - local-first defaults
  - optional hosted backends
- storage defaults
  - DuckDB
  - Parquet
  - FAISS
  - Chroma sidecar only for exploration
- artifact contract
  - manifests
  - parsed blocks
  - numeric observations
  - canonical candidates
  - registries
  - fit / validation / benchmark outputs
- scientific gates
  - no fabricated references
  - no fabricated statistical results
  - no silent fallback claims
  - prior-vs-direct-truth separation
- generic tensor conventions
  - time axis conventions
  - geo normalization conventions
  - provenance tiering
  - uncertainty tagging
- default compute guardrails
  - bounded working sets
  - resumable stages
  - no mandatory all-VRAM execution

The global config should **not** hardcode:

- one country hierarchy
- one disease ontology
- one benchmark target definition
- one policy regime

---

## 4. Country Pack

The `country_pack` is the main localization bundle.

It should define:

### 4.1 Geography hierarchy

- national level
- region / state / province hierarchy
- district / city / barangay or equivalent if needed
- canonical geo ids
- alias tables
- boundary version metadata

### 4.2 Time conventions

- default reporting frequency
  - monthly
  - weekly
  - quarterly
  - annual
- reporting-year conventions
- surveillance lag conventions

### 4.3 Language and alias layer

- multilingual aliases
- institution aliases
- disease-program aliases
- local administrative abbreviations
- subgroup terminology mapping

### 4.4 Population and subgroup taxonomy

For HIV-like models this may include:

- key-population taxonomy
- age-band conventions
- sex axis conventions
- partner and spillover categories

For other diseases, the subgroup layer may differ completely.

### 4.5 Country-specific ontology hints

This should remain soft at discovery time.

It should provide:

- preferred tags
- synonym banks
- country-specific measurement names
- reporting-program keywords
- domain hints for economics, mobility, stigma, biology, logistics, demography, and service delivery

### 4.6 Country priors and structural assumptions

Only at the level needed to start discovery and validation safely:

- plausible data-quality assumptions
- broad priors on reporting completeness
- country-specific caveats
- structural exclusions if necessary

The country pack localizes the system.
It does not rewrite the global framework.

---

## 5. Source Adapter Pack

The `source_adapter_pack` defines how the country actually connects to evidence.

It should specify:

### 5.1 Official anchor sources

- ministry of health / national public-health body
- national statistical office
- official registries
- country surveillance portals
- official dashboards
- official policy repositories

### 5.2 International overlays

- WHO country pages
- UNAIDS country pages where relevant
- UN system pages
- World Bank and similar auxiliary repositories

### 5.3 Scientific literature sources

- PubMed
- Crossref
- OpenAlex
- Semantic Scholar
- arXiv
- bioRxiv
- manual seeds
- targeted scraping

### 5.4 Structured repositories

- Kaggle when useful
- open public datasets
- country-specific open-data portals

### 5.5 Adapter behavior

Each adapter should define:

- discovery queries
- pagination behavior
- download behavior
- snapshot behavior
- metadata fields
- checksum rules
- eligibility defaults
- source tier mapping

The source adapter pack is where portability often fails in practice.

The framework is reusable.
The source connectors are local.

---

## 6. Benchmark Pack

The `benchmark_pack` defines what counts as success in that country.

It should specify:

### 6.1 Target definitions

For HIV this may include:

- first 95
- second 95
- third 95
- documented suppression
- regional/provincial variants

For other diseases this could be entirely different:

- incidence
- mortality
- hospitalization
- outbreak peaks
- treatment completion

### 6.2 Truth surfaces and eligibility

- what is direct truth
- what is proxy truth
- what is prior-only
- what is excluded from direct scoring

### 6.3 Holdout and split design

- year-based holdouts
- region-based holdouts
- province-based holdouts
- rolling validation windows

### 6.4 Comparator suite

- naive baselines
- trend baselines
- mechanistic baselines
- external models if available

### 6.5 Acceptance gates

- MAE tolerances
- calibration requirements
- reconciliation requirements
- integrity requirements
- claim-scope reduction rules

Without the benchmark pack, the system may run, but it cannot honestly claim anything.

---

## 7. Policy Pack

The `policy_pack` defines the downstream action layer and operational limits.

It should specify:

### 7.1 Intervention channels

For HIV-like systems these might include:

- testing expansion
- linkage support
- ART retention support
- diagnostics reach
- transport support
- stigma reduction
- workforce deployment
- KP-targeted service allocation

### 7.2 Legal and operational constraints

- budget constraints
- geographic targeting constraints
- service-capacity constraints
- subgroup targeting rules
- ethical and legal exclusions

### 7.3 Scenario semantics

- what counts as a scenario
- what counts as an intervention
- which outputs are advisory only
- which outputs are production-eligible

### 7.4 Control-family eligibility

The policy pack should not assume a heavy controller by default.

It should declare whether the country instance currently supports:

- heuristic frontier only
- MPC-capable scenario planning
- PPO-like RL experimentation

This matches the conversation: control families remain optional and maturity-dependent.

---

## 8. Portability Workflow

The country-port process is:

1. install or define the shared framework
2. create `country_pack`
3. create `source_adapter_pack`
4. create `benchmark_pack`
5. create `policy_pack`
6. run Phase 0 broad discovery
7. run Phase 1 alignment
8. run Phase 2 curation and promotion staging
9. run Phase 3 scientific fit and benchmarking
10. only then allow Phase 4 scenario or policy packaging

This is why a new country is a **port**, not a trivial restart.

---

## 9. Portability Rules

### 9.1 What should be reusable

- pipeline orchestration
- artifact contracts
- scientific gates
- provenance model
- retrieval architecture
- broad discovery logic
- curation logic
- promotion logic

### 9.2 What should be localized

- ontology hints
- source adapters
- geography hierarchy
- subgroup taxonomy
- benchmark definitions
- policy constraints

### 9.3 What must never be assumed

- that country A measurement definitions equal country B definitions
- that subgroup taxonomies are interchangeable
- that one policy pack fits all countries
- that a scientific benchmark can be ported unchanged
- that a local run is valid just because the code executed successfully

---

## 10. Global Disease Portability

This portability logic also extends across diseases, but with a stronger warning:

- the **framework** can be shared,
- the **disease model plugin** cannot be assumed to be identical.

For example:

- HIV cascade dynamics
- dengue outbreak dynamics
- TB treatment dynamics
- malaria transmission dynamics

may all live in the same broader framework, but they do **not** necessarily share:

- state topology
- time resolution
- observation model
- intervention semantics

So for disease portability, the correct object is:

\[
\mathfrak{I}_{country,disease}
=
\left(
\text{global\_config},
\text{disease\_plugin},
\text{country\_pack},
\text{source\_adapter\_pack},
\text{benchmark\_pack},
\text{policy\_pack}
\right)
\]

This is stronger and more honest than saying one YAML file alone can safely define every disease instance.

---

## 11. Final Portability Principle

The most important takeaway is:

> The software architecture can be global.
> The scientific instantiation is local.
> The benchmark is local.
> The policy layer is local.

That is the portability model most consistent with the Codex conversation memory.
