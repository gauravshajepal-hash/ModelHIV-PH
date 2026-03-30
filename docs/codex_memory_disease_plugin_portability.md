# Codex Memory Context: Disease-Plugin Portability

This document is a **conversation-memory-based disease-portability specification**, not a fresh repository audit.

Its purpose is to clarify what can be shared across diseases and what must remain disease-specific inside the broader framework.

The core principle is:

> The framework is reusable across diseases.
> The disease model is not just a config toggle.
> A disease plugin is required whenever the state topology, observation model, benchmark target, or intervention semantics materially change.

---

## 1. Shared Across Diseases

These parts of the system are intended to be reusable:

### 1.1 General pipeline structure

- Phase 0 discovery and evidence intake
- Phase 1 alignment and tensorization
- Phase 2 curation and promotion staging
- Phase 3 scientific fitting discipline
- Phase 4 downstream scenario / release discipline

### 1.2 General scientific rules

- verifiable references
- no fabricated numbers
- no fabricated statistical claims
- no silent fallback claims
- prior-vs-direct-truth separation
- explicit provenance

### 1.3 General storage and retrieval backbone

- DuckDB
- Parquet
- FAISS
- Chroma sidecar for curated exploration
- manifests and audit artifacts

### 1.4 General portability packs

- `global_config`
- `country_pack`
- `source_adapter_pack`
- `benchmark_pack`
- `policy_pack`

These are necessary for every disease instance, but they are not sufficient by themselves.

---

## 2. What Must Be Disease-Specific

The following cannot be assumed to be universal:

- state topology
- time resolution
- observation model
- disease-specific subgroup structure
- intervention channels
- benchmark targets
- policy semantics
- structural priors
- admissible causal and mechanistic constraints

This is why a disease plugin is needed.

---

## 3. Disease Plugin

The `disease_plugin` is the object that localizes the scientific model to a disease family.

It should define:

### 3.1 State topology

- main latent states
- allowed transitions
- duration / memory structure
- optional overlays

### 3.2 Time scale

- weekly vs monthly vs quarterly vs annual
- expected reporting cadence
- biological delay assumptions

### 3.3 Observation model

- what is observed directly
- what is latent
- what is a proxy
- reconciliation rules between program data and latent states

### 3.4 Subgroup model

- whether population subgroups are explicit axes
- whether age and sex are explicit axes
- whether co-infections are state variables or feature modifiers

### 3.5 Intervention model

- which intervention channels exist
- whether control is mostly prevention, detection, retention, treatment, vector control, hospital capacity, or something else

### 3.6 Benchmark semantics

- what targets matter
- how forecast quality is scored
- what safety / plausibility constraints are disease-specific

---

## 4. What Belongs in Config vs Plugin vs Pack

### 4.1 What belongs in `global_config`

- runtime defaults
- artifact conventions
- storage backends
- generic scientific gates
- compute guardrails

### 4.2 What belongs in `country_pack`

- geography hierarchy
- local aliases and language
- local subgroup labels
- country priors and caveats
- country reporting conventions

### 4.3 What belongs in `source_adapter_pack`

- local official sources
- scrape/API definitions
- source tiering defaults
- metadata and checksum rules

### 4.4 What belongs in `benchmark_pack`

- disease-country target definitions
- truth eligibility
- holdout structure
- comparator suite
- acceptance thresholds

### 4.5 What belongs in `policy_pack`

- action channels
- legal and operational constraints
- scenario semantics
- controller eligibility

### 4.6 What belongs in `disease_plugin`

- state topology
- observation model
- mechanistic transition logic
- duration structure
- disease-specific overlays
- disease-specific intervention semantics

That is the key separation:

- packs localize the environment and evidence
- the plugin localizes the scientific model

---

## 5. Example Topology Differences

### 5.1 HIV

Shared memory-based direction:

- grid: province x month
- state style: cascade-like and directional
- heavy care-continuum structure
- important duration effects
- explicit KP-by-age-by-sex decomposition
- coarse CD4 overlay in v1
- co-infections like TB/HBV/HCV mostly as feature modifiers in v1

Typical topology:

\[
U \rightarrow D \rightarrow A \rightarrow V
\]

with lapse/disengagement states and re-entry paths.

### 5.2 Dengue

Expected plugin differences:

- faster time resolution, often weekly
- outbreak-oriented dynamics
- vector ecology and weather more central
- cyclical transmission structure
- less emphasis on long care-continuum duration memory

Topology is more like:

\[
S \rightarrow E \rightarrow I \rightarrow R
\]

or a related epidemic compartment structure, often with seasonality and vector modifiers.

### 5.3 TB

Expected plugin differences:

- longer clinical course than dengue
- diagnosis and treatment completion matter heavily
- latent vs active disease may matter
- HIV/TB coupling can matter strongly without collapsing the two diseases into one giant tensor

Possible topology may include:

\[
S \rightarrow L \rightarrow A \rightarrow T \rightarrow C
\]

or another latent/active/treatment/completion style system depending on the exact TB model.

The important point is:

- HIV, dengue, and TB do not differ only by parameter values
- they differ in topology, observation logic, and intervention semantics

---

## 6. Disease Federation Without Tensor Collapse

When multiple diseases coexist in the same operational environment, the conversation memory supports:

- separate disease engines
- separate tensors
- shared external capacity or determinant variables where appropriate
- explicit syndemic coupling summaries rather than a forced shared mega-state

For example:

- hospital capacity
- diagnostics capacity
- workforce availability
- transport reach
- supply chain limits

This is a federation pattern, not a single merged mega-state.

That keeps the model tractable and preserves disease-specific scientific structure.

---

## 7. Final Disease-Portability Principle

The correct portability statement is:

> Same framework, same scientific discipline, same pack architecture.
> Different disease plugin.
> Different country pack.
> Different benchmark and policy pack.

That is more accurate than saying one config file alone can safely instantiate every disease.
