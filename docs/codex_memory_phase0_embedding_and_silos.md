# Codex Memory Context: Phase 0 Embedding and Silo Strategy

This document is a **conversation-memory-based Phase 0 embedding strategy note**, not a fresh repository audit.

Its purpose is to clarify how broad Phase 0 document discovery should use embedding models and corpus organization without collapsing into a brittle one-model-for-everything design.

The core rule is:

> Use a general embedding model for broad discovery.
> Keep domain silos explicit.
> Preserve provenance and soft ontology.
> Use domain-specific biomedical models only where they truly belong.

---

## 1. Default Embedder Choice

For the broadest Phase 0 discovery surface, the default memory-consistent choice is:

- a **general local embedding model**

not:

- a narrowly biomedical encoder as the default reader for all domains.

The reason is simple:

- Phase 0 has to span economics, logistics, mobility, sociology, policy, clinical science, biology, and official documents
- a narrowly biomedical encoder is too specialized for that full search surface

So the default broad-discovery embedder should be general-purpose.

---

## 2. When to Use BGE-M3 vs Lighter Local Models

### 2.1 BGE-M3

Treat **BGE-M3** as the strongest optional broad-retrieval upgrade when compute allows.

It is best suited for:

- very broad multilingual or multi-domain retrieval
- large-scale corpus indexing
- stronger semantic search over mixed-domain documents
- more ambitious Phase 0 widening runs

### 2.2 Lighter local models

Treat lighter local embedders as the practical default when:

- RAM/VRAM budget is tight
- quick iteration matters more than peak embedding quality
- the current task is curation or retrieval over already-narrowed corpora
- the machine should remain stable under local-only runs

This reflects the actual memory-based architecture:

- lighter local embeddings are the operational default
- BGE-M3 is a strong optional upgrade

---

## 3. Domain Silo Layout

The memory-consistent discovery strategy is not one giant undifferentiated folder.

Instead, organize corpora into broad **domain silos** such as:

- economic and affordability silo
- logistics and transport silo
- mobility and migration silo
- stigma and behavior silo
- clinical and biology silo
- policy and governance silo
- demographics and population silo
- environment and geography silo
- official anchor silo

For HIV-first work, the core special silos are:

- HIV-direct clinical and program literature
- upstream determinant literature
- Philippine official anchor sources

This helps the discovery process remain broad without becoming semantically muddy.

---

## 4. Collection and Query-Bank Design

The Phase 0 memory architecture wants multiple query families and multiple retrieval surfaces.

### 4.1 Query-bank design

The query bank should be partitioned by:

- lane
  - HIV-direct
  - upstream determinant
- domain
  - economics
  - logistics
  - mobility
  - stigma
  - biology
  - policy
  - environment
- geography
  - Philippines-specific
  - regional or mixed

### 4.2 Collection design

The retrieval surfaces should also remain split where possible:

- curated candidate collection
- raw candidate collection
- document chunk collection
- HIV-direct lane collection
- upstream-determinant collection
- optional domain-specific silo collections

This follows the lesson we already captured:

- do not throw everything into one polluted raw collection
- curated retrieval works better when collections are semantically coherent

---

## 5. Where Domain-Specific Biomedical Models Still Belong

Domain-specific biomedical models are still useful.

They just do **not** belong as the universal Phase 0 reader.

They fit better in:

- biomedical NER
- clinical term extraction
- relation extraction inside already-biomedical corpora
- disease-specific downstream refinement tasks
- focused literature review over strongly biomedical subsets

So the memory-consistent split is:

- **general embedder** for broad discovery
- **domain-specific biomedical model** for targeted biomedical refinement

---

## 6. Final Phase 0 Embedding Rule

The final rule is:

> For broad Phase 0 discovery, prefer a general embedding model plus siloed corpora and split collections.
> Use biomedical-specific models as downstream specialists, not as the universal discovery reader.
