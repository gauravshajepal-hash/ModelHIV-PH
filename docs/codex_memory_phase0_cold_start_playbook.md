# Codex Memory Context: Phase 0 Cold-Start Playbook

This document is a **conversation-memory-based Phase 0 playbook**, not a fresh repository audit.

Its purpose is to make the early discovery stage usable when the variable universe is extremely large and the operator does not want to guess every candidate variable by hand.

The governing rewrite is:

> Start with broad domain categories.
> Harvest widely.
> Extract softly.
> Preserve provenance.
> Let cheap filters and scoring narrow the field.
> Then use ablation, benchmark, and scientific gates to decide what survives.

---

## 1. Category List

The memory-consistent Phase 0 cold start begins with categories, not with a manually curated perfect variable list.

Recommended broad categories include:

- epidemiology and registry signals
- clinical severity and co-infections
- care cascade and service delivery
- economics and affordability
- logistics and transport friction
- mobility and migration
- stigma, behavior, and social determinants
- demography and population structure
- geography and environment
- policy and governance
- digital traces and information environment
- diagnostics and supply chain

The associated retrieval rule is:

- use a general embedder for broad discovery
- keep the categories as explicit domain silos where possible
- reserve narrow biomedical models for downstream refinement on already-biomedical corpora

These categories should stay broad at first.

---

## 2. Philippine Source Map

For the Philippine HIV-oriented baseline, the conversation repeatedly emphasized a source map like this:

### Official anchor layer

- DOH Philippines
- HARP
- WHO country pages
- UNAIDS country pages
- UN system sources
- PSA

### Behavioral and survey layer

- NDHS
- IHBSS
- related public survey and implementation-report sources when available

### Economic and household layer

- FIES
- PSA economic bulletins
- World Bank / ADB style overlays where useful

### Geospatial and logistics layer

- OpenStreetMap-derived geospatial resources
- transport and routing datasets
- mobility or colocation datasets where legitimately accessible

### Scientific literature layer

- PubMed
- Crossref
- OpenAlex
- Semantic Scholar
- arXiv
- bioRxiv
- manual seeds
- targeted scraping

---

## 3. Initial Query Bank

The initial query bank should cover both HIV-direct and upstream-determinant discovery.

### HIV-direct examples

- drivers of loss to follow-up in HIV care
- barriers to ART adherence
- causes of delayed HIV diagnosis
- predictors of virologic failure
- HIV stigma and testing
- HIV key population barriers

### Upstream examples

- determinants of out-of-pocket healthcare expenditure
- transport cost barriers to care
- rural travel-time barriers to clinic use
- migration and continuity of care
- stigma and disclosure barriers
- poverty and healthcare access
- diagnostics turnaround delays
- supply-chain bottlenecks in archipelagos
- climate and service disruption

The memory-consistent rule is to use multiple silos or domain buckets, not one narrow query family.

---

## 4. Evidence-Quality Rules

Cold-start discovery is intentionally broad, but it is not allowed to be sloppy about scientific status.

The rules remain:

- preserve source tier
- preserve locator metadata
- preserve extraction provenance
- separate candidate-only evidence from direct-truth-eligible evidence
- do not treat broad discovery hits as automatically fit-worthy

This means:

- harvest widely
- extract softly
- preserve provenance
- narrow later

not:

- force early certainty
- or promote early guesses into scientific claims

---

## 5. Candidate-Only vs Promotion-Eligible

### Candidate-only evidence

This includes:

- broad upstream literature hits
- weak overlap signals
- exploratory extracted concepts
- low-confidence or indirect measures
- chart-derived or proxy-derived quantities not yet validated

Candidate-only evidence is useful for:

- discovery
- retrieval
- ontology widening
- ablation design

### Promotion-eligible evidence

This requires stronger support:

- verifiable references
- acceptable provenance
- acceptable extraction quality
- usable alignment to the model grid
- successful ablation or benchmark contribution
- passing scientific gates

This is the line between Phase 0 curiosity and Phase 3 admissibility.

---

## 6. Cold-Start Operational Rule

The most important operational takeaway is:

> Do not begin by trying to enumerate the perfect variable list.
> Begin by defining broad categories, source maps, and query families.
> Then let the widening stack produce candidates and let the later gates decide what survives.

That is the most memory-consistent way to start Phase 0 without freezing on the size of the search space.
