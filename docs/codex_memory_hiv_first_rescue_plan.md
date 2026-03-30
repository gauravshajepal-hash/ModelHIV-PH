# Codex Memory Context: HIV-First Rescue Plan

This document is a **conversation-memory-based rescue plan**, not a fresh repository audit.

Its purpose is to convert the broad architecture into a **machine-safe, benchmark-oriented HIV-first build strategy**.

The central rule is:

> Keep Phase 0 wide.
> Keep the HIV scientific core narrow.
> Make every determinant earn entry by winning a controlled tournament.

---

## 1. Objective

The immediate goal is **not**:

- universal disease coverage
- full syndemic control
- RL-first policy optimization
- a giant determinant soup

The immediate goal is:

> build a narrow, mechanistically honest, Philippines-specific HIV model that can win a well-defined benchmark before the architecture expands further.

That is the shortest realistic path to scientific credibility.

---

## 2. The Main Problem

The architecture is strongest when:

- discovery is broad
- promotion is strict
- inference is mechanistic

It is weakest when:

- too many weak determinants leak into Phase 3
- Phase 2 remains too vague
- the observation model is under-specified

So the rescue plan solves one thing above all:

> prevent Phase 0 breadth from destabilizing the HIV scientific core.

---

## 3. Freeze the Minimum Viable HIV Core

Before adding more determinants, freeze a reference HIV engine with the smallest structure that still respects the science.

## 3.1 Frozen latent structure

Keep:

- province hierarchy
- month time scale
- cascade states:
  - `U`
  - `D`
  - `A`
  - `V`
  - `L`
- KP axis
- age axis
- sex axis
- coarse CD4 overlay
- duration buckets

Do **not** add:

- dense determinant interactions
- extra disease-state axes
- multi-disease coupling
- policy-control complexity

## 3.2 Frozen observation ladder

The frozen core should explicitly distinguish:

- direct observed diagnosis-related quantities
- direct observed ART-related quantities
- direct or bounded suppression-related quantities
- testing/documentation quantities
- prior-only or bridge-only quantities

The model should not be allowed to invent epidemic structure that outruns this ladder.

---

## 4. Observation-First Rebuild

This is the most important scientific correction.

The HIV engine must be anchored by the observation process first, not by determinant ambition.

## 4.1 Hard observation priorities

The first-fitting backbone should be:

- diagnosed counts or diagnosed-stock surfaces
- ART counts or ART-stock surfaces
- suppression-related observables
- testing/documentation surfaces
- mortality or attrition signals when available
- province / region / national reconciliation

## 4.2 What determinants are allowed to do

Determinants may modify:

- incidence allocation
- diagnosis rate
- linkage rate
- retention / attrition
- suppression transition behavior

Determinants may **not**:

- freely replace observation structure
- dominate latent stock reconstruction
- override hierarchy and reconciliation discipline

---

## 5. Replace Phase 2 Causal Ambiguity with Determinant Tournaments

Do not let one giant causal-discovery dream sit between Phase 0 and Phase 3.

Instead, use **block tournaments**.

## 5.1 Candidate blocks

Use the blocks already defined in memory:

- population
- behavior
- biology / clinical
- logistics / health systems
- mobility / network
- economics / access
- policy / implementation
- environment / disruption

## 5.2 Tournament rule

Each block is tested **against the frozen HIV core**, not against the entire wide candidate universe at once.

For each block:

1. pick a small shortlist
2. fit the shortlist only on top of the frozen core
3. compare against the frozen core
4. keep only variables or composites that improve protected metrics without harming mechanistic honesty

## 5.3 Champion policy

Each block gets:

- `1` champion
- `1` backup
- optional `1` composite

No block gets to flood the model with dozens of variables.

---

## 6. Promotion Budget

The main defense against overfitting is a hard promotion budget.

## 6.1 Suggested per-block budget

- population: max `2`
- behavior: max `2`
- biology / clinical: max `3`
- logistics / health systems: max `2`
- mobility / network: max `2`
- economics / access: max `2`
- policy / implementation: max `1`
- environment / disruption: max `1`

## 6.2 Suggested global budget

For the first strong HIV scientific pass:

- total promoted determinant features: max `10–15`
- total interaction terms: max `5`
- total subgroup-specific determinant modifiers: max `5`

This forces discipline.

---

## 7. Three Evidence Levels for Every Variable

Every candidate should be explicitly tagged as one of:

- `exploratory`
- `scientific_retained`
- `mechanistic_active`

## 7.1 Exploratory

Useful for:

- retrieval
- ontology widening
- linkage discovery
- hypothesis generation

Not allowed to affect the main fitted HIV engine directly.

## 7.2 Scientific retained

Useful for:

- structured ablation
- benchmark comparison
- candidate promotion contests

Still not guaranteed to enter the active mechanistic engine.

## 7.3 Mechanistic active

Reserved for variables that:

- survive tournament screening
- survive holdout benchmarking
- preserve mechanistic honesty
- preserve hierarchy and observation discipline

This is the only class that enters the narrow HIV scientific core.

---

## 8. Stress Lab Before Any Claim

Every promoted HIV model must pass a stress lab.

## 8.1 Required stress tests

- year holdout
- region holdout
- province holdout where feasible
- missing-data perturbation
- noisy-input perturbation
- determinant dropout
- anchor removal sensitivity
- subgroup sparsity stress

## 8.2 Failure rule

If a determinant or block only helps under ideal conditions, it is not real enough to keep.

The stress lab exists to kill fragile wins.

---

## 9. Machine-Safe Run Design

This machine is roughly:

- `16 GB` RAM
- `8 GB` usable VRAM through PyTorch
- JAX CPU-only

So the rescue plan must be operationally conservative.

## 9.1 Safe workload split

CPU:

- harvest
- parse
- DuckDB / Parquet
- manifests
- provenance
- registry logic

GPU:

- embeddings in batches
- selective Phase 1 tensor assembly
- Phase 2 scoring / ranking math
- Phase 3 transition and forward-state computations

## 9.2 Hard safety rules

- no full-corpus-in-VRAM run
- no monolithic end-to-end run without stage outputs
- no giant dense all-axis tensor materialized eagerly
- no JAX migration yet
- no Ray yet

## 9.3 Two-resolution workflow

Use:

- **screening pass**
  - large candidate field
  - cheap scores
  - cheap retrieval
  - cheap ablations
- **scientific pass**
  - tiny promoted set
  - narrow Phase 3 model
  - expensive fit / validation / benchmark

This is how the architecture stays tractable on this machine.

---

## 10. Benchmark Strategy

Do not try to beat every external model everywhere immediately.

## 10.1 Wrong target

Wrong initial claim:

- “better than Spectrum, Naomi, and Optima overall”

That is too broad.

## 10.2 Right target

Right initial claim:

> better on a narrowly defined Philippines-specific benchmark:
> province-aware HIV cascade forecasting and subnational mechanistic consistency under messy local evidence.

## 10.3 Benchmark ladder

The model should first beat:

1. naive baselines
2. trend baselines
3. stripped local mechanistic baselines
4. internal rival variants

Only then should it claim progress against external model families, and even then only on the precise benchmark task actually tested.

---

## 11. Local Rival Suite

Before comparing to Spectrum / Naomi / Optima-style families, build a local rival suite:

- cascade-only model
- hierarchy-only model
- KP-only model
- KP + age + sex model
- KP + age + sex + CD4 model
- determinant-light model
- determinant-heavy model

This reveals which pieces are actually earning their keep.

---

## 12. Path to a First Benchmark Win

The shortest realistic path is:

1. freeze the narrow HIV core
2. strengthen the observation ladder
3. enforce hierarchy and mechanistic constraints
4. add KP x age x sex
5. add coarse CD4 overlay
6. run determinant tournaments block by block
7. keep only block champions
8. run the stress lab
9. compare against local rivals and naive baselines
10. define a narrow external-competitor claim only if the benchmark really supports it

---

## 13. What to Defer

Until the first HIV benchmark win exists, defer:

- big syndemic coupling ambitions
- large RL/control infrastructure
- Ray rollout systems
- JAX probabilistic migration
- giant multi-disease policy optimization
- overgrown determinant sets

That work is downstream.

---

## 14. Final Rescue Rule

The final rule is:

> The system wins by being wider in discovery, harder in promotion, narrower in the mechanistic core, and more adversarial in validation.

That is the most realistic path to a machine-safe, scientifically credible HIV-first model.
