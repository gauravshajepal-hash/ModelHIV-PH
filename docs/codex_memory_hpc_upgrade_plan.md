# Codex Memory HPC Upgrade Plan

This document records the **agreed acceleration roadmap** from our Codex conversations.

It is intentionally conservative.

The goal is not to rewrite the full stack into a GPU-first system immediately.
The goal is to identify:

- what the current baseline is,
- where tensorization is safe and worthwhile,
- where runtime boundaries should exist,
- what this specific machine can handle,
- and what conditions must be met before migrating toward GPyTorch, JAX/NumPyro, or Ray.

---

## 1. Current Baseline

The current architecture is a **hybrid scientific pipeline**, not a pure HPC stack.

### 1.1 Core scientific backbone

The authoritative backbone remains:

- DuckDB + Parquet for canonical storage
- JSON for manifests and summaries
- FAISS for production retrieval over curated candidate space
- explicit provenance and reference verification
- dual-lane literature scoring
- candidate / curated / active separation
- mechanistic HIV modeling with strong scientific gates

### 1.2 Discovery and exploration sidecars

The sidecar layer includes:

- OpenResearcher as optional discovery
- Docling as a strong parser when safe
- BGE-M3 as an optional embedding upgrade
- Chroma for document/chunk retrieval and curated filtered exploration

These tools are useful, but they are not the canonical storage and scientific-gating backbone.

### 1.3 Current numerical posture

At present, the system is not tensor-native end to end.

That is acceptable.

This pipeline includes:

- ETL
- parsing
- schema validation
- provenance handling
- registry management
- mechanistic model updates

Only some of those steps benefit materially from tensorization.

---

## 2. Safe Tensorization Upgrades

We agreed that tensorization should be introduced **selectively**, not universally.

## 2.1 Phase 0

Phase 0 should remain mostly:

- file-centric
- ETL-centric
- provenance-centric
- storage-centric

Safe tensorization in Phase 0:

- embedding generation
- similarity scoring
- some bulk numeric normalization

Do **not** force all Phase 0 data structures into tensors.

That would:

- increase complexity,
- increase memory risk,
- and make the pipeline more fragile without proportionate scientific benefit.

## 2.2 Phase 1

Phase 1 is the best early target for more tensor-native math.

Safe upgrades:

- unit normalization as tensor operations
- scaling / robust standardization as tensor operations
- aligned feature assembly as tensor operations
- subgroup and mask assembly as tensor operations

Keep outside the hot tensor path:

- manifests
- provenance tables
- source metadata
- artifact summaries

## 2.3 Phase 2

Phase 2 is another good target for tensorization.

Safe upgrades:

- scoring
- ranking
- linkage scoring
- selection math
- blockwise candidate screening

Do **not** hide:

- registry semantics
- provenance semantics
- admissibility logic

inside an opaque tensor-only implementation.

## 2.4 Phase 3

Phase 3 is the main place where heavier numerical acceleration may matter.

Safe upgrades:

- transition matrix construction
- state update math
- low-rank parameter effects
- subgroup-axis computations
- batched forward simulation

This is where PyTorch tensor use can become much more important.

## 2.5 Phase 4

Phase 4 should only become a major tensor/distributed engine if:

- the scientific core is stable,
- the rollout workload is real,
- and distributed simulation is the actual bottleneck.

---

## 3. Runtime Boundary Design

One of the most important agreed principles is:

> be tensor-native **within** a runtime, but be explicit about crossing runtime boundaries

This means:

- do not casually bounce data between CPU and GPU,
- do not casually bounce data between PyTorch and JAX,
- do not create hidden conversions inside loops.

## 3.1 Canonical runtime today

The safest current posture is:

- CPU / storage / ETL for Phase 0 and metadata-heavy logic
- PyTorch for selected tensor-heavy numerical work

## 3.2 PyTorch boundary

If a stage is accelerated with PyTorch, the rule should be:

- convert once into a tensor-friendly representation,
- do the heavy math there,
- materialize stable artifacts out again

not:

- repeatedly convert row tables to tensors and back inside the inner loop.

## 3.3 JAX boundary

We did discuss the idea of an explicit boundary operator \(\mathcal B\) between PyTorch and JAX.

That is a good future design, but only if we intentionally adopt a split-runtime architecture.

The rule would be:

- PyTorch block finishes
- explicit conversion once
- JAX block begins

It should **not** be:

- PyTorch -> JAX -> Python -> CPU -> JAX again in the middle of simulation

## 3.4 Distributed boundary

Ray or other distributed orchestration should sit **outside** the core numerical engine.

That means:

- Ray coordinates workers
- workers run the already-defined model
- the model itself should not be written around Ray assumptions from day one

---

## 4. Machine-Specific RAM / VRAM Guardrails

These guardrails are specific to the local machine context we checked.

## 4.1 Current machine facts

Current local machine state from our checks:

- RAM: about `16 GB`
- CPU: `32` logical processors
- GPU: `RTX 4070 Laptop GPU`
- usable GPU memory visible via PyTorch: about `8 GB`
- JAX currently detects **CPU only**, not GPU

## 4.2 Immediate implications

This means:

- a pure GPU-first design is unsafe as a baseline
- Phase 0 must remain conservative with downloads, parsing, and working-set size
- very large PDF/layout jobs must be explicitly bounded
- tensorized model work should be batched, not “load everything into VRAM”

## 4.3 VRAM guardrails

Practical rules:

- do not assume the full corpus can live in VRAM
- keep embeddings and candidate matrices batched
- avoid simultaneous large PDF OCR/layout workloads and large model tensors on GPU
- free GPU tensors aggressively between major stages if needed

## 4.4 RAM guardrails

Practical rules:

- do not parse the entire large corpus in one unbounded pass
- use bounded working sets
- prefer metadata-first harvest over full-document download everywhere
- make heavy binary documents a capped subset
- write stage outputs early and often so long runs are resumable

## 4.5 Safety rule after prior destructive failure

Because you reported an earlier run that crashed and deleted files, the operational rule should be:

- no long monolithic run without stage outputs
- every heavy phase should be resumable
- every heavy phase should have explicit resource budgets
- no “best effort” hidden fallback that silently keeps going with corrupted assumptions

---

## 5. Phased Migration Triggers

The point is not “use every HPC library now.”
The point is to know **when** each one becomes justified.

## 5.1 GPyTorch trigger

Use GPyTorch only if all of the following are true:

- a specific alignment subproblem really is GP-like,
- that GP interpolation is a measured bottleneck,
- simpler alignment or weighted reconciliation is not sufficient,
- the data volume and kernel structure are appropriate for GPyTorch.

Do **not** use GPyTorch just because it is available.

## 5.2 NumPyro / JAX trigger

Consider NumPyro / JAX only if:

- we explicitly decide to migrate the Phase 3 probabilistic core,
- we define a clear runtime boundary from the existing implementation,
- JAX GPU is working on the target machine,
- and the expected gain justifies the migration cost.

Until then:

- JAX is a future migration option, not the current operational baseline.

## 5.3 Ray trigger

Consider Ray only if:

- Phase 4 is operationally real,
- many-rollout simulation is truly the bottleneck,
- distributed worker orchestration is actually needed,
- and the core model can already run cleanly in a single-process setting.

Do **not** introduce Ray just to look “production-grade.”

## 5.4 BGE-M3 trigger

Use BGE-M3 as a default only if:

- embedding quality clearly beats the lighter local backend,
- resource usage remains acceptable,
- and retrieval quality improvement is worth the cost.

Otherwise keep it as an optional higher-quality mode.

## 5.5 Docling-heavy parsing trigger

Use heavier Docling-first parsing only when:

- document size and page count are within safe bounds,
- the document class benefits from layout parsing,
- and the parse stage is not dominated by large PDFs that can be handled more cheaply by fallback paths.

---

## 6. Recommended Near-Term Build Order

The near-term roadmap we implicitly settled on is:

1. keep the current scientific backbone
2. add more PyTorch tensor math in Phase 1
3. add more PyTorch tensor math in Phase 2
4. add more PyTorch tensor math in Phase 3 transitions
5. measure bottlenecks
6. only then decide whether:
   - GPyTorch is warranted
   - JAX/NumPyro migration is warranted
   - Ray is warranted

This preserves:

- scientific rigor,
- provenance,
- reproducibility,
- and machine safety

while still moving toward a higher-performance system.

---

## 7. Bottom Line

The agreed HPC strategy is:

- **not** “rewrite everything for GPU immediately”
- **not** “ban all CPU-native structures”
- **not** “introduce JAX and Ray before the model is stable”

It is:

- keep the scientific core stable,
- tensorize the mathematically hot parts,
- enforce explicit runtime boundaries,
- respect the real limits of this machine,
- and promote heavier HPC infrastructure only when the bottlenecks justify it.

