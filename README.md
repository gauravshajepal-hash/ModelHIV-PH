# ModelHIV-PH

ModelHIV-PH is a staged HIV evidence-mining, causal-structure, and semi-Markov cascade modeling platform focused on the Philippines.

The system is designed to do three things safely:

1. turn literature and official records into structured subparameters
2. reduce those subparameters into smaller, auditable factor sets
3. use the retained factors to forecast the HIV cascade and compare against reference series such as HARP and official anchors

This repo is intentionally built as a pipeline rather than one giant model. That keeps the system inspectable, easier to debug, and safer to run on a single Windows laptop with bounded RAM and an 8 GB GPU.

## What The Repo Does

At a high level, the repository:

- harvests literature and official data from broad source families
- parses metadata, PDFs, OCR text, and chunked document spans
- extracts candidate subparameters with provenance, geography, time, and typed payloads
- normalizes those candidates into Province x Month x Feature tensors
- learns latent province / region / national block states with uncertainty-aware reconciliation
- builds multiscale factor surfaces and propagated factor uncertainty
- learns temporal sparse-plus-low-rank structure over latent block innovations
- freezes structural Phase 2 payloads for reproducible downstream use
- feeds those structural payloads into the transition-research frontier and keeps broad rescue-core as a benchmark
- backtests that model against frozen historical reference trajectories
- emits policy-facing outputs downstream

## Architecture

```mermaid
flowchart LR
    A["Phase 0\nHarvest, parse, OCR, extract"] --> B["Registry\nAccepted subparameters"]
    B --> C["Phase 1\nNormalization and tensors"]
    C --> D["Phase 15\nLatent block states,\nreconciliation, factor tournament"]
    D --> E["Phase 2\nLatent temporal structure learning\nplus multiscale support"]
    E --> F["Phase 3\nTR-V2 frontier and\nbroad rescue-core benchmark"]
    F --> G["Phase 4\nPolicy simulation and runtime assurance"]
```

## Why The Pipeline Is Split

The repository is not trying to learn one giant graph over every mined variable at once.

Instead, it uses a layered latent-to-structure strategy:

```mermaid
flowchart TD
    A["Accepted subparameters"] --> B["Within-block candidate banks"]
    B --> C["Phase 15 latent block states\nwith posterior uncertainty"]
    C --> D["Multiscale factor construction\nand uncertainty propagation"]
    C --> E["Phase 2 latent temporal graph\n(sparse direct + low-rank hidden)"]
    D --> F["Phase 2 multiscale support graph"]
    E --> G["Frozen structural payload"]
    F --> G
    G --> H["TR-V2 hazard priors,\nhidden shock channels,\nsupport modulation"]
```

This is deliberate.

- It keeps memory use bounded.
- It avoids forcing unrelated variables into one dense graph.
- It makes the causal structure easier to audit.
- It reduces the chance of a huge unstable adjacency matrix blowing up the run.

## Repository Layout

- `src/epigraph_ph/core`
  Shared contracts, disease-plugin interfaces, province archetypes, and runtime assurance support.
- `src/epigraph_ph/plugins/hiv.py`
  The HIV plugin. This is the domain rulebook for the repo.
- `src/epigraph_ph/adapters/structured_sources.py`
  First-class structured adapters for official and literature source families.
- `src/epigraph_ph/phase0`
  Harvest, parse, OCR, extract, boundary validation, and support artifacts.
- `src/epigraph_ph/registry`
  Source and subparameter registry builders.
- `src/epigraph_ph/phase1`
  Measurement normalization and tensor construction.
- `src/epigraph_ph/phase15`
  Latent block-state estimation, missing-information reconciliation, multiscale factor construction, and survival-tournament selection.
- `src/epigraph_ph/phase2`
  Latent temporal graph estimation, exact sparse-plus-low-rank optimization, multiscale temporal support, frozen structural payloads, and benchmark compatibility artifacts.
- `src/epigraph_ph/phase3`
  Broad rescue-core benchmark lineage plus the transition-research frontier, including `TR-V2` structural Phase 2 consumption.
- `src/epigraph_ph/phase4`
  Policy evaluation and runtime assurance outputs.
- `tests`
  Contract, math, artifact, and regression coverage.
- `scripts`
  Local helpers including OCR serving, WSL bootstrap, and probes.
- `artifacts`
  Run outputs and analysis products.

## The HIV Plugin

The HIV plugin in [D:\EpiGraph_PH\src\epigraph_ph\plugins\hiv.py](/D:/EpiGraph_PH/src/epigraph_ph/plugins/hiv.py) declares the core modeling contract:

- determinant silos
- query banks
- source adapters
- Phase 0 boundary rules
- Phase 1 normalization rules
- Phase 1.5 tournament and Bayesian search ranges
- Phase 2 graph constraints
- Phase 3 priors and frozen-backtest tournament settings
- Phase 4 policy settings
- gold-standard checks for each phase

In plain English, the plugin is where the repo says:

"For HIV in the Philippines, these are the kinds of evidence, states, priors, constraints, and outputs we accept."

## Phase 0: Harvest, Parse, OCR, Extract

Main modules:

- [D:\EpiGraph_PH\src\epigraph_ph\phase0\pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase0/pipeline.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase0\boundary_models.py](/D:/EpiGraph_PH/src/epigraph_ph/phase0/boundary_models.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase0\literature_candidates.py](/D:/EpiGraph_PH/src/epigraph_ph/phase0/literature_candidates.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase0\semantic_benchmark.py](/D:/EpiGraph_PH/src/epigraph_ph/phase0/semantic_benchmark.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase0\shard_materializer.py](/D:/EpiGraph_PH/src/epigraph_ph/phase0/shard_materializer.py)

### What Phase 0 does

Phase 0 is the evidence factory.

It:

- harvests source metadata from official and literature platforms
- snapshots remote PDFs and pages when budget allows
- parses born-digital PDFs and metadata docs
- optionally runs GPU-backed OCR for hard documents
- extracts numeric observations and soft text-derived candidates
- validates extracted candidates through strict boundary models
- writes accepted and rejected outputs separately
- builds family-specific candidate banks
- writes alignment tensors and retrieval artifacts

### Structured source families

The current source stack includes:

- WHO
- UNAIDS
- DOH Philippines
- UN
- NDHS
- YAFS
- FIES
- PhilGIS / PSGC
- PhilHealth
- Google Mobility
- World Bank WDI
- DOH facility statistics
- transport proxies
- PubMed
- arXiv
- bioRxiv
- OpenAlex
- Crossref
- Semantic Scholar

### The evolving Phase 0 boundary

Phase 0 does not use one giant flat schema anymore.

It uses:

- one stable outer envelope
- multiple family-specific payloads
- multiple JSON outputs instead of one monolith

The stable envelope contains:

- provenance
- geo binding
- time binding
- confidence
- extraction mode
- evidence references
- signal family

The current family payloads are:

- `PopulationMeasure`
- `LogisticsAccess`
- `BehaviorSignal`
- `ServiceCapacity`
- `EconomicConstraint`
- `PolicyEnvironment`
- `CascadeObservation`

This design matters because the ontology is intentionally soft.

We want Phase 0 to expand subparameters aggressively, but we still need typed structure at the boundary so later phases can reason over the rows safely.

### Phase 0 outputs

Phase 0 writes split outputs such as:

- accepted candidates
- rejected candidates
- boundary validation summary
- family-specific candidate banks
- alignment tensors
- schema summaries
- literature review reports
- curated bibliography
- review queues
- tool stack and resource manifests

### Phase 0 acceptance policy

The current acceptance logic is designed to stay soft without becoming sloppy.

It keeps:

- validity checks
- support checks
- leakage prevention
- finite-value checks
- shape checks

It now also does better geo binding for literature candidates by using:

- explicit geo fields
- geo mentions
- literature titles
- source title context
- `query_geo_focus`

And it uses softer acceptance for well-supported text signals by recognizing:

- typed signal families
- evidence-rich titles and excerpts
- literature references
- soft ontology tags
- linkage targets
- text-supported prior signals even when the row is not a direct official anchor

## Registry

Main module:

- [D:\EpiGraph_PH\src\epigraph_ph\registry\subparameters.py](/D:/EpiGraph_PH/src/epigraph_ph/registry/subparameters.py)

The registry stage is the bridge between extraction and modeling.

It combines:

- accepted Phase 0 candidates
- wide-sweep literature bank rows
- determinant silo context

into one explicit subparameter registry.

This matters because Phase 1 reads the registry.

The code now also falls back safely to accepted Phase 0 candidates if the registry is missing or empty, so a manual run cannot silently collapse to zero normalized rows just because the registry command was skipped.

## Phase 1: Measurement Normalization

Main modules:

- [D:\EpiGraph_PH\src\epigraph_ph\phase1\pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase1/pipeline.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase1\normalization_helpers.py](/D:/EpiGraph_PH/src/epigraph_ph/phase1/normalization_helpers.py)

### What Phase 1 does

Phase 1 turns heterogeneous evidence into comparable model inputs.

It produces:

- `aligned_tensor`
- `standardized_tensor`
- `denominator_tensor`
- `missing_mask`
- `quality_weight_tensor`
- `normalized_subparameters.json`
- `parameter_catalog.json`

### Important plain-English terms

- `denominator tensor`
  A table of the population or reference bases needed to convert counts into comparable rates or shares. Without this, counts from large and small provinces get mixed unfairly.
- `missing mask`
  A yes-or-no table showing where data is actually missing. It stops the model from confusing missing with zero.
- `density conversion`
  Turning raw counts into relative quantities like per-capita or per-PLHIV values.
- `winsorization`
  Capping extreme outliers so a few broken values do not distort the whole scale.
- `log1p`
  A log transform that can handle zero values.
- `Box-Cox`
  Another way to reshape skewed data so it behaves more cleanly.
- `robust scaling`
  Scaling by statistics like the median and IQR instead of more outlier-sensitive choices.

### Why Phase 1 is necessary

Without Phase 1, the repo would be mixing:

- counts
- percentages
- rates
- capacities
- soft supports
- national rows
- subnational rows
- sparse and dense evidence

as if they were all the same kind of number.

They are not.

## Phase 15: Latent States, Reconciliation, And Factor Selection

Main modules:

- [D:\EpiGraph_PH\src\epigraph_ph\phase15\pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/pipeline.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase15\v2_engine.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/v2_engine.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase15\multiscale_factors.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/multiscale_factors.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase15\bayesian_survival.py](/D:/EpiGraph_PH/src/epigraph_ph/phase15/bayesian_survival.py)

### What Phase 15 does

Phase 15 is the latent-state bridge between normalized evidence and temporal structure learning.

It:

- estimates latent province, region, and national block states
- propagates posterior uncertainty for those latent states
- solves missing-information reconciliation problems against stronger aggregate evidence
- builds multiscale factor surfaces and factor-level uncertainty tensors
- runs the mesoscopic survival tournament
- optionally tunes the tournament with Bayesian optimization on top of the same objective

### Why this phase exists now

The repo no longer jumps directly from normalized tensors to Phase 2 graph structure.

Instead, it first estimates:

- a latent block trajectory
- uncertainty around that trajectory
- a reconciled correction field when stronger national or regional evidence disagrees with weak provincial support

In plain English:

- Phase 15 says what the hidden HIV system probably looks like
- and how sure we are about each latent estimate

That is what Phase 2 now learns temporal structure over.

### Survival tournament in Phase 15

Bayesian optimization now tunes the survival tournament; it does not replace it.

The search space includes:

- tournament score weights
- survivor budgets per block
- representation mix preference across unclumped, clumped, and network factors

If the optimized tournament beats the baseline objective, the optimized pool becomes active. If not, the system keeps the baseline pool.

## Phase 2: Latent Temporal Structure Learning

Main modules:

- [D:\EpiGraph_PH\src\epigraph_ph\phase2\pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/pipeline.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase2\latent_temporal_graph.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/latent_temporal_graph.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase2\temporal_optimizer.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/temporal_optimizer.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase2\multiscale_dag.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/multiscale_dag.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase2\structural_payload.py](/D:/EpiGraph_PH/src/epigraph_ph/phase2/structural_payload.py)

### What Phase 2 does

Phase 2 is now a latent, temporal, uncertainty-aware structure-learning layer.

It does:

- removes self-persistence from latent block trajectories
- builds multi-lag temporal designs over those latent trajectories
- fits an exact joint sparse-plus-low-rank temporal operator
- keeps direct temporal effects and hidden shared structure separate
- propagates Phase 15 uncertainty into the fitting and selection objective
- runs multiscale temporal support estimation as a support-only corroboration layer
- freezes a structural payload for reproducible downstream use
- keeps a separate compatibility payload for benchmark consumers

### What the three Phase 2 surfaces mean

- `direct temporal surface`
  The lagged block-to-block temporal hypotheses that Phase 3 is allowed to use structurally.
- `hidden-driver surface`
  Shared low-rank structure that becomes hidden shock channels, not direct mechanistic edges.
- `multiscale support surface`
  Factor-level corroboration that can strengthen or weaken prior confidence, but does not create new direct hazard terms.

### What changed scientifically

Phase 2 is no longer a same-time NOTEARS / block-DAG layer.

It now learns temporal hypotheses from latent innovations. That makes the output closer to:

- direct lagged predictive structure
- hidden shared-driver structure
- uncertainty-aware support summaries

instead of one mixed graph over observed factor snapshots.

## Phase 3: Broad Benchmark Plus Structural Frontier

Main modules:

- [D:\EpiGraph_PH\src\epigraph_ph\phase3\pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/pipeline.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase3\_lineage\rescue_core.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/_lineage/rescue_core.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase3\frontier\tr_v2.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/frontier/tr_v2.py)

### States

The model uses explicit HIV cascade states:

- `U` undiagnosed
- `D` diagnosed
- `A` on ART
- `V` virally suppressed / documented suppression
- `L` lost or disengaged from care

### What Phase 3 does now

Phase 3:

- builds observation ladders from official and HARP-aligned sources
- keeps the broad rescue-core path as a benchmark lineage
- runs the transition-research frontier as the main structural consumer of Phase 2
- fits quarter-level transition-hazard branches on top of the HIV cascade states
- lets direct Phase 2 edges become priors on hazard transitions
- lets hidden Phase 2 structure become separate hidden shock channels
- keeps multiscale Phase 2 outputs as support-only prior modulation
- runs frozen-history backtests
- compares against simple baselines
- keeps explicit experiment families such as `MECH`, `DECOMP`, `PEAK`, `AGE`, and `TR-V2`

### Broad benchmark vs frontier

- `rescue_core`
  The broad multiyear benchmark and compatibility consumer.
- `transition_research`
  The winning branch family for quarter-level transition experiments.
- `TR-V2`
  The new structural frontier that consumes frozen Phase 2 structural payloads directly.

This split is deliberate. The repo no longer treats broad rescue-core as the main place where new Phase 2 semantics live.

## Testing Strategy

The repository now keeps two useful regression layers for the structural frontier:

- fast checked-in synthetic tests for interface semantics and failure modes
- slower artifact-backed integration tests against the frozen `smoke-latent-blocks` run

The slow frontier benchmark is intentionally opt-in and can be enabled with:

`EPIGRAPH_RUN_SLOW_INTEGRATION=1`

### Inference engines

Phase 3 supports multiple inference paths:

- Torch MAP
- JAX SVI
- JAX NUTS

WSL2 GPU interop is now supported for true Torch-CUDA to JAX-GPU DLPack handoff on machines where the Linux environment is provisioned correctly.

## Phase 4: Policy Layer

Main modules:

- [D:\EpiGraph_PH\src\epigraph_ph\phase4\pipeline.py](/D:/EpiGraph_PH/src/epigraph_ph/phase4/pipeline.py)
- [D:\EpiGraph_PH\src\epigraph_ph\phase4\policy_analysis.py](/D:/EpiGraph_PH/src/epigraph_ph/phase4/policy_analysis.py)

Phase 4 takes the cascade outputs and turns them into policy comparisons and sensitivity outputs.

This is the action layer, not the disease-dynamics layer.

## Large-Corpus Strategy

The repo does not try to parse or graph 100,000 papers in one pass.

The safe production strategy is:

```mermaid
flowchart TD
    A["Sharded metadata harvest"] --> B["Bounded parse/extract slices"]
    B --> C["Registry merge"]
    C --> D["Phase 1 normalization"]
    D --> E["Phase 1.5 survival tournament"]
    E --> F["Phase 2 sharded block graphs"]
    F --> G["Merged retained factors"]
    G --> H["Phase 3 backtest and forecast"]
```

### Why this matters

- RAM is tighter than disk on the laptop.
- OCR is expensive even with GPU support.
- Exact DAG discovery should happen on retained factor sets, not giant raw candidate pools.
- The bridge graph should stay small by design.

## Typical Commands

### Install and test

```powershell
python -m pip install -e .[core]
python -m pytest tests -q
python -m epigraph_ph.cli.main --help
```

### Bounded end-to-end run

```powershell
python -m epigraph_ph.cli.main phase0 build --run-id demo-bounded --plugin hiv --corpus-mode massive --target-records 500 --working-set-size 150
python -m epigraph_ph.cli.main registry build --run-id demo-bounded --plugin hiv
python -m epigraph_ph.cli.main phase1 build --run-id demo-bounded --plugin hiv --profile hiv_rescue_v2
python -m epigraph_ph.cli.main phase15 build --run-id demo-bounded --plugin hiv --profile hiv_rescue_v2
python -m epigraph_ph.cli.main phase2 build --run-id demo-bounded --plugin hiv --profile hiv_rescue_v2
python -m epigraph_ph.cli.main phase3 tournament-frozen-backtest --run-id demo-bounded --plugin hiv --profile hiv_rescue_v2 --phase3-inference torch_map
```

### Metadata-first large harvest

```powershell
python -m epigraph_ph.cli.main phase0 harvest --run-id prod-harvest-s00 --plugin hiv --corpus-mode massive --target-records 100000 --metadata-only --query-shard-count 12 --query-shard-index 0
```

Repeat across shard indices, merge the shard manifests, then parse bounded slices instead of trying to parse the full merged corpus in one shot.

## What To Look At In Artifacts

Typical high-value artifacts are:

- `phase0/extracted/boundary_validation_report.json`
- `phase0/extracted/family_candidate_banks_manifest.json`
- `phase1/normalization_report.json`
- `phase15/factor_survival_tournament.json`
- `phase15/factor_survival_bayesian_optimization.json`
- `phase2/feature_matrix_mix_report.json`
- `phase2/dag_projection_report.json`
- `phase2/phase3_target_blankets.json`
- `phase3_frozen_backtest_tournament/representation_tournament.json`

## Current Practical Constraints

The repo is strongest on:

- staged validation
- provenance
- graph sparsity discipline
- bounded computation
- frozen-history backtesting

The repo is weaker on:

- Phase 0 extraction precision on broad literature
- geo binding on generic international papers
- very large live OCR budgets
- making JAX/NUTS robust enough to be the default inference path

## Current Design Commitments

The current design deliberately prefers:

- soft ontology at extraction
- strict typed boundaries at phase transitions
- smaller graphs instead of one giant graph
- measurable tournaments instead of handwavy promotion rules
- backtests over intuition

## Quick Mental Model

If you want the shortest plain-English summary:

- Phase 0 finds possible signals.
- Registry keeps the candidate bank explicit.
- Phase 1 puts the numbers on a comparable scale.
- Phase 1.5 asks which clumps survive a holdout tournament.
- Phase 2 learns a sparse graph over the survivors.
- Phase 3 uses those survivors to simulate the HIV cascade.
- Phase 4 compares decisions on top of that cascade.

That is the current system.
