# Codex Memory: Subgroup Topology, State Space, and Compute-Safe Implementation

This document records the subgroup and state-space design that best matches our Codex conversations.

It focuses on:

- the correct subgroup topology
- the correct tensor shapes
- sparse coupling rules
- what belongs in state vs feature vs overlay
- compute-safe implementation guidance for this machine

This is the **conversation-memory design target**, not a claim that every element is already fully implemented in code.

---

## 1. Core Principle

We agreed on a strong modeling rule:

> do not explode the main clinical state space unless the scientific gain clearly justifies the computational cost

This is the central design principle behind the subgroup topology.

It means:

- key populations should not be naively hard-coded into state labels,
- co-infections should not become a giant Cartesian multiplier in v1,
- age and sex should be explicit where they matter,
- CD4 should enter as a controlled severity overlay rather than a full primary axis,
- sparse coupling is preferred over dense all-to-all interaction structure.

---

## 2. Correct Subgroup Topology

We discussed **two very different kinds of subgroup structure**.

They should not live in the same place mathematically.

## 2.1 Population subgroups

These include:

- MSM
- TGW
- FSW
- clients of sex workers
- PWID
- non-KP partners
- remaining population

These are **population strata**, not alternative cascade states.

So they should live on an explicit subgroup axis:

\[
k \in \mathcal{K}
\]

and **not** inside the base state alphabet.

That means we do **not** want:

- `U_MSM`
- `D_MSM`
- `A_MSM`
- `U_PWID`
- `D_PWID`
- etc.

as separate main state labels.

Instead, we want:

- subgroup axis
- age axis
- sex axis
- cascade state axis

This matches the modeling decision we converged on in the Phase 3 extension.

## 2.2 Disease subgroups / co-infections

These include things like:

- TB
- hepatitis
- other co-infections or severity modifiers

These are not stable demographic strata in the same sense as KP structure.

They are better treated in v1 as:

- covariates
- dynamic modifiers
- severity context
- hazard modifiers

not as a new main Cartesian axis on the core latent state.

So:

- population subgroup structure belongs on explicit model axes
- co-infection structure belongs mainly in the feature/modifier layer in v1

## 2.3 CD4 structure

CD4 is important, but we explicitly decided:

- yes to CD4
- no to full primary-axis explosion in v1

So CD4 belongs in an **overlay**:

- coarse bins
- severity distribution
- progression / attrition / clinical severity modifier

not as a full top-level expansion of the state tensor on day one.

---

## 3. Correct Tensor Shapes

## 3.1 Minimal mechanistic latent tensor

The correct enriched latent tensor from our conversation is:

\[
X_{p,k,a,x,\sigma,\delta,t}
\]

where:

- \(p\): province
- \(k\): key-population group
- \(a\): age band
- \(x\): sex
- \(\sigma\): cascade state
- \(\delta\): duration bucket
- \(t\): time

This is the main care-state tensor.

## 3.2 Why this is the right structure

This preserves:

- province structure
- subgroup structure
- age structure
- sex structure
- clinical cascade structure
- time-in-state memory

without collapsing everything into one oversized state label space.

## 3.3 Coarse CD4 overlay

CD4 should be represented separately as:

\[
C_{p,k,a,x,g,t}
\]

where:

- \(g\): coarse CD4 severity bin

This is an overlay, not the main latent tensor.

For valid cells:

\[
\sum_g C_{p,k,a,x,g,t}=1
\]

and

\[
C_{p,k,a,x,g,t}\ge 0
\]

## 3.4 Feature/environment tensor

External drivers and promoted subparameters belong in a feature tensor like:

\[
Z_{p,t,f}
\]

or, when subgroup-resolution evidence exists:

\[
Z_{p,t,f,k,a,x}
\]

Examples:

- poverty
- transport friction
- migration
- remoteness
- stigma
- diagnostics reach
- service delivery quality
- TB incidence
- hepatitis burden
- social protection
- collective behavior metrics

## 3.5 Aggregated public outputs

The public cascade outputs are derived by summing across subgroup axes, not fit as separate disconnected systems.

Examples:

\[
\text{first95}_t = \frac{\text{diagnosed stock}}{\text{PLHIV stock}}
\]

\[
\text{second95}_t = \frac{\text{on-ART stock}}{\text{diagnosed stock}}
\]

\[
\text{third95}_t = \frac{\text{suppressed stock}}{\text{on-ART stock}}
\]

The key point:

- subgroup decomposition is internal,
- aggregate outputs are derived views.

---

## 4. Sparse Coupling Rules

This is where the earlier “independent subgroup batch” story becomes too strong.

We did **not** agree that subgroup trajectories are fully independent.

We explicitly discussed:

- partner spillover
- key-population turnover
- non-KP partners
- network and mobility effects

So the correct rule is:

> subgroups should be mostly factorized for computational efficiency, but sparsely coupled where the science requires it

## 4.1 Within-group transitions

Most care transitions are within the same subgroup cell:

- \(U \rightarrow D\)
- \(D \rightarrow A\)
- \(A \rightarrow V\)
- \(A \rightarrow L\)
- \(L \rightarrow A\)

These are primarily modeled within each:

\[
(p,k,a,x,\delta,t)
\]

cell.

## 4.2 Sparse cross-group coupling

Cross-group coupling should be allowed through:

### 1. Incidence decomposition

New infections are not just a national total.
They are allocated across subgroup strata:

\[
I_{p,k,a,x,t}
\]

and that allocation can depend on:

- subgroup structure
- partner structure
- mixing
- mobility

### 2. Partner spillover

The model should allow structured spillover from KP groups into partner groups and the broader population, not zero it out by assumption.

### 3. KP turnover

KP membership is not static forever.
Sparse turnover between categories can exist.

### 4. Shared environment

All subgroups in a province still experience common external conditions:

- logistics
- poverty
- transport friction
- stigma climate
- policy environment
- co-infection burden

This is another coupling mechanism.

## 4.3 What sparse coupling means mathematically

It means:

- do **not** make the subgroup coupling matrix dense by default
- do **not** make it identically zero by default

Instead:

- define a sparse coupling structure
- only allow scientifically plausible links
- heavily regularize them

This is the compute-safe middle ground.

---

## 5. What Belongs in State vs Feature vs Overlay

This is the most important design boundary.

## 5.1 State

Things that belong in the main latent state:

- HIV care-cascade status
  - `U`
  - `D`
  - `A`
  - `V`
  - `L`
- duration / time-in-state bucket
- subgroup axes that define the stratified population view:
  - province
  - KP group
  - age band
  - sex

These define the core mechanistic occupancy tensor.

## 5.2 Feature / environment

Things that belong in the feature or environment layer:

- poverty
- affordability
- remoteness
- migration
- transport friction
- service delivery quality
- diagnostics reach
- stigma indicators
- governance indicators
- social protection
- collective behavior indicators
- TB incidence
- hepatitis burden
- broader co-infection burdens
- broad upstream determinants generally

These modify transition behavior, incidence allocation, or subnational structure.

## 5.3 Overlay

Things that belong in overlays:

- CD4 severity
- optional future clinical severity overlays
- optional adherence-risk overlays
- optional regime / resistance overlays later if justified

Overlay means:

- attached to the main latent structure
- influences hazards or transitions
- does not become a full top-level primary axis immediately

---

## 6. Compute-Safe Implementation Guidance for This Machine

This section matters because the local machine is not an infinite cluster.

We explicitly checked:

- RAM: about `16 GB`
- GPU memory effectively visible through PyTorch: about `8 GB`
- JAX currently CPU-only on this machine

So the subgroup/state-space implementation must be conservative.

## 6.1 Do not build the full dense tensor eagerly

Do not try to materialize every possible combination of:

- province
- month
- KP
- age
- sex
- state
- duration
- CD4

as one giant dense tensor if much of it is empty or weakly identified.

Instead:

- keep some axes explicit,
- but prune or aggregate where necessary,
- and prefer structured low-rank parameterization over dense expansion.

## 6.2 Good implementation order

For this machine, the safe implementation order is:

1. main latent tensor:
   - province
   - KP
   - age
   - sex
   - state
   - duration
2. coarse CD4 overlay
3. sparse coupling terms
4. only later, if justified:
   - richer clinical overlays
   - more interaction structure

## 6.3 Use tensor math where it helps

Safe use of PyTorch tensors:

- transition construction
- batched state updates
- subgroup-axis batched operations
- low-rank factor application
- forward simulation

Keep outside the hot tensor loop:

- manifests
- registry logic
- provenance tables
- artifact summaries

## 6.4 Do not assume JAX `vmap` is the implementation baseline

Conceptually, `vmap`-like subgroup batching is a good future computational pattern.

But right now:

- JAX is CPU-only on this machine
- we have not migrated the probabilistic core to JAX

So `vmap` is a useful **future implementation analogy**, not the current required backend.

## 6.5 Heavy regularization and pruning

To prevent state explosion:

- use strong shrinkage
- keep subgroup interactions sparse
- keep cross-group coupling sparse
- cap CD4 bins
- cap duration granularity if needed

## 6.6 Respect the prior crash history

Because you reported an earlier destructive crash, the subgroup build path should follow these operational rules:

- stage outputs early
- checkpoint intermediate tensors
- never rely on one massive monolithic run
- cap heavy binary parsing separately from model tensor work
- use working-set budgets
- prefer resumable runs

---

## 7. Bottom Line

The correct subgroup topology from our conversation is:

- **Population subgroups**
  - explicit subgroup axis
  - not hard-coded into state labels
  - not fully independent
  - sparsely coupled

- **Age and sex**
  - explicit axes

- **Co-infections**
  - primarily feature/modifier layer in v1

- **CD4**
  - coarse overlay in v1

- **Cross-group dynamics**
  - sparse, structured, regularized

This gives the right balance between:

- scientific fidelity,
- subgroup interpretability,
- and compute safety on the current machine.

