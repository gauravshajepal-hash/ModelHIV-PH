# Codex Memory Context: Syndemic Coupling

This document is a **conversation-memory-based syndemic-coupling specification**, not a fresh repository audit.

Its purpose is to describe how multiple disease engines can coexist, exchange limited information, and be compared or coupled without collapsing into one giant joint state space.

The core principle is:

> Keep disease engines separate by default.
> Exchange only low-dimensional summaries where coupling is scientifically justified.
> Start with cheap, robust coupling diagnostics before adopting heavier nonlinear or information-theoretic coupling machinery.

---

## 1. Separate Disease Engines

The memory-consistent design is a federation pattern:

- one disease plugin per disease
- one fitted engine per disease
- one main latent tensor per disease
- one benchmark pack per disease
- one policy pack per disease, unless a joint operational layer is explicitly defined

So for diseases \(z_1, z_2, \dots, z_m\), the system is better written as:

\[
\mathcal{M}^{(z_1)}, \mathcal{M}^{(z_2)}, \dots, \mathcal{M}^{(z_m)}
\]

not as one monolithic merged mega-model.

This avoids:

- state-space explosion
- mixed observation semantics
- impossible transition definitions
- unmanageable tensor sizes on constrained hardware

Examples:

- HIV remains a care-continuum model
- TB may remain a latent/active/treatment model
- dengue may remain an epidemic/vector model

The engines are allowed to differ in:

- state topology
- time resolution
- subgroup axes
- intervention semantics
- benchmark targets

---

## 2. Exchanged Exogenous Summaries

Cross-disease interaction should occur through **summaries**, not through a full tensor merge.

Let disease \(a\) export a summary vector:

\[
E^{(a \rightarrow b)}_t \in \mathbb{R}^{q_{ab}}
\]

which is consumed by disease \(b\) as an exogenous feature block.

Examples of such summaries:

- hospital capacity pressure
- diagnostics capacity
- workforce pressure
- drug supply pressure
- transport disruption
- clinic reach
- HIV ART coverage as a modifier for TB outcomes
- TB burden as a modifier for HIV attrition or severity
- shared environmental burden indicators

Then disease \(b\) consumes:

\[
Z^{(b)}_t = \left[ Z^{(b)}_{local,t}, E^{(a \rightarrow b)}_t \right]
\]

where \(Z^{(b)}_{local,t}\) is the disease-local feature set.

This is the memory-consistent coupling pattern:

- separate disease engines
- shared low-dimensional exogenous summaries
- no forced disease-state tensor union

---

## 3. Coupling Diagnostics: Cheap to Expensive

We did **not** commit to one universal cross-disease coupling metric.

The safe design is a ladder of diagnostics from cheap and robust to expensive and assumption-heavy.

This diagnostic ladder should be interpreted together with a separate **coupling-evidence ladder**:

- weak evidence for exploration
- medium evidence for structured scientific retention
- strong evidence for cautious scientific or operational claims

### 3.1 Cheap diagnostics

These should come first.

#### Shared-driver overlap

Compare whether two disease engines depend on the same broad blocks:

- economics
- logistics
- climate
- mobility
- health-system reach
- stigma / behavior
- diagnostics capacity

This can be measured using block-level overlap scores or ranked-driver overlap.

#### Lagged predictive gain

Ask whether adding a disease-\(a\) summary improves forecasting for disease \(b\):

\[
\Delta \text{score}_{a \rightarrow b}
=
\text{score}\!\left(\mathcal{M}^{(b)} \mid E^{(a \rightarrow b)}\right)
-
\text{score}\!\left(\mathcal{M}^{(b)} \mid \varnothing\right)
\]

This is often more useful operationally than exotic graph metrics.

#### Ablation benefit

Run paired ablations:

- without cross-disease summary
- with cross-disease summary

and compare:

- forecast improvement
- calibration improvement
- regional/province improvement

This is one of the safest practical syndemic diagnostics.

### 3.2 Medium-cost diagnostics

These are informative, but should come only after the cheap layer is stable.

#### Shared-feature or blanket-style overlap

If both disease engines expose comparable driver sets, measure overlap of their promoted drivers:

\[
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
\]

where \(A\) and \(B\) are comparable driver sets or block sets.

This is a heuristic structural comparison, not proof of shared biology.

#### Cross-model sensitivity comparison

Estimate how strongly each disease responds to the same exogenous summary family:

\[
\nabla^{cf}_{u, z}
\]

for disease \(z\) and intervention or driver \(u\).

This can reveal shared operational leverage points without merging disease states.

### 3.3 Expensive diagnostics

These are optional research tools, not baseline requirements.

#### CCM

Convergent Cross Mapping may be useful as an exploratory nonlinear coupling diagnostic when:

- time series are long enough
- measurement quality is acceptable
- common forcing is handled carefully

But it is not a mandatory baseline tool.

#### Transfer entropy

Transfer entropy may be used as an optional dynamic coupling diagnostic:

\[
T_{a \rightarrow b}
\]

when we want to estimate information flow from disease \(a\) to disease \(b\).

But it is estimation-heavy, sensitive to noise, and not a universal default.

---

## 4. What Is Safe on This Machine

The current machine constraints from conversation memory are roughly:

- about 16 GB RAM
- about 8 GB effective GPU VRAM available through PyTorch
- JAX currently CPU-only

So the safe syndemic strategy on this machine is:

- separate disease engines
- low-dimensional exchanged summaries
- cheap or medium-cost diagnostics first
- bounded working sets
- resumable stage outputs
- no giant cross-disease tensor union

Safe now:

- driver-block overlap
- lagged predictive gain
- ablation benefit
- small cross-disease sensitivity comparisons
- summary-feature exchange between disease engines

Not safe as a default now:

- giant joint HIV-TB-dengue-malaria latent tensor
- heavy JAX-first coupled simulation on GPU
- large transfer-entropy grids over many provinces and long horizons
- expensive nonlinear coupling scans over every disease pair by default

---

## 5. What Is Future HPC Work

Future HPC-only or later-stage work may include:

- large cross-disease rollout ensembles
- dense pairwise coupling experiments across many diseases
- high-resolution CCM or transfer-entropy scans
- joint optimization of disease portfolios under shared resource constraints
- distributed simulation or control layers if Phase 4 matures enough

These belong to the future acceleration roadmap, not the immediate baseline.

---

## 6. Final Syndemic Principle

The final memory-consistent rule is:

> Syndemics should be modeled as federated disease engines with explicit coupling summaries and progressively more expensive diagnostics.

That is more robust than:

- forcing everything into one tensor
- assuming one graph explains all diseases
- or jumping straight to expensive nonlinear coupling machinery before the basic coupled ablations are working.
