# Codex Memory Context: Syndemic Coupling Evidence Ladder

This document is a **conversation-memory-based evidence ladder** for cross-disease or syndemic coupling claims.

It is intended to prevent a common failure mode:

- seeing one interesting shared driver or nonlinear diagnostic,
- then over-claiming that two disease systems are mechanistically coupled.

The core rule is:

> Coupling evidence should be graded.
> Cheap diagnostics can motivate exploration.
> Strong scientific claims require stronger evidence than exploratory structure alone.

---

## 1. Evidence Classes

We divide syndemic coupling evidence into three broad levels:

- **weak evidence**
- **medium evidence**
- **strong evidence**

These levels are about claim strength, not mathematical elegance.

---

## 2. Weak Coupling Evidence

Weak evidence is useful for hypothesis generation, discovery, and prioritization.

It is **not** enough for strong scientific or policy claims by itself.

### 2.1 Typical weak evidence

- shared broad driver blocks
  - both diseases appear sensitive to climate, mobility, poverty, or health-system reach
- ranked-feature overlap
- simple correlation
- broad temporal co-movement
- shared ontology tags from Phase 0
- literature overlap without disease-specific validation
- exploratory graph overlap without outcome improvement

### 2.2 Interpretation

Weak evidence supports statements like:

- “these diseases may share upstream determinants”
- “this coupling is worth testing further”
- “this shared driver family should enter ablation”

Weak evidence does **not** support statements like:

- “the diseases are mechanistically coupled”
- “one policy will reliably control both diseases”
- “the shared driver is proven causal”

---

## 3. Medium Coupling Evidence

Medium evidence is suitable for stronger modeling decisions, but still should be described carefully.

### 3.1 Typical medium evidence

- lagged predictive gain when one disease summary is added to another disease model
- ablation benefit from cross-disease summaries
- improved calibration or forecast accuracy after adding exchanged summaries
- stable shared-driver overlap at the block or promoted-feature level
- consistent cross-model sensitivity comparisons
- replicated coupling signal across regions or time windows

### 3.2 Interpretation

Medium evidence supports statements like:

- “adding cross-disease summaries improves predictive performance”
- “the diseases appear operationally coupled through shared drivers”
- “the evidence supports joint monitoring or joint scenario analysis”

Medium evidence still does **not** fully support:

- strong mechanistic causal claims
- definitive biological interpretation
- automatic joint policy deployment

---

## 4. Strong Coupling Evidence

Strong evidence is what we would need before making robust cross-disease scientific claims or strong joint-policy recommendations.

### 4.1 Typical strong evidence

- repeated ablation gains across multiple holdouts
- consistent predictive gain across regions and time windows
- stable coupling under perturbation and sensitivity checks
- externally plausible mechanism supported by domain knowledge
- agreement between multiple evidence types:
  - predictive gain
  - ablation benefit
  - comparable-driver overlap
  - counterfactual sensitivity comparison
- optional heavier diagnostics such as CCM or transfer entropy that remain stable after confounding and seasonality checks

### 4.2 Interpretation

Strong evidence supports statements like:

- “these disease systems are strongly coupled through these identified pathways”
- “joint monitoring or selective joint intervention is scientifically defensible”
- “the cross-disease signal is robust enough for controlled operational use”

Even strong evidence should still be narrower than:

- “we have proven the complete biology”
- “the policy is guaranteed to be optimal”

---

## 5. What Is Acceptable for Scientific Claims

For scientific claims, the minimum acceptable evidence should usually be **medium to strong**, depending on the claim.

### 5.1 Acceptable lower-strength scientific claims

Acceptable with medium evidence:

- “disease A provides useful predictive information for disease B”
- “shared-driver blocks improve model fit or forecast skill”
- “cross-disease summaries should be retained in the scientific candidate set”

### 5.2 Acceptable stronger scientific claims

Require strong evidence:

- “disease A and disease B are materially coupled in this setting”
- “joint intervention on this shared driver is scientifically justified for operational testing”
- “shared leverage points are robust across validation settings”

---

## 6. What Is Only Exploratory Research Evidence

The following should usually remain exploratory unless backed by stronger downstream evidence:

- pure graph-overlap heuristics
- Jaccard overlap of feature or blanket sets by itself
- simple contemporaneous correlation
- unvalidated nonlinear diagnostics
- one-off CCM signal
- one-off transfer entropy estimate
- literature-only overlap
- intuition that a shared biology “must” exist

These are useful for:

- prioritizing investigations
- designing ablations
- motivating data collection
- hypothesis generation

They are not enough for strong scientific or policy claims on their own.

---

## 7. Practical Rule for This Framework

The memory-consistent workflow is:

1. detect weak signals broadly
2. test medium-strength predictive and ablation evidence
3. reserve strong claims for repeated, stable, multi-view evidence

This matches the broader philosophy of the pipeline:

- wide discovery
- strict promotion
- mechanistic validation before operational claims

---

## 8. Final Coupling-Evidence Rule

The final rule is:

> Weak evidence is for exploration.
> Medium evidence is for scientific retention and structured testing.
> Strong evidence is for cautious scientific claims and tightly controlled operational use.
