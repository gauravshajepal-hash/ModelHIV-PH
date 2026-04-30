# Phase 2 Seeded Champions: Validity Skeptic Audit

**Date:** 2026-04-18

## Scope

This memo attacks two candidate directions:

1. the current bounded seeded-scenario layer in [tr_v3_phase2_seeded_champion_batch.py](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_phase2_seeded_champion_batch.py)
2. the proposed next-step alternative of adding a stricter regime/sojourn layer on top of the same Phase 2 monthly structure

The standard is validity, not creativity. The question is what is too weak to trust, what must be falsified first, and which path is less indefensible.

## Executive Judgment

The current seeded-scenario layer is only defensible as a **bounded structural sandbox**. It is not yet defensible as a scientific finding about future structural shocks, plateau dynamics, or mechanistic propagation.

The stricter regime/sojourn idea is currently **less defensible** than the bounded seeded layer. It adds hidden-state complexity before the current structural seed has cleared basic falsification gates.

If forced to choose between:

- expanding scenario families on the same bounded kernel
- adding a regime/sojourn layer now

the less-wrong choice is:

- **same-kernel scenario work only after a falsification batch**

The correct immediate move is not expansion. It is a narrow audit package that attacks circularity, archive mismatch, effective sample size, and placebo sensitivity.

## Direct Evidence

| Evidence | Observation | Validity read |
|---|---|---|
| [tr_v3_phase2_seeded_champion_batch.py](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_phase2_seeded_champion_batch.py#L215) | Scenario shocks are hand-coded as fixed magnitude templates on a small set of blocks. | This is scenario design, not learned shock structure. |
| [tr_v3_phase2_seeded_champion_batch.py](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_phase2_seeded_champion_batch.py#L275) | Residual readout is fit on quarterly holdout rows using Phase 2 quarter features. | This is a retrospective residual map, not independent forward evidence. |
| [tr_v3_phase2_seeded_champion_batch.py](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_phase2_seeded_champion_batch.py#L304) | Exact-lane fit gate is only `max(8, feature_dim + 2)` rows. With 4 blocks plus deltas, that means 10 rows minimum. | Exact-lane readout is operating in a tiny-sample regime. |
| [tr_v3_phase2_seeded_champion_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/tr_v3_phase2_seeded_champion_batch_report.md#L26) | Exact residual counts are `12-13` per metric. | Effective sample size is too small for a stable 8-feature readout. |
| [tr_v3_phase2_seeded_champion_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-phase2-seeded-champions-20260418-s00/analysis/tr_v3_phase2_seeded_champion_batch_report.md#L4) | The seeded batch used archive run `harp-archive-wdi-standard-20260412-s19`, not the newer merged HIV-anchor archive. | Base forecasts and structural seed are not aligned to the latest evidence geometry. |
| [plugins/hiv.py](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/plugins/hiv.py#L1316) | `testing_engagement` directly includes `diagnosed_plhiv` and `new_diagnosed_cases_period`. | The structural seed contains the same outputs the seeded layer later perturbs. |
| [plugins/hiv.py](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/plugins/hiv.py#L1347) | `care_access_continuity` directly includes `alive_on_art`. | The structural seed contains another target the scenario layer later perturbs. |
| [tr_v3_monthly_edge_audit_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-edge-audit-20260418-s01/analysis/tr_v3_monthly_edge_audit_batch_report.md#L27) | `care_access_continuity -> suppression_capacity` drops while `testing_engagement -> suppression_capacity` appears. | The edge shift is real enough for follow-up, but still fragile and not yet interpretable as causal discovery. |

## Contextual Evidence

| Concern | Current status |
|---|---|
| Monthly axis defect | fixed |
| HARP enrichment | improved |
| Structural graph richness | still very thin: 2 direct edges, 0 hidden rows |
| Hidden-state support | absent |
| Independent shock labels | absent |
| Exogenous lead indicators | absent |
| Structural-feature independence from target heads | absent |
| Prospective claim support | absent |

## Top Validity Threats

### 1. Circularity between Phase 2 blocks and Phase 3 outcomes

This is the biggest problem.

The seeded layer uses Phase 2 blocks as structural causes, but those blocks already directly contain:

- `diagnosed_plhiv`
- `new_diagnosed_cases_period`
- `alive_on_art`

That means the model can narrate:

- testing state changes diagnosed stock
- care continuity changes ART stock

when in fact the structural state already includes those same heads as indicators.

This is not fatal for a descriptive latent model. It is fatal for any claim that the seeded layer has discovered an independent structural mechanism.

### 2. Archive mismatch between base forecast and structural seed

The seeded batch is driven by:

- old archive run for the champion forecast/evaluation
- newer monthly structural run for Phase 2

That breaks support parity. It means the seeded layer is not being applied to a forecast base built from the same evidence revision that produced the structural seed.

At minimum, this can change:

- annual anchors
- observation weights
- care-cascade support
- induced structural edge geometry

### 3. Effective sample size is too small, especially on exact

The readout fit uses:

- 8 features: 4 block levels plus 4 block deltas
- only `12-13` exact residual rows per metric

That is barely above the minimum gate and is not enough to trust coefficient signs or scenario ranking.

Worse, the rows come from rolling-origin splits. The same quarter-level features can reappear in multiple split contexts, so nominal row count overstates independent information.

### 4. Hand-coded shocks are not identified

The current scenarios are policy templates, not discovered shock modes.

Examples in the code:

- `testing_pulse`
- `testing_plateau`
- `disruption_recovery`
- `mobility_spike`

These are acceptable scenario names for a sandbox. They are not acceptable evidence for future-shock structure.

### 5. Missingness and block reinterpretation can create fake edges

The monthly lane improved materially, but the graph is still thin. The appearance of a new `testing -> suppression` edge after HARP enrichment could mean:

- a real structural relationship
- changed indicator composition
- changed weighting of annual anchors
- block relabeling through direct indicator mix

Without ablations, those are not separable.

### 6. Fake novelty risk is high

The current system risks rediscovering an obvious cascade:

- testing
- care
- suppression

because the HARP block design and priors already encode that cascade semantically.

If the strongest structural result is “testing-centered cascade appears,” the burden is to prove it survives:

- indicator ablation
- lag-prior ablation
- archive alignment
- placebo shocks

Otherwise this is architecture-induced patterning, not discovery.

## What Must Be Falsified First

### Falsification Gate 1: Outcome-derived indicator ablation

Rebuild the monthly Phase 2 lane with these removed from latent-block indicators:

- `diagnosed_plhiv`
- `new_diagnosed_cases_period`
- `alive_on_art`

Keep only non-target or less-direct heads where possible.

If the testing-centered structure collapses, the current seeded story was partly circular.

### Falsification Gate 2: Archive alignment rerun

Rerun the seeded batch on the same merged HARP-enriched archive used to build the corrected monthly lane.

If scenario responses or fitted readout signs change materially, the current seeded output is not stable enough to interpret.

### Falsification Gate 3: Unique-quarter residual fit

Refit the residual readout using one row per quarter per metric, not repeated split rows.

Then bootstrap the coefficient signs and scenario terminal deltas.

If signs are unstable, the current readout should be treated as illustrative only.

### Falsification Gate 4: Placebo and orthogonalized shocks

Test:

- random block perturbations with matched norm
- shuffled-quarter shock patterns
- perturbations on blocks that should have weak readout

If the model produces similar endpoint deltas under placebo shocks, the scenario semantics are weak.

### Falsification Gate 5: Indicator-to-block contribution audit

For each retained block, report:

- direct indicator composition
- support counts
- loadings
- how much each HARP head contributes

This is required before any scientific language about structural states.

## What To Avoid Next

- Do not add a regime/sojourn layer now.
- Do not call the current scenarios “future shock prediction.”
- Do not publish the seeded layer as mechanistic novelty.
- Do not add more hand-designed scenario families before the ablations above.
- Do not evaluate this with point-forecast MAE as if it were competing with the champions.
- Do not treat the current `testing -> suppression` edge as a discovered causal pathway.
- Do not use a large HSMM on the current single national sequence.

## Recommendation: Same-Kernel Expansion vs Regime/Sojourn

### Recommendation

If one path must be kept alive now, keep:

- **same-kernel bounded scenario work**

and reject for now:

- **regime/sojourn layer**

### Why same-kernel is less wrong

- It is honest about being a scenario layer.
- It can remain bounded.
- It does not require hidden-state identifiability claims.
- It can be stress-tested with ablations and placebo shocks.

### Why regime/sojourn is less defensible now

- The structural seed itself has not cleared circularity checks.
- There is only one national sequence.
- Hidden rows are still zero.
- No independent regime labels exist.
- Plateau duration is not estimated from clean event structure.
- Any dwell-time model now would mostly formalize unsupported semantics.

## Proper Plan

### Phase A: Validity Gates

1. `EXP-P2-SEED-GATE-01`
   Archive-aligned seeded rerun.

2. `EXP-P2-SEED-GATE-02`
   Outcome-derived indicator ablation in Phase 2 block definitions.

3. `EXP-P2-SEED-GATE-03`
   Unique-quarter residual readout plus coefficient stability audit.

4. `EXP-P2-SEED-GATE-04`
   Placebo shock sensitivity.

5. `EXP-P2-SEED-GATE-05`
   Indicator contribution and support audit by block.

### Phase B: Conditional Follow-up

Only if Phase A survives:

6. `EXP-P2-SEED-SCEN-01`
   Expand same-kernel scenario families.

7. `EXP-P2-SEED-SCEN-02`
   Add scenario envelopes and sensitivity plots over bounded shock magnitudes.

### Phase C: Only If Stronger Evidence Appears

Only if:

- block composition is no longer circular
- unique-quarter readout is stable
- placebo shocks separate clearly from targeted shocks
- scenario semantics remain stable under archive-aligned reruns

then consider:

8. `EXP-P2-SEED-RS-01`
   Tiny sticky regime layer with explicit justification for dwell time.

Not before.

## AutoResearch Handoff

- Variant: `evidence-to-model-loop`
- Evaluation harness:
  - archive-aligned seeded rerun
  - unique-quarter residual fit
  - placebo shock audit
  - block indicator ablation
  - coefficient/sign stability
- Mutation units:
  - archive-run alignment
  - outcome-indicator removal from block definitions
  - unique-quarter aggregation
  - placebo shock generator
  - block contribution reporter
- Stop rule:
  - keep only if the seeded interpretation remains stable after the falsification batch
  - otherwise demote the seeded layer to an exploratory sandbox and stop there
