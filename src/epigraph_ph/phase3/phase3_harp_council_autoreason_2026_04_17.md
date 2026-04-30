# HARP Monthly Lane Council + Autoreason

- Date: `2026-04-17`
- Scope: monthly HARP-enriched Phase 1 / Phase 15 / Phase 2 lane after rebuild `tr-v3-monthly-phase2-lane-20260416-s01`
- Council mode: same-model council
- Autoreason arbitration: `A / B / AB`
- Winning plan: `AB`

## Chairman Synthesis

HARP materially improved the monthly national lane:

- contiguous month axis: `192` months from `2010-01` to `2025-12`
- monthly national rows: `637`
- injected HARP Phase 1 rows: `562`
- retained Phase 15 blocks: `4`
- retained Phase 2 direct edges: `2`

The strongest surviving structure is a small lag-1 program cascade:

- `testing_engagement -> care_access_continuity @ lag1`
- `care_access_continuity -> suppression_capacity @ lag1`

What does **not** survive scrutiny yet:

- a larger HSMM or large hidden-state sweep
- future-shock emulation off the current monthly structural output
- any claim that the monthly cascade is fully emergent or causal

The main reason is validity risk:

- HARP heads were injected directly into Phase 1
- those heads were then mapped into HIV latent blocks with cascade-shaped priors
- the larger lag sweep has an unresolved scoring inconsistency
- hidden-driver rows remain `0`

So the next phase should be gated:

1. hardening and falsification first
2. only then a narrow monthly modeling phase

## Strongest Hypotheses

1. HARP successfully repaired the monthly lane enough to reveal a real, but modest, lag-1 program cascade.
2. The visible cascade may be partly induced by block design and lag priors rather than discovered purely from the data.
3. The national monthly lane is still too thin for a larger hidden-state model, but it may support a tiny sticky regime overlay after falsification gates pass.

## Evidence For And Against

### H1: HARP repaired the lane enough to reveal usable structure

Direct evidence for:

- [tr_v3_monthly_phase2_lane_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260416-s01/analysis/tr_v3_monthly_phase2_lane_batch_report.md): `192` contiguous months, `637` rows, `562` HARP rows
- [phase15_v2_fit_summary.json](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260416-s01/phase15/phase15_v2_fit_summary.json): `4` retained blocks
- [phase2_structural_payload.json](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260416-s01/phase2/phase2_structural_payload.json): 2 retained lag-1 direct edges

Evidence against overclaiming:

- same payload has `0` hidden temporal edges and `0` hidden driver rows
- [phase2_larger_temporal_sweep.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260416-s01/analysis/phase2_larger_temporal_sweep.md): baseline and extended search still choose `lag=1`

### H2: The cascade may be partly engineered

Direct evidence for:

- [tr_v3_monthly_phase2_lane_batch.py](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_monthly_phase2_lane_batch.py): direct HARP injection into Phase 1 rows
- [hiv.py](/media/gaurav/New_Volume/EpiGraph_PH/src/epigraph_ph/plugins/hiv.py): expanded latent block assignments and cascade-shaped lag priors

Evidence against total dismissal:

- the lane was thin before the HARP rebuild and is materially richer now
- the same small cascade survives the unconstrained searches, not just forced lag settings

### H3: A larger hidden-state model is not justified yet

Direct evidence for:

- only `4` retained blocks
- only `2` direct edges
- `0` hidden rows
- national-only monthly lane

Evidence that keeps the door open later:

- the monthly axis defect is fixed
- HARP support is now strong enough to justify a stricter multi-sequence monthly build

## Checks The Council Agrees Must Happen First

1. Artifact consistency parity between markdown, JSON summaries, and plots.
2. Lag-selection consistency under one pre-registered scoring rule.
3. Block-support sufficiency audit by retained Phase 15 block.
4. Ablation falsification:
   - remove new HARP-to-block assignments
   - flatten/remove hand-authored cascade lag priors
5. Blocked monthly stability audit for retained direct edges.

## Interventions Worth Trying

1. `EXP-HARP-GATE-01`
   Artifact consistency and support audit.

2. `EXP-HARP-GATE-02`
   Lag-selection consistency and blocked stability audit.

3. `EXP-HARP-GATE-03`
   HARP block-mapping plus lag-prior ablation falsification.

4. `EXP-HARP-GATE-04`
   Quarterly summary extraction audit from monthly structural outputs.

5. Conditional on the gates passing:
   - `P2-MON-R1`: split-stable lag graph freeze
   - `P2-MON-R2`: tiny sticky monthly regime overlay with `K=2..3`
   - `P2-MON-R3`: bounded quarterly champion sidecar from monthly summaries

## Interventions To Avoid

- large HSMM sweep now
- future-shock emulation off the current monthly structure
- forcing long-lag discovery claims
- feeding raw monthly edge counts into Phase 3 champions
- treating hidden-row absence as a bug rather than weak support

## Direct Evidence Table

| Artifact | Evidence | Implication |
|---|---|---|
| [tr_v3_monthly_phase2_lane_batch_report.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260416-s01/analysis/tr_v3_monthly_phase2_lane_batch_report.md) | `192` months, `637` rows, `562` HARP rows, `4` blocks, `2` edges, `0` hidden rows | HARP fixed lane geometry and yielded a small direct structure |
| [phase1_monthly_rebuild_summary.json](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260416-s01/phase1/phase1_monthly_rebuild_summary.json) | `21` canonicals | report-layer canonical count bug should not be treated as scientific evidence |
| [phase15_v2_fit_summary.json](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260416-s01/phase15/phase15_v2_fit_summary.json) | `4` retained blocks, `testing_engagement` retained | measurement model is viable but still compact |
| [phase2_structural_payload.json](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260416-s01/phase2/phase2_structural_payload.json) | lag-1 cascade only, no hidden rows | no support yet for larger hidden-state structure |
| [phase2_larger_temporal_sweep.md](/media/gaurav/New_Volume/EpiGraph_PH/artifacts/runs/tr-v3-monthly-phase2-lane-20260416-s01/analysis/phase2_larger_temporal_sweep.md) | baseline and extended searches pick `lag=1`; forced `lag=6` reports lower loss | scoring / selection path must be reconciled before interpretation |

## Contextual Evidence Table

| Context | Relevance |
|---|---|
| Monthly axis is now contiguous | fixes the prior construction defect |
| HARP heads are mainly program stocks and flows | can induce cascade-shaped lag structure even without richer hidden dynamics |
| National-only monthly lane | too weak for aggressive state-model claims |
| Hidden rows remain zero | current structure is direct and sparse, not hidden and rich |

## Autoreason Arbitration

### A

Benchmark hardening first. No new model phase until all evaluation and validity gates pass.

### B

Proceed directly to a small monthly modeling phase: split-stable lag graph refinement plus a tiny sticky regime overlay.

### AB

Run hardening and falsification first. If the gates pass, proceed immediately to the narrow monthly modeling phase while deferring large HSMM and future-shock work.

### Winner

`AB`

Reason:

- `A` alone is too conservative because a narrow modeling step is plausible if the lane survives falsification.
- `B` alone is not defensible because the current evidence still has unresolved validity issues.
- `AB` preserves rigor without stalling progress.

## AutoResearch Handoff

- Variant: `evidence-to-model-loop`
- Evaluation harness:
  - stage 1 hardening gates on the rebuilt monthly national lane
  - artifact parity across markdown / JSON / plots
  - lag-selection consistency under a pre-registered score
  - block-support sufficiency by retained Phase 15 block
  - ablation falsification removing new HARP-to-block assignments and flattening hand-authored cascade lag priors
  - blocked monthly folds:
    - train `2010-2016`, validate `2017-2019`, test `2020-2021`
    - train `2010-2018`, validate `2019-2021`, test `2022-2023`
    - train `2010-2020`, validate `2021-2023`, test `2024-2025`
  - promotion criteria:
    - artifact parity holds
    - selected lag is reproducible
    - direct edges remain stable across blocked folds
    - ablated cascade still survives materially
    - hidden rows remain optional, not required
- Mutation units:
  - report / JSON consistency fixes
  - lag-selection scoring reconciliation
  - per-block monthly support audit
  - HARP block-mapping ablation
  - lag-prior ablation
  - blocked edge-stability audit
  - tiny sticky regime overlay with `K=2..3`
  - quarterly structural summary extraction
- First experiments:
  1. `EXP-HARP-GATE-01`
  2. `EXP-HARP-GATE-02`
  3. `EXP-HARP-GATE-03`
  4. `EXP-HARP-GATE-04`
  5. conditional: `P2-MON-R1`
  6. conditional: `P2-MON-R2`
  7. conditional: `P2-MON-R3`
