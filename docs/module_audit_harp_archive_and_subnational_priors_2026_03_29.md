## Scope

This audit covers the implementation pass that added:

- a dedicated `harp_archive` build path
- gap-aware historical HARP/Spectrum archive assembly
- province/region-specific subgroup priors in Phase 3
- explicit Phase 2 promotion admission when no main predictive factor is justified
- bounded replay on:
  - `bounded-v2-replay-allprovinces-harparchive-20260329`

Key comparison artifact:

- [harp_archive_upgrade_report.json](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-harparchive-20260329/analysis/harp_archive_upgrade_report.json)

## Module Findings

### `harp_archive`

Status: implemented and working, but not yet sufficient for a real frozen-history backtest.

What is now correct:
- local official PDFs are parsed into a structured archive
- the archive distinguishes:
  - Spectrum/model estimates
  - HARP/program-observed counts
- the builder emits:
  - `historical_harp_panel.json`
  - `historical_harp_panel.csv`
  - `subgroup_anchor_pack.json`
  - `backtest_assessment.json`

Current scientific limit:
- the archive is gap-aware, not complete
- `backtest_ready` is still `false`
- `observed_program_year_count = 1`

This is the correct outcome from the currently available local PDFs. The code is not hallucinating a 2010–2025 HARP panel that does not exist in the assembled source set.

Primary artifact:
- [backtest_assessment.json](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-harparchive-20260329/harp_archive/backtest_assessment.json)

### `phase15`

Status: improved materially.

What is now correct:
- factor stability now includes:
  - `subnational_anomaly_gain`
  - `region_contrast_score`
- Phase 1.5 now writes `network_feature_catalog.json`
- this unblocks Phase 3 from actually consuming network-derived prior signals

What remains weak:
- on the bounded replay, Phase 1.5 still does not produce a clearly dominant main predictive factor set
- many useful signals are still arriving as network/support features rather than robust mesoscopic block champions

### `phase2`

Status: behavior is now honest.

What is now correct:
- Phase 2 no longer silently returns an empty promotion set
- it now emits:
  - `promotion_admission.json`
- if no main predictive factor clears the gates, it says so explicitly and lists near misses

Current bounded replay result:
- `status = "none_admitted"`
- strongest near misses are network/service fragility features

Primary artifact:
- [promotion_admission.json](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-harparchive-20260329/phase2/promotion_admission.json)

Interpretation:
- this is a scientifically safer result than fake promotion
- but it also means the current subnational evidence stack is still not strong enough to justify calling those factors main predictive

### `phase3`

Status: subgroup priors are no longer flat, but calibration is still not good enough.

What is now correct:
- subgroup priors are no longer province-constant
- priors now combine:
  - national counts
  - region shrinkage
  - subgroup anchor pack
  - network-derived prior signals

Current bounded replay evidence:
- `anchor_pack_present = true`
- `network_signal_count = 103`
- provinces now have visibly different KP priors

Primary artifact:
- [subgroup_weight_summary.json](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-harparchive-20260329/phase3/subgroup_weight_summary.json)

What remains wrong:
- the model still over-diagnoses badly against HARP
- HARP comparison at `2025-01` still shows:
  - diagnosed stock error about `0.223`
  - ART stock error about `0.098`
  - second 95 error about `0.061`

Primary artifact:
- [reference_check_harp.json](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-harparchive-20260329/phase3/reference_check_harp.json)

### `phase0` / `phase1`

Status: unchanged in this pass, but still the main scientific bottleneck.

The new archive/prior work improves downstream honesty, but it does not solve the upstream fact that:
- subnational anchor density is still thin
- the model still leans too hard on national anchors plus shrinkage

That is why:
- subgroup priors are now more realistic
- but official and HARP calibration remain weak

## Bounded Replay Outcome

Run:
- [bounded-v2-replay-allprovinces-harparchive-20260329](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-harparchive-20260329)

Memory discipline used:
- copied existing `phase0` and `phase1` instead of reharvesting
- separate processes for:
  - `harp_archive`
  - `phase15`
  - `phase2`
  - `phase3`
  - `phase4`
- single-threaded BLAS/OpenMP
- CPU-forced Phase 3

Behavioral summary:
- archive builder works
- subgroup priors vary by province/region
- network signals are now actually consumed
- Phase 2 explicitly admits no main predictive factor
- Phase 4 remains correctly blocked

## Comparison Against Prior Cleaned Run

Old run:
- [bounded-v2-replay-allprovinces-cleanaxes-20260329](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-cleanaxes-20260329)

New comparison package:
- [harp_archive_upgrade_report.json](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-harparchive-20260329/analysis/harp_archive_upgrade_report.json)

Net result:
- subgroup priors improved structurally
- archive support exists
- promotion logic is more honest
- but official and HARP fit did not improve enough to claim a scientific win

This is important:
- the new pass made the pipeline more correct
- it did not make the epidemic fit substantially better

## Most Important Remaining Problems

1. Historical HARP program counts are still incomplete.

Without a real 2010–2025 program panel, a true frozen-history backtest is still blocked.

2. Phase 2 still cannot justify a main predictive subnational factor set.

That means the model is still mostly:
- national anchors
- shrinkage
- support features

3. Diagnosis remains too optimistic.

The biggest remaining failure is still diagnosis inflation relative to HARP.

4. Network-informed priors help structure, but they do not substitute for real subnational observed anchors.

## Verdict

This pass was necessary and successful at the implementation level.

It fixed real architecture gaps:
- no archive path
- flat subgroup priors
- silent empty promotion
- missing network feature catalog

But the current system is still not publication-ready as a predictive HIV model because:
- the historical HARP archive is incomplete
- the main predictive factor layer is still not scientifically justified
- the bounded replay remains materially miscalibrated on diagnosis and treatment linkage

The next serious work item is not another architectural layer. It is evidence assembly:

1. expand the historical HARP/PNAC archive
2. assemble region/province anchor panels
3. rerun the frozen-history backtest once the archive is actually real
