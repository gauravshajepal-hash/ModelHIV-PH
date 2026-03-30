# Clean-Axes Replay And HARP Backtest Audit

Date: 2026-03-29
Run: `bounded-v2-replay-allprovinces-cleanaxes-20260329`
Baseline comparison run: `bounded-v2-replay-allprovinces-anchorpack2-20260329`

## What changed

This replay was executed after two upstream fixes:

1. Geography was split into:
   - `national`
   - `region`
   - `province`
   - `city`
   - `global`
   - `unknown`
2. Time normalization now rejects impossible observation years and prevents generic chart years from entering the aligned tensor.

The replay was intentionally memory-safe:

- reused an existing harvested raw corpus
- ran each phase in a separate process
- pinned BLAS/OpenMP thread counts to `1`
- forced CPU for Phase 3

## What improved

### Phase 0 / Phase 1 axis integrity

Compared with `bounded-v2-replay-allprovinces-anchorpack2-20260329`:

- province axis: `108 -> 103`
- month axis: `76 -> 64`
- duplicate national labels remained fixed
- region labels no longer appear in the province axis
- impossible years like `1913-01` and `2050-01` no longer appear

### National calibration

Mean error versus references improved:

- official ladder mean error: `0.204115 -> 0.157748`
- HARP mean error: `0.090132 -> 0.073516`

The run is still not calibrated enough for publication, but the clean-axis replay is materially better than the prior anchor-pack replay.

## Why regional outputs are still too similar

The cleaned replay proves that the earlier fake-flatness was not only an axis bug.

There is still a deeper modeling problem:

### 1. No truly promoted main predictive factors

`phase2/promoted_factor_set.json` is still empty.

That means the model is still being driven mostly by:

- shared observation anchors
- shared shrinkage structure
- supporting network factors

Without main predictive factors, there is not enough learned regional differentiation pressure.

### 2. Subgroup priors are still identical by province

`phase3/subgroup_weight_summary.json` shows the same subgroup distributions repeated across provinces.

This is a direct cause of regional homogenization.

If every province starts from the same:

- KP mix
- age mix
- sex mix

then the regional cascades will remain smooth unless strong subnational anchors or promoted factors counteract that.

### 3. ART and suppression regional MAE are still tightly clustered

The new replay no longer produces perfectly identical regional error lines, but the spread is still too small for many targets:

- `art_stock_mae` range: `0.035858`
- `documented_suppression_mae` range is larger only because a few regions break away from the pack

Many regions still sit in a narrow band around:

- diagnosed share: about `0.93`
- ART share: about `0.46`
- suppressed share: about `0.317`

This is not realistic enough.

### 4. The model still over-diagnoses nationally

In the clean replay HARP comparison:

- diagnosed stock model: `0.843177`
- diagnosed stock HARP: `0.622527`

That national optimism leaks downward into the regional surfaces.

## Regional variation after the clean replay

From `analysis/cleanaxes_comparison_report.json`:

- diagnosed share range: `0.072732`
- ART share range: `0.024089`
- suppressed share range: `0.016914`

So the regions are no longer numerically identical.
But the model is still too homogeneous on ART and suppression.

## HARP 2010-2025 frozen-history backtest status

A scientifically honest 2010-2025 HARP backtest is **not yet runnable** from the currently assembled corpus.

Current blockers:

1. The local PDFs only provide:
   - a 2024/2025 core-team care-cascade snapshot
   - surveillance and context material
   - not a clean annual HARP time series from 2010 onward
2. The old downloader in the earlier GitHub repo depends on a saved HTML page from the DOH/PNAC site to extract Google Drive surveillance links.
3. The current repo does not yet contain a verified historical HARP table spanning 2010-2025.

What is feasible now:

- assemble a dedicated HARP archive builder
- download and parse the monthly/quarterly surveillance PDFs from PNAC/DOH archives
- build a validated annual HARP panel
- then run a real frozen-history backtest:
  - train on early years
  - freeze
  - predict held-out later years

What is **not** scientifically acceptable now:

- pretending the current corpus already supports a full 2010-2025 predictive backtest

## Module-level verdict after the clean replay

### Phase 0

Improved significantly.

Strengths:

- local anchor PDFs are now being harvested and parsed
- impossible observation years are removed
- province axis is cleaner

Remaining weakness:

- still no formal Phase 0 truth package emitted on disk in the replay path

### Phase 1

Improved.

Strengths:

- split geography is respected
- normalized rows carry bias/reliability fields
- time axis is much cleaner

Remaining weakness:

- anchor-eligible rows dropped from `137` to `33`
- this may be the correct result after removing polluted anchors, but it also means the subnational anchor pack is still thin

### Phase 1.5

Architecturally real, scientifically underpowered.

The mesoscopic factor engine runs, but it is still mostly producing supporting context rather than factors that genuinely differentiate regions.

### Phase 2

Still too conservative.

No main predictive factors were promoted in this replay.
That is safe, but it means the rescue-v2 middle layer is not yet delivering real predictive leverage.

### Phase 3

Still the strongest module overall, but still not publication-ready.

Strengths:

- observation-first
- HARP/official reference checks are explicit
- cleaned replay improved national fit

Remaining weaknesses:

- diagnosed optimism still high
- hierarchy reconciliation still weak
- regional outputs still too homogeneous

### Phase 4

Correctly blocked.

No policy layer should be trusted until the subnational scientific core is stronger.

## Immediate next steps

1. Build a real historical HARP archive and parser for 2010-2025.
2. Replace province-constant subgroup priors with province- and region-specific priors from anchorable sources.
3. Force Phase 2 to either:
   - promote at least a small, stable predictive factor set
   - or explicitly report that v2 is still only a calibration shell
4. Add region-level anchor packs so ART and suppression are not learned mostly from national pressure.
