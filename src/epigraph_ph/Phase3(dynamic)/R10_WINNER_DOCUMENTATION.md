# R10 Repair Experiment — Key Finding

**Date:** 2026-04-12
**Run:** `tr-v3-repair-search-20260412-s05`
**Champion:** `SEARCH-R10E-a100-f25-delta-sup75`

---

## Result Summary

| Contract | Candidate MAE | Baseline MAE | Improvement |
|----------|--------------:|------------:|------------|
| **exact_only** | **0.072** | 0.228 | **3.2x better** |
| dense_train_observed_score | 0.126 | 0.136 | 1.1x better |

## Contract Champions

| Contract | Experiment | MAE |
|----------|-----------|-----:|
| `exact_only` | `SEARCH-R10E-a100-f25-delta-sup75` | **0.072** |
| `dense` | `SEARCH-R10D-artdelta-ab100-f100` | **0.087** |

---

## What Changed: R10 vs R7

| Approach | Exact MAE | Dense MAE |
|----------|----------:|----------:|
| R7 (best before) | 0.295 | 0.365 |
| **R10 (new champion)** | **0.072** | **0.126** |

**4x improvement on exact_only contract.**

---

## Key Insight

The R10 approach succeeds because it:

1. **Does NOT fit sparse quarterly hazards** — Instead of trying to learn fake dynamics from synthetic early-year data
2. **Forecasts scored metrics directly** — `diagnosed_plhiv`, `alive_on_art`, `new_diagnosed_cases_period` 
3. **Uses delta-mode for diagnosis flow** — Forecasts net changes, not levels
4. **Keeps suppression on carry** — 75% carry weight for viral suppression

The model no longer tries to learn dynamics from reconstructed quarterly panels. It forecasts what actually gets scored.

---

## Technical Parameters

**Champion config (SEARCH-R10E-a100-f25-delta-sup75):**
- diagnosed stock weight: `1.0` (100%)
- ART stock weight: `1.0` (100%)
- diagnosis flow weight: `0.25` (25%)
- diagnosis flow mode: `delta` (net change)
- suppression carry weight: `0.75` (75%)

**Dense champion config (SEARCH-R10D-artdelta-ab100-f100):**
- ART level forecast from delta-mode ART series
- full diagnosis flow weight: `1.0`
- full suppression carry weight: `1.0`

---

## Interpretation

The repair strategy works. The core problem was:

- Model tried to fit quarterly hazards in early years where data is sparse/synthetic
- This caused over-trending on logit-transformed hazards
- Fix: Don't fit sparse hazards. Forecast the actual scored metrics directly.

This is an observation-first approach that treats derived transition hazards as diagnostics, not as the primary fitted object.

---

## Visual Outputs

Located at:
```
/mnt/d/EpiGraph_PH/artifacts/runs/tr-v3-repair-search-20260412-s05/analysis/
```

Key files:
- `contracts/exact_only/SEARCH-R10E-a100-f25-delta-sup75.png` — Main prediction vs actual
- `contracts/exact_only/SEARCH-R10E-a100-f25-delta-sup75_hazard_curves.png` — Hazard curves
- `contracts/dense_train_observed_score/SEARCH-R10D-artdelta-ab100-f100.png` — Dense contract champion
- `repair_search_primary_overview.png` — All candidates comparison
- `repair_search_frontier_scatter.png` — Pareto frontier

---

## Next Steps

1. Use R10 as the baseline for future experiments
2. Explore if any additional tuning can improve further
3. Consider adding Phase 2 priors on top of R10 backbone
4. Document in paper as key Phase 3 finding