# Frozen Comparison Section For Paper Draft

**Date:** 2026-04-13  
**Source artifacts:** [tr_v3_publishability_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/tr_v3_publishability_batch_report.md)
**Adjacent methods subsection:** [phase3_tr_v3_methods_contracts_section_2026_04_13.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_methods_contracts_section_2026_04_13.md)

---

## Results: Frozen Contract Comparison

We evaluated the leading Phase 3 TR-V3 candidates under a frozen comparison package designed for manuscript use rather than open-ended model search. The package combined:

- an `exact_only` contract, scored only on exact quarterly observations;
- a `purged_dense` contract, trained on dense mixed-evidence rows but scored only on exact and bridge observations;
- a `legacy_dense` reference contract for comparison;
- a retroactive-from-now paper lockbox using `2025` as a fixed holdout year;
- calibration and residual summaries for the promoted exact and dense candidates.

The lockbox is intentionally described as a **retroactive paper lockbox** rather than a pristine untouched historical split, because the model family had already been tuned on this archive before the present freezing step. That limitation should be stated explicitly in the manuscript.

### Main frozen comparison

Under the `exact_only` contract, the best candidate was `EXP-R10-M1`, which improved mean quarterly normalized MAE from `0.228036` for carry-forward to `0.071418`. The previous exact-lane champion, `EXP-R10-EXACT-CHAMPION`, remained close at `0.072106`, while the mechanistic anchor `EXP-R1` remained substantially worse at `0.319273`.

Under the `purged_dense` contract, the best candidate was `EXP-R10-DENSE-M1`, which reduced mean quarterly normalized MAE from `0.144010` for carry-forward to `0.085418`. The previously promoted dense champion, `EXP-R10-DENSE-CHAMPION`, remained competitive at `0.087028`. A suppression-honesty variant, `EXP-R10-DENSE-H1`, matched the same dense MAE `0.087028` while replacing five `unsupported_level_carry` suppression flags with five `unsupported_or_unclaimed` flags. By contrast, the minimal mechanistic overlay `EXP-R11` degraded to `0.125836`, and the mechanistic anchor `EXP-R1` remained poor at `0.358783`.

These results support three practical promotions for the manuscript:

- `EXP-R10-M1` as the exact-lane paper candidate;
- `EXP-R10-DENSE-M1` as the dense rolling-origin paper candidate;
- `EXP-R10-DENSE-H1` as the honesty-improved dense companion model.

The corresponding mechanistic overlay `EXP-R11` should be retired from the main paper cycle and mentioned only as a negative or secondary result.

### Frozen comparison table

| Contract | Model | Quarterly mean MAE | Baseline MAE | Raw diagnosed MAE | Raw ART MAE | Raw diagnosis-flow MAE | Suppression honesty |
|---|---|---:|---:|---:|---:|---:|---|
| `exact_only` | `EXP-R10-M1` | `0.071418` | `0.228036` | `2156.801` | `2790.245` | `828.490` | `{'scored_direct_support': 5}` |
| `exact_only` | `EXP-R10-EXACT-CHAMPION` | `0.072106` | `0.228036` | `2444.502` | `2790.245` | `828.490` | `{'scored_direct_support': 5}` |
| `exact_only` | `EXP-R1` | `0.319273` | `0.228036` | `35035.621` | `16393.908` | `2763.010` | `{'scored_direct_support': 2, 'unsupported_or_unclaimed': 3}` |
| `purged_dense` | `EXP-R10-DENSE-M1` | `0.085418` | `0.144010` | `4111.096` | `1548.916` | `588.591` | `{'scored_direct_support': 8, 'unsupported_level_carry': 5}` |
| `purged_dense` | `EXP-R10-DENSE-CHAMPION` | `0.087028` | `0.144010` | `4261.963` | `1548.916` | `588.591` | `{'scored_direct_support': 8, 'unsupported_level_carry': 5}` |
| `purged_dense` | `EXP-R10-DENSE-H1` | `0.087028` | `0.144010` | `4261.963` | `1548.916` | `588.591` | `{'scored_direct_support': 8, 'unsupported_or_unclaimed': 5}` |
| `purged_dense` | `EXP-R10-DENSE-M2` | `0.087055` | `0.144010` | `4221.454` | `1548.916` | `590.311` | `{'scored_direct_support': 8, 'unsupported_level_carry': 5}` |
| `purged_dense` | `EXP-R11` | `0.125836` | `0.144010` | `4261.963` | `3364.019` | `736.439` | `{'scored_direct_support': 8, 'unsupported_level_carry': 5}` |
| `purged_dense` | `EXP-R1` | `0.358783` | `0.144010` | `34197.687` | `21067.651` | `941.556` | `{'scored_direct_support': 2, 'unsupported_or_unclaimed': 11}` |

### Lockbox comparison

On the frozen `2025` holdout, `EXP-R10-M1` remained the strongest exact-lane candidate, with quarterly MAE `0.053281` against a carry-forward baseline of `0.082391`. The dense-lane candidates also beat the baseline on the same holdout, but the lockbox ranking compressed strongly, with `EXP-R10-EXACT-CHAMPION`, `EXP-R10-M1`, and `EXP-R11` tying at `0.063449` under both `legacy_dense` and `purged_dense` scoring. This means the lockbox confirms that the promoted candidates are not artifacts of the rolling-origin average, but it does **not** cleanly separate the dense-lane variants from each other.

That is an important paper point: the rolling-origin dense benchmark and the fixed holdout answer slightly different questions. The rolling-origin dense benchmark is more discriminating for model selection, while the fixed `2025` lockbox is better interpreted as a final stability check.

### Calibration and residual behavior

Residual summaries show that both promoted candidates still have structured error.

For `EXP-R10-M1`, mean residuals remain negative for all three scored endpoints:

- diagnosed stock: `-962.890`
- ART stock: `-1178.304`
- diagnosis flow: `-648.321`

For `EXP-R10-DENSE-CHAMPION`, the largest remaining issue is a strong negative bias on diagnosed stock:

- diagnosed stock: `-4244.585`
- ART stock: `-192.366`
- diagnosis flow: `+307.392`

These calibration results imply that the promoted models are publishable as forecasting candidates, but not yet as perfectly calibrated operational systems. In the paper, this should be presented as residual room for improvement rather than hidden.

### Interpretation

The frozen comparison supports four manuscript-level claims.

First, the best-performing models remain observation-first rather than mechanistic hidden-state models. The mechanistic branch is still scientifically useful as a comparator, but it is not the empirical frontier.

Second, the benchmark contract matters. The exact-lane and dense-lane winners are not the same model, and the dense-lane honesty repair changes the interpretation of suppression handling even when headline MAE is unchanged.

Third, the promoted winners are stable enough to survive a frozen `2025` lockbox against carry-forward, which makes them defensible paper candidates.

Fourth, the current evidence does not justify a stronger mechanistic claim. `EXP-R11` did not improve the frozen comparison enough to carry the paper’s main narrative.

### Recommended paper wording

The safest manuscript wording is:

> Under frozen exact-only and purged-dense evaluation contracts, observation-first quarterly stock/flow models outperformed the current mechanistic hidden-state branch. The best exact-lane candidate was a joint-consistency refinement (`EXP-R10-M1`), while the best dense rolling-origin candidate was the dense joint-consistency variant (`EXP-R10-DENSE-M1`). A suppression-honesty companion (`EXP-R10-DENSE-H1`) preserved dense performance while removing unsupported level-carry claims. These results support a provenance-aware, contract-sensitive forecasting interpretation rather than a strong mechanistic epidemic-model claim.

### Figure references

- Exact-lane promoted model curves: [EXP-R10-M1_observation_curves.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/EXP-R10-M1_observation_curves.png)
- Exact-lane calibration: [EXP-R10-M1_calibration.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/EXP-R10-M1_calibration.png)
- Dense-lane promoted model curves: [EXP-R10-DENSE-M1_observation_curves.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/EXP-R10-DENSE-M1_observation_curves.png)
- Dense-lane champion curves: [EXP-R10-DENSE-CHAMPION_observation_curves.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/EXP-R10-DENSE-CHAMPION_observation_curves.png)
- Dense-lane honesty companion curves: [EXP-R10-DENSE-H1_observation_curves.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/EXP-R10-DENSE-H1_observation_curves.png)
- Dense-lane calibration: [EXP-R10-DENSE-CHAMPION_calibration.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/EXP-R10-DENSE-CHAMPION_calibration.png)
- Lockbox exact comparison: [lockbox_exact_only.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/lockbox_exact_only.png)
- Lockbox dense comparison: [lockbox_purged_dense.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/lockbox_purged_dense.png)

---

## Addendum: Frozen Probabilistic Package Update (2026-04-14)

**Source artifacts:** [tr_v3_probabilistic_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-probabilistic-20260414-s00/analysis/tr_v3_probabilistic_batch_report.md), [tr_v3_probabilistic_extension_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-probabilistic-extension-20260414-s03/analysis/tr_v3_probabilistic_extension_batch_report.md)

The frozen comparison package should now distinguish between the promoted deterministic mean models and the promoted uncertainty sidecars.

The deterministic mean promotions remain:

- exact-lane mean model: `EXP-R10-M1-F1-C1`
- dense-lane mean model: `EXP-R10-DENSE-M1-C1-H1`

The uncertainty promotions are now:

- exact-lane uncertainty sidecar: `EXP-UQ-01Z-exact`
- dense-lane uncertainty sidecar: `EXP-UQ-02`

### Exact-lane uncertainty result

`EXP-UQ-01Z-exact` replaced the earlier exact interval repair variants. Relative to the exact bootstrap baseline `EXP-UQ-02`, it improved all three headline uncertainty metrics:

- rolling normalized WIS: `0.340561` vs `0.933561`
- rolling coverage gap: `0.019737` vs `0.269737`
- lockbox normalized WIS: `0.317263` vs `0.438169`

It also outperformed the previous exact repair variants:

- `EXP-UQ-01X-exact`: rolling normalized WIS `0.922658`, coverage gap `0.085526`, lockbox normalized WIS `0.460773`
- `EXP-UQ-01Y-exact`: rolling normalized WIS `0.925497`, coverage gap `0.098684`, lockbox normalized WIS `0.544336`

The winning exact uncertainty construction is not era-stratified. It is bias-aware and asymmetric:

- train-only median residual recentering by endpoint
- lower and upper interval inflation searched separately by endpoint

The learned asymmetric settings were:

- diagnosed stock: center shift `681.27`, lower scale `2.5`, upper scale `1.0`
- ART stock: center shift `618.49`, lower scale `2.5`, upper scale `1.25`
- diagnosis flow: center shift `532.48`, lower scale `2.0`, upper scale `1.25`

### Dense-lane uncertainty result

The dense-lane uncertainty winner remains `EXP-UQ-02`. None of the newer dense probabilistic extensions beat it on the primary rolling benchmark:

- `EXP-UQ-03-dense`: `1.040465`
- `EXP-UQ-04-dense`: `0.854325`
- `EXP-UQ-05`: `0.786941`
- `EXP-PLAT-01`: `0.735846`
- dense `EXP-UQ-02` baseline: `0.688204`

So the publishable uncertainty package is asymmetric across contracts:

- exact lane needs explicit bias-aware interval repair
- dense lane still prefers the simpler moving-block bootstrap sidecar

### Recommended updated manuscript wording

> We froze separate deterministic and probabilistic components for the exact and dense contracts. The exact-lane mean model remained `EXP-R10-M1-F1-C1`, but its uncertainty layer required a bias-aware asymmetric interval repair (`EXP-UQ-01Z-exact`) that substantially improved rolling weighted interval score, coverage-gap error, and lockbox interval performance. By contrast, the dense-lane uncertainty frontier remained dominated by the simpler moving-block bootstrap sidecar (`EXP-UQ-02`); richer jump-, regime-, and plateau-conditioned interval extensions did not improve the primary dense benchmark.

### New figure references

- Exact-lane uncertainty extension report: [tr_v3_probabilistic_extension_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-probabilistic-extension-20260414-s03/analysis/tr_v3_probabilistic_extension_batch_report.md)
- Exact-lane endpoint-by-tier coverage: [EXP-UQ-01Z-exact_endpoint_tier_coverage.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-probabilistic-extension-20260414-s03/analysis/EXP-UQ-01Z-exact_endpoint_tier_coverage.png)
- Probabilistic extension overview: [probabilistic_extension_overview.png](/D:/EpiGraph_PH/artifacts/runs/tr-v3-probabilistic-extension-20260414-s03/analysis/probabilistic_extension_overview.png)
