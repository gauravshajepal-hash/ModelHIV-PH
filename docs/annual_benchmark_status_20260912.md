# Annual benchmark status and next defensible model experiment

**Verdict: no UNAIDS/AEM/Spectrum superiority claim.** This pass preserves model parameters, historical predictions, and the R97 prospective Q4 lock. It changes the observation likelihood available to a new branch and narrows annual publication claims that the audit cannot support.

## Executed work

- R98 extracted six reconciled AHD status partitions, quarantined four ambiguous partitions, implemented a three-category missingness likelihood, and ran three conditional status forecasts under event-time and mirror-availability splits. [Results and figures](phase3_r98_ahd_missingness_20260911.md).
- R93 was rerun as `p3d-r93-incumbent-recheck-20260911`. This **re-adjudicates frozen R78/R79 scores**, not a fresh model fit. Their source hashes remain recorded in the result.
- The annual design-role audit executed all 12 locked splits, with three metrics each. All 36 metric/split designs contain rows labeled `validation_only` that the legacy array constructor accepts. No new model was fitted to these inadmissible rows. [Machine-readable audit](annual_benchmark_integrity_20260912.json).
- The effective claim registry was regenerated as `p3d-r53-claim-registry-r98-annual-integrity-20260912`. R41 remains the national research reference; R98 is diagnostic-only. Annual R75/R76/R78/R80/R86/R88/R90 claims are limited to retrospective estimate agreement. Historical statuses and scores are preserved.

[Matched benchmark snapshot](phase3_r93_benchmark_recheck_20260911.json) | [Effective claim registry](phase3_claim_registry_20260912.json)

## Actual matched comparison

The R93 comparator is `public_train_selected_annual_proxy_v2`, a local public-data model, **not** a recovered AEM or Spectrum implementation/output. Its selected families include piecewise log trends and a local-level mass-balance head. It is a reproducible internal baseline, not verified official forecasting skill.

| Annual metric | Existing matched Phase 3 head | Local public-data comparator | Interpretation |
|---|---:|---:|---|
| New infections | 0.267854 | 0.042749 | Largest relative gap |
| AIDS deaths | 0.394716 | 0.380824 | Both weak; comparator slightly lower |
| PLHIV | 0.067862 | 0.067862 | Tie |
| Pooled | 0.243477 | 0.163811 | Phase 3 head loses this internal test |

These are mean absolute errors normalized by the largest training-window value for each metric, not percentages of the outcome. There are 28 scored forecast/year entries per metric across overlapping 1/3/5-year windows, not 28 independent annual observations. Source estimates cover 2010-2024; evaluation targets span 2019-2024. This does not score R98 or a new model against 2026 data.

## Why the claim boundary changed

**Training roles:** `r75_bulk_unaids_annual_challenge.py::_bulk_unaids_target_rows` emits `observation_role=validation_only` and `allowed_use=validation_only`. R78's `_metric_train_arrays` consumes pre-cutoff numeric entries without checking those fields. R86 likewise passes raw pre-cutoff annual rows into its annual-head fitter. This is a role-contract inconsistency, not evidence that target-year labels were directly passed into a fit. The older design intended weak annual calibration, but must declare a separate permissible auxiliary input rather than implicitly exempt a validation-only row. The new `require_training_role` helper fails closed. It is used for this audit; legacy fitters remain frozen rather than silently rerun under different semantics.

**Availability:** training years before an origin do not prove that a revised 2025 estimate of those years existed at that origin. Current-vintage retrospective reconstruction must be labeled accordingly. A forecasting claim needs historical releases or a genuinely locked future test.

**Uncertainty:** legacy interval coverage tests whether a *point prediction* lies between the target estimate's reported lower and upper bounds. This is target-interval containment, not the frequency with which independent outcomes fall inside model predictive intervals. The reported 0.75 versus 0.9286 values are not calibrated predictive coverage.

**Identity:** matching UNAIDS estimates is neither independent validation of incidence nor evidence of outperforming UNAIDS. Actual official forecasts and independent outcomes must share a declared horizon, target, vintage, and uncertainty score.

## Latest public-source check

[AIDSinfo](https://aidsinfo.unaids.org/dataset) advertises a 2026 estimates ZIP and recommends original unrounded Spectrum estimates for calculations. The advertised `Estimates_2026_en.zip` endpoint returned HTTP 403 to HEAD and ordinary GET on this retrieval attempt. No access restrictions were bypassed.

The [UNAIDS uncertainty workbook webpage](https://www.unaids.org/en/resources/documents/2025/HIV_estimates_with_uncertainty_bounds_1990-present) titles its coverage 1990-2025, but its linked workbook contains `Source: UNAIDS 2025 estimates`, sheets named `HIV2025Estimates_*`, and Philippine data ending in **2024**. This mismatch is recorded instead of treating a URL or current download date as a new vintage.

The 5.6 MiB workbook was processed read-only, row by row; duplicate ByArea sheets were excluded. SHA-256: `ecdba947ff1589663348350f01b16b289bbef8f0a2f47914da9e43712c507be9`. The [Philippine extraction and vintage audit](unaids_workbook_vintage_audit_20260911.json) preserves 105 core annual estimate/uncertainty rows, raw country rows and headers, source coordinates, rounded/censored cells, and validation-only roles. No downloaded row entered training. Latest displayed all-age estimates are 31,000 infections, 2,300 AIDS deaths, and 220,000 PLHIV for 2024, with published uncertainty retained. These are estimates, not registry case counts or fresh holdout observations.

## Next model program

1. **Repair origin-specific roles.** Keep genuinely reserved annual estimates validation-only. Admit weak annual calibration only through a separately declared auxiliary input with source version, eligibility date, and explicit fit target. Never relabel held-out evidence to rescue a model. Retrospective estimate-emulation experiments need a distinct nonforecast scope.
2. **Fit diagnosis/reporting and clinical classification jointly on admissible surveillance evidence.** Compare the frozen reference with the R98 classification adapter, keeping downstream states fixed. Profile classification probabilities rather than forcing unknown status to early disease. If incidence/backlog remains underidentified, report a feasible range and reject a point-mechanism claim.
3. **Narrow the diagnosis kernel with independent progression evidence.** CD4/AHD labels require a progression/observation model before identifying time since infection. Compare the clinical-emission hypothesis with a no-clinical-information control; do not convert median-CD4 ranks into probabilities.
4. **Run a small nested blocked-time comparison after the contract passes.** Score diagnosis flow and stock reconciliation, with annual incidence agreement separate. Keep R41, carry-forward, R10, and the local annual comparator distinct; require no stock-cone or conditional VL/suppression regression.
5. **Reserve genuine future outcomes.** R97 Q4-2026 forecasts remain immutable and unscored until a report is available. Q1/Q2-2026 have already been repeatedly inspected and cannot become a fresh lockbox. Subnational and determinant-intervention claims remain separately gated.

This is an evidence-standard correction, not a claim that code hygiene or a lower in-sample score improves epidemic predictions. The finish condition remains matched improvement with valid roles, realistic information availability, conserved dynamics, and out-of-sample uncertainty evaluation.

## Verification

89 targeted tests passed across the new likelihood, source-vintage reader, annual contract audit, existing diagnosis/incidence repair, and R90-R98 modules. Tests cover analytic-versus-numerical profile optimization, conservation, quarantine and validation-role blocking, target-mutation isolation, report availability, row/source/figure hashes, and effective claim limitations. The full project test suite was not run. No R41 parameters or R97 Q4 forecast bytes changed. The 2026 estimates ZIP was not obtained; the completed workbook remains cached outside Git and its compact extraction is tracked.
