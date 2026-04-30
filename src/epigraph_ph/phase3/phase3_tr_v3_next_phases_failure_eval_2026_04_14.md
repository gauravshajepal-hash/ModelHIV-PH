# Phase 3 TR-V3 Next Phases: Failure / Evaluation Plan

**Date:** 2026-04-14  
**Method:** council-style failure analysis with bounded autoreason arbitration  
**Question:** What next experimental phases most improve publishability by reducing uncertainty, not just lowering MAE?

---

## 1. Inputs

Reviewed local artifacts only:

- [summary.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-autoresearch-r10m1f1-s1a1-l2a1-20260414-s00/analysis/summary.md)
- [tr_v3_publishability_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/tr_v3_publishability_batch_report.md)
- [phase3_tr_v3_publishability_strategy_2026_04_13.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_publishability_strategy_2026_04_13.md)

Current frontier:

- exact lane winner: `EXP-R10-M1-F1`
- dense lane winner: `EXP-R10-DENSE-M1-H1`
- dense calibration companion: `EXP-R10-DENSE-M1-B1-H1`
- mechanistic line: still secondary, not main-paper backbone

---

## 2. Chairman Read

The highest-value next work is **not** another wide model-family search.

The fastest route to a paper-grade result is:

1. harden the evaluation package until the predictive claim is difficult to attack,
2. run one narrow dense-lane transfer experiment with high prior probability,
3. add one creative stress test that turns the provenance story into a stronger methods contribution.

The biggest remaining risks are:

- repeated-search skepticism,
- endpoint concentration risk,
- contract-specific overclaiming,
- underdeveloped uncertainty reporting,
- lack of a stress test showing that the ranking is not an accident of the current evidence mix.

---

## 3. Autoreason Arbitration

### A
Pure paper-hardening:

- protocol freeze,
- endpoint/tier/era robustness,
- calibration and interval coverage,
- provenance stress test,
- no more model changes.

### B
Score-first continuation:

- port `F1` into dense lane,
- keep tuning `R10`,
- defer most evaluation hardening until later.

### AB
Paper-hardening first, but include one narrow dense-lane transfer test with strong prior support:

- protocol freeze,
- endpoint/tier/era robustness,
- calibration and interval coverage,
- `EXP-R10-DENSE-M1-F1-H1`,
- provenance stress test.

### Winner
`AB`

Reason:

- it keeps the strongest publishability path,
- it still allows one plausible dense-lane improvement,
- it avoids reopening broad exploratory churn,
- it directly addresses the most likely reviewer attacks.

---

## 4. The Five Highest-Information Next Experiments

## 4.1 `EXP-EVAL-01`: Frozen Protocol Rebuild

**Purpose:** Prove that the current headline numbers survive a clean rerun under frozen contracts and selected champion rows.

**Harness**

- Contracts:
  - `exact_only`
  - `legacy_dense`
  - `purged_dense`
  - retroactive `2025` lockbox
- Models:
  - carry-forward baseline
  - `EXP-R10-M1-F1`
  - `EXP-R10-DENSE-M1-H1`
  - `EXP-R10-DENSE-M1-B1-H1`
  - `EXP-R1` as mechanistic anchor
- Outputs:
  - quarterly mean MAE
  - raw endpoint MAE
  - suppression honesty flags
  - split-level metrics
  - artifact hash / config hash

**Keep rule**

- keep the frozen comparison if rankings are unchanged and every primary MAE stays within a narrow reproducibility tolerance of the current values
- revert any paper-grade claim that depends on a ranking flip or unexplained drift

**False win prevented**

- accidental code drift
- cache artifact dependence
- benchmark instability hidden by repeated tuning

---

## 4.2 `EXP-EVAL-02`: Endpoint × Tier × Era Robustness Audit

**Purpose:** Show whether the winner is broad-based or is winning because of one endpoint, one provenance tier, or one time regime.

**Harness**

- Models:
  - carry-forward
  - `EXP-R10-M1-F1`
  - `EXP-R10-DENSE-M1-H1`
  - `EXP-R10-DENSE-M1-B1-H1`
- Break out scoring by:
  - endpoint:
    - `diagnosed_plhiv`
    - `alive_on_art`
    - `new_diagnosed_cases_period`
  - tier:
    - `exact_observed`
    - `bridge_observed`
  - era:
    - pre-2020
    - 2020-2022
    - 2023-2025
- Report:
  - raw MAE
  - normalized MAE
  - split-level win counts
  - paired sign test or equivalent nonparametric win summary

**Keep rule**

- keep the promoted winners only if they beat carry-forward on at least two of the three endpoints and do not collapse on one tier or era
- if a winner depends mostly on diagnosed stock or only on exact rows, downgrade the paper claim to a narrower endpoint-specific forecasting claim

**False win prevented**

- one-endpoint domination
- bridge-tier collapse hidden by pooled MAE
- late-era-only or recent-tail overclaiming

---

## 4.3 `EXP-EVAL-03`: Calibration And Interval Coverage Audit

**Purpose:** Turn the current point-forecast package into a paper-grade uncertainty package.

**Harness**

- Models:
  - `EXP-R10-M1-F1`
  - `EXP-R10-DENSE-M1-H1`
- Use rolling-origin residuals only
- Build empirical prediction intervals per endpoint and per tier
- Report:
  - 50%, 80%, and 95% empirical coverage
  - interval width
  - mean residual
  - calibration by split
  - underprediction / overprediction asymmetry

**Keep rule**

- keep if interval coverage is materially close to nominal and residual bias is stable and explicitly characterizable
- if coverage is badly miscalibrated, do not claim calibrated uncertainty; downgrade to empirical error bands only

**False win prevented**

- good MAE with unusable uncertainty
- hidden directional bias
- overclaiming forecast reliability from point estimates alone

---

## 4.4 `EXP-R10-DENSE-M1-F1-H1`: Dense-Lane Flow-Consistency Transfer

**Purpose:** Port the exact-lane `F1` improvement into the dense champion while preserving the honesty-safe suppression handling from `H1`.

**Harness**

- Primary contract:
  - `purged_dense`
- Secondary contracts:
  - `legacy_dense`
  - `exact_only`
  - retroactive `2025` lockbox
- Compare:
  - `EXP-R10-DENSE-M1-H1`
  - `EXP-R10-DENSE-M1-B1-H1`
  - `EXP-R10-DENSE-M1-F1-H1`
  - carry-forward
- Core outputs:
  - quarterly mean MAE
  - raw diagnosed / ART / flow MAE
  - suppression honesty flags

**Keep rule**

- keep only if `purged_dense` MAE improves below `0.085418`, or ties within negligible tolerance while improving both raw diagnosed MAE and raw flow MAE
- suppression flags must remain `unsupported_or_unclaimed`
- lockbox must remain competitive

**False win prevented**

- importing an exact-lane fix that only improves one metric while quietly worsening the dense contract
- regressing suppression honesty while chasing score

---

## 4.5 `EXP-STRESS-01`: Provenance Perturbation / Contract Stress Test

**Purpose:** Strengthen the paper’s main methods claim by showing how ranking behaves when the evidence mix is degraded or perturbed in controlled ways.

**Harness**

- Start from the current archive-backed panel
- Create controlled perturbation panels:
  - hide a fraction of exact rows and reclassify them as unavailable
  - replace a fraction of exact quarterly rows with bridge-equivalent observations sampled from the historical bridge pattern
  - mask one endpoint at a time
- Rerun:
  - carry-forward
  - `EXP-R10-M1-F1`
  - `EXP-R10-DENSE-M1-H1`
  - `EXP-R1`
- Report:
  - ranking stability
  - degradation curves
  - breakpoint where the winner changes

**Keep rule**

- keep the provenance-aware paper claim if degradation is monotone, ranking changes are interpretable, and observation-first models remain more robust than the mechanistic anchor
- if rankings flip erratically under mild perturbation, narrow the paper claim substantially

**False win prevented**

- lucky leaderboard outcomes driven by one particular evidence mix
- overclaiming generality from a single archive composition

---

## 5. Phased Execution Order

## Week 1: Freeze the paper-grade evaluation package

1. `EXP-EVAL-01`
2. `EXP-EVAL-02`
3. `EXP-EVAL-03`

**Deliverable:** a frozen results package with reproducibility, endpoint/tier/era robustness, and interval coverage.

This week is mandatory. If this package weakens materially, model tuning should pause.

## Week 2: One narrow model extension

4. `EXP-R10-DENSE-M1-F1-H1`

**Decision gate**

- promote it only if it clears the dense keep rule
- otherwise freeze:
  - exact: `EXP-R10-M1-F1`
  - dense: `EXP-R10-DENSE-M1-H1`
  - dense sensitivity: `EXP-R10-DENSE-M1-B1-H1`

## Week 3: Make the methods claim stronger than the benchmark table

5. `EXP-STRESS-01`

**Deliverable:** the creative methods figure showing contract/provenance sensitivity and ranking robustness.

This is the experiment most likely to make the work feel like a publishable methodology paper rather than a one-off benchmark note.

---

## 6. Lower-Priority Parallel Track

These are still worth keeping alive, but not as the next mainline phase:

- `EXP-S1-A1` follow-up only if more annual denominator support appears
- `EXP-L2-A1` follow-up only as a diagnostic appendix
- one constrained mechanistic comparator later, only for a secondary or appendix claim

These should not displace the five experiments above.

---

## 7. AutoResearch Handoff

- **Variant:** `evidence-to-model-loop`
- **Primary evaluation harness:** `exact_only`, `legacy_dense`, `purged_dense`, retroactive `2025` lockbox
- **Mutation units:** evaluation contract package, dense `F1` transfer, provenance perturbation
- **Promotion rule:** prefer uncertainty reduction and claim hardening over incremental MAE gains
- **Stop rule:** once the frozen evaluation package and provenance stress test are complete, stop broad experiment churn and move into manuscript assembly
