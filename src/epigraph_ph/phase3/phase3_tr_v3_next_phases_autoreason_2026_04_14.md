# Phase 3 TR-V3 Next Phases Autoreason

**Date:** 2026-04-14  
**Method:** same-model council plus bounded `A / B / AB` autoreason arbitration  
**Question:** What are the next phases of experiments for the current Phase 3 TR-V3 model if the goal is to stay scientifically honest while getting to a publishable result as soon as possible?

---

## 1. Bottom Line

The winning plan is `AB`:

- keep the live `R10` observation-first frontier as the production line,
- harden the paper-grade evaluation package before reopening broad model churn,
- then run one narrow dense-lane transfer and one new structured observation-model phase,
- keep explicit quarterly `S(t)` and richer leakage as sidecars, not promoted benchmark blocks.

The current evidence does **not** support:

- broad mechanistic reopening,
- quarterly explicit `S(t)`,
- richer leakage promotion,
- or more blind `R10` micro-sweeps on the same benchmark.

The current evidence **does** support:

- one more dense-lane transfer from the improved exact winner,
- stronger calibration and uncertainty work,
- a hierarchical observation-model phase,
- a regime/changepoint phase,
- and a provenance stress-test phase.

---

## 2. Strongest Hypotheses

1. The remaining performance upside is still in the observation-first family, especially in cross-head coordination between diagnosed stock, ART stock, and diagnosis flow.
2. The biggest paper risk is now evaluation credibility, not lack of another clever model.
3. The most interesting publishable extension is not a latent epidemic expansion but a more structured observation model:
   - hierarchical reporting structure,
   - train-only regime segmentation,
   - and honest uncertainty.
4. `S(t)` and richer leakage are still useful, but only as diagnostic sidecars under the current archive.

---

## 3. Direct Evidence Table

| Claim | Direct evidence | Read |
|---|---|---|
| Exact frontier is still moving | [summary.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-autoresearch-r10m1f1-s1a1-l2a1-20260414-s00/analysis/summary.md) | `EXP-R10-M1-F1` improved exact MAE from `0.071109` to `0.069471`, and `2025` lockbox MAE from `0.053306` to `0.045118` |
| Flow is the cleanest remaining exact-lane lever | [summary.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-autoresearch-r10m1f1-s1a1-l2a1-20260414-s00/analysis/summary.md) | diagnosis-flow raw MAE improved from `828.490` to `795.799` under `F1` |
| Dense lane is strong but less calibrated | [tr_v3_publishability_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/tr_v3_publishability_batch_report.md) | `EXP-R10-DENSE-M1 = 0.085418`, but dense diagnosed raw MAE remains `4111.096` |
| Honesty repair matters | [tr_v3_publishability_batch_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-publishability-batch-20260413-s04/analysis/tr_v3_publishability_batch_report.md) | `EXP-R10-DENSE-H1` preserved dense MAE while replacing unsupported suppression carry with unclaimed suppression |
| Mechanistic branch remains a comparator, not the frontier | [phase3_tr_v3_frozen_comparison_section_2026_04_13.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_frozen_comparison_section_2026_04_13.md) | `EXP-R1` and `EXP-R11` remain well behind the live `R10` branch |
| `S(t)` does not clear support | [exp_s1_a1_susceptible_sidecar.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-autoresearch-r10m1f1-s1a1-l2a1-20260414-s00/analysis/exp_s1_a1_susceptible_sidecar.md) | only `9` joint annual support years; status stays `defer` |
| Richer leakage does not clear support | [exp_l2_a1_late_leakage_sensitivity.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-autoresearch-r10m1f1-s1a1-l2a1-20260414-s00/analysis/exp_l2_a1_late_leakage_sensitivity.md) | late suppression shortfall is only an upper-bound proxy; status is `late_window_sensitivity_only` |

---

## 4. Contextual Evidence Table

| Context | Source | Why it matters |
|---|---|---|
| The winning family is observation-first by construction | [tr_v3_experiment_suite.py](/D:/EpiGraph_PH/src/epigraph_ph/phase3/tr_v3_experiment_suite.py) | the direct-observation path still discards mechanistic and observation configs internally, so claims must match the actual model class |
| Exact, legacy dense, purged dense, and retroactive lockbox are already formalized | [phase3_tr_v3_methods_contracts_section_2026_04_13.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_methods_contracts_section_2026_04_13.md) | paper-grade evaluation language is already in place; this supports benchmark hardening as a real contribution |
| The publishability backbone is benchmark/provenance, not biology | [phase3_tr_v3_publishability_strategy_2026_04_13.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_publishability_strategy_2026_04_13.md) | the current evidence supports a forecasting/benchmark paper first |
| The main paper should not claim a mechanistic breakthrough | [phase3_tr_v3_frozen_comparison_section_2026_04_13.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_frozen_comparison_section_2026_04_13.md) | the current comparator set is already enough to support a negative or secondary mechanistic story |

---

## 5. Autoreason Inputs

### A: Evaluation-hardening-first

Main idea:

- freeze the candidate set,
- rebuild the paper-grade benchmark package,
- add calibration and coverage,
- run one untouched outer evaluation,
- stop broad mutation until that is done.

Strength:

- strongest defense against reviewer attack.

Weakness:

- improves credibility more than model capability.

### B: New-model-first

Main idea:

- move immediately into a new structured observation family,
- especially hierarchical observation heads and train-only changepoints,
- using the current `R10` winners as the base class.

Strength:

- strongest chance of a genuinely new methods result.

Weakness:

- too easy to slip back into repeated-search risk before the current paper package is frozen.

### AB: Freeze first, then one narrow transfer plus one new structured phase

Main idea:

- harden the benchmark package now,
- allow one dense-lane transfer from the improved exact winner,
- then open one structured observation-model phase and one evaluation-strengthening phase,
- keep `S(t)` and leakage diagnostic-only.

Strength:

- preserves publishability discipline while still moving the model frontier.

Weakness:

- slower than a pure score-chasing plan.

---

## 6. Arbitration Result

**Winner:** `AB`

### Why `AB` won

`A` was too conservative. It strengthens the paper, but it leaves a credible remaining upside untouched.

`B` was too aggressive. It invites repeated-search criticism before the current frontier is properly frozen.

`AB` wins because it:

- respects the main validity threats,
- preserves one high-probability model improvement path,
- opens one genuinely new methods phase,
- and keeps unsupported biological structure outside the live benchmark loop.

---

## 7. Next Phases Of Experiments

### Phase 0: Freeze and Rebuild the Paper-Grade Evaluation Package

Run:

- `EXP-EVAL-01` Frozen Protocol Rebuild
- `EXP-EVAL-02` Endpoint × Tier × Era Robustness Audit
- `EXP-EVAL-03` Calibration and Interval Coverage Audit

Purpose:

- reduce reviewer attack surface,
- make the paper backbone stable,
- and stop accidental drift in the published tables.

Keep/revert:

- keep only if rankings and MAEs reproduce within tight tolerance,
- and coverage/calibration claims are numerically defensible.

### Phase 1: Dense-Lane Flow-Consistency Transfer

Run:

- `EXP-R10-DENSE-M1-F1-H1`

Definition:

- start from `EXP-R10-DENSE-M1-H1`
- transfer the successful `F1` flow-consistency idea from the exact lane
- preserve `unsupported_or_unclaimed` suppression honesty

Primary harness:

- `purged_dense`

Secondary harness:

- `legacy_dense`
- `exact_only`
- retroactive `2025` lockbox

Keep/revert:

- keep if dense MAE beats `0.085418`,
- or ties within negligible tolerance while improving both raw diagnosed and raw flow MAE,
- and honesty flags remain `unsupported_or_unclaimed`.

### Phase 2: Hierarchical Observation Model

Run:

- `EXP-R10-H1-01`
- `EXP-R10-H1-02`
- `EXP-R10-H1-03`

Definition:

- introduce a shared reporting-regime factor across:
  - `diagnosed_plhiv`
  - `alive_on_art`
  - `new_diagnosed_cases_period`
- let the factor move observed heads together, instead of correcting each head independently

Examples:

- low-rank reporting factor
- shared residual factor with bounded endpoint-specific loadings
- province/report-cycle proxy factor if available from the archive

Keep/revert:

- exact improves below `0.069471`,
- or exact ties while diagnosed and flow raw MAE both improve,
- dense stays at or below `0.085418`,
- no honesty regression.

Paper value:

- this is the strongest creative next phase with real novelty:
  provenance-aware hierarchical stock/flow forecasting.

### Phase 3: Regime Segmentation / Changepoint-Aware Forecasting

Run:

- `EXP-R10-CP-01`
- `EXP-R10-CP-02`

Definition:

- infer train-only changepoints or regimes,
- then fit regime-aware observation models,
- compare against the single-regime `R10` line.

Allowed regime drivers:

- purely train-only changepoints,
- archive-defined reporting eras,
- COVID/post-COVID split only if learned or externally justified and kept secondary.

Keep/revert:

- exact or dense must improve materially,
- changepoints must be stable across rolling splits,
- and the result must improve interpretation rather than just add another tuning layer.

Paper value:

- contract-sensitive regime discovery under mixed epidemic evidence.

### Phase 4: Provenance Perturbation / Archive-Noise Stress Test

Run:

- `EXP-STRESS-01`

Definition:

- degrade the panel in controlled ways:
  - hide some exact rows,
  - replace some exact rows with bridge-like rows,
  - mask one endpoint at a time,
  - perturb row-level support tiers
- rerun:
  - carry-forward
  - `EXP-R10-M1-F1`
  - dense winner
  - `EXP-R1`

Keep/revert:

- keep the provenance-aware paper claim only if degradation is monotone,
- and ranking changes are interpretable rather than chaotic.

Paper value:

- this turns provenance from a bookkeeping detail into a tested methodological claim.

### Phase 5: Conformal / Calibration Layer

Run:

- `EXP-CAL-01`
- `EXP-CAL-02`

Definition:

- build 50/80/95% empirical or conformal intervals by:
  - endpoint
  - tier
  - contract
- keep point forecasts unchanged initially

Keep/revert:

- coverage near nominal,
- intervals not trivially wide,
- no material MAE regression if integrated into the live forecast runner.

Paper value:

- honest uncertainty under provenance-stratified epidemic forecasting.

### Phase 6: Creative Parallel Sidecars

These are worthwhile, but not on the main benchmark line.

1. `Subnational / KP sidecar forecasting`
   - bounded auxiliary heads only
   - reconciled back to national totals

2. `Archive-noise-aware robust scoring`
   - noise classes by source type
   - robust loss vs standard loss comparisons

3. `Constrained mechanistic sidecar comparator`
   - one observation-first winner plus annual-incidence coherence sidecar
   - no quarterly `S(t)`
   - no richer leakage
   - no suppression claims

---

## 8. Interventions Worth Trying

- dense-lane `F1` transfer
- hierarchical observation heads
- train-only changepoint/regime inference
- provenance perturbation stress tests
- calibration and conformal interval layers
- archive-noise-aware residual analysis

## 9. Interventions To Avoid

- more blind `R10` micro-sweeps on the same exact/dense benchmarks
- quarterly explicit `S(t)` promotion
- richer leakage promotion into the live benchmark
- turning unsupported suppression back into a forecast endpoint
- using the retroactive `2025` lockbox as a tuning target
- claiming the current winner is a mechanistic epidemic model

---

## 10. Checks The Council Agrees Must Happen First

1. Freeze the candidate set for the paper line.
2. Rebuild the frozen protocol package and verify it reproduces.
3. Report endpoint × tier × era robustness, not just overall MAE.
4. Preserve suppression honesty flags as a first-class gate.
5. Keep `S(t)` and richer leakage outside the benchmark loop.

---

## 11. AutoResearch Handoff

**Variant:** `evidence-to-model-loop`

**Why this variant still fits**

- the project already has stable metrics,
- but the real problem is still evidence contracts, calibration, and validity hardening,
- not pure score chasing.

**Evaluation harness**

- `exact_only`
- `legacy_dense`
- `purged_dense`
- retroactive `2025` lockbox
- endpoint-level raw MAE
- tier-level scoring
- era-level scoring
- suppression honesty flags
- calibration and interval coverage

**Mutation units**

1. one evaluation hardening change
2. one dense transfer experiment
3. one structured observation-model phase
4. one stress-test phase
5. one uncertainty/calibration phase

**Stop rule**

- keep only if the frozen benchmark or outer evaluation improves without honesty regression,
- otherwise revert and retain the simpler candidate.

**First 5 iterations**

1. `EXP-EVAL-01` Frozen Protocol Rebuild
2. `EXP-EVAL-02` Endpoint × Tier × Era Robustness Audit
3. `EXP-EVAL-03` Calibration and Interval Coverage Audit
4. `EXP-R10-DENSE-M1-F1-H1`
5. `EXP-R10-H1-01`

**Second batch**

6. `EXP-R10-H1-02`
7. `EXP-R10-CP-01`
8. `EXP-STRESS-01`
9. `EXP-CAL-01`
10. one constrained mechanistic sidecar comparator

---

## 12. Practical Recommendation

If the goal is to move forward immediately:

1. freeze the current paper candidates:
   - exact: `EXP-R10-M1-F1`
   - dense: `EXP-R10-DENSE-M1-H1`
   - comparator: `EXP-R1`
2. run the evaluation-hardening package,
3. then run `EXP-R10-DENSE-M1-F1-H1`,
4. then start the hierarchical observation-model phase.

That is the highest-value path that is both creative and scientifically defensible.
