# Phase 3 TR-V3 Publishability Strategy

**Date:** 2026-04-13  
**Method:** same-model council with critique round  
**Question:** What is the best publishable way forward for the current Phase 3 TR-V3 model, including tangential or parallel directions that could produce a strong paper quickly?

---

## 1. Bottom Line

The fastest publishable path is **not**:

- "we built a better mechanistic HIV epidemic model for the Philippines"

The fastest publishable path **is**:

- "honest quarterly epidemic forecasting under mixed, provenance-tagged evidence requires explicit train/test contracts, benchmark hardening, and model-class separation"

Under the current archive and benchmark:

- the `R10` observation-first family is the only live predictive frontier,
- the mechanistic `TR-V3` branch is mostly a negative result or secondary research track,
- the exact-vs-dense split is real and publishable,
- the purged dense audit did **not** overturn the dense winner,
- the benchmark story is stronger than the biology story.

This does **not** mean the mechanistic branch is useless. It means it is currently a **secondary paper track**, not the main submission backbone.

---

## 2. Council Consensus

The council converged strongly. No autoreason arbitration was needed.

Three independent roles all agreed on the same core judgment:

1. The strongest current evidence is the `R10` observation-first forecasting line.
2. The cleanest paper is about **provenance-aware, contract-sensitive forecasting and evaluation**.
3. The mechanistic branch is still worth developing, but mostly as:
   - a constrained secondary research track,
   - an identifiability/result-interpretation story,
   - or a follow-up paper.

The critique round sharpened one central warning:

- the weak assumption across almost every tempting plan is that more mechanism is the shortest route to publishability.

The current results do not support that.

---

## 3. Direct Evidence Table

| Claim | Direct evidence | Read |
|---|---|---|
| The exact-lane winner is strong | [tr_v3_experiment_suite_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-experiment-suite-design-v1v2-r10m-r11-20260413-s00/analysis/tr_v3_experiment_suite_report.md) | `EXP-R10-EXACT-CHAMPION = 0.072106` vs baseline `0.228036`; `EXP-R10-M1 = 0.071418` |
| The dense-lane winner is strong | [exp_v1_purged_dense_contract.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-exp-v1-20260413-s00/analysis/exp_v1_purged_dense_contract.md) | Purged dense winner remains `EXP-R10-DENSE-CHAMPION = 0.087028`; ranking unchanged |
| The dense-lane conclusion survives purging | [exp_v1_purged_dense_contract.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-exp-v1-20260413-s00/analysis/exp_v1_purged_dense_contract.md) | `winner_mae_delta = 0.000000`; this removes the strongest immediate attack on the dense lane |
| The mechanistic branch is far behind | [tr_v3_experiment_suite_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-experiment-suite-design-v1v2-r10m-r11-20260413-s00/analysis/tr_v3_experiment_suite_report.md) | `EXP-R1 = 0.319273`; old `05a/05b/L1` families revert badly |
| The frontier is still the `R10` family | [tr_v3_repair_search_report.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-repair-search-20260412-s05/analysis/tr_v3_repair_search_report.md) | Pareto frontier is entirely `R10`-family variants |
| The current winner is not a latent epidemic generator | [phase3_tr_v3_representation_audit_2026_04_12.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_representation_audit_2026_04_12.md) | Winner is an observation-first quarterly stock/flow forecaster |
| The exact-lane improvement can still move | [exp_v2_endpoint_tier_audit.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-experiment-suite-design-v1v2-r10m-r11-20260413-s00/analysis/exp_v2_endpoint_tier_audit.md) | `R10-M1` improves exact MAE and diagnosed-stock raw MAE |
| Mechanistic overlay has not yet justified itself | [dense_r10m_r11.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-dense-r10m-r11-20260413-s00/analysis/dense_r10m_r11.md) | `R11` does not improve the frontier; it mostly stays neutral or worse |

---

## 4. Contextual Evidence Table

| Contextual factor | Source | Why it matters |
|---|---|---|
| Mixed evidence must be treated explicitly | [phase3_tr_v3_autoresearch_design_2026_04_10.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_autoresearch_design_2026_04_10.md) | The design memo already defines the provenance ladder and the split between predictive and mechanistic tracks |
| Hybrid mechanistic-statistical models fail when both sides explain the same thing | [phase3_tr_v3_paper_comparison_2026_04_12.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_paper_comparison_2026_04_12.md) | This supports the current finding that observation-first and mechanism-first goals have diverged |
| The earlier mechanistic branch failed mostly from identifiability and support issues | [phase3_tr_v3_failure_analysis_2026_04_12.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_failure_analysis_2026_04_12.md) | Good negative-result material, but not yet a positive paper backbone |
| The current archive is valuable as a benchmarked mixed-evidence case study | [exp_v2_endpoint_tier_audit.md](/D:/EpiGraph_PH/artifacts/runs/tr-v3-dense-v2-20260413-s00/analysis/exp_v2_endpoint_tier_audit.md) | The support tiers, honesty flags, and endpoint asymmetry are unusually explicit and useful for a methods paper |

---

## 5. Best Paper Strategy

### Paper A: Fastest and strongest

**Working title:**  
`Provenance-aware quarterly epidemic forecasting under mixed evidence: a contract-sensitive benchmark from the Philippines HIV archive`

**Core claim:**  
When surveillance archives mix exact observations, bridge observations, and rule-based reconstructions, model quality depends as much on the evidence contract as on the model class. Under honest blocked-time evaluation, an observation-first stock/flow forecaster outperforms a more mechanistic hidden-state family.

**Why this is the best near-term paper**

- strongest current evidence
- least dependent on new biology claims
- clean benchmark and audit story already exists
- purged dense result removed the most obvious benchmark-validity criticism
- can be defended with current artifacts plus a small final experiment package

**What makes it publishable**

- explicit evidence provenance ladder
- exact vs dense contract split
- purged dense evaluation
- endpoint-tier audit
- direct demonstration that the winning model changes with contract and support
- negative result on under-identified latent hazards

### Paper B: Secondary or follow-up

**Working title:**  
`When quarterly latent epidemic hazards are not identified: a representation audit for partially observed HIV surveillance`

**Core claim:**  
The mechanistic branch does not fail because biology is wrong; it fails because quarterly support is too weak for the free latent transition structure being asked of it.

**Why it is second**

- scientifically interesting
- but it is mostly a negative result at the moment
- needs at least one cleaner constrained mechanistic comparator to avoid looking post-hoc

### Paper C: Tangential but strong

**Working title:**  
`Benchmark contracts for mixed-evidence public-health archives`

**Core claim:**  
Data provenance and contract design change leaderboard outcomes in partially observed epidemiological archives.

**Why it is attractive**

- broader methods audience than HIV alone
- can generalize beyond this disease while still using the Philippines case study
- closer to a framework paper than a disease paper

---

## 6. Publication Decision

If the goal is **publishable as soon as possible**, choose:

- **Paper A as the main target**

Use:

- Paper C as the broader framing language
- Paper B as the secondary discussion or future-work track

Do **not** make the first paper a claim that the repo now contains the best mechanistic HIV model for the Philippines. The evidence does not support that claim.

---

## 7. Highest-Probability Next Experiments

These are ranked by:

1. expected information gain for the paper,
2. probability of success,
3. how much they reduce reviewer attack surface.

### Tier 1: Must happen first

#### 7.1 Frozen contract audit package

**Goal:** build one paper-grade benchmark figure/table package that cannot drift as the repo keeps changing.

Run on the same locked holdout:

- `EXP-R10-EXACT-CHAMPION`
- `EXP-R10-DENSE-CHAMPION`
- `EXP-R10-M1`
- `EXP-R1`

Under:

- exact-only
- legacy dense
- purged dense

With:

- per-metric raw MAE
- per-tier scoring
- suppression honesty flags
- support counts per split

**Why first:** this directly supports the paper backbone.

#### 7.2 Lockbox split

Freeze one final untouched blocked-time holdout window that no further tuning may see.

**Why first:** current results are strong, but repeated search over the same benchmark will be an obvious reviewer concern.

#### 7.3 Calibration and uncertainty report for the winners

For:

- `EXP-R10-M1`
- `EXP-R10-DENSE-CHAMPION`

Add:

- residual distributions
- rolling-origin error spread
- empirical forecast intervals by endpoint
- calibration by exact vs bridge support

**Why first:** papers need more than point MAE.

### Tier 2: Highest-probability model work

#### 7.4 Promote `R10-M1` if it survives lockbox

This is the current best exact-lane specialist.

Keep only if:

- exact lockbox stays better than `0.072106`,
- diagnosed-stock raw MAE remains improved,
- no honesty regression appears.

#### 7.5 Dense champion suppression-honesty repair

Target the remaining `unsupported_level_carry` in the dense champion.

Keep only if:

- dense MAE stays near or below `0.087028`,
- unsupported suppression carries go down,
- purged dense ranking does not break.

#### 7.6 Dense champion plus a transferred M1-style consistency constraint

This is the most plausible way to get one model closer to both lanes without reviving the older mechanistic failures.

Keep only if:

- exact performance improves or stays close,
- dense performance does not collapse toward `0.124`,
- diagnosed-stock raw MAE does not worsen.

#### 7.7 Dense champion plus train-only bias correction

This is lower risk than a broader structural change and can be paper-worthy if it improves calibration more than MAE.

### Tier 3: Secondary research track

#### 7.8 One constrained mechanistic comparator

Build **one** clean mechanistic reconciliation model, not a whole new family.

Properties:

- observation-first outer heads remain
- constrained diagnosis/infection sidecar only
- annual incidence anchor only
- no quarterly mortality block
- no richer leakage
- no suppression claims

Purpose:

- strengthen the negative-result discussion
- test whether minimal mechanism improves annual coherence without losing forecast skill

This is the safe version of the old `R11` idea.

---

## 8. Creative Parallel Directions

These are not the fastest path, but they are good parallel bets if time or agents are available.

### 8.1 General methods paper beyond HIV

Use this repo as the first case study for a broader framework:

- provenance-aware forecasting under mixed evidence
- exact / bridge / extrapolated / latent contracts
- contract-sensitive leaderboard shifts

This could be more publishable than a disease-specific paper if written cleanly.

### 8.2 Archive paper or data descriptor

The archive/hydration/provenance machinery is strong enough that a separate data-methods note may be viable:

- local corpus recovery
- OCR and bridge extraction
- quarter-end stock anchor repair
- provenance ladder
- benchmark-ready historical panel

### 8.3 Negative-result paper on identifiability

This is viable later if paired with one clean constrained mechanistic comparator and one synthetic stress test.

Core claim:

- quarterly latent hazard estimation can look mathematically elegant while being scientifically unsupported in mixed-evidence archives

### 8.4 State-space hybrid paper

If a later `R11`-style model works, this becomes a better second paper than the current mechanistic branch:

- observation-first heads
- constrained latent reconciliation
- annual anchors
- honest unsupported-state flags

But this is not the fastest route today.

---

## 9. Experiments To Avoid Right Now

Do **not** spend the next cycle on:

- full `S(t)` introduction
- richer leakage network
- quarterly mortality block
- more blind sweeps around the old `R1` to `R9` family
- broad `05a`, `N1-N3`, or `L1` revival unless the goal is explicit falsification
- papers that pretend the current winner is already a mechanistic epidemic generator

These are low-probability uses of time under the current evidence.

---

## 10. Checks The Council Agrees Must Happen First

1. Freeze one lockbox holdout.
2. Freeze the benchmark contract package.
3. Report exact vs bridge tier errors separately.
4. Report support counts and suppression honesty flags.
5. Stop using leaderboard MAE alone as the paper backbone.

If those are not done, the paper stays vulnerable to the criticism that it is just a well-tuned local benchmark note.

---

## 11. Winning Plan

### Recommendation

Push **Paper A** first:

- provenance-aware, contract-sensitive quarterly forecasting under mixed evidence

Make the main empirical comparison:

- `EXP-R10-M1`
- `EXP-R10-DENSE-CHAMPION`
- `EXP-R1`
- carry-forward baseline

Under:

- exact-only
- legacy dense
- purged dense
- lockbox holdout

### What the paper should honestly say

- the best predictive model is observation-first
- the evidence contract materially changes what "best" means
- mixed-evidence forecasting needs explicit provenance-aware evaluation
- richer mechanistic claims remain unsupported under the current archive

This is honest, interesting, and publishable sooner than another mechanistic chase.

---

## 12. AutoResearch Handoff

**Variant:** `evidence-to-model-loop`

**Evaluation harness:**

- exact-only frozen contract
- dense-train-observed-score frozen contract
- one untouched lockbox split
- per-metric raw MAE
- per-tier scoring
- suppression honesty flags
- calibration report

**Mutation units:**

- one benchmark-contract package change
- one winner-lane refinement
- one calibration or uncertainty addition
- one constrained mechanistic comparator

**First batch of experiments:**

1. frozen contract audit package
2. lockbox split
3. calibration/uncertainty report for `R10-M1` and `R10-DENSE-CHAMPION`
4. dense champion suppression-honesty repair
5. dense champion plus transferred `M1`-style consistency constraint
6. dense champion plus train-only bias correction
7. one constrained mechanistic comparator for the discussion section

**Stop rule:**

- keep only if the frozen benchmark or lockbox improves without honesty regression
- otherwise revert and strengthen the paper around the already defensible benchmark result

