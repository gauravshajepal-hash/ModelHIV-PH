# Phase 3 TR-V2 Implementation Plan

**Date:** 2026-04-06
**Author:** Epigraph PH (G S + AI agent)
**Status:** Draft — awaiting review

**Document purpose:**
This plan implements the full TR-V2 experimental program for Phase 3, building on the architecture note (`phase3_architecture_and_frontier_2026_04_05.md`) and the implementation plan (`phase3_transition_frontier_implementation_plan_2026_04_05.md`). It extends both with concrete implementation details, file-level specifications, and a new "trajectory understanding" test.

---

## 1. Current State Assessment

### 1.1 Directory Structure (Post-Restructure)

```
src/epigraph_ph/phase3/
├── _lineage/              # Broad benchmark + infrastructure reservoir
│   ├── rescue_core.py     # Multiyear, multilevel Phase 3 model (broad benchmark)
│   ├── national_reset_core.py
│   ├── national_reset_pipeline.py
│   ├── peak_search.py
│   ├── shocks.py
│   └── pipeline.py
├── frontier/              # Active research frontier (was transition_research/)
│   ├── transition_engine.py  # Core hazard model (~3092 lines)
│   ├── age_research.py    # AGE-01A, AGE-01B, AGE-01C
│   ├── peak_windows.py    # PEAK-01A through PEAK-01F
│   ├── decomposition.py   # DECOMP-01A through DECOMP-01F
│   ├── analytics.py       # Factor analytics and scoring
│   ├── artifacts.py       # Context, artifact writing, numeric policy
│   ├── cli.py             # CLI entry points
│   ├── registry.py        # Experiment definitions (AGE, PEAK, DECOMP, MECH)
│   ├── sources.py         # Data loading
│   └── numeric_policy.py  # Numerical guards
├── incidence/             # Dependent incidence extension
│   ├── modeling.py
│   ├── registry.py
│   ├── sources.py
│   ├── artifacts.py
│   ├── audits.py
│   └── cli.py
└── shared/                # Shared platform utilities
    ├── broad_backtest_support.py
    ├── evaluation_regimes.py
    ├── mixed_frequency.py
    ├── temporal_scaffold.py
    └── numerics.py
```

### 1.2 Winning Cascade

The current winning branch lineage is:

```
MECH-01A → MECH-01E → DECOMP-01E/F → PEAK-01F → AGE-01B → INC-01D
```

**AGE-01B** is the selected base for TR-V2-00 because:
- Inherits PEAK-01F (region-plus-KP-modifier gated fused forecast)
- Adds narrow, support-aware youth diagnosis modifier
- Uses leave-one-out gating
- Does not reopen unsupported downstream structure
- Cleanest identifiability among all winning branches

### 1.3 What Is Implemented vs What Is Not

| Workstream | Status | Notes |
|---|---|---|
| Slice 1: Platform extraction | ✅ **Done** | `shared/` + `_lineage/` created, imports fixed |
| Slice 2: Baseline lock (TR-V2-00) | ❌ **Not started** | No TR-V2-* in registry, no baseline_lock.py |
| Slice 3: Phase 2 direct priors (TR-V2-01) | ❌ **Not started** | No phase2_hazard_priors.py |
| Slice 4: Hidden shocks (TR-V2-02) | ❌ **Not started** | No hidden_shocks.py |
| Slice 5: Ablation suite (TR-V2-03) | ❌ **Not started** | No ablation runner |
| Slice 6: Incidence explicitness | ❌ **Not started** | No INC-V2-* entries |
| Slice 7: Reintegration readiness | ⏸️ **Readiness only** | Interface docs, no code changes |

### 1.4 Key Mathematical Objects Already Present

From `transition_engine.py` (lines 1–3092):

- **State vector:** `x_t = (U_t, D_t, A_t, V_t, L_t)` — national quarter-level
- **Transition names:** `U_to_D`, `D_to_A`, `A_to_V`, `A_to_L`, `L_to_A`
- **Hazard model:** `h_r(t)` per transition, clipped to `[0, 1]`
- **Helper model:** `hazard_t = intercept + persistence * hazard_{t-1} + helper_scale * signal_t`
- **Residual model:** `hazard_t = hazard_mean + residual_scale * signal_t`, clipped to `[hazard_mean - std, hazard_mean + std]`
- **State evolution:** `x_{t+1} = F(x_t, h_t)` with explicit flow accounting and A-compartment mass conservation
- **Peak detection (PEAK-01F):** `region-plus-KP-modifier` detector gates downstream shock/residual fusion

### 1.5 Pre-Existing Issues

1. `test_inc_00b_and_inc_00c_live_runs` — assertion mismatch on `identifiability_class` (test expectation drift)
2. `test_inc_01d_and_inc_01b_live_runs` — `FileNotFoundError: 'D:/EpiGraph_PH/artifacts/runs'` (Windows path in sandbox)
3. `_discover_latest_transition_experiment()` in `transition_engine.py` searches `transition_research/` path — needs update to `frontier/`

---

## 2. New Experiments Registry

### 2.1 Transition Research TR-V2 Series

| ID | Name | Description | Expected Outputs |
|---|---|---|---|
| `TR-V2-00-age01b-baseline-lock` | Locked AGE-01B reproduction | Exact reproduction of AGE-01B as regression baseline | `baseline_lock.json`, `baseline_comparison.json`, `evaluation.json`, `frontier_lineage.json` |
| `TR-V2-01-phase2-direct-hazard-priors` | Phase 2 sparse direct edges | Phase 15 latent block states enter as structured lagged hazard priors `Gamma[r,b,l]` | `phase2_direct_hazard_prior_summary.json`, `baseline_comparison.json`, `evaluation.json` |
| `TR-V2-02-phase2-hidden-shock-hazards` | Phase 2 low-rank hidden shocks | Hidden temporal modes as separate latent shock channels `u_m(t)` with `Lambda` coefficients | `phase2_hidden_shock_summary.json`, `baseline_comparison.json`, `evaluation.json` |
| `TR-V2-03-phase2-ablation-suite` | Ablation suite | Direct-only, hidden-only, both, with/without peak gating | `transition_frontier_ablation_summary.json` |
| `TR-V2-04-region-pooled-transition-frontier` | Region-pooled frontier | Extending winning national model to region-level pooling | Region-pooled outputs |
| `TR-V2-05-trajectory-understanding` | Peak prediction without detector | Model projects forward and finds peaks from its own dynamics | `trajectory_projection.json`, `predicted_peaks.json`, `peak_prediction_accuracy.json` |

### 2.2 Incidence INC-V2 Series

| ID | Name | Description | Expected Outputs |
|---|---|---|---|
| `INC-V2-00-lineage-explicitness-audit` | Incidence lineage audit | Standardized recording of parent transition branch for all incidence runs | `incidence_lineage_audit.json` |
| `INC-V2-01-phase2-compatible-incidence-extension` | Phase 2-compatible incidence | Incidence model that respects Phase 2 direct/hidden structure separation | `phase2_incidence_summary.json`, `fit_artifact.json`, `evaluation.json` |

---

## 3. Implementation Slices

### Slice 2: TR-V2-00 Baseline Lock

**Status:** Not started
**Dependency:** Slice 1 (done)
**Target base:** `AGE-01B-youth-diagnosis-modifier`

#### 3.1 Files to Add

**`src/epigraph_ph/phase3/frontier/baseline_lock.py`**

Purpose: Create a locked, immutable reproduction of AGE-01B. This becomes the regression baseline against which every TR-V2-* experiment is compared.

Implementation:
```python
# baseline_lock.py — locked reproduction of AGE-01B

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epigraph_ph.runtime import ROOT_DIR, read_json, write_json
from .artifacts import write_experiment_artifacts
from .registry import make_transition_run_id

@dataclass(frozen=True, slots=True)
class BaselineLockRecord:
    parent_experiment_id: str  # "AGE-01B-youth-diagnosis-modifier"
    parent_run_id: str         # The actual run that produced the winning result
    inherited_hazards: dict    # Transition hazards from parent
    inherited_holdout_quarters: list[str]
    inherited_metrics: dict    # MAE, sMAPE, diagnosis-flow MAE, peak alive-on-ART error
    lock_version: int          # Incremented if re-locked from a newer parent run

def lock_baseline(
    *,
    parent_experiment_id: str = "AGE-01B-youth-diagnosis-modifier",
    seed: int = 0,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Create a locked baseline record from the latest AGE-01B run.
    
    This function:
    1. Discovers the latest AGE-01B experiment artifact
    2. Extracts all transition hazards, holdout quarters, and metrics
    3. Writes an immutable baseline_lock.json
    4. Returns the lock record for use by downstream experiments
    
    The lock record becomes the immutable reference. If AGE-01B is
    re-run with different parameters, the lock must be explicitly
    renewed — it does not auto-update.
    """
    pass

def reproduce_baseline(
    lock_record: BaselineLockRecord,
    ctx: TransitionResearchContext,
) -> dict[str, Any]:
    """
    Run the exact same computation as AGE-01B using the locked parameters.
    Returns evaluation.json, baseline_comparison.json, and state trajectories.
    
    Acceptance: metrics match parent within machine tolerance (1e-10).
    """
    pass
```

#### 3.2 Registry Changes

Add to `registry.py`:
```python
"TR-V2-00-age01b-baseline-lock": ExperimentDefinition(
    experiment_id="TR-V2-00-age01b-baseline-lock",
    cli_name="tr-v2-00",
    description="Locked baseline reproduction of AGE-01B-youth-diagnosis-modifier. "
                "This is the immutable regression baseline for all TR-V2-* experiments.",
    expected_outputs=(
        "baseline_lock.json",
        "baseline_comparison.json",
        "evaluation.json",
        "frontier_lineage.json",
    ),
),
```

#### 3.3 Acceptance Criteria

1. `TR-V2-00` reproduces `AGE-01B` metrics within machine tolerance (1e-10 relative).
2. `baseline_lock.json` explicitly documents:
   - Parent experiment ID and run ID
   - Inherited hazard values for all 5 transitions
   - Inherited holdout quarters
   - Inherited MAE, sMAPE, diagnosis-flow MAE, peak alive-on-ART error
3. The lock record is versioned. If a new AGE-01B run supersedes the current one, the lock version increments.
4. CLI: `epigraph-ph phase3 frontier tr-v2-00` runs the lock reproduction.

#### 3.4 Implementation Steps

1. Create `baseline_lock.py` with `lock_baseline()` and `reproduce_baseline()`
2. Add TR-V2-00 entry to `registry.py`
3. Add CLI handler in `cli.py` for `tr-v2-00`
4. Update `_discover_latest_transition_experiment()` in `transition_engine.py` to search `frontier/` path instead of `transition_research/` (pre-existing bug fix)
5. Run `epigraph-ph phase3 frontier tr-v2-00` and verify output matches AGE-01B
6. Write `baseline_lock.json` to `artifacts/runs/<run_id>/frontier/TR-V2-00-age01b-baseline-lock/`

---

### Slice 3: TR-V2-01 – Phase 2 Direct Hazard Priors

**Status:** Not started
**Dependency:** Slice 2 (baseline lock)
**Mathematical target:** `logit h_r(t) = alpha_r(t) + base_r(t) + sum_{b,l} Gamma[r,b,l] z_b(t-l)`

#### 3.1 Files to Add

**`src/epigraph_ph/phase3/frontier/phase2_hazard_priors.py`**

Purpose: Read Phase 15 latent block output, map eligible block-lag pairs to transition channels, introduce `Gamma[r,b,l]` as structured hazard priors.

Data sources (Phase 2 / Phase 15):
- Phase 15 latent block state tensor: aggregated to national quarter scale
- Phase 2 sparse direct temporal edges from the kept latent graph bundle
- Support/stability summaries for weighting prior strength

Implementation outline:

```python
# phase2_hazard_priors.py — Phase 2 direct hazard priors for transition model

from __future__ import annotations
from typing import Any
import numpy as np

from epigraph_ph.phase3.frontier.registry import TRANSITION_NAMES
from epigraph_ph.runtime import ROOT_DIR, read_json

def load_phase15_latent_blocks(
    *,
    archive_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load Phase 15 latent block state tensor and aggregate to national quarter scale.
    
    Returns:
        block_states: dict mapping block_id to (time_steps,) array of national-level values
        block_metadata: dict with block descriptions and support scores
    """
    pass

def load_phase2_sparse_edges(
    *,
    archive_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load Phase 2 sparse direct temporal edges.
    
    These are edges of the form (block_b, lag_l) -> (target) with support/stability scores.
    
    Returns:
        edges: list of dicts with 'block_id', 'lag', 'target', 'weight', 'support'
    """
    pass

def build_hazard_prior_matrix(
    *,
    block_states: dict[str, np.ndarray],
    sparse_edges: dict[str, Any],
    transitions: tuple[str, ...] = TRANSITION_NAMES,
    max_lag: int = 4,
    support_threshold: float = 0.3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build the Gamma[r,b,l] prior matrix for hazard model inclusion.
    
    For each transition r, block b, and lag l:
    - If a sparse edge exists with support >= threshold, include it
    - Prior scale is proportional to support * phase2_stability
    - Priors are zero-mean with learned scale (empirical Bayes)
    
    Returns:
        gamma_prior: (n_transitions, n_blocks, max_lag) array
        prior_metadata: dict with which edges were included/excluded and why
    """
    pass

def apply_hazard_priors(
    *,
    base_hazards: np.ndarray,  # (n_quarters, n_transitions)
    block_states: dict[str, np.ndarray],
    gamma_prior: np.ndarray,   # (n_transitions, n_blocks, max_lag)
    prior_scale: dict[str, float],  # per-transition prior scale
) -> np.ndarray:
    """
    Apply Phase 2 direct priors to base hazards.
    
    For transition r at quarter t:
        prior_contribution = sum_{b,l} Gamma[r,b,l] * z_b(t-l)
        adjusted_hazard_r(t) = base_hazard_r(t) + prior_contribution
    
    The prior contribution is clipped to reasonable range to prevent
    over-correction. The base hazard must still dominate unless the
    Phase 2 signal is extremely strong.
    
    Returns:
        adjusted_hazards: (n_quarters, n_transitions) array
    """
    pass
```

#### 3.2 Engine Changes

In `transition_engine.py`, add a new experiment handler for TR-V2-01 that:
1. Loads the TR-V2-00 baseline lock
2. Loads Phase 15 latent blocks and Phase 2 sparse edges
3. Builds the `Gamma[r,b,l]` prior matrix
4. Applies priors to the base hazard model from `AGE-01B`
5. Runs holdout simulation with adjusted hazards
6. Outputs comparison against TR-V2-00

#### 3.3 Acceptance Criteria

1. The implementation keeps direct Phase 2 structure separate from hidden low-rank structure.
2. Emits a table of:
   - Retained direct block-lag terms
   - Corresponding hazard channels
   - Prior mean and prior scale
3. The model can be ablated against TR-V2-00 (direct priors ON vs OFF).
4. No Phase 2 signal is injected as a generic modifier — all entries are structured as `Gamma[r,b,l]`.

---

### Slice 4: TR-V2-02 – Phase 2 Hidden Shock Layer

**Status:** Not started
**Dependency:** Slice 3 (direct priors implemented)
**Mathematical target:**
```
u_m(t) = rho_m u_m(t-1) + xi_m(t)
logit h_r(t) = ... + sum_m Lambda[r,m] u_m(t)
```

#### 4.1 Files to Add

**`src/epigraph_ph/phase3/frontier/hidden_shocks.py`**

Purpose: Read Phase 2 low-rank hidden driver output, represent hidden temporal modes at quarter scale, add hazard shock terms with separate coefficients `Lambda`.

```python
# hidden_shocks.py — Phase 2 hidden shared shock channels

from __future__ import annotations
from typing import Any
import numpy as np

from epigraph_ph.phase3.frontier.registry import TRANSITION_NAMES

def load_phase2_hidden_drivers(
    *,
    archive_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load Phase 2 low-rank hidden structure output.
    
    Phase 2 decomposed temporal structure into:
    - sparse direct edges (handled by phase2_hazard_priors.py)
    - low-rank hidden shared structure (handled here)
    
    Returns:
        hidden_modes: dict mapping mode_id to (time_steps,) array
        mode_metadata: AR coefficients (rho_m), innovation variance, support scores
    """
    pass

def build_hidden_shock_model(
    *,
    hidden_modes: dict[str, np.ndarray],
    train_quarters: list[str],
    holdout_quarters: list[str],
    transitions: tuple[str, ...] = TRANSITION_NAMES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Fit the hidden shock layer.
    
    For each hidden mode m:
    u_m(t) = rho_m u_m(t-1) + xi_m(t)
    
    For each transition r:
    Learn Lambda[r,m] via regularized regression on train hazards,
    after removing the direct Phase 2 prior contribution.
    
    The key constraint: Lambda coefficients must be identifiable
    from the Gamma coefficients. This is achieved by:
    1. Fitting Gamma first (direct priors only)
    2. Computing residuals = train_hazards - Gamma_contribution
    3. Fitting Lambda on residuals only
    
    Returns:
        shock_models: dict with AR parameters (rho_m) per mode
        lambda_matrix: dict mapping (transition, mode) -> coefficient
    """
    pass

def project_hidden_shocks(
    *,
    shock_models: dict[str, Any],
    lambda_matrix: dict[tuple[str, str], float],
    last_hidden_states: dict[str, float],
    n_forecast_steps: int,
) -> dict[str, np.ndarray]:
    """
    Project hidden shocks forward into forecast horizon.
    
    For each mode m, project:
        u_m(t+k) = rho_m^k u_m(t)  (mean projection, xi=0)
    
    Then compute shock contribution to each transition:
        shock_r(t+k) = sum_m Lambda[r,m] u_m(t+k)
    
    Returns:
        projected_shocks: dict mapping transition to (n_forecast_steps,) array
    """
    pass
```

#### 4.2 Engine Changes

In `transition_engine.py`, add TR-V2-02 experiment handler that:
1. Loads TR-V2-00 baseline
2. Applies TR-V2-01 direct priors (or skips for "hidden-only" ablation)
3. Fits hidden shock layer on train residuals
4. Runs holdout simulation with hidden shock contributions
5. Outputs comparison against TR-V2-00 and TR-V2-01

#### 4.3 Acceptance Criteria

1. Direct and hidden Phase 2 structure are logged separately in artifacts.
2. The model can run with:
   - direct only
   - hidden only
   - both
3. Hidden shock magnitude can be ablated independently from direct edge magnitude.
4. `project_hidden_shocks()` produces reasonable forecasts:
   - If `rho_m < 1`, shocks decay toward zero (stationary)
   - If `rho_m ≈ 1`, shocks persist (unit root behavior)
   - No blow-up in long-horizon projection

---

### Slice 5: TR-V2-03 – Ablation Suite

**Status:** Not started
**Dependency:** Slices 2–4

#### 5.1 Files to Add

**`src/epigraph_ph/phase3/frontier/ablation.py`**

Purpose: Run controlled comparisons to prevent over-claiming.

```python
# ablation.py — Transition frontier ablation suite

from __future__ import annotations
from typing import Any
import numpy as np

METRICS = [
    "mean_absolute_error",
    "sMAPE",
    "diagnosis_flow_mean_absolute_error",
    "peak_alive_on_art_absolute_error",
]

def run_ablation_suite(
    *,
    baseline: dict[str, Any],        # TR-V2-00 lock record
    context: TransitionResearchContext,
    holdout_quarters: list[str],
) -> dict[str, Any]:
    """
    Run the full ablation suite.
    
    Configurations tested:
    1. TR-V2-00: baseline only (no Phase 2)
    2. TR-V2-01: direct Phase 2 priors only
    3. TR-V2-02-hidden: hidden shocks only (no direct priors)
    4. TR-V2-02-both: direct + hidden
    5. TR-V2-02-both-gated: direct + hidden + peak gating (PEAK-01F)
    6. TR-V2-02-both-nogating: direct + hidden, no peak gating
    
    For each configuration, compute all METRICS on holdout quarters.
    Then compare each against TR-V2-00.
    
    Returns:
        ablation_summary: dict with per-configuration metrics,
                         pairwise deltas vs baseline, and
                         keep/revert recommendation per config.
    """
    pass

def decide_keep_or_revert(
    *,
    config_name: str,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    improvement_threshold: float = 0.01,  # 1% improvement required
) -> dict[str, Any]:
    """
    Apply keep-or-revert rules per the plan:
    - Keeps if improves on at least one metric without materially
      regressing any other metric (>1% regression = material)
    - Reverts if any metric regresses materially
    - Flags if improvement is < threshold (no material gain)
    """
    pass
```

#### 5.2 Acceptance Criteria

1. Every new TR-V2-* experiment records parent lineage.
2. Every new TR-V2-* experiment is directly compared against TR-V2-00.
3. No new branch is considered "kept" unless it beats TR-V2-00 on the agreed metric rule.
4. `transition_frontier_ablation_summary.json` is emitted with complete results.

---

### Slice 6: TR-V2-05 – Trajectory Understanding (GRasp Test)

**Status:** Not started (new experiment added to plan)
**Dependency:** Slices 2–4 (need Phase 2 structure implemented)

**Rationale:**
Inspired by the GRasp benchmark concept: if a model truly "understands" the internal dynamics of a system, it should be able to predict turning points (peaks/troughs) from first principles, without external detectors.

For our model: can we remove PEAK-01F's explicit peak detector and still predict peaks just by projecting the model's own dynamics forward?

#### 6.1 Mathematical Approach

Given the model with Phase 2 structure:
```
logit h_r(t) = alpha_r(t) + base_r(t) + Gamma_contrib(t) + Lambda_contrib(t)
x_{t+1} = F(x_t, h_t)
```

**Peak detection from projection:**
A peak in diagnosis flow `new_diagnosed_cases_period` (which equals `h_U→D(t) * U_t`) occurs when:
```
d/dt [h_U→D(t) * U_t] = 0  AND  d²/dt² [h_U→D(t) * U_t] < 0
```

For discrete quarterly time:
```
peak_detected_at_t if: diagnosis_flow(t) > diagnosis_flow(t-1) AND diagnosis_flow(t) > diagnosis_flow(t+1)
```

**Projecting forward:**
1. Run the model forward from the last training state (no observations injected)
2. Compute `diagnosis_flow(t) = h_U→D(t) * U_t` at each projected step
3. Find t* where `diagnosis_flow(t*)` is maximized
4. Compare t* with the actual observed peak quarter

#### 6.2 Files to Add

**`src/epigraph_ph/phase3/frontier/trajectory.py`**

```python
# trajectory.py — Forward trajectory projection and peak prediction

from __future__ import annotations
from typing import Any
import numpy as np

def project_trajectory(
    *,
    initial_state: dict[str, float],  # Last training state (U,D,A,V,L)
    initial_hazards: dict[str, float],  # Last training hazards
    helper_models: dict[str, dict[str, float]],  # intercept, persistence, helper_scale
    phase2_priors: dict[str, Any] | None,  # Gamma matrix (TR-V2-01)
    hidden_shocks: dict[str, Any] | None,  # Lambda, rho, u (TR-V2-02)
    n_steps: int = 8,  # Forecast horizon in quarters
    eps: float = 1e-8,
) -> dict[str, Any]:
    """
    Project the model forward for n_steps without external observations.
    
    The model uses its own internal dynamics:
    - Base hazard trends (from helper models)
    - Phase 2 direct prior contributions (if provided)
    - Phase 2 hidden shock projections (if provided)
    - State evolution F(x_t, h_t)
    
    Returns:
        state_trajectory: list of state dicts per quarter
        hazard_trajectory: list of hazard dicts per quarter
        diagnosis_flow: array of h_U→D(t) * U_t values
    """
    pass

def find_projected_peaks(
    *,
    diagnosis_flow: np.ndarray,
    quarters: list[str],
    min_prominence: float = 0.01,  # Minimum peak prominence to count
) -> list[dict[str, Any]]:
    """
    Find peaks in the projected diagnosis flow.
    
    A peak is a local maximum with prominence > threshold.
    
    Returns:
        peaks: list of dicts with quarter, flow_value, prominence, rank
    """
    pass

def compare_peaks(
    *,
    predicted_peaks: list[dict[str, Any]],
    observed_peaks: list[dict[str, Any]],
    tolerance_quarters: int = 1,  # Within ±1 quarter counts as correct
) -> dict[str, Any]:
    """
    Compare predicted vs observed peaks.
    
    Metrics:
    - Timing accuracy: quarters between predicted and actual peak
    - Magnitude accuracy: |predicted_flow - actual_flow| / actual_flow
    - Detection rate: fraction of observed peaks that were predicted
    - False positive rate: fraction of predicted peaks that don't match observations
    """
    pass
```

#### 6.3 Experiment Configurations for TR-V2-05

| Config | Phase 2 Direct | Phase 2 Hidden | Peak Gating | Expected |
|---|---|---|---|---|
| TR-V2-05-baseline | OFF | OFF | OFF | Pure model extrapolation (likely poor) |
| TR-V2-05-direct | ON | OFF | OFF | Direct edges improve timing |
| TR-V2-05-hidden | OFF | ON | OFF | Hidden shocks capture latent momentum |
| TR-V2-05-both | ON | ON | OFF | Best chance of accurate peak prediction |
| TR-V2-05-full | ON | ON | ON | Compare with detector-aided version |

#### 6.4 Acceptance Criteria

1. The model produces a smooth trajectory without oscillation or divergence in the first 8 forecast quarters.
2. At least one peak is detected in the projected trajectory (otherwise the model is trivially flat).
3. The predicted peak quarter is within ±1 quarter of the observed peak quarter in at least one configuration.
4. If the model cannot predict peaks without the detector, this is documented as a limitation — it means the detector is necessary and the model alone doesn't have sufficient "trajectory understanding."

---

### Slice 7: INC-V2 Incidence Lineage

**Status:** Not started
**Dependency:** Slice 2 (baseline lock)

#### 7.1 Files to Modify

**`src/epigraph_ph/phase3/incidence/modeling.py`**
- Add `parent_transition_branch_id` field to all run metadata
- Emission explicit lineage in `incidence_flow_summary.json`

**`src/epigraph_ph/phase3/incidence/registry.py`**
- Add `INC-V2-00-lineage-explicitness-audit` and `INC-V2-01-phase2-compatible-incidence-extension`

#### 7.2 Acceptance Criteria

1. Every incidence artifact records the parent transition branch.
2. Incidence branch success is not conflated with independent forecast success.
3. Separation of forecast inheritance quality vs incidence compatibility quality.

---

### Slice 8: Broad Reintegration Readiness

**Status:** Readiness only
**Dependency:** All preceding slices

#### 8.1 Purpose

Define interfaces for future rebuild of `rescue_core` around winning transition logic. Do NOT rebuild yet.

#### 8.2 Interface Definition

Create `src/epigraph_ph/phase3/transition_interface.py`:

```python
# transition_interface.py — Interface between frontier and broad model

from __future__ import annotations
from typing import Any, Protocol

class TransitionFrontierProvider(Protocol):
    """Interface that any transition frontier must implement."""
    
    def get_transition_hazards(
        self,
        quarter: str,
        state_vector: dict[str, float],
    ) -> dict[str, float]:
        """Return hazards for all 5 transitions given state."""
        ...
    
    def get_frontier_metadata(self) -> dict[str, Any]:
        """Return experiment ID, lineage, and configuration."""
        ...
    
    def project_forward(
        self,
        initial_state: dict[str, float],
        n_steps: int,
    ) -> dict[str, Any]:
        """Project trajectory forward (for trajectory understanding test)."""
        ...

class BroadModelConsumer(Protocol):
    """Interface that the broad rescue_core model must satisfy to consume frontier."""
    
    def accept_transition_hazards(
        self,
        hazards: dict[str, float],
        source: str,
    ) -> None:
        """Accept external transition hazards."""
        ...
    
    def run_broad_simulation(
        self,
        hazards_function: TransitionFrontierProvider,
    ) -> dict[str, Any]:
        """Run broad model using frontier hazards."""
        ...
```

#### 8.3 Output

- `transition_interface_spec.json`: Interface definition + compatibility matrix
- `broadening_candidates.json`: List of frontier components that are vs aren't broadening candidates

---

## 4. Pre-Existing Bugs to Fix

Before implementing the above, these pre-existing issues must be addressed:

### 4.1 `transition_engine.py` Path Bug

`_discover_latest_transition_experiment()` (line 56) searches:
```python
experiment_dir = run_dir / "transition_research" / experiment_id
```
Should search:
```python
experiment_dir = run_dir / "frontier" / experiment_id
```

### 4.2 Test: `identifiability_class` Assertion Drift

`test_inc_00b_and_inc_00c_live_runs` has an assertion that no longer matches actual values. The test expectation needs to be updated to match the current output.

### 4.3 Test: Windows Path in Sandbox

`test_inc_01d_and_inc_01b_live_runs` fails with `FileNotFoundError: 'D:/EpiGraph_PH/artifacts/runs'`. The sandbox environment needs the correct path or the test needs to skip on non-Windows.

---

## 5. Implementation Order

The plan specifies this order:

| Order | Slice | Experiment | Estimated Effort |
|---|---|---|---|
| 1 | Pre-bug fixes | 3 bug fixes | Low |
| 2 | Slice 2 | TR-V2-00 baseline lock | Medium |
| 3 | Slice 3 | TR-V2-01 Phase 2 direct priors | Medium |
| 4 | Slice 4 | TR-V2-02 Hidden shocks | Medium |
| 5 | Slice 5 | TR-V2-03 Ablation suite | Low-Medium |
| 6 | Slice 6 | TR-V2-05 Trajectory understanding | Medium |
| 7 | Slice 7 | INC-V2-00, INC-V2-01 | Medium |
| 8 | Slice 8 | Reintegration readiness (docs only) | Low |

---

## 6. Required Artifacts

### 6.1 Every TR-V2-* Experiment Should Emit

- `baseline_comparison.json` — vs TR-V2-00
- `evaluation.json` — standard evaluation
- `frontier_lineage.json` — explicit parent chain
- `transition_hazard_summary.json` — per-transition hazard values

### 6.2 Phase 2-Integrated Experiments Additionally Emit

- `phase2_direct_hazard_prior_summary.json` — TR-V2-01
- `phase2_hidden_shock_summary.json` — TR-V2-02
- `transition_frontier_ablation_summary.json` — TR-V2-03
- `trajectory_projection.json` — TR-V2-05

### 6.3 Required Plots

- `forecast_vs_baseline.png` — overlaid forecast trajectories
- `hazard_comparison.png` — per-transition hazard comparison
- `phase2_contribution.png` — direct vs hidden contribution
- `trajectory_peak_prediction.png` — predicted vs actual peaks (TR-V2-05)

---

## 7. Research References

### 7.1 Directly Relevant Papers

| # | Paper | Relevance |
|---|---|---|
| 1 | Eletti, Marra, Radice (2023). **Spline-Based Multi-State Models for Analyzing Disease Progression.** arXiv:2312.05345v4. | Directly applicable to `logit h_r(t)` transition hazards. Handles non-homogeneous Markov models with spline-based transition intensities and multiple observation schemes. |
| 2 | Hohberg & Groll (2020). **A flexible adaptive lasso Cox frailty model based on the full likelihood.** arXiv:2003.14118v1. | Template for penalized Cox with time-varying covariates, P-spline baselines, group lasso. Matches TR-V2-01 needs. |
| 3 | Gross, Meshkat, Shiu (2017). **Identifiability of linear compartmental models: the singular locus.** arXiv:1709.10013v3. | Framework for parameter identifiability. Critical for incidence identifiability problem. |
| 4 | Ryalen, Stensrud, Røysland (2017). **Transforming cumulative hazard estimates.** arXiv:1710.07422v4. | Addresses built-in selection effect in hazard modeling. Important for scientific honesty. |
| 5 | Narci et al. (2020). **Inference for partially observed epidemic dynamics guided by Kalman filtering techniques.** arXiv:2007.08974v3. | Relevant for unobserved U/D/A/V/L state inference. |
| 6 | Birrell, De Angelis, Presanis (2017). **Evidence synthesis for stochastic epidemic models.** arXiv:1706.02624v1. | Multi-source evidence synthesis. Directly parallel to our data integration challenge. |
| 7 | Foygel & Drton (2010). **Exact block-wise optimization in group lasso and sparse group lasso.** arXiv:1010.3320v2. | Optimization algorithm for group lasso. Useful for structured prior selection in TR-V2-01. |

### 7.2 Foundational References (Not on arXiv)

- Cox, D.R. (1972). **Regression models and life-tables.** J. Royal Statistical Society B.
- Anderson, P.K., Borgan, Ø., Gill, R.D., Keiding, N. (1993). **Statistical Models Based on Counting Processes.** Springer.
- Jackson, C.H. (2011). **Multi-state models for panel data: the msm package for R.** Journal of Statistical Software.

---

## 8. Decision Rules

### 8.1 Keep-or-Revert

Keep a new transition-frontier mutation only if:
1. It preserves artifact completeness,
2. It improves or at least does not materially regress the winning branch metrics,
3. It does not silently blur the distinction between direct and hidden Phase 2 structure,
4. It does not rely on unsupported broadening assumptions,
5. It keeps lineage and evaluation regime explicit.

### 8.2 Scientific Posture

No new branch should be described as:
- Broad winner
- Final mechanistic truth
- Independent incidence winner

Unless it actually satisfies those broader evaluation criteria.

### 8.3 Trajectory Understanding Gate

For TR-V2-05:
- If the model predicts peaks without a detector: the trajectory understanding test passes
- If it needs the detector: document this as a finding, not a failure
- The finding "detector is necessary" is itself valuable scientific information

---

## 9. File Dependency Graph

```
baseline_lock.py
    └── Uses: transition_engine (AGE-01B reproduction logic)
    └── Uses: registry.py (experiment definitions)
    └── Outputs: baseline_lock.json

phase2_hazard_priors.py
    └── Uses: Phase 15 latent block outputs
    └── Uses: Phase 2 sparse edge outputs
    └── Outputs: gamma_prior matrix, prior summary

hidden_shocks.py
    └── Uses: phase2_hazard_priors residuals
    └── Uses: Phase 2 hidden driver outputs
    └── Outputs: lambda_matrix, shock projections

ablation.py
    └── Uses: baseline_lock (TR-V2-00 record)
    └── Uses: phase2_hazard_priors (TR-V2-01 config)
    └── Uses: hidden_shocks (TR-V2-02 configs)
    └── Outputs: ablation summary

trajectory.py
    └── Uses: baseline_lock + all Phase 2 configs
    └── Uses: _simulate_holdout helpers from transition_engine
    └── Outputs: predicted peaks, accuracy metrics
```

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Phase 2 data doesn't load/runs in the sandbox | Blocks Slices 3–5 entirely | Verify Phase 15/Phase 2 artifact availability before starting Slice 3 |
| AGE-01B baseline can't be exactly reproduced | Blocks all TR-V2 work (Slice 2 gate) | First priority: reproduce AGE-01B and verify within tolerance |
| Direct priors destabilize the model | Over-correction in hazards | Clip prior contributions, use empirical Bayes scaling (Hohberg & Groll template) |
| Hidden shocks don't converge | Lambda estimation fails | Use regularization, ensure direct priors are fit first |
| Trajectory test produces flat/noisy projections | Model doesn't "understand" dynamics | This is a valid finding — document it |

---

## 11. Success Criteria for the Full TR-V2 Program

The TR-V2 program succeeds if:

1. TR-V2-00 locks the AGE-01B baseline within machine tolerance.
2. At least one TR-V2-0* configuration beats TR-V2-00 on the agreed metric rule (MAE improvement ≥ 1% without material regression elsewhere).
3. The ablation suite clearly separates direct vs hidden Phase 2 contributions.
4. The trajectory understanding test produces a clear result (pass or documented limitation).
5. Incidence lineage is explicitly recorded for all INC-V2 runs.
6. The transition interface is defined for future broad reintegration.

---

*End of plan.*
