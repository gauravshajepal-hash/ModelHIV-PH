from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from epigraph_ph.runtime import ROOT_DIR, read_json, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .integrated_autoresearch import (
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    STATE_NAMES,
    TOTAL_METRIC,
    CandidateConfig,
    FitResult,
    IntegratedSnapshot,
    TRANSITION_NAMES,
    _build_snapshot,
    _cloglog_inverse,
    _compute_window_metrics,
    _fit_candidate,
    _fit_residuals,
    _historical_mask,
    _safe_exp,
    _sigmoid,
)
from .numeric_policy import numerical_guard_entry


BRANCH_REFERENCE_KEYS: dict[str, tuple[str, str]] = {
    "diagnosis": ("delay", "markov"),
    "care": ("hazard", "semi_markov"),
    "combined": ("delay", "semi_markov"),
}
BRANCH_LABELS: dict[str, str] = {
    "diagnosis": "Diagnosis Delay Branch",
    "care": "Semi-Markov Care Branch",
    "combined": "Combined Delay + Semi-Markov Branch",
}
CARE_TRANSITIONS: tuple[str, ...] = ("D_to_A", "A_to_V", "A_to_L", "L_to_A")
SPEC_ROW_ORDER: tuple[str, ...] = (
    "incidence",
    "diagnosis_hazard",
    "diagnosis_delay",
    "downstream_semi_markov",
    "a_competing_risks",
    "hidden_structure",
    "phase2_insertion",
    "observation_layer",
)
_CANDIDATE_PATTERN = re.compile(
    r"^diag-(?P<diagnosis>hazard|delay)-care-(?P<care>markov|semi_markov)-h(?P<hidden>\d+)-obs(?P<obs>on|off)$"
)


@dataclass(slots=True)
class BranchAuditResult:
    branch_name: str
    current_fit: FitResult
    current_bic: float
    current_age_param_count: int
    strict_width: int
    strict_bic: float
    strict_metrics: dict[str, float]
    strict_simulation: dict[str, Any]
    strict_age_param_count: int
    simultaneous_competing: bool
    width_trace: list[dict[str, Any]]
    prediction_gap: dict[str, dict[str, float]]


def _latest_integrated_experiment_dir() -> Path | None:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "transition_research" / "PHASE3-V2-INT-explicit-incidence-autoresearch"
        frontier_path = experiment_dir / "stage1_calibration_frontier.json"
        if frontier_path.exists():
            candidates.append((frontier_path.stat().st_mtime, experiment_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item[0]))[1]


def _parse_candidate_id(candidate_id: str) -> CandidateConfig | None:
    match = _CANDIDATE_PATTERN.match(str(candidate_id))
    if match is None:
        return None
    return CandidateConfig(
        candidate_id=str(candidate_id),
        diagnosis_family=str(match.group("diagnosis")),
        care_family=str(match.group("care")),
        hidden_rank=int(match.group("hidden")),
        use_observation_covariates=str(match.group("obs")) == "on",
    )


def _score_tuple_from_row(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("historical_primary_loss", float("inf"))),
        float(row.get("historical_diag_flow_loss", float("inf"))),
        float(row.get("historical_total_loss", float("inf"))),
        float(row.get("historical_secondary_loss", float("inf"))),
    )


def _select_reference_configs() -> dict[str, CandidateConfig]:
    experiment_dir = _latest_integrated_experiment_dir()
    if experiment_dir is None:
        raise FileNotFoundError("No PHASE3-V2-INT stage1 frontier artifact was found under artifacts/runs")
    frontier_rows = list(read_json(experiment_dir / "stage1_calibration_frontier.json", default=[]))
    best_rows: dict[str, dict[str, Any]] = {}
    for row in frontier_rows:
        candidate_id = str(row.get("candidate_id") or "")
        config = _parse_candidate_id(candidate_id)
        if config is None:
            continue
        for branch_name, key in BRANCH_REFERENCE_KEYS.items():
            if (config.diagnosis_family, config.care_family) != key:
                continue
            current_best = best_rows.get(branch_name)
            if current_best is None or _score_tuple_from_row(row) < _score_tuple_from_row(current_best):
                best_rows[branch_name] = dict(row)
    if set(best_rows) != set(BRANCH_REFERENCE_KEYS):
        missing = sorted(set(BRANCH_REFERENCE_KEYS) - set(best_rows))
        raise ValueError(f"Missing PHASE3-V2-INT branch candidates in stage1 frontier: {missing}")
    selected: dict[str, CandidateConfig] = {}
    for branch_name, row in best_rows.items():
        config = _parse_candidate_id(str(row.get("candidate_id") or ""))
        if config is None:
            raise ValueError(f"Could not parse candidate id for {branch_name}: {row}")
        selected[branch_name] = config
    return selected


def _reference_fit(snapshot: IntegratedSnapshot, config: CandidateConfig) -> FitResult:
    fit = _fit_candidate(snapshot, config)
    if not fit.success:
        raise RuntimeError(f"Reference fit failed for {config.candidate_id}: {fit.message}")
    historical = dict(fit.window_metrics.get("historical") or {})
    if float(historical.get("non_finite_count", 0.0)) > 0.0 or float(historical.get("population_violation_count", 0.0)) > 0.0:
        raise RuntimeError(f"Reference fit is numerically invalid for {config.candidate_id}")
    return fit


def _age_shift(values: np.ndarray) -> np.ndarray:
    shifted = np.zeros_like(values, dtype=np.float64)
    if values.size == 0:
        return shifted
    if values.size > 1:
        shifted[1:] = values[:-1]
    shifted[-1] += float(values[-1])
    return shifted


def _current_age_profile(fit: FitResult, transition: str, age_count: int) -> np.ndarray:
    age_basis = np.log1p(np.arange(age_count, dtype=np.float64))
    transition_slice = fit.param_slices[f"transition::{transition}"]
    params = np.asarray(fit.param_vector[transition_slice], dtype=np.float64)
    include_age = (transition == "U_to_D" and fit.config.diagnosis_family == "delay") or (
        transition != "U_to_D" and fit.config.care_family == "semi_markov"
    )
    if not include_age or params.size < 2:
        return np.zeros((age_count,), dtype=np.float64)
    return float(params[1]) * age_basis


def _age_param_count(fit: FitResult, branch_name: str) -> int:
    if branch_name == "diagnosis":
        return 1 if fit.config.diagnosis_family == "delay" else 0
    if branch_name == "care":
        return 4 if fit.config.care_family == "semi_markov" else 0
    if branch_name == "combined":
        count = 0
        if fit.config.diagnosis_family == "delay":
            count += 1
        if fit.config.care_family == "semi_markov":
            count += 4
        return count
    raise KeyError(branch_name)


def _age_basis_dct(age_count: int, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros((age_count, 0), dtype=np.float64)
    age_index = np.arange(age_count, dtype=np.float64)
    basis = np.zeros((age_count, width), dtype=np.float64)
    for col in range(width):
        frequency = float(col + 1)
        basis[:, col] = np.cos(math.pi * frequency * (age_index + 0.5) / float(age_count))
    norms = np.linalg.norm(basis, axis=0)
    norms = np.where(norms > np.finfo(np.float64).eps, norms, 1.0)
    return basis / norms[None, :]


def _coverage_count(snapshot: IntegratedSnapshot, metric_name: str) -> int:
    return int(np.sum(np.asarray(snapshot.metric_masks[metric_name], dtype=bool)))


def _basis_width_limit(snapshot: IntegratedSnapshot, branch_name: str) -> int:
    if branch_name == "diagnosis":
        informative = _coverage_count(snapshot, "new_diagnosed_cases_period")
    elif branch_name == "care":
        informative = min(
            _coverage_count(snapshot, "alive_on_art"),
            _coverage_count(snapshot, "tested_for_viral_load"),
            _coverage_count(snapshot, "virally_suppressed"),
        )
    elif branch_name == "combined":
        informative = min(
            _basis_width_limit(snapshot, "diagnosis"),
            _basis_width_limit(snapshot, "care"),
        )
    else:
        raise KeyError(branch_name)
    informative = max(int(informative), 1)
    return min(int(len(snapshot.model_quarters) - 1), informative)


def _initial_state_arrays(fit: FitResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    age_count = len(fit.simulation["states"])
    eps = float(np.finfo(np.float64).eps)
    init_u = float(_safe_exp(fit.param_vector[fit.param_slices["init_u_log"]])[0])
    init_art_share = float(_sigmoid(fit.param_vector[fit.param_slices["init_art_share"]])[0])
    init_v_share = float(_sigmoid(fit.param_vector[fit.param_slices["init_v_share"]])[0])
    diagnosed_stock = max(float(fit.simulation["predictions"]["diagnosed_plhiv"][0]), 0.0)
    init_art_stock = diagnosed_stock * init_art_share
    diag_gap = max(diagnosed_stock - init_art_stock, 0.0)
    init_l_share = float(_sigmoid(fit.param_vector[fit.param_slices["init_l_share"]])[0]) if diag_gap > eps else 0.0

    u_age = np.zeros((age_count,), dtype=np.float64)
    d_age = np.zeros((age_count,), dtype=np.float64)
    a_age = np.zeros((age_count,), dtype=np.float64)
    l_age = np.zeros((age_count,), dtype=np.float64)

    u_age[-1] = max(init_u, 0.0)
    v_stock = init_art_stock * init_v_share
    a_age[-1] = max(init_art_stock - v_stock, 0.0)
    l_age[-1] = diag_gap * init_l_share
    d_age[-1] = max(diag_gap - l_age[-1], 0.0)
    return u_age, d_age, a_age, l_age, float(v_stock)


def _branch_current_profiles(fit: FitResult, branch_name: str) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    age_count = len(fit.simulation["states"])
    diagnosis_profile = None
    if branch_name in {"diagnosis", "combined"}:
        diagnosis_profile = _current_age_profile(fit, "U_to_D", age_count)
    care_profiles: dict[str, np.ndarray] = {}
    if branch_name in {"care", "combined"}:
        for transition in CARE_TRANSITIONS:
            care_profiles[transition] = _current_age_profile(fit, transition, age_count)
    return diagnosis_profile, care_profiles


def _simulate_conditional_gap(
    snapshot: IntegratedSnapshot,
    fit: FitResult,
    *,
    diagnosis_profile: np.ndarray | None,
    care_profiles: dict[str, np.ndarray],
    simultaneous_competing: bool,
) -> dict[str, Any]:
    eps = float(np.finfo(np.float64).eps)
    quarter_count = len(snapshot.model_quarters)
    age_count = quarter_count
    u_age, d_age, a_age, l_age, v_stock = _initial_state_arrays(fit)
    states = np.zeros((quarter_count, len(STATE_NAMES)), dtype=np.float64)
    flow_map = {transition: np.zeros((quarter_count,), dtype=np.float64) for transition in TRANSITION_NAMES}
    hazard_map = {transition: np.zeros((quarter_count,), dtype=np.float64) for transition in TRANSITION_NAMES}
    states[0, STATE_NAMES.index("U")] = float(np.sum(u_age))
    states[0, STATE_NAMES.index("D")] = float(np.sum(d_age))
    states[0, STATE_NAMES.index("A")] = float(np.sum(a_age))
    states[0, STATE_NAMES.index("V")] = float(v_stock)
    states[0, STATE_NAMES.index("L")] = float(np.sum(l_age))

    base_eta = {key: np.asarray(value, dtype=np.float64) for key, value in dict(fit.simulation["eta"]).items()}
    incidence_flow = np.asarray(fit.simulation["incidence_flow"], dtype=np.float64)
    pi_test = np.asarray(_sigmoid(base_eta["tested_for_viral_load"]), dtype=np.float64)
    pi_doc = np.asarray(_sigmoid(base_eta["virally_suppressed"]), dtype=np.float64)
    current_profiles = {
        transition: _current_age_profile(fit, transition, age_count)
        for transition in TRANSITION_NAMES
    }
    current_hazards = {transition: np.asarray(fit.simulation["hazards"][transition], dtype=np.float64) for transition in TRANSITION_NAMES}

    for idx in range(1, quarter_count):
        u_prev = float(np.sum(u_age))
        d_prev = float(np.sum(d_age))
        a_prev = float(np.sum(a_age))
        l_prev = float(np.sum(l_age))

        if diagnosis_profile is not None:
            diagnosis_prob = np.asarray(_sigmoid(base_eta["U_to_D"][idx] + diagnosis_profile), dtype=np.float64)
            diagnosed_by_age = np.minimum(u_age, diagnosis_prob * u_age)
        elif fit.config.diagnosis_family == "delay":
            diagnosis_prob = np.asarray(_sigmoid(base_eta["U_to_D"][idx] + current_profiles["U_to_D"]), dtype=np.float64)
            diagnosed_by_age = np.minimum(u_age, diagnosis_prob * u_age)
        else:
            diagnosed_by_age = np.minimum(u_age, current_hazards["U_to_D"][idx] * u_age)

        u_to_d = float(np.sum(diagnosed_by_age))
        u_age_next = _age_shift(np.maximum(u_age - diagnosed_by_age, 0.0))
        u_age_next[0] += float(incidence_flow[idx])

        def _care_hazard(transition: str) -> np.ndarray:
            if transition in care_profiles:
                return np.asarray(_cloglog_inverse(base_eta[transition][idx] + care_profiles[transition]), dtype=np.float64)
            if fit.config.care_family == "semi_markov" and transition != "U_to_D":
                return np.asarray(_cloglog_inverse(base_eta[transition][idx] + current_profiles[transition]), dtype=np.float64)
            return np.full((age_count,), float(current_hazards[transition][idx]), dtype=np.float64)

        d_to_a_hazard = _care_hazard("D_to_A")
        d_to_a_by_age = np.minimum(d_age, d_to_a_hazard * d_age)
        d_to_a = float(np.sum(d_to_a_by_age))
        d_age_next = _age_shift(np.maximum(d_age - d_to_a_by_age, 0.0))
        d_age_next[0] += u_to_d

        a_to_v_hazard = _care_hazard("A_to_V")
        a_to_l_hazard = _care_hazard("A_to_L")
        if simultaneous_competing and (care_profiles or fit.config.care_family == "semi_markov"):
            if "A_to_V" in care_profiles:
                eta_v = base_eta["A_to_V"][idx] + care_profiles["A_to_V"]
            elif fit.config.care_family == "semi_markov":
                eta_v = base_eta["A_to_V"][idx] + current_profiles["A_to_V"]
            else:
                eta_v_scalar = float(np.log(-np.log(np.maximum(1.0 - np.minimum(current_hazards["A_to_V"][idx], 1.0 - eps), eps))))
                eta_v = np.full((age_count,), eta_v_scalar, dtype=np.float64)

            if "A_to_L" in care_profiles:
                eta_l = base_eta["A_to_L"][idx] + care_profiles["A_to_L"]
            elif fit.config.care_family == "semi_markov":
                eta_l = base_eta["A_to_L"][idx] + current_profiles["A_to_L"]
            else:
                eta_l_scalar = float(np.log(-np.log(np.maximum(1.0 - np.minimum(current_hazards["A_to_L"][idx], 1.0 - eps), eps))))
                eta_l = np.full((age_count,), eta_l_scalar, dtype=np.float64)

            lambda_v = np.asarray(_safe_exp(eta_v), dtype=np.float64)
            lambda_l = np.asarray(_safe_exp(eta_l), dtype=np.float64)
            lambda_total = lambda_v + lambda_l
            event_prob = 1.0 - np.exp(-lambda_total)
            share_v = np.divide(lambda_v, np.maximum(lambda_total, eps), out=np.zeros_like(lambda_v), where=lambda_total > eps)
            share_l = 1.0 - share_v
            a_to_v_by_age = np.minimum(a_age, event_prob * share_v * a_age)
            a_to_l_by_age = np.minimum(np.maximum(a_age - a_to_v_by_age, 0.0), event_prob * share_l * a_age)
            a_survivors = np.maximum(a_age - a_to_v_by_age - a_to_l_by_age, 0.0)
        else:
            a_to_v_by_age = np.minimum(a_age, a_to_v_hazard * a_age)
            a_after_v = np.maximum(a_age - a_to_v_by_age, 0.0)
            a_to_l_by_age = np.minimum(a_after_v, a_to_l_hazard * a_age)
            a_survivors = np.maximum(a_after_v - a_to_l_by_age, 0.0)

        a_to_v = float(np.sum(a_to_v_by_age))
        a_to_l = float(np.sum(a_to_l_by_age))

        l_to_a_hazard = _care_hazard("L_to_A")
        l_to_a_by_age = np.minimum(l_age, l_to_a_hazard * l_age)
        l_to_a = float(np.sum(l_to_a_by_age))
        l_survivors = np.maximum(l_age - l_to_a_by_age, 0.0)

        a_age_next = _age_shift(a_survivors)
        a_age_next[0] += d_to_a + l_to_a
        l_age_next = _age_shift(l_survivors)
        l_age_next[0] += a_to_l
        v_stock = max(v_stock + a_to_v, 0.0)

        u_age = u_age_next
        d_age = d_age_next
        a_age = a_age_next
        l_age = l_age_next

        states[idx, STATE_NAMES.index("U")] = max(float(np.sum(u_age)), 0.0)
        states[idx, STATE_NAMES.index("D")] = max(float(np.sum(d_age)), 0.0)
        states[idx, STATE_NAMES.index("A")] = max(float(np.sum(a_age)), 0.0)
        states[idx, STATE_NAMES.index("V")] = max(float(v_stock), 0.0)
        states[idx, STATE_NAMES.index("L")] = max(float(np.sum(l_age)), 0.0)
        flow_map["U_to_D"][idx] = u_to_d
        flow_map["D_to_A"][idx] = d_to_a
        flow_map["A_to_V"][idx] = a_to_v
        flow_map["A_to_L"][idx] = a_to_l
        flow_map["L_to_A"][idx] = l_to_a
        hazard_map["U_to_D"][idx] = float(u_to_d / max(u_prev, eps)) if u_prev > 0.0 else 0.0
        hazard_map["D_to_A"][idx] = float(d_to_a / max(d_prev, eps)) if d_prev > 0.0 else 0.0
        hazard_map["A_to_V"][idx] = float(a_to_v / max(a_prev, eps)) if a_prev > 0.0 else 0.0
        hazard_map["A_to_L"][idx] = float(a_to_l / max(a_prev, eps)) if a_prev > 0.0 else 0.0
        hazard_map["L_to_A"][idx] = float(l_to_a / max(l_prev, eps)) if l_prev > 0.0 else 0.0

    diagnosed = states[:, STATE_NAMES.index("D")] + states[:, STATE_NAMES.index("A")] + states[:, STATE_NAMES.index("V")] + states[:, STATE_NAMES.index("L")]
    art = states[:, STATE_NAMES.index("A")] + states[:, STATE_NAMES.index("V")]
    total = np.sum(states, axis=1)
    predictions = {
        "diagnosed_plhiv": diagnosed,
        "alive_on_art": art,
        "new_diagnosed_cases_period": flow_map["U_to_D"],
        "tested_for_viral_load": art * pi_test,
        "virally_suppressed": states[:, STATE_NAMES.index("V")] * pi_test * pi_doc,
        "estimated_plhiv": total,
    }
    plausibility = {
        "non_finite_count": int(sum(int(np.sum(~np.isfinite(values))) for values in list(predictions.values()) + [incidence_flow])),
        "population_violation_count": int(np.sum(total - snapshot.population_denominator > eps)),
    }
    return {
        "states": states,
        "flows": flow_map,
        "hazards": hazard_map,
        "eta": dict(base_eta),
        "predictions": predictions,
        "incidence_flow": incidence_flow,
        "pi_test": pi_test,
        "pi_doc": pi_doc,
        "plausibility": plausibility,
        "direct_components": dict(fit.simulation["direct_components"]),
        "hidden_components": dict(fit.simulation["hidden_components"]),
    }


def _bic_from_residuals(residuals: np.ndarray, parameter_count: int) -> float:
    residual_array = np.asarray(residuals, dtype=np.float64)
    if residual_array.size <= 0:
        return float("inf")
    sse = float(np.sum(np.square(residual_array)))
    if not np.isfinite(sse) or sse <= np.finfo(np.float64).eps:
        sse = np.finfo(np.float64).eps
    n_obs = float(residual_array.size)
    return float(n_obs * np.log(sse / n_obs) + float(parameter_count) * np.log(n_obs))


def _prediction_gap(snapshot: IntegratedSnapshot, current_simulation: dict[str, Any], strict_simulation: dict[str, Any]) -> dict[str, dict[str, float]]:
    historical_mask = _historical_mask(snapshot)
    future_mask = np.logical_not(historical_mask)
    rows: dict[str, dict[str, float]] = {}
    for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,):
        current_values = np.asarray(current_simulation["predictions"][metric_name], dtype=np.float64)
        strict_values = np.asarray(strict_simulation["predictions"][metric_name], dtype=np.float64)
        observed_historical_mask = np.logical_and(historical_mask, np.asarray(snapshot.metric_masks[metric_name], dtype=bool))
        hist_diff = strict_values[historical_mask] - current_values[historical_mask]
        hist_observed_diff = strict_values[observed_historical_mask] - current_values[observed_historical_mask]
        future_diff = strict_values[future_mask] - current_values[future_mask]
        rows[metric_name] = {
            "historical_all_quarter_mae_gap": float(np.mean(np.abs(hist_diff))) if hist_diff.size else 0.0,
            "historical_all_quarter_max_abs_gap": float(np.max(np.abs(hist_diff))) if hist_diff.size else 0.0,
            "historical_observed_mae_gap": float(np.mean(np.abs(hist_observed_diff))) if hist_observed_diff.size else 0.0,
            "historical_observed_max_abs_gap": float(np.max(np.abs(hist_observed_diff))) if hist_observed_diff.size else 0.0,
            "future_mae_gap": float(np.mean(np.abs(future_diff))) if future_diff.size else 0.0,
            "future_max_abs_gap": float(np.max(np.abs(future_diff))) if future_diff.size else 0.0,
        }
    return rows


def _optimize_branch_gap(snapshot: IntegratedSnapshot, fit: FitResult, branch_name: str) -> BranchAuditResult:
    diagnosis_active = branch_name in {"diagnosis", "combined"}
    care_active = branch_name in {"care", "combined"}
    simultaneous_competing = bool(care_active)
    basis_limit = _basis_width_limit(snapshot, branch_name)
    historical_mask = _historical_mask(snapshot)
    current_residuals = _fit_residuals(snapshot, fit.simulation, historical_mask)
    current_param_count = _age_param_count(fit, branch_name)
    current_bic = _bic_from_residuals(current_residuals, current_param_count)
    diagnosis_profile_current, care_profiles_current = _branch_current_profiles(fit, branch_name)
    age_count = len(snapshot.model_quarters)
    width_trace: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    non_improving_widths = 0

    for width in range(1, basis_limit + 1):
        basis = _age_basis_dct(age_count, width)
        initial_segments: list[np.ndarray] = []
        if diagnosis_active and diagnosis_profile_current is not None:
            diag_coef, *_ = np.linalg.lstsq(basis, diagnosis_profile_current, rcond=None)
            initial_segments.append(np.asarray(diag_coef, dtype=np.float64))
        if care_active:
            for transition in CARE_TRANSITIONS:
                current_profile = np.asarray(care_profiles_current.get(transition, np.zeros((age_count,), dtype=np.float64)), dtype=np.float64)
                care_coef, *_ = np.linalg.lstsq(basis, current_profile, rcond=None)
                initial_segments.append(np.asarray(care_coef, dtype=np.float64))
        x0 = np.concatenate(initial_segments) if initial_segments else np.zeros((0,), dtype=np.float64)

        def _decode(params: np.ndarray) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
            cursor = 0
            diag_profile: np.ndarray | None = None
            if diagnosis_active:
                diag_profile = np.asarray(basis @ np.asarray(params[cursor : cursor + width], dtype=np.float64), dtype=np.float64)
                cursor += width
            care_profiles: dict[str, np.ndarray] = {}
            if care_active:
                for transition in CARE_TRANSITIONS:
                    care_profiles[transition] = np.asarray(basis @ np.asarray(params[cursor : cursor + width], dtype=np.float64), dtype=np.float64)
                    cursor += width
            return diag_profile, care_profiles

        def residual_fn(params: np.ndarray) -> np.ndarray:
            diag_profile, care_profiles = _decode(np.asarray(params, dtype=np.float64))
            simulation = _simulate_conditional_gap(
                snapshot,
                fit,
                diagnosis_profile=diag_profile,
                care_profiles=care_profiles,
                simultaneous_competing=simultaneous_competing,
            )
            residuals = _fit_residuals(snapshot, simulation, historical_mask)
            if simulation["plausibility"]["non_finite_count"] or simulation["plausibility"]["population_violation_count"]:
                penalty = np.asarray(
                    [
                        float(simulation["plausibility"]["non_finite_count"]),
                        float(simulation["plausibility"]["population_violation_count"]),
                    ],
                    dtype=np.float64,
                )
                return penalty if residuals.size == 0 else np.concatenate([residuals, penalty])
            return residuals

        max_nfev = int(max(200, 50 * max(1, x0.size)))
        result = optimize.least_squares(residual_fn, x0=x0, method="trf", max_nfev=max_nfev)
        fitted_params = np.asarray(result.x, dtype=np.float64)
        diag_profile, care_profiles = _decode(fitted_params)
        strict_simulation = _simulate_conditional_gap(
            snapshot,
            fit,
            diagnosis_profile=diag_profile,
            care_profiles=care_profiles,
            simultaneous_competing=simultaneous_competing,
        )
        strict_residuals = _fit_residuals(snapshot, strict_simulation, historical_mask)
        metrics = _compute_window_metrics(snapshot, strict_simulation, historical_mask)
        parameter_count = int(fitted_params.size)
        bic = _bic_from_residuals(strict_residuals, parameter_count)
        row = {
            "basis_width": int(width),
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "strict_bic": float(bic),
            "strict_primary_loss": float(metrics["primary_loss"]),
            "strict_diag_flow_loss": float(metrics["diag_flow_loss"]),
            "strict_secondary_loss": float(metrics["secondary_loss"]),
            "strict_total_loss": float(metrics["total_loss"]),
            "strict_population_violation_count": float(metrics["population_violation_count"]),
            "strict_non_finite_count": float(metrics["non_finite_count"]),
            "strict_age_param_count": parameter_count,
        }
        width_trace.append(row)

        hard_valid = bool(
            result.success
            and float(metrics["non_finite_count"]) <= 0.0
            and float(metrics["population_violation_count"]) <= 0.0
        )
        if hard_valid and (best_payload is None or float(bic) < float(best_payload["strict_bic"])):
            best_payload = {
                "basis_width": int(width),
                "strict_bic": float(bic),
                "strict_metrics": dict(metrics),
                "strict_simulation": strict_simulation,
                "strict_age_param_count": parameter_count,
            }
            non_improving_widths = 0
        else:
            non_improving_widths += 1
            if non_improving_widths >= 3:
                break

    if best_payload is None:
        raise RuntimeError(f"Strict surrogate optimization did not produce a valid solution for {branch_name}")

    return BranchAuditResult(
        branch_name=branch_name,
        current_fit=fit,
        current_bic=float(current_bic),
        current_age_param_count=int(current_param_count),
        strict_width=int(best_payload["basis_width"]),
        strict_bic=float(best_payload["strict_bic"]),
        strict_metrics=dict(best_payload["strict_metrics"]),
        strict_simulation=dict(best_payload["strict_simulation"]),
        strict_age_param_count=int(best_payload["strict_age_param_count"]),
        simultaneous_competing=simultaneous_competing,
        width_trace=width_trace,
        prediction_gap=_prediction_gap(snapshot, fit.simulation, dict(best_payload["strict_simulation"])),
    )


def _spec_vs_code_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "row_id": "incidence",
            "module_name": "Module A: Incidence",
            "spec_equation": "log(lambda_t) = alpha_I + beta_I^T w_t + sum_(b,l) Gamma_(I,b,l) z_b(t-l) + sum_m Psi_m u_m(t) + xi_t ; I_t = N_t * lambda_t",
            "code_equation": "eta_I(t) = alpha_I + X_inc(t) beta_I + H(t) psi ; lambda_t = exp(eta_I(t)) ; I_t = N_t * lambda_t",
            "spec_reference": "phase3_mathematical_spec_2026_04_09.md:199-217",
            "code_reference": "integrated_autoresearch.py:676-686",
            "mathematically_equivalent": "Explicit denominator-times-hazard incidence is implemented.",
            "missing_terms": "No optional renewal term, no separate w_t block beyond direct Phase 2 covariates, no stochastic xi_t process, and hidden modes are static SVD scores rather than dynamic latent states.",
        },
        {
            "row_id": "diagnosis_hazard",
            "module_name": "Module B1: Explicit U -> D Hazard",
            "spec_equation": "d_t = h_U_to_D(t) U_t ; h_U_to_D(t) = 1 - exp(-exp(eta_U_to_D(t)))",
            "code_equation": "diagnosed_by_age = min(U_age, h_U_to_D(t) * U_age) ; d_t = sum_a diagnosed_by_age(a) when diagnosis_family = hazard",
            "spec_reference": "phase3_mathematical_spec_2026_04_09.md:226-246",
            "code_reference": "integrated_autoresearch.py:733-739 and 794-799",
            "mathematically_equivalent": "Yes. The implemented hazard branch matches the explicit cloglog U -> D hazard family.",
            "missing_terms": "No meaningful family mismatch.",
        },
        {
            "row_id": "diagnosis_delay",
            "module_name": "Module B2: Diagnosis Delay / Back-Calculation",
            "spec_equation": "E[y_newdiag_t] = sum_k I_(t-k) pi_k(t) with pi_k(t) = g_k(t) prod_(j<k)(1 - g_j(t-k+j))",
            "code_equation": "U cohorts are aged forward; g_t(a) = sigmoid(eta_U_to_D(t) + beta_D log(1+a)); d_t = sum_a g_t(a) U_t(a)",
            "spec_reference": "phase3_mathematical_spec_2026_04_09.md:248-265",
            "code_reference": "integrated_autoresearch.py:619-620 and 733-740",
            "mathematically_equivalent": "Equivalent only in the restricted sense that the code induces an implicit delay kernel through cohort survival in U.",
            "missing_terms": "The delay kernel is restricted to one age-shape parameter beta_D on log(1+a), not a general g_k(t) or explicit pi_k(t) family.",
        },
        {
            "row_id": "downstream_semi_markov",
            "module_name": "Module C: Downstream Semi-Markov Care",
            "spec_equation": "h_r(t,a) = 1 - exp(-exp(eta_r(t,a))) with eta_r(t,a) = alpha_r + f_r(a) + q_r(t) + ...",
            "code_equation": "h_r(t,a) = 1 - exp(-exp(eta_r(t) + beta_r log(1+a))) for r in {D_to_A, A_to_V, A_to_L, L_to_A} when care_family = semi_markov",
            "spec_reference": "phase3_mathematical_spec_2026_04_09.md:267-292",
            "code_reference": "integrated_autoresearch.py:690-705 and 742-772",
            "mathematically_equivalent": "Equivalent only for the narrow subclass f_r(a) = beta_r log(1+a).",
            "missing_terms": "No flexible dwell-time function f_r(a); only one age-shape coefficient per downstream transition.",
        },
        {
            "row_id": "a_competing_risks",
            "module_name": "Module E: A -> V and A -> L Exit Mechanics",
            "spec_equation": "v_t = h_A_to_V(t,a_A) A_t ; l_t = h_A_to_L(t,a_A) A_t with concurrent exits from A",
            "code_equation": "A_to_V is applied first; survivors are then exposed to A_to_L using a_to_l_by_age = min(A_after_V, h_A_to_L(t,a) * A_age)",
            "spec_reference": "phase3_mathematical_spec_2026_04_09.md:316-330",
            "code_reference": "integrated_autoresearch.py:752-768",
            "mathematically_equivalent": "Not exactly. The current implementation is a sequential clipping approximation, not a simultaneous competing-risks construction.",
            "missing_terms": "Order dependence between suppression and attrition exits remains in code.",
        },
        {
            "row_id": "hidden_structure",
            "module_name": "Module D: Shared Hidden Structure",
            "spec_equation": "u_t = A u_(t-1) + epsilon_t ; epsilon_t ~ N(0, Sigma_u)",
            "code_equation": "Historical national factor matrix F is SVD-decomposed and hidden scores are H = F V_r^T",
            "spec_reference": "phase3_mathematical_spec_2026_04_09.md:294-310",
            "code_reference": "integrated_autoresearch.py:313-332",
            "mathematically_equivalent": "No. The code uses static latent coordinates extracted from the observed factor matrix, not a dynamic latent state evolution model.",
            "missing_terms": "No transition dynamics A, no epsilon_t process, and no province-resolved hidden state.",
        },
        {
            "row_id": "phase2_insertion",
            "module_name": "Module G: Where Phase 2 Enters",
            "spec_equation": "Direct temporal surface enters as structured priors or covariates; hidden modes enter as shared shocks; coefficients remain module-specific",
            "code_equation": "eta_module(t) = alpha_module + X_direct,module(t) beta_module + H(t) lambda_module",
            "spec_reference": "phase3_mathematical_spec_2026_04_09.md:367-382",
            "code_reference": "integrated_autoresearch.py:400-446 and 597-616",
            "mathematically_equivalent": "Module-specific direct and hidden terms are separated correctly.",
            "missing_terms": "The direct surface is inserted as ordinary fitted covariates, not structured priors, and hidden scores are national-only.",
        },
        {
            "row_id": "observation_layer",
            "module_name": "Module F: Observation / Ascertainment",
            "spec_equation": "E[y_vltest_t] = (A_t + V_t) pi_test_t ; E[y_vs_t] = V_t pi_test_t pi_doc_t ; logit(pi_test_t), logit(pi_doc_t) depend on observation covariates and hidden shocks",
            "code_equation": "tested_for_viral_load = (A + V) * pi_test ; virally_suppressed = V * pi_test * pi_doc ; pi_test = sigmoid(eta_test), pi_doc = sigmoid(eta_doc)",
            "spec_reference": "phase3_mathematical_spec_2026_04_09.md:341-365",
            "code_reference": "integrated_autoresearch.py:706-725 and 808-815",
            "mathematically_equivalent": "Yes. This is the strongest spec-to-code match in the current implementation.",
            "missing_terms": "No major family mismatch; the main issue is sparse observation coverage rather than the observation equations themselves.",
        },
    ]
    order_lookup = {row_id: idx for idx, row_id in enumerate(SPEC_ROW_ORDER)}
    rows.sort(key=lambda row: order_lookup.get(str(row["row_id"]), len(order_lookup)))
    return rows


def _write_spec_vs_code_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# AN-03C Phase 3 Strict Spec vs Code Audit",
        "",
        "| Module | Spec Equation | Implemented Code Equation | Equivalent Part | Missing |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {module} | `{spec}` | `{code}` | {equivalent} | {missing} |".format(
                module=str(row["module_name"]).replace("|", "/"),
                spec=str(row["spec_equation"]).replace("|", "/"),
                code=str(row["code_equation"]).replace("|", "/"),
                equivalent=str(row["mathematically_equivalent"]).replace("|", "/"),
                missing=str(row["missing_terms"]).replace("|", "/"),
            )
        )
    lines.extend(["", "## References", ""])
    for row in rows:
        lines.append(f"- **{row['module_name']}**: spec `{row['spec_reference']}` ; code `{row['code_reference']}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def _branch_summary_payload(result: BranchAuditResult) -> dict[str, Any]:
    current_metrics = dict(result.current_fit.window_metrics["historical"] or {})
    metric_deltas = {
        "primary_loss_delta": float(result.strict_metrics["primary_loss"] - current_metrics["primary_loss"]),
        "diag_flow_loss_delta": float(result.strict_metrics["diag_flow_loss"] - current_metrics["diag_flow_loss"]),
        "secondary_loss_delta": float(result.strict_metrics["secondary_loss"] - current_metrics["secondary_loss"]),
        "total_loss_delta": float(result.strict_metrics["total_loss"] - current_metrics["total_loss"]),
    }
    return {
        "branch_name": result.branch_name,
        "label": BRANCH_LABELS[result.branch_name],
        "current_candidate_id": result.current_fit.config.candidate_id,
        "current_bic": float(result.current_bic),
        "current_age_param_count": int(result.current_age_param_count),
        "current_metrics": {
            key: float(current_metrics[key])
            for key in ("primary_loss", "diag_flow_loss", "secondary_loss", "total_loss")
        },
        "strict_basis_width": int(result.strict_width),
        "strict_bic": float(result.strict_bic),
        "strict_age_param_count": int(result.strict_age_param_count),
        "strict_metrics": {
            key: float(result.strict_metrics[key])
            for key in ("primary_loss", "diag_flow_loss", "secondary_loss", "total_loss")
        },
        "strict_simultaneous_competing": bool(result.simultaneous_competing),
        "metric_deltas": metric_deltas,
        "prediction_gap": result.prediction_gap,
        "width_trace": list(result.width_trace),
    }


def _write_metric_delta_chart(path: Path, branch_results: dict[str, BranchAuditResult]) -> None:
    branch_names = [branch for branch in ("diagnosis", "care", "combined") if branch in branch_results]
    x_axis = np.arange(len(branch_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for offset, metric_name in enumerate(("primary_loss", "diag_flow_loss", "secondary_loss", "total_loss")):
        deltas = [
            float(branch_results[branch].strict_metrics[metric_name] - branch_results[branch].current_fit.window_metrics["historical"][metric_name])
            for branch in branch_names
        ]
        ax.bar(x_axis + (offset - 1.5) * width, deltas, width=width, label=metric_name)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("AN-03C Strict-Spec Surrogate Minus Current Branch Loss")
    ax.set_ylabel("strict minus current")
    ax.set_xticks(x_axis)
    ax.set_xticklabels([BRANCH_LABELS[branch] for branch in branch_names], rotation=15, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_prediction_gap_chart(path: Path, branch_results: dict[str, BranchAuditResult]) -> None:
    metrics = list(PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,))
    branch_names = [branch for branch in ("diagnosis", "care", "combined") if branch in branch_results]
    matrix = np.zeros((len(branch_names), len(metrics)), dtype=np.float64)
    for branch_idx, branch_name in enumerate(branch_names):
        for metric_idx, metric_name in enumerate(metrics):
            matrix[branch_idx, metric_idx] = float(
                branch_results[branch_name].prediction_gap[metric_name]["historical_observed_mae_gap"]
            )
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("AN-03C Historical Observed-Quarter Prediction Divergence")
    ax.set_yticks(range(len(branch_names)))
    ax.set_yticklabels([BRANCH_LABELS[branch] for branch in branch_names])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_bic_trace_chart(path: Path, branch_results: dict[str, BranchAuditResult]) -> None:
    ordered_names = [branch for branch in ("diagnosis", "care", "combined") if branch in branch_results]
    fig, axes = plt.subplots(len(ordered_names), 1, figsize=(11.5, 3.6 * max(len(ordered_names), 1)))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    for axis, branch_name in zip(axes, ordered_names):
        result = branch_results[branch_name]
        widths = [int(row["basis_width"]) for row in result.width_trace]
        bic_values = [float(row["strict_bic"]) for row in result.width_trace]
        axis.plot(widths, bic_values, marker="o", linewidth=1.5)
        axis.axhline(float(result.current_bic), color="tab:red", linestyle="--", linewidth=1.0, label="current branch BIC")
        axis.set_title(f"AN-03C {BRANCH_LABELS[branch_name]} BIC Trace")
        axis.set_ylabel("conditional BIC")
        axis.set_xlabel("strict basis width")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_dashboard_markdown(path: Path, branch_payloads: dict[str, Any]) -> None:
    lines = [
        "# AN-03C Strict Spec Gap Dashboard",
        "",
        "This audit holds the current branch's time-varying baseline terms fixed and relaxes only the age/dwell representation toward a stricter spec surrogate.",
        "",
    ]
    for branch_name in ("diagnosis", "care", "combined"):
        payload = branch_payloads.get(branch_name)
        if payload is None:
            continue
        lines.extend(
            [
                f"## {payload['label']}",
                "",
                f"- Current candidate: `{payload['current_candidate_id']}`",
                f"- Current BIC: `{payload['current_bic']:.6f}`",
                f"- Strict surrogate width: `{payload['strict_basis_width']}`",
                f"- Strict surrogate BIC: `{payload['strict_bic']:.6f}`",
                f"- Primary loss delta: `{payload['metric_deltas']['primary_loss_delta']:.6f}`",
                f"- Diagnosis-flow loss delta: `{payload['metric_deltas']['diag_flow_loss_delta']:.6f}`",
                f"- Secondary loss delta: `{payload['metric_deltas']['secondary_loss_delta']:.6f}`",
                f"- Total loss delta: `{payload['metric_deltas']['total_loss_delta']:.6f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _export_paper_figures(ctx: TransitionResearchContext, chart_paths: dict[str, Path]) -> dict[str, Any]:
    archive_dir = ROOT_DIR / "artifacts" / "paper_figures" / "phase3_hmba_20260409"
    archive_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = list(read_json(archive_dir / "figure_manifest.json", default=[]))
    retained = [
        row
        for row in existing_manifest
        if isinstance(row, dict) and not str(row.get("figure_id") or "").startswith("fig_an03c_")
    ]
    new_rows: list[dict[str, Any]] = []
    figure_specs = [
        (
            "fig_an03c_01",
            "Strict Spec Gap Loss Delta",
            "Historical loss deltas between the current branch implementation and the stricter spec surrogate.",
            "an03c_strict_gap_loss_delta.png",
            chart_paths.get("metric_deltas"),
        ),
        (
            "fig_an03c_02",
            "Strict Spec Prediction Divergence",
            "Historical prediction divergence between the current branch implementation and the stricter spec surrogate.",
            "an03c_prediction_divergence.png",
            chart_paths.get("prediction_gap"),
        ),
        (
            "fig_an03c_03",
            "Strict Spec BIC Trace",
            "Conditional BIC trace as age/dwell basis width is increased beyond the current single-parameter age-in-state extension.",
            "an03c_bic_trace.png",
            chart_paths.get("bic_trace"),
        ),
    ]
    for figure_id, title, caption, filename, source_path in figure_specs:
        if source_path is None or not source_path.exists():
            continue
        destination = archive_dir / filename
        shutil.copy2(source_path, destination)
        new_rows.append(
            {
                "figure_id": figure_id,
                "title": title,
                "caption": caption,
                "source_path": str(source_path),
                "archived_path": str(destination),
            }
        )
    manifest = retained + new_rows
    manifest_path = archive_dir / "figure_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "archive_dir": str(archive_dir),
        "figure_manifest": str(manifest_path),
        "figure_count_added": int(len(new_rows)),
        "figure_count_total": int(len(manifest)),
    }


def run_an03c(ctx: TransitionResearchContext) -> dict[str, Any]:
    snapshot = _build_snapshot(ctx)
    selected_configs = _select_reference_configs()
    reference_fits = {
        branch_name: _reference_fit(snapshot, config)
        for branch_name, config in selected_configs.items()
    }
    branch_results = {
        branch_name: _optimize_branch_gap(snapshot, fit, branch_name)
        for branch_name, fit in reference_fits.items()
    }
    spec_rows = _spec_vs_code_rows()

    spec_table_path = ctx.experiment_dir / "spec_vs_code_table.json"
    spec_markdown_path = ctx.experiment_dir / "spec_vs_code_table.md"
    divergence_summary_path = ctx.experiment_dir / "strict_spec_divergence_summary.json"
    diagnosis_path = ctx.experiment_dir / "diagnosis_divergence.json"
    care_path = ctx.experiment_dir / "care_divergence.json"
    combined_path = ctx.experiment_dir / "combined_divergence.json"
    dashboard_markdown_path = ctx.experiment_dir / "strict_spec_gap_dashboard.md"
    metric_chart_path = ctx.experiment_dir / "strict_spec_gap_loss_delta.png"
    prediction_chart_path = ctx.experiment_dir / "strict_spec_prediction_divergence.png"
    bic_trace_path = ctx.experiment_dir / "strict_spec_bic_trace.png"

    write_json(spec_table_path, spec_rows)
    _write_spec_vs_code_markdown(spec_markdown_path, spec_rows)

    branch_payloads = {branch_name: _branch_summary_payload(result) for branch_name, result in branch_results.items()}
    write_json(diagnosis_path, branch_payloads["diagnosis"])
    write_json(care_path, branch_payloads["care"])
    write_json(combined_path, branch_payloads["combined"])
    divergence_summary = {
        "reference_branch_configs": {
            branch_name: result.current_fit.config.candidate_id for branch_name, result in branch_results.items()
        },
        "branch_summaries": branch_payloads,
    }
    write_json(divergence_summary_path, divergence_summary)
    _write_dashboard_markdown(dashboard_markdown_path, branch_payloads)

    _write_metric_delta_chart(metric_chart_path, branch_results)
    _write_prediction_gap_chart(prediction_chart_path, branch_results)
    _write_bic_trace_chart(bic_trace_path, branch_results)

    archive_payload = _export_paper_figures(
        ctx,
        {
            "metric_deltas": metric_chart_path,
            "prediction_gap": prediction_chart_path,
            "bic_trace": bic_trace_path,
        },
    )

    branch_scorecard = {
        branch_name: {
            "current_candidate_id": result.current_fit.config.candidate_id,
            "current_primary_loss": float(result.current_fit.window_metrics["historical"]["primary_loss"]),
            "strict_primary_loss": float(result.strict_metrics["primary_loss"]),
            "current_diag_flow_loss": float(result.current_fit.window_metrics["historical"]["diag_flow_loss"]),
            "strict_diag_flow_loss": float(result.strict_metrics["diag_flow_loss"]),
            "current_secondary_loss": float(result.current_fit.window_metrics["historical"]["secondary_loss"]),
            "strict_secondary_loss": float(result.strict_metrics["secondary_loss"]),
            "current_total_loss": float(result.current_fit.window_metrics["historical"]["total_loss"]),
            "strict_total_loss": float(result.strict_metrics["total_loss"]),
            "current_bic": float(result.current_bic),
            "strict_bic": float(result.strict_bic),
            "strict_basis_width": int(result.strict_width),
            "strict_simultaneous_competing": bool(result.simultaneous_competing),
        }
        for branch_name, result in branch_results.items()
    }

    decision = {
        "kept_current_labels": {branch_name: result.current_fit.config.candidate_id for branch_name, result in branch_results.items()},
        "strict_surrogate_selected_widths": {branch_name: int(result.strict_width) for branch_name, result in branch_results.items()},
        "strict_surrogate_beats_current_on_bic": {
            branch_name: bool(float(result.strict_bic) < float(result.current_bic))
            for branch_name, result in branch_results.items()
        },
        "strict_surrogate_beats_current_on_primary_loss": {
            branch_name: bool(float(result.strict_metrics["primary_loss"]) < float(result.current_fit.window_metrics["historical"]["primary_loss"]))
            for branch_name, result in branch_results.items()
        },
        "paper_figure_archive": archive_payload,
    }

    experiment_spec = {
        "audit_question": "How far is the current Phase 3 code from the strict written spec on diagnosis delay and semi-Markov care?",
        "source_run_id": ctx.source_run_id,
        "reference_integrated_run": str(_latest_integrated_experiment_dir()) if _latest_integrated_experiment_dir() is not None else None,
        "method": "Hold the current branch's time-varying baseline terms fixed, replace the single-parameter age-in-state extension with a stricter DCT-basis surrogate, and compare conditional historical fit and BIC.",
        "spec_rows": len(spec_rows),
        "branch_scorecard": branch_scorecard,
        "artifacts": {
            "spec_vs_code_table": str(spec_table_path),
            "spec_vs_code_markdown": str(spec_markdown_path),
            "divergence_summary": str(divergence_summary_path),
            "diagnosis_divergence": str(diagnosis_path),
            "care_divergence": str(care_path),
            "combined_divergence": str(combined_path),
            "dashboard_markdown": str(dashboard_markdown_path),
        },
    }
    coverage_summary = {
        "historical_quarter_count": int(len(snapshot.historical_quarters)),
        "future_quarter_count": int(len(snapshot.future_quarters)),
        "metric_observation_coverage": {
            metric_name: int(_coverage_count(snapshot, metric_name))
            for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,)
        },
        "basis_width_limits": {
            branch_name: int(_basis_width_limit(snapshot, branch_name)) for branch_name in BRANCH_REFERENCE_KEYS
        },
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            numerical_guard_entry(
                name="strict_spec_gap_machine_epsilon",
                role="conditional gap simulator lower bound",
                why_needed="Protects denominator, probability, and hazard calculations in the strict-spec conditional audit.",
            )
        ],
    )
    return {
        "experiment_id": ctx.experiment.experiment_id,
        "experiment_dir": str(ctx.experiment_dir),
        "artifacts": {
            **artifacts,
            "spec_vs_code_table": str(spec_table_path),
            "spec_vs_code_markdown": str(spec_markdown_path),
            "divergence_summary": str(divergence_summary_path),
            "diagnosis_divergence": str(diagnosis_path),
            "care_divergence": str(care_path),
            "combined_divergence": str(combined_path),
            "dashboard_markdown": str(dashboard_markdown_path),
            "strict_spec_gap_loss_delta": str(metric_chart_path),
            "strict_spec_prediction_divergence": str(prediction_chart_path),
            "strict_spec_bic_trace": str(bic_trace_path),
        },
        "decision": decision,
        "coverage_summary": coverage_summary,
    }
