from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from epigraph_ph.runtime import ROOT_DIR, utc_now_iso, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .integrated_autoresearch import (
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    STATE_NAMES,
    TOTAL_METRIC,
    TRANSITION_NAMES,
    _cloglog_inverse,
    _fit_residuals,
    _module_linear_predictor,
    _safe_exp,
    _sigmoid,
)
from .numeric_policy import numerical_guard_entry
from .strict_diagnosis_kernel_research import (
    BlockedTimeContract,
    DiagnosisKernelCandidate,
    _age_basis_log,
    _age_profile,
    _age_shift,
    _baseline_payload,
    _build_blocked_time_contract,
    _compute_window_metrics,
    _initial_param_vector,
    _matrices_for_reference,
    _score_tuple,
    _select_promoted_strict_reference,
    _build_param_slices as _build_reference_param_slices,
)


@dataclass(slots=True)
class IncidenceFamilyCandidate:
    candidate_id: str
    incidence_family: str
    lag_count: int
    use_direct_covariates: bool
    diagnosis_candidate: DiagnosisKernelCandidate


@dataclass(slots=True)
class IncidenceFamilyFit:
    candidate: IncidenceFamilyCandidate
    success: bool
    status: int
    message: str
    train_cost: float
    param_vector: np.ndarray
    param_slices: dict[str, slice]
    matrices: dict[str, Any]
    simulation: dict[str, Any]
    split_metrics: dict[str, dict[str, float]]
    train_bic: float
    nfev: int
    max_nfev: int
    wall_seconds: float


def _build_family_candidates(
    reference_candidate: DiagnosisKernelCandidate,
    contract: BlockedTimeContract,
) -> list[IncidenceFamilyCandidate]:
    lag_limit = max(1, int(len(contract.train_diagnosis_quarters)) - 1)
    candidates: list[IncidenceFamilyCandidate] = []
    for use_direct_covariates in (True, False):
        direct_label = "dirall" if use_direct_covariates else "dirnone"
        candidates.append(
            IncidenceFamilyCandidate(
                candidate_id=f"inc-level-{direct_label}",
                incidence_family="level",
                lag_count=0,
                use_direct_covariates=use_direct_covariates,
                diagnosis_candidate=reference_candidate,
            )
        )
        for lag_count in range(1, lag_limit + 1):
            candidates.append(
                IncidenceFamilyCandidate(
                    candidate_id=f"inc-renewal-lag{lag_count:02d}-{direct_label}",
                    incidence_family="renewal",
                    lag_count=int(lag_count),
                    use_direct_covariates=use_direct_covariates,
                    diagnosis_candidate=reference_candidate,
                )
            )
            candidates.append(
                IncidenceFamilyCandidate(
                    candidate_id=f"inc-diagnosis-feedback-lag{lag_count:02d}-{direct_label}",
                    incidence_family="diagnosis_feedback",
                    lag_count=int(lag_count),
                    use_direct_covariates=use_direct_covariates,
                    diagnosis_candidate=reference_candidate,
                )
            )
    return candidates


def _family_matrices(
    snapshot: Any,
    reference_config: Any,
    candidate: IncidenceFamilyCandidate,
) -> dict[str, Any]:
    matrices = _matrices_for_reference(snapshot, reference_config)
    if candidate.use_direct_covariates:
        return matrices
    return {
        "incidence_direct": np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64),
        "transition_direct": matrices["transition_direct"],
        "observation_direct": matrices["observation_direct"],
        "hidden": matrices["hidden"],
        "feature_ids": {
            "incidence": [],
            "transition": dict(matrices["feature_ids"]["transition"]),
            "observation": dict(matrices["feature_ids"]["observation"]),
        },
    }


def _build_param_slices(candidate: IncidenceFamilyCandidate, matrices: dict[str, Any]) -> dict[str, slice]:
    reference_slices = _build_reference_param_slices(candidate.diagnosis_candidate, matrices)
    incidence_width = 1 + matrices["incidence_direct"].shape[1] + matrices["hidden"].shape[1] + int(candidate.lag_count)
    shift = incidence_width - (reference_slices["incidence"].stop - reference_slices["incidence"].start)
    if shift == 0:
        return reference_slices
    slices: dict[str, slice] = {}
    for key, value in reference_slices.items():
        if key == "full":
            continue
        if key == "incidence":
            slices[key] = slice(value.start, value.start + incidence_width)
        elif value.start >= reference_slices["incidence"].stop:
            slices[key] = slice(value.start + shift, value.stop + shift)
        else:
            slices[key] = slice(value.start, value.stop)
    slices["full"] = slice(0, reference_slices["full"].stop + shift)
    return slices


def _history_vector(
    candidate: IncidenceFamilyCandidate,
    idx: int,
    incidence_flow: np.ndarray,
    diagnosis_flow: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    if candidate.lag_count <= 0:
        return np.zeros((0,), dtype=np.float64)
    values = np.zeros((int(candidate.lag_count),), dtype=np.float64)
    for lag in range(1, int(candidate.lag_count) + 1):
        history_idx = idx - lag
        if history_idx < 0:
            continue
        if candidate.incidence_family == "renewal":
            source_value = float(incidence_flow[history_idx])
        elif candidate.incidence_family == "diagnosis_feedback":
            source_value = float(diagnosis_flow[history_idx])
        else:
            source_value = 0.0
        share = source_value / max(float(denominator[history_idx]), float(np.finfo(np.float64).eps))
        values[lag - 1] = np.log1p(max(share, 0.0))
    return values


def _simulate_candidate(
    snapshot: Any,
    candidate: IncidenceFamilyCandidate,
    fit_vector: np.ndarray,
    param_slices: dict[str, slice],
    matrices: dict[str, Any],
) -> dict[str, Any]:
    eps = float(np.finfo(np.float64).eps)
    quarter_count = len(snapshot.model_quarters)
    age_count = quarter_count
    age_basis = _age_basis_log(age_count)
    states = np.zeros((quarter_count, len(STATE_NAMES)), dtype=np.float64)
    flow_map = {transition: np.zeros((quarter_count,), dtype=np.float64) for transition in TRANSITION_NAMES}
    hazard_map = {transition: np.zeros((quarter_count,), dtype=np.float64) for transition in TRANSITION_NAMES}
    eta_map: dict[str, np.ndarray] = {}
    direct_component_map: dict[str, np.ndarray] = {}
    hidden_component_map: dict[str, np.ndarray] = {}
    age_slope_map: dict[str, float] = {}

    init_u = float(_safe_exp(fit_vector[param_slices["init_u_log"]])[0])
    init_art_share = float(_sigmoid(fit_vector[param_slices["init_art_share"]])[0])
    init_v_share = float(_sigmoid(fit_vector[param_slices["init_v_share"]])[0])
    diagnosed_stock = max(snapshot.start_diag, 0.0)
    init_art_stock = diagnosed_stock * init_art_share
    diag_gap = max(diagnosed_stock - init_art_stock, 0.0)
    init_l_share = float(_sigmoid(fit_vector[param_slices["init_l_share"]])[0]) if diag_gap > eps else 0.0

    u_age = np.zeros((age_count,), dtype=np.float64)
    d_age = np.zeros((age_count,), dtype=np.float64)
    a_age = np.zeros((age_count,), dtype=np.float64)
    l_age = np.zeros((age_count,), dtype=np.float64)
    u_age[-1] = max(init_u, 0.0)
    v_stock = init_art_stock * init_v_share
    a_age[-1] = init_art_stock - v_stock
    l_age[-1] = diag_gap * init_l_share
    d_age[-1] = diag_gap - l_age[-1]

    states[0, STATE_NAMES.index("U")] = float(np.sum(u_age))
    states[0, STATE_NAMES.index("D")] = float(np.sum(d_age))
    states[0, STATE_NAMES.index("A")] = float(np.sum(a_age))
    states[0, STATE_NAMES.index("V")] = float(v_stock)
    states[0, STATE_NAMES.index("L")] = float(np.sum(l_age))

    incidence_params = np.asarray(fit_vector[param_slices["incidence"]], dtype=np.float64)
    direct_width = matrices["incidence_direct"].shape[1]
    hidden_width = matrices["hidden"].shape[1]
    cursor = 1
    incidence_direct_beta = (
        np.asarray(incidence_params[cursor : cursor + direct_width], dtype=np.float64)
        if direct_width
        else np.zeros((0,), dtype=np.float64)
    )
    cursor += direct_width
    incidence_hidden_beta = (
        np.asarray(incidence_params[cursor : cursor + hidden_width], dtype=np.float64)
        if hidden_width
        else np.zeros((0,), dtype=np.float64)
    )
    cursor += hidden_width
    incidence_history_beta = np.asarray(incidence_params[cursor : cursor + int(candidate.lag_count)], dtype=np.float64)
    incidence_intercept = float(incidence_params[0]) if incidence_params.size else 0.0
    incidence_direct_component = (
        np.asarray(matrices["incidence_direct"], dtype=np.float64) @ incidence_direct_beta
        if direct_width
        else np.zeros((quarter_count,), dtype=np.float64)
    )
    incidence_hidden_component = (
        np.asarray(matrices["hidden"], dtype=np.float64) @ incidence_hidden_beta
        if hidden_width
        else np.zeros((quarter_count,), dtype=np.float64)
    )
    eta_map["incidence"] = np.zeros((quarter_count,), dtype=np.float64)
    direct_component_map["incidence"] = incidence_direct_component
    hidden_component_map["incidence"] = incidence_hidden_component
    incidence_flow = np.zeros((quarter_count,), dtype=np.float64)

    diagnosis_age_profile = np.zeros((age_count,), dtype=np.float64)
    for transition in TRANSITION_NAMES:
        transition_params = np.asarray(fit_vector[param_slices[f"transition::{transition}"]], dtype=np.float64)
        if transition == "U_to_D":
            intercept = np.full((quarter_count,), float(transition_params[0]), dtype=np.float64)
            cursor = 1
            if candidate.diagnosis_candidate.diagnosis_kind == "delay":
                diagnosis_age_profile = _age_profile(candidate.diagnosis_candidate, transition_params, age_count)
                cursor += 1
            elif candidate.diagnosis_candidate.diagnosis_kind == "strict":
                diagnosis_age_profile = _age_profile(candidate.diagnosis_candidate, transition_params, age_count)
                cursor += int(candidate.diagnosis_candidate.kernel_width)
            transition_direct_width = matrices["transition_direct"][transition].shape[1]
            transition_hidden_width = matrices["hidden"].shape[1]
            direct_beta = (
                np.asarray(transition_params[cursor : cursor + transition_direct_width], dtype=np.float64)
                if transition_direct_width
                else np.zeros((0,), dtype=np.float64)
            )
            cursor += transition_direct_width
            hidden_beta = (
                np.asarray(transition_params[cursor : cursor + transition_hidden_width], dtype=np.float64)
                if transition_hidden_width
                else np.zeros((0,), dtype=np.float64)
            )
            direct_component = (
                np.asarray(matrices["transition_direct"][transition], dtype=np.float64) @ direct_beta
                if transition_direct_width
                else np.zeros((quarter_count,), dtype=np.float64)
            )
            hidden_component = (
                np.asarray(matrices["hidden"], dtype=np.float64) @ hidden_beta
                if transition_hidden_width
                else np.zeros((quarter_count,), dtype=np.float64)
            )
            eta_map[transition] = intercept + direct_component + hidden_component
            direct_component_map[transition] = direct_component
            hidden_component_map[transition] = hidden_component
            if candidate.diagnosis_candidate.diagnosis_kind == "hazard":
                hazard_map[transition] = np.asarray(_cloglog_inverse(eta_map[transition]), dtype=np.float64)
        else:
            intercept_component, age_slope, direct_component, hidden_component = _module_linear_predictor(
                transition_params,
                np.asarray(matrices["transition_direct"][transition], dtype=np.float64),
                np.asarray(matrices["hidden"], dtype=np.float64),
                include_age_slope=candidate.diagnosis_candidate.care_family == "semi_markov",
            )
            eta_map[transition] = intercept_component + direct_component + hidden_component
            direct_component_map[transition] = direct_component
            hidden_component_map[transition] = hidden_component
            age_slope_map[transition] = float(age_slope)
            if candidate.diagnosis_candidate.care_family != "semi_markov":
                hazard_map[transition] = np.asarray(_cloglog_inverse(eta_map[transition]), dtype=np.float64)

    test_params = np.asarray(fit_vector[param_slices["observation::tested_for_viral_load"]], dtype=np.float64)
    test_intercept, _test_age_slope, test_direct, test_hidden = _module_linear_predictor(
        test_params,
        np.asarray(matrices["observation_direct"]["tested_for_viral_load"], dtype=np.float64),
        np.asarray(matrices["hidden"], dtype=np.float64),
    )
    doc_params = np.asarray(fit_vector[param_slices["observation::virally_suppressed"]], dtype=np.float64)
    doc_intercept, _doc_age_slope, doc_direct, doc_hidden = _module_linear_predictor(
        doc_params,
        np.asarray(matrices["observation_direct"]["virally_suppressed"], dtype=np.float64),
        np.asarray(matrices["hidden"], dtype=np.float64),
    )
    eta_map["tested_for_viral_load"] = test_intercept + test_direct + test_hidden
    eta_map["virally_suppressed"] = doc_intercept + doc_direct + doc_hidden
    direct_component_map["tested_for_viral_load"] = test_direct
    direct_component_map["virally_suppressed"] = doc_direct
    hidden_component_map["tested_for_viral_load"] = test_hidden
    hidden_component_map["virally_suppressed"] = doc_hidden
    pi_test = np.asarray(_sigmoid(eta_map["tested_for_viral_load"]), dtype=np.float64)
    pi_doc = np.asarray(_sigmoid(eta_map["virally_suppressed"]), dtype=np.float64)

    eta_map["incidence"][0] = incidence_intercept + incidence_direct_component[0] + incidence_hidden_component[0]
    incidence_flow[0] = float(snapshot.population_denominator[0]) * float(_safe_exp(np.asarray([eta_map["incidence"][0]], dtype=np.float64))[0])

    for idx in range(1, quarter_count):
        history_vector = _history_vector(candidate, idx, incidence_flow, flow_map["U_to_D"], snapshot.population_denominator)
        history_term = float(np.dot(incidence_history_beta, history_vector)) if history_vector.size else 0.0
        eta_map["incidence"][idx] = incidence_intercept + incidence_direct_component[idx] + incidence_hidden_component[idx] + history_term
        incidence_flow[idx] = float(snapshot.population_denominator[idx]) * float(_safe_exp(np.asarray([eta_map["incidence"][idx]], dtype=np.float64))[0])

        u_prev = float(np.sum(u_age))
        d_prev = float(np.sum(d_age))
        a_prev = float(np.sum(a_age))
        l_prev = float(np.sum(l_age))

        if candidate.diagnosis_candidate.diagnosis_kind == "hazard":
            diagnosed_by_age = np.minimum(u_age, hazard_map["U_to_D"][idx] * u_age)
        else:
            diagnosis_prob = np.asarray(_sigmoid(eta_map["U_to_D"][idx] + diagnosis_age_profile), dtype=np.float64)
            diagnosed_by_age = np.minimum(u_age, diagnosis_prob * u_age)
        u_to_d = float(np.sum(diagnosed_by_age))
        u_age_next = _age_shift(np.maximum(u_age - diagnosed_by_age, 0.0))
        u_age_next[0] += float(incidence_flow[idx])

        d_to_a_hazard = (
            np.asarray(_cloglog_inverse(eta_map["D_to_A"][idx] + age_slope_map.get("D_to_A", 0.0) * age_basis), dtype=np.float64)
            if candidate.diagnosis_candidate.care_family == "semi_markov"
            else np.full((age_count,), float(hazard_map["D_to_A"][idx]), dtype=np.float64)
        )
        d_to_a_by_age = np.minimum(d_age, d_to_a_hazard * d_age)
        d_to_a = float(np.sum(d_to_a_by_age))
        d_age_next = _age_shift(np.maximum(d_age - d_to_a_by_age, 0.0))
        d_age_next[0] += u_to_d

        a_to_v_hazard = (
            np.asarray(_cloglog_inverse(eta_map["A_to_V"][idx] + age_slope_map.get("A_to_V", 0.0) * age_basis), dtype=np.float64)
            if candidate.diagnosis_candidate.care_family == "semi_markov"
            else np.full((age_count,), float(hazard_map["A_to_V"][idx]), dtype=np.float64)
        )
        a_to_l_hazard = (
            np.asarray(_cloglog_inverse(eta_map["A_to_L"][idx] + age_slope_map.get("A_to_L", 0.0) * age_basis), dtype=np.float64)
            if candidate.diagnosis_candidate.care_family == "semi_markov"
            else np.full((age_count,), float(hazard_map["A_to_L"][idx]), dtype=np.float64)
        )
        a_to_v_by_age = np.minimum(a_age, a_to_v_hazard * a_age)
        a_after_v = np.maximum(a_age - a_to_v_by_age, 0.0)
        a_to_l_by_age = np.minimum(a_after_v, a_to_l_hazard * a_age)
        a_survivors = np.maximum(a_after_v - a_to_l_by_age, 0.0)
        a_to_v = float(np.sum(a_to_v_by_age))
        a_to_l = float(np.sum(a_to_l_by_age))

        l_to_a_hazard = (
            np.asarray(_cloglog_inverse(eta_map["L_to_A"][idx] + age_slope_map.get("L_to_A", 0.0) * age_basis), dtype=np.float64)
            if candidate.diagnosis_candidate.care_family == "semi_markov"
            else np.full((age_count,), float(hazard_map["L_to_A"][idx]), dtype=np.float64)
        )
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
        "eta": eta_map,
        "predictions": predictions,
        "incidence_flow": incidence_flow,
        "pi_test": pi_test,
        "pi_doc": pi_doc,
        "plausibility": plausibility,
        "direct_components": direct_component_map,
        "hidden_components": hidden_component_map,
    }


def _fit_candidate(
    snapshot: Any,
    reference_config: Any,
    candidate: IncidenceFamilyCandidate,
    contract: BlockedTimeContract,
) -> IncidenceFamilyFit:
    matrices = _family_matrices(snapshot, reference_config, candidate)
    param_slices = _build_param_slices(candidate, matrices)
    x0 = _initial_param_vector(snapshot, candidate.diagnosis_candidate, matrices, param_slices, contract.train_mask)
    residual_budget = sum(
        int(np.sum(np.logical_and(snapshot.metric_masks[metric_name], contract.train_mask)))
        for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,)
    )
    max_nfev = max(1, int(param_slices["full"].stop + residual_budget + int(np.sum(contract.train_mask))))

    def residual_fn(params: np.ndarray) -> np.ndarray:
        simulation = _simulate_candidate(snapshot, candidate, params, param_slices, matrices)
        residuals = _fit_residuals(snapshot, simulation, contract.train_mask)
        if simulation["plausibility"]["non_finite_count"] or simulation["plausibility"]["population_violation_count"]:
            penalty_terms = np.asarray(
                [
                    float(simulation["plausibility"]["non_finite_count"]),
                    float(simulation["plausibility"]["population_violation_count"]),
                ],
                dtype=np.float64,
            )
            return penalty_terms if residuals.size == 0 else np.concatenate([residuals, penalty_terms])
        return residuals

    started = float(np.datetime64("now").astype("datetime64[ns]").astype(np.int64))
    result = optimize.least_squares(residual_fn, x0=x0, method="trf", max_nfev=max_nfev)
    ended = float(np.datetime64("now").astype("datetime64[ns]").astype(np.int64))
    fitted_vector = np.asarray(result.x, dtype=np.float64)
    simulation = _simulate_candidate(snapshot, candidate, fitted_vector, param_slices, matrices)
    train_residuals = _fit_residuals(snapshot, simulation, contract.train_mask)
    split_metrics = {
        "train": _compute_window_metrics(snapshot, simulation, contract.train_mask),
        "validation": _compute_window_metrics(snapshot, simulation, contract.validation_mask),
        "holdout": _compute_window_metrics(snapshot, simulation, contract.holdout_mask),
    }
    sse = max(float(np.sum(np.square(train_residuals))), float(np.finfo(np.float64).eps))
    n_obs = max(float(train_residuals.size), 1.0)
    train_bic = float(n_obs * np.log(sse / n_obs) + float(param_slices["full"].stop) * np.log(n_obs))
    return IncidenceFamilyFit(
        candidate=candidate,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        train_cost=float(result.cost),
        param_vector=fitted_vector,
        param_slices=param_slices,
        matrices=matrices,
        simulation=simulation,
        split_metrics=split_metrics,
        train_bic=train_bic,
        nfev=int(getattr(result, "nfev", 0) or 0),
        max_nfev=int(max_nfev),
        wall_seconds=max(0.0, float(ended - started) / 1_000_000_000.0),
    )


def _fit_payload(fit: IncidenceFamilyFit) -> dict[str, Any]:
    return {
        "candidate_id": fit.candidate.candidate_id,
        "incidence_family": fit.candidate.incidence_family,
        "lag_count": int(fit.candidate.lag_count),
        "use_direct_covariates": bool(fit.candidate.use_direct_covariates),
        "success": bool(fit.success),
        "status": int(fit.status),
        "message": str(fit.message),
        "train_bic": float(fit.train_bic),
        "nfev": int(fit.nfev),
        "max_nfev": int(fit.max_nfev),
        "wall_seconds": float(fit.wall_seconds),
        "split_metrics": {
            split_name: {metric_name: float(metric_value) for metric_name, metric_value in metrics.items()}
            for split_name, metrics in fit.split_metrics.items()
        },
        "incidence_feature_ids": list(fit.matrices["feature_ids"]["incidence"]),
    }


def _fit_is_valid(fit: IncidenceFamilyFit) -> bool:
    validation = fit.split_metrics["validation"]
    holdout = fit.split_metrics["holdout"]
    return bool(
        np.isfinite(float(fit.train_cost))
        and float(validation.get("non_finite_count", 0.0)) <= 0.0
        and float(validation.get("population_violation_count", 0.0)) <= 0.0
        and float(holdout.get("non_finite_count", 0.0)) <= 0.0
        and float(holdout.get("population_violation_count", 0.0)) <= 0.0
        and np.isfinite(float(validation.get("primary_loss", float("inf"))))
        and np.isfinite(float(holdout.get("primary_loss", float("inf"))))
    )


def _frontier_rows(fits: list[IncidenceFamilyFit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_validation_primary = float("inf")
    for fit in fits:
        validation_primary = float(fit.split_metrics["validation"]["primary_loss"])
        best_validation_primary = min(best_validation_primary, validation_primary)
        rows.append(
            {
                "candidate_id": fit.candidate.candidate_id,
                "incidence_family": fit.candidate.incidence_family,
                "lag_count": int(fit.candidate.lag_count),
                "use_direct_covariates": bool(fit.candidate.use_direct_covariates),
                "success": bool(fit.success),
                "valid": bool(_fit_is_valid(fit)),
                "validation_primary_loss": validation_primary,
                "validation_diag_flow_loss": float(fit.split_metrics["validation"]["diag_flow_loss"]),
                "holdout_primary_loss": float(fit.split_metrics["holdout"]["primary_loss"]),
                "holdout_diag_flow_loss": float(fit.split_metrics["holdout"]["diag_flow_loss"]),
                "train_bic": float(fit.train_bic),
                "best_validation_primary_loss": float(best_validation_primary),
            }
        )
    return rows


def _write_frontier_chart(path: Path, frontier_rows: list[dict[str, Any]], reference_candidate_id: str) -> None:
    x_labels = [str(row["candidate_id"]) for row in frontier_rows]
    x_values = np.arange(len(frontier_rows))
    ref_index = next((idx for idx, row in enumerate(frontier_rows) if str(row["candidate_id"]) == str(reference_candidate_id)), None)
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.2), sharex=True)
    axes[0].plot(x_values, [float(row["validation_primary_loss"]) for row in frontier_rows], marker="o", linewidth=1.4, label="validation primary")
    axes[0].plot(x_values, [float(row["holdout_primary_loss"]) for row in frontier_rows], marker="s", linewidth=1.2, label="holdout primary")
    if ref_index is not None:
        axes[0].axvline(float(ref_index), color="#666666", linestyle="--", linewidth=1.0, alpha=0.7, label="strict reference")
    axes[0].set_title("DIAG-02C Incidence Family Frontier")
    axes[0].set_ylabel("primary loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(x_values, [float(row["validation_diag_flow_loss"]) for row in frontier_rows], marker="o", linewidth=1.4, label="validation diagnosis flow")
    axes[1].plot(x_values, [float(row["holdout_diag_flow_loss"]) for row in frontier_rows], marker="s", linewidth=1.2, label="holdout diagnosis flow")
    if ref_index is not None:
        axes[1].axvline(float(ref_index), color="#666666", linestyle="--", linewidth=1.0, alpha=0.7, label="strict reference")
    axes[1].set_ylabel("diagnosis-flow loss")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(x_labels, rotation=35, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_model_comparison_chart(path: Path, reference_fit: IncidenceFamilyFit, best_fit: IncidenceFamilyFit, baselines: dict[str, Any]) -> None:
    labels = ["carry_forward", "simple_compartmental", "strict_reference", "incidence_best"]
    validation_primary = [
        float(baselines["carry_forward"]["validation"]["primary_loss"]),
        float(baselines["simple_compartmental"]["validation"]["primary_loss"]),
        float(reference_fit.split_metrics["validation"]["primary_loss"]),
        float(best_fit.split_metrics["validation"]["primary_loss"]),
    ]
    holdout_primary = [
        float(baselines["carry_forward"]["holdout"]["primary_loss"]),
        float(baselines["simple_compartmental"]["holdout"]["primary_loss"]),
        float(reference_fit.split_metrics["holdout"]["primary_loss"]),
        float(best_fit.split_metrics["holdout"]["primary_loss"]),
    ]
    validation_diag = [
        float(baselines["carry_forward"]["validation"]["diag_flow_loss"]),
        float(baselines["simple_compartmental"]["validation"]["diag_flow_loss"]),
        float(reference_fit.split_metrics["validation"]["diag_flow_loss"]),
        float(best_fit.split_metrics["validation"]["diag_flow_loss"]),
    ]
    holdout_diag = [
        float(baselines["carry_forward"]["holdout"]["diag_flow_loss"]),
        float(baselines["simple_compartmental"]["holdout"]["diag_flow_loss"]),
        float(reference_fit.split_metrics["holdout"]["diag_flow_loss"]),
        float(best_fit.split_metrics["holdout"]["diag_flow_loss"]),
    ]
    x_values = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.0), sharex=True)
    axes[0].bar(x_values - width / 2, validation_primary, width=width, label="validation")
    axes[0].bar(x_values + width / 2, holdout_primary, width=width, label="holdout")
    axes[0].set_title("DIAG-02C Blocked-Time Primary Loss Comparison")
    axes[0].set_ylabel("primary loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x_values - width / 2, validation_diag, width=width, label="validation")
    axes[1].bar(x_values + width / 2, holdout_diag, width=width, label="holdout")
    axes[1].set_title("DIAG-02C Blocked-Time Diagnosis-Flow Loss Comparison")
    axes[1].set_ylabel("diagnosis-flow loss")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_prediction_chart(path: Path, snapshot: Any, contract: BlockedTimeContract, reference_fit: IncidenceFamilyFit, best_fit: IncidenceFamilyFit) -> None:
    relevant_mask = np.logical_or(contract.validation_mask, contract.holdout_mask)
    indices = np.where(relevant_mask)[0]
    quarter_labels = [snapshot.model_quarters[idx] for idx in indices]
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.5), sharex=True)
    for axis, metric_name, title in (
        (axes[0], "new_diagnosed_cases_period", "New Diagnosed Cases"),
        (axes[1], "diagnosed_plhiv", "Diagnosed PLHIV"),
    ):
        observed_mask = np.logical_and(relevant_mask, snapshot.metric_masks[metric_name])
        axis.plot(
            quarter_labels,
            [float(snapshot.metric_values[metric_name][idx]) if bool(observed_mask[idx]) else np.nan for idx in indices],
            marker="o",
            linewidth=1.6,
            label="observed",
        )
        axis.plot(
            quarter_labels,
            [float(reference_fit.simulation["predictions"][metric_name][idx]) for idx in indices],
            linewidth=1.4,
            label="strict reference",
        )
        axis.plot(
            quarter_labels,
            [float(best_fit.simulation["predictions"][metric_name][idx]) for idx in indices],
            linewidth=1.4,
            label="incidence best",
        )
        axis.set_title(f"DIAG-02C {title} on Validation + Holdout")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xticks(range(len(quarter_labels)))
    axes[-1].set_xticklabels(quarter_labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_incidence_chart(path: Path, snapshot: Any, contract: BlockedTimeContract, reference_fit: IncidenceFamilyFit, best_fit: IncidenceFamilyFit) -> None:
    relevant_mask = np.logical_or(contract.validation_mask, contract.holdout_mask)
    indices = np.where(relevant_mask)[0]
    quarter_labels = [snapshot.model_quarters[idx] for idx in indices]
    reference_share = reference_fit.simulation["incidence_flow"][indices] / np.maximum(snapshot.population_denominator[indices], np.finfo(np.float64).eps)
    best_share = best_fit.simulation["incidence_flow"][indices] / np.maximum(snapshot.population_denominator[indices], np.finfo(np.float64).eps)
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.2), sharex=True)
    axes[0].plot(quarter_labels, reference_fit.simulation["incidence_flow"][indices], linewidth=1.5, label="strict reference")
    axes[0].plot(quarter_labels, best_fit.simulation["incidence_flow"][indices], linewidth=1.5, label="incidence best")
    axes[0].set_title("DIAG-02C Latent Incidence Flow")
    axes[0].set_ylabel("latent inflow")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(quarter_labels, reference_share, linewidth=1.5, label="strict reference")
    axes[1].plot(quarter_labels, best_share, linewidth=1.5, label="incidence best")
    axes[1].set_title("DIAG-02C Latent Incidence Share")
    axes[1].set_ylabel("incidence / denominator")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(range(len(quarter_labels)))
    axes[1].set_xticklabels(quarter_labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _export_paper_figures(ctx: TransitionResearchContext, chart_paths: dict[str, Path]) -> dict[str, Any]:
    archive_dir = ROOT_DIR / "artifacts" / "paper_figures" / "phase3_incidence_module_20260410"
    archive_dir.mkdir(parents=True, exist_ok=True)
    figure_specs = [
        ("fig_diag02c_01", "Incidence Family Frontier", "Validation and holdout losses across blocked-time incidence-family candidates with the strict diagnosis kernel and downstream care held fixed.", "diag02c_incidence_frontier.png", chart_paths.get("frontier")),
        ("fig_diag02c_02", "Incidence Family Model Comparison", "Blocked-time primary-loss and diagnosis-flow-loss comparison against carry-forward, simple compartmental, the strict diagnosis reference, and the best incidence-family branch.", "diag02c_model_comparison.png", chart_paths.get("comparison")),
        ("fig_diag02c_03", "Incidence Family Predictions", "Observed and predicted new diagnoses and diagnosed stock over the blocked validation and holdout periods for the strict reference and the best incidence-family branch.", "diag02c_predictions.png", chart_paths.get("predictions")),
        ("fig_diag02c_04", "Latent Incidence Trajectories", "Model-implied latent incidence inflow and denominator-normalized incidence share over validation and holdout for the strict reference and the best incidence-family branch.", "diag02c_incidence_trajectories.png", chart_paths.get("incidence")),
    ]
    manifest: list[dict[str, Any]] = []
    for figure_id, title, caption, filename, source_path in figure_specs:
        if source_path is None or not source_path.exists():
            continue
        destination = archive_dir / filename
        shutil.copy2(source_path, destination)
        manifest.append(
            {
                "figure_id": figure_id,
                "title": title,
                "caption": caption,
                "source_path": str(source_path),
                "archived_path": str(destination),
            }
        )
    manifest_path = archive_dir / "figure_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "archive_dir": str(archive_dir),
        "figure_manifest": str(manifest_path),
        "figure_count_added": int(len(manifest)),
        "figure_count_total": int(len(manifest)),
    }


def run_diag_02c(ctx: TransitionResearchContext) -> dict[str, Any]:
    from .integrated_autoresearch import _build_snapshot

    snapshot = _build_snapshot(ctx)
    reference_config, reference_candidate = _select_promoted_strict_reference()
    contract = _build_blocked_time_contract(snapshot)
    candidates = _build_family_candidates(reference_candidate, contract)
    fits = [_fit_candidate(snapshot, reference_config, candidate, contract) for candidate in candidates]
    frontier_rows = _frontier_rows(fits)
    baselines = _baseline_payload(snapshot, contract)

    reference_fit = next(
        fit
        for fit in fits
        if fit.candidate.incidence_family == "level" and fit.candidate.use_direct_covariates
    )
    valid_fits = [fit for fit in fits if _fit_is_valid(fit)]
    if not valid_fits:
        raise RuntimeError("DIAG-02C produced no valid incidence-family fits")
    best_fit = min(valid_fits, key=lambda fit: _score_tuple(fit.split_metrics["validation"]))

    contract_path = ctx.experiment_dir / "incidence_family_contract.json"
    frontier_path = ctx.experiment_dir / "incidence_family_frontier.json"
    evaluation_path = ctx.experiment_dir / "incidence_family_evaluation.json"
    comparison_path = ctx.experiment_dir / "incidence_family_model_comparison.json"
    prediction_rows_path = ctx.experiment_dir / "incidence_family_prediction_rows.json"
    dashboard_path = ctx.experiment_dir / "incidence_family_dashboard.md"
    frontier_chart_path = ctx.experiment_dir / "incidence_family_frontier.png"
    comparison_chart_path = ctx.experiment_dir / "incidence_family_model_comparison.png"
    prediction_chart_path = ctx.experiment_dir / "incidence_family_predictions.png"
    incidence_chart_path = ctx.experiment_dir / "incidence_family_trajectories.png"

    contract_payload = {
        "generated_at": utc_now_iso(),
        "reference_candidate_id": str(reference_candidate.candidate_id),
        "reference_integrated_candidate_id": str(reference_config.candidate_id),
        "train_diagnosis_quarters": list(contract.train_diagnosis_quarters),
        "validation_quarters": list(contract.validation_quarters),
        "holdout_quarters": list(contract.holdout_quarters),
        "incidence_family_count": int(len(candidates)),
        "incidence_lag_limit": max(1, int(len(contract.train_diagnosis_quarters)) - 1),
    }
    write_json(contract_path, contract_payload)
    write_json(frontier_path, frontier_rows)
    write_json(
        comparison_path,
        {
            "carry_forward": baselines["carry_forward"],
            "simple_compartmental": baselines["simple_compartmental"],
            "strict_reference": _fit_payload(reference_fit),
            "incidence_best": _fit_payload(best_fit),
        },
    )

    relevant_mask = np.logical_or(contract.validation_mask, contract.holdout_mask)
    prediction_rows: list[dict[str, Any]] = []
    for idx in np.where(relevant_mask)[0]:
        prediction_rows.append(
            {
                "quarter": str(snapshot.model_quarters[idx]),
                "window": "validation" if bool(contract.validation_mask[idx]) else "holdout",
                "observed_new_diagnosed_cases_period": float(snapshot.metric_values["new_diagnosed_cases_period"][idx]) if bool(snapshot.metric_masks["new_diagnosed_cases_period"][idx]) else None,
                "reference_new_diagnosed_cases_period": float(reference_fit.simulation["predictions"]["new_diagnosed_cases_period"][idx]),
                "best_new_diagnosed_cases_period": float(best_fit.simulation["predictions"]["new_diagnosed_cases_period"][idx]),
                "observed_diagnosed_plhiv": float(snapshot.metric_values["diagnosed_plhiv"][idx]) if bool(snapshot.metric_masks["diagnosed_plhiv"][idx]) else None,
                "reference_diagnosed_plhiv": float(reference_fit.simulation["predictions"]["diagnosed_plhiv"][idx]),
                "best_diagnosed_plhiv": float(best_fit.simulation["predictions"]["diagnosed_plhiv"][idx]),
                "reference_incidence_flow": float(reference_fit.simulation["incidence_flow"][idx]),
                "best_incidence_flow": float(best_fit.simulation["incidence_flow"][idx]),
            }
        )
    write_json(prediction_rows_path, prediction_rows)

    _write_frontier_chart(frontier_chart_path, frontier_rows, reference_fit.candidate.candidate_id)
    _write_model_comparison_chart(comparison_chart_path, reference_fit, best_fit, baselines)
    _write_prediction_chart(prediction_chart_path, snapshot, contract, reference_fit, best_fit)
    _write_incidence_chart(incidence_chart_path, snapshot, contract, reference_fit, best_fit)
    paper_archive = _export_paper_figures(
        ctx,
        {
            "frontier": frontier_chart_path,
            "comparison": comparison_chart_path,
            "predictions": prediction_chart_path,
            "incidence": incidence_chart_path,
        },
    )

    dashboard_lines = [
        "# DIAG-02C Blocked-Time Incidence Family Search Dashboard",
        "",
        "This experiment freezes the promoted strict diagnosis kernel as the reference, keeps downstream care fixed, and reworks only the incidence module under the blocked-time contract.",
        "",
        f"- Best incidence candidate: `{best_fit.candidate.candidate_id}`",
        f"- Reference candidate: `{reference_fit.candidate.candidate_id}`",
        f"- Validation primary delta vs reference: `{float(best_fit.split_metrics['validation']['primary_loss'] - reference_fit.split_metrics['validation']['primary_loss']):.6f}`",
        f"- Holdout primary delta vs reference: `{float(best_fit.split_metrics['holdout']['primary_loss'] - reference_fit.split_metrics['holdout']['primary_loss']):.6f}`",
    ]
    dashboard_path.write_text("\n".join(dashboard_lines), encoding="utf-8")

    evaluation_payload = {
        "reference_candidate_id": str(reference_fit.candidate.candidate_id),
        "best_candidate_id": str(best_fit.candidate.candidate_id),
        "strict_reference": _fit_payload(reference_fit),
        "incidence_best": _fit_payload(best_fit),
        "best_vs_reference_validation_delta": {
            "primary_loss": float(best_fit.split_metrics["validation"]["primary_loss"] - reference_fit.split_metrics["validation"]["primary_loss"]),
            "diag_flow_loss": float(best_fit.split_metrics["validation"]["diag_flow_loss"] - reference_fit.split_metrics["validation"]["diag_flow_loss"]),
        },
        "best_vs_reference_holdout_delta": {
            "primary_loss": float(best_fit.split_metrics["holdout"]["primary_loss"] - reference_fit.split_metrics["holdout"]["primary_loss"]),
            "diag_flow_loss": float(best_fit.split_metrics["holdout"]["diag_flow_loss"] - reference_fit.split_metrics["holdout"]["diag_flow_loss"]),
        },
        "best_vs_carry_forward_validation_delta": {
            "primary_loss": float(best_fit.split_metrics["validation"]["primary_loss"] - baselines["carry_forward"]["validation"]["primary_loss"]),
            "diag_flow_loss": float(best_fit.split_metrics["validation"]["diag_flow_loss"] - baselines["carry_forward"]["validation"]["diag_flow_loss"]),
        },
        "best_vs_carry_forward_holdout_delta": {
            "primary_loss": float(best_fit.split_metrics["holdout"]["primary_loss"] - baselines["carry_forward"]["holdout"]["primary_loss"]),
            "diag_flow_loss": float(best_fit.split_metrics["holdout"]["diag_flow_loss"] - baselines["carry_forward"]["holdout"]["diag_flow_loss"]),
        },
        "paper_archive": paper_archive,
    }
    write_json(evaluation_path, evaluation_payload)

    decision = {
        "reference_candidate_id": str(reference_fit.candidate.candidate_id),
        "best_candidate_id": str(best_fit.candidate.candidate_id),
        "best_validation_beats_reference": bool(_score_tuple(best_fit.split_metrics["validation"]) < _score_tuple(reference_fit.split_metrics["validation"])),
        "best_holdout_beats_reference": bool(_score_tuple(best_fit.split_metrics["holdout"]) < _score_tuple(reference_fit.split_metrics["holdout"])),
        "best_validation_beats_carry_forward": bool(_score_tuple(best_fit.split_metrics["validation"]) < _score_tuple(baselines["carry_forward"]["validation"])),
        "best_holdout_beats_carry_forward": bool(_score_tuple(best_fit.split_metrics["holdout"]) < _score_tuple(baselines["carry_forward"]["holdout"])),
        "promote_incidence_branch": bool(
            _score_tuple(best_fit.split_metrics["validation"]) < _score_tuple(reference_fit.split_metrics["validation"])
            and _score_tuple(best_fit.split_metrics["holdout"]) < _score_tuple(reference_fit.split_metrics["holdout"])
        ),
        "paper_figure_archive": paper_archive,
    }

    experiment_spec = {
        "variant": "evidence-to-model-loop",
        "goal": "Freeze the promoted strict diagnosis branch as the reference, keep downstream care fixed, and search stronger incidence-module families under the blocked-time contract.",
        "source_run_id": ctx.source_run_id,
        "reference_candidate_id": str(reference_fit.candidate.candidate_id),
        "mutation_unit": "incidence-module family only",
        "blocked_time_contract": contract_payload,
        "artifacts": {
            "incidence_family_contract": str(contract_path),
            "incidence_family_frontier": str(frontier_path),
            "incidence_family_evaluation": str(evaluation_path),
            "incidence_family_model_comparison": str(comparison_path),
            "incidence_family_prediction_rows": str(prediction_rows_path),
            "incidence_family_dashboard": str(dashboard_path),
        },
    }
    coverage_summary = {
        "historical_quarter_count": int(len(snapshot.historical_quarters)),
        "train_quarter_count": int(np.sum(contract.train_mask)),
        "validation_quarter_count": int(np.sum(contract.validation_mask)),
        "holdout_quarter_count": int(np.sum(contract.holdout_mask)),
        "incidence_family_count": int(len(candidates)),
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            numerical_guard_entry(name="diag02c_incidence_family_eps", role="diag02c_numerical_guard", why_needed="Prevents divide-by-zero and log singularities while fitting recursive blocked-time incidence-family candidates."),
            {
                "name": "incidence_lag_limit",
                "value": int(max(1, len(contract.train_diagnosis_quarters) - 1)),
                "role": "diag02c_incidence_family_budget",
                "source_type": "estimated",
                "estimation_data": "Blocked-time train diagnosis quarter count",
                "estimation_method": "Set the maximum recursive incidence lag count to train diagnosis quarter count minus one, floored at one",
                "uncertainty": "deterministic given the frozen blocked-time contract",
                "why_needed": "Bounds recursive incidence families by observed front-half support instead of an ad hoc fixed lag ceiling.",
            },
            {
                "name": "incidence_family_count",
                "value": int(len(candidates)),
                "role": "diag02c_incidence_family_search_budget",
                "source_type": "estimated",
                "estimation_data": "Blocked-time lag limit and the fixed family ladder {level, renewal, diagnosis_feedback} x {direct on, direct off}",
                "estimation_method": "Enumerate the bounded family set implied by the frozen strict diagnosis reference and blocked-time lag limit",
                "uncertainty": "deterministic given the frozen contract",
                "why_needed": "Makes the incidence-family search budget explicit and reproducible.",
            },
        ],
    )
    return {
        "artifacts": artifacts,
        "incidence_family_contract": contract_payload,
        "frontier_rows": frontier_rows,
        "evaluation": evaluation_payload,
        "decision": decision,
        "paper_archive": paper_archive,
    }


__all__ = ["run_diag_02c"]
