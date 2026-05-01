from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from epigraph_ph.phase3._lineage.national_reset_core import quarter_sort_key
from epigraph_ph.runtime import ROOT_DIR, read_json, utc_now_iso, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .integrated_autoresearch import (
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    STATE_NAMES,
    TOTAL_METRIC,
    CandidateConfig,
    IntegratedSnapshot,
    TRANSITION_NAMES,
    _build_snapshot,
    _candidate_matrices,
    _cloglog_inverse,
    _compute_window_metrics,
    _fit_residuals,
    _historical_mask,
    _module_linear_predictor,
    _safe_exp,
    _select_matrix,
    _sigmoid,
)
from .numeric_policy import numerical_guard_entry
from .strict_spec_gap_audit import _latest_integrated_experiment_dir, _select_reference_configs


@dataclass(slots=True)
class DiagnosisKernelCandidate:
    candidate_id: str
    diagnosis_kind: str
    care_family: str
    kernel_width: int
    reference_hidden_rank: int
    use_observation_covariates: bool


@dataclass(slots=True)
class BlockedTimeContract:
    diagnosis_flow_observed_quarters: list[str]
    train_diagnosis_quarters: list[str]
    validation_quarters: list[str]
    holdout_quarters: list[str]
    train_end_quarter: str
    validation_start_quarter: str
    validation_end_quarter: str
    holdout_start_quarter: str
    holdout_end_quarter: str
    train_mask: np.ndarray
    validation_mask: np.ndarray
    holdout_mask: np.ndarray


@dataclass(slots=True)
class DiagnosisKernelFit:
    candidate: DiagnosisKernelCandidate
    budget_multiplier: float
    success: bool
    status: int
    message: str
    train_cost: float
    param_vector: np.ndarray
    param_slices: dict[str, slice]
    matrices: dict[str, Any]
    simulation: dict[str, Any]
    split_metrics: dict[str, dict[str, float]]
    age_profile: np.ndarray
    train_bic: float
    nfev: int
    max_nfev: int
    wall_seconds: float


_INTEGRATED_CANDIDATE_PATTERN = re.compile(
    r"^diag-(?P<diagnosis>hazard|delay)-care-(?P<care>markov|semi_markov)-h(?P<hidden>\d+)-obs(?P<obs>on|off)$"
)
_DIAG_CANDIDATE_PATTERN = re.compile(
    r"^diag-(?:(?P<kind>hazard|delay)|strictw(?P<width>\d+))-care-(?P<care>markov|semi_markov)-h(?P<hidden>\d+)-obs(?P<obs>on|off)$"
)


def _age_basis_log(age_count: int) -> np.ndarray:
    return np.log1p(np.arange(age_count, dtype=np.float64))


def _age_basis_dct(age_count: int, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros((age_count, 0), dtype=np.float64)
    age_index = np.arange(age_count, dtype=np.float64)
    basis = np.zeros((age_count, width), dtype=np.float64)
    for col in range(width):
        frequency = float(col + 1)
        basis[:, col] = np.cos(np.pi * frequency * (age_index + 0.5) / float(age_count))
    norms = np.linalg.norm(basis, axis=0)
    norms = np.where(norms > np.finfo(np.float64).eps, norms, 1.0)
    return basis / norms[None, :]


def _quarter_mask_between(snapshot: IntegratedSnapshot, start_quarter: str, end_quarter: str) -> np.ndarray:
    return np.asarray(
        [
            quarter_sort_key(start_quarter) <= quarter_sort_key(quarter) <= quarter_sort_key(end_quarter)
            for quarter in snapshot.model_quarters
        ],
        dtype=bool,
    )


def _build_blocked_time_contract(snapshot: IntegratedSnapshot) -> BlockedTimeContract:
    diagnosis_mask = np.asarray(snapshot.metric_masks["new_diagnosed_cases_period"], dtype=bool)
    diagnosis_quarters = [
        quarter
        for quarter in snapshot.historical_quarters
        if bool(diagnosis_mask[snapshot.quarter_index[quarter]])
    ]
    if len(diagnosis_quarters) < 3:
        raise ValueError("Strict diagnosis blocked-time experiment requires at least three observed diagnosis-flow quarters")
    split_blocks = [list(block.astype(str)) for block in np.array_split(np.asarray(diagnosis_quarters, dtype=object), 3)]
    if any(len(block) == 0 for block in split_blocks):
        raise ValueError("Blocked-time diagnosis-flow split produced an empty contiguous block")
    train_diag_quarters, validation_quarters, holdout_quarters = split_blocks
    train_end_quarter = str(train_diag_quarters[-1])
    validation_start_quarter = str(validation_quarters[0])
    validation_end_quarter = str(validation_quarters[-1])
    holdout_start_quarter = str(holdout_quarters[0])
    holdout_end_quarter = str(holdout_quarters[-1])
    historical_mask = _historical_mask(snapshot)
    train_mask = np.asarray(
        [
            bool(historical_mask[idx]) and quarter_sort_key(quarter) <= quarter_sort_key(train_end_quarter)
            for idx, quarter in enumerate(snapshot.model_quarters)
        ],
        dtype=bool,
    )
    validation_mask = np.logical_and(historical_mask, _quarter_mask_between(snapshot, validation_start_quarter, validation_end_quarter))
    holdout_mask = np.logical_and(historical_mask, _quarter_mask_between(snapshot, holdout_start_quarter, holdout_end_quarter))
    return BlockedTimeContract(
        diagnosis_flow_observed_quarters=list(diagnosis_quarters),
        train_diagnosis_quarters=[str(value) for value in train_diag_quarters],
        validation_quarters=[str(value) for value in validation_quarters],
        holdout_quarters=[str(value) for value in holdout_quarters],
        train_end_quarter=train_end_quarter,
        validation_start_quarter=validation_start_quarter,
        validation_end_quarter=validation_end_quarter,
        holdout_start_quarter=holdout_start_quarter,
        holdout_end_quarter=holdout_end_quarter,
        train_mask=train_mask,
        validation_mask=validation_mask,
        holdout_mask=holdout_mask,
    )


def _build_candidates(reference_config: CandidateConfig, contract: BlockedTimeContract) -> list[DiagnosisKernelCandidate]:
    width_limit = max(int(len(contract.train_diagnosis_quarters)), 1)
    candidates = [
        DiagnosisKernelCandidate(
            candidate_id=f"diag-hazard-care-{reference_config.care_family}-h{reference_config.hidden_rank:02d}-obs{'on' if reference_config.use_observation_covariates else 'off'}",
            diagnosis_kind="hazard",
            care_family=str(reference_config.care_family),
            kernel_width=0,
            reference_hidden_rank=int(reference_config.hidden_rank),
            use_observation_covariates=bool(reference_config.use_observation_covariates),
        ),
        DiagnosisKernelCandidate(
            candidate_id=f"diag-delay-care-{reference_config.care_family}-h{reference_config.hidden_rank:02d}-obs{'on' if reference_config.use_observation_covariates else 'off'}",
            diagnosis_kind="delay",
            care_family=str(reference_config.care_family),
            kernel_width=1,
            reference_hidden_rank=int(reference_config.hidden_rank),
            use_observation_covariates=bool(reference_config.use_observation_covariates),
        ),
    ]
    for width in range(2, width_limit + 1):
        candidates.append(
            DiagnosisKernelCandidate(
                candidate_id=f"diag-strictw{width:02d}-care-{reference_config.care_family}-h{reference_config.hidden_rank:02d}-obs{'on' if reference_config.use_observation_covariates else 'off'}",
                diagnosis_kind="strict",
                care_family=str(reference_config.care_family),
                kernel_width=int(width),
                reference_hidden_rank=int(reference_config.hidden_rank),
                use_observation_covariates=bool(reference_config.use_observation_covariates),
            )
        )
    return candidates


def _parse_integrated_candidate_config(candidate_id: str) -> CandidateConfig | None:
    match = _INTEGRATED_CANDIDATE_PATTERN.match(str(candidate_id))
    if match is None:
        return None
    return CandidateConfig(
        candidate_id=str(candidate_id),
        diagnosis_family=str(match.group("diagnosis")),
        care_family=str(match.group("care")),
        hidden_rank=int(match.group("hidden")),
        use_observation_covariates=str(match.group("obs")) == "on",
    )


def _select_integrated_stage1_champion_config() -> CandidateConfig:
    experiment_dir = _latest_integrated_experiment_dir()
    if experiment_dir is None:
        raise FileNotFoundError("No PHASE3-V2-INT experiment directory was found under artifacts/runs")
    decision = read_json(experiment_dir / "decision.json", default={})
    candidate_id = str(decision.get("stage1_champion_candidate_id") or "")
    config = _parse_integrated_candidate_config(candidate_id)
    if config is None:
        raise ValueError(f"Could not parse PHASE3-V2-INT stage1 champion candidate id: {candidate_id}")
    return config


def _parse_diag_candidate(candidate_id: str) -> DiagnosisKernelCandidate | None:
    match = _DIAG_CANDIDATE_PATTERN.match(str(candidate_id))
    if match is None:
        return None
    kind = str(match.group("kind") or "strict")
    if kind == "hazard":
        kernel_width = 0
    elif kind == "delay":
        kernel_width = 1
    else:
        kernel_width = int(match.group("width") or 0)
    return DiagnosisKernelCandidate(
        candidate_id=str(candidate_id),
        diagnosis_kind=kind,
        care_family=str(match.group("care")),
        kernel_width=int(kernel_width),
        reference_hidden_rank=int(match.group("hidden")),
        use_observation_covariates=str(match.group("obs")) == "on",
    )


def _latest_diag02a_experiment_dir() -> Path | None:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "transition_research" / "DIAG-02A-integrated-champion-strict-promotion"
        decision_path = experiment_dir / "decision.json"
        if decision_path.exists():
            candidates.append((decision_path.stat().st_mtime, experiment_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item[0]))[1]


def _select_promoted_strict_reference() -> tuple[CandidateConfig, DiagnosisKernelCandidate]:
    experiment_dir = _latest_diag02a_experiment_dir()
    if experiment_dir is None:
        raise FileNotFoundError("No DIAG-02A experiment directory was found under artifacts/runs")
    decision = read_json(experiment_dir / "decision.json", default={})
    reference_id = str(decision.get("reference_candidate_id") or "")
    promoted_id = str(decision.get("promoted_candidate_id") or "")
    reference_config = _parse_integrated_candidate_config(reference_id)
    promoted_candidate = _parse_diag_candidate(promoted_id)
    if reference_config is None:
        raise ValueError(f"Could not parse DIAG-02A reference candidate id: {reference_id}")
    if promoted_candidate is None:
        raise ValueError(f"Could not parse DIAG-02A promoted candidate id: {promoted_id}")
    return reference_config, promoted_candidate


def _reference_candidate_config(reference_config: CandidateConfig) -> CandidateConfig:
    return CandidateConfig(
        candidate_id=str(reference_config.candidate_id),
        diagnosis_family=str(reference_config.diagnosis_family),
        care_family=str(reference_config.care_family),
        hidden_rank=int(reference_config.hidden_rank),
        use_observation_covariates=bool(reference_config.use_observation_covariates),
    )


def _matrices_for_reference(snapshot: IntegratedSnapshot, reference_config: CandidateConfig) -> dict[str, Any]:
    return _candidate_matrices(snapshot, _reference_candidate_config(reference_config))


def _build_param_slices(candidate: DiagnosisKernelCandidate, matrices: dict[str, Any]) -> dict[str, slice]:
    cursor = 0
    slices: dict[str, slice] = {}
    slices["init_u_log"] = slice(cursor, cursor + 1)
    cursor += 1
    slices["init_art_share"] = slice(cursor, cursor + 1)
    cursor += 1
    slices["init_v_share"] = slice(cursor, cursor + 1)
    cursor += 1
    slices["init_l_share"] = slice(cursor, cursor + 1)
    cursor += 1
    incidence_width = 1 + matrices["incidence_direct"].shape[1] + matrices["hidden"].shape[1]
    slices["incidence"] = slice(cursor, cursor + incidence_width)
    cursor += incidence_width
    for transition in TRANSITION_NAMES:
        width = 1 + matrices["transition_direct"][transition].shape[1] + matrices["hidden"].shape[1]
        if transition == "U_to_D":
            if candidate.diagnosis_kind == "delay":
                width += 1
            elif candidate.diagnosis_kind == "strict":
                width += int(candidate.kernel_width)
        elif candidate.care_family == "semi_markov":
            width += 1
        slices[f"transition::{transition}"] = slice(cursor, cursor + width)
        cursor += width
    for metric_name in ("tested_for_viral_load", "virally_suppressed"):
        width = 1 + matrices["observation_direct"][metric_name].shape[1] + matrices["hidden"].shape[1]
        slices[f"observation::{metric_name}"] = slice(cursor, cursor + width)
        cursor += width
    slices["full"] = slice(0, cursor)
    return slices


def _initial_param_vector(
    snapshot: IntegratedSnapshot,
    candidate: DiagnosisKernelCandidate,
    matrices: dict[str, Any],
    param_slices: dict[str, slice],
    train_mask: np.ndarray,
) -> np.ndarray:
    eps = float(np.finfo(np.float64).eps)
    params = np.zeros((param_slices["full"].stop,), dtype=np.float64)

    initial_total = float(snapshot.start_total)
    initial_diag = float(snapshot.start_diag)
    initial_undiagnosed = max(initial_total - initial_diag, eps)
    params[param_slices["init_u_log"]] = np.log(initial_undiagnosed)

    art_share_mask = np.logical_and(
        train_mask,
        np.logical_and(snapshot.metric_masks["alive_on_art"], np.logical_and(snapshot.metric_masks["diagnosed_plhiv"], snapshot.metric_values["diagnosed_plhiv"] > 0.0)),
    )
    if not np.any(art_share_mask):
        raise ValueError("Strict diagnosis kernel experiment requires at least one diagnosed-plus-ART observation in training")
    initial_art_share = float(
        np.clip(
            snapshot.metric_values["alive_on_art"][art_share_mask][0] / max(snapshot.metric_values["diagnosed_plhiv"][art_share_mask][0], eps),
            eps,
            1.0 - eps,
        )
    )
    params[param_slices["init_art_share"]] = np.log(initial_art_share / max(1.0 - initial_art_share, eps))

    initial_vs_mask = np.logical_and(train_mask, np.logical_and(snapshot.metric_masks["virally_suppressed"], snapshot.metric_masks["alive_on_art"]))
    if not np.any(initial_vs_mask):
        raise ValueError("Strict diagnosis kernel experiment requires at least one virally-suppressed observation in training")
    initial_v_share = float(
        np.clip(
            snapshot.metric_values["virally_suppressed"][initial_vs_mask][0] / max(snapshot.metric_values["alive_on_art"][initial_vs_mask][0], eps),
            eps,
            1.0 - eps,
        )
    )
    params[param_slices["init_v_share"]] = np.log(initial_v_share / max(1.0 - initial_v_share, eps))
    params[param_slices["init_l_share"]] = 0.0

    total_observed_mask = np.logical_and(train_mask, snapshot.metric_masks[TOTAL_METRIC])
    total_indices = np.where(total_observed_mask)[0]
    total_pred = snapshot.metric_values[TOTAL_METRIC][total_observed_mask]
    if total_pred.size >= 2:
        total_deltas = np.diff(total_pred)
        denominator = snapshot.population_denominator[total_indices[1:]]
        incidence_hazard = np.mean(np.maximum(total_deltas, 0.0) / np.maximum(denominator, eps))
    else:
        diag_train_mask = np.logical_and(train_mask, snapshot.metric_masks["new_diagnosed_cases_period"])
        incidence_hazard = np.mean(
            np.nan_to_num(snapshot.metric_values["new_diagnosed_cases_period"][diag_train_mask], nan=0.0)
            / np.maximum(snapshot.population_denominator[diag_train_mask], eps)
        )
    incidence_hazard = max(float(incidence_hazard), eps)
    params[param_slices["incidence"]][0] = np.log(incidence_hazard)

    newdiag_mask = np.logical_and(train_mask, snapshot.metric_masks["new_diagnosed_cases_period"])
    total_mask = np.logical_and(train_mask, snapshot.metric_masks[TOTAL_METRIC])
    diag_mask = np.logical_and(train_mask, snapshot.metric_masks["diagnosed_plhiv"])
    art_mask = np.logical_and(train_mask, snapshot.metric_masks["alive_on_art"])
    newdiag_values = snapshot.metric_values["new_diagnosed_cases_period"][newdiag_mask]
    total_values = snapshot.metric_values[TOTAL_METRIC][total_mask]
    diag_values = snapshot.metric_values["diagnosed_plhiv"][diag_mask]
    art_values = snapshot.metric_values["alive_on_art"][art_mask]
    if newdiag_values.size and total_values.size and diag_values.size:
        overlap = max(1, min(len(newdiag_values), len(total_values), len(diag_values)))
        u_stock = np.maximum(total_values[:overlap] - diag_values[:overlap], 0.0)
        u_hazard = np.mean(newdiag_values[:overlap] / np.maximum(u_stock, eps))
    else:
        u_hazard = eps
    if art_values.size >= 2 and diag_values.size >= 2:
        art_increase = np.maximum(np.diff(art_values), 0.0)
        overlap = max(1, min(len(art_increase), len(diag_values), len(art_values)))
        diag_gap = np.maximum(diag_values[:overlap] - art_values[:overlap], 0.0)
        d_to_a_hazard = np.mean(art_increase[:overlap] / np.maximum(diag_gap, eps))
    else:
        d_to_a_hazard = eps
    vs_values = snapshot.metric_values["virally_suppressed"][np.logical_and(train_mask, snapshot.metric_masks["virally_suppressed"])]
    if vs_values.size >= 2 and art_values.size >= 2:
        vs_increase = np.maximum(np.diff(vs_values), 0.0)
        overlap = max(1, min(len(vs_increase), len(art_values)))
        art_base = np.maximum(art_values[:overlap], eps)
        a_to_v_hazard = np.mean(vs_increase[:overlap] / art_base)
    else:
        a_to_v_hazard = eps
    transition_starts = {
        "U_to_D": max(float(u_hazard), eps),
        "D_to_A": max(float(d_to_a_hazard), eps),
        "A_to_V": max(float(a_to_v_hazard), eps),
        "A_to_L": eps,
        "L_to_A": eps,
    }
    for transition, hazard in transition_starts.items():
        transition_slice = param_slices[f"transition::{transition}"]
        transition_params = params[transition_slice]
        if transition == "U_to_D":
            if candidate.diagnosis_kind == "hazard":
                transition_params[0] = np.log(-np.log(max(1.0 - min(hazard, 1.0 - eps), eps)))
            else:
                bounded = min(max(hazard, eps), 1.0 - eps)
                transition_params[0] = np.log(bounded / max(1.0 - bounded, eps))
        else:
            transition_params[0] = np.log(-np.log(max(1.0 - min(hazard, 1.0 - eps), eps)))
            if candidate.care_family == "semi_markov" and transition_params.size >= 2:
                transition_params[1] = 0.0
        params[transition_slice] = transition_params

    art_nonzero_mask = np.logical_and(train_mask, np.logical_and(snapshot.metric_masks["alive_on_art"], snapshot.metric_values["alive_on_art"] > 0.0))
    test_mask = np.logical_and(np.logical_and(train_mask, snapshot.metric_masks["tested_for_viral_load"]), art_nonzero_mask)
    if not np.any(test_mask):
        raise ValueError("Strict diagnosis kernel experiment requires at least one VL-testing observation in training")
    test_share = np.mean(snapshot.metric_values["tested_for_viral_load"][test_mask] / snapshot.metric_values["alive_on_art"][test_mask])
    test_share = float(np.clip(test_share, eps, 1.0 - eps))
    params[param_slices["observation::tested_for_viral_load"]][0] = np.log(test_share / max(1.0 - test_share, eps))

    test_nonzero_mask = np.logical_and(train_mask, np.logical_and(snapshot.metric_masks["tested_for_viral_load"], snapshot.metric_values["tested_for_viral_load"] > 0.0))
    doc_mask = np.logical_and(np.logical_and(train_mask, snapshot.metric_masks["virally_suppressed"]), test_nonzero_mask)
    if not np.any(doc_mask):
        raise ValueError("Strict diagnosis kernel experiment requires at least one documented-suppression observation in training")
    doc_share = np.mean(snapshot.metric_values["virally_suppressed"][doc_mask] / snapshot.metric_values["tested_for_viral_load"][doc_mask])
    doc_share = float(np.clip(doc_share, eps, 1.0 - eps))
    params[param_slices["observation::virally_suppressed"]][0] = np.log(doc_share / max(1.0 - doc_share, eps))
    return params


def _age_profile(candidate: DiagnosisKernelCandidate, transition_params: np.ndarray, age_count: int) -> np.ndarray:
    if candidate.diagnosis_kind == "hazard":
        return np.zeros((age_count,), dtype=np.float64)
    if candidate.diagnosis_kind == "delay":
        if transition_params.size < 2:
            return np.zeros((age_count,), dtype=np.float64)
        return float(transition_params[1]) * _age_basis_log(age_count)
    basis = _age_basis_dct(age_count, int(candidate.kernel_width))
    return np.asarray(basis @ np.asarray(transition_params[1 : 1 + candidate.kernel_width], dtype=np.float64), dtype=np.float64)


def _age_shift(values: np.ndarray) -> np.ndarray:
    shifted = np.zeros_like(values, dtype=np.float64)
    if values.size == 0:
        return shifted
    if values.size > 1:
        shifted[1:] = values[:-1]
    shifted[-1] += float(values[-1])
    return shifted


def _simulate_candidate(
    snapshot: IntegratedSnapshot,
    candidate: DiagnosisKernelCandidate,
    fit_vector: np.ndarray,
    param_slices: dict[str, slice],
    matrices: dict[str, Any],
) -> dict[str, Any]:
    eps = float(np.finfo(np.float64).eps)
    quarter_count = len(snapshot.model_quarters)
    age_count = quarter_count
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
    incidence_intercept, _inc_age_slope, incidence_direct, incidence_hidden = _module_linear_predictor(
        incidence_params,
        np.asarray(matrices["incidence_direct"], dtype=np.float64),
        np.asarray(matrices["hidden"], dtype=np.float64),
    )
    eta_map["incidence"] = incidence_intercept + incidence_direct + incidence_hidden
    direct_component_map["incidence"] = incidence_direct
    hidden_component_map["incidence"] = incidence_hidden
    incidence_flow = snapshot.population_denominator * np.asarray(_safe_exp(eta_map["incidence"]), dtype=np.float64)

    diagnosis_age_profile = np.zeros((age_count,), dtype=np.float64)
    for transition in TRANSITION_NAMES:
        transition_params = np.asarray(fit_vector[param_slices[f"transition::{transition}"]], dtype=np.float64)
        if transition == "U_to_D":
            intercept = np.full((quarter_count,), float(transition_params[0]), dtype=np.float64)
            cursor = 1
            if candidate.diagnosis_kind == "delay":
                diagnosis_age_profile = _age_profile(candidate, transition_params, age_count)
                cursor += 1
            elif candidate.diagnosis_kind == "strict":
                diagnosis_age_profile = _age_profile(candidate, transition_params, age_count)
                cursor += int(candidate.kernel_width)
            direct_width = matrices["transition_direct"][transition].shape[1]
            hidden_width = matrices["hidden"].shape[1]
            direct_beta = (
                np.asarray(transition_params[cursor : cursor + direct_width], dtype=np.float64)
                if direct_width
                else np.zeros((0,), dtype=np.float64)
            )
            cursor += direct_width
            hidden_beta = (
                np.asarray(transition_params[cursor : cursor + hidden_width], dtype=np.float64)
                if hidden_width
                else np.zeros((0,), dtype=np.float64)
            )
            direct_component = matrices["transition_direct"][transition] @ direct_beta if direct_width else np.zeros((quarter_count,), dtype=np.float64)
            hidden_component = matrices["hidden"] @ hidden_beta if hidden_width else np.zeros((quarter_count,), dtype=np.float64)
            eta_map[transition] = intercept + direct_component + hidden_component
            direct_component_map[transition] = direct_component
            hidden_component_map[transition] = hidden_component
            if candidate.diagnosis_kind == "hazard":
                hazard_map[transition] = np.asarray(_cloglog_inverse(eta_map[transition]), dtype=np.float64)
        else:
            intercept_component, age_slope, direct_component, hidden_component = _module_linear_predictor(
                transition_params,
                np.asarray(matrices["transition_direct"][transition], dtype=np.float64),
                np.asarray(matrices["hidden"], dtype=np.float64),
                include_age_slope=candidate.care_family == "semi_markov",
            )
            eta_map[transition] = intercept_component + direct_component + hidden_component
            direct_component_map[transition] = direct_component
            hidden_component_map[transition] = hidden_component
            age_slope_map[transition] = float(age_slope)
            if candidate.care_family != "semi_markov":
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

    for idx in range(1, quarter_count):
        u_prev = float(np.sum(u_age))
        d_prev = float(np.sum(d_age))
        a_prev = float(np.sum(a_age))
        l_prev = float(np.sum(l_age))

        if candidate.diagnosis_kind == "hazard":
            diagnosed_by_age = np.minimum(u_age, hazard_map["U_to_D"][idx] * u_age)
        else:
            diagnosis_prob = np.asarray(_sigmoid(eta_map["U_to_D"][idx] + diagnosis_age_profile), dtype=np.float64)
            diagnosed_by_age = np.minimum(u_age, diagnosis_prob * u_age)
        u_to_d = float(np.sum(diagnosed_by_age))
        u_age_next = _age_shift(np.maximum(u_age - diagnosed_by_age, 0.0))
        u_age_next[0] += float(incidence_flow[idx])

        d_to_a_hazard = (
            np.asarray(_cloglog_inverse(eta_map["D_to_A"][idx] + age_slope_map.get("D_to_A", 0.0) * _age_basis_log(age_count)), dtype=np.float64)
            if candidate.care_family == "semi_markov"
            else np.full((age_count,), float(hazard_map["D_to_A"][idx]), dtype=np.float64)
        )
        d_to_a_by_age = np.minimum(d_age, d_to_a_hazard * d_age)
        d_to_a = float(np.sum(d_to_a_by_age))
        d_age_next = _age_shift(np.maximum(d_age - d_to_a_by_age, 0.0))
        d_age_next[0] += u_to_d

        a_to_v_hazard = (
            np.asarray(_cloglog_inverse(eta_map["A_to_V"][idx] + age_slope_map.get("A_to_V", 0.0) * _age_basis_log(age_count)), dtype=np.float64)
            if candidate.care_family == "semi_markov"
            else np.full((age_count,), float(hazard_map["A_to_V"][idx]), dtype=np.float64)
        )
        a_to_l_hazard = (
            np.asarray(_cloglog_inverse(eta_map["A_to_L"][idx] + age_slope_map.get("A_to_L", 0.0) * _age_basis_log(age_count)), dtype=np.float64)
            if candidate.care_family == "semi_markov"
            else np.full((age_count,), float(hazard_map["A_to_L"][idx]), dtype=np.float64)
        )
        a_to_v_by_age = np.minimum(a_age, a_to_v_hazard * a_age)
        a_after_v = np.maximum(a_age - a_to_v_by_age, 0.0)
        a_to_l_by_age = np.minimum(a_after_v, a_to_l_hazard * a_age)
        a_survivors = np.maximum(a_after_v - a_to_l_by_age, 0.0)
        a_to_v = float(np.sum(a_to_v_by_age))
        a_to_l = float(np.sum(a_to_l_by_age))

        l_to_a_hazard = (
            np.asarray(_cloglog_inverse(eta_map["L_to_A"][idx] + age_slope_map.get("L_to_A", 0.0) * _age_basis_log(age_count)), dtype=np.float64)
            if candidate.care_family == "semi_markov"
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
        "diagnosis_age_profile": diagnosis_age_profile,
    }


def _bic_from_residuals(residuals: np.ndarray, parameter_count: int) -> float:
    residual_array = np.asarray(residuals, dtype=np.float64)
    if residual_array.size <= 0:
        return float("inf")
    sse = float(np.sum(np.square(residual_array)))
    sse = max(sse, float(np.finfo(np.float64).eps))
    n_obs = float(residual_array.size)
    return float(n_obs * np.log(sse / n_obs) + float(parameter_count) * np.log(n_obs))


def _fit_candidate(
    snapshot: IntegratedSnapshot,
    candidate: DiagnosisKernelCandidate,
    reference_config: CandidateConfig,
    contract: BlockedTimeContract,
    *,
    budget_multiplier: float = 1.0,
    x0_override: np.ndarray | None = None,
) -> DiagnosisKernelFit:
    matrices = _matrices_for_reference(snapshot, reference_config)
    return _fit_candidate_with_matrices(
        snapshot,
        candidate,
        matrices,
        contract,
        budget_multiplier=budget_multiplier,
        x0_override=x0_override,
    )


def _fit_candidate_with_matrices(
    snapshot: IntegratedSnapshot,
    candidate: DiagnosisKernelCandidate,
    matrices: dict[str, Any],
    contract: BlockedTimeContract,
    *,
    budget_multiplier: float = 1.0,
    x0_override: np.ndarray | None = None,
) -> DiagnosisKernelFit:
    param_slices = _build_param_slices(candidate, matrices)
    x0 = (
        np.asarray(x0_override, dtype=np.float64)
        if x0_override is not None
        else _initial_param_vector(snapshot, candidate, matrices, param_slices, contract.train_mask)
    )
    residual_budget = sum(
        int(np.sum(np.logical_and(snapshot.metric_masks[metric_name], contract.train_mask)))
        for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,)
    )
    base_nfev = int(param_slices["full"].stop + residual_budget + int(np.sum(contract.train_mask)))
    max_nfev = max(1, int(np.ceil(float(base_nfev) * max(float(budget_multiplier), 1.0))))

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

    started_at = time.perf_counter()
    result = optimize.least_squares(residual_fn, x0=x0, method="trf", max_nfev=max_nfev)
    wall_seconds = float(time.perf_counter() - started_at)
    fitted_vector = np.asarray(result.x, dtype=np.float64)
    simulation = _simulate_candidate(snapshot, candidate, fitted_vector, param_slices, matrices)
    train_residuals = _fit_residuals(snapshot, simulation, contract.train_mask)
    split_metrics = {
        "train": _compute_window_metrics(snapshot, simulation, contract.train_mask),
        "validation": _compute_window_metrics(snapshot, simulation, contract.validation_mask),
        "holdout": _compute_window_metrics(snapshot, simulation, contract.holdout_mask),
    }
    return DiagnosisKernelFit(
        candidate=candidate,
        budget_multiplier=float(max(float(budget_multiplier), 1.0)),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        train_cost=float(result.cost),
        param_vector=fitted_vector,
        param_slices=param_slices,
        matrices=matrices,
        simulation=simulation,
        split_metrics=split_metrics,
        age_profile=np.asarray(simulation["diagnosis_age_profile"], dtype=np.float64),
        train_bic=_bic_from_residuals(train_residuals, int(param_slices["full"].stop)),
        nfev=int(getattr(result, "nfev", 0) or 0),
        max_nfev=int(max_nfev),
        wall_seconds=wall_seconds,
    )


def _carry_forward_blocked_time(snapshot: IntegratedSnapshot, contract: BlockedTimeContract) -> dict[str, Any]:
    predictions: dict[str, np.ndarray] = {}
    for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,):
        metric_mask = np.asarray(snapshot.metric_masks[metric_name], dtype=bool)
        values = np.asarray(snapshot.metric_values[metric_name], dtype=np.float64)
        prediction = np.full((len(snapshot.model_quarters),), np.nan, dtype=np.float64)
        train_observed_mask = np.logical_and(contract.train_mask, metric_mask)
        if np.any(train_observed_mask):
            last_index = int(np.where(train_observed_mask)[0][-1])
            last_value = float(values[last_index])
            prediction[contract.train_mask] = values[contract.train_mask]
            prediction[np.logical_or(contract.validation_mask, contract.holdout_mask)] = last_value
        predictions[metric_name] = prediction
    return {"predictions": predictions, "plausibility": {"population_violation_count": 0, "non_finite_count": 0}}


def _score_tuple(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("primary_loss", float("inf"))),
        float(metrics.get("diag_flow_loss", float("inf"))),
        float(metrics.get("total_loss", float("inf"))),
        float(metrics.get("secondary_loss", float("inf"))),
    )


def _candidate_valid(fit: DiagnosisKernelFit) -> bool:
    validation = fit.split_metrics["validation"]
    holdout = fit.split_metrics["holdout"]
    return bool(
        np.isfinite(float(fit.train_cost))
        and float(validation.get("non_finite_count", 0.0)) <= 0.0
        and float(validation.get("population_violation_count", 0.0)) <= 0.0
        and float(holdout.get("non_finite_count", 0.0)) <= 0.0
        and float(holdout.get("population_violation_count", 0.0)) <= 0.0
        and np.isfinite(float(validation.get("primary_loss", float("inf"))))
        and np.isfinite(float(validation.get("diag_flow_loss", float("inf"))))
        and np.isfinite(float(holdout.get("primary_loss", float("inf"))))
        and np.isfinite(float(holdout.get("diag_flow_loss", float("inf"))))
    )


def _frontier_rows(fits: list[DiagnosisKernelFit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_validation_primary = float("inf")
    for fit in fits:
        validation_primary = float(fit.split_metrics["validation"]["primary_loss"])
        best_validation_primary = min(best_validation_primary, validation_primary)
        rows.append(
            {
                "candidate_id": fit.candidate.candidate_id,
                "diagnosis_kind": fit.candidate.diagnosis_kind,
                "kernel_width": int(fit.candidate.kernel_width),
                "success": bool(fit.success),
                "validation_primary_loss": validation_primary,
                "validation_diag_flow_loss": float(fit.split_metrics["validation"]["diag_flow_loss"]),
                "validation_secondary_loss": float(fit.split_metrics["validation"]["secondary_loss"]),
                "validation_total_loss": float(fit.split_metrics["validation"]["total_loss"]),
                "holdout_primary_loss": float(fit.split_metrics["holdout"]["primary_loss"]),
                "holdout_diag_flow_loss": float(fit.split_metrics["holdout"]["diag_flow_loss"]),
                "holdout_secondary_loss": float(fit.split_metrics["holdout"]["secondary_loss"]),
                "holdout_total_loss": float(fit.split_metrics["holdout"]["total_loss"]),
                "train_primary_loss": float(fit.split_metrics["train"]["primary_loss"]),
                "train_diag_flow_loss": float(fit.split_metrics["train"]["diag_flow_loss"]),
                "train_bic": float(fit.train_bic),
                "best_validation_primary_loss": float(best_validation_primary),
                "valid": _candidate_valid(fit),
            }
        )
    return rows


def _best_fit(fits: list[DiagnosisKernelFit], diagnosis_kind: str | None = None) -> DiagnosisKernelFit:
    filtered = [fit for fit in fits if diagnosis_kind is None or fit.candidate.diagnosis_kind == diagnosis_kind]
    valid = [fit for fit in filtered if _candidate_valid(fit)]
    if not valid:
        raise RuntimeError(f"No valid fits found for diagnosis kind {diagnosis_kind or 'any'}")
    return min(valid, key=lambda fit: (_score_tuple(fit.split_metrics["validation"]), int(fit.candidate.kernel_width)))


def _fit_by_candidate_id(fits: list[DiagnosisKernelFit], candidate_id: str) -> DiagnosisKernelFit:
    for fit in fits:
        if str(fit.candidate.candidate_id) == str(candidate_id):
            return fit
    raise KeyError(f"Candidate not found in fit list: {candidate_id}")


def _fit_payload(fit: DiagnosisKernelFit) -> dict[str, Any]:
    return {
        "candidate_id": fit.candidate.candidate_id,
        "diagnosis_kind": fit.candidate.diagnosis_kind,
        "care_family": fit.candidate.care_family,
        "kernel_width": int(fit.candidate.kernel_width),
        "budget_multiplier": float(fit.budget_multiplier),
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
    }


def _baseline_payload(snapshot: IntegratedSnapshot, contract: BlockedTimeContract) -> dict[str, Any]:
    carry_forward = _carry_forward_blocked_time(snapshot, contract)
    zero_matrices = {
        "incidence_direct": np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64),
        "transition_direct": {transition: np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64) for transition in TRANSITION_NAMES},
        "observation_direct": {
            "tested_for_viral_load": np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64),
            "virally_suppressed": np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64),
        },
        "hidden": np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64),
        "feature_ids": {
            "incidence": [],
            "transition": {transition: [] for transition in TRANSITION_NAMES},
            "observation": {"tested_for_viral_load": [], "virally_suppressed": []},
        },
    }
    simple_candidate = DiagnosisKernelCandidate(
        candidate_id="simple-compartmental-constant-hazard",
        diagnosis_kind="hazard",
        care_family="markov",
        kernel_width=0,
        reference_hidden_rank=0,
        use_observation_covariates=False,
    )
    simple_param_slices = _build_param_slices(simple_candidate, zero_matrices)
    x0 = _initial_param_vector(snapshot, simple_candidate, zero_matrices, simple_param_slices, contract.train_mask)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        simulation = _simulate_candidate(snapshot, simple_candidate, params, simple_param_slices, zero_matrices)
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

    result = optimize.least_squares(residual_fn, x0=x0, method="trf", max_nfev=int(simple_param_slices["full"].stop + 100))
    simple_simulation = _simulate_candidate(snapshot, simple_candidate, np.asarray(result.x, dtype=np.float64), simple_param_slices, zero_matrices)
    return {
        "carry_forward": {
            "validation": _compute_window_metrics(snapshot, carry_forward, contract.validation_mask),
            "holdout": _compute_window_metrics(snapshot, carry_forward, contract.holdout_mask),
        },
        "simple_compartmental": {
            "validation": _compute_window_metrics(snapshot, simple_simulation, contract.validation_mask),
            "holdout": _compute_window_metrics(snapshot, simple_simulation, contract.holdout_mask),
        },
    }


def _write_frontier_chart(path: Path, frontier_rows: list[dict[str, Any]]) -> None:
    x_labels = [str(row["candidate_id"]) for row in frontier_rows]
    x_values = np.arange(len(frontier_rows))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    axes[0].plot(x_values, [float(row["validation_primary_loss"]) for row in frontier_rows], marker="o", linewidth=1.5, label="validation primary")
    axes[0].plot(x_values, [float(row["holdout_primary_loss"]) for row in frontier_rows], marker="s", linewidth=1.2, label="holdout primary")
    axes[0].set_title("DIAG-01A Candidate Frontier")
    axes[0].set_ylabel("primary loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(x_values, [float(row["validation_diag_flow_loss"]) for row in frontier_rows], marker="o", linewidth=1.5, label="validation diagnosis flow")
    axes[1].plot(x_values, [float(row["holdout_diag_flow_loss"]) for row in frontier_rows], marker="s", linewidth=1.2, label="holdout diagnosis flow")
    axes[1].set_ylabel("diagnosis-flow loss")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(x_labels, rotation=30, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_model_comparison_chart(
    path: Path,
    current_hazard: DiagnosisKernelFit,
    current_delay: DiagnosisKernelFit,
    strict_winner: DiagnosisKernelFit,
    baselines: dict[str, Any],
) -> None:
    labels = ["carry_forward", "simple_compartmental", "hazard", "delay", "strict_kernel"]
    validation_primary = [
        float(baselines["carry_forward"]["validation"]["primary_loss"]),
        float(baselines["simple_compartmental"]["validation"]["primary_loss"]),
        float(current_hazard.split_metrics["validation"]["primary_loss"]),
        float(current_delay.split_metrics["validation"]["primary_loss"]),
        float(strict_winner.split_metrics["validation"]["primary_loss"]),
    ]
    holdout_primary = [
        float(baselines["carry_forward"]["holdout"]["primary_loss"]),
        float(baselines["simple_compartmental"]["holdout"]["primary_loss"]),
        float(current_hazard.split_metrics["holdout"]["primary_loss"]),
        float(current_delay.split_metrics["holdout"]["primary_loss"]),
        float(strict_winner.split_metrics["holdout"]["primary_loss"]),
    ]
    validation_diag = [
        float(baselines["carry_forward"]["validation"]["diag_flow_loss"]),
        float(baselines["simple_compartmental"]["validation"]["diag_flow_loss"]),
        float(current_hazard.split_metrics["validation"]["diag_flow_loss"]),
        float(current_delay.split_metrics["validation"]["diag_flow_loss"]),
        float(strict_winner.split_metrics["validation"]["diag_flow_loss"]),
    ]
    holdout_diag = [
        float(baselines["carry_forward"]["holdout"]["diag_flow_loss"]),
        float(baselines["simple_compartmental"]["holdout"]["diag_flow_loss"]),
        float(current_hazard.split_metrics["holdout"]["diag_flow_loss"]),
        float(current_delay.split_metrics["holdout"]["diag_flow_loss"]),
        float(strict_winner.split_metrics["holdout"]["diag_flow_loss"]),
    ]
    x_values = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True)
    axes[0].bar(x_values - width / 2, validation_primary, width=width, label="validation")
    axes[0].bar(x_values + width / 2, holdout_primary, width=width, label="holdout")
    axes[0].set_title("DIAG-01A Blocked-Time Primary Loss Comparison")
    axes[0].set_ylabel("primary loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x_values - width / 2, validation_diag, width=width, label="validation")
    axes[1].bar(x_values + width / 2, holdout_diag, width=width, label="holdout")
    axes[1].set_title("DIAG-01A Blocked-Time Diagnosis-Flow Loss Comparison")
    axes[1].set_ylabel("diagnosis-flow loss")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_prediction_chart(
    path: Path,
    snapshot: IntegratedSnapshot,
    contract: BlockedTimeContract,
    current_hazard: DiagnosisKernelFit,
    current_delay: DiagnosisKernelFit,
    strict_winner: DiagnosisKernelFit,
) -> None:
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
        for label, fit in (("hazard", current_hazard), ("delay", current_delay), ("strict", strict_winner)):
            axis.plot(
                quarter_labels,
                [float(fit.simulation["predictions"][metric_name][idx]) for idx in indices],
                linewidth=1.4,
                label=label,
            )
        axis.set_title(f"DIAG-01A {title} on Validation + Holdout")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xticks(range(len(quarter_labels)))
    axes[-1].set_xticklabels(quarter_labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_kernel_profile_chart(path: Path, current_delay: DiagnosisKernelFit, strict_winner: DiagnosisKernelFit) -> None:
    age_axis = np.arange(len(current_delay.age_profile), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.plot(age_axis, current_delay.age_profile, linewidth=1.7, label="current delay slope")
    ax.plot(age_axis, strict_winner.age_profile, linewidth=1.7, label=f"strict kernel (width={strict_winner.candidate.kernel_width})")
    ax.set_title("DIAG-01A Diagnosis Age-Kernel Profile")
    ax.set_xlabel("quarters since entry into U")
    ax.set_ylabel("diagnosis logit offset by age")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_dashboard_markdown(
    path: Path,
    *,
    contract: BlockedTimeContract,
    current_hazard: DiagnosisKernelFit,
    current_delay: DiagnosisKernelFit,
    strict_winner: DiagnosisKernelFit,
    baselines: dict[str, Any],
) -> None:
    lines = [
        "# DIAG-01A Strict Diagnosis Kernel Blocked-Time Dashboard",
        "",
        "This experiment keeps incidence, downstream care, hidden-rank capacity, and observation settings fixed to the current diagnosis branch configuration and changes only the diagnosis-delay kernel.",
        "",
        "## Blocked-Time Contract",
        "",
        f"- Train diagnosis-flow quarters: `{', '.join(contract.train_diagnosis_quarters)}`",
        f"- Validation quarters: `{', '.join(contract.validation_quarters)}`",
        f"- Holdout quarters: `{', '.join(contract.holdout_quarters)}`",
        "",
        "## Model Comparison",
        "",
        "| Model | Validation Primary | Holdout Primary | Validation Diagnosis Flow | Holdout Diagnosis Flow |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    rows = [
        ("carry_forward", baselines["carry_forward"]),
        ("simple_compartmental", baselines["simple_compartmental"]),
        ("hazard", {"validation": current_hazard.split_metrics["validation"], "holdout": current_hazard.split_metrics["holdout"]}),
        ("delay", {"validation": current_delay.split_metrics["validation"], "holdout": current_delay.split_metrics["holdout"]}),
        ("strict_kernel", {"validation": strict_winner.split_metrics["validation"], "holdout": strict_winner.split_metrics["holdout"]}),
    ]
    for label, metrics in rows:
        lines.append(
            f"| `{label}` | {float(metrics['validation']['primary_loss']):.6f} | {float(metrics['holdout']['primary_loss']):.6f} | "
            f"{float(metrics['validation']['diag_flow_loss']):.6f} | {float(metrics['holdout']['diag_flow_loss']):.6f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _export_paper_figures(ctx: TransitionResearchContext, chart_paths: dict[str, Path]) -> dict[str, Any]:
    archive_dir = ROOT_DIR / "artifacts" / "paper_figures" / "phase3_diag_kernel_20260409"
    archive_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = list(read_json(archive_dir / "figure_manifest.json", default=[]))
    retained = [
        row
        for row in existing_manifest
        if isinstance(row, dict) and not str(row.get("figure_id") or "").startswith("fig_diag01a_")
    ]
    figure_specs = [
        (
            "fig_diag01a_01",
            "Strict Diagnosis Blocked-Time Frontier",
            "Validation and holdout losses across hazard, current delay, and strict diagnosis-kernel candidates under the frozen blocked-time contract.",
            "diag01a_candidate_frontier.png",
            chart_paths.get("frontier"),
        ),
        (
            "fig_diag01a_02",
            "Strict Diagnosis Blocked-Time Model Comparison",
            "Validation and holdout primary-loss and diagnosis-flow-loss comparison against carry-forward, simple compartmental, current hazard, current delay, and selected strict diagnosis kernel.",
            "diag01a_model_comparison.png",
            chart_paths.get("comparison"),
        ),
        (
            "fig_diag01a_03",
            "Strict Diagnosis Blocked-Time Predictions",
            "Observed and predicted new diagnoses and diagnosed stock over the blocked validation and holdout periods for hazard, current delay, and strict diagnosis kernel branches.",
            "diag01a_predictions.png",
            chart_paths.get("predictions"),
        ),
        (
            "fig_diag01a_04",
            "Strict Diagnosis Kernel Profile",
            "Age-kernel comparison between the current one-slope diagnosis-delay branch and the selected strict diagnosis kernel.",
            "diag01a_kernel_profile.png",
            chart_paths.get("kernel"),
        ),
    ]
    new_rows: list[dict[str, Any]] = []
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


def _budget_multipliers(contract: BlockedTimeContract) -> list[float]:
    ceiling = max(2, int(len(contract.train_diagnosis_quarters)))
    multipliers: list[float] = [1.0]
    current = 1
    while current < ceiling:
        current = min(current * 2, ceiling)
        multipliers.append(float(current))
        if current >= ceiling:
            break
    return multipliers


def _fit_budget_ladder(
    snapshot: IntegratedSnapshot,
    candidate: DiagnosisKernelCandidate,
    reference_config: CandidateConfig,
    contract: BlockedTimeContract,
    multipliers: list[float],
) -> list[DiagnosisKernelFit]:
    fits: list[DiagnosisKernelFit] = []
    warm_start: np.ndarray | None = None
    for multiplier in multipliers:
        fit = _fit_candidate(
            snapshot,
            candidate,
            reference_config,
            contract,
            budget_multiplier=float(multiplier),
            x0_override=warm_start,
        )
        fits.append(fit)
        warm_start = np.asarray(fit.param_vector, dtype=np.float64)
        if fit.success and fit.nfev < fit.max_nfev:
            break
    return fits


def _fit_at_or_before(fits: list[DiagnosisKernelFit], budget_multiplier: float) -> DiagnosisKernelFit:
    eligible = [fit for fit in fits if float(fit.max_nfev) <= float(fits[0].max_nfev) * float(budget_multiplier) + 1e-9]
    return eligible[-1] if eligible else fits[0]


def _budget_row(fit: DiagnosisKernelFit) -> dict[str, Any]:
    return {
        "candidate_id": str(fit.candidate.candidate_id),
        "diagnosis_kind": str(fit.candidate.diagnosis_kind),
        "kernel_width": int(fit.candidate.kernel_width),
        "budget_multiplier": float(fit.budget_multiplier),
        "nfev": int(fit.nfev),
        "max_nfev": int(fit.max_nfev),
        "wall_seconds": float(fit.wall_seconds),
        "success": bool(fit.success),
        "status": int(fit.status),
        "message": str(fit.message),
        "validation_primary_loss": float(fit.split_metrics["validation"]["primary_loss"]),
        "validation_diag_flow_loss": float(fit.split_metrics["validation"]["diag_flow_loss"]),
        "holdout_primary_loss": float(fit.split_metrics["holdout"]["primary_loss"]),
        "holdout_diag_flow_loss": float(fit.split_metrics["holdout"]["diag_flow_loss"]),
    }


def _budget_frontier_payload(
    ladder_fits: dict[str, list[DiagnosisKernelFit]],
    multipliers: list[float],
) -> dict[str, Any]:
    first_fit = next(iter(ladder_fits.values()))[0]
    base_budget = int(first_fit.max_nfev)
    rows = [_budget_row(fit) for fits in ladder_fits.values() for fit in fits]
    champion_rows: list[dict[str, Any]] = []
    for multiplier in multipliers:
        candidates = [_fit_at_or_before(fits, float(multiplier)) for fits in ladder_fits.values()]
        champion_validation = min(candidates, key=lambda fit: _score_tuple(fit.split_metrics["validation"]))
        champion_holdout = min(candidates, key=lambda fit: _score_tuple(fit.split_metrics["holdout"]))
        champion_rows.append(
            {
                "budget_multiplier": float(multiplier),
                "validation_champion_candidate_id": str(champion_validation.candidate.candidate_id),
                "validation_champion_primary_loss": float(champion_validation.split_metrics["validation"]["primary_loss"]),
                "validation_champion_diag_flow_loss": float(champion_validation.split_metrics["validation"]["diag_flow_loss"]),
                "holdout_champion_candidate_id": str(champion_holdout.candidate.candidate_id),
                "holdout_champion_primary_loss": float(champion_holdout.split_metrics["holdout"]["primary_loss"]),
                "holdout_champion_diag_flow_loss": float(champion_holdout.split_metrics["holdout"]["diag_flow_loss"]),
            }
        )
    return {"rows": rows, "champion_rows": champion_rows, "base_budget_nfev": int(base_budget)}


def _write_budget_loss_curves(path: Path, ladder_fits: dict[str, list[DiagnosisKernelFit]], multipliers: list[float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex=True)
    metric_specs = (
        ("validation", "primary_loss", axes[0, 0], "Validation Primary Loss"),
        ("holdout", "primary_loss", axes[0, 1], "Holdout Primary Loss"),
        ("validation", "diag_flow_loss", axes[1, 0], "Validation Diagnosis-Flow Loss"),
        ("holdout", "diag_flow_loss", axes[1, 1], "Holdout Diagnosis-Flow Loss"),
    )
    for candidate_id, fits in ladder_fits.items():
        label = str(candidate_id)
        for split_name, metric_name, axis, title in metric_specs:
            y_values = [float(_fit_at_or_before(fits, multiplier).split_metrics[split_name][metric_name]) for multiplier in multipliers]
            axis.plot(multipliers, y_values, marker="o", linewidth=1.5, label=label)
            axis.set_title(title)
            axis.set_xlabel("budget multiplier")
            axis.set_ylabel(metric_name.replace("_", " "))
            axis.grid(alpha=0.3)
    axes[0, 1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_budget_champion_trace(path: Path, champion_rows: list[dict[str, Any]]) -> None:
    multipliers = [float(row["budget_multiplier"]) for row in champion_rows]
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.6), sharex=True)
    axes[0].plot(multipliers, [float(row["validation_champion_primary_loss"]) for row in champion_rows], marker="o", linewidth=1.6, label="validation primary")
    axes[0].plot(multipliers, [float(row["holdout_champion_primary_loss"]) for row in champion_rows], marker="s", linewidth=1.6, label="holdout primary")
    axes[0].set_title("DIAG-01B Champion Primary Loss vs Compute Budget")
    axes[0].set_ylabel("primary loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(multipliers, [float(row["validation_champion_diag_flow_loss"]) for row in champion_rows], marker="o", linewidth=1.6, label="validation diagnosis flow")
    axes[1].plot(multipliers, [float(row["holdout_champion_diag_flow_loss"]) for row in champion_rows], marker="s", linewidth=1.6, label="holdout diagnosis flow")
    axes[1].set_title("DIAG-01B Champion Diagnosis-Flow Loss vs Compute Budget")
    axes[1].set_xlabel("budget multiplier")
    axes[1].set_ylabel("diagnosis-flow loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_budget_compute_scatter(path: Path, budget_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=False)
    diagnosis_kinds = sorted({str(row["diagnosis_kind"]) for row in budget_rows})
    color_map = {"hazard": "#4C78A8", "delay": "#F58518", "strict": "#54A24B"}
    for diagnosis_kind in diagnosis_kinds:
        kind_rows = [row for row in budget_rows if str(row["diagnosis_kind"]) == diagnosis_kind]
        axes[0].scatter(
            [float(row["nfev"]) for row in kind_rows],
            [float(row["validation_primary_loss"]) for row in kind_rows],
            label=diagnosis_kind,
            color=color_map.get(diagnosis_kind, None),
            alpha=0.8,
        )
        axes[1].scatter(
            [float(row["wall_seconds"]) for row in kind_rows],
            [float(row["holdout_diag_flow_loss"]) for row in kind_rows],
            label=diagnosis_kind,
            color=color_map.get(diagnosis_kind, None),
            alpha=0.8,
        )
    axes[0].set_title("DIAG-01B Validation Primary vs Function Evaluations")
    axes[0].set_xlabel("nfev")
    axes[0].set_ylabel("validation primary loss")
    axes[0].grid(alpha=0.3)
    axes[1].set_title("DIAG-01B Holdout Diagnosis Flow vs Wall Time")
    axes[1].set_xlabel("wall seconds")
    axes[1].set_ylabel("holdout diagnosis-flow loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_diag_01a(ctx: TransitionResearchContext) -> dict[str, Any]:
    snapshot = _build_snapshot(ctx)
    reference_config = _select_reference_configs()["diagnosis"]
    contract = _build_blocked_time_contract(snapshot)
    candidates = _build_candidates(reference_config, contract)
    fits = [_fit_candidate(snapshot, candidate, reference_config, contract) for candidate in candidates]
    frontier_rows = _frontier_rows(fits)
    current_hazard = _best_fit(fits, diagnosis_kind="hazard")
    current_delay = _best_fit(fits, diagnosis_kind="delay")
    strict_winner = _best_fit(fits, diagnosis_kind="strict")
    best_overall = _best_fit(fits, diagnosis_kind=None)
    baselines = _baseline_payload(snapshot, contract)

    split_contract_path = ctx.experiment_dir / "blocked_time_contract.json"
    frontier_path = ctx.experiment_dir / "candidate_frontier.json"
    evaluation_path = ctx.experiment_dir / "blocked_time_evaluation.json"
    dashboard_path = ctx.experiment_dir / "strict_diagnosis_dashboard.md"
    comparison_path = ctx.experiment_dir / "blocked_time_model_comparison.json"
    prediction_rows_path = ctx.experiment_dir / "blocked_time_prediction_rows.json"
    frontier_chart_path = ctx.experiment_dir / "blocked_time_candidate_frontier.png"
    model_comparison_chart_path = ctx.experiment_dir / "blocked_time_model_comparison.png"
    prediction_chart_path = ctx.experiment_dir / "blocked_time_predictions.png"
    kernel_chart_path = ctx.experiment_dir / "blocked_time_kernel_profile.png"

    split_contract_payload = {
        "generated_at": utc_now_iso(),
        "diagnosis_flow_observed_quarters": list(contract.diagnosis_flow_observed_quarters),
        "train_diagnosis_quarters": list(contract.train_diagnosis_quarters),
        "validation_quarters": list(contract.validation_quarters),
        "holdout_quarters": list(contract.holdout_quarters),
        "train_end_quarter": str(contract.train_end_quarter),
        "validation_start_quarter": str(contract.validation_start_quarter),
        "validation_end_quarter": str(contract.validation_end_quarter),
        "holdout_start_quarter": str(contract.holdout_start_quarter),
        "holdout_end_quarter": str(contract.holdout_end_quarter),
    }
    write_json(split_contract_path, split_contract_payload)
    write_json(frontier_path, frontier_rows)

    comparison_payload = {
        "carry_forward": baselines["carry_forward"],
        "simple_compartmental": baselines["simple_compartmental"],
        "hazard": _fit_payload(current_hazard),
        "delay": _fit_payload(current_delay),
        "strict_kernel": _fit_payload(strict_winner),
        "best_overall_candidate_id": str(best_overall.candidate.candidate_id),
    }
    write_json(comparison_path, comparison_payload)

    prediction_rows: list[dict[str, Any]] = []
    relevant_mask = np.logical_or(contract.validation_mask, contract.holdout_mask)
    for idx in np.where(relevant_mask)[0]:
        prediction_rows.append(
            {
                "quarter": str(snapshot.model_quarters[idx]),
                "window": "validation" if bool(contract.validation_mask[idx]) else "holdout",
                "observed_new_diagnosed_cases_period": float(snapshot.metric_values["new_diagnosed_cases_period"][idx])
                if bool(snapshot.metric_masks["new_diagnosed_cases_period"][idx])
                else None,
                "hazard_new_diagnosed_cases_period": float(current_hazard.simulation["predictions"]["new_diagnosed_cases_period"][idx]),
                "delay_new_diagnosed_cases_period": float(current_delay.simulation["predictions"]["new_diagnosed_cases_period"][idx]),
                "strict_new_diagnosed_cases_period": float(strict_winner.simulation["predictions"]["new_diagnosed_cases_period"][idx]),
                "observed_diagnosed_plhiv": float(snapshot.metric_values["diagnosed_plhiv"][idx]) if bool(snapshot.metric_masks["diagnosed_plhiv"][idx]) else None,
                "hazard_diagnosed_plhiv": float(current_hazard.simulation["predictions"]["diagnosed_plhiv"][idx]),
                "delay_diagnosed_plhiv": float(current_delay.simulation["predictions"]["diagnosed_plhiv"][idx]),
                "strict_diagnosed_plhiv": float(strict_winner.simulation["predictions"]["diagnosed_plhiv"][idx]),
            }
        )
    write_json(prediction_rows_path, prediction_rows)

    _write_frontier_chart(frontier_chart_path, frontier_rows)
    _write_model_comparison_chart(model_comparison_chart_path, current_hazard, current_delay, strict_winner, baselines)
    _write_prediction_chart(prediction_chart_path, snapshot, contract, current_hazard, current_delay, strict_winner)
    _write_kernel_profile_chart(kernel_chart_path, current_delay, strict_winner)
    _write_dashboard_markdown(
        dashboard_path,
        contract=contract,
        current_hazard=current_hazard,
        current_delay=current_delay,
        strict_winner=strict_winner,
        baselines=baselines,
    )
    paper_archive = _export_paper_figures(
        ctx,
        {
            "frontier": frontier_chart_path,
            "comparison": model_comparison_chart_path,
            "predictions": prediction_chart_path,
            "kernel": kernel_chart_path,
        },
    )

    evaluation_payload = {
        "reference_candidate_id": str(reference_config.candidate_id),
        "reference_hidden_rank": int(reference_config.hidden_rank),
        "reference_use_observation_covariates": bool(reference_config.use_observation_covariates),
        "current_hazard": _fit_payload(current_hazard),
        "current_delay": _fit_payload(current_delay),
        "strict_kernel": _fit_payload(strict_winner),
        "best_overall": _fit_payload(best_overall),
        "strict_vs_delay_validation_delta": {
            "primary_loss": float(strict_winner.split_metrics["validation"]["primary_loss"] - current_delay.split_metrics["validation"]["primary_loss"]),
            "diag_flow_loss": float(strict_winner.split_metrics["validation"]["diag_flow_loss"] - current_delay.split_metrics["validation"]["diag_flow_loss"]),
        },
        "strict_vs_delay_holdout_delta": {
            "primary_loss": float(strict_winner.split_metrics["holdout"]["primary_loss"] - current_delay.split_metrics["holdout"]["primary_loss"]),
            "diag_flow_loss": float(strict_winner.split_metrics["holdout"]["diag_flow_loss"] - current_delay.split_metrics["holdout"]["diag_flow_loss"]),
        },
        "strict_vs_hazard_validation_delta": {
            "primary_loss": float(strict_winner.split_metrics["validation"]["primary_loss"] - current_hazard.split_metrics["validation"]["primary_loss"]),
            "diag_flow_loss": float(strict_winner.split_metrics["validation"]["diag_flow_loss"] - current_hazard.split_metrics["validation"]["diag_flow_loss"]),
        },
        "strict_vs_hazard_holdout_delta": {
            "primary_loss": float(strict_winner.split_metrics["holdout"]["primary_loss"] - current_hazard.split_metrics["holdout"]["primary_loss"]),
            "diag_flow_loss": float(strict_winner.split_metrics["holdout"]["diag_flow_loss"] - current_hazard.split_metrics["holdout"]["diag_flow_loss"]),
        },
        "paper_archive": paper_archive,
    }
    write_json(evaluation_path, evaluation_payload)

    decision = {
        "reference_candidate_id": str(reference_config.candidate_id),
        "best_overall_candidate_id": str(best_overall.candidate.candidate_id),
        "strict_candidate_id": str(strict_winner.candidate.candidate_id),
        "strict_validation_beats_delay": bool(_score_tuple(strict_winner.split_metrics["validation"]) < _score_tuple(current_delay.split_metrics["validation"])),
        "strict_holdout_beats_delay": bool(_score_tuple(strict_winner.split_metrics["holdout"]) < _score_tuple(current_delay.split_metrics["holdout"])),
        "strict_validation_beats_hazard": bool(_score_tuple(strict_winner.split_metrics["validation"]) < _score_tuple(current_hazard.split_metrics["validation"])),
        "strict_holdout_beats_hazard": bool(_score_tuple(strict_winner.split_metrics["holdout"]) < _score_tuple(current_hazard.split_metrics["holdout"])),
        "keep_on_validation_gate": bool(_score_tuple(strict_winner.split_metrics["validation"]) < _score_tuple(current_delay.split_metrics["validation"])),
        "paper_figure_archive": paper_archive,
    }

    experiment_spec = {
        "variant": "evidence-to-model-loop",
        "goal": "Implement a stricter diagnosis-delay kernel while holding incidence, downstream care, hidden rank, and observation settings fixed to the current diagnosis branch.",
        "source_run_id": ctx.source_run_id,
        "reference_integrated_run": str(_latest_integrated_experiment_dir()) if _latest_integrated_experiment_dir() is not None else None,
        "reference_candidate_id": str(reference_config.candidate_id),
        "mutation_unit": "diagnosis kernel shape only",
        "blocked_time_contract": split_contract_payload,
        "candidate_ids": [str(candidate.candidate_id) for candidate in candidates],
        "artifacts": {
            "blocked_time_contract": str(split_contract_path),
            "candidate_frontier": str(frontier_path),
            "blocked_time_evaluation": str(evaluation_path),
            "dashboard_markdown": str(dashboard_path),
            "blocked_time_model_comparison": str(comparison_path),
            "blocked_time_prediction_rows": str(prediction_rows_path),
        },
    }
    coverage_summary = {
        "historical_quarter_count": int(len(snapshot.historical_quarters)),
        "train_quarter_count": int(np.sum(contract.train_mask)),
        "validation_quarter_count": int(np.sum(contract.validation_mask)),
        "holdout_quarter_count": int(np.sum(contract.holdout_mask)),
        "diagnosis_flow_observed_quarter_count": int(len(contract.diagnosis_flow_observed_quarters)),
        "train_diagnosis_flow_observed_quarter_count": int(len(contract.train_diagnosis_quarters)),
        "validation_diagnosis_flow_observed_quarter_count": int(len(contract.validation_quarters)),
        "holdout_diagnosis_flow_observed_quarter_count": int(len(contract.holdout_quarters)),
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            numerical_guard_entry(
                name="strict_diagnosis_kernel_eps",
                role="strict_diagnosis_kernel_numerical_guard",
                why_needed="Prevents divide-by-zero and log singularities while fitting blocked-time diagnosis hazards and observation shares.",
            ),
            {
                "name": "diagnosis_flow_observed_quarter_count",
                "value": int(len(contract.diagnosis_flow_observed_quarters)),
                "role": "blocked_time_split_total_support",
                "source_type": "estimated",
                "estimation_data": "Historical quarters with observed new_diagnosed_cases_period in the active national snapshot",
                "estimation_method": "count(snapshot.historical_quarters intersect diagnosis-flow observation mask)",
                "uncertainty": "deterministic given the frozen snapshot",
                "why_needed": "Defines the admissible blocked-time partition without introducing manual year cutoffs.",
            },
            {
                "name": "strict_kernel_width_limit",
                "value": int(len(contract.train_diagnosis_quarters)),
                "role": "strict_diagnosis_kernel_search_ceiling",
                "source_type": "estimated",
                "estimation_data": "Training diagnosis-flow support after contiguous blocked-time partition",
                "estimation_method": "count(train diagnosis-flow quarters)",
                "uncertainty": "deterministic given the frozen blocked-time split",
                "why_needed": "Caps diagnosis-kernel flexibility by the amount of direct diagnosis-flow support available in training.",
            },
        ],
    )
    return {
        "artifacts": artifacts,
        "blocked_time_contract": split_contract_payload,
        "candidate_frontier": frontier_rows,
        "evaluation": evaluation_payload,
        "decision": decision,
        "paper_archive": paper_archive,
    }


def _write_promotion_frontier_chart(path: Path, frontier_rows: list[dict[str, Any]], reference_candidate_id: str) -> None:
    x_labels = [str(row["candidate_id"]) for row in frontier_rows]
    x_values = np.arange(len(frontier_rows))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    axes[0].plot(x_values, [float(row["validation_primary_loss"]) for row in frontier_rows], marker="o", linewidth=1.5, label="validation primary")
    axes[0].plot(x_values, [float(row["holdout_primary_loss"]) for row in frontier_rows], marker="s", linewidth=1.2, label="holdout primary")
    ref_index = next((idx for idx, row in enumerate(frontier_rows) if str(row["candidate_id"]) == str(reference_candidate_id)), None)
    if ref_index is not None:
        axes[0].axvline(float(ref_index), color="#666666", linestyle="--", linewidth=1.0, alpha=0.7, label="integrated champion")
    axes[0].set_title("DIAG-02A Integrated-Champion Promotion Frontier")
    axes[0].set_ylabel("primary loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(x_values, [float(row["validation_diag_flow_loss"]) for row in frontier_rows], marker="o", linewidth=1.5, label="validation diagnosis flow")
    axes[1].plot(x_values, [float(row["holdout_diag_flow_loss"]) for row in frontier_rows], marker="s", linewidth=1.2, label="holdout diagnosis flow")
    if ref_index is not None:
        axes[1].axvline(float(ref_index), color="#666666", linestyle="--", linewidth=1.0, alpha=0.7, label="integrated champion")
    axes[1].set_ylabel("diagnosis-flow loss")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(x_labels, rotation=30, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_promotion_model_comparison_chart(
    path: Path,
    reference_fit: DiagnosisKernelFit,
    strict_winner: DiagnosisKernelFit,
    baselines: dict[str, Any],
) -> None:
    labels = ["carry_forward", "simple_compartmental", "integrated_champion", "strict_promoted"]
    validation_primary = [
        float(baselines["carry_forward"]["validation"]["primary_loss"]),
        float(baselines["simple_compartmental"]["validation"]["primary_loss"]),
        float(reference_fit.split_metrics["validation"]["primary_loss"]),
        float(strict_winner.split_metrics["validation"]["primary_loss"]),
    ]
    holdout_primary = [
        float(baselines["carry_forward"]["holdout"]["primary_loss"]),
        float(baselines["simple_compartmental"]["holdout"]["primary_loss"]),
        float(reference_fit.split_metrics["holdout"]["primary_loss"]),
        float(strict_winner.split_metrics["holdout"]["primary_loss"]),
    ]
    validation_diag = [
        float(baselines["carry_forward"]["validation"]["diag_flow_loss"]),
        float(baselines["simple_compartmental"]["validation"]["diag_flow_loss"]),
        float(reference_fit.split_metrics["validation"]["diag_flow_loss"]),
        float(strict_winner.split_metrics["validation"]["diag_flow_loss"]),
    ]
    holdout_diag = [
        float(baselines["carry_forward"]["holdout"]["diag_flow_loss"]),
        float(baselines["simple_compartmental"]["holdout"]["diag_flow_loss"]),
        float(reference_fit.split_metrics["holdout"]["diag_flow_loss"]),
        float(strict_winner.split_metrics["holdout"]["diag_flow_loss"]),
    ]
    x_values = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True)
    axes[0].bar(x_values - width / 2, validation_primary, width=width, label="validation")
    axes[0].bar(x_values + width / 2, holdout_primary, width=width, label="holdout")
    axes[0].set_title("DIAG-02A Blocked-Time Primary Loss Comparison")
    axes[0].set_ylabel("primary loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x_values - width / 2, validation_diag, width=width, label="validation")
    axes[1].bar(x_values + width / 2, holdout_diag, width=width, label="holdout")
    axes[1].set_title("DIAG-02A Blocked-Time Diagnosis-Flow Loss Comparison")
    axes[1].set_ylabel("diagnosis-flow loss")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_promotion_prediction_chart(
    path: Path,
    snapshot: IntegratedSnapshot,
    contract: BlockedTimeContract,
    reference_fit: DiagnosisKernelFit,
    strict_winner: DiagnosisKernelFit,
) -> None:
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
        for label, fit in (("integrated champion", reference_fit), ("strict promoted", strict_winner)):
            axis.plot(
                quarter_labels,
                [float(fit.simulation["predictions"][metric_name][idx]) for idx in indices],
                linewidth=1.4,
                label=label,
            )
        axis.set_title(f"DIAG-02A {title} on Validation + Holdout")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xticks(range(len(quarter_labels)))
    axes[-1].set_xticklabels(quarter_labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_promotion_kernel_profile_chart(path: Path, reference_fit: DiagnosisKernelFit, strict_winner: DiagnosisKernelFit) -> None:
    age_axis = np.arange(len(strict_winner.age_profile), dtype=np.float64)
    reference_profile = (
        np.asarray(reference_fit.age_profile, dtype=np.float64)
        if reference_fit.candidate.diagnosis_kind != "hazard"
        else np.zeros_like(strict_winner.age_profile, dtype=np.float64)
    )
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.plot(age_axis, reference_profile, linewidth=1.7, label=f"integrated champion ({reference_fit.candidate.diagnosis_kind})")
    ax.plot(age_axis, strict_winner.age_profile, linewidth=1.7, label=f"strict promoted (width={strict_winner.candidate.kernel_width})")
    ax.set_title("DIAG-02A Diagnosis Age-Kernel Profile")
    ax.set_xlabel("quarters since entry into U")
    ax.set_ylabel("diagnosis offset by age")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_promotion_dashboard_markdown(
    path: Path,
    *,
    contract: BlockedTimeContract,
    reference_fit: DiagnosisKernelFit,
    strict_winner: DiagnosisKernelFit,
    baselines: dict[str, Any],
) -> None:
    lines = [
        "# DIAG-02A Integrated Champion Strict-Diagnosis Promotion Dashboard",
        "",
        "This experiment freezes the current PHASE3-V2 integrated stage-1 champion outside the diagnosis branch, keeps downstream care fixed, and mutates only the diagnosis kernel under the same blocked-time contract.",
        "",
        "## Blocked-Time Contract",
        "",
        f"- Train diagnosis-flow quarters: `{', '.join(contract.train_diagnosis_quarters)}`",
        f"- Validation quarters: `{', '.join(contract.validation_quarters)}`",
        f"- Holdout quarters: `{', '.join(contract.holdout_quarters)}`",
        "",
        "## Model Comparison",
        "",
        "| Model | Validation Primary | Holdout Primary | Validation Diagnosis Flow | Holdout Diagnosis Flow |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    rows = [
        ("carry_forward", baselines["carry_forward"]),
        ("simple_compartmental", baselines["simple_compartmental"]),
        ("integrated_champion", {"validation": reference_fit.split_metrics["validation"], "holdout": reference_fit.split_metrics["holdout"]}),
        ("strict_promoted", {"validation": strict_winner.split_metrics["validation"], "holdout": strict_winner.split_metrics["holdout"]}),
    ]
    for label, metrics in rows:
        lines.append(
            f"| `{label}` | {float(metrics['validation']['primary_loss']):.6f} | {float(metrics['holdout']['primary_loss']):.6f} | "
            f"{float(metrics['validation']['diag_flow_loss']):.6f} | {float(metrics['holdout']['diag_flow_loss']):.6f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _export_promotion_paper_figures(ctx: TransitionResearchContext, chart_paths: dict[str, Path]) -> dict[str, Any]:
    archive_dir = ROOT_DIR / "artifacts" / "paper_figures" / "phase3_diag_kernel_20260409"
    archive_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = list(read_json(archive_dir / "figure_manifest.json", default=[]))
    retained = [
        row
        for row in existing_manifest
        if isinstance(row, dict) and not str(row.get("figure_id") or "").startswith("fig_diag02a_")
    ]
    figure_specs = [
        (
            "fig_diag02a_01",
            "Integrated Champion Promotion Frontier",
            "Validation and holdout losses across diagnosis-kernel candidates after freezing the current integrated champion outside the diagnosis branch.",
            "diag02a_candidate_frontier.png",
            chart_paths.get("frontier"),
        ),
        (
            "fig_diag02a_02",
            "Integrated Champion Promotion Comparison",
            "Blocked-time comparison between carry-forward, simple compartmental, the current integrated champion, and the strict-diagnosis promoted branch.",
            "diag02a_model_comparison.png",
            chart_paths.get("comparison"),
        ),
        (
            "fig_diag02a_03",
            "Integrated Champion Promotion Predictions",
            "Observed and predicted new diagnoses and diagnosed stock for the current integrated champion and the strict-diagnosis promoted branch on validation and holdout quarters.",
            "diag02a_predictions.png",
            chart_paths.get("predictions"),
        ),
        (
            "fig_diag02a_04",
            "Integrated Champion Promotion Kernel Profile",
            "Diagnosis age-kernel comparison between the current integrated champion and the promoted strict diagnosis branch.",
            "diag02a_kernel_profile.png",
            chart_paths.get("kernel"),
        ),
    ]
    new_rows: list[dict[str, Any]] = []
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


def run_diag_02a(ctx: TransitionResearchContext) -> dict[str, Any]:
    snapshot = _build_snapshot(ctx)
    reference_config = _select_integrated_stage1_champion_config()
    contract = _build_blocked_time_contract(snapshot)
    candidates = _build_candidates(reference_config, contract)
    fits = [_fit_candidate(snapshot, candidate, reference_config, contract) for candidate in candidates]
    frontier_rows = _frontier_rows(fits)
    reference_fit = _fit_by_candidate_id(fits, reference_config.candidate_id)
    strict_winner = _best_fit(fits, diagnosis_kind="strict")
    best_overall = _best_fit(fits, diagnosis_kind=None)
    baselines = _baseline_payload(snapshot, contract)

    split_contract_path = ctx.experiment_dir / "promotion_blocked_time_contract.json"
    frontier_path = ctx.experiment_dir / "promotion_candidate_frontier.json"
    evaluation_path = ctx.experiment_dir / "promotion_evaluation.json"
    dashboard_path = ctx.experiment_dir / "promotion_dashboard.md"
    comparison_path = ctx.experiment_dir / "promotion_model_comparison.json"
    prediction_rows_path = ctx.experiment_dir / "promotion_prediction_rows.json"
    frontier_chart_path = ctx.experiment_dir / "promotion_candidate_frontier.png"
    model_comparison_chart_path = ctx.experiment_dir / "promotion_model_comparison.png"
    prediction_chart_path = ctx.experiment_dir / "promotion_predictions.png"
    kernel_chart_path = ctx.experiment_dir / "promotion_kernel_profile.png"

    split_contract_payload = {
        "generated_at": utc_now_iso(),
        "reference_candidate_id": str(reference_config.candidate_id),
        "reference_diagnosis_family": str(reference_config.diagnosis_family),
        "reference_care_family": str(reference_config.care_family),
        "reference_hidden_rank": int(reference_config.hidden_rank),
        "reference_use_observation_covariates": bool(reference_config.use_observation_covariates),
        "train_diagnosis_quarters": list(contract.train_diagnosis_quarters),
        "validation_quarters": list(contract.validation_quarters),
        "holdout_quarters": list(contract.holdout_quarters),
    }
    write_json(split_contract_path, split_contract_payload)
    write_json(frontier_path, frontier_rows)

    comparison_payload = {
        "carry_forward": baselines["carry_forward"],
        "simple_compartmental": baselines["simple_compartmental"],
        "integrated_champion": _fit_payload(reference_fit),
        "strict_promoted": _fit_payload(strict_winner),
        "best_overall_candidate_id": str(best_overall.candidate.candidate_id),
    }
    write_json(comparison_path, comparison_payload)

    prediction_rows: list[dict[str, Any]] = []
    relevant_mask = np.logical_or(contract.validation_mask, contract.holdout_mask)
    for idx in np.where(relevant_mask)[0]:
        prediction_rows.append(
            {
                "quarter": str(snapshot.model_quarters[idx]),
                "window": "validation" if bool(contract.validation_mask[idx]) else "holdout",
                "observed_new_diagnosed_cases_period": float(snapshot.metric_values["new_diagnosed_cases_period"][idx])
                if bool(snapshot.metric_masks["new_diagnosed_cases_period"][idx])
                else None,
                "integrated_champion_new_diagnosed_cases_period": float(reference_fit.simulation["predictions"]["new_diagnosed_cases_period"][idx]),
                "strict_promoted_new_diagnosed_cases_period": float(strict_winner.simulation["predictions"]["new_diagnosed_cases_period"][idx]),
                "observed_diagnosed_plhiv": float(snapshot.metric_values["diagnosed_plhiv"][idx]) if bool(snapshot.metric_masks["diagnosed_plhiv"][idx]) else None,
                "integrated_champion_diagnosed_plhiv": float(reference_fit.simulation["predictions"]["diagnosed_plhiv"][idx]),
                "strict_promoted_diagnosed_plhiv": float(strict_winner.simulation["predictions"]["diagnosed_plhiv"][idx]),
            }
        )
    write_json(prediction_rows_path, prediction_rows)

    _write_promotion_frontier_chart(frontier_chart_path, frontier_rows, reference_config.candidate_id)
    _write_promotion_model_comparison_chart(model_comparison_chart_path, reference_fit, strict_winner, baselines)
    _write_promotion_prediction_chart(prediction_chart_path, snapshot, contract, reference_fit, strict_winner)
    _write_promotion_kernel_profile_chart(kernel_chart_path, reference_fit, strict_winner)
    _write_promotion_dashboard_markdown(
        dashboard_path,
        contract=contract,
        reference_fit=reference_fit,
        strict_winner=strict_winner,
        baselines=baselines,
    )
    paper_archive = _export_promotion_paper_figures(
        ctx,
        {
            "frontier": frontier_chart_path,
            "comparison": model_comparison_chart_path,
            "predictions": prediction_chart_path,
            "kernel": kernel_chart_path,
        },
    )

    evaluation_payload = {
        "reference_candidate_id": str(reference_config.candidate_id),
        "integrated_champion": _fit_payload(reference_fit),
        "strict_promoted": _fit_payload(strict_winner),
        "strict_vs_reference_validation_delta": {
            "primary_loss": float(strict_winner.split_metrics["validation"]["primary_loss"] - reference_fit.split_metrics["validation"]["primary_loss"]),
            "diag_flow_loss": float(strict_winner.split_metrics["validation"]["diag_flow_loss"] - reference_fit.split_metrics["validation"]["diag_flow_loss"]),
        },
        "strict_vs_reference_holdout_delta": {
            "primary_loss": float(strict_winner.split_metrics["holdout"]["primary_loss"] - reference_fit.split_metrics["holdout"]["primary_loss"]),
            "diag_flow_loss": float(strict_winner.split_metrics["holdout"]["diag_flow_loss"] - reference_fit.split_metrics["holdout"]["diag_flow_loss"]),
        },
        "strict_vs_carry_forward_validation_delta": {
            "primary_loss": float(strict_winner.split_metrics["validation"]["primary_loss"] - baselines["carry_forward"]["validation"]["primary_loss"]),
            "diag_flow_loss": float(strict_winner.split_metrics["validation"]["diag_flow_loss"] - baselines["carry_forward"]["validation"]["diag_flow_loss"]),
        },
        "strict_vs_carry_forward_holdout_delta": {
            "primary_loss": float(strict_winner.split_metrics["holdout"]["primary_loss"] - baselines["carry_forward"]["holdout"]["primary_loss"]),
            "diag_flow_loss": float(strict_winner.split_metrics["holdout"]["diag_flow_loss"] - baselines["carry_forward"]["holdout"]["diag_flow_loss"]),
        },
        "paper_archive": paper_archive,
    }
    write_json(evaluation_path, evaluation_payload)

    decision = {
        "reference_candidate_id": str(reference_config.candidate_id),
        "promoted_candidate_id": str(strict_winner.candidate.candidate_id),
        "promoted_validation_beats_reference": bool(_score_tuple(strict_winner.split_metrics["validation"]) < _score_tuple(reference_fit.split_metrics["validation"])),
        "promoted_holdout_beats_reference": bool(_score_tuple(strict_winner.split_metrics["holdout"]) < _score_tuple(reference_fit.split_metrics["holdout"])),
        "promoted_validation_beats_carry_forward": bool(_score_tuple(strict_winner.split_metrics["validation"]) < _score_tuple(baselines["carry_forward"]["validation"])),
        "promoted_holdout_beats_carry_forward": bool(_score_tuple(strict_winner.split_metrics["holdout"]) < _score_tuple(baselines["carry_forward"]["holdout"])),
        "promote_to_integrated_branch": bool(
            _score_tuple(strict_winner.split_metrics["validation"]) < _score_tuple(reference_fit.split_metrics["validation"])
            and _score_tuple(strict_winner.split_metrics["holdout"]) < _score_tuple(reference_fit.split_metrics["holdout"])
        ),
        "paper_figure_archive": paper_archive,
    }

    experiment_spec = {
        "variant": "benchmark-hardening-loop",
        "goal": "Promote the strict diagnosis kernel into the current integrated PHASE3-V2 champion while keeping downstream care and all non-diagnosis structure fixed, then compare against the champion under a blocked-time gate.",
        "source_run_id": ctx.source_run_id,
        "reference_integrated_run": str(_latest_integrated_experiment_dir()) if _latest_integrated_experiment_dir() is not None else None,
        "reference_candidate_id": str(reference_config.candidate_id),
        "mutation_unit": "diagnosis kernel only on top of the integrated stage1 champion",
        "blocked_time_contract": split_contract_payload,
        "candidate_ids": [str(candidate.candidate_id) for candidate in candidates],
        "artifacts": {
            "promotion_blocked_time_contract": str(split_contract_path),
            "promotion_candidate_frontier": str(frontier_path),
            "promotion_evaluation": str(evaluation_path),
            "promotion_dashboard": str(dashboard_path),
            "promotion_model_comparison": str(comparison_path),
            "promotion_prediction_rows": str(prediction_rows_path),
        },
    }
    coverage_summary = {
        "historical_quarter_count": int(len(snapshot.historical_quarters)),
        "train_quarter_count": int(np.sum(contract.train_mask)),
        "validation_quarter_count": int(np.sum(contract.validation_mask)),
        "holdout_quarter_count": int(np.sum(contract.holdout_mask)),
        "diagnosis_flow_observed_quarter_count": int(len(contract.diagnosis_flow_observed_quarters)),
        "train_diagnosis_flow_observed_quarter_count": int(len(contract.train_diagnosis_quarters)),
        "validation_diagnosis_flow_observed_quarter_count": int(len(contract.validation_quarters)),
        "holdout_diagnosis_flow_observed_quarter_count": int(len(contract.holdout_quarters)),
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            numerical_guard_entry(
                name="diag02a_strict_promotion_eps",
                role="diag02a_numerical_guard",
                why_needed="Prevents divide-by-zero and log singularities while promoting the strict diagnosis kernel into the integrated champion under blocked-time evaluation.",
            ),
            {
                "name": "diagnosis_flow_observed_quarter_count",
                "value": int(len(contract.diagnosis_flow_observed_quarters)),
                "role": "diag02a_blocked_time_support",
                "source_type": "estimated",
                "estimation_data": "Historical quarters with observed new_diagnosed_cases_period in the active national snapshot",
                "estimation_method": "count(snapshot.historical_quarters intersect diagnosis-flow observation mask)",
                "uncertainty": "deterministic given the frozen snapshot",
                "why_needed": "Defines the admissible blocked-time split for integrated-champion diagnosis-kernel promotion without manual year cutoffs.",
            },
            {
                "name": "strict_kernel_width_limit",
                "value": int(len(contract.train_diagnosis_quarters)),
                "role": "diag02a_strict_kernel_search_ceiling",
                "source_type": "estimated",
                "estimation_data": "Training diagnosis-flow support after contiguous blocked-time partition",
                "estimation_method": "count(train diagnosis-flow quarters)",
                "uncertainty": "deterministic given the frozen blocked-time split",
                "why_needed": "Caps strict diagnosis-kernel flexibility by the amount of direct diagnosis-flow support available while keeping the integrated champion fixed outside the diagnosis branch.",
            },
        ],
    )
    return {
        "artifacts": artifacts,
        "promotion_blocked_time_contract": split_contract_payload,
        "candidate_frontier": frontier_rows,
        "evaluation": evaluation_payload,
        "decision": decision,
        "paper_archive": paper_archive,
    }


def _export_budget_paper_figures(ctx: TransitionResearchContext, chart_paths: dict[str, Path]) -> dict[str, Any]:
    archive_dir = ROOT_DIR / "artifacts" / "paper_figures" / "phase3_diag_kernel_20260409"
    archive_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = list(read_json(archive_dir / "figure_manifest.json", default=[]))
    retained = [
        row
        for row in existing_manifest
        if isinstance(row, dict) and not str(row.get("figure_id") or "").startswith("fig_diag01b_")
    ]
    figure_specs = [
        (
            "fig_diag01b_01",
            "Diagnosis Budget Sweep Loss Curves",
            "Validation and holdout loss curves for hazard, current delay, and strict diagnosis-kernel candidates under a predeclared compute-budget ladder.",
            "diag01b_budget_loss_curves.png",
            chart_paths.get("loss_curves"),
        ),
        (
            "fig_diag01b_02",
            "Diagnosis Budget Sweep Champion Trace",
            "Best validation and holdout scores attained at each compute-budget rung in the diagnosis-kernel sweep.",
            "diag01b_budget_champion_trace.png",
            chart_paths.get("champion_trace"),
        ),
        (
            "fig_diag01b_03",
            "Diagnosis Budget Sweep Compute Scatter",
            "Relationship between optimizer effort and diagnosis-kernel score across all evaluated budget rungs.",
            "diag01b_budget_compute_scatter.png",
            chart_paths.get("compute_scatter"),
        ),
    ]
    new_rows: list[dict[str, Any]] = []
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


def run_diag_01b(ctx: TransitionResearchContext) -> dict[str, Any]:
    snapshot = _build_snapshot(ctx)
    reference_config = _select_reference_configs()["diagnosis"]
    contract = _build_blocked_time_contract(snapshot)
    candidates = _build_candidates(reference_config, contract)
    multipliers = _budget_multipliers(contract)
    ladder_fits = {
        str(candidate.candidate_id): _fit_budget_ladder(snapshot, candidate, reference_config, contract, multipliers)
        for candidate in candidates
    }
    budget_payload = _budget_frontier_payload(ladder_fits, multipliers)
    budget_rows = list(budget_payload["rows"])
    champion_rows = list(budget_payload["champion_rows"])
    validation_champions = [str(row["validation_champion_candidate_id"]) for row in champion_rows]
    holdout_champions = [str(row["holdout_champion_candidate_id"]) for row in champion_rows]

    split_contract_path = ctx.experiment_dir / "budget_sweep_contract.json"
    rows_path = ctx.experiment_dir / "budget_sweep_rows.json"
    champion_path = ctx.experiment_dir / "budget_sweep_champions.json"
    summary_path = ctx.experiment_dir / "budget_sweep_summary.json"
    dashboard_path = ctx.experiment_dir / "budget_sweep_dashboard.md"
    loss_curves_path = ctx.experiment_dir / "budget_sweep_loss_curves.png"
    champion_trace_path = ctx.experiment_dir / "budget_sweep_champion_trace.png"
    compute_scatter_path = ctx.experiment_dir / "budget_sweep_compute_scatter.png"

    split_contract_payload = {
        "generated_at": utc_now_iso(),
        "blocked_time_contract": {
            "train_diagnosis_quarters": list(contract.train_diagnosis_quarters),
            "validation_quarters": list(contract.validation_quarters),
            "holdout_quarters": list(contract.holdout_quarters),
        },
        "budget_multipliers": [float(value) for value in multipliers],
        "reference_candidate_id": str(reference_config.candidate_id),
    }
    write_json(split_contract_path, split_contract_payload)
    write_json(rows_path, budget_rows)
    write_json(champion_path, champion_rows)

    champion_candidates = [fits[-1] for fits in ladder_fits.values()]
    best_final_validation = min(champion_candidates, key=lambda fit: _score_tuple(fit.split_metrics["validation"]))
    best_final_holdout = min(champion_candidates, key=lambda fit: _score_tuple(fit.split_metrics["holdout"]))
    summary_payload = {
        "best_final_validation": _fit_payload(best_final_validation),
        "best_final_holdout": _fit_payload(best_final_holdout),
        "validation_champions_by_budget": validation_champions,
        "holdout_champions_by_budget": holdout_champions,
        "validation_ranking_stable": len(set(validation_champions)) == 1,
        "holdout_ranking_stable": len(set(holdout_champions)) == 1,
        "candidate_budget_depth": {candidate_id: int(len(fits)) for candidate_id, fits in ladder_fits.items()},
    }
    write_json(summary_path, summary_payload)

    _write_budget_loss_curves(loss_curves_path, ladder_fits, multipliers)
    _write_budget_champion_trace(champion_trace_path, champion_rows)
    _write_budget_compute_scatter(compute_scatter_path, budget_rows)
    paper_archive = _export_budget_paper_figures(
        ctx,
        {
            "loss_curves": loss_curves_path,
            "champion_trace": champion_trace_path,
            "compute_scatter": compute_scatter_path,
        },
    )

    markdown_lines = [
        "# DIAG-01B Diagnosis Budget Sweep Dashboard",
        "",
        "This audit reruns the diagnosis candidates under a predeclared compute-budget ladder with warm starts and identical blocked-time splits.",
        "",
        f"- Budget multipliers: `{', '.join(str(int(value)) if float(value).is_integer() else str(value) for value in multipliers)}`",
        f"- Validation champions by budget: `{', '.join(validation_champions)}`",
        f"- Holdout champions by budget: `{', '.join(holdout_champions)}`",
        f"- Final validation champion: `{best_final_validation.candidate.candidate_id}`",
        f"- Final holdout champion: `{best_final_holdout.candidate.candidate_id}`",
        "",
    ]
    dashboard_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    decision = {
        "reference_candidate_id": str(reference_config.candidate_id),
        "budget_multipliers": [float(value) for value in multipliers],
        "best_final_validation_candidate_id": str(best_final_validation.candidate.candidate_id),
        "best_final_holdout_candidate_id": str(best_final_holdout.candidate.candidate_id),
        "validation_ranking_stable": bool(len(set(validation_champions)) == 1),
        "holdout_ranking_stable": bool(len(set(holdout_champions)) == 1),
        "paper_figure_archive": paper_archive,
    }
    experiment_spec = {
        "variant": "benchmark-hardening-loop",
        "goal": "Audit whether the diagnosis-kernel ranking is stable under a predeclared compute-budget ladder before increasing optimizer effort in a publication setting.",
        "source_run_id": ctx.source_run_id,
        "reference_candidate_id": str(reference_config.candidate_id),
        "budget_multipliers": [float(value) for value in multipliers],
        "mutation_unit": "optimizer budget only",
        "artifacts": {
            "budget_sweep_contract": str(split_contract_path),
            "budget_sweep_rows": str(rows_path),
            "budget_sweep_champions": str(champion_path),
            "budget_sweep_summary": str(summary_path),
            "budget_sweep_dashboard": str(dashboard_path),
        },
    }
    coverage_summary = {
        "candidate_count": int(len(candidates)),
        "budget_count": int(len(multipliers)),
        "evaluated_budget_rows": int(len(budget_rows)),
        "train_diagnosis_flow_observed_quarter_count": int(len(contract.train_diagnosis_quarters)),
        "validation_diagnosis_flow_observed_quarter_count": int(len(contract.validation_quarters)),
        "holdout_diagnosis_flow_observed_quarter_count": int(len(contract.holdout_quarters)),
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            numerical_guard_entry(
                name="diag_budget_sweep_eps",
                role="diagnosis_budget_sweep_numerical_guard",
                why_needed="Prevents divide-by-zero and log singularities while sweeping optimizer budgets over the strict diagnosis kernel candidates.",
            ),
            {
                "name": "budget_multiplier_ceiling",
                "value": float(max(multipliers)),
                "role": "diagnosis_budget_sweep_ceiling",
                "source_type": "estimated",
                "estimation_data": "Training diagnosis-flow support count after blocked-time split",
                "estimation_method": "powers-of-two ladder capped at train diagnosis-flow quarter count",
                "uncertainty": "deterministic given the frozen contract",
                "why_needed": "Predeclares a finite compute ladder instead of increasing budget post hoc until a preferred branch wins.",
            },
        ],
    )
    return {
        "artifacts": artifacts,
        "budget_sweep_contract": split_contract_payload,
        "budget_rows": budget_rows,
        "budget_champions": champion_rows,
        "summary": summary_payload,
        "decision": decision,
        "paper_archive": paper_archive,
    }


__all__ = ["run_diag_01a", "run_diag_01b", "run_diag_02a", "_build_blocked_time_contract", "_build_candidates", "_budget_multipliers"]
