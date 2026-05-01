from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.special import expit

from epigraph_ph.phase3._lineage.national_reset_core import quarter_sort_key
from epigraph_ph.runtime import read_json, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .numeric_policy import numerical_guard_entry
from .registry import TRANSITION_NAMES
from .sources import load_transition_research_inputs
from .transition_engine import _build_observation_payload


STATE_NAMES: tuple[str, ...] = ("U", "D", "A", "V", "L")
PRIMARY_METRICS: tuple[str, ...] = ("new_diagnosed_cases_period", "diagnosed_plhiv", "alive_on_art")
SECONDARY_METRICS: tuple[str, ...] = ("tested_for_viral_load", "virally_suppressed")
TOTAL_METRIC = "estimated_plhiv"
CASCADE_TARGET = 0.95
TRANSITION_HOOKS: dict[str, str] = {
    "U_to_D": "diagnosis_transitions",
    "D_to_A": "linkage_transitions",
    "A_to_V": "suppression_transitions",
    "A_to_L": "retention_attrition_transitions",
    "L_to_A": "retention_attrition_transitions",
}


@dataclass(slots=True)
class CandidateConfig:
    candidate_id: str
    diagnosis_family: str
    care_family: str
    hidden_rank: int
    use_observation_covariates: bool


@dataclass(slots=True)
class IntegratedSnapshot:
    source_run_id: str
    model_quarters: list[str]
    historical_quarters: list[str]
    future_quarters: list[str]
    quarter_index: dict[str, int]
    metric_values: dict[str, np.ndarray]
    metric_masks: dict[str, np.ndarray]
    metric_scales: dict[str, float]
    population_denominator: np.ndarray
    factor_matrix: np.ndarray
    feature_ids: list[str]
    feature_rows_by_id: dict[str, dict[str, Any]]
    incidence_feature_ids: list[str]
    transition_feature_ids: dict[str, list[str]]
    observation_feature_ids: dict[str, list[str]]
    hidden_scores: np.ndarray
    hidden_rank_max: int
    start_total: float
    start_diag: float
    start_art: float
    start_art_observed: bool
    historical_end_quarter: str
    historical_years: list[int]
    target_horizon: str
    payload_rows: list[dict[str, Any]]
    evidence_provenance: dict[str, Any]
    phase2_insertion_contract: dict[str, Any]


@dataclass(slots=True)
class FitResult:
    config: CandidateConfig
    success: bool
    status: int
    message: str
    train_cost: float
    param_vector: np.ndarray
    param_slices: dict[str, slice]
    matrices: dict[str, Any]
    simulation: dict[str, Any]
    window_metrics: dict[str, dict[str, float]]
    year_metrics: dict[str, dict[str, float]]
    feature_summary: dict[str, Any]


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


def _quarter_from_month(month_label: str) -> str:
    year_text, month_text = str(month_label).split("-", 1)
    month_value = int(month_text[:2])
    return f"{int(year_text):04d}-Q{((month_value - 1) // 3) + 1}"


def _quarter_year(quarter: str) -> int:
    return int(str(quarter).split("-", 1)[0])


def _quarter_sequence(start_quarter: str, end_quarter: str) -> list[str]:
    start_year, start_q = _quarter_year(start_quarter), int(str(start_quarter).split("Q", 1)[1])
    end_year, end_q = _quarter_year(end_quarter), int(str(end_quarter).split("Q", 1)[1])
    quarters: list[str] = []
    year_value = start_year
    quarter_value = start_q
    while (year_value, quarter_value) <= (end_year, end_q):
        quarters.append(f"{year_value:04d}-Q{quarter_value}")
        quarter_value += 1
        if quarter_value > 4:
            year_value += 1
            quarter_value = 1
    return quarters


def _quarter_mask_between(snapshot: IntegratedSnapshot, start_quarter: str, end_quarter: str) -> np.ndarray:
    return np.asarray(
        [
            quarter_sort_key(start_quarter) <= quarter_sort_key(quarter) <= quarter_sort_key(end_quarter)
            for quarter in snapshot.model_quarters
        ],
        dtype=bool,
    )


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return expit(np.asarray(values, dtype=np.float64))


def _cloglog_inverse(eta: np.ndarray | float) -> np.ndarray | float:
    eta_array = np.asarray(eta, dtype=np.float64)
    log_float_max = float(np.log(np.finfo(np.float64).max))
    clipped = np.minimum(eta_array, log_float_max)
    return 1.0 - np.exp(-np.exp(clipped))


def _safe_exp(values: np.ndarray | float) -> np.ndarray | float:
    value_array = np.asarray(values, dtype=np.float64)
    log_float_max = float(np.log(np.finfo(np.float64).max))
    clipped = np.minimum(value_array, log_float_max)
    return np.exp(clipped)


def _smooth_abs(values: np.ndarray, eps: float) -> np.ndarray:
    return np.sqrt(np.square(values) + eps)


def _safe_std(values: np.ndarray, eps: float) -> float:
    std_value = float(np.std(np.asarray(values, dtype=np.float64)))
    return std_value if std_value > eps else eps


def _safe_scale(values: list[float], eps: float) -> float:
    finite = [abs(float(value)) for value in values if np.isfinite(value)]
    return max(finite) if finite else eps


def _write_line_chart(path: Path, title: str, x_labels: list[str], series: list[tuple[str, list[float]]], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for label, values in series:
        sanitized = [float(value) if np.isfinite(value) else np.nan for value in values]
        ax.plot(x_labels, sanitized, marker="o", linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_population_rows(_ctx: TransitionResearchContext) -> tuple[list[dict[str, Any]], Path]:
    wpp_path = Path("D:/EpiGraph_PH/src/epigraph_ph/phase3/incidence/official_population_denominator_phl_wpp2024_2010_2025.json")
    wb_path = Path("D:/EpiGraph_PH/src/epigraph_ph/phase3/incidence/official_population_denominator_phl_wb_2010_2024.json")
    for path in (wpp_path, wb_path):
        if not path.exists():
            continue
        payload = read_json(path, default={})
        rows = list(payload.get("rows") or []) if isinstance(payload, dict) else []
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            year_value = row.get("year")
            population_total = row.get("population_total")
            if year_value in (None, "") or population_total in (None, ""):
                continue
            normalized.append({"year": int(year_value), "population_total": float(population_total)})
        if normalized:
            normalized.sort(key=lambda row: int(row["year"]))
            return normalized, path
    raise FileNotFoundError("No official Phase 3 population denominator file found")


def _latest_supported_total_lookup(payload: dict[str, Any]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for quarter, value in dict(payload.get("estimated_plhiv_by_quarter") or {}).items():
        try:
            lookup[str(quarter)] = float(value)
        except (TypeError, ValueError):
            continue
    return lookup


def _first_supported_quarter(rows: list[dict[str, Any]], total_lookup: dict[str, float]) -> str:
    candidates = [
        str(row.get("quarter") or "")
        for row in rows
        if row.get("diagnosed_plhiv") is not None
        and str(row.get("quarter") or "") in total_lookup
    ]
    if not candidates:
        raise ValueError("Integrated Phase 3 frontier requires at least one quarter with diagnosed and estimated-total support")
    return min(candidates, key=quarter_sort_key)


def _metric_evidence_tier(metric_name: str, provenance: dict[str, Any] | None) -> str:
    provenance = dict(provenance or {})
    measurement_class = str(provenance.get("measurement_class") or "").lower()
    source_bank = str(provenance.get("source_bank") or "").lower()
    series_kind = str(provenance.get("series_kind") or "").lower()
    value_semantics = str(provenance.get("value_semantics") or "").lower()
    if metric_name == TOTAL_METRIC:
        return "model_estimated_total"
    if measurement_class == "model_estimate":
        return "model_estimate"
    if value_semantics == "direct_observed":
        return "direct_observed_anchor"
    if measurement_class == "program_observed_harp" or source_bank == "phase0_extracted":
        if series_kind == "monthly_aggregated_to_quarter":
            return "archive_aggregated_program_observation"
        return "archive_program_observation"
    if provenance:
        return "archive_derived_observation"
    return "unknown"


def _estimated_total_provenance_summary(payload: dict[str, Any]) -> dict[str, Any]:
    provenance_by_quarter = dict(payload.get("estimated_plhiv_provenance_by_quarter") or {})
    quarters = sorted((str(quarter) for quarter in provenance_by_quarter.keys()), key=quarter_sort_key)
    tier_counts: dict[str, int] = {}
    source_ids: set[str] = set()
    source_labels: set[str] = set()
    for quarter in quarters:
        provenance = dict(provenance_by_quarter.get(quarter) or {})
        tier = _metric_evidence_tier(TOTAL_METRIC, provenance)
        tier_counts[tier] = int(tier_counts.get(tier, 0) + 1)
        source_ids.update(str(item) for item in list(provenance.get("source_ids") or []) if str(item))
        source_labels.update(str(item) for item in list(provenance.get("source_labels") or []) if str(item))
        if provenance.get("source_id"):
            source_ids.add(str(provenance["source_id"]))
        if provenance.get("source_label"):
            source_labels.add(str(provenance["source_label"]))
    return {
        "metric_name": TOTAL_METRIC,
        "role": "auxiliary_latent_population_size",
        "claim_boundary": "Estimated PLHIV is not direct training truth; it constrains latent total size and cascade denominators.",
        "observed_quarter_count": int(len(quarters)),
        "first_observed_quarter": quarters[0] if quarters else None,
        "last_observed_quarter": quarters[-1] if quarters else None,
        "evidence_tier_counts": tier_counts,
        "source_ids": sorted(source_ids),
        "source_labels": sorted(source_labels),
    }


def _build_evidence_provenance(payload: dict[str, Any], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_metric_rows: dict[str, list[tuple[str, dict[str, Any]]]] = {metric_name: [] for metric_name in PRIMARY_METRICS + SECONDARY_METRICS}
    for row in source_rows:
        quarter = str(row.get("quarter") or "")
        metric_provenance = dict(row.get("metric_provenance") or {})
        for metric_name in PRIMARY_METRICS + SECONDARY_METRICS:
            if row.get(metric_name) is None:
                continue
            provenance = dict(metric_provenance.get(metric_name) or {})
            per_metric_rows[metric_name].append((quarter, provenance))
    metric_rows: dict[str, Any] = {}
    for metric_name, quarter_rows in per_metric_rows.items():
        if not quarter_rows:
            metric_rows[metric_name] = {
                "metric_name": metric_name,
                "role": "unused",
                "claim_boundary": "No active observed support",
                "observed_quarter_count": 0,
                "first_observed_quarter": None,
                "last_observed_quarter": None,
                "evidence_tier_counts": {},
                "source_ids": [],
                "source_labels": [],
            }
            continue
        ordered_quarters = sorted((quarter for quarter, _ in quarter_rows), key=quarter_sort_key)
        source_ids: set[str] = set()
        source_labels: set[str] = set()
        tier_counts: dict[str, int] = {}
        for _quarter, provenance in quarter_rows:
            tier = _metric_evidence_tier(metric_name, provenance)
            tier_counts[tier] = int(tier_counts.get(tier, 0) + 1)
            source_ids.update(str(item) for item in list(provenance.get("source_ids") or []) if str(item))
            source_labels.update(str(item) for item in list(provenance.get("source_labels") or []) if str(item))
        if metric_name in PRIMARY_METRICS:
            role = "primary_training_target"
        elif metric_name in SECONDARY_METRICS:
            role = "secondary_observation_process_target"
        else:
            role = "auxiliary"
        metric_rows[metric_name] = {
            "metric_name": metric_name,
            "role": role,
            "claim_boundary": (
                "Primary calibration target"
                if metric_name in PRIMARY_METRICS
                else "Observed service/documentation process target with weaker identifiability than the front-half cascade"
            ),
            "observed_quarter_count": int(len(ordered_quarters)),
            "first_observed_quarter": ordered_quarters[0],
            "last_observed_quarter": ordered_quarters[-1],
            "evidence_tier_counts": tier_counts,
            "source_ids": sorted(source_ids),
            "source_labels": sorted(source_labels),
        }
    metric_rows[TOTAL_METRIC] = _estimated_total_provenance_summary(payload)
    return {
        "summary": {
            "direct_vs_contextual_split": {
                "direct_observed_anchor_metrics": [
                    metric_name
                    for metric_name, row in metric_rows.items()
                    if int(dict(row.get("evidence_tier_counts") or {}).get("direct_observed_anchor", 0)) > 0
                ],
                "archive_program_observation_metrics": [
                    metric_name
                    for metric_name, row in metric_rows.items()
                    if sum(
                        int(dict(row.get("evidence_tier_counts") or {}).get(tier, 0))
                        for tier in ("archive_program_observation", "archive_aggregated_program_observation", "archive_derived_observation")
                    )
                    > 0
                ],
                "model_estimated_metrics": [
                    metric_name
                    for metric_name, row in metric_rows.items()
                    if int(dict(row.get("evidence_tier_counts") or {}).get("model_estimated_total", 0)) > 0
                    or int(dict(row.get("evidence_tier_counts") or {}).get("model_estimate", 0)) > 0
                ],
            },
            "publication_note": "National Phase 3 uses a mixed evidence stack: archive-derived program observations for most quarterly targets, direct extracted anchors in limited places, and model-estimated totals for latent population size.",
        },
        "metrics": metric_rows,
    }


def _phase2_insertion_contract(snapshot: IntegratedSnapshot) -> dict[str, Any]:
    return {
        "direct_phase2_terms_role": "ordinary fitted covariates in incidence and transition linear predictors",
        "hidden_phase2_terms_role": "shared latent shocks extracted by SVD from the national historical factor matrix",
        "structured_prior_status": False,
        "province_resolved_hidden_dynamics_status": False,
        "claim_boundary": "This implementation uses Phase 2 as direct covariates plus national hidden scores. It does not yet implement literature-strength structured priors or province-resolved hidden dynamics.",
        "feature_count": int(len(snapshot.feature_ids)),
        "hidden_rank_max": int(snapshot.hidden_rank_max),
    }


def _feature_matrix_for_quarters(
    *,
    model_quarters: list[str],
    feature_ids: list[str],
    month_axis: list[str],
    factor_index: dict[str, int],
    tensor: np.ndarray,
    train_mask: np.ndarray,
    eps: float,
) -> np.ndarray:
    month_quarters = [_quarter_from_month(month_label) for month_label in month_axis]
    ordered_indices = [idx for _, idx in sorted((quarter_sort_key(month_quarters[idx]), idx) for idx in range(len(month_axis)))]
    sorted_month_quarters = [month_quarters[idx] for idx in ordered_indices]
    quarter_matrix = np.zeros((len(model_quarters), len(feature_ids)), dtype=np.float64)
    for feature_col, factor_id in enumerate(feature_ids):
        factor_idx = factor_index.get(factor_id)
        if factor_idx is None:
            continue
        sorted_values = np.asarray(tensor[0, ordered_indices, factor_idx], dtype=np.float64)
        pointer = 0
        latest_value = float(sorted_values[0]) if sorted_values.size else 0.0
        for quarter_row, quarter in enumerate(model_quarters):
            while pointer + 1 < len(sorted_month_quarters) and quarter_sort_key(sorted_month_quarters[pointer + 1]) <= quarter_sort_key(quarter):
                pointer += 1
                latest_value = float(sorted_values[pointer])
            quarter_matrix[quarter_row, feature_col] = latest_value
    train_values = quarter_matrix[train_mask]
    means = np.mean(train_values, axis=0) if train_values.size else np.zeros((quarter_matrix.shape[1],), dtype=np.float64)
    stds = np.std(train_values, axis=0) if train_values.size else np.ones((quarter_matrix.shape[1],), dtype=np.float64)
    stds = np.where(stds > eps, stds, 1.0)
    return (quarter_matrix - means[None, :]) / stds[None, :]


def _transition_feature_ids(retained_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {transition: [] for transition in TRANSITION_NAMES}
    for row in retained_rows:
        factor_id = str(row.get("factor_id") or "")
        hooks = {str(value) for value in list(row.get("transition_hooks") or [])}
        for transition, hook_name in TRANSITION_HOOKS.items():
            if hook_name in hooks:
                mapping[transition].append(factor_id)
    return {key: sorted(set(values)) for key, values in mapping.items()}


def _observation_feature_ids(retained_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    testing_ids = sorted({str(row.get("factor_id") or "") for row in retained_rows if str(row.get("best_target") or "") == "testing_coverage"})
    suppression_ids = sorted({str(row.get("factor_id") or "") for row in retained_rows if str(row.get("best_target") or "") == "documented_suppression"})
    return {"tested_for_viral_load": testing_ids, "virally_suppressed": suppression_ids}


def _build_snapshot(ctx: TransitionResearchContext) -> IntegratedSnapshot:
    eps = float(np.finfo(np.float64).eps)
    payload = _build_observation_payload(ctx)
    inputs = load_transition_research_inputs(ctx)
    total_lookup = _latest_supported_total_lookup(payload)
    source_rows = [dict(row) for row in list(payload.get("rows") or [])]
    start_quarter = _first_supported_quarter(source_rows, total_lookup)
    population_rows, _population_path = _load_population_rows(ctx)
    population_lookup = {int(row["year"]): float(row["population_total"]) for row in population_rows}
    max_factor_year = max(int(str(month_label).split("-", 1)[0]) for month_label in inputs.month_axis)
    horizon_year = min(max(population_lookup), max_factor_year)
    target_horizon = f"{horizon_year:04d}-Q4"
    model_quarters = _quarter_sequence(start_quarter, target_horizon)
    quarter_index = {quarter: idx for idx, quarter in enumerate(model_quarters)}
    metric_values: dict[str, np.ndarray] = {
        metric_name: np.full((len(model_quarters),), np.nan, dtype=np.float64)
        for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,)
    }
    metric_masks: dict[str, np.ndarray] = {
        metric_name: np.zeros((len(model_quarters),), dtype=bool)
        for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,)
    }
    for row in source_rows:
        quarter = str(row.get("quarter") or "")
        if quarter not in quarter_index:
            continue
        idx = quarter_index[quarter]
        for metric_name in PRIMARY_METRICS + SECONDARY_METRICS:
            value = row.get(metric_name)
            if value is None:
                continue
            metric_values[metric_name][idx] = float(value)
            metric_masks[metric_name][idx] = True
        total_value = total_lookup.get(quarter)
        if total_value is not None:
            metric_values[TOTAL_METRIC][idx] = float(total_value)
            metric_masks[TOTAL_METRIC][idx] = True
    observed_quarters = [
        quarter
        for quarter in model_quarters
        if any(bool(metric_masks[metric_name][quarter_index[quarter]]) for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,))
    ]
    if not observed_quarters:
        raise ValueError("Integrated Phase 3 frontier requires at least one observed quarter in the national payload")
    historical_end_quarter = max(observed_quarters, key=quarter_sort_key)
    historical_quarters = [quarter for quarter in model_quarters if quarter_sort_key(quarter) <= quarter_sort_key(historical_end_quarter)]
    future_quarters = [quarter for quarter in model_quarters if quarter_sort_key(quarter) > quarter_sort_key(historical_end_quarter)]
    historical_mask = np.asarray([quarter_sort_key(quarter) <= quarter_sort_key(historical_end_quarter) for quarter in model_quarters], dtype=bool)
    start_idx = quarter_index[start_quarter]
    start_total = float(metric_values[TOTAL_METRIC][start_idx])
    start_diag = float(metric_values["diagnosed_plhiv"][start_idx])
    start_art = float(metric_values["alive_on_art"][start_idx]) if metric_masks["alive_on_art"][start_idx] else float("nan")
    if not np.isfinite(start_total) or not np.isfinite(start_diag):
        raise ValueError("Integrated Phase 3 frontier requires finite diagnosed and estimated-total support at the start quarter")
    population_denominator = np.asarray([float(population_lookup[_quarter_year(quarter)]) for quarter in model_quarters], dtype=np.float64)
    retained_rows = [dict(row) for row in list(inputs.retained_factor_rows or []) if str(row.get("factor_id") or "") in inputs.factor_index]
    feature_ids = sorted({str(row.get("factor_id") or "") for row in retained_rows if str(row.get("factor_id") or "")})
    feature_rows_by_id = {str(row.get("factor_id") or ""): dict(row) for row in retained_rows}
    factor_matrix = _feature_matrix_for_quarters(
        model_quarters=model_quarters,
        feature_ids=feature_ids,
        month_axis=inputs.month_axis,
        factor_index=inputs.factor_index,
        tensor=np.asarray(inputs.national_tensor, dtype=np.float64),
        train_mask=historical_mask,
        eps=eps,
    )
    historical_feature_matrix = factor_matrix[historical_mask]
    if historical_feature_matrix.size == 0:
        raise ValueError("Integrated Phase 3 frontier requires historical feature rows")
    _u_matrix, singular_values, vh_matrix = np.linalg.svd(historical_feature_matrix, full_matrices=False)
    if singular_values.size == 0:
        hidden_rank_max = 0
        hidden_scores = np.zeros((len(model_quarters), 0), dtype=np.float64)
    else:
        tolerance = singular_values[0] * float(max(historical_feature_matrix.shape)) * np.finfo(np.float64).eps
        hidden_rank_max = int(np.sum(singular_values > tolerance))
        hidden_scores = factor_matrix @ vh_matrix[:hidden_rank_max].T if hidden_rank_max > 0 else np.zeros((len(model_quarters), 0), dtype=np.float64)
    metric_scales: dict[str, float] = {}
    for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,):
        observed_historical = metric_values[metric_name][np.logical_and(metric_masks[metric_name], historical_mask)]
        metric_scales[metric_name] = _safe_scale(list(observed_historical), eps)
    transition_feature_ids = _transition_feature_ids(retained_rows)
    observation_feature_ids = _observation_feature_ids(retained_rows)
    evidence_provenance = _build_evidence_provenance(payload, source_rows)
    snapshot = IntegratedSnapshot(
        source_run_id=ctx.source_run_id,
        model_quarters=model_quarters,
        historical_quarters=historical_quarters,
        future_quarters=future_quarters,
        quarter_index=quarter_index,
        metric_values=metric_values,
        metric_masks=metric_masks,
        metric_scales=metric_scales,
        population_denominator=population_denominator,
        factor_matrix=factor_matrix,
        feature_ids=feature_ids,
        feature_rows_by_id=feature_rows_by_id,
        incidence_feature_ids=list(feature_ids),
        transition_feature_ids=transition_feature_ids,
        observation_feature_ids=observation_feature_ids,
        hidden_scores=hidden_scores,
        hidden_rank_max=hidden_rank_max,
        start_total=start_total,
        start_diag=start_diag,
        start_art=start_art,
        start_art_observed=bool(metric_masks["alive_on_art"][start_idx]),
        historical_end_quarter=historical_end_quarter,
        historical_years=sorted({_quarter_year(quarter) for quarter in historical_quarters}),
        target_horizon=target_horizon,
        payload_rows=source_rows,
        evidence_provenance=evidence_provenance,
        phase2_insertion_contract={},
    )
    snapshot.phase2_insertion_contract = _phase2_insertion_contract(snapshot)
    return snapshot


def _build_candidate_configs(snapshot: IntegratedSnapshot) -> list[CandidateConfig]:
    observation_modes = [False]
    if snapshot.observation_feature_ids["tested_for_viral_load"] or snapshot.observation_feature_ids["virally_suppressed"]:
        observation_modes.append(True)
    configs: list[CandidateConfig] = []
    for use_observation_covariates in observation_modes:
        for hidden_rank in range(snapshot.hidden_rank_max + 1):
            configs.append(
                CandidateConfig(
                    candidate_id=f"diag-hazard-care-markov-h{hidden_rank:02d}-obs{'on' if use_observation_covariates else 'off'}",
                    diagnosis_family="hazard",
                    care_family="markov",
                    hidden_rank=hidden_rank,
                    use_observation_covariates=use_observation_covariates,
                )
            )
    return configs


def _branch_candidate_configs(snapshot: IntegratedSnapshot, champion: CandidateConfig) -> list[CandidateConfig]:
    observation_modes = [False]
    if snapshot.observation_feature_ids["tested_for_viral_load"] or snapshot.observation_feature_ids["virally_suppressed"]:
        observation_modes.append(True)
    branch_pairs = [
        ("delay", "markov"),
        ("hazard", "semi_markov"),
        ("delay", "semi_markov"),
    ]
    configs: list[CandidateConfig] = []
    for diagnosis_family, care_family in branch_pairs:
        for use_observation_covariates in observation_modes:
            configs.append(
                CandidateConfig(
                    candidate_id=f"diag-{diagnosis_family}-care-{care_family}-h{champion.hidden_rank:02d}-obs{'on' if use_observation_covariates else 'off'}",
                    diagnosis_family=diagnosis_family,
                    care_family=care_family,
                    hidden_rank=champion.hidden_rank,
                    use_observation_covariates=use_observation_covariates,
                )
            )
    deduped: list[CandidateConfig] = []
    seen: set[str] = set()
    for config in configs:
        if config.candidate_id in seen:
            continue
        seen.add(config.candidate_id)
        deduped.append(config)
    return deduped


def _select_matrix(snapshot: IntegratedSnapshot, feature_ids: list[str]) -> np.ndarray:
    if not feature_ids:
        return np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64)
    indices = [snapshot.feature_ids.index(feature_id) for feature_id in feature_ids]
    return np.asarray(snapshot.factor_matrix[:, indices], dtype=np.float64)


def _candidate_matrices(snapshot: IntegratedSnapshot, config: CandidateConfig) -> dict[str, Any]:
    hidden = np.asarray(snapshot.hidden_scores[:, : config.hidden_rank], dtype=np.float64)
    return {
        "incidence_direct": _select_matrix(snapshot, snapshot.incidence_feature_ids),
        "transition_direct": {
            transition: _select_matrix(snapshot, snapshot.transition_feature_ids.get(transition, [])) for transition in TRANSITION_NAMES
        },
        "observation_direct": {
            "tested_for_viral_load": _select_matrix(snapshot, snapshot.observation_feature_ids["tested_for_viral_load"])
            if config.use_observation_covariates
            else np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64),
            "virally_suppressed": _select_matrix(snapshot, snapshot.observation_feature_ids["virally_suppressed"])
            if config.use_observation_covariates
            else np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64),
        },
        "hidden": hidden,
        "feature_ids": {
            "incidence": list(snapshot.incidence_feature_ids),
            "transition": {transition: list(snapshot.transition_feature_ids.get(transition, [])) for transition in TRANSITION_NAMES},
            "observation": {
                "tested_for_viral_load": list(snapshot.observation_feature_ids["tested_for_viral_load"]) if config.use_observation_covariates else [],
                "virally_suppressed": list(snapshot.observation_feature_ids["virally_suppressed"]) if config.use_observation_covariates else [],
            },
        },
    }


def _build_param_slices(config: CandidateConfig, matrices: dict[str, Any], _snapshot: IntegratedSnapshot) -> dict[str, slice]:
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
        if transition == "U_to_D" and config.diagnosis_family == "delay":
            width += 1
        if transition != "U_to_D" and config.care_family == "semi_markov":
            width += 1
        slices[f"transition::{transition}"] = slice(cursor, cursor + width)
        cursor += width
    for metric_name in ("tested_for_viral_load", "virally_suppressed"):
        width = 1 + matrices["observation_direct"][metric_name].shape[1] + matrices["hidden"].shape[1]
        slices[f"observation::{metric_name}"] = slice(cursor, cursor + width)
        cursor += width
    slices["full"] = slice(0, cursor)
    return slices


def _initial_param_vector(snapshot: IntegratedSnapshot, config: CandidateConfig, matrices: dict[str, Any], param_slices: dict[str, slice]) -> np.ndarray:
    eps = float(np.finfo(np.float64).eps)
    params = np.zeros((param_slices["full"].stop,), dtype=np.float64)

    total_indices = np.where(snapshot.metric_masks[TOTAL_METRIC])[0]
    if total_indices.size == 0:
        raise ValueError("Integrated Phase 3 frontier requires at least one estimated-total observation for state initialization")
    initial_total = float(snapshot.start_total)
    initial_diag = float(snapshot.start_diag)
    initial_undiagnosed = max(initial_total - initial_diag, eps)
    params[param_slices["init_u_log"]] = np.log(initial_undiagnosed)

    art_share_mask = np.logical_and(
        snapshot.metric_masks["alive_on_art"],
        np.logical_and(snapshot.metric_masks["diagnosed_plhiv"], snapshot.metric_values["diagnosed_plhiv"] > 0.0),
    )
    if not np.any(art_share_mask):
        raise ValueError("Integrated Phase 3 frontier requires at least one diagnosed-plus-ART observation for latent ART initialization")
    initial_art_share = float(
        np.clip(
            snapshot.metric_values["alive_on_art"][art_share_mask][0] / max(snapshot.metric_values["diagnosed_plhiv"][art_share_mask][0], eps),
            eps,
            1.0 - eps,
        )
    )
    params[param_slices["init_art_share"]] = np.log(initial_art_share / max(1.0 - initial_art_share, eps))

    initial_vs_mask = np.logical_and(snapshot.metric_masks["virally_suppressed"], snapshot.metric_masks["alive_on_art"])
    if not np.any(initial_vs_mask):
        raise ValueError("Integrated Phase 3 frontier requires at least one virally-suppressed observation for state initialization")
    initial_v_share = float(
        np.clip(
            snapshot.metric_values["virally_suppressed"][initial_vs_mask][0] / max(snapshot.metric_values["alive_on_art"][initial_vs_mask][0], eps),
            eps,
            1.0 - eps,
        )
    )
    params[param_slices["init_v_share"]] = np.log(initial_v_share / max(1.0 - initial_v_share, eps))
    params[param_slices["init_l_share"]] = 0.0

    total_pred = snapshot.metric_values[TOTAL_METRIC][snapshot.metric_masks[TOTAL_METRIC]]
    if total_pred.size >= 2:
        total_deltas = np.diff(total_pred)
        denominator = snapshot.population_denominator[total_indices[1:]]
        incidence_hazard = np.mean(np.maximum(total_deltas, 0.0) / np.maximum(denominator, eps))
    else:
        incidence_hazard = np.mean(
            np.nan_to_num(snapshot.metric_values["new_diagnosed_cases_period"], nan=0.0) / np.maximum(snapshot.population_denominator, eps)
        )
    incidence_hazard = max(float(incidence_hazard), eps)
    params[param_slices["incidence"]][0] = np.log(incidence_hazard)

    newdiag_values = snapshot.metric_values["new_diagnosed_cases_period"][snapshot.metric_masks["new_diagnosed_cases_period"]]
    total_values = snapshot.metric_values[TOTAL_METRIC][snapshot.metric_masks[TOTAL_METRIC]]
    diag_values = snapshot.metric_values["diagnosed_plhiv"][snapshot.metric_masks["diagnosed_plhiv"]]
    art_values = snapshot.metric_values["alive_on_art"][snapshot.metric_masks["alive_on_art"]]
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
    vs_values = snapshot.metric_values["virally_suppressed"][snapshot.metric_masks["virally_suppressed"]]
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
        if transition == "U_to_D" and config.diagnosis_family == "delay":
            bounded = min(max(hazard, eps), 1.0 - eps)
            transition_params[0] = np.log(bounded / max(1.0 - bounded, eps))
        else:
            transition_params[0] = np.log(-np.log(max(1.0 - min(hazard, 1.0 - eps), eps)))
        if transition == "U_to_D" and config.diagnosis_family == "delay":
            transition_params[1] = 0.0
        if transition != "U_to_D" and config.care_family == "semi_markov":
            transition_params[1] = 0.0
        params[transition_slice] = transition_params

    art_nonzero_mask = np.logical_and(snapshot.metric_masks["alive_on_art"], snapshot.metric_values["alive_on_art"] > 0.0)
    test_mask = np.logical_and(snapshot.metric_masks["tested_for_viral_load"], art_nonzero_mask)
    if not np.any(test_mask):
        raise ValueError("Integrated Phase 3 frontier requires at least one VL-testing observation for observation initialization")
    test_share = np.mean(snapshot.metric_values["tested_for_viral_load"][test_mask] / snapshot.metric_values["alive_on_art"][test_mask])
    test_share = float(np.clip(test_share, eps, 1.0 - eps))
    params[param_slices["observation::tested_for_viral_load"]][0] = np.log(test_share / max(1.0 - test_share, eps))

    test_nonzero_mask = np.logical_and(snapshot.metric_masks["tested_for_viral_load"], snapshot.metric_values["tested_for_viral_load"] > 0.0)
    doc_mask = np.logical_and(snapshot.metric_masks["virally_suppressed"], test_nonzero_mask)
    if not np.any(doc_mask):
        raise ValueError("Integrated Phase 3 frontier requires at least one documented-suppression observation for observation initialization")
    doc_share = np.mean(snapshot.metric_values["virally_suppressed"][doc_mask] / snapshot.metric_values["tested_for_viral_load"][doc_mask])
    doc_share = float(np.clip(doc_share, eps, 1.0 - eps))
    params[param_slices["observation::virally_suppressed"]][0] = np.log(doc_share / max(1.0 - doc_share, eps))
    return params


def _module_linear_predictor(
    module_params: np.ndarray,
    direct_matrix: np.ndarray,
    hidden_matrix: np.ndarray,
    *,
    include_age_slope: bool = False,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    intercept = float(module_params[0]) if module_params.size else 0.0
    cursor = 1
    age_slope = float(module_params[cursor]) if include_age_slope and module_params.size > cursor else 0.0
    if include_age_slope:
        cursor += 1
    direct_width = direct_matrix.shape[1]
    hidden_width = hidden_matrix.shape[1]
    direct_beta = np.asarray(module_params[cursor : cursor + direct_width], dtype=np.float64) if direct_width else np.zeros((0,), dtype=np.float64)
    cursor += direct_width
    hidden_beta = np.asarray(module_params[cursor : cursor + hidden_width], dtype=np.float64) if hidden_width else np.zeros((0,), dtype=np.float64)
    direct_component = direct_matrix @ direct_beta if direct_width else np.zeros((direct_matrix.shape[0],), dtype=np.float64)
    hidden_component = hidden_matrix @ hidden_beta if hidden_width else np.zeros((hidden_matrix.shape[0],), dtype=np.float64)
    return np.full((direct_matrix.shape[0],), intercept, dtype=np.float64), age_slope, direct_component, hidden_component


def _age_basis(age_bin_count: int) -> np.ndarray:
    return np.log1p(np.arange(age_bin_count, dtype=np.float64))


def _age_shift(values: np.ndarray) -> np.ndarray:
    shifted = np.zeros_like(values, dtype=np.float64)
    if values.size == 0:
        return shifted
    if values.size > 1:
        shifted[1:] = values[:-1]
    shifted[-1] += float(values[-1])
    return shifted


def _simulate(
    snapshot: IntegratedSnapshot,
    config: CandidateConfig,
    fit_vector: np.ndarray,
    param_slices: dict[str, slice],
    matrices: dict[str, Any],
    levers: dict[str, float] | None = None,
) -> dict[str, Any]:
    eps = float(np.finfo(np.float64).eps)
    levers = levers or {}
    quarter_count = len(snapshot.model_quarters)
    age_bin_count = quarter_count
    age_basis = _age_basis(age_bin_count)
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
    u_age = np.zeros((age_bin_count,), dtype=np.float64)
    d_age = np.zeros((age_bin_count,), dtype=np.float64)
    a_age = np.zeros((age_bin_count,), dtype=np.float64)
    l_age = np.zeros((age_bin_count,), dtype=np.float64)
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
    incidence_intercept, _incidence_age_slope, incidence_direct, incidence_hidden = _module_linear_predictor(
        incidence_params,
        np.asarray(matrices["incidence_direct"], dtype=np.float64),
        np.asarray(matrices["hidden"], dtype=np.float64),
    )
    direct_component_map["incidence"] = incidence_direct
    hidden_component_map["incidence"] = incidence_hidden
    eta_map["incidence"] = incidence_intercept + incidence_direct + incidence_hidden + float(levers.get("incidence", 0.0)) * incidence_direct
    incidence_hazard = np.asarray(_safe_exp(eta_map["incidence"]), dtype=np.float64)
    incidence_flow = snapshot.population_denominator * incidence_hazard

    for transition in TRANSITION_NAMES:
        transition_params = np.asarray(fit_vector[param_slices[f"transition::{transition}"]], dtype=np.float64)
        include_age_slope = (transition == "U_to_D" and config.diagnosis_family == "delay") or (
            transition != "U_to_D" and config.care_family == "semi_markov"
        )
        intercept_component, age_slope, direct_component, hidden_component = _module_linear_predictor(
            transition_params,
            np.asarray(matrices["transition_direct"][transition], dtype=np.float64),
            np.asarray(matrices["hidden"], dtype=np.float64),
            include_age_slope=include_age_slope,
        )
        direct_component_map[transition] = direct_component
        hidden_component_map[transition] = hidden_component
        age_slope_map[transition] = float(age_slope)
        eta_map[transition] = intercept_component + direct_component + hidden_component + float(levers.get(transition, 0.0)) * direct_component
        if not include_age_slope:
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

        if config.diagnosis_family == "delay":
            diagnosis_prob = np.asarray(_sigmoid(eta_map["U_to_D"][idx] + age_slope_map["U_to_D"] * age_basis), dtype=np.float64)
            diagnosed_by_age = np.minimum(u_age, diagnosis_prob * u_age)
        else:
            diagnosed_by_age = np.minimum(u_age, hazard_map["U_to_D"][idx] * u_age)
        u_to_d = float(np.sum(diagnosed_by_age))
        u_age_next = _age_shift(np.maximum(u_age - diagnosed_by_age, 0.0))
        u_age_next[0] += float(incidence_flow[idx])

        d_to_a_hazard = (
            np.asarray(_cloglog_inverse(eta_map["D_to_A"][idx] + age_slope_map["D_to_A"] * age_basis), dtype=np.float64)
            if config.care_family == "semi_markov"
            else np.full((age_bin_count,), float(hazard_map["D_to_A"][idx]), dtype=np.float64)
        )
        d_to_a_by_age = np.minimum(d_age, d_to_a_hazard * d_age)
        d_to_a = float(np.sum(d_to_a_by_age))
        d_age_next = _age_shift(np.maximum(d_age - d_to_a_by_age, 0.0))
        d_age_next[0] += u_to_d

        a_to_v_hazard = (
            np.asarray(_cloglog_inverse(eta_map["A_to_V"][idx] + age_slope_map["A_to_V"] * age_basis), dtype=np.float64)
            if config.care_family == "semi_markov"
            else np.full((age_bin_count,), float(hazard_map["A_to_V"][idx]), dtype=np.float64)
        )
        a_to_l_hazard = (
            np.asarray(_cloglog_inverse(eta_map["A_to_L"][idx] + age_slope_map["A_to_L"] * age_basis), dtype=np.float64)
            if config.care_family == "semi_markov"
            else np.full((age_bin_count,), float(hazard_map["A_to_L"][idx]), dtype=np.float64)
        )
        a_to_v_by_age = np.minimum(a_age, a_to_v_hazard * a_age)
        a_after_v = np.maximum(a_age - a_to_v_by_age, 0.0)
        a_to_l_by_age = np.minimum(a_after_v, a_to_l_hazard * a_age)
        a_survivors = np.maximum(a_after_v - a_to_l_by_age, 0.0)
        a_to_v = float(np.sum(a_to_v_by_age))
        a_to_l = float(np.sum(a_to_l_by_age))

        l_to_a_hazard = (
            np.asarray(_cloglog_inverse(eta_map["L_to_A"][idx] + age_slope_map["L_to_A"] * age_basis), dtype=np.float64)
            if config.care_family == "semi_markov"
            else np.full((age_bin_count,), float(hazard_map["L_to_A"][idx]), dtype=np.float64)
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


def _fit_residuals(snapshot: IntegratedSnapshot, simulation: dict[str, Any], quarter_mask: np.ndarray) -> np.ndarray:
    eps = float(np.finfo(np.float64).eps)
    residuals: list[float] = []
    for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,):
        metric_mask = np.logical_and(snapshot.metric_masks[metric_name], quarter_mask)
        if not np.any(metric_mask):
            continue
        scale = max(float(snapshot.metric_scales[metric_name]), eps)
        predicted = np.asarray(simulation["predictions"][metric_name], dtype=np.float64)[metric_mask]
        observed = np.asarray(snapshot.metric_values[metric_name], dtype=np.float64)[metric_mask]
        residuals.extend(((predicted - observed) / scale).tolist())
    return np.asarray(residuals, dtype=np.float64)


def _historical_mask(snapshot: IntegratedSnapshot) -> np.ndarray:
    return np.asarray(
        [quarter_sort_key(quarter) <= quarter_sort_key(snapshot.historical_end_quarter) for quarter in snapshot.model_quarters],
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
        raise ValueError("Integrated Phase 3 blocked-time contract requires at least three observed diagnosis-flow quarters")
    split_blocks = [list(block.astype(str)) for block in np.array_split(np.asarray(diagnosis_quarters, dtype=object), 3)]
    if any(len(block) == 0 for block in split_blocks):
        raise ValueError("Integrated Phase 3 blocked-time split produced an empty contiguous block")
    train_diag_quarters, validation_quarters, holdout_quarters = split_blocks
    historical_mask = _historical_mask(snapshot)
    train_end_quarter = str(train_diag_quarters[-1])
    validation_start_quarter = str(validation_quarters[0])
    validation_end_quarter = str(validation_quarters[-1])
    holdout_start_quarter = str(holdout_quarters[0])
    holdout_end_quarter = str(holdout_quarters[-1])
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


def _blocked_report_masks(snapshot: IntegratedSnapshot, contract: BlockedTimeContract) -> dict[str, np.ndarray]:
    return {
        "train": np.asarray(contract.train_mask, dtype=bool),
        "validation": np.asarray(contract.validation_mask, dtype=bool),
        "holdout": np.asarray(contract.holdout_mask, dtype=bool),
        "historical": _historical_mask(snapshot),
    }


def _identifiability_contract(snapshot: IntegratedSnapshot, contract: BlockedTimeContract) -> dict[str, Any]:
    coverage = _metric_observation_coverage(snapshot)
    return {
        "stage1_selection_window": {
            "selection_window": "validation",
            "report_only_window": "holdout",
            "fit_window_end_quarter": contract.train_end_quarter,
            "validation_start_quarter": contract.validation_start_quarter,
            "validation_end_quarter": contract.validation_end_quarter,
            "holdout_start_quarter": contract.holdout_start_quarter,
            "holdout_end_quarter": contract.holdout_end_quarter,
        },
        "metric_identifiability": {
            "new_diagnosed_cases_period": {
                "role": "primary_front_half_identification_signal",
                "coverage": dict(coverage.get("new_diagnosed_cases_period") or {}),
            },
            "tested_for_viral_load": {
                "role": "secondary_service_ascertainment_signal",
                "coverage": dict(coverage.get("tested_for_viral_load") or {}),
            },
            "virally_suppressed": {
                "role": "secondary_documented_suppression_signal",
                "coverage": dict(coverage.get("virally_suppressed") or {}),
            },
        },
        "claim_boundary": {
            "third_95_supported_as_primary_fit_target": False,
            "stage2_95_95_95_role": "exploratory_bounded_lever_search_only",
            "publication_note": "The back half of the cascade remains weakly identified relative to diagnosis and ART. Stage 2 target search is exploratory and is not an acceptance criterion.",
        },
    }


def _year_mask(snapshot: IntegratedSnapshot, year: int) -> np.ndarray:
    return np.asarray([_quarter_year(quarter) == int(year) for quarter in snapshot.model_quarters], dtype=bool)


def _metric_observation_flags(snapshot: IntegratedSnapshot, quarter_mask: np.ndarray) -> dict[str, bool]:
    metric_names = PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,)
    return {
        metric_name: bool(np.any(np.logical_and(snapshot.metric_masks[metric_name], quarter_mask)))
        for metric_name in metric_names
    }


def _serialize_year_metrics(snapshot: IntegratedSnapshot, year_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    serialized: dict[str, dict[str, Any]] = {}
    for year, metrics in year_metrics.items():
        quarter_mask = _year_mask(snapshot, int(year))
        observation_flags = _metric_observation_flags(snapshot, quarter_mask)
        diagnosis_flow_observed = bool(observation_flags["new_diagnosed_cases_period"])
        year_payload: dict[str, Any] = {}
        for key, value in metrics.items():
            metric_key = str(key)
            if metric_key in {"diag_flow_loss", "metric::new_diagnosed_cases_period"} and not diagnosis_flow_observed:
                year_payload[metric_key] = None
                continue
            value_float = float(value)
            year_payload[metric_key] = value_float if np.isfinite(value_float) else None
        year_payload["diagnosis_flow_observed"] = diagnosis_flow_observed
        year_payload["diagnosis_flow_status"] = "observed" if diagnosis_flow_observed else "not observed"
        year_payload["observation_flags"] = {metric_name: bool(observed) for metric_name, observed in observation_flags.items()}
        year_payload["observed_metric_count"] = int(sum(bool(observed) for observed in observation_flags.values()))
        serialized[str(year)] = year_payload
    return serialized


def _metric_observation_coverage(snapshot: IntegratedSnapshot) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,):
        observed_quarters = [
            quarter
            for quarter in snapshot.model_quarters
            if bool(snapshot.metric_masks[metric_name][snapshot.quarter_index[quarter]])
        ]
        coverage[metric_name] = {
            "observed_quarter_count": int(len(observed_quarters)),
            "first_observed_quarter": observed_quarters[0] if observed_quarters else None,
            "last_observed_quarter": observed_quarters[-1] if observed_quarters else None,
        }
    return coverage


def _compute_window_metrics(snapshot: IntegratedSnapshot, simulation: dict[str, Any], quarter_mask: np.ndarray) -> dict[str, float]:
    eps = float(np.finfo(np.float64).eps)
    metric_mae: dict[str, float] = {}
    for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,):
        metric_mask = np.logical_and(snapshot.metric_masks[metric_name], quarter_mask)
        if not np.any(metric_mask):
            continue
        scale = max(float(snapshot.metric_scales[metric_name]), eps)
        predicted = np.asarray(simulation["predictions"][metric_name], dtype=np.float64)[metric_mask]
        observed = np.asarray(snapshot.metric_values[metric_name], dtype=np.float64)[metric_mask]
        metric_mae[metric_name] = float(np.mean(_smooth_abs((predicted - observed) / scale, eps)))
    primary_values = [metric_mae[name] for name in PRIMARY_METRICS if name in metric_mae]
    secondary_values = [metric_mae[name] for name in SECONDARY_METRICS if name in metric_mae]
    metrics = {
        "primary_loss": float(np.mean(primary_values)) if primary_values else float("inf"),
        "diag_flow_loss": float(metric_mae.get("new_diagnosed_cases_period", float("inf"))),
        "secondary_loss": float(np.mean(secondary_values)) if secondary_values else 0.0,
        "total_loss": float(metric_mae.get(TOTAL_METRIC, 0.0)),
        "population_violation_count": float(simulation["plausibility"]["population_violation_count"]),
        "non_finite_count": float(simulation["plausibility"]["non_finite_count"]),
    }
    metrics.update({f"metric::{metric_name}": float(value) for metric_name, value in metric_mae.items()})
    return metrics


def _fit_with_matrices(
    snapshot: IntegratedSnapshot,
    config: CandidateConfig,
    matrices: dict[str, Any],
    *,
    fit_mask: np.ndarray | None = None,
    report_masks: dict[str, np.ndarray] | None = None,
) -> FitResult:
    param_slices = _build_param_slices(config, matrices, snapshot)
    x0 = _initial_param_vector(snapshot, config, matrices, param_slices)
    historical_mask = _historical_mask(snapshot)
    fit_window_mask = np.asarray(fit_mask if fit_mask is not None else historical_mask, dtype=bool)
    report_window_masks = {
        str(window_name): np.asarray(window_mask, dtype=bool)
        for window_name, window_mask in dict(report_masks or {"historical": historical_mask}).items()
    }
    report_window_masks.setdefault("historical", historical_mask)
    residual_budget = sum(
        int(np.sum(np.logical_and(snapshot.metric_masks[metric_name], fit_window_mask)))
        for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,)
    )
    max_nfev = int(param_slices["full"].stop + residual_budget + int(np.sum(fit_window_mask)))

    def residual_fn(params: np.ndarray) -> np.ndarray:
        simulation = _simulate(snapshot, config, params, param_slices, matrices)
        residuals = _fit_residuals(snapshot, simulation, fit_window_mask)
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

    result = optimize.least_squares(residual_fn, x0=x0, method="trf", max_nfev=max_nfev)
    fitted_vector = np.asarray(result.x, dtype=np.float64)
    simulation = _simulate(snapshot, config, fitted_vector, param_slices, matrices)
    window_metrics = {
        window_name: _compute_window_metrics(snapshot, simulation, window_mask)
        for window_name, window_mask in report_window_masks.items()
    }
    year_metrics = {
        str(year): _compute_window_metrics(snapshot, simulation, _year_mask(snapshot, year))
        for year in snapshot.historical_years
        if np.any(
            np.logical_and(
                _year_mask(snapshot, year),
                np.logical_or.reduce([snapshot.metric_masks[metric_name] for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,)]),
            )
        )
    }
    feature_summary = {
        "diagnosis_family": config.diagnosis_family,
        "care_family": config.care_family,
        "hidden_rank": config.hidden_rank,
        "use_observation_covariates": config.use_observation_covariates,
        "incidence_feature_ids": list(matrices["feature_ids"]["incidence"]),
        "transition_feature_ids": {transition: list(value) for transition, value in dict(matrices["feature_ids"]["transition"]).items()},
        "observation_feature_ids": dict(matrices["feature_ids"]["observation"]),
    }
    return FitResult(
        config=config,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        train_cost=float(result.cost),
        param_vector=fitted_vector,
        param_slices=param_slices,
        matrices=matrices,
        simulation=simulation,
        window_metrics=window_metrics,
        year_metrics=year_metrics,
        feature_summary=feature_summary,
    )


def _fit_candidate(
    snapshot: IntegratedSnapshot,
    config: CandidateConfig,
    *,
    fit_mask: np.ndarray | None = None,
    report_masks: dict[str, np.ndarray] | None = None,
) -> FitResult:
    matrices = _candidate_matrices(snapshot, config)
    return _fit_with_matrices(snapshot, config, matrices, fit_mask=fit_mask, report_masks=report_masks)


def _constant_baseline_fit(
    snapshot: IntegratedSnapshot,
    *,
    fit_mask: np.ndarray | None = None,
    report_masks: dict[str, np.ndarray] | None = None,
) -> FitResult:
    zero_matrix = np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64)
    config = CandidateConfig(
        candidate_id="simple-compartmental-constant-hazard",
        diagnosis_family="hazard",
        care_family="markov",
        hidden_rank=0,
        use_observation_covariates=False,
    )
    matrices = {
        "incidence_direct": zero_matrix,
        "transition_direct": {transition: zero_matrix for transition in TRANSITION_NAMES},
        "observation_direct": {
            "tested_for_viral_load": zero_matrix,
            "virally_suppressed": zero_matrix,
        },
        "hidden": zero_matrix,
        "feature_ids": {
            "incidence": [],
            "transition": {transition: [] for transition in TRANSITION_NAMES},
            "observation": {
                "tested_for_viral_load": [],
                "virally_suppressed": [],
            },
        },
    }
    return _fit_with_matrices(snapshot, config, matrices, fit_mask=fit_mask, report_masks=report_masks)


def _carry_forward_prediction_array(snapshot: IntegratedSnapshot, metric_name: str) -> np.ndarray:
    prediction = np.full((len(snapshot.model_quarters),), np.nan, dtype=np.float64)
    observed = np.asarray(snapshot.metric_values[metric_name], dtype=np.float64)
    observed_mask = np.asarray(snapshot.metric_masks[metric_name], dtype=bool)
    last_value = float("nan")
    for idx in range(len(snapshot.model_quarters)):
        if idx == 0 and observed_mask[idx]:
            prediction[idx] = float(observed[idx])
            last_value = float(observed[idx])
            continue
        if np.isfinite(last_value):
            prediction[idx] = float(last_value)
        elif observed_mask[idx]:
            prediction[idx] = float(observed[idx])
        if observed_mask[idx]:
            last_value = float(observed[idx])
    return prediction


def _carry_forward_simulation(snapshot: IntegratedSnapshot) -> dict[str, Any]:
    predictions = {
        metric_name: _carry_forward_prediction_array(snapshot, metric_name)
        for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,)
    }
    return {
        "predictions": predictions,
        "plausibility": {"population_violation_count": 0, "non_finite_count": 0},
    }


def _carry_forward_blocked_time_simulation(snapshot: IntegratedSnapshot, contract: BlockedTimeContract) -> dict[str, Any]:
    predictions: dict[str, np.ndarray] = {}
    for metric_name in PRIMARY_METRICS + SECONDARY_METRICS + (TOTAL_METRIC,):
        prediction = np.full((len(snapshot.model_quarters),), np.nan, dtype=np.float64)
        values = np.asarray(snapshot.metric_values[metric_name], dtype=np.float64)
        observed_indices = np.where(np.logical_and(snapshot.metric_masks[metric_name], contract.train_mask))[0]
        if observed_indices.size > 0:
            last_index = int(observed_indices[-1])
            last_value = float(values[last_index])
            prediction[contract.train_mask] = values[contract.train_mask]
            prediction[np.logical_or(contract.validation_mask, contract.holdout_mask)] = last_value
        predictions[metric_name] = prediction
    return {"predictions": predictions, "plausibility": {"population_violation_count": 0, "non_finite_count": 0}}


def _baseline_comparison(
    snapshot: IntegratedSnapshot,
    champion: FitResult,
    *,
    fit_mask: np.ndarray | None = None,
    report_masks: dict[str, np.ndarray] | None = None,
    blocked_contract: BlockedTimeContract | None = None,
) -> dict[str, Any]:
    historical_mask = _historical_mask(snapshot)
    resolved_report_masks = {
        str(window_name): np.asarray(window_mask, dtype=bool)
        for window_name, window_mask in dict(report_masks or {"historical": historical_mask}).items()
    }
    resolved_report_masks.setdefault("historical", historical_mask)
    carry_forward_simulation = (
        _carry_forward_blocked_time_simulation(snapshot, blocked_contract)
        if blocked_contract is not None
        else _carry_forward_simulation(snapshot)
    )
    carry_forward_metrics = {
        window_name: _compute_window_metrics(snapshot, carry_forward_simulation, window_mask)
        for window_name, window_mask in resolved_report_masks.items()
    }
    simple_baseline = _constant_baseline_fit(snapshot, fit_mask=fit_mask, report_masks=resolved_report_masks)
    simple_metrics_by_window = dict(simple_baseline.window_metrics)
    model_metrics = champion.window_metrics["historical"]
    carry_historical_metrics = carry_forward_metrics["historical"]
    simple_metrics = simple_metrics_by_window["historical"]
    validation_metrics = champion.window_metrics.get("validation")
    holdout_metrics = champion.window_metrics.get("holdout")
    split_comparison: dict[str, Any] = {}
    for window_name in ("validation", "holdout"):
        if window_name not in resolved_report_masks or window_name not in champion.window_metrics:
            continue
        model_window = champion.window_metrics[window_name]
        carry_window = carry_forward_metrics[window_name]
        simple_window = simple_metrics_by_window[window_name]
        split_comparison[window_name] = {
            "model_primary_loss": round(float(model_window["primary_loss"]), 6),
            "model_diag_flow_loss": round(float(model_window["diag_flow_loss"]), 6),
            "carry_forward_primary_loss": round(float(carry_window["primary_loss"]), 6),
            "carry_forward_diag_flow_loss": round(float(carry_window["diag_flow_loss"]), 6),
            "simple_compartmental_primary_loss": round(float(simple_window["primary_loss"]), 6),
            "simple_compartmental_diag_flow_loss": round(float(simple_window["diag_flow_loss"]), 6),
            "model_beats_carry_forward": bool(float(model_window["primary_loss"]) < float(carry_window["primary_loss"])),
            "model_beats_simple_compartmental": bool(float(model_window["primary_loss"]) < float(simple_window["primary_loss"])),
            "diagnosis_flow_beats_carry_forward": bool(float(model_window["diag_flow_loss"]) < float(carry_window["diag_flow_loss"])),
            "diagnosis_flow_beats_simple_compartmental": bool(float(model_window["diag_flow_loss"]) < float(simple_window["diag_flow_loss"])),
        }
    return {
        "comparison_window_start_quarter": snapshot.historical_quarters[0],
        "comparison_window_end_quarter": snapshot.historical_end_quarter,
        "comparison_years": list(snapshot.historical_years),
        "metric_definition": "normalized historical loss on diagnosed_plhiv, alive_on_art, and new_diagnosed_cases_period over the active national window",
        "model_primary_loss": round(float(model_metrics["primary_loss"]), 6),
        "model_diag_flow_loss": round(float(model_metrics["diag_flow_loss"]), 6),
        "carry_forward_primary_loss": round(float(carry_historical_metrics["primary_loss"]), 6),
        "carry_forward_diag_flow_loss": round(float(carry_historical_metrics["diag_flow_loss"]), 6),
        "simple_compartmental_primary_loss": round(float(simple_metrics["primary_loss"]), 6),
        "simple_compartmental_diag_flow_loss": round(float(simple_metrics["diag_flow_loss"]), 6),
        "simple_compartmental_optimizer_success": bool(simple_baseline.success),
        "model_beats_carry_forward": float(model_metrics["primary_loss"]) < float(carry_historical_metrics["primary_loss"]),
        "model_beats_simple_compartmental": float(model_metrics["primary_loss"]) < float(simple_metrics["primary_loss"]),
        "diagnosis_flow_beats_carry_forward": float(model_metrics["diag_flow_loss"]) < float(carry_historical_metrics["diag_flow_loss"]),
        "diagnosis_flow_beats_simple_compartmental": float(model_metrics["diag_flow_loss"]) < float(simple_metrics["diag_flow_loss"]),
        "split_comparison": split_comparison,
        "selection_window": "validation" if validation_metrics is not None else "historical",
        "validation_score_tuple": list(_score_tuple(validation_metrics)) if validation_metrics is not None else None,
        "holdout_score_tuple": list(_score_tuple(holdout_metrics)) if holdout_metrics is not None else None,
    }


def _score_tuple(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(metrics["primary_loss"]),
        float(metrics["diag_flow_loss"]),
        float(metrics["total_loss"]),
        float(metrics["secondary_loss"]),
    )


def _fit_is_hard_valid(fit: FitResult) -> bool:
    if not fit.success:
        return False
    for window_name in ("validation", "holdout", "historical"):
        metrics = fit.window_metrics.get(window_name)
        if metrics is None:
            continue
        if float(metrics.get("non_finite_count", 0.0)) > 0.0:
            return False
        if float(metrics.get("population_violation_count", 0.0)) > 0.0:
            return False
    return True


def _promotable(candidate: FitResult, champion: FitResult | None, *, selection_window: str = "validation") -> bool:
    if not _fit_is_hard_valid(candidate):
        return False
    if champion is None:
        return True
    candidate_window = dict(candidate.window_metrics.get(selection_window) or candidate.window_metrics["historical"])
    champion_window = dict(champion.window_metrics.get(selection_window) or champion.window_metrics["historical"])
    return _score_tuple(candidate_window) < _score_tuple(champion_window)


def _state_rows(snapshot: IntegratedSnapshot, simulation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, quarter in enumerate(snapshot.model_quarters):
        state_values = {state_name: round(float(simulation["states"][idx, state_idx]), 6) for state_idx, state_name in enumerate(STATE_NAMES)}
        rows.append(
            {
                "quarter": quarter,
                "state_values": state_values,
                "estimated_plhiv": round(float(simulation["predictions"]["estimated_plhiv"][idx]), 6),
                "diagnosed_plhiv": round(float(simulation["predictions"]["diagnosed_plhiv"][idx]), 6),
                "alive_on_art": round(float(simulation["predictions"]["alive_on_art"][idx]), 6),
                "new_diagnosed_cases_period": round(float(simulation["predictions"]["new_diagnosed_cases_period"][idx]), 6),
                "tested_for_viral_load": round(float(simulation["predictions"]["tested_for_viral_load"][idx]), 6),
                "virally_suppressed": round(float(simulation["predictions"]["virally_suppressed"][idx]), 6),
                "incidence_flow": round(float(simulation["incidence_flow"][idx]), 6),
            }
        )
    return rows


def _transition_hazard_rows(snapshot: IntegratedSnapshot, simulation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, quarter in enumerate(snapshot.model_quarters):
        rows.append(
            {
                "quarter": quarter,
                "hazards": {transition: round(float(simulation["hazards"][transition][idx]), 6) for transition in TRANSITION_NAMES},
                "flows": {transition: round(float(simulation["flows"][transition][idx]), 6) for transition in TRANSITION_NAMES},
                "linear_predictors": {
                    transition: round(float(simulation["eta"][transition][idx]), 6) for transition in ("incidence",) + TRANSITION_NAMES
                },
            }
        )
    return rows


def _observation_process_summary(snapshot: IntegratedSnapshot, fit: FitResult) -> dict[str, Any]:
    observation_rows = []
    for idx, quarter in enumerate(snapshot.model_quarters):
        observation_rows.append(
            {
                "quarter": quarter,
                "pi_test": round(float(fit.simulation["pi_test"][idx]), 6),
                "pi_doc": round(float(fit.simulation["pi_doc"][idx]), 6),
                "predicted_tested_for_viral_load": round(float(fit.simulation["predictions"]["tested_for_viral_load"][idx]), 6),
                "predicted_virally_suppressed": round(float(fit.simulation["predictions"]["virally_suppressed"][idx]), 6),
            }
        )
    return {
        "config": {
            "candidate_id": fit.config.candidate_id,
            "diagnosis_family": fit.config.diagnosis_family,
            "care_family": fit.config.care_family,
            "hidden_rank": fit.config.hidden_rank,
            "use_observation_covariates": fit.config.use_observation_covariates,
        },
        "feature_summary": fit.feature_summary,
        "rows": observation_rows,
    }


def _stage1_fit_chart(snapshot: IntegratedSnapshot, fit: FitResult, path: Path) -> None:
    x_labels = list(snapshot.historical_quarters)
    index_lookup = [snapshot.quarter_index[quarter] for quarter in snapshot.historical_quarters]
    series = [
        ("diagnosed target", [float(snapshot.metric_values["diagnosed_plhiv"][idx]) for idx in index_lookup]),
        ("diagnosed prediction", [float(fit.simulation["predictions"]["diagnosed_plhiv"][idx]) for idx in index_lookup]),
        ("ART target", [float(snapshot.metric_values["alive_on_art"][idx]) for idx in index_lookup]),
        ("ART prediction", [float(fit.simulation["predictions"]["alive_on_art"][idx]) for idx in index_lookup]),
        ("new diagnoses target", [float(snapshot.metric_values["new_diagnosed_cases_period"][idx]) for idx in index_lookup]),
        ("new diagnoses prediction", [float(fit.simulation["predictions"]["new_diagnosed_cases_period"][idx]) for idx in index_lookup]),
    ]
    _write_line_chart(path, "Stage 1 Full-Window Fit Vs Observed", x_labels, series, "count")


def _stage1_frontier_chart(frontier_rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["candidate_id"]) for row in frontier_rows]
    candidate_primary = [float(row.get("validation_primary_loss") or row["historical_primary_loss"]) for row in frontier_rows]
    champion_primary = [
        float(row.get("best_validation_primary_loss") or row.get("best_historical_primary_loss") or row["historical_primary_loss"])
        for row in frontier_rows
    ]
    _write_line_chart(
        path,
        "Stage 1 Blocked-Time Calibration Frontier",
        labels,
        [("candidate validation primary", candidate_primary), ("best-so-far validation primary", champion_primary)],
        "normalized MAE",
    )


def _stage1_yearly_chart(fit: FitResult, path: Path) -> None:
    years = list(fit.year_metrics)
    _write_line_chart(
        path,
        "Stage 1 Yearly Historical Loss",
        years,
        [
            ("primary loss", [float(fit.year_metrics[year]["primary_loss"]) for year in years]),
            (
                "diagnosis-flow loss",
                [
                    float(value) if np.isfinite(value := float(fit.year_metrics[year]["diag_flow_loss"])) else float("nan")
                    for year in years
                ],
            ),
        ],
        "normalized MAE",
    )


def _baseline_comparison_chart(comparison: dict[str, Any], path: Path) -> None:
    split_comparison = dict(comparison.get("split_comparison") or {})
    if "validation" in split_comparison and "holdout" in split_comparison:
        labels = ["model", "carry_forward", "simple_compartmental"]
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
        for axis, window_name in zip(axes, ("validation", "holdout"), strict=False):
            window = dict(split_comparison.get(window_name) or {})
            axis.plot(
                labels,
                [
                    float(window["model_primary_loss"]),
                    float(window["carry_forward_primary_loss"]),
                    float(window["simple_compartmental_primary_loss"]),
                ],
                marker="o",
                linewidth=1.5,
                label=f"{window_name} primary loss",
            )
            axis.plot(
                labels,
                [
                    float(window["model_diag_flow_loss"]),
                    float(window["carry_forward_diag_flow_loss"]),
                    float(window["simple_compartmental_diag_flow_loss"]),
                ],
                marker="o",
                linewidth=1.5,
                label=f"{window_name} diagnosis-flow loss",
            )
            axis.set_title(f"Blocked-Time Benchmark Comparison ({window_name.title()})")
            axis.set_ylabel("normalized MAE")
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return
    labels = ["model", "carry_forward", "simple_compartmental"]
    _write_line_chart(
        path,
        "Historical Benchmark Comparison",
        labels,
        [
            (
                "primary loss",
                [
                    float(comparison["model_primary_loss"]),
                    float(comparison["carry_forward_primary_loss"]),
                    float(comparison["simple_compartmental_primary_loss"]),
                ],
            ),
            (
                "diagnosis-flow loss",
                [
                    float(comparison["model_diag_flow_loss"]),
                    float(comparison["carry_forward_diag_flow_loss"]),
                    float(comparison["simple_compartmental_diag_flow_loss"]),
                ],
            ),
        ],
        "normalized MAE",
    )


def _cascade_rows(snapshot: IntegratedSnapshot, simulation: dict[str, Any]) -> list[dict[str, Any]]:
    eps = float(np.finfo(np.float64).eps)
    rows: list[dict[str, Any]] = []
    for idx, quarter in enumerate(snapshot.model_quarters):
        diagnosed = float(simulation["predictions"]["diagnosed_plhiv"][idx])
        art = float(simulation["predictions"]["alive_on_art"][idx])
        total = float(simulation["predictions"]["estimated_plhiv"][idx])
        suppressed = float(simulation["states"][idx, STATE_NAMES.index("V")])
        rows.append(
            {
                "quarter": quarter,
                "cascade_1": diagnosed / max(total, eps),
                "cascade_2": art / max(diagnosed, eps),
                "cascade_3": suppressed / max(art, eps),
            }
        )
    return rows


def _lever_bounds(fit: FitResult, snapshot: IntegratedSnapshot) -> tuple[list[str], np.ndarray]:
    observed_mask = _historical_mask(snapshot)
    lever_names: list[str] = []
    bound_values: list[float] = []
    for module_name in ("incidence",) + TRANSITION_NAMES:
        direct_component = np.asarray(fit.simulation["direct_components"].get(module_name, np.zeros((len(snapshot.model_quarters),), dtype=np.float64)), dtype=np.float64)
        eta_values = np.asarray(fit.simulation["eta"].get(module_name, np.zeros((len(snapshot.model_quarters),), dtype=np.float64)), dtype=np.float64)
        direct_std = np.std(direct_component[observed_mask])
        if direct_std <= np.finfo(np.float64).eps:
            continue
        eta_std = _safe_std(eta_values[observed_mask], np.finfo(np.float64).eps)
        lever_names.append(module_name)
        bound_values.append(float(eta_std / direct_std))
    return lever_names, np.asarray(bound_values, dtype=np.float64)


def _target_objective_rows(snapshot: IntegratedSnapshot, simulation: dict[str, Any]) -> dict[str, Any]:
    cascade_rows = _cascade_rows(snapshot, simulation)
    horizon_row = next(row for row in cascade_rows if row["quarter"] == snapshot.target_horizon)
    shortfall = sum(max(0.0, CASCADE_TARGET - float(horizon_row[f"cascade_{index}"])) ** 2 for index in (1, 2, 3))
    return {
        "cascade_rows": cascade_rows,
        "horizon_quarter": snapshot.target_horizon,
        "horizon_shortfall": float(shortfall),
        "horizon_cascade": {
            "cascade_1": float(horizon_row["cascade_1"]),
            "cascade_2": float(horizon_row["cascade_2"]),
            "cascade_3": float(horizon_row["cascade_3"]),
        },
    }


def _stage2_frontier_chart(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["step"]) for row in rows]
    shortfall = [float(row["shortfall"]) for row in rows]
    best = [float(row["best_shortfall"]) for row in rows]
    _write_line_chart(path, "Stage 2 Target Frontier", labels, [("candidate shortfall", shortfall), ("best-so-far shortfall", best)], "shortfall")


def _stage2_projection_chart(snapshot: IntegratedSnapshot, baseline_rows: list[dict[str, Any]], optimized_rows: list[dict[str, Any]], path: Path) -> None:
    target_quarters = set(snapshot.historical_quarters + snapshot.future_quarters)
    quarters = [row["quarter"] for row in baseline_rows if row["quarter"] in target_quarters]
    baseline_lookup = {row["quarter"]: row for row in baseline_rows}
    optimized_lookup = {row["quarter"]: row for row in optimized_rows}
    series = [
        ("baseline C1", [float(baseline_lookup[quarter]["cascade_1"]) for quarter in quarters]),
        ("optimized C1", [float(optimized_lookup[quarter]["cascade_1"]) for quarter in quarters]),
        ("baseline C2", [float(baseline_lookup[quarter]["cascade_2"]) for quarter in quarters]),
        ("optimized C2", [float(optimized_lookup[quarter]["cascade_2"]) for quarter in quarters]),
        ("baseline C3", [float(baseline_lookup[quarter]["cascade_3"]) for quarter in quarters]),
        ("optimized C3", [float(optimized_lookup[quarter]["cascade_3"]) for quarter in quarters]),
    ]
    _write_line_chart(path, "Stage 2 Cascade Projection", quarters, series, "ratio")


def _stage2_target_search(snapshot: IntegratedSnapshot, champion: FitResult) -> dict[str, Any]:
    lever_names, bound_values = _lever_bounds(champion, snapshot)
    if not lever_names:
        baseline_target = _target_objective_rows(snapshot, champion.simulation)
        row = {
            "step": 0,
            "label": "baseline",
            "shortfall": float(baseline_target["horizon_shortfall"]),
            "best_shortfall": float(baseline_target["horizon_shortfall"]),
            "historical_primary_loss": float(champion.window_metrics["historical"]["primary_loss"]),
            "feasible": True,
            "lever_values": {},
        }
        return {
            "optimized_simulation": champion.simulation,
            "frontier_rows": [row],
            "baseline_target": baseline_target,
            "optimized_target": baseline_target,
            "lever_values": {},
            "optimization_success": False,
            "optimization_message": "no non-zero direct Phase 2 surfaces were available for target search",
        }

    baseline_target = _target_objective_rows(snapshot, champion.simulation)
    baseline_historical = champion.window_metrics["historical"]
    frontier_rows: list[dict[str, Any]] = []

    def _lever_map(vector: np.ndarray) -> dict[str, float]:
        return {lever_names[idx]: float(vector[idx]) for idx in range(len(lever_names))}

    def _evaluate(vector: np.ndarray) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
        simulation = _simulate(snapshot, champion.config, champion.param_vector, champion.param_slices, champion.matrices, _lever_map(vector))
        historical_metrics = _compute_window_metrics(snapshot, simulation, _historical_mask(snapshot))
        target_metrics = _target_objective_rows(snapshot, simulation)
        return simulation, historical_metrics, target_metrics

    best_shortfall = float(baseline_target["horizon_shortfall"])
    best_vector = np.zeros((len(lever_names),), dtype=np.float64)
    best_simulation = champion.simulation
    best_target = baseline_target

    def _record(step: int, label: str, vector: np.ndarray, historical_metrics: dict[str, float], target_metrics: dict[str, Any], feasible: bool) -> None:
        nonlocal best_shortfall, best_vector, best_simulation, best_target
        if feasible and float(target_metrics["horizon_shortfall"]) < best_shortfall:
            best_shortfall = float(target_metrics["horizon_shortfall"])
            best_vector = np.asarray(vector, dtype=np.float64)
            best_simulation = _simulate(snapshot, champion.config, champion.param_vector, champion.param_slices, champion.matrices, _lever_map(best_vector))
            best_target = target_metrics
        frontier_rows.append(
            {
                "step": int(step),
                "label": label,
                "shortfall": float(target_metrics["horizon_shortfall"]),
                "best_shortfall": float(best_shortfall),
                "historical_primary_loss": float(historical_metrics["primary_loss"]),
                "feasible": bool(feasible),
                "lever_values": _lever_map(vector),
            }
        )

    zero_vector = np.zeros((len(lever_names),), dtype=np.float64)
    _record(0, "baseline", zero_vector, baseline_historical, baseline_target, True)

    def objective(vector: np.ndarray) -> float:
        _simulation, historical_metrics, target_metrics = _evaluate(vector)
        violation = sum(
            max(0.0, candidate - baseline)
            for candidate, baseline in (
                (float(historical_metrics["primary_loss"]), float(baseline_historical["primary_loss"])),
                (float(historical_metrics["diag_flow_loss"]), float(baseline_historical["diag_flow_loss"])),
            )
        )
        return float(target_metrics["horizon_shortfall"] + violation)

    constraints = [
        {"type": "ineq", "fun": lambda x: float(baseline_historical["primary_loss"] - _evaluate(x)[1]["primary_loss"])},
        {"type": "ineq", "fun": lambda x: float(baseline_historical["diag_flow_loss"] - _evaluate(x)[1]["diag_flow_loss"])},
    ]

    callback_counter = {"value": 1}

    def callback(vector: np.ndarray) -> None:
        simulation, historical_metrics, target_metrics = _evaluate(vector)
        feasible = (
            float(historical_metrics["primary_loss"]) <= float(baseline_historical["primary_loss"])
            and float(historical_metrics["diag_flow_loss"]) <= float(baseline_historical["diag_flow_loss"])
        )
        _record(callback_counter["value"], f"iter-{callback_counter['value']}", vector, historical_metrics, target_metrics, feasible)
        callback_counter["value"] += 1

    bounds = optimize.Bounds(lb=-bound_values, ub=bound_values)
    maxiter = int(len(lever_names) + len(snapshot.historical_quarters) + len(snapshot.future_quarters))
    optimization = optimize.minimize(
        objective,
        x0=zero_vector,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={"maxiter": maxiter},
    )
    final_vector = np.asarray(optimization.x, dtype=np.float64)
    _final_simulation, final_historical, final_target = _evaluate(final_vector)
    final_feasible = (
        float(final_historical["primary_loss"]) <= float(baseline_historical["primary_loss"])
        and float(final_historical["diag_flow_loss"]) <= float(baseline_historical["diag_flow_loss"])
    )
    _record(callback_counter["value"], "final", final_vector, final_historical, final_target, final_feasible)
    return {
        "optimized_simulation": best_simulation,
        "frontier_rows": frontier_rows,
        "baseline_target": baseline_target,
        "optimized_target": best_target,
        "lever_values": _lever_map(best_vector),
        "optimization_success": bool(optimization.success),
        "optimization_message": str(optimization.message),
    }


def run_phase3_v2_int(ctx: TransitionResearchContext) -> dict[str, Any]:
    snapshot = _build_snapshot(ctx)
    contract = _build_blocked_time_contract(snapshot)
    report_masks = _blocked_report_masks(snapshot, contract)
    base_candidate_configs = _build_candidate_configs(snapshot)
    frontier_rows: list[dict[str, Any]] = []
    challenger_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []
    champion: FitResult | None = None
    best_validation_primary = float("inf")
    epoch_index = 0

    def _evaluate_candidate(config: CandidateConfig) -> None:
        nonlocal champion, best_validation_primary, epoch_index
        epoch_index += 1
        fit = _fit_candidate(snapshot, config, fit_mask=contract.train_mask, report_masks=report_masks)
        train_metrics = fit.window_metrics["train"]
        validation_metrics = fit.window_metrics["validation"]
        holdout_metrics = fit.window_metrics["holdout"]
        historical_metrics = fit.window_metrics["historical"]
        serialized_year_metrics = _serialize_year_metrics(snapshot, fit.year_metrics)
        promoted = _promotable(fit, champion, selection_window="validation")
        if promoted:
            champion = fit
            best_validation_primary = float(validation_metrics["primary_loss"])
        frontier_rows.append(
            {
                "step": int(epoch_index),
                "candidate_id": config.candidate_id,
                "train_primary_loss": float(train_metrics["primary_loss"]),
                "train_diag_flow_loss": float(train_metrics["diag_flow_loss"]),
                "validation_primary_loss": float(validation_metrics["primary_loss"]),
                "validation_diag_flow_loss": float(validation_metrics["diag_flow_loss"]),
                "holdout_primary_loss": float(holdout_metrics["primary_loss"]),
                "holdout_diag_flow_loss": float(holdout_metrics["diag_flow_loss"]),
                "historical_primary_loss": float(historical_metrics["primary_loss"]),
                "historical_diag_flow_loss": float(historical_metrics["diag_flow_loss"]),
                "historical_total_loss": float(historical_metrics["total_loss"]),
                "historical_secondary_loss": float(historical_metrics["secondary_loss"]),
                "best_validation_primary_loss": float(best_validation_primary),
                "promoted": bool(promoted),
                "success": bool(fit.success),
            }
        )
        challenger_rows.append(
            {
                "candidate_id": config.candidate_id,
                "diagnosis_family": config.diagnosis_family,
                "care_family": config.care_family,
                "hidden_rank": int(config.hidden_rank),
                "use_observation_covariates": bool(config.use_observation_covariates),
                "optimizer_success": bool(fit.success),
                "optimizer_status": int(fit.status),
                "optimizer_message": fit.message,
                "train_cost": float(fit.train_cost),
                "split_metrics": {window_name: {key: float(value) for key, value in metrics.items()} for window_name, metrics in fit.window_metrics.items()},
                "historical_metrics": {key: float(value) for key, value in historical_metrics.items()},
                "year_metrics": serialized_year_metrics,
            }
        )
        if promoted:
            promotion_rows.append(
                {
                    "candidate_id": config.candidate_id,
                    "diagnosis_family": config.diagnosis_family,
                    "care_family": config.care_family,
                    "selection_window": "validation",
                    "validation_score_tuple": list(_score_tuple(validation_metrics)),
                    "holdout_report_tuple": list(_score_tuple(holdout_metrics)),
                }
            )
        else:
            reasons: list[str] = []
            if not fit.success:
                reasons.append("optimizer_failed")
            if validation_metrics["non_finite_count"] > 0.0 or holdout_metrics["non_finite_count"] > 0.0:
                reasons.append("non_finite_predictions")
            if validation_metrics["population_violation_count"] > 0.0 or holdout_metrics["population_violation_count"] > 0.0:
                reasons.append("population_constraint_violation")
            if champion is not None:
                reasons.append("promotion_gate_not_met")
            rejected_rows.append(
                {
                    "candidate_id": config.candidate_id,
                    "diagnosis_family": config.diagnosis_family,
                    "care_family": config.care_family,
                    "reasons": reasons or ["candidate_not_selected"],
                    "validation_score_tuple": list(_score_tuple(validation_metrics)),
                    "holdout_report_tuple": list(_score_tuple(holdout_metrics)),
                }
            )
        write_json(ctx.experiment_dir / "challenger_log.json", challenger_rows)
        write_json(ctx.experiment_dir / "rejected_mutations.json", rejected_rows)
        write_json(ctx.experiment_dir / "stage1_calibration_frontier.json", frontier_rows)
        write_json(ctx.experiment_dir / "promotion_audit.json", promotion_rows)

    for config in base_candidate_configs:
        _evaluate_candidate(config)

    if champion is None:
        raise RuntimeError("Integrated Phase 3 autoresearch could not find a feasible Stage 1 champion")

    branch_candidate_configs = _branch_candidate_configs(snapshot, champion.config)
    for config in branch_candidate_configs:
        _evaluate_candidate(config)

    candidate_configs = base_candidate_configs + branch_candidate_configs

    stage2 = _stage2_target_search(snapshot, champion)
    baseline_comparison = _baseline_comparison(
        snapshot,
        champion,
        fit_mask=contract.train_mask,
        report_masks=report_masks,
        blocked_contract=contract,
    )
    serialized_champion_year_metrics = _serialize_year_metrics(snapshot, champion.year_metrics)
    experiment_spec = {
        "experiment_id": ctx.experiment.experiment_id,
        "description": ctx.experiment.description,
        "source_run_id": snapshot.source_run_id,
        "candidate_count": int(len(candidate_configs)),
        "feature_count": int(len(snapshot.feature_ids)),
        "hidden_rank_max": int(snapshot.hidden_rank_max),
        "historical_end_quarter": snapshot.historical_end_quarter,
        "historical_years": list(snapshot.historical_years),
        "target_horizon": snapshot.target_horizon,
        "selection_window": "validation",
        "holdout_window": "holdout",
    }
    coverage_summary = {
        "model_quarter_count": int(len(snapshot.model_quarters)),
        "historical_quarter_count": int(len(snapshot.historical_quarters)),
        "future_quarter_count": int(len(snapshot.future_quarters)),
        "feature_count": int(len(snapshot.feature_ids)),
        "train_quarter_count": int(np.sum(contract.train_mask)),
        "validation_quarter_count": int(np.sum(contract.validation_mask)),
        "holdout_quarter_count": int(np.sum(contract.holdout_mask)),
    }
    decision = {
        "passed": True,
        "stage1_champion_candidate_id": champion.config.candidate_id,
        "stage1_champion_diagnosis_family": champion.config.diagnosis_family,
        "stage1_champion_care_family": champion.config.care_family,
        "stage1_selection_window": "validation",
        "stage1_validation_score_tuple": list(_score_tuple(champion.window_metrics["validation"])),
        "stage1_holdout_report_tuple": list(_score_tuple(champion.window_metrics["holdout"])),
        "stage1_historical_score_tuple": list(_score_tuple(champion.window_metrics["historical"])),
        "publication_gate_passed": bool(
            bool((baseline_comparison.get("split_comparison") or {}).get("validation", {}).get("model_beats_carry_forward"))
            and bool((baseline_comparison.get("split_comparison") or {}).get("validation", {}).get("model_beats_simple_compartmental"))
            and bool((baseline_comparison.get("split_comparison") or {}).get("holdout", {}).get("model_beats_carry_forward"))
            and bool((baseline_comparison.get("split_comparison") or {}).get("holdout", {}).get("model_beats_simple_compartmental"))
        ),
        "stage2_optimized_shortfall": float(stage2["optimized_target"]["horizon_shortfall"]),
        "stage2_baseline_shortfall": float(stage2["baseline_target"]["horizon_shortfall"]),
        "stage2_optimization_success": bool(stage2["optimization_success"]),
    }
    numeric_policy = [
        numerical_guard_entry(
            name="float64_machine_epsilon",
            role="finite arithmetic guard for division, scaling, and residual calculations",
            why_needed="The integrated autoresearch loop computes normalized residuals, SVD tolerances, and finite hazard transforms.",
        ),
        {
            "name": "cascade_target_probability",
            "value": CASCADE_TARGET,
            "role": "target ratio for each component of the 95-95-95 cascade objective",
            "source_type": "physical_constraint",
            "estimation_data": "UNAIDS 95-95-95 target definition",
            "estimation_method": "fixed public program target",
            "uncertainty": "program target constant",
            "why_needed": "Stage 2 shortfall is defined against the 95-95-95 target and must use the exact target ratio.",
        },
    ]
    write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=numeric_policy,
    )
    write_json(ctx.experiment_dir / "epoch_spec.json", experiment_spec)
    write_json(
        ctx.experiment_dir / "data_snapshot_manifest.json",
        {
            "source_run_id": snapshot.source_run_id,
            "model_quarters": list(snapshot.model_quarters),
            "historical_quarters": list(snapshot.historical_quarters),
            "future_quarters": list(snapshot.future_quarters),
            "historical_end_quarter": snapshot.historical_end_quarter,
            "target_horizon": snapshot.target_horizon,
            "metric_observation_coverage": _metric_observation_coverage(snapshot),
            "blocked_time_contract": {
                "diagnosis_flow_observed_quarters": list(contract.diagnosis_flow_observed_quarters),
                "train_diagnosis_quarters": list(contract.train_diagnosis_quarters),
                "validation_quarters": list(contract.validation_quarters),
                "holdout_quarters": list(contract.holdout_quarters),
                "train_end_quarter": contract.train_end_quarter,
                "validation_start_quarter": contract.validation_start_quarter,
                "validation_end_quarter": contract.validation_end_quarter,
                "holdout_start_quarter": contract.holdout_start_quarter,
                "holdout_end_quarter": contract.holdout_end_quarter,
            },
            "evidence_provenance_summary": snapshot.evidence_provenance["summary"],
        },
    )
    write_json(
        ctx.experiment_dir / "evaluation_contract.json",
        {
            "stage1_primary_metrics": list(PRIMARY_METRICS),
            "stage1_secondary_metrics": list(SECONDARY_METRICS),
            "stage1_total_metric": TOTAL_METRIC,
            "stage1_fit_window": "blocked_train",
            "stage1_selection_window": "blocked_validation",
            "stage1_holdout_window": "blocked_holdout_report_only",
            "stage1_keep_gate": "promote only by validation score after fitting on blocked train quarters; holdout is report-only and not used for candidate promotion",
            "stage2_target": "exploratory bounded direct-surface lever search toward latent 95-95-95; not an acceptance criterion and not a validated policy claim",
            "blocked_time_contract": {
                "train_end_quarter": contract.train_end_quarter,
                "validation_start_quarter": contract.validation_start_quarter,
                "validation_end_quarter": contract.validation_end_quarter,
                "holdout_start_quarter": contract.holdout_start_quarter,
                "holdout_end_quarter": contract.holdout_end_quarter,
            },
            "yearly_reporting_contract": {
                "diag_flow_loss_when_unobserved": None,
                "diagnosis_flow_status_key": "diagnosis_flow_status",
                "diagnosis_flow_status_values": ["observed", "not observed"],
                "observation_flags_key": "observation_flags",
            },
        },
    )
    write_json(ctx.experiment_dir / "evidence_provenance.json", snapshot.evidence_provenance)
    write_json(ctx.experiment_dir / "phase2_insertion_contract.json", snapshot.phase2_insertion_contract)
    write_json(ctx.experiment_dir / "identifiability_contract.json", _identifiability_contract(snapshot, contract))
    write_json(ctx.experiment_dir / "challenger_log.json", challenger_rows)
    write_json(ctx.experiment_dir / "rejected_mutations.json", rejected_rows)
    write_json(ctx.experiment_dir / "stage1_calibration_frontier.json", frontier_rows)
    write_json(ctx.experiment_dir / "promotion_audit.json", promotion_rows)
    write_json(
        ctx.experiment_dir / "champion_table.json",
        {
            "stage1_champion": {
                "candidate_id": champion.config.candidate_id,
                "train_metrics": champion.window_metrics["train"],
                "validation_metrics": champion.window_metrics["validation"],
                "holdout_metrics": champion.window_metrics["holdout"],
                "historical_metrics": champion.window_metrics["historical"],
                "year_metrics": serialized_champion_year_metrics,
            },
            "baseline_comparison": baseline_comparison,
            "stage2": {
                "baseline_shortfall": stage2["baseline_target"]["horizon_shortfall"],
                "optimized_shortfall": stage2["optimized_target"]["horizon_shortfall"],
                "lever_values": stage2["lever_values"],
                "interpretation": "exploratory_only",
            },
        },
    )
    write_json(ctx.experiment_dir / "mechanistic_spec.json", champion.feature_summary)
    write_json(ctx.experiment_dir / "fit_artifact.json", {"train_cost": champion.train_cost, "optimizer_message": champion.message, "optimizer_status": champion.status})
    write_json(
        ctx.experiment_dir / "evaluation.json",
        {
            "train_metrics": champion.window_metrics["train"],
            "validation_metrics": champion.window_metrics["validation"],
            "holdout_metrics": champion.window_metrics["holdout"],
            "historical_metrics": champion.window_metrics["historical"],
            "year_metrics": serialized_champion_year_metrics,
            "baseline_comparison": baseline_comparison,
            "historical_end_quarter": snapshot.historical_end_quarter,
            "target_horizon": snapshot.target_horizon,
            "stage2_baseline_shortfall": stage2["baseline_target"]["horizon_shortfall"],
            "stage2_optimized_shortfall": stage2["optimized_target"]["horizon_shortfall"],
            "stage2_interpretation": "exploratory_only",
        },
    )
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "state_trajectory_rows.json", _state_rows(snapshot, champion.simulation))
    write_json(ctx.experiment_dir / "transition_hazard_summary.json", {"rows": _transition_hazard_rows(snapshot, champion.simulation)})
    write_json(ctx.experiment_dir / "observation_process_summary.json", _observation_process_summary(snapshot, champion))
    write_json(ctx.experiment_dir / "stage2_target_frontier.json", stage2["frontier_rows"])
    write_json(
        ctx.experiment_dir / "cascade_target_projection.json",
        {
            "baseline": stage2["baseline_target"],
            "optimized": stage2["optimized_target"],
            "lever_values": stage2["lever_values"],
        },
    )
    write_json(
        ctx.experiment_dir / "mechanistic_forecast.json",
        {
            "stage1_rows": _state_rows(snapshot, champion.simulation),
            "stage2_rows": _state_rows(snapshot, stage2["optimized_simulation"]),
        },
    )
    write_json(ctx.experiment_dir / "stage1_yearly_metrics.json", serialized_champion_year_metrics)
    _stage1_frontier_chart(frontier_rows, ctx.experiment_dir / "stage1_calibration_frontier.png")
    _stage1_fit_chart(snapshot, champion, ctx.experiment_dir / "stage1_fit_vs_observed.png")
    _stage1_yearly_chart(champion, ctx.experiment_dir / "stage1_yearly_primary_loss.png")
    _baseline_comparison_chart(baseline_comparison, ctx.experiment_dir / "baseline_comparison.png")
    _stage2_frontier_chart(stage2["frontier_rows"], ctx.experiment_dir / "stage2_target_frontier.png")
    baseline_cascade_rows = _target_objective_rows(snapshot, champion.simulation)["cascade_rows"]
    optimized_cascade_rows = _target_objective_rows(snapshot, stage2["optimized_simulation"])["cascade_rows"]
    _stage2_projection_chart(snapshot, baseline_cascade_rows, optimized_cascade_rows, ctx.experiment_dir / "stage2_cascade_projection.png")
    return {
        "decision": decision,
        "stage1_champion": champion.config.candidate_id,
        "stage2_shortfall": stage2["optimized_target"]["horizon_shortfall"],
        "experiment_dir": str(ctx.experiment_dir),
    }


__all__ = ["run_phase3_v2_int"]
