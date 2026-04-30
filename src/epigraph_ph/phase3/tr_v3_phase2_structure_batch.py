from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase2.latent_temporal_graph import _weighted_standardize
from epigraph_ph.phase2.temporal_optimizer import fit_weighted_sparse_plus_low_rank, regularization_maxima
from epigraph_ph.phase3 import tr_v3_eval_hardening_batch as hardening
from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_publishability_batch as publish
from epigraph_ph.phase3.tr_v3_05_autoresearch import PRIMARY_METRICS, build_annual_anchor_rows, quarter_gap, quarter_sort_key
from epigraph_ph.runtime import ensure_dir, write_json


EXACT_BASE_EXPERIMENT_ID = "EXP-R10-M1-F1-C1"
DENSE_BASE_EXPERIMENT_ID = "EXP-R10-DENSE-M1-C1-H1"
LOCKBOX_YEARS: list[int] = [2025]
MIN_STATE_ROWS = 5
MIN_SAMPLE_ROWS = 4
MIN_QUARTER_SUPPORT = 2


@dataclass(frozen=True, slots=True)
class StructuralOverlayConfig:
    config_id: str
    sparse_frac: float
    lowrank_frac: float
    correction_weight: float
    component_mode: str
    clip_multiplier: float = 1.25


STRUCTURAL_CANDIDATES: tuple[StructuralOverlayConfig, ...] = (
    StructuralOverlayConfig("P2-LR-l10-w25", sparse_frac=0.0, lowrank_frac=0.10, correction_weight=0.25, component_mode="low_rank"),
    StructuralOverlayConfig("P2-LR-l25-w25", sparse_frac=0.0, lowrank_frac=0.25, correction_weight=0.25, component_mode="low_rank"),
    StructuralOverlayConfig("P2-LR-l10-w50", sparse_frac=0.0, lowrank_frac=0.10, correction_weight=0.50, component_mode="low_rank"),
    StructuralOverlayConfig("P2-LR-l25-w50", sparse_frac=0.0, lowrank_frac=0.25, correction_weight=0.50, component_mode="low_rank"),
    StructuralOverlayConfig("P2-FULL-s10-l10-w25", sparse_frac=0.10, lowrank_frac=0.10, correction_weight=0.25, component_mode="full"),
    StructuralOverlayConfig("P2-FULL-s20-l20-w25", sparse_frac=0.20, lowrank_frac=0.20, correction_weight=0.25, component_mode="full"),
    StructuralOverlayConfig("P2-FULL-s10-l10-w50", sparse_frac=0.10, lowrank_frac=0.10, correction_weight=0.50, component_mode="full"),
    StructuralOverlayConfig("P2-FULL-s20-l20-w50", sparse_frac=0.20, lowrank_frac=0.20, correction_weight=0.50, component_mode="full"),
)


def _suite_result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): dict(row) for row in list(payload.get("results") or []) if row.get("status") == "executed"}


def _metric_diag_lookup(transition_diagnostics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    d_to_a = dict(transition_diagnostics.get("D_to_A") or {})
    a_to_v = dict(transition_diagnostics.get("A_to_V") or {})
    return {
        "diagnosed_plhiv": dict(d_to_a.get("diagnosed_level_diagnostics") or {}),
        "alive_on_art": dict(d_to_a.get("art_level_diagnostics") or {}),
        "new_diagnosed_cases_period": dict(a_to_v.get("flow_level_diagnostics") or {}),
    }


def _build_train_residual_state(transition_diagnostics: dict[str, Any]) -> dict[str, Any]:
    diag_lookup = _metric_diag_lookup(transition_diagnostics)
    quarter_rows: dict[str, dict[str, float]] = {}
    for metric_name in PRIMARY_METRICS:
        diag = dict(diag_lookup.get(metric_name) or {})
        for quarter, observed, fitted, supported in zip(
            list(diag.get("train_quarters") or []),
            list(diag.get("train_observed") or []),
            list(diag.get("train_fitted") or []),
            list(diag.get("train_supported") or []),
            strict=False,
        ):
            if not supported or observed is None or fitted is None:
                continue
            quarter_rows.setdefault(str(quarter), {})[metric_name] = float(observed) - float(fitted)
    ordered_quarters = sorted(list(quarter_rows.keys()), key=quarter_sort_key)
    rows: list[dict[str, Any]] = []
    for quarter in ordered_quarters:
        metric_map = dict(quarter_rows[quarter])
        support_count = sum(1 for metric_name in PRIMARY_METRICS if metric_name in metric_map)
        if support_count < int(MIN_QUARTER_SUPPORT):
            continue
        rows.append(
            {
                "quarter": str(quarter),
                "vector": [float(metric_map.get(metric_name, 0.0)) for metric_name in PRIMARY_METRICS],
                "support_count": int(support_count),
            }
        )
    return {
        "quarters": [str(row["quarter"]) for row in rows],
        "matrix": np.asarray([row["vector"] for row in rows], dtype=np.float32) if rows else np.zeros((0, len(PRIMARY_METRICS)), dtype=np.float32),
        "support_counts": np.asarray([float(row["support_count"]) for row in rows], dtype=np.float32) if rows else np.zeros((0,), dtype=np.float32),
    }


def _fit_phase2_structural_model(
    transition_diagnostics: dict[str, Any],
    cfg: StructuralOverlayConfig,
) -> dict[str, Any] | None:
    state_payload = _build_train_residual_state(transition_diagnostics)
    quarters = list(state_payload["quarters"])
    state_matrix = np.asarray(state_payload["matrix"], dtype=np.float32)
    support_counts = np.asarray(state_payload["support_counts"], dtype=np.float32)
    if state_matrix.shape[0] < int(MIN_STATE_ROWS):
        return None
    state_weights = np.clip(support_counts / max(float(len(PRIMARY_METRICS)), 1.0), 0.25, 1.0)
    standardized_state, state_mean, state_scale = _weighted_standardize(state_matrix, state_weights)
    design_rows: list[np.ndarray] = []
    response_rows: list[np.ndarray] = []
    sample_weights: list[float] = []
    pair_quarters: list[str] = []
    for idx in range(1, len(quarters)):
        if quarter_gap(str(quarters[idx - 1]), str(quarters[idx])) != 1:
            continue
        design_rows.append(np.asarray(standardized_state[idx - 1], dtype=np.float32))
        response_rows.append(np.asarray(standardized_state[idx], dtype=np.float32))
        sample_weights.append(float(np.mean([state_weights[idx - 1], state_weights[idx]])))
        pair_quarters.append(str(quarters[idx]))
    if len(design_rows) < int(MIN_SAMPLE_ROWS):
        return None
    design_matrix = np.asarray(design_rows, dtype=np.float32)
    response_matrix = np.asarray(response_rows, dtype=np.float32)
    sample_weight_vector = np.asarray(sample_weights, dtype=np.float32)
    maxima = regularization_maxima(
        response_matrix=response_matrix,
        design_matrix=design_matrix,
        sample_weights=sample_weight_vector,
        block_count=len(PRIMARY_METRICS),
        max_lag=1,
    )
    sparse_penalty = float(cfg.sparse_frac) * float(maxima["lambda_sparse_max"])
    lowrank_penalty = float(cfg.lowrank_frac) * float(maxima["lambda_low_rank_max"])
    sparse_matrix, low_rank_matrix, diagnostics = fit_weighted_sparse_plus_low_rank(
        response_matrix=response_matrix,
        design_matrix=design_matrix,
        sample_weights=sample_weight_vector,
        lambda_sparse=sparse_penalty,
        lambda_low_rank=lowrank_penalty,
        max_iterations=80,
        convergence_tol=1e-4,
        block_count=len(PRIMARY_METRICS),
        max_lag=1,
    )
    if str(cfg.component_mode) == "low_rank":
        coefficient = np.asarray(low_rank_matrix, dtype=np.float32)
    elif str(cfg.component_mode) == "sparse":
        coefficient = np.asarray(sparse_matrix, dtype=np.float32)
    else:
        coefficient = np.asarray(sparse_matrix + low_rank_matrix, dtype=np.float32)
    train_min = np.min(standardized_state, axis=0)
    train_max = np.max(standardized_state, axis=0)
    spread = np.maximum(train_max - train_min, 1e-3)
    lower = train_min - (float(cfg.clip_multiplier) * spread)
    upper = train_max + (float(cfg.clip_multiplier) * spread)
    return {
        "config_id": str(cfg.config_id),
        "quarters": list(quarters),
        "state_mean": np.asarray(state_mean, dtype=np.float32),
        "state_scale": np.asarray(state_scale, dtype=np.float32),
        "last_state": np.asarray(standardized_state[-1], dtype=np.float32),
        "coefficient": np.asarray(coefficient, dtype=np.float32),
        "sparse_matrix": np.asarray(sparse_matrix, dtype=np.float32),
        "low_rank_matrix": np.asarray(low_rank_matrix, dtype=np.float32),
        "clip_lower": np.asarray(lower, dtype=np.float32),
        "clip_upper": np.asarray(upper, dtype=np.float32),
        "hidden_rank": int(diagnostics.get("complexity", {}).get("hidden_rank", 0)),
        "nnz_sparse": int(diagnostics.get("complexity", {}).get("nnz_sparse", 0)),
        "train_pair_count": int(len(pair_quarters)),
        "objective": dict(diagnostics.get("objective") or {}),
    }


def _corrected_prediction_rows(
    base_prediction_rows: list[dict[str, Any]],
    model: dict[str, Any] | None,
    cfg: StructuralOverlayConfig | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if model is None or cfg is None:
        return [dict(row) for row in base_prediction_rows], []
    correction_rows: list[dict[str, Any]] = []
    corrected_rows: list[dict[str, Any]] = []
    state = np.asarray(model["last_state"], dtype=np.float32)
    coefficient = np.asarray(model["coefficient"], dtype=np.float32)
    lower = np.asarray(model["clip_lower"], dtype=np.float32)
    upper = np.asarray(model["clip_upper"], dtype=np.float32)
    state_mean = np.asarray(model["state_mean"], dtype=np.float32)
    state_scale = np.asarray(model["state_scale"], dtype=np.float32)
    weight = float(cfg.correction_weight)
    for base_row in list(base_prediction_rows):
        next_state = np.clip(state @ coefficient, lower, upper)
        residual_vector = (next_state * state_scale) + state_mean
        diagnosed_correction = float(weight * residual_vector[0])
        art_correction = float(weight * residual_vector[1])
        flow_correction = float(weight * residual_vector[2])
        diagnosed_value = float(max(float(base_row.get("diagnosed_plhiv") or 0.0) + diagnosed_correction, 0.0))
        art_value = float(np.clip(float(base_row.get("alive_on_art") or 0.0) + art_correction, 0.0, diagnosed_value))
        flow_value = float(max(float(base_row.get("new_diagnosed_cases_period") or 0.0) + flow_correction, 0.0))
        corrected_row = dict(base_row)
        corrected_row["diagnosed_plhiv"] = diagnosed_value
        corrected_row["alive_on_art"] = art_value
        corrected_row["new_diagnosed_cases_period"] = flow_value
        if corrected_row.get("virally_suppressed") is not None:
            corrected_row["virally_suppressed"] = float(np.clip(float(corrected_row["virally_suppressed"]), 0.0, art_value))
        corrected_rows.append(corrected_row)
        correction_rows.append(
            {
                "quarter": str(base_row["quarter"]),
                "diagnosed_correction": float(diagnosed_correction),
                "art_correction": float(art_correction),
                "flow_correction": float(flow_correction),
                "state_norm": float(np.linalg.norm(next_state)),
            }
        )
        state = next_state.astype(np.float32)
    return corrected_rows, correction_rows


def _raw_endpoint_row(
    prediction_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    allowed_tiers: set[str],
    metric_scales: dict[str, float],
    eps: float,
) -> dict[str, Any]:
    audit = suite._raw_endpoint_audit(
        prediction_rows,
        holdout_rows,
        metric_scales,
        allowed_tiers=allowed_tiers,
        eps=eps,
    )
    by_metric = dict(audit.get("by_metric") or {})
    return {
        "diagnosed_raw_mae": float(dict(by_metric.get("diagnosed_plhiv") or {}).get("raw_mae") or 0.0),
        "art_raw_mae": float(dict(by_metric.get("alive_on_art") or {}).get("raw_mae") or 0.0),
        "flow_raw_mae": float(dict(by_metric.get("new_diagnosed_cases_period") or {}).get("raw_mae") or 0.0),
    }


def _evaluate_split_overlay(
    split: dict[str, Any],
    *,
    allowed_tiers: set[str],
    cfg: StructuralOverlayConfig,
) -> dict[str, Any] | None:
    model = _fit_phase2_structural_model(dict(split.get("transition_diagnostics") or {}), cfg)
    if model is None:
        return None
    corrected_rows, correction_rows = _corrected_prediction_rows(list(split.get("candidate_prediction_rows") or []), model, cfg)
    holdout_rows = list(split.get("holdout_target_rows") or [])
    metric_scales = dict(split.get("candidate", {}).get("metric_scales") or split.get("metric_scales") or {})
    if not metric_scales:
        metric_scales = {metric_name: 1.0 for metric_name in PRIMARY_METRICS}
        for metric_name in PRIMARY_METRICS:
            targets = [abs(float(row[metric_name])) for row in holdout_rows if row.get(metric_name) is not None]
            if targets:
                metric_scales[metric_name] = max(float(np.median(np.asarray(targets, dtype=np.float64))), 1.0)
    eps = float(split.get("candidate", {}).get("eps") or 1e-6)
    split_mae = suite._filtered_normalized_mae(
        corrected_rows,
        holdout_rows,
        metric_scales,
        allowed_tiers=allowed_tiers,
        eps=eps,
    )
    raw_metrics = _raw_endpoint_row(corrected_rows, holdout_rows, allowed_tiers=allowed_tiers, metric_scales=metric_scales, eps=eps)
    return {
        "holdout_years": list(split.get("holdout_years") or []),
        "quarterly_mean_mae": float(split_mae),
        "prediction_rows": corrected_rows,
        "correction_rows": correction_rows,
        "raw_metrics": raw_metrics,
        "hidden_rank": int(model["hidden_rank"]),
        "nnz_sparse": int(model["nnz_sparse"]),
        "train_pair_count": int(model["train_pair_count"]),
        "objective": dict(model["objective"]),
    }


def _aggregate_overlay_rows(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not split_rows:
        return {
            "quarterly_mean_mae": None,
            "diagnosed_raw_mae": None,
            "art_raw_mae": None,
            "flow_raw_mae": None,
            "hidden_rank_mean": None,
            "nnz_sparse_mean": None,
            "split_count": 0,
        }
    return {
        "quarterly_mean_mae": float(np.mean(np.asarray([float(row["quarterly_mean_mae"]) for row in split_rows], dtype=np.float64))),
        "diagnosed_raw_mae": float(np.mean(np.asarray([float(row["raw_metrics"]["diagnosed_raw_mae"]) for row in split_rows], dtype=np.float64))),
        "art_raw_mae": float(np.mean(np.asarray([float(row["raw_metrics"]["art_raw_mae"]) for row in split_rows], dtype=np.float64))),
        "flow_raw_mae": float(np.mean(np.asarray([float(row["raw_metrics"]["flow_raw_mae"]) for row in split_rows], dtype=np.float64))),
        "hidden_rank_mean": float(np.mean(np.asarray([float(row["hidden_rank"]) for row in split_rows], dtype=np.float64))),
        "nnz_sparse_mean": float(np.mean(np.asarray([float(row["nnz_sparse"]) for row in split_rows], dtype=np.float64))),
        "split_count": int(len(split_rows)),
    }


def _evaluate_contract_structural_candidates(
    base_result: dict[str, Any],
    *,
    allowed_tiers: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_summary = dict(base_result.get("quarterly_summary") or {})
    base_endpoint = dict(dict(base_summary.get("endpoint_audit_summary") or {}).get("candidate") or {})
    base_by_metric = dict(base_endpoint.get("by_metric") or {})
    rows.append(
        {
            "config_id": "BASE",
            "status": "baseline",
            "quarterly_mean_mae": float(base_summary.get("candidate_mean_mae") or 0.0),
            "diagnosed_raw_mae": float(dict(base_by_metric.get("diagnosed_plhiv") or {}).get("raw_mae") or 0.0),
            "art_raw_mae": float(dict(base_by_metric.get("alive_on_art") or {}).get("raw_mae") or 0.0),
            "flow_raw_mae": float(dict(base_by_metric.get("new_diagnosed_cases_period") or {}).get("raw_mae") or 0.0),
            "hidden_rank_mean": None,
            "nnz_sparse_mean": None,
            "split_count": int(len(list(base_result.get("quarterly_rows") or []))),
            "split_rows": [],
        }
    )
    for cfg in STRUCTURAL_CANDIDATES:
        split_rows = []
        for split in list(base_result.get("quarterly_rows") or []):
            split_payload = _evaluate_split_overlay(dict(split), allowed_tiers=allowed_tiers, cfg=cfg)
            if split_payload is not None:
                split_rows.append(split_payload)
        aggregate = _aggregate_overlay_rows(split_rows)
        rows.append(
            {
                "config_id": str(cfg.config_id),
                "status": "executed" if split_rows else "insufficient_history",
                **aggregate,
                "split_rows": split_rows,
            }
        )
    return rows


def _custom_fixed_holdout_base_candidate(
    *,
    archive_run_id: str,
    contract_name: str,
    base_experiment_id: str,
    frozen_config: dict[str, Any],
    holdout_years: list[int],
) -> tuple[Any, dict[str, Any], set[str]]:
    observation_rows = publish._observation_rows_for_lockbox(archive_run_id, contract_name=contract_name, holdout_years=holdout_years)
    annual_rows = build_annual_anchor_rows(archive_run_id)
    dataset = suite.build_quarterly_dataset(observation_rows, list(holdout_years))
    scoring_tiers = publish._scoring_tiers(contract_name)
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    dynamic_cfg = suite.DynamicControlConfig(**dict(frozen_config.get("dynamic_cfg") or {}))
    observation_cfg = suite.ObservationConfig(**dict(frozen_config.get("observation_cfg") or {}))
    candidate = suite._fit_05a_experiment_candidate(dataset, annual_rows, dynamic_cfg, observation_cfg, spec_map[base_experiment_id])
    return dataset, candidate, scoring_tiers


def _evaluate_fixed_holdout_overlay(
    *,
    archive_run_id: str,
    contract_name: str,
    base_experiment_id: str,
    frozen_config: dict[str, Any],
    holdout_years: list[int],
    cfg: StructuralOverlayConfig,
) -> dict[str, Any] | None:
    dataset, base_candidate, scoring_tiers = _custom_fixed_holdout_base_candidate(
        archive_run_id=archive_run_id,
        contract_name=contract_name,
        base_experiment_id=base_experiment_id,
        frozen_config=frozen_config,
        holdout_years=holdout_years,
    )
    model = _fit_phase2_structural_model(dict(base_candidate.get("transition_diagnostics") or {}), cfg)
    if model is None:
        return None
    corrected_rows, correction_rows = _corrected_prediction_rows(list(base_candidate.get("prediction_rows") or []), model, cfg)
    mae = suite._filtered_normalized_mae(
        corrected_rows,
        dataset.holdout_rows,
        dataset.metric_scales,
        allowed_tiers=scoring_tiers,
        eps=dataset.eps,
    )
    raw_metrics = _raw_endpoint_row(corrected_rows, dataset.holdout_rows, allowed_tiers=scoring_tiers, metric_scales=dataset.metric_scales, eps=dataset.eps)
    return {
        "config_id": str(cfg.config_id),
        "quarterly_mean_mae": float(mae),
        "raw_metrics": raw_metrics,
        "prediction_rows": corrected_rows,
        "correction_rows": correction_rows,
        "hidden_rank": int(model["hidden_rank"]),
        "nnz_sparse": int(model["nnz_sparse"]),
    }


def _evaluate_fixed_holdout_base(
    *,
    archive_run_id: str,
    contract_name: str,
    base_experiment_id: str,
    frozen_config: dict[str, Any],
    holdout_years: list[int],
) -> dict[str, Any]:
    dataset, base_candidate, scoring_tiers = _custom_fixed_holdout_base_candidate(
        archive_run_id=archive_run_id,
        contract_name=contract_name,
        base_experiment_id=base_experiment_id,
        frozen_config=frozen_config,
        holdout_years=holdout_years,
    )
    mae = suite._filtered_normalized_mae(
        list(base_candidate.get("prediction_rows") or []),
        dataset.holdout_rows,
        dataset.metric_scales,
        allowed_tiers=scoring_tiers,
        eps=dataset.eps,
    )
    raw_metrics = _raw_endpoint_row(
        list(base_candidate.get("prediction_rows") or []),
        dataset.holdout_rows,
        allowed_tiers=scoring_tiers,
        metric_scales=dataset.metric_scales,
        eps=dataset.eps,
    )
    return {
        "experiment_id": str(base_experiment_id),
        "quarterly_mean_mae": float(mae),
        "raw_metrics": raw_metrics,
        "prediction_rows": list(base_candidate.get("prediction_rows") or []),
    }


def _save_overview(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    executed = [row for row in rows if row.get("quarterly_mean_mae") is not None]
    if not executed:
        suite._plot_placeholder(path, title=title, body="No executed structural candidates.")
        return
    labels = [str(row["config_id"]) for row in executed]
    values = [float(row["quarterly_mean_mae"]) for row in executed]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, max(4.0, len(labels) * 0.45)))
    ax.barh(y, values, color=["#999999" if label == "BASE" else "#1f78b4" for label in labels])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Quarterly normalized MAE")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_lockbox_curve(path: Path, *, title: str, target_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], corrected_rows: list[dict[str, Any]]) -> None:
    if not target_rows or not base_rows or not corrected_rows:
        suite._plot_placeholder(path, title=title, body="No lockbox rows.")
        return
    quarters = [str(row["quarter"]) for row in target_rows]
    x = np.arange(len(quarters))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, metric_name, label in zip(axes, PRIMARY_METRICS, ("Diagnosed stock", "ART stock", "Diagnosis flow"), strict=False):
        target = [float(row.get(metric_name) or 0.0) for row in target_rows]
        base = [float(row.get(metric_name) or 0.0) for row in base_rows]
        corrected = [float(row.get(metric_name) or 0.0) for row in corrected_rows]
        ax.plot(x, target, label="Observed", color="#333333", marker="o")
        ax.plot(x, base, label="Base", color="#1f78b4", linestyle="--", marker="o")
        ax.plot(x, corrected, label="Phase2-struct", color="#d95f02", marker="o")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    step = max(len(quarters) // 8, 1)
    axes[-1].set_xticks(x[::step])
    axes[-1].set_xticklabels(quarters[::step], rotation=45, ha="right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _pick_best_nonbaseline(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("status") == "executed" and row.get("config_id") != "BASE" and row.get("quarterly_mean_mae") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["quarterly_mean_mae"]))


def _lane_report_payload(
    *,
    lane_name: str,
    contract_name: str,
    base_experiment_id: str,
    base_result: dict[str, Any],
    archive_run_id: str,
) -> dict[str, Any]:
    allowed_tiers = publish._scoring_tiers(contract_name)
    rows = _evaluate_contract_structural_candidates(base_result, allowed_tiers=allowed_tiers)
    base_row = next(row for row in rows if row["config_id"] == "BASE")
    base_lockbox = _evaluate_fixed_holdout_base(
        archive_run_id=archive_run_id,
        contract_name=contract_name,
        base_experiment_id=base_experiment_id,
        frozen_config=dict(base_result.get("best_candidate") or {}),
        holdout_years=list(LOCKBOX_YEARS),
    )
    best_row = _pick_best_nonbaseline(rows)
    if best_row is None or float(best_row["quarterly_mean_mae"]) >= float(base_row["quarterly_mean_mae"]):
        decision = {
            "status": "revert",
            "winner_id": "BASE",
            "why": "No Phase-2-compatible structural residual overlay beat the frozen champion on the primary rolling-origin contract.",
        }
        lockbox = None
    else:
        fixed = _evaluate_fixed_holdout_overlay(
            archive_run_id=archive_run_id,
            contract_name=contract_name,
            base_experiment_id=base_experiment_id,
            frozen_config=dict(base_result.get("best_candidate") or {}),
            holdout_years=list(LOCKBOX_YEARS),
            cfg=next(cfg for cfg in STRUCTURAL_CANDIDATES if cfg.config_id == best_row["config_id"]),
        )
        if fixed is not None and float(fixed["quarterly_mean_mae"]) < float(base_lockbox["quarterly_mean_mae"]) - 1e-6:
            status = "keep"
            why = "Phase-2-compatible sparse-plus-low-rank residual structure improved rolling-origin performance and also beat the frozen champion on the fixed holdout."
        elif fixed is not None and float(fixed["quarterly_mean_mae"]) <= float(base_lockbox["quarterly_mean_mae"]) + 0.001:
            status = "sensitivity_only"
            why = "Phase-2-compatible structure improved rolling-origin performance, but the fixed holdout was only tied within tolerance rather than decisively better."
        else:
            status = "sensitivity_only"
            why = "Phase-2-compatible structure improved rolling-origin performance, but the fixed holdout did not improve cleanly."
        decision = {
            "status": status,
            "winner_id": str(best_row["config_id"]),
            "why": why,
        }
        lockbox = fixed
    return {
        "lane_name": lane_name,
        "contract_name": contract_name,
        "base_experiment_id": base_experiment_id,
        "rows": rows,
        "decision": decision,
        "base_lockbox": base_lockbox,
        "lockbox": lockbox,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Phase2 Structure Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        "- Note: no frozen Phase 2 source run survives on disk in this workspace, so this batch uses Phase-2-compatible sparse-plus-low-rank residual dynamics learned train-only from the frozen champions.",
        "",
    ]
    for lane_name in ("exact", "dense"):
        lane = dict(payload[lane_name])
        lines.extend(
            [
                f"## {lane_name.title()} lane",
                "",
                f"- Base experiment: `{lane['base_experiment_id']}`",
                f"- Contract: `{lane['contract_name']}`",
                f"- Decision: `{lane['decision']['status']}`",
                f"- Winner: `{lane['decision']['winner_id']}`",
                f"- Why: {lane['decision']['why']}",
                "",
                "| Config | Status | Quarterly MAE | Raw diagnosed MAE | Raw ART MAE | Raw flow MAE | Hidden rank | Sparse nnz |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in list(lane.get("rows") or []):
            q_mae = "" if row.get("quarterly_mean_mae") is None else f"{float(row['quarterly_mean_mae']):.6f}"
            d_mae = "" if row.get("diagnosed_raw_mae") is None else f"{float(row['diagnosed_raw_mae']):.3f}"
            a_mae = "" if row.get("art_raw_mae") is None else f"{float(row['art_raw_mae']):.3f}"
            f_mae = "" if row.get("flow_raw_mae") is None else f"{float(row['flow_raw_mae']):.3f}"
            rank = "" if row.get("hidden_rank_mean") is None else f"{float(row['hidden_rank_mean']):.2f}"
            nnz = "" if row.get("nnz_sparse_mean") is None else f"{float(row['nnz_sparse_mean']):.2f}"
            lines.append(f"| {row['config_id']} | {row['status']} | {q_mae} | {d_mae} | {a_mae} | {f_mae} | {rank} | {nnz} |")
        if lane.get("lockbox"):
            lockbox = dict(lane["lockbox"])
            lines.extend(
                [
                    "",
                    f"- Base lockbox MAE (`{','.join(str(year) for year in LOCKBOX_YEARS)}`): `{float(dict(lane['base_lockbox'])['quarterly_mean_mae']):.6f}`",
                    f"- Structural lockbox MAE (`{','.join(str(year) for year in LOCKBOX_YEARS)}`): `{float(lockbox['quarterly_mean_mae']):.6f}`",
                    f"- Lockbox diagnosed raw MAE: `{float(lockbox['raw_metrics']['diagnosed_raw_mae']):.3f}`",
                ]
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_tr_v3_phase2_structure_batch(
    *,
    run_id: str,
    archive_run_id: str | None = None,
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    archive_run = str(archive_run_id or suite._latest_standard_archive_run())
    analysis_dir = ensure_dir(suite.repo_root() / "artifacts" / "runs" / run_id / "analysis")

    exact_payload = hardening._run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="exact_only",
        experiment_ids=[EXACT_BASE_EXPERIMENT_ID],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    dense_payload = hardening._run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="purged_dense",
        experiment_ids=[DENSE_BASE_EXPERIMENT_ID],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    exact_result = _suite_result_map(exact_payload)[EXACT_BASE_EXPERIMENT_ID]
    dense_result = _suite_result_map(dense_payload)[DENSE_BASE_EXPERIMENT_ID]

    exact_lane = _lane_report_payload(
        lane_name="exact",
        contract_name="exact_only",
        base_experiment_id=EXACT_BASE_EXPERIMENT_ID,
        base_result=exact_result,
        archive_run_id=archive_run,
    )
    dense_lane = _lane_report_payload(
        lane_name="dense",
        contract_name="purged_dense",
        base_experiment_id=DENSE_BASE_EXPERIMENT_ID,
        base_result=dense_result,
        archive_run_id=archive_run,
    )

    _save_overview(list(exact_lane["rows"]), analysis_dir / "phase2_structure_exact_overview.png", title="Phase2-compatible structural overlay | exact lane")
    _save_overview(list(dense_lane["rows"]), analysis_dir / "phase2_structure_dense_overview.png", title="Phase2-compatible structural overlay | dense lane")

    for lane_name, lane, base_result in (("exact", exact_lane, exact_result), ("dense", dense_lane, dense_result)):
        if not lane.get("lockbox"):
            continue
        base_dataset, base_candidate, _ = _custom_fixed_holdout_base_candidate(
            archive_run_id=archive_run,
            contract_name=str(lane["contract_name"]),
            base_experiment_id=str(lane["base_experiment_id"]),
            frozen_config=dict(base_result.get("best_candidate") or {}),
            holdout_years=list(LOCKBOX_YEARS),
        )
        corrected_rows = list(dict(lane["lockbox"]).get("prediction_rows") or [])
        target_rows = [
            {
                "quarter": str(row["quarter"]),
                **{metric_name: row.get(metric_name) for metric_name in PRIMARY_METRICS},
            }
            for row in list(base_dataset.holdout_rows)
        ]
        _save_lockbox_curve(
            analysis_dir / f"phase2_structure_{lane_name}_lockbox.png",
            title=f"Phase2-compatible structural overlay | {lane_name} lockbox",
            target_rows=target_rows,
            base_rows=list(base_candidate.get("prediction_rows") or []),
            corrected_rows=corrected_rows,
        )

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_run_id": archive_run,
        "exact": exact_lane,
        "dense": dense_lane,
        "artifacts": {
            "exact_overview": "phase2_structure_exact_overview.png",
            "dense_overview": "phase2_structure_dense_overview.png",
            "exact_lockbox_curve": "phase2_structure_exact_lockbox.png",
            "dense_lockbox_curve": "phase2_structure_dense_lockbox.png",
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_structure_batch_report.json", report_payload)
    (analysis_dir / "tr_v3_phase2_structure_batch_report.md").write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-2-compatible structural residual sidecars on the frozen TR-V3 champions.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=None)
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    args = parser.parse_args()
    run_tr_v3_phase2_structure_batch(
        run_id=str(args.run_id),
        archive_run_id=args.archive_run_id,
        quarterly_start_year=int(args.quarterly_start_year),
        quarterly_end_year=int(args.quarterly_end_year),
        quarterly_min_train_years=int(args.quarterly_min_train_years),
        annual_start_year=int(args.annual_start_year),
        annual_end_year=int(args.annual_end_year),
        annual_min_train_years=int(args.annual_min_train_years),
        horizon_years=int(args.horizon_years),
    )


if __name__ == "__main__":
    main()
