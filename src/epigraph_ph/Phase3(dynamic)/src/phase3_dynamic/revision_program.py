from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    AUXILIARY_METRICS,
    FLOW_METRICS,
    OBSERVATION_METRICS,
    SNAPSHOT_METRICS,
    build_blocked_time_dataset,
    build_observation_rows,
    default_epigraph_root,
    rolling_origin_splits,
    sandbox_repo_root,
)
from .experiment_tracker import write_experiment_registry
from .failure_anatomy import build_failure_anatomy_payload, markdown_failure_anatomy_report
from .incidence import IncidenceFlowConfig, carry_forward_incidence_flow_paths
from .lineage import write_project_lineage_manifest
from .loop import PHASE2_PRIOR_CONTRACT, _build_data_provenance_payload, _build_observation_score_ledger
from .metrics import PRIMARY_METRICS, quarter_sort_key
from .model import (
    DampingConfig,
    DirectPriorConfig,
    DynamicBaselineConfig,
    HiddenDriverConfig,
    ObservationModelConfig,
    ShockConfig,
    _simulate_sequence,
    carry_forward_hazards,
    forecast_dynamic_baseline,
    simulate_holdout,
)
from .observation_ledger import (
    build_observation_role_ledger,
    resolve_active_source_run_id,
    resolve_baseline_source_run_id,
)
from .phase2 import (
    allowed_direct_edge_keys_from_robustness,
    build_direct_prior_features,
    build_hidden_driver_features,
    direct_prior_feature_count,
    filter_direct_prior_features_by_edge_keys,
    load_phase2_structural_inputs,
    resolve_phase2_determinant_robustness_report,
    resolve_phase2_structural_source_run_id,
)
from .priors import PHASE2_TRANSITION_PRIOR_MAP
from .runtime import ensure_dir, write_json
from .scientific_contracts import build_hazard_semantics, build_model_contract, write_project_contract_artifacts


CLAIM_CARD_SCHEMA_VERSION = "phase3_dynamic_claim_card.v1"
RUN_REPRO_MANIFEST_SCHEMA_VERSION = "phase3_dynamic_run_reproducibility_manifest.v1"
DEFAULT_ACTIVE_SOURCE_RUN_ID = "tr-v3-current-champion-expanded-harp-compatibility-20260419-s00"
DEFAULT_BASELINE_SOURCE_RUN_ID = "harp-archive-wdi-standard-20260412-s19"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _write_markdown(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _spec_payload(spec: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in spec.items():
        payload[key] = asdict(value) if is_dataclass(value) else value
    return payload


def _phase2_feature_count(features_by_transition: Any | None) -> int:
    return int(sum(len(rows) for rows in dict(features_by_transition or {}).values()))


def _candidate_model_contract(
    spec: dict[str, Any],
    *,
    direct_prior_features: Any | None = None,
    hidden_driver_features: Any | None = None,
) -> dict[str, Any]:
    return build_model_contract(
        incidence_enabled=spec.get("incidence_cfg") is not None,
        observation_calibration_enabled=spec.get("observation_cfg") is not None,
        phase2_direct_enabled=spec.get("prior_cfg") is not None and _phase2_feature_count(direct_prior_features) > 0,
        phase2_hidden_enabled=spec.get("hidden_cfg") is not None and _phase2_feature_count(hidden_driver_features) > 0,
    )


def _candidate_hazard_semantics(
    spec: dict[str, Any],
    *,
    direct_prior_features: Any | None = None,
    hidden_driver_features: Any | None = None,
) -> dict[str, Any]:
    return build_hazard_semantics(
        incidence_enabled=spec.get("incidence_cfg") is not None,
        observation_calibration_enabled=spec.get("observation_cfg") is not None,
        phase2_direct_enabled=spec.get("prior_cfg") is not None and _phase2_feature_count(direct_prior_features) > 0,
        phase2_hidden_enabled=spec.get("hidden_cfg") is not None and _phase2_feature_count(hidden_driver_features) > 0,
    )


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    carry_forward_maes = [float(row["carry_forward"]["mae"]) for row in rows]
    candidate_maes = [float(row["candidate"]["mae"]) for row in rows]
    carry_forward_smape = [float(row["carry_forward"]["smape"]) for row in rows]
    candidate_smape = [float(row["candidate"]["smape"]) for row in rows]
    return {
        "carry_forward_mean_mae": float(np.mean(np.asarray(carry_forward_maes, dtype=np.float64))) if carry_forward_maes else float("inf"),
        "candidate_mean_mae": float(np.mean(np.asarray(candidate_maes, dtype=np.float64))) if candidate_maes else float("inf"),
        "carry_forward_mean_smape": float(np.mean(np.asarray(carry_forward_smape, dtype=np.float64))) if carry_forward_smape else float("inf"),
        "candidate_mean_smape": float(np.mean(np.asarray(candidate_smape, dtype=np.float64))) if candidate_smape else float("inf"),
        "carry_forward_worst_mae": max(carry_forward_maes) if carry_forward_maes else float("inf"),
        "candidate_worst_mae": max(candidate_maes) if candidate_maes else float("inf"),
    }


def _mean_share(rows: list[dict[str, Any]], numerator_key: str, denominator_key: str) -> float | None:
    shares: list[float] = []
    for row in rows:
        numerator = row.get(numerator_key)
        denominator = row.get(denominator_key)
        if numerator is None or denominator is None:
            continue
        denominator_value = float(denominator)
        if denominator_value <= 0.0:
            continue
        shares.append(float(numerator) / denominator_value)
    if not shares:
        return None
    return float(np.mean(np.asarray(shares, dtype=np.float64)))


def _train_replay(dataset: Any, candidate_result: dict[str, Any]) -> dict[str, Any]:
    train_rows = sorted(list(dataset.train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if len(train_rows) < 2 or not dataset.train_state_rows:
        return {"prediction_rows": [], "trajectory_rows": []}
    return _simulate_sequence(
        dict(dataset.train_state_rows[0]["state_values"]),
        train_rows[1:],
        dict((candidate_result.get("hazard_paths") or {}).get("train_hazard_map") or {}),
        incidence_inflow_map=dict((candidate_result.get("incidence_paths") or {}).get("train_incidence_inflow_map") or {}),
        incidence_hazard_map=dict((candidate_result.get("incidence_paths") or {}).get("train_incidence_hazard_map") or {}),
        population_denominator_map=dict((candidate_result.get("incidence_paths") or {}).get("train_population_denominator_map") or {}),
        attrition_outflow_map=dict((candidate_result.get("incidence_paths") or {}).get("train_attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict((candidate_result.get("incidence_paths") or {}).get("train_state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=dict((candidate_result.get("incidence_paths") or {}).get("train_exit_channel_state_outflow_map") or {}),
    )


def _fit_linear_head(
    *,
    train_features: list[list[float]],
    train_targets: list[float],
    holdout_features: list[list[float]],
    ridge: float = 0.1,
) -> tuple[list[float], dict[str, Any]] | None:
    if len(train_features) < 4 or len(train_targets) < 4 or len(holdout_features) == 0:
        return None
    x_train = np.asarray(train_features, dtype=np.float64)
    y_train = np.asarray(train_targets, dtype=np.float64)
    ridge_matrix = np.eye(x_train.shape[1], dtype=np.float64) * float(ridge)
    ridge_matrix[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + ridge_matrix, x_train.T @ y_train)
    x_holdout = np.asarray(holdout_features, dtype=np.float64)
    predictions = np.asarray(x_holdout @ beta, dtype=np.float64)
    return (
        [float(max(value, 0.0)) for value in predictions],
        {"coefficients": [float(value) for value in beta]},
    )


def _cd4_features(
    trajectory_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[list[float]]:
    features: list[list[float]] = []
    for trajectory_row, prediction_row in zip(trajectory_rows, prediction_rows):
        state_values = dict(trajectory_row.get("state_values") or {})
        undiagnosed = float(state_values.get("U") or 0.0)
        diagnosed_flow = float(prediction_row.get("new_diagnosed_cases_period") or 0.0)
        features.append(
            [
                1.0,
                float(np.log1p(max(undiagnosed, 0.0))),
                float(np.log1p(max(diagnosed_flow, 0.0))),
            ]
        )
    return features


def _auxiliary_metric_entries(
    *,
    holdout_rows: list[dict[str, Any]],
    candidate_predictions: dict[str, list[float | None]],
    baseline_predictions: dict[str, list[float | None]],
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric_name in metric_names:
        candidate_series = list(candidate_predictions.get(metric_name) or [])
        baseline_series = list(baseline_predictions.get(metric_name) or [])
        entries: list[tuple[float, float, float]] = []
        for row, candidate_value, baseline_value in zip(holdout_rows, candidate_series, baseline_series):
            target_value = row.get(metric_name)
            if target_value is None or candidate_value is None or baseline_value is None:
                continue
            entries.append((float(target_value), float(candidate_value), float(baseline_value)))
        if not entries:
            metrics[metric_name] = {"entry_count": 0}
            continue
        targets = np.asarray([value[0] for value in entries], dtype=np.float64)
        candidate_values = np.asarray([value[1] for value in entries], dtype=np.float64)
        baseline_values = np.asarray([value[2] for value in entries], dtype=np.float64)
        scale = max(float(np.max(np.abs(targets))), 1.0)
        metrics[metric_name] = {
            "entry_count": int(len(entries)),
            "candidate_mae": float(np.mean(np.abs(candidate_values - targets))),
            "baseline_mae": float(np.mean(np.abs(baseline_values - targets))),
            "candidate_norm_mae": float(np.mean(np.abs(candidate_values - targets) / scale)),
            "baseline_norm_mae": float(np.mean(np.abs(baseline_values - targets) / scale)),
        }
    scored_metrics = [row for row in metrics.values() if int(row.get("entry_count") or 0) > 0]
    return {
        "metrics": metrics,
        "scored_metric_count": len(scored_metrics),
        "candidate_mean_norm_mae": None if not scored_metrics else float(np.mean([float(row["candidate_norm_mae"]) for row in scored_metrics])),
        "baseline_mean_norm_mae": None if not scored_metrics else float(np.mean([float(row["baseline_norm_mae"]) for row in scored_metrics])),
    }


def _evaluate_auxiliary_module(
    *,
    dataset: Any,
    candidate_result: dict[str, Any],
    carry_forward_result: dict[str, Any],
    module_name: str,
) -> dict[str, Any]:
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if module_name == "diagnosis_delay":
        train_simulation = _train_replay(dataset, candidate_result)
        train_rows = sorted(list(dataset.train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))[1:]
        holdout_trajectory = list(candidate_result.get("trajectory_rows") or [])
        holdout_predictions = list(candidate_result.get("prediction_rows") or [])
        train_features = []
        train_targets = []
        for row, feature_row in zip(train_rows, _cd4_features(list(train_simulation.get("trajectory_rows") or []), list(train_simulation.get("prediction_rows") or []))):
            target = row.get("median_cd4_at_enrollment")
            if target is None:
                continue
            train_features.append(feature_row)
            train_targets.append(float(target))
        holdout_features = _cd4_features(holdout_trajectory, holdout_predictions)
        fitted = _fit_linear_head(
            train_features=train_features,
            train_targets=train_targets,
            holdout_features=holdout_features,
        )
        if fitted is None:
            return {
                "module_name": module_name,
                "status": "not_identifiable",
                "reason": "insufficient_cd4_training_rows",
                "metrics": {},
            }
        candidate_series, diagnostics = fitted
        baseline_mean = float(np.mean(np.asarray(train_targets, dtype=np.float64)))
        return {
            "module_name": module_name,
            "status": "scored",
            "diagnostics": diagnostics,
            **_auxiliary_metric_entries(
                holdout_rows=holdout_rows,
                candidate_predictions={"median_cd4_at_enrollment": candidate_series},
                baseline_predictions={"median_cd4_at_enrollment": [baseline_mean for _ in holdout_rows]},
                metric_names=("median_cd4_at_enrollment",),
            ),
        }
    if module_name == "art_ltfu_vl":
        train_rows = sorted(list(dataset.train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
        testing_share = _mean_share(train_rows, "tested_for_viral_load", "alive_on_art")
        suppression_share = _mean_share(train_rows, "virally_suppressed", "alive_on_art")
        candidate_rows = list(candidate_result.get("prediction_rows") or [])
        carry_forward_rows = list(carry_forward_result.get("prediction_rows") or [])
        candidate_predictions = {
            "tested_for_viral_load": [row.get("tested_for_viral_load") for row in candidate_rows],
            "virally_suppressed": [row.get("virally_suppressed") for row in candidate_rows],
            "on_art_not_suppressed": [
                None
                if row.get("alive_on_art") is None or row.get("virally_suppressed") is None
                else max(float(row.get("alive_on_art") or 0.0) - float(row.get("virally_suppressed") or 0.0), 0.0)
                for row in candidate_rows
            ],
        }
        baseline_predictions = {
            "tested_for_viral_load": [
                None if testing_share is None else float(testing_share) * float(row.get("alive_on_art") or 0.0)
                for row in carry_forward_rows
            ],
            "virally_suppressed": [
                None if suppression_share is None else float(suppression_share) * float(row.get("alive_on_art") or 0.0)
                for row in carry_forward_rows
            ],
            "on_art_not_suppressed": [
                None
                if suppression_share is None
                else max(float(row.get("alive_on_art") or 0.0) - float(suppression_share) * float(row.get("alive_on_art") or 0.0), 0.0)
                for row in carry_forward_rows
            ],
        }
        summary = _auxiliary_metric_entries(
            holdout_rows=holdout_rows,
            candidate_predictions=candidate_predictions,
            baseline_predictions=baseline_predictions,
            metric_names=("tested_for_viral_load", "virally_suppressed", "on_art_not_suppressed"),
        )
        if int(summary.get("scored_metric_count") or 0) <= 0:
            return {
                "module_name": module_name,
                "status": "not_identifiable",
                "reason": "insufficient_vl_or_suppression_targets",
                "metrics": {},
            }
        return {
            "module_name": module_name,
            "status": "scored",
            "transition_diagnostics": {
                "mean_a_to_l_train_hazard": float(
                    np.mean(
                        np.asarray(
                            [
                                float((row.get("hazards") or {}).get("A_to_L") or 0.0)
                                for row in dataset.train_transition_rows
                            ],
                            dtype=np.float64,
                        )
                    )
                ) if dataset.train_transition_rows else 0.0,
                "mean_l_to_a_train_hazard": float(
                    np.mean(
                        np.asarray(
                            [
                                float((row.get("hazards") or {}).get("L_to_A") or 0.0)
                                for row in dataset.train_transition_rows
                            ],
                            dtype=np.float64,
                        )
                    )
                ) if dataset.train_transition_rows else 0.0,
            },
            **summary,
        }
    return {
        "module_name": module_name,
        "status": "not_applicable",
        "reason": "module_not_configured",
        "metrics": {},
    }


def _evaluate_split(
    *,
    observation_rows: list[dict[str, Any]],
    split: dict[str, Any],
    spec: dict[str, Any],
    module_name: str | None = None,
    structural_inputs: Any | None = None,
    direct_prior_features: Any | None = None,
    hidden_driver_features: Any | None = None,
) -> dict[str, Any]:
    dataset = build_blocked_time_dataset(observation_rows, list(split["holdout_years"]))
    carry_forward_incidence = None
    if spec.get("incidence_cfg") is not None:
        carry_forward_incidence = carry_forward_incidence_flow_paths(dataset, mode="last_train")
    carry_forward_result = simulate_holdout(
        dataset,
        carry_forward_hazards(dataset, mode="last_train"),
        incidence_inflow_map=None if carry_forward_incidence is None else dict(carry_forward_incidence.get("incidence_inflow_map") or {}),
        incidence_hazard_map=None if carry_forward_incidence is None else dict(carry_forward_incidence.get("incidence_hazard_map") or {}),
        population_denominator_map=None if carry_forward_incidence is None else dict(carry_forward_incidence.get("population_denominator_map") or {}),
        attrition_outflow_map=None if carry_forward_incidence is None else dict(carry_forward_incidence.get("attrition_outflow_map") or {}),
        state_attrition_outflow_map=None if carry_forward_incidence is None else dict(carry_forward_incidence.get("state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=None if carry_forward_incidence is None else dict(carry_forward_incidence.get("exit_channel_state_outflow_map") or {}),
    )
    candidate_result = forecast_dynamic_baseline(
        dataset,
        spec["dynamic_cfg"],
        incidence_cfg=spec.get("incidence_cfg"),
        observation_cfg=spec.get("observation_cfg"),
        shock_cfg=spec.get("shock_cfg"),
        damping_cfg=spec.get("damping_cfg"),
        structural_inputs=structural_inputs,
        direct_prior_features=direct_prior_features,
        prior_cfg=spec.get("prior_cfg"),
        hidden_driver_features=hidden_driver_features,
        hidden_cfg=spec.get("hidden_cfg"),
    )
    auxiliary_summary = None
    if module_name is not None:
        auxiliary_summary = _evaluate_auxiliary_module(
            dataset=dataset,
            candidate_result=candidate_result,
            carry_forward_result=carry_forward_result,
            module_name=module_name,
        )
    return {
        "train_end_year": int(split["train_end_year"]),
        "train_years": list(split["train_years"]),
        "holdout_years": list(split["holdout_years"]),
        "carry_forward": {
            "mae": float(carry_forward_result["mae"]),
            "smape": float(carry_forward_result["smape"]),
        },
        "candidate": {
            "mae": float(candidate_result["mae"]),
            "smape": float(candidate_result["smape"]),
        },
        "auxiliary_module": auxiliary_summary,
        "dataset_provenance": dict(dataset.provenance_summary),
        "scoring_details": {
            "metric_scales": {str(key): float(value) for key, value in dict(dataset.metric_scales).items()},
            "eps": float(dataset.eps),
            "holdout_rows": [dict(row) for row in dataset.holdout_rows],
            "candidate_prediction_rows": [dict(row) for row in list(candidate_result.get("prediction_rows") or [])],
            "carry_forward_prediction_rows": [dict(row) for row in list(carry_forward_result.get("prediction_rows") or [])],
        },
        "candidate_result": candidate_result,
    }


def _scored_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        scoring = dict(row.get("scoring_details") or {})
        metric_scales = {str(key): float(value) for key, value in dict(scoring.get("metric_scales") or {}).items()}
        holdout_rows = list(scoring.get("holdout_rows") or [])
        candidate_rows = list(scoring.get("candidate_prediction_rows") or [])
        carry_forward_rows = list(scoring.get("carry_forward_prediction_rows") or [])
        for target_row, candidate_row, carry_forward_row in zip(holdout_rows, candidate_rows, carry_forward_rows):
            metric_provenance = dict(target_row.get("metric_provenance") or {})
            for metric_name in PRIMARY_METRICS:
                target_value = target_row.get(metric_name)
                candidate_value = candidate_row.get(metric_name)
                carry_forward_value = carry_forward_row.get(metric_name)
                if target_value is None or candidate_value is None or carry_forward_value is None:
                    continue
                provenance = dict(metric_provenance.get(metric_name) or {})
                if str(provenance.get("observation_role") or "") != "direct_target":
                    continue
                scale = max(float(metric_scales.get(metric_name) or 1.0), 1.0)
                entries.append(
                    {
                        "quarter": str(target_row.get("quarter") or ""),
                        "metric_name": metric_name,
                        "candidate_abs_error": abs(float(candidate_value) - float(target_value)),
                        "carry_forward_abs_error": abs(float(carry_forward_value) - float(target_value)),
                        "candidate_norm_error": abs(float(candidate_value) - float(target_value)) / scale,
                        "carry_forward_norm_error": abs(float(carry_forward_value) - float(target_value)) / scale,
                        "support_partition": str(provenance.get("support_partition") or "common_support"),
                        "observation_role": str(provenance.get("observation_role") or "direct_target"),
                        "row_hash": str(provenance.get("row_hash") or ""),
                        "source_id": str(provenance.get("source_id") or ""),
                    }
                )
    return entries


def _support_partition_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for metric_name in sorted({str(entry["metric_name"]) for entry in entries}):
        metric_entries = [entry for entry in entries if str(entry["metric_name"]) == metric_name]
        for partition in sorted({str(entry["support_partition"]) for entry in metric_entries}):
            subset = [entry for entry in metric_entries if str(entry["support_partition"]) == partition]
            rows.append(
                {
                    "metric_name": metric_name,
                    "support_partition": partition,
                    "entry_count": int(len(subset)),
                    "candidate_norm_mae": float(np.mean([float(entry["candidate_norm_error"]) for entry in subset])),
                    "carry_forward_norm_mae": float(np.mean([float(entry["carry_forward_norm_error"]) for entry in subset])),
                }
            )
    return {"rows": rows}


def _residual_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {
            "overall_candidate_p90": None,
            "overall_carry_forward_p90": None,
            "metric_rows": [],
        }
    metric_rows: list[dict[str, Any]] = []
    for metric_name in sorted({str(entry["metric_name"]) for entry in entries}):
        subset = [entry for entry in entries if str(entry["metric_name"]) == metric_name]
        metric_rows.append(
            {
                "metric_name": metric_name,
                "candidate_p90": float(np.percentile(np.asarray([float(entry["candidate_abs_error"]) for entry in subset], dtype=np.float64), 90)),
                "carry_forward_p90": float(np.percentile(np.asarray([float(entry["carry_forward_abs_error"]) for entry in subset], dtype=np.float64), 90)),
            }
        )
    return {
        "overall_candidate_p90": float(np.percentile(np.asarray([float(entry["candidate_abs_error"]) for entry in entries], dtype=np.float64), 90)),
        "overall_carry_forward_p90": float(np.percentile(np.asarray([float(entry["carry_forward_abs_error"]) for entry in entries], dtype=np.float64), 90)),
        "metric_rows": metric_rows,
    }


def _raw_endpoint_error_report(entries: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    for metric_name in sorted({str(entry["metric_name"]) for entry in entries}):
        subset = [entry for entry in entries if str(entry["metric_name"]) == metric_name]
        metric_rows.append(
            {
                "metric_name": metric_name,
                "entry_count": int(len(subset)),
                "candidate_mean_abs_error": float(
                    np.mean(np.asarray([float(entry["candidate_abs_error"]) for entry in subset], dtype=np.float64))
                ),
                "carry_forward_mean_abs_error": float(
                    np.mean(np.asarray([float(entry["carry_forward_abs_error"]) for entry in subset], dtype=np.float64))
                ),
            }
        )
    return {
        "metric_rows": metric_rows,
        "candidate_mean_abs_error": None if not entries else float(np.mean(np.asarray([float(entry["candidate_abs_error"]) for entry in entries], dtype=np.float64))),
        "carry_forward_mean_abs_error": None if not entries else float(np.mean(np.asarray([float(entry["carry_forward_abs_error"]) for entry in entries], dtype=np.float64))),
    }


def _archive_drift_gate(
    *,
    score: dict[str, float],
    benchmark_comparison: dict[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name in ("frozen_exact_r10", "dense_r10_readout", "inflow_diagnostic"):
        row = dict(benchmark_comparison.get(name) or {})
        mean_mae = row.get("candidate_mean_mae")
        worst_mae = row.get("candidate_worst_mae")
        if mean_mae is None or not np.isfinite(float(mean_mae)):
            continue
        comparisons[name] = {
            "reference_mean_mae": float(mean_mae),
            "reference_worst_mae": None if worst_mae is None or not np.isfinite(float(worst_mae)) else float(worst_mae),
            "candidate_mean_mae_delta": float(score["candidate_mean_mae"]) - float(mean_mae),
            "candidate_worst_mae_delta": None if worst_mae is None or not np.isfinite(float(worst_mae)) else float(score["candidate_worst_mae"]) - float(worst_mae),
        }
    return {
        "status": "emitted",
        "comparisons": comparisons,
    }


def _support_tier_stability_gate(support_partition: dict[str, Any]) -> dict[str, Any]:
    rows = list(support_partition.get("rows") or [])
    common_rows = [row for row in rows if str(row.get("support_partition")) == "common_support"]
    expanded_rows = [row for row in rows if str(row.get("support_partition")) == "expanded_support"]
    return {
        "status": "emitted",
        "common_support_pass": (
            all(float(row["candidate_norm_mae"]) <= float(row["carry_forward_norm_mae"]) for row in common_rows)
            if common_rows
            else None
        ),
        "expanded_support_pass": (
            all(float(row["candidate_norm_mae"]) <= float(row["carry_forward_norm_mae"]) for row in expanded_rows)
            if expanded_rows
            else None
        ),
        "common_support_entry_count": int(sum(int(row.get("entry_count") or 0) for row in common_rows)),
        "expanded_support_entry_count": int(sum(int(row.get("entry_count") or 0) for row in expanded_rows)),
    }


def _lockbox_contract() -> dict[str, Any]:
    return {
        "status": "deferred_until_final_publication_lockbox",
        "allowed_use": "development_and_model_screening_only",
        "disallowed_use": "publication_claim_without_external_lockbox",
    }


def _decision_reason(*, keep: bool, identifiability_status: str) -> str:
    if identifiability_status != "identifiable":
        return "Stage evidence is not identifiable under the active observation contract."
    if keep:
        return "Blocked-time primary, support-partition, and residual diagnostics remain compatible with carry-forward."
    return "Blocked-time carry-forward comparison failed on at least one primary forecast gate."


def _load_reference_benchmarks(epigraph_root: Path, sandbox_root: Path) -> dict[str, Any]:
    benchmarks: dict[str, Any] = {}
    r10_path = epigraph_root / "artifacts" / "runs" / "tr-v3-current-champion-r10-neighborhood-20260419-s00" / "analysis" / "tr_v3_current_champion_r10_neighborhood_batch_report.json"
    if r10_path.exists():
        payload = json.loads(r10_path.read_text(encoding="utf-8"))
        for contract in list(payload.get("contracts") or []):
            contract_name = str(contract.get("contract") or "")
            winner = dict(contract.get("merged_winner") or {})
            if contract_name == "exact_only":
                benchmarks["frozen_exact_r10"] = {
                    "source_report": r10_path.as_posix(),
                    "candidate_mean_mae": float(winner.get("quarterly_mean_mae") or float("nan")),
                    "candidate_worst_mae": float(winner.get("quarterly_worst_mae") or float("nan")),
                }
            if contract_name == "purged_dense":
                benchmarks["dense_r10_readout"] = {
                    "source_report": r10_path.as_posix(),
                    "candidate_mean_mae": float(winner.get("quarterly_mean_mae") or float("nan")),
                    "candidate_worst_mae": float(winner.get("quarterly_worst_mae") or float("nan")),
                }
    inflow_path = sandbox_root / "artifacts" / "runs" / "tr-v3-05a-inflow-20260424" / "analysis" / "tr_v3_05a_inflow_report.json"
    if inflow_path.exists():
        payload = json.loads(inflow_path.read_text(encoding="utf-8"))
        score = dict((payload.get("best_candidate") or {}).get("score") or {})
        benchmarks["inflow_diagnostic"] = {
            "source_report": inflow_path.as_posix(),
            "candidate_mean_mae": float(score.get("candidate_mean_mae") or float("nan")),
            "candidate_worst_mae": float(score.get("candidate_worst_mae") or float("nan")),
        }
    return benchmarks


def _claim_card(
    *,
    run_id: str,
    revision_stage: str,
    source_run_id: str,
    baseline_source_run_id: str,
    score: dict[str, float],
    support_partition: dict[str, Any],
    residual_summary: dict[str, Any],
    reference_benchmarks: dict[str, Any],
    decision: str,
    auxiliary_summary: dict[str, Any] | None = None,
    identifiability_status: str = "identifiable",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    score_gate_applicable = str(decision) in {"keep", "revert"}
    support_rows = list(support_partition.get("rows") or [])
    common_rows = [row for row in support_rows if str(row.get("support_partition")) == "common_support"]
    expanded_rows = [row for row in support_rows if str(row.get("support_partition")) == "expanded_support"]
    support_blockers: list[str] = []
    if score_gate_applicable and common_rows and any(float(row["candidate_norm_mae"]) > float(row["carry_forward_norm_mae"]) for row in common_rows):
        support_blockers.append("fails_common_support_gate")
    if score_gate_applicable and expanded_rows and any(float(row["candidate_norm_mae"]) > float(row["carry_forward_norm_mae"]) for row in expanded_rows):
        support_blockers.append("fails_expanded_support_gate")
    residual_blockers: list[str] = []
    candidate_p90 = residual_summary.get("overall_candidate_p90")
    carry_forward_p90 = residual_summary.get("overall_carry_forward_p90")
    if score_gate_applicable and candidate_p90 is not None and carry_forward_p90 is not None and float(candidate_p90) > float(carry_forward_p90):
        residual_blockers.append("fails_residual_p90_gate")
    primary_blockers: list[str] = []
    if score_gate_applicable and float(score["candidate_mean_mae"]) >= float(score["carry_forward_mean_mae"]):
        primary_blockers.append("fails_carry_forward_mean_gate")
    if score_gate_applicable and float(score["candidate_worst_mae"]) > float(score["carry_forward_worst_mae"]):
        primary_blockers.append("fails_carry_forward_worst_gate")
    auxiliary_blockers: list[str] = []
    if auxiliary_summary is not None and auxiliary_summary.get("status") == "scored":
        candidate_aux = auxiliary_summary.get("candidate_mean_norm_mae")
        baseline_aux = auxiliary_summary.get("baseline_mean_norm_mae")
        if candidate_aux is not None and baseline_aux is not None and float(candidate_aux) > float(baseline_aux):
            auxiliary_blockers.append("fails_auxiliary_module_gate")
    if identifiability_status != "identifiable":
        auxiliary_blockers.append("module_not_identifiable")
    if not score_gate_applicable:
        auxiliary_blockers.append("non_model_or_diagnostic_stage")
    blockers = primary_blockers + support_blockers + residual_blockers + auxiliary_blockers
    promotion_eligible = decision == "keep" and not blockers
    return {
        "schema_version": CLAIM_CARD_SCHEMA_VERSION,
        "run_id": run_id,
        "revision_stage": revision_stage,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "decision": decision,
        "identifiability_status": identifiability_status,
        "promotion_eligible": bool(promotion_eligible),
        "blockers": blockers,
        "warnings": list(warnings or []),
        "primary_gate": {
            "candidate_mean_mae": float(score["candidate_mean_mae"]),
            "carry_forward_mean_mae": float(score["carry_forward_mean_mae"]),
            "candidate_worst_mae": float(score["candidate_worst_mae"]),
            "carry_forward_worst_mae": float(score["carry_forward_worst_mae"]),
        },
        "support_partition_gate": support_partition,
        "residual_gate": residual_summary,
        "reference_benchmarks": reference_benchmarks,
        "auxiliary_module": auxiliary_summary,
        "allowed_claims": [
            "blocked_time_development_benchmark",
            "typed_observation_role_contract",
            "support_partition_validation",
        ],
        "disallowed_claims": [
            "validated_policy_effect",
            "validated_provincial_truth",
            "validated_incidence_truth",
        ],
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    claim_card = dict(payload.get("claim_card") or {})
    auxiliary = dict(payload.get("auxiliary_module_summary") or {})
    lines = [
        f"# {payload['revision_stage']} Report",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Baseline run: `{payload['baseline_source_run_id']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Promotion eligible: `{claim_card.get('promotion_eligible')}`",
        "",
        "## Primary Benchmark",
        "",
        f"- Candidate mean MAE: `{payload['best_candidate']['score']['candidate_mean_mae']:.6f}`",
        f"- Carry-forward mean MAE: `{payload['best_candidate']['score']['carry_forward_mean_mae']:.6f}`",
        f"- Candidate worst MAE: `{payload['best_candidate']['score']['candidate_worst_mae']:.6f}`",
        f"- Carry-forward worst MAE: `{payload['best_candidate']['score']['carry_forward_worst_mae']:.6f}`",
        "",
        "## External Benchmarks",
        "",
        "| Baseline | Mean MAE | Worst MAE |",
        "|---|---:|---:|",
    ]
    for name, row in sorted(dict(payload.get("benchmark_comparison") or {}).items()):
        lines.append(
            f"| `{name}` | {float(row.get('candidate_mean_mae') or float('nan')):.6f} | {float(row.get('candidate_worst_mae') or float('nan')):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Support Partition",
            "",
            "| Metric | Partition | Entries | Candidate Norm MAE | Carry-forward Norm MAE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in list((payload.get("support_partition_table") or {}).get("rows") or []):
        lines.append(
            f"| `{row['metric_name']}` | `{row['support_partition']}` | {int(row['entry_count'])} | "
            f"{float(row['candidate_norm_mae']):.6f} | {float(row['carry_forward_norm_mae']):.6f} |"
        )
    if auxiliary:
        lines.extend(
            [
                "",
                "## Auxiliary Module",
                "",
                f"- Module: `{auxiliary.get('module_name')}`",
                f"- Status: `{auxiliary.get('status')}`",
            ]
        )
        if auxiliary.get("reason"):
            lines.append(f"- Reason: `{auxiliary.get('reason')}`")
        metrics = dict(auxiliary.get("metrics") or {})
        if metrics:
            lines.extend(
                [
                    "",
                    "| Metric | Entries | Candidate Norm MAE | Baseline Norm MAE |",
                    "|---|---:|---:|---:|",
                ]
            )
            for metric_name, row in sorted(metrics.items()):
                if int(row.get("entry_count") or 0) <= 0:
                    continue
                lines.append(
                    f"| `{metric_name}` | {int(row['entry_count'])} | {float(row['candidate_norm_mae']):.6f} | {float(row['baseline_norm_mae']):.6f} |"
                )
    lines.extend(
        [
            "",
            "## Claim Card",
            "",
            f"- Identifiability: `{claim_card.get('identifiability_status')}`",
            f"- Blockers: `{', '.join(str(value) for value in claim_card.get('blockers') or []) or 'none'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _svg_dashboard(
    *,
    benchmark_comparison: dict[str, Any],
    support_partition: dict[str, Any],
    output_path: Path,
) -> None:
    panel_width = 520
    panel_height = 240
    width = 1100
    height = 560
    bars = [("candidate", None, None)]
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="40" font-size="24" font-family="monospace" fill="#111111">REV Benchmark Dashboard</text>',
        '<text x="40" y="68" font-size="13" font-family="monospace" fill="#555555">Mean MAE and support-partition normalized MAE</text>',
    ]
    benchmark_rows = []
    for name, row in sorted(benchmark_comparison.items()):
        benchmark_rows.append((name, float(row.get("candidate_mean_mae") or 0.0)))
    if benchmark_rows:
        max_value = max(value for _name, value in benchmark_rows) or 1.0
        x0 = 60
        y0 = 270
        bar_width = 90
        gap = 20
        lines.extend(
            [
                '<text x="60" y="105" font-size="16" font-family="monospace" fill="#111111">Benchmark Mean MAE</text>',
                '<line x1="60" y1="260" x2="520" y2="260" stroke="#333333" stroke-width="1"/>',
            ]
        )
        for index, (name, value) in enumerate(benchmark_rows):
            height_value = 150.0 * (value / max_value if max_value > 0.0 else 0.0)
            x = x0 + index * (bar_width + gap)
            y = y0 - height_value
            lines.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{height_value:.2f}" fill="#3b82f6"/>')
            lines.append(f'<text x="{x}" y="{y - 8:.2f}" font-size="12" font-family="monospace" fill="#111111">{value:.3f}</text>')
            lines.append(f'<text x="{x}" y="{y0 + 20}" font-size="11" font-family="monospace" fill="#111111">{name}</text>')
    support_rows = list((support_partition.get("rows") or []))[:8]
    if support_rows:
        max_value = max(max(float(row["candidate_norm_mae"]), float(row["carry_forward_norm_mae"])) for row in support_rows) or 1.0
        x0 = 600
        y0 = 270
        group_gap = 18
        bar_width = 24
        lines.extend(
            [
                '<text x="600" y="105" font-size="16" font-family="monospace" fill="#111111">Support Partition Norm MAE</text>',
                '<line x1="600" y1="260" x2="1040" y2="260" stroke="#333333" stroke-width="1"/>',
            ]
        )
        for index, row in enumerate(support_rows):
            base_x = x0 + index * (bar_width * 2 + group_gap)
            candidate_height = 150.0 * (float(row["candidate_norm_mae"]) / max_value if max_value > 0.0 else 0.0)
            baseline_height = 150.0 * (float(row["carry_forward_norm_mae"]) / max_value if max_value > 0.0 else 0.0)
            lines.append(f'<rect x="{base_x}" y="{y0 - candidate_height:.2f}" width="{bar_width}" height="{candidate_height:.2f}" fill="#ef4444"/>')
            lines.append(f'<rect x="{base_x + bar_width}" y="{y0 - baseline_height:.2f}" width="{bar_width}" height="{baseline_height:.2f}" fill="#10b981"/>')
            label = f"{row['metric_name']}:{row['support_partition']}"
            lines.append(f'<text x="{base_x}" y="{y0 + 20}" font-size="9" font-family="monospace" fill="#111111">{label}</text>')
        lines.append('<text x="600" y="300" font-size="11" font-family="monospace" fill="#ef4444">candidate</text>')
        lines.append('<text x="700" y="300" font-size="11" font-family="monospace" fill="#10b981">carry-forward</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_repro_manifest(analysis_dir: Path, *, run_id: str, source_run_id: str) -> Path:
    files = sorted([path for path in analysis_dir.iterdir() if path.is_file()])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": RUN_REPRO_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "source_run_id": source_run_id,
        "files": [
            {
                "path": path.as_posix(),
                "name": path.name,
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in files
        ],
    }
    target_path = analysis_dir / "reproducibility_manifest.json"
    write_json(target_path, payload)
    return target_path


def _finalize_run_artifacts(
    *,
    analysis_dir: Path,
    report_stem: str,
    payload: dict[str, Any],
    epigraph_root: Path,
) -> dict[str, Path]:
    report_json_path = analysis_dir / f"{report_stem}.json"
    report_md_path = analysis_dir / f"{report_stem}.md"
    ledger_path = analysis_dir / "observation_role_ledger_snapshot.json"
    claim_card_json_path = analysis_dir / "claim_card.json"
    claim_card_md_path = analysis_dir / "claim_card.md"
    support_path = analysis_dir / "support_partition_table.json"
    benchmark_path = analysis_dir / "benchmark_comparison.json"
    residual_path = analysis_dir / "residual_summary.json"
    dashboard_path = analysis_dir / "residual_dashboard.svg"
    failure_json_path = analysis_dir / "failure_anatomy.json"
    failure_md_path = analysis_dir / "failure_anatomy.md"
    artifact_paths = {
        "report_json": report_json_path,
        "report_markdown": report_md_path,
        "observation_ledger_snapshot": ledger_path,
        "claim_card_json": claim_card_json_path,
        "claim_card_markdown": claim_card_md_path,
        "support_partition_table": support_path,
        "benchmark_comparison": benchmark_path,
        "residual_summary": residual_path,
        "dashboard": dashboard_path,
        "failure_anatomy_json": failure_json_path,
        "failure_anatomy_markdown": failure_md_path,
    }
    payload["artifact_paths"] = {key: path.as_posix() for key, path in artifact_paths.items()}
    write_json(report_json_path, payload)
    _write_markdown(report_md_path, _markdown_report(payload))
    write_json(ledger_path, payload["observation_role_ledger"])
    write_json(claim_card_json_path, payload["claim_card"])
    _write_markdown(claim_card_md_path, json.dumps(payload["claim_card"], indent=2, sort_keys=True, default=_json_default))
    write_json(support_path, payload["support_partition_table"])
    write_json(benchmark_path, payload["benchmark_comparison"])
    write_json(residual_path, payload["residual_summary"])
    _svg_dashboard(
        benchmark_comparison=payload["benchmark_comparison"],
        support_partition=payload["support_partition_table"],
        output_path=dashboard_path,
    )
    failure_payload = build_failure_anatomy_payload(
        epigraph_root=epigraph_root,
        report_paths=[report_json_path],
        top_years=3,
    )
    write_json(failure_json_path, failure_payload)
    _write_markdown(failure_md_path, markdown_failure_anatomy_report(failure_payload))
    repro_path = _write_repro_manifest(
        analysis_dir,
        run_id=str(payload["run_id"]),
        source_run_id=str(payload["source_run_id"]),
    )
    artifact_paths["reproducibility_manifest"] = repro_path
    payload["artifact_paths"] = {key: path.as_posix() for key, path in artifact_paths.items()}
    write_json(report_json_path, payload)
    return artifact_paths


def _base_dynamic_specs() -> list[dict[str, Any]]:
    dynamic_cfgs = [
        DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0),
        DynamicBaselineConfig(ridge_penalty=0.1, rho_clip=0.9, trend_scale=1.0),
    ]
    observation_cfgs = [
        ObservationModelConfig(calibration_ridge=0.01, share_ridge_penalty=0.01, share_rho_clip=0.8, share_trend_scale=0.5),
        ObservationModelConfig(calibration_ridge=0.1, share_ridge_penalty=0.1, share_rho_clip=0.95, share_trend_scale=1.0),
    ]
    shock_cfgs = [
        None,
        ShockConfig(shock_phi=0.8, shock_scale=1.5, gate_z=1.5),
    ]
    damping_cfgs = [
        DampingConfig(min_dynamic_weight=0.15, gain_weight=0.5, residual_weight=0.15, horizon_decay=0.4, calibration_floor=0.2, calibration_gain_weight=0.7),
        DampingConfig(min_dynamic_weight=0.25, gain_weight=0.75, residual_weight=0.15, horizon_decay=0.6, calibration_floor=0.35, calibration_gain_weight=0.7),
    ]
    specs: list[dict[str, Any]] = []
    for dynamic_cfg in dynamic_cfgs:
        for observation_cfg in observation_cfgs:
            for shock_cfg in shock_cfgs:
                for damping_cfg in damping_cfgs:
                    specs.append(
                        {
                            "dynamic_cfg": dynamic_cfg,
                            "observation_cfg": observation_cfg,
                            "shock_cfg": shock_cfg,
                            "damping_cfg": damping_cfg,
                        }
                    )
    return specs


def _incidence_specs(base_spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for incidence_cfg in (
        IncidenceFlowConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=0.5),
        IncidenceFlowConfig(ridge_penalty=0.1, rho_clip=0.9, trend_scale=1.0),
        IncidenceFlowConfig(ridge_penalty=0.1, rho_clip=0.8, trend_scale=0.5),
        IncidenceFlowConfig(ridge_penalty=0.01, rho_clip=0.9, trend_scale=1.0),
    ):
        spec = dict(base_spec)
        spec["incidence_cfg"] = incidence_cfg
        rows.append(spec)
    return rows


def _phase2_specs(base_spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direct_cfgs = [
        DirectPriorConfig(effect_scale=1.0, precision_scale=1.0, residual_ridge=0.1, max_effect=0.2),
        DirectPriorConfig(effect_scale=1.5, precision_scale=2.0, residual_ridge=0.1, max_effect=0.35),
    ]
    hidden_cfgs = [
        HiddenDriverConfig(precision_scale=0.25, residual_ridge=0.01, max_effect=0.1, rank_cap=1),
        HiddenDriverConfig(
            precision_scale=0.25,
            residual_ridge=0.01,
            max_effect=0.1,
            rank_cap=1,
            min_effect_weight=1.0,
            gain_weight=0.0,
            shock_penalty_weight=0.0,
            horizon_decay=1.0,
            gate_gain_weight=2.0,
            gate_recent_penalty=0.25,
            gate_threshold=0.05,
            blocked_weight=0.15,
        ),
    ]
    for prior_cfg in direct_cfgs:
        spec = dict(base_spec)
        spec["prior_cfg"] = prior_cfg
        rows.append(spec)
        for hidden_cfg in hidden_cfgs:
            richer = dict(spec)
            richer["hidden_cfg"] = hidden_cfg
            rows.append(richer)
    return rows


def _resolve_phase2_source_for_revision(
    epigraph_root: Path,
    *,
    observation_source_run_id: str,
    preferred: str | None,
) -> tuple[str, dict[str, Any]]:
    if preferred:
        return resolve_phase2_structural_source_run_id(
            epigraph_root,
            observation_source_run_id=observation_source_run_id,
            preferred=preferred,
        )
    candidates: list[dict[str, Any]] = []
    for path in sorted((epigraph_root / "artifacts" / "runs").glob("*/phase2/phase2_structural_payload.json")):
        run_id = path.parent.parent.name
        try:
            structural_inputs = load_phase2_structural_inputs(epigraph_root, run_id)
            direct_prior_features = build_direct_prior_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
            hidden_driver_features = build_hidden_driver_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
        except Exception as exc:
            candidates.append(
                {
                    "run_id": run_id,
                    "usable": False,
                    "error": type(exc).__name__,
                    "direct_prior_feature_count": 0,
                    "hidden_driver_feature_count": 0,
                    "direct_edge_count": 0,
                    "hidden_edge_count": 0,
                }
            )
            continue
        direct_count = int(sum(len(rows) for rows in direct_prior_features.values()))
        hidden_count = int(sum(len(rows) for rows in hidden_driver_features.values()))
        candidates.append(
            {
                "run_id": run_id,
                "usable": direct_count + hidden_count > 0,
                "direct_prior_feature_count": direct_count,
                "hidden_driver_feature_count": hidden_count,
                "direct_edge_count": len(structural_inputs.direct_edge_rows),
                "hidden_edge_count": len(structural_inputs.hidden_driver_rows),
            }
        )
    usable = [row for row in candidates if bool(row.get("usable"))]
    if usable:
        selected = max(
            usable,
            key=lambda row: (
                int(row["direct_prior_feature_count"]) + int(row["hidden_driver_feature_count"]),
                int(row["hidden_driver_feature_count"]),
                int(row["direct_prior_feature_count"]),
                str(row["run_id"]),
            ),
        )
        return str(selected["run_id"]), {
            "resolution": "feature_aware_max_admissible_phase3_features",
            "candidate_count": len(candidates),
            "usable_candidate_count": len(usable),
            "selected_direct_prior_feature_count": int(selected["direct_prior_feature_count"]),
            "selected_hidden_driver_feature_count": int(selected["hidden_driver_feature_count"]),
        }
    fallback_run_id, fallback = resolve_phase2_structural_source_run_id(
        epigraph_root,
        observation_source_run_id=observation_source_run_id,
        preferred=None,
    )
    return fallback_run_id, {
        **fallback,
        "resolution": "fallback_no_admissible_phase3_features",
        "candidate_count": len(candidates),
        "usable_candidate_count": 0,
    }


def _best_candidate(
    *,
    observation_rows: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    candidate_specs: list[dict[str, Any]],
    module_name: str | None,
    structural_inputs: Any | None = None,
    direct_prior_features: Any | None = None,
    hidden_driver_features: Any | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for index, spec in enumerate(candidate_specs):
        rows = [
            _evaluate_split(
                observation_rows=observation_rows,
                split=split,
                spec=spec,
                module_name=module_name,
                structural_inputs=structural_inputs,
                direct_prior_features=direct_prior_features if spec.get("prior_cfg") is not None else None,
                hidden_driver_features=hidden_driver_features if spec.get("hidden_cfg") is not None else None,
            )
            for split in splits
        ]
        stored_rows = []
        for row in rows:
            stored = dict(row)
            stored.pop("candidate_result", None)
            stored_rows.append(stored)
        auxiliary_rows = [
            dict(row.get("auxiliary_module") or {})
            for row in stored_rows
            if isinstance(row.get("auxiliary_module"), dict) and str((row.get("auxiliary_module") or {}).get("status") or "") == "scored"
        ]
        auxiliary_candidate_score = None
        if auxiliary_rows:
            auxiliary_candidate_score = float(
                np.mean(
                    np.asarray(
                        [float(row.get("candidate_mean_norm_mae") or 0.0) for row in auxiliary_rows],
                        dtype=np.float64,
                    )
                )
            )
        candidates.append(
            {
                "candidate_index": int(index),
                "config": _spec_payload(spec),
                "model_contract": _candidate_model_contract(
                    spec,
                    direct_prior_features=direct_prior_features if spec.get("prior_cfg") is not None else None,
                    hidden_driver_features=hidden_driver_features if spec.get("hidden_cfg") is not None else None,
                ),
                "hazard_semantics": _candidate_hazard_semantics(
                    spec,
                    direct_prior_features=direct_prior_features if spec.get("prior_cfg") is not None else None,
                    hidden_driver_features=hidden_driver_features if spec.get("hidden_cfg") is not None else None,
                ),
                "rows": stored_rows,
                "score": _score_rows(stored_rows),
                "auxiliary_candidate_score": auxiliary_candidate_score,
            }
        )
    best_candidate = min(
        candidates,
        key=lambda row: (
            float(row["score"]["candidate_mean_mae"]),
            float("inf") if row["auxiliary_candidate_score"] is None else float(row["auxiliary_candidate_score"]),
            float(row["score"]["candidate_worst_mae"]),
        ),
    )
    return {
        "spec": dict(candidate_specs[int(best_candidate["candidate_index"])]),
        "candidate": best_candidate,
        "all_candidates": candidates,
    }


def _build_stage_payload(
    *,
    revision_stage: str,
    run_id: str,
    source_run_id: str,
    baseline_source_run_id: str,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
    observation_rows: list[dict[str, Any]],
    observation_role_ledger: dict[str, Any],
    best_candidate: dict[str, Any],
    all_candidates: list[dict[str, Any]],
    module_name: str | None,
    epigraph_root: Path,
    phase2_context: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = list(best_candidate["rows"])
    score = dict(best_candidate["score"])
    entries = _scored_entries(rows)
    support_partition = _support_partition_summary(entries)
    residual_summary = _residual_summary(entries)
    raw_endpoint_errors = _raw_endpoint_error_report(entries)
    benchmark_comparison = _load_reference_benchmarks(epigraph_root, sandbox_repo_root())
    benchmark_comparison = {
        "candidate": {
            "candidate_mean_mae": float(score["candidate_mean_mae"]),
            "candidate_worst_mae": float(score["candidate_worst_mae"]),
        },
        "carry_forward": {
            "candidate_mean_mae": float(score["carry_forward_mean_mae"]),
            "candidate_worst_mae": float(score["carry_forward_worst_mae"]),
        },
        **benchmark_comparison,
    }
    auxiliary_rows = [
        dict(row.get("auxiliary_module") or {})
        for row in rows
        if isinstance(row.get("auxiliary_module"), dict)
    ]
    identifiability_status = "identifiable"
    auxiliary_summary = None
    if module_name is not None:
        scored_rows = [row for row in auxiliary_rows if str(row.get("status") or "") == "scored"]
        if scored_rows:
            metric_payload: dict[str, Any] = {}
            for metric_name in sorted(
                {
                    str(metric_name)
                    for row in scored_rows
                    for metric_name, metric_payload_row in dict(row.get("metrics") or {}).items()
                    if int((metric_payload_row or {}).get("entry_count") or 0) > 0
                }
            ):
                metric_rows = [
                    dict((row.get("metrics") or {}).get(metric_name) or {})
                    for row in scored_rows
                    if int(((row.get("metrics") or {}).get(metric_name) or {}).get("entry_count") or 0) > 0
                ]
                metric_payload[metric_name] = {
                    "entry_count": int(sum(int(row.get("entry_count") or 0) for row in metric_rows)),
                    "candidate_norm_mae": float(np.mean(np.asarray([float(row["candidate_norm_mae"]) for row in metric_rows], dtype=np.float64))),
                    "baseline_norm_mae": float(np.mean(np.asarray([float(row["baseline_norm_mae"]) for row in metric_rows], dtype=np.float64))),
                }
            auxiliary_summary = {
                "module_name": module_name,
                "status": "scored",
                "metrics": metric_payload,
                "candidate_mean_norm_mae": float(np.mean(np.asarray([float(row["candidate_mean_norm_mae"]) for row in scored_rows], dtype=np.float64))),
                "baseline_mean_norm_mae": float(np.mean(np.asarray([float(row["baseline_mean_norm_mae"]) for row in scored_rows], dtype=np.float64))),
            }
        else:
            identifiability_status = "not_identifiable"
            auxiliary_summary = auxiliary_rows[0] if auxiliary_rows else {"module_name": module_name, "status": "not_identifiable"}
    keep = float(score["candidate_mean_mae"]) < float(score["carry_forward_mean_mae"]) and float(score["candidate_worst_mae"]) <= float(score["carry_forward_worst_mae"])
    decision = "keep" if keep and identifiability_status == "identifiable" else ("not_identifiable" if identifiability_status != "identifiable" else "revert")
    observation_score_ledger = _build_observation_score_ledger({"rows": rows})
    model_contract = dict(best_candidate.get("model_contract") or {})
    hazard_semantics = dict(best_candidate.get("hazard_semantics") or {})
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "family_name": revision_stage,
        "revision_stage": revision_stage,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "loop_variant": "evidence-to-model-loop",
        "decision": decision,
        "decision_reason": _decision_reason(keep=keep, identifiability_status=identifiability_status),
        "module_name": module_name,
        "benchmark_contract": {
            "start_year": int(start_year),
            "end_year": int(end_year),
            "min_train_years": int(min_train_years),
            "horizon_years": int(horizon_years),
        },
        "model_contract": model_contract,
        "hazard_semantics": hazard_semantics,
        "phase2_prior_contract": None if phase2_context is None else dict(PHASE2_PRIOR_CONTRACT),
        "phase2_context": phase2_context,
        "data_provenance": _build_data_provenance_payload(observation_rows, {"rows": rows}),
        "observation_role_ledger": observation_role_ledger,
        "observation_score_ledger": list(observation_score_ledger["entries"]),
        "observation_score_ledger_summary": dict(observation_score_ledger["summary"]),
        "lockbox_contract": _lockbox_contract(),
        "best_candidate": {
            "config": dict(best_candidate["config"]),
            "model_contract": model_contract,
            "hazard_semantics": hazard_semantics,
            "score": score,
            "rows": rows,
        },
        "all_candidates": all_candidates,
        "support_partition_table": support_partition,
        "residual_summary": residual_summary,
        "raw_endpoint_error_report": raw_endpoint_errors,
        "raw_endpoint_errors": raw_endpoint_errors,
        "archive_drift_gate": _archive_drift_gate(score=score, benchmark_comparison=benchmark_comparison),
        "support_tier_stability_gate": _support_tier_stability_gate(support_partition),
        "benchmark_comparison": benchmark_comparison,
        "auxiliary_module_summary": auxiliary_summary,
    }
    payload["claim_card"] = _claim_card(
        run_id=run_id,
        revision_stage=revision_stage,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        score=score,
        support_partition=support_partition,
        residual_summary=residual_summary,
        reference_benchmarks=benchmark_comparison,
        decision=decision,
        auxiliary_summary=auxiliary_summary,
        identifiability_status=identifiability_status,
    )
    return payload


def _run_revision_stage(
    *,
    revision_stage: str,
    run_id: str,
    source_run_id: str,
    baseline_source_run_id: str,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
    candidate_specs: list[dict[str, Any]],
    module_name: str | None = None,
    use_phase2: bool = False,
    phase2_source_run_id: str | None = None,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    observation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    ledger = build_observation_role_ledger(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    splits = rolling_origin_splits(
        observation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    structural_inputs = None
    direct_prior_features = None
    hidden_driver_features = None
    phase2_context = None
    if use_phase2:
        resolved_phase2_source_run_id, phase2_resolution = _resolve_phase2_source_for_revision(
            epigraph_root,
            observation_source_run_id=source_run_id,
            preferred=phase2_source_run_id,
        )
        structural_inputs = load_phase2_structural_inputs(epigraph_root, resolved_phase2_source_run_id)
        raw_direct_prior_features = build_direct_prior_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
        robustness_report, robustness_resolution = resolve_phase2_determinant_robustness_report(
            epigraph_root,
            resolved_phase2_source_run_id,
        )
        allowed_direct_edge_keys = allowed_direct_edge_keys_from_robustness(robustness_report)
        direct_prior_features = filter_direct_prior_features_by_edge_keys(
            raw_direct_prior_features,
            allowed_direct_edge_keys,
        )
        hidden_driver_features = build_hidden_driver_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
        phase2_context = {
            "phase2_source_run_id": resolved_phase2_source_run_id,
            "phase2_source_resolution": phase2_resolution,
            "determinant_robustness_resolution": robustness_resolution,
            "determinant_robustness_contract": {
                "mode": "fail_closed_strict_source_stable_edges_only",
                "direct_terms_before_robustness_gate": direct_prior_feature_count(raw_direct_prior_features),
                "direct_terms_after_robustness_gate": direct_prior_feature_count(direct_prior_features),
                "allowed_direct_edge_keys": sorted(allowed_direct_edge_keys),
                "missing_report_policy": "zero_direct_phase2_terms",
            },
            "direct_edge_count": len(structural_inputs.direct_edge_rows),
            "hidden_edge_count": len(structural_inputs.hidden_driver_rows),
            "multiscale_support_count": len(structural_inputs.multiscale_support_rows),
            "quarter_count": len(structural_inputs.quarter_axis),
            "block_count": len(structural_inputs.block_axis),
            "direct_prior_feature_count": direct_prior_feature_count(direct_prior_features),
            "hidden_driver_feature_count": int(sum(len(rows) for rows in hidden_driver_features.values())),
        }
    best = _best_candidate(
        observation_rows=observation_rows,
        splits=splits,
        candidate_specs=candidate_specs,
        module_name=module_name,
        structural_inputs=structural_inputs,
        direct_prior_features=direct_prior_features,
        hidden_driver_features=hidden_driver_features,
    )
    payload = _build_stage_payload(
        revision_stage=revision_stage,
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
        observation_rows=observation_rows,
        observation_role_ledger=ledger,
        best_candidate=best["candidate"],
        all_candidates=best["all_candidates"],
        module_name=module_name,
        epigraph_root=epigraph_root,
        phase2_context=phase2_context,
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    artifact_paths = _finalize_run_artifacts(
        analysis_dir=analysis_dir,
        report_stem=revision_stage.lower().replace("-", "_") + "_report",
        payload=payload,
        epigraph_root=epigraph_root,
    )
    payload["artifact_paths"] = {key: path.as_posix() for key, path in artifact_paths.items()}
    return payload


def run_rev_00_observation_role_ledger(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    ledger = build_observation_role_ledger(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "family_name": "REV-00",
        "revision_stage": "REV-00",
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "decision": "contract_only",
        "decision_reason": "Typed observation-role ledger emitted for the active evidence universe.",
        "loop_variant": "evidence-to-model-loop",
        "benchmark_contract": {"status": "contract_only_stage"},
        "data_provenance": {"observation_role_ledger_summary": dict((ledger.get("summary") or {}))},
        "observation_role_ledger": ledger,
        "observation_score_ledger": [],
        "observation_score_ledger_summary": {
            "schema_version": "phase3_dynamic_observation_score_ledger.v1",
            "entry_count": 0,
            "candidate_scored_entry_count": 0,
            "carry_forward_scored_entry_count": 0,
            "metrics": {},
        },
        "lockbox_contract": _lockbox_contract(),
        "best_candidate": {
            "config": {},
            "score": {
                "candidate_mean_mae": 0.0,
                "candidate_worst_mae": 0.0,
                "carry_forward_mean_mae": 0.0,
                "carry_forward_worst_mae": 0.0,
            },
            "rows": [],
        },
        "all_candidates": [],
        "support_partition_table": {"rows": []},
        "residual_summary": {"overall_candidate_p90": None, "overall_carry_forward_p90": None, "metric_rows": []},
        "raw_endpoint_error_report": {"status": "not_applicable"},
        "raw_endpoint_errors": {"status": "not_applicable"},
        "archive_drift_gate": {"status": "not_applicable"},
        "support_tier_stability_gate": {"status": "not_applicable"},
        "benchmark_comparison": {},
        "auxiliary_module_summary": None,
    }
    payload["claim_card"] = {
        "schema_version": CLAIM_CARD_SCHEMA_VERSION,
        "run_id": run_id,
        "revision_stage": "REV-00",
        "decision": "contract_only",
        "promotion_eligible": False,
        "identifiability_status": "not_applicable",
        "blockers": ["non_model_or_diagnostic_stage"],
        "warnings": [],
        "observation_role_counts": dict((ledger.get("summary") or {}).get("observation_role_counts") or {}),
    }
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    artifact_paths = _finalize_run_artifacts(
        analysis_dir=analysis_dir,
        report_stem="rev_00_observation_role_ledger_report",
        payload=payload,
        epigraph_root=epigraph_root,
    )
    payload["artifact_paths"] = {key: path.as_posix() for key, path in artifact_paths.items()}
    return payload


def run_rev_01_conserved_national_ssm(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    return _run_revision_stage(
        revision_stage="REV-01",
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
        candidate_specs=_base_dynamic_specs(),
        module_name=None,
    )


def run_rev_02_diagnosis_delay(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    base_spec = _base_dynamic_specs()[0]
    return _run_revision_stage(
        revision_stage="REV-02",
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
        candidate_specs=_incidence_specs(base_spec),
        module_name="diagnosis_delay",
    )


def run_rev_03_art_ltfu_vl(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    return _run_revision_stage(
        revision_stage="REV-03",
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
        candidate_specs=_incidence_specs(_base_dynamic_specs()[0]),
        module_name="art_ltfu_vl",
    )


def run_rev_04_prep_persistence(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    observation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    prep_rows = [
        {
            "quarter": str(row.get("quarter") or ""),
            "prep_newly_enrolled_period": row.get("prep_newly_enrolled_period"),
            "prep_people_receiving": row.get("prep_people_receiving"),
        }
        for row in observation_rows
        if row.get("prep_newly_enrolled_period") is not None or row.get("prep_people_receiving") is not None
    ]
    status = "scored" if len(prep_rows) >= 4 else "not_identifiable"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "family_name": "REV-04",
        "revision_stage": "REV-04",
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "decision": "not_identifiable" if status != "scored" else "diagnostic_only",
        "decision_reason": "PrEP persistence evidence is emitted as a typed diagnostic before mechanistic promotion.",
        "loop_variant": "evidence-to-model-loop",
        "benchmark_contract": {"status": "status_only_stage"},
        "data_provenance": {
            "observation_role_ledger_summary": dict(
                (
                    build_observation_role_ledger(
                        epigraph_root,
                        source_run_id=source_run_id,
                        baseline_source_run_id=baseline_source_run_id,
                    ).get("summary")
                    or {}
                )
            )
        },
        "observation_role_ledger": build_observation_role_ledger(
            epigraph_root,
            source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
        ),
        "observation_score_ledger": [],
        "observation_score_ledger_summary": {
            "schema_version": "phase3_dynamic_observation_score_ledger.v1",
            "entry_count": 0,
            "candidate_scored_entry_count": 0,
            "carry_forward_scored_entry_count": 0,
            "metrics": {},
        },
        "lockbox_contract": _lockbox_contract(),
        "best_candidate": {
            "config": {},
            "score": {
                "candidate_mean_mae": 0.0,
                "candidate_worst_mae": 0.0,
                "carry_forward_mean_mae": 0.0,
                "carry_forward_worst_mae": 0.0,
            },
            "rows": [],
        },
        "all_candidates": [],
        "support_partition_table": {"rows": []},
        "residual_summary": {"overall_candidate_p90": None, "overall_carry_forward_p90": None, "metric_rows": []},
        "raw_endpoint_error_report": {"status": "not_applicable"},
        "raw_endpoint_errors": {"status": "not_applicable"},
        "archive_drift_gate": {"status": "not_applicable"},
        "support_tier_stability_gate": {"status": "not_applicable"},
        "benchmark_comparison": {},
        "auxiliary_module_summary": {
            "module_name": "prep_persistence",
            "status": status,
            "reason": None if status == "scored" else "insufficient_prep_history",
            "row_count": len(prep_rows),
            "preview_rows": prep_rows[-8:],
        },
    }
    payload["claim_card"] = _claim_card(
        run_id=run_id,
        revision_stage="REV-04",
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        score=payload["best_candidate"]["score"],
        support_partition=payload["support_partition_table"],
        residual_summary=payload["residual_summary"],
        reference_benchmarks=payload["benchmark_comparison"],
        decision=str(payload["decision"]),
        auxiliary_summary=dict(payload["auxiliary_module_summary"]),
        identifiability_status="not_identifiable" if status != "scored" else "identifiable",
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    artifact_paths = _finalize_run_artifacts(
        analysis_dir=analysis_dir,
        report_stem="rev_04_prep_persistence_report",
        payload=payload,
        epigraph_root=epigraph_root,
    )
    payload["artifact_paths"] = {key: path.as_posix() for key, path in artifact_paths.items()}
    return payload


def run_rev_05_determinant_shrinkage(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    phase2_source_run_id: str | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    return _run_revision_stage(
        revision_stage="REV-05",
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
        candidate_specs=_phase2_specs(_incidence_specs(_base_dynamic_specs()[0])[0]),
        module_name="art_ltfu_vl",
        use_phase2=True,
        phase2_source_run_id=phase2_source_run_id,
    )


def _support_evidence_counts(ledger: dict[str, Any], *, name_patterns: tuple[str, ...]) -> int:
    count = 0
    for row in list(ledger.get("rows") or []):
        metric_id = str(row.get("metric_id") or "").lower()
        if any(pattern in metric_id for pattern in name_patterns):
            count += 1
    return int(count)


def _write_status_only_stage(
    *,
    revision_stage: str,
    run_id: str,
    source_run_id: str,
    baseline_source_run_id: str,
    epigraph_root: Path,
    ledger: dict[str, Any],
    status: str,
    reason: str,
    evidence_counts: dict[str, int],
) -> dict[str, Any]:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "family_name": revision_stage,
        "revision_stage": revision_stage,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "decision": status,
        "decision_reason": reason,
        "loop_variant": "evidence-to-model-loop",
        "benchmark_contract": {"status": "status_only_stage"},
        "data_provenance": {"observation_role_ledger_summary": dict((ledger.get("summary") or {}))},
        "observation_role_ledger": ledger,
        "observation_score_ledger": [],
        "observation_score_ledger_summary": {
            "schema_version": "phase3_dynamic_observation_score_ledger.v1",
            "entry_count": 0,
            "candidate_scored_entry_count": 0,
            "carry_forward_scored_entry_count": 0,
            "metrics": {},
        },
        "lockbox_contract": _lockbox_contract(),
        "best_candidate": {
            "config": {},
            "score": {
                "candidate_mean_mae": 0.0,
                "candidate_worst_mae": 0.0,
                "carry_forward_mean_mae": 0.0,
                "carry_forward_worst_mae": 0.0,
            },
            "rows": [],
        },
        "all_candidates": [],
        "support_partition_table": {"rows": []},
        "residual_summary": {"overall_candidate_p90": None, "overall_carry_forward_p90": None, "metric_rows": []},
        "raw_endpoint_error_report": {"status": "not_applicable"},
        "raw_endpoint_errors": {"status": "not_applicable"},
        "archive_drift_gate": {"status": "not_applicable"},
        "support_tier_stability_gate": {"status": "not_applicable"},
        "benchmark_comparison": {},
        "auxiliary_module_summary": {
            "module_name": revision_stage.lower(),
            "status": status,
            "reason": reason,
            "evidence_counts": evidence_counts,
        },
    }
    payload["claim_card"] = _claim_card(
        run_id=run_id,
        revision_stage=revision_stage,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        score=payload["best_candidate"]["score"],
        support_partition=payload["support_partition_table"],
        residual_summary=payload["residual_summary"],
        reference_benchmarks=payload["benchmark_comparison"],
        decision=status,
        auxiliary_summary=dict(payload["auxiliary_module_summary"]),
        identifiability_status="not_identifiable",
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    artifact_paths = _finalize_run_artifacts(
        analysis_dir=analysis_dir,
        report_stem=revision_stage.lower().replace("-", "_") + "_report",
        payload=payload,
        epigraph_root=epigraph_root,
    )
    payload["artifact_paths"] = {key: path.as_posix() for key, path in artifact_paths.items()}
    return payload


def run_rev_06_kp_lite_overlay(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    ledger = build_observation_role_ledger(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    counts = {
        "msm_rows": _support_evidence_counts(ledger, name_patterns=("msm", "men_who_have_sex_with_men")),
        "tgw_rows": _support_evidence_counts(ledger, name_patterns=("transgender", "tgw")),
        "sex_worker_rows": _support_evidence_counts(ledger, name_patterns=("sex_worker", "transactional")),
        "pwid_rows": _support_evidence_counts(ledger, name_patterns=("pwid", "inject_drugs")),
    }
    return _write_status_only_stage(
        revision_stage="REV-06",
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        epigraph_root=epigraph_root,
        ledger=ledger,
        status="not_identifiable",
        reason="no_supported_key_population_time_series_in_active_evidence_universe",
        evidence_counts=counts,
    )


def run_rev_07_region_hierarchy(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    ledger = build_observation_role_ledger(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    region_rows = [
        row
        for row in list(ledger.get("rows") or [])
        if str(row.get("geography") or "").lower() not in {"national", "philippines"}
    ]
    counts = {
        "subnational_row_count": int(len(region_rows)),
        "direct_target_subnational_rows": int(
            sum(1 for row in region_rows if str(row.get("observation_role") or "") == "direct_target")
        ),
    }
    return _write_status_only_stage(
        revision_stage="REV-07",
        run_id=run_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        epigraph_root=epigraph_root,
        ledger=ledger,
        status="auxiliary_only",
        reason="subnational_truth_is_not_validated_for_champion_promotion",
        evidence_counts=counts,
    )


def run_publication_revision_program(
    *,
    run_id_prefix: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    phase2_source_run_id: str | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    contracts = write_project_contract_artifacts(
        project_root=epigraph_root,
        phase3_dynamic_root=sandbox_repo_root(),
    )
    lineage_path = write_project_lineage_manifest(
        project_root=epigraph_root,
        phase3_dynamic_root=sandbox_repo_root(),
    )
    rev00 = run_rev_00_observation_role_ledger(
        run_id=f"{run_id_prefix}-rev00",
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    rev01 = run_rev_01_conserved_national_ssm(
        run_id=f"{run_id_prefix}-rev01",
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    rev02 = run_rev_02_diagnosis_delay(
        run_id=f"{run_id_prefix}-rev02",
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    rev03 = run_rev_03_art_ltfu_vl(
        run_id=f"{run_id_prefix}-rev03",
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
    )
    rev04 = run_rev_04_prep_persistence(
        run_id=f"{run_id_prefix}-rev04",
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    rev05 = run_rev_05_determinant_shrinkage(
        run_id=f"{run_id_prefix}-rev05",
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=horizon_years,
        phase2_source_run_id=phase2_source_run_id,
    )
    rev06 = run_rev_06_kp_lite_overlay(
        run_id=f"{run_id_prefix}-rev06",
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    rev07 = run_rev_07_region_hierarchy(
        run_id=f"{run_id_prefix}-rev07",
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    registry = write_experiment_registry(sandbox_root=sandbox_repo_root())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id_prefix": run_id_prefix,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "phase2_source_run_id": phase2_source_run_id,
        "contract_paths": {key: path.as_posix() for key, path in contracts.items()},
        "lineage_manifest_path": lineage_path.as_posix(),
        "experiment_registry_path": (sandbox_repo_root() / "artifacts" / "experiment_tracking" / "experiment_registry.json").as_posix(),
        "claim_aware_champion": None
        if registry.get("champion_by_claim_aware_promotion") is None
        else str((registry.get("champion_by_claim_aware_promotion") or {}).get("run_id") or ""),
        "stages": {
            "REV-00": rev00.get("artifact_paths"),
            "REV-01": rev01.get("artifact_paths"),
            "REV-02": rev02.get("artifact_paths"),
            "REV-03": rev03.get("artifact_paths"),
            "REV-04": rev04.get("artifact_paths"),
            "REV-05": rev05.get("artifact_paths"),
            "REV-06": rev06.get("artifact_paths"),
            "REV-07": rev07.get("artifact_paths"),
        },
    }
