from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import MISSING_DATA_LADDER, build_blocked_time_dataset, build_early_partial_target_rows, build_observation_rows, default_epigraph_root, partial_observation_splits, rolling_origin_splits, sandbox_repo_root, summarize_early_partial_provenance, summarize_observation_provenance
from .incidence import IncidenceFlowConfig, carry_forward_incidence_flow_paths
from .metrics import PRIMARY_METRICS
from .model import DampingConfig, DirectPriorConfig, DynamicBaselineConfig, HiddenDriverConfig, ObservationModelConfig, ShockConfig, carry_forward_hazards, forecast_dynamic_baseline, simulate_holdout
from .phase2 import build_direct_prior_features, build_hidden_driver_features, load_phase2_structural_inputs
from .priors import PHASE2_TRANSITION_PRIOR_MAP
from .runtime import ensure_dir, write_json
from .scientific_contracts import build_hazard_semantics, build_model_contract


PHASE2_PRIOR_CONTRACT: dict[str, str] = {
    "direct_terms": "train_history_only; requested holdout source quarters after forecast origin are deterministic carry-forward from the last training quarter",
    "hidden_terms": "train_history_only; holdout hidden-mode values are deterministic carry-forward from the last training quarter",
    "fit_scope": "coefficients and standardization are fitted on train transitions only",
}
OBSERVATION_SCORE_LEDGER_SCHEMA_VERSION = "phase3_dynamic_observation_score_ledger.v1"


def _dynamic_candidate_grid() -> list[DynamicBaselineConfig]:
    return [DynamicBaselineConfig(ridge_penalty=ridge_penalty, rho_clip=rho_clip, trend_scale=trend_scale) for ridge_penalty in (0.01, 0.1, 1.0, 5.0) for rho_clip in (0.80, 0.90, 0.98) for trend_scale in (0.25, 0.50, 1.00)]


def _observation_candidate_grid() -> list[ObservationModelConfig]:
    return [ObservationModelConfig(calibration_ridge=calibration_ridge, share_ridge_penalty=share_ridge_penalty, share_rho_clip=share_rho_clip, share_trend_scale=share_trend_scale) for calibration_ridge in (0.01, 0.1, 1.0) for share_ridge_penalty in (0.01, 0.1) for share_rho_clip in (0.80, 0.95) for share_trend_scale in (0.50, 1.00)]


def _shock_candidate_grid() -> list[ShockConfig]:
    return [ShockConfig(shock_phi=shock_phi, shock_scale=shock_scale, gate_z=gate_z) for shock_phi in (0.25, 0.50, 0.80) for shock_scale in (0.50, 1.00, 1.50) for gate_z in (0.50, 1.00, 1.50)]


def _anchored_dynamic_candidates() -> list[DynamicBaselineConfig]:
    return [DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0), DynamicBaselineConfig(ridge_penalty=0.1, rho_clip=0.8, trend_scale=1.0)]


def _anchored_observation_candidates() -> list[ObservationModelConfig]:
    return [ObservationModelConfig(calibration_ridge=0.01, share_ridge_penalty=0.01, share_rho_clip=0.8, share_trend_scale=0.5), ObservationModelConfig(calibration_ridge=0.1, share_ridge_penalty=0.01, share_rho_clip=0.8, share_trend_scale=0.5)]


def _anchored_shock_candidates() -> list[ShockConfig | None]:
    return [None, ShockConfig(shock_phi=0.8, shock_scale=1.0, gate_z=1.5), ShockConfig(shock_phi=0.8, shock_scale=1.5, gate_z=1.5)]


def _anchored_incidence_candidates() -> list[IncidenceFlowConfig]:
    return [
        IncidenceFlowConfig(ridge_penalty=ridge_penalty, rho_clip=rho_clip, trend_scale=trend_scale)
        for ridge_penalty in (0.01, 0.1)
        for rho_clip in (0.8, 0.9)
        for trend_scale in (0.5, 1.0)
    ]


def _damping_candidate_grid() -> list[DampingConfig]:
    rows: list[DampingConfig] = []
    for min_dynamic_weight in (0.15, 0.25, 0.40):
        for gain_weight in (0.50, 0.75):
            for horizon_decay in (0.40, 0.60, 0.80):
                for calibration_floor in (0.20, 0.35):
                    rows.append(DampingConfig(min_dynamic_weight=min_dynamic_weight, gain_weight=gain_weight, residual_weight=0.15, horizon_decay=horizon_decay, calibration_floor=calibration_floor, calibration_gain_weight=0.70))
    return rows


def _direct_prior_candidate_grid() -> list[DirectPriorConfig]:
    return [DirectPriorConfig(effect_scale=effect_scale, precision_scale=precision_scale, residual_ridge=residual_ridge, max_effect=max_effect) for effect_scale in (0.5, 1.0, 1.5) for precision_scale in (0.5, 1.0, 2.0) for residual_ridge in (0.01, 0.1) for max_effect in (0.20, 0.35)]


def _hidden_driver_candidate_grid() -> list[HiddenDriverConfig]:
    return [HiddenDriverConfig(precision_scale=precision_scale, residual_ridge=residual_ridge, max_effect=max_effect, rank_cap=1) for precision_scale in (0.25, 0.5, 1.0, 2.0) for residual_ridge in (0.01, 0.1) for max_effect in (0.05, 0.10, 0.20)]


def _run_candidate(observation_rows: list[dict[str, Any]], split: dict[str, Any], dynamic_cfg: DynamicBaselineConfig, *, incidence_cfg: IncidenceFlowConfig | None = None, observation_cfg: ObservationModelConfig | None = None, shock_cfg: ShockConfig | None = None, damping_cfg: DampingConfig | None = None, structural_inputs: Any | None = None, direct_prior_features: Any | None = None, prior_cfg: DirectPriorConfig | None = None, hidden_driver_features: Any | None = None, hidden_cfg: HiddenDriverConfig | None = None, include_scoring_details: bool = False) -> dict[str, Any]:
    dataset = build_blocked_time_dataset(observation_rows, list(split["holdout_years"]))
    carry_forward_stock_balance = None if incidence_cfg is None else carry_forward_incidence_flow_paths(dataset, mode="last_train")
    carry_forward_result = simulate_holdout(
        dataset,
        carry_forward_hazards(dataset, mode="last_train"),
        incidence_inflow_map=None if carry_forward_stock_balance is None else dict(carry_forward_stock_balance.get("incidence_inflow_map") or {}),
        incidence_hazard_map=None if carry_forward_stock_balance is None else dict(carry_forward_stock_balance.get("incidence_hazard_map") or {}),
        population_denominator_map=None if carry_forward_stock_balance is None else dict(carry_forward_stock_balance.get("population_denominator_map") or {}),
        attrition_outflow_map=None if carry_forward_stock_balance is None else dict(carry_forward_stock_balance.get("attrition_outflow_map") or {}),
        state_attrition_outflow_map=None if carry_forward_stock_balance is None else dict(carry_forward_stock_balance.get("state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=None if carry_forward_stock_balance is None else dict(carry_forward_stock_balance.get("exit_channel_state_outflow_map") or {}),
    )
    candidate_result = forecast_dynamic_baseline(dataset, dynamic_cfg, incidence_cfg=incidence_cfg, observation_cfg=observation_cfg, shock_cfg=shock_cfg, damping_cfg=damping_cfg, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, prior_cfg=prior_cfg, hidden_driver_features=hidden_driver_features, hidden_cfg=hidden_cfg)
    payload = {
        "train_end_year": int(split["train_end_year"]),
        "train_years": list(split["train_years"]),
        "holdout_years": list(split["holdout_years"]),
        "carry_forward": {"mae": float(carry_forward_result["mae"]), "smape": float(carry_forward_result["smape"])},
        "candidate": {
            "mae": float(candidate_result["mae"]),
            "smape": float(candidate_result["smape"]),
            "model_contract": candidate_result.get("model_contract"),
            "hazard_semantics": candidate_result.get("hazard_semantics"),
            "incidence_diagnostics": None if candidate_result.get("incidence_paths") is None else dict((candidate_result.get("incidence_paths") or {}).get("diagnostics") or {}),
        },
        "dataset_provenance": dict(dataset.provenance_summary),
    }
    if include_scoring_details:
        payload["scoring_details"] = {
            "metric_scales": {str(key): float(value) for key, value in dict(dataset.metric_scales).items()},
            "eps": float(dataset.eps),
            "holdout_rows": [dict(row) for row in dataset.holdout_rows],
            "candidate_prediction_rows": [dict(row) for row in list(candidate_result.get("prediction_rows") or [])],
            "carry_forward_prediction_rows": [dict(row) for row in list(carry_forward_result.get("prediction_rows") or [])],
        }
    return payload


def _run_spec_rows(
    observation_rows: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    structural_inputs: Any | None = None,
    direct_prior_features: Any | None = None,
    hidden_driver_features: Any | None = None,
    include_scoring_details: bool = False,
) -> list[dict[str, Any]]:
    use_phase2 = spec.get("prior_cfg") is not None or spec.get("hidden_cfg") is not None
    return [
        _run_candidate(
            observation_rows,
            split,
            spec["dynamic_cfg"],
            incidence_cfg=spec.get("incidence_cfg"),
            observation_cfg=spec.get("observation_cfg"),
            shock_cfg=spec.get("shock_cfg"),
            damping_cfg=spec.get("damping_cfg"),
            structural_inputs=structural_inputs if use_phase2 else None,
            direct_prior_features=direct_prior_features if spec.get("prior_cfg") is not None else None,
            prior_cfg=spec.get("prior_cfg"),
            hidden_driver_features=hidden_driver_features if spec.get("hidden_cfg") is not None else None,
            hidden_cfg=spec.get("hidden_cfg"),
            include_scoring_details=include_scoring_details,
        )
        for split in splits
    ]


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    carry_forward_maes = [float(row["carry_forward"]["mae"]) for row in rows]
    candidate_maes = [float(row["candidate"]["mae"]) for row in rows]
    carry_forward_smape = [float(row["carry_forward"]["smape"]) for row in rows]
    candidate_smape = [float(row["candidate"]["smape"]) for row in rows]
    return {"carry_forward_mean_mae": sum(carry_forward_maes) / max(len(carry_forward_maes), 1), "candidate_mean_mae": sum(candidate_maes) / max(len(candidate_maes), 1), "carry_forward_mean_smape": sum(carry_forward_smape) / max(len(carry_forward_smape), 1), "candidate_mean_smape": sum(candidate_smape) / max(len(candidate_smape), 1), "carry_forward_worst_mae": max(carry_forward_maes) if carry_forward_maes else float("inf"), "candidate_worst_mae": max(candidate_maes) if candidate_maes else float("inf")}


def _single_holdout_year(row: dict[str, Any]) -> int:
    return int(list(row.get("holdout_years") or [0])[0])


def _score_future_window(rows: list[dict[str, Any]], *, start_year: int) -> float:
    filtered = [float(row["candidate"]["mae"]) for row in rows if _single_holdout_year(row) >= int(start_year)]
    return sum(filtered) / max(len(filtered), 1)


def _candidate_config(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "dynamic_cfg": asdict(spec["dynamic_cfg"]),
        "incidence_cfg": None if spec.get("incidence_cfg") is None else asdict(spec["incidence_cfg"]),
        "observation_cfg": None if spec.get("observation_cfg") is None else asdict(spec["observation_cfg"]),
        "shock_cfg": None if spec.get("shock_cfg") is None else asdict(spec["shock_cfg"]),
        "damping_cfg": None if spec.get("damping_cfg") is None else asdict(spec["damping_cfg"]),
        "prior_cfg": None if spec.get("prior_cfg") is None else asdict(spec["prior_cfg"]),
        "hidden_cfg": None if spec.get("hidden_cfg") is None else asdict(spec["hidden_cfg"]),
    }


def _keep_decision(score: dict[str, float]) -> tuple[str, str]:
    keep = float(score["candidate_mean_mae"]) < float(score["carry_forward_mean_mae"]) and float(score["candidate_worst_mae"]) <= (1.10 * float(score["carry_forward_worst_mae"]))
    return ("keep", "Candidate beat carry-forward on mean blocked-time MAE without unacceptable worst-split regression.") if keep else ("revert", "Candidate did not clear the blocked-time benchmark gate against carry-forward.")


def _merge_count_tree(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, dict):
            node = base.setdefault(key, {})
            if isinstance(node, dict):
                _merge_count_tree(node, value)
        elif isinstance(value, int):
            base[key] = int(base.get(key, 0)) + int(value)


def _aggregate_candidate_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"ladder": list(MISSING_DATA_LADDER)}
    for row in rows:
        provenance = dict(row.get("dataset_provenance") or {})
        for key, value in provenance.items():
            if key == "ladder":
                continue
            if isinstance(value, dict):
                _merge_count_tree(aggregate.setdefault(key, {}), value)
    return aggregate


def _time_support_from_aggregation_mode(aggregation_mode: str) -> str:
    mode = str(aggregation_mode or "")
    if mode == "annual_anchor_to_q4":
        return "annual_period"
    if mode in {"monthly_to_quarter_sum", "intraquarter_snapshot_bridge"}:
        return "multi_period_bridge"
    return "quarterly_period"


def _value_role_from_tier(tier: str) -> str:
    return {
        "exact_observed": "direct_measurement",
        "bridge_observed": "bridge_observed",
        "rule_based_extrapolated": "rule_based_extrapolated",
        "latent_imputed": "latent_imputed",
    }.get(str(tier or ""), "context_only")


def _score_reason(*, target_value: Any, prediction_value: Any) -> str:
    if target_value is None:
        return "missing_target"
    if prediction_value is None:
        return "missing_prediction"
    return "scored"


def _role_score_reason(*, observation_role: str, target_value: Any, prediction_value: Any) -> str:
    if str(observation_role or "") != "direct_target":
        return "non_direct_observation_role"
    return _score_reason(target_value=target_value, prediction_value=prediction_value)


def _normalized_error(*, prediction_value: Any, target_value: Any, scale: float, eps: float) -> float | None:
    if prediction_value is None or target_value is None:
        return None
    return abs(float(prediction_value) - float(target_value)) / max(float(scale), float(eps))


def _build_observation_score_ledger(candidate: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for split_row in list(candidate.get("rows") or []):
        detail = dict(split_row.get("scoring_details") or {})
        holdout_rows = list(detail.get("holdout_rows") or [])
        candidate_prediction_rows = list(detail.get("candidate_prediction_rows") or [])
        carry_forward_prediction_rows = list(detail.get("carry_forward_prediction_rows") or [])
        metric_scales = {str(key): float(value) for key, value in dict(detail.get("metric_scales") or {}).items()}
        eps = float(detail.get("eps") or np.finfo(np.float32).eps)
        for row_index, target_row in enumerate(holdout_rows):
            candidate_prediction = dict(candidate_prediction_rows[row_index]) if row_index < len(candidate_prediction_rows) else {}
            carry_forward_prediction = dict(carry_forward_prediction_rows[row_index]) if row_index < len(carry_forward_prediction_rows) else {}
            metric_provenance = dict(target_row.get("metric_provenance") or {})
            for metric_name in PRIMARY_METRICS:
                provenance = dict(metric_provenance.get(metric_name) or {})
                target_value = target_row.get(metric_name)
                candidate_value = candidate_prediction.get(metric_name)
                carry_forward_value = carry_forward_prediction.get(metric_name)
                observation_role = str(provenance.get("observation_role") or "")
                candidate_reason = _role_score_reason(
                    observation_role=observation_role,
                    target_value=target_value,
                    prediction_value=candidate_value,
                )
                carry_forward_reason = _role_score_reason(
                    observation_role=observation_role,
                    target_value=target_value,
                    prediction_value=carry_forward_value,
                )
                tier = str(provenance.get("tier") or target_row.get("row_provenance_tier") or "rejected_or_quarantined")
                aggregation_mode = str(provenance.get("aggregation_mode") or "")
                scale = float(metric_scales.get(metric_name) or eps)
                entries.append(
                    {
                        "train_end_year": int(split_row.get("train_end_year") or 0),
                        "holdout_years": [int(value) for value in list(split_row.get("holdout_years") or [])],
                        "quarter": str(target_row.get("quarter") or ""),
                        "geography": "national",
                        "metric_name": str(metric_name),
                        "time_support": _time_support_from_aggregation_mode(aggregation_mode),
                        "value_role": _value_role_from_tier(tier),
                        "row_provenance_tier": str(target_row.get("row_provenance_tier") or "rejected_or_quarantined"),
                        "metric_provenance_tier": tier,
                        "aggregation_mode": aggregation_mode,
                        "source_id": str(provenance.get("source_id") or ""),
                        "source_quality_tier": str(provenance.get("source_quality_tier") or ""),
                        "measurement_class": str(provenance.get("measurement_class") or ""),
                        "series_kind": str(provenance.get("series_kind") or ""),
                        "observation_role": observation_role,
                        "allowed_use": str(provenance.get("allowed_use") or observation_role),
                        "target_value": None if target_value is None else float(target_value),
                        "candidate_prediction_value": None if candidate_value is None else float(candidate_value),
                        "carry_forward_prediction_value": None if carry_forward_value is None else float(carry_forward_value),
                        "metric_scale": float(scale),
                        "candidate_scored": candidate_reason == "scored",
                        "candidate_score_reason": candidate_reason,
                        "carry_forward_scored": carry_forward_reason == "scored",
                        "carry_forward_score_reason": carry_forward_reason,
                        "candidate_normalized_abs_error": _normalized_error(prediction_value=candidate_value, target_value=target_value, scale=scale, eps=eps),
                        "carry_forward_normalized_abs_error": _normalized_error(prediction_value=carry_forward_value, target_value=target_value, scale=scale, eps=eps),
                    }
                )
    metric_summary: dict[str, dict[str, Any]] = {}
    row_tier_counts = {tier: 0 for tier in MISSING_DATA_LADDER}
    value_role_counts: dict[str, int] = {}
    for entry in entries:
        metric_name = str(entry["metric_name"])
        metric_entry = metric_summary.setdefault(
            metric_name,
            {
                "entry_count": 0,
                "candidate_scored_count": 0,
                "carry_forward_scored_count": 0,
                "metric_tier_counts": {tier: 0 for tier in MISSING_DATA_LADDER},
                "value_role_counts": {},
                "time_support_counts": {},
            },
        )
        metric_entry["entry_count"] += 1
        metric_entry["candidate_scored_count"] += 1 if entry["candidate_scored"] else 0
        metric_entry["carry_forward_scored_count"] += 1 if entry["carry_forward_scored"] else 0
        tier = str(entry["metric_provenance_tier"])
        metric_entry["metric_tier_counts"][tier] = int(metric_entry["metric_tier_counts"].get(tier, 0)) + 1
        row_tier = str(entry["row_provenance_tier"])
        row_tier_counts[row_tier] = int(row_tier_counts.get(row_tier, 0)) + 1
        value_role = str(entry["value_role"])
        metric_entry["value_role_counts"][value_role] = int(metric_entry["value_role_counts"].get(value_role, 0)) + 1
        metric_entry["time_support_counts"][str(entry["time_support"])] = int(metric_entry["time_support_counts"].get(str(entry["time_support"]), 0)) + 1
        value_role_counts[value_role] = int(value_role_counts.get(value_role, 0)) + 1
    return {
        "entries": entries,
        "summary": {
            "schema_version": OBSERVATION_SCORE_LEDGER_SCHEMA_VERSION,
            "entry_count": len(entries),
            "candidate_scored_entry_count": sum(1 for entry in entries if bool(entry["candidate_scored"])),
            "carry_forward_scored_entry_count": sum(1 for entry in entries if bool(entry["carry_forward_scored"])),
            "row_tier_counts": row_tier_counts,
            "value_role_counts": value_role_counts,
            "metrics": metric_summary,
        },
    }


def _build_data_provenance_payload(observation_rows: list[dict[str, Any]], candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "ladder": list(MISSING_DATA_LADDER),
        "overall_observation_rows": summarize_observation_provenance(observation_rows),
    }
    if candidate is not None:
        payload["benchmark_weighted"] = _aggregate_candidate_provenance(list(candidate.get("rows") or []))
    return payload


def _candidate_model_contract(spec: dict[str, Any]) -> dict[str, Any]:
    return build_model_contract(
        incidence_enabled=spec.get("incidence_cfg") is not None,
        observation_calibration_enabled=spec.get("observation_cfg") is not None,
        phase2_direct_enabled=spec.get("prior_cfg") is not None,
        phase2_hidden_enabled=spec.get("hidden_cfg") is not None,
    )


def _candidate_hazard_semantics(spec: dict[str, Any]) -> dict[str, Any]:
    return build_hazard_semantics(
        incidence_enabled=spec.get("incidence_cfg") is not None,
        observation_calibration_enabled=spec.get("observation_cfg") is not None,
        phase2_direct_enabled=spec.get("prior_cfg") is not None,
        phase2_hidden_enabled=spec.get("hidden_cfg") is not None,
    )


def _attach_observation_score_ledger(payload: dict[str, Any], best_candidate: dict[str, Any]) -> None:
    ledger = _build_observation_score_ledger(best_candidate)
    payload["observation_score_ledger"] = list(ledger["entries"])
    payload["observation_score_ledger_summary"] = dict(ledger["summary"])


def _append_ladder_table(lines: list[str], label: str, row_tier_counts: dict[str, Any]) -> None:
    lines.append(f"| {label} | {int(row_tier_counts.get('exact_observed', 0))} | {int(row_tier_counts.get('bridge_observed', 0))} | {int(row_tier_counts.get('rule_based_extrapolated', 0))} | {int(row_tier_counts.get('latent_imputed', 0))} | {int(row_tier_counts.get('rejected_or_quarantined', 0))} |")


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [f"# {payload['family_name']} Autoresearch Report", "", f"Source run: `{payload['source_run_id']}`", f"Generated at: `{payload['generated_at']}`", "", "## Decision", "", f"- Decision: `{payload['decision']}`", f"- Reason: `{payload['decision_reason']}`", "", "## Best Candidate", "", f"- Config: `{payload['best_candidate']['config']}`", f"- Candidate mean MAE: `{payload['best_candidate']['score']['candidate_mean_mae']:.6f}`", f"- Carry-forward mean MAE: `{payload['best_candidate']['score']['carry_forward_mean_mae']:.6f}`", f"- Candidate worst MAE: `{payload['best_candidate']['score']['candidate_worst_mae']:.6f}`", f"- Carry-forward worst MAE: `{payload['best_candidate']['score']['carry_forward_worst_mae']:.6f}`"]
    model_contract = dict(payload.get("model_contract") or {})
    hazard_semantics = dict(payload.get("hazard_semantics") or {})
    if model_contract or hazard_semantics:
        lines.extend([
            "",
            "## Model Contract",
            "",
            f"- Model kind: `{model_contract.get('model_kind')}`",
            f"- Primary output contract: `{model_contract.get('primary_output_contract')}`",
            f"- Transition contract: `{model_contract.get('transition_contract')}`",
            f"- Incidence contract: `{model_contract.get('incidence_contract')}`",
            f"- Observation contract: `{model_contract.get('observation_contract')}`",
            f"- Fitted hazard semantics: `{(hazard_semantics.get('fitted_transition_hazard') or {}).get('meaning')}`",
            f"- Incidence semantics: `{(hazard_semantics.get('latent_incidence_inflow') or {}).get('status')}`",
            f"- Diagnostic derived hazard status: `{(hazard_semantics.get('diagnostic_derived_hazard') or {}).get('status')}`",
        ])
    phase2_contract = dict(payload.get("phase2_prior_contract") or {})
    if phase2_contract:
        lines.extend([
            "",
            "## Phase 2 Prior Contract",
            "",
            f"- Direct terms: `{phase2_contract.get('direct_terms')}`",
            f"- Hidden terms: `{phase2_contract.get('hidden_terms')}`",
            f"- Fit scope: `{phase2_contract.get('fit_scope')}`",
        ])
    data_provenance = dict(payload.get("data_provenance") or {})
    if data_provenance:
        overall = dict(data_provenance.get("overall_observation_rows") or {})
        benchmark_weighted = dict(data_provenance.get("benchmark_weighted") or {})
        lines.extend([
            "",
            "## Data Provenance",
            "",
            f"- Missing-data ladder: `{', '.join(str(value) for value in data_provenance.get('ladder') or [])}`",
            "",
            "| Slice | Exact | Bridge | Rule-based | Latent | Rejected |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        if overall:
            _append_ladder_table(lines, "Overall observation rows", dict(overall.get("row_tier_counts") or {}))
        benchmark_observation = dict(benchmark_weighted.get("observation_rows") or {})
        benchmark_train_obs = dict(benchmark_weighted.get("train_observation_rows") or {})
        benchmark_holdout_obs = dict(benchmark_weighted.get("holdout_observation_rows") or {})
        benchmark_train_state = dict(benchmark_weighted.get("train_state_rows") or {})
        benchmark_holdout_state = dict(benchmark_weighted.get("holdout_state_rows") or {})
        if benchmark_observation:
            _append_ladder_table(lines, "Benchmark-weighted observation rows", dict(benchmark_observation.get("row_tier_counts") or {}))
        if benchmark_train_obs:
            _append_ladder_table(lines, "Benchmark-weighted train observation", dict(benchmark_train_obs.get("row_tier_counts") or {}))
        if benchmark_holdout_obs:
            _append_ladder_table(lines, "Benchmark-weighted holdout observation", dict(benchmark_holdout_obs.get("row_tier_counts") or {}))
        if benchmark_train_state:
            _append_ladder_table(lines, "Benchmark-weighted train state", dict(benchmark_train_state.get("row_tier_counts") or {}))
        if benchmark_holdout_state:
            _append_ladder_table(lines, "Benchmark-weighted holdout state", dict(benchmark_holdout_state.get("row_tier_counts") or {}))
    observation_score_ledger_summary = dict(payload.get("observation_score_ledger_summary") or {})
    if observation_score_ledger_summary:
        lines.extend([
            "",
            "## Observation-to-Score Ledger",
            "",
            f"- Schema: `{observation_score_ledger_summary.get('schema_version')}`",
            f"- Ledger entries: `{int(observation_score_ledger_summary.get('entry_count') or 0)}`",
            f"- Candidate scored entries: `{int(observation_score_ledger_summary.get('candidate_scored_entry_count') or 0)}`",
            f"- Carry-forward scored entries: `{int(observation_score_ledger_summary.get('carry_forward_scored_entry_count') or 0)}`",
            "",
            "| Metric | Entries | Candidate scored | Carry-forward scored | Exact | Bridge | Rule-based | Latent |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for metric_name, summary in sorted(dict(observation_score_ledger_summary.get("metrics") or {}).items()):
            tier_counts = dict(summary.get("metric_tier_counts") or {})
            lines.append(
                f"| `{metric_name}` | {int(summary.get('entry_count') or 0)} | {int(summary.get('candidate_scored_count') or 0)} | "
                f"{int(summary.get('carry_forward_scored_count') or 0)} | {int(tier_counts.get('exact_observed', 0))} | "
                f"{int(tier_counts.get('bridge_observed', 0))} | {int(tier_counts.get('rule_based_extrapolated', 0))} | "
                f"{int(tier_counts.get('latent_imputed', 0))} |"
            )
    control_candidate = payload.get("control_candidate")
    if control_candidate is not None:
        control_family_name = str(payload.get("control_family_name") or "TR-V3-03")
        gate = dict(payload.get("hidden_channel_gate") or payload.get("tr_v3_04c_gate") or payload.get("tr_v3_04b_gate") or payload.get("tr_v3_04_gate") or {})
        lines.extend([
            "",
            f"## {control_family_name} Control",
            "",
            f"- Control config: `{control_candidate['config']}`",
            f"- Control mean MAE: `{control_candidate['score']['candidate_mean_mae']:.6f}`",
            f"- Control worst MAE: `{control_candidate['score']['candidate_worst_mae']:.6f}`",
            "",
            "## Hidden-Channel Gate",
            "",
            f"- Tail-window candidate MAE (`2024-2025`): `{gate.get('candidate_tail_mean_mae', gate.get('candidate_future_mean_mae', float('nan'))):.6f}`",
            f"- Tail-window control MAE (`2024-2025`): `{gate.get('control_tail_mean_mae', gate.get('control_future_mean_mae', float('nan'))):.6f}`",
            f"- 2023 candidate MAE: `{gate.get('candidate_2023_mae', float('nan')):.6f}`",
            f"- 2023 control MAE: `{gate.get('control_2023_mae', float('nan')):.6f}`",
            f"- 2022 candidate MAE: `{gate.get('candidate_2022_mae', float('nan')):.6f}`",
            f"- 2022 control MAE: `{gate.get('control_2022_mae', float('nan')):.6f}`",
            f"- Allowed 2022 ceiling: `{gate.get('allowed_2022_ceiling', float('nan')):.6f}`",
        ])
    lines.extend(["", "## Split Summary", "", "| Train end | Holdout | Carry-forward MAE | Candidate MAE | Delta |", "|---|---:|---:|---:|---:|"])
    control_by_year = {_single_holdout_year(row): row for row in list(control_candidate.get("rows") or [])} if control_candidate is not None else {}
    for row in payload["best_candidate"]["rows"]:
        carry_forward_mae = float(row["carry_forward"]["mae"])
        candidate_mae = float(row["candidate"]["mae"])
        lines.append(f"| {row['train_end_year']} | {','.join(str(value) for value in row['holdout_years'])} | {carry_forward_mae:.6f} | {candidate_mae:.6f} | {candidate_mae - carry_forward_mae:+.6f} |")
        if control_candidate is not None:
            year = _single_holdout_year(row)
            control_row = control_by_year.get(year)
            if control_row is not None:
                control_mae = float(control_row["candidate"]["mae"])
                lines.append(f"| control | {year} | n/a | {control_mae:.6f} | {candidate_mae - control_mae:+.6f} vs control |")
    return "\n".join(lines) + "\n"

def _run_family_loop(*, family_name: str, report_stem: str, run_id: str, source_run_id: str, epigraph_root: Path | None, start_year: int, end_year: int, min_train_years: int, horizon_years: int, candidate_builders: list[dict[str, Any]]) -> dict[str, Any]:
    epigraph_root = Path(epigraph_root) if epigraph_root is not None else default_epigraph_root()
    observation_rows = build_observation_rows(epigraph_root, source_run_id)
    structural_inputs = load_phase2_structural_inputs(epigraph_root, source_run_id)
    direct_prior_features = build_direct_prior_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    hidden_driver_features = build_hidden_driver_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    splits = rolling_origin_splits(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)
    candidate_payloads: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(candidate_builders):
        rows = _run_spec_rows(observation_rows, splits, spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
        candidate_payloads.append({"candidate_index": int(candidate_index), "config": _candidate_config(spec), "model_contract": _candidate_model_contract(spec), "hazard_semantics": _candidate_hazard_semantics(spec), "rows": rows, "score": _score_rows(rows)})
    best_candidate = min(candidate_payloads, key=lambda row: (row["score"]["candidate_mean_mae"], row["score"]["candidate_worst_mae"]))
    best_spec = dict(candidate_builders[int(best_candidate["candidate_index"])])
    best_candidate = dict(best_candidate)
    best_candidate["rows"] = _run_spec_rows(observation_rows, splits, best_spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features, include_scoring_details=True)
    decision, decision_reason = _keep_decision(best_candidate["score"])
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "family_name": family_name, "source_run_id": source_run_id, "loop_variant": "evidence-to-model-loop", "model_contract": best_candidate.get("model_contract"), "hazard_semantics": best_candidate.get("hazard_semantics"), "phase2_prior_contract": dict(PHASE2_PRIOR_CONTRACT), "benchmark_contract": {"start_year": int(start_year), "end_year": int(end_year), "min_train_years": int(min_train_years), "horizon_years": int(horizon_years)}, "phase2_context": {"direct_edge_count": len(structural_inputs.direct_edge_rows), "hidden_edge_count": len(structural_inputs.hidden_driver_rows), "multiscale_support_count": len(structural_inputs.multiscale_support_rows), "quarter_count": len(structural_inputs.quarter_axis), "block_count": len(structural_inputs.block_axis), "direct_prior_feature_count": int(sum(len(rows) for rows in direct_prior_features.values())), "hidden_driver_feature_count": int(sum(len(rows) for rows in hidden_driver_features.values()))}, "split_count": len(splits), "candidate_count": len(candidate_builders), "decision": decision, "decision_reason": decision_reason, "data_provenance": _build_data_provenance_payload(observation_rows, best_candidate), "best_candidate": best_candidate, "all_candidates": candidate_payloads}
    _attach_observation_score_ledger(payload, best_candidate)
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    write_json(analysis_dir / f"{report_stem}.json", payload)
    (analysis_dir / f"{report_stem}.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def run_tr_v3_00_loop(*, run_id: str, source_run_id: str = "smoke-latent-blocks", epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    return _run_family_loop(family_name="TR-V3-00", report_stem="tr_v3_00_autoresearch_report", run_id=run_id, source_run_id=source_run_id, epigraph_root=epigraph_root, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years, candidate_builders=[{"dynamic_cfg": cfg} for cfg in _dynamic_candidate_grid()])


def run_tr_v3_01_loop(*, run_id: str, source_run_id: str = "smoke-latent-blocks", epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    builders = [{"dynamic_cfg": dynamic_cfg, "observation_cfg": observation_cfg} for dynamic_cfg in _dynamic_candidate_grid() for observation_cfg in _observation_candidate_grid()]
    return _run_family_loop(family_name="TR-V3-01", report_stem="tr_v3_01_autoresearch_report", run_id=run_id, source_run_id=source_run_id, epigraph_root=epigraph_root, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years, candidate_builders=builders)


def run_tr_v3_02_loop(*, run_id: str, source_run_id: str = "smoke-latent-blocks", epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    builders = [{"dynamic_cfg": dynamic_cfg, "observation_cfg": observation_cfg, "shock_cfg": shock_cfg} for dynamic_cfg in _dynamic_candidate_grid() for observation_cfg in _observation_candidate_grid() for shock_cfg in _shock_candidate_grid()]
    return _run_family_loop(family_name="TR-V3-02", report_stem="tr_v3_02_autoresearch_report", run_id=run_id, source_run_id=source_run_id, epigraph_root=epigraph_root, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years, candidate_builders=builders)


def run_tr_v3_02b_loop(*, run_id: str, source_run_id: str = "smoke-latent-blocks", epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    builders = [{"dynamic_cfg": dynamic_cfg, "observation_cfg": observation_cfg, "shock_cfg": shock_cfg, "damping_cfg": damping_cfg} for dynamic_cfg in _anchored_dynamic_candidates() for observation_cfg in _anchored_observation_candidates() for shock_cfg in _anchored_shock_candidates() for damping_cfg in _damping_candidate_grid()]
    return _run_family_loop(family_name="TR-V3-02b", report_stem="tr_v3_02b_autoresearch_report", run_id=run_id, source_run_id=source_run_id, epigraph_root=epigraph_root, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years, candidate_builders=builders)


def run_tr_v3_03_loop(*, run_id: str, source_run_id: str = "smoke-latent-blocks", epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    backbone_dynamic = DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0)
    backbone_observation = ObservationModelConfig(calibration_ridge=0.01, share_ridge_penalty=0.01, share_rho_clip=0.8, share_trend_scale=0.5)
    backbone_shock = ShockConfig(shock_phi=0.8, shock_scale=1.5, gate_z=1.5)
    backbone_damping = DampingConfig(min_dynamic_weight=0.15, gain_weight=0.5, residual_weight=0.15, horizon_decay=0.4, calibration_floor=0.2, calibration_gain_weight=0.7)
    builders = [{"dynamic_cfg": backbone_dynamic, "observation_cfg": backbone_observation, "shock_cfg": backbone_shock, "damping_cfg": backbone_damping, "prior_cfg": prior_cfg} for prior_cfg in _direct_prior_candidate_grid()]
    return _run_family_loop(family_name="TR-V3-03", report_stem="tr_v3_03_autoresearch_report", run_id=run_id, source_run_id=source_run_id, epigraph_root=epigraph_root, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years, candidate_builders=builders)


def run_tr_v3_04_loop(*, run_id: str, source_run_id: str = "smoke-latent-blocks", epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    epigraph_root = Path(epigraph_root) if epigraph_root is not None else default_epigraph_root()
    observation_rows = build_observation_rows(epigraph_root, source_run_id)
    structural_inputs = load_phase2_structural_inputs(epigraph_root, source_run_id)
    direct_prior_features = build_direct_prior_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    hidden_driver_features = build_hidden_driver_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    splits = rolling_origin_splits(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)

    backbone_dynamic = DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0)
    backbone_observation = ObservationModelConfig(calibration_ridge=0.01, share_ridge_penalty=0.01, share_rho_clip=0.8, share_trend_scale=0.5)
    backbone_shock = ShockConfig(shock_phi=0.8, shock_scale=1.5, gate_z=1.5)
    backbone_damping = DampingConfig(min_dynamic_weight=0.15, gain_weight=0.5, residual_weight=0.15, horizon_decay=0.4, calibration_floor=0.2, calibration_gain_weight=0.7)
    backbone_prior = DirectPriorConfig(effect_scale=1.5, precision_scale=2.0, residual_ridge=0.1, max_effect=0.35)

    control_spec = {"dynamic_cfg": backbone_dynamic, "observation_cfg": backbone_observation, "shock_cfg": backbone_shock, "damping_cfg": backbone_damping, "prior_cfg": backbone_prior}
    control_rows = _run_spec_rows(observation_rows, splits, control_spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
    control_candidate = {"config": _candidate_config(control_spec), "model_contract": _candidate_model_contract(control_spec), "hazard_semantics": _candidate_hazard_semantics(control_spec), "rows": control_rows, "score": _score_rows(control_rows)}

    hidden_specs = [{"dynamic_cfg": backbone_dynamic, "observation_cfg": backbone_observation, "shock_cfg": backbone_shock, "damping_cfg": backbone_damping, "prior_cfg": backbone_prior, "hidden_cfg": hidden_cfg} for hidden_cfg in _hidden_driver_candidate_grid()]
    candidate_payloads: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(hidden_specs):
        rows = _run_spec_rows(observation_rows, splits, spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
        candidate_payloads.append({"candidate_index": int(candidate_index), "config": _candidate_config(spec), "model_contract": _candidate_model_contract(spec), "hazard_semantics": _candidate_hazard_semantics(spec), "rows": rows, "score": _score_rows(rows)})
    best_candidate = min(candidate_payloads, key=lambda row: (row["score"]["candidate_mean_mae"], row["score"]["candidate_worst_mae"]))
    best_candidate = dict(best_candidate)
    best_candidate["rows"] = _run_spec_rows(observation_rows, splits, hidden_specs[int(best_candidate["candidate_index"])], structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features, include_scoring_details=True)

    control_by_year = {_single_holdout_year(row): row for row in control_rows}
    best_by_year = {_single_holdout_year(row): row for row in best_candidate["rows"]}
    control_future_mean = _score_future_window(control_rows, start_year=2023)
    candidate_future_mean = _score_future_window(best_candidate["rows"], start_year=2023)
    control_2022 = float(control_by_year[2022]["candidate"]["mae"])
    candidate_2022 = float(best_by_year[2022]["candidate"]["mae"])
    allowed_2022_ceiling = float(control_2022 + max(0.025, 0.15 * control_2022))
    keep = candidate_future_mean < control_future_mean and candidate_2022 <= allowed_2022_ceiling
    decision = "keep" if keep else "revert"
    decision_reason = "Hidden-driver channels improved the 2023-2025 window without unacceptable 2022 degradation." if keep else "Hidden-driver channels did not improve 2023-2025 enough, or they degraded the 2022 split beyond the allowed ceiling."

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "family_name": "TR-V3-04",
        "source_run_id": source_run_id,
        "loop_variant": "evidence-to-model-loop",
        "model_contract": best_candidate.get("model_contract"),
        "hazard_semantics": best_candidate.get("hazard_semantics"),
        "phase2_prior_contract": dict(PHASE2_PRIOR_CONTRACT),
        "benchmark_contract": {"start_year": int(start_year), "end_year": int(end_year), "min_train_years": int(min_train_years), "horizon_years": int(horizon_years)},
        "phase2_context": {"direct_edge_count": len(structural_inputs.direct_edge_rows), "hidden_edge_count": len(structural_inputs.hidden_driver_rows), "multiscale_support_count": len(structural_inputs.multiscale_support_rows), "quarter_count": len(structural_inputs.quarter_axis), "block_count": len(structural_inputs.block_axis), "direct_prior_feature_count": int(sum(len(rows) for rows in direct_prior_features.values())), "hidden_driver_feature_count": int(sum(len(rows) for rows in hidden_driver_features.values()))},
        "split_count": len(splits),
        "candidate_count": len(hidden_specs),
        "decision": decision,
        "decision_reason": decision_reason,
        "data_provenance": _build_data_provenance_payload(observation_rows, best_candidate),
        "control_candidate": control_candidate,
        "best_candidate": best_candidate,
        "all_candidates": candidate_payloads,
        "tr_v3_04_gate": {
            "control_future_mean_mae": float(control_future_mean),
            "candidate_future_mean_mae": float(candidate_future_mean),
            "control_2022_mae": float(control_2022),
            "candidate_2022_mae": float(candidate_2022),
            "allowed_2022_ceiling": float(allowed_2022_ceiling),
        },
    }
    _attach_observation_score_ledger(payload, best_candidate)
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    write_json(analysis_dir / "tr_v3_04_autoresearch_report.json", payload)
    (analysis_dir / "tr_v3_04_autoresearch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload

def _score_year_window(rows: list[dict[str, Any]], years: set[int]) -> float:
    filtered = [float(row["candidate"]["mae"]) for row in rows if _single_holdout_year(row) in set(years)]
    return sum(filtered) / max(len(filtered), 1)


def _hidden_driver_damped_candidate_grid() -> list[HiddenDriverConfig]:
    rows: list[HiddenDriverConfig] = []
    for min_effect_weight in (0.35, 0.50, 0.70):
        for gain_weight in (0.00, 0.50, 1.00):
            for shock_penalty_weight in (0.60, 1.00, 1.40):
                for horizon_decay in (0.50, 0.70, 0.90):
                    rows.append(
                        HiddenDriverConfig(
                            precision_scale=0.25,
                            residual_ridge=0.01,
                            max_effect=0.10,
                            rank_cap=1,
                            min_effect_weight=min_effect_weight,
                            gain_weight=gain_weight,
                            shock_penalty_weight=shock_penalty_weight,
                            horizon_decay=horizon_decay,
                        )
                    )
    return rows




def _hidden_driver_transition_gate_grid() -> list[HiddenDriverConfig]:
    rows: list[HiddenDriverConfig] = []
    for gate_gain_weight in (1.0, 2.0, 3.0):
        for gate_recent_penalty in (0.0, 0.25, 0.5):
            for gate_threshold in (0.02, 0.05, 0.08, 0.12):
                for blocked_weight in (0.0, 0.15, 0.30):
                    rows.append(
                        HiddenDriverConfig(
                            precision_scale=0.25,
                            residual_ridge=0.01,
                            max_effect=0.10,
                            rank_cap=1,
                            min_effect_weight=1.0,
                            gain_weight=0.0,
                            shock_penalty_weight=0.0,
                            horizon_decay=1.0,
                            gate_gain_weight=gate_gain_weight,
                            gate_recent_penalty=gate_recent_penalty,
                            gate_threshold=gate_threshold,
                            blocked_weight=blocked_weight,
                        )
                    )
    return rows



def _transition_specific_hidden_weight_grid() -> list[HiddenDriverConfig]:
    rows: list[HiddenDriverConfig] = []
    u_to_d_weights = (1.0, 0.7)
    d_to_a_weights = (0.0, 0.3, 1.0)
    a_to_v_weights = (0.0, 0.3, 1.0)
    a_to_l_weights = (0.0, 0.3, 1.0)
    l_to_a_weights = (0.0, 0.3)
    for u_to_d in u_to_d_weights:
        for d_to_a in d_to_a_weights:
            for a_to_v in a_to_v_weights:
                for a_to_l in a_to_l_weights:
                    for l_to_a in l_to_a_weights:
                        rows.append(
                            HiddenDriverConfig(
                                precision_scale=0.25,
                                residual_ridge=0.01,
                                max_effect=0.10,
                                rank_cap=1,
                                min_effect_weight=1.0,
                                gain_weight=0.0,
                                shock_penalty_weight=0.0,
                                horizon_decay=1.0,
                                gate_gain_weight=0.0,
                                gate_recent_penalty=0.0,
                                gate_threshold=-1.0,
                                blocked_weight=1.0,
                                transition_weights={
                                    'U_to_D': u_to_d,
                                    'D_to_A': d_to_a,
                                    'A_to_V': a_to_v,
                                    'A_to_L': a_to_l,
                                    'L_to_A': l_to_a,
                                },
                            )
                        )
    return rows

def _write_early_history_partial_report(*, family_name: str, run_id: str, source_run_id: str, epigraph_root: Path | None, start_year: int, end_year: int, min_train_years: int, horizon_years: int) -> dict[str, Any]:
    epigraph_root = Path(epigraph_root) if epigraph_root is not None else default_epigraph_root()
    target_rows = build_early_partial_target_rows(epigraph_root, source_run_id, start_year=start_year, end_year=end_year)
    if not target_rows:
        raise ValueError("No early-history partial target rows were available.")
    split_defs = partial_observation_splits(target_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)
    split_rows: list[dict[str, Any]] = []
    for metric_name in sorted({str(row['metric_name']) for row in target_rows}):
        metric_rows = [dict(row) for row in target_rows if str(row['metric_name']) == metric_name]
        scale = max([abs(float(row['value'])) for row in metric_rows], default=1.0)
        for split in split_defs:
            train_targets = [row for row in metric_rows if int(row['year']) <= int(split['train_end_year'])]
            holdout_target = next((row for row in metric_rows if int(row['year']) == int(split['holdout_year'])), None)
            if holdout_target is None or len(train_targets) < int(min_train_years):
                continue
            no_signal_prediction = float(train_targets[-1]['value'])
            train_years = np.asarray([float(row['year']) for row in train_targets], dtype=np.float64)
            train_values = np.asarray([float(row['value']) for row in train_targets], dtype=np.float64)
            trend_prediction = float(np.mean(train_values)) if len(train_targets) < 2 else float(np.poly1d(np.polyfit(train_years, train_values, deg=1))(float(holdout_target['year'])))
            split_rows.append({"metric_name": metric_name, "train_end_year": int(split['train_end_year']), "holdout_year": int(split['holdout_year']), "variant_id": 'no_signal', "prediction": round(no_signal_prediction, 6), "target": round(float(holdout_target['value']), 6), "normalized_error": round(abs(no_signal_prediction - float(holdout_target['value'])) / max(scale, 1e-6), 6)})
            split_rows.append({"metric_name": metric_name, "train_end_year": int(split['train_end_year']), "holdout_year": int(split['holdout_year']), "variant_id": family_name, "prediction": round(trend_prediction, 6), "target": round(float(holdout_target['value']), 6), "normalized_error": round(abs(trend_prediction - float(holdout_target['value'])) / max(scale, 1e-6), 6)})
    if not split_rows:
        raise ValueError("No early-history partial-observation split rows were produced.")
    summary_rows = []
    for metric_name in sorted({str(row['metric_name']) for row in split_rows}):
        metric_split_rows = [row for row in split_rows if str(row['metric_name']) == metric_name]
        baseline_rows = [row for row in metric_split_rows if str(row['variant_id']) == 'no_signal']
        for variant_id in sorted({str(row['variant_id']) for row in metric_split_rows}):
            variant_rows = [row for row in metric_split_rows if str(row['variant_id']) == variant_id]
            summary_rows.append({
                "metric_name": metric_name,
                "variant_id": variant_id,
                "mean_normalized_error": round(float(np.mean([float(row['normalized_error']) for row in variant_rows])), 6),
                "median_normalized_error": round(float(np.median([float(row['normalized_error']) for row in variant_rows])), 6),
                "win_count_vs_no_signal": int(sum(float(candidate['normalized_error']) < float(baseline['normalized_error']) for candidate, baseline in zip(sorted(variant_rows, key=lambda row: (int(row['train_end_year']), int(row['holdout_year']))), sorted(baseline_rows, key=lambda row: (int(row['train_end_year']), int(row['holdout_year'])))))) if variant_id != 'no_signal' else 0,
            })
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "source_run_id": source_run_id, "family_name": family_name, "benchmark_kind": 'early_history_partial_observation', "quality_regime": 'partial_observation_not_comparable_to_full_transition_benchmark', "split_config": {"start_year": int(start_year), "end_year": int(end_year), "min_train_years": int(min_train_years), "horizon_years": int(horizon_years)}, "data_provenance": {"ladder": list(MISSING_DATA_LADDER), "target_rows": summarize_early_partial_provenance(target_rows)}, "target_rows": target_rows, "split_rows": split_rows, "summary_rows": summary_rows}
    analysis_dir = ensure_dir(sandbox_repo_root() / 'artifacts' / 'runs' / run_id / 'analysis')
    stem = family_name.lower().replace('-', '_') + '_early_history_partial_report'
    json_path = analysis_dir / f'{stem}.json'
    write_json(json_path, payload)
    lines = [f'# {family_name} Early-History Partial Observation Report', '', f'Source run: `{source_run_id}`', '', 'This report is a partial-observation benchmark for 2010-2016. It is not directly comparable to the full blocked-time transition benchmark because the early years do not have the same national transition targets.', '', '## Data Provenance', '', f"- Missing-data ladder: `{', '.join(MISSING_DATA_LADDER)}`", '', '| Slice | Exact | Bridge | Rule-based | Latent | Rejected |', '|---|---:|---:|---:|---:|---:|']
    _append_ladder_table(lines, 'Early partial target rows', dict((payload.get('data_provenance') or {}).get('target_rows', {}).get('row_tier_counts') or {}))
    lines.extend(['', '## Summary', '', '| Metric | Variant | Mean Normalized Error | Median Normalized Error | Wins vs No Signal |', '|---|---|---:|---:|---:|'])
    for row in sorted(summary_rows, key=lambda item: (str(item['metric_name']), float(item['mean_normalized_error']))):
        lines.append(f"| `{row['metric_name']}` | `{row['variant_id']}` | {float(row['mean_normalized_error']):.6f} | {float(row['median_normalized_error']):.6f} | {int(row['win_count_vs_no_signal'])} |")
    lines.extend(['', '## Split Table', '', '| Metric | Train End | Holdout | Variant | Prediction | Target | Normalized Error |', '|---|---:|---:|---|---:|---:|---:|'])
    for row in sorted(split_rows, key=lambda item: (str(item['metric_name']), int(item['train_end_year']), str(item['variant_id']))):
        lines.append(f"| `{row['metric_name']}` | {int(row['train_end_year'])} | {int(row['holdout_year'])} | `{row['variant_id']}` | {float(row['prediction']):.6f} | {float(row['target']):.6f} | {float(row['normalized_error']):.6f} |")
    md_path = analysis_dir / f'{stem}.md'
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return payload


def run_tr_v3_04b_loop(*, run_id: str, source_run_id: str = 'smoke-latent-blocks', epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    epigraph_root = Path(epigraph_root) if epigraph_root is not None else default_epigraph_root()
    observation_rows = build_observation_rows(epigraph_root, source_run_id)
    structural_inputs = load_phase2_structural_inputs(epigraph_root, source_run_id)
    direct_prior_features = build_direct_prior_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    hidden_driver_features = build_hidden_driver_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    splits = rolling_origin_splits(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)
    backbone_dynamic = DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0)
    backbone_observation = ObservationModelConfig(calibration_ridge=0.01, share_ridge_penalty=0.01, share_rho_clip=0.8, share_trend_scale=0.5)
    backbone_shock = ShockConfig(shock_phi=0.8, shock_scale=1.5, gate_z=1.5)
    backbone_damping = DampingConfig(min_dynamic_weight=0.15, gain_weight=0.5, residual_weight=0.15, horizon_decay=0.4, calibration_floor=0.2, calibration_gain_weight=0.7)
    backbone_prior = DirectPriorConfig(effect_scale=1.5, precision_scale=2.0, residual_ridge=0.1, max_effect=0.35)
    control_hidden = HiddenDriverConfig(precision_scale=0.25, residual_ridge=0.01, max_effect=0.10, rank_cap=1)
    control_spec = {'dynamic_cfg': backbone_dynamic, 'observation_cfg': backbone_observation, 'shock_cfg': backbone_shock, 'damping_cfg': backbone_damping, 'prior_cfg': backbone_prior, 'hidden_cfg': control_hidden}
    control_rows = _run_spec_rows(observation_rows, splits, control_spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
    control_candidate = {'config': _candidate_config(control_spec), 'model_contract': _candidate_model_contract(control_spec), 'hazard_semantics': _candidate_hazard_semantics(control_spec), 'rows': control_rows, 'score': _score_rows(control_rows)}
    candidate_specs = [{'dynamic_cfg': backbone_dynamic, 'observation_cfg': backbone_observation, 'shock_cfg': backbone_shock, 'damping_cfg': backbone_damping, 'prior_cfg': backbone_prior, 'hidden_cfg': hidden_cfg} for hidden_cfg in _hidden_driver_damped_candidate_grid()]
    candidate_payloads: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(candidate_specs):
        rows = _run_spec_rows(observation_rows, splits, spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
        candidate_payloads.append({'candidate_index': int(candidate_index), 'config': _candidate_config(spec), 'model_contract': _candidate_model_contract(spec), 'hazard_semantics': _candidate_hazard_semantics(spec), 'rows': rows, 'score': _score_rows(rows)})
    control_by_year = {_single_holdout_year(row): row for row in control_rows}
    control_tail_mean = _score_year_window(control_rows, {2024, 2025})
    control_2023 = float(control_by_year[2023]['candidate']['mae'])
    control_2022 = float(control_by_year[2022]['candidate']['mae'])
    allowed_2022_ceiling = float(control_2022 + max(0.02, 0.10 * control_2022))

    def _gate_payload(candidate: dict[str, Any]) -> dict[str, float]:
        by_year = {_single_holdout_year(row): row for row in candidate['rows']}
        return {
            'candidate_tail_mean_mae': float(_score_year_window(candidate['rows'], {2024, 2025})),
            'candidate_2023_mae': float(by_year[2023]['candidate']['mae']),
            'candidate_2022_mae': float(by_year[2022]['candidate']['mae']),
        }

    keepable_candidates: list[tuple[dict[str, Any], dict[str, float]]] = []
    gated_candidates: list[tuple[dict[str, Any], dict[str, float]]] = []
    for candidate in candidate_payloads:
        gate = _gate_payload(candidate)
        gated_candidates.append((candidate, gate))
        if gate['candidate_tail_mean_mae'] <= control_tail_mean and gate['candidate_2023_mae'] <= control_2023 and gate['candidate_2022_mae'] <= allowed_2022_ceiling:
            keepable_candidates.append((candidate, gate))

    if keepable_candidates:
        best_candidate, gate_values = min(keepable_candidates, key=lambda item: (item[0]['score']['candidate_mean_mae'], item[0]['score']['candidate_worst_mae']))
        decision = 'keep'
        decision_reason = 'Split-aware hidden damping preserved the 2024-2025 gain, improved 2023, and kept 2022 within tolerance.'
    else:
        best_candidate, gate_values = min(gated_candidates, key=lambda item: (item[0]['score']['candidate_mean_mae'], item[0]['score']['candidate_worst_mae']))
        decision = 'revert'
        decision_reason = 'Split-aware hidden damping failed to protect the 2024-2025 gain, failed to improve 2023, or degraded 2022 beyond tolerance.'

    best_candidate = dict(best_candidate)
    best_candidate['rows'] = _run_spec_rows(observation_rows, splits, candidate_specs[int(best_candidate['candidate_index'])], structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features, include_scoring_details=True)
    candidate_tail_mean = float(gate_values['candidate_tail_mean_mae'])
    candidate_2023 = float(gate_values['candidate_2023_mae'])
    candidate_2022 = float(gate_values['candidate_2022_mae'])
    payload = {'generated_at': datetime.now(timezone.utc).isoformat(), 'run_id': run_id, 'family_name': 'TR-V3-04b', 'source_run_id': source_run_id, 'loop_variant': 'evidence-to-model-loop', 'model_contract': best_candidate.get('model_contract'), 'hazard_semantics': best_candidate.get('hazard_semantics'), 'phase2_prior_contract': dict(PHASE2_PRIOR_CONTRACT), 'benchmark_contract': {'start_year': int(start_year), 'end_year': int(end_year), 'min_train_years': int(min_train_years), 'horizon_years': int(horizon_years)}, 'phase2_context': {'direct_edge_count': len(structural_inputs.direct_edge_rows), 'hidden_edge_count': len(structural_inputs.hidden_driver_rows), 'multiscale_support_count': len(structural_inputs.multiscale_support_rows), 'quarter_count': len(structural_inputs.quarter_axis), 'block_count': len(structural_inputs.block_axis), 'direct_prior_feature_count': int(sum(len(rows) for rows in direct_prior_features.values())), 'hidden_driver_feature_count': int(sum(len(rows) for rows in hidden_driver_features.values()))}, 'split_count': len(splits), 'candidate_count': len(candidate_specs), 'decision': decision, 'decision_reason': decision_reason, 'data_provenance': _build_data_provenance_payload(observation_rows, best_candidate), 'control_candidate': control_candidate, 'best_candidate': best_candidate, 'all_candidates': candidate_payloads, 'tr_v3_04b_gate': {'control_tail_mean_mae': float(control_tail_mean), 'candidate_tail_mean_mae': float(candidate_tail_mean), 'control_2023_mae': float(control_2023), 'candidate_2023_mae': float(candidate_2023), 'control_2022_mae': float(control_2022), 'candidate_2022_mae': float(candidate_2022), 'allowed_2022_ceiling': float(allowed_2022_ceiling)}}
    _attach_observation_score_ledger(payload, best_candidate)
    analysis_dir = ensure_dir(sandbox_repo_root() / 'artifacts' / 'runs' / run_id / 'analysis')
    write_json(analysis_dir / 'tr_v3_04b_autoresearch_report.json', payload)
    (analysis_dir / 'tr_v3_04b_autoresearch_report.md').write_text(_markdown_report(payload), encoding='utf-8')
    return payload


def run_tr_v3_04b_early_history_partial(*, run_id: str, source_run_id: str = 'smoke-latent-blocks', epigraph_root: Path | None = None, start_year: int = 2010, end_year: int = 2016, min_train_years: int = 3, horizon_years: int = 1) -> dict[str, Any]:
    return _write_early_history_partial_report(family_name='TR-V3-04b', run_id=run_id, source_run_id=source_run_id, epigraph_root=epigraph_root, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)



def run_tr_v3_04c_loop(*, run_id: str, source_run_id: str = 'smoke-latent-blocks', epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    epigraph_root = Path(epigraph_root) if epigraph_root is not None else default_epigraph_root()
    observation_rows = build_observation_rows(epigraph_root, source_run_id)
    structural_inputs = load_phase2_structural_inputs(epigraph_root, source_run_id)
    direct_prior_features = build_direct_prior_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    hidden_driver_features = build_hidden_driver_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    splits = rolling_origin_splits(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)
    backbone_dynamic = DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0)
    backbone_observation = ObservationModelConfig(calibration_ridge=0.01, share_ridge_penalty=0.01, share_rho_clip=0.8, share_trend_scale=0.5)
    backbone_shock = ShockConfig(shock_phi=0.8, shock_scale=1.5, gate_z=1.5)
    backbone_damping = DampingConfig(min_dynamic_weight=0.15, gain_weight=0.5, residual_weight=0.15, horizon_decay=0.4, calibration_floor=0.2, calibration_gain_weight=0.7)
    backbone_prior = DirectPriorConfig(effect_scale=1.5, precision_scale=2.0, residual_ridge=0.1, max_effect=0.35)
    control_hidden = HiddenDriverConfig(precision_scale=0.25, residual_ridge=0.01, max_effect=0.10, rank_cap=1)
    control_spec = {'dynamic_cfg': backbone_dynamic, 'observation_cfg': backbone_observation, 'shock_cfg': backbone_shock, 'damping_cfg': backbone_damping, 'prior_cfg': backbone_prior, 'hidden_cfg': control_hidden}
    control_rows = _run_spec_rows(observation_rows, splits, control_spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
    control_candidate = {'config': _candidate_config(control_spec), 'model_contract': _candidate_model_contract(control_spec), 'hazard_semantics': _candidate_hazard_semantics(control_spec), 'rows': control_rows, 'score': _score_rows(control_rows)}
    candidate_specs = [{'dynamic_cfg': backbone_dynamic, 'observation_cfg': backbone_observation, 'shock_cfg': backbone_shock, 'damping_cfg': backbone_damping, 'prior_cfg': backbone_prior, 'hidden_cfg': hidden_cfg} for hidden_cfg in _hidden_driver_transition_gate_grid()]
    candidate_payloads: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(candidate_specs):
        rows = _run_spec_rows(observation_rows, splits, spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
        candidate_payloads.append({'candidate_index': int(candidate_index), 'config': _candidate_config(spec), 'model_contract': _candidate_model_contract(spec), 'hazard_semantics': _candidate_hazard_semantics(spec), 'rows': rows, 'score': _score_rows(rows)})
    control_by_year = {_single_holdout_year(row): row for row in control_rows}
    control_tail_mean = _score_year_window(control_rows, {2024, 2025})
    control_2023 = float(control_by_year[2023]['candidate']['mae'])
    control_2022 = float(control_by_year[2022]['candidate']['mae'])
    allowed_2022_ceiling = float(control_2022 + max(0.02, 0.10 * control_2022))

    def _gate_payload(candidate: dict[str, Any]) -> dict[str, float]:
        by_year = {_single_holdout_year(row): row for row in candidate['rows']}
        return {
            'candidate_tail_mean_mae': float(_score_year_window(candidate['rows'], {2024, 2025})),
            'candidate_2023_mae': float(by_year[2023]['candidate']['mae']),
            'candidate_2022_mae': float(by_year[2022]['candidate']['mae']),
        }

    keepable_candidates: list[tuple[dict[str, Any], dict[str, float]]] = []
    gated_candidates: list[tuple[dict[str, Any], dict[str, float]]] = []
    for candidate in candidate_payloads:
        gate = _gate_payload(candidate)
        gated_candidates.append((candidate, gate))
        if gate['candidate_tail_mean_mae'] <= control_tail_mean and gate['candidate_2023_mae'] <= control_2023 and gate['candidate_2022_mae'] <= allowed_2022_ceiling:
            keepable_candidates.append((candidate, gate))

    if keepable_candidates:
        best_candidate, gate_values = min(keepable_candidates, key=lambda item: (item[0]['score']['candidate_mean_mae'], item[0]['score']['candidate_worst_mae']))
        decision = 'keep'
        decision_reason = 'Transition-specific hidden gating preserved the 2024-2025 gain, improved 2023, and kept 2022 within tolerance.'
    else:
        best_candidate, gate_values = min(gated_candidates, key=lambda item: (item[0]['score']['candidate_mean_mae'], item[0]['score']['candidate_worst_mae']))
        decision = 'revert'
        decision_reason = 'Transition-specific hidden gating failed to protect the 2024-2025 gain, failed to improve 2023, or degraded 2022 beyond tolerance.'

    best_candidate = dict(best_candidate)
    best_candidate['rows'] = _run_spec_rows(observation_rows, splits, candidate_specs[int(best_candidate['candidate_index'])], structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features, include_scoring_details=True)
    candidate_tail_mean = float(gate_values['candidate_tail_mean_mae'])
    candidate_2023 = float(gate_values['candidate_2023_mae'])
    candidate_2022 = float(gate_values['candidate_2022_mae'])
    payload = {'generated_at': datetime.now(timezone.utc).isoformat(), 'run_id': run_id, 'family_name': 'TR-V3-04c', 'source_run_id': source_run_id, 'loop_variant': 'evidence-to-model-loop', 'model_contract': best_candidate.get('model_contract'), 'hazard_semantics': best_candidate.get('hazard_semantics'), 'phase2_prior_contract': dict(PHASE2_PRIOR_CONTRACT), 'benchmark_contract': {'start_year': int(start_year), 'end_year': int(end_year), 'min_train_years': int(min_train_years), 'horizon_years': int(horizon_years)}, 'phase2_context': {'direct_edge_count': len(structural_inputs.direct_edge_rows), 'hidden_edge_count': len(structural_inputs.hidden_driver_rows), 'multiscale_support_count': len(structural_inputs.multiscale_support_rows), 'quarter_count': len(structural_inputs.quarter_axis), 'block_count': len(structural_inputs.block_axis), 'direct_prior_feature_count': int(sum(len(rows) for rows in direct_prior_features.values())), 'hidden_driver_feature_count': int(sum(len(rows) for rows in hidden_driver_features.values()))}, 'split_count': len(splits), 'candidate_count': len(candidate_specs), 'decision': decision, 'decision_reason': decision_reason, 'data_provenance': _build_data_provenance_payload(observation_rows, best_candidate), 'control_family_name': 'TR-V3-04', 'control_candidate': control_candidate, 'best_candidate': best_candidate, 'all_candidates': candidate_payloads, 'hidden_channel_gate': {'control_tail_mean_mae': float(control_tail_mean), 'candidate_tail_mean_mae': float(candidate_tail_mean), 'control_2023_mae': float(control_2023), 'candidate_2023_mae': float(candidate_2023), 'control_2022_mae': float(control_2022), 'candidate_2022_mae': float(candidate_2022), 'allowed_2022_ceiling': float(allowed_2022_ceiling)}}
    _attach_observation_score_ledger(payload, best_candidate)
    analysis_dir = ensure_dir(sandbox_repo_root() / 'artifacts' / 'runs' / run_id / 'analysis')
    write_json(analysis_dir / 'tr_v3_04c_autoresearch_report.json', payload)
    (analysis_dir / 'tr_v3_04c_autoresearch_report.md').write_text(_markdown_report(payload), encoding='utf-8')
    return payload


def run_tr_v3_04c_early_history_partial(*, run_id: str, source_run_id: str = 'smoke-latent-blocks', epigraph_root: Path | None = None, start_year: int = 2010, end_year: int = 2016, min_train_years: int = 3, horizon_years: int = 1) -> dict[str, Any]:
    return _write_early_history_partial_report(family_name='TR-V3-04c', run_id=run_id, source_run_id=source_run_id, epigraph_root=epigraph_root, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)


def run_tr_v3_04d_loop(*, run_id: str, source_run_id: str = 'smoke-latent-blocks', epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    epigraph_root = Path(epigraph_root) if epigraph_root is not None else default_epigraph_root()
    observation_rows = build_observation_rows(epigraph_root, source_run_id)
    structural_inputs = load_phase2_structural_inputs(epigraph_root, source_run_id)
    direct_prior_features = build_direct_prior_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    hidden_driver_features = build_hidden_driver_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    splits = rolling_origin_splits(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)
    backbone_dynamic = DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0)
    backbone_observation = ObservationModelConfig(calibration_ridge=0.01, share_ridge_penalty=0.01, share_rho_clip=0.8, share_trend_scale=0.5)
    backbone_shock = ShockConfig(shock_phi=0.8, shock_scale=1.5, gate_z=1.5)
    backbone_damping = DampingConfig(min_dynamic_weight=0.15, gain_weight=0.5, residual_weight=0.15, horizon_decay=0.4, calibration_floor=0.2, calibration_gain_weight=0.7)
    backbone_prior = DirectPriorConfig(effect_scale=1.5, precision_scale=2.0, residual_ridge=0.1, max_effect=0.35)
    control_hidden = HiddenDriverConfig(precision_scale=0.25, residual_ridge=0.01, max_effect=0.10, rank_cap=1)
    control_spec = {'dynamic_cfg': backbone_dynamic, 'observation_cfg': backbone_observation, 'shock_cfg': backbone_shock, 'damping_cfg': backbone_damping, 'prior_cfg': backbone_prior, 'hidden_cfg': control_hidden}
    control_rows = _run_spec_rows(observation_rows, splits, control_spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
    control_candidate = {'config': _candidate_config(control_spec), 'model_contract': _candidate_model_contract(control_spec), 'hazard_semantics': _candidate_hazard_semantics(control_spec), 'rows': control_rows, 'score': _score_rows(control_rows)}
    candidate_specs = [{'dynamic_cfg': backbone_dynamic, 'observation_cfg': backbone_observation, 'shock_cfg': backbone_shock, 'damping_cfg': backbone_damping, 'prior_cfg': backbone_prior, 'hidden_cfg': hidden_cfg} for hidden_cfg in _transition_specific_hidden_weight_grid()]
    candidate_payloads: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(candidate_specs):
        rows = _run_spec_rows(observation_rows, splits, spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
        candidate_payloads.append({'candidate_index': int(candidate_index), 'config': _candidate_config(spec), 'model_contract': _candidate_model_contract(spec), 'hazard_semantics': _candidate_hazard_semantics(spec), 'rows': rows, 'score': _score_rows(rows)})
    control_by_year = {_single_holdout_year(row): row for row in control_rows}
    control_tail_mean = _score_year_window(control_rows, {2024, 2025})
    control_2023 = float(control_by_year[2023]['candidate']['mae'])
    control_2022 = float(control_by_year[2022]['candidate']['mae'])
    allowed_2022_ceiling = float(control_2022 + max(0.02, 0.10 * control_2022))

    def _gate_payload(candidate: dict[str, Any]) -> dict[str, float]:
        by_year = {_single_holdout_year(row): row for row in candidate['rows']}
        return {
            'candidate_tail_mean_mae': float(_score_year_window(candidate['rows'], {2024, 2025})),
            'candidate_2023_mae': float(by_year[2023]['candidate']['mae']),
            'candidate_2022_mae': float(by_year[2022]['candidate']['mae']),
        }

    keepable_candidates: list[tuple[dict[str, Any], dict[str, float]]] = []
    gated_candidates: list[tuple[dict[str, Any], dict[str, float]]] = []
    for candidate in candidate_payloads:
        gate = _gate_payload(candidate)
        gated_candidates.append((candidate, gate))
        if gate['candidate_tail_mean_mae'] <= control_tail_mean and gate['candidate_2023_mae'] <= control_2023 and gate['candidate_2022_mae'] <= allowed_2022_ceiling:
            keepable_candidates.append((candidate, gate))

    if keepable_candidates:
        best_candidate, gate_values = min(keepable_candidates, key=lambda item: (item[0]['score']['candidate_mean_mae'], item[0]['score']['candidate_worst_mae']))
        decision = 'keep'
        decision_reason = 'Transition-specific hidden weights preserved the 2024-2025 gain, improved 2023, and kept 2022 within tolerance.'
    else:
        best_candidate, gate_values = min(gated_candidates, key=lambda item: (item[0]['score']['candidate_mean_mae'], item[0]['score']['candidate_worst_mae']))
        decision = 'revert'
        decision_reason = 'Transition-specific hidden weights failed to protect the 2024-2025 gain, failed to improve 2023, or degraded 2022 beyond tolerance.'

    best_candidate = dict(best_candidate)
    best_candidate['rows'] = _run_spec_rows(observation_rows, splits, candidate_specs[int(best_candidate['candidate_index'])], structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features, include_scoring_details=True)
    candidate_tail_mean = float(gate_values['candidate_tail_mean_mae'])
    candidate_2023 = float(gate_values['candidate_2023_mae'])
    candidate_2022 = float(gate_values['candidate_2022_mae'])
    payload = {'generated_at': datetime.now(timezone.utc).isoformat(), 'run_id': run_id, 'family_name': 'TR-V3-04d', 'source_run_id': source_run_id, 'loop_variant': 'evidence-to-model-loop', 'model_contract': best_candidate.get('model_contract'), 'hazard_semantics': best_candidate.get('hazard_semantics'), 'phase2_prior_contract': dict(PHASE2_PRIOR_CONTRACT), 'benchmark_contract': {'start_year': int(start_year), 'end_year': int(end_year), 'min_train_years': int(min_train_years), 'horizon_years': int(horizon_years)}, 'phase2_context': {'direct_edge_count': len(structural_inputs.direct_edge_rows), 'hidden_edge_count': len(structural_inputs.hidden_driver_rows), 'multiscale_support_count': len(structural_inputs.multiscale_support_rows), 'quarter_count': len(structural_inputs.quarter_axis), 'block_count': len(structural_inputs.block_axis), 'direct_prior_feature_count': int(sum(len(rows) for rows in direct_prior_features.values())), 'hidden_driver_feature_count': int(sum(len(rows) for rows in hidden_driver_features.values()))}, 'split_count': len(splits), 'candidate_count': len(candidate_specs), 'decision': decision, 'decision_reason': decision_reason, 'data_provenance': _build_data_provenance_payload(observation_rows, best_candidate), 'control_family_name': 'TR-V3-04', 'control_candidate': control_candidate, 'best_candidate': best_candidate, 'all_candidates': candidate_payloads, 'hidden_channel_gate': {'control_tail_mean_mae': float(control_tail_mean), 'candidate_tail_mean_mae': float(candidate_tail_mean), 'control_2023_mae': float(control_2023), 'candidate_2023_mae': float(candidate_2023), 'control_2022_mae': float(control_2022), 'candidate_2022_mae': float(candidate_2022), 'allowed_2022_ceiling': float(allowed_2022_ceiling)}}
    _attach_observation_score_ledger(payload, best_candidate)
    analysis_dir = ensure_dir(sandbox_repo_root() / 'artifacts' / 'runs' / run_id / 'analysis')
    write_json(analysis_dir / 'tr_v3_04d_autoresearch_report.json', payload)
    (analysis_dir / 'tr_v3_04d_autoresearch_report.md').write_text(_markdown_report(payload), encoding='utf-8')
    return payload


def run_tr_v3_04d_early_history_partial(*, run_id: str, source_run_id: str = 'smoke-latent-blocks', epigraph_root: Path | None = None, start_year: int = 2010, end_year: int = 2016, min_train_years: int = 3, horizon_years: int = 1) -> dict[str, Any]:
    return _write_early_history_partial_report(family_name='TR-V3-04d', run_id=run_id, source_run_id=source_run_id, epigraph_root=epigraph_root, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)


def run_tr_v3_05a_inflow_loop(*, run_id: str, source_run_id: str = 'smoke-latent-blocks', epigraph_root: Path | None = None, start_year: int = 2017, end_year: int = 2025, min_train_years: int = 5, horizon_years: int = 1) -> dict[str, Any]:
    epigraph_root = Path(epigraph_root) if epigraph_root is not None else default_epigraph_root()
    observation_rows = build_observation_rows(epigraph_root, source_run_id)
    structural_inputs = load_phase2_structural_inputs(epigraph_root, source_run_id)
    direct_prior_features = build_direct_prior_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    hidden_driver_features = build_hidden_driver_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    splits = rolling_origin_splits(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)

    backbone_dynamic = DynamicBaselineConfig(ridge_penalty=0.01, rho_clip=0.8, trend_scale=1.0)
    backbone_observation = ObservationModelConfig(calibration_ridge=0.01, share_ridge_penalty=0.01, share_rho_clip=0.8, share_trend_scale=0.5)
    backbone_shock = ShockConfig(shock_phi=0.8, shock_scale=1.5, gate_z=1.5)
    backbone_damping = DampingConfig(min_dynamic_weight=0.15, gain_weight=0.5, residual_weight=0.15, horizon_decay=0.4, calibration_floor=0.2, calibration_gain_weight=0.7)
    backbone_prior = DirectPriorConfig(effect_scale=1.5, precision_scale=2.0, residual_ridge=0.1, max_effect=0.35)
    backbone_hidden = HiddenDriverConfig(precision_scale=0.25, residual_ridge=0.01, max_effect=0.10, rank_cap=1)

    control_spec = {
        'dynamic_cfg': backbone_dynamic,
        'observation_cfg': backbone_observation,
        'shock_cfg': backbone_shock,
        'damping_cfg': backbone_damping,
        'prior_cfg': backbone_prior,
        'hidden_cfg': backbone_hidden,
    }
    control_rows = _run_spec_rows(observation_rows, splits, control_spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
    control_candidate = {
        'config': _candidate_config(control_spec),
        'model_contract': _candidate_model_contract(control_spec),
        'hazard_semantics': _candidate_hazard_semantics(control_spec),
        'rows': control_rows,
        'score': _score_rows(control_rows),
    }

    candidate_specs = [
        {
            'dynamic_cfg': backbone_dynamic,
            'incidence_cfg': incidence_cfg,
            'observation_cfg': backbone_observation,
            'shock_cfg': backbone_shock,
            'damping_cfg': backbone_damping,
            'prior_cfg': backbone_prior,
            'hidden_cfg': backbone_hidden,
        }
        for incidence_cfg in _anchored_incidence_candidates()
    ]
    candidate_payloads: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(candidate_specs):
        rows = _run_spec_rows(observation_rows, splits, spec, structural_inputs=structural_inputs, direct_prior_features=direct_prior_features, hidden_driver_features=hidden_driver_features)
        candidate_payloads.append({
            'candidate_index': int(candidate_index),
            'config': _candidate_config(spec),
            'model_contract': _candidate_model_contract(spec),
            'hazard_semantics': _candidate_hazard_semantics(spec),
            'rows': rows,
            'score': _score_rows(rows),
        })

    best_candidate = min(candidate_payloads, key=lambda row: (row['score']['candidate_mean_mae'], row['score']['candidate_worst_mae']))
    best_candidate = dict(best_candidate)
    best_spec = candidate_specs[int(best_candidate['candidate_index'])]
    best_candidate['rows'] = _run_spec_rows(
        observation_rows,
        splits,
        best_spec,
        structural_inputs=structural_inputs,
        direct_prior_features=direct_prior_features,
        hidden_driver_features=hidden_driver_features,
        include_scoring_details=True,
    )

    control_score = dict(control_candidate['score'])
    best_score = dict(best_candidate['score'])
    keep = float(best_score['candidate_mean_mae']) < float(control_score['candidate_mean_mae']) and float(best_score['candidate_worst_mae']) <= (1.10 * float(control_score['candidate_worst_mae']))
    decision = 'keep' if keep else 'revert'
    decision_reason = (
        'Latent inflow branch beat the integrated reference on blocked-time mean MAE without unacceptable worst-split regression.'
        if keep
        else 'Latent inflow branch did not beat the integrated reference strongly enough under the blocked-time gate.'
    )

    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'run_id': run_id,
        'family_name': 'TR-V3-05a',
        'source_run_id': source_run_id,
        'loop_variant': 'evidence-to-model-loop',
        'model_contract': best_candidate.get('model_contract'),
        'hazard_semantics': best_candidate.get('hazard_semantics'),
        'phase2_prior_contract': dict(PHASE2_PRIOR_CONTRACT),
        'benchmark_contract': {'start_year': int(start_year), 'end_year': int(end_year), 'min_train_years': int(min_train_years), 'horizon_years': int(horizon_years)},
        'phase2_context': {
            'direct_edge_count': len(structural_inputs.direct_edge_rows),
            'hidden_edge_count': len(structural_inputs.hidden_driver_rows),
            'multiscale_support_count': len(structural_inputs.multiscale_support_rows),
            'quarter_count': len(structural_inputs.quarter_axis),
            'block_count': len(structural_inputs.block_axis),
            'direct_prior_feature_count': int(sum(len(rows) for rows in direct_prior_features.values())),
            'hidden_driver_feature_count': int(sum(len(rows) for rows in hidden_driver_features.values())),
        },
        'split_count': len(splits),
        'candidate_count': len(candidate_specs),
        'decision': decision,
        'decision_reason': decision_reason,
        'data_provenance': _build_data_provenance_payload(observation_rows, best_candidate),
        'control_family_name': 'TR-V3-04d_reference',
        'control_candidate': control_candidate,
        'best_candidate': best_candidate,
        'all_candidates': candidate_payloads,
        'inflow_gate': {
            'control_mean_mae': float(control_score['candidate_mean_mae']),
            'candidate_mean_mae': float(best_score['candidate_mean_mae']),
            'control_worst_mae': float(control_score['candidate_worst_mae']),
            'candidate_worst_mae': float(best_score['candidate_worst_mae']),
        },
    }
    _attach_observation_score_ledger(payload, best_candidate)
    analysis_dir = ensure_dir(sandbox_repo_root() / 'artifacts' / 'runs' / run_id / 'analysis')
    write_json(analysis_dir / 'tr_v3_05a_inflow_report.json', payload)
    (analysis_dir / 'tr_v3_05a_inflow_report.md').write_text(_markdown_report(payload), encoding='utf-8')
    return payload
