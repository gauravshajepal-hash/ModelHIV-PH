from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3._lineage.national_reset_core import PRIMARY_METRICS
from epigraph_ph.runtime import read_json, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .decomposition import (
    CHANNEL_NAMES,
    DECOMP01E_EXPERIMENT_ID,
    _discover_latest_decomp_experiment,
    _simulate_holdout_from_hazard_map,
    _load_decomp_01a_reference,
)
from .analytics import _collapse_kp_distribution, _factor_transition_rows
from .numeric_policy import numerical_guard_entry
from .registry import KP_COLLAPSED_NAMES, TRANSITION_NAMES
from .sources import load_transition_research_inputs
from .transition_engine import (
    DOWNSTREAM_TRANSITIONS,
    STATE_NAMES,
    _build_observation_payload,
    _diagnosis_flow_mae,
    _holdout_rows,
    _holdout_state_rows_from_reference,
    _holdout_years_from_reference,
    _load_transition_experiment_reference,
    _metric_scales,
    _normalized_mae,
    _quarter_factor_surface,
    _reference_holdout_hazards,
    _smape,
    _state_trajectory_rows,
    _train_rows,
)

PEAK01A_EXPERIMENT_ID = "PEAK-01A-regional-kp-window-detector"
PEAK01B_EXPERIMENT_ID = "PEAK-01B-detector-gated-fused-forecast"
PEAK01C_EXPERIMENT_ID = "PEAK-01C-region-only-window-detector"
PEAK01D_EXPERIMENT_ID = "PEAK-01D-region-only-gated-fused-forecast"
PEAK01E_EXPERIMENT_ID = "PEAK-01E-region-plus-kp-modifier-detector"
PEAK01F_EXPERIMENT_ID = "PEAK-01F-region-plus-kp-modifier-gated-fused-forecast"
PEAK_CHANNELS: tuple[str, ...] = ("shock", "residual")


def _quarter_from_month(month_label: str) -> str:
    year_text, month_text = str(month_label).split("-", 1)
    return f"{int(year_text):04d}-Q{((int(month_text[:2]) - 1) // 3) + 1}"


def _quarter_region_factor_surface(ctx: TransitionResearchContext) -> dict[str, dict[str, dict[str, float]]]:
    inputs = load_transition_research_inputs(ctx)
    grouped_months: dict[str, list[int]] = defaultdict(list)
    for month_idx, month_label in enumerate(inputs.month_axis):
        grouped_months[_quarter_from_month(str(month_label))].append(month_idx)
    surface: dict[str, dict[str, dict[str, float]]] = {}
    for region_idx, region_name in enumerate(inputs.region_axis):
        factor_map: dict[str, dict[str, float]] = {}
        for factor_id, factor_idx in inputs.factor_index.items():
            values = np.asarray(inputs.region_tensor[region_idx, :, factor_idx], dtype=np.float64)
            factor_map[factor_id] = {
                quarter: float(np.mean(values[month_indices])) if month_indices else 0.0
                for quarter, month_indices in grouped_months.items()
            }
        surface[str(region_name)] = factor_map
    return surface


def _zscore_by_quarter(values_by_quarter: dict[str, float], train_quarters: list[str], eps: float) -> dict[str, float]:
    train_values = [float(values_by_quarter.get(quarter) or 0.0) for quarter in train_quarters]
    mean_value = float(np.mean(train_values)) if train_values else 0.0
    std_value = float(np.std(train_values)) if train_values else 0.0
    if std_value <= eps:
        return {str(quarter): 0.0 for quarter in values_by_quarter}
    return {
        str(quarter): (float(value) - mean_value) / std_value
        for quarter, value in values_by_quarter.items()
    }


def _run_peak_detector(
    ctx: TransitionResearchContext,
    *,
    include_kp: bool,
    experiment_label: str,
    heatmap_title: str,
    decision_reason: str,
) -> dict[str, Any]:
    payload = _detector_payload(ctx, include_kp=include_kp)
    detector_json = {
        "quarter_axis": payload["quarter_axis"],
        "pair_rows": payload["pair_rows"],
        "quarter_rows": payload["quarter_rows"],
    }
    write_json(ctx.experiment_dir / "peak_window_detector.json", detector_json)
    fig, ax = plt.subplots()
    image = ax.imshow(payload["gate_matrix"], aspect="auto")
    ax.set_title(heatmap_title)
    ax.set_xticks(np.arange(len(payload["quarter_axis"])), labels=list(payload["quarter_axis"]))
    ax.set_yticks(np.arange(len(payload["heatmap_labels"])), labels=list(payload["heatmap_labels"]))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "peak_window_heatmap.png")
    plt.close(fig)
    decision = {
        "completed": True,
        "keep": bool(payload["coverage_summary"]["supported_detector_pair_count"] > 0),
        "reason": decision_reason,
    }
    decomp_01a_reference = _load_decomp_01a_reference()
    mech_01e_reference = _load_transition_experiment_reference("MECH-01E-anchored-downstream-residual-helpers")
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "detector_label": experiment_label,
            "detector_mode": str(payload["coverage_summary"]["detector_mode"]),
            "decomp_01a_reference_run_id": str(decomp_01a_reference["reference_run_id"]),
            "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        },
        coverage_summary=payload["coverage_summary"],
        decision=decision,
        numeric_justification=payload["numeric_justification"],
    )
    return {
        "artifacts": artifacts,
        "decision": decision,
        **payload,
    }


def run_peak_01a(ctx: TransitionResearchContext) -> dict[str, Any]:
    return _run_peak_detector(
        ctx,
        include_kp=True,
        experiment_label="PEAK-01A",
        heatmap_title="PEAK-01A Regional/KP Peak-Window Gates",
        decision_reason="Regional and KP-conditioned peak-window detector fitted successfully with explicit out-of-sample skill gating.",
    )


def run_peak_01c(ctx: TransitionResearchContext) -> dict[str, Any]:
    return _run_peak_detector(
        ctx,
        include_kp=False,
        experiment_label="PEAK-01C",
        heatmap_title="PEAK-01C Region-Only Peak-Window Gates",
        decision_reason="Region-only peak-window detector fitted successfully with explicit out-of-sample skill gating.",
    )


def _run_peak_gated_forecast(
    ctx: TransitionResearchContext,
    *,
    detector_reference_experiment_id: str,
    experiment_label: str,
    forecast_title: str,
    decision_reason: str,
    comparison_reference_experiment_id: str | None = None,
    comparison_reference_label: str | None = None,
) -> dict[str, Any]:
    detector_reference = _load_peak_experiment_reference(detector_reference_experiment_id, "peak_window_detector.json")
    decomp_01e_reference = _load_decomp_01e_reference()
    payload = _build_observation_payload(ctx)
    holdout_years = _holdout_years_from_reference(decomp_01e_reference)
    actual_rows = [dict(row) for row in list(payload["rows"])]
    train_rows = _train_rows(actual_rows, holdout_years)
    holdout_rows = _holdout_rows(actual_rows, holdout_years)
    if not train_rows or len(holdout_rows) < 2:
        raise ValueError("PEAK-01B requires non-empty train rows and at least two holdout rows")

    fit_reference = dict(decomp_01e_reference.get("fit_artifact") or {})
    testing_share_mean = float(fit_reference.get("testing_share_train_mean") or 0.0)
    eps = float(np.finfo(np.float32).eps)
    probability_lower_bound = eps
    probability_upper_bound = 1.0 - eps
    locked_transitions = {"U_to_D"}

    detector_payload = dict(detector_reference.get("payload") or {})
    detector_rows = list(detector_payload.get("quarter_rows") or [])
    detector_pair_rows = list(detector_payload.get("pair_rows") or [])
    gate_lookup = {
        (str(row.get("quarter") or ""), str(row.get("transition") or ""), str(row.get("channel") or "")): float(row.get("detector_gate") or 1.0)
        for row in detector_rows
    }
    supported_pair_lookup = {
        (str(row.get("transition") or ""), str(row.get("channel") or "")): bool(row.get("detector_supported"))
        for row in detector_pair_rows
    }

    hazard_reconstruction = dict(decomp_01e_reference.get("hazard_reconstruction") or {})
    hazard_rows = [dict(row) for row in list(hazard_reconstruction.get("rows") or [])]
    adjusted_hazard_map: dict[str, dict[str, float]] = {str(row["quarter"]): {} for row in holdout_rows[1:]}
    gate_rows: list[dict[str, Any]] = []
    for row in hazard_rows:
        quarter = str(row.get("quarter") or "")
        transition = str(row.get("transition") or "")
        channel_name = str(row.get("channel") or "")
        if quarter not in adjusted_hazard_map:
            continue
        baseline_channel = float(row.get("baseline_channel") or 0.0)
        mapped_channel = float(row.get("mapped_channel") or 0.0)
        fused_channel = float(row.get("fused_channel") or 0.0)
        effective_blend_weight = float(row.get("effective_blend_weight") or 0.0)
        if transition in DOWNSTREAM_TRANSITIONS and channel_name in PEAK_CHANNELS and supported_pair_lookup.get((transition, channel_name), False):
            detector_gate = float(gate_lookup.get((quarter, transition, channel_name), 1.0))
            adjusted_effective_weight = effective_blend_weight * detector_gate
            adjusted_channel = baseline_channel + adjusted_effective_weight * (mapped_channel - baseline_channel)
        else:
            detector_gate = 1.0
            adjusted_effective_weight = effective_blend_weight
            adjusted_channel = fused_channel
        adjusted_hazard_map[quarter].setdefault(transition, 0.0)
        adjusted_hazard_map[quarter][transition] += adjusted_channel
        gate_rows.append(
            {
                "quarter": quarter,
                "transition": transition,
                "channel": channel_name,
                "detector_gate": round(float(detector_gate), 6),
                "base_effective_blend_weight": round(float(effective_blend_weight), 6),
                "adjusted_effective_blend_weight": round(float(adjusted_effective_weight), 6),
                "baseline_channel": round(float(baseline_channel), 6),
                "mapped_channel": round(float(mapped_channel), 6),
                "adjusted_channel": round(float(adjusted_channel), 6),
            }
        )
    active_gate_row_count = sum(
        1
        for row in gate_rows
        if abs(float(row["adjusted_effective_blend_weight"]) - float(row["base_effective_blend_weight"])) > eps
    )

    decomp_01e_holdout_states = _holdout_state_rows_from_reference(decomp_01e_reference, reference_label="DECOMP-01E reference")
    decomp_01e_holdout_predictions = [
        dict((row.get("prediction") or {}), quarter=str(row.get("quarter") or ""))
        for row in list((decomp_01e_reference.get("evaluation") or {}).get("holdout_rows") or [])
    ]
    if len(decomp_01e_holdout_states) != len(decomp_01e_holdout_predictions):
        raise ValueError("PEAK-01B requires aligned DECOMP-01E holdout states and prediction rows")
    anchored_state_row = dict(decomp_01e_holdout_states[0])
    anchored_prediction_row = dict(decomp_01e_holdout_predictions[0])
    anchored_forecast_row = {
        "quarter": str(anchored_state_row["quarter"]),
        "diagnosed_plhiv": float(anchored_prediction_row["diagnosed_plhiv"]),
        "alive_on_art": float(anchored_prediction_row["alive_on_art"]),
        "new_diagnosed_cases_period": float(anchored_prediction_row["new_diagnosed_cases_period"]),
        "tested_for_viral_load": float(testing_share_mean * (float(anchored_state_row["state_values"]["A"]) + float(anchored_state_row["state_values"]["V"]))),
        "virally_suppressed": float(anchored_state_row["state_values"]["V"]),
        "state_values": {state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
    }

    simulated_forecast_rows, simulated_holdout_transition_rows = _simulate_holdout_from_hazard_map(
        holdout_rows=holdout_rows[1:],
        hazard_map=adjusted_hazard_map,
        testing_share_mean=testing_share_mean,
        current_state_override={state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
        probability_lower_bound=probability_lower_bound,
        probability_upper_bound=probability_upper_bound,
        eps=eps,
    )
    forecast_rows = [anchored_forecast_row] + simulated_forecast_rows
    holdout_state_rows = [
        {
            "quarter": str(anchored_state_row["quarter"]),
            "diagnosed_plhiv": float(anchored_prediction_row["diagnosed_plhiv"]),
            "alive_on_art": float(anchored_prediction_row["alive_on_art"]),
            "new_diagnosed_cases_period": float(anchored_prediction_row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(anchored_forecast_row["tested_for_viral_load"]),
            "virally_suppressed": float(anchored_state_row["state_values"]["V"]),
            "estimated_plhiv": float(sum(float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES)),
            "state_values": {state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
    ] + [
        {
            "quarter": str(row["quarter"]),
            "diagnosed_plhiv": float(row["diagnosed_plhiv"]),
            "alive_on_art": float(row["alive_on_art"]),
            "new_diagnosed_cases_period": float(row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(row["tested_for_viral_load"]),
            "virally_suppressed": float(row["virally_suppressed"]),
            "estimated_plhiv": float(sum(float(row["state_values"][state_name]) for state_name in STATE_NAMES)),
            "state_values": {state_name: float(row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
        for row in simulated_forecast_rows
    ]

    metric_scales = _metric_scales(train_rows, eps)
    model_mae = round(_normalized_mae(forecast_rows, holdout_rows, metric_scales, eps), 6)
    model_smape = round(_smape(forecast_rows, holdout_rows, eps), 6)
    diagnosis_flow_mae, diagnosis_flow_rows = _diagnosis_flow_mae(
        forecast_rows=forecast_rows,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    decomp_01e_baseline = dict(decomp_01e_reference.get("baseline_comparison") or {})
    target_peak_quarter = max(
        [str(row["quarter"]) for row in holdout_rows],
        key=lambda quarter: float(next(row for row in holdout_rows if str(row["quarter"]) == quarter)["alive_on_art"]),
    )
    model_peak_alive_on_art_error = abs(
        float(next(row for row in forecast_rows if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
        - float(next(row for row in holdout_rows if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
    )
    decomp_01e_peak_alive_on_art_error = abs(
        float(next(row for row in decomp_01e_holdout_predictions if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
        - float(next(row for row in holdout_rows if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
    )
    comparison_reference = (
        _load_peak_gated_reference(comparison_reference_experiment_id)
        if comparison_reference_experiment_id
        else None
    )
    branch_reference_baseline = (
        dict(comparison_reference.get("baseline_comparison") or {})
        if comparison_reference is not None
        else dict(decomp_01e_baseline)
    )
    branch_reference_label = comparison_reference_label or ("DECOMP-01E" if comparison_reference is None else str(comparison_reference_experiment_id))
    branch_reference_mae = float(branch_reference_baseline.get("model_mean_absolute_error") or float("inf"))
    branch_reference_peak_alive_on_art_error = float(
        branch_reference_baseline.get("model_peak_alive_on_art_absolute_error")
        or decomp_01e_peak_alive_on_art_error
    )
    baseline_comparison = {
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
        "decomp_01e_reference_mean_absolute_error": float(decomp_01e_baseline.get("model_mean_absolute_error") or 0.0),
        "decomp_01e_reference_diagnosis_flow_mean_absolute_error": round(float(decomp_01e_baseline.get("diagnosis_flow_mean_absolute_error") or 0.0), 6),
        "decomp_01e_reference_peak_alive_on_art_absolute_error": round(float(decomp_01e_peak_alive_on_art_error), 6),
        "branch_reference_experiment_id": comparison_reference_experiment_id or DECOMP01E_EXPERIMENT_ID,
        "branch_reference_label": branch_reference_label,
        "branch_reference_mean_absolute_error": round(float(branch_reference_mae), 6),
        "branch_reference_peak_alive_on_art_absolute_error": round(float(branch_reference_peak_alive_on_art_error), 6),
        "model_peak_alive_on_art_absolute_error": round(float(model_peak_alive_on_art_error), 6),
        "peak_target_quarter": target_peak_quarter,
        "carry_forward_mean_absolute_error": float(decomp_01e_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(decomp_01e_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_beats_branch_reference_on_mae_or_peak_error": (
            model_mae < float(branch_reference_mae)
            or float(model_peak_alive_on_art_error) < float(branch_reference_peak_alive_on_art_error)
        ),
        "model_beats_carry_forward": model_mae < float(decomp_01e_baseline.get("carry_forward_mean_absolute_error") or float("inf")),
        "model_beats_simple_compartmental": model_mae < float(decomp_01e_baseline.get("simple_compartmental_mean_absolute_error") or float("inf")),
    }
    evaluation = {
        "mode": "transition_research_peak_detector_gated_fused_forecast",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "locked_transitions": sorted(locked_transitions),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "peak_detector_reference_run_id": str(detector_reference["reference_run_id"]),
        "comparison_reference_run_id": (
            None
            if comparison_reference is None
            else str(comparison_reference["reference_run_id"])
        ),
        "decomp_01e_reference_run_id": str(decomp_01e_reference["reference_run_id"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "metric_scales": {name: round(float(value), 6) for name, value in metric_scales.items()},
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "peak_target_quarter": target_peak_quarter,
        "peak_alive_on_art_absolute_error": round(float(model_peak_alive_on_art_error), 6),
        "holdout_rows": [
            {
                "quarter": str(target["quarter"]),
                "prediction": {metric_name: round(float(prediction[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "target": {metric_name: round(float(target[metric_name]), 6) for metric_name in PRIMARY_METRICS},
            }
            for prediction, target in zip(forecast_rows, holdout_rows)
        ],
        "holdout_state_rows": _state_trajectory_rows(holdout_state_rows),
        "diagnosis_flow_evaluation": {
            "mean_absolute_error": round(float(diagnosis_flow_mae), 6),
            "rows": diagnosis_flow_rows,
        },
    }
    upper_bound_rows = [
        dict(row)
        for row in list((decomp_01e_reference.get("mechanistic_forecast") or {}).get("transition_hazards") or [])
    ]
    empirical_upper_bounds = {
        transition: max(
            [
                float((row.get("upper_bounds") or {}).get(transition) or 0.0)
                for row in upper_bound_rows
            ]
            or [0.0]
        )
        for transition in TRANSITION_NAMES
    }
    no_empirical_spike = all(
        float((row.get("hazards") or {}).get(transition) or 0.0) <= float(empirical_upper_bounds.get(transition) or 0.0) + eps
        for row in simulated_holdout_transition_rows
        for transition in TRANSITION_NAMES
    )
    baseline_holdout_hazards = _reference_holdout_hazards(decomp_01e_reference)
    diagnosis_locked_match = all(
        abs(
            float((baseline_holdout_hazards.get(str(row.get("quarter"))) or {}).get("U_to_D") or 0.0)
            - float((row.get("hazards") or {}).get("U_to_D") or 0.0)
        ) <= eps
        for row in simulated_holdout_transition_rows
    )
    decision = {
        "completed": True,
        "keep": bool(
            baseline_comparison["model_beats_branch_reference_on_mae_or_peak_error"]
            and sum(1 for row in detector_pair_rows if bool(row.get("detector_supported"))) > 0
            and active_gate_row_count > 0
            and no_empirical_spike
            and diagnosis_locked_match
        ),
        "reason": decision_reason,
        "checks": [
            {
                "name": "beats_branch_reference_on_mae_or_peak_error",
                "passed": bool(baseline_comparison["model_beats_branch_reference_on_mae_or_peak_error"]),
                "actual": {
                    "model_mean_absolute_error": model_mae,
                    "model_peak_alive_on_art_absolute_error": round(float(model_peak_alive_on_art_error), 6),
                    "peak_target_quarter": target_peak_quarter,
                },
                "target": {
                    "branch_reference_label": branch_reference_label,
                    "branch_reference_mean_absolute_error": round(float(branch_reference_mae), 6),
                    "branch_reference_peak_alive_on_art_absolute_error": round(float(branch_reference_peak_alive_on_art_error), 6),
                    "peak_target_quarter": target_peak_quarter,
                },
            },
            {
                "name": "no_empirical_hazard_spikes",
                "passed": bool(no_empirical_spike),
                "actual": max(
                    float((row.get("hazards") or {}).get(transition) or 0.0)
                    for row in simulated_holdout_transition_rows
                    for transition in TRANSITION_NAMES
                ) if simulated_holdout_transition_rows else 0.0,
                "target": {transition: round(float(value), 6) for transition, value in empirical_upper_bounds.items()},
            },
            {
                "name": "detector_has_supported_pairs",
                "passed": bool(sum(1 for row in detector_pair_rows if bool(row.get("detector_supported"))) > 0),
                "actual": sum(1 for row in detector_pair_rows if bool(row.get("detector_supported"))),
                "target": ">=1",
            },
            {
                "name": "detector_changes_at_least_one_channel_weight",
                "passed": bool(active_gate_row_count > 0),
                "actual": int(active_gate_row_count),
                "target": ">=1",
            },
            {
                "name": "diagnosis_hazard_remains_locked_to_decomp_01e",
                "passed": bool(diagnosis_locked_match),
                "actual": bool(diagnosis_locked_match),
                "target": True,
            },
        ],
    }
    peak_window_gate_summary = {
        "peak_detector_reference_run_id": str(detector_reference["reference_run_id"]),
        "decomp_01e_reference_run_id": str(decomp_01e_reference["reference_run_id"]),
        "rows": gate_rows,
    }

    fig, ax = plt.subplots()
    comparison_labels = [experiment_label, branch_reference_label, "Carry-forward", "Simple compartmental"]
    comparison_values = [
        model_mae,
        float(decomp_01e_baseline.get("model_mean_absolute_error") or 0.0),
        float(decomp_01e_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        float(decomp_01e_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
    ]
    ax.bar(comparison_labels, comparison_values)
    ax.set_ylabel("Normalized MAE")
    ax.set_title(forecast_title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "forecast_vs_baselines.png")
    plt.close(fig)

    write_json(ctx.experiment_dir / "peak_window_gate_summary.json", peak_window_gate_summary)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(
        ctx.experiment_dir / "mechanistic_forecast.json",
        {
            "baseline_comparison": baseline_comparison,
            "evaluation": evaluation,
            "transition_hazards": simulated_holdout_transition_rows,
            "forecast_rows": [
                {
                    "quarter": str(row["quarter"]),
                    "prediction": {metric_name: round(float(row[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                    "state_values": {state_name: round(float(row["state_values"][state_name]), 6) for state_name in STATE_NAMES},
                }
                for row in forecast_rows
            ],
        },
    )
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "peak_detector_reference_experiment_id": detector_reference_experiment_id,
            "peak_detector_reference_run_id": str(detector_reference["reference_run_id"]),
            "comparison_reference_experiment_id": comparison_reference_experiment_id or DECOMP01E_EXPERIMENT_ID,
            "comparison_reference_run_id": (
                None
                if comparison_reference is None
                else str(comparison_reference["reference_run_id"])
            ),
            "decomp_01e_reference_run_id": str(decomp_01e_reference["reference_run_id"]),
            "holdout_years": holdout_years,
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "peak_gate_row_count": len(gate_rows),
            "active_gate_row_count": int(active_gate_row_count),
            "locked_transition_count": len(locked_transitions),
            "detector_supported_pair_count": sum(1 for row in detector_pair_rows if bool(row.get("detector_supported"))),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "peak_detector_supported_pair_count",
                "value": sum(1 for row in detector_pair_rows if bool(row.get("detector_supported"))),
                "role": "peak_detector_supported_pair_count",
                "source_type": "estimated",
                "estimation_data": f"{detector_reference_experiment_id} detector pair summary",
                "estimation_method": "count detector pairs with positive leave-one-quarter-out skill and nonzero train peak support",
                "uncertainty": "depends on train-quarter support",
                "why_needed": f"Defines how many transition-channel pairs are eligible for detector gating in {experiment_label}.",
            },
            {
                "name": "active_gate_row_count",
                "value": int(active_gate_row_count),
                "role": "detector_effective_gate_activation_count",
                "source_type": "estimated",
                "estimation_data": f"{experiment_label} gate rows",
                "estimation_method": "count rows where the detector changed the effective blend weight relative to DECOMP-01E by more than float32 epsilon",
                "uncertainty": "depends on detector support and quarter-level gate values",
                "why_needed": "Prevents branches with no effective detector action from being promoted on numerical drift alone.",
            },
            {
                "name": "peak_channel_count",
                "value": len(PEAK_CHANNELS),
                "role": "detector_gated_channel_grid_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "peak detector design",
                "estimation_method": "count of downstream channels gated by the detector",
                "uncertainty": "none",
                "why_needed": "Restricts detector gating to shock and residual channels instead of introducing hidden extra gates.",
            },
            {
                "name": "branch_reference_mean_absolute_error",
                "value": round(float(branch_reference_mae), 6),
                "role": "branch_reference_forecast_gate",
                "source_type": "estimated",
                "estimation_data": f"kept {branch_reference_label} baseline_comparison artifact",
                "estimation_method": f"reuse the latest kept {branch_reference_label} normalized holdout MAE as the {experiment_label} reference gate",
                "uncertainty": f"inherits {branch_reference_label} holdout uncertainty",
                "why_needed": f"Ensures the peak-window detector branch is judged against the current kept {branch_reference_label} baseline.",
            },
            {
                "name": "branch_reference_peak_alive_on_art_absolute_error",
                "value": round(float(branch_reference_peak_alive_on_art_error), 6),
                "role": "peak_quarter_reference_gate",
                "source_type": "estimated",
                "estimation_data": f"kept {branch_reference_label} prediction and holdout alive_on_art target at {target_peak_quarter}",
                "estimation_method": "absolute error at the holdout quarter with the highest alive_on_art target",
                "uncertainty": "depends on holdout alive_on_art observation uncertainty",
                "why_needed": f"Provides the explicit peak-error comparator for {experiment_label}.",
            },
            {
                "name": "probability_lower_bound",
                "value": probability_lower_bound,
                "role": "hazard_probability_floor",
                "source_type": "physical_constraint",
                "estimation_data": "probability simplex",
                "estimation_method": "lower probability bound for transition hazards",
                "uncertainty": "none",
                "why_needed": "Prevents negative transition hazards during mechanistic simulation.",
            },
            {
                "name": "probability_upper_bound",
                "value": probability_upper_bound,
                "role": "hazard_probability_ceiling",
                "source_type": "physical_constraint",
                "estimation_data": "probability simplex",
                "estimation_method": "upper probability bound for transition hazards",
                "uncertainty": "none",
                "why_needed": "Prevents transition hazards from exceeding available probability mass during mechanistic simulation.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="peak_detector_gated_fusion_guard",
                why_needed="Prevents undefined equality checks and degenerate state outflow scaling during detector-gated simulation.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "decision": decision,
        "baseline_comparison": baseline_comparison,
        "evaluation": evaluation,
        "peak_window_gate_summary": peak_window_gate_summary,
    }


def run_peak_01b(ctx: TransitionResearchContext) -> dict[str, Any]:
    return _run_peak_gated_forecast(
        ctx,
        detector_reference_experiment_id=PEAK01A_EXPERIMENT_ID,
        experiment_label="PEAK-01B",
        forecast_title="PEAK-01B Forecast vs Baselines",
        decision_reason="PEAK-01B gates downstream shock and residual fusion with a regional and KP-conditioned peak-window detector on top of the kept DECOMP-01E baseline.",
    )


def run_peak_01d(ctx: TransitionResearchContext) -> dict[str, Any]:
    return _run_peak_gated_forecast(
        ctx,
        detector_reference_experiment_id=PEAK01C_EXPERIMENT_ID,
        experiment_label="PEAK-01D",
        forecast_title="PEAK-01D Forecast vs Baselines",
        decision_reason="PEAK-01D gates downstream shock and residual fusion with a region-only peak-window detector on top of the kept DECOMP-01E baseline.",
    )


def run_peak_01e(ctx: TransitionResearchContext) -> dict[str, Any]:
    region_reference = _load_peak_experiment_reference(PEAK01C_EXPERIMENT_ID, "peak_window_detector.json")
    kp_reference = _load_peak_experiment_reference(PEAK01A_EXPERIMENT_ID, "peak_window_detector.json")
    mech_01e_reference = _load_transition_experiment_reference("MECH-01E-anchored-downstream-residual-helpers")
    train_quarters = [str(value) for value in list((mech_01e_reference.get("evaluation") or {}).get("train_quarters") or []) if str(value)]
    if not train_quarters:
        raise ValueError("PEAK-01E requires MECH-01E train quarter support")
    region_payload = dict(region_reference.get("payload") or {})
    kp_payload = dict(kp_reference.get("payload") or {})
    quarter_axis = [str(value) for value in list(region_payload.get("quarter_axis") or []) if str(value)]
    if not quarter_axis:
        raise ValueError("PEAK-01E requires a non-empty PEAK-01C quarter axis")
    eps = float(np.finfo(np.float32).eps)

    region_pair_rows = [dict(row) for row in list(region_payload.get("pair_rows") or [])]
    region_quarter_lookup = {
        (str(row.get("quarter") or ""), str(row.get("transition") or ""), str(row.get("channel") or "")): dict(row)
        for row in list(region_payload.get("quarter_rows") or [])
    }
    kp_quarter_lookup = {
        (str(row.get("quarter") or ""), str(row.get("transition") or ""), str(row.get("channel") or "")): dict(row)
        for row in list(kp_payload.get("quarter_rows") or [])
    }
    kp_low_confidence_reasons = list((kp_reference.get("payload") or {}).get("coverage_summary", {}).get("low_confidence_reasons") or [])

    pair_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    gate_matrix_rows: list[np.ndarray] = []
    heatmap_labels: list[str] = []
    active_modifier_pair_count = 0
    max_kp_modifier_strength = 0.0
    for pair_row in region_pair_rows:
        transition = str(pair_row.get("transition") or "")
        channel_name = str(pair_row.get("channel") or "")
        heatmap_labels.append(f"{transition}|{channel_name}")
        base_supported = bool(pair_row.get("detector_supported"))
        region_train_signals = [
            float((region_quarter_lookup.get((quarter, transition, channel_name)) or {}).get("regional_signal") or 0.0)
            for quarter in quarter_axis
            if quarter in set(train_quarters)
        ]
        kp_train_signals = [
            float((kp_quarter_lookup.get((quarter, transition, channel_name)) or {}).get("kp_signal") or 0.0)
            for quarter in quarter_axis
            if quarter in set(train_quarters)
        ]
        max_train_regional_signal = max(region_train_signals or [0.0])
        max_train_kp_signal = max(kp_train_signals or [0.0])
        kp_modifier_strength = (
            float(max_train_kp_signal / max(max_train_regional_signal + max_train_kp_signal, eps))
            if base_supported and max_train_kp_signal > eps
            else 0.0
        )
        max_kp_modifier_strength = max(max_kp_modifier_strength, float(kp_modifier_strength))
        modifier_supported = bool(base_supported and kp_modifier_strength > eps)
        if modifier_supported:
            active_modifier_pair_count += 1
        pair_gate_values: list[float] = []
        for quarter in quarter_axis:
            region_row = dict(region_quarter_lookup.get((quarter, transition, channel_name)) or {})
            kp_row = dict(kp_quarter_lookup.get((quarter, transition, channel_name)) or {})
            regional_signal = float(region_row.get("regional_signal") or 0.0)
            kp_signal = float(kp_row.get("kp_signal") or 0.0)
            region_gate = float(region_row.get("detector_gate") or 1.0)
            kp_normalized_support = float(np.clip(kp_signal / max(max_train_kp_signal, eps), 0.0, 1.0)) if max_train_kp_signal > eps else 0.0
            kp_modifier_gate = (
                float(1.0 - (kp_modifier_strength * (1.0 - kp_normalized_support)))
                if modifier_supported
                else 1.0
            )
            combined_gate = float(region_gate * kp_modifier_gate) if modifier_supported else float(region_gate)
            pair_gate_values.append(combined_gate)
            quarter_rows.append(
                {
                    "quarter": quarter,
                    "transition": transition,
                    "channel": channel_name,
                    "detector_mode": "region_plus_kp_modifier",
                    "regional_signal": round(float(regional_signal), 6),
                    "kp_signal": round(float(kp_signal), 6),
                    "kp_normalized_support": round(float(kp_normalized_support), 6),
                    "region_gate": round(float(region_gate), 6),
                    "kp_modifier_gate": round(float(kp_modifier_gate), 6),
                    "detector_gate": round(float(combined_gate), 6),
                    "target_peak_signal": round(float(region_row.get("target_peak_signal") or 0.0), 6),
                    "kp_group_signals": dict(kp_row.get("kp_group_signals") or {kp_name: 0.0 for kp_name in KP_COLLAPSED_NAMES}),
                }
            )
        gate_matrix_rows.append(np.asarray(pair_gate_values, dtype=np.float32))
        pair_rows.append(
            {
                "transition": transition,
                "channel": channel_name,
                "detector_mode": "region_plus_kp_modifier",
                "base_region_detector_supported": base_supported,
                "detector_supported": modifier_supported,
                "region_loo_r2": round(float(pair_row.get("loo_r2") or 0.0), 6),
                "kp_modifier_strength": round(float(kp_modifier_strength), 6),
                "max_train_regional_signal": round(float(max_train_regional_signal), 6),
                "max_train_kp_signal": round(float(max_train_kp_signal), 6),
                "low_confidence_reasons": kp_low_confidence_reasons,
            }
        )

    gate_matrix = np.vstack(gate_matrix_rows) if gate_matrix_rows else np.zeros((0, len(quarter_axis)), dtype=np.float32)
    detector_json = {
        "quarter_axis": quarter_axis,
        "pair_rows": pair_rows,
        "quarter_rows": quarter_rows,
    }
    write_json(ctx.experiment_dir / "peak_window_detector.json", detector_json)
    fig, ax = plt.subplots()
    image = ax.imshow(gate_matrix, aspect="auto")
    ax.set_title("PEAK-01E Region + KP Modifier Peak-Window Gates")
    ax.set_xticks(np.arange(len(quarter_axis)), labels=list(quarter_axis))
    ax.set_yticks(np.arange(len(heatmap_labels)), labels=list(heatmap_labels))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "peak_window_heatmap.png")
    plt.close(fig)

    coverage_summary = {
        "source_run_id": ctx.source_run_id,
        "detector_mode": "region_plus_kp_modifier",
        "train_quarter_count": len(train_quarters),
        "holdout_quarter_count": len(list((mech_01e_reference.get("evaluation") or {}).get("holdout_quarters") or [])),
        "detector_pair_count": len(pair_rows),
        "base_region_supported_pair_count": sum(1 for row in pair_rows if bool(row["base_region_detector_supported"])),
        "supported_detector_pair_count": sum(1 for row in pair_rows if bool(row["detector_supported"])),
        "active_modifier_pair_count": int(active_modifier_pair_count),
        "low_confidence_reasons": kp_low_confidence_reasons,
    }
    decision = {
        "completed": True,
        "keep": bool(coverage_summary["supported_detector_pair_count"] > 0 and active_modifier_pair_count > 0),
        "reason": "PEAK-01E reintroduces KP only as a modifier on top of supported region-only detector pairs.",
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "base_region_detector_reference_run_id": str(region_reference["reference_run_id"]),
            "kp_signal_reference_run_id": str(kp_reference["reference_run_id"]),
            "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        },
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            {
                "name": "base_region_supported_pair_count",
                "value": int(coverage_summary["base_region_supported_pair_count"]),
                "role": "region_detector_support_floor",
                "source_type": "estimated",
                "estimation_data": "PEAK-01C detector pair summary",
                "estimation_method": "count PEAK-01C pairs with positive leave-one-quarter-out detector support",
                "uncertainty": "depends on train-quarter support",
                "why_needed": "Ensures KP modifier logic only attaches to region-only detector pairs that already showed out-of-sample skill.",
            },
            {
                "name": "active_modifier_pair_count",
                "value": int(active_modifier_pair_count),
                "role": "kp_modifier_active_pair_count",
                "source_type": "estimated",
                "estimation_data": "PEAK-01E pair summary",
                "estimation_method": "count supported region detector pairs with positive train-time KP signal scale ratio",
                "uncertainty": "depends on sparse KP signal support",
                "why_needed": "Prevents the KP modifier branch from promoting itself when the modifier has no active pair-level effect.",
            },
            {
                "name": "max_kp_modifier_strength",
                "value": round(float(max_kp_modifier_strength), 6),
                "role": "kp_modifier_strength_ceiling",
                "source_type": "estimated",
                "estimation_data": "PEAK-01C regional train signals and PEAK-01A KP train signals",
                "estimation_method": "per-pair ratio max_train_kp_signal / (max_train_kp_signal + max_train_regional_signal), then take the maximum active pair value",
                "uncertainty": "depends on sparse KP signal support",
                "why_needed": "Records the strongest KP modifier scale actually learned from the data instead of introducing a manual modifier coefficient.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="kp_modifier_guard",
                why_needed="Prevents undefined division and degenerate gate updates when train-time KP signal support is absent.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "decision": decision,
        "quarter_axis": quarter_axis,
        "pair_rows": pair_rows,
        "quarter_rows": quarter_rows,
        "gate_matrix": gate_matrix,
        "heatmap_labels": heatmap_labels,
        "coverage_summary": coverage_summary,
    }


def run_peak_01f(ctx: TransitionResearchContext) -> dict[str, Any]:
    return _run_peak_gated_forecast(
        ctx,
        detector_reference_experiment_id=PEAK01E_EXPERIMENT_ID,
        experiment_label="PEAK-01F",
        forecast_title="PEAK-01F Forecast vs Baselines",
        decision_reason="PEAK-01F gates downstream shock and residual fusion with a region-first detector and KP-only modifier on top of the kept DECOMP-01E baseline.",
        comparison_reference_experiment_id=PEAK01D_EXPERIMENT_ID,
        comparison_reference_label="PEAK-01D",
    )


def _nonnegative_two_feature_fit(
    regional_values: np.ndarray,
    kp_values: np.ndarray,
    target_values: np.ndarray,
    eps: float,
) -> tuple[float, float, float]:
    if regional_values.size == 0 or target_values.size == 0:
        return 0.0, 0.0, 0.0
    design = np.column_stack([regional_values.astype(np.float64), kp_values.astype(np.float64)])
    if not np.any(np.abs(design) > eps):
        residual_mse = float(np.mean(np.square(target_values.astype(np.float64)))) if target_values.size else 0.0
        return 0.0, 0.0, residual_mse
    coefficients, _, _, _ = np.linalg.lstsq(design, target_values.astype(np.float64), rcond=None)
    beta_regional = max(float(coefficients[0]), 0.0)
    beta_kp = max(float(coefficients[1]), 0.0)
    prediction = (beta_regional * regional_values.astype(np.float64)) + (beta_kp * kp_values.astype(np.float64))
    residual_mse = float(np.mean(np.square(target_values.astype(np.float64) - prediction))) if target_values.size else 0.0
    return beta_regional, beta_kp, residual_mse


def _leave_one_out_two_feature_r2(
    regional_values: np.ndarray,
    kp_values: np.ndarray,
    target_values: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    if regional_values.size != kp_values.size or regional_values.size != target_values.size or regional_values.size < 3:
        return 0.0, 0.0
    predictions: list[float] = []
    targets: list[float] = []
    sample_count = int(target_values.size)
    for holdout_index in range(sample_count):
        mask = np.ones(sample_count, dtype=bool)
        mask[holdout_index] = False
        beta_regional, beta_kp, _ = _nonnegative_two_feature_fit(
            regional_values[mask],
            kp_values[mask],
            target_values[mask],
            eps,
        )
        predictions.append(
            max(
                0.0,
                (beta_regional * float(regional_values[holdout_index]))
                + (beta_kp * float(kp_values[holdout_index])),
            )
        )
        targets.append(float(target_values[holdout_index]))
    prediction_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    residual_mse = float(np.mean(np.square(target_array - prediction_array))) if target_array.size else 0.0
    target_variance = float(np.var(target_array)) if target_array.size else 0.0
    loo_r2 = max(0.0, 1.0 - (residual_mse / max(target_variance, eps))) if target_variance > eps else 0.0
    return float(loo_r2), float(residual_mse)


def _load_peak_experiment_reference(experiment_id: str, required_path: str) -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_decomp_experiment(experiment_id, required_path)
    payload = read_json(experiment_dir / required_path, default={})
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "payload": dict(payload or {}),
    }


def _load_peak_gated_reference(experiment_id: str) -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_decomp_experiment(experiment_id, "baseline_comparison.json")
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "baseline_comparison": dict(read_json(experiment_dir / "baseline_comparison.json", default={}) or {}),
        "evaluation": dict(read_json(experiment_dir / "evaluation.json", default={}) or {}),
        "peak_window_gate_summary": dict(read_json(experiment_dir / "peak_window_gate_summary.json", default={}) or {}),
        "mechanistic_forecast": dict(read_json(experiment_dir / "mechanistic_forecast.json", default={}) or {}),
    }


def _load_decomp_01e_reference() -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_decomp_experiment(
        DECOMP01E_EXPERIMENT_ID,
        "hazard_reconstruction.json",
    )
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "evaluation": dict(read_json(experiment_dir / "evaluation.json", default={}) or {}),
        "baseline_comparison": dict(read_json(experiment_dir / "baseline_comparison.json", default={}) or {}),
        "fit_artifact": dict(read_json(experiment_dir / "fit_artifact.json", default={}) or {}),
        "transition_hazard_summary": dict(read_json(experiment_dir / "transition_hazard_summary.json", default={}) or {}),
        "hazard_reconstruction": dict(read_json(experiment_dir / "hazard_reconstruction.json", default={}) or {}),
        "mechanistic_forecast": dict(read_json(experiment_dir / "mechanistic_forecast.json", default={}) or {}),
    }


def _detector_payload(ctx: TransitionResearchContext, *, include_kp: bool = True) -> dict[str, Any]:
    inputs = load_transition_research_inputs(ctx)
    decomp_01a_reference = _load_decomp_01a_reference()
    mech_01e_reference = _load_transition_experiment_reference("MECH-01E-anchored-downstream-residual-helpers")
    train_quarters = [str(value) for value in list((mech_01e_reference.get("evaluation") or {}).get("train_quarters") or []) if str(value)]
    holdout_quarters = [str(value) for value in list((mech_01e_reference.get("evaluation") or {}).get("holdout_quarters") or []) if str(value)]
    quarter_axis = [str(value) for value in list((decomp_01a_reference.get("channel_summary") or {}).get("quarter_axis") or []) if str(value)]
    if not train_quarters or not holdout_quarters or not quarter_axis:
        raise ValueError("Peak detector requires train/holdout quarter support and DECOMP-01A quarter axis")
    eps = float(np.finfo(np.float32).eps)

    national_surface = _quarter_factor_surface(ctx)
    regional_surface = _quarter_region_factor_surface(ctx)
    transition_rows = _factor_transition_rows(inputs)
    transition_weight_lookup: dict[str, dict[str, float]] = {transition: {} for transition in TRANSITION_NAMES}
    for row in transition_rows:
        transition_weight_lookup[str(row["transition"])][str(row["factor_id"])] = float(row["relevance"])
    collapsed_distribution, kp_low_confidence_reasons = _collapse_kp_distribution(inputs)
    subgroup_hook_lookup = {
        factor_id: "subgroup_allocation_priors" in list((inputs.retained_factor_lookup.get(factor_id) or {}).get("transition_hooks", []))
        for factor_id in inputs.retained_factor_lookup
    }
    if include_kp:
        active_collapsed_distribution = {kp_name: float(collapsed_distribution.get(kp_name, 0.0)) for kp_name in KP_COLLAPSED_NAMES}
        low_confidence_reasons = list(kp_low_confidence_reasons)
        kp_group_count = sum(1 for kp_name in KP_COLLAPSED_NAMES if float(active_collapsed_distribution.get(kp_name, 0.0)) > 0.0)
        detector_mode = "regional_kp"
        detector_feature_count = 2
    else:
        active_collapsed_distribution = {kp_name: 0.0 for kp_name in KP_COLLAPSED_NAMES}
        low_confidence_reasons = []
        kp_group_count = 0
        detector_mode = "region_only"
        detector_feature_count = 1

    regional_factor_signals: dict[str, dict[str, float]] = {}
    for factor_id in sorted(inputs.retained_factor_lookup):
        quarter_signal: dict[str, float] = {}
        region_zscores = {
            region_name: _zscore_by_quarter(region_surface[factor_id], train_quarters, eps)
            for region_name, region_surface in regional_surface.items()
        }
        for quarter in quarter_axis:
            positive_region_scores = [max(float(zscores.get(quarter) or 0.0), 0.0) for zscores in region_zscores.values()]
            mean_positive = float(np.mean(positive_region_scores)) if positive_region_scores else 0.0
            positive_fraction = (
                float(sum(1 for value in positive_region_scores if value > 0.0)) / float(len(positive_region_scores))
                if positive_region_scores
                else 0.0
            )
            quarter_signal[quarter] = mean_positive * positive_fraction
        regional_factor_signals[factor_id] = quarter_signal

    national_factor_zscores = {
        factor_id: _zscore_by_quarter(values_by_quarter, train_quarters, eps)
        for factor_id, values_by_quarter in national_surface.items()
    }

    channel_tensor = np.asarray(decomp_01a_reference["channel_tensor"], dtype=np.float32)
    transition_index = {transition: idx for idx, transition in enumerate(TRANSITION_NAMES)}
    channel_index = {channel_name: idx for idx, channel_name in enumerate(CHANNEL_NAMES)}

    pair_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    gate_matrix_rows: list[np.ndarray] = []
    heatmap_labels: list[str] = []
    for transition in DOWNSTREAM_TRANSITIONS:
        factor_weights = dict(transition_weight_lookup.get(transition) or {})
        weight_total = float(sum(factor_weights.values()))
        for channel_name in PEAK_CHANNELS:
            heatmap_labels.append(f"{transition}|{channel_name}")
            target_series = np.asarray(
                [
                    max(float(channel_tensor[transition_index[transition], quarter_axis.index(quarter), channel_index[channel_name]]), 0.0)
                    for quarter in quarter_axis
                ],
                dtype=np.float64,
            )
            regional_signal_map: dict[str, float] = {}
            kp_signal_map: dict[str, float] = {}
            kp_detail_map: dict[str, dict[str, float]] = {}
            for quarter in quarter_axis:
                regional_total = 0.0
                kp_total = 0.0
                kp_detail: dict[str, float] = {}
                for factor_id in sorted(inputs.retained_factor_lookup):
                    weight = float(factor_weights.get(factor_id) or 0.0)
                    if weight_total > 0.0:
                        weight = weight / weight_total
                    regional_total += weight * float((regional_factor_signals.get(factor_id) or {}).get(quarter) or 0.0)
                    if include_kp and subgroup_hook_lookup.get(factor_id):
                        positive_kp_signal = max(float((national_factor_zscores.get(factor_id) or {}).get(quarter) or 0.0), 0.0)
                        for kp_name in KP_COLLAPSED_NAMES:
                            kp_value = weight * float(active_collapsed_distribution.get(kp_name, 0.0)) * positive_kp_signal
                            kp_detail[kp_name] = float(kp_detail.get(kp_name, 0.0)) + kp_value
                            kp_total += kp_value
                regional_signal_map[quarter] = float(regional_total)
                kp_signal_map[quarter] = float(kp_total)
                kp_detail_map[quarter] = {kp_name: round(float(kp_detail.get(kp_name, 0.0)), 6) for kp_name in KP_COLLAPSED_NAMES}

            train_mask = np.asarray([quarter in set(train_quarters) for quarter in quarter_axis], dtype=bool)
            regional_train = np.asarray([regional_signal_map[quarter] for quarter in quarter_axis], dtype=np.float64)[train_mask]
            kp_train = np.asarray([kp_signal_map[quarter] for quarter in quarter_axis], dtype=np.float64)[train_mask]
            target_train = target_series[train_mask]
            beta_regional, beta_kp, residual_mse = _nonnegative_two_feature_fit(regional_train, kp_train, target_train, eps)
            loo_r2, loo_residual_mse = _leave_one_out_two_feature_r2(regional_train, kp_train, target_train, eps)
            predicted_series = np.asarray(
                [
                    max(0.0, (beta_regional * float(regional_signal_map[quarter])) + (beta_kp * float(kp_signal_map[quarter])))
                    for quarter in quarter_axis
                ],
                dtype=np.float64,
            )
            train_target_max = float(np.max(target_train)) if target_train.size else 0.0
            detector_supported = bool(loo_r2 > 0.0 and train_target_max > eps)
            if detector_supported:
                normalized_support = np.clip(predicted_series / max(train_target_max, eps), 0.0, 1.0).astype(np.float64)
                detector_gate = (1.0 - float(loo_r2) * (1.0 - normalized_support)).astype(np.float64)
            else:
                normalized_support = np.zeros_like(predicted_series, dtype=np.float64)
                detector_gate = np.ones_like(predicted_series, dtype=np.float64)
            pair_rows.append(
                {
                    "detector_mode": detector_mode,
                    "transition": transition,
                    "channel": channel_name,
                    "regional_beta": round(float(beta_regional), 6),
                    "kp_beta": round(float(beta_kp), 6),
                    "loo_r2": round(float(loo_r2), 6),
                    "train_residual_mse": round(float(residual_mse), 10),
                    "loo_residual_mse": round(float(loo_residual_mse), 10),
                    "train_target_max": round(float(train_target_max), 6),
                    "detector_supported": detector_supported,
                    "kp_positive_group_count": int(kp_group_count),
                    "low_confidence_reasons": low_confidence_reasons,
                }
            )
            gate_matrix_rows.append(np.asarray(detector_gate, dtype=np.float32))
            for quarter_index, quarter in enumerate(quarter_axis):
                quarter_rows.append(
                    {
                        "detector_mode": detector_mode,
                        "quarter": quarter,
                        "transition": transition,
                        "channel": channel_name,
                        "regional_signal": round(float(regional_signal_map[quarter]), 6),
                        "kp_signal": round(float(kp_signal_map[quarter]), 6),
                        "predicted_peak_support": round(float(predicted_series[quarter_index]), 6),
                        "normalized_peak_support": round(float(normalized_support[quarter_index]), 6),
                        "detector_gate": round(float(detector_gate[quarter_index]), 6),
                        "target_peak_signal": round(float(target_series[quarter_index]), 6),
                        "kp_group_signals": kp_detail_map[quarter],
                    }
                )
    gate_matrix = np.vstack(gate_matrix_rows) if gate_matrix_rows else np.zeros((0, len(quarter_axis)), dtype=np.float32)
    return {
        "quarter_axis": quarter_axis,
        "pair_rows": pair_rows,
        "quarter_rows": quarter_rows,
        "gate_matrix": gate_matrix,
        "heatmap_labels": heatmap_labels,
        "coverage_summary": {
            "source_run_id": ctx.source_run_id,
            "detector_mode": detector_mode,
            "train_quarter_count": len(train_quarters),
            "holdout_quarter_count": len(holdout_quarters),
            "region_count": len(inputs.region_axis),
            "kp_group_count": int(kp_group_count),
            "detector_pair_count": len(pair_rows),
            "supported_detector_pair_count": sum(1 for row in pair_rows if bool(row["detector_supported"])),
            "low_confidence_reasons": low_confidence_reasons,
        },
        "numeric_justification": [
            {
                "name": "peak_channel_count",
                "value": len(PEAK_CHANNELS),
                "role": "peak_detector_channel_grid_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "peak detector design",
                "estimation_method": "count of downstream channels eligible for peak-window gating: shock and residual",
                "uncertainty": "none",
                "why_needed": "Defines the detector channel grid without introducing an implicit extra channel family.",
            },
            {
                "name": "downstream_transition_count",
                "value": len(DOWNSTREAM_TRANSITIONS),
                "role": "peak_detector_transition_grid_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of downstream transitions eligible for peak-window gating",
                "uncertainty": "none",
                "why_needed": "Limits the detector to non-diagnosis transitions while diagnosis stays locked on the kept baseline.",
            },
            {
                "name": "detector_feature_count",
                "value": int(detector_feature_count),
                "role": "peak_detector_feature_count",
                "source_type": "physical_constraint",
                "estimation_data": "detector model design",
                "estimation_method": (
                    "count of fitted detector features: regional signal and KP-conditioned signal"
                    if include_kp
                    else "count of fitted detector features: regional signal only"
                ),
                "uncertainty": "none",
                "why_needed": (
                    "Records the exact detector feature set so there are no hidden peak heuristics in the regional/KP detector."
                    if include_kp
                    else "Records the exact detector feature set so the region-only detector does not silently reuse KP conditioning."
                ),
            },
            {
                "name": "region_count",
                "value": len(inputs.region_axis),
                "role": "regional_peak_signal_cardinality",
                "source_type": "estimated",
                "estimation_data": "phase15 multiscale_factor_axes.region",
                "estimation_method": "count of retained region labels in the source tensor",
                "uncertainty": "none",
                "why_needed": "Normalizes regional cluster evidence to the active source grid instead of a manual region count.",
            },
            {
                "name": "kp_positive_group_count",
                "value": int(kp_group_count),
                "role": "nonzero_collapsed_kp_group_count",
                "source_type": "estimated",
                "estimation_data": "phase3 subgroup_weight_summary national_kp_distribution collapsed to msm/tgw/other",
                "estimation_method": "count of collapsed KP groups with positive mass",
                "uncertainty": "depends on Phase 3 subgroup learning support",
                "why_needed": "Prevents the KP-conditioned detector signal from implying unsupported group mass.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="peak_detector_fit_guard",
                why_needed="Prevents undefined z-score normalization and degenerate least-squares scaling when quarter support is sparse.",
            ),
        ],
    }
