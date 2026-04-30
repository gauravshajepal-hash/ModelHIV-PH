from __future__ import annotations

import argparse
import csv
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .data import (
    STATE_NAMES,
    build_blocked_time_dataset,
    build_observation_rows,
    default_epigraph_root,
    rolling_origin_splits,
    sandbox_repo_root,
)
from .hybrid_champion import (
    _detect_shock_catalog,
    _load_r10_baseline,
    _r10_reference_scores,
    _shock_aware_lifted_gate_from_evaluated,
)
from .incidence import fit_incidence_flow_paths
from .incidence_module_search import (
    _annual_incidence_targets,
    _evaluate_split as _evaluate_incidence_split,
    _r10_annual_incidence_error,
)
from .metrics import normalized_mae, quarter_sort_key, quarter_year
from .model import (
    _apply_observation_model,
    _quarter_positions,
    _simulate_sequence,
    carry_forward_hazards,
    fit_dynamic_hazard_paths,
    fit_observation_model,
    simulate_holdout,
)
from .observation_ledger import (
    build_observation_role_ledger,
    resolve_active_source_run_id,
    resolve_baseline_source_run_id,
)
from .runtime import ensure_dir, write_json
from .scenario_lab import (
    DEFAULT_ACTIVE_SOURCE_RUN_ID,
    DEFAULT_BASELINE_SOURCE_RUN_ID,
    _apply_projection_constraints,
    _build_projection_constraints,
    _filter_rows_before_holdout,
    _future_quarters,
    _load_reference_config,
    _projection_dataset,
    _scale_state_attrition_to_aggregate,
)


INCIDENCE_READOUT_FULL_GATE_SCHEMA_VERSION = "phase3_dynamic_incidence_readout_full_gate.v1"
PROMOTED_INCIDENCE_FAMILY = "s_eff_hazard_r10_mechanistic_annual_measurement_readout"


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = [float(row["candidate_mae"]) for row in rows if _finite_float(row.get("candidate_mae")) is not None]
    carry = [float(row["carry_forward_mae"]) for row in rows if _finite_float(row.get("carry_forward_mae")) is not None]
    if not candidate or not carry:
        return {
            "split_count": 0,
            "candidate_mean_mae": None,
            "carry_forward_mean_mae": None,
            "candidate_worst_mae": None,
            "carry_forward_worst_mae": None,
            "candidate_minus_carry_forward_mean_mae": None,
            "candidate_minus_carry_forward_worst_mae": None,
        }
    candidate_array = np.asarray(candidate, dtype=np.float64)
    carry_array = np.asarray(carry, dtype=np.float64)
    return {
        "split_count": len(candidate),
        "candidate_mean_mae": float(np.mean(candidate_array)),
        "carry_forward_mean_mae": float(np.mean(carry_array)),
        "candidate_worst_mae": float(np.max(candidate_array)),
        "carry_forward_worst_mae": float(np.max(carry_array)),
        "candidate_minus_carry_forward_mean_mae": float(np.mean(candidate_array) - np.mean(carry_array)),
        "candidate_minus_carry_forward_worst_mae": float(np.max(candidate_array) - np.max(carry_array)),
    }


def _select_state_incidence_map(incidence_result: dict[str, Any]) -> dict[str, float]:
    return {
        str(quarter): float(value)
        for quarter, value in dict(incidence_result.get("incidence_map") or {}).items()
    }


def _select_measurement_readout_map(incidence_result: dict[str, Any]) -> dict[str, float]:
    return {
        str(quarter): float(value)
        for quarter, value in dict(
            incidence_result.get("annual_measurement_readout_incidence_map")
            or incidence_result.get("weak_measurement_incidence_map")
            or {}
        ).items()
    }


def _stock_balance_paths(
    *,
    dataset: Any,
    reference_config: dict[str, Any],
    incidence_result: dict[str, Any],
) -> dict[str, Any]:
    base_paths = dict(fit_incidence_flow_paths(dataset, reference_config["incidence_cfg"]))
    state_incidence_map = _select_state_incidence_map(incidence_result)
    measurement_readout_map = _select_measurement_readout_map(incidence_result)
    updated = dict(base_paths)
    updated["holdout_incidence_inflow_map"] = state_incidence_map
    updated["holdout_incidence_hazard_map"] = {}
    updated["holdout_incidence_cap_map"] = {}
    updated["holdout_state_incidence_contract"] = (
        "full-cascade state path uses the latent interval-aware S_eff incidence inflow; "
        "annual measurement readout is retained as readout-only evidence and is not injected into U"
    )
    updated["holdout_measurement_readout_incidence_map"] = measurement_readout_map
    updated["holdout_measurement_readout_contract"] = (
        "annual_new_infections readout rescales annual incidence measurements for validation only; "
        "it is not used as a state-transition truth source"
    )
    return updated


def _fit_candidate_backbone(
    *,
    dataset: Any,
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    incidence_result: dict[str, Any],
) -> dict[str, Any]:
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    hazard_paths = fit_dynamic_hazard_paths(
        dataset,
        reference_config["dynamic_cfg"],
        shock_cfg=reference_config["shock_cfg"],
        damping_cfg=reference_config["damping_cfg"],
    )
    incidence_paths = _stock_balance_paths(
        dataset=dataset,
        reference_config=reference_config,
        incidence_result=incidence_result,
    )
    constraints = _build_projection_constraints(
        dataset=dataset,
        constraint_rows=constraint_rows,
        future_quarters=holdout_quarters,
    )
    incidence_map, attrition_map, constraint_application = _apply_projection_constraints(
        incidence_map=dict(incidence_paths.get("holdout_incidence_inflow_map") or {}),
        attrition_map=dict(incidence_paths.get("holdout_attrition_outflow_map") or {}),
        constraints=constraints,
    )
    state_attrition_map = _scale_state_attrition_to_aggregate(
        dict(incidence_paths.get("holdout_state_attrition_outflow_map") or {}),
        attrition_map,
    )
    raw = _simulate_sequence(
        dict(dataset.train_state_rows[-1]["state_values"]),
        holdout_rows,
        dict(hazard_paths.get("holdout_hazard_map") or {}),
        incidence_inflow_map=incidence_map,
        incidence_hazard_map={},
        incidence_cap_map={quarter: float(constraints.get("incidence_cap_per_quarter") or 0.0) for quarter in holdout_quarters},
        population_denominator_map=dict(incidence_paths.get("holdout_population_denominator_map") or {}),
        attrition_outflow_map=attrition_map,
        state_attrition_outflow_map=state_attrition_map,
        exit_channel_state_outflow_map=dict(incidence_paths.get("holdout_exit_channel_state_outflow_map") or {}),
    )
    observation_model = fit_observation_model(
        dataset,
        dict(hazard_paths.get("train_hazard_map") or {}),
        reference_config["observation_cfg"],
        incidence_paths=incidence_paths,
        damping_cfg=reference_config["damping_cfg"],
    )
    prediction_rows = _apply_observation_model(
        list(raw.get("prediction_rows") or []),
        observation_model,
        positions=_quarter_positions(dataset),
        damping_cfg=reference_config["damping_cfg"],
    )
    return {
        "prediction_rows": prediction_rows,
        "raw_prediction_rows": list(raw.get("prediction_rows") or []),
        "trajectory_rows": list(raw.get("trajectory_rows") or []),
        "mae": float(normalized_mae(prediction_rows, holdout_rows, dataset.metric_scales, eps=dataset.eps)),
        "hazard_paths": hazard_paths,
        "incidence_paths": incidence_paths,
        "constraints": constraints,
        "constraint_application": constraint_application,
        "observation_model": observation_model,
    }


def _carry_forward_full_result(*, dataset: Any, reference_config: dict[str, Any]) -> dict[str, Any]:
    incidence_paths = fit_incidence_flow_paths(dataset, reference_config["incidence_cfg"])
    return simulate_holdout(
        dataset,
        carry_forward_hazards(dataset, mode="last_train"),
        incidence_inflow_map=dict(incidence_paths.get("holdout_incidence_inflow_map") or {}),
        incidence_hazard_map={},
        population_denominator_map=dict(incidence_paths.get("holdout_population_denominator_map") or {}),
        attrition_outflow_map=dict(incidence_paths.get("holdout_attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict(incidence_paths.get("holdout_state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=dict(incidence_paths.get("holdout_exit_channel_state_outflow_map") or {}),
    )


def _evaluate_full_split(
    *,
    family: str,
    observation_rows: list[dict[str, Any]],
    validation_targets: dict[int, dict[str, Any]],
    split: dict[str, Any],
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    dataset = build_blocked_time_dataset(observation_rows, list(split["holdout_years"]))
    if not dataset.train_transition_rows or not dataset.holdout_rows:
        return None
    split_constraints = _filter_rows_before_holdout(constraint_rows, list(split["holdout_years"]))
    incidence_result = _evaluate_incidence_split(
        family=family,
        dataset=dataset,
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        targets=validation_targets,
        train_end_year=int(split["train_end_year"]),
        horizon_years=len(list(split["holdout_years"])),
    )
    if str(incidence_result.get("fit_status") or "") != "completed":
        return {
            "family": family,
            "train_end_year": int(split["train_end_year"]),
            "train_years": list(split["train_years"]),
            "holdout_years": list(split["holdout_years"]),
            "fit_status": str(incidence_result.get("fit_status") or "failed_closed"),
            "fit_blocker": str(incidence_result.get("fit_blocker") or "incidence_branch_failed"),
            "candidate_mae": float("inf"),
            "carry_forward_mae": float("inf"),
            "prediction_rows": [],
            "carry_forward_prediction_rows": [],
            "holdout_rows": list(dataset.holdout_rows),
            "metric_scales": dict(dataset.metric_scales),
        }
    candidate = _fit_candidate_backbone(
        dataset=dataset,
        reference_config=reference_config,
        constraint_rows=split_constraints,
        incidence_result=incidence_result,
    )
    carry = _carry_forward_full_result(dataset=dataset, reference_config=reference_config)
    return {
        "family": family,
        "train_end_year": int(split["train_end_year"]),
        "train_years": list(split["train_years"]),
        "holdout_years": list(split["holdout_years"]),
        "fit_status": "completed",
        "candidate_mae": float(candidate["mae"]),
        "carry_forward_mae": float(carry["mae"]),
        "candidate_minus_carry_forward_mae": float(candidate["mae"]) - float(carry["mae"]),
        "incidence_annual_score_entries": list(incidence_result.get("annual_score_entries") or []),
        "incidence_contract": dict(candidate.get("incidence_paths") or {}),
        "constraint_application": dict(candidate.get("constraint_application") or {}),
        "prediction_rows": list(candidate.get("prediction_rows") or []),
        "carry_forward_prediction_rows": list(carry.get("prediction_rows") or []),
        "holdout_rows": list(dataset.holdout_rows),
        "metric_scales": dict(dataset.metric_scales),
    }


def _annual_incidence_gate_from_rows(rows: list[dict[str, Any]], r10_reference: dict[str, Any]) -> dict[str, Any]:
    entries = [
        entry
        for row in rows
        for entry in list(row.get("incidence_annual_score_entries") or [])
        if _finite_float(entry.get("candidate_norm_error")) is not None
        and _finite_float(entry.get("carry_forward_norm_error")) is not None
    ]
    candidate = _mean([float(entry["candidate_norm_error"]) for entry in entries])
    carry = _mean([float(entry["carry_forward_norm_error"]) for entry in entries])
    r10_annual = _finite_float(r10_reference.get("reference_annual_incidence_error"))
    blockers: list[str] = []
    if not entries:
        blockers.append("no_annual_incidence_validation_entries")
    if candidate is None or carry is None:
        blockers.append("annual_incidence_score_not_finite")
    elif candidate >= carry:
        blockers.append("annual_incidence_not_better_than_carry_forward")
    if r10_annual is not None and candidate is not None and candidate >= r10_annual:
        blockers.append("annual_incidence_not_better_than_r10")
    return {
        "status": "pass" if not blockers else "fail",
        "entry_count": len(entries),
        "candidate_norm_mae": candidate,
        "carry_forward_norm_mae": carry,
        "r10_annual_incidence_error": r10_annual,
        "blockers": blockers,
        "contract": "annual_new_infections are validation-only; pass means the readout is useful as an annual measurement forecast, not a fitted quarterly incidence truth source",
    }


def _full_path_gate(*, rows: list[dict[str, Any]], gate_name: str) -> dict[str, Any]:
    score = _score_rows(rows)
    blockers: list[str] = []
    if int(score.get("split_count") or 0) <= 0:
        blockers.append("no_scored_full_path_splits")
    candidate_mean = _finite_float(score.get("candidate_mean_mae"))
    carry_mean = _finite_float(score.get("carry_forward_mean_mae"))
    candidate_worst = _finite_float(score.get("candidate_worst_mae"))
    carry_worst = _finite_float(score.get("carry_forward_worst_mae"))
    if candidate_mean is None or carry_mean is None or candidate_mean >= carry_mean:
        blockers.append("candidate_full_path_mean_not_better_than_carry_forward")
    if candidate_worst is None or carry_worst is None or candidate_worst > carry_worst:
        blockers.append("candidate_full_path_worst_worse_than_carry_forward")
    return {
        "schema_version": "phase3_dynamic_incidence_readout_full_path_gate.v1",
        "gate_name": gate_name,
        "status": "pass" if not blockers else "fail",
        **score,
        "blockers": blockers,
        "contract": "state trajectory is scored with latent interval-aware incidence inflow only; annual incidence measurement readout is not injected into U",
    }


def _long_horizon_status(
    *,
    family: str,
    observation_rows: list[dict[str, Any]],
    validation_targets: dict[int, dict[str, Any]],
    epigraph_root: Path,
    source_run_id: str,
    baseline_source_run_id: str,
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    scenario_start_year: int,
    scenario_end_year: int,
) -> dict[str, Any]:
    future_quarters = _future_quarters(scenario_start_year, scenario_end_year)
    dataset = _projection_dataset(observation_rows, future_quarters)
    train_end_year = max(quarter_year(str(row.get("quarter") or "")) for row in dataset.train_rows)
    incidence_result = _evaluate_incidence_split(
        family=family,
        dataset=dataset,
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        targets=validation_targets,
        train_end_year=int(train_end_year),
        horizon_years=int(scenario_end_year) - int(scenario_start_year) + 1,
    )
    if str(incidence_result.get("fit_status") or "") != "completed":
        return {
            "status": "fail",
            "reason": str(incidence_result.get("fit_blocker") or "incidence_branch_failed"),
            "trust_projection": False,
            "unstable_zero_stock_quarter_count": 0,
            "first_zero_stock_quarter": None,
        }
    candidate = _fit_candidate_backbone(
        dataset=dataset,
        reference_config=reference_config,
        constraint_rows=constraint_rows,
        incidence_result=incidence_result,
    )
    zero_quarters = []
    cone_violations = []
    for prediction, trajectory in zip(list(candidate.get("prediction_rows") or []), list(candidate.get("trajectory_rows") or [])):
        quarter = str(prediction.get("quarter") or "")
        state_values = dict(trajectory.get("state_values") or {})
        if any(float(state_values.get(name) or 0.0) <= 0.0 for name in ("D", "A")):
            zero_quarters.append(quarter)
        diagnosed = float(prediction.get("diagnosed_plhiv") or 0.0)
        art = float(prediction.get("alive_on_art") or 0.0)
        suppressed = float(prediction.get("virally_suppressed") or 0.0)
        if suppressed > art + 1e-6 or art > diagnosed + 1e-6:
            cone_violations.append(quarter)
    final_prediction = list(candidate.get("prediction_rows") or [])[-1] if candidate.get("prediction_rows") else {}
    return {
        "status": "pass" if not zero_quarters and not cone_violations else "fail",
        "family": family,
        "scenario_start_year": int(scenario_start_year),
        "scenario_end_year": int(scenario_end_year),
        "trust_projection": False,
        "trust_projection_reason": "projection can be trusted only if one-year, five-year, shock-aware, and long-horizon gates all pass",
        "unstable_zero_stock_quarter_count": len(zero_quarters),
        "first_zero_stock_quarter": zero_quarters[0] if zero_quarters else None,
        "cascade_cone_violation_count": len(cone_violations),
        "first_cascade_cone_violation_quarter": cone_violations[0] if cone_violations else None,
        "final_prediction": final_prediction,
        "constraint_application": dict(candidate.get("constraint_application") or {}),
    }


def _promotion_gate(
    *,
    annual_gate: dict[str, Any],
    one_year_gate: dict[str, Any],
    five_year_gate: dict[str, Any],
    shock_gate: dict[str, Any],
    long_horizon_status: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if str(annual_gate.get("status") or "") != "pass":
        blockers.append("fails_annual_incidence_measurement_gate")
        blockers.extend(str(value) for value in list(annual_gate.get("blockers") or []))
    if str(one_year_gate.get("status") or "") != "pass":
        blockers.append("fails_one_year_full_path_gate")
        blockers.extend(str(value) for value in list(one_year_gate.get("blockers") or []))
    if str(five_year_gate.get("status") or "") != "pass":
        blockers.append("fails_five_year_full_path_gate")
        blockers.extend(str(value) for value in list(five_year_gate.get("blockers") or []))
    if str(shock_gate.get("status") or "") != "pass":
        blockers.append("fails_shock_aware_lifted_trajectory_gate")
        blockers.extend(str(value) for value in list(shock_gate.get("blockers") or []))
    if str(long_horizon_status.get("status") or "") != "pass":
        blockers.append("fails_long_horizon_bounded_state_gate")
    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "promote_full_cascade_claim" if not blockers else "reject_full_cascade_claim",
        "promotion_eligible": bool(not blockers),
        "blockers": blockers,
        "claim_boundary": (
            "If rejected, retain the branch only as the narrower annual incidence measurement-readout result; "
            "do not claim a full-cascade mechanistic champion."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    stored_rows: list[dict[str, Any]] = []
    for row in rows:
        stored = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "prediction_rows",
                "carry_forward_prediction_rows",
                "holdout_rows",
                "metric_scales",
                "incidence_contract",
            }
        }
        stored_rows.append(stored)
    if not stored_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in stored_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stored_rows:
            writer.writerow(row)


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    one_year = dict(payload.get("one_year_full_path_gate") or {})
    five_year = dict(payload.get("five_year_full_path_gate") or {})
    annual = dict(payload.get("annual_incidence_measurement_gate") or {})
    shock = dict(payload.get("shock_aware_lifted_trajectory_gate") or {})
    long_horizon = dict(payload.get("long_horizon_status") or {})
    promotion = dict(payload.get("promotion_gate") or {})
    colors = {"candidate": "#9f1239", "carry": "#0f766e", "r10": "#111827"}

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6), constrained_layout=True)
        _write_dashboard_panels(fig, axes, annual, one_year, five_year, shock, long_horizon, promotion, colors)
        fig.savefig(path, dpi=330, bbox_inches="tight")
        plt.close(fig)


def _write_dashboard_panels(
    fig: Any,
    axes: Any,
    annual: dict[str, Any],
    one_year: dict[str, Any],
    five_year: dict[str, Any],
    shock: dict[str, Any],
    long_horizon: dict[str, Any],
    promotion: dict[str, Any],
    colors: dict[str, str],
) -> None:
    flat = axes.ravel()

    annual_labels = ["candidate", "carry-forward"]
    annual_values = [
        float(annual.get("candidate_norm_mae") or 0.0),
        float(annual.get("carry_forward_norm_mae") or 0.0),
    ]
    x = np.arange(len(annual_labels))
    flat[0].bar(x, annual_values, color=[colors["candidate"], colors["carry"]])
    r10_value = _finite_float(annual.get("r10_annual_incidence_error"))
    if r10_value is not None:
        flat[0].axhline(float(r10_value), color=colors["r10"], linewidth=1.2, linestyle="--", label="R10")
        flat[0].legend(frameon=False)
    flat[0].set_xticks(x, annual_labels, rotation=0)
    flat[0].set_title("A. Annual Incidence Readout Gate", loc="left", fontweight="bold")
    flat[0].set_ylabel("Norm. MAE")

    gate_labels = ["1-year", "5-year"]
    candidate_values = [
        float(one_year.get("candidate_mean_mae") or 0.0),
        float(five_year.get("candidate_mean_mae") or 0.0),
    ]
    carry_values = [
        float(one_year.get("carry_forward_mean_mae") or 0.0),
        float(five_year.get("carry_forward_mean_mae") or 0.0),
    ]
    gx = np.arange(len(gate_labels))
    flat[1].bar(gx - 0.18, candidate_values, width=0.36, color=colors["candidate"], label="candidate latent incidence path")
    flat[1].bar(gx + 0.18, carry_values, width=0.36, color=colors["carry"], label="carry-forward")
    flat[1].set_xticks(gx, gate_labels)
    flat[1].set_title("B. Full Cascade Blocked-Time Gate", loc="left", fontweight="bold")
    flat[1].set_ylabel("Norm. MAE")
    flat[1].legend(frameon=False)

    shock_overall = dict(shock.get("overall") or {})
    shock_values = [
        float(shock_overall.get("candidate_mean_mae") or 0.0),
        float(shock_overall.get("carry_forward_mean_mae") or 0.0),
    ]
    sx = np.arange(2)
    flat[2].bar(sx, shock_values, color=[colors["candidate"], colors["carry"]])
    r10_quarter = _finite_float((shock.get("r10_reference") or {}).get("reference_quarterly_mean_mae"))
    if r10_quarter is not None:
        flat[2].axhline(float(r10_quarter), color=colors["r10"], linewidth=1.2, linestyle="--", label="R10")
        flat[2].legend(frameon=False)
    flat[2].set_xticks(sx, ["candidate", "carry-forward"])
    flat[2].set_title("C. Shock-Aware Lifted Trajectory Gate", loc="left", fontweight="bold")
    flat[2].set_ylabel("Quarter-level norm. MAE")

    claim = textwrap.wrap(str(promotion.get("claim_boundary") or ""), width=68)
    text = [
        "Full-gate interpretation",
        f"Annual readout: {annual.get('status')}",
        f"1-year full path: {one_year.get('status')}",
        f"5-year full path: {five_year.get('status')}",
        f"Shock path: {shock.get('status')}",
        f"Long horizon: {long_horizon.get('status')}",
        f"Promotion: {promotion.get('status')}",
        "",
        "Claim boundary:",
        *claim,
        "",
        "Top blockers:",
    ]
    text.extend([f"- {value}" for value in list(promotion.get("blockers") or [])[:8]] or ["- none"])
    flat[3].axis("off")
    flat[3].text(0.0, 1.0, "\n".join(text), va="top", ha="left", family="monospace", fontsize=8.2)

    for ax in flat[:3]:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Phase 3 Incidence Readout Full-Cascade Gate", fontweight="bold")


def run_incidence_readout_full_gate(
    *,
    run_id: str = "p3d-incidence-readout-full-gate-20260429-s00",
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    reference_report_path: str | Path | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    scenario_start_year: int = 2026,
    scenario_end_year: int = 2035,
    family: str = PROMOTED_INCIDENCE_FAMILY,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    active_source_run_id = resolve_active_source_run_id(
        epigraph_root,
        str(source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID),
    )
    active_baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=active_source_run_id,
        preferred=str(baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID),
    )
    observation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=active_source_run_id,
        baseline_source_run_id=active_baseline_source_run_id,
    )
    validation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=active_source_run_id,
        baseline_source_run_id=active_baseline_source_run_id,
        include_validation_only=True,
    )
    targets = _annual_incidence_targets(validation_rows)
    reference_config = _load_reference_config(Path(reference_report_path) if reference_report_path else None)
    r10_baseline = _load_r10_baseline(epigraph_root)
    r10_reference = _r10_reference_scores(r10_baseline)
    r10_reference["reference_annual_incidence_error"] = _r10_annual_incidence_error(epigraph_root)
    one_year_rows: list[dict[str, Any]] = []
    five_year_rows: list[dict[str, Any]] = []
    for horizon_years, target_rows in ((1, one_year_rows), (5, five_year_rows)):
        splits = rolling_origin_splits(
            observation_rows,
            start_year=int(start_year),
            end_year=int(end_year),
            min_train_years=int(min_train_years),
            horizon_years=int(horizon_years),
        )
        for split in splits:
            result = _evaluate_full_split(
                family=family,
                observation_rows=observation_rows,
                validation_targets=targets,
                split=split,
                epigraph_root=epigraph_root,
                source_run_id=active_source_run_id,
                baseline_source_run_id=active_baseline_source_run_id,
                reference_config=reference_config,
                constraint_rows=validation_rows,
            )
            if result is not None:
                target_rows.append(result)
    annual_gate = _annual_incidence_gate_from_rows(one_year_rows + five_year_rows, r10_reference)
    one_year_gate = _full_path_gate(rows=one_year_rows, gate_name="one_year_full_cascade")
    five_year_gate = _full_path_gate(rows=five_year_rows, gate_name="five_year_full_cascade")
    shock_catalog = _detect_shock_catalog(observation_rows)
    shock_gate = _shock_aware_lifted_gate_from_evaluated(
        family=family,
        evaluated_rows=one_year_rows + five_year_rows,
        shock_catalog=shock_catalog,
        r10_baseline=r10_baseline,
    )
    long_status = _long_horizon_status(
        family=family,
        observation_rows=observation_rows,
        validation_targets=targets,
        epigraph_root=epigraph_root,
        source_run_id=active_source_run_id,
        baseline_source_run_id=active_baseline_source_run_id,
        reference_config=reference_config,
        constraint_rows=validation_rows,
        scenario_start_year=scenario_start_year,
        scenario_end_year=scenario_end_year,
    )
    promotion = _promotion_gate(
        annual_gate=annual_gate,
        one_year_gate=one_year_gate,
        five_year_gate=five_year_gate,
        shock_gate=shock_gate,
        long_horizon_status=long_status,
    )
    long_status["trust_projection"] = bool(promotion.get("promotion_eligible"))
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report_path = analysis_dir / "incidence_readout_full_gate_report.json"
    rows_csv_path = analysis_dir / "incidence_readout_full_gate_rows.csv"
    dashboard_path = analysis_dir / "incidence_readout_full_gate_dashboard.png"
    payload = {
        "schema_version": INCIDENCE_READOUT_FULL_GATE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": active_source_run_id,
        "baseline_source_run_id": active_baseline_source_run_id,
        "family": family,
        "loop_variant": "evidence-to-model-loop",
        "contract": {
            "tested_question": "does the promoted annual incidence readout survive full Phase3 cascade gates without converting validation-only annual incidence into fitted quarterly state truth",
            "state_incidence_path": "latent interval-aware S_eff incidence inflow from the incidence module enters U",
            "measurement_readout_path": "annual measurement readout remains validation/readout-only and is not injected into U",
            "promotion_rule": "must pass annual incidence, one-year full path, five-year full path, shock-aware lifted path, and bounded long-horizon gates",
        },
        "benchmark_contract": {
            "start_year": int(start_year),
            "end_year": int(end_year),
            "min_train_years": int(min_train_years),
            "one_year_split_count": len(one_year_rows),
            "five_year_split_count": len(five_year_rows),
        },
        "scenario_gate_contract": {
            "scenario_start_year": int(scenario_start_year),
            "scenario_end_year": int(scenario_end_year),
        },
        "observation_role_ledger_summary": dict(
            build_observation_role_ledger(
                epigraph_root,
                source_run_id=active_source_run_id,
                baseline_source_run_id=active_baseline_source_run_id,
            ).get("summary")
            or {}
        ),
        "r10_reference": r10_reference,
        "annual_incidence_measurement_gate": annual_gate,
        "one_year_full_path_gate": one_year_gate,
        "five_year_full_path_gate": five_year_gate,
        "shock_aware_lifted_trajectory_gate": shock_gate,
        "long_horizon_status": long_status,
        "promotion_gate": promotion,
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "prediction_rows",
                    "carry_forward_prediction_rows",
                    "holdout_rows",
                    "metric_scales",
                    "incidence_contract",
                }
            }
            for row in one_year_rows + five_year_rows
        ],
        "artifact_paths": {
            "report_json": report_path.as_posix(),
            "rows_csv": rows_csv_path.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(report_path, payload)
    _write_csv(rows_csv_path, one_year_rows + five_year_rows)
    _write_dashboard(payload, dashboard_path)
    write_json(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate the promoted incidence readout through full Phase3 cascade criteria.")
    parser.add_argument("--run-id", default="p3d-incidence-readout-full-gate-20260429-s00")
    parser.add_argument("--source-run-id")
    parser.add_argument("--baseline-source-run-id")
    parser.add_argument("--reference-report-path")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--scenario-start-year", type=int, default=2026)
    parser.add_argument("--scenario-end-year", type=int, default=2035)
    parser.add_argument("--family", default=PROMOTED_INCIDENCE_FAMILY)
    args = parser.parse_args()
    payload = run_incidence_readout_full_gate(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        reference_report_path=args.reference_report_path,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
        scenario_start_year=args.scenario_start_year,
        scenario_end_year=args.scenario_end_year,
        family=args.family,
    )
    print(payload["artifact_paths"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
