from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3._lineage.national_reset_core import quarter_sort_key
from epigraph_ph.phase3.frontier.transition_engine import _discover_latest_transition_experiment
from epigraph_ph.runtime import ROOT_DIR, read_json, write_json

from .artifacts import IncidenceResearchContext, write_experiment_artifacts
from .audits import DIRECT_INFLOW_METRICS, _base_numeric_policy, _metric_rows, _year_from_row
from epigraph_ph.phase3.shared.numerics import safe_floor
from .sources import load_incidence_audit_inputs


INC01B_EXPERIMENT_ID = "INC-01B-backlog-vs-incidence-swap-stress-test"
INC01D_EXPERIMENT_ID = "INC-01D-diagnosis-locked-incidence-branch"
INCV201_EXPERIMENT_ID = "INC-V2-01-observed-denominator-explicit-incidence"
AGE01B_EXPERIMENT_ID = "AGE-01B-youth-diagnosis-modifier"
MECH01E_EXPERIMENT_ID = "MECH-01E-anchored-downstream-residual-helpers"
STATE_NAMES: tuple[str, ...] = ("U", "D", "A", "V", "L")


def _quarter_year(quarter: str) -> int:
    return int(quarter_sort_key(str(quarter))[0])


def _discover_latest_incidence_experiment(experiment_id: str) -> tuple[str, Path]:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, str, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "incidence_research" / experiment_id
        decision_path = experiment_dir / "decision.json"
        if decision_path.exists():
            candidates.append((decision_path.stat().st_mtime, run_dir.name, experiment_dir))
    if not candidates:
        raise FileNotFoundError(f"No incidence research experiment found for {experiment_id}")
    preferred = [row for row in candidates if "pytest" not in str(row[1]).lower()]
    pool = preferred or candidates
    _, run_id, experiment_dir = max(pool, key=lambda row: row[0])
    return run_id, experiment_dir


def _load_transition_reference_with_forecast(experiment_id: str) -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_transition_experiment(experiment_id)
    baseline_comparison = read_json(experiment_dir / "baseline_comparison.json", default={})
    evaluation = read_json(experiment_dir / "evaluation.json", default={})
    mechanistic_forecast = read_json(experiment_dir / "mechanistic_forecast.json", default={})
    state_trajectory_rows = read_json(experiment_dir / "state_trajectory_rows.json", default=[])
    transition_hazard_summary = read_json(experiment_dir / "transition_hazard_summary.json", default={})
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "baseline_comparison": dict(baseline_comparison or {}),
        "evaluation": dict(evaluation or {}),
        "mechanistic_forecast": dict(mechanistic_forecast or {}),
        "state_trajectory_rows": [dict(row) for row in list(state_trajectory_rows or [])],
        "transition_hazard_summary": dict(transition_hazard_summary or {}),
    }


def _load_incidence_reference(experiment_id: str) -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_incidence_experiment(experiment_id)
    decision = read_json(experiment_dir / "decision.json", default={})
    evaluation = read_json(experiment_dir / "evaluation.json", default={})
    fit_artifact = read_json(experiment_dir / "fit_artifact.json", default={})
    incidence_flow_summary = read_json(experiment_dir / "incidence_flow_summary.json", default={})
    diagnosis_locked_incidence_summary = read_json(experiment_dir / "diagnosis_locked_incidence_summary.json", default={})
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "decision": dict(decision or {}),
        "evaluation": dict(evaluation or {}),
        "fit_artifact": dict(fit_artifact or {}),
        "incidence_flow_summary": dict(incidence_flow_summary or {}),
        "diagnosis_locked_incidence_summary": dict(diagnosis_locked_incidence_summary or {}),
    }


def _diagnosis_flow_from_rows(previous_row: dict[str, Any], current_row: dict[str, Any]) -> float:
    direct_value = current_row.get("new_diagnosed_cases_period")
    if direct_value is not None:
        return max(float(direct_value), 0.0)
    previous_state = dict(previous_row.get("state_values") or {})
    current_state = dict(current_row.get("state_values") or {})
    previous_total = sum(float(previous_state.get(name) or 0.0) for name in ("D", "A", "V", "L"))
    current_total = sum(float(current_state.get(name) or 0.0) for name in ("D", "A", "V", "L"))
    return max(float(current_total - previous_total), 0.0)


def _combined_locked_state_rows() -> dict[str, Any]:
    age_01b_reference = _load_transition_reference_with_forecast(AGE01B_EXPERIMENT_ID)
    mech_01e_reference = _load_transition_reference_with_forecast(MECH01E_EXPERIMENT_ID)
    age_evaluation = dict(age_01b_reference.get("evaluation") or {})
    holdout_quarters = [str(value) for value in list(age_evaluation.get("holdout_quarters") or []) if str(value)]
    if not holdout_quarters:
        raise ValueError("INC-01D requires holdout quarters from AGE-01B")
    first_holdout_quarter = min(holdout_quarters, key=quarter_sort_key)
    holdout_forecast_rows = [dict(row) for row in list((age_01b_reference.get("mechanistic_forecast") or {}).get("forecast_rows") or [])]
    if not holdout_forecast_rows:
        raise ValueError("INC-01D requires forecast_rows from AGE-01B mechanistic forecast")
    historical_rows = [
        dict(row)
        for row in list(mech_01e_reference.get("state_trajectory_rows") or [])
        if quarter_sort_key(str(row.get("quarter") or "")) < quarter_sort_key(first_holdout_quarter)
    ]
    combined_rows = historical_rows + holdout_forecast_rows
    combined_rows.sort(key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    return {
        "age_01b_reference": age_01b_reference,
        "mech_01e_reference": mech_01e_reference,
        "combined_state_rows": combined_rows,
        "holdout_quarters": holdout_quarters,
    }


def _build_incidence_flow_rows(combined_state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for previous_row, current_row in zip(combined_state_rows[:-1], combined_state_rows[1:]):
        previous_state = dict(previous_row.get("state_values") or {})
        current_state = dict(current_row.get("state_values") or {})
        prev_u = float(previous_state.get("U") or 0.0)
        curr_u = float(current_state.get("U") or 0.0)
        diagnosis_flow = _diagnosis_flow_from_rows(previous_row, current_row)
        mass_balance_term = float(curr_u - prev_u + diagnosis_flow)
        latent_incidence_inflow = max(mass_balance_term, 0.0)
        residual_undiagnosed_clearance = max(-mass_balance_term, 0.0)
        rows.append(
            {
                "previous_quarter": str(previous_row.get("quarter") or ""),
                "quarter": str(current_row.get("quarter") or ""),
                "year": int(_quarter_year(str(current_row.get("quarter") or ""))),
                "u_previous": prev_u,
                "u_current": curr_u,
                "u_delta": float(curr_u - prev_u),
                "diagnosis_flow": diagnosis_flow,
                "mass_balance_term": mass_balance_term,
                "latent_incidence_inflow": latent_incidence_inflow,
                "residual_undiagnosed_clearance": residual_undiagnosed_clearance,
            }
        )
    return rows


def _annual_validation_rows(
    incidence_flow_rows: list[dict[str, Any]],
    historical_metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annual_targets = {
        int(_year_from_row(row)): float(row.get("value") or 0.0)
        for row in _metric_rows(historical_metric_rows, DIRECT_INFLOW_METRICS)
        if _year_from_row(row) is not None
    }
    annual_inflow: dict[int, float] = {}
    annual_diagnosis: dict[int, float] = {}
    annual_delta_u: dict[int, float] = {}
    annual_clearance: dict[int, float] = {}
    for row in incidence_flow_rows:
        year_value = int(row["year"])
        annual_inflow[year_value] = float(annual_inflow.get(year_value, 0.0)) + float(row["latent_incidence_inflow"])
        annual_diagnosis[year_value] = float(annual_diagnosis.get(year_value, 0.0)) + float(row["diagnosis_flow"])
        annual_delta_u[year_value] = float(annual_delta_u.get(year_value, 0.0)) + float(row["u_delta"])
        annual_clearance[year_value] = float(annual_clearance.get(year_value, 0.0)) + float(row["residual_undiagnosed_clearance"])
    validation_rows: list[dict[str, Any]] = []
    for year_value in sorted(set(annual_inflow) | set(annual_targets)):
        predicted_inflow = float(annual_inflow.get(year_value, 0.0))
        target_inflow = annual_targets.get(year_value)
        absolute_error = abs(predicted_inflow - float(target_inflow)) if target_inflow is not None else None
        validation_rows.append(
            {
                "year": int(year_value),
                "predicted_inflow": predicted_inflow,
                "target_annual_new_infections": float(target_inflow) if target_inflow is not None else None,
                "absolute_error": float(absolute_error) if absolute_error is not None else None,
                "annual_diagnosis_flow": float(annual_diagnosis.get(year_value, 0.0)),
                "annual_u_delta": float(annual_delta_u.get(year_value, 0.0)),
                "annual_residual_undiagnosed_clearance": float(annual_clearance.get(year_value, 0.0)),
            }
        )
    return validation_rows


def _official_population_lookup(inputs: Any) -> tuple[dict[int, float], str | None]:
    rows = [dict(row) for row in list(getattr(inputs, "official_population_denominator_rows", []) or [])]
    lookup: dict[int, float] = {}
    for row in rows:
        year = row.get("year")
        population_total = row.get("population_total")
        if year in (None, "") or population_total in (None, ""):
            continue
        try:
            lookup[int(year)] = float(population_total)
        except (TypeError, ValueError):
            continue
    path = getattr(inputs, "official_population_denominator_path", None)
    return lookup, (str(path) if path is not None else None)


def _build_explicit_incidence_rows(
    incidence_flow_rows: list[dict[str, Any]],
    population_lookup: dict[int, float],
) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    missing_years: list[int] = []
    for row in incidence_flow_rows:
        year_value = int(row["year"])
        denominator = population_lookup.get(year_value)
        if denominator is None:
            missing_years.append(year_value)
            rows.append(
                {
                    "year": year_value,
                    "quarter": str(row.get("quarter") or ""),
                    "latent_incidence_inflow": float(row.get("latent_incidence_inflow") or 0.0),
                    "diagnosis_flow": float(row.get("diagnosis_flow") or 0.0),
                    "population_denominator": None,
                    "incidence_hazard": None,
                }
            )
            continue
        inflow = float(row.get("latent_incidence_inflow") or 0.0)
        hazard = inflow / safe_floor(denominator)
        rows.append(
            {
                "year": year_value,
                "quarter": str(row.get("quarter") or ""),
                "latent_incidence_inflow": inflow,
                "diagnosis_flow": float(row.get("diagnosis_flow") or 0.0),
                "population_denominator": float(denominator),
                "incidence_hazard": float(hazard),
            }
        )
    return rows, sorted(set(missing_years))


def _population_denominator_summary_rows(population_lookup: dict[int, float]) -> list[dict[str, Any]]:
    return [
        {
            "year": int(year),
            "population_denominator": float(value),
        }
        for year, value in sorted(population_lookup.items())
    ]


def _save_state_npz(path: Path, rows: list[dict[str, Any]]) -> None:
    quarter = np.asarray([str(row.get("quarter") or "") for row in rows], dtype="<U16")
    payload: dict[str, Any] = {"quarter": quarter}
    for state_name in STATE_NAMES:
        payload[state_name] = np.asarray(
            [float(dict(row.get("state_values") or {}).get(state_name) or 0.0) for row in rows],
            dtype=np.float32,
        )
    payload["diagnosed_plhiv"] = np.asarray([float(row.get("diagnosed_plhiv") or 0.0) for row in rows], dtype=np.float32)
    payload["alive_on_art"] = np.asarray([float(row.get("alive_on_art") or 0.0) for row in rows], dtype=np.float32)
    payload["new_diagnosed_cases_period"] = np.asarray([float(row.get("new_diagnosed_cases_period") or 0.0) for row in rows], dtype=np.float32)
    np.savez(path, **payload)


def _plot_locked_baseline_vs_incidence(
    path: Path,
    holdout_rows: list[dict[str, Any]],
    annual_validation_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    diagnosed_prediction = [float(dict(row.get("prediction") or {}).get("diagnosed_plhiv") or 0.0) for row in holdout_rows]
    diagnosed_target = [float(dict(row.get("target") or {}).get("diagnosed_plhiv") or 0.0) for row in holdout_rows]
    art_prediction = [float(dict(row.get("prediction") or {}).get("alive_on_art") or 0.0) for row in holdout_rows]
    art_target = [float(dict(row.get("target") or {}).get("alive_on_art") or 0.0) for row in holdout_rows]
    axes[0].plot(holdout_quarters, diagnosed_prediction, label="diagnosed prediction", marker="o")
    axes[0].plot(holdout_quarters, diagnosed_target, label="diagnosed target", marker="o")
    axes[0].plot(holdout_quarters, art_prediction, label="ART prediction", marker="o")
    axes[0].plot(holdout_quarters, art_target, label="ART target", marker="o")
    axes[0].set_title("Locked Holdout Forecast")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8)

    years = [str(row["year"]) for row in annual_validation_rows if row.get("target_annual_new_infections") is not None]
    predicted = [float(row["predicted_inflow"]) for row in annual_validation_rows if row.get("target_annual_new_infections") is not None]
    target = [float(row["target_annual_new_infections"]) for row in annual_validation_rows if row.get("target_annual_new_infections") is not None]
    axes[1].plot(years, predicted, label="implied inflow", marker="o")
    axes[1].plot(years, target, label="annual new infections", marker="o")
    axes[1].set_title("Annual Inflow Compatibility")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_inc_01d(ctx: IncidenceResearchContext) -> dict[str, Any]:
    inputs = load_incidence_audit_inputs(ctx)
    lineage = _combined_locked_state_rows()
    age_01b_reference = dict(lineage["age_01b_reference"])
    combined_state_rows = [dict(row) for row in list(lineage["combined_state_rows"])]
    holdout_quarters = [str(value) for value in list(lineage["holdout_quarters"])]
    incidence_flow_rows = _build_incidence_flow_rows(combined_state_rows)
    annual_validation_rows = _annual_validation_rows(incidence_flow_rows, inputs.historical_metric_rows)

    holdout_rows = [dict(row) for row in list((age_01b_reference.get("evaluation") or {}).get("holdout_rows") or [])]
    baseline_comparison = dict(age_01b_reference.get("baseline_comparison") or {})
    model_mae = float(baseline_comparison.get("model_mean_absolute_error") or 0.0)
    model_smape = float(baseline_comparison.get("model_smape") or 0.0)
    diagnosis_flow_mae = float(baseline_comparison.get("diagnosis_flow_mean_absolute_error") or 0.0)
    annual_validation_errors = [float(row["absolute_error"]) for row in annual_validation_rows if row.get("absolute_error") is not None]
    annual_validation_mae = float(np.mean(annual_validation_errors)) if annual_validation_errors else 0.0
    total_inflow = float(sum(float(row["latent_incidence_inflow"]) for row in incidence_flow_rows))
    total_clearance = float(sum(float(row["residual_undiagnosed_clearance"]) for row in incidence_flow_rows))
    forecast_locked_exactly = True

    baseline_summary = {
        "branch_reference_experiment_id": AGE01B_EXPERIMENT_ID,
        "branch_reference_label": "AGE-01B",
        "branch_reference_mean_absolute_error": model_mae,
        "branch_reference_diagnosis_flow_mean_absolute_error": diagnosis_flow_mae,
        "carry_forward_mean_absolute_error": float(baseline_comparison.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(baseline_comparison.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "diagnosis_flow_mean_absolute_error": diagnosis_flow_mae,
        "model_matches_locked_baseline": forecast_locked_exactly,
        "annual_incidence_validation_mean_absolute_error": round(float(annual_validation_mae), 6),
    }
    evaluation = {
        "mode": "diagnosis_locked_incidence_branch",
        "comparison_reference_run_id": str(age_01b_reference["reference_run_id"]),
        "holdout_quarters": holdout_quarters,
        "holdout_rows": holdout_rows,
        "annual_incidence_validation": annual_validation_rows,
        "forecast_locked_exactly": forecast_locked_exactly,
    }
    fit_artifact = {
        "annual_validation_row_count": int(len(annual_validation_errors)),
        "annual_validation_mean_absolute_error": round(float(annual_validation_mae), 6),
        "total_latent_incidence_inflow": round(float(total_inflow), 6),
        "total_residual_undiagnosed_clearance": round(float(total_clearance), 6),
        "clearance_share_of_inflow": round(float(total_clearance / max(total_inflow, np.finfo(np.float32).eps)), 6) if total_inflow > 0.0 else None,
        "forecast_locked_exactly": forecast_locked_exactly,
    }
    flow_summary = {
        "incidence_flow_rows": incidence_flow_rows,
        "annual_validation_rows": annual_validation_rows,
    }
    diagnosis_locked_summary = {
        "reference_experiment_id": AGE01B_EXPERIMENT_ID,
        "reference_run_id": str(age_01b_reference["reference_run_id"]),
        "train_quarters": [str(row.get("quarter") or "") for row in combined_state_rows if str(row.get("quarter") or "") not in set(holdout_quarters)],
        "holdout_quarters": holdout_quarters,
        "annual_validation_row_count": int(len(annual_validation_errors)),
        "annual_validation_mean_absolute_error": round(float(annual_validation_mae), 6),
        "forecast_locked_exactly": forecast_locked_exactly,
    }

    _save_state_npz(ctx.experiment_dir / "state_estimates.npz", combined_state_rows)
    holdout_state_rows = [row for row in combined_state_rows if str(row.get("quarter") or "") in set(holdout_quarters)]
    _save_state_npz(ctx.experiment_dir / "forecast_states.npz", holdout_state_rows)
    write_json(ctx.experiment_dir / "incidence_flow_summary.json", flow_summary)
    write_json(ctx.experiment_dir / "diagnosis_locked_incidence_summary.json", diagnosis_locked_summary)
    write_json(ctx.experiment_dir / "fit_artifact.json", fit_artifact)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_summary)
    _plot_locked_baseline_vs_incidence(
        ctx.experiment_dir / "forecast_vs_locked_baseline.png",
        holdout_rows=holdout_rows,
        annual_validation_rows=annual_validation_rows,
    )

    experiment_spec = {
        "experiment_id": INC01D_EXPERIMENT_ID,
        "description": ctx.experiment.description,
        "source_run_id": ctx.source_run_id,
        "expected_outputs": list(ctx.experiment.expected_outputs),
        "reference_experiment_id": AGE01B_EXPERIMENT_ID,
    }
    coverage_summary = {
        "combined_state_row_count": int(len(combined_state_rows)),
        "incidence_flow_row_count": int(len(incidence_flow_rows)),
        "holdout_quarter_count": int(len(holdout_quarters)),
        "annual_validation_row_count": int(len(annual_validation_errors)),
    }
    decision = {
        "completed": True,
        "keep": bool(forecast_locked_exactly) and float(annual_validation_mae) == 0.0,
        "reason": "INC-01D makes a pre-U inflow explicit while inheriting the kept AGE-01B diagnosis path. The branch is rejected as a forecasting baseline unless the explicit inflow is also compatible with the annual incidence series.",
        "checks": [
            {
                "name": "forecast_locked_exactly",
                "passed": forecast_locked_exactly,
                "actual": bool(forecast_locked_exactly),
                "target": True,
            },
            {
                "name": "annual_incidence_exact_compatibility",
                "passed": float(annual_validation_mae) == 0.0,
                "actual": round(float(annual_validation_mae), 6),
                "target": 0.0,
            },
        ],
    }
    numeric_justification = _base_numeric_policy() + [
        {
            "name": "non_negative_undiagnosed_mass_boundary",
            "value": 0,
            "role": "Boundary used when converting mass-balance terms into non-negative latent inflow and undiagnosed-clearance components.",
            "source_type": "physical_constraint",
            "estimation_data": "Negative incidence and negative undiagnosed clearance are not physically meaningful in the branch.",
            "estimation_method": "non-negative compartment flow constraint",
            "uncertainty": "none",
            "why_needed": "INC-01D splits the U mass-balance term into non-negative inflow and non-negative residual clearance without handwritten thresholds.",
        }
    ]
    write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=numeric_justification,
    )
    return {
        "baseline_comparison": baseline_summary,
        "evaluation": evaluation,
        "fit_artifact": fit_artifact,
        "decision": decision,
    }


def run_inc_v2_01(ctx: IncidenceResearchContext) -> dict[str, Any]:
    inputs = load_incidence_audit_inputs(ctx)
    lineage = _combined_locked_state_rows()
    age_01b_reference = dict(lineage["age_01b_reference"])
    combined_state_rows = [dict(row) for row in list(lineage["combined_state_rows"])]
    holdout_quarters = [str(value) for value in list(lineage["holdout_quarters"])]
    incidence_flow_rows = _build_incidence_flow_rows(combined_state_rows)
    annual_validation_rows = _annual_validation_rows(incidence_flow_rows, inputs.historical_metric_rows)
    population_lookup, population_source_path = _official_population_lookup(inputs)
    explicit_incidence_rows, missing_denominator_years = _build_explicit_incidence_rows(incidence_flow_rows, population_lookup)
    denominator_summary_rows = _population_denominator_summary_rows(population_lookup)

    holdout_rows = [dict(row) for row in list((age_01b_reference.get("evaluation") or {}).get("holdout_rows") or [])]
    baseline_comparison = dict(age_01b_reference.get("baseline_comparison") or {})
    model_mae = float(baseline_comparison.get("model_mean_absolute_error") or 0.0)
    model_smape = float(baseline_comparison.get("model_smape") or 0.0)
    diagnosis_flow_mae = float(baseline_comparison.get("diagnosis_flow_mean_absolute_error") or 0.0)
    annual_validation_errors = [float(row["absolute_error"]) for row in annual_validation_rows if row.get("absolute_error") is not None]
    annual_validation_mae = float(np.mean(annual_validation_errors)) if annual_validation_errors else 0.0
    hazard_values = [float(row["incidence_hazard"]) for row in explicit_incidence_rows if row.get("incidence_hazard") is not None]
    denominator_covered_years = sorted({int(row["year"]) for row in explicit_incidence_rows if row.get("population_denominator") is not None})
    flow_years = sorted({int(row["year"]) for row in incidence_flow_rows})
    denominator_coverage_complete = denominator_covered_years == flow_years

    baseline_summary = {
        "branch_reference_experiment_id": AGE01B_EXPERIMENT_ID,
        "branch_reference_label": "AGE-01B",
        "branch_reference_mean_absolute_error": model_mae,
        "branch_reference_diagnosis_flow_mean_absolute_error": diagnosis_flow_mae,
        "carry_forward_mean_absolute_error": float(baseline_comparison.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(baseline_comparison.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "diagnosis_flow_mean_absolute_error": diagnosis_flow_mae,
        "annual_incidence_validation_mean_absolute_error": round(float(annual_validation_mae), 6),
        "denominator_coverage_complete": bool(denominator_coverage_complete),
    }
    evaluation = {
        "mode": "observed_denominator_explicit_incidence",
        "comparison_reference_run_id": str(age_01b_reference["reference_run_id"]),
        "holdout_quarters": holdout_quarters,
        "holdout_rows": holdout_rows,
        "annual_incidence_validation": annual_validation_rows,
        "denominator_covered_years": denominator_covered_years,
        "missing_denominator_years": missing_denominator_years,
    }
    fit_artifact = {
        "population_source_path": population_source_path,
        "denominator_row_count": int(len(denominator_summary_rows)),
        "denominator_covered_year_count": int(len(denominator_covered_years)),
        "missing_denominator_year_count": int(len(missing_denominator_years)),
        "missing_denominator_years": missing_denominator_years,
        "annual_validation_row_count": int(len(annual_validation_errors)),
        "annual_validation_mean_absolute_error": round(float(annual_validation_mae), 6),
        "minimum_incidence_hazard": round(float(min(hazard_values)), 12) if hazard_values else None,
        "maximum_incidence_hazard": round(float(max(hazard_values)), 12) if hazard_values else None,
        "mean_incidence_hazard": round(float(np.mean(hazard_values)), 12) if hazard_values else None,
        "denominator_coverage_complete": bool(denominator_coverage_complete),
    }
    flow_summary = {
        "population_denominator_rows": denominator_summary_rows,
        "explicit_incidence_rows": explicit_incidence_rows,
        "annual_validation_rows": annual_validation_rows,
    }
    explicit_summary = {
        "reference_experiment_id": AGE01B_EXPERIMENT_ID,
        "reference_run_id": str(age_01b_reference["reference_run_id"]),
        "population_source_path": population_source_path,
        "population_source_provider": "World Bank WDI SP.POP.TOTL",
        "denominator_covered_years": denominator_covered_years,
        "missing_denominator_years": missing_denominator_years,
        "incidence_hazard_row_count": int(len(hazard_values)),
        "denominator_coverage_complete": bool(denominator_coverage_complete),
    }

    _save_state_npz(ctx.experiment_dir / "state_estimates.npz", combined_state_rows)
    holdout_state_rows = [row for row in combined_state_rows if str(row.get("quarter") or "") in set(holdout_quarters)]
    _save_state_npz(ctx.experiment_dir / "forecast_states.npz", holdout_state_rows)
    write_json(ctx.experiment_dir / "population_denominator_series.json", {"rows": denominator_summary_rows, "source_path": population_source_path})
    write_json(ctx.experiment_dir / "explicit_incidence_hazard_summary.json", explicit_summary)
    write_json(ctx.experiment_dir / "incidence_flow_summary.json", flow_summary)
    write_json(ctx.experiment_dir / "fit_artifact.json", fit_artifact)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_summary)
    _plot_locked_baseline_vs_incidence(
        ctx.experiment_dir / "forecast_vs_locked_baseline.png",
        holdout_rows=holdout_rows,
        annual_validation_rows=annual_validation_rows,
    )

    experiment_spec = {
        "experiment_id": INCV201_EXPERIMENT_ID,
        "description": ctx.experiment.description,
        "source_run_id": ctx.source_run_id,
        "expected_outputs": list(ctx.experiment.expected_outputs),
        "reference_experiment_id": AGE01B_EXPERIMENT_ID,
        "population_source_provider": "World Bank WDI SP.POP.TOTL",
    }
    coverage_summary = {
        "combined_state_row_count": int(len(combined_state_rows)),
        "incidence_flow_row_count": int(len(incidence_flow_rows)),
        "holdout_quarter_count": int(len(holdout_quarters)),
        "denominator_row_count": int(len(denominator_summary_rows)),
        "denominator_covered_year_count": int(len(denominator_covered_years)),
        "missing_denominator_year_count": int(len(missing_denominator_years)),
    }
    decision = {
        "completed": True,
        "keep": bool(denominator_coverage_complete) and float(annual_validation_mae) == 0.0,
        "reason": "INC-V2-01 makes incidence explicit as I_t = N_t * lambda_t using an observed official denominator. The branch is rejected unless denominator coverage is complete for all modeled years and the implied annual inflow remains exactly compatible with the annual incidence series.",
        "checks": [
            {
                "name": "observed_denominator_coverage_complete",
                "passed": bool(denominator_coverage_complete),
                "actual": bool(denominator_coverage_complete),
                "target": True,
            },
            {
                "name": "annual_incidence_exact_compatibility",
                "passed": float(annual_validation_mae) == 0.0,
                "actual": round(float(annual_validation_mae), 6),
                "target": 0.0,
            },
        ],
    }
    numeric_justification = _base_numeric_policy() + [
        {
            "name": "non_negative_undiagnosed_mass_boundary",
            "value": 0,
            "role": "Boundary used when converting mass-balance terms into non-negative latent inflow and non-negative undiagnosed-clearance components.",
            "source_type": "physical_constraint",
            "estimation_data": "Negative incidence and negative undiagnosed clearance are not physically meaningful in the explicit incidence branch.",
            "estimation_method": "non-negative compartment flow constraint",
            "uncertainty": "none",
            "why_needed": "INC-V2-01 keeps explicit inflow and residual clearance physically interpretable without handwritten thresholds.",
        }
    ]
    write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=numeric_justification,
    )
    return {
        "baseline_comparison": baseline_summary,
        "evaluation": evaluation,
        "fit_artifact": fit_artifact,
        "decision": decision,
    }


def run_inc_01b(ctx: IncidenceResearchContext) -> dict[str, Any]:
    incidence_reference = _load_incidence_reference(INC01D_EXPERIMENT_ID)
    flow_summary = dict(incidence_reference.get("incidence_flow_summary") or {})
    annual_validation_rows = [dict(row) for row in list(flow_summary.get("annual_validation_rows") or [])]

    margin_rows: list[dict[str, Any]] = []
    for row in annual_validation_rows:
        target_inflow = row.get("target_annual_new_infections")
        if target_inflow in (None, ""):
            continue
        target_value = float(target_inflow)
        lower_bound = max(float(row.get("predicted_inflow") or 0.0), 0.0)
        swap_slack = max(target_value - lower_bound, 0.0)
        incompatibility_excess = max(lower_bound - target_value, 0.0)
        identifiability_margin = (
            min(lower_bound, target_value) / max(target_value, np.finfo(np.float32).eps)
            if target_value > 0.0
            else 0.0
        )
        margin_rows.append(
            {
                "year": int(row["year"]),
                "validation_target_inflow": target_value,
                "mass_balance_lower_bound": lower_bound,
                "swap_slack": float(swap_slack),
                "incompatibility_excess": float(incompatibility_excess),
                "identifiability_margin": float(identifiability_margin),
                "annual_diagnosis_flow": float(row.get("annual_diagnosis_flow") or 0.0),
                "annual_u_delta": float(row.get("annual_u_delta") or 0.0),
                "annual_residual_undiagnosed_clearance": float(row.get("annual_residual_undiagnosed_clearance") or 0.0),
            }
        )

    summary_payload = {
        "reference_experiment_id": INC01D_EXPERIMENT_ID,
        "reference_run_id": str(incidence_reference["reference_run_id"]),
        "rows": margin_rows,
        "mean_identifiability_margin": round(float(np.mean([row["identifiability_margin"] for row in margin_rows])) if margin_rows else 0.0, 6),
        "min_identifiability_margin": round(float(np.min([row["identifiability_margin"] for row in margin_rows])) if margin_rows else 0.0, 6),
        "mean_incompatibility_excess": round(float(np.mean([row["incompatibility_excess"] for row in margin_rows])) if margin_rows else 0.0, 6),
        "max_incompatibility_excess": round(float(np.max([row["incompatibility_excess"] for row in margin_rows])) if margin_rows else 0.0, 6),
    }
    write_json(ctx.experiment_dir / "identifiability_margin.json", margin_rows)
    write_json(ctx.experiment_dir / "swap_stress_summary.json", summary_payload)

    experiment_spec = {
        "experiment_id": INC01B_EXPERIMENT_ID,
        "description": ctx.experiment.description,
        "source_run_id": ctx.source_run_id,
        "expected_outputs": list(ctx.experiment.expected_outputs),
        "reference_experiment_id": INC01D_EXPERIMENT_ID,
    }
    coverage_summary = {
        "annual_margin_row_count": int(len(margin_rows)),
        "mean_identifiability_margin": float(summary_payload["mean_identifiability_margin"]),
    }
    decision = {
        "completed": True,
        "keep": bool(margin_rows),
        "reason": "INC-01B quantifies how much annual incidence can still swap with backlog-compatible residual structure after the diagnosis-locked branch is made explicit.",
        "checks": [
            {
                "name": "annual_margin_rows_available",
                "passed": bool(margin_rows),
                "actual": int(len(margin_rows)),
                "target": "greater_than_zero",
            }
        ],
    }
    numeric_justification = _base_numeric_policy() + [
        {
            "name": "non_negative_swap_slack_boundary",
            "value": 0,
            "role": "Boundary for allowable swap slack after enforcing non-negative incidence and non-negative residual clearance.",
            "source_type": "physical_constraint",
            "estimation_data": "Slack cannot be negative once the annual validation target and mass-balance lower bound are defined.",
            "estimation_method": "non-negative residual slack constraint",
            "uncertainty": "none",
            "why_needed": "INC-01B reports how much annual incidence remains free to swap with backlog-compatible residual structure.",
        }
    ]
    write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=numeric_justification,
    )
    return {
        "decision": decision,
        "summary": summary_payload,
        "coverage_summary": coverage_summary,
    }


# Numeric Policy re-exports (merged from numeric_policy.py)
from epigraph_ph.phase3.shared.numerics import safe_floor  # noqa: F401
