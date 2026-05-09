from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import _finite_float, _generated_at, _sha256
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r71_service_intensity_capacity_branch import _default_evidence_root
from .r75_bulk_unaids_annual_challenge import (
    _annual_carry_forward_prediction,
    _annual_score_row,
    _annual_target_scale,
    _bulk_unaids_target_rows,
    _merge_external_targets_into_observations,
    _r69_annual_path,
    _rolling_annual_splits,
    _score_summary_by_fields,
    _write_csv,
)
from .r89_incidence_mortality_mechanism_support_gate import _metric_support_summary
from .r91_mechanism_support_expansion_gate import (
    BRIDGE_SPECS,
    _ablate_proxy_metric_source_family,
    _proxy_annual_series,
    _source_family_rows,
)
from .runtime import ensure_dir, read_json, write_json


R92_SCHEMA_VERSION = "phase3_dynamic.r92_process_repair_experiment_queue.v1"
R92_RUN_ID = "p3d-r92-process-repair-experiment-queue-20260509-s00"
R92_FAMILY = "process_repair_train_selected_observation_bridge"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)

PROCESS_REPAIR_FAMILIES: tuple[str, ...] = (
    "proxy_ar_ratio_median",
    "proxy_ar_ratio_ar",
    "proxy_log_linear_target",
    "target_ar_readout_only",
)
MECHANISM_ELIGIBLE_FAMILIES = frozenset(
    {
        "proxy_ar_ratio_median",
        "proxy_ar_ratio_ar",
        "proxy_log_linear_target",
    }
)


def _target_by_year(rows: list[dict[str, Any]], metric_name: str) -> dict[int, float]:
    output: dict[int, float] = {}
    for row in rows:
        quarter = str(row.get("quarter") or "")
        if not quarter.endswith("-Q4"):
            continue
        value = _finite_float(row.get(metric_name))
        if value is None:
            continue
        output[quarter_year(quarter)] = max(float(value), 0.0)
    return dict(sorted(output.items()))


def _series_until(series: dict[int, float], train_end_year: int) -> dict[int, float]:
    return {int(year): float(value) for year, value in sorted(series.items()) if int(year) <= int(train_end_year)}


def _log_ar_forecast(series: dict[int, float], *, train_end_year: int, forecast_years: list[int]) -> dict[int, float]:
    train = _series_until(series, train_end_year)
    if not train:
        return {}
    years = sorted(train)
    values = [float(train[year]) for year in years]
    if len(values) < 2:
        return {int(year): float(values[-1]) for year in forecast_years}
    transformed = np.asarray([np.log1p(max(value, 0.0)) for value in values], dtype=np.float64)
    positions = {year: idx for idx, year in enumerate(years + [year for year in forecast_years if year not in years])}
    x_rows = []
    y_values = []
    for idx in range(1, len(years)):
        year = years[idx]
        x_rows.append([1.0, float(positions[year]), float(transformed[idx - 1])])
        y_values.append(float(transformed[idx]))
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    beta = np.linalg.pinv(x) @ y
    alpha = float(beta[0])
    slope = float(beta[1])
    rho = float(beta[2])
    last_eta = float(transformed[-1])
    last_year = int(years[-1])
    output: dict[int, float] = {}
    for year in sorted(forecast_years):
        eta = last_eta
        for step_year in range(last_year + 1, int(year) + 1):
            eta = alpha + slope * float(positions.get(step_year, positions[last_year] + step_year - last_year)) + rho * eta
        output[int(year)] = max(float(np.expm1(eta)), 0.0)
    return output


def _ratio_series(target: dict[int, float], proxy: dict[int, float], *, train_end_year: int) -> dict[int, float]:
    output: dict[int, float] = {}
    for year, target_value in sorted(target.items()):
        if int(year) > int(train_end_year):
            continue
        proxy_value = _finite_float(proxy.get(int(year)))
        if proxy_value is None or proxy_value <= 0.0:
            continue
        output[int(year)] = float(target_value) / float(proxy_value)
    return output


def _ridge_log_linear_prediction(
    *,
    target: dict[int, float],
    proxy: dict[int, float],
    proxy_forecast: dict[int, float],
    train_end_year: int,
    forecast_years: list[int],
) -> dict[int, float]:
    pairs = [
        (int(year), float(target[year]), float(proxy[year]))
        for year in sorted(target)
        if int(year) <= int(train_end_year) and int(year) in proxy and float(proxy[int(year)]) > 0.0
    ]
    if len(pairs) < 2:
        return {}
    first_year = int(pairs[0][0])
    x = np.asarray(
        [[1.0, np.log1p(max(proxy_value, 0.0)), float(year - first_year)] for year, _target_value, proxy_value in pairs],
        dtype=np.float64,
    )
    y = np.asarray([np.log1p(max(target_value, 0.0)) for _year, target_value, _proxy_value in pairs], dtype=np.float64)
    beta = np.linalg.pinv(x) @ y
    output: dict[int, float] = {}
    for year in forecast_years:
        proxy_value = _finite_float(proxy_forecast.get(int(year)))
        if proxy_value is None:
            continue
        eta = float(np.asarray([1.0, np.log1p(max(float(proxy_value), 0.0)), float(int(year) - first_year)]) @ beta)
        output[int(year)] = max(float(np.expm1(eta)), 0.0)
    return output


def _family_predictions(
    rows: list[dict[str, Any]],
    *,
    target_metric: str,
    proxy_metric: str,
    family: str,
    train_end_year: int,
    forecast_years: list[int],
) -> dict[int, float]:
    target = _target_by_year(rows, target_metric)
    proxy_rows = _proxy_annual_series(rows, proxy_metric)
    proxy = {
        int(year): float(row["annualized_proxy_value"])
        for year, row in proxy_rows.items()
        if _finite_float(row.get("annualized_proxy_value")) is not None
    }
    proxy_forecast = _log_ar_forecast(proxy, train_end_year=train_end_year, forecast_years=forecast_years)
    if family == "target_ar_readout_only":
        return _log_ar_forecast(target, train_end_year=train_end_year, forecast_years=forecast_years)
    ratios = _ratio_series(target, proxy, train_end_year=train_end_year)
    if not ratios:
        return {}
    if family == "proxy_ar_ratio_median":
        ratio = float(np.median(np.asarray(list(ratios.values()), dtype=np.float64)))
        return {
            int(year): float(proxy_forecast[int(year)]) * ratio
            for year in forecast_years
            if int(year) in proxy_forecast
        }
    if family == "proxy_ar_ratio_ar":
        ratio_forecast = _log_ar_forecast(ratios, train_end_year=train_end_year, forecast_years=forecast_years)
        return {
            int(year): float(proxy_forecast[int(year)]) * float(ratio_forecast[int(year)])
            for year in forecast_years
            if int(year) in proxy_forecast and int(year) in ratio_forecast
        }
    if family == "proxy_log_linear_target":
        return _ridge_log_linear_prediction(
            target=target,
            proxy=proxy,
            proxy_forecast=proxy_forecast,
            train_end_year=train_end_year,
            forecast_years=forecast_years,
        )
    raise ValueError(f"Unknown R92 process-repair family: {family}")


def _internal_family_score_rows(
    rows: list[dict[str, Any]],
    *,
    bridge_id: str,
    target_metric: str,
    proxy_metric: str,
    train_end_year: int,
) -> list[dict[str, Any]]:
    q4_rows = [dict(row) for row in rows if str(row.get("quarter") or "").endswith("-Q4")]
    available_years = sorted(
        {
            quarter_year(str(row.get("quarter") or ""))
            for row in q4_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
            and _finite_float(row.get(target_metric)) is not None
        }
    )
    output: list[dict[str, Any]] = []
    for validation_year in available_years:
        prior_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) < int(validation_year)]
        holdout_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) == int(validation_year)]
        if len(prior_rows) < 2 or not holdout_rows:
            continue
        for family in PROCESS_REPAIR_FAMILIES:
            predictions = _family_predictions(
                rows,
                target_metric=target_metric,
                proxy_metric=proxy_metric,
                family=family,
                train_end_year=int(validation_year) - 1,
                forecast_years=[int(validation_year)],
            )
            for holdout_row in holdout_rows:
                candidate_value = _finite_float(predictions.get(int(validation_year)))
                carry_value = _annual_carry_forward_prediction(prior_rows, holdout_row, target_metric)
                score = _annual_score_row(
                    family=f"{bridge_id}:{family}:internal",
                    horizon=1,
                    train_end_year=int(validation_year) - 1,
                    holdout_years=[int(validation_year)],
                    holdout_row=holdout_row,
                    metric_name=target_metric,
                    candidate_value=None if candidate_value is None else float(candidate_value),
                    carry_value=None if carry_value is None else float(carry_value),
                    scale=_annual_target_scale(prior_rows, target_metric),
                )
                score["bridge_id"] = bridge_id
                score["process_family"] = family
                score["training_use"] = "train_internal_process_family_selection"
                score["mechanism_eligible"] = family in MECHANISM_ELIGIBLE_FAMILIES
                output.append(score)
    return output


def _select_process_family(internal_rows: list[dict[str, Any]], *, mechanism_only: bool) -> dict[str, Any]:
    candidates = [
        row
        for row in internal_rows
        if (not mechanism_only or bool(row.get("mechanism_eligible")))
    ]
    summaries = _score_summary_by_fields(candidates, group_fields=("process_family",))
    evaluable = [
        row
        for row in summaries
        if _finite_float(row.get("candidate_mean_norm_error")) is not None
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    if not evaluable:
        default_family = "proxy_ar_ratio_median" if mechanism_only else "target_ar_readout_only"
        return {
            "selected_family": default_family,
            "selection_status": "default_no_internal_scores",
            "internal_summary_rows": summaries,
        }
    order = {family: index for index, family in enumerate(PROCESS_REPAIR_FAMILIES)}
    evaluable.sort(key=lambda row: (float(row.get("candidate_mean_norm_error") or np.inf), order.get(str(row.get("process_family") or ""), 999)))
    selected = dict(evaluable[0])
    return {
        "selected_family": str(selected.get("process_family") or ""),
        "selection_status": "train_internal_min_candidate_mean_norm_error",
        "selected_internal_candidate_mean_norm_error": selected.get("candidate_mean_norm_error"),
        "selected_internal_carry_forward_mean_norm_error": selected.get("carry_forward_mean_norm_error"),
        "internal_summary_rows": summaries,
    }


def _score_process_repair(
    rows: list[dict[str, Any]],
    *,
    bridge_id: str,
    target_metric: str,
    proxy_metric: str,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
    mechanism_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    q4_rows = [dict(row) for row in rows if str(row.get("quarter") or "").endswith("-Q4")]
    score_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    for split in _rolling_annual_splits(q4_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizons=horizons):
        train_end_year = int(split["train_end_year"])
        horizon = int(split["horizon_years"])
        holdout_years = [int(year) for year in split["holdout_years"]]
        raw_train_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) <= train_end_year]
        holdout_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        if not raw_train_rows or not holdout_rows:
            continue
        split_internal_rows = _internal_family_score_rows(
            rows,
            bridge_id=bridge_id,
            target_metric=target_metric,
            proxy_metric=proxy_metric,
            train_end_year=train_end_year,
        )
        selection = _select_process_family(split_internal_rows, mechanism_only=mechanism_only)
        selected_family = str(selection.get("selected_family") or "")
        predictions = _family_predictions(
            rows,
            target_metric=target_metric,
            proxy_metric=proxy_metric,
            family=selected_family,
            train_end_year=train_end_year,
            forecast_years=holdout_years,
        )
        model_rows.append(
            {
                "candidate_family": f"{bridge_id}:r92_process_repair:{'mechanism' if mechanism_only else 'readout'}",
                "bridge_id": bridge_id,
                "target_metric": target_metric,
                "proxy_metric": proxy_metric,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                "selected_process_family": selected_family,
                "mechanism_only": bool(mechanism_only),
                "mechanism_eligible": selected_family in MECHANISM_ELIGIBLE_FAMILIES,
                "selection_status": selection.get("selection_status"),
                "selected_internal_candidate_mean_norm_error": selection.get("selected_internal_candidate_mean_norm_error"),
                "selected_internal_carry_forward_mean_norm_error": selection.get("selected_internal_carry_forward_mean_norm_error"),
            }
        )
        for internal_row in split_internal_rows:
            row = dict(internal_row)
            row["outer_train_end_year"] = train_end_year
            row["outer_horizon_years"] = horizon
            row["mechanism_only_outer"] = bool(mechanism_only)
            internal_rows.append(row)
        for holdout_row in holdout_rows:
            holdout_year = quarter_year(str(holdout_row.get("quarter") or ""))
            if _finite_float(holdout_row.get(target_metric)) is None:
                continue
            candidate_value = _finite_float(predictions.get(int(holdout_year)))
            carry_value = _annual_carry_forward_prediction(raw_train_rows, holdout_row, target_metric)
            score = _annual_score_row(
                family=f"{bridge_id}:r92_process_repair:{'mechanism' if mechanism_only else 'readout'}",
                horizon=horizon,
                train_end_year=train_end_year,
                holdout_years=holdout_years,
                holdout_row=holdout_row,
                metric_name=target_metric,
                candidate_value=None if candidate_value is None else float(candidate_value),
                carry_value=None if carry_value is None else float(carry_value),
                scale=_annual_target_scale(raw_train_rows, target_metric),
            )
            score["bridge_id"] = bridge_id
            score["proxy_metric"] = proxy_metric
            score["selected_process_family"] = selected_family
            score["mechanism_only"] = bool(mechanism_only)
            score["mechanism_eligible"] = selected_family in MECHANISM_ELIGIBLE_FAMILIES
            score["training_use"] = "train_origin_process_repair_selected_inside_train_only"
            score_rows.append(score)
    return score_rows, model_rows, internal_rows


def _score_all_bridges(
    rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
    mechanism_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    for spec in BRIDGE_SPECS:
        rows_part, models_part, internal_part = _score_process_repair(
            rows,
            bridge_id=str(spec["bridge_id"]),
            target_metric=str(spec["target_metric"]),
            proxy_metric=str(spec["proxy_metric"]),
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            horizons=horizons,
            mechanism_only=mechanism_only,
        )
        score_rows.extend(rows_part)
        model_rows.extend(models_part)
        internal_rows.extend(internal_part)
    return score_rows, model_rows, internal_rows


def _ablation_rows(
    rows: list[dict[str, Any]],
    *,
    source_family_rows: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    full_scores, _, _ = _score_all_bridges(
        rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
        mechanism_only=True,
    )
    full_by_bridge = {
        str(row.get("candidate_family") or "").split(":")[0]: dict(row)
        for row in _score_summary_by_fields(full_scores, group_fields=("candidate_family",))
    }
    for spec in BRIDGE_SPECS:
        bridge_id = str(spec["bridge_id"])
        proxy_metric = str(spec["proxy_metric"])
        target_metric = str(spec["target_metric"])
        for family_row in source_family_rows:
            if str(family_row.get("proxy_metric") or "") != proxy_metric:
                continue
            source_family = str(family_row.get("source_family") or "")
            ablated = _ablate_proxy_metric_source_family(rows, proxy_metric=proxy_metric, source_family=source_family)
            ablated_scores, _, _ = _score_process_repair(
                ablated,
                bridge_id=bridge_id,
                target_metric=target_metric,
                proxy_metric=proxy_metric,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                horizons=horizons,
                mechanism_only=True,
            )
            ablated_summary = (_score_summary_by_fields(ablated_scores, group_fields=("candidate_family",)) or [{}])[0]
            full_summary = full_by_bridge.get(bridge_id) or {}
            full_mean = _finite_float(full_summary.get("candidate_mean_norm_error"))
            ablated_mean = _finite_float(ablated_summary.get("candidate_mean_norm_error"))
            output.append(
                {
                    "bridge_id": bridge_id,
                    "target_metric": target_metric,
                    "proxy_metric": proxy_metric,
                    "ablated_source_family": source_family,
                    "ablated_entry_count": family_row.get("entry_count"),
                    "full_candidate_mean_norm_error": full_mean,
                    "full_carry_forward_mean_norm_error": _finite_float(full_summary.get("carry_forward_mean_norm_error")),
                    "ablated_candidate_mean_norm_error": ablated_mean,
                    "ablated_carry_forward_mean_norm_error": _finite_float(ablated_summary.get("carry_forward_mean_norm_error")),
                    "ablation_delta_candidate_mean_norm_error": None
                    if full_mean is None or ablated_mean is None
                    else float(ablated_mean - full_mean),
                    "ablation_status": "evaluable" if ablated_scores else "not_evaluable_after_ablation",
                }
            )
    return output


def _bridge_summary(family_rows: list[dict[str, Any]], bridge_id: str, mode: str) -> dict[str, Any]:
    prefix = f"{bridge_id}:r92_process_repair:{mode}"
    row = next((dict(item) for item in family_rows if str(item.get("candidate_family") or "") == prefix), {})
    candidate_mean = _finite_float(row.get("candidate_mean_norm_error"))
    carry_mean = _finite_float(row.get("carry_forward_mean_norm_error"))
    candidate_coverage = _finite_float(row.get("candidate_interval_coverage"))
    carry_coverage = _finite_float(row.get("carry_forward_interval_coverage"))
    return {
        "bridge_id": bridge_id,
        "mode": mode,
        "candidate_mean_norm_error": candidate_mean,
        "carry_forward_mean_norm_error": carry_mean,
        "candidate_minus_carry_forward_mean_norm_error": None
        if candidate_mean is None or carry_mean is None
        else float(candidate_mean - carry_mean),
        "candidate_interval_coverage": candidate_coverage,
        "carry_forward_interval_coverage": carry_coverage,
        "beats_carry_forward_mean": bool(candidate_mean is not None and carry_mean is not None and candidate_mean < carry_mean),
        "coverage_nonregression": bool(
            candidate_coverage is not None and carry_coverage is not None and candidate_coverage >= carry_coverage
        ),
    }


def _gate(
    *,
    support_rows: list[dict[str, Any]],
    mechanism_family_rows: list[dict[str, Any]],
    readout_family_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    support_by_metric = {str(row.get("metric_name") or ""): dict(row) for row in support_rows}
    direct_incidence_count = int((support_by_metric.get("incident_infections_period") or {}).get("count") or 0)
    diagnosis_flow_proxy_count = int((support_by_metric.get("new_diagnosed_cases_period") or {}).get("count") or 0)
    direct_death_count = int((support_by_metric.get("deaths_reported_period") or {}).get("count") or 0)
    incidence_mechanism = _bridge_summary(mechanism_family_rows, "incidence_proxy_diagnosis_flow_bridge", "mechanism")
    mortality_mechanism = _bridge_summary(mechanism_family_rows, "mortality_reported_death_bridge", "mechanism")
    incidence_readout = _bridge_summary(readout_family_rows, "incidence_proxy_diagnosis_flow_bridge", "readout")
    mortality_readout = _bridge_summary(readout_family_rows, "mortality_reported_death_bridge", "readout")
    mortality_ablation = [row for row in ablation_rows if str(row.get("bridge_id") or "") == "mortality_reported_death_bridge"]
    mortality_stable = bool(mortality_ablation) and all(
        str(row.get("ablation_status") or "") == "evaluable"
        and _finite_float(row.get("ablated_candidate_mean_norm_error")) is not None
        and _finite_float(row.get("ablated_carry_forward_mean_norm_error")) is not None
        and float(row["ablated_candidate_mean_norm_error"]) < float(row["ablated_carry_forward_mean_norm_error"])
        for row in mortality_ablation
    )
    blockers: list[str] = []
    if direct_incidence_count <= 0:
        blockers.append("direct_incidence_process_support_absent")
    if diagnosis_flow_proxy_count <= 0:
        blockers.append("diagnosis_flow_proxy_support_absent")
    if direct_death_count <= 0:
        blockers.append("direct_reported_death_support_absent")
    if not incidence_mechanism["beats_carry_forward_mean"]:
        blockers.append("incidence_process_repair_not_better_than_carry_forward")
    if not incidence_mechanism["coverage_nonregression"]:
        blockers.append("incidence_process_repair_interval_coverage_worse_than_carry_forward")
    if not mortality_mechanism["beats_carry_forward_mean"]:
        blockers.append("mortality_process_repair_not_better_than_carry_forward")
    if not mortality_mechanism["coverage_nonregression"]:
        blockers.append("mortality_process_repair_interval_coverage_worse_than_carry_forward")
    if not mortality_stable:
        blockers.append("mortality_process_repair_not_source_family_stable")
    if not blockers:
        status = "process_repair_mechanism_support_ready"
    elif mortality_mechanism["beats_carry_forward_mean"] and mortality_mechanism["coverage_nonregression"]:
        status = "mortality_process_signal_detected_mechanism_claim_blocked"
    elif incidence_readout["beats_carry_forward_mean"] or mortality_readout["beats_carry_forward_mean"]:
        status = "readout_process_signal_detected_mechanism_claim_blocked"
    else:
        status = "process_repair_diagnostic_only"
    return {
        "status": status,
        "blockers": blockers,
        "direct_incidence_process_support_count": direct_incidence_count,
        "diagnosis_flow_proxy_support_count": diagnosis_flow_proxy_count,
        "direct_reported_death_support_count": direct_death_count,
        "incidence_mechanism_bridge": incidence_mechanism,
        "mortality_mechanism_bridge": mortality_mechanism,
        "incidence_readout_bridge": incidence_readout,
        "mortality_readout_bridge": mortality_readout,
        "mortality_source_family_stable": mortality_stable,
        "contract": (
            "R92 is a train-origin process-repair queue. It tests richer proxy/ascertainment families but separates "
            "mechanism-eligible proxy models from target-only readout models. Mechanism claims require direct incidence "
            "support, carry-forward improvement, coverage non-regression, and mortality source-family stability."
        ),
    }


def _experiment_queue_rows() -> list[dict[str, Any]]:
    return [
        {
            "priority": 1,
            "experiment_id": "R92",
            "name": "mortality_incidence_process_repair",
            "reason": "R89-R91 block raw incidence/death mechanism claims.",
            "status_after_this_run": "implemented",
        },
        {
            "priority": 2,
            "experiment_id": "R93",
            "name": "open_aem_spectrum_public_comparator",
            "reason": "Official Philippines Spectrum/AEM files are absent; construct open comparator from public data.",
            "status_after_this_run": "queued",
        },
        {
            "priority": 3,
            "experiment_id": "R94",
            "name": "direct_incidence_evidence_scanner",
            "reason": "Direct incidence process support count is zero.",
            "status_after_this_run": "queued",
        },
        {
            "priority": 4,
            "experiment_id": "R95",
            "name": "subnational_sparse_hierarchy",
            "reason": "Regional adapters are mean-promoted but not split-stable.",
            "status_after_this_run": "queued",
        },
        {
            "priority": 5,
            "experiment_id": "R96",
            "name": "phase2_source_stable_prior_retest",
            "reason": "Phase 2 determinants are sensitivity-only until source-stable priors survive.",
            "status_after_this_run": "queued",
        },
        {
            "priority": 6,
            "experiment_id": "R97",
            "name": "third95_vl_suppression_process_gate",
            "reason": "Back-half process claims remain evidence-limited.",
            "status_after_this_run": "queued",
        },
    ]


def run_r92_process_repair_experiment_queue(
    *,
    run_id: str = R92_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    r69_report_path: Path | None = None,
    external_start_year: int = 2010,
    start_year: int = 2019,
    end_year: int = 2024,
    min_train_years: int = 5,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    root = Path(epigraph_root) if epigraph_root is not None else _default_evidence_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(root, source_run_id=source_run_id, preferred=baseline_source_run_id)
    observation_rows = build_observation_rows(root, source_run_id, baseline_source_run_id=baseline_source_run_id)
    r69_path = R69_DEFAULT_REPORT if r69_report_path is None else Path(r69_report_path)
    r69 = dict(read_json(r69_path, default={}) or {}) if r69_path.exists() else {}
    annual_csv = _r69_annual_path(r69)
    target_rows = [] if annual_csv is None else _bulk_unaids_target_rows(annual_csv, external_start_year=external_start_year)
    rows = _merge_external_targets_into_observations(observation_rows, target_rows)
    support_metrics = (
        "incident_infections_period",
        "new_diagnosed_cases_period",
        "annual_new_infections",
        "deaths_reported_period",
        "annual_aids_deaths",
    )
    support_rows = [_metric_support_summary(rows, metric) for metric in support_metrics]
    proxy_series_rows = []
    for spec in BRIDGE_SPECS:
        proxy_series_rows.extend(list(_proxy_annual_series(rows, str(spec["proxy_metric"])).values()))
    mechanism_scores, mechanism_models, mechanism_internal = _score_all_bridges(
        rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
        mechanism_only=True,
    )
    readout_scores, readout_models, readout_internal = _score_all_bridges(
        rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
        mechanism_only=False,
    )
    mechanism_family_rows = _score_summary_by_fields(mechanism_scores, group_fields=("candidate_family",))
    readout_family_rows = _score_summary_by_fields(readout_scores, group_fields=("candidate_family",))
    source_family_rows = _source_family_rows(rows, tuple(spec["proxy_metric"] for spec in BRIDGE_SPECS))
    ablation_rows = _ablation_rows(
        rows,
        source_family_rows=source_family_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
    )
    gate = _gate(
        support_rows=support_rows,
        mechanism_family_rows=mechanism_family_rows,
        readout_family_rows=readout_family_rows,
        ablation_rows=ablation_rows,
    )
    report_path = analysis_dir / "r92_process_repair_experiment_queue_report.json"
    markdown_path = analysis_dir / "r92_process_repair_experiment_queue_report.md"
    paths = {
        "support_rows_csv": analysis_dir / "r92_support_rows.csv",
        "proxy_series_rows_csv": analysis_dir / "r92_proxy_series_rows.csv",
        "mechanism_score_rows_csv": analysis_dir / "r92_mechanism_score_rows.csv",
        "readout_score_rows_csv": analysis_dir / "r92_readout_score_rows.csv",
        "mechanism_model_rows_csv": analysis_dir / "r92_mechanism_model_rows.csv",
        "readout_model_rows_csv": analysis_dir / "r92_readout_model_rows.csv",
        "mechanism_internal_rows_csv": analysis_dir / "r92_mechanism_internal_rows.csv",
        "readout_internal_rows_csv": analysis_dir / "r92_readout_internal_rows.csv",
        "mechanism_family_rows_csv": analysis_dir / "r92_mechanism_family_rows.csv",
        "readout_family_rows_csv": analysis_dir / "r92_readout_family_rows.csv",
        "source_family_rows_csv": analysis_dir / "r92_source_family_rows.csv",
        "source_family_ablation_rows_csv": analysis_dir / "r92_source_family_ablation_rows.csv",
        "experiment_queue_rows_csv": analysis_dir / "r92_experiment_queue_rows.csv",
    }
    report = {
        "schema_version": R92_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "candidate_family": R92_FAMILY,
        "process_repair_gate": gate,
        "experiment_queue_rows": _experiment_queue_rows(),
        "support_rows": support_rows,
        "proxy_series_rows": proxy_series_rows,
        "mechanism_score_rows": mechanism_scores,
        "readout_score_rows": readout_scores,
        "mechanism_model_rows": mechanism_models,
        "readout_model_rows": readout_models,
        "mechanism_internal_rows": mechanism_internal,
        "readout_internal_rows": readout_internal,
        "mechanism_family_rows": mechanism_family_rows,
        "readout_family_rows": readout_family_rows,
        "source_family_rows": source_family_rows,
        "source_family_ablation_rows": ablation_rows,
        "source_artifacts": {
            "r69": {"path": r69_path.as_posix(), "sha256": _sha256(r69_path) if r69_path.exists() else None},
            "annual_external_challenge_csv": None if annual_csv is None else annual_csv.as_posix(),
            "annual_external_challenge_csv_sha256": None if annual_csv is None or not annual_csv.exists() else _sha256(annual_csv),
            "epigraph_root": root.as_posix(),
            "source_run_id": source_run_id,
            "baseline_source_run_id": baseline_source_run_id,
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            **{name: path.as_posix() for name, path in paths.items()},
        },
    }
    _write_csv(paths["support_rows_csv"], support_rows)
    _write_csv(paths["proxy_series_rows_csv"], proxy_series_rows)
    _write_csv(paths["mechanism_score_rows_csv"], mechanism_scores)
    _write_csv(paths["readout_score_rows_csv"], readout_scores)
    _write_csv(paths["mechanism_model_rows_csv"], mechanism_models)
    _write_csv(paths["readout_model_rows_csv"], readout_models)
    _write_csv(paths["mechanism_internal_rows_csv"], mechanism_internal)
    _write_csv(paths["readout_internal_rows_csv"], readout_internal)
    _write_csv(paths["mechanism_family_rows_csv"], mechanism_family_rows)
    _write_csv(paths["readout_family_rows_csv"], readout_family_rows)
    _write_csv(paths["source_family_rows_csv"], source_family_rows)
    _write_csv(paths["source_family_ablation_rows_csv"], ablation_rows)
    _write_csv(paths["experiment_queue_rows_csv"], report["experiment_queue_rows"])
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("process_repair_gate") or {})
    lines = [
        "# Phase 3 R92 Process-Repair Experiment Queue",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        f"- Direct incidence support count: `{gate.get('direct_incidence_process_support_count')}`",
        f"- Diagnosis-flow proxy support count: `{gate.get('diagnosis_flow_proxy_support_count')}`",
        f"- Direct reported-death support count: `{gate.get('direct_reported_death_support_count')}`",
        "",
        "## Bridge Summary",
        "",
        "| Bridge | Mode | Candidate Mean | Carry Mean | Candidate Coverage | Carry Coverage | Mean Beats Carry | Coverage Nonregression |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for key in ("incidence_mechanism_bridge", "mortality_mechanism_bridge", "incidence_readout_bridge", "mortality_readout_bridge"):
        row = dict(gate.get(key) or {})
        lines.append(
            f"| `{row.get('bridge_id')}` | `{row.get('mode')}` | `{row.get('candidate_mean_norm_error')}` | "
            f"`{row.get('carry_forward_mean_norm_error')}` | `{row.get('candidate_interval_coverage')}` | "
            f"`{row.get('carry_forward_interval_coverage')}` | `{row.get('beats_carry_forward_mean')}` | "
            f"`{row.get('coverage_nonregression')}` |"
        )
    lines.extend(["", "## Experiment Queue", "", "| Priority | Experiment | Name | Status |", "|---:|---|---|---|"])
    for row in report.get("experiment_queue_rows") or []:
        lines.append(
            f"| `{row.get('priority')}` | `{row.get('experiment_id')}` | `{row.get('name')}` | `{row.get('status_after_this_run')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R92 process-repair experiment queue.")
    parser.add_argument("--run-id", default=R92_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    args = parser.parse_args()
    run_r92_process_repair_experiment_queue(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
    )


if __name__ == "__main__":
    _main()
