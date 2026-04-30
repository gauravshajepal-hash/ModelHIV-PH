from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase3._lineage.national_reset_core import (
    PRIMARY_METRICS,
    fit_national_uda_baseline,
    fit_national_uda_delay_aux,
    fit_national_uda_vl_observation_process,
    quarter_end_month,
    quarter_sort_key,
)
from epigraph_ph.runtime import ROOT_DIR, RunContext, ensure_dir, read_json, write_json


_HIV_PLUGIN = get_disease_plugin("hiv")
NR00_EXPERIMENT_ID = "NR-00-observation-table"
NR01_EXPERIMENT_ID = "NR-01-national-uda-baseline"
NR02_EXPERIMENT_ID = "NR-02-delay-aux"
NR03_EXPERIMENT_ID = "NR-03-vl-observation-process"
NR05_EXPERIMENT_ID = "NR-05-deferred-complexity-scan"
_PRIMARY_PRIORITY = {
    "quarterly_snapshot": 30,
    "monthly_aggregated_to_quarter": 20,
    "multi_month_snapshot": 15,
    "monthly_snapshot": 10,
    "annual_snapshot": 5,
}
_OBSERVATION_METRICS = (
    "diagnosed_plhiv",
    "alive_on_art",
    "new_diagnosed_cases_period",
    "advanced_hiv_cases_period",
    "median_cd4_at_enrollment",
    "tested_for_viral_load",
    "virally_suppressed",
)
_AUXILIARY_METRICS = ("advanced_hiv_cases_period", "median_cd4_at_enrollment")
_VL_METRICS = ("tested_for_viral_load", "virally_suppressed")


def _national_reset_cfg() -> dict[str, Any]:
    return dict((((_HIV_PLUGIN.constraint_settings or {}).get("phase3", {}) or {}).get("national_reset", {}) or {}))


def _national_reset_required_section(key: str) -> dict[str, Any]:
    value = _national_reset_cfg().get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing HIV phase3 national_reset section: {key}")
    return dict(value)


def _national_reset_defaults() -> dict[str, Any]:
    return _national_reset_required_section("defaults")


def _national_reset_benchmark_gates() -> dict[str, Any]:
    return _national_reset_required_section("benchmark_gates")


def _national_reset_deferred_scan_cfg() -> dict[str, Any]:
    return _national_reset_required_section("deferred_scan")


def _national_reset_start_quarter(start_quarter: str | None) -> str:
    if start_quarter:
        return str(start_quarter)
    configured = _national_reset_defaults().get("start_quarter")
    if not configured:
        raise KeyError("Missing HIV phase3 national_reset.defaults.start_quarter")
    return str(configured)


def _national_reset_holdout_years(holdout_years: list[int] | None) -> list[int]:
    if holdout_years:
        return sorted({int(year) for year in holdout_years})
    defaults = list(_national_reset_defaults().get("holdout_years", []) or [])
    if not defaults:
        raise KeyError("Missing HIV phase3 national_reset.defaults.holdout_years")
    return sorted({int(year) for year in defaults})


def _plausible_cd4_range() -> tuple[float, float]:
    observation_table_cfg = _national_reset_required_section("observation_table")
    lower, upper = list(observation_table_cfg.get("plausible_cd4_range") or [])
    return float(lower), float(upper)


def _required_float(metric_map: dict[str, Any], key: str, *, context: str) -> float:
    if key not in metric_map or metric_map.get(key) is None:
        raise KeyError(f"Missing {context}: {key}")
    return float(metric_map[key])


def _load_legacy_benchmark_reference() -> dict[str, float]:
    reference_cfg = dict((((_HIV_PLUGIN.reference_data or {}).get("phase3", {}) or {}).get("national_reset_benchmarks", {}) or {}))
    legacy_run_id = str(reference_cfg.get("legacy_run_id") or "")
    if not legacy_run_id:
        raise KeyError("Missing HIV phase3 national_reset_benchmarks.legacy_run_id")
    validation_path = ROOT_DIR / "artifacts" / "runs" / legacy_run_id / str(reference_cfg.get("legacy_validation_artifact") or "")
    tournament_path = ROOT_DIR / "artifacts" / "runs" / legacy_run_id / str(reference_cfg.get("legacy_tournament_artifact") or "")
    validation = read_json(validation_path, default={})
    tournament = read_json(tournament_path, default={})
    diagnosis_flow_summary = dict(validation.get("diagnosis_flow_summary") or {})
    frozen_backtest = dict(validation.get("frozen_history_backtest") or {})
    carry_forward = dict(frozen_backtest.get("carry_forward_baseline") or {})
    simple_compartmental = dict(frozen_backtest.get("simple_compartmental_baseline") or {})
    required_metric_keys = [str(item) for item in list(reference_cfg.get("required_metric_keys") or []) if str(item)]
    if not required_metric_keys:
        raise KeyError("Missing HIV phase3 national_reset_benchmarks.required_metric_keys")
    raw_metrics = {
        "legacy_carry_forward_mae": carry_forward.get("mean_absolute_error"),
        "legacy_simple_compartmental_mae": simple_compartmental.get("mean_absolute_error"),
        "legacy_baseline_model_mae": tournament.get("winner_model_mean_absolute_error"),
        "legacy_diagnosis_flow_mae": diagnosis_flow_summary.get("mean_absolute_error")
        if diagnosis_flow_summary.get("mean_absolute_error") is not None
        else (
            float(
                np.mean(
                    [
                        float(row.get("absolute_error") or 0.0)
                        for row in list(diagnosis_flow_summary.get("rows") or [])
                        if row.get("absolute_error") is not None
                    ]
                )
            )
            if list(diagnosis_flow_summary.get("rows") or [])
            else None
        ),
    }
    return {
        key: _required_float(raw_metrics, key, context="legacy national-reset benchmark metric")
        for key in required_metric_keys
    }


def _quarter_from_month(month_label: str) -> str:
    year_text, month_text = str(month_label).split("-", 1)
    quarter_number = ((int(month_text[:2]) - 1) // 3) + 1
    return f"{int(year_text):04d}-Q{quarter_number}"


def _quarter_start_month(quarter: str) -> str:
    year, quarter_number = quarter_sort_key(quarter)
    month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter_number]
    return f"{year:04d}-{month:02d}"


def _iter_quarters(start_quarter: str, end_quarter: str) -> list[str]:
    start_year, start_index = quarter_sort_key(start_quarter)
    end_year, end_index = quarter_sort_key(end_quarter)
    rows: list[str] = []
    year = start_year
    quarter_number = start_index
    while (year, quarter_number) <= (end_year, end_index):
        rows.append(f"{year:04d}-Q{quarter_number}")
        quarter_number += 1
        if quarter_number > 4:
            year += 1
            quarter_number = 1
    return rows


def _discover_archive_dir(preferred_run_id: str | None = None) -> tuple[str, Path]:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    if preferred_run_id:
        archive_dir = runs_root / preferred_run_id / "harp_archive"
        if not (archive_dir / "historical_metric_rows.json").exists():
            raise FileNotFoundError(f"Missing harp archive for run_id={preferred_run_id}")
        return preferred_run_id, archive_dir
    candidates: list[tuple[float, str, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        archive_dir = run_dir / "harp_archive"
        metric_path = archive_dir / "historical_metric_rows.json"
        if metric_path.exists():
            candidates.append((metric_path.stat().st_mtime, run_dir.name, archive_dir))
    if not candidates:
        raise FileNotFoundError("No harp archive run found under artifacts/runs")
    _, run_id, archive_dir = max(candidates, key=lambda row: row[0])
    return run_id, archive_dir


def _phase_run_dir(run_id: str, phase_name: str) -> Path:
    phase_dir = ROOT_DIR / "artifacts" / "runs" / run_id / phase_name
    if not phase_dir.exists():
        raise FileNotFoundError(f"Missing {phase_name} artifact directory for run_id={run_id}")
    return phase_dir


def _dx_queue_items() -> list[dict[str, Any]]:
    return [
        {
            "deferred_id": "DX-01",
            "item": "determinant modifiers as active forecast covariates",
            "earliest_eligible_phase": "after NR-04",
            "metric_target": "improve holdout MAE without scaffold fallback",
            "revert_condition": "selected_determinant_count remains 0 or scaffold_only reappears",
        },
        {
            "deferred_id": "DX-02",
            "item": "province archetypes",
            "earliest_eligible_phase": "after NR-04",
            "metric_target": "improve subgroup attribution or holdout MAE",
            "revert_condition": "variance increases without holdout gain",
        },
        {
            "deferred_id": "DX-03",
            "item": "region/province hierarchy",
            "earliest_eligible_phase": "after NR-04",
            "metric_target": "improve subnational validation after national stabilization",
            "revert_condition": "national fit degrades or hierarchy is weakly identified",
        },
        {
            "deferred_id": "DX-04",
            "item": "metapopulation and network-family pressure terms",
            "earliest_eligible_phase": "after NR-04",
            "metric_target": "standalone out-of-sample improvement",
            "revert_condition": "benefit appears only in-sample",
        },
        {
            "deferred_id": "DX-05",
            "item": "representation tournament variants",
            "earliest_eligible_phase": "after NR-04",
            "metric_target": "variants produce materially different scores",
            "revert_condition": "variant scores collapse again",
        },
        {
            "deferred_id": "DX-06",
            "item": "full back-half detail",
            "earliest_eligible_phase": "after NR-03",
            "metric_target": "back-half improvement without front-half regression",
            "revert_condition": "diagnosis/linkage errors worsen",
        },
        {
            "deferred_id": "DX-07",
            "item": "subgroup microstructure beyond essential splits",
            "earliest_eligible_phase": "after NR-04",
            "metric_target": "improve holdout fit with explicit observational support",
            "revert_condition": "microstructure is unobserved or prior-dominated",
        },
    ]


def _month_window_start_index(month_axis: list[str], start_month: str) -> int:
    for idx, month_label in enumerate(month_axis):
        if str(month_label) >= start_month:
            return idx
    raise ValueError(f"Active deferred-scan month {start_month} is outside the phase15 month axis")


def _observation_adjacent_tokens(member_names: list[str]) -> list[str]:
    configured_tokens = [str(item).lower() for item in list(_national_reset_deferred_scan_cfg().get("observation_adjacent_tokens") or []) if str(item)]
    matched: set[str] = set()
    normalized_members = [str(name).lower() for name in member_names]
    for token in configured_tokens:
        if any(token in member_name for member_name in normalized_members):
            matched.add(token)
    return sorted(matched)


def _deferred_factor_screen(phase_run_id: str) -> dict[str, Any]:
    scan_cfg = _national_reset_deferred_scan_cfg()
    phase2_dir = _phase_run_dir(phase_run_id, "phase2")
    phase15_dir = _phase_run_dir(phase_run_id, "phase15")
    blankets = dict(read_json(phase2_dir / "multiscale_phase3_target_blankets.json", default={}) or {})
    factor_support_rows = [dict(row) for row in list(blankets.get("factor_support_rows") or [])]
    blanket_factor_ids = [str(item) for item in list(blankets.get("merged_blanket_factor_ids") or []) if str(item)]
    target_factor_ids = {str(item) for item in list(blankets.get("merged_target_factor_ids") or []) if str(item)}
    support_by_factor = {
        str(row.get("factor_id") or ""): int(row.get("support_count") or 0)
        for row in factor_support_rows
        if str(row.get("factor_id") or "")
    }
    factor_catalog = [dict(row) for row in list(read_json(phase15_dir / "multiscale_factor_catalog.json", default=[])) if isinstance(row, dict)]
    catalog_by_factor = {str(row.get("factor_id") or ""): row for row in factor_catalog}
    factor_axes = dict(read_json(phase15_dir / "multiscale_factor_axes.json", default={}) or {})
    factor_axis = [str(item) for item in list(factor_axes.get("factor") or []) if str(item)]
    month_axis = [str(item) for item in list(factor_axes.get("month") or []) if str(item)]
    active_window_start = str(scan_cfg.get("active_window_start_month") or "")
    active_start_idx = _month_window_start_index(month_axis, active_window_start)
    factor_tensor = np.asarray(np.load(phase15_dir / "multiscale_national_factor_tensor.npz")["values"], dtype=np.float32)
    national_surface = factor_tensor[0] if factor_tensor.ndim == 3 else factor_tensor
    minimum_support_count = int(scan_cfg.get("minimum_support_count") or 0)
    minimum_dynamic_std = float(scan_cfg.get("minimum_dynamic_std") or 0.0)
    minimum_dynamic_range = float(scan_cfg.get("minimum_dynamic_range") or 0.0)
    minimum_unique_points = int(scan_cfg.get("minimum_unique_points") or 1)
    dominant_step_share_max = float(scan_cfg.get("dominant_step_share_max") or 1.0)

    factor_rows: list[dict[str, Any]] = []
    for factor_id in blanket_factor_ids:
        catalog_row = dict(catalog_by_factor.get(factor_id) or {})
        if factor_id not in factor_axis:
            continue
        factor_idx = factor_axis.index(factor_id)
        series = np.asarray(national_surface[active_start_idx:, factor_idx], dtype=np.float32)
        delta = np.abs(np.diff(series))
        delta_l1 = float(delta.sum()) if delta.size else 0.0
        dominant_step_share = float(delta.max() / delta_l1) if delta_l1 > 0.0 else 0.0
        member_names = [str(name) for name in list(catalog_row.get("member_canonical_names") or []) if str(name)]
        observation_tokens = _observation_adjacent_tokens(member_names)
        unique_point_count = len({round(float(value), 6) for value in series.tolist()})
        is_dynamic = (
            float(series.std()) >= minimum_dynamic_std
            and float(series.max() - series.min()) >= minimum_dynamic_range
            and unique_point_count >= minimum_unique_points
        )
        one_off_spike_risk = dominant_step_share > dominant_step_share_max
        support_count = int(support_by_factor.get(factor_id, 0))
        status = "candidate_for_standalone_dx01"
        if support_count < minimum_support_count:
            status = "reject_low_support"
        elif factor_id.startswith("network_factor_") and not member_names:
            status = "defer_to_dx04_network_operator"
        elif observation_tokens:
            status = "reject_observation_adjacent"
        elif not is_dynamic:
            status = "reject_flat_in_active_window"
        elif one_off_spike_risk:
            status = "reject_one_off_spike"
        factor_rows.append(
            {
                "factor_id": factor_id,
                "support_count": support_count,
                "in_phase2_target_blanket": factor_id in target_factor_ids,
                "active_window_start_month": active_window_start,
                "active_window_months": month_axis[active_start_idx:],
                "active_window_std": round(float(series.std()), 6),
                "active_window_range": round(float(series.max() - series.min()), 6),
                "active_window_unique_point_count": unique_point_count,
                "dominant_step_share": round(dominant_step_share, 6),
                "dynamic_in_active_window": is_dynamic,
                "one_off_spike_risk": one_off_spike_risk,
                "observation_adjacent_tokens": observation_tokens,
                "member_canonical_names": member_names,
                "transition_hooks": [str(item) for item in list(catalog_row.get("transition_hooks") or []) if str(item)],
                "series_preview": [round(float(value), 6) for value in series.tolist()],
                "screening_status": status,
            }
        )

    factor_rows.sort(
        key=lambda row: (
            row["screening_status"] != "candidate_for_standalone_dx01",
            -int(row["support_count"]),
            -float(row["active_window_std"]),
            row["factor_id"],
        )
    )
    promotable_factor_ids = [str(row["factor_id"]) for row in factor_rows if str(row["screening_status"]) == "candidate_for_standalone_dx01"]
    blocked_factor_ids = [str(row["factor_id"]) for row in factor_rows if str(row["screening_status"]) != "candidate_for_standalone_dx01"]
    summary = {
        "phase_run_id": phase_run_id,
        "phase2_dir": str(phase2_dir),
        "phase15_dir": str(phase15_dir),
        "blanket_factor_count": len(blanket_factor_ids),
        "target_factor_count": len(target_factor_ids),
        "promotable_factor_count": len(promotable_factor_ids),
        "promotable_factor_ids": promotable_factor_ids,
        "blocked_factor_ids": blocked_factor_ids,
        "active_window_start_month": active_window_start,
        "active_window_month_count": len(month_axis[active_start_idx:]),
        "scan_thresholds": {
            "minimum_support_count": minimum_support_count,
            "minimum_dynamic_std": minimum_dynamic_std,
            "minimum_dynamic_range": minimum_dynamic_range,
            "minimum_unique_points": minimum_unique_points,
            "dominant_step_share_max": dominant_step_share_max,
        },
    }
    return {"summary": summary, "factor_rows": factor_rows}

def _select_best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            int(_PRIMARY_PRIORITY.get(str(row.get("series_kind") or ""), 0)),
            float(row.get("evidence_confidence") or 0.0),
            str(row.get("source_id") or ""),
        ),
    )


def _aggregate_monthly_to_quarter(rows: list[dict[str, Any]], *, aggregation: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        month_label = str(row.get("time") or row.get("period_end") or "")
        if not month_label:
            continue
        grouped[_quarter_from_month(month_label)].append(dict(row))
    payload: list[dict[str, Any]] = []
    for quarter, quarter_rows in grouped.items():
        values = [float(row.get("value") or 0.0) for row in quarter_rows]
        if aggregation == "sum":
            quarter_value = float(sum(values))
        elif aggregation == "mean":
            quarter_value = float(sum(values) / len(values))
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        payload.append(
            {
                "quarter": quarter,
                "period_start": _quarter_start_month(quarter),
                "period_end": quarter_end_month(quarter),
                "series_kind": "monthly_aggregated_to_quarter",
                "value": quarter_value,
                "source_ids": sorted({str(row.get("source_id") or "") for row in quarter_rows if row.get("source_id")}),
                "source_labels": sorted({str(row.get("source_label") or "") for row in quarter_rows if row.get("source_label")}),
                "months_covered": len(quarter_rows),
                "aggregation_method": f"quarter_{aggregation}",
            }
        )
    payload.sort(key=lambda row: quarter_sort_key(str(row["quarter"])))
    return payload


def _sanitize_metric_value(metric_name: str, value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    numeric = float(value)
    if metric_name == "median_cd4_at_enrollment":
        lower, upper = _plausible_cd4_range()
        if numeric < lower or numeric > upper:
            return None, f"outside_plausible_cd4_range_{int(lower)}_{int(upper)}"
    return numeric, None


def _primary_metric_rows(metric_rows: list[dict[str, Any]], *, metric_name: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        if str(row.get("metric_name") or "") != metric_name:
            continue
        if str(row.get("region") or "").lower() != "national":
            continue
        month_label = str(row.get("period_end") or row.get("time") or "")
        if not month_label:
            continue
        quarter = _quarter_from_month(month_label)
        candidate = dict(row)
        candidate["quarter"] = quarter
        current = selected.get(quarter)
        if current is None or _select_best_row([current, candidate]) == candidate:
            selected[quarter] = candidate
    return selected


def _estimated_plhiv_by_quarter(metric_rows: list[dict[str, Any]]) -> dict[str, float]:
    selected_rows = _primary_metric_rows(metric_rows, metric_name="estimated_plhiv")
    return {
        quarter: float(row.get("value") or 0.0)
        for quarter, row in selected_rows.items()
        if row.get("value") is not None
    }


def _estimated_plhiv_provenance_by_quarter(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected_rows = _primary_metric_rows(metric_rows, metric_name="estimated_plhiv")
    return {
        quarter: {
            "series_kind": row.get("series_kind"),
            "measurement_class": row.get("measurement_class"),
            "source_quality_tier": row.get("source_quality_tier"),
            "source_bank": row.get("source_bank"),
            "source_id": row.get("source_id"),
            "source_label": row.get("source_label"),
            "source_ids": [str(row.get("source_id"))] if row.get("source_id") else [],
            "source_labels": [str(row.get("source_label"))] if row.get("source_label") else [],
            "extraction_method": row.get("extraction_method"),
            "value_semantics": row.get("value_semantics"),
            "is_direct_measurement": row.get("is_direct_measurement"),
        }
        for quarter, row in selected_rows.items()
    }


def _diagnosis_flow_targets_by_quarter(archive_dir: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(archive_dir / "diagnosis_flow_points.json", default={})
    points = list(payload.get("points") or []) if isinstance(payload, dict) else list(payload or [])
    targets: dict[str, dict[str, Any]] = {}
    for point in points:
        quarter = _quarter_from_month(str(point.get("period_end") or point.get("effective_month") or point.get("month") or ""))
        entry = dict(point)
        entry["quarter"] = quarter
        targets[quarter] = entry
    return targets


def build_national_reset_observation_table_payload(
    *,
    archive_dir: Path,
    archive_run_id: str,
    start_quarter: str | None = None,
) -> dict[str, Any]:
    start_quarter = _national_reset_start_quarter(start_quarter)
    raw_rows = read_json(archive_dir / "historical_metric_rows.json", default=[])
    if not isinstance(raw_rows, list):
        raw_rows = []
    metric_rows = [dict(row) for row in raw_rows]
    diagnosis_flow_targets = _diagnosis_flow_targets_by_quarter(archive_dir)

    primary_rows: dict[str, dict[str, Any]] = {}
    for metric_name in ("diagnosed_plhiv", "alive_on_art", "advanced_hiv_cases_period", "tested_for_viral_load", "virally_suppressed"):
        primary_rows[metric_name] = _primary_metric_rows(metric_rows, metric_name=metric_name)

    quarterly_flows = _primary_metric_rows(metric_rows, metric_name="new_diagnosed_cases_period")
    monthly_flow_rows = [
        dict(row)
        for row in metric_rows
        if str(row.get("metric_name") or "") == "new_diagnosed_cases_period"
        and str(row.get("region") or "").lower() == "national"
        and str(row.get("series_kind") or "") == "monthly_snapshot"
    ]
    for row in _aggregate_monthly_to_quarter(monthly_flow_rows, aggregation="sum"):
        quarter = str(row["quarter"])
        current = quarterly_flows.get(quarter)
        if current is None or _PRIMARY_PRIORITY.get(str(row["series_kind"]), 0) >= _PRIMARY_PRIORITY.get(str(current.get("series_kind") or ""), 0):
            quarterly_flows[quarter] = dict(row)
    primary_rows["new_diagnosed_cases_period"] = quarterly_flows

    monthly_cd4_rows = [
        dict(row)
        for row in metric_rows
        if str(row.get("metric_name") or "") == "median_cd4_at_enrollment"
        and str(row.get("region") or "").lower() == "national"
        and str(row.get("series_kind") or "") in {"monthly_snapshot", "multi_month_snapshot", "quarterly_snapshot"}
    ]
    quarterly_cd4: dict[str, dict[str, Any]] = {}
    for row in monthly_cd4_rows:
        if str(row.get("series_kind") or "") == "quarterly_snapshot":
            quarter = _quarter_from_month(str(row.get("period_end") or row.get("time") or ""))
            quarterly_cd4[quarter] = dict(row) | {"quarter": quarter}
    for row in _aggregate_monthly_to_quarter([row for row in monthly_cd4_rows if str(row.get("series_kind") or "") != "quarterly_snapshot"], aggregation="mean"):
        quarter = str(row["quarter"])
        current = quarterly_cd4.get(quarter)
        if current is None or _PRIMARY_PRIORITY.get(str(row["series_kind"]), 0) >= _PRIMARY_PRIORITY.get(str(current.get("series_kind") or ""), 0):
            quarterly_cd4[quarter] = dict(row)
    primary_rows["median_cd4_at_enrollment"] = quarterly_cd4

    latest_quarter_candidates = [
        quarter
        for metric_map in primary_rows.values()
        for quarter in metric_map.keys()
        if quarter >= start_quarter
    ]
    if not latest_quarter_candidates:
        raise ValueError("No national reset quarters available from archive")
    latest_quarter = max(latest_quarter_candidates, key=quarter_sort_key)

    rows: list[dict[str, Any]] = []
    for quarter in _iter_quarters(start_quarter, latest_quarter):
        record: dict[str, Any] = {
            "quarter": quarter,
            "period_start": _quarter_start_month(quarter),
            "period_end": quarter_end_month(quarter),
            "source_ids": [],
            "source_labels": [],
            "metric_provenance": {},
        }
        for metric_name in _OBSERVATION_METRICS:
            source_row = dict(primary_rows.get(metric_name, {}).get(quarter) or {})
            value = source_row.get("value")
            sanitized_value, quality_filter = _sanitize_metric_value(metric_name, value)
            record[metric_name] = sanitized_value
            record[f"{metric_name}_observed"] = sanitized_value is not None
            provenance = {
                "series_kind": source_row.get("series_kind"),
                "source_ids": sorted({str(item) for item in list(source_row.get("source_ids") or [source_row.get("source_id")]) if item}),
                "source_labels": sorted({str(item) for item in list(source_row.get("source_labels") or [source_row.get("source_label")]) if item}),
                "aggregation_method": source_row.get("aggregation_method"),
                "months_covered": source_row.get("months_covered"),
                "quality_filter": quality_filter,
                "measurement_class": source_row.get("measurement_class"),
                "source_quality_tier": source_row.get("source_quality_tier"),
                "source_bank": source_row.get("source_bank"),
                "extraction_method": source_row.get("extraction_method"),
                "value_semantics": source_row.get("value_semantics"),
                "is_direct_measurement": source_row.get("is_direct_measurement"),
            }
            record["metric_provenance"][metric_name] = provenance
            record["source_ids"].extend(provenance["source_ids"])
            record["source_labels"].extend(provenance["source_labels"])
        record["source_ids"] = sorted({item for item in record["source_ids"] if item})
        record["source_labels"] = sorted({item for item in record["source_labels"] if item})
        if quarter in diagnosis_flow_targets:
            target = diagnosis_flow_targets[quarter]
            record["diagnosis_flow_reference_share"] = float(target.get("diagnosed_share") or 0.0)
            record["diagnosis_flow_reference_estimated_plhiv"] = float(target.get("estimated_plhiv") or 0.0)
        else:
            record["diagnosis_flow_reference_share"] = None
            record["diagnosis_flow_reference_estimated_plhiv"] = None
        rows.append(record)

    coverage_summary = {
        metric_name: sum(1 for row in rows if row.get(metric_name) is not None)
        for metric_name in _OBSERVATION_METRICS
    }
    missingness_report = {
        metric_name: [row["quarter"] for row in rows if row.get(metric_name) is None]
        for metric_name in _AUXILIARY_METRICS + _VL_METRICS
    }
    quality_filter_counts: dict[str, int] = {}
    for metric_name in _OBSERVATION_METRICS:
        quality_filter_counts[metric_name] = sum(
            1
            for row in rows
            if str(row.get("metric_provenance", {}).get(metric_name, {}).get("quality_filter") or "")
        )
    summary = {
        "archive_run_id": archive_run_id,
        "archive_dir": str(archive_dir),
        "start_quarter": start_quarter,
        "latest_quarter": latest_quarter,
        "row_count": len(rows),
        "coverage_summary": coverage_summary,
        "missingness_report": missingness_report,
        "quality_filter_counts": quality_filter_counts,
    }
    return {
        "summary": summary,
        "rows": rows,
        "estimated_plhiv_by_quarter": _estimated_plhiv_by_quarter(metric_rows),
        "estimated_plhiv_provenance_by_quarter": _estimated_plhiv_provenance_by_quarter(metric_rows),
        "diagnosis_flow_targets_by_quarter": diagnosis_flow_targets,
    }


def run_phase3_national_reset_observation_table(
    *,
    run_id: str,
    plugin_id: str,
    archive_run_id: str | None = None,
    start_quarter: str | None = None,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    start_quarter = _national_reset_start_quarter(start_quarter)
    resolved_archive_run_id, archive_dir = _discover_archive_dir(archive_run_id)
    experiment_dir = ensure_dir(ctx.run_dir / "phase3_national_reset" / NR00_EXPERIMENT_ID)

    payload = build_national_reset_observation_table_payload(
        archive_dir=archive_dir,
        archive_run_id=resolved_archive_run_id,
        start_quarter=start_quarter,
    )
    observation_table = {"summary": payload["summary"], "rows": payload["rows"]}
    experiment_spec = {
        "experiment_id": NR00_EXPERIMENT_ID,
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": resolved_archive_run_id,
        "start_quarter": start_quarter,
        "required_metrics": list(PRIMARY_METRICS) + ["advanced_hiv_cases_period", "median_cd4_at_enrollment"],
        "decision_rule": "keep_only_if_observation_table_is_reproducible_from_archive_data_alone",
    }
    decision_checks = [
        {
            "name": "unique_quarter_rows",
            "passed": len({row["quarter"] for row in payload["rows"]}) == len(payload["rows"]),
            "actual": len(payload["rows"]),
            "target": len(payload["rows"]),
        },
        {
            "name": "diagnosed_quarter_coverage_at_least_12",
            "passed": int(payload["summary"]["coverage_summary"]["diagnosed_plhiv"]) >= 12,
            "actual": int(payload["summary"]["coverage_summary"]["diagnosed_plhiv"]),
            "target": 12,
        },
        {
            "name": "art_quarter_coverage_at_least_12",
            "passed": int(payload["summary"]["coverage_summary"]["alive_on_art"]) >= 12,
            "actual": int(payload["summary"]["coverage_summary"]["alive_on_art"]),
            "target": 12,
        },
        {
            "name": "diagnosis_flow_coverage_at_least_8",
            "passed": int(payload["summary"]["coverage_summary"]["new_diagnosed_cases_period"]) >= 8,
            "actual": int(payload["summary"]["coverage_summary"]["new_diagnosed_cases_period"]),
            "target": 8,
        },
        {
            "name": "missingness_report_present_for_auxiliary_metrics",
            "passed": "advanced_hiv_cases_period" in payload["summary"]["missingness_report"]
            and "median_cd4_at_enrollment" in payload["summary"]["missingness_report"],
            "actual": sorted(payload["summary"]["missingness_report"].keys()),
            "target": ["advanced_hiv_cases_period", "median_cd4_at_enrollment"],
        },
        {
            "name": "provenance_preserved",
            "passed": all(row.get("source_ids") is not None and row.get("source_labels") is not None for row in payload["rows"]),
            "actual": len(payload["rows"]),
            "target": len(payload["rows"]),
        },
    ]
    decision = {
        "experiment_id": NR00_EXPERIMENT_ID,
        "keep": all(bool(row["passed"]) for row in decision_checks),
        "decision_rule": "keep_only_if_table_is_reproducible_with_explicit_missingness_and_provenance",
        "checks": decision_checks,
        "summary": payload["summary"],
    }

    artifact_paths = {
        "observation_table": str(experiment_dir / "observation_table.json"),
        "experiment_spec": str(experiment_dir / "experiment_spec.json"),
        "decision": str(experiment_dir / "decision.json"),
    }
    write_json(experiment_dir / "observation_table.json", observation_table)
    write_json(experiment_dir / "experiment_spec.json", experiment_spec)
    write_json(experiment_dir / "decision.json", decision)
    write_json(experiment_dir / "phase3_national_reset_manifest.json", {"experiment_id": NR00_EXPERIMENT_ID, "artifact_paths": artifact_paths})
    ctx.record_stage_outputs("phase3_national_reset", [Path(path) for path in artifact_paths.values()])
    return {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": resolved_archive_run_id,
        "experiment_id": NR00_EXPERIMENT_ID,
        "phase_dir": str(experiment_dir),
        "artifact_paths": artifact_paths,
        "observation_table": observation_table,
        "estimated_plhiv_by_quarter": payload["estimated_plhiv_by_quarter"],
        "diagnosis_flow_targets_by_quarter": payload["diagnosis_flow_targets_by_quarter"],
        "decision": decision,
    }


def run_phase3_national_reset_baseline(
    *,
    run_id: str,
    plugin_id: str,
    archive_run_id: str | None = None,
    start_quarter: str | None = None,
    holdout_years: list[int] | None = None,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    start_quarter = _national_reset_start_quarter(start_quarter)
    nr00 = run_phase3_national_reset_observation_table(
        run_id=run_id,
        plugin_id=plugin_id,
        archive_run_id=archive_run_id,
        start_quarter=start_quarter,
    )
    experiment_dir = ensure_dir(ctx.run_dir / "phase3_national_reset" / NR01_EXPERIMENT_ID)
    holdout_years = _national_reset_holdout_years(holdout_years)
    observation_rows = list(nr00["observation_table"]["rows"])
    legacy_reference = _load_legacy_benchmark_reference()
    fit_result = fit_national_uda_baseline(
        observation_rows=observation_rows,
        estimated_plhiv_by_quarter=dict(nr00["estimated_plhiv_by_quarter"]),
        diagnosis_flow_targets=dict(nr00["diagnosis_flow_targets_by_quarter"]),
        artifact_dir=experiment_dir,
        holdout_years=holdout_years,
        legacy_reference_metrics=legacy_reference,
    )
    experiment_spec = {
        "experiment_id": NR01_EXPERIMENT_ID,
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": nr00["archive_run_id"],
        "start_quarter": start_quarter,
        "holdout_years": holdout_years,
        "state_names": ["U", "D", "A"],
        "primary_observations": list(PRIMARY_METRICS),
        "auxiliary_observations": ["advanced_hiv_cases_period", "median_cd4_at_enrollment"],
        "decision_rule": "keep_only_if_model_beats_baselines_and_diagnosis_flow_error_improves",
    }
    write_json(experiment_dir / "experiment_spec.json", experiment_spec)
    write_json(experiment_dir / "observation_table.json", nr00["observation_table"])
    artifact_paths = {
        "experiment_spec": str(experiment_dir / "experiment_spec.json"),
        "observation_table": str(experiment_dir / "observation_table.json"),
        "fit_artifact": str(experiment_dir / "fit_artifact.json"),
        "evaluation": str(experiment_dir / "evaluation.json"),
        "baseline_comparison": str(experiment_dir / "baseline_comparison.json"),
        "diagnosis_flow_evaluation": str(experiment_dir / "diagnosis_flow_evaluation.json"),
        "decision": str(experiment_dir / "decision.json"),
        "state_estimates": str(experiment_dir / "state_estimates.npz"),
        "forecast_states": str(experiment_dir / "forecast_states.npz"),
    }
    write_json(experiment_dir / "phase3_national_reset_manifest.json", {"experiment_id": NR01_EXPERIMENT_ID, "artifact_paths": artifact_paths})
    ctx.record_stage_outputs("phase3_national_reset", [Path(path) for path in artifact_paths.values()])
    return {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": nr00["archive_run_id"],
        "experiment_id": NR01_EXPERIMENT_ID,
        "phase_dir": str(experiment_dir),
        "artifact_paths": artifact_paths,
        **fit_result,
    }


def run_phase3_national_reset_delay_aux(
    *,
    run_id: str,
    plugin_id: str,
    archive_run_id: str | None = None,
    start_quarter: str | None = None,
    holdout_years: list[int] | None = None,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    start_quarter = _national_reset_start_quarter(start_quarter)
    nr00 = run_phase3_national_reset_observation_table(
        run_id=run_id,
        plugin_id=plugin_id,
        archive_run_id=archive_run_id,
        start_quarter=start_quarter,
    )
    experiment_dir = ensure_dir(ctx.run_dir / "phase3_national_reset" / NR02_EXPERIMENT_ID)
    holdout_years = _national_reset_holdout_years(holdout_years)
    observation_rows = list(nr00["observation_table"]["rows"])
    nr01 = run_phase3_national_reset_baseline(
        run_id=run_id,
        plugin_id=plugin_id,
        archive_run_id=archive_run_id,
        start_quarter=start_quarter,
        holdout_years=holdout_years,
    )
    fit_result = fit_national_uda_delay_aux(
        observation_rows=observation_rows,
        estimated_plhiv_by_quarter=dict(nr00["estimated_plhiv_by_quarter"]),
        diagnosis_flow_targets=dict(nr00["diagnosis_flow_targets_by_quarter"]),
        artifact_dir=experiment_dir,
        holdout_years=holdout_years,
        reference_metrics={
            "reference_model_mean_absolute_error": float(nr01["baseline_comparison"]["model_mean_absolute_error"]),
            "reference_diagnosis_flow_mean_absolute_error": float(nr01["diagnosis_flow_evaluation"]["mean_absolute_error"]),
        },
    )
    experiment_spec = {
        "experiment_id": NR02_EXPERIMENT_ID,
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": nr00["archive_run_id"],
        "start_quarter": start_quarter,
        "holdout_years": holdout_years,
        "state_names": ["U", "D", "A"],
        "primary_observations": list(PRIMARY_METRICS),
        "auxiliary_observations": ["advanced_hiv_cases_period", "median_cd4_at_enrollment"],
        "decision_rule": "keep_only_if_diagnosis_flow_improves_without_front_half_regression",
    }
    write_json(experiment_dir / "experiment_spec.json", experiment_spec)
    write_json(experiment_dir / "observation_table.json", nr00["observation_table"])
    artifact_paths = {
        "experiment_spec": str(experiment_dir / "experiment_spec.json"),
        "observation_table": str(experiment_dir / "observation_table.json"),
        "fit_artifact": str(experiment_dir / "fit_artifact.json"),
        "evaluation": str(experiment_dir / "evaluation.json"),
        "baseline_comparison": str(experiment_dir / "baseline_comparison.json"),
        "diagnosis_flow_evaluation": str(experiment_dir / "diagnosis_flow_evaluation.json"),
        "delay_aux_summary": str(experiment_dir / "delay_aux_summary.json"),
        "decision": str(experiment_dir / "decision.json"),
        "state_estimates": str(experiment_dir / "state_estimates.npz"),
        "forecast_states": str(experiment_dir / "forecast_states.npz"),
    }
    write_json(experiment_dir / "phase3_national_reset_manifest.json", {"experiment_id": NR02_EXPERIMENT_ID, "artifact_paths": artifact_paths})
    ctx.record_stage_outputs("phase3_national_reset", [Path(path) for path in artifact_paths.values()])
    return {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": nr00["archive_run_id"],
        "experiment_id": NR02_EXPERIMENT_ID,
        "phase_dir": str(experiment_dir),
        "artifact_paths": artifact_paths,
        **fit_result,
    }


def run_phase3_national_reset_vl_observation_process(
    *,
    run_id: str,
    plugin_id: str,
    archive_run_id: str | None = None,
    start_quarter: str | None = None,
    holdout_years: list[int] | None = None,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    start_quarter = _national_reset_start_quarter(start_quarter)
    nr00 = run_phase3_national_reset_observation_table(
        run_id=run_id,
        plugin_id=plugin_id,
        archive_run_id=archive_run_id,
        start_quarter=start_quarter,
    )
    nr02 = run_phase3_national_reset_delay_aux(
        run_id=run_id,
        plugin_id=plugin_id,
        archive_run_id=archive_run_id,
        start_quarter=start_quarter,
        holdout_years=holdout_years,
    )
    upstream_result = nr02
    if not bool(nr02.get("decision", {}).get("keep")):
        nr01 = run_phase3_national_reset_baseline(
            run_id=run_id,
            plugin_id=plugin_id,
            archive_run_id=archive_run_id,
            start_quarter=start_quarter,
            holdout_years=holdout_years,
        )
        upstream_result = nr01
    experiment_dir = ensure_dir(ctx.run_dir / "phase3_national_reset" / NR03_EXPERIMENT_ID)
    holdout_years = _national_reset_holdout_years(holdout_years)
    observation_rows = list(nr00["observation_table"]["rows"])
    fit_result = fit_national_uda_vl_observation_process(
        observation_rows=observation_rows,
        upstream_result=upstream_result,
        artifact_dir=experiment_dir,
        holdout_years=holdout_years,
    )
    experiment_spec = {
        "experiment_id": NR03_EXPERIMENT_ID,
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": nr00["archive_run_id"],
        "start_quarter": start_quarter,
        "holdout_years": holdout_years,
        "state_names": ["U", "D", "A"],
        "primary_observations": list(PRIMARY_METRICS),
        "observation_process_metrics": ["tested_for_viral_load", "virally_suppressed"],
        "front_half_reference_experiment": str(upstream_result["experiment_id"]),
        "decision_rule": "keep_only_if_front_half_survives_and_vl_process_beats_naive_service_baseline",
    }
    write_json(experiment_dir / "experiment_spec.json", experiment_spec)
    write_json(experiment_dir / "observation_table.json", nr00["observation_table"])
    artifact_paths = {
        "experiment_spec": str(experiment_dir / "experiment_spec.json"),
        "observation_table": str(experiment_dir / "observation_table.json"),
        "fit_artifact": str(experiment_dir / "fit_artifact.json"),
        "evaluation": str(experiment_dir / "evaluation.json"),
        "baseline_comparison": str(experiment_dir / "baseline_comparison.json"),
        "diagnosis_flow_evaluation": str(experiment_dir / "diagnosis_flow_evaluation.json"),
        "decision": str(experiment_dir / "decision.json"),
        "vl_observation_process": str(experiment_dir / "vl_observation_process.json"),
    }
    if (experiment_dir / "delay_aux_summary.json").exists():
        artifact_paths["delay_aux_summary"] = str(experiment_dir / "delay_aux_summary.json")
    write_json(experiment_dir / "phase3_national_reset_manifest.json", {"experiment_id": NR03_EXPERIMENT_ID, "artifact_paths": artifact_paths})
    ctx.record_stage_outputs("phase3_national_reset", [Path(path) for path in artifact_paths.values()])
    return {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": nr00["archive_run_id"],
        "experiment_id": NR03_EXPERIMENT_ID,
        "phase_dir": str(experiment_dir),
        "artifact_paths": artifact_paths,
        **fit_result,
    }


def run_phase3_national_reset_deferred_complexity_scan(
    *,
    run_id: str,
    plugin_id: str,
    archive_run_id: str | None = None,
    start_quarter: str | None = None,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    start_quarter = _national_reset_start_quarter(start_quarter)
    nr00 = run_phase3_national_reset_observation_table(
        run_id=run_id,
        plugin_id=plugin_id,
        archive_run_id=archive_run_id,
        start_quarter=start_quarter,
    )
    experiment_dir = ensure_dir(ctx.run_dir / "phase3_national_reset" / NR05_EXPERIMENT_ID)
    factor_scan = _deferred_factor_screen(str(nr00["archive_run_id"]))
    dx_items = _dx_queue_items()
    dx01 = next(item for item in dx_items if str(item["deferred_id"]) == "DX-01")
    dx01["phase2_phase15_scan"] = dict(factor_scan["summary"])
    dx01["recommended_candidate_factor_ids"] = list(factor_scan["summary"]["promotable_factor_ids"])
    dx01["standalone_candidate_required"] = True
    dx01["promotion_status"] = (
        "blocked_no_promotable_factor"
        if not factor_scan["summary"]["promotable_factor_ids"]
        else "candidate_ready_for_standalone_experiment"
    )
    experiment_spec = {
        "experiment_id": NR05_EXPERIMENT_ID,
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": nr00["archive_run_id"],
        "start_quarter": start_quarter,
        "scan_inputs": [
            "phase2/multiscale_phase3_target_blankets.json",
            "phase15/multiscale_factor_catalog.json",
            "phase15/multiscale_factor_axes.json",
            "phase15/multiscale_national_factor_tensor.npz",
        ],
        "decision_rule": "keep_only_if_deferred_queue_and_phase2_screening_are_explicit",
    }
    complexity_queue = {
        "archive_run_id": nr00["archive_run_id"],
        "observation_table_reference": str(experiment_dir / "observation_table.json"),
        "deferred_items": dx_items,
        "phase2_factor_scan": factor_scan,
    }
    decision_checks = [
        {
            "name": "every_deferred_item_has_metric_target",
            "passed": all(str(item.get("metric_target") or "").strip() for item in dx_items),
            "actual": len(dx_items),
            "target": len(dx_items),
        },
        {
            "name": "every_deferred_item_has_revert_condition",
            "passed": all(str(item.get("revert_condition") or "").strip() for item in dx_items),
            "actual": len(dx_items),
            "target": len(dx_items),
        },
        {
            "name": "phase2_phase15_scan_loaded",
            "passed": int(factor_scan["summary"]["blanket_factor_count"]) > 0 and int(factor_scan["summary"]["active_window_month_count"]) > 0,
            "actual": {
                "blanket_factor_count": int(factor_scan["summary"]["blanket_factor_count"]),
                "active_window_month_count": int(factor_scan["summary"]["active_window_month_count"]),
            },
            "target": {"blanket_factor_count": ">0", "active_window_month_count": ">0"},
        },
        {
            "name": "dx01_has_explicit_screening_verdict",
            "passed": int(len(factor_scan["factor_rows"])) > 0,
            "actual": int(len(factor_scan["factor_rows"])),
            "target": ">0",
        },
    ]
    decision = {
        "experiment_id": NR05_EXPERIMENT_ID,
        "keep": all(bool(row["passed"]) for row in decision_checks),
        "decision_rule": "keep_only_if_deferred_queue_and_phase2_screening_are_explicit",
        "checks": decision_checks,
        "summary": {
            "promotable_factor_count": int(factor_scan["summary"]["promotable_factor_count"]),
            "promotable_factor_ids": list(factor_scan["summary"]["promotable_factor_ids"]),
            "blocked_factor_ids": list(factor_scan["summary"]["blocked_factor_ids"]),
        },
    }

    write_json(experiment_dir / "experiment_spec.json", experiment_spec)
    write_json(experiment_dir / "observation_table.json", nr00["observation_table"])
    write_json(experiment_dir / "phase2_factor_scan.json", factor_scan)
    write_json(experiment_dir / "complexity_queue.json", complexity_queue)
    write_json(experiment_dir / "decision.json", decision)
    artifact_paths = {
        "experiment_spec": str(experiment_dir / "experiment_spec.json"),
        "observation_table": str(experiment_dir / "observation_table.json"),
        "phase2_factor_scan": str(experiment_dir / "phase2_factor_scan.json"),
        "complexity_queue": str(experiment_dir / "complexity_queue.json"),
        "decision": str(experiment_dir / "decision.json"),
    }
    write_json(experiment_dir / "phase3_national_reset_manifest.json", {"experiment_id": NR05_EXPERIMENT_ID, "artifact_paths": artifact_paths})
    ctx.record_stage_outputs("phase3_national_reset", [Path(path) for path in artifact_paths.values()])
    return {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "archive_run_id": nr00["archive_run_id"],
        "experiment_id": NR05_EXPERIMENT_ID,
        "phase_dir": str(experiment_dir),
        "artifact_paths": artifact_paths,
        "complexity_queue": complexity_queue,
        "decision": decision,
    }
