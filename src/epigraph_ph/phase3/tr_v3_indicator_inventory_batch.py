from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
DEFAULT_ACTIVE_MONTHLY_RUN_ID = "tr-v3-phase2-sidecar-ablation-20260419-s00-monthly"
DEFAULT_CANDIDATE_MONTHLY_RUN_ID = "tr-v3-phase2-testing-prevention-rebuild-20260419-s01-monthly"
DERIVED_CANONICAL_MAP = {
    "annual_hiv_tests_volume": "annual_hiv_tests_volume_per_100k",
    "prep_people_receiving": "prep_people_receiving_per_100k",
}
HISTORICAL_PANEL_NUMERIC_FIELDS = (
    "estimated_plhiv",
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
)
RAW_DIAGNOSIS_FLOW_FIELDS = ("diagnosed_count", "diagnosed_share", "estimated_plhiv")
RAW_PROGRAM_POINT_FIELDS = ("diagnosed", "estimated_plhiv", "on_art", "suppressed", "viral_load_tested")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return float(numeric)


def _month_ordinal(month_label: str) -> int:
    year_str, month_str = str(month_label).split("-", 1)
    return int(year_str) * 12 + int(month_str) - 1


def _is_national_row(row: dict[str, Any]) -> bool:
    geo_resolution = str(row.get("geo_resolution") or "").strip().lower()
    geo = str(row.get("geo") or row.get("province") or row.get("country") or "").strip().lower()
    region = str(row.get("region") or "").strip().lower()
    return geo_resolution == "national" or geo == "philippines" or region == "national"


def _extract_time_label(row: dict[str, Any]) -> str:
    for key in ("time", "effective_month", "source_month", "period_end", "period_start"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    year = row.get("year")
    try:
        return f"{int(year):04d}"
    except Exception:
        return ""


def _extract_year(row: dict[str, Any]) -> int | None:
    raw_year = row.get("year")
    try:
        return int(raw_year)
    except Exception:
        pass
    time_label = _extract_time_label(row)
    if MONTH_PATTERN.fullmatch(time_label):
        return int(time_label[:4])
    if re.fullmatch(r"^\d{4}$", time_label):
        return int(time_label)
    return None


def _infer_resolution(row: dict[str, Any], time_label: str) -> str:
    time_resolution = str(row.get("time_resolution") or "").strip().lower()
    if time_resolution:
        return time_resolution
    temporal_precision = str(row.get("temporal_precision") or "").strip().lower()
    if "month" in temporal_precision:
        return "monthly"
    if "quarter" in temporal_precision:
        return "quarterly"
    if "annual" in temporal_precision or "year" in temporal_precision:
        return "annual"
    if MONTH_PATTERN.fullmatch(time_label):
        return "monthly"
    if re.fullmatch(r"^\d{4}$", time_label):
        return "annual"
    return "unknown"


def _cadence_class(time_labels: set[str], resolutions: set[str], numeric_obs_count: int) -> str:
    if numeric_obs_count <= 0:
        return "no_numeric_support"
    if numeric_obs_count == 1 or len(time_labels) <= 1:
        return "single_reading"
    normalized = {str(label).strip() for label in time_labels if str(label).strip()}
    resolution_set = {str(value).strip().lower() for value in resolutions if str(value).strip()}
    infer_monthly_from_labels = not any(value in resolution_set for value in {"annual", "quarterly"})
    infer_annual_from_labels = "monthly" not in resolution_set
    has_monthly = "monthly" in resolution_set or (infer_monthly_from_labels and any(MONTH_PATTERN.fullmatch(label) for label in normalized))
    has_annual = "annual" in resolution_set or (infer_annual_from_labels and any(re.fullmatch(r"^\d{4}$", label) for label in normalized))
    if has_monthly and has_annual:
        return "mixed_frequency"
    if has_monthly:
        month_points = sorted({_month_ordinal(label) for label in normalized if MONTH_PATTERN.fullmatch(label)})
        if len(month_points) <= 1:
            return "single_reading"
        diffs = np.diff(np.asarray(month_points, dtype=np.int32))
        median_gap = float(np.median(diffs)) if diffs.size else 0.0
        if median_gap <= 1.5:
            return "monthly_dense" if len(month_points) >= 12 else "monthly_sparse"
        if median_gap <= 3.5:
            return "quarterly_like"
        return "monthly_irregular"
    year_points: list[int] = []
    for label in normalized:
        if re.fullmatch(r"^\d{4}$", label):
            year_points.append(int(label))
            continue
        if MONTH_PATTERN.fullmatch(label):
            year_points.append(int(label[:4]))
    year_points = sorted(set(year_points))
    if len(year_points) <= 1:
        return "single_reading"
    gaps = np.diff(np.asarray(year_points, dtype=np.int32))
    median_gap = float(np.median(gaps)) if gaps.size else 0.0
    if median_gap <= 1.25:
        return "annual"
    if median_gap <= 2.5:
        return "biannual"
    return "sparse_annual"


def _empty_summary(name: str) -> dict[str, Any]:
    return {
        "indicator_name": name,
        "obs_count": 0,
        "numeric_obs_count": 0,
        "time_labels": set(),
        "resolutions": set(),
        "years": [],
        "source_ids": set(),
        "source_banks": set(),
        "candidate_blocks": set(),
        "measurement_roles": set(),
        "units": set(),
    }


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    time_labels = {str(label).strip() for label in set(summary.get("time_labels") or set()) if str(label).strip()}
    years = sorted({int(year) for year in list(summary.get("years") or [])})
    resolutions = {str(value).strip().lower() for value in set(summary.get("resolutions") or set()) if str(value).strip()}
    return {
        "indicator_name": str(summary.get("indicator_name") or "").strip(),
        "obs_count": int(summary.get("obs_count") or 0),
        "numeric_obs_count": int(summary.get("numeric_obs_count") or 0),
        "first_time": min(time_labels) if time_labels else "",
        "last_time": max(time_labels) if time_labels else "",
        "year_start": int(min(years)) if years else None,
        "year_end": int(max(years)) if years else None,
        "time_resolution_labels": sorted(resolutions),
        "cadence_class": _cadence_class(time_labels, resolutions, int(summary.get("numeric_obs_count") or 0)),
        "source_ids": sorted(str(value) for value in set(summary.get("source_ids") or set()) if str(value)),
        "source_banks": sorted(str(value) for value in set(summary.get("source_banks") or set()) if str(value)),
        "candidate_blocks": sorted(str(value) for value in set(summary.get("candidate_blocks") or set()) if str(value)),
        "measurement_roles": sorted(str(value) for value in set(summary.get("measurement_roles") or set()) if str(value)),
        "units": sorted(str(value) for value in set(summary.get("units") or set()) if str(value)),
    }


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    name_fn: Callable[[dict[str, Any]], str],
    numeric_fn: Callable[[dict[str, Any]], float | None],
    row_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row_filter is not None and not row_filter(row):
            continue
        name = str(name_fn(row) or "").strip()
        if not name:
            continue
        summary = summaries.setdefault(name, _empty_summary(name))
        summary["obs_count"] += 1
        numeric_value = numeric_fn(row)
        if numeric_value is not None:
            summary["numeric_obs_count"] += 1
        time_label = _extract_time_label(row)
        if time_label:
            summary["time_labels"].add(time_label)
        resolution = _infer_resolution(row, time_label)
        if resolution:
            summary["resolutions"].add(resolution)
        year = _extract_year(row)
        if year is not None:
            summary["years"].append(int(year))
        source_id = str(row.get("source_id") or "").strip()
        if source_id:
            summary["source_ids"].add(source_id)
        source_bank = str(row.get("source_bank") or "").strip()
        if source_bank:
            summary["source_banks"].add(source_bank)
        candidate_block = str(row.get("candidate_block") or "").strip()
        if candidate_block:
            summary["candidate_blocks"].add(candidate_block)
        measurement_role = str(row.get("measurement_role") or "").strip()
        if measurement_role:
            summary["measurement_roles"].add(measurement_role)
        unit = str(row.get("unit") or row.get("normalized_unit") or "").strip()
        if unit:
            summary["units"].add(unit)
    return {name: _finalize_summary(summary) for name, summary in summaries.items()}


def _summarize_historical_panel_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        time_label = _extract_time_label(row)
        for field in HISTORICAL_PANEL_NUMERIC_FIELDS:
            numeric_value = _safe_float(row.get(field))
            if numeric_value is None:
                continue
            summary = summaries.setdefault(field, _empty_summary(field))
            summary["obs_count"] += 1
            summary["numeric_obs_count"] += 1
            if time_label:
                summary["time_labels"].add(time_label)
            summary["resolutions"].add("annual")
            year = _extract_year(row)
            if year is not None:
                summary["years"].append(int(year))
            summary["source_ids"].add("historical_harp_panel")
            summary["source_banks"].add("historical_harp_panel")
            summary["units"].add("count_people")
    return {name: _finalize_summary(summary) for name, summary in summaries.items()}


def _summarize_raw_field_rows(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    source_bank: str,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        time_label = _extract_time_label(row)
        for field in fields:
            numeric_value = _safe_float(row.get(field))
            if numeric_value is None:
                continue
            summary = summaries.setdefault(field, _empty_summary(field))
            summary["obs_count"] += 1
            summary["numeric_obs_count"] += 1
            if time_label:
                summary["time_labels"].add(time_label)
            summary["resolutions"].add(_infer_resolution(row, time_label))
            year = _extract_year(row)
            if year is not None:
                summary["years"].append(int(year))
            summary["source_ids"].add(source_bank)
            summary["source_banks"].add(source_bank)
    return {name: _finalize_summary(summary) for name, summary in summaries.items()}


def _summary_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("indicator_name") or "").strip(): dict(row) for row in rows if str(row.get("indicator_name") or "").strip()}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default=[])
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if "rows" in payload:
            return [dict(row) for row in list(payload.get("rows") or []) if isinstance(row, dict)]
        if "points" in payload:
            return [dict(row) for row in list(payload.get("points") or []) if isinstance(row, dict)]
    return []


def _derive_source_run_id(active_monthly_run_dir: Path) -> str:
    payload = dict(read_json(active_monthly_run_dir / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.json", default={}))
    source_run_id = str(payload.get("source_run_id") or "").strip()
    if not source_run_id:
        raise ValueError(f"could not derive source_run_id from {active_monthly_run_dir}")
    return source_run_id


def _current_status(
    *,
    indicator_name: str,
    active_structural_names: set[str],
    active_full_eval_names: set[str],
    active_excluded_names: set[str],
    candidate_structural_names: set[str],
    national_numeric_obs: int,
    subnational_numeric_obs: int,
) -> str:
    if indicator_name in active_structural_names:
        return "active_structural"
    if indicator_name in active_excluded_names or indicator_name in active_full_eval_names:
        return "evaluation_only"
    if indicator_name in candidate_structural_names:
        return "candidate_only"
    if national_numeric_obs > 0:
        return "unused_numeric_source"
    if subnational_numeric_obs > 0:
        return "subnational_only_numeric"
    return "unused_nonnumeric_source"


def _plot_status_counts(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    counter = Counter(str(row.get("current_status") or "") for row in rows)
    labels = [label for label, _ in counter.most_common()]
    values = [counter[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values, color="#1f77b4")
    ax.set_title("Indicator count by current use status")
    ax.set_ylabel("indicator count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_top_indicator_bars(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    title: str,
    value_key: str,
    limit: int = 15,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ranked = sorted(rows, key=lambda row: (float(row.get(value_key) or 0.0), str(row.get("indicator_name") or "")), reverse=True)[:limit]
    if not ranked:
        return
    labels = [str(row.get("indicator_name") or "") for row in ranked]
    values = [float(row.get(value_key) or 0.0) for row in ranked]
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.35 * len(labels))))
    ax.barh(labels, values, color="#ff7f0e")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(value_key.replace("_", " "))
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_sparse_indicators(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    sparse_rows = [
        row
        for row in rows
        if str(row.get("cadence_class") or "") in {"single_reading", "biannual", "sparse_annual", "monthly_sparse"}
        and str(row.get("current_status") or "") != "active_structural"
    ]
    ranked = sorted(sparse_rows, key=lambda row: (float(row.get("national_numeric_obs") or 0.0), str(row.get("indicator_name") or "")), reverse=True)[:20]
    if not ranked:
        return
    labels = [str(row.get("indicator_name") or "") for row in ranked]
    values = [float(row.get("national_numeric_obs") or 0.0) for row in ranked]
    colors = []
    for row in ranked:
        cadence = str(row.get("cadence_class") or "")
        colors.append(
            {
                "single_reading": "#d62728",
                "biannual": "#9467bd",
                "sparse_annual": "#8c564b",
                "monthly_sparse": "#2ca02c",
            }.get(cadence, "#7f7f7f")
        )
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.35 * len(labels))))
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_title("Sparse national indicators outside the active structural kernel")
    ax.set_xlabel("national numeric observations")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, tuple, set)):
                    normalized[key] = "|".join(str(item) for item in value)
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def _report_markdown(payload: dict[str, Any]) -> str:
    top_unused = list(payload.get("top_unused_numeric") or [])
    top_sparse = list(payload.get("top_sparse_indicators") or [])
    top_subnational = list(payload.get("top_subnational_only") or [])
    lines = [
        "# Indicator Inventory Audit",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Active monthly run: `{payload['active_monthly_run_id']}`",
        f"- Candidate monthly run: `{payload.get('candidate_monthly_run_id') or 'none'}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Total inventoried indicators: `{payload['inventory_count']}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in dict(payload.get("status_counts") or {}).items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Current Read",
            "",
            "- `active_structural` means the indicator is present in the live national structural lane.",
            "- `evaluation_only` means it is available in the monthly evaluation bundle or explicitly excluded from structural use.",
            "- `candidate_only` means it only appeared in the reverted testing-prevention rebuild.",
            "- `unused_numeric_source` means the data exists nationally with numeric values but is not currently live.",
            "- `subnational_only_numeric` means the source signal exists numerically but mostly below the national aggregation level.",
            "- `unused_nonnumeric_source` means the repo knows about the indicator but does not have usable numeric support yet.",
            "",
            "## Highest-Value Unused Numeric Indicators",
            "",
        ]
    )
    if top_unused:
        for row in top_unused:
            lines.append(
                f"- `{row['indicator_name']}`: national numeric obs `{row['national_numeric_obs']}`, cadence `{row['cadence_class']}`, source families `{', '.join(list(row.get('source_families') or []))}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Sparse / Singleton Indicators", ""])
    if top_sparse:
        for row in top_sparse:
            lines.append(
                f"- `{row['indicator_name']}`: cadence `{row['cadence_class']}`, national numeric obs `{row['national_numeric_obs']}`, years `{row.get('year_start')}` to `{row.get('year_end')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Subnational-Only Numeric Indicators", ""])
    if top_subnational:
        for row in top_subnational:
            lines.append(
                f"- `{row['indicator_name']}`: subnational numeric obs `{row['subnational_numeric_obs']}`, candidate blocks `{', '.join(list(row.get('source_candidate_blocks') or [])) or 'none'}`"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `analysis/indicator_inventory.csv`",
            "- `analysis/unused_numeric_indicator_inventory.csv`",
            "- `analysis/raw_program_field_inventory.csv`",
            "- `analysis/indicator_status_counts.png`",
            "- `analysis/unused_numeric_national_indicators.png`",
            "- `analysis/subnational_only_numeric_indicators.png`",
            "- `analysis/sparse_indicator_inventory.png`",
            "",
        ]
    )
    return "\n".join(lines)


def run_tr_v3_indicator_inventory_batch(
    *,
    run_id: str,
    active_monthly_run_id: str = DEFAULT_ACTIVE_MONTHLY_RUN_ID,
    candidate_monthly_run_id: str = DEFAULT_CANDIDATE_MONTHLY_RUN_ID,
) -> dict[str, Any]:
    active_monthly_run_dir = ROOT_DIR / "artifacts" / "runs" / str(active_monthly_run_id)
    if not active_monthly_run_dir.exists():
        raise FileNotFoundError(f"active monthly run does not exist: {active_monthly_run_dir}")
    source_run_id = _derive_source_run_id(active_monthly_run_dir)
    source_run_dir = ROOT_DIR / "artifacts" / "runs" / str(source_run_id)
    if not source_run_dir.exists():
        raise FileNotFoundError(f"source run does not exist: {source_run_dir}")
    candidate_monthly_run_dir = ROOT_DIR / "artifacts" / "runs" / str(candidate_monthly_run_id)
    if not candidate_monthly_run_dir.exists():
        candidate_monthly_run_id = ""
        candidate_monthly_run_dir = Path()

    source_phase1_rows = _load_rows(source_run_dir / "phase1" / "normalized_subparameters.json")
    active_structural_rows = _load_rows(active_monthly_run_dir / "phase1" / "normalized_subparameters.json")
    active_full_eval_rows = _load_rows(active_monthly_run_dir / "phase1" / "normalized_subparameters_full_for_evaluation.json")
    active_exclusion_summary = dict(read_json(active_monthly_run_dir / "phase1" / "structural_exclusion_summary.json", default={}))
    candidate_structural_rows = _load_rows(candidate_monthly_run_dir / "phase1" / "normalized_subparameters.json") if candidate_monthly_run_id else []

    observed_program_rows = _load_rows(source_run_dir / "harp_archive" / "observed_program_panel.json")
    historical_metric_rows = _load_rows(source_run_dir / "harp_archive" / "historical_metric_rows.json")
    multinational_metric_rows = _load_rows(source_run_dir / "harp_archive" / "multinational_hiv_metric_rows.json")
    historical_panel_rows = _load_rows(source_run_dir / "harp_archive" / "historical_harp_panel.json")
    diagnosis_flow_rows = _load_rows(source_run_dir / "harp_archive" / "diagnosis_flow_points.json")
    program_point_rows = _load_rows(source_run_dir / "harp_archive" / "harp_program_points.json")

    source_phase1_national = _summarize_rows(
        source_phase1_rows,
        name_fn=lambda row: str(row.get("canonical_name") or ""),
        numeric_fn=lambda row: _safe_float(row.get("model_numeric_value")),
        row_filter=_is_national_row,
    )
    source_phase1_all_geo = _summarize_rows(
        source_phase1_rows,
        name_fn=lambda row: str(row.get("canonical_name") or ""),
        numeric_fn=lambda row: _safe_float(row.get("model_numeric_value")),
    )
    active_structural = _summarize_rows(
        active_structural_rows,
        name_fn=lambda row: str(row.get("canonical_name") or ""),
        numeric_fn=lambda row: _safe_float(row.get("model_numeric_value")),
    )
    active_full_eval = _summarize_rows(
        active_full_eval_rows,
        name_fn=lambda row: str(row.get("canonical_name") or ""),
        numeric_fn=lambda row: _safe_float(row.get("model_numeric_value")),
    )
    candidate_structural = _summarize_rows(
        candidate_structural_rows,
        name_fn=lambda row: str(row.get("canonical_name") or ""),
        numeric_fn=lambda row: _safe_float(row.get("model_numeric_value")),
    )
    observed_program = _summarize_rows(
        observed_program_rows,
        name_fn=lambda row: str(row.get("metric_name") or ""),
        numeric_fn=lambda row: _safe_float(row.get("value")),
    )
    historical_metric = _summarize_rows(
        historical_metric_rows,
        name_fn=lambda row: str(row.get("metric_name") or ""),
        numeric_fn=lambda row: _safe_float(row.get("value")),
    )
    multinational_metric = _summarize_rows(
        multinational_metric_rows,
        name_fn=lambda row: str(row.get("metric_name") or ""),
        numeric_fn=lambda row: _safe_float(row.get("value")),
    )
    historical_panel = _summarize_historical_panel_rows(historical_panel_rows)
    raw_diagnosis_fields = _summarize_raw_field_rows(
        diagnosis_flow_rows,
        fields=RAW_DIAGNOSIS_FLOW_FIELDS,
        source_bank="diagnosis_flow_points",
    )
    raw_program_fields = _summarize_raw_field_rows(
        program_point_rows,
        fields=RAW_PROGRAM_POINT_FIELDS,
        source_bank="harp_program_points",
    )

    active_structural_names = set(active_structural.keys())
    active_full_eval_names = set(active_full_eval.keys())
    active_excluded_names = {str(value).strip() for value in list(active_exclusion_summary.get("excluded_canonicals") or []) if str(value).strip()}
    candidate_structural_names = set(candidate_structural.keys())

    inventory_names = set().union(
        source_phase1_national.keys(),
        source_phase1_all_geo.keys(),
        active_structural.keys(),
        active_full_eval.keys(),
        candidate_structural.keys(),
        observed_program.keys(),
        historical_metric.keys(),
        multinational_metric.keys(),
        historical_panel.keys(),
    )
    inventory_names.update(DERIVED_CANONICAL_MAP.values())

    inventory_rows: list[dict[str, Any]] = []
    for indicator_name in sorted(inventory_names):
        national_source = dict(source_phase1_national.get(indicator_name) or {})
        all_geo_source = dict(source_phase1_all_geo.get(indicator_name) or {})
        active_eval_summary = dict(active_full_eval.get(indicator_name) or {})
        active_struct_summary = dict(active_structural.get(indicator_name) or {})
        candidate_summary = dict(candidate_structural.get(indicator_name) or {})
        observed_summary = dict(observed_program.get(indicator_name) or {})
        historical_metric_summary = dict(historical_metric.get(indicator_name) or {})
        multinational_summary = dict(multinational_metric.get(indicator_name) or {})
        historical_panel_summary = dict(historical_panel.get(indicator_name) or {})

        national_numeric_obs = int(national_source.get("numeric_obs_count") or 0)
        subnational_numeric_obs = max(0, int(all_geo_source.get("numeric_obs_count") or 0) - national_numeric_obs)
        source_families: list[str] = []
        for family_name, summary in (
            ("source_phase1_national", national_source),
            ("source_phase1_all_geo", all_geo_source),
            ("observed_program_panel", observed_summary),
            ("historical_metric_rows", historical_metric_summary),
            ("multinational_hiv_metric_rows", multinational_summary),
            ("historical_harp_panel", historical_panel_summary),
        ):
            if summary:
                source_families.append(family_name)

        derived_from_metric = ""
        admitted_as_derived_canonical = ""
        for raw_metric, derived_name in DERIVED_CANONICAL_MAP.items():
            if indicator_name == raw_metric:
                admitted_as_derived_canonical = derived_name
            if indicator_name == derived_name:
                derived_from_metric = raw_metric

        derived_historical_summary = dict(historical_metric.get(derived_from_metric) or {}) if derived_from_metric else {}
        derived_multinational_summary = dict(multinational_metric.get(derived_from_metric) or {}) if derived_from_metric else {}
        national_numeric_obs_total = national_numeric_obs
        national_numeric_obs_total += int(observed_summary.get("numeric_obs_count") or 0)
        national_numeric_obs_total += int(historical_metric_summary.get("numeric_obs_count") or 0)
        national_numeric_obs_total += int(multinational_summary.get("numeric_obs_count") or 0)
        national_numeric_obs_total += int(historical_panel_summary.get("numeric_obs_count") or 0)
        if derived_from_metric:
            national_numeric_obs_total += int(derived_historical_summary.get("numeric_obs_count") or 0)
            national_numeric_obs_total += int(derived_multinational_summary.get("numeric_obs_count") or 0)

        current_status = _current_status(
            indicator_name=indicator_name,
            active_structural_names=active_structural_names,
            active_full_eval_names=active_full_eval_names,
            active_excluded_names=active_excluded_names,
            candidate_structural_names=candidate_structural_names,
            national_numeric_obs=national_numeric_obs_total,
            subnational_numeric_obs=subnational_numeric_obs,
        )

        primary_summary = (
            national_source
            or historical_metric_summary
            or observed_summary
            or multinational_summary
            or historical_panel_summary
            or derived_historical_summary
            or derived_multinational_summary
            or active_eval_summary
            or candidate_summary
            or {}
        )
        if derived_from_metric and (derived_historical_summary or derived_multinational_summary):
            for family_name, summary in (
                ("historical_metric_rows", derived_historical_summary),
                ("multinational_hiv_metric_rows", derived_multinational_summary),
            ):
                if summary and family_name not in source_families:
                    source_families.append(family_name)
        row = {
            "indicator_name": indicator_name,
            "current_status": current_status,
            "cadence_class": str(primary_summary.get("cadence_class") or "unknown"),
            "national_numeric_obs": national_numeric_obs_total,
            "subnational_numeric_obs": subnational_numeric_obs,
            "active_full_eval_obs": int(active_eval_summary.get("numeric_obs_count") or 0),
            "active_structural_obs": int(active_struct_summary.get("numeric_obs_count") or 0),
            "candidate_structural_obs": int(candidate_summary.get("numeric_obs_count") or 0),
            "observed_program_numeric_obs": int(observed_summary.get("numeric_obs_count") or 0),
            "historical_metric_numeric_obs": int(historical_metric_summary.get("numeric_obs_count") or 0),
            "multinational_metric_numeric_obs": int(multinational_summary.get("numeric_obs_count") or 0),
            "historical_panel_numeric_obs": int(historical_panel_summary.get("numeric_obs_count") or 0),
            "first_time": str(primary_summary.get("first_time") or ""),
            "last_time": str(primary_summary.get("last_time") or ""),
            "year_start": primary_summary.get("year_start"),
            "year_end": primary_summary.get("year_end"),
            "source_candidate_blocks": sorted(set(national_source.get("candidate_blocks") or all_geo_source.get("candidate_blocks") or [])),
            "source_measurement_roles": sorted(set(national_source.get("measurement_roles") or all_geo_source.get("measurement_roles") or [])),
            "source_families": sorted(source_families),
            "excluded_from_active_structural": indicator_name in active_excluded_names,
            "in_active_full_eval": indicator_name in active_full_eval_names,
            "in_active_structural": indicator_name in active_structural_names,
            "in_candidate_structural": indicator_name in candidate_structural_names,
            "derived_from_metric": derived_from_metric,
            "admitted_as_derived_canonical": admitted_as_derived_canonical,
        }
        inventory_rows.append(row)

    raw_field_inventory: list[dict[str, Any]] = []
    for family_name, lookup in (("diagnosis_flow_points", raw_diagnosis_fields), ("harp_program_points", raw_program_fields)):
        for indicator_name, summary in sorted(lookup.items()):
            raw_field_inventory.append(
                {
                    "source_family": family_name,
                    "indicator_name": indicator_name,
                    "numeric_obs_count": int(summary.get("numeric_obs_count") or 0),
                    "cadence_class": str(summary.get("cadence_class") or "unknown"),
                    "first_time": str(summary.get("first_time") or ""),
                    "last_time": str(summary.get("last_time") or ""),
                    "year_start": summary.get("year_start"),
                    "year_end": summary.get("year_end"),
                }
            )

    unused_numeric_rows = [row for row in inventory_rows if str(row.get("current_status") or "") == "unused_numeric_source"]
    subnational_only_rows = [row for row in inventory_rows if str(row.get("current_status") or "") == "subnational_only_numeric"]
    sparse_rows = [
        row
        for row in inventory_rows
        if str(row.get("cadence_class") or "") in {"single_reading", "biannual", "sparse_annual", "monthly_sparse"}
        and str(row.get("current_status") or "") != "active_structural"
    ]
    top_unused = sorted(unused_numeric_rows, key=lambda row: (float(row.get("national_numeric_obs") or 0.0), str(row.get("indicator_name") or "")), reverse=True)[:12]
    top_subnational = sorted(subnational_only_rows, key=lambda row: (float(row.get("subnational_numeric_obs") or 0.0), str(row.get("indicator_name") or "")), reverse=True)[:12]
    top_sparse = sorted(sparse_rows, key=lambda row: (float(row.get("national_numeric_obs") or 0.0), str(row.get("indicator_name") or "")), reverse=True)[:12]

    run_dir = ROOT_DIR / "artifacts" / "runs" / str(run_id)
    analysis_dir = ensure_dir(run_dir / "analysis")
    _write_csv(analysis_dir / "indicator_inventory.csv", inventory_rows)
    _write_csv(analysis_dir / "unused_numeric_indicator_inventory.csv", unused_numeric_rows)
    _write_csv(analysis_dir / "raw_program_field_inventory.csv", raw_field_inventory)
    _plot_status_counts(inventory_rows, analysis_dir / "indicator_status_counts.png")
    _plot_top_indicator_bars(
        unused_numeric_rows,
        output_path=analysis_dir / "unused_numeric_national_indicators.png",
        title="Top unused national numeric indicators",
        value_key="national_numeric_obs",
    )
    _plot_top_indicator_bars(
        subnational_only_rows,
        output_path=analysis_dir / "subnational_only_numeric_indicators.png",
        title="Top subnational-only numeric indicators",
        value_key="subnational_numeric_obs",
    )
    _plot_sparse_indicators(inventory_rows, analysis_dir / "sparse_indicator_inventory.png")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "active_monthly_run_id": str(active_monthly_run_id),
        "candidate_monthly_run_id": str(candidate_monthly_run_id),
        "source_run_id": str(source_run_id),
        "inventory_count": int(len(inventory_rows)),
        "status_counts": {str(name): int(value) for name, value in Counter(str(row.get("current_status") or "") for row in inventory_rows).most_common()},
        "top_unused_numeric": top_unused,
        "top_subnational_only": top_subnational,
        "top_sparse_indicators": top_sparse,
        "artifacts": {
            "indicator_inventory_csv": str(analysis_dir / "indicator_inventory.csv"),
            "unused_numeric_csv": str(analysis_dir / "unused_numeric_indicator_inventory.csv"),
            "raw_program_field_inventory_csv": str(analysis_dir / "raw_program_field_inventory.csv"),
            "status_plot": str(analysis_dir / "indicator_status_counts.png"),
            "unused_numeric_plot": str(analysis_dir / "unused_numeric_national_indicators.png"),
            "subnational_plot": str(analysis_dir / "subnational_only_numeric_indicators.png"),
            "sparse_plot": str(analysis_dir / "sparse_indicator_inventory.png"),
        },
    }
    write_json(analysis_dir / "tr_v3_indicator_inventory_batch_report.json", payload)
    (analysis_dir / "tr_v3_indicator_inventory_batch_report.md").write_text(_report_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory all currently available Phase0/HARP/Phase1 indicators and tag which ones are active, excluded, or unused.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--active-monthly-run-id", default=DEFAULT_ACTIVE_MONTHLY_RUN_ID)
    parser.add_argument("--candidate-monthly-run-id", default=DEFAULT_CANDIDATE_MONTHLY_RUN_ID)
    args = parser.parse_args()
    run_tr_v3_indicator_inventory_batch(
        run_id=str(args.run_id),
        active_monthly_run_id=str(args.active_monthly_run_id),
        candidate_monthly_run_id=str(args.candidate_monthly_run_id),
    )


if __name__ == "__main__":
    main()
