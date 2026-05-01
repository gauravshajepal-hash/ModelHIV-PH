from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.geography import geo_resolution_label, infer_region_code, is_national_geo, normalize_geo_label
from epigraph_ph.latent_blocks import annotate_latent_indicator_fields
from epigraph_ph.phase1.latent_observability import build_direct_contextual_split, build_latent_observability_audit
from epigraph_ph.phase1.observation_noise import build_observation_noise_model
from epigraph_ph.phase1.pipeline import _build_denominator_tensor, _build_missing_mask, _tensor_preprocess
from epigraph_ph.phase15 import run_phase15_build
from epigraph_ph.phase15.latent_measurements import build_archive_derived_indicator_rows, build_sparse_indicator_cube
from epigraph_ph.phase2 import run_phase2_build
from epigraph_ph.runtime import ROOT_DIR, RunContext, ensure_dir, read_json, save_tensor_artifact, to_numpy, write_json


MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
DEFAULT_SOURCE_RUN_ID = "phase2-replay-source-20260414-s00"
DEFAULT_START_MONTH = "2010-01"
HARP_OBSERVED_METRICS = (
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
    "deaths_reported_period",
    "newly_enrolled_to_treatment",
    "advanced_hiv_cases_period",
)
HARP_DIAGNOSIS_FLOW_CANONICAL = "new_diagnosed_cases_period"
HARP_HISTORICAL_PANEL_METRICS = (
    "estimated_plhiv",
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
)
HARP_ANNUAL_UPSTREAM_COUNT_METRICS: dict[str, tuple[str, str]] = {
    "annual_hiv_tests_volume": ("annual_hiv_tests_volume_per_100k", "count_people_per_100k_population"),
    "prep_people_receiving": ("prep_people_receiving_per_100k", "count_people_per_100k_population"),
}
HARP_ANNUAL_UPSTREAM_PASS_THROUGH_METRICS = (
    "hiv_test_positivity_percent",
    "late_hiv_diagnosis_percent",
    "prevention_access",
    "prevention_coverage",
    "hiv_knowledge_index",
    "unaids_known_status_share_percent",
)
DEFAULT_STRUCTURAL_EXCLUDED_CANONICALS: tuple[str, ...] = ()
SOURCE_QUALITY_PRIORITY = {
    "official_doh_archive": 4,
    "official_user_provided_slide": 4,
    "official_mirror": 3,
    "official_doh_release": 3,
}


def _month_ordinal(month_label: str) -> int:
    year_str, month_str = str(month_label).split("-", 1)
    return int(year_str) * 12 + int(month_str) - 1


def _month_from_ordinal(ordinal: int) -> str:
    year = int(ordinal) // 12
    month = (int(ordinal) % 12) + 1
    return f"{year:04d}-{month:02d}"


def _contiguous_month_axis(start_month: str, end_month: str) -> list[str]:
    start = _month_ordinal(start_month)
    end = _month_ordinal(end_month)
    return [_month_from_ordinal(ordinal) for ordinal in range(start, end + 1)]


def _extract_month_tokens(payload: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(payload, dict):
        for key in ("time", "month", "effective_month", "period_end", "period_start", "source_month"):
            value = str(payload.get(key) or "").strip()
            if MONTH_PATTERN.fullmatch(value):
                tokens.append(value)
        for value in payload.values():
            tokens.extend(_extract_month_tokens(value))
    elif isinstance(payload, list):
        for value in payload:
            tokens.extend(_extract_month_tokens(value))
    return tokens


def _harp_month_axis(source_run_dir: Path, *, start_month: str) -> list[str]:
    harp_dir = source_run_dir / "harp_archive"
    payloads = [
        read_json(harp_dir / "observed_program_panel.json", default={}),
        read_json(harp_dir / "diagnosis_flow_points.json", default={}),
        read_json(harp_dir / "harp_program_points.json", default={}),
    ]
    month_tokens = sorted({token for payload in payloads for token in _extract_month_tokens(payload) if _month_ordinal(token) >= _month_ordinal(start_month)})
    if not month_tokens:
        raise ValueError(f"no monthly HARP evidence found at or after {start_month}")
    return _contiguous_month_axis(start_month, max(month_tokens, key=_month_ordinal))


def _is_national_monthly_row(row: dict[str, Any], *, start_month: str, end_month: str) -> bool:
    time_value = str(row.get("time") or "").strip()
    if not MONTH_PATTERN.fullmatch(time_value):
        return False
    ordinal = _month_ordinal(time_value)
    if ordinal < _month_ordinal(start_month) or ordinal > _month_ordinal(end_month):
        return False
    geo_value = normalize_geo_label(str(row.get("geo") or row.get("province") or ""), default_country_focus=is_national_geo(str(row.get("geo") or "")))
    region_value = str(row.get("region") or "").strip().lower()
    geo_resolution = str(row.get("geo_resolution") or "").strip().lower()
    return geo_resolution == "national" or geo_value == "Philippines" or region_value == "national"


def _filtered_monthly_rows(normalized_rows: list[dict[str, Any]], *, start_month: str, end_month: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_row in normalized_rows:
        row = dict(source_row)
        if not _is_national_monthly_row(row, start_month=start_month, end_month=end_month):
            continue
        time_value = str(row.get("time") or "").strip()
        row["time"] = time_value
        row["time_resolution"] = "monthly"
        row["year"] = int(time_value[:4])
        row["month"] = int(time_value[5:7])
        row["geo"] = "Philippines"
        row["province"] = "Philippines"
        row["region"] = "national"
        row["geo_resolution"] = "national"
        rows.append(row)
    return rows


def _within_month_window(month_label: str, *, start_month: str, end_month: str) -> bool:
    if not MONTH_PATTERN.fullmatch(str(month_label or "").strip()):
        return False
    ordinal = _month_ordinal(str(month_label))
    return _month_ordinal(start_month) <= ordinal <= _month_ordinal(end_month)


def _source_quality_rank(source_tier: str) -> int:
    return int(SOURCE_QUALITY_PRIORITY.get(str(source_tier or "").strip().lower(), 0))


def _dedupe_best_rows(rows: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in key_fields)
        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue
        current_score = (
            float(current.get("evidence_confidence") or 0.0),
            _source_quality_rank(str(current.get("source_quality_tier") or "")),
            1 if "official" in str(current.get("source_bank") or "").lower() else 0,
        )
        candidate_score = (
            float(row.get("evidence_confidence") or 0.0),
            _source_quality_rank(str(row.get("source_quality_tier") or "")),
            1 if "official" in str(row.get("source_bank") or "").lower() else 0,
        )
        if candidate_score > current_score:
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (str(row.get("time") or ""), str(row.get("canonical_name") or "")))


def _harp_numeric_row(
    *,
    canonical_name: str,
    time_value: str,
    numeric_value: float,
    plugin_id: str,
    source_bank: str,
    source_id: str,
    source_title: str,
    source_url: str,
    temporal_precision: str,
    normalized_unit: str,
    evidence_confidence: float,
    source_quality_tier: str,
    source_note: str = "",
) -> dict[str, Any]:
    confidence = float(np.clip(float(evidence_confidence), 0.25, 1.0))
    time_resolution = "monthly"
    if "annual" in str(temporal_precision or "").lower():
        time_resolution = "annual"
    elif "quarter" in str(temporal_precision or "").lower():
        time_resolution = "quarterly"
    base_row = {
        "canonical_name": canonical_name,
        "source_bank": source_bank,
        "source_id": source_id,
        "source_title": source_title,
        "source_url": source_url,
        "source_note": source_note,
        "source_quality_tier": source_quality_tier,
        "source_reliability_class": "official_routine_anchor",
        "geo": "Philippines",
        "province": "Philippines",
        "country": "Philippines",
        "region": "national",
        "geo_resolution": "national",
        "time": time_value,
        "time_resolution": time_resolution,
        "year": int(time_value[:4]),
        "month": int(time_value[5:7]),
        "temporal_precision": temporal_precision,
        "raw_numeric_value": float(numeric_value),
        "model_numeric_value": float(numeric_value),
        "normalized_unit": normalized_unit,
        "original_unit": normalized_unit,
        "signal_family": "harp_program_observed",
        "payload_family": "monthly_program_head",
        "evidence_class": "observed_numeric",
        "evidence_weight": confidence,
        "quality_weight": confidence,
        "measurement_quality_weight": confidence,
        "temporal_freshness_weight": 1.0,
        "spatial_relevance_weight": 1.0,
        "replication_weight": 1.0,
        "bias_penalty": 0.0,
        "is_anchor_eligible": True,
        "is_direct_measurement": True,
        "is_prior_only": False,
        "sex": "",
        "age_band": "",
        "kp_group": "remaining_population",
        "domain_family": "hiv_program_cascade",
        "pathway_family": "care_cascade",
        "literature_basis": [
            "Direct monthly or quarter-end program head extracted from the DOH/HARP archive and injected into the monthly Phase 1 rebuild lane.",
        ],
    }
    base_row.update(annotate_latent_indicator_fields(base_row, plugin_id))
    return base_row


def _harp_rows_from_observed_program_panel(
    *,
    source_run_dir: Path,
    plugin_id: str,
    start_month: str,
    end_month: str,
) -> list[dict[str, Any]]:
    payload = dict(read_json(source_run_dir / "harp_archive" / "observed_program_panel.json", default={}))
    program_rows = list(payload.get("rows") or [])
    collected: list[dict[str, Any]] = []
    for item in program_rows:
        if not isinstance(item, dict):
            continue
        canonical_name = str(item.get("metric_name") or "").strip()
        if canonical_name not in HARP_OBSERVED_METRICS:
            continue
        time_value = str(item.get("time") or item.get("period_end") or "").strip()
        if not _within_month_window(time_value, start_month=start_month, end_month=end_month):
            continue
        numeric_value = item.get("value")
        try:
            numeric_float = float(numeric_value)
        except Exception:
            continue
        collected.append(
            _harp_numeric_row(
                canonical_name=canonical_name,
                time_value=time_value,
                numeric_value=numeric_float,
                plugin_id=plugin_id,
                source_bank="phase1_harp_observed_program_panel",
                source_id=str(item.get("source_id") or canonical_name),
                source_title=str(item.get("source_label") or canonical_name),
                source_url=str(item.get("source_url") or ""),
                temporal_precision=str(item.get("temporal_precision") or item.get("series_kind") or "monthly_snapshot"),
                normalized_unit=str(item.get("unit") or "count_people"),
                evidence_confidence=float(item.get("evidence_confidence") or 0.92),
                source_quality_tier=str(item.get("source_quality_tier") or ""),
                source_note=str(item.get("source_note") or ""),
            )
        )
    return _dedupe_best_rows(collected, key_fields=("canonical_name", "time"))


def _harp_rows_from_diagnosis_flow(
    *,
    source_run_dir: Path,
    plugin_id: str,
    start_month: str,
    end_month: str,
) -> list[dict[str, Any]]:
    payload = dict(read_json(source_run_dir / "harp_archive" / "diagnosis_flow_points.json", default={}))
    points = list(payload.get("points") or [])
    collected: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        time_value = str(point.get("effective_month") or point.get("month") or "").strip()
        if not _within_month_window(time_value, start_month=start_month, end_month=end_month):
            continue
        try:
            numeric_float = float(point.get("diagnosed_count"))
        except Exception:
            continue
        collected.append(
            _harp_numeric_row(
                canonical_name=HARP_DIAGNOSIS_FLOW_CANONICAL,
                time_value=time_value,
                numeric_value=numeric_float,
                plugin_id=plugin_id,
                source_bank="phase1_harp_diagnosis_flow",
                source_id=str(point.get("source_id") or HARP_DIAGNOSIS_FLOW_CANONICAL),
                source_title=str(point.get("source_label") or point.get("label") or HARP_DIAGNOSIS_FLOW_CANONICAL),
                source_url=str(point.get("source_url") or ""),
                temporal_precision=str(point.get("temporal_precision") or point.get("series_kind") or "monthly_snapshot"),
                normalized_unit="count_people",
                evidence_confidence=float(point.get("evidence_confidence") or 0.92),
                source_quality_tier="official_doh_archive",
            )
        )
    return _dedupe_best_rows(collected, key_fields=("canonical_name", "time"))


def _harp_rows_from_program_derived(
    *,
    source_run_dir: Path,
    plugin_id: str,
    start_month: str,
    end_month: str,
) -> list[dict[str, Any]]:
    rows = build_archive_derived_indicator_rows(run_dir=source_run_dir, plugin_id=plugin_id)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        time_value = str(row.get("time") or "").strip()
        if not _within_month_window(time_value, start_month=start_month, end_month=end_month):
            continue
        row_copy = dict(row)
        row_copy["province"] = "Philippines"
        row_copy["country"] = "Philippines"
        row_copy["source_reliability_class"] = "official_routine_anchor"
        row_copy["bias_penalty"] = 0.0
        row_copy["replication_weight"] = 1.0
        row_copy["measurement_quality_weight"] = 1.0
        row_copy["temporal_freshness_weight"] = 1.0
        row_copy["spatial_relevance_weight"] = 1.0
        row_copy["evidence_class"] = "observed_numeric"
        filtered.append(row_copy)
    return _dedupe_best_rows(filtered, key_fields=("canonical_name", "time"))


def _harp_rows_from_historical_panel(
    *,
    source_run_dir: Path,
    plugin_id: str,
    start_month: str,
    end_month: str,
) -> list[dict[str, Any]]:
    payload = dict(read_json(source_run_dir / "harp_archive" / "historical_harp_panel.json", default={}))
    panel_rows = list(payload.get("rows") or [])
    collected: list[dict[str, Any]] = []
    for item in panel_rows:
        if not isinstance(item, dict):
            continue
        time_value = str(item.get("time") or "").strip()
        if not _within_month_window(time_value, start_month=start_month, end_month=end_month):
            continue
        for canonical_name in HARP_HISTORICAL_PANEL_METRICS:
            numeric_value = item.get(canonical_name)
            try:
                numeric_float = float(numeric_value)
            except Exception:
                continue
            collected.append(
                _harp_numeric_row(
                    canonical_name=canonical_name,
                    time_value=time_value,
                    numeric_value=numeric_float,
                    plugin_id=plugin_id,
                    source_bank="phase1_harp_historical_panel",
                    source_id="historical_harp_panel",
                    source_title="Historical HARP Panel",
                    source_url="",
                    temporal_precision="annual_snapshot",
                    normalized_unit="count_people",
                    evidence_confidence=0.88,
                    source_quality_tier="official_mirror",
                    source_note="Annual HARP-derived panel anchor carried into the monthly Phase1 rebuild lane.",
                )
            )
    return _dedupe_best_rows(collected, key_fields=("canonical_name", "time"))


def _load_metric_row_list(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default=[])
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [dict(row) for row in list(payload.get("rows") or []) if isinstance(row, dict)]
    return []


def _metric_row_year(row: dict[str, Any]) -> int | None:
    raw_year = row.get("year")
    try:
        return int(raw_year)
    except Exception:
        pass
    time_value = str(row.get("time") or "").strip()
    if MONTH_PATTERN.fullmatch(time_value):
        return int(time_value[:4])
    return None


def _annual_population_lookup(source_run_dir: Path) -> dict[int, float]:
    rows = _load_metric_row_list(source_run_dir / "harp_archive" / "historical_metric_rows.json")
    lookup: dict[int, float] = {}
    for row in rows:
        if str(row.get("metric_name") or "").strip() != "population_total":
            continue
        year = _metric_row_year(row)
        if year is None:
            continue
        try:
            numeric_value = float(row.get("value"))
        except Exception:
            continue
        if numeric_value <= 0.0:
            continue
        lookup[year] = float(numeric_value)
    return lookup


def _harp_rows_from_historical_metrics(
    *,
    source_run_dir: Path,
    plugin_id: str,
    start_month: str,
    end_month: str,
) -> list[dict[str, Any]]:
    multinational_rows = _load_metric_row_list(source_run_dir / "harp_archive" / "multinational_hiv_metric_rows.json")
    historical_rows = _load_metric_row_list(source_run_dir / "harp_archive" / "historical_metric_rows.json")
    population_lookup = _annual_population_lookup(source_run_dir)
    collected: list[dict[str, Any]] = []

    for row in multinational_rows:
        metric_name = str(row.get("metric_name") or "").strip()
        if metric_name not in HARP_ANNUAL_UPSTREAM_COUNT_METRICS:
            continue
        year = _metric_row_year(row)
        if year is None:
            continue
        time_value = str(row.get("time") or f"{year:04d}-12").strip()
        if not _within_month_window(time_value, start_month=start_month, end_month=end_month):
            continue
        denominator = float(population_lookup.get(int(year)) or 0.0)
        if denominator <= 0.0:
            continue
        try:
            numeric_value = float(row.get("value"))
        except Exception:
            continue
        derived_name, normalized_unit = HARP_ANNUAL_UPSTREAM_COUNT_METRICS[metric_name]
        per_100k_value = 100000.0 * float(numeric_value) / denominator
        collected.append(
            _harp_numeric_row(
                canonical_name=derived_name,
                time_value=time_value,
                numeric_value=per_100k_value,
                plugin_id=plugin_id,
                source_bank="phase1_harp_historical_metric_rows",
                source_id=f"{str(row.get('source_id') or metric_name)}_per_100k_population",
                source_title=f"{str(row.get('source_label') or metric_name)} per 100k population",
                source_url=str(row.get("source_url") or ""),
                temporal_precision=str(row.get("temporal_precision") or "annual_series"),
                normalized_unit=normalized_unit,
                evidence_confidence=float(row.get("evidence_confidence") or 0.82),
                source_quality_tier=str(row.get("source_quality_tier") or "external_multinational_hiv_panel"),
                source_note=(
                    f"{str(row.get('source_note') or '').strip()} "
                    "Normalized per 100k population using annual population_total from historical_metric_rows.json."
                ).strip(),
            )
        )

    for row in historical_rows + multinational_rows:
        metric_name = str(row.get("metric_name") or "").strip()
        if metric_name not in HARP_ANNUAL_UPSTREAM_PASS_THROUGH_METRICS:
            continue
        year = _metric_row_year(row)
        if year is None:
            continue
        time_value = str(row.get("time") or f"{year:04d}-12").strip()
        if not _within_month_window(time_value, start_month=start_month, end_month=end_month):
            continue
        try:
            numeric_value = float(row.get("value"))
        except Exception:
            continue
        collected.append(
            _harp_numeric_row(
                canonical_name=metric_name,
                time_value=time_value,
                numeric_value=numeric_value,
                plugin_id=plugin_id,
                source_bank="phase1_harp_historical_metric_rows",
                source_id=str(row.get("source_id") or metric_name),
                source_title=str(row.get("source_label") or metric_name),
                source_url=str(row.get("source_url") or ""),
                temporal_precision=str(row.get("temporal_precision") or "annual_series"),
                normalized_unit=str(row.get("unit") or "annual_value"),
                evidence_confidence=float(row.get("evidence_confidence") or 0.82),
                source_quality_tier=str(row.get("source_quality_tier") or "external_multinational_hiv_panel"),
                source_note=str(row.get("source_note") or "Annual historical metric carried into the monthly Phase1 rebuild lane."),
            )
        )

    return _dedupe_best_rows(collected, key_fields=("canonical_name", "time", "source_id"))


def _harp_monthly_phase1_rows(
    *,
    source_run_dir: Path,
    plugin_id: str,
    start_month: str,
    end_month: str,
) -> list[dict[str, Any]]:
    observed_rows = _harp_rows_from_observed_program_panel(
        source_run_dir=source_run_dir,
        plugin_id=plugin_id,
        start_month=start_month,
        end_month=end_month,
    )
    flow_rows = _harp_rows_from_diagnosis_flow(
        source_run_dir=source_run_dir,
        plugin_id=plugin_id,
        start_month=start_month,
        end_month=end_month,
    )
    derived_rows = _harp_rows_from_program_derived(
        source_run_dir=source_run_dir,
        plugin_id=plugin_id,
        start_month=start_month,
        end_month=end_month,
    )
    historical_panel_rows = _harp_rows_from_historical_panel(
        source_run_dir=source_run_dir,
        plugin_id=plugin_id,
        start_month=start_month,
        end_month=end_month,
    )
    historical_metric_rows = _harp_rows_from_historical_metrics(
        source_run_dir=source_run_dir,
        plugin_id=plugin_id,
        start_month=start_month,
        end_month=end_month,
    )
    combined = observed_rows + flow_rows + derived_rows + historical_panel_rows + historical_metric_rows
    return _dedupe_best_rows(combined, key_fields=("canonical_name", "time", "source_bank"))


def _canonical_axis(source_axis_catalogs: dict[str, Any], monthly_rows: list[dict[str, Any]]) -> list[str]:
    source_axis = [str(value) for value in list(source_axis_catalogs.get("canonical_name") or [])]
    monthly_canonical_names = {str(row.get("canonical_name") or "").strip() for row in monthly_rows if str(row.get("canonical_name") or "").strip()}
    axis = [name for name in source_axis if name in monthly_canonical_names]
    remaining = sorted(name for name in monthly_canonical_names if name not in axis)
    if axis or remaining:
        return axis + remaining
    return sorted(monthly_canonical_names)


def _filtered_parameter_catalog(source_catalog: list[dict[str, Any]], canonical_axis: list[str]) -> list[dict[str, Any]]:
    canonical_set = {str(name) for name in canonical_axis}
    rows = [dict(row) for row in source_catalog if str(row.get("canonical_name") or "") in canonical_set]
    present = {str(row.get("canonical_name") or "") for row in rows}
    fallback: list[dict[str, Any]] = []
    for canonical_name in canonical_axis:
        if canonical_name in present:
            continue
        fallback.append(
            {
                "canonical_name": canonical_name,
                "row_count": 0,
                "numeric_row_count": 0,
                "model_numeric_row_count": 0,
                "source_banks": {},
                "units": {},
                "geo_resolutions": {"national": 0},
                "regions": {"national": 0},
                "time_resolutions": {"monthly": 0},
                "domain_families": {"mixed": 0},
                "pathway_families": {"mixed": 0},
                "soft_ontology_tags": {},
                "linkage_targets": {},
                "evidence_classes": {},
                "candidate_blocks": {},
                "expected_signs": {},
                "measurement_roles": {},
            }
        )
    return rows + fallback


def _dominant_label(payload: dict[str, Any], key: str, default: str) -> str:
    counter = Counter({str(name): int(value) for name, value in dict(payload.get(key) or {}).items() if str(name)})
    return counter.most_common(1)[0][0] if counter else default


def _tensor_rows(
    *,
    aligned_tensor: np.ndarray,
    standardized_tensor: np.ndarray,
    missing_mask: np.ndarray,
    quality_weight_tensor: np.ndarray,
    canonical_axis: list[str],
    month_axis: list[str],
    parameter_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parameter_lookup = {str(row.get("canonical_name") or ""): dict(row) for row in parameter_catalog}
    province = "Philippines"
    rows: list[dict[str, Any]] = []
    for month_idx, month in enumerate(month_axis):
        for canonical_idx, canonical_name in enumerate(canonical_axis):
            rollup = dict(parameter_lookup.get(canonical_name) or {})
            rows.append(
                {
                    "tensor_row_id": f"{province}:{month}:{canonical_name}",
                    "canonical_name": canonical_name,
                    "model_numeric_value": float(standardized_tensor[0, month_idx, canonical_idx]),
                    "raw_numeric_value": float(aligned_tensor[0, month_idx, canonical_idx]),
                    "normalized_unit": "standard_score",
                    "geo_resolution": geo_resolution_label(province),
                    "geo": province,
                    "country": "Philippines",
                    "region": infer_region_code(province),
                    "province": province,
                    "time_resolution": "monthly",
                    "time": month,
                    "year": int(month[:4]),
                    "month": int(month[5:7]),
                    "sex": "",
                    "age_band": "",
                    "kp_group": "remaining_population",
                    "domain_family": _dominant_label(rollup, "domain_families", "mixed"),
                    "pathway_family": _dominant_label(rollup, "pathway_families", "mixed"),
                    "evidence_class": "phase1_standardized_tensor",
                    "evidence_weight": 1.0,
                    "source_bank": "phase1_standardized_tensor",
                    "missing_mask": float(missing_mask[0, month_idx, canonical_idx]),
                    "quality_weight": float(quality_weight_tensor[0, month_idx, canonical_idx]),
                }
            )
    return rows


def _axis_gap_summary(month_axis: list[str]) -> dict[str, Any]:
    if len(month_axis) < 2:
        return {"gap_count": 0, "largest_gap_months": 0, "irregular": False}
    ordinals = np.asarray([_month_ordinal(month) for month in month_axis], dtype=np.int32)
    gaps = np.diff(ordinals)
    return {
        "gap_count": int(np.sum(gaps != 1)),
        "largest_gap_months": int(np.max(gaps)) if gaps.size else 0,
        "irregular": bool(np.any(gaps != 1)),
    }


def _safe_reset_run_dir(run_dir: Path) -> None:
    runs_root = (ROOT_DIR / "artifacts" / "runs").resolve()
    resolved = run_dir.resolve()
    if resolved == runs_root or runs_root not in resolved.parents:
        raise ValueError(f"refusing to reset path outside runs root: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    ensure_dir(run_dir)


def _archive_merge_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("metric_name") or row.get("canonical_name") or "").strip(),
        str(row.get("time") or row.get("year") or "").strip(),
        str(row.get("source_id") or "").strip(),
        str(row.get("subgroup") or row.get("population") or "").strip(),
        str(row.get("sex") or "").strip(),
        str(row.get("age_group") or "").strip(),
    )


def _dedupe_archive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        row_copy = dict(row)
        key = _archive_merge_key(row_copy)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row_copy)
    return deduped


def _copy_harp_archive(
    source_run_dir: Path,
    target_run_dir: Path,
    *,
    coverage_run_dir: Path | None = None,
) -> dict[str, Any]:
    source_dir = source_run_dir / "harp_archive"
    if not source_dir.exists():
        raise FileNotFoundError(f"source HARP archive missing: {source_dir}")
    target_dir = target_run_dir / "harp_archive"
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(source_dir, target_dir)
    summary: dict[str, Any] = {
        "mode": "copy_only",
        "source_run_id": str(source_run_dir.name),
    }
    if coverage_run_dir is None:
        return summary
    coverage_dir = coverage_run_dir / "harp_archive"
    if not coverage_dir.exists():
        raise FileNotFoundError(f"coverage HARP archive missing: {coverage_dir}")

    base_harp_only = _load_metric_row_list(target_dir / "historical_metric_rows_harp_only.json")
    base_historical = _load_metric_row_list(target_dir / "historical_metric_rows.json")
    coverage_multinational = _load_metric_row_list(coverage_dir / "multinational_hiv_metric_rows.json")
    merged_harp_only = _dedupe_archive_rows(base_harp_only + coverage_multinational)
    merged_historical = _dedupe_archive_rows(base_historical + coverage_multinational)
    write_json(target_dir / "multinational_hiv_metric_rows.json", coverage_multinational)
    write_json(target_dir / "historical_metric_rows_harp_only.json", merged_harp_only)
    write_json(target_dir / "historical_metric_rows.json", merged_historical)

    coverage_inventory_json = coverage_dir / "multinational_hiv_series_inventory.json"
    coverage_inventory_csv = coverage_dir / "multinational_hiv_series_inventory.csv"
    if coverage_inventory_json.exists():
        shutil.copy2(coverage_inventory_json, target_dir / "multinational_hiv_series_inventory.json")
    if coverage_inventory_csv.exists():
        shutil.copy2(coverage_inventory_csv, target_dir / "multinational_hiv_series_inventory.csv")

    summary = {
        "mode": "baseline_plus_coverage_multinational_merge",
        "source_run_id": str(source_run_dir.name),
        "coverage_run_id": str(coverage_run_dir.name),
        "base_harp_only_row_count": int(len(base_harp_only)),
        "base_historical_row_count": int(len(base_historical)),
        "coverage_multinational_row_count": int(len(coverage_multinational)),
        "merged_harp_only_row_count": int(len(merged_harp_only)),
        "merged_historical_row_count": int(len(merged_historical)),
        "preserved_observed_program_panel": True,
        "preserved_diagnosis_flow_points": True,
    }
    write_json(target_dir / "harp_archive_merge_summary.json", summary)
    return summary


def _plot_axis_comparison(old_axis: list[str], new_axis: list[str], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    old_ordinals = np.asarray([_month_ordinal(month) for month in old_axis], dtype=np.int32) if old_axis else np.zeros((0,), dtype=np.int32)
    new_ordinals = np.asarray([_month_ordinal(month) for month in new_axis], dtype=np.int32) if new_axis else np.zeros((0,), dtype=np.int32)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    if old_ordinals.size:
        ax.scatter(old_ordinals, np.full_like(old_ordinals, 1.0, dtype=np.float32), s=22, label="legacy phase15 axis", alpha=0.85)
    if new_ordinals.size:
        ax.scatter(new_ordinals, np.full_like(new_ordinals, 0.0, dtype=np.float32), s=18, label="rebuilt monthly axis", alpha=0.85)
    ax.set_yticks([0.0, 1.0], labels=["monthly rebuild", "legacy"])
    ax.set_xlabel("month ordinal")
    ax.set_title("Phase15 month-axis comparison")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_monthly_support(month_axis: list[str], monthly_rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    month_counts = Counter(str(row.get("time") or "") for row in monthly_rows)
    values = np.asarray([int(month_counts.get(month, 0)) for month in month_axis], dtype=np.int32)
    positions = np.arange(len(month_axis), dtype=np.int32)
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(positions, values, color="#1f77b4", linewidth=1.8)
    ax.set_title("Monthly national-row support used for rebuild")
    ax.set_xlabel("month index")
    ax.set_ylabel("row count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_canonical_support(monthly_rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    counts = Counter(str(row.get("canonical_name") or "") for row in monthly_rows if str(row.get("canonical_name") or "").strip())
    labels = [item[0] for item in counts.most_common()]
    values = [item[1] for item in counts.most_common()]
    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.35 * len(labels))))
    ax.barh(labels, values, color="#2ca02c")
    ax.invert_yaxis()
    ax.set_title("Canonical support in rebuilt monthly Phase1 lane")
    ax.set_xlabel("row count")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _apply_structural_canonical_exclusions(
    monthly_rows: list[dict[str, Any]],
    *,
    excluded_canonicals: tuple[str, ...] | list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded = tuple(sorted({str(name).strip() for name in excluded_canonicals if str(name).strip()}))
    if not excluded:
        return list(monthly_rows), {
            "excluded_canonicals": [],
            "full_row_count": int(len(monthly_rows)),
            "structural_row_count": int(len(monthly_rows)),
            "excluded_row_count": 0,
            "excluded_counts_by_canonical": {},
        }
    excluded_set = set(excluded)
    structural_rows: list[dict[str, Any]] = []
    excluded_counts = Counter()
    for row in monthly_rows:
        canonical_name = str(row.get("canonical_name") or "").strip()
        if canonical_name in excluded_set:
            excluded_counts[canonical_name] += 1
            continue
        structural_rows.append(dict(row))
    return structural_rows, {
        "excluded_canonicals": list(excluded),
        "full_row_count": int(len(monthly_rows)),
        "structural_row_count": int(len(structural_rows)),
        "excluded_row_count": int(sum(int(value) for value in excluded_counts.values())),
        "excluded_counts_by_canonical": {str(name): int(value) for name, value in sorted(excluded_counts.items())},
    }


def _write_monthly_phase1(
    *,
    run_dir: Path,
    plugin_id: str,
    source_axis_catalogs: dict[str, Any],
    source_parameter_catalog: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    full_monthly_rows: list[dict[str, Any]] | None = None,
    structural_exclusion_summary: dict[str, Any] | None = None,
    month_axis: list[str],
) -> dict[str, Any]:
    phase1_dir = ensure_dir(run_dir / "phase1")
    canonical_axis = _canonical_axis(source_axis_catalogs, monthly_rows)
    parameter_catalog = _filtered_parameter_catalog(source_parameter_catalog, canonical_axis)
    axis_catalogs = dict(source_axis_catalogs)
    axis_catalogs["province"] = ["Philippines"]
    axis_catalogs["month"] = list(month_axis)
    axis_catalogs["canonical_name"] = list(canonical_axis)
    axis_catalogs["canonical"] = list(canonical_axis)
    axis_catalogs["region"] = ["national"]
    sparse_payload = build_sparse_indicator_cube(
        normalized_rows=monthly_rows,
        province_axis=["Philippines"],
        month_axis=month_axis,
        canonical_names=canonical_axis,
        region_labels=["national"],
        include_national_rows=True,
    )
    aligned_tensor = np.asarray(sparse_payload["raw_cube"], dtype=np.float32)
    missing_mask = _build_missing_mask(
        aligned_tensor=aligned_tensor,
        normalized_rows=monthly_rows,
        province_axis=["Philippines"],
        month_axis=month_axis,
        canonical_axis=canonical_axis,
    )
    denominator_tensor, denominator_map = _build_denominator_tensor(
        aligned_tensor=aligned_tensor,
        normalized_rows=monthly_rows,
        canonical_axis=canonical_axis,
    )
    _, denominator_tensor_out, standardized_tensor, quality_weight_tensor, preprocess_meta = _tensor_preprocess(
        aligned_tensor,
        denominator_tensor=denominator_tensor,
        missing_mask=missing_mask,
    )
    standardized_tensor_np = to_numpy(standardized_tensor)
    denominator_tensor_np = to_numpy(denominator_tensor_out)
    quality_weight_tensor_np = to_numpy(quality_weight_tensor)
    tensor_rows = _tensor_rows(
        aligned_tensor=aligned_tensor,
        standardized_tensor=standardized_tensor_np,
        missing_mask=np.asarray(missing_mask, dtype=np.float32),
        quality_weight_tensor=np.asarray(quality_weight_tensor_np, dtype=np.float32),
        canonical_axis=canonical_axis,
        month_axis=month_axis,
        parameter_catalog=parameter_catalog,
    )
    latent_observability_audit = build_latent_observability_audit(
        normalized_rows=monthly_rows,
        parameter_catalog=parameter_catalog,
        plugin_id=plugin_id,
    )
    direct_contextual_split = build_direct_contextual_split(
        normalized_rows=monthly_rows,
        plugin_id=plugin_id,
    )
    observation_noise_model = build_observation_noise_model(
        normalized_rows=monthly_rows,
        plugin_id=plugin_id,
    )
    standardized_artifact = save_tensor_artifact(
        array=standardized_tensor_np,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="standardized_tensor",
        backend=str(preprocess_meta.get("compute_backend") or "numpy"),
        device=str(preprocess_meta.get("device") or "cpu"),
        notes=["phase1_standardized_tensor", "monthly_harp_2010_plus_rebuild"],
    )
    denominator_artifact = save_tensor_artifact(
        array=denominator_tensor_np,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="denominator_tensor",
        backend=str(preprocess_meta.get("compute_backend") or "numpy"),
        device=str(preprocess_meta.get("device") or "cpu"),
        notes=["phase1_denominator_tensor", "monthly_harp_2010_plus_rebuild"],
    )
    missing_artifact = save_tensor_artifact(
        array=np.asarray(missing_mask, dtype=np.float32),
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="missing_mask",
        backend=str(preprocess_meta.get("compute_backend") or "numpy"),
        device=str(preprocess_meta.get("device") or "cpu"),
        notes=["phase1_missing_mask", "monthly_harp_2010_plus_rebuild"],
    )
    quality_artifact = save_tensor_artifact(
        array=np.asarray(quality_weight_tensor_np, dtype=np.float32),
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="quality_weight_tensor",
        backend=str(preprocess_meta.get("compute_backend") or "numpy"),
        device=str(preprocess_meta.get("device") or "cpu"),
        notes=["phase1_quality_weight_tensor", "monthly_harp_2010_plus_rebuild"],
    )
    write_json(phase1_dir / "normalized_subparameters.json", monthly_rows)
    if full_monthly_rows is not None:
        write_json(phase1_dir / "normalized_subparameters_full_for_evaluation.json", list(full_monthly_rows))
    if structural_exclusion_summary is not None:
        write_json(phase1_dir / "structural_exclusion_summary.json", dict(structural_exclusion_summary))
    write_json(phase1_dir / "parameter_catalog.json", parameter_catalog)
    write_json(phase1_dir / "axis_catalogs.json", axis_catalogs)
    write_json(phase1_dir / "tensor_rows.json", tensor_rows)
    write_json(phase1_dir / "latent_observability_audit.json", latent_observability_audit)
    write_json(phase1_dir / "direct_vs_contextual_split.json", direct_contextual_split)
    write_json(phase1_dir / "observation_noise_model.json", observation_noise_model)
    write_json(phase1_dir / "interop_report.json", dict(preprocess_meta.get("interop") or {}))
    write_json(
        phase1_dir / "tensor_schema.json",
        {
            "axes": axis_catalogs,
            "value_fields": {
                "standardized_tensor": "phase1/standardized_tensor",
                "denominator_tensor": "phase1/denominator_tensor",
                "missing_mask": "phase1/missing_mask",
                "quality_weight_tensor": "phase1/quality_weight_tensor",
                "raw_numeric_value": "raw_numeric_value",
                "model_numeric_value": "model_numeric_value",
            },
            "default_value_field": "standardized_tensor",
        },
    )
    write_json(
        phase1_dir / "normalization_report.json",
        {
            "normalized_row_count": int(len(monthly_rows)),
            "tensor_row_count": int(len(tensor_rows)),
            "month_count": int(len(month_axis)),
            "canonical_count": int(len(canonical_axis)),
            "province_count": 1,
            "missing_mask_fraction": float(1.0 - np.asarray(missing_mask, dtype=np.float32).mean()) if np.asarray(missing_mask).size else 0.0,
            "denominator_map": denominator_map,
            "preprocess_meta": preprocess_meta,
            "source_scope": "national_monthly_harp_2010_plus",
        },
    )
    write_json(
        phase1_dir / "phase1_monthly_rebuild_summary.json",
        {
            "source_scope": "national_monthly_harp_2010_plus",
            "month_start": month_axis[0] if month_axis else "",
            "month_end": month_axis[-1] if month_axis else "",
            "month_count": int(len(month_axis)),
            "canonical_count": int(len(canonical_axis)),
            "monthly_row_count": int(len(monthly_rows)),
            "full_monthly_row_count": int(len(full_monthly_rows or monthly_rows)),
            "structural_exclusion_summary": dict(structural_exclusion_summary or {}),
            "artifacts": {
                "standardized_tensor": standardized_artifact["value_path"],
                "denominator_tensor": denominator_artifact["value_path"],
                "missing_mask": missing_artifact["value_path"],
                "quality_weight_tensor": quality_artifact["value_path"],
            },
        },
    )
    return {
        "month_axis": list(month_axis),
        "canonical_axis": list(canonical_axis),
        "month_count": int(len(month_axis)),
        "canonical_count": int(len(canonical_axis)),
        "monthly_row_count": int(len(monthly_rows)),
        "full_monthly_row_count": int(len(full_monthly_rows or monthly_rows)),
        "structural_exclusion_summary": dict(structural_exclusion_summary or {}),
    }


def _report_markdown(payload: dict[str, Any]) -> str:
    source_gap = dict(payload.get("legacy_phase15_gap_summary") or {})
    rebuilt_gap = dict(payload.get("rebuilt_phase15_gap_summary") or {})
    phase2_summary = dict(payload.get("rebuilt_phase2_summary") or {})
    merge_summary = dict(payload.get("harp_archive_merge_summary") or {})
    lines = [
        "# Monthly Phase1/Phase15 Rebuild Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Coverage run: `{payload.get('coverage_run_id') or 'none'}`",
        f"- Rebuilt run: `{payload['run_id']}`",
        f"- Start month: `{payload['start_month']}`",
        f"- HARP month range: `{payload['harp_month_axis'][0]}` to `{payload['harp_month_axis'][-1]}`",
        f"- HARP month count: `{len(payload['harp_month_axis'])}`",
        f"- Filtered monthly national rows: `{payload['monthly_row_count']}`",
        f"- Structural monthly rows used for Phase15/Phase2: `{payload.get('structural_monthly_row_count', payload['monthly_row_count'])}`",
        f"- Injected HARP Phase1 rows: `{payload.get('harp_phase1_row_count', 0)}`",
        f"- Historical HARP panel rows: `{payload.get('harp_historical_panel_row_count', 0)}`",
        f"- Historical HARP metric rows: `{payload.get('harp_historical_metric_row_count', 0)}`",
        f"- Rebuilt canonical count: `{payload.get('phase1_summary', {}).get('canonical_count', 0)}`",
        f"- Structural exclusions: `{', '.join(list(payload.get('structural_exclusion_summary', {}).get('excluded_canonicals') or [])) or 'none'}`",
        "",
        "## HARP Archive Merge",
        "",
        f"- Merge mode: `{merge_summary.get('mode', 'copy_only')}`",
        f"- Coverage multinational rows: `{merge_summary.get('coverage_multinational_row_count', 0)}`",
        f"- Merged HARP-only historical rows: `{merge_summary.get('merged_harp_only_row_count', payload.get('harp_historical_metric_row_count', 0))}`",
        f"- Preserved observed program panel: `{merge_summary.get('preserved_observed_program_panel', False)}`",
        f"- Preserved diagnosis flow points: `{merge_summary.get('preserved_diagnosis_flow_points', False)}`",
        "",
        "## Axis Diagnosis",
        "",
        f"- Legacy Phase15 month count: `{len(payload['legacy_phase15_month_axis'])}`",
        f"- Legacy gap count: `{source_gap.get('gap_count')}`",
        f"- Legacy largest gap (months): `{source_gap.get('largest_gap_months')}`",
        f"- Rebuilt Phase15 month count: `{len(payload['rebuilt_phase15_month_axis'])}`",
        f"- Rebuilt gap count: `{rebuilt_gap.get('gap_count')}`",
        f"- Rebuilt largest gap (months): `{rebuilt_gap.get('largest_gap_months')}`",
        "",
        "## Rebuilt Structural Outputs",
        "",
        f"- Phase15 national state tensor shape: `{payload['rebuilt_phase15_shape']}`",
        f"- Phase2 structural block count: `{phase2_summary.get('block_count')}`",
        f"- Phase2 direct temporal edges: `{phase2_summary.get('direct_temporal_edge_count')}`",
        f"- Phase2 hidden temporal edges: `{phase2_summary.get('hidden_temporal_edge_count')}`",
        f"- Phase2 hidden driver rows: `{phase2_summary.get('hidden_driver_row_count')}`",
        f"- Phase2 month axis count: `{phase2_summary.get('month_count')}`",
        "",
        "## Artifacts",
        "",
        "- `analysis/month_axis_comparison.png`",
        "- `analysis/monthly_support_counts.png`",
        "- `analysis/canonical_support_counts.png`",
        "- `phase1/phase1_monthly_rebuild_summary.json`",
        "- `phase15/phase15_v2_uncertainty.json`",
        "- `phase2/phase2_structural_payload.json`",
        "",
    ]
    return "\n".join(lines)


def run_tr_v3_monthly_phase2_lane_batch(
    *,
    run_id: str,
    source_run_id: str = DEFAULT_SOURCE_RUN_ID,
    coverage_run_id: str | None = None,
    plugin_id: str = "hiv",
    start_month: str = DEFAULT_START_MONTH,
    structural_excluded_canonicals: tuple[str, ...] | list[str] = DEFAULT_STRUCTURAL_EXCLUDED_CANONICALS,
) -> dict[str, Any]:
    source_run_dir = ROOT_DIR / "artifacts" / "runs" / str(source_run_id)
    if not source_run_dir.exists():
        raise FileNotFoundError(f"source run does not exist: {source_run_dir}")
    coverage_run_dir: Path | None = None
    if str(coverage_run_id or "").strip():
        coverage_run_dir = ROOT_DIR / "artifacts" / "runs" / str(coverage_run_id)
        if not coverage_run_dir.exists():
            raise FileNotFoundError(f"coverage run does not exist: {coverage_run_dir}")
    target_run_dir = ROOT_DIR / "artifacts" / "runs" / str(run_id)
    _safe_reset_run_dir(target_run_dir)
    merge_summary = _copy_harp_archive(source_run_dir, target_run_dir, coverage_run_dir=coverage_run_dir)

    source_phase1_dir = source_run_dir / "phase1"
    source_rows = list(read_json(source_phase1_dir / "normalized_subparameters.json", default=[]))
    source_axis_catalogs = dict(read_json(source_phase1_dir / "axis_catalogs.json", default={}))
    source_parameter_catalog = list(read_json(source_phase1_dir / "parameter_catalog.json", default=[]))
    harp_month_axis = _harp_month_axis(source_run_dir, start_month=start_month)
    monthly_rows = _filtered_monthly_rows(source_rows, start_month=harp_month_axis[0], end_month=harp_month_axis[-1])
    harp_phase1_rows = _harp_monthly_phase1_rows(
        source_run_dir=source_run_dir,
        plugin_id=plugin_id,
        start_month=harp_month_axis[0],
        end_month=harp_month_axis[-1],
    )
    historical_panel_rows = _harp_rows_from_historical_panel(
        source_run_dir=source_run_dir,
        plugin_id=plugin_id,
        start_month=harp_month_axis[0],
        end_month=harp_month_axis[-1],
    )
    historical_metric_rows = _harp_rows_from_historical_metrics(
        source_run_dir=source_run_dir,
        plugin_id=plugin_id,
        start_month=harp_month_axis[0],
        end_month=harp_month_axis[-1],
    )
    full_monthly_rows = monthly_rows + harp_phase1_rows
    if not full_monthly_rows:
        raise ValueError("no national monthly phase1 rows survived the rebuild filter")
    structural_rows, structural_exclusion_summary = _apply_structural_canonical_exclusions(
        full_monthly_rows,
        excluded_canonicals=structural_excluded_canonicals,
    )
    if not structural_rows:
        raise ValueError("no structural monthly phase1 rows survived the requested canonical exclusions")
    phase1_summary = _write_monthly_phase1(
        run_dir=target_run_dir,
        plugin_id=plugin_id,
        source_axis_catalogs=source_axis_catalogs,
        source_parameter_catalog=source_parameter_catalog,
        monthly_rows=structural_rows,
        full_monthly_rows=full_monthly_rows,
        structural_exclusion_summary=structural_exclusion_summary,
        month_axis=harp_month_axis,
    )

    run_phase15_build(run_id=str(run_id), plugin_id=plugin_id, reuse_existing_harp_archive=True)
    run_phase2_build(run_id=str(run_id), plugin_id=plugin_id)

    analysis_dir = ensure_dir(target_run_dir / "analysis")
    legacy_phase15_month_axis = list(read_json(source_run_dir / "phase15" / "phase15_v2_uncertainty.json", default={}).get("month_axis") or [])
    rebuilt_phase15_uncertainty = dict(read_json(target_run_dir / "phase15" / "phase15_v2_uncertainty.json", default={}))
    rebuilt_phase15_month_axis = list(rebuilt_phase15_uncertainty.get("month_axis") or [])
    rebuilt_phase15_shape = list(read_json(target_run_dir / "phase15" / "phase15_v2_national_state_tensor.summary.json", default={}).get("shape") or [])
    rebuilt_phase2_structural = dict(read_json(target_run_dir / "phase2" / "phase2_structural_payload.json", default={}))
    hidden_temporal_rows = list(rebuilt_phase2_structural.get("hidden_temporal_edge_rows") or [])
    direct_temporal_rows = list(rebuilt_phase2_structural.get("direct_temporal_edge_rows") or [])
    hidden_driver_rows = list(rebuilt_phase2_structural.get("hidden_driver_rows") or [])
    _plot_axis_comparison(legacy_phase15_month_axis, rebuilt_phase15_month_axis, analysis_dir / "month_axis_comparison.png")
    _plot_monthly_support(harp_month_axis, structural_rows, analysis_dir / "monthly_support_counts.png")
    _plot_canonical_support(structural_rows, analysis_dir / "canonical_support_counts.png")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "source_run_id": str(source_run_id),
        "coverage_run_id": str(coverage_run_id or ""),
        "start_month": str(start_month),
        "harp_month_axis": harp_month_axis,
        "monthly_row_count": int(len(full_monthly_rows)),
        "structural_monthly_row_count": int(len(structural_rows)),
        "harp_phase1_row_count": int(len(harp_phase1_rows)),
        "harp_historical_panel_row_count": int(len(historical_panel_rows)),
        "harp_historical_metric_row_count": int(len(historical_metric_rows)),
        "harp_archive_merge_summary": merge_summary,
        "structural_exclusion_summary": structural_exclusion_summary,
        "legacy_phase15_month_axis": legacy_phase15_month_axis,
        "legacy_phase15_gap_summary": _axis_gap_summary(legacy_phase15_month_axis),
        "rebuilt_phase15_month_axis": rebuilt_phase15_month_axis,
        "rebuilt_phase15_gap_summary": _axis_gap_summary(rebuilt_phase15_month_axis),
        "rebuilt_phase15_shape": rebuilt_phase15_shape,
        "rebuilt_phase2_summary": {
            "month_count": int(len(list(rebuilt_phase2_structural.get("month_axis") or rebuilt_phase15_month_axis))),
            "block_count": int(len(list(rebuilt_phase2_structural.get("block_axis") or []))),
            "direct_temporal_edge_count": int(len(direct_temporal_rows)),
            "hidden_temporal_edge_count": int(len(hidden_temporal_rows)),
            "hidden_driver_row_count": int(len(hidden_driver_rows)),
        },
        "phase1_summary": phase1_summary,
        "artifacts": {
            "axis_plot": str(analysis_dir / "month_axis_comparison.png"),
            "support_plot": str(analysis_dir / "monthly_support_counts.png"),
            "canonical_support_plot": str(analysis_dir / "canonical_support_counts.png"),
            "phase1_summary": str(target_run_dir / "phase1" / "phase1_monthly_rebuild_summary.json"),
            "phase15_uncertainty": str(target_run_dir / "phase15" / "phase15_v2_uncertainty.json"),
            "phase2_structural_payload": str(target_run_dir / "phase2" / "phase2_structural_payload.json"),
        },
    }
    write_json(analysis_dir / "tr_v3_monthly_phase2_lane_batch_report.json", payload)
    (analysis_dir / "tr_v3_monthly_phase2_lane_batch_report.md").write_text(_report_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a true monthly national Phase1/Phase15 lane from HARP 2010+ and fit the real Phase2 structural model on it.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--coverage-run-id", default="")
    parser.add_argument("--plugin", default="hiv")
    parser.add_argument("--start-month", default=DEFAULT_START_MONTH)
    parser.add_argument("--structural-exclude-canonical", action="append", default=[])
    args = parser.parse_args()
    run_tr_v3_monthly_phase2_lane_batch(
        run_id=str(args.run_id),
        source_run_id=str(args.source_run_id),
        coverage_run_id=str(args.coverage_run_id or ""),
        plugin_id=str(args.plugin),
        start_month=str(args.start_month),
        structural_excluded_canonicals=tuple(str(value) for value in list(args.structural_exclude_canonical or [])),
    )


if __name__ == "__main__":
    main()
