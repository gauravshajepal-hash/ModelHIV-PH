from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from epigraph_ph.geography import infer_region_code, normalize_geo_label
from epigraph_ph.latent_blocks import annotate_latent_indicator_fields
from epigraph_ph.runtime import read_json


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_region_label(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return "unknown"
    if token.lower() in {"national", "philippines"}:
        return "national"
    inferred = infer_region_code(token)
    if inferred:
        return inferred
    return re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_") or "unknown"


def province_lookup_tokens(province_axis: list[str]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, province in enumerate(province_axis):
        normalized = normalize_geo_label(str(province), default_country_focus=True)
        lookup[normalized.lower()] = idx
        lookup[str(province).strip().lower()] = idx
    return lookup


def province_region_codes(province_axis: list[str], region_labels: list[str] | None = None) -> list[str]:
    rows: list[str] = []
    fallback_labels = list(region_labels or [])
    for idx, province in enumerate(province_axis):
        inferred = infer_region_code(str(province), str(province))
        if inferred:
            rows.append(inferred)
            continue
        fallback = fallback_labels[idx] if idx < len(fallback_labels) else ""
        rows.append(normalize_region_label(str(fallback)))
    return rows


def province_region_members(region_codes: list[str]) -> dict[str, list[int]]:
    members: dict[str, list[int]] = defaultdict(list)
    for province_idx, label in enumerate(region_codes):
        members[normalize_region_label(label)].append(province_idx)
    return dict(members)


def month_slots_for_row(row: Mapping[str, Any], month_axis: list[str]) -> list[int]:
    time_value = str(row.get("time") or row.get("effective_month") or "").strip()
    if not time_value:
        return []
    if re.fullmatch(r"\d{4}-\d{2}", time_value):
        try:
            return [month_axis.index(time_value)]
        except ValueError:
            return []
    if re.fullmatch(r"\d{4}", time_value):
        return [idx for idx, month in enumerate(month_axis) if str(month).startswith(f"{time_value}-")]
    return []


def row_weight(row: Mapping[str, Any], *, floor: float) -> float:
    learned_weight = safe_float(row.get("observation_weight"), default=float("nan"))
    evidence_weight = safe_float(row.get("evidence_weight"), default=1.0)
    quality_weight = safe_float(row.get("quality_weight"), default=1.0)
    if quality_weight <= 0.0:
        quality_weight = 1.0
    bias_penalty = safe_float(row.get("bias_penalty"), default=0.0)
    reliability = max(0.25, 1.0 - max(0.0, bias_penalty))
    base_weight = learned_weight if np.isfinite(learned_weight) and learned_weight > 0.0 else evidence_weight * reliability
    return max(float(floor), base_weight * quality_weight)


def target_province_indices(
    row: Mapping[str, Any],
    *,
    province_lookup: Mapping[str, int],
    region_members: Mapping[str, list[int]],
    include_national_rows: bool,
) -> tuple[list[int], str]:
    geo_resolution = str(row.get("geo_resolution") or "").strip().lower()
    province_value = str(row.get("province") or row.get("geo") or "").strip()
    region_value = str(row.get("region") or "").strip()
    if geo_resolution in {"province", "city"} or province_value:
        normalized = normalize_geo_label(province_value, default_country_focus=True).lower()
        if normalized in province_lookup:
            return [int(province_lookup[normalized])], "province"
        fallback = province_value.lower()
        if fallback in province_lookup:
            return [int(province_lookup[fallback])], "province"
    if geo_resolution == "region" or region_value:
        token = normalize_region_label(region_value or province_value)
        if token in region_members:
            return [int(idx) for idx in region_members[token]], "region"
    if include_national_rows:
        geo_value = normalize_geo_label(str(row.get("geo") or ""), default_country_focus=True)
        if geo_resolution == "national" or str(row.get("region") or "").strip().lower() == "national" or geo_value == "Philippines":
            ordered = sorted(int(idx) for indices in region_members.values() for idx in indices)
            return ordered, "national"
    return [], "unsupported"


def build_sparse_indicator_cube(
    *,
    normalized_rows: list[dict[str, Any]],
    province_axis: list[str],
    month_axis: list[str],
    canonical_names: list[str],
    region_labels: list[str] | None = None,
    include_national_rows: bool = True,
    observation_weight_floor: float = 0.25,
) -> dict[str, Any]:
    canonical_axis = [str(name) for name in canonical_names]
    canonical_index = {name: idx for idx, name in enumerate(canonical_axis)}
    province_lookup = province_lookup_tokens(province_axis)
    region_codes = province_region_codes(province_axis, region_labels)
    region_members = province_region_members(region_codes)
    province_count = len(province_axis)
    month_count = len(month_axis)
    canonical_count = len(canonical_axis)
    raw_sum = np.zeros((province_count, month_count, canonical_count), dtype=np.float32)
    weight_sum = np.zeros((province_count, month_count, canonical_count), dtype=np.float32)

    for row in normalized_rows:
        measurement_role = str(row.get("measurement_role") or "")
        if measurement_role == "context_only":
            continue
        canonical_name = str(row.get("canonical_name") or "")
        canonical_idx = canonical_index.get(canonical_name)
        if canonical_idx is None:
            continue
        value = row.get("model_numeric_value")
        if value in (None, ""):
            value = row.get("value")
        numeric_value = safe_float(value, default=float("nan"))
        if not np.isfinite(numeric_value):
            continue
        month_slots = month_slots_for_row(row, month_axis)
        if not month_slots:
            continue
        province_indices, _scope = target_province_indices(
            row,
            province_lookup=province_lookup,
            region_members=region_members,
            include_national_rows=include_national_rows,
        )
        if not province_indices:
            continue
        base_weight = row_weight(row, floor=observation_weight_floor)
        cell_weight = float(base_weight) / float(max(len(province_indices) * len(month_slots), 1))
        for province_idx in province_indices:
            for month_idx in month_slots:
                raw_sum[province_idx, month_idx, canonical_idx] += float(cell_weight) * float(numeric_value)
                weight_sum[province_idx, month_idx, canonical_idx] += float(cell_weight)

    observed_mask = weight_sum > 0.0
    raw_cube = np.zeros_like(raw_sum)
    raw_cube[observed_mask] = raw_sum[observed_mask] / np.clip(weight_sum[observed_mask], 1e-6, None)
    standardized_cube = np.zeros_like(raw_cube)
    for canonical_idx in range(canonical_count):
        mask = observed_mask[:, :, canonical_idx]
        if not np.any(mask):
            continue
        values = raw_cube[:, :, canonical_idx][mask]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        if std_value > 1e-8:
            standardized_cube[:, :, canonical_idx][mask] = (raw_cube[:, :, canonical_idx][mask] - mean_value) / std_value
        else:
            standardized_cube[:, :, canonical_idx][mask] = raw_cube[:, :, canonical_idx][mask] - mean_value
    return {
        "canonical_axis": canonical_axis,
        "region_codes": region_codes,
        "raw_cube": raw_cube,
        "weight_cube": weight_sum,
        "standardized_cube": standardized_cube,
        "observed_mask": observed_mask.astype(np.float32),
    }


def build_archive_derived_indicator_rows(*, run_dir: Path, plugin_id: str) -> list[dict[str, Any]]:
    archive_points_path = Path(run_dir) / "harp_archive" / "harp_program_points.json"
    payload = read_json(archive_points_path, default={})
    points = list(dict(payload).get("points") or [])
    rows: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        effective_month = str(point.get("effective_month") or point.get("month") or "").strip()
        if not effective_month:
            continue
        temporal_precision = str(point.get("temporal_precision") or "monthly_snapshot")
        time_resolution = "annual" if "annual" in temporal_precision else "monthly"
        diagnosed = safe_float(point.get("diagnosed"), default=0.0)
        on_art = safe_float(point.get("on_art"), default=0.0)
        viral_load_tested = safe_float(point.get("viral_load_tested"), default=0.0)
        suppressed = safe_float(point.get("suppressed"), default=0.0)
        derived_specs: list[tuple[str, float | None]] = [
            ("art_uptake_rate", (on_art / diagnosed) if diagnosed > 0.0 else None),
            ("retention_adherence", (viral_load_tested / on_art) if on_art > 0.0 else None),
            ("suppression_outcomes", (suppressed / on_art) if on_art > 0.0 else None),
            ("viral_suppression_rate", (suppressed / viral_load_tested) if viral_load_tested > 0.0 else None),
        ]
        for canonical_name, numeric_value in derived_specs:
            if numeric_value is None or not np.isfinite(float(numeric_value)):
                continue
            base_row = {
                "canonical_name": canonical_name,
                "source_bank": "phase15_harp_derived_indicators",
                "source_id": str(point.get("label") or canonical_name),
                "source_title": str(point.get("label") or canonical_name),
                "source_url": str(point.get("source_url") or ""),
                "geo": "Philippines",
                "province": "",
                "region": "national",
                "geo_resolution": "national",
                "time": effective_month,
                "time_resolution": time_resolution,
                "temporal_precision": temporal_precision,
                "measurement_role": "direct_indicator",
                "model_numeric_value": round(float(numeric_value), 6),
                "raw_numeric_value": round(float(numeric_value), 6),
                "evidence_weight": 1.0,
                "quality_weight": 1.0,
                "is_direct_measurement": True,
                "is_anchor_eligible": True,
                "literature_basis": [
                    "Derived from existing DOH/HARP care continuum counts already present in the local archive build.",
                ],
            }
            base_row.update(annotate_latent_indicator_fields(base_row, plugin_id))
            rows.append(base_row)
    return rows
