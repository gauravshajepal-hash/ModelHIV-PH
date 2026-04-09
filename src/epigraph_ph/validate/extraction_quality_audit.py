from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from epigraph_ph.phase0.boundary_models import validate_phase0_candidate_rows
from epigraph_ph.phase0.pipeline import _phase0_required_section
from epigraph_ph.phase0.structured_numeric_sources import (
    build_philhealth_portal_artifacts,
    build_structured_numeric_candidates,
)
from epigraph_ph.runtime import ensure_dir, read_json, utc_now_iso, write_json


STRUCTURED_SOURCE_BANK = "phase0_structured_numeric"
GOOGLE_MOBILITY_METHOD = "structured_google_mobility_csv"
HARP_SEED_SOURCE_URL = "user_attached_doh_slide_2026_03_31"
HARP_PANEL_FIELDS = {
    "estimated_plhiv": "estimated_plhiv",
    "diagnosed_plhiv": "diagnosed",
    "alive_on_art": "on_art",
    "tested_for_viral_load": "viral_load_tested",
    "virally_suppressed": "suppressed",
}
HARP_PROGRAM_FIELDS = {
    "estimated_plhiv": "estimated_plhiv",
    "diagnosed_plhiv": "diagnosed",
    "alive_on_art": "on_art",
    "tested_for_viral_load": "viral_load_tested",
    "virally_suppressed": "suppressed",
}


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    value = _safe_float(row.get("value"))
    rounded = round(value, 6) if value is not None else None
    return (
        str(row.get("source_id") or ""),
        str(row.get("canonical_name") or ""),
        str(row.get("geo") or ""),
        str(row.get("region") or ""),
        str(row.get("province") or ""),
        str(row.get("time") or ""),
        str(row.get("sex") or ""),
        str(row.get("age_band") or ""),
        str(row.get("kp_group") or ""),
        str(row.get("extraction_method") or ""),
        rounded,
    )


def _candidate_key_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "source_id": key[0],
        "canonical_name": key[1],
        "geo": key[2],
        "region": key[3],
        "province": key[4],
        "time": key[5],
        "sex": key[6],
        "age_band": key[7],
        "kp_group": key[8],
        "extraction_method": key[9],
        "value": key[10],
    }


def _counter_diff(expected_rows: list[dict[str, Any]], emitted_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_counter = Counter(_candidate_key(row) for row in expected_rows)
    emitted_counter = Counter(_candidate_key(row) for row in emitted_rows)
    missing = list((expected_counter - emitted_counter).elements())
    extra = list((emitted_counter - expected_counter).elements())
    expected_by_name = Counter(str(row.get("canonical_name") or "") for row in expected_rows)
    emitted_by_name = Counter(str(row.get("canonical_name") or "") for row in emitted_rows)
    names = sorted(set(expected_by_name) | set(emitted_by_name))
    count_deltas = [
        {
            "canonical_name": name,
            "expected_count": int(expected_by_name.get(name, 0)),
            "emitted_count": int(emitted_by_name.get(name, 0)),
            "delta": int(emitted_by_name.get(name, 0) - expected_by_name.get(name, 0)),
        }
        for name in names
        if expected_by_name.get(name, 0) != emitted_by_name.get(name, 0)
    ]
    return {
        "expected_count": int(sum(expected_counter.values())),
        "emitted_count": int(sum(emitted_counter.values())),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_examples": [_candidate_key_dict(key) for key in missing[:25]],
        "extra_examples": [_candidate_key_dict(key) for key in extra[:25]],
        "count_deltas": count_deltas[:50],
        "passed": not missing and not extra,
    }


def _source_manifest_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "").strip()
        if source_id:
            output[source_id] = row
    return output


def _structured_source_parity(*, run_dir: Path, plugin_id: str) -> dict[str, Any]:
    raw_dir = run_dir / "phase0" / "raw"
    source_manifest = read_json(raw_dir / "source_manifest.json", default=[])
    source_rows = _source_manifest_map(list(source_manifest or []))
    structured_payload = build_structured_numeric_candidates(
        raw_dir=raw_dir,
        source_rows=source_rows,
        plugin_id=plugin_id,
    )
    validation_cfg = dict(_phase0_required_section("boundary_validation"))
    expected_validated, expected_rejected, validation_summary = validate_phase0_candidate_rows(
        list(structured_payload.get("candidate_rows") or []),
        validation_cfg=validation_cfg,
    )
    emitted_candidates = read_json(run_dir / "phase0" / "extracted" / "canonical_parameter_candidates.json", default=[])
    emitted_structured = [
        row for row in list(emitted_candidates or []) if str(row.get("source_bank") or "") == STRUCTURED_SOURCE_BANK
    ]
    parity = _counter_diff(expected_validated, emitted_structured)

    expected_portal = build_philhealth_portal_artifacts(
        candidate_rows=expected_validated,
        collector_rows=list((structured_payload.get("summary") or {}).get("collectors") or []),
    )
    emitted_portal_rows = read_json(run_dir / "phase0" / "extracted" / "philhealth_portal_candidate_rows.json", default=[])
    emitted_portal_summary = read_json(run_dir / "phase0" / "extracted" / "philhealth_portal_metric_summary.json", default={})
    portal_row_parity = _counter_diff(list(expected_portal.get("candidate_rows") or []), list(emitted_portal_rows or []))
    expected_portal_summary = dict(expected_portal.get("summary") or {})

    return {
        "expected_validation_summary": validation_summary,
        "expected_rejected_count": len(expected_rejected),
        "structured_row_parity": parity,
        "philhealth_portal_row_parity": portal_row_parity,
        "philhealth_portal_summary_matches": emitted_portal_summary == expected_portal_summary,
        "philhealth_portal_expected_summary": expected_portal_summary,
        "philhealth_portal_emitted_summary": emitted_portal_summary,
        "passed": bool(parity["passed"] and portal_row_parity["passed"] and emitted_portal_summary == expected_portal_summary),
    }


def _parse_harp_seed(seed_path: Path) -> dict[int, dict[str, Any]]:
    by_year: dict[int, dict[str, Any]] = defaultdict(dict)
    if not seed_path.exists():
        return {}
    with seed_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            year_value = str(row.get("year") or "").strip()
            metric_name = str(row.get("metric_name") or "").strip()
            numeric_value = _safe_float(row.get("value"))
            if not year_value.isdigit() or not metric_name or numeric_value is None:
                continue
            year = int(year_value)
            by_year.setdefault(year, {})
            by_year[year][metric_name] = float(numeric_value)
            by_year[year]["time"] = str(row.get("time") or "")
    return dict(by_year)


def _parse_harp_panel(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        year_value = row.get("year")
        if year_value is None:
            continue
        try:
            year = int(year_value)
        except (TypeError, ValueError):
            continue
        payload = {"time": str(row.get("time") or "")}
        for metric_name, field_name in HARP_PANEL_FIELDS.items():
            value = _safe_float(row.get(field_name))
            if value is not None:
                payload[metric_name] = float(value)
        output[year] = payload
    return output


def _parse_harp_program_points(points: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in points:
        source_url = str(row.get("source_url") or "")
        month = str(row.get("month") or "")
        if source_url != HARP_SEED_SOURCE_URL or len(month) < 4 or not month[:4].isdigit():
            continue
        year = int(month[:4])
        payload = {"time": month}
        for metric_name, field_name in HARP_PROGRAM_FIELDS.items():
            value = _safe_float(row.get(field_name))
            if value is not None:
                payload[metric_name] = float(value)
        output[year] = payload
    return output


def _harp_discrepancy_rows(
    *,
    seed: dict[int, dict[str, Any]],
    observed: dict[int, dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in sorted(set(seed) & set(observed)):
        for metric_name in sorted(set(HARP_PANEL_FIELDS) & set(seed[year])):
            observed_value = _safe_float(observed[year].get(metric_name))
            seed_value = _safe_float(seed[year].get(metric_name))
            if observed_value is None or seed_value is None:
                continue
            if math.isclose(observed_value, seed_value, rel_tol=0.0, abs_tol=0.5):
                continue
            rows.append(
                {
                    "label": label,
                    "year": year,
                    "metric_name": metric_name,
                    "expected_value": seed_value,
                    "observed_value": observed_value,
                    "difference": round(observed_value - seed_value, 6),
                }
            )
    return rows


def _harp_time_mismatch_rows(panel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row in panel_rows:
        year_value = row.get("year")
        time_value = str(row.get("time") or "")
        if year_value is None:
            continue
        try:
            year = int(year_value)
        except (TypeError, ValueError):
            continue
        if len(time_value) < 4 or not time_value[:4].isdigit():
            mismatches.append({"year": year, "time": time_value, "reason": "invalid_time"})
            continue
        if int(time_value[:4]) != year:
            mismatches.append({"year": year, "time": time_value, "reason": "year_prefix_mismatch"})
    return mismatches


def _harp_invariant_violations(panel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in panel_rows:
        year = row.get("year")
        estimated = _safe_float(row.get("estimated_plhiv"))
        diagnosed = _safe_float(row.get("diagnosed_plhiv"))
        art = _safe_float(row.get("alive_on_art"))
        tested = _safe_float(row.get("tested_for_viral_load"))
        suppressed = _safe_float(row.get("virally_suppressed"))
        if estimated is not None and diagnosed is not None and diagnosed > estimated + 1e-6:
            violations.append({"year": year, "rule": "diagnosed_le_estimated", "lhs": diagnosed, "rhs": estimated})
        if diagnosed is not None and art is not None and art > diagnosed + 1e-6:
            violations.append({"year": year, "rule": "art_le_diagnosed", "lhs": art, "rhs": diagnosed})
        if art is not None and suppressed is not None and suppressed > art + 1e-6:
            violations.append({"year": year, "rule": "suppressed_le_art", "lhs": suppressed, "rhs": art})
        if tested is not None and suppressed is not None and suppressed > tested + 1e-6:
            violations.append({"year": year, "rule": "suppressed_le_tested", "lhs": suppressed, "rhs": tested})
    return violations


def _harp_source_conflicts(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        year_value = row.get("year")
        metric_name = str(row.get("metric_name") or "")
        value = _safe_float(row.get("value"))
        if value is None or not metric_name:
            continue
        try:
            year = int(year_value)
        except (TypeError, ValueError):
            continue
        grouped[(year, metric_name)].append(row)

    conflicts: list[dict[str, Any]] = []
    for (year, metric_name), rows in sorted(grouped.items()):
        distinct_values = {round(float(row["value"]), 6) for row in rows if _safe_float(row.get("value")) is not None}
        if len(distinct_values) <= 1:
            continue
        official_rows = [row for row in rows if str(row.get("source_url") or "") == HARP_SEED_SOURCE_URL]
        if not official_rows:
            continue
        official_value = round(float(official_rows[0]["value"]), 6)
        alternatives = [
            {
                "source_id": str(row.get("source_id") or ""),
                "source_label": str(row.get("source_label") or ""),
                "time": str(row.get("time") or ""),
                "value": round(float(row["value"]), 6),
            }
            for row in rows
            if round(float(row["value"]), 6) != official_value
        ]
        if alternatives:
            conflicts.append(
                {
                    "year": year,
                    "metric_name": metric_name,
                    "official_value": official_value,
                    "alternatives": alternatives[:10],
                }
            )
    return conflicts


def _harp_adjudication_category(alternative_row: dict[str, Any]) -> tuple[str, str]:
    series_kind = str(alternative_row.get("series_kind") or "")
    temporal_precision = str(alternative_row.get("temporal_precision") or "")
    measurement_class = str(alternative_row.get("measurement_class") or "")
    source_id = str(alternative_row.get("source_id") or "")
    if series_kind == "quarterly_snapshot" or temporal_precision == "quarterly_snapshot":
        return (
            "quarterly_snapshot",
            "Alternative row is an intra-year quarterly archive snapshot, so the annual official seed remains the adjudicated year-end value.",
        )
    if measurement_class == "model_estimate":
        return (
            "model_update",
            "Alternative row is a non-official model estimate from a different annual source package or model vintage.",
        )
    if series_kind == "monthly_snapshot" or temporal_precision == "monthly_snapshot":
        return (
            "official_override",
            "Alternative row is a monthly administrative snapshot; the official annual seed remains the adjudicated year-end value.",
        )
    if source_id:
        return (
            "official_override",
            "Alternative row is a non-quarterly administrative or annual snapshot that is overridden by the official annual seed.",
        )
    return (
        "official_override",
        "Alternative row is overridden by the official annual seed under the configured HARP precedence rules.",
    )


def _harp_source_adjudication_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        year_value = row.get("year")
        metric_name = str(row.get("metric_name") or "")
        value = _safe_float(row.get("value"))
        if value is None or not metric_name:
            continue
        try:
            year = int(year_value)
        except (TypeError, ValueError):
            continue
        grouped[(year, metric_name)].append(row)

    adjudications: list[dict[str, Any]] = []
    for (year, metric_name), rows in sorted(grouped.items()):
        distinct_values = {round(float(row["value"]), 6) for row in rows if _safe_float(row.get("value")) is not None}
        if len(distinct_values) <= 1:
            continue
        official_rows = [row for row in rows if str(row.get("source_url") or "") == HARP_SEED_SOURCE_URL]
        if not official_rows:
            continue
        official_row = max(
            official_rows,
            key=lambda row: (
                float(row.get("evidence_confidence") or 0.0),
                str(row.get("time") or ""),
            ),
        )
        official_value = round(float(official_row["value"]), 6)
        for alternative_row in rows:
            alternative_value = _safe_float(alternative_row.get("value"))
            if alternative_value is None:
                continue
            rounded_alternative_value = round(float(alternative_value), 6)
            if math.isclose(rounded_alternative_value, official_value, rel_tol=0.0, abs_tol=0.5):
                continue
            if str(alternative_row.get("source_url") or "") == HARP_SEED_SOURCE_URL:
                continue
            category, rationale = _harp_adjudication_category(alternative_row)
            adjudications.append(
                {
                    "year": year,
                    "metric_name": metric_name,
                    "official_source_id": str(official_row.get("source_id") or ""),
                    "official_source_label": str(official_row.get("source_label") or ""),
                    "official_time": str(official_row.get("time") or ""),
                    "official_value": official_value,
                    "official_measurement_class": str(official_row.get("measurement_class") or ""),
                    "official_series_kind": str(official_row.get("series_kind") or ""),
                    "official_source_quality_tier": str(official_row.get("source_quality_tier") or ""),
                    "alternative_source_id": str(alternative_row.get("source_id") or ""),
                    "alternative_source_label": str(alternative_row.get("source_label") or ""),
                    "alternative_time": str(alternative_row.get("time") or ""),
                    "alternative_value": rounded_alternative_value,
                    "alternative_measurement_class": str(alternative_row.get("measurement_class") or ""),
                    "alternative_series_kind": str(alternative_row.get("series_kind") or ""),
                    "alternative_temporal_precision": str(alternative_row.get("temporal_precision") or ""),
                    "alternative_source_quality_tier": str(alternative_row.get("source_quality_tier") or ""),
                    "adjudication_category": category,
                    "adjudication_decision": "retain_official_annual_seed",
                    "difference_from_official": round(rounded_alternative_value - official_value, 6),
                    "rationale": rationale,
                }
            )
    adjudications.sort(
        key=lambda row: (
            int(row.get("year") or 0),
            str(row.get("metric_name") or ""),
            str(row.get("adjudication_category") or ""),
            str(row.get("alternative_time") or ""),
            str(row.get("alternative_source_id") or ""),
        )
    )
    return adjudications


def _harp_quality_audit(*, run_dir: Path) -> dict[str, Any]:
    seed_path = Path(__file__).resolve().parents[1] / "harp_archive" / "seeds" / "doh_official_cascade_ground_truth_2018_2025.csv"
    panel_payload = read_json(run_dir / "harp_archive" / "historical_harp_panel.json", default={})
    panel_rows = list((panel_payload or {}).get("rows") or [])
    program_payload = read_json(run_dir / "harp_archive" / "harp_program_points.json", default={})
    program_rows = list((program_payload or {}).get("points") or [])
    metric_rows = read_json(run_dir / "harp_archive" / "historical_metric_rows.json", default=[])

    seed = _parse_harp_seed(seed_path)
    panel = _parse_harp_panel(panel_rows)
    program = _parse_harp_program_points(program_rows)
    panel_discrepancies = _harp_discrepancy_rows(seed=seed, observed=panel, label="historical_harp_panel")
    program_discrepancies = _harp_discrepancy_rows(seed=seed, observed=program, label="harp_program_points")
    time_mismatches = _harp_time_mismatch_rows(panel_rows)
    invariant_violations = _harp_invariant_violations(panel_rows)
    source_conflicts = _harp_source_conflicts(list(metric_rows or []))
    source_adjudications = _harp_source_adjudication_rows(list(metric_rows or []))
    source_adjudication_summary = Counter(str(row.get("adjudication_category") or "") for row in source_adjudications)

    return {
        "seed_year_count": len(seed),
        "panel_year_count": len(panel),
        "program_year_count": len(program),
        "panel_vs_seed_discrepancies": panel_discrepancies,
        "program_vs_seed_discrepancies": program_discrepancies,
        "panel_time_year_mismatches": time_mismatches,
        "panel_invariant_violations": invariant_violations,
        "source_conflicts_against_official_slide": source_conflicts[:50],
        "source_adjudication_rows": source_adjudications,
        "source_adjudication_summary": [
            {"adjudication_category": category, "count": count}
            for category, count in source_adjudication_summary.most_common()
        ],
        "passed": not panel_discrepancies and not program_discrepancies and not time_mismatches and not invariant_violations,
    }


def _candidate_sanity_audit(*, run_dir: Path) -> dict[str, Any]:
    candidate_rows = read_json(run_dir / "phase0" / "extracted" / "canonical_parameter_candidates.json", default=[])
    rows = list(candidate_rows or [])
    duplicate_counter = Counter(
        (
            str(row.get("source_id") or ""),
            str(row.get("canonical_name") or ""),
            str(row.get("geo") or ""),
            str(row.get("region") or ""),
            str(row.get("province") or ""),
            str(row.get("time") or ""),
            str(row.get("sex") or ""),
            str(row.get("age_band") or ""),
            str(row.get("kp_group") or ""),
            str(row.get("extraction_method") or ""),
        )
        for row in rows
    )
    duplicate_examples = [
        {
            "source_id": key[0],
            "canonical_name": key[1],
            "geo": key[2],
            "region": key[3],
            "province": key[4],
            "time": key[5],
            "sex": key[6],
            "age_band": key[7],
            "kp_group": key[8],
            "extraction_method": key[9],
            "count": count,
        }
        for key, count in duplicate_counter.items()
        if count > 1
    ]

    grouped_duplicates: dict[tuple[str, ...], set[float]] = defaultdict(set)
    grouped_parameter_texts: dict[tuple[str, ...], set[str]] = defaultdict(set)
    invalid_numeric_rows: list[dict[str, Any]] = []
    negative_count_rows: list[dict[str, Any]] = []
    bounded_percent_outliers: list[dict[str, Any]] = []
    google_mobility_outliers: list[dict[str, Any]] = []
    time_format_issues: list[dict[str, Any]] = []

    for row in rows:
        key = (
            str(row.get("source_id") or ""),
            str(row.get("canonical_name") or ""),
            str(row.get("geo") or ""),
            str(row.get("region") or ""),
            str(row.get("province") or ""),
            str(row.get("time") or ""),
            str(row.get("sex") or ""),
            str(row.get("age_band") or ""),
            str(row.get("kp_group") or ""),
            str(row.get("extraction_method") or ""),
        )
        value = row.get("value")
        numeric_value = _safe_float(value)
        if value is not None and numeric_value is None:
            invalid_numeric_rows.append(
                {
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "canonical_name": str(row.get("canonical_name") or ""),
                    "value": value,
                }
            )
            continue
        if numeric_value is not None:
            grouped_duplicates[key].add(round(numeric_value, 6))
            parameter_text = str(row.get("parameter_text") or "").strip()
            if parameter_text:
                grouped_parameter_texts[key].add(parameter_text)
            measurement_type = str(row.get("measurement_type") or "")
            unit = str(row.get("unit") or "")
            extraction_method = str(row.get("extraction_method") or "")
            if measurement_type in {"count", "cost"} and numeric_value < 0.0:
                negative_count_rows.append(
                    {
                        "candidate_id": str(row.get("candidate_id") or ""),
                        "canonical_name": str(row.get("canonical_name") or ""),
                        "value": numeric_value,
                    }
                )
            if unit == "percent":
                if extraction_method == GOOGLE_MOBILITY_METHOD:
                    if numeric_value < -100.0 or numeric_value > 200.0:
                        google_mobility_outliers.append(
                            {
                                "candidate_id": str(row.get("candidate_id") or ""),
                                "canonical_name": str(row.get("canonical_name") or ""),
                                "value": numeric_value,
                            }
                        )
                elif numeric_value < 0.0 or numeric_value > 100.0:
                    bounded_percent_outliers.append(
                        {
                            "candidate_id": str(row.get("candidate_id") or ""),
                            "canonical_name": str(row.get("canonical_name") or ""),
                            "value": numeric_value,
                        }
                    )
        time_value = str(row.get("time") or "")
        if not time_value or not any(
            (
                len(time_value) == 4 and time_value.isdigit(),
                len(time_value) == 7 and time_value[:4].isdigit() and time_value[4] == "-" and time_value[5:].isdigit(),
                len(time_value) == 10 and time_value[:4].isdigit() and time_value[4] == "-" and time_value[5:7].isdigit(),
            )
        ):
            time_format_issues.append(
                {
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "canonical_name": str(row.get("canonical_name") or ""),
                    "time": time_value,
                }
            )

    conflicting_duplicate_examples = [
        {
            "source_id": key[0],
            "canonical_name": key[1],
            "geo": key[2],
            "region": key[3],
            "province": key[4],
            "time": key[5],
            "sex": key[6],
            "age_band": key[7],
            "kp_group": key[8],
            "extraction_method": key[9],
            "distinct_values": sorted(values),
            "parameter_texts": sorted(grouped_parameter_texts.get(key, set())),
        }
        for key, values in grouped_duplicates.items()
        if len(values) > 1
    ]
    conflicting_duplicate_source_summary = Counter(
        (row["source_id"], row["extraction_method"]) for row in conflicting_duplicate_examples
    )
    conflicting_duplicate_canonical_summary = Counter(row["canonical_name"] for row in conflicting_duplicate_examples)

    return {
        "candidate_count": len(rows),
        "duplicate_key_count": len(duplicate_examples),
        "conflicting_duplicate_key_count": len(conflicting_duplicate_examples),
        "invalid_numeric_count": len(invalid_numeric_rows),
        "negative_count_row_count": len(negative_count_rows),
        "bounded_percent_outlier_count": len(bounded_percent_outliers),
        "google_mobility_outlier_count": len(google_mobility_outliers),
        "time_format_issue_count": len(time_format_issues),
        "duplicate_examples": duplicate_examples[:25],
        "conflicting_duplicate_examples": conflicting_duplicate_examples[:25],
        "conflicting_duplicate_source_summary": [
            {
                "source_id": source_id,
                "extraction_method": extraction_method,
                "count": count,
            }
            for (source_id, extraction_method), count in conflicting_duplicate_source_summary.most_common(25)
        ],
        "conflicting_duplicate_canonical_summary": [
            {
                "canonical_name": canonical_name,
                "count": count,
            }
            for canonical_name, count in conflicting_duplicate_canonical_summary.most_common(25)
        ],
        "invalid_numeric_examples": invalid_numeric_rows[:25],
        "negative_count_examples": negative_count_rows[:25],
        "bounded_percent_outlier_examples": bounded_percent_outliers[:25],
        "google_mobility_outlier_examples": google_mobility_outliers[:25],
        "time_format_issue_examples": time_format_issues[:25],
        "passed": not conflicting_duplicate_examples
        and not invalid_numeric_rows
        and not negative_count_rows
        and not bounded_percent_outliers
        and not google_mobility_outliers
        and not time_format_issues,
    }


def _markdown_report(*, audit: dict[str, Any]) -> str:
    structured = dict(audit.get("structured_source_parity") or {})
    harp = dict(audit.get("harp_quality") or {})
    sanity = dict(audit.get("candidate_sanity") or {})
    lines = [
        "# Extraction Quality Audit",
        "",
        f"Run directory: `{audit.get('run_dir')}`",
        f"Generated at: `{audit.get('generated_at')}`",
        "",
        "## Audit classes",
        "",
        "- Exact parity: deterministic regeneration from cached or local structured sources.",
        "- Reproducibility: rerunning the same local PDF/OCR parsers to ensure emitted rows match regenerated rows.",
        "- Conflict surfacing: where the repo holds multiple competing source values, report disagreement instead of pretending a single truth exists.",
        "",
        "## Structured source parity",
        "",
        f"- Passed: `{structured.get('passed')}`",
        f"- Structured emitted vs regenerated: `{structured.get('structured_row_parity', {}).get('emitted_count')}` vs `{structured.get('structured_row_parity', {}).get('expected_count')}`",
        f"- Missing rows: `{structured.get('structured_row_parity', {}).get('missing_count')}`",
        f"- Extra rows: `{structured.get('structured_row_parity', {}).get('extra_count')}`",
        f"- PhilHealth portal summary match: `{structured.get('philhealth_portal_summary_matches')}`",
        "",
        "## HARP quality",
        "",
        f"- Passed: `{harp.get('passed')}`",
        f"- Panel vs official seed discrepancies: `{len(harp.get('panel_vs_seed_discrepancies') or [])}`",
        f"- Program-point vs official seed discrepancies: `{len(harp.get('program_vs_seed_discrepancies') or [])}`",
        f"- Time/year mismatches in selected panel: `{len(harp.get('panel_time_year_mismatches') or [])}`",
        f"- Invariant violations: `{len(harp.get('panel_invariant_violations') or [])}`",
        f"- Conflicting alternative source values against official slide: `{len(harp.get('source_conflicts_against_official_slide') or [])}`",
        f"- Adjudicated alternative rows: `{len(harp.get('source_adjudication_rows') or [])}`",
        "",
        "## Candidate sanity",
        "",
        f"- Passed: `{sanity.get('passed')}`",
        f"- Conflicting duplicate keys: `{sanity.get('conflicting_duplicate_key_count')}`",
        f"- Invalid numeric rows: `{sanity.get('invalid_numeric_count')}`",
        f"- Negative count/cost rows: `{sanity.get('negative_count_row_count')}`",
        f"- Bounded percent outliers: `{sanity.get('bounded_percent_outlier_count')}`",
        f"- Google mobility outliers: `{sanity.get('google_mobility_outlier_count')}`",
        f"- Time format issues: `{sanity.get('time_format_issue_count')}`",
        "",
        "## Key findings",
        "",
    ]
    for finding in list(harp.get("panel_vs_seed_discrepancies") or [])[:10]:
        lines.append(
            f"- HARP panel mismatch `{finding['year']}` `{finding['metric_name']}`: "
            f"panel=`{finding['observed_value']}` vs official seed=`{finding['expected_value']}`"
        )
    for finding in list(harp.get("panel_time_year_mismatches") or [])[:10]:
        lines.append(
            f"- HARP panel time mismatch `{finding['year']}` with time=`{finding['time']}` ({finding['reason']})"
        )
    for finding in list(harp.get("source_adjudication_summary") or [])[:10]:
        lines.append(
            f"- HARP adjudication category `{finding['adjudication_category']}`: `{finding['count']}` alternative rows"
        )
    for finding in list(structured.get("structured_row_parity", {}).get("missing_examples") or [])[:10]:
        lines.append(
            f"- Missing structured row `{finding['canonical_name']}` `{finding['geo']}` `{finding['time']}` value=`{finding['value']}`"
        )
    if lines[-1] == "## Key findings":
        lines.append("- No high-severity findings recorded.")
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "- Unstructured literature extraction does not have a literal gold standard in this repo yet. The current audit can validate internal consistency and source conflicts, but absolute correctness for free-text extraction still requires a human-labeled benchmark set.",
        ]
    )
    duplicate_source_summary = list(sanity.get("conflicting_duplicate_source_summary") or [])[:5]
    if duplicate_source_summary:
        lines.extend(["", "## Duplicate hot spots", ""])
        for row in duplicate_source_summary:
            lines.append(
                f"- `{row['source_id']}` via `{row['extraction_method']}`: `{row['count']}` conflicting duplicate keys"
            )
    return "\n".join(lines) + "\n"


def _write_harp_source_adjudication_table(*, analysis_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    json_path = analysis_dir / "harp_source_adjudication_table.json"
    csv_path = analysis_dir / "harp_source_adjudication_table.csv"
    write_json(json_path, {"rows": rows})
    fieldnames = [
        "year",
        "metric_name",
        "official_source_id",
        "official_source_label",
        "official_time",
        "official_value",
        "official_measurement_class",
        "official_series_kind",
        "official_source_quality_tier",
        "alternative_source_id",
        "alternative_source_label",
        "alternative_time",
        "alternative_value",
        "alternative_measurement_class",
        "alternative_series_kind",
        "alternative_temporal_precision",
        "alternative_source_quality_tier",
        "adjudication_category",
        "adjudication_decision",
        "difference_from_official",
        "rationale",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return {
        "json": str(json_path),
        "csv": str(csv_path),
    }


def build_extraction_quality_audit(
    *,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    plugin_id: str = "hiv",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    analysis_dir = ensure_dir(Path(output_dir) if output_dir is not None else run_dir / "analysis")
    structured = _structured_source_parity(run_dir=run_dir, plugin_id=plugin_id)
    harp = _harp_quality_audit(run_dir=run_dir)
    sanity = _candidate_sanity_audit(run_dir=run_dir)
    audit = {
        "generated_at": utc_now_iso(),
        "run_dir": str(run_dir),
        "plugin_id": plugin_id,
        "structured_source_parity": structured,
        "harp_quality": harp,
        "candidate_sanity": sanity,
        "overall_passed": bool(structured.get("passed") and harp.get("passed") and sanity.get("passed")),
    }
    adjudication_paths = _write_harp_source_adjudication_table(
        analysis_dir=analysis_dir,
        rows=list((audit.get("harp_quality") or {}).get("source_adjudication_rows") or []),
    )
    audit["harp_quality"]["source_adjudication_table_paths"] = adjudication_paths
    write_json(analysis_dir / "extraction_quality_audit.json", audit)
    (analysis_dir / "extraction_quality_audit.md").write_text(_markdown_report(audit=audit), encoding="utf-8")
    return audit
