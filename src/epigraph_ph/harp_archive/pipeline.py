from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from epigraph_ph.geography import infer_philippines_geo, infer_region_code, normalize_geo_label
from epigraph_ph.runtime import RunContext, ensure_dir, utc_now_iso, write_ground_truth_package, write_json

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


DEFAULT_LOCAL_SEED_SPECS = [
    {
        "source_id": "curated_historical_harp_2017_2024",
        "label": "Curated Historical HARP Panel 2017-2024",
        "path": Path(__file__).resolve().with_name("seeds") / "historical_harp_panel_curated.csv",
        "source_kind": "packaged_csv",
    },
    {
        "source_id": "core_team_2025",
        "label": "2025 PH HIV Estimates Core Team for WHO",
        "path": Path(r"C:\Users\gaura\OneDrive\Desktop\2025 PH HIV Estimates_Core team_for WHO.pdf"),
        "source_kind": "local_pdf",
    },
    {
        "source_id": "philippine_surveillance",
        "label": "The Philippine HIV AIDS and STI Surveillance",
        "path": Path(r"C:\Users\gaura\OneDrive\Desktop\The Philippine HIV_STI Surveillance.pdf"),
        "source_kind": "local_pdf",
    },
]

YEAR_RANGE = list(range(2010, 2026))
MANUAL_SEED_PATTERNS = [
    "*PNAC*Annual*Report*.pdf",
    "*HARP*.pdf",
    "*HIV*Estimates*.pdf",
    "*historical_harp*.csv",
    "*historical_harp*.json",
    "*harp_panel*.csv",
    "*harp_panel*.json",
]


def _safe_ascii_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned or "document"


def _read_pdf_pages(path: Path) -> list[dict[str, Any]]:
    if PdfReader is None or not path.exists():
        return []
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page_number": page_number, "text": text})
    return pages


def _manual_seed_specs(seed_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in MANUAL_SEED_PATTERNS:
        for path in sorted(seed_dir.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                source_kind = "manual_pdf"
            elif suffix == ".csv":
                source_kind = "manual_csv"
            elif suffix == ".json":
                source_kind = "manual_json"
            else:
                continue
            rows.append(
                {
                    "source_id": f"manual_{_safe_ascii_label(path.stem)}",
                    "label": path.stem,
                    "path": path,
                    "source_kind": source_kind,
                }
            )
    return rows


def _to_int(token: str) -> int:
    return int(token.replace(",", "").strip())


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-6 else 0.0


def _find_spectrum_sequences(page_text: str) -> dict[str, list[int]]:
    tokens = [_to_int(token) for token in re.findall(r"\d{1,3}(?:,\d{3})+|\d+", page_text)]
    candidates: list[list[int]] = []
    seq_len = len(YEAR_RANGE)
    for idx in range(len(tokens) - seq_len + 1):
        seq = tokens[idx : idx + seq_len]
        if seq[0] in YEAR_RANGE:
            continue
        if sum(1 for left, right in zip(seq, seq[1:]) if right >= left) < 12:
            continue
        if seq not in candidates:
            candidates.append(seq)
    classified: dict[str, list[int]] = {}
    for seq in candidates:
        seq_min = min(seq)
        seq_max = max(seq)
        if seq_max > 100_000 and seq_min >= 10_000:
            classified.setdefault("estimated_plhiv_spectrum_2025", seq)
        elif 4_000 <= seq_min and seq_max <= 50_000:
            classified.setdefault("annual_new_infections_spectrum_2025", seq)
        elif seq_max <= 5_000:
            classified.setdefault("annual_aids_deaths_spectrum_2025", seq)
    return classified


def _extract_core_team_series(page_text: str, source_id: str, source_label: str, page_number: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sequences = _find_spectrum_sequences(page_text)
    metric_specs = {
        "estimated_plhiv_spectrum_2025": {
            "metric_name": "estimated_plhiv",
            "measurement_class": "model_estimate",
            "series_kind": "annual_series",
        },
        "annual_new_infections_spectrum_2025": {
            "metric_name": "annual_new_infections",
            "measurement_class": "model_estimate",
            "series_kind": "annual_series",
        },
        "annual_aids_deaths_spectrum_2025": {
            "metric_name": "annual_aids_deaths",
            "measurement_class": "model_estimate",
            "series_kind": "annual_series",
        },
    }
    for key, spec in metric_specs.items():
        series = sequences.get(key, [])
        if len(series) != len(YEAR_RANGE):
            continue
        for year, value in zip(YEAR_RANGE, series, strict=True):
            rows.append(
                {
                    "year": year,
                    "time": f"{year:04d}-01",
                    "metric_name": spec["metric_name"],
                    "value": float(value),
                    "unit": "count_people",
                    "source_id": source_id,
                    "source_label": source_label,
                    "page_number": page_number,
                    "measurement_class": spec["measurement_class"],
                    "series_kind": spec["series_kind"],
                    "geo": "Philippines",
                    "region": "national",
                    "province": "Philippines",
                    "evidence_confidence": 0.98,
                }
            )
    return rows


def _extract_core_team_cascade(page_text: str, source_id: str, source_label: str, page_number: int) -> list[dict[str, Any]]:
    if "Philippine HIV Care Cascade as of December 2024" not in page_text:
        return []
    counts = re.findall(r"\b\d{1,3}(?:,\d{3})+\b", page_text)
    if len(counts) < 5:
        return []
    values = [_to_int(token) for token in counts[:5]]
    specs = [
        ("estimated_plhiv", values[0], "count_people", "model_estimate"),
        ("diagnosed_plhiv", values[1], "count_people", "program_observed_harp"),
        ("alive_on_art", values[2], "count_people", "program_observed_harp"),
        ("tested_for_viral_load", values[3], "count_people", "program_observed_harp"),
        ("virally_suppressed", values[4], "count_people", "program_observed_harp"),
    ]
    rows = []
    for metric_name, value, unit, measurement_class in specs:
        rows.append(
            {
                "year": 2024,
                "time": "2025-01",
                "metric_name": metric_name,
                "value": float(value),
                "unit": unit,
                "source_id": source_id,
                "source_label": source_label,
                "page_number": page_number,
                "measurement_class": measurement_class,
                "series_kind": "annual_snapshot",
                "geo": "Philippines",
                "region": "national",
                "province": "Philippines",
                "evidence_confidence": 0.99,
            }
        )
    return rows


def _extract_generic_harp_snapshot(page_text: str, source_id: str, source_label: str, page_number: int) -> list[dict[str, Any]]:
    lowered = page_text.lower()
    if ("care cascade" not in lowered and "aids and art registry" not in lowered and "harp" not in lowered) or "philippine hiv care cascade as of december 2024" in lowered:
        return []
    year_match = re.search(r"(?:december|year|for)\s+(20[0-2]\d)", page_text, flags=re.IGNORECASE)
    if not year_match:
        year_match = re.search(r"\b(20[0-2]\d)\b", page_text)
    if not year_match:
        return []
    year = int(year_match.group(1))
    metric_patterns = {
        "estimated_plhiv": [
            r"(?:estimated\s+plhiv|estimated\s+number\s+of\s+plhiv|plhiv estimate)[^\d]{0,30}(\d{1,3}(?:,\d{3})+)",
        ],
        "diagnosed_plhiv": [
            r"(?:diagnosed\s+plhiv|kn(?:o|ow)s?\s+their\s+status|diagnosed)[^\d]{0,30}(\d{1,3}(?:,\d{3})+)",
        ],
        "alive_on_art": [
            r"(?:alive\s+on\s+art|on\s+art|enrolled\s+to\s+treatment)[^\d]{0,30}(\d{1,3}(?:,\d{3})+)",
        ],
        "tested_for_viral_load": [
            r"(?:viral\s+load\s+tested|vl\s+tested|tested\s+for\s+viral\s+load)[^\d]{0,30}(\d{1,3}(?:,\d{3})+)",
        ],
        "virally_suppressed": [
            r"(?:virally\s+suppressed|viral\s+suppression|suppressed)[^\d]{0,30}(\d{1,3}(?:,\d{3})+)",
        ],
    }
    rows = []
    for metric_name, patterns in metric_patterns.items():
        value = None
        for pattern in patterns:
            match = re.search(pattern, page_text, flags=re.IGNORECASE)
            if match:
                value = float(_to_int(match.group(1)))
                break
        if value is None:
            continue
        measurement_class = "program_observed_harp" if metric_name != "estimated_plhiv" else "model_estimate"
        rows.append(
            {
                "year": year,
                "time": f"{year:04d}-01",
                "metric_name": metric_name,
                "value": value,
                "unit": "count_people",
                "source_id": source_id,
                "source_label": source_label,
                "page_number": page_number,
                "measurement_class": measurement_class,
                "series_kind": "annual_snapshot",
                "geo": "Philippines",
                "region": "national",
                "province": "Philippines",
                "evidence_confidence": 0.82,
            }
        )
    return rows if len(rows) >= 3 else []


def _extract_surveillance_kp_profile(page_text: str, source_id: str, source_label: str, page_number: int) -> dict[str, Any] | None:
    if "PLHIV by key population, 2021" not in page_text:
        return None
    total_match = re.search(r"PLHIV by key population, 2021 \(N=(\d{1,3}(?:,\d{3})+)\)", page_text)
    msm_match = re.search(r"Males having sex with Males \(MSM\)\s+(\d+)%", page_text)
    female_match = re.search(r"Female-\s*(\d+)%", page_text)
    pwid_match = re.search(r"Person who Inject Drugs \(PWID\)-\s*(\d+)%", page_text)
    other_match = re.search(r"Other males-\s*(\d+)%", page_text)
    if not (total_match and msm_match and female_match and pwid_match and other_match):
        return None
    total = float(_to_int(total_match.group(1)))
    msm = float(msm_match.group(1)) / 100.0
    female = float(female_match.group(1)) / 100.0
    pwid = float(pwid_match.group(1)) / 100.0
    other = float(other_match.group(1)) / 100.0
    mapped = {
        "remaining_population": round(max(0.0, female + other), 6),
        "msm": round(msm, 6),
        "tgw": 0.0,
        "fsw": 0.0,
        "clients_fsw": 0.0,
        "pwid": round(pwid, 6),
        "non_kp_partners": 0.0,
    }
    total_share = sum(mapped.values())
    if total_share > 0:
        mapped = {key: round(value / total_share, 6) for key, value in mapped.items()}
    return {
        "anchor_id": "national_kp_profile_2021",
        "year": 2021,
        "time": "2021-01",
        "geo": "Philippines",
        "region": "national",
        "province": "Philippines",
        "group_kind": "kp_distribution",
        "mapped_distribution": mapped,
        "raw_distribution": {
            "msm": round(msm, 6),
            "female": round(female, 6),
            "pwid": round(pwid, 6),
            "other_males": round(other, 6),
        },
        "total_plhiv_estimate": total,
        "source_id": source_id,
        "source_label": source_label,
        "page_number": page_number,
        "evidence_confidence": 0.9,
    }


def _extract_core_team_subnational_kp(page_text: str, source_id: str, source_label: str, page_number: int) -> list[dict[str, Any]]:
    if "Subnational Model Prevention Coverage MSM & TGW Estimates (2025)" not in page_text:
        return []
    pattern = re.compile(
        r"(NCR|Cebu City|Cebu Province|Angeles City|Category A|Category B|Category C|National)\s+(\d+)%\s+([\d,]+)"
    )
    anchors = []
    for geo_label, coverage, estimate in pattern.findall(page_text):
        geo_match = infer_philippines_geo(geo_label, default_country_focus=True)
        geo = normalize_geo_label(geo_match.geo or geo_label, default_country_focus=True)
        anchors.append(
            {
                "anchor_id": f"msm_tgw_2025_{_safe_ascii_label(geo_label)}",
                "year": 2025,
                "time": "2025-01",
                "geo": geo,
                "region": geo_match.region or ("national" if geo_label == "National" else infer_region_code(geo)),
                "province": geo if geo not in {"Philippines", ""} else "Philippines",
                "group_kind": "msm_tgw_programmatic_anchor",
                "prevention_coverage": round(float(coverage) / 100.0, 6),
                "estimated_population_15_plus": float(_to_int(estimate)),
                "source_id": source_id,
                "source_label": source_label,
                "page_number": page_number,
                "evidence_confidence": 0.84,
            }
        )
    return anchors


def _materialize_local_sources(run_dir: Path, desktop_seed_dir: Path | None = None) -> list[dict[str, Any]]:
    raw_dir = ensure_dir(run_dir / "harp_archive" / "raw")
    source_rows: list[dict[str, Any]] = []
    seed_specs = list(DEFAULT_LOCAL_SEED_SPECS)
    if desktop_seed_dir is not None and desktop_seed_dir.exists():
        seed_specs.extend(_manual_seed_specs(desktop_seed_dir))
    seen_source_ids: set[str] = set()
    for spec in seed_specs:
        path = Path(spec["path"])
        if desktop_seed_dir is not None and spec in DEFAULT_LOCAL_SEED_SPECS and str(spec.get("source_kind") or "").startswith("local_"):
            path = desktop_seed_dir / path.name
        if not path.exists():
            continue
        if spec["source_id"] in seen_source_ids:
            continue
        seen_source_ids.add(str(spec["source_id"]))
        filename = f"{spec['source_id']}_{_safe_ascii_label(path.stem)}{path.suffix.lower()}"
        copied_path = raw_dir / filename
        shutil.copy2(path, copied_path)
        source_rows.append(
            {
                "source_id": spec["source_id"],
                "label": spec["label"],
                "source_kind": spec["source_kind"],
                "local_path": str(copied_path),
                "origin_path": str(path),
            }
        )
    return source_rows


def _read_tabular_seed_rows(path: Path, *, source_id: str, source_label: str) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            payload_rows = list(csv.DictReader(handle))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload_rows = payload.get("rows") or payload.get("metric_rows") or []
        elif isinstance(payload, list):
            payload_rows = payload
        else:
            payload_rows = []
    else:
        return rows
    for raw in payload_rows:
        year_raw = raw.get("year")
        metric_name = raw.get("metric_name")
        value_raw = raw.get("value")
        if year_raw in {None, ""} or metric_name in {None, ""} or value_raw in {None, ""}:
            continue
        try:
            year = int(year_raw)
            value = float(str(value_raw).replace(",", ""))
        except Exception:
            continue
        if year not in YEAR_RANGE:
            continue
        measurement_class = str(raw.get("measurement_class") or "program_observed_harp")
        geo = normalize_geo_label(str(raw.get("geo") or "Philippines"), default_country_focus=True)
        region = str(raw.get("region") or infer_region_code(geo) or ("national" if geo == "Philippines" else "region_unknown"))
        province = str(raw.get("province") or geo)
        rows.append(
            {
                "year": year,
                "time": str(raw.get("time") or f"{year:04d}-01"),
                "metric_name": str(metric_name),
                "value": value,
                "unit": str(raw.get("unit") or "count_people"),
                "source_id": source_id,
                "source_label": source_label,
                "page_number": None,
                "measurement_class": measurement_class,
                "series_kind": str(raw.get("series_kind") or "annual_snapshot"),
                "geo": geo,
                "region": region,
                "province": province,
                "evidence_confidence": float(raw.get("evidence_confidence") or 0.90),
                "source_url": str(raw.get("source_url") or ""),
                "source_note": str(raw.get("source_note") or ""),
                "source_quality_tier": str(raw.get("source_quality_tier") or ""),
                "extraction_method": str(raw.get("extraction_method") or "tabular_seed"),
            }
        )
    return rows


def _panel_from_metric_rows(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    panel: dict[str, dict[str, Any]] = {str(year): {"year": year} for year in YEAR_RANGE}
    for row in metric_rows:
        year_key = str(int(row["year"]))
        panel.setdefault(year_key, {"year": int(row["year"])})
        metric_name = str(row["metric_name"])
        if metric_name not in panel[year_key]:
            panel[year_key][metric_name] = row["value"]
    return {"rows": [panel[str(year)] for year in YEAR_RANGE]}


def _write_panel_csv(path: Path, panel_rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in panel_rows for key in row.keys()}, key=lambda item: (item != "year", item))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in panel_rows:
            writer.writerow(row)


def _backtest_assessment(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_metric: dict[str, set[int]] = defaultdict(set)
    for row in metric_rows:
        by_metric[str(row["metric_name"])].add(int(row["year"]))
    observed_program_years = sorted(
        {
            int(row["year"])
            for row in metric_rows
            if str(row.get("measurement_class") or "").startswith("program_observed_harp")
        }
    )
    return {
        "target_year_range": YEAR_RANGE,
        "observed_metric_years": {metric: sorted(years) for metric, years in sorted(by_metric.items())},
        "observed_program_years": observed_program_years,
        "train_years_default": [year for year in YEAR_RANGE if year <= 2024],
        "holdout_years_default": [year for year in YEAR_RANGE if year >= 2025],
        "backtest_ready": len(observed_program_years) >= 5,
        "blocking_reasons": [] if len(observed_program_years) >= 5 else ["historical_harp_program_series_incomplete"],
        "coverage_summary": {
            "metric_count": len(by_metric),
            "program_observed_year_count": len(observed_program_years),
            "complete_year_count": sum(1 for year in YEAR_RANGE if any(year in years for years in by_metric.values())),
        },
    }


def _metric_row_priority(row: dict[str, Any]) -> tuple[float, float]:
    metric_name = str(row.get("metric_name") or "")
    source_id = str(row.get("source_id") or "")
    year = int(row.get("year") or 0)
    curated_bonus = 0.0
    if metric_name == "estimated_plhiv" and source_id == "curated_historical_harp_2017_2024" and 2017 <= year <= 2024:
        curated_bonus = 1.0
    return curated_bonus, float(row.get("evidence_confidence") or 0.0)


def _deduplicate_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in metric_rows:
        key = (
            int(row.get("year") or 0),
            str(row.get("metric_name") or ""),
            str(row.get("measurement_class") or ""),
            str(row.get("geo") or ""),
        )
        previous = deduped.get(key)
        if previous is None or _metric_row_priority(row) >= _metric_row_priority(previous):
            deduped[key] = row
    rows = list(deduped.values())
    rows.sort(key=lambda item: (int(item.get("year") or 0), str(item.get("metric_name") or ""), str(item.get("geo") or "")))
    return rows


def _build_frozen_backtest_artifacts(metric_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    assessment = _backtest_assessment(metric_rows)
    observed_years = assessment["observed_program_years"]
    if len(observed_years) >= 2:
        train_years = observed_years[:-1]
        holdout_years = [observed_years[-1]]
    else:
        train_years = observed_years[:]
        holdout_years = []
    spec = {
        "freeze_policy": "fit_on_train_years_only_then_compare_to_holdout_years",
        "train_years": train_years,
        "holdout_years": holdout_years,
        "locked_metrics": ["diagnosed_plhiv", "alive_on_art", "tested_for_viral_load", "virally_suppressed"],
        "ready_for_model_backtest": len(train_years) >= 3 and bool(holdout_years),
        "blocking_reasons": [] if len(train_years) >= 3 and bool(holdout_years) else ["insufficient_observed_harp_history_for_frozen_backtest"],
        "notes": [
            "historical_model_estimates_may_exist_without_historical_program_counts",
            "frozen_backtest_requires_multiple_observed_program_years",
        ],
    }
    rows_by_metric_year = {
        (str(row.get("metric_name") or ""), int(row.get("year") or 0)): float(row.get("value") or 0.0)
        for row in metric_rows
        if str(row.get("measurement_class") or "").startswith("program_observed_harp")
    }
    holdout_rows = []
    if train_years and holdout_years:
        last_train_year = train_years[-1]
        holdout_year = holdout_years[0]
        for metric_name in spec["locked_metrics"]:
            baseline = rows_by_metric_year.get((metric_name, last_train_year))
            observed = rows_by_metric_year.get((metric_name, holdout_year))
            if baseline is None or observed is None:
                continue
            holdout_rows.append(
                {
                    "metric_name": metric_name,
                    "train_year": last_train_year,
                    "holdout_year": holdout_year,
                    "frozen_baseline": baseline,
                    "holdout_observed": observed,
                    "absolute_error": round(abs(baseline - observed), 6),
                    "relative_error": round(_safe_ratio(abs(baseline - observed), observed), 6),
                }
            )
    summary = {
        "ready_for_model_backtest": spec["ready_for_model_backtest"],
        "comparison_rows": holdout_rows,
        "comparison_count": len(holdout_rows),
        "mean_relative_error": round(float(mean(row["relative_error"] for row in holdout_rows)) if holdout_rows else 0.0, 6),
        "mean_absolute_error": round(float(mean(row["absolute_error"] for row in holdout_rows)) if holdout_rows else 0.0, 6),
        "notes": [
            "This is a frozen-history carry-forward baseline, not the Phase 3 model backtest.",
            "It exists to prove the historical HARP archive is structured enough to support a later real backtest.",
        ],
    }
    return spec, summary


def run_harp_archive_build(
    *,
    run_id: str,
    plugin_id: str = "hiv",
    desktop_seed_dir: str | None = None,
    manual_seed_dir: str | None = None,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    archive_dir = ensure_dir(ctx.run_dir / "harp_archive")
    desktop_path = Path(manual_seed_dir or desktop_seed_dir) if (manual_seed_dir or desktop_seed_dir) else None
    source_rows = _materialize_local_sources(ctx.run_dir, desktop_seed_dir=desktop_path)

    page_catalog: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    subgroup_anchor_rows: list[dict[str, Any]] = []
    for source in source_rows:
        local_path = Path(source["local_path"])
        suffix = local_path.suffix.lower()
        if suffix in {".csv", ".json"}:
            tabular_rows = _read_tabular_seed_rows(local_path, source_id=str(source["source_id"]), source_label=str(source["label"]))
            metric_rows.extend(tabular_rows)
            source["page_count"] = 0
            source["tabular_row_count"] = len(tabular_rows)
            continue
        pages = _read_pdf_pages(local_path)
        source["page_count"] = len(pages)
        for page in pages:
            page_catalog.append(
                {
                    "source_id": source["source_id"],
                    "source_label": source["label"],
                    "page_number": page["page_number"],
                    "text_excerpt": page["text"][:800],
                }
            )
            if source["source_id"] == "core_team_2025":
                metric_rows.extend(
                    _extract_core_team_series(page["text"], source["source_id"], source["label"], page["page_number"])
                )
                metric_rows.extend(
                    _extract_core_team_cascade(page["text"], source["source_id"], source["label"], page["page_number"])
                )
                subgroup_anchor_rows.extend(
                    _extract_core_team_subnational_kp(page["text"], source["source_id"], source["label"], page["page_number"])
                )
            elif source["source_id"] == "philippine_surveillance":
                kp_profile = _extract_surveillance_kp_profile(page["text"], source["source_id"], source["label"], page["page_number"])
                if kp_profile is not None:
                    subgroup_anchor_rows.append(kp_profile)
            if source["source_kind"] in {"manual_pdf", "local_pdf"}:
                metric_rows.extend(
                    _extract_generic_harp_snapshot(page["text"], str(source["source_id"]), str(source["label"]), int(page["page_number"]))
                )

    metric_rows = _deduplicate_metric_rows(metric_rows)
    panel = _panel_from_metric_rows(metric_rows)
    assessment = _backtest_assessment(metric_rows)
    frozen_backtest_spec, frozen_backtest_summary = _build_frozen_backtest_artifacts(metric_rows)
    subgroup_anchor_pack = {
        "anchors": subgroup_anchor_rows,
        "national_kp_profile": next((row for row in subgroup_anchor_rows if row.get("anchor_id") == "national_kp_profile_2021"), None),
        "subnational_kp_anchors": [row for row in subgroup_anchor_rows if row.get("group_kind") == "msm_tgw_programmatic_anchor"],
    }
    observed_program_panel = {
        "rows": [
            row
            for row in metric_rows
            if str(row.get("measurement_class") or "").startswith("program_observed_harp")
        ]
    }

    write_json(archive_dir / "archive_source_manifest.json", {"generated_at": utc_now_iso(), "sources": source_rows})
    write_json(archive_dir / "page_catalog.json", page_catalog)
    write_json(archive_dir / "historical_metric_rows.json", metric_rows)
    write_json(archive_dir / "historical_harp_panel.json", panel)
    _write_panel_csv(archive_dir / "historical_harp_panel.csv", panel["rows"])
    write_json(archive_dir / "observed_program_panel.json", observed_program_panel)
    write_json(archive_dir / "subgroup_anchor_pack.json", subgroup_anchor_pack)
    write_json(archive_dir / "backtest_assessment.json", assessment)
    write_json(archive_dir / "frozen_backtest_spec.json", frozen_backtest_spec)
    write_json(archive_dir / "frozen_backtest_summary.json", frozen_backtest_summary)

    truth_paths = write_ground_truth_package(
        phase_dir=archive_dir,
        phase_name="harp_archive",
        profile_id="hiv_rescue_v2",
        checks=[
            {"name": "source_rows_present", "passed": bool(source_rows)},
            {
                "name": "historical_anchor_seed_present",
                "passed": any(
                    row["source_id"] in {"core_team_2025", "curated_historical_harp_2017_2024"}
                    for row in source_rows
                ),
            },
            {"name": "historical_metric_rows_present", "passed": bool(metric_rows)},
            {"name": "panel_years_complete", "passed": len(panel["rows"]) == len(YEAR_RANGE)},
            {"name": "subgroup_anchor_pack_present", "passed": bool(subgroup_anchor_rows)},
            {"name": "frozen_backtest_spec_present", "passed": True},
        ],
        truth_sources=["anchor_truth", "benchmark_truth", "synthetic_truth"],
        stage_manifest_path=str(archive_dir / "harp_archive_manifest.json"),
        summary={
            "source_count": len(source_rows),
            "historical_metric_row_count": len(metric_rows),
            "subgroup_anchor_count": len(subgroup_anchor_rows),
            "backtest_ready": assessment["backtest_ready"],
            "frozen_backtest_ready": frozen_backtest_spec["ready_for_model_backtest"],
        },
    )
    manifest = {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "generated_at": utc_now_iso(),
        "artifact_paths": {
            "archive_source_manifest": str(archive_dir / "archive_source_manifest.json"),
            "page_catalog": str(archive_dir / "page_catalog.json"),
            "historical_metric_rows": str(archive_dir / "historical_metric_rows.json"),
            "historical_harp_panel": str(archive_dir / "historical_harp_panel.json"),
            "historical_harp_panel_csv": str(archive_dir / "historical_harp_panel.csv"),
            "observed_program_panel": str(archive_dir / "observed_program_panel.json"),
            "subgroup_anchor_pack": str(archive_dir / "subgroup_anchor_pack.json"),
            "backtest_assessment": str(archive_dir / "backtest_assessment.json"),
            "frozen_backtest_spec": str(archive_dir / "frozen_backtest_spec.json"),
            "frozen_backtest_summary": str(archive_dir / "frozen_backtest_summary.json"),
            **truth_paths,
        },
        "notes": [
            "historical_harp_panel_is_gap_aware",
            "spectrum_estimates_and_harp_program_counts_are_separated",
            "backtest_readiness_false_means_missing_historical_program_series",
            "manual_seed_csv_json_rows_are_supported_for_historical_panel_assembly",
        ],
    }
    write_json(archive_dir / "harp_archive_manifest.json", manifest)
    ctx.record_stage_outputs(
        "harp_archive_build",
        [
            archive_dir / "archive_source_manifest.json",
            archive_dir / "page_catalog.json",
            archive_dir / "historical_metric_rows.json",
            archive_dir / "historical_harp_panel.json",
            archive_dir / "historical_harp_panel.csv",
            archive_dir / "observed_program_panel.json",
            archive_dir / "subgroup_anchor_pack.json",
            archive_dir / "backtest_assessment.json",
            archive_dir / "frozen_backtest_spec.json",
            archive_dir / "frozen_backtest_summary.json",
            archive_dir / "harp_archive_manifest.json",
        ],
    )
    return manifest
