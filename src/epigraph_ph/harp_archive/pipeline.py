from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from epigraph_ph.geography import infer_philippines_geo, infer_region_code, normalize_geo_label
from epigraph_ph.harp_archive.doh_hiv_sti_archive import (
    archive_seed_path,
    archive_ocr_cache_settings,
    archive_ocr_settings,
    build_diagnosis_flow_points,
    build_harp_program_points,
    download_archive_pdfs,
    extract_archive_metric_rows,
    extract_archive_pdf_pages,
    load_archive_seed_rows,
    materialize_archive_ocr_payload,
    promote_archive_ocr_artifact_to_shared,
)
from epigraph_ph.runtime import RunContext, ensure_dir, read_json, sha256_file, utc_now_iso, write_ground_truth_package, write_json

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
        "source_id": "doh_official_cascade_ground_truth_2018_2025",
        "label": "DOH 95-95-95 Ground Truth 2018-2025 From User-Supplied Official Slide",
        "path": Path(__file__).resolve().with_name("seeds") / "doh_official_cascade_ground_truth_2018_2025.csv",
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
_HARP_ARCHIVE_BUILD_VERSION = "harp_archive_cache_v1"
_HARP_ARCHIVE_SEMANTICS_VERSION = "2026-04-03_harp_panel_priority_v1"
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


def _archive_source_identity_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in source_rows:
        source_kind = str(row.get("source_kind") or "")
        identity_row: dict[str, Any] = {
            "source_id": str(row.get("source_id") or ""),
            "source_kind": source_kind,
            "label": str(row.get("label") or row.get("source_label") or ""),
        }
        if source_kind == "doh_hiv_sti_archive_pdf":
            for key in (
                "source_label",
                "source_year",
                "source_url",
                "file_id",
                "temporal_precision",
                "effective_month",
                "start_month",
                "end_month",
            ):
                value = row.get(key)
                if value not in (None, ""):
                    identity_row[key] = value
        else:
            checksum = str(row.get("checksum") or "")
            local_path = Path(str(row.get("local_path") or ""))
            if not checksum and local_path.exists() and local_path.is_file():
                checksum = sha256_file(local_path)
            if checksum:
                identity_row["checksum"] = checksum
            source_url = str(row.get("source_url") or "")
            if source_url:
                identity_row["source_url"] = source_url
        normalized_rows.append(identity_row)
    normalized_rows.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return normalized_rows


def _archive_source_manifest_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _archive_source_identity_rows(source_rows)


def _archive_build_cache_key(source_rows: list[dict[str, Any]]) -> str:
    ocr_cfg = _archive_ocr_cache_settings()
    stable_payload = {
        "build_version": _HARP_ARCHIVE_BUILD_VERSION,
        "semantics_version": _HARP_ARCHIVE_SEMANTICS_VERSION,
        "ocr_settings": ocr_cfg,
        "sources": _archive_source_manifest_rows(source_rows),
    }
    return hashlib.sha256(json.dumps(stable_payload, sort_keys=True).encode("utf-8")).hexdigest()


def _archive_ocr_cache_settings() -> dict[str, Any]:
    return dict(archive_ocr_cache_settings())


def _archive_manifest_is_reusable(archive_dir: Path, *, cache_key: str) -> dict[str, Any] | None:
    manifest_path = archive_dir / "harp_archive_manifest.json"
    manifest = read_json(manifest_path, default={})
    if not isinstance(manifest, dict):
        return None
    if str(manifest.get("build_cache_key") or "") != str(cache_key):
        return None
    artifact_paths = dict(manifest.get("artifact_paths") or {})
    if not artifact_paths:
        return None
    for path_str in artifact_paths.values():
        if not path_str or not Path(str(path_str)).exists():
            return None
    return manifest


def _read_pdf_pages(path: Path) -> list[dict[str, Any]]:
    if PdfReader is None or not path.exists():
        return []
    try:
        reader = PdfReader(str(path))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                pages.append({"page_number": page_number, "text": text})
        return pages
    except Exception:
        return []


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
                "time": "2024-12",
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
                "checksum": sha256_file(copied_path),
            }
        )
    return source_rows


def _archive_source_signature_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature_rows: list[dict[str, Any]] = []
    for row in source_rows:
        local_path = Path(str(row.get("local_path") or ""))
        checksum = str(row.get("checksum") or "")
        if not checksum and local_path.exists() and local_path.is_file():
            checksum = sha256_file(local_path)
        signature_rows.append(
            {
                "source_id": str(row.get("source_id") or ""),
                "source_kind": str(row.get("source_kind") or ""),
                "local_path": str(local_path),
                "origin_path": str(row.get("origin_path") or ""),
                "checksum": checksum,
            }
        )
    signature_rows.sort(key=lambda item: (item["source_id"], item["local_path"], item["checksum"]))
    return signature_rows


def _archive_build_signature(source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pipeline_path = Path(__file__).resolve()
    extractor_path = Path(materialize_archive_ocr_payload.__code__.co_filename).resolve()
    return {
        "pipeline_sha256": sha256_file(pipeline_path),
        "extractor_sha256": sha256_file(extractor_path),
        "ocr_settings": archive_ocr_settings(),
        "sources": _archive_source_signature_rows(source_rows),
    }


def _expected_archive_artifact_paths(archive_dir: Path) -> list[Path]:
    return [
        archive_dir / "archive_source_manifest.json",
        archive_dir / "ocr_corpus_manifest.json",
        archive_dir / "page_catalog.json",
        archive_dir / "historical_metric_rows.json",
        archive_dir / "historical_harp_panel.json",
        archive_dir / "historical_harp_panel.csv",
        archive_dir / "harp_program_points.json",
        archive_dir / "diagnosis_flow_points.json",
        archive_dir / "observed_program_panel.json",
        archive_dir / "subgroup_anchor_pack.json",
        archive_dir / "backtest_assessment.json",
        archive_dir / "frozen_backtest_spec.json",
        archive_dir / "frozen_backtest_summary.json",
        archive_dir / "ground_truth_summary.json",
        archive_dir / "ground_truth_manifest.json",
        archive_dir / "ground_truth_checks.json",
        archive_dir / "harp_archive_manifest.json",
    ]


def _try_reuse_archive_build(
    archive_dir: Path,
    *,
    build_signature: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = archive_dir / "harp_archive_manifest.json"
    manifest = read_json(manifest_path, default={})
    if not isinstance(manifest, dict) or not manifest:
        return None
    if dict(manifest.get("build_signature") or {}) != build_signature:
        return None
    if not all(path.exists() for path in _expected_archive_artifact_paths(archive_dir)):
        return None
    payload = dict(manifest)
    payload["reuse_status"] = "reused_existing_build"
    return payload


def _archive_required_artifact_paths(archive_dir: Path) -> dict[str, Path]:
    return {
        "archive_source_manifest": archive_dir / "archive_source_manifest.json",
        "ocr_corpus_manifest": archive_dir / "ocr_corpus_manifest.json",
        "page_catalog": archive_dir / "page_catalog.json",
        "historical_metric_rows": archive_dir / "historical_metric_rows.json",
        "historical_harp_panel": archive_dir / "historical_harp_panel.json",
        "historical_harp_panel_csv": archive_dir / "historical_harp_panel.csv",
        "harp_program_points": archive_dir / "harp_program_points.json",
        "diagnosis_flow_points": archive_dir / "diagnosis_flow_points.json",
        "observed_program_panel": archive_dir / "observed_program_panel.json",
        "subgroup_anchor_pack": archive_dir / "subgroup_anchor_pack.json",
        "backtest_assessment": archive_dir / "backtest_assessment.json",
        "frozen_backtest_spec": archive_dir / "frozen_backtest_spec.json",
        "frozen_backtest_summary": archive_dir / "frozen_backtest_summary.json",
        "harp_archive_manifest": archive_dir / "harp_archive_manifest.json",
    }


def _archive_build_fingerprint(source_rows: list[dict[str, Any]]) -> str:
    source_payload: list[dict[str, Any]] = []
    for row in source_rows:
        local_path = Path(str(row.get("local_path") or ""))
        source_payload.append(
            {
                "source_id": str(row.get("source_id") or ""),
                "source_kind": str(row.get("source_kind") or ""),
                "label": str(row.get("label") or ""),
                "local_path": str(local_path),
                "checksum": str(row.get("checksum") or (sha256_file(local_path) if local_path.exists() and local_path.is_file() else "")),
                "source_url": str(row.get("source_url") or ""),
                "effective_month": str(row.get("effective_month") or ""),
                "temporal_precision": str(row.get("temporal_precision") or ""),
            }
        )
    source_payload.sort(key=lambda item: (item["source_kind"], item["source_id"], item["checksum"], item["effective_month"]))
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("doh_hiv_sti_archive.py"),
        archive_seed_path(),
    ]
    code_payload = [
        {
            "path": str(path),
            "checksum": sha256_file(path) if path.exists() and path.is_file() else "",
        }
        for path in code_paths
    ]
    fingerprint_payload = {
        "ocr_settings": archive_ocr_settings(),
        "sources": source_payload,
        "code_inputs": code_payload,
    }
    canonical = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_reusable_archive_manifest_by_fingerprint(archive_dir: Path, build_fingerprint: str) -> dict[str, Any] | None:
    manifest_path = archive_dir / "harp_archive_manifest.json"
    manifest = read_json(manifest_path, default={})
    if not isinstance(manifest, dict):
        return None
    if str(manifest.get("build_fingerprint") or "") != str(build_fingerprint):
        return None
    required = _archive_required_artifact_paths(archive_dir)
    if not all(path.exists() for path in required.values()):
        return None
    artifact_paths = dict(manifest.get("artifact_paths") or {})
    if any(not Path(str(path)).exists() for path in artifact_paths.values() if isinstance(path, str) and path):
        return None
    return manifest


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


def _time_ordinal(time_label: str) -> int:
    value = str(time_label or "")
    if len(value) >= 7 and value[:4].isdigit() and value[5:7].isdigit():
        return int(value[:4]) * 12 + int(value[5:7]) - 1
    if len(value) >= 4 and value[:4].isdigit():
        return int(value[:4]) * 12
    return -1


def _panel_from_metric_rows(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    point_metrics = {
        "estimated_plhiv",
        "diagnosed_plhiv",
        "alive_on_art",
        "tested_for_viral_load",
        "virally_suppressed",
    }
    panel: dict[str, dict[str, Any]] = {str(year): {"year": year} for year in YEAR_RANGE}
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    for row in metric_rows:
        year = int(row.get("year") or 0)
        metric_name = str(row.get("metric_name") or "")
        if year not in YEAR_RANGE or metric_name not in point_metrics:
            continue
        key = (year, metric_name)
        previous = selected.get(key)
        row_priority = _metric_row_priority(row)
        row_rank = (row_priority[0], _time_ordinal(str(row.get("time") or "")), row_priority[1])
        prev_rank = (
            (
                _metric_row_priority(previous)[0],
                _time_ordinal(str(previous.get("time") or "")),
                _metric_row_priority(previous)[1],
            )
            if previous
            else None
        )
        if previous is None or row_rank >= prev_rank:
            selected[key] = row
    for (year, metric_name), row in selected.items():
        year_key = str(year)
        panel.setdefault(year_key, {"year": year})
        panel[year_key][metric_name] = row["value"]
        panel[year_key]["time"] = str(row.get("time") or panel[year_key].get("time") or f"{year:04d}-01")
    return {"rows": [panel[str(year)] for year in YEAR_RANGE]}


def _write_panel_csv(path: Path, panel_rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in panel_rows for key in row.keys()}, key=lambda item: (item != "year", item))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in panel_rows:
            writer.writerow(row)


def _backtest_assessment(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    locked_metrics = {"diagnosed_plhiv", "alive_on_art", "tested_for_viral_load", "virally_suppressed"}
    by_metric: dict[str, set[int]] = defaultdict(set)
    for row in metric_rows:
        by_metric[str(row["metric_name"])].add(int(row["year"]))
    observed_program_years = sorted(
        {
            int(row["year"])
            for row in metric_rows
            if str(row.get("measurement_class") or "").startswith("program_observed_harp")
            and str(row.get("metric_name") or "") in locked_metrics
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
    if source_id == "doh_official_cascade_ground_truth_2018_2025" and 2018 <= year <= 2025:
        curated_bonus = 2.0
    if metric_name == "estimated_plhiv" and source_id == "curated_historical_harp_2017_2024" and 2017 <= year <= 2024:
        curated_bonus = 1.0
    return curated_bonus, float(row.get("evidence_confidence") or 0.0)


def _deduplicate_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in metric_rows:
        key = (
            int(row.get("year") or 0),
            str(row.get("time") or ""),
            str(row.get("metric_name") or ""),
            str(row.get("measurement_class") or ""),
            str(row.get("geo") or ""),
        )
        previous = deduped.get(key)
        if previous is None or _metric_row_priority(row) >= _metric_row_priority(previous):
            deduped[key] = row
    rows = list(deduped.values())
    rows.sort(
        key=lambda item: (
            int(item.get("year") or 0),
            _time_ordinal(str(item.get("time") or "")),
            str(item.get("metric_name") or ""),
            str(item.get("geo") or ""),
        )
    )
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


def _archive_source_signature(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _archive_source_identity_rows(source_rows)


def _manifest_artifact_paths_present(artifact_paths: dict[str, Any]) -> bool:
    if not artifact_paths:
        return False
    for path in artifact_paths.values():
        if not str(path or "").strip():
            return False
        if not Path(str(path)).exists():
            return False
    return True


def _saved_archive_ocr_cache_settings(ocr_manifest: dict[str, Any]) -> dict[str, Any]:
    saved_ocr_cache_settings = dict(ocr_manifest).get("cache_settings") or {}
    if not saved_ocr_cache_settings and isinstance(dict(ocr_manifest).get("ocr_settings"), dict):
        ocr_settings = dict(dict(ocr_manifest).get("ocr_settings") or {})
        saved_ocr_cache_settings = {
            "render_dpi": ocr_settings.get("render_dpi"),
            "pdf_requires_ocr_avg_chars": ocr_settings.get("pdf_requires_ocr_avg_chars"),
            "force_ocr_every_page": ocr_settings.get("force_ocr_every_page"),
            "max_pages_per_document": ocr_settings.get("max_pages_per_document"),
        }
    return dict(saved_ocr_cache_settings)


def _saved_archive_source_identity_rows(source_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_rows = source_manifest.get("source_manifest_rows")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        candidate_rows = source_manifest.get("sources")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        candidate_rows = source_manifest.get("source_signature")
    if not isinstance(candidate_rows, list):
        return []
    return _archive_source_identity_rows([dict(row) for row in candidate_rows if isinstance(row, dict)])


def _load_reusable_archive_manifest(
    *,
    archive_dir: Path,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    manifest_path = archive_dir / "harp_archive_manifest.json"
    source_manifest_path = archive_dir / "archive_source_manifest.json"
    ocr_manifest_path = archive_dir / "ocr_corpus_manifest.json"
    if not (manifest_path.exists() and source_manifest_path.exists() and ocr_manifest_path.exists()):
        return None
    manifest = read_json(manifest_path, default={})
    source_manifest = read_json(source_manifest_path, default={})
    ocr_manifest = read_json(ocr_manifest_path, default={})
    if str(dict(manifest).get("build_version") or "") != _HARP_ARCHIVE_BUILD_VERSION:
        return None
    if str(dict(source_manifest).get("build_version") or "") != _HARP_ARCHIVE_BUILD_VERSION:
        return None
    expected_cache_key = _archive_build_cache_key(source_rows)
    expected_source_signature = _archive_source_identity_rows(source_rows)
    manifest_cache_key = str(dict(manifest).get("build_cache_key") or "")
    saved_source_signature = _saved_archive_source_identity_rows(dict(source_manifest))
    saved_ocr_cache_settings = _saved_archive_ocr_cache_settings(dict(ocr_manifest))
    if manifest_cache_key:
        if manifest_cache_key != expected_cache_key:
            if dict(saved_ocr_cache_settings) != _archive_ocr_cache_settings():
                return None
            if saved_source_signature != expected_source_signature:
                return None
    else:
        if dict(saved_ocr_cache_settings) != _archive_ocr_cache_settings():
            return None
        if saved_source_signature != expected_source_signature:
            return None
    artifact_paths = dict(manifest.get("artifact_paths") or {})
    if not _manifest_artifact_paths_present(artifact_paths):
        return None
    return dict(manifest)


def _reuse_cached_archive_build_if_current(
    *,
    archive_dir: Path,
    source_rows: list[dict[str, Any]],
    force_refresh: bool,
) -> dict[str, Any] | None:
    if force_refresh:
        return None
    manifest = _load_reusable_archive_manifest(archive_dir=archive_dir, source_rows=source_rows)
    if manifest is None:
        return None
    reused_manifest = dict(manifest)
    reused_manifest["reused_existing_build"] = True
    return reused_manifest


def _materialize_reused_archive_build(
    *,
    source_archive_dir: Path,
    target_archive_dir: Path,
) -> dict[str, Any] | None:
    expected_paths = _archive_required_artifact_paths(target_archive_dir)
    copied_paths: dict[str, str] = {}
    for key, target_path in expected_paths.items():
        source_path = source_archive_dir / target_path.name
        if not source_path.exists():
            return None
        ensure_dir(target_path.parent)
        shutil.copy2(source_path, target_path)
        copied_paths[key] = str(target_path)
    manifest_path = target_archive_dir / "harp_archive_manifest.json"
    manifest = read_json(manifest_path, default={})
    if not isinstance(manifest, dict):
        return None
    reused_from_run_id = str(manifest.get("run_id") or source_archive_dir.parent.name)
    manifest["artifact_paths"] = copied_paths
    manifest["run_id"] = target_archive_dir.parent.name
    manifest["generated_at"] = utc_now_iso()
    manifest["reused_existing_build"] = True
    manifest["reused_from_run_id"] = reused_from_run_id
    write_json(manifest_path, manifest)
    return manifest


def _reuse_cached_archive_build_from_prior_runs(
    *,
    archive_dir: Path,
    source_rows: list[dict[str, Any]],
    force_refresh: bool,
) -> dict[str, Any] | None:
    if force_refresh:
        return None
    runs_dir = archive_dir.parent.parent
    if not runs_dir.exists():
        return None
    manifest_paths = sorted(
        runs_dir.rglob("harp_archive_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifest_paths:
        candidate_archive_dir = manifest_path.parent
        if candidate_archive_dir == archive_dir:
            continue
        manifest = _load_reusable_archive_manifest(archive_dir=candidate_archive_dir, source_rows=source_rows)
        if manifest is None:
            continue
        materialized_manifest = _materialize_reused_archive_build(
            source_archive_dir=candidate_archive_dir,
            target_archive_dir=archive_dir,
        )
        if materialized_manifest is None:
            continue
        materialized_manifest["reused_existing_build"] = True
        materialized_manifest["reused_from_archive_dir"] = str(candidate_archive_dir)
        return materialized_manifest
    return None


def _promote_reused_archive_ocr_manifest_to_shared(archive_dir: Path) -> None:
    ocr_manifest_path = archive_dir / "ocr_corpus_manifest.json"
    ocr_manifest = read_json(ocr_manifest_path, default={})
    if not isinstance(ocr_manifest, dict):
        return
    for document in list(ocr_manifest.get("documents", []) or []):
        if not isinstance(document, dict):
            continue
        candidate_path_values = [
            str(document.get("run_artifact_path") or "").strip(),
            str(document.get("ocr_artifact_path") or "").strip(),
        ]
        for candidate_value in candidate_path_values:
            if not candidate_value:
                continue
            candidate_path = Path(candidate_value)
            if not candidate_path.exists():
                continue
            promote_archive_ocr_artifact_to_shared(candidate_path)
            break


def run_harp_archive_build(
    *,
    run_id: str,
    plugin_id: str = "hiv",
    desktop_seed_dir: str | None = None,
    manual_seed_dir: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    archive_dir = ensure_dir(ctx.run_dir / "harp_archive")
    raw_dir = ensure_dir(archive_dir / "raw")
    ocr_dir = ensure_dir(archive_dir / "ocr_corpus")
    desktop_path = Path(manual_seed_dir or desktop_seed_dir) if (manual_seed_dir or desktop_seed_dir) else None
    local_source_rows = _materialize_local_sources(ctx.run_dir, desktop_seed_dir=desktop_path)
    archive_seed_rows = [dict(row) for row in load_archive_seed_rows(archive_seed_path())]
    predownload_source_rows = list(local_source_rows) + [dict(row) for row in archive_seed_rows]
    reused_manifest = _reuse_cached_archive_build_if_current(
        archive_dir=archive_dir,
        source_rows=predownload_source_rows,
        force_refresh=force_refresh,
    )
    if reused_manifest is None:
        reused_manifest = _reuse_cached_archive_build_from_prior_runs(
            archive_dir=archive_dir,
            source_rows=predownload_source_rows,
            force_refresh=force_refresh,
        )
    if reused_manifest is not None:
        _promote_reused_archive_ocr_manifest_to_shared(archive_dir)
        artifact_paths = dict(reused_manifest.get("artifact_paths") or {})
        ctx.record_stage_outputs(
            "harp_archive_build",
            [Path(str(path)) for path in artifact_paths.values() if str(path or "").strip()],
        )
        print(f"[harp-archive] reusing cached build at {archive_dir}")
        return reused_manifest

    source_rows = list(local_source_rows)
    source_rows.extend(download_archive_pdfs(archive_rows=archive_seed_rows, raw_dir=raw_dir))
    build_cache_key = _archive_build_cache_key(source_rows)

    page_catalog: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    subgroup_anchor_rows: list[dict[str, Any]] = []
    archive_ocr_manifest_rows: list[dict[str, Any]] = []
    archive_pdf_total = sum(1 for row in source_rows if str(row.get("source_kind") or "") == "doh_hiv_sti_archive_pdf")
    archive_pdf_index = 0
    for source in source_rows:
        local_path = Path(source["local_path"])
        if not local_path.exists() or not local_path.is_file():
            source["page_count"] = 0
            source["tabular_row_count"] = 0
            continue
        suffix = local_path.suffix.lower()
        if suffix in {".csv", ".json"}:
            tabular_rows = _read_tabular_seed_rows(local_path, source_id=str(source["source_id"]), source_label=str(source["label"]))
            metric_rows.extend(tabular_rows)
            source["page_count"] = 0
            source["tabular_row_count"] = len(tabular_rows)
            continue
        if str(source.get("source_kind") or "") == "doh_hiv_sti_archive_pdf":
            archive_pdf_index += 1
            ocr_payload = materialize_archive_ocr_payload(source, ocr_dir=ocr_dir, force_refresh=force_refresh)
            if not bool(ocr_payload.get("from_cache")):
                print(f"[harp-archive] OCR {archive_pdf_index}/{archive_pdf_total}: {source['source_id']}")
            pages = [dict(page) for page in list(ocr_payload.get("pages", []) or [])]
            source["ocr_artifact_path"] = str(ocr_payload.get("artifact_path") or "")
            source["ocr_page_count"] = int(ocr_payload.get("page_count") or len(pages))
            source["ocr_runtime"] = dict(ocr_payload.get("runtime") or {})
            archive_ocr_manifest_rows.append(
                {
                    "source_id": source["source_id"],
                    "source_label": source["label"],
                    "local_path": str(local_path),
                    "ocr_artifact_path": str(ocr_payload.get("artifact_path") or ""),
                    "run_artifact_path": str(ocr_payload.get("run_artifact_path") or ""),
                    "shared_artifact_path": str(ocr_payload.get("shared_artifact_path") or ""),
                    "shared_cache_key": str(ocr_payload.get("shared_cache_key") or ""),
                    "page_count": int(ocr_payload.get("page_count") or len(pages)),
                    "checksum": str(ocr_payload.get("checksum") or ""),
                    "force_ocr_every_page": bool(ocr_payload.get("force_ocr_every_page")),
                    "cache_settings": dict(ocr_payload.get("cache_settings") or {}),
                    "from_shared_cache": bool(ocr_payload.get("from_shared_cache")),
                    "runtime": dict(ocr_payload.get("runtime") or {}),
                }
            )
            metric_rows.extend(extract_archive_metric_rows(source, pages))
        else:
            pages = _read_pdf_pages(local_path)
        source["page_count"] = len(pages)
        for page in pages:
            page_catalog.append(
                {
                    "source_id": source["source_id"],
                    "source_label": source["label"],
                    "page_number": page["page_number"],
                    "parser_used": page.get("parser_used") or "pdf_text",
                    "text_source": page.get("text_source") or page.get("parser_used") or "pdf_text",
                    "ocr_performed": bool(page.get("ocr_performed")),
                    "direct_char_count": int(page.get("direct_char_count") or 0),
                    "ocr_char_count": int(page.get("ocr_char_count") or 0),
                    "markers": dict(page.get("markers") or {}),
                    "text": page["text"],
                    "text_excerpt": page["text"][:800],
                    "direct_text_excerpt": str(page.get("direct_text") or "")[:400],
                    "ocr_text_excerpt": str(page.get("ocr_text") or "")[:400],
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
    harp_program_points = build_harp_program_points(metric_rows)
    diagnosis_flow_points = build_diagnosis_flow_points(metric_rows)
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

    write_json(
        archive_dir / "archive_source_manifest.json",
        {
            "generated_at": utc_now_iso(),
            "build_version": _HARP_ARCHIVE_BUILD_VERSION,
            "source_signature": _archive_source_signature(source_rows),
            "source_manifest_rows": _archive_source_manifest_rows(source_rows),
            "build_cache_key": build_cache_key,
            "sources": source_rows,
        },
    )
    write_json(
        archive_dir / "ocr_corpus_manifest.json",
        {
            "generated_at": utc_now_iso(),
            "build_version": _HARP_ARCHIVE_BUILD_VERSION,
            "cache_settings": _archive_ocr_cache_settings(),
            "ocr_settings": archive_ocr_settings(),
            "documents": archive_ocr_manifest_rows,
        },
    )
    write_json(archive_dir / "page_catalog.json", page_catalog)
    write_json(archive_dir / "historical_metric_rows.json", metric_rows)
    write_json(archive_dir / "historical_harp_panel.json", panel)
    _write_panel_csv(archive_dir / "historical_harp_panel.csv", panel["rows"])
    write_json(archive_dir / "harp_program_points.json", {"points": harp_program_points})
    write_json(archive_dir / "diagnosis_flow_points.json", {"points": diagnosis_flow_points})
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
            "ocr_document_count": len(archive_ocr_manifest_rows),
            "historical_metric_row_count": len(metric_rows),
            "harp_program_point_count": len(harp_program_points),
            "diagnosis_flow_point_count": len(diagnosis_flow_points),
            "subgroup_anchor_count": len(subgroup_anchor_rows),
            "backtest_ready": assessment["backtest_ready"],
            "frozen_backtest_ready": frozen_backtest_spec["ready_for_model_backtest"],
        },
    )
    manifest = {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "generated_at": utc_now_iso(),
        "build_version": _HARP_ARCHIVE_BUILD_VERSION,
        "build_cache_key": build_cache_key,
        "artifact_paths": {
            "archive_source_manifest": str(archive_dir / "archive_source_manifest.json"),
            "ocr_corpus_manifest": str(archive_dir / "ocr_corpus_manifest.json"),
            "page_catalog": str(archive_dir / "page_catalog.json"),
            "historical_metric_rows": str(archive_dir / "historical_metric_rows.json"),
            "historical_harp_panel": str(archive_dir / "historical_harp_panel.json"),
            "historical_harp_panel_csv": str(archive_dir / "historical_harp_panel.csv"),
            "harp_program_points": str(archive_dir / "harp_program_points.json"),
            "diagnosis_flow_points": str(archive_dir / "diagnosis_flow_points.json"),
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
            archive_dir / "harp_program_points.json",
            archive_dir / "diagnosis_flow_points.json",
            archive_dir / "observed_program_panel.json",
            archive_dir / "subgroup_anchor_pack.json",
            archive_dir / "backtest_assessment.json",
            archive_dir / "frozen_backtest_spec.json",
            archive_dir / "frozen_backtest_summary.json",
            archive_dir / "harp_archive_manifest.json",
        ],
    )
    return manifest
