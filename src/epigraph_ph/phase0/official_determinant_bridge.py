from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

from epigraph_ph.phase0.phase3_target_contract import PHASE3_MODULE_CONTRACT, phase3_target_canonical_names
from epigraph_ph.runtime import ensure_dir, read_json, write_json


OFFICIAL_DETERMINANT_BRIDGE_SCHEMA_VERSION = "phase0_official_determinant_bridge.v1"
UNAIDS_AIDSINFO_URL = "https://aidsinfo.unaids.org/"
WDI_DATABANK_URL = "https://databank.worldbank.org/source/world-development-indicators"

WDI_DETERMINANT_SERIES: tuple[dict[str, Any], ...] = (
    {
        "indicator_code": "EN.POP.DNST",
        "canonical_name": "population_density",
        "source_id": "world_bank_wdi_en_pop_dnst",
        "source_label": "World Bank WDI Population density",
        "parameter_text": "population density from World Development Indicators",
        "unit": "people_per_sq_km",
        "value_semantics": "direct_observed",
        "measurement_role": "direct_indicator",
        "confidence": 0.9,
    },
    {
        "indicator_code": "SP.URB.TOTL.IN.ZS",
        "canonical_name": "urbanization_pressure",
        "source_id": "world_bank_wdi_sp_urb_totl_in_zs",
        "source_label": "World Bank WDI Urban population share",
        "parameter_text": "urban population as percent of total population from World Development Indicators",
        "unit": "percent",
        "value_semantics": "direct_observed",
        "measurement_role": "direct_indicator",
        "confidence": 0.9,
    },
    {
        "indicator_code": "SH.XPD.CHEX.PC.CD",
        "canonical_name": "health_expenditure",
        "source_id": "world_bank_wdi_sh_xpd_chex_pc_cd",
        "source_label": "World Bank WDI Current health expenditure per capita",
        "parameter_text": "current health expenditure per capita from World Development Indicators",
        "unit": "usd_per_capita",
        "value_semantics": "direct_observed",
        "measurement_role": "direct_indicator",
        "confidence": 0.88,
    },
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _row_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "points", "source_manifest_rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _row_list(read_json(path, default=[]))


def _target_meta_by_name() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for module_id, module in PHASE3_MODULE_CONTRACT.items():
        for canonical_name in list(module.get("canonical_names") or []):
            row = meta.setdefault(
                str(canonical_name),
                {"module_ids": [], "transitions": [], "latent_blocks": []},
            )
            row["module_ids"].append(module_id)
            row["transitions"].extend(str(value) for value in list(module.get("transitions") or []))
            row["latent_blocks"].extend(str(value) for value in list(module.get("latent_blocks") or []))
    return {
        name: {
            "module_ids": sorted(set(row["module_ids"])),
            "transitions": sorted(set(row["transitions"])),
            "latent_blocks": sorted(set(row["latent_blocks"])),
        }
        for name, row in meta.items()
    }


def _candidate_block(canonical_name: str) -> str:
    meta = _target_meta_by_name().get(canonical_name) or {}
    blocks = [str(value) for value in list(meta.get("latent_blocks") or []) if str(value)]
    return blocks[0] if blocks else "unassigned"


def _source_url(row: Mapping[str, Any]) -> str:
    url = _clean_text(row.get("source_url") or row.get("url"))
    if url:
        return url
    organization = _clean_text(row.get("source_organization") or row.get("organization")).lower()
    source_id = _clean_text(row.get("source_id")).lower()
    if organization == "unaids" or source_id.startswith("unaids"):
        return UNAIDS_AIDSINFO_URL
    return ""


def _platform(row: Mapping[str, Any]) -> str:
    source_family = _clean_text(row.get("source_family")).lower()
    measurement_class = _clean_text(row.get("measurement_class")).lower()
    organization = _clean_text(row.get("source_organization") or row.get("organization")).lower()
    source_id = _clean_text(row.get("source_id")).lower()
    if source_family == "wdi_unaids_hiv" or source_id.startswith("wdi_") or measurement_class == "external_reference_wdi":
        return "world_bank_wdi"
    if organization == "unaids" or source_id.startswith("unaids"):
        return "unaids"
    if measurement_class == "program_observed_harp" or source_id.startswith(("curated_historical_harp", "doh_official")):
        return "doh_harp_hasp"
    return _clean_text(row.get("platform")) or "official_archive"


def _source_tier(row: Mapping[str, Any]) -> str:
    platform = _platform(row)
    if platform == "doh_harp_hasp":
        return "tier1_official_program_observation"
    if platform == "unaids":
        return "tier1_official_anchor"
    if platform == "world_bank_wdi":
        return "tier3_structured_repository"
    if platform == "fies":
        return "tier2_official_survey"
    return _clean_text(row.get("source_quality_tier") or row.get("source_tier")) or "official_archive"


def _allowed_use_hint(row: Mapping[str, Any], canonical_name: str) -> str:
    explicit = _clean_text(row.get("allowed_use_hint"))
    if explicit:
        return explicit
    platform = _platform(row)
    if platform in {"unaids", "world_bank_wdi"}:
        return "validation_or_auxiliary_only"
    if platform == "doh_harp_hasp":
        return "program_observation_head_only"
    if canonical_name in {"economic_access_constraint", "cash_instability"}:
        return "direct_determinant_covariate_candidate"
    return "prior_context_only"


def _measurement_semantics(canonical_name: str, row: Mapping[str, Any]) -> str:
    unit = _clean_text(row.get("unit")).lower()
    if canonical_name in {
        "documented_suppression",
        "suppression_outcomes",
        "viral_load_testing_coverage",
        "testing_rate",
        "prep_active_refill",
        "key_population_burden",
    }:
        return "flow_count" if "count" in unit else "proportion"
    if canonical_name in {"population_density", "clinics_per_capita"}:
        return "denominator_or_capacity_covariate"
    if canonical_name in {
        "testing_uptake",
        "viral_suppression_rate",
        "late_hiv_diagnosis_percent",
        "condom_use_barrier",
        "community_testing_reach",
        "stigma_barrier",
        "service_delivery_reach",
        "urbanization_pressure",
    }:
        return "proportion"
    return "determinant_covariate"


def _literature_ref(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_id = _clean_text(row.get("source_id")) or _stable_hash(row)[:20]
    title = _clean_text(row.get("source_label") or row.get("source_title") or row.get("title")) or source_id
    url = _source_url(row)
    detail = {
        "source_id": source_id,
        "title": title,
        "year": row.get("year"),
        "source_tier": _source_tier(row),
        "url": url or None,
        "doi": None,
        "pmid": None,
        "openalex_id": None,
    }
    return [detail] if url else []


def _candidate_row(
    *,
    source_row: Mapping[str, Any],
    canonical_name: str,
    parameter_text: str,
    value_semantics: str,
    measurement_role: str,
    confidence: float,
    extraction_method: str,
) -> dict[str, Any] | None:
    target_names = set(phase3_target_canonical_names())
    if canonical_name not in target_names:
        return None
    value = _safe_float(source_row.get("value"))
    if value is None:
        return None
    source_id = _clean_text(source_row.get("source_id")) or f"official_{_stable_hash(source_row)[:16]}"
    time_label = _clean_text(source_row.get("time") or source_row.get("year"))
    token = {
        "source_id": source_id,
        "canonical_name": canonical_name,
        "time": time_label,
        "geo": source_row.get("geo"),
        "value": value,
        "value_semantics": value_semantics,
    }
    row_hash = _stable_hash(token)
    geo = _clean_text(source_row.get("geo") or source_row.get("geography")) or "Philippines"
    region = _clean_text(source_row.get("region")) or "national"
    province = _clean_text(source_row.get("province"))
    title = _clean_text(source_row.get("source_label") or source_row.get("source_title") or source_row.get("title")) or source_id
    year = source_row.get("year")
    if year is None and len(time_label) >= 4 and time_label[:4].isdigit():
        year = int(time_label[:4])
    unit = _clean_text(source_row.get("unit")) or "annual_value"
    platform = _platform(source_row)
    allowed_use = _allowed_use_hint(source_row, canonical_name)
    evidence_span = (
        f"{title}; metric={_clean_text(source_row.get('metric_name'))}; "
        f"time={time_label}; value={value}; unit={unit}; semantics={value_semantics}"
    )
    return {
        "candidate_id": f"cand-official-bridge-{row_hash[:20]}",
        "document_id": f"doc-{source_id}",
        "source_id": source_id,
        "canonical_name": canonical_name,
        "candidate_block": _candidate_block(canonical_name),
        "candidate_text": f"{canonical_name} {geo} {time_label}",
        "parameter_text": parameter_text,
        "evidence_span": evidence_span,
        "extraction_method": extraction_method,
        "confidence": round(float(confidence), 6),
        "source_bank": "phase0_official_determinant_bridge",
        "source_tier": _source_tier(source_row),
        "source_title": title,
        "platform": platform,
        "query_geo_focus": "philippines",
        "observation_id": f"obs-official-bridge-{row_hash[:20]}",
        "geo": geo,
        "region": region,
        "province": "" if province == geo and region == "national" else province,
        "population": _clean_text(source_row.get("population")) or ("national" if region == "national" else region),
        "time": time_label,
        "year": year,
        "sex": _clean_text(source_row.get("sex")),
        "age_band": _clean_text(source_row.get("age_group") or source_row.get("age_band")),
        "kp_group": _kp_group(source_row),
        "geo_mentions": [geo],
        "literature_ref_details": _literature_ref(source_row),
        "linkage_targets": _transition_targets(canonical_name),
        "soft_ontology_tags": _soft_tags(canonical_name, platform),
        "soft_subparameter_hints": [canonical_name],
        "measurement_type": "rate" if unit == "percent" or "rate" in unit else "count",
        "denominator_type": "reported_metric_denominator",
        "normalization_basis": unit,
        "value_semantics": value_semantics,
        "measurement_semantics": _measurement_semantics(canonical_name, source_row),
        "measurement_role": measurement_role,
        "allowed_use_hint": allowed_use,
        "value": round(float(value), 6),
        "unit": unit,
        "source_url": _source_url(source_row),
        "source_organization": _clean_text(source_row.get("source_organization") or source_row.get("organization")),
        "source_metric_name": _clean_text(source_row.get("metric_name")),
        "is_anchor_eligible": False,
        "is_direct_measurement": True,
        "is_prior_only": allowed_use not in {"direct_determinant_covariate_candidate"},
    }


def _transition_targets(canonical_name: str) -> list[str]:
    meta = _target_meta_by_name().get(canonical_name) or {}
    return [str(value) for value in list(meta.get("transitions") or []) if str(value)]


def _soft_tags(canonical_name: str, platform: str) -> list[str]:
    tags = ["official_archive_bridge", platform]
    if canonical_name in {
        "key_population_burden",
        "condom_use_barrier",
        "prep_active_refill",
        "community_testing_reach",
    }:
        tags.append("incidence_pressure")
    if canonical_name in {"viral_suppression_rate", "suppression_outcomes", "documented_suppression", "viral_load_testing_coverage"}:
        tags.append("cascade_observation")
    if canonical_name in {"economic_access_constraint", "stigma_barrier"}:
        tags.append("structural_barrier")
    if canonical_name in {"population_density", "urbanization_pressure"}:
        tags.append("population_structure")
    if canonical_name in {"transport_friction", "travel_time", "mobility_network_mixing"}:
        tags.append("mobility_exposure")
    if canonical_name in {"clinics_per_capita", "service_delivery_reach", "philhealth_coverage", "health_expenditure"}:
        tags.append("service_capacity")
    return tags


def _kp_group(row: Mapping[str, Any]) -> str:
    text = f"{row.get('metric_name') or ''} {row.get('source_label') or ''}".lower()
    if "men_who_have_sex_with_men" in text or "men who have sex with men" in text:
        return "msm"
    if "transgender" in text:
        return "tgw"
    if "sex_workers" in text or "sex workers" in text:
        return "sex_workers"
    if "inject_drugs" in text or "inject drugs" in text:
        return "pwid"
    if "prisoners" in text:
        return "prisoners"
    return ""


def _metric_mappings(row: Mapping[str, Any]) -> list[tuple[str, str, str, str, float]]:
    metric = _clean_text(row.get("metric_name")).lower()
    label = _clean_text(row.get("source_label")).lower()
    text = f"{metric} {label}"
    mappings: list[tuple[str, str, str, str, float]] = []

    def add(name: str, parameter_text: str, semantics: str, role: str = "auxiliary_likelihood", confidence: float = 0.86) -> None:
        mappings.append((name, parameter_text, semantics, role, confidence))

    if metric == "late_hiv_diagnosis_percent":
        add("late_hiv_diagnosis_percent", "late HIV diagnosis with CD4 threshold from UNAIDS AIDSInfo", "reported_late_diagnosis_percent")
    if metric == "annual_hiv_tests_volume":
        add("testing_rate", "annual HIV test volume from UNAIDS AIDSInfo", "testing_volume_count")
    if metric == "unaids_known_status_share_percent" or "percent_of_people_living_with_hiv_who_know_their_status" in text:
        add("testing_uptake", "share of people living with HIV who know their status from UNAIDS AIDSInfo", "known_status_share_percent")
    if metric == "prep_people_receiving":
        add("prep_active_refill", "people receiving PrEP from UNAIDS AIDSInfo", "people_receiving_prep_count")
    if "condom_use" in metric:
        add(
            "condom_use_barrier",
            "condom-use metric from UNAIDS AIDSInfo; value is not inverted into a barrier score",
            "protective_condom_use_indicator_no_numeric_inversion",
            confidence=0.78,
        )
    if "coverage_of_hiv_prevention_programmes" in metric:
        add("community_testing_reach", "coverage of HIV prevention programmes among key populations from UNAIDS AIDSInfo", "prevention_programme_coverage_percent")
    if "hiv_testing_and_status_awareness" in metric:
        add("testing_uptake", "HIV testing and status awareness among key populations from UNAIDS AIDSInfo", "kp_testing_status_awareness")
    if "avoidance_of_health_care_because_of_stigma" in metric or "stigma_and_discrimination" in metric:
        add("stigma_barrier", "stigma/discrimination barrier metric from UNAIDS AIDSInfo", "stigma_barrier_indicator")
    if "size_estimate" in metric and any(term in metric for term in ("men_who_have_sex_with_men", "sex_workers", "transgender_people", "people_who_inject_drugs", "prisoners")):
        add("key_population_burden", "key-population size estimate from UNAIDS AIDSInfo", "kp_size_estimate_count")
    if "hiv_prevalence_among" in metric and any(term in metric for term in ("men_who_have_sex_with_men", "sex_workers", "transgender_people", "people_who_inject_drugs", "prisoners")):
        add("key_population_burden", "key-population HIV prevalence from UNAIDS AIDSInfo", "kp_hiv_prevalence_percent")
    if "antiretroviral_therapy_coverage_among" in metric or metric == "unaids_art_coverage_percent" or metric == "art_coverage_percent":
        add("service_delivery_reach", "ART coverage/service reach metric from UNAIDS or WDI", "art_coverage_percent")
    if "percent_of_people_on_art_who_achieve_viral_suppression" in text:
        add("viral_suppression_rate", "viral suppression among people on ART from UNAIDS AIDSInfo", "viral_suppression_rate_percent")
    if "percent_of_people_living_with_hiv_who_have_suppressed_viral_loads" in text:
        add("suppression_outcomes", "suppressed viral-load outcome among PLHIV from UNAIDS AIDSInfo", "population_viral_suppression_percent")
    if metric == "tested_for_viral_load":
        add(
            "viral_load_testing_coverage",
            "observed count tested for viral load in HARP/HASP cascade panel; denominator is not recomputed here",
            "viral_load_test_count_numerator",
            role="observation_head",
            confidence=0.92,
        )
    if metric == "virally_suppressed":
        add(
            "documented_suppression",
            "observed count documented virally suppressed in HARP/HASP cascade panel",
            "documented_suppression_count",
            role="observation_head",
            confidence=0.92,
        )
        add(
            "suppression_outcomes",
            "observed count virally suppressed in HARP/HASP cascade panel",
            "suppressed_count",
            role="observation_head",
            confidence=0.92,
        )
    return mappings


def _official_metric_rows(harp_archive_dir: Path) -> Iterable[dict[str, Any]]:
    for file_name in ("multinational_hiv_metric_rows.json", "wdi_hiv_rows.json", "observed_program_panel.json"):
        for row in _read_rows(harp_archive_dir / file_name):
            yield row


def _source_manifest_has_platform(run_dir: Path, platform: str) -> bool:
    source_rows = _read_rows(run_dir / "phase0" / "raw" / "source_manifest.json")
    target = platform.strip().lower()
    for row in source_rows:
        row_platform = _clean_text(row.get("platform")).lower()
        source_id = _clean_text(row.get("source_id")).lower()
        adapter_id = _clean_text(row.get("adapter_id")).lower()
        if row_platform == target and (adapter_id == target or source_id == target or source_id.startswith(f"{target}_")):
            return True
    return False


def _load_or_fetch_wdi_rows(*, cache_dir: Path, indicator_code: str) -> tuple[list[dict[str, Any]], bool]:
    cache_path = cache_dir / f"world_bank_{indicator_code.lower()}.json"
    if cache_path.exists():
        return list(read_json(cache_path, default=[]) or []), True
    url = f"https://api.worldbank.org/v2/country/PHL/indicator/{indicator_code}?format=json&per_page=20000"
    response = requests.get(url, timeout=(10.0, 45.0))
    response.raise_for_status()
    payload = response.json()
    rows = list(payload[1] or []) if isinstance(payload, list) and len(payload) >= 2 else []
    write_json(cache_path, rows)
    return rows, False


def _wdi_determinant_rows(*, run_dir: Path, cache_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not _source_manifest_has_platform(run_dir, "world_bank_wdi"):
        return [], []
    rows: list[dict[str, Any]] = []
    collector_rows: list[dict[str, Any]] = []
    for spec in WDI_DETERMINANT_SERIES:
        indicator_code = str(spec["indicator_code"])
        try:
            api_rows, cache_used = _load_or_fetch_wdi_rows(cache_dir=cache_dir, indicator_code=indicator_code)
        except Exception as exc:
            collector_rows.append(
                {
                    "collector": "official_bridge_world_bank_wdi",
                    "indicator_code": indicator_code,
                    "canonical_name": str(spec["canonical_name"]),
                    "status": "fetch_error",
                    "row_count": 0,
                    "cache_used": False,
                    "error": repr(exc),
                }
            )
            continue
        kept = 0
        for api_row in api_rows:
            year_text = _clean_text(api_row.get("date"))
            value = _safe_float(api_row.get("value"))
            if value is None or not year_text.isdigit() or int(year_text) < 2010:
                continue
            rows.append(
                {
                    "metric_name": str(spec["canonical_name"]),
                    "source_id": str(spec["source_id"]),
                    "source_label": str(spec["source_label"]),
                    "source_organization": "World Bank",
                    "source_url": WDI_DATABANK_URL,
                    "time": year_text,
                    "year": int(year_text),
                    "value": float(value),
                    "unit": str(spec["unit"]),
                    "geo": "Philippines",
                    "region": "national",
                    "platform": "world_bank_wdi",
                    "measurement_class": "official_wdi_determinant",
                    "allowed_use_hint": "direct_determinant_covariate_candidate",
                    "wdi_indicator_code": indicator_code,
                    "parameter_text": str(spec["parameter_text"]),
                    "value_semantics": str(spec["value_semantics"]),
                    "measurement_role": str(spec["measurement_role"]),
                    "confidence": float(spec["confidence"]),
                }
            )
            kept += 1
        collector_rows.append(
            {
                "collector": "official_bridge_world_bank_wdi",
                "indicator_code": indicator_code,
                "canonical_name": str(spec["canonical_name"]),
                "status": "ok",
                "row_count": kept,
                "cache_used": bool(cache_used),
            }
        )
    return rows, collector_rows


def _bridge_alias_rows(candidate_rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in candidate_rows:
        canonical = _clean_text(row.get("canonical_name"))
        platform = _clean_text(row.get("platform")).lower()
        if canonical == "household_expenditure_burden" and platform == "fies":
            bridged = dict(row)
            bridged["canonical_name"] = "economic_access_constraint"
            bridged["candidate_block"] = _candidate_block("economic_access_constraint")
            bridged["parameter_text"] = (
                "FIES household expenditure burden used as an economic-access proxy; "
                "the numeric value is the already-extracted official expenditure/income ratio"
            )
            bridged["value_semantics"] = "household_expenditure_burden_proxy_no_additional_transform"
            bridged["measurement_role"] = "direct_indicator"
            bridged["allowed_use_hint"] = "direct_determinant_covariate_candidate"
            bridged["source_bank"] = "phase0_official_determinant_bridge"
            bridged["candidate_id"] = f"cand-official-bridge-{_stable_hash(bridged)[:20]}"
            bridged["observation_id"] = f"obs-official-bridge-{_stable_hash({'obs': bridged})[:20]}"
            bridged["soft_subparameter_hints"] = ["economic_access_constraint", "household_expenditure_burden"]
            yield bridged
        elif canonical == "poverty_rate" and platform in {"world_bank_wdi", "psa_poverty", "psa"}:
            bridged = dict(row)
            bridged["canonical_name"] = "economic_access_constraint"
            bridged["candidate_block"] = _candidate_block("economic_access_constraint")
            bridged["parameter_text"] = "official poverty-rate row used as an economic-access proxy; numeric value is not transformed"
            bridged["value_semantics"] = "poverty_rate_proxy_no_additional_transform"
            bridged["measurement_role"] = "direct_indicator"
            bridged["allowed_use_hint"] = "direct_determinant_covariate_candidate"
            bridged["source_bank"] = "phase0_official_determinant_bridge"
            bridged["candidate_id"] = f"cand-official-bridge-{_stable_hash(bridged)[:20]}"
            bridged["observation_id"] = f"obs-official-bridge-{_stable_hash({'obs': bridged})[:20]}"
            bridged["soft_subparameter_hints"] = ["economic_access_constraint", "poverty_rate"]
            yield bridged
        elif canonical == "congestion_travel_time" and platform == "google_mobility":
            bridged = dict(row)
            bridged["canonical_name"] = "transport_friction"
            bridged["candidate_block"] = _candidate_block("transport_friction")
            bridged["parameter_text"] = (
                "Google transit-station mobility change used as a transport-friction proxy; "
                "the signed mobility-change value is not inverted or rescaled"
            )
            bridged["value_semantics"] = "transit_station_mobility_change_proxy_no_numeric_transform"
            bridged["measurement_role"] = "direct_indicator"
            bridged["allowed_use_hint"] = "direct_determinant_covariate_candidate"
            bridged["source_bank"] = "phase0_official_determinant_bridge"
            bridged["candidate_id"] = f"cand-official-bridge-{_stable_hash(bridged)[:20]}"
            bridged["observation_id"] = f"obs-official-bridge-{_stable_hash({'obs': bridged})[:20]}"
            bridged["soft_subparameter_hints"] = ["transport_friction", "congestion_travel_time"]
            yield bridged
        elif canonical.startswith("service_delivery_reach_") and platform == "philhealth_open_portal":
            bridged = dict(row)
            bridged["canonical_name"] = "service_delivery_reach"
            bridged["candidate_block"] = _candidate_block("service_delivery_reach")
            bridged["parameter_text"] = (
                "PhilHealth program-specific service delivery reach row collapsed to the Phase 3 service_delivery_reach determinant; "
                "the program-specific value is not rescaled"
            )
            bridged["value_semantics"] = f"{canonical}_proxy_no_numeric_transform"
            bridged["measurement_role"] = "direct_indicator"
            bridged["allowed_use_hint"] = "direct_determinant_covariate_candidate"
            bridged["source_bank"] = "phase0_official_determinant_bridge"
            bridged["candidate_id"] = f"cand-official-bridge-{_stable_hash(bridged)[:20]}"
            bridged["observation_id"] = f"obs-official-bridge-{_stable_hash({'obs': bridged})[:20]}"
            bridged["soft_subparameter_hints"] = ["service_delivery_reach", canonical]
            yield bridged
        elif canonical.startswith("clinics_per_capita_") and platform == "philhealth_open_portal":
            bridged = dict(row)
            bridged["canonical_name"] = "clinics_per_capita"
            bridged["candidate_block"] = _candidate_block("clinics_per_capita")
            bridged["parameter_text"] = (
                "PhilHealth program-specific facilities-per-capita row collapsed to the Phase 3 clinics_per_capita determinant; "
                "the program-specific value is not rescaled"
            )
            bridged["value_semantics"] = f"{canonical}_proxy_no_numeric_transform"
            bridged["measurement_role"] = "direct_indicator"
            bridged["allowed_use_hint"] = "direct_determinant_covariate_candidate"
            bridged["source_bank"] = "phase0_official_determinant_bridge"
            bridged["candidate_id"] = f"cand-official-bridge-{_stable_hash(bridged)[:20]}"
            bridged["observation_id"] = f"obs-official-bridge-{_stable_hash({'obs': bridged})[:20]}"
            bridged["soft_subparameter_hints"] = ["clinics_per_capita", canonical]
            yield bridged


def build_official_determinant_bridge_rows(
    *,
    run_dir: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    harp_archive_dir = run_dir / "harp_archive"
    extracted_dir = run_dir / "phase0" / "extracted"
    rows: list[dict[str, Any]] = []
    unmapped_metrics: Counter[str] = Counter()
    mapped_metric_counts: Counter[str] = Counter()
    collector_rows: list[dict[str, Any]] = []
    cache_dir = ensure_dir((out_dir or (run_dir / "phase0" / "evidence_ledger")) / "official_bridge_cache")
    wdi_rows, wdi_collectors = _wdi_determinant_rows(run_dir=run_dir, cache_dir=cache_dir)
    collector_rows.extend(wdi_collectors)
    for source_row in wdi_rows:
        candidate = _candidate_row(
            source_row=source_row,
            canonical_name=str(source_row["metric_name"]),
            parameter_text=str(source_row["parameter_text"]),
            value_semantics=str(source_row["value_semantics"]),
            measurement_role=str(source_row["measurement_role"]),
            confidence=float(source_row["confidence"]),
            extraction_method="official_bridge_world_bank_wdi_api",
        )
        if candidate is not None:
            rows.append(candidate)
            mapped_metric_counts[_clean_text(source_row.get("metric_name"))] += 1
    for source_row in _official_metric_rows(harp_archive_dir):
        mappings = _metric_mappings(source_row)
        if not mappings:
            metric = _clean_text(source_row.get("metric_name"))
            if metric:
                unmapped_metrics[metric] += 1
            continue
        for canonical_name, parameter_text, semantics, role, confidence in mappings:
            candidate = _candidate_row(
                source_row=source_row,
                canonical_name=canonical_name,
                parameter_text=parameter_text,
                value_semantics=semantics,
                measurement_role=role,
                confidence=confidence,
                extraction_method="official_archive_metric_bridge_exact_name_mapping",
            )
            if candidate is not None:
                rows.append(candidate)
                mapped_metric_counts[_clean_text(source_row.get("metric_name"))] += 1

    base_candidates = _read_rows(extracted_dir / "canonical_parameter_candidates.json")
    alias_rows = list(_bridge_alias_rows(base_candidates))
    rows.extend(alias_rows)

    destination = ensure_dir(out_dir or (run_dir / "phase0" / "evidence_ledger"))
    rows_path = destination / "official_determinant_candidate_rows.json"
    summary_path = destination / "official_determinant_candidate_rows_summary.json"
    write_json(rows_path, rows)
    by_canonical = Counter(str(row.get("canonical_name") or "unknown") for row in rows)
    by_platform = Counter(str(row.get("platform") or "unknown") for row in rows)
    by_allowed_use = Counter(str(row.get("allowed_use_hint") or "unknown") for row in rows)
    by_source_platform: dict[tuple[str, str], int] = Counter((str(row.get("platform") or "unknown"), str(row.get("source_id") or "")) for row in rows)
    compression_by_platform = []
    for platform in sorted(by_platform):
        source_count = len({source_id for plat, source_id in by_source_platform if plat == platform})
        raw_count = int(by_platform[platform])
        compression_by_platform.append(
            {
                "platform": platform,
                "candidate_row_count": raw_count,
                "source_level_count": source_count,
                "rows_per_source": round(raw_count / max(source_count, 1), 6),
            }
        )
    summary = {
        "schema_version": OFFICIAL_DETERMINANT_BRIDGE_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "candidate_row_count": len(rows),
        "alias_bridge_row_count": len(alias_rows),
        "metric_bridge_row_count": len(rows) - len(alias_rows),
        "collector_rows": collector_rows,
        "canonical_name_counts": dict(sorted(by_canonical.items())),
        "platform_counts": dict(sorted(by_platform.items())),
        "row_compression_by_platform": compression_by_platform,
        "allowed_use_hint_counts": dict(sorted(by_allowed_use.items())),
        "mapped_metric_counts": dict(sorted(mapped_metric_counts.items())),
        "unmapped_metric_counts_top25": dict(unmapped_metrics.most_common(25)),
        "artifact_paths": {"candidate_rows": str(rows_path), "summary_json": str(summary_path)},
        "contract_note": (
            "This bridge makes official rows visible to the determinant ledger. "
            "UNAIDS/WDI model-estimate rows remain validation_or_auxiliary_only; HARP cascade rows remain observation-head evidence."
        ),
    }
    write_json(summary_path, summary)
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build official-source determinant bridge rows for the Phase 3 contract.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = build_official_determinant_bridge_rows(run_dir=args.run_dir, out_dir=args.out_dir)
    print(json.dumps(summary.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
