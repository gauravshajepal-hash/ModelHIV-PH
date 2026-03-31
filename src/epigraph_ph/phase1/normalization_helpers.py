from __future__ import annotations

import re
from typing import Any

from epigraph_ph.geography import geo_resolution_label, infer_region_code

SEX_TOKENS = {
    "male": [" male ", " men ", " man ", " boys ", " msm "],
    "female": [" female ", " women ", " woman ", " girls ", " fsw "],
}
AGE_TOKENS = {
    "15_24": ["15-24", "15 24", "adolescent", "youth", "young people"],
    "25_34": ["25-34", "25 34"],
    "35_49": ["35-49", "35 49"],
    "50_plus": ["50+", "older adult", "aging"],
}
KP_TOKENS = {
    "msm": [" msm ", "men who have sex with men"],
    "tgw": [" tgw ", "transgender women", "trans women"],
    "fsw": [" fsw ", "female sex worker", "sex worker"],
    "clients_fsw": ["client of sex worker", "clients of female sex workers"],
    "pwid": [" pwid ", "people who inject drugs", "inject drugs"],
    "non_kp_partners": ["partner", "spouse", "non-key-population partner"],
}
DOMAIN_HINTS = {
    "economics": ["econom", "poverty", "income", "afford", "cash", "insurance"],
    "logistics": ["transport", "mobility", "travel", "supply chain", "commute", "remoteness"],
    "behavior": ["stigma", "behavior", "norm", "risk", "health seeking", "condom"],
    "population": ["population", "demograph", "urbanization", "migration", "fertility", "age structure"],
    "biology": ["cd4", "viral", "immune", "drug resistance", "antiviral", "reservoir"],
    "policy": ["policy", "governance", "implementation", "service delivery", "program"],
}
PATHWAY_HINTS = {
    "prevention_access": ["prevent", "condom", "prophylaxis", "access"],
    "testing_uptake": ["testing", "screening", "diagnos", "uptake"],
    "linkage_to_care": ["linkage", "referral", "clinic", "care"],
    "retention_adherence": ["retention", "adherence", "loss to follow up", "dropout"],
    "suppression_outcomes": ["suppression", "viral load", "treatment success", "art"],
    "mobility_network_mixing": ["mobility", "migration", "network", "mixing"],
    "health_system_reach": ["facility", "telehealth", "coverage", "health system"],
    "biological_progression": ["cd4", "immune", "viral", "reservoir", "drug resistance"],
}


def is_valid_observation_year(year: int) -> bool:
    return 1980 <= year <= 2026


def safe_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def normalize_unit(unit: str) -> str:
    unit = (unit or "").strip().lower()
    mapping = {
        "%": "percent",
        "percent": "percent",
        "usd": "currency_usd",
        "php": "currency_php",
        "peso": "currency_php",
        "pesos": "currency_php",
        "people": "count_people",
        "cases": "count_cases",
        "deaths": "count_deaths",
        "million": "count_million",
        "billion": "count_billion",
    }
    return mapping.get(unit, unit or "unitless")


def normalize_numeric_value(value: float | None, normalized_unit: str) -> float | None:
    if value is None:
        return None
    if normalized_unit == "percent":
        return value / 100.0
    if normalized_unit == "count_million":
        return value * 1_000_000.0
    if normalized_unit == "count_billion":
        return value * 1_000_000_000.0
    return value


def geo_resolution(geo: str) -> str:
    return geo_resolution_label(geo)


def infer_region(geo: str, text: str) -> str:
    return infer_region_code(geo, text)


def time_components(value: str) -> tuple[str, str, int | None, int | None]:
    value = (value or "").strip()
    if len(value) == 4 and value.isdigit():
        year = int(value)
        return ("annual", value, year, None) if is_valid_observation_year(year) else ("unknown", "unknown", None, None)
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if is_valid_observation_year(year) and 1 <= month <= 12:
            return "monthly", value, year, month
        return "unknown", "unknown", None, None
    return "unknown", value, None, None


def text_blob(row: dict[str, Any]) -> str:
    pieces = [
        row.get("candidate_text"),
        row.get("parameter_text"),
        row.get("canonical_name"),
        row.get("geo"),
        " ".join(row.get("soft_ontology_tags") or []),
        " ".join(row.get("soft_subparameter_hints") or []),
        " ".join(row.get("linkage_targets") or []),
    ]
    return f" {' '.join(str(piece or '') for piece in pieces)} ".lower()


def infer_from_tokens(blob: str, token_map: dict[str, list[str]], default: str = "") -> str:
    for label, tokens in token_map.items():
        if any(token in blob for token in tokens):
            return label
    return default


def infer_domain_family(row: dict[str, Any], blob: str) -> str:
    explicit = list(row.get("soft_ontology_tags") or [])
    if explicit:
        return explicit[0]
    return infer_from_tokens(blob, DOMAIN_HINTS, default="mixed")


def infer_pathway_family(row: dict[str, Any], blob: str) -> str:
    explicit = list(row.get("linkage_targets") or [])
    if explicit:
        return explicit[0]
    return infer_from_tokens(blob, PATHWAY_HINTS, default="mixed")


def evidence_class(row: dict[str, Any]) -> str:
    source_bank = str(row.get("source_bank") or "")
    canonical_name = str(row.get("canonical_name") or "")
    linkage_targets = {str(item) for item in (row.get("linkage_targets") or [])}
    main_predictive = {"prevention_access", "testing_uptake", "linkage_to_care", "retention_adherence", "suppression_outcomes"}
    if source_bank == "phase0_extracted" and row.get("is_direct_measurement"):
        return "observed_numeric"
    if source_bank == "phase0_extracted":
        return "numeric_prior"
    if source_bank == "phase0_chunk_soft_candidates":
        return "hiv_literature_seed" if canonical_name in main_predictive or bool(linkage_targets & main_predictive) else "upstream_literature_seed"
    if source_bank == "phase0_wide_sweep_hiv_direct":
        return "hiv_literature_seed"
    if source_bank == "phase0_wide_sweep_upstream_determinants":
        return "upstream_literature_seed"
    return "literature_seed"


def evidence_weight(row: dict[str, Any], evidence_class_value: str, evidence_class_weights: dict[str, Any]) -> float:
    weight = float(evidence_class_weights["base"])
    if evidence_class_value == "observed_numeric":
        weight = float(evidence_class_weights["observed_numeric"])
    elif evidence_class_value == "numeric_prior":
        weight = float(evidence_class_weights["numeric_prior"])
    elif evidence_class_value == "hiv_literature_seed":
        weight = float(evidence_class_weights["hiv_literature_seed"])
    elif evidence_class_value == "upstream_literature_seed":
        weight = float(evidence_class_weights["upstream_literature_seed"])
    if row.get("is_anchor_eligible"):
        weight += float(evidence_class_weights["anchor_bonus"])
    if row.get("is_direct_measurement"):
        weight += float(evidence_class_weights["direct_measurement_bonus"])
    return min(float(evidence_class_weights["ceiling"]), round(weight, 4))


def source_reliability_class(row: dict[str, Any], evidence_class_value: str) -> str:
    source_bank = str(row.get("source_bank") or "").lower()
    source_tier = str(row.get("source_tier") or "").lower()
    if row.get("is_anchor_eligible") and source_bank == "phase0_extracted":
        return "official_routine_anchor"
    if source_bank == "phase0_chunk_soft_candidates":
        if "official_survey" in source_tier:
            return "official_survey"
        if "scientific_literature" in source_tier:
            return "scientific_qualitative_study"
        if "structured_repository" in source_tier:
            return "structured_repository"
        return "literature_only_seed"
    if evidence_class_value == "observed_numeric":
        return "official_survey" if any(token in source_bank for token in ("survey", "ndhs", "ihbss")) else "scientific_numeric_study"
    if evidence_class_value == "numeric_prior":
        return "structured_repository" if "repository" in source_bank else "scientific_numeric_study"
    if source_bank in {"phase0_wide_sweep_hiv_direct", "phase0_wide_sweep_upstream_determinants"}:
        details = row.get("literature_ref_details") or []
        if any(str(detail.get("kind") or "").lower() == "quantitative" for detail in details if isinstance(detail, dict)):
            return "scientific_numeric_study"
        return "literature_only_seed"
    if "repository" in source_bank:
        return "structured_repository"
    if row.get("is_direct_measurement"):
        return "scientific_numeric_study"
    if row.get("is_prior_only"):
        return "proxy_inferred"
    return "literature_only_seed"


def bias_fields(
    row: dict[str, Any],
    reliability_class: str,
    time_resolution: str,
    geo_resolution_value: str,
    *,
    reliability_class_rules: dict[str, Any],
    bias_class_thresholds: dict[str, Any],
    spatial_relevance_by_geo: dict[str, Any],
    bias_penalty_mix: dict[str, Any],
) -> dict[str, Any]:
    profile = reliability_class_rules[reliability_class]
    measurement_bias_class = (
        "low_bias"
        if profile["measurement"] >= float(bias_class_thresholds["measurement_low"])
        else ("moderate_bias" if profile["measurement"] >= float(bias_class_thresholds["measurement_moderate"]) else "high_bias")
    )
    sampling_bias_class = (
        "low_sampling_bias"
        if profile["sampling"] >= float(bias_class_thresholds["sampling_low"])
        else ("moderate_sampling_bias" if profile["sampling"] >= float(bias_class_thresholds["sampling_moderate"]) else "high_sampling_bias")
    )
    if time_resolution == "monthly":
        reporting_delay_class = "short_delay"
    elif time_resolution == "annual":
        reporting_delay_class = "moderate_delay"
    else:
        reporting_delay_class = "unknown_delay"
    spatial_relevance = float(spatial_relevance_by_geo.get(geo_resolution_value, spatial_relevance_by_geo["unknown"]))
    temporal_freshness = profile["delay"]
    return {
        "source_reliability_class": reliability_class,
        "measurement_bias_class": measurement_bias_class,
        "sampling_bias_class": sampling_bias_class,
        "reporting_delay_class": reporting_delay_class,
        "promotion_eligibility_hint": profile["hint"],
        "source_tier_weight": round(float(profile["tier"]), 4),
        "measurement_quality_weight": round(float(profile["measurement"]), 4),
        "temporal_freshness_weight": round(float(temporal_freshness), 4),
        "spatial_relevance_weight": round(float(spatial_relevance), 4),
        "replication_weight": 0.0,
        "bias_penalty": round(
            float(
                max(
                    0.0,
                    1.0
                    - (
                        float(bias_penalty_mix["measurement"]) * profile["measurement"]
                        + float(bias_penalty_mix["sampling"]) * profile["sampling"]
                        + float(bias_penalty_mix["temporal_freshness"]) * temporal_freshness
                        + float(bias_penalty_mix["spatial_relevance"]) * spatial_relevance
                    ),
                )
            ),
            4,
        ),
    }


__all__ = [
    "AGE_TOKENS",
    "bias_fields",
    "DOMAIN_HINTS",
    "evidence_class",
    "evidence_weight",
    "geo_resolution",
    "infer_region",
    "infer_domain_family",
    "infer_from_tokens",
    "infer_pathway_family",
    "is_valid_observation_year",
    "KP_TOKENS",
    "normalize_numeric_value",
    "normalize_unit",
    "safe_float",
    "SEX_TOKENS",
    "source_reliability_class",
    "text_blob",
    "time_components",
]
