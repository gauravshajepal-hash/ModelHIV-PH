from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.geography import geo_resolution_label, infer_philippines_geo, infer_region_code, is_national_geo, normalize_geo_label
from epigraph_ph.phase0.models import Phase0BackendStatus, Phase0ManifestArtifact
from epigraph_ph.runtime import (
    RunContext,
    choose_torch_device,
    detect_backends,
    ensure_dir,
    load_tensor_artifact,
    read_json,
    save_tensor_artifact,
    to_numpy,
    to_torch_tensor,
    utc_now_iso,
    write_ground_truth_package,
    write_json,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

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

_HIV_PLUGIN = get_disease_plugin("hiv")


def _phase1_cfg() -> dict[str, Any]:
    return dict((_HIV_PLUGIN.constraint_settings or {}).get("phase1", {}) or {})


def _phase1_required(key: str) -> Any:
    cfg = _phase1_cfg()
    if key not in cfg:
        raise KeyError(f"Missing HIV phase1 constraint setting: {key}")
    return cfg[key]


def _phase1_required_section(key: str) -> dict[str, Any]:
    value = _phase1_required(key)
    if not isinstance(value, dict):
        raise TypeError(f"HIV phase1 constraint setting '{key}' must be a mapping")
    return dict(value)


def _is_valid_observation_year(year: int) -> bool:
    return 1980 <= year <= 2026


def _safe_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _normalize_unit(unit: str) -> str:
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


def _normalize_numeric_value(value: float | None, normalized_unit: str) -> float | None:
    if value is None:
        return None
    if normalized_unit == "percent":
        return value / 100.0
    if normalized_unit == "count_million":
        return value * 1_000_000.0
    if normalized_unit == "count_billion":
        return value * 1_000_000_000.0
    return value


def _geo_resolution(geo: str) -> str:
    return geo_resolution_label(geo)


def _infer_region(geo: str, text: str) -> str:
    return infer_region_code(geo, text)


def _time_components(value: str) -> tuple[str, str, int | None, int | None]:
    value = (value or "").strip()
    if len(value) == 4 and value.isdigit():
        year = int(value)
        return ("annual", value, year, None) if _is_valid_observation_year(year) else ("unknown", "unknown", None, None)
    match = re.fullmatch(r"(\d{4})-(\d{2})", value)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if _is_valid_observation_year(year) and 1 <= month <= 12:
            return "monthly", value, year, month
        return "unknown", "unknown", None, None
    return "unknown", value, None, None


def _text_blob(row: dict[str, Any]) -> str:
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


def _infer_from_tokens(blob: str, token_map: dict[str, list[str]], default: str = "") -> str:
    for label, tokens in token_map.items():
        if any(token in blob for token in tokens):
            return label
    return default


def _infer_domain_family(row: dict[str, Any], blob: str) -> str:
    explicit = list(row.get("soft_ontology_tags") or [])
    if explicit:
        return explicit[0]
    return _infer_from_tokens(blob, DOMAIN_HINTS, default="mixed")


def _infer_pathway_family(row: dict[str, Any], blob: str) -> str:
    explicit = list(row.get("linkage_targets") or [])
    if explicit:
        return explicit[0]
    return _infer_from_tokens(blob, PATHWAY_HINTS, default="mixed")


def _evidence_class(row: dict[str, Any]) -> str:
    if row.get("source_bank") == "phase0_extracted" and row.get("is_direct_measurement"):
        return "observed_numeric"
    if row.get("source_bank") == "phase0_extracted":
        return "numeric_prior"
    if row.get("source_bank") == "phase0_wide_sweep_hiv_direct":
        return "hiv_literature_seed"
    if row.get("source_bank") == "phase0_wide_sweep_upstream_determinants":
        return "upstream_literature_seed"
    return "literature_seed"


def _evidence_weight(row: dict[str, Any], evidence_class: str) -> float:
    cfg = _phase1_required_section("evidence_class_weights")
    weight = float(cfg["base"])
    if evidence_class == "observed_numeric":
        weight = float(cfg["observed_numeric"])
    elif evidence_class == "numeric_prior":
        weight = float(cfg["numeric_prior"])
    elif evidence_class == "hiv_literature_seed":
        weight = float(cfg["hiv_literature_seed"])
    elif evidence_class == "upstream_literature_seed":
        weight = float(cfg["upstream_literature_seed"])
    if row.get("is_anchor_eligible"):
        weight += float(cfg["anchor_bonus"])
    if row.get("is_direct_measurement"):
        weight += float(cfg["direct_measurement_bonus"])
    return min(float(cfg["ceiling"]), round(weight, 4))


def _source_reliability_class(row: dict[str, Any], evidence_class: str) -> str:
    source_bank = str(row.get("source_bank") or "").lower()
    if row.get("is_anchor_eligible") and source_bank == "phase0_extracted":
        return "official_routine_anchor"
    if evidence_class == "observed_numeric":
        return "official_survey" if any(token in source_bank for token in ("survey", "ndhs", "ihbss")) else "scientific_numeric_study"
    if evidence_class == "numeric_prior":
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


def _bias_fields(row: dict[str, Any], reliability_class: str, time_resolution: str, geo_resolution: str) -> dict[str, Any]:
    profile = _phase1_required_section("reliability_class_rules")[reliability_class]
    thresholds = _phase1_required_section("bias_class_thresholds")
    spatial_cfg = _phase1_required_section("spatial_relevance_by_geo")
    penalty_mix = _phase1_required_section("bias_penalty_mix")
    measurement_bias_class = "low_bias" if profile["measurement"] >= float(thresholds["measurement_low"]) else ("moderate_bias" if profile["measurement"] >= float(thresholds["measurement_moderate"]) else "high_bias")
    sampling_bias_class = "low_sampling_bias" if profile["sampling"] >= float(thresholds["sampling_low"]) else ("moderate_sampling_bias" if profile["sampling"] >= float(thresholds["sampling_moderate"]) else "high_sampling_bias")
    if time_resolution == "monthly":
        reporting_delay_class = "short_delay"
    elif time_resolution == "annual":
        reporting_delay_class = "moderate_delay"
    else:
        reporting_delay_class = "unknown_delay"
    spatial_relevance = float(spatial_cfg.get(geo_resolution, spatial_cfg["unknown"]))
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
                        float(penalty_mix["measurement"]) * profile["measurement"]
                        + float(penalty_mix["sampling"]) * profile["sampling"]
                        + float(penalty_mix["temporal_freshness"]) * temporal_freshness
                        + float(penalty_mix["spatial_relevance"]) * spatial_relevance
                    ),
                )
            ),
            4,
        ),
    }


def _tensor_preprocess(aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if aligned.size == 0:
        empty = np.zeros_like(aligned)
        return empty, empty, empty, {"density_scale_default": True}
    if torch is None:
        denominator = np.ones_like(aligned, dtype=np.float32)
        density = aligned / np.where(denominator == 0, 1.0, denominator)
        log_tensor = np.log1p(np.clip(density, a_min=0.0, a_max=None))
        median = np.median(log_tensor, axis=(0, 1), keepdims=True)
        q75 = np.quantile(log_tensor, 0.75, axis=(0, 1), keepdims=True)
        q25 = np.quantile(log_tensor, 0.25, axis=(0, 1), keepdims=True)
        iqr = np.where((q75 - q25) == 0, 1.0, q75 - q25)
        standardized = np.clip((log_tensor - median) / iqr, -6.0, 6.0)
        return density.astype(np.float32), denominator.astype(np.float32), standardized.astype(np.float32), {"density_scale_default": True}
    device = choose_torch_device(prefer_gpu=True)
    tensor = to_torch_tensor(aligned, device=device, dtype=torch.float32)
    denominator = torch.ones_like(tensor, dtype=torch.float32)
    density = tensor / torch.clamp(denominator, min=1.0)
    log_tensor = torch.log1p(torch.clamp(density, min=0.0))
    flat = log_tensor.reshape(-1, log_tensor.shape[-1])
    median = torch.quantile(flat, 0.5, dim=0)
    q75 = torch.quantile(flat, 0.75, dim=0)
    q25 = torch.quantile(flat, 0.25, dim=0)
    iqr = torch.clamp(q75 - q25, min=1e-6)
    standardized = (log_tensor - median.view(1, 1, -1)) / iqr.view(1, 1, -1)
    standardized = torch.clamp(standardized, min=-6.0, max=6.0)
    return (
        to_numpy(density).astype(np.float32),
        to_numpy(denominator).astype(np.float32),
        to_numpy(standardized).astype(np.float32),
        {"density_scale_default": True, "torch_device": device},
    )


def run_phase1_build(*, run_id: str, plugin_id: str, profile: str = "legacy") -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase1_dir = ensure_dir(ctx.run_dir / "phase1")
    registry_path = ctx.run_dir / "registry" / "subparameter_registry.json"
    registry = read_json(registry_path, default={})
    subparameters = registry.get("subparameters", [])
    alignment_summary = read_json(ctx.run_dir / "phase0" / "extracted" / "alignment_summary.json", default={})
    aligned_tensor = load_tensor_artifact(ctx.run_dir / "phase0" / "extracted" / "aligned_tensor.npz")

    density_tensor, denominator_tensor, standardized_tensor, preprocess_meta = _tensor_preprocess(aligned_tensor)
    axis_catalogs = {
        "province": alignment_summary.get("province_axis", []),
        "month": alignment_summary.get("month_axis", []),
        "canonical_name": alignment_summary.get("canonical_name_axis", []),
    }

    normalized_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    extra_axis_values: dict[str, set[str]] = defaultdict(set)
    canonical_source_counts: Counter[tuple[str, str]] = Counter()
    catalog_rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "canonical_name": "",
            "row_count": 0,
            "numeric_row_count": 0,
            "model_numeric_row_count": 0,
            "source_banks": Counter(),
            "units": Counter(),
            "geo_resolutions": Counter(),
            "regions": Counter(),
            "time_resolutions": Counter(),
            "domain_families": Counter(),
            "pathway_families": Counter(),
            "soft_ontology_tags": Counter(),
            "linkage_targets": Counter(),
            "evidence_classes": Counter(),
        }
    )

    for row in subparameters:
        raw_numeric_value = _safe_float(row.get("value"))
        normalized_unit = _normalize_unit(str(row.get("unit") or ""))
        model_numeric_value = _normalize_numeric_value(raw_numeric_value, normalized_unit)
        blob = _text_blob(row)
        time_resolution, normalized_time, year, month = _time_components(str(row.get("time") or ""))
        sex = row.get("sex") or _infer_from_tokens(blob, SEX_TOKENS, default="")
        age_band = row.get("age_band") or _infer_from_tokens(blob, AGE_TOKENS, default="")
        kp_group = row.get("kp_group") or _infer_from_tokens(blob, KP_TOKENS, default="remaining_population")
        geo = normalize_geo_label(str(row.get("geo") or ""), default_country_focus="philippines" in blob) or ("Philippines" if "philippines" in blob else "")
        geo_match = infer_philippines_geo(f"{geo} {blob}", default_country_focus="philippines" in blob)
        geo_resolution = _geo_resolution(str(geo))
        region = str(row.get("region") or geo_match.region or _infer_region(str(geo), blob))
        province = str(row.get("province") or geo_match.province or (geo if geo_resolution in {"province", "city"} else ""))
        domain_family = _infer_domain_family(row, blob)
        pathway_family = _infer_pathway_family(row, blob)
        evidence_class = _evidence_class(row)
        evidence_weight = _evidence_weight(row, evidence_class)
        reliability_class = _source_reliability_class(row, evidence_class)
        bias_fields = _bias_fields(row, reliability_class, time_resolution, geo_resolution)
        normalized = {
            "normalized_id": row.get("subparameter_id") or row.get("candidate_id"),
            "canonical_name": row.get("canonical_name") or "unknown",
            "candidate_text": row.get("candidate_text") or row.get("parameter_text") or "",
            "raw_numeric_value": raw_numeric_value,
            "model_numeric_value": model_numeric_value,
            "raw_value": row.get("value"),
            "original_unit": row.get("unit") or "",
            "normalized_unit": normalized_unit,
            "geo": geo,
            "geo_resolution": geo_resolution,
            "country": "Philippines" if geo_resolution in {"national", "region", "province", "city"} else ("global" if geo_resolution == "global" else ""),
            "region": region,
            "province": province,
            "time": normalized_time,
            "time_resolution": time_resolution,
            "year": year,
            "month": month,
            "sex": sex,
            "age_band": age_band,
            "kp_group": kp_group,
            "domain_family": domain_family,
            "pathway_family": pathway_family,
            "source_bank": row.get("source_bank") or "",
            "source_id": row.get("source_id") or "",
            "evidence_class": evidence_class,
            "evidence_weight": evidence_weight,
            "is_anchor_eligible": bool(row.get("is_anchor_eligible")),
            "is_direct_measurement": bool(row.get("is_direct_measurement")),
            "is_prior_only": bool(row.get("is_prior_only")),
            "soft_ontology_tags": list(row.get("soft_ontology_tags") or []),
            "soft_subparameter_hints": list(row.get("soft_subparameter_hints") or []),
            "linkage_targets": list(row.get("linkage_targets") or []),
            "literature_ref_details": list(row.get("literature_ref_details") or []),
            **bias_fields,
        }
        normalized_rows.append(normalized)
        canonical_source_counts[(normalized["canonical_name"], str(normalized["source_id"]))] += 1
        roll = catalog_rollup[normalized["canonical_name"]]
        roll["canonical_name"] = normalized["canonical_name"]
        roll["row_count"] += 1
        roll["numeric_row_count"] += 1 if raw_numeric_value is not None else 0
        roll["model_numeric_row_count"] += 1 if model_numeric_value is not None else 0
        roll["source_banks"][normalized["source_bank"]] += 1
        roll["units"][normalized["normalized_unit"]] += 1
        roll["geo_resolutions"][normalized["geo_resolution"]] += 1
        if normalized["region"]:
            roll["regions"][normalized["region"]] += 1
        roll["time_resolutions"][normalized["time_resolution"]] += 1
        roll["domain_families"][normalized["domain_family"]] += 1
        roll["pathway_families"][normalized["pathway_family"]] += 1
        roll["evidence_classes"][normalized["evidence_class"]] += 1
        for tag in normalized["soft_ontology_tags"]:
            roll["soft_ontology_tags"][tag] += 1
        for target in normalized["linkage_targets"]:
            roll["linkage_targets"][target] += 1
        for axis_name in ("geo_resolution", "geo", "country", "region", "time_resolution", "sex", "age_band", "kp_group", "domain_family", "pathway_family", "evidence_class", "source_bank", "normalized_unit"):
            value = normalized.get(axis_name)
            if value not in {"", None}:
                extra_axis_values[axis_name].add(str(value))

    for row in normalized_rows:
        key = (row["canonical_name"], str(row.get("source_id") or ""))
        replication_cfg = _phase1_required_section("replication_weight")
        replication_weight = min(
            float(replication_cfg["ceiling"]),
            float(replication_cfg["base"]) + float(replication_cfg["step"]) * canonical_source_counts[key],
        )
        row["replication_weight"] = round(float(replication_weight), 4)
        row["bias_penalty"] = round(
            float(
                max(
                    0.0,
                    row["bias_penalty"]
                    - float(replication_cfg["bias_penalty_discount"])
                    * min(
                        float(replication_cfg["ceiling"]),
                        max(0.0, replication_weight - float(replication_cfg["bias_penalty_reference"])),
                    ),
                )
            ),
            4,
        )

    province_axis = [normalize_geo_label(str(item), default_country_focus=is_national_geo(str(item))) or str(item) for item in (axis_catalogs["province"] or ["Philippines"])]
    province_axis = list(dict.fromkeys(province_axis))
    axis_catalogs["province"] = province_axis
    month_axis = axis_catalogs["month"] or ["unknown"]
    canonical_axis = axis_catalogs["canonical_name"] or ["numeric_observation"]
    for pi, province in enumerate(province_axis):
        for ti, month in enumerate(month_axis):
            for fi, canonical_name in enumerate(canonical_axis):
                tensor_rows.append(
                    {
                        "tensor_row_id": f"{province}:{month}:{canonical_name}",
                        "canonical_name": canonical_name,
                        "model_numeric_value": float(standardized_tensor[pi, ti, fi]),
                        "raw_numeric_value": float(aligned_tensor[pi, ti, fi]),
                        "normalized_unit": "standard_score",
                        "geo_resolution": geo_resolution_label(province),
                        "geo": province,
                        "country": "Philippines" if province != "global" else "global",
                        "region": infer_region_code(province),
                        "province": province,
                        "time_resolution": "monthly",
                        "time": month,
                        "year": int(month.split("-")[0]) if re.fullmatch(r"\d{4}-\d{2}", month) else None,
                        "month": int(month.split("-")[1]) if re.fullmatch(r"\d{4}-\d{2}", month) else None,
                        "sex": "",
                        "age_band": "",
                        "kp_group": "remaining_population",
                        "domain_family": next(iter(catalog_rollup.get(canonical_name, {}).get("domain_families", {"mixed": 1})), "mixed"),
                        "pathway_family": next(iter(catalog_rollup.get(canonical_name, {}).get("pathway_families", {"mixed": 1})), "mixed"),
                        "evidence_class": "phase1_standardized_tensor",
                        "evidence_weight": 1.0,
                        "source_bank": "phase1_standardized_tensor",
                    }
                )

    parameter_catalog = []
    for canonical_name, roll in sorted(catalog_rollup.items()):
        parameter_catalog.append(
            {
                "canonical_name": canonical_name,
                "row_count": roll["row_count"],
                "numeric_row_count": roll["numeric_row_count"],
                "model_numeric_row_count": roll["model_numeric_row_count"],
                "source_banks": dict(roll["source_banks"]),
                "units": dict(roll["units"]),
                "geo_resolutions": dict(roll["geo_resolutions"]),
                "regions": dict(roll["regions"]),
                "time_resolutions": dict(roll["time_resolutions"]),
                "domain_families": dict(roll["domain_families"]),
                "pathway_families": dict(roll["pathway_families"]),
                "soft_ontology_tags": dict(roll["soft_ontology_tags"]),
                "linkage_targets": dict(roll["linkage_targets"]),
                "evidence_classes": dict(roll["evidence_classes"]),
            }
        )

    axis_catalogs.update({axis: sorted(values) for axis, values in sorted(extra_axis_values.items())})
    normalization_report = {
        "normalized_row_count": len(normalized_rows),
        "tensor_row_count": len(tensor_rows),
        "model_numeric_coverage": round(sum(1 for row in normalized_rows if row["model_numeric_value"] is not None) / len(normalized_rows), 4) if normalized_rows else 0.0,
        "direct_measurement_count": sum(1 for row in normalized_rows if row["is_direct_measurement"]),
        "anchor_eligible_count": sum(1 for row in normalized_rows if row["is_anchor_eligible"]),
        "source_reliability_counts": dict(Counter(row["source_reliability_class"] for row in normalized_rows)),
        "promotion_hint_counts": dict(Counter(row["promotion_eligibility_hint"] for row in normalized_rows)),
        "domain_family_counts": dict(Counter(row["domain_family"] for row in normalized_rows)),
        "pathway_family_counts": dict(Counter(row["pathway_family"] for row in normalized_rows)),
        "kp_group_counts": dict(Counter(row["kp_group"] for row in normalized_rows if row["kp_group"])),
        "sex_counts": dict(Counter(row["sex"] for row in normalized_rows if row["sex"])),
        "age_band_counts": dict(Counter(row["age_band"] for row in normalized_rows if row["age_band"])),
        "preprocess_meta": preprocess_meta,
    }

    backend_map = detect_backends()
    standardized_artifact = save_tensor_artifact(
        array=to_torch_tensor(standardized_tensor, device=choose_torch_device(prefer_gpu=True), dtype=torch.float32) if torch is not None else standardized_tensor,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="standardized_tensor",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=True) if torch is not None else "cpu",
        notes=["phase1_standardized_tensor"],
    )
    denominator_artifact = save_tensor_artifact(
        array=to_torch_tensor(denominator_tensor, device=choose_torch_device(prefer_gpu=False), dtype=torch.float32) if torch is not None else denominator_tensor,
        axis_names=["province", "month", "canonical_name"],
        artifact_dir=phase1_dir,
        stem="denominator_tensor",
        backend="torch" if torch is not None else "numpy",
        device=choose_torch_device(prefer_gpu=False) if torch is not None else "cpu",
        notes=["phase1_denominator_tensor"],
    )

    write_json(phase1_dir / "normalized_subparameters.json", normalized_rows)
    write_json(phase1_dir / "parameter_catalog.json", parameter_catalog)
    write_json(phase1_dir / "axis_catalogs.json", axis_catalogs)
    write_json(phase1_dir / "tensor_rows.json", tensor_rows)
    tensor_schema = {
        "axes": axis_catalogs,
        "value_fields": {
            "aligned_tensor": "phase0/aligned_tensor",
            "standardized_tensor": "phase1/standardized_tensor",
            "denominator_tensor": "phase1/denominator_tensor",
            "raw_numeric_value": "raw_numeric_value",
            "model_numeric_value": "model_numeric_value",
        },
        "default_value_field": "standardized_tensor",
    }
    write_json(phase1_dir / "tensor_schema.json", tensor_schema)
    write_json(phase1_dir / "normalization_report.json", normalization_report)

    manifest = Phase0ManifestArtifact(
        plugin_id=plugin_id,
        run_id=run_id,
        generated_at=utc_now_iso(),
        raw_dir=str(ctx.run_dir / "phase0" / "raw"),
        parsed_dir=str(ctx.run_dir / "phase0" / "parsed"),
        extracted_dir=str(phase1_dir),
        index_dir=str(ctx.run_dir / "phase0" / "index"),
        stage_status={"phase1": "completed"},
        artifact_paths={
            "standardized_tensor": standardized_artifact["value_path"],
            "denominator_tensor": denominator_artifact["value_path"],
            "normalized_subparameters": str(phase1_dir / "normalized_subparameters.json"),
            "parameter_catalog": str(phase1_dir / "parameter_catalog.json"),
            "axis_catalogs": str(phase1_dir / "axis_catalogs.json"),
            "tensor_rows": str(phase1_dir / "tensor_rows.json"),
            "tensor_schema": str(phase1_dir / "tensor_schema.json"),
            "normalization_report": str(phase1_dir / "normalization_report.json"),
        },
        backend_status={
            "torch": Phase0BackendStatus("torch", backend_map["torch"].available, backend_map["torch"].selected, notes=backend_map["torch"].device),
            "jax": Phase0BackendStatus("jax", backend_map["jax"].available, False, notes=backend_map["jax"].device),
        },
        source_count=registry.get("subparameter_count", 0),
        canonical_candidate_count=len(normalized_rows),
        numeric_observation_count=len(tensor_rows),
        notes=["phase1_preprocessing:robust_tensor_scaling"],
    ).to_dict()
    manifest["profile_id"] = profile
    truth_paths = write_ground_truth_package(
        phase_dir=phase1_dir,
        phase_name="phase1",
        profile_id=profile,
        checks=[
            {"name": "normalized_rows_present", "passed": bool(normalized_rows)},
            {"name": "standardized_tensor_finite", "passed": bool(np.isfinite(standardized_tensor).all())},
            {"name": "denominator_tensor_finite", "passed": bool(np.isfinite(denominator_tensor).all())},
            {
                "name": "no_duplicate_national_labels",
                "passed": not (("Philippines" in province_axis) and any(is_national_geo(name) for name in province_axis if name != "Philippines")),
            },
            {
                "name": "truth_fields_present",
                "passed": all("source_reliability_class" in row and "bias_penalty" in row for row in normalized_rows),
            },
        ],
        truth_sources=sorted(set(str(row.get("source_reliability_class") or "") for row in normalized_rows if row.get("source_reliability_class"))),
        stage_manifest_path=str(phase1_dir / "phase1_manifest.json"),
        summary={
            "normalized_row_count": len(normalized_rows),
            "tensor_row_count": len(tensor_rows),
            "province_count": len(province_axis),
            "month_count": len(month_axis),
            "canonical_name_count": len(canonical_axis),
            "anchor_eligible_count": int(sum(1 for row in normalized_rows if row.get("is_anchor_eligible"))),
        },
    )
    manifest["artifact_paths"].update(truth_paths)
    write_json(phase1_dir / "phase1_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase1_build",
        [
            phase1_dir / "standardized_tensor.npz",
            phase1_dir / "denominator_tensor.npz",
            phase1_dir / "normalized_subparameters.json",
            phase1_dir / "parameter_catalog.json",
            phase1_dir / "axis_catalogs.json",
            phase1_dir / "tensor_rows.json",
            phase1_dir / "tensor_schema.json",
            phase1_dir / "normalization_report.json",
            phase1_dir / "phase1_manifest.json",
        ],
    )
    return manifest
