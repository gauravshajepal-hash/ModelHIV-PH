from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .r64_leakage_support_gap_prioritizer import R64_RUN_ID
from .r65_transmission_model_readiness_gate import R65_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R66_SCHEMA_VERSION = "phase3_dynamic.r66_scientific_source_base.v1"
R66_RUN_ID = "p3d-r66-scientific-source-base-20260506-s00"
R64_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R64_RUN_ID
    / "analysis"
    / "r64_leakage_support_gap_prioritizer_report.json"
)
R65_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R65_RUN_ID
    / "analysis"
    / "r65_transmission_model_readiness_gate_report.json"
)
HIV_DATA_DIR = sandbox_repo_root().parent / "harp_archive" / "HIV_Data"
DOWNLOADED_HASP_DIR = HIV_DATA_DIR / "downloaded_hasp"


def _row_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _source_id(source_family: str, path_or_url: str) -> str:
    digest = hashlib.sha256(f"{source_family}|{path_or_url}".encode("utf-8")).hexdigest()[:16]
    safe_family = source_family.replace("/", "_").replace(" ", "_")
    return f"{safe_family}:{digest}"


def _classify_local_source(path: Path) -> dict[str, Any]:
    name = path.name
    lower = name.lower()
    suffix = path.suffix.lower()
    module_targets: list[str] = []
    measurement_semantics: list[str] = []
    source_family = "unclassified_local_source"
    observation_role = "quarantined"
    allowed_use = "quarantined_until_lineage_review"
    evidence_note = "Local file not recognized by the R66 scientific source classifier."

    if suffix == ".pdf" and ("hasp" in lower or "hiv-aids-surveillance" in lower):
        source_family = "doh_hasp_report_pdf"
        observation_role = "direct_target"
        allowed_use = "direct_program_observation_after_row_extraction"
        module_targets = ["diagnosis_reporting", "art_retention", "vl_suppression_service", "mortality_reporting", "prep_persistence"]
        measurement_semantics = ["flow_count", "stock_anchor", "reporting_process_covariate"]
        evidence_note = "HASP PDFs define diagnosis, ART outcome, VL/suppression, PrEP, AHD, and reported death program observations."
    elif suffix == ".pdf" and "ihbss" in lower:
        source_family = "philippines_ihbss_survey_pdf"
        observation_role = "prior_context"
        allowed_use = "key_population_context_and_determinant_covariate"
        module_targets = ["incidence_pressure", "kp_overlay", "behavioral_risk"]
        measurement_semantics = ["determinant_covariate", "denominator", "proportion"]
        evidence_note = "IHBSS is survey evidence for key-population denominators, prevalence, testing, behavior, and prevention context."
    elif suffix == ".csv" and any(token in lower for token in ("new hiv infections", "hiv incidence")):
        source_family = "unaids_aidsinfo_incidence_csv"
        observation_role = "validation_only"
        allowed_use = "annual_incidence_weak_measurement_or_external_challenge"
        module_targets = ["incidence_validation", "annual_challenge"]
        measurement_semantics = ["modeled_estimate", "flow_count"]
        evidence_note = "Annual modeled incidence must not be converted into quarterly training truth."
    elif suffix == ".csv" and "aids-related deaths" in lower:
        source_family = "unaids_aidsinfo_deaths_csv"
        observation_role = "validation_only"
        allowed_use = "annual_aids_death_weak_measurement_or_external_challenge"
        module_targets = ["mortality_reporting", "annual_challenge"]
        measurement_semantics = ["modeled_estimate", "flow_count"]
        evidence_note = "Annual modeled AIDS-death estimates are challenge/weak-measurement evidence, not direct quarterly death truth."
    elif suffix == ".csv" and ("people living with hiv" in lower or ("hiv prevalence" in lower and not _is_key_population_file(lower))):
        source_family = "unaids_aidsinfo_plhiv_csv"
        observation_role = "validation_only"
        allowed_use = "annual_plhiv_weak_measurement_or_external_challenge"
        module_targets = ["plhiv_stock_validation", "annual_challenge"]
        measurement_semantics = ["modeled_estimate", "stock_anchor", "proportion"]
        evidence_note = "Annual modeled PLHIV/prevalence estimates are external challenge or weak measurement evidence."
    elif suffix == ".csv" and "treatment cascade" in lower:
        source_family = "unaids_aidsinfo_cascade_csv"
        observation_role = "auxiliary_likelihood"
        allowed_use = "annual_cascade_auxiliary_or_external_challenge"
        module_targets = ["diagnosis_reporting", "art_retention", "vl_suppression_service", "annual_challenge"]
        measurement_semantics = ["modeled_estimate", "stock_anchor", "proportion"]
        evidence_note = "Annual cascade estimates can constrain broad scale but should not override direct HASP program rows."
    elif suffix == ".csv" and _is_key_population_file(lower):
        source_family = "unaids_key_population_csv"
        observation_role = "prior_context"
        allowed_use = "determinant_sensitivity_until_source_stable"
        module_targets = ["incidence_pressure", "kp_overlay", "regional_shrinkage"]
        measurement_semantics = ["determinant_covariate", "denominator", "proportion"]
        evidence_note = "KP rows are high-value determinant/context evidence but need R46-style source-family falsification before strict priors."
    elif suffix == ".csv" and any(token in lower for token in ("stigma", "young people", "combination prevention", "hiv expenditure")):
        source_family = "unaids_determinant_context_csv"
        observation_role = "prior_context"
        allowed_use = "determinant_sensitivity_until_source_stable"
        module_targets = ["incidence_pressure", "diagnosis_reporting", "art_retention", "prep_persistence"]
        measurement_semantics = ["determinant_covariate", "proportion", "flow_count"]
        evidence_note = "Prevention, stigma, youth knowledge, testing, PrEP, and expenditure rows are contextual determinant evidence."
    elif suffix == ".csv" and "epidemic transition metrics" in lower:
        source_family = "unaids_epidemic_transition_csv"
        observation_role = "validation_only"
        allowed_use = "annual_epidemic_transition_external_challenge"
        module_targets = ["annual_challenge", "incidence_validation", "mortality_reporting"]
        measurement_semantics = ["modeled_estimate", "proportion"]
        evidence_note = "Incidence/mortality and incidence/prevalence ratios are external annual challenge metrics."
    elif suffix in {".xlsx", ".xls"}:
        source_family = "local_spreadsheet_needs_lineage_review"
        observation_role = "quarantined"
        allowed_use = "quarantined_until_source_owner_and_semantics_are_documented"
        module_targets = ["unknown"]
        measurement_semantics = ["unknown"]
        evidence_note = "Spreadsheet lineage is ambiguous; R66 blocks it from scoring until reviewed."

    payload = {
        "source_id": _source_id(source_family, path.as_posix()),
        "source_family": source_family,
        "source_path": path.as_posix(),
        "source_url": "",
        "filename": name,
        "support_status": "local_present",
        "source_tier": "local_archive",
        "observation_role": observation_role,
        "allowed_use": allowed_use,
        "module_targets": "|".join(module_targets),
        "measurement_semantics": "|".join(measurement_semantics),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if path.exists() and path.is_file() else None,
        "evidence_note": evidence_note,
    }
    payload["row_hash"] = _row_hash(payload)
    return payload


def _is_key_population_file(lower_name: str) -> bool:
    return any(
        token in lower_name
        for token in (
            "men who have sex with men",
            "transgender people",
            "sex workers",
            "people who inject drugs",
            "prisoners",
        )
    )


def _local_source_rows(hiv_data_dir: Path = HIV_DATA_DIR) -> list[dict[str, Any]]:
    if not hiv_data_dir.exists():
        return []
    rows = [_classify_local_source(path) for path in sorted(hiv_data_dir.glob("*")) if path.is_file()]
    downloaded = hiv_data_dir / "downloaded_hasp"
    if downloaded.exists():
        rows.extend(_classify_local_source(path) for path in sorted(downloaded.glob("*")) if path.is_file() and path.suffix.lower() == ".pdf")
    rows.sort(key=lambda row: (str(row.get("source_family") or ""), str(row.get("filename") or "")))
    return rows


def _wdi_specs() -> list[dict[str, str]]:
    return [
        {
            "indicator": "SP.POP.TOTL",
            "name": "population_total",
            "module_targets": "population_denominator|regional_shrinkage",
            "allowed_use": "determinant_covariate_denominator_context",
        },
        {
            "indicator": "SP.URB.TOTL.IN.ZS",
            "name": "urban_population_percent",
            "module_targets": "incidence_pressure|mobility_proxy|regional_shrinkage",
            "allowed_use": "determinant_covariate_prior_context",
        },
        {
            "indicator": "NY.GDP.PCAP.KD",
            "name": "gdp_per_capita_constant_2015_usd",
            "module_targets": "structural_access|regional_shrinkage",
            "allowed_use": "determinant_covariate_prior_context",
        },
        {
            "indicator": "SH.XPD.CHEX.GD.ZS",
            "name": "current_health_expenditure_percent_gdp",
            "module_targets": "service_capacity|art_retention|vl_suppression_service",
            "allowed_use": "determinant_covariate_prior_context",
        },
        {
            "indicator": "IT.NET.USER.ZS",
            "name": "internet_users_percent",
            "module_targets": "app_network_proxy|diagnosis_reporting|incidence_pressure",
            "allowed_use": "determinant_covariate_prior_context",
        },
        {
            "indicator": "SE.SEC.ENRR",
            "name": "secondary_school_enrollment_gross_percent",
            "module_targets": "sex_education_proxy|diagnosis_reporting|incidence_pressure",
            "allowed_use": "weak_determinant_proxy_prior_context",
        },
        {
            "indicator": "SH.MED.PHYS.ZS",
            "name": "physicians_per_1000_people",
            "module_targets": "health_access|diagnosis_reporting|art_retention",
            "allowed_use": "determinant_covariate_prior_context",
        },
    ]


def _download_url(url: str, path: Path, *, timeout_seconds: int = 45) -> tuple[str, str]:
    ensure_dir(path.parent)
    request = urllib.request.Request(url, headers={"User-Agent": "ModelHIV-PH-R66/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
        path.write_bytes(body)
        return "downloaded", ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "download_failed", str(exc)


def _wdi_download_rows(source_dir: Path, *, allow_network: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _wdi_specs():
        indicator = spec["indicator"]
        url = f"https://api.worldbank.org/v2/country/PHL/indicator/{indicator}?format=json&per_page=20000"
        filename = f"world_bank_wdi_PHL_{indicator}.json"
        path = source_dir / "world_bank_wdi" / filename
        if allow_network:
            status, error = _download_url(url, path)
        else:
            status = "network_skipped"
            error = "allow_network=false"
        payload = {
            "source_id": _source_id("world_bank_wdi_api", url),
            "source_family": "world_bank_wdi_api",
            "source_path": path.as_posix() if path.exists() else "",
            "source_url": url,
            "filename": filename,
            "support_status": status,
            "source_tier": "official_api",
            "observation_role": "prior_context",
            "allowed_use": spec["allowed_use"],
            "module_targets": spec["module_targets"],
            "measurement_semantics": "determinant_covariate|denominator",
            "file_size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256(path) if path.exists() else None,
            "evidence_note": f"Compact WDI API pull for Philippines indicator {indicator}: {spec['name']}.",
            "download_error": error,
        }
        payload["row_hash"] = _row_hash(payload)
        rows.append(payload)
    return rows


def _external_reference_rows() -> list[dict[str, Any]]:
    references = [
        {
            "source_family": "unaids_aidsinfo_dataset",
            "source_url": "https://aidsinfo.unaids.org/dataset",
            "filename": "UNAIDS AIDSinfo dataset landing page",
            "observation_role": "validation_only",
            "allowed_use": "annual_official_hiv_estimates_external_challenge",
            "module_targets": "incidence_validation|mortality_reporting|plhiv_stock_validation|annual_challenge",
            "measurement_semantics": "modeled_estimate|stock_anchor|flow_count",
            "evidence_note": "Use AIDSinfo annual estimates as external challenge/weak-measurement evidence; do not downcast annual estimates into quarterly truth.",
        },
        {
            "source_family": "doh_hasp_primary_lineage_reference",
            "source_url": "https://doh.gov.ph",
            "filename": "DOH HASP primary lineage reference",
            "observation_role": "direct_target",
            "allowed_use": "preferred_primary_source_for_hasp_rows_when_available",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service|mortality_reporting",
            "measurement_semantics": "flow_count|stock_anchor|reporting_process_covariate",
            "evidence_note": "R66 keeps mirrored HASP PDFs but prefers primary DOH/EB lineage when available.",
        },
        {
            "source_family": "google_mobility_or_trends_unscoped",
            "source_url": "https://www.google.com/covid19/mobility/",
            "filename": "Google mobility/trends candidate source",
            "observation_role": "prior_context",
            "allowed_use": "not_downloaded_until_scoped_region_time_extraction_exists",
            "module_targets": "mobility_proxy|app_network_proxy|reporting_disruption",
            "measurement_semantics": "determinant_covariate|reporting_process_covariate",
            "evidence_note": "Large or methodologically shifting Google data should be scoped before download to avoid noisy overfit pressure.",
        },
        {
            "source_family": "philippine_statistics_authority_unscoped",
            "source_url": "https://psa.gov.ph",
            "filename": "PSA population/demography candidate source",
            "observation_role": "prior_context",
            "allowed_use": "not_downloaded_until_indicator_and_geography_scope_are_locked",
            "module_targets": "population_denominator|regional_shrinkage|migration_proxy",
            "measurement_semantics": "denominator|determinant_covariate",
            "evidence_note": "PSA is a high-priority denominator source, but R66 does not bulk-download unscoped tables.",
        },
        {
            "source_family": "hiv_care_continuum_method_reference",
            "source_url": "https://clinicalinfo.hiv.gov/en/glossary/hiv-treatment-cascade",
            "filename": "NIH HIV continuum of care glossary",
            "observation_role": "prior_context",
            "allowed_use": "model_family_design_only",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service",
            "measurement_semantics": "conceptual_model_reference",
            "evidence_note": "Supports treating diagnosis, linkage, retention, and viral suppression as distinct care-continuum stages.",
        },
        {
            "source_family": "hiv_states_transitions_method_reference",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4506700/",
            "filename": "HIV States and Transitions framework",
            "observation_role": "prior_context",
            "allowed_use": "model_family_design_only",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service",
            "measurement_semantics": "conceptual_model_reference",
            "evidence_note": "Supports splitting cascade endpoints into dynamic states and transitions instead of a one-way cascade.",
        },
        {
            "source_family": "hiv_multistate_cascade_method_reference",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11986931/",
            "filename": "Modeling HIV cascade of care with multistate models",
            "observation_role": "prior_context",
            "allowed_use": "model_family_design_only",
            "module_targets": "art_retention|vl_suppression_service",
            "measurement_semantics": "statistical_method_reference",
            "evidence_note": "Supports multistate/cyclical care models with engagement and re-engagement rather than linear endpoint tuning.",
        },
        {
            "source_family": "hiv_incidence_backcalculation_method_reference",
            "source_url": "https://tbiomed.biomedcentral.com/articles/10.1186/s12976-019-0118-0",
            "filename": "Mathematical review of HIV incidence estimation",
            "observation_role": "prior_context",
            "allowed_use": "model_family_design_only",
            "module_targets": "incidence_validation|diagnosis_reporting",
            "measurement_semantics": "statistical_method_reference",
            "evidence_note": "Supports diagnosis-delay/back-calculation and joint use of diagnosis and CD4/late-diagnosis data.",
        },
        {
            "source_family": "hiv_flow_queue_method_reference",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/28471841/",
            "filename": "Flow-based model of the HIV care continuum",
            "observation_role": "prior_context",
            "allowed_use": "model_family_design_only",
            "module_targets": "diagnosis_reporting|art_retention|vl_suppression_service",
            "measurement_semantics": "queueing_method_reference",
            "evidence_note": "Supports service-capacity and queueing interpretations of care-continuum flows.",
        },
        {
            "source_family": "hiv_hierarchical_subnational_method_reference",
            "source_url": "https://arxiv.org/abs/1411.4219",
            "filename": "Hierarchical model for estimating HIV epidemics",
            "observation_role": "prior_context",
            "allowed_use": "model_family_design_only",
            "module_targets": "regional_shrinkage|incidence_pressure",
            "measurement_semantics": "hierarchical_method_reference",
            "evidence_note": "Supports borrowing strength across sparse subnational epidemics instead of independent region fits.",
        },
        {
            "source_family": "hiv_dynamic_hierarchical_subnational_method_reference",
            "source_url": "https://arxiv.org/abs/2401.04753",
            "filename": "Dynamic models augmented by hierarchical data for subnational HIV",
            "observation_role": "prior_context",
            "allowed_use": "model_family_design_only",
            "module_targets": "regional_shrinkage|annual_challenge",
            "measurement_semantics": "hierarchical_method_reference",
            "evidence_note": "Supports dynamic hierarchical estimation for subnational/subpopulation HIV with uncertainty-aware prediction.",
        },
        {
            "source_family": "hiv_prep_model_method_reference",
            "source_url": "https://arxiv.org/abs/1703.06446",
            "filename": "Modeling and optimal control of HIV prevention through PrEP",
            "observation_role": "prior_context",
            "allowed_use": "model_family_design_only",
            "module_targets": "prep_persistence|incidence_pressure",
            "measurement_semantics": "mechanistic_method_reference",
            "evidence_note": "Supports modeling PrEP as an incidence-pressure modifier, with persistence and active protection kept separate from enrollment counts.",
        },
    ]
    rows: list[dict[str, Any]] = []
    for ref in references:
        payload = {
            "source_id": _source_id(ref["source_family"], ref["source_url"]),
            "source_path": "",
            "support_status": "reference_only_not_downloaded",
            "source_tier": "official_or_candidate_reference",
            "file_size_bytes": None,
            "sha256": None,
            "download_error": "",
            **ref,
        }
        payload["row_hash"] = _row_hash(payload)
        rows.append(payload)
    return rows


def _module_coverage_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    modules = [
        "incidence_pressure",
        "incidence_validation",
        "diagnosis_reporting",
        "art_retention",
        "vl_suppression_service",
        "mortality_reporting",
        "prep_persistence",
        "kp_overlay",
        "regional_shrinkage",
        "annual_challenge",
    ]
    rows: list[dict[str, Any]] = []
    for module in modules:
        matched = [row for row in source_rows if module in str(row.get("module_targets") or "").split("|")]
        strict_or_direct = [
            row
            for row in matched
            if str(row.get("observation_role") or "") in {"direct_target", "auxiliary_likelihood", "validation_only"}
            and str(row.get("support_status") or "") in {"local_present", "downloaded"}
        ]
        determinant_only = [row for row in matched if "determinant" in str(row.get("allowed_use") or "")]
        rows.append(
            {
                "module_target": module,
                "source_count": len(matched),
                "usable_supported_count": len(strict_or_direct),
                "determinant_context_count": len(determinant_only),
                "source_families": "|".join(sorted({str(row.get("source_family") or "") for row in matched})),
                "readiness_hint": _readiness_hint(module, len(strict_or_direct), len(determinant_only)),
            }
        )
    rows.sort(key=lambda row: (str(row.get("readiness_hint")), str(row.get("module_target"))))
    return rows


def _readiness_hint(module: str, usable_count: int, determinant_count: int) -> str:
    if module in {"vl_suppression_service", "art_retention", "diagnosis_reporting"} and usable_count > 0:
        return "source_supported_needs_row_extraction_or_split_test"
    if module == "prep_persistence" and usable_count > 0:
        return "source_supported_needs_persistence_split"
    if module in {"incidence_pressure", "regional_shrinkage", "kp_overlay"} and determinant_count > 0:
        return "determinant_supported_but_strict_prior_locked"
    if module in {"incidence_validation", "mortality_reporting", "annual_challenge"} and usable_count > 0:
        return "external_challenge_supported"
    if usable_count == 0 and determinant_count == 0:
        return "source_gap"
    return "context_only"


def _support_gap_rows(r64_report: dict[str, Any], module_coverage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage_by_module = {str(row.get("module_target") or ""): row for row in module_coverage_rows}
    stream_to_module = {
        "vl_suppression_service": "vl_suppression_service",
        "art_stock_retention": "art_retention",
        "diagnosis_stock": "diagnosis_reporting",
        "population_burden_anchor": "annual_challenge",
    }
    rows: list[dict[str, Any]] = []
    for row in list(r64_report.get("metric_stream_priority_rows") or []):
        stream = str(row.get("metric_stream") or "")
        module = stream_to_module.get(stream, stream)
        coverage = coverage_by_module.get(module) or {}
        rows.append(
            {
                "r64_metric_stream": stream,
                "mapped_module_target": module,
                "r64_entry_count": row.get("entry_count"),
                "r64_student_gap_sum": row.get("sum_positive_student_gap_vs_oracle"),
                "supported_source_count": coverage.get("usable_supported_count", 0),
                "determinant_context_count": coverage.get("determinant_context_count", 0),
                "source_families": coverage.get("source_families", ""),
                "evidence_action": _gap_action(module, int(coverage.get("usable_supported_count") or 0)),
            }
        )
    return rows


def _gap_action(module: str, supported_count: int) -> str:
    if module == "vl_suppression_service":
        return "extract regional VL-tested and suppression table history from HASP PDFs; do not fit new regional hazards first"
    if module == "art_retention":
        return "extract active ART, LTFU definition, refill/return evidence, and deaths/transfer-out where available"
    if module == "diagnosis_reporting":
        return "separate diagnosis stock support from reporting-intensity and facility-count support"
    if supported_count == 0:
        return "acquire scoped evidence before model expansion"
    return "rerun blocked-time gate after row extraction"


def _gate(source_rows: list[dict[str, Any]], module_rows: list[dict[str, Any]]) -> dict[str, Any]:
    downloaded = sum(1 for row in source_rows if row.get("support_status") == "downloaded")
    local_present = sum(1 for row in source_rows if row.get("support_status") == "local_present")
    quarantined = sum(1 for row in source_rows if row.get("observation_role") == "quarantined")
    determinant_modules = {
        str(row.get("module_target") or "")
        for row in module_rows
        if int(row.get("determinant_context_count") or 0) > 0
    }
    return {
        "status": "source_base_ready_for_model_family_queue",
        "downloaded_source_count": downloaded,
        "local_source_count": local_present,
        "quarantined_source_count": quarantined,
        "determinant_module_count": len(determinant_modules),
        "contract": (
            "R66 is an evidence/source-base gate, not a model result. It permits creative model families only when each "
            "family declares source modules, allowed-use roles, leakage guards, and a blocked-time promotion test."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("source_base_gate") or {})
    lines = [
        "# Phase 3 R66 Scientific Source Base",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Local sources: `{gate.get('local_source_count')}`",
        f"- Downloaded compact sources: `{gate.get('downloaded_source_count')}`",
        f"- Quarantined sources: `{gate.get('quarantined_source_count')}`",
        "",
        "## Module Coverage",
        "",
        "| Module | Usable supported | Determinant context | Readiness |",
        "|---|---:|---:|---|",
    ]
    for row in report.get("module_coverage_rows") or []:
        lines.append(
            f"| `{row.get('module_target')}` | {int(row.get('usable_supported_count') or 0)} | "
            f"{int(row.get('determinant_context_count') or 0)} | `{row.get('readiness_hint')}` |"
        )
    lines.extend(["", "## R64-Guided Gaps", "", "| Stream | Module | Action |", "|---|---|---|"])
    for row in report.get("r64_support_gap_rows") or []:
        lines.append(f"| `{row.get('r64_metric_stream')}` | `{row.get('mapped_module_target')}` | {row.get('evidence_action')} |")
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r66_scientific_source_base(
    *,
    run_id: str = R66_RUN_ID,
    allow_network: bool = True,
    hiv_data_dir: Path | None = None,
    r64_report_path: Path | None = None,
    r65_report_path: Path | None = None,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    source_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "external_sources")
    local_rows = _local_source_rows(HIV_DATA_DIR if hiv_data_dir is None else Path(hiv_data_dir))
    wdi_rows = _wdi_download_rows(source_dir, allow_network=allow_network)
    reference_rows = _external_reference_rows()
    source_rows = local_rows + wdi_rows + reference_rows
    module_rows = _module_coverage_rows(source_rows)
    r64_path = R64_DEFAULT_REPORT if r64_report_path is None else Path(r64_report_path)
    r65_path = R65_DEFAULT_REPORT if r65_report_path is None else Path(r65_report_path)
    r64 = dict(read_json(r64_path, default={}) or {}) if r64_path.exists() else {}
    support_gap_rows = _support_gap_rows(r64, module_rows)
    gate = _gate(source_rows, module_rows)
    report_path = analysis_dir / "r66_scientific_source_base_report.json"
    markdown_path = analysis_dir / "r66_scientific_source_base_report.md"
    source_csv = analysis_dir / "r66_source_manifest_rows.csv"
    module_csv = analysis_dir / "r66_module_coverage_rows.csv"
    gap_csv = analysis_dir / "r66_r64_support_gap_rows.csv"
    report = {
        "schema_version": R66_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_base_gate": gate,
        "source_manifest_rows": source_rows,
        "module_coverage_rows": module_rows,
        "r64_support_gap_rows": support_gap_rows,
        "source_artifacts": {
            "r64": {"path": r64_path.as_posix(), "sha256": _sha256(r64_path) if r64_path.exists() else None},
            "r65": {"path": r65_path.as_posix(), "sha256": _sha256(r65_path) if r65_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "source_manifest_csv": source_csv.as_posix(),
            "module_coverage_csv": module_csv.as_posix(),
            "r64_support_gap_csv": gap_csv.as_posix(),
            "external_source_dir": source_dir.as_posix(),
        },
    }
    _write_csv(source_csv, source_rows)
    _write_csv(module_csv, module_rows)
    _write_csv(gap_csv, support_gap_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build R66 scientific source base.")
    parser.add_argument("--run-id", default=R66_RUN_ID)
    parser.add_argument("--no-network", action="store_true", help="Skip compact external downloads.")
    parser.add_argument("--hiv-data-dir", default=None)
    parser.add_argument("--r64-report-path", default=None)
    parser.add_argument("--r65-report-path", default=None)
    args = parser.parse_args()
    run_r66_scientific_source_base(
        run_id=str(args.run_id),
        allow_network=not bool(args.no_network),
        hiv_data_dir=None if args.hiv_data_dir is None else Path(args.hiv_data_dir),
        r64_report_path=None if args.r64_report_path is None else Path(args.r64_report_path),
        r65_report_path=None if args.r65_report_path is None else Path(args.r65_report_path),
    )


if __name__ == "__main__":
    _main()
