from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from epigraph_ph.runtime import write_json

from epigraph_ph.phase3.incidence.artifacts import IncidenceResearchContext, write_experiment_artifacts
from epigraph_ph.phase3.incidence.audits import _base_numeric_policy, _write_heatmap_png
from epigraph_ph.phase3.incidence.sources import load_incidence_audit_inputs


SHOCK00A_EXPERIMENT_ID = "SHOCK-00A-covid-shock-subparameter-audit"

BACKLOG_RELEASE_TERMS: tuple[str, ...] = (
    "reopen",
    "re-opening",
    "resume",
    "resum",
    "restore",
    "catch-up",
    "catch up",
    "recovery",
    "restart",
    "backlog",
    "post-lockdown",
    "easing",
)

GENERAL_CONTEXT_CANONICALS: frozenset[str] = frozenset(
    {
        "policy_implementation_weakness",
        "unemployment_rate",
        "economic_access_constraint",
        "travel_time",
        "remoteness",
        "cash_instability",
    }
)
EXCLUDED_OBSERVATION_CANONICALS: frozenset[str] = frozenset({"population_count", "case_count"})
DIAGNOSIS_SERVICE_CANONICALS: frozenset[str] = frozenset(
    {
        "testing_rate",
        "testing_uptake",
        "health_system_reach",
        "clinics_per_capita",
        "travel_time",
        "economic_access_constraint",
        "poverty_rate",
        "policy_implementation_weakness",
        "stigma_barrier",
        "late_diagnosis_rate",
        "service_delivery_reach",
    }
)
INCIDENCE_SIDE_CANONICALS: frozenset[str] = frozenset(
    {
        "mobility_network_mixing",
        "sexual_risk",
        "idu_prevalence",
        "social_media_use",
        "remoteness",
        "prevention_access",
        "labor_migration",
        "prevention_coverage",
    }
)
DOWNSTREAM_CARE_CANONICALS: frozenset[str] = frozenset(
    {
        "art_uptake_rate",
        "suppression_outcomes",
        "viral_load",
        "retention_adherence",
        "viral_suppression_rate",
        "linkage_to_care",
    }
)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value in (None, ""):
        return []
    return [str(value)]


def _row_text_blob(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "canonical_name",
        "signal_family",
        "payload_family",
        "pathway_family",
        "candidate_text",
    ):
        parts.extend(_string_list(row.get(key)))
    for key in ("soft_ontology_tags", "soft_subparameter_hints", "linkage_targets"):
        parts.extend(_string_list(row.get(key)))
    return " ".join(parts).lower()


def _family_from_row(row: dict[str, Any]) -> str | None:
    canonical_name = str(row.get("canonical_name") or "")
    signal_family = str(row.get("signal_family") or "")
    pathway_family = str(row.get("pathway_family") or "")
    payload_family = str(row.get("payload_family") or "")
    text_blob = _row_text_blob(row)

    if canonical_name in EXCLUDED_OBSERVATION_CANONICALS:
        if any(term in text_blob for term in BACKLOG_RELEASE_TERMS) and pathway_family in {"testing_uptake", "linkage_to_care"}:
            return "backlog_release_shock"
        return None

    shock_text_present = any(term in text_blob for term in BACKLOG_RELEASE_TERMS) or any(
        term in text_blob
        for term in (
            "covid",
            "lockdown",
            "quarantine",
            "disruption",
            "reopen",
            "recovery",
            "transport",
            "outreach",
            "telemedicine",
            "mobility",
        )
    )

    diagnosis_relevant = (
        pathway_family in {"testing_uptake", "linkage_to_care"}
        or canonical_name in DIAGNOSIS_SERVICE_CANONICALS
        or signal_family in {"service_delivery", "behavior_stigma", "economics_access"}
    )
    incidence_relevant = (
        pathway_family in {"prevention_access", "mobility_network_mixing"}
        or canonical_name in INCIDENCE_SIDE_CANONICALS
        or signal_family == "mobility_logistics"
    )
    downstream_relevant = (
        pathway_family in {"suppression_outcomes", "retention_adherence"}
        or canonical_name in DOWNSTREAM_CARE_CANONICALS
    )
    context_relevant = canonical_name in GENERAL_CONTEXT_CANONICALS or (
        signal_family in {"general_context", "policy_environment"} and shock_text_present
    )

    if any(term in text_blob for term in BACKLOG_RELEASE_TERMS):
        if diagnosis_relevant:
            return "backlog_release_shock"

    if downstream_relevant:
        return "downstream_care_shock"

    if diagnosis_relevant:
        return "diagnosis_service_shock"

    if incidence_relevant:
        return "incidence_side_shock"

    if context_relevant:
        return "general_context_only"
    if payload_family in {"PolicyEnvironment", "EconomicConstraint"} and shock_text_present:
        return "general_context_only"

    return None


def _retained_row_payload(row: dict[str, Any], family: str) -> dict[str, Any]:
    return {
        "normalized_id": str(row.get("normalized_id") or ""),
        "family": family,
        "canonical_name": str(row.get("canonical_name") or ""),
        "signal_family": str(row.get("signal_family") or ""),
        "payload_family": str(row.get("payload_family") or ""),
        "pathway_family": str(row.get("pathway_family") or ""),
        "year": row.get("year"),
        "month": row.get("month"),
        "time": str(row.get("time") or ""),
        "geo_resolution": str(row.get("geo_resolution") or ""),
        "region": str(row.get("region") or ""),
        "kp_group": str(row.get("kp_group") or ""),
        "age_band": str(row.get("age_band") or ""),
        "source_reliability_class": str(row.get("source_reliability_class") or ""),
        "source_bank": str(row.get("source_bank") or ""),
        "soft_ontology_tags": _string_list(row.get("soft_ontology_tags")),
        "soft_subparameter_hints": _string_list(row.get("soft_subparameter_hints")),
        "linkage_targets": _string_list(row.get("linkage_targets")),
        "candidate_text": str(row.get("candidate_text") or ""),
    }


def run_shock_00a(ctx: IncidenceResearchContext) -> dict[str, Any]:
    inputs = load_incidence_audit_inputs(ctx)
    retained_rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    family_year_counts: dict[str, Counter[int]] = defaultdict(Counter)
    family_canonical_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_reliability_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in inputs.normalized_subparameter_rows:
        family = _family_from_row(row)
        if family is None:
            continue
        retained_payload = _retained_row_payload(row, family)
        retained_rows.append(retained_payload)
        family_counts[family] += 1
        canonical_name = str(retained_payload["canonical_name"])
        if canonical_name:
            family_canonical_counts[family][canonical_name] += 1
        reliability_class = str(retained_payload["source_reliability_class"])
        if reliability_class:
            family_reliability_counts[family][reliability_class] += 1
        year_value = retained_payload.get("year")
        if isinstance(year_value, int):
            family_year_counts[family][year_value] += 1

    families = [
        "diagnosis_service_shock",
        "backlog_release_shock",
        "incidence_side_shock",
        "downstream_care_shock",
        "general_context_only",
    ]
    years = sorted({year for counter in family_year_counts.values() for year in counter})
    heatmap_years = years or ["unavailable"]
    heatmap_values = np.asarray(
        [[float(family_year_counts[family].get(year, 0)) for year in years] for family in families]
        if years
        else [[0.0] for _ in families],
        dtype=np.float32,
    )
    _write_heatmap_png(
        ctx.experiment_dir / "shock_factor_family_heatmap.png",
        title="Shock Subparameter Coverage By Year",
        row_labels=families,
        column_labels=[str(year) for year in heatmap_years],
        values=heatmap_values,
    )

    coverage_summary = {
        "normalized_subparameter_path": str(inputs.normalized_subparameters_path) if inputs.normalized_subparameters_path is not None else None,
        "normalized_subparameter_row_count": int(len(inputs.normalized_subparameter_rows)),
        "retained_shock_row_count": int(len(retained_rows)),
        "family_row_counts": {family: int(family_counts.get(family, 0)) for family in families},
        "family_year_counts": {
            family: {str(year): int(count) for year, count in sorted(family_year_counts.get(family, Counter()).items())}
            for family in families
        },
        "family_reliability_counts": {
            family: {name: int(count) for name, count in sorted(family_reliability_counts.get(family, Counter()).items())}
            for family in families
        },
    }
    write_json(ctx.experiment_dir / "shock_signal_coverage_summary.json", coverage_summary)

    diagnosis_supported = family_counts.get("diagnosis_service_shock", 0) > 0
    backlog_supported = family_counts.get("backlog_release_shock", 0) > 0
    incidence_supported = family_counts.get("incidence_side_shock", 0) > 0
    downstream_supported = family_counts.get("downstream_care_shock", 0) > 0
    shock_split_justified = diagnosis_supported and backlog_supported and incidence_supported
    recommended_next_branch = (
        "INC-01E-covid-shock-diagnosis-release-split"
        if shock_split_justified
        else "SHOCK-00B-quarterly-shock-factor-surfaces"
        if diagnosis_supported and (backlog_supported or incidence_supported or downstream_supported)
        else "do_not_add_shock_split_yet"
    )

    audit_payload = {
        "source_run_id": ctx.source_run_id,
        "normalized_subparameter_path": str(inputs.normalized_subparameters_path) if inputs.normalized_subparameters_path is not None else None,
        "retained_shock_row_count": int(len(retained_rows)),
        "family_row_counts": {family: int(family_counts.get(family, 0)) for family in families},
        "family_year_coverage": {
            family: sorted(int(year) for year in family_year_counts.get(family, Counter()).keys())
            for family in families
        },
        "family_canonical_counts": {
            family: [
                {"canonical_name": canonical_name, "row_count": int(count)}
                for canonical_name, count in sorted(
                    family_canonical_counts.get(family, Counter()).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            for family in families
        },
        "shock_diagnosis_release_split_justified": shock_split_justified,
        "recommended_next_branch": recommended_next_branch,
        "retained_rows": retained_rows,
        "notes": [
            "Rows are typed into shock families using normalized subparameter metadata and recovery-language support from source text.",
            "General context rows are retained separately so later experiments can decide whether they should remain validation-only context.",
        ],
    }
    write_json(ctx.experiment_dir / "shock_subparameter_audit.json", audit_payload)

    experiment_spec = {
        "experiment_id": SHOCK00A_EXPERIMENT_ID,
        "description": ctx.experiment.description,
        "source_run_id": ctx.source_run_id,
        "expected_outputs": list(ctx.experiment.expected_outputs),
    }
    decision = {
        "passed": True,
        "shock_split_justified": shock_split_justified,
        "recommended_next_branch": recommended_next_branch,
    }
    write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=_base_numeric_policy(),
    )
    return {
        "decision": decision,
        "coverage_summary": coverage_summary,
        "audit": audit_payload,
    }
