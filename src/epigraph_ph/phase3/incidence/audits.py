from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.runtime import write_json

from .artifacts import IncidenceResearchContext, write_experiment_artifacts
from epigraph_ph.phase3.frontier.numeric_policy import numerical_guard_entry
from .sources import IncidenceAuditInputs, load_incidence_audit_inputs


INC00A_EXPERIMENT_ID = "INC-00A-incidence-evidence-audit"
INC00B_EXPERIMENT_ID = "INC-00B-population-denominator-audit"
INC00C_EXPERIMENT_ID = "INC-00C-incidence-identifiability-audit"

DIRECT_INFLOW_METRICS: tuple[str, ...] = ("annual_new_infections",)
DIAGNOSIS_FLOW_METRICS: tuple[str, ...] = ("new_diagnosed_cases_monthly", "new_diagnosed_cases_period")
STOCK_ANCHOR_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
)
VALIDATION_ONLY_METRICS: tuple[str, ...] = ("annual_aids_deaths", "estimated_plhiv")
INDIRECT_PROXY_METRICS: tuple[str, ...] = (
    "advanced_hiv_cases_period",
    "median_cd4_at_enrollment",
    "art_median_age",
    "youth_cases_15_24_period",
    "average_cases_per_day",
    "prep_newly_enrolled_period",
    "new_hiv_positive_pregnant_women",
    "transactional_sex_cases_period",
)


def _year_from_row(row: dict[str, Any]) -> int | None:
    for key in ("time", "period_end", "period_start"):
        raw_value = str(row.get(key) or "").strip()
        if len(raw_value) >= 4 and raw_value[:4].isdigit():
            return int(raw_value[:4])
    raw_year = row.get("year")
    if raw_year not in (None, ""):
        try:
            return int(raw_year)
        except (TypeError, ValueError):
            return None
    return None


def _metric_rows(rows: list[dict[str, Any]], metric_names: tuple[str, ...]) -> list[dict[str, Any]]:
    target_names = set(metric_names)
    return [row for row in rows if str(row.get("metric_name") or "") in target_names]


def _is_program_observed(row: dict[str, Any]) -> bool:
    return str(row.get("measurement_class") or "") == "program_observed_harp"


def _is_model_estimate(row: dict[str, Any]) -> bool:
    return str(row.get("measurement_class") or "") == "model_estimate"


def _year_counter(rows: list[dict[str, Any]]) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for row in rows:
        year_value = _year_from_row(row)
        if year_value is not None:
            counter[int(year_value)] += 1
    return {int(year): int(count) for year, count in sorted(counter.items())}


def _annualize_diagnosis_flow(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_metric_year: dict[str, dict[int, float]] = {"new_diagnosed_cases_monthly": defaultdict(float), "new_diagnosed_cases_period": defaultdict(float)}
    for row in rows:
        metric_name = str(row.get("metric_name") or "")
        if metric_name not in by_metric_year:
            continue
        year_value = _year_from_row(row)
        value = row.get("value")
        if year_value is None or value in (None, ""):
            continue
        try:
            by_metric_year[metric_name][int(year_value)] += float(value)
        except (TypeError, ValueError):
            continue
    years = sorted(set(by_metric_year["new_diagnosed_cases_monthly"]) | set(by_metric_year["new_diagnosed_cases_period"]))
    annualized: list[dict[str, Any]] = []
    for year in years:
        if year in by_metric_year["new_diagnosed_cases_monthly"]:
            annualized.append(
                {
                    "year": int(year),
                    "value": float(by_metric_year["new_diagnosed_cases_monthly"][year]),
                    "source_metric": "new_diagnosed_cases_monthly",
                }
            )
        elif year in by_metric_year["new_diagnosed_cases_period"]:
            annualized.append(
                {
                    "year": int(year),
                    "value": float(by_metric_year["new_diagnosed_cases_period"][year]),
                    "source_metric": "new_diagnosed_cases_period",
                }
            )
    return annualized


def _write_heatmap_png(path: Path, title: str, row_labels: list[str], column_labels: list[str], values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * max(1, len(column_labels))), max(3.0, 0.7 * max(1, len(row_labels)))))
    image = ax.imshow(values, aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            ax.text(col_index, row_index, f"{values[row_index, col_index]:.0f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _base_numeric_policy() -> list[dict[str, Any]]:
    return [
        {
            "name": "support_presence_boundary",
            "value": 0,
            "role": "Support presence and absence boundary in audits.",
            "source_type": "physical_constraint",
            "estimation_data": "count support is absent below zero and present above zero",
            "estimation_method": "non-negative count support boundary",
            "uncertainty": "none",
            "why_needed": "Audit decisions need an explicit zero-support boundary without using undocumented literals.",
        },
        numerical_guard_entry(
            name="float32_machine_epsilon",
            role="Numerical stability guard for finite division and correlation calculations.",
            why_needed="Audit summaries use guarded finite arithmetic when overlap is sparse.",
        ),
    ]


def run_inc_00a(ctx: IncidenceResearchContext) -> dict[str, Any]:
    inputs = load_incidence_audit_inputs(ctx)
    direct_inflow_rows = _metric_rows(inputs.historical_metric_rows, DIRECT_INFLOW_METRICS)
    diagnosis_flow_rows = _metric_rows(inputs.historical_metric_rows, DIAGNOSIS_FLOW_METRICS)
    stock_anchor_rows = _metric_rows(inputs.historical_metric_rows, STOCK_ANCHOR_METRICS)
    validation_only_rows = _metric_rows(inputs.historical_metric_rows, VALIDATION_ONLY_METRICS)
    indirect_proxy_rows = _metric_rows(inputs.historical_metric_rows, INDIRECT_PROXY_METRICS)

    direct_inflow_training_rows = [row for row in direct_inflow_rows if _is_program_observed(row)]
    direct_inflow_validation_rows = [row for row in direct_inflow_rows if _is_model_estimate(row)]
    target_leaky_rows = [
        row
        for row in validation_only_rows + direct_inflow_validation_rows
        if _is_model_estimate(row)
    ]

    coverage_rows = [
        {
            "support_family": "direct_inflow_training",
            "metric_names": list(DIRECT_INFLOW_METRICS),
            "row_count": int(len(direct_inflow_training_rows)),
            "year_counts": _year_counter(direct_inflow_training_rows),
        },
        {
            "support_family": "direct_inflow_validation_only",
            "metric_names": list(DIRECT_INFLOW_METRICS),
            "row_count": int(len(direct_inflow_validation_rows)),
            "year_counts": _year_counter(direct_inflow_validation_rows),
        },
        {
            "support_family": "diagnosis_flow",
            "metric_names": list(DIAGNOSIS_FLOW_METRICS),
            "row_count": int(len(diagnosis_flow_rows)),
            "year_counts": _year_counter(diagnosis_flow_rows),
        },
        {
            "support_family": "stock_anchor",
            "metric_names": list(STOCK_ANCHOR_METRICS),
            "row_count": int(len(stock_anchor_rows)),
            "year_counts": _year_counter(stock_anchor_rows),
        },
        {
            "support_family": "indirect_proxy",
            "metric_names": list(INDIRECT_PROXY_METRICS),
            "row_count": int(len(indirect_proxy_rows)),
            "year_counts": _year_counter(indirect_proxy_rows),
        },
        {
            "support_family": "validation_only",
            "metric_names": list(VALIDATION_ONLY_METRICS),
            "row_count": int(len(validation_only_rows)),
            "year_counts": _year_counter(validation_only_rows),
        },
    ]
    support_summary = {
        "ground_truth_summary": inputs.ground_truth_summary,
        "coverage_rows": coverage_rows,
        "target_leaky_metric_names": sorted({str(row.get("metric_name") or "") for row in target_leaky_rows}),
    }
    write_json(ctx.experiment_dir / "incidence_signal_coverage_summary.json", support_summary)

    all_years = sorted({year for coverage in coverage_rows for year in coverage["year_counts"].keys()})
    row_labels = [str(coverage["support_family"]) for coverage in coverage_rows]
    matrix = np.asarray(
        [[float(coverage["year_counts"].get(year, 0)) for year in all_years] for coverage in coverage_rows],
        dtype=np.float32,
    )
    _write_heatmap_png(
        ctx.experiment_dir / "incidence_support_heatmap.png",
        title="Incidence Support By Year",
        row_labels=row_labels,
        column_labels=[str(year) for year in all_years],
        values=matrix,
    )

    latent_incidence_layer_justified = bool(direct_inflow_validation_rows and diagnosis_flow_rows and stock_anchor_rows)
    recommendation = (
        "latent_inflow_with_locked_diagnosis_and_validation_only_incidence"
        if latent_incidence_layer_justified and not direct_inflow_training_rows
        else "direct_inflow_training_supported"
        if direct_inflow_training_rows
        else "do_not_add_pre_u_layer_yet"
    )
    audit_payload = {
        "source_run_id": ctx.source_run_id,
        "direct_inflow_training_row_count": int(len(direct_inflow_training_rows)),
        "direct_inflow_validation_only_row_count": int(len(direct_inflow_validation_rows)),
        "diagnosis_flow_row_count": int(len(diagnosis_flow_rows)),
        "stock_anchor_row_count": int(len(stock_anchor_rows)),
        "indirect_proxy_row_count": int(len(indirect_proxy_rows)),
        "validation_only_row_count": int(len(validation_only_rows)),
        "target_leaky_row_count": int(len(target_leaky_rows)),
        "latent_incidence_layer_justified": latent_incidence_layer_justified,
        "recommended_starting_branch": recommendation,
        "notes": [
            "Annual new infections are present, but currently arrive as model_estimate rows and must be treated as validation-only evidence.",
            "Diagnosis-flow and stock-anchor evidence are present and justify testing a locked-diagnosis pre-U branch.",
        ],
    }
    write_json(ctx.experiment_dir / "incidence_evidence_audit.json", audit_payload)

    experiment_spec = {
        "experiment_id": INC00A_EXPERIMENT_ID,
        "description": ctx.experiment.description,
        "source_run_id": ctx.source_run_id,
        "expected_outputs": list(ctx.experiment.expected_outputs),
    }
    coverage_summary = {
        "direct_inflow_training_row_count": int(len(direct_inflow_training_rows)),
        "direct_inflow_validation_only_row_count": int(len(direct_inflow_validation_rows)),
        "diagnosis_flow_row_count": int(len(diagnosis_flow_rows)),
        "stock_anchor_row_count": int(len(stock_anchor_rows)),
        "indirect_proxy_row_count": int(len(indirect_proxy_rows)),
        "validation_only_row_count": int(len(validation_only_rows)),
        "target_leaky_row_count": int(len(target_leaky_rows)),
        "heatmap_year_count": int(len(all_years)),
    }
    decision = {
        "passed": True,
        "latent_incidence_layer_justified": latent_incidence_layer_justified,
        "recommended_starting_branch": recommendation,
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
        "audit": audit_payload,
        "coverage_summary": coverage_summary,
    }


def _classify_population_candidate(row: dict[str, Any]) -> str:
    payload = dict(row.get("payload") or {})
    envelope = dict(row.get("envelope") or {})
    canonical_name = str(payload.get("canonical_name") or "")
    denominator_type = str(payload.get("denominator_type") or "")
    geo_scope = str(payload.get("geo_scope") or "")
    measurement_type = str(payload.get("measurement_type") or "")
    value_semantics = str(payload.get("value_semantics") or "")
    kp_group = str(payload.get("kp_group") or "")
    age_band = str(payload.get("age_band") or "")
    if canonical_name != "population_count":
        return "unsupported_population_candidate"
    if measurement_type != "count":
        return "unsupported_population_candidate"
    if value_semantics != "direct_observed":
        return "unsupported_population_candidate"
    if not bool(envelope.get("is_direct_measurement")):
        return "unsupported_population_candidate"
    if not bool(envelope.get("is_anchor_eligible")):
        return "unsupported_population_candidate"
    if geo_scope != "national":
        return "non_national"
    if denominator_type == "population" and not kp_group and not age_band:
        return "raw_population_candidate"
    if denominator_type == "population" and bool(kp_group):
        return "national_kp_population_candidate"
    if denominator_type == "population" and bool(age_band):
        return "age_specific_population_candidate"
    if denominator_type == "plhiv":
        return "plhiv_population_candidate"
    return "unsupported_population_candidate"


def _candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    candidate_class = str(row.get("candidate_class") or "")
    return (
        candidate_class != "raw_population_candidate",
        candidate_class == "unsupported_population_candidate",
        not bool(row.get("is_direct_measurement")),
        not bool(row.get("is_anchor_eligible")),
        str(row.get("time") or ""),
        -float(row.get("confidence") or 0.0),
        str(row.get("candidate_id") or ""),
    )


def run_inc_00b(ctx: IncidenceResearchContext) -> dict[str, Any]:
    inputs = load_incidence_audit_inputs(ctx)
    classified_rows: list[dict[str, Any]] = []
    for row in inputs.population_candidate_rows:
        payload = dict(row.get("payload") or {})
        envelope = dict(row.get("envelope") or {})
        classified_rows.append(
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "candidate_class": _classify_population_candidate(row),
                "canonical_name": str(payload.get("canonical_name") or ""),
                "geo_scope": str(payload.get("geo_scope") or ""),
                "denominator_type": str(payload.get("denominator_type") or ""),
                "kp_group": str(payload.get("kp_group") or ""),
                "age_band": str(payload.get("age_band") or ""),
                "measurement_type": str(payload.get("measurement_type") or ""),
                "value": payload.get("value"),
                "value_semantics": str(payload.get("value_semantics") or ""),
                "time": str(envelope.get("time") or ""),
                "confidence": float(envelope.get("confidence") or 0.0),
                "is_anchor_eligible": bool(envelope.get("is_anchor_eligible")),
                "is_direct_measurement": bool(envelope.get("is_direct_measurement")),
                "candidate_text": str(envelope.get("candidate_text") or ""),
            }
        )
    classified_rows.sort(key=_candidate_sort_key)
    selected_row = next((row for row in classified_rows if row["candidate_class"] == "raw_population_candidate"), None)
    selected_denominator_supported = selected_row is not None
    recommendation = (
        "use_observed_population_denominator"
        if selected_denominator_supported
        else "do_not_add_raw_population_denominator_yet"
    )
    summary_payload = {
        "source_candidate_bank_path": str(inputs.population_candidate_bank_path) if inputs.population_candidate_bank_path is not None else None,
        "candidate_class_counts": dict(Counter(str(row["candidate_class"]) for row in classified_rows)),
        "selected_denominator_supported": selected_denominator_supported,
        "selected_denominator": selected_row,
        "recommendation": recommendation,
        "top_candidates": classified_rows[:10],
    }
    write_json(ctx.experiment_dir / "population_denominator_audit.json", summary_payload)
    alignment_summary = {
        "candidate_bank_found": inputs.population_candidate_bank_path is not None,
        "population_candidate_row_count": int(len(classified_rows)),
        "raw_population_candidate_count": int(sum(row["candidate_class"] == "raw_population_candidate" for row in classified_rows)),
        "national_kp_population_candidate_count": int(sum(row["candidate_class"] == "national_kp_population_candidate" for row in classified_rows)),
        "plhiv_population_candidate_count": int(sum(row["candidate_class"] == "plhiv_population_candidate" for row in classified_rows)),
        "selected_denominator_supported": selected_denominator_supported,
    }
    write_json(ctx.experiment_dir / "population_alignment_summary.json", alignment_summary)

    experiment_spec = {
        "experiment_id": INC00B_EXPERIMENT_ID,
        "description": ctx.experiment.description,
        "source_run_id": ctx.source_run_id,
        "expected_outputs": list(ctx.experiment.expected_outputs),
    }
    decision = {
        "passed": True,
        "selected_denominator_supported": selected_denominator_supported,
        "recommendation": recommendation,
    }
    write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=alignment_summary,
        decision=decision,
        numeric_justification=_base_numeric_policy(),
    )
    return {
        "decision": decision,
        "audit": summary_payload,
        "coverage_summary": alignment_summary,
    }


def _pearson_correlation(paired_values: list[tuple[float, float]]) -> float | None:
    if len(paired_values) < 2:
        return None
    x = np.asarray([left for left, _ in paired_values], dtype=np.float64)
    y = np.asarray([right for _, right in paired_values], dtype=np.float64)
    if float(np.std(x)) <= float(np.finfo(np.float32).eps):
        return None
    if float(np.std(y)) <= float(np.finfo(np.float32).eps):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_inc_00c(ctx: IncidenceResearchContext) -> dict[str, Any]:
    inputs = load_incidence_audit_inputs(ctx)
    annual_inflow_rows = _metric_rows(inputs.historical_metric_rows, DIRECT_INFLOW_METRICS)
    diagnosis_flow_rows = _metric_rows(inputs.historical_metric_rows, DIAGNOSIS_FLOW_METRICS)
    stock_anchor_rows = _metric_rows(inputs.historical_metric_rows, STOCK_ANCHOR_METRICS)

    annualized_diagnosis_rows = _annualize_diagnosis_flow(diagnosis_flow_rows)
    inflow_validation_rows = [row for row in annual_inflow_rows if _is_model_estimate(row)]
    inflow_training_rows = [row for row in annual_inflow_rows if _is_program_observed(row)]

    annual_inflow_by_year = {
        int(_year_from_row(row)): float(row["value"])
        for row in inflow_validation_rows + inflow_training_rows
        if _year_from_row(row) is not None and row.get("value") not in (None, "")
    }
    annual_diagnosis_by_year = {int(row["year"]): float(row["value"]) for row in annualized_diagnosis_rows}
    overlapping_years = sorted(set(annual_inflow_by_year) & set(annual_diagnosis_by_year))
    paired_values = [(annual_inflow_by_year[year], annual_diagnosis_by_year[year]) for year in overlapping_years]
    inflow_vs_diagnosis_corr = _pearson_correlation(paired_values)

    if inflow_training_rows:
        identifiability_class = "direct_inflow_training_support_present"
    elif inflow_validation_rows and annualized_diagnosis_rows and stock_anchor_rows:
        identifiability_class = "weakly_identified_requires_locked_diagnosis"
    elif annualized_diagnosis_rows and stock_anchor_rows:
        identifiability_class = "diagnosis_and_stock_supported_but_inflow_unidentified"
    else:
        identifiability_class = "not_identified"

    missing_evidence: list[str] = []
    if not inflow_training_rows:
        missing_evidence.append("independent_direct_inflow_training_observations")
    if not any(_classify_population_candidate(row) == "raw_population_candidate" for row in inputs.population_candidate_rows):
        missing_evidence.append("observed_national_population_denominator")
    if len(overlapping_years) < 2:
        missing_evidence.append("multi_year_overlap_between_inflow_and_annualized_diagnosis")

    stress_rows = [
        {
            "scenario": "direct_inflow_training_support",
            "supported": bool(inflow_training_rows),
            "row_count": int(len(inflow_training_rows)),
        },
        {
            "scenario": "validation_only_inflow_support",
            "supported": bool(inflow_validation_rows),
            "row_count": int(len(inflow_validation_rows)),
        },
        {
            "scenario": "annualized_diagnosis_support",
            "supported": bool(annualized_diagnosis_rows),
            "row_count": int(len(annualized_diagnosis_rows)),
        },
        {
            "scenario": "stock_anchor_support",
            "supported": bool(stock_anchor_rows),
            "row_count": int(len(stock_anchor_rows)),
        },
    ]
    write_json(ctx.experiment_dir / "identifiability_stress_table.json", stress_rows)

    identifiability_payload = {
        "source_run_id": ctx.source_run_id,
        "identifiability_class": identifiability_class,
        "inflow_training_row_count": int(len(inflow_training_rows)),
        "inflow_validation_row_count": int(len(inflow_validation_rows)),
        "annualized_diagnosis_row_count": int(len(annualized_diagnosis_rows)),
        "stock_anchor_row_count": int(len(stock_anchor_rows)),
        "overlapping_years": overlapping_years,
        "inflow_vs_diagnosis_pearson_r": inflow_vs_diagnosis_corr,
        "missing_evidence": missing_evidence,
    }
    write_json(ctx.experiment_dir / "incidence_identifiability_audit.json", identifiability_payload)

    experiment_spec = {
        "experiment_id": INC00C_EXPERIMENT_ID,
        "description": ctx.experiment.description,
        "source_run_id": ctx.source_run_id,
        "expected_outputs": list(ctx.experiment.expected_outputs),
    }
    coverage_summary = {
        "inflow_training_row_count": int(len(inflow_training_rows)),
        "inflow_validation_row_count": int(len(inflow_validation_rows)),
        "annualized_diagnosis_row_count": int(len(annualized_diagnosis_rows)),
        "stock_anchor_row_count": int(len(stock_anchor_rows)),
        "overlap_year_count": int(len(overlapping_years)),
    }
    decision = {
        "passed": True,
        "identifiability_class": identifiability_class,
        "requires_locked_diagnosis": identifiability_class == "weakly_identified_requires_locked_diagnosis",
    }
    numeric_policy = _base_numeric_policy() + [
        {
            "name": "minimum_paired_years_for_pearson_correlation",
            "value": 2,
            "role": "Minimum paired observations required before a Pearson correlation is mathematically meaningful.",
            "source_type": "physical_constraint",
            "estimation_data": "Pearson correlation requires at least two paired observations.",
            "estimation_method": "definition of two-point sample correlation",
            "uncertainty": "none",
            "why_needed": "The identifiability audit reports inflow-versus-diagnosis correlation only when overlap is sufficient.",
        }
    ]
    write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=numeric_policy,
    )
    return {
        "decision": decision,
        "audit": identifiability_payload,
        "coverage_summary": coverage_summary,
    }
