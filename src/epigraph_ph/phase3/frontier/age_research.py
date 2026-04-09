from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.runtime import read_json, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .decomposition import _leave_one_out_slope_only_r2, _simulate_holdout_from_hazard_map
from .numeric_policy import numerical_guard_entry
from .peak_windows import PEAK01F_EXPERIMENT_ID, _load_peak_gated_reference
from .sources import load_transition_research_inputs
from .transition_engine import (
    _build_observation_payload,
    _discover_latest_transition_experiment,
    _diagnosis_flow_mae,
    _holdout_rows,
    _holdout_years_from_reference,
    _metric_scales,
    _normalized_mae,
    _smape,
)


AGE01A_EXPERIMENT_ID = "AGE-01A-age-evidence-audit"
AGE01B_EXPERIMENT_ID = "AGE-01B-youth-diagnosis-modifier"
AGE01C_EXPERIMENT_ID = "AGE-01C-youth-downstream-modifier"
AGE_GROUP_ORDER: tuple[str, ...] = ("0-14", "15-24", "25+")
SUPPORT_CLASS_ORDER: tuple[str, ...] = ("direct_observation", "auxiliary_proxy", "prior_only")
AGE01C_TRANSITIONS: tuple[str, ...] = ("D_to_A", "A_to_V")
MONTH_NAME_TO_NUMBER: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _year_from_row(row: dict[str, Any]) -> int | None:
    raw_year = row.get("year")
    if raw_year not in (None, ""):
        try:
            return int(raw_year)
        except (TypeError, ValueError):
            pass
    for key in ("time", "period_end", "period_start"):
        raw_value = str(row.get(key) or "").strip()
        if len(raw_value) >= 4 and raw_value[:4].isdigit():
            return int(raw_value[:4])
    return None


def _count_metric_by_year(rows: list[dict[str, Any]], metric_name: str, analysis_years: list[int]) -> dict[int, int]:
    counts = {int(year): 0 for year in analysis_years}
    for row in rows:
        if str(row.get("metric_name") or "") != metric_name:
            continue
        year_value = _year_from_row(row)
        if year_value in counts:
            counts[int(year_value)] += 1
    return counts


def _to_float(value: str | None) -> float | None:
    token = str(value or "").replace(",", "").strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _capture_first_number(text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if not match:
            continue
        value = _to_float(match.group(1))
        if value is not None:
            return value
    return None


def _page_catalog_year(source_label: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", str(source_label))
    if not match:
        return None
    return int(match.group(1))


def _page_catalog_quarter(source_label: str) -> str | None:
    year_value = _page_catalog_year(source_label)
    if year_value is None:
        return None
    month_matches = re.findall(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        str(source_label),
        flags=re.I,
    )
    if not month_matches:
        return None
    end_month_number = MONTH_NAME_TO_NUMBER.get(str(month_matches[-1]).lower())
    if end_month_number is None:
        return None
    return f"{int(year_value):04d}-Q{((int(end_month_number) - 1) // 3) + 1}"


def _page_catalog_total_diagnosed_cases(text: str) -> float | None:
    return _capture_first_number(
        text,
        (
            r"there were\s*(\d{1,3}(?:,\d{3})*)\s*new hiv",
            r"there were\s*(\d{1,3}(?:,\d{3})*)\s*confirmed hiv-positive individuals reported",
            r"there were\s*(\d{1,3}(?:,\d{3})*)\s*people with hiv diagnosed",
            r"the newly diagnosed cases[^\d]{0,30}(\d{1,3}(?:,\d{3})*)",
            r"Total Reported Cases[^\d]{0,40}(\d{1,3}(?:,\d{3})*)",
        ),
    )


def _page_catalog_youth_percent(text: str) -> float | None:
    return _capture_first_number(
        text,
        (
            r"\b\d{1,3}(?:,\d{3})*\s*\((\d{1,3})%\)\s*of the reported cases\s*this month were among the youth aged 15-24",
            r"\b\d{1,3}(?:,\d{3})*\s*\((\d{1,3})%\)\s*were\s*15-24\s*years?\s*old",
            r"\b(\d{1,3})%\s*\(\d{1,3}(?:,\d{3})*\)\s*were\s*15-24\s*years?\s*old",
            r"\b\d{1,3}(?:,\d{3})*\s*\((\d{1,3})%\)\s*were\s*youth aged 15-24",
            r"\b(\d{1,3})%\s*\(\d{1,3}(?:,\d{3})*\)\s*were\s*youth aged 15-24",
            r"while\s*(\d{1,3})%\s*were\s*youth aged 15-24",
            r"(\d{1,3})%\s*were\s*youth aged 15-24",
            r"while\s*(\d{1,3})%\s*were\s*15-24\s*years?\s*old",
            r"(\d{1,3})%\s*were\s*15-24\s*years?\s*old",
        ),
    )


def _page_catalog_youth_count(text: str) -> tuple[float | None, str]:
    explicit_count = _capture_first_number(
        text,
        (
            r"\b(\d{1,3}(?:,\d{3})*)\s*\(\d{1,3}%\)\s*of the reported cases\s*this month were among the youth aged 15-24",
            r"\b(\d{1,3}(?:,\d{3})*)\s*\(\d{1,3}%\)\s*were\s*15-24\s*years?\s*old",
            r"\b\d{1,3}%\s*\((\d{1,3}(?:,\d{3})*)\)\s*were\s*15-24\s*years?\s*old",
            r"\b(\d{1,3}(?:,\d{3})*)\s*\(\d{1,3}%\)\s*were\s*youth aged 15-24",
            r"\b\d{1,3}%\s*\((\d{1,3}(?:,\d{3})*)\)\s*were\s*youth aged 15-24",
            r"Youth\s*15-24yo[^\d]{0,20}(\d{1,3}(?:,\d{3})*)",
            r"15-24\s*y/o[^\d]{0,20}(\d{1,3}(?:,\d{3})*)",
        ),
    )
    if explicit_count is not None:
        return float(explicit_count), "direct_count"
    explicit_share = _capture_first_number(
        text,
        (
            r"while\s*(\d{1,3})%\s*were\s*youth aged 15-24",
            r"(\d{1,3})%\s*were\s*youth aged 15-24",
            r"while\s*(\d{1,3})%\s*were\s*15-24\s*years?\s*old",
            r"(\d{1,3})%\s*were\s*15-24\s*years?\s*old",
        ),
    )
    total_cases = _page_catalog_total_diagnosed_cases(text)
    if explicit_share is None or total_cases is None:
        return None, "unsupported"
    derived_count = round((float(explicit_share) / 100.0) * float(total_cases))
    return float(derived_count), "derived_from_reported_share"


def _page_catalog_youth_observation(row: dict[str, Any]) -> dict[str, Any] | None:
    if int(row.get("page_number") or 0) != 1:
        return None
    source_label = str(row.get("source_label") or "")
    year_value = _page_catalog_year(source_label)
    quarter_value = _page_catalog_quarter(source_label)
    if year_value is None or quarter_value is None:
        return None
    text = str(row.get("text") or "")
    youth_count, extraction_kind = _page_catalog_youth_count(text)
    if youth_count is None:
        return None
    total_cases = _page_catalog_total_diagnosed_cases(text)
    youth_percent = _page_catalog_youth_percent(text)
    youth_share = None
    if total_cases is not None and float(total_cases) >= float(youth_count) and float(total_cases) > 0.0:
        youth_share = float(youth_count) / float(total_cases)
    elif youth_percent is not None and 0.0 < float(youth_percent) <= 100.0:
        youth_share = float(youth_percent) / 100.0
        total_cases = None
    return {
        "source_id": str(row.get("source_id") or ""),
        "source_label": source_label,
        "year": int(year_value),
        "quarter": quarter_value,
        "youth_count": float(youth_count),
        "total_cases": float(total_cases) if total_cases is not None else None,
        "youth_share": float(youth_share) if youth_share is not None else None,
        "extraction_kind": extraction_kind,
    }


def _page_catalog_youth_rows(page_catalog_rows: list[dict[str, Any]], analysis_years: list[int]) -> list[dict[str, Any]]:
    supported_years = {int(year) for year in analysis_years}
    seen_sources: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in page_catalog_rows:
        source_id = str(row.get("source_id") or "")
        if not source_id or source_id in seen_sources:
            continue
        observation = _page_catalog_youth_observation(row)
        if observation is None:
            continue
        if int(observation["year"]) not in supported_years:
            continue
        rows.append(observation)
        seen_sources.add(source_id)
    return rows


def _metric_row_quarter(row: dict[str, Any]) -> str | None:
    month_label = str(row.get("period_end") or row.get("time") or "").strip()
    if not month_label:
        return None
    try:
        year_text, month_text = month_label.split("-", maxsplit=1)
        month_number = int(month_text[:2])
        return f"{int(year_text):04d}-Q{((month_number - 1) // 3) + 1}"
    except (TypeError, ValueError):
        return None


def _quarter_metric_stats(
    historical_metric_rows: list[dict[str, Any]],
    *,
    metric_name: str,
    analysis_years: list[int],
) -> dict[str, dict[str, Any]]:
    supported_years = {int(year) for year in analysis_years}
    grouped: dict[str, dict[str, Any]] = {}
    for row in historical_metric_rows:
        if str(row.get("metric_name") or "") != metric_name:
            continue
        if str(row.get("region") or "").lower() != "national":
            continue
        quarter = _metric_row_quarter(row)
        if not quarter:
            continue
        if int(str(quarter).split("-", maxsplit=1)[0]) not in supported_years:
            continue
        bucket = grouped.setdefault(
            quarter,
            {
                "value": 0.0,
                "row_count": 0,
                "temporal_precisions": set(),
            },
        )
        bucket["value"] = float(bucket["value"]) + float(row.get("value") or 0.0)
        bucket["row_count"] = int(bucket["row_count"]) + 1
        bucket["temporal_precisions"].add(str(row.get("temporal_precision") or ""))
    return grouped


def _count_youth_page_catalog_by_year(page_catalog_rows: list[dict[str, Any]], analysis_years: list[int]) -> dict[int, int]:
    counts = {int(year): 0 for year in analysis_years}
    for observation in _page_catalog_youth_rows(page_catalog_rows, analysis_years):
        year_value = int(observation["year"])
        if year_value in counts:
            counts[year_value] += 1
    return counts


def _aggregate_prior_groups(subgroup_weight_summary: dict[str, Any]) -> dict[str, float]:
    age_distribution = dict(subgroup_weight_summary.get("national_age_distribution") or {})
    fifteen_twenty_four = float(age_distribution.get("15_24") or 0.0)
    twenty_five_plus = float(
        float(age_distribution.get("25_34") or 0.0)
        + float(age_distribution.get("35_49") or 0.0)
        + float(age_distribution.get("50_plus") or 0.0)
    )
    return {
        "0-14": 0.0,
        "15-24": fifteen_twenty_four,
        "25+": twenty_five_plus,
    }


def _support_presence_row(*, age_group: str, support_class: str, counts_by_year: dict[int, int]) -> dict[str, Any]:
    year_count_with_support = sum(1 for value in counts_by_year.values() if int(value) > 0)
    return {
        "age_group": age_group,
        "support_class": support_class,
        "yearly_counts": {str(year): int(count) for year, count in counts_by_year.items()},
        "year_count_with_support": int(year_count_with_support),
        "row_count": int(sum(int(count) for count in counts_by_year.values())),
    }


def _quarter_youth_share_series(
    historical_metric_rows: list[dict[str, Any]],
    page_catalog_rows: list[dict[str, Any]],
    analysis_years: list[int],
) -> list[dict[str, Any]]:
    youth_metric_stats = _quarter_metric_stats(
        historical_metric_rows,
        metric_name="youth_cases_15_24_period",
        analysis_years=analysis_years,
    )
    diagnosed_metric_stats = _quarter_metric_stats(
        historical_metric_rows,
        metric_name="new_diagnosed_cases_period",
        analysis_years=analysis_years,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in _page_catalog_youth_rows(page_catalog_rows, analysis_years):
        grouped[str(observation["quarter"])].append(observation)
    rows: list[dict[str, Any]] = []
    quarter_axis = sorted(set(grouped) | set(youth_metric_stats) | set(diagnosed_metric_stats))
    for quarter in quarter_axis:
        observations = grouped.get(quarter, [])
        youth_metric_bucket = dict(youth_metric_stats.get(quarter) or {})
        diagnosed_metric_bucket = dict(diagnosed_metric_stats.get(quarter) or {})
        metric_youth_cases = youth_metric_bucket.get("value")
        metric_total_cases = diagnosed_metric_bucket.get("value")
        denominator_temporal_precisions = {str(value) for value in list(diagnosed_metric_bucket.get("temporal_precisions") or [])}
        denominator_is_complete = bool(
            ("quarterly_snapshot" in denominator_temporal_precisions)
            or int(diagnosed_metric_bucket.get("row_count") or 0) >= 3
        )
        if (
            denominator_is_complete
            and
            metric_youth_cases is not None
            and metric_total_cases is not None
            and float(metric_total_cases) > 0.0
            and float(metric_youth_cases) <= float(metric_total_cases)
        ):
            rows.append(
                {
                    "quarter": str(quarter),
                    "year": int(str(quarter).split("-", maxsplit=1)[0]),
                    "report_count": int(len(observations)),
                    "youth_cases": round(float(metric_youth_cases), 6),
                    "total_cases": round(float(metric_total_cases), 6),
                    "youth_share": round(float(metric_youth_cases / metric_total_cases), 6),
                    "extraction_kinds": ["archive_metric_ratio"],
                    "source_labels": [str(row["source_label"]) for row in observations],
                }
            )
            continue
        observed_total_cases = sum(
            float(row["total_cases"])
            for row in observations
            if row.get("total_cases") is not None and float(row.get("total_cases") or 0.0) > 0.0
        )
        observed_youth_cases = sum(float(row["youth_count"]) for row in observations)
        if observed_total_cases > 0.0:
            youth_share = float(observed_youth_cases / observed_total_cases)
        else:
            youth_share_values = [float(row["youth_share"]) for row in observations if row.get("youth_share") is not None]
            youth_share = float(np.mean(youth_share_values)) if youth_share_values else 0.0
        rows.append(
            {
                "quarter": str(quarter),
                "year": int(str(quarter).split("-", maxsplit=1)[0]),
                "report_count": int(len(observations)),
                "youth_cases": round(float(observed_youth_cases), 6),
                "total_cases": round(float(observed_total_cases), 6) if observed_total_cases > 0.0 else None,
                "youth_share": round(float(youth_share), 6),
                "extraction_kinds": sorted({str(row["extraction_kind"]) for row in observations}),
                "source_labels": [str(row["source_label"]) for row in observations],
            }
        )
    return rows


def _least_squares_intercept_and_slope(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    if x_values.size == 0 or y_values.size == 0 or x_values.size != y_values.size:
        return 0.0, 0.0
    design = np.column_stack(
        [
            np.ones((int(x_values.size),), dtype=np.float64),
            x_values.astype(np.float64),
        ]
    )
    solution, *_ = np.linalg.lstsq(design, y_values.astype(np.float64), rcond=None)
    return float(solution[0]), float(solution[1])


def _load_age_experiment_reference(experiment_id: str) -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_transition_experiment(experiment_id)
    baseline_comparison = read_json(experiment_dir / "baseline_comparison.json", default={})
    evaluation = read_json(experiment_dir / "evaluation.json", default={})
    mechanistic_forecast = read_json(experiment_dir / "mechanistic_forecast.json", default={})
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "baseline_comparison": dict(baseline_comparison or {}),
        "evaluation": dict(evaluation or {}),
        "mechanistic_forecast": dict(mechanistic_forecast or {}),
    }


def run_age_01a(ctx: TransitionResearchContext) -> dict[str, Any]:
    inputs = load_transition_research_inputs(ctx)
    analysis_years = [int(year) for year in inputs.analysis_years]
    historical_metric_rows = [
        dict(row)
        for row in list(read_json(ctx.source_run_dir / "harp_archive" / "historical_metric_rows.json", default=[]) or [])
    ]
    page_catalog_rows = [
        dict(row) for row in list(read_json(ctx.source_run_dir / "harp_archive" / "page_catalog.json", default=[]) or [])
    ]
    subgroup_weight_summary = dict(inputs.subgroup_weight_summary or {})
    subgroup_prior_learning_summary = dict(inputs.subgroup_prior_learning_summary or {})

    youth_metric_counts = _count_metric_by_year(historical_metric_rows, "youth_cases_15_24_period", analysis_years)
    youth_page_catalog_counts = _count_youth_page_catalog_by_year(page_catalog_rows, analysis_years)
    youth_direct_counts = (
        youth_metric_counts
        if sum(int(value) for value in youth_metric_counts.values()) > 0
        else youth_page_catalog_counts
    )
    youth_direct_evidence_kind = (
        "archive_extracted_metric"
        if sum(int(value) for value in youth_metric_counts.values()) > 0
        else "archive_page_catalog_direct_extraction"
    )
    youth_direct_metric_names = (
        ["youth_cases_15_24_period"]
        if youth_direct_evidence_kind == "archive_extracted_metric"
        else ["Youth 15-24yo quick facts"]
    )
    auxiliary_counts = _count_metric_by_year(historical_metric_rows, "art_median_age", analysis_years)
    prior_groups = _aggregate_prior_groups(subgroup_weight_summary)
    prior_counts_by_group = {
        age_group: {int(year): (1 if float(prior_groups.get(age_group) or 0.0) > 0.0 else 0) for year in analysis_years}
        for age_group in AGE_GROUP_ORDER
    }

    support_rows = [
        {
            **_support_presence_row(
                age_group="15-24",
                support_class="direct_observation",
                counts_by_year=youth_direct_counts,
            ),
            "metric_names": youth_direct_metric_names,
            "evidence_kind": youth_direct_evidence_kind,
        },
        {
            **_support_presence_row(
                age_group="all-ages",
                support_class="auxiliary_proxy",
                counts_by_year=auxiliary_counts,
            ),
            "metric_names": ["art_median_age"],
            "evidence_kind": "archive_extracted_metric",
        },
        {
            **_support_presence_row(
                age_group="15-24",
                support_class="prior_only",
                counts_by_year=prior_counts_by_group["15-24"],
            ),
            "metric_names": ["subgroup_weight_summary.national_age_distribution.15_24"],
            "evidence_kind": "phase3_prior_summary",
        },
        {
            **_support_presence_row(
                age_group="25+",
                support_class="prior_only",
                counts_by_year=prior_counts_by_group["25+"],
            ),
            "metric_names": [
                "subgroup_weight_summary.national_age_distribution.25_34",
                "subgroup_weight_summary.national_age_distribution.35_49",
                "subgroup_weight_summary.national_age_distribution.50_plus",
            ],
            "evidence_kind": "phase3_prior_summary",
        },
        {
            **_support_presence_row(
                age_group="0-14",
                support_class="prior_only",
                counts_by_year=prior_counts_by_group["0-14"],
            ),
            "metric_names": [],
            "evidence_kind": "phase3_prior_summary",
        },
    ]

    youth_direct_year_count = sum(1 for value in youth_direct_counts.values() if int(value) > 0)
    auxiliary_year_count = sum(1 for value in auxiliary_counts.values() if int(value) > 0)
    prior_fifteen_twenty_four_present = bool(float(prior_groups.get("15-24") or 0.0) > 0.0)
    prior_twenty_five_plus_present = bool(float(prior_groups.get("25+") or 0.0) > 0.0)
    prior_zero_fourteen_present = bool(float(prior_groups.get("0-14") or 0.0) > 0.0)

    if youth_direct_year_count > 0 and prior_fifteen_twenty_four_present and prior_twenty_five_plus_present:
        recommended_grouping = "15-24_vs_25+"
        recommended_action = "age_modifier_branch_justified"
        recommendation_reason = "Direct youth observation exists and can be paired with existing prior support for an older residual band."
    elif prior_fifteen_twenty_four_present and prior_twenty_five_plus_present and auxiliary_year_count > 0:
        recommended_grouping = "15-24_vs_25+"
        recommended_action = "audit_only_direct_youth_signal_missing"
        recommendation_reason = "The residual two-band grouping exists in prior space and auxiliary age proxy space, but no direct youth observation is present in the packaged archive metrics."
    else:
        recommended_grouping = "defer_age_branch"
        recommended_action = "insufficient_age_support"
        recommendation_reason = "The current transition-research inputs do not provide enough age support to justify an age-conditioned forecast branch."

    age_evidence_audit = {
        "source_run_id": ctx.source_run_id,
        "analysis_years": [str(year) for year in analysis_years],
        "support_rows": support_rows,
        "prior_group_distribution": {age_group: round(float(value), 6) for age_group, value in prior_groups.items()},
        "prior_learning_present": bool(subgroup_prior_learning_summary),
        "prior_learning_mean_age_distribution": {
            "15-24": round(float((list(subgroup_prior_learning_summary.get("mean_age_distribution") or [0.0, 0.0, 0.0, 0.0]) + [0.0, 0.0, 0.0, 0.0])[0]), 6),
            "25+": round(
                float(
                    sum(
                        float(value)
                        for value in list(subgroup_prior_learning_summary.get("mean_age_distribution") or [0.0, 0.0, 0.0, 0.0])[1:4]
                    )
                ),
                6,
            ),
        },
        "recommendation": {
            "recommended_grouping": recommended_grouping,
            "recommended_action": recommended_action,
            "reason": recommendation_reason,
            "direct_youth_observation_year_count": int(youth_direct_year_count),
            "direct_youth_evidence_kind": youth_direct_evidence_kind,
            "auxiliary_proxy_year_count": int(auxiliary_year_count),
            "prior_support": {
                "15-24": prior_fifteen_twenty_four_present,
                "25+": prior_twenty_five_plus_present,
                "0-14": prior_zero_fourteen_present,
            },
        },
    }
    write_json(ctx.experiment_dir / "age_evidence_audit.json", age_evidence_audit)

    coverage_summary = {
        "source_run_id": ctx.source_run_id,
        "analysis_year_count": len(analysis_years),
        "direct_youth_row_count": int(sum(youth_direct_counts.values())),
        "direct_youth_year_count": int(youth_direct_year_count),
        "direct_youth_evidence_kind": youth_direct_evidence_kind,
        "auxiliary_proxy_row_count": int(sum(auxiliary_counts.values())),
        "auxiliary_proxy_year_count": int(auxiliary_year_count),
        "prior_15_24_present": bool(prior_fifteen_twenty_four_present),
        "prior_25_plus_present": bool(prior_twenty_five_plus_present),
        "prior_0_14_present": bool(prior_zero_fourteen_present),
        "recommended_grouping": recommended_grouping,
        "recommended_action": recommended_action,
    }
    write_json(ctx.experiment_dir / "age_signal_coverage_summary.json", coverage_summary)

    heatmap_rows = [
        ("15-24 | direct_observation", youth_direct_counts),
        ("all-ages | auxiliary_proxy", auxiliary_counts),
        ("15-24 | prior_only", prior_counts_by_group["15-24"]),
        ("25+ | prior_only", prior_counts_by_group["25+"]),
        ("0-14 | prior_only", prior_counts_by_group["0-14"]),
    ]
    heatmap_labels = [label for label, _ in heatmap_rows]
    heatmap_matrix = np.asarray(
        [[float(counts.get(year) or 0.0) for year in analysis_years] for _, counts in heatmap_rows],
        dtype=np.float32,
    )
    fig, ax = plt.subplots()
    image = ax.imshow(heatmap_matrix, aspect="auto")
    ax.set_title("AGE-01A Age Evidence Coverage")
    ax.set_xticks(np.arange(len(analysis_years)), labels=[str(year) for year in analysis_years])
    ax.set_yticks(np.arange(len(heatmap_labels)), labels=heatmap_labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "age_evidence_heatmap.png")
    plt.close(fig)

    decision = {
        "completed": True,
        "keep": True,
        "reason": recommendation_reason,
        "age_branch_justified": bool(recommended_action == "age_modifier_branch_justified"),
        "checks": [
            {
                "name": "direct_youth_observation_present",
                "passed": bool(youth_direct_year_count > 0),
                "actual": int(youth_direct_year_count),
                "target": ">=1 year",
            },
            {
                "name": "two_band_prior_support_present",
                "passed": bool(prior_fifteen_twenty_four_present and prior_twenty_five_plus_present),
                "actual": {
                    "15-24": bool(prior_fifteen_twenty_four_present),
                    "25+": bool(prior_twenty_five_plus_present),
                },
                "target": True,
            },
            {
                "name": "0_14_remains_deferred_without_direct_support",
                "passed": bool(not prior_zero_fourteen_present),
                "actual": bool(prior_zero_fourteen_present),
                "target": False,
            },
        ],
    }

    numeric_justification = [
        {
            "name": "age_group_boundary_15_24_lower",
            "value": 15,
            "role": "archive_native_youth_band_lower_bound",
            "source_type": "estimated",
            "estimation_data": "archive-native metric name youth_cases_15_24_period and AGE-01A plan grouping",
            "estimation_method": "use the archive-native youth surveillance band exactly as written rather than inventing a new lower bound",
            "uncertainty": "none once the archive-native band is fixed",
            "why_needed": "Defines the only direct youth age band currently supported by the packaged archive metrics.",
        },
        {
            "name": "age_group_boundary_15_24_upper",
            "value": 24,
            "role": "archive_native_youth_band_upper_bound",
            "source_type": "estimated",
            "estimation_data": "archive-native metric name youth_cases_15_24_period and AGE-01A plan grouping",
            "estimation_method": "use the archive-native youth surveillance band exactly as written rather than inventing a new upper bound",
            "uncertainty": "none once the archive-native band is fixed",
            "why_needed": "Defines the only direct youth age band currently supported by the packaged archive metrics.",
        },
        {
            "name": "age_group_boundary_25_plus_lower",
            "value": 25,
            "role": "residual_older_band_lower_bound",
            "source_type": "estimated",
            "estimation_data": "AGE-01A plan grouping plus subgroup_weight_summary national_age_distribution bins 25_34, 35_49, and 50_plus",
            "estimation_method": "aggregate all prior-supported adult age bins starting at 25 into a single residual older band",
            "uncertainty": "inherits subgroup prior aggregation uncertainty",
            "why_needed": "Allows the audit to test a minimal two-band split without inventing additional unsupported age bands.",
        },
        {
            "name": "analysis_year_floor",
            "value": int(min(analysis_years)),
            "role": "age_audit_year_floor",
            "source_type": "estimated",
            "estimation_data": "transition research analysis years derived from phase15 month axis and archive support",
            "estimation_method": "minimum year in the typed transition-research analysis window",
            "uncertainty": "none once the source run is fixed",
            "why_needed": "Pins the audit window to the same evidence horizon used elsewhere in the transition-research branch.",
        },
        {
            "name": "analysis_year_ceiling",
            "value": int(max(analysis_years)),
            "role": "age_audit_year_ceiling",
            "source_type": "estimated",
            "estimation_data": "transition research analysis years derived from phase15 month axis and archive support",
            "estimation_method": "maximum year in the typed transition-research analysis window",
            "uncertainty": "none once the source run is fixed",
            "why_needed": "Pins the audit window to the same evidence horizon used elsewhere in the transition-research branch.",
        },
    ]

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "age_group_order": list(AGE_GROUP_ORDER),
            "support_class_order": list(SUPPORT_CLASS_ORDER),
            "direct_youth_evidence_kind": youth_direct_evidence_kind,
        },
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=numeric_justification,
    )
    return {
        "artifacts": artifacts,
        "decision": decision,
        "coverage_summary": coverage_summary,
        "age_evidence_audit": age_evidence_audit,
    }


def run_age_01b(ctx: TransitionResearchContext) -> dict[str, Any]:
    inputs = load_transition_research_inputs(ctx)
    analysis_years = [int(year) for year in inputs.analysis_years]
    historical_metric_rows = [
        dict(row)
        for row in list(read_json(ctx.source_run_dir / "harp_archive" / "historical_metric_rows.json", default=[]) or [])
    ]
    page_catalog_rows = [
        dict(row) for row in list(read_json(ctx.source_run_dir / "harp_archive" / "page_catalog.json", default=[]) or [])
    ]
    youth_quarter_rows = _quarter_youth_share_series(historical_metric_rows, page_catalog_rows, analysis_years)
    youth_share_by_quarter = {
        str(row["quarter"]): float(row["youth_share"])
        for row in youth_quarter_rows
        if row.get("youth_share") is not None
    }

    peak_01f_reference = _load_peak_gated_reference(PEAK01F_EXPERIMENT_ID)
    actual_payload = _build_observation_payload(ctx)
    actual_rows = [dict(row) for row in list(actual_payload["rows"])]
    holdout_years = _holdout_years_from_reference(peak_01f_reference)
    holdout_rows = _holdout_rows(actual_rows, holdout_years)
    if len(holdout_rows) < 2:
        raise ValueError("AGE-01B requires at least two holdout rows from the PEAK-01F reference")

    evaluation_reference = dict(peak_01f_reference.get("evaluation") or {})
    anchored_state_rows = [dict(row) for row in list(evaluation_reference.get("holdout_state_rows") or [])]
    anchored_prediction_rows = [
        dict((row.get("prediction") or {}), quarter=str(row.get("quarter") or ""))
        for row in list(evaluation_reference.get("holdout_rows") or [])
    ]
    if len(anchored_state_rows) != len(anchored_prediction_rows):
        raise ValueError("AGE-01B requires aligned PEAK-01F holdout states and prediction rows")
    anchored_state_row = dict(anchored_state_rows[0])
    anchored_prediction_row = dict(anchored_prediction_rows[0])

    from .transition_engine import MECH01E_EXPERIMENT_ID, _load_transition_experiment_reference  # local import

    mech_01e_reference = _load_transition_experiment_reference(MECH01E_EXPERIMENT_ID)
    train_hazard_rows = [
        dict(row)
        for row in list((mech_01e_reference.get("transition_hazard_summary") or {}).get("train_rows") or [])
    ]
    train_pairs = [
        (
            str(row["quarter"]),
            float((row.get("hazards") or {}).get("U_to_D") or 0.0),
            float(youth_share_by_quarter[str(row["quarter"])]),
        )
        for row in train_hazard_rows
        if str(row.get("quarter") or "") in youth_share_by_quarter
    ]
    if len(train_pairs) < 3:
        raise ValueError("AGE-01B requires at least three aligned train quarters with youth-share support")

    train_quarters = [quarter for quarter, _, _ in train_pairs]
    train_time_index = np.asarray([float(index) for index, _ in enumerate(train_quarters)], dtype=np.float64)
    train_hazards = np.asarray([float(hazard) for _, hazard, _ in train_pairs], dtype=np.float64)
    train_youth_share = np.asarray([float(share) for _, _, share in train_pairs], dtype=np.float64)

    eps = float(np.finfo(np.float32).eps)
    probability_lower_bound = eps
    probability_upper_bound = 1.0 - eps

    trend_intercept, trend_slope = _least_squares_intercept_and_slope(train_time_index, train_hazards)
    train_trend = trend_intercept + (trend_slope * train_time_index)
    train_residual = train_hazards - train_trend
    youth_share_mean = float(np.mean(train_youth_share))
    youth_share_std = float(np.std(train_youth_share))
    if youth_share_std <= eps:
        standardized_train_share = np.zeros_like(train_youth_share, dtype=np.float64)
    else:
        standardized_train_share = (train_youth_share - youth_share_mean) / youth_share_std
    denominator = float(np.dot(standardized_train_share, standardized_train_share))
    numerator = float(np.dot(standardized_train_share, train_residual))
    youth_slope = (numerator / denominator) if denominator > eps else 0.0
    youth_fitted_residual = youth_slope * standardized_train_share
    youth_residual_mse = float(np.mean(np.square(train_residual - youth_fitted_residual))) if train_residual.size else 0.0
    youth_loo_r2, youth_loo_mse = _leave_one_out_slope_only_r2(
        standardized_train_share.astype(np.float32),
        train_residual.astype(np.float32),
        eps,
    )
    residual_std = float(np.std(train_residual)) if train_residual.size else 0.0

    peak_mechanistic_forecast = dict(peak_01f_reference.get("mechanistic_forecast") or {})
    baseline_hazard_map = {
        str(row.get("quarter") or ""): {
            str(transition): float(value)
            for transition, value in dict(row.get("hazards") or {}).items()
        }
        for row in list(peak_mechanistic_forecast.get("transition_hazards") or [])
    }
    holdout_quarters_after_anchor = [str(row["quarter"]) for row in holdout_rows[1:]]
    adjusted_hazard_map: dict[str, dict[str, float]] = {}
    modifier_rows: list[dict[str, Any]] = []
    for holdout_index, quarter in enumerate(holdout_quarters_after_anchor, start=len(train_quarters)):
        base_hazards = {transition: float(value) for transition, value in dict(baseline_hazard_map.get(quarter) or {}).items()}
        youth_share = float(youth_share_by_quarter.get(quarter) or youth_share_mean)
        youth_z = (youth_share - youth_share_mean) / max(youth_share_std, eps) if youth_share_std > eps else 0.0
        raw_correction = float(youth_slope * youth_z)
        gated_correction = float(np.clip(youth_loo_r2 * raw_correction, -residual_std, residual_std))
        adjusted_hazard_map[quarter] = {
            transition: float(base_hazards.get(transition) or 0.0)
            for transition in ("U_to_D", "D_to_A", "A_to_V", "A_to_L", "L_to_A")
        }
        adjusted_hazard_map[quarter]["U_to_D"] = float(
            np.clip(
                float(base_hazards.get("U_to_D") or 0.0) + gated_correction,
                probability_lower_bound,
                probability_upper_bound,
            )
        )
        modifier_rows.append(
            {
                "quarter": quarter,
                "time_index": int(holdout_index),
                "youth_share": round(float(youth_share), 6),
                "youth_share_z": round(float(youth_z), 6),
                "base_u_to_d_hazard": round(float(base_hazards.get("U_to_D") or 0.0), 6),
                "raw_correction": round(float(raw_correction), 6),
                "gated_correction": round(float(gated_correction), 6),
                "adjusted_u_to_d_hazard": round(float(adjusted_hazard_map[quarter]["U_to_D"]), 6),
                "support_available": bool(quarter in youth_share_by_quarter),
            }
        )

    train_rows = [
        dict(row)
        for row in actual_rows
        if str(row.get("quarter") or "") not in {str(item["quarter"]) for item in holdout_rows}
    ]
    testing_shares = [
        float(row["tested_for_viral_load"]) / max(float(row["alive_on_art"]), eps)
        for row in train_rows
        if row.get("tested_for_viral_load") is not None and row.get("alive_on_art") not in (None, 0)
    ]
    testing_share_mean = float(np.mean(testing_shares)) if testing_shares else 0.0

    simulated_forecast_rows, simulated_hazard_rows = _simulate_holdout_from_hazard_map(
        holdout_rows=holdout_rows[1:],
        hazard_map=adjusted_hazard_map,
        testing_share_mean=testing_share_mean,
        current_state_override=dict(anchored_state_row["state_values"]),
        probability_lower_bound=probability_lower_bound,
        probability_upper_bound=probability_upper_bound,
        eps=eps,
    )

    anchored_forecast_row = {
        "quarter": str(anchored_state_row["quarter"]),
        "diagnosed_plhiv": float(anchored_prediction_row["diagnosed_plhiv"]),
        "alive_on_art": float(anchored_prediction_row["alive_on_art"]),
        "new_diagnosed_cases_period": float(anchored_prediction_row["new_diagnosed_cases_period"]),
        "tested_for_viral_load": float(testing_share_mean * (float(anchored_state_row["state_values"]["A"]) + float(anchored_state_row["state_values"]["V"]))),
        "virally_suppressed": float(anchored_state_row["state_values"]["V"]),
        "state_values": {state_name: float(anchored_state_row["state_values"][state_name]) for state_name in anchored_state_row["state_values"]},
    }
    combined_forecast_rows = [anchored_forecast_row] + simulated_forecast_rows

    primary_metrics = ("diagnosed_plhiv", "alive_on_art", "new_diagnosed_cases_period")
    evaluation_holdout_rows: list[dict[str, Any]] = []
    for target_row, forecast_row in zip(holdout_rows, combined_forecast_rows):
        evaluation_holdout_rows.append(
            {
                "quarter": str(target_row["quarter"]),
                "target": {
                    metric_name: float(target_row[metric_name])
                    for metric_name in primary_metrics
                    if target_row.get(metric_name) is not None
                },
                "prediction": {
                    metric_name: float(forecast_row[metric_name])
                    for metric_name in primary_metrics
                    if forecast_row.get(metric_name) is not None
                },
            }
        )

    metric_scales = _metric_scales(train_rows, eps)
    model_mae = _normalized_mae(combined_forecast_rows, holdout_rows, metric_scales, eps)
    model_smape = _smape(combined_forecast_rows, holdout_rows, eps)
    diagnosis_flow_mae, diagnosis_flow_rows = _diagnosis_flow_mae(
        forecast_rows=combined_forecast_rows,
        diagnosis_flow_targets=dict(actual_payload["diagnosis_flow_targets_by_quarter"]),
    )
    peak_01f_baseline = dict(peak_01f_reference.get("baseline_comparison") or {})
    baseline_comparison = {
        "branch_reference_experiment_id": PEAK01F_EXPERIMENT_ID,
        "branch_reference_label": "PEAK-01F",
        "branch_reference_mean_absolute_error": float(peak_01f_baseline.get("model_mean_absolute_error") or 0.0),
        "branch_reference_diagnosis_flow_mean_absolute_error": float(peak_01f_baseline.get("diagnosis_flow_mean_absolute_error") or 0.0),
        "carry_forward_mean_absolute_error": float(peak_01f_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(peak_01f_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_mean_absolute_error": round(float(model_mae), 6),
        "model_smape": round(float(model_smape), 6),
        "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
        "model_beats_peak_01f_on_mae": bool(float(model_mae) < float(peak_01f_baseline.get("model_mean_absolute_error") or 0.0)),
        "diagnosis_flow_non_regression": bool(
            float(diagnosis_flow_mae)
            <= float(peak_01f_baseline.get("diagnosis_flow_mean_absolute_error") or 0.0)
        ),
    }
    evaluation = {
        "mode": "age_youth_diagnosis_modifier",
        "comparison_reference_run_id": str(peak_01f_reference["reference_run_id"]),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "holdout_rows": evaluation_holdout_rows,
        "diagnosis_flow_evaluation": {
            "mean_absolute_error": round(float(diagnosis_flow_mae), 6),
            "rows": diagnosis_flow_rows,
        },
        "metric_scales": {key: round(float(value), 6) for key, value in metric_scales.items()},
    }
    mechanistic_forecast = {
        "baseline_comparison": baseline_comparison,
        "evaluation": evaluation,
        "forecast_rows": combined_forecast_rows,
        "transition_hazards": simulated_hazard_rows,
    }
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "mechanistic_forecast.json", mechanistic_forecast)
    write_json(
        ctx.experiment_dir / "age_diagnosis_modifier_summary.json",
        {
            "train_quarters": train_quarters,
            "train_youth_share_rows": [
                {"quarter": quarter, "u_to_d_hazard": round(float(hazard), 6), "youth_share": round(float(share), 6)}
                for quarter, hazard, share in train_pairs
            ],
            "modifier_rows": modifier_rows,
            "youth_share_series": youth_quarter_rows,
            "trend_intercept": round(float(trend_intercept), 6),
            "trend_slope": round(float(trend_slope), 6),
            "youth_slope": round(float(youth_slope), 6),
            "youth_loo_r2": round(float(youth_loo_r2), 6),
            "youth_residual_mse": round(float(youth_residual_mse), 6),
            "youth_loo_mse": round(float(youth_loo_mse), 6),
            "residual_std_cap": round(float(residual_std), 6),
        },
    )

    fig, ax = plt.subplots()
    ax.bar(
        ["AGE-01B", "PEAK-01F", "Carry-forward", "Simple compartmental"],
        [
            float(model_mae),
            float(peak_01f_baseline.get("model_mean_absolute_error") or 0.0),
            float(peak_01f_baseline.get("carry_forward_mean_absolute_error") or 0.0),
            float(peak_01f_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        ],
    )
    ax.set_ylabel("Normalized MAE")
    ax.set_title("AGE-01B Forecast vs Baselines")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "forecast_vs_baselines.png")
    plt.close(fig)

    decision_checks = [
        {
            "name": "model_beats_peak_01f_on_mae",
            "passed": bool(baseline_comparison["model_beats_peak_01f_on_mae"]),
            "actual": round(float(model_mae), 6),
            "target": round(float(peak_01f_baseline.get("model_mean_absolute_error") or 0.0), 6),
        },
        {
            "name": "diagnosis_flow_non_regression",
            "passed": bool(baseline_comparison["diagnosis_flow_non_regression"]),
            "actual": round(float(diagnosis_flow_mae), 6),
            "target": round(float(peak_01f_baseline.get("diagnosis_flow_mean_absolute_error") or 0.0), 6),
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "AGE-01B applies a youth-share-based modifier to the PEAK-01F diagnosis hazard while leaving downstream hazards unchanged.",
        "checks": decision_checks,
    }

    numeric_justification = [
        {
            "name": "age_group_boundary_15_24_lower",
            "value": 15,
            "role": "archive_native_youth_band_lower_bound",
            "source_type": "estimated",
            "estimation_data": "archive-native youth phrasing and AGE-01B plan grouping",
            "estimation_method": "use the archive-native youth boundary exactly as reported",
            "uncertainty": "none once the archive-native band is fixed",
            "why_needed": "Defines the youth band being mapped into the U_to_D diagnosis modifier.",
        },
        {
            "name": "age_group_boundary_15_24_upper",
            "value": 24,
            "role": "archive_native_youth_band_upper_bound",
            "source_type": "estimated",
            "estimation_data": "archive-native youth phrasing and AGE-01B plan grouping",
            "estimation_method": "use the archive-native youth boundary exactly as reported",
            "uncertainty": "none once the archive-native band is fixed",
            "why_needed": "Defines the youth band being mapped into the U_to_D diagnosis modifier.",
        },
        {
            "name": "youth_modifier_train_sample_count",
            "value": int(len(train_pairs)),
            "role": "age_modifier_train_support",
            "source_type": "estimated",
            "estimation_data": "quarters with both MECH-01E train U_to_D hazards and recovered youth-share observations",
            "estimation_method": "count aligned train quarters after quarter-level age-signal aggregation",
            "uncertainty": "depends on age extraction coverage",
            "why_needed": "Records the empirical support behind the youth diagnosis modifier fit.",
        },
        {
            "name": "youth_share_train_mean",
            "value": round(float(youth_share_mean), 6),
            "role": "age_modifier_centering_mean",
            "source_type": "estimated",
            "estimation_data": "quarter-level youth share series aligned to MECH-01E train quarters",
            "estimation_method": "empirical mean of the aligned train youth-share observations",
            "uncertainty": "sampling variability over the aligned train quarters",
            "why_needed": "Centers the youth diagnosis modifier on the observed train-age baseline rather than a manual reference level.",
        },
        {
            "name": "youth_share_train_std",
            "value": round(float(youth_share_std), 6),
            "role": "age_modifier_standardization_scale",
            "source_type": "estimated",
            "estimation_data": "quarter-level youth share series aligned to MECH-01E train quarters",
            "estimation_method": "empirical standard deviation of the aligned train youth-share observations",
            "uncertainty": "sampling variability over the aligned train quarters",
            "why_needed": "Standardizes the youth-share predictor without introducing a manual scaling constant.",
        },
        {
            "name": "u_to_d_time_trend_intercept",
            "value": round(float(trend_intercept), 6),
            "role": "baseline_diagnosis_hazard_trend_intercept",
            "source_type": "estimated",
            "estimation_data": "MECH-01E train U_to_D hazards over aligned train quarters",
            "estimation_method": "ordinary least squares intercept from a quarter-index time trend",
            "uncertainty": "captured indirectly by the residual variance and the downstream youth slope gate",
            "why_needed": "Removes the slow diagnosis-hazard trend before fitting the youth-share residual effect.",
        },
        {
            "name": "u_to_d_time_trend_slope",
            "value": round(float(trend_slope), 6),
            "role": "baseline_diagnosis_hazard_trend_slope",
            "source_type": "estimated",
            "estimation_data": "MECH-01E train U_to_D hazards over aligned train quarters",
            "estimation_method": "ordinary least squares slope from a quarter-index time trend",
            "uncertainty": "captured indirectly by the residual variance and the downstream youth slope gate",
            "why_needed": "Separates the slow time trend from the youth-share residual effect before applying the modifier on PEAK-01F.",
        },
        {
            "name": "youth_modifier_slope",
            "value": round(float(youth_slope), 6),
            "role": "u_to_d_youth_residual_effect",
            "source_type": "estimated",
            "estimation_data": "detrended MECH-01E train U_to_D hazards versus standardized youth-share observations",
            "estimation_method": "slope-only least squares fit on centered youth-share residual variation",
            "uncertainty": "summarized by leave-one-quarter-out R^2 and residual MSE",
            "why_needed": "Provides the estimated youth-driven residual correction applied to the PEAK-01F diagnosis hazard.",
        },
        {
            "name": "youth_modifier_loo_r2",
            "value": round(float(youth_loo_r2), 6),
            "role": "u_to_d_youth_gate_strength",
            "source_type": "bayesian_posterior",
            "estimation_data": "leave-one-quarter-out residual predictions from the youth slope-only fit",
            "estimation_method": "convert leave-one-quarter-out residual MSE into a non-negative out-of-sample R^2 gate",
            "uncertainty": "already encoded as out-of-sample performance on the aligned train quarters",
            "why_needed": "Suppresses the youth diagnosis modifier when it lacks out-of-sample skill.",
        },
        {
            "name": "u_to_d_residual_std_cap",
            "value": round(float(residual_std), 6),
            "role": "diagnosis_modifier_cap",
            "source_type": "estimated",
            "estimation_data": "detrended MECH-01E train U_to_D residual distribution",
            "estimation_method": "empirical standard deviation of the residual train diagnosis hazard after removing the slow trend",
            "uncertainty": "sampling variability over aligned train quarters",
            "why_needed": "Caps the youth-driven diagnosis correction using an observed residual scale instead of a manual constant.",
        },
        numerical_guard_entry(
            name="float32_epsilon",
            role="hazard_and_standardization_guard",
            why_needed="Prevents undefined standardization and invalid hazard clipping when age-share variation is numerically zero.",
        ),
    ]

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "comparison_reference_run_id": str(peak_01f_reference["reference_run_id"]),
            "age_group_order": list(AGE_GROUP_ORDER),
            "supported_age_quarter_count": int(len(youth_quarter_rows)),
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "comparison_reference_run_id": str(peak_01f_reference["reference_run_id"]),
            "age_support_quarter_count": int(len(youth_quarter_rows)),
            "train_support_quarter_count": int(len(train_pairs)),
            "holdout_support_quarter_count": int(sum(1 for quarter in holdout_quarters_after_anchor if quarter in youth_share_by_quarter)),
            "recommended_grouping": "15-24_vs_25+",
        },
        decision=decision,
        numeric_justification=numeric_justification,
    )
    return {
        "artifacts": artifacts,
        "decision": decision,
        "baseline_comparison": baseline_comparison,
        "evaluation": evaluation,
        "mechanistic_forecast": mechanistic_forecast,
    }


def run_age_01c(ctx: TransitionResearchContext) -> dict[str, Any]:
    inputs = load_transition_research_inputs(ctx)
    analysis_years = [int(year) for year in inputs.analysis_years]
    historical_metric_rows = [
        dict(row)
        for row in list(read_json(ctx.source_run_dir / "harp_archive" / "historical_metric_rows.json", default=[]) or [])
    ]
    page_catalog_rows = [
        dict(row) for row in list(read_json(ctx.source_run_dir / "harp_archive" / "page_catalog.json", default=[]) or [])
    ]
    youth_quarter_rows = _quarter_youth_share_series(historical_metric_rows, page_catalog_rows, analysis_years)
    youth_share_by_quarter = {
        str(row["quarter"]): float(row["youth_share"])
        for row in youth_quarter_rows
        if row.get("youth_share") is not None
    }

    age_01b_reference = _load_age_experiment_reference(AGE01B_EXPERIMENT_ID)
    actual_payload = _build_observation_payload(ctx)
    actual_rows = [dict(row) for row in list(actual_payload["rows"])]
    holdout_years = _holdout_years_from_reference(age_01b_reference)
    holdout_rows = _holdout_rows(actual_rows, holdout_years)
    if len(holdout_rows) < 2:
        raise ValueError("AGE-01C requires at least two holdout rows from the AGE-01B reference")

    age_01b_mechanistic = dict(age_01b_reference.get("mechanistic_forecast") or {})
    age_01b_forecast_rows = [dict(row) for row in list(age_01b_mechanistic.get("forecast_rows") or [])]
    age_01b_hazard_rows = [dict(row) for row in list(age_01b_mechanistic.get("transition_hazards") or [])]
    if len(age_01b_forecast_rows) < len(holdout_rows):
        raise ValueError("AGE-01C requires AGE-01B forecast rows with anchored holdout state support")
    anchored_forecast_row = dict(age_01b_forecast_rows[0])
    if "state_values" not in anchored_forecast_row:
        raise ValueError("AGE-01C requires AGE-01B anchored forecast rows to include state_values")
    baseline_hazard_map = {
        str(row.get("quarter") or ""): {
            str(transition): float(value)
            for transition, value in dict(row.get("hazards") or {}).items()
        }
        for row in age_01b_hazard_rows
    }

    from .transition_engine import MECH01E_EXPERIMENT_ID, _load_transition_experiment_reference  # local import

    mech_01e_reference = _load_transition_experiment_reference(MECH01E_EXPERIMENT_ID)
    train_hazard_rows = [
        dict(row)
        for row in list((mech_01e_reference.get("transition_hazard_summary") or {}).get("train_rows") or [])
    ]

    eps = float(np.finfo(np.float32).eps)
    probability_lower_bound = eps
    probability_upper_bound = 1.0 - eps

    transition_fit_rows: list[dict[str, Any]] = []
    transition_models: dict[str, dict[str, Any]] = {}
    for transition_name in AGE01C_TRANSITIONS:
        train_pairs = [
            (
                str(row["quarter"]),
                float((row.get("hazards") or {}).get(transition_name) or 0.0),
                float(youth_share_by_quarter[str(row["quarter"])]),
            )
            for row in train_hazard_rows
            if str(row.get("quarter") or "") in youth_share_by_quarter
        ]
        if len(train_pairs) < 3:
            transition_models[transition_name] = {"supported": False, "reason": "insufficient_aligned_train_quarters"}
            transition_fit_rows.append(
                {
                    "transition": transition_name,
                    "supported": False,
                    "aligned_train_quarter_count": int(len(train_pairs)),
                    "reason": "insufficient_aligned_train_quarters",
                }
            )
            continue
        train_quarters = [quarter for quarter, _, _ in train_pairs]
        train_time_index = np.asarray([float(index) for index, _ in enumerate(train_quarters)], dtype=np.float64)
        train_hazards = np.asarray([float(hazard) for _, hazard, _ in train_pairs], dtype=np.float64)
        train_youth_share = np.asarray([float(share) for _, _, share in train_pairs], dtype=np.float64)
        trend_intercept, trend_slope = _least_squares_intercept_and_slope(train_time_index, train_hazards)
        train_trend = trend_intercept + (trend_slope * train_time_index)
        train_residual = train_hazards - train_trend
        youth_share_mean = float(np.mean(train_youth_share))
        youth_share_std = float(np.std(train_youth_share))
        if youth_share_std <= eps:
            standardized_train_share = np.zeros_like(train_youth_share, dtype=np.float64)
        else:
            standardized_train_share = (train_youth_share - youth_share_mean) / youth_share_std
        denominator = float(np.dot(standardized_train_share, standardized_train_share))
        numerator = float(np.dot(standardized_train_share, train_residual))
        youth_slope = (numerator / denominator) if denominator > eps else 0.0
        youth_fitted_residual = youth_slope * standardized_train_share
        youth_residual_mse = float(np.mean(np.square(train_residual - youth_fitted_residual))) if train_residual.size else 0.0
        youth_loo_r2, youth_loo_mse = _leave_one_out_slope_only_r2(
            standardized_train_share.astype(np.float32),
            train_residual.astype(np.float32),
            eps,
        )
        residual_std = float(np.std(train_residual)) if train_residual.size else 0.0
        transition_models[transition_name] = {
            "supported": True,
            "train_quarters": train_quarters,
            "trend_intercept": float(trend_intercept),
            "trend_slope": float(trend_slope),
            "youth_share_mean": float(youth_share_mean),
            "youth_share_std": float(youth_share_std),
            "youth_slope": float(youth_slope),
            "youth_loo_r2": float(youth_loo_r2),
            "youth_residual_mse": float(youth_residual_mse),
            "youth_loo_mse": float(youth_loo_mse),
            "residual_std_cap": float(residual_std),
            "train_pairs": train_pairs,
        }
        transition_fit_rows.append(
            {
                "transition": transition_name,
                "supported": True,
                "aligned_train_quarter_count": int(len(train_pairs)),
                "trend_intercept": round(float(trend_intercept), 6),
                "trend_slope": round(float(trend_slope), 6),
                "youth_share_mean": round(float(youth_share_mean), 6),
                "youth_share_std": round(float(youth_share_std), 6),
                "youth_slope": round(float(youth_slope), 6),
                "youth_loo_r2": round(float(youth_loo_r2), 6),
                "youth_residual_mse": round(float(youth_residual_mse), 6),
                "youth_loo_mse": round(float(youth_loo_mse), 6),
                "residual_std_cap": round(float(residual_std), 6),
            }
        )

    holdout_quarters_after_anchor = [str(row["quarter"]) for row in holdout_rows[1:]]
    adjusted_hazard_map: dict[str, dict[str, float]] = {}
    modifier_rows: list[dict[str, Any]] = []
    max_abs_gated_correction = 0.0
    for quarter in holdout_quarters_after_anchor:
        base_hazards = {transition: float(value) for transition, value in dict(baseline_hazard_map.get(quarter) or {}).items()}
        adjusted_hazards = {
            transition: float(base_hazards.get(transition) or 0.0)
            for transition in ("U_to_D", "D_to_A", "A_to_V", "A_to_L", "L_to_A")
        }
        youth_share = float(youth_share_by_quarter.get(quarter) or 0.0)
        for transition_name in AGE01C_TRANSITIONS:
            model = dict(transition_models.get(transition_name) or {})
            if not bool(model.get("supported")):
                modifier_rows.append(
                    {
                        "quarter": quarter,
                        "transition": transition_name,
                        "support_available": False,
                        "active": False,
                        "reason": str(model.get("reason") or "unsupported"),
                        "base_hazard": round(float(base_hazards.get(transition_name) or 0.0), 6),
                        "adjusted_hazard": round(float(adjusted_hazards.get(transition_name) or 0.0), 6),
                    }
                )
                continue
            youth_share_mean = float(model["youth_share_mean"])
            youth_share_std = float(model["youth_share_std"])
            youth_z = (youth_share - youth_share_mean) / max(youth_share_std, eps) if youth_share_std > eps else 0.0
            raw_correction = float(float(model["youth_slope"]) * youth_z)
            gated_correction = float(
                np.clip(
                    float(model["youth_loo_r2"]) * raw_correction,
                    -float(model["residual_std_cap"]),
                    float(model["residual_std_cap"]),
                )
            )
            adjusted_hazards[transition_name] = float(
                np.clip(
                    float(base_hazards.get(transition_name) or 0.0) + gated_correction,
                    probability_lower_bound,
                    probability_upper_bound,
                )
            )
            max_abs_gated_correction = max(max_abs_gated_correction, abs(gated_correction))
            modifier_rows.append(
                {
                    "quarter": quarter,
                    "transition": transition_name,
                    "support_available": bool(quarter in youth_share_by_quarter),
                    "active": bool(abs(gated_correction) > 0.0),
                    "youth_share": round(float(youth_share), 6),
                    "youth_share_z": round(float(youth_z), 6),
                    "base_hazard": round(float(base_hazards.get(transition_name) or 0.0), 6),
                    "raw_correction": round(float(raw_correction), 6),
                    "gated_correction": round(float(gated_correction), 6),
                    "adjusted_hazard": round(float(adjusted_hazards[transition_name]), 6),
                }
            )
        adjusted_hazard_map[quarter] = adjusted_hazards

    train_rows = [
        dict(row)
        for row in actual_rows
        if str(row.get("quarter") or "") not in {str(item["quarter"]) for item in holdout_rows}
    ]
    testing_shares = [
        float(row["tested_for_viral_load"]) / max(float(row["alive_on_art"]), eps)
        for row in train_rows
        if row.get("tested_for_viral_load") is not None and row.get("alive_on_art") not in (None, 0)
    ]
    testing_share_mean = float(np.mean(testing_shares)) if testing_shares else 0.0
    simulated_forecast_rows, simulated_hazard_rows = _simulate_holdout_from_hazard_map(
        holdout_rows=holdout_rows[1:],
        hazard_map=adjusted_hazard_map,
        testing_share_mean=testing_share_mean,
        current_state_override=dict(anchored_forecast_row["state_values"]),
        probability_lower_bound=probability_lower_bound,
        probability_upper_bound=probability_upper_bound,
        eps=eps,
    )
    combined_forecast_rows = [anchored_forecast_row] + simulated_forecast_rows

    primary_metrics = ("diagnosed_plhiv", "alive_on_art", "new_diagnosed_cases_period")
    evaluation_holdout_rows: list[dict[str, Any]] = []
    for target_row, forecast_row in zip(holdout_rows, combined_forecast_rows):
        evaluation_holdout_rows.append(
            {
                "quarter": str(target_row["quarter"]),
                "target": {
                    metric_name: float(target_row[metric_name])
                    for metric_name in primary_metrics
                    if target_row.get(metric_name) is not None
                },
                "prediction": {
                    metric_name: float(forecast_row[metric_name])
                    for metric_name in primary_metrics
                    if forecast_row.get(metric_name) is not None
                },
            }
        )

    metric_scales = _metric_scales(train_rows, eps)
    model_mae = _normalized_mae(combined_forecast_rows, holdout_rows, metric_scales, eps)
    model_smape = _smape(combined_forecast_rows, holdout_rows, eps)
    diagnosis_flow_mae, diagnosis_flow_rows = _diagnosis_flow_mae(
        forecast_rows=combined_forecast_rows,
        diagnosis_flow_targets=dict(actual_payload["diagnosis_flow_targets_by_quarter"]),
    )

    age_01b_baseline = dict(age_01b_reference.get("baseline_comparison") or {})
    age_01b_evaluation = dict(age_01b_reference.get("evaluation") or {})
    age_01b_holdout_rows = [dict(row) for row in list(age_01b_evaluation.get("holdout_rows") or [])]
    target_peak_quarter = max(
        [str(row["quarter"]) for row in holdout_rows],
        key=lambda quarter: float(next(row for row in holdout_rows if str(row["quarter"]) == quarter)["alive_on_art"]),
    )
    model_peak_alive_on_art_error = abs(
        float(next(row for row in combined_forecast_rows if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
        - float(next(row for row in holdout_rows if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
    )
    age_01b_peak_alive_on_art_error = abs(
        float(next(row for row in age_01b_holdout_rows if str(row["quarter"]) == target_peak_quarter)["prediction"]["alive_on_art"])
        - float(next(row for row in holdout_rows if str(row["quarter"]) == target_peak_quarter)["alive_on_art"])
    )

    active_transition_count = sum(1 for row in modifier_rows if bool(row.get("active")))
    baseline_comparison = {
        "branch_reference_experiment_id": AGE01B_EXPERIMENT_ID,
        "branch_reference_label": "AGE-01B",
        "branch_reference_mean_absolute_error": float(age_01b_baseline.get("model_mean_absolute_error") or 0.0),
        "branch_reference_diagnosis_flow_mean_absolute_error": float(age_01b_baseline.get("diagnosis_flow_mean_absolute_error") or 0.0),
        "branch_reference_peak_alive_on_art_absolute_error": round(float(age_01b_peak_alive_on_art_error), 6),
        "carry_forward_mean_absolute_error": float(age_01b_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(age_01b_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_mean_absolute_error": round(float(model_mae), 6),
        "model_smape": round(float(model_smape), 6),
        "model_peak_alive_on_art_absolute_error": round(float(model_peak_alive_on_art_error), 6),
        "peak_target_quarter": target_peak_quarter,
        "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
        "model_beats_age_01b_on_mae": bool(float(model_mae) < float(age_01b_baseline.get("model_mean_absolute_error") or 0.0)),
        "model_beats_age_01b_on_peak_error": bool(float(model_peak_alive_on_art_error) < float(age_01b_peak_alive_on_art_error)),
        "diagnosis_flow_non_regression": bool(
            float(diagnosis_flow_mae)
            <= float(age_01b_baseline.get("diagnosis_flow_mean_absolute_error") or 0.0)
        ),
        "active_transition_count": int(active_transition_count),
        "max_absolute_gated_correction": round(float(max_abs_gated_correction), 6),
    }
    evaluation = {
        "mode": "age_youth_downstream_modifier",
        "comparison_reference_run_id": str(age_01b_reference["reference_run_id"]),
        "anchored_holdout_quarter": str(anchored_forecast_row["quarter"]),
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "holdout_rows": evaluation_holdout_rows,
        "diagnosis_flow_evaluation": {
            "mean_absolute_error": round(float(diagnosis_flow_mae), 6),
            "rows": diagnosis_flow_rows,
        },
        "metric_scales": {key: round(float(value), 6) for key, value in metric_scales.items()},
    }
    mechanistic_forecast = {
        "baseline_comparison": baseline_comparison,
        "evaluation": evaluation,
        "forecast_rows": combined_forecast_rows,
        "transition_hazards": simulated_hazard_rows,
    }
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "mechanistic_forecast.json", mechanistic_forecast)
    write_json(
        ctx.experiment_dir / "age_downstream_modifier_summary.json",
        {
            "transition_fit_rows": transition_fit_rows,
            "modifier_rows": modifier_rows,
            "youth_share_series": youth_quarter_rows,
            "target_peak_quarter": target_peak_quarter,
            "branch_reference_peak_alive_on_art_absolute_error": round(float(age_01b_peak_alive_on_art_error), 6),
            "model_peak_alive_on_art_absolute_error": round(float(model_peak_alive_on_art_error), 6),
        },
    )

    heatmap_quarters = holdout_quarters_after_anchor
    heatmap_matrix = np.asarray(
        [
            [
                float(
                    next(
                        (
                            row.get("gated_correction") or 0.0
                            for row in modifier_rows
                            if str(row.get("quarter") or "") == quarter and str(row.get("transition") or "") == transition_name
                        ),
                        0.0,
                    )
                )
                for quarter in heatmap_quarters
            ]
            for transition_name in AGE01C_TRANSITIONS
        ],
        dtype=np.float32,
    )
    fig, ax = plt.subplots()
    image = ax.imshow(heatmap_matrix, aspect="auto")
    ax.set_title("AGE-01C Downstream Age Effects")
    ax.set_xticks(np.arange(len(heatmap_quarters)), labels=heatmap_quarters)
    ax.set_yticks(np.arange(len(AGE01C_TRANSITIONS)), labels=list(AGE01C_TRANSITIONS))
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "age_transition_effect_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(
        ["AGE-01C", "AGE-01B", "Carry-forward", "Simple compartmental"],
        [
            float(model_mae),
            float(age_01b_baseline.get("model_mean_absolute_error") or 0.0),
            float(age_01b_baseline.get("carry_forward_mean_absolute_error") or 0.0),
            float(age_01b_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        ],
    )
    ax.set_ylabel("Normalized MAE")
    ax.set_title("AGE-01C Forecast vs AGE-01B")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "forecast_vs_age01b.png")
    plt.close(fig)

    decision_checks = [
        {
            "name": "age_modifier_active",
            "passed": bool(active_transition_count > 0 and max_abs_gated_correction > 0.0),
            "actual": {
                "active_transition_count": int(active_transition_count),
                "max_absolute_gated_correction": round(float(max_abs_gated_correction), 6),
            },
            "target": {"active_transition_count": ">=1", "max_absolute_gated_correction": ">0"},
        },
        {
            "name": "model_beats_age_01b_on_mae_or_peak_error",
            "passed": bool(
                baseline_comparison["model_beats_age_01b_on_mae"]
                or baseline_comparison["model_beats_age_01b_on_peak_error"]
            ),
            "actual": {
                "model_mean_absolute_error": round(float(model_mae), 6),
                "model_peak_alive_on_art_absolute_error": round(float(model_peak_alive_on_art_error), 6),
            },
            "target": {
                "branch_reference_mean_absolute_error": round(float(age_01b_baseline.get("model_mean_absolute_error") or 0.0), 6),
                "branch_reference_peak_alive_on_art_absolute_error": round(float(age_01b_peak_alive_on_art_error), 6),
            },
        },
        {
            "name": "diagnosis_flow_non_regression",
            "passed": bool(baseline_comparison["diagnosis_flow_non_regression"]),
            "actual": round(float(diagnosis_flow_mae), 6),
            "target": round(float(age_01b_baseline.get("diagnosis_flow_mean_absolute_error") or 0.0), 6),
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "AGE-01C applies youth-share-based downstream age modulation only on D_to_A and A_to_V on top of the kept AGE-01B branch.",
        "checks": decision_checks,
    }

    numeric_justification = [
        {
            "name": "age_group_boundary_15_24_lower",
            "value": 15,
            "role": "archive_native_youth_band_lower_bound",
            "source_type": "estimated",
            "estimation_data": "archive-native youth phrasing and AGE-01C plan grouping",
            "estimation_method": "use the archive-native youth boundary exactly as reported",
            "uncertainty": "none once the archive-native band is fixed",
            "why_needed": "Defines the youth band being mapped into downstream age modifiers.",
        },
        {
            "name": "age_group_boundary_15_24_upper",
            "value": 24,
            "role": "archive_native_youth_band_upper_bound",
            "source_type": "estimated",
            "estimation_data": "archive-native youth phrasing and AGE-01C plan grouping",
            "estimation_method": "use the archive-native youth boundary exactly as reported",
            "uncertainty": "none once the archive-native band is fixed",
            "why_needed": "Defines the youth band being mapped into downstream age modifiers.",
        },
    ]
    for transition_name in AGE01C_TRANSITIONS:
        model = dict(transition_models.get(transition_name) or {})
        if not bool(model.get("supported")):
            continue
        numeric_justification.extend(
            [
                {
                    "name": f"{transition_name.lower()}_age_modifier_train_sample_count",
                    "value": int(len(list(model.get('train_pairs') or []))),
                    "role": f"{transition_name}_age_modifier_train_support",
                    "source_type": "estimated",
                    "estimation_data": f"quarters with MECH-01E train {transition_name} hazards and quarter-level youth-share support",
                    "estimation_method": "count aligned train quarters after canonical youth-share aggregation",
                    "uncertainty": "depends on age extraction coverage",
                    "why_needed": f"Records empirical support behind the {transition_name} age modifier fit.",
                },
                {
                    "name": f"{transition_name.lower()}_youth_share_train_mean",
                    "value": round(float(model["youth_share_mean"]), 6),
                    "role": f"{transition_name}_age_modifier_centering_mean",
                    "source_type": "estimated",
                    "estimation_data": f"quarter-level youth-share series aligned to MECH-01E train {transition_name} hazards",
                    "estimation_method": "empirical mean of aligned train youth-share observations",
                    "uncertainty": "sampling variability over aligned train quarters",
                    "why_needed": f"Centers the {transition_name} age modifier on the observed train-age baseline.",
                },
                {
                    "name": f"{transition_name.lower()}_youth_share_train_std",
                    "value": round(float(model["youth_share_std"]), 6),
                    "role": f"{transition_name}_age_modifier_standardization_scale",
                    "source_type": "estimated",
                    "estimation_data": f"quarter-level youth-share series aligned to MECH-01E train {transition_name} hazards",
                    "estimation_method": "empirical standard deviation of aligned train youth-share observations",
                    "uncertainty": "sampling variability over aligned train quarters",
                    "why_needed": f"Standardizes the {transition_name} age predictor without manual scaling constants.",
                },
                {
                    "name": f"{transition_name.lower()}_time_trend_intercept",
                    "value": round(float(model["trend_intercept"]), 6),
                    "role": f"{transition_name}_hazard_trend_intercept",
                    "source_type": "estimated",
                    "estimation_data": f"MECH-01E train {transition_name} hazards over aligned train quarters",
                    "estimation_method": "ordinary least squares intercept from quarter-index time trend",
                    "uncertainty": "captured indirectly by residual variance and downstream age gate",
                    "why_needed": f"Removes slow trend before fitting the {transition_name} age residual effect.",
                },
                {
                    "name": f"{transition_name.lower()}_time_trend_slope",
                    "value": round(float(model["trend_slope"]), 6),
                    "role": f"{transition_name}_hazard_trend_slope",
                    "source_type": "estimated",
                    "estimation_data": f"MECH-01E train {transition_name} hazards over aligned train quarters",
                    "estimation_method": "ordinary least squares slope from quarter-index time trend",
                    "uncertainty": "captured indirectly by residual variance and downstream age gate",
                    "why_needed": f"Separates slow trend from the youth-share residual effect for {transition_name}.",
                },
                {
                    "name": f"{transition_name.lower()}_age_modifier_slope",
                    "value": round(float(model["youth_slope"]), 6),
                    "role": f"{transition_name}_youth_residual_effect",
                    "source_type": "estimated",
                    "estimation_data": f"detrended MECH-01E train {transition_name} hazards versus standardized youth-share observations",
                    "estimation_method": "slope-only least squares fit on centered youth-share residual variation",
                    "uncertainty": "summarized by leave-one-quarter-out R^2 and residual MSE",
                    "why_needed": f"Provides the youth-driven downstream residual correction applied to {transition_name}.",
                },
                {
                    "name": f"{transition_name.lower()}_age_modifier_loo_r2",
                    "value": round(float(model["youth_loo_r2"]), 6),
                    "role": f"{transition_name}_age_gate_strength",
                    "source_type": "bayesian_posterior",
                    "estimation_data": f"leave-one-quarter-out residual predictions from the {transition_name} youth slope-only fit",
                    "estimation_method": "convert leave-one-quarter-out residual MSE into a non-negative out-of-sample R^2 gate",
                    "uncertainty": "already encoded as out-of-sample performance on aligned train quarters",
                    "why_needed": f"Suppresses the {transition_name} age modifier when it lacks out-of-sample skill.",
                },
                {
                    "name": f"{transition_name.lower()}_residual_std_cap",
                    "value": round(float(model["residual_std_cap"]), 6),
                    "role": f"{transition_name}_age_modifier_cap",
                    "source_type": "estimated",
                    "estimation_data": f"detrended MECH-01E train {transition_name} residual distribution",
                    "estimation_method": "empirical standard deviation of the residual train hazard after removing the slow trend",
                    "uncertainty": "sampling variability over aligned train quarters",
                    "why_needed": f"Caps the youth-driven {transition_name} correction using observed residual scale rather than a manual constant.",
                },
            ]
        )
    numeric_justification.append(
        numerical_guard_entry(
            name="float32_epsilon",
            role="hazard_and_standardization_guard",
            why_needed="Prevents undefined standardization and invalid hazard clipping in downstream age modulation when age-share variation is numerically zero.",
        )
    )

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "comparison_reference_run_id": str(age_01b_reference["reference_run_id"]),
            "age_group_order": list(AGE_GROUP_ORDER),
            "active_transition_names": list(AGE01C_TRANSITIONS),
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "comparison_reference_run_id": str(age_01b_reference["reference_run_id"]),
            "age_support_quarter_count": int(len(youth_quarter_rows)),
            "train_support_quarter_count_by_transition": {
                transition_name: int(len(list(dict(transition_models.get(transition_name) or {}).get("train_pairs") or [])))
                for transition_name in AGE01C_TRANSITIONS
            },
            "holdout_support_quarter_count": int(sum(1 for quarter in holdout_quarters_after_anchor if quarter in youth_share_by_quarter)),
            "recommended_grouping": "15-24_vs_25+",
            "active_transition_count": int(active_transition_count),
        },
        decision=decision,
        numeric_justification=numeric_justification,
    )
    return {
        "artifacts": artifacts,
        "decision": decision,
        "baseline_comparison": baseline_comparison,
        "evaluation": evaluation,
        "mechanistic_forecast": mechanistic_forecast,
    }
