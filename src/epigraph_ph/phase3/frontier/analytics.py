from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.runtime import write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts

from .sources import TransitionResearchInputs, load_transition_research_inputs


def _year_to_month_indices(month_axis: list[str], analysis_years: list[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {year: [] for year in analysis_years}
    for month_idx, month in enumerate(month_axis):
        year = int(str(month).split("-", maxsplit=1)[0])
        if year in grouped:
            grouped[year].append(month_idx)
    return grouped


def _rank_fraction(values_by_id: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values_by_id.items(), key=lambda item: (float(item[1]), item[0]))
    denominator = len(ordered)
    return {
        factor_id: float(rank_position + 1) / float(denominator)
        for rank_position, (factor_id, _) in enumerate(ordered)
    }


def _geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    if np.any(array <= 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _factor_static_metrics(inputs: TransitionResearchInputs) -> dict[str, dict[str, object]]:
    survival = {
        factor_id: float((inputs.retained_factor_lookup.get(factor_id) or {}).get("survival_score") or 0.0)
        for factor_id in inputs.retained_factor_lookup
    }
    predictive = {
        factor_id: float((inputs.retained_factor_lookup.get(factor_id) or {}).get("predictive_gain") or 0.0)
        for factor_id in inputs.retained_factor_lookup
    }
    stability = {
        factor_id: float((inputs.retained_factor_lookup.get(factor_id) or {}).get("stability_score") or 0.0)
        for factor_id in inputs.retained_factor_lookup
    }
    support = {
        factor_id: float(inputs.factor_support_counts.get(factor_id, 0))
        for factor_id in inputs.retained_factor_lookup
    }
    phase3_target = {
        factor_id: float((inputs.retained_factor_lookup.get(factor_id) or {}).get("phase3_target_score") or 0.0)
        for factor_id in inputs.retained_factor_lookup
    }
    survival_rank = _rank_fraction(survival)
    predictive_rank = _rank_fraction(predictive)
    stability_rank = _rank_fraction(stability)
    support_rank = _rank_fraction(support)
    phase3_target_rank = _rank_fraction(phase3_target)
    metrics: dict[str, dict[str, object]] = {}
    for factor_id, row in inputs.retained_factor_lookup.items():
        metrics[factor_id] = {
            "factor_id": factor_id,
            "factor_name": str(row.get("factor_name") or factor_id),
            "best_target": str(row.get("best_target") or ""),
            "transition_hooks": list(row.get("transition_hooks", [])),
            "network_feature_family": str(row.get("network_feature_family") or ""),
            "structural_strength": _geometric_mean(
                [
                    float(survival_rank[factor_id]),
                    float(predictive_rank[factor_id]),
                    float(stability_rank[factor_id]),
                ]
            ),
            "support_rank": float(support_rank[factor_id]),
            "phase3_target_rank": float(phase3_target_rank[factor_id]),
            "support_count": int(inputs.factor_support_counts.get(factor_id, 0)),
        }
    return metrics


def _heatmap(*, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, output_path) -> None:
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _trajectory_plot(*, series: dict[str, list[float]], x_labels: list[str], title: str, ylabel: str, output_path) -> None:
    fig, ax = plt.subplots()
    x_values = np.arange(len(x_labels))
    for label, values in series.items():
        ax.plot(x_values, values, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values, labels=x_labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compute_an01a_outputs(inputs: TransitionResearchInputs) -> dict[str, object]:
    factor_metrics = _factor_static_metrics(inputs)
    year_month_indices = _year_to_month_indices(inputs.month_axis, inputs.analysis_years)
    max_month_count = max(len(indices) for indices in year_month_indices.values()) if year_month_indices else 0
    eps = np.finfo(np.float32).eps
    rows: list[dict[str, object]] = []
    ranked_yearly_tables: dict[str, list[dict[str, object]]] = {}
    years = [str(year) for year in inputs.analysis_years]
    factor_ids = sorted(inputs.retained_factor_lookup)
    score_matrix = np.zeros((len(factor_ids), len(inputs.analysis_years)), dtype=np.float32)
    trajectories: dict[str, list[float]] = {}
    for factor_position, factor_id in enumerate(factor_ids):
        factor_idx = inputs.factor_index[factor_id]
        factor_values = np.asarray(inputs.national_tensor[0, :, factor_idx], dtype=np.float32)
        factor_peak = float(np.max(np.abs(factor_values))) if factor_values.size else 0.0
        series: list[float] = []
        for year_position, year in enumerate(inputs.analysis_years):
            month_indices = year_month_indices[year]
            activation_mean_abs = float(np.mean(np.abs(factor_values[month_indices]))) if month_indices else 0.0
            activation_relative = activation_mean_abs / factor_peak if factor_peak > eps else 0.0
            year_coverage_fraction = float(len(month_indices)) / float(max_month_count) if max_month_count else 0.0
            metrics = factor_metrics[factor_id]
            score = _geometric_mean(
                [
                    float(max(float(metrics["structural_strength"]), 0.0)),
                    float(max(float(metrics["support_rank"]), 0.0)),
                    float(max(activation_relative, 0.0)),
                ]
            )
            confidence = _geometric_mean(
                [
                    float(max(float(metrics["support_rank"]), 0.0)),
                    float(max(float(metrics["phase3_target_rank"]), 0.0)),
                    float(max(year_coverage_fraction, 0.0)),
                ]
            )
            rows.append(
                {
                    "year": year,
                    "factor_id": factor_id,
                    "factor_name": metrics["factor_name"],
                    "best_target": metrics["best_target"],
                    "transition_hooks": metrics["transition_hooks"],
                    "activation_mean_abs": round(activation_mean_abs, 6),
                    "activation_relative": round(activation_relative, 6),
                    "structural_strength": round(float(metrics["structural_strength"]), 6),
                    "support_rank": round(float(metrics["support_rank"]), 6),
                    "year_coverage_fraction": round(year_coverage_fraction, 6),
                    "score": round(score, 6),
                    "confidence": round(confidence, 6),
                }
            )
            score_matrix[factor_position, year_position] = float(score)
            series.append(float(score))
        trajectories[factor_id] = series
    for year in inputs.analysis_years:
        year_rows = [dict(row) for row in rows if int(row["year"]) == year]
        year_rows.sort(key=lambda row: (-float(row["score"]), str(row["factor_id"])))
        for rank_position, row in enumerate(year_rows):
            row["rank_within_year"] = rank_position + 1
        ranked_yearly_tables[str(year)] = year_rows
    return {
        "rows": rows,
        "ranked_yearly_tables": ranked_yearly_tables,
        "factor_ids": factor_ids,
        "years": years,
        "score_matrix": score_matrix,
        "trajectories": trajectories,
        "coverage_summary": {
            "analysis_years": inputs.analysis_years,
            "factor_count": len(factor_ids),
            "month_axis_count": len(inputs.month_axis),
            "phase15_source_run_id": inputs.ctx.source_run_id,
            "year_month_counts": {str(year): len(year_month_indices[year]) for year in inputs.analysis_years},
        },
        "numeric_justification": [
            {
                "name": "analysis_year_floor",
                "value": int(min(inputs.analysis_years)),
                "role": "year_window_start",
                "source_type": "estimated",
                "estimation_data": "phase15 multiscale month axis",
                "estimation_method": "minimum observed year in the retained month axis",
                "uncertainty": "none",
                "why_needed": "Defines the first analysis year without introducing a manual boundary.",
            },
            {
                "name": "analysis_year_ceiling",
                "value": int(max(inputs.analysis_years)),
                "role": "year_window_end",
                "source_type": "estimated",
                "estimation_data": "phase15 archive_observation_target_support.harp_program_support.reference_points",
                "estimation_method": "maximum year among supported HARP reference points",
                "uncertainty": "none",
                "why_needed": "Excludes unsupported projected years from the yearly score path.",
            },
            {
                "name": "year_window_month_capacity",
                "value": int(max_month_count),
                "role": "coverage_normalizer",
                "source_type": "estimated",
                "estimation_data": "phase15 multiscale month axis grouped by year",
                "estimation_method": "maximum count of retained month stamps inside a supported analysis year",
                "uncertainty": "source-axis dependent",
                "why_needed": "Converts per-year retained month counts into a coverage fraction without a manual divisor.",
            },
            {
                "name": "structural_strength_component_count",
                "value": 3,
                "role": "rank_aggregation_component_count",
                "source_type": "physical_constraint",
                "estimation_data": "retained factor metadata",
                "estimation_method": "count of structural rank components: survival, predictive gain, stability",
                "uncertainty": "none",
                "why_needed": "The geometric mean uses the exact number of structural evidence channels present in the retained factor rows.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="division_guard",
                why_needed="Prevents undefined normalization when a factor tensor is identically zero.",
            ),
        ],
    }


def run_an01a(ctx: TransitionResearchContext) -> dict[str, object]:
    inputs = load_transition_research_inputs(ctx)
    payload = compute_an01a_outputs(inputs)
    write_json(ctx.experiment_dir / "yearly_factor_scores_national.json", payload["rows"])
    write_json(ctx.experiment_dir / "ranked_yearly_tables.json", payload["ranked_yearly_tables"])
    _heatmap(
        matrix=payload["score_matrix"],
        row_labels=payload["factor_ids"],
        col_labels=payload["years"],
        title="AN-01A National Yearly Factor Scores",
        output_path=ctx.experiment_dir / "national_factor_heatmap.png",
    )
    _trajectory_plot(
        series=payload["trajectories"],
        x_labels=payload["years"],
        title="AN-01A National Yearly Factor Trajectories",
        ylabel="Score",
        output_path=ctx.experiment_dir / "national_factor_trajectories.png",
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
        },
        coverage_summary=payload["coverage_summary"],
        decision={"completed": True, "keep": True, "reason": "National yearly factor evolution artifacts generated successfully."},
        numeric_justification=payload["numeric_justification"],
    )
    return {**payload, "artifacts": artifacts}


def compute_an01b_outputs(inputs: TransitionResearchInputs) -> dict[str, object]:
    factor_metrics = _factor_static_metrics(inputs)
    year_month_indices = _year_to_month_indices(inputs.month_axis, inputs.analysis_years)
    max_month_count = max(len(indices) for indices in year_month_indices.values()) if year_month_indices else 0
    eps = np.finfo(np.float32).eps
    factor_ids = sorted(inputs.retained_factor_lookup)
    rows: list[dict[str, object]] = []
    heatmap_labels: list[str] = []
    heatmap_rows: list[np.ndarray] = []
    trajectories: dict[str, list[float]] = {factor_id: [] for factor_id in factor_ids}
    for region_idx, region in enumerate(inputs.region_axis):
        for factor_id in factor_ids:
            factor_idx = inputs.factor_index[factor_id]
            factor_values = np.asarray(inputs.region_tensor[region_idx, :, factor_idx], dtype=np.float32)
            factor_peak = float(np.max(np.abs(inputs.region_tensor[:, :, factor_idx]))) if inputs.region_tensor.size else 0.0
            label = f"{region}|{factor_id}"
            heatmap_labels.append(label)
            yearly_scores: list[float] = []
            for year in inputs.analysis_years:
                month_indices = year_month_indices[year]
                activation_mean_abs = float(np.mean(np.abs(factor_values[month_indices]))) if month_indices else 0.0
                activation_relative = activation_mean_abs / factor_peak if factor_peak > eps else 0.0
                year_coverage_fraction = float(len(month_indices)) / float(max_month_count) if max_month_count else 0.0
                metrics = factor_metrics[factor_id]
                score = _geometric_mean(
                    [
                        float(max(float(metrics["structural_strength"]), 0.0)),
                        float(max(float(metrics["support_rank"]), 0.0)),
                        float(max(activation_relative, 0.0)),
                    ]
                )
                confidence = _geometric_mean(
                    [
                        float(max(float(metrics["support_rank"]), 0.0)),
                        float(max(float(metrics["phase3_target_rank"]), 0.0)),
                        float(max(year_coverage_fraction, 0.0)),
                    ]
                )
                rows.append(
                    {
                        "region": region,
                        "year": year,
                        "factor_id": factor_id,
                        "factor_name": metrics["factor_name"],
                        "score": round(score, 6),
                        "confidence": round(confidence, 6),
                        "activation_relative": round(activation_relative, 6),
                    }
                )
                yearly_scores.append(float(score))
            heatmap_rows.append(np.asarray(yearly_scores, dtype=np.float32))
    for factor_id in factor_ids:
        for year in inputs.analysis_years:
            year_rows = [row for row in rows if row["factor_id"] == factor_id and int(row["year"]) == year]
            mean_score = float(np.mean([float(row["score"]) for row in year_rows])) if year_rows else 0.0
            trajectories[factor_id].append(mean_score)
    score_matrix = np.vstack(heatmap_rows) if heatmap_rows else np.zeros((0, len(inputs.analysis_years)), dtype=np.float32)
    return {
        "rows": rows,
        "factor_ids": factor_ids,
        "years": [str(year) for year in inputs.analysis_years],
        "heatmap_labels": heatmap_labels,
        "score_matrix": score_matrix,
        "trajectories": trajectories,
        "coverage_summary": {
            "analysis_years": inputs.analysis_years,
            "factor_count": len(factor_ids),
            "region_count": len(inputs.region_axis),
            "year_region_support_count": len(rows),
            "phase15_source_run_id": inputs.ctx.source_run_id,
        },
        "numeric_justification": [
            {
                "name": "region_count",
                "value": len(inputs.region_axis),
                "role": "regional_grid_cardinality",
                "source_type": "estimated",
                "estimation_data": "phase15 multiscale factor axes region axis",
                "estimation_method": "count of region labels in multiscale_factor_axes.region",
                "uncertainty": "none",
                "why_needed": "Defines the regional score grid directly from the source artifact.",
            },
            {
                "name": "year_window_month_capacity",
                "value": int(max_month_count),
                "role": "coverage_normalizer",
                "source_type": "estimated",
                "estimation_data": "phase15 multiscale month axis grouped by year",
                "estimation_method": "maximum retained month count across supported analysis years",
                "uncertainty": "source-axis dependent",
                "why_needed": "Normalizes year-level regional coverage without a manual divisor.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="division_guard",
                why_needed="Prevents undefined normalization when a regional factor surface is identically zero.",
            ),
        ],
    }


def run_an01b(ctx: TransitionResearchContext) -> dict[str, object]:
    inputs = load_transition_research_inputs(ctx)
    payload = compute_an01b_outputs(inputs)
    write_json(ctx.experiment_dir / "yearly_factor_scores_region.json", payload["rows"])
    _heatmap(
        matrix=payload["score_matrix"],
        row_labels=payload["heatmap_labels"],
        col_labels=payload["years"],
        title="AN-01B Regional Yearly Factor Scores",
        output_path=ctx.experiment_dir / "regional_factor_heatmap.png",
    )
    _trajectory_plot(
        series=payload["trajectories"],
        x_labels=payload["years"],
        title="AN-01B Mean Regional Factor Trajectories",
        ylabel="Mean regional score",
        output_path=ctx.experiment_dir / "regional_factor_trajectories.png",
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
        },
        coverage_summary=payload["coverage_summary"],
        decision={"completed": True, "keep": True, "reason": "Regional yearly factor evolution artifacts generated successfully."},
        numeric_justification=payload["numeric_justification"],
    )
    return {**payload, "artifacts": artifacts}


def run_an01c(ctx: TransitionResearchContext) -> dict[str, object]:
    inputs = load_transition_research_inputs(ctx)
    national = compute_an01a_outputs(inputs)
    regional = compute_an01b_outputs(inputs)
    factor_ids = national["factor_ids"]
    first_year = int(min(inputs.analysis_years))
    last_year = int(max(inputs.analysis_years))
    regional_rows = regional["rows"]
    report_rows: list[dict[str, object]] = []
    for factor_id in factor_ids:
        national_rows = [row for row in national["rows"] if row["factor_id"] == factor_id]
        first_national = next(row for row in national_rows if int(row["year"]) == first_year)
        last_national = next(row for row in national_rows if int(row["year"]) == last_year)
        first_region_rows = [row for row in regional_rows if row["factor_id"] == factor_id and int(row["year"]) == first_year]
        last_region_rows = [row for row in regional_rows if row["factor_id"] == factor_id and int(row["year"]) == last_year]
        regional_first_mean = float(np.mean([float(row["score"]) for row in first_region_rows])) if first_region_rows else 0.0
        regional_last_mean = float(np.mean([float(row["score"]) for row in last_region_rows])) if last_region_rows else 0.0
        report_rows.append(
            {
                "factor_id": factor_id,
                "factor_name": str(first_national["factor_name"]),
                "national_first_year_score": round(float(first_national["score"]), 6),
                "national_last_year_score": round(float(last_national["score"]), 6),
                "national_delta": round(float(last_national["score"]) - float(first_national["score"]), 6),
                "regional_first_year_mean_score": round(regional_first_mean, 6),
                "regional_last_year_mean_score": round(regional_last_mean, 6),
                "regional_delta": round(regional_last_mean - regional_first_mean, 6),
            }
        )
    report_rows.sort(key=lambda row: (-float(row["national_delta"]), str(row["factor_id"])))
    write_json(ctx.experiment_dir / "factor_drift_report.json", report_rows)
    fig, ax = plt.subplots()
    labels = [str(row["factor_id"]) for row in report_rows]
    deltas = [float(row["national_delta"]) for row in report_rows]
    ax.barh(labels, deltas)
    ax.set_title("AN-01C National Factor Drift")
    ax.set_xlabel("Last year minus first year score")
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "factor_drift_summary.png")
    plt.close(fig)
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
        },
        coverage_summary={
            "analysis_years": inputs.analysis_years,
            "factor_count": len(report_rows),
            "drift_reference_years": [first_year, last_year],
        },
        decision={"completed": True, "keep": True, "reason": "Factor drift report generated successfully."},
        numeric_justification=[
            {
                "name": "drift_start_year",
                "value": first_year,
                "role": "drift_reference_start",
                "source_type": "estimated",
                "estimation_data": "supported analysis year set",
                "estimation_method": "minimum supported analysis year",
                "uncertainty": "none",
                "why_needed": "Anchors drift to the first supported year instead of a manual start point.",
            },
            {
                "name": "drift_end_year",
                "value": last_year,
                "role": "drift_reference_end",
                "source_type": "estimated",
                "estimation_data": "supported analysis year set",
                "estimation_method": "maximum supported analysis year",
                "uncertainty": "none",
                "why_needed": "Anchors drift to the last supported year instead of a manual end point.",
            },
        ],
    )
    return {"rows": report_rows, "artifacts": artifacts}


#============================================================
# KP Mapping Experiments (AN-02A/B/C)
#============================================================

from .analytics import _factor_static_metrics, _geometric_mean, compute_an01a_outputs
from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .numeric_policy import numerical_guard_entry
from .registry import KP_COLLAPSED_NAMES, TRANSITION_NAMES
from .sources import TransitionResearchInputs, load_transition_research_inputs

HOOK_TO_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "diagnosis_transitions": ("U_to_D",),
    "linkage_transitions": ("D_to_A", "L_to_A"),
    "suppression_transitions": ("A_to_V",),
    "retention_attrition_transitions": ("A_to_L", "L_to_A"),
    "subgroup_allocation_priors": (),
}

BEST_TARGET_TO_TRANSITION: dict[str, str] = {
    "diagnosed_stock": "U_to_D",
    "testing_coverage": "U_to_D",
    "art_stock": "D_to_A",
    "alive_on_art": "D_to_A",
    "documented_suppression": "A_to_V",
    "viral_load_tested_stock": "A_to_V",
    "viral_load_tested_among_art": "A_to_V",
    "suppressed_among_art": "A_to_V",
}


def _factor_transition_rows(inputs: TransitionResearchInputs) -> list[dict[str, object]]:
    metrics = _factor_static_metrics(inputs)
    rows: list[dict[str, object]] = []
    for factor_id in sorted(inputs.retained_factor_lookup):
        factor_row = inputs.retained_factor_lookup[factor_id]
        hooks = [str(value) for value in list(factor_row.get("transition_hooks", []))]
        raw_scores = {transition: 0.0 for transition in TRANSITION_NAMES}
        for hook in hooks:
            mapped = HOOK_TO_TRANSITIONS.get(hook, ())
            contribution = 1.0 / float(len(mapped)) if mapped else 0.0
            for transition in mapped:
                raw_scores[transition] += contribution
        best_target_transition = BEST_TARGET_TO_TRANSITION.get(str(factor_row.get("best_target") or ""))
        if best_target_transition:
            raw_scores[best_target_transition] += 1.0
        raw_total = float(sum(raw_scores.values()))
        confidence = _geometric_mean(
            [
                float(metrics[factor_id]["structural_strength"]),
                float(metrics[factor_id]["support_rank"]),
                float(metrics[factor_id]["phase3_target_rank"]),
            ]
        )
        for transition in TRANSITION_NAMES:
            share = raw_scores[transition] / raw_total if raw_total > 0.0 else 0.0
            rows.append(
                {
                    "factor_id": factor_id,
                    "factor_name": str(factor_row.get("factor_name") or factor_id),
                    "transition": transition,
                    "relevance": round(share * confidence, 6),
                    "normalized_transition_share": round(share, 6),
                    "confidence": round(confidence, 6),
                    "best_target": str(factor_row.get("best_target") or ""),
                    "transition_hooks": hooks,
                    "support_count": int(inputs.factor_support_counts.get(factor_id, 0)),
                    "provenance": {"hooks": hooks, "best_target": str(factor_row.get("best_target") or "")},
                }
            )
    return rows


def _collapse_kp_distribution(inputs: TransitionResearchInputs) -> tuple[dict[str, float], list[str]]:
    national_distribution = dict((inputs.subgroup_weight_summary.get("national_kp_distribution") or {}))
    other_mass = sum(
        float(value)
        for key, value in national_distribution.items()
        if str(key).lower() not in {"msm", "tgw"}
    )
    collapsed = {
        "msm": float(national_distribution.get("msm", 0.0)),
        "tgw": float(national_distribution.get("tgw", 0.0)),
        "other": other_mass,
    }
    total = float(sum(collapsed.values()))
    if total > 0.0:
        collapsed = {key: float(value) / total for key, value in collapsed.items()}
    reasons: list[str] = []
    if "tgw" not in national_distribution:
        reasons.append("current_phase3_subgroup_summary_has_no_explicit_tgw_mass")
    if not inputs.subgroup_anchor_pack.get("national_kp_profile"):
        reasons.append("subgroup_anchor_pack_has_no_national_kp_profile")
    return collapsed, reasons


def run_an02a(ctx: TransitionResearchContext) -> dict[str, object]:
    inputs = load_transition_research_inputs(ctx)
    rows = _factor_transition_rows(inputs)
    factor_ids = sorted(inputs.retained_factor_lookup)
    matrix = np.zeros((len(factor_ids), len(TRANSITION_NAMES)), dtype=np.float32)
    for factor_idx, factor_id in enumerate(factor_ids):
        factor_rows = [row for row in rows if row["factor_id"] == factor_id]
        for transition_idx, transition in enumerate(TRANSITION_NAMES):
            matrix[factor_idx, transition_idx] = float(
                next(row["relevance"] for row in factor_rows if row["transition"] == transition)
            )
    write_json(ctx.experiment_dir / "factor_transition_relevance.json", rows)
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title("AN-02A Factor-to-Transition Relevance")
    ax.set_xticks(np.arange(len(TRANSITION_NAMES)), labels=list(TRANSITION_NAMES))
    ax.set_yticks(np.arange(len(factor_ids)), labels=factor_ids)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "factor_transition_heatmap.png")
    plt.close(fig)
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
        },
        coverage_summary={"factor_count": len(factor_ids), "transition_count": len(TRANSITION_NAMES), "phase15_source_run_id": ctx.source_run_id},
        decision={"completed": True, "keep": True, "reason": "Factor-to-transition relevance map generated successfully."},
        numeric_justification=[
            {
                "name": "transition_count",
                "value": len(TRANSITION_NAMES),
                "role": "exact_transition_grid_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of exact HIV transition channels in the transition-centric plan",
                "uncertainty": "none",
                "why_needed": "Normalizes hook mass across the exact transition grid without ad hoc weighting.",
            },
            {
                "name": "transition_confidence_component_count",
                "value": 3,
                "role": "confidence_component_count",
                "source_type": "physical_constraint",
                "estimation_data": "factor metadata",
                "estimation_method": "count of confidence channels: structural strength, blanket support, phase3 target alignment",
                "uncertainty": "none",
                "why_needed": "The geometric mean uses the exact number of confidence channels available for each factor.",
            },
        ],
    )
    return {"rows": rows, "artifacts": artifacts}


def run_an02b(ctx: TransitionResearchContext) -> dict[str, object]:
    inputs = load_transition_research_inputs(ctx)
    transition_rows = _factor_transition_rows(inputs)
    collapsed_distribution, low_confidence_reasons = _collapse_kp_distribution(inputs)
    transition_strength = {
        factor_id: float(sum(float(row["relevance"]) for row in transition_rows if row["factor_id"] == factor_id))
        for factor_id in sorted(inputs.retained_factor_lookup)
    }
    factor_ids = sorted(inputs.retained_factor_lookup)
    rows: list[dict[str, object]] = []
    matrix = np.zeros((len(factor_ids), len(KP_COLLAPSED_NAMES)), dtype=np.float32)
    for factor_idx, factor_id in enumerate(factor_ids):
        factor_row = inputs.retained_factor_lookup[factor_id]
        hooks = [str(value) for value in list(factor_row.get("transition_hooks", []))]
        subgroup_hook_active = "subgroup_allocation_priors" in hooks
        kp_total_strength = float(transition_strength[factor_id]) if subgroup_hook_active else 0.0
        for kp_idx, kp_name in enumerate(KP_COLLAPSED_NAMES):
            relevance = kp_total_strength * float(collapsed_distribution.get(kp_name, 0.0))
            matrix[factor_idx, kp_idx] = relevance
            rows.append(
                {
                    "factor_id": factor_id,
                    "factor_name": str(factor_row.get("factor_name") or factor_id),
                    "kp_group": kp_name,
                    "relevance": round(float(relevance), 6),
                    "confidence": round(float(kp_total_strength), 6),
                    "subgroup_hook_active": subgroup_hook_active,
                    "base_distribution_share": round(float(collapsed_distribution.get(kp_name, 0.0)), 6),
                    "low_confidence_reasons": low_confidence_reasons,
                }
            )
    write_json(ctx.experiment_dir / "factor_kp_relevance.json", rows)
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title("AN-02B Factor-to-KP Relevance")
    ax.set_xticks(np.arange(len(KP_COLLAPSED_NAMES)), labels=list(KP_COLLAPSED_NAMES))
    ax.set_yticks(np.arange(len(factor_ids)), labels=factor_ids)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "factor_kp_heatmap.png")
    plt.close(fig)
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
        },
        coverage_summary={
            "factor_count": len(factor_ids),
            "kp_group_count": len(KP_COLLAPSED_NAMES),
            "phase3_result_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "low_confidence_reasons": low_confidence_reasons,
        },
        decision={"completed": True, "keep": True, "reason": "Factor-to-KP relevance map generated successfully with explicit low-confidence flags."},
        numeric_justification=[
            {
                "name": "kp_group_count",
                "value": len(KP_COLLAPSED_NAMES),
                "role": "collapsed_kp_grid_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of requested collapsed KP groups: msm, tgw, other",
                "uncertainty": "none",
                "why_needed": "Defines the KP relevance grid without introducing a hidden grouping choice.",
            },
            {
                "name": "collapsed_other_kp_mass",
                "value": float(collapsed_distribution.get("other", 0.0)),
                "role": "collapsed_kp_distribution_share",
                "source_type": "estimated",
                "estimation_data": "phase3 subgroup_weight_summary national_kp_distribution",
                "estimation_method": "sum of all non-MSM and non-TGW KP shares, then normalize within the collapsed grid",
                "uncertainty": "depends on phase3 subgroup prior learning",
                "why_needed": "Produces the requested MSM/TGW/other view from the active Phase 3 subgroup output instead of inventing a manual collapse.",
            },
        ],
    )
    return {"rows": rows, "artifacts": artifacts}


def run_an02c(ctx: TransitionResearchContext) -> dict[str, object]:
    inputs = load_transition_research_inputs(ctx)
    yearly_payload = compute_an01a_outputs(inputs)
    transition_rows = _factor_transition_rows(inputs)
    collapsed_distribution, low_confidence_reasons = _collapse_kp_distribution(inputs)
    factor_year_score = {
        (str(row["factor_id"]), int(row["year"])): float(row["score"]) for row in yearly_payload["rows"]
    }
    factor_transition_share = {
        (str(row["factor_id"]), str(row["transition"])): float(row["normalized_transition_share"]) for row in transition_rows
    }
    factor_has_subgroup_hook = {
        factor_id: "subgroup_allocation_priors" in list((inputs.retained_factor_lookup.get(factor_id) or {}).get("transition_hooks", []))
        for factor_id in inputs.retained_factor_lookup
    }
    rows: list[dict[str, object]] = []
    matrix_rows: list[list[float]] = []
    labels: list[str] = []
    years = inputs.analysis_years
    for kp_name in KP_COLLAPSED_NAMES:
        for transition in TRANSITION_NAMES:
            labels.append(f"{kp_name}|{transition}")
            yearly_scores: list[float] = []
            for year in years:
                contributions: list[dict[str, object]] = []
                total = 0.0
                for factor_id in sorted(inputs.retained_factor_lookup):
                    subgroup_share = float(collapsed_distribution.get(kp_name, 0.0)) if factor_has_subgroup_hook[factor_id] else 0.0
                    contribution = (
                        float(factor_year_score[(factor_id, year)])
                        * float(factor_transition_share[(factor_id, transition)])
                        * subgroup_share
                    )
                    if contribution > 0.0:
                        contributions.append({"factor_id": factor_id, "contribution": round(float(contribution), 6)})
                    total += contribution
                contributions.sort(key=lambda row: (-float(row["contribution"]), str(row["factor_id"])))
                rows.append(
                    {
                        "year": year,
                        "kp_group": kp_name,
                        "transition": transition,
                        "score": round(float(total), 6),
                        "top_factor_contributions": contributions,
                        "low_confidence_reasons": low_confidence_reasons,
                    }
                )
                yearly_scores.append(float(total))
            matrix_rows.append(yearly_scores)
    matrix = np.asarray(matrix_rows, dtype=np.float32) if matrix_rows else np.zeros((0, len(years)), dtype=np.float32)
    write_json(ctx.experiment_dir / "kp_transition_relevance.json", rows)
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title("AN-02C KP-Transition Relevance Drift")
    ax.set_xticks(np.arange(len(years)), labels=[str(year) for year in years])
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(ctx.experiment_dir / "kp_transition_drift.png")
    plt.close(fig)
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
        },
        coverage_summary={"analysis_years": years, "kp_group_count": len(KP_COLLAPSED_NAMES), "transition_count": len(TRANSITION_NAMES), "low_confidence_reasons": low_confidence_reasons},
        decision={"completed": True, "keep": True, "reason": "KP-transition relevance drift artifacts generated successfully."},
        numeric_justification=[
            {
                "name": "kp_transition_grid_size",
                "value": len(KP_COLLAPSED_NAMES) * len(TRANSITION_NAMES),
                "role": "heatmap_row_count",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "Cartesian product of collapsed KP groups and exact transitions",
                "uncertainty": "none",
                "why_needed": "Defines the KP-transition drift surface directly from the requested analytic grid.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="future_normalization_guard",
                why_needed="Keeps the drift surface numerically stable if later normalization is added.",
            ),
        ],
    )
    return {"rows": rows, "artifacts": artifacts}
