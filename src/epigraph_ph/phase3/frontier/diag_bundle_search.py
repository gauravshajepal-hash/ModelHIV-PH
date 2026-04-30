from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.runtime import ROOT_DIR, read_json, utc_now_iso, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .integrated_autoresearch import IntegratedSnapshot, _build_snapshot, _select_matrix
from .numeric_policy import numerical_guard_entry
from .strict_diagnosis_kernel_research import (
    BlockedTimeContract,
    DiagnosisKernelCandidate,
    DiagnosisKernelFit,
    _baseline_payload,
    _best_fit,
    _bic_from_residuals,
    _build_blocked_time_contract,
    _compute_window_metrics,
    _fit_candidate_with_matrices,
    _fit_payload,
    _frontier_rows,
    _latest_diag02a_experiment_dir,
    _matrices_for_reference,
    _parse_diag_candidate,
    _score_tuple,
    _select_promoted_strict_reference,
)


def _latest_hmba00_experiment_dir() -> Path | None:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "transition_research" / "HMBA-00-hierarchical-contract-freeze"
        manifest_path = experiment_dir / "module_seed_manifest.json"
        if manifest_path.exists():
            candidates.append((manifest_path.stat().st_mtime, experiment_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item[0]))[1]


def _bundle_seed_rows(snapshot: IntegratedSnapshot, module_name: str, *, max_candidates: int = 4) -> list[dict[str, Any]]:
    experiment_dir = _latest_hmba00_experiment_dir()
    if experiment_dir is None:
        raise FileNotFoundError("No HMBA-00 module seed manifest was found under artifacts/runs")
    manifest = read_json(experiment_dir / "module_seed_manifest.json", default={})
    rankings = dict(manifest.get("module_bundle_rankings") or {})
    rows = [dict(row) for row in list(rankings.get(module_name) or []) if isinstance(row, dict)]
    filtered: list[dict[str, Any]] = []
    for row in rows:
        member_ids = [str(value) for value in list(row.get("member_factor_ids") or []) if str(value) in snapshot.feature_ids]
        if not member_ids:
            continue
        if float(row.get("bundle_score") or 0.0) <= 0.0:
            continue
        filtered.append(
            {
                "bundle_name": str(row.get("bundle_name") or ""),
                "bundle_score": float(row.get("bundle_score") or 0.0),
                "member_factor_ids": member_ids,
            }
        )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in filtered:
        bundle_name = str(row["bundle_name"])
        if bundle_name in seen:
            continue
        seen.add(bundle_name)
        deduped.append(row)
    return deduped[: max(1, int(max_candidates))]


def _selected_feature_ids(
    reference_feature_ids: list[str],
    bundle_rows: list[dict[str, Any]],
    selected_bundles: list[str] | None,
) -> list[str]:
    if selected_bundles is None:
        return list(reference_feature_ids)
    allowed: set[str] = set()
    selected_set = {str(value) for value in selected_bundles}
    for row in bundle_rows:
        if str(row["bundle_name"]) not in selected_set:
            continue
        allowed.update(str(value) for value in list(row.get("member_factor_ids") or []))
    return [feature_id for feature_id in reference_feature_ids if feature_id in allowed]


def _bundle_state_label(incidence_bundles: list[str] | None, utd_bundles: list[str] | None) -> str:
    def _label(values: list[str] | None) -> str:
        if values is None:
            return "all"
        if not values:
            return "none"
        return "+".join(str(value) for value in values)

    return f"inc={_label(incidence_bundles)}|ud={_label(utd_bundles)}"


def _bundle_candidate_from_reference(
    reference_candidate: DiagnosisKernelCandidate,
    incidence_bundles: list[str] | None,
    utd_bundles: list[str] | None,
) -> DiagnosisKernelCandidate:
    return DiagnosisKernelCandidate(
        candidate_id=_bundle_state_label(incidence_bundles, utd_bundles),
        diagnosis_kind=str(reference_candidate.diagnosis_kind),
        care_family=str(reference_candidate.care_family),
        kernel_width=int(reference_candidate.kernel_width),
        reference_hidden_rank=int(reference_candidate.reference_hidden_rank),
        use_observation_covariates=bool(reference_candidate.use_observation_covariates),
    )


def _bundle_filtered_matrices(
    snapshot: IntegratedSnapshot,
    reference_config: Any,
    incidence_bundle_rows: list[dict[str, Any]],
    utd_bundle_rows: list[dict[str, Any]],
    incidence_bundles: list[str] | None,
    utd_bundles: list[str] | None,
) -> dict[str, Any]:
    base = _matrices_for_reference(snapshot, reference_config)
    incidence_feature_ids = _selected_feature_ids(list(base["feature_ids"]["incidence"]), incidence_bundle_rows, incidence_bundles)
    utd_feature_ids = _selected_feature_ids(list(base["feature_ids"]["transition"]["U_to_D"]), utd_bundle_rows, utd_bundles)
    transition_feature_ids = {transition: list(value) for transition, value in dict(base["feature_ids"]["transition"]).items()}
    transition_feature_ids["U_to_D"] = list(utd_feature_ids)
    return {
        "incidence_direct": _select_matrix(snapshot, incidence_feature_ids),
        "transition_direct": {
            transition: (
                _select_matrix(snapshot, utd_feature_ids)
                if transition == "U_to_D"
                else np.asarray(base["transition_direct"][transition], dtype=np.float64)
            )
            for transition in base["transition_direct"]
        },
        "observation_direct": {
            "tested_for_viral_load": np.asarray(base["observation_direct"]["tested_for_viral_load"], dtype=np.float64),
            "virally_suppressed": np.asarray(base["observation_direct"]["virally_suppressed"], dtype=np.float64),
        },
        "hidden": np.asarray(base["hidden"], dtype=np.float64),
        "feature_ids": {
            "incidence": list(incidence_feature_ids),
            "transition": transition_feature_ids,
            "observation": {
                "tested_for_viral_load": list(base["feature_ids"]["observation"]["tested_for_viral_load"]),
                "virally_suppressed": list(base["feature_ids"]["observation"]["virally_suppressed"]),
            },
        },
    }


def _frontier_row(
    fit: DiagnosisKernelFit,
    *,
    stage: str,
    module_name: str,
    incidence_bundles: list[str] | None,
    utd_bundles: list[str] | None,
) -> dict[str, Any]:
    return {
        "candidate_id": str(fit.candidate.candidate_id),
        "stage": str(stage),
        "module_name": str(module_name),
        "selected_incidence_bundles": list(incidence_bundles) if incidence_bundles is not None else None,
        "selected_u_to_d_bundles": list(utd_bundles) if utd_bundles is not None else None,
        "validation_primary_loss": float(fit.split_metrics["validation"]["primary_loss"]),
        "validation_diag_flow_loss": float(fit.split_metrics["validation"]["diag_flow_loss"]),
        "holdout_primary_loss": float(fit.split_metrics["holdout"]["primary_loss"]),
        "holdout_diag_flow_loss": float(fit.split_metrics["holdout"]["diag_flow_loss"]),
        "train_bic": float(fit.train_bic),
        "success": bool(fit.success),
    }


def _trace_row(step: int, selected_bundles: list[str], fit: DiagnosisKernelFit) -> dict[str, Any]:
    return {
        "step": int(step),
        "selected_bundles": list(selected_bundles),
        "validation_primary_loss": float(fit.split_metrics["validation"]["primary_loss"]),
        "validation_diag_flow_loss": float(fit.split_metrics["validation"]["diag_flow_loss"]),
        "holdout_primary_loss": float(fit.split_metrics["holdout"]["primary_loss"]),
        "holdout_diag_flow_loss": float(fit.split_metrics["holdout"]["diag_flow_loss"]),
    }


def _write_frontier_chart(path: Path, frontier_rows: list[dict[str, Any]], reference_candidate_id: str) -> None:
    x_labels = [str(row["candidate_id"]) for row in frontier_rows]
    x_values = np.arange(len(frontier_rows))
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.2), sharex=True)
    axes[0].plot(x_values, [float(row["validation_primary_loss"]) for row in frontier_rows], marker="o", linewidth=1.4, label="validation primary")
    axes[0].plot(x_values, [float(row["holdout_primary_loss"]) for row in frontier_rows], marker="s", linewidth=1.2, label="holdout primary")
    ref_index = next((idx for idx, row in enumerate(frontier_rows) if str(row["candidate_id"]) == str(reference_candidate_id)), None)
    if ref_index is not None:
        axes[0].axvline(float(ref_index), color="#666666", linestyle="--", linewidth=1.0, alpha=0.7, label="strict reference")
    axes[0].set_title("DIAG-02B Bundle Search Frontier")
    axes[0].set_ylabel("primary loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(x_values, [float(row["validation_diag_flow_loss"]) for row in frontier_rows], marker="o", linewidth=1.4, label="validation diagnosis flow")
    axes[1].plot(x_values, [float(row["holdout_diag_flow_loss"]) for row in frontier_rows], marker="s", linewidth=1.2, label="holdout diagnosis flow")
    if ref_index is not None:
        axes[1].axvline(float(ref_index), color="#666666", linestyle="--", linewidth=1.0, alpha=0.7, label="strict reference")
    axes[1].set_ylabel("diagnosis-flow loss")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(x_labels, rotation=35, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_trace_chart(path: Path, incidence_trace: list[dict[str, Any]], utd_trace: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.4), sharex=False)
    for axis, trace_rows, title in (
        (axes[0], incidence_trace, "Incidence Bundle Search Trace"),
        (axes[1], utd_trace, "U_to_D Bundle Search Trace"),
    ):
        steps = [int(row["step"]) for row in trace_rows]
        axis.plot(steps, [float(row["validation_primary_loss"]) for row in trace_rows], marker="o", linewidth=1.4, label="validation primary")
        axis.plot(steps, [float(row["holdout_primary_loss"]) for row in trace_rows], marker="s", linewidth=1.2, label="holdout primary")
        axis.set_title(title)
        axis.set_xlabel("forward-selection step")
        axis.set_ylabel("primary loss")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_model_comparison_chart(
    path: Path,
    reference_fit: DiagnosisKernelFit,
    incidence_fit: DiagnosisKernelFit,
    utd_fit: DiagnosisKernelFit,
    best_fit: DiagnosisKernelFit,
    baselines: dict[str, Any],
) -> None:
    labels = ["carry_forward", "simple_compartmental", "strict_reference", "incidence_best", "u_to_d_best", "bundle_best"]
    validation_primary = [
        float(baselines["carry_forward"]["validation"]["primary_loss"]),
        float(baselines["simple_compartmental"]["validation"]["primary_loss"]),
        float(reference_fit.split_metrics["validation"]["primary_loss"]),
        float(incidence_fit.split_metrics["validation"]["primary_loss"]),
        float(utd_fit.split_metrics["validation"]["primary_loss"]),
        float(best_fit.split_metrics["validation"]["primary_loss"]),
    ]
    holdout_primary = [
        float(baselines["carry_forward"]["holdout"]["primary_loss"]),
        float(baselines["simple_compartmental"]["holdout"]["primary_loss"]),
        float(reference_fit.split_metrics["holdout"]["primary_loss"]),
        float(incidence_fit.split_metrics["holdout"]["primary_loss"]),
        float(utd_fit.split_metrics["holdout"]["primary_loss"]),
        float(best_fit.split_metrics["holdout"]["primary_loss"]),
    ]
    validation_diag = [
        float(baselines["carry_forward"]["validation"]["diag_flow_loss"]),
        float(baselines["simple_compartmental"]["validation"]["diag_flow_loss"]),
        float(reference_fit.split_metrics["validation"]["diag_flow_loss"]),
        float(incidence_fit.split_metrics["validation"]["diag_flow_loss"]),
        float(utd_fit.split_metrics["validation"]["diag_flow_loss"]),
        float(best_fit.split_metrics["validation"]["diag_flow_loss"]),
    ]
    holdout_diag = [
        float(baselines["carry_forward"]["holdout"]["diag_flow_loss"]),
        float(baselines["simple_compartmental"]["holdout"]["diag_flow_loss"]),
        float(reference_fit.split_metrics["holdout"]["diag_flow_loss"]),
        float(incidence_fit.split_metrics["holdout"]["diag_flow_loss"]),
        float(utd_fit.split_metrics["holdout"]["diag_flow_loss"]),
        float(best_fit.split_metrics["holdout"]["diag_flow_loss"]),
    ]
    x_values = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.4), sharex=True)
    axes[0].bar(x_values - width / 2, validation_primary, width=width, label="validation")
    axes[0].bar(x_values + width / 2, holdout_primary, width=width, label="holdout")
    axes[0].set_title("DIAG-02B Blocked-Time Primary Loss Comparison")
    axes[0].set_ylabel("primary loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x_values - width / 2, validation_diag, width=width, label="validation")
    axes[1].bar(x_values + width / 2, holdout_diag, width=width, label="holdout")
    axes[1].set_title("DIAG-02B Blocked-Time Diagnosis-Flow Loss Comparison")
    axes[1].set_ylabel("diagnosis-flow loss")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_prediction_chart(
    path: Path,
    snapshot: IntegratedSnapshot,
    contract: BlockedTimeContract,
    reference_fit: DiagnosisKernelFit,
    best_fit: DiagnosisKernelFit,
) -> None:
    relevant_mask = np.logical_or(contract.validation_mask, contract.holdout_mask)
    indices = np.where(relevant_mask)[0]
    quarter_labels = [snapshot.model_quarters[idx] for idx in indices]
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 8.5), sharex=True)
    for axis, metric_name, title in (
        (axes[0], "new_diagnosed_cases_period", "New Diagnosed Cases"),
        (axes[1], "diagnosed_plhiv", "Diagnosed PLHIV"),
    ):
        observed_mask = np.logical_and(relevant_mask, snapshot.metric_masks[metric_name])
        axis.plot(
            quarter_labels,
            [float(snapshot.metric_values[metric_name][idx]) if bool(observed_mask[idx]) else np.nan for idx in indices],
            marker="o",
            linewidth=1.6,
            label="observed",
        )
        for label, fit in (("strict reference", reference_fit), ("bundle best", best_fit)):
            axis.plot(
                quarter_labels,
                [float(fit.simulation["predictions"][metric_name][idx]) for idx in indices],
                linewidth=1.4,
                label=label,
            )
        axis.set_title(f"DIAG-02B {title} on Validation + Holdout")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].set_xticks(range(len(quarter_labels)))
    axes[-1].set_xticklabels(quarter_labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_selection_chart(path: Path, incidence_bundles: list[str], utd_bundles: list[str]) -> None:
    labels = ["incidence", "U_to_D"]
    counts = [len(incidence_bundles), len(utd_bundles)]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(labels, counts, color=["#4C78A8", "#F58518"])
    ax.set_title("DIAG-02B Selected Bundle Counts")
    ax.set_ylabel("accepted bundles")
    ax.grid(axis="y", alpha=0.3)
    for bar, bundle_names in zip(bars, (incidence_bundles, utd_bundles)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            "\n".join(bundle_names) if bundle_names else "none",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _export_paper_figures(ctx: TransitionResearchContext, chart_paths: dict[str, Path]) -> dict[str, Any]:
    archive_dir = ROOT_DIR / "artifacts" / "paper_figures" / "phase3_diag_kernel_20260409"
    archive_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = list(read_json(archive_dir / "figure_manifest.json", default=[]))
    retained = [
        row
        for row in existing_manifest
        if isinstance(row, dict) and not str(row.get("figure_id") or "").startswith("fig_diag02b_")
    ]
    figure_specs = [
        ("fig_diag02b_01", "Blocked-Time Bundle Search Frontier", "Validation and holdout losses across bundle-search candidates after fixing the promoted strict branch as the reference.", "diag02b_bundle_frontier.png", chart_paths.get("frontier")),
        ("fig_diag02b_02", "Blocked-Time Bundle Search Traces", "Forward-selection traces for incidence and U_to_D bundle search under the blocked-time contract.", "diag02b_bundle_traces.png", chart_paths.get("traces")),
        ("fig_diag02b_03", "Blocked-Time Bundle Search Comparison", "Blocked-time comparison between carry-forward, simple compartmental, the strict reference, module-local bundle winners, and the final bundle-search winner.", "diag02b_model_comparison.png", chart_paths.get("comparison")),
        ("fig_diag02b_04", "Blocked-Time Bundle Search Predictions", "Observed and predicted new diagnoses and diagnosed stock for the strict reference and the best bundle-search candidate.", "diag02b_predictions.png", chart_paths.get("predictions")),
        ("fig_diag02b_05", "Blocked-Time Bundle Selection Summary", "Accepted incidence and U_to_D bundle counts and labels under blocked-time forward selection.", "diag02b_bundle_selection.png", chart_paths.get("selection")),
    ]
    new_rows: list[dict[str, Any]] = []
    for figure_id, title, caption, filename, source_path in figure_specs:
        if source_path is None or not source_path.exists():
            continue
        destination = archive_dir / filename
        shutil.copy2(source_path, destination)
        new_rows.append(
            {
                "figure_id": figure_id,
                "title": title,
                "caption": caption,
                "source_path": str(source_path),
                "archived_path": str(destination),
            }
        )
    manifest = retained + new_rows
    manifest_path = archive_dir / "figure_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "archive_dir": str(archive_dir),
        "figure_manifest": str(manifest_path),
        "figure_count_added": int(len(new_rows)),
        "figure_count_total": int(len(manifest)),
    }


def run_diag_02b(ctx: TransitionResearchContext) -> dict[str, Any]:
    snapshot = _build_snapshot(ctx)
    reference_config, reference_candidate = _select_promoted_strict_reference()
    contract = _build_blocked_time_contract(snapshot)
    incidence_bundle_rows = _bundle_seed_rows(snapshot, "incidence", max_candidates=4)
    utd_bundle_rows = _bundle_seed_rows(snapshot, "U_to_D", max_candidates=4)

    fit_cache: dict[tuple[Any, Any], DiagnosisKernelFit] = {}
    frontier_rows: list[dict[str, Any]] = []

    def _fit_state(incidence_bundles: list[str] | None, utd_bundles: list[str] | None, *, stage: str, module_name: str) -> DiagnosisKernelFit:
        cache_key = (
            None if incidence_bundles is None else tuple(incidence_bundles),
            None if utd_bundles is None else tuple(utd_bundles),
        )
        if cache_key not in fit_cache:
            matrices = _bundle_filtered_matrices(snapshot, reference_config, incidence_bundle_rows, utd_bundle_rows, incidence_bundles, utd_bundles)
            candidate = _bundle_candidate_from_reference(reference_candidate, incidence_bundles, utd_bundles)
            fit_cache[cache_key] = _fit_candidate_with_matrices(snapshot, candidate, matrices, contract)
        fit = fit_cache[cache_key]
        frontier_rows.append(_frontier_row(fit, stage=stage, module_name=module_name, incidence_bundles=incidence_bundles, utd_bundles=utd_bundles))
        return fit

    reference_fit = _fit_state(None, None, stage="reference", module_name="reference")

    def _forward_search(module_name: str, bundle_rows: list[dict[str, Any]]) -> tuple[list[str], DiagnosisKernelFit, list[dict[str, Any]]]:
        selected: list[str] = []
        current_fit = _fit_state(
            [] if module_name == "incidence" else None,
            [] if module_name == "U_to_D" else None,
            stage=f"{module_name}_baseline",
            module_name=module_name,
        )
        trace_rows = [_trace_row(0, selected, current_fit)]
        remaining = [str(row["bundle_name"]) for row in bundle_rows]
        step = 1
        while remaining and len(selected) < 2:
            candidate_results: list[tuple[str, DiagnosisKernelFit]] = []
            for bundle_name in remaining:
                candidate_selected = selected + [bundle_name]
                fit = _fit_state(
                    candidate_selected if module_name == "incidence" else None,
                    candidate_selected if module_name == "U_to_D" else None,
                    stage=f"{module_name}_candidate",
                    module_name=module_name,
                )
                candidate_results.append((bundle_name, fit))
            best_bundle, best_fit = min(candidate_results, key=lambda item: _score_tuple(item[1].split_metrics["validation"]))
            if _score_tuple(best_fit.split_metrics["validation"]) < _score_tuple(current_fit.split_metrics["validation"]):
                selected.append(best_bundle)
                remaining.remove(best_bundle)
                current_fit = best_fit
                trace_rows.append(_trace_row(step, selected, current_fit))
                step += 1
                continue
            break
        return selected, current_fit, trace_rows

    selected_incidence_bundles, incidence_fit, incidence_trace = _forward_search("incidence", incidence_bundle_rows)
    selected_utd_bundles, utd_fit, utd_trace = _forward_search("U_to_D", utd_bundle_rows)
    combined_fit = _fit_state(selected_incidence_bundles, selected_utd_bundles, stage="combined", module_name="combined")
    baselines = _baseline_payload(snapshot, contract)

    final_candidates = {
        "strict_reference": reference_fit,
        "incidence_best": incidence_fit,
        "u_to_d_best": utd_fit,
        "combined": combined_fit,
    }
    best_label, best_fit = min(final_candidates.items(), key=lambda item: _score_tuple(item[1].split_metrics["validation"]))

    contract_path = ctx.experiment_dir / "bundle_search_contract.json"
    frontier_path = ctx.experiment_dir / "bundle_search_frontier.json"
    trace_path = ctx.experiment_dir / "bundle_search_traces.json"
    evaluation_path = ctx.experiment_dir / "bundle_search_evaluation.json"
    comparison_path = ctx.experiment_dir / "bundle_search_model_comparison.json"
    prediction_rows_path = ctx.experiment_dir / "bundle_search_prediction_rows.json"
    dashboard_path = ctx.experiment_dir / "bundle_search_dashboard.md"
    frontier_chart_path = ctx.experiment_dir / "bundle_search_frontier.png"
    trace_chart_path = ctx.experiment_dir / "bundle_search_traces.png"
    comparison_chart_path = ctx.experiment_dir / "bundle_search_model_comparison.png"
    prediction_chart_path = ctx.experiment_dir / "bundle_search_predictions.png"
    selection_chart_path = ctx.experiment_dir / "bundle_search_selection.png"

    contract_payload = {
        "generated_at": utc_now_iso(),
        "reference_candidate_id": str(reference_candidate.candidate_id),
        "reference_integrated_candidate_id": str(reference_config.candidate_id),
        "train_diagnosis_quarters": list(contract.train_diagnosis_quarters),
        "validation_quarters": list(contract.validation_quarters),
        "holdout_quarters": list(contract.holdout_quarters),
        "incidence_bundle_seed_rows": incidence_bundle_rows,
        "u_to_d_bundle_seed_rows": utd_bundle_rows,
    }
    write_json(contract_path, contract_payload)
    write_json(frontier_path, frontier_rows)
    write_json(trace_path, {"incidence_trace": incidence_trace, "u_to_d_trace": utd_trace, "selected_incidence_bundles": selected_incidence_bundles, "selected_u_to_d_bundles": selected_utd_bundles})

    comparison_payload = {
        "carry_forward": baselines["carry_forward"],
        "simple_compartmental": baselines["simple_compartmental"],
        "strict_reference": _fit_payload(reference_fit),
        "incidence_best": _fit_payload(incidence_fit),
        "u_to_d_best": _fit_payload(utd_fit),
        "combined": _fit_payload(combined_fit),
        "best_label": str(best_label),
        "best_bundle_candidate_id": str(best_fit.candidate.candidate_id),
    }
    write_json(comparison_path, comparison_payload)

    relevant_mask = np.logical_or(contract.validation_mask, contract.holdout_mask)
    prediction_rows: list[dict[str, Any]] = []
    for idx in np.where(relevant_mask)[0]:
        prediction_rows.append(
            {
                "quarter": str(snapshot.model_quarters[idx]),
                "window": "validation" if bool(contract.validation_mask[idx]) else "holdout",
                "observed_new_diagnosed_cases_period": float(snapshot.metric_values["new_diagnosed_cases_period"][idx]) if bool(snapshot.metric_masks["new_diagnosed_cases_period"][idx]) else None,
                "reference_new_diagnosed_cases_period": float(reference_fit.simulation["predictions"]["new_diagnosed_cases_period"][idx]),
                "best_new_diagnosed_cases_period": float(best_fit.simulation["predictions"]["new_diagnosed_cases_period"][idx]),
                "observed_diagnosed_plhiv": float(snapshot.metric_values["diagnosed_plhiv"][idx]) if bool(snapshot.metric_masks["diagnosed_plhiv"][idx]) else None,
                "reference_diagnosed_plhiv": float(reference_fit.simulation["predictions"]["diagnosed_plhiv"][idx]),
                "best_diagnosed_plhiv": float(best_fit.simulation["predictions"]["diagnosed_plhiv"][idx]),
            }
        )
    write_json(prediction_rows_path, prediction_rows)

    _write_frontier_chart(frontier_chart_path, frontier_rows, reference_candidate.candidate_id)
    _write_trace_chart(trace_chart_path, incidence_trace, utd_trace)
    _write_model_comparison_chart(comparison_chart_path, reference_fit, incidence_fit, utd_fit, best_fit, baselines)
    _write_prediction_chart(prediction_chart_path, snapshot, contract, reference_fit, best_fit)
    _write_selection_chart(selection_chart_path, selected_incidence_bundles, selected_utd_bundles)
    paper_archive = _export_paper_figures(
        ctx,
        {
            "frontier": frontier_chart_path,
            "traces": trace_chart_path,
            "comparison": comparison_chart_path,
            "predictions": prediction_chart_path,
            "selection": selection_chart_path,
        },
    )

    dashboard_path.write_text(
        "\n".join(
            [
                "# DIAG-02B Blocked-Time Incidence + U_to_D Bundle Search Dashboard",
                "",
                "This experiment fixes the promoted strict diagnosis branch as the reference, keeps downstream care fixed, and searches only direct Phase 2 bundle subsets for incidence and U_to_D under the blocked-time contract.",
                "",
                f"- Selected incidence bundles: `{', '.join(selected_incidence_bundles) if selected_incidence_bundles else 'none'}`",
                f"- Selected U_to_D bundles: `{', '.join(selected_utd_bundles) if selected_utd_bundles else 'none'}`",
                f"- Best validation candidate: `{best_label}` / `{best_fit.candidate.candidate_id}`",
            ]
        ),
        encoding="utf-8",
    )

    evaluation_payload = {
        "reference_candidate_id": str(reference_candidate.candidate_id),
        "best_label": str(best_label),
        "strict_reference": _fit_payload(reference_fit),
        "incidence_best": _fit_payload(incidence_fit),
        "u_to_d_best": _fit_payload(utd_fit),
        "combined": _fit_payload(combined_fit),
        "best_bundle_candidate": _fit_payload(best_fit),
        "selected_incidence_bundles": selected_incidence_bundles,
        "selected_u_to_d_bundles": selected_utd_bundles,
        "best_vs_reference_validation_delta": {
            "primary_loss": float(best_fit.split_metrics["validation"]["primary_loss"] - reference_fit.split_metrics["validation"]["primary_loss"]),
            "diag_flow_loss": float(best_fit.split_metrics["validation"]["diag_flow_loss"] - reference_fit.split_metrics["validation"]["diag_flow_loss"]),
        },
        "best_vs_reference_holdout_delta": {
            "primary_loss": float(best_fit.split_metrics["holdout"]["primary_loss"] - reference_fit.split_metrics["holdout"]["primary_loss"]),
            "diag_flow_loss": float(best_fit.split_metrics["holdout"]["diag_flow_loss"] - reference_fit.split_metrics["holdout"]["diag_flow_loss"]),
        },
        "best_vs_carry_forward_validation_delta": {
            "primary_loss": float(best_fit.split_metrics["validation"]["primary_loss"] - baselines["carry_forward"]["validation"]["primary_loss"]),
            "diag_flow_loss": float(best_fit.split_metrics["validation"]["diag_flow_loss"] - baselines["carry_forward"]["validation"]["diag_flow_loss"]),
        },
        "best_vs_carry_forward_holdout_delta": {
            "primary_loss": float(best_fit.split_metrics["holdout"]["primary_loss"] - baselines["carry_forward"]["holdout"]["primary_loss"]),
            "diag_flow_loss": float(best_fit.split_metrics["holdout"]["diag_flow_loss"] - baselines["carry_forward"]["holdout"]["diag_flow_loss"]),
        },
        "paper_archive": paper_archive,
    }
    write_json(evaluation_path, evaluation_payload)

    decision = {
        "reference_candidate_id": str(reference_candidate.candidate_id),
        "best_label": str(best_label),
        "best_bundle_candidate_id": str(best_fit.candidate.candidate_id),
        "selected_incidence_bundles": selected_incidence_bundles,
        "selected_u_to_d_bundles": selected_utd_bundles,
        "best_validation_beats_reference": bool(_score_tuple(best_fit.split_metrics["validation"]) < _score_tuple(reference_fit.split_metrics["validation"])),
        "best_holdout_beats_reference": bool(_score_tuple(best_fit.split_metrics["holdout"]) < _score_tuple(reference_fit.split_metrics["holdout"])),
        "best_validation_beats_carry_forward": bool(_score_tuple(best_fit.split_metrics["validation"]) < _score_tuple(baselines["carry_forward"]["validation"])),
        "best_holdout_beats_carry_forward": bool(_score_tuple(best_fit.split_metrics["holdout"]) < _score_tuple(baselines["carry_forward"]["holdout"])),
        "promote_bundle_branch": bool(_score_tuple(best_fit.split_metrics["validation"]) < _score_tuple(reference_fit.split_metrics["validation"]) and _score_tuple(best_fit.split_metrics["holdout"]) < _score_tuple(reference_fit.split_metrics["holdout"])),
        "paper_figure_archive": paper_archive,
    }

    experiment_spec = {
        "variant": "evidence-to-model-loop",
        "goal": "Fix the promoted strict diagnosis branch as the reference, keep downstream care fixed, and search only incidence and U_to_D direct Phase 2 bundles under the blocked-time contract.",
        "source_run_id": ctx.source_run_id,
        "reference_diag02a_run": str(_latest_diag02a_experiment_dir()) if _latest_diag02a_experiment_dir() is not None else None,
        "reference_candidate_id": str(reference_candidate.candidate_id),
        "mutation_unit": "incidence and U_to_D direct bundle subsets only",
        "blocked_time_contract": contract_payload,
        "artifacts": {
            "bundle_search_contract": str(contract_path),
            "bundle_search_frontier": str(frontier_path),
            "bundle_search_traces": str(trace_path),
            "bundle_search_evaluation": str(evaluation_path),
            "bundle_search_model_comparison": str(comparison_path),
            "bundle_search_prediction_rows": str(prediction_rows_path),
            "bundle_search_dashboard": str(dashboard_path),
        },
    }
    coverage_summary = {
        "historical_quarter_count": int(len(snapshot.historical_quarters)),
        "train_quarter_count": int(np.sum(contract.train_mask)),
        "validation_quarter_count": int(np.sum(contract.validation_mask)),
        "holdout_quarter_count": int(np.sum(contract.holdout_mask)),
        "incidence_bundle_seed_count": int(len(incidence_bundle_rows)),
        "u_to_d_bundle_seed_count": int(len(utd_bundle_rows)),
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            numerical_guard_entry(name="diag02b_bundle_search_eps", role="diag02b_numerical_guard", why_needed="Prevents divide-by-zero and log singularities while searching direct incidence and U_to_D Phase 2 bundle subsets under blocked-time evaluation."),
            {
                "name": "incidence_bundle_seed_count",
                "value": int(len(incidence_bundle_rows)),
                "role": "diag02b_incidence_bundle_search_budget",
                "source_type": "estimated",
                "estimation_data": "Positive-scoring HMBA-00 incidence bundle seeds that overlap the active national feature set",
                "estimation_method": "filter(module_seed_manifest incidence rows by positive score and feature overlap, then cap at four)",
                "uncertainty": "deterministic given the frozen HMBA-00 artifact and active snapshot",
                "why_needed": "Bounds the incidence bundle search to evidence-backed seeds instead of blind subset search.",
            },
            {
                "name": "u_to_d_bundle_seed_count",
                "value": int(len(utd_bundle_rows)),
                "role": "diag02b_u_to_d_bundle_search_budget",
                "source_type": "estimated",
                "estimation_data": "Positive-scoring HMBA-00 U_to_D bundle seeds that overlap the active national feature set",
                "estimation_method": "filter(module_seed_manifest U_to_D rows by positive score and feature overlap, then cap at four)",
                "uncertainty": "deterministic given the frozen HMBA-00 artifact and active snapshot",
                "why_needed": "Bounds the U_to_D bundle search to evidence-backed seeds instead of blind subset search.",
            },
        ],
    )
    return {
        "artifacts": artifacts,
        "bundle_search_contract": contract_payload,
        "frontier_rows": frontier_rows,
        "evaluation": evaluation_payload,
        "decision": decision,
        "paper_archive": paper_archive,
    }


__all__ = ["run_diag_02b"]
