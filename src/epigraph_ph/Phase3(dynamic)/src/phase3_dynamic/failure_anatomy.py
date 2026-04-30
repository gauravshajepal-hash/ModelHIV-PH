from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import BlockedTimeDataset, build_blocked_time_dataset, build_observation_rows, default_epigraph_root, sandbox_repo_root
from .metrics import PRIMARY_METRICS, quarter_sort_key
from .model import (
    DampingConfig,
    DirectPriorConfig,
    DynamicBaselineConfig,
    HiddenDriverConfig,
    IncidenceFlowConfig,
    ObservationModelConfig,
    ShockConfig,
    _simulate_sequence,
    forecast_dynamic_baseline,
)
from .phase2 import build_direct_prior_features, build_hidden_driver_features, load_phase2_structural_inputs
from .priors import PHASE2_TRANSITION_PRIOR_MAP
from .runtime import ensure_dir, read_json, write_json

SUPPLEMENTAL_METRICS: tuple[str, ...] = ("tested_for_viral_load", "virally_suppressed")


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _metric_triplet_summary(
    holdout_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    calibrated_rows: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, float | None]:
    observed_values = [float(row[metric_name]) for row in holdout_rows if row.get(metric_name) is not None]
    raw_values = [float(row[metric_name]) for row in raw_rows if row.get(metric_name) is not None]
    calibrated_values = [float(row[metric_name]) for row in calibrated_rows if row.get(metric_name) is not None]
    paired_raw = [
        abs(float(raw_row[metric_name]) - float(observed_row[metric_name]))
        for raw_row, observed_row in zip(raw_rows, holdout_rows)
        if raw_row.get(metric_name) is not None and observed_row.get(metric_name) is not None
    ]
    paired_calibrated = [
        abs(float(calibrated_row[metric_name]) - float(observed_row[metric_name]))
        for calibrated_row, observed_row in zip(calibrated_rows, holdout_rows)
        if calibrated_row.get(metric_name) is not None and observed_row.get(metric_name) is not None
    ]
    return {
        "observed_mean": _mean_or_none(observed_values),
        "raw_mean": _mean_or_none(raw_values),
        "calibrated_mean": _mean_or_none(calibrated_values),
        "raw_mae": _mean_or_none(paired_raw),
        "calibrated_mae": _mean_or_none(paired_calibrated),
    }


def _train_mass_diagnostics(dataset: BlockedTimeDataset, candidate_result: dict[str, Any]) -> dict[str, Any]:
    train_rows = sorted(list(dataset.train_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not train_rows or not dataset.train_state_rows:
        return {}
    raw_train = _simulate_sequence(
        dict(dataset.train_state_rows[0]["state_values"]),
        train_rows[1:],
        candidate_result["hazard_paths"]["train_hazard_map"],
        incidence_inflow_map=dict((candidate_result.get("incidence_paths") or {}).get("train_incidence_inflow_map") or {}),
        incidence_hazard_map=dict((candidate_result.get("incidence_paths") or {}).get("train_incidence_hazard_map") or {}),
        population_denominator_map=dict((candidate_result.get("incidence_paths") or {}).get("train_population_denominator_map") or {}),
        attrition_outflow_map=dict((candidate_result.get("incidence_paths") or {}).get("train_attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict((candidate_result.get("incidence_paths") or {}).get("train_state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=dict((candidate_result.get("incidence_paths") or {}).get("train_exit_channel_state_outflow_map") or {}),
    )
    first_state = dict(dataset.train_state_rows[0]["state_values"])
    last_state = dict(dataset.train_state_rows[-1]["state_values"])
    raw_last_state = dict(raw_train["trajectory_rows"][-1]["state_values"]) if raw_train["trajectory_rows"] else dict(first_state)
    raw_train_prediction_rows = list(raw_train["prediction_rows"])
    total_inferred_inflow = float(
        sum(float(((row.get("stock_balance") or {}).get("incidence_inflow")) or 0.0) for row in raw_train.get("trajectory_rows") or [])
    )
    total_inferred_attrition = float(
        sum(float(((row.get("stock_balance") or {}).get("attrition_outflow")) or 0.0) for row in raw_train.get("trajectory_rows") or [])
    )
    observed_last_total = float(sum(float(value) for value in last_state.values()))
    raw_last_total = float(sum(float(value) for value in raw_last_state.values()))
    return {
        "first_train_quarter": str(dataset.train_state_rows[0]["quarter"]),
        "last_train_quarter": str(dataset.train_state_rows[-1]["quarter"]),
        "first_train_state_total": float(sum(float(value) for value in first_state.values())),
        "last_train_state_total": observed_last_total,
        "raw_train_last_state_total": raw_last_total,
        "train_mass_gap": float(observed_last_total - raw_last_total),
        "total_inferred_inflow": total_inferred_inflow,
        "total_inferred_attrition": total_inferred_attrition,
        "raw_train_metric_ranges": {
            metric_name: {
                "min": float(min(float(row.get(metric_name) or 0.0) for row in raw_train_prediction_rows)),
                "max": float(max(float(row.get(metric_name) or 0.0) for row in raw_train_prediction_rows)),
            }
            for metric_name in PRIMARY_METRICS
            if raw_train_prediction_rows
        },
        "observed_train_metric_ranges": {
            metric_name: {
                "min": float(min(float(row.get(metric_name) or 0.0) for row in train_rows[1:])),
                "max": float(max(float(row.get(metric_name) or 0.0) for row in train_rows[1:])),
            }
            for metric_name in PRIMARY_METRICS
            if len(train_rows) > 1
        },
    }


def _config_from_report(report_config: dict[str, Any]) -> dict[str, Any]:
    dynamic_cfg = DynamicBaselineConfig(**dict(report_config.get("dynamic_cfg") or {}))
    incidence_payload = report_config.get("incidence_cfg")
    observation_payload = report_config.get("observation_cfg")
    shock_payload = report_config.get("shock_cfg")
    damping_payload = report_config.get("damping_cfg")
    prior_payload = report_config.get("prior_cfg")
    hidden_payload = report_config.get("hidden_cfg")
    return {
        "dynamic_cfg": dynamic_cfg,
        "incidence_cfg": None if incidence_payload is None else IncidenceFlowConfig(**dict(incidence_payload)),
        "observation_cfg": None if observation_payload is None else ObservationModelConfig(**dict(observation_payload)),
        "shock_cfg": None if shock_payload is None else ShockConfig(**dict(shock_payload)),
        "damping_cfg": None if damping_payload is None else DampingConfig(**dict(damping_payload)),
        "prior_cfg": None if prior_payload is None else DirectPriorConfig(**dict(prior_payload)),
        "hidden_cfg": None if hidden_payload is None else HiddenDriverConfig(**dict(hidden_payload)),
    }


def _flatten_scored_entries(best_candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in best_candidate_rows:
        scoring = dict(row.get("scoring_details") or {})
        holdout_rows = list(scoring.get("holdout_rows") or [])
        candidate_rows = list(scoring.get("candidate_prediction_rows") or [])
        carry_forward_rows = list(scoring.get("carry_forward_prediction_rows") or [])
        metric_scales = dict(scoring.get("metric_scales") or {})
        for observed_row, candidate_row, carry_forward_row in zip(holdout_rows, candidate_rows, carry_forward_rows):
            metric_provenance = dict(observed_row.get("metric_provenance") or {})
            for metric_name in PRIMARY_METRICS:
                observed_value = observed_row.get(metric_name)
                candidate_value = candidate_row.get(metric_name)
                carry_forward_value = carry_forward_row.get(metric_name)
                if observed_value is None or candidate_value is None or carry_forward_value is None:
                    continue
                scale = float(metric_scales.get(metric_name) or 0.0)
                candidate_abs_error = abs(float(candidate_value) - float(observed_value))
                carry_forward_abs_error = abs(float(carry_forward_value) - float(observed_value))
                entries.append(
                    {
                        "quarter": str(observed_row.get("quarter") or ""),
                        "year": int(str(observed_row.get("quarter") or "0-Q0").split("-Q", 1)[0]),
                        "metric_name": metric_name,
                        "observed_value": float(observed_value),
                        "candidate_value": float(candidate_value),
                        "carry_forward_value": float(carry_forward_value),
                        "candidate_abs_error": float(candidate_abs_error),
                        "carry_forward_abs_error": float(carry_forward_abs_error),
                        "candidate_norm_error": None if scale <= 0.0 else float(candidate_abs_error / scale),
                        "carry_forward_norm_error": None if scale <= 0.0 else float(carry_forward_abs_error / scale),
                        "scale": None if scale <= 0.0 else float(scale),
                        "tier": str((metric_provenance.get(metric_name) or {}).get("tier") or ""),
                        "source_id": str((metric_provenance.get(metric_name) or {}).get("source_id") or ""),
                    }
                )
    return entries


def summarize_scored_entries(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    for metric_name in sorted({str(entry["metric_name"]) for entry in entries}):
        metric_entries = [entry for entry in entries if str(entry["metric_name"]) == metric_name]
        norm_entries = [entry for entry in metric_entries if entry.get("candidate_norm_error") is not None and entry.get("carry_forward_norm_error") is not None]
        metric_rows.append(
            {
                "metric_name": metric_name,
                "entry_count": int(len(metric_entries)),
                "candidate_mae": float(np.mean([float(entry["candidate_abs_error"]) for entry in metric_entries])),
                "carry_forward_mae": float(np.mean([float(entry["carry_forward_abs_error"]) for entry in metric_entries])),
                "candidate_norm_mae": None if not norm_entries else float(np.mean([float(entry["candidate_norm_error"]) for entry in norm_entries])),
                "carry_forward_norm_mae": None if not norm_entries else float(np.mean([float(entry["carry_forward_norm_error"]) for entry in norm_entries])),
            }
        )
    year_rows: list[dict[str, Any]] = []
    for year in sorted({int(entry["year"]) for entry in entries}):
        year_entries = [entry for entry in entries if int(entry["year"]) == year]
        norm_entries = [entry for entry in year_entries if entry.get("candidate_norm_error") is not None and entry.get("carry_forward_norm_error") is not None]
        year_rows.append(
            {
                "year": int(year),
                "entry_count": int(len(year_entries)),
                "candidate_mae": float(np.mean([float(entry["candidate_abs_error"]) for entry in year_entries])),
                "carry_forward_mae": float(np.mean([float(entry["carry_forward_abs_error"]) for entry in year_entries])),
                "candidate_norm_mae": None if not norm_entries else float(np.mean([float(entry["candidate_norm_error"]) for entry in norm_entries])),
                "carry_forward_norm_mae": None if not norm_entries else float(np.mean([float(entry["carry_forward_norm_error"]) for entry in norm_entries])),
            }
        )
    worst_rows = sorted(
        entries,
        key=lambda entry: float(entry["candidate_abs_error"]) - float(entry["carry_forward_abs_error"]),
        reverse=True,
    )[:12]
    return {"metric_rows": metric_rows, "year_rows": year_rows, "worst_rows": worst_rows}


def _replay_top_problem_years(
    *,
    epigraph_root: Path,
    source_run_id: str,
    phase2_source_run_id: str | None = None,
    report_config: dict[str, Any],
    holdout_years: list[int],
) -> list[dict[str, Any]]:
    if not holdout_years:
        return []
    spec = _config_from_report(report_config)
    observation_rows = build_observation_rows(epigraph_root, source_run_id)
    structural_inputs = None
    direct_prior_features = None
    hidden_driver_features = None
    if spec["prior_cfg"] is not None or spec["hidden_cfg"] is not None:
        structural_inputs = load_phase2_structural_inputs(epigraph_root, phase2_source_run_id or source_run_id)
        if spec["prior_cfg"] is not None:
            direct_prior_features = build_direct_prior_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
        if spec["hidden_cfg"] is not None:
            hidden_driver_features = build_hidden_driver_features(structural_inputs, PHASE2_TRANSITION_PRIOR_MAP)
    replay_rows: list[dict[str, Any]] = []
    for holdout_year in holdout_years:
        dataset = build_blocked_time_dataset(observation_rows, [int(holdout_year)])
        candidate_result = forecast_dynamic_baseline(
            dataset,
            spec["dynamic_cfg"],
            incidence_cfg=spec["incidence_cfg"],
            observation_cfg=spec["observation_cfg"],
            shock_cfg=spec["shock_cfg"],
            damping_cfg=spec["damping_cfg"],
            structural_inputs=structural_inputs,
            direct_prior_features=direct_prior_features if spec["prior_cfg"] is not None else None,
            prior_cfg=spec["prior_cfg"],
            hidden_driver_features=hidden_driver_features if spec["hidden_cfg"] is not None else None,
            hidden_cfg=spec["hidden_cfg"],
        )
        replay_rows.append(
            {
                "holdout_year": int(holdout_year),
                "candidate_mae": float(candidate_result["mae"]),
                "candidate_smape": float(candidate_result["smape"]),
                "raw_vs_calibrated": {
                    metric_name: _metric_triplet_summary(
                        list(dataset.holdout_rows),
                        list(candidate_result.get("raw_prediction_rows") or []),
                        list(candidate_result.get("prediction_rows") or []),
                        metric_name,
                    )
                    for metric_name in (*PRIMARY_METRICS, *SUPPLEMENTAL_METRICS)
                },
                "observation_coefficients": dict((candidate_result.get("observation_model") or {}).get("primary_coefficients") or {}),
                "observation_weights": dict((candidate_result.get("observation_model") or {}).get("primary_weights") or {}),
                "share_weights": dict((candidate_result.get("observation_model") or {}).get("share_weights") or {}),
                "incidence_diagnostics": None if candidate_result.get("incidence_paths") is None else dict((candidate_result.get("incidence_paths") or {}).get("diagnostics") or {}),
                "train_mass_diagnostics": _train_mass_diagnostics(dataset, candidate_result),
            }
        )
    return replay_rows


def build_failure_anatomy_payload(*, epigraph_root: Path, report_paths: list[Path], top_years: int = 3) -> dict[str, Any]:
    payload_rows: list[dict[str, Any]] = []
    for report_path in report_paths:
        report = dict(read_json(report_path, default={}) or {})
        best_candidate = dict(report.get("best_candidate") or {})
        best_candidate_rows = list(best_candidate.get("rows") or [])
        scored_entries = _flatten_scored_entries(best_candidate_rows)
        error_summary = summarize_scored_entries(scored_entries)
        top_problem_years = [
            int(year)
            for year, _score in sorted(
                {
                    int(row["holdout_years"][0]): float(row["candidate"]["mae"])
                    for row in best_candidate_rows
                    if row.get("holdout_years")
                }.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: max(int(top_years), 0)]
        ]
        payload_rows.append(
            {
                "report_path": str(report_path),
                "run_id": str(report.get("run_id") or report_path.parent.parent.name),
                "family_name": str(report.get("family_name") or ""),
                "source_run_id": str(report.get("source_run_id") or ""),
                "decision": str(report.get("decision") or ""),
                "decision_reason": str(report.get("decision_reason") or ""),
                "score_summary": dict(best_candidate.get("score") or {}),
                "phase2_context": dict(report.get("phase2_context") or {}),
                "metric_error_summary": error_summary["metric_rows"],
                "year_error_summary": error_summary["year_rows"],
                "worst_quarter_deltas": error_summary["worst_rows"],
                "problem_year_replays": _replay_top_problem_years(
                    epigraph_root=epigraph_root,
                    source_run_id=str(report.get("source_run_id") or ""),
                    phase2_source_run_id=str((report.get("phase2_context") or {}).get("phase2_source_run_id") or "") or None,
                    report_config=dict(best_candidate.get("config") or {}),
                    holdout_years=top_problem_years,
                ),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "epigraph_root": str(epigraph_root),
        "top_problem_year_count": int(top_years),
        "reports": payload_rows,
    }


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def markdown_failure_anatomy_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase3(dynamic) Failure Anatomy",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
    ]
    for report in list(payload.get("reports") or []):
        lines.extend(
            [
                f"## {report.get('family_name')} :: {report.get('run_id')}",
                "",
                f"- Report: `{report.get('report_path')}`",
                f"- Source run: `{report.get('source_run_id')}`",
                f"- Decision: `{report.get('decision')}`",
                f"- Reason: {report.get('decision_reason')}",
                "",
                "| Score | Candidate | Carry-forward |",
                "|---|---:|---:|",
            ]
        )
        score = dict(report.get("score_summary") or {})
        lines.extend(
            [
                f"| Mean MAE | {_fmt_optional(score.get('candidate_mean_mae'))} | {_fmt_optional(score.get('carry_forward_mean_mae'))} |",
                f"| Worst MAE | {_fmt_optional(score.get('candidate_worst_mae'))} | {_fmt_optional(score.get('carry_forward_worst_mae'))} |",
                "",
                "### Scored Metrics",
                "",
                "| Metric | Entries | Candidate MAE | Carry-forward MAE | Candidate Norm MAE | Carry-forward Norm MAE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in list(report.get("metric_error_summary") or []):
            lines.append(
                f"| `{row['metric_name']}` | {int(row['entry_count'])} | {_fmt_optional(row.get('candidate_mae'))} | {_fmt_optional(row.get('carry_forward_mae'))} | {_fmt_optional(row.get('candidate_norm_mae'))} | {_fmt_optional(row.get('carry_forward_norm_mae'))} |"
            )
        lines.extend(
            [
                "",
                "### Holdout Years",
                "",
                "| Year | Entries | Candidate MAE | Carry-forward MAE | Candidate Norm MAE | Carry-forward Norm MAE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in list(report.get("year_error_summary") or []):
            lines.append(
                f"| {int(row['year'])} | {int(row['entry_count'])} | {_fmt_optional(row.get('candidate_mae'))} | {_fmt_optional(row.get('carry_forward_mae'))} | {_fmt_optional(row.get('candidate_norm_mae'))} | {_fmt_optional(row.get('carry_forward_norm_mae'))} |"
            )
        lines.extend(
            [
                "",
                "### Worst Quarter Deltas",
                "",
                "| Quarter | Metric | Actual | Candidate | Carry-forward | Candidate Minus CF Abs Error | Tier | Source |",
                "|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in list(report.get("worst_quarter_deltas") or []):
            delta = float(row["candidate_abs_error"]) - float(row["carry_forward_abs_error"])
            lines.append(
                f"| `{row['quarter']}` | `{row['metric_name']}` | {float(row['observed_value']):.6f} | {float(row['candidate_value']):.6f} | {float(row['carry_forward_value']):.6f} | {delta:.6f} | `{row.get('tier') or ''}` | `{row.get('source_id') or ''}` |"
            )
        lines.extend(
            [
                "",
                "### Problem-Year Replays",
                "",
            ]
        )
        for replay in list(report.get("problem_year_replays") or []):
            lines.extend(
                [
                    f"#### Holdout {int(replay['holdout_year'])}",
                    "",
                    "| Metric | Observed Mean | Raw Mean | Calibrated Mean | Raw MAE | Calibrated MAE |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for metric_name, summary in dict(replay.get("raw_vs_calibrated") or {}).items():
                lines.append(
                    f"| `{metric_name}` | {_fmt_optional(summary.get('observed_mean'))} | {_fmt_optional(summary.get('raw_mean'))} | {_fmt_optional(summary.get('calibrated_mean'))} | {_fmt_optional(summary.get('raw_mae'))} | {_fmt_optional(summary.get('calibrated_mae'))} |"
                )
            train_mass = dict(replay.get("train_mass_diagnostics") or {})
            lines.extend(
                [
                    "",
                    "| Train Mass Diagnostic | Value |",
                    "|---|---:|",
                    f"| First train state total | {_fmt_optional(train_mass.get('first_train_state_total'))} |",
                    f"| Last train state total | {_fmt_optional(train_mass.get('last_train_state_total'))} |",
                    f"| Raw train last state total | {_fmt_optional(train_mass.get('raw_train_last_state_total'))} |",
                    f"| Train mass gap | {_fmt_optional(train_mass.get('train_mass_gap'))} |",
                    f"| Total inferred inflow | {_fmt_optional(train_mass.get('total_inferred_inflow'))} |",
                    f"| Total inferred attrition | {_fmt_optional(train_mass.get('total_inferred_attrition'))} |",
                    "",
                    "| Observation Coefficient | Value |",
                    "|---|---:|",
                ]
            )
            for metric_name, coeffs in dict(replay.get("observation_coefficients") or {}).items():
                lines.append(f"| `{metric_name}.raw_scale` | {_fmt_optional((coeffs or {}).get('raw_scale'))} |")
                lines.append(f"| `{metric_name}.time_slope` | {_fmt_optional((coeffs or {}).get('time_slope'))} |")
            incidence_diag = dict(replay.get("incidence_diagnostics") or {})
            if incidence_diag:
                lines.extend(
                    [
                        "",
                        "| Incidence Diagnostic | Train Mean | Forecast Mean |",
                        "|---|---:|---:|",
                    ]
                )
                for name, summary in incidence_diag.items():
                    lines.append(
                        f"| `{name}` | {_fmt_optional((summary or {}).get('train_mean'))} | {_fmt_optional((summary or {}).get('forecast_mean'))} |"
                    )
            lines.append("")
    return "\n".join(lines) + "\n"


def write_failure_anatomy_report(
    *,
    report_paths: list[Path],
    output_stem: str = "phase3_failure_anatomy",
    epigraph_root: Path | None = None,
    top_years: int = 3,
) -> dict[str, Any]:
    epigraph_root = Path(epigraph_root) if epigraph_root is not None else default_epigraph_root()
    payload = build_failure_anatomy_payload(epigraph_root=epigraph_root, report_paths=[Path(path) for path in report_paths], top_years=top_years)
    output_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "scientific_audits")
    json_path = output_dir / f"{output_stem}.json"
    md_path = output_dir / f"{output_stem}.md"
    write_json(json_path, payload)
    md_path.write_text(markdown_failure_anatomy_report(payload), encoding="utf-8")
    return payload
