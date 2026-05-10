from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS, _finite_float, _generated_at, _sha256
from .r75_bulk_unaids_annual_challenge import _write_csv
from .r77_public_proxy_annual_gate import _comparison_summary_by_fields
from .r78_public_annual_family_expansion import R78_RUN_ID, R78_SELECTED_FAMILY
from .r79_expanded_public_proxy_annual_gate import R79_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R93_SCHEMA_VERSION = "phase3_dynamic.r93_open_public_incumbent_comparator.v1"
R93_RUN_ID = "p3d-r93-open-public-incumbent-comparator-20260510-s00"
R93_FAMILY = "open_public_aem_spectrum_style_annual_incumbent_comparator"
R78_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R78_RUN_ID
    / "analysis"
    / "r78_public_annual_family_expansion_report.json"
)
R79_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R79_RUN_ID
    / "analysis"
    / "r79_expanded_public_proxy_annual_gate_report.json"
)


def _target_counts_by_metric(target_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in target_rows:
        metric = str(row.get("metric_name") or "")
        if metric:
            counts[metric] += 1
    return {metric: int(counts.get(metric, 0)) for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS}


def _selected_family_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        (dict(row) for row in rows if str(row.get("candidate_family") or "") == R78_SELECTED_FAMILY),
        {},
    )


def _validation_role_leakage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("observation_role") or "") != "validation_only"
        or str(row.get("allowed_use") or "") != "validation_only"
    ]


def _incumbent_selection_summary(selection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in selection_rows:
        selected = dict(row.get("selected_metric_families") or {})
        for metric, family in selected.items():
            counts[(str(metric), str(family))] += 1
    return [
        {
            "metric_name": metric,
            "selected_public_family": family,
            "selection_count": int(count),
        }
        for (metric, family), count in sorted(counts.items())
    ]


def _gate(r78: dict[str, Any], r79: dict[str, Any]) -> dict[str, Any]:
    r78_gate = dict(r78.get("expanded_public_annual_gate") or {})
    r79_gate = dict(r79.get("expanded_public_proxy_annual_gate") or {})
    target_counts = _target_counts_by_metric(list(r78.get("target_rows") or []))
    selected_row = _selected_family_row(list(r78.get("family_rows") or []))
    selected_mean = _finite_float(selected_row.get("candidate_mean_norm_error"))
    selected_coverage = _finite_float(selected_row.get("candidate_interval_coverage"))
    model_mean = _finite_float(r79_gate.get("model_mean_norm_error"))
    incumbent_mean = _finite_float(r79_gate.get("public_proxy_mean_norm_error"))
    model_coverage = _finite_float(r79_gate.get("model_interval_coverage"))
    incumbent_coverage = _finite_float(r79_gate.get("public_proxy_interval_coverage"))
    blockers: list[str] = []
    if str(r78_gate.get("status") or "") != "expanded_public_annual_comparator_promoted":
        blockers.append("expanded_public_incumbent_not_promoted")
    missing_metrics = [metric for metric, count in target_counts.items() if int(count) <= 0]
    if missing_metrics:
        blockers.append("public_target_scope_incomplete")
    selected_scores = [
        dict(row)
        for row in list(r78.get("score_rows") or [])
        if str(row.get("candidate_family") or "") == R78_SELECTED_FAMILY
    ]
    if not selected_scores:
        blockers.append("incumbent_selected_scores_absent")
    if _validation_role_leakage_rows(selected_scores):
        blockers.append("incumbent_validation_role_leakage")
    if int(r79_gate.get("score_row_count") or 0) <= 0:
        blockers.append("model_incumbent_matched_scores_absent")
    r79_status = str(r79_gate.get("status") or "")
    if blockers:
        status = "public_incumbent_comparator_blocked"
        annual_superiority_status = "not_evaluable"
    elif r79_status == "annual_model_beats_expanded_public_proxy":
        status = "public_incumbent_comparator_ready_model_beats_incumbent"
        annual_superiority_status = "allowed_against_open_public_incumbent"
    else:
        status = "public_incumbent_comparator_ready_model_blocked"
        annual_superiority_status = "blocked_by_open_public_incumbent"
    return {
        "status": status,
        "blockers": blockers,
        "annual_superiority_status": annual_superiority_status,
        "incumbent_family": R78_SELECTED_FAMILY,
        "target_counts_by_metric": target_counts,
        "required_metrics": list(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "incumbent_mean_norm_error": selected_mean,
        "incumbent_interval_coverage": selected_coverage,
        "matched_model_family": r79_gate.get("model_family"),
        "matched_model_mean_norm_error": model_mean,
        "matched_incumbent_mean_norm_error": incumbent_mean,
        "matched_model_minus_incumbent_mean_norm_error": None
        if model_mean is None or incumbent_mean is None
        else float(model_mean - incumbent_mean),
        "matched_model_interval_coverage": model_coverage,
        "matched_incumbent_interval_coverage": incumbent_coverage,
        "r78_status": r78_gate.get("status"),
        "r79_status": r79_status,
        "contract": (
            "R93 freezes the promoted R78 public-domain annual proxy as the open AEM/Spectrum-style incumbent "
            "comparator for public annual incidence, AIDS deaths, and PLHIV targets. This is not official "
            "AEM/Spectrum output. Broad annual superiority is allowed only if the Phase 3 annual head beats the "
            "incumbent on matched blocked-time R79 scores without role leakage."
        ),
    }


def run_r93_open_public_incumbent_comparator(
    *,
    run_id: str = R93_RUN_ID,
    r78_report_path: Path | None = None,
    r79_report_path: Path | None = None,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r78_path = R78_DEFAULT_REPORT if r78_report_path is None else Path(r78_report_path)
    r79_path = R79_DEFAULT_REPORT if r79_report_path is None else Path(r79_report_path)
    r78 = dict(read_json(r78_path, default={}) or {}) if r78_path.exists() else {}
    r79 = dict(read_json(r79_path, default={}) or {}) if r79_path.exists() else {}
    gate = _gate(r78, r79)
    incumbent_metric_rows = [
        dict(row)
        for row in list(r78.get("metric_rows") or [])
        if str(row.get("candidate_family") or "") == R78_SELECTED_FAMILY
    ]
    incumbent_horizon_rows = [
        dict(row)
        for row in list(r78.get("horizon_rows") or [])
        if str(row.get("candidate_family") or "") == R78_SELECTED_FAMILY
    ]
    matched_metric_rows = list(r79.get("metric_rows") or [])
    matched_horizon_rows = list(r79.get("horizon_rows") or [])
    selection_summary_rows = _incumbent_selection_summary(list(r78.get("selection_rows") or []))
    matched_rows = list(r79.get("matched_score_rows") or [])
    matched_summary_rows = _comparison_summary_by_fields(matched_rows, group_fields=("metric_name", "horizon_years"))
    report_path = analysis_dir / "r93_open_public_incumbent_comparator_report.json"
    markdown_path = analysis_dir / "r93_open_public_incumbent_comparator_report.md"
    paths = {
        "incumbent_metric_rows_csv": analysis_dir / "r93_incumbent_metric_rows.csv",
        "incumbent_horizon_rows_csv": analysis_dir / "r93_incumbent_horizon_rows.csv",
        "matched_metric_rows_csv": analysis_dir / "r93_matched_metric_rows.csv",
        "matched_horizon_rows_csv": analysis_dir / "r93_matched_horizon_rows.csv",
        "selection_summary_rows_csv": analysis_dir / "r93_selection_summary_rows.csv",
        "matched_summary_rows_csv": analysis_dir / "r93_matched_summary_rows.csv",
    }
    report = {
        "schema_version": R93_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "candidate_family": R93_FAMILY,
        "open_public_incumbent_comparator_gate": gate,
        "incumbent_metric_rows": incumbent_metric_rows,
        "incumbent_horizon_rows": incumbent_horizon_rows,
        "matched_metric_rows": matched_metric_rows,
        "matched_horizon_rows": matched_horizon_rows,
        "selection_summary_rows": selection_summary_rows,
        "matched_summary_rows": matched_summary_rows,
        "source_artifacts": {
            "r78": {"path": r78_path.as_posix(), "sha256": _sha256(r78_path) if r78_path.exists() else None},
            "r79": {"path": r79_path.as_posix(), "sha256": _sha256(r79_path) if r79_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            **{key: value.as_posix() for key, value in paths.items()},
        },
    }
    _write_csv(paths["incumbent_metric_rows_csv"], incumbent_metric_rows)
    _write_csv(paths["incumbent_horizon_rows_csv"], incumbent_horizon_rows)
    _write_csv(paths["matched_metric_rows_csv"], matched_metric_rows)
    _write_csv(paths["matched_horizon_rows_csv"], matched_horizon_rows)
    _write_csv(paths["selection_summary_rows_csv"], selection_summary_rows)
    _write_csv(paths["matched_summary_rows_csv"], matched_summary_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("open_public_incumbent_comparator_gate") or {})
    lines = [
        "# Phase 3 R93 Open Public Incumbent Comparator",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Annual superiority status: `{gate.get('annual_superiority_status')}`",
        f"- Incumbent family: `{gate.get('incumbent_family')}`",
        f"- Incumbent mean normalized error: `{gate.get('incumbent_mean_norm_error')}`",
        f"- Matched model mean normalized error: `{gate.get('matched_model_mean_norm_error')}`",
        f"- Matched incumbent mean normalized error: `{gate.get('matched_incumbent_mean_norm_error')}`",
        f"- Delta model minus incumbent: `{gate.get('matched_model_minus_incumbent_mean_norm_error')}`",
        f"- Matched model interval coverage: `{gate.get('matched_model_interval_coverage')}`",
        f"- Matched incumbent interval coverage: `{gate.get('matched_incumbent_interval_coverage')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Incumbent Metric Scores",
        "",
        "| Metric | Entries | Mean Error | P90 Error | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("incumbent_metric_rows") or []:
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('entry_count')}` | "
            f"`{row.get('candidate_mean_norm_error')}` | `{row.get('candidate_p90_norm_error')}` | "
            f"`{row.get('candidate_interval_coverage')}` |"
        )
    lines.extend(
        [
            "",
            "## Matched Model Versus Incumbent",
            "",
            "| Metric | Entries | Model Mean | Incumbent Mean | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report.get("matched_metric_rows") or []:
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('entry_count')}` | "
            f"`{row.get('model_mean_norm_error')}` | `{row.get('public_proxy_mean_norm_error')}` | "
            f"`{row.get('model_minus_public_proxy_mean_norm_error')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R93 open public incumbent comparator.")
    parser.add_argument("--run-id", default=R93_RUN_ID)
    parser.add_argument("--r78-report-path", default=None)
    parser.add_argument("--r79-report-path", default=None)
    args = parser.parse_args()
    run_r93_open_public_incumbent_comparator(
        run_id=str(args.run_id),
        r78_report_path=None if args.r78_report_path is None else Path(args.r78_report_path),
        r79_report_path=None if args.r79_report_path is None else Path(args.r79_report_path),
    )


if __name__ == "__main__":
    _main()
