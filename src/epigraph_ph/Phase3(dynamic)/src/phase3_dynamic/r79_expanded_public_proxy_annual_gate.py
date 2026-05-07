from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .r75_bulk_unaids_annual_challenge import R75_RUN_ID, _write_csv
from .r77_public_proxy_annual_gate import (
    _comparison_summary_by_fields,
    _gate as _r77_gate,
    _matched_proxy_rows,
)
from .r78_public_annual_family_expansion import R78_RUN_ID, R78_SELECTED_FAMILY
from .runtime import ensure_dir, read_json, write_json


R79_SCHEMA_VERSION = "phase3_dynamic.r79_expanded_public_proxy_annual_gate.v1"
R79_RUN_ID = "p3d-r79-expanded-public-proxy-annual-gate-20260507-s00"
R75_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R75_RUN_ID
    / "analysis"
    / "r75_bulk_unaids_annual_challenge_report.json"
)
R78_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R78_RUN_ID
    / "analysis"
    / "r78_public_annual_family_expansion_report.json"
)


def _expanded_gate(matched_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate = _r77_gate(matched_rows, family_rows)
    status = str(gate.get("status") or "")
    gate["status"] = "annual_model_beats_expanded_public_proxy" if status == "annual_model_beats_public_proxy" else "annual_model_blocked_by_expanded_public_proxy"
    gate["contract"] = (
        "R79 replaces the weaker R76 annual proxy with the promoted R78 expanded public-domain annual "
        "incumbent. Phase 3 annual superiority claims are blocked unless the model annual head beats "
        "the R78 selected proxy on matched held-out UNAIDS-style annual targets."
    )
    return gate


def run_r79_expanded_public_proxy_annual_gate(
    *,
    run_id: str = R79_RUN_ID,
    r75_report_path: Path | None = None,
    r78_report_path: Path | None = None,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r75_path = R75_DEFAULT_REPORT if r75_report_path is None else Path(r75_report_path)
    r78_path = R78_DEFAULT_REPORT if r78_report_path is None else Path(r78_report_path)
    r75 = dict(read_json(r75_path, default={}) or {}) if r75_path.exists() else {}
    r78 = dict(read_json(r78_path, default={}) or {}) if r78_path.exists() else {}
    matched_rows = _matched_proxy_rows(r75, r78, r76_family=R78_SELECTED_FAMILY)
    metric_rows = _comparison_summary_by_fields(matched_rows, group_fields=("model_family", "public_proxy_family", "metric_name"))
    horizon_rows = _comparison_summary_by_fields(matched_rows, group_fields=("model_family", "public_proxy_family", "horizon_years"))
    family_rows = _comparison_summary_by_fields(matched_rows, group_fields=("model_family", "public_proxy_family"))
    gate = _expanded_gate(matched_rows, family_rows)
    report_path = analysis_dir / "r79_expanded_public_proxy_annual_gate_report.json"
    markdown_path = analysis_dir / "r79_expanded_public_proxy_annual_gate_report.md"
    matched_csv = analysis_dir / "r79_matched_score_rows.csv"
    family_csv = analysis_dir / "r79_family_rows.csv"
    metric_csv = analysis_dir / "r79_metric_rows.csv"
    horizon_csv = analysis_dir / "r79_horizon_rows.csv"
    report = {
        "schema_version": R79_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "expanded_public_proxy_annual_gate": gate,
        "matched_score_rows": matched_rows,
        "family_rows": family_rows,
        "metric_rows": metric_rows,
        "horizon_rows": horizon_rows,
        "source_artifacts": {
            "r75": {"path": r75_path.as_posix(), "sha256": _sha256(r75_path) if r75_path.exists() else None},
            "r78": {"path": r78_path.as_posix(), "sha256": _sha256(r78_path) if r78_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "matched_score_rows_csv": matched_csv.as_posix(),
            "family_rows_csv": family_csv.as_posix(),
            "metric_rows_csv": metric_csv.as_posix(),
            "horizon_rows_csv": horizon_csv.as_posix(),
        },
    }
    _write_csv(matched_csv, matched_rows)
    _write_csv(family_csv, family_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(horizon_csv, horizon_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("expanded_public_proxy_annual_gate") or {})
    lines = [
        "# Phase 3 R79 Expanded Public-Proxy Annual Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Model family: `{gate.get('model_family')}`",
        f"- Public proxy family: `{gate.get('public_proxy_family')}`",
        f"- Model mean normalized error: `{gate.get('model_mean_norm_error')}`",
        f"- Expanded public proxy mean normalized error: `{gate.get('public_proxy_mean_norm_error')}`",
        f"- Delta model minus expanded proxy: `{gate.get('model_minus_public_proxy_mean_norm_error')}`",
        f"- Model interval coverage: `{gate.get('model_interval_coverage')}`",
        f"- Expanded public proxy interval coverage: `{gate.get('public_proxy_interval_coverage')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Metric Comparison",
        "",
        "| Metric | Entries | Model mean | Expanded proxy mean | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("metric_rows") or []:
        lines.append(
            f"| `{row.get('metric_name')}` | {int(row.get('entry_count') or 0)} | "
            f"{float(row.get('model_mean_norm_error') or 0.0):.6f} | "
            f"{float(row.get('public_proxy_mean_norm_error') or 0.0):.6f} | "
            f"{float(row.get('model_minus_public_proxy_mean_norm_error') or 0.0):.6f} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R79 expanded public-proxy annual gate.")
    parser.add_argument("--run-id", default=R79_RUN_ID)
    parser.add_argument("--r75-report-path", default=None)
    parser.add_argument("--r78-report-path", default=None)
    args = parser.parse_args()
    run_r79_expanded_public_proxy_annual_gate(
        run_id=str(args.run_id),
        r75_report_path=None if args.r75_report_path is None else Path(args.r75_report_path),
        r78_report_path=None if args.r78_report_path is None else Path(args.r78_report_path),
    )


if __name__ == "__main__":
    _main()
