from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS, R11_EVALUATION_METRICS, _generated_at, _sha256
from .r75_bulk_unaids_annual_challenge import R75_RUN_ID, _write_csv
from .r80_public_annual_projection_head import R80_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R82_SCHEMA_VERSION = "phase3_dynamic.r82_quarterly_annual_bridge_gate.v1"
R82_RUN_ID = "p3d-r82-quarterly-annual-bridge-gate-20260507-s00"
R75_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R75_RUN_ID
    / "analysis"
    / "r75_bulk_unaids_annual_challenge_report.json"
)
R80_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R80_RUN_ID
    / "analysis"
    / "r80_public_annual_projection_head_report.json"
)

BRIDGE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "annual_new_infections": {
        "required_quarterly_outputs": ("incident_infections_period",),
        "aggregation_operator": "sum_four_quarters",
        "mechanistic_requirement": "explicit S_eff -> incidence -> U inflow state emitted by the quarterly champion",
    },
    "annual_aids_deaths": {
        "required_quarterly_outputs": ("aids_deaths_period",),
        "aggregation_operator": "sum_four_quarters",
        "mechanistic_requirement": "explicit AIDS mortality/removal channel emitted by the quarterly champion",
    },
    "estimated_plhiv": {
        "required_quarterly_outputs": ("estimated_plhiv",),
        "aggregation_operator": "q4_stock",
        "mechanistic_requirement": "explicit total PLHIV stock, including undiagnosed and diagnosed states, emitted by the quarterly champion",
    },
}


def _bridge_rows(
    *,
    quarterly_outputs: tuple[str, ...] = R11_EVALUATION_METRICS,
    annual_metrics: tuple[str, ...] = OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS,
) -> list[dict[str, Any]]:
    output_set = {str(metric) for metric in quarterly_outputs}
    rows: list[dict[str, Any]] = []
    for metric in annual_metrics:
        requirement = dict(BRIDGE_REQUIREMENTS.get(metric) or {})
        required = tuple(str(value) for value in requirement.get("required_quarterly_outputs") or ())
        missing = [name for name in required if name not in output_set]
        if not required:
            status = "blocked_no_bridge_rule"
        elif missing:
            status = "blocked_missing_quarterly_outputs"
        else:
            status = "bridge_available"
        rows.append(
            {
                "annual_metric": metric,
                "bridge_status": status,
                "required_quarterly_outputs": list(required),
                "available_required_outputs": [name for name in required if name in output_set],
                "missing_quarterly_outputs": missing,
                "aggregation_operator": str(requirement.get("aggregation_operator") or ""),
                "mechanistic_requirement": str(requirement.get("mechanistic_requirement") or ""),
            }
        )
    return rows


def _annual_head_status_rows(r75: dict[str, Any]) -> list[dict[str, Any]]:
    family = str(dict(r75.get("bulk_unaids_annual_gate") or {}).get("best_candidate_family") or "")
    rows: list[dict[str, Any]] = []
    for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
        score_count = sum(
            1
            for row in list(r75.get("score_rows") or [])
            if str(row.get("candidate_family") or "") == family and str(row.get("metric_name") or "") == metric
        )
        rows.append(
            {
                "annual_metric": metric,
                "annual_head_family": family,
                "annual_head_score_count": int(score_count),
                "annual_head_is_quarterly_bridge": False,
                "reason": "R75 annual weak-measurement head is train-origin annual-only; it does not prove the quarterly cascade emitted this annual process.",
            }
        )
    return rows


def _gate(bridge_rows: list[dict[str, Any]], r80: dict[str, Any]) -> dict[str, Any]:
    r80_status = str(dict(r80.get("public_annual_projection_gate") or {}).get("status") or "")
    blocked = [row for row in bridge_rows if str(row.get("bridge_status") or "") != "bridge_available"]
    blockers: list[str] = []
    if r80_status != "public_annual_projection_head_ready":
        blockers.append("public_annual_projection_head_not_ready")
    for row in blocked:
        blockers.append(f"{row.get('annual_metric')}_{row.get('bridge_status')}")
    return {
        "status": "quarterly_annual_bridge_ready" if not blockers else "quarterly_annual_bridge_blocked",
        "blockers": blockers,
        "bridge_available_metric_count": len(bridge_rows) - len(blocked),
        "bridge_blocked_metric_count": len(blocked),
        "annual_metric_count": len(bridge_rows),
        "r80_projection_gate_status": r80_status,
        "quarterly_output_scope": list(R11_EVALUATION_METRICS),
        "contract": (
            "R82 checks whether public annual targets can be evaluated from quarterly Phase 3 champion outputs "
            "without annual-target leakage. Annual weak-measurement heads do not count as quarterly-mechanistic "
            "bridges. A bridge is ready only when the quarterly champion emits the required incidence, mortality, "
            "and total PLHIV state outputs directly."
        ),
    }


def run_r82_quarterly_annual_bridge_gate(
    *,
    run_id: str = R82_RUN_ID,
    r75_report_path: Path | None = None,
    r80_report_path: Path | None = None,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r75_path = R75_DEFAULT_REPORT if r75_report_path is None else Path(r75_report_path)
    r80_path = R80_DEFAULT_REPORT if r80_report_path is None else Path(r80_report_path)
    r75 = dict(read_json(r75_path, default={}) or {}) if r75_path.exists() else {}
    r80 = dict(read_json(r80_path, default={}) or {}) if r80_path.exists() else {}
    bridge = _bridge_rows()
    annual_head_rows = _annual_head_status_rows(r75)
    gate = _gate(bridge, r80)
    report_path = analysis_dir / "r82_quarterly_annual_bridge_gate_report.json"
    markdown_path = analysis_dir / "r82_quarterly_annual_bridge_gate_report.md"
    bridge_csv = analysis_dir / "r82_bridge_rows.csv"
    annual_head_csv = analysis_dir / "r82_annual_head_status_rows.csv"
    report = {
        "schema_version": R82_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "quarterly_annual_bridge_gate": gate,
        "bridge_rows": bridge,
        "annual_head_status_rows": annual_head_rows,
        "source_artifacts": {
            "r75": {"path": r75_path.as_posix(), "sha256": _sha256(r75_path) if r75_path.exists() else None},
            "r80": {"path": r80_path.as_posix(), "sha256": _sha256(r80_path) if r80_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "bridge_rows_csv": bridge_csv.as_posix(),
            "annual_head_status_rows_csv": annual_head_csv.as_posix(),
        },
    }
    _write_csv(bridge_csv, bridge)
    _write_csv(annual_head_csv, annual_head_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("quarterly_annual_bridge_gate") or {})
    lines = [
        "# Phase 3 R82 Quarterly-Annual Bridge Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Bridge available metrics: `{gate.get('bridge_available_metric_count')}`",
        f"- Bridge blocked metrics: `{gate.get('bridge_blocked_metric_count')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Bridge Requirements",
        "",
        "| Annual Metric | Status | Required Quarterly Outputs | Missing Outputs |",
        "|---|---|---|---|",
    ]
    for row in report.get("bridge_rows") or []:
        lines.append(
            f"| `{row.get('annual_metric')}` | `{row.get('bridge_status')}` | "
            f"`{', '.join(row.get('required_quarterly_outputs') or [])}` | "
            f"`{', '.join(row.get('missing_quarterly_outputs') or [])}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R82 quarterly-annual bridge gate.")
    parser.add_argument("--run-id", default=R82_RUN_ID)
    parser.add_argument("--r75-report-path", default=None)
    parser.add_argument("--r80-report-path", default=None)
    args = parser.parse_args()
    run_r82_quarterly_annual_bridge_gate(
        run_id=str(args.run_id),
        r75_report_path=None if args.r75_report_path is None else Path(args.r75_report_path),
        r80_report_path=None if args.r80_report_path is None else Path(args.r80_report_path),
    )


if __name__ == "__main__":
    _main()
