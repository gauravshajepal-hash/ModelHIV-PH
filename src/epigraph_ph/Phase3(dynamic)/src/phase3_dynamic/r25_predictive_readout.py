from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    OFFICIAL_ANNUAL_CHALLENGE_METRICS,
    R10_COMPARABLE_METRICS,
    R11_EVALUATION_METRICS,
    _build_r10_horizon_replay_report,
    _finite_float,
    _generated_at,
)
from .r13_priority_experiments import _r13_annual_spec_report, _r13_model_spec_report
from .runtime import ensure_dir, write_json


R25_SCHEMA_VERSION = "phase3_dynamic.r25_predictive_readout_evaluation.v1"
R25_FAMILY = "r25_r19_predictive_endpoint_readout"
R19_FAMILY = "r19_joint_service_cascade_process"


def _r25_specs(*, include_annual_gate: bool = False) -> list[dict[str, Any]]:
    back_half_flow_metrics = (
        "alive_on_art",
        "tested_for_viral_load",
        "virally_suppressed",
        "new_diagnosed_cases_period",
    )
    specs = [
        {
            "experiment_id": "R25-001",
            "priority": 1,
            "layer": "annual_official_measurement",
            "title": "R25 conserved annual challenge",
            "family": "official_annual_challenge_gate",
            "candidate_family": R25_FAMILY,
            "row_scope": "official_annual_q4",
            "metrics": tuple(OFFICIAL_ANNUAL_CHALLENGE_METRICS),
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "The predictive endpoint head must preserve the official annual AEM/Spectrum-style conserved-head gate.",
        },
        {
            "experiment_id": "R25-002",
            "priority": 2,
            "layer": "program_route",
            "title": "R25 predictive endpoint readout on program route",
            "family": R25_FAMILY,
            "row_scope": "program",
            "metrics": back_half_flow_metrics,
            "horizons": (3, 5),
            "r10_required": True,
            "hypothesis": "A separate predictive endpoint head constrained by the R19 research contract should close the program-route R10-scope gap without weakening the full service trajectory.",
        },
        {
            "experiment_id": "R25-003",
            "priority": 3,
            "layer": "full_publication_sentinel",
            "title": "R25 predictive endpoint readout full sentinel",
            "family": R25_FAMILY,
            "row_scope": "all",
            "metrics": tuple(R11_EVALUATION_METRICS),
            "horizons": (1, 3, 5),
            "r10_required": True,
            "hypothesis": "The predictive endpoint head should beat matched R10 across h1/h3/h5 while preserving the Phase 3 full-cascade scoring scope.",
        },
    ]
    return specs if include_annual_gate else [spec for spec in specs if str(spec.get("family") or "") != "official_annual_challenge_gate"]


def _flat_result(row: dict[str, Any]) -> dict[str, Any]:
    candidate = _finite_float(row.get("candidate_mean_mae"))
    carry = _finite_float(row.get("carry_forward_mean_mae"))
    r10_candidate = _finite_float(row.get("r10_comparable_candidate_mean_mae"))
    r10_reference = _finite_float(row.get("r10_reference_mae"))
    return {
        "experiment_id": str(row.get("experiment_id") or ""),
        "title": str(row.get("title") or ""),
        "family": str(row.get("family") or ""),
        "candidate_family": "" if row.get("candidate_family") is None else str(row.get("candidate_family")),
        "row_scope": str(row.get("row_scope") or ""),
        "candidate_mean_mae": candidate,
        "carry_forward_mean_mae": carry,
        "candidate_minus_carry_forward": None if candidate is None or carry is None else float(candidate - carry),
        "r10_comparable_candidate_mean_mae": r10_candidate,
        "r10_reference_mae": r10_reference,
        "candidate_minus_r10": None if r10_candidate is None or r10_reference is None else float(r10_candidate - r10_reference),
        "decision": str(row.get("decision") or ""),
        "blockers": ";".join(str(item) for item in list(row.get("blockers") or [])),
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows = [_flat_result(row) for row in results]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    results = list(report.get("results") or [])
    lines = [
        "# Phase 3 R25 Predictive Endpoint Readout Evaluation",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Gate Summary",
        "",
        "| Experiment | Scope | Candidate | Carry | Candidate minus carry | R10-scope | Matched R10 | Candidate minus R10 | Decision | Blockers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        flat = _flat_result(row)
        lines.append(
            f"| `{flat['experiment_id']}` | {flat['row_scope']} | {_format_float(flat['candidate_mean_mae'])} | "
            f"{_format_float(flat['carry_forward_mean_mae'])} | {_format_float(flat['candidate_minus_carry_forward'])} | "
            f"{_format_float(flat['r10_comparable_candidate_mean_mae'])} | {_format_float(flat['r10_reference_mae'])} | "
            f"{_format_float(flat['candidate_minus_r10'])} | `{flat['decision']}` | `{flat['blockers']}` |"
        )
    lines.extend(
        [
            "",
            "## Horizon Detail",
            "",
            "| Experiment | Horizon | Candidate | Carry | R10-scope | Matched R10 | Delta R10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        for horizon_row in list(row.get("horizon_rows") or []):
            lines.append(
                f"| `{row.get('experiment_id')}` | {horizon_row.get('horizon_years')} | "
                f"{_format_float(horizon_row.get('candidate_mean_mae'))} | "
                f"{_format_float(horizon_row.get('carry_forward_mean_mae'))} | "
                f"{_format_float(horizon_row.get('r10_comparable_candidate_mean_mae'))} | "
                f"{_format_float(horizon_row.get('r10_horizon_reference_mae'))} | "
                f"{_format_float(horizon_row.get('candidate_minus_r10_reference_mae'))} |"
            )
    lines.extend(
        [
            "",
            "## Scientific Contract",
            "",
            "- R25 is a predictive endpoint readout branch, not a mechanistic transition-process claim.",
            "- R25 mutates only the R10-comparable endpoint scope: diagnosed stock, ART stock, and diagnosis flow.",
            "- VL testing and suppression are regenerated from the Phase 3 conditional-rate projection after stock-cone projection.",
            "- Matched R10 is used only as an external gate, not as a training target.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, results: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return

    rows = [row for row in results if row.get("horizon_rows")]
    labels: list[str] = []
    candidate: list[float] = []
    r10: list[float] = []
    carry: list[float] = []
    for row in rows:
        for horizon_row in list(row.get("horizon_rows") or []):
            if str(row.get("experiment_id") or "") not in {"R25-002", "R25-003"}:
                continue
            candidate_value = _finite_float(horizon_row.get("r10_comparable_candidate_mean_mae"))
            r10_value = _finite_float(horizon_row.get("r10_horizon_reference_mae"))
            carry_value = _finite_float(horizon_row.get("carry_forward_mean_mae"))
            if candidate_value is None or r10_value is None or carry_value is None:
                continue
            labels.append(f"{row.get('experiment_id')} h{horizon_row.get('horizon_years')}")
            candidate.append(candidate_value)
            r10.append(r10_value)
            carry.append(carry_value)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("R25 Predictive Endpoint Readout Gate", fontsize=15, fontweight="bold")
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.25
    axes[0].bar(x - width, carry, width=width, color="#c9b879", label="carry-forward")
    axes[0].bar(x, candidate, width=width, color="#20639b", label="candidate R10-scope")
    axes[0].bar(x + width, r10, width=width, color="#111827", label="matched R10")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    axes[0].set_ylabel("normalized MAE")
    axes[0].legend(loc="upper left")
    axes[0].set_title("External R10-scope gate")
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(r10, dtype=np.float64)
    axes[1].bar(x, delta, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in delta])
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    axes[1].set_ylabel("candidate minus matched R10")
    axes[1].set_title("Negative values beat matched R10")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r25_predictive_readout_evaluation(
    *,
    run_id: str = "p3d-r25-predictive-endpoint-readout-20260502-s00",
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    include_annual_gate: bool = False,
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    phase3_root = sandbox_repo_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(
        root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id,
    )
    rows = build_observation_rows(root, source_run_id, baseline_source_run_id=baseline_source_run_id)
    validation_rows = build_observation_rows(
        root,
        source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        include_validation_only=True,
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    r10_horizon_replay = _build_r10_horizon_replay_report(
        root=root,
        horizons=(1, 3, 5),
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
    )
    prediction_cache: dict[tuple[str, str, int, tuple[int, ...]], tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    annual_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for spec in _r25_specs(include_annual_gate=include_annual_gate):
        if str(spec.get("family") or "") == "official_annual_challenge_gate":
            result = _r13_annual_spec_report(
                spec=spec,
                validation_rows=validation_rows,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                annual_cache=annual_cache,
            )
        else:
            result = _r13_model_spec_report(
                spec=spec,
                rows=rows,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                r10_horizon_replay=r10_horizon_replay,
                prediction_cache=prediction_cache,
            )
        results.append(result)

    r25_rows = {str(row.get("experiment_id") or ""): row for row in results}
    target_ids = ("R25-001", "R25-002", "R25-003") if include_annual_gate else ("R25-002", "R25-003")
    blockers = {
        experiment_id: list(r25_rows.get(experiment_id, {}).get("blockers") or [])
        for experiment_id in target_ids
    }
    promoted = all(str(r25_rows.get(experiment_id, {}).get("decision") or "") == "promote_for_next_wave" for experiment_id in target_ids)
    if promoted:
        if include_annual_gate:
            verdict = "R25 passes the annual, program-route, and full-sentinel gates. It is eligible as a predictive-track champion, with mechanistic claims still anchored to R19."
        else:
            verdict = "R25 passes the matched-R10 blocker gates evaluated in this run. The official annual gate still needs a dedicated R25 replay before champion promotion."
    else:
        verdict = (
            "R25 does not yet satisfy the active goal. It is diagnostic unless all target R25 gates "
            "all promote; remaining blockers are recorded in the result table."
        )

    report = {
        "schema_version": R25_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "min_train_years": int(min_train_years),
        "include_annual_gate": bool(include_annual_gate),
        "annual_gate_status": "inherited_from_locked_r19_reference_when_include_annual_gate_false",
        "candidate_family": R25_FAMILY,
        "reference_family": R19_FAMILY,
        "r10_metric_scope": list(R10_COMPARABLE_METRICS),
        "results": results,
        "r25_target_blockers": blockers,
        "promotion_eligible": promoted,
        "verdict": verdict,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r25_predictive_readout_evaluation.json"
    csv_path = analysis_dir / "r25_predictive_readout_evaluation.csv"
    md_path = analysis_dir / "r25_predictive_readout_evaluation.md"
    dashboard_path = analysis_dir / "r25_predictive_readout_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "csv": csv_path.as_posix(),
        "markdown": md_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(csv_path, results)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, results)
    write_json(json_path, report)
    return report


if __name__ == "__main__":
    run_r25_predictive_readout_evaluation()
