from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at
from .r48_subnational_proxy_champion_contract import COUNT_METRICS, _finite_float
from .r63_region_metric_online_expert_gate import R63_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R64_SCHEMA_VERSION = "phase3_dynamic.r64_leakage_support_gap_prioritizer.v1"
R64_RUN_ID = "p3d-r64-leakage-support-gap-prioritizer-20260503-s00"
R63_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R63_RUN_ID
    / "analysis"
    / "r63_region_metric_online_expert_gate_report.json"
)


def _read_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_stream(metric: str) -> str:
    if metric in {"estimated_plhiv"}:
        return "population_burden_anchor"
    if metric in {"diagnosed_plhiv"}:
        return "diagnosis_stock"
    if metric in {"alive_on_art"}:
        return "art_stock_retention"
    if metric in {"tested_for_viral_load", "virally_suppressed"}:
        return "vl_suppression_service"
    return "other"


def _evidence_recommendation(metric: str) -> str:
    return {
        "estimated_plhiv": "regional PLHIV burden anchors or model-estimate lineage audit by region/year",
        "diagnosed_plhiv": "regional diagnosed-stock support, diagnosis backlog, and reporting-intensity lineages",
        "alive_on_art": "regional ART active-stock, initiation, interruption, transfer-out, and retention support",
        "tested_for_viral_load": "regional VL testing numerator, lab capacity, and testing-coverage denominator support",
        "virally_suppressed": "regional suppression conditional on VL testing, lab reporting, and ART continuity support",
    }.get(metric, "metric-specific regional support evidence")


def _error_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (
            str(row.get("candidate_family") or ""),
            str(row.get("holdout_period") or ""),
            str(row.get("region") or ""),
            str(row.get("metric_name") or ""),
        ): dict(row)
        for row in rows
    }


def _selection_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (
            str(row.get("holdout_period") or ""),
            str(row.get("region") or ""),
            str(row.get("metric_name") or ""),
        ): dict(row)
        for row in rows
    }


def _entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    return float(-sum((count / total) * math.log(max(count / total, 1e-12)) for count in counts.values()))


def _gap_rows(
    *,
    oracle_rows: list[dict[str, Any]],
    student_rows: list[dict[str, Any]],
    candidate_error_rows: list[dict[str, Any]],
    r60_family: str,
) -> list[dict[str, Any]]:
    student_index = _selection_index(student_rows)
    errors = _error_index(candidate_error_rows)
    rows: list[dict[str, Any]] = []
    for oracle in oracle_rows:
        period = str(oracle.get("holdout_period") or "")
        region = str(oracle.get("region") or "")
        metric = str(oracle.get("metric_name") or "")
        oracle_family = str(oracle.get("selected_source_family") or "")
        student = student_index.get((period, region, metric)) or {}
        student_family = str(student.get("selected_source_family") or "")
        oracle_error = _finite_float(oracle.get("selected_normalized_absolute_error"))
        student_error = _finite_float((errors.get((student_family, period, region, metric)) or {}).get("normalized_absolute_error"))
        r60_error = _finite_float((errors.get((r60_family, period, region, metric)) or {}).get("normalized_absolute_error"))
        target_value = _finite_float((errors.get((oracle_family, period, region, metric)) or {}).get("target_value"))
        if oracle_error is None:
            continue
        student_gap = None if student_error is None else float(student_error) - float(oracle_error)
        r60_gap = None if r60_error is None else float(r60_error) - float(oracle_error)
        rows.append(
            {
                "holdout_period": period,
                "region": region,
                "metric_name": metric,
                "metric_stream": _metric_stream(metric),
                "oracle_source_family": oracle_family,
                "student_source_family": student_family,
                "r60_source_family": r60_family,
                "oracle_normalized_absolute_error": oracle_error,
                "student_normalized_absolute_error": student_error,
                "r60_normalized_absolute_error": r60_error,
                "student_gap_vs_oracle": student_gap,
                "r60_gap_vs_oracle": r60_gap,
                "target_value": target_value,
                "student_matches_oracle": student_family == oracle_family,
                "evidence_recommendation": _evidence_recommendation(metric),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row.get("student_gap_vs_oracle") or 0.0),
            -float(row.get("r60_gap_vs_oracle") or 0.0),
            str(row.get("metric_name") or ""),
            str(row.get("region") or ""),
            str(row.get("holdout_period") or ""),
        )
    )
    return rows


def _aggregate_priority(rows: list[dict[str, Any]], keys: tuple[str, ...], *, label: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key) or "") for key in keys)].append(dict(row))
    output: list[dict[str, Any]] = []
    for key_values, group in grouped.items():
        student_gaps = [float(row.get("student_gap_vs_oracle") or 0.0) for row in group if _finite_float(row.get("student_gap_vs_oracle")) is not None]
        r60_gaps = [float(row.get("r60_gap_vs_oracle") or 0.0) for row in group if _finite_float(row.get("r60_gap_vs_oracle")) is not None]
        oracle_families = [str(row.get("oracle_source_family") or "") for row in group if row.get("oracle_source_family")]
        payload = {
            "priority_scope": label,
            "entry_count": len(group),
            "mean_student_gap_vs_oracle": float(np.mean(np.asarray(student_gaps, dtype=np.float64))) if student_gaps else None,
            "sum_positive_student_gap_vs_oracle": float(sum(max(value, 0.0) for value in student_gaps)),
            "mean_r60_gap_vs_oracle": float(np.mean(np.asarray(r60_gaps, dtype=np.float64))) if r60_gaps else None,
            "sum_positive_r60_gap_vs_oracle": float(sum(max(value, 0.0) for value in r60_gaps)),
            "oracle_family_entropy": _entropy(oracle_families),
            "unique_oracle_family_count": len(set(oracle_families)),
            "top_oracle_source_family": Counter(oracle_families).most_common(1)[0][0] if oracle_families else None,
            "top_evidence_recommendation": Counter(str(row.get("evidence_recommendation") or "") for row in group).most_common(1)[0][0],
        }
        for key_name, key_value in zip(keys, key_values):
            payload[key_name] = key_value
        output.append(payload)
    output.sort(
        key=lambda row: (
            -float(row.get("sum_positive_student_gap_vs_oracle") or 0.0),
            -float(row.get("sum_positive_r60_gap_vs_oracle") or 0.0),
            -float(row.get("oracle_family_entropy") or 0.0),
            tuple(str(row.get(key) or "") for key in keys),
        )
    )
    return output


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("support_gap_gate") or {})
    lines = [
        "# Phase 3 R64 Leakage Support Gap Prioritizer",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Gap row count: `{gate.get('gap_row_count')}`",
        f"- Top metric stream: `{gate.get('top_metric_stream')}`",
        f"- Top region: `{gate.get('top_region')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Top Metric Priorities",
        "",
        "| Metric | Entries | Student gap sum | R60 gap sum | Recommendation |",
        "|---|---:|---:|---:|---|",
    ]
    for row in list(report.get("metric_priority_rows") or [])[:10]:
        lines.append(
            f"| `{row.get('metric_name')}` | {int(row.get('entry_count') or 0)} | "
            f"{float(row.get('sum_positive_student_gap_vs_oracle') or 0.0):.6f} | "
            f"{float(row.get('sum_positive_r60_gap_vs_oracle') or 0.0):.6f} | "
            f"{row.get('top_evidence_recommendation')} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, metric_rows: list[dict[str, Any]], region_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    top_metrics = metric_rows[:5]
    top_regions = region_rows[:10]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    axes[0].bar(
        [str(row.get("metric_name") or "") for row in top_metrics],
        [float(row.get("sum_positive_student_gap_vs_oracle") or 0.0) for row in top_metrics],
        color="#846a3a",
    )
    axes[0].set_title("Metric support gaps")
    axes[0].set_ylabel("student minus oracle NAE gap, positive sum")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(
        [str(row.get("region") or "") for row in top_regions],
        [float(row.get("sum_positive_student_gap_vs_oracle") or 0.0) for row in top_regions],
        color="#4f6f8f",
    )
    axes[1].set_title("Regional support gaps")
    axes[1].set_ylabel("student minus oracle NAE gap, positive sum")
    axes[1].tick_params(axis="x", rotation=30)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r64_leakage_support_gap_prioritizer(
    *,
    run_id: str = R64_RUN_ID,
    r63_report_path: Path | None = None,
) -> dict[str, Any]:
    r63_path = Path(r63_report_path) if r63_report_path is not None else R63_DEFAULT_REPORT
    r63 = _read_report(r63_path)
    artifact_paths = dict(r63.get("artifact_paths") or {})
    candidate_error_rows = _read_csv(Path(str(artifact_paths.get("candidate_error_csv") or "")))
    oracle_rows = [dict(row) for row in list(r63.get("oracle_selection_rows") or [])]
    student_rows = [dict(row) for row in list(r63.get("student_selection_rows") or [])]
    r60_family = str((r63.get("online_expert_gate") or {}).get("r60_best_candidate_family") or "")
    gap_rows = _gap_rows(
        oracle_rows=oracle_rows,
        student_rows=student_rows,
        candidate_error_rows=candidate_error_rows,
        r60_family=r60_family,
    )
    metric_priority_rows = _aggregate_priority(gap_rows, ("metric_name",), label="metric")
    metric_stream_priority_rows = _aggregate_priority(gap_rows, ("metric_stream",), label="metric_stream")
    region_priority_rows = _aggregate_priority(gap_rows, ("region",), label="region")
    region_metric_priority_rows = _aggregate_priority(gap_rows, ("region", "metric_name"), label="region_metric")
    period_metric_priority_rows = _aggregate_priority(gap_rows, ("holdout_period", "metric_name"), label="period_metric")
    blockers: list[str] = []
    if not gap_rows:
        blockers.append("no_region_metric_oracle_gap_rows")
    gate = {
        "status": "leakage_support_gap_priorities_ready" if not blockers else "leakage_support_gap_prioritizer_blocked",
        "blockers": blockers,
        "gap_row_count": len(gap_rows),
        "top_metric": (metric_priority_rows[0] or {}).get("metric_name") if metric_priority_rows else None,
        "top_metric_stream": (metric_stream_priority_rows[0] or {}).get("metric_stream") if metric_stream_priority_rows else None,
        "top_region": (region_priority_rows[0] or {}).get("region") if region_priority_rows else None,
        "r60_best_candidate_family": r60_family,
        "contract": (
            "R64 does not create forecasts. It converts the R63 same-holdout oracle/student gap into ranked "
            "evidence-acquisition priorities. These priorities may guide data onboarding and model-family design, "
            "but cannot be used as fitted corrections."
        ),
    }
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r64_leakage_support_gap_prioritizer_report.json"
    md_path = analysis_dir / "r64_leakage_support_gap_prioritizer_report.md"
    gap_csv = analysis_dir / "r64_region_metric_gap_rows.csv"
    metric_csv = analysis_dir / "r64_metric_priority_rows.csv"
    stream_csv = analysis_dir / "r64_metric_stream_priority_rows.csv"
    region_csv = analysis_dir / "r64_region_priority_rows.csv"
    region_metric_csv = analysis_dir / "r64_region_metric_priority_rows.csv"
    period_metric_csv = analysis_dir / "r64_period_metric_priority_rows.csv"
    dashboard_path = analysis_dir / "r64_leakage_support_gap_prioritizer_dashboard.png"
    report = {
        "schema_version": R64_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": blockers,
        "verdict": (
            "R64 identified ranked leakage-derived evidence priorities."
            if not blockers
            else "R64 could not build leakage-derived evidence priorities."
        ),
        "r63_report_path": r63_path.as_posix(),
        "support_gap_gate": gate,
        "gap_rows": gap_rows,
        "metric_priority_rows": metric_priority_rows,
        "metric_stream_priority_rows": metric_stream_priority_rows,
        "region_priority_rows": region_priority_rows,
        "region_metric_priority_rows": region_metric_priority_rows,
        "period_metric_priority_rows": period_metric_priority_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "gap_csv": gap_csv.as_posix(),
            "metric_priority_csv": metric_csv.as_posix(),
            "metric_stream_priority_csv": stream_csv.as_posix(),
            "region_priority_csv": region_csv.as_posix(),
            "region_metric_priority_csv": region_metric_csv.as_posix(),
            "period_metric_priority_csv": period_metric_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(gap_csv, gap_rows)
    _write_csv(metric_csv, metric_priority_rows)
    _write_csv(stream_csv, metric_stream_priority_rows)
    _write_csv(region_csv, region_priority_rows)
    _write_csv(region_metric_csv, region_metric_priority_rows)
    _write_csv(period_metric_csv, period_metric_priority_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, metric_priority_rows, region_priority_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R64 leakage support gap prioritizer.")
    parser.add_argument("--run-id", default=R64_RUN_ID)
    parser.add_argument("--r63-report-path", default=None)
    args = parser.parse_args()
    run_r64_leakage_support_gap_prioritizer(
        run_id=str(args.run_id),
        r63_report_path=None if args.r63_report_path is None else Path(args.r63_report_path),
    )


if __name__ == "__main__":
    _main()
