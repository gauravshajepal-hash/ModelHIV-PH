from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at
from .r48_subnational_proxy_champion_contract import _finite_float
from .runtime import ensure_dir, read_json, write_json


R55_SCHEMA_VERSION = "phase3_dynamic.r55_adapter_split_stability_gate.v1"
R55_RUN_ID = "p3d-r55-adapter-split-stability-gate-20260503-s00"
R54_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r54-national-total-regional-adapter-20260503-s00"
    / "analysis"
    / "r54_national_total_regional_adapter_report.json"
)


def _load_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {})


def _regional_score_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("metric_name") or "")): dict(row)
        for row in rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("metric_name")
    }


def _coherence_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("metric_name") or "")): dict(row)
        for row in rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("metric_name")
    }


def _nonregression(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    # Avoid flagging machine-roundoff differences as scientific regressions.
    return float(left) <= float(right) + sys.float_info.epsilon * max(1.0, abs(float(right)))


def _comparison_rows(
    *,
    regional_score_rows: list[dict[str, Any]],
    coherence_rows: list[dict[str, Any]],
    promoted_family: str,
    carry_family: str = "regional_carry_forward",
    reference_family: str = "module_local_selector",
) -> list[dict[str, Any]]:
    regional = _regional_score_index(regional_score_rows)
    coherence = _coherence_index(coherence_rows)
    split_keys = sorted(
        {
            (period, metric)
            for family, period, metric in set(regional) | set(coherence)
            if family == promoted_family
        }
    )
    rows: list[dict[str, Any]] = []
    for period, metric in split_keys:
        promoted_regional = _finite_float((regional.get((promoted_family, period, metric)) or {}).get("normalized_absolute_error"))
        carry_regional = _finite_float((regional.get((carry_family, period, metric)) or {}).get("normalized_absolute_error"))
        reference_regional = _finite_float((regional.get((reference_family, period, metric)) or {}).get("normalized_absolute_error"))
        promoted_coherence = coherence.get((promoted_family, period, metric)) or {}
        carry_coherence = coherence.get((carry_family, period, metric)) or {}
        reference_coherence = coherence.get((reference_family, period, metric)) or {}
        promoted_mass = _finite_float(promoted_coherence.get("aggregate_mass_normalized_absolute_error"))
        carry_mass = _finite_float(carry_coherence.get("aggregate_mass_normalized_absolute_error"))
        reference_mass = _finite_float(reference_coherence.get("aggregate_mass_normalized_absolute_error"))
        promoted_share = _finite_float(promoted_coherence.get("regional_share_half_l1_error"))
        carry_share = _finite_float(carry_coherence.get("regional_share_half_l1_error"))
        reference_share = _finite_float(reference_coherence.get("regional_share_half_l1_error"))
        rows.append(
            {
                "holdout_period": period,
                "metric_name": metric,
                "promoted_candidate_family": promoted_family,
                "carry_forward_family": carry_family,
                "reference_family": reference_family,
                "promoted_regional_normalized_absolute_error": promoted_regional,
                "carry_forward_regional_normalized_absolute_error": carry_regional,
                "reference_regional_normalized_absolute_error": reference_regional,
                "promoted_aggregate_mass_normalized_absolute_error": promoted_mass,
                "carry_forward_aggregate_mass_normalized_absolute_error": carry_mass,
                "reference_aggregate_mass_normalized_absolute_error": reference_mass,
                "promoted_regional_share_half_l1_error": promoted_share,
                "carry_forward_regional_share_half_l1_error": carry_share,
                "reference_regional_share_half_l1_error": reference_share,
                "regional_nonregression_vs_carry_forward": (
                    _nonregression(promoted_regional, carry_regional)
                ),
                "mass_nonregression_vs_carry_forward": _nonregression(promoted_mass, carry_mass),
                "share_nonregression_vs_carry_forward": _nonregression(promoted_share, carry_share),
                "regional_nonregression_vs_reference": (
                    _nonregression(promoted_regional, reference_regional)
                ),
                "mass_nonregression_vs_reference": _nonregression(promoted_mass, reference_mass),
                "share_nonregression_vs_reference": _nonregression(promoted_share, reference_share),
            }
        )
    return rows


def _gate(comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    if not comparison_rows:
        blockers.append("no_split_metric_comparisons")
    checks = [
        "regional_nonregression_vs_carry_forward",
        "mass_nonregression_vs_carry_forward",
        "share_nonregression_vs_carry_forward",
        "regional_nonregression_vs_reference",
        "mass_nonregression_vs_reference",
        "share_nonregression_vs_reference",
    ]
    failure_counts: dict[str, int] = {}
    for check in checks:
        count = sum(1 for row in comparison_rows if row.get(check) is not True)
        failure_counts[check] = int(count)
        if count:
            blockers.append(f"{check}_failed")
    period_failures: Counter[str] = Counter()
    metric_failures: Counter[str] = Counter()
    for row in comparison_rows:
        if any(row.get(check) is not True for check in checks):
            period_failures[str(row.get("holdout_period") or "")] += 1
            metric_failures[str(row.get("metric_name") or "")] += 1
    status = "strict_split_stable_adapter_promoted" if not blockers else "adapter_mean_promoted_but_split_stability_limited"
    return {
        "status": status,
        "blockers": blockers,
        "split_metric_count": len(comparison_rows),
        "failure_counts": failure_counts,
        "period_failure_counts": dict(sorted(period_failures.items())),
        "metric_failure_counts": dict(sorted(metric_failures.items())),
        "contract": (
            "R55 is a strict nonregression stability gate. The R54 adapter must not regress against carry-forward "
            "or the R52 reference on regional error, aggregate mass error, or regional share error for any scored "
            "period-metric split. Failure does not erase the R54 mean-level claim; it limits the wording."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("split_stability_gate") or {})
    lines = [
        "# Phase 3 R55 Adapter Split Stability Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Split-metrics: `{gate.get('split_metric_count')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Failure Counts",
        "",
        "| Check | Failed split-metrics |",
        "|---|---:|",
    ]
    for key, value in dict(gate.get("failure_counts") or {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Period Failures",
            "",
            "| Period | Failed checks |",
            "|---|---:|",
        ]
    )
    for key, value in dict(gate.get("period_failure_counts") or {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r55_adapter_split_stability_gate(
    *,
    run_id: str = R55_RUN_ID,
    r54_report_path: Path | None = None,
) -> dict[str, Any]:
    r54_path = Path(r54_report_path) if r54_report_path is not None else R54_DEFAULT_REPORT
    r54 = _load_report(r54_path)
    promoted_family = str((r54.get("adapter_gate") or {}).get("promoted_candidate_family") or "")
    comparison_rows = _comparison_rows(
        regional_score_rows=[dict(row) for row in list(r54.get("split_metric_score_rows") or [])],
        coherence_rows=[dict(row) for row in list(r54.get("coherence_rows") or [])],
        promoted_family=promoted_family,
    )
    gate = _gate(comparison_rows)
    verdict = (
        "R55 promoted the R54 adapter as split-stable."
        if gate["status"] == "strict_split_stable_adapter_promoted"
        else "R55 found that the R54 adapter is mean-promoted but not split-stable; publication language must report localized regressions."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r55_adapter_split_stability_gate_report.json"
    md_path = analysis_dir / "r55_adapter_split_stability_gate_report.md"
    csv_path = analysis_dir / "r55_split_comparison_rows.csv"
    report = {
        "schema_version": R55_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r54_report_path": r54_path.as_posix(),
        "promoted_candidate_family": promoted_family,
        "split_stability_gate": gate,
        "comparison_rows": comparison_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "comparison_csv": csv_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(csv_path, comparison_rows)
    _write_markdown(md_path, report)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R55 adapter split stability gate.")
    parser.add_argument("--run-id", default=R55_RUN_ID)
    parser.add_argument("--r54-report-path", default=None)
    args = parser.parse_args()
    run_r55_adapter_split_stability_gate(
        run_id=str(args.run_id),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
    )


if __name__ == "__main__":
    _main()
