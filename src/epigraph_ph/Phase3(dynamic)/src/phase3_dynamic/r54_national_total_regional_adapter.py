from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at
from .r48_subnational_proxy_champion_contract import (
    COUNT_METRICS,
    _finite_float,
    _load_r44_rows,
    _projection_cascade,
    _rows_by_period_region,
    _score_predictions,
)
from .r52_subnational_coherence_gate import (
    _coherence_rows,
    _coherence_summary_table,
    _combined_table,
    _regional_summary_table,
)
from .runtime import ensure_dir, read_json, write_json


R54_SCHEMA_VERSION = "phase3_dynamic.r54_national_total_regional_adapter.v1"
R54_RUN_ID = "p3d-r54-national-total-regional-adapter-20260503-s00"
R48_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r48-subnational-proxy-champion-contract-20260503-s00"
    / "analysis"
    / "r48_subnational_proxy_champion_contract_report.json"
)
R49_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r49-subnational-module-champion-gate-20260503-s00"
    / "analysis"
    / "r49_subnational_module_champion_gate_report.json"
)
R51_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r51-train-only-regional-selector-20260503-s00"
    / "analysis"
    / "r51_train_only_regional_selector_report.json"
)
R52_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r52-subnational-coherence-gate-20260503-s00"
    / "analysis"
    / "r52_subnational_coherence_gate_report.json"
)


def _load_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {})


def _base_prediction_rows(
    r48_report: dict[str, Any],
    r49_report: dict[str, Any],
    r51_report: dict[str, Any],
) -> list[dict[str, Any]]:
    return [dict(row) for row in list(r48_report.get("prediction_rows") or [])] + [
        dict(row) for row in list(r49_report.get("module_prediction_rows") or [])
    ] + [
        dict(row) for row in list(r51_report.get("prediction_rows") or [])
    ]


def _prediction_index(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("region") or "")): dict(row)
        for row in prediction_rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("region")
    }


def _available_families(prediction_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("candidate_family") or "") for row in prediction_rows if row.get("candidate_family")})


def _period_regions_for_family(indexed: dict[tuple[str, str, str], dict[str, Any]], family: str, period: str) -> list[str]:
    return sorted({region for candidate_family, candidate_period, region in indexed if candidate_family == family and candidate_period == period})


def _family_period_metric_total(
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    *,
    family: str,
    period: str,
    metric: str,
    regions: list[str],
) -> float | None:
    values: list[float] = []
    for region in regions:
        row = indexed.get((family, period, region))
        if row is None:
            return None
        value = _finite_float(row.get(metric))
        if value is None:
            return None
        values.append(max(float(value), 0.0))
    return float(sum(values))


def _family_period_metric_shares(
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    *,
    family: str,
    period: str,
    metric: str,
    regions: list[str],
) -> dict[str, float] | None:
    values: list[float] = []
    for region in regions:
        row = indexed.get((family, period, region))
        if row is None:
            return None
        value = _finite_float(row.get(metric))
        if value is None:
            return None
        values.append(max(float(value), 0.0))
    total = float(sum(values))
    if total <= 0.0:
        return {region: 1.0 / float(len(regions)) for region in regions} if regions else None
    return {region: float(value / total) for region, value in zip(regions, values)}


def _adapter_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    *,
    total_families: tuple[str, ...] = (
        "regional_carry_forward",
        "aggregate_log_trend",
        "module_local_selector",
        "train_only_region_metric_selector",
    ),
    share_families: tuple[str, ...] = (
        "regional_carry_forward",
        "aggregate_log_trend",
        "similarity_proxy_log_delta",
        "module_local_selector",
        "train_only_region_metric_selector",
    ),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = _prediction_index(prediction_rows)
    available = set(_available_families(prediction_rows))
    periods = sorted({period for _family, period, _region in indexed})
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for total_family in total_families:
        if total_family not in available:
            continue
        for share_family in share_families:
            if share_family not in available:
                continue
            candidate_family = f"national_total_adapter__total={total_family}__share={share_family}"
            for period in periods:
                total_regions = set(_period_regions_for_family(indexed, total_family, period))
                share_regions = set(_period_regions_for_family(indexed, share_family, period))
                regions = sorted(total_regions & share_regions)
                if not regions:
                    diagnostics.append(
                        {
                            "candidate_family": candidate_family,
                            "period_id": period,
                            "status": "skipped_no_common_regions",
                            "total_family": total_family,
                            "share_family": share_family,
                        }
                    )
                    continue
                per_region: dict[str, dict[str, Any]] = {
                    region: {
                        "candidate_family": candidate_family,
                        "holdout_period": period,
                        "region": region,
                        "national_total_source_family": total_family,
                        "regional_share_source_family": share_family,
                    }
                    for region in regions
                }
                metric_status: dict[str, str] = {}
                for metric in COUNT_METRICS:
                    total = _family_period_metric_total(
                        indexed,
                        family=total_family,
                        period=period,
                        metric=metric,
                        regions=regions,
                    )
                    shares = _family_period_metric_shares(
                        indexed,
                        family=share_family,
                        period=period,
                        metric=metric,
                        regions=regions,
                    )
                    if total is None or shares is None:
                        metric_status[metric] = "missing"
                        continue
                    for region in regions:
                        per_region[region][metric] = float(total * float(shares.get(region, 0.0)))
                        per_region[region][f"{metric}_national_total_source_family"] = total_family
                        per_region[region][f"{metric}_regional_share_source_family"] = share_family
                    metric_status[metric] = "completed"
                for region in regions:
                    raw = per_region[region]
                    projected = _projection_cascade(raw)
                    projected["projection_adjusted"] = any(
                        abs(float(projected.get(metric) or 0.0) - float(raw.get(metric) or 0.0)) > 1e-8
                        for metric in COUNT_METRICS
                    )
                    rows.append(projected)
                diagnostics.append(
                    {
                        "candidate_family": candidate_family,
                        "period_id": period,
                        "status": "completed",
                        "total_family": total_family,
                        "share_family": share_family,
                        "region_count": len(regions),
                        "metric_status": metric_status,
                    }
                )
    return rows, diagnostics


def _score_candidate_rows(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    regional_score_rows, regional_split_metric_rows = _score_predictions(rows_by_key, prediction_rows)
    coherence = _coherence_rows(rows_by_key, prediction_rows)
    regional_summary_rows = _regional_summary_table(regional_split_metric_rows)
    coherence_summary_rows = _coherence_summary_table(coherence)
    combined_rows = _combined_table(regional_summary_rows, coherence_summary_rows)
    return regional_split_metric_rows, coherence, combined_rows


def _gate(
    combined_rows: list[dict[str, Any]],
    *,
    r52_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_family = {str(row.get("candidate_family") or ""): dict(row) for row in combined_rows}
    carry = by_family.get("regional_carry_forward") or {}
    carry_regional = _finite_float(carry.get("mean_regional_normalized_absolute_error"))
    carry_mass = _finite_float(carry.get("mean_aggregate_mass_normalized_absolute_error"))
    carry_share = _finite_float(carry.get("mean_regional_share_half_l1_error"))
    r52 = dict(r52_gate or {})
    reference_regional = _finite_float(r52.get("promoted_mean_regional_normalized_absolute_error"))
    reference_mass = _finite_float(r52.get("promoted_mean_aggregate_mass_normalized_absolute_error"))
    reference_share = _finite_float(r52.get("promoted_mean_regional_share_half_l1_error"))
    eligible: list[dict[str, Any]] = []
    for row in combined_rows:
        family = str(row.get("candidate_family") or "")
        if not family.startswith("national_total_adapter__"):
            continue
        regional = _finite_float(row.get("mean_regional_normalized_absolute_error"))
        mass = _finite_float(row.get("mean_aggregate_mass_normalized_absolute_error"))
        share = _finite_float(row.get("mean_regional_share_half_l1_error"))
        if regional is None or mass is None or share is None:
            continue
        if carry_regional is not None and not regional < carry_regional:
            continue
        if carry_mass is not None and mass > carry_mass:
            continue
        if carry_share is not None and share > carry_share:
            continue
        if reference_regional is not None and regional > reference_regional:
            continue
        if reference_mass is not None and mass > reference_mass:
            continue
        if reference_share is not None and share > reference_share:
            continue
        eligible.append(dict(row))
    eligible.sort(key=lambda row: float(row.get("mean_regional_normalized_absolute_error") or float("inf")))
    promoted = eligible[0] if eligible else {}
    blockers: list[str] = []
    if not promoted:
        blockers.append("no_national_total_adapter_beats_carry_forward_and_r52_reference")
    status = "national_total_regional_adapter_promoted" if not blockers else "national_total_regional_adapter_diagnostic_only"
    return {
        "status": status,
        "blockers": blockers,
        "promoted_candidate_family": promoted.get("candidate_family"),
        "promoted_mean_regional_normalized_absolute_error": promoted.get("mean_regional_normalized_absolute_error"),
        "promoted_mean_aggregate_mass_normalized_absolute_error": promoted.get("mean_aggregate_mass_normalized_absolute_error"),
        "promoted_mean_regional_share_half_l1_error": promoted.get("mean_regional_share_half_l1_error"),
        "carry_forward_mean_regional_normalized_absolute_error": carry_regional,
        "carry_forward_mean_aggregate_mass_normalized_absolute_error": carry_mass,
        "carry_forward_mean_regional_share_half_l1_error": carry_share,
        "r52_reference_mean_regional_normalized_absolute_error": reference_regional,
        "r52_reference_mean_aggregate_mass_normalized_absolute_error": reference_mass,
        "r52_reference_mean_regional_share_half_l1_error": reference_share,
        "contract": (
            "R54 may promote only a train-origin regional adapter whose national totals come from an existing "
            "forecast candidate and whose regional allocation comes from an existing regional candidate. It must "
            "beat carry-forward and not regress against the R52 coherent subnational reference on regional, mass, "
            "or share error."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, list):
                normalized[key] = "|".join(str(item) for item in value)
            elif isinstance(value, dict):
                normalized[key] = str(value)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("adapter_gate") or {})
    lines = [
        "# Phase 3 R54 National-Total Regional Adapter",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Promoted candidate: `{gate.get('promoted_candidate_family') or 'none'}`",
        f"- Promoted regional / mass / share errors: `{gate.get('promoted_mean_regional_normalized_absolute_error')}` / `{gate.get('promoted_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('promoted_mean_regional_share_half_l1_error')}`",
        f"- R52 reference regional / mass / share errors: `{gate.get('r52_reference_mean_regional_normalized_absolute_error')}` / `{gate.get('r52_reference_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('r52_reference_mean_regional_share_half_l1_error')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Candidate Table",
        "",
        "| Candidate | Regional NAE | Aggregate mass NAE | Share half-L1 | Worst regional NAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in list(report.get("combined_candidate_table") or [])[:30]:
        lines.append(
            f"| `{row.get('candidate_family')}` | {float(row.get('mean_regional_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('mean_aggregate_mass_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('mean_regional_share_half_l1_error') or 0.0):.6f} | "
            f"{float(row.get('worst_regional_normalized_absolute_error') or 0.0):.6f} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, combined_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    top = list(combined_rows)[:12]
    labels = [str(row.get("candidate_family") or "") for row in top]
    x = np.arange(len(labels))
    width = 0.25
    regional = [float(row.get("mean_regional_normalized_absolute_error") or 0.0) for row in top]
    mass = [float(row.get("mean_aggregate_mass_normalized_absolute_error") or 0.0) for row in top]
    share = [float(row.get("mean_regional_share_half_l1_error") or 0.0) for row in top]
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.bar(x - width, regional, width, label="regional NAE", color="#516b5f")
    ax.bar(x, mass, width, label="aggregate mass NAE", color="#496f9e")
    ax.bar(x + width, share, width, label="regional share half-L1", color="#8a6f2a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("error")
    ax.set_title("R54 national-total regional adapter")
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r54_national_total_regional_adapter(
    *,
    run_id: str = R54_RUN_ID,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
    r51_report_path: Path | None = None,
    r52_report_path: Path | None = None,
) -> dict[str, Any]:
    r48_path = Path(r48_report_path) if r48_report_path is not None else R48_DEFAULT_REPORT
    r49_path = Path(r49_report_path) if r49_report_path is not None else R49_DEFAULT_REPORT
    r51_path = Path(r51_report_path) if r51_report_path is not None else R51_DEFAULT_REPORT
    r52_path = Path(r52_report_path) if r52_report_path is not None else R52_DEFAULT_REPORT
    r48 = _load_report(r48_path)
    r49 = _load_report(r49_path)
    r51 = _load_report(r51_path)
    r52 = _load_report(r52_path)
    r44_path = Path(str(r48.get("r44_report_path") or r51.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    base_rows = _base_prediction_rows(r48, r49, r51)
    adapter_rows, adapter_diagnostics = _adapter_prediction_rows(base_rows)
    all_prediction_rows = base_rows + adapter_rows
    split_scores, coherence_rows, combined_rows = _score_candidate_rows(rows_by_key, all_prediction_rows)
    gate = _gate(combined_rows, r52_gate=dict(r52.get("coherence_gate") or {}))
    verdict = (
        "R54 promoted a national-total-constrained regional adapter."
        if gate["status"] == "national_total_regional_adapter_promoted"
        else "R54 found no national-total-constrained adapter that improves on the R52 coherent subnational reference."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r54_national_total_regional_adapter_report.json"
    md_path = analysis_dir / "r54_national_total_regional_adapter_report.md"
    combined_csv = analysis_dir / "r54_combined_candidate_table.csv"
    prediction_csv = analysis_dir / "r54_adapter_prediction_rows.csv"
    diagnostics_csv = analysis_dir / "r54_adapter_diagnostics.csv"
    split_csv = analysis_dir / "r54_split_metric_scores.csv"
    coherence_csv = analysis_dir / "r54_coherence_rows.csv"
    dashboard_path = analysis_dir / "r54_national_total_regional_adapter_dashboard.png"
    report = {
        "schema_version": R54_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r44_report_path": r44_path.as_posix(),
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r52_report_path": r52_path.as_posix(),
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "adapter_rule": "regional prediction = train-origin national total forecast from one candidate times train-origin regional share forecast from another candidate",
            "target_leakage_status": "no holdout observed regional total is used as a scaling target",
            "claim_limit": "regional adapter/readout claim only; not a determinant or province-level mechanistic claim",
        },
        "adapter_gate": gate,
        "combined_candidate_table": combined_rows,
        "adapter_prediction_rows": adapter_rows,
        "adapter_diagnostics": adapter_diagnostics,
        "split_metric_score_rows": split_scores,
        "coherence_rows": coherence_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "combined_candidate_csv": combined_csv.as_posix(),
            "adapter_prediction_csv": prediction_csv.as_posix(),
            "adapter_diagnostics_csv": diagnostics_csv.as_posix(),
            "split_metric_scores_csv": split_csv.as_posix(),
            "coherence_rows_csv": coherence_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(combined_csv, combined_rows)
    _write_csv(prediction_csv, adapter_rows)
    _write_csv(diagnostics_csv, adapter_diagnostics)
    _write_csv(split_csv, split_scores)
    _write_csv(coherence_csv, coherence_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, combined_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R54 national-total regional adapter.")
    parser.add_argument("--run-id", default=R54_RUN_ID)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    parser.add_argument("--r52-report-path", default=None)
    args = parser.parse_args()
    run_r54_national_total_regional_adapter(
        run_id=str(args.run_id),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
        r52_report_path=None if args.r52_report_path is None else Path(args.r52_report_path),
    )


if __name__ == "__main__":
    _main()
