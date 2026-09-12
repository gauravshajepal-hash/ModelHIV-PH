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
    _rows_by_period_region,
    _score_predictions,
)
from .runtime import ensure_dir, read_json, write_json


R52_SCHEMA_VERSION = "phase3_dynamic.r52_subnational_coherence_gate.v1"
R52_RUN_ID = "p3d-r52-subnational-coherence-gate-20260503-s00"
R48_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r48-subnational-proxy-champion-contract-20260503-s00"
    / "analysis"
    / "r48_subnational_proxy_champion_contract_report.json"
)
R51_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r51-train-only-regional-selector-20260503-s00"
    / "analysis"
    / "r51_train_only_regional_selector_report.json"
)
R49_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r49-subnational-module-champion-gate-20260503-s00"
    / "analysis"
    / "r49_subnational_module_champion_gate_report.json"
)


def _load_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {})


def _candidate_prediction_rows(
    r48_report: dict[str, Any],
    r49_report: dict[str, Any],
    r51_report: dict[str, Any],
) -> list[dict[str, Any]]:
    return [dict(row) for row in list(r48_report.get("prediction_rows") or [])] + [
        dict(row) for row in list(r49_report.get("module_prediction_rows") or [])
    ] + [
        dict(row) for row in list(r51_report.get("prediction_rows") or [])
    ]


def _split_metric_regional_scores(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _score_predictions(rows_by_key, prediction_rows)


def _period_region_candidates(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("region") or "")): dict(row)
        for row in prediction_rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("region")
    }


def _coherence_rows(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = _period_region_candidates(prediction_rows)
    families = sorted({family for family, _period, _region in indexed})
    periods = sorted({period for _family, period, _region in indexed})
    rows: list[dict[str, Any]] = []
    for family in families:
        for period in periods:
            regions = sorted({region for candidate_family, candidate_period, region in indexed if candidate_family == family and candidate_period == period})
            if not regions:
                continue
            for metric in COUNT_METRICS:
                predicted_values: list[float] = []
                target_values: list[float] = []
                for region in regions:
                    prediction = indexed.get((family, period, region))
                    target = rows_by_key.get((period, region))
                    if prediction is None or target is None:
                        continue
                    predicted = _finite_float(prediction.get(metric))
                    observed = _finite_float(target.get(metric))
                    if predicted is None or observed is None:
                        continue
                    predicted_values.append(max(float(predicted), 0.0))
                    target_values.append(max(float(observed), 0.0))
                predicted_sum = float(sum(predicted_values))
                target_sum = float(sum(target_values))
                if not predicted_values or target_sum <= 0.0:
                    continue
                predicted_share = np.asarray(predicted_values, dtype=np.float64) / max(predicted_sum, float(np.finfo(np.float32).eps))
                target_share = np.asarray(target_values, dtype=np.float64) / target_sum
                share_l1 = float(np.sum(np.abs(predicted_share - target_share)))
                rows.append(
                    {
                        "candidate_family": family,
                        "holdout_period": period,
                        "metric_name": metric,
                        "region_count": len(predicted_values),
                        "predicted_sum": predicted_sum,
                        "target_sum": target_sum,
                        "aggregate_mass_absolute_error": abs(predicted_sum - target_sum),
                        "aggregate_mass_normalized_absolute_error": abs(predicted_sum - target_sum) / target_sum,
                        "regional_share_l1_error": share_l1,
                        "regional_share_half_l1_error": 0.5 * share_l1,
                    }
                )
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_finite_float(row.get(key)) for row in rows]
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _max(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_finite_float(row.get(key)) for row in rows]
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(np.max(np.asarray(finite, dtype=np.float64)))


def _regional_summary_table(split_metric_regional_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_metric_regional_scores:
        grouped[str(row.get("candidate_family") or "")].append(dict(row))
    rows: list[dict[str, Any]] = []
    for family, family_rows in sorted(grouped.items()):
        rows.append(
            {
                "candidate_family": family,
                "mean_regional_normalized_absolute_error": _mean(family_rows, "normalized_absolute_error"),
                "worst_regional_normalized_absolute_error": _max(family_rows, "normalized_absolute_error"),
                "scored_split_metric_count": len(family_rows),
            }
        )
    return rows


def _coherence_summary_table(coherence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in coherence_rows:
        grouped[str(row.get("candidate_family") or "")].append(dict(row))
    rows: list[dict[str, Any]] = []
    for family, family_rows in sorted(grouped.items()):
        rows.append(
            {
                "candidate_family": family,
                "mean_aggregate_mass_normalized_absolute_error": _mean(family_rows, "aggregate_mass_normalized_absolute_error"),
                "worst_aggregate_mass_normalized_absolute_error": _max(family_rows, "aggregate_mass_normalized_absolute_error"),
                "mean_regional_share_half_l1_error": _mean(family_rows, "regional_share_half_l1_error"),
                "worst_regional_share_half_l1_error": _max(family_rows, "regional_share_half_l1_error"),
                "scored_split_metric_count": len(family_rows),
            }
        )
    return rows


def _combined_table(
    regional_summary_rows: list[dict[str, Any]],
    coherence_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_family: dict[str, dict[str, Any]] = {}
    for row in regional_summary_rows:
        by_family.setdefault(str(row.get("candidate_family") or ""), {}).update(dict(row))
    for row in coherence_summary_rows:
        by_family.setdefault(str(row.get("candidate_family") or ""), {}).update(dict(row))
    rows = [row for family, row in by_family.items() if family]
    rows.sort(key=lambda row: float(row.get("mean_regional_normalized_absolute_error") or float("inf")))
    return rows


def _gate(combined_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {str(row.get("candidate_family") or ""): dict(row) for row in combined_rows}
    selector = by_family.get("train_only_region_metric_selector") or {}
    carry = by_family.get("regional_carry_forward") or {}
    selector_regional = _finite_float(selector.get("mean_regional_normalized_absolute_error"))
    carry_regional = _finite_float(carry.get("mean_regional_normalized_absolute_error"))
    selector_mass = _finite_float(selector.get("mean_aggregate_mass_normalized_absolute_error"))
    carry_mass = _finite_float(carry.get("mean_aggregate_mass_normalized_absolute_error"))
    selector_share = _finite_float(selector.get("mean_regional_share_half_l1_error"))
    carry_share = _finite_float(carry.get("mean_regional_share_half_l1_error"))
    eligible: list[dict[str, Any]] = []
    for row in combined_rows:
        family = str(row.get("candidate_family") or "")
        if not family or family == "regional_carry_forward":
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
        eligible.append(dict(row))
    eligible.sort(key=lambda row: float(row.get("mean_regional_normalized_absolute_error") or float("inf")))
    promoted = eligible[0] if eligible else {}
    blockers: list[str] = []
    if not promoted:
        if selector_regional is None:
            blockers.append("train_only_selector_missing")
        if selector_regional is not None and carry_regional is not None and not selector_regional < carry_regional:
            blockers.append("selector_does_not_beat_carry_forward_regional_error")
        if selector_mass is not None and carry_mass is not None and selector_mass > carry_mass:
            blockers.append("selector_worsens_aggregate_mass_error")
        if selector_share is not None and carry_share is not None and selector_share > carry_share:
            blockers.append("selector_worsens_regional_share_error")
        if not blockers:
            blockers.append("no_non_carry_forward_candidate_passes_coherence_gate")
    status = "coherent_subnational_champion_promoted" if not blockers else "subnational_selector_not_coherent_enough"
    return {
        "status": status,
        "blockers": blockers,
        "promoted_candidate_family": promoted.get("candidate_family"),
        "promoted_mean_regional_normalized_absolute_error": promoted.get("mean_regional_normalized_absolute_error"),
        "promoted_mean_aggregate_mass_normalized_absolute_error": promoted.get("mean_aggregate_mass_normalized_absolute_error"),
        "promoted_mean_regional_share_half_l1_error": promoted.get("mean_regional_share_half_l1_error"),
        "selector_mean_regional_normalized_absolute_error": selector_regional,
        "carry_forward_mean_regional_normalized_absolute_error": carry_regional,
        "selector_mean_aggregate_mass_normalized_absolute_error": selector_mass,
        "carry_forward_mean_aggregate_mass_normalized_absolute_error": carry_mass,
        "selector_mean_regional_share_half_l1_error": selector_share,
        "carry_forward_mean_regional_share_half_l1_error": carry_share,
        "contract": (
            "A subnational champion must improve regional point errors without worsening aggregate national "
            "mass reconstruction or regional allocation/share error versus carry-forward. This separates "
            "total-count forecasting from true subnational allocation skill."
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
    gate = dict(report.get("coherence_gate") or {})
    lines = [
        "# Phase 3 R52 Subnational Coherence Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Promoted candidate: `{gate.get('promoted_candidate_family') or 'none'}`",
        f"- Promoted regional / mass / share errors: `{gate.get('promoted_mean_regional_normalized_absolute_error')}` / `{gate.get('promoted_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('promoted_mean_regional_share_half_l1_error')}`",
        f"- Regional error selector vs carry-forward: `{gate.get('selector_mean_regional_normalized_absolute_error')}` vs `{gate.get('carry_forward_mean_regional_normalized_absolute_error')}`",
        f"- Aggregate mass error selector vs carry-forward: `{gate.get('selector_mean_aggregate_mass_normalized_absolute_error')}` vs `{gate.get('carry_forward_mean_aggregate_mass_normalized_absolute_error')}`",
        f"- Regional share error selector vs carry-forward: `{gate.get('selector_mean_regional_share_half_l1_error')}` vs `{gate.get('carry_forward_mean_regional_share_half_l1_error')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Candidate Table",
        "",
        "| Candidate | Regional NAE | Aggregate mass NAE | Share half-L1 | Worst regional NAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in list(report.get("combined_candidate_table") or []):
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
    labels = [str(row.get("candidate_family") or "") for row in combined_rows]
    x = np.arange(len(labels))
    width = 0.25
    regional = [float(row.get("mean_regional_normalized_absolute_error") or 0.0) for row in combined_rows]
    mass = [float(row.get("mean_aggregate_mass_normalized_absolute_error") or 0.0) for row in combined_rows]
    share = [float(row.get("mean_regional_share_half_l1_error") or 0.0) for row in combined_rows]
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.bar(x - width, regional, width, label="regional NAE", color="#516b5f")
    ax.bar(x, mass, width, label="aggregate mass NAE", color="#496f9e")
    ax.bar(x + width, share, width, label="regional share half-L1", color="#8a6f2a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("error")
    ax.set_title("R52 subnational coherence gate")
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r52_subnational_coherence_gate(
    *,
    run_id: str = R52_RUN_ID,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
    r51_report_path: Path | None = None,
) -> dict[str, Any]:
    r48_path = Path(r48_report_path) if r48_report_path is not None else R48_DEFAULT_REPORT
    r49_path = Path(r49_report_path) if r49_report_path is not None else R49_DEFAULT_REPORT
    r51_path = Path(r51_report_path) if r51_report_path is not None else R51_DEFAULT_REPORT
    r48_report = _load_report(r48_path)
    r49_report = _load_report(r49_path)
    r51_report = _load_report(r51_path)
    r44_path = Path(str(r48_report.get("r44_report_path") or r51_report.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    prediction_rows = _candidate_prediction_rows(r48_report, r49_report, r51_report)
    regional_score_rows, regional_split_metric_rows = _split_metric_regional_scores(rows_by_key, prediction_rows)
    coherence_rows = _coherence_rows(rows_by_key, prediction_rows)
    regional_summary_rows = _regional_summary_table(regional_split_metric_rows)
    coherence_summary_rows = _coherence_summary_table(coherence_rows)
    combined_rows = _combined_table(regional_summary_rows, coherence_summary_rows)
    gate = _gate(combined_rows)
    verdict = (
        "R52 promoted a subnational champion that improves regional errors without breaking national mass or allocation coherence."
        if gate["status"] == "coherent_subnational_champion_promoted"
        else "R52 blocked the subnational selector from a full coherence claim; keep it as a readout diagnostic until mass/allocation errors are fixed."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r52_subnational_coherence_gate_report.json"
    md_path = analysis_dir / "r52_subnational_coherence_gate_report.md"
    combined_csv = analysis_dir / "r52_combined_candidate_table.csv"
    coherence_csv = analysis_dir / "r52_coherence_rows.csv"
    regional_csv = analysis_dir / "r52_regional_split_metric_scores.csv"
    dashboard_path = analysis_dir / "r52_subnational_coherence_dashboard.png"
    report = {
        "schema_version": R52_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r44_report_path": r44_path.as_posix(),
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "aggregate_mass_error": "absolute error of summed regional predictions divided by summed regional targets",
            "regional_share_error": "half L1 distance between predicted regional shares and observed regional shares",
            "claim_limit": "subnational coherence gate; determinant and mechanistic regional claims remain separately gated",
        },
        "coherence_gate": gate,
        "combined_candidate_table": combined_rows,
        "regional_score_rows": regional_score_rows,
        "regional_split_metric_score_rows": regional_split_metric_rows,
        "coherence_rows": coherence_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "combined_candidate_csv": combined_csv.as_posix(),
            "coherence_rows_csv": coherence_csv.as_posix(),
            "regional_split_metric_scores_csv": regional_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(combined_csv, combined_rows)
    _write_csv(coherence_csv, coherence_rows)
    _write_csv(regional_csv, regional_split_metric_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, combined_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R52 subnational coherence gate.")
    parser.add_argument("--run-id", default=R52_RUN_ID)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    args = parser.parse_args()
    run_r52_subnational_coherence_gate(
        run_id=str(args.run_id),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
    )


if __name__ == "__main__":
    _main()
