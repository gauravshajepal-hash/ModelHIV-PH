from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .metrics import quarter_sort_key
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
from .r54_national_total_regional_adapter import (
    R48_DEFAULT_REPORT,
    R49_DEFAULT_REPORT,
    R51_DEFAULT_REPORT,
    _base_prediction_rows,
    _load_report,
)
from .runtime import ensure_dir, read_json, write_json


R57_SCHEMA_VERSION = "phase3_dynamic.r57_regional_candidate_ceiling_diagnostic.v1"
R57_RUN_ID = "p3d-r57-regional-candidate-ceiling-diagnostic-20260503-s00"
CEILING_FAMILY = "leakage_labeled_candidate_ceiling"
R54_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r54-national-total-regional-adapter-20260503-s00"
    / "analysis"
    / "r54_national_total_regional_adapter_report.json"
)


def _read_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _prediction_index(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("region") or "")): dict(row)
        for row in prediction_rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("region")
    }


def _score_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("metric_name") or "")): dict(row)
        for row in rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("metric_name")
    }


def _candidate_metric_error_tuple(
    *,
    regional_index: dict[tuple[str, str, str], dict[str, Any]],
    coherence_index: dict[tuple[str, str, str], dict[str, Any]],
    family: str,
    period: str,
    metric: str,
) -> tuple[float, float, float] | None:
    regional = _finite_float((regional_index.get((family, period, metric)) or {}).get("normalized_absolute_error"))
    mass = _finite_float((coherence_index.get((family, period, metric)) or {}).get("aggregate_mass_normalized_absolute_error"))
    share = _finite_float((coherence_index.get((family, period, metric)) or {}).get("regional_share_half_l1_error"))
    if regional is None or mass is None or share is None:
        return None
    return float(regional), float(mass), float(share)


def _ceiling_selection_rows(
    *,
    prediction_rows: list[dict[str, Any]],
    regional_score_rows: list[dict[str, Any]],
    coherence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = _prediction_index(prediction_rows)
    regional_index = _score_index(regional_score_rows)
    coherence_index = _score_index(coherence_rows)
    periods = sorted({period for _family, period, _region in indexed}, key=quarter_sort_key)
    families = sorted({family for family, _period, _region in indexed})
    rows: list[dict[str, Any]] = []
    for period in periods:
        regions = sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        for metric in COUNT_METRICS:
            choices: list[tuple[float, float, float, str]] = []
            for family in families:
                if not all((family, period, region) in indexed for region in regions):
                    continue
                errors = _candidate_metric_error_tuple(
                    regional_index=regional_index,
                    coherence_index=coherence_index,
                    family=family,
                    period=period,
                    metric=metric,
                )
                if errors is None:
                    continue
                choices.append((*errors, family))
            if not choices:
                continue
            regional, mass, share, family = min(choices)
            rows.append(
                {
                    "holdout_period": period,
                    "metric_name": metric,
                    "selected_candidate_family": family,
                    "selected_regional_normalized_absolute_error": regional,
                    "selected_aggregate_mass_normalized_absolute_error": mass,
                    "selected_regional_share_half_l1_error": share,
                    "candidate_count": len(choices),
                    "leakage_status": "uses_same_split_targets_for_diagnostic_ceiling_only",
                }
            )
    return rows


def _ceiling_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    *,
    ceiling_family: str = CEILING_FAMILY,
) -> list[dict[str, Any]]:
    indexed = _prediction_index(prediction_rows)
    selections = {
        (str(row.get("holdout_period") or ""), str(row.get("metric_name") or "")): str(row.get("selected_candidate_family") or "")
        for row in selection_rows
    }
    periods = sorted({period for _family, period, _region in indexed}, key=quarter_sort_key)
    output_rows: list[dict[str, Any]] = []
    for period in periods:
        regions = sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        for region in regions:
            row: dict[str, Any] = {
                "candidate_family": ceiling_family,
                "holdout_period": period,
                "region": region,
                "leakage_status": "diagnostic_ceiling_not_promotable",
            }
            for metric in COUNT_METRICS:
                selected_family = selections.get((period, metric))
                selected = indexed.get((selected_family or "", period, region))
                if selected is None:
                    continue
                value = _finite_float(selected.get(metric))
                if value is None:
                    continue
                row[metric] = float(value)
                row[f"{metric}_selected_candidate_family"] = selected_family
            projected = _projection_cascade(row)
            projected["projection_adjusted"] = any(
                abs(float(projected.get(metric) or 0.0) - float(row.get(metric) or 0.0)) > 1e-8
                for metric in COUNT_METRICS
            )
            output_rows.append(projected)
    return output_rows


def _score_ceiling_rows(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _raw, split_rows = _score_predictions(rows_by_key, prediction_rows)
    coherence = _coherence_rows(rows_by_key, prediction_rows)
    combined = _combined_table(_regional_summary_table(split_rows), _coherence_summary_table(coherence))
    return split_rows, coherence, combined


def _family_row(combined_rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    return next((dict(row) for row in combined_rows if str(row.get("candidate_family") or "") == family), {})


def _gate(
    combined_rows: list[dict[str, Any]],
    *,
    r54_gate: dict[str, Any] | None = None,
    ceiling_family: str = CEILING_FAMILY,
) -> dict[str, Any]:
    ceiling = _family_row(combined_rows, ceiling_family)
    r54 = dict(r54_gate or {})
    ceiling_regional = _finite_float(ceiling.get("mean_regional_normalized_absolute_error"))
    ceiling_mass = _finite_float(ceiling.get("mean_aggregate_mass_normalized_absolute_error"))
    ceiling_share = _finite_float(ceiling.get("mean_regional_share_half_l1_error"))
    r54_regional = _finite_float(r54.get("promoted_mean_regional_normalized_absolute_error"))
    r54_mass = _finite_float(r54.get("promoted_mean_aggregate_mass_normalized_absolute_error"))
    r54_share = _finite_float(r54.get("promoted_mean_regional_share_half_l1_error"))
    improvement = {
        "regional_error_delta_vs_r54": None if ceiling_regional is None or r54_regional is None else float(r54_regional - ceiling_regional),
        "mass_error_delta_vs_r54": None if ceiling_mass is None or r54_mass is None else float(r54_mass - ceiling_mass),
        "share_error_delta_vs_r54": None if ceiling_share is None or r54_share is None else float(r54_share - ceiling_share),
    }
    has_unrealized_signal = all(value is not None and value > 0.0 for value in improvement.values())
    return {
        "status": (
            "candidate_space_has_unrealized_signal_diagnostic"
            if has_unrealized_signal
            else "candidate_space_ceiling_not_better_than_r54"
        ),
        "claim_status": "diagnostic_only",
        "blockers": ["uses_same_split_holdout_targets", "not_train_origin_selectable"],
        "ceiling_family": ceiling_family,
        "ceiling_mean_regional_normalized_absolute_error": ceiling_regional,
        "ceiling_mean_aggregate_mass_normalized_absolute_error": ceiling_mass,
        "ceiling_mean_regional_share_half_l1_error": ceiling_share,
        "r54_reference_mean_regional_normalized_absolute_error": r54_regional,
        "r54_reference_mean_aggregate_mass_normalized_absolute_error": r54_mass,
        "r54_reference_mean_regional_share_half_l1_error": r54_share,
        "improvement": improvement,
        "contract": (
            "R57 is an intentionally leakage-labeled ceiling diagnostic. It may reveal whether the existing "
            "regional candidate family contains unrealized signal, but it cannot promote a model because it "
            "selects candidates using same-split holdout errors."
        ),
    }


def _selection_summary(selection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter(
        (str(row.get("metric_name") or ""), str(row.get("selected_candidate_family") or "")) for row in selection_rows
    )
    return [
        {
            "metric_name": metric,
            "selected_candidate_family": family,
            "selected_period_count": count,
        }
        for (metric, family), count in sorted(counter.items(), key=lambda item: (item[0][0], -item[1], item[0][1]))
    ]


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
    gate = dict(report.get("ceiling_gate") or {})
    improvement = dict(gate.get("improvement") or {})
    lines = [
        "# Phase 3 R57 Regional Candidate Ceiling Diagnostic",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Claim status: `{gate.get('claim_status')}`",
        f"- Ceiling regional / mass / share errors: `{gate.get('ceiling_mean_regional_normalized_absolute_error')}` / `{gate.get('ceiling_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('ceiling_mean_regional_share_half_l1_error')}`",
        f"- R54 regional / mass / share errors: `{gate.get('r54_reference_mean_regional_normalized_absolute_error')}` / `{gate.get('r54_reference_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('r54_reference_mean_regional_share_half_l1_error')}`",
        f"- Delta versus R54 regional / mass / share: `{improvement.get('regional_error_delta_vs_r54')}` / `{improvement.get('mass_error_delta_vs_r54')}` / `{improvement.get('share_error_delta_vs_r54')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Selection Summary",
        "",
        "| Metric | Selected family | Period count |",
        "|---|---|---:|",
    ]
    for row in list(report.get("selection_summary_rows") or []):
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('selected_candidate_family')}` | {row.get('selected_period_count')} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, combined_rows: list[dict[str, Any]], r54_gate: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    rows = [
        _family_row(combined_rows, CEILING_FAMILY),
        {
            "candidate_family": "r54_reference",
            "mean_regional_normalized_absolute_error": r54_gate.get("promoted_mean_regional_normalized_absolute_error"),
            "mean_aggregate_mass_normalized_absolute_error": r54_gate.get("promoted_mean_aggregate_mass_normalized_absolute_error"),
            "mean_regional_share_half_l1_error": r54_gate.get("promoted_mean_regional_share_half_l1_error"),
        },
    ]
    labels = [str(row.get("candidate_family") or "") for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    regional = [float(row.get("mean_regional_normalized_absolute_error") or 0.0) for row in rows]
    mass = [float(row.get("mean_aggregate_mass_normalized_absolute_error") or 0.0) for row in rows]
    share = [float(row.get("mean_regional_share_half_l1_error") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - width, regional, width, label="regional NAE", color="#516b5f")
    ax.bar(x, mass, width, label="aggregate mass NAE", color="#496f9e")
    ax.bar(x + width, share, width, label="regional share half-L1", color="#8a6f2a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("error")
    ax.set_title("R57 leakage-labeled regional candidate ceiling")
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r57_regional_candidate_ceiling_diagnostic(
    *,
    run_id: str = R57_RUN_ID,
    r54_report_path: Path | None = None,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
    r51_report_path: Path | None = None,
) -> dict[str, Any]:
    r54_path = Path(r54_report_path) if r54_report_path is not None else R54_DEFAULT_REPORT
    r54 = _read_report(r54_path)
    r48_path = Path(r48_report_path or r54.get("r48_report_path") or R48_DEFAULT_REPORT)
    r49_path = Path(r49_report_path or r54.get("r49_report_path") or R49_DEFAULT_REPORT)
    r51_path = Path(r51_report_path or r54.get("r51_report_path") or R51_DEFAULT_REPORT)
    r48 = _load_report(r48_path)
    r49 = _load_report(r49_path)
    r51 = _load_report(r51_path)
    r44_path = Path(str(r54.get("r44_report_path") or r48.get("r44_report_path") or r51.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    prediction_rows = _base_prediction_rows(r48, r49, r51) + [dict(row) for row in list(r54.get("adapter_prediction_rows") or [])]
    selection_rows = _ceiling_selection_rows(
        prediction_rows=prediction_rows,
        regional_score_rows=[dict(row) for row in list(r54.get("split_metric_score_rows") or [])],
        coherence_rows=[dict(row) for row in list(r54.get("coherence_rows") or [])],
    )
    ceiling_rows = _ceiling_prediction_rows(prediction_rows, selection_rows)
    split_rows, coherence_rows, combined_rows = _score_ceiling_rows(rows_by_key, ceiling_rows)
    gate = _gate(combined_rows, r54_gate=dict(r54.get("adapter_gate") or {}))
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r57_regional_candidate_ceiling_diagnostic_report.json"
    md_path = analysis_dir / "r57_regional_candidate_ceiling_diagnostic_report.md"
    selection_csv = analysis_dir / "r57_selection_rows.csv"
    prediction_csv = analysis_dir / "r57_prediction_rows.csv"
    split_csv = analysis_dir / "r57_split_metric_scores.csv"
    coherence_csv = analysis_dir / "r57_coherence_rows.csv"
    combined_csv = analysis_dir / "r57_combined_candidate_table.csv"
    dashboard_path = analysis_dir / "r57_regional_candidate_ceiling_diagnostic_dashboard.png"
    report = {
        "schema_version": R57_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": (
            "R57 found unrealized signal inside the existing regional candidate family, but this is diagnostic-only because selection used same-split holdout errors."
            if gate["status"] == "candidate_space_has_unrealized_signal_diagnostic"
            else "R57 found that the existing regional candidate family ceiling does not improve over R54."
        ),
        "r44_report_path": r44_path.as_posix(),
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r54_report_path": r54_path.as_posix(),
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "leakage_status": "same-split target errors are used; diagnostic ceiling only",
            "claim_limit": "not promotable and not a forecast model; use only to diagnose whether selection or candidate family is the bottleneck",
        },
        "ceiling_gate": gate,
        "selection_summary_rows": _selection_summary(selection_rows),
        "selection_rows": selection_rows,
        "prediction_rows": ceiling_rows,
        "split_metric_score_rows": split_rows,
        "coherence_rows": coherence_rows,
        "combined_candidate_table": combined_rows,
        "r54_gate_snapshot": dict(r54.get("adapter_gate") or {}),
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "selection_csv": selection_csv.as_posix(),
            "prediction_csv": prediction_csv.as_posix(),
            "split_metric_scores_csv": split_csv.as_posix(),
            "coherence_rows_csv": coherence_csv.as_posix(),
            "combined_candidate_csv": combined_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(selection_csv, selection_rows)
    _write_csv(prediction_csv, ceiling_rows)
    _write_csv(split_csv, split_rows)
    _write_csv(coherence_csv, coherence_rows)
    _write_csv(combined_csv, combined_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, combined_rows, dict(r54.get("adapter_gate") or {}))
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R57 regional candidate ceiling diagnostic.")
    parser.add_argument("--run-id", default=R57_RUN_ID)
    parser.add_argument("--r54-report-path", default=None)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    args = parser.parse_args()
    run_r57_regional_candidate_ceiling_diagnostic(
        run_id=str(args.run_id),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
    )


if __name__ == "__main__":
    _main()
