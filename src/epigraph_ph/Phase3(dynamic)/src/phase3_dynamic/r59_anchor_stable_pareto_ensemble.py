from __future__ import annotations

import argparse
import csv
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
from .r55_adapter_split_stability_gate import (
    _comparison_rows as _split_comparison_rows,
    _gate as _split_stability_gate,
)
from .r58_pareto_simplex_regional_ensemble import (
    CARRY_FORWARD_FAMILY,
    REFERENCE_FAMILY,
    _fit_simplex_weights,
    _gate as _r58_gate,
    _pareto_candidate_families,
    _prediction_index,
    _write_csv,
)
from .runtime import ensure_dir, read_json, write_json


R59_SCHEMA_VERSION = "phase3_dynamic.r59_anchor_stable_pareto_ensemble.v1"
R59_RUN_ID = "p3d-r59-anchor-stable-pareto-ensemble-20260503-s00"
ENSEMBLE_FAMILY = "anchor_stable_pareto_ensemble"
ANCHOR_METRICS = ("estimated_plhiv",)
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


def _periods(prediction_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("holdout_period") or "") for row in prediction_rows if row.get("holdout_period")}, key=quarter_sort_key)


def _default_weights(families: list[str], default_family: str) -> np.ndarray:
    weights = np.zeros(len(families), dtype=np.float64)
    if not families:
        return weights
    if default_family in families:
        weights[families.index(default_family)] = 1.0
    else:
        weights[0] = 1.0
    return weights


def _available_families(
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    *,
    period: str,
    regions: list[str],
    families: list[str],
    metric: str,
) -> list[str]:
    return [
        family
        for family in families
        if all((family, period, region) in indexed and _finite_float(indexed[(family, period, region)].get(metric)) is not None for region in regions)
    ]


def _prior_design(
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    prior_periods: list[str],
    regions: list[str],
    families: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    design_rows: list[list[float]] = []
    target_values: list[float] = []
    for period in prior_periods:
        for region in regions:
            target = rows_by_key.get((period, region))
            target_value = _finite_float((target or {}).get(metric))
            if target_value is None:
                continue
            values: list[float] = []
            complete = True
            for family in families:
                value = _finite_float((indexed.get((family, period, region)) or {}).get(metric))
                if value is None:
                    complete = False
                    break
                values.append(float(value))
            if complete:
                design_rows.append(values)
                target_values.append(float(target_value))
    return (
        np.asarray(design_rows, dtype=np.float64).reshape((len(design_rows), len(families))),
        np.asarray(target_values, dtype=np.float64),
    )


def _anchor_stable_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    pareto_families: list[str],
    default_family: str,
    anchor_family: str = REFERENCE_FAMILY,
    anchor_metrics: tuple[str, ...] = ANCHOR_METRICS,
    ensemble_family: str = ENSEMBLE_FAMILY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = _prediction_index(prediction_rows)
    periods = _periods(prediction_rows)
    output_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for period in periods:
        regions = sorted({region for family, candidate_period, region in indexed if family == anchor_family and candidate_period == period})
        if not regions:
            regions = sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        period_rows: dict[str, dict[str, Any]] = {
            region: {"candidate_family": ensemble_family, "holdout_period": period, "region": region}
            for region in regions
        }
        prior_periods = [prior for prior in periods if quarter_sort_key(prior) < quarter_sort_key(period)]
        for metric in COUNT_METRICS:
            if metric in anchor_metrics:
                for region in regions:
                    value = _finite_float((indexed.get((anchor_family, period, region)) or {}).get(metric))
                    if value is not None:
                        period_rows[region][metric] = float(value)
                        period_rows[region][f"{metric}_source_family"] = anchor_family
                weight_rows.append(
                    {
                        "holdout_period": period,
                        "metric_name": metric,
                        "weight_status": "anchor_metric_reference_family",
                        "anchor_family": anchor_family,
                        "available_candidate_families": [anchor_family],
                        "weight_by_candidate_family": {anchor_family: 1.0},
                        "prior_period_count": len(prior_periods),
                        "prior_design_row_count": None,
                        "candidate_count": 1,
                    }
                )
                continue
            available = _available_families(
                indexed,
                period=period,
                regions=regions,
                families=pareto_families,
                metric=metric,
            )
            weights = _default_weights(available, default_family)
            design, target = _prior_design(
                indexed,
                rows_by_key,
                prior_periods=prior_periods,
                regions=regions,
                families=available,
                metric=metric,
            )
            fitted = False
            if available and design.shape[0] >= design.shape[1] and design.shape[1] > 0:
                weights = _fit_simplex_weights(design, target)
                fitted = True
            weight_payload = {family: float(weights[index]) for index, family in enumerate(available)}
            weight_rows.append(
                {
                    "holdout_period": period,
                    "metric_name": metric,
                    "weight_status": "simplex_fitted" if fitted else "default_reference_weight",
                    "available_candidate_families": available,
                    "weight_by_candidate_family": weight_payload,
                    "prior_period_count": len(prior_periods),
                    "prior_design_row_count": int(design.shape[0]),
                    "candidate_count": len(available),
                }
            )
            for region in regions:
                value = 0.0
                for index, family in enumerate(available):
                    metric_value = _finite_float((indexed.get((family, period, region)) or {}).get(metric))
                    if metric_value is not None:
                        value += float(weights[index]) * float(metric_value)
                period_rows[region][metric] = float(value)
                period_rows[region][f"{metric}_weight_by_candidate_family"] = weight_payload
        for region in regions:
            raw = period_rows[region]
            projected = _projection_cascade(raw)
            projected["projection_adjusted"] = any(
                abs(float(projected.get(metric) or 0.0) - float(raw.get(metric) or 0.0)) > 1e-8
                for metric in COUNT_METRICS
            )
            output_rows.append(projected)
    return output_rows, weight_rows


def _score_rows(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _raw, split_rows = _score_predictions(rows_by_key, prediction_rows)
    coherence = _coherence_rows(rows_by_key, prediction_rows)
    combined = _combined_table(_regional_summary_table(split_rows), _coherence_summary_table(coherence))
    return split_rows, coherence, combined


def _gate(
    combined_rows: list[dict[str, Any]],
    *,
    split_gate: dict[str, Any],
    r54_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate = _r58_gate(
        combined_rows,
        split_gate=split_gate,
        r54_gate=r54_gate,
        ensemble_family=ENSEMBLE_FAMILY,
    )
    gate["contract"] = (
        "R59 anchors stock metrics with direct support semantics to the module-local reference family, then applies "
        "the R58 Pareto-simplex train-origin ensemble only to non-anchor cascade streams. Mean promotion uses the "
        "same carry-forward/R52/R54 nonregression contract; strict promotion still requires split stability."
    )
    if gate["status"] == "pareto_simplex_regional_ensemble_promoted":
        gate["status"] = "anchor_stable_pareto_ensemble_promoted"
    elif gate["status"] == "pareto_simplex_regional_ensemble_mean_promoted_split_limited":
        gate["status"] = "anchor_stable_pareto_ensemble_mean_promoted_split_limited"
    elif gate["status"] == "pareto_simplex_regional_ensemble_diagnostic_only":
        gate["status"] = "anchor_stable_pareto_ensemble_diagnostic_only"
    return gate


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("ensemble_gate") or {})
    split_gate = dict(report.get("split_stability_gate") or {})
    lines = [
        "# Phase 3 R59 Anchor-Stable Pareto Ensemble",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Ensemble regional / mass / share errors: `{gate.get('ensemble_mean_regional_normalized_absolute_error')}` / `{gate.get('ensemble_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('ensemble_mean_regional_share_half_l1_error')}`",
        f"- R54 regional / mass / share errors: `{gate.get('r54_reference_mean_regional_normalized_absolute_error')}` / `{gate.get('r54_reference_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('r54_reference_mean_regional_share_half_l1_error')}`",
        f"- Anchor metrics: `{', '.join(report.get('metric_contract', {}).get('anchor_metrics') or [])}`",
        f"- Split-stability status: `{split_gate.get('status')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Pareto Families",
        "",
    ]
    for family in list(report.get("pareto_candidate_families") or []):
        lines.append(f"- `{family}`")
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, combined_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    keep = {
        ENSEMBLE_FAMILY,
        CARRY_FORWARD_FAMILY,
        REFERENCE_FAMILY,
    }
    rows = [
        row
        for row in combined_rows
        if str(row.get("candidate_family") or "") in keep
        or str(row.get("candidate_family") or "").startswith(
            "national_total_adapter__total=train_only_region_metric_selector__share=similarity_proxy_log_delta"
        )
    ]
    labels = [str(row.get("candidate_family") or "") for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    regional = [float(row.get("mean_regional_normalized_absolute_error") or 0.0) for row in rows]
    mass = [float(row.get("mean_aggregate_mass_normalized_absolute_error") or 0.0) for row in rows]
    share = [float(row.get("mean_regional_share_half_l1_error") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.bar(x - width, regional, width, label="regional NAE", color="#516b5f")
    ax.bar(x, mass, width, label="aggregate mass NAE", color="#496f9e")
    ax.bar(x + width, share, width, label="regional share half-L1", color="#8a6f2a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("error")
    ax.set_title("R59 anchor-stable Pareto regional ensemble")
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r59_anchor_stable_pareto_ensemble(
    *,
    run_id: str = R59_RUN_ID,
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
    pareto_families = _pareto_candidate_families([dict(row) for row in list(r54.get("combined_candidate_table") or [])])
    default_family = str((r54.get("adapter_gate") or {}).get("promoted_candidate_family") or (pareto_families[0] if pareto_families else ""))
    prediction_rows_out, weight_rows = _anchor_stable_prediction_rows(
        prediction_rows,
        rows_by_key,
        pareto_families=pareto_families,
        default_family=default_family,
    )
    split_rows, coherence_rows, _combined = _score_rows(rows_by_key, prediction_rows_out)
    comparison_split_rows = [dict(row) for row in list(r54.get("split_metric_score_rows") or [])] + split_rows
    comparison_coherence_rows = [dict(row) for row in list(r54.get("coherence_rows") or [])] + coherence_rows
    split_comparison = _split_comparison_rows(
        regional_score_rows=comparison_split_rows,
        coherence_rows=comparison_coherence_rows,
        promoted_family=ENSEMBLE_FAMILY,
    )
    split_gate = _split_stability_gate(split_comparison)
    combined_rows = _combined_table(
        _regional_summary_table(comparison_split_rows),
        _coherence_summary_table(comparison_coherence_rows),
    )
    gate = _gate(combined_rows, split_gate=split_gate, r54_gate=dict(r54.get("adapter_gate") or {}))
    verdict = (
        "R59 promoted an anchor-stable Pareto ensemble with strict split stability."
        if gate["status"] == "anchor_stable_pareto_ensemble_promoted"
        else (
            "R59 improved mean regional/mass/share errors but remains split-limited."
            if gate["status"] == "anchor_stable_pareto_ensemble_mean_promoted_split_limited"
            else "R59 did not beat the current regional references."
        )
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r59_anchor_stable_pareto_ensemble_report.json"
    md_path = analysis_dir / "r59_anchor_stable_pareto_ensemble_report.md"
    weight_csv = analysis_dir / "r59_weight_rows.csv"
    prediction_csv = analysis_dir / "r59_prediction_rows.csv"
    split_csv = analysis_dir / "r59_split_metric_scores.csv"
    coherence_csv = analysis_dir / "r59_coherence_rows.csv"
    comparison_csv = analysis_dir / "r59_split_comparison_rows.csv"
    combined_csv = analysis_dir / "r59_combined_candidate_table.csv"
    dashboard_path = analysis_dir / "r59_anchor_stable_pareto_ensemble_dashboard.png"
    report = {
        "schema_version": R59_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r44_report_path": r44_path.as_posix(),
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r54_report_path": r54_path.as_posix(),
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "anchor_metrics": list(ANCHOR_METRICS),
            "anchor_family": REFERENCE_FAMILY,
            "pareto_rule": "non-anchor streams use the R54 Pareto frontier over mean regional, mass, and share errors",
            "weight_rule": "non-anchor stream weights are fitted from earlier same-metric regional rows only",
            "target_leakage_status": "no same-period target error enters anchor or ensemble weights",
            "claim_limit": "regional readout ensemble only; no determinant, province-level, or causal transition claim",
        },
        "pareto_candidate_families": pareto_families,
        "ensemble_gate": gate,
        "split_stability_gate": split_gate,
        "weight_rows": weight_rows,
        "prediction_rows": prediction_rows_out,
        "split_metric_score_rows": split_rows,
        "coherence_rows": coherence_rows,
        "split_comparison_rows": split_comparison,
        "combined_candidate_table": combined_rows,
        "r54_gate_snapshot": dict(r54.get("adapter_gate") or {}),
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "weight_csv": weight_csv.as_posix(),
            "prediction_csv": prediction_csv.as_posix(),
            "split_metric_scores_csv": split_csv.as_posix(),
            "coherence_rows_csv": coherence_csv.as_posix(),
            "split_comparison_csv": comparison_csv.as_posix(),
            "combined_candidate_csv": combined_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(weight_csv, weight_rows)
    _write_csv(prediction_csv, prediction_rows_out)
    _write_csv(split_csv, split_rows)
    _write_csv(coherence_csv, coherence_rows)
    _write_csv(comparison_csv, split_comparison)
    _write_csv(combined_csv, combined_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, combined_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R59 anchor-stable Pareto ensemble.")
    parser.add_argument("--run-id", default=R59_RUN_ID)
    parser.add_argument("--r54-report-path", default=None)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    args = parser.parse_args()
    run_r59_anchor_stable_pareto_ensemble(
        run_id=str(args.run_id),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
    )


if __name__ == "__main__":
    _main()
