from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .r11_sparse_state_space import (
    OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS,
    _finite_float,
    _generated_at,
    _sha256,
)
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r75_bulk_unaids_annual_challenge import (
    _annual_target_scale,
    _bulk_unaids_target_rows,
    _merge_external_targets_into_observations,
    _r69_annual_path,
    _write_csv,
)
from .r78_public_annual_family_expansion import (
    R78_RUN_ID,
    R78_SELECTED_FAMILY,
    _select_metric_families,
    _selected_prediction_rows,
)
from .runtime import ensure_dir, read_json, write_json


R80_SCHEMA_VERSION = "phase3_dynamic.r80_public_annual_projection_head.v1"
R80_RUN_ID = "p3d-r80-public-annual-projection-head-20260507-s00"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)
R78_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R78_RUN_ID
    / "analysis"
    / "r78_public_annual_family_expansion_report.json"
)
INTERVAL_QUANTILES: tuple[float, ...] = (0.5, 0.8, 0.95)


def _horizon_bucket(horizon_years: int) -> int:
    if int(horizon_years) <= 1:
        return 1
    if int(horizon_years) <= 3:
        return 3
    return 5


def _residual_quantiles(
    score_rows: list[dict[str, Any]],
    *,
    metric_name: str,
    horizon_bucket: int,
    quantiles: tuple[float, ...] = INTERVAL_QUANTILES,
) -> dict[str, float]:
    exact = [
        abs(float(row["candidate_norm_error"]))
        for row in score_rows
        if str(row.get("candidate_family") or "") == R78_SELECTED_FAMILY
        and str(row.get("metric_name") or "") == str(metric_name)
        and int(row.get("horizon_years") or 0) == int(horizon_bucket)
        and _finite_float(row.get("candidate_norm_error")) is not None
    ]
    fallback = [
        abs(float(row["candidate_norm_error"]))
        for row in score_rows
        if str(row.get("candidate_family") or "") == R78_SELECTED_FAMILY
        and str(row.get("metric_name") or "") == str(metric_name)
        and _finite_float(row.get("candidate_norm_error")) is not None
    ]
    values = np.asarray(exact or fallback or [0.0], dtype=np.float64)
    return {f"q{int(round(q * 100))}_norm_abs_error": float(np.quantile(values, q)) for q in quantiles}


def _projection_rows(
    panel_rows: list[dict[str, Any]],
    r78_score_rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    selected = _select_metric_families(panel_rows, min_train_years=min_train_years, horizons=horizons)
    future_rows = [{"quarter": f"{year}-Q4"} for year in range(int(start_year), int(end_year) + 1)]
    predictions = _selected_prediction_rows(panel_rows, future_rows, selected)
    prediction_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in predictions}
    last_public_year = max(int(str(row.get("quarter") or "0-Q4").split("-")[0]) for row in panel_rows if row.get("quarter"))
    rows: list[dict[str, Any]] = []
    for year in range(int(start_year), int(end_year) + 1):
        quarter = f"{year}-Q4"
        horizon = int(year - last_public_year)
        bucket = _horizon_bucket(horizon)
        prediction = prediction_by_quarter.get(quarter, {})
        for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
            value = _finite_float(prediction.get(metric))
            scale = _annual_target_scale(panel_rows, metric)
            quantiles_by_name = _residual_quantiles(r78_score_rows, metric_name=metric, horizon_bucket=bucket)
            q50 = quantiles_by_name["q50_norm_abs_error"] * scale
            q80 = quantiles_by_name["q80_norm_abs_error"] * scale
            q95 = quantiles_by_name["q95_norm_abs_error"] * scale
            point = None if value is None else max(float(value), 0.0)
            rows.append(
                {
                    "year": int(year),
                    "quarter": quarter,
                    "metric_name": metric,
                    "point_prediction": point,
                    "p50_lower": None if point is None else max(float(point) - q50, 0.0),
                    "p50_upper": None if point is None else max(float(point) + q50, 0.0),
                    "p80_lower": None if point is None else max(float(point) - q80, 0.0),
                    "p80_upper": None if point is None else max(float(point) + q80, 0.0),
                    "p95_lower": None if point is None else max(float(point) - q95, 0.0),
                    "p95_upper": None if point is None else max(float(point) + q95, 0.0),
                    "horizon_years_from_last_public_target": horizon,
                    "horizon_bucket": bucket,
                    "long_horizon_extrapolation": bool(horizon > max(horizons)),
                    "selected_public_family": selected.get(metric),
                    "uncertainty_source": "R78 blocked-time selected-proxy normalized residual quantiles",
                    **quantiles_by_name,
                }
            )
    return rows, selected


def _mass_balance_rows(projection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_year_metric = {
        (int(row.get("year") or 0), str(row.get("metric_name") or "")): _finite_float(row.get("point_prediction"))
        for row in projection_rows
    }
    years = sorted({int(row.get("year") or 0) for row in projection_rows})
    rows: list[dict[str, Any]] = []
    previous_plhiv: float | None = None
    for year in years:
        incidence = by_year_metric.get((year, "annual_new_infections"))
        deaths = by_year_metric.get((year, "annual_aids_deaths"))
        plhiv = by_year_metric.get((year, "estimated_plhiv"))
        implied_next = None if previous_plhiv is None or incidence is None or deaths is None else float(previous_plhiv + incidence - deaths)
        rows.append(
            {
                "year": year,
                "estimated_plhiv": plhiv,
                "annual_new_infections": incidence,
                "annual_aids_deaths": deaths,
                "previous_plhiv_plus_infections_minus_aids_deaths": implied_next,
                "reported_minus_simple_aids_death_balance": None if implied_next is None or plhiv is None else float(plhiv - implied_next),
                "balance_note": "diagnostic_only: non-AIDS mortality, migration, and estimate-process changes are not identified by public annual targets alone",
            }
        )
        previous_plhiv = plhiv
    return rows


def _gate(projection_rows: list[dict[str, Any]], r78: dict[str, Any], selected: dict[str, str]) -> dict[str, Any]:
    r78_status = str(dict(r78.get("expanded_public_annual_gate") or {}).get("status") or "")
    blockers: list[str] = []
    if r78_status != "expanded_public_annual_comparator_promoted":
        blockers.append("r78_expanded_public_comparator_not_promoted")
    if not projection_rows:
        blockers.append("no_projection_rows")
    if set(selected) != set(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS):
        blockers.append("missing_required_metric_selection")
    return {
        "status": "public_annual_projection_head_ready" if not blockers else "public_annual_projection_head_diagnostic_only",
        "blockers": blockers,
        "projection_row_count": len(projection_rows),
        "selected_metric_families": selected,
        "contract": (
            "R80 emits public-domain AEM/Spectrum-style annual projections for incidence, AIDS deaths, and PLHIV. "
            "It uses only public annual target history, R78 train-origin family selection, and uncertainty bands "
            "derived from R78 blocked-time residuals. It is not official AEM/Spectrum output and does not train "
            "quarterly cascade dynamics."
        ),
    }


def run_r80_public_annual_projection_head(
    *,
    run_id: str = R80_RUN_ID,
    r69_report_path: Path | None = None,
    r78_report_path: Path | None = None,
    external_start_year: int = 2010,
    projection_start_year: int = 2025,
    projection_end_year: int = 2035,
    min_train_years: int = 5,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r69_path = R69_DEFAULT_REPORT if r69_report_path is None else Path(r69_report_path)
    r78_path = R78_DEFAULT_REPORT if r78_report_path is None else Path(r78_report_path)
    r69 = dict(read_json(r69_path, default={}) or {}) if r69_path.exists() else {}
    r78 = dict(read_json(r78_path, default={}) or {}) if r78_path.exists() else {}
    annual_csv = _r69_annual_path(r69)
    target_rows = [] if annual_csv is None else _bulk_unaids_target_rows(annual_csv, external_start_year=external_start_year)
    panel_rows = _merge_external_targets_into_observations([], target_rows)
    projection_rows, selected = _projection_rows(
        panel_rows,
        list(r78.get("score_rows") or []),
        start_year=projection_start_year,
        end_year=projection_end_year,
        min_train_years=min_train_years,
        horizons=horizons,
    )
    balance_rows = _mass_balance_rows(projection_rows)
    gate = _gate(projection_rows, r78, selected)
    report_path = analysis_dir / "r80_public_annual_projection_head_report.json"
    markdown_path = analysis_dir / "r80_public_annual_projection_head_report.md"
    projection_csv = analysis_dir / "r80_projection_rows.csv"
    balance_csv = analysis_dir / "r80_mass_balance_rows.csv"
    figure_path = analysis_dir / "r80_public_annual_projection_fanchart.png"
    report = {
        "schema_version": R80_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "public_annual_projection_gate": gate,
        "projection_start_year": int(projection_start_year),
        "projection_end_year": int(projection_end_year),
        "target_rows": target_rows,
        "projection_rows": projection_rows,
        "mass_balance_rows": balance_rows,
        "source_artifacts": {
            "r69": {"path": r69_path.as_posix(), "sha256": _sha256(r69_path) if r69_path.exists() else None},
            "r78": {"path": r78_path.as_posix(), "sha256": _sha256(r78_path) if r78_path.exists() else None},
            "annual_external_challenge_csv": None if annual_csv is None else annual_csv.as_posix(),
            "annual_external_challenge_csv_sha256": None if annual_csv is None or not annual_csv.exists() else _sha256(annual_csv),
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "projection_rows_csv": projection_csv.as_posix(),
            "mass_balance_rows_csv": balance_csv.as_posix(),
            "projection_fanchart_png": figure_path.as_posix(),
        },
    }
    _write_csv(projection_csv, projection_rows)
    _write_csv(balance_csv, balance_rows)
    _write_markdown(markdown_path, report)
    _write_projection_figure(figure_path, target_rows, projection_rows)
    write_json(report_path, report)
    return report


def _write_projection_figure(path: Path, target_rows: list[dict[str, Any]], projection_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    ensure_dir(path.parent)
    label_by_metric = {
        "annual_new_infections": "Annual new infections",
        "annual_aids_deaths": "Annual AIDS deaths",
        "estimated_plhiv": "Estimated PLHIV",
    }
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, metric in zip(axes, OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS, strict=True):
        hist = [row for row in target_rows if str(row.get("metric_name") or "") == metric]
        proj = [row for row in projection_rows if str(row.get("metric_name") or "") == metric]
        hist_years = [int(row["year"]) for row in hist]
        hist_values = [float(row["target_value"]) for row in hist]
        years = [int(row["year"]) for row in proj]
        point = [float(row["point_prediction"]) for row in proj]
        p80_lower = [float(row["p80_lower"]) for row in proj]
        p80_upper = [float(row["p80_upper"]) for row in proj]
        p95_lower = [float(row["p95_lower"]) for row in proj]
        p95_upper = [float(row["p95_upper"]) for row in proj]
        ax.plot(hist_years, hist_values, color="#1f2933", linewidth=1.8, label="public annual target")
        ax.plot(years, point, color="#b42318", linewidth=1.8, label="R80 projection")
        ax.fill_between(years, p95_lower, p95_upper, color="#f6b7ae", alpha=0.35, label="95% residual band")
        ax.fill_between(years, p80_lower, p80_upper, color="#e85d4f", alpha=0.25, label="80% residual band")
        ax.set_title(label_by_metric[metric])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Year")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("R80 public-domain annual HIV projection head, Philippines", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("public_annual_projection_gate") or {})
    lines = [
        "# Phase 3 R80 Public Annual Projection Head",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Projection years: `{report.get('projection_start_year')}-{report.get('projection_end_year')}`",
        f"- Projection rows: `{gate.get('projection_row_count')}`",
        f"- Selected metric families: `{gate.get('selected_metric_families')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## 2030 Snapshot",
        "",
        "| Metric | Point | 80% lower | 80% upper | 95% lower | 95% upper |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("projection_rows") or []:
        if int(row.get("year") or 0) != 2030:
            continue
        lines.append(
            f"| `{row.get('metric_name')}` | {float(row.get('point_prediction') or 0.0):.3f} | "
            f"{float(row.get('p80_lower') or 0.0):.3f} | {float(row.get('p80_upper') or 0.0):.3f} | "
            f"{float(row.get('p95_lower') or 0.0):.3f} | {float(row.get('p95_upper') or 0.0):.3f} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R80 public annual projection head.")
    parser.add_argument("--run-id", default=R80_RUN_ID)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument("--r78-report-path", default=None)
    parser.add_argument("--external-start-year", type=int, default=2010)
    parser.add_argument("--projection-start-year", type=int, default=2025)
    parser.add_argument("--projection-end-year", type=int, default=2035)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--horizon", type=int, action="append", default=None)
    args = parser.parse_args()
    run_r80_public_annual_projection_head(
        run_id=str(args.run_id),
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
        r78_report_path=None if args.r78_report_path is None else Path(args.r78_report_path),
        external_start_year=int(args.external_start_year),
        projection_start_year=int(args.projection_start_year),
        projection_end_year=int(args.projection_end_year),
        min_train_years=int(args.min_train_years),
        horizons=tuple(int(value) for value in (args.horizon or [1, 3, 5])),
    )


if __name__ == "__main__":
    _main()
