from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import quarter_sort_key
from .r11_sparse_state_space import _generated_at
from .runtime import ensure_dir, read_json, write_json
from .data import sandbox_repo_root


R48_SCHEMA_VERSION = "phase3_dynamic.r48_subnational_proxy_champion_contract.v1"
R48_RUN_ID = "p3d-r48-subnational-proxy-champion-contract-20260503-s00"
R44_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r44-subnational-horizon-gate-20260502-s00"
    / "analysis"
    / "r44_subnational_horizon_gate_report.json"
)

COUNT_METRICS: tuple[str, ...] = (
    "estimated_plhiv",
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
)

RATE_FEATURES: tuple[str, ...] = (
    "first_95",
    "second_95",
    "vl_testing_coverage",
    "suppression_among_tested",
    "third_95",
)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if not math.isfinite(result):
        return None
    return result


def _load_r44_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = dict(read_json(path, default={}) or {})
    rows = [dict(row) for row in list(report.get("regional_cascade_rows") or [])]
    return rows, report


def _periods(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("period_id") or "") for row in rows if row.get("period_id")}, key=quarter_sort_key)


def _rows_by_period_region(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("period_id") or ""), str(row.get("region") or "")): dict(row)
        for row in rows
        if row.get("period_id") and row.get("region")
    }


def _overlapping_regions(rows: list[dict[str, Any]], periods: list[str]) -> list[str]:
    regions_by_period: list[set[str]] = []
    for period in periods:
        regions_by_period.append({str(row.get("region") or "") for row in rows if str(row.get("period_id") or "") == period})
    return sorted(set.intersection(*regions_by_period)) if regions_by_period else []


def _feature_row(row: dict[str, Any], aggregate_by_metric: dict[str, float]) -> dict[str, float]:
    features: dict[str, float] = {}
    for metric in COUNT_METRICS:
        value = max(float(row.get(metric) or 0.0), 0.0)
        aggregate = max(float(aggregate_by_metric.get(metric) or 0.0), float(np.finfo(np.float32).eps))
        features[f"log1p_{metric}"] = float(np.log1p(value))
        features[f"share_{metric}"] = float(value / aggregate)
    for metric in RATE_FEATURES:
        value = _finite_float(row.get(metric))
        if value is not None:
            features[metric] = float(value)
    return features


def _aggregate_by_metric(period_rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: float(sum(max(float(row.get(metric) or 0.0), 0.0) for row in period_rows))
        for metric in COUNT_METRICS
    }


def _standardized_feature_matrix(feature_rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    feature_names = sorted({key for row in feature_rows for key in row.keys()})
    if not feature_names:
        return [], np.zeros((len(feature_rows), 0), dtype=np.float64)
    matrix = np.asarray(
        [[float(row.get(name) or 0.0) for name in feature_names] for row in feature_rows],
        dtype=np.float64,
    )
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales = np.where(scales <= 0.0, 1.0, scales)
    return feature_names, (matrix - means) / scales


def _similarity_kernel(feature_matrix: np.ndarray, regions: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(regions)
    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            distances[i, j] = float(np.linalg.norm(feature_matrix[i] - feature_matrix[j])) if feature_matrix.size else 0.0
    positive = distances[distances > 0.0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    if bandwidth <= 0.0:
        bandwidth = 1.0
    weights = np.exp(-0.5 * np.square(distances / bandwidth))
    np.fill_diagonal(weights, 0.0)
    row_sums = weights.sum(axis=1)
    for i in range(n):
        if row_sums[i] > 0.0:
            weights[i, :] = weights[i, :] / row_sums[i]
        elif n > 1:
            weights[i, :] = 1.0 / float(n - 1)
            weights[i, i] = 0.0
    return distances, weights, bandwidth


def _projection_cascade(row: dict[str, Any]) -> dict[str, Any]:
    projected = dict(row)
    estimated = max(float(projected.get("estimated_plhiv") or 0.0), 0.0)
    diagnosed = max(float(projected.get("diagnosed_plhiv") or 0.0), 0.0)
    art = max(float(projected.get("alive_on_art") or 0.0), 0.0)
    tested = max(float(projected.get("tested_for_viral_load") or 0.0), 0.0)
    suppressed = max(float(projected.get("virally_suppressed") or 0.0), 0.0)
    estimated = max(estimated, diagnosed)
    diagnosed = min(diagnosed, estimated)
    art = min(art, diagnosed)
    tested = min(tested, art)
    suppressed = min(suppressed, tested)
    projected.update(
        {
            "estimated_plhiv": float(estimated),
            "diagnosed_plhiv": float(diagnosed),
            "alive_on_art": float(art),
            "tested_for_viral_load": float(tested),
            "virally_suppressed": float(suppressed),
        }
    )
    return projected


def _region_features_for_train_period(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    period: str,
    regions: list[str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, float, list[str]]:
    period_rows = [rows_by_key[(period, region)] for region in regions if (period, region) in rows_by_key]
    aggregate = _aggregate_by_metric(period_rows)
    feature_payload_rows: list[dict[str, Any]] = []
    feature_values: list[dict[str, float]] = []
    present_regions: list[str] = []
    for region in regions:
        row = rows_by_key.get((period, region))
        if row is None:
            continue
        features = _feature_row(row, aggregate)
        present_regions.append(region)
        feature_values.append(features)
        feature_payload_rows.append(
            {
                "period_id": period,
                "region": region,
                **features,
            }
        )
    _feature_names, standardized = _standardized_feature_matrix(feature_values)
    distances, weights, bandwidth = _similarity_kernel(standardized, present_regions)
    return feature_payload_rows, distances, weights, bandwidth, present_regions


def _latest_two_periods(train_periods: list[str]) -> tuple[str, str] | None:
    if len(train_periods) < 2:
        return None
    return train_periods[-2], train_periods[-1]


def _log_growth(value_old: float, value_new: float) -> float:
    return float(np.log1p(max(float(value_new), 0.0)) - np.log1p(max(float(value_old), 0.0)))


def _candidate_prediction_rows(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    periods: list[str],
    regions: list[str],
    holdout_period: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_periods = [period for period in periods if quarter_sort_key(period) < quarter_sort_key(holdout_period)]
    pair = _latest_two_periods(train_periods)
    if pair is None:
        return [], {"status": "not_enough_train_periods", "holdout_period": holdout_period}
    previous_period, last_period = pair
    feature_rows, distances, weights, bandwidth, present_regions = _region_features_for_train_period(
        rows_by_key,
        period=last_period,
        regions=regions,
    )
    region_index = {region: idx for idx, region in enumerate(present_regions)}
    aggregate_previous = _aggregate_by_metric([rows_by_key[(previous_period, region)] for region in present_regions])
    aggregate_last = _aggregate_by_metric([rows_by_key[(last_period, region)] for region in present_regions])
    rows: list[dict[str, Any]] = []
    for region in present_regions:
        last_row = rows_by_key.get((last_period, region))
        previous_row = rows_by_key.get((previous_period, region))
        if last_row is None or previous_row is None:
            continue
        predictions: dict[str, dict[str, Any]] = {
            "regional_carry_forward": {"candidate_family": "regional_carry_forward", "holdout_period": holdout_period, "region": region},
            "aggregate_log_trend": {"candidate_family": "aggregate_log_trend", "holdout_period": holdout_period, "region": region},
            "similarity_proxy_log_delta": {
                "candidate_family": "similarity_proxy_log_delta",
                "holdout_period": holdout_period,
                "region": region,
            },
        }
        idx = region_index[region]
        proxy_indices = [j for j in range(len(present_regions)) if j != idx]
        nearest_proxy = None
        if proxy_indices:
            nearest_proxy = present_regions[min(proxy_indices, key=lambda j: distances[idx, j])]
        for metric in COUNT_METRICS:
            last_value = max(float(last_row.get(metric) or 0.0), 0.0)
            predictions["regional_carry_forward"][metric] = last_value
            aggregate_growth = _log_growth(aggregate_previous.get(metric, 0.0), aggregate_last.get(metric, 0.0))
            predictions["aggregate_log_trend"][metric] = float(np.expm1(np.log1p(last_value) + aggregate_growth))
            proxy_growths: list[float] = []
            proxy_weights: list[float] = []
            for proxy_region in present_regions:
                if proxy_region == region:
                    continue
                old_row = rows_by_key.get((previous_period, proxy_region))
                new_row = rows_by_key.get((last_period, proxy_region))
                if old_row is None or new_row is None:
                    continue
                proxy_growths.append(_log_growth(float(old_row.get(metric) or 0.0), float(new_row.get(metric) or 0.0)))
                proxy_weights.append(float(weights[idx, region_index[proxy_region]]))
            if proxy_growths and sum(proxy_weights) > 0.0:
                weighted_growth = float(np.average(np.asarray(proxy_growths, dtype=np.float64), weights=np.asarray(proxy_weights, dtype=np.float64)))
            elif proxy_growths:
                weighted_growth = float(np.mean(np.asarray(proxy_growths, dtype=np.float64)))
            else:
                weighted_growth = 0.0
            predictions["similarity_proxy_log_delta"][metric] = float(np.expm1(np.log1p(last_value) + weighted_growth))
        for family, prediction in predictions.items():
            projected = _projection_cascade(prediction)
            projected["train_previous_period"] = previous_period
            projected["train_last_period"] = last_period
            projected["proxy_region"] = nearest_proxy
            projected["similarity_bandwidth"] = bandwidth
            rows.append(projected)
    diagnostics = {
        "status": "completed",
        "holdout_period": holdout_period,
        "train_previous_period": previous_period,
        "train_last_period": last_period,
        "feature_rows": feature_rows,
        "region_count": len(present_regions),
        "similarity_bandwidth": bandwidth,
    }
    return rows, diagnostics


def _score_predictions(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        period = str(prediction.get("holdout_period") or "")
        region = str(prediction.get("region") or "")
        target = rows_by_key.get((period, region))
        if target is None:
            continue
        for metric in COUNT_METRICS:
            target_value = _finite_float(target.get(metric))
            candidate_value = _finite_float(prediction.get(metric))
            if target_value is None or candidate_value is None:
                continue
            score_rows.append(
                {
                    "candidate_family": prediction.get("candidate_family"),
                    "holdout_period": period,
                    "region": region,
                    "metric_name": metric,
                    "target_value": float(target_value),
                    "candidate_value": float(candidate_value),
                    "absolute_error": abs(float(candidate_value) - float(target_value)),
                    "target_abs": abs(float(target_value)),
                }
            )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[(str(row["candidate_family"]), str(row["holdout_period"]), str(row["metric_name"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    eps = float(np.finfo(np.float32).eps)
    for (family, period, metric), rows in sorted(grouped.items()):
        numerator = float(sum(float(row["absolute_error"]) for row in rows))
        denominator = max(float(sum(float(row["target_abs"]) for row in rows)), eps)
        summary_rows.append(
            {
                "candidate_family": family,
                "holdout_period": period,
                "metric_name": metric,
                "region_count": len(rows),
                "normalized_absolute_error": numerator / denominator,
                "absolute_error_sum": numerator,
                "target_abs_sum": denominator,
            }
        )
    return score_rows, summary_rows


def _candidate_table(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row.get("candidate_family") or "")].append(row)
    table: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        values = [float(row.get("normalized_absolute_error") or 0.0) for row in rows]
        table.append(
            {
                "candidate_family": family,
                "mean_normalized_absolute_error": float(np.mean(np.asarray(values, dtype=np.float64))) if values else None,
                "worst_normalized_absolute_error": float(np.max(np.asarray(values, dtype=np.float64))) if values else None,
                "scored_split_metric_count": len(values),
            }
        )
    table.sort(key=lambda row: float(row.get("mean_normalized_absolute_error") or float("inf")))
    return table


def _proxy_assignment_rows(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    periods: list[str],
    regions: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not periods:
        return [], []
    latest_period = periods[-1]
    feature_rows, distances, weights, bandwidth, present_regions = _region_features_for_train_period(
        rows_by_key,
        period=latest_period,
        regions=regions,
    )
    rows: list[dict[str, Any]] = []
    for i, region in enumerate(present_regions):
        candidates = [j for j in range(len(present_regions)) if j != i]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda j: distances[i, j])
        rows.append(
            {
                "period_id": latest_period,
                "region": region,
                "proxy_region": present_regions[nearest],
                "distance": float(distances[i, nearest]),
                "kernel_weight": float(weights[i, nearest]),
                "kernel_bandwidth": float(bandwidth),
            }
        )
    return rows, feature_rows


def _gate(candidate_table: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {str(row.get("candidate_family") or ""): dict(row) for row in candidate_table}
    best = candidate_table[0] if candidate_table else {}
    best_family = str(best.get("candidate_family") or "")
    best_score = _finite_float(best.get("mean_normalized_absolute_error"))
    carry_score = _finite_float((by_family.get("regional_carry_forward") or {}).get("mean_normalized_absolute_error"))
    aggregate_score = _finite_float((by_family.get("aggregate_log_trend") or {}).get("mean_normalized_absolute_error"))
    blockers: list[str] = []
    if best_score is None:
        blockers.append("no_scored_subnational_candidates")
    if best_family == "regional_carry_forward":
        blockers.append("best_candidate_is_carry_forward")
    if carry_score is not None and best_score is not None and not best_score < carry_score:
        blockers.append("does_not_beat_regional_carry_forward")
    if aggregate_score is not None and best_score is not None and not best_score <= aggregate_score:
        blockers.append("does_not_match_or_beat_aggregate_trend_baseline")
    status = "subnational_proxy_champion_promoted" if not blockers else "no_subnational_proxy_champion"
    return {
        "status": status,
        "blockers": blockers,
        "best_candidate_family": best_family,
        "best_mean_normalized_absolute_error": best_score,
        "regional_carry_forward_mean_normalized_absolute_error": carry_score,
        "aggregate_log_trend_mean_normalized_absolute_error": aggregate_score,
        "contract": (
            "A subnational proxy champion must beat regional carry-forward and match or beat the aggregate "
            "trend baseline under blocked periods. The first R48 pass does not fit region-specific free "
            "parameters; it borrows only train-window growth from similar regions."
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
            normalized[key] = "|".join(str(item) for item in value) if isinstance(value, list) else value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("subnational_proxy_gate") or {})
    lines = [
        "# Phase 3 R48 Subnational Proxy Champion Contract",
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
        f"- Best candidate: `{gate.get('best_candidate_family')}`",
        f"- Best score: `{gate.get('best_mean_normalized_absolute_error')}`",
        f"- Carry-forward score: `{gate.get('regional_carry_forward_mean_normalized_absolute_error')}`",
        f"- Aggregate-trend score: `{gate.get('aggregate_log_trend_mean_normalized_absolute_error')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Candidate Table",
        "",
        "| Candidate | Mean normalized AE | Worst normalized AE | Split-metrics |",
        "|---|---:|---:|---:|",
    ]
    for row in list(report.get("candidate_table") or []):
        lines.append(
            f"| `{row.get('candidate_family')}` | {float(row.get('mean_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('worst_normalized_absolute_error') or 0.0):.6f} | {row.get('scored_split_metric_count')} |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            str(gate.get("contract") or ""),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(
    path: Path,
    *,
    candidate_table: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    labels = [str(row.get("candidate_family") or "") for row in candidate_table]
    scores = [float(row.get("mean_normalized_absolute_error") or 0.0) for row in candidate_table]
    axes[0].barh(np.arange(len(labels)), scores, color="#5c6f83")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean normalized absolute error")
    axes[0].set_title("Blocked subnational candidate comparison")
    proxy_labels = [str(row.get("region") or "") for row in proxy_rows]
    proxy_distances = [float(row.get("distance") or 0.0) for row in proxy_rows]
    axes[1].barh(np.arange(len(proxy_labels)), proxy_distances, color="#8a6f2a")
    axes[1].set_yticks(np.arange(len(proxy_labels)))
    axes[1].set_yticklabels(proxy_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("nearest-proxy feature distance")
    axes[1].set_title("Latest-period regional proxy assignments")
    fig.suptitle("R48 subnational proxy champion contract", fontsize=14, fontweight="bold")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r48_subnational_proxy_champion_contract(
    *,
    run_id: str = R48_RUN_ID,
    r44_report_path: Path | None = None,
    min_train_periods: int = 2,
) -> dict[str, Any]:
    r44_path = Path(r44_report_path) if r44_report_path is not None else R44_DEFAULT_REPORT
    regional_rows, r44_report = _load_r44_rows(r44_path)
    periods = _periods(regional_rows)
    regions = _overlapping_regions(regional_rows, periods)
    rows_by_key = _rows_by_period_region(regional_rows)
    prediction_rows: list[dict[str, Any]] = []
    split_diagnostics: list[dict[str, Any]] = []
    for holdout_period in periods[int(min_train_periods) :]:
        rows, diagnostics = _candidate_prediction_rows(
            rows_by_key,
            periods=periods,
            regions=regions,
            holdout_period=holdout_period,
        )
        prediction_rows.extend(rows)
        split_diagnostics.append(diagnostics)
    score_rows, split_metric_score_rows = _score_predictions(rows_by_key, prediction_rows)
    candidate_table = _candidate_table(split_metric_score_rows)
    gate = _gate(candidate_table)
    proxy_rows, feature_rows = _proxy_assignment_rows(rows_by_key, periods=periods, regions=regions)
    verdict = (
        "R48 promoted a sparse subnational proxy champion under blocked regional cascade periods."
        if gate["status"] == "subnational_proxy_champion_promoted"
        else "R48 did not promote a subnational proxy champion; regional cascade data are usable for diagnostics, but current proxy borrowing does not beat required baselines."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r48_subnational_proxy_champion_contract_report.json"
    md_path = analysis_dir / "r48_subnational_proxy_champion_contract_report.md"
    candidate_csv = analysis_dir / "r48_candidate_table.csv"
    split_metric_csv = analysis_dir / "r48_split_metric_scores.csv"
    score_csv = analysis_dir / "r48_region_metric_score_rows.csv"
    prediction_csv = analysis_dir / "r48_prediction_rows.csv"
    proxy_csv = analysis_dir / "r48_proxy_assignments.csv"
    feature_csv = analysis_dir / "r48_region_feature_matrix.csv"
    dashboard_path = analysis_dir / "r48_subnational_proxy_dashboard.png"
    report = {
        "schema_version": R48_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r44_report_path": r44_path.as_posix(),
        "r44_status": r44_report.get("status"),
        "periods": periods,
        "overlapping_regions": regions,
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "rate_features": list(RATE_FEATURES),
            "normalization": "sum absolute regional error divided by sum observed regional target for each split and metric",
            "similarity": "Gaussian kernel over standardized latest-train cascade size, share, and rate features with data-derived median-distance bandwidth",
            "free_region_parameters": 0,
        },
        "subnational_proxy_gate": gate,
        "split_diagnostics": split_diagnostics,
        "candidate_table": candidate_table,
        "split_metric_score_rows": split_metric_score_rows,
        "score_rows": score_rows,
        "prediction_rows": prediction_rows,
        "proxy_assignment_rows": proxy_rows,
        "feature_rows": feature_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "candidate_csv": candidate_csv.as_posix(),
            "split_metric_scores_csv": split_metric_csv.as_posix(),
            "region_metric_scores_csv": score_csv.as_posix(),
            "prediction_csv": prediction_csv.as_posix(),
            "proxy_assignment_csv": proxy_csv.as_posix(),
            "feature_matrix_csv": feature_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(candidate_csv, candidate_table)
    _write_csv(split_metric_csv, split_metric_score_rows)
    _write_csv(score_csv, score_rows)
    _write_csv(prediction_csv, prediction_rows)
    _write_csv(proxy_csv, proxy_rows)
    _write_csv(feature_csv, feature_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, candidate_table=candidate_table, proxy_rows=proxy_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R48 subnational proxy champion contract.")
    parser.add_argument("--run-id", default=R48_RUN_ID)
    parser.add_argument("--r44-report-path", default=None)
    parser.add_argument("--min-train-periods", type=int, default=2)
    args = parser.parse_args()
    run_r48_subnational_proxy_champion_contract(
        run_id=str(args.run_id),
        r44_report_path=None if args.r44_report_path is None else Path(args.r44_report_path),
        min_train_periods=int(args.min_train_periods),
    )


if __name__ == "__main__":
    _main()
