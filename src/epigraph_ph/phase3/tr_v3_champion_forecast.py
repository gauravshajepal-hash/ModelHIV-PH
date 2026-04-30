from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3.tr_v3_05_autoresearch import (
    PRIMARY_METRICS,
    DynamicControlConfig,
    ObservationConfig,
    QuarterlyDataset,
    build_annual_anchor_rows,
    build_quarterly_dataset,
    quarter_ordinal,
    quarter_sort_key,
    quarter_year,
    repo_root,
)
from epigraph_ph.runtime import ensure_dir, write_json


CONTRACT_CHOICES: tuple[str, ...] = ("exact_only", "dense_train_observed_score")


def _ordinal_to_quarter(value: int) -> str:
    year = int(value) // 4
    quarter = (int(value) % 4) + 1
    return f"{year:04d}-Q{quarter}"


def _future_quarters(last_quarter: str, horizon_quarters: int) -> list[str]:
    base = int(quarter_ordinal(str(last_quarter)))
    return [_ordinal_to_quarter(base + offset) for offset in range(1, int(horizon_quarters) + 1)]


def _contract_payload(archive_run_id: str, contract_name: str) -> tuple[list[dict[str, Any]], set[str], dict[str, Any] | None]:
    if contract_name == "exact_only":
        return suite.build_quarterly_observation_rows(archive_run_id), {"exact_observed"}, None
    if contract_name == "dense_train_observed_score":
        dense_payload = suite._build_dense_contract_payload(archive_run_id)
        return list(dense_payload["rows"]), {"exact_observed", "bridge_observed"}, dense_payload
    raise ValueError(f"Unsupported contract_name: {contract_name}")


def _target_metric_tier(row: dict[str, Any], metric_name: str) -> str:
    metric_tiers = row.get("metric_tiers")
    if isinstance(metric_tiers, dict) and metric_name in metric_tiers:
        return str(metric_tiers[metric_name])
    return "exact_observed"


def _collect_metric_residuals(
    quarterly_rows: list[dict[str, Any]],
    *,
    allowed_tiers: set[str],
) -> dict[str, list[float]]:
    residuals = {metric_name: [] for metric_name in PRIMARY_METRICS}
    for split_row in quarterly_rows:
        targets = list(split_row.get("holdout_target_rows") or [])
        predictions = list(split_row.get("candidate_prediction_rows") or [])
        for target_row, prediction_row in zip(targets, predictions, strict=False):
            for metric_name in PRIMARY_METRICS:
                if _target_metric_tier(target_row, metric_name) not in allowed_tiers:
                    continue
                target_value = target_row.get(metric_name)
                prediction_value = prediction_row.get(metric_name)
                if target_value is None or prediction_value is None:
                    continue
                residuals[metric_name].append(float(prediction_value) - float(target_value))
    return residuals


def _residual_interval_payload(
    quarterly_rows: list[dict[str, Any]],
    *,
    allowed_tiers: set[str],
    alpha: float,
) -> dict[str, Any]:
    residuals = _collect_metric_residuals(quarterly_rows, allowed_tiers=allowed_tiers)
    lower_q = float(alpha / 2.0)
    upper_q = float(1.0 - lower_q)
    metrics: dict[str, Any] = {}
    for metric_name, values in residuals.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            metrics[metric_name] = {
                "count": 0,
                "mean_residual": 0.0,
                "median_residual": 0.0,
                "lower_quantile": lower_q,
                "upper_quantile": upper_q,
                "q_low": 0.0,
                "q_high": 0.0,
            }
            continue
        metrics[metric_name] = {
            "count": int(arr.size),
            "mean_residual": float(np.mean(arr)),
            "median_residual": float(np.median(arr)),
            "lower_quantile": lower_q,
            "upper_quantile": upper_q,
            "q_low": float(np.quantile(arr, lower_q)),
            "q_high": float(np.quantile(arr, upper_q)),
        }
    return {
        "alpha": float(alpha),
        "metrics": metrics,
    }


def _future_holdout_rows(forecast_quarters: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for quarter in forecast_quarters:
        row = {"quarter": str(quarter)}
        for metric_name in PRIMARY_METRICS:
            row[metric_name] = None
            row[f"{metric_name}_tier"] = "forecast"
        rows.append(row)
    return rows


def _forecast_future_rows(
    observation_rows: list[dict[str, Any]],
    *,
    annual_rows: list[dict[str, Any]],
    spec: suite.ExperimentSpec,
    best_candidate: dict[str, Any],
    forecast_quarters: list[str],
) -> dict[str, Any]:
    base_dataset = build_quarterly_dataset(observation_rows, [9999])
    future_dataset = QuarterlyDataset(
        holdout_years=sorted({quarter_year(quarter) for quarter in forecast_quarters}),
        train_rows=list(base_dataset.train_rows),
        holdout_rows=_future_holdout_rows(forecast_quarters),
        train_state_rows=list(base_dataset.train_state_rows),
        holdout_state_rows=[],
        train_transition_rows=list(base_dataset.train_transition_rows),
        metric_scales=dict(base_dataset.metric_scales),
        eps=float(base_dataset.eps),
    )
    dynamic_cfg = DynamicControlConfig(**dict(best_candidate["dynamic_cfg"]))
    observation_cfg = ObservationConfig(**dict(best_candidate["observation_cfg"]))
    return suite._fit_05a_experiment_candidate(future_dataset, annual_rows, dynamic_cfg, observation_cfg, spec)


def _apply_empirical_intervals(
    prediction_rows: list[dict[str, Any]],
    residual_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics = dict(residual_payload.get("metrics") or {})
    out: list[dict[str, Any]] = []
    for row in prediction_rows:
        enriched = {"quarter": str(row["quarter"])}
        for metric_name in PRIMARY_METRICS:
            value = row.get(metric_name)
            enriched[metric_name] = value
            if value is None:
                enriched[f"{metric_name}_lower"] = None
                enriched[f"{metric_name}_upper"] = None
                continue
            interval = dict(metrics.get(metric_name) or {})
            lower = float(value) + float(interval.get("q_low") or 0.0)
            upper = float(value) + float(interval.get("q_high") or 0.0)
            enriched[f"{metric_name}_lower"] = float(max(lower, 0.0))
            enriched[f"{metric_name}_upper"] = float(max(upper, 0.0))
        out.append(enriched)
    return out


def _save_forecast_curve_graph(
    observation_rows: list[dict[str, Any]],
    forecast_rows: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
) -> bool:
    metrics = [metric_name for metric_name in PRIMARY_METRICS if any(row.get(metric_name) is not None for row in observation_rows + forecast_rows)]
    if not metrics:
        return False
    cols = 2
    rows = int(math.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(4.5, rows * 3.6)))
    axes_list = list(np.asarray(axes).reshape(-1))
    tier_styles = {
        "exact_observed": {"color": "#111111", "marker": "o", "label": "Exact"},
        "bridge_observed": {"color": "#c66a00", "marker": "s", "label": "Bridge"},
        "rule_based_extrapolated": {"color": "#7a7a7a", "marker": "^", "label": "Rule-based"},
        "latent_imputed": {"color": "#9467bd", "marker": "x", "label": "Latent"},
    }
    for ax, metric_name in zip(axes_list, metrics, strict=False):
        observed_rows = [row for row in observation_rows if row.get(metric_name) is not None]
        for tier_name, style in tier_styles.items():
            tier_points = [row for row in observed_rows if suite._metric_tier(row, metric_name) == tier_name]
            if not tier_points:
                continue
            ax.plot(
                [str(row["quarter"]) for row in tier_points],
                [float(row[metric_name]) for row in tier_points],
                linestyle="None",
                marker=str(style["marker"]),
                color=str(style["color"]),
                markersize=4,
                label=str(style["label"]),
            )
        future_rows = [row for row in forecast_rows if row.get(metric_name) is not None]
        if future_rows:
            ax.plot(
                [str(row["quarter"]) for row in future_rows],
                [float(row[metric_name]) for row in future_rows],
                color="#1f77b4",
                linewidth=2.0,
                label="Forecast",
            )
            lower = [row.get(f"{metric_name}_lower") for row in future_rows]
            upper = [row.get(f"{metric_name}_upper") for row in future_rows]
            if all(value is not None for value in lower) and all(value is not None for value in upper):
                ax.fill_between(
                    [str(row["quarter"]) for row in future_rows],
                    [float(value) for value in lower],
                    [float(value) for value in upper],
                    color="#1f77b4",
                    alpha=0.18,
                    label="Empirical band",
                )
        ax.set_title(metric_name.replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.2)
    for ax in axes_list[len(metrics):]:
        ax.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Champion Forecast",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Forecast horizon (quarters): `{payload['forecast_horizon_quarters']}`",
        "",
    ]
    for contract_name, contract_payload in (payload.get("contracts") or {}).items():
        lines.extend(
            [
                f"## {contract_name}",
                "",
                f"- Champion: `{contract_payload['experiment_id']}`",
                f"- Backtest quarterly mean MAE: `{contract_payload['quarterly_mean_mae']:.6f}`",
                f"- Backtest baseline MAE: `{contract_payload['quarterly_baseline_mean_mae']:.6f}`",
                f"- Observation curve: `{contract_payload['forecast_curve_file']}`",
                "",
                "| Metric | Residual count | Mean residual | Lower band | Upper band |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric_name, metric_payload in (contract_payload.get("residual_intervals", {}).get("metrics") or {}).items():
            lines.append(
                f"| `{metric_name}` | `{metric_payload['count']}` | `{metric_payload['mean_residual']:.3f}` | `{metric_payload['q_low']:.3f}` | `{metric_payload['q_high']:.3f}` |"
            )
        lines.extend(
            [
                "",
                "| Quarter | Diagnosed | ART | New diagnosed | Virally suppressed |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in contract_payload.get("future_forecast_rows") or []:
            lines.append(
                f"| `{row['quarter']}` | `{(row.get('diagnosed_plhiv') or 0.0):,.0f}` | `{(row.get('alive_on_art') or 0.0):,.0f}` | `{(row.get('new_diagnosed_cases_period') or 0.0):,.0f}` | `{(row.get('virally_suppressed') or 0.0):,.0f}` |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_tr_v3_champion_forecast(
    *,
    run_id: str,
    archive_run_id: str,
    contracts: list[str] | None = None,
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
    forecast_horizon_quarters: int = 4,
    interval_alpha: float = 0.1,
) -> dict[str, Any]:
    selected_contracts = list(contracts or CONTRACT_CHOICES)
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    availability = suite._build_availability_payload(archive_run_id)
    annual_rows = build_annual_anchor_rows(archive_run_id)
    analysis_dir = ensure_dir(repo_root() / "artifacts" / "runs" / run_id / "analysis")
    contract_payloads: dict[str, Any] = {}
    for contract_name in selected_contracts:
        observation_rows, scoring_tiers, dense_payload = _contract_payload(archive_run_id, contract_name)
        champion_id = suite.default_predictive_candidate_id(contract_name)
        spec = spec_map[champion_id]
        result = suite._evaluate_experiment_spec(
            spec,
            observation_rows=observation_rows,
            annual_rows=annual_rows,
            availability=availability,
            scoring_tiers=scoring_tiers,
            quarterly_start_year=quarterly_start_year,
            quarterly_end_year=quarterly_end_year,
            quarterly_min_train_years=quarterly_min_train_years,
            annual_start_year=annual_start_year,
            annual_end_year=annual_end_year,
            annual_min_train_years=annual_min_train_years,
            horizon_years=horizon_years,
        )
        latest_quarter = max((str(row["quarter"]) for row in observation_rows), key=quarter_sort_key)
        forecast_quarters = _future_quarters(latest_quarter, forecast_horizon_quarters)
        forecast_candidate = _forecast_future_rows(
            observation_rows,
            annual_rows=annual_rows,
            spec=spec,
            best_candidate=dict(result["best_candidate"]),
            forecast_quarters=forecast_quarters,
        )
        residual_intervals = _residual_interval_payload(
            list(result["quarterly_rows"]),
            allowed_tiers=scoring_tiers,
            alpha=interval_alpha,
        )
        forecast_rows = _apply_empirical_intervals(
            list(forecast_candidate["prediction_rows"]),
            residual_intervals,
        )
        graph_path = analysis_dir / f"{champion_id}_{contract_name}_forecast_curves.png"
        _save_forecast_curve_graph(
            observation_rows,
            forecast_rows,
            graph_path,
            title=f"{champion_id} forecast curves ({contract_name})",
        )
        contract_payloads[contract_name] = {
            "experiment_id": champion_id,
            "decision": str(result["decision"]),
            "decision_reason": str(result["decision_reason"]),
            "best_candidate": dict(result["best_candidate"]),
            "quarterly_mean_mae": float(result["quarterly_summary"]["candidate_mean_mae"]),
            "quarterly_baseline_mean_mae": float(result["quarterly_summary"]["carry_forward_mean_mae"]),
            "annual_mean_incidence_error": float(result["annual_summary"]["candidate_mean_incidence_error"]),
            "annual_baseline_incidence_error": float(result["annual_summary"]["baseline_mean_incidence_error"]),
            "dense_contract": dense_payload,
            "residual_intervals": residual_intervals,
            "future_forecast_rows": forecast_rows,
            "future_hazard_rows": list(forecast_candidate.get("trajectory_rows") or []),
            "forecast_curve_file": graph_path.name,
            "backtest_quarterly_rows": list(result["quarterly_rows"]),
            "backtest_annual_rows": list(result["annual_rows"]),
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "archive_run_id": archive_run_id,
        "forecast_horizon_quarters": int(forecast_horizon_quarters),
        "interval_alpha": float(interval_alpha),
        "contracts": contract_payloads,
    }
    write_json(analysis_dir / "tr_v3_champion_forecast_report.json", payload)
    (analysis_dir / "tr_v3_champion_forecast_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tr-v3-champion-forecast")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=suite._latest_standard_archive_run())
    parser.add_argument(
        "--contracts",
        nargs="+",
        default=list(CONTRACT_CHOICES),
        choices=list(CONTRACT_CHOICES),
    )
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=4)
    parser.add_argument("--interval-alpha", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_champion_forecast(
        run_id=args.run_id,
        archive_run_id=args.archive_run_id,
        contracts=list(args.contracts),
        quarterly_start_year=args.quarterly_start_year,
        quarterly_end_year=args.quarterly_end_year,
        quarterly_min_train_years=args.quarterly_min_train_years,
        annual_start_year=args.annual_start_year,
        annual_end_year=args.annual_end_year,
        annual_min_train_years=args.annual_min_train_years,
        horizon_years=args.horizon_years,
        forecast_horizon_quarters=args.forecast_horizon_quarters,
        interval_alpha=args.interval_alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
