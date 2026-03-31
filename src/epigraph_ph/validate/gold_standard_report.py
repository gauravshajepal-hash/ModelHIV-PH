from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.runtime import ensure_dir, read_json, write_json

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PHASE_ORDER = ["phase0", "phase1", "phase15", "phase2", "phase3", "phase4"]
HARP_METRICS = [
    ("diagnosed_stock", "Diagnosed / PLHIV"),
    ("art_stock", "On ART / PLHIV"),
    ("documented_suppression", "Suppressed / PLHIV"),
    ("second95", "On ART / Diagnosed"),
]


def _phase_payload(run_dir: Path, phase_name: str) -> dict[str, Any]:
    phase_dir = run_dir / phase_name
    return {
        "manifest": read_json(phase_dir / "gold_standard_manifest.json", default={}),
        "checks": read_json(phase_dir / "gold_standard_checks.json", default=[]),
        "summary": read_json(phase_dir / "gold_standard_summary.json", default={}),
    }


def _plot_phase_overview(rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return
    labels = [row["phase_name"] for row in rows]
    ratios = [row["pass_ratio"] for row in rows]
    colors = ["#2e8b57" if row["overall_passed"] else "#c44e52" for row in rows]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(labels, ratios, color=colors)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Gold-standard pass ratio")
    ax.set_title("Module Gold-Standard Overview")
    for bar, row in zip(bars, rows, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(1.02, bar.get_height() + 0.03),
            f"{row['passed_check_count']}/{row['passed_check_count'] + row['failed_check_count']}\n{row['gold_standard_mode']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_phase_check_dashboard(rows: list[dict[str, Any]], phase_checks: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, max(9, 2.2 * len(rows))), constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, rows, strict=False):
        checks = phase_checks[row["phase_name"]]
        names = [str(check.get("name") or "") for check in checks]
        values = [1 if bool(check.get("passed")) else 0 for check in checks]
        colors = ["#2e8b57" if value else "#c44e52" for value in values]
        y = np.arange(len(names))
        ax.barh(y, values, color=colors)
        ax.set_yticks(y, names, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 1], ["Fail", "Pass"])
        ax.set_title(f"{row['phase_name']} | {row['truth_claim_level']} | {row['benchmark_policy']}", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _national_training_series(state_rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    by_year_state: dict[int, dict[str, float]] = defaultdict(dict)
    for row in state_rows:
        if str(row.get("province")) != "Philippines":
            continue
        time_value = str(row.get("time") or "")
        if len(time_value) < 4 or not time_value[:4].isdigit():
            continue
        year = int(time_value[:4])
        by_year_state[year][str(row.get("state"))] = float(row.get("value") or 0.0)
    output: dict[int, dict[str, float]] = {}
    for year, states in sorted(by_year_state.items()):
        u = float(states.get("U", 0.0))
        a = float(states.get("A", 0.0))
        v = float(states.get("V", 0.0))
        diagnosed = max(0.0, min(1.0, 1.0 - u))
        art = max(0.0, min(1.0, a + v))
        suppression = max(0.0, min(1.0, v))
        second95 = art / max(diagnosed, 1e-6)
        output[year] = {
            "diagnosed_stock": diagnosed,
            "art_stock": art,
            "documented_suppression": suppression,
            "second95": second95,
        }
    return output


def _harp_observed_series(panel_rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for row in panel_rows:
        year = row.get("year")
        plhiv = float(row.get("estimated_plhiv") or 0.0)
        diagnosed = float(row.get("diagnosed_plhiv") or 0.0)
        art = float(row.get("alive_on_art") or 0.0)
        suppressed = float(row.get("virally_suppressed") or 0.0)
        if not year or plhiv <= 0.0:
            continue
        output[int(year)] = {
            "diagnosed_stock": diagnosed / plhiv if diagnosed else np.nan,
            "art_stock": art / plhiv if art else np.nan,
            "documented_suppression": suppressed / plhiv if suppressed else np.nan,
            "second95": art / diagnosed if diagnosed else np.nan,
        }
    return output


def _plot_harp_training_holdout(
    *,
    run_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    phase3_backtest_dir = run_dir / "phase3_tuned_backtest"
    if not (phase3_backtest_dir / "frozen_history_backtest_evaluation.json").exists():
        phase3_backtest_dir = run_dir / "phase3_frozen_backtest"
    evaluation = read_json(phase3_backtest_dir / "frozen_history_backtest_evaluation.json", default={})
    state_rows = read_json(phase3_backtest_dir / "state_estimates_rows.json", default=[])
    panel = read_json(run_dir / "harp_archive" / "historical_harp_panel.json", default={})
    observed = _harp_observed_series(list(panel.get("rows", [])))
    model_train = _national_training_series(state_rows)
    holdout_rows = list(evaluation.get("holdout_reference_check", {}).get("comparisons", []))
    carry_rows = list(evaluation.get("carry_forward_baseline", {}).get("comparisons", []))
    holdout_by_year = {
        int(str(row.get("date") or "0000")[:4]): row
        for row in holdout_rows
        if str(row.get("date") or "")[:4].isdigit()
    }
    carry_by_year = {
        int(str(row.get("date") or "0000")[:4]): row
        for row in carry_rows
        if str(row.get("date") or "")[:4].isdigit()
    }
    holdout_years = sorted(holdout_by_year.keys())

    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.flatten()
        observed_years = sorted(observed.keys())
        model_years = sorted(model_train.keys())
        for ax, (metric_key, label) in zip(axes, HARP_METRICS, strict=False):
            observed_values = [observed[year].get(metric_key, np.nan) for year in observed_years]
            model_values = [model_train[year].get(metric_key, np.nan) for year in model_years]
            ax.plot(observed_years, observed_values, color="#222222", marker="o", label="HARP observed")
            if model_years:
                ax.plot(model_years, model_values, color="#1f77b4", marker="o", label="Model fit (train)")
            if holdout_years:
                model_holdout_values = [float(holdout_by_year[year].get("model", {}).get(metric_key, np.nan)) for year in holdout_years]
                harp_holdout_values = [float(holdout_by_year[year].get("reference", {}).get(metric_key, np.nan)) for year in holdout_years]
                carry_values = [float(carry_by_year.get(year, {}).get("baseline_reference", {}).get(metric_key, np.nan)) for year in holdout_years]
                ax.scatter(holdout_years, model_holdout_values, color="#d62728", marker="*", s=110, label="Model holdout")
                ax.scatter(holdout_years, harp_holdout_values, color="#2ca02c", marker="s", s=45, label="HARP holdout")
                ax.scatter(holdout_years, carry_values, color="#7f7f7f", marker="x", s=55, label="Carry-forward")
                ax.axvspan(min(holdout_years) - 0.3, max(holdout_years) + 0.3, color="#f2f2f2", alpha=0.4)
            ax.set_title(label)
            ax.set_ylim(0.0, 1.05)
            ax.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4)
        fig.suptitle("Phase 3 vs HARP: training years and frozen holdout years", y=1.02, fontsize=14)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    return {
        "phase3_backtest_dir": str(phase3_backtest_dir),
        "holdout_years": holdout_years,
        "train_years_with_model": sorted(model_train.keys()),
        "observed_years": sorted(observed.keys()),
        "model_beats_carry_forward": bool(evaluation.get("summary", {}).get("model_beats_carry_forward")),
        "model_mean_absolute_error": float(evaluation.get("summary", {}).get("model_mean_absolute_error", np.nan)),
        "carry_forward_mean_absolute_error": float(evaluation.get("summary", {}).get("carry_forward_mean_absolute_error", np.nan)),
    }


def build_gold_standard_report(*, run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    analysis_dir = ensure_dir(Path(output_dir) if output_dir is not None else run_dir / "analysis")

    rows: list[dict[str, Any]] = []
    phase_checks: dict[str, list[dict[str, Any]]] = {}
    for phase_name in PHASE_ORDER:
        payload = _phase_payload(run_dir, phase_name)
        summary = dict(payload["summary"] or {})
        if not summary:
            continue
        checks = list(payload["checks"] or [])
        phase_checks[phase_name] = checks
        passed = int(summary.get("passed_check_count", 0))
        failed = int(summary.get("failed_check_count", 0))
        total = max(1, passed + failed)
        rows.append(
            {
                "phase_name": phase_name,
                "passed_check_count": passed,
                "failed_check_count": failed,
                "pass_ratio": passed / total,
                "overall_passed": bool(summary.get("overall_passed")),
                "gold_standard_mode": str(summary.get("gold_standard_mode") or ""),
                "truth_claim_level": str(summary.get("truth_claim_level") or ""),
                "benchmark_policy": str(summary.get("benchmark_policy") or ""),
                "failed_checks": [str(check.get("name") or "") for check in checks if not bool(check.get("passed"))],
            }
        )

    overview_path = analysis_dir / "gold_standard_phase_overview.png"
    dashboard_path = analysis_dir / "gold_standard_check_dashboard.png"
    _plot_phase_overview(rows, overview_path)
    _plot_phase_check_dashboard(rows, phase_checks, dashboard_path)

    harp_plot_path = analysis_dir / "phase3_harp_train_vs_holdout.png"
    harp_summary = _plot_harp_training_holdout(run_dir=run_dir, output_path=harp_plot_path)

    markdown_lines = [
        "# Gold Standard Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Phase summary",
        "",
    ]
    for row in rows:
        markdown_lines.append(
            f"- `{row['phase_name']}`: {row['passed_check_count']}/{row['passed_check_count'] + row['failed_check_count']} checks passed, "
            f"mode=`{row['gold_standard_mode']}`, truth=`{row['truth_claim_level']}`"
        )
        if row["failed_checks"]:
            markdown_lines.append(f"  failed: {', '.join(row['failed_checks'])}")
    markdown_lines.extend(
        [
            "",
            "## Phase 3 HARP backtest",
            "",
            f"- model beats carry-forward: `{harp_summary.get('model_beats_carry_forward')}`",
            f"- model MAE: `{harp_summary.get('model_mean_absolute_error')}`",
            f"- carry-forward MAE: `{harp_summary.get('carry_forward_mean_absolute_error')}`",
        ]
    )

    report = {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "phase_rows": rows,
        "harp_backtest_summary": harp_summary,
        "artifacts": {
            "gold_standard_phase_overview": str(overview_path),
            "gold_standard_check_dashboard": str(dashboard_path),
            "phase3_harp_train_vs_holdout": str(harp_plot_path),
            "gold_standard_report_json": str(analysis_dir / "gold_standard_report.json"),
            "gold_standard_report_md": str(analysis_dir / "gold_standard_report.md"),
        },
    }
    write_json(analysis_dir / "gold_standard_report.json", report)
    (analysis_dir / "gold_standard_report.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return report
