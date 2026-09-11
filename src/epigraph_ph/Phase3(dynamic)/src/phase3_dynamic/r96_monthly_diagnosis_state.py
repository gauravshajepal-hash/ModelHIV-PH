"""Report-vintage-preserving monthly diagnosis experiment; no incidence targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from .data import sandbox_repo_root
from .r94_2026_q1_hasp_intake_gate import (
    DEFAULT_PDF_PATH as Q1_PDF,
    extract_hasp_2026_q1_rows,
    extract_pdf_text,
)
from .r95_2026_q2_hasp_intake_gate import (
    DEFAULT_PDF_PATH as Q2_PDF,
    R94_DEFAULT_REPORT,
    R95_RUN_ID,
    extract_hasp_2026_q2_rows,
)


RUN_ID = "p3d-r96-monthly-diagnosis-state-20260911-s01"
FAMILIES = ("quarter_carry", "last_month", "historical_mean", "local_level", "local_level_drift")
# A complete prior calendar year is the fixed initial evaluation support, not a fitted coefficient.
INITIAL_MONTHS = 12
QUARTER_MONTHS = 3
CONTRACT = {
    "schema_version": "r96.monthly_diagnosis_state.v1",
    "families": FAMILIES,
    "initial_months": INITIAL_MONTHS,
    "selection": "minimum mean complete-quarter absolute error on inner rolling origins",
    "outer_evaluation": "nonoverlapping quarters; refit on prefix only; latest Q1 vintage frozen for Q2 inputs",
    "measurement": "reported monthly new diagnoses, not infections or reporting completeness",
    "equations": [
        "z_t = log(1 + Y_t) = ell_t + epsilon_t",
        "ell_t = ell_(t-1) + d + eta_t",
        "Var(eta_t) = theta*sigma2; Var(epsilon_t) = (1-theta)*sigma2",
        "theta in [0,1]; sigma2 and optional d profiled by train-prefix Gaussian likelihood",
        "point_t = max(0, exp(E[ell_t|train])-1); quarter point = sum(monthly marginal medians)",
    ],
    "uncertainty": "95% conditional observation intervals; excludes parameter and source-revision uncertainty",
    "strict_promotion_requires": [
        "historical_nested_MAE_and_p90_not_worse_than_quarter_carry",
        "Q1_Q2_each_not_worse_than_carry_and_frozen_R41",
        "frozen_stock_values_and_conditional_rates_preserved",
        "independently_timestamped_forecast_and_report_vintages",
    ],
    "availability": "historical issue dates unavailable; time-blocked retrospective replay only",
    "q2_status": "already inspected before candidate design; not an untouched prospective test",
    "identified_process": "reported diagnosis intensity only; reporting vs diagnosis volume not separately identified",
    "forbidden_uses": ["incidence_truth", "regional_effect_claim", "causal_reporting_shock", "AEM_superiority"],
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def monthly_series(rows: list[dict]) -> list[dict]:
    selected = {}
    for row in rows:
        if (row.get("metric_id") != "new_diagnosed_cases_period"
                or row.get("time_granularity") != "monthly"
                or row.get("geography") != "national" or row.get("population") != "all"):
            continue
        if row.get("observation_role") != "direct_target" or row.get("allowed_use") != "direct_target":
            raise ValueError("Monthly diagnosis input is not permitted as a direct target")
        if row.get("measurement_semantics") != "flow_count" or row.get("unit") != "people":
            raise ValueError("Monthly diagnosis input has incompatible units/semantics")
        month = row["time_start"]
        if month != row["time_end"] or month in selected:
            raise ValueError("Duplicate or interval-ambiguous monthly observation")
        value = float(row["value"])
        if not np.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("Monthly counts must be finite nonnegative integers")
        selected[month] = row
    result = [selected[key] for key in sorted(selected)]
    indices = [int(r["time_start"][:4]) * 12 + int(r["time_start"][5:]) for r in result]
    if not indices or any(b - a != 1 for a, b in zip(indices, indices[1:])):
        raise ValueError("Monthly support must be nonempty and contiguous; missing months cannot be zero-filled")
    return result


def _filter(z: np.ndarray, theta: float, drift: float) -> tuple:
    # Exact diffuse first observation: ell_0|z_0 has normalized variance R, not an arbitrary large prior.
    measurement = 1.0 - theta
    level, covariance = float(z[0]), measurement
    errors, variances = [], []
    for observation in z[1:]:
        level += drift
        covariance += theta
        variance = covariance + measurement
        error = float(observation) - level
        gain = covariance / variance
        level += gain * error
        covariance *= 1.0 - gain
        errors.append(error)
        variances.append(variance)
    return np.asarray(errors), np.asarray(variances), level, covariance


def fit_local_level(values: list[float], *, drift: bool = False) -> dict:
    values_array = np.asarray(values, dtype=float)
    if (len(values_array) < 4 or not np.isfinite(values_array).all() or (values_array < 0).any()):
        raise ValueError("State estimation requires at least four finite nonnegative counts")
    z = np.log1p(values_array)

    def profile(theta: float) -> tuple:
        errors, variances, _, _ = _filter(z, theta, 0.0)
        d = 0.0
        if drift:
            unit_errors, _, _, _ = _filter(z, theta, 1.0)
            sensitivity = errors - unit_errors
            d = float(np.sum(sensitivity * errors / variances) / np.sum(sensitivity**2 / variances))
        errors, variances, level, covariance = _filter(z, theta, d)
        sigma2 = float(np.mean(errors**2 / variances))
        # Machine precision only, not a fitted biological or observation variance floor.
        objective = (len(errors) * np.log(max(sigma2, np.finfo(float).tiny)) + np.log(variances).sum()) / 2
        return float(objective), sigma2, d, level, covariance, errors, variances

    optimum = minimize_scalar(lambda t: profile(t)[0], bounds=(0.0, 1.0), method="bounded")
    if not optimum.success:
        raise RuntimeError(f"State likelihood optimization failed: {optimum.message}")
    theta = min((0.0, float(optimum.x), 1.0), key=lambda t: profile(t)[0])
    objective, sigma2, d, level, covariance, errors, variances = profile(theta)
    return {
        "theta": theta, "sigma2": sigma2, "drift": d, "level": level,
        "state_variance": covariance * sigma2, "process_variance": theta * sigma2,
        "measurement_variance": (1.0 - theta) * sigma2,
        "profile_negative_log_likelihood_without_constant": objective,
        "optimizer_success": True, "boundary_variance_estimate": theta in (0.0, 1.0),
        "standardized_innovations": (errors / np.sqrt(np.maximum(variances * sigma2, np.finfo(float).tiny))).tolist(),
        "training_count": len(values),
    }


def forecast(values: list[float], family: str, horizon: int = QUARTER_MONTHS) -> dict:
    if family not in FAMILIES or horizon < 1 or len(values) < QUARTER_MONTHS:
        raise ValueError("Invalid family, horizon, or insufficient training support")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Training values must be finite and nonnegative")
    if family == "quarter_carry":
        points = [float(values[-QUARTER_MONTHS + i % QUARTER_MONTHS]) for i in range(horizon)]
    elif family == "last_month":
        points = [float(values[-1])] * horizon
    elif family == "historical_mean":
        points = [float(np.mean(values))] * horizon
    else:
        fit = fit_local_level(values, drift=family == "local_level_drift")
        means = fit["level"] + np.arange(1, horizon + 1) * fit["drift"]
        variances = fit["state_variance"] + np.arange(1, horizon + 1) * fit["process_variance"] + fit["measurement_variance"]
        width = norm.ppf(0.975) * np.sqrt(variances)
        with np.errstate(over="raise", invalid="raise"):
            points = np.maximum(0, np.expm1(means)).tolist()
            lower = np.maximum(0, np.expm1(means - width)).tolist()
            upper = np.maximum(0, np.expm1(means + width)).tolist()
        return {"points": points, "lower95": lower, "upper95": upper, "log_means": means.tolist(),
                "log_variances": variances.tolist(), "fit": fit}
    return {"points": points, "lower95": None, "upper95": None, "fit": None}


def backtest(series: list[dict]) -> tuple[list[dict], list[dict]]:
    values = [float(r["value"]) for r in series]
    rows, choices = [], []
    for origin in range(INITIAL_MONTHS, len(values) - QUARTER_MONTHS + 1, QUARTER_MONTHS):
        # Selection sees only completed inner blocks; no observation from this target block is visible.
        history = {family: [r["absolute_error"] for r in rows if r["family"] == family] for family in FAMILIES}
        chosen = min(FAMILIES, key=lambda f: np.mean(history[f])) if rows else "quarter_carry"
        target = sum(values[origin:origin + QUARTER_MONTHS])
        scale = float(np.mean(values[:origin]) * QUARTER_MONTHS)
        for family in FAMILIES:
            predicted = sum(forecast(values[:origin], family)["points"])
            error = abs(predicted - target)
            row = {"origin": series[origin - 1]["time_start"], "target_end": series[origin + 2]["time_end"],
                   "training_rows": origin, "training_max_month": series[origin - 1]["time_end"],
                   "family": family, "actual": target, "prediction": predicted, "absolute_error": error,
                   "train_scaled_error": error / scale if scale > 0 else None}
            rows.append(row)
            if family == chosen:
                choices.append({**row, "inner_fold_count": len(history[family]), "selection_scope": "past_complete_quarters_only"})
    return rows, choices


def choose_family(series: list[dict]) -> tuple[str, dict]:
    rows, _ = backtest(series)
    if not rows:
        return "quarter_carry", {}
    scores = {f: float(np.mean([r["absolute_error"] for r in rows if r["family"] == f])) for f in FAMILIES}
    return min(FAMILIES, key=scores.get), scores


def _summary(rows: list[dict]) -> dict:
    errors = [r["absolute_error"] for r in rows]
    normalized = [r["train_scaled_error"] for r in rows if r["train_scaled_error"] is not None]
    return {"folds": len(rows), "MAE": float(np.mean(errors)) if errors else None,
            "p90_absolute_error": float(np.quantile(errors, .9)) if errors else None,
            "mean_train_scaled_error": float(np.mean(normalized)) if normalized else None}


def _stock_guard(r95: dict) -> dict:
    stock = dict(r95["q2_candidate_prediction_row"])
    metrics = ("diagnosed_plhiv", "alive_on_art", "tested_for_viral_load", "virally_suppressed")
    values = [float(stock[m]) for m in metrics]
    valid = all(np.isfinite(values)) and values[-1] >= 0 and all(a >= b for a, b in zip(values, values[1:]))
    return {"stock_cone_valid": bool(valid), "values_unchanged": {m: stock[m] for m in metrics},
            "conditional_rates_unchanged": True,
            "dynamics_coupled": False, "boundary": "diagnosis readout comparison does not validate state-flow reconciliation"}


def promotion_gate(history: dict, comparisons: list[dict], stock: dict) -> dict:
    baseline, selected = history["quarter_carry"], history["nested_selector"]
    history_pass = bool(selected["folds"] and selected["MAE"] < baseline["MAE"]
                        and selected["p90_absolute_error"] <= baseline["p90_absolute_error"])
    challenge_pass = all(r["selected_absolute_error"] <= min(r["carry_absolute_error"], r["R41_absolute_error"])
                         for r in comparisons) and len(comparisons) == 2
    blockers = ["historical_issue_dates_unverified", "Q2_already_inspected_before_experiment_design",
                "reporting_vs_diagnosis_volume_not_identified"]
    if not history_pass:
        blockers.append("nested_history_MAE_or_p90_fails_carry")
    if not challenge_pass:
        blockers.append("Q1_or_Q2_fails_matched_carry_or_R41")
    if not stock["stock_cone_valid"]:
        blockers.append("frozen_stock_cone_invalid")
    return {"status": "diagnostic_only", "champion": None, "history_pass": history_pass,
            "challenge_pass": challenge_pass, "blockers": blockers, "R41_changed": False,
            "allowed_claim": "retrospective monthly reported-diagnosis forecasting comparison"}


def run(output_dir: Path, *, q1_pdf: Path = Q1_PDF, q2_pdf: Path = Q2_PDF,
        r94_path: Path = R94_DEFAULT_REPORT, r95_path: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "report.json").exists():
        raise FileExistsError("Completed experiment is immutable; choose a fresh output directory")
    _write(output_dir / "contract.json", CONTRACT)
    q1_rows, q1_flags = extract_hasp_2026_q1_rows(extract_pdf_text(q1_pdf), q1_pdf)
    q2_rows, q2_flags = extract_hasp_2026_q2_rows(extract_pdf_text(q2_pdf), q2_pdf)
    q1, q2 = monthly_series(q1_rows), monthly_series(q2_rows)
    if (len(q1), len(q2), q1[0]["time_start"], q1[-1]["time_end"], q2[-1]["time_end"]) != (39, 42, "2023-01", "2026-03", "2026-06"):
        raise ValueError("R96 requires the locked January 2023 to June 2026 support contract")
    r94 = json.loads(r94_path.read_text())
    r95_path = r95_path or sandbox_repo_root() / "artifacts/runs" / R95_RUN_ID / "analysis/r95_2026_q2_hasp_intake_gate_report.json"
    r95 = json.loads(r95_path.read_text())
    evidence = [{**row, "report_vintage": vintage, "report_issue_date": None,
                 "availability_status": "historical_issue_date_unverified"}
                for vintage, rows in (("2026-Q1", q1_rows), ("2026-Q2", q2_rows)) for row in rows]
    _write(output_dir / "observation_ledger.json", evidence)
    revisions = [{"month": a["time_start"], "Q1_value": a["value"], "Q2_value": b["value"],
                  "Q1_row_hash": a["row_hash"], "Q2_row_hash": b["row_hash"]}
                 for a, b in zip(q1, q2) if a["value"] != b["value"]]
    if [r["time_start"] for r in q1] != [r["time_start"] for r in q2[:len(q1)]]:
        raise ValueError("Report vintages do not share aligned monthly support")
    # New Q2 rows are evaluation-only here; revised historical Q2 rows never enter Q1-origin fitting.
    combined = q1 + [r for r in q2 if r["time_start"] > q1[-1]["time_start"]]
    rows, choices = backtest([r for r in q1 if r["time_start"] < "2026-01"])
    summaries = {f: _summary([r for r in rows if r["family"] == f]) for f in FAMILIES}
    summaries["nested_selector"] = _summary(choices)
    comparisons, monthly_predictions, candidate_challenges = [], [], []
    for quarter, origin, frozen in (("2026-Q1", 36, r94), ("2026-Q2", 39, r95)):
        train, target = combined[:origin], combined[origin:origin + QUARTER_MONTHS]
        if len(target) != QUARTER_MONTHS:
            raise ValueError("Incomplete quarterly target")
        values = [r["value"] for r in train]
        selected, scores = choose_family(train)
        actual = sum(r["value"] for r in target)
        key = "q1" if quarter.endswith("Q1") else "q2"
        published_actual = float(frozen[f"{key}_target_row"]["new_diagnosed_cases_period"])
        if actual != published_actual:
            raise ValueError("Monthly diagnosis sum disagrees with matching published quarterly target")
        R41 = float(frozen[f"{key}_candidate_prediction_row"]["new_diagnosed_cases_period"])
        for family in FAMILIES:
            result = forecast(values, family)
            prediction = sum(result["points"])
            candidate_challenges.append({"quarter": quarter, "family": family, "prediction": prediction,
                                         "actual": actual, "absolute_error": abs(prediction - actual), "selected": family == selected})
            for i, row in enumerate(target):
                monthly_predictions.append({"quarter": quarter, "month": row["time_start"], "family": family,
                                            "actual": row["value"], "prediction": result["points"][i],
                                            "lower95": result["lower95"][i] if result["lower95"] else None,
                                            "upper95": result["upper95"][i] if result["upper95"] else None,
                                            "origin": train[-1]["time_start"], "selected": family == selected})
        selected_prediction = sum(forecast(values, selected)["points"])
        carry = sum(values[-QUARTER_MONTHS:])
        comparisons.append({"quarter": quarter, "selected_family": selected, "inner_scores": scores,
                            "actual": actual, "selected_prediction": selected_prediction,
                            "selected_absolute_error": abs(selected_prediction - actual),
                            "carry_prediction": carry, "carry_absolute_error": abs(carry - actual),
                            "R41_prediction": R41, "R41_absolute_error": abs(R41 - actual),
                            "training_vintage": "2026-Q1", "training_max_month": train[-1]["time_start"],
                            "frozen_R41_carry_prediction": frozen[f"{key}_carry_forward_prediction_row"]["new_diagnosed_cases_period"]})
    stock = _stock_guard(r95)
    gate = promotion_gate(summaries, comparisons, stock)
    future_family, future_scores = choose_family(q2)
    future = forecast([r["value"] for r in q2], future_family)
    state = forecast([r["value"] for r in q1], "local_level")
    shocks = []
    for i, row in enumerate(q2[-QUARTER_MONTHS:]):
        variance = state["log_variances"][i]
        z = (np.log1p(row["value"]) - state["log_means"][i]) / np.sqrt(variance) if variance > 0 else None
        shocks.append({"month": row["time_start"], "standardized_surprise": float(z) if z is not None else None,
                       "bounded_descriptive_driver": float(z / np.hypot(1, z)) if z is not None else None,
                       "used_in_forecast": False, "interpretation": "post-outcome innovation; cause unidentifiable"})
    report = {"run_id": output_dir.name, "generated_at": datetime.now(timezone.utc).isoformat(),
              "contract": CONTRACT, "gate": gate, "history": summaries,
              "historical_folds": rows, "nested_choices": choices, "quarter_comparisons": comparisons,
              "candidate_challenges": candidate_challenges, "monthly_predictions": monthly_predictions,
              "report_revisions": revisions, "stock_guard": stock, "Q2_innovations": shocks,
              "extraction": {"Q1_rows": len(q1_rows), "Q2_rows": len(q2_rows), "Q1_months": len(q1),
                             "Q2_months": len(q2), "Q1_flags": q1_flags, "Q2_flags": q2_flags},
              "future_Q3": {"family": future_family, "selection_scores": future_scores, **future,
                            "status": "provisional_research_forecast", "origin": "2026-06",
                            "months": ["2026-07", "2026-08", "2026-09"],
                            "available_at_generation": datetime.now(timezone.utc).date().isoformat(), "prospective": False,
                            "quarter_prediction": sum(future["points"]), "champion_replacement": False}}
    _write(output_dir / "report.json", report)
    import scipy
    source_files = [Path(__file__), Path(__file__).with_name("hasp_monthly_table.py"),
                    Path(__file__).with_name("r94_2026_q1_hasp_intake_gate.py"),
                    Path(__file__).with_name("r95_2026_q2_hasp_intake_gate.py")]
    manifest = {"run_id": output_dir.name, "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "inputs": {str(p): _digest(p) for p in (q1_pdf, q2_pdf, r94_path, r95_path)},
                "code": {str(p): _digest(p) for p in source_files},
                "outputs": {p.name: _digest(p) for p in output_dir.glob("*.json")}}
    _write(output_dir / "claim_card.json", gate)
    for name, table in (("historical_folds", rows), ("candidate_challenges", candidate_challenges),
                        ("monthly_predictions", monthly_predictions)):
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    manifest["outputs"] = {p.name: _digest(p) for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json"}
    _write(output_dir / "manifest.json", manifest)
    return report


def plot_report(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    ax = axes[0, 0]
    families = [*FAMILIES, "nested_selector"]
    errors = [report["history"][f]["MAE"] for f in families]
    ax.barh(families, errors, color=["#8c9296"] * 5 + ["#167d8d"])
    ax.set(xlabel="Quarterly MAE (reported diagnoses)", title="A  Eight historical quarterly blocks, 2024-2025")
    for i, value in enumerate(errors):
        ax.text(value, i, f" {value:,.0f}", va="center")
    ax = axes[0, 1]
    comparison = report["quarter_comparisons"]
    x = np.arange(len(comparison))
    for offset, prefix, label, color in ((-.26, "carry", "Quarter carry", "#8c9296"), (0, "R41", "Frozen R41", "#d79039"), (.26, "selected", "Train-selected monthly model", "#167d8d")):
        ax.bar(x + offset, [r[f"{prefix}_absolute_error"] for r in comparison], .25, label=label, color=color)
    ax.set(xticks=x, xticklabels=[r["quarter"] for r in comparison], ylabel="Absolute error (reported diagnoses)", title="B  Retrospective 2026 quarterly challenges")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    data = [r for r in report["monthly_predictions"] if r["family"] == "local_level"]
    x = np.arange(len(data))
    ax.fill_between(x, [r["lower95"] for r in data], [r["upper95"] for r in data], color="#167d8d", alpha=.17, label="95% conditional observation interval")
    ax.plot(x, [r["prediction"] for r in data], color="#167d8d", marker="o", label="Local-level forecast")
    ax.plot(x, [r["actual"] for r in data], color="#242b32", marker="s", label="Reported count")
    ax.axvline(2.5, color="#aaa", linestyle=":")
    ax.set(xticks=x, xticklabels=[r["month"] for r in data], ylabel="Diagnoses / month", title="C  Forecasts fixed at each quarter's starting origin")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.axis("off")
    text = (f"D  Evidence and decision\n\n"
            f"Monthly rows recovered: Q1 24 -> {report['extraction']['Q1_months']}; Q2 12 -> {report['extraction']['Q2_months']}\n"
            "March 2026 revised: 1,536 (Q1) -> 1,533 (Q2)\n"
            "Q2 forecasts use the unrevised Q1 training vintage.\n\n"
            f"Nested historical gate: {report['gate']['history_pass']}\n"
            f"Both-quarter carry/R41 gate: {report['gate']['challenge_pass']}\n"
            f"Frozen stock cone valid: {report['stock_guard']['stock_cone_valid']}\n\n"
            "No model promotion. Historical release dates unverified.\n"
            "Reporting availability and diagnosis volume remain confounded.\n"
            "Q2 was already inspected before this experiment.\n"
            "Intervals omit parameter/source-revision uncertainty.")
    ax.text(0, 1, text, transform=ax.transAxes, va="top", fontsize=11, linespacing=1.6)
    fig.suptitle("R96 | Monthly diagnosis state: extraction repair and forecast falsification", fontsize=16)
    for ax in axes.flat[:3]:
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=200)
    plt.close(fig)
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["matplotlib"] = matplotlib.__version__
        manifest["outputs"].update({path.with_suffix("." + ext).name: _digest(path.with_suffix("." + ext))
                                    for ext in ("png", "svg", "pdf")})
        _write(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=sandbox_repo_root() / "artifacts/runs" / RUN_ID)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = run(args.output_dir)
    if args.plot:
        plot_report(report, args.output_dir / "dashboard")
    print(json.dumps({"gate": report["gate"], "history": report["history"], "quarters": report["quarter_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
