from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_source_row_path
from .art_retention_evidence import ART_RETENTION_PROCESS_METRICS
from .metrics import inv_logit, logit, quarter_sort_key
from .observation_ledger import build_contract_row_hash, build_observation_contract_lookup
from .runtime import read_json


MONTHLY_SHOCK_SCHEMA_VERSION = "phase3_dynamic_monthly_native_shock_process.v1"
MONTHLY_SIGNAL_METRICS: tuple[str, ...] = (
    "new_diagnosed_cases_period",
    "new_diagnosed_cases_monthly",
    "diagnosed_plhiv",
    "alive_on_art",
    "newly_enrolled_to_treatment",
    "tested_for_viral_load",
    "virally_suppressed",
    "on_art_not_suppressed",
    "deaths_reported_period",
    "median_cd4_at_enrollment",
    *ART_RETENTION_PROCESS_METRICS,
)
MONTHLY_SIGNAL_ALLOWED_ROLES = {"direct_target", "auxiliary_likelihood", "prior_context"}


@dataclass(frozen=True, slots=True)
class MonthlyShockContext:
    epigraph_root: Path
    source_run_id: str
    baseline_source_run_id: str | None = None


def _month_index(month_label: str) -> int:
    year_text, month_text = str(month_label).split("-", 1)
    return int(year_text) * 12 + int(month_text[:2]) - 1


def _month_label(month_index: int) -> str:
    year = int(month_index) // 12
    month = int(month_index) % 12 + 1
    return f"{year:04d}-{month:02d}"


def _quarter_month_range(quarter: str) -> tuple[int, int]:
    year_text, q_text = str(quarter).split("-Q", 1)
    quarter_index = int(q_text)
    start_month = (quarter_index - 1) * 3 + 1
    start = int(year_text) * 12 + start_month - 1
    return start, start + 2


def _time_label(row: dict[str, Any]) -> str:
    return str(row.get("time") or row.get("period_end") or row.get("period_start") or "")


def _provenance_signature(row: dict[str, Any], contract: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("metric_name") or ""),
        str(row.get("source_quality_tier") or ""),
        str(row.get("measurement_class") or ""),
        str(row.get("series_kind") or ""),
        str(contract.get("observation_role") or ""),
        str(contract.get("allowed_use") or ""),
        str(contract.get("support_partition") or ""),
    )


def _metric_priority(row: dict[str, Any]) -> tuple[float, float, str]:
    quality = str(row.get("source_quality_tier") or "")
    quality_rank = {
        "official_user_provided_slide": 0.0,
        "official_doh_archive": 1.0,
        "official_local_corpus": 1.0,
        "official": 2.0,
    }.get(quality, 5.0)
    confidence = -float(row.get("evidence_confidence") or 0.0)
    return quality_rank, confidence, str(row.get("source_id") or "")


def load_monthly_signal_rows(context: MonthlyShockContext) -> list[dict[str, Any]]:
    archive_path = (
        Path(context.epigraph_root)
        / "artifacts"
        / "runs"
        / context.source_run_id
        / "harp_archive"
        / "historical_metric_rows.json"
    )
    contract_lookup, _ledger = build_observation_contract_lookup(
        Path(context.epigraph_root),
        source_run_id=context.source_run_id,
        baseline_source_run_id=context.baseline_source_run_id,
    )
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(list(read_json(archive_path, default=[]) or [])):
        metric_name = str(row.get("metric_name") or "")
        if metric_name not in MONTHLY_SIGNAL_METRICS:
            continue
        if not str(row.get("series_kind") or "").startswith(("monthly", "quarterly")):
            continue
        time_label = _time_label(row)
        if not time_label:
            continue
        source_path = build_source_row_path(archive_path, row_index)
        row_hash = build_contract_row_hash(row=row, source_path=source_path)
        contract = dict(contract_lookup.get(row_hash) or {})
        if str(contract.get("observation_role") or "") not in MONTHLY_SIGNAL_ALLOWED_ROLES:
            continue
        enriched = dict(row)
        enriched["_contract"] = contract
        enriched["_month_index"] = _month_index(time_label)
        enriched["_source_path"] = source_path
        enriched["_row_hash"] = row_hash
        rows.append(enriched)
    return sorted(rows, key=lambda row: (int(row["_month_index"]), str(row.get("metric_name") or "")))


def _chosen_month_metric_values(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["_month_index"]), str(row.get("metric_name") or "")), []).append(row)
    chosen: dict[tuple[int, str], dict[str, Any]] = {}
    for key, candidates in grouped.items():
        chosen[key] = min(candidates, key=_metric_priority)
    return chosen


def _robust_monthly_deviation(values: list[float]) -> list[float]:
    return [score for score, _surprise in _robust_monthly_deviation_with_surprise(values)]


def _robust_monthly_deviation_with_surprise(values: list[float]) -> list[tuple[float, float]]:
    if len(values) < 2:
        return [(0.0, 0.0) for _ in values]
    deltas = np.diff(np.asarray(values, dtype=np.float64))
    scores = [(0.0, 0.0)]
    for index, delta in enumerate(deltas):
        reference = deltas[:index] if index > 0 else deltas[:1]
        center = float(np.median(reference))
        scale = float(np.median(np.abs(reference - center)))
        if scale <= 0.0:
            scale = float(np.mean(np.abs(reference - center))) if reference.size else 0.0
        if scale <= 0.0:
            scale = max(float(np.max(np.abs(reference))) if reference.size else 0.0, 1.0)
        surprise = float((float(delta) - center) / scale)
        scores.append((float(np.tanh(surprise)), abs(surprise)))
    return scores


def _positive_autocorrelation(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    previous = array[:-1]
    current = array[1:]
    if float(np.std(previous)) <= 0.0 or float(np.std(current)) <= 0.0:
        return 0.0
    correlation = float(np.corrcoef(previous, current)[0, 1])
    if not np.isfinite(correlation):
        return 0.0
    return float(np.clip(correlation, 0.0, 1.0))


def _tukey_upper_fence(values: list[float]) -> float:
    finite = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return 0.0
    if finite.size < 4:
        return float(np.max(finite))
    q1, q3 = np.percentile(finite, [25.0, 75.0])
    return float(q3 + 1.5 * float(q3 - q1))


def _monthly_feature_table(rows: list[dict[str, Any]], *, train_end_month: int) -> dict[int, dict[str, float]]:
    chosen = _chosen_month_metric_values([row for row in rows if int(row["_month_index"]) <= int(train_end_month)])
    metric_scores_by_month: dict[int, list[float]] = {}
    metric_surprise_by_month: dict[int, list[float]] = {}
    support_shift_by_month: dict[int, float] = {}
    for metric_name in MONTHLY_SIGNAL_METRICS:
        metric_items = [
            (month, row)
            for (month, metric), row in chosen.items()
            if metric == metric_name
        ]
        if not metric_items:
            continue
        metric_items = sorted(metric_items)
        values = [float(np.log1p(max(float(row.get("value") or 0.0), 0.0))) for _month, row in metric_items]
        scores = _robust_monthly_deviation_with_surprise(values)
        previous_signature: tuple[str, ...] | None = None
        for (month, row), (score, surprise) in zip(metric_items, scores):
            metric_scores_by_month.setdefault(month, []).append(float(score))
            metric_surprise_by_month.setdefault(month, []).append(float(surprise))
            signature = _provenance_signature(row, dict(row.get("_contract") or {}))
            if previous_signature is not None and signature != previous_signature:
                support_shift_by_month[month] = 1.0
            previous_signature = signature
    months = sorted(set(metric_scores_by_month).union(support_shift_by_month))
    composite_values = [
        float(np.mean(np.asarray(metric_scores_by_month.get(month) or [0.0], dtype=np.float64)))
        for month in months
    ]
    surprise_values = [
        float(np.max(np.asarray(metric_surprise_by_month.get(month) or [0.0], dtype=np.float64)))
        for month in months
    ]
    residual_threshold = _tukey_upper_fence(surprise_values)
    table: dict[int, dict[str, float]] = {}
    for month, composite, surprise in zip(months, composite_values, surprise_values):
        support_shift = float(support_shift_by_month.get(month) or 0.0)
        residual_shock = 1.0 if residual_threshold > 0.0 and float(surprise) > residual_threshold else 0.0
        table[month] = {
            "monthly_deviation": float(np.clip(composite, -1.0, 1.0)),
            "support_shift": support_shift,
            "residual_shock": residual_shock,
            "support_shift_deviation": float(support_shift * np.clip(composite, -1.0, 1.0)),
        }
    return table


def _feature_autocorrelation(table: dict[int, dict[str, float]]) -> float:
    values = [float(table[month].get("monthly_deviation") or 0.0) for month in sorted(table)]
    return _positive_autocorrelation(values)


def _feature_for_month(table: dict[int, dict[str, float]], month: int, *, train_end_month: int, phi: float) -> dict[str, float]:
    if month in table and month <= train_end_month:
        return dict(table[month])
    if not table:
        return {
            "monthly_deviation": 0.0,
            "support_shift": 0.0,
            "residual_shock": 0.0,
            "support_shift_deviation": 0.0,
        }
    last_month = max(month for month in table if month <= train_end_month)
    last = dict(table[last_month])
    step = max(int(month) - int(last_month), 0)
    decay = float(phi) ** step
    return {
        "monthly_deviation": float(last.get("monthly_deviation") or 0.0) * decay,
        "support_shift": float(last.get("support_shift") or 0.0) * decay,
        "residual_shock": float(last.get("residual_shock") or 0.0) * decay,
        "support_shift_deviation": float(last.get("support_shift_deviation") or 0.0) * decay,
    }


def build_quarterly_monthly_shock_features(
    context: MonthlyShockContext,
    *,
    train_end_quarter: str,
    quarters: list[str],
) -> dict[str, Any]:
    _start, train_end_month = _quarter_month_range(train_end_quarter)
    rows = load_monthly_signal_rows(context)
    table = _monthly_feature_table(rows, train_end_month=train_end_month)
    phi = _feature_autocorrelation(table)
    feature_names = ["monthly_deviation", "support_shift", "residual_shock", "support_shift_deviation"]
    quarter_features: dict[str, dict[str, float]] = {}
    for quarter in quarters:
        start_month, end_month = _quarter_month_range(quarter)
        month_features = [_feature_for_month(table, month, train_end_month=train_end_month, phi=phi) for month in range(start_month, end_month + 1)]
        quarter_features[quarter] = {
            name: float(np.mean(np.asarray([features[name] for features in month_features], dtype=np.float64)))
            for name in feature_names
        }
    return {
        "schema_version": MONTHLY_SHOCK_SCHEMA_VERSION,
        "feature_names": feature_names,
        "quarter_features": quarter_features,
        "monthly_row_count": len(rows),
        "train_end_quarter": train_end_quarter,
        "train_end_month": _month_label(train_end_month),
        "feature_autocorrelation": phi,
        "contract": "Monthly HARP signal is estimated only from rows at or before the forecast-origin quarter; holdout months receive decayed last-observed monthly signal.",
    }


def _fit_linear_effect(
    train_quarters: list[str],
    features: dict[str, dict[str, float]],
    residuals: dict[str, float],
    feature_names: list[str],
) -> dict[str, Any]:
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for quarter in train_quarters:
        row = dict(features.get(quarter) or {})
        if quarter not in residuals:
            continue
        x_rows.append([float(row.get(name) or 0.0) for name in feature_names])
        y_rows.append(float(residuals[quarter]))
    if not x_rows:
        return {"coefficients": [0.0 for _ in feature_names], "effect_bound": 0.0, "train_row_count": 0}
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    coefficients = np.linalg.pinv(x) @ y
    fitted = x @ coefficients
    effect_bound = _tukey_upper_fence([abs(float(value)) for value in y - fitted])
    if effect_bound <= 0.0:
        effect_bound = _tukey_upper_fence([abs(float(value)) for value in y])
    return {
        "coefficients": [float(value) for value in coefficients],
        "effect_bound": float(effect_bound),
        "train_row_count": len(x_rows),
    }


def _linear_effect(feature_row: dict[str, float], fit: dict[str, Any], feature_names: list[str]) -> float:
    coefficients = np.asarray(list(fit.get("coefficients") or []), dtype=np.float64)
    x = np.asarray([float(feature_row.get(name) or 0.0) for name in feature_names], dtype=np.float64)
    if coefficients.size != x.size:
        return 0.0
    raw = float(x @ coefficients)
    bound = float(fit.get("effect_bound") or 0.0)
    if bound > 0.0:
        return float(np.clip(raw, -bound, bound))
    return raw


def apply_monthly_shock_to_paths(
    *,
    context: MonthlyShockContext,
    dataset: Any,
    hazard_paths: dict[str, Any],
    incidence_paths: dict[str, Any],
) -> dict[str, Any]:
    train_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not train_rows or not holdout_rows:
        return {"hazard_paths": hazard_paths, "incidence_paths": incidence_paths, "diagnostics": {"status": "not_evaluable"}}
    train_quarters = [str(row.get("quarter") or "") for row in train_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    all_quarters = train_quarters + holdout_quarters
    features_payload = build_quarterly_monthly_shock_features(
        context,
        train_end_quarter=train_quarters[-1],
        quarters=all_quarters,
    )
    features = dict(features_payload.get("quarter_features") or {})
    feature_names = list(features_payload.get("feature_names") or [])
    adjusted_hazards = {
        key: {transition: float(value) for transition, value in dict(values).items()}
        for key, values in dict(hazard_paths.get("holdout_hazard_map") or {}).items()
    }
    adjusted_train_hazards = {
        key: {transition: float(value) for transition, value in dict(values).items()}
        for key, values in dict(hazard_paths.get("train_hazard_map") or {}).items()
    }
    diagnostics: dict[str, Any] = {"monthly_features": features_payload, "targets": {}}
    eps = float(getattr(dataset, "eps", np.finfo(np.float32).eps))
    for transition in ("U_to_D",):
        residuals = {}
        for row in train_rows:
            quarter = str(row.get("quarter") or "")
            observed = float((row.get("hazards") or {}).get(transition) or 0.0)
            base = float((adjusted_train_hazards.get(quarter) or {}).get(transition) or 0.0)
            residuals[quarter] = logit(observed, eps=eps) - logit(base, eps=eps)
        fit = _fit_linear_effect(train_quarters, features, residuals, feature_names)
        diagnostics["targets"][transition] = fit
        for target_map in (adjusted_train_hazards, adjusted_hazards):
            for quarter, values in target_map.items():
                base = float(values.get(transition) or 0.0)
                effect = _linear_effect(dict(features.get(quarter) or {}), fit, feature_names)
                values[transition] = float(inv_logit(logit(base, eps=eps) + effect))
    adjusted_incidence = dict(incidence_paths)
    train_incidence_hazard = {quarter: float(value) for quarter, value in dict(incidence_paths.get("train_incidence_hazard_map") or {}).items()}
    holdout_incidence_hazard = {quarter: float(value) for quarter, value in dict(incidence_paths.get("holdout_incidence_hazard_map") or {}).items()}
    residuals = {}
    for row in train_rows:
        quarter = str(row.get("quarter") or "")
        observed = float((row.get("stock_balance") or {}).get("incidence_hazard_per_s_eff") or 0.0)
        base = float(train_incidence_hazard.get(quarter) or 0.0)
        residuals[quarter] = float(np.log1p(max(observed, 0.0)) - np.log1p(max(base, 0.0)))
    fit = _fit_linear_effect(train_quarters, features, residuals, feature_names)
    diagnostics["targets"]["incidence_hazard_per_s_eff"] = fit
    for target_map in (train_incidence_hazard, holdout_incidence_hazard):
        for quarter, base in list(target_map.items()):
            effect = _linear_effect(dict(features.get(quarter) or {}), fit, feature_names)
            target_map[quarter] = float(max(np.expm1(np.log1p(max(float(base), 0.0)) + effect), 0.0))
    adjusted_incidence["train_incidence_hazard_map"] = train_incidence_hazard
    adjusted_incidence["holdout_incidence_hazard_map"] = holdout_incidence_hazard
    adjusted_paths = dict(hazard_paths)
    adjusted_paths["train_hazard_map"] = adjusted_train_hazards
    adjusted_paths["holdout_hazard_map"] = adjusted_hazards
    return {
        "hazard_paths": adjusted_paths,
        "incidence_paths": adjusted_incidence,
        "diagnostics": diagnostics,
    }
