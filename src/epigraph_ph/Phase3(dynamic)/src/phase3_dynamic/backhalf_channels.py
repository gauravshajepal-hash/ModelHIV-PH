from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .art_retention_evidence import (
    ART_RETENTION_PROCESS_METRICS,
    REENGAGEMENT_SENSITIVITY_MODES,
    build_art_retention_quarter_evidence,
)
from .metrics import inv_logit, logit, quarter_ordinal, quarter_sort_key
from .monthly_shock import (
    MonthlyShockContext,
    _chosen_month_metric_values,
    _fit_linear_effect,
    _linear_effect,
    _month_label,
    _quarter_month_range,
    load_monthly_signal_rows,
)


BACKHALF_CHANNEL_SCHEMA_VERSION = "phase3_dynamic_backhalf_transition_channels.v1"
BACKHALF_MONTHLY_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "newly_enrolled_to_treatment",
    "tested_for_viral_load",
    "virally_suppressed",
    "on_art_not_suppressed",
    "deaths_reported_period",
    *ART_RETENTION_PROCESS_METRICS,
)
BACKHALF_TRANSITION_FEATURES: dict[str, tuple[str, ...]] = {
    "D_to_A": (
        "art_initiation_pressure",
        "art_stock_growth",
        "service_capacity",
        "backhalf_support",
    ),
    "A_to_T": (
        "vl_testing_loss_pressure",
        "service_capacity",
        "backhalf_support",
    ),
    "T_to_V": (
        "suppression_capacity",
        "unsuppressed_pressure",
        "service_capacity",
        "backhalf_support",
    ),
    "A_to_L": (
        "art_ltfu_pressure",
        "art_interruption_evidence_pressure",
        "art_retention_gap_pressure",
        "art_retention_support",
        "art_process_removal_pressure",
        "latent_unobserved_attrition_pressure",
        "art_transfer_out_pressure",
        "art_stopped_pressure",
        "service_mortality_pressure",
        "vl_testing_loss_pressure",
        "backhalf_support",
    ),
    "T_to_L": (
        "art_ltfu_pressure",
        "art_interruption_evidence_pressure",
        "art_retention_gap_pressure",
        "art_retention_support",
        "art_process_removal_pressure",
        "latent_unobserved_attrition_pressure",
        "art_transfer_out_pressure",
        "art_stopped_pressure",
        "unsuppressed_pressure",
        "service_mortality_pressure",
        "backhalf_support",
    ),
    "V_to_L": (
        "art_ltfu_pressure",
        "art_interruption_evidence_pressure",
        "art_retention_gap_pressure",
        "art_retention_support",
        "art_process_removal_pressure",
        "latent_unobserved_attrition_pressure",
        "art_transfer_out_pressure",
        "art_stopped_pressure",
        "service_mortality_pressure",
        "backhalf_support",
    ),
    "L_to_R": (
        "reengagement_pressure",
        "art_reengagement_evidence_pressure",
        "latent_reengagement_balance_pressure",
        "art_retention_support",
        "art_stock_growth",
        "service_capacity",
        "backhalf_support",
    ),
    "R_to_A": (
        "art_reengagement_evidence_pressure",
        "latent_reengagement_balance_pressure",
        "art_retention_support",
        "service_capacity",
        "art_stock_growth",
        "backhalf_support",
    ),
}


@dataclass(frozen=True, slots=True)
class BackHalfChannelContext:
    epigraph_root: Path
    source_run_id: str
    baseline_source_run_id: str | None = None
    reengagement_sensitivity_mode: str = "public_stock_flow_proxy"


def _shock_context(context: BackHalfChannelContext) -> MonthlyShockContext:
    return MonthlyShockContext(
        epigraph_root=Path(context.epigraph_root),
        source_run_id=context.source_run_id,
        baseline_source_run_id=context.baseline_source_run_id,
    )


def _quarter_from_ordinal(index: int) -> str:
    year = int(index) // 4
    quarter = int(index) % 4 + 1
    return f"{year:04d}-Q{quarter}"


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(np.log1p(max(float(numerator), 0.0)) - np.log1p(max(float(denominator), 0.0)))


def _robust_standardize(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    scale = float(np.median(np.abs(array - center)))
    if scale <= 0.0:
        scale = float(np.std(array))
    if scale <= 0.0:
        scale = max(float(np.max(np.abs(array - center))), 1.0)
    return [float((value - center) / scale) for value in array]


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


def _monthly_metric_values(rows: list[dict[str, Any]], train_end_month: int) -> dict[int, dict[str, float]]:
    chosen = _chosen_month_metric_values([row for row in rows if int(row["_month_index"]) <= int(train_end_month)])
    values: dict[int, dict[str, float]] = {}
    for (month, metric_name), row in sorted(chosen.items()):
        if metric_name not in BACKHALF_MONTHLY_METRICS:
            continue
        values.setdefault(int(month), {})[metric_name] = float(row.get("value") or 0.0)
    return values


def _month_to_quarter(month_index: int) -> str:
    year = int(month_index) // 12
    month = int(month_index) % 12 + 1
    quarter = ((month - 1) // 3) + 1
    return f"{year:04d}-Q{quarter}"


def _quarter_months(monthly_values: dict[int, dict[str, float]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for month in sorted(monthly_values):
        grouped.setdefault(_month_to_quarter(month), []).append(int(month))
    return grouped


def _quarter_summary(months: list[int], monthly_values: dict[int, dict[str, float]]) -> dict[str, Any]:
    ordered = sorted(months)
    snapshots = {
        "diagnosed_plhiv": [],
        "alive_on_art": [],
        "tested_for_viral_load": [],
        "virally_suppressed": [],
        "on_art_not_suppressed": [],
    }
    newly_enrolled_sum = 0.0
    newly_enrolled_count = 0
    deaths_sum = 0.0
    observed_metric_count = 0
    process_sums = {metric: 0.0 for metric in ART_RETENTION_PROCESS_METRICS}
    process_counts = {metric: 0 for metric in ART_RETENTION_PROCESS_METRICS}
    process_last = {metric: None for metric in ART_RETENTION_PROCESS_METRICS}
    for month in ordered:
        values = monthly_values.get(month) or {}
        observed_metric_count += sum(1 for metric in BACKHALF_MONTHLY_METRICS if values.get(metric) is not None)
        if values.get("newly_enrolled_to_treatment") is not None:
            newly_enrolled_sum += float(values.get("newly_enrolled_to_treatment") or 0.0)
            newly_enrolled_count += 1
        deaths_sum += float(values.get("deaths_reported_period") or 0.0)
        for metric_name in snapshots:
            if values.get(metric_name) is not None:
                snapshots[metric_name].append(float(values[metric_name]))
        for metric_name in ART_RETENTION_PROCESS_METRICS:
            if values.get(metric_name) is not None:
                process_sums[metric_name] += float(values.get(metric_name) or 0.0)
                process_counts[metric_name] += 1
                process_last[metric_name] = float(values.get(metric_name) or 0.0)
    result = {
        "diagnosed_last": float(snapshots["diagnosed_plhiv"][-1]) if snapshots["diagnosed_plhiv"] else None,
        "art_last": float(snapshots["alive_on_art"][-1]) if snapshots["alive_on_art"] else None,
        "tested_last": float(snapshots["tested_for_viral_load"][-1]) if snapshots["tested_for_viral_load"] else None,
        "suppressed_last": float(snapshots["virally_suppressed"][-1]) if snapshots["virally_suppressed"] else None,
        "unsuppressed_last": float(snapshots["on_art_not_suppressed"][-1]) if snapshots["on_art_not_suppressed"] else None,
        "newly_enrolled_sum": float(newly_enrolled_sum),
        "newly_enrolled_count": int(newly_enrolled_count),
        "deaths_sum": float(deaths_sum),
        "observed_metric_count": int(observed_metric_count),
        "possible_metric_count": int(len(ordered) * len(BACKHALF_MONTHLY_METRICS)),
    }
    for metric_name in ART_RETENTION_PROCESS_METRICS:
        result[f"{metric_name}_sum"] = float(process_sums[metric_name])
        result[f"{metric_name}_count"] = int(process_counts[metric_name])
        result[f"{metric_name}_last"] = process_last[metric_name]
    return result


def _raw_backhalf_features_with_evidence(
    monthly_values: dict[int, dict[str, float]],
    *,
    reengagement_sensitivity_mode: str = "public_stock_flow_proxy",
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    grouped = _quarter_months(monthly_values)
    summaries = {quarter: _quarter_summary(months, monthly_values) for quarter, months in grouped.items()}
    raw: dict[str, dict[str, float]] = {}
    retention_evidence: dict[str, dict[str, Any]] = {}
    ordered_quarters = sorted(summaries, key=quarter_sort_key)
    for quarter in ordered_quarters:
        summary = summaries[quarter]
        previous_quarter = _quarter_from_ordinal(quarter_ordinal(quarter) - 1)
        previous = summaries.get(previous_quarter) or {}
        art = float(summary.get("art_last") or 0.0)
        diagnosed = float(summary.get("diagnosed_last") or 0.0)
        previous_art = float(previous.get("art_last") or 0.0)
        diagnosed_gap = max(diagnosed - art, 0.0)
        art_growth = art - previous_art
        tested = summary.get("tested_last")
        suppressed = summary.get("suppressed_last")
        unsuppressed = summary.get("unsuppressed_last")
        if unsuppressed is None and suppressed is not None:
            unsuppressed = max(art - float(suppressed), 0.0)
        art_retention = build_art_retention_quarter_evidence(
            summary=summary,
            previous_summary=previous,
            diagnosed_gap=diagnosed_gap,
            reengagement_sensitivity_mode=reengagement_sensitivity_mode,
        )
        retention_evidence[quarter] = art_retention
        retention_features = dict(art_retention.get("pressure_features") or {})
        raw[quarter] = {
            "art_initiation_pressure": _safe_log_ratio(float(summary.get("newly_enrolled_sum") or 0.0), diagnosed_gap),
            "art_stock_growth": _safe_log_ratio(max(art_growth, 0.0), max(previous_art, 0.0)),
            "art_ltfu_pressure": float(retention_features.get("art_ltfu_pressure") or 0.0),
            "art_interruption_evidence_pressure": float(retention_features.get("art_interruption_evidence_pressure") or 0.0),
            "reengagement_pressure": float(retention_features.get("reengagement_pressure") or 0.0),
            "art_reengagement_evidence_pressure": float(retention_features.get("art_reengagement_evidence_pressure") or 0.0),
            "latent_reengagement_balance_pressure": float(retention_features.get("latent_reengagement_balance_pressure") or 0.0),
            "latent_unobserved_attrition_pressure": float(retention_features.get("latent_unobserved_attrition_pressure") or 0.0),
            "art_retention_gap_pressure": float(retention_features.get("art_retention_gap_pressure") or 0.0),
            "art_retention_support": float(retention_features.get("art_retention_support") or 0.0),
            "art_transfer_out_pressure": float(retention_features.get("art_transfer_out_pressure") or 0.0),
            "art_stopped_pressure": float(retention_features.get("art_stopped_pressure") or 0.0),
            "art_death_pressure": float(retention_features.get("art_death_pressure") or 0.0),
            "art_process_removal_pressure": float(retention_features.get("art_process_removal_pressure") or 0.0),
            "vl_testing_loss_pressure": 0.0 if tested is None else _safe_log_ratio(max(art - float(tested), 0.0), art),
            "suppression_capacity": 0.0 if suppressed is None else _safe_log_ratio(float(suppressed), art),
            "unsuppressed_pressure": 0.0 if unsuppressed is None else _safe_log_ratio(float(unsuppressed), art),
            "service_mortality_pressure": _safe_log_ratio(float(summary.get("deaths_sum") or 0.0), art),
            "service_capacity": _safe_log_ratio(art, diagnosed),
            "backhalf_support": float(summary.get("observed_metric_count") or 0.0) / max(float(summary.get("possible_metric_count") or 0.0), 1.0),
        }
    return raw, retention_evidence


def _raw_backhalf_features(monthly_values: dict[int, dict[str, float]]) -> dict[str, dict[str, float]]:
    raw, _retention_evidence = _raw_backhalf_features_with_evidence(monthly_values)
    return raw


def _standardize_feature_table(raw: dict[str, dict[str, float]], feature_names: list[str]) -> dict[str, dict[str, float]]:
    ordered_quarters = sorted(raw, key=quarter_sort_key)
    passthrough_features = {"backhalf_support", "art_retention_support"}
    standardized_columns = {
        feature: _robust_standardize([float((raw.get(quarter) or {}).get(feature) or 0.0) for quarter in ordered_quarters])
        for feature in feature_names
        if feature not in passthrough_features
    }
    result: dict[str, dict[str, float]] = {}
    for index, quarter in enumerate(ordered_quarters):
        row = dict(raw[quarter])
        for feature, values in standardized_columns.items():
            row[feature] = float(np.tanh(float(values[index])))
        result[quarter] = {feature: float(row.get(feature) or 0.0) for feature in feature_names}
    return result


def _feature_for_quarter(
    table: dict[str, dict[str, float]],
    quarter: str,
    *,
    train_end_quarter: str,
    feature_names: list[str],
    phis: dict[str, float],
) -> dict[str, float]:
    if quarter in table and quarter_sort_key(quarter) <= quarter_sort_key(train_end_quarter):
        return dict(table[quarter])
    valid = [item for item in table if quarter_sort_key(item) <= quarter_sort_key(train_end_quarter)]
    if not valid:
        return {feature: 0.0 for feature in feature_names}
    last_quarter = max(valid, key=quarter_sort_key)
    last_index = quarter_ordinal(last_quarter)
    target_index = quarter_ordinal(quarter)
    step = max(int(target_index - last_index), 0)
    last = dict(table[last_quarter])
    return {
        feature: float(last.get(feature) or 0.0) * (float(phis.get(feature) or 0.0) ** step)
        for feature in feature_names
    }


def build_backhalf_channel_features(
    context: BackHalfChannelContext,
    *,
    train_end_quarter: str,
    quarters: list[str],
) -> dict[str, Any]:
    if context.reengagement_sensitivity_mode not in REENGAGEMENT_SENSITIVITY_MODES:
        raise ValueError(
            "Unsupported reengagement_sensitivity_mode "
            f"{context.reengagement_sensitivity_mode!r}; expected one of {REENGAGEMENT_SENSITIVITY_MODES!r}."
        )
    _start_month, train_end_month = _quarter_month_range(train_end_quarter)
    rows = load_monthly_signal_rows(_shock_context(context))
    monthly_values = _monthly_metric_values(rows, train_end_month)
    feature_names = sorted({feature for features in BACKHALF_TRANSITION_FEATURES.values() for feature in features})
    raw, retention_evidence = _raw_backhalf_features_with_evidence(
        monthly_values,
        reengagement_sensitivity_mode=context.reengagement_sensitivity_mode,
    )
    table = _standardize_feature_table(raw, feature_names)
    ordered_train_quarters = [quarter for quarter in sorted(table, key=quarter_sort_key) if quarter_sort_key(quarter) <= quarter_sort_key(train_end_quarter)]
    phis = {
        feature: _positive_autocorrelation([float((table.get(quarter) or {}).get(feature) or 0.0) for quarter in ordered_train_quarters])
        for feature in feature_names
    }
    quarter_features = {
        quarter: _feature_for_quarter(
            table,
            quarter,
            train_end_quarter=train_end_quarter,
            feature_names=feature_names,
            phis=phis,
        )
        for quarter in quarters
    }
    return {
        "schema_version": BACKHALF_CHANNEL_SCHEMA_VERSION,
        "feature_names": feature_names,
        "transition_feature_map": {transition: list(features) for transition, features in BACKHALF_TRANSITION_FEATURES.items()},
        "quarter_features": quarter_features,
        "train_end_quarter": train_end_quarter,
        "train_end_month": _month_label(train_end_month),
        "reengagement_sensitivity_mode": context.reengagement_sensitivity_mode,
        "monthly_row_count": len(rows),
        "aligned_train_quarter_count": len(ordered_train_quarters),
        "feature_autocorrelation": phis,
        "art_retention_evidence": retention_evidence,
        "art_retention_evidence_summary": {
            "support_class_counts": {
                support_class: sum(
                    1
                    for row in retention_evidence.values()
                    if str(row.get("support_class") or "") == support_class
                )
                for support_class in (
                    "direct_process_observed",
                    "cohort_balance_proxy",
                    "art_stock_balance_proxy",
                    "not_observed",
                )
            },
            "direct_process_observed_quarter_count": sum(
                1
                for row in retention_evidence.values()
                if str(row.get("support_class") or "") == "direct_process_observed"
            ),
            "proxy_supported_quarter_count": sum(
                1
                for row in retention_evidence.values()
                if str(row.get("support_class") or "") in {"cohort_balance_proxy", "art_stock_balance_proxy"}
            ),
            "direct_reengagement_quarter_count": sum(
                1
                for row in retention_evidence.values()
                if bool((row.get("reengagement_evidence") or {}).get("direct_process_observed"))
            ),
            "proxy_reengagement_quarter_count": sum(
                1
                for row in retention_evidence.values()
                if str((row.get("reengagement_evidence") or {}).get("claim_status") or "")
                == "proxy_only_no_public_treatment_cohort"
            ),
            "zero_sensitivity_reengagement_quarter_count": sum(
                1
                for row in retention_evidence.values()
                if str((row.get("reengagement_evidence") or {}).get("claim_status") or "")
                == "zero_sensitivity_no_public_treatment_cohort"
            ),
            "upper_bound_reengagement_quarter_count": sum(
                1
                for row in retention_evidence.values()
                if str((row.get("reengagement_evidence") or {}).get("claim_status") or "")
                == "upper_bound_proxy_no_public_treatment_cohort"
            ),
        },
        "contract": (
            "Back-half monthly HARP/HASP support states are estimated only through the forecast-origin "
            "quarter and enter ART initiation, VL testing, suppression, ART interruption, and "
            "re-engagement hazards before state simulation. ART interruption channels prefer explicit "
            "process rows when present. Re-engagement is direct only when restart/return rows exist; "
            "otherwise the public-data path is a stock-flow residual proxy and cannot support a "
            "process-level re-engagement claim."
        ),
    }


def apply_backhalf_channel_adjustment_to_paths(
    *,
    context: BackHalfChannelContext,
    dataset: Any,
    hazard_paths: dict[str, Any],
    incidence_paths: dict[str, Any],
) -> dict[str, Any]:
    del incidence_paths
    train_rows = sorted(list(dataset.train_transition_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    holdout_rows = sorted(list(dataset.holdout_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    if not train_rows or not holdout_rows:
        return {"hazard_paths": hazard_paths, "diagnostics": {"status": "not_evaluable"}}
    train_quarters = [str(row.get("quarter") or "") for row in train_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in holdout_rows]
    features_payload = build_backhalf_channel_features(
        context,
        train_end_quarter=train_quarters[-1],
        quarters=train_quarters + holdout_quarters,
    )
    features = dict(features_payload.get("quarter_features") or {})
    transition_feature_map = {
        transition: list(feature_names)
        for transition, feature_names in dict(features_payload.get("transition_feature_map") or {}).items()
    }
    adjusted_hazards = {
        key: {transition: float(value) for transition, value in dict(values).items()}
        for key, values in dict(hazard_paths.get("holdout_hazard_map") or {}).items()
    }
    adjusted_train_hazards = {
        key: {transition: float(value) for transition, value in dict(values).items()}
        for key, values in dict(hazard_paths.get("train_hazard_map") or {}).items()
    }
    diagnostics: dict[str, Any] = {
        "backhalf_channel_features": features_payload,
        "targets": {},
        "contract": "Back-half transition channels alter hazards before quarterly state simulation and before endpoint readout.",
    }
    eps = float(getattr(dataset, "eps", np.finfo(np.float32).eps))
    for transition, feature_names in transition_feature_map.items():
        residuals = {}
        for row in train_rows:
            quarter = str(row.get("quarter") or "")
            observed = float((row.get("hazards") or {}).get(transition) or 0.0)
            base = float((adjusted_train_hazards.get(quarter) or {}).get(transition) or 0.0)
            residuals[quarter] = logit(observed, eps=eps) - logit(base, eps=eps)
        fit = _fit_linear_effect(train_quarters, features, residuals, feature_names)
        diagnostics["targets"][transition] = {
            **fit,
            "feature_names": feature_names,
            "channel_role": {
                "D_to_A": "ART initiation",
                "A_to_T": "VL testing among active ART without recent VL",
                "T_to_V": "suppression after VL-tested unsuppressed state",
                "A_to_L": "ART interruption before recent VL",
                "T_to_L": "ART interruption after VL-tested unsuppressed state",
                "V_to_L": "ART interruption after suppression",
                "L_to_R": "re-engagement from interrupted ART",
                "R_to_A": "recently re-engaged return to active ART without recent VL",
            }.get(transition, transition),
        }
        for target_map in (adjusted_train_hazards, adjusted_hazards):
            for quarter, values in target_map.items():
                base = float(values.get(transition) or 0.0)
                effect = _linear_effect(dict(features.get(quarter) or {}), fit, feature_names)
                values[transition] = float(inv_logit(logit(base, eps=eps) + effect))
    adjusted_paths = dict(hazard_paths)
    adjusted_paths["train_hazard_map"] = adjusted_train_hazards
    adjusted_paths["holdout_hazard_map"] = adjusted_hazards
    return {
        "hazard_paths": adjusted_paths,
        "diagnostics": diagnostics,
    }
