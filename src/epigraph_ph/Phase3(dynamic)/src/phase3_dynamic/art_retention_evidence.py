from __future__ import annotations

from typing import Any

import numpy as np


ART_RETENTION_EVIDENCE_SCHEMA_VERSION = "phase3_dynamic_art_retention_evidence.v1"

ART_INTERRUPTION_METRICS: tuple[str, ...] = (
    "art_interruption_period",
    "art_ltfu_period",
    "art_treatment_interruption_period",
)
ART_REENGAGEMENT_METRICS: tuple[str, ...] = (
    "art_reengagement_period",
    "art_restarted_period",
    "returned_to_care_period",
)
ART_RETENTION_METRICS: tuple[str, ...] = (
    "art_retained_period",
    "art_retention_period",
)
ART_INTERRUPTION_CUMULATIVE_METRICS: tuple[str, ...] = (
    "art_ltfu_cumulative",
    "art_interruption_cumulative",
)
ART_REENGAGEMENT_CUMULATIVE_METRICS: tuple[str, ...] = (
    "art_reengagement_cumulative",
    "art_restarted_cumulative",
)
ART_TRANSFER_OUT_CUMULATIVE_METRICS: tuple[str, ...] = ("art_transfer_out_overseas_cumulative",)
ART_STOPPED_CUMULATIVE_METRICS: tuple[str, ...] = ("art_stopped_refused_cumulative",)
ART_DEATH_CUMULATIVE_METRICS: tuple[str, ...] = ("art_deaths_cumulative",)
ART_EVER_ENROLLED_CUMULATIVE_METRICS: tuple[str, ...] = ("art_ever_enrolled_cumulative",)
ART_NO_LONGER_CUMULATIVE_METRICS: tuple[str, ...] = ("art_no_longer_on_treatment_cumulative",)
ART_RETENTION_PROCESS_METRICS: tuple[str, ...] = (
    *ART_INTERRUPTION_METRICS,
    *ART_REENGAGEMENT_METRICS,
    *ART_RETENTION_METRICS,
    *ART_INTERRUPTION_CUMULATIVE_METRICS,
    *ART_REENGAGEMENT_CUMULATIVE_METRICS,
    *ART_TRANSFER_OUT_CUMULATIVE_METRICS,
    *ART_STOPPED_CUMULATIVE_METRICS,
    *ART_DEATH_CUMULATIVE_METRICS,
    *ART_EVER_ENROLLED_CUMULATIVE_METRICS,
    *ART_NO_LONGER_CUMULATIVE_METRICS,
)
ART_RETENTION_CUMULATIVE_METRICS: tuple[str, ...] = (
    *ART_INTERRUPTION_CUMULATIVE_METRICS,
    *ART_REENGAGEMENT_CUMULATIVE_METRICS,
    *ART_TRANSFER_OUT_CUMULATIVE_METRICS,
    *ART_STOPPED_CUMULATIVE_METRICS,
    *ART_DEATH_CUMULATIVE_METRICS,
    *ART_EVER_ENROLLED_CUMULATIVE_METRICS,
    *ART_NO_LONGER_CUMULATIVE_METRICS,
)
REENGAGEMENT_PUBLIC_PROXY_METHOD = "public_art_stock_flow_residual"
REENGAGEMENT_PROXY_CLAIM_STATUS = "proxy_only_no_public_treatment_cohort"
REENGAGEMENT_SENSITIVITY_MODES: tuple[str, ...] = (
    "zero",
    "public_stock_flow_proxy",
    "upper_bound_proxy",
)


def safe_log_ratio(numerator: float, denominator: float) -> float:
    return float(np.log1p(max(float(numerator), 0.0)) - np.log1p(max(float(denominator), 0.0)))


def _sum_metrics(summary: dict[str, Any], metrics: tuple[str, ...]) -> float:
    return float(sum(float(summary.get(f"{metric}_sum") or 0.0) for metric in metrics))


def _count_metrics(summary: dict[str, Any], metrics: tuple[str, ...]) -> int:
    return int(sum(int(summary.get(f"{metric}_count") or 0) for metric in metrics))


def _last_metric(summary: dict[str, Any], metrics: tuple[str, ...]) -> float | None:
    for metric in metrics:
        if int(summary.get(f"{metric}_count") or 0) <= 0:
            continue
        value = summary.get(f"{metric}_last")
        if value is not None:
            return float(value)
    return None


def _cumulative_delta(
    *,
    summary: dict[str, Any],
    previous_summary: dict[str, Any],
    metrics: tuple[str, ...],
) -> tuple[float | None, int]:
    current = _last_metric(summary, metrics)
    previous = _last_metric(previous_summary, metrics)
    count = _count_metrics(summary, metrics)
    if current is None:
        return None, 0
    if previous is None:
        return None, count
    return max(float(current) - float(previous), 0.0), count


def build_art_retention_quarter_evidence(
    *,
    summary: dict[str, Any],
    previous_summary: dict[str, Any],
    diagnosed_gap: float,
    reengagement_sensitivity_mode: str = "public_stock_flow_proxy",
) -> dict[str, Any]:
    """Build ART retention/interruption/re-engagement evidence without endpoint tuning.

    Direct source rows win when available. Otherwise the function exposes deterministic
    stock-balance bounds from ART stock, new ART starts, and reported removals.
    """
    if reengagement_sensitivity_mode not in REENGAGEMENT_SENSITIVITY_MODES:
        raise ValueError(
            "Unsupported reengagement_sensitivity_mode "
            f"{reengagement_sensitivity_mode!r}; expected one of {REENGAGEMENT_SENSITIVITY_MODES!r}."
        )
    art = float(summary.get("art_last") or 0.0)
    previous_art = float(previous_summary.get("art_last") or 0.0)
    newly_enrolled = float(summary.get("newly_enrolled_sum") or 0.0)
    deaths = float(summary.get("deaths_sum") or 0.0)

    explicit_interruption = _sum_metrics(summary, ART_INTERRUPTION_METRICS)
    explicit_reengagement = _sum_metrics(summary, ART_REENGAGEMENT_METRICS)
    explicit_retention = _sum_metrics(summary, ART_RETENTION_METRICS)
    explicit_interruption_count = _count_metrics(summary, ART_INTERRUPTION_METRICS)
    explicit_reengagement_count = _count_metrics(summary, ART_REENGAGEMENT_METRICS)
    explicit_retention_count = _count_metrics(summary, ART_RETENTION_METRICS)
    cumulative_interruption_delta, cumulative_interruption_count = _cumulative_delta(
        summary=summary,
        previous_summary=previous_summary,
        metrics=ART_INTERRUPTION_CUMULATIVE_METRICS,
    )
    cumulative_reengagement_delta, cumulative_reengagement_count = _cumulative_delta(
        summary=summary,
        previous_summary=previous_summary,
        metrics=ART_REENGAGEMENT_CUMULATIVE_METRICS,
    )
    cumulative_transfer_delta, cumulative_transfer_count = _cumulative_delta(
        summary=summary,
        previous_summary=previous_summary,
        metrics=ART_TRANSFER_OUT_CUMULATIVE_METRICS,
    )
    cumulative_stopped_delta, cumulative_stopped_count = _cumulative_delta(
        summary=summary,
        previous_summary=previous_summary,
        metrics=ART_STOPPED_CUMULATIVE_METRICS,
    )
    cumulative_death_delta, cumulative_death_count = _cumulative_delta(
        summary=summary,
        previous_summary=previous_summary,
        metrics=ART_DEATH_CUMULATIVE_METRICS,
    )
    cumulative_ever_enrolled = _last_metric(summary, ART_EVER_ENROLLED_CUMULATIVE_METRICS)
    cumulative_no_longer = _last_metric(summary, ART_NO_LONGER_CUMULATIVE_METRICS)
    cumulative_interruption_stock = _last_metric(summary, ART_INTERRUPTION_CUMULATIVE_METRICS)

    starts_observed = int(summary.get("newly_enrolled_count") or 0) > 0
    has_art_stock_pair = previous_art > 0.0 and art > 0.0
    expected_after_new_starts_and_removal = max(previous_art + newly_enrolled - deaths, 0.0)
    if has_art_stock_pair and starts_observed:
        interruption_balance = max(expected_after_new_starts_and_removal - art, 0.0)
        reengagement_balance = max(art - expected_after_new_starts_and_removal, 0.0)
        retained_balance = max(min(previous_art, art - newly_enrolled - reengagement_balance), 0.0)
    elif has_art_stock_pair:
        interruption_balance = max(previous_art - deaths - art, 0.0)
        reengagement_balance = 0.0
        retained_balance = max(min(previous_art - deaths, art), 0.0)
    else:
        interruption_balance = 0.0
        reengagement_balance = 0.0
        retained_balance = 0.0
    if explicit_retention_count > 0:
        retained_balance = explicit_retention

    direct_interruption_available = explicit_interruption_count > 0 or cumulative_interruption_delta is not None
    direct_reengagement_available = explicit_reengagement_count > 0 or cumulative_reengagement_delta is not None

    interruption_value = (
        explicit_interruption
        if explicit_interruption_count > 0
        else float(cumulative_interruption_delta)
        if cumulative_interruption_delta is not None
        else interruption_balance
    )
    reengagement_value = (
        explicit_reengagement
        if explicit_reengagement_count > 0
        else float(cumulative_reengagement_delta)
        if cumulative_reengagement_delta is not None
        else reengagement_balance
    )
    transfer_out_value = float(cumulative_transfer_delta or 0.0)
    stopped_refused_value = float(cumulative_stopped_delta or 0.0)
    art_death_value = float(cumulative_death_delta or 0.0)
    process_removal_value = float(transfer_out_value + stopped_refused_value + art_death_value)
    observed_direct_outflow_value = (
        float(interruption_value if direct_interruption_available else 0.0)
        + float(process_removal_value)
    )
    stock_flow_expected_after_observed_process = max(previous_art + newly_enrolled - observed_direct_outflow_value, 0.0)
    public_stock_flow_residual = None
    latent_reengagement_proxy = 0.0
    latent_unobserved_attrition = 0.0
    latent_reengagement_upper_bound = 0.0
    if has_art_stock_pair and starts_observed and direct_interruption_available:
        public_stock_flow_residual = float(art - stock_flow_expected_after_observed_process)
        latent_reengagement_proxy = max(public_stock_flow_residual, 0.0)
        latent_unobserved_attrition = max(-public_stock_flow_residual, 0.0)
        interrupted_pool = max(
            float(cumulative_no_longer or 0.0),
            float(cumulative_interruption_stock or 0.0),
            float(interruption_value),
            0.0,
        )
        upper_denominator = max(float(diagnosed_gap), float(art), float(previous_art), interrupted_pool, 0.0)
        latent_reengagement_upper_bound = max(
            latent_reengagement_proxy,
            min(interrupted_pool, upper_denominator),
        )

    direct_components = (
        int(explicit_interruption_count > 0)
        + int(explicit_reengagement_count > 0)
        + int(explicit_retention_count > 0)
        + int(cumulative_interruption_delta is not None)
        + int(cumulative_reengagement_delta is not None)
        + int(cumulative_transfer_delta is not None)
        + int(cumulative_stopped_delta is not None)
        + int(cumulative_death_delta is not None)
    )
    balance_components = int(previous_art > 0.0) + int(art > 0.0) + int(starts_observed)
    if direct_components > 0:
        support_class = "direct_process_observed"
    elif previous_art > 0.0 and art > 0.0 and starts_observed:
        support_class = "cohort_balance_proxy"
    elif previous_art > 0.0 and art > 0.0:
        support_class = "art_stock_balance_proxy"
    else:
        support_class = "not_observed"

    retention_denominator = max(expected_after_new_starts_and_removal, previous_art, 0.0)
    retention_ratio = None
    if retention_denominator > 0.0:
        retention_ratio = float(np.clip(art / retention_denominator, 0.0, 1.0))

    support_weight = 0.0
    if support_class == "direct_process_observed":
        support_weight = float(direct_components) / 3.0
    elif support_class in {"cohort_balance_proxy", "art_stock_balance_proxy"}:
        support_weight = float(balance_components) / 3.0

    if direct_reengagement_available:
        reengagement_claim_status = "direct_process_observed"
        reengagement_proxy_value = 0.0
    elif public_stock_flow_residual is not None:
        if reengagement_sensitivity_mode == "zero":
            reengagement_claim_status = "zero_sensitivity_no_public_treatment_cohort"
            reengagement_proxy_value = 0.0
        elif reengagement_sensitivity_mode == "upper_bound_proxy":
            reengagement_claim_status = "upper_bound_proxy_no_public_treatment_cohort"
            reengagement_proxy_value = latent_reengagement_upper_bound
        else:
            reengagement_claim_status = REENGAGEMENT_PROXY_CLAIM_STATUS
            reengagement_proxy_value = latent_reengagement_proxy
        if reengagement_value <= 0.0:
            reengagement_value = latent_reengagement_proxy
    else:
        reengagement_claim_status = "not_identifiable_from_public_aggregate_data"
        reengagement_proxy_value = 0.0
    if not direct_reengagement_available:
        reengagement_value = float(reengagement_proxy_value)
    publishable_reengagement_claim = bool(direct_reengagement_available)

    return {
        "schema_version": ART_RETENTION_EVIDENCE_SCHEMA_VERSION,
        "support_class": support_class,
        "support_weight": float(support_weight),
        "previous_art": float(previous_art),
        "current_art": float(art),
        "newly_enrolled_sum": float(newly_enrolled),
        "deaths_sum": float(deaths),
        "expected_after_new_starts_and_removal": float(expected_after_new_starts_and_removal),
        "stock_flow_expected_after_observed_process": float(stock_flow_expected_after_observed_process),
        "public_stock_flow_residual": public_stock_flow_residual,
        "reengagement_sensitivity_mode": reengagement_sensitivity_mode,
        "interruption_count": float(interruption_value),
        "reengagement_count": float(reengagement_value),
        "direct_reengagement_count": float(
            explicit_reengagement
            if explicit_reengagement_count > 0
            else float(cumulative_reengagement_delta)
            if cumulative_reengagement_delta is not None
            else 0.0
        ),
        "latent_reengagement_proxy_count": float(reengagement_proxy_value),
        "latent_reengagement_public_point_count": float(latent_reengagement_proxy),
        "latent_reengagement_upper_bound_count": float(latent_reengagement_upper_bound),
        "latent_unobserved_attrition_count": float(latent_unobserved_attrition),
        "transfer_out_count": float(transfer_out_value),
        "stopped_refused_count": float(stopped_refused_value),
        "art_death_count": float(art_death_value),
        "process_removal_count": float(process_removal_value),
        "retained_count": float(retained_balance),
        "retention_ratio": retention_ratio,
        "cumulative_ever_enrolled": cumulative_ever_enrolled,
        "cumulative_no_longer_on_treatment": cumulative_no_longer,
        "direct_metric_counts": {
            "interruption": int(explicit_interruption_count),
            "reengagement": int(explicit_reengagement_count),
            "retention": int(explicit_retention_count),
            "cumulative_interruption": int(cumulative_interruption_count),
            "cumulative_reengagement": int(cumulative_reengagement_count),
            "cumulative_transfer_out": int(cumulative_transfer_count),
            "cumulative_stopped_refused": int(cumulative_stopped_count),
            "cumulative_art_deaths": int(cumulative_death_count),
        },
        "balance_proxy_counts": {
            "interruption": float(interruption_balance),
            "reengagement": float(reengagement_balance),
            "retained": float(retained_balance),
        },
        "reengagement_evidence": {
            "claim_status": reengagement_claim_status,
            "publishable_process_claim": publishable_reengagement_claim,
            "direct_process_observed": bool(direct_reengagement_available),
            "proxy_method": REENGAGEMENT_PUBLIC_PROXY_METHOD if reengagement_claim_status == REENGAGEMENT_PROXY_CLAIM_STATUS else None,
            "proxy_point_estimate": float(reengagement_proxy_value),
            "proxy_identification_lower_bound": 0.0,
            "proxy_identification_upper_bound": float(latent_reengagement_upper_bound),
            "limitation": (
                "Public HARP/HASP aggregate outcome rows do not identify restart/return-to-care timing. "
                "The proxy is a stock-flow residual after observed ART starts and observed LTFU/removal deltas; "
                "it is a bounded driver, not direct re-engagement evidence."
            ),
        },
        "pressure_features": {
            "art_ltfu_pressure": safe_log_ratio(interruption_value, max(previous_art, 0.0)),
            "art_interruption_evidence_pressure": safe_log_ratio(interruption_value, max(previous_art, 0.0)),
            "reengagement_pressure": safe_log_ratio(reengagement_value, max(diagnosed_gap, 0.0)),
            "art_reengagement_evidence_pressure": safe_log_ratio(
                explicit_reengagement
                if explicit_reengagement_count > 0
                else float(cumulative_reengagement_delta)
                if cumulative_reengagement_delta is not None
                else 0.0,
                max(diagnosed_gap, 0.0),
            ),
            "latent_reengagement_balance_pressure": safe_log_ratio(reengagement_proxy_value, max(diagnosed_gap, 0.0)),
            "latent_unobserved_attrition_pressure": safe_log_ratio(latent_unobserved_attrition, max(previous_art, 0.0)),
            "art_retention_gap_pressure": safe_log_ratio(max(1.0 - float(retention_ratio or 0.0), 0.0), 1.0),
            "art_retention_support": float(support_weight),
            "art_transfer_out_pressure": safe_log_ratio(transfer_out_value, max(previous_art, 0.0)),
            "art_stopped_pressure": safe_log_ratio(stopped_refused_value, max(previous_art, 0.0)),
            "art_death_pressure": safe_log_ratio(art_death_value, max(previous_art, 0.0)),
            "art_process_removal_pressure": safe_log_ratio(process_removal_value, max(previous_art, 0.0)),
        },
    }
