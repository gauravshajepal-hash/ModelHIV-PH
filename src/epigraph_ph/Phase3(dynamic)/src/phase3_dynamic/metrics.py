from __future__ import annotations

from typing import Iterable

import numpy as np

PRIMARY_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "new_diagnosed_cases_period",
)

SUPPORT_AWARE_BACK_HALF_METRICS: tuple[str, ...] = (
    "tested_for_viral_load",
    "virally_suppressed",
)

SUPPORT_AWARE_BACK_HALF_RATE_METRICS: tuple[str, ...] = (
    "vl_testing_given_art",
    "suppression_given_vl_tested",
)

RATE_METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "vl_testing_given_art": {
        "numerator": "tested_for_viral_load",
        "denominator": "alive_on_art",
        "meaning": "viral-load testing coverage among people alive on ART",
    },
    "suppression_given_vl_tested": {
        "numerator": "virally_suppressed",
        "denominator": "tested_for_viral_load",
        "meaning": "viral suppression among people with viral-load testing evidence",
    },
}

OBSERVED_SUPPORT_TIERS: tuple[str, ...] = (
    "exact_observed",
    "bridge_observed",
)

SUPPORTED_SCORING_ROLES: tuple[str, ...] = (
    "direct_target",
)


def quarter_sort_key(value: str) -> tuple[int, int]:
    year_text, quarter_text = str(value).split("-Q", 1)
    return int(year_text), int(quarter_text)


def quarter_year(value: str) -> int:
    return int(quarter_sort_key(value)[0])


def quarter_ordinal(value: str) -> int:
    year, quarter = quarter_sort_key(value)
    return year * 4 + quarter - 1


def logit(value: float, *, eps: float = 1e-5) -> float:
    clipped = float(np.clip(value, eps, 1.0 - eps))
    return float(np.log(clipped / max(1.0 - clipped, eps)))


def inv_logit(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def metric_scales(rows: Iterable[dict[str, float]], *, eps: float) -> dict[str, float]:
    scales: dict[str, float] = {}
    row_list = list(rows)
    for metric_name in PRIMARY_METRICS:
        observed = [abs(float(row.get(metric_name) or 0.0)) for row in row_list if row.get(metric_name) is not None]
        scales[metric_name] = max(observed) if observed else eps
    return scales


def _metric_provenance(row: dict[str, float], metric_name: str) -> dict[str, object]:
    return dict((row.get("metric_provenance") or {}).get(metric_name) or {})


def is_supported_scoring_target(
    row: dict[str, float],
    metric_name: str,
    *,
    allowed_roles: tuple[str, ...] = SUPPORTED_SCORING_ROLES,
    observed_tiers: tuple[str, ...] = OBSERVED_SUPPORT_TIERS,
) -> bool:
    if row.get(metric_name) is None:
        return False
    provenance = _metric_provenance(row, metric_name)
    if not provenance:
        return False
    role = str(provenance.get("observation_role") or "")
    tier = str(provenance.get("tier") or "")
    return role in set(allowed_roles) and tier in set(observed_tiers)


def support_aware_metric_scales(
    rows: Iterable[dict[str, float]],
    metric_names: tuple[str, ...],
    *,
    eps: float,
    allowed_roles: tuple[str, ...] = SUPPORTED_SCORING_ROLES,
    observed_tiers: tuple[str, ...] = OBSERVED_SUPPORT_TIERS,
) -> dict[str, float]:
    scales: dict[str, float] = {}
    row_list = list(rows)
    for metric_name in metric_names:
        observed = [
            abs(float(row.get(metric_name) or 0.0))
            for row in row_list
            if is_supported_scoring_target(
                row,
                metric_name,
                allowed_roles=allowed_roles,
                observed_tiers=observed_tiers,
            )
        ]
        scales[metric_name] = max(observed) if observed else eps
    return scales


def support_aware_normalized_mae(
    prediction_rows: list[dict[str, float]],
    target_rows: list[dict[str, float]],
    scales: dict[str, float],
    *,
    metric_names: tuple[str, ...],
    eps: float,
    allowed_roles: tuple[str, ...] = SUPPORTED_SCORING_ROLES,
    observed_tiers: tuple[str, ...] = OBSERVED_SUPPORT_TIERS,
) -> dict[str, object]:
    errors: list[float] = []
    metric_rows: list[dict[str, object]] = []
    for metric_name in metric_names:
        metric_errors: list[float] = []
        missing_prediction_count = 0
        missing_train_scale_count = 0
        supported_target_count = 0
        support_partition_counts: dict[str, int] = {}
        tier_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}
        for prediction, target in zip(prediction_rows, target_rows):
            if not is_supported_scoring_target(
                target,
                metric_name,
                allowed_roles=allowed_roles,
                observed_tiers=observed_tiers,
            ):
                continue
            supported_target_count += 1
            provenance = _metric_provenance(target, metric_name)
            partition = str(provenance.get("support_partition") or "unknown")
            tier = str(provenance.get("tier") or "unknown")
            role = str(provenance.get("observation_role") or "unknown")
            support_partition_counts[partition] = support_partition_counts.get(partition, 0) + 1
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            role_counts[role] = role_counts.get(role, 0) + 1
            prediction_value = prediction.get(metric_name)
            if prediction_value is None:
                missing_prediction_count += 1
                continue
            scale = float(scales.get(metric_name) or eps)
            if scale <= eps:
                missing_train_scale_count += 1
                continue
            target_value = float(target.get(metric_name) or 0.0)
            metric_errors.append(abs(float(prediction_value) - target_value) / max(scale, eps))
        errors.extend(metric_errors)
        if missing_prediction_count or missing_train_scale_count:
            metric_status = "not_evaluable"
            metric_mae = float("inf")
        elif metric_errors:
            metric_status = "scored"
            metric_mae = float(np.mean(metric_errors))
        elif supported_target_count:
            metric_status = "not_train_scalable"
            metric_mae = float("inf")
        else:
            metric_status = "not_observed"
            metric_mae = float("inf")
        metric_rows.append(
            {
                "metric_name": metric_name,
                "status": metric_status,
                "supported_target_count": int(supported_target_count),
                "scored_entry_count": int(len(metric_errors)),
                "missing_prediction_count": int(missing_prediction_count),
                "missing_train_scale_count": int(missing_train_scale_count),
                "normalized_mae": metric_mae,
                "support_partition_counts": support_partition_counts,
                "tier_counts": tier_counts,
                "role_counts": role_counts,
            }
        )
    blocker_count = int(
        sum(int(row["missing_prediction_count"]) + int(row["missing_train_scale_count"]) for row in metric_rows)
    )
    scored_count = int(sum(int(row["scored_entry_count"]) for row in metric_rows))
    supported_count = int(sum(int(row["supported_target_count"]) for row in metric_rows))
    if blocker_count:
        status = "not_evaluable"
        mean_mae = float("inf")
    elif scored_count:
        status = "scored"
        mean_mae = float(np.mean(errors))
    elif supported_count:
        status = "not_train_scalable"
        mean_mae = float("inf")
    else:
        status = "not_observed"
        mean_mae = float("inf")
    return {
        "status": status,
        "mean_normalized_mae": mean_mae,
        "supported_target_count": supported_count,
        "scored_entry_count": scored_count,
        "metric_rows": metric_rows,
        "metric_names": list(metric_names),
        "target_support_contract": {
            "allowed_roles": list(allowed_roles),
            "observed_tiers": list(observed_tiers),
            "scale_source": "train_rows_only",
        },
    }


def _rate_definition(rate_name: str) -> dict[str, str]:
    if rate_name not in RATE_METRIC_DEFINITIONS:
        raise ValueError(f"Unsupported conditional rate metric: {rate_name}")
    return RATE_METRIC_DEFINITIONS[rate_name]


def _rate_support_value(
    row: dict[str, float],
    rate_name: str,
    *,
    eps: float,
    require_supported_target: bool,
    allowed_roles: tuple[str, ...],
    observed_tiers: tuple[str, ...],
) -> float | None:
    definition = _rate_definition(rate_name)
    numerator_name = definition["numerator"]
    denominator_name = definition["denominator"]
    numerator = row.get(numerator_name)
    denominator = row.get(denominator_name)
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value <= eps:
        return None
    if require_supported_target:
        if not is_supported_scoring_target(
            row,
            numerator_name,
            allowed_roles=allowed_roles,
            observed_tiers=observed_tiers,
        ):
            return None
        if not is_supported_scoring_target(
            row,
            denominator_name,
            allowed_roles=allowed_roles,
            observed_tiers=observed_tiers,
        ):
            return None
    return float(np.clip(float(numerator) / denominator_value, 0.0, 1.0))


def _rate_support_provenance(
    row: dict[str, float],
    rate_name: str,
) -> dict[str, object]:
    definition = _rate_definition(rate_name)
    numerator_provenance = _metric_provenance(row, definition["numerator"])
    denominator_provenance = _metric_provenance(row, definition["denominator"])
    return {
        "numerator": definition["numerator"],
        "denominator": definition["denominator"],
        "meaning": definition["meaning"],
        "numerator_tier": str(numerator_provenance.get("tier") or "unknown"),
        "denominator_tier": str(denominator_provenance.get("tier") or "unknown"),
        "numerator_role": str(numerator_provenance.get("observation_role") or "unknown"),
        "denominator_role": str(denominator_provenance.get("observation_role") or "unknown"),
        "support_partition": str(numerator_provenance.get("support_partition") or denominator_provenance.get("support_partition") or "unknown"),
    }


def support_aware_rate_mae(
    prediction_rows: list[dict[str, float]],
    target_rows: list[dict[str, float]],
    train_rows: list[dict[str, float]],
    *,
    rate_names: tuple[str, ...] = SUPPORT_AWARE_BACK_HALF_RATE_METRICS,
    eps: float,
    allowed_roles: tuple[str, ...] = SUPPORTED_SCORING_ROLES,
    observed_tiers: tuple[str, ...] = OBSERVED_SUPPORT_TIERS,
) -> dict[str, object]:
    errors: list[float] = []
    metric_rows: list[dict[str, object]] = []
    for rate_name in rate_names:
        train_supported_count = sum(
            _rate_support_value(
                row,
                rate_name,
                eps=eps,
                require_supported_target=True,
                allowed_roles=allowed_roles,
                observed_tiers=observed_tiers,
            )
            is not None
            for row in train_rows
        )
        rate_errors: list[float] = []
        missing_prediction_count = 0
        supported_target_count = 0
        support_partition_counts: dict[str, int] = {}
        numerator_tier_counts: dict[str, int] = {}
        denominator_tier_counts: dict[str, int] = {}
        for prediction, target in zip(prediction_rows, target_rows):
            target_rate = _rate_support_value(
                target,
                rate_name,
                eps=eps,
                require_supported_target=True,
                allowed_roles=allowed_roles,
                observed_tiers=observed_tiers,
            )
            if target_rate is None:
                continue
            supported_target_count += 1
            provenance = _rate_support_provenance(target, rate_name)
            partition = str(provenance.get("support_partition") or "unknown")
            numerator_tier = str(provenance.get("numerator_tier") or "unknown")
            denominator_tier = str(provenance.get("denominator_tier") or "unknown")
            support_partition_counts[partition] = support_partition_counts.get(partition, 0) + 1
            numerator_tier_counts[numerator_tier] = numerator_tier_counts.get(numerator_tier, 0) + 1
            denominator_tier_counts[denominator_tier] = denominator_tier_counts.get(denominator_tier, 0) + 1
            prediction_rate = _rate_support_value(
                prediction,
                rate_name,
                eps=eps,
                require_supported_target=False,
                allowed_roles=allowed_roles,
                observed_tiers=observed_tiers,
            )
            if prediction_rate is None:
                missing_prediction_count += 1
                continue
            if train_supported_count <= 0:
                continue
            rate_errors.append(abs(float(prediction_rate) - float(target_rate)))
        errors.extend(rate_errors)
        if missing_prediction_count:
            metric_status = "not_evaluable"
            metric_mae = float("inf")
        elif supported_target_count and train_supported_count <= 0:
            metric_status = "not_train_supported"
            metric_mae = float("inf")
        elif rate_errors:
            metric_status = "scored"
            metric_mae = float(np.mean(rate_errors))
        elif supported_target_count:
            metric_status = "not_evaluable"
            metric_mae = float("inf")
        else:
            metric_status = "not_observed"
            metric_mae = float("inf")
        metric_rows.append(
            {
                "metric_name": rate_name,
                "status": metric_status,
                "train_supported_count": int(train_supported_count),
                "supported_target_count": int(supported_target_count),
                "scored_entry_count": int(len(rate_errors)),
                "missing_prediction_count": int(missing_prediction_count),
                "normalized_mae": metric_mae,
                "support_partition_counts": support_partition_counts,
                "numerator_tier_counts": numerator_tier_counts,
                "denominator_tier_counts": denominator_tier_counts,
                "definition": dict(_rate_definition(rate_name)),
            }
        )
    missing_prediction_total = int(sum(int(row["missing_prediction_count"]) for row in metric_rows))
    train_unsupported_total = int(
        sum(
            int(row["supported_target_count"])
            for row in metric_rows
            if int(row["train_supported_count"]) <= 0
        )
    )
    scored_count = int(sum(int(row["scored_entry_count"]) for row in metric_rows))
    supported_count = int(sum(int(row["supported_target_count"]) for row in metric_rows))
    if missing_prediction_total:
        status = "not_evaluable"
        mean_mae = float("inf")
    elif scored_count:
        status = "scored"
        mean_mae = float(np.mean(errors))
    elif train_unsupported_total:
        status = "not_train_supported"
        mean_mae = float("inf")
    elif supported_count:
        status = "not_evaluable"
        mean_mae = float("inf")
    else:
        status = "not_observed"
        mean_mae = float("inf")
    return {
        "status": status,
        "mean_normalized_mae": mean_mae,
        "supported_target_count": supported_count,
        "scored_entry_count": scored_count,
        "metric_rows": metric_rows,
        "metric_names": list(rate_names),
        "target_support_contract": {
            "allowed_roles": list(allowed_roles),
            "observed_tiers": list(observed_tiers),
            "scale_source": "unit_interval_rate_no_holdout_scale",
            "train_support_required": True,
        },
    }


def normalized_mae(
    prediction_rows: list[dict[str, float]],
    target_rows: list[dict[str, float]],
    scales: dict[str, float],
    *,
    eps: float,
) -> float:
    errors: list[float] = []
    for prediction, target in zip(prediction_rows, target_rows):
        for metric_name in PRIMARY_METRICS:
            prediction_value = prediction.get(metric_name)
            target_value = target.get(metric_name)
            if prediction_value is None or target_value is None:
                continue
            errors.append(abs(float(prediction_value) - float(target_value)) / max(float(scales.get(metric_name) or eps), eps))
    return float(np.mean(errors)) if errors else float("inf")


def smape(prediction_rows: list[dict[str, float]], target_rows: list[dict[str, float]], *, eps: float) -> float:
    scores: list[float] = []
    for prediction, target in zip(prediction_rows, target_rows):
        for metric_name in PRIMARY_METRICS:
            prediction_value = prediction.get(metric_name)
            target_value = target.get(metric_name)
            if prediction_value is None or target_value is None:
                continue
            denom = abs(float(prediction_value)) + abs(float(target_value))
            if denom <= eps:
                continue
            scores.append((2.0 * abs(float(prediction_value) - float(target_value))) / denom)
    return float(np.mean(scores)) if scores else 0.0
