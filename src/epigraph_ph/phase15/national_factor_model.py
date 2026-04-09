from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from epigraph_ph.latent_blocks import latent_block_specs
from epigraph_ph.phase15.latent_measurements import build_sparse_indicator_cube


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1e-8 or right_std <= 1e-8:
        return 0.0
    corr = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _national_indicator_cube(standardized_tensor: np.ndarray, province_axis: list[str]) -> tuple[np.ndarray, str]:
    if standardized_tensor.ndim != 3:
        raise ValueError("standardized_tensor must have rank 3 [province, month, canonical_name]")
    if "Philippines" in province_axis:
        return np.asarray(standardized_tensor[province_axis.index("Philippines"), :, :], dtype=np.float32), "Philippines"
    return np.asarray(np.mean(standardized_tensor, axis=0), dtype=np.float32), "province_mean_fallback"


def build_national_measurement_spec(
    *,
    normalized_rows: list[dict[str, Any]],
    observability_audit: dict[str, Any],
    month_axis: list[str],
    plugin_id: str,
) -> dict[str, Any]:
    audit_rows = {str(row.get("canonical_name") or ""): dict(row) for row in list(observability_audit.get("rows") or [])}
    retained_blocks: list[dict[str, Any]] = []
    dropped_blocks: list[dict[str, Any]] = []
    for block in latent_block_specs(plugin_id):
        indicator_rows: list[dict[str, Any]] = []
        direct_indicator_count = 0
        for canonical_name, indicator in dict(block.get("indicators") or {}).items():
            audit_row = dict(audit_rows.get(canonical_name) or {})
            if not audit_row:
                continue
            direct_count = int(audit_row.get("direct_indicator_count") or 0)
            proxy_count = int(audit_row.get("proxy_indicator_count") or 0)
            context_count = int(audit_row.get("context_only_count") or 0)
            numeric_count = int(audit_row.get("numeric_row_count") or 0)
            eligible = bool(direct_count > 0 or (proxy_count > 0 and numeric_count > 0))
            if not eligible:
                continue
            direct_indicator_count += 1 if direct_count > 0 else 0
            loading_weight_hint = max(
                0.25,
                float(direct_count)
                + 0.5 * float(proxy_count)
                + 0.25 * float(int(audit_row.get("monthly_support_count") or 0) > 0)
                + 0.25 * float(int(audit_row.get("national_support_count") or 0) > 0),
            )
            indicator_rows.append(
                {
                    "canonical_name": canonical_name,
                    "expected_sign": str(indicator.get("expected_sign") or audit_row.get("expected_sign") or "neutral"),
                    "direct_indicator_count": direct_count,
                    "proxy_indicator_count": proxy_count,
                    "context_only_count": context_count,
                    "national_support_count": int(audit_row.get("national_support_count") or 0),
                    "regional_support_count": int(audit_row.get("regional_support_count") or 0),
                    "province_support_count": int(audit_row.get("province_support_count") or 0),
                    "monthly_support_count": int(audit_row.get("monthly_support_count") or 0),
                    "annual_support_count": int(audit_row.get("annual_support_count") or 0),
                    "aggregate_support_score": float(audit_row.get("aggregate_support_score") or 0.0),
                    "subnational_support_score": float(audit_row.get("subnational_support_score") or 0.0),
                    "eligible_for_national_likelihood": bool(audit_row.get("eligible_for_national_likelihood")),
                    "eligible_for_subnational_inference": bool(audit_row.get("eligible_for_subnational_inference")),
                    "literature_basis": list(indicator.get("literature_basis") or audit_row.get("literature_basis") or []),
                    "loading_weight_hint": round(float(loading_weight_hint), 6),
                }
            )
        block_summary = {
            "block_id": str(block["block_id"]),
            "display_name": str(block["display_name"]),
            "description": str(block["description"]),
            "minimum_direct_indicators": int(block["minimum_direct_indicators"]),
            "minimum_indicator_count": int(block["minimum_indicator_count"]),
            "literature_basis": list(block.get("literature_basis") or []),
            "indicator_rows": indicator_rows,
            "direct_indicator_count": int(direct_indicator_count),
            "eligible_indicator_count": len(indicator_rows),
        }
        if direct_indicator_count >= int(block["minimum_direct_indicators"]) and len(indicator_rows) >= int(block["minimum_indicator_count"]):
            retained_blocks.append(block_summary)
        else:
            dropped_blocks.append(block_summary)

    return {
        "method": "signed_weighted_national_scaffold_v1",
        "plugin_id": plugin_id,
        "source_row_count": len(normalized_rows),
        "month_axis": list(month_axis),
        "retained_block_count": len(retained_blocks),
        "dropped_block_count": len(dropped_blocks),
        "retained_blocks": retained_blocks,
        "dropped_blocks": dropped_blocks,
    }


def fit_national_factor_model_scaffold(
    *,
    standardized_tensor: np.ndarray,
    axis_catalogs: Mapping[str, list[str]],
    normalized_rows: list[dict[str, Any]],
    measurement_spec: dict[str, Any],
    plugin_id: str,
) -> dict[str, Any]:
    province_axis = list(axis_catalogs.get("province") or [])
    month_axis = list(axis_catalogs.get("month") or [])
    canonical_axis = list(axis_catalogs.get("canonical_name") or [])
    canonical_index = {str(name): idx for idx, name in enumerate(canonical_axis)}
    national_cube, national_geo_label = _national_indicator_cube(np.asarray(standardized_tensor, dtype=np.float32), province_axis)
    province_mean_cube = np.asarray(np.mean(np.asarray(standardized_tensor, dtype=np.float32), axis=0), dtype=np.float32)
    retained_indicator_names = sorted(
        {
            str(indicator.get("canonical_name") or "")
            for block in list(measurement_spec.get("retained_blocks") or [])
            for indicator in list(block.get("indicator_rows") or [])
            if str(indicator.get("canonical_name") or "").strip()
        }
    )
    sparse_measurements = build_sparse_indicator_cube(
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        canonical_names=retained_indicator_names,
        include_national_rows=True,
    )
    sparse_standardized_cube = np.asarray(sparse_measurements["standardized_cube"], dtype=np.float32)
    sparse_weight_cube = np.asarray(sparse_measurements["weight_cube"], dtype=np.float32)
    sparse_canonical_index = {str(name): idx for idx, name in enumerate(list(sparse_measurements.get("canonical_axis") or []))}
    used_sparse_observed_mean = False
    used_province_mean_fallback = False

    loading_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    ppc_rows: list[dict[str, Any]] = []
    retained_block_ids: list[str] = []
    for block in list(measurement_spec.get("retained_blocks") or []):
        weighted_series: list[np.ndarray] = []
        raw_weights: list[float] = []
        kept_indicators: list[dict[str, Any]] = []
        for indicator in list(block.get("indicator_rows") or []):
            canonical_name = str(indicator.get("canonical_name") or "")
            if canonical_name not in canonical_index:
                continue
            if canonical_name in sparse_canonical_index:
                sparse_idx = sparse_canonical_index[canonical_name]
                month_weights = np.sum(sparse_weight_cube[:, :, sparse_idx], axis=0, dtype=np.float32)
                if float(np.sum(month_weights, dtype=np.float32)) > 1e-8:
                    numerator = np.sum(
                        sparse_standardized_cube[:, :, sparse_idx] * sparse_weight_cube[:, :, sparse_idx],
                        axis=0,
                        dtype=np.float32,
                    )
                    series = np.zeros((len(month_axis),), dtype=np.float32)
                    observed_mask = month_weights > 0.0
                    series[observed_mask] = numerator[observed_mask] / np.clip(month_weights[observed_mask], 1e-8, None)
                    used_sparse_observed_mean = True
                else:
                    series = np.asarray(national_cube[:, canonical_index[canonical_name]], dtype=np.float32)
            else:
                series = np.asarray(national_cube[:, canonical_index[canonical_name]], dtype=np.float32)
            fallback_series = np.asarray(province_mean_cube[:, canonical_index[canonical_name]], dtype=np.float32)
            if float(np.sum(np.abs(series), dtype=np.float32)) <= 1e-8 and float(np.sum(np.abs(fallback_series), dtype=np.float32)) > 1e-8:
                series = fallback_series
                used_province_mean_fallback = True
            sign = str(indicator.get("expected_sign") or "neutral")
            sign_multiplier = -1.0 if sign == "negative" else 1.0
            weight = max(0.25, _safe_float(indicator.get("loading_weight_hint"), default=1.0))
            weighted_series.append(series * sign_multiplier)
            raw_weights.append(weight)
            kept_indicators.append(dict(indicator))
        if not weighted_series:
            continue
        weights = np.asarray(raw_weights, dtype=np.float32)
        weights = weights / max(float(np.sum(weights)), 1e-8)
        state = np.zeros((len(month_axis),), dtype=np.float32)
        for weight, series in zip(weights, weighted_series, strict=False):
            state += float(weight) * np.asarray(series, dtype=np.float32)
        state_mean = float(np.mean(state))
        state_std = float(np.std(state))
        if state_std > 1e-8:
            state = (state - state_mean) / state_std
        retained_block_ids.append(str(block.get("block_id") or ""))
        state_rows.append(
            {
                "block_id": str(block.get("block_id") or ""),
                "display_name": str(block.get("display_name") or ""),
                "state_values": [round(float(value), 6) for value in state.tolist()],
            }
        )
        for weight, indicator, series in zip(weights, kept_indicators, weighted_series, strict=False):
            canonical_name = str(indicator.get("canonical_name") or "")
            sign = str(indicator.get("expected_sign") or "neutral")
            signed_loading = float(weight) * (-1.0 if sign == "negative" else 1.0)
            loading_rows.append(
                {
                    "block_id": str(block.get("block_id") or ""),
                    "display_name": str(block.get("display_name") or ""),
                    "canonical_name": canonical_name,
                    "expected_sign": sign,
                    "loading": round(signed_loading, 6),
                    "direct_indicator_count": int(indicator.get("direct_indicator_count") or 0),
                    "proxy_indicator_count": int(indicator.get("proxy_indicator_count") or 0),
                    "literature_basis": list(indicator.get("literature_basis") or []),
                }
            )
            ppc_rows.append(
                {
                    "block_id": str(block.get("block_id") or ""),
                    "canonical_name": canonical_name,
                    "expected_sign": sign,
                    "correlation_with_state": round(float(_safe_corr(np.asarray(series, dtype=np.float32), state)), 6),
                    "mean_absolute_error": round(float(np.mean(np.abs(np.asarray(series, dtype=np.float32) - state))), 6),
                }
            )

    if used_sparse_observed_mean:
        national_geo_label = "sparse_observed_mean"
    elif used_province_mean_fallback:
        national_geo_label = "province_mean_fallback"

    identification_report = {
        "method": "signed_weighted_national_scaffold_v1",
        "plugin_id": plugin_id,
        "is_scaffold": True,
        "national_geo_label": national_geo_label,
        "month_axis": list(month_axis),
        "retained_block_count": len(state_rows),
        "requested_block_count": len(list(measurement_spec.get("retained_blocks") or [])),
        "retained_blocks": retained_block_ids,
        "dropped_blocks": [str(row.get("block_id") or "") for row in list(measurement_spec.get("dropped_blocks") or [])],
        "notes": [
            "This is a signed weighted national scaffold, not a full mixed-frequency Bayesian estimator.",
            "Indicator signs are fixed from latent-block priors and local literature rationale captured in plugin config.",
        ],
    }

    return {
        "measurement_spec": measurement_spec,
        "loadings": {
            "method": "signed_weighted_national_scaffold_v1",
            "rows": loading_rows,
        },
        "states": {
            "method": "signed_weighted_national_scaffold_v1",
            "month_axis": list(month_axis),
            "rows": state_rows,
        },
        "ppc": {
            "method": "signed_weighted_national_scaffold_v1",
            "rows": ppc_rows,
        },
        "identification_report": identification_report,
    }
