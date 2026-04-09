from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.geography import infer_region_code, normalize_geo_label
from epigraph_ph.phase15.latent_measurements import build_sparse_indicator_cube


def _phase15_factor_graph_cfg(plugin_id: str) -> dict[str, Any]:
    plugin = get_disease_plugin(plugin_id)
    phase15_cfg = dict((plugin.constraint_settings or {}).get("phase15", {}) or {})
    latent_cfg = dict(phase15_cfg.get("latent_blocks", {}) or {})
    graph_cfg = dict(latent_cfg.get("province_factor_graph", {}) or {})
    return {
        "enabled": bool(graph_cfg.get("enabled", True)),
        "national_precision": float(graph_cfg.get("national_precision") or 4.0),
        "region_precision": float(graph_cfg.get("region_precision") or 2.0),
        "loading_prior_precision": float(graph_cfg.get("loading_prior_precision") or 8.0),
        "loading_scale_min": float(graph_cfg.get("loading_scale_min") or 0.25),
        "loading_scale_max": float(graph_cfg.get("loading_scale_max") or 1.75),
        "observation_weight_floor": float(graph_cfg.get("observation_weight_floor") or 0.25),
        "posterior_precision_eps": float(graph_cfg.get("posterior_precision_eps") or 1e-6),
    }


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


def _normalize_region_label(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return "unknown"
    if token.lower() in {"national", "philippines"}:
        return "national"
    inferred = infer_region_code(token)
    if inferred:
        return inferred
    return re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_") or "unknown"


def _province_lookup_tokens(province_axis: list[str]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, province in enumerate(province_axis):
        normalized = normalize_geo_label(str(province), default_country_focus=True)
        lookup[normalized.lower()] = idx
        lookup[str(province).strip().lower()] = idx
    return lookup


def _province_region_members(region_labels: list[str]) -> dict[str, list[int]]:
    members: dict[str, list[int]] = defaultdict(list)
    for province_idx, label in enumerate(region_labels):
        members[_normalize_region_label(label)].append(province_idx)
    return dict(members)


def _month_slots_for_row(row: Mapping[str, Any], month_axis: list[str]) -> list[int]:
    time_value = str(row.get("time") or row.get("effective_month") or "").strip()
    if not time_value:
        return []
    if re.fullmatch(r"\d{4}-\d{2}", time_value):
        try:
            return [month_axis.index(time_value)]
        except ValueError:
            return []
    if re.fullmatch(r"\d{4}", time_value):
        return [idx for idx, month in enumerate(month_axis) if str(month).startswith(f"{time_value}-")]
    return []


def _row_weight(row: Mapping[str, Any], *, floor: float) -> float:
    learned_weight = _safe_float(row.get("observation_weight"), default=float("nan"))
    evidence_weight = _safe_float(row.get("evidence_weight"), default=1.0)
    quality_weight = _safe_float(row.get("quality_weight"), default=1.0)
    if quality_weight <= 0.0:
        quality_weight = 1.0
    bias_penalty = _safe_float(row.get("bias_penalty"), default=0.0)
    reliability = max(0.25, 1.0 - max(0.0, bias_penalty))
    base_weight = learned_weight if np.isfinite(learned_weight) and learned_weight > 0.0 else evidence_weight * reliability
    return max(float(floor), base_weight * quality_weight)


def _target_province_indices(
    row: Mapping[str, Any],
    *,
    province_lookup: Mapping[str, int],
    region_members: Mapping[str, list[int]],
) -> tuple[list[int], str]:
    geo_resolution = str(row.get("geo_resolution") or "").strip().lower()
    province_value = str(row.get("province") or row.get("geo") or "").strip()
    region_value = str(row.get("region") or "").strip()
    if geo_resolution in {"province", "city"} or province_value:
        normalized = normalize_geo_label(province_value, default_country_focus=True).lower()
        if normalized in province_lookup:
            return [int(province_lookup[normalized])], "province"
        fallback = province_value.lower()
        if fallback in province_lookup:
            return [int(province_lookup[fallback])], "province"
    if geo_resolution == "region" or region_value:
        token = _normalize_region_label(region_value or province_value)
        if token in region_members:
            return [int(idx) for idx in region_members[token]], "region"
    return [], "unsupported"


def _retained_block_loading_rows(national_scaffold: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in list(dict(national_scaffold.get("loadings") or {}).get("rows") or []):
        block_id = str(row.get("block_id") or "")
        if block_id:
            rows_by_block[block_id].append(dict(row))
    return dict(rows_by_block)


def _retained_block_states(national_scaffold: Mapping[str, Any]) -> dict[str, np.ndarray]:
    state_lookup: dict[str, np.ndarray] = {}
    for row in list(dict(national_scaffold.get("states") or {}).get("rows") or []):
        block_id = str(row.get("block_id") or "")
        values = np.asarray(list(row.get("state_values") or []), dtype=np.float32)
        if block_id and values.size:
            state_lookup[block_id] = values
    return state_lookup


def build_province_factor_graph_scaffold(
    *,
    standardized_tensor: np.ndarray,
    axis_catalogs: Mapping[str, list[str]],
    normalized_rows: list[dict[str, Any]],
    national_scaffold: Mapping[str, Any],
    region_labels: list[str],
    plugin_id: str,
) -> dict[str, Any]:
    cfg = _phase15_factor_graph_cfg(plugin_id)
    province_axis = list(axis_catalogs.get("province") or [])
    month_axis = list(axis_catalogs.get("month") or [])
    canonical_axis = list(axis_catalogs.get("canonical_name") or [])
    province_count = len(province_axis)
    month_count = len(month_axis)
    province_region_labels = [
        infer_region_code(str(province), str(province)) or _normalize_region_label(region_labels[idx] if idx < len(region_labels) else "")
        for idx, province in enumerate(province_axis)
    ]
    block_loading_rows = _retained_block_loading_rows(national_scaffold)
    national_state_lookup = _retained_block_states(national_scaffold)
    block_ids = [block_id for block_id in block_loading_rows if block_id in national_state_lookup]
    if not cfg["enabled"] or not block_ids:
        empty_tensor = np.zeros((province_count, month_count, 0), dtype=np.float32)
        return {
            "method": "province_factor_graph_scaffold_v1",
            "state_tensor": empty_tensor,
            "states": {"method": "province_factor_graph_scaffold_v1", "province_axis": province_axis, "month_axis": month_axis, "rows": []},
            "uncertainty": {"method": "province_factor_graph_scaffold_v1", "province_axis": province_axis, "month_axis": month_axis, "rows": []},
            "loading_deviations": {"method": "province_factor_graph_scaffold_v1", "rows": []},
            "identification_report": {
                "method": "province_factor_graph_scaffold_v1",
                "is_scaffold": True,
                "retained_block_count": 0,
                "block_ids": [],
                "notes": ["No retained national latent blocks were available for province factor-graph deployment."],
            },
        }

    canonical_index = {str(name): idx for idx, name in enumerate(canonical_axis)}
    province_lookup = _province_lookup_tokens(province_axis)
    region_members = _province_region_members(province_region_labels)
    region_axis = list(region_members.keys())
    region_index = {label: idx for idx, label in enumerate(region_axis)}
    retained_indicator_names = sorted(
        {
            str(row.get("canonical_name") or "")
            for block_id in block_ids
            for row in list(block_loading_rows.get(block_id) or [])
            if str(row.get("canonical_name") or "").strip()
        }
    )
    sparse_measurements = build_sparse_indicator_cube(
        normalized_rows=normalized_rows,
        province_axis=province_axis,
        month_axis=month_axis,
        canonical_names=retained_indicator_names,
        region_labels=province_region_labels,
        include_national_rows=True,
        observation_weight_floor=cfg["observation_weight_floor"],
    )
    sparse_standardized_cube = np.asarray(sparse_measurements["standardized_cube"], dtype=np.float32)
    sparse_canonical_index = {str(name): idx for idx, name in enumerate(list(sparse_measurements.get("canonical_axis") or []))}

    block_count = len(block_ids)
    state_tensor = np.zeros((province_count, month_count, block_count), dtype=np.float32)
    region_state_tensor = np.zeros((len(region_axis), month_count, block_count), dtype=np.float32)
    uncertainty_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    loading_deviation_rows: list[dict[str, Any]] = []

    province_support_weights: dict[tuple[str, str], np.ndarray] = {}
    region_support_weights: dict[tuple[str, str], np.ndarray] = {}
    block_display_names: dict[str, str] = {}
    retained_indicator_summary: dict[str, list[str]] = {}
    for block_id in block_ids:
        loading_rows = list(block_loading_rows.get(block_id) or [])
        retained_indicator_summary[block_id] = [str(row.get("canonical_name") or "") for row in loading_rows]
        display_name = str(loading_rows[0].get("display_name") or block_id.replace("_", " ").title())
        block_display_names[block_id] = display_name
        for loading_row in loading_rows:
            canonical_name = str(loading_row.get("canonical_name") or "")
            province_support_weights[(block_id, canonical_name)] = np.zeros((province_count, month_count), dtype=np.float32)
            region_support_weights[(block_id, canonical_name)] = np.zeros((province_count, month_count), dtype=np.float32)

    for row in normalized_rows:
        measurement_role = str(row.get("measurement_role") or "")
        if measurement_role == "context_only":
            continue
        canonical_name = str(row.get("canonical_name") or "")
        block_id = str(row.get("candidate_block") or "")
        if (block_id, canonical_name) not in province_support_weights:
            continue
        month_slots = _month_slots_for_row(row, month_axis)
        if not month_slots:
            continue
        province_indices, support_scope = _target_province_indices(
            row,
            province_lookup=province_lookup,
            region_members=region_members,
        )
        if not province_indices:
            continue
        base_weight = _row_weight(row, floor=cfg["observation_weight_floor"])
        cell_weight = float(base_weight) / float(max(len(province_indices) * len(month_slots), 1))
        if support_scope in {"province", "region"}:
            region_support = region_support_weights[(block_id, canonical_name)]
            for province_idx in province_indices:
                for month_idx in month_slots:
                    region_support[province_idx, month_idx] += cell_weight
        if support_scope == "province":
            province_support = province_support_weights[(block_id, canonical_name)]
            for province_idx in province_indices:
                for month_idx in month_slots:
                    province_support[province_idx, month_idx] += cell_weight

    for block_idx, block_id in enumerate(block_ids):
        z_national = np.asarray(national_state_lookup[block_id], dtype=np.float32)
        loading_rows = list(block_loading_rows.get(block_id) or [])
        local_precision = np.zeros((province_count, month_count), dtype=np.float32)
        local_numerator = np.zeros((province_count, month_count), dtype=np.float32)
        region_precision = np.zeros((province_count, month_count), dtype=np.float32)
        region_numerator = np.zeros((province_count, month_count), dtype=np.float32)
        for loading_row in loading_rows:
            canonical_name = str(loading_row.get("canonical_name") or "")
            canonical_idx = canonical_index.get(canonical_name)
            loading_value = _safe_float(loading_row.get("loading"), default=0.0)
            if canonical_name in sparse_canonical_index:
                indicator_surface = np.asarray(sparse_standardized_cube[:, :, sparse_canonical_index[canonical_name]], dtype=np.float32)
            elif canonical_idx is not None:
                indicator_surface = np.asarray(standardized_tensor[:, :, canonical_idx], dtype=np.float32)
            else:
                continue
            local_weight = province_support_weights[(block_id, canonical_name)]
            region_weight = region_support_weights[(block_id, canonical_name)]
            local_precision += local_weight * float(loading_value**2)
            local_numerator += local_weight * float(loading_value) * indicator_surface
            region_precision += region_weight * float(loading_value**2)
            region_numerator += region_weight * float(loading_value) * indicator_surface

        for region_label, members in region_members.items():
            members_array = np.asarray(members, dtype=np.int32)
            member_precision = np.sum(region_precision[members_array, :], axis=0, dtype=np.float32)
            member_numerator = np.sum(region_numerator[members_array, :], axis=0, dtype=np.float32)
            posterior_precision = cfg["national_precision"] + member_precision
            z_region = (cfg["national_precision"] * z_national + member_numerator) / np.clip(posterior_precision, cfg["posterior_precision_eps"], None)
            region_state_tensor[int(region_index[region_label]), :, block_idx] = z_region.astype(np.float32)
            for province_idx in members:
                province_posterior_precision = cfg["national_precision"] + cfg["region_precision"] + local_precision[province_idx, :]
                province_state = (
                    cfg["national_precision"] * z_national
                    + cfg["region_precision"] * z_region
                    + local_numerator[province_idx, :]
                ) / np.clip(province_posterior_precision, cfg["posterior_precision_eps"], None)
                state_tensor[province_idx, :, block_idx] = province_state.astype(np.float32)

        for province_idx, province in enumerate(province_axis):
            province_state = np.asarray(state_tensor[province_idx, :, block_idx], dtype=np.float32)
            posterior_precision = cfg["national_precision"] + cfg["region_precision"] + local_precision[province_idx, :]
            posterior_std = np.sqrt(1.0 / np.clip(posterior_precision, cfg["posterior_precision_eps"], None))
            local_indicator_count = np.zeros((month_count,), dtype=np.int32)
            local_support_mass = np.zeros((month_count,), dtype=np.float32)
            regional_indicator_count = np.zeros((month_count,), dtype=np.int32)
            regional_support_mass = np.zeros((month_count,), dtype=np.float32)
            for loading_row in loading_rows:
                canonical_name = str(loading_row.get("canonical_name") or "")
                local_support = province_support_weights[(block_id, canonical_name)][province_idx, :]
                regional_support = region_support_weights[(block_id, canonical_name)][province_idx, :]
                local_indicator_count += (local_support > 0.0).astype(np.int32)
                local_support_mass += local_support
                regional_indicator_count += (regional_support > 0.0).astype(np.int32)
                regional_support_mass += regional_support
            state_rows.append(
                {
                    "block_id": block_id,
                    "display_name": block_display_names[block_id],
                    "province": province,
                    "region": province_region_labels[province_idx] if province_idx < len(province_region_labels) else "unknown",
                    "state_values": [round(float(value), 6) for value in province_state.tolist()],
                }
            )
            uncertainty_rows.append(
                {
                    "block_id": block_id,
                    "display_name": block_display_names[block_id],
                    "province": province,
                    "region": province_region_labels[province_idx] if province_idx < len(province_region_labels) else "unknown",
                    "posterior_std_values": [round(float(value), 6) for value in posterior_std.tolist()],
                    "local_precision_values": [round(float(value), 6) for value in local_precision[province_idx, :].tolist()],
                    "region_precision_values": [round(float(value), 6) for value in region_precision[province_idx, :].tolist()],
                    "local_support_mass_values": [round(float(value), 6) for value in local_support_mass.tolist()],
                    "regional_support_mass_values": [round(float(value), 6) for value in regional_support_mass.tolist()],
                    "observed_indicator_count_values": [int(value) for value in local_indicator_count.tolist()],
                    "regional_indicator_count_values": [int(value) for value in regional_indicator_count.tolist()],
                }
            )

            for loading_row in loading_rows:
                canonical_name = str(loading_row.get("canonical_name") or "")
                canonical_idx = canonical_index.get(canonical_name)
                support = province_support_weights[(block_id, canonical_name)][province_idx, :]
                if canonical_name in sparse_canonical_index:
                    indicator_surface = np.asarray(sparse_standardized_cube[province_idx, :, sparse_canonical_index[canonical_name]], dtype=np.float32)
                elif canonical_idx is not None:
                    indicator_surface = np.asarray(standardized_tensor[province_idx, :, canonical_idx], dtype=np.float32)
                else:
                    continue
                national_loading = _safe_float(loading_row.get("loading"), default=0.0)
                prior_precision = cfg["loading_prior_precision"]
                target = float(national_loading) * province_state
                numerator = float(np.sum(support * indicator_surface * target, dtype=np.float32)) + prior_precision
                denominator = float(np.sum(support * np.square(target), dtype=np.float32)) + prior_precision
                scale_multiplier = numerator / max(denominator, cfg["posterior_precision_eps"])
                scale_multiplier = float(np.clip(scale_multiplier, cfg["loading_scale_min"], cfg["loading_scale_max"]))
                effective_loading = float(national_loading) * scale_multiplier
                loading_deviation_rows.append(
                    {
                        "block_id": block_id,
                        "display_name": block_display_names[block_id],
                        "province": province,
                        "region": province_region_labels[province_idx] if province_idx < len(province_region_labels) else "unknown",
                        "canonical_name": canonical_name,
                        "national_loading": round(float(national_loading), 6),
                        "effective_loading": round(float(effective_loading), 6),
                        "loading_deviation": round(float(effective_loading - national_loading), 6),
                        "scale_multiplier": round(float(scale_multiplier), 6),
                        "observed_month_count": int(np.sum(support > 0.0)),
                        "support_mass": round(float(np.sum(support, dtype=np.float32)), 6),
                        "correlation_with_block_state": round(float(_safe_corr(indicator_surface[support > 0.0], province_state[support > 0.0])), 6)
                        if np.any(support > 0.0)
                        else 0.0,
                    }
                )

    identification_report = {
        "method": "province_factor_graph_scaffold_v1",
        "is_scaffold": True,
        "plugin_id": plugin_id,
        "province_count": province_count,
        "region_count": len(region_axis),
        "retained_block_count": len(block_ids),
        "block_ids": list(block_ids),
        "block_indicator_summary": retained_indicator_summary,
        "notes": [
            "Province states are inferred with a closed-form shrinkage graph using national latent states as priors.",
            "Only province and region observations contribute local evidence; national observations are already absorbed by the national scaffold.",
            "Loading signs remain fixed globally while province-level loading magnitudes are shrunk toward the national loading.",
        ],
    }
    return {
        "method": "province_factor_graph_scaffold_v1",
        "state_tensor": state_tensor,
        "region_state_tensor": region_state_tensor,
        "states": {
            "method": "province_factor_graph_scaffold_v1",
            "province_axis": province_axis,
            "month_axis": month_axis,
            "block_axis": list(block_ids),
            "rows": state_rows,
        },
        "uncertainty": {
            "method": "province_factor_graph_scaffold_v1",
            "province_axis": province_axis,
            "month_axis": month_axis,
            "block_axis": list(block_ids),
            "rows": uncertainty_rows,
        },
        "loading_deviations": {
            "method": "province_factor_graph_scaffold_v1",
            "rows": loading_deviation_rows,
        },
        "identification_report": identification_report,
    }
