from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin


ARCHETYPE_ORDER = [
    "urban_high_throughput",
    "migrant_corridor",
    "remote_island",
    "fragile_service_network",
    "under_reporting_province",
]

TRANSITION_ORDER = ["U_to_D", "D_to_A", "A_to_V", "A_to_L", "L_to_A"]


@dataclass(slots=True)
class ProvinceArchetypeDefinition:
    archetype_id: str
    display_name: str
    description: str
    prototype_features: dict[str, float]
    reporting_noise: float
    diagnosis_prior_shift: float
    linkage_prior_shift: float
    retention_prior_shift: float
    vl_testing_prior_shift: float
    documentation_prior_shift: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hiv_archetype_cfg() -> dict[str, Any]:
    try:
        plugin = get_disease_plugin("hiv")
    except KeyError:  # pragma: no cover - import side effect fallback
        from epigraph_ph.plugins import hiv as _hiv  # noqa: F401

        plugin = get_disease_plugin("hiv")
    return dict((plugin.prior_hyperparameters or {}).get("province_archetypes", {}) or {})


def _archetype_library() -> dict[str, ProvinceArchetypeDefinition]:
    definitions = dict(_hiv_archetype_cfg().get("definitions", {}) or {})
    library: dict[str, ProvinceArchetypeDefinition] = {}
    for archetype_id in ARCHETYPE_ORDER:
        row = dict(definitions.get(archetype_id, {}) or {})
        library[archetype_id] = ProvinceArchetypeDefinition(
            archetype_id=archetype_id,
            display_name=str(row.get("display_name") or archetype_id.replace("_", " ").title()),
            description=str(row.get("description") or ""),
            prototype_features={str(key): float(value) for key, value in dict(row.get("prototype_features", {}) or {}).items()},
            reporting_noise=float(row.get("reporting_noise") or 0.0),
            diagnosis_prior_shift=float(row.get("diagnosis_prior_shift") or 0.0),
            linkage_prior_shift=float(row.get("linkage_prior_shift") or 0.0),
            retention_prior_shift=float(row.get("retention_prior_shift") or 0.0),
            vl_testing_prior_shift=float(row.get("vl_testing_prior_shift") or 0.0),
            documentation_prior_shift=float(row.get("documentation_prior_shift") or 0.0),
            notes=list(row.get("notes") or []),
        )
    return library


def _safe_ratio(numerator: float, denominator: float) -> float:
    cfg = _hiv_archetype_cfg()
    min_denominator = float(cfg.get("min_denominator", 1e-6))
    return float(numerator / denominator) if denominator > min_denominator else 0.0


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _softmax(values: np.ndarray, temperature: float | None = None) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    cfg = _hiv_archetype_cfg()
    used_temperature = float(cfg.get("softmax_temperature", 0.35)) if temperature is None else float(temperature)
    scaled = values / max(used_temperature, float(cfg.get("softmax_min_temperature", 1e-3)))
    shifted = scaled - float(np.max(scaled))
    weights = np.exp(shifted)
    weights = weights / np.clip(weights.sum(), float(cfg.get("min_denominator", 1e-6)), None)
    return weights.astype(np.float32)


def _normalize_feature(values: list[float]) -> list[float]:
    if not values:
        return []
    cfg = _hiv_archetype_cfg()
    lo = min(values)
    hi = max(values)
    if hi - lo < float(cfg.get("min_denominator", 1e-6)):
        return [float(cfg.get("zero_variance_midpoint", 0.5)) for _ in values]
    return [float(np.clip((value - lo) / (hi - lo), 0.0, 1.0)) for value in values]


def _province_feature_table(
    *,
    province_axis: list[str],
    month_axis: list[str],
    subgroup_summary: dict[str, Any],
    observation_targets: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    cfg = _hiv_archetype_cfg()
    cue_cfg = dict(cfg.get("under_reporting_cue_mix", {}) or {})
    default_feature = float(cfg.get("default_feature_value", 0.5))
    by_province = {str(row.get("province") or ""): row for row in subgroup_summary.get("rows", [])}
    diagnosed_mean = observation_targets.get("diagnosed_stock", np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)).mean(axis=1)
    art_mean = observation_targets.get("art_stock", np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)).mean(axis=1)
    sup_mean = observation_targets.get("documented_suppression", np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)).mean(axis=1)
    test_mean = observation_targets.get("testing_coverage", np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)).mean(axis=1)
    evidence_values = []
    raw_rows: list[dict[str, float]] = []
    for idx, province in enumerate(province_axis):
        row = by_province.get(province, {})
        network = row.get("network_signal") or {}
        kp_distribution = row.get("kp_distribution") or {}
        evidence_strength = float(row.get("evidence_strength") or 0.0)
        raw_rows.append(
            {
                "urbanity": float(network.get("urbanity") or default_feature),
                "accessibility": float(network.get("accessibility") or default_feature),
                "awareness": float(network.get("awareness") or default_feature),
                "stress": float(network.get("stress") or default_feature),
                "kp_focus": float(kp_distribution.get("msm") or 0.0) + 0.5 * float(kp_distribution.get("tgw") or 0.0),
                "evidence_strength": evidence_strength,
                "diagnosed_share": float(diagnosed_mean[idx]),
                "art_share": float(art_mean[idx]),
                "suppression_share": float(sup_mean[idx]),
                "testing_share": float(test_mean[idx]),
            }
        )
        evidence_values.append(evidence_strength)
    normalized_evidence = _normalize_feature(evidence_values)
    out: dict[str, dict[str, float]] = {}
    for idx, province in enumerate(province_axis):
        row = raw_rows[idx]
        out[province] = {
            **row,
            "evidence_strength": normalized_evidence[idx] if normalized_evidence else default_feature,
        }
        # Under-reporting cue: relatively decent access with weak observed diagnosis/testing.
        access = out[province]["accessibility"]
        diag = out[province]["diagnosed_share"]
        test = out[province]["testing_share"]
        evidence = out[province]["evidence_strength"]
        out[province]["under_reporting_cue"] = float(
            np.clip(
                float(cue_cfg.get("accessibility", 0.55)) * access
                + float(cue_cfg.get("urbanity", 0.20)) * out[province]["urbanity"]
                + float(cue_cfg.get("diagnosed_share", -0.45)) * diag
                + float(cue_cfg.get("testing_share", -0.30)) * test
                + float(cue_cfg.get("evidence_gap", 0.25)) * (1.0 - evidence),
                0.0,
                1.0,
            )
        )
    return out


def build_synthetic_province_library(
    *,
    month_axis: list[str],
) -> dict[str, Any]:
    cfg = _hiv_archetype_cfg()
    library = _archetype_library()
    generator_cfg = dict(cfg.get("synthetic_generator", {}) or {})
    min_denominator = float(cfg.get("min_denominator", 1e-6))
    month_count = max(1, len(month_axis))
    month_grid = np.linspace(0.0, 1.0, month_count, dtype=np.float32)
    archetype_rows = []
    for archetype_id in ARCHETYPE_ORDER:
        definition = library[archetype_id]
        progress = 1.0 / (
            1.0
            + np.exp(
                -float(generator_cfg.get("logistic_scale", 6.0))
                * (month_grid - float(generator_cfg.get("logistic_center", 0.45)))
            )
        )
        diagnosed = np.clip(
            definition.prototype_features["diagnosed_share"]
            * (float(generator_cfg.get("diagnosed_base", 0.72)) + float(generator_cfg.get("diagnosed_scale", 0.28)) * progress),
            float(generator_cfg.get("diagnosed_floor", 0.02)),
            float(generator_cfg.get("diagnosed_ceiling", 0.97)),
        )
        art = np.clip(
            definition.prototype_features["art_share"]
            * (float(generator_cfg.get("art_base", 0.65)) + float(generator_cfg.get("art_scale", 0.35)) * progress),
            float(generator_cfg.get("art_floor", 0.01)),
            diagnosed - float(generator_cfg.get("art_floor", 0.01)),
        )
        suppression = np.clip(
            definition.prototype_features["suppression_share"]
            * (float(generator_cfg.get("suppression_base", 0.55)) + float(generator_cfg.get("suppression_scale", 0.45)) * progress),
            float(generator_cfg.get("suppression_floor", 0.005)),
            art - float(generator_cfg.get("suppression_floor", 0.005)),
        )
        testing = np.clip(
            definition.prototype_features["testing_share"]
            * (float(generator_cfg.get("testing_base", 0.60)) + float(generator_cfg.get("testing_scale", 0.40)) * progress),
            float(generator_cfg.get("testing_floor", 0.005)),
            art,
        )
        state_trajectory = np.stack(
            [
                np.clip(1.0 - diagnosed, 0.0, 1.0),
                np.clip(diagnosed - art, 0.0, 1.0),
                np.clip(art - suppression, 0.0, 1.0),
                suppression,
                np.clip(
                    float(generator_cfg.get("death_base", 0.04))
                    + float(generator_cfg.get("death_reporting_scale", 0.05)) * definition.reporting_noise
                    + float(generator_cfg.get("death_time_scale", 0.02)) * (1.0 - progress),
                    float(generator_cfg.get("death_floor", 0.01)),
                    float(generator_cfg.get("death_ceiling", 0.18)),
                ),
            ],
            axis=-1,
        ).astype(np.float32)
        state_trajectory = state_trajectory / np.clip(state_trajectory.sum(axis=-1, keepdims=True), min_denominator, None)
        archetype_rows.append(
            {
                "archetype_id": archetype_id,
                "display_name": definition.display_name,
                "state_trajectory": state_trajectory.round(6).tolist(),
                "observed_trajectory": {
                    "diagnosed_stock": diagnosed.round(6).tolist(),
                    "art_stock": art.round(6).tolist(),
                    "documented_suppression": suppression.round(6).tolist(),
                    "testing_coverage": testing.round(6).tolist(),
                },
                "prior_parameters": {
                    "reporting_noise": definition.reporting_noise,
                    "diagnosis_prior_shift": definition.diagnosis_prior_shift,
                    "linkage_prior_shift": definition.linkage_prior_shift,
                    "retention_prior_shift": definition.retention_prior_shift,
                    "vl_testing_prior_shift": definition.vl_testing_prior_shift,
                    "documentation_prior_shift": definition.documentation_prior_shift,
                },
            }
        )
    return {
        "archetypes": [library[name].to_dict() for name in ARCHETYPE_ORDER],
        "month_axis": list(month_axis),
        "synthetic_trajectories": archetype_rows,
        "notes": [
            "synthetic_library_is_for_prior_pretraining_and_sparse_province_regularization",
            "synthetic_trajectories_do_not_override_observed_harp_or_official_anchor_surfaces",
        ],
    }


def infer_province_archetype_priors(
    *,
    province_axis: list[str],
    month_axis: list[str],
    subgroup_summary: dict[str, Any],
    observation_targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    cfg = _hiv_archetype_cfg()
    library = _archetype_library()
    default_feature = float(cfg.get("default_feature_value", 0.5))
    feature_table = _province_feature_table(
        province_axis=province_axis,
        month_axis=month_axis,
        subgroup_summary=subgroup_summary,
        observation_targets=observation_targets,
    )
    bonus_cfg = dict(cfg.get("under_reporting_bonus", {}) or {})
    archetype_vectors = {
        archetype_id: {
            **definition.prototype_features,
            "under_reporting_cue": float(
                bonus_cfg.get(
                    archetype_id,
                    bonus_cfg.get("default", 0.15),
                )
            ),
        }
        for archetype_id, definition in library.items()
    }
    feature_names = [
        "urbanity",
        "accessibility",
        "awareness",
        "stress",
        "kp_focus",
        "evidence_strength",
        "diagnosed_share",
        "art_share",
        "suppression_share",
        "testing_share",
        "under_reporting_cue",
    ]
    feature_weights = {key: float(value) for key, value in dict(cfg.get("feature_weights", {}) or {}).items()}
    migrant_bonus_cfg = dict(cfg.get("migrant_corridor_bonus", {}) or {})
    sparse_cfg = dict(cfg.get("sparse_pretraining_weight", {}) or {})
    transition_mix_cfg = dict(cfg.get("transition_shift_mix", {}) or {})
    observation_weight_cfg = dict(cfg.get("observation_weight_mix", {}) or {})
    mixture_rows = []
    mixture_matrix = np.zeros((len(province_axis), len(ARCHETYPE_ORDER)), dtype=np.float32)
    transition_prior_shift = np.zeros((len(province_axis), len(TRANSITION_ORDER)), dtype=np.float32)
    reporting_noise = np.zeros((len(province_axis),), dtype=np.float32)
    observation_weight = np.zeros((len(province_axis),), dtype=np.float32)
    vl_testing_prior_shift = np.zeros((len(province_axis),), dtype=np.float32)
    documentation_prior_shift = np.zeros((len(province_axis),), dtype=np.float32)
    synthetic_pretraining_weight = np.zeros((len(province_axis),), dtype=np.float32)
    for province_idx, province in enumerate(province_axis):
        features = feature_table.get(province, {})
        scores = []
        for archetype_id in ARCHETYPE_ORDER:
            prototype = archetype_vectors[archetype_id]
            distance = 0.0
            for feature_name in feature_names:
                lhs = float(features.get(feature_name, default_feature))
                rhs = float(prototype.get(feature_name, default_feature))
                distance += float(feature_weights.get(feature_name, 1.0)) * (lhs - rhs) ** 2
            under_reporting_bonus = 0.0
            if archetype_id == "under_reporting_province":
                under_reporting_bonus = float(bonus_cfg.get("under_reporting_province", 0.85)) * float(features.get("under_reporting_cue", 0.0))
            if archetype_id == "migrant_corridor":
                under_reporting_bonus = (
                    float(migrant_bonus_cfg.get("urbanity", 0.20)) * float(features.get("urbanity", default_feature))
                    + float(migrant_bonus_cfg.get("stress", 0.30)) * float(features.get("stress", default_feature))
                )
            scores.append(-distance + under_reporting_bonus)
        mixture = _softmax(np.asarray(scores, dtype=np.float32), temperature=float(cfg.get("softmax_temperature", 0.35)))
        mixture_matrix[province_idx] = mixture
        sparse_weight = float(
            np.clip(
                float(sparse_cfg.get("base", 0.20))
                + float(sparse_cfg.get("scale", 0.80)) * (1.0 - float(features.get("evidence_strength", default_feature))),
                float(sparse_cfg.get("floor", 0.20)),
                float(sparse_cfg.get("ceiling", 0.95)),
            )
        )
        synthetic_pretraining_weight[province_idx] = sparse_weight
        diag_shift = 0.0
        linkage_shift = 0.0
        retention_shift = 0.0
        vl_shift = 0.0
        doc_shift = 0.0
        noise = 0.0
        for archetype_id, weight in zip(ARCHETYPE_ORDER, mixture, strict=True):
            definition = library[archetype_id]
            diag_shift += float(weight) * definition.diagnosis_prior_shift
            linkage_shift += float(weight) * definition.linkage_prior_shift
            retention_shift += float(weight) * definition.retention_prior_shift
            vl_shift += float(weight) * definition.vl_testing_prior_shift
            doc_shift += float(weight) * definition.documentation_prior_shift
            noise += float(weight) * definition.reporting_noise
        transition_prior_shift[province_idx] = np.asarray(
            [
                sparse_weight * diag_shift,
                sparse_weight * linkage_shift,
                sparse_weight
                * (
                    float(transition_mix_cfg.get("a_to_v_retention_weight", 0.20)) * retention_shift
                    + float(transition_mix_cfg.get("a_to_v_documentation_weight", 0.10)) * doc_shift
                ),
                sparse_weight
                * (
                    float(transition_mix_cfg.get("a_to_l_retention_weight", -0.65)) * retention_shift
                    + float(transition_mix_cfg.get("a_to_l_noise_weight", 0.08)) * noise
                ),
                sparse_weight * (float(transition_mix_cfg.get("l_to_a_retention_weight", 0.75)) * retention_shift),
            ],
            dtype=np.float32,
        )
        reporting_noise[province_idx] = float(np.clip(noise, float(cfg.get("reporting_noise_floor", 0.05)), float(cfg.get("reporting_noise_ceiling", 0.85))))
        observation_weight[province_idx] = float(
            np.clip(
                (
                    1.0
                    - float(observation_weight_cfg.get("reporting_noise_weight", 0.55)) * reporting_noise[province_idx]
                )
                * (
                    float(observation_weight_cfg.get("evidence_base", 0.60))
                    + float(observation_weight_cfg.get("evidence_scale", 0.40)) * float(features.get("evidence_strength", default_feature))
                ),
                float(observation_weight_cfg.get("floor", 0.35)),
                float(observation_weight_cfg.get("ceiling", 1.0)),
            )
        )
        vl_testing_prior_shift[province_idx] = float(sparse_weight * vl_shift)
        documentation_prior_shift[province_idx] = float(sparse_weight * doc_shift)
        dominant_idx = int(np.argmax(mixture))
        dominant = ARCHETYPE_ORDER[dominant_idx]
        mixture_rows.append(
            {
                "province": province,
                "dominant_archetype": dominant,
                "dominant_weight": round(float(mixture[dominant_idx]), 6),
                "synthetic_pretraining_weight": round(sparse_weight, 6),
                "reporting_noise": round(float(reporting_noise[province_idx]), 6),
                "observation_weight": round(float(observation_weight[province_idx]), 6),
                "transition_prior_shift": {TRANSITION_ORDER[idx]: round(float(transition_prior_shift[province_idx, idx]), 6) for idx in range(len(TRANSITION_ORDER))},
                "vl_testing_prior_shift": round(float(vl_testing_prior_shift[province_idx]), 6),
                "documentation_prior_shift": round(float(documentation_prior_shift[province_idx]), 6),
                "features": {key: round(float(value), 6) for key, value in features.items()},
                "archetype_mixture": {ARCHETYPE_ORDER[idx]: round(float(mixture[idx]), 6) for idx in range(len(ARCHETYPE_ORDER))},
            }
        )
    region_mixture: dict[str, dict[str, float]] = {}
    for row in subgroup_summary.get("rows", []):
        province = str(row.get("province") or "")
        region = str(row.get("region") or "")
        if not province or not region:
            continue
        province_row = next((item for item in mixture_rows if item["province"] == province), None)
        if province_row is None:
            continue
        store = region_mixture.setdefault(region, {name: 0.0 for name in ARCHETYPE_ORDER})
        for name in ARCHETYPE_ORDER:
            store[name] += float(province_row["archetype_mixture"][name])
    for region, mixture in region_mixture.items():
        total = max(sum(mixture.values()), float(cfg.get("min_denominator", 1e-6)))
        region_mixture[region] = {name: round(value / total, 6) for name, value in mixture.items()}
    return {
        "archetype_definitions": [library[name].to_dict() for name in ARCHETYPE_ORDER],
        "rows": mixture_rows,
        "mixture_matrix": mixture_matrix.astype(np.float32),
        "transition_prior_shift": transition_prior_shift.astype(np.float32),
        "reporting_noise": reporting_noise.astype(np.float32),
        "observation_weight": observation_weight.astype(np.float32),
        "vl_testing_prior_shift": vl_testing_prior_shift.astype(np.float32),
        "documentation_prior_shift": documentation_prior_shift.astype(np.float32),
        "synthetic_pretraining_weight": synthetic_pretraining_weight.astype(np.float32),
        "region_archetype_mixture": region_mixture,
        "summary": {
            "province_count": len(province_axis),
            "dominant_counts": {
                archetype_id: sum(1 for row in mixture_rows if row["dominant_archetype"] == archetype_id)
                for archetype_id in ARCHETYPE_ORDER
            },
            "mean_reporting_noise": round(float(mean(reporting_noise.tolist())) if len(reporting_noise) else 0.0, 6),
            "mean_observation_weight": round(float(mean(observation_weight.tolist())) if len(observation_weight) else 0.0, 6),
            "mean_synthetic_pretraining_weight": round(float(mean(synthetic_pretraining_weight.tolist())) if len(synthetic_pretraining_weight) else 0.0, 6),
        },
    }
