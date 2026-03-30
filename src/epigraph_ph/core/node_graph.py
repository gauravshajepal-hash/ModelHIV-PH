from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from epigraph_ph.core.disease_plugin import DiseasePlugin, get_disease_plugin
from epigraph_ph.runtime import read_json


@dataclass(slots=True)
class NodeDefinition:
    block_id: str
    node_type: str
    core_distance: int
    acts_on: str
    positive_support_allowed: bool
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BlockEvidence:
    block_id: str
    prior_reliability: float
    truth_gain: float
    stability: float
    overlap_penalty: float
    drift_penalty: float
    posterior_reliability: float
    source_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RegionNodeState:
    region_id: str
    block_id: str
    raw_signal: float
    signal_direction: str
    regional_consistency: float
    regional_confidence: float
    final_node_weight: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DecisionNodeBundle:
    block_evidence: dict[str, BlockEvidence]
    region_node_states: list[RegionNodeState]
    risk_bonus_by_region: dict[str, float]
    risk_penalty_by_region: dict[str, float]
    risk_veto_flag_by_region: dict[str, bool]
    decision_penalty_by_region: dict[str, float]
    decision_support_by_region: dict[str, float]
    debug_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_evidence": {key: value.to_dict() for key, value in self.block_evidence.items()},
            "region_node_states": [row.to_dict() for row in self.region_node_states],
            "risk_bonus_by_region": dict(self.risk_bonus_by_region),
            "risk_penalty_by_region": dict(self.risk_penalty_by_region),
            "risk_veto_flag_by_region": dict(self.risk_veto_flag_by_region),
            "decision_penalty_by_region": dict(self.decision_penalty_by_region),
            "decision_support_by_region": dict(self.decision_support_by_region),
            "debug_summary": dict(self.debug_summary),
        }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _logit(value: float) -> float:
    plugin = get_disease_plugin("hiv")
    stabilizers = dict((plugin.numerical_stabilizers or {}).get("node_graph", {}) or {})
    floor = float(stabilizers.get("probability_floor", 1e-6))
    ceiling = float(stabilizers.get("probability_ceiling", 1.0 - floor))
    value = min(ceiling, max(floor, value))
    return math.log(value / (1.0 - value))


def _node_graph_cfg(plugin: DiseasePlugin) -> dict[str, Any]:
    return dict(plugin.constraint_settings.get("node_graph", {}) or {})


def _posterior(
    prior: float,
    *,
    truth_gain: float,
    stability: float,
    overlap_penalty: float,
    drift_penalty: float,
    cfg: dict[str, Any],
) -> float:
    update_cfg = dict(cfg.get("posterior_update", {}) or {})
    z_clip = float(update_cfg.get("z_clip", 2.0))
    floor = float(update_cfg.get("posterior_floor", 0.05))
    ceiling = float(update_cfg.get("posterior_ceiling", 0.95))

    def _bounded(value: float) -> float:
        return max(-z_clip, min(z_clip, float(value)))

    logit = _logit(prior)
    logit += float(update_cfg.get("truth_gain_weight", 1.2)) * _bounded(truth_gain)
    logit += float(update_cfg.get("stability_weight", 0.9)) * _bounded(stability)
    logit -= float(update_cfg.get("overlap_penalty_weight", 1.1)) * _bounded(overlap_penalty)
    logit -= float(update_cfg.get("drift_penalty_weight", 0.8)) * _bounded(drift_penalty)
    return round(min(ceiling, max(floor, _sigmoid(logit))), 6)


def _load_phase_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "factor_stability": read_json(run_dir / "phase15" / "factor_stability_report.json", default=[]),
        "factor_diagnostics": read_json(run_dir / "phase2" / "factor_diagnostics.json", default=[]),
        "factor_tournament": read_json(run_dir / "phase2" / "factor_tournament_results.json", default=[]),
        "promotion_admission": read_json(run_dir / "phase2" / "promotion_admission.json", default={}),
        "validation_artifact": read_json(run_dir / "phase3" / "validation_artifact.json", default={}),
        "benchmark_gates": read_json(run_dir / "phase3" / "benchmark_gate_report.json", default={}),
        "subgroup_weights": read_json(run_dir / "phase3" / "subgroup_weight_summary.json", default={}),
    }


def _node_definitions(plugin: DiseasePlugin) -> list[NodeDefinition]:
    return [NodeDefinition(**row) for row in plugin.node_graph_defaults]


def _evidence_snapshot(run_dir: Path, plugin: DiseasePlugin, phase_artifacts: dict[str, Any]) -> dict[str, BlockEvidence]:
    cfg = _node_graph_cfg(plugin)
    prior_map = dict(cfg.get("node_priors", {}) or {})
    fallback_cfg = dict(cfg.get("evidence_fallback", {}) or {})
    by_block: dict[str, dict[str, Any]] = {}
    for row in phase_artifacts["factor_tournament"]:
        block_id = str(row.get("block_name") or "")
        if not block_id:
            continue
        if block_id not in by_block or float(row.get("diagnostic_score") or 0.0) > float(by_block[block_id].get("diagnostic_score") or 0.0):
            by_block[block_id] = row
    stability_by_block: dict[str, dict[str, Any]] = {}
    for row in phase_artifacts["factor_stability"]:
        block_id = str(row.get("block_name") or "")
        if not block_id:
            continue
        if block_id not in stability_by_block or float(row.get("stability_score") or 0.0) > float(stability_by_block[block_id].get("stability_score") or 0.0):
            stability_by_block[block_id] = row
    primary_gates = phase_artifacts["benchmark_gates"].get("primary_gates", [])
    pass_rate = mean(1.0 if gate.get("passed") else 0.0 for gate in primary_gates) if primary_gates else 0.0
    evidence: dict[str, BlockEvidence] = {}
    for definition in _node_definitions(plugin):
        prior = float(prior_map.get(definition.node_type, prior_map.get("decision_modifier", 0.5)))
        diagnostic = by_block.get(definition.block_id, {})
        stability = stability_by_block.get(definition.block_id, {})
        truth_gain = float(diagnostic.get("diagnostic_score") or diagnostic.get("predictive_gain") or pass_rate)
        stability_score = float(stability.get("stability_score") or diagnostic.get("region_contrast_score") or 0.0)
        overlap_penalty = max(0.0, float(fallback_cfg.get("overlap_contrast_target", 0.4)) - float(diagnostic.get("region_contrast_score") or 0.0))
        drift_penalty = max(
            0.0,
            float(diagnostic.get("predictive_gain") or 0.0) - float(stability.get("predictive_gain_stability") or 0.0),
        )
        if definition.block_id == "benchmark_guardrails":
            truth_gain = pass_rate
            stability_score = max(pass_rate, float(fallback_cfg.get("benchmark_pass_floor", 0.25)))
            overlap_penalty = 0.0
            drift_penalty = max(0.0, float(fallback_cfg.get("benchmark_drift_target", 0.5)) - pass_rate)
        posterior = _posterior(
            prior,
            truth_gain=truth_gain,
            stability=stability_score,
            overlap_penalty=overlap_penalty,
            drift_penalty=drift_penalty,
            cfg=cfg,
        )
        evidence[definition.block_id] = BlockEvidence(
            block_id=definition.block_id,
            prior_reliability=prior,
            truth_gain=round(truth_gain, 6),
            stability=round(stability_score, 6),
            overlap_penalty=round(overlap_penalty, 6),
            drift_penalty=round(drift_penalty, 6),
            posterior_reliability=posterior,
            source_paths=[
                str(run_dir / "phase15" / "factor_stability_report.json"),
                str(run_dir / "phase2" / "factor_diagnostics.json"),
                str(run_dir / "phase3" / "benchmark_gate_report.json"),
            ],
        )
    return evidence


def _region_inputs(phase_artifacts: dict[str, Any]) -> dict[str, dict[str, float]]:
    plugin = get_disease_plugin("hiv")
    cfg = _node_graph_cfg(plugin)
    default_signal = dict(cfg.get("default_region_signal", {}) or {})
    regional_metrics = {
        str(row.get("region") or ""): {
            "diagnosed_mae": float(row.get("diagnosed_mae") or 0.0),
            "art_mae": float(row.get("art_mae") or 0.0),
            "suppression_mae": float(row.get("suppression_mae") or 0.0),
        }
        for row in phase_artifacts["validation_artifact"].get("regional_metrics", [])
        if row.get("region")
    }
    subgroup_rows = phase_artifacts["subgroup_weights"].get("rows", [])
    signals: dict[str, dict[str, list[float]]] = {}
    for row in subgroup_rows:
        region = str(row.get("region") or "")
        if not region:
            continue
        store = signals.setdefault(
            region,
            {
                "accessibility": [],
                "awareness": [],
                "stress": [],
                "urbanity": [],
                "kp_focus": [],
            },
        )
        network = row.get("network_signal") or {}
        store["accessibility"].append(float(network.get("accessibility") or 0.0))
        store["awareness"].append(float(network.get("awareness") or 0.0))
        store["stress"].append(float(network.get("stress") or 0.0))
        store["urbanity"].append(float(network.get("urbanity") or 0.0))
        kp_distribution = row.get("kp_distribution") or {}
        store["kp_focus"].append(float(kp_distribution.get("msm") or 0.0))
    out: dict[str, dict[str, float]] = {}
    for region, values in signals.items():
        out[region] = {
            key: round(mean(series), 6) if series else 0.0
            for key, series in values.items()
        }
        out[region].update(regional_metrics.get(region, {}))
    for region, metrics in regional_metrics.items():
        out.setdefault(region, {}).update(metrics)
    if not out:
        national = phase_artifacts["validation_artifact"].get("national_metrics", {})
        out["national"] = {
            "accessibility": float(default_signal.get("accessibility", 0.5)),
            "awareness": float(default_signal.get("awareness", 0.0)),
            "stress": float(default_signal.get("stress", 0.0)),
            "urbanity": float(default_signal.get("urbanity", 0.5)),
            "kp_focus": float(default_signal.get("kp_focus", 0.5)),
            "diagnosed_mae": float(national.get("diagnosed_mae") or 0.0),
            "art_mae": float(national.get("art_mae") or 0.0),
            "suppression_mae": float(national.get("suppression_mae") or 0.0),
        }
    return out


def _raw_signal(definition: NodeDefinition, region_values: dict[str, float], *, cfg: dict[str, Any]) -> tuple[float, str, str]:
    accessibility = float(region_values.get("accessibility") or 0.0)
    awareness = float(region_values.get("awareness") or 0.0)
    stress = float(region_values.get("stress") or 0.0)
    urbanity = float(region_values.get("urbanity") or 0.0)
    kp_focus = float(region_values.get("kp_focus") or 0.0)
    diagnosed_mae = float(region_values.get("diagnosed_mae") or 0.0)
    art_mae = float(region_values.get("art_mae") or 0.0)
    suppression_mae = float(region_values.get("suppression_mae") or 0.0)
    direction_cfg = dict(cfg.get("signal_direction", {}) or {})
    weights = dict((cfg.get("signal_weights", {}) or {}).get(definition.block_id, {}) or {})
    if definition.block_id == "mobility_network_mixing":
        raw = float(weights.get("accessibility", 0.0)) * accessibility + float(weights.get("urbanity", 0.0)) * urbanity + float(weights.get("stress", 0.0)) * stress
        explanation = "Mobility node favors accessible, urban, lower-stress corridors and cautions stressed continuity regions."
    elif definition.block_id == "logistics_access":
        raw = -abs(float(weights.get("stress", 0.0)) * stress + abs(float(weights.get("accessibility_gap", 0.0))) * max(0.0, float((cfg.get("default_region_signal", {}) or {}).get("accessibility", 0.5)) - accessibility))
        explanation = "Logistics node is negative-only and penalizes stressed, weak-access regions."
    elif definition.block_id == "service_delivery_infrastructure":
        raw = -abs(float(weights.get("art_mae", 0.0)) * art_mae + float(weights.get("suppression_mae", 0.0)) * suppression_mae)
        explanation = "Service-delivery node penalizes regions where treatment and suppression errors remain high."
    elif definition.block_id == "stigma_behavior_information":
        raw = float(weights.get("awareness", 0.0)) * awareness + float(weights.get("stress", 0.0)) * stress + float(weights.get("diagnosed_mae", 0.0)) * diagnosed_mae
        explanation = "Stigma-information node rewards awareness reach and cautions regions with stress and diagnosis slippage."
    elif definition.block_id == "population_structure_demography":
        raw = float(weights.get("kp_focus", 0.0)) * kp_focus + float(weights.get("urbanity", 0.0)) * urbanity + float(weights.get("stress", 0.0)) * stress
        explanation = "Population-structure node supports regions with stronger targetable subgroup concentration and urban reach."
    elif definition.block_id == "policy_implementation":
        raw = -abs(
            abs(float(weights.get("diagnosed_mae", 0.0))) * diagnosed_mae
            + abs(float(weights.get("art_mae", 0.0))) * art_mae
            + abs(float(weights.get("suppression_mae", 0.0))) * suppression_mae
        )
        explanation = "Policy-implementation node is negative-only and tracks execution weakness through regional validation error."
    else:
        raw = -abs(
            abs(float(weights.get("diagnosed_mae", 0.0))) * diagnosed_mae
            + abs(float(weights.get("art_mae", 0.0))) * art_mae
            + abs(float(weights.get("suppression_mae", 0.0))) * suppression_mae
        )
        explanation = "Benchmark-guardrails node penalizes regions where regional validation error remains elevated."
    raw = max(-1.0, min(1.0, raw))
    if definition.node_type == "constraint":
        raw = min(0.0, raw)
        direction = "veto" if raw <= -0.8 else "caution" if raw < 0.0 else "neutral"
    elif raw > float(direction_cfg.get("support_floor", 0.0)):
        direction = "support"
    elif raw < float(direction_cfg.get("caution_ceiling", 0.0)):
        direction = "caution"
    else:
        direction = "neutral"
    return round(raw, 6), direction, explanation


def build_node_graph_bundle(*, run_dir: str | Path, plugin_id: str) -> DecisionNodeBundle:
    run_root = Path(run_dir)
    plugin = get_disease_plugin(plugin_id)
    cfg = _node_graph_cfg(plugin)
    cap_cfg = dict(cfg.get("caps", {}) or {})
    veto_cfg = dict(cfg.get("veto", {}) or {})
    consistency_cfg = dict(cfg.get("consistency_weights", {}) or {})
    confidence_cfg = dict(cfg.get("confidence_mix", {}) or {})
    risk_bonus_max = float(cap_cfg.get("risk_bonus_max", 0.08))
    risk_penalty_max = float(cap_cfg.get("risk_penalty_max", 0.12))
    support_max = float(cap_cfg.get("regional_support_max", 1.0))
    penalty_max = float(cap_cfg.get("regional_penalty_max", 1.0))
    support_scale = float(cap_cfg.get("support_scale", 0.08))
    penalty_scale = float(cap_cfg.get("penalty_scale", 0.12))
    veto_weight_threshold = float(veto_cfg.get("weight_threshold", 0.75))
    veto_signal_threshold = float(veto_cfg.get("signal_threshold", -0.80))
    phase_artifacts = _load_phase_artifacts(run_root)
    evidence = _evidence_snapshot(run_root, plugin, phase_artifacts)
    region_values = _region_inputs(phase_artifacts)
    definitions = _node_definitions(plugin)
    region_states: list[RegionNodeState] = []
    risk_bonus: dict[str, float] = {region: 0.0 for region in region_values}
    risk_penalty: dict[str, float] = {region: 0.0 for region in region_values}
    risk_veto: dict[str, bool] = {region: False for region in region_values}
    decision_penalty: dict[str, float] = {region: 0.0 for region in region_values}
    decision_support: dict[str, float] = {region: 0.0 for region in region_values}
    for definition in definitions:
        evidence_row = evidence[definition.block_id]
        for region, values in region_values.items():
            raw_signal, direction, explanation = _raw_signal(definition, values, cfg=cfg)
            signal_consistency = 1.0 - min(
                1.0,
                float(consistency_cfg.get("diagnosed_mae", 0.0)) * float(values.get("diagnosed_mae") or 0.0)
                + float(consistency_cfg.get("suppression_mae", 0.0)) * float(values.get("suppression_mae") or 0.0),
            )
            regional_consistency = round(max(0.0, min(1.0, signal_consistency)), 6)
            regional_confidence = round(
                max(
                    0.0,
                    min(
                        1.0,
                        float(confidence_cfg.get("posterior_weight", 0.0)) * evidence_row.posterior_reliability
                        + float(confidence_cfg.get("consistency_weight", 0.0)) * regional_consistency,
                    ),
                ),
                6,
            )
            final_weight = round(max(0.0, min(1.0, evidence_row.posterior_reliability * regional_consistency)), 6)
            state = RegionNodeState(
                region_id=region,
                block_id=definition.block_id,
                raw_signal=raw_signal,
                signal_direction=direction,
                regional_consistency=regional_consistency,
                regional_confidence=regional_confidence,
                final_node_weight=final_weight,
                explanation=explanation,
            )
            region_states.append(state)
            if raw_signal > 0.0 and definition.positive_support_allowed:
                bonus = min(risk_bonus_max, raw_signal * final_weight * support_scale)
                risk_bonus[region] = round(min(risk_bonus_max, risk_bonus[region] + bonus), 6)
                decision_support[region] = round(min(support_max, decision_support[region] + raw_signal * final_weight), 6)
            elif raw_signal < 0.0:
                penalty = min(risk_penalty_max, abs(raw_signal) * final_weight * penalty_scale)
                risk_penalty[region] = round(min(risk_penalty_max, risk_penalty[region] + penalty), 6)
                decision_penalty[region] = round(min(penalty_max, decision_penalty[region] + abs(raw_signal) * final_weight), 6)
            if definition.node_type == "constraint" and final_weight >= veto_weight_threshold and raw_signal <= veto_signal_threshold:
                risk_veto[region] = True
    return DecisionNodeBundle(
        block_evidence=evidence,
        region_node_states=region_states,
        risk_bonus_by_region=risk_bonus,
        risk_penalty_by_region=risk_penalty,
        risk_veto_flag_by_region=risk_veto,
        decision_penalty_by_region=decision_penalty,
        decision_support_by_region=decision_support,
        debug_summary={
            "plugin_id": plugin_id,
            "region_count": len(region_values),
            "node_count": len(definitions),
            "enabled_node_count": sum(1 for row in definitions if row.enabled),
            "advisory_only_default": True,
        },
    )
