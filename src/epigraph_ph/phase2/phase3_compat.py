from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.runtime import load_tensor_artifact, read_json, save_tensor_artifact, write_json


def select_retained_factor_rows(
    *,
    phase15_dir: Path,
    predictive_budget: int,
    context_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pool_rows = list(read_json(phase15_dir / "factor_promotion_pool.json", default=[]))
    multiscale_rows = list(read_json(phase15_dir / "multiscale_factor_catalog.json", default=[]))
    by_id: dict[str, dict[str, Any]] = {}
    multiscale_by_id: dict[str, dict[str, Any]] = {}
    for row in multiscale_rows:
        factor_id = str(row.get("factor_id") or "")
        if factor_id:
            payload = dict(row)
            by_id[factor_id] = payload
            multiscale_by_id[factor_id] = payload
    for row in pool_rows:
        factor_id = str(row.get("factor_id") or "")
        if not factor_id:
            continue
        merged = dict(by_id.get(factor_id, {}))
        merged.update(dict(row))
        by_id[factor_id] = merged

    def _score(row: dict[str, Any]) -> tuple[float, float, float, str]:
        return (
            float(row.get("survival_score") or 0.0),
            float(row.get("predictive_gain") or 0.0),
            float(row.get("stability_score") or 0.0),
            str(row.get("factor_id") or ""),
        )

    def _eligible(classes: set[str]) -> list[dict[str, Any]]:
        rows = []
        for row in by_id.values():
            survival_class = str(row.get("survival_class") or "")
            promotion_class = str(row.get("promotion_class") or "")
            if survival_class in classes or promotion_class in classes:
                rows.append(dict(row))
        rows.sort(key=_score, reverse=True)
        return rows

    selection_method = "phase15_survival_classes_intersect_multiscale_catalog"
    predictive_rows = _eligible({"survivor_primary", "retained_predictive"})[: max(0, int(predictive_budget))]
    for row in predictive_rows:
        row["promotion_class"] = "retained_predictive"
        row.setdefault("selection_variant", "phase2_latent_multiscale_restore")

    context_rows = _eligible({"survivor_secondary", "retained_context"})[: max(0, int(context_budget))]
    seen_predictive = {str(row.get("factor_id") or "") for row in predictive_rows}
    context_rows = [row for row in context_rows if str(row.get("factor_id") or "") not in seen_predictive]
    for row in context_rows:
        row["promotion_class"] = "retained_context"
        row.setdefault("selection_variant", "phase2_latent_multiscale_restore")

    multiscale_ids = set(multiscale_by_id)
    predictive_rows = [row for row in predictive_rows if str(row.get("factor_id") or "") in multiscale_ids]
    context_rows = [row for row in context_rows if str(row.get("factor_id") or "") in multiscale_ids]

    if not predictive_rows and not context_rows:
        selection_method = "multiscale_catalog_fallback"
        ranked_multiscale = [
            dict(row)
            for row in multiscale_rows
            if bool(row.get("hard_checks_passed"))
        ]
        ranked_multiscale.sort(key=_score, reverse=True)
        predictive_rows = [
            row
            for row in ranked_multiscale
            if list(row.get("transition_hooks") or []) or str(row.get("best_target") or "")
        ][: max(0, int(predictive_budget))]
        predictive_ids = {str(row.get("factor_id") or "") for row in predictive_rows}
        context_rows = [
            row
            for row in ranked_multiscale
            if str(row.get("factor_id") or "") not in predictive_ids
        ][: max(0, int(context_budget))]
        for row in predictive_rows:
            row["promotion_class"] = "retained_predictive"
            row.setdefault("selection_variant", "phase2_multiscale_catalog_fallback")
        for row in context_rows:
            row["promotion_class"] = "retained_context"
            row.setdefault("selection_variant", "phase2_multiscale_catalog_fallback")

    retained_catalog_rows = [dict(row) for row in [*predictive_rows, *context_rows]]
    if not retained_catalog_rows:
        selection_method = "hard_checks_fallback"
        fallback_rows = [
            dict(row)
            for row in by_id.values()
            if bool(row.get("hard_checks_passed"))
        ]
        fallback_rows.sort(key=_score, reverse=True)
        retained_catalog_rows = fallback_rows[: max(int(predictive_budget) + int(context_budget), 8)]

    retained_catalog = {
        "rows": retained_catalog_rows,
        "predictive_count": len(predictive_rows),
        "context_count": len(context_rows),
        "selection_method": selection_method,
    }
    return predictive_rows, context_rows, retained_catalog


def _legacy_primary_block(row: dict[str, Any]) -> str:
    pathway = str(row.get("dominant_pathway_family") or "").lower()
    domain = str(row.get("dominant_domain_family") or "").lower()
    canonical = str(row.get("canonical_name") or "").lower()
    text = " ".join([pathway, domain, canonical])
    if any(token in text for token in ("poverty", "economic", "cost", "cash", "housing", "education")):
        return "economics"
    if any(token in text for token in ("travel", "transport", "remoteness", "mobility", "clinic", "facility", "service", "health_system")):
        return "logistics"
    if any(token in text for token in ("population", "demograph", "migration", "household")):
        return "population"
    if any(token in text for token in ("viral", "suppression", "cd4", "biolog", "progression")):
        return "biology"
    if "policy" in text:
        return "policy"
    return "behavior"


def _factor_transition_linkage_targets(row: dict[str, Any]) -> list[str]:
    targets: set[str] = set()
    best_target = str(row.get("best_target") or "").strip()
    if best_target:
        targets.add(best_target)
    for hook in list(row.get("transition_hooks") or []):
        hook_name = str(hook).strip()
        if hook_name == "diagnosis_transitions":
            targets.update({"diagnosed_stock", "testing_uptake"})
        elif hook_name == "linkage_transitions":
            targets.update({"art_stock", "linkage_to_care"})
        elif hook_name == "suppression_transitions":
            targets.update({"documented_suppression", "suppression_outcomes"})
        elif hook_name == "retention_attrition_transitions":
            targets.update({"testing_coverage", "retention_adherence"})
    return sorted(targets)


def _factor_priority(row: dict[str, Any]) -> float:
    numeric_values = [
        float(row.get("predictive_gain") or 0.0),
        float(row.get("survival_score") or 0.0),
        float(row.get("stability_score") or 0.0),
        float(row.get("calibration_score") or 0.0),
    ]
    return max(numeric_values) if numeric_values else 0.0


def _canonical_phase1_rollup(
    *,
    normalized_rows: list[dict[str, Any]],
    canonical_axis: list[str],
) -> dict[str, dict[str, Any]]:
    axis_set = {str(name) for name in canonical_axis}
    rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "support_count": 0,
            "numeric_support": 0,
            "anchor_support": 0,
            "direct_support": 0,
            "evidence_weight": 0.0,
            "observation_weight": 0.0,
            "domain_families": Counter(),
            "pathway_families": Counter(),
            "source_banks": Counter(),
            "geo_resolutions": Counter(),
            "regions": Counter(),
            "times": Counter(),
            "tags": Counter(),
            "targets": Counter(),
        }
    )
    for row in normalized_rows:
        canonical_name = str(row.get("canonical_name") or "")
        if canonical_name not in axis_set:
            continue
        item = rollup[canonical_name]
        item["support_count"] += 1
        item["numeric_support"] += 1 if row.get("model_numeric_value") is not None else 0
        item["anchor_support"] += 1 if row.get("is_anchor_eligible") else 0
        item["direct_support"] += 1 if str(row.get("observation_role") or "") == "direct_indicator" else 0
        item["evidence_weight"] += float(row.get("evidence_weight") or 0.0)
        item["observation_weight"] += float(row.get("observation_weight") or row.get("quality_weight") or 0.0)
        item["domain_families"][str(row.get("domain_family") or "mixed")] += 1
        item["pathway_families"][str(row.get("pathway_family") or "mixed")] += 1
        item["source_banks"][str(row.get("source_bank") or "unknown")] += 1
        item["geo_resolutions"][str(row.get("geo_resolution") or "unknown")] += 1
        item["regions"][str(row.get("region") or row.get("macro_region") or "")] += 1
        item["times"][str(row.get("time") or row.get("year") or "")] += 1
        for tag in list(row.get("soft_ontology_tags") or []):
            item["tags"][str(tag)] += 1
        for target in list(row.get("linkage_targets") or []):
            item["targets"][str(target)] += 1
    return rollup


def _candidate_profiles(
    *,
    normalized_rows: list[dict[str, Any]],
    canonical_axis: list[str],
    direct_temporal_nodes: list[str],
    hidden_driver_nodes: list[str],
    multiscale_support_nodes: list[str],
    predictive_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    retained_catalog_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    direct_set = {str(node) for node in direct_temporal_nodes if str(node)}
    hidden_set = {str(node) for node in hidden_driver_nodes if str(node)}
    multiscale_set = {str(node) for node in multiscale_support_nodes if str(node)}
    rollup = _canonical_phase1_rollup(normalized_rows=normalized_rows, canonical_axis=canonical_axis)
    factor_rows = [dict(row) for row in [*predictive_rows, *context_rows, *retained_catalog_rows]]
    canonical_to_factors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictive_factor_ids = {str(row.get("factor_id") or "") for row in predictive_rows if str(row.get("factor_id") or "")}
    context_factor_ids = {str(row.get("factor_id") or "") for row in context_rows if str(row.get("factor_id") or "")}
    for factor_row in factor_rows:
        member_names = [str(name) for name in list(factor_row.get("member_canonical_names") or []) if str(name)]
        for canonical_name in member_names:
            canonical_to_factors[canonical_name].append(factor_row)

    rows: list[dict[str, Any]] = []
    for canonical_name in canonical_axis:
        item = rollup.get(str(canonical_name), {})
        support_count = int(item.get("support_count") or 0)
        numeric_support = int(item.get("numeric_support") or 0)
        anchor_support = int(item.get("anchor_support") or 0)
        direct_support = int(item.get("direct_support") or 0)
        support_denom = max(support_count, 1)
        evidence_score = float(item.get("evidence_weight") or 0.0) / support_denom
        observation_score = float(item.get("observation_weight") or 0.0) / support_denom
        attached_factors = canonical_to_factors.get(str(canonical_name), [])
        factor_priority = max((_factor_priority(row) for row in attached_factors), default=0.0)
        factor_stability = float(np.mean([float(row.get("stability_score") or 0.0) for row in attached_factors])) if attached_factors else 0.0
        factor_blocks = [str(row.get("block_name") or row.get("interpretability_label") or "") for row in attached_factors if str(row.get("block_name") or row.get("interpretability_label") or "")]
        dominant_domain = item.get("domain_families", Counter()).most_common(1)[0][0] if item.get("domain_families") else "mixed"
        dominant_pathway = item.get("pathway_families", Counter()).most_common(1)[0][0] if item.get("pathway_families") else "mixed"
        primary_block = (
            Counter(factor_blocks).most_common(1)[0][0]
            if factor_blocks
            else _legacy_primary_block(
                {
                    "canonical_name": canonical_name,
                    "dominant_domain_family": dominant_domain,
                    "dominant_pathway_family": dominant_pathway,
                }
            )
        )
        direct_temporal_member = str(canonical_name) in direct_set
        hidden_driver_member = str(canonical_name) in hidden_set
        multiscale_support_member = str(canonical_name) in multiscale_set
        blanket_member = direct_temporal_member
        promotion_class = "retained_predictive" if any(str(row.get("factor_id") or "") in predictive_factor_ids for row in attached_factors) else (
            "retained_context" if any(str(row.get("factor_id") or "") in context_factor_ids for row in attached_factors) else "blanket_only"
        )
        if promotion_class == "retained_predictive" or (direct_temporal_member and factor_priority > 0.0):
            curation_status = "promoted_candidate"
        elif promotion_class in {"retained_context", "blanket_only"} or direct_temporal_member or support_count > 0:
            curation_status = "research_candidate"
        else:
            curation_status = "review"
        linkage_targets = set(str(target) for target in item.get("targets", Counter()).keys())
        for factor_row in attached_factors:
            linkage_targets.update(_factor_transition_linkage_targets(factor_row))
        secondary_blocks = sorted({block for block in factor_blocks if block and block != primary_block})
        source_diversity = len(item.get("source_banks", Counter()))
        geo_diversity = len(item.get("geo_resolutions", Counter()))
        region_diversity = len({key for key in item.get("regions", Counter()) if key})
        time_diversity = len({key for key in item.get("times", Counter()) if key})
        domain_diversity = len(item.get("domain_families", Counter()))
        pathway_diversity = len(item.get("pathway_families", Counter()))
        tag_diversity = len(item.get("tags", Counter()))
        target_diversity = len(linkage_targets)
        dag_score = 1.0 if blanket_member else 0.0
        relevance_score = max(float(evidence_score), float(factor_priority))
        curation_score = max(float(observation_score), float(factor_priority), float(dag_score))
        rows.append(
            {
                "canonical_name": canonical_name,
                "blanket_member": blanket_member,
                "direct_temporal_member": direct_temporal_member,
                "hidden_driver_member": hidden_driver_member,
                "multiscale_support_member": multiscale_support_member,
                "primary_block": primary_block,
                "curation_status": curation_status,
                "promotion_class": promotion_class,
                "curation_score": round(curation_score, 4),
                "dag_score": round(dag_score, 4),
                "relevance_score": round(relevance_score, 4),
                "stability_score": round(factor_stability, 4),
                "evidence_score": round(evidence_score, 4),
                "support_count": support_count,
                "numeric_support": numeric_support,
                "anchor_support": anchor_support,
                "direct_support": direct_support,
                "linkage_targets": sorted(linkage_targets),
                "source_banks": dict(item.get("source_banks", Counter())),
                "dominant_domain_family": dominant_domain,
                "dominant_pathway_family": dominant_pathway,
                "secondary_blocks": secondary_blocks,
                "soft_ontology_tags": [tag for tag, _ in item.get("tags", Counter()).most_common(8)],
                "source_diversity": source_diversity,
                "geo_diversity": geo_diversity,
                "region_diversity": region_diversity,
                "time_diversity": time_diversity,
                "domain_diversity": domain_diversity,
                "pathway_diversity": pathway_diversity,
                "tag_diversity": tag_diversity,
                "target_diversity": target_diversity,
                "source_id_count": source_diversity,
                "numeric_support_ratio": round(numeric_support / support_denom, 4),
                "anchor_support_ratio": round(anchor_support / support_denom, 4),
                "direct_support_ratio": round(direct_support / support_denom, 4),
                "in_strength": round(dag_score, 6),
                "out_strength": round(dag_score * 0.5, 6),
                "eligibility_sources": [
                    source_name
                    for source_name, active in (
                        ("direct_temporal", direct_temporal_member),
                        ("hidden_driver", hidden_driver_member),
                        ("multiscale_support", multiscale_support_member),
                    )
                    if active
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            float(row.get("curation_score") or 0.0),
            float(row.get("stability_score") or 0.0),
            float(row.get("evidence_score") or 0.0),
            str(row.get("canonical_name") or ""),
        ),
        reverse=True,
    )
    return rows


def _curated_candidate_blocks(candidate_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_profiles:
        grouped[str(row.get("primary_block") or "mixed")].append(row)
    rows: list[dict[str, Any]] = []
    for block_name, members in sorted(grouped.items()):
        kept = [row for row in members if str(row.get("curation_status") or "") in {"promoted_candidate", "research_candidate", "review"}]
        if not kept:
            continue
        rows.append(
            {
                "block_name": block_name,
                "candidate_count": len(kept),
                "blanket_member_count": sum(1 for row in kept if row.get("blanket_member")),
                "top_candidates": [str(row.get("canonical_name") or "") for row in kept[:6]],
            }
        )
    return rows


def _ranked_linkages(candidate_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidate_profiles:
        canonical_name = str(row.get("canonical_name") or "")
        targets = list(row.get("linkage_targets") or [])
        if not targets:
            continue
        linkage_score = max(
            float(row.get("curation_score") or 0.0),
            float(row.get("stability_score") or 0.0),
            float(row.get("evidence_score") or 0.0),
        )
        for target in targets:
            rows.append(
                {
                    "canonical_name": canonical_name,
                    "linkage_target": str(target),
                    "linkage_score": round(linkage_score, 4),
                    "support_count": int(row.get("support_count") or 0),
                    "numeric_support": int(row.get("numeric_support") or 0),
                    "primary_block": str(row.get("primary_block") or "mixed"),
                    "source_bank_count": len(dict(row.get("source_banks") or {})),
                }
            )
    rows.sort(key=lambda row: (float(row.get("linkage_score") or 0.0), int(row.get("support_count") or 0), str(row.get("canonical_name") or "")), reverse=True)
    return rows


def _surface_payload(
    *,
    name: str,
    nodes: list[str],
    canonical_axis: list[str],
) -> dict[str, Any]:
    canonical_set = {str(name) for name in canonical_axis}
    deduped = sorted({str(node) for node in nodes if str(node) in canonical_set})
    index = {name: idx for idx, name in enumerate(canonical_axis)}
    return {
        "name": name,
        "blanket_nodes": deduped,
        "blanket_indices": [int(index[node]) for node in deduped if node in index],
        "available": bool(deduped),
    }


def _surface_tensor(
    standardized_tensor: np.ndarray,
    *,
    indices: list[int],
) -> np.ndarray:
    if indices:
        return np.asarray(standardized_tensor[:, :, indices], dtype=np.float32)
    return np.zeros((*standardized_tensor.shape[:2], 0), dtype=np.float32)


def _edge_score_surfaces(multiscale_bundle: dict[str, Any], latent_bundle: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    direct_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    multiscale_rows: list[dict[str, Any]] = []
    for scale_name, scale_bundle in dict(multiscale_bundle.get("scales") or {}).items():
        for edge in list(scale_bundle.get("edges") or []):
            weight = float(edge.get("weight") or edge.get("score") or 0.0)
            multiscale_rows.append(
                {
                    "source": str(edge.get("source") or ""),
                    "target": str(edge.get("target") or ""),
                    "weight": round(weight, 6),
                    "abs_weight": round(abs(weight), 6),
                    "scale": scale_name,
                    "family": "multiscale_support",
                }
            )
    for scale_name, scale_bundle in dict(latent_bundle.get("scales") or {}).items():
        for edge in list(scale_bundle.get("edges") or []):
            weight = float(edge.get("weight") or 0.0)
            direct_rows.append(
                {
                    "source": f"{edge.get('source')}@lag{edge.get('lag')}",
                    "target": str(edge.get("target") or ""),
                    "weight": round(weight, 6),
                    "abs_weight": round(abs(weight), 6),
                    "scale": scale_name,
                    "family": "direct_temporal",
                }
            )
        for edge in list(scale_bundle.get("hidden_driver_rows") or []):
            weight = float(edge.get("weight") or 0.0)
            hidden_rows.append(
                {
                    "source": f"{edge.get('source')}@lag{edge.get('lag')}",
                    "target": str(edge.get("target") or ""),
                    "weight": round(weight, 6),
                    "abs_weight": round(abs(weight), 6),
                    "scale": scale_name,
                    "family": "hidden_driver",
                }
            )
    for rows in (direct_rows, hidden_rows, multiscale_rows):
        rows.sort(key=lambda row: (float(row.get("abs_weight") or 0.0), str(row.get("source") or ""), str(row.get("target") or "")), reverse=True)
    return {
        "direct_temporal": direct_rows,
        "hidden_driver": hidden_rows,
        "multiscale_support": multiscale_rows,
    }


def build_phase3_compatibility_payload(
    *,
    phase1_dir: Path,
    predictive_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    retained_catalog: dict[str, Any],
    multiscale_blankets: dict[str, Any],
    latent_blankets: dict[str, Any],
    multiscale_bundle: dict[str, Any],
    latent_bundle: dict[str, Any],
) -> dict[str, Any]:
    axis_catalogs = read_json(phase1_dir / "axis_catalogs.json", default={})
    canonical_axis = [str(value) for value in list(axis_catalogs.get("canonical", []) or axis_catalogs.get("canonical_name", []))]
    standardized_tensor = np.asarray(load_tensor_artifact(phase1_dir / "standardized_tensor.npz"), dtype=np.float32)
    normalized_rows = list(read_json(phase1_dir / "normalized_subparameters.json", default=[]))
    direct_temporal_surface = _surface_payload(
        name="direct_temporal",
        nodes=list(latent_blankets.get("direct_phase3_member_canonical_names") or latent_blankets.get("phase3_member_canonical_names") or []),
        canonical_axis=canonical_axis,
    )
    hidden_driver_surface = _surface_payload(
        name="hidden_driver",
        nodes=list(latent_blankets.get("hidden_phase3_member_canonical_names") or []),
        canonical_axis=canonical_axis,
    )
    multiscale_support_surface = _surface_payload(
        name="multiscale_support",
        nodes=list(multiscale_blankets.get("phase3_member_canonical_names") or []),
        canonical_axis=canonical_axis,
    )

    candidate_profiles = _candidate_profiles(
        normalized_rows=normalized_rows,
        canonical_axis=canonical_axis,
        direct_temporal_nodes=list(direct_temporal_surface["blanket_nodes"]),
        hidden_driver_nodes=list(hidden_driver_surface["blanket_nodes"]),
        multiscale_support_nodes=list(multiscale_support_surface["blanket_nodes"]),
        predictive_rows=predictive_rows,
        context_rows=context_rows,
        retained_catalog_rows=list(retained_catalog.get("rows") or []),
    )
    blanket_nodes = list(direct_temporal_surface["blanket_nodes"])
    blanket_indices = list(direct_temporal_surface["blanket_indices"])
    blanket_set = set(blanket_nodes)
    for row in candidate_profiles:
        row["blanket_member"] = str(row.get("canonical_name") or "") in blanket_set
    curated_candidate_blocks = _curated_candidate_blocks(candidate_profiles)
    ranked_linkages = _ranked_linkages(candidate_profiles)
    edge_score_surfaces = _edge_score_surfaces(multiscale_bundle, latent_bundle)

    phase3_target_blankets = {
        "blanket_factor_ids": [],
        "target_factor_ids": [],
        "blanket_indices": list(blanket_indices),
        "phase3_member_canonical_names": list(blanket_nodes),
        "surface_name": "direct_temporal",
    }
    markov_blanket = {
        "blanket_indices": list(blanket_indices),
        "blanket_nodes": list(blanket_nodes),
        "target_nodes": list(blanket_nodes),
        "surface_name": "direct_temporal",
    }
    direct_tensor = _surface_tensor(standardized_tensor, indices=blanket_indices)
    hidden_tensor = _surface_tensor(standardized_tensor, indices=list(hidden_driver_surface["blanket_indices"]))
    multiscale_tensor = _surface_tensor(standardized_tensor, indices=list(multiscale_support_surface["blanket_indices"]))
    eligibility_surfaces = {
        "direct_temporal": direct_temporal_surface,
        "hidden_driver": hidden_driver_surface,
        "multiscale_support": multiscale_support_surface,
    }
    return {
        "candidate_profiles": candidate_profiles,
        "curated_candidate_blocks": curated_candidate_blocks,
        "markov_blanket": markov_blanket,
        "edge_scores": edge_score_surfaces["direct_temporal"],
        "hidden_driver_edge_scores": edge_score_surfaces["hidden_driver"],
        "multiscale_edge_scores": edge_score_surfaces["multiscale_support"],
        "ranked_linkages": ranked_linkages,
        "phase3_target_blankets": phase3_target_blankets,
        "eligibility_surfaces": eligibility_surfaces,
        "retained_predictive_factor_set": predictive_rows,
        "retained_context_factor_set": context_rows,
        "promoted_factor_set": predictive_rows,
        "supporting_factor_set": context_rows,
        "retained_mesoscopic_factor_catalog": retained_catalog,
        "core_feature_tensor_array": direct_tensor,
        "direct_feature_tensor_array": direct_tensor,
        "hidden_driver_feature_tensor_array": hidden_tensor,
        "multiscale_support_feature_tensor_array": multiscale_tensor,
    }


def build_phase3_compatibility_artifacts(
    *,
    phase2_dir: Path,
    phase1_dir: Path,
    predictive_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    retained_catalog: dict[str, Any],
    multiscale_blankets: dict[str, Any],
    latent_blankets: dict[str, Any],
    multiscale_bundle: dict[str, Any],
    latent_bundle: dict[str, Any],
) -> dict[str, str]:
    payload = build_phase3_compatibility_payload(
        phase1_dir=phase1_dir,
        predictive_rows=predictive_rows,
        context_rows=context_rows,
        retained_catalog=retained_catalog,
        multiscale_blankets=multiscale_blankets,
        latent_blankets=latent_blankets,
        multiscale_bundle=multiscale_bundle,
        latent_bundle=latent_bundle,
    )
    core_artifact = save_tensor_artifact(
        array=np.asarray(payload["core_feature_tensor_array"], dtype=np.float32),
        axis_names=["province", "month", "feature"],
        artifact_dir=phase2_dir,
        stem="core_feature_tensor",
        backend="numpy",
        device="cpu",
        notes=["phase2_phase3_compatibility_blanket_tensor"],
        save_pt=False,
    )
    hidden_artifact = save_tensor_artifact(
        array=np.asarray(payload["hidden_driver_feature_tensor_array"], dtype=np.float32),
        axis_names=["province", "month", "feature"],
        artifact_dir=phase2_dir,
        stem="hidden_driver_feature_tensor",
        backend="numpy",
        device="cpu",
        notes=["phase2_phase3_hidden_driver_tensor"],
        save_pt=False,
    )
    multiscale_artifact = save_tensor_artifact(
        array=np.asarray(payload["multiscale_support_feature_tensor_array"], dtype=np.float32),
        axis_names=["province", "month", "feature"],
        artifact_dir=phase2_dir,
        stem="multiscale_support_feature_tensor",
        backend="numpy",
        device="cpu",
        notes=["phase2_phase3_multiscale_support_tensor"],
        save_pt=False,
    )

    write_json(phase2_dir / "candidate_profiles.json", payload["candidate_profiles"])
    write_json(phase2_dir / "curated_candidate_blocks.json", payload["curated_candidate_blocks"])
    write_json(phase2_dir / "markov_blanket.json", payload["markov_blanket"])
    write_json(phase2_dir / "edge_scores.json", payload["edge_scores"])
    write_json(phase2_dir / "hidden_driver_edge_scores.json", payload["hidden_driver_edge_scores"])
    write_json(phase2_dir / "multiscale_edge_scores.json", payload["multiscale_edge_scores"])
    write_json(phase2_dir / "ranked_linkages.json", payload["ranked_linkages"])
    write_json(phase2_dir / "phase3_target_blankets.json", payload["phase3_target_blankets"])
    write_json(phase2_dir / "phase3_eligibility_surfaces.json", payload["eligibility_surfaces"])
    write_json(phase2_dir / "retained_predictive_factor_set.json", payload["retained_predictive_factor_set"])
    write_json(phase2_dir / "retained_context_factor_set.json", payload["retained_context_factor_set"])
    write_json(phase2_dir / "promoted_factor_set.json", payload["promoted_factor_set"])
    write_json(phase2_dir / "supporting_factor_set.json", payload["supporting_factor_set"])
    write_json(phase2_dir / "retained_mesoscopic_factor_catalog.json", payload["retained_mesoscopic_factor_catalog"])
    frozen_payload = {
        "candidate_profiles": payload["candidate_profiles"],
        "curated_candidate_blocks": payload["curated_candidate_blocks"],
        "markov_blanket": payload["markov_blanket"],
        "edge_scores": payload["edge_scores"],
        "hidden_driver_edge_scores": payload["hidden_driver_edge_scores"],
        "multiscale_edge_scores": payload["multiscale_edge_scores"],
        "ranked_linkages": payload["ranked_linkages"],
        "phase3_target_blankets": payload["phase3_target_blankets"],
        "eligibility_surfaces": payload["eligibility_surfaces"],
        "retained_predictive_factor_set": payload["retained_predictive_factor_set"],
        "retained_context_factor_set": payload["retained_context_factor_set"],
        "promoted_factor_set": payload["promoted_factor_set"],
        "supporting_factor_set": payload["supporting_factor_set"],
        "retained_mesoscopic_factor_catalog": payload["retained_mesoscopic_factor_catalog"],
        "core_feature_tensor_path": str(core_artifact["value_path"]),
        "direct_feature_tensor_path": str(core_artifact["value_path"]),
        "hidden_driver_feature_tensor_path": str(hidden_artifact["value_path"]),
        "multiscale_support_feature_tensor_path": str(multiscale_artifact["value_path"]),
    }
    write_json(phase2_dir / "phase3_compatibility_payload.json", frozen_payload)

    return {
        "candidate_profiles": str(phase2_dir / "candidate_profiles.json"),
        "curated_candidate_blocks": str(phase2_dir / "curated_candidate_blocks.json"),
        "markov_blanket": str(phase2_dir / "markov_blanket.json"),
        "core_feature_tensor": str(core_artifact["value_path"]),
        "edge_scores": str(phase2_dir / "edge_scores.json"),
        "hidden_driver_edge_scores": str(phase2_dir / "hidden_driver_edge_scores.json"),
        "multiscale_edge_scores": str(phase2_dir / "multiscale_edge_scores.json"),
        "ranked_linkages": str(phase2_dir / "ranked_linkages.json"),
        "phase3_target_blankets": str(phase2_dir / "phase3_target_blankets.json"),
        "phase3_eligibility_surfaces": str(phase2_dir / "phase3_eligibility_surfaces.json"),
        "retained_predictive_factor_set": str(phase2_dir / "retained_predictive_factor_set.json"),
        "retained_context_factor_set": str(phase2_dir / "retained_context_factor_set.json"),
        "promoted_factor_set": str(phase2_dir / "promoted_factor_set.json"),
        "supporting_factor_set": str(phase2_dir / "supporting_factor_set.json"),
        "retained_mesoscopic_factor_catalog": str(phase2_dir / "retained_mesoscopic_factor_catalog.json"),
        "phase3_compatibility_payload": str(phase2_dir / "phase3_compatibility_payload.json"),
        "hidden_driver_feature_tensor": str(hidden_artifact["value_path"]),
        "multiscale_support_feature_tensor": str(multiscale_artifact["value_path"]),
    }


__all__ = [
    "build_phase3_compatibility_payload",
    "build_phase3_compatibility_artifacts",
    "select_retained_factor_rows",
]
