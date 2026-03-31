from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from epigraph_ph.runtime import RunContext, ensure_dir, read_json, utc_now_iso, write_json


def _stable_signature(parts: list[str]) -> str:
    joined = "||".join(parts)
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _factor_signature(row: dict[str, Any]) -> tuple[str, ...]:
    hooks = sorted(str(item) for item in (row.get("transition_hooks") or []))
    members = sorted(str(item) for item in (row.get("member_canonical_names") or []))
    return (
        str(row.get("factor_class") or ""),
        str(row.get("block_name") or ""),
        str(row.get("factor_name") or ""),
        str(row.get("interpretability_label") or ""),
        str(row.get("best_target") or ""),
        str(row.get("network_feature_family") or ""),
        "|".join(hooks),
        "|".join(members),
    )


def _block_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def merge_phase2_shard_factor_summaries(
    *,
    run_id: str,
    plugin_id: str,
    source_run_ids: list[str],
    bridge_edge_budget_per_block_pair: int = 4,
) -> dict[str, Any]:
    if not source_run_ids:
        raise ValueError("phase2 merge-shard-summaries requires at least one source run id")
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase2_dir = ensure_dir(ctx.run_dir / "phase2")

    merged_rows_by_sig: dict[tuple[str, ...], dict[str, Any]] = {}
    shard_factor_maps: dict[str, dict[str, tuple[str, ...]]] = {}
    cross_block_counter: Counter[tuple[str, ...]] = Counter()
    phase3_member_counter: Counter[str] = Counter()

    for shard_run_id in source_run_ids:
        shard_ctx = RunContext.create(run_id=shard_run_id, plugin_id=plugin_id)
        shard_phase2_dir = shard_ctx.run_dir / "phase2"
        retained_rows = list(read_json(shard_phase2_dir / "retained_mesoscopic_factor_catalog.json", default={}).get("rows", []))
        blanket = read_json(shard_phase2_dir / "phase3_target_blankets.json", default={})
        blanket_ids = {str(item) for item in (blanket.get("blanket_factor_ids") or [])}
        factor_id_to_sig: dict[str, tuple[str, ...]] = {}
        shard_factor_maps[shard_run_id] = factor_id_to_sig

        for row in retained_rows:
            factor_id = str(row.get("factor_id") or "")
            signature = _factor_signature(row)
            factor_id_to_sig[factor_id] = signature
            merged_id = f"merged_factor_{_stable_signature(list(signature))}"
            merged = merged_rows_by_sig.setdefault(
                signature,
                {
                    "merged_factor_id": merged_id,
                    "factor_class": str(row.get("factor_class") or ""),
                    "block_name": str(row.get("block_name") or ""),
                    "factor_name": str(row.get("factor_name") or ""),
                    "interpretability_label": str(row.get("interpretability_label") or ""),
                    "best_target": str(row.get("best_target") or ""),
                    "network_feature_family": str(row.get("network_feature_family") or ""),
                    "transition_hooks": sorted(str(item) for item in (row.get("transition_hooks") or [])),
                    "member_canonical_names": sorted(str(item) for item in (row.get("member_canonical_names") or [])),
                    "promotion_class_counts": Counter(),
                    "shard_run_ids": [],
                    "shard_count": 0,
                    "phase3_target_relevant_shard_count": 0,
                    "phase3_blanket_shard_count": 0,
                    "predictive_gain_values": [],
                    "stability_score_values": [],
                    "region_contrast_score_values": [],
                    "subnational_anomaly_gain_values": [],
                    "phase3_target_score_values": [],
                },
            )
            merged["promotion_class_counts"][str(row.get("promotion_class") or "")] += 1
            if shard_run_id not in merged["shard_run_ids"]:
                merged["shard_run_ids"].append(shard_run_id)
                merged["shard_count"] += 1
            if bool(row.get("phase3_target_relevant")):
                merged["phase3_target_relevant_shard_count"] += 1
            if factor_id in blanket_ids:
                merged["phase3_blanket_shard_count"] += 1
            merged["predictive_gain_values"].append(float(row.get("predictive_gain") or 0.0))
            merged["stability_score_values"].append(float(row.get("stability_score") or 0.0))
            merged["region_contrast_score_values"].append(float(row.get("region_contrast_score") or 0.0))
            merged["subnational_anomaly_gain_values"].append(float(row.get("subnational_anomaly_gain") or 0.0))
            merged["phase3_target_score_values"].append(float(row.get("phase3_target_score") or 0.0))

        blanket_signatures = sorted({factor_id_to_sig[item] for item in blanket_ids if item in factor_id_to_sig}, key=lambda item: merged_rows_by_sig[item]["merged_factor_id"])
        blanket_rows = [merged_rows_by_sig[item] for item in blanket_signatures]
        for row in blanket_rows:
            for member in row.get("member_canonical_names") or []:
                phase3_member_counter[str(member)] += 1
        for left_idx, left_row in enumerate(blanket_rows):
            for right_row in blanket_rows[left_idx + 1 :]:
                if str(left_row.get("block_name") or "") == str(right_row.get("block_name") or ""):
                    continue
                pair = _block_pair(str(left_row.get("block_name") or ""), str(right_row.get("block_name") or ""))
                edge_key = (
                    left_row["merged_factor_id"],
                    right_row["merged_factor_id"],
                    pair[0],
                    pair[1],
                )
                cross_block_counter[edge_key] += 1

    merged_rows: list[dict[str, Any]] = []
    for row in merged_rows_by_sig.values():
        merged_rows.append(
            {
                "merged_factor_id": row["merged_factor_id"],
                "factor_class": row["factor_class"],
                "block_name": row["block_name"],
                "factor_name": row["factor_name"],
                "interpretability_label": row["interpretability_label"],
                "best_target": row["best_target"],
                "network_feature_family": row["network_feature_family"],
                "transition_hooks": row["transition_hooks"],
                "member_canonical_names": row["member_canonical_names"],
                "shard_count": int(row["shard_count"]),
                "shard_run_ids": sorted(row["shard_run_ids"]),
                "phase3_target_relevant_shard_count": int(row["phase3_target_relevant_shard_count"]),
                "phase3_blanket_shard_count": int(row["phase3_blanket_shard_count"]),
                "promotion_class_counts": dict(row["promotion_class_counts"]),
                "mean_predictive_gain": round(sum(row["predictive_gain_values"]) / max(1, len(row["predictive_gain_values"])), 6),
                "max_predictive_gain": round(max(row["predictive_gain_values"] or [0.0]), 6),
                "mean_stability_score": round(sum(row["stability_score_values"]) / max(1, len(row["stability_score_values"])), 6),
                "max_stability_score": round(max(row["stability_score_values"] or [0.0]), 6),
                "mean_region_contrast_score": round(sum(row["region_contrast_score_values"]) / max(1, len(row["region_contrast_score_values"])), 6),
                "max_region_contrast_score": round(max(row["region_contrast_score_values"] or [0.0]), 6),
                "mean_subnational_anomaly_gain": round(sum(row["subnational_anomaly_gain_values"]) / max(1, len(row["subnational_anomaly_gain_values"])), 6),
                "max_subnational_anomaly_gain": round(max(row["subnational_anomaly_gain_values"] or [0.0]), 6),
                "mean_phase3_target_score": round(sum(row["phase3_target_score_values"]) / max(1, len(row["phase3_target_score_values"])), 6),
                "max_phase3_target_score": round(max(row["phase3_target_score_values"] or [0.0]), 6),
            }
        )
    merged_rows.sort(
        key=lambda row: (
            int(row["phase3_blanket_shard_count"]),
            int(row["shard_count"]),
            float(row["max_stability_score"]),
            float(row["max_predictive_gain"]),
            str(row["merged_factor_id"]),
        ),
        reverse=True,
    )

    block_summary_rows = []
    block_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        block_groups[str(row.get("block_name") or "mixed")].append(row)
    for block_name, rows in sorted(block_groups.items()):
        block_summary_rows.append(
            {
                "block_name": block_name,
                "merged_factor_count": len(rows),
                "phase3_blanket_factor_count": sum(1 for row in rows if int(row.get("phase3_blanket_shard_count", 0)) > 0),
                "top_merged_factor_ids": [row["merged_factor_id"] for row in rows[:12]],
                "top_factor_names": [row["factor_name"] for row in rows[:12]],
            }
        )

    raw_edge_rows = []
    for (left_id, right_id, left_block, right_block), support_count in cross_block_counter.items():
        left_row = next(row for row in merged_rows if row["merged_factor_id"] == left_id)
        right_row = next(row for row in merged_rows if row["merged_factor_id"] == right_id)
        score = (
            float(support_count)
            + 0.5 * min(float(left_row.get("mean_phase3_target_score") or 0.0), float(right_row.get("mean_phase3_target_score") or 0.0))
            + 0.5 * min(float(left_row.get("mean_stability_score") or 0.0), float(right_row.get("mean_stability_score") or 0.0))
        )
        raw_edge_rows.append(
            {
                "source_merged_factor_id": left_id,
                "target_merged_factor_id": right_id,
                "source_block": left_block,
                "target_block": right_block,
                "support_count": int(support_count),
                "score": round(score, 6),
            }
        )
    raw_edge_rows.sort(key=lambda row: (int(row["support_count"]), float(row["score"])), reverse=True)

    pruned_edge_rows = []
    pair_budgets: Counter[tuple[str, str]] = Counter()
    for row in raw_edge_rows:
        pair = _block_pair(str(row["source_block"]), str(row["target_block"]))
        if pair_budgets[pair] >= bridge_edge_budget_per_block_pair:
            continue
        pair_budgets[pair] += 1
        pruned_edge_rows.append(dict(row))

    merged_target_rows = [
        row
        for row in merged_rows
        if int(row.get("phase3_blanket_shard_count", 0)) > 0 or int(row.get("phase3_target_relevant_shard_count", 0)) > 0
    ]
    phase3_target_blanket_summary = {
        "merged_factor_ids": [row["merged_factor_id"] for row in merged_target_rows],
        "member_canonical_names": [name for name, _ in phase3_member_counter.most_common(40)],
        "factor_count": len(merged_target_rows),
        "member_canonical_count": len(phase3_member_counter),
    }

    bridge_summary = {
        "status": "completed",
        "source_run_ids": list(source_run_ids),
        "raw_edge_count": len(raw_edge_rows),
        "retained_edge_count": len(pruned_edge_rows),
        "bridge_edge_budget_per_block_pair": int(bridge_edge_budget_per_block_pair),
        "edges": pruned_edge_rows,
    }
    merge_manifest = {
        "run_id": run_id,
        "plugin_id": plugin_id,
        "generated_at": utc_now_iso(),
        "source_run_ids": list(source_run_ids),
        "merged_factor_count": len(merged_rows),
        "block_summary_count": len(block_summary_rows),
        "bridge_edge_count": len(pruned_edge_rows),
        "phase3_target_factor_count": len(merged_target_rows),
    }

    write_json(phase2_dir / "merged_retained_factor_summary.json", {"rows": merged_rows})
    write_json(phase2_dir / "merged_block_factor_summary.json", {"rows": block_summary_rows})
    write_json(phase2_dir / "merged_bridge_summary.json", bridge_summary)
    write_json(phase2_dir / "merged_phase3_target_blanket_summary.json", phase3_target_blanket_summary)
    write_json(phase2_dir / "phase2_merge_manifest.json", merge_manifest)
    ctx.record_stage_outputs(
        "phase2_merge_shard_summaries",
        [
            phase2_dir / "merged_retained_factor_summary.json",
            phase2_dir / "merged_block_factor_summary.json",
            phase2_dir / "merged_bridge_summary.json",
            phase2_dir / "merged_phase3_target_blanket_summary.json",
            phase2_dir / "phase2_merge_manifest.json",
        ],
    )
    return {
        "artifact_paths": {
            "merged_retained_factor_summary": str(phase2_dir / "merged_retained_factor_summary.json"),
            "merged_block_factor_summary": str(phase2_dir / "merged_block_factor_summary.json"),
            "merged_bridge_summary": str(phase2_dir / "merged_bridge_summary.json"),
            "merged_phase3_target_blanket_summary": str(phase2_dir / "merged_phase3_target_blanket_summary.json"),
            "phase2_merge_manifest": str(phase2_dir / "phase2_merge_manifest.json"),
        },
        "summary": merge_manifest,
    }
