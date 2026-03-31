from __future__ import annotations

from collections import defaultdict
from typing import Any

from epigraph_ph.phase15 import PHASE15_PROFILE_ID
from epigraph_ph.runtime import RunContext, ensure_dir, load_tensor_artifact, read_json, write_ground_truth_package, write_json


def transition_hook_rank(hooks: list[str]) -> list[str]:
    order = [
        "diagnosis_transitions",
        "linkage_transitions",
        "retention_attrition_transitions",
        "suppression_transitions",
        "subgroup_allocation_priors",
    ]
    return [hook for hook in order if hook in hooks]


def augment_phase2_for_rescue_v2(*, run_id: str, plugin_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    from epigraph_ph.phase2.pipeline import _phase2_required, _phase2_required_section

    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase2_dir = ensure_dir(ctx.run_dir / "phase2")
    factor_catalog = read_json(ctx.run_dir / "phase15" / "mesoscopic_factor_catalog.json", default=[])
    factor_pool = read_json(ctx.run_dir / "phase15" / "factor_promotion_pool.json", default=[])
    factor_tensor = load_tensor_artifact(ctx.run_dir / "phase15" / "mesoscopic_factor_tensor.npz")
    stability_report = read_json(ctx.run_dir / "phase15" / "factor_stability_report.json", default=[])
    if not factor_catalog or factor_tensor.size == 0:
        empty = {"rows": [], "summary": {"factor_count": 0}}
        write_json(phase2_dir / "survival_tournament_plan.json", empty)
        write_json(phase2_dir / "survival_tournament_results.json", empty)
        write_json(phase2_dir / "retained_predictive_factor_set.json", [])
        write_json(phase2_dir / "retained_context_factor_set.json", [])
        write_json(phase2_dir / "factor_tournament_plan.json", empty)
        write_json(phase2_dir / "factor_tournament_results.json", empty)
        write_json(phase2_dir / "promoted_factor_set.json", [])
        write_json(phase2_dir / "supporting_factor_set.json", [])
        write_json(
            phase2_dir / "survival_admission.json",
            {
                "status": "none_admitted",
                "reason": "no_factor_catalog_or_tensor",
                "top_near_misses": [],
            },
        )
        write_json(phase2_dir / "promotion_admission.json", {"status": "none_admitted", "reason": "no_factor_catalog_or_tensor", "top_near_misses": []})
        write_json(phase2_dir / "survival_budget_report.json", {"retained_predictive_count": 0, "retained_context_count": 0})
        write_json(phase2_dir / "promotion_budget_report.json", {"retained_predictive_count": 0, "retained_context_count": 0})
        write_json(phase2_dir / "survival_gate_report.json", {"rows": stability_report})
        write_json(phase2_dir / "stability_gate_report.json", {"rows": stability_report})
        write_json(phase2_dir / "factor_diagnostics.json", {"rows": []})
        manifest["profile_id"] = PHASE15_PROFILE_ID
        manifest.setdefault("artifact_paths", {})
        manifest["artifact_paths"].update(
            {
                "survival_tournament_plan": str(phase2_dir / "survival_tournament_plan.json"),
                "survival_tournament_results": str(phase2_dir / "survival_tournament_results.json"),
                "retained_predictive_factor_set": str(phase2_dir / "retained_predictive_factor_set.json"),
                "retained_context_factor_set": str(phase2_dir / "retained_context_factor_set.json"),
                "survival_admission": str(phase2_dir / "survival_admission.json"),
                "survival_budget_report": str(phase2_dir / "survival_budget_report.json"),
                "survival_gate_report": str(phase2_dir / "survival_gate_report.json"),
                "factor_tournament_plan": str(phase2_dir / "factor_tournament_plan.json"),
                "factor_tournament_results": str(phase2_dir / "factor_tournament_results.json"),
                "promoted_factor_set": str(phase2_dir / "promoted_factor_set.json"),
                "supporting_factor_set": str(phase2_dir / "supporting_factor_set.json"),
                "promotion_admission": str(phase2_dir / "promotion_admission.json"),
                "promotion_budget_report": str(phase2_dir / "promotion_budget_report.json"),
                "stability_gate_report": str(phase2_dir / "stability_gate_report.json"),
                "factor_diagnostics": str(phase2_dir / "factor_diagnostics.json"),
            }
        )
        write_json(phase2_dir / "phase2_manifest.json", manifest)
        return manifest

    pool_by_id = {row["factor_id"]: row for row in factor_pool}
    catalog_by_id = {row["factor_id"]: row for row in factor_catalog}
    shortlist_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics = []
    factor_diag_cfg = _phase2_required_section("factor_diagnostic_weights")
    for factor in factor_catalog:
        pool = pool_by_id.get(factor["factor_id"], {})
        score = (
            float(factor_diag_cfg["survival_score"]) * float(pool.get("survival_score", 0.0))
            + float(factor_diag_cfg["holdout_mae_improvement"]) * float(pool.get("holdout_mae_improvement", 0.0))
            + float(factor_diag_cfg["holdout_smape_improvement"]) * float(pool.get("holdout_smape_improvement", 0.0))
            + float(factor_diag_cfg["calibration_score"]) * float(pool.get("calibration_score", 0.0))
            + float(factor_diag_cfg["stability_score"]) * float(pool.get("stability_score", 0.0))
            + float(factor_diag_cfg["predictive_gain"]) * float(pool.get("predictive_gain", 0.0))
            + float(factor_diag_cfg["subnational_anomaly_gain"]) * float(pool.get("subnational_anomaly_gain", 0.0))
            + float(factor_diag_cfg["region_contrast_score"]) * float(pool.get("region_contrast_score", 0.0))
            - float(factor_diag_cfg["sparsity_penalty"]) * float(pool.get("sparsity_penalty", 0.0))
            - float(factor_diag_cfg["resampling_stability_penalty"]) * float(pool.get("resampling_stability_penalty", 0.0))
        )
        diag_row = {
            "factor_id": factor["factor_id"],
            "factor_name": factor["factor_name"],
            "block_name": factor["block_name"],
            "diagnostic_score": round(float(score), 6),
            "survival_class": pool.get("survival_class", pool.get("promotion_class", "discarded")),
            "hard_checks_passed": bool(pool.get("hard_checks_passed")),
            "survives_holdout": bool(pool.get("survives_holdout")),
            "network_feature_family": pool.get("network_feature_family", factor.get("network_feature_family", "")),
            "best_target": pool.get("best_target", ""),
            "best_subnational_target": pool.get("best_subnational_target", ""),
            "subnational_anomaly_gain": round(float(pool.get("subnational_anomaly_gain", 0.0)), 6),
            "region_contrast_score": round(float(pool.get("region_contrast_score", 0.0)), 6),
            "holdout_mae_improvement": round(float(pool.get("holdout_mae_improvement", 0.0)), 6),
            "holdout_smape_improvement": round(float(pool.get("holdout_smape_improvement", 0.0)), 6),
            "calibration_score": round(float(pool.get("calibration_score", 0.0)), 6),
            "sparsity_penalty": round(float(pool.get("sparsity_penalty", 0.0)), 6),
            "resampling_stability_penalty": round(float(pool.get("resampling_stability_penalty", 0.0)), 6),
            "transition_hooks": transition_hook_rank(list(factor.get("transition_hooks", []))),
        }
        diagnostics.append(diag_row)
        shortlist_by_block[factor["block_name"]].append(diag_row)

    for block_name, rows in shortlist_by_block.items():
        class_rank = {"survivor_primary": 2, "survivor_secondary": 1, "reserve": 0, "discarded": -1}
        rows.sort(
            key=lambda item: (
                bool(item["hard_checks_passed"]),
                bool(item["survives_holdout"]),
                class_rank.get(str(item["survival_class"]), -1),
                float(item["diagnostic_score"]),
            ),
            reverse=True,
        )
        shortlist_by_block[block_name] = rows[: int(_phase2_required("shortlist_per_block"))]

    tournament_plan = []
    tournament_results = []
    budget_cfg = _phase2_required_section("budgets")
    main_budget = int(budget_cfg["main"])
    support_budget = int(budget_cfg["support"])
    network_main_budget = int(budget_cfg["network_main"])
    promoted_main = []
    promoted_support = []
    network_main_count = 0
    for block_name, rows in shortlist_by_block.items():
        tournament_plan.append({"block_name": block_name, "shortlist_factor_ids": [row["factor_id"] for row in rows], "shortlist_count": len(rows)})
        for rank, row in enumerate(rows):
            pool = pool_by_id.get(row["factor_id"], {})
            factor = catalog_by_id[row["factor_id"]]
            class_decision = "exploratory"
            if (
                row["survival_class"] == "survivor_primary"
                and row["hard_checks_passed"]
                and main_budget > 0
                and (network_main_count < network_main_budget or not row["network_feature_family"])
            ):
                class_decision = "retained_predictive"
                main_budget -= 1
                if row["network_feature_family"]:
                    network_main_count += 1
            elif row["survival_class"] in {"survivor_primary", "survivor_secondary"} and row["hard_checks_passed"] and support_budget > 0:
                class_decision = "retained_context"
                support_budget -= 1
            factor_row = {
                "factor_id": row["factor_id"],
                "factor_name": row["factor_name"],
                "block_name": block_name,
                "promotion_class": class_decision,
                "factor_class": factor.get("factor_class", "mesoscopic_factor"),
                "transition_hooks": row["transition_hooks"],
                "best_target": row["best_target"],
                "diagnostic_score": row["diagnostic_score"],
                "stability_score": pool.get("stability_score", 0.0),
                "predictive_gain": pool.get("predictive_gain", 0.0),
                "survival_score": pool.get("survival_score", 0.0),
                "holdout_mae_improvement": pool.get("holdout_mae_improvement", 0.0),
                "holdout_smape_improvement": pool.get("holdout_smape_improvement", 0.0),
                "calibration_score": pool.get("calibration_score", 0.0),
                "sparsity_penalty": pool.get("sparsity_penalty", 0.0),
                "resampling_stability_penalty": pool.get("resampling_stability_penalty", 0.0),
                "network_feature_family": row["network_feature_family"],
                "tournament_rank": rank + 1,
            }
            tournament_results.append(factor_row)
            if class_decision == "retained_predictive":
                promoted_main.append(factor_row)
            elif class_decision == "retained_context":
                promoted_support.append(factor_row)

    budget_report = {
        "retained_predictive_count": len(promoted_main),
        "retained_context_count": len(promoted_support),
        "remaining_retained_predictive_budget": main_budget,
        "remaining_retained_context_budget": support_budget,
        "network_retained_predictive_count": network_main_count,
        "max_retained_predictive": int(budget_cfg["main"]),
        "max_retained_context": int(budget_cfg["support"]),
        "max_network_retained_predictive": int(budget_cfg["network_main"]),
    }
    stability_gate_report = {"rows": stability_report}
    top_near_misses = [
        {
            "factor_id": row["factor_id"],
            "factor_name": row["factor_name"],
            "block_name": row["block_name"],
            "diagnostic_score": row["diagnostic_score"],
            "subnational_anomaly_gain": row.get("subnational_anomaly_gain", 0.0),
            "region_contrast_score": row.get("region_contrast_score", 0.0),
            "reason": "failed_survival_competition",
        }
        for row in sorted(diagnostics, key=lambda item: float(item["diagnostic_score"]), reverse=True)[:5]
        if row["factor_id"] not in {item["factor_id"] for item in promoted_main}
    ]
    promotion_admission = (
        {
            "status": "admitted_retained_predictive",
            "retained_predictive_factor_ids": [row["factor_id"] for row in promoted_main],
            "retained_context_factor_ids": [row["factor_id"] for row in promoted_support],
            "top_near_misses": top_near_misses,
        }
        if promoted_main
        else {
            "status": "none_admitted",
            "reason": "no_factor_survived_retained_predictive_competition",
            "retained_context_factor_ids": [row["factor_id"] for row in promoted_support],
            "top_near_misses": top_near_misses,
        }
    )

    write_json(phase2_dir / "survival_tournament_plan.json", tournament_plan)
    write_json(phase2_dir / "survival_tournament_results.json", tournament_results)
    write_json(phase2_dir / "retained_predictive_factor_set.json", promoted_main)
    write_json(phase2_dir / "retained_context_factor_set.json", promoted_support)
    write_json(phase2_dir / "survival_admission.json", promotion_admission)
    write_json(phase2_dir / "survival_budget_report.json", budget_report)
    write_json(phase2_dir / "survival_gate_report.json", stability_gate_report)
    write_json(phase2_dir / "factor_tournament_plan.json", tournament_plan)
    write_json(phase2_dir / "factor_tournament_results.json", tournament_results)
    write_json(phase2_dir / "promoted_factor_set.json", promoted_main)
    write_json(phase2_dir / "supporting_factor_set.json", promoted_support)
    write_json(phase2_dir / "promotion_admission.json", promotion_admission)
    write_json(phase2_dir / "promotion_budget_report.json", budget_report)
    write_json(phase2_dir / "stability_gate_report.json", stability_gate_report)
    write_json(phase2_dir / "factor_diagnostics.json", diagnostics)

    manifest = dict(manifest)
    manifest["profile_id"] = PHASE15_PROFILE_ID
    manifest.setdefault("artifact_paths", {})
    manifest["artifact_paths"].update(
        {
            "survival_tournament_plan": str(phase2_dir / "survival_tournament_plan.json"),
            "survival_tournament_results": str(phase2_dir / "survival_tournament_results.json"),
            "retained_predictive_factor_set": str(phase2_dir / "retained_predictive_factor_set.json"),
            "retained_context_factor_set": str(phase2_dir / "retained_context_factor_set.json"),
            "survival_admission": str(phase2_dir / "survival_admission.json"),
            "survival_budget_report": str(phase2_dir / "survival_budget_report.json"),
            "survival_gate_report": str(phase2_dir / "survival_gate_report.json"),
            "factor_tournament_plan": str(phase2_dir / "factor_tournament_plan.json"),
            "factor_tournament_results": str(phase2_dir / "factor_tournament_results.json"),
            "promoted_factor_set": str(phase2_dir / "promoted_factor_set.json"),
            "supporting_factor_set": str(phase2_dir / "supporting_factor_set.json"),
            "promotion_admission": str(phase2_dir / "promotion_admission.json"),
            "promotion_budget_report": str(phase2_dir / "promotion_budget_report.json"),
            "stability_gate_report": str(phase2_dir / "stability_gate_report.json"),
            "factor_diagnostics": str(phase2_dir / "factor_diagnostics.json"),
        }
    )
    manifest.setdefault("notes", [])
    manifest["notes"] = list(manifest["notes"]) + ["phase2_rescue_v2:mesoscopic_factor_tournaments"]
    truth_paths = write_ground_truth_package(
        phase_dir=phase2_dir,
        phase_name="phase2",
        profile_id=PHASE15_PROFILE_ID,
        checks=[
            {"name": "tournament_results_present", "passed": bool(tournament_results)},
            {"name": "main_budget_respected", "passed": len(promoted_main) <= 8},
            {"name": "support_budget_respected", "passed": len(promoted_support) <= 12},
            {"name": "promotion_admission_present", "passed": True},
            {
                "name": "no_exploratory_promotions",
                "passed": all(pool_by_id.get(row["factor_id"], {}).get("promotion_class") in {"survivor_primary", "survivor_secondary"} for row in promoted_main),
            },
            {"name": "network_main_budget_respected", "passed": network_main_count <= 3},
        ],
        truth_sources=["benchmark_truth", "null_test", "synthetic_truth"],
        stage_manifest_path=str(phase2_dir / "phase2_manifest.json"),
        summary={
            "retained_predictive_count": len(promoted_main),
            "retained_context_count": len(promoted_support),
            "diagnostic_factor_count": len(diagnostics),
            "promotion_admission_status": promotion_admission["status"],
        },
    )
    manifest["artifact_paths"].update(truth_paths)
    write_json(phase2_dir / "phase2_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase2_build",
        [
            phase2_dir / "survival_tournament_plan.json",
            phase2_dir / "survival_tournament_results.json",
            phase2_dir / "retained_predictive_factor_set.json",
            phase2_dir / "retained_context_factor_set.json",
            phase2_dir / "survival_admission.json",
            phase2_dir / "survival_budget_report.json",
            phase2_dir / "survival_gate_report.json",
            phase2_dir / "factor_tournament_plan.json",
            phase2_dir / "factor_tournament_results.json",
            phase2_dir / "promoted_factor_set.json",
            phase2_dir / "supporting_factor_set.json",
            phase2_dir / "promotion_admission.json",
            phase2_dir / "promotion_budget_report.json",
            phase2_dir / "stability_gate_report.json",
            phase2_dir / "factor_diagnostics.json",
            phase2_dir / "phase2_manifest.json",
        ],
    )
    return manifest
