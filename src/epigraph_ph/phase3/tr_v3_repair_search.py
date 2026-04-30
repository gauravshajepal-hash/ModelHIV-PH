from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.runtime import ensure_dir, write_json


CONTRACT_CHOICES: tuple[str, ...] = ("exact_only", "dense_train_observed_score")


@dataclass(slots=True)
class RepairSearchCandidate:
    base_experiment_id: str
    search_id: str
    spec: suite.ExperimentSpec


def _clone_spec(
    base: suite.ExperimentSpec,
    *,
    experiment_id: str,
    description: str | None = None,
    transition_ridge_multipliers: dict[str, float] | None = None,
    repair_params: dict[str, Any] | None = None,
) -> suite.ExperimentSpec:
    payload = asdict(base)
    payload["experiment_id"] = experiment_id
    if description is not None:
        payload["description"] = description
    if transition_ridge_multipliers is not None:
        payload["transition_ridge_multipliers"] = {
            **dict(base.transition_ridge_multipliers),
            **{str(name): float(value) for name, value in transition_ridge_multipliers.items()},
        }
    if repair_params is not None:
        payload["repair_params"] = {**dict(base.repair_params), **repair_params}
    return suite.ExperimentSpec(**payload)


def build_repair_search_specs() -> list[RepairSearchCandidate]:
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    candidates: list[RepairSearchCandidate] = []

    r1 = spec_map["EXP-R1"]
    for ridge_value in (1.0, 2.0, 4.0, 8.0):
        ridge_label = str(int(ridge_value))
        spec = _clone_spec(
            r1,
            experiment_id=f"SEARCH-R1-d2a-ridge{ridge_label}",
            description=(
                "Strict-support bounded-drift repair with "
                f"D_to_A ridge multiplier {ridge_label}."
            ),
            transition_ridge_multipliers={"D_to_A": float(ridge_value)},
        )
        candidates.append(RepairSearchCandidate("EXP-R1", spec.experiment_id, spec))

    for base_experiment_id in ("EXP-R6", "EXP-R7"):
        base = spec_map[base_experiment_id]
        family_label = base_experiment_id.replace("EXP-", "")
        for min_count in (3, 4, 5):
            for min_fraction in (0.25, 0.35, 0.45):
                fraction_label = int(round(min_fraction * 100.0))
                spec = _clone_spec(
                    base,
                    experiment_id=f"SEARCH-{family_label}-c{min_count}-f{fraction_label:02d}",
                    description=(
                        f"{family_label} with D_to_A support thresholds "
                        f"min_count={min_count}, min_fraction={min_fraction:.2f}."
                    ),
                    repair_params={
                        "d_to_a_min_count": int(min_count),
                        "d_to_a_min_fraction": float(min_fraction),
                    },
                )
                candidates.append(RepairSearchCandidate(base_experiment_id, spec.experiment_id, spec))

    r9 = spec_map["EXP-R9"]
    for min_count in (3, 4):
        for min_fraction in (0.25, 0.35):
            for model_weight in (0.5, 0.75, 1.0):
                for use_blend in (False, True):
                    fraction_label = int(round(min_fraction * 100.0))
                    weight_label = int(round(model_weight * 100.0))
                    mode_label = "blend" if use_blend else "repair"
                    spec = _clone_spec(
                        r9,
                        experiment_id=f"SEARCH-R9-c{min_count}-f{fraction_label:02d}-w{weight_label:02d}-{mode_label}",
                        description=(
                            "Net ART-delta D_to_A repair with "
                            f"min_count={min_count}, min_fraction={min_fraction:.2f}, "
                            f"delta_weight={model_weight:.2f}, mode={mode_label}."
                        ),
                        repair_params={
                            "d_to_a_min_count": int(min_count),
                            "d_to_a_min_fraction": float(min_fraction),
                            "d_to_a_use_blend": bool(use_blend),
                            "art_delta_model_weight": float(model_weight),
                        },
                    )
                    candidates.append(RepairSearchCandidate("EXP-R9", spec.experiment_id, spec))

    r10 = spec_map["EXP-R10"]
    for stock_weight in (0.25, 0.5, 0.75, 1.0):
        for flow_weight in (0.5, 1.0):
            stock_label = int(round(stock_weight * 100.0))
            flow_label = int(round(flow_weight * 100.0))
            spec = _clone_spec(
                r10,
                experiment_id=f"SEARCH-R10-s{stock_label:02d}-f{flow_label:02d}",
                description=(
                    "Observation-first stock-flow repair with "
                    f"stock_weight={stock_weight:.2f}, flow_weight={flow_weight:.2f}."
                ),
                repair_params={
                    "diagnosed_weight": float(stock_weight),
                    "art_weight": float(stock_weight),
                    "flow_weight": float(flow_weight),
                },
            )
            candidates.append(RepairSearchCandidate("EXP-R10", spec.experiment_id, spec))

    r10_exact = spec_map["EXP-R10-EXACT-CHAMPION"]
    for art_weight in (0.85, 1.0):
        for flow_weight in (0.25, 0.5, 0.75, 1.0):
            for flow_model in ("level", "delta"):
                for suppression_weight in (0.75, 1.0):
                    art_label = int(round(art_weight * 100.0))
                    flow_label = int(round(flow_weight * 100.0))
                    suppression_label = int(round(suppression_weight * 100.0))
                    spec = _clone_spec(
                        r10_exact,
                        experiment_id=f"SEARCH-R10E-a{art_label:02d}-f{flow_label:02d}-{flow_model}-sup{suppression_label:02d}",
                        description=(
                            "Exact-champion-adjacent observation-first repair with "
                            f"art_weight={art_weight:.2f}, flow_weight={flow_weight:.2f}, "
                            f"flow_model={flow_model}, suppression_weight={suppression_weight:.2f}."
                        ),
                        repair_params={
                            "diagnosed_weight": 1.0,
                            "art_weight": float(art_weight),
                            "flow_weight": float(flow_weight),
                            "diagnosed_series_model": "level",
                            "art_series_model": "level",
                            "flow_series_model": str(flow_model),
                            "diagnosed_recent_blend_weight": 1.0,
                            "art_recent_blend_weight": 1.0,
                            "flow_recent_blend_weight": 0.75 if flow_model == "delta" else 1.0,
                            "suppression_carry_weight": float(suppression_weight),
                        },
                    )
                    candidates.append(RepairSearchCandidate("EXP-R10-EXACT-CHAMPION", spec.experiment_id, spec))

    r10_dense = spec_map["EXP-R10-DENSE-CHAMPION"]
    for art_recent_blend_weight in (0.5, 0.75, 1.0):
        for flow_weight in (0.5, 1.0):
            art_blend_label = int(round(art_recent_blend_weight * 100.0))
            flow_label = int(round(flow_weight * 100.0))
            spec = _clone_spec(
                r10_dense,
                experiment_id=f"SEARCH-R10D-artdelta-ab{art_blend_label:02d}-f{flow_label:02d}",
                description=(
                    "Dense-champion-adjacent observation-first repair with "
                    f"art_series_model=delta, art_recent_blend_weight={art_recent_blend_weight:.2f}, "
                    f"flow_weight={flow_weight:.2f}."
                ),
                repair_params={
                    "diagnosed_weight": 1.0,
                    "art_weight": 1.0,
                    "flow_weight": float(flow_weight),
                    "diagnosed_series_model": "level",
                    "art_series_model": "delta",
                    "flow_series_model": "level",
                    "diagnosed_recent_blend_weight": 1.0,
                    "art_recent_blend_weight": float(art_recent_blend_weight),
                    "flow_recent_blend_weight": 1.0,
                    "suppression_carry_weight": 1.0,
                },
            )
            candidates.append(RepairSearchCandidate("EXP-R10-DENSE-CHAMPION", spec.experiment_id, spec))
    return candidates


def _build_contract_context(
    archive_run_id: str,
    quarterly_contract: str,
    *,
    annual_rows: list[dict[str, Any]],
    availability: dict[str, Any],
) -> dict[str, Any]:
    if quarterly_contract == "exact_only":
        observation_rows = suite.build_quarterly_observation_rows(archive_run_id)
        scoring_tiers = {"exact_observed"}
        execution_contract = "exact_quarterly_only_for_model_loop; bridge_and_annual_history_for_availability_and_diagnostics"
    elif quarterly_contract == "dense_train_observed_score":
        dense_contract = suite._build_dense_contract_payload(archive_run_id)
        observation_rows = list(dense_contract["rows"])
        scoring_tiers = {"exact_observed", "bridge_observed"}
        execution_contract = "dense_quarterly_train_on_exact_bridge_rule_based; score_holdout_only_on_exact_and_bridge"
    else:
        raise ValueError(f"Unsupported quarterly contract: {quarterly_contract}")
    return {
        "quarterly_contract": quarterly_contract,
        "quarterly_execution_contract": execution_contract,
        "observation_rows": observation_rows,
        "annual_rows": annual_rows,
        "availability": availability,
        "scoring_tiers": scoring_tiers,
    }


def _contract_summary(result: dict[str, Any]) -> dict[str, Any]:
    quarterly_summary = dict(result["quarterly_summary"])
    annual_summary = dict(result["annual_summary"])
    return {
        "decision": str(result["decision"]),
        "decision_reason": str(result["decision_reason"]),
        "quarterly_mean_mae": float(quarterly_summary["candidate_mean_mae"]),
        "quarterly_baseline_mean_mae": float(quarterly_summary["carry_forward_mean_mae"]),
        "quarterly_worst_mae": float(quarterly_summary["candidate_worst_mae"]),
        "quarterly_baseline_worst_mae": float(quarterly_summary["carry_forward_worst_mae"]),
        "annual_mean_incidence_error": float(annual_summary["candidate_mean_incidence_error"]),
        "annual_baseline_incidence_error": float(annual_summary["baseline_mean_incidence_error"]),
        "score_tuple": [float(value) for value in suite._score_experiment_result(result)],
        "candidate_count": int(result["candidate_count"]),
    }


def _contract_objective_vector(
    contract_results: dict[str, dict[str, Any]],
    contracts: list[str],
) -> tuple[float, ...]:
    values: list[float] = []
    for contract in contracts:
        values.extend(float(value) for value in suite._score_experiment_result(contract_results[contract]))
    return tuple(values)


def _dominates(lhs: tuple[float, ...], rhs: tuple[float, ...], *, eps: float = 1e-12) -> bool:
    if len(lhs) != len(rhs):
        raise ValueError("Objective vectors must have the same length.")
    not_worse = all(float(l) <= float(r) + eps for l, r in zip(lhs, rhs, strict=True))
    strictly_better = any(float(l) + eps < float(r) for l, r in zip(lhs, rhs, strict=True))
    return bool(not_worse and strictly_better)


def _equivalent(lhs: tuple[float, ...], rhs: tuple[float, ...], *, eps: float = 1e-12) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(abs(float(l) - float(r)) <= eps for l, r in zip(lhs, rhs, strict=True))


def _candidate_sort_key(
    row: dict[str, Any],
    *,
    primary_contract: str,
    contracts: list[str],
) -> tuple[float, ...]:
    values: list[float] = []
    primary_summary = row["contract_summaries"][primary_contract]
    values.extend(float(value) for value in primary_summary["score_tuple"])
    for contract in contracts:
        if contract == primary_contract:
            continue
        values.extend(float(value) for value in row["contract_summaries"][contract]["score_tuple"])
    return tuple(values)


def _save_candidate_contract_artifacts(
    analysis_dir: Path,
    contract_name: str,
    candidate_result: dict[str, Any],
) -> dict[str, str]:
    contract_dir = ensure_dir(analysis_dir / "contracts" / contract_name)
    json_path = contract_dir / f"{candidate_result['experiment_id']}.json"
    write_json(json_path, candidate_result)
    graph_path = contract_dir / f"{candidate_result['experiment_id']}.png"
    suite._save_experiment_graph(candidate_result, graph_path)
    artifact_paths = {
        "json_file": str(Path("contracts") / contract_name / json_path.name),
        "graph_file": str(Path("contracts") / contract_name / graph_path.name),
    }
    if candidate_result.get("family") == "repair":
        hazard_path = contract_dir / f"{candidate_result['experiment_id']}_hazard_curves.png"
        if suite._save_hazard_curve_graph(candidate_result, hazard_path):
            artifact_paths["hazard_graph_file"] = str(Path("contracts") / contract_name / hazard_path.name)
    return artifact_paths


def _save_primary_overview(
    path: Path,
    candidates: list[dict[str, Any]],
    *,
    primary_contract: str,
) -> None:
    ordered = sorted(candidates, key=lambda row: float(row["contract_summaries"][primary_contract]["quarterly_mean_mae"]))
    labels = [str(row["experiment_id"]) for row in ordered]
    candidate_values = [float(row["contract_summaries"][primary_contract]["quarterly_mean_mae"]) for row in ordered]
    baseline_values = [float(row["contract_summaries"][primary_contract]["quarterly_baseline_mean_mae"]) for row in ordered]
    colors = ["tab:green" if row["search_decision"] == "kept" else "tab:gray" for row in ordered]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, max(5.0, len(labels) * 0.42)))
    ax.barh(y - 0.18, baseline_values, height=0.34, color="tab:blue", alpha=0.45, label="Carry-forward")
    ax.barh(y + 0.18, candidate_values, height=0.34, color=colors, alpha=0.8, label="Candidate")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Quarterly mean normalized MAE ({primary_contract})")
    ax.set_title("Bounded Repair Search: primary-contract quarterly MAE")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_frontier_scatter(
    path: Path,
    candidates: list[dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for row in candidates:
        exact = float(row["contract_summaries"]["exact_only"]["quarterly_mean_mae"])
        dense = float(row["contract_summaries"]["dense_train_observed_score"]["quarterly_mean_mae"])
        color = "tab:green" if row["search_decision"] == "kept" else "tab:gray"
        marker = "o" if row["search_decision"] == "kept" else "x"
        ax.scatter(exact, dense, color=color, marker=marker, s=64, alpha=0.85)
        ax.text(exact, dense, str(row["experiment_id"]), fontsize=7, alpha=0.9)
    ax.set_xlabel("Exact-contract quarterly mean MAE")
    ax.set_ylabel("Dense-contract quarterly mean MAE")
    ax.set_title("Bounded Repair Search Frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_search_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Repair Search",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Primary contract: `{payload['primary_contract']}`",
        f"- Contracts: `{', '.join(payload['contracts'])}`",
        f"- Candidate count: `{payload['candidate_count']}`",
        "",
        "## Champion",
        "",
        f"- Experiment: `{payload['champion']['experiment_id']}`",
        f"- Based on: `{payload['champion']['base_experiment_id']}`",
        "",
        "| Contract | Candidate mean MAE | Baseline mean MAE | Annual incidence error | Annual baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    champion = payload["champion"]
    for contract in payload["contracts"]:
        summary = champion["contract_summaries"][contract]
        lines.append(
            f"| `{contract}` | {float(summary['quarterly_mean_mae']):.6f} | "
            f"{float(summary['quarterly_baseline_mean_mae']):.6f} | "
            f"{float(summary['annual_mean_incidence_error']):.6f} | "
            f"{float(summary['annual_baseline_incidence_error']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Contract Champions",
            "",
            "| Contract | Experiment | Based on | Candidate mean MAE | Baseline mean MAE | Annual incidence error |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for contract, row in payload["contract_champions"].items():
        summary = row["contract_summaries"][contract]
        lines.append(
            f"| `{contract}` | `{row['experiment_id']}` | `{row['base_experiment_id']}` | "
            f"{float(summary['quarterly_mean_mae']):.6f} | "
            f"{float(summary['quarterly_baseline_mean_mae']):.6f} | "
            f"{float(summary['annual_mean_incidence_error']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Pareto Frontier",
            "",
            "| Experiment | Based on | Exact MAE | Dense MAE | Search decision |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in payload["frontier"]:
        exact = row["contract_summaries"]["exact_only"]
        dense = row["contract_summaries"]["dense_train_observed_score"]
        lines.append(
            f"| `{row['experiment_id']}` | `{row['base_experiment_id']}` | "
            f"{float(exact['quarterly_mean_mae']):.6f} | {float(dense['quarterly_mean_mae']):.6f} | `{row['search_decision']}` |"
        )

    lines.extend(
        [
            "",
            "## Accepted Timeline",
            "",
            "| Step | Experiment | Removed frontier entries |",
            "|---:|---|---|",
        ]
    )
    for idx, row in enumerate(payload["accepted_timeline"], start=1):
        removed = ", ".join(f"`{value}`" for value in row["removed_frontier_ids"]) or ""
        lines.append(f"| {idx} | `{row['experiment_id']}` | {removed} |")

    lines.extend(
        [
            "",
            "## All Candidates",
            "",
            "| Experiment | Based on | Exact MAE | Dense MAE | Exact annual | Dense annual | Search decision | Primary graph |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["candidates"]:
        exact = row["contract_summaries"]["exact_only"]
        dense = row["contract_summaries"]["dense_train_observed_score"]
        primary_graph = row["contract_artifacts"][payload["primary_contract"]]["graph_file"]
        lines.append(
            f"| `{row['experiment_id']}` | `{row['base_experiment_id']}` | "
            f"{float(exact['quarterly_mean_mae']):.6f} | {float(dense['quarterly_mean_mae']):.6f} | "
            f"{float(exact['annual_mean_incidence_error']):.6f} | {float(dense['annual_mean_incidence_error']):.6f} | "
            f"`{row['search_decision']}` | `{primary_graph}` |"
        )

    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Primary overview graph: `{payload['primary_overview_graph_file']}`",
            f"- Frontier scatter graph: `{payload['frontier_scatter_graph_file']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_tr_v3_repair_search(
    *,
    run_id: str,
    archive_run_id: str | None,
    contracts: list[str] | None = None,
    primary_contract: str = "exact_only",
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
) -> dict[str, Any]:
    resolved_archive_run_id = str(archive_run_id or suite._latest_standard_archive_run())
    chosen_contracts = list(contracts or ["exact_only", "dense_train_observed_score"])
    for contract in chosen_contracts:
        if contract not in CONTRACT_CHOICES:
            raise ValueError(f"Unsupported contract: {contract}")
    if primary_contract not in chosen_contracts:
        raise ValueError("primary_contract must be included in contracts.")

    annual_rows = suite.build_annual_anchor_rows(resolved_archive_run_id)
    availability = suite._build_availability_payload(resolved_archive_run_id)
    contract_contexts = {
        contract: _build_contract_context(
            resolved_archive_run_id,
            contract,
            annual_rows=annual_rows,
            availability=availability,
        )
        for contract in chosen_contracts
    }

    analysis_dir = ensure_dir(suite.repo_root() / "artifacts" / "runs" / run_id / "analysis")
    candidate_rows: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    accepted_timeline: list[dict[str, Any]] = []

    for candidate in build_repair_search_specs():
        contract_results: dict[str, dict[str, Any]] = {}
        contract_summaries: dict[str, dict[str, Any]] = {}
        contract_artifacts: dict[str, dict[str, str]] = {}
        for contract in chosen_contracts:
            context = contract_contexts[contract]
            result = suite._evaluate_experiment_spec(
                candidate.spec,
                observation_rows=context["observation_rows"],
                annual_rows=context["annual_rows"],
                availability=context["availability"],
                scoring_tiers=context["scoring_tiers"],
                quarterly_start_year=quarterly_start_year,
                quarterly_end_year=quarterly_end_year,
                quarterly_min_train_years=quarterly_min_train_years,
                annual_start_year=annual_start_year,
                annual_end_year=annual_end_year,
                annual_min_train_years=annual_min_train_years,
                horizon_years=horizon_years,
            )
            contract_results[contract] = result
            contract_summaries[contract] = _contract_summary(result)
            contract_artifacts[contract] = _save_candidate_contract_artifacts(analysis_dir, contract, result)

        objective_vector = _contract_objective_vector(contract_results, chosen_contracts)
        row = {
            "experiment_id": candidate.search_id,
            "base_experiment_id": candidate.base_experiment_id,
            "family": candidate.spec.family,
            "transition_model": candidate.spec.transition_model,
            "description": candidate.spec.description,
            "transition_ridge_multipliers": dict(candidate.spec.transition_ridge_multipliers),
            "repair_params": dict(candidate.spec.repair_params),
            "contract_summaries": contract_summaries,
            "contract_artifacts": contract_artifacts,
            "objective_vector": [float(value) for value in objective_vector],
        }

        equivalent_ids = [
            frontier_row["experiment_id"]
            for frontier_row in frontier
            if _equivalent(tuple(frontier_row["objective_vector"]), objective_vector)
        ]
        dominators = [
            frontier_row["experiment_id"]
            for frontier_row in frontier
            if _dominates(tuple(frontier_row["objective_vector"]), objective_vector)
        ]
        if equivalent_ids:
            row["search_decision"] = "discarded"
            row["search_reason"] = f"Equivalent to frontier candidate(s): {', '.join(equivalent_ids)}."
        elif dominators:
            row["search_decision"] = "discarded"
            row["search_reason"] = f"Dominated by frontier candidate(s): {', '.join(dominators)}."
        else:
            removed_frontier = [
                frontier_row["experiment_id"]
                for frontier_row in frontier
                if _dominates(objective_vector, tuple(frontier_row["objective_vector"]))
            ]
            frontier = [
                frontier_row
                for frontier_row in frontier
                if not _dominates(objective_vector, tuple(frontier_row["objective_vector"]))
            ]
            row["search_decision"] = "kept"
            row["search_reason"] = "Entered the Pareto frontier for the chosen contracts."
            frontier.append(row)
            accepted_timeline.append(
                {
                    "experiment_id": row["experiment_id"],
                    "base_experiment_id": row["base_experiment_id"],
                    "removed_frontier_ids": list(removed_frontier),
                }
            )
        candidate_rows.append(row)

    frontier = sorted(frontier, key=lambda row: _candidate_sort_key(row, primary_contract=primary_contract, contracts=chosen_contracts))
    champion = frontier[0]
    contract_champions = {
        contract: min(
            candidate_rows,
            key=lambda row: tuple(float(value) for value in row["contract_summaries"][contract]["score_tuple"]),
        )
        for contract in chosen_contracts
    }

    primary_overview_graph = analysis_dir / "repair_search_primary_overview.png"
    _save_primary_overview(primary_overview_graph, candidate_rows, primary_contract=primary_contract)
    frontier_scatter_graph = analysis_dir / "repair_search_frontier_scatter.png"
    _save_frontier_scatter(frontier_scatter_graph, candidate_rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "archive_run_id": resolved_archive_run_id,
        "contracts": chosen_contracts,
        "primary_contract": primary_contract,
        "quarterly_window": {
            "start_year": quarterly_start_year,
            "end_year": quarterly_end_year,
            "min_train_years": quarterly_min_train_years,
            "horizon_years": horizon_years,
        },
        "annual_window": {
            "start_year": annual_start_year,
            "end_year": annual_end_year,
            "min_train_years": annual_min_train_years,
            "horizon_years": horizon_years,
        },
        "candidate_count": len(candidate_rows),
        "champion": champion,
        "contract_champions": contract_champions,
        "frontier": frontier,
        "accepted_timeline": accepted_timeline,
        "candidates": candidate_rows,
        "primary_overview_graph_file": primary_overview_graph.name,
        "frontier_scatter_graph_file": frontier_scatter_graph.name,
    }
    write_json(analysis_dir / "tr_v3_repair_search_report.json", payload)
    (analysis_dir / "tr_v3_repair_search_report.md").write_text(_markdown_search_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tr-v3-repair-search")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=suite._latest_standard_archive_run())
    parser.add_argument(
        "--contracts",
        nargs="+",
        choices=CONTRACT_CHOICES,
        default=list(CONTRACT_CHOICES),
    )
    parser.add_argument("--primary-contract", choices=CONTRACT_CHOICES, default="exact_only")
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_repair_search(
        run_id=args.run_id,
        archive_run_id=args.archive_run_id,
        contracts=list(args.contracts),
        primary_contract=args.primary_contract,
        quarterly_start_year=args.quarterly_start_year,
        quarterly_end_year=args.quarterly_end_year,
        quarterly_min_train_years=args.quarterly_min_train_years,
        annual_start_year=args.annual_start_year,
        annual_end_year=args.annual_end_year,
        annual_min_train_years=args.annual_min_train_years,
        horizon_years=args.horizon_years,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
