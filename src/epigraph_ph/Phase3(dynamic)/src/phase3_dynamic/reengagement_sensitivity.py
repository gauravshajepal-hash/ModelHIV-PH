from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .backhalf_channels import BackHalfChannelContext, build_backhalf_channel_features
from .data import default_epigraph_root, sandbox_repo_root
from .runtime import ensure_dir, write_json
from .hybrid_champion import run_hybrid_champion_search


REENGAGEMENT_SENSITIVITY_SCHEMA_VERSION = "phase3_dynamic_reengagement_sensitivity_gate.v1"
REENGAGEMENT_SENSITIVITY_MODES: tuple[str, ...] = (
    "zero",
    "public_stock_flow_proxy",
    "upper_bound_proxy",
)
FOCUSED_REENGAGEMENT_FAMILIES: tuple[str, ...] = (
    "monthly_joint_horizon_gated_endpoint_state",
    "backhalf_channel_horizon_gated_endpoint_state",
    "backhalf_channel_state_multihorizon_support_decay",
)


def _family_score_rows(report_path: Path) -> list[dict[str, Any]]:
    csv_path = report_path.with_name("hybrid_family_scores.csv")
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "family": row["family"],
                "one_year_mae": float(row["candidate_mean_mae"]),
                "carry_forward_mae": float(row["carry_forward_mean_mae"]),
                "worst_one_year_mae": float(row["candidate_worst_mae"]),
                "lifted_mae": float(row["shock_trajectory_candidate_mean_mae"]),
                "r10_reference_mae": float(row["shock_trajectory_r10_reference_mae"]),
                "promotion_status": row["promotion_status"],
                "blockers": row["blockers"],
            }
            for row in csv.DictReader(handle)
        ]


def _branch_summary(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(str((payload.get("artifact_paths") or {}).get("report_json") or ""))
    rows = _family_score_rows(report_path)
    best = dict(payload.get("best_family") or {})
    best_family = str(best.get("family") or "")
    best_score = dict(best.get("score") or {})
    shock_gate = dict(best.get("shock_trajectory_gate") or {})
    overall = dict(shock_gate.get("overall") or {})
    r10_reference = dict(shock_gate.get("r10_reference") or {})
    promotion_gate = dict(best.get("promotion_gate") or {})
    champion = payload.get("champion_by_claim_aware_promotion")
    best_row = next((row for row in rows if row["family"] == best_family), None)
    lifted_mae = float(overall.get("candidate_mean_mae") or (best_row or {}).get("lifted_mae") or float("inf"))
    r10_mae = float(r10_reference.get("reference_quarterly_mean_mae") or (best_row or {}).get("r10_reference_mae") or float("inf"))
    one_year_mae = float(best_score.get("candidate_mean_mae") or (best_row or {}).get("one_year_mae") or float("inf"))
    carry_forward_mae = float(best_score.get("carry_forward_mean_mae") or (best_row or {}).get("carry_forward_mae") or float("inf"))
    return {
        "mode": mode,
        "run_id": payload.get("run_id"),
        "report_path": str(report_path),
        "best_family": best_family,
        "champion_family": None if not champion else str((champion or {}).get("family") or ""),
        "promotion_status": str(promotion_gate.get("status") or ""),
        "promotion_eligible": bool(promotion_gate.get("promotion_eligible")),
        "one_year_mae": one_year_mae,
        "carry_forward_mae": carry_forward_mae,
        "lifted_mae": lifted_mae,
        "r10_reference_mae": r10_mae,
        "beats_carry_forward_one_year": one_year_mae < carry_forward_mae,
        "beats_r10_lifted": lifted_mae < r10_mae,
        "family_scores": rows,
        "blockers": list(promotion_gate.get("blockers") or []),
    }


def _evidence_mode_summary(source_run_id: str, baseline_source_run_id: str | None) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    quarters = [f"{year}-Q{quarter}" for year in range(2010, 2026) for quarter in range(1, 5)]
    summaries: dict[str, Any] = {}
    for mode in REENGAGEMENT_SENSITIVITY_MODES:
        feature_payload = build_backhalf_channel_features(
            BackHalfChannelContext(
                epigraph_root=epigraph_root,
                source_run_id=source_run_id,
                baseline_source_run_id=baseline_source_run_id,
                reengagement_sensitivity_mode=mode,
            ),
            train_end_quarter="2025-Q4",
            quarters=quarters,
        )
        evidence = dict(feature_payload.get("art_retention_evidence") or {})
        late_rows = {
            quarter: row
            for quarter, row in evidence.items()
            if str(quarter) >= "2024-Q1"
        }
        summaries[mode] = {
            "feature_summary": dict(feature_payload.get("art_retention_evidence_summary") or {}),
            "late_quarter_count": len(late_rows),
            "late_direct_reengagement_sum": float(
                sum(float(row.get("direct_reengagement_count") or 0.0) for row in late_rows.values())
            ),
            "late_reengagement_count_sum": float(
                sum(float(row.get("reengagement_count") or 0.0) for row in late_rows.values())
            ),
            "late_public_point_proxy_sum": float(
                sum(float(row.get("latent_reengagement_public_point_count") or 0.0) for row in late_rows.values())
            ),
            "late_upper_bound_proxy_sum": float(
                sum(float(row.get("latent_reengagement_upper_bound_count") or 0.0) for row in late_rows.values())
            ),
            "late_unobserved_attrition_sum": float(
                sum(float(row.get("latent_unobserved_attrition_count") or 0.0) for row in late_rows.values())
            ),
        }
    return summaries


def _stability_assessment(branches: list[dict[str, Any]]) -> dict[str, Any]:
    if not branches:
        return {"status": "not_evaluable", "conclusion_changed": True}
    best_families = {str(row.get("best_family") or "") for row in branches}
    champion_families = {str(row.get("champion_family") or "") for row in branches}
    promotion_states = {bool(row.get("promotion_eligible")) for row in branches}
    r10_states = {bool(row.get("beats_r10_lifted")) for row in branches}
    carry_states = {bool(row.get("beats_carry_forward_one_year")) for row in branches}
    lifted_values = [float(row.get("lifted_mae") or float("inf")) for row in branches]
    one_year_values = [float(row.get("one_year_mae") or float("inf")) for row in branches]
    finite_lifted = [value for value in lifted_values if np.isfinite(value)]
    finite_one_year = [value for value in one_year_values if np.isfinite(value)]
    lifted_spread = float(max(finite_lifted) - min(finite_lifted)) if finite_lifted else float("inf")
    one_year_spread = float(max(finite_one_year) - min(finite_one_year)) if finite_one_year else float("inf")
    conclusion_changed = (
        len(best_families) > 1
        or len(champion_families) > 1
        or len(promotion_states) > 1
        or len(r10_states) > 1
        or len(carry_states) > 1
    )
    return {
        "status": "sensitivity_parameter" if conclusion_changed else "stable_under_tested_reengagement_modes",
        "conclusion_changed": bool(conclusion_changed),
        "best_family_set": sorted(best_families),
        "champion_family_set": sorted(champion_families),
        "promotion_state_set": sorted(promotion_states),
        "beats_r10_lifted_state_set": sorted(r10_states),
        "beats_carry_forward_one_year_state_set": sorted(carry_states),
        "lifted_mae_spread": lifted_spread,
        "one_year_mae_spread": one_year_spread,
        "interpretation": (
            "Re-engagement must remain a sensitivity parameter because best family, promotion, "
            "or benchmark pass/fail status changed across R=0/public/upper-bound branches."
            if conclusion_changed
            else "The headline benchmark conclusion is stable across R=0/public/upper-bound branches, "
            "but re-engagement remains proxy-only because direct restart/return-to-care rows are absent."
        ),
    }


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    branches = list(payload.get("branches") or [])
    if not branches:
        return
    labels = [
        {
            "zero": "R=0",
            "public_stock_flow_proxy": "public\nproxy",
            "upper_bound_proxy": "upper\nbound",
        }.get(str(row.get("mode")), str(row.get("mode")))
        for row in branches
    ]
    one_year = [float(row.get("one_year_mae") or 0.0) for row in branches]
    lifted = [float(row.get("lifted_mae") or 0.0) for row in branches]
    carry = [float(row.get("carry_forward_mae") or 0.0) for row in branches]
    r10 = [float(row.get("r10_reference_mae") or 0.0) for row in branches]
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.18, top=0.78, wspace=0.30)
    fig.text(0.08, 0.95, "Re-engagement sensitivity gate", fontsize=15, fontweight="bold", ha="left")
    fig.text(
        0.08,
        0.89,
        "R=0 vs public stock-flow proxy vs upper-bound proxy; direct process claim remains blocked without restart/return rows.",
        fontsize=9.5,
        color="#444444",
        ha="left",
    )
    x = np.arange(len(branches))
    axes[0].bar(x - 0.18, one_year, width=0.34, color="#4c78a8", label="1-year MAE")
    axes[0].bar(x + 0.18, lifted, width=0.34, color="#e45756", label="lifted trajectory MAE")
    axes[0].plot(x, carry, color="#222222", linestyle="--", marker="o", linewidth=1.1, label="carry-forward")
    axes[0].plot(x, r10, color="#54a24b", linestyle=":", marker="o", linewidth=1.8, label="R10")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("normalized MAE, lower is better")
    axes[0].set_title("A. Benchmark stability")
    axes[0].grid(axis="y", color="#e6e6e6")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    promoted = [1.0 if bool(row.get("promotion_eligible")) else 0.0 for row in branches]
    r10_pass = [1.0 if bool(row.get("beats_r10_lifted")) else 0.0 for row in branches]
    carry_pass = [1.0 if bool(row.get("beats_carry_forward_one_year")) else 0.0 for row in branches]
    axes[1].bar(x - 0.24, promoted, width=0.22, color="#b279a2", label="promoted")
    axes[1].bar(x, carry_pass, width=0.22, color="#4c78a8", label="beats carry-forward")
    axes[1].bar(x + 0.24, r10_pass, width=0.22, color="#e45756", label="beats R10")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["no", "yes"])
    axes[1].set_ylim(0, 1.25)
    axes[1].set_title("B. Conclusion-changing gates")
    axes[1].grid(axis="y", color="#e6e6e6")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    evidence_summary = dict(payload.get("evidence_mode_summary") or {})
    r_used = [float((evidence_summary.get(str(row.get("mode"))) or {}).get("late_reengagement_count_sum") or 0.0) for row in branches]
    r_public = [float((evidence_summary.get(str(row.get("mode"))) or {}).get("late_public_point_proxy_sum") or 0.0) for row in branches]
    r_upper = [float((evidence_summary.get(str(row.get("mode"))) or {}).get("late_upper_bound_proxy_sum") or 0.0) for row in branches]
    axes[2].bar(x - 0.22, r_used, width=0.20, color="#4c78a8", label="R used")
    axes[2].bar(x, r_public, width=0.20, color="#72b7b2", label="public point")
    axes[2].bar(x + 0.22, r_upper, width=0.20, color="#f58518", label="upper bound")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("late-quarter count sum")
    axes[2].set_title("C. Branch assumptions differ")
    axes[2].grid(axis="y", color="#e6e6e6")
    axes[2].legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 Re-engagement Sensitivity Gate",
        "",
        "This gate evaluates whether the current Phase 3 conclusion depends on the unobserved re-engagement assumption.",
        "",
        f"![Re-engagement sensitivity gate]({payload['artifact_paths']['dashboard_png']})",
        "",
        "## Branches",
        "",
        "- `zero`: no re-engagement proxy when direct restart/return rows are absent.",
        "- `public_stock_flow_proxy`: point proxy from public ART stock-flow residuals.",
        "- `upper_bound_proxy`: conservative upper-bound proxy from the public interrupted/no-longer-on-treatment pool.",
        "",
        "## Results",
        "",
        "| mode | best family | promoted | 1-year MAE | lifted MAE | beats carry-forward | beats R10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("branches") or []:
        lines.append(
            "| {mode} | {best_family} | {promotion_eligible} | {one_year_mae:.6f} | {lifted_mae:.6f} | {beats_carry_forward_one_year} | {beats_r10_lifted} |".format(
                **row
            )
        )
    assessment = dict(payload.get("stability_assessment") or {})
    lines.extend(
        [
            "",
            "## Stability Assessment",
            "",
            f"- Status: `{assessment.get('status')}`",
            f"- Conclusion changed: `{assessment.get('conclusion_changed')}`",
            f"- Lifted MAE spread: `{float(assessment.get('lifted_mae_spread') or 0.0):.6f}`",
            f"- One-year MAE spread: `{float(assessment.get('one_year_mae_spread') or 0.0):.6f}`",
            f"- Interpretation: {assessment.get('interpretation')}",
            "",
            "## Evidence Branch Check",
            "",
            "| mode | late R used | public point proxy | upper-bound proxy | unobserved attrition |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode, row in dict(payload.get("evidence_mode_summary") or {}).items():
        lines.append(
            "| {mode} | {late_reengagement_count_sum:.1f} | {late_public_point_proxy_sum:.1f} | {late_upper_bound_proxy_sum:.1f} | {late_unobserved_attrition_sum:.1f} |".format(
                mode=mode,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Claim Contract",
            "",
            "- If branch conclusions differ, re-engagement is a sensitivity parameter.",
            "- If branch conclusions are stable, re-engagement is still not a fitted process claim because direct restart/return-to-care rows are absent.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_reengagement_sensitivity_gate(
    *,
    run_id: str,
    source_run_id: str,
    baseline_source_run_id: str | None = None,
    head_families: tuple[str, ...] = FOCUSED_REENGAGEMENT_FAMILIES,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    branches: list[dict[str, Any]] = []
    for mode in REENGAGEMENT_SENSITIVITY_MODES:
        branch_run_id = f"{run_id}-{mode.replace('_', '-')}"
        branch_payload = run_hybrid_champion_search(
            run_id=branch_run_id,
            source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
            reengagement_sensitivity_mode=mode,
            head_families=head_families,
        )
        branches.append(_branch_summary(mode, branch_payload))
    payload = {
        "schema_version": REENGAGEMENT_SENSITIVITY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "head_families": list(head_families),
        "branches": branches,
        "evidence_mode_summary": _evidence_mode_summary(source_run_id, baseline_source_run_id),
        "stability_assessment": _stability_assessment(branches),
        "claim_contract": {
            "direct_reengagement_required_for_process_claim": True,
            "sensitivity_parameter_rule": "If best family, promotion, carry-forward pass/fail, or R10 pass/fail differs by branch, re-engagement remains a sensitivity parameter.",
            "proxy_role": "bounded latent driver only, not direct restart/return-to-care evidence",
        },
    }
    dashboard_path = analysis_dir / "reengagement_sensitivity_gate.png"
    json_path = analysis_dir / "reengagement_sensitivity_gate.json"
    markdown_path = analysis_dir / "reengagement_sensitivity_gate.md"
    payload["artifact_paths"] = {
        "dashboard_png": dashboard_path.as_posix(),
        "report_json": json_path.as_posix(),
        "report_markdown": markdown_path.as_posix(),
    }
    _write_dashboard(payload, dashboard_path)
    write_json(json_path, payload)
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Phase3 re-engagement sensitivity gate.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--baseline-source-run-id")
    args = parser.parse_args()
    payload = run_reengagement_sensitivity_gate(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )
    print(json.dumps(payload.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
