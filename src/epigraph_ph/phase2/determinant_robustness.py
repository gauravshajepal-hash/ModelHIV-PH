from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.runtime import ensure_dir, read_json, utc_now_iso, write_json


DETERMINANT_ROBUSTNESS_SCHEMA_VERSION = "phase2_determinant_robustness.v1"


def _edge_parts(edge_key: str) -> dict[str, Any]:
    kind, rest = str(edge_key).split(":", 1)
    transition, lag_text = rest.rsplit(":lag", 1)
    source, target = transition.split("->", 1)
    return {
        "edge_kind": kind,
        "source": source,
        "target": target,
        "lag": int(lag_text),
    }


def _binomial_upper_tail(k: int, n: int, p: float = 0.5) -> float | None:
    if n <= 0:
        return None
    k = max(0, min(int(k), int(n)))
    probability = 0.0
    for value in range(k, n + 1):
        probability += math.comb(n, value) * (float(p) ** value) * ((1.0 - float(p)) ** (n - value))
    return float(probability)


def _edge_report_by_key(edge_falsification_report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("edge_key") or ""): dict(row)
        for row in list(edge_falsification_report.get("edge_reports") or [])
        if str(row.get("edge_key") or "")
    }


def _status_for_edge(
    *,
    edge_key: str,
    source_reestimated_report: Mapping[str, Any],
    edge_falsification_row: Mapping[str, Any],
) -> dict[str, Any]:
    parts = _edge_parts(edge_key)
    summary = dict((source_reestimated_report.get("edge_summary") or {}).get(edge_key) or {})
    evaluation_rows = [
        dict(row)
        for row in list(source_reestimated_report.get("edge_family_rows") or [])
        if str(row.get("edge_key") or "") == edge_key
    ]
    completed_rows = [row for row in evaluation_rows if str(row.get("ablation_status") or "") == "completed"]
    evaluated = int(summary.get("evaluated_family_count") or len(completed_rows))
    survived = int(summary.get("survived_family_count") or 0)
    absent = int(summary.get("absent_family_count") or 0)
    sign_conflict = int(summary.get("sign_conflict_family_count") or 0)
    survival_fraction = None if evaluated <= 0 else float(survived) / float(evaluated)
    source_reestimated_passed = bool(summary.get("passed")) and evaluated > 0
    support_ablation = dict(edge_falsification_row.get("source_ablation") or {})
    time_window = dict(edge_falsification_row.get("time_window_falsification") or {})
    support_passed = bool(support_ablation.get("passed"))
    time_passed = bool(time_window.get("passed"))
    direct_edge = str(parts["edge_kind"]) == "direct"
    phase3_prior_eligible = bool(edge_falsification_row.get("phase3_prior_eligible"))
    source_stable_covariate = bool(direct_edge and support_passed and time_passed and source_reestimated_passed)
    exploratory_covariate = bool(
        direct_edge
        and support_passed
        and evaluated > 0
        and survived > 0
        and sign_conflict == 0
        and not source_stable_covariate
    )
    if phase3_prior_eligible:
        scientific_status = "source_stable_phase3_prior"
    elif source_stable_covariate:
        scientific_status = "source_stable_covariate"
    elif exploratory_covariate:
        scientific_status = "exploratory_source_sensitive_covariate_only"
    elif not direct_edge:
        scientific_status = "hidden_diagnostic_not_direct_covariate"
    else:
        scientific_status = "blocked_unstable_determinant"
    return {
        "edge_key": edge_key,
        **parts,
        "baseline_weight": edge_falsification_row.get("weight"),
        "support_ablation_passed": support_passed,
        "source_reestimated_passed": source_reestimated_passed,
        "blocked_time_passed": time_passed,
        "phase3_prior_eligible": phase3_prior_eligible,
        "source_stable_covariate": source_stable_covariate,
        "exploratory_covariate_only": exploratory_covariate,
        "phase3_default_allowed": bool(phase3_prior_eligible or source_stable_covariate),
        "scientific_status": scientific_status,
        "evaluated_family_count": evaluated,
        "survived_family_count": survived,
        "absent_family_count": absent,
        "sign_conflict_family_count": sign_conflict,
        "survival_fraction": survival_fraction,
        "binomial_upper_tail_p_value": _binomial_upper_tail(survived, evaluated),
        "evaluation_rows": evaluation_rows,
        "promotion_blockers": list(edge_falsification_row.get("promotion_blockers") or []),
    }


def _bundle_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bundles: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in edge_rows:
        key = (str(row.get("edge_kind") or ""), str(row.get("source") or ""), str(row.get("target") or ""))
        bundles.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (edge_kind, source, target), members in sorted(bundles.items()):
        direct_members = [row for row in members if str(row.get("edge_kind")) == "direct"]
        fractions = [
            float(row["survival_fraction"])
            for row in members
            if row.get("survival_fraction") is not None
        ]
        rows.append(
            {
                "bundle_key": f"{edge_kind}:{source}->{target}",
                "edge_kind": edge_kind,
                "source": source,
                "target": target,
                "lags": sorted(int(row.get("lag") or 0) for row in members),
                "edge_count": len(members),
                "direct_edge_count": len(direct_members),
                "phase3_default_allowed_count": sum(1 for row in members if bool(row.get("phase3_default_allowed"))),
                "exploratory_covariate_count": sum(1 for row in members if bool(row.get("exploratory_covariate_only"))),
                "blocked_edge_count": sum(1 for row in members if not bool(row.get("phase3_default_allowed"))),
                "min_survival_fraction": None if not fractions else float(min(fractions)),
                "max_survival_fraction": None if not fractions else float(max(fractions)),
            }
        )
    return rows


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 2 Determinant Robustness",
        "",
        f"- Source re-estimation report: `{report.get('source_reestimated_report_path')}`",
        f"- Edge falsification report: `{report.get('edge_falsification_report_path')}`",
        f"- Phase 3 default-allowed direct edges: `{report.get('phase3_default_allowed_direct_edge_count')}`",
        f"- Exploratory source-sensitive direct covariates: `{report.get('exploratory_direct_covariate_count')}`",
        "",
        "## Contract",
        "",
        "Direct Phase 2 determinants may enter strict Phase 3 only if they are direct edges and pass support ablation, true source-family re-estimation, and blocked-time gates. Edges failing that all-gates contract may be reported as exploratory covariates, but they are not default Phase 3 inputs.",
        "",
        "## Edge Rows",
        "",
        "| Edge | Status | Survival | Source Refit | Blocked Time | Default Allowed |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in list(report.get("edge_rows") or []):
        survival = row.get("survival_fraction")
        lines.append(
            f"| `{row.get('edge_key')}` | `{row.get('scientific_status')}` | "
            f"{'' if survival is None else f'{float(survival):.3f}'} | "
            f"`{bool(row.get('source_reestimated_passed'))}` | `{bool(row.get('blocked_time_passed'))}` | "
            f"`{bool(row.get('phase3_default_allowed'))}` |"
        )
    lines.extend(["", "## Bundle Rows", "", "| Bundle | Edges | Default Allowed | Exploratory | Survival Range |", "| --- | ---: | ---: | ---: | --- |"])
    for row in list(report.get("bundle_rows") or []):
        min_fraction = row.get("min_survival_fraction")
        max_fraction = row.get("max_survival_fraction")
        range_text = "" if min_fraction is None else f"{float(min_fraction):.3f}-{float(max_fraction):.3f}"
        lines.append(
            f"| `{row.get('bundle_key')}` | {int(row.get('edge_count') or 0)} | "
            f"{int(row.get('phase3_default_allowed_count') or 0)} | "
            f"{int(row.get('exploratory_covariate_count') or 0)} | {range_text} |"
        )
    return "\n".join(lines) + "\n"


def _write_dashboard(report: Mapping[str, Any], output_path: Path) -> None:
    edge_rows = list(report.get("edge_rows") or [])
    direct_rows = [row for row in edge_rows if str(row.get("edge_kind") or "") == "direct"]
    labels = [
        str(row.get("edge_key") or "")
        .replace("direct:", "")
        .replace("structural_barrier_pressure", "barrier")
        .replace("mobility_exposure_pressure", "mobility")
        .replace("care_access_continuity", "care")
        for row in direct_rows
    ]
    survival = [0.0 if row.get("survival_fraction") is None else float(row["survival_fraction"]) for row in direct_rows]
    allowed = [1 if bool(row.get("phase3_default_allowed")) else 0 for row in direct_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].bar(range(len(labels)), survival, color="#2f6f73", edgecolor="#111111", linewidth=0.6)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Survived completed source-family refits")
    axes[0].set_title("A. Direct determinant source-family survival")
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[0].spines[["top", "right"]].set_visible(False)
    status_counts = {
        "default_allowed": sum(allowed),
        "exploratory_only": sum(1 for row in direct_rows if bool(row.get("exploratory_covariate_only"))),
        "blocked": sum(1 for row in direct_rows if not bool(row.get("phase3_default_allowed"))),
    }
    axes[1].bar(status_counts.keys(), status_counts.values(), color=["#2f6f73", "#f2a541", "#b23a48"], edgecolor="#111111", linewidth=0.6)
    axes[1].set_title("B. Strict Phase 3 determinant gate")
    axes[1].set_ylabel("Direct edges")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Phase 2 Determinant Robustness Gate", fontweight="bold")
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_determinant_robustness_report(
    *,
    source_reestimated_ablation_path: Path,
    edge_falsification_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_reestimated = read_json(source_reestimated_ablation_path, default={}) or {}
    edge_falsification = read_json(edge_falsification_path, default={}) or {}
    edge_reports = _edge_report_by_key(edge_falsification)
    edge_rows = []
    for edge_key in sorted((source_reestimated.get("edge_summary") or {}).keys()):
        edge_rows.append(
            _status_for_edge(
                edge_key=edge_key,
                source_reestimated_report=source_reestimated,
                edge_falsification_row=edge_reports.get(edge_key, {}),
            )
        )
    completed_family_rows = [
        dict(row)
        for row in list(source_reestimated.get("family_rows") or [])
        if str(row.get("status") or "") == "completed"
    ]
    output_dir = ensure_dir(output_dir)
    report = {
        "schema_version": DETERMINANT_ROBUSTNESS_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "source_reestimated_report_path": str(source_reestimated_ablation_path),
        "edge_falsification_report_path": str(edge_falsification_path),
        "contract": {
            "strict_phase3_default_rule": "direct edges only; support ablation passed; true source-family re-estimation passed; blocked-time falsification passed",
            "exploratory_covariate_rule": "reported for diagnostics when a direct edge has at least one source-family survival and no sign conflict, but not used by strict Phase 3 by default",
            "hidden_edge_rule": "hidden edges remain shared latent diagnostics and are never direct determinant covariates",
            "thresholds": "none; strict rule is all completed falsification gates",
        },
        "completed_source_family_count": len(completed_family_rows),
        "completed_source_families": [str(row.get("excluded_source_family") or "") for row in completed_family_rows],
        "edge_rows": edge_rows,
        "bundle_rows": _bundle_rows(edge_rows),
        "phase3_default_allowed_edge_keys": [
            str(row.get("edge_key") or "")
            for row in edge_rows
            if bool(row.get("phase3_default_allowed"))
        ],
        "phase3_default_allowed_direct_edge_count": sum(
            1
            for row in edge_rows
            if str(row.get("edge_kind") or "") == "direct" and bool(row.get("phase3_default_allowed"))
        ),
        "exploratory_direct_covariate_count": sum(
            1
            for row in edge_rows
            if str(row.get("edge_kind") or "") == "direct" and bool(row.get("exploratory_covariate_only"))
        ),
    }
    report_path = output_dir / "phase2_determinant_robustness_report.json"
    md_path = output_dir / "phase2_determinant_robustness_report.md"
    dashboard_path = output_dir / "phase2_determinant_robustness_dashboard.png"
    report["artifact_paths"] = {
        "report_json": str(report_path),
        "report_markdown": str(md_path),
        "dashboard_png": str(dashboard_path),
    }
    write_json(report_path, report)
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    _write_dashboard(report, dashboard_path)
    write_json(report_path, report)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase2 determinant robustness gate from source-family re-estimation and edge falsification.")
    parser.add_argument("--source-reestimated-ablation-path", required=True)
    parser.add_argument("--edge-falsification-path", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    report = build_determinant_robustness_report(
        source_reestimated_ablation_path=Path(args.source_reestimated_ablation_path),
        edge_falsification_path=Path(args.edge_falsification_path),
        output_dir=Path(args.out_dir),
    )
    print(json.dumps(report.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
