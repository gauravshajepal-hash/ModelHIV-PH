from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .runtime import ensure_dir, read_json, write_json


R24_SCHEMA_VERSION = "phase3_dynamic.r24_matched_r10_fairness_audit.v1"


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() and (parent / "src" / "epigraph_ph" / "Phase3(dynamic)").exists():
            return parent
    raise RuntimeError(f"Cannot resolve repository root from {current}")


def _phase3_root(repo_root: Path) -> Path:
    return repo_root / "src" / "epigraph_ph" / "Phase3(dynamic)"


def _float_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(output):
        return None
    return output


def _require_dict(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _result_by_experiment(r13_report: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    for row in list(r13_report.get("results") or []):
        if isinstance(row, dict) and str(row.get("experiment_id") or "") == experiment_id:
            return row
    raise KeyError(f"Missing {experiment_id} in R13 report")


def _r10_references(r10_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    refs: dict[int, dict[str, Any]] = {}
    for row in list(r10_report.get("horizon_rows") or []):
        if not isinstance(row, dict):
            continue
        horizon = row.get("horizon_years")
        try:
            horizon_i = int(horizon)
        except (TypeError, ValueError):
            continue
        refs[horizon_i] = {
            "horizon_years": horizon_i,
            "reference_experiment_id": row.get("reference_experiment_id"),
            "reference_metric_scope": list(row.get("reference_metric_scope") or []),
            "reference_quarterly_mean_mae": _float_or_none(row.get("reference_quarterly_mean_mae")),
            "reference_carry_forward_mean_mae": _float_or_none(row.get("reference_carry_forward_mean_mae")),
            "reference_selection_policy": row.get("reference_selection_policy"),
            "reference_quarterly_worst_mae": _float_or_none(row.get("reference_quarterly_worst_mae")),
            "quarterly_window": dict(row.get("quarterly_window") or {}),
        }
    return refs


def _program_family_rows(program_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = dict(program_report.get("families") or {})
    for family_name, family_payload in families.items():
        if not isinstance(family_payload, dict):
            continue
        for row in list(family_payload.get("horizon_rows") or []):
            if not isinstance(row, dict):
                continue
            candidate = _float_or_none(row.get("r10_comparable_candidate_mean_mae"))
            reference = _float_or_none(row.get("r10_horizon_reference_mae"))
            rows.append(
                {
                    "family": str(family_name),
                    "horizon_years": int(row.get("horizon_years") or 0),
                    "candidate_mean_mae": _float_or_none(row.get("candidate_mean_mae")),
                    "carry_forward_mean_mae": _float_or_none(row.get("carry_forward_mean_mae")),
                    "r10_comparable_candidate_mean_mae": candidate,
                    "r10_horizon_reference_mae": reference,
                    "candidate_minus_r10_reference_mae": None
                    if candidate is None or reference is None
                    else float(candidate - reference),
                }
            )
    return sorted(rows, key=lambda row: (int(row["horizon_years"]), str(row["family"])))


def _all_support_rows(r13_050: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in list(r13_050.get("horizon_rows") or []):
        if not isinstance(row, dict):
            continue
        candidate = _float_or_none(row.get("r10_comparable_candidate_mean_mae"))
        reference = _float_or_none(row.get("r10_horizon_reference_mae"))
        rows.append(
            {
                "family": "r19_joint_service_cascade_process",
                "horizon_years": int(row.get("horizon_years") or 0),
                "candidate_mean_mae": _float_or_none(row.get("candidate_mean_mae")),
                "carry_forward_mean_mae": _float_or_none(row.get("carry_forward_mean_mae")),
                "r10_comparable_candidate_mean_mae": candidate,
                "r10_horizon_reference_mae": reference,
                "candidate_minus_r10_reference_mae": None
                if candidate is None or reference is None
                else float(candidate - reference),
                "split_count": row.get("split_count"),
            }
        )
    return sorted(rows, key=lambda row: int(row["horizon_years"]))


def _format_float(value: Any) -> str:
    finite = _float_or_none(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_dashboard(path: Path, report: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    all_rows = list(report["all_support_r19_rows"])
    program_rows = list(report["program_route_rows"])
    horizons = [int(row["horizon_years"]) for row in all_rows]
    x = np.arange(len(horizons), dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    fig.suptitle("R24 Matched-R10 Fairness Audit", fontsize=15, fontweight="bold")

    all_candidate = np.asarray([float(row["r10_comparable_candidate_mean_mae"]) for row in all_rows], dtype=np.float64)
    all_r10 = np.asarray([float(row["r10_horizon_reference_mae"]) for row in all_rows], dtype=np.float64)
    all_carry = np.asarray([float(row["carry_forward_mean_mae"]) for row in all_rows], dtype=np.float64)
    width = 0.24
    axes[0].bar(x - width, all_carry, width=width, color="#c9b879", label="carry-forward")
    axes[0].bar(x, all_candidate, width=width, color="#20639b", label="R19 R10-scope")
    axes[0].bar(x + width, all_r10, width=width, color="#111827", label="matched R10")
    axes[0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[0].set_title("Full publication sentinel: R19 beats R10 at 1y/3y but not 5y")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{h}y" for h in horizons])
    axes[0].set_ylabel("normalized MAE")
    axes[0].legend(loc="upper left")

    families = [
        "r19_joint_service_cascade_process",
        "r22_program_metric_coupled_process",
        "r23_recent_origin_program_coupled_process",
    ]
    family_labels = ["R19", "R22", "R23"]
    program_horizons = sorted({int(row["horizon_years"]) for row in program_rows})
    px = np.arange(len(program_horizons), dtype=np.float64)
    pwidth = 0.18
    colors = ["#20639b", "#f28e2b", "#59a14f"]
    for index, family in enumerate(families):
        values = []
        for horizon in program_horizons:
            match = next(
                row
                for row in program_rows
                if row["family"] == family and int(row["horizon_years"]) == int(horizon)
            )
            values.append(float(match["r10_comparable_candidate_mean_mae"]))
        axes[1].bar(px + (index - 1) * pwidth, values, width=pwidth, color=colors[index], label=family_labels[index])
    r10_values = []
    for horizon in program_horizons:
        match = next(
            row
            for row in program_rows
            if row["family"] == "r19_joint_service_cascade_process" and int(row["horizon_years"]) == int(horizon)
        )
        r10_values.append(float(match["r10_horizon_reference_mae"]))
    axes[1].plot(px, r10_values, marker="D", color="#111827", linewidth=2.0, label="matched R10")
    axes[1].set_title("Program route: process variants still miss R10 at 3y/5y")
    axes[1].set_xticks(px)
    axes[1].set_xticklabels([f"{h}y" for h in program_horizons])
    axes[1].set_ylabel("R10-scope normalized MAE")
    axes[1].legend(loc="upper left")

    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown(report: dict[str, Any]) -> str:
    all_rows = list(report["all_support_r19_rows"])
    program_rows = list(report["program_route_rows"])
    r10_refs = dict(report["matched_r10_reference_by_horizon"])
    annual = dict(report["official_annual_gate"])
    paths = dict(report["source_artifacts"])

    lines: list[str] = [
        "# Phase 3 R24 Matched-R10 Fairness and Readout Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        "R19 remains the active mechanistic reference because it beats carry-forward and the conserved annual official-style gate, but it has not achieved the active goal: it still fails the matched R10 endpoint/readout benchmark on the long-horizon program route and narrowly at 5y on the full sentinel.",
        "",
        "The audit changes the next experiment choice. More deterministic R19/R16 process selector variants are low value. The evidence says matched R10 is a direct endpoint/readout forecast family over a narrower metric scope, while R19 is a process-family cascade model. If beating matched R10 remains mandatory, the next branch must be an explicitly labeled predictive readout layer on top of the conserved state model, not another hidden process knob pretending to be mechanistic.",
        "",
        "## Source Artifacts",
        "",
    ]
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Matched R10 Contract",
            "",
            "| Horizon | R10 reference | Carry-forward | Reference experiment | Metric scope |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for horizon in sorted(int(key) for key in r10_refs.keys()):
        ref = dict(r10_refs[str(horizon)] if str(horizon) in r10_refs else r10_refs[horizon])
        scope = ", ".join(str(item) for item in list(ref.get("reference_metric_scope") or []))
        lines.append(
            f"| {horizon}y | {_format_float(ref.get('reference_quarterly_mean_mae'))} | "
            f"{_format_float(ref.get('reference_carry_forward_mean_mae'))} | "
            f"`{ref.get('reference_experiment_id')}` | {scope} |"
        )
    lines.extend(
        [
            "",
            "R10 is therefore a valid predictive benchmark, but it is not evidence that the same equations identify cascade transition mechanisms. It forecasts the scored endpoints directly and scores only diagnosed stock, ART stock, and diagnosis flow.",
            "",
            "## R19 Annual Gate",
            "",
            "| Gate | Decision | Candidate MAE | Carry-forward MAE | Conservation residual |",
            "|---|---|---:|---:|---:|",
            f"| official annual conserved-head gate | `{annual.get('decision')}` | {_format_float(annual.get('candidate_mean_mae'))} | {_format_float(annual.get('carry_forward_mean_mae'))} | {_format_float(annual.get('annual_max_conservation_residual'))} |",
            "",
            "## Full Sentinel R19 vs Matched R10",
            "",
            "| Horizon | R19 R10-scope | Matched R10 | Delta | Carry-forward | Decision |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in all_rows:
        delta = _float_or_none(row.get("candidate_minus_r10_reference_mae"))
        decision = "beats R10" if delta is not None and delta < 0.0 else "fails R10"
        lines.append(
            f"| {row['horizon_years']}y | {_format_float(row.get('r10_comparable_candidate_mean_mae'))} | "
            f"{_format_float(row.get('r10_horizon_reference_mae'))} | {_format_float(delta)} | "
            f"{_format_float(row.get('carry_forward_mean_mae'))} | {decision} |"
        )
    lines.extend(
        [
            "",
            "## Program Route R19/R22/R23",
            "",
            "| Branch | Horizon | Full candidate MAE | R10-scope candidate | Matched R10 | Delta | Carry-forward |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    label_by_family = {
        "r19_joint_service_cascade_process": "R19",
        "r22_program_metric_coupled_process": "R22",
        "r23_recent_origin_program_coupled_process": "R23",
    }
    for row in program_rows:
        lines.append(
            f"| {label_by_family.get(str(row.get('family')), row.get('family'))} | {row['horizon_years']}y | "
            f"{_format_float(row.get('candidate_mean_mae'))} | "
            f"{_format_float(row.get('r10_comparable_candidate_mean_mae'))} | "
            f"{_format_float(row.get('r10_horizon_reference_mae'))} | "
            f"{_format_float(row.get('candidate_minus_r10_reference_mae'))} | "
            f"{_format_float(row.get('carry_forward_mean_mae'))} |"
        )
    lines.extend(
        [
            "",
            "## Scientific Interpretation",
            "",
            "- R19 is scientifically better than R18 and passes the annual conserved-head challenge, so it remains useful as a mechanistic research reference.",
            "- R22 proves service/back-half signal exists in the earlier support-cadence branch, but it worsens the h3 R10-scope ART path.",
            "- R23 proves a recent-origin guard can prevent that h3 ART regression, but the improvement is too small and it gives up the full program-route service gain.",
            "- The remaining R10 gap is not mainly a missing VL/suppression stock-flow equation; it is the advantage of an endpoint/readout family over the exact R10 scoring scope.",
            "",
            "## Next Experiment Contract",
            "",
            "R24 should not be promoted as a model. It is a fairness/readout audit. The next model branch, if the active goal still requires beating matched R10, should be a two-track hybrid:",
            "",
            "- Mechanistic track: keep R19 conserved states, annual mass balance, stock cone, and conditional VL/suppression gates.",
            "- Predictive track: add a train-origin endpoint readout head for the R10 metric scope only, labeled as predictive readout rather than transition mechanism.",
            "- Promotion rule: the predictive head must beat carry-forward and matched R10 on the same horizon-matched gate, while the mechanistic state remains valid under annual conservation and stock-cone checks.",
            "",
            "Ralph check: helpful. This audit stops the loop from spending more budget on low-yield selector variants and defines the only next branch that directly attacks the active blocker without mislabeling readout performance as mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def build_r24_matched_r10_fairness_audit(repo_root: Path | None = None) -> dict[str, Any]:
    repo = Path(repo_root) if repo_root else _repo_root()
    phase3 = _phase3_root(repo)
    audits = ensure_dir(phase3 / "artifacts" / "scientific_audits")
    r13_path = (
        phase3
        / "artifacts"
        / "runs"
        / "p3d-r19-joint-service-r13-queue-20260502-final-v3"
        / "analysis"
        / "r13_priority_experiment_results.json"
    )
    r10_path = (
        phase3
        / "artifacts"
        / "runs"
        / "p3d-r12-joint-annual-latent-monthly-20260501-s00"
        / "analysis"
        / "r10_horizon_matched_replay_report.json"
    )
    r22_path = audits / "phase3_r22_program_metric_coupled_diagnostic_results_20260502.json"
    r23_path = audits / "phase3_r23_recent_origin_program_coupling_diagnostic_results_20260502.json"
    r10_doc_path = phase3 / "R10_WINNER_DOCUMENTATION.md"

    r13_report = _require_dict(read_json(r13_path), r13_path)
    r10_report = _require_dict(read_json(r10_path), r10_path)
    r22_report = _require_dict(read_json(r22_path), r22_path)
    r23_report = _require_dict(read_json(r23_path), r23_path)

    annual = _result_by_experiment(r13_report, "R13-001")
    r13_050 = _result_by_experiment(r13_report, "R13-050")
    all_rows = _all_support_rows(r13_050)
    program_rows = _program_family_rows(r23_report)
    r10_refs = _r10_references(r10_report)
    worst_program_gap = max(
        (
            float(row["candidate_minus_r10_reference_mae"])
            for row in program_rows
            if _float_or_none(row.get("candidate_minus_r10_reference_mae")) is not None
        ),
        default=None,
    )
    five_year_all_gap = next(
        (
            row.get("candidate_minus_r10_reference_mae")
            for row in all_rows
            if int(row.get("horizon_years") or 0) == 5
        ),
        None,
    )

    report = {
        "schema_version": R24_SCHEMA_VERSION,
        "generated_at": "2026-05-02",
        "source_artifacts": {
            "r13_final_results": r13_path.as_posix(),
            "matched_r10_horizon_replay": r10_path.as_posix(),
            "r22_program_diagnostic": r22_path.as_posix(),
            "r23_program_diagnostic": r23_path.as_posix(),
            "r10_winner_documentation": r10_doc_path.as_posix(),
        },
        "matched_r10_reference_by_horizon": {str(key): value for key, value in sorted(r10_refs.items())},
        "official_annual_gate": {
            "experiment_id": annual.get("experiment_id"),
            "decision": annual.get("decision"),
            "candidate_family": annual.get("candidate_family"),
            "candidate_mean_mae": _float_or_none(annual.get("candidate_mean_mae")),
            "carry_forward_mean_mae": _float_or_none(annual.get("carry_forward_mean_mae")),
            "annual_status": annual.get("annual_status"),
            "annual_max_conservation_residual": _float_or_none(annual.get("annual_max_conservation_residual")),
            "stock_cone_violation_count": annual.get("stock_cone_violation_count"),
        },
        "all_support_r19_rows": all_rows,
        "program_route_rows": program_rows,
        "r22_decision": r22_report.get("decision"),
        "r22_blockers": list(r22_report.get("blockers") or []),
        "r23_decision": r23_report.get("decision"),
        "r23_blockers": list(r23_report.get("blockers") or []),
        "audit_decision": "do_not_promote_r24_as_model",
        "active_reference_after_audit": "r19_joint_service_cascade_process",
        "blockers": [
            "matched_r10_endpoint_readout_advantage",
            "program_route_h3_h5_still_fail_r10",
            "full_sentinel_h5_still_fails_r10",
        ],
        "worst_program_candidate_minus_r10": worst_program_gap,
        "full_sentinel_h5_candidate_minus_r10": _float_or_none(five_year_all_gap),
        "next_branch_contract": (
            "If matched-R10 superiority remains required, build a transparently predictive endpoint readout head "
            "on top of the conserved R19 state model; keep the mechanistic and predictive claims separate."
        ),
    }

    json_path = audits / "phase3_r24_matched_r10_fairness_audit_results_20260502.json"
    md_path = audits / "phase3_r24_matched_r10_fairness_audit_20260502.md"
    dashboard_path = audits / "phase3_r24_matched_r10_fairness_audit_dashboard_20260502.png"
    report["artifact_paths"] = {
        "results_json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    md_path.write_text(_markdown(report), encoding="utf-8")
    _write_dashboard(dashboard_path, report)
    return report


if __name__ == "__main__":
    build_r24_matched_r10_fairness_audit()
