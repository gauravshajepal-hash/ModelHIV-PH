from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epigraph_ph.phase3 import tr_v3_monthly_phase2_lane_batch as monthly_lane
from epigraph_ph.phase3 import tr_v3_phase2_champion_equivalence_batch as champion_equivalence
from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


DEFAULT_BASELINE_RUN_ID = champion_equivalence.DEFAULT_BASELINE_RUN_ID
DEFAULT_CANDIDATE_RUN_ID = champion_equivalence.DEFAULT_CANDIDATE_RUN_ID
TESTING_FAMILY_CANONICALS: tuple[str, ...] = (
    "annual_hiv_tests_volume_per_100k",
    "prep_people_receiving_per_100k",
    "hiv_test_positivity_percent",
    "late_hiv_diagnosis_percent",
    "unaids_known_status_share_percent",
)
NON_TESTING_SCENARIOS: tuple[str, ...] = ("disruption_recovery", "mobility_spike")
PRESERVED_FAMILIES: tuple[str, ...] = ("care_access_continuity", "suppression_capacity", "mobility_exposure_pressure")


def _monthly_report(run_id: str) -> dict[str, Any]:
    path = ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload:
        raise FileNotFoundError(f"missing monthly lane report: {path}")
    return payload


def _max_abs_non_testing_delta(seed_report: dict[str, Any]) -> float:
    max_value = 0.0
    for row in list(seed_report.get("terminal_delta_rows") or []):
        if str(row.get("scenario") or "") not in set(NON_TESTING_SCENARIOS):
            continue
        for metric_name in seeded.METRIC_PLOT_ORDER:
            max_value = max(max_value, abs(float(row.get(f"{metric_name}_delta") or 0.0)))
    return float(max_value)


def _plot_demotion_scenario_summary(summary_rows: list[dict[str, Any]], path: Path) -> None:
    if not summary_rows:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No scenario summary rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    labels = [str(row["scenario_group"]) for row in summary_rows]
    values = [float(row["max_abs_terminal_delta"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values, color=["#c44e52", "#4c72b0"])
    ax.set_ylabel("Max absolute terminal delta")
    ax.set_title("Demoted testing-family scenario amplitudes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Phase 2 Champion Testing Demotion Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Baseline monthly run: `{payload['baseline_monthly_run_id']}`",
        f"- Source merged monthly run: `{payload['candidate_monthly_run_id']}`",
        f"- Demoted monthly run: `{payload['demoted_monthly_run_id']}`",
        f"- Decision: `{payload['decision']}`",
        "",
        "## Decision summary",
        "",
        f"- Testing family removed from structural substrate: `{payload['decision_summary']['testing_family_removed']}`",
        f"- Minimum preserved-family Jaccard: `{float(payload['decision_summary']['min_preserved_family_jaccard']):.3f}`",
        f"- Max testing-scenario abs terminal delta: `{float(payload['decision_summary']['max_testing_abs_terminal_delta']):.3f}`",
        f"- Max non-testing abs terminal delta: `{float(payload['decision_summary']['max_non_testing_abs_terminal_delta']):.3f}`",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in dict(payload.get("artifacts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def run_tr_v3_phase2_champion_testing_demotion_batch(
    *,
    run_id: str,
    baseline_monthly_run_id: str = DEFAULT_BASELINE_RUN_ID,
    candidate_monthly_run_id: str = DEFAULT_CANDIDATE_RUN_ID,
) -> dict[str, Any]:
    candidate_report = _monthly_report(candidate_monthly_run_id)
    demoted_monthly_run_id = f"{run_id}-monthly"
    equivalence_run_id = f"{run_id}-equivalence"

    monthly_lane.run_tr_v3_monthly_phase2_lane_batch(
        run_id=demoted_monthly_run_id,
        source_run_id=str(candidate_report.get("source_run_id") or monthly_lane.DEFAULT_SOURCE_RUN_ID),
        coverage_run_id=str(candidate_report.get("coverage_run_id") or "") or None,
        start_month=str(candidate_report.get("start_month") or monthly_lane.DEFAULT_START_MONTH),
        structural_excluded_canonicals=TESTING_FAMILY_CANONICALS,
    )
    equivalence_payload = champion_equivalence.run_tr_v3_phase2_champion_equivalence_batch(
        run_id=equivalence_run_id,
        baseline_monthly_run_id=baseline_monthly_run_id,
        candidate_monthly_run_id=demoted_monthly_run_id,
    )
    candidate_seeded = champion_equivalence._seeded_report(f"{equivalence_run_id}-candidate-seeded")
    testing_family_row = next(
        (row for row in list(equivalence_payload.get("family_overlap_rows") or []) if str(row.get("block_family") or "") == "testing_family"),
        {},
    )
    preserved_rows = [
        row
        for row in list(equivalence_payload.get("family_overlap_rows") or [])
        if str(row.get("block_family") or "") in set(PRESERVED_FAMILIES)
    ]
    max_testing_abs_terminal_delta = max(
        (float(row.get("candidate_abs_delta") or 0.0) for row in list(equivalence_payload.get("testing_scenario_rows") or [])),
        default=0.0,
    )
    max_non_testing_abs_terminal_delta = _max_abs_non_testing_delta(candidate_seeded)
    min_preserved_family_jaccard = min((float(row.get("indicator_jaccard") or 0.0) for row in preserved_rows), default=0.0)
    testing_family_removed = bool(int(testing_family_row.get("candidate_indicator_count") or 0) == 0)
    decision = (
        "keep_testing_as_measurement_sidecar"
        if testing_family_removed and min_preserved_family_jaccard >= 0.75 and max_testing_abs_terminal_delta <= 1e-6 and max_non_testing_abs_terminal_delta > 0.0
        else "revisit_testing_demotion"
    )

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / str(run_id) / "analysis")
    scenario_summary_rows = [
        {"scenario_group": "testing_scenarios", "max_abs_terminal_delta": float(max_testing_abs_terminal_delta)},
        {"scenario_group": "non_testing_scenarios", "max_abs_terminal_delta": float(max_non_testing_abs_terminal_delta)},
    ]
    scenario_plot = analysis_dir / "testing_demotion_scenario_summary.png"
    _plot_demotion_scenario_summary(scenario_summary_rows, scenario_plot)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "baseline_monthly_run_id": str(baseline_monthly_run_id),
        "candidate_monthly_run_id": str(candidate_monthly_run_id),
        "demoted_monthly_run_id": str(demoted_monthly_run_id),
        "equivalence_run_id": str(equivalence_run_id),
        "excluded_testing_canonicals": list(TESTING_FAMILY_CANONICALS),
        "decision": decision,
        "decision_summary": {
            "testing_family_removed": bool(testing_family_removed),
            "min_preserved_family_jaccard": float(min_preserved_family_jaccard),
            "max_testing_abs_terminal_delta": float(max_testing_abs_terminal_delta),
            "max_non_testing_abs_terminal_delta": float(max_non_testing_abs_terminal_delta),
        },
        "scenario_summary_rows": scenario_summary_rows,
        "artifacts": {
            "scenario_summary_plot": scenario_plot.name,
            "demoted_monthly_report": str(Path(demoted_monthly_run_id) / "analysis" / "tr_v3_monthly_phase2_lane_batch_report.md"),
            "equivalence_report": str(Path(equivalence_run_id) / "analysis" / "tr_v3_phase2_champion_equivalence_batch_report.md"),
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_champion_testing_demotion_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_champion_testing_demotion_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demote the merged testing family to measurement-only and re-audit against the current predictive champions.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-monthly-run-id", default=DEFAULT_BASELINE_RUN_ID)
    parser.add_argument("--candidate-monthly-run-id", default=DEFAULT_CANDIDATE_RUN_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_champion_testing_demotion_batch(
        run_id=str(args.run_id),
        baseline_monthly_run_id=str(args.baseline_monthly_run_id),
        candidate_monthly_run_id=str(args.candidate_monthly_run_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
