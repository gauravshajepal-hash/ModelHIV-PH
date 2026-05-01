from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.phase3 import tr_v3_indicator_inventory_batch as inventory_batch
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


DEFAULT_INVENTORY_RUN_ID = "tr-v3-indicator-inventory-20260419-s00"
DEFAULT_ACTIVE_MONTHLY_RUN_ID = inventory_batch.DEFAULT_ACTIVE_MONTHLY_RUN_ID

RAW_PROGRAM_FIELD_CANONICAL_CANDIDATES = {
    "diagnosed_share": "candidate_first_class_canonical",
    "diagnosed_count": "canonical_alias_review",
    "on_art": "canonical_alias_review",
    "suppressed": "canonical_alias_review",
    "viral_load_tested": "canonical_alias_review",
    "diagnosed": "canonical_alias_review",
}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _aggregate_source_banks(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    banks: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        canonical = str(row.get("canonical_name") or "").strip()
        source_bank = str(row.get("source_bank") or "").strip()
        if canonical and source_bank:
            banks[canonical].add(source_bank)
    return banks


def _provenance_class(row: dict[str, Any], active_source_banks: set[str]) -> str:
    source_families = {str(value).strip() for value in str(row.get("source_families") or "").split("|") if str(value).strip()}
    national_numeric_obs = _to_int(row.get("national_numeric_obs"))
    if active_source_banks and active_source_banks.issubset({"phase15_harp_derived_indicators"}):
        if national_numeric_obs > 0:
            return "mixed_raw_and_derived"
        return "derived_numeric"
    if national_numeric_obs > 0:
        if source_families & {"observed_program_panel", "historical_metric_rows", "multinational_hiv_metric_rows", "historical_harp_panel"}:
            return "raw_source_numeric"
        if source_families & {"source_phase1_all_geo", "source_phase1_national"}:
            return "phase0_phase1_numeric"
    if row.get("current_status") == "unused_nonnumeric_source":
        return "placeholder_nonnumeric"
    if active_source_banks:
        return "derived_numeric"
    return "unknown"


def _recommended_action(row: dict[str, Any], provenance_class: str, active_source_banks: set[str]) -> tuple[str, str]:
    indicator_name = str(row.get("indicator_name") or "").strip()
    status = str(row.get("current_status") or "")
    cadence = str(row.get("cadence_class") or "")
    national_numeric_obs = _to_int(row.get("national_numeric_obs"))

    if status == "active_structural":
        if provenance_class in {"derived_numeric", "mixed_raw_and_derived"} and active_source_banks & {"phase15_harp_derived_indicators"}:
            return "derived_support_review", "active in the structural lane but at least part of the support comes from Phase 15 derived indicators"
        return "keep_active", "already active in the current structural kernel"

    if status == "evaluation_only":
        return "keep_evaluation_only", "high-support outcome or cascade head intentionally excluded from structural use"

    if status == "candidate_only":
        return "measurement_sidecar_only", "clean enough to measure but not stable enough as a live structural driver"

    if status == "subnational_only_numeric":
        return "aggregate_or_subnational_track", "numeric support exists but mostly below the national aggregation level"

    if provenance_class == "placeholder_nonnumeric":
        return "needs_extraction", "present in the repo as text or literature context but still missing numeric support"

    if status == "unused_numeric_source":
        if indicator_name in {"youth_cases_15_24_period", "art_median_age", "diagnosed_cases_cumulative", "median_cd4_at_enrollment", "new_diagnosed_cases_monthly"}:
            return "promote_measurement_anchor", "dense national numeric support makes this a strong measurement or anchor candidate"
        if indicator_name.startswith("wdi_") or indicator_name in {"annual_new_infections", "annual_aids_deaths"}:
            return "annual_anchor_candidate", "annual national anchor with usable support but not a likely monthly structural driver"
        if cadence in {"single_reading", "biannual", "sparse_annual", "monthly_sparse"}:
            return "sparse_anchor_candidate", "numeric but too sparse for a live structural role without special handling"
        if national_numeric_obs >= 12:
            return "promote_measurement_anchor", "national numeric support is strong enough for a first measurement-only promotion pass"
        return "low_priority_numeric", "numeric support exists but is too thin to prioritize before denser unused signals"

    return "defer", "not currently actionable beyond documentation"


def _raw_program_field_action(row: dict[str, Any]) -> tuple[str, str]:
    indicator_name = str(row.get("indicator_name") or "").strip()
    if indicator_name in RAW_PROGRAM_FIELD_CANONICAL_CANDIDATES:
        action = RAW_PROGRAM_FIELD_CANONICAL_CANDIDATES[indicator_name]
        if action == "candidate_first_class_canonical":
            return action, "monthly dense ratio series with direct field support and no current first-class canonical"
        return action, "raw HARP field overlaps an existing canonical but should be reviewed as an explicit alias or merge path"
    return "raw_field_reference_only", "raw field is informative but does not obviously justify a new canonical by itself"


def _plot_action_counts(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    counter = Counter(str(row.get("recommended_action") or "") for row in rows)
    labels = [label for label, _ in counter.most_common()]
    values = [counter[label] for label in labels]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(labels, values, color="#1f77b4")
    ax.set_title("Indicator triage recommendation counts")
    ax.set_ylabel("indicator count")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_top_candidates(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    candidates = [
        row
        for row in rows
        if str(row.get("recommended_action") or "") in {"promote_measurement_anchor", "annual_anchor_candidate", "sparse_anchor_candidate"}
    ]
    ranked = sorted(candidates, key=lambda row: (_to_float(row.get("national_numeric_obs")), str(row.get("indicator_name") or "")), reverse=True)[:15]
    if not ranked:
        return
    labels = [str(row.get("indicator_name") or "") for row in ranked]
    values = [_to_float(row.get("national_numeric_obs")) for row in ranked]
    colors = []
    for row in ranked:
        action = str(row.get("recommended_action") or "")
        colors.append(
            {
                "promote_measurement_anchor": "#2ca02c",
                "annual_anchor_candidate": "#9467bd",
                "sparse_anchor_candidate": "#ff7f0e",
            }.get(action, "#7f7f7f")
        )
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.35 * len(labels))))
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_title("Top measurement and anchor promotion candidates")
    ax.set_xlabel("national numeric observations")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_provenance_status(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    status_labels = sorted({str(row.get("current_status") or "") for row in rows if str(row.get("current_status") or "")})
    provenance_labels = sorted({str(row.get("provenance_class") or "") for row in rows if str(row.get("provenance_class") or "")})
    matrix = np.zeros((len(provenance_labels), len(status_labels)), dtype=np.float32)
    lookup_status = {label: idx for idx, label in enumerate(status_labels)}
    lookup_prov = {label: idx for idx, label in enumerate(provenance_labels)}
    for row in rows:
        status = str(row.get("current_status") or "")
        prov = str(row.get("provenance_class") or "")
        if status in lookup_status and prov in lookup_prov:
            matrix[lookup_prov[prov], lookup_status[status]] += 1.0
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(status_labels)), labels=status_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(provenance_labels)), labels=provenance_labels)
    ax.set_title("Indicator provenance by current status")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_diagnosed_share_audit(source_run_dir: Path, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    diagnosis_rows = inventory_batch._load_rows(source_run_dir / "harp_archive" / "diagnosis_flow_points.json")
    if not diagnosis_rows:
        return
    diagnosis_rows = sorted(diagnosis_rows, key=lambda row: inventory_batch._extract_time_label(row))
    months = [inventory_batch._extract_time_label(row) for row in diagnosis_rows]
    diagnosed_share = [inventory_batch._safe_float(row.get("diagnosed_share")) for row in diagnosis_rows]
    diagnosed_count = [inventory_batch._safe_float(row.get("diagnosed_count")) for row in diagnosis_rows]
    estimated_plhiv = [inventory_batch._safe_float(row.get("estimated_plhiv")) for row in diagnosis_rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    positions = np.arange(len(months), dtype=np.int32)
    axes[0].plot(positions, diagnosed_share, color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("diagnosed share")
    axes[0].set_title("Diagnosed share raw-field audit")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(positions, diagnosed_count, color="#d62728", linewidth=1.5, label="diagnosed_count")
    axes[1].plot(positions, estimated_plhiv, color="#2ca02c", linewidth=1.2, label="estimated_plhiv")
    axes[1].set_ylabel("count")
    axes[1].set_xlabel("month index")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, tuple, set)):
                    normalized[key] = "|".join(str(item) for item in value)
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def _report_markdown(payload: dict[str, Any]) -> str:
    def _listish(value: Any) -> list[str]:
        if isinstance(value, str):
            return [part for part in value.split("|") if part]
        if isinstance(value, (list, tuple, set)):
            return [str(part) for part in value if str(part)]
        return []

    top_anchor = list(payload.get("top_anchor_candidates") or [])
    top_derived = list(payload.get("top_derived_reviews") or [])
    top_extract = list(payload.get("top_extraction_needed") or [])
    raw_field_candidates = list(payload.get("raw_field_candidates") or [])
    lines = [
        "# Indicator Triage Audit",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Inventory run: `{payload['inventory_run_id']}`",
        f"- Active monthly run: `{payload['active_monthly_run_id']}`",
        f"- Source run: `{payload['source_run_id']}`",
        "",
        "## Recommendation Counts",
        "",
    ]
    for action, count in dict(payload.get("recommendation_counts") or {}).items():
        lines.append(f"- `{action}`: `{count}`")
    lines.extend(["", "## Highest-Value Anchor Candidates", ""])
    for row in top_anchor:
        lines.append(
            f"- `{row['indicator_name']}`: action `{row['recommended_action']}`, national numeric obs `{row['national_numeric_obs']}`, cadence `{row['cadence_class']}`"
        )
    if not top_anchor:
        lines.append("- none")
    lines.extend(["", "## Derived Indicators That Need Review", ""])
    for row in top_derived:
        lines.append(
            f"- `{row['indicator_name']}`: active status `{row['current_status']}`, source banks `{', '.join(_listish(row.get('active_source_banks')))}"
        )
    if not top_derived:
        lines.append("- none")
    lines.extend(["", "## Placeholder Indicators That Still Need Extraction", ""])
    for row in top_extract:
        lines.append(
            f"- `{row['indicator_name']}`: candidate blocks `{', '.join(_listish(row.get('source_candidate_blocks'))) or 'none'}`"
        )
    if not top_extract:
        lines.append("- none")
    lines.extend(["", "## Raw Program Field Candidates", ""])
    for row in raw_field_candidates:
        lines.append(
            f"- `{row['indicator_name']}`: action `{row['recommended_action']}`, numeric obs `{row['numeric_obs_count']}`, cadence `{row['cadence_class']}`"
        )
    if not raw_field_candidates:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Diagnosed Share Decision",
            "",
            f"- Decision: `{payload.get('diagnosed_share_decision') or 'none'}`",
            f"- Rationale: {payload.get('diagnosed_share_rationale') or 'none'}",
            "",
            "## Artifacts",
            "",
            "- `analysis/indicator_triage.csv`",
            "- `analysis/raw_field_triage.csv`",
            "- `analysis/triage_action_counts.png`",
            "- `analysis/top_anchor_candidates.png`",
            "- `analysis/provenance_by_status.png`",
            "- `analysis/diagnosed_share_audit.png`",
            "",
        ]
    )
    return "\n".join(lines)


def run_tr_v3_indicator_triage_batch(
    *,
    run_id: str,
    inventory_run_id: str = DEFAULT_INVENTORY_RUN_ID,
    active_monthly_run_id: str = DEFAULT_ACTIVE_MONTHLY_RUN_ID,
) -> dict[str, Any]:
    inventory_run_dir = ROOT_DIR / "artifacts" / "runs" / str(inventory_run_id)
    active_monthly_run_dir = ROOT_DIR / "artifacts" / "runs" / str(active_monthly_run_id)
    if not inventory_run_dir.exists():
        raise FileNotFoundError(f"inventory run does not exist: {inventory_run_dir}")
    if not active_monthly_run_dir.exists():
        raise FileNotFoundError(f"active monthly run does not exist: {active_monthly_run_dir}")

    inventory_report = dict(read_json(inventory_run_dir / "analysis" / "tr_v3_indicator_inventory_batch_report.json", default={}))
    source_run_id = str(inventory_report.get("source_run_id") or "").strip()
    if not source_run_id:
        raise ValueError("inventory report is missing source_run_id")
    source_run_dir = ROOT_DIR / "artifacts" / "runs" / source_run_id
    if not source_run_dir.exists():
        raise FileNotFoundError(f"source run does not exist: {source_run_dir}")

    inventory_rows = _read_csv_rows(inventory_run_dir / "analysis" / "indicator_inventory.csv")
    raw_field_rows = _read_csv_rows(inventory_run_dir / "analysis" / "raw_program_field_inventory.csv")
    active_structural_rows = inventory_batch._load_rows(active_monthly_run_dir / "phase1" / "normalized_subparameters.json")
    active_full_eval_rows = inventory_batch._load_rows(active_monthly_run_dir / "phase1" / "normalized_subparameters_full_for_evaluation.json")
    structural_banks = _aggregate_source_banks(active_structural_rows)
    evaluation_banks = _aggregate_source_banks(active_full_eval_rows)

    triage_rows: list[dict[str, Any]] = []
    for row in inventory_rows:
        indicator_name = str(row.get("indicator_name") or "").strip()
        active_source_banks = sorted(structural_banks.get(indicator_name) or evaluation_banks.get(indicator_name) or set())
        provenance_class = _provenance_class(row, set(active_source_banks))
        recommended_action, rationale = _recommended_action(row, provenance_class, set(active_source_banks))
        triage_rows.append(
            {
                **row,
                "active_source_banks": active_source_banks,
                "provenance_class": provenance_class,
                "recommended_action": recommended_action,
                "triage_rationale": rationale,
            }
        )

    raw_field_triage_rows: list[dict[str, Any]] = []
    for row in raw_field_rows:
        action, rationale = _raw_program_field_action(row)
        raw_field_triage_rows.append(
            {
                **row,
                "provenance_class": "raw_program_field_numeric",
                "recommended_action": action,
                "triage_rationale": rationale,
            }
        )

    recommendation_counts = Counter(str(row.get("recommended_action") or "") for row in triage_rows)
    top_anchor_candidates = sorted(
        [row for row in triage_rows if str(row.get("recommended_action") or "") in {"promote_measurement_anchor", "annual_anchor_candidate", "sparse_anchor_candidate"}],
        key=lambda row: (_to_float(row.get("national_numeric_obs")), str(row.get("indicator_name") or "")),
        reverse=True,
    )[:15]
    top_derived_reviews = sorted(
        [row for row in triage_rows if str(row.get("recommended_action") or "") == "derived_support_review"],
        key=lambda row: (len(list(row.get("active_source_banks") or [])), str(row.get("indicator_name") or "")),
        reverse=True,
    )[:15]
    top_extraction_needed = sorted(
        [row for row in triage_rows if str(row.get("recommended_action") or "") == "needs_extraction"],
        key=lambda row: (str(row.get("indicator_name") or "")),
    )[:15]
    raw_field_candidates = sorted(
        [row for row in raw_field_triage_rows if str(row.get("recommended_action") or "") != "raw_field_reference_only"],
        key=lambda row: (_to_float(row.get("numeric_obs_count")), str(row.get("indicator_name") or "")),
        reverse=True,
    )

    diagnosed_share_decision = "defer"
    diagnosed_share_rationale = "raw diagnosed_share field was not found"
    for row in raw_field_triage_rows:
        if str(row.get("indicator_name") or "") == "diagnosed_share":
            diagnosed_share_decision = str(row.get("recommended_action") or "")
            diagnosed_share_rationale = str(row.get("triage_rationale") or "")
            break

    run_dir = ROOT_DIR / "artifacts" / "runs" / str(run_id)
    analysis_dir = ensure_dir(run_dir / "analysis")
    _write_csv(analysis_dir / "indicator_triage.csv", triage_rows)
    _write_csv(analysis_dir / "raw_field_triage.csv", raw_field_triage_rows)
    _plot_action_counts(triage_rows, analysis_dir / "triage_action_counts.png")
    _plot_top_candidates(triage_rows, analysis_dir / "top_anchor_candidates.png")
    _plot_provenance_status(triage_rows, analysis_dir / "provenance_by_status.png")
    _plot_diagnosed_share_audit(source_run_dir, analysis_dir / "diagnosed_share_audit.png")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "inventory_run_id": str(inventory_run_id),
        "active_monthly_run_id": str(active_monthly_run_id),
        "source_run_id": str(source_run_id),
        "recommendation_counts": {str(name): int(value) for name, value in recommendation_counts.most_common()},
        "top_anchor_candidates": top_anchor_candidates,
        "top_derived_reviews": top_derived_reviews,
        "top_extraction_needed": top_extraction_needed,
        "raw_field_candidates": raw_field_candidates,
        "diagnosed_share_decision": diagnosed_share_decision,
        "diagnosed_share_rationale": diagnosed_share_rationale,
        "artifacts": {
            "indicator_triage_csv": str(analysis_dir / "indicator_triage.csv"),
            "raw_field_triage_csv": str(analysis_dir / "raw_field_triage.csv"),
            "action_counts_plot": str(analysis_dir / "triage_action_counts.png"),
            "top_candidates_plot": str(analysis_dir / "top_anchor_candidates.png"),
            "provenance_plot": str(analysis_dir / "provenance_by_status.png"),
            "diagnosed_share_plot": str(analysis_dir / "diagnosed_share_audit.png"),
        },
    }
    write_json(analysis_dir / "tr_v3_indicator_triage_batch_report.json", payload)
    (analysis_dir / "tr_v3_indicator_triage_batch_report.md").write_text(_report_markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage available indicators into raw-source, derived, placeholder, and action buckets before additional structural modeling.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--inventory-run-id", default=DEFAULT_INVENTORY_RUN_ID)
    parser.add_argument("--active-monthly-run-id", default=DEFAULT_ACTIVE_MONTHLY_RUN_ID)
    args = parser.parse_args()
    run_tr_v3_indicator_triage_batch(
        run_id=str(args.run_id),
        inventory_run_id=str(args.inventory_run_id),
        active_monthly_run_id=str(args.active_monthly_run_id),
    )


if __name__ == "__main__":
    main()
