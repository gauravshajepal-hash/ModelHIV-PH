from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, utc_now_iso, write_json
from epigraph_ph.validate.extraction_quality_audit import _harp_quality_audit, _write_harp_source_adjudication_table


DEFAULT_ARCHIVE_RUN_ID = "phase2-replay-source-20260414-s00"
CORE_MONTHLY_METRICS = (
    "new_diagnosed_cases_period",
    "deaths_reported_period",
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
)


def _month_ordinal(month_label: str) -> int:
    year_str, month_str = str(month_label).split("-", 1)
    return int(year_str) * 12 + int(month_str) - 1


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _sorted_month_keys(months: set[str]) -> list[str]:
    return sorted((str(month) for month in months if isinstance(month, str) and len(str(month)) == 7), key=_month_ordinal)


def _metric_inventory_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row.get("metric_name") or row.get("canonical_name") or "")].append(row)
    summary_rows: list[dict[str, Any]] = []
    for metric_name, rows in grouped.items():
        times = sorted({str(row.get("time") or "") for row in rows if str(row.get("time") or "")})
        source_ids = sorted({str(row.get("source_id") or "") for row in rows if str(row.get("source_id") or "")})
        units = sorted({str(row.get("unit") or row.get("normalized_unit") or "") for row in rows if str(row.get("unit") or row.get("normalized_unit") or "")})
        temporal_precisions = sorted({str(row.get("temporal_precision") or row.get("series_kind") or "") for row in rows if str(row.get("temporal_precision") or row.get("series_kind") or "")})
        measurement_classes = sorted({str(row.get("measurement_class") or "") for row in rows if str(row.get("measurement_class") or "")})
        summary_rows.append(
            {
                "metric_name": metric_name,
                "row_count": len(rows),
                "source_count": len(source_ids),
                "first_time": times[0] if times else "",
                "last_time": times[-1] if times else "",
                "units": "|".join(units),
                "temporal_precisions": "|".join(temporal_precisions),
                "measurement_classes": "|".join(measurement_classes),
                "example_source_ids": "|".join(source_ids[:5]),
            }
        )
    return sorted(summary_rows, key=lambda row: (-int(row["row_count"]), str(row["metric_name"])))


def _source_inventory_rows(metric_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row.get("source_id") or "")].append(row)
    manifest_by_id = {str(row.get("source_id") or ""): row for row in manifest_rows}
    summary_rows: list[dict[str, Any]] = []
    for source_id, rows in grouped.items():
        times = sorted({str(row.get("time") or "") for row in rows if str(row.get("time") or "")})
        metrics = Counter(str(row.get("metric_name") or row.get("canonical_name") or "") for row in rows)
        example_metrics = "|".join(metric for metric, _count in metrics.most_common(5))
        source_labels = {str(row.get("source_label") or "") for row in rows if str(row.get("source_label") or "")}
        source_urls = {str(row.get("source_url") or "") for row in rows if str(row.get("source_url") or "")}
        manifest_row = manifest_by_id.get(source_id, {})
        summary_rows.append(
            {
                "source_id": source_id,
                "row_count": len(rows),
                "metric_count": len(metrics),
                "first_time": times[0] if times else "",
                "last_time": times[-1] if times else "",
                "source_label": next(iter(sorted(source_labels)), str(manifest_row.get("label") or "")),
                "source_kind": str(manifest_row.get("source_kind") or ""),
                "source_url": next(iter(sorted(source_urls)), str(manifest_row.get("source_url") or "")),
                "example_metrics": example_metrics,
            }
        )
    return sorted(summary_rows, key=lambda row: (-int(row["row_count"]), str(row["source_id"])))


def _monthly_core_table(
    observed_program_rows: list[dict[str, Any]],
    diagnosis_flow_points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    month_rows: dict[str, dict[str, Any]] = defaultdict(dict)
    months: set[str] = set()
    observed_metrics = set(CORE_MONTHLY_METRICS) - {"new_diagnosed_cases_period"}
    for row in observed_program_rows:
        metric_name = str(row.get("metric_name") or "")
        month = str(row.get("time") or row.get("period_end") or "")
        if metric_name not in observed_metrics or len(month) != 7:
            continue
        numeric_value = _safe_float(row.get("value"))
        if numeric_value is None:
            continue
        months.add(month)
        target = month_rows[month]
        target["month"] = month
        target[metric_name] = numeric_value
        target[f"{metric_name}_source_id"] = str(row.get("source_id") or "")
        target[f"{metric_name}_source_label"] = str(row.get("source_label") or "")
    for row in diagnosis_flow_points:
        month = str(row.get("effective_month") or row.get("month") or "")
        if len(month) != 7:
            continue
        numeric_value = _safe_float(row.get("diagnosed_count"))
        if numeric_value is None:
            continue
        months.add(month)
        target = month_rows[month]
        target["month"] = month
        target["new_diagnosed_cases_period"] = numeric_value
        target["new_diagnosed_cases_period_source_id"] = str(row.get("source_id") or "")
        target["new_diagnosed_cases_period_source_label"] = str(row.get("source_label") or "")
    table: list[dict[str, Any]] = []
    for month in _sorted_month_keys(months):
        row = dict(month_rows.get(month) or {})
        row["month"] = month
        diagnosed = _safe_float(row.get("diagnosed_plhiv"))
        on_art = _safe_float(row.get("alive_on_art"))
        tested = _safe_float(row.get("tested_for_viral_load"))
        suppressed = _safe_float(row.get("virally_suppressed"))
        row["art_uptake_rate"] = round(on_art / diagnosed, 6) if diagnosed and diagnosed > 0.0 and on_art is not None else ""
        row["viral_suppression_rate_tested"] = round(suppressed / tested, 6) if tested and tested > 0.0 and suppressed is not None else ""
        row["viral_suppression_rate_on_art"] = round(suppressed / on_art, 6) if on_art and on_art > 0.0 and suppressed is not None else ""
        table.append(row)
    return table


def _annual_cascade_table(panel_rows: list[dict[str, Any]], program_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annual: dict[str, dict[str, Any]] = {}
    for row in panel_rows:
        year = str(row.get("year") or "")
        if not year:
            continue
        annual.setdefault(year, {"year": year}).update(
            {
                "time": str(row.get("time") or ""),
                "diagnosed_plhiv": _safe_float(row.get("diagnosed_plhiv")) or "",
                "estimated_plhiv": _safe_float(row.get("estimated_plhiv")) or "",
                "alive_on_art": _safe_float(row.get("alive_on_art")) or "",
                "tested_for_viral_load": _safe_float(row.get("tested_for_viral_load")) or "",
                "virally_suppressed": _safe_float(row.get("virally_suppressed")) or "",
            }
        )
    for row in program_points:
        month = str(row.get("month") or row.get("effective_month") or "")
        year = month[:4] if len(month) >= 4 else ""
        if not year:
            continue
        target = annual.setdefault(year, {"year": year})
        target["program_point_time"] = month
        target["program_point_label"] = str(row.get("label") or row.get("source_label") or "")
        for source_name, target_name in (
            ("diagnosed", "program_point_diagnosed"),
            ("estimated_plhiv", "program_point_estimated_plhiv"),
            ("on_art", "program_point_alive_on_art"),
            ("viral_load_tested", "program_point_tested_for_viral_load"),
            ("suppressed", "program_point_virally_suppressed"),
        ):
            target[target_name] = _safe_float(row.get(source_name)) or ""
    rows: list[dict[str, Any]] = []
    for year in sorted(annual.keys()):
        row = dict(annual[year])
        diagnosed = _safe_float(row.get("diagnosed_plhiv"))
        estimated = _safe_float(row.get("estimated_plhiv"))
        on_art = _safe_float(row.get("alive_on_art"))
        suppressed = _safe_float(row.get("virally_suppressed"))
        row["diagnosed_share"] = round(diagnosed / estimated, 6) if estimated and estimated > 0.0 and diagnosed is not None else ""
        row["art_share_of_diagnosed"] = round(on_art / diagnosed, 6) if diagnosed and diagnosed > 0.0 and on_art is not None else ""
        row["suppressed_share_of_art"] = round(suppressed / on_art, 6) if on_art and on_art > 0.0 and suppressed is not None else ""
        rows.append(row)
    return rows


def _source_regime(source_id: str) -> str:
    value = str(source_id or "")
    if value.startswith("doh_hiv_sti_"):
        return "doh_archive"
    if value.startswith("doh_official_cascade_ground_truth"):
        return "official_ground_truth"
    if value.startswith("core_team_"):
        return "core_team"
    if value.startswith("curated_historical_harp"):
        return "curated_harp_panel"
    return value


def _source_transition_rows(core_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    last_source_by_metric: dict[str, tuple[str, str, str]] = {}
    for row in sorted(core_table, key=lambda item: str(item.get("month") or "")):
        month = str(row.get("month") or "")
        for metric_name in CORE_MONTHLY_METRICS:
            if row.get(metric_name, "") in ("", None):
                continue
            source_id = str(row.get(f"{metric_name}_source_id") or "")
            source_label = str(row.get(f"{metric_name}_source_label") or "")
            if not source_id:
                continue
            source_regime = _source_regime(source_id)
            previous = last_source_by_metric.get(metric_name)
            if previous is not None and previous[2] != source_regime:
                transitions.append(
                    {
                        "metric_name": metric_name,
                        "month": month,
                        "from_source_id": previous[0],
                        "to_source_id": source_id,
                        "from_source_label": previous[1],
                        "to_source_label": source_label,
                        "from_source_regime": previous[2],
                        "to_source_regime": source_regime,
                    }
                )
            last_source_by_metric[metric_name] = (source_id, source_label, source_regime)
    return transitions


def _monthly_density_rows(observed_program_rows: list[dict[str, Any]], diagnosis_flow_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in observed_program_rows:
        month = str(row.get("time") or row.get("period_end") or "")
        if len(month) == 7:
            counts[month] += 1
    for row in diagnosis_flow_points:
        month = str(row.get("effective_month") or row.get("month") or "")
        if len(month) == 7:
            counts[month] += 1
    return [{"month": month, "row_count": counts[month]} for month in _sorted_month_keys(set(counts.keys()))]


def _metric_jump_thresholds(core_table: list[dict[str, Any]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for metric_name in CORE_MONTHLY_METRICS:
        diffs: list[float] = []
        previous_value: float | None = None
        for row in core_table:
            raw = row.get(metric_name, "")
            if raw in ("", None):
                continue
            value = float(raw)
            if previous_value is not None and previous_value > 0.0 and value > 0.0:
                diffs.append(abs(math.log(value / previous_value)))
            previous_value = value
        if not diffs:
            thresholds[metric_name] = float("inf")
            continue
        med = median(diffs)
        deviations = [abs(item - med) for item in diffs]
        mad = median(deviations) if deviations else 0.0
        thresholds[metric_name] = max(0.35, med + 3.0 * max(mad, 0.05))
    return thresholds


def _suspicious_event_rows(core_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = _metric_jump_thresholds(core_table)
    events: list[dict[str, Any]] = []
    previous_state: dict[str, dict[str, Any]] = {}
    for row in sorted(core_table, key=lambda item: str(item.get("month") or "")):
        month = str(row.get("month") or "")
        for metric_name in CORE_MONTHLY_METRICS:
            raw_value = row.get(metric_name, "")
            source_id = str(row.get(f"{metric_name}_source_id") or "")
            source_regime = _source_regime(source_id) if source_id else ""
            current_present = raw_value not in ("", None)
            current_value = float(raw_value) if current_present else None
            state = previous_state.get(metric_name, {})
            previous_present = bool(state.get("present"))
            previous_value = state.get("value")
            previous_source_id = str(state.get("source_id") or "")
            previous_source_regime = str(state.get("source_regime") or "")

            flags: list[str] = []
            score = 0
            log_change = ""
            if previous_present and not current_present:
                flags.append("gap_open")
                score += 2
            if not previous_present and current_present:
                flags.append("gap_close")
                score += 2
            if previous_present and current_present and previous_source_regime and source_regime and previous_source_regime != source_regime:
                flags.append("source_regime_handoff")
                score += 3
            if (
                previous_present
                and current_present
                and previous_value not in (None, 0.0)
                and current_value not in (None, 0.0)
                and previous_value > 0.0
                and current_value > 0.0
            ):
                log_change_value = abs(math.log(current_value / previous_value))
                log_change = round(log_change_value, 6)
                if log_change_value >= thresholds.get(metric_name, float("inf")):
                    flags.append("large_jump")
                    score += 4

            if flags:
                events.append(
                    {
                        "month": month,
                        "metric_name": metric_name,
                        "score": score,
                        "flags": "|".join(flags),
                        "previous_value": previous_value if previous_value is not None else "",
                        "current_value": current_value if current_value is not None else "",
                        "log_change_abs": log_change,
                        "previous_source_id": previous_source_id,
                        "current_source_id": source_id,
                        "previous_source_regime": previous_source_regime,
                        "current_source_regime": source_regime,
                    }
                )

            previous_state[metric_name] = {
                "present": current_present,
                "value": current_value,
                "source_id": source_id,
                "source_regime": source_regime,
            }
    return events


def _top_suspicious_month_rows(event_rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in event_rows:
        month = str(row.get("month") or "")
        target = grouped.setdefault(
            month,
            {
                "month": month,
                "total_score": 0,
                "event_count": 0,
                "metrics": [],
                "flags": set(),
            },
        )
        target["total_score"] += int(row.get("score") or 0)
        target["event_count"] += 1
        metric_name = str(row.get("metric_name") or "")
        if metric_name:
            target["metrics"].append(metric_name)
        for flag in str(row.get("flags") or "").split("|"):
            if flag:
                target["flags"].add(flag)
    rows = []
    for month, row in grouped.items():
        rows.append(
            {
                "month": month,
                "total_score": row["total_score"],
                "event_count": row["event_count"],
                "metrics": "|".join(sorted(set(row["metrics"]))),
                "flags": "|".join(sorted(row["flags"])),
            }
        )
    rows.sort(key=lambda item: (-int(item["total_score"]), -int(item["event_count"]), str(item["month"])))
    return rows[:limit]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_core_series(core_table: list[dict[str, Any]], output_path: Path) -> str | None:
    if not core_table:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    months = [str(row.get("month") or "") for row in core_table]
    x = list(range(len(months)))
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes_flat = list(axes.flatten())
    for index, metric_name in enumerate(CORE_MONTHLY_METRICS):
        values = [
            float(row.get(metric_name))
            if row.get(metric_name, "") not in ("", None)
            else float("nan")
            for row in core_table
        ]
        axis = axes_flat[index]
        axis.plot(x, values, linewidth=1.8)
        axis.set_title(metric_name)
        axis.grid(alpha=0.25)
    tick_step = max(1, len(x) // 10)
    tick_positions = list(range(0, len(x), tick_step))
    tick_labels = [months[pos] for pos in tick_positions]
    for axis in axes_flat[-2:]:
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, rotation=45, ha="right")
    fig.suptitle("HARP core monthly series for manual review")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_metric_coverage_heatmap(core_table: list[dict[str, Any]], output_path: Path) -> str | None:
    if not core_table:
        return None
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None
    months = [str(row.get("month") or "") for row in core_table]
    matrix = np.asarray(
        [
            [0.0 if row.get(metric_name, "") in ("", None) else 1.0 for row in core_table]
            for metric_name in CORE_MONTHLY_METRICS
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
    ax.set_yticks(list(range(len(CORE_MONTHLY_METRICS))))
    ax.set_yticklabels(list(CORE_MONTHLY_METRICS))
    tick_step = max(1, len(months) // 10)
    tick_positions = list(range(0, len(months), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([months[pos] for pos in tick_positions], rotation=45, ha="right")
    ax.set_title("HARP core-metric month coverage")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_monthly_density(density_rows: list[dict[str, Any]], output_path: Path) -> str | None:
    if not density_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    months = [str(row.get("month") or "") for row in density_rows]
    values = [int(row.get("row_count") or 0) for row in density_rows]
    x = list(range(len(months)))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(x, values, linewidth=1.8)
    ax.set_title("HARP monthly row density used for review")
    ax.set_ylabel("row count")
    ax.grid(alpha=0.25)
    tick_step = max(1, len(x) // 10)
    tick_positions = list(range(0, len(x), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([months[pos] for pos in tick_positions], rotation=45, ha="right")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _review_markdown(payload: dict[str, Any]) -> str:
    audit = dict(payload.get("quality_audit_summary") or {})
    lines = [
        "# HARP Human Review Package",
        "",
        f"- Generated at: `{payload.get('generated_at')}`",
        f"- Review run: `{payload.get('run_id')}`",
        f"- Archive run: `{payload.get('archive_run_id')}`",
        f"- Pure HARP metric rows: `{payload.get('historical_metric_row_count')}`",
        f"- Observed program panel rows: `{payload.get('observed_program_panel_row_count')}`",
        f"- Diagnosis-flow points: `{payload.get('diagnosis_flow_point_count')}`",
        f"- Annual HARP panel rows: `{payload.get('historical_harp_panel_row_count')}`",
        "",
        "## How To Review",
        "",
        "1. Open `analysis/harp_monthly_core_table.csv` and scan the key monthly heads directly.",
        "2. Open `analysis/harp_annual_cascade_table.csv` and verify the annual December cascade values look plausible.",
        "3. Use the plots to look for impossible jumps, long gaps, or source handoffs that look suspicious.",
        "4. If something looks wrong, open `analysis/harp_source_transitions.csv` and `analysis/harp_source_adjudication_table.csv` to inspect where the series changed source or where alternative rows were adjudicated.",
        "5. Open `analysis/harp_top20_suspicious_months.csv` for the fastest manual triage pass.",
        "",
        "## Manual Sanity Questions",
        "",
        "- Does `alive_on_art` stay below or near `diagnosed_plhiv` when both are present?",
        "- Does `virally_suppressed` stay below `tested_for_viral_load` and `alive_on_art`?",
        "- Do diagnosis flow and deaths behave like plausible monthly counts, not annual values accidentally snapped into a month?",
        "- Do any sharp jumps align with a source change rather than a real program change?",
        "",
        "## Machine Audit Snapshot",
        "",
        f"- Overall extraction audit passed: `{payload.get('quality_audit_overall_passed')}`",
        f"- HARP quality passed: `{audit.get('passed')}`",
        f"- Panel vs official seed discrepancies: `{len(audit.get('panel_vs_seed_discrepancies') or [])}`",
        f"- Program-point vs official seed discrepancies: `{len(audit.get('program_vs_seed_discrepancies') or [])}`",
        f"- Time/year mismatches: `{len(audit.get('panel_time_year_mismatches') or [])}`",
        f"- Invariant violations: `{len(audit.get('panel_invariant_violations') or [])}`",
        f"- Conflicting alternative source values: `{len(audit.get('source_conflicts_against_official_slide') or [])}`",
        "",
        "## Most Useful Files",
        "",
        "- `analysis/harp_monthly_core_table.csv`",
        "- `analysis/harp_annual_cascade_table.csv`",
        "- `analysis/harp_metric_inventory.csv`",
        "- `analysis/harp_source_inventory.csv`",
        "- `analysis/harp_source_transitions.csv`",
        "- `analysis/harp_top20_suspicious_months.csv`",
        "- `analysis/harp_suspicious_events.csv`",
        "- `analysis/harp_core_series.png`",
        "- `analysis/harp_core_coverage_heatmap.png`",
        "- `analysis/harp_monthly_density.png`",
        "- `analysis/harp_quality_audit.md`",
        "- `analysis/harp_source_adjudication_table.csv`",
        "",
    ]
    return "\n".join(lines)


def _harp_quality_markdown(payload: dict[str, Any]) -> str:
    harp = dict(payload.get("harp_quality") or {})
    lines = [
        "# HARP Quality Audit",
        "",
        f"Run directory: `{payload.get('run_dir')}`",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        f"- Passed: `{harp.get('passed')}`",
        f"- Panel vs official seed discrepancies: `{len(harp.get('panel_vs_seed_discrepancies') or [])}`",
        f"- Program-point vs official seed discrepancies: `{len(harp.get('program_vs_seed_discrepancies') or [])}`",
        f"- Time/year mismatches in selected panel: `{len(harp.get('panel_time_year_mismatches') or [])}`",
        f"- Invariant violations: `{len(harp.get('panel_invariant_violations') or [])}`",
        f"- Conflicting alternative source values against official slide: `{len(harp.get('source_conflicts_against_official_slide') or [])}`",
        f"- Adjudicated alternative rows: `{len(harp.get('source_adjudication_rows') or [])}`",
        "",
    ]
    return "\n".join(lines)


def run_tr_v3_harp_review_batch(
    *,
    run_id: str,
    archive_run_id: str = DEFAULT_ARCHIVE_RUN_ID,
    plugin_id: str = "hiv",
) -> dict[str, Any]:
    archive_run_dir = ROOT_DIR / "artifacts" / "runs" / str(archive_run_id)
    if not archive_run_dir.exists():
        raise FileNotFoundError(f"archive run does not exist: {archive_run_dir}")
    archive_dir = archive_run_dir / "harp_archive"
    if not archive_dir.exists():
        raise FileNotFoundError(f"archive dir does not exist: {archive_dir}")

    target_run_dir = ROOT_DIR / "artifacts" / "runs" / str(run_id)
    analysis_dir = ensure_dir(target_run_dir / "analysis")

    observed_program_panel = dict(read_json(archive_dir / "observed_program_panel.json", default={}))
    observed_program_rows = list(observed_program_panel.get("rows") or [])
    diagnosis_flow_payload = dict(read_json(archive_dir / "diagnosis_flow_points.json", default={}))
    diagnosis_flow_points = list(diagnosis_flow_payload.get("points") or [])
    program_points_payload = dict(read_json(archive_dir / "harp_program_points.json", default={}))
    program_points = list(program_points_payload.get("points") or [])
    historical_panel_payload = dict(read_json(archive_dir / "historical_harp_panel.json", default={}))
    historical_panel_rows = list(historical_panel_payload.get("rows") or [])
    metric_rows = list(read_json(archive_dir / "historical_metric_rows_harp_only.json", default=[]))
    manifest_payload = read_json(archive_dir / "archive_source_manifest.json", default={})
    if isinstance(manifest_payload, dict):
        manifest_rows = list(manifest_payload.get("source_manifest_rows") or manifest_payload.get("sources") or [])
    else:
        manifest_rows = list(manifest_payload or [])

    metric_inventory = _metric_inventory_rows(metric_rows)
    source_inventory = _source_inventory_rows(metric_rows, manifest_rows)
    monthly_core_table = _monthly_core_table(observed_program_rows, diagnosis_flow_points)
    annual_cascade_table = _annual_cascade_table(historical_panel_rows, program_points)
    source_transitions = _source_transition_rows(monthly_core_table)
    monthly_density = _monthly_density_rows(observed_program_rows, diagnosis_flow_points)
    suspicious_events = _suspicious_event_rows(monthly_core_table)
    top_suspicious_months = _top_suspicious_month_rows(suspicious_events, limit=20)

    write_json(analysis_dir / "harp_metric_inventory.json", {"rows": metric_inventory})
    _write_csv(analysis_dir / "harp_metric_inventory.csv", metric_inventory)
    write_json(analysis_dir / "harp_source_inventory.json", {"rows": source_inventory})
    _write_csv(analysis_dir / "harp_source_inventory.csv", source_inventory)
    write_json(analysis_dir / "harp_monthly_core_table.json", {"rows": monthly_core_table})
    _write_csv(analysis_dir / "harp_monthly_core_table.csv", monthly_core_table)
    write_json(analysis_dir / "harp_annual_cascade_table.json", {"rows": annual_cascade_table})
    _write_csv(analysis_dir / "harp_annual_cascade_table.csv", annual_cascade_table)
    write_json(analysis_dir / "harp_source_transitions.json", {"rows": source_transitions})
    _write_csv(analysis_dir / "harp_source_transitions.csv", source_transitions)
    write_json(analysis_dir / "harp_monthly_density.json", {"rows": monthly_density})
    _write_csv(analysis_dir / "harp_monthly_density.csv", monthly_density)
    write_json(analysis_dir / "harp_suspicious_events.json", {"rows": suspicious_events})
    _write_csv(analysis_dir / "harp_suspicious_events.csv", suspicious_events)
    write_json(analysis_dir / "harp_top20_suspicious_months.json", {"rows": top_suspicious_months})
    _write_csv(analysis_dir / "harp_top20_suspicious_months.csv", top_suspicious_months)

    core_plot_path = _plot_core_series(monthly_core_table, analysis_dir / "harp_core_series.png")
    coverage_plot_path = _plot_metric_coverage_heatmap(monthly_core_table, analysis_dir / "harp_core_coverage_heatmap.png")
    density_plot_path = _plot_monthly_density(monthly_density, analysis_dir / "harp_monthly_density.png")

    quality_harp = _harp_quality_audit(run_dir=archive_run_dir)
    quality_audit = {
        "generated_at": utc_now_iso(),
        "run_dir": str(archive_run_dir),
        "plugin_id": plugin_id,
        "harp_quality": quality_harp,
        "overall_passed": bool(quality_harp.get("passed")),
    }
    adjudication_paths = _write_harp_source_adjudication_table(
        analysis_dir=analysis_dir,
        rows=list(quality_harp.get("source_adjudication_rows") or []),
    )
    quality_harp["source_adjudication_table_paths"] = adjudication_paths
    write_json(analysis_dir / "harp_quality_audit.json", quality_audit)
    (analysis_dir / "harp_quality_audit.md").write_text(_harp_quality_markdown(quality_audit), encoding="utf-8")

    payload = {
        "generated_at": utc_now_iso(),
        "run_id": str(run_id),
        "archive_run_id": str(archive_run_id),
        "plugin_id": plugin_id,
        "historical_metric_row_count": int(len(metric_rows)),
        "observed_program_panel_row_count": int(len(observed_program_rows)),
        "diagnosis_flow_point_count": int(len(diagnosis_flow_points)),
        "historical_harp_panel_row_count": int(len(historical_panel_rows)),
        "metric_inventory_count": int(len(metric_inventory)),
        "source_inventory_count": int(len(source_inventory)),
        "source_transition_count": int(len(source_transitions)),
        "suspicious_event_count": int(len(suspicious_events)),
        "quality_audit_overall_passed": bool(quality_audit.get("overall_passed")),
        "quality_audit_summary": quality_harp,
        "artifacts": {
            "metric_inventory_csv": str(analysis_dir / "harp_metric_inventory.csv"),
            "source_inventory_csv": str(analysis_dir / "harp_source_inventory.csv"),
            "monthly_core_table_csv": str(analysis_dir / "harp_monthly_core_table.csv"),
            "annual_cascade_table_csv": str(analysis_dir / "harp_annual_cascade_table.csv"),
            "source_transitions_csv": str(analysis_dir / "harp_source_transitions.csv"),
            "top_suspicious_months_csv": str(analysis_dir / "harp_top20_suspicious_months.csv"),
            "suspicious_events_csv": str(analysis_dir / "harp_suspicious_events.csv"),
            "monthly_density_csv": str(analysis_dir / "harp_monthly_density.csv"),
            "core_plot": core_plot_path or "",
            "coverage_plot": coverage_plot_path or "",
            "density_plot": density_plot_path or "",
            "quality_audit_md": str(analysis_dir / "harp_quality_audit.md"),
            "quality_audit_json": str(analysis_dir / "harp_quality_audit.json"),
            "source_adjudication_csv": str(analysis_dir / "harp_source_adjudication_table.csv"),
        },
    }
    write_json(analysis_dir / "tr_v3_harp_review_batch_report.json", payload)
    (analysis_dir / "tr_v3_harp_review_batch_report.md").write_text(_review_markdown(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a human-review package for a HARP archive run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=DEFAULT_ARCHIVE_RUN_ID)
    parser.add_argument("--plugin", default="hiv")
    args = parser.parse_args()
    run_tr_v3_harp_review_batch(
        run_id=args.run_id,
        archive_run_id=args.archive_run_id,
        plugin_id=args.plugin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
