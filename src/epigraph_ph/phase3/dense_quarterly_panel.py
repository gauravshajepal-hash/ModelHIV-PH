from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from epigraph_ph.phase3.bridge_quarterly_panel import build_bridge_quarterly_panel_rows
from epigraph_ph.phase3.tr_v3_05_autoresearch import build_annual_anchor_rows, quarter_sort_key, repo_root
from epigraph_ph.runtime import ensure_dir, utc_now_iso, write_json

PRIMARY_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "new_diagnosed_cases_period",
)

MISSING_DATA_LADDER: tuple[str, ...] = (
    "exact_observed",
    "bridge_observed",
    "rule_based_extrapolated",
    "latent_imputed",
    "rejected_or_quarantined",
)


def _quarter_index(quarter: str) -> int:
    year_text, quarter_text = str(quarter).split("-Q", 1)
    return (int(year_text) * 4) + int(quarter_text) - 1


def _quarter_from_index(index: int) -> str:
    year, quarter_idx = divmod(int(index), 4)
    return f"{year:04d}-Q{quarter_idx + 1}"


def _all_quarters(start_quarter: str, end_quarter: str) -> list[str]:
    start_idx = _quarter_index(start_quarter)
    end_idx = _quarter_index(end_quarter)
    return [_quarter_from_index(idx) for idx in range(start_idx, end_idx + 1)]


def _tier_from_bridge_source(source_kind: str) -> str:
    kind = str(source_kind or "").lower()
    if kind in {"exact_snapshot", "quarterly_exact"}:
        return "exact_observed"
    if kind in {"bridge_cumulative_minus_deaths", "monthly_bridge", "monthly_bridge_latest_in_quarter", "monthly_aggregate"}:
        return "bridge_observed"
    return "rejected_or_quarantined"


def _annual_tier(row: dict[str, Any]) -> str:
    quality = str(row.get("source_quality_tier") or row.get("measurement_class") or "other").lower()
    if quality.startswith("official"):
        return "exact_observed"
    if quality == "derived":
        return "bridge_observed"
    if quality == "model_estimate":
        return "rule_based_extrapolated"
    return "rejected_or_quarantined"


def _linear_interpolate(
    quarters: list[str],
    anchors: dict[str, float],
    *,
    floor: float = 0.0,
    cap: float | None = None,
) -> dict[str, float]:
    if not anchors:
        return {quarter: floor for quarter in quarters}
    anchor_points = sorted((_quarter_index(quarter), float(value)) for quarter, value in anchors.items())
    out: dict[str, float] = {}
    for quarter in quarters:
        idx = _quarter_index(quarter)
        if quarter in anchors:
            value = float(anchors[quarter])
        else:
            left = [point for point in anchor_points if point[0] <= idx]
            right = [point for point in anchor_points if point[0] >= idx]
            if left and right:
                left_idx, left_value = left[-1]
                right_idx, right_value = right[0]
                if right_idx == left_idx:
                    value = float(left_value)
                else:
                    weight = float(idx - left_idx) / float(right_idx - left_idx)
                    value = float(left_value + weight * (right_value - left_value))
            elif left:
                value = float(left[-1][1])
            else:
                value = float(right[0][1])
        value = max(float(value), floor)
        if cap is not None:
            value = min(value, cap)
        out[quarter] = float(value)
    return out


def _reconciled_stock_path(
    quarters: list[str],
    *,
    anchors: dict[str, float],
    inflows: dict[str, float],
    outflows: dict[str, float],
    floor_map: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    if not quarters:
        return {}, {}
    floor_map = dict(floor_map or {})
    if not anchors:
        empty = {quarter: float(floor_map.get(quarter, 0.0)) for quarter in quarters}
        methods = {quarter: "no_anchor_floor" for quarter in quarters}
        return empty, methods
    quarter_index = {quarter: idx for idx, quarter in enumerate(quarters)}
    anchor_quarters = sorted([quarter for quarter in anchors if quarter in quarter_index], key=quarter_sort_key)
    if not anchor_quarters:
        empty = {quarter: float(floor_map.get(quarter, 0.0)) for quarter in quarters}
        methods = {quarter: "no_anchor_floor" for quarter in quarters}
        return empty, methods
    values: dict[str, float] = {}
    methods: dict[str, str] = {}
    first_anchor = anchor_quarters[0]
    first_idx = quarter_index[first_anchor]
    values[first_anchor] = float(max(anchors[first_anchor], floor_map.get(first_anchor, 0.0)))
    methods[first_anchor] = "observed_anchor"
    current = float(values[first_anchor])
    for idx in range(first_idx - 1, -1, -1):
        next_quarter = quarters[idx + 1]
        quarter = quarters[idx]
        current = max(current - float(inflows.get(next_quarter, 0.0)) + float(outflows.get(next_quarter, 0.0)), float(floor_map.get(quarter, 0.0)))
        values[quarter] = float(current)
        methods[quarter] = "reverse_flow_death"
    for left_anchor, right_anchor in zip(anchor_quarters[:-1], anchor_quarters[1:]):
        left_idx = quarter_index[left_anchor]
        right_idx = quarter_index[right_anchor]
        current = float(max(anchors[left_anchor], floor_map.get(left_anchor, 0.0)))
        provisional: dict[str, float] = {left_anchor: current}
        for idx in range(left_idx + 1, right_idx + 1):
            quarter = quarters[idx]
            current = max(current + float(inflows.get(quarter, 0.0)) - float(outflows.get(quarter, 0.0)), float(floor_map.get(quarter, 0.0)))
            provisional[quarter] = float(current)
        discrepancy = float(max(anchors[right_anchor], floor_map.get(right_anchor, 0.0)) - provisional[right_anchor])
        span = max(right_idx - left_idx, 1)
        values[left_anchor] = float(max(anchors[left_anchor], floor_map.get(left_anchor, 0.0)))
        methods[left_anchor] = "observed_anchor"
        for step, idx in enumerate(range(left_idx + 1, right_idx + 1), start=1):
            quarter = quarters[idx]
            adjusted = max(float(provisional[quarter]) + discrepancy * (float(step) / float(span)), float(floor_map.get(quarter, 0.0)))
            values[quarter] = float(adjusted)
            methods[quarter] = "flow_death_segment_adjusted" if quarter != right_anchor else "observed_anchor"
    last_anchor = anchor_quarters[-1]
    last_idx = quarter_index[last_anchor]
    current = float(max(anchors[last_anchor], floor_map.get(last_anchor, 0.0)))
    values[last_anchor] = current
    methods[last_anchor] = "observed_anchor"
    for idx in range(last_idx + 1, len(quarters)):
        quarter = quarters[idx]
        current = max(current + float(inflows.get(quarter, 0.0)) - float(outflows.get(quarter, 0.0)), float(floor_map.get(quarter, 0.0)))
        values[quarter] = float(current)
        methods[quarter] = "forward_flow_death"
    return values, methods


def _year_q4_anchors(annual_rows: list[dict[str, Any]], metric_name: str) -> dict[str, float]:
    return {
        f"{int(row['year']):04d}-Q4": float(row["value"])
        for row in annual_rows
        if str(row.get("metric_name") or "") == metric_name
    }


def _share_anchors(bridge_rows: list[dict[str, Any]], numerator: str, denominator: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in bridge_rows:
        numer = row.get(numerator)
        denom = row.get(denominator)
        if numer is None or denom is None:
            continue
        denom_value = float(denom)
        if denom_value <= 1e-9:
            continue
        out[str(row["quarter"])] = max(min(float(numer) / denom_value, 1.0), 0.0)
    return out


def _row_tier(row: dict[str, Any]) -> str:
    tiers = [str(row.get(f"{metric}_tier") or "rejected_or_quarantined") for metric in PRIMARY_METRICS]
    if any(tier == "rule_based_extrapolated" for tier in tiers):
        return "rule_based_extrapolated"
    if any(tier == "bridge_observed" for tier in tiers):
        return "bridge_observed"
    if any(tier == "exact_observed" for tier in tiers):
        return "exact_observed"
    return "rejected_or_quarantined"


def _filter_bridge_rows_through_quarter(
    bridge_rows: list[dict[str, Any]],
    *,
    max_quarter: str | None,
) -> list[dict[str, Any]]:
    if not max_quarter:
        return [dict(row) for row in bridge_rows]
    return [dict(row) for row in bridge_rows if quarter_sort_key(str(row["quarter"])) <= quarter_sort_key(str(max_quarter))]


def _filter_annual_rows_through_quarter(
    annual_rows: list[dict[str, Any]],
    *,
    max_quarter: str | None,
) -> list[dict[str, Any]]:
    if not max_quarter:
        return [dict(row) for row in annual_rows]
    max_year = int(str(max_quarter)[:4])
    return [dict(row) for row in annual_rows if int(row["year"]) <= max_year]


def build_dense_quarterly_panel_rows(archive_run_id: str, *, max_quarter: str | None = None) -> dict[str, Any]:
    bridge_payload = build_bridge_quarterly_panel_rows(archive_run_id)
    bridge_rows = _filter_bridge_rows_through_quarter(list(bridge_payload["rows"]), max_quarter=max_quarter)
    annual_rows = _filter_annual_rows_through_quarter(build_annual_anchor_rows(archive_run_id), max_quarter=max_quarter)
    bridge_map = {str(row["quarter"]): dict(row) for row in bridge_rows}
    annual_estimated = _year_q4_anchors(annual_rows, "estimated_plhiv")
    annual_deaths = _year_q4_anchors(annual_rows, "annual_aids_deaths")
    if not bridge_rows:
        return {"rows": [], "summary": {"archive_run_id": archive_run_id, "available_through_quarter": max_quarter}}
    start_quarter = min(str(row["quarter"]) for row in bridge_rows)
    end_quarter = max(str(row["quarter"]) for row in bridge_rows)
    quarters = _all_quarters(start_quarter, end_quarter)

    diagnosed_observed = {
        str(row["quarter"]): float(row["diagnosed_plhiv"])
        for row in bridge_rows
        if row.get("diagnosed_plhiv") is not None
    }
    alive_observed = {
        str(row["quarter"]): float(row["alive_on_art"])
        for row in bridge_rows
        if row.get("alive_on_art") is not None
    }
    flow_observed = {
        str(row["quarter"]): float(row["new_diagnosed_cases_period"])
        for row in bridge_rows
        if row.get("new_diagnosed_cases_period") is not None
    }
    deaths_flow_observed = {
        str(row["quarter"]): float(row["deaths_reported_period"])
        for row in bridge_rows
        if row.get("deaths_reported_period") is not None
    }
    tested_share_observed = _share_anchors(bridge_rows, "tested_for_viral_load", "alive_on_art")
    suppressed_share_observed = _share_anchors(bridge_rows, "virally_suppressed", "alive_on_art")
    alive_interp = _linear_interpolate(quarters, alive_observed, floor=0.0)
    estimated_interp = _linear_interpolate(quarters, annual_estimated, floor=0.0)
    tested_share_interp = _linear_interpolate(quarters, tested_share_observed, floor=0.0, cap=1.0)
    suppressed_share_interp = _linear_interpolate(quarters, suppressed_share_observed, floor=0.0, cap=1.0)
    death_quarter_estimate = {
        quarter: float(deaths_flow_observed.get(quarter, annual_deaths.get(f"{int(quarter[:4]):04d}-Q4", 0.0) / 4.0))
        for quarter in quarters
    }
    diagnosed_floor_map = {quarter: float(alive_observed.get(quarter, 0.0)) for quarter in quarters}
    diagnosed_stock_estimate, diagnosed_methods = _reconciled_stock_path(
        quarters,
        anchors=diagnosed_observed,
        inflows=flow_observed,
        outflows=death_quarter_estimate,
        floor_map=diagnosed_floor_map,
    )

    rows: list[dict[str, Any]] = []
    for quarter in quarters:
        bridge_row = bridge_map.get(quarter, {})
        diagnosed_value = bridge_row.get("diagnosed_plhiv")
        if diagnosed_value is not None:
            diagnosed_tier = _tier_from_bridge_source(str(bridge_row.get("diagnosed_plhiv_source_kind") or "missing"))
            diagnosed = float(diagnosed_value)
            diagnosed_method = "observed_anchor"
        else:
            diagnosed_tier = "rule_based_extrapolated"
            diagnosed = float(diagnosed_stock_estimate[quarter])
            diagnosed_method = str(diagnosed_methods.get(quarter) or "rule_based_extrapolated")

        alive_value = bridge_row.get("alive_on_art")
        if alive_value is not None:
            alive_tier = _tier_from_bridge_source(str(bridge_row.get("alive_on_art_source_kind") or "missing"))
            alive = float(alive_value)
        else:
            alive_tier = "rule_based_extrapolated"
            alive = float(min(alive_interp[quarter], diagnosed))

        estimated_value = bridge_row.get("estimated_plhiv")
        if estimated_value is not None:
            estimated_tier = _tier_from_bridge_source(str(bridge_row.get("estimated_plhiv_source_kind") or "missing"))
            estimated = float(estimated_value)
        else:
            estimated_tier = "rule_based_extrapolated"
            estimated = float(max(estimated_interp[quarter], diagnosed))

        tested_value = bridge_row.get("tested_for_viral_load")
        if tested_value is not None:
            tested_tier = _tier_from_bridge_source(str(bridge_row.get("tested_for_viral_load_source_kind") or "missing"))
            tested = float(tested_value)
        else:
            tested_tier = "rule_based_extrapolated"
            tested = float(max(min(tested_share_interp[quarter] * alive, alive), 0.0))

        suppressed_value = bridge_row.get("virally_suppressed")
        if suppressed_value is not None:
            suppressed_tier = _tier_from_bridge_source(str(bridge_row.get("virally_suppressed_source_kind") or "missing"))
            suppressed = float(suppressed_value)
        else:
            suppressed_tier = "rule_based_extrapolated"
            suppressed = float(max(min(suppressed_share_interp[quarter] * alive, alive), 0.0))

        if bridge_row.get("new_diagnosed_cases_period") is not None:
            flow_tier = _tier_from_bridge_source(str(bridge_row.get("new_diagnosed_cases_source_kind") or "missing"))
            new_diagnosed = float(bridge_row["new_diagnosed_cases_period"])
        else:
            flow_tier = "rule_based_extrapolated"
            new_diagnosed = float(max(flow_observed.get(quarter, 0.0), 0.0))

        row = {
            "quarter": quarter,
            "diagnosed_plhiv": float(max(diagnosed, alive)),
            "diagnosed_plhiv_tier": diagnosed_tier,
            "diagnosed_plhiv_imputation_method": diagnosed_method,
            "alive_on_art": float(min(max(alive, 0.0), max(diagnosed, alive))),
            "alive_on_art_tier": alive_tier,
            "new_diagnosed_cases_period": float(max(new_diagnosed, 0.0)),
            "new_diagnosed_cases_period_tier": flow_tier,
            "tested_for_viral_load": float(max(tested, 0.0)),
            "tested_for_viral_load_tier": tested_tier,
            "virally_suppressed": float(max(min(suppressed, alive), 0.0)),
            "virally_suppressed_tier": suppressed_tier,
            "estimated_plhiv": float(max(estimated, diagnosed)),
            "estimated_plhiv_tier": estimated_tier,
        }
        row["row_tier"] = _row_tier(row)
        row["score_eligible_metrics"] = [
            metric
            for metric in PRIMARY_METRICS
            if str(row.get(f"{metric}_tier") or "") in {"exact_observed", "bridge_observed"}
        ]
        row["score_eligible"] = bool(row["score_eligible_metrics"])
        rows.append(row)

    summary = {
        "archive_run_id": archive_run_id,
        "available_through_quarter": max_quarter,
        "quarter_count": len(rows),
        "quarter_range": [rows[0]["quarter"], rows[-1]["quarter"]],
        "score_eligible_quarters": sum(1 for row in rows if bool(row["score_eligible"])),
        "score_eligible_years": sorted({int(str(row["quarter"])[:4]) for row in rows if bool(row["score_eligible"])}),
        "row_tier_counts": {
            tier: sum(1 for row in rows if str(row.get("row_tier") or "") == tier)
            for tier in MISSING_DATA_LADDER
        },
        "metric_tier_counts": {
            metric: {
                tier: sum(1 for row in rows if str(row.get(f"{metric}_tier") or "") == tier)
                for tier in MISSING_DATA_LADDER
            }
            for metric in ("diagnosed_plhiv", "alive_on_art", "new_diagnosed_cases_period", "tested_for_viral_load", "virally_suppressed", "estimated_plhiv")
        },
        "diagnosed_imputation_method_counts": {
            method: sum(1 for row in rows if str(row.get("diagnosed_plhiv_imputation_method") or "") == method)
            for method in sorted({str(row.get("diagnosed_plhiv_imputation_method") or "") for row in rows})
        },
    }
    return {"rows": rows, "summary": summary}


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = dict(payload["summary"])
    lines = [
        "# Dense Quarterly Panel",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Quarter range: `{summary['quarter_range'][0]}` -> `{summary['quarter_range'][1]}`",
        f"- Quarter count: `{summary['quarter_count']}`",
        f"- Score-eligible quarters: `{summary['score_eligible_quarters']}`",
        f"- Score-eligible years: `{', '.join(str(year) for year in summary['score_eligible_years'])}`",
        "",
        "## Row Tier Counts",
        "",
        "| Tier | Count |",
        "|---|---:|",
    ]
    for tier, count in dict(summary["row_tier_counts"]).items():
        lines.append(f"| {tier} | {count} |")
    lines.extend(
        [
            "",
            "## Primary Metric Tier Counts",
            "",
            "| Metric | Exact | Bridge | Rule-based | Latent | Rejected |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric_name in PRIMARY_METRICS:
        counts = dict(summary["metric_tier_counts"][metric_name])
        lines.append(
            f"| {metric_name} | {counts.get('exact_observed', 0)} | {counts.get('bridge_observed', 0)} | "
            f"{counts.get('rule_based_extrapolated', 0)} | {counts.get('latent_imputed', 0)} | "
            f"{counts.get('rejected_or_quarantined', 0)} |"
        )
    return "\n".join(lines) + "\n"


def run_dense_quarterly_panel(*, run_id: str, archive_run_id: str) -> dict[str, Any]:
    panel = build_dense_quarterly_panel_rows(archive_run_id)
    payload = {
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "archive_run_id": archive_run_id,
        "rows": panel["rows"],
        "summary": panel["summary"],
    }
    analysis_dir = ensure_dir(repo_root() / "artifacts" / "runs" / run_id / "analysis")
    write_json(analysis_dir / "dense_quarterly_panel.json", payload)
    _write_rows_csv(analysis_dir / "dense_quarterly_panel.csv", payload["rows"])
    (analysis_dir / "dense_quarterly_panel.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dense-quarterly-panel")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dense_quarterly_panel(run_id=args.run_id, archive_run_id=args.archive_run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
