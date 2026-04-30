from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from epigraph_ph.harp_archive.doh_hiv_sti_archive import extract_art_summary_rows, extract_diagnosis_summary_rows
from epigraph_ph.runtime import ensure_dir, read_json, utc_now_iso, write_json

_PRIMARY_PRIORITY: dict[str, int] = {
    "official_user_provided_slide": 0,
    "official_mirror": 1,
    "official": 2,
    "official_local_corpus": 2,
    "official_doh_archive": 2,
    "official_unaids_country_data": 2,
    "official_wdi_unaids_hiv_series": 2,
    "model_estimate": 3,
    "derived": 4,
    "other": 5,
}

_EXACT_SNAPSHOT_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
    "estimated_plhiv",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def quarter_sort_key(value: str) -> tuple[int, int]:
    year_text, quarter_text = str(value).split("-Q", 1)
    return int(year_text), int(quarter_text)


def _quarter_from_month(value: str) -> str:
    year_text, month_text = str(value).split("-", 1)
    month = int(month_text[:2])
    return f"{int(year_text):04d}-Q{((month - 1) // 3) + 1}"


def _quarter_end_month_label(quarter: str) -> str:
    year_text, quarter_text = str(quarter).split("-Q", 1)
    month = {1: 3, 2: 6, 3: 9, 4: 12}[int(quarter_text)]
    return f"{int(year_text):04d}-{month:02d}"


def _month_ordinal(value: str) -> int | None:
    text = str(value or "")
    if len(text) < 7 or not text[:4].isdigit() or not text[5:7].isdigit():
        return None
    return (int(text[:4]) * 12) + int(text[5:7]) - 1


def _priority_key(row: dict[str, Any]) -> tuple[float, float, str]:
    quality = str(row.get("source_quality_tier") or row.get("measurement_class") or "other")
    quality_rank = _PRIMARY_PRIORITY.get(quality, _PRIMARY_PRIORITY["other"])
    confidence = -float(row.get("evidence_confidence") or 0.0)
    source_id = str(row.get("source_id") or "")
    return float(quality_rank), confidence, source_id


def _snapshot_selection_key(quarter: str, row: dict[str, Any]) -> tuple[int, int, float, float, str]:
    time_label = str(row.get("time") or row.get("period_end") or "")
    month_ordinal = _month_ordinal(time_label) or -1
    quarter_end_month = _quarter_end_month_label(quarter)
    quarter_end_rank = 0 if time_label[:7] == quarter_end_month else 1
    quality_rank, confidence, source_id = _priority_key(row)
    return quarter_end_rank, -month_ordinal, quality_rank, confidence, source_id


def _load_harp_only_rows(archive_run_id: str, *, repo: Path | None = None) -> list[dict[str, Any]]:
    root = repo or repo_root()
    path = root / "artifacts" / "runs" / archive_run_id / "harp_archive" / "historical_metric_rows_harp_only.json"
    return list(read_json(path, default=[]) or [])


def _load_archive_source_rows(archive_run_id: str, *, repo: Path | None = None) -> list[dict[str, Any]]:
    root = repo or repo_root()
    path = root / "artifacts" / "runs" / archive_run_id / "harp_archive" / "archive_source_manifest.json"
    payload = dict(read_json(path, default={}) or {})
    return [dict(row) for row in list(payload.get("sources") or [])]


def _national_rows(rows: list[dict[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("region") or "").lower() == "national" and str(row.get("metric_name") or "") == metric_name
    ]


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("metric_name") or ""),
            str(row.get("time") or row.get("period_end") or ""),
            str(row.get("source_id") or ""),
            str(row.get("period_start") or ""),
            str(row.get("period_end") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
    return deduped


def _load_ocr_pages(source_row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("ocr_artifact_path", "shared_artifact_path", "run_artifact_path"):
        path_text = str(source_row.get(key) or "")
        if not path_text:
            continue
        payload = dict(read_json(Path(path_text), default={}) or {})
        pages = list(payload.get("pages") or [])
        if pages:
            return [dict(page) for page in pages]
    return []


def _supplement_rows_from_monthly_corpus(archive_run_id: str, *, repo: Path | None = None) -> list[dict[str, Any]]:
    source_rows = _load_archive_source_rows(archive_run_id, repo=repo)
    supplement: list[dict[str, Any]] = []
    for source in source_rows:
        if str(source.get("source_kind") or "") != "doh_hiv_sti_archive_pdf":
            continue
        if str(source.get("temporal_precision") or "").lower() != "monthly_snapshot":
            continue
        pages = _load_ocr_pages(source)
        if not pages:
            continue
        supplement.extend(extract_diagnosis_summary_rows(source, pages))
        supplement.extend(extract_art_summary_rows(source, pages))
    return _deduplicate_rows(supplement)


def _choose_month_rows(rows: list[dict[str, Any]], metric_name: str) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for row in _national_rows(rows, metric_name):
        time_label = str(row.get("time") or row.get("period_end") or "")
        if len(time_label) < 7:
            continue
        month = time_label[:7]
        previous = chosen.get(month)
        if previous is None or _priority_key(row) < _priority_key(previous):
            chosen[month] = row
    return chosen


def _choose_month_bridge_rows_by_quarter(rows: list[dict[str, Any]], metric_name: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _national_rows(rows, metric_name):
        series_kind = str(row.get("series_kind") or row.get("temporal_precision") or "").lower()
        if "monthly" not in series_kind:
            continue
        time_label = str(row.get("time") or row.get("period_end") or "")
        if len(time_label) < 7:
            continue
        quarter = _quarter_from_month(time_label[:7])
        grouped.setdefault(quarter, []).append(row)
    chosen: dict[str, dict[str, Any]] = {}
    for quarter, qrows in grouped.items():
        selected = min(qrows, key=lambda row: _snapshot_selection_key(quarter, row))
        time_label = str(selected.get("time") or selected.get("period_end") or "")[:7]
        source_kind = "monthly_bridge" if time_label == _quarter_end_month_label(quarter) else "monthly_bridge_latest_in_quarter"
        chosen[quarter] = {
            **dict(selected),
            "source_kind": source_kind,
            "source_ids": [str(selected.get("source_id") or "")] if str(selected.get("source_id") or "") else [],
        }
    return chosen


def _choose_snapshot_rows(rows: list[dict[str, Any]], metric_name: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _national_rows(rows, metric_name):
        series_kind = str(row.get("series_kind") or row.get("temporal_precision") or "").lower()
        if "monthly" in series_kind:
            continue
        time_label = str(row.get("time") or row.get("period_end") or "")
        if len(time_label) < 7:
            continue
        quarter = _quarter_from_month(time_label[:7])
        grouped.setdefault(quarter, []).append(row)
    chosen: dict[str, dict[str, Any]] = {}
    for quarter, qrows in grouped.items():
        chosen[quarter] = min(qrows, key=lambda row: _snapshot_selection_key(quarter, row))
    return chosen


def _choose_flow_rows(rows: list[dict[str, Any]], metric_name: str) -> dict[str, dict[str, Any]]:
    quarter_rows: dict[str, list[dict[str, Any]]] = {}
    monthly_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _national_rows(rows, metric_name):
        period_end = str(row.get("period_end") or row.get("time") or "")
        period_start = str(row.get("period_start") or period_end)
        if len(period_end) < 7:
            continue
        series_kind = str(row.get("series_kind") or row.get("temporal_precision") or "").lower()
        quarter = _quarter_from_month(period_end[:7])
        if "quarterly" in series_kind:
            quarter_rows.setdefault(quarter, []).append(row)
            continue
        start_ord = _month_ordinal(period_start[:7])
        end_ord = _month_ordinal(period_end[:7])
        if start_ord is None or end_ord is None or start_ord != end_ord:
            continue
        month = period_end[:7]
        monthly_rows.setdefault(quarter, {})
        previous = monthly_rows[quarter].get(month)
        if previous is None or _priority_key(row) < _priority_key(previous):
            monthly_rows[quarter][month] = row
    chosen: dict[str, dict[str, Any]] = {}
    for quarter, qrows in quarter_rows.items():
        selected = min(qrows, key=_priority_key)
        chosen[quarter] = {
            "value": float(selected.get("value") or 0.0),
            "source_kind": "quarterly_exact",
            "source_ids": [str(selected.get("source_id") or "")],
        }
    for quarter, month_map in monthly_rows.items():
        if quarter in chosen:
            continue
        if len(month_map) < 3:
            continue
        selected_rows = [month_map[month] for month in sorted(month_map.keys())]
        chosen[quarter] = {
            "value": float(sum(float(row.get("value") or 0.0) for row in selected_rows)),
            "source_kind": "monthly_aggregate",
            "source_ids": sorted({str(row.get("source_id") or "") for row in selected_rows if str(row.get("source_id") or "")}),
            "month_count": len(selected_rows),
        }
    return chosen


def _source_ids(*rows: dict[str, Any] | None, extra_ids: list[str] | None = None) -> list[str]:
    values = {str(row.get("source_id") or "") for row in rows if row is not None and str(row.get("source_id") or "")}
    if extra_ids:
        values.update(str(value or "") for value in extra_ids if str(value or ""))
    return sorted(value for value in values if value)


def _quarter_row(
    quarter: str,
    exact_snapshots: dict[str, dict[str, dict[str, Any]]],
    month_rows: dict[str, dict[str, dict[str, Any]]],
    bridge_stock_rows: dict[str, dict[str, dict[str, Any]]],
    flow_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    quarter_end_month = _quarter_end_month_label(quarter)
    diagnosed_exact = exact_snapshots["diagnosed_plhiv"].get(quarter)
    alive_exact = exact_snapshots["alive_on_art"].get(quarter)
    tested_exact = exact_snapshots["tested_for_viral_load"].get(quarter)
    suppressed_exact = exact_snapshots["virally_suppressed"].get(quarter)
    estimated_exact = exact_snapshots["estimated_plhiv"].get(quarter)
    diagnosed_direct_bridge = bridge_stock_rows["diagnosed_plhiv"].get(quarter)
    alive_direct_bridge = bridge_stock_rows["alive_on_art"].get(quarter)
    diagnosed_cases_cumulative = month_rows["diagnosed_cases_cumulative"].get(quarter_end_month)
    deaths_reported_cumulative = month_rows["deaths_reported_cumulative"].get(quarter_end_month)
    diagnosed_bridge_value = None
    if diagnosed_cases_cumulative is not None and deaths_reported_cumulative is not None:
        diagnosed_bridge_value = float(diagnosed_cases_cumulative.get("value") or 0.0) - float(deaths_reported_cumulative.get("value") or 0.0)
    diagnosed_value = (
        float(diagnosed_exact.get("value") or 0.0)
        if diagnosed_exact is not None
        else float(diagnosed_direct_bridge.get("value") or 0.0)
        if diagnosed_direct_bridge is not None
        else diagnosed_bridge_value
    )
    diagnosed_source_kind = (
        "exact_snapshot"
        if diagnosed_exact is not None
        else str(diagnosed_direct_bridge.get("source_kind") or "monthly_bridge")
        if diagnosed_direct_bridge is not None
        else "bridge_cumulative_minus_deaths"
        if diagnosed_bridge_value is not None
        else "missing"
    )
    alive_value = (
        float(alive_exact.get("value") or 0.0)
        if alive_exact is not None
        else float(alive_direct_bridge.get("value") or 0.0)
        if alive_direct_bridge is not None
        else None
    )
    alive_source_kind = (
        "exact_snapshot"
        if alive_exact is not None
        else str(alive_direct_bridge.get("source_kind") or "monthly_bridge")
        if alive_direct_bridge is not None
        else "missing"
    )
    diagnosis_flow = flow_rows["new_diagnosed_cases_period"].get(quarter)
    deaths_flow = flow_rows["deaths_reported_period"].get(quarter)
    exact_used = any(row is not None for row in (diagnosed_exact, alive_exact, tested_exact, suppressed_exact, estimated_exact))
    bridge_used = diagnosed_source_kind != "exact_snapshot" and diagnosed_source_kind != "missing" or alive_source_kind != "exact_snapshot" and alive_source_kind != "missing"
    if exact_used and bridge_used:
        anchor_class = "mixed"
    elif bridge_used:
        anchor_class = "bridge"
    elif exact_used:
        anchor_class = "exact"
    else:
        anchor_class = "flow_only"
    return {
        "quarter": quarter,
        "anchor_month": quarter_end_month,
        "anchor_class": anchor_class,
        "diagnosed_plhiv": diagnosed_value,
        "diagnosed_plhiv_source_kind": diagnosed_source_kind,
        "alive_on_art": alive_value,
        "alive_on_art_source_kind": alive_source_kind,
        "tested_for_viral_load": float(tested_exact.get("value") or 0.0) if tested_exact is not None else None,
        "tested_for_viral_load_source_kind": "exact_snapshot" if tested_exact is not None else "missing",
        "virally_suppressed": float(suppressed_exact.get("value") or 0.0) if suppressed_exact is not None else None,
        "virally_suppressed_source_kind": "exact_snapshot" if suppressed_exact is not None else "missing",
        "estimated_plhiv": float(estimated_exact.get("value") or 0.0) if estimated_exact is not None else None,
        "estimated_plhiv_source_kind": "exact_snapshot" if estimated_exact is not None else "missing",
        "diagnosed_cases_cumulative": float(diagnosed_cases_cumulative.get("value") or 0.0) if diagnosed_cases_cumulative is not None else None,
        "deaths_reported_cumulative": float(deaths_reported_cumulative.get("value") or 0.0) if deaths_reported_cumulative is not None else None,
        "new_diagnosed_cases_period": float(diagnosis_flow.get("value") or 0.0) if diagnosis_flow is not None else None,
        "new_diagnosed_cases_source_kind": str(diagnosis_flow.get("source_kind") or "missing") if diagnosis_flow is not None else "missing",
        "deaths_reported_period": float(deaths_flow.get("value") or 0.0) if deaths_flow is not None else None,
        "deaths_reported_source_kind": str(deaths_flow.get("source_kind") or "missing") if deaths_flow is not None else "missing",
        "stock_anchor_complete": diagnosed_value is not None and alive_value is not None,
        "source_ids": _source_ids(
            diagnosed_exact,
            alive_exact,
            tested_exact,
            suppressed_exact,
            estimated_exact,
            diagnosed_direct_bridge,
            diagnosed_cases_cumulative,
            deaths_reported_cumulative,
            alive_direct_bridge,
            extra_ids=(diagnosis_flow or {}).get("source_ids"),
        ),
    }


def build_bridge_quarterly_panel_rows(archive_run_id: str, *, repo: Path | None = None) -> dict[str, Any]:
    rows = _load_harp_only_rows(archive_run_id, repo=repo)
    supplement_rows = _supplement_rows_from_monthly_corpus(archive_run_id, repo=repo)
    rows = _deduplicate_rows(list(rows) + list(supplement_rows))
    exact_snapshots = {metric_name: _choose_snapshot_rows(rows, metric_name) for metric_name in _EXACT_SNAPSHOT_METRICS}
    bridge_stock_rows = {
        "diagnosed_plhiv": _choose_month_bridge_rows_by_quarter(rows, "diagnosed_plhiv"),
        "alive_on_art": _choose_month_bridge_rows_by_quarter(rows, "alive_on_art"),
    }
    month_rows = {
        "diagnosed_cases_cumulative": _choose_month_rows(rows, "diagnosed_cases_cumulative"),
        "deaths_reported_cumulative": _choose_month_rows(rows, "deaths_reported_cumulative"),
    }
    flow_rows = {
        "new_diagnosed_cases_period": _choose_flow_rows(rows, "new_diagnosed_cases_period"),
        "deaths_reported_period": _choose_flow_rows(rows, "deaths_reported_period"),
    }
    quarter_set = set()
    for snapshot_map in exact_snapshots.values():
        quarter_set.update(snapshot_map.keys())
    for flow_map in flow_rows.values():
        quarter_set.update(flow_map.keys())
    for bridge_map in bridge_stock_rows.values():
        quarter_set.update(bridge_map.keys())
    for month_map in month_rows.values():
        quarter_set.update(_quarter_from_month(month) for month in month_map.keys())
    panel_rows = [
        _quarter_row(quarter, exact_snapshots, month_rows, bridge_stock_rows, flow_rows)
        for quarter in sorted(quarter_set, key=quarter_sort_key)
    ]
    exact_complete = {
        quarter
        for quarter in quarter_set
        if exact_snapshots["diagnosed_plhiv"].get(quarter) is not None and exact_snapshots["alive_on_art"].get(quarter) is not None
    }
    bridge_complete = {str(row["quarter"]) for row in panel_rows if bool(row["stock_anchor_complete"])}
    summary = {
        "archive_run_id": archive_run_id,
        "supplement_row_count": len(supplement_rows),
        "total_quarters": len(panel_rows),
        "exact_complete_stock_quarters": len(exact_complete),
        "bridge_complete_stock_quarters": len(bridge_complete),
        "bridge_added_complete_stock_quarters": sorted(bridge_complete - exact_complete, key=quarter_sort_key),
        "diagnosed_bridge_quarters": [str(row["quarter"]) for row in panel_rows if str(row["diagnosed_plhiv_source_kind"]) in {"bridge_cumulative_minus_deaths", "monthly_bridge", "monthly_bridge_latest_in_quarter"}],
        "diagnosed_direct_bridge_quarters": [str(row["quarter"]) for row in panel_rows if str(row["diagnosed_plhiv_source_kind"]) in {"monthly_bridge", "monthly_bridge_latest_in_quarter"}],
        "alive_bridge_quarters": [str(row["quarter"]) for row in panel_rows if str(row["alive_on_art_source_kind"]) in {"monthly_bridge", "monthly_bridge_latest_in_quarter"}],
        "quarterly_death_flow_quarters": [str(row["quarter"]) for row in panel_rows if row.get("deaths_reported_period") is not None],
        "quarterly_diagnosis_flow_quarters": [str(row["quarter"]) for row in panel_rows if row.get("new_diagnosed_cases_period") is not None],
    }
    return {"rows": panel_rows, "summary": summary}


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
    summary = dict(payload.get("summary") or {})
    lines = [
        "# Bridge Quarterly Panel",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Supplement rows re-parsed from local monthly corpus: `{summary.get('supplement_row_count', 0)}`",
        f"- Total quarters: `{summary.get('total_quarters', 0)}`",
        f"- Exact complete stock quarters: `{summary.get('exact_complete_stock_quarters', 0)}`",
        f"- Bridge complete stock quarters: `{summary.get('bridge_complete_stock_quarters', 0)}`",
        "",
        "## Coverage Summary",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Bridge-added complete stock quarters | {len(summary.get('bridge_added_complete_stock_quarters', []))} |",
        f"| Diagnosed bridge quarters | {len(summary.get('diagnosed_bridge_quarters', []))} |",
        f"| Diagnosed direct-stock bridge quarters | {len(summary.get('diagnosed_direct_bridge_quarters', []))} |",
        f"| Alive-on-ART bridge quarters | {len(summary.get('alive_bridge_quarters', []))} |",
        f"| Quarterly diagnosis-flow quarters | {len(summary.get('quarterly_diagnosis_flow_quarters', []))} |",
        f"| Quarterly deaths-flow quarters | {len(summary.get('quarterly_death_flow_quarters', []))} |",
        "",
        "## Bridge-Added Complete Stock Quarters",
        "",
        ", ".join(summary.get("bridge_added_complete_stock_quarters", [])) or "None",
        "",
        "## Quarter Rows",
        "",
        "| Quarter | Anchor class | Diagnosed | Diagnosed source | Alive on ART | Alive source | Diagnosis flow | Deaths flow |",
        "|---|---|---:|---|---:|---|---:|---:|",
    ]
    for row in payload["rows"]:
        diagnosed = "None" if row.get("diagnosed_plhiv") is None else f"{float(row['diagnosed_plhiv']):.0f}"
        alive = "None" if row.get("alive_on_art") is None else f"{float(row['alive_on_art']):.0f}"
        diag_flow = "None" if row.get("new_diagnosed_cases_period") is None else f"{float(row['new_diagnosed_cases_period']):.0f}"
        death_flow = "None" if row.get("deaths_reported_period") is None else f"{float(row['deaths_reported_period']):.0f}"
        lines.append(
            f"| {row['quarter']} | {row['anchor_class']} | {diagnosed} | {row['diagnosed_plhiv_source_kind']} | {alive} | {row['alive_on_art_source_kind']} | {diag_flow} | {death_flow} |"
        )
    return "\n".join(lines) + "\n"


def run_bridge_quarterly_panel(*, run_id: str, archive_run_id: str) -> dict[str, Any]:
    panel = build_bridge_quarterly_panel_rows(archive_run_id)
    payload = {
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "archive_run_id": archive_run_id,
        "rows": panel["rows"],
        "summary": panel["summary"],
    }
    analysis_dir = ensure_dir(repo_root() / "artifacts" / "runs" / run_id / "analysis")
    write_json(analysis_dir / "bridge_quarterly_panel.json", payload)
    _write_rows_csv(analysis_dir / "bridge_quarterly_panel.csv", payload["rows"])
    (analysis_dir / "bridge_quarterly_panel.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bridge-quarterly-panel")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_bridge_quarterly_panel(run_id=args.run_id, archive_run_id=args.archive_run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
