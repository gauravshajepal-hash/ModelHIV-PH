from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .metrics import quarter_sort_key
from .r11_sparse_state_space import _generated_at, _sha256
from .r68_bulk_external_source_ingest import R68_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R69_SCHEMA_VERSION = "phase3_dynamic.r69_bulk_signal_feature_compiler.v1"
R69_RUN_ID = "p3d-r69-bulk-signal-feature-compiler-20260506-s00"
R68_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R68_RUN_ID
    / "analysis"
    / "r68_bulk_external_source_ingest_report.json"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _quarter_from_date(date_text: str) -> str:
    year, month, *_ = date_text.split("-")
    quarter = ((int(month) - 1) // 3) + 1
    return f"{year}-Q{quarter}"


def _quarter_from_year(year_text: str) -> str:
    return f"{int(float(year_text))}-Q4"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _path_for_source(r68_report: dict[str, Any], source_id: str) -> Path | None:
    for row in r68_report.get("extracted_table_rows") or []:
        if row.get("source_id") == source_id and row.get("output_path"):
            return Path(str(row["output_path"]))
    return None


def _indicator_module(indicator: str) -> str:
    lower = indicator.lower()
    if "new hiv infection" in lower or "incidence" in lower:
        return "incidence_validation"
    if "aids-related death" in lower or "aids mortality" in lower:
        return "mortality_reporting"
    if "people living with hiv" in lower or "prevalence" in lower:
        return "plhiv_stock_validation"
    if "viral" in lower or "suppressed" in lower:
        return "vl_suppression_service"
    if "art" in lower or "treatment" in lower or "enrolled" in lower:
        return "art_retention"
    if "prep" in lower or "pre-exposure" in lower:
        return "prep_persistence"
    if "test" in lower or "diagnos" in lower:
        return "diagnosis_reporting"
    return "other"


def _compile_unaids_estimates(path: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path is None:
        return rows
    for row in _read_csv(path):
        indicator = str(row.get("Indicator") or "")
        module = _indicator_module(indicator)
        if module not in {"incidence_validation", "mortality_reporting", "plhiv_stock_validation"}:
            continue
        value = _float_or_none(row.get("Data value"))
        if value is None:
            continue
        rows.append(
            {
                "signal_id": f"unaids_estimates::{row.get('Indicator_GId')}::{row.get('Subgroup_Val_GId')}::{row.get('Time Period')}",
                "source_id": "unaids_estimates_2025",
                "module_target": module,
                "time_period": _quarter_from_year(str(row.get("Time Period") or "")),
                "time_granularity": "annual_as_q4_anchor",
                "indicator": indicator,
                "indicator_gid": row.get("Indicator_GId"),
                "subgroup": row.get("Subgroup"),
                "unit": row.get("Unit"),
                "value": value,
                "allowed_use": "validation_only_or_weak_measurement",
                "measurement_semantics": "modeled_estimate",
            }
        )
    rows.sort(key=lambda item: (str(item["module_target"]), quarter_sort_key(str(item["time_period"])), str(item["indicator_gid"])))
    return rows


def _compile_gam_program_rows(path: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path is None:
        return rows
    for row in _read_csv(path):
        indicator = str(row.get("Indicator") or "")
        module = _indicator_module(indicator)
        if module == "other":
            continue
        value = _float_or_none(row.get("Data value"))
        if value is None:
            continue
        rows.append(
            {
                "signal_id": f"unaids_gam::{row.get('Indicator_GId')}::{row.get('Subgroup_Val_GId')}::{row.get('Time Period')}",
                "source_id": "unaids_gam_2025",
                "module_target": module,
                "time_period": _quarter_from_year(str(row.get("Time Period") or "")),
                "time_granularity": "annual_as_q4_program_support",
                "indicator": indicator,
                "indicator_gid": row.get("Indicator_GId"),
                "subgroup": row.get("Subgroup"),
                "unit": row.get("Unit"),
                "value": value,
                "allowed_use": "auxiliary_program_support",
                "measurement_semantics": "country_reported_or_program_context",
            }
        )
    rows.sort(key=lambda item: (str(item["module_target"]), quarter_sort_key(str(item["time_period"])), str(item["indicator_gid"])))
    return rows


def _compile_kp_context_rows(path: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path is None:
        return rows
    for row in _read_csv(path):
        value = _float_or_none(row.get("Data value"))
        if value is None:
            continue
        rows.append(
            {
                "signal_id": f"unaids_kp::{row.get('Indicator_GId')}::{row.get('Subgroup_Val_GId')}::{row.get('Time Period')}",
                "source_id": "unaids_kp_atlas_2025",
                "module_target": "kp_overlay",
                "time_period": _quarter_from_year(str(row.get("Time Period") or "")),
                "time_granularity": "annual_or_survey_context",
                "indicator": row.get("Indicator"),
                "indicator_gid": row.get("Indicator_GId"),
                "subgroup": row.get("Subgroup"),
                "unit": row.get("Unit"),
                "value": value,
                "allowed_use": "determinant_sensitivity_until_source_stable",
                "measurement_semantics": "determinant_covariate_or_denominator",
            }
        )
    rows.sort(key=lambda item: (quarter_sort_key(str(item["time_period"])), str(item["indicator_gid"]), str(item["subgroup"])))
    return rows


def _compile_ncpi_policy_rows(path: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path is None:
        return rows
    for row in _read_csv(path):
        formatted = str(row.get("Formatted") or "").strip()
        value = _float_or_none(row.get("Data value"))
        if value is None and not formatted:
            continue
        rows.append(
            {
                "signal_id": f"unaids_ncpi::{row.get('Indicator_GId')}::{row.get('Subgroup_Val_GId')}::{row.get('Time Period')}",
                "source_id": "unaids_ncpi_2025",
                "module_target": "structural_policy",
                "linked_module_target": _indicator_module(str(row.get("Indicator") or "")),
                "time_period": _quarter_from_year(str(row.get("Time Period") or "")),
                "time_granularity": "annual_policy_context",
                "indicator": row.get("Indicator"),
                "indicator_gid": row.get("Indicator_GId"),
                "subgroup": row.get("Subgroup"),
                "unit": row.get("Unit"),
                "value": value if value is not None else formatted,
                "allowed_use": "structural_policy_context_not_target",
                "measurement_semantics": "policy_context",
            }
        )
    rows.sort(key=lambda item: (quarter_sort_key(str(item["time_period"])), str(item["indicator_gid"]), str(item["subgroup"])))
    return rows


def _compile_google_quarterly_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    metric_columns = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            geography = str(row.get("sub_region_1") or "national")
            period = _quarter_from_date(str(row.get("date") or ""))
            for column in metric_columns:
                value = _float_or_none(row.get(column))
                if value is not None:
                    grouped[(geography, period)][column].append(value)
    rows: list[dict[str, Any]] = []
    for (geography, period), metrics in grouped.items():
        for column, values in metrics.items():
            rows.append(
                {
                    "signal_id": f"google_mobility::{geography}::{period}::{column}",
                    "source_id": "google_global_mobility_report",
                    "module_target": "reporting_disruption",
                    "geography": geography,
                    "time_period": period,
                    "time_granularity": "quarterly_from_daily",
                    "indicator": column,
                    "value": sum(values) / len(values),
                    "days_observed": len(values),
                    "allowed_use": "reporting_disruption_or_mobility_sensitivity_covariate_2020_2022",
                    "measurement_semantics": "determinant_covariate_or_reporting_process_covariate",
                }
            )
    rows.sort(key=lambda item: (str(item["geography"]), quarter_sort_key(str(item["time_period"])), str(item["indicator"])))
    return rows


def _readiness_rows(
    *,
    annual_rows: list[dict[str, Any]],
    gam_rows: list[dict[str, Any]],
    kp_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    mobility_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets = {
        "R67-M01_hidden_service_intensity_state_space": [
            ("diagnosis_reporting", gam_rows),
            ("art_retention", gam_rows),
            ("vl_suppression_service", gam_rows),
            ("reporting_disruption", mobility_rows),
        ],
        "R67-M02_competing_risk_semi_markov_cascade": [
            ("diagnosis_reporting", gam_rows),
            ("art_retention", gam_rows),
            ("vl_suppression_service", gam_rows),
            ("mortality_reporting", annual_rows),
        ],
        "R67-M03_cd4_ahd_backcalculation_incidence": [
            ("incidence_validation", annual_rows),
            ("diagnosis_reporting", gam_rows),
        ],
        "R67-M04_kp_metapopulation_transmission_patch": [
            ("kp_overlay", kp_rows),
            ("incidence_validation", annual_rows),
            ("structural_policy", policy_rows),
        ],
        "R67-M06_service_capacity_queue_control": [
            ("diagnosis_reporting", gam_rows),
            ("art_retention", gam_rows),
            ("vl_suppression_service", gam_rows),
            ("prep_persistence", gam_rows),
            ("reporting_disruption", mobility_rows),
        ],
    }
    rows: list[dict[str, Any]] = []
    for family, requirements in buckets.items():
        counts: dict[str, int] = {}
        for module, source_rows in requirements:
            counts[module] = sum(1 for row in source_rows if row.get("module_target") == module or module in str(row.get("module_target") or ""))
        missing = [module for module, count in counts.items() if count == 0]
        rows.append(
            {
                "family_id": family,
                "readiness_status": "feature_table_ready" if not missing else "feature_table_partial",
                "missing_modules": "|".join(missing),
                "module_signal_counts": json_dumps_sorted(counts),
                "next_action": _next_action_for_family(family, missing),
            }
        )
    return rows


def json_dumps_sorted(payload: Any) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _next_action_for_family(family: str, missing: list[str]) -> str:
    if missing:
        return f"extract_missing_modules_before_fit:{'|'.join(missing)}"
    if family == "R67-M01_hidden_service_intensity_state_space":
        return "fit_train_only_reporting_intensity_state_with_google_mobility_and_GAM_support"
    if family == "R67-M06_service_capacity_queue_control":
        return "fit_capacity_queue_state_for_diagnosis_ART_VL_PrEP_with_stock_cone_gate"
    if family == "R67-M04_kp_metapopulation_transmission_patch":
        return "keep_sensitivity_only_until_R46_source_stable_determinants_pass"
    return "run_bounded_blocked_time_branch"


def _gate(readiness_rows: list[dict[str, Any]], total_rows: int) -> dict[str, Any]:
    ready = [row for row in readiness_rows if row.get("readiness_status") == "feature_table_ready"]
    return {
        "status": "bulk_signal_features_ready" if ready and total_rows > 0 else "bulk_signal_features_incomplete",
        "feature_table_ready_family_count": len(ready),
        "compiled_signal_row_count": total_rows,
        "contract": (
            "R69 compiles extracted bulk sources into model-ready signal tables. It still does not promote determinants: "
            "validation-only estimates remain annual weak/external evidence, KP and policy rows remain sensitivity until "
            "source-stable, and Google mobility is restricted to reporting/mobility disruption support."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("signal_feature_gate") or {})
    lines = [
        "# Phase 3 R69 Bulk Signal Feature Compiler",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Compiled signal rows: `{gate.get('compiled_signal_row_count')}`",
        f"- Ready model families: `{gate.get('feature_table_ready_family_count')}`",
        "",
        "## Family Readiness",
        "",
        "| Family | Status | Next action |",
        "|---|---|---|",
    ]
    for row in report.get("family_readiness_rows") or []:
        lines.append(f"| `{row.get('family_id')}` | `{row.get('readiness_status')}` | {row.get('next_action')} |")
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r69_bulk_signal_feature_compiler(
    *,
    run_id: str = R69_RUN_ID,
    r68_report_path: Path | None = None,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r68_path = R68_DEFAULT_REPORT if r68_report_path is None else Path(r68_report_path)
    r68 = dict(read_json(r68_path, default={}) or {}) if r68_path.exists() else {}
    annual_rows = _compile_unaids_estimates(_path_for_source(r68, "unaids_estimates_2025"))
    gam_rows = _compile_gam_program_rows(_path_for_source(r68, "unaids_gam_2025"))
    kp_rows = _compile_kp_context_rows(_path_for_source(r68, "unaids_kp_atlas_2025"))
    policy_rows = _compile_ncpi_policy_rows(_path_for_source(r68, "unaids_ncpi_2025"))
    mobility_rows = _compile_google_quarterly_rows(_path_for_source(r68, "google_global_mobility_report"))
    readiness_rows = _readiness_rows(
        annual_rows=annual_rows,
        gam_rows=gam_rows,
        kp_rows=kp_rows,
        policy_rows=policy_rows,
        mobility_rows=mobility_rows,
    )
    total_rows = len(annual_rows) + len(gam_rows) + len(kp_rows) + len(policy_rows) + len(mobility_rows)
    gate = _gate(readiness_rows, total_rows)
    report_path = analysis_dir / "r69_bulk_signal_feature_compiler_report.json"
    markdown_path = analysis_dir / "r69_bulk_signal_feature_compiler_report.md"
    annual_csv = analysis_dir / "r69_annual_external_challenge_rows.csv"
    gam_csv = analysis_dir / "r69_gam_program_support_rows.csv"
    kp_csv = analysis_dir / "r69_kp_context_rows.csv"
    policy_csv = analysis_dir / "r69_policy_context_rows.csv"
    mobility_csv = analysis_dir / "r69_google_mobility_quarterly_rows.csv"
    readiness_csv = analysis_dir / "r69_family_readiness_rows.csv"
    report = {
        "schema_version": R69_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "signal_feature_gate": gate,
        "family_readiness_rows": readiness_rows,
        "row_counts": {
            "annual_external_challenge_rows": len(annual_rows),
            "gam_program_support_rows": len(gam_rows),
            "kp_context_rows": len(kp_rows),
            "policy_context_rows": len(policy_rows),
            "google_mobility_quarterly_rows": len(mobility_rows),
        },
        "source_artifacts": {
            "r68": {"path": r68_path.as_posix(), "sha256": _sha256(r68_path) if r68_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "annual_external_challenge_csv": annual_csv.as_posix(),
            "gam_program_support_csv": gam_csv.as_posix(),
            "kp_context_csv": kp_csv.as_posix(),
            "policy_context_csv": policy_csv.as_posix(),
            "google_mobility_quarterly_csv": mobility_csv.as_posix(),
            "family_readiness_csv": readiness_csv.as_posix(),
        },
    }
    _write_csv(annual_csv, annual_rows)
    _write_csv(gam_csv, gam_rows)
    _write_csv(kp_csv, kp_rows)
    _write_csv(policy_csv, policy_rows)
    _write_csv(mobility_csv, mobility_rows)
    _write_csv(readiness_csv, readiness_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R69 bulk signal feature compiler.")
    parser.add_argument("--run-id", default=R69_RUN_ID)
    parser.add_argument("--r68-report-path", default=None)
    args = parser.parse_args()
    run_r69_bulk_signal_feature_compiler(
        run_id=str(args.run_id),
        r68_report_path=None if args.r68_report_path is None else Path(args.r68_report_path),
    )


if __name__ == "__main__":
    _main()
