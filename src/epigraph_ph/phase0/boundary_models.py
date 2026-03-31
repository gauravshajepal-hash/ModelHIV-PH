from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from epigraph_ph.geography import infer_philippines_geo, normalize_geo_label


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


class Phase0LiteratureRefBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str = Field(min_length=1)
    title: str | None = None
    year: int | None = None
    source_tier: str | None = None
    url: str | None = None
    doi: str | None = None
    pmid: str | None = None
    openalex_id: str | None = None


class Phase0CandidateBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    boundary_schema_version: str = "phase0_candidate_v1"
    candidate_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    candidate_text: str = Field(min_length=1)
    parameter_text: str = Field(min_length=1)
    evidence_span: str = Field(min_length=1)
    extraction_method: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_bank: str = Field(min_length=1)

    source_tier: str = ""
    source_title: str = ""
    platform: str = ""
    query_geo_focus: str = ""
    observation_id: str | None = None
    block_id: str | None = None
    chunk_id: str | None = None
    subparameter_id: str | None = None
    canonicalization_reason: str | None = None

    geo: str = ""
    region: str = ""
    province: str = ""
    population: str = ""
    time: str = Field(min_length=4)
    sex: str = ""
    age_band: str = ""
    kp_group: str = ""
    geo_mentions: list[str] = Field(default_factory=list)

    literature_ref_details: list[Phase0LiteratureRefBoundary] = Field(default_factory=list)
    linkage_targets: list[str] = Field(default_factory=list)
    soft_ontology_tags: list[str] = Field(default_factory=list)
    soft_subparameter_hints: list[str] = Field(default_factory=list)

    measurement_type: str = "unknown"
    denominator_type: str = "unknown"
    normalization_basis: str = "unknown"
    value_semantics: str = "bounded_proxy"
    value: float | None = None
    unit: str = ""

    is_anchor_eligible: bool = False
    is_direct_measurement: bool = False
    is_prior_only: bool = False

    geo_id: str = ""
    geo_scope: str = "unknown"
    geo_binding_class: str = "missing"
    signal_family: str = "general_context"
    value_presence_class: str = "soft_support"
    signal_evidence_count: int = 0
    textual_signal_support_count: int = 0
    support_signal_count: int = 0

    @field_validator("geo_mentions", "linkage_targets", "soft_ontology_tags", "soft_subparameter_hints", mode="before")
    @classmethod
    def _dedupe_string_list(cls, value: Any) -> list[str]:
        items = value or []
        if not isinstance(items, list):
            items = [items]
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(text)
        return cleaned

    @field_validator("time")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        text = str(value or "").strip()
        if not re.fullmatch(r"(?:19|20)\d{2}(?:-(?:0[1-9]|1[0-2]))?", text):
            raise ValueError("invalid_time_binding")
        return text

    @model_validator(mode="after")
    def _derive_fields(self) -> "Phase0CandidateBoundary":
        self.geo, self.region, self.province, self.geo_binding_class = _infer_boundary_geo(self)
        self.geo_scope = _infer_geo_scope(self.geo, self.region, self.province)
        self.geo_id = _infer_geo_id(self.geo, self.region, self.province)
        self.signal_family = _infer_signal_family(
            canonical_name=self.canonical_name,
            soft_tags=self.soft_ontology_tags,
            soft_hints=self.soft_subparameter_hints,
            linkage_targets=self.linkage_targets,
            text=" ".join(
                part
                for part in (self.canonical_name, self.parameter_text, self.candidate_text, self.evidence_span)
                if part
            ),
        )
        self.value_presence_class = "numeric_observed" if self.value is not None else "soft_support"
        self.signal_evidence_count = int(self.value is not None) + len(self.soft_ontology_tags) + len(self.soft_subparameter_hints) + len(self.linkage_targets)
        self.textual_signal_support_count = _textual_signal_support_count(self)
        self.support_signal_count = self.signal_evidence_count + self.textual_signal_support_count
        return self


class Phase0CandidateEnvelopeBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    boundary_schema_version: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_bank: str = Field(min_length=1)
    source_tier: str = ""
    extraction_method: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    candidate_text: str = Field(min_length=1)
    parameter_text: str = Field(min_length=1)
    evidence_span: str = Field(min_length=1)
    geo: str = ""
    region: str = ""
    province: str = ""
    geo_id: str = ""
    geo_scope: str = "unknown"
    geo_binding_class: str = "missing"
    time: str = Field(min_length=4)
    geo_mentions: list[str] = Field(default_factory=list)
    signal_family: str = Field(min_length=1)
    value_presence_class: str = Field(min_length=1)
    signal_evidence_count: int = Field(ge=0)
    textual_signal_support_count: int = Field(ge=0)
    support_signal_count: int = Field(ge=0)
    linkage_targets: list[str] = Field(default_factory=list)
    literature_ref_details: list[Phase0LiteratureRefBoundary] = Field(default_factory=list)
    is_anchor_eligible: bool = False
    is_direct_measurement: bool = False
    is_prior_only: bool = False

    @field_validator("geo_mentions", "linkage_targets", mode="before")
    @classmethod
    def _dedupe_envelope_lists(cls, value: Any) -> list[str]:
        items = value or []
        if not isinstance(items, list):
            items = [items]
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in items:
            text = str(item or "").strip()
            lowered = text.lower()
            if not text or lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(text)
        return cleaned


class Phase0FamilyPayloadBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    payload_schema_version: str = Field(min_length=1)
    payload_family: str = Field(min_length=1)
    signal_family: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    measurement_type: str = "unknown"
    denominator_type: str = "unknown"
    normalization_basis: str = "unknown"
    value_semantics: str = "bounded_proxy"
    value: float | None = None
    unit: str = ""
    geo_scope: str = "unknown"
    geo_id: str = ""
    linkage_targets: list[str] = Field(default_factory=list)
    soft_ontology_tags: list[str] = Field(default_factory=list)
    soft_subparameter_hints: list[str] = Field(default_factory=list)


class PopulationMeasurePayload(Phase0FamilyPayloadBase):
    payload_family: Literal["PopulationMeasure"] = "PopulationMeasure"
    population: str = ""
    sex: str = ""
    age_band: str = ""
    kp_group: str = ""


class LogisticsAccessPayload(Phase0FamilyPayloadBase):
    payload_family: Literal["LogisticsAccess"] = "LogisticsAccess"
    region: str = ""
    province: str = ""
    geo_mentions: list[str] = Field(default_factory=list)


class BehaviorSignalPayload(Phase0FamilyPayloadBase):
    payload_family: Literal["BehaviorSignal"] = "BehaviorSignal"
    sex: str = ""
    age_band: str = ""
    kp_group: str = ""


class ServiceCapacityPayload(Phase0FamilyPayloadBase):
    payload_family: Literal["ServiceCapacity"] = "ServiceCapacity"
    is_anchor_eligible: bool = False
    is_direct_measurement: bool = False
    observation_id: str | None = None


class EconomicConstraintPayload(Phase0FamilyPayloadBase):
    payload_family: Literal["EconomicConstraint"] = "EconomicConstraint"
    population: str = ""
    region: str = ""
    province: str = ""


class PolicyEnvironmentPayload(Phase0FamilyPayloadBase):
    payload_family: Literal["PolicyEnvironment"] = "PolicyEnvironment"
    source_tier: str = ""
    extraction_method: str = ""


class CascadeObservationPayload(Phase0FamilyPayloadBase):
    payload_family: Literal["CascadeObservation"] = "CascadeObservation"
    observation_id: str | None = None
    is_anchor_eligible: bool = False
    is_direct_measurement: bool = False
    is_prior_only: bool = False


def _infer_geo_scope(geo: str, region: str, province: str) -> str:
    province_text = str(province or "").strip()
    region_text = str(region or "").strip().lower()
    geo_text = str(geo or "").strip().lower()
    if province_text:
        return "province"
    if region_text and region_text != "national":
        return "region"
    if region_text == "national" or geo_text == "philippines":
        return "national"
    if geo_text:
        return "geo"
    return "unknown"


def _infer_geo_id(geo: str, region: str, province: str) -> str:
    province_text = str(province or "").strip()
    region_text = str(region or "").strip()
    geo_text = str(geo or "").strip()
    if province_text:
        return f"province:{_slug(province_text)}"
    if region_text and region_text.lower() != "national":
        return f"region:{_slug(region_text)}"
    if region_text.lower() == "national" or geo_text.lower() == "philippines":
        return "national:philippines"
    if geo_text:
        return f"geo:{_slug(geo_text)}"
    return ""


def _infer_signal_family(*, canonical_name: str, soft_tags: list[str], soft_hints: list[str], linkage_targets: list[str], text: str) -> str:
    tokens = " ".join([canonical_name, *soft_tags, *soft_hints, *linkage_targets, text]).lower()
    rules = [
        ("mobility_logistics", ("mobility", "migration", "transport", "travel", "logistics", "remoteness", "corridor", "congestion")),
        ("behavior_stigma", ("stigma", "behavior", "knowledge", "awareness", "testing_uptake", "sexual_risk", "retention", "adherence", "linkage_to_care")),
        ("service_delivery", ("clinic", "facility", "hub", "service", "viral_load", "suppression", "art", "health_system")),
        ("population_demography", ("population", "demograph", "household", "msm", "kp_population", "education", "literacy")),
        ("economics_access", ("poverty", "income", "cash", "housing", "philhealth", "expenditure", "gdp", "afford")),
        ("policy_environment", ("policy", "governance", "implementation", "disruption", "typhoon", "climate")),
        ("epidemiology_cascade", ("case_count", "population_count", "prevalence", "incidence", "diagnosed", "plhiv")),
    ]
    for family, family_tokens in rules:
        if any(token in tokens for token in family_tokens):
            return family
    return "general_context"


def _payload_family_for_signal_family(signal_family: str) -> str:
    mapping = {
        "population_demography": "PopulationMeasure",
        "mobility_logistics": "LogisticsAccess",
        "behavior_stigma": "BehaviorSignal",
        "service_delivery": "ServiceCapacity",
        "economics_access": "EconomicConstraint",
        "policy_environment": "PolicyEnvironment",
        "epidemiology_cascade": "CascadeObservation",
        "general_context": "PolicyEnvironment",
    }
    return mapping.get(str(signal_family or "").strip(), "PolicyEnvironment")


def _payload_schema_version(payload_family: str) -> str:
    return f"phase0_{_slug(payload_family)}_v1"


_GENERIC_CANONICAL_NAMES = {"", "unknown", "numeric_observation", "literature_seed"}
_TEXTUAL_SIGNAL_TOKENS = (
    "hiv",
    "art",
    "suppression",
    "diagnos",
    "testing",
    "mobility",
    "transport",
    "stigma",
    "clinic",
    "facility",
    "poverty",
    "policy",
    "migration",
    "population",
    "linkage",
    "retention",
    "prevention",
)


def _combined_candidate_text(candidate: Phase0CandidateBoundary) -> str:
    literature_titles = " ".join(str(ref.title or "") for ref in candidate.literature_ref_details)
    return " ".join(
        part
        for part in (
            candidate.document_id,
            candidate.source_id,
            candidate.source_title,
            literature_titles,
            candidate.candidate_text,
            candidate.parameter_text,
            candidate.evidence_span,
            " ".join(candidate.geo_mentions),
            candidate.query_geo_focus,
            candidate.geo,
            candidate.region,
            candidate.province,
        )
        if part
    ).strip()


def _infer_boundary_geo(candidate: Phase0CandidateBoundary) -> tuple[str, str, str, str]:
    geo = str(candidate.geo or "").strip()
    region = str(candidate.region or "").strip()
    province = str(candidate.province or "").strip()
    explicit_geo_present = bool(geo or region or province)
    explicit_country_focus = str(candidate.query_geo_focus or "").strip().lower() == "philippines"
    combined_text = _combined_candidate_text(candidate)
    match = infer_philippines_geo(
        combined_text,
        default_country_focus=explicit_country_focus or any(
            "philippines" in str(ref.title or "").lower() for ref in candidate.literature_ref_details
        ),
    )
    if province:
        resolved_geo = normalize_geo_label(province, default_country_focus=explicit_country_focus) or province
        resolved_region = region or (match.region_display if match.region_display and match.region != "national" else "")
        return resolved_geo, resolved_region, province, "explicit_geo"
    if geo:
        resolved_geo = normalize_geo_label(geo, default_country_focus=explicit_country_focus) or geo
        resolved_region = region or (match.region_display if match.region_display and match.region != "national" else ("national" if match.region == "national" else ""))
        resolved_province = province or match.province
        binding_class = "explicit_geo" if explicit_geo_present else "text_inferred"
        return resolved_geo, resolved_region, resolved_province, binding_class
    if match.resolution != "unknown":
        inferred_geo = normalize_geo_label(match.geo, default_country_focus=explicit_country_focus) or match.geo
        inferred_region = "national" if match.region == "national" else (match.region_display or region)
        binding_class = "text_inferred_subnational" if match.resolution in {"province", "region"} else "text_inferred_national"
        return inferred_geo, inferred_region, match.province, binding_class
    if explicit_country_focus:
        return "Philippines", "national", "", "query_geo_focus_national"
    return geo, region, province, "missing"


def _textual_signal_support_count(candidate: Phase0CandidateBoundary) -> int:
    combined_text = _combined_candidate_text(candidate).lower()
    count = 0
    if str(candidate.canonical_name or "").strip().lower() not in _GENERIC_CANONICAL_NAMES:
        count += 1
    if candidate.value is not None:
        count += 1
    if candidate.literature_ref_details:
        count += 1
    if candidate.geo_id or candidate.geo_binding_class != "missing" or str(candidate.query_geo_focus or "").strip():
        count += 1
    if candidate.signal_family != "general_context":
        count += 1
    if re.search(r"[a-z]{3,}", str(candidate.parameter_text or "").lower()) or re.search(r"[a-z]{3,}", str(candidate.evidence_span or "").lower()):
        count += 1
    if any(token in combined_text for token in _TEXTUAL_SIGNAL_TOKENS):
        count += 1
    return count


def build_phase0_candidate_envelope(row: dict[str, Any]) -> dict[str, Any]:
    envelope = Phase0CandidateEnvelopeBoundary.model_validate(
        {
            "boundary_schema_version": row.get("boundary_schema_version") or "phase0_candidate_v1",
            "candidate_id": row.get("candidate_id"),
            "document_id": row.get("document_id"),
            "source_id": row.get("source_id"),
            "source_bank": row.get("source_bank"),
            "source_tier": row.get("source_tier") or "",
            "extraction_method": row.get("extraction_method"),
            "confidence": row.get("confidence"),
            "candidate_text": row.get("candidate_text"),
            "parameter_text": row.get("parameter_text"),
            "evidence_span": row.get("evidence_span"),
            "geo": row.get("geo") or "",
            "region": row.get("region") or "",
            "province": row.get("province") or "",
            "geo_id": row.get("geo_id") or "",
            "geo_scope": row.get("geo_scope") or "unknown",
            "geo_binding_class": row.get("geo_binding_class") or "missing",
            "time": row.get("time"),
            "geo_mentions": row.get("geo_mentions") or [],
            "signal_family": row.get("signal_family") or "general_context",
            "value_presence_class": row.get("value_presence_class") or "soft_support",
            "signal_evidence_count": row.get("signal_evidence_count") or 0,
            "textual_signal_support_count": row.get("textual_signal_support_count") or 0,
            "support_signal_count": row.get("support_signal_count") or 0,
            "linkage_targets": row.get("linkage_targets") or [],
            "literature_ref_details": row.get("literature_ref_details") or [],
            "is_anchor_eligible": bool(row.get("is_anchor_eligible")),
            "is_direct_measurement": bool(row.get("is_direct_measurement")),
            "is_prior_only": bool(row.get("is_prior_only")),
        }
    )
    return envelope.model_dump(mode="json")


def build_phase0_candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload_family = str(row.get("payload_family") or _payload_family_for_signal_family(str(row.get("signal_family") or "")))
    payload_common = {
        "payload_schema_version": str(row.get("payload_schema_version") or _payload_schema_version(payload_family)),
        "payload_family": payload_family,
        "signal_family": row.get("signal_family") or "general_context",
        "canonical_name": row.get("canonical_name"),
        "measurement_type": row.get("measurement_type") or "unknown",
        "denominator_type": row.get("denominator_type") or "unknown",
        "normalization_basis": row.get("normalization_basis") or "unknown",
        "value_semantics": row.get("value_semantics") or "bounded_proxy",
        "value": row.get("value"),
        "unit": row.get("unit") or "",
        "geo_scope": row.get("geo_scope") or "unknown",
        "geo_id": row.get("geo_id") or "",
        "linkage_targets": row.get("linkage_targets") or [],
        "soft_ontology_tags": row.get("soft_ontology_tags") or [],
        "soft_subparameter_hints": row.get("soft_subparameter_hints") or [],
    }
    payload_model: type[Phase0FamilyPayloadBase]
    payload_kwargs: dict[str, Any] = dict(payload_common)
    if payload_family == "PopulationMeasure":
        payload_model = PopulationMeasurePayload
        payload_kwargs.update(
            {
                "population": row.get("population") or "",
                "sex": row.get("sex") or "",
                "age_band": row.get("age_band") or "",
                "kp_group": row.get("kp_group") or "",
            }
        )
    elif payload_family == "LogisticsAccess":
        payload_model = LogisticsAccessPayload
        payload_kwargs.update(
            {
                "region": row.get("region") or "",
                "province": row.get("province") or "",
                "geo_mentions": row.get("geo_mentions") or [],
            }
        )
    elif payload_family == "BehaviorSignal":
        payload_model = BehaviorSignalPayload
        payload_kwargs.update(
            {
                "sex": row.get("sex") or "",
                "age_band": row.get("age_band") or "",
                "kp_group": row.get("kp_group") or "",
            }
        )
    elif payload_family == "ServiceCapacity":
        payload_model = ServiceCapacityPayload
        payload_kwargs.update(
            {
                "is_anchor_eligible": bool(row.get("is_anchor_eligible")),
                "is_direct_measurement": bool(row.get("is_direct_measurement")),
                "observation_id": row.get("observation_id"),
            }
        )
    elif payload_family == "EconomicConstraint":
        payload_model = EconomicConstraintPayload
        payload_kwargs.update(
            {
                "population": row.get("population") or "",
                "region": row.get("region") or "",
                "province": row.get("province") or "",
            }
        )
    elif payload_family == "CascadeObservation":
        payload_model = CascadeObservationPayload
        payload_kwargs.update(
            {
                "observation_id": row.get("observation_id"),
                "is_anchor_eligible": bool(row.get("is_anchor_eligible")),
                "is_direct_measurement": bool(row.get("is_direct_measurement")),
                "is_prior_only": bool(row.get("is_prior_only")),
            }
        )
    else:
        payload_model = PolicyEnvironmentPayload
        payload_kwargs.update(
            {
                "source_tier": row.get("source_tier") or "",
                "extraction_method": row.get("extraction_method") or "",
            }
        )
    payload = payload_model.model_validate(payload_kwargs)
    return payload.model_dump(mode="json")


def build_phase0_family_candidate_banks(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    banks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        envelope = build_phase0_candidate_envelope(row)
        payload = build_phase0_candidate_payload(row)
        banks[payload["payload_family"]].append(
            {
                "candidate_id": envelope["candidate_id"],
                "canonical_name": payload["canonical_name"],
                "envelope": envelope,
                "payload": payload,
            }
        )
    manifest_rows = []
    for family_name, family_rows in sorted(banks.items()):
        manifest_rows.append(
            {
                "payload_family": family_name,
                "payload_schema_version": _payload_schema_version(family_name),
                "row_count": len(family_rows),
                "direct_measurement_count": sum(1 for row in family_rows if row["envelope"].get("is_direct_measurement")),
                "prior_only_count": sum(1 for row in family_rows if row["envelope"].get("is_prior_only")),
            }
        )
    return dict(banks), {
        "boundary_schema_version": "phase0_candidate_v1",
        "family_bank_count": len(manifest_rows),
        "families": manifest_rows,
    }


def validate_phase0_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    validation_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    allowed_measurement_types = {str(item) for item in validation_cfg.get("allowed_measurement_types", [])}
    allowed_denominator_types = {str(item) for item in validation_cfg.get("allowed_denominator_types", [])}
    allowed_normalization_bases = {str(item) for item in validation_cfg.get("allowed_normalization_basis", [])}
    allowed_value_semantics = {str(item) for item in validation_cfg.get("allowed_value_semantics", [])}
    require_geo_binding = bool(validation_cfg.get("require_geo_binding", True))
    require_time_binding = bool(validation_cfg.get("require_time_binding", True))
    minimum_signal_fields = int(validation_cfg.get("minimum_signal_fields", 1))
    soft_text_support_floor = int(validation_cfg.get("soft_text_support_floor", 3))
    numeric_prior_signal_floor = int(validation_cfg.get("numeric_prior_signal_floor", 2))
    schema_version = str(validation_cfg.get("schema_version") or "phase0_candidate_v1")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    signal_family_counts: Counter[str] = Counter()
    geo_scope_counts: Counter[str] = Counter()
    geo_binding_class_counts: Counter[str] = Counter()
    value_presence_counts: Counter[str] = Counter()
    payload_family_counts: Counter[str] = Counter()

    for row in rows:
        patched = dict(row)
        patched["boundary_schema_version"] = schema_version
        try:
            model = Phase0CandidateBoundary.model_validate(patched)
            semantic_errors: list[str] = []
            if allowed_measurement_types and model.measurement_type not in allowed_measurement_types:
                semantic_errors.append("invalid_measurement_type")
            if allowed_denominator_types and model.denominator_type not in allowed_denominator_types:
                semantic_errors.append("invalid_denominator_type")
            if allowed_normalization_bases and model.normalization_basis not in allowed_normalization_bases:
                semantic_errors.append("invalid_normalization_basis")
            if allowed_value_semantics and model.value_semantics not in allowed_value_semantics:
                semantic_errors.append("invalid_value_semantics")
            if require_geo_binding and not model.geo_id:
                semantic_errors.append("missing_geo_binding")
            if require_time_binding and not model.time:
                semantic_errors.append("missing_time_binding")
            if model.is_direct_measurement and model.value is None:
                semantic_errors.append("direct_measurement_requires_value")
            has_explicit_signal = bool(model.soft_ontology_tags or model.soft_subparameter_hints or model.linkage_targets)
            has_textual_support = model.support_signal_count >= soft_text_support_floor
            has_numeric_prior_support = model.value is not None and model.textual_signal_support_count >= numeric_prior_signal_floor
            if model.is_prior_only and not (has_explicit_signal or has_textual_support or has_numeric_prior_support):
                semantic_errors.append("soft_candidate_requires_signal")
            if model.support_signal_count < minimum_signal_fields:
                semantic_errors.append("insufficient_signal_fields")
            if semantic_errors:
                raise ValueError(",".join(semantic_errors))
            dumped = model.model_dump(mode="json")
            dumped["payload_family"] = _payload_family_for_signal_family(dumped["signal_family"])
            dumped["payload_schema_version"] = _payload_schema_version(dumped["payload_family"])
            accepted.append(dumped)
            signal_family_counts[dumped["signal_family"]] += 1
            geo_scope_counts[dumped["geo_scope"]] += 1
            geo_binding_class_counts[dumped["geo_binding_class"]] += 1
            value_presence_counts[dumped["value_presence_class"]] += 1
            payload_family_counts[dumped["payload_family"]] += 1
        except ValidationError as exc:
            reasons = [str(error.get("msg") or error.get("type") or "validation_error") for error in exc.errors()]
            for reason in reasons:
                rejection_counts[reason] += 1
            rejected.append(
                {
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "canonical_name": str(row.get("canonical_name") or ""),
                    "source_id": str(row.get("source_id") or ""),
                    "errors": exc.errors(),
                    "row": row,
                }
            )
        except ValueError as exc:
            reasons = [item for item in str(exc).split(",") if item]
            for reason in reasons:
                rejection_counts[reason] += 1
            rejected.append(
                {
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "canonical_name": str(row.get("canonical_name") or ""),
                    "source_id": str(row.get("source_id") or ""),
                    "errors": [{"msg": reason, "type": "semantic_validation"} for reason in reasons],
                    "row": row,
                }
            )

    summary = {
        "schema_version": schema_version,
        "candidate_count_in": len(rows),
        "candidate_count_accepted": len(accepted),
        "candidate_count_rejected": len(rejected),
        "acceptance_rate": round(len(accepted) / max(1, len(rows)), 4),
        "rejection_reason_counts": dict(rejection_counts),
        "signal_family_counts": dict(signal_family_counts),
        "payload_family_counts": dict(payload_family_counts),
        "geo_scope_counts": dict(geo_scope_counts),
        "geo_binding_class_counts": dict(geo_binding_class_counts),
        "value_presence_counts": dict(value_presence_counts),
    }
    return accepted, rejected, summary
