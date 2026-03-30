from __future__ import annotations

from epigraph_ph.core.disease_plugin import StructuredSourceAdapterSpec


def _hiv_structured_source_adapters() -> list[StructuredSourceAdapterSpec]:
    return [
        StructuredSourceAdapterSpec(
            adapter_id="ndhs",
            source_name="National Demographic and Health Survey",
            organization="Philippine Statistics Authority",
            source_tier="tier2_official_survey",
            access_mode="public_survey_report",
            spatial_resolution="national_region_survey_wave",
            temporal_resolution="survey_wave",
            landing_url="https://psa.gov.ph/",
            determinant_silos=["sexual_risk", "prevention_access", "testing_uptake", "education", "social_capital"],
            promotion_track="supporting_context",
            platform="ndhs",
            seed_queries=[
                "Philippines NDHS sexual behavior HIV testing uptake",
                "Philippines demographic health survey prevention access education health literacy",
            ],
            preferred_file_patterns=["*ndhs*", "*demographic*health*survey*"],
            fallback_urls=["https://dhsprogram.com/"],
            notes=[
                "Use as a survey-wave behavior and access source with uncertainty retained through alignment.",
                "Do not coerce to province-month without partial pooling or explicit downscaling uncertainty.",
            ],
        ),
        StructuredSourceAdapterSpec(
            adapter_id="yafs",
            source_name="Young Adult Fertility and Sexuality Study",
            organization="UP Population Institute / DRDF",
            source_tier="tier2_official_survey",
            access_mode="public_survey_report",
            spatial_resolution="national_region_survey_wave",
            temporal_resolution="survey_wave",
            landing_url="https://www.uppi.upd.edu.ph/",
            determinant_silos=["sexual_risk", "collective_risk_behavior", "education", "testing_uptake"],
            promotion_track="supporting_context",
            platform="yafs",
            seed_queries=[
                "Philippines YAFS sexual risk youth testing uptake",
                "Young adult fertility sexuality study Philippines risk behavior education",
            ],
            preferred_file_patterns=["*yafs*", "*fertility*sexuality*"],
            fallback_urls=["https://www.uppi.upd.edu.ph/"],
            notes=[
                "Treat YAFS as a youth and emerging-adult risk-behavior substrate rather than a direct cascade anchor.",
            ],
        ),
        StructuredSourceAdapterSpec(
            adapter_id="fies",
            source_name="Family Income and Expenditure Survey",
            organization="Philippine Statistics Authority",
            source_tier="tier2_official_survey",
            access_mode="public_aggregate_survey",
            spatial_resolution="national_region_province_wave",
            temporal_resolution="survey_wave",
            landing_url="https://psa.gov.ph/",
            determinant_silos=["poverty", "cash_instability", "housing_precarity", "education"],
            promotion_track="supporting_context",
            platform="fies",
            seed_queries=[
                "Philippines FIES poverty cash instability healthcare affordability",
                "Family income expenditure survey Philippines remoteness housing precarity",
            ],
            preferred_file_patterns=["*fies*", "*family*income*expenditure*"],
            fallback_urls=["https://psada.psa.gov.ph/"],
            notes=[
                "Use as the main household economic capability and instability adapter.",
                "Supports affordability and continuity-of-care factors rather than direct HIV state replacement.",
            ],
        ),
        StructuredSourceAdapterSpec(
            adapter_id="philgis_boundary_proxy",
            source_name="PhilGIS / PSGC Boundary Proxy",
            organization="PSA PSGC / archival boundary references",
            source_tier="tier3_structured_repository",
            access_mode="public_boundary_catalog",
            spatial_resolution="province_city_region_static",
            temporal_resolution="static",
            landing_url="https://psa.gov.ph/classification/psgc/",
            determinant_silos=["remoteness", "transport_friction", "health_system_reach", "mobility_network_mixing"],
            promotion_track="supporting_context",
            platform="philgis_psgc",
            seed_queries=[
                "Philippines province region shapefile remoteness health access",
                "Philippines PSGC boundaries transport reach travel geometry",
            ],
            preferred_file_patterns=["*psgc*", "*boundary*", "*shapefile*", "*geojson*"],
            fallback_urls=["https://www.openstreetmap.org/", "https://data.humdata.org/"],
            notes=[
                "The old public PhilGIS domain is not trusted as a current landing page; use PSGC and open boundary mirrors as the safer active proxy.",
                "This adapter seeds geometry and travel-friction literature and repository retrieval.",
            ],
        ),
        StructuredSourceAdapterSpec(
            adapter_id="transport_network_proxies",
            source_name="Transport and Network Proxy Sources",
            organization="OpenStreetMap / HDX / open route proxies",
            source_tier="tier4_proxy_dataset",
            access_mode="public_proxy_repository",
            spatial_resolution="segment_hub_route",
            temporal_resolution="mixed",
            landing_url="https://www.openstreetmap.org/",
            determinant_silos=["transport_friction", "congestion_travel_time", "mobility_network_mixing", "remoteness", "health_system_reach"],
            promotion_track="supporting_context",
            platform="transport_proxy",
            seed_queries=[
                "Philippines ferry flight road network travel time health access",
                "Philippines commuting congestion treatment access route proxies",
                "Philippines transport network remoteness mobility mixing care continuity",
            ],
            preferred_file_patterns=["*road*", "*route*", "*travel*time*", "*airport*", "*ferry*"],
            fallback_urls=["https://data.humdata.org/", "https://www.geofabrik.de/"],
            notes=[
                "Use as open fallback when gated mobility feeds are unavailable.",
                "These are proxies and should remain down-weighted unless validated against direct program patterns.",
            ],
        ),
    ]


def get_structured_source_adapters(plugin_id: str) -> list[StructuredSourceAdapterSpec]:
    if plugin_id == "hiv":
        return _hiv_structured_source_adapters()
    return []
