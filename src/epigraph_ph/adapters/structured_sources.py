from __future__ import annotations

from epigraph_ph.core.disease_plugin import StructuredSourceAdapterSpec


def _historical_record(*, title: str, url: str, year: int | None, document_type: str = "dataset_catalog", notes: list[str] | None = None) -> dict[str, object]:
    return {
        "title": title,
        "url": url,
        "year": year,
        "document_type": document_type,
        "notes": list(notes or []),
    }


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
            historical_records=[
                _historical_record(
                    title="Philippines National Demographic and Health Survey 2013",
                    url="https://dhsprogram.com/publications/publication-fr294-dhs-final-reports.cfm",
                    year=2013,
                    document_type="survey_report_pdf",
                ),
                _historical_record(
                    title="Philippines National Demographic and Health Survey 2017",
                    url="https://psa.gov.ph/content/national-demographic-and-health-survey-ndhs",
                    year=2017,
                    document_type="survey_report_pdf",
                ),
                _historical_record(
                    title="Philippines National Demographic and Health Survey 2022",
                    url="https://psa.gov.ph/content/2022-national-demographic-and-health-survey-ndhs-key-findings",
                    year=2022,
                    document_type="survey_report_pdf",
                ),
            ],
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
            historical_records=[
                _historical_record(
                    title="Young Adult Fertility and Sexuality Study 4",
                    url="https://www.uppi.upd.edu.ph/yafs4",
                    year=2013,
                    document_type="survey_report_pdf",
                ),
                _historical_record(
                    title="Young Adult Fertility and Sexuality Study 5 National Report",
                    url="https://www.uppi.upd.edu.ph/yafs5",
                    year=2021,
                    document_type="survey_report_pdf",
                ),
                _historical_record(
                    title="YAFS 5 Key Indicators and Regional Tables",
                    url="https://www.uppi.upd.edu.ph/yafs5",
                    year=2021,
                    document_type="survey_table_pack",
                ),
            ],
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
            historical_records=[
                _historical_record(
                    title="Family Income and Expenditure Survey 2012",
                    url="https://psa.gov.ph/statistics/income-expenditure/fies",
                    year=2012,
                    document_type="survey_report_pdf",
                ),
                _historical_record(
                    title="Family Income and Expenditure Survey 2015",
                    url="https://psa.gov.ph/statistics/income-expenditure/fies",
                    year=2015,
                    document_type="survey_report_pdf",
                ),
                _historical_record(
                    title="Family Income and Expenditure Survey 2018",
                    url="https://psa.gov.ph/statistics/income-expenditure/fies",
                    year=2018,
                    document_type="survey_report_pdf",
                ),
                _historical_record(
                    title="Family Income and Expenditure Survey 2021",
                    url="https://psa.gov.ph/statistics/income-expenditure/fies",
                    year=2021,
                    document_type="survey_report_pdf",
                ),
            ],
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
            historical_records=[
                _historical_record(
                    title="Philippine Standard Geographic Code Province List 2015",
                    url="https://psa.gov.ph/classification/psgc/",
                    year=2015,
                    document_type="boundary_catalog",
                ),
                _historical_record(
                    title="Philippine Standard Geographic Code Province List 2020",
                    url="https://psa.gov.ph/classification/psgc/",
                    year=2020,
                    document_type="boundary_catalog",
                ),
                _historical_record(
                    title="Philippine Standard Geographic Code Province List 2024",
                    url="https://psa.gov.ph/classification/psgc/",
                    year=2024,
                    document_type="boundary_catalog",
                ),
            ],
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
            historical_records=[
                _historical_record(
                    title="OpenStreetMap Philippines Extract",
                    url="https://download.geofabrik.de/asia/philippines.html",
                    year=2024,
                    document_type="network_extract",
                ),
                _historical_record(
                    title="HDX Philippines Transport Accessibility Proxy",
                    url="https://data.humdata.org/",
                    year=2021,
                    document_type="proxy_dataset",
                ),
                _historical_record(
                    title="OpenFlights Philippines Airport and Route Proxy",
                    url="https://openflights.org/data.html",
                    year=2024,
                    document_type="network_extract",
                ),
            ],
            notes=[
                "Use as open fallback when gated mobility feeds are unavailable.",
                "These are proxies and should remain down-weighted unless validated against direct program patterns.",
            ],
        ),
        StructuredSourceAdapterSpec(
            adapter_id="google_mobility",
            source_name="Google Community Mobility Reports",
            organization="Google",
            source_tier="tier3_structured_repository",
            access_mode="public_csv_archive",
            spatial_resolution="region_province_week",
            temporal_resolution="daily_weekly",
            landing_url="https://www.google.com/covid19/mobility/",
            determinant_silos=["mobility_network_mixing", "congestion_travel_time", "transport_friction", "health_system_reach"],
            promotion_track="supporting_context",
            platform="google_mobility",
            seed_queries=[
                "Philippines Google mobility transit workplace retail province",
                "Philippines mobility reports travel time congestion access",
            ],
            preferred_file_patterns=["*mobility*", "*region*csv*", "*philippines*mobility*"],
            fallback_urls=["https://www.google.com/covid19/mobility/", "https://www.kaggle.com/"],
            historical_records=[
                _historical_record(
                    title="Google Community Mobility Reports Philippines 2020",
                    url="https://www.google.com/covid19/mobility/",
                    year=2020,
                    document_type="csv_archive",
                ),
                _historical_record(
                    title="Google Community Mobility Reports Philippines 2021",
                    url="https://www.google.com/covid19/mobility/",
                    year=2021,
                    document_type="csv_archive",
                ),
                _historical_record(
                    title="Google Community Mobility Reports Philippines 2022",
                    url="https://www.google.com/covid19/mobility/",
                    year=2022,
                    document_type="csv_archive",
                ),
            ],
            notes=[
                "Use as a relative movement and congestion proxy rather than a direct epidemiologic anchor.",
                "Historical coverage matters more than current update cadence because the public Google series ended in 2022.",
            ],
        ),
        StructuredSourceAdapterSpec(
            adapter_id="world_bank_wdi",
            source_name="World Bank World Development Indicators",
            organization="World Bank",
            source_tier="tier3_structured_repository",
            access_mode="public_api_csv",
            spatial_resolution="national_annual",
            temporal_resolution="annual",
            landing_url="https://data.worldbank.org/",
            determinant_silos=["poverty", "education", "health_system_reach", "cash_instability"],
            promotion_track="supporting_context",
            platform="world_bank_wdi",
            seed_queries=[
                "Philippines World Bank WDI poverty health expenditure education",
                "Philippines WDI GDP per capita health spending poverty rate",
            ],
            preferred_file_patterns=["*wdi*", "*world*development*indicators*", "*api*csv*"],
            fallback_urls=["https://api.worldbank.org/"],
            historical_records=[
                _historical_record(
                    title="WDI Poverty Headcount Ratio Philippines",
                    url="https://api.worldbank.org/v2/country/PHL/indicator/SI.POV.NAHC?format=json",
                    year=2024,
                    document_type="api_series",
                ),
                _historical_record(
                    title="WDI Current Health Expenditure per Capita Philippines",
                    url="https://api.worldbank.org/v2/country/PHL/indicator/SH.XPD.CHEX.PC.CD?format=json",
                    year=2024,
                    document_type="api_series",
                ),
                _historical_record(
                    title="WDI Lower Secondary Completion Rate Philippines",
                    url="https://api.worldbank.org/v2/country/PHL/indicator/SE.SEC.CMPT.LO.ZS?format=json",
                    year=2024,
                    document_type="api_series",
                ),
            ],
            notes=[
                "Treat WDI as a national macro-structural context layer.",
                "Do not over-interpret WDI as province-specific evidence without explicit downscaling uncertainty.",
            ],
        ),
        StructuredSourceAdapterSpec(
            adapter_id="philhealth_reports",
            source_name="PhilHealth Annual Reports",
            organization="PhilHealth",
            source_tier="tier2_official_survey",
            access_mode="public_report_pdf_excel",
            spatial_resolution="national_region_annual",
            temporal_resolution="annual",
            landing_url="https://www.philhealth.gov.ph/about_us/statsncharts/",
            determinant_silos=["health_system_reach", "poverty", "cash_instability", "policy_implementation_weakness"],
            promotion_track="supporting_context",
            platform="philhealth",
            seed_queries=[
                "PhilHealth annual report enrollment claims regional coverage",
                "PhilHealth Philippines health insurance coverage annual report",
            ],
            preferred_file_patterns=["*philhealth*", "*annual*report*", "*claims*", "*coverage*"],
            fallback_urls=["https://www.philhealth.gov.ph/"],
            historical_records=[
                _historical_record(
                    title="PhilHealth Annual Report 2018",
                    url="https://www.philhealth.gov.ph/about_us/statsncharts/",
                    year=2018,
                    document_type="annual_report_pdf",
                ),
                _historical_record(
                    title="PhilHealth Annual Report 2020",
                    url="https://www.philhealth.gov.ph/about_us/statsncharts/",
                    year=2020,
                    document_type="annual_report_pdf",
                ),
                _historical_record(
                    title="PhilHealth Annual Report 2023",
                    url="https://www.philhealth.gov.ph/about_us/statsncharts/",
                    year=2023,
                    document_type="annual_report_pdf",
                ),
            ],
            notes=[
                "Useful for insured-access context and financing coverage, not as a direct HIV cascade truth source.",
            ],
        ),
        StructuredSourceAdapterSpec(
            adapter_id="doh_facility_stats",
            source_name="DOH Facility Statistics and Master Lists",
            organization="Department of Health Philippines",
            source_tier="tier1_official_anchor",
            access_mode="public_report_excel_request",
            spatial_resolution="facility_province_annual",
            temporal_resolution="annual",
            landing_url="https://doh.gov.ph/",
            determinant_silos=["health_system_reach", "linkage_to_care", "transport_friction", "policy_implementation_weakness"],
            promotion_track="main_predictive_candidate",
            platform="doh_facility_stats",
            seed_queries=[
                "DOH Philippines facility master list HIV treatment hub laboratory",
                "DOH Philippines clinic treatment site physician facility statistics",
            ],
            preferred_file_patterns=["*facility*", "*master*list*", "*treatment*hub*", "*laboratory*"],
            fallback_urls=["https://doh.gov.ph/"],
            historical_records=[
                _historical_record(
                    title="DOH Health Facility Master List 2019",
                    url="https://doh.gov.ph/",
                    year=2019,
                    document_type="facility_master_list",
                ),
                _historical_record(
                    title="DOH HIV Treatment Hub and Laboratory List 2022",
                    url="https://doh.gov.ph/",
                    year=2022,
                    document_type="facility_master_list",
                ),
                _historical_record(
                    title="DOH Health Facility Statistics and Licensing Tables 2024",
                    url="https://doh.gov.ph/",
                    year=2024,
                    document_type="facility_master_list",
                ),
            ],
            notes=[
                "Prefer direct facility inventories, treatment hubs, and laboratory lists where available.",
                "This is the strongest source family in the non-HARP infrastructure layer and should be treated as promotion-eligible when numeric and geographically resolvable.",
            ],
        ),
    ]


def get_structured_source_adapters(plugin_id: str) -> list[StructuredSourceAdapterSpec]:
    if plugin_id == "hiv":
        return _hiv_structured_source_adapters()
    return []
