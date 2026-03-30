from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

NATIONAL_GEO_ALIASES = {
    "national",
    "philippines",
    "republic of the philippines",
}


REGION_DEFINITIONS = {
    "ncr": {
        "display": "National Capital Region",
        "macro_region": "luzon",
        "aliases": ["ncr", "national capital region", "metro manila"],
    },
    "car": {
        "display": "Cordillera Administrative Region",
        "macro_region": "luzon",
        "aliases": ["car", "cordillera administrative region", "cordillera"],
    },
    "region_i": {
        "display": "Ilocos Region",
        "macro_region": "luzon",
        "aliases": ["region i", "ilocos region", "ilocos"],
    },
    "region_ii": {
        "display": "Cagayan Valley",
        "macro_region": "luzon",
        "aliases": ["region ii", "cagayan valley"],
    },
    "region_iii": {
        "display": "Central Luzon",
        "macro_region": "luzon",
        "aliases": ["region iii", "central luzon"],
    },
    "region_iv_a": {
        "display": "CALABARZON",
        "macro_region": "luzon",
        "aliases": ["region iv-a", "iv-a", "calabarzon"],
    },
    "region_iv_b": {
        "display": "MIMAROPA",
        "macro_region": "luzon",
        "aliases": ["region iv-b", "iv-b", "mimaropa"],
    },
    "region_v": {
        "display": "Bicol Region",
        "macro_region": "luzon",
        "aliases": ["region v", "bicol region", "bicol"],
    },
    "region_vi": {
        "display": "Western Visayas",
        "macro_region": "visayas",
        "aliases": ["region vi", "western visayas"],
    },
    "region_vii": {
        "display": "Central Visayas",
        "macro_region": "visayas",
        "aliases": ["region vii", "central visayas"],
    },
    "region_viii": {
        "display": "Eastern Visayas",
        "macro_region": "visayas",
        "aliases": ["region viii", "eastern visayas"],
    },
    "region_ix": {
        "display": "Zamboanga Peninsula",
        "macro_region": "mindanao",
        "aliases": ["region ix", "zamboanga peninsula"],
    },
    "region_x": {
        "display": "Northern Mindanao",
        "macro_region": "mindanao",
        "aliases": ["region x", "northern mindanao"],
    },
    "region_xi": {
        "display": "Davao Region",
        "macro_region": "mindanao",
        "aliases": ["region xi", "davao region", "davao"],
    },
    "region_xii": {
        "display": "SOCCSKSARGEN",
        "macro_region": "mindanao",
        "aliases": ["region xii", "soccsksargen"],
    },
    "region_xiii": {
        "display": "Caraga",
        "macro_region": "mindanao",
        "aliases": ["region xiii", "caraga"],
    },
    "barmm": {
        "display": "BARMM",
        "macro_region": "mindanao",
        "aliases": ["barmm", "bangsamoro", "bangsamoro autonomous region in muslim mindanao"],
    },
}

PROVINCE_DEFINITIONS = {
    "Abra": {"region": "car", "aliases": ["abra"]},
    "Apayao": {"region": "car", "aliases": ["apayao"]},
    "Benguet": {"region": "car", "aliases": ["benguet"]},
    "Baguio": {"region": "car", "aliases": ["baguio", "baguio city"]},
    "Ifugao": {"region": "car", "aliases": ["ifugao"]},
    "Kalinga": {"region": "car", "aliases": ["kalinga"]},
    "Mountain Province": {"region": "car", "aliases": ["mountain province"]},
    "Ilocos Norte": {"region": "region_i", "aliases": ["ilocos norte"]},
    "Ilocos Sur": {"region": "region_i", "aliases": ["ilocos sur"]},
    "La Union": {"region": "region_i", "aliases": ["la union"]},
    "Pangasinan": {"region": "region_i", "aliases": ["pangasinan"]},
    "Batanes": {"region": "region_ii", "aliases": ["batanes"]},
    "Cagayan": {"region": "region_ii", "aliases": ["cagayan"]},
    "Isabela": {"region": "region_ii", "aliases": ["isabela"]},
    "Nueva Vizcaya": {"region": "region_ii", "aliases": ["nueva vizcaya"]},
    "Quirino": {"region": "region_ii", "aliases": ["quirino"]},
    "Aurora": {"region": "region_iii", "aliases": ["aurora"]},
    "Bataan": {"region": "region_iii", "aliases": ["bataan"]},
    "Bulacan": {"region": "region_iii", "aliases": ["bulacan"]},
    "Nueva Ecija": {"region": "region_iii", "aliases": ["nueva ecija"]},
    "Pampanga": {"region": "region_iii", "aliases": ["pampanga"]},
    "Tarlac": {"region": "region_iii", "aliases": ["tarlac"]},
    "Zambales": {"region": "region_iii", "aliases": ["zambales"]},
    "Olongapo": {"region": "region_iii", "aliases": ["olongapo", "olongapo city"]},
    "Metro Manila": {
        "region": "ncr",
        "aliases": [
            "metro manila",
            "manila",
            "city of manila",
            "quezon city",
            "caloocan",
            "las pinas",
            "makati",
            "malabon",
            "mandaluyong",
            "marikina",
            "muntinlupa",
            "navotas",
            "paranaque",
            "pasay",
            "pasig",
            "pateros",
            "san juan",
            "taguig",
            "valenzuela",
        ],
    },
    "Batangas": {"region": "region_iv_a", "aliases": ["batangas"]},
    "Cavite": {"region": "region_iv_a", "aliases": ["cavite"]},
    "Laguna": {"region": "region_iv_a", "aliases": ["laguna"]},
    "Quezon": {"region": "region_iv_a", "aliases": ["quezon province", "province of quezon", "quezon"]},
    "Rizal": {"region": "region_iv_a", "aliases": ["rizal"]},
    "Lucena": {"region": "region_iv_a", "aliases": ["lucena", "lucena city"]},
    "Marinduque": {"region": "region_iv_b", "aliases": ["marinduque"]},
    "Occidental Mindoro": {"region": "region_iv_b", "aliases": ["occidental mindoro"]},
    "Oriental Mindoro": {"region": "region_iv_b", "aliases": ["oriental mindoro"]},
    "Palawan": {"region": "region_iv_b", "aliases": ["palawan"]},
    "Puerto Princesa": {"region": "region_iv_b", "aliases": ["puerto princesa", "puerto princesa city"]},
    "Romblon": {"region": "region_iv_b", "aliases": ["romblon"]},
    "Albay": {"region": "region_v", "aliases": ["albay"]},
    "Camarines Norte": {"region": "region_v", "aliases": ["camarines norte"]},
    "Camarines Sur": {"region": "region_v", "aliases": ["camarines sur"]},
    "Catanduanes": {"region": "region_v", "aliases": ["catanduanes"]},
    "Masbate": {"region": "region_v", "aliases": ["masbate"]},
    "Sorsogon": {"region": "region_v", "aliases": ["sorsogon"]},
    "Aklan": {"region": "region_vi", "aliases": ["aklan"]},
    "Antique": {"region": "region_vi", "aliases": ["antique"]},
    "Capiz": {"region": "region_vi", "aliases": ["capiz"]},
    "Guimaras": {"region": "region_vi", "aliases": ["guimaras"]},
    "Iloilo": {"region": "region_vi", "aliases": ["iloilo"]},
    "Iloilo City": {"region": "region_vi", "aliases": ["iloilo city"]},
    "Negros Occidental": {"region": "region_vi", "aliases": ["negros occidental"]},
    "Bacolod": {"region": "region_vi", "aliases": ["bacolod", "bacolod city"]},
    "Bohol": {"region": "region_vii", "aliases": ["bohol"]},
    "Cebu": {"region": "region_vii", "aliases": ["cebu"]},
    "Cebu City": {"region": "region_vii", "aliases": ["cebu city"]},
    "Lapu-Lapu": {"region": "region_vii", "aliases": ["lapu-lapu", "lapu lapu", "lapu-lapu city"]},
    "Mandaue": {"region": "region_vii", "aliases": ["mandaue", "mandaue city"]},
    "Negros Oriental": {"region": "region_vii", "aliases": ["negros oriental"]},
    "Siquijor": {"region": "region_vii", "aliases": ["siquijor"]},
    "Biliran": {"region": "region_viii", "aliases": ["biliran"]},
    "Eastern Samar": {"region": "region_viii", "aliases": ["eastern samar"]},
    "Leyte": {"region": "region_viii", "aliases": ["leyte"]},
    "Southern Leyte": {"region": "region_viii", "aliases": ["southern leyte"]},
    "Northern Samar": {"region": "region_viii", "aliases": ["northern samar"]},
    "Samar": {"region": "region_viii", "aliases": ["samar", "western samar"]},
    "Tacloban": {"region": "region_viii", "aliases": ["tacloban", "tacloban city"]},
    "Zamboanga del Norte": {"region": "region_ix", "aliases": ["zamboanga del norte"]},
    "Zamboanga del Sur": {"region": "region_ix", "aliases": ["zamboanga del sur"]},
    "Zamboanga Sibugay": {"region": "region_ix", "aliases": ["zamboanga sibugay"]},
    "Zamboanga City": {"region": "region_ix", "aliases": ["zamboanga city"]},
    "Bukidnon": {"region": "region_x", "aliases": ["bukidnon"]},
    "Camiguin": {"region": "region_x", "aliases": ["camiguin"]},
    "Lanao del Norte": {"region": "region_x", "aliases": ["lanao del norte"]},
    "Misamis Occidental": {"region": "region_x", "aliases": ["misamis occidental"]},
    "Misamis Oriental": {"region": "region_x", "aliases": ["misamis oriental"]},
    "Cagayan de Oro": {"region": "region_x", "aliases": ["cagayan de oro", "cdo"]},
    "Iligan": {"region": "region_x", "aliases": ["iligan", "iligan city"]},
    "Davao de Oro": {"region": "region_xi", "aliases": ["davao de oro", "compostela valley"]},
    "Davao del Norte": {"region": "region_xi", "aliases": ["davao del norte"]},
    "Davao del Sur": {"region": "region_xi", "aliases": ["davao del sur"]},
    "Davao Oriental": {"region": "region_xi", "aliases": ["davao oriental"]},
    "Davao Occidental": {"region": "region_xi", "aliases": ["davao occidental"]},
    "Davao City": {"region": "region_xi", "aliases": ["davao city"]},
    "Cotabato": {"region": "region_xii", "aliases": ["cotabato province", "north cotabato", "cotabato"]},
    "Sarangani": {"region": "region_xii", "aliases": ["sarangani"]},
    "South Cotabato": {"region": "region_xii", "aliases": ["south cotabato"]},
    "Sultan Kudarat": {"region": "region_xii", "aliases": ["sultan kudarat"]},
    "General Santos": {"region": "region_xii", "aliases": ["general santos", "gensan", "general santos city"]},
    "Agusan del Norte": {"region": "region_xiii", "aliases": ["agusan del norte"]},
    "Agusan del Sur": {"region": "region_xiii", "aliases": ["agusan del sur"]},
    "Dinagat Islands": {"region": "region_xiii", "aliases": ["dinagat islands"]},
    "Surigao del Norte": {"region": "region_xiii", "aliases": ["surigao del norte"]},
    "Surigao del Sur": {"region": "region_xiii", "aliases": ["surigao del sur"]},
    "Butuan": {"region": "region_xiii", "aliases": ["butuan", "butuan city"]},
    "Basilan": {"region": "barmm", "aliases": ["basilan"]},
    "Lanao del Sur": {"region": "barmm", "aliases": ["lanao del sur"]},
    "Maguindanao del Norte": {"region": "barmm", "aliases": ["maguindanao del norte"]},
    "Maguindanao del Sur": {"region": "barmm", "aliases": ["maguindanao del sur"]},
    "Sulu": {"region": "barmm", "aliases": ["sulu"]},
    "Tawi-Tawi": {"region": "barmm", "aliases": ["tawi-tawi", "tawi tawi"]},
    "Cotabato City": {"region": "barmm", "aliases": ["cotabato city"]},
    "Marawi": {"region": "barmm", "aliases": ["marawi", "marawi city"]},
}


@dataclass(frozen=True)
class PhilippinesGeoMatch:
    geo: str
    region: str
    region_display: str
    province: str
    macro_region: str
    resolution: str
    mentions: list[str]


def _pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", flags=re.I)


PROVINCE_PATTERNS = [
    (province, province_info["region"], alias, _pattern(alias))
    for province, province_info in PROVINCE_DEFINITIONS.items()
    for alias in province_info["aliases"]
]
REGION_PATTERNS = [
    (region_code, region_info["display"], alias, _pattern(alias))
    for region_code, region_info in REGION_DEFINITIONS.items()
    for alias in region_info["aliases"]
]

REGION_DISPLAY_NAMES = {info["display"] for info in REGION_DEFINITIONS.values()}
REGION_LIKE_CANONICALS = REGION_DISPLAY_NAMES | {"Metro Manila"}
CITY_LIKE_CANONICALS = {
    "Bacolod",
    "Baguio",
    "Butuan",
    "Cagayan de Oro",
    "Cebu City",
    "Cotabato City",
    "Davao City",
    "General Santos",
    "Iligan",
    "Iloilo City",
    "Lapu-Lapu",
    "Lucena",
    "Mandaue",
    "Marawi",
    "Olongapo",
    "Puerto Princesa",
    "Tacloban",
    "Zamboanga City",
}


def infer_philippines_geo(text: str, *, default_country_focus: bool = False) -> PhilippinesGeoMatch:
    lowered = f" {(text or '').strip().lower()} "
    province_hits: dict[str, dict[str, Any]] = {}
    for province, region_code, alias, pattern in PROVINCE_PATTERNS:
        matches = list(pattern.finditer(lowered))
        if not matches:
            continue
        province_hits[province] = {
            "count": len(matches),
            "earliest": matches[0].start(),
            "alias_length": len(alias),
            "region": region_code,
            "mentions": [match.group(0).strip() for match in matches],
        }
    if province_hits:
        province = sorted(
            province_hits.items(),
            key=lambda item: (item[1]["count"], item[1]["alias_length"], -item[1]["earliest"]),
            reverse=True,
        )[0][0]
        region_code = province_hits[province]["region"]
        region_info = REGION_DEFINITIONS[region_code]
        return PhilippinesGeoMatch(
            geo=province,
            region=region_code,
            region_display=region_info["display"],
            province=province,
            macro_region=region_info["macro_region"],
            resolution="province",
            mentions=province_hits[province]["mentions"],
        )

    region_hits: dict[str, dict[str, Any]] = {}
    for region_code, region_display, alias, pattern in REGION_PATTERNS:
        matches = list(pattern.finditer(lowered))
        if not matches:
            continue
        region_hits[region_code] = {
            "count": len(matches),
            "earliest": matches[0].start(),
            "alias_length": len(alias),
            "mentions": [match.group(0).strip() for match in matches],
            "display": region_display,
        }
    if region_hits:
        region_code = sorted(
            region_hits.items(),
            key=lambda item: (item[1]["count"], item[1]["alias_length"], -item[1]["earliest"]),
            reverse=True,
        )[0][0]
        region_info = REGION_DEFINITIONS[region_code]
        return PhilippinesGeoMatch(
            geo=region_info["display"],
            region=region_code,
            region_display=region_info["display"],
            province="",
            macro_region=region_info["macro_region"],
            resolution="region",
            mentions=region_hits[region_code]["mentions"],
        )

    if " philippines " in lowered or default_country_focus:
        return PhilippinesGeoMatch(
            geo="Philippines",
            region="national",
            region_display="Philippines",
            province="",
            macro_region="national",
            resolution="national",
            mentions=["philippines"] if " philippines " in lowered else [],
        )
    return PhilippinesGeoMatch(
        geo="",
        region="",
        region_display="",
        province="",
        macro_region="",
        resolution="unknown",
        mentions=[],
    )


def is_national_geo(name: str) -> bool:
    return (name or "").strip().lower() in NATIONAL_GEO_ALIASES


def normalize_geo_label(geo: str, *, default_country_focus: bool = False) -> str:
    value = (geo or "").strip()
    lowered = value.lower()
    if not value and default_country_focus:
        return "Philippines"
    if lowered in NATIONAL_GEO_ALIASES:
        return "Philippines"
    if lowered in {"global", "international"}:
        return "global"
    match = infer_philippines_geo(value, default_country_focus=default_country_focus)
    if match.resolution == "province" and match.province:
        return match.province
    if match.resolution == "region" and match.region_display:
        return match.region_display
    if match.resolution == "national":
        return "Philippines"
    return value


def geo_resolution_label(geo: str) -> str:
    lowered = (geo or "").strip().lower()
    if not lowered:
        return "unknown"
    if lowered in {"philippines", "national"}:
        return "national"
    if lowered in {"global", "international"}:
        return "global"
    match = infer_philippines_geo(geo, default_country_focus=lowered in NATIONAL_GEO_ALIASES)
    if match.resolution == "region" or (geo or "").strip() in REGION_LIKE_CANONICALS:
        return "region"
    canonical = match.province or normalize_geo_label(geo, default_country_focus=lowered in NATIONAL_GEO_ALIASES)
    if canonical in CITY_LIKE_CANONICALS or canonical.endswith(" City"):
        return "city"
    if canonical in PROVINCE_DEFINITIONS:
        return "province"
    return "unknown"


def infer_region_code(geo: str, text: str = "") -> str:
    match = infer_philippines_geo(f"{geo} {text}", default_country_focus=(geo or "").strip().lower() in {"philippines", "national"})
    return match.region


def macro_region_label(name: str) -> str:
    match = infer_philippines_geo(name)
    if match.macro_region:
        return match.macro_region
    lowered = (name or "").strip().lower()
    if lowered in {"national", "philippines"}:
        return "national"
    return "mixed"


def philippines_modeling_geos(*, include_national: bool = True) -> list[str]:
    geos = sorted(PROVINCE_DEFINITIONS.keys())
    if include_national:
        return ["Philippines", *geos]
    return geos
