"""Domain constants shared by the app, training script, and tests."""

from __future__ import annotations

import math
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Pollutants and units
# ---------------------------------------------------------------------------
# The Kaggle "Air Quality Data in India" dataset reports every pollutant in
# µg/m³ except CO, which is reported in mg/m³.  The previous version of this
# project incorrectly labelled NOx as ppb; that has been corrected here.
POLLUTANTS: dict[str, dict[str, object]] = {
    "PM2.5": {"unit": "µg/m³", "default": 67.45, "label": "PM2.5"},
    "PM10": {"unit": "µg/m³", "default": 118.13, "label": "PM10"},
    "NO": {"unit": "µg/m³", "default": 17.57, "label": "NO"},
    "NO2": {"unit": "µg/m³", "default": 28.56, "label": "NO2"},
    "NOx": {"unit": "µg/m³", "default": 32.31, "label": "NOx"},
    "NH3": {"unit": "µg/m³", "default": 23.48, "label": "NH3"},
    "CO": {"unit": "mg/m³", "default": 2.25, "label": "CO"},
    "SO2": {"unit": "µg/m³", "default": 14.53, "label": "SO2"},
    "O3": {"unit": "µg/m³", "default": 34.49, "label": "O3"},
    "Benzene": {"unit": "µg/m³", "default": 3.28, "label": "Benzene"},
    "Toluene": {"unit": "µg/m³", "default": 8.70, "label": "Toluene"},
    "Xylene": {"unit": "µg/m³", "default": 3.07, "label": "Xylene"},
}

POLLUTANT_ORDER: list[str] = list(POLLUTANTS)
MAIN_POLLUTANTS: list[str] = ["PM2.5", "PM10"]
OPTIONAL_POLLUTANTS: list[str] = [
    name for name in POLLUTANT_ORDER if name not in MAIN_POLLUTANTS
]

UNIT_GROUPS: dict[str, list[str]] = {
    "µg/m³": [p for p in POLLUTANT_ORDER if POLLUTANTS[p]["unit"] == "µg/m³"],
    "mg/m³": [p for p in POLLUTANT_ORDER if POLLUTANTS[p]["unit"] == "mg/m³"],
}

CATEGORICAL_FEATURE = "City"
TARGET_COLUMN = "AQI"
FEATURE_COLUMNS: list[str] = [CATEGORICAL_FEATURE, *POLLUTANT_ORDER]

# ---------------------------------------------------------------------------
# Indian CPCB AQI bands (used by the ML prediction page)
# ---------------------------------------------------------------------------
AQI_BANDS: list[dict[str, object]] = [
    {"label": "Good", "min": 0, "max": 50, "message": "Air quality is in the Good range. Pollution-related risk is comparatively low for most people."},
    {"label": "Satisfactory", "min": 51, "max": 100, "message": "Air quality is Satisfactory. Some sensitive individuals may experience discomfort."},
    {"label": "Moderate", "min": 101, "max": 200, "message": "Air quality is Moderate. Sensitive individuals should consider reducing strenuous outdoor activity if symptoms occur."},
    {"label": "Poor", "min": 201, "max": 300, "message": "Air quality is Poor. Consider reducing prolonged or strenuous outdoor activity."},
    {"label": "Very Poor", "min": 301, "max": 400, "message": "Air quality is Very Poor. Reduce outdoor exposure and follow local public-health advice."},
    {"label": "Severe", "min": 401, "max": math.inf, "message": "Air quality is Severe. Minimize outdoor pollution exposure and follow official public-health advisories."},
]

AQI_CATEGORY_ORDER = [band["label"] for band in AQI_BANDS]
AQI_MAX_DISPLAY = 500


def aqi_band(value: float) -> dict[str, object]:
    """Return the CPCB band for ``value`` (negative values clamp to Good)."""
    value = max(0.0, float(value))
    for band in AQI_BANDS:
        if band["min"] <= value <= band["max"]:
            return band
    return AQI_BANDS[-1]


def aqi_category(value: float) -> str:
    return str(aqi_band(value)["label"])


def aqi_health_message(value: float) -> str:
    return str(aqi_band(value)["message"])


def aqi_progress(value: float) -> float:
    """Fraction of the 0-500 CPCB display scale, clamped to [0, 1]."""
    return min(max(float(value), 0.0) / AQI_MAX_DISPLAY, 1.0)


# ---------------------------------------------------------------------------
# US EPA AQI bands (used by the live Open-Meteo feed, which exposes US AQI)
# ---------------------------------------------------------------------------
US_AQI_BANDS: list[dict[str, object]] = [
    {"label": "Good", "min": 0, "max": 50},
    {"label": "Moderate", "min": 51, "max": 100},
    {"label": "Unhealthy for Sensitive Groups", "min": 101, "max": 150},
    {"label": "Unhealthy", "min": 151, "max": 200},
    {"label": "Very Unhealthy", "min": 201, "max": 300},
    {"label": "Hazardous", "min": 301, "max": math.inf},
]


def us_aqi_category(value: float) -> str:
    value = max(0.0, float(value))
    for band in US_AQI_BANDS:
        if band["min"] <= value <= band["max"]:
            return str(band["label"])
    return str(US_AQI_BANDS[-1]["label"])


# ---------------------------------------------------------------------------
# City coordinates for the dataset's 26 cities (used by the dashboard map)
# ---------------------------------------------------------------------------
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "Ahmedabad": (23.0225, 72.5714),
    "Aizawl": (23.7271, 92.7176),
    "Amaravati": (16.5131, 80.5165),
    "Amritsar": (31.6340, 74.8723),
    "Bengaluru": (12.9716, 77.5946),
    "Bhopal": (23.2599, 77.4126),
    "Brajrajnagar": (21.8167, 83.9167),
    "Chandigarh": (30.7333, 76.7794),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Delhi": (28.7041, 77.1025),
    "Ernakulam": (9.9816, 76.2999),
    "Gurugram": (28.4595, 77.0266),
    "Guwahati": (26.1445, 91.7362),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
    "Jorapokhar": (23.7167, 86.4000),
    "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639),
    "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777),
    "Patna": (25.5941, 85.1376),
    "Shillong": (25.5788, 91.8933),
    "Talcher": (20.9500, 85.2333),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Visakhapatnam": (17.6868, 83.2185),
}


def nearest_supported_city(latitude: float, longitude: float) -> str:
    """Return the dataset city closest to a coordinate (great-circle distance)."""

    def distance_sq(city: str) -> float:
        lat2, lon2 = CITY_COORDINATES[city]
        # Equirectangular approximation is more than accurate enough here.
        x = math.radians(lon2 - longitude) * math.cos(
            math.radians((latitude + lat2) / 2)
        )
        y = math.radians(lat2 - latitude)
        return x * x + y * y

    return min(CITY_COORDINATES, key=distance_sq)


def pollutant_label(pollutant: str) -> str:
    return f"{POLLUTANTS[pollutant]['label']} ({POLLUTANTS[pollutant]['unit']})"


def format_number(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:,.{digits}f}"


def missing_pollutants(values: dict[str, float | None]) -> list[str]:
    return [
        name
        for name in POLLUTANT_ORDER
        if values.get(name) is None or _is_nan(values.get(name))
    ]


def _is_nan(value: object) -> bool:
    try:
        return math.isnan(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def first_non_empty(values: Iterable[object]) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None
