"""Live air-quality data from the Open-Meteo APIs (no API key required)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from .constants import POLLUTANT_ORDER, aqi_category
from .india_aqi import calculate_indian_aqi

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

CURRENT_VARIABLES = [
    "us_aqi",
    "european_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "ammonia",
]
HOURLY_VARIABLES = [
    "us_aqi",
    "european_aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "ammonia",
]

# Open-Meteo returns these in µg/m³.  The app stores CO in mg/m³.
OPEN_METEO_TO_APP = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "nitrogen_dioxide": "NO2",
    "carbon_monoxide": "CO",
    "sulphur_dioxide": "SO2",
    "ozone": "O3",
    "ammonia": "NH3",
}

ATTRIBUTION = "Live data: Open-Meteo Air Quality API (open-meteo.com), CC BY 4.0."


class LiveDataError(RuntimeError):
    """Raised when live data cannot be retrieved or parsed."""


@dataclass(frozen=True)
class Location:
    name: str
    country: str
    admin1: str | None
    latitude: float
    longitude: float
    timezone: str | None = None

    @property
    def display_name(self) -> str:
        parts = [self.name]
        if self.admin1 and self.admin1 != self.name:
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)


def _request_json(
    url: str,
    params: dict[str, Any],
    *,
    error_label: str,
    timeout: float,
    attempts: int = 3,
    base_delay: float = 0.5,
) -> dict[str, Any]:
    """GET JSON with a small exponential-backoff retry.

    Open-Meteo occasionally rate-limits or drops a request; retrying keeps the
    live page usable without hiding a persistent failure.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except ValueError as exc:
            # ``response.json()`` raises a ValueError subclass for bad JSON.
            raise LiveDataError(f"{error_label} returned invalid JSON") from exc
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(base_delay * (2 ** (attempt - 1)))
    raise LiveDataError(
        f"{error_label} failed after {attempts} attempts: {last_error}"
    )


def geocode_city(query: str, count: int = 6, timeout: float = 15.0) -> list[Location]:
    """Look up a city name using the Open-Meteo geocoding API."""
    query = query.strip()
    if not query:
        return []

    payload = _request_json(
        GEOCODING_URL,
        {"name": query, "count": count, "language": "en", "format": "json"},
        error_label="City lookup",
        timeout=timeout,
    )

    results = payload.get("results") or []
    locations: list[Location] = []
    for result in results:
        try:
            locations.append(
                Location(
                    name=str(result.get("name", "Unknown")),
                    country=str(result.get("country", "")),
                    admin1=result.get("admin1"),
                    latitude=float(result["latitude"]),
                    longitude=float(result["longitude"]),
                    timezone=result.get("timezone"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return locations


def fetch_air_quality(
    latitude: float,
    longitude: float,
    forecast_days: int = 5,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Fetch current conditions plus an hourly AQI forecast."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(CURRENT_VARIABLES),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "auto",
        "forecast_days": max(1, min(int(forecast_days), 7)),
    }
    payload = _request_json(
        AIR_QUALITY_URL,
        params,
        error_label="Live air-quality request",
        timeout=timeout,
    )

    if "current" not in payload or "hourly" not in payload:
        raise LiveDataError("Live air-quality response is missing expected fields")
    return payload


def current_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise the API's ``current`` block for display and model input."""
    current = payload.get("current") or {}
    snapshot: dict[str, Any] = {
        "time": current.get("time"),
        "us_aqi": _as_float(current.get("us_aqi")),
        "european_aqi": _as_float(current.get("european_aqi")),
        "pollutants": {},
    }

    for source, target in OPEN_METEO_TO_APP.items():
        value = _as_float(current.get(source))
        if value is not None and target == "CO":
            value = value / 1000.0
        snapshot["pollutants"][target] = value

    snapshot["indian_aqi"] = calculate_indian_aqi(snapshot["pollutants"])
    snapshot["timezone"] = payload.get("timezone")
    snapshot["timezone_abbreviation"] = payload.get("timezone_abbreviation")
    snapshot["latitude"] = _as_float(payload.get("latitude"))
    snapshot["longitude"] = _as_float(payload.get("longitude"))
    return snapshot


def hourly_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Return an hourly forecast frame with a parsed timestamp index."""
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        raise LiveDataError("Hourly forecast is empty")

    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).set_index("time").sort_index()

    for column in HOURLY_VARIABLES:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def hourly_indian_aqi(frame: pd.DataFrame) -> pd.Series:
    """Compute the CPCB AQI for every row of an hourly pollutant frame."""
    values: list[float | None] = []
    for _, row in frame.iterrows():
        pollutants: dict[str, float | None] = {}
        for source, target in OPEN_METEO_TO_APP.items():
            value = row.get(source)
            if value is None or pd.isna(value):
                pollutants[target] = None
            else:
                numeric = float(value)
                pollutants[target] = numeric / 1000.0 if target == "CO" else numeric
        values.append(calculate_indian_aqi(pollutants).aqi)
    return pd.Series(values, index=frame.index, name="india_aqi", dtype="float64")


def daily_forecast(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly CPCB AQI into a simple daily outlook."""
    if hourly.empty:
        return pd.DataFrame()

    if "india_aqi" not in hourly.columns:
        if any(source in hourly.columns for source in OPEN_METEO_TO_APP):
            hourly = hourly.assign(india_aqi=hourly_indian_aqi(hourly))
        elif "us_aqi" in hourly.columns:
            # Backwards-compatible fallback for callers that only have US AQI.
            hourly = hourly.assign(india_aqi=hourly["us_aqi"])
        else:
            hourly = hourly.assign(india_aqi=float("nan"))

    grouped = hourly.resample("D")
    peak = grouped["india_aqi"].max()

    daily = pd.DataFrame(
        {
            "Date": peak.index,
            "Peak Indian AQI": peak.to_numpy(),
            "Mean PM2.5": grouped["pm2_5"].mean().to_numpy()
            if "pm2_5" in hourly.columns
            else float("nan"),
            "Mean PM10": grouped["pm10"].mean().to_numpy()
            if "pm10" in hourly.columns
            else float("nan"),
        }
    )
    daily["Category"] = daily["Peak Indian AQI"].map(
        lambda value: aqi_category(value) if pd.notna(value) else "—"
    )
    return daily.reset_index(drop=True)


def hourly_feature_frame(frame: pd.DataFrame, city: str) -> pd.DataFrame:
    """Build model-input rows for each hourly forecast timestamp.

    The returned frame has the exact feature columns the pipeline expects.
    Pollutants that Open-Meteo does not publish remain ``NaN`` so the model's
    own imputation is used, matching training-time behaviour.
    """
    data: dict[str, Any] = {"City": city}
    for source, target in OPEN_METEO_TO_APP.items():
        if source in frame.columns:
            values = pd.to_numeric(frame[source], errors="coerce")
        else:
            values = pd.Series(float("nan"), index=frame.index, dtype="float64")
        data[target] = values / 1000.0 if target == "CO" else values

    for pollutant in POLLUTANT_ORDER:
        data.setdefault(pollutant, float("nan"))

    return pd.DataFrame(data, index=frame.index)[["City", *POLLUTANT_ORDER]]


def model_features(current: dict[str, Any]) -> dict[str, float | None]:
    """Map live values onto the prediction model's pollutant inputs.

    Open-Meteo does not publish NO, NOx, Benzene, Toluene, or Xylene.  Those
    entries are returned as ``None`` so the model pipeline can impute them.
    """
    pollutants = current.get("pollutants", {})
    return {name: pollutants.get(name) for name in POLLUTANT_ORDER}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if pd.notna(numeric) else None
