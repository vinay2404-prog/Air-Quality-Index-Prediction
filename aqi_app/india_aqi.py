"""CPCB (Indian) AQI calculation from pollutant concentrations.

Open-Meteo publishes US and European AQI values, but not the Indian CPCB AQI.
This module reconstructs the CPCB AQI from the published pollutant
concentrations using the official sub-index breakpoints.

The CPCB AQI is the maximum of the pollutant sub-indices.  A valid AQI needs
at least three pollutants; with fewer, no AQI is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constants import aqi_category

# CPCB sub-index breakpoints.
#
# Concentration breakpoints are inclusive upper bounds and AQI breakpoints are
# the matching sub-index values.  Values between two concentration breakpoints
# are linearly interpolated; values above the last breakpoint are capped at 500.
CPCB_BREAKPOINTS: dict[str, dict[str, list[float]]] = {
    "PM2.5": {
        "concentration": [0, 30, 60, 90, 120, 250, 500],
        "aqi": [0, 50, 100, 200, 300, 400, 500],
    },
    "PM10": {
        "concentration": [0, 50, 100, 250, 350, 430, 600],
        "aqi": [0, 50, 100, 200, 300, 400, 500],
    },
    "NO2": {
        "concentration": [0, 40, 80, 180, 280, 400, 600],
        "aqi": [0, 50, 100, 200, 300, 400, 500],
    },
    "SO2": {
        "concentration": [0, 40, 80, 380, 800, 1600, 2000],
        "aqi": [0, 50, 100, 200, 300, 400, 500],
    },
    "CO": {
        "concentration": [0, 1, 2, 10, 17, 34, 50],
        "aqi": [0, 50, 100, 200, 300, 400, 500],
    },
    "O3": {
        "concentration": [0, 50, 100, 168, 208, 748, 1000],
        "aqi": [0, 50, 100, 200, 300, 400, 500],
    },
    "NH3": {
        "concentration": [0, 200, 400, 800, 1200, 1800, 2500],
        "aqi": [0, 50, 100, 200, 300, 400, 500],
    },
    # Pb is part of the CPCB AQI but is not published by Open-Meteo.  The
    # breakpoints are included so callers that do have a Pb measurement can
    # use the same calculation.
    "Pb": {
        "concentration": [0, 0.5, 1, 2, 3, 4, 5],
        "aqi": [0, 50, 100, 200, 300, 400, 500],
    },
}

CPCB_POLLUTANTS = list(CPCB_BREAKPOINTS)
CPCB_MIN_POLLUTANTS = 3
CPCB_MAX_AQI = 500


@dataclass(frozen=True)
class IndiaAQIResult:
    """Result of a CPCB AQI calculation."""

    aqi: float | None
    category: str | None
    dominant_pollutant: str | None
    sub_indices: dict[str, float] = field(default_factory=dict)
    pollutants_used: list[str] = field(default_factory=list)
    missing_pollutants: list[str] = field(default_factory=list)
    message: str | None = None

    @property
    def valid(self) -> bool:
        return self.aqi is not None


def sub_index(pollutant: str, concentration: float) -> float | None:
    """Return the CPCB sub-index for one pollutant concentration.

    Returns ``None`` for missing, non-finite, or negative values.  Values above
    the highest breakpoint are capped at 500.
    """
    if pollutant not in CPCB_BREAKPOINTS:
        return None
    if concentration is None:
        return None
    try:
        value = float(concentration)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None

    table = CPCB_BREAKPOINTS[pollutant]
    concentrations = table["concentration"]
    aqi_values = table["aqi"]

    if value >= concentrations[-1]:
        return float(CPCB_MAX_AQI)

    for index in range(1, len(concentrations)):
        lower_c = concentrations[index - 1]
        upper_c = concentrations[index]
        if value <= upper_c:
            lower_i = aqi_values[index - 1]
            upper_i = aqi_values[index]
            if upper_c == lower_c:
                return float(upper_i)
            fraction = (value - lower_c) / (upper_c - lower_c)
            return float(lower_i + fraction * (upper_i - lower_i))

    return float(CPCB_MAX_AQI)


def calculate_indian_aqi(values: dict[str, float | None]) -> IndiaAQIResult:
    """Compute the CPCB AQI from a pollutant -> concentration mapping.

    ``CO`` must be supplied in mg/m³; all other pollutants are in µg/m³.
    """
    sub_indices: dict[str, float] = {}
    for pollutant in CPCB_POLLUTANTS:
        value = values.get(pollutant)
        index = sub_index(pollutant, value)
        if index is not None:
            sub_indices[pollutant] = index

    pollutants_used = [p for p in CPCB_POLLUTANTS if p in sub_indices]
    missing = [p for p in CPCB_POLLUTANTS if p not in sub_indices]

    if len(sub_indices) < CPCB_MIN_POLLUTANTS:
        return IndiaAQIResult(
            aqi=None,
            category=None,
            dominant_pollutant=None,
            sub_indices=sub_indices,
            pollutants_used=pollutants_used,
            missing_pollutants=missing,
            message=(
                f"CPCB AQI needs at least {CPCB_MIN_POLLUTANTS} pollutants; "
                f"only {len(sub_indices)} were available."
            ),
        )

    dominant = max(sub_indices, key=sub_indices.get)
    aqi = min(max(round(sub_indices[dominant]), 0), CPCB_MAX_AQI)

    return IndiaAQIResult(
        aqi=float(aqi),
        category=aqi_category(aqi),
        dominant_pollutant=dominant,
        sub_indices=sub_indices,
        pollutants_used=pollutants_used,
        missing_pollutants=missing,
    )
