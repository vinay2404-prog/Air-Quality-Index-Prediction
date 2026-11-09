import math

import pytest

from aqi_app.india_aqi import (
    CPCB_MAX_AQI,
    CPCB_MIN_POLLUTANTS,
    calculate_indian_aqi,
    sub_index,
)


def test_sub_index_breakpoint_boundaries():
    assert sub_index("PM2.5", 0) == 0
    assert sub_index("PM2.5", 30) == 50
    assert sub_index("PM2.5", 60) == 100
    assert sub_index("PM2.5", 90) == 200
    assert sub_index("PM2.5", 120) == 300
    assert sub_index("PM2.5", 250) == 400
    assert sub_index("PM2.5", 500) == 500


def test_sub_index_linear_interpolation():
    assert sub_index("PM2.5", 45) == pytest.approx(75.0)
    assert sub_index("PM10", 75) == pytest.approx(75.0)
    assert sub_index("CO", 1.5) == pytest.approx(75.0)
    assert sub_index("NO2", 60) == pytest.approx(75.0)


def test_sub_index_caps_above_last_breakpoint():
    assert sub_index("PM2.5", 900) == CPCB_MAX_AQI
    assert sub_index("PM10", 5000) == CPCB_MAX_AQI


def test_lead_breakpoints_are_supported():
    assert sub_index("Pb", 0.25) == pytest.approx(25.0)
    assert sub_index("Pb", 0.75) == pytest.approx(75.0)
    assert sub_index("Pb", 9.0) == CPCB_MAX_AQI


def test_sub_index_rejects_invalid_values():
    assert sub_index("PM2.5", None) is None
    assert sub_index("PM2.5", -1) is None
    assert sub_index("PM2.5", math.inf) is None
    assert sub_index("PM2.5", math.nan) is None
    assert sub_index("Lead", 1.0) is None


def test_calculate_indian_aqi_uses_dominant_pollutant():
    result = calculate_indian_aqi(
        {
            "PM2.5": 102.4,
            "PM10": 192.3,
            "NO2": 67.3,
            "SO2": 29.9,
            "CO": 1.199,
            "O3": 19.0,
            "NH3": None,
        }
    )
    assert result.valid
    assert result.aqi == 241
    assert result.category == "Poor"
    assert result.dominant_pollutant == "PM2.5"
    assert "NH3" in result.missing_pollutants
    assert "PM2.5" in result.pollutants_used
    assert result.sub_indices["PM2.5"] == pytest.approx(241.33, abs=0.01)


def test_calculate_indian_aqi_needs_three_pollutants():
    result = calculate_indian_aqi({"PM2.5": 100.0, "PM10": 100.0})
    assert not result.valid
    assert result.aqi is None
    assert result.category is None
    assert CPCB_MIN_POLLUTANTS == 3
    assert "at least 3 pollutants" in (result.message or "")


def test_calculate_indian_aqi_ignores_missing_pollutants():
    result = calculate_indian_aqi(
        {"PM2.5": 40.0, "PM10": 60.0, "NO2": None, "SO2": 20.0}
    )
    assert result.valid
    assert result.dominant_pollutant == "PM2.5"
    assert result.aqi == 67
    assert "NO2" in result.missing_pollutants
