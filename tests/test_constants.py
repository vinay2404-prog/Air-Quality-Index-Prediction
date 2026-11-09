from aqi_app.constants import (
    AQI_BANDS,
    CITY_COORDINATES,
    POLLUTANT_ORDER,
    POLLUTANTS,
    UNIT_GROUPS,
    aqi_category,
    aqi_progress,
    nearest_supported_city,
    pollutant_label,
    us_aqi_category,
)


def test_cpcb_band_boundaries():
    assert aqi_category(0) == "Good"
    assert aqi_category(50) == "Good"
    assert aqi_category(51) == "Satisfactory"
    assert aqi_category(100) == "Satisfactory"
    assert aqi_category(101) == "Moderate"
    assert aqi_category(200) == "Moderate"
    assert aqi_category(201) == "Poor"
    assert aqi_category(300) == "Poor"
    assert aqi_category(301) == "Very Poor"
    assert aqi_category(400) == "Very Poor"
    assert aqi_category(401) == "Severe"
    assert aqi_category(2500) == "Severe"


def test_negative_aqi_clamps_to_good():
    assert aqi_category(-10) == "Good"
    assert aqi_progress(-10) == 0.0
    assert aqi_progress(250) == 0.5
    assert aqi_progress(900) == 1.0


def test_bands_cover_full_range():
    assert AQI_BANDS[0]["min"] == 0
    assert AQI_BANDS[-1]["max"] == float("inf")


def test_us_aqi_bands():
    assert us_aqi_category(50) == "Good"
    assert us_aqi_category(100) == "Moderate"
    assert us_aqi_category(150) == "Unhealthy for Sensitive Groups"
    assert us_aqi_category(200) == "Unhealthy"
    assert us_aqi_category(300) == "Very Unhealthy"
    assert us_aqi_category(301) == "Hazardous"


def test_unit_groups_cover_every_pollutant():
    grouped = {pollutant for group in UNIT_GROUPS.values() for pollutant in group}
    assert grouped == set(POLLUTANT_ORDER)
    assert POLLUTANTS["CO"]["unit"] == "mg/m³"
    assert POLLUTANTS["NOx"]["unit"] == "µg/m³"


def test_pollutant_label_contains_unit():
    assert "µg/m³" in pollutant_label("PM2.5")
    assert "mg/m³" in pollutant_label("CO")


def test_city_coordinates_cover_supported_cities():
    assert len(CITY_COORDINATES) == 26
    assert "Delhi" in CITY_COORDINATES


def test_nearest_supported_city():
    delhi_lat, delhi_lon = CITY_COORDINATES["Delhi"]
    assert nearest_supported_city(delhi_lat, delhi_lon) == "Delhi"
    mumbai_lat, mumbai_lon = CITY_COORDINATES["Mumbai"]
    assert nearest_supported_city(mumbai_lat, mumbai_lon) == "Mumbai"
