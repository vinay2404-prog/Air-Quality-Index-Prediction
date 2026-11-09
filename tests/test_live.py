import pandas as pd
import pytest
import requests

from aqi_app.constants import POLLUTANT_ORDER
from aqi_app.live import (
    LiveDataError,
    _request_json,
    current_snapshot,
    daily_forecast,
    hourly_feature_frame,
    hourly_frame,
    hourly_indian_aqi,
    model_features,
)

PAYLOAD = {
    "latitude": 28.65,
    "longitude": 77.23,
    "timezone": "Asia/Kolkata",
    "timezone_abbreviation": "IST",
    "current": {
        "time": "2026-11-08T01:30",
        "interval": 3600,
        "us_aqi": 171,
        "european_aqi": 85,
        "pm10": 192.3,
        "pm2_5": 102.4,
        "carbon_monoxide": 1199.0,
        "nitrogen_dioxide": 67.3,
        "sulphur_dioxide": 29.9,
        "ozone": 19.0,
        "ammonia": None,
    },
    "hourly": {
        "time": [
            "2026-11-08T00:00",
            "2026-11-08T01:00",
            "2026-11-09T00:00",
            "2026-11-09T01:00",
        ],
        "us_aqi": [160, 170, 180, 190],
        "european_aqi": [80, 85, 90, 95],
        "pm2_5": [90.0, 100.0, 110.0, 120.0],
        "pm10": [180.0, 190.0, 200.0, 210.0],
        "carbon_monoxide": [1000.0, 1100.0, 1200.0, 1300.0],
        "nitrogen_dioxide": [60.0, 65.0, 70.0, 75.0],
        "sulphur_dioxide": [25.0, 30.0, 35.0, 40.0],
        "ozone": [15.0, 20.0, 25.0, 30.0],
        "ammonia": [None, None, None, None],
    },
}


def test_current_snapshot_converts_co_and_computes_indian_aqi():
    snapshot = current_snapshot(PAYLOAD)
    assert snapshot["us_aqi"] == 171
    assert snapshot["european_aqi"] == 85
    assert snapshot["pollutants"]["PM2.5"] == 102.4
    assert snapshot["pollutants"]["CO"] == 1.199
    assert snapshot["pollutants"]["NH3"] is None
    assert snapshot["timezone"] == "Asia/Kolkata"

    indian = snapshot["indian_aqi"]
    assert indian.valid
    assert indian.aqi == 241
    assert indian.category == "Poor"
    assert indian.dominant_pollutant == "PM2.5"


def test_model_features_include_missing_values():
    features = model_features(current_snapshot(PAYLOAD))
    assert set(features) == set(POLLUTANT_ORDER)
    assert features["PM2.5"] == 102.4
    assert features["NO"] is None
    assert features["Benzene"] is None


def test_hourly_frame_and_daily_forecast_use_indian_aqi():
    frame = hourly_frame(PAYLOAD)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == 4
    assert frame["us_aqi"].max() == 190

    indian = hourly_indian_aqi(frame)
    assert indian.tolist() == [200.0, 233.0, 267.0, 300.0]

    daily = daily_forecast(frame.assign(india_aqi=indian))
    assert len(daily) == 2
    assert daily.iloc[0]["Peak Indian AQI"] == 233
    assert daily.iloc[1]["Peak Indian AQI"] == 300
    assert "Peak US AQI" not in daily.columns
    assert daily.iloc[0]["Category"] == "Poor"
    assert daily.iloc[1]["Category"] == "Poor"


def test_daily_forecast_computes_indian_aqi_when_absent():
    frame = hourly_frame(PAYLOAD)
    daily = daily_forecast(frame)
    assert daily.iloc[0]["Peak Indian AQI"] == 233
    assert daily.iloc[1]["Peak Indian AQI"] == 300


def test_hourly_feature_frame_matches_model_columns():
    frame = hourly_frame(PAYLOAD)
    features = hourly_feature_frame(frame, "Delhi")

    assert list(features.columns) == ["City", *POLLUTANT_ORDER]
    assert features.index.equals(frame.index)
    assert (features["City"] == "Delhi").all()
    assert features["PM2.5"].tolist() == [90.0, 100.0, 110.0, 120.0]
    assert features["CO"].iloc[0] == 1.0
    assert features["NO"].isna().all()
    assert features["Benzene"].isna().all()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_request_json_retries_then_succeeds(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("temporary failure")
        return _FakeResponse({"ok": True})

    monkeypatch.setattr("aqi_app.live.requests.get", fake_get)
    monkeypatch.setattr("aqi_app.live.time.sleep", lambda _: None)

    payload = _request_json("https://example.test", {}, error_label="Test", timeout=1)
    assert payload == {"ok": True}
    assert len(calls) == 2


def test_request_json_raises_after_all_attempts(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(1)
        raise requests.ConnectionError("still down")

    monkeypatch.setattr("aqi_app.live.requests.get", fake_get)
    monkeypatch.setattr("aqi_app.live.time.sleep", lambda _: None)

    with pytest.raises(LiveDataError):
        _request_json("https://example.test", {}, error_label="Test", timeout=1)
    assert len(calls) == 3
