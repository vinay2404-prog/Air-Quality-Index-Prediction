import numpy as np
import pandas as pd

from aqi_app.cleaning import clean_dataset
from aqi_app.constants import POLLUTANT_ORDER


def make_frame() -> pd.DataFrame:
    row = {column: np.nan for column in POLLUTANT_ORDER}
    rows = []

    ahmedabad = dict(row)
    ahmedabad.update(
        {
            "City": "Ahmedabad",
            "Date": pd.Timestamp("2018-02-19"),
            "PM2.5": 60.0,
            "PM10": 50.0,
            "NO": 10.0,
            "NO2": 5.0,
            "NOx": 3.0,
            "CO": 10.0,
            "Benzene": 0.0,
            "Toluene": 2.0,
            "Xylene": 1.0,
            "AQI": 2049.0,
            "AQI_Bucket": "Severe",
        }
    )
    rows.append(ahmedabad)

    delhi = dict(row)
    delhi.update(
        {
            "City": "Delhi",
            "Date": pd.Timestamp("2018-02-20"),
            "PM2.5": 40.0,
            "PM10": 80.0,
            "NO": 10.0,
            "NO2": 5.0,
            "NOx": 20.0,
            "CO": 0.0,
            "Benzene": 0.0,
            "Toluene": 0.0,
            "Xylene": 0.0,
            "AQI": 100.0,
            "AQI_Bucket": "Satisfactory",
        }
    )
    rows.append(delhi)

    # Completely empty row: should be dropped.
    empty = dict(row)
    empty.update(
        {
            "City": "Delhi",
            "Date": pd.Timestamp("2018-02-21"),
            "AQI": np.nan,
            "AQI_Bucket": np.nan,
        }
    )
    rows.append(empty)

    # Target-only row: keep it.
    target_only = dict(row)
    target_only.update(
        {
            "City": "Delhi",
            "Date": pd.Timestamp("2018-02-22"),
            "AQI": 50.0,
            "AQI_Bucket": "Good",
        }
    )
    rows.append(target_only)

    # Negative value: should be blanked.
    negative = dict(row)
    negative.update(
        {
            "City": "Delhi",
            "Date": pd.Timestamp("2018-02-23"),
            "PM2.5": -5.0,
            "PM10": 20.0,
            "NO": 1.0,
            "NO2": 1.0,
            "NOx": 2.0,
            "AQI": 60.0,
            "AQI_Bucket": "Satisfactory",
        }
    )
    rows.append(negative)

    return pd.DataFrame(rows)


def test_clean_dataset_applies_all_corrections():
    cleaned, report = clean_dataset(make_frame())

    assert report["rows_before"] == 5
    assert report["rows_after"] == 4
    assert report["rows_dropped_all_empty"] == 1
    assert report["aqi_capped_at_500"] == 1
    assert report["ahmedabad_co_set_missing"] == 1
    assert report["co_zeros_set_missing"] == 1
    assert report["benzene_zeros_set_missing"] == 2
    assert report["toluene_zeros_set_missing"] == 1
    assert report["xylene_zeros_set_missing"] == 1
    assert report["nox_inconsistent_set_missing"] == 1
    assert report["pm10_set_missing_when_below_pm25"] == 1
    assert report["negative_values_set_missing"] == 1

    assert cleaned["AQI"].max() == 500
    assert cleaned["AQI"].isna().sum() == 0
    assert cleaned["AQI_Bucket"].tolist() == ["Severe", "Satisfactory", "Good", "Satisfactory"]

    ahmedabad = cleaned[cleaned["City"] == "Ahmedabad"].iloc[0]
    assert pd.isna(ahmedabad["CO"])
    assert pd.isna(ahmedabad["PM10"])
    assert pd.isna(ahmedabad["NOx"])
    assert ahmedabad["PM2.5"] == 60.0

    delhi = cleaned[(cleaned["City"] == "Delhi") & (cleaned["Date"] == pd.Timestamp("2018-02-20"))].iloc[0]
    assert pd.isna(delhi["CO"])
    assert pd.isna(delhi["Benzene"])
    assert pd.isna(delhi["Toluene"])
    assert pd.isna(delhi["Xylene"])

    negative = cleaned[cleaned["Date"] == pd.Timestamp("2018-02-23")].iloc[0]
    assert pd.isna(negative["PM2.5"])


def test_clean_dataset_returns_expected_columns():
    cleaned, _ = clean_dataset(make_frame())
    assert list(cleaned.columns) == [
        "City",
        "Date",
        *POLLUTANT_ORDER,
        "AQI",
        "AQI_Bucket",
    ]
    assert cleaned["Date"].is_monotonic_increasing
