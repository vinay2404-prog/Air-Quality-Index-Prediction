import pandas as pd

from aqi_app.cleaning import ZERO_AS_MISSING
from aqi_app.constants import POLLUTANT_ORDER, TARGET_COLUMN, aqi_category
from aqi_app.data import (
    city_average_frame,
    dataset_summary,
    filter_dataset,
    load_dataset,
    pollutant_ranges,
    target_frame,
    training_frame,
)


def test_dataset_loads_and_has_expected_shape():
    df = load_dataset()
    assert len(df) == 27_462
    assert df["City"].nunique() == 26
    assert list(df.columns[:3]) == ["City", "Date", "PM2.5"]
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])


def test_training_frame_excludes_empty_targets():
    df = load_dataset()
    target = target_frame(df)
    training = training_frame(df)

    assert len(target) == 24_850
    assert len(training) == 24_798
    assert training[TARGET_COLUMN].isna().sum() == 0
    assert not training[POLLUTANT_ORDER].isna().all(axis=1).any()
    assert len(df) - len(training) == 2_664

    featureless = target[POLLUTANT_ORDER].isna().all(axis=1)
    assert int(featureless.sum()) == 52


def test_pollutant_ranges_match_data():
    df = load_dataset()
    ranges = pollutant_ranges(df)
    assert set(ranges) == set(POLLUTANT_ORDER)
    for values in ranges.values():
        assert values["max"] >= values["min"]
        assert values["median"] >= 0
    assert ranges["PM2.5"]["max"] >= 900
    assert ranges["CO"]["max"] >= 40
    assert ranges["CO"]["max"] <= 100


def test_filter_dataset_by_city_and_date():
    df = load_dataset()
    subset = filter_dataset(
        df,
        start_date="2019-01-01",
        end_date="2019-12-31",
        cities=["Delhi", "Mumbai"],
    )
    assert not subset.empty
    assert set(subset["City"].unique()) <= {"Delhi", "Mumbai"}
    assert subset["Date"].min() >= pd.Timestamp("2019-01-01")
    assert subset["Date"].max() <= pd.Timestamp("2019-12-31")

    assert filter_dataset(df, cities=[]).empty


def test_city_average_frame_has_coordinates():
    df = load_dataset()
    averages = city_average_frame(df)
    assert not averages.empty
    assert {"City", "Average AQI", "lat", "lon"} <= set(averages.columns)
    assert averages["lat"].between(-90, 90).all()
    assert averages["lon"].between(-180, 180).all()


def test_dataset_summary():
    summary = dataset_summary(load_dataset())
    assert summary["cities"] == 26
    assert summary["target_rows"] == 24_850
    assert str(summary["start"]) == "2015-01-01"
    assert str(summary["end"]) == "2020-07-01"


def test_cleaned_dataset_consistency():
    df = load_dataset()

    assert df[TARGET_COLUMN].max() <= 500
    assert (df[TARGET_COLUMN].dropna() >= 0).all()

    numeric = [*POLLUTANT_ORDER, TARGET_COLUMN]
    assert (df[numeric].dropna() >= 0).all().all()

    # Ahmedabad's CO column duplicated NO and was blanked during cleaning.
    assert df.loc[df["City"] == "Ahmedabad", "CO"].isna().all()

    # Exact zeros were treated as below-detection placeholders.
    for column in ZERO_AS_MISSING:
        assert not (df[column] == 0).any()

    nox_no = df["NOx"].notna() & df["NO"].notna()
    assert (df.loc[nox_no, "NOx"] >= df.loc[nox_no, "NO"]).all()

    nox_no2 = df["NOx"].notna() & df["NO2"].notna()
    assert (df.loc[nox_no2, "NOx"] >= df.loc[nox_no2, "NO2"]).all()

    both_pm = df["PM2.5"].notna() & df["PM10"].notna()
    assert (df.loc[both_pm, "PM2.5"] <= df.loc[both_pm, "PM10"]).all()

    labelled = df["AQI_Bucket"].notna() & df[TARGET_COLUMN].notna()
    expected = df.loc[labelled, TARGET_COLUMN].map(aqi_category)
    assert (df.loc[labelled, "AQI_Bucket"] == expected).all()

    assert not (df[POLLUTANT_ORDER].isna().all(axis=1) & df[TARGET_COLUMN].isna()).any()
