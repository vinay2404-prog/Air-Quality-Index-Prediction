import math

import pandas as pd
import pytest

from aqi_app.constants import POLLUTANT_ORDER
from aqi_app.models import (
    ModelError,
    best_model_name,
    best_temporal_model_name,
    cities,
    clip_prediction,
    dataset_aqi_percentile,
    error_quantiles,
    input_ranges,
    load_bundle,
    metrics_frame,
    model_names,
    predict,
    predict_frame,
    prediction_interval,
    target_stats,
    temporal_metrics_frame,
    temporal_split_info,
)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_bundle_metadata(bundle):
    assert bundle["format_version"] == 1
    assert bundle["feature_columns"] == ["City", *POLLUTANT_ORDER]
    assert bundle["training_rows"] > 18_000
    assert bundle["test_rows"] > 6_000
    assert bundle["excluded_empty_targets"] == 2_612
    assert bundle["excluded_featureless_targets"] == 52
    assert len(cities(bundle)) == 26
    assert len(model_names(bundle)) == 6
    assert best_model_name(bundle) == "Gradient Boosting"


def test_metrics_frame(bundle):
    frame = metrics_frame(bundle)
    assert len(frame) == 6
    assert {"Model", "R²", "MAE", "RMSE"} <= set(frame.columns)
    assert frame["R²"].max() >= 0.9
    assert frame.iloc[0]["Model"] == best_model_name(bundle)


def test_temporal_metrics_frame(bundle):
    frame = temporal_metrics_frame(bundle)
    assert len(frame) == 6
    assert frame["R²"].notna().all()
    assert frame["MAE"].notna().all()
    assert frame["R²"].max() >= 0.8

    split = temporal_split_info(bundle)
    assert split["cutoff"] == "2019-07-01"
    assert split["training_rows"] + split["test_rows"] == 24_798
    assert best_temporal_model_name(bundle) in model_names(bundle)


def test_input_ranges_cover_all_pollutants(bundle):
    ranges = input_ranges(bundle)
    assert set(ranges) == set(POLLUTANT_ORDER)
    assert ranges["PM2.5"]["max"] >= 900
    assert ranges["PM10"]["max"] >= 900


def test_predict_with_complete_profile(bundle):
    values = {
        "PM2.5": 80.0,
        "PM10": 140.0,
        "NO": 20.0,
        "NO2": 30.0,
        "NOx": 35.0,
        "NH3": 25.0,
        "CO": 2.5,
        "SO2": 15.0,
        "O3": 35.0,
        "Benzene": 3.0,
        "Toluene": 9.0,
        "Xylene": 3.0,
    }
    result = predict(bundle, best_model_name(bundle), "Delhi", values)
    assert math.isfinite(result)
    assert result > 0


def test_predict_with_missing_pollutants(bundle):
    result = predict(bundle, "Random Forest", "Delhi", {"PM2.5": 80.0})
    assert math.isfinite(result)
    assert result > 0


def test_predict_unknown_city_is_handled(bundle):
    result = predict(bundle, "Random Forest", "Atlantis", {"PM2.5": 50.0})
    assert math.isfinite(result)


def test_predict_rejects_bad_input(bundle):
    with pytest.raises(ModelError):
        predict(bundle, "Random Forest", "", {"PM2.5": 50.0})
    with pytest.raises(ModelError):
        predict(bundle, "Random Forest", "Delhi", {"PM2.5": -1.0})
    with pytest.raises(ModelError):
        predict(bundle, "Not a model", "Delhi", {"PM2.5": 50.0})


def test_error_quantiles_and_prediction_interval(bundle):
    quantiles = error_quantiles(bundle, "Gradient Boosting")
    assert {"p05", "p10", "p90", "p95"} <= set(quantiles)
    assert quantiles["p05"] <= quantiles["p10"] <= quantiles["p90"] <= quantiles["p95"]

    interval = prediction_interval(bundle, "Gradient Boosting", 120.0, level=0.8)
    assert interval is not None
    low, high = interval
    assert 0.0 <= low <= high <= 500.0
    assert low <= 120.0 <= high

    wider = prediction_interval(bundle, "Gradient Boosting", 120.0, level=0.9)
    assert wider is not None
    assert wider[0] <= low and wider[1] >= high

    with pytest.raises(ModelError):
        error_quantiles(bundle, "Not a model")
    with pytest.raises(ModelError):
        prediction_interval(bundle, "Gradient Boosting", 120.0, level=0.5)


def test_predict_frame_batch(bundle):
    base = {name: None for name in POLLUTANT_ORDER}
    base.update({"PM2.5": 80.0, "PM10": 140.0})
    frame = pd.DataFrame(
        [
            {"City": "Delhi", **base},
            {"City": "Mumbai", **base},
        ]
    )

    predictions = predict_frame(bundle, best_model_name(bundle), frame)
    assert predictions.shape == (2,)
    assert all(math.isfinite(value) for value in predictions)

    with pytest.raises(ModelError):
        predict_frame(bundle, best_model_name(bundle), frame[["City", "PM2.5"]])


def test_clip_prediction():
    assert clip_prediction(-5.0) == (0.0, True)
    assert clip_prediction(42.0) == (42.0, False)
    assert clip_prediction(650.0) == (500.0, True)
    assert clip_prediction(650.0, upper=600) == (600.0, True)
    assert clip_prediction(500.0) == (500.0, False)


def test_target_statistics_and_percentile(bundle):
    stats = target_stats(bundle)
    assert stats["min"] <= stats["median"] <= stats["max"]
    percentile = dataset_aqi_percentile(bundle, stats["median"])
    assert 0 <= percentile <= 100
