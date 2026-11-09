import numpy as np
import pandas as pd

from aqi_app.preprocessing import CityMedianImputer


def test_city_median_imputer_uses_city_then_global_medians():
    train = pd.DataFrame(
        {
            "City": ["Delhi", "Delhi", "Mumbai", "Mumbai", "Mumbai"],
            "A": [10.0, 20.0, 100.0, np.nan, 300.0],
            "B": [1.0, np.nan, 5.0, 7.0, 9.0],
        }
    )
    imputer = CityMedianImputer().fit(train)
    transformed = imputer.transform(
        pd.DataFrame(
            {
                "City": ["Delhi", "Mumbai", "Chennai"],
                "A": [np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan],
            }
        )
    )

    # Delhi A median is 15; Mumbai A median is 200 (NaN ignored).
    assert transformed[0, 0] == 15.0
    assert transformed[1, 0] == 200.0
    # Chennai is unseen: global A median is (20 + 100) / 2 = 60.
    assert transformed[2, 0] == 60.0
    # B is only missing for Delhi, which uses the Delhi median of 1.
    assert transformed[0, 1] == 1.0


def test_city_median_imputer_handles_unknown_and_all_missing_columns():
    train = pd.DataFrame(
        {
            "City": ["Delhi", "Mumbai"],
            "A": [np.nan, np.nan],
        }
    )
    imputer = CityMedianImputer().fit(train)
    transformed = imputer.transform(
        pd.DataFrame({"City": ["Atlantis"], "A": [np.nan]})
    )
    assert transformed.shape == (1, 1)
    assert transformed[0, 0] == 0.0
