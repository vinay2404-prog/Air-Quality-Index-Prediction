"""Reproducible data-quality corrections for ``Data/city_day.csv``.

The raw Kaggle extract has several documented issues:

* AQI values above the CPCB maximum of 500 (some above 1,000).
* Ahmedabad's CO column is effectively a copy of NO, despite the different
  units (CO is mg/m3, NO is ug/m3).
* Exact zeros in CO, benzene, toluene, and xylene look like below-detection
  placeholders rather than real measurements.
* NOx is sometimes lower than NO or NO2, which is physically impossible.
* PM2.5 is occasionally larger than PM10, although PM2.5 is a subset of PM10.
* Some rows contain no measurements and no target at all.

This module applies conservative corrections and returns a report with the
number of affected values.  The raw file is never modified.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import AQI_MAX_DISPLAY, POLLUTANT_ORDER, TARGET_COLUMN, aqi_category

# Columns expected in the cleaned file, in their original order.
CLEAN_COLUMNS: list[str] = [
    "City",
    "Date",
    *POLLUTANT_ORDER,
    TARGET_COLUMN,
    "AQI_Bucket",
]

# Exact zeros in these pollutants are treated as below-detection/missing.
ZERO_AS_MISSING: list[str] = ["CO", "Benzene", "Toluene", "Xylene"]

# Ahmedabad's CO values duplicate its NO values, so the whole city's CO series
# is unusable and is blanked.
INVALID_CO_CITY = "Ahmedabad"


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a corrected copy of ``df`` and a report of what changed."""
    cleaned = df.copy()
    report: dict[str, Any] = {
        "rows_before": int(len(cleaned)),
    }

    pollutant_columns = [column for column in POLLUTANT_ORDER if column in cleaned]
    numeric_columns = [*pollutant_columns, TARGET_COLUMN]

    # 1. Remove rows that are already completely empty.
    empty_mask = (
        cleaned[pollutant_columns].isna().all(axis=1)
        & cleaned[TARGET_COLUMN].isna()
    )
    report["rows_dropped_initially_empty"] = int(empty_mask.sum())
    cleaned = cleaned.loc[~empty_mask].copy()

    # 2. Defensive numeric coercion and a final negative-value guard.
    negative_values = 0
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        negatives = cleaned[column] < 0
        negative_values += int(negatives.sum())
        cleaned.loc[negatives, column] = np.nan
    report["negative_values_set_missing"] = negative_values

    # 3. AQI is defined on the CPCB 0-500 scale.
    above_max = cleaned[TARGET_COLUMN] > AQI_MAX_DISPLAY
    report["aqi_capped_at_500"] = int(above_max.sum())
    cleaned.loc[above_max, TARGET_COLUMN] = AQI_MAX_DISPLAY

    # 4. Keep AQI_Bucket consistent with the (capped) numeric AQI.
    expected_bucket = cleaned[TARGET_COLUMN].map(
        lambda value: aqi_category(value) if pd.notna(value) else None
    )
    report["aqi_bucket_recomputed"] = int(
        (
            cleaned["AQI_Bucket"].notna()
            & (cleaned["AQI_Bucket"] != expected_bucket)
        ).sum()
    )
    cleaned["AQI_Bucket"] = expected_bucket

    # 5. Ahmedabad's CO column is a copy of NO in different units.
    ahmedabad = cleaned["City"] == INVALID_CO_CITY
    report["ahmedabad_co_set_missing"] = int(
        cleaned.loc[ahmedabad, "CO"].notna().sum()
    )
    cleaned.loc[ahmedabad, "CO"] = np.nan

    # 6. Exact zeros in these pollutants are treated as missing.
    for column in ZERO_AS_MISSING:
        if column not in cleaned:
            continue
        zeros = cleaned[column] == 0
        report[f"{column.lower()}_zeros_set_missing"] = int(zeros.sum())
        cleaned.loc[zeros, column] = np.nan

    # 7. NOx cannot be lower than NO or NO2.
    invalid_nox = cleaned["NOx"].notna() & (
        (cleaned["NO"].notna() & (cleaned["NOx"] < cleaned["NO"]))
        | (cleaned["NO2"].notna() & (cleaned["NOx"] < cleaned["NO2"]))
    )
    report["nox_inconsistent_set_missing"] = int(invalid_nox.sum())
    cleaned.loc[invalid_nox, "NOx"] = np.nan

    # 8. PM2.5 is a fraction of PM10.  If PM2.5 > PM10, the PM10 value is
    #    invalid; keep the PM2.5 measurement.
    invalid_pm10 = (
        cleaned["PM2.5"].notna()
        & cleaned["PM10"].notna()
        & (cleaned["PM2.5"] > cleaned["PM10"])
    )
    report["pm10_set_missing_when_below_pm25"] = int(invalid_pm10.sum())
    cleaned.loc[invalid_pm10, "PM10"] = np.nan

    # 9. Some rows become empty after the value corrections above (for example
    #    a row whose only values were zero placeholders).  Drop those too.
    empty_after = (
        cleaned[pollutant_columns].isna().all(axis=1)
        & cleaned[TARGET_COLUMN].isna()
    )
    report["rows_dropped_after_cleaning"] = int(empty_after.sum())
    cleaned = cleaned.loc[~empty_after].copy()
    report["rows_dropped_all_empty"] = (
        report["rows_dropped_initially_empty"]
        + report["rows_dropped_after_cleaning"]
    )

    cleaned = cleaned.sort_values(["Date", "City"]).reset_index(drop=True)
    report["rows_after"] = int(len(cleaned))
    return cleaned[CLEAN_COLUMNS], report
