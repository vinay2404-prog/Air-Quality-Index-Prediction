"""Dataset loading, validation, filtering, and derived metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    CITY_COORDINATES,
    POLLUTANT_ORDER,
    TARGET_COLUMN,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "Data" / "city_day.csv"

REQUIRED_COLUMNS = ["City", "Date", *POLLUTANT_ORDER, TARGET_COLUMN]


class DataError(RuntimeError):
    """Raised when the historical dataset cannot be loaded or validated."""


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Resolve the dataset path independently of the process working dir."""
    if path is None:
        return DEFAULT_DATA_PATH
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate ``city_day.csv``.

    The cleaned file contains 2,612 rows with an empty AQI target. Training
    and analytics intentionally keep those rows out of the target-bearing
    frame; the model training script uses :func:`training_frame` so no
    synthetic AQI labels are created.
    """
    csv_path = resolve_data_path(path)
    if not csv_path.is_file():
        raise DataError(f"Dataset not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
    except Exception as exc:  # pragma: no cover - pandas message is enough
        raise DataError(f"Unable to read {csv_path}: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise DataError(
            f"{csv_path.name} is missing required columns: {', '.join(missing)}"
        )

    if df.empty:
        raise DataError(f"{csv_path.name} contains no rows")

    numeric = [*POLLUTANT_ORDER, TARGET_COLUMN]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["Date", "City"]).reset_index(drop=True)
    return df


def target_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real (non-imputed) AQI target.

    Used for analytics and evaluation.  Missing pollutant values are allowed;
    they are imputed inside each model pipeline.
    """
    return df.dropna(subset=[TARGET_COLUMN]).copy()


def training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows used to fit the models: real AQI and at least one pollutant.

    Rows without an AQI target are excluded rather than imputed. Rows whose
    AQI is present but whose pollutant columns are all missing are also
    excluded: they carry no features, so a model can only fall back to the
    city average for them.
    """
    frame = target_frame(df)
    featureless = frame[POLLUTANT_ORDER].isna().all(axis=1)
    frame = frame.loc[~featureless].copy()
    if frame.empty:
        raise DataError("No rows with a valid AQI target and pollutant data were found")
    return frame


def pollutant_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Observed min/max/median for each pollutant.

    The prediction UI uses these values instead of hand-written caps, so a
    user can enter every value present in the training data.
    """
    ranges: dict[str, dict[str, float]] = {}
    for pollutant in POLLUTANT_ORDER:
        series = pd.to_numeric(df[pollutant], errors="coerce")
        ranges[pollutant] = {
            "min": float(max(series.min(skipna=True), 0.0)) if series.notna().any() else 0.0,
            "max": float(series.max(skipna=True)) if series.notna().any() else 1000.0,
            "median": float(series.median(skipna=True)) if series.notna().any() else 0.0,
        }
    return ranges


def filter_dataset(
    df: pd.DataFrame,
    start_date: object | None = None,
    end_date: object | None = None,
    cities: list[str] | None = None,
) -> pd.DataFrame:
    """Filter by a closed date range and an optional city subset."""
    filtered = df
    if start_date is not None:
        filtered = filtered[filtered["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered = filtered[filtered["Date"] <= pd.Timestamp(end_date)]
    if cities is not None:
        filtered = filtered[filtered["City"].isin(cities)]
    return filtered.copy()


def city_average_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Average AQI per city with coordinates for ``st.map``."""
    averages = (
        target_frame(df)
        .groupby("City", as_index=False)[TARGET_COLUMN]
        .mean()
        .rename(columns={TARGET_COLUMN: "Average AQI"})
    )
    averages["lat"] = averages["City"].map(
        lambda city: CITY_COORDINATES.get(city, (np.nan, np.nan))[0]
    )
    averages["lon"] = averages["City"].map(
        lambda city: CITY_COORDINATES.get(city, (np.nan, np.nan))[1]
    )
    return averages.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict[str, object]:
    target = target_frame(df)
    return {
        "rows": int(len(df)),
        "target_rows": int(len(target)),
        "cities": int(df["City"].nunique()),
        "start": pd.Timestamp(df["Date"].min()).date(),
        "end": pd.Timestamp(df["Date"].max()).date(),
        "average_aqi": float(target[TARGET_COLUMN].mean()) if not target.empty else None,
        "max_aqi": float(target[TARGET_COLUMN].max()) if not target.empty else None,
    }
