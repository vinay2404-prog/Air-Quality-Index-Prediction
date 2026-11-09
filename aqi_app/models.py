"""Loading the trained model bundle and making predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .constants import (
    AQI_MAX_DISPLAY,
    CATEGORICAL_FEATURE,
    FEATURE_COLUMNS,
    POLLUTANT_ORDER,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "Models" / "aqi_models.joblib"
METADATA_PATH = PROJECT_ROOT / "Models" / "metadata.json"

BUNDLE_FORMAT_VERSION = 1


class ModelError(RuntimeError):
    """Raised when model artifacts are missing or incompatible."""


def load_bundle(path: str | Path | None = None) -> dict:
    """Load the joblib bundle produced by ``scripts/train_models.py``."""
    bundle_path = Path(path) if path is not None else DEFAULT_BUNDLE_PATH
    if not bundle_path.is_file():
        raise ModelError(
            f"Model bundle not found: {bundle_path}. "
            "Run `python scripts/train_models.py` first."
        )

    try:
        bundle = joblib.load(bundle_path)
    except Exception as exc:  # pragma: no cover - depends on artifact state
        raise ModelError(f"Unable to load {bundle_path.name}: {exc}") from exc

    if not isinstance(bundle, dict):
        raise ModelError("Model bundle has an unexpected format")
    if bundle.get("format_version") != BUNDLE_FORMAT_VERSION:
        raise ModelError(
            "Model bundle format is incompatible with this app. "
            "Re-run `python scripts/train_models.py`."
        )
    if not bundle.get("models"):
        raise ModelError("Model bundle does not contain any models")
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ModelError(
            "Model bundle feature order does not match the app. "
            "Re-run `python scripts/train_models.py`."
        )
    return bundle


def model_names(bundle: dict) -> list[str]:
    return list(bundle["models"].keys())


def best_model_name(bundle: dict) -> str:
    return str(bundle.get("best_model") or model_names(bundle)[0])


def metrics_frame(bundle: dict) -> pd.DataFrame:
    rows = []
    for name, payload in bundle["models"].items():
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "Model": name,
                "R²": metrics.get("r2"),
                "MAE": metrics.get("mae"),
                "RMSE": metrics.get("rmse"),
                "Fit time (s)": metrics.get("fit_seconds"),
            }
        )
    frame = pd.DataFrame(rows).sort_values("R²", ascending=False, na_position="last")
    return frame.reset_index(drop=True)


def temporal_metrics_frame(bundle: dict) -> pd.DataFrame:
    """Metrics from the time-based holdout (train on older dates)."""
    rows = []
    for name, payload in bundle["models"].items():
        metrics = payload.get("temporal_metrics", {})
        rows.append(
            {
                "Model": name,
                "R²": metrics.get("r2"),
                "MAE": metrics.get("mae"),
                "RMSE": metrics.get("rmse"),
                "Fit time (s)": metrics.get("fit_seconds"),
            }
        )
    frame = pd.DataFrame(rows).sort_values("R²", ascending=False, na_position="last")
    return frame.reset_index(drop=True)


def temporal_split_info(bundle: dict) -> dict[str, object]:
    return dict(bundle.get("temporal_split", {}))


def best_temporal_model_name(bundle: dict) -> str:
    return str(
        bundle.get("best_model_temporal_split")
        or bundle.get("best_model")
        or model_names(bundle)[0]
    )


def _build_input_frame(city: str, values: dict[str, float | None]) -> pd.DataFrame:
    if not city or not str(city).strip():
        raise ModelError("A city must be selected")

    row: dict[str, object] = {CATEGORICAL_FEATURE: str(city).strip()}
    for pollutant in POLLUTANT_ORDER:
        value = values.get(pollutant)
        if value is None:
            row[pollutant] = np.nan
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ModelError(
                    f"Invalid value for {pollutant}: {value!r}"
                ) from exc
            if not np.isfinite(numeric):
                raise ModelError(f"Invalid value for {pollutant}: {value!r}")
            if numeric < 0:
                raise ModelError(f"{pollutant} cannot be negative")
            row[pollutant] = numeric

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict(
    bundle: dict,
    model_name: str,
    city: str,
    values: dict[str, float | None],
) -> float:
    """Predict AQI for one city and a partial pollutant profile.

    Missing pollutants are imputed by the trained pipeline (median values
    learned from the training split).  Predictions are not silently clipped;
    the caller decides how to display them.
    """
    if model_name not in bundle["models"]:
        raise ModelError(f"Unknown model: {model_name}")

    frame = _build_input_frame(city, values)
    pipeline = bundle["models"][model_name]["pipeline"]

    try:
        prediction = np.asarray(pipeline.predict(frame), dtype=float).reshape(-1)
    except Exception as exc:  # pragma: no cover - depends on artifact state
        raise ModelError(f"Prediction failed with {model_name}: {exc}") from exc

    if prediction.size == 0:
        raise ModelError("The model returned no prediction")
    value = float(prediction[0])
    if not np.isfinite(value):
        raise ModelError("The model returned a non-finite prediction")
    return value


def clip_prediction(
    value: float,
    upper: float | None = None,
) -> tuple[float, bool]:
    """Clamp a prediction to ``[0, upper]`` and report any clamping.

    The CPCB AQI scale is defined from 0 to 500, so the display default caps
    predictions at ``AQI_MAX_DISPLAY``.  Negative model output is never
    meaningful for AQI and is clamped to zero.
    """
    upper_bound = AQI_MAX_DISPLAY if upper is None else float(upper)
    raw = float(value)
    clipped = min(max(raw, 0.0), upper_bound)
    return clipped, clipped != raw


def error_quantiles(bundle: dict, model_name: str) -> dict[str, float]:
    """Signed residual quantiles learned on the time-based holdout."""
    if model_name not in bundle["models"]:
        raise ModelError(f"Unknown model: {model_name}")
    payload = bundle["models"][model_name]
    return {
        key: float(value)
        for key, value in payload.get("error_quantiles", {}).items()
    }


def prediction_interval(
    bundle: dict,
    model_name: str,
    value: float,
    level: float = 0.8,
) -> tuple[float, float] | None:
    """Return a prediction interval around ``value``, or ``None``.

    The interval is based on residual quantiles from the time-based test set,
    then clamped to the CPCB 0–500 scale.  ``level`` must be 0.8 or 0.9.
    """
    pair = {0.8: ("p10", "p90"), 0.9: ("p05", "p95")}.get(round(float(level), 2))
    if pair is None:
        raise ModelError("Prediction intervals support level 0.8 or 0.9")

    quantiles = error_quantiles(bundle, model_name)
    if pair[0] not in quantiles or pair[1] not in quantiles:
        return None

    raw = float(value)
    low, _ = clip_prediction(raw + quantiles[pair[0]])
    high, _ = clip_prediction(raw + quantiles[pair[1]])
    return (min(low, high), max(low, high))


def predict_frame(
    bundle: dict,
    model_name: str,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Predict every row of a feature frame (used for hourly forecasts)."""
    if model_name not in bundle["models"]:
        raise ModelError(f"Unknown model: {model_name}")

    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ModelError(f"Missing feature columns: {', '.join(missing)}")
    if frame.empty:
        return np.asarray([], dtype=float)

    pipeline = bundle["models"][model_name]["pipeline"]
    try:
        predictions = np.asarray(
            pipeline.predict(frame[FEATURE_COLUMNS]),
            dtype=float,
        ).reshape(-1)
    except Exception as exc:  # pragma: no cover - depends on artifact state
        raise ModelError(f"Batch prediction failed with {model_name}: {exc}") from exc

    if predictions.size != len(frame):
        raise ModelError("The model returned an unexpected number of predictions")
    if not np.isfinite(predictions).all():
        raise ModelError("The model returned non-finite predictions")
    return predictions


def input_ranges(bundle: dict) -> dict[str, dict[str, float]]:
    return dict(bundle.get("input_ranges", {}))


def dataset_aqi_percentile(bundle: dict, value: float) -> float:
    """Percentile of ``value`` in the training target distribution."""
    sample = np.asarray(bundle.get("target_sample", []), dtype=float)
    if sample.size == 0:
        return float("nan")
    return float((sample <= value).mean() * 100.0)


def target_stats(bundle: dict) -> dict[str, float]:
    return dict(bundle.get("target_stats", {}))


def cities(bundle: dict) -> list[str]:
    return list(bundle.get("cities", []))


def trained_at(bundle: dict) -> str:
    return str(bundle.get("trained_at", "unknown"))
