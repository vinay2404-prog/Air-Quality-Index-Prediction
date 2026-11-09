"""Train, evaluate, and persist the AQI regression models.

Run from the project root:

    python scripts/train_models.py

The script writes ``Models/aqi_models.joblib`` and ``Models/metadata.json``.
Everything the app needs (feature order, input ranges, metrics) is stored in
the bundle, so the app never has to guess how the models were built.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aqi_app.constants import (  # noqa: E402
    CATEGORICAL_FEATURE,
    FEATURE_COLUMNS,
    POLLUTANT_ORDER,
    TARGET_COLUMN,
)
from aqi_app.data import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_dataset,
    pollutant_ranges,
    target_frame,
    training_frame,
)
from aqi_app.preprocessing import CityMedianImputer  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "Models"
BUNDLE_PATH = MODELS_DIR / "aqi_models.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.25
TEMPORAL_CUTOFF = pd.Timestamp("2019-07-01")



def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False,
    )


def _numeric_imputer(
    add_indicator: bool,
    imputation: str,
) -> object:
    if imputation == "city":
        return CityMedianImputer()
    return SimpleImputer(strategy="median", add_indicator=add_indicator)


def tree_preprocessor(
    numeric_features: list[str] | None = None,
    add_indicator: bool = True,
    imputation: str = "global",
) -> ColumnTransformer:
    numeric_features = list(numeric_features or POLLUTANT_ORDER)
    imputer_columns = (
        [CATEGORICAL_FEATURE, *numeric_features]
        if imputation == "city"
        else numeric_features
    )
    return ColumnTransformer(
        [
            ("city", _one_hot(), [CATEGORICAL_FEATURE]),
            (
                "numeric",
                _numeric_imputer(add_indicator, imputation),
                imputer_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def scaled_preprocessor(
    numeric_features: list[str] | None = None,
    add_indicator: bool = True,
    imputation: str = "global",
) -> ColumnTransformer:
    numeric_features = list(numeric_features or POLLUTANT_ORDER)
    imputer_columns = (
        [CATEGORICAL_FEATURE, *numeric_features]
        if imputation == "city"
        else numeric_features
    )
    return ColumnTransformer(
        [
            ("city", _one_hot(), [CATEGORICAL_FEATURE]),
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            _numeric_imputer(add_indicator, imputation),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                imputer_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def native_nan_preprocessor(
    numeric_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_features = list(numeric_features or POLLUTANT_ORDER)
    return ColumnTransformer(
        [
            ("city", _one_hot(), [CATEGORICAL_FEATURE]),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(
    numeric_features: list[str] | None = None,
    add_indicator: bool = True,
    imputation: str = "global",
) -> dict[str, tuple[Pipeline, str]]:
    """Return ``name -> (pipeline, short description)``."""
    return {
        "Random Forest": (
            Pipeline(
                [
                    (
                        "preprocess",
                        tree_preprocessor(
                            numeric_features,
                            add_indicator,
                            imputation,
                        ),
                    ),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=200,
                            min_samples_leaf=2,
                            max_features=0.8,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Ensemble of decision trees; usually the most accurate option here.",
        ),
        "Gradient Boosting": (
            Pipeline(
                [
                    ("preprocess", native_nan_preprocessor(numeric_features)),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            learning_rate=0.08,
                            max_iter=350,
                            max_leaf_nodes=31,
                            l2_regularization=0.1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Boosted trees with native missing-value support.",
        ),
        "Polynomial Regression": (
            Pipeline(
                [
                    (
                        "preprocess",
                        scaled_preprocessor(
                            numeric_features,
                            add_indicator,
                            imputation,
                        ),
                    ),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("model", LinearRegression()),
                ]
            ),
            "Linear regression over degree-2 polynomial features.",
        ),
        "Multiple Regression": (
            Pipeline(
                [
                    (
                        "preprocess",
                        scaled_preprocessor(
                            numeric_features,
                            add_indicator,
                            imputation,
                        ),
                    ),
                    ("model", LinearRegression()),
                ]
            ),
            "Linear regression baseline.",
        ),
        "Decision Tree": (
            Pipeline(
                [
                    (
                        "preprocess",
                        tree_preprocessor(
                            numeric_features,
                            add_indicator,
                            imputation,
                        ),
                    ),
                    (
                        "model",
                        DecisionTreeRegressor(
                            min_samples_leaf=2,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Single decision tree; interpretable but less accurate.",
        ),
        "SVR": (
            Pipeline(
                [
                    (
                        "preprocess",
                        scaled_preprocessor(
                            numeric_features,
                            add_indicator,
                            imputation,
                        ),
                    ),
                    (
                        "model",
                        TransformedTargetRegressor(
                            regressor=SVR(
                                C=100.0,
                                epsilon=0.5,
                                gamma="scale",
                                cache_size=500,
                            ),
                            transformer=StandardScaler(),
                        ),
                    ),
                ]
            ),
            "Support Vector Regression with target scaling.",
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "medae": float(median_absolute_error(y_true, y_pred)),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    print(f"Loading {DEFAULT_DATA_PATH} ...")
    raw = load_dataset()
    target_rows = target_frame(raw)
    frame = training_frame(raw)

    excluded_empty_targets = int(len(raw) - len(target_rows))
    excluded_featureless_targets = int(len(target_rows) - len(frame))
    print(
        f"Training on {len(frame):,} rows with a real AQI target and at least "
        f"one pollutant ({excluded_empty_targets:,} empty targets and "
        f"{excluded_featureless_targets:,} featureless targets excluded)."
    )

    X = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"Random split — train rows: {len(X_train):,} | test rows: {len(X_test):,}")

    trained_models: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for name, (pipeline, description) in build_models().items():
        print(f"\nFitting {name} ...", flush=True)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        predictions = np.asarray(pipeline.predict(X_test), dtype=float).reshape(-1)
        metrics = evaluate(y_test.to_numpy(), predictions)
        metrics["fit_seconds"] = float(fit_seconds)

        trained_models[name] = {
            "pipeline": pipeline,
            "description": description,
            "metrics": metrics,
        }
        summary_rows.append({"model": name, **metrics})

        print(
            f"  R²={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}  "
            f"RMSE={metrics['rmse']:.2f}  fit={fit_seconds:.1f}s"
        )

    best_model = max(summary_rows, key=lambda row: row["r2"])["model"]
    print(f"\nBest random-split model: {best_model}")

    # ------------------------------------------------------------------
    # Time-based evaluation
    # ------------------------------------------------------------------
    temporal_train = frame[frame["Date"] < TEMPORAL_CUTOFF]
    temporal_test = frame[frame["Date"] >= TEMPORAL_CUTOFF]
    if temporal_train.empty or temporal_test.empty:
        raise SystemExit(
            f"Temporal split at {TEMPORAL_CUTOFF.date()} produced an empty "
            "training or test set."
        )

    print(
        f"\nTemporal split at {TEMPORAL_CUTOFF.date()} — "
        f"train rows: {len(temporal_train):,} | test rows: {len(temporal_test):,}"
    )

    temporal_rows: list[dict] = []
    for name, (pipeline, _) in build_models().items():
        print(f"Fitting {name} on temporal split ...", flush=True)
        started = time.perf_counter()
        pipeline.fit(
            temporal_train[FEATURE_COLUMNS],
            temporal_train[TARGET_COLUMN].astype(float),
        )
        fit_seconds = time.perf_counter() - started

        truth = temporal_test[TARGET_COLUMN].to_numpy(dtype=float)
        predictions = np.asarray(
            pipeline.predict(temporal_test[FEATURE_COLUMNS]),
            dtype=float,
        ).reshape(-1)
        metrics = evaluate(truth, predictions)
        metrics["fit_seconds"] = float(fit_seconds)

        # Signed residual quantiles let the app show calibrated prediction
        # intervals.  They are learned on the time-based holdout, which is the
        # more honest evaluation for future/unseen data.
        residuals = truth - predictions
        quantiles = {
            "p05": float(np.percentile(residuals, 5)),
            "p10": float(np.percentile(residuals, 10)),
            "p90": float(np.percentile(residuals, 90)),
            "p95": float(np.percentile(residuals, 95)),
        }

        trained_models[name]["temporal_metrics"] = metrics
        trained_models[name]["error_quantiles"] = quantiles
        temporal_rows.append({"model": name, **metrics, **quantiles})

        print(
            f"  R²={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}  "
            f"RMSE={metrics['rmse']:.2f}  fit={fit_seconds:.1f}s  "
            f"80% interval ±{quantiles['p90']:.1f}/-{abs(quantiles['p10']):.1f}"
        )

    best_temporal_model = max(temporal_rows, key=lambda row: row["r2"])["model"]
    print(f"\nBest temporal-split model: {best_temporal_model}")

    target = y.to_numpy()
    sample_size = min(4000, target.size)
    rng = np.random.default_rng(RANDOM_STATE)
    target_sample = rng.choice(target, size=sample_size, replace=False)

    temporal_split = {
        "cutoff": TEMPORAL_CUTOFF.date().isoformat(),
        "training_rows": int(len(temporal_train)),
        "test_rows": int(len(temporal_test)),
        "training_start": temporal_train["Date"].min().date().isoformat(),
        "training_end": temporal_train["Date"].max().date().isoformat(),
        "test_start": temporal_test["Date"].min().date().isoformat(),
        "test_end": temporal_test["Date"].max().date().isoformat(),
    }

    bundle = {
        "format_version": 1,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "data_file": DEFAULT_DATA_PATH.name,
        "data_sha256": file_sha256(DEFAULT_DATA_PATH),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "excluded_empty_targets": excluded_empty_targets,
        "excluded_featureless_targets": excluded_featureless_targets,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": [CATEGORICAL_FEATURE],
        "numeric_features": POLLUTANT_ORDER,
        "preprocessing": {
            "numeric_imputation": "global_median",
            "missing_indicators": True,
        },
        "target_column": TARGET_COLUMN,
        "cities": sorted(frame[CATEGORICAL_FEATURE].dropna().unique().tolist()),
        "input_ranges": pollutant_ranges(frame),
        "target_stats": {
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
            "median": float(np.median(target)),
            "std": float(target.std()),
            "p90": float(np.percentile(target, 90)),
            "p95": float(np.percentile(target, 95)),
            "p99": float(np.percentile(target, 99)),
        },
        "target_sample": target_sample.tolist(),
        "metrics": summary_rows,
        "temporal_metrics": temporal_rows,
        "temporal_split": temporal_split,
        "best_model": best_model,
        "best_model_random_split": best_model,
        "best_model_temporal_split": best_temporal_model,
        "models": trained_models,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=5)
    print(f"Wrote {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1024 / 1024:.1f} MiB)")

    metadata = {
        "format_version": bundle["format_version"],
        "trained_at": bundle["trained_at"],
        "sklearn_version": bundle["sklearn_version"],
        "pandas_version": bundle["pandas_version"],
        "numpy_version": bundle["numpy_version"],
        "data_file": bundle["data_file"],
        "data_sha256": bundle["data_sha256"],
        "training_rows": bundle["training_rows"],
        "test_rows": bundle["test_rows"],
        "excluded_empty_targets": bundle["excluded_empty_targets"],
        "excluded_featureless_targets": bundle["excluded_featureless_targets"],
        "feature_columns": bundle["feature_columns"],
        "preprocessing": bundle["preprocessing"],
        "cities": bundle["cities"],
        "input_ranges": bundle["input_ranges"],
        "target_stats": bundle["target_stats"],
        "metrics": bundle["metrics"],
        "temporal_metrics": bundle["temporal_metrics"],
        "temporal_split": bundle["temporal_split"],
        "best_model": bundle["best_model"],
        "best_model_random_split": bundle["best_model_random_split"],
        "best_model_temporal_split": bundle["best_model_temporal_split"],
        "error_quantiles": {
            name: payload.get("error_quantiles", {})
            for name, payload in trained_models.items()
        },
        "model_descriptions": {
            name: payload["description"]
            for name, payload in trained_models.items()
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {METADATA_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
