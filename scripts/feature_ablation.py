"""Compare feature sets and missing-value handling.

Run from the project root:

    python scripts/feature_ablation.py

Writes ``Models/feature_ablation.json``.  This is an evaluation script: it does
not change the model bundle or the app's feature order.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aqi_app.constants import CATEGORICAL_FEATURE, POLLUTANT_ORDER, TARGET_COLUMN  # noqa: E402
from aqi_app.data import load_dataset, training_frame  # noqa: E402
from scripts.train_models import (  # noqa: E402
    RANDOM_STATE,
    TEMPORAL_CUTOFF,
    TEST_SIZE,
    build_models,
    evaluate,
)

OUTPUT_PATH = PROJECT_ROOT / "Models" / "feature_ablation.json"

# Feature sets are evaluated with Gradient Boosting, which handles NaN natively.
FEATURE_SETS: dict[str, list[str]] = {
    "full_12": list(POLLUTANT_ORDER),
    "cpcb_7": ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"],
    "cpcb_plus_no": ["PM2.5", "PM10", "NO", "NO2", "SO2", "CO", "O3", "NH3"],
    "no_high_missing": [
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NH3",
        "CO",
        "SO2",
        "O3",
        "Benzene",
    ],
}

# Models that currently impute with medians are tested with different
# imputation strategies: global median, global median + missing indicators,
# and city median.
IMPUTATION_MODELS = [
    "Random Forest",
    "SVR",
    "Polynomial Regression",
    "Multiple Regression",
    "Decision Tree",
]

PREPROCESSING_VARIANTS: dict[str, dict[str, object]] = {
    "global_median": {"imputation": "global", "add_indicator": False},
    "global_median_indicators": {"imputation": "global", "add_indicator": True},
    "city_median": {"imputation": "city", "add_indicator": False},
}


def fit_score(
    pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    columns = [CATEGORICAL_FEATURE, *features]
    started = time.perf_counter()
    pipeline.fit(train[columns], train[TARGET_COLUMN].astype(float))
    fit_seconds = time.perf_counter() - started
    predictions = np.asarray(pipeline.predict(test[columns]), dtype=float).reshape(-1)
    metrics = evaluate(test[TARGET_COLUMN].to_numpy(dtype=float), predictions)
    metrics["fit_seconds"] = float(fit_seconds)
    return metrics


def main() -> int:
    frame = training_frame(load_dataset())
    random_train, random_test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    temporal_train = frame[frame["Date"] < TEMPORAL_CUTOFF]
    temporal_test = frame[frame["Date"] >= TEMPORAL_CUTOFF]

    print(f"Random split: {len(random_train):,} / {len(random_test):,}")
    print(
        f"Temporal split ({TEMPORAL_CUTOFF.date()}): "
        f"{len(temporal_train):,} / {len(temporal_test):,}"
    )

    feature_results: list[dict] = []
    for label, features in FEATURE_SETS.items():
        print(f"\nGradient Boosting · {label} ({len(features)} features)")
        pipeline = build_models(features)["Gradient Boosting"][0]
        random_metrics = fit_score(pipeline, random_train, random_test, features)

        pipeline = build_models(features)["Gradient Boosting"][0]
        temporal_metrics = fit_score(pipeline, temporal_train, temporal_test, features)

        feature_results.append(
            {
                "feature_set": label,
                "features": features,
                "random": random_metrics,
                "temporal": temporal_metrics,
            }
        )
        print(
            f"  random R²={random_metrics['r2']:.4f}  "
            f"temporal R²={temporal_metrics['r2']:.4f}  "
            f"temporal MAE={temporal_metrics['mae']:.2f}"
        )

    preprocessing_results: list[dict] = []
    for name in IMPUTATION_MODELS:
        for label, options in PREPROCESSING_VARIANTS.items():
            print(f"\n{name} · {label}")
            pipeline = build_models(POLLUTANT_ORDER, **options)[name][0]
            metrics = fit_score(pipeline, temporal_train, temporal_test, POLLUTANT_ORDER)
            preprocessing_results.append(
                {
                    "model": name,
                    "variant": label,
                    **options,
                    "temporal": metrics,
                }
            )
            print(
                f"  temporal R²={metrics['r2']:.4f}  "
                f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}"
            )

    best_features = max(feature_results, key=lambda row: row["temporal"]["r2"])
    preprocessing_winners = {}
    for name in IMPUTATION_MODELS:
        rows = [row for row in preprocessing_results if row["model"] == name]
        winner = max(rows, key=lambda row: row["temporal"]["r2"])
        preprocessing_winners[name] = winner["variant"]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "temporal_cutoff": TEMPORAL_CUTOFF.date().isoformat(),
        "random_state": RANDOM_STATE,
        "feature_sets": feature_results,
        "best_feature_set_by_temporal_r2": best_features["feature_set"],
        "preprocessing_experiment": preprocessing_results,
        "preprocessing_winners": preprocessing_winners,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nBest feature set: {best_features['feature_set']}")
    for name, variant in preprocessing_winners.items():
        print(f"Best preprocessing for {name}: {variant}")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
