"""Optional preprocessing helpers for models that need explicit imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CityMedianImputer(BaseEstimator, TransformerMixin):
    """Impute numeric columns with city medians, then global medians.

    The input must be a DataFrame containing the city column plus the numeric
    pollutant columns.  City medians are learned on the training split only;
    unseen cities and columns with no training values fall back to the global
    median (and finally to zero if a column is entirely missing).
    """

    def __init__(self, city_column: str = "City") -> None:
        self.city_column = city_column

    def fit(self, X: pd.DataFrame, y: object | None = None) -> CityMedianImputer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CityMedianImputer expects a pandas DataFrame")
        if self.city_column not in X.columns:
            raise ValueError(f"Missing city column: {self.city_column}")

        numeric = X.drop(columns=[self.city_column])
        self.numeric_columns_ = list(numeric.columns)
        self.global_medians_ = numeric.median(numeric_only=True)
        self.city_medians_ = numeric.groupby(X[self.city_column]).median(
            numeric_only=True
        )
        self.fill_values_ = self.global_medians_.fillna(0.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CityMedianImputer expects a pandas DataFrame")

        numeric = X[self.numeric_columns_].copy()
        city = X[self.city_column]
        medians = self.city_medians_.reindex(city).fillna(self.fill_values_)
        medians.index = numeric.index
        numeric = numeric.fillna(medians)
        return numeric.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features: object | None = None) -> np.ndarray:
        return np.asarray(self.numeric_columns_, dtype=object)
