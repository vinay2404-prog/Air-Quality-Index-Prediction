"""Compare the reported AQI label with a CPCB AQI recomputed from pollutants.

Run from the project root:

    python scripts/label_diagnostics.py

Writes ``Data/aqi_label_diagnostics.json``.  This is a diagnostic only: the
reported AQI is never replaced in the cleaned CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aqi_app.constants import TARGET_COLUMN  # noqa: E402
from aqi_app.data import load_dataset  # noqa: E402
from aqi_app.india_aqi import calculate_indian_aqi  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "Data" / "aqi_label_diagnostics.json"
CPCB_INPUTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]


def recomputed_aqi(row: pd.Series) -> float | None:
    values = {
        name: (None if pd.isna(row.get(name)) else float(row[name]))
        for name in CPCB_INPUTS
    }
    result = calculate_indian_aqi(values)
    return result.aqi if result.valid else None


def summary(frame: pd.DataFrame) -> dict[str, float | int]:
    difference = (frame["_recomputed"] - frame[TARGET_COLUMN]).abs()
    return {
        "rows": int(len(frame)),
        "exact_within_1": int((difference < 1).sum()),
        "within_10": int((difference <= 10).sum()),
        "within_25": int((difference <= 25).sum()),
        "within_50": int((difference <= 50).sum()),
        "over_100": int((difference > 100).sum()),
        "median_abs_difference": float(difference.median()),
        "mean_abs_difference": float(difference.mean()),
        "correlation": float(frame[TARGET_COLUMN].corr(frame["_recomputed"])),
    }


def main() -> int:
    df = load_dataset()
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df["_recomputed"] = df.apply(recomputed_aqi, axis=1)
    comparable = df.dropna(subset=["_recomputed"]).copy()

    overall = summary(comparable)
    overall["share_exact_pct"] = round(
        overall["exact_within_1"] / max(overall["rows"], 1) * 100,
        2,
    )
    overall["share_within_10_pct"] = round(
        overall["within_10"] / max(overall["rows"], 1) * 100,
        2,
    )
    overall["share_within_50_pct"] = round(
        overall["within_50"] / max(overall["rows"], 1) * 100,
        2,
    )

    by_city = []
    for city, group in comparable.groupby("City"):
        city_summary = summary(group)
        city_summary["city"] = city
        by_city.append(city_summary)
    by_city.sort(key=lambda row: row["median_abs_difference"], reverse=True)

    payload = {
        "reported_target": TARGET_COLUMN,
        "recomputed_with": CPCB_INPUTS,
        "rows_with_reported_aqi": int(len(df)),
        "rows_comparable_at_least_3_pollutants": int(len(comparable)),
        "overall": overall,
        "by_city": by_city,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Rows with reported AQI: {len(df):,}")
    print(f"Rows comparable: {len(comparable):,}")
    print(
        f"Exact match: {overall['share_exact_pct']}% | "
        f"within 10: {overall['share_within_10_pct']}% | "
        f"within 50: {overall['share_within_50_pct']}%"
    )
    print(
        f"Median absolute difference: {overall['median_abs_difference']:.1f} | "
        f"correlation: {overall['correlation']:.3f}"
    )
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
