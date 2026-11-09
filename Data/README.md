# Data

## Files

| File | Description |
|---|---|
| `city_day_raw.csv` | Original Kaggle download, kept unchanged for reproducibility |
| `city_day.csv` | Cleaned dataset used by the app and training script |
| `cleaning_report.json` | Counts of every correction applied by the cleaning script |
| `aqi_label_diagnostics.json` | Reported-vs-recomputed AQI comparison |

Source: <https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india>

- Coverage: 2015-01-01 → 2020-07-01
- Cleaned rows: 27,462
- Cities: 26
- Target: `AQI` (2,612 rows have an empty target)
- Pollutants: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene,
  Toluene, Xylene

## Cleaning

`Data/city_day.csv` is generated from `Data/city_day_raw.csv` by:

```bash
python scripts/clean_data.py
```

The rules live in `aqi_app/cleaning.py` and are deliberately conservative:
values are blanked rather than invented.

- AQI is capped at the CPCB maximum of 500, and `AQI_Bucket` is recomputed.
- Ahmedabad's `CO` column is blanked because it duplicates `NO` in different
  units (mg/m³ vs µg/m³).
- Exact zeros in `CO`, `Benzene`, `Toluene`, and `Xylene` are treated as
  below-detection placeholders and blanked.
- `NOx` is blanked when it is lower than `NO` or `NO2`.
- `PM10` is blanked when it is lower than `PM2.5`.
- Negative values are blanked.
- Rows with no pollutants and no AQI are dropped.

Missing pollutant values remain missing in the CSV. They are imputed inside
each scikit-learn pipeline using training-split medians, so the model never
sees a value that was not learned from the data. Missing `AQI` targets are
excluded from training rather than filled. An additional 52 rows have an AQI
target but no pollutant measurements at all; they are kept in the CSV for
analysis but excluded from model training.

See `cleaning_report.json` for the number of affected values.

## Diagnostics

`aqi_label_diagnostics.json` compares the reported AQI target with a CPCB AQI
recomputed from PM2.5, PM10, NO2, SO2, CO, O3, and NH3. Generate it with:

```bash
python scripts/label_diagnostics.py
```

Only about 2.1% of rows match exactly, the median absolute difference is 15
AQI points, and 86% agree within 50 points. The reported AQI is kept as the
training label; the diagnostic exists so the label noise is visible rather
than silently replaced.
