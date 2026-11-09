# AirWatch India — Air Quality Dashboard & AQI Estimator

A Streamlit application for exploring historical air quality in India, checking
live conditions and forecasts, and estimating AQI from pollutant measurements
with transparent, reproducible machine-learning models.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.60%2B-ff4b4b)
![Tests](https://img.shields.io/badge/tests-pytest-green)

## Features

- **Historical dashboard** — filter by date and city, compare annual AQI,
  inspect category distributions, view pollutant trends grouped by unit, see a
  city map, and download the filtered data as CSV.
- **Live AQI page** — computes the **Indian CPCB AQI** from live pollutant
  concentrations (dominant pollutant, sub-index table, category, and hourly
  forecast), and shows both a CPCB forecast and an **ML forecast** for the next
  five days. US/European indices appear in a reference expander. Uses the free
  Open-Meteo API (no API key required) with retry/backoff.
- **ML prediction page** — six regression models, optional pollutant inputs
  (blank values are imputed from training medians), time-based and random-split
  metrics, an **80% prediction interval**, a model-comparison table, and health
  guidance.
- **Reproducible model bundle** — `Models/aqi_models.joblib` stores pipelines,
  feature order, input ranges, target statistics, evaluation metrics,
  prediction-interval residual quantiles, the scikit-learn version, and a
  SHA-256 hash of the training data.
- **Graceful fallback** — if the live API is unreachable, the Live AQI page
  shows recent historical observations instead.

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run Air.py
```

The app opens at <http://localhost:8501>.

## Project structure

```
Air.py                       # Entry point and navigation
aqi_app/
  constants.py               # Pollutants, units, AQI bands, city coordinates
  data.py                    # Dataset loading, validation, filters, map data
  cleaning.py                # Reproducible data-quality corrections
  preprocessing.py           # Optional city-aware median imputation
  india_aqi.py               # CPCB sub-indices and Indian AQI calculation
  live.py                    # Open-Meteo geocoding, live AQI, forecast parsing
  models.py                  # Model bundle loading, prediction, metrics
pages/
  home.py                    # Overview and health information
  dashboard.py               # Historical analytics and CSV export
  live.py                    # Live conditions and 5-day forecast
  prediction.py              # ML-based AQI estimation
scripts/
  clean_data.py              # Rebuild the cleaned CSV from the raw backup
  feature_ablation.py        # Compare feature sets and imputation strategies
  label_diagnostics.py       # Compare reported AQI with a CPCB recomputation
  train_models.py            # Reproducible training/evaluation entry point
Models/
  aqi_models.joblib          # Trained pipelines + metadata (generated)
  metadata.json              # Human-readable training summary (generated)
  feature_ablation.json      # Feature/imputation experiment results (generated)
Data/
  city_day_raw.csv           # Original Kaggle download (unchanged)
  city_day.csv               # Cleaned dataset used by the app
  cleaning_report.json       # Counts of every cleaning correction
  aqi_label_diagnostics.json # Reported-vs-recomputed AQI diagnostics
tests/                       # Unit and Streamlit AppTest coverage
.github/workflows/ci.yml     # Lint + test on every push/PR
```

## Data

| Source | Coverage | Notes |
|---|---|---|
| [Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) | 2015-01-01 → 2020-07-01 | Raw: 29,531 rows · cleaned: 27,462 rows, 26 cities, 12 pollutants |
| [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) | Live + 5-day forecast | CC BY 4.0; no API key |

The cleaned file contains **2,612 rows with an empty AQI target**. The
training script excludes those rows instead of filling AQI with its mean, so
the model is fitted only on real reported targets. It also excludes **52 rows**
whose AQI is present but whose pollutant columns are all missing, since those
rows carry no features. Missing *pollutant* values are imputed inside each
scikit-learn pipeline using training-split medians.

### Data cleaning

The original Kaggle file has several documented quality problems, so
`Data/city_day.csv` is generated from the untouched
`Data/city_day_raw.csv` backup:

```bash
python scripts/clean_data.py
```

Corrections in `aqi_app/cleaning.py`:

- AQI is capped at the CPCB maximum of **500**, and `AQI_Bucket` is recomputed.
- **Ahmedabad's CO column is blanked** because it duplicates NO in different
  units (mg/m³ vs µg/m³).
- Exact zeros in CO, Benzene, Toluene, and Xylene are treated as
  below-detection placeholders and blanked.
- NOx is blanked when it is lower than NO or NO2.
- PM10 is blanked when it is lower than PM2.5.
- Negative values are blanked, and rows with no pollutants and no AQI are
  dropped.

`Data/cleaning_report.json` records how many values each rule affected.

### AQI label diagnostics

The reported AQI column does not exactly match a CPCB AQI recomputed from the
pollutant columns: only **2.1%** of rows agree exactly, **37.5%** agree within
10 points, and **86%** agree within 50 points. The median absolute difference
is **15 points** and the correlation is **0.84**. This means the target is
noisy relative to the features; the app keeps the reported AQI as the training
label and documents the mismatch instead of silently replacing it.

```bash
python scripts/label_diagnostics.py
```

Writes `Data/aqi_label_diagnostics.json` with overall and per-city diagnostics.

### Indian AQI calculation

The Live AQI page does not display Open-Meteo's US AQI as the main value.
Instead, `aqi_app/india_aqi.py` reconstructs the **CPCB AQI** from the live
pollutant concentrations using the official sub-index breakpoints (linear
interpolation between breakpoints, capped at 500). The reported AQI is the
maximum sub-index, and at least three pollutants are required. Open-Meteo
publishes PM2.5, PM10, NO2, SO2, CO, O3, and NH3, but not Pb, so the
calculation uses those seven pollutants. US and European indices remain
available only as a labelled reference.

## Model performance

The model is fitted on **24,798 usable rows**: rows with a real AQI target and
at least one pollutant measurement. This excludes 2,612 rows with no AQI
target and 52 rows whose AQI is present but whose pollutant columns are all
missing.

Two evaluations are reported. The **random split** (25% held out,
`random_state=42`) is comparable to the previous version but can put the same
city and nearby dates in both sets. The **time-based split** trains on data
before 2019-07-01 (16,605 rows) and tests on later data (8,193 rows), which is
a more honest estimate for unseen periods.

| Model | Random R² | Random MAE | Time-based R² | Time-based MAE |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.9308 | 18.21 | 0.9118 | 17.21 |
| Random Forest | 0.9299 | 18.01 | 0.8989 | 17.29 |
| Polynomial Regression | 0.9044 | 22.21 | −1.0119 | 55.50 |
| SVR | 0.8852 | 27.83 | 0.7015 | 36.36 |
| Decision Tree | 0.8692 | 24.43 | 0.7721 | 25.20 |
| Multiple Regression | 0.8163 | 30.52 | 0.1471 | 52.41 |

Gradient Boosting remains best on both evaluations, and Random Forest is close
on the time-based split. The linear and SVR models fit the random split but
extrapolate poorly to later dates, which is expected for pollutant
distributions that shift over time. The prediction page shows the **time-based
metrics first** (the honest evaluation), with the random-split metrics as a
secondary comparison, and warns when a lower-performing model is selected.
Model predictions are clamped to the CPCB 0–500 display scale.

### Prediction intervals

The training script also stores 5th/10th/90th/95th percentile **signed residual
quantiles** from the time-based holdout for every model. The app turns the 10th
and 90th percentiles into an **80% prediction interval** shown next to each
estimate (and the 5th/95th percentiles are available for a 90% interval).
These intervals quantify model error; they do not cover measurement error or
forecast uncertainty, and they cannot remove the underlying model uncertainty.

### Feature and imputation experiments

`python scripts/feature_ablation.py` writes `Models/feature_ablation.json`.
Results on the time-based split:

- **Feature set:** the full 12 pollutants (R² 0.9118) is effectively tied with
  the best reduced set, CPCB pollutants plus NO (R² 0.9140). A CPCB-only set
  drops to R² 0.8813. The app keeps all 12 features because the difference is
  within noise and the extra measurements are useful context.
- **Missingness indicators:** adding median imputation + missing indicators
  clearly improved every imputing model — Random Forest R² 0.839 → 0.899, SVR
  0.358 → 0.702, Decision Tree 0.659 → 0.772, Polynomial Regression
  −1.400 → −1.012, Multiple Regression 0.134 → 0.147. This configuration is
  now used for all non-Gradient-Boosting models.
- **City medians:** city-level imputation did not beat global medians
  consistently (only Multiple Regression improved marginally), so it is
  available in `aqi_app/preprocessing.py` but not used by the bundle.
  Gradient Boosting continues to receive raw `NaN` values and handles them
  natively.

The ablation is an evaluation script; it does not change the app's feature
order or model bundle.

To rebuild the cleaned dataset and retrain the models:

```bash
pip install -r requirements-dev.txt
python scripts/clean_data.py
python scripts/train_models.py
```

Optional evaluation reports (they do not change the app bundle):

```bash
python scripts/feature_ablation.py    # Models/feature_ablation.json
python scripts/label_diagnostics.py   # Data/aqi_label_diagnostics.json
```

## Tests and quality checks

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The test-suite covers CPCB sub-index breakpoints and interpolation, AQI band
boundaries, dataset validation, filtering, model predictions with complete and
missing inputs, prediction intervals, live-payload parsing, Indian-AQI
computation, API retry behavior, and Streamlit page rendering
(`streamlit.testing.v1.AppTest`).

CI runs the same checks on Python 3.12 via
`.github/workflows/ci.yml`.

## Deployment

The app is ready for Streamlit Community Cloud or any platform that runs a
Streamlit process:

1. Push the repository (including `Models/aqi_models.joblib`).
2. Create a new Streamlit app and select `Air.py` as the entry point.
3. No secrets are required — the Open-Meteo API is keyless.

If the model bundle is too large for your host, run
`python scripts/train_models.py` during deployment and add the output to the
image instead of committing it.

## Known limitations and how the app handles them

Some limitations are inherent to the data sources or to what AQI means; others
are mitigated in code. They are listed here so the app's claims stay honest.

### Inherent trade-offs

- **AQI is derived from the pollutants used as features.** The ML page
  estimates the CPCB calculation rather than measuring AQI independently. The
  Live page now also runs the model over Open-Meteo's hourly forecast, but both
  the estimate and the forecast remain derived values.
- **The live page computes an indicative CPCB AQI, not an official one.** Only
  CPCB/state pollution-control boards publish official readings. The page uses
  CPCB sub-index breakpoints on Open-Meteo concentrations and labels the result
  as indicative.
- **Open-Meteo does not publish Pb**, so the live CPCB calculation uses seven
  of the eight CPCB pollutants. It withholds the AQI when fewer than three
  pollutants are available.
- **Historical coverage is limited** to 26 Indian cities from 2015 to July
  2020. For any other location, the Live page maps the model's city feature to
  the nearest supported city and displays which city was used.
- **The live feed does not publish NO, NOx, Benzene, Toluene, or Xylene.**
  Those pollutants are not part of the CPCB AQI calculation, but the ML model
  imputes training medians for them; the page states how many inputs were
  live versus imputed.
- **US AQI ≠ CPCB AQI.** The live page's primary value is the Indian CPCB
  scale; US and European indices are shown only in a labelled reference
  expander.
- **Exact model uncertainty cannot be removed.** The best time-based MAE is
  about 17 AQI points. The app now shows an 80% prediction interval derived
  from time-based residuals (and stores 5th/95th percentiles for a 90%
  interval). Predictions and intervals are clamped to the CPCB 0–500 scale.
- **The reported AQI is a noisy label.** Only about 2.1% of rows exactly match
  a CPCB AQI recomputed from their pollutant columns; the median absolute
  difference is 15 points and 86% agree within 50 points. The model learns the
  reported label rather than replacing it. Run
  `python scripts/label_diagnostics.py` to regenerate
  `Data/aqi_label_diagnostics.json`.
- **Map coordinates are approximate city centres**, not monitoring-station
  locations.
- **Live data depends on a third-party API.** Requests now retry with
  exponential backoff, but the service can still be unavailable; the app then
  falls back to recent historical observations.

### Mitigations in this version

- **Random split is optimistic:** the prediction page shows time-based metrics
  first and treats them as the honest evaluation.
- **Unknown cities:** the Live page resolves them to the nearest supported
  city instead of silently using an unknown encoder row.
- **Model uncertainty:** every estimate includes an 80% prediction interval.
- **Forecasting:** the Live page shows both a CPCB forecast and an ML forecast
  for the next five days.
- **API fragility:** geocoding and air-quality requests retry before failing.

This project is for education and awareness. It is not an official monitoring
or regulatory tool. Follow CPCB/state pollution-control advisories for
health-critical decisions.

## License and attribution

- Historical data: Kaggle, *Air Quality Data in India* (rohanrao).
- Live data: [Open-Meteo](https://open-meteo.com/), CC BY 4.0.
- Built with [Streamlit](https://streamlit.io/), Pandas, NumPy, and
  scikit-learn.
