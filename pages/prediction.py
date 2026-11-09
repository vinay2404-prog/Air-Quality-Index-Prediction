import pandas as pd
import streamlit as st

from aqi_app.constants import (
    AQI_BANDS,
    AQI_MAX_DISPLAY,
    MAIN_POLLUTANTS,
    OPTIONAL_POLLUTANTS,
    POLLUTANT_ORDER,
    aqi_category,
    aqi_health_message,
    aqi_progress,
    pollutant_label,
)
from aqi_app.models import (
    ModelError,
    best_model_name,
    clip_prediction,
    dataset_aqi_percentile,
    input_ranges,
    load_bundle,
    metrics_frame,
    model_names,
    predict,
    prediction_interval,
    target_stats,
    temporal_metrics_frame,
    temporal_split_info,
    trained_at,
)
from aqi_app.models import (
    cities as bundle_cities,
)


@st.cache_resource(show_spinner=False)
def cached_bundle():
    return load_bundle()


try:
    bundle = cached_bundle()
except ModelError as exc:
    st.error("Unable to load the prediction models.")
    st.code(str(exc))
    st.info("Run `python scripts/train_models.py` to rebuild the model bundle.")
    st.stop()

cities = bundle_cities(bundle)
if not cities:
    st.error("The model bundle does not contain any city categories.")
    st.stop()

ranges = input_ranges(bundle)
models = model_names(bundle)
best_model = best_model_name(bundle)
metrics = metrics_frame(bundle)
temporal_metrics = temporal_metrics_frame(bundle)
temporal_split = temporal_split_info(bundle)
temporal_cutoff = temporal_split.get("cutoff", "the holdout date")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Know the air. Breathe with awareness.")
st.caption("Machine learning · AQI estimation · transparent test metrics")
st.write(
    """
    Estimate air quality from pollutant measurements using machine learning.
    Every pollutant is optional: leave a field blank if it was not measured
    and the model will use the training median for that value.
    """
)
st.caption(
    "AQI is calculated from the same pollutants used as model inputs, so this "
    "page is an approximation of the CPCB calculation, not a future forecast. "
    "Use the Live AQI page for current conditions and forecasts."
)
st.divider()

# ---------------------------------------------------------------------------
# Prefill from the live page
# ---------------------------------------------------------------------------
prefill = st.session_state.pop("prediction_prefill", None)
if prefill:
    if prefill.get("city") in cities:
        st.session_state["predicted_city"] = prefill["city"]
    # Clear every field first so values from an earlier session or a previous
    # prefill cannot leak into this one.
    for pollutant in POLLUTANT_ORDER:
        st.session_state[f"pollutant_{pollutant}"] = None
    for name, value in (prefill.get("values") or {}).items():
        if name in POLLUTANT_ORDER and value is not None:
            st.session_state[f"pollutant_{name}"] = float(value)
    st.info("Live values were copied from the Live AQI page. Review them before predicting.")

if "predicted_city" not in st.session_state:
    st.session_state["predicted_city"] = "Delhi" if "Delhi" in cities else cities[0]

for pollutant in POLLUTANT_ORDER:
    st.session_state.setdefault(f"pollutant_{pollutant}", None)

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
st.header("Prediction workspace")
st.write("Choose a city, select a model, and enter any pollutant measurements you have.")

model_col, info_col = st.columns([1.4, 1], gap="large")

with model_col:
    selected_model = st.selectbox(
        "Machine-learning model",
        models,
        index=models.index(best_model),
        key="selected_model",
    )
    st.caption(bundle["models"][selected_model]["description"])

with info_col:
    temporal_row = temporal_metrics[temporal_metrics["Model"] == selected_model].iloc[0]
    a, b, c = st.columns(3)
    with a:
        st.metric("Time-based R²", f"{temporal_row['R²']:.3f}")
    with b:
        st.metric("MAE", f"{temporal_row['MAE']:.1f}")
    with c:
        st.metric("RMSE", f"{temporal_row['RMSE']:.1f}")

    selected_row = metrics[metrics["Model"] == selected_model].iloc[0]
    st.caption(
        f"Time-based holdout (trained before {temporal_cutoff}). "
        f"Random split (optimistic): R² {selected_row['R²']:.3f} · "
        f"MAE {selected_row['MAE']:.1f} · RMSE {selected_row['RMSE']:.1f}."
    )

if selected_model != best_model:
    st.info(
        f"**{best_model}** has the best time-based R² in this bundle. "
        "The comparison table below shows every model on both evaluations."
    )

with st.expander("Compare all six models"):
    temporal_tab, random_tab = st.tabs(
        [f"Time-based split (before {temporal_cutoff})", "Random split (optimistic)"]
    )
    format_spec = {
        "R²": "{:.4f}",
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "Fit time (s)": "{:.1f}",
    }
    with temporal_tab:
        st.dataframe(
            temporal_metrics.style.format(format_spec),
            width="stretch",
            hide_index=True,
        )
    with random_tab:
        st.dataframe(
            metrics.style.format(format_spec),
            width="stretch",
            hide_index=True,
        )

    stats = target_stats(bundle)
    st.caption(
        f"Time-based split: {temporal_split.get('training_rows', 0):,} training rows "
        f"and {temporal_split.get('test_rows', 0):,} later rows "
        f"({temporal_split.get('test_start', '?')} → "
        f"{temporal_split.get('test_end', '?')}). "
        f"Random split: {bundle['test_rows']:,} held-out rows, which can include "
        "the same city and nearby dates as the training set. "
        f"Training target range: {stats.get('min', 0):.0f}–"
        f"{stats.get('max', 0):.0f} AQI. Bundle trained {trained_at(bundle)} "
        f"with scikit-learn {bundle.get('sklearn_version', 'unknown')}."
    )

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.header("Enter pollutant measurements")

action_col, clear_col, _ = st.columns([1.2, 1, 2])
with action_col:
    if st.button("Fill with training medians", width="stretch"):
        for pollutant in POLLUTANT_ORDER:
            st.session_state[f"pollutant_{pollutant}"] = round(
                float(ranges.get(pollutant, {}).get("median", 0.0)), 2
            )
        st.rerun()

with clear_col:
    if st.button("Clear all fields", width="stretch"):
        for pollutant in POLLUTANT_ORDER:
            st.session_state[f"pollutant_{pollutant}"] = None
        st.rerun()

st.caption(
    "Maximum values shown below are the observed maxima in the training data, "
    "so extreme episodes are not artificially blocked."
)

with st.form("aqi_prediction_form"):
    left, right = st.columns([1.15, 0.85], gap="large")
    values: dict[str, float | None] = {}

    with left:
        st.markdown("#### Location")
        st.selectbox("Select city", cities, key="predicted_city")
        st.caption(
            "The model has learned city features for these 26 cities. For any "
            "other location, use the Live AQI page, which maps to the nearest "
            "supported city."
        )

        st.divider()
        st.markdown("#### Primary pollutants")

        primary_columns = st.columns(2)
        for index, name in enumerate(MAIN_POLLUTANTS):
            observed = ranges.get(name, {})
            max_value = max(float(observed.get("max", 1000.0)), 1.0)
            with primary_columns[index]:
                values[name] = st.number_input(
                    pollutant_label(name),
                    min_value=0.0,
                    max_value=max_value,
                    value=None,
                    step=0.5,
                    key=f"pollutant_{name}",
                    placeholder=f"Not measured · max {max_value:g}",
                )

    with right:
        st.markdown("#### Other pollutants")
        st.caption("Leave blank when a pollutant was not measured.")
        with st.expander("Configure the other 10 measurements", expanded=False):
            for name in OPTIONAL_POLLUTANTS:
                observed = ranges.get(name, {})
                max_value = max(float(observed.get("max", 1000.0)), 1.0)
                values[name] = st.number_input(
                    pollutant_label(name),
                    min_value=0.0,
                    max_value=max_value,
                    value=None,
                    step=0.1,
                    key=f"pollutant_{name}",
                    placeholder=f"Not measured · max {max_value:g}",
                )

    submitted = st.form_submit_button(
        "Estimate AQI",
        type="primary",
        width="stretch",
    )

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
if submitted:
    provided = {name: value for name, value in values.items() if value is not None}
    if not provided:
        st.error("Enter at least one pollutant measurement before estimating AQI.")
        st.session_state.pop("prediction_result", None)
    else:
        try:
            with st.spinner(f"Estimating with {selected_model}..."):
                raw = predict(
                    bundle,
                    selected_model,
                    st.session_state["predicted_city"],
                    values,
                )
            clipped, was_clipped = clip_prediction(raw, upper=AQI_MAX_DISPLAY)
            st.session_state["prediction_result"] = {
                "aqi": clipped,
                "raw": raw,
                "clipped": was_clipped,
                "city": st.session_state["predicted_city"],
                "model": selected_model,
                "provided": provided,
                "missing": [p for p in POLLUTANT_ORDER if p not in provided],
            }
        except ModelError as exc:
            st.session_state.pop("prediction_result", None)
            st.error(f"Prediction failed using {selected_model}.")
            st.code(str(exc))

st.divider()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
st.header("Prediction result")
result = st.session_state.get("prediction_result")

if result is None:
    st.info(
        "Enter your measurements and click **Estimate AQI** to see a result. "
        "Blank pollutants are imputed from training medians."
    )
else:
    aqi = float(result["aqi"])
    category = aqi_category(aqi)
    percentile = dataset_aqi_percentile(bundle, aqi)

    result_col, guidance_col = st.columns([1, 1], gap="large")

    with result_col:
        st.subheader("Estimated air quality")
        st.metric("Estimated CPCB AQI", f"{aqi:.1f}")
        st.progress(
            aqi_progress(aqi),
            text=f"{category} · {aqi:.1f} on a 0–500 display scale",
        )

        if category in {"Good", "Satisfactory"}:
            st.success(f"AQI category: {category}")
        elif category == "Moderate":
            st.warning(f"AQI category: {category}")
        else:
            st.error(f"AQI category: {category}")

        interval = prediction_interval(bundle, result["model"], aqi, level=0.8)
        if interval is not None:
            low, high = interval
            st.caption(
                f"Approx. 80% prediction interval: **{low:.0f}–{high:.0f} AQI**, "
                "derived from time-based test residuals. This covers model error, "
                "not measurement uncertainty."
            )

        st.caption(f"City: {result['city']} · Model: {result['model']}")
        if percentile == percentile:  # not NaN
            st.caption(
                f"This estimate is higher than about {percentile:.0f}% of the "
                "historical AQI values in the training data."
            )

    with guidance_col:
        st.subheader("Health guidance")
        st.write(aqi_health_message(aqi))
        st.caption(
            "General educational guidance only. Follow official local "
            "air-quality and public-health advisories."
        )

    if result["clipped"]:
        st.warning(
            f"The raw model output was {result['raw']:.1f}, outside the CPCB "
            f"0–{AQI_MAX_DISPLAY} AQI scale, so the displayed estimate is "
            f"clamped to {aqi:.1f}."
        )

    if result["missing"]:
        st.info(
            f"{len(result['missing'])} of {len(POLLUTANT_ORDER)} pollutant inputs "
            "were left blank and imputed with training medians: "
            + ", ".join(result["missing"])
            + "."
        )

    with st.expander("Inputs used for this prediction"):
        rows = [
            {
                "Pollutant": pollutant_label(name),
                "Value": (
                    f"{result['provided'][name]:,.2f}"
                    if name in result["provided"]
                    else "Not provided (median imputed)"
                ),
            }
            for name in POLLUTANT_ORDER
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()
    st.subheader("AQI reference scale")
    scale = pd.DataFrame(
        [
            {
                "Range": (
                    f"{band['min']}–{band['max']}"
                    if band["max"] != float("inf")
                    else f"{band['min']}+"
                ),
                "Category": band["label"],
                "This estimate": "←" if band["label"] == category else "",
            }
            for band in AQI_BANDS
        ]
    )
    st.dataframe(scale, width="stretch", hide_index=True)
    st.caption(
        "The AQI scale here is the Indian CPCB scale used by the historical "
        "dataset. The model reproduces that calculation from pollutant inputs; "
        "it is not an official monitoring-station reading."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.subheader("AirWatch India")
st.caption(
    "Machine-learning-based AQI estimation for educational use. "
    "This app does not provide official AQI readings. Results depend on the "
    "training data, model quality, and the measurements you supply."
)
