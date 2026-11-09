import pandas as pd
import streamlit as st

from aqi_app.constants import (
    AQI_MAX_DISPLAY,
    CITY_COORDINATES,
    POLLUTANT_ORDER,
    POLLUTANTS,
    aqi_health_message,
    aqi_progress,
    nearest_supported_city,
    pollutant_label,
    us_aqi_category,
)
from aqi_app.data import DataError, load_dataset
from aqi_app.india_aqi import CPCB_MIN_POLLUTANTS
from aqi_app.live import (
    ATTRIBUTION,
    LiveDataError,
    current_snapshot,
    daily_forecast,
    fetch_air_quality,
    geocode_city,
    hourly_feature_frame,
    hourly_frame,
    hourly_indian_aqi,
    model_features,
)
from aqi_app.models import (
    ModelError,
    best_temporal_model_name,
    clip_prediction,
    load_bundle,
    predict,
    predict_frame,
    prediction_interval,
)
from aqi_app.models import (
    cities as bundle_cities,
)

st.title("📡 Live Air Quality — Indian CPCB AQI")
st.caption("Current conditions and a 5-day forecast · Open-Meteo Air Quality API")

st.write(
    """
    Fetch live pollutant concentrations for a city and compute the **Indian
    CPCB AQI** from them. The page shows the AQI category, the dominant
    pollutant, every sub-index, and an hourly CPCB AQI forecast. It also runs
    the app's best ML model on the same pollutants, both now and across the
    five-day forecast, so you can compare the two approaches.
    """
)

st.caption(
    f"CPCB AQI is the maximum of the pollutant sub-indices and needs at least "
    f"{CPCB_MIN_POLLUTANTS} pollutants. Open-Meteo does not publish Pb, so the "
    "calculation uses the 7 CPCB pollutants available from the live feed. "
    "This is an indicative calculation, not an official CPCB reading."
)

st.divider()

# ---------------------------------------------------------------------------
# Location selection
# ---------------------------------------------------------------------------
mode = st.radio(
    "Location",
    ["Supported city", "Search any city"],
    horizontal=True,
)

latitude: float | None = None
longitude: float | None = None
location_name = ""

if mode == "Supported city":
    supported = sorted(CITY_COORDINATES)
    selected_city = st.selectbox("Select a city", supported, index=supported.index("Delhi"))
    latitude, longitude = CITY_COORDINATES[selected_city]
    location_name = selected_city
else:
    query = st.text_input("City name", value="Delhi", placeholder="e.g. Pune, London, Tokyo")
    if st.button("Search", type="primary"):
        try:
            st.session_state["geocode_results"] = geocode_city(query)
        except LiveDataError as exc:
            st.error(str(exc))

    results = st.session_state.get("geocode_results", [])
    if results:
        match = st.selectbox(
            "Matches",
            results,
            format_func=lambda item: item.display_name,
        )
        latitude, longitude = match.latitude, match.longitude
        location_name = match.name
    else:
        st.info("Search for a city to load live data.")

if latitude is None or longitude is None:
    st.stop()


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def cached_air_quality(lat: float, lon: float, days: int):
    return fetch_air_quality(lat, lon, forecast_days=days)


@st.cache_resource(show_spinner=False)
def cached_bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return load_dataset()


try:
    payload = cached_air_quality(latitude, longitude, 5)
except LiveDataError as exc:
    st.error(str(exc))
    st.info(
        "Live data is unavailable. The historical dataset is still available "
        "on the Dashboard page."
    )
    with st.expander("Show recent historical observations instead"):
        try:
            frame = cached_dataset()
            fallback_city = (
                location_name
                if location_name in frame["City"].unique()
                else nearest_supported_city(latitude, longitude)
            )
            recent = (
                frame[frame["City"] == fallback_city]
                .dropna(subset=["AQI"])
                .sort_values("Date")
                .tail(90)
                .set_index("Date")[["AQI", "PM2.5", "PM10"]]
            )
            st.caption(f"Last 90 reported observations for {fallback_city}.")
            st.line_chart(recent)
        except DataError as data_exc:
            st.warning(str(data_exc))
    st.stop()

snapshot = current_snapshot(payload)
indian = snapshot["indian_aqi"]

# ---------------------------------------------------------------------------
# Current conditions
# ---------------------------------------------------------------------------
st.header(f"Current conditions — {location_name}")

if snapshot.get("time"):
    st.caption(
        f"Observed {snapshot['time']} "
        f"({snapshot.get('timezone_abbreviation') or snapshot.get('timezone') or 'local time'})"
    )

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Indian CPCB AQI",
        f"{indian.aqi:.0f}" if indian.valid else "—",
    )
    if indian.valid:
        st.caption(f"CPCB category: {indian.category}")

with c2:
    dominant = (
        pollutant_label(indian.dominant_pollutant)
        if indian.dominant_pollutant
        else "—"
    )
    st.metric("Dominant pollutant", dominant)

with c3:
    pm25 = snapshot["pollutants"].get("PM2.5")
    st.metric("PM2.5", f"{pm25:.1f} µg/m³" if pm25 is not None else "—")

with c4:
    pm10 = snapshot["pollutants"].get("PM10")
    st.metric("PM10", f"{pm10:.1f} µg/m³" if pm10 is not None else "—")

if indian.valid:
    st.progress(
        aqi_progress(indian.aqi),
        text=f"{indian.category} · CPCB AQI {indian.aqi:.0f}",
    )
    st.write(aqi_health_message(indian.aqi))
else:
    st.warning(indian.message or "The CPCB AQI could not be calculated.")

st.subheader("Pollutant concentrations and CPCB sub-indices")
pollutant_rows = []
for pollutant in POLLUTANTS:
    value = snapshot["pollutants"].get(pollutant)
    sub_index = indian.sub_indices.get(pollutant)
    pollutant_rows.append(
        {
            "Pollutant": pollutant_label(pollutant),
            "Live value": f"{value:,.2f}" if value is not None else "Not provided",
            "CPCB sub-index": f"{sub_index:.0f}" if sub_index is not None else "—",
        }
    )
st.dataframe(pd.DataFrame(pollutant_rows), width="stretch", hide_index=True)

st.caption(
    "CPCB sub-indices are calculated for PM2.5, PM10, NO2, SO2, CO, O3, and NH3. "
    "NO, NOx, Benzene, Toluene, and Xylene are not part of the CPCB AQI "
    "calculation, so they are shown for information only."
)

with st.expander("International indices (reference only)"):
    us_col, eu_col = st.columns(2)
    with us_col:
        st.metric(
            "US AQI",
            f"{snapshot['us_aqi']:.0f}" if snapshot.get("us_aqi") is not None else "—",
        )
        if snapshot.get("us_aqi") is not None:
            st.caption(f"US EPA category: {us_aqi_category(snapshot['us_aqi'])}")
    with eu_col:
        st.metric(
            "European AQI",
            f"{snapshot['european_aqi']:.0f}"
            if snapshot.get("european_aqi") is not None
            else "—",
        )
    st.caption(
        "US and European indices use different breakpoints from the Indian CPCB "
        "scale. They are shown only for reference and are not used by the rest "
        "of the app."
    )

st.divider()

# ---------------------------------------------------------------------------
# Live model estimate
# ---------------------------------------------------------------------------
st.header("Computed CPCB AQI vs ML estimate")

bundle = None
try:
    bundle = cached_bundle()
    supported = set(bundle_cities(bundle))
    model_city = (
        location_name
        if location_name in supported
        else nearest_supported_city(latitude, longitude)
    )
    features = model_features(snapshot)
    available = sum(value is not None for value in features.values())

    if available == 0:
        st.warning("No modelled pollutants were returned, so an ML estimate is not possible.")
    else:
        model_name = best_temporal_model_name(bundle)
        estimate = predict(bundle, model_name, model_city, features)
        estimate, clipped = clip_prediction(estimate, upper=AQI_MAX_DISPLAY)

        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric(
                "Computed CPCB AQI",
                f"{indian.aqi:.0f}" if indian.valid else "—",
            )
            if indian.valid:
                st.caption(f"Category: {indian.category}")
        with e2:
            st.metric("ML estimate", f"{estimate:.1f}")
            st.caption(f"Model: {model_name}")
        with e3:
            if indian.valid:
                difference = estimate - indian.aqi
                st.metric("ML − CPCB", f"{difference:+.1f}")
            else:
                st.metric("ML − CPCB", "—")

        if model_city != location_name:
            st.caption(f"Nearest supported city for the model's city feature: {model_city}")
        if clipped:
            st.warning(
                f"The raw model output was outside the CPCB 0–{AQI_MAX_DISPLAY} "
                f"AQI scale and has been clamped to {estimate:.1f}."
            )

        interval = prediction_interval(bundle, model_name, estimate, level=0.8)
        if interval is not None:
            low, high = interval
            st.caption(
                f"Approx. 80% prediction interval: {low:.0f}–{high:.0f} AQI, "
                "based on time-based test residuals. This covers model error, "
                "not measurement or forecast uncertainty."
            )

        st.caption(
            f"{available} of {len(features)} model inputs came from the live feed; "
            "the rest were imputed from training medians. The ML estimate is "
            "expected to differ slightly from the direct CPCB calculation."
        )

        if st.button("Prefill the prediction page with these values"):
            st.session_state["prediction_prefill"] = {
                "city": model_city,
                "values": features,
            }
            st.switch_page("pages/prediction.py")
except ModelError as exc:
    st.warning(f"Model estimate unavailable: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
st.header("5-day hourly forecast")

try:
    hourly = hourly_frame(payload)
    hourly = hourly.assign(india_aqi=hourly_indian_aqi(hourly))

    st.subheader("Indian CPCB AQI forecast")
    if hourly["india_aqi"].notna().any():
        st.line_chart(hourly[["india_aqi"]], height=280)
    else:
        st.warning(
            f"Not enough pollutants were forecast to compute a CPCB AQI "
            f"(needs at least {CPCB_MIN_POLLUTANTS})."
        )

    if bundle is not None:
        try:
            supported = set(bundle_cities(bundle))
            forecast_city = (
                location_name
                if location_name in supported
                else nearest_supported_city(latitude, longitude)
            )
            forecast_model = best_temporal_model_name(bundle)
            feature_frame = hourly_feature_frame(hourly, forecast_city)
            has_inputs = feature_frame[POLLUTANT_ORDER].notna().any(axis=1)

            if has_inputs.any():
                raw_forecast = predict_frame(
                    bundle,
                    forecast_model,
                    feature_frame.loc[has_inputs],
                )
                ml_forecast = pd.Series(
                    [
                        clip_prediction(value, upper=AQI_MAX_DISPLAY)[0]
                        for value in raw_forecast
                    ],
                    index=feature_frame.index[has_inputs.to_numpy()],
                    name="ML estimate",
                )
                comparison = pd.DataFrame(
                    {
                        "CPCB AQI": hourly["india_aqi"],
                        "ML estimate": ml_forecast,
                    }
                ).dropna(how="all")

                st.subheader("ML AQI forecast")
                st.line_chart(comparison, height=280)
                st.caption(
                    f"Hourly forecast from **{forecast_model}** using Open-Meteo "
                    f"pollutants and city feature: {forecast_city}. Pollutants "
                    "the live feed does not publish are imputed by the model."
                )
        except (ModelError, ValueError) as exc:
            st.caption(f"ML forecast unavailable: {exc}")

    pm_columns = [column for column in ["pm2_5", "pm10"] if column in hourly.columns]
    if pm_columns:
        st.subheader("Particulate matter forecast (µg/m³)")
        st.line_chart(hourly[pm_columns], height=280)

    daily = daily_forecast(hourly)
    if not daily.empty:
        st.subheader("Daily outlook")
        st.dataframe(
            daily.style.format(
                {
                    "Peak Indian AQI": "{:.0f}",
                    "Mean PM2.5": "{:.1f}",
                    "Mean PM10": "{:.1f}",
                },
                na_rep="—",
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Daily peak is the highest hourly CPCB AQI. Forecast values are "
            "model output from Open-Meteo, not official CPCB forecasts."
        )
except LiveDataError as exc:
    st.warning(f"Forecast unavailable: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Location map
# ---------------------------------------------------------------------------
st.header("Location")
st.map(
    pd.DataFrame(
        {"lat": [latitude], "lon": [longitude], "name": [location_name]}
    ),
    latitude="lat",
    longitude="lon",
    zoom=9,
    height=300,
)

st.caption(ATTRIBUTION)
st.caption(
    "The Indian AQI on this page is calculated locally from Open-Meteo "
    "pollutant concentrations using CPCB sub-index breakpoints. It is not an "
    "official CPCB/state pollution-control reading. Use official CPCB data for "
    "regulatory decisions."
)
