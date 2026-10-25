import html
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

from theme import icon, sub_heading


# =========================================================
# 3. HEADER
# =========================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">{icon("cpu", 13)} Machine Learning · AQI Estimation</div>
        <div class="hero-title">
            Know the air.<br>
            <span class="text-muted">Breathe with awareness.</span>
        </div>
        <div class="hero-description">
            Estimate air quality from pollutant measurements using
            machine learning. Understand the predicted AQI category
            and explore general pollution-related guidance.
        </div>
    </div>
    
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. LOAD MODELS
# =========================================================
APP_DIR = Path(__file__).resolve().parent


def find_models_dir():
    possible_paths = [
        APP_DIR / "Models",
        APP_DIR.parent / "Models",
    ]

    for folder in possible_paths:
        if folder.is_dir():
            return folder

    return APP_DIR / "Models"


MODELS_DIR = find_models_dir()


@st.cache_resource
def load_pickle(path_string):
    path = Path(path_string)

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model file: {path}"
        )

    with path.open("rb") as file:
        header = file.read(100)

        if header.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{path.name} is a Git LFS pointer. "
                "Download the actual model file."
            )

        file.seek(0)
        return pickle.load(file)


try:
    mreg = load_pickle(
        str(MODELS_DIR / "Multiple Regression.pkl")
    )
    preg = load_pickle(
        str(MODELS_DIR / "pregression.pkl")
    )
    dec_tree = load_pickle(
        str(MODELS_DIR / "Decision tree.pkl")
    )
    rt_reg = load_pickle(
        str(MODELS_DIR / "RandomForest.pkl")
    )
    svr_reg = load_pickle(
        str(MODELS_DIR / "svrression.pkl")
    )
    poly_reg = load_pickle(
        str(MODELS_DIR / "ploy_reg.pkl")
    )
    ohe = load_pickle(
        str(MODELS_DIR / "OneHotEncoder_Featureset.pkl")
    )
    sc_x = load_pickle(
        str(MODELS_DIR / "Standard_Scaler_X_SVR.pkl")
    )

except Exception as exc:
    st.error("Unable to load the prediction models.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Check the Models folder, filenames, and model files."
    )
    st.stop()


# =========================================================
# 5. POLLUTANT SETTINGS
# =========================================================
POLLUTANTS = {
    "PM2.5": ("PM2.5", 67.45, 500.0, "µg/m³"),
    "PM10": ("PM10", 118.13, 600.0, "µg/m³"),
    "NO": ("NO", 17.57, 200.0, "µg/m³"),
    "NO2": ("NO2", 28.56, 200.0, "µg/m³"),
    "NOx": ("NOx", 32.31, 300.0, "ppb"),
    "NH3": ("NH3", 23.48, 200.0, "µg/m³"),
    "CO": ("CO", 2.25, 50.0, "mg/m³"),
    "SO2": ("SO2", 14.53, 200.0, "µg/m³"),
    "O3": ("O3", 34.49, 300.0, "µg/m³"),
    "Benzene": ("Benzene", 3.28, 100.0, "µg/m³"),
    "Toluene": ("Toluene", 8.70, 200.0, "µg/m³"),
    "Xylene": ("Xylene", 3.07, 100.0, "µg/m³"),
}

MAIN_POLLUTANTS = ["PM2.5", "PM10"]

OPTIONAL_POLLUTANTS = [
    name
    for name in POLLUTANTS
    if name not in MAIN_POLLUTANTS
]


# =========================================================
# 6. AQI CATEGORY AND HEALTH GUIDANCE
# =========================================================
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#4ade80"
    elif aqi <= 100:
        return "Satisfactory", "#a3e635"
    elif aqi <= 200:
        return "Moderate", "#facc15"
    elif aqi <= 300:
        return "Poor", "#fb923c"
    elif aqi <= 400:
        return "Very Poor", "#f87171"
    return "Severe", "#c084fc"


def get_health_message(aqi):
    if aqi <= 50:
        return (
            "Air quality is in the Good range. Pollution-related "
            "risk is comparatively low for most people."
        )
    elif aqi <= 100:
        return (
            "Air quality is Satisfactory. Some sensitive individuals "
            "may experience discomfort."
        )
    elif aqi <= 200:
        return (
            "Air quality is Moderate. Sensitive individuals should "
            "consider reducing strenuous outdoor activity if symptoms "
            "occur."
        )
    elif aqi <= 300:
        return (
            "Air quality is Poor. Consider reducing prolonged or "
            "strenuous outdoor activity."
        )
    elif aqi <= 400:
        return (
            "Air quality is Very Poor. Reduce outdoor exposure and "
            "follow local public-health advice."
        )
    return (
        "Air quality is Severe. Minimize outdoor pollution exposure "
        "and follow official public-health advisories."
    )


# =========================================================
# 7. MODEL MAPPING AND PREDICTION
# =========================================================
MODEL_MAP = {
    "Multiple Regression": mreg,
    "Polynomial Regression": preg,
    "Decision Tree": dec_tree,
    "Random Forest": rt_reg,
    "SVR": svr_reg,
}

MODEL_DESCRIPTIONS = {
    "Multiple Regression": "Linear regression baseline.",
    "Polynomial Regression": "Regression with polynomial features.",
    "Decision Tree": "Predicts AQI using decision-based rules.",
    "Random Forest": "Combines multiple decision trees to predict AQI.",
    "SVR": "Uses Support Vector Regression to estimate AQI.",
}


def predict_aqi(city, values, model_name):
    # Encode the selected city.
    encoded_city = ohe.transform([[city]])

    if hasattr(encoded_city, "toarray"):
        encoded_city = encoded_city.toarray()

    encoded_city = np.asarray(encoded_city, dtype=float)

    # Keep the pollutant order consistent with the training code.
    pollutant_row = np.array(
        [[values[name] for name in POLLUTANTS.keys()]],
        dtype=float,
    )

    # Assumes the original training features were city encoding
    # followed by the 12 pollutant values.
    features = np.hstack([encoded_city, pollutant_row])

    model = MODEL_MAP[model_name]

    if model_name == "Polynomial Regression":
        model_input = poly_reg.transform(features)

    elif model_name == "SVR":
        model_input = sc_x.transform(features)

    else:
        model_input = features

    expected_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    if (
        expected_features is not None
        and model_input.shape[1] != expected_features
    ):
        raise ValueError(
            f"{model_name} expects {expected_features} input "
            f"features, but received {model_input.shape[1]}. "
            "The saved model's training feature order or "
            "preprocessing does not match this app."
        )

    prediction = np.asarray(
        model.predict(model_input)
    ).reshape(-1)

    if prediction.size == 0:
        raise ValueError("The model returned no prediction.")

    if not np.isfinite(prediction[0]):
        raise ValueError("The model returned an invalid prediction.")

    return float(prediction[0])


# =========================================================
# 8. PREDICTION WORKSPACE
# =========================================================
st.markdown(
    '<div class="section-heading">Prediction workspace</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Choose your city, select a machine-learning model, and enter '
    'pollutant measurements.'
    '</div>',
    unsafe_allow_html=True,
)


# Get city categories from the saved encoder.
@st.cache_data
def get_cities():
    return sorted(str(city) for city in ohe.categories_[0])

try:
    cities = get_cities()
except Exception as exc:
    st.error(f"Unable to read city categories: {exc}")
    st.stop()

if not cities:
    st.error("No city categories were found in the encoder.")
    st.stop()


# Model selection stays outside the form so its description
# updates as soon as the dropdown changes.
with st.container():
    st.markdown(sub_heading("cpu", "Prediction model"), unsafe_allow_html=True)

    model_names = [
        "Random Forest",
        "Multiple Regression",
        "Polynomial Regression",
        "Decision Tree",
        "SVR",
    ]

    selected_model = st.selectbox(
        "Choose model",
        model_names,
        key="selected_prediction_model",
    )

    st.caption(MODEL_DESCRIPTIONS[selected_model])


# =========================================================
# 9. INPUT FORM
# =========================================================
with st.form("aqi_prediction_form"):
    left, right = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    values = {}

    with left:
        with st.container():
            st.markdown(sub_heading("map-pin", "Location"), unsafe_allow_html=True)

            selected_city = st.selectbox(
                "Select city",
                cities,
                key="selected_city",
            )

            st.caption(
                "Choose a city supported by the saved encoder."
            )

        with st.container():
            st.markdown(sub_heading("wind", "Primary pollutants"), unsafe_allow_html=True)

            st.caption(
                "Default values are illustrative, not live readings."
            )

            col1, col2 = st.columns(2)

            for index, name in enumerate(MAIN_POLLUTANTS):
                label, default, max_value, unit = POLLUTANTS[name]
                target_column = col1 if index == 0 else col2

                with target_column:
                    values[name] = st.number_input(
                        f"{label} ({unit})",
                        min_value=0.0,
                        max_value=max_value,
                        value=float(default),
                        step=0.5,
                        key=f"main_{name}",
                    )

    with right:
        with st.container():
            st.markdown(sub_heading("sliders-horizontal", "Other pollutants"), unsafe_allow_html=True)

            with st.expander("Configure 10 more measurements"):
                for name in OPTIONAL_POLLUTANTS:
                    label, default, max_value, unit = POLLUTANTS[name]

                    values[name] = st.number_input(
                        f"{label} ({unit})",
                        min_value=0.0,
                        max_value=max_value,
                        value=float(default),
                        step=0.1,
                        key=f"optional_{name}",
                    )

            st.caption(
                "Unchanged values use defaults, not live measurements."
            )

    submitted = st.form_submit_button(
        "Predict Air Quality",
        icon=":material/auto_awesome:",
        use_container_width=True,
    )


# =========================================================
# 10. RUN PREDICTION
# =========================================================
if "aqi_result" not in st.session_state:
    st.session_state.aqi_result = None

if submitted:
    try:
        with st.spinner(f"Predicting with {selected_model}..."):
            prediction = predict_aqi(
                selected_city,
                values,
                selected_model,
            )

        st.session_state.aqi_result = {
            "aqi": prediction,
            "city": selected_city,
            "model": selected_model,
        }

    except Exception as exc:
        st.session_state.aqi_result = None

        st.error(f"Prediction failed using {selected_model}.")
        st.code(f"{type(exc).__name__}: {exc}")


# =========================================================
# 11. RESULTS — NATIVE STREAMLIT, NOT HTML
# =========================================================
st.markdown(
    '<div class="section-heading">Prediction results</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Your model estimate and its indicative AQI category.'
    '</div>',
    unsafe_allow_html=True,
)

result = st.session_state.aqi_result

if result is None:
    with st.container():
        st.markdown(sub_heading("sparkles", "Ready to analyze?"), unsafe_allow_html=True)
        st.write(
            "Enter your measurements and click **Predict Air Quality** "
            "to see your result."
        )

else:
    aqi = result["aqi"]

    if aqi < 0:
        st.error(f"Predicted AQI: {aqi:.2f}")

        st.warning(
            "A negative estimate is not a valid AQI reading. "
            "Check the training target, input features, and "
            "model preprocessing."
        )

    else:
        category, category_color = get_aqi_category(aqi)

        result_col, details_col = st.columns(
            [1, 1],
            gap="large",
        )

        with result_col:
            with st.container():
                st.markdown(sub_heading("activity", "Predicted air quality"), unsafe_allow_html=True)

                # Native Streamlit components display values as text,
                # rather than inserting a custom HTML result card.
                st.metric(
                    label="Predicted AQI",
                    value=f"{aqi:.2f}",
                )

                st.markdown(
                    "<span class='badge'>"
                    f"<span class='dot' style='background:{category_color}'></span>"
                    f"{html.escape(category)}</span>",
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"City: {result['city']}  \n"
                    f"Model: {result['model']}"
                )

        with details_col:
            with st.container():
                st.markdown(sub_heading("heart-pulse", "Health guidance"), unsafe_allow_html=True)
                st.write(get_health_message(aqi))

                st.caption(
                    "General educational guidance only. Follow official "
                    "local air-quality and public-health advisories."
                )

        st.markdown(sub_heading("chart-column", "AQI reference scale"), unsafe_allow_html=True)

        scale_columns = st.columns(6)

        bands = [
            ("0–50", "Good", "#4ade80"),
            ("51–100", "Satisfactory", "#a3e635"),
            ("101–200", "Moderate", "#facc15"),
            ("201–300", "Poor", "#fb923c"),
            ("301–400", "Very Poor", "#f87171"),
            ("401+", "Severe", "#c084fc"),
        ]

        for column, (limits, label, color) in zip(
            scale_columns,
            bands,
        ):
            with column:
                active = " active" if label == category else ""
                st.markdown(
                    f"<div class='scale-item{active}'>"
                    f"<span class='dot' style='background:{color}'></span>"
                    f"<div class='scale-range'>{limits}</div>"
                    f"<div class='scale-label'>{html.escape(label)}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.caption(
            "Make sure the model predicts AQI on the same scale used "
            "by this reference. This is a model estimate, not an "
            "official monitoring-station reading."
        )


# =========================================================
# 12. FOOTER
# =========================================================
st.markdown(
    f"""
    <div class="footer">
        <strong>{icon("cloud-fog", 14)} AirWatch India</strong><br>
        Machine-learning-based AQI estimation for educational use.
        This app does not provide live monitoring or official AQI
        readings. Results depend on the training data, model quality,
        feature order, preprocessing, and supplied measurements.
    </div>
    """,
    unsafe_allow_html=True,
)