from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError

from aqi_app.data import DataError, dataset_summary, load_dataset


def page_link(path: str, label: str, icon: str) -> None:
    """Render a page link, or plain text when the page is rendered in isolation."""
    try:
        st.page_link(path, label=label, icon=icon)
    except StreamlitPageNotFoundError:
        st.caption(f"{icon} {label}")

st.title("🌫️ AirWatch India")
st.subheader("Understand the air. Protect your health.")

st.write(
    """
    Explore historical air pollution across Indian cities, check live
    conditions and a short forecast, and estimate AQI from pollutant
    measurements with transparent machine-learning models.
    """
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

hero_left, hero_right = st.columns([1.3, 1], gap="large")

with hero_left:
    st.markdown("#### Three ways to use AirWatch")
    st.markdown(
        """
        - **Dashboard** — explore 2015–2020 trends by city, year, category,
          and pollutant.
        - **Live AQI** — fetch current conditions and a 5-day forecast for
          any city using the Open-Meteo API.
        - **Prediction** — estimate AQI from pollutant measurements and
          compare six models with published test metrics.
        """
    )
    page_link("pages/dashboard.py", label="Open the historical dashboard", icon="📊")
    page_link("pages/live.py", label="Check live air quality", icon="📡")
    page_link("pages/prediction.py", label="Estimate AQI with ML", icon="🤖")

with hero_right:
    hero_image = PROJECT_ROOT / "air.jpeg"
    if hero_image.is_file():
        st.image(str(hero_image), width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Data snapshot
# ---------------------------------------------------------------------------
st.header("Data at a glance")


@st.cache_data(show_spinner=False)
def cached_summary():
    return dataset_summary(load_dataset())


try:
    summary = cached_summary()
except DataError as exc:
    st.error(f"Historical dataset unavailable: {exc}")
else:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Historical records", f"{summary['rows']:,}")
    with m2:
        st.metric("Cities covered", f"{summary['cities']}")
    with m3:
        st.metric("Date range", f"{summary['start']} → {summary['end']}")
    with m4:
        st.metric("Average AQI", f"{summary['average_aqi']:.1f}")
    st.caption(
        "Source: Kaggle Air Quality Data in India. The dataset ends in "
        "July 2020; use the Live AQI page for current conditions."
    )

st.divider()

# ---------------------------------------------------------------------------
# India AQI image
# ---------------------------------------------------------------------------
st.header("Air Quality in India")
st.write("A visual overview of air pollution across India.")

image_path = PROJECT_ROOT / "India-2023.png"

if image_path.is_file():
    st.image(
        str(image_path),
        caption="Air Quality Index across India — 2023",
        width="stretch",
    )
else:
    st.warning(
        "Image not found. Place India-2023.png in the project root "
        f"({PROJECT_ROOT})."
    )

st.divider()

# ---------------------------------------------------------------------------
# Health impact
# ---------------------------------------------------------------------------
st.header("Health & Well-being")
st.write("How exposure to air pollution can affect your body.")

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.subheader("🫁 Respiratory Health")
        st.write(
            """
            Pollutants such as PM2.5 and nitrogen dioxide can irritate
            the airways, worsen asthma, and contribute to respiratory
            disease. Fine particles can travel deep into the lungs,
            making prolonged exposure a serious health concern.
            """
        )

with col2:
    with st.container(border=True):
        st.subheader("❤️ Cardiovascular Health")
        st.write(
            """
            Air pollution is associated with an increased risk of heart
            disease and stroke. Fine particulate matter can contribute
            to inflammation and affect blood vessels, especially in
            people with existing cardiovascular conditions.
            """
        )

st.divider()

# ---------------------------------------------------------------------------
# Life expectancy
# ---------------------------------------------------------------------------
st.header("Air Pollution & Life Expectancy")
st.write("Long-term exposure can affect health over many years.")

col3, col4, col5 = st.columns(3, gap="medium")

with col3:
    with st.container(border=True):
        st.subheader("⏳ Long-term Exposure")
        st.write(
            """
            Persistent exposure to polluted air is associated with
            chronic health risks and premature mortality.
            """
        )

with col4:
    with st.container(border=True):
        st.subheader("🏙️ Urban Pollution")
        st.write(
            """
            Traffic, industrial emissions, construction dust, and
            burning fuels can worsen urban air quality.
            """
        )

with col5:
    with st.container(border=True):
        st.subheader("🔬 PM2.5 Particles")
        st.write(
            """
            These tiny airborne particles can penetrate deep into
            the lungs. Long-term exposure is linked to serious
            health effects.
            """
        )

st.divider()

# ---------------------------------------------------------------------------
# Practical recommendations
# ---------------------------------------------------------------------------
st.header("What Can You Do?")
st.write("Simple precautions to reduce your exposure on polluted days.")

with st.container(border=True):
    st.subheader("🔎 Check local air quality")
    st.write(
        """
        Check your city's current AQI before planning outdoor activities.
        The Live AQI page shows current conditions and a 5-day forecast;
        the dashboard shows the historical record.
        """
    )

    st.divider()

    st.subheader("🏠 Reduce exposure on polluted days")
    st.write(
        """
        When pollution levels are high, reduce strenuous outdoor exercise
        and spend more time in cleaner indoor environments when possible.
        """
    )

    st.divider()

    st.subheader("🛡️ Protect vulnerable people")
    st.write(
        """
        Children, older adults, and people with heart or lung conditions
        may need extra precautions during pollution episodes.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------
st.header("Our Mission")
st.write("Why AirWatch India exists.")

with st.container(border=True):
    st.subheader("🎯 Cleaner air starts with awareness.")

    st.write(
        """
        AirWatch India aims to make air pollution information easier
        to understand. By exploring pollution trends, understanding
        health risks, and encouraging informed decisions, we can help
        people take practical steps towards healthier communities.
        """
    )

    st.info("Better data. Better awareness. Healthier cities.")

st.divider()

st.subheader("🌫️ AirWatch India")

st.caption(
    """
    An air quality awareness, data visualization, and machine-learning project.

    Educational information only. Health effects vary by exposure and
    individual circumstances. Model estimates are not official monitoring
    readings.
    """
)
