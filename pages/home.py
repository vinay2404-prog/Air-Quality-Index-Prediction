
import streamlit as st
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AirWatch India | Awareness",
    page_icon="🌫️",
    layout="wide",
)

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.title("🌫️ AirWatch India")
st.subheader("Understand the air. Protect your health.")

st.write(
    """
    Explore air pollution across Indian cities, understand the health risks,
    and learn how air quality affects everyday life. Make more informed
    decisions about the air you breathe.
    """
)

st.divider()

# --------------------------------------------------
# INDIA AQI IMAGE
# --------------------------------------------------
st.header("Air Quality in India")
st.write("A visual overview of air pollution across India.")

image_path = Path(__file__).parent / "India-2023.png"

if image_path.exists():
    st.image(
        str(image_path),
        caption="Air Quality Index across India — 2023",
        width="stretch",
    )
else:
    st.warning(
        "Image not found. Place India-2023.png in the same "
        "folder as this Streamlit page."
    )

st.divider()

# --------------------------------------------------
# HEALTH IMPACT
# --------------------------------------------------
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

# --------------------------------------------------
# LIFE EXPECTANCY
# --------------------------------------------------
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

# --------------------------------------------------
# PRACTICAL RECOMMENDATIONS
# --------------------------------------------------
st.header("What Can You Do?")
st.write("Simple precautions to reduce your exposure on polluted days.")

with st.container(border=True):
    st.subheader("🔎 Check local air quality")
    st.write(
        """
        Check your city's current AQI before planning outdoor activities.
        This page displays a 2023 image, not real-time monitoring data.
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

# --------------------------------------------------
# MISSION
# --------------------------------------------------
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

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.subheader("🌫️ AirWatch India")

st.caption(
    """
    An air quality awareness and data visualization project.

    Educational information only. Health effects vary by exposure
    and individual circumstances.
    """
)
