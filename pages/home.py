import streamlit as st

from theme import icon


def card(icon_name, title, text):
    return (
        '<div class="info-card">'
        f'<div class="card-icon">{icon(icon_name, 20)}</div>'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-description">{text}</div>'
        '</div>'
    )


def section(title, description):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-description">{description}</div>', unsafe_allow_html=True)


# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">{icon("sparkles", 13)} Air Quality Awareness</div>
    <div class="hero-title">
        Understand the air.<br>
        <span class="text-muted">Protect your health.</span>
    </div>
    <div class="hero-description">
        Explore air pollution across Indian cities, understand
        the health risks, and learn how air quality affects
        everyday life. Make more informed decisions about
        the air you breathe.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INDIA AQI IMAGE
# --------------------------------------------------
section("Air Quality in India", "A visual overview of air pollution across India.")

try:
    st.image(
        "India-2023.png",
        caption="Air Quality Index across India — 2023",
        width="stretch",
    )
except Exception:
    st.warning(
        "Image not found. Place India-2023.png in the same "
        "folder as your Streamlit application."
    )


# --------------------------------------------------
# HEALTH IMPACT
# --------------------------------------------------
section("Health & Well-being", "How exposure to air pollution can affect your body.")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(card(
        "wind", "Respiratory Health",
        "Pollutants such as PM2.5 and nitrogen dioxide can irritate the "
        "airways, worsen asthma, and contribute to respiratory disease. "
        "Fine particles can travel deep into the lungs, making prolonged "
        "exposure a serious health concern.",
    ), unsafe_allow_html=True)

with col2:
    st.markdown(card(
        "heart-pulse", "Cardiovascular Health",
        "Air pollution is associated with an increased risk of heart "
        "disease and stroke. Fine particulate matter can contribute to "
        "inflammation and affect blood vessels, especially in people "
        "with existing cardiovascular conditions.",
    ), unsafe_allow_html=True)


# --------------------------------------------------
# LIFE EXPECTANCY
# --------------------------------------------------
section("Air Pollution & Life Expectancy", "Long-term exposure can affect health over many years.")

col3, col4, col5 = st.columns(3, gap="medium")

with col3:
    st.markdown(card(
        "hourglass", "Long-term Exposure",
        "Persistent exposure to polluted air is associated with chronic "
        "health risks and premature mortality.",
    ), unsafe_allow_html=True)

with col4:
    st.markdown(card(
        "building-2", "Urban Pollution",
        "Traffic, industrial emissions, construction dust, and burning "
        "fuels can worsen urban air quality.",
    ), unsafe_allow_html=True)

with col5:
    st.markdown(card(
        "microscope", "PM2.5 Particles",
        "These tiny airborne particles can penetrate deep into the "
        "lungs. Long-term exposure is linked to serious health effects.",
    ), unsafe_allow_html=True)


# --------------------------------------------------
# PRACTICAL RECOMMENDATIONS
# --------------------------------------------------
section("What Can You Do?", "Simple precautions to reduce your exposure on polluted days.")


def tip(icon_name, title, text):
    return (
        '<div class="tip">'
        f'{icon(icon_name, 18)}'
        '<div>'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-description">{text}</div>'
        '</div></div>'
    )


st.markdown(
    '<div class="info-card">'
    + tip(
        "search", "Check local air quality",
        "Check your city's current AQI before planning outdoor "
        "activities. This page displays a 2023 image, not real-time "
        "monitoring data.",
    )
    + "<hr>"
    + tip(
        "house", "Reduce exposure on polluted days",
        "When pollution levels are high, reduce strenuous outdoor "
        "exercise and spend more time in cleaner indoor environments "
        "when possible.",
    )
    + "<hr>"
    + tip(
        "shield-check", "Protect vulnerable people",
        "Children, older adults, and people with heart or lung "
        "conditions may need extra precautions during pollution episodes.",
    )
    + "</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MISSION
# --------------------------------------------------
section("Our Mission", "Why AirWatch India exists.")

st.markdown(f"""
<div class="hero">
    <div class="card-icon">{icon("target", 20)}</div>
    <div class="card-title">Cleaner air starts with awareness.</div>
    <div class="card-description">
        AirWatch India aims to make air pollution information easier to
        understand. By exploring pollution trends, understanding health
        risks, and encouraging informed decisions, we can help people
        take practical steps towards healthier communities.
        <br><br>
        Better data. Better awareness. Healthier cities.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(f"""
<div class="footer">
    <strong>{icon("cloud-fog", 14)} AirWatch India</strong>
    <br>
    An air quality awareness and data visualization project.
    <br><br>
    Educational information only. Health effects vary by
    exposure and individual circumstances.
</div>
""", unsafe_allow_html=True)