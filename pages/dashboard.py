import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from theme import icon

# =========================================================
# LOAD DATASET — UNCHANGED
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('Data/city_day.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['AQI'])
    return df


data = load_data()

# Introduction
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">{icon("chart-column", 13)} Air Quality Analytics</div>
    <div class="hero-title">
        Understand the air.<br>
        <span class="text-muted">Make better decisions.</span>
    </div>
    <div class="hero-description">
        Explore air pollution levels across Indian cities.
        Discover historical AQI patterns, compare city performance,
        and monitor pollutant concentrations through interactive analytics.
    </div>
</div>
""", unsafe_allow_html=True)

# Data cleaning (date parsing and dropping missing AQI) happens in load_data()

# =========================================================
# CHART STYLE
# =========================================================
BACKGROUND = "#09090b"
ACCENT = "#3b82f6"
TEXT = "#d4d4d8"
MUTED = "#a1a1aa"
GRID = "#27272a"

PALETTE = [
    "#60a5fa",
    "#e4e4e7",
    "#3b82f6",
    "#a1a1aa",
    "#93c5fd",
    "#71717a"
]

sns.set_theme(style="white")

def style_chart(fig, ax):
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    ax.set_title(
        ax.get_title(),
        color="#fafafa",
        fontsize=13,
        fontweight="bold",
        pad=16
    )

    ax.set_xlabel(ax.get_xlabel(), color=TEXT, fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), color=TEXT, fontsize=10)

    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=2)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.caption("WORKSPACE")
st.sidebar.markdown("### Dashboard")
st.sidebar.divider()

st.sidebar.markdown("##### EXPLORE ANALYTICS")

display_aqi_by_city = st.sidebar.checkbox(
    ':material/location_city: AQI by City'
)

display_aqi_by_year = st.sidebar.checkbox(
    ':material/calendar_month: AQI by Year'
)

display_aqi_by_bucket = st.sidebar.checkbox(
    ':material/donut_small: AQI Categories'
)

display_pollutants_over_time = st.sidebar.checkbox(
    ':material/show_chart: Pollutant Trends',
    value=True
)

st.sidebar.divider()
st.sidebar.caption("DATA SOURCE")
st.sidebar.markdown("**Kaggle · city_day.csv**")
st.sidebar.caption("Historical air quality analysis")

# =========================================================
# OVERVIEW METRICS
# =========================================================
st.markdown('<div class="section-heading">Overview</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">'
    'A quick snapshot of your air quality dataset.'
    '</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Average AQI", f"{data['AQI'].mean():.1f}")

with m2:
    st.metric("Highest AQI", f"{data['AQI'].max():.1f}")

with m3:
    st.metric("Cities Covered", f"{data['City'].nunique():,}")

with m4:
    st.metric("Total Records", f"{len(data):,}")

st.divider()

# =========================================================
# AQI BY CITY
# =========================================================
if display_aqi_by_city:

    st.markdown('<div class="section-heading">AQI by City</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Explore historical AQI patterns for a selected city.'
        '</div>',
        unsafe_allow_html=True
    )

    cities = sorted(data['City'].dropna().unique())
    selected_city = st.selectbox('Select a city', cities)
    city_data = data[data['City'] == selected_city]

    yearly_city = (
        city_data.groupby(city_data['Date'].dt.year)['AQI']
        .mean()
        .reset_index()
    )
    yearly_city.columns = ['Year', 'AQI']

    left, right = st.columns([2, 1])

    with left:
        fig_aqi_city, ax_aqi_city = plt.subplots(figsize=(9, 4.5))

        sns.barplot(
            data=yearly_city,
            x='Year',
            y='AQI',
            color=ACCENT,
            ax=ax_aqi_city,
            errorbar=None
        )

        ax_aqi_city.set_title(f'Annual Average AQI — {selected_city}')
        ax_aqi_city.set_xlabel('Year')
        ax_aqi_city.set_ylabel('Average AQI')

        style_chart(fig_aqi_city, ax_aqi_city)
        st.pyplot(fig_aqi_city, use_container_width=True)
        plt.close(fig_aqi_city)

    with right:
        st.markdown("##### City statistics")
        st.metric("Average AQI", f"{city_data['AQI'].mean():.1f}")
        st.metric("Maximum AQI", f"{city_data['AQI'].max():.1f}")
        st.metric("Records", f"{len(city_data):,}")

    st.info(
        "Higher AQI generally indicates worse air quality. "
        "Annual averages help reveal longer-term patterns."
    )

    st.divider()

# =========================================================
# AQI BY YEAR
# =========================================================
if display_aqi_by_year:

    st.markdown('<div class="section-heading">AQI by Year</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Compare average air quality between cities in a selected year.'
        '</div>',
        unsafe_allow_html=True
    )

    years = sorted(data['Date'].dt.year.dropna().unique())
    selected_year = st.selectbox('Select a year', years)
    year_data = data[data['Date'].dt.year == selected_year]

    city_year_avg = (
        year_data.groupby('City', as_index=False)['AQI']
        .mean()
        .sort_values('AQI', ascending=False)
    )

    fig_aqi_year, ax_aqi_year = plt.subplots(figsize=(11, 5))

    sns.barplot(
        data=city_year_avg,
        x='City',
        y='AQI',
        color=ACCENT,
        ax=ax_aqi_year,
        errorbar=None
    )

    ax_aqi_year.set_title(f'Average AQI Across Cities — {selected_year}')
    ax_aqi_year.set_xlabel('City')
    ax_aqi_year.set_ylabel('Average AQI')
    ax_aqi_year.tick_params(axis='x', rotation=45)

    style_chart(fig_aqi_year, ax_aqi_year)
    st.pyplot(fig_aqi_year, use_container_width=True)
    plt.close(fig_aqi_year)

    if not city_year_avg.empty:
        best_city = city_year_avg.iloc[-1]
        worst_city = city_year_avg.iloc[0]

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Highest average AQI",
                f"{worst_city['AQI']:.1f}",
                str(worst_city['City'])
            )

        with c2:
            st.metric(
                "Lowest average AQI",
                f"{best_city['AQI']:.1f}",
                str(best_city['City'])
            )

    st.divider()

# =========================================================
# AQI CATEGORIES
# =========================================================
if display_aqi_by_bucket:

    st.markdown('<div class="section-heading">AQI Categories</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Understand the distribution of reported air quality categories.'
        '</div>',
        unsafe_allow_html=True
    )

    if 'AQI_Bucket' in data.columns:

        bucket_data = data.dropna(subset=['AQI_Bucket'])

        fig_aqi_bucket, ax_aqi_bucket = plt.subplots(figsize=(9, 4.5))

        sns.countplot(
            data=bucket_data,
            x='AQI_Bucket',
            order=bucket_data['AQI_Bucket'].value_counts().index,
            color=ACCENT,
            ax=ax_aqi_bucket
        )

        ax_aqi_bucket.set_title('Distribution of AQI Categories')
        ax_aqi_bucket.set_xlabel('AQI Category')
        ax_aqi_bucket.set_ylabel('Number of Records')
        ax_aqi_bucket.tick_params(axis='x', rotation=25)

        style_chart(fig_aqi_bucket, ax_aqi_bucket)
        st.pyplot(fig_aqi_bucket, use_container_width=True)
        plt.close(fig_aqi_bucket)

        bucket_counts = bucket_data['AQI_Bucket'].value_counts()

        cols = st.columns(min(len(bucket_counts), 4))

        for i, (bucket, count) in enumerate(bucket_counts.items()):
            with cols[i % len(cols)]:
                st.metric(str(bucket), f"{count:,}")

    else:
        st.warning("The dataset does not contain an AQI_Bucket column.")

    st.divider()

# =========================================================
# POLLUTANTS OVER TIME
# =========================================================
if display_pollutants_over_time:

    st.markdown('<div class="section-heading">Pollutant Trends</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-description">'
        'Monitor pollutant concentrations over time for a selected city.'
        '</div>',
        unsafe_allow_html=True
    )

    pollutants = [
        'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
        'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
    ]

    # Keep only pollutant columns available in the dataset
    pollutants = [p for p in pollutants if p in data.columns]

    selected_pollutants = st.multiselect(
        'Select pollutants',
        pollutants,
        default=[p for p in ['PM2.5', 'PM10'] if p in pollutants]
    )

    cities = sorted(data['City'].dropna().unique())

    selected_city = st.selectbox(
        'Select a city',
        cities,
        key='pollutant_city'
    )

    city_data = data[data['City'] == selected_city].sort_values('Date')

    if selected_pollutants:

        fig_pollutants, ax_pollutants = plt.subplots(figsize=(11, 5))

        for i, pollutant in enumerate(selected_pollutants):
            sns.lineplot(
                x=city_data['Date'],
                y=city_data[pollutant],
                label=pollutant,
                color=PALETTE[i % len(PALETTE)],
                linewidth=1.8,
                ax=ax_pollutants
            )

        ax_pollutants.set_title(
            f'Pollutant Concentrations — {selected_city}'
        )
        ax_pollutants.set_xlabel('Date')
        ax_pollutants.set_ylabel('Pollutant Level')

        ax_pollutants.legend(
            frameon=False,
            labelcolor=TEXT,
            ncol=2
        )

        style_chart(fig_pollutants, ax_pollutants)
        st.pyplot(fig_pollutants, use_container_width=True)
        plt.close(fig_pollutants)

        st.info(
            "Pollutants may use different measurement units. "
            "Compare individual pollutant trends carefully rather "
            "than assuming all lines are directly comparable."
        )

    else:
        st.warning("Select at least one pollutant to display the chart.")

    st.divider()

if not any([
    display_aqi_by_city,
    display_aqi_by_year,
    display_aqi_by_bucket,
    display_pollutants_over_time,
]):
    st.info("Select at least one analytics view from the sidebar to see charts.")

# =========================================================
# FOOTER
# =========================================================
st.markdown(f"""
<div class="footer">
    <strong>{icon("cloud-fog", 14)} AirWatch India</strong>
    <br>
    An air quality awareness and data visualization project.
    <br><br>
    Built with Streamlit, Pandas, Seaborn & Matplotlib.
</div>
""", unsafe_allow_html=True)