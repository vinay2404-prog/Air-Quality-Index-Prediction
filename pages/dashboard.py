import streamlit as st
import pandas as pd

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AirWatch India | AQI Dashboard",
    page_icon="🌫️",
    layout="wide",
)

# =========================================================
# LOAD DATASET — UNCHANGED
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("Data/city_day.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["AQI"])
    return df


data = load_data()

# =========================================================
# INTRODUCTION
# =========================================================
st.title("🌫️ AirWatch India")
st.subheader("Understand the air. Make better decisions.")

st.write(
    """
    Explore air pollution levels across Indian cities. Discover historical
    AQI patterns, compare city performance, and monitor pollutant
    concentrations through interactive analytics.
    """
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.caption("WORKSPACE")
st.sidebar.title("Dashboard")
st.sidebar.divider()
st.sidebar.subheader("Explore Analytics")

display_aqi_by_city = st.sidebar.checkbox("🏙️ AQI by City")
display_aqi_by_year = st.sidebar.checkbox("📅 AQI by Year")
display_aqi_by_bucket = st.sidebar.checkbox("◉ AQI Categories")
display_pollutants_over_time = st.sidebar.checkbox(
    "📈 Pollutant Trends",
    value=True,
)

st.sidebar.divider()
st.sidebar.caption("DATA SOURCE")
st.sidebar.write("**Kaggle · city_day.csv**")
st.sidebar.caption("Historical air quality analysis")

# =========================================================
# OVERVIEW METRICS
# =========================================================
st.header("Overview")
st.write("A quick snapshot of your air quality dataset.")

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
    st.header("AQI by City")
    st.write("Explore historical AQI patterns for a selected city.")

    cities = sorted(data["City"].dropna().unique())
    selected_city = st.selectbox("Select a city", cities)
    city_data = data[data["City"] == selected_city]

    yearly_city = (
        city_data.groupby(city_data["Date"].dt.year)["AQI"]
        .mean()
        .reset_index()
    )
    yearly_city.columns = ["Year", "AQI"]
    yearly_city["Year"] = yearly_city["Year"].astype(str)

    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"Annual Average AQI — {selected_city}")
        st.bar_chart(yearly_city, x="Year", y="AQI")

    with right:
        st.subheader("City statistics")
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
    st.header("AQI by Year")
    st.write("Compare average air quality between cities in a selected year.")

    years = sorted(data["Date"].dt.year.dropna().unique())
    selected_year = st.selectbox("Select a year", years)
    year_data = data[data["Date"].dt.year == selected_year]

    city_year_avg = (
        year_data.groupby("City", as_index=False)["AQI"]
        .mean()
        .sort_values("AQI", ascending=False)
    )

    st.subheader(f"Average AQI Across Cities — {selected_year}")
    st.bar_chart(city_year_avg, x="City", y="AQI")

    if not city_year_avg.empty:
        best_city = city_year_avg.iloc[-1]
        worst_city = city_year_avg.iloc[0]

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Highest average AQI",
                f"{worst_city['AQI']:.1f}",
                str(worst_city["City"]),
            )

        with c2:
            st.metric(
                "Lowest average AQI",
                f"{best_city['AQI']:.1f}",
                str(best_city["City"]),
            )

    st.divider()

# =========================================================
# AQI CATEGORIES
# =========================================================
if display_aqi_by_bucket:
    st.header("AQI Categories")
    st.write("Understand the distribution of reported air quality categories.")

    if "AQI_Bucket" in data.columns:
        bucket_data = data.dropna(subset=["AQI_Bucket"])
        bucket_counts = (
            bucket_data["AQI_Bucket"]
            .value_counts()
            .rename_axis("AQI Category")
            .reset_index(name="Number of Records")
        )

        st.subheader("Distribution of AQI Categories")
        st.bar_chart(
            bucket_counts,
            x="AQI Category",
            y="Number of Records",
        )

        metric_columns = st.columns(min(len(bucket_counts), 4))

        for i, row in enumerate(bucket_counts.itertuples(index=False)):
            with metric_columns[i % len(metric_columns)]:
                st.metric(str(row[0]), f"{row[1]:,}")

    else:
        st.warning("The dataset does not contain an AQI_Bucket column.")

    st.divider()

# =========================================================
# POLLUTANTS OVER TIME
# =========================================================
if display_pollutants_over_time:
    st.header("Pollutant Trends")
    st.write("Monitor pollutant concentrations over time for a selected city.")

    pollutants = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
        "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
    ]

    # Keep only pollutant columns available in the dataset.
    pollutants = [p for p in pollutants if p in data.columns]

    selected_pollutants = st.multiselect(
        "Select pollutants",
        pollutants,
        default=[p for p in ["PM2.5", "PM10"] if p in pollutants],
    )

    cities = sorted(data["City"].dropna().unique())

    selected_city = st.selectbox(
        "Select a city",
        cities,
        key="pollutant_city",
    )

    city_data = data[data["City"] == selected_city].sort_values("Date")

    if selected_pollutants:
        pollutant_data = (
            city_data.set_index("Date")[selected_pollutants]
            .apply(pd.to_numeric, errors="coerce")
        )

        st.subheader(f"Pollutant Concentrations — {selected_city}")
        st.line_chart(pollutant_data)

        st.info(
            "Pollutants may use different measurement units. Compare "
            "individual pollutant trends carefully rather than assuming "
            "all lines are directly comparable."
        )
    else:
        st.warning("Select at least one pollutant to display the chart.")

    st.divider()

# =========================================================
# EMPTY STATE
# =========================================================
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
st.divider()
st.subheader("🌫️ AirWatch India")
st.caption(
    "An air quality awareness and data visualization project. "
    "Built with Streamlit and Pandas."
)
