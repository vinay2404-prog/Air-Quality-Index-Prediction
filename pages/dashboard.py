import pandas as pd
import streamlit as st

from aqi_app.constants import (
    AQI_CATEGORY_ORDER,
    POLLUTANT_ORDER,
    TARGET_COLUMN,
    UNIT_GROUPS,
    aqi_category,
)
from aqi_app.data import (
    DataError,
    city_average_frame,
    filter_dataset,
    load_dataset,
    target_frame,
)


@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return load_dataset()


try:
    data = cached_dataset()
except DataError as exc:
    st.error(f"Unable to load the historical dataset: {exc}")
    st.stop()

st.title("🌫️ AirWatch India — Dashboard")
st.subheader("Understand the air. Make better decisions.")
st.write(
    """
    Explore historical air pollution levels across Indian cities. Filter the
    record, compare cities, inspect pollutant trends, and download the data
    behind each view.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.caption("FILTERS")
st.sidebar.title("Dashboard")

min_date = data["Date"].min().date()
max_date = data["Date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (tuple, list)):
    start_date = date_range[0] if len(date_range) > 0 else min_date
    end_date = date_range[1] if len(date_range) > 1 else max_date
else:
    start_date, end_date = date_range, date_range

all_cities = sorted(data["City"].dropna().unique().tolist())
preferred = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai"]
default_cities = [city for city in preferred if city in all_cities] or all_cities[:5]

selected_cities = st.sidebar.multiselect(
    "Cities",
    all_cities,
    default=default_cities,
    help="Charts and summary metrics are limited to this selection.",
)

st.sidebar.divider()
st.sidebar.caption("VIEWS")
display_aqi_by_city = st.sidebar.checkbox("🏙️ AQI by city", value=True)
display_aqi_by_year = st.sidebar.checkbox("📅 AQI by year", value=True)
display_aqi_by_bucket = st.sidebar.checkbox("◉ AQI categories", value=True)
display_pollutants_over_time = st.sidebar.checkbox("📈 Pollutant trends", value=True)
display_city_map = st.sidebar.checkbox("🗺️ City map", value=True)

st.sidebar.divider()
st.sidebar.caption("DATA SOURCE")
st.sidebar.write("**Kaggle · city_day.csv**")
st.sidebar.caption(f"{min_date} → {max_date} · historical analysis")

filtered = filter_dataset(data, start_date, end_date, selected_cities)

if filtered.empty:
    st.warning("No records match the selected filters. Widen the date range or add cities.")
    st.stop()

target_data = target_frame(filtered)

# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------
st.header("Overview")
st.write("A quick snapshot of the filtered dataset.")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Average AQI", f"{target_data[TARGET_COLUMN].mean():.1f}")
with m2:
    st.metric("Highest AQI", f"{target_data[TARGET_COLUMN].max():.1f}")
with m3:
    st.metric("Cities in view", f"{filtered['City'].nunique():,}")
with m4:
    st.metric("Records in view", f"{len(filtered):,}")

st.caption(
    f"Showing {start_date} → {end_date}. "
    f"{len(filtered) - len(target_data):,} rows have no reported AQI and are "
    "excluded from AQI statistics."
)
st.divider()

if target_data.empty:
    st.warning("The selected records contain no reported AQI values.")
    st.stop()

# ---------------------------------------------------------------------------
# AQI by city
# ---------------------------------------------------------------------------
if display_aqi_by_city:
    st.header("AQI by City")
    st.write("Compare annual average AQI for the selected cities.")

    yearly_city = (
        target_data.assign(Year=target_data["Date"].dt.year)
        .groupby(["Year", "City"], as_index=False)[TARGET_COLUMN]
        .mean()
        .pivot(index="Year", columns="City", values=TARGET_COLUMN)
        .sort_index()
    )

    if not yearly_city.empty:
        st.subheader("Annual average AQI")
        st.line_chart(yearly_city)

        city_stats = (
            target_data.groupby("City", as_index=False)
            .agg(
                **{
                    "Average AQI": (TARGET_COLUMN, "mean"),
                    "Maximum AQI": (TARGET_COLUMN, "max"),
                    "Minimum AQI": (TARGET_COLUMN, "min"),
                    "Records": (TARGET_COLUMN, "count"),
                }
            )
            .sort_values("Average AQI", ascending=False)
        )
        st.subheader("City statistics")
        st.dataframe(
            city_stats.style.format(
                {
                    "Average AQI": "{:.1f}",
                    "Maximum AQI": "{:.1f}",
                    "Minimum AQI": "{:.1f}",
                    "Records": "{:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.info(
            "Higher AQI generally indicates worse air quality. Annual averages "
            "help reveal longer-term patterns, while the table shows the spread."
        )
    else:
        st.warning("No annual data is available for the selected filters.")
    st.divider()

# ---------------------------------------------------------------------------
# AQI by year
# ---------------------------------------------------------------------------
if display_aqi_by_year:
    st.header("AQI by Year")
    st.write("Compare average air quality between cities in a selected year.")

    years = sorted(target_data["Date"].dt.year.dropna().unique().tolist())
    selected_year = st.selectbox("Select a year", years, index=len(years) - 1)
    year_data = target_data[target_data["Date"].dt.year == selected_year]

    city_year_avg = (
        year_data.groupby("City", as_index=False)[TARGET_COLUMN]
        .mean()
        .sort_values(TARGET_COLUMN, ascending=False)
    )

    if not city_year_avg.empty:
        st.subheader(f"Average AQI Across Cities — {selected_year}")
        st.bar_chart(city_year_avg, x="City", y=TARGET_COLUMN)

        best_city = city_year_avg.iloc[-1]
        worst_city = city_year_avg.iloc[0]

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Highest average AQI",
                f"{worst_city[TARGET_COLUMN]:.1f}",
                str(worst_city["City"]),
            )
        with c2:
            st.metric(
                "Lowest average AQI",
                f"{best_city[TARGET_COLUMN]:.1f}",
                str(best_city["City"]),
            )
    else:
        st.warning("No city data is available for the selected year.")
    st.divider()

# ---------------------------------------------------------------------------
# AQI categories
# ---------------------------------------------------------------------------
if display_aqi_by_bucket:
    st.header("AQI Categories")
    st.write("Understand the distribution of reported air quality categories.")

    if "AQI_Bucket" in target_data.columns:
        bucket_series = target_data["AQI_Bucket"]
    else:
        bucket_series = target_data[TARGET_COLUMN].map(aqi_category)

    bucket_data = (
        bucket_series.dropna()
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Records")
    )
    bucket_data["Share"] = bucket_data["Records"] / bucket_data["Records"].sum() * 100
    bucket_data["Category"] = pd.Categorical(
        bucket_data["Category"],
        categories=AQI_CATEGORY_ORDER,
        ordered=True,
    )
    bucket_data = bucket_data.sort_values("Category").reset_index(drop=True)

    if not bucket_data.empty:
        st.subheader("Distribution of AQI categories")
        st.bar_chart(bucket_data, x="Category", y="Records")

        st.dataframe(
            bucket_data.assign(
                Category=bucket_data["Category"].astype(str),
                Share=bucket_data["Share"].map(lambda value: f"{value:.1f}%"),
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("The selected records do not contain AQI category values.")
    st.divider()

# ---------------------------------------------------------------------------
# Pollutant trends
# ---------------------------------------------------------------------------
if display_pollutants_over_time:
    st.header("Pollutant Trends")
    st.write("Monitor pollutant concentrations over time for a selected city.")

    trend_city = st.selectbox("Select a city for pollutant trends", selected_cities)
    city_data = target_data[target_data["City"] == trend_city].sort_values("Date")

    available_pollutants = [
        pollutant for pollutant in POLLUTANT_ORDER if pollutant in city_data.columns
    ]
    selected_pollutants = st.multiselect(
        "Select pollutants",
        available_pollutants,
        default=[p for p in ["PM2.5", "PM10"] if p in available_pollutants],
    )

    if selected_pollutants:
        for unit, unit_pollutants in UNIT_GROUPS.items():
            chosen = [p for p in selected_pollutants if p in unit_pollutants]
            if not chosen:
                continue

            pollutant_data = (
                city_data.set_index("Date")[chosen]
                .apply(pd.to_numeric, errors="coerce")
                .dropna(how="all")
            )

            st.subheader(f"Concentrations in {unit} — {trend_city}")
            st.line_chart(pollutant_data)
            st.caption(
                "Pollutants are grouped by unit so lines on the same chart are "
                "directly comparable. Missing measurements are skipped."
            )
    else:
        st.warning("Select at least one pollutant to display the chart.")
    st.divider()

# ---------------------------------------------------------------------------
# City map
# ---------------------------------------------------------------------------
if display_city_map:
    st.header("City Map")
    st.write("Average AQI for each city in the filtered dataset.")

    map_data = city_average_frame(filtered)
    if not map_data.empty:
        st.map(
            map_data,
            latitude="lat",
            longitude="lon",
            size="Average AQI",
            color="#ef4444",
            zoom=4,
            width="stretch",
        )
        st.dataframe(
            map_data[["City", "Average AQI", "lat", "lon"]].style.format(
                {"Average AQI": "{:.1f}", "lat": "{:.4f}", "lon": "{:.4f}"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Marker size reflects average AQI. Coordinates are approximate "
            "city centres, not monitoring-station locations."
        )
    else:
        st.warning("No map coordinates are available for the selected cities.")
    st.divider()

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.header("Data export")
st.write("Download the filtered records or inspect them in the table below.")

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data as CSV",
    data=csv_bytes,
    file_name=f"airwatch_{start_date}_{end_date}.csv",
    mime="text/csv",
    width="stretch",
)

with st.expander("Preview filtered records"):
    st.dataframe(filtered, width="stretch", height=320)

st.divider()
st.subheader("🌫️ AirWatch India")
st.caption(
    "An air quality awareness and data visualization project. "
    "Built with Streamlit, Pandas, and scikit-learn."
)
