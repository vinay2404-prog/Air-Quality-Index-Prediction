from pathlib import Path

import streamlit as st

# set_page_config must be the first Streamlit call, and runs once for ALL pages.
st.set_page_config(
    page_title="AirWatch India",
    page_icon=":material/air:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES_DIR = Path(__file__).resolve().parent / "pages"

PAGE_SPECS = [
    # (file name, title, icon, is_default)
    ("home.py", "Home", ":material/home:", True),
    ("dashboard.py", "Dashboard", ":material/monitoring:", False),
    ("live.py", "Live AQI", ":material/sensors:", False),
    ("prediction.py", "Prediction", ":material/online_prediction:", False),
]

missing = [name for name, *_ in PAGE_SPECS if not (PAGES_DIR / name).is_file()]

if missing:
    st.error(f"Missing page file(s) in `{PAGES_DIR}`: {', '.join(missing)}")
    found = sorted(p.name for p in PAGES_DIR.glob("*.py")) if PAGES_DIR.is_dir() else []
    st.write("Files currently in the pages folder:")
    st.code("\n".join(found) if found else "(folder not found or empty)")
    st.info("Expected: " + ", ".join(name for name, *_ in PAGE_SPECS))
    st.stop()

pages = [
    st.Page(str(PAGES_DIR / name), title=title, icon=icon, default=default)
    for name, title, icon, default in PAGE_SPECS
]

navigation = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption(
        "Historical data: Kaggle · city_day.csv (2015–2020)\n\n"
        "Live data: Open-Meteo Air Quality API"
    )

navigation.run()
