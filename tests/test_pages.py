from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_page(relative_path: str, timeout: int = 120) -> AppTest:
    app = AppTest.from_file(str(PROJECT_ROOT / relative_path), default_timeout=timeout)
    app.run()
    return app


@pytest.mark.parametrize(
    "page",
    [
        "pages/home.py",
        "pages/dashboard.py",
        "pages/prediction.py",
    ],
)
def test_pages_render_without_exceptions(page):
    app = run_page(page)
    assert not app.exception, [exc.value for exc in app.exception]
    assert not app.error, [err.value for err in app.error]


def test_entrypoint_renders_without_exceptions():
    app = run_page("Air.py")
    assert not app.exception, [exc.value for exc in app.exception]
    assert not app.error, [err.value for err in app.error]


def test_prediction_form_estimates_aqi():
    app = run_page("pages/prediction.py")
    app.number_input(key="pollutant_PM2.5").set_value(80.0)
    app.number_input(key="pollutant_PM10").set_value(140.0)
    app.form_submit_button[0].click()
    app.run()

    assert not app.exception, [exc.value for exc in app.exception]
    assert not app.error, [err.value for err in app.error]
    assert any(metric.label == "Estimated CPCB AQI" for metric in app.metric)
    assert any("pollutant inputs were left blank" in info.value for info in app.info)
