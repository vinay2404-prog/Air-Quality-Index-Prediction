"""Shared domain logic for the AirWatch India Streamlit app.

Keeping this logic outside ``pages/`` makes it reusable by the training
script, the Streamlit pages, and the test-suite.
"""

__all__ = [
    "constants",
    "data",
    "india_aqi",
    "live",
    "models",
]
