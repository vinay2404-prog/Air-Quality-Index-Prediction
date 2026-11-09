"""Clean ``Data/city_day_raw.csv`` and write ``Data/city_day.csv``.

Run from the project root:

    python scripts/clean_data.py

The raw Kaggle download is kept unchanged in ``Data/city_day_raw.csv`` so the
cleaning is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aqi_app.cleaning import clean_dataset  # noqa: E402

RAW_PATH = PROJECT_ROOT / "Data" / "city_day_raw.csv"
CLEAN_PATH = PROJECT_ROOT / "Data" / "city_day.csv"
REPORT_PATH = PROJECT_ROOT / "Data" / "cleaning_report.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not RAW_PATH.is_file():
        raise SystemExit(
            f"Raw dataset not found: {RAW_PATH}\n"
            "Restore the original Kaggle file as Data/city_day_raw.csv."
        )

    raw = pd.read_csv(RAW_PATH, parse_dates=["Date"])
    cleaned, report = clean_dataset(raw)
    cleaned.to_csv(CLEAN_PATH, index=False)

    report.update(
        {
            "source": RAW_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "output": CLEAN_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "raw_sha256": file_sha256(RAW_PATH),
            "clean_sha256": file_sha256(CLEAN_PATH),
        }
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Raw rows:    {report['rows_before']:,}")
    print(f"Cleaned rows:{report['rows_after']:,}")
    for key, value in report.items():
        if isinstance(value, int):
            print(f"  {key}: {value:,}")
    print(f"Wrote {CLEAN_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
