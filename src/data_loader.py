"""Data loading, provenance register and schema inspection helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from . import DATASET_GEOGRAPHY, DATASET_NAME, DATASET_PERIOD, N_RECORDS, RAW_DATA_PATH

REQUIRED_COLUMNS = [
    "accident_id", "city", "state", "latitude", "longitude", "date", "time",
    "hour", "day_of_week", "is_weekend", "road_type", "lanes", "traffic_signal",
    "weather", "visibility", "temperature", "traffic_density", "cause",
    "accident_severity", "vehicles_involved", "casualties", "is_peak_hour",
    "festival", "risk_score",
]


def data_source_register() -> dict[str, Any]:
    """Data-source register (provenance record) for the dataset in use.

    NOTE ON PROVENANCE: the master brief originally asked for the UK STATS19
    dataset. The execution environment has no verified download of that file,
    and the project was directed to use the dataset already present on disk:
    ``data/raw/indian_roads_dataset.csv``. Its characteristics (perfectly
    balanced city sizes, no missing values except ``festival``, severity
    frequencies nearly independent of every recorded condition, timestamps
    rounded to whole hours) strongly suggest a *synthetic / illustrative*
    file rather than an official government release. This is disclosed in the
    README, the About-the-Data page and the notebook; the modelling pipeline
    is written so that swapping in an official file (e.g. STATS19) only
    requires a new loader mapping.
    """
    return {
        "publisher": "Unknown / not an official statistical release",
        "dataset_name": DATASET_NAME,
        "description": (
            "Row-level traffic-accident records for 8 Indian cities "
            "(city, state, coordinates, date/time, road, weather, traffic "
            "conditions, severity label, casualties)."
        ),
        "geography": DATASET_GEOGRAPHY,
        "period": DATASET_PERIOD,
        "records": N_RECORDS,
        "source_path": str(RAW_DATA_PATH),
        "obtained_from": "Provided with the project workspace (bundled CSV).",
        "licence": "No licence file accompanies the CSV; treat as "
                   "project-internal illustrative data, do not redistribute.",
        "provenance_caveat": (
            "File properties indicate a synthetic/illustrative dataset, not an "
            "official government release. Findings demonstrate methodology, "
            "not measured real-world road-safety statistics."
        ),
        "register_date": str(date.today()),
        "files_used": [DATASET_NAME],
        "intended_official_alternative": (
            "UK Department for Transport STATS19 road collision data "
            "(https://www.gov.uk/government/collections/road-accidents-and-"
            "safety-statistics) - swap-in requires a column-mapping loader."
        ),
    }


def load_accidents(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw accident CSV and run basic integrity checks.

    Parameters
    ----------
    path:
        Optional explicit path; defaults to ``data/raw/indian_roads_dataset.csv``.

    Raises
    ------
    FileNotFoundError
        If the raw CSV is missing, with instructions on where to place it.
    ValueError
        If required columns are absent.
    """
    path = Path(path) if path is not None else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'.\n"
            "Place the accident CSV (indian_roads_dataset.csv) inside data/raw/. "
            "See data/README.md for details."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column schema/quality report as a DataFrame."""
    n = len(df)
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    })
    report["example"] = [
        repr(df[c].dropna().iloc[0]) if df[c].notna().any() else "ALL MISSING"
        for c in df.columns
    ]
    report.attrs["rows"] = n
    return report.sort_values("missing_pct", ascending=False)


def year_row_counts(df: pd.DataFrame) -> pd.Series:
    """Row counts per calendar year of the ``date`` column."""
    return pd.to_datetime(df["date"]).dt.year.value_counts().sort_index()


def duplicate_checks(df: pd.DataFrame) -> dict[str, int]:
    """Duplicate checks: fully identical rows and repeated accident ids."""
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_accident_ids": int(df["accident_id"].duplicated().sum()),
    }
