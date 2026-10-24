"""Traffic Safety Intelligence — shared configuration and paths.

Everything that both the training scripts, the notebook and the Streamlit app
need to agree on (random seed, file locations, split dates, severity weights,
spatial resolution) lives here so there is exactly one source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "indian_roads_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SEVERITY_PIPELINE_PATH = MODELS_DIR / "severity_pipeline.joblib"
HOTSPOT_PIPELINE_PATH = MODELS_DIR / "hotspot_count_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# --- Dataset facts (verified by inspection, see docs/data_dictionary.md) ---
DATASET_NAME = "indian_roads_dataset.csv"
DATASET_PERIOD = "2022-01-01 to 2025-04-15"
DATASET_GEOGRAPHY = "8 cities across 7 states/UTs of India"
N_RECORDS = 20_000

# --- Chronological splits --------------------------------------------------
# Severity model: train on 2022-2023, validate on 2024, final test on 2025
# (2025 is a partial year: 2025-01-01 to 2025-04-15).
SEVERITY_TRAIN_END = "2023-12-31"
SEVERITY_VAL_END = "2024-12-31"  # test = dates after this

# Count model: train months 2022-01 .. 2024-09, forecast-evaluate on the six
# complete months 2024-10 .. 2025-03. 2025-04 is excluded everywhere from
# evaluation because it only covers 15 days of data.
COUNT_TRAIN_END_MONTH = "2024-09"
COUNT_TEST_START_MONTH = "2024-10"
COUNT_TEST_END_MONTH = "2025-03"
PANEL_END_MONTH = "2025-03"  # last complete month in the study period

# --- Severity task ---------------------------------------------------------
# Order is fixed everywhere (training, app, harm index).
SEVERITY_CLASSES = ["minor", "major", "fatal"]

# Transparent prioritisation weights for the expected-harm index.
# These are a *policy* choice, not a measured monetary or human cost.
SEVERITY_WEIGHTS = {"minor": 1.0, "major": 4.0, "fatal": 10.0}

# Columns that must never be used as predictors (leakage / post-outcome).
LEAKAGE_COLUMNS = [
    "accident_id",       # identifier, not a condition
    "casualties",        # post-collision outcome
    "risk_score",        # pre-computed score strongly tied to the label
    "cause",             # assigned after investigation, not known at prediction time
    "accident_severity",  # the target itself
]

# Kept out of the model as well: vehicles involved is knowable at the scene but
# is an immediate collision property rather than an environmental condition; the
# platform's stated scenario is "conditions-only" prediction.
OPTIONAL_EXCLUDED_COLUMNS = ["vehicles_involved"]

# --- Hotspot / count task --------------------------------------------------
CELL_SIZE_DEG = 0.2  # ~22 km lat; documented grid resolution
MIN_TRAIN_COLLISIONS_PER_CELL = 20   # eligibility for forecasting
MIN_OBSERVED_MONTHS_PER_CELL = 12    # eligibility for forecasting

# Relative collision-intensity bands (counts per cell-month), derived from the
# training-period distribution in hotspot_analysis.intensity_thresholds().
INTENSITY_LABELS = ["Low", "Moderate", "High"]

# --- Emerging-hotspot rules ------------------------------------------------
EMERGING_RECENT_MONTHS = 3
EMERGING_BASELINE_MONTHS = 12
EMERGING_RISING_RATIO = 1.5   # recent mean / baseline mean -> "Emerging"
EMERGING_DECLINING_RATIO = 0.7
PERSISTENT_HIGH_SHARE = 0.5   # share of recent months in High band -> "Persistent"
MIN_COUNTS_FOR_TREND = 5      # minimum collisions in baseline window to trust a ratio


def ensure_dirs() -> None:
    """Create the project's output directories if they do not exist yet."""
    for d in (MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
