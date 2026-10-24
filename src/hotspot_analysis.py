"""Geographic hotspot analysis: grid cells, monthly panels, intensity bands,
emerging-hotspot rules and data-coverage indicators.

Study-area definition (reproducible, documented)
------------------------------------------------
* Grid: fixed 0.2 deg x 0.2 deg cells (``CELL_SIZE_DEG``) anchored at 0/0,
  assigned by ``features.assign_grid_cells``.
* Panel universe: every cell that recorded **at least one collision during
  the count-model training window (2022-01 .. 2024-09)**, crossed with every
  **complete calendar month** from 2022-01 to 2025-03. Zero-count rows exist
  only inside this documented universe - no synthetic cells elsewhere.
* 2025-04 is excluded everywhere: the file only covers 15 days of that month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import (
    CELL_SIZE_DEG,
    COUNT_TEST_START_MONTH,
    COUNT_TRAIN_END_MONTH,
    EMERGING_BASELINE_MONTHS,
    EMERGING_DECLINING_RATIO,
    EMERGING_RECENT_MONTHS,
    EMERGING_RISING_RATIO,
    INTENSITY_LABELS,
    MIN_COUNTS_FOR_TREND,
    MIN_OBSERVED_MONTHS_PER_CELL,
    MIN_TRAIN_COLLISIONS_PER_CELL,
    PANEL_END_MONTH,
    PERSISTENT_HIGH_SHARE,
)
from .features import assign_grid_cells

PANEL_START_MONTH = "2022-01"


def prepare_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate coordinates, add ``cell_id`` and ``month`` (Period[str] YYYY-MM)."""
    out = df.copy()
    out["dt"] = pd.to_datetime(out["date"])
    out["month"] = out["dt"].dt.to_period("M").astype(str)
    out["cell_id"] = assign_grid_cells(out["latitude"], out["longitude"], CELL_SIZE_DEG)
    # Drop the incomplete final month from every hotspot/count calculation.
    out = out[out["month"] <= PANEL_END_MONTH].copy()
    return out


def build_monthly_panel(collisions: pd.DataFrame) -> pd.DataFrame:
    """Cell x month collision counts, including documented zero-count rows.

    Eligibility (training window only): a cell must have
    >= MIN_TRAIN_COLLISIONS_PER_CELL collisions and
    >= MIN_OBSERVED_MONTHS_PER_CELL months with >=1 collision in 2022-01..2024-09.
    Cells failing eligibility are excluded from the panel entirely (they are
    reported separately as 'insufficient data', never labelled 'safe').
    """
    train_mask = collisions["month"] <= COUNT_TRAIN_END_MONTH
    train = collisions[train_mask]

    stats = train.groupby("cell_id").agg(
        train_collisions=("accident_id", "size"),
        observed_months=("month", "nunique"),
    )
    eligible = stats[
        (stats["train_collisions"] >= MIN_TRAIN_COLLISIONS_PER_CELL)
        & (stats["observed_months"] >= MIN_OBSERVED_MONTHS_PER_CELL)
    ].index

    months = pd.period_range(PANEL_START_MONTH, PANEL_END_MONTH, freq="M").astype(str)
    index = pd.MultiIndex.from_product([sorted(eligible), months], names=["cell_id", "month"])

    counts = (
        collisions[collisions["cell_id"].isin(eligible)]
        .groupby(["cell_id", "month"])
        .size()
        .reindex(index, fill_value=0)
        .rename("collision_count")
        .reset_index()
    )
    counts["is_test_period"] = counts["month"] >= COUNT_TEST_START_MONTH
    counts["month_num"] = pd.PeriodIndex(counts["month"], freq="M").astype(int)
    return counts, stats.loc[eligible]


def cell_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-cell coverage indicator: months with data, share of panel months, band.

    A cell with few recorded months must be flagged - low coverage means
    'little data', NOT 'safe'.
    """
    cov = panel.groupby("cell_id").agg(
        panel_months=("month", "nunique"),
        months_with_collisions=("collision_count", lambda s: int((s > 0).sum())),
        total_collisions=("collision_count", "sum"),
        first_month=("month", "min"),
        last_month=("month", "max"),
    )
    cov["coverage_share"] = (cov["months_with_collisions"] / cov["panel_months"]).round(3)
    cov["coverage_band"] = pd.cut(
        cov["coverage_share"],
        bins=[-0.01, 0.5, 0.9, 1.01],
        labels=["Sparse", "Partial", "Complete"],
    )
    return cov.reset_index()


def intensity_thresholds(train_counts: np.ndarray) -> dict[str, float]:
    """Low/Moderate/High cut-points from the training-period count distribution.

    Cuts are the 40th and 80th percentiles of *observed* (incl. zero) cell-month
    counts in the training window - documented, reproducible, and derived from
    data rather than arbitrary labels.
    """
    q40, q80 = np.percentile(train_counts, [40, 80])
    return {"low_moderate_cut": float(q40), "moderate_high_cut": float(q80)}


def assign_intensity(count: float, cuts: dict[str, float]) -> str:
    """Map a cell-month count to Low / Moderate / High using training cuts."""
    if count <= cuts["low_moderate_cut"]:
        return "Low"
    if count <= cuts["moderate_high_cut"]:
        return "Moderate"
    return "High"


def add_intensity_labels(panel: pd.DataFrame, cuts: dict[str, float]) -> pd.DataFrame:
    out = panel.copy()
    out["intensity"] = out["collision_count"].map(lambda c: assign_intensity(c, cuts))
    return out


def cell_trends(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-cell recent-vs-baseline comparison used by the emerging-hotspot view.

    Recent window : last EMERGING_RECENT_MONTHS months of the panel.
    Baseline      : the EMERGING_BASELINE_MONTHS months before that window.
    Classification rules (documented, deterministic):
      * 'Insufficient evidence' - baseline collisions < MIN_COUNTS_FOR_TREND.
      * 'Persistent hotspot'    - >= PERSISTENT_HIGH_SHARE of recent months High.
      * 'Emerging hotspot'      - recent mean / baseline mean >= EMERGING_RISING_RATIO.
      * 'Declining'             - ratio <= EMERGING_DECLINING_RATIO.
      * 'Stable'                - everything else.
    """
    months_sorted = sorted(panel["month"].unique())
    recent_months = set(months_sorted[-EMERGING_RECENT_MONTHS:])
    baseline_months = set(months_sorted[-(EMERGING_RECENT_MONTHS + EMERGING_BASELINE_MONTHS):-EMERGING_RECENT_MONTHS])

    rows = []
    for cell_id, grp in panel.groupby("cell_id"):
        recent = grp[grp["month"].isin(recent_months)]["collision_count"]
        baseline = grp[grp["month"].isin(baseline_months)]["collision_count"]
        cuts = intensity_thresholds(grp["collision_count"].values)
        high_share = float((recent.map(lambda c: assign_intensity(c, cuts)) == "High").mean())
        base_sum, recent_sum = int(baseline.sum()), int(recent.sum())
        base_mean = float(baseline.mean()) if len(baseline) else 0.0
        recent_mean = float(recent.mean()) if len(recent) else 0.0

        if base_sum < MIN_COUNTS_FOR_TREND:
            label, ratio = "Insufficient evidence", np.nan
        elif high_share >= PERSISTENT_HIGH_SHARE:
            label, ratio = "Persistent hotspot", recent_mean / base_mean if base_mean else np.nan
        else:
            ratio = recent_mean / base_mean if base_mean else np.nan
            if ratio >= EMERGING_RISING_RATIO:
                label = "Emerging hotspot"
            elif ratio <= EMERGING_DECLINING_RATIO:
                label = "Declining"
            else:
                label = "Stable"

        rows.append({
            "cell_id": cell_id,
            "baseline_collisions": base_sum,
            "recent_collisions": recent_sum,
            "baseline_monthly_mean": round(base_mean, 2),
            "recent_monthly_mean": round(recent_mean, 2),
            "trend_ratio": round(float(ratio), 3) if ratio == ratio else np.nan,
            "recent_high_share": round(high_share, 2),
            "trend_label": label,
        })
    return pd.DataFrame(rows).sort_values("trend_ratio", ascending=False, na_position="last")
