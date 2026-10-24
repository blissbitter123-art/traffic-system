# Traffic Safety Intelligence — Accident Risk Prediction & Hotspot Analysis

An intelligent traffic-analysis platform that identifies accident-prone
locations and predicts collision severity / risk categories from historical
accident records and environmental conditions.

**Three differentiators over a standard accident dashboard:**

1. **Time–Place Safety Twin (main USP)** — for any grid cell and month, the
   platform combines a *count forecast* with a *severity distribution* into a
   transparent, user-editable **expected-harm index**, side-by-side with
   scenario comparison and coverage warnings.
2. **Emerging Hotspot Early Warning** — deterministic rules classify each cell
   as Persistent / Emerging / Declining / Stable / Insufficient-evidence by
   comparing the last 3 months against the prior 12-month baseline.
3. **Explainable Intervention Studio** — global permutation importance plus
   local single-record ablation explanations, always framed as *model
   behaviour*, never as causal proof.

---

## Honest headline finding

On the bundled dataset the best severity model **does not beat the
class-prior baseline** on validation log-loss or macro-F1. Recorded conditions
(weather, road, time, place) carry no learnable severity signal in this file.
This is reported openly in `outputs/metrics.json`, the notebook, and the app
rather than hidden. The pipeline itself is fully functional and would be
re-run unchanged on official data (e.g. UK STATS19) that does contain signal.

---

## Project structure

```
traffic-safety-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── raw/indian_roads_dataset.csv   (local only, not version-controlled)
├── notebooks/
│   └── 01_traffic_safety_intelligence.ipynb
├── app.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train_severity.py
│   ├── train_hotspots.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── hotspot_analysis.py
│   ├── prediction.py
│   ├── visualization.py
│   └── build_metadata.py
├── models/
│   ├── severity_pipeline.joblib
│   ├── hotspot_count_pipeline.joblib
│   └── metadata.json
├── outputs/
│   ├── metrics.json
│   ├── classification_report.csv
│   ├── model_comparison.csv
│   ├── hotspot_cells.csv
│   ├── count_model_comparison.csv
│   └── permutation_importance.csv
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
└── docs/
    └── data_dictionary.md
```

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset

The raw file `data/raw/indian_roads_dataset.csv` (20,000 rows, 8 Indian
cities, 2022-01-01 → 2025-04-15) must be present. See `data/README.md` for
provenance, licence, and how to swap in the official UK STATS19 dataset.

**Important provenance caveat:** the bundled CSV shows properties of a
synthetic/illustrative file (near-uniform severity across every recorded
condition, perfectly balanced city sizes, hourly-rounded timestamps). It is
**not** an official government release. All findings demonstrate methodology,
not measured real-world road-safety statistics.

## Running the full pipeline

```bash
# 1. Train severity models (train 2022-23, validate 2024, test 2025-partial)
python -m src.train_severity

# 2. Build hotspot grid + monthly panel + count forecast
python -m src.train_hotspots

# 3. Write the combined metadata manifest
python -m src.build_metadata

# 4. Generate permutation-importance table
python -m src.explainability

# 5. Run the automated tests
python -m pytest tests -q

# 6. Launch the demo
streamlit run app.py
```

## Deploying on Render

This repository includes a Render Blueprint at `render.yaml`. In Render, choose
**New > Blueprint**, connect this GitHub repository, and deploy the Blueprint.
It creates a free Python web service and runs Streamlit on Render's assigned
port. The free service may spin down when inactive.

The raw dataset is intentionally excluded from Git because it has no
redistribution licence. The deployed dashboard and saved model pages can use
the checked-in outputs and model artifacts, but the Time-Place Safety Twin's
count forecast needs `data/raw/indian_roads_dataset.csv`; that dataset-dependent
forecast will be unavailable unless you provide the file through an approved,
private data source. Google Maps is optional and can be enabled by adding
`GOOGLE_MAPS_API_KEY` as a secret environment variable in the Render service.

Google Maps is optional. Set `GOOGLE_MAPS_API_KEY` in the environment to enable
its map; without it, the app displays the standard hotspot map instead. On
Windows PowerShell, set it for the current terminal with:

```powershell
$env:GOOGLE_MAPS_API_KEY = "your-key"
```

## Running the notebook

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/01_traffic_safety_intelligence.ipynb \
  --output 01_traffic_safety_intelligence.ipynb
```
or open it in Jupyter/VS Code and *Run All*.

---

## Validation summary (genuine, from executed runs)

### Severity model (chronological split)

| Model | Val macro-F1 | Val balanced acc | Val log-loss |
|---|---|---|---|
| prior (dummy) | 0.237 | 0.333 | 0.976 |
| stratified dummy | 0.334 | 0.334 | 21.02 |
| LogisticRegression | 0.305 | 0.333 | 1.100 |
| RandomForest | 0.305 | 0.328 | 1.039 |
| HistGradientBoosting | 0.238 | 0.334 | 0.980 |

Best real model (HGB) **does not clear the 0.01 margin** over the prior
baseline on either metric → `beats_prior_baseline: false` in
`outputs/metrics.json`. The fatal-class recall of every model is poor, which
is the honest consequence of the data carrying no signal.

### Count forecast (held-out 2024-10 → 2025-03, 46 eligible cells)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| cell historical mean | **2.50** | **3.27** | 23.7% |
| seasonal average | 2.87 | 3.70 | 27.4% |
| Poisson regression | 2.99 | 3.98 | 33.4% |
| previous month | 3.74 | 4.82 | 35.6% |

The learned Poisson model beats the naive previous-month baseline but not the
cell-mean baseline — expected on a flat synthetic series with no trend or
seasonality. On real data with genuine hot-spot dynamics we would expect the
learned model to win.

---

## Key design decisions

* **Leakage control:** `casualties`, `risk_score`, `cause`, `accident_id`
  (and optionally `vehicles_involved`) are excluded from predictors because
  they are post-outcome or post-investigation fields.
* **Chronological splits** everywhere — the 2025 test period is never used
  for model selection, threshold tuning, or preprocessing fitting.
* **No SMOTE.** Class weights (`class_weight="balanced"`) handle imbalance
  inside the training fold only.
* **Grid resolution 0.2°** (~22 km) → 46 eligible cells, ~3 collisions per
  cell-month: enough signal for monthly forecasting without sparse noise.
* **Coverage indicator** on every cell so a data-poor area is labelled
  "Sparse", never "safe".
* **Incomplete month 2025-04 (15 days only)** is excluded from every metric.

---

## Limitations

* Dataset appears synthetic; results demonstrate methodology, not real-world
  statistics.
* No traffic-exposure denominator → we report *collision intensity*, never
  absolute per-vehicle risk.
* Severity predictions are conditional on a reported collision; they are not
  proof a collision will occur.
* 2025 test period is partial (3.5 months).
* No live data feed — this is a historical-analysis prototype.

## Hackathon pitch (30 seconds)

*"Every accident dashboard shows the past. This platform predicts the
severity distribution of a reported collision, forecasts next month's
collision count per neighbourhood, and — the Time–Place Safety Twin — fuses
both into one transparent expected-harm index you can interrogate and
re-weight live. It flags emerging hotspots before they become persistent
ones, and explains every model output. We validated on a held-out future
period, disclosed honestly that this particular dataset carries no severity
signal, and built the pipeline so that plugging in official STATS19 data
makes it production-relevant."*

## Live-demo sequence (~3 min)

1. **Overview** — show the model-status panel and the honest baseline note.
2. **Hotspot Explorer** — open the map, hover a Persistent-hotspot circle,
   show the coverage band column.
3. **Emerging Hotspots** — point at the one Emerging cell and the rule table.
4. **Severity Predictor** — run one scenario, show the probability bars.
5. **Time–Place Safety Twin** — pick the emerging cell, adjust the fatal
   weight, recompute the expected-harm index live.
6. **Explainability** — run the local ablation, show that importances are ≈0,
   which *is* the honest story of this dataset.

## Licence & attribution

Bundled CSV: no licence file provided — treat as project-internal illustrative
data. Intended official alternative: UK Department for Transport STATS19
(Open Government Licence v3.0). See `data/README.md`.
