# Data directory

## `raw/`

Place the raw collision CSV here. The project expects:

```
data/raw/indian_roads_dataset.csv
```

This file is **not** committed to version control (see `../.gitignore`).
Keep the original untouched — every processed/derived table is written to
`outputs/` or held in memory.

## Provenance & licence

| Field | Value |
|---|---|
| Publisher | Unknown / not an official statistical release |
| Dataset name | `indian_roads_dataset.csv` |
| Geography | 8 cities, 7 states/UTs, India |
| Period | 2022-01-01 → 2025-04-15 |
| Records | 20,000 |
| Obtained from | Bundled with the project workspace |
| Licence | No licence file accompanies the CSV. Treat as project-internal illustrative data; do not redistribute. |
| Register date | See `models/metadata.json` → `dataset.source_register` |

### Provenance caveat (read this)

The brief originally specified the **UK Department for Transport STATS19**
road-collision dataset. That file could not be downloaded in this execution
environment, and the project was directed to use the CSV already present on
disk.

Inspection shows the bundled file has properties of a **synthetic /
illustrative** dataset:

* city sizes almost perfectly balanced (~2,500 rows each);
* zero missing values except `festival` (99.4 % missing);
* severity frequencies statistically independent of every recorded condition
  (weather, visibility, road type, city, hour …);
* timestamps rounded to whole hours;
* internally consistent derived columns (`hour` always matches `time`,
  `day_of_week` always matches `date`, `is_peak_hour` deterministic from
  `hour`).

**Therefore:** findings from this file demonstrate the *methodology*, not
measured real-world road-safety statistics. Do not cite the numbers as
official accident statistics.

### Swapping in official data (e.g. STATS19)

1. Download the annual CSVs (accidents, casualties, vehicles) from
   <https://www.gov.uk/government/collections/road-accidents-and-safety-statistics>
   under the Open Government Licence v3.0.
2. Place them in `data/raw/`.
3. Write a loader that maps the STATS19 columns onto the schema this project
   expects (see `docs/data_dictionary.md` for the target schema and
   `src/data_loader.py::REQUIRED_COLUMNS`).
4. Re-run the pipeline:

```bash
python -m src.train_severity
python -m src.train_hotspots
python -m src.build_metadata
python -m src.explainability
```

No other code changes are needed — preprocessing, modelling, evaluation,
hotspot logic and the app all read from the loader.

## Integrity rules honoured by this project

1. Data-source register maintained in `src/data_loader.py::data_source_register`
   and exported to `models/metadata.json`.
2. Columns, dtypes, row counts, missingness and category codes inspected
   before preprocessing (notebook section 5-6).
3. No value codes (`-1`, `9`, `99`, …) treated as missing without inspection —
   in this file no such sentinel codes exist.
4. Duplicates checked via `accident_id` and full-row hash
   (`src/data_loader.py::duplicate_checks`): 0 duplicates found.
5. No invented weather, coordinates, records, labels or scores.
6. No randomly generated records used for training or evaluation.
7. Raw file kept unchanged; processed data lives elsewhere.
8. Full reproducibility with `RANDOM_SEED = 42` in `src/__init__.py`.
