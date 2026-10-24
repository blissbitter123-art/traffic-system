# Data dictionary — `indian_roads_dataset.csv`

Target schema expected by `src/data_loader.py::REQUIRED_COLUMNS`.
Geography: 8 Indian cities. Period: 2022-01-01 → 2025-04-15. Rows: 20,000.

## Columns

| Column | Type | Missing | Meaning | Used as predictor? |
|---|---|---|---|---|
| `accident_id` | int | 0 | Unique row identifier (0…19999). Verified unique. | **No** — identifier |
| `city` | str | 0 | City name (8 values: Bangalore, Chandigarh, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Pune) | Yes (categorical) |
| `state` | str | 0 | State / union territory (7 values) | Yes (categorical) |
| `latitude` | float | 0 | WGS84 latitude, range 12.80–30.80. All within India bbox. | Yes (numeric) |
| `longitude` | float | 0 | WGS84 longitude, range 72.70–88.50. All within India bbox. | Yes (numeric) |
| `date` | str (YYYY-MM-DD) | 0 | Collision date. Verified consistent with `day_of_week`. | Derived → `month`, `season`, `weekday_num` |
| `time` | str (H:MM) | 0 | Collision time, rounded to the hour in this file. Verified consistent with `hour`. | Derived → validated against `hour` |
| `hour` | int 0–23 | 0 | Hour of day. Verified equal to `time`'s hour in every row. | Yes (numeric) |
| `day_of_week` | str | 0 | Weekday name. Verified equal to `date`'s weekday in every row. | Yes (categorical) |
| `is_weekend` | int {0,1} | 0 | 1 if Saturday/Sunday. Verified consistent with `date`. | Yes (categorical) |
| `road_type` | str | 0 | `urban` / `highway` / `rural` | Yes (categorical) |
| `lanes` | int 1–6 | 0 | Number of lanes at the location | Yes (numeric) |
| `traffic_signal` | int {0,1} | 0 | Traffic signal present at the location | Yes (categorical) |
| `weather` | str | 0 | `clear` / `rain` / `fog` | Yes (categorical) |
| `visibility` | str | 0 | `high` / `medium` / `low` | Yes (categorical) |
| `temperature` | int 15–40 | 0 | Ambient temperature °C | Yes (numeric) |
| `traffic_density` | str | 0 | `low` / `medium` / `high` | Yes (categorical) |
| `cause` | str | 0 | `weather` / `distraction` / `overspeeding` / `drunk driving` / `poor road` — **assigned after investigation** | **No** — post-investigation, leaks outcome knowledge |
| `accident_severity` | str | 0 | **Target.** `minor` (11 025) / `major` (5 988) / `fatal` (2 987) | **Target** |
| `vehicles_involved` | int 1–5 | 0 | Vehicles involved. Knowable at the scene but an immediate collision property, not an environmental condition. | Excluded by default (see `OPTIONAL_EXCLUDED_COLUMNS`) |
| `casualties` | int 0–5 | 0 | People harmed. Strongly determines `accident_severity` (fatal mean 3.0 vs 1.5). | **No** — post-outcome leakage |
| `is_peak_hour` | int {0,1} | 0 | 1 at hours 8–10 and 17–19 (verified deterministic from `hour`) | Yes (categorical) |
| `festival` | str | **19 885 (99.4 %)** | Festival name (`Diwali`, `Holi`, `Eid`, `New Year`) when recorded; NaN = no festival recorded → filled with `"none"` | Yes (categorical) |
| `risk_score` | float 0.1–1.0 | 0 | Pre-computed score; mean 0.61 for fatal vs 0.41 for major/minor — strongly tied to the label. | **No** — target proxy / leakage |

## Target class frequencies

| Severity | Count | Share |
|---|---|---|
| minor | 11 025 | 55.1 % |
| major | 5 988 | 29.9 % |
| fatal | 2 987 | 14.9 % |

## Missing-value handling

* `festival` NaN → `"none"` (an *observed* state: no festival recorded — not
  an imputed measurement).
* No other column has missing values in this file. The preprocessing pipeline
  nevertheless includes median/most-frequent imputers so that a swapped-in
  official dataset with gaps will not crash inference.

## Invalid / out-of-range value checks performed

* Coordinates checked against the India bounding box (6–38 °N, 68–98 °E) —
  0 violations.
* `hour` cross-checked against `time` — 0 mismatches.
* `day_of_week` cross-checked against `date` — 0 mismatches.
* `is_weekend` cross-checked against `date` — 0 mismatches.
* Duplicate `accident_id` — 0. Fully duplicated rows — 0.
* `2025-04` covers only 15 days; excluded from every metric and panel.

## Leakage policy (why columns are excluded)

A feature is admitted only if it would be known when a collision is
*reported*, before outcomes are known.

| Excluded column | Reason |
|---|---|
| `accident_id` | Identifier, not a condition |
| `casualties` | Post-collision outcome; near-deterministically reveals severity |
| `risk_score` | Pre-computed score strongly correlated with the label |
| `cause` | Determined by post-collision investigation, not available at prediction time |
| `accident_severity` | The target itself |
| `vehicles_involved` | Immediate collision property; excluded to keep the scenario purely environmental (configurable) |
