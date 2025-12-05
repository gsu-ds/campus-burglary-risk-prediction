# Data Dictionary

**Dataset**: Core Atlanta Burglary-Related Crimes (2021-2025)  
**Version**: 1.0.0  
**Last Updated**: December 2025

This document describes all variables in the processed dataset files.

---

## Files Overview

| File | Records | Columns | Description |
|------|---------|---------|-------------|
| `target_crimes.parquet` | 117,749 | 64 | Raw incident-level data (burglary/larceny reports) |
| `target_crimes_panel.parquet` | 99,965 | 52 | Hourly NPU panel with lagged features |
| `npu_sparse_panel.parquet` | 99,965 | 46 | Hourly NPU panel without lags (only hours with incidents) |
| `npu_dense_panel.parquet` | 1,074,500 | 31-46 | Complete hourly NPU grid (all NPU-hour combinations) |
| `all_apd_crimes.parquet` | 265,990 | 70 | All APD crimes before filtering to target crimes |

**Panel Creation Flow**: 
1. Raw APD data → `all_apd_crimes` (265,990 incidents)
2. Filter to burglary/larceny → `target_crimes` (117,749 incidents)
3. Aggregate to hourly + add lags → `target_crimes_panel` (99,965 NPU-hours)
4. Remove lags → `sparse_panel` (99,965 NPU-hours, observed only)
5. Fill complete grid → `dense_panel` (1,074,500 NPU-hours, all combinations)

**Note**: Dense panel has fewer columns (31) because some high-cardinality features are excluded to reduce file size.

---

## Core Identifiers

### Temporal Identifiers

| Variable | Type | Description | Example | Notes |
|----------|------|-------------|---------|-------|
| `hour_ts` | datetime | Hour timestamp (UTC) | `2024-01-15 14:00:00` | Rounded to nearest hour |
| `datetime` | datetime | Alias for hour_ts | `2024-01-15 14:00:00` | Used interchangeably |
| `report_date` | datetime | Original incident report datetime | `2024-01-15 14:23:45` | Before aggregation (in target_crimes only) |
| `date` | date | Calendar date | `2024-01-15` | Derived from hour_ts |

### Spatial Identifiers

| Variable | Type | Description | Example | Notes |
|----------|------|-------------|---------|-------|
| `npu` | string | Neighborhood Planning Unit ID | `"N"` | A-Z (25 total) |
| `zone` | string | APD Police Zone | `"Zone 5"` | 6 zones total |
| `beat` | string | APD Beat | `"502"` | Finer than zone |
| `neighborhood` | string | Neighborhood name | `"Downtown"` | City-defined neighborhoods |

---

## Target Variable

| Variable | Type | Description | Range | Notes |
|----------|------|-------------|-------|-------|
| `burglary_count` | integer | Hourly burglary/larceny incident count | 0-8 | ~85% are zeros |
| `crime_count` | integer | Alias for burglary_count | 0-8 | Used in sparse/dense panels |

**UCR Codes Included**:
- **220**: Burglary/Breaking & Entering
- **23A-23H**: Larceny/Theft (all subcategories)

---

## Temporal Features

### Basic Time Features

| Variable | Type | Description | Range | Notes |
|----------|------|-------------|-------|-------|
| `hour` | integer | Hour of day | 0-23 | 0 = midnight |
| `day_of_week` | integer | Day of week | 0-6 | 0=Monday, 6=Sunday |
| `month` | integer | Month | 1-12 | January=1 |
| `year` | integer | Year | 2021-2025 | Calendar year |
| `day_of_year` | integer | Day of year | 1-366 | Julian day |
| `day_number` | integer | Days since epoch | varies | Continuous time measure |

### Cyclical Encodings

| Variable | Type | Description | Range | Formula |
|----------|------|-------------|-------|---------|
| `hour_sin` | float | Hour sine encoding | -1 to 1 | `sin(2π * hour / 24)` |
| `hour_cos` | float | Hour cosine encoding | -1 to 1 | `cos(2π * hour / 24)` |
| `day_sin` | float | Day-of-week sine | -1 to 1 | `sin(2π * dow / 7)` |
| `day_cos` | float | Day-of-week cosine | -1 to 1 | `cos(2π * dow / 7)` |

**Why cyclical encoding?** Captures circular nature of time (23:00 is close to 00:00, Sunday is close to Monday).

### Time Indicators

| Variable | Type | Description | Values | Notes |
|----------|------|-------------|--------|-------|
| `is_weekend` | boolean | Weekend indicator | 0, 1 | 1 if Saturday or Sunday |
| `is_holiday` | boolean | Federal holiday | 0, 1 | US federal holidays |
| `is_daylight` | boolean | Daylight hours | 0, 1 | Based on sunrise/sunset |

### Time Blocks

| Variable | Type | Description | Values | Notes |
|----------|------|-------------|--------|-------|
| `time_block` | string | Time of day category | See below | Human-readable |
| `time_block_label` | string | Alias for time_block | See below | Used in some files |
| `time_block_code` | integer | Numeric encoding | 0-5 | For modeling |

**Time Block Categories**:
- `"Late Night (0-4)"` — 12am-4am
- `"Early Morning (5-8)"` — 5am-8am
- `"Late Morning (9-12)"` — 9am-12pm
- `"Afternoon (13-16)"` — 1pm-4pm
- `"Evening (17-20)"` — 5pm-8pm
- `"Late Night (21-24)"` — 9pm-11pm

---

## Weather Features

| Variable | Type | Description | Range | Source |
|----------|------|-------------|-------|--------|
| `temp_f` | float | Temperature (Fahrenheit) | -10 to 110 | Open-Meteo API |
| `temperature` | float | Alias for temp_f | -10 to 110 | Used interchangeably |
| `precipitation` | float | Precipitation (inches) | 0 to 5 | Hourly accumulation |
| `is_raining` | boolean | Rain indicator | 0, 1 | 1 if precipitation > 0 |
| `is_hot` | boolean | Hot weather | 0, 1 | 1 if temp > 85°F |
| `is_cold` | boolean | Cold weather | 0, 1 | 1 if temp < 40°F |

**Missing Data**: <1% of hours, forward-filled from last known value.

---

## Spatial Features

### Campus Proximity

| Variable | Type | Description | Range | Notes |
|----------|------|-------------|-------|-------|
| `campus_distance_m` | float | Distance to nearest campus (meters) | 0-15000 | Euclidean distance |
| `gsu_distance` | float | Distance to Georgia State | 0-15000 | Meters |
| `gt_distance` | float | Distance to Georgia Tech | 0-15000 | Meters |
| `emory_distance` | float | Distance to Emory | 0-15000 | Meters |

**Campuses Included**:
- Georgia State University (Downtown)
- Georgia Institute of Technology (Midtown)
- Emory University (Druid Hills)
- Clark Atlanta University
- Morehouse College
- Spelman College
- Morehouse School of Medicine
- Atlanta Metropolitan State College
- Atlanta Technical College
- SCAD Atlanta
- John Marshall Law School

### Density & Context

| Variable | Type | Description | Calculation | Notes |
|----------|------|-------------|-------------|-------|
| `grid_density_7d` | float | 7-day rolling crime density | Mean count in NPU over 7 days | Captures recent activity |
| `npu_crime_avg_30d` | float | 30-day NPU average | Mean count over 30 days | Baseline risk |
| `location_type_count` | integer | Number of location types | 0-50 | Diversity of locations |

### Geographic

| Variable | Type | Description | Example | Notes |
|----------|------|-------------|---------|-------|
| `lat` | float | Latitude | 33.7490 | WGS84 / EPSG:4326 |
| `lon` | float | Longitude | -84.3880 | WGS84 / EPSG:4326 |
| `latitude` | float | Alias for lat | 33.7490 | Used in some files |
| `longitude` | float | Alias for lon | -84.3880 | Used in some files |

---

## Lagged Features

### Historical Crime Counts

| Variable | Type | Description | Notes |
|----------|------|-------------|-------|
| `lag_1` | integer | Crime count 1 hour ago | Most recent |
| `lag_3` | integer | Crime count 3 hours ago | |
| `lag_6` | integer | Crime count 6 hours ago | |
| `lag_12` | integer | Crime count 12 hours ago | Half day |
| `lag_24` | integer | Crime count 24 hours ago | Same time yesterday |
| `lag_168` | integer | Crime count 168 hours ago | Same time last week |

**Important**: Lagged features are computed per NPU. Missing values occur in the first week of data per NPU and are dropped.

**Data Leakage Prevention**: These features only use **past** information, never future data.

---

## Additional Features (Target Crimes File Only)

These variables appear in the raw `target_crimes.parquet` file before aggregation:

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `incident_number` | string | Unique incident ID | `202400012345` |
| `ucr_code` | string | Uniform Crime Reporting code | `220`, `23H` |
| `offense` | string | Offense description | `"BURGLARY"` |
| `location_type` | string | Location category | `"RESIDENCE"` |
| `address` | string | Incident address (obfuscated) | `"100 BLOCK PEACHTREE ST"` |

**Privacy**: Exact addresses are obfuscated to block-level in public data.

---

## Data Types Summary

| Type | Example Variables | Notes |
|------|-------------------|-------|
| **datetime** | hour_ts, report_date | Timezone-aware UTC |
| **integer** | burglary_count, hour, lags | Whole numbers |
| **float** | temp_f, distances, densities | Decimal precision |
| **boolean** | is_weekend, is_raining | 0 or 1 |
| **string** | npu, time_block, neighborhood | Categorical |

---

## Missing Data

| Variable | Missing % | Handling |
|----------|-----------|----------|
| `temp_f` | <1% | Forward-fill from last hour |
| `precipitation` | <1% | Forward-fill, assume 0 if unavailable |
| `lat`, `lon` | <0.1% | Dropped (incidents outside NPU boundaries) |
| `lag_*` | First 168 hours per NPU | Rows dropped (cannot compute) |

---

## Variable Naming Conventions

- **Snake_case**: All variable names use lowercase with underscores
- **Units in name**: When applicable (e.g., `_m` for meters, `_f` for Fahrenheit)
- **Prefixes**:
  - `is_` — Boolean indicators
  - `lag_` — Historical/lagged features
  - `npu_` — NPU-aggregated statistics
- **Suffixes**:
  - `_count` — Count variables
  - `_sin`, `_cos` — Cyclical encodings
  - `_avg` — Averages
  - `_7d`, `_30d` — Time windows

---

## Feature Engineering Notes

### Cyclical Encoding

Time features (hour, day of week) are encoded as sine/cosine pairs to capture their circular nature:

```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
```

This ensures the model understands that 23:00 is close to 00:00.

### Lagged Features

Lagged features are computed per NPU to capture temporal autocorrelation:

```python
df['lag_24'] = df.groupby('npu')['burglary_count'].shift(24)
```

### Rolling Statistics

Rolling features use a trailing window (past data only):

```python
df['grid_density_7d'] = (
    df.groupby('npu')['burglary_count']
    .rolling(7*24, min_periods=1)
    .mean()
)
```

---

## Quality Checks

All processed files pass these validation checks:

- ✓ No missing NPU assignments
- ✓ No duplicate hour-NPU combinations (dense panel)
- ✓ Temporal continuity (no gaps in hourly series)
- ✓ No future data leakage in lagged features
- ✓ Weather coverage >95%
- ✓ All counts ≥ 0
- ✓ Lat/lon within Atlanta bounds

---

## Example Row

**Dense Panel** (`npu_dense_panel.parquet`):

```python
{
    'hour_ts': '2024-06-15 14:00:00',
    'npu': 'N',
    'burglary_count': 2,
    'hour': 14,
    'day_of_week': 5,  # Friday
    'month': 6,
    'is_weekend': 0,
    'is_holiday': 0,
    'hour_sin': 0.866,
    'hour_cos': 0.5,
    'temp_f': 82.3,
    'is_raining': 0,
    'is_hot': 0,
    'is_cold': 0,
    'is_daylight': 1,
    'campus_distance_m': 450.2,
    'grid_density_7d': 0.31,
    'npu_crime_avg_30d': 0.28,
    'lag_1': 0,
    'lag_24': 1,
    'lag_168': 2,
    'time_block': 'Afternoon (13-16)'
}
```

---

## Usage Examples

### Load Data

```python
import pandas as pd

# Load dense panel
df = pd.read_parquet('data/processed/npu_dense_panel.parquet')

# Ensure datetime type
df['hour_ts'] = pd.to_datetime(df['hour_ts'])
```

### Filter to NPU

```python
# Get data for NPU N (Georgia State area)
npu_n = df[df['npu'] == 'N'].copy()
```

### Train/Test Split

```python
# Time-based split (never random!)
train = df[df['hour_ts'] < '2024-01-01']
test = df[df['hour_ts'] >= '2024-01-01']
```

### Feature Selection

```python
# Temporal features
temporal_features = [
    'hour', 'day_of_week', 'month',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_weekend', 'is_holiday'
]

# Spatial features
spatial_features = [
    'campus_distance_m', 'grid_density_7d', 
    'npu_crime_avg_30d'
]

# Weather features
weather_features = [
    'temp_f', 'is_raining', 'is_hot', 'is_cold'
]

# Lagged features
lag_features = [
    'lag_1', 'lag_3', 'lag_6', 'lag_12', 
    'lag_24', 'lag_168'
]

# All features
all_features = (
    temporal_features + 
    spatial_features + 
    weather_features + 
    lag_features
)
```

---

## Related Documentation

- **Dataset Card**: [DATASET_CARD.md](DATASET_CARD.md) — Comprehensive dataset documentation
- **Pipeline Guide**: [DATA_GENERATION_PIPELINE.md](DATA_GENERATION_PIPELINE.md) — Reproduction instructions
- **README**: [README.md](README.md) — Project overview

---

## Contact

Questions about variables or feature engineering:
- **GitHub Issues**: https://github.com/gsu-ds/campus-burglary-risk-prediction/issues
- **Email**: jpina4@student.gsu.edu

---

**Last Updated**: December 2025  
**Version**: 1.0.0