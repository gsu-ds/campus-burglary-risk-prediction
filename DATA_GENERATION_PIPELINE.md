# Data Generation Pipeline Documentation

## Overview

This document describes the complete pipeline for generating the "Core Atlanta Burglary-Related Crimes (2021-2025)" benchmark dataset. The pipeline is fully reproducible and extensible.

## Architecture

```
┌─────────────────┐
│  Raw Data       │
│  Sources        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  1. Ingestion   │  ← atl_model_pipelines/ingestion/
│  - APD data     │
│  - Weather data │
│  - Spatial data │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Transform   │  ← atl_model_pipelines/transform/
│  - NPU agg      │
│  - Time features│
│  - Lags/rolling │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Validate    │  ← atl_model_pipelines/validate/
│  - Quality check│
│  - Data leakage │
│  - Completeness │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Dataset  │
│  (Parquet/CSV)  │
└─────────────────┘
```

## Prerequisites

### System Requirements
- Python 3.11+
- 8GB RAM minimum
- 10GB disk space

### Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- pandas >= 2.0
- geopandas >= 0.14
- scikit-learn >= 1.3
- requests (for API calls)

## Step-by-Step Reproduction

### Step 1: Clone Repository

```bash
git clone https://github.com/gsu-ds/campus-burglary-risk-prediction.git
cd campus-burglary-risk-prediction
```

### Step 2: Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure Paths

Edit `config.py` to set data directories:

```python
# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
```

### Step 4: Run Ingestion Pipeline

**Purpose**: Download and ingest raw data sources

```bash
python -m atl_model_pipelines.ingestion.ingestion_master
```

**What it does**:
1. Downloads APD crime data (2021-2025)
2. Fetches Open-Meteo weather data
3. Loads NPU shapefiles
4. Geocodes university campuses

**Outputs**:
- `data/raw/apd_crimes_2021_2025.csv`
- `data/raw/weather_atlanta_2021_2025.parquet`
- `data/raw/npu_boundaries.geojson`

**Data Sources**:

| Source | URL | License |
|--------|-----|---------|
| APD Crime Data | https://www.atlantapd.org/i-want-to/crime-data-downloads | Public Domain |
| Weather | https://open-meteo.com/en/docs/historical-weather-api | CC BY 4.0 |
| NPU Boundaries | City of Atlanta Open Data | Public Domain |

### Step 5: Run Transform Pipeline

**Purpose**: Engineer features and aggregate to hourly NPU level

```bash
python -m atl_model_pipelines.transform.transform_master
```

**What it does**:

**5.1. Spatial Aggregation**
```python
# Aggregate crimes to NPU-hour level
def aggregate_to_npu_hour(crime_df, npu_boundaries):
    # Spatial join crimes to NPUs
    gdf = gpd.GeoDataFrame(
        crime_df,
        geometry=gpd.points_from_xy(crime_df.lon, crime_df.lat),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(gdf, npu_boundaries, how="left", predicate="within")
    
    # Aggregate to hourly counts
    hourly = (
        joined
        .groupby(["npu", pd.Grouper(key="datetime", freq="H")])
        .size()
        .reset_index(name="burglary_count")
    )
    return hourly
```

**5.2. Temporal Features**
```python
def add_temporal_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    return df
```

**5.3. Spatial Features**
```python
def add_spatial_features(df, campus_locations):
    # Distance to nearest campus
    def min_distance(npu_centroid, campuses):
        return min(npu_centroid.distance(c) for c in campuses)
    
    df["campus_distance_m"] = df["npu"].apply(
        lambda x: min_distance(get_npu_centroid(x), campus_locations)
    )
    
    # 7-day rolling density
    df["grid_density_7d"] = (
        df.groupby("npu")["burglary_count"]
        .rolling(7*24, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
    
    return df
```

**5.4. Lagged Features**
```python
def add_lagged_features(df):
    lags = [1, 3, 6, 12, 24, 168]  # hours
    
    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby("npu")["burglary_count"]
            .shift(lag)
        )
    
    # Drop rows with NaN lags (first week per NPU)
    df = df.dropna(subset=[f"lag_{l}" for l in lags])
    
    return df
```

**5.5. Weather Merge**
```python
def merge_weather(df, weather_df):
    # Round to nearest hour
    weather_df["hour_ts"] = weather_df["datetime"].dt.floor("H")
    
    merged = df.merge(
        weather_df[["hour_ts", "temp_f", "precipitation"]],
        left_on="datetime",
        right_on="hour_ts",
        how="left"
    )
    
    # Engineer weather features
    merged["is_raining"] = (merged["precipitation"] > 0).astype(int)
    merged["is_hot"] = (merged["temp_f"] > 85).astype(int)
    merged["is_cold"] = (merged["temp_f"] < 40).astype(int)
    
    return merged
```

**Outputs**:
- `data/processed/npu_dense_panel.parquet` (full features)
- `data/processed/npu_sparse_panel.parquet` (minimal features)
- `data/processed/target_crimes.parquet` (for modeling)

### Step 6: Run Validation Pipeline

**Purpose**: Quality assurance and data integrity checks

```bash
python -m atl_model_pipelines.validate.orchestrator
```

**What it checks**:

```python
def validate_dataset(df):
    checks = {
        "no_missing_npu": df["npu"].notna().all(),
        "no_missing_datetime": df["datetime"].notna().all(),
        "temporal_continuity": check_hourly_continuity(df),
        "no_data_leakage": check_lag_integrity(df),
        "valid_counts": (df["burglary_count"] >= 0).all(),
        "weather_coverage": (df["temp_f"].notna().sum() / len(df)) > 0.95,
    }
    
    for check, passed in checks.items():
        print(f"{check}: {'✓' if passed else '✗'}")
    
    return all(checks.values())
```

**Outputs**:
- `data/processed/validation_report.json`
- Raises errors if validation fails

### Step 7: Export Final Dataset

```bash
# Export to CSV for Kaggle
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/npu_dense_panel.parquet')
df.to_csv('data/final/atlanta_burglary_hourly_npu.csv', index=False)
print(f'Exported {len(df):,} rows')
"
```

## Pipeline Configuration

### Key Parameters

Edit `config.py` to customize:

```python
# Temporal scope
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

# Spatial scope
TARGET_NPUS = ["N", "M", "D", "E", "F"]  # Or None for all

# Feature engineering
LAG_HOURS = [1, 3, 6, 12, 24, 168]
ROLLING_WINDOWS = [7, 14, 30]  # days

# Validation thresholds
MAX_MISSING_PCT = 0.01
MIN_WEATHER_COVERAGE = 0.95
```

## Extending the Pipeline

### Adding New Features

**Example: Add day-of-year feature**

Edit `atl_model_pipelines/transform/temporal_features.py`:

```python
def add_temporal_features(df):
    # ... existing features ...
    
    # Add day of year
    df["day_of_year"] = df["datetime"].dt.dayofyear
    
    # Cyclical encoding
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    
    return df
```

### Adding New Data Sources

**Example: Add demographic data**

1. Create ingestion script:
```python
# atl_model_pipelines/ingestion/demographics.py
def ingest_demographics():
    # Download census data
    url = "https://api.census.gov/..."
    df = pd.read_csv(url)
    
    # Save to raw
    df.to_parquet(RAW_DIR / "demographics.parquet")
    return df
```

2. Register in master:
```python
# atl_model_pipelines/ingestion/ingestion_master.py
def run_ingestion():
    # ... existing ingestion ...
    demographics = ingest_demographics()
    return {"demographics": demographics, ...}
```

3. Add merge logic in transforms

## Troubleshooting

### Common Issues

**Issue**: Missing NPU assignments
```bash
# Check spatial join
python -c "
df = pd.read_csv('data/raw/apd_crimes_2021_2025.csv')
missing = df[df['npu'].isna()]
print(f'{len(missing)} crimes missing NPU')
print(missing[['lat', 'lon']].head())
"
```

**Solution**: Crimes outside NPU boundaries. Either filter or assign to nearest NPU.

**Issue**: Weather data gaps
```bash
# Check coverage
python -c "
weather = pd.read_parquet('data/raw/weather_atlanta_2021_2025.parquet')
print(f'Coverage: {weather.notna().mean()} ')
"
```

**Solution**: Open-Meteo has hourly data. Check date range matches.

## Performance Tips

### Memory Optimization
```python
# Use chunked processing for large datasets
chunksize = 100_000
for chunk in pd.read_csv(path, chunksize=chunksize):
    process_chunk(chunk)
```

### Parallel Processing
```python
from joblib import Parallel, delayed

npus = df["npu"].unique()
results = Parallel(n_jobs=-1)(
    delayed(process_npu)(df[df["npu"] == npu])
    for npu in npus
)
```

## Validation Checklist

Before publishing dataset:

- [ ] All validation checks pass
- [ ] No PII in data
- [ ] Temporal continuity verified
- [ ] No data leakage (future → past)
- [ ] Spatial coverage complete
- [ ] Weather data merged correctly
- [ ] Lagged features computed properly
- [ ] CSV exports without errors
- [ ] Documentation complete
- [ ] Example code tested

## Contact & Support

**Issues**: https://github.com/gsu-ds/campus-burglary-risk-prediction/issues  
**Questions**: joshuadariuspina@gmail.com

## References

1. Atlanta Police Department. (2025). Crime Data Downloads.
2. Open-Meteo. (2025). Historical Weather API.
3. City of Atlanta. (2025). NPU Boundaries Open Data.