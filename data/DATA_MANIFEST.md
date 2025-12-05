# Complete File Manifest

**Dataset**: Core Atlanta Burglary-Related Crimes (2021-2025)  
**Last Updated**: December 2025  
**Source**: Actual file analysis

---

## Raw Data Sources

### APD Crime Data (3 files, 372,864 total incidents)

| File | Records | Columns | Time Period | Format |
|------|---------|---------|-------------|--------|
| `apd_2020_2024.csv` | 220,767 | 26 | 2020-2024 | CSV |
| `apd_2023_2025.csv` | 135,961 | 28 | 2023-2025 | CSV |
| `apd_2021_2024.csv` | 16,136 | 26 | 2021-2024 | CSV |
| **Total** | **372,864** | - | 2020-2025 | - |

**Note**: Some overlap between files (2023-2024 appears in multiple files). After deduplication: 265,990 unique incidents.

### Weather Data (2 files)

| File | Records | Columns | Resolution | Formats Available |
|------|---------|---------|------------|-------------------|
| `atlanta_hourly_weather_2021_to_current` | 43,128 | 7 | Hourly | CSV, Parquet |
| `atlanta_daily_weather_2021_to_current` | 1,797 | 8 | Daily | CSV, Parquet |

**Source**: Open-Meteo Historical Weather API  
**Coverage**: January 2021 - December 2025 (5 years)

---

## Interim Data (Enriched but Pre-Panel)

| File | Records | Columns | Description | Formats Available |
|------|---------|---------|-------------|-------------------|
| `apd_model_data_target_crimes` | 117,749 | 70 | Target crimes with all features | CSV, Parquet |

**Description**: Raw incidents filtered to burglary/larceny (UCR codes 220, 23*), enriched with:
- Temporal features (hour, day, cyclical encodings)
- Spatial features (NPU, zones, campus distances)
- Weather features (merged from hourly weather)
- Location metadata

---

## Processed Data (Final Panel Datasets)

### Core Files

| File | Records | Columns | Description | Formats Available |
|------|---------|---------|-------------|-------------------|
| `all_apd_crimes` | 265,990 | 70 | All APD crimes (deduplicated) | CSV, Parquet |
| `target_crimes` | 117,749 | 64 | Burglary/larceny only | CSV, Parquet |
| `target_crimes_panel` | 99,965 | 52 | Hourly NPU with lags | Parquet only |
| `npu_sparse_panel` | 99,965-104,286 | 31-46 | Observed hours only | CSV, Parquet |
| `npu_dense_panel` | 1,063,175-1,074,500 | 31-46 | Complete NPU-hour grid | CSV, Parquet |

**Note**: Row counts vary slightly between CSV and Parquet due to different processing timestamps.

### File Locations

```
data/
├── raw/apd/                                 # Raw APD downloads
│   ├── apd_2020_2024.csv                    # 220,767 incidents
│   ├── apd_2023_2025.csv                    # 135,961 incidents
│   └── apd_2021_2024.csv                    # 16,136 incidents
│
├── external/                                # Weather data
│   ├── atlanta_hourly_weather_2021_to_current.csv/parquet    # 43,128 hours
│   └── atlanta_daily_weather_2021_to_current.csv/parquet     # 1,797 days
│
├── interim/apd/                             # Intermediate processing
│   └── apd_model_data_target_crimes.csv/parquet              # 117,749 enriched incidents
│
└── processed/
    ├── panels/                              # PRIMARY - Latest panels
    │   ├── target_crimes_panel.parquet      # 99,965 × 52
    │   ├── npu_sparse_panel.parquet         # 99,965 × 46
    │   └── npu_dense_panel.parquet          # 1,074,500 × 31
    │
    └── apd/                                 # LEGACY - Older versions + CSVs
        ├── all_apd_crimes.csv/parquet       # 265,990 × 70
        ├── target_crimes.csv/parquet        # 117,749 × 64
        ├── target_crimes_panel.parquet      # 99,965 × 52
        ├── npu_sparse_panel.csv/parquet     # 104,286/99,965 × 31/46
        └── npu_dense_panel.csv/parquet      # 1,074,500/1,063,175 × 31/46
```

---

## Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: RAW DATA INGESTION                                 │
└─────────────────────────────────────────────────────────────┘

APD Downloads (3 files)          Weather API
├─ apd_2020_2024.csv (220K)     ├─ Hourly (43K hours)
├─ apd_2023_2025.csv (136K)     └─ Daily (1.8K days)
└─ apd_2021_2024.csv (16K)

            │                            │
            └────────────┬───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: CLEANING & DEDUPLICATION                           │
└─────────────────────────────────────────────────────────────┘

Deduplicate APD files → all_apd_crimes.parquet (265,990)
                              │
                              ├─ Filter UCR codes (220, 23*)
                              │
                              ▼
                    target_crimes.parquet (117,749)
                              │
                              ├─ Geocode to lat/lon
                              ├─ Spatial join to NPUs
                              ├─ Merge weather by hour
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: FEATURE ENGINEERING                                 │
└─────────────────────────────────────────────────────────────┘

apd_model_data_target_crimes.parquet (117,749 × 70 cols)
├─ Temporal: hour, day_of_week, cyclical encodings
├─ Spatial: NPU, zones, campus distances, densities
├─ Weather: temp_f, precipitation, indicators
└─ Location: location_type, address (obfuscated)

            │
            │ Aggregate to hourly NPU-level
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: PANEL CREATION                                      │
└─────────────────────────────────────────────────────────────┘

Hourly NPU Aggregation
├─ Count incidents per NPU × hour
├─ Average numeric features
└─ Mode of categorical features

            │
            ├─ Add lagged features (1h, 3h, 6h, 12h, 24h, 168h)
            │
            ▼
target_crimes_panel.parquet (99,965 × 52 cols)
            │
            ├─ Remove lags
            │
            ▼
npu_sparse_panel.parquet (99,965 × 46 cols)
[Only NPU-hours with incidents]
            │
            ├─ Create complete time grid
            ├─ Fill missing hours with zeros
            │
            ▼
npu_dense_panel.parquet (1,074,500 × 31 cols)
[All NPU-hour combinations including zeros]
```

---

## File Format Comparison

### CSV vs Parquet

| Aspect | CSV | Parquet |
|--------|-----|---------|
| **File Size** | Larger (~3-5x) | Smaller (compressed) |
| **Read Speed** | Slower | Faster |
| **Precision** | Can lose float precision | Preserves exact types |
| **Schema** | No schema enforcement | Strict schema |
| **Use Case** | Human inspection, Excel | Production modeling |

**Recommendation**: Use Parquet files for modeling, CSV for inspection.

---

## Column Count Variations

### Why different column counts?

| File | Columns | Reason |
|------|---------|--------|
| `all_apd_crimes` | 70 | All original APD columns + enrichment |
| `target_crimes` | 64 | Removed some APD-specific columns |
| `target_crimes_panel` | 52 | Aggregated + 6 lag features |
| `npu_sparse_panel` | 46 | Removed lags |
| `npu_dense_panel` | 31 | Removed high-cardinality categoricals |

**Dense panel optimization**: To reduce file size for 1M+ rows, removed:
- High-cardinality categoricals (location_type, offense details)
- Redundant features
- Features with >50% missing values in zero-count hours

---

## Storage Requirements

### By Format

| File Type | Total Size | Count |
|-----------|------------|-------|
| Parquet files | ~450 MB | 11 files |
| CSV files | ~1.2 GB | 11 files |
| **Total** | **~1.65 GB** | **22 files** |

### By Stage

| Stage | Files | Storage |
|-------|-------|---------|
| Raw | 3 CSVs | ~350 MB |
| External (Weather) | 4 files | ~15 MB |
| Interim | 2 files | ~120 MB |
| Processed (Parquet) | 8 files | ~280 MB |
| Processed (CSV) | 5 files | ~900 MB |

---

## Data Quality Summary

### Deduplication
- **Raw records**: 372,864 (across 3 files)
- **After deduplication**: 265,990 unique incidents
- **Duplicates removed**: 106,874 (28.7%)

### Filtering
- **All crimes**: 265,990
- **Target crimes (burglary/larceny)**: 117,749
- **Percentage**: 44.3% of all crimes

### Aggregation
- **Incident-level**: 117,749 incidents
- **Hourly NPU-level (observed)**: 99,965 hours
- **Compression ratio**: 1.18 (slight reduction from multiple incidents per hour)

### Grid Completion
- **Observed hours**: 99,965
- **Complete grid**: 1,074,500
- **Zero-inflation**: 90.7% of hours have 0 incidents

---

## Kaggle Upload Recommendation

### Primary Files to Upload

1. **Processed Panels** (use Parquet for efficiency):
   - `target_crimes_panel.parquet` (99,965 × 52) — For time-series modeling
   - `npu_sparse_panel.parquet` (99,965 × 46) — For quick prototyping
   - `npu_dense_panel.parquet` (1,074,500 × 31) — For zero-inflated models

2. **Raw Incidents** (for reproducibility):
   - `target_crimes.parquet` (117,749 × 64) — Incident-level data
   - `all_apd_crimes.parquet` (265,990 × 70) — Optional, for context

3. **Weather Data** (external dependency):
   - `atlanta_hourly_weather_2021_to_current.parquet` (43,128 × 7)

4. **Documentation**:
   - `DATASET_CARD.md`
   - `data_dictionary.md`
   - `README.md`

**Total upload size**: ~280 MB (Parquet only)

### Optional CSV Versions

For users without Parquet libraries:
- `npu_sparse_panel.csv` (smaller, easier to open in Excel)
- `target_crimes.csv` (for inspection)

---

## Version Control

### Primary Dataset
**Location**: `data/processed/panels/`  
**Version**: Latest (December 2025)  
**Files**: 3 Parquet files

### Legacy Dataset
**Location**: `data/processed/apd/`  
**Version**: Earlier processing run  
**Files**: CSVs + slightly older Parquets  
**Note**: Keep for backward compatibility

**This manifest reflects the ACTUAL data structure as of December 2025.**