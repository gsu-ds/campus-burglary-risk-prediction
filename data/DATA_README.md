# Data Directory

This directory contains all raw, interim, and processed data for the Atlanta Burglary Risk Prediction project.

---

## Directory Structure

```
data/
├── external/          # Weather data from Open-Meteo API
├── interim/           # Intermediate processing outputs
├── raw/               # Raw APD crime data downloads
└── processed/
    ├── panels/        # Final modeling datasets (USE THESE)
    └── apd/           # Legacy files and CSVs
```

---

## Quick Reference

### Primary Modeling Files

**Location**: `data/processed/panels/`

| File | Records | Columns | Use Case |
|------|---------|---------|----------|
| `target_crimes_panel.parquet` | 99,965 | 52 | Time-series models (with lags) |
| `npu_sparse_panel.parquet` | 99,965 | 46 | Baseline models (observed hours) |
| `npu_dense_panel.parquet` | 1,074,500 | 31 | Zero-inflated models (complete grid) |

### Raw Data Sources

| File | Records | Description |
|------|---------|-------------|
| `raw/apd/*.csv` | 372,864 | Raw APD downloads (3 files) |
| `external/atlanta_hourly_weather_*.parquet` | 43,128 | Hourly weather data |
| `processed/apd/all_apd_crimes.parquet` | 265,990 | All crimes (deduplicated) |
| `processed/apd/target_crimes.parquet` | 117,749 | Burglary/larceny only |

---

## Dataset Statistics

- **Time Period**: January 2021 - December 2025
- **Spatial Coverage**: 25 NPUs (Neighborhood Planning Units)
- **Target Crimes**: 117,749 burglary/larceny incidents
- **Panel Observations**: 99,965 observed hours, 1,074,500 complete grid
- **Zero-Inflation**: 90.7% of hours have 0 incidents

---

## Gitignore Policy

**Tracked** (lightweight, essential):
- Processed Parquet files (`processed/**/*.parquet`)
- Weather data (`external/*.parquet`)
- Documentation

**Ignored** (large, regenerable):
- Raw APD CSVs (`raw/*.csv`) - 300MB-1.5GB each
- CSV duplicates (`processed/**/*.csv`)
- NIBRS data (`external/nibrs_ga_2021_2024/`) - 6GB+
- Intermediate outputs (`interim/`)
- Temporary files (`tmp/`)

**Total tracked size**: ~450MB (Parquet only)  
**Total untracked size**: ~2-3GB (raw CSVs)

---

## Reproduction

To regenerate all processed files from raw data:

```bash
# 1. Download raw APD data (see ingestion scripts)
python -m atl_model_pipelines.ingestion.ingestion_master

# 2. Process and enrich
python -m atl_model_pipelines.transform.transform_master

# 3. Validate quality
python -m atl_model_pipelines.validate.orchestrator

# Output: data/processed/panels/*.parquet
```

See [DATA_GENERATION_PIPELINE.md](../DATA_GENERATION_PIPELINE.md) for detailed steps.

---

## File Sizes

| Type | Size | Notes |
|------|------|-------|
| Raw APD CSVs | 1-2 GB | Not tracked in Git |
| Weather data | ~15 MB | Tracked (Parquet) |
| Processed panels | ~280 MB | Tracked (Parquet) |
| **Total** | **~1.65 GB** | Only 450MB tracked |

---

## Parquet vs CSV

**For modeling**: Use `.parquet` files
- Faster read/write
- Smaller file size (~3-5x compression)
- Preserves exact data types
- Optimized for pandas/polars

**For inspection**: CSVs available in `processed/apd/`
- Human-readable
- Compatible with Excel
- Useful for debugging

---

## Panel Creation Flow

```
Raw APD (372,864) 
    → Deduplicate (265,990)
    → Filter burglary/larceny (117,749)
    → Aggregate hourly + features (99,965)
    → Complete grid (1,074,500)
```

See [data_dictionary.md](../data_dictionary.md) for detailed schema.

---

## Common Questions

**Q: Which file should I use for modeling?**  
A: `processed/panels/target_crimes_panel.parquet` - includes lagged features

**Q: Why are there two `npu_dense_panel.parquet` files?**  
A: Use `processed/panels/` (1,074,500 rows) - it's the latest version

**Q: How do I get the raw APD data?**  
A: Run ingestion pipeline or download from APD Open Data Portal

**Q: Can I use CSV instead of Parquet?**  
A: Yes, but Parquet is 5x faster and smaller. CSVs available in `processed/apd/`

---

## Contact

Questions about data structure or pipeline:
- **GitHub Issues**: https://github.com/gsu-ds/campus-burglary-risk-prediction/issues
- **Documentation**: See [data_dictionary.md](../data_dictionary.md)

---

**Last Updated**: December 2025