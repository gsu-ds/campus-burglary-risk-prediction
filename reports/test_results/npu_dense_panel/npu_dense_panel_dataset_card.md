# 📦 Dataset Card: npu_dense_panel

## Overview
This dataset is part of the **Campus Burglary Risk Prediction** project for Atlanta.

- **Rows:** 1,074,500
- **Columns:** 20
- **Date column:** `hour_ts`
- **Date range:** 2021-01-01 05:00:00 → 2025-11-27 00:00:00
- **Target column:** `burglary_count`
- **Grouping column:** `npu` (panel data)

## Intended Use
- Spatio-temporal modeling of burglary risk
- Evaluation of baseline ML models and tree-based models
- Comparison of rolling-origin CV vs simple train/test

## Basic Schema (first 25 columns)

| column              | dtype          |
|:--------------------|:---------------|
| npu                 | object         |
| hour_ts             | datetime64[ns] |
| burglary_count      | float64        |
| grid_density_7d     | float64        |
| npu_crime_avg_30d   | float64        |
| temp_f              | float32        |
| is_raining          | int64          |
| is_hot              | int64          |
| is_cold             | int64          |
| is_daylight         | int64          |
| is_weekend          | int64          |
| is_holiday          | int64          |
| day_number          | float64        |
| month               | float64        |
| year                | float64        |
| hour_sin            | float32        |
| hour_cos            | float32        |
| day_of_week         | object         |
| campus_distance_m   | float64        |
| location_type_count | float64        |


## Train/Test Recommendation
- Train on data **before 2024-01-01**
- Test on data **from 2024-01-01 onwards**

## License / Ethics
- Derived from Atlanta Police Department open data and external enrichment.
- For research and educational use only.

