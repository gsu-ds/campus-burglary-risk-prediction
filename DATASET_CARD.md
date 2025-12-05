# Dataset Card: Core Atlanta Burglary-Related Crimes (2021-2025)

## Dataset Description

### Summary
This dataset contains enriched burglary and larceny incident data for Atlanta, Georgia (2021-2025), aggregated at the Neighborhood Planning Unit (NPU) and hourly level. It has been specifically designed for spatiotemporal crime forecasting research and supports reproducible model development in urban safety analytics.

**Repository**: https://github.com/gsu-ds/campus-burglary-risk-prediction  
**Paper/Report**: [Pending link to final report]  
**Point of Contact**: Joshua Piña (jpina4@student.gsu.edu)

### Dataset Details

**Dataset Type**: Time-series panel data  
**Task Categories**: 
- Time-series forecasting
- Count regression
- Spatial-temporal prediction
- Public safety analytics

**Language(s)**: N/A (numeric/categorical data)  
**License**: [MIT]

## Dataset Structure

### Data Instances

Each row represents one hour in one NPU (Neighborhood Planning Unit) with:
- **Temporal features**: Hour, day of week, month, cyclical encodings
- **Spatial features**: NPU identifier, campus proximity, location density
- **Weather features**: Temperature, precipitation, daylight status
- **Target variable**: Hourly burglary/larceny count
- **Lagged features**: Previous 1h, 3h, 6h, 12h, 24h, 168h counts

**Example record**:
```json
{
  "npu": "N",
  "hour_ts": "2024-01-15 14:00:00",
  "burglary_count": 2,
  "temp_f": 65.3,
  "is_raining": 0,
  "is_weekend": 0,
  "hour_sin": 0.866,
  "hour_cos": 0.5,
  "lag_24": 1,
  "campus_distance_m": 450.2
}
```

### Data Fields

#### Target Variable
- `burglary_count` (int): Count of burglary/larceny incidents in this NPU-hour

#### Temporal Features
- `hour_ts` (datetime): Hour timestamp (YYYY-MM-DD HH:00:00)
- `hour` (int): Hour of day (0-23)
- `day_of_week` (int): Day of week (0=Monday, 6=Sunday)
- `month` (int): Month (1-12)
- `is_weekend` (bool): Weekend indicator
- `is_holiday` (bool): Federal holiday indicator
- `hour_sin`, `hour_cos` (float): Cyclical hour encoding
- `day_sin`, `day_cos` (float): Cyclical day-of-week encoding

#### Spatial Features
- `npu` (string): Neighborhood Planning Unit identifier (A-Z)
- `campus_distance_m` (float): Distance to nearest university campus (meters)
- `location_type_count` (int): Number of location types in NPU
- `grid_density_7d` (float): 7-day moving average of nearby NPU activity

#### Weather Features
- `temp_f` (float): Temperature in Fahrenheit
- `is_raining` (bool): Precipitation indicator
- `is_hot` (bool): Temperature > 85°F
- `is_cold` (bool): Temperature < 40°F
- `is_daylight` (bool): Daylight hours indicator

#### Lagged Features
- `lag_1`, `lag_3`, `lag_6`, `lag_12`, `lag_24`, `lag_168` (int): Historical counts at various lags

### Data Splits

**Recommended splits for reproducibility**:
- **Training**: 2021-01-01 to 2023-12-31 (3 years)
- **Validation**: 2024-01-01 to 2024-06-30 (6 months)
- **Test**: 2024-07-01 to 2025-12-31 (18 months)

**Cross-validation**: Rolling origin with 6-month folds

## Dataset Creation

### Source Data

#### Data Collection
- **Primary Source**: Atlanta Police Department (APD) public crime records
- **Weather Data**: Open-Meteo Historical Weather API
- **Spatial Data**: 
  - City of Atlanta NPU boundaries (shapefile)
  - University campus locations (manually geocoded)
  - OpenStreetMap for additional spatial features

#### Data Processing Pipeline

**Step 1: Crime Data Ingestion** (`atl_model_pipelines/ingestion/`)
- Downloaded APD incident reports (2021-2025)
- Filtered to burglary (`220`) and larceny (`23*`) UCR codes
- Geocoded incidents to lat/lon coordinates
- Spatial join to NPU boundaries

**Step 2: Feature Engineering** (`atl_model_pipelines/transform/`)
- Aggregated to hourly NPU-level counts
- Generated temporal features (cyclical encodings)
- Computed spatial features (distance to campuses, density metrics)
- Merged weather data by hour
- Created lagged features for time-series modeling

**Step 3: Validation** (`atl_model_pipelines/validate/`)
- Checked for missing NPU assignments
- Validated temporal continuity
- Ensured no data leakage in lagged features
- Quality checks on outliers and anomalies

**Complete pipeline available**: `atl_model_pipelines/` directory in repository

### Annotations

This dataset does not contain human annotations. All features are programmatically derived from:
1. Official crime reports (factual records)
2. Weather API data (measured values)
3. Spatial calculations (geometric computations)

**No personally identifiable information (PII)** is included. Incident locations are aggregated to NPU level (neighborhood-scale).

## Dataset Statistics

### Basic Statistics
- **Total rows**: ~350,000 (varies by data pull)
- **Time span**: January 2021 - November 2025
- **NPUs covered**: 25 planning units
- **Temporal resolution**: Hourly
- **Missing data**: <1% (imputed with forward-fill for weather)

### Target Distribution
- **Mean incidents/hour**: 0.15
- **Std incidents/hour**: 0.42
- **Max incidents/hour**: 8
- **Zero inflation**: ~85% of hours have 0 incidents

### Temporal Patterns
- **Peak hours**: 10am-2pm, 6pm-8pm
- **Peak days**: Friday, Saturday
- **Seasonal trends**: Higher in summer months

## Considerations for Using the Data

### Social Impact

**Intended Use**:
- Academic research in spatiotemporal forecasting
- Resource allocation optimization for campus safety
- Comparative studies of crime prediction methods

**Out-of-Scope Uses**:
- Individual-level predictions or profiling
- Real-time surveillance systems without human oversight
- Decisions affecting individual liberty or employment

### Discussion of Biases

**Known Limitations**:
1. **Reporting bias**: Crimes must be reported to appear in data
2. **Spatial bias**: NPUs vary in size and population
3. **Temporal bias**: Recent years may differ from historical patterns
4. **University focus**: Dataset emphasizes NPUs near campuses

**Mitigation Strategies**:
- Aggregate to NPU level (not addresses)
- No demographic information included
- Predictions provide risk scores, not certainties
- Intended for strategic planning, not individual enforcement

### Recommendations

**For Researchers**:
- Use rolling cross-validation for time-series evaluation
- Consider zero-inflation in model selection
- Account for spatial autocorrelation between NPUs
- Report performance separately for high-risk and low-risk NPUs

**For Practitioners**:
- Combine predictions with local knowledge
- Use for resource allocation, not individual targeting
- Update models regularly as patterns evolve
- Validate predictions with domain experts

## Additional Information

### Dataset Curators
Georgia State University Data Science Capstone (Fall 2025)
- Gunn Madan
- Harini Mohan
- Joshua Piña
- Yuntian 'Robin' Wu

### Licensing Information
[MIT]

**Attribution**: Please cite this dataset as:
```
Piña, J., et al. (2025). Core Atlanta Burglary-Related Crimes (2021-2025). 
Kaggle. https://www.kaggle.com/datasets/joshuapina/core-atlanta-burglary-related-crimes-2021-2025
```

### Citation Information
```bibtex
@dataset{pina2025atlanta,
  title={Core Atlanta Burglary-Related Crimes (2021-2025)},
  author={Piña, Joshua, and Wu, Yuntian and Mohan, Harini and Madan, Gunn},
  year={2025},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/joshuapina/core-atlanta-burglary-related-crimes-2021-2025}
}
```

### Contributions
Thanks to:
- Atlanta Police Department for public data
- Open-Meteo for weather data
- City of Atlanta for NPU boundaries

### Contact
For questions or issues: jpina4@student.gsu.edu, ywu49@student.gsu.edu, gmadan1@student.gsu.edu, hmohan1@student.gsu.edu

### Changelog

**v2.0.2** (2025-12)
- Hourly NPU-level aggregation for 3-level paneling (Target, Sparse, Dense)
- Full feature engineering pipeline