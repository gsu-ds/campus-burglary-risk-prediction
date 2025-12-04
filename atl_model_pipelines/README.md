# ATL Model Pipelines

**Production-ready data pipeline for Atlanta crime prediction and forecasting**

A modular, end-to-end machine learning pipeline that transforms raw Atlanta Police Department data into actionable crime forecasts using spatial-temporal analysis, advanced feature engineering, and ensemble modeling.

---

## **Project Overview**

This pipeline processes raw crime incident data from the Atlanta Police Department (2020-2025) and produces:
- ✅ Cleaned, enriched crime datasets with 40+ engineered features
- ✅ Hourly NPU × time panels (sparse and dense) ready for modeling
- ✅ Multi-horizon crime forecasts (1-24 hours ahead)
- ✅ Model performance benchmarks across multiple algorithms
- ✅ Comprehensive validation reports and visualizations

**Key Capabilities:**
- Spatial enrichment (NPU, zone, neighborhood, campus proximity)
- Temporal features (cyclical encodings, holidays, hour blocks)
- Weather integration (hourly + daily conditions)
- Grid-based density computation
- Rolling NPU statistics
- Automated model evaluation with cross-validation

---

##  **Repository Structure**

```
atl_model_pipelines/
├── __init__.py
│
├── utils/                          # Shared utilities
│   ├── __init__.py
│   └── logging.py                  # Pipeline logging with rich tables
│
├── ingestion/                      # Data loading
│   ├── __init__.py
│   ├── apd_ingest.py              # APD CSV ingestion
│   ├── weather_fetcher.py         # Weather API + caching
│   ├── shapefile_loader.py        # Spatial data loading
│   └── ingestion_master.py        # Orchestrator
│
├── transform/                      # Feature engineering
│   ├── __init__.py
│   ├── cleaning.py                # Date parsing, standardization
│   ├── spatial.py                 # Spatial joins (NPU, zone, etc.)
│   ├── temporal.py                # Temporal features
│   ├── weather.py                 # Weather merging
│   ├── campus_distance.py         # Campus proximity (11 campuses)
│   ├── grid_density.py            # Grid-based density
│   ├── rolling_stats.py           # NPU rolling averages
│   ├── spatial_repair.py          # Spatial data repairs
│   ├── filters.py                 # Target offense filtering
│   ├── finalize.py                # Final dataset prep
│   ├── encoding.py                # Categorical encoding
│   ├── optimization.py            # Memory optimization
│   ├── crime_lags.py              # Lag/rolling features
│   ├── panel_builder.py           # Sparse & dense panel creation
│   └── transform_master.py        # Orchestrator
│
├── validate/                       # Data quality checks
│   ├── __init__.py
│   ├── core.py                    # Basic integrity checks
│   ├── panel_checks.py            # Panel-specific validation
│   ├── orchestrator.py            # Validation orchestrator
│   ├── wandb_logging.py           # W&B artifact logging
│   └── VALIDATION_GUIDE.md        # Complete usage guide
│
└── models/                         # ML models & evaluation
    ├── __init__.py
    ├── rolling_cv.py              # Cross-validation framework
    ├── horizon_models.py          # Multi-horizon forecasting
    ├── poisson_models.py          # Count-based models
    ├── pipeline.py                # End-to-end pipeline runner
    └── README.md                  # Model documentation

config.py                           # Central configuration
```

---

##  **Quick Start**

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/atl-crime-prediction.git
cd atl-crime-prediction

# Install dependencies
pip install -r requirements.txt

# Optional: Install model libraries
pip install xgboost lightgbm catboost prophet wandb
```

### **2. Configure Paths**

Edit `config.py` to set your data directories:
```python
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
# ... paths configured automatically
```

### **3. Run Complete Pipeline**

```bash
# Option A: Run full pipeline (ingestion → transforms → validation)
python -m atl_model_pipelines.models.pipeline

# Option B: Run step-by-step
python -c "
from atl_model_pipelines.ingestion.ingestion_master import run_ingestion
from atl_model_pipelines.transform.transform_master import run_transforms
from atl_model_pipelines.transform.panel_builder import build_all_panels

# Step 1: Ingest raw data
ingestion_output = run_ingestion()

# Step 2: Transform & enrich
transform_output = run_transforms(ingestion_output)

# Step 3: Build panels
panels = build_all_panels(df=transform_output['target_crimes'], save=True)
"
```

### **4. Run Models**

```bash
# Baseline + ML comparison with cross-validation
python -m atl_model_pipelines.models.rolling_cv

# Multi-horizon forecasting (1-24 hours)
python -m atl_model_pipelines.models.horizon_models

# Poisson count models
python -m atl_model_pipelines.models.poisson_models
```

---

##  **Data Pipeline Flow**

```
┌─────────────────┐
│  Raw APD CSVs   │
└────────┬────────┘
         │
    [INGESTION]
         │
         ├──→ Combine & deduplicate
         ├──→ Fetch weather data
         └──→ Load shapefiles
         │
┌────────▼────────┐
│ Combined Dataset│
└────────┬────────┘
         │
   [TRANSFORM]
         │
         ├──→ Spatial enrichment (NPU, zone, neighborhood)
         ├──→ Temporal features (hour blocks, cyclical)
         ├──→ Weather merging (hourly + daily)
         ├──→ Campus distance (11 campuses)
         ├──→ Grid density (7-day rolling)
         ├──→ NPU rolling averages (30-day)
         └──→ Target filtering
         │
┌────────▼────────┐
│ Enriched Dataset│
└────────┬────────┘
         │
   [VALIDATION]
         │
         ├──→ Integrity checks
         ├──→ Completeness validation
         └──→ Quality reports
         │
┌────────▼────────┐
│ Panel Building  │
└────────┬────────┘
         │
         ├──→ Sparse panel (incidents only)
         └──→ Dense panel (complete grid)
         │
┌────────▼────────┐
│  Ready for ML   │
└─────────────────┘
         │
    [MODELING]
         │
         ├──→ Rolling CV (4 folds)
         ├──→ Horizon forecasting
         └──→ Poisson models
         │
┌────────▼────────┐
│ Results & Cards │
└─────────────────┘
```

---

##  **Module Documentation**

### **Utils** - Shared Utilities

**logging.py**
- `log_step(step_name, df)` - Track pipeline steps with row/col counts
- `show_pipeline_table()` - Display comprehensive pipeline summary
- `clear_pipeline_log()` - Reset logging state

```python
from atl_model_pipelines.utils.logging import log_step, show_pipeline_table

df = transform_data(df)
log_step("After transformation", df)

# At end of pipeline
show_pipeline_table()
```

---

### **Ingestion** - Data Loading

Load raw data from multiple sources:
- APD CSV files (multiple years)
- Weather data via Open-Meteo API
- Shapefiles (NPU, zones, neighborhoods, cities, campuses)

```python
from atl_model_pipelines.ingestion.ingestion_master import run_ingestion

# Load all data sources
ingestion_output = run_ingestion()
# Returns: {
#   'apd': DataFrame,
#   'weather_hourly': DataFrame,
#   'weather_daily': DataFrame,
#   'shapefiles': Dict[str, GeoDataFrame]
# }
```

---

### **Transform** - Feature Engineering

Complete transformation pipeline with 13+ specialized transformations:

```python
from atl_model_pipelines.transform.transform_master import run_transforms

# Run all transforms
transform_output = run_transforms(ingestion_output)
# Returns: {
#   'target_crimes': DataFrame,   # Filtered target offenses
#   'all_crimes': DataFrame        # Complete enriched dataset
# }
```

**Key Transformations:**
- ✅ **Spatial** - NPU/zone/neighborhood/city assignment
- ✅ **Temporal** - Hour blocks, cyclical encodings, holidays
- ✅ **Weather** - Hourly + daily conditions merged
- ✅ **Campus** - Distance + proximity flags (11 campuses)
- ✅ **Grid Density** - 7-day rolling spatial density
- ✅ **NPU Stats** - 30-day rolling crime averages

---

### **Validate** - Quality Assurance

Comprehensive validation suite:

```python
from atl_model_pipelines.validate import run_validations, run_panel_validations

# Validate transformed data
df = run_validations(df, step_name="After transforms", check_npu=True)

# Validate panels
panel = run_panel_validations(
    panel_df,
    panel_type="crime",
    check_schema=True,
    check_completeness=True
)
```

**Validation Checks:**
- Incident number uniqueness
- Coordinate bounds (Atlanta area)
- Feature completeness
- Panel schema requirements
- Time range coverage
- NPU category integrity

---

### **Models** - ML Pipeline

Three main modeling approaches:

**1. Rolling Cross-Validation** (`rolling_cv.py`)
- Evaluates multiple algorithms across 3 datasets
- 4-fold temporal cross-validation
- Generates leaderboards and model cards

**2. Multi-Horizon Forecasting** (`horizon_models.py`)
- Predicts crime counts 1-24 hours ahead
- Single multi-output XGBoost model
- Tracks RMSE/MAE per horizon

**3. Poisson Count Models** (`poisson_models.py`)
- Specialized models for count data
- Compares XGBoost, LightGBM, CatBoost, ZIP, Prophet
- Handles zero-inflation properly

See [models/README.md](atl_model_pipelines/models/README.md) for complete documentation.

---

## **Output Files**

### **Data Outputs**
```
data/
└── processed/apd/
    ├── target_crimes.parquet     # Filtered targets
    ├── npu_sparse_panel.parquet  # Sparse panel 
    └── npu_dense_panel.parquet   # Dense panel
```

### **Model Results**
```
reports/
├── test_results/
│   ├── npu_sparse_panel/
│   │   ├── simple_results.csv
│   │   ├── model_leaderboard_summary.csv
│   │   └── cv_results/
│   ├── npu_dense_panel/
│   ├── horizon_models/
│   │   ├── horizon_metrics.csv
│   │   └── predictions/
│   └── current_modeling/
│       ├── model_metrics.json
│       └── predictions/
├── cards/
│   ├── <dataset>_dataset_card.md
│   └── <dataset>_model_card.md
└── figures/leaderboard/
    └── combined_leaderboard_scatter.png
```

### **Artifacts**
```
artifacts/
├── wandb/                # W&B run logs
├── catboost_info/        #CatBoost training logs
├── horizon_models/
│   ├── xgb_multioutput_hourly_npu_lite.joblib
│   └── meta_hourly_npu_lite.json
└── assorted/
```

---

## **Key Features**

### **1. Campus Proximity Analysis**

Computes distance and proximity flags for **11 Atlanta campuses:**
- Georgia State University
- Georgia Tech
- Emory University
- Clark Atlanta
- Spelman College
- Morehouse College
- Morehouse School of Medicine
- Atlanta Metropolitan State College
- Atlanta Technical College
- SCAD Atlanta
- John Marshall Law School

```python
from atl_model_pipelines.transform.campus_distance import compute_campus_distance

df = compute_campus_distance(df)
# Adds: campus_label, campus_distance_m, campus_code
#       near_gsu, near_ga_tech, near_emory, ... (11 binary flags)
```

### **2. Panel Building**

Creates two panel types on a NPU x Hour grid

**Sparse Panel** - Only hours with incidents
```python
from atl_model_pipelines.transform.panel_builder import build_sparse_panel

sparse = build_sparse_panel(df)
# ~10k-50k rows, includes aggregated features
```

**Dense Panel** - Complete NPU × hour grid
```python
from atl_model_pipelines.transform.panel_builder import build_dense_panel

dense = build_dense_panel(df)
# ~780k rows (26 NPUs × 30k hours), zero-filled
```

### **3. Comprehensive Logging**

Rich console output with pipeline tracking:

```python
from atl_model_pipelines.utils.logging import log_step, show_pipeline_table

log_step("Initial APD data", df)
log_step("After spatial enrichment", df)
log_step("After temporal features", df)

# At end
show_pipeline_table()
```

**Output:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ Step                     ┃ Rows    ┃ Cols ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ Initial APD data         │ 100,000 │ 30   │
│ After spatial enrichment │ 95,000  │ 35   │
│ After temporal features  │ 95,000  │ 45   │
└──────────────────────────┴─────────┴──────┘
```

---

##  **Performance Benchmarks**

### **Model Performance (NPU Dense Panel)**

| Model | MAE | RMSE | R² | Training Time |
|-------|-----|------|----|---------------|
| XGBoost (Poisson) | 0.85 | 1.20 | 0.45 | ~5 min |
| LightGBM (Poisson) | 0.88 | 1.22 | 0.43 | ~3 min |
| CatBoost (Poisson) | 0.87 | 1.21 | 0.44 | ~8 min |
| Random Forest | 0.95 | 1.30 | 0.38 | ~12 min |
| Linear Regression | 1.10 | 1.45 | 0.28 | <1 min |

*Example, will be updated*

### **Pipeline Performance**

| Stage | Typical Runtime | Memory Usage |
|-------|----------------|--------------|
| Ingestion | 2-5 minutes | ~2 GB |
| Transform | 5-10 minutes | ~4 GB |
| Panel Building | 3-5 minutes | ~6 GB |
| Model Training (CV) | 15-30 minutes | ~8 GB |

*On standard laptop (16GB RAM, 4 cores)*

---

##  **Research & Academic Use**

### **Data Sources**
- **Crime Data:** Atlanta Police Department Open Data Portal
- **Weather:** Open-Meteo Archive API
- **Spatial:** Atlanta GIS, US Census Bureau

### **Methodology**
- **Time Period:** 2020-2025 (training through 2023, testing 2024)
- **Spatial Resolution:** NPU (Neighborhood Planning Unit) level
- **Temporal Resolution:** Hourly
- **Target Offenses:** Larceny, theft, robbery, burglary, fraud

### **Citation**
```bibtex
@software{atl_crime_prediction,
  title={'placeholder until we complete'},
  author={Joshua Darius Pina},
  year={2025},
  institution={Georgia State University},
  note={Data Science Capstone Project}
}
```

---

##  **Troubleshooting**

### **Common Issues**

**1. Import Errors**
```python
# If you get: ModuleNotFoundError: No module named 'atl_model_pipelines'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

**2. Missing Dependencies**
```bash
# Check optional dependencies
python -c "
from atl_model_pipelines.models.rolling_cv import HAS_XGB, HAS_CAT
print(f'XGBoost: {HAS_XGB}, CatBoost: {HAS_CAT}')
"

# Install missing
pip install xgboost catboost lightgbm
```

**3. Out of Memory**
```python
# Use sparse panel instead of dense
from config import DATA_SPARSE
df = pd.read_parquet(DATA_SPARSE)  # Much smaller

# Or optimize dtypes
from atl_model_pipelines.transform.optimization import optimize_dtypes
df = optimize_dtypes(df)
```

**4. Slow Training**
```python
# Reduce estimators in models
model = XGBRegressor(n_estimators=100)  # Instead of 400

# Use fewer CV folds
# Edit rolling_cv.py, reduce folds list
```

---

## **Additional Documentation**

- **[Models README](atl_model_pipelines/models/README.md)** - Complete model documentation
- **[Validation Guide](atl_model_pipelines/validate/VALIDATION_GUIDE.md)** - Validation functions
- **[Config Reference](config.py)** - All path configurations

---

## **Contributing**

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## **License**

For educational and research use only. Derived from Atlanta Police Department Open Data Portal.

**Restrictions:**
- Not for commercial use
- Not for production deployment
- Research and educational purposes only

---

## 👥 **Contact**

**Authors:** Gunn Madan, Harini Mohan, Joshua Piña, Yuntian "Robin" Wu
**Institution:** Georgia State University  
**Program:** Data Science Capstone  
**Year:** 2025

---

## 🎓 **Acknowledgments**

- Atlanta Police Department for open crime data
- Open-Meteo for weather data API
- Georgia State University Data Science Program
- Dr. Berkay Aydin for project guidance

---

**Last Updated:** 2025-12-04  
**Version:** 1.0.2  
**Status:**  Production Ready