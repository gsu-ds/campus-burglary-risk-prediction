# ATL Model Pipelines - Models

Complete modeling suite for Atlanta crime prediction with multiple model types, evaluation strategies, and automated reporting.

---

##  **Module Structure**

```
atl_model_pipelines/
└── models/
    ├── __init__.py              # Model module exports
    ├── rolling_cv.py            # Rolling cross-validation framework
    ├── horizon_models.py        # Multi-horizon forecasting
    ├── poisson_models.py        # Count-based models
    ├── pipeline.py              # End-to-end pipeline runner
    └── README.md                # This file
```

---

##  **Model Scripts Overview**

### **1. rolling_cv.py** - Baseline & ML Model Comparison

**Purpose:** Comprehensive model evaluation across multiple datasets using both simple train/test splits and rolling-origin cross-validation.

**Datasets Processed:**
- `npu_sparse_panel.parquet` - Only hours with incidents
- `npu_dense_panel.parquet` - Complete NPU × hour grid
- `target_crimes.parquet` - Converted to hourly NPU × time panel

**Models Included:**
- **BaselineMean** - Simple mean predictor
- **LinearRegression** - Linear baseline
- **RandomForest** - Ensemble decision trees (200 estimators)
- **XGBRegressor** - Gradient boosting (400 estimators, optional)
- **CatBoostRegressor** - Categorical boosting (optional)

**Evaluation Strategy:**

1. **Simple Train/Test Split**
   - Train: Before 2024-01-01
   - Test: 2024-01-01 onwards
   - Metrics: MAE, RMSE, R², MAPE

2. **Rolling-Origin Cross-Validation** (4 folds)
   - Fold 1: Train <2022-01-01, Test 2022-01-01 to 2022-07-01
   - Fold 2: Train <2022-07-01, Test 2022-07-01 to 2023-01-01
   - Fold 3: Train <2023-01-01, Test 2023-01-01 to 2023-07-01
   - Fold 4: Train <2023-07-01, Test 2023-07-01 to 2024-01-01

**Output Structure:**
```
reports/test_results/<dataset_name>/
├── simple_results.csv                 # Simple split metrics
├── model_leaderboard_summary.csv      # CV aggregated results
├── cv_results/
│   ├── <Model>_cv_metrics.csv        # Per-fold metrics
│   ├── folds/
│   │   ├── <Model>_fold1_train.csv   # Training data splits
│   │   └── <Model>_fold1_test.csv    # Test data splits
│   └── predictions/
│       └── <Model>_all_predictions.csv  # All fold predictions
└── model_cards/
    ├── <dataset>_dataset_card.md      # Dataset description
    ├── <dataset>_model_card.md        # Model performance summary
    └── <dataset>_kaggle_card.md       # Kaggle-formatted card
```

**Global Outputs:**
```
reports/test_results/
├── combined_leaderboard.csv           # All datasets, all models
├── combined_leaderboard.md            # Markdown table
└── figures/leaderboard/
    └── combined_leaderboard_scatter.png  # CV vs Simple MAE plot
```

**Usage:**
```bash
python -m atl_model_pipelines.models.rolling_cv
```

**W&B Integration:**
- Project: "Data Science Capstone - Final Tests"
- Group: "rolling_cv"
- Logs: Per-model CV metrics, fold-level performance

---

### **2. horizon_models.py** - Multi-Horizon Forecasting

**Purpose:** Predict crime counts at multiple future time horizons (1, 2, 3, 6, 12, 24 hours ahead) using a single multi-output model.

**Model Architecture:**
- Base: XGBRegressor with Poisson objective
- Wrapper: MultiOutputRegressor for simultaneous horizon prediction
- Parameters:
  - n_estimators: 250
  - max_depth: 5
  - learning_rate: 0.05
  - subsample: 0.9
  - colsample_bytree: 0.9

**Features Used:**
- Current crime count
- Temporal: hour_of_day, day_of_week, month, is_weekend
- Cyclical: sin/cos encodings for hour and day
- Lags: 1, 2, 3, 6, 12, 24 hours
- Rolling means: 3, 6, 12, 24 hours

**Horizons:**
- h1: 1 hour ahead
- h2: 2 hours ahead
- h3: 3 hours ahead
- h6: 6 hours ahead
- h12: 12 hours ahead
- h24: 24 hours ahead

**Data Split:**
- Train: 2020-01-01 to 2023-12-31
- Test: 2024-01-01 to 2025-12-31

**Output Structure:**
```
reports/test_results/horizon_models/
├── horizon_metrics.csv                    # Per-horizon RMSE/MAE
├── predictions/
│   └── hourly_npu_predictions_multi_horizon.csv  # All predictions
└── artifacts/horizon_models/
    ├── xgb_multioutput_hourly_npu_lite.joblib   # Trained model
    └── meta_hourly_npu_lite.json                # Model metadata
```

**Metrics Tracked:**
- RMSE per horizon (rmse_h1, rmse_h2, ...)
- MAE per horizon (mae_h1, mae_h2, ...)
- Average RMSE across all horizons
- Average MAE across all horizons

**Usage:**
```bash
python -m atl_model_pipelines.models.horizon_models --parquet data/processed/apd/target_crimes.parquet
```

**W&B Integration:**
- Project: "Data Science Capstone - Final Tests"
- Group: "multi_horizon"
- Name: "Multi_Horizon_XGB_Base"
- Artifacts: Model + metadata

---

### **3. poisson_models.py** - Count-Based Model Comparison

**Purpose:** Compare specialized count/Poisson regression models for crime prediction with proper handling of zero-inflated distributions.

**Models Included:**

1. **XGBoost (Poisson)**
   - Objective: count:poisson
   - 800 estimators, depth 6
   - Learning rate: 0.05

2. **LightGBM (Poisson)**
   - Objective: poisson
   - 1000 estimators, 64 leaves
   - Learning rate: 0.05

3. **CatBoost (Poisson)**
   - Loss: Poisson
   - 800 iterations, depth 6
   - Handles categorical features natively

4. **Zero-Inflated Poisson (ZIP)**
   - Statistical model for excess zeros
   - Uses statsmodels implementation
   - Sampled training (150k rows) for efficiency

5. **Prophet (Hourly Total)**
   - Time series model on aggregated citywide counts
   - Captures seasonal patterns
   - Daily/weekly/yearly seasonality

**Baselines:**
- Naive 24h: Previous day same hour
- Naive 168h: Previous week same hour

**Features Used:**
```python
feature_cols = [
    "grid_density_7d", "npu_crime_avg_30d",
    "temp_f", "is_raining", "is_hot", "is_cold", "is_daylight",
    "is_weekend", "is_holiday", "day_number", "month",
    "hour_sin", "hour_cos", "day_of_week_code",
    "campus_distance_m", "location_type_count",
    "lag_1", "lag_3", "lag_6", "lag_12", "lag_24", "lag_168",
    "npu_code", "time_block_code"
]
```

**Data Requirements:**
- Input: `npu_dense_panel.parquet` (recommended)
- Time blocks: 6 predefined blocks (Late Night, Early Morning, etc.)
- Lags: 1, 3, 6, 12, 24, 168 hours

**Output Structure:**
```
reports/test_results/current_modeling/
├── model_metrics.json                     # All model metrics
└── predictions/
    ├── baselines/
    │   ├── naive_24h_predictions.csv
    │   └── naive_168h_predictions.csv
    ├── xgboost_poisson.csv
    ├── lightgbm_poisson.csv
    ├── catboost_poisson.csv
    ├── zip.csv
    └── prophet_hourly_total.csv
```

**Metrics Tracked:**
- RMSE: Root mean squared error
- MAE: Mean absolute error (primary ranking metric)
- R²: Coefficient of determination

**Usage:**
```bash
# Default (dense panel)
python -m atl_model_pipelines.models.poisson_models

# Custom data and cutoff
python -m atl_model_pipelines.models.poisson_models \
    --data data/processed/apd/npu_dense_panel.parquet \
    --cutoff "2024-01-01 00:00:00" \
    --output reports/test_results/poisson_test
```

**W&B Integration:**
- Project: "Data Science Capstone - Final Tests"
- Group: "poisson_models"
- Name: "Poisson_Count_Model_Comparison"
- Logs: All model metrics + best model selection

---

### **4. pipeline.py** - End-to-End Pipeline Runner

**Purpose:** Orchestrate the complete data pipeline from raw ingestion through validation.

**Pipeline Stages:**

1. **Ingestion** (`run_ingestion()`)
   - Load APD CSV files
   - Fetch/cache weather data
   - Load shapefiles
   - Combine and deduplicate

2. **Transformation** (`run_transforms()`)
   - Spatial enrichment
   - Temporal features
   - Weather merging
   - Campus distance calculation
   - Grid density computation
   - NPU rolling averages
   - Target offense filtering
   - Final dataset preparation

3. **Validation** (`run_validations()`)
   - Integrity checks
   - Completeness validation
   - Missingness reports
   - Schema validation

4. **Output**
   - Saves: `data/processed/apd/target_crimes_final.parquet`
   - Full pipeline logging
   - Validation reports

**Usage:**
```bash
python -m atl_model_pipelines.models.pipeline
```

**Pipeline Flow:**
```
Raw CSVs → Ingestion → Transforms → Validation → Final Dataset
    ↓          ↓            ↓            ↓            ↓
  Combine   Spatial    Integrity    Quality    Ready for
  Dedupe    Temporal   Checks       Reports    Modeling
           Weather
           Campus
```

---

##  **Common Output Locations**

All paths are defined in `config.py`:

```python
# Data
DATA_SPARSE = PROCESSED_DIR / "npu_sparse_panel.parquet"
DATA_DENSE = PROCESSED_DIR / "npu_dense_panel.parquet"
DATA_TARGET = PROCESSED_DIR / "target_crimes.parquet"

# Results
TEST_RESULTS_DIR = REPORTS_DIR / "test_results"
LEADERBOARD_DIR = FIGURES_DIR / "leaderboard"
CARDS_DIR = REPORTS_DIR / "cards"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WANDB_OUTPUT_DIR = ARTIFACTS_DIR / "wandb"
CATBOOST_OUTPUT_DIR = ARTIFACTS_DIR / "catboost_info"
```

---

##  **Quick Start**

### **Run Complete Pipeline**
```bash
# 1. Full data pipeline
python -m atl_model_pipelines.models.pipeline

# 2. Build panels
python -m atl_model_pipelines.transform.panel_builder

# 3. Run all evaluations
python -m atl_model_pipelines.models.rolling_cv
python -m atl_model_pipelines.models.horizon_models
python -m atl_model_pipelines.models.poisson_models
```

### **Run Individual Models**
```bash
# Rolling CV only
python -m atl_model_pipelines.models.rolling_cv

# Horizon forecasting
python -m atl_model_pipelines.models.horizon_models --parquet data/processed/apd/target_crimes.parquet

# Poisson models
python -m atl_model_pipelines.models.poisson_models --data data/processed/apd/npu_dense_panel.parquet
```

---

##  **Weights & Biases Integration**

All model scripts integrate with W&B for experiment tracking:

**Project Configuration:**
- **Project:** "Data Science Capstone - Final Tests"
- **Entity:** "joshuadariuspina-georgia-state-university"
- **Output:** `artifacts/wandb/`

**Groups:**
- `rolling_cv` - Cross-validation experiments
- `multi_horizon` - Horizon forecasting runs
- `poisson_models` - Count model comparisons

**Logged Metrics:**
- Training/test set sizes
- Per-fold metrics (CV)
- Per-horizon metrics (multi-horizon)
- Per-model metrics (Poisson)
- Feature importance (when available)
- Model artifacts

**Disable W&B:**
```bash
export WANDB_MODE=disabled
python -m atl_model_pipelines.models.rolling_cv
```

---

##  **Dependencies**

### **Required:**
```bash
pip install pandas numpy scikit-learn joblib
```

### **Optional (Enhanced Functionality):**
```bash
# Tree-based models
pip install xgboost lightgbm catboost

# Time series
pip install prophet statsmodels

# Tracking & visualization
pip install wandb matplotlib seaborn rich
```

### **Check Availability:**
Models gracefully handle missing optional dependencies:
- XGBoost: `HAS_XGB`
- CatBoost: `HAS_CAT` / `HAS_CATBOOST`
- Prophet: `HAS_PROPHET`
- W&B: `HAS_WANDB`

---

## **Model Cards & Documentation**

Each modeling run generates comprehensive documentation:

### **Dataset Cards** (`reports/cards/`)
- Schema description
- Date range coverage
- Row/column counts
- Train/test recommendations
- Intended use and limitations

### **Model Cards** (`reports/cards/`)
- Model architecture
- Hyperparameters
- Performance metrics (CV + simple split)
- Feature importance
- Training/inference time

### **Kaggle Cards** (for dataset sharing)
- Public-facing dataset documentation
- Citation information
- License details

---

##  **Performance Benchmarks**

### **Typical Results (NPU Dense Panel)**

| Model | MAE | RMSE | R² | Notes |
|-------|-----|------|----|----|
| XGBoost (Poisson) | ~0.85 | ~1.20 | ~0.45 | Best overall |
| LightGBM (Poisson) | ~0.88 | ~1.22 | ~0.43 | Fast training |
| CatBoost (Poisson) | ~0.87 | ~1.21 | ~0.44 | Handles categoricals |
| Random Forest | ~0.95 | ~1.30 | ~0.38 | Good baseline |
| Linear Regression | ~1.10 | ~1.45 | ~0.28 | Simple baseline |
| Naive 24h | ~1.25 | ~1.60 | ~0.15 | Persistence baseline |

*Results vary by dataset and time period*

---

##  **Troubleshooting**

### **Missing Dependencies**
```python
# Check what's available
from atl_model_pipelines.models.rolling_cv import (
    HAS_XGB, HAS_CAT, HAS_WANDB
)
print(f"XGBoost: {HAS_XGB}, CatBoost: {HAS_CAT}, W&B: {HAS_WANDB}")
```

### **Out of Memory**
```python
# In poisson_models.py, reduce ZIP sample size
train_zip(train_df, test_df, feature_cols, sample_size=50000)  # Default: 150000
```

### **Slow Training**
```python
# Reduce estimators in rolling_cv.py
models["XGBRegressor"] = XGBRegressor(
    n_estimators=100,  # Default: 400
    n_jobs=-1
)
```

### **W&B Errors**
```bash
# Login to W&B
wandb login

# Or disable
export WANDB_MODE=disabled
```

---

##  **Additional Resources**

- **Main Pipeline README:** `atl_model_pipelines/README.md`
- **Transform Documentation:** `atl_model_pipelines/transform/`
- **Validation Guide:** `atl_model_pipelines/validate/VALIDATION_GUIDE.md`
- **Config Reference:** `config.py`

---

##  **Citation**

If using this code for research or publication:

```bibtex
@software{atl_crime_prediction,
  title={"placeholder until we complete"},
  author={Your Name},
  year={2025},
  institution={Georgia State University},
  note={Data Science Capstone Project}
}
```

---

## 📄 **License**

For educational and research use only. Derived from Atlanta Police Department open data.

---

**Last Updated:** 2025-12-04  
**Maintained By:** Joshua Darius Piña 
**Institution:** Georgia State University

---

## 👥 **Contact**

**Authors:** Gunn Madan, Harini Mohan, Joshua Piña, Yuntian "Robin" Wu
**Institution:** Georgia State University  
**Program:** Data Science Capstone  
**Year:** 2025

---
