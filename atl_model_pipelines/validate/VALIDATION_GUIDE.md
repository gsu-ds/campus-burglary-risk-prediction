# Validation Module - Usage Guide

## Structure

```
atl_model_pipelines/
└── validate/
    ├── __init__.py           # Module exports
    ├── core.py               # Basic validation checks
    ├── panel_checks.py       # Panel-specific validation
    ├── orchestrator.py       # Master validation runners
    └── wandb_logging.py      # W&B artifact logging
```

## Core Validation Functions

### 1. **Basic Data Validation**

```python
from atl_model_pipelines.validate import run_validation_checks

# Run integrity checks
run_validation_checks(
    df,
    step_name="After spatial enrichment",
    check_npu=True  # Optional: validate NPU categories
)
```

**Checks:**
- ✅ `incident_number` uniqueness
- ✅ Coordinate bounds (Atlanta area)
- ✅ Core column completeness (report_date, nibrs_offense)
- ✅ NPU category integrity (optional)

### 2. **Missing Data Comparison**

```python
from atl_model_pipelines.validate import show_missing_comparison, create_snapshot

# Before transformation
snapshot = create_snapshot(df)

# ... apply transformations ...

# After transformation
show_missing_comparison(df, snapshot, "Spatial Enrichment")
```

Shows before/after missing counts with a nice Rich table.

---

## Panel Validation

### 1. **Quality Report**

```python
from atl_model_pipelines.validate import panel_quality_report

# Generate comprehensive quality report
panel_quality_report(panel_df)
```

**Output:**
- Row/column counts
- Top 20 columns with missing values
- Lag feature distribution
- Sample rows

### 2. **Schema Validation**

```python
from atl_model_pipelines.validate import validate_panel_schema

# Validate required columns exist
validate_panel_schema(panel_df, panel_type="burglary")
```

**Required columns:**
- Target: `burglary_count` or `crime_count`
- Time: `hour_ts`, `npu`
- Lags: `lag_1`, `lag_3`, `lag_6`, `lag_12`, `lag_24`, `lag_168`
- Weather: `temp_f`, `is_raining`, `is_hot`, `is_cold`
- Features: `grid_density_7d`, `npu_crime_avg_30d`
- Temporal: `hour_sin`, `hour_cos`, `day_number`, `day_of_week`

### 3. **Completeness Check**

```python
from atl_model_pipelines.validate import validate_panel_completeness

# Check critical features aren't too sparse
validate_panel_completeness(panel_df, max_missing_pct=0.1)
```

Ensures critical features have <10% missing values.

### 4. **Time Range Check**

```python
from atl_model_pipelines.validate import validate_panel_time_range

# Ensure sufficient time coverage
validate_panel_time_range(panel_df, min_hours=168)
```

Validates panel covers at least 1 week of data.

---

## Master Orchestrators

### 1. **Standard Data Validation**

```python
from atl_model_pipelines.validate import run_validations

# Run all standard checks
df = run_validations(
    df,
    step_name="Final dataset",
    check_npu=True
)
```

Runs:
- Integrity checks
- Missingness report

### 2. **Complete Panel Validation**

```python
from atl_model_pipelines.validate import run_panel_validations

# Run all panel checks
panel_df = run_panel_validations(
    panel_df,
    panel_type="burglary",
    check_schema=True,
    check_completeness=True,
    check_time_range=True,
    show_quality_report=True
)
```

Runs:
- Quality report
- Schema validation
- Completeness check
- Time range validation

---

## W&B Logging

### 1. **Log Single Panel**

```python
from atl_model_pipelines.validate import log_panel_artifact
import wandb

# Initialize W&B
wandb.init(project="atl-crime-prediction", job_type="panel_creation")

# Log panel
log_panel_artifact(
    df=sparse_panel,
    name="npu_sparse_panel",
    description="Sparse NPU × hour panel"
)
```

### 2. **Log Multiple Panels**

```python
from atl_model_pipelines.validate import log_multiple_panels

# Log all panels at once
log_multiple_panels(
    sparse_df=sparse_panel,
    dense_df=dense_panel,
    target_df=target_crimes,
    project_name="atl-crime-prediction"
)
```

**Note:** Install wandb first: `pip install wandb`

---

## Integration with Transform Pipeline

### Example: Full Pipeline with Validation

```python
from atl_model_pipelines.ingestion.ingestion_master import run_ingestion
from atl_model_pipelines.transform.transform_master import run_transforms
from atl_model_pipelines.transform.panel_builder import build_all_panels
from atl_model_pipelines.validate import (
    run_validations,
    run_panel_validations,
    log_multiple_panels
)
from atl_model_pipelines.utils.logging import show_pipeline_table

# 1. Ingestion
ingestion_output = run_ingestion()

# 2. Transforms with validation
transform_output = run_transforms(ingestion_output)
target_crimes = run_validations(
    transform_output["target_crimes"],
    step_name="Final target crimes",
    check_npu=True
)

# 3. Build panels with validation
panels = build_all_panels(df=target_crimes, save=True)

sparse_validated = run_panel_validations(
    panels["sparse"],
    panel_type="crime",
    show_quality_report=True
)

dense_validated = run_panel_validations(
    panels["dense"],
    panel_type="crime",
    show_quality_report=True
)

# 4. Log to W&B
log_multiple_panels(
    sparse_df=sparse_validated,
    dense_df=dense_validated,
    target_df=target_crimes
)

# 5. Show pipeline summary
show_pipeline_table()
```

---

## Usage in Notebooks

```python
# At the end of your notebook
from atl_model_pipelines.validate import (
    panel_quality_report,
    validate_panel_schema
)

# Quick quality check
panel_quality_report(panel_df)

# Validate schema before modeling
validate_panel_schema(panel_df, panel_type="burglary")
```

---

## Error Handling

All validation functions raise `ValueError` if checks fail:

```python
from atl_model_pipelines.validate import validate_panel_schema

try:
    validate_panel_schema(panel_df)
    print("✓ Panel ready for modeling")
except ValueError as e:
    print(f"✗ Panel validation failed: {e}")
    # Fix issues...
```

---

## Summary

| Function | Use Case |
|----------|----------|
| `run_validation_checks()` | Basic integrity checks |
| `show_missing_comparison()` | Before/after missingness |
| `panel_quality_report()` | Panel overview |
| `validate_panel_schema()` | Required columns check |
| `validate_panel_completeness()` | Feature sparsity check |
| `validate_panel_time_range()` | Time coverage check |
| `run_validations()` | Standard data validation |
| `run_panel_validations()` | Complete panel validation |
| `log_panel_artifact()` | W&B single panel logging |
| `log_multiple_panels()` | W&B multiple panels logging |

All validation functions are designed to be:
✅ Non-destructive (return original DataFrame)  
✅ Rich console output with colors  
✅ Informative error messages  
✅ Configurable thresholds