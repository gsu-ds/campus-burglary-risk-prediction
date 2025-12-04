# Logging Module Implementation Guide

## What I Created

I've created a proper logging utility module based on your notebook version that tracks pipeline execution.

## File Structure

Place these files in your project:

```
your-project/
├── config.py
├── atl_model_pipelines/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py          ← Rename utils_init.py to this
│   │   └── logging.py           ← New logging module
│   ├── transform/
│   │   ├── __init__.py
│   │   ├── cleaning.py          ← Updated
│   │   ├── spatial.py           ← Updated
│   │   └── ...
```

## What Changed

### 1. Created `atl_model_pipelines/utils/logging.py`
- Contains your original `log_step()` function
- Contains your original `show_pipeline_table()` function  
- Added `clear_pipeline_log()` helper function
- Maintains global `pipeline_log` list to track all steps

### 2. Updated Transform Scripts
- `cleaning.py` - Now imports from `atl_model_pipelines.utils.logging`
- `spatial.py` - Now imports from `atl_model_pipelines.utils.logging`

### 3. Created `atl_model_pipelines/utils/__init__.py`
- Exports logging functions for easy imports

## How to Use

### In Your Transform Scripts
```python
from atl_model_pipelines.utils.logging import log_step

def my_transform(df):
    df = df.copy()
    # ... transformations ...
    log_step("My Transform Step", df)
    return df
```

### In Your Orchestrator/Main Pipeline
```python
from atl_model_pipelines.utils.logging import log_step, show_pipeline_table, clear_pipeline_log

# Clear log at start of pipeline
clear_pipeline_log()

# Run your pipeline steps
df = step1(df)
df = step2(df)
df = step3(df)

# Show summary table at the end
show_pipeline_table()
```

### Example Output

When you call `log_step()`:
```
Step: Clean data shape: 50,000 x 25
```

When you call `show_pipeline_table()`:
```
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ Step                 ┃ Rows    ┃ Cols ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ Load raw data        │ 100,000 │ 30   │
│ Clean data           │ 95,000  │ 25   │
│ Add spatial features │ 95,000  │ 32   │
│ Final output         │ 95,000  │ 32   │
└──────────────────────┴─────────┴──────┘
```

## Features

✅ **Global tracking** - All pipeline steps logged automatically  
✅ **Rich formatting** - Beautiful console output with colors  
✅ **Table summary** - See entire pipeline at a glance  
✅ **Handles edge cases** - Works with empty DataFrames  
✅ **Easy to clear** - Reset log between pipeline runs  

## Installation Steps

1. Create the directory: `mkdir -p atl_model_pipelines/utils`
2. Place `logging.py` in `atl_model_pipelines/utils/`
3. Rename `utils_init.py` to `__init__.py` and place in `atl_model_pipelines/utils/`
4. Replace your existing `cleaning.py` and `spatial.py` with the updated versions
5. Make sure you have `from rich.table import Table` available (it's part of the `rich` package)

## Dependencies

- pandas
- rich (for console output and tables)
