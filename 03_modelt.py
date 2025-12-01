# %% [markdown]
# # 03 - Model T: Standardized Crime Prediction Pipeline
# 
# > **"Any model you want, as long as it's properly cross-validated."** 
# > *— Henry Ford (probably)*
# 
# **Purpose:** Mass-produce model evaluations with a standardized, efficient pipeline
# 
# **Input:** 
# - `data/processed/apd/target_crimes.csv`
# 
# **Output:** 
# - `data/processed/apd/cv_results/*_cv_metrics.csv` (performance summaries)
# - `data/processed/apd/cv_results/predictions/*.csv` (out-of-sample predictions)
# - `models/*.pkl` (trained model artifacts)
# 
# **The Assembly Line (Rolling CV):**
# ```
# Raw Data → Feature Prep → Train/Test Split → Model Fit → Predict → Evaluate → Repeat
#                              ↑_______________|_________________________|
#                                           (4 Folds)
# ```
# 
# **Models on the Line:**
# 1. ⚙️ Baseline Models (Quality Control)
#    - Seasonal Naive (1-day, 7-day)
#    - Historical Mean (30-day)
# 
# 2. 🚀 Production Models
#    - XGBoost (Poisson objective)
#    - CatBoost (Poisson loss)
#    - LightGBM (Poisson objective)
#    - Zero-Inflated Poisson (ZIP) - if needed
# 
# **Quality Metrics:**
# - MAE (Mean Absolute Error)
# - RMSE (Root Mean Squared Error)
# - R² (Coefficient of Determination)
# - MAPE (Mean Absolute Percentage Error)
# 
# **Timeline:** 4-fold rolling origin (2022 H1 → 2022 H2 → 2023 H1 → 2023 H2)
# 
# **Team:** Run this after `02_explorer.ipynb` to evaluate model performance
# 
# **Runtime:** ~15-30 minutes (with hyperparameter tuning)
# **Est. Completion:** 1913 (or whenever your laptop finishes) 🕰️

# %%
# === Imports =================================================================
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Initialize console
console = Console()

console.print(
    Panel.fit(
        "Libraries imported successfully for modeling.\n\n"
        "Ready to:\n"
        "- Build baseline models\n"
        "- Train ML models (XGBoost, CatBoost, LightGBM)\n"
        "- Perform rolling cross-validation\n"
        "- Evaluate and compare performance",
        title="03 Model T - Imports Complete",
        border_style="cyan",
    )
)

# %%
# === Configuration ===========================================================
DATA_DIR = Path("../data")
PROCESSED_DATA_FOLDER = DATA_DIR / "processed" / "apd"
MODELS_DIR = Path("../models")

# Create models directory
MODELS_DIR.mkdir(parents=True, exist_ok=True)

console.print(
    Panel.fit(
        "[bold cyan]Paths configured.[/bold cyan]\n\n"
        f"Data: [yellow]{PROCESSED_DATA_FOLDER}[/yellow]\n"
        f"Models: [yellow]{MODELS_DIR}[/yellow]",
        title="Configuration",
        border_style="cyan",
    )
)

# %%
# === Pipeline Logging ========================================================
pipeline_log: List[Dict[str, Any]] = []

def log_step(step_name: str, df: pd.DataFrame) -> None:
    """Record a pipeline step with shape info."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        rows_val = "N/A"
        cols_val = "N/A"
        rows_str = rows_val
        cols_str = cols_val
    else:
        rows_val = int(df.shape[0])
        cols_val = int(df.shape[1])
        rows_str = f"{rows_val:,}"
        cols_str = str(cols_val)

    pipeline_log.append({"step": step_name, "rows": rows_val, "cols": cols_val})
    console.print(f"[green]✓ {step_name}[/green] [cyan]→ shape: {rows_str} x {cols_str}[/cyan]")

def show_pipeline_table() -> None:
    """Display a Rich table summarizing all steps."""
    if not pipeline_log:
        console.print("[red]No steps logged yet.[/red]")
        return

    table = Table(title="📊 Model T Pipeline Summary", show_lines=True)
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Rows", style="green")
    table.add_column("Cols", style="yellow")

    for entry in pipeline_log:
        rows_val = entry['rows']
        cols_val = entry['cols']
        rows_str = f"{rows_val:,}" if isinstance(rows_val, int) else str(rows_val)
        cols_str = str(cols_val) if isinstance(cols_val, int) else str(cols_val)
        table.add_row(entry["step"], rows_str, cols_str)
    
    console.print(table)

console.print(Panel("[bold green]Logger configured.[/bold green]", border_style="green"))

# %%
# === Load Processed Data =====================================================
console.print(Panel("[bold cyan]Loading processed crime data...[/bold cyan]", border_style="cyan"))

INPUT_PATH = PROCESSED_DATA_FOLDER / "target_crimes.csv"

if not INPUT_PATH.exists():
    console.print(f"[bold red]ERROR:[/bold red] File not found: {INPUT_PATH}")
    console.print("[yellow]Please run 01_wrangler.ipynb first![/yellow]")
    raise FileNotFoundError(f"Required file missing: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
df['report_date'] = pd.to_datetime(df['report_date'])

log_step("Step 0: Loaded processed data", df)

console.print(
    Panel.fit(
        f"[bold green]✓ Data loaded successfully![/bold green]\n\n"
        f"Records: [cyan]{len(df):,}[/cyan]\n"
        f"Columns: [cyan]{len(df.columns)}[/cyan]\n"
        f"Date range: [cyan]{df['report_date'].min().date()} to {df['report_date'].max().date()}[/cyan]",
        title="Data Summary",
        border_style="green",
    )
)

# %% [markdown]
# ## 🏭 Section 1: The Factory Floor (Setup)
# Setting up our production line...

# %%
console.print(
    Panel(
        "[bold magenta]STEP 1: Define modeling utilities for time-series cross-validation[/bold magenta]",
        border_style="magenta",
    )
)

# Define output directories for CV results
CV_RESULTS_DIR = PROCESSED_DATA_FOLDER / "cv_results"
CV_FOLDS_DIR = CV_RESULTS_DIR / "folds"
CV_PREDICTIONS_DIR = CV_RESULTS_DIR / "predictions"

# Create directories
CV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CV_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
CV_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

console.print(
    f"[green]✓ CV output directories created:[/green]\n"
    f"  Results: {CV_RESULTS_DIR}\n"
    f"  Folds: {CV_FOLDS_DIR}\n"
    f"  Predictions: {CV_PREDICTIONS_DIR}"
)


def run_rolling_cv(
    model, 
    df, 
    target_col='crime_count', 
    date_col='report_date',
    feature_cols=None,
    group_col='npu',
    save_outputs=True,
    model_name=None
):
    """
    Performs Rolling Origin Cross-Validation on Panel Data with auto-save.
    
    Args:
        model: The initialized model (e.g., XGBRegressor, CatBoostRegressor)
        df: The full dataframe (Must contain date_col, group_col, features)
        target_col: Name of the target variable
        date_col: Name of the date column
        feature_cols: List of feature columns (if None, auto-detect)
        group_col: Column for grouping (NPU, zone, etc.)
        save_outputs: If True, saves train/test splits and predictions to CSV
        model_name: Name for output files (defaults to model class name)
    
    Returns:
        metrics_df: A dataframe showing MAE/RMSE/R² for each fold
        predictions_df: Out-of-sample predictions for all folds
        
    Saves:
        - {CV_FOLDS_DIR}/{model_name}_fold{i}_train.csv
        - {CV_FOLDS_DIR}/{model_name}_fold{i}_test.csv
        - {CV_PREDICTIONS_DIR}/{model_name}_all_predictions.csv
        - {CV_RESULTS_DIR}/{model_name}_cv_metrics.csv
    """
    
    # Determine model name for file outputs
    if model_name is None:
        model_name = model.__class__.__name__ if hasattr(model, '__class__') else str(model)
    
    # Clean model name for filenames
    model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    # 1. Prepare aggregated data if needed
    console.print(f"[cyan]Preparing data for rolling CV ({model_name})...[/cyan]")
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # If target_col doesn't exist, create aggregated crime counts
    if target_col not in df.columns:
        console.print(f"[yellow]Creating aggregated {target_col} by {group_col}...[/yellow]")
        df['date_only'] = df[date_col].dt.date
        
        # Aggregate to daily level by group
        agg_df = df.groupby([group_col, 'date_only']).size().reset_index(name=target_col)
        agg_df = agg_df.rename(columns={'date_only': date_col})
        df = agg_df
        df[date_col] = pd.to_datetime(df[date_col])
    
    # 2. Define Timeline Cutoffs
    folds = [
        ('2022-01-01', '2022-07-01'),  # Fold 1: Test H1 2022
        ('2022-07-01', '2023-01-01'),  # Fold 2: Test H2 2022
        ('2023-01-01', '2023-07-01'),  # Fold 3: Test H1 2023
        ('2023-07-01', '2024-01-01'),  # Fold 4: Test H2 2023
    ]
    
    results = []
    all_predictions = []
    
    console.print(f"[bold cyan]Starting Rolling CV for {model_name}...[/bold cyan]")
    
    for i, (train_end, test_end) in enumerate(folds):
        fold_num = i + 1
        fold_name = f"Fold {fold_num}"
        
        # 3. Create Time-Based Splits
        train_mask = df[date_col] < train_end
        test_mask = (df[date_col] >= train_end) & (df[date_col] < test_end)
        
        train_data = df[train_mask].copy()
        test_data = df[test_mask].copy()
        
        if len(test_data) == 0:
            console.print(f"[yellow]  {fold_name}: No test data, skipping.[/yellow]")
            continue
        
        # 4. Save train/test splits if requested
        if save_outputs:
            train_path = CV_FOLDS_DIR / f"{model_name}_fold{fold_num}_train.csv"
            test_path = CV_FOLDS_DIR / f"{model_name}_fold{fold_num}_test.csv"
            
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            console.print(f"[dim cyan]    Saved: {train_path.name} ({len(train_data):,} rows)[/dim cyan]")
            console.print(f"[dim cyan]    Saved: {test_path.name} ({len(test_data):,} rows)[/dim cyan]")
        
        # 5. Prepare Features
        if feature_cols is None:
            # Auto-detect: drop non-feature columns
            drop_cols = [target_col, date_col, 'date_only', group_col, 'geometry']
            drop_cols = [c for c in drop_cols if c in train_data.columns]
            feature_cols = [c for c in train_data.columns if c not in drop_cols]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # 6. Train & Predict
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            # Handle baseline models (custom predict signature)
            preds = model.predict(test_data, train_data)
        
        # 7. Calculate Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        # Additional metrics
        mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-10))) * 100
        
        console.print(
            f"[green]  {fold_name} ({train_end} to {test_end}):[/green] "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%"
        )
        
        results.append({
            'Fold': fold_name,
            'Fold_Number': fold_num,
            'Train_End': train_end,
            'Test_End': test_end,
            'Train_Size': len(train_data),
            'Test_Size': len(test_data),
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        })
        
        # Store predictions for analysis
        pred_df = test_data[[date_col, group_col, target_col]].copy()
        pred_df['predicted'] = preds
        pred_df['residual'] = y_test.values - preds
        pred_df['abs_error'] = np.abs(pred_df['residual'])
        pred_df['fold'] = fold_name
        pred_df['fold_number'] = fold_num
        all_predictions.append(pred_df)
    
    # 8. Compile results
    metrics_df = pd.DataFrame(results)
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    
    # 9. Save summary outputs
    if save_outputs and len(metrics_df) > 0:
        metrics_path = CV_RESULTS_DIR / f"{model_name}_cv_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        console.print(f"[bold green]  ✓ Saved metrics: {metrics_path.name}[/bold green]")
        
        if len(predictions_df) > 0:
            predictions_path = CV_PREDICTIONS_DIR / f"{model_name}_all_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            console.print(f"[bold green]  ✓ Saved predictions: {predictions_path.name}[/bold green]")
    
    # 10. Summary statistics
    console.print("\n[bold green]Cross-Validation Summary:[/bold green]")
    summary_table = Table(title=f"{model_name} - CV Performance", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Mean", style="green")
    summary_table.add_column("Std", style="yellow")
    summary_table.add_column("Min", style="red")
    summary_table.add_column("Max", style="magenta")
    
    for metric in ['MAE', 'RMSE', 'R²', 'MAPE']:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        min_val = metrics_df[metric].min()
        max_val = metrics_df[metric].max()
        
        if metric == 'MAPE':
            summary_table.add_row(
                metric, 
                f"{mean_val:.2f}%", 
                f"±{std_val:.2f}%",
                f"{min_val:.2f}%",
                f"{max_val:.2f}%"
            )
        else:
            summary_table.add_row(
                metric, 
                f"{mean_val:.4f}", 
                f"±{std_val:.4f}",
                f"{min_val:.4f}",
                f"{max_val:.4f}"
            )
    
    console.print(summary_table)
    
    return metrics_df, predictions_df

# %% [markdown]
# ## ⚙️ Section 2: Quality Control (Baseline Models)
# Before we build fancy models, let's establish minimum standards...

# %%
# Baseline Models (Seasonal Naive)
class SeasonalNaive:
    """Baseline: Uses last period's value as prediction."""
    
    def __init__(self, period=7):
        """
        Args:
            period: Seasonality period (7 for weekly, 1 for daily)
        """
        self.period = period
        self.name = f"Seasonal_Naive_{period}d"
    
    def __repr__(self):
        return self.name
    
    def predict(self, test_data, train_data):
        """Predict using last period's values."""
        predictions = []
        
        for idx, row in test_data.iterrows():
            target_date = row['report_date'] - pd.Timedelta(days=self.period)
            
            # Find matching historical record
            match = train_data[
                (train_data['report_date'] == target_date) & 
                (train_data['npu'] == row['npu'])
            ]
            
            if len(match) > 0:
                predictions.append(match['crime_count'].iloc[0])
            else:
                # Fallback to overall mean if no match
                predictions.append(train_data['crime_count'].mean())
        
        return np.array(predictions)


class HistoricalMean:
    """Baseline: Uses historical mean by group."""
    
    def __init__(self, window_days=30):
        self.window_days = window_days
        self.name = f"Historical_Mean_{window_days}d"
    
    def __repr__(self):
        return self.name
    
    def predict(self, test_data, train_data):
        """Predict using historical mean."""
        predictions = []
        
        for idx, row in test_data.iterrows():
            # Calculate mean from last N days in training data
            cutoff_date = row['report_date'] - pd.Timedelta(days=self.window_days)
            
            historical = train_data[
                (train_data['report_date'] >= cutoff_date) &
                (train_data['npu'] == row['npu'])
            ]
            
            if len(historical) > 0:
                predictions.append(historical['crime_count'].mean())
            else:
                predictions.append(train_data['crime_count'].mean())
        
        return np.array(predictions)


console.print("[green]✓ Rolling CV utilities and baseline models defined.[/green]")
console.print(f"[cyan]Output structure:[/cyan]\n"
             f"  📁 {CV_RESULTS_DIR.relative_to(DATA_DIR)}/\n"
             f"    ├─ 📊 [model_name]_cv_metrics.csv\n"
             f"    ├─ 📁 folds/\n"
             f"    │   ├─ [model_name]_fold1_train.csv\n"
             f"    │   ├─ [model_name]_fold1_test.csv\n"
             f"    │   └─ ... (8 files per model)\n"
             f"    └─ 📁 predictions/\n"
             f"        └─ [model_name]_all_predictions.csv")

log_step("Step 1: Modeling utilities with auto-save functionality", pd.DataFrame())

# %% [markdown]
# ### 📖 For Data Science Team: Accessing CV Results
# 
# **Location:** `data/processed/apd/cv_results/`
# 
# **Quick Access:**
# ```python
# # Load summary metrics for a model
# xgb_metrics = pd.read_csv('data/processed/apd/cv_results/XGBRegressor_cv_metrics.csv')
# 
# # Load predictions for error analysis
# xgb_preds = pd.read_csv('data/processed/apd/cv_results/predictions/XGBRegressor_all_predictions.csv')
# 
# # Load specific fold data
# fold1_train = pd.read_csv('data/processed/apd/cv_results/folds/XGBRegressor_fold1_train.csv')
# fold1_test = pd.read_csv('data/processed/apd/cv_results/folds/XGBRegressor_fold1_test.csv')
# ```
# 
# **Compare all models:**
# ```python
# import glob
# 
# # Load all CV metrics
# metrics_files = glob.glob('data/processed/apd/cv_results/*_cv_metrics.csv')
# all_metrics = []
# 
# for file in metrics_files:
#     df_temp = pd.read_csv(file)
#     model_name = Path(file).stem.replace('_cv_metrics', '')
#     df_temp['Model'] = model_name
#     all_metrics.append(df_temp)
# 
# comparison = pd.concat(all_metrics, ignore_index=True)
# print(comparison.groupby('Model')[['MAE', 'RMSE', 'R²']].mean().sort_values('MAE'))
# ```

# %% [markdown]
# ## 🚀 Section 3: The Production Line (Ready to Run Models)
# 
# **Your models are ready to run!** Add execution cells below to:
# 
# 1. **Prepare aggregated data** (daily crime counts by NPU)
# 2. **Run baseline models** (Seasonal Naive 1d, 7d, Historical Mean)
# 3. **Run ML models** (XGBoost, CatBoost, LightGBM with Poisson objective)
# 4. **Compare results** across all models
# 
# **Example execution code:**
# ```python
# # Aggregate to daily NPU level
# daily_npu = df.groupby(['npu', df['report_date'].dt.date]).size().reset_index(name='crime_count')
# daily_npu = daily_npu.rename(columns={'report_date': 'date'})
# daily_npu['report_date'] = pd.to_datetime(daily_npu['date'])
# 
# # Add temporal features
# daily_npu['day_of_week'] = daily_npu['report_date'].dt.dayofweek
# daily_npu['month'] = daily_npu['report_date'].dt.month
# daily_npu['is_weekend'] = daily_npu['day_of_week'].isin([5, 6]).astype(int)
# 
# # Run baseline
# naive_1d = SeasonalNaive(period=1)
# results_1d, preds_1d = run_rolling_cv(naive_1d, daily_npu)
# 
# # Run XGBoost
# from xgboost import XGBRegressor
# xgb_model = XGBRegressor(objective='count:poisson', n_estimators=100, random_state=42)
# results_xgb, preds_xgb = run_rolling_cv(xgb_model, daily_npu)
# ```

# %% [markdown]
# ## ✅ Model Selection & Deployment
# 
# **Best performing model:** [To be filled after running]
# 
# **Next steps:**
# 1. Review error analysis - which NPUs/times have highest errors?
# 2. Consider ensemble methods (averaging top 3 models)
# 3. Deploy best model for real-time predictions
# 4. Set up monitoring dashboard
# 
# **Results location:** `data/processed/apd/cv_results/`

# %%
# === Final Summary ===========================================================
console.print("\n[bold magenta]═══ Model T Setup Complete ═══[/bold magenta]\n")

setup_summary = {
    "Data Loaded": f"{len(df):,} records",
    "CV Framework": "4-fold rolling origin (2022-2024)",
    "Baseline Models": "Seasonal Naive (1d, 7d), Historical Mean",
    "ML Models": "XGBoost, CatBoost, LightGBM (Poisson)",
    "Output Directory": str(CV_RESULTS_DIR),
}

setup_table = Table(show_header=False, show_lines=True)
setup_table.add_column("Component", style="cyan")
setup_table.add_column("Status", style="green")

for component, status in setup_summary.items():
    setup_table.add_row(component, status)

console.print(setup_table)
console.print(f"\n[bold green]✓ 03_modelt.ipynb setup complete![/bold green]")
console.print(f"[yellow]Ready to run models! Add execution cells in Section 3.[/yellow]")

show_pipeline_table()