# %% [markdown]
# # 03 – NPU–Hour Burglary Modeling Notebook
#
# Uses `npu_sparse_panel` from 01_wrangler to:
# - Load panel data
# - Define rolling-origin CV utilities
# - Train multiple models (RF / XGB / CatBoost / Poisson + baselines)
# - Log metrics & leaderboard to disk + Weights & Biases

# %% [markdown]
# --- Section 0: Imports, Paths, Data Load ---

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor

# External ML libs (XGBoost / CatBoost)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Weights & Biases (for experiment tracking)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

console = Console()

# %% [markdown]
# ### 0.1 Resolve project paths

# %%
# Robust project root detection (works in notebook or script)
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # In a notebook, __file__ is not defined; assume we're in /notebooks
    ROOT = Path.cwd().resolve().parent

DATA_DIR = ROOT / "data"
PROCESSED_DATA_FOLDER = DATA_DIR / "processed" / "apd"

# Test results + CV outputs (as requested)
TEST_RESULTS_DIR = ROOT / "reports" / "test_results"
CV_RESULTS_DIR = TEST_RESULTS_DIR / "cv_results"
CV_FOLDS_DIR = CV_RESULTS_DIR / "folds"
CV_PREDICTIONS_DIR = CV_RESULTS_DIR / "predictions"
WANDB_DIR = TEST_RESULTS_DIR / "wandb"

for d in [TEST_RESULTS_DIR, CV_RESULTS_DIR, CV_FOLDS_DIR, CV_PREDICTIONS_DIR, WANDB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Direct catboost logs into test_results instead of notebook dir
CATBOOST_TRAIN_DIR = TEST_RESULTS_DIR / "catboost_info"
CATBOOST_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Point wandb to the correct directory
if HAS_WANDB:
    os.environ["WANDB_DIR"] = str(WANDB_DIR)

console.print(
    Panel.fit(
        f"[bold green]Paths configured[/bold green]\n"
        f"ROOT: {ROOT}\n"
        f"Processed: {PROCESSED_DATA_FOLDER}\n"
        f"Test results: {TEST_RESULTS_DIR}",
        border_style="green",
        title="Path Summary"
    )
)

# %% [markdown]
# ### 0.2 Load sparse NPU–hour panel

# %%
pipeline_log = []


def log_step(step_name: str, df: pd.DataFrame | None) -> None:
    """Simple pipeline logger for this notebook."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        rows_val, cols_val = "N/A", "N/A"
    else:
        rows_val, cols_val = f"{len(df):,}", df.shape[1]

    pipeline_log.append({"step": step_name, "rows": rows_val, "cols": cols_val})

    console.print(
        f"[green]{step_name}[/green] → "
        f"[cyan]rows={rows_val}, cols={cols_val}[/cyan]"
    )


def show_pipeline_table() -> None:
    table = Table(title="Model T Pipeline Summary")
    table.add_column("Step", style="cyan")
    table.add_column("Rows", style="green", justify="right")
    table.add_column("Cols", style="yellow", justify="right")
    for entry in pipeline_log:
        table.add_row(entry["step"], str(entry["rows"]), str(entry["cols"]))
    console.print(table)


console.print(
    Panel(
        "[bold cyan]Loading sparse NPU–hour panel...[/bold cyan]",
        border_style="cyan",
    )
)

PANEL_PARQUET = PROCESSED_DATA_FOLDER / "npu_sparse_panel.parquet"
PANEL_CSV = PROCESSED_DATA_FOLDER / "npu_sparse_panel.csv"

if PANEL_PARQUET.exists():
    df = pd.read_parquet(PANEL_PARQUET)
elif PANEL_CSV.exists():
    df = pd.read_csv(PANEL_CSV)
else:
    raise FileNotFoundError(
        f"Could not find npu_sparse_panel at {PANEL_PARQUET} or {PANEL_CSV}"
    )

df["hour_ts"] = pd.to_datetime(df["hour_ts"])
df = df.sort_values(["hour_ts", "npu"]).reset_index(drop=True)

log_step("Step 0: Loaded sparse NPU–hour panel", df)

console.print(
    Panel.fit(
        f"[bold green]✓ Panel loaded successfully![/bold green]\n\n"
        f"Rows: [cyan]{len(df):,}[/cyan]\n"
        f"Columns: [cyan]{df.shape[1]}[/cyan]\n"
        f"Date range: [cyan]{df['hour_ts'].min()} to {df['hour_ts'].max()}[/cyan]\n"
        f"NPUs: [cyan]{df['npu'].nunique()}[/cyan]",
        border_style="green",
        title="Panel Summary",
    )
)

# %% [markdown]
# ### 0.3 Feature preparation

# %%
console.print(
    Panel(
        "[bold cyan]Preparing features (one-hot, numeric only)[/bold cyan]",
        border_style="cyan",
    )
)

# One-hot encode day_of_week if present
if "day_of_week" in df.columns:
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow", drop_first=True)
    df = pd.concat([df, dow_dummies], axis=1)

# Potential feature set from panel script
POTENTIAL_FEATURES = [
    "grid_density_7d",
    "npu_crime_avg_30d",
    "temp_f",
    "is_raining",
    "is_hot",
    "is_cold",
    "is_daylight",
    "is_weekend",
    "is_holiday",
    "day_number",
    "month",
    "year",
    "hour_sin",
    "hour_cos",
    "campus_distance_m",
    "location_type_count",
]

# Add any dow_ columns
POTENTIAL_FEATURES += [c for c in df.columns if c.startswith("dow_")]

FEATURE_COLS = [c for c in POTENTIAL_FEATURES if c in df.columns]

TARGET_COL = "burglary_count"
DATE_COL = "hour_ts"
GROUP_COL = "npu"

console.print(
    Panel.fit(
        f"[bold green]Features configured[/bold green]\n"
        f"Target: [cyan]{TARGET_COL}[/cyan]\n"
        f"Date: [cyan]{DATE_COL}[/cyan]\n"
        f"Group: [cyan]{GROUP_COL}[/cyan]\n"
        f"Num features: [cyan]{len(FEATURE_COLS)}[/cyan]",
        border_style="green",
        title="Modeling Config",
    )
)

# %% [markdown]
# --- Section 1: Rolling-Origin CV Utilities ---

# %%
console.print(
    Panel(
        "[bold magenta]STEP 1: Define modeling utilities for time-series cross-validation[/bold magenta]",
        border_style="magenta",
    )
)


def run_rolling_cv(
    model,
    df_in: pd.DataFrame,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
    feature_cols: list[str] | None = None,
    group_col: str = GROUP_COL,
    save_outputs: bool = True,
    model_name: str | None = None,
):
    """
    Rolling-origin CV on NPU–hour panel.

    Assumes df_in already contains aggregated target per (npu, hour_ts).
    If model has .fit, uses standard sklearn-style API.
    Otherwise expects: preds = model.predict(test_df, train_df)  (for baselines).
    """

    # ------------------ Model name normalization ------------------
    if model_name is None:
        model_name = model.__class__.__name__ if hasattr(model, "__class__") else str(model)
    model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

    console.print(f"[cyan]Preparing data for rolling CV ({model_name})...[/cyan]")

    df_local = df_in.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])

    # ------------------ CV folds ------------------
    folds = [
        ("2022-01-01", "2022-07-01"),  # Fold 1
        ("2022-07-01", "2023-01-01"),  # Fold 2
        ("2023-01-01", "2023-07-01"),  # Fold 3
        ("2023-07-01", "2024-01-01"),  # Fold 4
    ]

    results = []
    all_predictions = []

    console.print(f"[bold cyan]Starting Rolling CV for {model_name}...[/bold cyan]")

    for i, (train_end, test_end) in enumerate(folds):
        fold_num = i + 1
        fold_name = f"Fold {fold_num}"

        train_mask = df_local[date_col] < train_end
        test_mask = (df_local[date_col] >= train_end) & (df_local[date_col] < test_end)

        train_data = df_local[train_mask].copy()
        test_data = df_local[test_mask].copy()

        if len(test_data) == 0:
            console.print(f"[yellow]  {fold_name}: No test data, skipping.[/yellow]")
            continue

        # ------------------ Save splits ------------------
        if save_outputs:
            train_path = CV_FOLDS_DIR / f"{model_name}_fold{fold_num}_train.csv"
            test_path = CV_FOLDS_DIR / f"{model_name}_fold{fold_num}_test.csv"
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

        # ------------------ Feature detection (per fold) ------------------
        if feature_cols is None:
            drop_cols = [target_col, date_col, group_col, "geometry"]
            drop_cols = [c for c in drop_cols if c in train_data.columns]
            local_feature_cols = [c for c in train_data.columns if c not in drop_cols]
        else:
            local_feature_cols = list(feature_cols)

        X_train = train_data[local_feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[local_feature_cols]
        y_test = test_data[target_col]

        # ------------------ Fit & predict ------------------
        if hasattr(model, "fit"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            # For custom baselines that don't implement .fit
            preds = model.predict(test_data, train_data)

        # ------------------ Metrics ------------------
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-10))) * 100

        console.print(
            f"[green]  {fold_name} ({train_end} → {test_end}):[/green] "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%"
        )

        results.append(
            {
                "Model": model_name,
                "Fold": fold_name,
                "Fold_Number": fold_num,
                "Train_End": train_end,
                "Test_End": test_end,
                "Train_Size": len(train_data),
                "Test_Size": len(test_data),
                "MAE": mae,
                "RMSE": rmse,
                "R²": r2,
                "MAPE": mape,
            }
        )

        pred_df = test_data[[date_col, group_col, target_col]].copy()
        pred_df["predicted"] = preds
        pred_df["residual"] = y_test.values - preds
        pred_df["abs_error"] = np.abs(pred_df["residual"])
        pred_df["fold"] = fold_name
        pred_df["fold_number"] = fold_num
        all_predictions.append(pred_df)

    metrics_df = pd.DataFrame(results)
    predictions_df = (
        pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    )

    # ------------------ Save summary ------------------
    if save_outputs and len(metrics_df) > 0:
        metrics_path = CV_RESULTS_DIR / f"{model_name}_cv_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        if len(predictions_df) > 0:
            predictions_path = CV_PREDICTIONS_DIR / f"{model_name}_all_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)

    # ------------------ Rich summary table ------------------
    console.print("\n[bold green]Cross-Validation Summary:[/bold green]")
    if len(metrics_df) > 0:
        summary_table = Table(title=f"{model_name} - CV Performance", show_header=True)
        summary_table.add_column("Metric")
        summary_table.add_column("Mean")
        summary_table.add_column("Std")
        summary_table.add_column("Min")
        summary_table.add_column("Max")

        for metric in ["MAE", "RMSE", "R²", "MAPE"]:
            summary_table.add_row(
                metric,
                f"{metrics_df[metric].mean():.4f}",
                f"±{metrics_df[metric].std():.4f}",
                f"{metrics_df[metric].min():.4f}",
                f"{metrics_df[metric].max():.4f}",
            )

        console.print(summary_table)
    else:
        console.print("[yellow]No folds produced metrics.[/yellow]")

    return metrics_df, predictions_df


log_step("Step 1: Modeling utilities with auto-save functionality", None)

# %% [markdown]
# --- Section 2: Baseline & Model Zoo Definitions ---

# %%
console.print(
    Panel(
        "[bold magenta]STEP 2: Define baselines and model zoo[/bold magenta]",
        border_style="magenta",
    )
)

# --- Simple baseline models (no .fit) ---------------------------------------
class NaiveMeanModel:
    """Predicts global mean of target from training set."""

    def predict(self, test_df: pd.DataFrame, train_df: pd.DataFrame):
        mean_val = train_df[TARGET_COL].mean()
        return np.full(shape=len(test_df), fill_value=mean_val)


class NaiveLastHourModel:
    """
    For each (npu, hour_ts) in test:
    predict last observed value for that NPU (or global mean if none).
    """

    def predict(self, test_df: pd.DataFrame, train_df: pd.DataFrame):
        # Build lookup from train
        train_sorted = train_df.sort_values(DATE_COL)
        last_vals = (
            train_sorted.groupby(GROUP_COL)[TARGET_COL].last().to_dict()
        )
        global_mean = train_sorted[TARGET_COL].mean()

        preds = []
        for _, row in test_df.iterrows():
            npu = row[GROUP_COL]
            preds.append(last_vals.get(npu, global_mean))
        return np.array(preds)


class SeasonalWeeklyMeanModel:
    """
    Seasonal weekly: mean per (NPU, hour-of-week) from training.
    hour_of_week = day_number * 24 + hour_of_day (approx, using day_number).
    """

    def predict(self, test_df: pd.DataFrame, train_df: pd.DataFrame):
        # Ensure day_number + hour exist
        tr = train_df.copy()
        ts = test_df.copy()

        if "day_number" not in tr.columns:
            tr["day_number"] = pd.to_datetime(tr[DATE_COL]).dt.weekday + 1
        if "day_number" not in ts.columns:
            ts["day_number"] = pd.to_datetime(ts[DATE_COL]).dt.weekday + 1

        tr["hour"] = pd.to_datetime(tr[DATE_COL]).dt.hour
        ts["hour"] = pd.to_datetime(ts[DATE_COL]).dt.hour

        tr["hour_of_week"] = (tr["day_number"] - 1) * 24 + tr["hour"]
        ts["hour_of_week"] = (ts["day_number"] - 1) * 24 + ts["hour"]

        key_cols = [GROUP_COL, "hour_of_week"]

        mean_lookup = (
            tr.groupby(key_cols)[TARGET_COL].mean().reset_index().set_index(key_cols)
        )

        global_mean = tr[TARGET_COL].mean()
        preds = []
        for _, row in ts.iterrows():
            key = (row[GROUP_COL], row["hour_of_week"])
            preds.append(mean_lookup[TARGET_COL].get(key, global_mean))

        return np.array(preds)


# --- Tree / GLM models ------------------------------------------------------
MODEL_ZOO: dict[str, object] = {}

# Baselines
MODEL_ZOO["NaiveMean"] = NaiveMeanModel()
MODEL_ZOO["NaiveLastHour"] = NaiveLastHourModel()
MODEL_ZOO["SeasonalWeekly"] = SeasonalWeeklyMeanModel()

# Random Forest
MODEL_ZOO["RandomForest"] = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

# XGBoost (if available)
if HAS_XGB:
    MODEL_ZOO["XGBRegressor"] = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
        random_state=42,
    )
else:
    console.print("[yellow]XGBoost not installed; skipping XGBRegressor.[/yellow]")

# CatBoost (if available)
if HAS_CATBOOST:
    MODEL_ZOO["CatBoost"] = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        iterations=400,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        train_dir=str(CATBOOST_TRAIN_DIR),
    )
else:
    console.print("[yellow]CatBoost not installed; skipping CatBoost.[/yellow]")

# Poisson GLM (sklearn)
MODEL_ZOO["PoissonGLM"] = PoissonRegressor(alpha=0.0, max_iter=1000)

console.print(
    Panel.fit(
        "[bold green]Model zoo initialized:[/bold green]\n"
        + "\n".join(f" • {name}" for name in MODEL_ZOO.keys()),
        border_style="green",
        title="Models",
    )
)

log_step("Step 2: Model zoo defined", None)

# %% [markdown]
# --- Section 3: Auto-Train Loop with W&B + Leaderboard ---

# %%
console.print(
    Panel(
        "[bold magenta]STEP 3.3: Running rolling CV for all models in MODEL_ZOO[/bold magenta]",
        border_style="magenta",
    )
)

# Configure W&B project/entity (edit these!)
WANDB_PROJECT = "campus-burglary-models"
WANDB_ENTITY = "your_wandb_username_or_team"  # <-- EDIT ME

all_cv_metrics: list[pd.DataFrame] = []
model_summary_rows: list[dict] = []

for model_name, model in MODEL_ZOO.items():

    # 1. Initialize W&B run (if available)
    if HAS_WANDB:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"{model_name}_rolling_cv",
            config={
                "model": model_name,
                "group": "rolling_cv",
                "features": FEATURE_COLS,
            },
            reinit=True,
        )
    else:
        run = None

    console.print(
        Panel(
            f"[bold cyan]Running rolling CV for model:[/bold cyan] [yellow]{model_name}[/yellow]",
            border_style="cyan",
        )
    )

    # 2. Run rolling CV
    metrics_df, preds_df = run_rolling_cv(
        model=model,
        df_in=df,
        target_col=TARGET_COL,
        date_col=DATE_COL,
        feature_cols=FEATURE_COLS,
        group_col=GROUP_COL,
        save_outputs=True,
        model_name=model_name,
    )

    if metrics_df is None or metrics_df.empty:
        console.print(f"[red]No metrics for {model_name}, skipping.[/red]")
        if run is not None:
            run.finish()
        continue

    all_cv_metrics.append(metrics_df)

    # 3. Aggregate metrics across folds
    summary_row = {
        "Model": model_name,
        "Mean_MAE": metrics_df["MAE"].mean(),
        "Mean_RMSE": metrics_df["RMSE"].mean(),
        "Mean_R2": metrics_df["R²"].mean(),
        "Mean_MAPE": metrics_df["MAPE"].mean(),
    }
    model_summary_rows.append(summary_row)

    # 4. Log to W&B
    if run is not None:
        wandb.log(
            {
                "Mean_MAE": summary_row["Mean_MAE"],
                "Mean_RMSE": summary_row["Mean_RMSE"],
                "Mean_R2": summary_row["Mean_R2"],
                "Mean_MAPE": summary_row["Mean_MAPE"],
            }
        )
        run.finish()

# 5. Build leaderboard
if model_summary_rows:
    model_leaderboard = pd.DataFrame(model_summary_rows)
    model_leaderboard = model_leaderboard.sort_values("Mean_MAE")

    console.print("\n[bold green]Model Leaderboard (lower MAE is better):[/bold green]")
    leaderboard_table = Table(show_header=True, header_style="bold magenta")
    leaderboard_table.add_column("Model", style="cyan")
    leaderboard_table.add_column("Mean MAE", justify="right")
    leaderboard_table.add_column("Mean RMSE", justify="right")
    leaderboard_table.add_column("Mean R²", justify="right")
    leaderboard_table.add_column("Mean MAPE (%)", justify="right")

    for _, row in model_leaderboard.iterrows():
        leaderboard_table.add_row(
            row["Model"],
            f"{row['Mean_MAE']:.4f}",
            f"{row['Mean_RMSE']:.4f}",
            f"{row['Mean_R2']:.4f}",
            f"{row['Mean_MAPE']:.2f}",
        )

    console.print(leaderboard_table)

    LEADERBOARD_PATH = CV_RESULTS_DIR / "model_leaderboard_summary.csv"
    model_leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    console.print(f"[bold green]✓ Saved leaderboard summary → {LEADERBOARD_PATH}[/bold green]")
else:
    console.print("[red]No models produced metrics.[/red]")

log_step("Step 3: Auto-train + leaderboard", None)

# %% [markdown]
# --- Section 4: (Optional) Local CV comparison helper ---

# %%
# Simple helper to reload all *_cv_metrics.csv files under CV_RESULTS_DIR

def load_all_local_cv_metrics(cv_dir: Path = CV_RESULTS_DIR) -> pd.DataFrame:
    """Combine all local *_cv_metrics.csv into a single dataframe."""
    files = list(cv_dir.glob("*_cv_metrics.csv"))
    dfs = []
    for f in files:
        m = pd.read_csv(f)
        # Try to infer model name from filename
        model_name = f.stem.replace("_cv_metrics", "")
        if "Model" not in m.columns:
            m["Model"] = model_name
        dfs.append(m)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


all_local_cv = load_all_local_cv_metrics()
if not all_local_cv.empty:
    console.print(
        Panel(
            f"[bold cyan]Loaded {all_local_cv['Model'].nunique()} models from local CV metrics.[/bold cyan]",
            border_style="cyan",
        )
    )
else:
    console.print("[yellow]No local CV metrics found yet.[/yellow]")

# %% [markdown]
# --- Section 5: Pipeline Summary ---

# %%
show_pipeline_table()
console.print("[bold green]✓ 03_modelt.ipynb setup complete![/bold green]\n"
              "Ready to iterate on best models, add SHAP, or export model cards.")
