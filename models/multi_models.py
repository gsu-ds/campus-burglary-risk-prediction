#!/usr/bin/env python
"""
multi_models.py

Run multiple models on:
  - npu_sparse_panel.parquet
  - npu_dense_panel.parquet
  - target_crimes.parquet (aggregated to NPU × hour)

For each dataset:
  1) Simple train/test split (no rolling CV)
  2) Rolling-origin CV (4 folds)

Outputs:
  reports/test_results/<dataset_name>/
    - simple_results.csv
    - cv_results/<Model>_cv_metrics.csv
    - model_leaderboard_summary.csv
    - <dataset_name>_dataset_card.md
    - <dataset_name>_kaggle_card.md
    - <dataset_name>_model_card.md
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Optional imports: XGBoost / CatBoost / W&B
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

# Pretty console
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    # Fallback minimal console
    class _Dummy:
        def print(self, *args, **kwargs):
            print(*args)
    Console = _Dummy  # type: ignore
    Panel = lambda x, **kwargs: x  # type: ignore
    Table = None  # type: ignore

console = Console()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_SPARSE = PROJECT_ROOT / "data/processed/apd/npu_sparse_panel.parquet"
DATA_DENSE  = PROJECT_ROOT / "data/processed/apd/npu_dense_panel.parquet"
DATA_TARGET = PROJECT_ROOT / "data/processed/apd/target_crimes.parquet"

TEST_RESULTS_ROOT = PROJECT_ROOT / "reports/test_results"
WANDB_PROJECT = "rolling-cv"
WANDB_ENTITY = "joshuadariuspina-georgia-state-university"  # your entity

TRAIN_END = "2024-01-01"

# -----------------------------------------------------------------------------
# MODEL ZOO
# -----------------------------------------------------------------------------

def build_model_zoo() -> Dict[str, object]:
    models: Dict[str, object] = {}

    # Very simple baseline: predicts train-set mean
    class MeanBaseline:
        def fit(self, X, y):
            self.mean_ = float(np.mean(y))
            return self
        def predict(self, X):
            return np.full(shape=(len(X),), fill_value=self.mean_, dtype=float)

    models["BaselineMean"] = MeanBaseline()
    models["LinearRegression"] = LinearRegression()

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    if HAS_XGB:
        models["XGBRegressor"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

    if HAS_CAT:
        models["CatBoostRegressor"] = CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
        )

    return models


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    """
    Create:
      base_dir/
        simple_results.csv
        cv_results/
        cv_results/folds/
        cv_results/predictions/
        model_cards/
    """
    cv_results = base_dir / "cv_results"
    cv_folds = cv_results / "folds"
    cv_preds = cv_results / "predictions"
    model_cards = base_dir / "model_cards"

    for d in [base_dir, cv_results, cv_folds, cv_preds, model_cards]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir,
        "cv_results": cv_results,
        "cv_folds": cv_folds,
        "cv_preds": cv_preds,
        "model_cards": model_cards,
    }


def get_feature_columns(df: pd.DataFrame, target_col: str, date_col: str, group_col: str) -> List[str]:
    """
    Select numeric features only, excluding target, date, grouping, and any obvious leakage columns.
    """
    drop = {target_col, date_col, group_col, "burglary_count", "hour_ts", "report_date"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in drop]
    return feature_cols


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# -----------------------------------------------------------------------------
# ROLLING ORIGIN CROSS-VALIDATION
# -----------------------------------------------------------------------------

def run_rolling_cv(
    model,
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    group_col: str,
    feature_cols: List[str],
    cv_dirs: Dict[str, Path],
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling-origin CV over fixed time folds.
    Returns (metrics_df, predictions_df).
    """

    df_local = df.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])

    folds = [
        ("2022-01-01", "2022-07-01"),
        ("2022-07-01", "2023-01-01"),
        ("2023-01-01", "2023-07-01"),
        ("2023-07-01", "2024-01-01"),
    ]

    metrics_rows: List[Dict] = []
    all_preds: List[pd.DataFrame] = []

    console.print(
        Panel(
            f"[bold cyan]Rolling CV for {model_name}[/bold cyan]",
            border_style="cyan",
        )
    )

    for fold_idx, (train_end, test_end) in enumerate(folds, start=1):
        fold_name = f"Fold {fold_idx}"

        train_mask = df_local[date_col] < train_end
        test_mask = (df_local[date_col] >= train_end) & (df_local[date_col] < test_end)

        train_df = df_local[train_mask].copy()
        test_df = df_local[test_mask].copy()

        if test_df.empty:
            console.print(f"[yellow]{model_name} - {fold_name}: No test data, skipping.[/yellow]")
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        # Save splits
        train_path = cv_dirs["cv_folds"] / f"{model_name}_fold{fold_idx}_train.csv"
        test_path = cv_dirs["cv_folds"] / f"{model_name}_fold{fold_idx}_test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Fit / predict
        if hasattr(model, "fit"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            # Fallback for weird baselines; shouldn't hit for our MeanBaseline
            preds = model.predict(X_test)

        m = compute_metrics(y_test, preds)
        console.print(
            f"[green]{model_name} - {fold_name}[/green] "
            f"MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R²={m['R2']:.4f}, MAPE={m['MAPE']:.2f}%"
        )

        metrics_rows.append(
            {
                "Fold": fold_name,
                "Fold_Number": fold_idx,
                "Train_End": train_end,
                "Test_End": test_end,
                "Train_Size": len(train_df),
                "Test_Size": len(test_df),
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R²": m["R2"],
                "MAPE": m["MAPE"],
            }
        )

        pred_df = test_df[[date_col, group_col, target_col]].copy()
        pred_df["predicted"] = preds
        pred_df["residual"] = y_test.values - preds
        pred_df["abs_error"] = np.abs(pred_df["residual"])
        pred_df["fold"] = fold_name
        pred_df["fold_number"] = fold_idx
        all_preds.append(pred_df)

    if not metrics_rows:
        return pd.DataFrame(), pd.DataFrame()

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(all_preds, ignore_index=True)

    # Save
    metrics_path = cv_dirs["cv_results"] / f"{model_name}_cv_metrics.csv"
    preds_path = cv_dirs["cv_preds"] / f"{model_name}_all_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    return metrics_df, preds_df


# -----------------------------------------------------------------------------
# SIMPLE TRAIN/TEST (NO ROLLING CV)
# -----------------------------------------------------------------------------

def run_simple_split(
    models: Dict[str, object],
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    feature_cols: List[str],
    output_path: Path,
) -> pd.DataFrame:
    """
    Train on [min_date, TRAIN_END), test on [TRAIN_END, max_date].
    """
    df_local = df.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])

    train_df = df_local[df_local[date_col] < TRAIN_END]
    test_df = df_local[df_local[date_col] >= TRAIN_END]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    simple_rows: List[Dict] = []

    console.print(
        Panel(
            f"[bold cyan]Simple train/test split ({TRAIN_END} cutoff)[/bold cyan]",
            border_style="cyan",
        )
    )

    for name, model in models.items():
        console.print(f"[cyan]Training {name} (simple split)...[/cyan]")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        m = compute_metrics(y_test, preds)
        console.print(
            f"[green]{name}[/green] MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, "
            f"R²={m['R2']:.4f}, MAPE={m['MAPE']:.2f}%"
        )

        simple_rows.append(
            {
                "Model": name,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
                "MAPE": m["MAPE"],
            }
        )

    simple_df = pd.DataFrame(simple_rows).sort_values("MAE")
    simple_df.to_csv(output_path, index=False)
    return simple_df


# -----------------------------------------------------------------------------
# DATASET PREPARATION
# -----------------------------------------------------------------------------

def load_sparse_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # assume already has columns: npu, hour_ts, burglary_count, + features
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    return df


def load_dense_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    return df


def load_target_crimes_as_panel(path: Path) -> pd.DataFrame:
    """
    Convert target_crimes.parquet (row-level incidents) into NPU × hour panel
    with burglary_count.
    """
    df = pd.read_parquet(path)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["hour_ts"] = df["report_date"].dt.floor("h")
    df["npu"] = df["npu"].astype(str).str.upper().str.strip()

    panel = (
        df.groupby(["npu", "hour_ts"])
        .size()
        .reset_index(name="burglary_count")
    )
    return panel


# -----------------------------------------------------------------------------
# CARDS (DATASET + MODEL)
# -----------------------------------------------------------------------------

def generate_dataset_card(
    dataset_name: str,
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    group_col: str,
    out_dir: Path,
) -> None:
    """
    Kaggle-style and internal dataset description.
    """
    n_rows, n_cols = df.shape
    date_min = pd.to_datetime(df[date_col]).min()
    date_max = pd.to_datetime(df[date_col]).max()

    is_panel = group_col in df.columns

    card = f"""# 📦 Dataset Card: {dataset_name}

## Overview
This dataset is part of the **Campus Burglary Risk Prediction** project for Atlanta.

- **Rows:** {n_rows:,}
- **Columns:** {n_cols}
- **Date column:** `{date_col}`
- **Date range:** {date_min} → {date_max}
- **Target column:** `{target_col}`
- **Grouping column:** `{group_col}` ({'panel data' if is_panel else 'flat data'})

## Intended Use
- Spatio-temporal modeling of burglary risk
- Evaluation of baseline ML models and tree-based models
- Comparison of rolling-origin CV vs simple train/test

## Basic Schema (first 25 columns)
"""

    schema_preview = (
        df.dtypes.reset_index()
        .rename(columns={"index": "column", 0: "dtype"})
        .head(25)
        .to_markdown(index=False)
    )
    card += f"\n{schema_preview}\n\n"

    card += f"""
## Train/Test Recommendation
- Train on data **before {TRAIN_END}**
- Test on data **from {TRAIN_END} onwards**

## License / Ethics
- Derived from Atlanta Police Department open data and external enrichment.
- For research and educational use only.

"""

    # Save internal card
    card_path = out_dir / f"{dataset_name}_dataset_card.md"
    card_path.write_text(card, encoding="utf-8")

    # Kaggle-facing version (slightly tweaked wording)
    kaggle_card_path = out_dir / f"{dataset_name}_kaggle_card.md"
    kaggle_card_path.write_text(card, encoding="utf-8")


def generate_model_card_for_dataset(
    dataset_name: str,
    cv_leaderboard: pd.DataFrame,
    simple_results: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Single markdown model card summarizing all models for this dataset.
    """
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # align on model name
    simple = simple_results.set_index("Model")
    cv = cv_leaderboard.set_index("Model")

    models = sorted(set(simple.index) | set(cv.index))

    rows = []
    for m in models:
        s = simple.loc[m] if m in simple.index else None
        c = cv.loc[m] if m in cv.index else None

        row = {
            "Model": m,
            "Simple_MAE": float(s["MAE"]) if s is not None else np.nan,
            "CV_Mean_MAE": float(c["Mean_MAE"]) if c is not None else np.nan,
            "CV_Mean_RMSE": float(c["Mean_RMSE"]) if c is not None else np.nan,
            "CV_Mean_R2": float(c["Mean_R2"]) if c is not None else np.nan,
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    table_md = summary_df.to_markdown(index=False)

    card = f"""# 📘 Model Card: {dataset_name}

Generated: **{timestamp}**

## Overview
This card summarizes model performance on the **{dataset_name}** dataset, using:

- **Simple train/test split** (train < {TRAIN_END}, test ≥ {TRAIN_END})
- **Rolling-origin cross-validation** (4 folds from 2022–2023)

## Leaderboard (Simple vs Rolling CV)

{table_md}

- **Simple_MAE:** MAE on holdout 2024 period
- **CV_Mean_MAE:** Mean MAE across rolling CV folds
- **CV_Mean_RMSE:** Mean RMSE across rolling CV folds
- **CV_Mean_R2:** Mean R² across rolling CV folds

## Interpretation
- Prefer models with **low CV_Mean_MAE** and **high CV_Mean_R2**.
- Compare **Simple_MAE** to **CV_Mean_MAE** to see if the model generalizes
  to the final holdout period (stability gap).

"""

    card_path = out_dir / f"{dataset_name}_model_card.md"
    card_path.write_text(card, encoding="utf-8")


# -----------------------------------------------------------------------------
# MAIN ROUTINE PER DATASET
# -----------------------------------------------------------------------------

def run_for_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    group_col: str,
) -> None:
    console.print(
        Panel(
            f"[bold magenta]=== DATASET: {dataset_name} ===[/bold magenta]",
            border_style="magenta",
        )
    )

    out_dirs = ensure_output_dirs(TEST_RESULTS_ROOT / dataset_name)
    models = build_model_zoo()

    # Feature selection
    feature_cols = get_feature_columns(df, target_col, date_col, group_col)
    console.print(f"[green]{dataset_name}[/green] feature columns ({len(feature_cols)}): {feature_cols}")

    # --- 1) Simple train/test (no rolling CV) ---
    simple_results_path = out_dirs["base"] / "simple_results.csv"
    simple_results = run_simple_split(
        models=models,
        df=df,
        target_col=target_col,
        date_col=date_col,
        feature_cols=feature_cols,
        output_path=simple_results_path,
    )

    # --- 2) Rolling-origin CV ---
    cv_summary_rows: List[Dict] = []

    for model_name, model in models.items():
        metrics_df, preds_df = run_rolling_cv(
            model=model,
            df=df,
            target_col=target_col,
            date_col=date_col,
            group_col=group_col,
            feature_cols=feature_cols,
            cv_dirs=out_dirs,
            model_name=model_name,
        )

        if metrics_df.empty:
            continue

        # aggregate to single row for leaderboard
        row = {
            "Model": model_name,
            "Mean_MAE": metrics_df["MAE"].mean(),
            "Mean_RMSE": metrics_df["RMSE"].mean(),
            "Mean_R2": metrics_df["R²"].mean(),
            "Mean_MAPE": metrics_df["MAPE"].mean(),
        }
        cv_summary_rows.append(row)

        # optional W&B
        if HAS_WANDB:
            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{dataset_name}_{model_name}_rolling_cv",
                config={"dataset": dataset_name, "model": model_name},
                reinit=True,
            )
            wandb.log(row)
            run.finish()

    if not cv_summary_rows:
        console.print(f"[red]No CV results for {dataset_name}[/red]")
        return

    cv_leaderboard = pd.DataFrame(cv_summary_rows).sort_values("Mean_MAE")

    # save leaderboard
    leaderboard_path = out_dirs["base"] / "model_leaderboard_summary.csv"
    cv_leaderboard.to_csv(leaderboard_path, index=False)

    # --- Cards ---
    generate_dataset_card(
        dataset_name=dataset_name,
        df=df,
        target_col=target_col,
        date_col=date_col,
        group_col=group_col,
        out_dir=out_dirs["base"],
    )

    generate_model_card_for_dataset(
        dataset_name=dataset_name,
        cv_leaderboard=cv_leaderboard,
        simple_results=simple_results,
        out_dir=out_dirs["base"],
    )

    console.print(
        Panel(
            f"[bold green]Finished dataset {dataset_name}. "
            f"Results in: {out_dirs['base']}[/bold green]",
            border_style="green",
        )
    )


# -----------------------------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------------------------

def main() -> None:
    console.print(
        Panel(
            "[bold cyan]Multi-dataset modeling: sparse, dense, target[/bold cyan]",
            border_style="cyan",
        )
    )

    # Sparse panel
    if DATA_SPARSE.exists():
        df_sparse = load_sparse_panel(DATA_SPARSE)
        run_for_dataset(
            dataset_name="npu_sparse_panel",
            df=df_sparse,
            target_col="burglary_count",
            date_col="hour_ts",
            group_col="npu",
        )
    else:
        console.print(f"[red]Missing: {DATA_SPARSE}[/red]")

    # Dense panel
    if DATA_DENSE.exists():
        df_dense = load_dense_panel(DATA_DENSE)
        run_for_dataset(
            dataset_name="npu_dense_panel",
            df=df_dense,
            target_col="burglary_count",
            date_col="hour_ts",
            group_col="npu",
        )
    else:
        console.print(f"[red]Missing: {DATA_DENSE}[/red]")

    # Target crimes → panel
    if DATA_TARGET.exists():
        df_target_panel = load_target_crimes_as_panel(DATA_TARGET)
        run_for_dataset(
            dataset_name="target_crimes_panel",
            df=df_target_panel,
            target_col="burglary_count",
            date_col="hour_ts",
            group_col="npu",
        )
    else:
        console.print(f"[red]Missing: {DATA_TARGET}[/red]")

# -----------------------------------------------------------------------------
# GLOBAL COMBINED LEADERBOARD
# -----------------------------------------------------------------------------
def build_combined_leaderboard():
    console.print(
        Panel(
            "[bold magenta]Building Combined Leaderboard (All Datasets)[/bold magenta]",
            border_style="magenta",
        )
    )

    combined_rows = []

    for dataset_folder in ["npu_sparse_panel", "npu_dense_panel", "target_crimes_panel"]:
        base_dir = TEST_RESULTS_ROOT / dataset_folder
        cv_path = base_dir / "model_leaderboard_summary.csv"
        simple_path = base_dir / "simple_results.csv"

        if not (cv_path.exists() and simple_path.exists()):
            console.print(f"[yellow]Skipping {dataset_folder} (missing files)[/yellow]")
            continue

        cv_df = pd.read_csv(cv_path)
        simple_df = pd.read_csv(simple_path)

        cv_df = cv_df.set_index("Model")
        simple_df = simple_df.set_index("Model")

        all_models = sorted(set(cv_df.index) | set(simple_df.index))

        for model in all_models:
            cv_row = cv_df.loc[model] if model in cv_df.index else None
            simple_row = simple_df.loc[model] if model in simple_df.index else None

            combined_rows.append({
                "Dataset": dataset_folder,
                "Model": model,
                "CV_Mean_MAE": cv_row["Mean_MAE"] if cv_row is not None else np.nan,
                "CV_Mean_RMSE": cv_row["Mean_RMSE"] if cv_row is not None else np.nan,
                "CV_Mean_R2": cv_row["Mean_R2"] if cv_row is not None else np.nan,
                "Simple_MAE": simple_row["MAE"] if simple_row is not None else np.nan,
                "Simple_RMSE": simple_row["RMSE"] if simple_row is not None else np.nan,
                "Simple_R2": simple_row["R2"] if simple_row is not None else np.nan,
                "Stability_Gap": (
                    (simple_row["MAE"] - cv_row["Mean_MAE"])
                    if (cv_row is not None and simple_row is not None)
                    else np.nan
                ),
            })

    combined_df = pd.DataFrame(combined_rows)

    if combined_df.empty:
        console.print("[red]No combined leaderboard could be built.[/red]")
        return

    # Rank models globally
    combined_df["Global_Rank"] = combined_df["CV_Mean_MAE"].rank()

    # Save CSV
    combined_path = TEST_RESULTS_ROOT / "combined_leaderboard.csv"
    combined_df.to_csv(combined_path, index=False)

    # Save Markdown
    md_path = TEST_RESULTS_ROOT / "combined_leaderboard.md"
    md_path.write_text(combined_df.to_markdown(index=False), encoding="utf-8")

    # Print table
    table = Table(title="Combined Leaderboard (All Datasets)", show_header=True)
    for col in ["Dataset", "Model", "CV_Mean_MAE", "Simple_MAE", "Stability_Gap", "Global_Rank"]:
        table.add_column(col, style="cyan")

    for _, row in combined_df.sort_values("Global_Rank").iterrows():
        table.add_row(
            row["Dataset"],
            row["Model"],
            f"{row['CV_Mean_MAE']:.4f}" if pd.notna(row["CV_Mean_MAE"]) else "-",
            f"{row['Simple_MAE']:.4f}" if pd.notna(row["Simple_MAE"]) else "-",
            f"{row['Stability_Gap']:.4f}" if pd.notna(row["Stability_Gap"]) else "-",
            f"{int(row['Global_Rank'])}"
        )

    console.print(table)

    # Scatterplot
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=combined_df,
        x="CV_Mean_MAE",
        y="Simple_MAE",
        hue="Model",
        style="Dataset",
        s=120,
    )
    plt.title("Stability: CV Mean MAE vs Simple MAE")
    plt.xlabel("Rolling CV Mean MAE")
    plt.ylabel("Simple Split MAE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = TEST_RESULTS_ROOT / "combined_leaderboard_scatter.png"
    plt.savefig(plot_path, dpi=200)

    console.print(f"[bold green]✓ Combined leaderboard saved[/bold green]: {combined_path}")
    console.print(f"[bold green]✓ Scatterplot saved[/bold green]: {plot_path}")


# ---------------------------------------------------------------------------
# CALL GLOBAL LEADERBOARD AFTER MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    build_combined_leaderboard()
