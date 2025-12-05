#!/usr/bin/env python
# To quick run: python -m atl_model_pipelines.models.rolling_cv
# Check atl_model_pipelines/README.md for quick guide.

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config import (
    PROCESSED_DIR,
    LEADERBOARD_DIR,
    TEST_RESULTS_DIR,
    WANDB_OUTPUT_DIR,
    CARDS_DIR,
    DATA_SPARSE,
    DATA_DENSE,
    DATA_TARGET,
    DATA_TARGET_PANEL,
    CATBOOST_OUTPUT_DIR,
    ARTIFACTS_DIR,
)

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

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except Exception:
    class _Dummy:
        def print(self, *args, **kwargs):
            print(*args)

    Console = _Dummy  # type: ignore
    Panel = lambda x, **kwargs: x  # type: ignore
    Table = None  # type: ignore

console = Console()

WANDB_PROJECT = "Data Science Capstone - Final Tests"
WANDB_ENTITY = "joshuadariuspina-georgia-state-university"
TRAIN_END = "2024-01-01"

TARGET_MAP = {
    "target_crimes_panel": "burglary_count",
    "npu_sparse_panel": "crime_count",
    "npu_dense_panel": "crime_count",
}


def build_model_zoo() -> Dict[str, object]:
    models: Dict[str, object] = {}

    class MeanBaseline:
        def fit(self, X, y):
            self.mean_ = float(np.mean(y))

        def predict(self, X):
            return np.full(len(X), self.mean_)

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
            train_dir=CATBOOST_OUTPUT_DIR,
        )

    return models


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
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


def get_feature_columns(
    df: pd.DataFrame, target_col: str, date_col: str, group_col: str
) -> List[str]:
    drop = {target_col, date_col, group_col, "burglary_count", "hour_ts", "report_date"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in drop]


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def run_rolling_cv(
    model,
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    group_col: str,
    feature_cols: List[str],
    cv_dirs: Dict[str, Path],
    model_name: str,
    batch_size: int | None = None,
):
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

    console.print(Panel(f"[bold cyan]Rolling CV for {model_name}[/bold cyan]"))

    for idx, (train_end, test_end) in enumerate(folds, start=1):
        f_name = f"Fold {idx}"

        train_df = df_local[df_local[date_col] < train_end]
        test_df = df_local[
            (df_local[date_col] >= train_end)
            & (df_local[date_col] < test_end)
        ]

        if test_df.empty:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col]

        model.fit(X_train, y_train)

        if batch_size is not None and len(X_test) > batch_size:
            preds_list = []
            for start in range(0, len(X_test), batch_size):
                end = start + batch_size
                preds_list.append(model.predict(X_test.iloc[start:end]))
            preds = np.concatenate(preds_list)
        else:
            preds = model.predict(X_test)

        m = compute_metrics(y_test, preds)
        console.print(
            f"[green]{model_name} - {f_name}[/green] "
            f"MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R²={m['R2']:.4f}"
        )

        if HAS_WANDB and wandb.run is not None:
            wandb.log(
                {
                    "split": "cv",
                    "model": model_name,
                    "fold": f_name,
                    "cv/MAE": m["MAE"],
                    "cv/RMSE": m["RMSE"],
                    "cv/R2": m["R2"],
                    "cv/MAPE": m["MAPE"],
                }
            )

        metrics_rows.append(
            {
                "Fold": f_name,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R²": m["R2"],
                "MAPE": m["MAPE"],
            }
        )

        pred_df = test_df[[date_col, group_col, target_col]].copy()
        pred_df["predicted"] = preds
        pred_df["fold"] = f_name
        all_preds.append(pred_df)

    if not metrics_rows:
        return pd.DataFrame(), pd.DataFrame()

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(all_preds, ignore_index=True)

    metrics_df.to_csv(
        cv_dirs["cv_results"] / f"{model_name}_cv_metrics.csv", index=False
    )
    preds_df.to_csv(
        cv_dirs["cv_preds"] / f"{model_name}_all_predictions.csv", index=False
    )

    return metrics_df, preds_df


def run_simple_split(
    models: Dict[str, object],
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    feature_cols: List[str],
    out_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df_local = df.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])

    train_df = df_local[df_local[date_col] < TRAIN_END]
    test_df = df_local[df_local[date_col] >= TRAIN_END]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col]

    rows: List[Dict] = []
    fitted_models: Dict[str, object] = {}

    console.print(
        Panel(
            f"[bold cyan]Simple train/test split ({TRAIN_END} cutoff)[/bold cyan]"
        )
    )

    for name, model in models.items():
        console.print(f"[cyan]Training {name}...[/cyan]")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = compute_metrics(y_test, preds)

        console.print(
            f"[green]{name}[/green] "
            f"MAE={m['MAE']:.4f}, RMSE={m['RMSE']:.4f}, R²={m['R2']:.4f}"
        )

        if HAS_WANDB and wandb.run is not None:
            wandb.log(
                {
                    "split": "simple",
                    "model": name,
                    "simple/MAE": m["MAE"],
                    "simple/RMSE": m["RMSE"],
                    "simple/R2": m["R2"],
                    "simple/MAPE": m["MAPE"],
                }
            )

        rows.append(
            {
                "Model": name,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
                "MAPE": m["MAPE"],
            }
        )

        fitted_models[name] = model

    simple_df = pd.DataFrame(rows).sort_values("MAE")
    simple_df.to_csv(out_path, index=False)
    return simple_df, fitted_models


def load_sparse_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    return df


def load_dense_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    return df


def load_target_crimes_as_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    df["report_date"] = pd.to_datetime(df["report_date"])
    df["hour_ts"] = df["report_date"].dt.floor("h")
    df["npu"] = df["npu"].astype(str).str.upper().str.strip()

    hourly_counts = (
        df.groupby(["npu", "hour_ts"])
          .size()
          .reset_index(name="burglary_count")
    )

    numeric_cols = [
        "location_type_count",
        "incident_hour",
        "year",
        "month",
        "hour_sin",
        "hour_cos",
        "temp_f",
        "precip_in",
        "rain_in",
        "apparent_temp_f",
        "daylight_duration_sec",
        "sunshine_duration_sec",
        "precip_hours",
        "rain_sum_in",
        "temp_mean_f",
        "grid_density_7d",
        "npu_crime_avg_30d",
        "campus_distance_m",
    ]

    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if numeric_cols:
        hourly_numeric = (
            df.groupby(["npu", "hour_ts"])[numeric_cols]
              .mean()
              .reset_index()
        )
    else:
        hourly_numeric = hourly_counts[["npu", "hour_ts"]].copy()

    cat_cols = [
        "day_number",
        "day_of_week",
        "hour_block",
        "is_holiday",
        "is_weekend",
        "is_daylight",
        "weather_code_hourly",
        "weather_code_daily",
        "offense_category",
        "campus_label",
        "campus_code",
        "event_watch_day_watch",
        "event_watch_evening_watch",
        "event_watch_morning_watch",
        "near_gsu",
        "near_ga_tech",
        "near_emory",
        "near_clark",
        "near_spelman",
        "near_morehouse",
        "near_morehouse_med",
        "near_atlanta_metro",
        "near_atlanta_tech",
        "near_scad",
        "near_john_marshall",
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]

    if cat_cols:
        hourly_cat = (
            df.groupby(["npu", "hour_ts"])[cat_cols]
              .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
              .reset_index()
        )
    else:
        hourly_cat = hourly_counts[["npu", "hour_ts"]].copy()

    panel = (
        hourly_counts
        .merge(hourly_numeric, on=["npu", "hour_ts"], how="left")
        .merge(hourly_cat, on=["npu", "hour_ts"], how="left")
        .sort_values(["npu", "hour_ts"])
        .reset_index(drop=True)
    )

    lags = [1, 3, 6, 12, 24, 168]
    for lag in lags:
        panel[f"lag_{lag}"] = panel.groupby("npu")["burglary_count"].shift(lag)

    panel = panel.dropna(subset=[f"lag_{l}" for l in lags])
    return panel


def generate_dataset_card(
    dataset_name: str,
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    group_col: str,
    out_dir: Path,
) -> None:
    n_rows, n_cols = df.shape
    date_vals = pd.to_datetime(df[date_col])
    date_min, date_max = date_vals.min(), date_vals.max()

    card = f"""# Dataset Card: {dataset_name}

Rows: {n_rows:,}
Cols: {n_cols}
Target: {target_col}
Date column: {date_col}
Group column: {group_col}
Date Range: {date_min} → {date_max}

First 25 dtypes:
{df.dtypes.head(25).to_markdown()}
"""

    (CARDS_DIR / f"{dataset_name}_dataset_card.md").write_text(
        card, encoding="utf-8"
    )


def generate_model_card_for_dataset(
    dataset_name: str,
    cv_leaderboard: pd.DataFrame,
    simple_results: pd.DataFrame,
    out_dir: Path,
) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    simple = simple_results.set_index("Model")
    cv = cv_leaderboard.set_index("Model")

    models = sorted(set(simple.index) | set(cv.index))
    rows: List[Dict] = []

    for m in models:
        s = simple.loc[m] if m in simple.index else None
        c = cv.loc[m] if m in cv.index else None
        rows.append(
            {
                "Model": m,
                "Simple_MAE": float(s["MAE"]) if s is not None else np.nan,
                "Simple_R2": float(s["R2"]) if s is not None else np.nan,
                "CV_Mean_MAE": float(c["Mean_MAE"]) if c is not None else np.nan,
                "CV_Mean_R2": float(c["Mean_R2"]) if c is not None else np.nan,
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df["CV_Mean_R2"].isna().all():
        best_model = (
            summary_df.sort_values("CV_Mean_R2", ascending=False)
            .iloc[0]["Model"]
        )
    else:
        best_model = summary_df.sort_values("CV_Mean_MAE").iloc[0]["Model"]

    card = f"""# Model Card: {dataset_name}

Generated: {timestamp}

## Summary Table

{summary_df.to_markdown(index=False)}

**Best model (by CV Mean R²):** **{best_model}**
"""

    (CARDS_DIR / f"{dataset_name}_model_card.md").write_text(
        card, encoding="utf-8"
    )


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

    wandb_run = None
    if HAS_WANDB:
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"{dataset_name}_rolling_cv",
            group=dataset_name,
            dir=WANDB_OUTPUT_DIR,
            config={
                "dataset": dataset_name,
                "target_col": target_col,
                "date_col": date_col,
                "group_col": group_col,
                "train_end": TRAIN_END,
            },
        )

    out_dirs = ensure_output_dirs(TEST_RESULTS_DIR / dataset_name)
    models = build_model_zoo()

    feature_cols = get_feature_columns(df, target_col, date_col, group_col)
    console.print(
        f"{dataset_name} features ({len(feature_cols)}): {feature_cols}"
    )

    simple_path = out_dirs["base"] / "simple_results.csv"
    simple_results, fitted_models = run_simple_split(
        models=models,
        df=df,
        target_col=target_col,
        date_col=date_col,
        feature_cols=feature_cols,
        out_path=simple_path,
    )

    cv_rows: List[Dict] = []
    is_dense = dataset_name == "npu_dense_panel"
    batch_size = 100_000 if is_dense else None

    for model_name, model in models.items():
        m_df, p_df = run_rolling_cv(
            model=model,
            df=df,
            target_col=target_col,
            date_col=date_col,
            group_col=group_col,
            feature_cols=feature_cols,
            cv_dirs=out_dirs,
            model_name=model_name,
            batch_size=batch_size,
        )

        if m_df.empty:
            continue

        cv_rows.append(
            {
                "Model": model_name,
                "Mean_MAE": m_df["MAE"].mean(),
                "Mean_RMSE": m_df["RMSE"].mean(),
                "Mean_R2": m_df["R²"].mean(),
                "Mean_MAPE": m_df["MAPE"].mean(),
            }
        )

    if not cv_rows:
        console.print("[red]No CV results.[/red]")
        if wandb_run is not None:
            wandb_run.finish()
        return

    cv_leaderboard = pd.DataFrame(cv_rows).sort_values("Mean_MAE")
    cv_leaderboard_path = out_dirs["base"] / "model_leaderboard_summary.csv"
    cv_leaderboard.to_csv(cv_leaderboard_path, index=False)

    if HAS_WANDB and wandb_run is not None:
        wandb.log(
            {
                "cv_leaderboard": wandb.Table(
                    dataframe=cv_leaderboard.reset_index(drop=True)
                )
            }
        )

    if not cv_leaderboard["Mean_R2"].isna().all():
        best_row = cv_leaderboard.sort_values(
            "Mean_R2", ascending=False
        ).iloc[0]
    else:
        best_row = cv_leaderboard.sort_values("Mean_MAE").iloc[0]

    best_model_name = best_row["Model"]
    console.print(
        f"[bold green]Best model for {dataset_name} (by CV Mean R²): "
        f"{best_model_name}[/bold green]"
    )

    df_local = df.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])

    train_df_full = df_local[df_local[date_col] < TRAIN_END]
    X_train_full = train_df_full[feature_cols].fillna(0)
    y_train_full = train_df_full[target_col]

    best_model_obj = build_model_zoo()[best_model_name]
    best_model_obj.fit(X_train_full, y_train_full)

    dataset_art_dir = ARTIFACTS_DIR / dataset_name
    dataset_art_dir.mkdir(parents=True, exist_ok=True)

    model_path = dataset_art_dir / f"{dataset_name}_{best_model_name}_best_model.joblib"
    joblib.dump(best_model_obj, model_path)

    metadata = {
        "dataset": dataset_name,
        "best_model": best_model_name,
        "selection_metric": "CV Mean R2",
        "CV_Mean_R2": float(best_row["Mean_R2"]),
        "CV_Mean_MAE": float(best_row["Mean_MAE"]),
        "CV_Mean_RMSE": float(best_row["Mean_RMSE"]),
        "CV_Mean_MAPE": float(best_row["Mean_MAPE"]),
        "TRAIN_END": TRAIN_END,
        "target_col": target_col,
        "feature_cols": list(feature_cols),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
    }

    metadata_path = dataset_art_dir / f"{dataset_name}_{best_model_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    console.print(
        f"[green]Saved best model artifact[/green]: {model_path.name}"
    )

    if HAS_WANDB and wandb_run is not None:
        model_artifact = wandb.Artifact(
            name=f"{dataset_name}-{best_model_name}-best-model",
            type="model",
            description=(
                f"Best {dataset_name} model selected by CV Mean R² "
                f"(train < {TRAIN_END})."
            ),
            metadata=metadata,
        )
        model_artifact.add_file(model_path)
        model_artifact.add_file(metadata_path)

        wandb_run.log_artifact(model_artifact)

        wandb_run.summary["best_model"] = best_model_name
        wandb_run.summary["best_CV_Mean_R2"] = metadata["CV_Mean_R2"]
        wandb_run.summary["best_CV_Mean_MAE"] = metadata["CV_Mean_MAE"]
        wandb_run.summary["best_CV_Mean_RMSE"] = metadata["CV_Mean_RMSE"]

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
            f"[green]Finished dataset {dataset_name}[/green]",
            border_style="green",
        )
    )

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    console.print(Panel("[bold cyan]Multi-dataset modeling: sparse, target, dense[/bold cyan]"))

    ds = "npu_sparse_panel"
    if DATA_SPARSE.exists():
        df_sparse = load_sparse_panel(DATA_SPARSE)
        run_for_dataset(
            dataset_name=ds,
            df=df_sparse,
            target_col=TARGET_MAP[ds],
            date_col="hour_ts",
            group_col="npu",
        )
    else:
        console.print(f"[red]Missing: {DATA_SPARSE}[/red]")

    ds = "target_crimes_panel"
    if DATA_TARGET_PANEL.exists():
        df_target_panel = load_sparse_panel(DATA_TARGET_PANEL)
        run_for_dataset(
            dataset_name=ds,
            df=df_target_panel,
            target_col=TARGET_MAP[ds],
            date_col="hour_ts",
            group_col="npu",
        )
    else:
        console.print(f"[red]Missing: {DATA_TARGET_PANEL}[/red]")

    ds = "npu_dense_panel"
    if DATA_DENSE.exists():
        df_dense = load_dense_panel(DATA_DENSE)
        MAX_DENSE_ROWS = 300_000
        n_dense = len(df_dense)
        if n_dense > MAX_DENSE_ROWS:
            console.print(
                f"[yellow]Downsampling dense panel from {n_dense:,} → "
                f"{MAX_DENSE_ROWS:,} rows for modeling (OOM guard).[/yellow]"
            )
            df_dense = df_dense.sample(
                n=MAX_DENSE_ROWS, random_state=42
            ).sort_values("hour_ts")

        run_for_dataset(
            dataset_name=ds,
            df=df_dense,
            target_col=TARGET_MAP[ds],
            date_col="hour_ts",
            group_col="npu",
        )
    else:
        console.print(f"[red]Missing: {DATA_DENSE}[/red]")


def build_combined_leaderboard() -> None:
    console.print(
        Panel(
            "[bold magenta]Building Combined Leaderboard[/bold magenta]",
            border_style="magenta",
        )
    )

    rows: List[Dict] = []
    for folder in ["npu_sparse_panel", "npu_dense_panel", "target_crimes_panel"]:
        base = TEST_RESULTS_DIR / folder
        cv_path = base / "model_leaderboard_summary.csv"
        simple_path = base / "simple_results.csv"

        if not (cv_path.exists() and simple_path.exists()):
            continue

        cv_df = pd.read_csv(cv_path).set_index("Model")
        s_df = pd.read_csv(simple_path).set_index("Model")

        for m in sorted(set(cv_df.index) | set(s_df.index)):
            cv_row = cv_df.loc[m] if m in cv_df.index else None
            s_row = s_df.loc[m] if m in s_df.index else None

            rows.append(
                {
                    "Dataset": folder,
                    "Model": m,
                    "CV_Mean_MAE": cv_row["Mean_MAE"]
                    if cv_row is not None
                    else np.nan,
                    "CV_Mean_R2": cv_row["Mean_R2"]
                    if cv_row is not None
                    else np.nan,
                    "Simple_MAE": s_row["MAE"]
                    if s_row is not None
                    else np.nan,
                    "Simple_R2": s_row["R2"]
                    if s_row is not None
                    else np.nan,
                }
            )

    if not rows:
        console.print("[red]No combined results.[/red]")
        return

    df = pd.DataFrame(rows)
    combined_path = TEST_RESULTS_DIR / "combined_leaderboard.csv"
    df.to_csv(combined_path, index=False)
    console.print(f"[green]Saved combined leaderboard![/green] → {combined_path}")


if __name__ == "__main__":
    main()
    build_combined_leaderboard()