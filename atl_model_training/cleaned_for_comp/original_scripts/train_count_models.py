import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from statsmodels.discrete.count_model import ZeroInflatedPoisson
import statsmodels.api as sm

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TIME_BLOCKS = [
    (0, 4, "Late Night (0-4)"),
    (5, 8, "Early Morning (5-8)"),
    (9, 12, "Late Morning (9-12)"),
    (13, 16, "Afternoon (13-16)"),
    (17, 20, "Evening (17-20)"),
    (21, 23, "Late Night (21-24)"),
]

LAGS = [1, 3, 6, 12, 24, 168]


def assign_time_block(hour: int) -> str:
    for start, end, label in TIME_BLOCKS:
        if start <= hour <= end:
            return label
    return TIME_BLOCKS[-1][2]


def ensure_output_dirs(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "predictions").mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values(["npu", "hour_ts"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["hour_ts"].dt.hour
    df["time_block_label"] = df["hour"].apply(assign_time_block)
    df["time_block_code"] = df["time_block_label"].astype("category").cat.codes
    df["npu_code"] = df["npu"].astype("category").cat.codes

    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    df["day_of_week_code"] = (
        df["day_of_week"].map(day_map).fillna(-1).astype(int)
    )

    for lag in LAGS:
        df[f"lag_{lag}"] = (
            df.groupby("npu", observed=True)["burglary_count"].shift(lag)
        )

    lag_cols = [f"lag_{lag}" for lag in LAGS]
    df = df.dropna(subset=lag_cols)

    numeric_na = [
        "grid_density_7d",
        "npu_crime_avg_30d",
        "campus_distance_m",
        "location_type_count",
    ]
    for col in numeric_na:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


def train_test_split(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_ts = pd.Timestamp(cutoff)
    train_df = df[df["hour_ts"] < cutoff_ts].copy()
    test_df = df[df["hour_ts"] >= cutoff_ts].copy()
    return train_df, test_df


def evaluate_predictions(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def add_baselines(test_df: pd.DataFrame, output_dir: Path) -> dict:
    results = {}
    for name, col in [("naive_24h", "lag_24"), ("naive_168h", "lag_168")]:
        target_col = test_df["burglary_count"]
        preds = test_df[col]
        valid_mask = preds.notna()
        metrics = evaluate_predictions(target_col[valid_mask], preds[valid_mask])
        results[name] = metrics
        (
            test_df.loc[valid_mask, ["npu", "hour_ts", "burglary_count", col]]
            .rename(columns={col: "prediction"})
            .to_csv(output_dir / f"{name}_predictions.csv", index=False)
        )
    return results


def prepare_matrices(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "burglary_count"
) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    return X, y


def train_xgboost(train_df, test_df, feature_cols):
    X_train, y_train = prepare_matrices(train_df, feature_cols)
    X_test, _ = prepare_matrices(test_df, feature_cols)
    model = XGBRegressor(
        objective="count:poisson",
        tree_method="hist",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=800,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)
    return model, preds


def train_lightgbm(train_df, test_df, feature_cols):
    X_train, y_train = prepare_matrices(train_df, feature_cols)
    X_test, _ = prepare_matrices(test_df, feature_cols)
    model = LGBMRegressor(
        objective="poisson",
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)
    return model, preds


def train_catboost(train_df, test_df, numeric_cols, cat_cols):
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost is not installed in this environment.")
    feature_cols = numeric_cols + cat_cols
    X_train = train_df[feature_cols]
    y_train = train_df["burglary_count"].values
    X_test = test_df[feature_cols]

    cat_indices = [feature_cols.index(col) for col in cat_cols]
    model = CatBoostRegressor(
        loss_function="Poisson",
        depth=6,
        learning_rate=0.05,
        iterations=800,
        subsample=0.8,
        verbose=False,
    )
    model.fit(X_train, y_train, cat_features=cat_indices)
    preds = np.clip(model.predict(X_test), 0, None)
    return model, preds


def train_zip(train_df, test_df, feature_cols, sample_size=150000):
    train_used = (
        train_df.sample(sample_size, random_state=42)
        if len(train_df) > sample_size
        else train_df
    )

    X_train = sm.add_constant(train_used[feature_cols])
    y_train = train_used["burglary_count"].values
    X_test = sm.add_constant(test_df[feature_cols], has_constant="add")

    zip_model = ZeroInflatedPoisson(
        endog=y_train,
        exog=X_train,
        exog_infl=X_train[["const", feature_cols[0]]],
        inflation="logit",
    )
    fit_res = zip_model.fit(method="bfgs", maxiter=100, disp=False)
    preds = np.clip(fit_res.predict(exog=X_test, exog_infl=X_test[["const", feature_cols[0]]]), 0, None)
    return fit_res, preds


def train_prophet(df, cutoff):
    if not HAS_PROPHET:
        raise RuntimeError("Prophet is not installed in this environment.")
    agg = (
        df.groupby("hour_ts", as_index=False)["burglary_count"]
        .sum()
        .rename(columns={"hour_ts": "ds", "burglary_count": "y"})
    )
    train = agg[agg["ds"] < cutoff]
    test = agg[agg["ds"] >= cutoff]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        interval_width=0.90,
    )
    model.fit(train)
    future = model.make_future_dataframe(
        periods=len(test),
        freq="H",
        include_history=True,
    )
    forecast = model.predict(future)
    preds = forecast.tail(len(test))["yhat"].to_numpy()
    preds = np.clip(preds, 0, None)
    metrics = evaluate_predictions(test["y"].values, preds)
    return model, test, preds, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Poisson-based models on the enriched NPU dense panel."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/apd/npu_dense_panel.parquet"),
        help="Path to the dense panel parquet file.",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2024-01-01 00:00:00",
        help="Timestamp to split train/test sets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/test_results/current_modeling"),
        help="Directory where metrics and predictions will be saved.",
    )
    # Kept for compatibility but overridden below for hardcoding
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="campus-burglary-risk",
        help="Weights & Biases project name (default: campus-burglary-risk).",
    )
    # Kept for compatibility but overridden below for hardcoding
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="dense-panel-poisson",
        help="Custom run name for wandb (default: dense-panel-poisson).",
    )
    args = parser.parse_args()

    ensure_output_dirs(args.output)

    # --- WANDB HARDCODED CONFIGURATION ---
    # Hardcode the project name to ensure all runs go to the same location
    WANDB_PROJECT_NAME = "Data Science Capstone - Final Tests"
    WANDB_RUN_NAME = "Poisson_Count_Model_Comparison"

    wandb_run = None

    if HAS_WANDB:
        print(f"Initializing wandb run: Project={WANDB_PROJECT_NAME}, Name={WANDB_RUN_NAME}")
        wandb_run = wandb.init(
            project="Data Science Capstone - Final Tests",
            entity="joshuadariuspina-georgia-state-university",
            group="poisson_models",
            name="Poisson_Count_Model_Comparison",
            tags=["crime-forecasting", "capstone", "npu", "time-series"]
            config={
                "dataset": str(args.data),
                "cutoff": args.cutoff,
                "models": [
                    "xgboost_poisson",
                    "lightgbm_poisson",
                    "catboost_poisson" if HAS_CATBOOST else "catboost_skipped",
                    "zip",
                    "prophet_hourly_total" if HAS_PROPHET else "prophet_skipped",
                ],
                "baselines": ["naive_24h", "naive_168h"],
                
            },
        )
    else:
        print("wandb package not installed; skipping wandb logging.")

    print("Loading dataset...")
    df = load_dataset(args.data)
    df = add_features(df)

    train_df, test_df = train_test_split(df, args.cutoff)
    if train_df.empty or test_df.empty:
        raise RuntimeError("Train or test split is empty. Check cutoff and data range.")

    baseline_dir = args.output / "predictions" / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating baselines...")
    baseline_metrics = add_baselines(test_df, baseline_dir)

    feature_cols = [
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
        "hour_sin",
        "hour_cos",
        "day_of_week_code",
        "campus_distance_m",
        "location_type_count",
        "lag_1",
        "lag_3",
        "lag_6",
        "lag_12",
        "lag_24",
        "lag_168",
        "npu_code",
        "time_block_code",
    ]

    metrics = {}
    predictions_dir = args.output / "predictions"

    print("Training XGBoost (Poisson)...")
    xgb_model, xgb_preds = train_xgboost(train_df, test_df, feature_cols)
    metrics["xgboost_poisson"] = evaluate_predictions(
        test_df["burglary_count"], xgb_preds
    )
    test_df.assign(prediction=xgb_preds).to_csv(
        predictions_dir / "xgboost_poisson.csv", index=False
    )

    print("Training LightGBM (Poisson)...")
    lgb_model, lgb_preds = train_lightgbm(train_df, test_df, feature_cols)
    metrics["lightgbm_poisson"] = evaluate_predictions(
        test_df["burglary_count"], lgb_preds
    )
    test_df.assign(prediction=lgb_preds).to_csv(
        predictions_dir / "lightgbm_poisson.csv", index=False
    )

    numeric_for_cat = [
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
        "hour_sin",
        "hour_cos",
        "day_of_week_code",
        "campus_distance_m",
        "location_type_count",
        "lag_1",
        "lag_3",
        "lag_6",
        "lag_12",
        "lag_24",
        "lag_168",
    ]
    cat_cols = ["npu", "time_block_label"]

    if HAS_CATBOOST:
        print("Training CatBoost (Poisson)...")
        cat_model, cat_preds = train_catboost(
            train_df, test_df, numeric_for_cat, cat_cols
        )
        metrics["catboost_poisson"] = evaluate_predictions(
            test_df["burglary_count"], cat_preds
        )
        test_df.assign(prediction=cat_preds).to_csv(
            predictions_dir / "catboost_poisson.csv", index=False
        )
    else:
        print("CatBoost not installed; skipping this model.")
        metrics["catboost_poisson"] = {
            "rmse": None,
            "mae": None,
            "r2": None,
            "note": "CatBoost unavailable in current environment",
        }

    zip_features = [
        "grid_density_7d",
        "npu_crime_avg_30d",
        "lag_1",
        "lag_24",
        "lag_168",
        "is_weekend",
        "is_holiday",
    ]

    print("Training Zero-Inflated Poisson...")
    zip_model, zip_preds = train_zip(train_df, test_df, zip_features)
    metrics["zip"] = evaluate_predictions(test_df["burglary_count"], zip_preds)
    test_df.assign(prediction=zip_preds).to_csv(
        predictions_dir / "zip.csv", index=False
    )

    if HAS_PROPHET:
        print("Training Prophet on aggregated series...")
        (
            prophet_model,
            prophet_test,
            prophet_preds,
            prophet_metrics,
        ) = train_prophet(df, pd.Timestamp(args.cutoff))
        metrics["prophet_hourly_total"] = prophet_metrics
        prophet_output = pd.DataFrame(
            {
                "ds": prophet_test["ds"].values,
                "y": prophet_test["y"].values,
                "prediction": prophet_preds,
            }
        )
        prophet_output.to_csv(
            predictions_dir / "prophet_hourly_total.csv", index=False
        )
    else:
        print("Prophet not installed; skipping aggregated time-series model.")
        metrics["prophet_hourly_total"] = {
            "rmse": None,
            "mae": None,
            "r2": None,
            "note": "Prophet unavailable in current environment",
        }

    metrics.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})

    metrics_path = args.output / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {metrics_path}")

    if wandb_run is not None:
        flat_metrics = {}
        for model_name, model_metrics in metrics.items():
            for metric_name, value in model_metrics.items():
                flat_metrics[f"{model_name}/{metric_name}"] = value
        wandb.log(flat_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()