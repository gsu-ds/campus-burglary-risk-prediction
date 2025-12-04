import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import wandb


### Time-based split:
###Train on data before this (2020-01-01 to 2023-12-31),
## Test on/after this (2024-01-01 to 2025-12-31)
CUTOFF = pd.Timestamp("2024-01-01 00:00:00")

## Forecast horizons: 1, 2, 3, 6, 12, 24 hours ahead
HORIZONS = [1, 2, 3, 6, 12, 24]


def build_hourly_panel(df: pd.DataFrame) -> pd.DataFrame:


    df["hour"] = df["report_date"].dt.floor("H")

    df = df.dropna(subset=["npu"]).copy()
    df["npu"] = df["npu"].astype(str)

    counts = df.groupby(["npu", "hour"]).size().reset_index(name="crime_count")

    all_hours = pd.date_range(
        df["hour"].min().floor("H"),
        df["hour"].max().ceil("H"),
        freq="H",
    )
    npus = counts["npu"].unique()
    idx = pd.MultiIndex.from_product([npus, all_hours], names=["npu", "hour"])

    base = pd.DataFrame(index=idx).reset_index()
    base = base.merge(counts, on=["npu", "hour"], how="left")
    base["crime_count"] = base["crime_count"].fillna(0).astype(int)

    ### Temporal features
    base["hour_of_day"] = base["hour"].dt.hour
    base["day_of_week"] = base["hour"].dt.dayofweek  # 0=Mon, 6=Sun
    base["month"] = base["hour"].dt.month
    base["is_weekend"] = (base["day_of_week"] >= 5).astype(int)

    ##Cyclical 
    base["sin_hour"] = np.sin(2 * np.pi * base["hour_of_day"] / 24.0)
    base["cos_hour"] = np.cos(2 * np.pi * base["hour_of_day"] / 24.0)
    base["sin_dow"] = np.sin(2 * np.pi * base["day_of_week"] / 7.0)
    base["cos_dow"] = np.cos(2 * np.pi * base["day_of_week"] / 7.0)

    return base


def build_supervised(panel: pd.DataFrame, horizons=None) -> pd.DataFrame:
    """
    Turn the hourly panel into a supervised dataset with:
    - Lag & rolling features
    - Targets y_h1, y_h2, ... for each horizon
    """
    if horizons is None:
        horizons = HORIZONS

    groups = []

    for npu, g in panel.groupby("npu"):
        g = g.sort_values("hour").reset_index(drop=True).copy()

        ##### Lags
        for lag in [1, 2, 3, 6, 12, 24]:
            g[f"lag_{lag}"] = g["crime_count"].shift(lag)

        ### Rolling means (using past data only)
        for win in [3, 6, 12, 24]:
            g[f"roll_mean_{win}"] = g["crime_count"].shift(1).rolling(win).mean()

        ## Targets for each horizon
        for h in horizons:
            g[f"y_h{h}"] = g["crime_count"].shift(-h)

        groups.append(g)

    full = pd.concat(groups, axis=0, ignore_index=True)
    full = full.dropna().reset_index(drop=True)
    return full


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hourly multi-horizon NPU-level crime count forecaster "
            "(predicts counts 1, 2, 3, 6, 12, and 24 hours ahead)."
        )
    )

    #### Use parquet input instead of CSV, with default file name
    parser.add_argument(
        "--parquet",
        default="target_crimes.parquet",
        help="Path to target_crimes.parquet",
    )
    parser.add_argument(
        "--project",
        default="atl-crime-hourly-forecast",
        help="Weights & Biases project name",
    )
    args = parser.parse_args()

    parquet_path = args.parquet
    if not os.path.exists(parquet_path):
        print(f"ERROR: Parquet file not found at {os.path.abspath(parquet_path)}")
        sys.exit(1)

    ### Read parquet
    df = pd.read_parquet(parquet_path)

    if "report_date" not in df.columns:
        raise RuntimeError("Expected 'report_date' column in parquet data.")
    if "npu" not in df.columns:
        raise RuntimeError("Expected 'npu' column in parquet data.")

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"]).copy()

    ## Filter to time range:
    ##### train window: 2020-01-01 to 2023-12-31 (handled by CUTOFF below)
    ### test window:  2024-01-01 to 2025-12-31
    df = df[
        (df["report_date"] >= "2020-01-01")
        & (df["report_date"] <= "2025-12-31")
    ].copy()

    wandb.init(
        project=args.project,
        job_type="train_regression",
        config={
            "model_type": "MultiOutput XGBRegressor",
            "target": (
                "hourly crime_count per NPU, forecasting counts "
                "1, 2, 3, 6, 12, and 24 hours in the future for each selected NPU "
                "using 2020-2023 for training and 2024-2025 for testing"
            ),
            "horizons": HORIZONS,
            "cutoff": str(CUTOFF),
            "train_range": "2020-01-01 to 2023-12-31",
            "test_range": "2024-01-01 to 2025-12-31",
        },
    )
    cfg = wandb.config

    ####  hourly NPU panel and supervised dataset
    panel = build_hourly_panel(df)
    supervised = build_supervised(panel, horizons=HORIZONS)

    ###### Time-based split
    train = supervised[supervised["hour"] < CUTOFF].copy()
    test = supervised[supervised["hour"] >= CUTOFF].copy()

    if train.empty or test.empty:
        raise RuntimeError(
            "Train or test set is empty; check CUTOFF date vs data range."
        )

    target_cols = [f"y_h{h}" for h in HORIZONS]

    feature_cols = (
        [
            "crime_count",
            "hour_of_day",
            "day_of_week",
            "month",
            "is_weekend",
            "sin_hour",
            "cos_hour",
            "sin_dow",
            "cos_dow",
        ]
        + [f"lag_{lag}" for lag in [1, 2, 3, 6, 12, 24]]
        + [f"roll_mean_{win}" for win in [3, 6, 12, 24]]
    )

    X_train = train[feature_cols]
    y_train = train[target_cols]
    X_test = test[feature_cols]
    y_test = test[target_cols]

    wandb.log(
        {
            "train_rows": len(train),
            "test_rows": len(test),
            "num_npus": panel["npu"].nunique(),
        }
    )

    print(
        f"Training on {len(X_train)} rows, "
        f"{len(feature_cols)} features, "
        f"{len(target_cols)} horizons: {HORIZONS}"
    )

    n_estimators = getattr(cfg, "n_estimators", 250)
    max_depth = getattr(cfg, "max_depth", 5)
    learning_rate = getattr(cfg, "learning_rate", 0.05)
    subsample = getattr(cfg, "subsample", 0.9)
    colsample_bytree = getattr(cfg, "colsample_bytree", 0.9)

    base_model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )

    model = MultiOutputRegressor(base_model)

    #### Fit
    model.fit(X_train, y_train)

    ### Predict
    y_pred = pd.DataFrame(
        model.predict(X_test),
        columns=target_cols,
        index=y_test.index,
    )

    ## Metrics per horizon
    metrics = {}
    for h in HORIZONS:
        col = f"y_h{h}"

        mse = mean_squared_error(y_test[col], y_pred[col])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[col], y_pred[col])
        metrics[f"rmse_h{h}"] = float(rmse)
        metrics[f"mae_h{h}"] = float(mae)

    metrics["rmse_mean"] = float(
        np.mean([metrics[f"rmse_h{h}"] for h in HORIZONS])
    )
    metrics["mae_mean"] = float(
        np.mean([metrics[f"mae_h{h}"] for h in HORIZONS])
    )

    print("Average RMSE across horizons:", metrics["rmse_mean"])
    print("Average MAE across horizons:", metrics["mae_mean"])

    wandb.log(metrics)

    #### Save predictions
    os.makedirs("outputs", exist_ok=True)
    out = test[["npu", "hour", "crime_count"]].copy()
    for h in HORIZONS:
        out[f"pred_h{h}"] = y_pred[f"y_h{h}"]

    out.to_csv(
        "outputs/hourly_npu_predictions_multi_horizon.csv",
        index=False,
    )

    ###### Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/xgb_multioutput_hourly_npu_lite.joblib")

    meta = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "cutoff": str(CUTOFF),
        "horizons": HORIZONS,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
    }
    with open("artifacts/meta_hourly_npu_lite.json", "w") as f:
        json.dump(meta, f, indent=2)

    model_art = wandb.Artifact("xgb_multioutput_hourly_npu_lite", type="model")
    model_art.add_file("artifacts/xgb_multioutput_hourly_npu_lite.joblib")
    model_art.add_file("artifacts/meta_hourly_npu_lite.json")
    wandb.log_artifact(model_art)

    wandb.finish()


if __name__ == "__main__":
    main()
