import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import wandb


# Time-based split: train before this, test on/after this
CUTOFF = pd.Timestamp("2024-01-01 00:00:00")

# Lighter set of horizons (instead of 1..24)
# Short term: 1, 2, 3 hrs | Medium: 6, 12 hrs | Longer: 24 hrs
HORIZONS = [1, 2, 3, 6, 12, 24]


def build_hourly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert incident-level data into an hourly time series per NPU.
    Assumes df has columns: 'report_date' (datetime), 'npu'.
    """

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

    # Temporal features
    base["hour_of_day"] = base["hour"].dt.hour
    base["day_of_week"] = base["hour"].dt.dayofweek  # 0=Mon, 6=Sun
    base["month"] = base["hour"].dt.month
    base["is_weekend"] = (base["day_of_week"] >= 5).astype(int)

    # Cyclical encodings
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


        for lag in [1, 2, 3, 6, 12, 24]:
            g[f"lag_{lag}"] = g["crime_count"].shift(lag)


        for win in [3, 6, 12, 24]:
            g[f"roll_mean_{win}"] = g["crime_count"].shift(1).rolling(win).mean()


        for h in horizons:
            g[f"y_h{h}"] = g["crime_count"].shift(-h)

        groups.append(g)

    full = pd.concat(groups, axis=0, ignore_index=True)
    full = full.dropna().reset_index(drop=True)
    return full



def main():
    parser = argparse.ArgumentParser(
        description="Hourly multi-horizon (lighter) crime count forecaster (NPU-level)."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to crime_dataset.csv",
    )
    parser.add_argument(
        "--project",
        default="atl-crime-hourly-forecast",
        help="Weights & Biases project name",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {os.path.abspath(csv_path)}")
        sys.exit(1)


    df = pd.read_csv(csv_path, low_memory=False)
    df.rename(columns={'ReportDate': 'report_date'}, inplace=True)
    df.rename(columns={'npu_label': 'npu'}, inplace=True)

    if "report_date" not in df.columns:
        raise RuntimeError("Expected 'report_date' column in CSV.")
    if "npu" not in df.columns:
        raise RuntimeError("Expected 'npu' column in CSV.")


    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"]).copy()


    df = df[
        (df["report_date"] >= "2021-01-01")
        & (df["report_date"] <= "2024-12-31")
    ].copy()


    wandb.init(
        project=args.project,
        job_type="train_regression",
        config={
            "model_type": "MultiOutput XGBRegressor",
            "target": "hourly crime_count per NPU",
            "horizons": HORIZONS,
            "cutoff": str(CUTOFF),
        },
    )
    cfg = wandb.config


    panel = build_hourly_panel(df)
    supervised = build_supervised(panel, horizons=HORIZONS)

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

    # --------------------------------------------------------------
    # Rolling / expanding-window validation across years.
    # Example folds:
    #   - 2021 train → 2022 validation
    #   - 2021–2022 train → 2023 validation
    # These use the same hyperparameters as the final model.
    # --------------------------------------------------------------
    years = supervised["hour"].dt.year
    cv_folds = [
        ("cv_train_2021_val_2022", [2021], [2022]),
        ("cv_train_2021_2022_val_2023", [2021, 2022], [2023]),
    ]

    cv_metrics = {}
    for fold_name, train_years, val_years in cv_folds:
        train_mask_cv = years.isin(train_years)
        val_mask_cv = years.isin(val_years)

        train_cv = supervised.loc[train_mask_cv].copy()
        val_cv = supervised.loc[val_mask_cv].copy()

        if train_cv.empty or val_cv.empty:
            continue

        X_train_cv = train_cv[feature_cols]
        y_train_cv = train_cv[target_cols]
        X_val_cv = val_cv[feature_cols]
        y_val_cv = val_cv[target_cols]

        base_model_cv = XGBRegressor(**base_model.get_params())
        model_cv = MultiOutputRegressor(base_model_cv)
        model_cv.fit(X_train_cv, y_train_cv)

        y_val_pred = pd.DataFrame(
            model_cv.predict(X_val_cv),
            columns=target_cols,
            index=y_val_cv.index,
        )

        rmse_list = []
        mae_list = []
        r2_list = []

        for h in HORIZONS:
            col = f"y_h{h}"
            y_true = y_val_cv[col]
            y_hat = y_val_pred[col]
            mse = mean_squared_error(y_true, y_hat)
            rmse_list.append(float(np.sqrt(mse)))
            mae_list.append(float(mean_absolute_error(y_true, y_hat)))
            r2_list.append(float(r2_score(y_true, y_hat)))

        cv_metrics[f"{fold_name}_rmse_mean"] = float(np.mean(rmse_list))
        cv_metrics[f"{fold_name}_mae_mean"] = float(np.mean(mae_list))
        cv_metrics[f"{fold_name}_r2_mean"] = float(np.mean(r2_list))

    if cv_metrics:
        wandb.log(cv_metrics)

    model = MultiOutputRegressor(base_model)

    # Fit
    model.fit(X_train, y_train)


    y_pred = pd.DataFrame(
        model.predict(X_test),
        columns=target_cols,
        index=y_test.index,
    )
    # ------------------------------------------------------------------
    # Metrics: per-horizon RMSE / MAE / R2, plus baseline + skill scores
    # Baseline definition: naive persistence at each row,
    # predicting all future horizons equal to the current crime_count.
    # ------------------------------------------------------------------
    metrics: dict[str, float] = {}

    baseline_source = test["crime_count"]

    for h in HORIZONS:
        col = f"y_h{h}"

        y_true = y_test[col]
        y_hat = y_pred[col]
        y_base = baseline_source

        mse_model = mean_squared_error(y_true, y_hat)
        rmse_model = np.sqrt(mse_model)
        mae_model = mean_absolute_error(y_true, y_hat)
        r2_model = r2_score(y_true, y_hat)

        mse_base = mean_squared_error(y_true, y_base)
        rmse_base = np.sqrt(mse_base)
        mae_base = mean_absolute_error(y_true, y_base)
        r2_base = r2_score(y_true, y_base)

        metrics[f"rmse_h{h}"] = float(rmse_model)
        metrics[f"mae_h{h}"] = float(mae_model)
        metrics[f"r2_h{h}"] = float(r2_model)

        metrics[f"baseline_rmse_h{h}"] = float(rmse_base)
        metrics[f"baseline_mae_h{h}"] = float(mae_base)
        metrics[f"baseline_r2_h{h}"] = float(r2_base)

        if rmse_base > 0:
            metrics[f"skill_rmse_h{h}"] = float(1.0 - rmse_model / rmse_base)
        else:
            metrics[f"skill_rmse_h{h}"] = float("nan")

        if mae_base > 0:
            metrics[f"skill_mae_h{h}"] = float(1.0 - mae_model / mae_base)
        else:
            metrics[f"skill_mae_h{h}"] = float("nan")

    metrics["rmse_mean"] = float(
        np.mean([metrics[f"rmse_h{h}"] for h in HORIZONS])
    )
    metrics["mae_mean"] = float(
        np.mean([metrics[f"mae_h{h}"] for h in HORIZONS])
    )
    metrics["r2_mean"] = float(
        np.mean([metrics[f"r2_h{h}"] for h in HORIZONS])
    )

    metrics["baseline_rmse_mean"] = float(
        np.mean([metrics[f"baseline_rmse_h{h}"] for h in HORIZONS])
    )
    metrics["baseline_mae_mean"] = float(
        np.mean([metrics[f"baseline_mae_h{h}"] for h in HORIZONS])
    )
    metrics["baseline_r2_mean"] = float(
        np.mean([metrics[f"baseline_r2_h{h}"] for h in HORIZONS])
    )

    metrics["skill_rmse_mean"] = float(
        np.nanmean([metrics[f"skill_rmse_h{h}"] for h in HORIZONS])
    )
    metrics["skill_mae_mean"] = float(
        np.nanmean([metrics[f"skill_mae_h{h}"] for h in HORIZONS])
    )

    print("Average RMSE across horizons:", metrics["rmse_mean"])
    print("Average MAE across horizons:", metrics["mae_mean"])
    print("Average R2 across horizons:", metrics["r2_mean"])
    print("Average RMSE skill vs baseline:", metrics["skill_rmse_mean"])
    print("Average MAE skill vs baseline:", metrics["skill_mae_mean"])

    wandb.log(metrics)


    os.makedirs("outputs", exist_ok=True)
    out = test[["npu", "hour", "crime_count"]].copy()
    for h in HORIZONS:
        out[f"pred_h{h}"] = y_pred[f"y_h{h}"]

    out.to_csv(
        "outputs/hourly_npu_predictions_multi_horizon.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Long-form export for Streamlit:
    # One row per (npu, reference_hour, horizon_hours) containing:
    # - crime_count_at_reference
    # - model prediction for that horizon
    # - naive baseline prediction (persistence)
    # - forecast_hour = reference_hour + horizon
    # ------------------------------------------------------------------
    baseline_aligned = baseline_source.reindex(out.index)

    long_rows = []
    for h in HORIZONS:
        df_h = out[["npu", "hour", "crime_count", f"pred_h{h}"]].copy()
        df_h["horizon_hours"] = h
        df_h["forecast_hour"] = df_h["hour"] + pd.to_timedelta(h, unit="h")
        df_h["baseline_pred"] = baseline_aligned.values
        df_h = df_h.rename(
            columns={
                "hour": "reference_hour",
                "crime_count": "crime_count_at_reference",
                f"pred_h{h}": "pred_model",
            }
        )
        long_rows.append(df_h)

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df.to_csv(
        "outputs/hourly_npu_predictions_long_for_streamlit.csv",
        index=False,
    )

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
