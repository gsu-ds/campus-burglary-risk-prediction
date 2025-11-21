import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "cleaned_full_atl_v8.csv"
RESULTS_DIR = PROJECT_ROOT / "reports" / "test_results" / "hourly_baseline"


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute MAPE while safely handling zero targets.

    Returns NaN if all targets are zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    non_zero = y_true != 0
    if not np.any(non_zero):
        return float("nan")

    return float(
        np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))
        * 100.0
    )


def build_hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate incident-level records into a single hourly time series.

    Expects a datetime column named 'ReportDate'.
    """
    if "ReportDate" not in df.columns:
        raise RuntimeError("Expected 'ReportDate' column in input CSV.")

    df = df.copy()
    df["ReportDate"] = pd.to_datetime(df["ReportDate"])
    df = df.sort_values("ReportDate")

    hourly = (
        df.groupby(df["ReportDate"].dt.floor("h"))
        .size()
        .reset_index(name="crime_count")
    )
    hourly = hourly.rename(columns={"ReportDate": "date"})

    # Ensure a continuous hourly index and fill missing hours with zero incidents.
    full_range = pd.date_range(
        start=hourly["date"].min(),
        end=hourly["date"].max(),
        freq="h",
    )
    hourly = (
        hourly.set_index("date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    return hourly


def add_baseline_columns(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple baseline forecasts to the hourly series:

    - naive_24: value from 24 hours ago (daily pattern)
    - naive_168: value from 168 hours ago (weekly pattern)
    - ma_24: rolling mean of the previous 24 hours

    All baselines are constructed so that they never use information
    from the future relative to the prediction time.
    """
    hourly = hourly.copy()

    # Simple lags
    hourly["naive_24"] = hourly["crime_count"].shift(24)
    hourly["naive_168"] = hourly["crime_count"].shift(168)

    # Rolling mean of the previous 24 hours (exclude current hour)
    hourly["ma_24"] = (
        hourly["crime_count"]
        .rolling(window=24, min_periods=1)
        .mean()
        .shift(1)
    )

    return hourly


def evaluate_baselines(
    hourly: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the hourly series into train / test by cutoff and evaluate
    the baseline forecasts on the test portion.

    Returns:
        results_df: one row per baseline model with summary metrics.
        preds_df:   test-set predictions for downstream analysis.
    """
    hourly = hourly.copy()

    mask_train = hourly["date"] < cutoff
    mask_test = hourly["date"] >= cutoff

    train = hourly.loc[mask_train].copy()
    test = hourly.loc[mask_test].copy()

    if train.empty or test.empty:
        raise RuntimeError(
            "Train or test set is empty. "
            "Check the cutoff date versus the data range."
        )

    # Fill any initial NaNs in baseline columns with the training mean
    train_mean = float(train["crime_count"].mean())
    baseline_cols = ["naive_24", "naive_168", "ma_24"]
    for col in baseline_cols:
        if col not in hourly.columns:
            raise RuntimeError(f"Missing baseline column '{col}' in hourly DataFrame.")
        hourly[col] = hourly[col].fillna(train_mean)

    test = hourly.loc[mask_test].copy()

    y_true = test["crime_count"].to_numpy()

    rows = []
    for name, col in [
        ("Naive Seasonal (24-hr)", "naive_24"),
        ("Naive Seasonal (168-hr)", "naive_168"),
        ("Moving Average (24-hr)", "ma_24"),
    ]:
        y_pred = test[col].to_numpy()
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = safe_mape(y_true, y_pred)

        rows.append(
            {
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
            }
        )

    results_df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)

    preds_df = test[["date", "crime_count"] + baseline_cols].copy()
    preds_df = preds_df.rename(columns={"crime_count": "actual"})

    return results_df, preds_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hourly baseline forecaster for Atlanta burglary/larceny counts.\n"
            "Builds simple naive and moving-average baselines on the full city "
            "time series and evaluates them on a held-out test period."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help=(
            "Path to cleaned_full_atl_v8.csv (or equivalent incident-level file) "
            f"[default: {DEFAULT_CSV}]"
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2024-01-01 00:00:00",
        help=(
            "Datetime string used to split train/test; "
            "observations strictly before this go to train, "
            "on/after go to test. [default: 2024-01-01 00:00:00]"
        ),
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    cutoff_ts = pd.to_datetime(args.cutoff)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading incidents from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    hourly = build_hourly_series(df)
    hourly = add_baseline_columns(hourly)

    print(
        f"Built hourly series with {len(hourly)} rows "
        f"from {hourly['date'].min()} to {hourly['date'].max()}."
    )
    print(f"Using cutoff: {cutoff_ts} (train before, test on/after).")

    results_df, preds_df = evaluate_baselines(hourly, cutoff_ts)

    results_path = RESULTS_DIR / "baseline_model_results.csv"
    preds_path = RESULTS_DIR / "baseline_predictions.csv"

    results_df.to_csv(results_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    print("\n=== Hourly Baseline Model Performance (test set) ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved metrics to: {results_path}")
    print(f"Saved predictions to: {preds_path}")


if __name__ == "__main__":
    main()


