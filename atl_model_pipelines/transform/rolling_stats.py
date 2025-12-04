# Compute NPU rolling stats

import pandas as pd
from rich.console import Console

console = Console()


def add_npu_rolling_average(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Compute 30-day rolling mean crimes per NPU."""

    df_local = df.copy()
    df_local["date"] = df_local["report_date"].dt.date

    daily = (
        df_local.groupby(["npu", "date"])
                .size()
                .reset_index(name="daily_npu_count")
    )
    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.sort_values(["npu", "date"])
    feature_name = f"npu_crime_avg_{window_days}d"

    daily[feature_name] = daily.groupby("npu")["daily_npu_count"].transform(
        lambda x: x.shift(1).rolling(window_days, min_periods=1).mean()
    )

    df_local["date"] = pd.to_datetime(df_local["report_date"].dt.date)

    df_local = df_local.merge(
        daily[["npu", "date", feature_name]],
        on=["npu", "date"],
        how="left"
    ).drop(columns=["date"], errors="ignore")

    df_local[feature_name] = df_local[feature_name].fillna(0)

    console.print(f"[green]{feature_name} added.")
    return df_local