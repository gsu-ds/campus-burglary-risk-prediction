# pipeline/transforms/npu_rolling.py

import pandas as pd
from rich.console import Console

console = Console()

def compute_npu_rolling_average(
    df: pd.DataFrame,
    window_days: int = 30,
    date_col: str = "report_date"
) -> pd.DataFrame:
    """
    Matches notebook Step 9.2:
    Compute daily NPU counts → 30-day rolling average → join back.
    """
    console.print(f"[cyan]Computing {window_days}-day NPU rolling averages...[/cyan]")

    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col]).dt.date

    daily = (
        df.groupby(["npu", "date"])
          .size()
          .reset_index(name="daily_npu_count")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["npu", "date"])

    feature_name = f"npu_crime_avg_{window_days}d"

    daily[feature_name] = (
        daily.groupby("npu")["daily_npu_count"]
             .transform(lambda x: x.shift(1).rolling(window_days, min_periods=1).mean())
    )

    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(
        daily[["npu", "date", feature_name]],
        on=["npu", "date"],
        how="left"
    ).drop(columns=["date"], errors="ignore")

    df[feature_name] = df[feature_name].fillna(0)

    console.print(f"[green]✓ Added column: {feature_name}")
    return df
