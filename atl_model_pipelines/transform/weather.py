
#Merge hourly and daily weather features into crime dataframe.


import pandas as pd
from rich.console import Console

console = Console()


def merge_weather(df: pd.DataFrame, hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly and daily weather features into the crime dataframe."""

    df = df.copy()

    df["report_hour"] = df["report_date"].dt.floor("h")
    hourly_df["weather_hour"] = pd.to_datetime(hourly_df["datetime"]).dt.floor("h")

    df = df.merge(
        hourly_df.drop(columns=["datetime"], errors="ignore"),
        left_on="report_hour",
        right_on="weather_hour",
        how="left"
    ).drop(columns=["report_hour", "weather_hour"], errors="ignore")

    df["incident_date"] = df["report_date"].dt.date
    daily_df["weather_date"] = pd.to_datetime(daily_df["date"]).dt.date

    df = df.merge(
        daily_df.drop(columns=["date"], errors="ignore"),
        left_on="incident_date",
        right_on="weather_date",
        how="left"
    ).drop(columns=["incident_date", "weather_date"], errors="ignore")

    console.print("[green]Weather merged.[/green]")
    return df


def add_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive categorical weather flags (hot, cold, raining)."""

    df = df.copy()

    if "temp_f" not in df.columns:
        console.print("[yellow]Missing `temp_f`, skipping weather flags.[/yellow]")
        return df

    p85 = df["temp_f"].dropna().quantile(0.85)
    p15 = df["temp_f"].dropna().quantile(0.15)

    df["is_raining"] = (df.get("precip_in", 0).fillna(0) > 0.01).astype(int)
    df["is_hot"] = (df["temp_f"] >= p85).astype(int)
    df["is_cold"] = (df["temp_f"] <= p15).astype(int)

    console.print("[green]Weather flags added.[/green]")
    return df