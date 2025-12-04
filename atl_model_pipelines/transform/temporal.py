# Adds temporal features: datetime, hour blocks, holidays, cyclic time encodings, weekend flags...

import numpy as np
import pandas as pd
import holidays
from rich.console import Console

console = Console()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and contextual features to crime dataframe."""
    
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    dt = df["report_date"].dt

    df["incident_datetime"] = df["report_date"]
    df["incident_date"] = dt.date
    df["incident_hour"] = dt.hour
    df["year"] = dt.year
    df["month"] = dt.month
    df["day_of_week"] = dt.day_name()

    dow = dt.dayofweek
    apd_map = {6: 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
    df["day_number"] = dow.map(apd_map).astype("int8")

    bin_edges = [0, 5, 9, 13, 17, 21, 24]
    bin_labels = [
        "Early Night (0–4)",
        "Early Morning (5–8)",
        "Late Morning (9–12)",
        "Afternoon (13–16)",
        "Evening (17–20)",
        "Late Night (21–24)",
    ]
    df["hour_block"] = pd.cut(
        df["incident_hour"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,
        include_lowest=True,
    )

    years = sorted(df["report_date"].dt.year.unique().tolist())
    holiday_dates = holidays.country_holidays("US", subdiv="GA", years=years)
    df["is_holiday"] = df["report_date"].dt.date.isin(holiday_dates)

    df["hour_sin"] = np.sin(2 * np.pi * df["incident_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["incident_hour"] / 24)
    df["is_weekend"] = (dow >= 5)

    if "day_of_the_week" in df.columns:
        df = df.drop(columns=["day_of_the_week"])

    console.print("[green]Temporal/context features added.[/green]")
    return df