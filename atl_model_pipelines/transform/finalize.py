# Finalize dataset for modeling


import pandas as pd
import numpy as np


def finalize_target_crimes(df: pd.DataFrame) -> pd.DataFrame:
    """Finalize dataset for modeling."""

    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values("report_date").reset_index(drop=True)

    dt = df["report_date"].dt
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

    drop_cols = ["day_of_the_week", "zone", "zone_raw", "npu_raw", "npu_shp",
                 "neighborhood_raw", "neighborhood_shp", "campus_label_shp"]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df