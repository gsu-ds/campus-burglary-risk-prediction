"""
campus_distance.py

Vectorized Haversine distance to nearest campus + binary flags + campus_code encoding.
Matches notebook Step 10 logic exactly.
"""

import numpy as np
import pandas as pd
from rich.console import Console
from typing import Tuple

from config import SCHOOL_CENTERS, CAMPUS_ENCODING, CAMPUS_DISTANCE_THRESHOLD_M

console = Console()


def calculate_nearest_campus_fully_vectorized(df_in: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Vectorized Haversine distance from each crime to nearest campus center.
    
    Returns:
        Tuple of (nearest_campus_label, nearest_distance_m)
    """
    df_local = df_in.copy()
    valid_mask = df_local["latitude"].notna() & df_local["longitude"].notna()
    valid_df = df_local[valid_mask].copy()

    nearest_campus = pd.Series("none", index=df_local.index)
    nearest_distance = pd.Series(np.nan, index=df_local.index)

    if len(valid_df) == 0:
        console.print("[yellow]No valid coordinates; skipping campus distance.[/yellow]")
        return nearest_campus, nearest_distance

    campus_names = list(SCHOOL_CENTERS.keys())
    campus_lats = np.array([lat for lat, lon in SCHOOL_CENTERS.values()])
    campus_lons = np.array([lon for lat, lon in SCHOOL_CENTERS.values()])

    crime_lats = valid_df["latitude"].values[:, np.newaxis]
    crime_lons = valid_df["longitude"].values[:, np.newaxis]

    R = 6371000
    lat1, lon1 = np.radians(crime_lats), np.radians(crime_lons)
    lat2, lon2 = np.radians(campus_lats), np.radians(campus_lons)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = R * c

    min_idx = distances.argmin(axis=1)
    min_distances = distances[np.arange(len(distances)), min_idx]

    nearest_distance.loc[valid_df.index] = min_distances
    within = min_distances <= CAMPUS_DISTANCE_THRESHOLD_M
    nearest_campus.loc[valid_df.index[within]] = [campus_names[i] for i in min_idx[within]]

    return nearest_campus, nearest_distance


def compute_campus_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add campus distance features matching notebook Step 10:
    - campus_label: Nearest campus name (or "none")
    - campus_distance_m: Distance to nearest campus in meters
    - campus_code: Integer encoding of campus_label
    - near_<campus>: Binary flag for each campus (1 if within threshold)
    
    Parameters:
        df: DataFrame with latitude/longitude columns
    
    Returns:
        DataFrame with campus features added
    """
    console.print(f"[cyan]Computing campus distance & proximity (threshold: {CAMPUS_DISTANCE_THRESHOLD_M:.0f}m)...[/cyan]")

    df = df.copy()

    df["campus_label"], df["campus_distance_m"] = calculate_nearest_campus_fully_vectorized(df)
    
    df["campus_distance_m"] = df["campus_distance_m"].round(4).fillna(0)
    df["campus_code"] = df["campus_label"].map(CAMPUS_ENCODING).fillna(0).astype(int)

    for campus in SCHOOL_CENTERS.keys():
        col = f"near_{campus.lower()}"
        df[col] = (df["campus_label"] == campus).astype(int)

    df = df.drop(columns=["campus_label_shp"], errors="ignore")

    console.print("[green]Campus distance & flags added.[/green]")
    return df