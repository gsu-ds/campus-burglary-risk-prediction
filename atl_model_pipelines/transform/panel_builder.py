"""
panel_builder.py

Builds NPU × hour panels matching notebook Step 13 logic exactly:
- Uses specific aggregation dictionary for features
- Sparse panel: Only hours with incidents
- Dense panel: Complete grid with proper filling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from config import DATA_TARGET, DATA_SPARSE, DATA_DENSE, SCHOOL_CENTERS
from atl_model_pipelines.utils.logging import log_step

console = Console()

BOOLISH_COLS = [
    "is_raining", "is_hot", "is_cold", "is_daylight",
    "is_weekend", "is_holiday"
]

# Exact aggregation from notebook
AGG_DICT = {
    "grid_density_7d": "mean",
    "npu_crime_avg_30d": "first",
    "temp_f": "mean",
    "is_raining": "max",
    "is_hot": "max",
    "is_cold": "max",
    "is_daylight": "max",
    "is_weekend": "first",
    "is_holiday": "first",
    "day_number": "first",
    "month": "first",
    "year": "first",
    "hour_sin": "first",
    "hour_cos": "first",
    "day_of_week": "first",
    "campus_distance_m": "mean",
    "location_type_count": "mean",
}

# Add campus binary flags to aggregation
for campus in SCHOOL_CENTERS.keys():
    AGG_DICT[f"near_{campus.lower()}"] = "max"


def load_target_crimes(path: Path = DATA_TARGET) -> pd.DataFrame:
    """Load enriched target_crimes.parquet."""
    df = pd.read_parquet(path)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["npu"] = df["npu"].astype(str).str.upper().str.strip()
    return df


def build_panel_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated crime counts per NPU × hour.
    """
    console.print("[cyan]Building NPU × hour crime counts...[/cyan]")
    
    df = df.copy()
    df["hour_ts"] = df["report_date"].dt.floor("H")
    
    panel_target = (
        df.groupby(["npu", "hour_ts"])
        .size()
        .reset_index(name="crime_count")
    )
    
    return panel_target


def build_panel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract enriched features at NPU × hour level using notebook aggregation logic.
    """
    console.print("[cyan]Extracting features per NPU × hour...[/cyan]")
    
    df = df.copy()
    df["hour_ts"] = df["report_date"].dt.floor("H")
    
    # Apply aggregation only for columns that exist
    agg_to_use = {col: func for col, func in AGG_DICT.items() if col in df.columns}
    
    panel_features = (
        df.groupby(["npu", "hour_ts"])
        .agg(agg_to_use)
        .reset_index()
    )
    
    return panel_features


def build_sparse_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build SPARSE panel: Only NPU × hour combinations where crimes occurred.
    Matches notebook sparse panel logic exactly.
    """
    console.print("\n[bold cyan]Building SPARSE Panel[/bold cyan]")
    
    panel_target = build_panel_target(df)
    panel_features = build_panel_features(df)
    
    console.print("[cyan]Merging aggregated features with crime target panel...[/cyan]")
    
    panel_merged = panel_target.merge(
        panel_features,
        on=["npu", "hour_ts"],
        how="left",
    )
    
    # Fill numeric with 0
    num_cols = panel_merged.select_dtypes(include=["number"]).columns
    for col in num_cols:
        panel_merged[col] = panel_merged[col].fillna(0)
    
    # Fill categorical with "MISSING"
    cat_cols = panel_merged.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        panel_merged[col] = panel_merged[col].astype(str).fillna("MISSING")
    
    console.print(
        f"[green]Sparse NPU-hour panel: {panel_merged.shape[0]:,} rows × {panel_merged.shape[1]} cols[/green]"
    )
    
    return panel_merged


def build_dense_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build DENSE panel: Complete NPU × hour grid.
    Matches notebook dense panel logic exactly.
    """
    console.print("\n[bold cyan]Building DENSE Panel[/bold cyan]")
    console.print("[cyan]Building dense NPU-hour panel...[/cyan]")
    
    df_temp = df.copy()
    df_temp["hour_ts"] = df_temp["report_date"].dt.floor("H")
    
    # Get valid NPUs (single letters A-Z)
    valid_npus = sorted([
        n for n in df_temp["npu"].unique() 
        if isinstance(n, str) and n in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ])
    
    start_ts = df_temp["hour_ts"].min()
    end_ts = df_temp["hour_ts"].max()
    all_hours = pd.date_range(start=start_ts, end=end_ts, freq="h")
    
    console.print(f"[cyan]NPUs: {len(valid_npus)}, Hours: {len(all_hours):,}[/cyan]")
    
    # Create complete grid
    panel_index = pd.MultiIndex.from_product(
        [valid_npus, all_hours],
        names=["npu", "hour_ts"],
    )
    panel_dense = pd.DataFrame(index=panel_index).reset_index()
    
    # Build target and features
    panel_target = build_panel_target(df)
    panel_features = build_panel_features(df)
    
    # Merge
    panel_dense = panel_dense.merge(panel_target, on=["npu", "hour_ts"], how="left")
    panel_dense = panel_dense.merge(panel_features, on=["npu", "hour_ts"], how="left")
    
    # Fill missing values
    num_cols = panel_dense.select_dtypes(include=["number"]).columns
    cat_cols = panel_dense.select_dtypes(include=["object", "category"]).columns
    
    panel_dense[num_cols] = panel_dense[num_cols].fillna(0)
    
    for col in cat_cols:
        if col in ["npu", "hour_ts"]:
            continue
            
        if pd.api.types.is_categorical_dtype(panel_dense[col]):
            if "MISSING" not in panel_dense[col].cat.categories:
                panel_dense[col] = panel_dense[col].cat.add_categories(["MISSING"])
            panel_dense[col] = panel_dense[col].fillna("MISSING")
        else:
            panel_dense[col] = panel_dense[col].fillna("MISSING")
    
    console.print(
        f"[green]Dense panel: {panel_dense.shape[0]:,} rows × {panel_dense.shape[1]} cols[/green]"
    )
    
    return panel_dense


def clean_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure boolean indicator columns are clean int64."""
    df = df.copy()
    
    for col in BOOLISH_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace("MISSING", 0)
                .fillna(0)
                .astype(int)
            )
    
    return df


def save_panels(sparse_df: pd.DataFrame, dense_df: pd.DataFrame) -> None:
    """Save both panels to parquet and CSV."""
    
    console.print(
        Panel("[bold cyan]Saving sparse and dense panels...[/bold cyan]",
              border_style="cyan")
    )
    
    # Clean boolean columns
    sparse_df = clean_boolean_columns(sparse_df)
    dense_df = clean_boolean_columns(dense_df)
    
    # Sparse panel
    sparse_df.to_parquet(DATA_SPARSE, index=False, compression="snappy")
    sparse_df.to_csv(DATA_SPARSE.with_suffix(".csv"), index=False)
    console.print(f"[green]Saved sparse panel → {DATA_SPARSE.name}[/green]")
    
    # Dense panel
    dense_df.to_parquet(DATA_DENSE, index=False, compression="snappy")
    dense_df.to_csv(DATA_DENSE.with_suffix(".csv"), index=False)
    console.print(f"[green]Saved dense panel → {DATA_DENSE.name}[/green]")
    
    console.print(
        Panel.fit(
            "[bold green]Panel exports complete (sparse + dense).[/bold green]",
            border_style="green"
        )
    )


def build_all_panels(df: pd.DataFrame = None, save: bool = True) -> dict:
    """
    Build both sparse and dense panels matching notebook Step 13.
    
    Parameters:
        df: DataFrame with enriched target crimes. If None, loads from DATA_TARGET
        save: Whether to save panels to disk
    
    Returns:
        Dict with "sparse" and "dense" DataFrames
    """
    if df is None:
        console.print("[cyan]Loading target crimes from disk...[/cyan]")
        df = load_target_crimes()
    
    sparse = build_sparse_panel(df)
    dense = build_dense_panel(df)
    
    if save:
        save_panels(sparse, dense)
    
    log_step("Step 13: NPU panels & feature store exports", sparse)
    
    return {
        "sparse": sparse,
        "dense": dense
    }


__all__ = [
    "build_sparse_panel",
    "build_dense_panel",
    "build_all_panels",
    "load_target_crimes"
]