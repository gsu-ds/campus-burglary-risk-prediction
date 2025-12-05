# To quickly run > python -m atl_model_pipelines.transfor
"""Builds three NPU × hour panels:

1. Target panel  (with lag features)
2. Sparse panel  (only observed NPU × hour)
3. Dense panel   (complete grid) — STREAMED to parquet so RAM never spikes

The dense panel is written NPU-by-NPU using Arrow row groups.
This prevents Codespaces OOM termination.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

import pyarrow as pa
import pyarrow.parquet as pq

from rich.console import Console
from rich.panel import Panel

from config import (
    DATA_TARGET,
    DATA_SPARSE,
    DATA_DENSE,
    DATA_TARGET_PANEL,
    SCHOOL_CENTERS,
)
from atl_model_pipelines.utils.logging import log_step

console = Console()

# -------------------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------------------

WRITE_CSV = False  # leave False for stability in Codespaces

BOOLISH_COLS = [
    "is_raining", "is_hot", "is_cold",
    "is_daylight", "is_weekend", "is_holiday",
]

NUMERIC_COLS_BASE = [
    "location_type_count", "incident_hour", "year", "month",
    "hour_sin", "hour_cos",
    "temp_f", "precip_in", "rain_in",
    "apparent_temp_f", "daylight_duration_sec", "sunshine_duration_sec",
    "precip_hours", "rain_sum_in", "temp_mean_f",
    "grid_density_7d", "npu_crime_avg_30d",
    "campus_distance_m",
]

CAT_COLS_BASE = [
    "day_number", "day_of_week", "hour_block",
    "is_holiday", "is_weekend", "is_daylight",
    "weather_code_hourly", "weather_code_daily",
    "offense_category", "campus_label", "campus_code",
    "event_watch_day_watch", "event_watch_evening_watch", "event_watch_morning_watch",
    "near_gsu", "near_ga_tech", "near_emory", "near_clark",
    "near_spelman", "near_morehouse", "near_morehouse_med",
    "near_atlanta_metro", "near_atlanta_tech",
    "near_scad", "near_john_marshall",
]

LAGS = [1, 3, 6, 12, 24, 168]
LAG_COLS = [f"lag_{l}" for l in LAGS]

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------

def load_target_crimes(path: Path = DATA_TARGET) -> pd.DataFrame:
    df = pd.read_parquet(path)

    df["report_date"] = pd.to_datetime(df["report_date"])
    df["npu"] = df["npu"].astype(str).str.upper().str.strip()

    return df


# -------------------------------------------------------------
# TARGET PANEL (with lag features)
# -------------------------------------------------------------

def build_target_panel(df: pd.DataFrame) -> pd.DataFrame:
    console.print("[cyan]Building enriched TARGET panel (optimized)...[/cyan]")

    df = df.copy()
    df["hour_ts"] = df["report_date"].dt.floor("h")

    numeric_cols = [c for c in NUMERIC_COLS_BASE if c in df.columns]
    cat_cols = [c for c in CAT_COLS_BASE if c in df.columns]

    cols = ["npu", "hour_ts"] + numeric_cols + cat_cols
    df = df[cols]

    pl_df = pl.from_pandas(df)

    # Burglary counts
    hourly_counts = (
        pl_df
        .group_by(["npu", "hour_ts"])
        .agg(pl.len().alias("burglary_count"))
        .to_pandas()
    )

    # Numeric means
    if numeric_cols:
        hourly_numeric = (
            pl_df.group_by(["npu", "hour_ts"])
            .agg([pl.col(c).mean().alias(c) for c in numeric_cols])
            .to_pandas()
        )
    else:
        hourly_numeric = hourly_counts[["npu", "hour_ts"]].copy()

    # Categorical modes
    if cat_cols:
        hourly_cat = (
            pl_df.group_by(["npu", "hour_ts"])
            .agg([pl.col(c).mode().first().alias(c) for c in cat_cols])
            .to_pandas()
        )
    else:
        hourly_cat = hourly_counts[["npu", "hour_ts"]].copy()

    # Merge panels
    panel = (
        hourly_counts
        .merge(hourly_numeric, on=["npu", "hour_ts"], how="left")
        .merge(hourly_cat, on=["npu", "hour_ts"], how="left")
        .sort_values(["npu", "hour_ts"])
        .reset_index(drop=True)
    )

    # LAG FEATURES
    for lag in LAGS:
        panel[f"lag_{lag}"] = panel.groupby("npu")["burglary_count"].shift(lag)

    panel = panel.dropna(subset=LAG_COLS)

    console.print(f"[yellow]TARGET panel → {panel.shape[0]:,} rows × {panel.shape[1]} cols[/yellow]")
    return panel


# -------------------------------------------------------------
# SPARSE PANEL
# -------------------------------------------------------------

def build_sparse_panel(base_panel: pd.DataFrame) -> pd.DataFrame:
    console.print("\n[bold cyan]Building SPARSE panel...[/bold cyan]")

    sparse = base_panel.copy()
    if "burglary_count" in sparse.columns:
        sparse = sparse.rename(columns={"burglary_count": "crime_count"})

    console.print(f"[green]Sparse panel → {sparse.shape[0]:,} rows × {sparse.shape[1]} cols[/green]")
    return sparse


# -------------------------------------------------------------
# STREAMING DENSE PANEL — THE FIX 
# -------------------------------------------------------------

def build_dense_panel_streaming(base_panel: pd.DataFrame, output_path: Path):
    """
    Build DENSE panel using an Arrow ParquetWriter streaming approach.
    This avoids ever holding the full dense panel in memory.
    """

    console.print("\n[bold cyan]Building DENSE panel (STREAMING, memory-safe)...[/bold cyan]")

    base_panel = base_panel.copy()
    base_panel["hour_ts"] = pd.to_datetime(base_panel["hour_ts"])

    npus = sorted(base_panel["npu"].unique())
    start = base_panel["hour_ts"].min()
    end   = base_panel["hour_ts"].max()
    all_hours = pd.date_range(start=start, end=end, freq="h")

    writer = None

    for npu in npus:
        console.print(f"[cyan]Processing NPU {npu}...[/cyan]")

        df_npu = base_panel[base_panel["npu"] == npu]

        # Build this NPU’s full time grid
        grid = pd.DataFrame({
            "npu": [npu] * len(all_hours),
            "hour_ts": all_hours,
        })

        merged = grid.merge(df_npu, on=["npu", "hour_ts"], how="left")

        # Fill missing
        num_cols = merged.select_dtypes(include=["number"]).columns
        cat_cols = merged.select_dtypes(include=["object", "category"]).columns

        merged[num_cols] = merged[num_cols].fillna(0)
        """merged[cat_cols] = merged[cat_cols].fillna("MISSING").astype(str)"""
        # Handle categorical columns safely
        for col in cat_cols:
            if str(merged[col].dtype).startswith("category"):
                # Add "MISSING" category if not exists
                if "MISSING" not in merged[col].cat.categories:
                    merged[col] = merged[col].cat.add_categories(["MISSING"])
                merged[col] = merged[col].fillna("MISSING")
                merged[col] = merged[col].astype(str)  # convert to string so parquet doesn't choke
            else:
                merged[col] = merged[col].fillna("MISSING").astype(str)


        # Convert to Arrow
        table = pa.Table.from_pandas(merged, preserve_index=False)

        # Initialize writer once
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")

        writer.write_table(table)

    if writer is not None:
        writer.close()

    console.print(f"[green]DENSE panel saved → {output_path}[/green]")
    console.print(f"[green]Dense panel shape: {len(all_hours) * len(npus):,} x {len(merged.columns)}[/green]"
)


# -------------------------------------------------------------
# BOOLEAN CLEANUP
# -------------------------------------------------------------

def clean_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BOOLISH_COLS:
        if col in df.columns:
            df[col] = df[col].replace("MISSING", 0).fillna(0).astype("int8")
    return df


# -------------------------------------------------------------
# SAVE TARGET + SPARSE (dense is saved via streaming)
# -------------------------------------------------------------

def save_target_and_sparse(target_panel, sparse_panel):
    console.print(
        Panel("[bold cyan]Saving target + sparse panels...[/bold cyan]", border_style="cyan")
    )

    # TARGET
    tp = clean_boolean_columns(target_panel)
    tp.to_parquet(DATA_TARGET_PANEL, index=False, compression="snappy")
    if WRITE_CSV:
        tp.to_csv(DATA_TARGET_PANEL.with_suffix(".csv"), index=False)
    console.print(f"[green]Saved target panel → {DATA_TARGET_PANEL.name}[/green]")

    # SPARSE
    sp = clean_boolean_columns(sparse_panel)
    sp.to_parquet(DATA_SPARSE, index=False, compression="snappy")
    if WRITE_CSV:
        sp.to_csv(DATA_SPARSE.with_suffix(".csv"), index=False)
    console.print(f"[green]Saved sparse panel → {DATA_SPARSE.name}[/green]")


# -------------------------------------------------------------
# ORCHESTRATOR
# -------------------------------------------------------------

def build_all_panels(df: pd.DataFrame | None = None, save=True):
    if df is None:
        df = load_target_crimes()

    target_panel = build_target_panel(df)
    base_panel = target_panel.drop(columns=LAG_COLS, errors="ignore")

    sparse_panel = build_sparse_panel(base_panel)

    # Save target & sparse
    if save:
        save_target_and_sparse(target_panel, sparse_panel)

        # STREAM the dense panel — no RAM spike
        build_dense_panel_streaming(base_panel, DATA_DENSE)

    # Log
    log_step("Target panel", target_panel)
    log_step("Sparse panel", sparse_panel)

    return {
        "target": target_panel,
        "sparse": sparse_panel,
        "dense_streamed": str(DATA_DENSE),
    }


# -------------------------------------------------------------
# MAIN ENTRYPOINT
# -------------------------------------------------------------

if __name__ == "__main__":
    console.print("[bold green]Running optimized panel_builder (streaming)...[/bold green]")

    df = load_target_crimes()
    build_all_panels(df=df, save=True)

    console.print("[bold green]Panels successfully built and saved![/bold green]")
