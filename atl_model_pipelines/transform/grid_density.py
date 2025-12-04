# Adds 7-day rolling grid-based density features

import numpy as np
import pandas as pd
import geopandas as gpd
from rich.console import Console

console = Console()


def build_spatial_grid(df: pd.DataFrame, cell_size_m: int = 500):
    """Project coordinates and build grid boundaries."""
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    xmin, ymin, xmax, ymax = gdf.total_bounds

    x_edges = np.arange(xmin, xmax + cell_size_m, cell_size_m)
    y_edges = np.arange(ymin, ymax + cell_size_m, cell_size_m)

    return gdf, x_edges, y_edges


def assign_grid_ids(gdf, x_edges, y_edges):
    """Assign each point a grid_x, grid_y, and grid_id."""
    gdf["grid_x"] = np.searchsorted(x_edges, gdf.geometry.x) - 1
    gdf["grid_y"] = np.searchsorted(y_edges, gdf.geometry.y) - 1
    gdf["grid_id"] = gdf["grid_x"].astype(str) + "_" + gdf["grid_y"].astype(str)
    return gdf


def compute_grid_density(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 7-day rolling grid density to df."""

    console.print("[cyan]Computing 7-day grid-based density...[/cyan]")

    gdf, x_edges, y_edges = build_spatial_grid(df)
    gdf = assign_grid_ids(gdf, x_edges, y_edges)

    gdf["date"] = gdf["report_date"].dt.date

    daily = (
        gdf.groupby(["grid_id", "date"])
           .size()
           .reset_index(name="grid_daily_count")
    )
    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.sort_values(["grid_id", "date"])
    daily["grid_density_7d"] = daily.groupby("grid_id")["grid_daily_count"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    )

    gdf["date"] = gdf["report_date"].dt.date
    gdf = gdf.merge(
        daily[["grid_id", "date", "grid_density_7d"]],
        on=["grid_id", "date"],
        how="left"
    )

    df["grid_density_7d"] = gdf["grid_density_7d"].values
    df["grid_id"] = gdf["grid_id"].values

    console.print("[green]Grid density added.")
    return df