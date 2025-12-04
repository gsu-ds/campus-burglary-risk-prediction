# Converts APD df to GeoDF for spatial attributes

from pathlib import Path
from typing import List
import pandas as pd
import geopandas as gpd
from rich.console import Console

from config import (
    NPU_SHP,
    APD_ZONE_SHP,
    CAMPUS_SHP,
    NEIGHBORHOOD_SHP,
    CITIES_SHP,
)
from atl_model_pipelines.utils.logging import log_step

console = Console()


def to_gdf(df: pd.DataFrame, lon_col: str = "longitude", lat_col: str = "latitude") -> gpd.GeoDataFrame:
    """Convert a DataFrame into a GeoDataFrame."""
    for col in (lon_col, lat_col):
        if col not in df.columns:
            raise KeyError(f"Expected coordinate column '{col}' not found.")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )


def load_shapefile(path: Path, target_crs: str) -> gpd.GeoDataFrame:
    """Load and reproject shapefile."""
    if not path.exists():
        raise FileNotFoundError(f"Shapefile missing: {path}")

    try:
        gdf = gpd.read_file(path)
    except Exception:
        console.print(f"[bold red]Could not load shapefile:[/bold red] {path}")
        raise

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    return gdf.to_crs(target_crs)


def attach_shapefile(
    gdf: gpd.GeoDataFrame,
    shp_path: Path,
    name_col_candidates: List[str],
    target_col: str,
    predicate: str = "within",
    nearest: bool = False,
) -> gpd.GeoDataFrame:
    """
    Generic wrapper for spatial joins.
    - nearest=True used for campus polygons
    - predicate="within" used for NPU, neighborhood, city, zone
    """
    if not shp_path.exists():
        console.print(f"[yellow]Skipping {target_col}: shapefile not found.[/yellow]")
        return gdf

    console.print(f"[cyan]Spatial attach:[/cyan] {target_col} (nearest={nearest})")

    shp = load_shapefile(shp_path, gdf.crs)

    name_col = next((c for c in name_col_candidates if c in shp.columns), shp.columns[0])

    shp = shp[[name_col, "geometry"]].rename(columns={name_col: "_temp_target"})

    if nearest:
        gdf_proj = gdf.to_crs("EPSG:3857")
        shp_proj = shp.to_crs("EPSG:3857")

        joined = gpd.sjoin_nearest(gdf_proj, shp_proj, how="left")
        joined = joined.to_crs(gdf.crs)
    else:
        joined = gpd.sjoin(gdf, shp, how="left", predicate=predicate)

    joined = joined.rename(columns={"_temp_target": target_col})
    joined = joined.drop(columns=["index_right"], errors="ignore")

    joined = joined[~joined.index.duplicated(keep="first")]

    return joined


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main spatial enrichment pipeline:
    - preserve raw APD NPU
    - convert to GeoDataFrame
    - join NPU, zone, campus, neighborhood, city
    - final NPU assignment logic
    """
    console.print("\n[bold cyan]Spatial enrichment...[/bold cyan]")

    df = df.copy()

    if "npu" in df.columns:
        df.rename(columns={"npu": "npu_raw"}, inplace=True)
        console.print("[green]Preserved APD NPU as 'npu_raw'.[/green]")

    df_geo = df.dropna(subset=["longitude", "latitude"]).copy()
    dropped = len(df) - len(df_geo)

    if dropped > 0:
        console.print(f"[yellow]{dropped:,} rows dropped due to missing coordinates.[/yellow]")

    gdf = to_gdf(df_geo)

    gdf = attach_shapefile(gdf, NPU_SHP, ["NPU", "NPU_ID", "NAME"], "npu_shp", nearest=True)
    gdf = attach_shapefile(gdf, APD_ZONE_SHP, ["ZONE", "Zone"], "zone_raw")
    gdf = attach_shapefile(gdf, CAMPUS_SHP, ["FULLNAME"], "campus_label_shp", nearest=True)
    gdf = attach_shapefile(gdf, NEIGHBORHOOD_SHP, ["NAME"], "neighborhood")
    gdf = attach_shapefile(gdf, CITIES_SHP, ["NAME"], "city_label")

    df_enriched = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))

    df_enriched["npu"] = (
        df_enriched["npu_shp"]
        .fillna(df_enriched.get("npu_raw"))
        .astype(str)
        .str.upper()
        .str.strip()
    )

    log_step("Spatial enrichment complete", df_enriched)

    return df_enriched