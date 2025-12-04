# Used to repair NPU, zone, and neighborhood using spatial joins

import pandas as pd
import geopandas as gpd
from rich.console import Console

from config import NPU_SHP, APD_ZONE_SHP, NEIGHBORHOOD_SHP

console = Console()


def repair_spatial_assignments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair NPU, zone, and neighborhood using spatial joins.
    
    Uses shapefile paths from config.py
    """

    df = df.copy()
    valid = df.dropna(subset=["longitude", "latitude"])
    gdf = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["longitude"], valid["latitude"]),
        crs="EPSG:4326"
    )

    if NPU_SHP.exists():
        npu_shp = gpd.read_file(NPU_SHP).to_crs("EPSG:4326")
        name_col = next((c for c in ["NPU", "NPU_ID", "NAME"] if c in npu_shp.columns), npu_shp.columns[0])
        missing = df["npu"].isna() | (df["npu"].astype(str).str.strip().isin(["", "0", "NAN"]))
        if missing.sum() > 0:
            gdf_missing = gdf.loc[df[missing].index]
            joined = gpd.sjoin(gdf_missing, npu_shp[[name_col, "geometry"]], how="left")
            joined = joined.drop(columns=["index_right"], errors="ignore").groupby(joined.index).first()
            filled_idx = joined[joined[name_col].notna()].index
            df.loc[filled_idx, "npu"] = joined.loc[filled_idx, name_col]

    if APD_ZONE_SHP.exists():
        zones = gpd.read_file(APD_ZONE_SHP).to_crs("EPSG:4326")
        name_col = next((c for c in ["ZONE"] if c in zones.columns), zones.columns[0])
        missing = df["zone_int"].isna()
        if missing.sum() > 0:
            gdf_missing = gdf.loc[df[missing].index]
            joined = gpd.sjoin(gdf_missing, zones[[name_col, "geometry"]], how="left")
            joined = joined.drop(columns=["index_right"], errors="ignore").groupby(joined.index).first()
            zone_vals = pd.to_numeric(joined[name_col].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
            filled_idx = zone_vals[zone_vals.notna()].index
            df.loc[filled_idx, "zone_int"] = zone_vals.loc[filled_idx]

    if NEIGHBORHOOD_SHP.exists():
        neigh = gpd.read_file(NEIGHBORHOOD_SHP).to_crs("EPSG:4326")
        missing = df["neighborhood"].isna() | (df["neighborhood"] == "")
        if missing.sum() > 0:
            gdf_missing = gdf.loc[df[missing].index]
            joined = gpd.sjoin(gdf_missing, neigh[["NAME", "geometry"]], how="left")
            joined = joined.drop(columns=["index_right"], errors="ignore").groupby(joined.index).first()
            filled_idx = joined["NAME"].notna()
            df.loc[filled_idx, "neighborhood"] = joined.loc[filled_idx, "NAME"].str.lower()

    console.print("[green]Spatial repair complete.")
    return df