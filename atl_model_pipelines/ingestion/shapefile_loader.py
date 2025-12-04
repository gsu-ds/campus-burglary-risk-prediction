# Loading Shapefiles

from pathlib import Path
import geopandas as gpd
from rich.console import Console
from config import (
    NPU_SHP, APD_ZONE_SHP, 
     CAMPUS_SHP, NEIGHBORHOOD_SHP, CITIES_SHP
)
console = Console()


def load_shapefile(path: Path, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Load shapefile + enforce CRS."""
    if not path.exists():
        raise FileNotFoundError(f"Shapefile not found: {path}")

    console.print(f"[cyan]Loading shapefile:[/cyan] {path.name}")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    return gdf.to_crs(target_crs)


def load_all_shapefiles():
    """Convenience loader if you need everything at once."""
    return {
        "npu": load_shapefile(NPU_SHP),
        "zones": load_shapefile(APD_ZONE_SHP),
        "campuses": load_shapefile(CAMPUS_SHP),
        "neighborhood": load_shapefile(NEIGHBORHOOD_SHP),
        "cities": load_shapefile(CITIES_SHP),
    }


__all__ = [
    "load_shapefile",
    "load_all_shapefiles",
]
