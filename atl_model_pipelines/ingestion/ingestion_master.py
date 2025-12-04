# pipeline/ingestion/ingestion_master.py
from pathlib import Path
from rich.console import Console

from .combine_apd_files import combine_and_deduplicate
from .weather_fetcher import fetch_or_load_weather
from .shapefile_loader import load_all_shapefiles
from validation.checks import run_validation_checks

from config import RAW_APD_DIR, INTERIM_DIR

console = Console()

def run_ingestion():
    console.print("\n[bold cyan]=== INGESTION PIPELINE START ===[/bold cyan]\n")

    # 1. Load all APD raw CSVs
    input_files = list(RAW_APD_DIR.glob("*.csv"))
    if not input_files:
        raise FileNotFoundError(f"No CSVs found in {RAW_APD_DIR}")

    df_apd = combine_and_deduplicate(input_files, dedupe_key="IncidentNumber")
    run_validation_checks(df_apd, "Ingestion → After APD combine")

    # 2. Weather ingestion (cached or fetched)
    hourly_df, daily_df = fetch_or_load_weather()

    # 3. Load shapefiles (NPU, Zones, Campus, Neighborhoods, Cities)
    shapefiles = load_all_shapefiles()

    console.print("\n[green]✓ Ingestion completed successfully.[/green]\n")

    return {
        "apd": df_apd,
        "weather_hourly": hourly_df,
        "weather_daily": daily_df,
        "shapefiles": shapefiles,
    }
