# %% [markdown]
# ### NOTEBOOK 1: APD DATA PIPELINE - STANDARDIZE, COMBINE, FEATURE ENGINEERING & EDA
# 
# Save as: `01_wrangler.ipynb`

# %%
# === Imports =================================================================
# Required libraries:
# pip install numpy pandas geopandas matplotlib seaborn contextily python-dateutil holidays
# pip install shapely openmeteo-requests requests-cache retry-requests rich plotly

# --- Core Python / Utilities ---
import re
from math import radians, sin, cos, sqrt, asin, atan2
from pathlib import Path
from typing import List, Optional, Dict, Any

# --- Data Handling ---
import numpy as np
import pandas as pd
import geopandas as gpd # Added import needed for notebook consistency

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import contextily as ctx

# --- Date/Time Utilities ---
from dateutil import parser
import holidays

# --- Geospatial Processing ---
from shapely.geometry import Point

# --- Weather API / Networking ---
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- Rich Console Debugging / Output ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# --- Spatial Density ---
from sklearn.neighbors import KernelDensity

# Initialize console
console = Console()

console.print(
    Panel.fit(
        "Libraries imported successfully.\n\n"
        "Groups loaded:\n"
        "- Core Python: regex, math, typing, paths\n"
        "- Data Handling: pandas, numpy, geopandas\n"
        "- Visualization: matplotlib, seaborn, plotly, contextily\n"
        "- Date & Time: dateutil, holidays\n"
        "- Geospatial: geopandas, shapely\n"
        "- Weather / Networking: open-meteo, caching, retry\n"
        "- Rich Output: console, tables, panels",
        title="Import Summary",
        border_style="cyan",
    )
)

# %% [markdown]
# #### APD Crime Data Processing Pipeline
# 
# 1. Configuration and logger
# 2. Ingest and standardize raw CSVs
# 3. Combine and deduplicate
# 4. Clean and basic feature engineering
# 5. Geospatial enrichment
# 6. Date/context features
# 7. Weather enrichment
# 8. Campus distance/label features
# 9. Final cleaning, export, and EDA

# %% [markdown]
# #### Section 1: Configuration

# %%
# Pipeline Logging Setup

pipeline_log: List[Dict[str, Any]] = []

# FIX: Stabilized log_step for ValueError: Cannot specify ',' with 's'
def log_step(step_name: str, df: pd.DataFrame) -> None:
    """Record a pipeline step with shape info, safely handling N/A."""
    
    # 1. Determine values and set format type
    if not isinstance(df, pd.DataFrame) or df.empty:
        rows_val = "N/A"
        cols_val = "N/A"
        rows_str = rows_val
        cols_str = cols_val
    else:
        rows_val = int(df.shape[0])
        cols_val = int(df.shape[1])
        # Apply comma formatting only when values are integers
        rows_str = f"{rows_val:,}"
        cols_str = str(cols_val)

    # 2. Append the original raw values to the log (int or string 'N/A')
    pipeline_log.append({"step": step_name, "rows": rows_val, "cols": cols_val})
    
    # 3. Print the result using the safely formatted strings
    console.print(f"[green]✓ {step_name}[/green] [cyan]→ shape: {rows_str} x {cols_str}[/cyan]")


def show_pipeline_table() -> None:
    """Display a Rich table summarizing all pipeline steps, safely formatting N/A values."""
    if not pipeline_log:
        console.print("[red]No pipeline steps logged yet.[/red]")
        return

    table = Table(title="📊 Data Pipeline Summary", show_lines=True)
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Rows", style="green")
    table.add_column("Cols", style="yellow")

    for entry in pipeline_log:
        # Determine values for printing
        rows_val = entry['rows']
        cols_val = entry['cols']
        
        # Safely format: use comma only for integers, use string otherwise
        rows_str = f"{rows_val:,}" if isinstance(rows_val, int) else str(rows_val)
        cols_str = str(cols_val) if isinstance(cols_val, int) else str(cols_val)
        
        table.add_row(entry["step"], rows_str, cols_str)
    
    console.print(table)


console.print(Panel("[bold green]Pipeline logger configured.[/bold green]", border_style="green"))

# %%
# Configure Data Paths

DATA_DIR = Path("../data")
RAW_DATA_FOLDER = DATA_DIR / "raw" / "apd"
INTERIM_DATA_FOLDER = DATA_DIR / "interim" / "apd"
PROCESSED_DATA_FOLDER = DATA_DIR / "processed" / "apd"
EXTERNAL_DATA_FOLDER = DATA_DIR / "external"
SHAPEFILES_DIR = DATA_DIR / "raw" / "shapefiles"

# Create necessary folders
EXTERNAL_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
INTERIM_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# Weather CSVs (initial run; later we fetch full 2021–2025)
HOURLY_WEATHER_PATH = EXTERNAL_DATA_FOLDER / "atlanta_hourly_weather_2024.csv"
DAILY_WEATHER_PATH = EXTERNAL_DATA_FOLDER / "atlanta_daily_weather_2024.csv"

# Shapefiles
CITIES_SHP = SHAPEFILES_DIR / "census_boundary_2024_sf" / "ga_census_places_2024.shp"
CAMPUS_SHP = SHAPEFILES_DIR / "area_landmark_2024_sf" / "ga_census_landmarks_2023.shp"
NEIGHBORHOOD_SHP = SHAPEFILES_DIR / "atl_neighborhood_sf" / "atl_neighborhoods.shp"
NPU_SHP = SHAPEFILES_DIR / "atl_npu_sf" / "atl_npu_boundaries.shp"
APD_ZONE_SHP = SHAPEFILES_DIR / "apd_zone_2019_sf" / "apd_police_zones_2019.shp"

# School center coordinates for distance-based enrichment
SCHOOL_CENTERS = {
    "GSU": (33.7530, -84.3863),
    "GA_Tech": (33.7756, -84.3963),
    "Emory": (33.7925, -84.3239),
    "Clark": (33.7533, -84.4124),
    "Spelman": (33.7460, -84.4129),
    "Morehouse": (33.7483, -84.4126),
    "Morehouse_Med": (33.7505, -84.4131),
    "Atlanta_Metro": (33.7145, -84.4020),
    "Atlanta_Tech": (33.7126, -84.4034),
    "SCAD": (33.7997, -84.3920),
    "John_Marshall": (33.7621, -84.3896),
}

console.print(
    Panel.fit(
        "[bold cyan]Data paths configured.[/bold cyan]\n\n"
        f"Raw Data: [yellow]{RAW_DATA_FOLDER}[/yellow]\n"
        f"Interim Data: [yellow]{INTERIM_DATA_FOLDER}[/yellow]\n"
        f"External Data: [yellow]{EXTERNAL_DATA_FOLDER}[/yellow]\n"
        f"Processed Data: [yellow]{PROCESSED_DATA_FOLDER}[/yellow]\n"
        f"Shapefiles: [yellow]{SHAPEFILES_DIR}[/yellow]",
        title="[bold cyan]Data Paths[/bold cyan]",
        border_style="cyan",
    )
)

log_step("Step 1: Logger, paths, settings, and constants configured", pd.DataFrame())

# %%
# Initial weather fetch (pre-fetch for initial merge)
def fetch_atlanta_weather(
    start_date="2021-01-01", end_date="2024-12-31", lat=33.749, lon=-84.388
):
    """Fetch weather data from Open-Meteo and save hourly/daily CSVs for reference."""
    console.print(
        Panel(
            "[bold cyan]Fetching initial weather data[/bold cyan]\n"
            f"Location: ({lat:.3f} N, {lon:.3f} W)\n"
            f"Date Range: {start_date} to {end_date}",
            border_style="cyan",
        )
    )

    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "rain",
            "apparent_temperature",
            "weather_code",
            "is_day",
        ],
        "daily": [
            "sunrise",
            "daylight_duration",
            "sunshine_duration",
            "precipitation_hours",
            "rain_sum",
            "temperature_2m_mean",
            "weather_code",
        ],
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Hourly
    hourly = response.Hourly()
    hourly_df = pd.DataFrame(
        {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temp_f": hourly.Variables(0).ValuesAsNumpy(),
            "precip_in": hourly.Variables(1).ValuesAsNumpy(),
            "rain_in": hourly.Variables(2).ValuesAsNumpy(),
            "apparent_temp_f": hourly.Variables(3).ValuesAsNumpy(),
            "weather_code": hourly.Variables(4).ValuesAsNumpy(),
            "is_daylight": hourly.Variables(5).ValuesAsNumpy().astype(int),
        }
    )
    hourly_df["datetime"] = (
        hourly_df["datetime"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    )

    # Daily
    daily = response.Daily()
    daily_df = pd.DataFrame(
        {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            ),
            "sunrise": daily.Variables(0).ValuesInt64AsNumpy(),
            "daylight_duration_sec": daily.Variables(1).ValuesAsNumpy(),
            "sunshine_duration_sec": daily.Variables(2).ValuesAsNumpy(),
            "precip_hours": daily.Variables(3).ValuesAsNumpy(),
            "rain_sum_in": daily.Variables(4).ValuesAsNumpy(),
            "temp_mean_f": daily.Variables(5).ValuesAsNumpy(),
            "weather_code": daily.Variables(6).ValuesAsNumpy(),
        }
    )
    daily_df["date"] = (
        daily_df["date"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.date
    )

    hourly_df.to_csv(HOURLY_WEATHER_PATH, index=False)
    daily_df.to_csv(DAILY_WEATHER_PATH, index=False)

    console.print(f"[green]✓ Saved hourly:[/green] {HOURLY_WEATHER_PATH}")
    console.print(f"[green]✓ Saved daily:[/green] {DAILY_WEATHER_PATH}")
    console.print(
        f"[green]✓ Total rows: {len(hourly_df):,} hourly, {len(daily_df):,} daily[/green]"
    )

    return hourly_df, daily_df


fetch_atlanta_weather()
log_step("Step 2: Initial Atlanta weather (2021–2024) saved to data/external", pd.DataFrame())

# %% [markdown]
# #### Section 2: Standardize & Combine

# %%
def standardize_column_name(col: str) -> str:
    """Convert column name to snake_case format."""
    col = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", col)
    col = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
    col = col.lower()
    col = re.sub(r"[\s\-\.\,\(\)\[\]\{\}]+", "_", col)
    col = re.sub(r"[^\w]", "", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def combine_and_deduplicate(files: List[Path], dedupe_key: str) -> pd.DataFrame:
    """Combine multiple CSVs, standardize columns, and drop duplicates."""
    console.print("\n[bold cyan]═══ Combining data files ═══[/bold cyan]\n")

    dfs = []
    for filepath in files:
        console.print(f"[cyan]Reading and standardizing:[/cyan] {filepath.name}")
        df = pd.read_csv(filepath)
        df.columns = [standardize_column_name(c) for c in df.columns]
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)
    total_rows = len(df_combined)

    # New logger step: Ingest Raw Data
    log_step("Ingest Raw Data (standardized columns, pre-dedup)", df_combined)

    if dedupe_key not in df_combined.columns:
        raise KeyError(
            f"dedupe_key='{dedupe_key}' not found in columns: {df_combined.columns.tolist()}"
        )

    df_dedup = df_combined.drop_duplicates(subset=[dedupe_key])
    duplicates = total_rows - len(df_dedup)

    console.print(f"[yellow]Combined:[/yellow] {total_rows:,} rows")
    console.print(f"[red]Removed:[/red] {duplicates:,} duplicate rows")

    log_step("Step 3: Standardize & Combine, deduplicated by incident_number", df_dedup)
    return df_dedup

# %%
# Function Execution: Combine Raw Data CSVs

input_files = list(RAW_DATA_FOLDER.glob("*.csv"))

if not input_files:
    console.print(f"[bold red]Warning:[/bold red] No CSV files found in '{RAW_DATA_FOLDER}'")
    df_combined = pd.DataFrame()
else:
    console.print(f"[bold cyan]Found {len(input_files)} CSV file(s).[/bold cyan]")
    df_combined = combine_and_deduplicate(
        files=input_files, dedupe_key="incident_number"
    )

# %% [markdown]
# #### Section 3: Clean

# %%
def clean_apd_data(df: pd.DataFrame) -> pd.DataFrame:
    """Column drops, normalization, and one-hot encoding."""
    console.print("\n[bold cyan]═══ Cleaning APD data ═══[/bold cyan]\n")

    df = df.copy()

    # 1. Define and drop unnecessary columns
    columns_to_drop = [
        "report_number",
        "zone",
        "fire_arm_involved",
        "object_id",
        "occurred_from_date",
        "occurred_to_date",
        "part",
        "vic_count",
        "is_bias_motivation_involved",
        "x",
        "y",
        "beat_text",
    ]

    existing_drops = [col for col in columns_to_drop if col in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
        console.print(f"[yellow]Dropped {len(existing_drops)} columns:[/yellow] {', '.join(existing_drops)}")

    # 2. Rename columns
    if "objectid" in df.columns:
        df = df.rename(columns={"objectid": "object_id"})
    
    # 3. One-hot encode 'event_watch'
    if "event_watch" in df.columns:
        console.print("[cyan]One-hot encoding:[/cyan] 'event_watch'")
        one_hot = pd.get_dummies(df["event_watch"], prefix="event_watch", dummy_na=False)
        one_hot.columns = [standardize_column_name(col) for col in one_hot.columns]
        df = pd.concat([df.drop(columns=["event_watch"]), one_hot], axis=1)
        console.print(f"[green]✓ Created {len(one_hot.columns)} standardized one-hot columns[/green]")

    # 4. Convert text to lowercase
    text_columns = ["location_type", "street_address", "nibrs_offense"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    log_step("Step 4: Clean and drop selected columns", df)
    return df


df_clean = clean_apd_data(df_combined)

# %%
# Date Standardization

console.print("\n[bold cyan]═══ Standardizing 'report_date' column with robust parser ═══[/bold cyan]\n")
total_rows = len(df_clean)

# Preserve raw values for comparison
df_clean["_raw_report_date"] = df_clean["report_date"].astype(str).str.strip()


def parse_report_date(x):
    """Safely parse mixed APD/NIBRS date formats."""
    if pd.isna(x):
        return pd.NaT

    x = str(x).strip()

    # Explicit patterns: MM/DD/YYYY HH:MM:SS AM/PM, etc.
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [APMapm]{2}$", x):
        return pd.to_datetime(x, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} [APMapm]{2}$", x):
        return pd.to_datetime(x, format="%m/%d/%Y %I:%M %p", errors="coerce")
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$", x):
        return pd.to_datetime(x, format="%m/%d/%Y %H:%M", errors="coerce")

    # Fallback
    try:
        return parser.parse(x, fuzzy=True)
    except Exception:
        return pd.NaT


df_clean["report_date"] = df_clean["_raw_report_date"].apply(parse_report_date)
invalid = df_clean["report_date"].isna().sum()
parsed = total_rows - invalid

console.print(f"[cyan]Total rows:[/cyan] {total_rows:,}")
console.print(f"[green]Successfully standardized:[/green] {parsed:,}")
console.print(f"[yellow]Unrecoverable dates dropped:[/yellow] {invalid:,}")

df_clean = df_clean.dropna(subset=["report_date"]).copy()
df_clean["report_date"] = pd.to_datetime(df_clean["report_date"])

log_step("Step 5: Robust date standardization", df_clean)

console.print("\n[bold blue]Examples of corrected formats:[/bold blue]")
changed = df_clean[
    df_clean["_raw_report_date"] != df_clean["report_date"].astype(str)
]
if len(changed) > 0:
    console.print(
        changed.sample(min(5, len(changed)))[["_raw_report_date", "report_date"]]
    )
else:
    console.print("[yellow]All dates already standardized.[/yellow]")

# %% [markdown]
# #### Section 4: Geospatial Enrichment (NPU, Zone, Campus Footprints, City, Neighborhood)

# %%
# Geospatial Helper Functions (NPU STANDARDIZED TO 'npu')

def haversine_distance(lon1, lat1, lon2, lat2):
    """Great-circle distance between two points in miles (R=3956)."""
    R = 3956
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return c * R


def to_gdf(df: pd.DataFrame, lon_col: str = "longitude", lat_col: str = "latitude"):
    """Convert a DataFrame with lon/lat into a GeoDataFrame in EPSG:4326."""
    for c in (lon_col, lat_col):
        if c not in df.columns:
            raise KeyError(f"Expected coordinate column '{c}' not found.")
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )


def load_shapefile(path: Path, target_crs) -> gpd.GeoDataFrame:
    """Load shapefile and ensure CRS alignment."""
    try:
        gdf = gpd.read_file(path)
    except Exception:
        console.print(f"[bold red]FATAL: Could not load shapefile:[/bold red] {path}")
        raise

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(target_crs)


def attach_campus(gdf: gpd.GeoDataFrame, campus_shp: Path) -> gpd.GeoDataFrame:
    """Attach closest campus footprint label where available (MTFCC S1400)."""
    console.print("[cyan]Attaching campus footprint labels...[/cyan]")
    campus = load_shapefile(campus_shp, gdf.crs)
    if "MTFCC" in campus.columns:
        campus = campus[campus["MTFCC"] == "S1400"]
    name_col = "FULLNAME" if "FULLNAME" in campus.columns else campus.columns[0]
    campus = campus[[name_col, "geometry"]].rename(columns={name_col: "campus_label"})

    gdf = gpd.sjoin_nearest(gdf, campus, how="left")
    gdf = gdf.drop(columns=["index_right"], errors="ignore")
    return gdf


def attach_neighborhood(
    gdf: gpd.GeoDataFrame, nhood_shp: Path
) -> gpd.GeoDataFrame:
    """Attach neighborhood labels (within polygon)."""
    console.print("[cyan]Attaching neighborhood labels...[/cyan]")
    nhoods = load_shapefile(nhood_shp, gdf.crs)
    name_col = "NAME" if "NAME" in nhoods.columns else nhoods.columns[0]
    nhoods = nhoods[[name_col, "geometry"]].rename(
        columns={name_col: "neighborhood_label"}
    )

    gdf = gpd.sjoin(gdf, nhoods, how="left", predicate="within")
    gdf = gdf.drop(columns=["index_right"], errors="ignore")
    return gdf


def attach_city(gdf: gpd.GeoDataFrame, cities_shp: Path) -> gpd.GeoDataFrame:
    """Attach city labels."""
    console.print("[cyan]Attaching city labels...[/cyan]")
    cities = load_shapefile(cities_shp, gdf.crs)
    name_col = "NAME" if "NAME" in cities.columns else cities.columns[0]
    cities = cities[[name_col, "geometry"]].rename(columns={name_col: "city_label"})

    gdf = gpd.sjoin(gdf, cities, how="left", predicate="within")
    gdf = gdf.drop(columns=["index_right"], errors="ignore")
    return gdf


# FIX: Standardize NPU column output name to 'npu'
def attach_npu(gdf: gpd.GeoDataFrame, npu_shp: Path) -> gpd.GeoDataFrame:
    """Attach NPU labels using nearest polygon, naming the output column 'npu'."""
    console.print("[cyan]Attaching NPU labels (output column: 'npu')...[/cyan]")
    npu_gdf = load_shapefile(npu_shp, gdf.crs)

    candidate_cols = ["NPU", "npu", "NPU_ID", "NPU_NUM", "NPU_NAME", "NAME"]
    name_col = None
    for c in candidate_cols:
        if c in npu_gdf.columns:
            name_col = c
            break
    if name_col is None:
        name_col = npu_gdf.columns[0]

    # Rename the source NPU column to the desired final column name ('npu')
    npu_gdf = npu_gdf[[name_col, "geometry"]].rename(columns={name_col: "npu"})

    gdf = gpd.sjoin_nearest(gdf, npu_gdf, how="left")
    gdf = gdf.drop(columns=["index_right"], errors="ignore")
    return gdf


def attach_apd_zone(gdf: gpd.GeoDataFrame, zone_shp: Path) -> gpd.GeoDataFrame:
    """Attach APD Zone numbers by predicate 'within'."""
    console.print("[cyan]Attaching APD zones...[/cyan]")
    zones = load_shapefile(zone_shp, gdf.crs)

    zone_col = None
    for c in ["ZONE", "zone", "Zone"]:
        if c in zones.columns:
            zone_col = c
            break
    if zone_col is None:
        zone_col = zones.columns[0]

    zones = zones[[zone_col, "geometry"]].rename(columns={zone_col: "zone_raw"})

    gdf = gpd.sjoin(gdf, zones, how="left", predicate="within")
    gdf = gdf.drop(columns=["index_right"], errors="ignore")

    gdf["zone_int"] = pd.to_numeric(
        gdf["zone_raw"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    return gdf


# FIX: Updated to use 'npu' as the grouping column
def impute_missing_zone_from_npu(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing zone_int using the modal zone_int within its NPU."""
    if "zone_int" not in df.columns or "npu" not in df.columns:
        return df

    missing_before = df["zone_int"].isna().sum()
    if missing_before == 0:
        return df

    mapping = (
        df.dropna(subset=["zone_int"])
        .groupby("npu")["zone_int"]
        .apply(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    )

    df["zone_int_cleaned"] = df["zone_int"].fillna(df["npu"].map(mapping))
    missing_after = df["zone_int_cleaned"].isna().sum()

    df = df.drop(columns=["zone_int"]).rename(columns={"zone_int_cleaned": "zone_int"})
    console.print(
        f"[green]✓ Imputed {missing_before - missing_after} zone_int values via NPU mode.[/green]"
    )
    return df


def build_location_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create location_label with priority campus > neighborhood > city."""
    conditions = [
        df.get("campus_label").notna() if "campus_label" in df.columns else False,
        df.get("neighborhood_label").notna()
        if "neighborhood_label" in df.columns
        else False,
        df.get("city_label").notna() if "city_label" in df.columns else False,
    ]
    choices = [
        df.get("campus_label"),
        df.get("neighborhood_label"),
        df.get("city_label"),
    ]
    df["location_label"] = np.select(conditions, choices, default="Other")
    return df


def enrich_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """Spatial enrichment pipeline."""
    console.print("\n[bold cyan]═══ Spatial enrichment ═══[/bold cyan]\n")

    initial_count = len(df)
    df_temp = df.dropna(subset=["longitude", "latitude"]).copy()
    dropped_count = initial_count - len(df_temp)
    if dropped_count > 0:
        console.print(f"[yellow]Dropped {dropped_count} rows missing coordinates.[/yellow]")

    gdf = to_gdf(df_temp, lon_col="longitude", lat_col="latitude")

    TARGET_PCS = "EPSG:3857"
    gdf = gdf.to_crs(TARGET_PCS)
    console.print(f"[green]✓ Data projected to {TARGET_PCS} for joins.[/green]")

    # NPU is attached as 'npu'
    gdf = attach_npu(gdf, NPU_SHP)
    gdf = attach_apd_zone(gdf, APD_ZONE_SHP)
    gdf = attach_campus(gdf, CAMPUS_SHP)
    gdf = attach_neighborhood(gdf, NEIGHBORHOOD_SHP)
    gdf = attach_city(gdf, CITIES_SHP)

    df_enriched = pd.DataFrame(gdf.drop(columns=["geometry", "zone_raw"], errors="ignore"))
    df_enriched = impute_missing_zone_from_npu(df_enriched)
    df_enriched = build_location_label(df_enriched)

    log_step("Step 6: Spatial enrichment (NPU, zone, campus footprints, city, neighborhood)", df_enriched)
    return df_enriched


df_spatial = enrich_spatial(df_clean)

# %%
# Fix raw date column name if present
if "_raw_report_date" in df_spatial.columns:
    df_spatial = df_spatial.rename(columns={"_raw_report_date": "raw_report_date"})
    log_step("Step 7: Rename raw date column", df_spatial)
    console.print("[green]✓ Renamed '_raw_report_date' to 'raw_report_date'.[/green]")
else:
    console.print("[yellow]No raw date column to rename.[/yellow]")

# %% [markdown]
# #### Section 5: Weather + Date/Context Features

# %%
def add_holiday_flag(
    df: pd.DataFrame,
    date_col: str = "report_date",
    country: str = "US",
    subdiv: str = "GA",
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add 'is_holiday' boolean column."""
    df = df.copy()
    if years is None:
        years = sorted(df[date_col].dt.year.unique().tolist())

    console.print(f"[cyan]Generating holidays for {country}-{subdiv}, years={years}[/cyan]")
    holiday_dates = holidays.country_holidays(country=country, subdiv=subdiv, years=years)
    holiday_set = set(holiday_dates.keys())

    df["_date_only"] = df[date_col].dt.date
    df["is_holiday"] = df["_date_only"].isin(holiday_set)
    df = df.drop(columns=["_date_only"])
    console.print("[green]✓ 'is_holiday' column created.[/green]")
    return df


# %%
# Feature Engineering: Date & Contextual Features
def engineer_date_context_features(
    df: pd.DataFrame, date_col: str = "report_date"
) -> pd.DataFrame:
    """Create core temporal + contextual features. Standardized to 6 x 4-hour bins."""
    df = df.copy()
    console.print("\n[bold cyan]═══ Engineering date and contextual features (Standardizing to 6 Bins) ═══[/bold cyan]\n")

    df[date_col] = pd.to_datetime(df[date_col])
    dt = df[date_col].dt

    # Core temporal features
    df["incident_datetime"] = df[date_col]
    df["incident_date"] = dt.date
    df["incident_hour"] = dt.hour
    df["incident_datetime_hour"] = dt.to_period("h").astype(str)
    df["day_of_week"] = dt.day_name()
    df["day_number"] = dt.weekday + 1  # Monday=1
    df["year"] = dt.year
    df["day_of_year"] = dt.dayofyear
    df["month"] = dt.month
    df["week_number"] = dt.isocalendar().week.astype(int)

    df = add_holiday_flag(df, date_col=date_col)

    if "nibrs_ucr_code" in df.columns:
        df["nibrs_code"] = df["nibrs_ucr_code"]

    df["offense_category"] = np.select(
        [
            df["nibrs_offense"].str.contains("burglary|robbery", case=False, na=False),
            df["nibrs_offense"].str.contains(
                "motor vehicle theft", case=False, na=False
            ),
            df["nibrs_offense"].str.contains(
                "theft|larceny|shoplift|fraud|swindle|embezzelment|stolen property|false pretenses",
                case=False,
                na=False,
            ),
            df["nibrs_offense"].str.contains(
                "assault|murder|rape|battery|intimidation|extortion|kidnapping",
                case=False,
                na=False,
            ),
        ],
        [
            "Burglary/Robbery",
            "Motor Vehicle Theft",
            "Theft/Larceny/Fraud",
            "Violent Crime",
        ],
        default="Other/Misc.",
    )

    month = dt.month
    df["semester"] = np.select(
        [month.isin([8, 9, 10, 11, 12]), month.isin([1, 2, 3, 4, 5])],
        ["Fall", "Spring"],
        default="Summer",
    )

    # STANDARDIZED TO SIX 4-HOUR BINS (0-4, 5-8, ..., 21-24)
    df['hour'] = df['incident_datetime'].dt.hour
    bins = [0, 5, 9, 13, 17, 21, 25]
    labels = [
        "Early Night (0–4)",
        "Early Morning (5–8)",
        "Late Morning (9–12)",
        "Afternoon (13–16)",
        "Evening (17–20)",
        "Late Night (21–24)",
    ]
    # This is the single, consistent binning column
    df["hour_block"] = pd.cut(df["hour"], bins=bins, labels=labels, right=False, include_lowest=True)
    
    df["is_weekend"] = dt.weekday >= 5

    df["loc_acc"] = np.where(
        df["latitude"].isna() | df["longitude"].isna(), 1, 0
    )

    log_step("Step 8: Date and context features (Standardized to 6-bin hour_block)", df)
    return df

df_features = engineer_date_context_features(df_spatial)

# %%
def merge_weather_data_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge hourly and daily weather into APD data using the initial external CSVs.
    """
    df = df.copy()
    console.print("\n[bold cyan]═══ Merging initial weather data (external CSVs) ═══[/bold cyan]\n")

    try:
        hourly_df = pd.read_csv(HOURLY_WEATHER_PATH)
        daily_df = pd.read_csv(DAILY_WEATHER_PATH)
    except FileNotFoundError:
        console.print("[bold red]Warning:[/bold red] Initial weather CSVs not found. Skipping this merge.")
        return df

    df["report_date_dt"] = pd.to_datetime(df["report_date"])
    hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    # Drop potential existing weather columns before merge
    weather_cols_to_drop = [col for col in df.columns if any(c in col for c in hourly_df.columns if c != 'datetime') or any(c in col for c in daily_df.columns if c != 'date')]
    df = df.drop(columns=weather_cols_to_drop, errors='ignore')

    df = df.merge(
        hourly_df,
        left_on=df["report_date_dt"].dt.floor("H"),
        right_on=hourly_df["datetime"].dt.floor("H"),
        how="left",
    )
    df = df.drop(columns=["key_0"], errors="ignore")

    df["date_only"] = df["report_date_dt"].dt.date
    daily_df["date_only"] = daily_df["date"].dt.date

    df = df.merge(
        daily_df,
        on="date_only",
        how="left",
        suffixes=("_hourly", "_daily"),
    )

    df = df.drop(columns=["report_date_dt", "date_only", "datetime", "date"], errors="ignore")
    console.print("[green]✓ Initial hourly + daily weather merged (2021-2024).[/green]")
    return df


df_features = merge_weather_data_basic(df_features)

# %%
def derive_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create boolean flags is_raining, is_hot, is_cold based on merged columns."""
    df = df.copy()
    console.print("\n[bold cyan]═══ Deriving basic weather flags ═══[/bold cyan]\n")

    temp_col = "apparent_temp_f"
    precip_col = "precip_in"

    if temp_col not in df.columns or precip_col not in df.columns:
        console.print(
            "[yellow]Weather columns missing or incomplete; skipping flag derivation.[/yellow]"
        )
        return df

    df["is_raining"] = (df[precip_col].fillna(0) > 0.01).astype(int)
    df["is_hot"] = (df[temp_col].fillna(-999) >= 90).astype(int)
    df["is_cold"] = (df[temp_col].fillna(999) <= 40).astype(int)

    log_step("Step 9: Basic weather flags", df)
    console.print("[green]✓ Weather flags (hot/cold/raining) created.[/green]")
    return df


df_features = derive_weather_flags(df_features)

# %% [markdown]
# #### Section 6: Filter Target Offenses & Interim Export

# %%
console.print("\n[bold cyan]═══ Final filtering for target offenses and interim export ═══[/bold cyan]\n")

TARGET_OFFENSES = [
    "larceny",
    "theft",
    "robbery",
    "burglary",
    "prowling",
    "shoplifting",
    "fraud",
    "swindle",
    "embezzelment",
    "credit card",
    "wire fraud",
    "impersonation",
]

if "nibrs_offense" not in df_features.columns:
    raise KeyError("'nibrs_offense' column not found; cannot filter TARGET_OFFENSES.")

mask = df_features["nibrs_offense"].str.contains(
    "|".join(TARGET_OFFENSES), case=False, na=False
)
df_model = df_features[mask].copy()

if "raw_report_date" in df_model.columns:
    df_model = df_model.drop(columns=["raw_report_date"])
    console.print("[yellow]Dropped debug column 'raw_report_date' prior to export.[/yellow]")

log_step("Step 10: Filter for target crimes", df_model)
console.print(
    f"[bold yellow]Filtered for modeling:[/bold yellow] {len(df_model):,} rows match target offenses."
)

df_final = df_model.sort_values("report_date", ascending=True, ignore_index=True)

final_output = INTERIM_DATA_FOLDER / "apd_model_data_target_crimes.csv"
df_final.to_csv(final_output, index=False)

log_step("Interim Data Export", df_final)
console.print("[bold green]✓ Interim dataset exported.[/bold green]")
console.print(
    f"[bold]Final interim dataset:[/bold] {len(df_final):,} rows x {df_final.shape[1]} columns"
)
console.print(f"[bold]Saved to:[/bold] {final_output}")

# %% [markdown]
# #### Section 7: Additional Spatial/Weather Repair (Load Interim)

# %%
console.print(
    Panel.fit(
        "[bold cyan]STEP 11: Load interim data for final spatial and weather repair[/bold cyan]\n\n"
        "Tasks:\n"
        "• Load interim CSV\n"
        "• Fill missing spatial data (NPU, district, neighborhood, city)\n"
        "• Refetch and merge complete 2021–2025 weather\n"
        "• Add campus-based features\n"
        "• Save to processed folder",
        border_style="cyan",
    )
)

INTERIM_PATH = INTERIM_DATA_FOLDER / "apd_model_data_target_crimes.csv"
df = pd.read_csv(INTERIM_PATH)

log_step("Step 11: Loaded interim data", df)

console.print("\n[yellow]Missing data before spatial cleaning:[/yellow]")
missing_cols_initial = [
    "zone_int", 
    "npu",      # Standardized NPU column
    "neighborhood",
    "neighborhood_label",
    "campus_label",
    "city_label",
]
missing_table = Table(show_header=True, header_style="bold magenta")
missing_table.add_column("Column", style="cyan")
missing_table.add_column("Missing", justify="right", style="red")
missing_table.add_column("%", justify="right", style="yellow")

for col in missing_cols_initial:
    if col in df.columns:
        missing = df[col].isna().sum()
        pct = (missing / len(df)) * 100
        missing_table.add_row(col, f"{missing:,}", f"{pct:.1f}%")

console.print(missing_table)

# %%
console.print(Panel("[bold cyan]STEP 12: Fill missing spatial data via spatial joins (Stabilized)[/bold cyan]", border_style="cyan"))


def show_missing_comparison(df_local, cols_snapshot, step_name):
    table = Table(
        title=f"{step_name} - Missing Data Comparison",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Column", style="cyan")
    table.add_column("Before", justify="right", style="red")
    table.add_column("Before %", justify="right", style="yellow")
    table.add_column("After", justify="right", style="green")
    table.add_column("After %", justify="right", style="blue")
    table.add_column("Filled", justify="right", style="white")

    for col, (before, before_pct) in cols_snapshot.items():
        if col in df_local.columns:
            after = df_local[col].isna().sum()
            after_pct = (after / len(df_local)) * 100
            filled = before - after
            table.add_row(
                col,
                f"{before:,}",
                f"{before_pct:.1f}%",
                f"{after:,}",
                f"{after_pct:.1f}%",
                f"{filled:,}",
            )

    console.print(table)


spatial_cols_before: Dict[str, Any] = {}
# FIX: Use standardized column names here
for col in ["zone_int", "npu", "neighborhood", "neighborhood_label", "city_label"]:
    if col in df.columns:
        missing = df[col].isna().sum()
        pct = (missing / len(df)) * 100
        spatial_cols_before[col] = (missing, pct)

# Create GeoDataFrame for ALL rows with valid coordinates
gdf = gpd.GeoDataFrame(
    df.dropna(subset=['longitude', 'latitude']).copy(),
    geometry=gpd.points_from_xy(df.longitude, df.latitude), 
    crs="EPSG:4326"
)
# Keep track of the indices we are working on (all non-null coordinate rows)
gdf_indices = gdf.index.intersection(df.index)

# --- Fill NPU ---
if NPU_SHP.exists() and "npu" in df.columns: # Targeted column is 'npu'
    console.print("\n[cyan]→ Filling NPU via spatial join (intersects)...[/cyan]")
    gdf_npu = gpd.read_file(NPU_SHP).to_crs("EPSG:4326")
    npu_col_shp = 'NAME' if 'NAME' in gdf_npu.columns else 'NPU' # Source column in shapefile
    npu_col = 'npu' # Target column in DataFrame
    
    # Identify rows that need NPU filling and have valid coordinates
    missing_mask = df[npu_col].isna()
    
    if missing_mask.sum() > 0:
        gdf_missing = gdf[gdf.index.isin(df[missing_mask].index)].copy()
        
        joined = gpd.sjoin(gdf_missing, gdf_npu[[npu_col_shp, 'geometry']], 
                           how="left", predicate="intersects").drop(columns=['index_right'])
        
        # Take the first match for each original index
        joined_clean = joined.groupby(joined.index).first()
        
        # Safely fill the missing values in the main DataFrame 'df'
        original_nans = df[npu_col].isna().sum()
        
        # Only fill if the join was successful (npu_col_shp value is not NaN)
        valid_joins = joined_clean[npu_col_shp].dropna().index
        
        df.loc[valid_joins, npu_col] = df.loc[valid_joins, npu_col].fillna(
            joined_clean.loc[valid_joins, npu_col_shp].values
        )
        
        filled = original_nans - df[npu_col].isna().sum()
        console.print(f"  [green]✓ Filled {filled:,} NPU values[/green]")


# --- Fill zone_int (District) ---
if APD_ZONE_SHP.exists() and "zone_int" in df.columns:
    console.print("\n[cyan]→ Filling Zone Integer (District) via spatial join (intersects)...[/cyan]")
    gdf_zones = gpd.read_file(APD_ZONE_SHP).to_crs("EPSG:4326")
    zone_col = 'ZONE' 
    
    missing_mask = df['zone_int'].isna()
    
    if missing_mask.sum() > 0:
        gdf_missing = gdf[gdf.index.isin(df[missing_mask].index)].copy()
        
        joined = gpd.sjoin(gdf_missing, gdf_zones[[zone_col, 'geometry']], 
                           how="left", predicate="intersects").drop(columns=['index_right'])
        joined_clean = joined.groupby(joined.index).first()
        
        original_nans = df['zone_int'].isna().sum()
        
        zone_int_filled = pd.to_numeric(
            joined_clean[zone_col].astype(str).str.extract(r"(\d+)", expand=False), 
            errors="coerce"
        )
        
        df.loc[zone_int_filled.index, 'zone_int'] = df.loc[zone_int_filled.index, 'zone_int'].fillna(zone_int_filled)
        
        filled = original_nans - df['zone_int'].isna().sum()
        console.print(f"  [green]✓ Filled {filled:,} Zone Int values[/green]")

# --- Fill neighborhood ---
if NEIGHBORHOOD_SHP.exists() and "neighborhood" in df.columns:
    console.print("\n[cyan]→ Filling Neighborhood via spatial join (intersects)...[/cyan]")
    gdf_neighborhoods = gpd.read_file(NEIGHBORHOOD_SHP).to_crs("EPSG:4326")
    name_col = 'NAME' 
    
    missing_mask = df['neighborhood'].isna()
    
    if missing_mask.sum() > 0:
        gdf_missing = gdf[gdf.index.isin(df[missing_mask].index)].copy()
        
        joined = gpd.sjoin(gdf_missing, gdf_neighborhoods[[name_col, 'geometry']], 
                           how="left", predicate="intersects").drop(columns=['index_right'])
        joined_clean = joined.groupby(joined.index).first()
        
        original_nans = df['neighborhood'].isna().sum()
        valid_joins = joined_clean[name_col].dropna().index
        
        df.loc[valid_joins, 'neighborhood'] = df.loc[valid_joins, 'neighborhood'].fillna(
            joined_clean.loc[valid_joins, name_col].str.lower().values
        )
        
        if 'neighborhood_label' in df.columns:
            df.loc[valid_joins, 'neighborhood_label'] = df.loc[valid_joins, 'neighborhood_label'].fillna(
                joined_clean.loc[valid_joins, name_col].str.upper().values
            )
        
        filled = original_nans - df['neighborhood'].isna().sum()
        console.print(f"  [green]✓ Filled {filled:,} Neighborhood values[/green]")

# --- Fill city_label ---
if CITIES_SHP.exists() and "city_label" in df.columns:
    console.print("\n[cyan]→ Filling City Label via spatial join (intersects)...[/cyan]")
    gdf_cities = gpd.read_file(CITIES_SHP).to_crs("EPSG:4326")
    city_col = 'NAME'
    
    missing_mask = df['city_label'].isna()
    
    if missing_mask.sum() > 0:
        gdf_missing = gdf[gdf.index.isin(df[missing_mask].index)].copy()
        
        joined = gpd.sjoin(gdf_missing, gdf_cities[[city_col, 'geometry']], 
                           how="left", predicate="intersects").drop(columns=['index_right'])
        joined_clean = joined.groupby(joined.index).first()
        
        original_nans = df['city_label'].isna().sum()
        valid_joins = joined_clean[city_col].dropna().index
        
        df.loc[valid_joins, 'city_label'] = df.loc[valid_joins, 'city_label'].fillna(
            joined_clean.loc[valid_joins, city_col].str.upper().values
        )
        
        filled = original_nans - df['city_label'].isna().sum()
        console.print(f"  [green]✓ Filled {filled:,} City Label values[/green]")


console.print("\n[green]✓ Spatial data filling complete.[/green]")
show_missing_comparison(df, spatial_cols_before, "Step 12")
log_step("Step 12: Filled missing spatial data via spatial joins", df)

# %% [markdown]
# #### Section 8: Campus Distance / Campus Code (Model-Friendly)

# %%
console.print(Panel("[bold cyan]STEP 13: Campus labels and distance features (Ensures campus_code is not missing)[/bold cyan]", border_style="cyan"))

campus_before: Dict[str, Any] = {}
for col in ["campus_label", "campus_distance_m", "campus_code"]:
    if col in df.columns:
        missing = df[col].isna().sum()
        pct = (missing / len(df)) * 100
        campus_before[col] = (missing, pct)

# 1.5-mile threshold in meters
DISTANCE_THRESHOLD_M = 2414.016
console.print(f"[cyan]Using 1.5-mile threshold ({DISTANCE_THRESHOLD_M:.0f} meters).[/cyan]")

# Haversine in meters
def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


CAMPUS_ENCODING = {
    "none": 0,
    "GSU": 1,
    "GA_Tech": 2,
    "Emory": 3,
    "Clark": 4,
    "Spelman": 5,
    "Morehouse": 6,
    "Morehouse_Med": 7,
    "Atlanta_Metro": 8,
    "Atlanta_Tech": 9,
    "SCAD": 10,
    "John_Marshall": 11,
}

console.print("[cyan]Calculating distance to each campus and nearest campus...[/cyan]")

# Calculate distance to nearest campus for every crime
nearest_campuses = []
nearest_distances = []

for idx, row in df.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    
    if pd.isna(lat) or pd.isna(lon):
        nearest_campuses.append("none")
        nearest_distances.append(np.nan)
        continue
        
    min_dist = float('inf')
    nearest = 'none'
    
    for campus, (c_lat, c_lon) in SCHOOL_CENTERS.items():
        dist = haversine_meters(lat, lon, c_lat, c_lon)
        if dist < min_dist:
            min_dist = dist
            nearest = campus
            
    if min_dist <= DISTISTANCE_THRESHOLD_M:
        nearest_campuses.append(nearest)
        nearest_distances.append(min_dist)
    else:
        # Distance is NaN if outside the proximity radius (expected/desired missingness)
        nearest_campuses.append("none")
        nearest_distances.append(np.nan) 

df["campus_label"] = nearest_campuses

# campus_distance_m is NaN if outside 1.5 miles (correct for a proximity metric)
df["campus_distance_m"] = nearest_distances
df["campus_distance_m"] = df["campus_distance_m"].round(4) # Round for consistency

# Numeric encoding (Fix: Ensures no missing values for campus_code)
df["campus_code"] = df["campus_label"].map(CAMPUS_ENCODING).fillna(0).astype(int)
df["campus_distance_m"] = df["campus_distance_m"].fillna(0)

# Binary flags for each campus
binary_cols_created = []
for campus in SCHOOL_CENTERS.keys():
    col_name = f"near_{campus.lower()}"
    df[col_name] = (df["campus_label"] == campus).astype(int)
    binary_cols_created.append(col_name)

console.print(
    f"[green]✓ Campus features created: {len(binary_cols_created)} near_* columns.[/green]"
)
console.print(f"[green]✓ **campus_code** column is complete (non-proximal crimes = 0).[/green]")

# Distribution (including 'none')
campus_table = Table(
    title="Campus Distribution (including 'none')",
    show_header=True,
    header_style="bold magenta",
)
campus_table.add_column("Campus", style="cyan")
campus_table.add_column("Count", justify="right", style="green")
campus_table.add_column("%", justify="right", style="yellow")
campus_table.add_column("Code", justify="right", style="white")

for campus, code in sorted(CAMPUS_ENCODING.items(), key=lambda x: x[1]):
    count = (df["campus_label"] == campus).sum()
    pct = (count / len(df)) * 100
    campus_table.add_row(campus, f"{count:,}", f"{pct:.1f}%", str(code))

console.print(campus_table)

show_missing_comparison(df, campus_before, "Step 13")
log_step("Step 13: Campus distance and campus_code features", df)

# %% [markdown]
# #### Section 9: Full 2021–2025 Weather Refresh

# %%
console.print(
    Panel("[bold cyan]STEP 14: Fetch and merge full weather dataset (2021–2025) - Fixes missing 2025 data[/bold cyan]", border_style="cyan")
)

weather_before: Dict[str, Any] = {}
weather_cols_check = [
    "temp_f",
    "precip_in",
    "rain_in",
    "apparent_temp_f",
    "weather_code_hourly",
    "is_daylight",
    "daylight_duration_sec",
    "sunshine_duration_sec",
]
for col in weather_cols_check:
    # Ensure checking against the column names created in the prior merge
    current_col = col if col in df.columns else (
        'weather_code_hrly' if col == 'weather_code_hourly' else col
    )
    if current_col in df.columns:
        missing = df[current_col].isna().sum()
        pct = (missing / len(df)) * 100
        weather_before[col] = (missing, pct)

df["report_date"] = pd.to_datetime(df["report_date"])

cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 33.749,
    "longitude": -84.388,
    "start_date": "2021-01-01",
    "end_date": "2025-11-30", # Extended to ensure all 2025 data is captured
    "hourly": [
        "temperature_2m",
        "precipitation",
        "rain",
        "apparent_temperature",
        "weather_code",
        "is_day",
    ],
    "daily": [
        "sunrise",
        "daylight_duration",
        "sunshine_duration",
        "precipitation_hours",
        "rain_sum",
        "temperature_2m_mean",
        "weather_code",
    ],
    "timezone": "America/New_York",
    "temperature_unit": "fahrenheit",
}

try:
    console.print("[cyan]Fetching weather from Open-Meteo API...[/cyan]")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_df = pd.DataFrame(
        {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temp_f": hourly.Variables(0).ValuesAsNumpy(),
            "precip_in": hourly.Variables(1).ValuesAsNumpy(),
            "rain_in": hourly.Variables(2).ValuesAsNumpy(),
            "apparent_temp_f": hourly.Variables(3).ValuesAsNumpy(),
            "weather_code_hourly": hourly.Variables(4).ValuesAsNumpy(),
            "is_daylight": hourly.Variables(5).ValuesAsNumpy().astype(int),
        }
    )
    hourly_df["datetime"] = (
        hourly_df["datetime"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    )

    daily = response.Daily()
    daily_df = pd.DataFrame(
        {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            ),
            "sunrise": daily.Variables(0).ValuesInt64AsNumpy(),
            "daylight_duration_sec": daily.Variables(1).ValuesAsNumpy(),
            "sunshine_duration_sec": daily.Variables(2).ValuesAsNumpy(),
            "precip_hours": daily.Variables(3).ValuesAsNumpy(),
            "rain_sum_in": daily.Variables(4).ValuesAsNumpy(),
            "temp_mean_f": daily.Variables(5).ValuesAsNumpy(),
            "weather_code_daily": daily.Variables(6).ValuesAsNumpy(),
        }
    )
    daily_df["date"] = (
        daily_df["date"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.date
    )

    console.print(
        f"[green]✓ Fetched {len(hourly_df):,} hourly and {len(daily_df):,} daily weather records.[/green]"
    )
    
    # === FIX: Save the newly fetched data to the external folder ===
    hourly_df.to_csv(HOURLY_WEATHER_PATH, index=False)
    daily_df.to_csv(DAILY_WEATHER_PATH, index=False)
    console.print(f"[green]✓ Saved updated hourly data to: {HOURLY_WEATHER_PATH}[/green]")
    console.print(f"[green]✓ Saved updated daily data to: {DAILY_WEATHER_PATH}[/green]")
    # ====================================================================

    # Merge hourly weather
    df["weather_datetime"] = df["report_date"].dt.floor("H")

    overlapping_hourly_cols = [
        col for col in df.columns if col in hourly_df.columns and col != "weather_datetime"
    ]
    if overlapping_hourly_cols:
        df = df.drop(columns=overlapping_hourly_cols)

    df = df.merge(
        hourly_df,
        left_on="weather_datetime",
        right_on="datetime",
        how="left",
    ).drop(columns=["datetime"])

    # Merge daily weather
    df["report_date_only"] = df["report_date"].dt.date
    overlapping_daily_cols = [
        col for col in df.columns if col in daily_df.columns and col != "date"
    ]
    if overlapping_daily_cols:
        df = df.drop(columns=overlapping_daily_cols)

    df = df.merge(
        daily_df,
        left_on="report_date_only",
        right_on=daily_df['date'].dt.date,
        how="left",
    ).drop(columns=["date", "report_date_only", "key_0"])

    df = df.drop(columns=['weather_datetime'], errors='ignore')

    console.print("[green]✓ Merged full 2021–2025 weather successfully.[/green]")
except Exception as e:
    console.print(f"[bold red]Error fetching weather:[/bold red] {e}")

show_missing_comparison(df, weather_before, "Step 14")
log_step("Step 14: Fetched and merged full weather dataset (2021–2025)", df)

# %%
console.print(Panel("[bold cyan]STEP 15: Recalculate temperature-based flags[/bold cyan]", border_style="cyan"))

if "temp_f" in df.columns:
    p85 = df["temp_f"].quantile(0.85)
    p15 = df["temp_f"].quantile(0.15)

    console.print("[cyan]Temperature percentile thresholds:[/cyan]")
    console.print(f"  85th percentile: [yellow]{p85:.1f}[/yellow] F")
    console.print(f"  15th percentile: [yellow]{p15:.1f}[/yellow] F")

    df["is_hot"] = (df["temp_f"] >= p85).astype(int)
    df["is_cold"] = (df["temp_f"] <= p15).astype(int)

    hot_pct = (df["is_hot"] == 1).sum() / len(df) * 100
    cold_pct = (df["is_cold"] == 1).sum() / len(df) * 100

    console.print("[green]✓ Recalculated temperature flags:[/green]")
    console.print(f"  is_hot (>= {p85:.1f} F): {hot_pct:.1f}%")
    console.print(f"  is_cold (<= {p15:.1f} F): {cold_pct:.1f}%")

log_step("Step 15: Recalculated temperature-based flags", df)

# %% [markdown]
# #### Section 10: Final Verification, Save, and Core EDA

# %%
console.print(Panel("[bold cyan]STEP 16: Final verification and save[/bold cyan]", border_style="cyan"))

# Final missing data summary
all_cols_to_check = ['campus_distance_m', 'neighborhood', 'neighborhood_label', 'zone_int', 'npu', 'city_label', 'weather_code_hrly', 'is_daylight', 'temp_f', 'precip_in'] # Top 10 columns only
final_missing: Dict[str, Any] = {}

console.print("\n[yellow]Final missing data summary (Top 10 columns only):[/yellow]")
final_table = Table(show_header=True, header_style="bold magenta")
final_table.add_column("Column", style="cyan")
final_table.add_column("Missing", justify="right", style="red")
final_table.add_column("%", justify="right", style="yellow")

for col in all_cols_to_check:
    # Use fallback column names if needed
    current_col = col if col in df.columns else (
        'zone_int' if col == 'district' else (
            'npu_label' if col == 'npu' else col
        )
    )
    if current_col in df.columns:
        missing = df[current_col].isna().sum()
        pct = (missing / len(df)) * 100
        final_table.add_row(col, f"{missing:,}", f"{pct:.1f}%")

console.print(final_table)

# Save final processed dataset
OUTPUT_PATH = PROCESSED_DATA_FOLDER / "target_crimes.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

log_step("Processed Data Export", df)

console.print(
    Panel.fit(
        "[bold green]✓ CLEANING COMPLETE![/bold green]\n\n"
        f"Saved to: [yellow]{OUTPUT_PATH}[/yellow]\n"
        f"Total records: [cyan]{len(df):,}[/cyan]\n"
        f"Total columns: [cyan]{len(df.columns)}[/cyan]",
        border_style="green",
    )
)

# %%
console.print(Panel("[bold magenta]STEP 17: Core EDA visualizations[/bold magenta]", border_style="magenta"))
eda_df = df.copy()
date_range_str = f"{eda_df['year'].min()}–{eda_df['year'].max()}"

# Build GeoDataFrames for spatial EDA (needed for NPU map)
gdf_full = gpd.GeoDataFrame(
    eda_df.dropna(subset=['longitude', 'latitude']),
    geometry=gpd.points_from_xy(eda_df.longitude, eda_df.latitude),
    crs="EPSG:4326",
)

# --- Quick Visuals -----------------------------------------------------------

sns.set(style="whitegrid")

# Crimes by hour
plt.figure(figsize=(10, 5))
sns.countplot(x="incident_hour", data=eda_df, color="steelblue")
plt.title("Crimes by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Crimes by day of week
if "day_of_week" in eda_df.columns:
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    plt.figure(figsize=(10, 5))
    sns.countplot(x="day_of_week", data=eda_df, order=order)
    plt.title("Crimes by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Heatmap: day_of_week × hour (KEEP)
pivot = (
    eda_df.pivot_table(
        index="day_of_week",
        columns="incident_hour",
        values="incident_number",
        aggfunc="count",
    )
    .fillna(0)
)

pivot = pivot.reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

plt.figure(figsize=(16, 6))
sns.heatmap(pivot, cmap="mako")
plt.title("Heatmap: Day of Week × Hour")
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.show()

# Weather distribution (Only temp_f)
if "temp_f" in eda_df.columns:
    plt.figure(figsize=(6, 4))
    eda_df["temp_f"].hist(bins=30)
    plt.title("Distribution of Temperature (temp_f)")
    plt.xlabel(r"Temperature ($\circ$F)") # Added degrees symbol
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# --- Plotly Interactive Monthly Comparison (KEEP) -----------------------------
monthly_data = eda_df.groupby([eda_df['year'], eda_df['month']]).size().reset_index(name='count')
monthly_data['year'] = monthly_data['year'].astype(str)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig = px.line(monthly_data, x='month', y='count', color='year',
              title='Monthly Crime Trends by Year (Interactive)',
              labels={'month': 'Month', 'count': 'Number of Crimes', 'year': 'Year'},
              markers=True)

fig.update_xaxes(tickmode='linear', tick0=1, dtick=1,
                 ticktext=month_names, tickvals=list(range(1, 13)))
fig.update_layout(hovermode='x unified', height=600)
fig.show()

log_step("Step 17: Core EDA visualizations (including interactive Plotly)", eda_df)

# %%
console.print(
    Panel(
        "[bold cyan]STEP 18: Advanced temporal binning (6 x 4-hour bins)[/bold cyan]",
        border_style="cyan",
    )
)

# Overall time bin distribution (6-bin logic)
time_bin_counts = (
    eda_df["hour_block"].value_counts()
    .reindex(["Early Night (0–4)", "Early Morning (5–8)", "Late Morning (9–12)", "Afternoon (13–16)", "Evening (17–20)", "Late Night (21–24)"])
)
colors_bins = ["#2C3E50", "#34495E", "#95A5A6", "#E67E22", "#E74C3C", "#8E44AD"] # Consistent colors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.bar(
    range(len(time_bin_counts)),
    time_bin_counts.values,
    color=colors_bins,
    edgecolor="black",
)
ax1.set_xticks(range(len(time_bin_counts)))
ax1.set_xticklabels([l.split(' ')[0] for l in time_bin_counts.index], rotation=45, ha='right')
ax1.set_title("Crime Distribution by Time Block (All Years)")
ax1.set_ylabel("Number of Crimes")
ax1.xlabel = 'Time Block'
ax1.grid(axis="y", alpha=0.3)

for i, v in enumerate(time_bin_counts.values):
    ax1.text(i, v + max(time_bin_counts.values) * 0.01, f"{v:,}", ha="center", fontweight='bold')

# Time bin by year
time_bin_year = eda_df.groupby(["year", "hour_block"]).size().unstack(fill_value=0)
time_bin_year = time_bin_year[
    ["Early Night (0–4)", "Early Morning (5–8)", "Late Morning (9–12)", "Afternoon (13–16)", "Evening (17–20)", "Late Night (21–24)"]
]

x = np.arange(len(time_bin_year.index))
width = 0.14

for i, (bin_name, color) in enumerate(zip(time_bin_year.columns, colors_bins)):
    ax2.bar(
        x + i * width,
        time_bin_year[bin_name].values,
        width,
        label=bin_name.split(' ')[0],
        color=color,
        edgecolor="black",
    )

ax2.set_title("Crime by Time Block and Year")
ax2.set_xlabel("Year")
ax2.set_ylabel("Number of Crimes")
ax2.set_xticks(x + width * 2.5)
ax2.set_xticklabels(time_bin_year.index)
ax2.legend(title="Time Block", loc="upper right", fontsize=9, ncol=2)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# Crimes per year
crimes_per_year = eda_df["year"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 6))
crimes_per_year.plot(kind="bar", color="steelblue", ax=ax, edgecolor="black")
ax.set_title(f"Total Crimes Per Year ({date_range_str})")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Crimes")
ax.grid(axis="y", alpha=0.3)

for i, v in enumerate(crimes_per_year.values):
    ax.text(i, v + max(crimes_per_year.values) * 0.01, f"{v:,}", ha="center", fontweight='bold')

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Crimes by day of week (again, but from this EDA branch)
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
day_crimes = eda_df["day_of_week"].value_counts().reindex(day_order)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    range(len(day_crimes)),
    day_crimes.values,
    color="lightblue",
    edgecolor="black",
)
ax.set_title("Crimes by Day of Week")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Number of Crimes")
ax.set_xticks(range(len(day_order)))
ax.set_xticklabels(day_order, rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)

for i, v in enumerate(day_crimes.values):
    ax.text(i, v + max(day_crimes.values) * 0.01, f"{v:,}", ha="center", fontweight='bold')

plt.tight_layout()
plt.show()

log_step("Step 18: Advanced temporal EDA (6-bin blocks and yearly trends)", eda_df)

# %%
console.print(
    Panel(
        "[bold cyan]STEP 19: Spatial EDA – NPU choropleth, KDE, hexbin, GSU buffers[/bold cyan]",
        border_style="cyan",
    )
)

# Build GeoDataFrames for spatial EDA
gdf_3857 = gdf_full.to_crs(epsg=3857)
gdf_npu = gpd.read_file(NPU_SHP).to_crs(3857)
gdf_cities = gpd.read_file(CITIES_SHP).to_crs(3857)
atlanta_boundary = gdf_cities[gdf_cities["NAME"] == "Atlanta"].iloc[0].geometry

# Identify the shapefile column with NPU name/label
candidate_npu_cols = ["NPU", "NPU_ID", "NPU_NUM", "NPU_NAME", "NAME"]
npu_col_shp = next((c for c in candidate_npu_cols if c in gdf_npu.columns), gdf_npu.columns[0])
npu_col = 'npu' # Use the standardized column name from the DataFrame

# 19.1 NPU choropleth with bold black outlines and labels
gdf_join_npu = gpd.sjoin(
    gdf_3857, gdf_npu.rename(columns={npu_col_shp: 'npu_col_shp_merged'}, errors='ignore')[['npu_col_shp_merged', 'geometry']], 
    how="left", predicate="intersects"
)
# Merge crime counts based on the cleaned DataFrame column
npu_counts = eda_df.groupby(npu_col).size().reset_index(name="crime_count")
gdf_npu_plot = gdf_npu.merge(
    npu_counts, left_on=npu_col_shp, right_on=npu_col, how="left"
).fillna(0)


fig, ax = plt.subplots(figsize=(16, 16))
gdf_npu_plot.plot(
    column="crime_count",
    cmap="Reds",
    legend=True,
    edgecolor="black",
    linewidth=1.5,
    ax=ax,
    legend_kwds={"label": "Total crimes", "shrink": 0.7},
)

for idx, row in gdf_npu_plot.iterrows():
    centroid = row.geometry.centroid
    label = row[npu_col_shp] if npu_col_shp in row else row[npu_col] # Use shapefile label first, then DataFrame label
    count = int(row["crime_count"])
    ax.text(
        centroid.x,
        centroid.y,
        f"{label}\n({count:,})",
        fontsize=9,
        ha="center",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

ax.set_title(f"Crime density by NPU ({date_range_str})")
ax.axis("off")
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.tight_layout()
plt.show()

# 19.2 KDE hotspot map
coords = np.vstack([gdf_3857.geometry.x, gdf_3857.geometry.y]).T
kde = KernelDensity(bandwidth=400, kernel="gaussian").fit(coords)

atl_bounds = atlanta_boundary.bounds
x_margin = (atl_bounds[2] - atl_bounds[0]) * 0.1
y_margin = (atl_bounds[3] - atl_bounds[1]) * 0.1

xmin = atl_bounds[0] - x_margin
xmax = atl_bounds[2] + x_margin
ymin = atl_bounds[1] - y_margin
ymax = atl_bounds[3] + y_margin

xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
z = np.exp(kde.score_samples(grid_points)).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(16, 16))
im = ax.imshow(
    z.T,
    extent=[xmin, xmax, ymin, ymax],
    origin="lower",
    cmap="hot",
    alpha=0.8,
    vmin=z.max() * 0.3,
)

# NPU boundaries in bold black
gdf_npu.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, alpha=0.7)

# Atlanta boundary in black, thicker
gpd.GeoSeries([atlanta_boundary]).boundary.plot(
    ax=ax, edgecolor="black", linewidth=2
)

plt.colorbar(im, ax=ax, label="Crime density", shrink=0.7)
ax.set_title("Kernel density crime hotspot map")
ax.axis("off")
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.tight_layout()
plt.show()

# 19.3 Hexbin with NPU overlay (bold black)
fig, ax = plt.subplots(figsize=(16, 16))
hb = ax.hexbin(
    gdf_3857.geometry.x,
    gdf_3857.geometry.y,
    gridsize=70,
    cmap="Purples",
    mincnt=1,
    extent=[xmin, xmax, ymin, ymax],
    vmin=1,
)

gdf_npu.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, alpha=0.7)

gpd.GeoSeries([atlanta_boundary]).boundary.plot(
    ax=ax, edgecolor="black", linewidth=2
)

plt.colorbar(hb, ax=ax, label="Crime count per hexagon", shrink=0.7)
ax.set_title("Hexbin spatial density with NPU overlay")
ax.axis("off")
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.tight_layout()
plt.show()

# 19.4 GSU campus buffers with proper circles and crime points
gsu_pt = Point(SCHOOL_CENTERS['GSU'][1], SCHOOL_CENTERS['GSU'][0])
gsu_3857 = gpd.GeoSeries([gsu_pt], crs=4326).to_crs(3857).iloc[0]

buffers = {
    "0.25 mile": gsu_3857.buffer(402.336),
    "0.5 mile": gsu_3857.buffer(804.672),
    "1 mile": gsu_3857.buffer(1609.34),
}

max_buffer = buffers["1 mile"]
bounds = gpd.GeoSeries([max_buffer]).total_bounds
buffer_margin = 800

fig, ax = plt.subplots(figsize=(16, 16))

# FIX: Buffer lines are distinct colors for visibility
buffer_styles = [
    ("0.25 mile", "red", 2.5),
    ("0.5 mile", "orange", 2),
    ("1 mile", "yellow", 1.5),
]

for name, color, lw in buffer_styles:
    gpd.GeoSeries([buffers[name]]).plot(
        ax=ax,
        alpha=0,
        edgecolor=color,
        linewidth=lw,
        label=f"{name} radius",
        facecolor="none",
        linestyle="-",
    )

crimes_in_area = gdf_3857.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]]
crimes_in_area.plot(
    ax=ax,
    color="red",
    markersize=2,
    alpha=0.4,
    label=f"Crime incidents (n={len(crimes_in_area):,})",
)

gpd.GeoSeries([gsu_3857]).plot(
    ax=ax,
    color="blue",
    markersize=60,
    marker="*",
    zorder=5,
    label="GSU campus center",
    edgecolor="white",
    linewidth=1,
)

ax.set_xlim(bounds[0] - buffer_margin, bounds[2] + buffer_margin)
ax.set_ylim(bounds[1] - buffer_margin, bounds[3] + buffer_margin)

ax.set_title(
    "Crimes within GSU campus buffers\nGSU center: (33.7538 N, 84.3880 W)"
)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
ax.axis("off")
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.tight_layout()
plt.show()

log_step("Step 19: Spatial EDA (NPU choropleth, KDE, hexbin, GSU buffers)", eda_df)

# %% [markdown]
# #### Section 12: Comprehensive Crime Statistics Summary (Rich Formatting)

# %%
console.print(
    Panel(
        "[bold cyan]STEP 20: Comprehensive crime statistics summary[/bold cyan]",
        border_style="cyan",
    )
)

# --- Rich Summary Tables (6-BIN STANDARD) -------------------------------------

# Offense breakdown (reusing eda_df from Step 17)
if "offense_category" in eda_df.columns:
    console.print("[cyan]Offense Category Breakdown[/cyan]")
    console.print(eda_df["offense_category"].value_counts().to_string())

summary_table = Table(
    title=f"COMPREHENSIVE CRIME STATISTICS SUMMARY ({date_range_str})",
    show_header=False,
    header_style="bold magenta",
    show_lines=True,
)
summary_table.add_column("Metric", style="cyan")
summary_table.add_column("Value", style="green")

summary_table.add_row("[bold]Total Crimes[/bold]", f"{len(eda_df):,}")
summary_table.add_row(
    "[bold]Date Range[/bold]",
    f"{eda_df['report_date'].min().date()} → {eda_df['report_date'].max().date()}",
)

# Temporal insights
temporal_table = Table(
    title="📊 Temporal Insights", show_header=True, header_style="bold yellow", show_lines=True
)
temporal_table.add_column("Time Metric", style="cyan")
temporal_table.add_column("Peak Value", style="green")
temporal_table.add_column("Count", justify="right", style="magenta")

temporal_table.add_row(
    "Peak Hour",
    f"{eda_df['incident_hour'].mode()[0]:02}:00", # Fixed formatting to 2 digits
    f"{eda_df['incident_hour'].value_counts().max():,}",
)
temporal_table.add_row(
    "Peak Day",
    f"{eda_df['day_of_week'].mode()[0]}",
    f"{eda_df['day_of_week'].value_counts().max():,}",
)
month_names = {
    i: name
    for i, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        1,
    )
}
temporal_table.add_row(
    "Peak Month",
    f"{month_names[eda_df['month'].mode()[0]]}",
    f"{eda_df['month'].value_counts().max():,}",
)
temporal_table.add_row(
    "Peak Year",
    f"{eda_df['year'].mode()[0]}",
    f"{eda_df['year'].value_counts().max():,}",
)

# Time block analysis (6-BIN STANDARD)
time_block_table = Table(
    title="🔥 Time Block Analysis (6 x 4-Hour Bins)",
    show_header=True,
    header_style="bold yellow",
    show_lines=True,
)
time_block_table.add_column("Time Block", style="cyan")
time_block_table.add_column("Count", justify="right", style="green")
time_block_table.add_column("Percent", justify="right", style="magenta")

time_block_order = [
    "Early Night (0–4)",
    "Early Morning (5–8)",
    "Late Morning (9–12)",
    "Afternoon (13–16)",
    "Evening (17–20)",
    "Late Night (21–24)",
]

for block_name in time_block_order:
    if "hour_block" in eda_df.columns:
        count = (eda_df["hour_block"] == block_name).sum()
        pct = (count / len(eda_df)) * 100
        time_block_table.add_row(block_name, f"{count:,}", f"{pct:.1f}%")

# Day-of-week table
day_of_week_table = Table(
    title="📅 Day of Week Analysis",
    show_header=True,
    header_style="bold yellow",
    show_lines=True,
)
day_of_week_table.add_column("Day", style="cyan")
day_of_week_table.add_column("Count", justify="right", style="green")
day_of_week_table.add_column("Percent", justify="right", style="magenta")

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
for day in day_order:
    count = (eda_df["day_of_week"] == day).sum()
    pct = (count / len(eda_df)) * 100
    day_of_week_table.add_row(day, f"{count:,}", f"{pct:.1f}%")

rprint(summary_table)
rprint(temporal_table)
rprint(time_block_table)
rprint(day_of_week_table)

log_step("Step 20: Comprehensive crime statistics summary (Rich)", eda_df)

console.print(Panel("[bold green]✓ All pipeline steps complete![/bold green]"))
show_pipeline_table()