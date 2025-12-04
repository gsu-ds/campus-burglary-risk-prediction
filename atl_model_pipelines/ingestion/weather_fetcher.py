# Weather Fetcher from OpenMeteo

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple

import requests_cache
import openmeteo_requests
from retry_requests import retry

from rich.console import Console
from rich.panel import Panel


from config import (
    EXTERNAL_DIR,
    HOURLY_WEATHER_CSV,
    DAILY_WEATHER_CSV,
    HOURLY_WEATHER_PARQUET,
    DAILY_WEATHER_PARQUET
)

console = Console()

def log_step(step: str, df):
    try:
        from ..validation.orchestrator import log_step as real_log
        real_log(step, df)
    except Exception:
        console.print(f"[blue]Log step:[/blue] {step}")



def fetch_atlanta_weather_full(lat: float = 33.749, lon: float = -84.388) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch full historical weather (2021 → today) via Open-Meteo API."""

    start_date = "2021-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    console.print(
        Panel(
            f"[bold cyan]Fetching Weather Data (2021 → {end_date})[/bold cyan]\n"
            f"Location: ({lat:.3f}, {lon:.3f})",
            border_style="cyan",
        )
    )

    # configure caching + retry
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
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

    # -------------------------
    # API RESPONSE
    # -------------------------
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # -------------------------
    # HOURLY
    # -------------------------
    hourly = response.Hourly()

    hourly_df = pd.DataFrame({
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
    })

    hourly_df["datetime"] = (
        hourly_df["datetime"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

    # -------------------------
    # DAILY
    # -------------------------
    daily = response.Daily()

    daily_df = pd.DataFrame({
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
    })

    daily_df["date"] = (
        daily_df["date"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.date
    )

    hourly_df.to_csv(HOURLY_WEATHER_PATH, index=False)
    daily_df.to_csv(DAILY_WEATHER_PATH, index=False)

    try:
        hourly_df.to_parquet(HOURLY_WEATHER_PARQUET, index=False, compression="snappy")
        daily_df.to_parquet(DAILY_WEATHER_PARQUET, index=False, compression="snappy")
    except Exception as e:
        console.print(f"[yellow]Parquet export failed: {e}[/yellow]")

    console.print(
        f"[green]Weather saved → {len(hourly_df):,} hourly rows, {len(daily_df):,} daily rows[/green]"
    )

    return hourly_df, daily_df

def load_or_fetch_weather(refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached weather if available, otherwise fetch fresh data."""

    cache_exists = HOURLY_WEATHER_PATH.exists() and DAILY_WEATHER_PATH.exists()

    console.print(Panel("Weather Data Initialization", border_style="cyan"))

    if refresh or not cache_exists:
        console.print("[yellow]Fetching full weather history from API...[/yellow]")
        hourly_df, daily_df = fetch_atlanta_weather_full()
        log_step("Weather fetched from API", pd.DataFrame())
        return hourly_df, daily_df

    # Load cached
    console.print("[green]Using cached weather CSV files.[/green]")
    hourly_df = pd.read_csv(HOURLY_WEATHER_PATH)
    daily_df = pd.read_csv(DAILY_WEATHER_PATH)
    log_step("Weather loaded from cache", pd.DataFrame())
    return hourly_df, daily_df
