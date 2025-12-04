"""
transform_master.py

Orchestrates all transformation steps for APD crime data.
"""

from rich.console import Console
from atl_model_pipelines.validation.core import run_validation_checks
from atl_model_pipelines.utils.logging import log_step, show_pipeline_table, clear_pipeline_log

from atl_model_pipelines.transform.temporal import add_temporal_features
from atl_model_pipelines.transform.weather import merge_weather, add_weather_flags
from atl_model_pipelines.transform.spatial import add_spatial_features
from atl_model_pipelines.transform.grid_density import compute_grid_density
from atl_model_pipelines.transform.rolling_stats import add_npu_rolling_average
from atl_model_pipelines.transform.campus_distance import compute_campus_distance
from atl_model_pipelines.transform.filters import filter_target_offenses
from atl_model_pipelines.transform.finalize import finalize_target_crimes

console = Console()


def run_transforms(ingestion_output: dict) -> dict:
    """
    Run complete transformation pipeline.
    
    Parameters:
        ingestion_output: Dict with keys:
            - apd: DataFrame with raw APD data
            - weather_hourly: Hourly weather DataFrame
            - weather_daily: Daily weather DataFrame
            - shapefiles: Dict of loaded shapefiles (optional, not used if spatial loads from config)
    
    Returns:
        Dict with transformed data:
            - target_crimes: Final filtered & cleaned DataFrame
            - all_crimes: Complete transformed DataFrame before filtering
    """
    console.print("\n[bold cyan]=== TRANSFORM PIPELINE START ===[/bold cyan]\n")
    clear_pipeline_log()

    df = ingestion_output["apd"].copy()
    hourly_df = ingestion_output["weather_hourly"]
    daily_df = ingestion_output["weather_daily"]

    log_step("Initial APD data", df)

    df = add_spatial_features(df)
    run_validation_checks(df, "Transform: After spatial")

    df = add_temporal_features(df)
    log_step("After temporal features", df)

    df = merge_weather(df, hourly_df, daily_df)
    df = add_weather_flags(df)
    log_step("After weather merge", df)

    df = compute_grid_density(df)
    log_step("After grid density", df)

    df = add_npu_rolling_average(df, window_days=30)
    log_step("After NPU rolling avg", df)

    df = compute_campus_distance(df)
    log_step("After campus distance", df)

    all_crimes = df.copy()

    df_target = filter_target_offenses(df)
    log_step("After filtering target offenses", df_target)

    df_final = finalize_target_crimes(df_target)
    log_step("Final target crimes dataset", df_final)

    console.print("\n[green]Transformation completed successfully.[/green]\n")
    show_pipeline_table()

    return {
        "target_crimes": df_final,
        "all_crimes": all_crimes
    }