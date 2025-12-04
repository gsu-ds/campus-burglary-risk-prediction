# Validation checks for the NPU × hour panels


import pandas as pd
from rich.console import Console

console = Console()


def panel_quality_report(df: pd.DataFrame) -> None:
    """
    Generate comprehensive quality report for panel data.
    Shows shape, missing values, lag distributions, and sample rows.
    """
    console.print("\n[bold cyan]=== PANEL QUALITY REPORT ===[/bold cyan]")
    console.print(f"[cyan]Rows:[/cyan] {df.shape[0]:,}")
    console.print(f"[cyan]Columns:[/cyan] {df.shape[1]}")
    
    console.print("\n[yellow]Missing values (top 20):[/yellow]")
    missing = df.isna().sum().sort_values(ascending=False).head(20)
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            console.print(f"  {col}: {count:,} ({pct:.2f}%)")
    
    console.print("\n[yellow]Lag distribution:[/yellow]")
    lag_cols = [1, 3, 6, 12, 24, 168]
    for lag in lag_cols:
        col = f"lag_{lag}"
        if col in df.columns:
            missing_count = df[col].isna().sum()
            pct = (missing_count / len(df)) * 100
            console.print(f"  {col}: {missing_count:,} missing ({pct:.2f}%)")
    
    console.print("\n[cyan]Sample rows (first 5):[/cyan]")
    console.print(df.head().to_string())


def validate_panel_schema(df: pd.DataFrame, panel_type: str = "burglary") -> None:
    """
    Validate that panel has all required columns for modeling.
    
    Parameters:
        df: Panel DataFrame
        panel_type: Type of panel ("burglary" for burglary_count or "crime" for crime_count)
    """
    if panel_type == "burglary":
        count_col = "burglary_count"
    else:
        count_col = "crime_count"
    
    required_cols = [
        count_col,
        "hour_ts",
        "npu",
        "lag_1", "lag_3", "lag_6", "lag_12", "lag_24", "lag_168",
        "temp_f", "is_raining", "is_hot", "is_cold",
        "grid_density_7d", "npu_crime_avg_30d",
        "hour_sin", "hour_cos",
        "day_number", "day_of_week",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        console.print(f"[bold red]Panel schema validation failed.[/bold red]")
        console.print(f"[red]Missing columns:[/red] {missing}")
        raise ValueError(f"Panel schema validation failed. Missing: {missing}")
    
    console.print("[green]Panel schema validated.[/green]")


def validate_panel_completeness(df: pd.DataFrame, max_missing_pct: float = 0.1) -> None:
    """
    Check that critical features have acceptable missingness.
    
    Parameters:
        df: Panel DataFrame
        max_missing_pct: Maximum allowed missing percentage (default 10%)
    """
    critical_cols = [
        "temp_f", "grid_density_7d", "npu_crime_avg_30d",
        "hour_sin", "hour_cos", "day_number"
    ]
    
    failures = []
    for col in critical_cols:
        if col in df.columns:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct > max_missing_pct:
                failures.append((col, missing_pct))
    
    if failures:
        console.print(f"[bold red]Completeness validation failed:[/bold red]")
        for col, pct in failures:
            console.print(f"  [red]{col}: {pct:.2%} missing (>{max_missing_pct:.0%} threshold)[/red]")
        raise ValueError(f"Panel completeness validation failed for: {[c for c, _ in failures]}")
    
    console.print("[green]Panel completeness validated.[/green]")


def validate_panel_time_range(df: pd.DataFrame, min_hours: int = 168) -> None:
    """
    Validate panel covers sufficient time range for modeling.
    
    Parameters:
        df: Panel DataFrame
        min_hours: Minimum required hours (default 168 = 1 week)
    """
    if "hour_ts" not in df.columns:
        console.print("[yellow]Warning: hour_ts column not found, skipping time range check.[/yellow]")
        return
    
    unique_hours = df["hour_ts"].nunique()
    
    if unique_hours < min_hours:
        console.print(
            f"[bold red]Time range validation failed:[/bold red] "
            f"Only {unique_hours} unique hours (need {min_hours}+)"
        )
        raise ValueError(f"Panel has insufficient time range: {unique_hours} < {min_hours}")
    
    time_range = df["hour_ts"].max() - df["hour_ts"].min()
    console.print(f"[green]Time range validated: {unique_hours} hours ({time_range})[/green]")


__all__ = [
    "panel_quality_report",
    "validate_panel_schema",
    "validate_panel_completeness",
    "validate_panel_time_range"
]