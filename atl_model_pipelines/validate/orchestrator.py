# Master validation orchestrator that runs all validation checks

import pandas as pd
from rich.console import Console
from typing import Dict, Any, Optional

from .core import run_validation_checks, show_missing_comparison
from .panel_checks import (
    panel_quality_report,
    validate_panel_schema,
    validate_panel_completeness,
    validate_panel_time_range
)

console = Console()


def run_validations(
    df: pd.DataFrame,
    step_name: str = "Final dataset",
    check_npu: bool = False
) -> pd.DataFrame:
    """
    Run standard validation checks on crime data.
    
    Parameters:
        df: DataFrame to validate
        step_name: Name of the validation step
        check_npu: Whether to check NPU categories
    
    Returns:
        Original DataFrame (unmodified)
    """
    console.print("\n[bold cyan]=== VALIDATION PIPELINE START ===[/bold cyan]\n")

    run_validation_checks(df, step_name, check_npu=check_npu)

    snapshot = {
        col: (df[col].isna().sum(), (df[col].isna().mean() * 100))
        for col in df.columns
    }

    show_missing_comparison(df, snapshot, f"{step_name} Missingness")

    console.print("\n[green]Validation completed successfully.[/green]\n")
    return df


def run_panel_validations(
    df: pd.DataFrame,
    panel_type: str = "burglary",
    check_schema: bool = True,
    check_completeness: bool = True,
    check_time_range: bool = True,
    show_quality_report: bool = True
) -> pd.DataFrame:
    """
    Run comprehensive validation checks on panel data.
    
    Parameters:
        df: Panel DataFrame to validate
        panel_type: Type of panel ("burglary" or "crime")
        check_schema: Whether to validate schema
        check_completeness: Whether to check feature completeness
        check_time_range: Whether to validate time coverage
        show_quality_report: Whether to display quality report
    
    Returns:
        Original DataFrame (unmodified)
    """
    console.print("\n[bold cyan]=== PANEL VALIDATION START ===[/bold cyan]\n")

    if show_quality_report:
        panel_quality_report(df)

    if check_schema:
        try:
            validate_panel_schema(df, panel_type=panel_type)
        except ValueError as e:
            console.print(f"[bold red]Schema validation failed:[/bold red] {e}")
            raise

    if check_completeness:
        try:
            validate_panel_completeness(df, max_missing_pct=0.1)
        except ValueError as e:
            console.print(f"[bold red]Completeness validation failed:[/bold red] {e}")
            raise

    if check_time_range:
        try:
            validate_panel_time_range(df, min_hours=168)
        except ValueError as e:
            console.print(f"[bold red]Time range validation failed:[/bold red] {e}")
            raise

    console.print("\n[green]Panel validation completed successfully.[/green]\n")
    return df


def create_snapshot(df: pd.DataFrame) -> Dict[str, tuple]:
    """
    Create a snapshot of DataFrame missingness for comparison.
    
    Parameters:
        df: DataFrame to snapshot
    
    Returns:
        Dict mapping column names to (count, percentage) of missing values
    """
    return {
        col: (df[col].isna().sum(), df[col].isna().mean() * 100)
        for col in df.columns
    }


__all__ = [
    "run_validations",
    "run_panel_validations",
    "create_snapshot"
]