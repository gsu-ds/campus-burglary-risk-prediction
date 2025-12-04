# Core data validation checks for APD crime pipeline

import pandas as pd
from rich.console import Console
from rich.table import Table
from typing import Dict, Any

console = Console()


def run_validation_checks(df: pd.DataFrame, step_name: str, check_npu: bool = False) -> None:
    """
    Key integrity checks:
    - incident_number uniqueness
    - coordinate bounds (rough ATL box)
    - core completeness
    - NPU category sanity (optional)
    """
    LAT_MIN, LAT_MAX = 33.5, 34.0
    LON_MIN, LON_MAX = -84.6, -84.2

    if "incident_number" in df.columns:
        duplicates = df.duplicated(subset=["incident_number"]).sum()
        if duplicates > 0:
            console.print(
                f"[bold red]FAIL: {step_name} - {duplicates:,} incident_number duplicates.[/bold red]"
            )
        else:
            console.print(f"[green]PASS: {step_name} - incident_number unique.[/green]")

    if all(c in df.columns for c in ["latitude", "longitude"]):
        out_of_bounds = df[
            (df["latitude"] < LAT_MIN)
            | (df["latitude"] > LAT_MAX)
            | (df["longitude"] < LON_MIN)
            | (df["longitude"] > LON_MAX)
        ].shape[0]
        if out_of_bounds > 0:
            console.print(
                f"[bold yellow]WARNING: {step_name} - {out_of_bounds:,} rows outside expected ATL bounds.[/bold yellow]"
            )
        else:
            console.print(f"[green]PASS: {step_name} - coordinates within expected bounds.[/green]")

    core_cols = ["report_date", "nibrs_offense"]
    for col in core_cols:
        if col in df.columns:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct > 0.01:
                console.print(
                    f"[bold red]FAIL: {step_name} - '{col}' missing {missing_pct:.2%} (>1%).[/bold red]"
                )
            else:
                console.print(
                    f"[green]PASS: {step_name} - '{col}' completeness OK ({missing_pct:.2%} missing).[/green]"
                )

    if check_npu and "npu" in df.columns and df["npu"].dtype == "category":
        valid_npus = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"OUT_OF_NPU_BOUNDS"}
        current_npus = set(df["npu"].cat.categories.str.upper().tolist())
        invalid_npus = current_npus - valid_npus
        if invalid_npus:
            console.print(
                f"[bold red]FAIL: {step_name} - unexpected NPU categories: {invalid_npus}[/bold red]"
            )
        else:
            console.print(
                f"[green]PASS: {step_name} - NPU categories valid (A-Z, OUT_OF_NPU_BOUNDS).[/green]"
            )


def show_missing_comparison(df_local: pd.DataFrame, snapshot: Dict[str, Any], step_name: str) -> None:
    """Compare missingness before/after a step."""
    table = Table(
        title=f"{step_name} - Missing Data Comparison",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Column", style="cyan")
    table.add_column("Before", justify="right", style="red")
    table.add_column("After", justify="right", style="green")
    table.add_column("Filled", justify="right", style="blue")

    for col, (before, before_pct) in snapshot.items():
        if col in df_local.columns:
            after = df_local[col].isna().sum()
            filled = before - after
            table.add_row(
                col,
                f"{before:,} ({before_pct:.1f}%)",
                f"{after:,} ({after/len(df_local)*100:.1f}%)",
                f"{filled:,}",
            )

    console.print(table)


__all__ = ["run_validation_checks", "show_missing_comparison"]