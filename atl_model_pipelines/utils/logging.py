# Logging utilities for pipeline steps.

from typing import List, Dict, Any
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

pipeline_log: List[Dict[str, Any]] = []


def log_step(step_name: str, df: pd.DataFrame) -> None:
    """
    Log pipeline step name + shape.
    
    Parameters:
        step_name: Description of the pipeline step
        df: DataFrame to log (or any object with shape attribute)
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        rows_val: Any = "N/A"
        cols_val: Any = "N/A"
        rows_str = "N/A"
        cols_str = "N/A"
    else:
        rows_val = int(df.shape[0])
        cols_val = int(df.shape[1])
        rows_str = f"{rows_val:,}"
        cols_str = str(cols_val)

    pipeline_log.append({"step": step_name, "rows": rows_val, "cols": cols_val})
    console.print(f"[green]{step_name}[/green] [cyan]shape: {rows_str} x {cols_str}[/cyan]")


def show_pipeline_table() -> None:
    """Pretty-print pipeline log as a table."""
    if not pipeline_log:
        console.print("[red]No pipeline steps logged yet.[/red]")
        return

    table = Table(title="Data Pipeline Summary", show_lines=True)
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Rows", style="green")
    table.add_column("Cols", style="yellow")

    for entry in pipeline_log:
        rows_val = entry["rows"]
        cols_val = entry["cols"]
        rows_str = f"{rows_val:,}" if isinstance(rows_val, int) else str(rows_val)
        cols_str = f"{cols_val:,}" if isinstance(cols_val, int) else str(cols_val)
        table.add_row(entry["step"], rows_str, cols_str)

    console.print(table)


def clear_pipeline_log() -> None:
    """Clear the pipeline log."""
    global pipeline_log
    pipeline_log = []
    console.print("[yellow]Pipeline log cleared.[/yellow]")


__all__ = ["log_step", "show_pipeline_table", "clear_pipeline_log", "pipeline_log"]