# Raw Data Ingestion from Atlanta Police Department's Open Data Portal
from pathlib import Path
from typing import List
import pandas as pd

from rich.console import Console
from config import RAW_APD_DIR, INTERIM_DIR, PROCESSED_DIR 
from atl_model_pipelines.helpers.standardize import standardize_column_name
from atl_model_pipelines.validation.core import run_validation_checks
from atl_model_pipelines.validation.orchestrator import log_step
from atl_model_pipelines.helpers.cleanup import cleanup_duplicate_columns

console = Console()



def combine_and_deduplicate(files: List[Path], dedupe_key: str) -> pd.DataFrame:
    """Read raw APD CSVs → standardize → combine → dedupe."""
    console.print("\n[bold cyan]Combining APD CSVs...[/bold cyan]")

    dfs = []
    for fp in files:
        console.print(f"[cyan]Reading:[/cyan] {fp.name}")
        df = pd.read_csv(fp)
        df.columns = [standardize_column_name(c) for c in df.columns]
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)
    total_rows = len(df_combined)

    # Standardize dedupe key
    dedupe_key_std = standardize_column_name(dedupe_key)
    if dedupe_key_std not in df_combined.columns:
        raise KeyError(
            f"Dedupe key '{dedupe_key_std}' missing after standardization. "
            "Check raw columns for IncidentNumber."
        )

    df_dedup = df_combined.drop_duplicates(subset=[dedupe_key_std])
    dupes = total_rows - len(df_dedup)

    console.print(f"[yellow]Combined rows:[/yellow] {total_rows:,}")
    console.print(f"[red]Removed duplicates:[/red] {dupes:,}")

    log_step(f"Step 3: Combined & deduped by {dedupe_key_std}", df_dedup)

    return df_dedup


def run_ingestion() -> pd.DataFrame:
    input_files = list(RAW_APD_DIR.glob("*.csv"))  
    if not input_files:
        raise FileNotFoundError(f"No CSVs found in {RAW_APD_DIR}")

    df_combined = combine_and_deduplicate(input_files, "IncidentNumber")
    df_combined = cleanup_duplicate_columns(df_combined)
    run_validation_checks(df_combined, "Step 3: Post-ingestion")
    return df_combined
