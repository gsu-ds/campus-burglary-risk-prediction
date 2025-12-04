# Core cleaning transformations used before any feature engineering
import re
from typing import Any
import numpy as np
import pandas as pd
from dateutil import parser
from rich.console import Console

from atl_model_pipelines.utils.logging import log_step

console = Console()


def standardize_column_name(col: str) -> str:
    """Convert arbitrary APD CSV column names into clean_snake_case."""
    col = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", col)
    col = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
    col = col.lower()
    col = re.sub(r"[\s\-\.\,\(\)\[\]\{\}]+", "_", col)
    col = re.sub(r"[^\w]", "", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def parse_report_date(x: Any) -> pd.Timestamp:
    """
    Safely parse APD-style date strings with mixed formats.
    Falls back to dateutil.parser when possible.
    """
    if pd.isna(x):
        return pd.NaT

    s = str(x).strip()
    if s.lower() == "nan":
        return pd.NaT

    if re.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [APMapm]{2}$", s):
        return pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

    try:
        return parser.parse(s, fuzzy=True)
    except Exception:
        return pd.NaT


def cleanup_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated columns (same name) while preserving first occurrence."""
    cols = df.columns
    seen = set()
    keep = []

    for c in cols:
        if c not in seen:
            keep.append(c)
            seen.add(c)
        else:
            console.print(f"[yellow]Dropped duplicate column:[/yellow] {c}")

    return df.loc[:, keep]


def standardize_report_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fully standardize and validate the 'report_date' field.
    - Preserves raw string version ('raw_report_date')
    - Applies robust parsing
    - Drops rows with unparseable dates
    - Ensures final type is datetime64
    """
    console.print("\n[bold cyan]Standardizing report_date...[/bold cyan]")

    df = df.copy()

    if "report_date" not in df.columns:
        raise KeyError("'report_date' column not found in APD data.")

    total_rows = len(df)

    df["raw_report_date"] = df["report_date"].apply(
        lambda x: str(x).strip() if pd.notna(x) else np.nan
    )

    df["report_date"] = df["raw_report_date"].apply(parse_report_date)

    invalid = df["report_date"].isna().sum()
    parsed = total_rows - invalid

    console.print(f"[cyan]Rows: {total_rows:,}")
    console.print(f"[green]Parsed dates: {parsed:,}")
    console.print(f"[yellow]Dropped invalid dates: {invalid:,}")

    df = df.dropna(subset=["report_date"]).copy()
    df["report_date"] = pd.to_datetime(df["report_date"])

    log_step("Step: report_date → datetime", df)

    return df


def clean_initial_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recommended cleaning sequence before any feature engineering.

    Steps:
        1. Drop duplicate-named columns
        2. Standardize report_date to datetime
    """
    df = cleanup_duplicate_columns(df)
    df = standardize_report_date(df)
    return df