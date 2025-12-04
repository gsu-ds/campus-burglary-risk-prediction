# Encoding for categorical columns

import pandas as pd
from rich.console import Console

console = Console()


def add_count_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Count-encode a categorical column."""
    df = df.copy()
    if col in df.columns:
        new_col = f"{col}_count"
        console.print(f"[cyan]Count-encoding:[/cyan] '{col}' → '{new_col}'")
        df[new_col] = df[col].map(df[col].value_counts()).fillna(0).astype(int)
    return df