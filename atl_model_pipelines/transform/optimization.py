# Memory Optimization

import pandas as pd
import pandas.api.types as ptypes
from rich.console import Console

console = Console()


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to save memory."""
    mem_before = df.memory_usage(deep=True).sum() / (1024**2)
    
    for col in list(df.columns):
        if ptypes.is_numeric_dtype(df[col]):
            if ptypes.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif ptypes.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast="float")
    
    mem_after = df.memory_usage(deep=True).sum() / (1024**2)
    console.print(
        f"[cyan]optimize_dtypes:[/cyan] {mem_before:.2f} → {mem_after:.2f} MB ({mem_after - mem_before:+.2f} MB)"
    )
    return df