# Lag & Rolling features

import pandas as pd
from rich.console import Console

console = Console()


def add_lag_features(df: pd.DataFrame, group_col: str, target_col: str, lags=None):
    """
    Add lag features for panel data grouped by NPU.

    Parameters:
        df: DataFrame containing panel data
        group_col: column used to group (e.g., 'npu')
        target_col: name of the target count col (e.g., 'burglary_count')
        lags: list of lag steps (default includes 1–24 hours + 7 days)

    Returns:
        DataFrame with lag columns added.
    """

    if lags is None:
        lags = [1, 3, 6, 12, 24, 168]

    df = df.copy()
    df = df.sort_values([group_col, df.columns[1]])

    for lag in lags:
        col = f"lag_{lag}"
        df[col] = df.groupby(group_col)[target_col].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame, group_col: str, target_col: str, windows=None):
    """Add rolling mean/sum features."""
    if windows is None:
        windows = [3, 6, 12, 24, 168]

    df = df.copy()
    df = df.sort_values([group_col, df.columns[1]])

    for win in windows:
        df[f"roll_mean_{win}"] = (
            df.groupby(group_col)[target_col]
              .shift(1)
              .rolling(win, min_periods=1)
              .mean()
        )

    return df


__all__ = ["add_lag_features", "add_rolling_features"]