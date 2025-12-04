# Filter APD Crimes to Target

import pandas as pd

TARGET_OFFENSES = [
    "larceny",
    "theft",
    "robbery",
    "burglary",
    "prowling",
    "shoplifting",
    "fraud",
    "swindle",
    "embezzelment",
    "credit card",
    "wire fraud",
    "impersonation",
    "motor vehicle theft",
]


def filter_target_offenses(df: pd.DataFrame) -> pd.DataFrame:
    """Filter APD dataset to target crime categories."""

    mask = df["nibrs_offense"].str.contains("|".join(TARGET_OFFENSES), case=False, na=False)
    df_filtered = df[mask].copy()

    if "incident_number" in df_filtered.columns:
        df_filtered = df_filtered.drop_duplicates(subset=["incident_number"])

    return df_filtered