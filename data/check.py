import pandas as pd
from pathlib import Path

# Quick check for csv
for f in Path("data").rglob("*.csv"):
    df = pd.read_csv(f)
    print(f"{str(f):60} {df.shape[0]:>10,} rows × {df.shape[1]:>3} cols")

# Quick check for parquet
for f in Path("data").rglob("*.parquet"):
    df = pd.read_parquet(f)
    print(f"{str(f):60} {df.shape[0]:>10,} rows × {df.shape[1]:>3} cols")