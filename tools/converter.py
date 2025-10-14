# tools/converters.py
from pathlib import Path
import pandas as pd
from numbers_parser import Document

def csv_to_xlsx(csv_file: Path, xlsx_file: Path):
    pd.read_csv(csv_file).to_excel(xlsx_file, index=False)

def xlsx_to_csv(xlsx_file: Path, csv_file: Path):
    pd.read_excel(xlsx_file).to_csv(csv_file, index=False)

def csv_to_parquet(csv_file: Path, parquet_file: Path, compression="snappy"):
    df = pd.read_csv(csv_file)
    df.to_parquet(parquet_file, engine="pyarrow", compression=compression)

def parquet_to_csv(parquet_file: Path, csv_file: Path):
    df = pd.read_parquet(parquet_file, engine="pyarrow")
    df.to_csv(csv_file, index=False)

def xlsx_to_parquet(xlsx_file: Path, parquet_file: Path, compression="snappy"):
    df = pd.read_excel(xlsx_file)
    df.to_parquet(parquet_file, engine="pyarrow", compression=compression)

def parquet_to_xlsx(parquet_file: Path, xlsx_file: Path):
    df = pd.read_parquet(parquet_file, engine="pyarrow")
    df.to_excel(xlsx_file, index=False)

if __name__ == "__main__":
    # examples:
    csv_to_xlsx(Path("data/OctLib.csv"), Path("data/OctLib.xlsx"))
    # csv_to_parquet(Path("data/OctLib.csv"), Path("data/OctLib.parquet"))