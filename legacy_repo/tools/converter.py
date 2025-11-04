# tools/converters.py
from pathlib import Path
import pandas as pd
from numbers_parser import Document

def csv_to_xlsx(csv_file: Path, xlsx_file: Path):
    """Converts a CSV file to an XLSX file."""
    xlsx_file.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(csv_file).to_excel(xlsx_file, index=False)

def xlsx_to_csv(xlsx_file: Path, csv_file: Path):
    """Converts an XLSX file to a CSV file."""
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    pd.read_excel(xlsx_file).to_csv(csv_file, index=False)

def csv_to_parquet(csv_file: Path, parquet_file: Path, compression="snappy"):
    """Converts a CSV file to a Parquet file."""
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_file)
    df.to_parquet(parquet_file, engine="pyarrow", compression=compression)

def parquet_to_csv(parquet_file: Path, csv_file: Path):
    """Converts a Parquet file to a CSV file."""
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_file, engine="pyarrow")
    df.to_csv(csv_file, index=False)

def xlsx_to_parquet(xlsx_file: Path, parquet_file: Path, compression="snappy"):
    """Converts an XLSX file to a Parquet file."""
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(xlsx_file)
    df.to_parquet(parquet_file, engine="pyarrow", compression=compression)

def parquet_to_xlsx(parquet_file: Path, xlsx_file: Path):
    """Converts a Parquet file to an XLSX file."""
    xlsx_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_file, engine="pyarrow")
    df.to_excel(xlsx_file, index=False)

def numbers_to_csv(numbers_file: Path, csv_file: Path, sheet_index: int = 0, table_index: int = 0):
    """
    Converts a specific table from an Apple Numbers file to a CSV file.
    
    Defaults to the first table (index 0) on the first sheet (index 0).
    """
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        doc = Document(str(numbers_file)) # numbers_parser may prefer string paths
        sheet = doc.sheets[sheet_index]
        table = sheet.tables[table_index]
        
        data = table.rows(values_only=True)
        
        if not data:
            print(f"Warning: No data found in {numbers_file} (Sheet: {sheet_index}, Table: {table_index}). Empty CSV created.")
            Path(csv_file).touch() # Create empty file
            return

        headers = data[0]
        df = pd.DataFrame(data[1:], columns=headers)
        df.to_csv(csv_file, index=False, encoding='utf-8')

    except Exception as e:
        print(f"Error converting {numbers_file}: {e}")
        print("Please check sheet/table indices and file integrity.")


if __name__ == "__main__":
    # Define a base data path
    data_path = Path("data")
    
    # Create sample data if it doesn't exist (for testing)
    sample_csv = data_path / "sample.csv"
    if not sample_csv.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        sample_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
        sample_df.to_csv(sample_csv, index=False)
        print(f"Created {sample_csv} for testing.")

    # --- Example Usage ---
    
    # Define output paths
    test_xlsx = data_path / "converted" / "sample.xlsx"
    test_parquet = data_path / "converted" / "sample.parquet"
    test_csv_from_xlsx = data_path / "converted" / "sample_from_xlsx.csv"

    print(f"Converting {sample_csv} to {test_xlsx}...")
    csv_to_xlsx(sample_csv, test_xlsx)

    print(f"Converting {sample_csv} to {test_parquet}...")
    csv_to_parquet(sample_csv, test_parquet)

    print(f"Converting {test_xlsx} back to {test_csv_from_xlsx}...")
    xlsx_to_csv(test_xlsx, test_csv_from_xlsx)

    print("Conversions complete. Check the 'data/converted' folder.")

    # Example for numbers_to_csv (assuming you have a file)
    # numbers_file = data_path / "MyReport.numbers"
    # numbers_csv_out = data_path / "converted" / "from_numbers.csv"
    # if numbers_file.exists():
    #     print(f"Converting {numbers_file}...")
    #     numbers_to_csv(numbers_file, numbers_csv_out)
    #     # Example with specific sheet/table
    #     # numbers_to_csv(numbers_file, numbers_csv_out, sheet_index=1, table_index=0)
    # else:
    #     print(f"Skipping Numbers conversion, {numbers_file} not found.")