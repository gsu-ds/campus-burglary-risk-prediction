import config
import pandas as pd

# Load data
df = pd.read_parquet(config.MODEL_GRID_PATH)
df['date'] = pd.to_datetime(df['date'])

# Create splits using config dates
train = df[df['date'] < config.TRAIN_END]
val = df[(df['date'] >= config.TRAIN_END) & (df['date'] < config.VAL_END)]
test = df[df['date'] >= config.VAL_END]

print(f"Train: {len(train):,} rows ({train['date'].min()} to {train['date'].max()})")
print(f"Val:   {len(val):,} rows ({val['date'].min()} to {val['date'].max()})")
print(f"Test:  {len(test):,} rows ({test['date'].min()} to {test['date'].max()})")



# Check Splits

tools/utils/validation.py

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

def validate_time_splits(train, val, test, date_col='date'):
    """
    Verifies train/val/test splits don't overlap and cover expected ranges.
    """
    table = Table(title="Time Split Validation", border_style="cyan")
    table.add_column("Split", style="bold")
    table.add_column("Start Date", style="cyan")
    table.add_column("End Date", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("% of Total", justify="right")
    
    total = len(train) + len(val) + len(test)
    
    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        start = df[date_col].min()
        end = df[date_col].max()
        count = len(df)
        pct = (count / total) * 100
        table.add_row(name, str(start), str(end), f"{count:,}", f"{pct:.1f}%")
    
    console.print(table)
    
    # Check for overlap
    train_max = train[date_col].max()
    val_min = val[date_col].min()
    val_max = val[date_col].max()
    test_min = test[date_col].min()
    
    if train_max >= val_min:
        console.print("[red]⚠ WARNING: Train and Val overlap![/red]")
    if val_max >= test_min:
        console.print("[red]⚠ WARNING: Val and Test overlap![/red]")
    else:
        console.print("[green]✓ No overlap detected. Splits are valid.[/green]")

# Usage in modeling scripts
validate_time_splits(train, val, test)