"""
This script provides a quick look at the raw data without requiring geopandas.
It uses basic pandas operations to understand the data structure and content.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Import configuration
from src.config.settings import CAMPUSES, CAMPUS_RADIUS_MILES, PROCESSED_DATA_PATH

# Data paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"

RADIUS_MILES = CAMPUS_RADIUS_MILES


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in miles.
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    
    # Radius of earth in miles
    r = 3956 
    
    return c * r


def web_mercator_to_latlon(x, y):
    """Convert Web Mercator coordinates to lat/lon."""
    lon = (x / 20037508.34) * 180
    lat = (y / 20037508.34) * 180
    lat = 180 / np.pi * (2 * np.arctan(np.exp(lat * np.pi / 180)) - np.pi / 2)
    return lat, lon


def load_and_explore_data():
    """Load and explore all data files."""
    print("=" * 80)
    print("QUICK DATA EXPLORATION - 6 CAMPUSES")
    print("=" * 80)
    print(f"Analyzing {len(CAMPUSES)} campuses:")
    for code, info in CAMPUSES.items():
        print(f"  - {info['name']} ({info['city']})")
    print(f"Radius: {RADIUS_MILES} mile(s)")
    print("=" * 80)
    
    # Load 2009-2020 data
    print("\n1. Loading 2009-2020 data...")
    file_2009_2020 = RAW_DATA_PATH / "APD" / "2009-2020-3-categories.csv"
    
    if file_2009_2020.exists():
        df_old = pd.read_csv(file_2009_2020)
        print(f"   Shape: {df_old.shape}")
        print(f"   Columns: {list(df_old.columns)}")
        print(f"\n   First few rows:")
        print(df_old.head(3))
        print(f"\n   Crime types: {df_old['Crime Type'].value_counts()}")
        
        # Convert projected coordinates to lat/lon
        print("\n   Converting coordinates...")
        df_old['latitude'], df_old['longitude'] = zip(*df_old.apply(
            lambda row: web_mercator_to_latlon(row['x'], row['y']), axis=1
        ))
        
        # Standardize
        df_old['report_date'] = pd.to_datetime(df_old['Report Date'])
        df_old['crime_type'] = df_old['Crime Type']
        df_old['source'] = '2009-2020'
        
    else:
        print(f"   File not found: {file_2009_2020}")
        df_old = pd.DataFrame()
    
    # Load 2021-2025 burglary data
    print("\n2. Loading 2021-2025 burglary data...")
    file_burglary = RAW_DATA_PATH / "APD" / "2021-2025-Burglary.csv"
    
    if file_burglary.exists():
        df_burglary = pd.read_csv(file_burglary)
        print(f"   Shape: {df_burglary.shape}")
        print(f"   Columns: {list(df_burglary.columns)}")
        print(f"\n   First few rows:")
        print(df_burglary.head(3))
        
        # Standardize
        df_burglary['report_date'] = pd.to_datetime(df_burglary['ReportDate'])
        df_burglary['crime_type'] = 'Burglary'
        df_burglary['source'] = '2021-2025-burglary'
        
    else:
        print(f"   File not found: {file_burglary}")
        df_burglary = pd.DataFrame()
    
    # Load 2021-2025 larceny data
    print("\n3. Loading 2021-2025 larceny data...")
    file_larceny = RAW_DATA_PATH / "APD" / "2021-2025-All-Other_larceny.csv"
    
    if file_larceny.exists():
        df_larceny = pd.read_csv(file_larceny)
        print(f"   Shape: {df_larceny.shape}")
        print(f"   Columns: {list(df_larceny.columns)}")
        
        # Standardize
        df_larceny['report_date'] = pd.to_datetime(df_larceny['ReportDate'])
        df_larceny['crime_type'] = 'Larceny'
        df_larceny['source'] = '2021-2025-larceny'
        
    else:
        print(f"   File not found: {file_larceny}")
        df_larceny = pd.DataFrame()
    
    return df_old, df_burglary, df_larceny


def filter_by_campus(df, lat_col, lon_col):
    """Filter data to 1-mile radius of campuses."""
    print("\n4. Filtering by campus proximity (1-mile radius)...")
    
    # Calculate distance to each campus
    for campus_code, campus_info in CAMPUSES.items():
        df[f'dist_to_{campus_code}'] = df.apply(
            lambda row: haversine_distance(
                row[lon_col], row[lat_col],
                campus_info['longitude'], campus_info['latitude']
            ),
            axis=1
        )
    
    # Find nearest campus and distance
    dist_cols = [f'dist_to_{code}' for code in CAMPUSES.keys()]
    df['nearest_campus'] = df[dist_cols].idxmin(axis=1).str.replace('dist_to_', '')
    df['distance_to_nearest'] = df[dist_cols].min(axis=1)
    
    # Filter to within radius
    df_filtered = df[df['distance_to_nearest'] <= RADIUS_MILES].copy()
    
    print(f"   Original: {len(df)} records")
    print(f"   Within {RADIUS_MILES} mile: {len(df_filtered)} records")
    print(f"   Filtered out: {len(df) - len(df_filtered)} records")
    
    return df_filtered


def analyze_burglary_data(df_old, df_burglary):
    """Analyze burglary-specific data."""
    print("\n" + "=" * 80)
    print("BURGLARY DATA ANALYSIS")
    print("=" * 80)
    
    # Extract burglary from old data
    df_old_burglary = df_old[df_old['crime_type'].str.contains('BURGLARY', case=False, na=False)].copy()
    print(f"\nBurglary records 2009-2020: {len(df_old_burglary)}")
    
    # Filter by campus proximity
    if len(df_old_burglary) > 0:
        df_old_burglary_filtered = filter_by_campus(df_old_burglary, 'latitude', 'longitude')
    else:
        df_old_burglary_filtered = df_old_burglary
    
    if len(df_burglary) > 0:
        df_new_burglary_filtered = filter_by_campus(df_burglary, 'Latitude', 'Longitude')
    else:
        df_new_burglary_filtered = df_burglary
    
    # Combine
    print("\n5. Combining burglary datasets...")
    total_burglary = len(df_old_burglary_filtered) + len(df_new_burglary_filtered)
    print(f"   Total burglary incidents near campuses: {total_burglary}")
    print(f"   - 2009-2020: {len(df_old_burglary_filtered)}")
    print(f"   - 2021-2025: {len(df_new_burglary_filtered)}")
    
    # Campus distribution
    print("\n6. Burglary incidents by campus:")
    if len(df_old_burglary_filtered) > 0:
        old_campus = df_old_burglary_filtered['nearest_campus'].value_counts()
        print("\n   2009-2020:")
        for campus, count in old_campus.items():
            campus_name = CAMPUSES[campus]['name']
            print(f"     {campus_name}: {count}")
    
    if len(df_new_burglary_filtered) > 0:
        new_campus = df_new_burglary_filtered['nearest_campus'].value_counts()
        print("\n   2021-2025:")
        for campus, count in new_campus.items():
            campus_name = CAMPUSES[campus]['name']
            print(f"     {campus_name}: {count}")
    
    # Temporal trends
    print("\n7. Temporal trends:")
    
    if len(df_old_burglary_filtered) > 0:
        df_old_burglary_filtered['year'] = df_old_burglary_filtered['report_date'].dt.year
        old_yearly = df_old_burglary_filtered['year'].value_counts().sort_index()
        print("\n   2009-2020 yearly counts:")
        for year, count in old_yearly.items():
            print(f"     {year}: {count}")
    
    if len(df_new_burglary_filtered) > 0:
        df_new_burglary_filtered['year'] = df_new_burglary_filtered['report_date'].dt.year
        new_yearly = df_new_burglary_filtered['year'].value_counts().sort_index()
        print("\n   2021-2025 yearly counts:")
        for year, count in new_yearly.items():
            print(f"     {year}: {count}")
    
    return df_old_burglary_filtered, df_new_burglary_filtered


def save_preliminary_data(df_old, df_new):
    """Save preliminary filtered data."""
    print("\n8. Saving preliminary data...")
    
    output_dir = PROCESSED_DATA_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    if len(df_old) > 0:
        df_old_save = df_old[['report_date', 'crime_type', 'latitude', 'longitude', 
                               'nearest_campus', 'distance_to_nearest', 'source']].copy()
        df_old_save.to_csv(output_dir / 'burglary_2009_2020_preliminary.csv', index=False)
        print(f"   Saved: {output_dir / 'burglary_2009_2020_preliminary.csv'}")
    
    if len(df_new) > 0:
        df_new_save = df_new[['report_date', 'crime_type', 'Latitude', 'Longitude', 
                               'nearest_campus', 'distance_to_nearest', 'source']].copy()
        df_new_save.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
        df_new_save.to_csv(output_dir / 'burglary_2021_2025_preliminary.csv', index=False)
        print(f"   Saved: {output_dir / 'burglary_2021_2025_preliminary.csv'}")


def main():
    """Run quick exploration."""
    df_old, df_burglary, df_larceny = load_and_explore_data()
    
    if not df_old.empty or not df_burglary.empty:
        df_old_burglary, df_new_burglary = analyze_burglary_data(df_old, df_burglary)
        save_preliminary_data(df_old_burglary, df_new_burglary)
    
    print("\n" + "=" * 80)
    print("QUICK EXPLORATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Install geopandas: pip install geopandas")
    print("2. Run full data processing: python scripts/01_data_processing.py")
    print("3. Run EDA: python scripts/02_exploratory_analysis.py")


if __name__ == "__main__":
    main()

