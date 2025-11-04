"""
Data Processing Pipeline - Load, Clean, and Filter Crime Data

This script performs the complete data processing pipeline:
1. Load raw APD crime data
2. Clean and standardize the data
3. Apply spatial filtering (1-mile radius from campuses)
4. Save processed data for EDA and modeling
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import geopandas as gpd
from src.data_processing import APDDataLoader, DataCleaner, SpatialFilter
from src.config.settings import PROCESSED_DATA_PATH
from src.utils.logger import setup_logger

logger = setup_logger("data_processing_pipeline", "data_processing.log")


def main():
    """Run the complete data processing pipeline."""
    
    logger.info("=" * 80)
    logger.info("STARTING DATA PROCESSING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Load all data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading raw data")
    logger.info("=" * 80)
    
    loader = APDDataLoader()
    df_raw = loader.load_all_data()
    
    logger.info(f"Raw data shape: {df_raw.shape}")
    logger.info(f"Date range: {df_raw['report_date'].min()} to {df_raw['report_date'].max()}")
    
    # Step 2: Clean data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Cleaning and standardizing data")
    logger.info("=" * 80)
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data(df_raw)
    
    logger.info(f"Cleaned data shape: {df_clean.shape}")
    logger.info(f"\nCrime category distribution:")
    logger.info(df_clean["crime_category"].value_counts())
    
    # Step 3: Convert to GeoDataFrame
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Converting to GeoDataFrame")
    logger.info("=" * 80)
    
    gdf_clean = loader.convert_to_geodataframe(df_clean)
    logger.info(f"GeoDataFrame created with CRS: {gdf_clean.crs}")
    
    # Step 4: Apply spatial filtering
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Applying spatial filtering (1-mile radius)")
    logger.info("=" * 80)
    
    spatial_filter = SpatialFilter()
    
    # Filter for campus proximity
    gdf_filtered = spatial_filter.filter_by_campus_proximity(gdf_clean)
    logger.info(f"Incidents near campuses: {len(gdf_filtered)}")
    
    # Assign nearest campus
    gdf_filtered = spatial_filter.assign_nearest_campus(gdf_filtered)
    
    # Get campus statistics
    campus_stats = spatial_filter.get_campus_statistics(gdf_filtered)
    logger.info("\nCampus statistics:")
    logger.info(campus_stats)
    
    # Step 5: Filter for burglary (primary focus)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Creating burglary-focused dataset")
    logger.info("=" * 80)
    
    gdf_burglary = cleaner.filter_by_crime_type(gdf_filtered, ["burglary"])
    logger.info(f"Burglary incidents near campuses: {len(gdf_burglary)}")
    
    # Step 6: Save processed data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Saving processed data")
    logger.info("=" * 80)
    
    # Save all campus-filtered data
    output_path_all = PROCESSED_DATA_PATH / "crime_data_campus_filtered.geojson"
    gdf_filtered.to_file(output_path_all, driver="GeoJSON")
    logger.info(f"Saved all campus-filtered data to: {output_path_all}")
    
    # Save burglary-only data
    output_path_burglary = PROCESSED_DATA_PATH / "burglary_data_campus_filtered.geojson"
    gdf_burglary.to_file(output_path_burglary, driver="GeoJSON")
    logger.info(f"Saved burglary data to: {output_path_burglary}")
    
    # Save as CSV for easy access
    csv_path_all = PROCESSED_DATA_PATH / "crime_data_campus_filtered.csv"
    df_to_save = pd.DataFrame(gdf_filtered.drop(columns="geometry"))
    df_to_save["latitude"] = gdf_filtered.geometry.y
    df_to_save["longitude"] = gdf_filtered.geometry.x
    df_to_save.to_csv(csv_path_all, index=False)
    logger.info(f"Saved CSV to: {csv_path_all}")
    
    csv_path_burglary = PROCESSED_DATA_PATH / "burglary_data_campus_filtered.csv"
    df_burglary = pd.DataFrame(gdf_burglary.drop(columns="geometry"))
    df_burglary["latitude"] = gdf_burglary.geometry.y
    df_burglary["longitude"] = gdf_burglary.geometry.x
    df_burglary.to_csv(csv_path_burglary, index=False)
    logger.info(f"Saved burglary CSV to: {csv_path_burglary}")
    
    # Save campus buffers
    buffer_path = PROCESSED_DATA_PATH / "campus_buffers.geojson"
    spatial_filter.export_campus_buffers(str(buffer_path))
    logger.info(f"Saved campus buffers to: {buffer_path}")
    
    # Save campus statistics
    stats_path = PROCESSED_DATA_PATH / "campus_statistics.csv"
    campus_stats.to_csv(stats_path, index=False)
    logger.info(f"Saved campus statistics to: {stats_path}")
    
    # Step 7: Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"Total raw records: {len(df_raw)}")
    logger.info(f"After cleaning: {len(df_clean)}")
    logger.info(f"Near campuses (all crimes): {len(gdf_filtered)}")
    logger.info(f"Burglary incidents near campuses: {len(gdf_burglary)}")
    
    logger.info("\nDate range of processed data:")
    logger.info(f"  Start: {gdf_burglary['occurred_date'].min()}")
    logger.info(f"  End: {gdf_burglary['occurred_date'].max()}")
    
    logger.info("\nIncidents by campus:")
    for _, row in campus_stats.iterrows():
        logger.info(f"  {row['campus_name']}: {row['incident_count']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DATA PROCESSING PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

