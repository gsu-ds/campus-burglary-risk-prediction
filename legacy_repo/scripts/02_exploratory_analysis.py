"""
Exploratory Data Analysis (EDA) for Campus Crime Data

This script performs comprehensive exploratory analysis on the processed crime data:
1. Temporal analysis (trends over time, seasonality, day/time patterns)
2. Spatial analysis (hotspot identification, campus comparisons)
3. Crime type analysis
4. Statistical summaries and visualizations
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

from src.config.settings import PROCESSED_DATA_PATH, VISUALIZATIONS_OUTPUT_PATH, CAMPUSES
from src.utils.logger import setup_logger

logger = setup_logger("eda_analysis", "eda_analysis.log")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def load_processed_data():
    """Load processed data."""
    logger.info("Loading processed data...")
    
    # Load burglary data
    burglary_path = PROCESSED_DATA_PATH / "burglary_data_campus_filtered.csv"
    df_burglary = pd.read_csv(burglary_path)
    df_burglary["occurred_date"] = pd.to_datetime(df_burglary["occurred_date"])
    
    # Load all crime data
    all_crime_path = PROCESSED_DATA_PATH / "crime_data_campus_filtered.csv"
    df_all = pd.read_csv(all_crime_path)
    df_all["occurred_date"] = pd.to_datetime(df_all["occurred_date"])
    
    logger.info(f"Loaded {len(df_burglary)} burglary records")
    logger.info(f"Loaded {len(df_all)} total crime records")
    
    return df_burglary, df_all


def temporal_analysis(df, output_dir):
    """Perform temporal analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("TEMPORAL ANALYSIS")
    logger.info("=" * 80)
    
    # 1. Yearly trend
    logger.info("\n1. Analyzing yearly trends...")
    yearly_counts = df.groupby("year").size().reset_index(name="count")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_counts["year"], yearly_counts["count"], marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Incidents", fontsize=12)
    ax.set_title("Burglary Incidents by Year (Campus Areas)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "01_yearly_trend.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Yearly trend: {yearly_counts['year'].min()} to {yearly_counts['year'].max()}")
    logger.info(f"Peak year: {yearly_counts.loc[yearly_counts['count'].idxmax(), 'year']} "
                f"({yearly_counts['count'].max()} incidents)")
    
    # 2. Monthly patterns
    logger.info("\n2. Analyzing monthly patterns...")
    monthly_counts = df.groupby("month").size().reset_index(name="count")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(monthly_counts["month"], monthly_counts["count"], color="steelblue", alpha=0.7)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Number of Incidents", fontsize=12)
    ax.set_title("Burglary Incidents by Month (All Years)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    plt.tight_layout()
    plt.savefig(output_dir / "02_monthly_pattern.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    peak_month = monthly_counts.loc[monthly_counts["count"].idxmax(), "month"]
    logger.info(f"Peak month: {month_names[peak_month-1]} ({monthly_counts['count'].max()} incidents)")
    
    # 3. Day of week patterns
    logger.info("\n3. Analyzing day of week patterns...")
    dow_counts = df.groupby("day_of_week").size().reset_index(name="count")
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(dow_counts["day_of_week"], dow_counts["count"], color="coral", alpha=0.7)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("Number of Incidents", fontsize=12)
    ax.set_title("Burglary Incidents by Day of Week", fontsize=14, fontweight="bold")
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "03_day_of_week_pattern.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    peak_dow = dow_counts.loc[dow_counts["count"].idxmax(), "day_of_week"]
    logger.info(f"Peak day: {dow_names[peak_dow]} ({dow_counts['count'].max()} incidents)")
    
    # 4. Weekend vs Weekday
    logger.info("\n4. Comparing weekend vs weekday...")
    weekend_counts = df.groupby("is_weekend").size()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Weekday", "Weekend"]
    colors = ["#3498db", "#e74c3c"]
    ax.pie(weekend_counts.values, labels=labels, autopct="%1.1f%%", 
           colors=colors, startangle=90)
    ax.set_title("Burglary Incidents: Weekday vs Weekend", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "04_weekend_vs_weekday.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    weekday_pct = (weekend_counts[0] / weekend_counts.sum()) * 100
    logger.info(f"Weekday incidents: {weekend_counts[0]} ({weekday_pct:.1f}%)")
    logger.info(f"Weekend incidents: {weekend_counts[1]} ({100-weekday_pct:.1f}%)")
    
    # 5. Quarterly trends
    logger.info("\n5. Analyzing quarterly patterns...")
    quarterly_counts = df.groupby(["year", "quarter"]).size().reset_index(name="count")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for quarter in range(1, 5):
        q_data = quarterly_counts[quarterly_counts["quarter"] == quarter]
        ax.plot(q_data["year"], q_data["count"], marker="o", label=f"Q{quarter}", linewidth=2)
    
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Incidents", fontsize=12)
    ax.set_title("Burglary Incidents by Quarter", fontsize=14, fontweight="bold")
    ax.legend(title="Quarter")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "05_quarterly_trends.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 6. Recent trend (2021-2025)
    logger.info("\n6. Analyzing recent trends (2021-2025)...")
    recent_df = df[df["year"] >= 2021]
    
    if len(recent_df) > 0:
        # Monthly trend for recent years
        recent_monthly = recent_df.groupby([recent_df["occurred_date"].dt.to_period("M")]).size()
        recent_monthly.index = recent_monthly.index.to_timestamp()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(recent_monthly.index, recent_monthly.values, linewidth=2, color="darkgreen")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Number of Incidents", fontsize=12)
        ax.set_title("Monthly Burglary Incidents (2021-2025)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "06_recent_monthly_trend.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Recent trend: {len(recent_df)} incidents from 2021-2025")


def spatial_analysis(df, output_dir):
    """Perform spatial analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("SPATIAL ANALYSIS")
    logger.info("=" * 80)
    
    # 1. Campus comparison
    logger.info("\n1. Comparing incidents across campuses...")
    campus_counts = df.groupby("campus_code").size().reset_index(name="count")
    campus_counts = campus_counts.sort_values("count", ascending=False)
    
    # Add campus names
    campus_name_map = {code: info["name"] for code, info in CAMPUSES.items()}
    campus_counts["campus_name"] = campus_counts["campus_code"].map(campus_name_map)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(campus_counts["campus_name"], campus_counts["count"], color="teal", alpha=0.7)
    ax.set_xlabel("Number of Incidents", fontsize=12)
    ax.set_ylabel("Campus", fontsize=12)
    ax.set_title("Burglary Incidents by Campus (1-Mile Radius)", fontsize=14, fontweight="bold")
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f"{int(width)}", 
                ha="left", va="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "07_campus_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info("\nIncidents by campus:")
    for _, row in campus_counts.iterrows():
        pct = (row["count"] / campus_counts["count"].sum()) * 100
        logger.info(f"  {row['campus_name']}: {row['count']} ({pct:.1f}%)")
    
    # 2. Campus temporal comparison
    logger.info("\n2. Analyzing temporal trends by campus...")
    campus_yearly = df.groupby(["campus_code", "year"]).size().reset_index(name="count")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for campus_code in campus_yearly["campus_code"].unique():
        campus_data = campus_yearly[campus_yearly["campus_code"] == campus_code]
        ax.plot(campus_data["year"], campus_data["count"], 
                marker="o", label=campus_name_map[campus_code], linewidth=2)
    
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Incidents", fontsize=12)
    ax.set_title("Burglary Trends by Campus", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "08_campus_temporal_trends.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Heatmap: Campus vs Month
    logger.info("\n3. Creating campus-month heatmap...")
    campus_month = df.groupby(["campus_code", "month"]).size().reset_index(name="count")
    heatmap_data = campus_month.pivot(index="campus_code", columns="month", values="count").fillna(0)
    
    # Rename index with campus names
    heatmap_data.index = heatmap_data.index.map(campus_name_map)
    heatmap_data.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd", 
                cbar_kws={"label": "Number of Incidents"}, ax=ax)
    ax.set_title("Burglary Incidents: Campus vs Month", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Campus", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "09_campus_month_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def statistical_summary(df_burglary, df_all):
    """Generate statistical summaries."""
    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL SUMMARY")
    logger.info("=" * 80)
    
    # Overall statistics
    logger.info("\n1. Overall Statistics (Burglary):")
    logger.info(f"   Total incidents: {len(df_burglary)}")
    logger.info(f"   Date range: {df_burglary['occurred_date'].min()} to {df_burglary['occurred_date'].max()}")
    logger.info(f"   Number of campuses: {df_burglary['campus_code'].nunique()}")
    
    # Yearly statistics
    logger.info("\n2. Yearly Statistics:")
    yearly_stats = df_burglary.groupby("year").agg({
        "report_number": "count",
    }).reset_index()
    yearly_stats.columns = ["year", "count"]
    
    logger.info(f"   Mean incidents per year: {yearly_stats['count'].mean():.1f}")
    logger.info(f"   Std deviation: {yearly_stats['count'].std():.1f}")
    logger.info(f"   Min year: {yearly_stats['count'].min()} incidents")
    logger.info(f"   Max year: {yearly_stats['count'].max()} incidents")
    
    # Campus statistics
    logger.info("\n3. Campus Statistics:")
    campus_stats = df_burglary.groupby("campus_code").agg({
        "report_number": "count",
    }).reset_index()
    campus_stats.columns = ["campus_code", "count"]
    
    campus_name_map = {code: info["name"] for code, info in CAMPUSES.items()}
    campus_stats["campus_name"] = campus_stats["campus_code"].map(campus_name_map)
    
    for _, row in campus_stats.iterrows():
        pct = (row["count"] / campus_stats["count"].sum()) * 100
        logger.info(f"   {row['campus_name']}: {row['count']} ({pct:.1f}%)")
    
    # Crime type comparison
    logger.info("\n4. Crime Type Distribution (All):")
    crime_dist = df_all["crime_category"].value_counts()
    for crime_type, count in crime_dist.items():
        pct = (count / len(df_all)) * 100
        logger.info(f"   {crime_type.capitalize()}: {count} ({pct:.1f}%)")
    
    # Recent trend analysis (2021-2025)
    logger.info("\n5. Recent Trend (2021-2025):")
    recent_df = df_burglary[df_burglary["year"] >= 2021]
    
    if len(recent_df) > 0:
        recent_yearly = recent_df.groupby("year").size()
        logger.info(f"   Total incidents (2021-2025): {len(recent_df)}")
        logger.info(f"   Average per year: {len(recent_df) / len(recent_yearly):.1f}")
        
        for year, count in recent_yearly.items():
            logger.info(f"   {year}: {count} incidents")


def main():
    """Run complete EDA pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = VISUALIZATIONS_OUTPUT_PATH / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_burglary, df_all = load_processed_data()
    
    # Perform analyses
    temporal_analysis(df_burglary, output_dir)
    spatial_analysis(df_burglary, output_dir)
    statistical_summary(df_burglary, df_all)
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPLORATORY DATA ANALYSIS COMPLETE")
    logger.info(f"Visualizations saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

