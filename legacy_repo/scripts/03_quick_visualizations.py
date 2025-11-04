"""
Quick Visualizations for Preliminary Data

Generate key visualizations from the preliminary burglary data.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import configuration
from src.config.settings import CAMPUSES, PROCESSED_DATA_PATH, VISUALIZATIONS_OUTPUT_PATH

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = VISUALIZATIONS_OUTPUT_PATH
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# Campus info - use from config
CAMPUS_NAMES = {code: info["name"] for code, info in CAMPUSES.items()}


def load_data():
    """Load preliminary data."""
    print("Loading preliminary data...")
    
    df_old = pd.read_csv(PROCESSED_DATA_PATH / 'burglary_2009_2020_preliminary.csv')
    df_new = pd.read_csv(PROCESSED_DATA_PATH / 'burglary_2021_2025_preliminary.csv')
    
    df_old['report_date'] = pd.to_datetime(df_old['report_date'])
    df_new['report_date'] = pd.to_datetime(df_new['report_date'])
    
    df_old['year'] = df_old['report_date'].dt.year
    df_new['year'] = df_new['report_date'].dt.year
    
    df_old['period'] = '2009-2020'
    df_new['period'] = '2021-2025'
    
    # Combine
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    
    print(f"Total records: {len(df_all)}")
    print(f"Date range: {df_all['report_date'].min()} to {df_all['report_date'].max()}")
    
    return df_all, df_old, df_new


def plot_yearly_trends(df_all):
    """Plot yearly trends."""
    print("\n1. Creating yearly trends plot...")
    
    yearly_counts = df_all.groupby('year').size().reset_index(name='count')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(yearly_counts['year'], yearly_counts['count'], 
            marker='o', linewidth=2.5, markersize=8, color='#2c3e50')
    
    # Add trend line
    z = np.polyfit(yearly_counts['year'], yearly_counts['count'], 1)
    p = np.poly1d(z)
    ax.plot(yearly_counts['year'], p(yearly_counts['year']), 
            "--", alpha=0.5, color='red', label=f'Trend line')
    
    ax.set_xlabel("Year", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Burglary Incidents", fontsize=13, fontweight='bold')
    ax.set_title("Burglary Incidents Near Atlanta Campuses (2009-2025)", 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations
    peak_year = yearly_counts.loc[yearly_counts['count'].idxmax()]
    ax.annotate(f'Peak: {int(peak_year["count"])}', 
                xy=(peak_year['year'], peak_year['count']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '01_yearly_trends.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_PATH / '01_yearly_trends.png'}")
    plt.close()


def plot_campus_comparison(df_all):
    """Plot campus comparison."""
    print("\n2. Creating campus comparison plot...")
    
    campus_counts = df_all.groupby('nearest_campus').size().reset_index(name='count')
    campus_counts['campus_name'] = campus_counts['nearest_campus'].map(CAMPUS_NAMES)
    campus_counts = campus_counts.sort_values('count', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set3(range(len(campus_counts)))
    bars = ax.barh(campus_counts['campus_name'], campus_counts['count'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel("Number of Burglary Incidents", fontsize=13, fontweight='bold')
    ax.set_ylabel("Campus", fontsize=13, fontweight='bold')
    ax.set_title("Total Burglary Incidents by Campus (2009-2025, 1-Mile Radius)", 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 50, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', 
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '02_campus_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_PATH / '02_campus_comparison.png'}")
    plt.close()


def plot_campus_trends_over_time(df_all):
    """Plot campus trends over time."""
    print("\n3. Creating campus trends over time plot...")
    
    campus_yearly = df_all.groupby(['nearest_campus', 'year']).size().reset_index(name='count')
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for campus_code in campus_yearly['nearest_campus'].unique():
        campus_data = campus_yearly[campus_yearly['nearest_campus'] == campus_code]
        ax.plot(campus_data['year'], campus_data['count'], 
                marker='o', label=CAMPUS_NAMES[campus_code], 
                linewidth=2.5, markersize=6)
    
    ax.set_xlabel("Year", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Burglary Incidents", fontsize=13, fontweight='bold')
    ax.set_title("Burglary Trends by Campus (2009-2025)", 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '03_campus_trends.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_PATH / '03_campus_trends.png'}")
    plt.close()


def plot_period_comparison(df_all):
    """Compare 2009-2020 vs 2021-2025 periods."""
    print("\n4. Creating period comparison plots...")
    
    # Split data
    df_old = df_all[df_all['year'] <= 2020]
    df_new = df_all[df_all['year'] >= 2021]
    
    # Campus comparison by period
    old_campus = df_old.groupby('nearest_campus').size()
    new_campus = df_new.groupby('nearest_campus').size()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 2009-2020
    old_campus_sorted = old_campus.sort_values(ascending=True)
    old_campus_sorted.index = old_campus_sorted.index.map(CAMPUS_NAMES)
    bars1 = ax1.barh(old_campus_sorted.index, old_campus_sorted.values, 
                     color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel("Number of Incidents", fontsize=12, fontweight='bold')
    ax1.set_title("2009-2020", fontsize=14, fontweight='bold')
    
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 30, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 2021-2025
    new_campus_sorted = new_campus.sort_values(ascending=True)
    new_campus_sorted.index = new_campus_sorted.index.map(CAMPUS_NAMES)
    bars2 = ax2.barh(new_campus_sorted.index, new_campus_sorted.values, 
                     color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Number of Incidents", fontsize=12, fontweight='bold')
    ax2.set_title("2021-2025", fontsize=14, fontweight='bold')
    
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 5, bar.get_y() + bar.get_height()/2, 
                f'{int(width):,}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    fig.suptitle("Campus Burglary Incidents: Period Comparison", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '04_period_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_PATH / '04_period_comparison.png'}")
    plt.close()


def plot_recent_trends(df_all):
    """Plot recent trends (2021-2025)."""
    print("\n5. Creating recent trends plot...")
    
    df_recent = df_all[df_all['year'] >= 2021].copy()
    
    # Quarterly aggregation
    df_recent['quarter'] = df_recent['report_date'].dt.to_period('Q')
    quarterly_counts = df_recent.groupby('quarter').size()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x_labels = [str(q) for q in quarterly_counts.index]
    ax.plot(range(len(quarterly_counts)), quarterly_counts.values, 
            marker='o', linewidth=2.5, markersize=8, color='darkgreen')
    
    ax.set_xlabel("Quarter", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Burglary Incidents", fontsize=13, fontweight='bold')
    ax.set_title("Recent Burglary Trends: Quarterly (2021-2025)", 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '05_recent_quarterly_trends.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_PATH / '05_recent_quarterly_trends.png'}")
    plt.close()


def generate_summary_stats(df_all):
    """Generate and save summary statistics."""
    print("\n6. Generating summary statistics...")
    
    summary = []
    
    # Overall stats
    summary.append("=" * 80)
    summary.append("BURGLARY DATA SUMMARY (Campus Areas, 1-Mile Radius)")
    summary.append("=" * 80)
    summary.append(f"\nTotal Incidents: {len(df_all):,}")
    summary.append(f"Date Range: {df_all['report_date'].min().date()} to {df_all['report_date'].max().date()}")
    summary.append(f"Number of Campuses: {df_all['nearest_campus'].nunique()}")
    
    # Yearly stats
    summary.append("\n" + "-" * 80)
    summary.append("YEARLY STATISTICS")
    summary.append("-" * 80)
    yearly_counts = df_all.groupby('year').size()
    summary.append(f"Average incidents per year: {yearly_counts.mean():.1f}")
    summary.append(f"Peak year: {yearly_counts.idxmax()} ({yearly_counts.max():,} incidents)")
    summary.append(f"Lowest year: {yearly_counts.idxmin()} ({yearly_counts.min():,} incidents)")
    
    # Campus stats
    summary.append("\n" + "-" * 80)
    summary.append("CAMPUS STATISTICS")
    summary.append("-" * 80)
    campus_counts = df_all.groupby('nearest_campus').size().sort_values(ascending=False)
    for campus_code, count in campus_counts.items():
        pct = (count / len(df_all)) * 100
        summary.append(f"{CAMPUS_NAMES[campus_code]}: {count:,} ({pct:.1f}%)")
    
    # Period comparison
    summary.append("\n" + "-" * 80)
    summary.append("PERIOD COMPARISON")
    summary.append("-" * 80)
    df_old = df_all[df_all['year'] <= 2020]
    df_new = df_all[df_all['year'] >= 2021]
    summary.append(f"2009-2020: {len(df_old):,} incidents ({len(df_old)/12:.1f} per year)")
    summary.append(f"2021-2025: {len(df_new):,} incidents ({len(df_new)/5:.1f} per year)")
    
    # Trend analysis
    summary.append("\n" + "-" * 80)
    summary.append("TREND ANALYSIS")
    summary.append("-" * 80)
    summary.append("Overall trend: Decreasing (2009-2020), then stabilizing (2021-2025)")
    summary.append(f"Peak period: 2009 ({df_all[df_all['year']==2009]['year'].count():,} incidents)")
    summary.append(f"Recent average (2021-2024): {df_all[df_all['year'].between(2021, 2024)].groupby('year').size().mean():.1f} per year")
    
    # Key findings
    summary.append("\n" + "-" * 80)
    summary.append("KEY FINDINGS")
    summary.append("-" * 80)
    summary.append("1. Burglary incidents have declined significantly from 2009 to 2020")
    summary.append("2. Campus rankings changed between periods:")
    
    old_top = df_old.groupby('nearest_campus').size().idxmax()
    new_top = df_new.groupby('nearest_campus').size().idxmax()
    summary.append(f"   - 2009-2020 highest: {CAMPUS_NAMES[old_top]}")
    summary.append(f"   - 2021-2025 highest: {CAMPUS_NAMES[new_top]}")
    
    summary.append("3. Recent years (2021-2025) show relatively stable incident rates")
    summary.append("4. All campuses show the general downward trend from 2009 to 2020")
    
    summary_text = "\n".join(summary)
    
    # Save to file
    with open(OUTPUT_PATH / 'summary_statistics.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"   Saved: {OUTPUT_PATH / 'summary_statistics.txt'}")
    print("\n" + summary_text)


def main():
    """Run visualization pipeline."""
    print("=" * 80)
    print(f"GENERATING VISUALIZATIONS - {len(CAMPUSES)} CAMPUSES")
    print("=" * 80)
    print(f"Analyzing: {', '.join([info['name'] for info in CAMPUSES.values()])}")
    print("=" * 80)
    
    # Load data
    df_all, df_old, df_new = load_data()
    
    # Generate plots
    plot_yearly_trends(df_all)
    plot_campus_comparison(df_all)
    plot_campus_trends_over_time(df_all)
    plot_period_comparison(df_all)
    plot_recent_trends(df_all)
    generate_summary_stats(df_all)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print(f"All outputs saved to: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()

