import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, asin, sqrt
from datetime import datetime, date

# ============================================
# 0. LOAD DATA
# ============================================
df = pd.read_csv("APD_01:01:2020_12:31:2024.csv", low_memory=False)

# Parse dates
df["ReportDate"] = pd.to_datetime(df["ReportDate"], errors="coerce")
df["OccurredFromDate"] = pd.to_datetime(df["OccurredFromDate"], errors="coerce")

# Keep only rows with lat/lon
df = df.dropna(subset=["Latitude", "x", "y"])

# ============================================
# 1. FILTER TO 1-MILE RADIUS OF GSU
# GSU (Downtown) approx: 33.7529, -84.3860
# We'll use haversine
# ============================================
GSU_LAT = 33.7529
GSU_LON = -84.3860

def haversine(lat1, lon1, lat2, lon2):
    """distance in miles"""
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    miles = 3956 * c   # radius of earth in miles
    return miles

df["dist_from_gsu_mi"] = df.apply(
    lambda r: haversine(GSU_LAT, GSU_LON, r["Latitude"], r["x"]), axis=1
)

df_gsu = df[df["dist_from_gsu_mi"] <= 1.0].copy()

print(f"Total records in file: {len(df):,}")
print(f"Records within 1 mile of GSU: {len(df_gsu):,}")

# ============================================
# 2. FEATURE ENGINEERING (TIME)
# ============================================
df_gsu["year"] = df_gsu["ReportDate"].dt.year
df_gsu["month"] = df_gsu["ReportDate"].dt.month
df_gsu["month_name"] = df_gsu["ReportDate"].dt.month_name()
df_gsu["date"] = df_gsu["ReportDate"].dt.date
df_gsu["hour"] = df_gsu["OccurredFromDate"].dt.hour
df_gsu["dow"] = df_gsu["ReportDate"].dt.day_name()
df_gsu["is_weekend"] = df_gsu["dow"].isin(["Saturday", "Sunday"])

# ============================================
# 3. ACADEMIC / SEMESTER FLAGS (GSU-LIKE)
# You can adjust the dates if your semester differs
# ============================================
def semester_flag(dt):
    """Return 'spring', 'summer', 'fall', or 'break' for a given date."""
    if pd.isna(dt):
        return "unknown"
    m = dt.month
    d = dt.day
    # rough GSU pattern:
    # Spring: Jan 6 – May 10
    # Summer: May 15 – Aug 5
    # Fall:   Aug 15 – Dec 15
    # everything else = break
    if (m == 1 and d >= 6) or (2 <= m <= 4) or (m == 5 and d <= 10):
        return "spring"
    if (m == 5 and d >= 15) or (m == 6) or (m == 7) or (m == 8 and d <= 5):
        return "summer"
    if (m == 8 and d >= 15) or (m in [9, 10, 11]) or (m == 12 and d <= 15):
        return "fall"
    return "break"

df_gsu["semester"] = df_gsu["ReportDate"].apply(semester_flag)

# ============================================
# 4. EVENT DATES (MBS + STATE FARM + BIG STUFF)
# pulled from public calendars; expand as needed
# ============================================
event_dates = [
    # 2024 examples from public pages (concerts, Copa opener, NBA games) :contentReference[oaicite:1]{index=1}
    date(2024, 6, 20),   # Copa America opener at MBS
    date(2024, 8, 11),   # (example) big concert at MBS :contentReference[oaicite:2]{index=2}
    date(2024, 5, 18),   # Luis Miguel at State Farm Arena :contentReference[oaicite:3]{index=3}
    date(2024, 5, 29),   # TXT concert at State Farm Arena :contentReference[oaicite:4]{index=4}
    # add Hawks home games / MLS / Falcons as needed
]

df_gsu["is_event_day"] = df_gsu["date"].isin(event_dates)

# ============================================
# 5. TEMPORAL EDA
# ============================================

# 5.1 Daily trend (with rolling mean)
daily_counts = (
    df_gsu.groupby("date")
          .size()
          .reset_index(name="crime_count")
          .sort_values("date")
)

daily_counts["roll7"] = daily_counts["crime_count"].rolling(7, min_periods=1).mean()

plt.figure(figsize=(14, 5))
plt.plot(daily_counts["date"], daily_counts["crime_count"], label="Daily crimes", alpha=0.4)
plt.plot(daily_counts["date"], daily_counts["roll7"], label="7-day rolling avg", linewidth=2)
plt.title("Daily Crimes within 1 Mile of GSU")
plt.xlabel("Date")
plt.ylabel("Number of crimes")
plt.legend()
plt.tight_layout()
plt.show()

# 5.2 Day-of-week pattern
dow_counts = (
    df_gsu.groupby("dow")
          .size()
          .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
)
plt.figure(figsize=(8, 4))
dow_counts.plot(kind="bar", edgecolor="black")
plt.title("Crimes by Day of Week (1 mi of GSU)")
plt.xlabel("Day of Week")
plt.ylabel("Number of crimes")
plt.tight_layout()
plt.show()

# 5.3 Hour-of-day vs weekend
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df_gsu,
    x="hour",
    hue="is_weekend",
    palette="Set2"
)
plt.title("Crimes by Hour: Weekend vs Weekday (1 mi of GSU)")
plt.xlabel("Hour of Day")
plt.ylabel("Crime Count")
plt.legend(title="Weekend")
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 5))
sns.heatmap(pivot_hour_dow, cmap="YlOrRd")
plt.title("Heatmap: Day of Week vs Hour (1 mi of GSU)")
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.show()

# 5.4 Semester vs break
sem_counts = df_gsu.groupby("semester").size().reindex(["spring","summer","fall","break"])
plt.figure(figsize=(6,4))
sem_counts.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Crimes by Semester Period (1 mi of GSU)")
plt.xlabel("Semester Period")
plt.ylabel("Crime Count")
plt.tight_layout()
plt.show()


# ============================================
# 6. SPATIAL-LIKE EDA (within buffer)
# we don't have zip, so use NhoodName
# ============================================
top_nhood = (
    df_gsu["NhoodName"]
    .fillna("Unknown")
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,5))
top_nhood.plot(kind="bar", edgecolor="black")
plt.title("Top 10 Neighborhoods within 1 Mile of GSU (by crime reports)")
plt.xlabel("Neighborhood")
plt.ylabel("Crime Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ============================================
# 7. SIMPLE RELATIONSHIPS / CORRELATIONS
# (hour + weekend + event)
# ============================================
# create a small modeling-style frame
eda_df = df_gsu[["crime_description" if "crime_description" in df_gsu.columns else "Crime_Against",
                 "hour", "is_weekend", "is_event_day", "semester"]].copy()

# crosstab: weekend x event
ct_weekend_event = pd.crosstab(df_gsu["is_weekend"], df_gsu["is_event_day"])
print("\nWeekend vs Event-day crosstab:\n", ct_weekend_event)

# hour distribution for event vs non-event
plt.figure(figsize=(10,5))
sns.kdeplot(data=df_gsu, x="hour", hue="is_event_day", common_norm=False)
plt.title("Hour-of-Day Distribution: Event vs Non-Event Days (1 mi of GSU)")
plt.tight_layout()
plt.show()
