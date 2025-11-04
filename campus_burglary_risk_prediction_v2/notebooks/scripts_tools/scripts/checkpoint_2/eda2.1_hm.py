import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# LOAD & PREP DATA
# =========================================================
df = pd.read_csv("APD_01:01:2020_12:31:2024.csv", low_memory=False)

# Parse date columns
df["ReportDate"] = pd.to_datetime(df["ReportDate"], errors="coerce")
df["OccurredFromDate"] = pd.to_datetime(df["OccurredFromDate"], errors="coerce")

# =========================================================
# 1) CRIMES BY YEAR AND MONTH
# =========================================================
df["Year"] = df["ReportDate"].dt.year
df["Month"] = df["ReportDate"].dt.month_name()

# Group by year/month
by_ym = (
    df.groupby(["Year", "Month"])
      .size()
      .reset_index(name="CrimeCount")
)

# Order months Jan→Dec
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
by_ym["Month"] = pd.Categorical(by_ym["Month"], categories=month_order, ordered=True)
by_ym = by_ym.sort_values(["Year", "Month"])

# Plot
plt.figure(figsize=(12, 6))
for year, sub in by_ym.groupby("Year"):
    plt.plot(sub["Month"], sub["CrimeCount"], marker="o", label=year)

plt.title("Crimes by Year and Month (Atlanta PD 2020–2024)")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45)
plt.legend(title="Year")
plt.tight_layout()
plt.show()

# =========================================================
# 2) CRIMES BY HOUR
# =========================================================
df["Hour"] = df["OccurredFromDate"].dt.hour
by_hour = (
    df.groupby("Hour")
      .size()
      .reset_index(name="CrimeCount")
      .sort_values("Hour")
)

plt.figure(figsize=(10, 5))
plt.bar(by_hour["Hour"], by_hour["CrimeCount"], edgecolor="black")
plt.title("Crimes by Hour of Day")
plt.xlabel("Hour (0–23)")
plt.ylabel("Number of Crimes")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

# =========================================================
# 3) CRIMES BY NEIGHBORHOOD (NhoodName)
# =========================================================
nhood_counts = (
    df["NhoodName"]
    .fillna("Unknown")
    .value_counts()
    .head(15)
)

plt.figure(figsize=(12, 6))
nhood_counts.plot(kind="bar", edgecolor="black", color="teal")
plt.title("Top 15 Neighborhoods by Crime Count (2020–2024)")
plt.xlabel("Neighborhood")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =========================================================
# 4) CRIMES BY MONTH/YEAR (1/2020 → 12/2024)
# =========================================================
df["Year"] = df["ReportDate"].dt.year
df["MonthNum"] = df["ReportDate"].dt.month

monthly_counts = (
    df.groupby(["Year", "MonthNum"])
      .size()
      .reset_index(name="CrimeCount")
      .sort_values(["Year", "MonthNum"])
)

# Create labels like "1/2020", "2/2020", ..., "12/2024"
monthly_counts["Label"] = monthly_counts["MonthNum"].astype(str) + "/" + monthly_counts["Year"].astype(str)

plt.figure(figsize=(16, 6))
plt.bar(monthly_counts["Label"], monthly_counts["CrimeCount"], color="coral", edgecolor="black")
plt.title("Total Crimes per Month (Jan 2020 – Dec 2024)", fontsize=14)
plt.xlabel("Month/Year", fontsize=12)
plt.ylabel("Number of Crimes", fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()
