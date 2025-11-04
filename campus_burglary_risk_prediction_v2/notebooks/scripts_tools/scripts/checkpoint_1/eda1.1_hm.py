import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("APD.csv", usecols=[4, 15, 16])  # E, P, Q by position
df.columns = ["DateTime", "Longitude", "Latitude"]

# Clean + features
df = df.dropna(subset=["DateTime", "Longitude", "Latitude"])
df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
df = df.dropna(subset=["DateTime"])

df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.to_period("M").astype(str)
df["DayOfWeek"] = df["DateTime"].dt.day_name()
df["Hour"] = df["DateTime"].dt.hour

# Simple plots
df = df[df["DateTime"] >= "2021-01-01"]
monthly = df["DateTime"].dt.to_period("M").value_counts().sort_index()
monthly.index = monthly.index.astype(str)
monthly.plot(kind="bar", figsize=(12,4), title="Monthly Burglary Incidents (2021–present)")
plt.tight_layout(); plt.show()

dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow = df.groupby("DayOfWeek").size().reindex(dow_order)
plt.figure(figsize=(8,4)); dow.plot(kind="bar"); plt.title("Incidents by Day of Week"); plt.tight_layout(); plt.show()

hourly = df.groupby("Hour").size()
plt.figure(figsize=(10,4)); hourly.plot(kind="bar"); plt.title("Incidents by Hour"); plt.tight_layout(); plt.show()

print("Total incidents:", len(df))
print("Peak month:", monthly.idxmax())
print("Peak day:", dow.idxmax())
print("Peak hour:", hourly.idxmax())
