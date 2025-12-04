### Cyclical Encoding

Models can get confused by 11pm (2300) and midnight (0000), by using sin and cos we can map hours into a circle.

import numpy as np
import matplotlib.pyplot as plt

hours = np.arange(24)
hour_sin = np.sin(2 * np.pi * hours / 24)
hour_cos = np.cos(2 * np.pi * hours / 24)

# Visualize the circle
plt.figure(figsize=(8, 8))
plt.scatter(hour_cos, hour_sin, c=hours, cmap='twilight', s=200, edgecolor='black')
for i, h in enumerate(hours):
    plt.text(hour_cos[i], hour_sin[i], str(h), ha='center', va='center', fontsize=8)
plt.xlabel('hour_cos')
plt.ylabel('hour_sin')
plt.title('Hours Mapped to a Circle')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()

> Good for hour of day (24-hour cycle), day of week (7-day cycle), month of year (12-month cycle)

def create_cyclical_features(df, date_col='datetime'):
    """
    Creates cyclical encodings for temporal features.
    """
    df = df.copy()
    
    # Hour of day (0-23)
    df['hour'] = df[date_col].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month of year (1-12)
    df['month'] = df[date_col].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


> Optional for xgboost, catboost...but can help. Critical for linear models like glm, nn