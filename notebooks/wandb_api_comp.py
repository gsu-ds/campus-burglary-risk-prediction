import wandb
import pandas as pd

api = wandb.Api()

ENTITY = "joshuadariuspina-georgia-state-university"

PROJECTS = [
    "rolling-cv",
    "campus-burglary-risk",
    "updated_burglary_forecast",
    "atl-crime-hourly-forecast",
    "atl-crime-risk"
]

all_rows = []

for project in PROJECTS:
    runs = api.runs(f"{ENTITY}/{project}")
    
    for r in runs:
        all_rows.append({
            "project": project,
            "run_id": r.id,
            "model": r.config.get("model"),
            "dataset": r.config.get("dataset"),
            "MAE": r.summary.get("Mean_MAE"),
            "RMSE": r.summary.get("Mean_RMSE"),
            "R2": r.summary.get("Mean_R2"),
            "MAPE": r.summary.get("Mean_MAPE"),
            "timestamp": r.created_at,
            "url": r.url,
        })

df_all = pd.DataFrame(all_rows)
df_all
