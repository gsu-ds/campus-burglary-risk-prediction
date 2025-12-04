from pathlib import Path
from typing import List, Optional
from .schemas import ForecastRequest
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# --- FastAPI app ---
app = FastAPI(
    title="Atlanta Burglary Risk API",
    description="Serve NPU-level burglary/larceny forecasts from the capstone project.",
    version="1.0.0",
)

# --- Paths to prediction CSVs (from your repo structure) ---
ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "december" / "cv_results" / "predictions"

# File names taken from your GitHub tree
MODEL_FILES = {
    "NaiveMean": "NaiveMean_all_predictions.csv",
    "NaiveLastHour": "NaiveLastHour_all_predictions.csv",
    "SeasonalWeekly": "SeasonalWeekly_all_predictions.csv",
    "RandomForest": "RandomForest_all_predictions.csv",
    "PoissonGLM": "PoissonGLM_all_predictions.csv",
    "XGBRegressor": "XGBRegressor_all_predictions.csv",
    "CatBoost": "CatBoost_all_predictions.csv",
}


class ForecastPoint(BaseModel):
    datetime: str
    npu: str
    actual: float
    predicted: float
    fold: Optional[str] = None


# ---------- Basic routes ----------

@app.get("/")
def root():
    """Simple landing route."""
    return {
        "message": "Atlanta Burglary Risk API",
        "docs": "/docs",
        "endpoints": ["/health", "/models", "/forecast"],
    }


@app.get("/health")
def health():
    """Healthcheck for monitoring."""
    return {"status": "ok"}


# ---------- Helper to load predictions ----------

def load_model_predictions(model: str) -> pd.DataFrame:
    """Load predictions CSV for a given model name and normalize columns."""
    if model not in MODEL_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")

    path = PRED_DIR / MODEL_FILES[model]
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Predictions file not found for model {model}: {path}",
        )

    df = pd.read_csv(path, parse_dates=["hour_ts"])

    # Standardize column names used by the API
    rename_map = {
        "hour_ts": "datetime",
        "burglary_count": "actual",
        # "predicted" is already the right name from the notebook
    }
    df = df.rename(columns=rename_map)

    # Sanity checks
    for col in ["datetime", "npu", "actual", "predicted"]:
        if col not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Column '{col}' missing from predictions file {path.name}",
            )

    return df


# ---------- Public API endpoints ----------

@app.get("/models")
def list_models() -> List[str]:
    """List models available through the API."""
    return sorted(MODEL_FILES.keys())


@app.get("/forecast", response_model=List[ForecastPoint])
def get_forecast(
    model: str = Query(..., description="Model name, e.g. 'RandomForest'"),
    npu: str = Query(..., description="NPU label, e.g. 'A' or 'M'"),
    max_points: int = Query(
        500,
        ge=1,
        le=2000,
        description="Maximum number of time points to return",
    ),
):
    """
    Return actual vs predicted counts over time for a given model + NPU.

    Uses the combined *_all_predictions.csv files from outputs/december/cv_results/predictions.
    """
    df = load_model_predictions(model)

    sub = df[df["npu"] == npu].sort_values("datetime")
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"No forecast rows for NPU {npu}")

    sub = sub.head(max_points)

    results: List[ForecastPoint] = []
    for _, row in sub.iterrows():
        results.append(
            ForecastPoint(
                datetime=row["datetime"].isoformat(),
                npu=str(row["npu"]),
                actual=float(row["actual"]),
                predicted=float(row["predicted"]),
                fold=row["fold"] if "fold" in sub.columns else None,
            )
        )

    return results
