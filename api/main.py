from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import pandas as pd
import joblib
import json
import numpy as np

app = FastAPI(
    title="Atlanta Burglary Prediction API",
    description="Serve ML predictions for burglary risk across Atlanta NPUs",
    version="1.0.0"
)

# NOTE: These paths reference the mounted location inside the API container
ARTIFACTS_DIR = Path("artifacts")
TEST_RESULTS_DIR = Path("reports/test_results")

class PredictionRequest(BaseModel):
    npu: str
    hour_ts: str
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    npu: str
    hour_ts: str
    predicted_count: float
    model: str
    dataset: str

class ModelInfo(BaseModel):
    dataset: str
    model_name: str
    cv_mean_r2: float
    cv_mean_mae: float
    cv_mean_rmse: float
    n_features: int

@app.get("/")
def root():
    return {
        "message": "Atlanta Burglary Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/models": "List available models with metrics",
            "/forecast/{dataset}/{model}": "Get historical predictions",
            "/predict/{dataset}": "Make new prediction with best model",
            "/health": "Health check"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "artifacts_exist": ARTIFACTS_DIR.exists(),
        "test_results_exist": TEST_RESULTS_DIR.exists()
    }

@app.get("/models", response_model=List[ModelInfo])
def list_models():
    """List all available trained models with their performance metrics."""
    models = []
    
    # Iterates through dataset directories inside the artifacts folder
    for dataset_dir in ARTIFACTS_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        
        # Looks for the metadata file
        for metadata_file in dataset_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                models.append(ModelInfo(
                    dataset=dataset_name,
                    model_name=metadata["best_model"],
                    cv_mean_r2=metadata["CV_Mean_R2"],
                    cv_mean_mae=metadata["CV_Mean_MAE"],
                    cv_mean_rmse=metadata["CV_Mean_RMSE"],
                    n_features=len(metadata["feature_cols"])
                ))
            except Exception as e:
                # Silently skip directories or metadata files that fail to load
                continue
    
    # Sorts models by R2 for easy selection on the dashboard
    return sorted(models, key=lambda x: x.cv_mean_r2, reverse=True)

@app.get("/forecast/{dataset}/{model}")
def get_forecast(
    dataset: str,
    model: str,
    npu: Optional[str] = Query(None, description="Filter by NPU"),
    limit: int = Query(10000, ge=1, le=1000000) 
):
    """Get historical predictions from rolling CV results."""
    
    # Accesses prediction files generated during CV runs
    pred_file = TEST_RESULTS_DIR / dataset / "cv_results" / "predictions" / f"{model}_all_predictions.csv"
    
    if not pred_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Predictions not found for dataset '{dataset}' and model '{model}'"
        )
    
    try:
        df = pd.read_csv(pred_file)
        
        # Normalizes column names for consistent API response structure
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["hour_ts", "datetime"]:
                rename_map[col] = "hour_ts"
            elif col_lower in ["burglary_count", "crime_count", "actual", "y_true"]:
                rename_map[col] = "actual"
            elif col_lower in ["predicted", "pred"]:
                rename_map[col] = "predicted"
        
        df = df.rename(columns=rename_map)
        
        if npu:
            df = df[df["npu"] == npu]
        
        # Limit result size for safety, although Streamlit requests a large limit
        df = df.head(limit)
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for NPU '{npu}'" if npu else "No predictions found"
            )
        
        # Convert DataFrame rows to a list of dicts for JSON response
        results = []
        for _, row in df.iterrows():
            results.append({
                "hour_ts": str(row["hour_ts"]),
                "npu": str(row["npu"]),
                "actual": float(row["actual"]),
                "predicted": float(row["predicted"]),
                "model": model,
                "dataset": dataset
            })
        
        return {
            "dataset": dataset,
            "model": model,
            "count": len(results),
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{dataset}", response_model=PredictionResponse)
def predict(dataset: str, request: PredictionRequest):
    """
    Make a new prediction using the best model for the dataset.
    This uses the robust loading logic: metadata -> model name -> joblib file.
    """
    
    dataset_dir = ARTIFACTS_DIR / dataset
    if not dataset_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset}' not found"
        )
    
    # 1. Find and load the metadata to get the BEST model name
    metadata_files = list(dataset_dir.glob("*_metadata.json"))
    
    if not metadata_files:
        raise HTTPException(
            status_code=404,
            detail=f"Metadata not found for dataset '{dataset}'"
        )
    
    try:
        # Load metadata
        metadata_file_path = metadata_files[0]
        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)
        
        best_model_name = metadata["best_model"]
        
        # 2. Construct the EXACT model file path
        # Searches for a file matching *{ModelName}_best_model.joblib
        model_files = list(dataset_dir.glob(f"*{best_model_name}_best_model.joblib"))
        
        if not model_files:
            raise HTTPException(
                status_code=404, 
                detail=f"Model file matching '*{best_model_name}_best_model.joblib' not found."
            )
        
        model_file_path = model_files[0]

        # 3. Load the explicitly named model
        model = joblib.load(model_file_path)
        
        feature_cols = metadata["feature_cols"]
        
        # 4. Process input features for prediction
        input_df = pd.DataFrame([request.features])
        
        missing_features = set(feature_cols) - set(input_df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing_features}"
            )
        
        X = input_df[feature_cols].fillna(0)
        
        prediction = model.predict(X)[0]
        
        return PredictionResponse(
            npu=request.npu,
            hour_ts=request.hour_ts,
            predicted_count=float(prediction),
            model=best_model_name,
            dataset=dataset
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load or predict: {str(e)}")

@app.get("/datasets")
def list_datasets():
    """List available datasets."""
    datasets = []
    
    for dataset_dir in ARTIFACTS_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        metadata_files = list(dataset_dir.glob("*_metadata.json"))
        if metadata_files:
            try:
                with open(metadata_files[0], 'r') as f:
                    metadata = json.load(f)
                
                datasets.append({
                    "name": dataset_dir.name,
                    "best_model": metadata["best_model"],
                    "cv_r2": metadata["CV_Mean_R2"],
                    "n_rows": metadata["n_rows"],
                    "n_features": len(metadata["feature_cols"]),
                    "target": metadata["target_col"]
                })
            except:
                continue
    
    return {"datasets": datasets}

@app.get("/features/{dataset}")
def get_features(dataset: str):
    """Get required feature list for a dataset."""
    
    dataset_dir = ARTIFACTS_DIR / dataset
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")
    
    metadata_files = list(dataset_dir.glob("*_metadata.json"))
    if not metadata_files:
        raise HTTPException(status_code=404, detail=f"Metadata not found for '{dataset}'")
    
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
    
    return {
        "dataset": dataset,
        "model": metadata["best_model"],
        "features": metadata["feature_cols"],
        "feature_count": len(metadata["feature_cols"])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)