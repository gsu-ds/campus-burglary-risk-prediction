# wandb utils for logging

import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def log_panel_artifact(
    df: pd.DataFrame,
    name: str = "crime_panel_enriched",
    description: str = "Unified panel",
    output_path: Optional[Path] = None
) -> None:
    """
    Log panel DataFrame as W&B artifact.
    
    Parameters:
        df: Panel DataFrame to log
        name: Artifact name in W&B
        description: Artifact description
        output_path: Optional path to save parquet locally (default: panel_output.parquet)
    """
    if not WANDB_AVAILABLE:
        print("W&B not available. Skipping artifact logging.")
        return
    
    if wandb.run is None:
        print("Warning: No active W&B run. Call wandb.init() first.")
        return
    
    if output_path is None:
        output_path = Path("panel_output.parquet")
    
    df.to_parquet(output_path, index=False)
    
    artifact = wandb.Artifact(name, type="dataset", description=description)
    artifact.add_file(str(output_path))
    wandb.log_artifact(artifact)
    
    print(f"✔ W&B Artifact logged: {name}")


def log_multiple_panels(
    sparse_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    target_df: Optional[pd.DataFrame] = None,
    project_name: str = "atl-crime-prediction"
) -> None:
    """
    Log multiple panel types as separate W&B artifacts.
    
    Parameters:
        sparse_df: Sparse panel DataFrame
        dense_df: Dense panel DataFrame
        target_df: Optional target crimes DataFrame
        project_name: W&B project name
    """
    if not WANDB_AVAILABLE:
        print("W&B not available. Skipping artifact logging.")
        return
    
    if wandb.run is None:
        print(f"Initializing W&B run for project: {project_name}")
        wandb.init(project=project_name, job_type="panel_creation")
    
    log_panel_artifact(
        sparse_df,
        name="npu_sparse_panel",
        description="Sparse NPU × hour panel (only hours with incidents)",
        output_path=Path("sparse_panel.parquet")
    )
    
    log_panel_artifact(
        dense_df,
        name="npu_dense_panel",
        description="Dense NPU × hour panel (complete grid)",
        output_path=Path("dense_panel.parquet")
    )
    
    if target_df is not None:
        log_panel_artifact(
            target_df,
            name="target_crimes",
            description="Filtered target crimes dataset",
            output_path=Path("target_crimes.parquet")
        )
    
    print("✔ All panels logged to W&B")


__all__ = ["log_panel_artifact", "log_multiple_panels"]