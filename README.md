# Spatiotemporal Forecasting of Burglary Risk in Atlanta

**Data Science Capstone Project - Fall 2025**

A modular, end-to-end machine learning pipeline that transforms raw Atlanta Police Department data into actionable crime forecasts using spatial-temporal analysis, advanced feature engineering, and ensemble modeling.

---

## Project Overview

Routine Activity Theory posits that crime requires the convergence of a motivated offender, a suitable target, and the absence of a capable guardian. Applied to Atlanta's university districts, this framework highlights the urgent need for proactive rather than reactive safety measures.

This project focuses on forecasting **burglary and larceny risks** across Atlanta's 25 Neighborhood Planning Units (NPUs) to improve campus safety for Metro Atlanta colleges.

### Goals

- Forecast **hourly burglary risk** across Atlanta's 25 NPUs
- Deliver actionable insights through an **interactive dashboard**
- Support proactive safety strategies for university administrators, law enforcement, and students
- Benchmark multiple forecasting approaches (Naive Seasonal, Random Forest, XGBoost, CatBoost, LightGBM, ZIP, Prophet)

### Methodology

Leveraging the Atlanta Police Department's Open Data Portal (2021–Present), we constructed an automated ETL pipeline with:

- **Spatial enrichment**: NPU boundaries, campus proximity, zone mappings
- **Temporal features**: Cyclical encodings, holidays, hour blocks, lagged values
- **Weather integration**: Hourly temperature and precipitation data
- **Grid-based density**: Rolling statistics and spatial crime patterns

Models are evaluated using **R²**, **RMSE**, and **MAE** via rolling cross-validation to assess predictive performance and real-world utility for resource allocation.

---

## Benchmark Dataset

This project produces a publicly available benchmark dataset for spatiotemporal crime forecasting research.

### Quick Links

- **Dataset**: [Kaggle - Core Atlanta Burglary-Related Crimes (2021-2025)](https://www.kaggle.com/datasets/joshuapina/core-atlanta-burglary-related-crimes-2021-2025)
- **Full Documentation**: [DATASET_CARD.md](DATASET_CARD.md)
- **Reproducibility Guide**: [DATA_GENERATION_PIPELINE.md](DATA_GENERATION_PIPELINE.md)
- **Data Dictionary**: [data_dictionary.md](data_dictionary.md)

### Dataset Highlights

- **117,749 incidents**: Raw burglary/larceny reports (2021-2025)
- **99,965 observations**: Target and sparse panels (hourly NPU-level with incidents)
- **1,074,500 observations**: Dense panel (complete hourly NPU grid)
- **Rich features**: Temporal, spatial, and weather variables
- **Fully reproducible**: Complete pipeline in `atl_model_pipelines/`
- **Research-grade**: Comprehensive documentation and quality checks

### Reproduce the Dataset

```bash
# Run data generation pipeline
python -m atl_model_pipelines.ingestion.ingestion_master
python -m atl_model_pipelines.transform.transform_master
python -m atl_model_pipelines.validate.orchestrator

# Output: data/processed/npu_dense_panel.parquet
```

### Citation

If you use this dataset in research or publication:

```bibtex
@dataset{atlanta_burglary_2025,
  title={Core Atlanta Burglary-Related Crimes (2021-2025)},
  author={Madan, Gunn and Mohan, Harini and Piña, Joshua and Wu, Yuntian Robin},
  year={2025},
  institution={Georgia State University},
  url={https://www.kaggle.com/datasets/joshuapina/core-atlanta-burglary-related-crimes-2021-2025}
}
```

---

## Quick Start

### Option 1: GitHub Codespaces (Recommended)

1. Click the green **Code** button on this repository
2. Select **Open with Codespaces** → **New codespace**
3. Wait for environment to build (includes all dependencies)

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/gsu-ds/campus-burglary-risk-prediction.git
cd campus-burglary-risk-prediction

# Install dependencies (Python 3.10+)
pip install -r requirements.txt

# Optional: Install all model libraries
pip install scikit-learn xgboost lightgbm catboost prophet wandb
```

---

## Infrastructure & Tech Stack

| Component | Tool | Link |
|-----------|------|------|
| **Frontend Dashboard** | Streamlit/Render | [Live Demo](https://atl-crime-api.onrender.com/) |
| **Dataset** | Kaggle | [Dataset Page](https://www.kaggle.com/datasets/joshuapina/core-atlanta-burglary-related-crimes-2021-2025) |
| **Version Control** | GitHub | [Repository](https://github.com/gsu-ds/campus-burglary-risk-prediction) |
| **Database** | Supabase (PostgreSQL) | [Dashboard](https://supabase.com/dashboard/project/huhkmlefmbxxsgewvrgm/settings/general) |
| **Experiment Tracking** | Weights & Biases | [Project Board](https://wandb.ai/joshuadariuspina-georgia-state-university/atl-crime-hourly-forecast) |

---

## Team

**Data Science Team** (Alphabetically):  
Gunn Madan, Harini Mohan, Joshua Piña, Yuntian Wu

**Institution**: Georgia State University

---

## Contact & Support

### GitHub Issues
[Report bugs or request features](https://github.com/gsu-ds/campus-burglary-risk-prediction/issues)

### Email
- Joshua Piña: jpina4@student.gsu.edu
- Yuntian Wu: ywu49@student.gsu.edu
- Gunn Madan: gmadan1@student.gsu.edu
- Harini Mohan: hmohan1@student.gsu.edu

---

## References

1. Atlanta Police Department. (2025). Crime Data Downloads.
2. Open-Meteo. (2025). Historical Weather API.
3. City of Atlanta. (2025). NPU Boundaries Open Data.

---

## License

**Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
**Code**: MIT License