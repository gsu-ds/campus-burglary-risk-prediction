# 📚 APD Crime Analysis Notebooks Guide

## Workflow

Run notebooks in sequential order:

1. **`01_wrangler.ipynb`** - Clean, standardize, and transform **raw data** into a feature set.
2. **`02_explorer.ipynb`** - Perform **Exploratory Data Analysis (EDA)** and visualize patterns.
3. **`03_modeler.ipynb`** - Build predictive **models** and evaluate performance.

---

## Typical Use Cases

| Task | Notebook Sequence | Notes |
| :--- | :--- | :--- |
| **New raw data update** | `01_wrangler.ipynb` only | Run **02** & **03** afterward to fully refresh analysis/model. |
| **Investigate crime patterns** | `02_explorer.ipynb` only | Requires latest output from **01**. |
| **Quarterly model refresh** | `01_wrangler.ipynb` $\rightarrow$ `03_modeler.ipynb` | Bypasses visualization if needed. |
| **Full pipeline from scratch** | All 3 in order | Clean data to model prediction. |

---

## Output Locations & Data Strategy

### Output Paths
- **Primary Processed Data:** `data/processed/apd/target_crimes.parquet`
- **Figures:** `reports/figures/`
- **Model Results:** `data/processed/apd/cv_results/`

### Data Format Strategy

1.  **Public Repo Format:** The compressed **`.parquet`** file (`target_crimes.parquet`) is the canonical dataset for ML model training and public sharing (e.g., Streamlit).
2.  **Local Format:** The large CSV file (`target_crimes.csv`) remains locally available on the disk for tools like **Microsoft Data Wrangler** but is permanently excluded from Git using `.gitignore` to reduce repo bloat.
3.  **Pipeline Consistency:** All subsequent notebooks (`02_explorer.ipynb` and `03_modeler.ipynb`) must be configured to read the compressed **`.parquet`** file for better performance.

---

## Runtime Estimates

| Notebook | Estimated Duration | Dependencies |
| :--- | :--- | :--- |
| **01_wrangler** | ~5-10 min | External weather API calls |
| **02_explorer** | ~2-3 min | Reads `target_crimes.parquet` |
| **03_modeler** | ~15-30 min | Depends on hyperparameter tuning settings |