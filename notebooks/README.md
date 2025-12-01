# APD Crime Analysis Notebooks

## Workflow

Run notebooks in order:

1. **`01_wrangler.ipynb`** - Clean raw data
2. **`02_explorer.ipynb`** - Explore patterns
3. **`03_modeler.ipynb`** - Build models

## Typical Use Cases

| Task | Notebook to Run |
|------|----------------|
| New raw data arrived | `01_wrangler.ipynb` only |
| Investigate crime patterns | `02_explorer.ipynb` only |
| Try new model | `03_modeler.ipynb` only |
| Full pipeline from scratch | All 3 in order |
| Quarterly model refresh | `01_wrangler.ipynb` → `03_modeler.ipynb` |

## Output Locations

- **Processed Data:** `data/processed/apd/target_crimes.csv`
- **Figures:** `reports/figures/`
- **Model Results:** `data/processed/apd/cv_results/`

## Runtime Estimates

- **01_wrangler:** ~5-10 min (depends on weather API)
- **02_explorer:** ~2-3 min
- **03_modeler:** ~15-30 min (with hyperparameter tuning)