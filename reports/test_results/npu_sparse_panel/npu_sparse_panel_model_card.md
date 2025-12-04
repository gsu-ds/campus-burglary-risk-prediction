# 📘 Model Card: npu_sparse_panel

Generated: **2025-12-04 17:54:40**

## Overview
This card summarizes model performance on the **npu_sparse_panel** dataset, using:

- **Simple train/test split** (train < 2024-01-01, test ≥ 2024-01-01)
- **Rolling-origin cross-validation** (4 folds from 2022–2023)

## Leaderboard (Simple vs Rolling CV)

| Model             |   Simple_MAE |   CV_Mean_MAE |   CV_Mean_RMSE |   CV_Mean_R2 |
|:------------------|-------------:|--------------:|---------------:|-------------:|
| BaselineMean      |    0.22577   |     0.23691   |       0.410422 | -0.000653149 |
| CatBoostRegressor |    0.0889993 |     0.0648281 |       0.194176 |  0.776023    |
| LinearRegression  |    0.210016  |     0.219082  |       0.399651 |  0.0511594   |
| RandomForest      |    0.096639  |     0.0700468 |       0.219547 |  0.712962    |
| XGBRegressor      |    0.101975  |     0.0791788 |       0.219733 |  0.713183    |

- **Simple_MAE:** MAE on holdout 2024 period
- **CV_Mean_MAE:** Mean MAE across rolling CV folds
- **CV_Mean_RMSE:** Mean RMSE across rolling CV folds
- **CV_Mean_R2:** Mean R² across rolling CV folds

## Interpretation
- Prefer models with **low CV_Mean_MAE** and **high CV_Mean_R2**.
- Compare **Simple_MAE** to **CV_Mean_MAE** to see if the model generalizes
  to the final holdout period (stability gap).

