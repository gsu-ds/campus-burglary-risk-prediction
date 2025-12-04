# 📘 Model Card: npu_dense_panel

Generated: **2025-12-02 19:36:46**

## Overview
This card summarizes model performance on the **npu_dense_panel** dataset, using:

- **Simple train/test split** (train < 2024-01-01, test ≥ 2024-01-01)
- **Rolling-origin cross-validation** (4 folds from 2022–2023)

## Leaderboard (Simple vs Rolling CV)

| Model             |   Simple_MAE |   CV_Mean_MAE |   CV_Mean_RMSE |   CV_Mean_R2 |
|:------------------|-------------:|--------------:|---------------:|-------------:|
| BaselineMean      |   0.189277   |    0.197673   |      0.359453  | -0.000206449 |
| CatBoostRegressor |   0.00817806 |    0.00625307 |      0.0613133 |  0.970875    |
| LinearRegression  |   0.018457   |    0.0216133  |      0.124861  |  0.879346    |
| RandomForest      |   0.00826434 |    0.00697513 |      0.0691407 |  0.96295     |
| XGBRegressor      |   0.00871636 |    0.00788454 |      0.0693538 |  0.962676    |

- **Simple_MAE:** MAE on holdout 2024 period
- **CV_Mean_MAE:** Mean MAE across rolling CV folds
- **CV_Mean_RMSE:** Mean RMSE across rolling CV folds
- **CV_Mean_R2:** Mean R² across rolling CV folds

## Interpretation
- Prefer models with **low CV_Mean_MAE** and **high CV_Mean_R2**.
- Compare **Simple_MAE** to **CV_Mean_MAE** to see if the model generalizes
  to the final holdout period (stability gap).

