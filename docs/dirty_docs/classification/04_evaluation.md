Root Mean Square Error (RMSE) -> Yes
			- Mean Absolute Error (MAE) -> Yes
			- Mean Absolute Percentage Error (MAPE)
			- Mean Absolute Scaled Error (MASE) -> **Yes**
		- Future checks: Rolling-Origin Cross-Validation (Time Series CV)
			- Train on increasing chunks
				- (1) Train: 2021, Test: 2022
				- (2) Train: 2021-22, Test: 2023
			- Assess model before testing on 2024.