Rough Draft



Explain each concept and decisions for modeling
# Supervised vs Unsupervised, Regression vs Classification


Rough Notes:
    **Baseline Models**
		- Linear Regression
		- Poisson Regression
		- ARIMA (Will probably fail because of 0's)
		- Naive Mean
		- Naive Seasonal (7-day) (Mean value)
		- Historical mean for each NPU x time_block?
	- **Statistical Models**
		- Generalized Linear Model (GLM) with Poisson distribution or negative binom.
		- Additional Considerations: 
			- [ZIP (Zero-Inflated Poisson Models)](https://towardsdatascience.com/zero-inflated-data-comparison-of-regression-models/)
	- **Machine Learning Models**
		- Prophet
		- XGBoost w/ Poisson
			- **Note** on possible params because of 0s:
			- ```
			  xgb_params ={
			  'objective': 'count:poisson',
			  'eval_metric': 'rmse',
			  'max_depth': 6,
			  'learning_rate': 0.1,
			  'subsample': 0.8,
			  'colsample_bytree': 0.8,
			  'min_child_weight': 1
			  }			  ```
		- CATBoost
        - LightGBM Model

		- Additional Considerations: 
			- train/test or train/val/test?
				- for train/test: 
					- `train = df[df['date'] < '2024-01-01']`
					- `test = df[df['date] >= '2024-01-01']`
				- for train/val/test
					- train: 2021, 2022
					- val: 2023
					- test: 2024