Stock Index Prediction Using Machine Learning

This notebook explores how machine learning can predict stock index prices (S&P 500, DJIA, NASDAQ Composite) using economic indicators such as interest rates, unemployment, CPI, GDP, and trade volumes.

Overview
- Objective: Predict 1-, 3-, and 6-month stock index prices and assess the impact of economic indicators.
- Models Used: LASSO Linear Regression, XGBoost, Random Forest, SVM
- Data Range: 1/1/1992 to 3/1/2023

Data Sources
- Stock Indexes: S&P 500, DJIA, NASDAQ Composite
- Economic Indicators: Federal interest rate, unemployment rate, CPI, GDP, import/export volumes

Key Findings
- Machine learning models can effectively predict stock prices based on economic indicators.
- Parametric models did not provide statistically valid descriptions of individual indicators' impacts.

Challenges
Multicollinearity, time-series data, and ensuring valid statistical inference.

How to Use
1. Download the relevant data.
2. Install dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn.
3. Run the notebook to replicate the analysis.
