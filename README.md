# ETF Diversification Potential Decision Support Dashboard

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-008DE4?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![YFinance](https://img.shields.io/badge/YFinance-Market%20Data-lightgrey)

## Executive Summary

This project is an interactive decision support dashboard for analyzing the diversification potential of Exchange-Traded Funds (ETFs).

The system helps long-term investors compare ETFs not only by historical return, but also by risk, correlation with the market benchmark, bear-market behavior, drawdowns, tail risk, and short-term forecasted dynamics.

The dashboard is built in Python using Dash and Plotly. Historical market data is downloaded with `yfinance`, processed with `pandas` and `NumPy`, stored locally in Parquet format, and visualized through an interactive one-page analytical interface.

Key insights supported by the dashboard:

- ETF diversification should not be evaluated only by return.
- Low correlation with SPY, lower downside capture, smaller tail dependence, and lower drawdowns may indicate stronger diversification value.
- Forecasts are used as an additional analytical layer, not as a standalone trading signal.

---

## Dashboard Preview

### Full Dashboard

<!-- Insert full dashboard screenshot here -->
<img width="2557" height="1265" alt="image" src="https://github.com/user-attachments/assets/3eec6148-18fb-4a52-a66c-f8ecaab0396d" />


### Sidebar Controls

<!-- Insert sidebar screenshot here -->
<img width="434" height="871" alt="image" src="https://github.com/user-attachments/assets/88b6fccb-5850-46a5-b1f9-6427e96d34ce" />


### Metrics Table

<!-- Insert metrics table screenshot here -->
<img width="2038" height="589" alt="image" src="https://github.com/user-attachments/assets/b433aab8-6297-494d-b88f-ccadc1230145" />


---

## Business Problem

Long-term investors often need to choose ETFs that can improve portfolio stability during market downturns. However, common ETF comparison tools usually focus on return, expense ratio, or basic volatility.

This project addresses the following questions:

- Which ETFs behave differently from the broad market benchmark?
- Which assets may reduce portfolio risk during crisis periods?
- How do ETFs compare by return, volatility, drawdown, beta, correlation, and tail-risk metrics?
- What is the expected short-term price direction over 1-month and 3-month horizons?

---

## Methodology

- **Data Source:** Yahoo Finance data accessed through `yfinance`
- **Data Storage:** Local Parquet files organized by ticker
- **Frontend:** Dash interactive web dashboard
- **Visualization:** Plotly Graph Objects
- **Data Processing:** pandas, NumPy
- **Dimensionality Reduction:** PCA using scikit-learn
- **Forecasting Models:** PatchTST and N-BEATS
- **Forecast Target:** Log-returns, later transformed back into price levels
- **Forecast Horizons:** 1 month / 21 trading days and 3 months / 63 trading days

The system supports a ticker-oriented data structure. Historical prices, asset metadata, and forecast outputs are stored separately in local folders such as `data/prices`, `data/asset_info`, `data/forecast_prices/1mo`, and `data/forecast_prices/3mo`.

---

## Tools & Technologies

- **Python:** main programming language
- **Dash:** interactive web application and dashboard layout
- **Plotly:** interactive financial charts
- **pandas:** time series processing, table construction, Parquet operations
- **NumPy:** numerical calculations and vectorized operations
- **scikit-learn:** PCA and data standardization
- **PyTorch:** neural network forecasting models
- **yfinance:** historical ETF market data
- **pathlib:** local file and directory management

---

## Main Functionality

The dashboard allows users to:

- select ETF tickers for comparison;
- change the analysis period using a year range slider;
- quickly switch to predefined crisis periods;
- enable 1-month or 3-month forecast mode;
- choose which analytical charts are displayed;
- compare ETFs in a structured metrics table;
- analyze historical behavior and forecasted price dynamics.

The dashboard recalculates results reactively after the user changes selected assets, time range, forecast horizon, or chart configuration.

---

## Analytical Metrics

The main metrics table groups ETF indicators into several analytical blocks:

- **YFinance:** ticker, short name, category, historical return, beta;
- **General Diversification:** correlation, volatility, beta, annualized return;
- **Bear-Market Behavior:** downside capture, upside capture, bear-market correlation;
- **Crisis & Tail Risk:** maximum drawdown, CVaR impact, tail dependence;
- **Profitability:** Sortino ratio, Calmar ratio, return-to-risk interpretation.

The table uses color highlighting to mark values that meet predefined positive interpretation rules or perform better than SPY.

---

## Visual Analytics

The dashboard includes five key charts.

### 1. Indexed Price Dynamics

<!-- Insert chart screenshot here -->
<img width="2036" height="885" alt="image" src="https://github.com/user-attachments/assets/695a7ac1-1861-4d3d-b659-5551d681d81c" />


Compares ETF price growth in a common indexed scale. This makes it easier to compare assets with different nominal prices.

### 2. PCA Return Space Map

<!-- Insert chart screenshot here -->
<img width="2031" height="757" alt="image" src="https://github.com/user-attachments/assets/ef29ef05-f5ca-4201-852f-9bd774404075" />


Projects ETFs into a two-dimensional space based on return similarity. Assets located close to each other have more similar behavior, while distant assets may provide stronger diversification value.

### 3. Rolling Correlation with SPY

<!-- Insert chart screenshot here -->
<img width="2034" height="829" alt="image" src="https://github.com/user-attachments/assets/cc3d060f-b4ac-4cc8-9efc-08c1d0d141f3" />


Shows how each ETF’s correlation with the SPY benchmark changes over time using a rolling 252-trading-day window.

### 4. Drawdown from Previous Peak

<!-- Insert chart screenshot here -->
<img width="2044" height="849" alt="image" src="https://github.com/user-attachments/assets/881f25a9-0fa7-4fce-a733-0ec9b02b7b94" />


Displays the decline from the previous local maximum. This chart helps evaluate how deeply an ETF has fallen during unfavorable market periods.

### 5. Rolling Annualized Volatility

<!-- Insert chart screenshot here -->
<img width="2027" height="837" alt="image" src="https://github.com/user-attachments/assets/0b5df9f7-9b10-4f0c-b03e-5a72ec851a8d" />


Shows how ETF risk changes over time based on rolling annualized volatility.

---

## Forecasting Module

The forecasting module estimates ETF price trajectories for:

- **1 month:** 21 trading days;
- **3 months:** 63 trading days.

The model predicts future **log-returns**, not prices directly. After prediction, the forecasted returns are accumulated and converted back into a price path using the last known real price.

Model architectures are used:

- **PatchTST:** a Transformer-based time series model that splits the input window into local patches;

For each ticker, the model is trained on historical return windows and evaluated using a chronological walk-forward split. The better-performing model is selected for inference.

### Forecast Accuracy

Walk-forward evaluation on the 3-month horizon produced the following median results:

- **Median RMSE:** 1.75275
- **Median MAE:** 1.31745
- **Median MAPE:** 2.77%
- **Median R²:** 0.75355
- **Median Accuracy:** approximately **97.23%**

The accuracy value is interpreted as `100% − Median MAPE`.

### Historical and Forecasted ETF Price Dynamics

<img width="609" height="590" alt="image" src="https://github.com/user-attachments/assets/2169161a-ce88-4bf1-b37d-e9b6a859c9c2" />


### Why the Forecast Line Is Smoother

The forecast does not fluctuate like real market prices because the model generates a direct multi-step expected trajectory from historical log-returns. It is not a stochastic simulator and does not attempt to reproduce random day-to-day market noise.

The final forecast line reflects the expected direction and intensity of price movement under patterns learned from the previous 252-trading-day context window. Therefore, the forecast should be interpreted as an additional analytical signal, not as a guaranteed future price or an automatic buy/sell recommendation.

---

## Results & Recommendations

The system provides a compact analytical environment for ETF comparison and diversification analysis.

Main outcomes:

- Historical ETF data is automatically downloaded and stored locally.
- The user can compare multiple ETFs by return, risk, drawdown, market sensitivity, and crisis behavior.
- PCA visualization helps identify similar and dissimilar ETF groups.
- Rolling correlation and drawdown charts show whether diversification benefits remain stable across time.
- Forecasting adds a short-term and medium-term perspective to the historical analysis.

The dashboard is most useful for long-term investors who want to evaluate whether an ETF can improve portfolio resilience rather than simply maximize historical return.

---

## Project Structure

```text
investment-dashboard/
│
├── main.py
├── dashboard_app.py
├── data_manager.py
├── forecast.py
│
├── data/
│   ├── prices/
│   ├── asset_info/
│   ├── forecast_prices/
│   │   ├── 1mo/
│   │   └── 3mo/
│   └── forecast_meta/
│
└── assets/
    ├── full_dashboard.png
    ├── sidebar.png
    ├── metrics_table.png
    ├── indexed_prices.png
    ├── pca_map.png
    ├── rolling_correlation.png
    ├── drawdown.png
    └── rolling_volatility.png
