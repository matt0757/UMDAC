# ML Forecast Documentation

## Overview

The `6_ml_forecast.py` script uses machine learning models to forecast weekly net cash flow for each entity. This document explains the data sources, features, and methodology used.

---

## Data Source

### Primary Data File
```
processed_data/weekly_{entity}.csv
```

Where `{entity}` is one of:
- `KR10` (Korea)
- `ID10` (Indonesia)
- `MY10` (Malaysia)
- `PH10` (Philippines)
- `SS10` (Singapore)
- `TH10` (Thailand)
- `TW10` (Taiwan)
- `VN20` (Vietnam)

### Data Structure
Each weekly CSV file contains pre-aggregated transaction data with the following structure:

| Column | Description | Example |
|--------|-------------|---------|
| `Entity` | Entity/Country code | KR10 |
| `Week_Num` | Week number in the dataset | 1, 2, 3... |
| `Week_Start` | Start date of the week | 2025-01-01 |
| `Week_End` | End date of the week | 2025-01-07 |
| `Total_Net` | **Target Variable** - Net cash flow (inflow - outflow) | 7,177.65 |
| `Total_Inflow` | Total money coming in | 65,429.59 |
| `Total_Outflow` | Total money going out (negative) | -58,251.93 |

---

## Features Used for ML Models

The script uses the following features from the weekly data:

### Time-Based Features
| Feature | Description | How It's Used |
|---------|-------------|---------------|
| `Week_of_Month` | Which week of the month (1-5) | Captures monthly seasonality patterns |
| `Is_Month_End` | Boolean - is this a month-end week? (day >= 25) | Month-end often has different cash patterns |
| `Month` | Month number (1-12) | Seasonal patterns |
| `Quarter` | Quarter number (1-4) | Quarterly business cycles |

### Lag Features (Historical Values)
| Feature | Description | How It's Used |
|---------|-------------|---------------|
| `Net_Lag1` | Net cash flow from 1 week ago | Most recent pattern |
| `Net_Lag2` | Net cash flow from 2 weeks ago | Short-term trend |
| `Net_Lag4` | Net cash flow from 4 weeks ago | Same week last month pattern |
| `Inflow_Lag1` | Inflow from 1 week ago | Recent inflow pattern |
| `Outflow_Lag1` | Outflow from 1 week ago | Recent outflow pattern |

### Rolling Statistics
| Feature | Description | How It's Used |
|---------|-------------|---------------|
| `Net_Rolling4_Mean` | 4-week moving average of net cash flow | Trend indicator |
| `Net_Rolling4_Std` | 4-week rolling standard deviation | Volatility indicator |

### Transaction Counts
| Feature | Description | How It's Used |
|---------|-------------|---------------|
| `Transaction_Count` | Total number of transactions | Activity level |
| `Inflow_Count` | Number of incoming transactions | Inflow activity |
| `Outflow_Count` | Number of outgoing transactions | Outflow activity |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raw Data Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. clean_transactions.csv (Original raw transactions)          │
│         ↓                                                        │
│  2. 2_weekly_aggregation.ipynb (Jupyter notebook)               │
│         ↓                                                        │
│  3. weekly_{entity}.csv (Pre-aggregated weekly data)            │
│         ↓                                                        │
│  4. 6_ml_forecast.py (This script - ML forecasting)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Train/Test Split

```
Total Data: ~44 weeks (Jan 2025 - Oct 2025)
├── Training Set: 40 weeks (Jan 2025 - Oct 01, 2025)
└── Backtest Set: 4 weeks (Oct 08 - Oct 29, 2025) ← ONLY THIS IS VISUALIZED
```

---

## ML Models Used

| Model | Description | Best For |
|-------|-------------|----------|
| **Random Forest** | Ensemble of decision trees | Robust, handles outliers |
| **Gradient Boosting** | Sequential boosting (sklearn) | Good baseline |
| **Ridge Regression** | Linear with L2 regularization | Simple, interpretable |
| **XGBoost** | Advanced gradient boosting | Best for tabular data (requires `pip install xgboost`) |
| **LightGBM** | Fast gradient boosting | Large datasets (requires `pip install lightgbm`) |
| **Ensemble** | Weighted average of all models | Combines strengths |

---

## Output

### Console Output
- Backtest metrics (MAE, RMSE, Direction Accuracy, Sign Accuracy)
- Best model selection per entity
- 4-week forecast with 95% confidence intervals

### Visualizations
```
outputs/ml_forecast/{entity}_ml_backtest.png
```

Each visualization includes:
1. Line chart comparing actual vs predicted for last 4 weeks
2. Bar chart comparison
3. Error metrics by model
4. Feature importance (for tree-based models)

---

## Example: How KR10 Data Looks

```
Week_Start    Total_Net        Week_of_Month  Is_Month_End  Net_Lag1        Net_Lag4
----------    ---------        -------------  ------------  --------        --------
2025-01-01    7,177.65         1              False         NaN             NaN
2025-01-08    -74,476.36       2              False         7,177.65        NaN
2025-01-15    -445,193.93      3              False         -74,476.36      NaN
2025-01-22    163,027.34       4              True          -445,193.93     NaN
2025-01-29    1,543,694.40     5              True          163,027.34      7,177.65
...
2025-10-22    -2,508,577.63    4              True          28,888.55       -2,436,663.99
2025-10-29    3,001,118.02     5              True          -2,508,577.63   2,290,883.10
```

---

## Key Insights

1. **High Volatility**: Cash flows swing between millions positive and negative
2. **Month-End Effects**: Significant cash movements around month-end (day >= 25)
3. **Week-of-Month Patterns**: Different weeks within a month have different patterns
4. **Lag Features**: Past values help predict future (autocorrelation)

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
xgboost>=2.0.0 (optional)
lightgbm>=4.0.0 (optional)
```

---

## Usage

```bash
# Install dependencies
pip install scikit-learn xgboost lightgbm

# Run the script
python 6_ml_forecast.py
```
