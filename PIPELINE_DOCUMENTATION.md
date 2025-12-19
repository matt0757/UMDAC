# Cash Flow Forecasting Pipeline - Technical Documentation

## AstraZeneca DATATHON 2025

**Version:** 2.0.0  
**Last Updated:** December 2025  
**Author:** ML Pipeline Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Data Flow Overview](#data-flow-overview)
4. [Feature Engineering](#feature-engineering)
5. [Machine Learning Models](#machine-learning-models)
6. [Forecasting Methodology](#forecasting-methodology)
7. [Dashboard Visualization](#dashboard-visualization)
8. [Usage Guide](#usage-guide)
9. [Technical Reference](#technical-reference)
10. [Troubleshooting](#troubleshooting)

---

## Executive Summary

This pipeline provides an end-to-end solution for **cash flow forecasting** across multiple business entities. It combines data cleaning, feature engineering, machine learning ensemble models, and interactive visualization into a single executable Python script.

### Business Problems Addressed

| Problem | Description | Solution |
|---------|-------------|----------|
| **Backtest Validation** | Compare model predictions against actual historical data | Hold-out validation on last 4 weeks |
| **Short-Term Forecast** | Tactical 4-week cash flow projections | ML model with weekly iteration |
| **Long-Term Forecast** | Strategic 6-month projections | Year-over-year seasonal patterns |

### Key Features

- ✅ **Automated Pipeline**: Single script execution from raw data to dashboard
- ✅ **Multi-Model Ensemble**: XGBoost, LightGBM, RandomForest, GradientBoosting, Ridge
- ✅ **Year-Over-Year Patterns**: "Every year rhymes" seasonal adjustment
- ✅ **Interactive Dashboard**: Plotly.js visualizations with AstraZeneca branding
- ✅ **Entity-Level Forecasting**: Separate models per business entity

---

## Pipeline Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐            │
│  │ DataCleaner  │───►│ WeeklyAggregator │───►│  MLForecaster   │            │
│  │              │    │                  │    │                 │            │
│  │ • Parse CSV  │    │ • Time features  │    │ • Train models  │            │
│  │ • Type conv. │    │ • Lag features   │    │ • Backtest      │            │
│  │ • Clean data │    │ • Rolling stats  │    │ • Forecast      │            │
│  └──────────────┘    └──────────────────┘    └────────┬────────┘            │
│                                                       │                      │
│                                                       ▼                      │
│                                          ┌────────────────────────┐          │
│                                          │ InteractiveDashboard   │          │
│                                          │ Builder                │          │
│                                          │                        │          │
│                                          │ • Plotly.js charts     │          │
│                                          │ • HTML/CSS generation  │          │
│                                          │ • AstraZeneca theme    │          │
│                                          └────────────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Class Structure

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `PathConfig` | Centralized file path management | `__init__()` |
| `DataCleaner` | Raw data preprocessing | `run()` |
| `WeeklyAggregator` | Feature engineering | `run()` |
| `MLForecaster` | Model training & prediction | `forecast_entity()`, `run()` |
| `InteractiveDashboardBuilder` | HTML dashboard generation | `build()` |

### Data Classes

| Dataclass | Purpose | Fields |
|-----------|---------|--------|
| `MonthlyForecast` | Single month forecast data | `month_num`, `dates`, `predictions`, `cumulative_net` |
| `ModelResult` | Training results for one model | `name`, `model`, `scaler`, `backtest_pred`, `metrics` |
| `ForecastArtifacts` | Complete entity forecast output | `best_model`, `backtest_*`, `future_*`, `metrics` |

---

## Data Flow Overview

### Input Data

**File:** `Data/Datathon Dataset.xlsx - Data - Main.csv`

| Column | Description | Type |
|--------|-------------|------|
| `Name` | Entity/company identifier | String |
| `Period` | Accounting period | String |
| `Account` | Account number | String |
| `PK` | Posting key (40=debit, 50=credit) | Integer |
| `Offst.acct` | Offsetting account | String |
| `Name of offsetting account` | Account description | String |
| `Pstng Date` | Posting date | Date |
| `Doc..Date` | Document date | Date |
| `Amount in USD` | Transaction amount | Numeric |
| `LCurr` | Local currency | String |
| `Category` | Transaction category | String |

### Output Files

| File | Description |
|------|-------------|
| `processed_data/clean_transactions.csv` | Cleaned transaction data |
| `processed_data/weekly_entity_features.csv` | Aggregated weekly features |
| `outputs/dashboards/interactive_dashboard.html` | Interactive visualization |

---

## Feature Engineering

### Feature Selection Rationale

The feature set was carefully selected based on time-series forecasting best practices and domain knowledge of cash flow patterns.

### Temporal Features (Calendar-Based)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Week_of_Month` | Week position (1-5) | Captures weekly patterns within months (payroll cycles, invoice collections) |
| `Is_Month_End` | Boolean flag | Month-end periods show concentrated activity (accounting close, batch processing) |
| `Month` | Calendar month (1-12) | Seasonal business patterns (Q4 spikes, summer slowdowns) |
| `Quarter` | Fiscal quarter (1-4) | Quarterly patterns (targets, bonuses, reporting requirements) |

### Lag Features (Autoregressive Components)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Net_Lag1` | Net cash flow from 1 week ago | Strongest predictor due to momentum effects and ongoing business activities |
| `Net_Lag2` | Net cash flow from 2 weeks ago | Captures bi-weekly patterns (payroll cycles, Net-14 payment terms) |
| `Net_Lag4` | Net cash flow from 4 weeks ago | Captures monthly patterns without explicit seasonal decomposition |
| `Inflow_Lag1` | Previous week's inflow | Separate dynamics for customer payments |
| `Outflow_Lag1` | Previous week's outflow | Separate dynamics for vendor payments |

### Rolling Statistics (Trend & Volatility)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Net_Rolling4_Mean` | 4-week moving average | Smooths noise to reveal underlying trend; prevents overfitting |
| `Net_Rolling4_Std` | 4-week rolling std | Captures recent volatility; enables conservative predictions during uncertain times |

### Activity Features (Transaction Volume)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Transaction_Count` | Total transactions | Higher volume correlates with larger absolute cash flows |
| `Inflow_Count` | Positive transaction count | Insight into business activity composition |
| `Outflow_Count` | Negative transaction count | Detects unusual activity patterns |

### Feature Importance

```
Typical Feature Importance Ranking:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Net_Lag1           ████████████████████████████  28%
Net_Rolling4_Mean  ███████████████████           19%
Net_Lag4           ██████████████                14%
Net_Lag2           ███████████                   11%
Month              ████████                       8%
Net_Rolling4_Std   ██████                         6%
Quarter            █████                          5%
Week_of_Month      ████                           4%
Other              █████                          5%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Machine Learning Models

### Model Configurations

#### XGBoost (eXtreme Gradient Boosting)

```python
XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42
)
```

**Strengths:**
- Excellent handling of non-linear patterns
- Built-in regularization prevents overfitting
- Handles missing values gracefully
- Fast training with parallelization

#### LightGBM (Light Gradient Boosting Machine)

```python
LGBMRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
```

**Strengths:**
- Faster training than XGBoost
- Excellent for large datasets
- Leaf-wise tree growth for better accuracy
- Lower memory usage

#### Random Forest

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

**Strengths:**
- Resistant to overfitting through bagging
- Provides feature importance
- Robust to outliers
- Handles non-linear relationships

#### Gradient Boosting (scikit-learn)

```python
GradientBoostingRegressor(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    random_state=42
)
```

**Strengths:**
- Robust baseline implementation
- Good out-of-box performance
- Interpretable feature importance
- Handles heteroscedasticity well

#### Ridge Regression

```python
Ridge(alpha=1.0)
# With StandardScaler preprocessing
```

**Strengths:**
- Fast training (baseline model)
- Interpretable coefficients
- Handles multicollinearity via L2 regularization
- Stable predictions

### Ensemble Strategy

The final ensemble combines predictions from all models using inverse-RMSE weighting:

```
                     1 / RMSE_model_i
weight_i = ─────────────────────────────────────
           Σ (1 / RMSE_model_j) for all models j

Ensemble_Prediction = Σ (weight_i × prediction_i)
```

**Rationale:**
- Better-performing models contribute more
- Reduces variance through averaging
- Improves generalization
- Research shows ensembles outperform single models (M4, M5 competitions)

### Model Selection

The best model is selected based on **RMSE (Root Mean Square Error)** on the backtest period:

```
RMSE = √(Σ(actual - predicted)² / n)
```

**Why RMSE?**
- Penalizes large errors more than MAE
- Same units as target variable (USD)
- Standard metric in forecasting literature
- Important for financial planning (large errors are costly)

---

## Forecasting Methodology

### Short-Term Forecast (1 Month / 4 Weeks)

#### Algorithm

1. Train the best-performing model on full historical data
2. Extract week-of-month seasonal patterns
3. For each forecast week:
   - Generate feature row using prediction history
   - Get base prediction from ML model
   - Apply seasonal adjustment
   - Add controlled noise
   - Append prediction to history

#### Seasonal Adjustment Formula

```
y_final = y_base + (seasonal_factor × |y_base| × 0.15) + noise

Where:
    seasonal_factor = (wom_mean - overall_mean) / overall_mean
    noise ~ N(0, wom_std × 0.25)
    wom_mean = mean cash flow for this week-of-month historically
    wom_std = std cash flow for this week-of-month historically
```

### Long-Term Forecast (6 Months / 24 Weeks)

#### The "Every Year Rhymes" Approach

The 6-month forecast is based on the assumption that **seasonal patterns from the same calendar month in previous years are likely to repeat**. This is particularly valid for business cash flows due to:

- Payroll cycles (same timing each year)
- Quarterly reporting requirements
- Seasonal business patterns
- Holiday effects
- Budget cycles

#### Algorithm

```
For each month m in [1, 2, 3, 4, 5, 6]:
    For each week w in [1, 2, 3, 4]:
        
        1. BUILD FEATURES from prediction history
        
        2. GET BASE PREDICTION from ML model
           y_base = Model.predict(features)
        
        3. LOOKUP HISTORICAL PATTERN for this calendar month
           μ_m,wom = mean(historical[month == forecast_month, week_of_month == w])
           σ_m,wom = std(historical[month == forecast_month, week_of_month == w])
        
        4. BLEND MODEL + HISTORY (40%/60% weighting)
           y_blended = 0.4 × y_base + 0.6 × μ_m,wom
        
        5. ADD REALISTIC NOISE
           noise ~ N(0, σ_m,wom × 0.4)
           y_final = y_blended + noise
        
        6. APPEND to history for next iteration
```

#### Weighting Rationale

| Component | Weight | Rationale |
|-----------|--------|-----------|
| ML Model | 40% | Captures recent trends, entity-specific dynamics |
| Historical Pattern | 60% | Strong seasonal prior, prevents unrealistic drift |

**Why 60% historical weight?**
- Business cash flows are highly seasonal
- Same month patterns are strong predictors
- Prevents ML model from drifting over long horizons
- Similar to SARIMA seasonal component

#### Variance Preservation

A key challenge in iterative forecasting is **variance collapse** where predictions become unrealistically smooth over long horizons. To combat this:

1. **Rolling std floor:** `Net_Rolling4_Std = max(hist_std × 0.8, recent_std)`
2. **Noise injection:** Sample from historical distribution of same month/week
3. **Historical variance:** Use actual variance from same calendar period

---

## Dashboard Visualization

### AstraZeneca Color Palette

| Color | Hex Code | Usage |
|-------|----------|-------|
| Mulberry | `#830051` | Primary brand, 6-month forecast |
| Lime Green | `#C4D600` | Positive values, good performance |
| Navy | `#003865` | Historical actual data |
| Graphite | `#3F4444` | Text, borders |
| Light Blue | `#68D2DF` | 1-month forecast |
| Magenta | `#D0006F` | Negative values, alerts |
| Purple | `#3C1053` | Headers, dark accent |
| Gold | `#F0AB00` | Backtest predictions |

### Dashboard Sections

#### 1. Executive Summary
- Total entities analyzed
- Date range of data
- Pipeline generation timestamp

#### 2. Per-Entity Views

Each entity section contains:

| Chart | Description |
|-------|-------------|
| **Backtest Validation** | Last 4 weeks actual vs predicted |
| **1-Month Forecast** | Historical + 4-week forecast |
| **6-Month Forecast** | Full timeline + 24-week forecast |
| **Monthly Breakdown** | Bar chart of month-by-month forecast |

#### 3. Metrics Panel

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Square Error (primary) |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| Direction Accuracy | Correct up/down predictions |
| Sign Accuracy | Correct positive/negative predictions |

#### 4. Methodology Section

- Model selection explanation
- Key features used
- Backtest validation approach
- Limitations and caveats

### Technical Implementation

- **Plotly.js** for interactive charts (zoom, pan, hover)
- **Pure HTML/CSS/JS** - no server required
- **CDN-hosted Plotly** - single file, no dependencies
- **Inter font family** - modern typography
- **Responsive design** - adapts to screen size

---

## Usage Guide

### Basic Usage

```bash
python run_full_pipeline.py
```

### Requirements

#### Required Packages
```
pandas
numpy
scikit-learn
```

#### Optional Packages (Recommended)
```
xgboost      # Better gradient boosting
lightgbm     # Faster gradient boosting
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Directory Structure

```
project_root/
├── run_full_pipeline.py          # Main pipeline script
├── requirements.txt              # Python dependencies
├── PIPELINE_DOCUMENTATION.md     # This documentation
├── Data/
│   └── Datathon Dataset.xlsx - Data - Main.csv
├── processed_data/
│   ├── clean_transactions.csv
│   └── weekly_entity_features.csv
└── outputs/
    └── dashboards/
        └── interactive_dashboard.html
```

### Configuration Options

The pipeline has several configurable parameters in the `MLForecaster` class:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backtest_weeks` | 4 | Weeks held out for validation |
| `forecast_weeks` | 4 | Short-term forecast horizon |
| `long_horizon_weeks` | 26 | Long-term forecast horizon |

---

## Technical Reference

### Evaluation Metrics

#### MAE (Mean Absolute Error)
```
MAE = (1/n) × Σ|actual_i - predicted_i|
```
- Interpretable in original units (USD)
- Robust to outliers
- Equal weight to all errors

#### RMSE (Root Mean Square Error)
```
RMSE = √((1/n) × Σ(actual_i - predicted_i)²)
```
- Penalizes large errors more heavily
- Same units as target variable
- Primary model selection metric

#### MAPE (Mean Absolute Percentage Error)
```
MAPE = (100/n) × Σ|actual_i - predicted_i| / |actual_i|
```
- Scale-independent percentage
- Allows cross-entity comparison
- Undefined when actual = 0

#### Direction Accuracy
```
Direction_Acc = (1/(n-1)) × Σ I(sign(Δactual_i) == sign(Δpred_i))
```
- Correct trend predictions
- Important for strategic planning

#### Sign Accuracy
```
Sign_Acc = (1/n) × Σ I(sign(actual_i) == sign(pred_i))
```
- Correct surplus/deficit predictions
- Critical for cash position planning

### Best Practices

1. **Handling Missing Data**
   - Forward-fill for lag features
   - Zero-fill for missing transaction counts
   - Appropriate for sparse time series

2. **Feature Scaling**
   - Ridge regression: StandardScaler normalization
   - Tree-based models: Raw features (scale-invariant)

3. **Reproducibility**
   - All random operations use `seed=42`
   - Deterministic results across runs

4. **Backtest Period**
   - 4 weeks balances statistical significance vs data availability
   - Validates model on unseen recent data

5. **Rolling Window**
   - 4-week window captures monthly patterns
   - Responsive to recent changes

---

## Troubleshooting

### Common Issues

#### Issue: "Raw file missing" Error
```
FileNotFoundError: Raw file missing: Data/Datathon Dataset.xlsx - Data - Main.csv
```
**Solution:** Ensure the raw data CSV is in the `Data/` directory with the exact filename.

#### Issue: "Missing required columns" Error
```
ValueError: Missing required columns: ['Amount in USD', ...]
```
**Solution:** Verify the CSV has all required columns. Check for column name typos.

#### Issue: XGBoost/LightGBM Not Available
```
Warning: XGBoost not available, using fallback models
```
**Solution:** Install optional packages:
```bash
pip install xgboost lightgbm
```

#### Issue: Forecast Too Smooth/Flat
**Cause:** Variance collapse in iterative forecasting  
**Solution:** The pipeline includes variance preservation mechanisms. If still too smooth:
- Increase noise scale in `_iterative_forecast_monthly`
- Increase historical weight from 60% to 70%

#### Issue: Dashboard Not Loading
**Cause:** Browser blocking CDN script  
**Solution:** 
- Check internet connection (Plotly.js loads from CDN)
- Try a different browser
- Check browser console for errors

### Performance Optimization

| Issue | Solution |
|-------|----------|
| Slow training | Reduce `n_estimators` for boosting models |
| Memory issues | Process entities in batches |
| Large output file | Reduce historical data points in charts |

---

## Appendix

### A. Complete Feature List

```python
FEATURE_COLS = [
    "Week_of_Month",      # Temporal
    "Is_Month_End",       # Temporal
    "Month",              # Temporal
    "Quarter",            # Temporal
    "Net_Lag1",           # Lag
    "Net_Lag2",           # Lag
    "Net_Lag4",           # Lag
    "Net_Rolling4_Mean",  # Rolling
    "Net_Rolling4_Std",   # Rolling
    "Transaction_Count",  # Activity
    "Inflow_Count",       # Activity
    "Outflow_Count",      # Activity
    "Inflow_Lag1",        # Lag
    "Outflow_Lag1",       # Lag
]
```

### B. Model Hyperparameter Summary

| Model | Key Hyperparameters |
|-------|---------------------|
| XGBoost | n_estimators=200, max_depth=4, lr=0.08 |
| LightGBM | n_estimators=200, max_depth=5, lr=0.08 |
| RandomForest | n_estimators=300, max_depth=8 |
| GradientBoosting | n_estimators=250, max_depth=4, lr=0.05 |
| Ridge | alpha=1.0, with StandardScaler |

### C. Output Schema

#### ForecastArtifacts Fields

| Field | Type | Description |
|-------|------|-------------|
| `best_model` | str | Name of selected model |
| `backtest_dates` | pd.Series | Dates for backtest period |
| `backtest_actual` | np.ndarray | Actual values in backtest |
| `backtest_pred` | np.ndarray | Predicted values in backtest |
| `future_short_dates` | List[datetime] | 1-month forecast dates |
| `future_short_pred` | np.ndarray | 1-month forecast values |
| `future_long_dates` | List[datetime] | 6-month forecast dates |
| `future_long_pred` | np.ndarray | 6-month forecast values |
| `monthly_forecasts` | List[MonthlyForecast] | Month-by-month breakdown |
| `metrics` | Dict[str, float] | Best model metrics |
| `all_model_metrics` | Dict[str, Dict] | All models' metrics |

---

## Change Log

### Version 2.0.0 (December 2025)
- Added year-over-year seasonal patterns for 6-month forecast
- Implemented iterative month-by-month forecasting
- Added variance preservation to prevent forecast collapse
- Enhanced documentation with full technical details
- AstraZeneca corporate theme integration

### Version 1.0.0 (Initial)
- Basic ML forecasting pipeline
- Simple direct multi-step forecasting
- Basic HTML dashboard

---

**© 2025 AstraZeneca Datathon - Cash Flow Analytics Team**
