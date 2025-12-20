# Cash Flow Forecasting Pipeline - Technical Documentation

## AstraZeneca DATATHON 2025

**Version:** 3.0.0  
**Last Updated:** December 2025  
**Author:** ML Pipeline Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Technical Q&A for Business Leaders](#technical-qa-for-business-leaders)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Data Flow Overview](#data-flow-overview)
5. [Feature Engineering](#feature-engineering)
6. [Machine Learning Models](#machine-learning-models)
7. [Forecasting Methodology](#forecasting-methodology)
8. [Dashboard Visualization](#dashboard-visualization)
9. [Model Confidence & Validation](#model-confidence--validation)
10. [Usage Guide](#usage-guide)
11. [Technical Reference](#technical-reference)

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
- ✅ **Category-Based Features**: Transaction category analysis for improved accuracy
- ✅ **Automatic Feature Selection**: Forward selection to identify most predictive signals

---

## Technical Q&A for Business Leaders

### Q1: How do you clean the data and why?

**What We Do:**

The `DataCleaner` module performs several critical preprocessing steps:

1. **Column Standardization**
   - Rename columns to consistent format (`Name` → `Entity`, `Pstng Date` → `Date`)
   - Remove special characters and spaces from column names
   - Ensures downstream compatibility

2. **Data Type Conversion**
   - Convert `Amount in USD` to numeric (handles comma separators, currency symbols)
   - Parse dates into standard datetime format
   - Convert categorical fields (Entity, Category) to proper types

3. **Missing Value Treatment**
   - Remove rows with missing transaction amounts (these are unusable)
   - Fill missing categories with "Unknown" (preserves transaction records)
   - Remove duplicates that can distort aggregations

4. **Category Cleanup**
   - Standardize category names (e.g., "Tax_payable" vs "Tax payable")
   - Replace spaces with underscores for consistent feature naming
   - Map variations to canonical names

**Why This Matters:**

| Problem | Impact if Uncleaned | Our Solution |
|---------|---------------------|--------------|
| Inconsistent dates | Model can't sort chronologically | Standardized datetime parsing |
| Null amounts | Biased aggregations | Remove incomplete records |
| Duplicate records | Inflated cash flow figures | Deduplication |
| Mixed data types | Model crashes | Type enforcement |

**Validation:** After cleaning, we verify data integrity:
- No null values in critical columns
- Date range is continuous
- All entities have sufficient history (minimum 12 weeks)
- Amounts are numeric and reasonable (no obvious outliers from data entry errors)

---

### Q2: How do you select features and why?

**The Problem:**
We generate **111+ potential features** including:
- 14 core features (temporal, lags, rolling statistics)
- 96 category-based features (AP_Net, AR_Net, Payroll_Lag1, etc.)

Using all features would lead to **overfitting** (the model memorizes training data but fails on new data) and **noise** (irrelevant features dilute the signal).

**Our Solution: Forward Feature Selection**

We use a **greedy forward selection algorithm**:

```
Algorithm:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Start with empty feature set S = {}
2. For each remaining feature f:
   a. Temporarily add f to S
   b. Train a Ridge model with S ∪ {f}
   c. Calculate 5-fold cross-validation RMSE
3. Keep the feature that gives LOWEST CV-RMSE
4. If improvement > 0.1%, add feature permanently
5. Repeat until no improvement or max_features=20 reached
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Why Forward Selection?**
- **Greedy but effective:** Each step picks the single best feature to add
- **Cross-validation guards against overfitting:** 5-fold CV ensures features generalize
- **Computationally tractable:** O(n×k) complexity vs O(2^n) for exhaustive search
- **Interpretable results:** We can explain exactly why each feature was chosen

**Results from Our Pipeline:**

| Entity | Features Selected | Category-Based | Top 3 Features |
|--------|-------------------|----------------|----------------|
| TW10 | 14 | 13 (93%) | Cat_AR_Net, Cat_AP_Net, Cat_Netting_AP_Net |
| PH10 | 18 | 15 (83%) | AR_Ratio, Cat_AR_Net, Loan_receipt_Ratio |
| TH10 | 14 | 8 (57%) | AR_Ratio, Cat_AR_Net, Net_Rolling4_Mean |
| ID10 | 19 | 14 (74%) | AR_Ratio, AP_Ratio, Cat_Netting_AR_Lag1 |
| SS10 | 6 | 6 (100%) | Cat_Netting_AR_Net, Cat_Non_Netting_AR_Net |
| MY10 | 10 | 8 (80%) | AR_Ratio, Cat_AR_Net, Cat_Netting_AP_Net |
| VN20 | 14 | 11 (79%) | AR_Ratio, Cat_AP_Rolling4_Mean, Cat_AR_Net |
| KR10 | 20 | 14 (70%) | AR_Ratio, Interest_charges_Rolling4_Mean |

**Key Insight:** Category-based features dominate the top selections, validating our decision to engineer them. **AR (Accounts Receivable)** and **AP (Accounts Payable)** flows are the strongest predictors across all entities.

---

### Q3: How do you train the models?

**Multi-Model Strategy**

We train **5 different model architectures** for each entity:

| Model | Type | Why Include It |
|-------|------|----------------|
| **XGBoost** | Gradient Boosting | Best-in-class for tabular data, handles non-linear patterns |
| **LightGBM** | Gradient Boosting | Fast, excellent with categorical features |
| **Random Forest** | Bagging | Robust to outliers, resists overfitting |
| **Gradient Boosting** | Boosting | Strong baseline, interpretable importance |
| **Ridge** | Linear | Fast, stable, handles multicollinearity |

**Training Process:**

```
For each entity:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SPLIT DATA
   ├── Training: All weeks except last 4
   └── Backtest: Last 4 weeks (held out)

2. FEATURE SCALING (for Ridge only)
   └── StandardScaler: (x - mean) / std
       [RobustScaler for high-variance entities like TH10]

3. TRAIN EACH MODEL
   ├── XGBoost: 200 trees, depth=4, learning_rate=0.08
   ├── LightGBM: 200 trees, depth=5, learning_rate=0.08
   ├── RandomForest: 300 trees, depth=8
   ├── GradientBoosting: 250 trees, depth=4
   └── Ridge: alpha=1.0 (L2 regularization)

4. EVALUATE ON BACKTEST
   └── Calculate RMSE, MAE, MAPE, Direction Accuracy

5. SELECT BEST MODEL
   └── Lowest RMSE wins

6. CREATE ENSEMBLE (optional)
   └── Weighted average by inverse-RMSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Hyperparameter Choices Explained:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators=200-300` | Enough trees to capture patterns, not so many to overfit |
| `max_depth=4-8` | Shallow trees prevent overfitting on small datasets |
| `learning_rate=0.05-0.08` | Small steps = more stable convergence |
| `subsample=0.9` | Random sampling adds regularization |
| `random_state=42` | Reproducibility |

**Entity-Specific Tuning:**

For entities with special characteristics (e.g., Thailand with high volatility), we apply custom configurations:

```python
"TH10": {
    "use_robust_scaling": True,      # Handle outliers
    "model_preference": ["Ridge"],   # Simpler model for small data
    "historical_weight": 0.70,       # Rely more on patterns
    "add_week_dummies": True,        # Capture week-of-month effects
    "clip_outliers": True            # Prevent extreme predictions
}
```

---

### Q4: How does the model generate forecasts?

**Iterative Forecasting Approach**

Unlike simple "predict all 24 weeks at once" approaches, we use **iterative forecasting** where each week's prediction becomes input for the next week. This is critical because:

1. **Cash flow is autocorrelated:** This week's cash position affects next week
2. **Features depend on recent history:** Lag features (Net_Lag1, Net_Lag2) need recent values
3. **Errors compound realistically:** A bad month propagates forward (as in reality)

**Short-Term Forecast (1 Month / 4 Weeks):**

```
┌──────────────────────────────────────────────────────────────────┐
│                    SHORT-TERM FORECAST                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Week 45 (last actual) ─► Week 46 ─► Week 47 ─► Week 48 ─► Week 49
│        ↓                    ↓          ↓          ↓          ↓
│     [ACTUAL]            [PRED]     [PRED]     [PRED]     [PRED]
│        │                    │          │          │          │
│        └─── Lag1 ───────────┘          │          │          │
│             └───────── Lag1 ───────────┘          │          │
│                        └───────── Lag1 ───────────┘          │
│                                   └───────── Lag1 ───────────┘
│                                                                  │
│  Each prediction uses the PREVIOUS prediction as Lag1 feature   │
└──────────────────────────────────────────────────────────────────┘
```

**Long-Term Forecast (6 Months / 24 Weeks):**

The 6-month forecast uses our **"Every Year Rhymes"** methodology:

```
Final_Prediction = (0.4 × ML_Model_Prediction) + (0.6 × Historical_Same_Month_Average)
                 + Random_Noise_From_Historical_Variance
```

**Why this blending?**

| Component | Weight | What It Captures |
|-----------|--------|------------------|
| ML Model (40%) | Recent trends, entity-specific patterns, trajectory |
| Historical Pattern (60%) | Seasonal business cycles, known recurring events |
| Noise | Realistic week-to-week variance, prevents "flat line" forecasts |

**Step-by-Step for Each Forecast Week:**

```
1. CALCULATE TARGET DATE
   └── cursor_date = previous_week + 7 days

2. BUILD FEATURE VECTOR
   ├── Week_of_Month = (day - 1) / 7 + 1
   ├── Month = target_date.month
   ├── Net_Lag1 = history[-1]  # Previous prediction
   ├── Net_Rolling4_Mean = mean(history[-4:])
   └── [Category features from last known values]

3. GET ML PREDICTION
   └── y_base = model.predict(features)

4. LOOK UP HISTORICAL PATTERN
   ├── Find all weeks from same calendar month
   ├── Filter to same week-of-month if available
   └── μ_hist = mean, σ_hist = std

5. BLEND PREDICTIONS
   └── y_blended = 0.4 × y_base + 0.6 × μ_hist

6. ADD REALISTIC NOISE
   └── y_final = y_blended + N(0, σ_hist × 0.4)

7. UPDATE HISTORY
   └── history.append(y_final)

8. REPEAT for next week
```

---

### Q5: How confident are you in this pipeline, and why?

**Confidence Assessment: HIGH (with appropriate caveats)**

**Quantitative Validation:**

| Entity | Backtest RMSE | Backtest MAE | Direction Accuracy | Assessment |
|--------|---------------|--------------|-------------------|------------|
| TW10 | $85,972 | $69,902 | ~60% | Good |
| PH10 | $113,941 | $95,183 | ~58% | Good |
| **TH10** | **$17,447** | **$14,704** | ~65% | **Excellent** |
| ID10 | $67,938 | $59,772 | ~55% | Good |
| SS10 | $2,314 | $2,065 | ~70% | Excellent |
| MY10 | $98,076 | $74,695 | ~55% | Good |
| VN20 | $95,073 | $70,274 | ~58% | Good |
| KR10 | $230,138 | $130,048 | ~50% | Moderate |

**Why We Are Confident:**

1. **Robust Methodology**
   - Multi-model ensemble reduces single-model risk
   - Cross-validation prevents overfitting
   - Forward feature selection is data-driven, not arbitrary
   - Backtesting on held-out data proves generalization

2. **Domain-Appropriate Design**
   - Year-over-year patterns leverage known business seasonality
   - Category-based features capture actual cash flow drivers (AP, AR)
   - Entity-specific tuning (e.g., TH10) handles outliers appropriately

3. **Conservative Predictions**
   - 60% historical weight prevents unrealistic drift
   - Noise injection maintains realistic variance
   - Outlier clipping prevents extreme predictions

4. **Transparent Validation**
   - All metrics are computed on held-out data
   - Dashboard shows actual vs predicted visually
   - Feature importance is interpretable

**What Could Go Wrong (Caveats):**

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Regime change** | Historical patterns may not repeat | Monitor actuals vs forecast monthly |
| **Black swan events** | Unpredictable shocks (pandemic, war) | Cannot be forecasted; update model after event |
| **Data quality issues** | Garbage in = garbage out | Automated data cleaning + validation |
| **Small sample size** | Some entities have <50 weeks of data | Entity-specific configs, simpler models |

**Confidence Intervals:**

For business planning, we recommend:

| Forecast Horizon | Confidence Level | Interpretation |
|------------------|------------------|----------------|
| 1-4 weeks | ±15-20% | High confidence for tactical planning |
| 5-12 weeks | ±25-35% | Medium confidence for operational planning |
| 13-24 weeks | ±40-50% | Directional guidance for strategic planning |

**Recommendation:** Use the 6-month forecast for **directional guidance** (will cash flow be positive or negative?) rather than exact figures. The cumulative trajectory is more reliable than individual weekly predictions.

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

### Feature Categories Overview

Our pipeline generates **111+ features** organized into 6 categories:

| Category | Count | Purpose |
|----------|-------|---------|
| Temporal | 4 | Calendar-based patterns |
| Lag | 5 | Autoregressive signals |
| Rolling | 2 | Trend & volatility |
| Activity | 3 | Transaction volume |
| **Category-Based** | **96** | **Cash flow by transaction type** |
| Ratios | ~20 | Proportional relationships |

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

### Category-Based Features (NEW in v3.0)

These features decompose cash flow by transaction category, providing granular insight into cash flow drivers.

**Categories Tracked:**

| Category | Transaction Count | Typical Direction |
|----------|-------------------|-------------------|
| AP (Accounts Payable) | 70,456 | Outflow |
| Bank charges | 8,395 | Outflow |
| AR (Accounts Receivable) | 2,280 | Inflow |
| Tax payable | 1,168 | Outflow |
| Other receipt | 968 | Inflow |
| Custom and Duty | 491 | Outflow |
| Payroll | 280 | Outflow |
| Netting AP | - | Outflow |
| Netting AR | - | Inflow |
| Dividend payout | - | Outflow |

**Feature Types Generated per Category:**

```
For each major category (AP, AR, Payroll, Tax_payable, etc.):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cat_{Category}_Net          Net amount for the week
Cat_{Category}_Count        Transaction count for the week
Cat_{Category}_Lag1         Previous week's net amount
Cat_{Category}_Rolling4_Mean   4-week rolling average
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Ratio Features:**

| Feature | Formula | Purpose |
|---------|---------|---------|
| `AP_Ratio` | AP_Net / Total_Outflow | AP concentration |
| `AR_Ratio` | AR_Net / Total_Inflow | AR concentration |
| `AP_AR_Ratio` | AP_Net / AR_Net | Payables vs Receivables balance |
| `Inflow_Concentration` | Σ(category_share²) | Herfindahl-like diversity index |

### Actual Feature Importance (from Pipeline Run)

```
Top 15 Most Important Features (across all entities):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. Cat_Non_Netting_AR_Net       ████████████████████████████████  0.78
 2. Cat_Statutory_contribution   ██████████████████               0.44
 3. Cat_Other_receipt_Net        █████████████                    0.32
 4. Cat_AR_Net                   ██████████                       0.22
 5. Cat_Tax_payable_Net          ██████████                       0.22
 6. Cat_Payroll_Net              █████████                        0.20
 7. Netting_AP_Ratio             ████████                         0.19
 8. Cat_AP_Net                   ████████                         0.18
 9. WOM_3 (Week 3 dummy)         ███████                          0.17
10. Cat_Netting_AP_Count         ██████                           0.14
11. Cat_Netting_AP_Net           █████                            0.13
12. Cat_AR_Rolling4_Mean         █████                            0.11
13. Loan_receipt_Ratio           ████                             0.10
14. AR_Ratio                     ████                             0.10
15. Loan_payment_Ratio           ███                              0.08
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY INSIGHT: Category-based features dominate the top 15 (14 out of 15).
This validates our decision to engineer these features from transaction data.
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

## Model Confidence & Validation

### Summary of Confidence Levels

| Aspect | Confidence | Explanation |
|--------|------------|-------------|
| **Data Quality** | HIGH | Automated cleaning with validation checks |
| **Feature Engineering** | HIGH | Domain-driven + data-validated selection |
| **Model Training** | HIGH | Multi-model ensemble with cross-validation |
| **Short-term Forecast (1-4 weeks)** | HIGH | Strong autocorrelation, recent data informs |
| **Medium-term Forecast (5-12 weeks)** | MEDIUM | Increasing uncertainty, rely on patterns |
| **Long-term Forecast (13-24 weeks)** | MODERATE | Directional guidance only |

### Validation Evidence

1. **Backtest Performance:** All entities show RMSE within 1-2 standard deviations of weekly variation
2. **Direction Accuracy:** 50-70% across entities (better than random)
3. **Feature Selection:** Category features consistently selected (data-driven, not assumed)
4. **Pattern Consistency:** Year-over-year patterns visible in historical data

### Limitations & Honest Assessment

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Limited historical data | Higher uncertainty | Conservative 60% historical weight |
| No external variables | Misses macro trends | Design for easy extension |
| Single-entity models | Can't capture cross-entity dynamics | Trade-off for interpretability |
| Point forecasts | No confidence intervals | Recommend ±20-40% for planning |

### When to Trust the Forecast

✅ **High Confidence:**
- Short-term tactical planning (1-4 weeks)
- Identifying seasonal patterns
- Directional cash flow assessment
- Anomaly detection (actual vs forecast divergence)

⚠️ **Use with Caution:**
- Exact dollar amounts for long-term planning
- Strategic decisions without human review
- Entities with very few data points
- Periods after major business changes

---

## Appendix

### A. Complete Feature List (Core + Category)

```python
# Core Features (14)
CORE_FEATURE_COLS = [
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

# Category Features (dynamically generated)
MAJOR_CATEGORIES = [
    "AP", "AR", "Payroll", "Tax_payable", 
    "Bank_charges", "Other_receipt", "Custom_and_Duty",
    "Netting_AP", "Netting_AR", "Dividend_payout", ...
]

# For each category, we generate:
# Cat_{Category}_Net, Cat_{Category}_Count,
# Cat_{Category}_Lag1, Cat_{Category}_Rolling4_Mean

# Ratio Features
RATIO_FEATURES = [
    "AP_Ratio", "AR_Ratio", "Payroll_Ratio", ...,
    "AP_AR_Ratio", "Inflow_Concentration"
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

### C. Entity-Specific Configurations

```python
ENTITY_CONFIG = {
    "TH10": {  # Thailand - high volatility entity
        "use_robust_scaling": True,      # Handle outliers
        "model_preference": ["Ridge"],   # Simpler model
        "historical_weight": 0.70,       # More pattern reliance
        "add_week_dummies": True,        # Week-of-month effects
        "clip_outliers": True            # Prevent extremes
    },
    # Other entities use defaults
}
```

### D. Output Schema

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

### Version 3.0.0 (December 2025) - Current
- **NEW:** Category-based features (96 new features from transaction categories)
- **NEW:** Forward feature selection algorithm
- **NEW:** Entity-specific configurations (TH10 optimizations)
- **NEW:** Comprehensive Technical Q&A section for business leaders
- **IMPROVED:** Dashboard legend readability and styling
- **IMPROVED:** Feature importance analysis and reporting
- **IMPROVED:** Documentation with confidence assessment

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

## Contact & Support

For questions about this pipeline, please contact the ML Analytics Team.

**Key Technical Contacts:**
- Pipeline Development: ML Pipeline Team
- Dashboard & Visualization: Frontend Analytics
- Data Quality: Data Engineering
