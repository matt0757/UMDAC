"""
================================================================================
END-TO-END CASH FLOW FORECASTING PIPELINE WITH INTERACTIVE DASHBOARD
================================================================================

AstraZeneca DATATHON 2025 - Cash Flow Prediction System
Author: ML Pipeline Team
Version: 2.0.0
Last Updated: December 2025

================================================================================
EXECUTIVE SUMMARY
================================================================================

This pipeline provides an end-to-end solution for cash flow forecasting across
multiple business entities. It combines data cleaning, feature engineering,
machine learning ensemble models, and interactive visualization into a single
executable script. The system is designed to address three key business problems:

1. BACKTEST VALIDATION: Compare model predictions against actual historical data
   to validate forecasting accuracy and build confidence in predictions.

2. SHORT-TERM FORECAST (1 Month): Provide tactical 4-week cash flow projections
   for immediate operational planning and liquidity management.

3. LONG-TERM FORECAST (6 Months): Deliver strategic 24-week projections using
   iterative month-by-month forecasting with year-over-year seasonal patterns.

================================================================================
PIPELINE ARCHITECTURE
================================================================================

The pipeline follows a modular Object-Oriented Programming (OOP) design with
four main components:

┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐            │
│  │ DataCleaner  │───►│ WeeklyAggregator │───►│  MLForecaster   │            │
│  │              │    │                  │    │                 │            │
│  │ • Parse CSV  │    │ • Time features  │    │ • Train models  │            │
│  │ • Type conv. │    │ • Lag features   │    │ • Backtest      │            │
│  │ • Clean data │    │ • Rolling stats  │    │ • Forecast      │            │
│  └──────────────┘    └──────────────────┘    └────────┬────────┘            │
│                                                       │                     │
│                                                       ▼                     │
│                                          ┌────────────────────────┐         │
│                                          │ InteractiveDashboard   │         │
│                                          │ Builder                │         │
│                                          │                        │         │
│                                          │ • Plotly.js charts     │         │
│                                          │ • HTML/CSS generation  │         │
│                                          │ • AstraZeneca theme    │         │
│                                          └────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
FEATURE ENGINEERING RATIONALE
================================================================================

The feature set was carefully selected based on time-series forecasting best
practices and domain knowledge of cash flow patterns. Each feature addresses
specific aspects of cash flow behavior:

TEMPORAL FEATURES (Calendar-Based)
----------------------------------
These features capture cyclical patterns tied to business calendars:

• Week_of_Month (1-5): Captures weekly patterns within each month.
  RATIONALE: Many businesses have predictable cash flows tied to payroll cycles
  (typically week 1 or 2), vendor payments (end of month), and invoice
  collection patterns. Week 1 often shows higher collections as customers
  pay invoices from the previous month.

• Is_Month_End (Boolean): Flag for the last week of each month.
  RATIONALE: Month-end periods typically show concentrated activity due to
  accounting close processes, deadline-driven payments, and batch processing.
  This binary feature helps the model capture month-end spikes or dips.

• Month (1-12): Calendar month indicator.
  RATIONALE: Seasonal business patterns vary by month. Retail sees Q4 spikes,
  pharmaceutical companies may see quarter-end pushes, and many industries
  experience summer slowdowns. Monthly encoding captures these macro cycles.

• Quarter (1-4): Fiscal quarter indicator.
  RATIONALE: Quarterly patterns are crucial in business finance. Many companies
  have quarterly targets, bonus payouts, and reporting requirements that
  create predictable cash flow patterns. Q4 often shows "budget flush" spending.

LAG FEATURES (Autoregressive Components)
----------------------------------------
These features capture the temporal dependency of cash flows on recent history:

• Net_Lag1: Net cash flow from 1 week ago.
  RATIONALE: The most recent observation is typically the strongest predictor
  of near-term cash flow. High autocorrelation at lag-1 is common in financial
  time series due to momentum effects and ongoing business activities.

• Net_Lag2: Net cash flow from 2 weeks ago.
  RATIONALE: Captures bi-weekly patterns common in payroll cycles (many
  companies pay employees bi-weekly) and invoice terms (Net-14 payment terms).

• Net_Lag4: Net cash flow from 4 weeks ago.
  RATIONALE: Captures monthly patterns by looking at the same week from the
  previous month. This helps the model understand month-over-month trends
  and detect monthly seasonality without explicit seasonal decomposition.

• Inflow_Lag1, Outflow_Lag1: Separate lags for inflows and outflows.
  RATIONALE: Inflows and outflows may have different lag structures. Customer
  payments (inflows) might follow different patterns than vendor payments
  (outflows). Separating these allows the model to learn distinct dynamics.

ROLLING STATISTICS (Trend & Volatility)
---------------------------------------
These features capture recent trends and variability:

• Net_Rolling4_Mean: 4-week rolling average of net cash flow.
  RATIONALE: Smooths out weekly noise to reveal the underlying trend. This
  helps the model distinguish between temporary fluctuations and sustained
  changes in cash flow levels. A moving average is more stable than raw
  values and helps prevent overfitting to noise.

• Net_Rolling4_Std: 4-week rolling standard deviation.
  RATIONALE: Captures recent volatility/variability in cash flows. High
  volatility periods may require different predictions than stable periods.
  This feature enables the model to adjust confidence and potentially
  be more conservative during uncertain times.

ACTIVITY FEATURES (Transaction Volume)
--------------------------------------
These features capture the intensity of business activity:

• Transaction_Count: Total number of transactions in the week.
  RATIONALE: Higher transaction volume often correlates with larger absolute
  cash flows. This helps the model scale predictions appropriately and
  detect unusual activity patterns.

• Inflow_Count, Outflow_Count: Separate counts for positive/negative flows.
  RATIONALE: The mix of inflow vs outflow transactions provides insight into
  business activity composition. A week with many small inflows vs few large
  outflows will have different characteristics.

================================================================================
MODEL SELECTION & ENSEMBLE STRATEGY
================================================================================

The pipeline employs a multi-model ensemble approach to maximize prediction
accuracy and robustness. Each model type has specific strengths:

GRADIENT BOOSTING MODELS
------------------------
• XGBoost (eXtreme Gradient Boosting)
  - Strengths: Excellent handling of non-linear patterns, built-in
    regularization, handles missing values, fast training
  - Configuration: n_estimators=200, max_depth=4, learning_rate=0.08
  - Use case: Complex patterns with interactions between features

• LightGBM (Light Gradient Boosting Machine)
  - Strengths: Faster training than XGBoost, handles large datasets well,
    excellent for high-cardinality features, leaf-wise tree growth
  - Configuration: n_estimators=200, max_depth=5, learning_rate=0.08
  - Use case: When speed is important with large datasets

• GradientBoostingRegressor (scikit-learn)
  - Strengths: Robust implementation, good baseline, interpretable
    feature importance, handles heteroscedasticity well
  - Configuration: n_estimators=250, max_depth=4, learning_rate=0.05
  - Use case: Stable baseline with strong out-of-box performance

TREE-BASED MODELS
-----------------
• RandomForestRegressor
  - Strengths: Resistant to overfitting through bagging, provides feature
    importance, handles non-linear relationships, robust to outliers
  - Configuration: n_estimators=300, max_depth=8, min_samples_split=3
  - Use case: When data has outliers or non-linear patterns

LINEAR MODELS
-------------
• Ridge Regression
  - Strengths: Fast training, interpretable coefficients, handles
    multicollinearity through L2 regularization, stable predictions
  - Configuration: alpha=1.0 with StandardScaler preprocessing
  - Use case: Baseline model, when relationships are approximately linear

ENSEMBLE APPROACH
-----------------
The final ensemble combines predictions from all available models:
  Final_Prediction = Average(XGBoost, LightGBM, RandomForest, 
                            GradientBoosting, Ridge)

RATIONALE: Ensemble methods reduce variance and improve generalization by
averaging out individual model biases. If one model overfits to noise,
others can compensate. Research consistently shows ensembles outperform
single models in forecasting competitions (e.g., M4, M5 competitions).

MODEL SELECTION: The pipeline automatically selects the best-performing
model based on RMSE (Root Mean Square Error) on the backtest period.
This ensures the deployed model has demonstrated accuracy on held-out data.

================================================================================
YEAR-OVER-YEAR FORECASTING METHODOLOGY
================================================================================

For the 6-month forecast, the pipeline implements a sophisticated iterative
forecasting approach based on the assumption that "every year rhymes" -
meaning seasonal patterns from previous years are likely to repeat.

ITERATIVE MONTH-BY-MONTH FORECASTING
------------------------------------
Rather than predicting all 6 months at once, the pipeline:

1. Forecasts Month 1 (4 weeks) using actual historical data
2. Uses Month 1 predictions as input features for Month 2 forecast
3. Continues iteratively until Month 6 is complete

This approach is superior to direct multi-step forecasting because:
- It allows the model to adapt to its own predictions
- Captures cascading effects (e.g., a low month affects subsequent months)
- More realistic simulation of how forecasts would be used in practice

YEAR-OVER-YEAR SEASONAL ADJUSTMENT
----------------------------------
The core innovation is blending ML predictions with historical same-month
patterns. For each forecast week, the algorithm:

1. EXTRACTS HISTORICAL PATTERNS:
   - For each calendar month (Jan-Dec), compute:
     * Mean net cash flow for that month across all years
     * Standard deviation of cash flows in that month
     * Week-of-month specific patterns within that month

2. BLENDS MODEL + HISTORY (40%/60% weighting):
   Final_Forecast = 0.4 × ML_Prediction + 0.6 × Historical_Same_Month_Mean

   RATIONALE for 60% historical weight:
   - Business cash flows are highly seasonal (payroll, quarterly cycles)
   - Historical patterns from the same month are strong predictors
   - ML model captures recent trends and entity-specific dynamics
   - Blending prevents the model from drifting too far from reality

3. ADDS REALISTIC VARIANCE:
   - Samples noise from historical distribution of that month/week
   - Preserves the natural fluctuation patterns observed in real data
   - Prevents unrealistically smooth forecasts

MATHEMATICAL FORMULATION:

For week w in forecast month m:
  
  y_base = ML_Model.predict(features)
  
  μ_m = mean(historical_data[month == m])
  σ_m = std(historical_data[month == m])
  
  y_adjusted = 0.4 × y_base + 0.6 × μ_m,w
  
  noise ~ N(0, 0.4 × σ_m,w)
  
  y_final = y_adjusted + noise

Where:
  - μ_m,w is the historical mean for month m, week-of-month w
  - σ_m,w is the historical std for month m, week-of-month w

================================================================================
DASHBOARD VISUALIZATION DESIGN
================================================================================

The interactive HTML dashboard uses Plotly.js for rich, client-side
visualizations without requiring a Python server.

DESIGN PRINCIPLES:
1. Executive-friendly: Clear metrics, intuitive navigation
2. AstraZeneca branded: Corporate color palette throughout
3. Interactive: Zoom, pan, hover tooltips for exploration
4. Responsive: Adapts to different screen sizes
5. Self-contained: Single HTML file, no external dependencies

CHART TYPES:
• Backtest Charts: Overlaid actual vs predicted lines
• Forecast Charts: Stacked timeline showing history + forecasts
• Monthly Breakdown: Bar charts for 6-month forecast by month

COLOR SEMANTICS:
• Navy (#003865): Historical actual values - authoritative, stable
• Gold (#F0AB00): Backtest predictions - highlighting comparison
• Light Blue (#68D2DF): 1-month forecast - near-term, actionable
• Mulberry (#830051): 6-month forecast - strategic, long-term
• Lime Green (#C4D600): Positive values / good performance
• Magenta (#D0006F): Negative values / areas of concern

================================================================================
USAGE INSTRUCTIONS
================================================================================

BASIC USAGE:
    python run_full_pipeline.py

REQUIREMENTS:
    - pandas, numpy, scikit-learn (required)
    - xgboost, lightgbm (optional, improves accuracy)

INPUT:
    Data/Datathon Dataset.xlsx - Data - Main.csv

OUTPUT:
    - processed_data/clean_transactions.csv
    - processed_data/weekly_entity_features.csv
    - outputs/dashboards/interactive_dashboard.html

================================================================================
TECHNICAL NOTES & BEST PRACTICES
================================================================================

1. HANDLING MISSING DATA: The pipeline uses forward-fill for lag features
   and zero-fill for missing transaction counts, which is appropriate for
   sparse time series data.

2. FEATURE SCALING: Ridge regression uses StandardScaler to normalize
   features, while tree-based models (XGBoost, RandomForest) operate on
   raw features since they're scale-invariant.

3. RANDOM SEED: All random operations use seed=42 for reproducibility.

4. BACKTEST PERIOD: Default 4 weeks held out for validation, balancing
   statistical significance with data availability.

5. ROLLING FEATURES: 4-week window chosen to capture monthly patterns
   while maintaining responsiveness to recent changes.

================================================================================
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# ASTRAZENECA CORPORATE COLOR PALETTE
# ==============================================================================
# These colors are used consistently throughout the dashboard to maintain
# brand identity and provide semantic meaning to different data elements.

AZ_COLORS = {
    "mulberry": "#830051",      # Primary brand color - used for 6-month forecast
    "lime_green": "#C4D600",    # Accent color - used for positive values
    "navy": "#003865",          # Secondary color - used for actual/historical data
    "graphite": "#3F4444",      # Neutral color - used for text and borders
    "light_blue": "#68D2DF",    # Highlight color - used for 1-month forecast
    "magenta": "#D0006F",       # Alert color - used for negative values
    "purple": "#3C1053",        # Dark accent - used for headers
    "gold": "#F0AB00",          # Warm accent - used for backtest predictions
    # Semantic mappings for dashboard consistency
    "positive": "#C4D600",      # Lime green for positive cash flows
    "negative": "#D0006F",      # Magenta for negative cash flows
    "actual": "#003865",        # Navy for actual historical values
    "forecast_short": "#68D2DF", # Light blue for 1-month forecast
    "forecast_long": "#830051",  # Mulberry for 6-month forecast
    "backtest": "#F0AB00",      # Gold for backtest predictions
}

# ==============================================================================
# OPTIONAL GRADIENT BOOSTING LIBRARIES
# ==============================================================================
# XGBoost and LightGBM provide superior performance but are optional.
# The pipeline gracefully degrades to scikit-learn models if unavailable.

try:  # pragma: no cover - optional dependency
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_XGB = False

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LGB = False

# ==============================================================================
# FEATURE CONFIGURATION
# ==============================================================================
# These features are used by all ML models. The selection is based on:
# - Temporal patterns (Week_of_Month, Month, Quarter, Is_Month_End)
# - Autoregressive components (Net_Lag1, Net_Lag2, Net_Lag4)
# - Trend/volatility indicators (Net_Rolling4_Mean, Net_Rolling4_Std)
# - Activity metrics (Transaction_Count, Inflow_Count, Outflow_Count)
# - Category-based features (AP, AR, Payroll, Tax, etc.)
# See module docstring for detailed rationale on each feature.

# Core features that are always included
CORE_FEATURE_COLS = [
    "Week_of_Month",      # Weekly position within month (1-5)
    "Is_Month_End",       # Binary flag for last week of month
    "Month",              # Calendar month (1-12)
    "Quarter",            # Fiscal quarter (1-4)
    "Net_Lag1",           # Net cash flow from 1 week ago
    "Net_Lag2",           # Net cash flow from 2 weeks ago
    "Net_Lag4",           # Net cash flow from 4 weeks ago (monthly pattern)
    "Net_Rolling4_Mean",  # 4-week moving average (trend)
    "Net_Rolling4_Std",   # 4-week rolling std (volatility)
    "Transaction_Count",  # Total transactions in week
    "Inflow_Count",       # Number of positive transactions
    "Outflow_Count",      # Number of negative transactions
    "Inflow_Lag1",        # Previous week's inflow (separated dynamics)
    "Outflow_Lag1",       # Previous week's outflow (separated dynamics)
]

# Major categories to create features from (based on transaction volume and business importance)
MAJOR_CATEGORIES = [
    "AP",                 # Accounts Payable - largest category
    "AR",                 # Accounts Receivable - key inflow
    "Payroll",            # Payroll - regular, predictable outflow
    "Tax_payable",        # Tax payments - significant periodic outflow
    "Bank_charges",       # Bank charges - regular small outflow
    "Other_receipt",      # Other receipts - miscellaneous inflows
    "Custom_and_Duty",    # Custom and Duty - import/export related
]

# Category-based features (generated dynamically based on available categories)
CATEGORY_FEATURE_COLS = [
    # Net amount per category
    "Cat_AP_Net",
    "Cat_AR_Net",
    "Cat_Payroll_Net",
    "Cat_Tax_payable_Net",
    "Cat_Bank_charges_Net",
    # Lag features for major categories
    "Cat_AP_Lag1",
    "Cat_AR_Lag1",
    "Cat_Payroll_Lag1",
    # Rolling stats for key categories
    "Cat_AP_Rolling4_Mean",
    "Cat_AR_Rolling4_Mean",
    # Category ratios (proportion of total)
    "AP_Ratio",
    "AR_Ratio",
]

# Combined feature list (will be filtered based on availability)
FEATURE_COLS = CORE_FEATURE_COLS + CATEGORY_FEATURE_COLS


# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
class PathConfig:
    """
    Centralized path management for all pipeline I/O operations.
    
    This class ensures consistent file paths across all pipeline components
    and automatically creates necessary directories on initialization.
    
    Directory Structure:
        project_root/
        ├── Data/                     # Raw input data
        │   └── Datathon Dataset.xlsx - Data - Main.csv
        ├── processed_data/           # Cleaned and feature-engineered data
        │   ├── clean_transactions.csv
        │   └── weekly_entity_features.csv
        └── outputs/
            ├── ml_pipeline/          # Model artifacts (if saved)
            └── dashboards/           # Generated HTML dashboards
                └── interactive_dashboard.html
    """
    
    def __init__(self) -> None:
        self.base = Path(__file__).resolve().parent
        self.project_root = self.base.parent  # Go up to UMDAC folder
        self.data_dir = self.project_root / "Data"
        self.raw_main = self.data_dir / "Datathon Dataset.xlsx - Data - Main.csv"
        self.processed = self.base / "processed_data"
        self.outputs = self.base / "outputs"
        self.processed.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.forecast_dir = self.outputs / "ml_pipeline"
        self.dashboard_dir = self.outputs / "dashboards"
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# DATA CLEANING MODULE
# ==============================================================================
class DataCleaner:
    """
    Cleans and standardizes raw transaction data from CSV exports.
    
    This class handles the initial data preprocessing step, transforming raw
    financial transaction data into a clean, analysis-ready format.
    
    Processing Steps:
    1. Load raw CSV data from the Data directory
    2. Select only required columns (reduces memory footprint)
    3. Parse and convert Amount in USD to numeric (handles comma formatting)
    4. Convert date columns to datetime objects
    5. Export cleaned data to processed_data directory
    
    Column Requirements:
    - Name: Entity/company identifier
    - Period: Accounting period
    - Account: Account number
    - PK: Posting key (40=debit, 50=credit)
    - Offst.acct: Offsetting account
    - Name of offsetting account: Account description
    - Pstng Date: Posting date (primary date for analysis)
    - Doc..Date: Document date
    - Amount in USD: Transaction amount (positive=inflow, negative=outflow)
    - LCurr: Local currency
    - Category: Transaction category for segmentation
    
    Attributes:
        paths: PathConfig instance for file path management
        KEEP_COLS: List of required column names from raw data
    """
    
    KEEP_COLS = [
        "Name",
        "Period",
        "Account",
        "PK",
        "Offst.acct",
        "Name of offsetting account",
        "Pstng Date",
        "Doc..Date",
        "Amount in USD",
        "LCurr",
        "Category",
    ]

    def __init__(self, paths: PathConfig) -> None:
        self.paths = paths

    def run(self) -> pd.DataFrame:
        """
        Execute the data cleaning pipeline.
        
        Returns:
            pd.DataFrame: Cleaned transaction data with proper types
            
        Raises:
            FileNotFoundError: If raw data file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.paths.raw_main.exists():
            raise FileNotFoundError(f"Raw file missing: {self.paths.raw_main}")

        df = pd.read_csv(self.paths.raw_main)
        missing = [c for c in self.KEEP_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[self.KEEP_COLS].copy()
        df["Amount in USD"] = (
            df["Amount in USD"].astype(str).str.replace(",", "", regex=False)
        )
        df["Amount in USD"] = pd.to_numeric(df["Amount in USD"], errors="coerce").fillna(0)
        df["Pstng Date"] = pd.to_datetime(df["Pstng Date"], format="mixed")
        df["Doc..Date"] = pd.to_datetime(df["Doc..Date"], format="mixed")

        output_path = self.paths.processed / "clean_transactions.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Clean data saved to {output_path} ({len(df):,} rows)")
        return df


# ==============================================================================
# WEEKLY AGGREGATION & FEATURE ENGINEERING MODULE
# ==============================================================================
class WeeklyAggregator:
    """
    Transforms daily transaction data into weekly aggregated features.
    
    This class is responsible for the critical feature engineering step that
    converts raw transaction-level data into ML-ready weekly time series with
    carefully designed features for cash flow forecasting.
    
    Aggregation Strategy:
    ---------------------
    Daily transactions are grouped into 7-day weeks starting from January 1st
    of the data's earliest year. This creates consistent, non-overlapping time
    periods that align across all entities for fair comparison.
    
    Feature Categories Generated:
    ----------------------------
    1. CASH FLOW AGGREGATES:
       - Total_Net: Sum of all transactions (net cash position change)
       - Total_Inflow: Sum of positive transactions (money in)
       - Total_Outflow: Sum of negative transactions (money out)
       - Outflow_Abs: Absolute value of outflows (for analysis)
    
    2. TRANSACTION VOLUME METRICS:
       - Transaction_Count: Total number of transactions
       - Inflow_Count: Number of positive transactions
       - Outflow_Count: Number of negative transactions
    
    3. STATISTICAL SUMMARIES:
       - Avg_Transaction: Mean transaction value
       - Avg_Inflow: Mean inflow value
       - Avg_Outflow: Mean outflow value
       - Max_Transaction: Largest transaction
       - Min_Transaction: Smallest transaction
    
    4. CATEGORY BREAKDOWNS:
       - Cat_{category}_Net: Net amount per category
       - Cat_{category}_Count: Transaction count per category
    
    5. POSTING KEY ANALYSIS:
       - PK40_Amount/Count: Debit transactions (PK=40)
       - PK50_Amount/Count: Credit transactions (PK=50)
    
    6. TEMPORAL FEATURES:
       - Month: Calendar month (1-12)
       - Week_of_Month: Week position within month (1-5)
       - Is_Month_End: Binary flag for last week of month
       - Quarter: Fiscal quarter (1-4)
    
    7. LAG FEATURES (for autoregression):
       - Net_Lag1, Net_Lag2, Net_Lag4: Previous weeks' net cash flow
       - Inflow_Lag1, Outflow_Lag1: Previous week's components
    
    8. ROLLING STATISTICS (for trend/volatility):
       - Net_Rolling4_Mean: 4-week moving average
       - Net_Rolling4_Std: 4-week rolling standard deviation
    
    Why Weekly Granularity?
    ----------------------
    Weekly aggregation was chosen because:
    - Daily data is too noisy for reliable pattern detection
    - Monthly data loses intra-month patterns (payroll, billing cycles)
    - Weekly aligns with business planning cycles
    - Provides sufficient data points for ML training (~52 per year)
    
    Attributes:
        paths: PathConfig instance for file path management
    """
    
    def __init__(self, paths: PathConfig) -> None:
        self.paths = paths

    def _clean_category(self, value: str) -> str:
        """Sanitize category names for use as column names."""
        return value.replace(" ", "_").replace("/", "_")

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        year_start = datetime(df["Pstng Date"].dt.year.min(), 1, 1)
        df = df.copy()
        df["Days_From_Start"] = (df["Pstng Date"] - year_start).dt.days
        df["Week_Num"] = (df["Days_From_Start"] // 7) + 1
        df["Week_Start"] = year_start + pd.to_timedelta((df["Week_Num"] - 1) * 7, unit="D")
        df["Week_End"] = df["Week_Start"] + timedelta(days=6)

        entities = df["Name"].unique()
        weeks = sorted(df["Week_Num"].unique())
        categories = df["Category"].unique()

        weekly_rows: List[Dict[str, object]] = []
        for entity in entities:
            entity_df = df[df["Name"] == entity]
            for week in weeks:
                week_df = entity_df[entity_df["Week_Num"] == int(week)]
                row: Dict[str, object] = {
                    "Entity": entity,
                    "Week_Num": int(week),
                    "Week_Start": year_start + timedelta(days=(int(week) - 1) * 7),
                    "Week_End": year_start + timedelta(days=int(week) * 7 - 1),
                    "Total_Net": week_df["Amount in USD"].sum(),
                    "Total_Inflow": week_df[week_df["Amount in USD"] > 0]["Amount in USD"].sum(),
                    "Total_Outflow": week_df[week_df["Amount in USD"] < 0]["Amount in USD"].sum(),
                }
                row["Outflow_Abs"] = abs(row["Total_Outflow"])
                row["Transaction_Count"] = len(week_df)
                row["Inflow_Count"] = len(week_df[week_df["Amount in USD"] > 0])
                row["Outflow_Count"] = len(week_df[week_df["Amount in USD"] < 0])
                row["Avg_Transaction"] = week_df["Amount in USD"].mean() if len(week_df) else 0
                row["Avg_Inflow"] = week_df[week_df["Amount in USD"] > 0]["Amount in USD"].mean() if row["Inflow_Count"] else 0
                row["Avg_Outflow"] = week_df[week_df["Amount in USD"] < 0]["Amount in USD"].mean() if row["Outflow_Count"] else 0
                row["Max_Transaction"] = week_df["Amount in USD"].max() if len(week_df) else 0
                row["Min_Transaction"] = week_df["Amount in USD"].min() if len(week_df) else 0
                for cat in categories:
                    cat_df = week_df[week_df["Category"] == cat]
                    cat_clean = self._clean_category(str(cat))
                    row[f"Cat_{cat_clean}_Net"] = cat_df["Amount in USD"].sum()
                    row[f"Cat_{cat_clean}_Count"] = len(cat_df)
                row["PK40_Amount"] = week_df[week_df["PK"] == 40]["Amount in USD"].sum()
                row["PK50_Amount"] = week_df[week_df["PK"] == 50]["Amount in USD"].sum()
                row["PK40_Count"] = len(week_df[week_df["PK"] == 40])
                row["PK50_Count"] = len(week_df[week_df["PK"] == 50])
                weekly_rows.append(row)

        weekly_df = pd.DataFrame(weekly_rows).fillna(0)
        weekly_df["Month"] = weekly_df["Week_Start"].dt.month
        weekly_df["Week_of_Month"] = ((weekly_df["Week_Start"].dt.day - 1) // 7) + 1
        weekly_df["Is_Month_End"] = weekly_df["Week_Start"].dt.day > 21
        weekly_df["Quarter"] = weekly_df["Week_Start"].dt.quarter
        weekly_df = weekly_df.sort_values(["Entity", "Week_Num"]).reset_index(drop=True)

        # Standard lag features for net cash flow
        for lag in [1, 2, 4]:
            weekly_df[f"Net_Lag{lag}"] = weekly_df.groupby("Entity")["Total_Net"].shift(lag)
            weekly_df[f"Inflow_Lag{lag}"] = weekly_df.groupby("Entity")["Total_Inflow"].shift(lag)
            weekly_df[f"Outflow_Lag{lag}"] = weekly_df.groupby("Entity")["Total_Outflow"].shift(lag)

        weekly_df["Net_Rolling4_Mean"] = weekly_df.groupby("Entity")["Total_Net"].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )
        weekly_df["Net_Rolling4_Std"] = weekly_df.groupby("Entity")["Total_Net"].transform(
            lambda x: x.rolling(window=4, min_periods=1).std()
        )
        weekly_df["Cumulative_Net"] = weekly_df.groupby("Entity")["Total_Net"].cumsum()

        # =====================================================================
        # CATEGORY-BASED FEATURE ENGINEERING
        # =====================================================================
        # Create lag and rolling features for major categories
        major_cat_cols = [col for col in weekly_df.columns if col.startswith("Cat_") and col.endswith("_Net")]
        
        for cat_col in major_cat_cols:
            cat_name = cat_col.replace("Cat_", "").replace("_Net", "")
            
            # Lag features for categories (last week's value)
            weekly_df[f"Cat_{cat_name}_Lag1"] = weekly_df.groupby("Entity")[cat_col].shift(1)
            
            # Rolling mean for categories (captures trend)
            weekly_df[f"Cat_{cat_name}_Rolling4_Mean"] = weekly_df.groupby("Entity")[cat_col].transform(
                lambda x: x.rolling(window=4, min_periods=1).mean()
            )
        
        # Category ratios (what proportion of total flow is from each category)
        # This helps capture composition changes
        total_abs_flow = weekly_df["Total_Inflow"].abs() + weekly_df["Total_Outflow"].abs()
        total_abs_flow = total_abs_flow.replace(0, 1)  # Avoid division by zero
        
        for cat_col in major_cat_cols:
            cat_name = cat_col.replace("Cat_", "").replace("_Net", "")
            weekly_df[f"{cat_name}_Ratio"] = weekly_df[cat_col].abs() / total_abs_flow
        
        # AP to AR ratio (key business metric - payables vs receivables)
        if "Cat_AP_Net" in weekly_df.columns and "Cat_AR_Net" in weekly_df.columns:
            ar_safe = weekly_df["Cat_AR_Net"].abs().replace(0, 1)
            weekly_df["AP_AR_Ratio"] = weekly_df["Cat_AP_Net"].abs() / ar_safe
        
        # Inflow concentration (is cash coming from few or many sources?)
        inflow_cats = [col for col in weekly_df.columns if col.startswith("Cat_") and col.endswith("_Net")]
        if inflow_cats:
            inflow_values = weekly_df[inflow_cats].clip(lower=0)  # Only positive values
            inflow_total = inflow_values.sum(axis=1).replace(0, 1)
            # Herfindahl-like concentration index
            weekly_df["Inflow_Concentration"] = ((inflow_values.div(inflow_total, axis=0) ** 2).sum(axis=1))
        
        weekly_df = weekly_df.fillna(0)

        weekly_df.to_csv(self.paths.processed / "weekly_entity_features.csv", index=False)
        entity_frames: Dict[str, pd.DataFrame] = {}
        for entity in entities:
            entity_df = weekly_df[weekly_df["Entity"] == entity].copy()
            entity_path = self.paths.processed / f"weekly_{entity}.csv"
            entity_df.to_csv(entity_path, index=False)
            entity_frames[entity] = entity_df
        print(f"✅ Weekly features saved to {self.paths.processed}")
        print(f"   📊 Created {len([c for c in weekly_df.columns if 'Cat_' in c or '_Ratio' in c])} category-based features")
        return weekly_df, entity_frames


# ==============================================================================
# DATA CLASSES FOR FORECAST RESULTS
# ==============================================================================
@dataclass
class MonthlyForecast:
    """
    Stores forecast data for a single month in the 6-month iterative forecast.
    
    This dataclass captures both the weekly predictions within a month and
    the cumulative running total, enabling month-by-month analysis of the
    long-term forecast trajectory.
    
    Attributes:
        month_num: Sequential month number (1-6 for 6-month forecast)
        dates: List of datetime objects for each forecasted week
        predictions: List of predicted net cash flow values for each week
        cumulative_net: Running total of all predictions up to and including
                       this month (useful for cumulative cash position analysis)
    
    Example:
        MonthlyForecast(
            month_num=2,
            dates=[datetime(2025, 2, 3), datetime(2025, 2, 10), ...],
            predictions=[150000, -80000, 200000, 50000],
            cumulative_net=520000  # Sum of month 1 + month 2
        )
    """
    month_num: int  # 1-6
    dates: List[datetime]
    predictions: List[float]
    cumulative_net: float  # Running total up to this month


@dataclass
class ModelResult:
    """
    Encapsulates the results from training a single ML model.
    
    This dataclass holds all artifacts needed to use and evaluate a trained
    model, including the model object itself, any preprocessing scalers,
    backtest predictions, and performance metrics.
    
    Attributes:
        name: Human-readable model name (e.g., "XGBoost", "Ridge")
        model: Trained model object (sklearn-compatible with .predict())
        scaler: StandardScaler instance if feature scaling was used (None for
               tree-based models which are scale-invariant)
        backtest_pred: Array of predictions on the held-out backtest period
        metrics: Dictionary of evaluation metrics (RMSE, MAE, MAPE, R²)
    """
    name: str
    model: object
    scaler: Optional[StandardScaler]
    backtest_pred: np.ndarray
    metrics: Dict[str, float]


@dataclass
class ForecastArtifacts:
    """
    Complete forecast output for a single entity.
    
    This dataclass aggregates all forecast-related data for one business entity,
    including backtest validation results, short-term (1-month) forecasts,
    long-term (6-month) forecasts with monthly breakdown, and model metrics.
    
    Used by the dashboard builder to generate visualizations and by downstream
    analysis code to access forecast data.
    
    Attributes:
        best_model: Name of the model selected based on backtest RMSE
        backtest_dates: DatetimeIndex of the held-out validation period
        backtest_actual: Actual observed values during backtest period
        backtest_pred: Model predictions for backtest period
        future_short_dates: Dates for 1-month (4-week) forecast
        future_short_pred: Predictions for 1-month forecast
        future_long_dates: Dates for 6-month (24-week) forecast
        future_long_pred: Predictions for 6-month forecast
        monthly_forecasts: List of MonthlyForecast objects breaking down the
                          6-month forecast into individual months
        metrics: Performance metrics for the best model
        all_model_metrics: Metrics for all trained models (for comparison)
    """
    best_model: str
    backtest_dates: pd.Series
    backtest_actual: np.ndarray
    backtest_pred: np.ndarray
    future_short_dates: List[datetime]
    future_short_pred: np.ndarray
    future_long_dates: List[datetime]
    future_long_pred: np.ndarray
    monthly_forecasts: List[MonthlyForecast] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    all_model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ==============================================================================
# MACHINE LEARNING FORECASTER MODULE
# ==============================================================================
class MLForecaster:
    """
    Multi-model ML forecaster with ensemble capabilities and iterative forecasting.
    
    This class implements the core forecasting logic, including:
    - Training multiple ML models (XGBoost, LightGBM, RandomForest, etc.)
    - Backtesting to validate model accuracy on held-out data
    - Short-term (1-month) iterative forecasting
    - Long-term (6-month) iterative forecasting with year-over-year patterns
    
    Forecasting Philosophy:
    ----------------------
    The forecaster uses an iterative approach where each future prediction
    becomes input for subsequent predictions. This is more realistic than
    direct multi-step forecasting because:
    
    1. It captures cascading effects (a bad month affects subsequent months)
    2. It allows the model to adapt to its own prediction trajectory
    3. It mirrors how forecasts would actually be used in practice
    
    Year-Over-Year Pattern Integration:
    ----------------------------------
    For 6-month forecasts, the model assumes "every year rhymes" - meaning
    seasonal patterns from the same calendar month in previous years are
    likely to repeat. The forecast blends:
    
    - 40% ML model prediction (captures recent trends, entity-specific patterns)
    - 60% Historical same-month average (captures seasonal business cycles)
    
    This weighting was chosen because:
    - Business cash flows are highly seasonal (payroll, quarter-end, holidays)
    - Historical month patterns provide strong prior information
    - The ML model captures deviations from typical patterns
    - Pure ML predictions tend to drift unrealistically over long horizons
    
    Model Selection Strategy:
    ------------------------
    Multiple models are trained, and the best is selected based on RMSE
    (Root Mean Square Error) on the backtest period. RMSE was chosen because:
    
    - It penalizes large errors more than MAE (important for financial planning)
    - It's in the same units as the target variable (interpretable)
    - It's the most common metric in forecasting literature
    
    Attributes:
        paths: PathConfig instance for file path management
        backtest_weeks: Number of weeks to hold out for validation (default: 4)
        forecast_weeks: Number of weeks for short-term forecast (default: 4)
        long_horizon_weeks: Total weeks for long-term forecast (default: 26)
        use_feature_selection: Whether to perform automatic feature selection
    """
    
    # Entity-specific configurations for entities with special characteristics
    ENTITY_CONFIG = {
        "TH10": {
            # Thailand has high volatility, outliers, and strong week-of-month patterns
            "use_robust_scaling": True,  # Handle outliers better
            "model_preference": ["Ridge", "GradientBoosting", "Ensemble"],  # Simpler models for small data
            "historical_weight": 0.70,  # Rely more on historical patterns
            "model_weight": 0.30,
            "add_week_dummies": True,  # One-hot encode week-of-month
            "clip_outliers": True,  # Clip extreme predictions
            "outlier_multiplier": 2.5,  # Clip at 2.5x IQR
        },
        # Other entities use defaults
    }
    
    def __init__(
        self,
        paths: PathConfig,
        backtest_weeks: int = 4,
        forecast_weeks: int = 4,
        long_horizon_weeks: int = 26,
        use_feature_selection: bool = True,
    ) -> None:
        self.paths = paths
        self.backtest_weeks = backtest_weeks
        self.forecast_weeks = forecast_weeks
        self.long_horizon_weeks = long_horizon_weeks
        self.use_feature_selection = use_feature_selection
        self.selected_features: Dict[str, List[str]] = {}  # Store selected features per entity
        self.feature_importance: Dict[str, Dict[str, float]] = {}  # Store feature importance per entity
        self.current_entity: Optional[str] = None  # Track current entity being processed

    def _get_entity_config(self, entity: str) -> Dict:
        """Get entity-specific configuration or defaults."""
        default_config = {
            "use_robust_scaling": False,
            "model_preference": None,
            "historical_weight": 0.60,
            "model_weight": 0.40,
            "add_week_dummies": False,
            "clip_outliers": False,
            "outlier_multiplier": 3.0,
        }
        if entity in self.ENTITY_CONFIG:
            config = default_config.copy()
            config.update(self.ENTITY_CONFIG[entity])
            return config
        return default_config

    def _get_all_available_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get all potential features from the dataframe, including category-based ones.
        
        This dynamically discovers all available features including:
        - Core temporal and lag features
        - Category net amounts and counts
        - Category lag and rolling features
        - Category ratios
        """
        potential_features = []
        
        # Core features
        potential_features.extend(CORE_FEATURE_COLS)
        
        # Category net amounts
        cat_net_cols = [c for c in df.columns if c.startswith("Cat_") and c.endswith("_Net")]
        potential_features.extend(cat_net_cols)
        
        # Category counts
        cat_count_cols = [c for c in df.columns if c.startswith("Cat_") and c.endswith("_Count")]
        potential_features.extend(cat_count_cols)
        
        # Category lags
        cat_lag_cols = [c for c in df.columns if c.startswith("Cat_") and "_Lag" in c]
        potential_features.extend(cat_lag_cols)
        
        # Category rolling means
        cat_rolling_cols = [c for c in df.columns if c.startswith("Cat_") and "_Rolling" in c]
        potential_features.extend(cat_rolling_cols)
        
        # Ratio features
        ratio_cols = [c for c in df.columns if c.endswith("_Ratio")]
        potential_features.extend(ratio_cols)
        
        # Concentration index
        if "Inflow_Concentration" in df.columns:
            potential_features.append("Inflow_Concentration")
        
        # Week-of-month dummy features (for entities with special config)
        wom_dummies = [c for c in df.columns if c.startswith("WOM_")]
        potential_features.extend(wom_dummies)
        
        # Filter to only those that exist in the dataframe
        available = [f for f in potential_features if f in df.columns]
        return list(dict.fromkeys(available))  # Remove duplicates while preserving order

    def _add_week_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add week-of-month dummy variables for entities with strong week-of-month patterns.
        
        This creates one-hot encoded features for weeks 1-5 of the month,
        which allows the model to learn different patterns for each week.
        Particularly useful for entities like TH10 that show strong
        week-of-month dependencies.
        """
        df = df.copy()
        if "Week_of_Month" not in df.columns:
            df["Week_of_Month"] = ((pd.to_datetime(df["Week_Start"]).dt.day - 1) // 7) + 1
        
        # Create dummy variables for weeks 1-5
        for week in range(1, 6):
            df[f"WOM_{week}"] = (df["Week_of_Month"] == week).astype(int)
        
        return df

    def _forward_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        max_features: int = 20,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Forward feature selection using cross-validation RMSE.
        
        Algorithm:
        1. Start with empty feature set
        2. For each remaining feature, try adding it
        3. Keep the feature that gives best CV RMSE improvement
        4. Repeat until no improvement or max_features reached
        
        Returns:
            Tuple of (selected_feature_names, feature_importance_dict)
        """
        from sklearn.model_selection import cross_val_score
        
        selected_idx = []
        remaining_idx = list(range(len(feature_names)))
        importance = {}
        
        best_score = float('inf')
        
        # Use a simple model for feature selection (faster)
        selector_model = Ridge(alpha=1.0)
        scaler = StandardScaler()
        
        for _ in range(min(max_features, len(feature_names))):
            best_new_feature = None
            best_new_score = best_score
            
            for idx in remaining_idx:
                # Try adding this feature
                trial_idx = selected_idx + [idx]
                X_trial = scaler.fit_transform(X[:, trial_idx])
                
                # Cross-validation score (negative MSE, so higher is better)
                try:
                    scores = cross_val_score(
                        selector_model, X_trial, y,
                        cv=min(5, len(y) // 2),
                        scoring='neg_mean_squared_error'
                    )
                    mean_score = -np.mean(scores)  # Convert to positive RMSE-like
                    
                    if mean_score < best_new_score:
                        best_new_score = mean_score
                        best_new_feature = idx
                except Exception:
                    continue
            
            # If we found an improvement, add the feature
            if best_new_feature is not None and best_new_score < best_score * 0.995:  # 0.5% improvement threshold
                selected_idx.append(best_new_feature)
                remaining_idx.remove(best_new_feature)
                improvement = (best_score - best_new_score) / max(best_score, 1e-6)
                importance[feature_names[best_new_feature]] = improvement
                best_score = best_new_score
            else:
                break  # No improvement, stop
        
        selected_names = [feature_names[i] for i in selected_idx]
        return selected_names, importance

    def _get_feature_importance_from_model(
        self,
        model,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """
        Extract feature importance from trained model.
        
        Works with tree-based models (feature_importances_) and linear models (coef_).
        """
        importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for name, imp in zip(feature_names, importances):
                    importance[name] = float(imp)
            elif hasattr(model, 'coef_'):
                # Linear models - use absolute coefficient values
                coefs = np.abs(model.coef_)
                total = coefs.sum() if coefs.sum() > 0 else 1
                for name, coef in zip(feature_names, coefs):
                    importance[name] = float(coef / total)
        except Exception:
            pass
        
        return importance

    def _metrics(self, actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive forecast evaluation metrics.
        
        Metrics Computed:
        - MAE (Mean Absolute Error): Average absolute prediction error.
          Interpretable in original units (USD). Robust to outliers.
          
        - RMSE (Root Mean Square Error): Square root of average squared error.
          Penalizes large errors more heavily. Primary selection metric.
          
        - MAPE (Mean Absolute Percentage Error): Percentage-based error metric.
          Allows cross-entity comparison regardless of scale.
          Note: Undefined when actual values are zero.
          
        - Direction_Accuracy: Percentage of times the model correctly predicts
          whether the value will increase or decrease from the previous period.
          Important for trend-following strategies.
          
        - Sign_Accuracy: Percentage of correct positive/negative sign predictions.
          Critical for cash flow direction (surplus vs deficit).
        
        Args:
            actual: Array of actual observed values
            pred: Array of predicted values
            
        Returns:
            Dictionary containing all computed metrics
        """
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape_mask = actual != 0
        mape = (
            np.mean(np.abs((actual[mape_mask] - pred[mape_mask]) / actual[mape_mask])) * 100
            if mape_mask.any()
            else np.nan
        )
        direction_acc = (
            np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))) * 100
            if len(actual) > 1
            else np.nan
        )
        sign_acc = np.mean(np.sign(actual) == np.sign(pred)) * 100
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Direction_Accuracy": direction_acc, "Sign_Accuracy": sign_acc}

    def _train_models(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
        feature_names: List[str] = None, entity: str = None
    ) -> Dict[str, ModelResult]:
        """
        Train all available ML models and evaluate on backtest data.
        
        This method implements a multi-model training strategy where several
        different model architectures are trained on the same data, allowing
        for automatic selection of the best performer.
        
        Models Trained:
        1. XGBoost (if available): Gradient boosting with regularization
        2. LightGBM (if available): Fast gradient boosting with leaf-wise growth
        3. RandomForest: Bagged decision trees for robust predictions
        4. GradientBoosting: Classic scikit-learn gradient boosting
        5. Ridge: Regularized linear regression (baseline)
        6. Ensemble: Weighted average of all models by inverse RMSE
        
        The ensemble is constructed using inverse-RMSE weighting, meaning
        better-performing models contribute more to the final prediction.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values
            X_test: Backtest feature matrix
            y_test: Backtest target values (for evaluation only)
            feature_names: List of feature names (for importance tracking)
            entity: Entity name (for entity-specific model configurations)
            
        Returns:
            Dictionary mapping model names to ModelResult objects
        """
        results: Dict[str, ModelResult] = {}
        
        # Get entity-specific config
        entity = entity or self.current_entity
        config = self._get_entity_config(entity) if entity else {}
        use_robust_scaling = config.get("use_robust_scaling", False)

        def add_model(name: str, estimator, use_scaler: bool = False) -> None:
            if use_robust_scaling and use_scaler:
                # Use RobustScaler for high-variance entities like TH10
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                scaler = StandardScaler() if use_scaler else None
            Xtr = scaler.fit_transform(X_train) if scaler is not None else X_train
            Xte = scaler.transform(X_test) if scaler is not None else X_test
            estimator.fit(Xtr, y_train)
            pred = estimator.predict(Xte)
            results[name] = ModelResult(
                name=name,
                model=estimator,
                scaler=scaler,
                backtest_pred=pred,
                metrics=self._metrics(y_test, pred),
            )

        if HAS_XGB:
            add_model(
                "XGBoost",
                xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                    verbosity=0,
                ),
            )

        if HAS_LGB:
            add_model(
                "LightGBM",
                lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    verbose=-1,
                ),
            )

        add_model(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        )
        add_model(
            "GradientBoosting",
            GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=42),
        )
        add_model("Ridge", Ridge(alpha=1.0), use_scaler=True)

        # Ensemble weighted by inverse RMSE
        if results:
            inv_weights = {k: 1 / max(v.metrics["RMSE"], 1e-6) for k, v in results.items()}
            total = sum(inv_weights.values())
            ensemble_pred = np.zeros_like(next(iter(results.values())).backtest_pred)
            for name, res in results.items():
                ensemble_pred += (inv_weights[name] / total) * res.backtest_pred
            results["Ensemble"] = ModelResult(
                name="Ensemble",
                model=None,
                scaler=None,
                backtest_pred=ensemble_pred,
                metrics=self._metrics(y_test, ensemble_pred),
            )
        return results

    def _future_feature_row(
        self,
        target_date: datetime,
        net_history: List[float],
        last_row: pd.Series,
        inflow_history: List[float],
        outflow_history: List[float],
        historical_df: pd.DataFrame = None,
    ) -> Dict[str, float]:
        """
        Construct feature vector for a future date during iterative forecasting.
        
        This method generates the feature row needed to predict the cash flow
        for a future week. It handles the challenge of iterative forecasting
        where true future values are unknown and must be approximated from
        the prediction history.
        
        Feature Construction Strategy:
        -----------------------------
        1. TEMPORAL FEATURES: Derived directly from target_date
           - Week_of_Month, Is_Month_End, Month, Quarter
           
        2. LAG FEATURES: Use prediction history as proxy for actual values
           - Net_Lag1: Most recent value (actual or predicted)
           - Net_Lag2, Net_Lag4: Earlier values from history
           
        3. ROLLING STATISTICS: Preserve historical variance patterns
           - Net_Rolling4_Mean: 4-period moving average
           - Net_Rolling4_Std: Explicitly preserve historical std to prevent
             variance collapse during long forecasting horizons
             
        4. ACTIVITY FEATURES: Use last known actual values as estimates
           - Transaction_Count, Inflow_Count, Outflow_Count
           - Inflow_Lag1, Outflow_Lag1
        
        Variance Preservation:
        ---------------------
        A key challenge in iterative forecasting is "variance collapse" where
        predictions become increasingly smooth over long horizons. To combat
        this, the rolling std is set to max(hist_std * 0.8, recent_std),
        ensuring the model sees sufficient variance even when predictions
        are smoothing out.
        
        Args:
            target_date: The future date to generate features for
            net_history: List of historical (and predicted) net cash flows
            last_row: Last actual data row (for activity features)
            inflow_history: List of historical inflows
            outflow_history: List of historical outflows
            historical_df: Full historical DataFrame (optional, for patterns)
            
        Returns:
            Dictionary mapping feature names to values
        """
        week_of_month = ((target_date.day - 1) // 7) + 1
        month = target_date.month
        
        # Get historical statistics for similar periods (same week_of_month)
        hist_std = float(np.std(net_history[-12:])) if len(net_history) >= 12 else float(np.std(net_history))
        
        # Use recent actual history for rolling stats to preserve variance
        # Mix predicted and actual to maintain pattern continuity
        recent_for_rolling = net_history[-4:] if len(net_history) >= 4 else net_history
        
        row = {
            "Week_of_Month": week_of_month,
            "Is_Month_End": int(target_date.day > 21),
            "Month": month,
            "Quarter": (month - 1) // 3 + 1,
            "Net_Lag1": net_history[-1],
            "Net_Lag2": net_history[-2] if len(net_history) > 1 else net_history[-1],
            "Net_Lag4": net_history[-4] if len(net_history) > 3 else net_history[0],
            "Net_Rolling4_Mean": float(np.mean(recent_for_rolling)),
            "Net_Rolling4_Std": max(hist_std * 0.8, float(np.std(recent_for_rolling))),
            "Transaction_Count": float(last_row.get("Transaction_Count", 0)),
            "Inflow_Count": float(last_row.get("Inflow_Count", 0)),
            "Outflow_Count": float(last_row.get("Outflow_Count", 0)),
            "Inflow_Lag1": inflow_history[-1] if inflow_history else float(last_row.get("Total_Inflow", 0)),
            "Outflow_Lag1": outflow_history[-1] if outflow_history else float(last_row.get("Total_Outflow", 0)),
        }
        
        # Add week-of-month dummies for entities that use them
        entity_config = self._get_entity_config(self.current_entity) if self.current_entity else {}
        if entity_config.get("add_week_dummies", False):
            for week in range(1, 6):
                row[f"WOM_{week}"] = 1 if week_of_month == week else 0
        
        return row

    def _iterative_forecast(
        self,
        model_name: str,
        available_features: List[str],
        base_df: pd.DataFrame,
        horizon: int,
    ) -> Tuple[List[datetime], np.ndarray]:
        """
        Perform short-term (1-month) iterative forecast with pattern preservation.
        
        This method generates a 4-week forecast using an iterative approach
        where each week's prediction becomes input for the next. Includes
        seasonal adjustments based on week-of-month patterns to maintain
        realistic fluctuations.
        
        Algorithm:
        1. Train the specified model on full historical data
        2. Extract week-of-month seasonal patterns from history
        3. For each forecast week:
           a. Generate feature row using prediction history
           b. Get base prediction from ML model
           c. Apply seasonal adjustment based on week-of-month
           d. Add controlled noise to prevent unrealistic smoothness
           e. Append prediction to history for next iteration
        
        Seasonal Adjustment Formula:
            y_final = y_base + (seasonal_factor × |y_base| × 0.15) + noise
            
        Where:
            - seasonal_factor = (wom_mean - overall_mean) / overall_mean
            - noise ~ N(0, wom_std × 0.25)
        
        Args:
            model_name: Name of the model to use (must be in trained results)
            available_features: List of feature column names
            base_df: Historical data DataFrame
            horizon: Number of weeks to forecast
            
        Returns:
            Tuple of (forecast_dates, forecast_values)
        """
        full_X = base_df[available_features].values
        full_y = base_df["Total_Net"].values

        def build_estimator() -> Tuple[object, Optional[StandardScaler]]:
            if model_name == "XGBoost" and HAS_XGB:
                return (
                    xgb.XGBRegressor(
                        n_estimators=200, max_depth=4, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9,
                        objective="reg:squarederror", random_state=42, verbosity=0,
                    ),
                    None,
                )
            if model_name == "LightGBM" and HAS_LGB:
                return (
                    lgb.LGBMRegressor(
                        n_estimators=200, max_depth=5, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1,
                    ),
                    None,
                )
            if model_name == "RandomForest":
                return (
                    RandomForestRegressor(
                        n_estimators=300, max_depth=8, min_samples_split=3,
                        min_samples_leaf=2, random_state=42, n_jobs=-1,
                    ),
                    None,
                )
            if model_name == "GradientBoosting":
                return (
                    GradientBoostingRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=42),
                    None,
                )
            return Ridge(alpha=1.0), StandardScaler()

        estimator, scaler = build_estimator()
        X_fit = scaler.fit_transform(full_X) if scaler else full_X
        estimator.fit(X_fit, full_y)

        # Calculate historical patterns
        base_df = base_df.copy()
        base_df["Week_of_Month"] = ((base_df["Week_Start"].dt.day - 1) // 7) + 1
        overall_mean = full_y.mean()
        week_of_month_means = base_df.groupby("Week_of_Month")["Total_Net"].mean().to_dict()
        week_of_month_stds = base_df.groupby("Week_of_Month")["Total_Net"].std().to_dict()

        history_net = list(full_y)
        inflow_history = list(base_df.get("Total_Inflow", pd.Series([0] * len(base_df))).values)
        outflow_history = list(base_df.get("Total_Outflow", pd.Series([0] * len(base_df))).values)
        last_row = base_df.iloc[-1]
        cursor_date = pd.to_datetime(last_row["Week_Start"]).to_pydatetime()

        np.random.seed(42)
        preds: List[float] = []
        dates: List[datetime] = []
        
        for _ in range(horizon):
            cursor_date = cursor_date + timedelta(days=7)
            week_of_month = ((cursor_date.day - 1) // 7) + 1
            
            feat_row = self._future_feature_row(cursor_date, history_net, last_row, inflow_history, outflow_history, base_df)
            X_next = np.array([[feat_row.get(col, 0) for col in available_features]])
            X_next = scaler.transform(X_next) if scaler else X_next
            
            # Base prediction
            y_hat_base = float(estimator.predict(X_next)[0])
            
            # Apply seasonal pattern
            wom_mean = week_of_month_means.get(week_of_month, overall_mean)
            wom_std = week_of_month_stds.get(week_of_month, float(np.std(full_y)))
            
            if overall_mean != 0:
                seasonal_factor = (wom_mean - overall_mean) / max(abs(overall_mean), 1)
            else:
                seasonal_factor = 0
            
            noise_scale = wom_std * 0.25
            seasonal_noise = np.random.normal(0, noise_scale)
            
            y_hat = y_hat_base + (seasonal_factor * abs(y_hat_base) * 0.15) + seasonal_noise
            
            preds.append(y_hat)
            dates.append(cursor_date)
            history_net.append(y_hat)


        return dates, np.array(preds)

    def _iterative_forecast_monthly(
        self,
        model_name: str,
        available_features: List[str],
        base_df: pd.DataFrame,
        num_months: int = 6,
        weeks_per_month: int = 4,
    ) -> Tuple[List[datetime], np.ndarray, List[MonthlyForecast]]:
        """
        Perform iterative month-by-month forecasting with year-over-year patterns.
        
        This method implements the core long-term forecasting logic based on the
        assumption that "every year rhymes" - meaning business cash flow patterns
        from the same calendar month in previous years are likely to repeat.
        
        METHODOLOGY
        ===========
        
        The forecast is generated iteratively, one month at a time, with each
        month's predictions feeding into the next month's feature inputs. This
        captures cascading effects and maintains realistic forecast trajectories.
        
        Year-Over-Year Pattern Integration:
        ----------------------------------
        For each forecast week, the algorithm blends:
        
        1. ML MODEL PREDICTION (40% weight):
           - Captures recent trends and entity-specific dynamics
           - Uses lag features from prediction history
           - Responds to trajectory of recent weeks
           
        2. HISTORICAL SAME-MONTH PATTERN (60% weight):
           - Extracts mean cash flow for same calendar month across all years
           - Further refined to week-of-month within that month
           - Strong prior based on seasonal business cycles
        
        RATIONALE FOR 60% HISTORICAL WEIGHT:
        - Business cash flows are highly seasonal (payroll cycles, quarter-end)
        - Same month in previous years is a strong predictor
        - Prevents ML model from drifting unrealistically over long horizons
        - Validated by time series forecasting literature (similar to SARIMA)
        
        REALISTIC VARIANCE PRESERVATION:
        --------------------------------
        To prevent "variance collapse" (predictions becoming unrealistically
        smooth), the algorithm adds controlled noise:
        
        noise ~ N(0, historical_std × 0.4)
        
        Where historical_std is the standard deviation of cash flows for the
        same month/week-of-month in historical data. This ensures forecasts
        maintain the natural fluctuation patterns observed in real data.
        
        ALGORITHM STEPS
        ===============
        
        1. PATTERN EXTRACTION:
           - For each calendar month (1-12), compute:
             * Mean net cash flow across all historical years
             * Standard deviation for that month
             * Week-of-month specific patterns within the month
        
        2. MODEL TRAINING:
           - Train specified model (XGBoost, LightGBM, etc.) on full history
        
        3. ITERATIVE FORECASTING (for each of 6 months):
           a. For each week in the month:
              - Build feature row from prediction history
              - Get base ML prediction
              - Look up historical pattern for this calendar month/week
              - Blend: y = 0.4 × ML_pred + 0.6 × historical_mean
              - Add noise sampled from historical variance
              - Append to history for next iteration
           
           b. After completing 4 weeks:
              - Create MonthlyForecast object
              - Calculate cumulative running total
              - Move to next month
        
        MATHEMATICAL FORMULATION
        ========================
        
        For forecast week w in calendar month m, week-of-month wom:
        
        y_base = Model.predict(features)
        
        μ_m,wom = mean(historical_data[month == m AND week_of_month == wom])
        σ_m,wom = std(historical_data[month == m AND week_of_month == wom])
        
        y_blended = 0.4 × y_base + 0.6 × μ_m,wom
        
        noise ~ N(0, σ_m,wom × 0.4)
        
        y_final = y_blended + noise
        
        Args:
            model_name: Name of the model architecture to use
            available_features: List of feature column names available in data
            base_df: Historical weekly data DataFrame
            num_months: Number of months to forecast (default: 6)
            weeks_per_month: Weeks per month assumption (default: 4)
            
        Returns:
            Tuple containing:
            - all_dates: List of all forecasted dates
            - all_preds: Array of all weekly predictions
            - monthly_forecasts: List of MonthlyForecast objects for breakdown
        """
        full_X = base_df[available_features].values
        full_y = base_df["Total_Net"].values

        def build_estimator() -> Tuple[object, Optional[StandardScaler]]:
            if model_name == "XGBoost" and HAS_XGB:
                return (
                    xgb.XGBRegressor(
                        n_estimators=200, max_depth=4, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9,
                        objective="reg:squarederror", random_state=42, verbosity=0,
                    ),
                    None,
                )
            if model_name == "LightGBM" and HAS_LGB:
                return (
                    lgb.LGBMRegressor(
                        n_estimators=200, max_depth=5, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1,
                    ),
                    None,
                )
            if model_name == "RandomForest":
                return (
                    RandomForestRegressor(
                        n_estimators=300, max_depth=8, min_samples_split=3,
                        min_samples_leaf=2, random_state=42, n_jobs=-1,
                    ),
                    None,
                )
            if model_name == "GradientBoosting":
                return (
                    GradientBoostingRegressor(
                        n_estimators=250, max_depth=4, learning_rate=0.05,
                        subsample=0.9, random_state=42
                    ),
                    None,
                )
            return Ridge(alpha=1.0), StandardScaler()

        estimator, scaler = build_estimator()
        X_fit = scaler.fit_transform(full_X) if scaler else full_X
        estimator.fit(X_fit, full_y)

        # Prepare historical data with month info for year-over-year patterns
        base_df = base_df.copy()
        base_df["Month"] = base_df["Week_Start"].dt.month
        base_df["Week_of_Month"] = ((base_df["Week_Start"].dt.day - 1) // 7) + 1
        base_df["Year"] = base_df["Week_Start"].dt.year
        
        # Build year-over-year patterns: for each month, get historical stats
        # This assumes "every year rhymes" - same month patterns repeat
        monthly_patterns = {}
        for month in range(1, 13):
            month_data = base_df[base_df["Month"] == month]["Total_Net"]
            if len(month_data) > 0:
                monthly_patterns[month] = {
                    "mean": float(month_data.mean()),
                    "std": float(month_data.std()) if len(month_data) > 1 else float(np.std(full_y)),
                    "min": float(month_data.min()),
                    "max": float(month_data.max()),
                    "weekly_values": list(month_data.values),  # Store actual weekly values from this month
                }
        
        # Also build week-of-month patterns within each calendar month
        month_week_patterns = {}
        for month in range(1, 13):
            month_week_patterns[month] = {}
            month_data = base_df[base_df["Month"] == month]
            for wom in range(1, 6):
                wom_data = month_data[month_data["Week_of_Month"] == wom]["Total_Net"]
                if len(wom_data) > 0:
                    month_week_patterns[month][wom] = {
                        "mean": float(wom_data.mean()),
                        "std": float(wom_data.std()) if len(wom_data) > 1 else float(np.std(full_y)) * 0.5,
                        "values": list(wom_data.values),
                    }
        
        # Overall stats as fallback
        overall_mean = full_y.mean()
        hist_std = float(np.std(full_y))

        # Initialize history with actual data
        history_net = list(full_y)
        inflow_history = list(base_df.get("Total_Inflow", pd.Series([0] * len(base_df))).values)
        outflow_history = list(base_df.get("Total_Outflow", pd.Series([0] * len(base_df))).values)
        last_row = base_df.iloc[-1]
        cursor_date = pd.to_datetime(last_row["Week_Start"]).to_pydatetime()

        all_dates: List[datetime] = []
        all_preds: List[float] = []
        monthly_forecasts: List[MonthlyForecast] = []
        cumulative_total = 0.0

        # Forecast month by month
        np.random.seed(42)  # For reproducibility
        
        for month_idx in range(num_months):
            month_dates: List[datetime] = []
            month_preds: List[float] = []

            # Forecast 4 weeks for this month
            for week_idx in range(weeks_per_month):
                cursor_date = cursor_date + timedelta(days=7)
                forecast_month = cursor_date.month
                week_of_month = ((cursor_date.day - 1) // 7) + 1
                
                # Build features using history (which includes previous months' predictions)
                feat_row = self._future_feature_row(
                    cursor_date, history_net, last_row, inflow_history, outflow_history, base_df
                )
                X_next = np.array([[feat_row.get(col, 0) for col in available_features]])
                X_next = scaler.transform(X_next) if scaler else X_next
                
                # Base prediction from model
                y_hat_base = float(estimator.predict(X_next)[0])
                
                # === Year-over-year adjustment: use same month from previous year(s) ===
                # Get historical pattern for this specific month
                if forecast_month in monthly_patterns:
                    month_pattern = monthly_patterns[forecast_month]
                    month_mean = month_pattern["mean"]
                    month_std = month_pattern["std"]
                    
                    # Get week-of-month specific pattern for this calendar month
                    if forecast_month in month_week_patterns and week_of_month in month_week_patterns[forecast_month]:
                        wom_pattern = month_week_patterns[forecast_month][week_of_month]
                        wom_mean = wom_pattern["mean"]
                        wom_std = wom_pattern["std"]
                        historical_values = wom_pattern["values"]
                    else:
                        wom_mean = month_mean
                        wom_std = month_std
                        historical_values = month_pattern["weekly_values"]
                else:
                    month_mean = overall_mean
                    month_std = hist_std
                    wom_mean = overall_mean
                    wom_std = hist_std
                    historical_values = []
                
                # Get entity-specific weights
                entity_config = self._get_entity_config(self.current_entity) if self.current_entity else {}
                model_weight = entity_config.get("model_weight", 0.4)
                historical_weight = entity_config.get("historical_weight", 0.6)
                
                # Calculate year-over-year seasonal adjustment
                if overall_mean != 0:
                    # How much does this month typically deviate from overall average?
                    month_seasonal_factor = (month_mean - overall_mean) / max(abs(overall_mean), 1)
                else:
                    month_seasonal_factor = 0
                
                # Adjust base prediction towards historical month pattern
                y_hat_adjusted = (model_weight * y_hat_base) + (historical_weight * wom_mean)
                
                # Add realistic noise based on historical variance for this specific month/week
                # Sample from historical distribution to maintain realistic fluctuations
                if len(historical_values) > 0 and len(historical_values) >= 2:
                    # Use historical values to inform noise
                    hist_variation = np.std(historical_values)
                    noise = np.random.normal(0, hist_variation * 0.4)
                else:
                    noise = np.random.normal(0, wom_std * 0.3)
                
                y_hat = y_hat_adjusted + noise
                
                # Apply outlier clipping for high-variance entities
                if entity_config.get("clip_outliers", False):
                    q1, q3 = np.percentile(history_net[-20:] if len(history_net) >= 20 else history_net, [25, 75])
                    iqr = q3 - q1
                    multiplier = entity_config.get("outlier_multiplier", 2.5)
                    lower_bound = q1 - multiplier * iqr
                    upper_bound = q3 + multiplier * iqr
                    y_hat = np.clip(y_hat, lower_bound, upper_bound)
                
                # Store prediction
                month_dates.append(cursor_date)
                month_preds.append(y_hat)
                all_dates.append(cursor_date)
                all_preds.append(y_hat)
                
                # Update history with this prediction for next iteration
                history_net.append(y_hat)

            # Calculate cumulative total up to this month
            cumulative_total += sum(month_preds)
            
            # Store monthly forecast
            monthly_forecasts.append(MonthlyForecast(
                month_num=month_idx + 1,
                dates=month_dates,
                predictions=month_preds,
                cumulative_net=cumulative_total,
            ))

        return all_dates, np.array(all_preds), monthly_forecasts

    def forecast_entity(self, entity: str, df: pd.DataFrame) -> Optional[ForecastArtifacts]:
        df = df.copy().sort_values("Week_Start")
        df["Week_Start"] = pd.to_datetime(df["Week_Start"])
        
        # Set current entity for use by other methods
        self.current_entity = entity
        config = self._get_entity_config(entity)
        
        # Add week-of-month dummy features for entities that benefit from it
        if config["add_week_dummies"]:
            df = self._add_week_dummies(df)
        
        # Get all available features dynamically (including category features)
        all_available_features = self._get_all_available_features(df)
        
        # Filter to non-null features
        df_clean = df.dropna(subset=[f for f in all_available_features if f in df.columns])
        if len(df_clean) < self.backtest_weeks + 8:
            print(f"  Skipping {entity}: not enough history after cleaning.")
            return None

        train_df = df_clean.iloc[:-self.backtest_weeks]
        test_df = df_clean.iloc[-self.backtest_weeks:]
        
        # Perform feature selection if enabled
        if self.use_feature_selection and len(all_available_features) > 15:
            print(f"    📊 Feature selection from {len(all_available_features)} candidates...")
            X_train_all = train_df[all_available_features].values
            y_train_all = train_df["Total_Net"].values
            
            selected_features, importance = self._forward_feature_selection(
                X_train_all, y_train_all, all_available_features, max_features=20
            )
            
            # Store for later use
            self.selected_features[entity] = selected_features
            self.feature_importance[entity] = importance
            
            # Ensure we have at least core features
            if len(selected_features) < 5:
                selected_features = [f for f in CORE_FEATURE_COLS if f in all_available_features][:10]
            
            available_features = selected_features
            print(f"    ✅ Selected {len(available_features)} features")
            
            # Show top features
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"    🔝 Top features: {', '.join([f[0] for f in top_features])}")
        else:
            available_features = [f for f in all_available_features if f in df_clean.columns]
            self.selected_features[entity] = available_features
        
        X_train, y_train = train_df[available_features].values, train_df["Total_Net"].values
        X_test, y_test = test_df[available_features].values, test_df["Total_Net"].values

        results = self._train_models(X_train, y_train, X_test, y_test, available_features, entity=entity)
        
        # Select best model (consider entity preference)
        if config["model_preference"]:
            # Prefer specified models if they perform reasonably well
            best_name = None
            best_rmse = float('inf')
            for preferred in config["model_preference"]:
                if preferred in results:
                    if results[preferred].metrics["RMSE"] < best_rmse:
                        best_rmse = results[preferred].metrics["RMSE"]
                        best_name = preferred
            if best_name is None:
                best_name = min(results.keys(), key=lambda k: results[k].metrics["RMSE"])
        else:
            best_name = min(results.keys(), key=lambda k: results[k].metrics["RMSE"])
        
        best = results[best_name]
        
        # Get feature importance from best model
        if best.model is not None:
            model_importance = self._get_feature_importance_from_model(best.model, available_features)
            if model_importance:
                self.feature_importance[entity] = model_importance

        # 1-month forecast (4 weeks)
        short_dates, short_pred = self._iterative_forecast(best_name, available_features, df_clean, self.forecast_weeks)
        
        # 6-month forecast with month-by-month iteration
        long_dates, long_pred, monthly_forecasts = self._iterative_forecast_monthly(
            best_name, available_features, df_clean, num_months=6, weeks_per_month=4
        )

        print(
            f"  {entity}: {best_name} | RMSE={best.metrics['RMSE']:.2f} | MAE={best.metrics['MAE']:.2f}"
        )
        print(f"    → 6-month forecast generated iteratively (month-by-month)")

        return ForecastArtifacts(
            best_model=best_name,
            backtest_dates=test_df["Week_Start"],
            backtest_actual=test_df["Total_Net"].values,
            backtest_pred=best.backtest_pred,
            future_short_dates=short_dates,
            future_short_pred=short_pred,
            future_long_dates=long_dates,
            future_long_pred=long_pred,
            monthly_forecasts=monthly_forecasts,
            metrics=best.metrics,
            all_model_metrics={name: res.metrics for name, res in results.items()},
        )

    def run(self, entity_frames: Dict[str, pd.DataFrame]) -> Dict[str, ForecastArtifacts]:
        """
        Run forecasting pipeline for all entities.
        
        Args:
            entity_frames: Dictionary mapping entity names to their weekly DataFrames
            
        Returns:
            Dictionary mapping entity names to their ForecastArtifacts
        """
        all_results: Dict[str, ForecastArtifacts] = {}
        print("\n=== Training & forecasting ===")
        for entity, frame in entity_frames.items():
            print(f"\nEntity: {entity}")
            art = self.forecast_entity(entity, frame)
            if art:
                all_results[entity] = art
        
        # Print feature importance summary
        self._print_feature_importance_summary()
        
        return all_results
    
    def _print_feature_importance_summary(self) -> None:
        """Print a summary of feature importance across all entities."""
        if not self.feature_importance:
            return
        
        print("\n" + "=" * 70)
        print("  📊 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)
        
        # Aggregate importance across all entities
        aggregated = {}
        for entity, importance in self.feature_importance.items():
            for feature, imp in importance.items():
                if feature not in aggregated:
                    aggregated[feature] = []
                aggregated[feature].append(imp)
        
        # Calculate mean importance (handle NaN)
        mean_importance = {}
        for f, imps in aggregated.items():
            valid_imps = [x for x in imps if not np.isnan(x)]
            if valid_imps:
                mean_importance[f] = np.mean(valid_imps)
            else:
                mean_importance[f] = 0.0
        
        sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n  Top 15 Most Important Features (averaged across entities):")
        print("  " + "-" * 50)
        
        # Normalize for bar visualization
        max_imp = max(mean_importance.values()) if mean_importance else 1.0
        if max_imp == 0:
            max_imp = 1.0
        
        for i, (feature, imp) in enumerate(sorted_features[:15], 1):
            # Determine feature category
            if feature.startswith("Cat_"):
                category = "📦 Category"
            elif "Lag" in feature:
                category = "⏮️  Lag"
            elif "Rolling" in feature:
                category = "📈 Rolling"
            elif feature in ["Month", "Quarter", "Week_of_Month", "Is_Month_End"]:
                category = "📅 Temporal"
            elif "Ratio" in feature:
                category = "📊 Ratio"
            else:
                category = "📋 Other"
            
            normalized_imp = imp / max_imp
            bar_len = int(normalized_imp * 30)
            bar = "█" * bar_len
            print(f"  {i:2d}. {feature:40s} {category:12s} {imp:.4f} {bar}")
        
        # Category feature analysis
        cat_features = [f for f, _ in sorted_features if f.startswith("Cat_") or "Ratio" in f]
        if cat_features:
            print(f"\n  📦 Category-based features in top 15: {len([f for f in cat_features if f in dict(sorted_features[:15])])}")
            print(f"  📦 Most important category feature: {cat_features[0] if cat_features else 'None'}")
        
        # Per-entity selected features
        print("\n  Selected Features per Entity:")
        print("  " + "-" * 50)
        for entity, features in self.selected_features.items():
            cat_count = len([f for f in features if f.startswith("Cat_") or "Ratio" in f])
            print(f"  {entity}: {len(features)} features ({cat_count} category-based)")
        
        print("=" * 70 + "\n")


# ==============================================================================
# INTERACTIVE DASHBOARD BUILDER MODULE
# ==============================================================================
class InteractiveDashboardBuilder:
    """
    Generates interactive HTML dashboard with Plotly.js visualizations.
    
    This class creates a self-contained HTML file that provides comprehensive
    cash flow forecast visualization with the AstraZeneca corporate theme.
    No Python server required - all interactivity is handled client-side.
    
    Dashboard Sections:
    ------------------
    1. ENTITY OVERVIEW: Summary metrics and model information
       - Best model name and key performance metrics
       - Total historical vs forecasted cash flow
       
    2. BACKTEST VALIDATION: Last month actual vs predicted
       - Line chart comparing actual values to model predictions
       - Validates model accuracy on held-out data
       
    3. 1-MONTH FORECAST: Short-term tactical projection
       - Historical data + 4-week forecast
       - Includes confidence context from recent patterns
       
    4. 6-MONTH FORECAST: Long-term strategic projection
       - Full historical timeline + 24-week forecast
       - Month-by-month breakdown showing iterative forecast structure
       
    5. CATEGORY ANALYSIS: Cash flow composition (if data available)
       - Top inflow categories
       - Top outflow categories
    
    Technical Implementation:
    ------------------------
    - Plotly.js for interactive charts (zoom, pan, hover tooltips)
    - Pure HTML/CSS/JavaScript - no external dependencies
    - Embedded Plotly.js from CDN for charting
    - Responsive design adapts to screen size
    - Inter font family for modern typography
    
    Color Theme (AstraZeneca):
    -------------------------
    - Mulberry (#830051): Primary color, 6-month forecast
    - Navy (#003865): Historical actual data
    - Light Blue (#68D2DF): 1-month forecast
    - Gold (#F0AB00): Backtest predictions
    - Lime Green (#C4D600): Positive values
    - Magenta (#D0006F): Negative values
    
    Attributes:
        paths: PathConfig instance for output file path management
    """

    def __init__(self, paths: PathConfig) -> None:
        self.paths = paths

    def _format_currency(self, value: float) -> str:
        """Format numeric value as currency string with appropriate scale."""
        if abs(value) >= 1e6:
            return f"${value/1e6:,.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:,.1f}K"
        return f"${value:,.0f}"

    def _scrape_country_sentiments(self, entity_countries: Dict[str, Dict]) -> Dict[str, Dict]:
        """Scrape fresh news sentiment for each country using the FinBERT-based scraper."""
        import json
        import sys
        import importlib.util
        
        country_sentiments = {}
        
        # Dynamically load main_scraper from News_scraper directory
        news_scraper_path = self.paths.project_root / "News_scraper"
        main_scraper_file = news_scraper_path / "main_scraper.py"
        
        if not main_scraper_file.exists():
            print(f"  ⚠️ main_scraper.py not found at {main_scraper_file}")
            return {}
        
        try:
            # Add News_scraper to path for its dependencies
            if str(news_scraper_path) not in sys.path:
                sys.path.insert(0, str(news_scraper_path))
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("main_scraper", main_scraper_file)
            main_scraper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_scraper)
            
            analyze_news_sync = main_scraper.analyze_news_sync
            get_full_summary = main_scraper.get_full_summary
            print("  🤖 Using FinBERT-based sentiment analysis from main_scraper.py")
        except Exception as e:
            print(f"  ⚠️ Could not load main_scraper: {e}")
            print("  ℹ️ Make sure News_scraper dependencies are installed (transformers, playwright, etc.)")
            return {}
        
        for entity, info in entity_countries.items():
            country = info['country']
            search_term = info['search_term']
            
            print(f"  🔍 Scraping news for {country}...")
            
            try:
                # Use the existing FinBERT-based scraper
                results = analyze_news_sync(
                    keywords=[search_term],
                    max_articles=1,
                    print_report=False,
                    save_json=False
                )
                
                # Get the full summary with articles
                summary = get_full_summary(results)
                
                if summary['total_articles'] > 0:
                    # Convert to our expected format
                    articles = []
                    for article in summary['articles']:
                        articles.append({
                            'title': article.get('title', 'No title'),
                            'url': article.get('url', '#'),
                            'publish_date': article.get('publish_date'),
                            'final_score': round(article.get('score', 0), 3),
                            'source': article.get('source', 'unknown')
                        })
                    
                    country_sentiments[entity] = {
                        'country': country,
                        'average_score': round(summary['score'], 3),
                        'total_articles': summary['total_articles'],
                        'articles': articles
                    }
                    print(f"    ✓ {country}: {summary['total_articles']} articles, avg score: {summary['score']:.2f} ({summary['verdict']})")
                else:
                    print(f"    ⚠️ No articles found for {country}")
                    country_sentiments[entity] = {
                        'country': country,
                        'average_score': 0,
                        'total_articles': 0,
                        'articles': []
                    }
                    
            except Exception as e:
                print(f"    ⚠️ Error scraping {country}: {e}")
                country_sentiments[entity] = {
                    'country': country,
                    'average_score': 0,
                    'total_articles': 0,
                    'articles': []
                }
        
        # Save the scraped data for future use
        if country_sentiments:
            sentiment_dir = self.paths.project_root / "News_scraper" / "country_sentiments"
            sentiment_dir.mkdir(parents=True, exist_ok=True)
            
            for entity, data in country_sentiments.items():
                country_name = data['country'].lower().replace(' ', '_')
                output_file = sentiment_dir / f"{country_name}_sentiment.json"
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, default=str)
                except Exception as e:
                    print(f"  ⚠️ Could not save {country_name}: {e}")
            
            print(f"  💾 Saved sentiment data to {sentiment_dir}")
        
        return country_sentiments

    def _load_country_sentiments(self, entity_countries: Dict[str, Dict]) -> Dict[str, Dict]:
        """Load country sentiment data from cached JSON files (fallback)."""
        import json
        country_sentiments = {}
        sentiment_dir = self.paths.project_root / "News_scraper" / "country_sentiments"
        
        if not sentiment_dir.exists():
            print(f"  ⚠️ Sentiment directory not found: {sentiment_dir}")
            return country_sentiments
        
        for entity, info in entity_countries.items():
            country = info['country']
            country_file = sentiment_dir / f"{country.lower().replace(' ', '_')}_sentiment.json"
            
            if country_file.exists():
                try:
                    with open(country_file, 'r', encoding='utf-8') as f:
                        country_sentiments[entity] = json.load(f)
                    print(f"  ✅ Loaded cached sentiment for {entity} ({country})")
                except Exception as e:
                    print(f"  ⚠️ Error loading {country}: {e}")
            else:
                print(f"  ℹ️ No sentiment file for {country}")
                # Create a default entry
                country_sentiments[entity] = {
                    'country': country,
                    'average_score': 0,
                    'total_articles': 0,
                    'articles': []
                }
        
        return country_sentiments

    def _generate_entity_news_section(
        self, 
        entity: str, 
        country: str, 
        articles: List[Dict], 
        sentiment_score: float
    ) -> str:
        """Generate the news section HTML for a specific entity."""
        # Determine sentiment badge
        if sentiment_score > 0.1:
            sentiment_label = "POSITIVE"
            sentiment_color = "#27ae60"
            sentiment_bg = "#27ae6020"
        elif sentiment_score < -0.1:
            sentiment_label = "NEGATIVE"
            sentiment_color = "#e74c3c"
            sentiment_bg = "#e74c3c20"
        else:
            sentiment_label = "NEUTRAL"
            sentiment_color = "#7f8c8d"
            sentiment_bg = "#7f8c8d20"
        
        # Generate news cards
        news_cards_html = ""
        if articles:
            for article in articles[:5]:  # Show max 5 articles
                score = article.get('final_score', 0)
                if score > 0.1:
                    card_color = "#27ae60"
                    tag_text = "Positive"
                    tag_bg = "#27ae6015"
                elif score < -0.1:
                    card_color = "#e74c3c"
                    tag_text = "Negative"
                    tag_bg = "#e74c3c15"
                else:
                    card_color = "#7f8c8d"
                    tag_text = "Neutral"
                    tag_bg = "#7f8c8d15"
                
                title = article.get('title', 'No title')
                url = article.get('url', '#')
                source = article.get('source', 'News')
                
                news_cards_html += f'''
                <div class="news-card" style="border-left-color: {card_color};">
                    <div class="news-card-title">
                        <a href="{url}" target="_blank">{title[:120]}{'...' if len(title) > 120 else ''}</a>
                    </div>
                    <div class="news-card-meta">
                        <span>📰 {source}</span>
                        <span class="news-sentiment-tag" style="background: {tag_bg}; color: {card_color};">
                            {tag_text} ({score:.2f})
                        </span>
                    </div>
                </div>
                '''
        else:
            news_cards_html = '''
            <div style="text-align: center; padding: 2rem; color: #888;">
                <p>📭 No recent news articles available for this country.</p>
            </div>
            '''
        
        return f'''
        <div class="entity-news-section">
            <div class="news-header">
                <div class="news-title">
                    📰 {country} Market News
                </div>
                <div class="sentiment-badge" style="background: {sentiment_bg}; color: {sentiment_color};">
                    Sentiment: {sentiment_label} ({sentiment_score:.2f})
                </div>
            </div>
            <div class="news-grid">
                {news_cards_html}
            </div>
        </div>
        '''

    def _get_category_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze top cash flow categories."""
        cat_cols = [c for c in df.columns if c.startswith("Cat_") and c.endswith("_Net")]
        if not cat_cols:
            return {"top_inflow": [], "top_outflow": []}

        cat_totals = {}
        for col in cat_cols:
            cat_name = col.replace("Cat_", "").replace("_Net", "").replace("_", " ")
            cat_totals[cat_name] = df[col].sum()

        sorted_cats = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
        top_inflow = [(k, v) for k, v in sorted_cats if v > 0][:5]
        top_outflow = [(k, abs(v)) for k, v in sorted_cats if v < 0][:5]

        return {"top_inflow": top_inflow, "top_outflow": top_outflow, "all": cat_totals}

    def _generate_entity_html(
        self,
        entity: str,
        df: pd.DataFrame,
        art: ForecastArtifacts,
    ) -> str:
        """Generate HTML section for one entity."""
        df = df.copy()
        df["Week_Start"] = pd.to_datetime(df["Week_Start"])

        # Calculate key metrics
        total_inflow = df["Total_Inflow"].sum()
        total_outflow = abs(df["Total_Outflow"].sum())
        net_position = df["Total_Net"].sum()
        avg_weekly_net = df["Total_Net"].mean()
        volatility = df["Total_Net"].std()

        # Short-term forecast metrics
        short_forecast_total = float(np.sum(art.future_short_pred))
        short_forecast_avg = float(np.mean(art.future_short_pred))

        # Long-term forecast metrics
        long_forecast_total = float(np.sum(art.future_long_pred))
        long_forecast_avg = float(np.mean(art.future_long_pred))

        # Category analysis
        cat_analysis = self._get_category_analysis(df)

        # Backtest accuracy
        mae = art.metrics.get("MAE", 0)
        rmse = art.metrics.get("RMSE", 0)
        mape = art.metrics.get("MAPE", 0)
        direction_acc = art.metrics.get("Direction_Accuracy", 0)

        # Prepare data for charts (convert to JSON-safe format)
        history_dates = [d.strftime("%Y-%m-%d") for d in df["Week_Start"]]
        history_net = df["Total_Net"].tolist()
        history_inflow = df["Total_Inflow"].tolist()
        history_outflow = [abs(x) for x in df["Total_Outflow"].tolist()]

        backtest_dates = [d.strftime("%Y-%m-%d") for d in art.backtest_dates]
        backtest_actual = art.backtest_actual.tolist()
        backtest_pred = art.backtest_pred.tolist()

        short_dates = [d.strftime("%Y-%m-%d") for d in art.future_short_dates]
        short_pred = art.future_short_pred.tolist()

        long_dates = [d.strftime("%Y-%m-%d") for d in art.future_long_dates]
        long_pred = art.future_long_pred.tolist()

        # Prepare monthly forecast data for visualization
        monthly_data = []
        month_colors = [
            AZ_COLORS["light_blue"],   # Month 1
            AZ_COLORS["lime_green"],   # Month 2
            AZ_COLORS["gold"],         # Month 3
            AZ_COLORS["magenta"],      # Month 4
            AZ_COLORS["purple"],       # Month 5
            AZ_COLORS["mulberry"],     # Month 6
        ]
        for mf in art.monthly_forecasts:
            monthly_data.append({
                "month_num": mf.month_num,
                "dates": [d.strftime("%Y-%m-%d") for d in mf.dates],
                "predictions": mf.predictions,
                "total": sum(mf.predictions),
                "cumulative": mf.cumulative_net,
                "color": month_colors[mf.month_num - 1] if mf.month_num <= len(month_colors) else AZ_COLORS["graphite"],
            })

        # Top categories for display
        top_inflow_cats = cat_analysis.get("top_inflow", [])[:3]
        top_outflow_cats = cat_analysis.get("top_outflow", [])[:3]

        # Liquidity risk assessment
        if short_forecast_total < 0:
            liquidity_status = "⚠️ At Risk"
            liquidity_color = AZ_COLORS["magenta"]
            liquidity_msg = f"Expected net outflow of {self._format_currency(abs(short_forecast_total))} over the next month. Monitor closely."
        elif short_forecast_avg < avg_weekly_net * 0.5:
            liquidity_status = "⚡ Moderate"
            liquidity_color = AZ_COLORS["gold"]
            liquidity_msg = f"Cash flow expected to decrease. Forecasted weekly average: {self._format_currency(short_forecast_avg)}"
        else:
            liquidity_status = "✅ Stable"
            liquidity_color = AZ_COLORS["lime_green"]
            liquidity_msg = f"Healthy cash position expected. Net inflow forecast: {self._format_currency(short_forecast_total)}"

        # Long-term sustainability
        if long_forecast_total < 0:
            sustainability = "Needs Attention"
            sust_color = AZ_COLORS["magenta"]
        elif long_forecast_total < net_position * 0.3:
            sustainability = "Monitor"
            sust_color = AZ_COLORS["gold"]
        else:
            sustainability = "Sustainable"
            sust_color = AZ_COLORS["lime_green"]

        return f'''
        <div class="entity-section" id="{entity}">
            <div class="entity-header">
                <h2><span class="entity-badge">{entity}</span> Cash Flow Analysis</h2>
                <div class="model-badge">Best Model: <strong>{art.best_model}</strong></div>
            </div>

            <!-- KPI Cards Row -->
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-icon">💰</div>
                    <div class="kpi-value" style="color: {AZ_COLORS['lime_green']}">{self._format_currency(total_inflow)}</div>
                    <div class="kpi-label">Total Inflow</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">💸</div>
                    <div class="kpi-value" style="color: {AZ_COLORS['magenta']}">{self._format_currency(total_outflow)}</div>
                    <div class="kpi-label">Total Outflow</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">📊</div>
                    <div class="kpi-value" style="color: {AZ_COLORS['navy']}">{self._format_currency(net_position)}</div>
                    <div class="kpi-label">Net Position</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">📈</div>
                    <div class="kpi-value" style="color: {AZ_COLORS['light_blue']}">{self._format_currency(short_forecast_total)}</div>
                    <div class="kpi-label">1-Month Forecast</div>
                </div>
            </div>

            <!-- Main Charts Section -->
            <div class="charts-grid">
                <!-- Backtest Chart -->
                <div class="chart-container full-width">
                    <h3>🔍 Backtest: Actual vs Forecast (Last 4 Weeks)</h3>
                    <p class="chart-description">Comparing model predictions against actual cash flows to evaluate forecast accuracy.</p>
                    <div id="backtest-{entity}" class="chart"></div>
                    <div class="metrics-row">
                        <div class="metric">
                            <span class="metric-label">MAE</span>
                            <span class="metric-value">{self._format_currency(mae)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">RMSE</span>
                            <span class="metric-value">{self._format_currency(rmse)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">MAPE</span>
                            <span class="metric-value">{mape:.1f}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Direction Accuracy</span>
                            <span class="metric-value">{direction_acc:.1f}%</span>
                        </div>
                    </div>
                </div>

                <!-- Inflow/Outflow Chart -->
                <div class="chart-container full-width">
                    <h3>💹 Cash Flow Overview: Inflow vs Outflow</h3>
                    <p class="chart-description">Weekly breakdown of cash inflows and outflows across the historical period.</p>
                    <div id="inflow-outflow-{entity}" class="chart"></div>
                </div>

                <!-- Short-term Forecast -->
                <div class="chart-container">
                    <h3>📅 1-Month Ahead Forecast</h3>
                    <p class="chart-description">Weekly cash flow predictions for the next 4 weeks based on {art.best_model} model.</p>
                    <div id="short-forecast-{entity}" class="chart"></div>
                    <div class="insight-box" style="border-left-color: {liquidity_color}">
                        <div class="insight-header">
                            <span class="status-badge" style="background: {liquidity_color}">{liquidity_status}</span>
                            <strong>Short-term Liquidity Assessment</strong>
                        </div>
                        <p>{liquidity_msg}</p>
                    </div>
                </div>

                <!-- Long-term Forecast -->
                <div class="chart-container">
                    <h3>📆 6-Month Ahead Forecast (Iterative Month-by-Month)</h3>
                    <p class="chart-description">Each month's forecast uses previous months' predictions as input features, building iteratively.</p>
                    <div id="long-forecast-{entity}" class="chart"></div>
                    <div class="insight-box" style="border-left-color: {sust_color}">
                        <div class="insight-header">
                            <span class="status-badge" style="background: {sust_color}">{sustainability}</span>
                            <strong>Long-term Sustainability</strong>
                        </div>
                        <p>6-month forecasted total: {self._format_currency(long_forecast_total)}. {"Consider strategic adjustments." if sustainability != "Sustainable" else "Position appears sustainable."}</p>
                    </div>
                </div>
            </div>

            <!-- Month-by-Month Forecast Breakdown -->
            <div class="monthly-breakdown">
                <h3>📊 Month-by-Month Forecast Breakdown</h3>
                <p class="chart-description">Iterative forecasting: each month's prediction feeds into the next month's features, creating a cascading forecast.</p>
                <div class="monthly-grid">
                    {"".join([f'''
                    <div class="monthly-card" style="border-top: 4px solid {md['color']}">
                        <div class="monthly-header">
                            <span class="month-badge" style="background: {md['color']}">Month {md['month_num']}</span>
                            <span class="month-dates">{md['dates'][0]} → {md['dates'][-1]}</span>
                        </div>
                        <div class="monthly-stats">
                            <div class="monthly-stat">
                                <span class="stat-value">{self._format_currency(md['total'])}</span>
                                <span class="stat-label">Monthly Net</span>
                            </div>
                            <div class="monthly-stat">
                                <span class="stat-value">{self._format_currency(md['cumulative'])}</span>
                                <span class="stat-label">Cumulative</span>
                            </div>
                        </div>
                        <div class="weekly-detail">
                            {"".join([f'<span class="week-val" style="color: {"#22c55e" if v >= 0 else "#ef4444"}">{self._format_currency(v)}</span>' for v in md['predictions']])}
                        </div>
                    </div>
                    ''' for md in monthly_data])}
                </div>
                <div id="monthly-chart-{entity}" class="chart" style="margin-top: 1.5rem;"></div>
            </div>

            <!-- Category Analysis -->
            <div class="category-analysis">
                <h3>🏷️ Key Cash Flow Drivers</h3>
                <div class="category-grid">
                    <div class="category-box inflow">
                        <h4>Top Inflow Categories</h4>
                        <ul>
                            {"".join([f'<li><span class="cat-name">{cat}</span><span class="cat-value">{self._format_currency(val)}</span></li>' for cat, val in top_inflow_cats]) if top_inflow_cats else '<li>No significant inflows</li>'}
                        </ul>
                    </div>
                    <div class="category-box outflow">
                        <h4>Top Outflow Categories</h4>
                        <ul>
                            {"".join([f'<li><span class="cat-name">{cat}</span><span class="cat-value">-{self._format_currency(val)}</span></li>' for cat, val in top_outflow_cats]) if top_outflow_cats else '<li>No significant outflows</li>'}
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Model Performance Table -->
            <div class="model-comparison">
                <h3>🤖 Model Performance Comparison</h3>
                <table class="model-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>MAPE (%)</th>
                            <th>Direction Acc (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([f'''<tr class="{'best-model' if name == art.best_model else ''}">
                            <td>{name} {'⭐' if name == art.best_model else ''}</td>
                            <td>{self._format_currency(m.get('MAE', 0))}</td>
                            <td>{self._format_currency(m.get('RMSE', 0))}</td>
                            <td>{m.get('MAPE', 0):.1f}</td>
                            <td>{m.get('Direction_Accuracy', 0):.1f}</td>
                        </tr>''' for name, m in art.all_model_metrics.items()])}
                    </tbody>
                </table>
                <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 8px; font-size: 0.85rem; color: #555;">
                    <strong>📊 Metric Definitions:</strong>
                    <ul style="margin: 0.5rem 0 0 1.5rem; line-height: 1.6;">
                        <li><strong>MAE</strong> (Mean Absolute Error): Average dollar difference between predicted and actual values — lower is better.</li>
                        <li><strong>RMSE</strong> (Root Mean Square Error): Similar to MAE but penalizes large errors more heavily — lower is better.</li>
                        <li><strong>MAPE</strong> (Mean Absolute Percentage Error): Average percentage error relative to actual values — lower is better.</li>
                        <li><strong>Direction Acc</strong> (Direction Accuracy): Percentage of times the model correctly predicted if cash flow would go up or down — higher is better.</li>
                    </ul>
                </div>
            </div>
        </div>

        <script>
        // Backtest Chart
        Plotly.newPlot('backtest-{entity}', [
            {{
                x: {json.dumps(backtest_dates)},
                y: {json.dumps(backtest_actual)},
                type: 'scatter',
                mode: 'lines+markers',
                name: '✓ Actual Values',
                line: {{color: '{AZ_COLORS["navy"]}', width: 4}},
                marker: {{size: 14, symbol: 'circle'}},
                hovertemplate: '<b>Actual</b><br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
            }},
            {{
                x: {json.dumps(backtest_dates)},
                y: {json.dumps(backtest_pred)},
                type: 'scatter',
                mode: 'lines+markers',
                name: '🔮 Model Prediction',
                line: {{color: '{AZ_COLORS["gold"]}', width: 4, dash: 'dash'}},
                marker: {{size: 12, symbol: 'diamond'}},
                hovertemplate: '<b>Prediction</b><br>Date: %{{x}}<br>Predicted: $%{{y:,.0f}}<extra></extra>'
            }}
        ], {{
            height: 350,
            margin: {{t: 50, b: 60, l: 80, r: 40}},
            xaxis: {{
                title: {{text: 'Week', font: {{size: 12, color: '#666'}}}},
                tickangle: -30,
                tickfont: {{size: 10}},
                gridcolor: '#eee'
            }},
            yaxis: {{
                title: {{text: 'Net Cash Flow (USD)', font: {{size: 12, color: '#666'}}}},
                tickformat: ',.0f',
                tickfont: {{size: 10}},
                gridcolor: '#eee',
                zerolinecolor: '#333',
                zerolinewidth: 2
            }},
            legend: {{
                x: 0.5,
                y: 1.12,
                xanchor: 'center',
                orientation: 'h',
                bgcolor: 'rgba(255,255,255,0.95)',
                bordercolor: '#ddd',
                borderwidth: 1,
                font: {{size: 11}}
            }},
            hovermode: 'x unified',
            shapes: [{{type: 'line', x0: '{backtest_dates[0]}', x1: '{backtest_dates[-1]}', y0: 0, y1: 0, line: {{color: '#888', width: 1, dash: 'dot'}}}}],
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        }}, {{responsive: true}});

        // Inflow/Outflow Chart
        Plotly.newPlot('inflow-outflow-{entity}', [
            {{
                x: {json.dumps(history_dates)},
                y: {json.dumps(history_inflow)},
                type: 'bar',
                name: '↑ Inflow',
                marker: {{color: '{AZ_COLORS["lime_green"]}', line: {{color: 'rgba(0,0,0,0.2)', width: 1}}}},
                hovertemplate: '<b>Inflow</b><br>Date: %{{x}}<br>Amount: $%{{y:,.0f}}<extra></extra>'
            }},
            {{
                x: {json.dumps(history_dates)},
                y: {json.dumps([-x for x in history_outflow])},
                type: 'bar',
                name: '↓ Outflow',
                marker: {{color: '{AZ_COLORS["magenta"]}', line: {{color: 'rgba(0,0,0,0.2)', width: 1}}}},
                hovertemplate: '<b>Outflow</b><br>Date: %{{x}}<br>Amount: $%{{y:,.0f}}<extra></extra>'
            }},
            {{
                x: {json.dumps(history_dates)},
                y: {json.dumps(history_net)},
                type: 'scatter',
                mode: 'lines+markers',
                name: '— Net Cash Flow',
                line: {{color: '{AZ_COLORS["navy"]}', width: 3}},
                marker: {{size: 6}},
                hovertemplate: '<b>Net</b><br>Date: %{{x}}<br>Amount: $%{{y:,.0f}}<extra></extra>'
            }}
        ], {{
            height: 400,
            barmode: 'relative',
            margin: {{t: 50, b: 60, l: 80, r: 40}},
            xaxis: {{
                title: {{text: 'Week', font: {{size: 12, color: '#666'}}}},
                tickangle: -45,
                tickfont: {{size: 10}},
                gridcolor: '#eee'
            }},
            yaxis: {{
                title: {{text: 'Amount (USD)', font: {{size: 12, color: '#666'}}}},
                tickformat: ',.0f',
                tickfont: {{size: 10}},
                gridcolor: '#eee',
                zerolinecolor: '#333',
                zerolinewidth: 2
            }},
            legend: {{
                x: 0.5,
                y: 1.12,
                xanchor: 'center',
                orientation: 'h',
                bgcolor: 'rgba(255,255,255,0.95)',
                bordercolor: '#ddd',
                borderwidth: 1,
                font: {{size: 11}}
            }},
            hovermode: 'x unified',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        }}, {{responsive: true}});

        // Short-term Forecast Chart
        var lastHistDate = '{history_dates[-1]}';
        var lastHistVal = {history_net[-1]};
        Plotly.newPlot('short-forecast-{entity}', [
            {{
                x: {json.dumps(history_dates[-8:])},
                y: {json.dumps(history_net[-8:])},
                type: 'scatter',
                mode: 'lines+markers',
                name: '📊 Historical (Last 8 Weeks)',
                line: {{color: '{AZ_COLORS["navy"]}', width: 3}},
                marker: {{size: 10}},
                hovertemplate: '<b>Historical</b><br>Date: %{{x}}<br>Net: $%{{y:,.0f}}<extra></extra>'
            }},
            {{
                x: [lastHistDate].concat({json.dumps(short_dates)}),
                y: [lastHistVal].concat({json.dumps(short_pred)}),
                type: 'scatter',
                mode: 'lines+markers',
                name: '🔮 1-Month Forecast',
                line: {{color: '{AZ_COLORS["light_blue"]}', width: 4}},
                marker: {{size: 12, symbol: 'diamond'}},
                fill: 'tozeroy',
                fillcolor: 'rgba(104, 210, 223, 0.2)',
                hovertemplate: '<b>Forecast</b><br>Date: %{{x}}<br>Predicted: $%{{y:,.0f}}<extra></extra>'
            }}
        ], {{
            height: 320,
            margin: {{t: 50, b: 60, l: 80, r: 40}},
            xaxis: {{
                title: {{text: 'Week', font: {{size: 12, color: '#666'}}}},
                tickangle: -30,
                tickfont: {{size: 10}},
                gridcolor: '#eee'
            }},
            yaxis: {{
                title: {{text: 'Net Cash Flow (USD)', font: {{size: 12, color: '#666'}}}},
                tickformat: ',.0f',
                tickfont: {{size: 10}},
                gridcolor: '#eee',
                zerolinecolor: '#333',
                zerolinewidth: 2
            }},
            legend: {{
                x: 0.5,
                y: 1.12,
                xanchor: 'center',
                orientation: 'h',
                bgcolor: 'rgba(255,255,255,0.95)',
                bordercolor: '#ddd',
                borderwidth: 1,
                font: {{size: 10}}
            }},
            hovermode: 'x unified',
            shapes: [{{type: 'line', x0: lastHistDate, x1: lastHistDate, y0: 0, y1: 0, yref: 'paper', line: {{color: '#888', width: 2, dash: 'dash'}}}}],
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        }}, {{responsive: true}});

        // Long-term Forecast Chart (Month-by-Month with different colors)
        var longForecastTraces = [
            {{
                x: {json.dumps(history_dates[-12:])},
                y: {json.dumps(history_net[-12:])},
                type: 'scatter',
                mode: 'lines+markers',
                name: '📊 Historical',
                line: {{color: '{AZ_COLORS["navy"]}', width: 3}},
                marker: {{size: 8}},
                hovertemplate: '<b>Historical</b><br>Date: %{{x}}<br>Net: $%{{y:,.0f}}<extra></extra>'
            }}
        ];
        
        // Add each month as a separate trace with different colors
        var monthColors = ['{AZ_COLORS["light_blue"]}', '{AZ_COLORS["lime_green"]}', '{AZ_COLORS["gold"]}', '{AZ_COLORS["magenta"]}', '{AZ_COLORS["purple"]}', '{AZ_COLORS["mulberry"]}'];
        var monthNames = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'];
        var monthlyData = {json.dumps(monthly_data)};
        var prevEndDate = lastHistDate;
        var prevEndVal = lastHistVal;
        
        monthlyData.forEach(function(month, idx) {{
            var xVals = [prevEndDate].concat(month.dates);
            var yVals = [prevEndVal].concat(month.predictions);
            var monthTotal = month.total;
            var sign = monthTotal >= 0 ? '+' : '';
            longForecastTraces.push({{
                x: xVals,
                y: yVals,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'M' + month.month_num + ' (' + sign + (monthTotal/1000).toFixed(0) + 'K)',
                line: {{color: monthColors[idx], width: 3}},
                marker: {{size: 8, symbol: 'circle'}},
                fill: 'tozeroy',
                fillcolor: monthColors[idx] + '15',
                hovertemplate: '<b>Month ' + month.month_num + '</b><br>Date: %{{x}}<br>Net: $%{{y:,.0f}}<extra></extra>'
            }});
            prevEndDate = month.dates[month.dates.length - 1];
            prevEndVal = month.predictions[month.predictions.length - 1];
        }});
        
        Plotly.newPlot('long-forecast-{entity}', longForecastTraces, {{
            height: 380,
            margin: {{t: 60, b: 60, l: 80, r: 40}},
            xaxis: {{
                title: {{text: 'Week', font: {{size: 12, color: '#666'}}}},
                tickangle: -30,
                tickfont: {{size: 10}},
                gridcolor: '#eee'
            }},
            yaxis: {{
                title: {{text: 'Net Cash Flow (USD)', font: {{size: 12, color: '#666'}}}},
                tickformat: ',.0f',
                tickfont: {{size: 10}},
                gridcolor: '#eee',
                zerolinecolor: '#333',
                zerolinewidth: 2
            }},
            legend: {{
                x: 0.5,
                y: 1.15,
                xanchor: 'center',
                orientation: 'h',
                bgcolor: 'rgba(255,255,255,0.95)',
                bordercolor: '#ccc',
                borderwidth: 1,
                font: {{size: 10}},
                itemwidth: 40
            }},
            hovermode: 'x unified',
            shapes: [{{
                type: 'line',
                x0: lastHistDate,
                x1: lastHistDate,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {{color: '#888', width: 2, dash: 'dash'}}
            }}],
            annotations: [{{
                x: lastHistDate,
                y: 1.02,
                yref: 'paper',
                text: 'Forecast →',
                showarrow: false,
                font: {{size: 10, color: '#666'}}
            }}],
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        }}, {{responsive: true}});

        // Monthly Summary Bar Chart
        var monthlyTotals = monthlyData.map(m => m.total);
        var monthlyLabels = monthlyData.map(m => 'Month ' + m.month_num);
        var monthlyBarColors = monthlyTotals.map((v, i) => v >= 0 ? monthColors[i] : '{AZ_COLORS["magenta"]}');
        
        // Calculate y-axis range to accommodate text labels
        var maxVal = Math.max(...monthlyTotals);
        var minVal = Math.min(...monthlyTotals);
        var yPadding = Math.max(Math.abs(maxVal), Math.abs(minVal)) * 0.25;
        var yMin = minVal < 0 ? minVal - yPadding : Math.min(0, minVal);
        var yMax = maxVal > 0 ? maxVal + yPadding : Math.max(0, maxVal);
        
        Plotly.newPlot('monthly-chart-{entity}', [
            {{
                x: monthlyLabels,
                y: monthlyTotals,
                type: 'bar',
                name: 'Monthly Net',
                marker: {{
                    color: monthlyBarColors,
                    line: {{color: 'rgba(0,0,0,0.3)', width: 1}}
                }},
                text: monthlyTotals.map(v => v >= 0 ? '+' + (v/1000).toFixed(0) + 'K' : (v/1000).toFixed(0) + 'K'),
                textposition: 'outside',
                textfont: {{size: 11, color: '#333', family: 'Inter, sans-serif', weight: 600}},
                cliponaxis: false,
                constraintext: 'none',
                hovertemplate: '<b>%{{x}}</b><br>Net: $%{{y:,.0f}}<extra></extra>'
            }},
            {{
                x: monthlyLabels,
                y: monthlyData.map(m => m.cumulative),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Cumulative Total',
                yaxis: 'y2',
                line: {{color: '{AZ_COLORS["navy"]}', width: 3}},
                marker: {{size: 10, symbol: 'circle'}},
                hovertemplate: '<b>%{{x}}</b><br>Cumulative: $%{{y:,.0f}}<extra></extra>'
            }}
        ], {{
            height: 380,
            margin: {{t: 80, b: 60, l: 80, r: 80}},
            xaxis: {{
                title: {{text: 'Forecast Period', font: {{size: 12, color: '#666'}}}},
                tickfont: {{size: 11}}
            }},
            yaxis: {{
                title: {{text: 'Monthly Net (USD)', font: {{size: 12, color: '{AZ_COLORS["mulberry"]}'}}}},
                tickformat: ',.0f',
                tickfont: {{size: 10}},
                gridcolor: '#eee',
                zerolinecolor: '#333',
                zerolinewidth: 2,
                range: [yMin, yMax]
            }},
            yaxis2: {{
                title: {{text: 'Cumulative (USD)', font: {{size: 12, color: '{AZ_COLORS["navy"]}'}}}},
                overlaying: 'y',
                side: 'right',
                tickformat: ',.0f',
                tickfont: {{size: 10}}
            }},
            showlegend: true,
            legend: {{
                x: 0.5,
                y: 1.12,
                xanchor: 'center',
                orientation: 'h',
                bgcolor: 'rgba(255,255,255,0.95)',
                bordercolor: '#ddd',
                borderwidth: 1,
                font: {{size: 11}}
            }},
            hovermode: 'x unified',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        }}, {{responsive: true}});
        </script>
        '''

    def build(
        self,
        entity_frames: Dict[str, pd.DataFrame],
        forecast_results: Dict[str, ForecastArtifacts],
        weekly_df: pd.DataFrame,
    ) -> Path:
        """Build complete interactive dashboard with tabbed entity selection."""
        import json
        
        # Aggregate metrics
        total_entities = len(forecast_results)
        total_inflow = weekly_df["Total_Inflow"].sum()
        total_outflow = abs(weekly_df["Total_Outflow"].sum())
        total_net = weekly_df["Total_Net"].sum()

        # Entity to country mapping
        entity_countries = {
            "TW10": {"country": "Taiwan", "search_term": "Taiwan economy"},
            "PH10": {"country": "Philippines", "search_term": "Philippines economy"},
            "TH10": {"country": "Thailand", "search_term": "Thailand economy"},
            "ID10": {"country": "Indonesia", "search_term": "Indonesia economy"},
            "SS10": {"country": "Singapore", "search_term": "Singapore economy"},
            "MY10": {"country": "Malaysia", "search_term": "Malaysia economy"},
            "VN20": {"country": "Vietnam", "search_term": "Vietnam economy"},
            "KR10": {"country": "South Korea", "search_term": "South Korea economy"},
        }
        
        # Try to scrape fresh news sentiment, fall back to cached data
        print("  📡 Attempting to scrape fresh news sentiment...")
        country_sentiments = self._scrape_country_sentiments(entity_countries)
        
        # If scraping failed or returned empty, try loading cached data
        if not country_sentiments:
            print("  ℹ️ Scraping failed, loading cached sentiment data...")
            country_sentiments = self._load_country_sentiments(entity_countries)
        
        # Build entity analysis with priority levels
        entity_priority_data = []
        for entity in forecast_results.keys():
            sentiment_data = country_sentiments.get(entity, {})
            sentiment_score = sentiment_data.get('average_score', 0)
            country = entity_countries.get(entity, {}).get('country', 'Unknown')
            
            # Priority based on sentiment score (more negative = higher priority)
            if sentiment_score <= -0.3:
                priority = "HIGH"
                priority_color = "#e74c3c"
                priority_icon = "🔴"
            elif sentiment_score <= -0.1:
                priority = "MEDIUM"
                priority_color = "#f39c12"
                priority_icon = "🟡"
            elif sentiment_score >= 0.3:
                priority = "LOW"
                priority_color = "#27ae60"
                priority_icon = "🟢"
            else:
                priority = "MEDIUM"
                priority_color = "#f39c12"
                priority_icon = "🟡"
            
            entity_priority_data.append({
                "entity": entity,
                "country": country,
                "priority": priority,
                "priority_color": priority_color,
                "priority_icon": priority_icon,
                "sentiment_score": sentiment_score,
                "articles": sentiment_data.get('articles', [])
            })
        
        # Sort by priority (HIGH first, then MEDIUM, then LOW)
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        entity_priority_data.sort(key=lambda x: (priority_order[x["priority"]], x["sentiment_score"]))
        
        # Generate entity tabs HTML
        entity_tabs_html = ""
        for i, ep in enumerate(entity_priority_data):
            active_class = "active" if i == 0 else ""
            entity_tabs_html += f'''
            <button class="entity-tab {active_class}" data-entity="{ep['entity']}" onclick="switchEntity('{ep['entity']}')"
                    style="--priority-color: {ep['priority_color']}">
                <span class="tab-priority-icon">{ep['priority_icon']}</span>
                <span class="tab-entity-name">{ep['entity']}</span>
                <span class="tab-country">{ep['country']}</span>
                <span class="tab-priority-badge" style="background: {ep['priority_color']}20; color: {ep['priority_color']}">{ep['priority']}</span>
            </button>
            '''
        
        # Generate entity sections (all hidden except first)
        entity_sections = ""
        for i, ep in enumerate(entity_priority_data):
            entity = ep['entity']
            display_style = "block" if i == 0 else "none"
            news_html = self._generate_entity_news_section(ep['entity'], ep['country'], ep['articles'], ep['sentiment_score'])
            entity_content = self._generate_entity_html(entity, entity_frames[entity], forecast_results[entity])
            entity_sections += f'''
            <div class="entity-content-wrapper" id="content-{entity}" style="display: {display_style};">
                {news_html}
                {entity_content}
            </div>
            '''
        
        # JSON data for JavaScript
        entity_data_json = json.dumps({ep['entity']: ep for ep in entity_priority_data})

        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cash Flow Forecasting Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --az-mulberry: {AZ_COLORS["mulberry"]};
            --az-lime: {AZ_COLORS["lime_green"]};
            --az-navy: {AZ_COLORS["navy"]};
            --az-graphite: {AZ_COLORS["graphite"]};
            --az-light-blue: {AZ_COLORS["light_blue"]};
            --az-magenta: {AZ_COLORS["magenta"]};
            --az-purple: {AZ_COLORS["purple"]};
            --az-gold: {AZ_COLORS["gold"]};
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            color: var(--az-graphite);
            line-height: 1.6;
        }}

        .dashboard-header {{
            background: linear-gradient(135deg, var(--az-mulberry) 0%, var(--az-purple) 100%);
            color: white;
            padding: 2rem 3rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}

        .header-content {{
            max-width: 1600px;
            margin: 0 auto;
        }}

        .dashboard-title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .dashboard-subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }}

        .nav-bar {{
            background: white;
            padding: 1rem 3rem;
            border-bottom: 3px solid var(--az-mulberry);
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .nav-content {{
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
        }}

        .nav-label {{
            font-weight: 600;
            color: var(--az-mulberry);
            margin-right: 1rem;
        }}

        .nav-link {{
            padding: 0.5rem 1.2rem;
            background: #f1f3f4;
            border-radius: 25px;
            text-decoration: none;
            color: var(--az-graphite);
            font-weight: 500;
            transition: all 0.3s ease;
        }}

        .nav-link:hover {{
            background: var(--az-mulberry);
            color: white;
            transform: translateY(-2px);
        }}

        /* Entity Tabs Styling */
        .entity-tabs-container {{
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .entity-tabs-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }}

        .entity-tabs-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--az-mulberry);
        }}

        .priority-legend {{
            display: flex;
            gap: 1rem;
            font-size: 0.85rem;
            margin-left: auto;
        }}

        .priority-legend-item {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .entity-tabs {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            padding: 0.5rem 0;
        }}

        .entity-tab {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 1rem 1.25rem;
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 140px;
            position: relative;
            overflow: hidden;
        }}

        .entity-tab::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: var(--priority-color);
            transition: width 0.3s ease;
        }}

        .entity-tab:hover {{
            border-color: var(--priority-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .entity-tab.active {{
            background: white;
            border-color: var(--priority-color);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }}

        .entity-tab.active::before {{
            width: 6px;
        }}

        .tab-priority-icon {{
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }}

        .tab-entity-name {{
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--az-navy);
        }}

        .tab-country {{
            font-size: 0.8rem;
            color: #666;
            margin: 0.25rem 0;
        }}

        .tab-priority-badge {{
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            border-radius: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }}

        /* News Section Styling */
        .entity-news-section {{
            background: linear-gradient(145deg, #f8f9fa, #ffffff);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .news-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .news-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--az-navy);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .sentiment-badge {{
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}

        .news-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}

        .news-card {{
            background: white;
            border-radius: 10px;
            padding: 1rem;
            border-left: 4px solid;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .news-card:hover {{
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .news-card-title {{
            font-size: 0.95rem;
            font-weight: 500;
            color: #333;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }}

        .news-card-title a {{
            text-decoration: none;
            color: inherit;
        }}

        .news-card-title a:hover {{
            color: var(--az-mulberry);
        }}

        .news-card-meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: #666;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .news-sentiment-tag {{
            padding: 0.15rem 0.5rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.75rem;
        }}

        /* News Fetch Control Panel */
        .news-fetch-control {{
            background: linear-gradient(145deg, #f0f4f8, #ffffff);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e1e8ed;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .news-fetch-left {{
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }}

        .news-fetch-label {{
            font-weight: 600;
            color: var(--az-navy);
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .news-fetch-input {{
            width: 80px;
            padding: 0.5rem 0.75rem;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            font-size: 1rem;
            text-align: center;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}

        .news-fetch-input:focus {{
            outline: none;
            border-color: var(--az-mulberry);
            box-shadow: 0 0 0 3px rgba(131, 0, 81, 0.1);
        }}

        .news-fetch-btn {{
            background: linear-gradient(135deg, var(--az-mulberry), var(--az-purple));
            color: white;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .news-fetch-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(131, 0, 81, 0.3);
        }}

        .news-fetch-btn:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }}

        .news-fetch-status {{
            font-size: 0.85rem;
            color: #666;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .news-fetch-status.loading {{
            color: var(--az-mulberry);
        }}

        .news-fetch-status.success {{
            color: #27ae60;
        }}

        .news-fetch-status.error {{
            color: #e74c3c;
        }}

        .spinner {{
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid var(--az-mulberry);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        .main-content {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }}

        /* Executive Summary */
        .executive-summary {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .summary-title {{
            font-size: 2rem;
            color: var(--az-mulberry);
            margin-bottom: 1rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
        }}

        .summary-card {{
            text-align: center;
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(145deg, #f8f9fa, #ffffff);
            border: 1px solid #e9ecef;
        }}

        .summary-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}

        .summary-label {{
            font-size: 0.9rem;
            color: #6c757d;
            font-weight: 500;
        }}

        /* Methodology Section */
        .methodology {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .methodology h2 {{
            color: var(--az-mulberry);
            margin-bottom: 1rem;
        }}

        .methodology-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}

        .method-card {{
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid var(--az-mulberry);
            background: #f8f9fa;
        }}

        .method-card h4 {{
            color: var(--az-navy);
            margin-bottom: 0.5rem;
        }}

        .method-card p {{
            font-size: 0.95rem;
            color: #555;
        }}

        /* Entity Section */
        .entity-section {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .entity-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .entity-header h2 {{
            font-size: 1.75rem;
            color: var(--az-graphite);
        }}

        .entity-badge {{
            background: var(--az-mulberry);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 8px;
            font-size: 1.2rem;
        }}

        .model-badge {{
            background: var(--az-light-blue);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
        }}

        /* KPI Cards */
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .kpi-card {{
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}

        .kpi-icon {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}

        .kpi-value {{
            font-size: 1.5rem;
            font-weight: 700;
        }}

        .kpi-label {{
            font-size: 0.85rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }}

        /* Charts */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .chart-container {{
            background: #fafbfc;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #e9ecef;
        }}

        .chart-container.full-width {{
            grid-column: 1 / -1;
        }}

        .chart-container h3 {{
            font-size: 1.1rem;
            color: var(--az-navy);
            margin-bottom: 0.5rem;
        }}

        .chart-description {{
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 1rem;
        }}

        .chart {{
            width: 100%;
            min-height: 300px;
        }}

        .metrics-row {{
            display: flex;
            gap: 1.5rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e9ecef;
            flex-wrap: wrap;
        }}

        .metric {{
            display: flex;
            flex-direction: column;
        }}

        .metric-label {{
            font-size: 0.8rem;
            color: #6c757d;
            text-transform: uppercase;
        }}

        .metric-value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--az-navy);
        }}

        /* Insight Box */
        .insight-box {{
            background: #f8f9fa;
            border-left: 4px solid var(--az-mulberry);
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin-top: 1rem;
        }}

        .insight-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}

        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            color: white;
            font-weight: 600;
        }}

        /* Category Analysis */
        .category-analysis {{
            margin-bottom: 2rem;
        }}

        .category-analysis h3 {{
            color: var(--az-navy);
            margin-bottom: 1rem;
        }}

        .category-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}

        .category-box {{
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e9ecef;
        }}

        .category-box.inflow {{
            background: linear-gradient(145deg, rgba(196, 214, 0, 0.1), rgba(196, 214, 0, 0.05));
            border-left: 4px solid var(--az-lime);
        }}

        .category-box.outflow {{
            background: linear-gradient(145deg, rgba(208, 0, 111, 0.1), rgba(208, 0, 111, 0.05));
            border-left: 4px solid var(--az-magenta);
        }}

        .category-box h4 {{
            margin-bottom: 1rem;
            color: var(--az-graphite);
        }}

        .category-box ul {{
            list-style: none;
        }}

        .category-box li {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px dashed #e9ecef;
        }}

        .cat-name {{
            font-weight: 500;
        }}

        .cat-value {{
            font-weight: 600;
        }}

        /* Monthly Breakdown */
        .monthly-breakdown {{
            background: #fafbfc;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid #e9ecef;
        }}

        .monthly-breakdown h3 {{
            color: var(--az-navy);
            margin-bottom: 0.5rem;
        }}

        .monthly-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}

        .monthly-card {{
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .monthly-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }}

        .monthly-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .month-badge {{
            padding: 0.25rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            color: white;
            font-weight: 600;
        }}

        .month-dates {{
            font-size: 0.7rem;
            color: #6c757d;
        }}

        .monthly-stats {{
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}

        .monthly-stat {{
            text-align: center;
            flex: 1;
        }}

        .stat-value {{
            display: block;
            font-size: 1rem;
            font-weight: 700;
            color: var(--az-navy);
        }}

        .stat-label {{
            font-size: 0.65rem;
            color: #6c757d;
            text-transform: uppercase;
        }}

        .weekly-detail {{
            display: flex;
            gap: 0.25rem;
            flex-wrap: wrap;
            justify-content: center;
        }}

        .week-val {{
            font-size: 0.65rem;
            padding: 0.15rem 0.4rem;
            background: #f1f3f4;
            border-radius: 4px;
            font-weight: 500;
        }}

        /* Model Table */
        .model-comparison {{
            margin-bottom: 1rem;
        }}

        .model-comparison h3 {{
            color: var(--az-navy);
            margin-bottom: 1rem;
        }}

        .model-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }}

        .model-table th {{
            background: var(--az-navy);
            color: white;
            padding: 0.75rem;
            text-align: left;
        }}

        .model-table td {{
            padding: 0.75rem;
            border-bottom: 1px solid #e9ecef;
        }}

        .model-table tr:hover {{
            background: #f8f9fa;
        }}

        .model-table .best-model {{
            background: rgba(196, 214, 0, 0.2);
            font-weight: 600;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            .dashboard-header {{
                padding: 1.5rem;
            }}
            .dashboard-title {{
                font-size: 1.75rem;
            }}
        }}

        /* Print styles */
        @media print {{
            .nav-bar {{ display: none; }}
            .entity-section {{ break-inside: avoid; }}
        }}

        /* Main Page Tabs */
        .main-tabs {{
            display: flex;
            gap: 0.5rem;
            margin-left: auto;
        }}

        .main-tab {{
            padding: 0.6rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .main-tab.overview-tab {{
            background: #e9ecef;
            color: var(--az-navy);
        }}

        .main-tab.overview-tab:hover {{
            background: #dee2e6;
        }}

        .main-tab.overview-tab.active {{
            background: linear-gradient(135deg, var(--az-navy) 0%, #1a3a52 100%);
            color: white;
            box-shadow: 0 4px 12px rgba(0, 59, 92, 0.3);
        }}

        .main-tab.entity-tab-main {{
            background: #e9ecef;
            color: var(--az-magenta);
        }}

        .main-tab.entity-tab-main:hover {{
            background: #dee2e6;
        }}

        .main-tab.entity-tab-main.active {{
            background: linear-gradient(135deg, #830051 0%, #3C1053 100%);
            color: white;
            box-shadow: 0 4px 12px rgba(131, 0, 81, 0.3);
        }}

        .page-section {{
            display: none;
        }}

        .page-section.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <header class="dashboard-header">
        <div class="header-content">
            <h1 class="dashboard-title">💰 Cash Flow Forecasting Dashboard</h1>
            <p class="dashboard-subtitle">Time-Series Analysis & ML-Based Predictions | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
    </header>

    <nav class="nav-bar">
        <div class="nav-content">
            <span class="nav-label">📊 Dashboard</span>
            <div class="main-tabs">
                <button class="main-tab overview-tab active" onclick="switchMainTab('overview')">
                    📋 Overview
                </button>
                <button class="main-tab entity-tab-main" onclick="switchMainTab('entity-analysis')">
                    🏢 Entity Analysis
                </button>
            </div>
        </div>
    </nav>

    <main class="main-content">
        <!-- OVERVIEW PAGE SECTION -->
        <div id="overview-page" class="page-section active">
            <!-- Executive Summary -->
            <section class="executive-summary" id="executive-summary">
                <h2 class="summary-title">📊 Executive Summary</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-value" style="color: var(--az-navy)">{total_entities}</div>
                        <div class="summary-label">Entities Analyzed</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value" style="color: var(--az-lime)">{self._format_currency(total_inflow)}</div>
                        <div class="summary-label">Total Inflows</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value" style="color: var(--az-magenta)">{self._format_currency(total_outflow)}</div>
                        <div class="summary-label">Total Outflows</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value" style="color: var(--az-mulberry)">{self._format_currency(total_net)}</div>
                        <div class="summary-label">Net Position</div>
                    </div>
                </div>
            </section>

            <!-- Methodology Section -->
            <section class="methodology" id="methodology">
                <h2>📚 Forecasting Methodology</h2>
                <div class="methodology-grid">
                    <div class="method-card">
                        <h4>🎯 Model Selection</h4>
                        <p>We employ an ensemble of ML models including <strong>XGBoost</strong>, <strong>LightGBM</strong>, <strong>Random Forest</strong>, <strong>Gradient Boosting</strong>, and <strong>Ridge Regression</strong>. The best model is selected based on lowest RMSE during backtesting.</p>
                    </div>
                    <div class="method-card">
                        <h4>📈 Key Features</h4>
                        <p>Models use <strong>lag features</strong> (1, 2, 4 weeks), <strong>rolling statistics</strong> (4-week mean/std), <strong>temporal patterns</strong> (week of month, quarter), and <strong>transaction counts</strong> to capture cash flow dynamics.</p>
                    </div>
                    <div class="method-card">
                        <h4>🔍 Backtest Validation</h4>
                        <p>The last <strong>4 weeks</strong> are held out for backtesting. We measure accuracy using MAE, RMSE, MAPE, and direction accuracy to ensure reliable forecasts.</p>
                    </div>
                    <div class="method-card">
                        <h4>⚠️ Limitations</h4>
                        <p>Forecasts assume historical patterns continue. External shocks, policy changes, or market disruptions may cause deviations. Longer horizons have higher uncertainty.</p>
                    </div>
                </div>
            </section>
        </div>

        <!-- ENTITY ANALYSIS PAGE SECTION -->
        <div id="entity-analysis-page" class="page-section">
            <!-- News Fetch Control Panel -->
            <div class="news-fetch-control">
                <div class="news-fetch-left">
                    <span class="news-fetch-label">📰 Market News</span>
                    <label style="display: flex; align-items: center; gap: 0.5rem; color: #555;">
                        Max articles per country:
                        <input type="number" id="maxArticlesInput" class="news-fetch-input" value="5" min="1" max="20">
                    </label>
                    <button type="button" id="fetchNewsBtn" class="news-fetch-btn" onclick="fetchLatestNews(event)">
                        🔄 Fetch Current News
                    </button>
                </div>
                <div id="newsFetchStatus" class="news-fetch-status">
                    Last updated: Dashboard generation time
                </div>
            </div>

            <!-- Entity Selection Tabs -->
            <section class="entity-tabs-container" id="entity-analysis">
                <div class="entity-tabs-header">
                    <h2 class="entity-tabs-title">🏢 Select Entity to View Analysis</h2>
                    <div class="priority-legend">
                        <span class="priority-legend-item">🔴 HIGH Priority</span>
                        <span class="priority-legend-item">🟡 MEDIUM Priority</span>
                        <span class="priority-legend-item">🟢 LOW Priority</span>
                    </div>
                </div>
                <p style="color: #666; margin-bottom: 1rem; font-size: 0.9rem;">
                    Priority is based on news sentiment analysis. Click on an entity tab to view its detailed cash flow analysis and related news.
                </p>
                <div class="entity-tabs">
                    {entity_tabs_html}
                </div>
            </section>

            <!-- Entity Content Sections (dynamically shown/hidden) -->
            <div id="entity-content-area">
                {entity_sections}
            </div>
        </div>

    </main>

    <footer style="text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;">
        <p>🏢 Global Finance Services & Group Controllers | Cash Flow Analytics Platform</p>
        <p style="margin-top: 0.5rem;">Dashboard generated using Python ML Pipeline with AstraZeneca Theme</p>
    </footer>

    <script>
        // Main page tab switching functionality
        function switchMainTab(tabId) {{
            // Update main tab active states
            document.querySelectorAll('.main-tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            if (tabId === 'overview') {{
                document.querySelector('.overview-tab').classList.add('active');
            }} else {{
                document.querySelector('.entity-tab-main').classList.add('active');
            }}
            
            // Show/hide page sections
            document.querySelectorAll('.page-section').forEach(section => {{
                section.classList.remove('active');
            }});
            
            const targetPage = document.getElementById(tabId + '-page');
            if (targetPage) {{
                targetPage.classList.add('active');
                // Scroll to top of page
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
                
                // Trigger Plotly resize for any charts
                setTimeout(() => {{
                    window.dispatchEvent(new Event('resize'));
                }}, 100);
            }}
        }}

        // Entity tab switching functionality
        function switchEntity(entityId) {{
            // Update tab active states
            document.querySelectorAll('.entity-tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            const activeTab = document.querySelector(`.entity-tab[data-entity="${{entityId}}"]`);
            if (activeTab) {{
                activeTab.classList.add('active');
            }}
            
            // Show/hide entity content
            document.querySelectorAll('.entity-content-wrapper').forEach(content => {{
                content.style.display = 'none';
            }});
            const activeContent = document.getElementById(`content-${{entityId}}`);
            if (activeContent) {{
                activeContent.style.display = 'block';
                // Scroll to the entity content
                activeContent.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                
                // Trigger Plotly resize for any charts that may need it
                setTimeout(() => {{
                    window.dispatchEvent(new Event('resize'));
                }}, 100);
            }}
        }}
        
        // Initialize - ensure charts resize properly
        window.addEventListener('load', function() {{
            setTimeout(() => {{
                window.dispatchEvent(new Event('resize'));
            }}, 500);
        }});

        // News Fetch Functionality
        const NEWS_SERVER_URL = 'http://localhost:5001';

        // Entity to country mapping
        const ENTITY_COUNTRY_MAP = {{
            'TH10': 'Thailand',
            'TW10': 'Taiwan',
            'SS10': 'Singapore',
            'MY10': 'Malaysia',
            'VN20': 'Vietnam',
            'KR10': 'South Korea',
            'ID10': 'Indonesia',
            'PH10': 'Philippines'
        }};

        async function fetchLatestNews(event) {{
            // Prevent any default behavior or event bubbling
            if (event) {{
                event.preventDefault();
                event.stopPropagation();
            }}
            
            const fetchBtn = document.getElementById('fetchNewsBtn');
            const statusDiv = document.getElementById('newsFetchStatus');
            const maxArticles = parseInt(document.getElementById('maxArticlesInput').value) || 5;

            // Validate input
            if (maxArticles < 1 || maxArticles > 20) {{
                statusDiv.className = 'news-fetch-status error';
                statusDiv.innerHTML = '❌ Please enter a number between 1 and 20';
                return;
            }}

            // Update UI to loading state
            fetchBtn.disabled = true;
            fetchBtn.innerHTML = '<span class="spinner"></span> Fetching...';
            statusDiv.className = 'news-fetch-status loading';
            statusDiv.innerHTML = '<span class="spinner"></span> Fetching news from multiple sources...';

            try {{
                const response = await fetch(`${{NEWS_SERVER_URL}}/api/fetch-news?max_articles=${{maxArticles}}`);
                
                if (!response.ok) {{
                    throw new Error(`Server returned ${{response.status}}`);
                }}

                const result = await response.json();

                if (result.success) {{
                    // Update the news sections in the DOM
                    updateNewsSections(result.data);
                    
                    const timestamp = new Date().toLocaleString();
                    statusDiv.className = 'news-fetch-status success';
                    statusDiv.innerHTML = `✅ Updated ${{Object.keys(result.data).length}} countries at ${{timestamp}}`;
                }} else {{
                    throw new Error(result.error || 'Unknown error');
                }}

            }} catch (error) {{
                console.error('News fetch error:', error);
                statusDiv.className = 'news-fetch-status error';
                
                if (error.message.includes('Failed to fetch')) {{
                    statusDiv.innerHTML = '❌ Cannot connect to news server. Please start the server with: <code>python news_server.py</code>';
                }} else {{
                    statusDiv.innerHTML = `❌ Error: ${{error.message}}`;
                }}
            }} finally {{
                fetchBtn.disabled = false;
                fetchBtn.innerHTML = '🔄 Fetch Current News';
            }}
        }}

        function getPriorityFromScore(score) {{
            // Priority based on sentiment score (more negative = higher priority)
            if (score <= -0.3) {{
                return {{ priority: 'HIGH', color: '#e74c3c', icon: '🔴' }};
            }} else if (score <= -0.1) {{
                return {{ priority: 'MEDIUM', color: '#f39c12', icon: '🟡' }};
            }} else if (score >= 0.3) {{
                return {{ priority: 'LOW', color: '#27ae60', icon: '🟢' }};
            }} else {{
                return {{ priority: 'MEDIUM', color: '#f39c12', icon: '🟡' }};
            }}
        }}

        function updateEntityTabPriority(entity, score) {{
            // Find the entity tab button
            const entityTab = document.querySelector(`.entity-tab[data-entity="${{entity}}"]`);
            if (!entityTab) return;

            const priorityInfo = getPriorityFromScore(score);

            // Update the priority icon
            const priorityIcon = entityTab.querySelector('.tab-priority-icon');
            if (priorityIcon) {{
                priorityIcon.textContent = priorityInfo.icon;
            }}

            // Update the priority badge
            const priorityBadge = entityTab.querySelector('.tab-priority-badge');
            if (priorityBadge) {{
                priorityBadge.textContent = priorityInfo.priority;
                priorityBadge.style.background = priorityInfo.color + '20';
                priorityBadge.style.color = priorityInfo.color;
            }}

            // Update the tab's priority color CSS variable
            entityTab.style.setProperty('--priority-color', priorityInfo.color);
        }}

        function updateNewsSections(newsData) {{
            for (const [entity, data] of Object.entries(newsData)) {{
                const country = ENTITY_COUNTRY_MAP[entity];
                if (!country) continue;

                // Update the entity tab priority based on new sentiment score
                const score = data.average_score || 0;
                updateEntityTabPriority(entity, score);

                // Find the news section for this entity
                const contentWrapper = document.getElementById(`content-${{entity}}`);
                if (!contentWrapper) continue;

                const newsSection = contentWrapper.querySelector('.entity-news-section');
                if (!newsSection) continue;

                // Generate new news HTML
                const newNewsHtml = generateNewsHtml(country, data);
                newsSection.innerHTML = newNewsHtml;
            }}
        }}

        function generateNewsHtml(country, data) {{
            const score = data.average_score || 0;
            const articles = data.articles || [];

            // Determine sentiment styling
            let sentimentLabel, sentimentColor, sentimentBg;
            if (score > 0.1) {{
                sentimentLabel = 'POSITIVE';
                sentimentColor = '#27ae60';
                sentimentBg = '#27ae6020';
            }} else if (score < -0.1) {{
                sentimentLabel = 'NEGATIVE';
                sentimentColor = '#e74c3c';
                sentimentBg = '#e74c3c20';
            }} else {{
                sentimentLabel = 'NEUTRAL';
                sentimentColor = '#7f8c8d';
                sentimentBg = '#7f8c8d20';
            }}

            // Generate news cards - show all articles returned by API
            let newsCardsHtml = '';
            if (articles.length > 0) {{
                for (const article of articles) {{
                    const articleScore = article.score || article.final_score || 0;
                    let cardColor, tagText, tagBg;
                    
                    if (articleScore > 0.1) {{
                        cardColor = '#27ae60';
                        tagText = 'Positive';
                        tagBg = '#27ae6015';
                    }} else if (articleScore < -0.1) {{
                        cardColor = '#e74c3c';
                        tagText = 'Negative';
                        tagBg = '#e74c3c15';
                    }} else {{
                        cardColor = '#7f8c8d';
                        tagText = 'Neutral';
                        tagBg = '#7f8c8d15';
                    }}

                    const title = article.title || 'No title';
                    const url = article.url || '#';
                    const source = article.source || 'News';

                    newsCardsHtml += `
                    <div class="news-card" style="border-left-color: ${{cardColor}};">
                        <div class="news-card-title">
                            <a href="${{url}}" target="_blank">${{title.substring(0, 120)}}${{title.length > 120 ? '...' : ''}}</a>
                        </div>
                        <div class="news-card-meta">
                            <span>📰 ${{source}}</span>
                            <span class="news-sentiment-tag" style="background: ${{tagBg}}; color: ${{cardColor}};">
                                ${{tagText}} (${{articleScore.toFixed(2)}})
                            </span>
                        </div>
                    </div>
                    `;
                }}
            }} else {{
                newsCardsHtml = `
                <div style="text-align: center; padding: 2rem; color: #888;">
                    <p>📭 No recent news articles available for this country.</p>
                </div>
                `;
            }}

            return `
            <div class="news-header">
                <div class="news-title">
                    📰 ${{country}} Market News
                </div>
                <div class="sentiment-badge" style="background: ${{sentimentBg}}; color: ${{sentimentColor}};">
                    Sentiment: ${{sentimentLabel}} (${{score.toFixed(2)}})
                </div>
            </div>
            <div class="news-grid">
                ${{newsCardsHtml}}
            </div>
            `;
        }}
    </script>
</body>
</html>'''

        output_path = self.paths.dashboard_dir / "interactive_dashboard.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main() -> None:
    """
    Execute the complete cash flow forecasting pipeline.
    
    This function orchestrates all pipeline components in sequence:
    
    STEP 1: DATA CLEANING
    --------------------
    - Load raw transaction CSV from Data directory
    - Select required columns and standardize types
    - Convert Amount in USD to numeric (handling comma formatting)
    - Parse date columns to datetime objects
    - Output: processed_data/clean_transactions.csv
    
    STEP 2: WEEKLY AGGREGATION & FEATURE ENGINEERING
    ------------------------------------------------
    - Aggregate daily transactions into weekly periods
    - Calculate summary statistics (sum, mean, count) per week
    - Generate temporal features (Month, Quarter, Week_of_Month)
    - Create lag features (Net_Lag1, Net_Lag2, Net_Lag4)
    - Compute rolling statistics (4-week mean and std)
    - Split data by entity for per-entity modeling
    - Output: processed_data/weekly_entity_features.csv
    
    STEP 3: ML MODEL TRAINING & FORECASTING
    ---------------------------------------
    For each entity:
    - Train multiple ML models (XGBoost, LightGBM, RF, GB, Ridge)
    - Evaluate on held-out backtest period (last 4 weeks)
    - Select best model by lowest RMSE
    - Generate 1-month (4-week) forecast with seasonal adjustment
    - Generate 6-month (24-week) forecast with year-over-year patterns
    - Track monthly breakdown for strategic planning
    
    STEP 4: DASHBOARD GENERATION
    ---------------------------
    - Generate interactive HTML dashboard with Plotly.js
    - Include backtest validation charts per entity
    - Show 1-month and 6-month forecast visualizations
    - Apply AstraZeneca corporate color theme
    - Output: outputs/dashboards/interactive_dashboard.html
    
    The pipeline is idempotent - safe to run multiple times as it
    regenerates all outputs from the source data.
    
    Returns:
        None (outputs are written to files)
    """
    paths = PathConfig()
    print("=" * 70)
    print("  CASH FLOW FORECASTING PIPELINE")
    print("  Generating Interactive Dashboard with AstraZeneca Theme")
    print("=" * 70)

    print("\n=== Step 1: Cleaning raw data ===")
    cleaner = DataCleaner(paths)
    clean_df = cleaner.run()

    print("\n=== Step 2: Building weekly features ===")
    aggregator = WeeklyAggregator(paths)
    weekly_df, entity_frames = aggregator.run(clean_df)

    print("\n=== Step 3: Training ML models & generating forecasts ===")
    forecaster = MLForecaster(paths)
    forecast_results = forecaster.run(entity_frames)

    print("\n=== Step 4: Building interactive HTML dashboard ===")
    dashboard = InteractiveDashboardBuilder(paths)
    dash_path = dashboard.build(entity_frames, forecast_results, weekly_df)
    print(f"  ✅ Interactive dashboard saved to: {dash_path}")

    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"  📁 Clean data:      {paths.processed / 'clean_transactions.csv'}")
    print(f"  📁 Weekly features: {paths.processed / 'weekly_entity_features.csv'}")
    print(f"  📊 Dashboard:       {dash_path}")
    print(f"\n  Open the dashboard in your browser to explore the interactive visualizations!")


if __name__ == "__main__":
    main()
