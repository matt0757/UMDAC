"""
Machine Learning Net Cash Flow Forecasting
===========================================
This script uses ML models to forecast net cash flow for 1 month ahead.
Visualizes ONLY the backtest of the latest month (last 4 weeks).

Models:
1. XGBoost - Gradient boosting, excellent for tabular data
2. LightGBM - Fast gradient boosting, handles large datasets
3. Random Forest - Ensemble method, robust to outliers
4. Ridge Regression - Linear model with regularization
5. Ensemble - Weighted average of best models

Author: AI Assistant
Date: 2025-12-19
"""

import pandas as pd
import numpy as np

# Set matplotlib backend BEFORE importing pyplot to avoid Tkinter threading issues
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Try importing XGBoost and LightGBM
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    print("LightGBM not installed. Install with: pip install lightgbm")
    HAS_LGB = False

# Configuration
DATA_DIR = 'd:/UM_DATATHON/UMDAC/processed_data'
OUTPUT_DIR = 'd:/UM_DATATHON/UMDAC/outputs/ml_forecast'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Forecast settings
BACKTEST_WEEKS = 4  # Last 1 month for backtest
FORECAST_WEEKS = 4  # Forecast 1 month ahead

# Feature columns to use (based on repository analysis)
FEATURE_COLS = [
    'Week_of_Month', 'Is_Month_End', 'Month', 'Quarter',
    'Net_Lag1', 'Net_Lag2', 'Net_Lag4',
    'Net_Rolling4_Mean', 'Net_Rolling4_Std',
    'Transaction_Count', 'Inflow_Count', 'Outflow_Count',
    'Inflow_Lag1', 'Outflow_Lag1'
]


def load_entity_data(entity_code):
    """Load and preprocess entity data"""
    filepath = os.path.join(DATA_DIR, f'weekly_{entity_code}.csv')
    df = pd.read_csv(filepath)
    
    # Convert to datetime
    df['Week_Start'] = pd.to_datetime(df['Week_Start'])
    
    # Sort by week
    df = df.sort_values('Week_Start').reset_index(drop=True)
    
    return df


def prepare_features(df):
    """
    Prepare features for ML models.
    Uses lag features and rolling statistics as predictors.
    """
    feature_df = df.copy()
    
    # Convert boolean to int if needed
    if 'Is_Month_End' in feature_df.columns:
        feature_df['Is_Month_End'] = feature_df['Is_Month_End'].astype(int)
    
    # Available features (check what exists in the data)
    available_features = []
    for col in FEATURE_COLS:
        if col in feature_df.columns:
            available_features.append(col)
    
    # Fill NaN in lag features with 0 (for first rows)
    for col in available_features:
        if 'Lag' in col or 'Rolling' in col:
            feature_df[col] = feature_df[col].fillna(0)
    
    return feature_df, available_features


def create_future_features(df, n_periods):
    """
    Create feature values for future periods.
    This is tricky for ML models - we need to estimate future feature values.
    """
    last_row = df.iloc[-1]
    last_date = last_row['Week_Start']
    
    future_rows = []
    
    # For recursive forecasting, we'll use a simplified approach
    # In practice, you'd want to update lag features with predictions
    for i in range(1, n_periods + 1):
        future_date = last_date + timedelta(days=7 * i)
        
        # Calculate time-based features
        week_of_month = ((future_date.day - 1) // 7) + 1
        is_month_end = 1 if future_date.day >= 25 else 0
        month = future_date.month
        quarter = (month - 1) // 3 + 1
        
        future_row = {
            'Week_Start': future_date,
            'Week_of_Month': week_of_month,
            'Is_Month_End': is_month_end,
            'Month': month,
            'Quarter': quarter,
            # Use last known values for lag features (simplified approach)
            'Net_Lag1': last_row['Total_Net'] if i == 1 else future_rows[-1].get('predicted', 0),
            'Net_Lag2': df['Total_Net'].iloc[-1] if i <= 2 else future_rows[-2].get('predicted', 0),
            'Net_Lag4': df['Total_Net'].iloc[-min(4, len(df))] if i <= 4 else 0,
            'Net_Rolling4_Mean': df['Total_Net'].tail(4).mean(),
            'Net_Rolling4_Std': df['Total_Net'].tail(4).std(),
            'Transaction_Count': df['Transaction_Count'].mean() if 'Transaction_Count' in df.columns else 0,
            'Inflow_Count': df['Inflow_Count'].mean() if 'Inflow_Count' in df.columns else 0,
            'Outflow_Count': df['Outflow_Count'].mean() if 'Outflow_Count' in df.columns else 0,
            'Inflow_Lag1': df['Total_Inflow'].iloc[-1] if 'Total_Inflow' in df.columns else 0,
            'Outflow_Lag1': df['Total_Outflow'].iloc[-1] if 'Total_Outflow' in df.columns else 0,
        }
        future_rows.append(future_row)
    
    return pd.DataFrame(future_rows)


def calculate_metrics(actual, forecast):
    """Calculate comprehensive forecast accuracy metrics"""
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    
    # MAPE - handle zeros
    mape_mask = actual != 0
    if mape_mask.any():
        mape = np.mean(np.abs((actual[mape_mask] - forecast[mape_mask]) / actual[mape_mask])) * 100
    else:
        mape = np.nan
    
    # Direction accuracy
    if len(actual) > 1:
        actual_direction = np.sign(np.diff(actual))
        forecast_direction = np.sign(np.diff(forecast))
        direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
    else:
        direction_accuracy = np.nan
    
    # Sign accuracy
    sign_accuracy = np.mean(np.sign(actual) == np.sign(forecast)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy,
        'Sign_Accuracy': sign_accuracy
    }


def train_xgboost(X_train, y_train, X_test):
    """Train XGBoost model with optimized hyperparameters"""
    if not HAS_XGB:
        return None
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, model


def train_lightgbm(X_train, y_train, X_test):
    """Train LightGBM model with optimized hyperparameters"""
    if not HAS_LGB:
        return None
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, model


def train_random_forest(X_train, y_train, X_test):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, model


def train_gradient_boosting(X_train, y_train, X_test):
    """Train Gradient Boosting model (sklearn version)"""
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions, model


def train_ridge(X_train, y_train, X_test, scaler=None):
    """Train Ridge Regression model"""
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    return predictions, model, scaler


def run_ml_forecast(entity_code):
    """
    Run complete ML forecast pipeline for one entity.
    Only visualizes backtest of the latest 4 weeks.
    """
    print(f"\n{'#'*70}")
    print(f"ML Forecasting for Entity: {entity_code}")
    print(f"{'#'*70}")
    
    # Load and prepare data
    df = load_entity_data(entity_code)
    feature_df, available_features = prepare_features(df)
    
    if len(feature_df) < 10:
        print(f"  Insufficient data ({len(feature_df)} weeks). Skipping.")
        return None
    
    print(f"  Data: {len(feature_df)} weeks from {df['Week_Start'].min().strftime('%Y-%m-%d')} to {df['Week_Start'].max().strftime('%Y-%m-%d')}")
    print(f"  Features: {len(available_features)}")
    
    # Remove rows with NaN in features (usually first few rows due to lags)
    feature_df = feature_df.dropna(subset=available_features)
    
    # Split data for backtest
    train_df = feature_df.iloc[:-BACKTEST_WEEKS].copy()
    test_df = feature_df.iloc[-BACKTEST_WEEKS:].copy()
    
    if len(train_df) < 5:
        print(f"  Insufficient training data after removing NaN. Skipping.")
        return None
    
    print(f"  Training period: {train_df['Week_Start'].iloc[0].strftime('%Y-%m-%d')} to {train_df['Week_Start'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Backtest period: {test_df['Week_Start'].iloc[0].strftime('%Y-%m-%d')} to {test_df['Week_Start'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # Prepare X and y
    X_train = train_df[available_features].values
    y_train = train_df['Total_Net'].values
    X_test = test_df[available_features].values
    y_test = test_df['Total_Net'].values
    
    results = {}
    
    # Model 1: XGBoost
    if HAS_XGB:
        print("\n  Training XGBoost...")
        try:
            xgb_pred, xgb_model = train_xgboost(X_train, y_train, X_test)
            xgb_metrics = calculate_metrics(y_test, xgb_pred)
            results['XGBoost'] = {
                'backtest': xgb_pred,
                'metrics': xgb_metrics,
                'model': xgb_model
            }
            print(f"    MAE: {xgb_metrics['MAE']:,.2f}, RMSE: {xgb_metrics['RMSE']:,.2f}")
        except Exception as e:
            print(f"    XGBoost failed: {e}")
    
    # Model 2: LightGBM
    if HAS_LGB:
        print("  Training LightGBM...")
        try:
            lgb_pred, lgb_model = train_lightgbm(X_train, y_train, X_test)
            lgb_metrics = calculate_metrics(y_test, lgb_pred)
            results['LightGBM'] = {
                'backtest': lgb_pred,
                'metrics': lgb_metrics,
                'model': lgb_model
            }
            print(f"    MAE: {lgb_metrics['MAE']:,.2f}, RMSE: {lgb_metrics['RMSE']:,.2f}")
        except Exception as e:
            print(f"    LightGBM failed: {e}")
    
    # Model 3: Random Forest
    print("  Training Random Forest...")
    try:
        rf_pred, rf_model = train_random_forest(X_train, y_train, X_test)
        rf_metrics = calculate_metrics(y_test, rf_pred)
        results['RandomForest'] = {
            'backtest': rf_pred,
            'metrics': rf_metrics,
            'model': rf_model
        }
        print(f"    MAE: {rf_metrics['MAE']:,.2f}, RMSE: {rf_metrics['RMSE']:,.2f}")
    except Exception as e:
        print(f"    Random Forest failed: {e}")
    
    # Model 4: Gradient Boosting (sklearn)
    print("  Training Gradient Boosting...")
    try:
        gb_pred, gb_model = train_gradient_boosting(X_train, y_train, X_test)
        gb_metrics = calculate_metrics(y_test, gb_pred)
        results['GradientBoosting'] = {
            'backtest': gb_pred,
            'metrics': gb_metrics,
            'model': gb_model
        }
        print(f"    MAE: {gb_metrics['MAE']:,.2f}, RMSE: {gb_metrics['RMSE']:,.2f}")
    except Exception as e:
        print(f"    Gradient Boosting failed: {e}")
    
    # Model 5: Ridge Regression
    print("  Training Ridge Regression...")
    try:
        ridge_pred, ridge_model, scaler = train_ridge(X_train, y_train, X_test)
        ridge_metrics = calculate_metrics(y_test, ridge_pred)
        results['Ridge'] = {
            'backtest': ridge_pred,
            'metrics': ridge_metrics,
            'model': ridge_model,
            'scaler': scaler
        }
        print(f"    MAE: {ridge_metrics['MAE']:,.2f}, RMSE: {ridge_metrics['RMSE']:,.2f}")
    except Exception as e:
        print(f"    Ridge failed: {e}")
    
    if not results:
        print("  No models trained successfully.")
        return None
    
    # Create Ensemble (weighted average based on inverse RMSE)
    print("  Creating Ensemble...")
    try:
        ensemble_weights = {}
        total_inv_rmse = 0
        for model_name, model_results in results.items():
            inv_rmse = 1.0 / max(model_results['metrics']['RMSE'], 1)
            ensemble_weights[model_name] = inv_rmse
            total_inv_rmse += inv_rmse
        
        # Normalize weights
        for model_name in ensemble_weights:
            ensemble_weights[model_name] /= total_inv_rmse
        
        # Create weighted ensemble prediction
        ensemble_pred = np.zeros(BACKTEST_WEEKS)
        for model_name, weight in ensemble_weights.items():
            ensemble_pred += weight * results[model_name]['backtest']
        
        ensemble_metrics = calculate_metrics(y_test, ensemble_pred)
        results['Ensemble'] = {
            'backtest': ensemble_pred,
            'metrics': ensemble_metrics,
            'weights': ensemble_weights
        }
        print(f"    MAE: {ensemble_metrics['MAE']:,.2f}, RMSE: {ensemble_metrics['RMSE']:,.2f}")
        print(f"    Weights: {', '.join([f'{k}: {v:.2%}' for k, v in ensemble_weights.items()])}")
    except Exception as e:
        print(f"    Ensemble failed: {e}")
    
    # Select best model based on RMSE
    best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['RMSE'])
    print(f"\n  *** Best Model: {best_model_name} (RMSE: {results[best_model_name]['metrics']['RMSE']:,.2f}) ***")
    
    # Generate future forecast
    print(f"\n  Generating {FORECAST_WEEKS}-week ahead forecast...")
    
    # Use all data to retrain best model
    full_X = feature_df[available_features].values
    full_y = feature_df['Total_Net'].values
    
    # Create future features
    future_df = create_future_features(df, FORECAST_WEEKS)
    future_X = future_df[available_features].values
    
    # Retrain and predict
    if best_model_name == 'XGBoost' and HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        model.fit(full_X, full_y)
        future_pred = model.predict(future_X)
    elif best_model_name == 'LightGBM' and HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        model.fit(full_X, full_y)
        future_pred = model.predict(future_X)
    elif best_model_name == 'RandomForest':
        model = RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_split=3,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        model.fit(full_X, full_y)
        future_pred = model.predict(future_X)
    elif best_model_name == 'GradientBoosting':
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(full_X, full_y)
        future_pred = model.predict(future_X)
    elif best_model_name == 'Ensemble':
        # Use ensemble for future prediction
        future_pred = np.zeros(FORECAST_WEEKS)
        for model_name, weight in results['Ensemble']['weights'].items():
            if model_name == 'XGBoost' and HAS_XGB:
                m = xgb.XGBRegressor(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
            elif model_name == 'LightGBM' and HAS_LGB:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, verbose=-1, random_state=42)
            elif model_name == 'RandomForest':
                m = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            elif model_name == 'GradientBoosting':
                m = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
            else:
                continue
            m.fit(full_X, full_y)
            future_pred += weight * m.predict(future_X)
    else:  # Ridge
        scaler = StandardScaler()
        full_X_scaled = scaler.fit_transform(full_X)
        future_X_scaled = scaler.transform(future_X)
        model = Ridge(alpha=1.0)
        model.fit(full_X_scaled, full_y)
        future_pred = model.predict(future_X_scaled)
    
    # Estimate confidence intervals (using historical std)
    hist_std = np.std(full_y)
    
    results['future'] = {
        'dates': future_df['Week_Start'],
        'values': future_pred,
        'conf_lower': future_pred - 1.96 * hist_std,
        'conf_upper': future_pred + 1.96 * hist_std
    }
    
    # Store data for visualization
    results['test_data'] = test_df
    results['train_data'] = train_df
    results['full_data'] = feature_df
    results['best_model'] = best_model_name
    results['available_features'] = available_features
    
    return results


def create_backtest_visualization(entity_code, results):
    """
    Create visualization ONLY for the backtest of the latest month (4 weeks).
    """
    fig = plt.figure(figsize=(16, 10))
    
    colors = {
        'actual': '#2C3E50',
        'XGBoost': '#E74C3C',
        'LightGBM': '#3498DB',
        'RandomForest': '#27AE60',
        'GradientBoosting': '#9B59B6',
        'Ridge': '#F39C12',
        'Ensemble': '#1ABC9C'
    }
    
    test_data = results['test_data']
    best_model = results['best_model']
    
    # Plot 1: Backtest Comparison - Line Chart
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Actual values
    ax1.plot(test_data['Week_Start'], test_data['Total_Net'], 
             'o-', color=colors['actual'], linewidth=3, markersize=10,
             label='Actual', alpha=0.9)
    
    # All model forecasts
    for model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'Ridge', 'Ensemble']:
        if model_name in results:
            linestyle = '-' if model_name == best_model else '--'
            linewidth = 3 if model_name == best_model else 1.5
            ax1.plot(test_data['Week_Start'], results[model_name]['backtest'],
                    linestyle=linestyle, linewidth=linewidth,
                    color=colors.get(model_name, 'gray'),
                    marker='s', markersize=8 if model_name == best_model else 5,
                    label=f'{model_name}{"*" if model_name == best_model else ""}', 
                    alpha=0.8)
    
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_title(f'{entity_code}: Backtest - Latest 4 Weeks (ML Models)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Net Cash Flow (USD)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Backtest Comparison - Bar Chart
    ax2 = fig.add_subplot(2, 2, 2)
    
    x = np.arange(BACKTEST_WEEKS)
    width = 0.12
    
    # Actual bars
    ax2.bar(x - 3*width, test_data['Total_Net'].values, width,
           label='Actual', color=colors['actual'], alpha=0.9, edgecolor='black')
    
    # Model forecast bars
    offset = -2*width
    for model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'Ridge', 'Ensemble']:
        if model_name in results:
            alpha = 0.9 if model_name == best_model else 0.6
            edge = 'black' if model_name == best_model else 'none'
            ax2.bar(x + offset, results[model_name]['backtest'], width,
                   label=model_name, color=colors.get(model_name, 'gray'), 
                   alpha=alpha, edgecolor=edge)
            offset += width
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%m/%d') for d in test_data['Week_Start']])
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title('Backtest Comparison (Bar Chart)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Net Cash Flow')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Error Metrics Comparison
    ax3 = fig.add_subplot(2, 2, 3)
    
    model_names = []
    rmse_values = []
    mae_values = []
    
    for model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'Ridge', 'Ensemble']:
        if model_name in results:
            model_names.append(model_name)
            rmse_values.append(results[model_name]['metrics']['RMSE'])
            mae_values.append(results[model_name]['metrics']['MAE'])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, rmse_values, width, label='RMSE', color='#E74C3C', alpha=0.8)
    bars2 = ax3.bar(x + width/2, mae_values, width, label='MAE', color='#3498DB', alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.set_title('Backtest Error Metrics by Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Error Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight best model
    if best_model in model_names:
        best_idx = model_names.index(best_model)
        ax3.get_xticklabels()[best_idx].set_weight('bold')
        ax3.get_xticklabels()[best_idx].set_color('green')
    
    # Plot 4: Feature Importance (if available)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Try to get feature importance from best model
    if best_model in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
        model = results[best_model].get('model')
        if model is not None and hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = results['available_features']
            
            # Sort by importance
            idx = np.argsort(importance)[::-1][:10]  # Top 10 features
            
            ax4.barh(range(len(idx)), importance[idx], color='#3498DB', alpha=0.8)
            ax4.set_yticks(range(len(idx)))
            ax4.set_yticklabels([features[i] for i in idx])
            ax4.set_xlabel('Importance')
            ax4.set_title(f'Top Feature Importance ({best_model})', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
    else:
        # Show backtest errors instead
        best_backtest = results[best_model]['backtest']
        actual_values = test_data['Total_Net'].values
        errors = best_backtest - actual_values
        
        colors_error = ['#27AE60' if e >= 0 else '#E74C3C' for e in errors]
        ax4.bar(range(len(errors)), errors, color=colors_error, alpha=0.8, edgecolor='black')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xticks(range(len(errors)))
        ax4.set_xticklabels([d.strftime('%m/%d') for d in test_data['Week_Start']])
        ax4.set_title(f'Forecast Errors ({best_model})', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Week')
        ax4.set_ylabel('Error (Forecast - Actual)')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'ML Backtest Analysis - {entity_code}\n(Latest 4 Weeks: {test_data["Week_Start"].iloc[0].strftime("%Y-%m-%d")} to {test_data["Week_Start"].iloc[-1].strftime("%Y-%m-%d")})', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f'{entity_code}_ml_backtest.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  Backtest visualization saved to: {output_path}")
    return output_path


def print_summary(all_results):
    """Print comprehensive summary of all entity forecasts"""
    print("\n" + "="*100)
    print("ML FORECAST SUMMARY - All Entities")
    print("="*100)
    
    print(f"\n{'Entity':<10} {'Best Model':<18} {'Backtest MAE':>15} {'Backtest RMSE':>15} {'Direction Acc':>15} {'Sign Acc':>10}")
    print("-"*90)
    
    for entity, results in all_results.items():
        if results:
            best = results['best_model']
            metrics = results[best]['metrics']
            print(f"{entity:<10} {best:<18} {metrics['MAE']:>15,.2f} {metrics['RMSE']:>15,.2f} {metrics['Direction_Accuracy']:>14.1f}% {metrics['Sign_Accuracy']:>9.1f}%")
    
    print("\n" + "="*100)
    print("1-MONTH FORECAST VALUES")
    print("="*100)
    
    for entity, results in all_results.items():
        if results and 'future' in results:
            fc = results['future']
            print(f"\n{entity} (Best Model: {results['best_model']}):")
            total = 0
            for i, (date, val, low, high) in enumerate(zip(
                fc['dates'], fc['values'], fc['conf_lower'], fc['conf_upper'])):
                print(f"  Week {i+1} ({date.strftime('%Y-%m-%d')}): {val:>15,.2f}  [95% CI: {low:>15,.2f} to {high:>15,.2f}]")
                total += val
            print(f"  {'Monthly Total':<30}: {total:>15,.2f}")


def main():
    """Main execution"""
    print("="*70)
    print("MACHINE LEARNING NET CASH FLOW FORECASTING")
    print("Backtest: Latest 4 Weeks (1 Month)")
    print("Forecast Horizon: 4 Weeks (1 Month) Ahead")
    print("="*70)
    
    print(f"\nAvailable Libraries:")
    print(f"  XGBoost: {'Yes' if HAS_XGB else 'No (install with: pip install xgboost)'}")
    print(f"  LightGBM: {'Yes' if HAS_LGB else 'No (install with: pip install lightgbm)'}")
    
    # List of entities
    entities = ['KR10', 'ID10', 'MY10', 'PH10', 'SS10', 'TH10', 'TW10', 'VN20']
    
    all_results = {}
    
    for entity in entities:
        try:
            results = run_ml_forecast(entity)
            if results:
                all_results[entity] = results
                create_backtest_visualization(entity, results)
        except Exception as e:
            print(f"Error processing {entity}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print_summary(all_results)
    
    print(f"\n✅ ML forecasting complete!")
    print(f"   Backtest visualizations saved to: {OUTPUT_DIR}")
    
    return all_results


if __name__ == "__main__":
    results = main()
