"""
Prophet Net Cash Flow Forecasting
==================================
This script uses Facebook Prophet to forecast the net cash flow for 1 month ahead.
Visualizes ONLY the backtest of the latest month (last 4 weeks).

Key Hyperparameters (derived from repository analysis):
- Weekly frequency with 4-week monthly seasonality
- Flexible changepoint detection for volatile financial data
- Additive seasonality for net cash flows (can be negative)
- Custom month-end effects as additional regressor

Author: AI Assistant
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Prophet installation check
try:
    from prophet import Prophet
except ImportError:
    print("Prophet not installed. Install with: pip install prophet")
    raise

from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Configuration
DATA_DIR = 'd:/UM_DATATHON/UMDAC/processed_data'
OUTPUT_DIR = 'd:/UM_DATATHON/UMDAC/outputs/prophet_1month'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Forecast settings (derived from repo analysis)
BACKTEST_WEEKS = 4  # Last 1 month for backtest
FORECAST_WEEKS = 4  # Forecast 1 month ahead

# Prophet Hyperparameters (optimized based on repository patterns)
# These are tuned for volatile financial time series data
PROPHET_PARAMS = {
    # Trend flexibility - higher value = more flexible trend (volatile cash flows need this)
    'changepoint_prior_scale': 0.5,
    
    # Seasonality strength - moderate to capture weekly patterns without overfitting
    'seasonality_prior_scale': 10.0,
    
    # Seasonality mode - additive because net cash flow can be negative
    'seasonality_mode': 'additive',
    
    # Changepoint range - allow changepoints up to 90% of history
    'changepoint_range': 0.9,
    
    # Yearly seasonality - disabled (we only have ~10 months of data)
    'yearly_seasonality': False,
    
    # Weekly seasonality - could be useful for day-level, but we're on weekly level
    'weekly_seasonality': False,
    
    # Daily seasonality - disabled for weekly data
    'daily_seasonality': False,
    
    # Growth model - linear for financial data
    'growth': 'linear',
    
    # Interval width for uncertainty
    'interval_width': 0.95,
    
    # Number of changepoints
    'n_changepoints': 15,
}


def load_entity_data(entity_code):
    """Load and preprocess entity data for Prophet"""
    filepath = os.path.join(DATA_DIR, f'weekly_{entity_code}.csv')
    df = pd.read_csv(filepath)
    
    # Convert to datetime
    df['Week_Start'] = pd.to_datetime(df['Week_Start'])
    
    # Sort by week
    df = df.sort_values('Week_Start').reset_index(drop=True)
    
    return df


def prepare_prophet_data(df):
    """
    Prepare data in Prophet format: 
    - ds: datetime column
    - y: target variable
    - Additional regressors for month-end effects
    """
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df['Week_Start']
    prophet_df['y'] = df['Total_Net']
    
    # Additional regressors based on repository patterns
    prophet_df['is_month_end'] = df['Is_Month_End'].astype(int)
    prophet_df['week_of_month'] = df['Week_of_Month']
    
    # Transaction volume can be a useful regressor (if available)
    if 'Transaction_Count' in df.columns:
        prophet_df['transaction_count'] = df['Transaction_Count']
    
    return prophet_df


def create_prophet_model(include_regressors=True):
    """
    Create and configure Prophet model with optimized hyperparameters
    
    Parameters are derived from analyzing the existing repository patterns:
    - High volatility needs flexible trend (changepoint_prior_scale=0.5)
    - Monthly patterns need custom seasonality (4-week period)
    - Additive mode for net cash flows that can be negative
    """
    model = Prophet(
        changepoint_prior_scale=PROPHET_PARAMS['changepoint_prior_scale'],
        seasonality_prior_scale=PROPHET_PARAMS['seasonality_prior_scale'],
        seasonality_mode=PROPHET_PARAMS['seasonality_mode'],
        changepoint_range=PROPHET_PARAMS['changepoint_range'],
        yearly_seasonality=PROPHET_PARAMS['yearly_seasonality'],
        weekly_seasonality=PROPHET_PARAMS['weekly_seasonality'],
        daily_seasonality=PROPHET_PARAMS['daily_seasonality'],
        growth=PROPHET_PARAMS['growth'],
        interval_width=PROPHET_PARAMS['interval_width'],
        n_changepoints=PROPHET_PARAMS['n_changepoints'],
    )
    
    # Add custom monthly seasonality (4-week period) - key pattern from repo
    model.add_seasonality(
        name='monthly',
        period=28,  # 4 weeks = 28 days
        fourier_order=3,  # Lower order to avoid overfitting with limited data
    )
    
    # Add regressors for month-end effects (important pattern from repo)
    if include_regressors:
        model.add_regressor('is_month_end', standardize=False)
    
    return model


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
    
    # Direction accuracy - did we get the direction of change right?
    actual_direction = np.sign(np.diff(np.concatenate([[0], actual])))
    forecast_direction = np.sign(np.diff(np.concatenate([[0], forecast])))
    direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
    
    # Sign accuracy - did we correctly predict positive vs negative?
    sign_accuracy = np.mean(np.sign(actual) == np.sign(forecast)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy,
        'Sign_Accuracy': sign_accuracy
    }


def run_prophet_forecast(entity_code):
    """
    Run complete Prophet forecast pipeline for one entity.
    Only visualizes backtest of the latest 4 weeks.
    """
    print(f"\n{'#'*70}")
    print(f"Prophet Forecasting for Entity: {entity_code}")
    print(f"{'#'*70}")
    
    # Load and prepare data
    df = load_entity_data(entity_code)
    prophet_df = prepare_prophet_data(df)
    
    if len(prophet_df) < 10:
        print(f"  Insufficient data ({len(prophet_df)} weeks). Skipping.")
        return None
    
    print(f"  Data: {len(prophet_df)} weeks from {prophet_df['ds'].min().strftime('%Y-%m-%d')} to {prophet_df['ds'].max().strftime('%Y-%m-%d')}")
    
    # Split data for backtest (last 4 weeks = 1 month)
    train_df = prophet_df.iloc[:-BACKTEST_WEEKS].copy()
    test_df = prophet_df.iloc[-BACKTEST_WEEKS:].copy()
    
    print(f"  Training period: {train_df['ds'].iloc[0].strftime('%Y-%m-%d')} to {train_df['ds'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Backtest period: {test_df['ds'].iloc[0].strftime('%Y-%m-%d')} to {test_df['ds'].iloc[-1].strftime('%Y-%m-%d')}")
    
    results = {}
    
    # Model 1: Prophet with basic settings
    print("\n  Fitting Prophet (basic)...")
    try:
        model_basic = Prophet(
            changepoint_prior_scale=0.1,  # More conservative
            seasonality_mode='additive',
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
        )
        model_basic.add_seasonality(name='monthly', period=28, fourier_order=2)
        model_basic.fit(train_df[['ds', 'y']])
        
        # Backtest forecast
        future_basic = model_basic.make_future_dataframe(periods=BACKTEST_WEEKS, freq='W')
        forecast_basic = model_basic.predict(future_basic)
        backtest_basic = forecast_basic.iloc[-BACKTEST_WEEKS:]['yhat'].values
        
        basic_metrics = calculate_metrics(test_df['y'].values, backtest_basic)
        results['Prophet_Basic'] = {
            'backtest': backtest_basic,
            'metrics': basic_metrics,
            'model': model_basic,
            'forecast_full': forecast_basic
        }
        print(f"    MAE: {basic_metrics['MAE']:,.2f}, RMSE: {basic_metrics['RMSE']:,.2f}")
    except Exception as e:
        print(f"    Prophet (basic) failed: {e}")
    
    # Model 2: Prophet with optimized hyperparameters (from repo analysis)
    print("  Fitting Prophet (optimized)...")
    try:
        model_opt = Prophet(
            changepoint_prior_scale=PROPHET_PARAMS['changepoint_prior_scale'],
            seasonality_prior_scale=PROPHET_PARAMS['seasonality_prior_scale'],
            seasonality_mode=PROPHET_PARAMS['seasonality_mode'],
            changepoint_range=PROPHET_PARAMS['changepoint_range'],
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
            n_changepoints=PROPHET_PARAMS['n_changepoints'],
        )
        model_opt.add_seasonality(name='monthly', period=28, fourier_order=3)
        model_opt.fit(train_df[['ds', 'y']])
        
        future_opt = model_opt.make_future_dataframe(periods=BACKTEST_WEEKS, freq='W')
        forecast_opt = model_opt.predict(future_opt)
        backtest_opt = forecast_opt.iloc[-BACKTEST_WEEKS:]['yhat'].values
        
        opt_metrics = calculate_metrics(test_df['y'].values, backtest_opt)
        results['Prophet_Optimized'] = {
            'backtest': backtest_opt,
            'metrics': opt_metrics,
            'model': model_opt,
            'forecast_full': forecast_opt
        }
        print(f"    MAE: {opt_metrics['MAE']:,.2f}, RMSE: {opt_metrics['RMSE']:,.2f}")
    except Exception as e:
        print(f"    Prophet (optimized) failed: {e}")
    
    # Model 3: Prophet with regressors (month-end effect)
    print("  Fitting Prophet (with regressors)...")
    try:
        model_reg = Prophet(
            changepoint_prior_scale=PROPHET_PARAMS['changepoint_prior_scale'],
            seasonality_prior_scale=PROPHET_PARAMS['seasonality_prior_scale'],
            seasonality_mode=PROPHET_PARAMS['seasonality_mode'],
            changepoint_range=PROPHET_PARAMS['changepoint_range'],
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
            n_changepoints=PROPHET_PARAMS['n_changepoints'],
        )
        model_reg.add_seasonality(name='monthly', period=28, fourier_order=3)
        model_reg.add_regressor('is_month_end', standardize=False)
        
        train_reg = train_df[['ds', 'y', 'is_month_end']].copy()
        model_reg.fit(train_reg)
        
        # Create future dataframe with regressor
        future_reg = model_reg.make_future_dataframe(periods=BACKTEST_WEEKS, freq='W')
        # Add regressor values for future dates
        full_is_month_end = pd.concat([
            train_df[['ds', 'is_month_end']], 
            test_df[['ds', 'is_month_end']]
        ], ignore_index=True)
        future_reg = future_reg.merge(full_is_month_end, on='ds', how='left')
        future_reg['is_month_end'] = future_reg['is_month_end'].fillna(0).astype(int)
        
        forecast_reg = model_reg.predict(future_reg)
        backtest_reg = forecast_reg.iloc[-BACKTEST_WEEKS:]['yhat'].values
        
        reg_metrics = calculate_metrics(test_df['y'].values, backtest_reg)
        results['Prophet_Regressor'] = {
            'backtest': backtest_reg,
            'metrics': reg_metrics,
            'model': model_reg,
            'forecast_full': forecast_reg
        }
        print(f"    MAE: {reg_metrics['MAE']:,.2f}, RMSE: {reg_metrics['RMSE']:,.2f}")
    except Exception as e:
        print(f"    Prophet (with regressors) failed: {e}")
    
    # Model 4: Prophet with aggressive flexibility (for high volatility)
    print("  Fitting Prophet (high flexibility)...")
    try:
        model_flex = Prophet(
            changepoint_prior_scale=1.0,  # Very flexible
            seasonality_prior_scale=15.0,
            seasonality_mode='additive',
            changepoint_range=0.95,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
            n_changepoints=20,
        )
        model_flex.add_seasonality(name='monthly', period=28, fourier_order=4)
        model_flex.fit(train_df[['ds', 'y']])
        
        future_flex = model_flex.make_future_dataframe(periods=BACKTEST_WEEKS, freq='W')
        forecast_flex = model_flex.predict(future_flex)
        backtest_flex = forecast_flex.iloc[-BACKTEST_WEEKS:]['yhat'].values
        
        flex_metrics = calculate_metrics(test_df['y'].values, backtest_flex)
        results['Prophet_Flexible'] = {
            'backtest': backtest_flex,
            'metrics': flex_metrics,
            'model': model_flex,
            'forecast_full': forecast_flex
        }
        print(f"    MAE: {flex_metrics['MAE']:,.2f}, RMSE: {flex_metrics['RMSE']:,.2f}")
    except Exception as e:
        print(f"    Prophet (high flexibility) failed: {e}")
    
    if not results:
        print("  No models fitted successfully.")
        return None
    
    # Select best model based on RMSE
    best_model_name = min(results.keys(), key=lambda k: results[k]['metrics']['RMSE'])
    print(f"\n  *** Best Model: {best_model_name} (RMSE: {results[best_model_name]['metrics']['RMSE']:,.2f}) ***")
    
    # Generate future forecast using FULL data with best model parameters
    print(f"\n  Generating {FORECAST_WEEKS}-week ahead forecast...")
    
    # Refit best model on full data
    best_params = get_best_model_params(best_model_name)
    final_model = Prophet(**best_params)
    final_model.add_seasonality(name='monthly', period=28, fourier_order=3)
    
    if 'Regressor' in best_model_name:
        final_model.add_regressor('is_month_end', standardize=False)
        final_model.fit(prophet_df[['ds', 'y', 'is_month_end']])
        
        # Create future dataframe
        future = final_model.make_future_dataframe(periods=FORECAST_WEEKS, freq='W')
        # Estimate month-end for future
        future['is_month_end'] = future['ds'].apply(lambda x: 1 if x.day >= 25 else 0)
    else:
        final_model.fit(prophet_df[['ds', 'y']])
        future = final_model.make_future_dataframe(periods=FORECAST_WEEKS, freq='W')
    
    final_forecast = final_model.predict(future)
    
    # Extract future forecast
    future_forecast = final_forecast.iloc[-FORECAST_WEEKS:]
    results['future'] = {
        'dates': future_forecast['ds'],
        'values': future_forecast['yhat'].values,
        'conf_lower': future_forecast['yhat_lower'].values,
        'conf_upper': future_forecast['yhat_upper'].values
    }
    
    # Store data for visualization
    results['test_data'] = test_df
    results['train_data'] = train_df
    results['full_data'] = prophet_df
    results['best_model'] = best_model_name
    results['final_model'] = final_model
    results['final_forecast'] = final_forecast
    
    return results


def get_best_model_params(model_name):
    """Get Prophet parameters for the best model"""
    if model_name == 'Prophet_Basic':
        return {
            'changepoint_prior_scale': 0.1,
            'seasonality_mode': 'additive',
            'yearly_seasonality': False,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'interval_width': 0.95,
        }
    elif model_name == 'Prophet_Flexible':
        return {
            'changepoint_prior_scale': 1.0,
            'seasonality_prior_scale': 15.0,
            'seasonality_mode': 'additive',
            'changepoint_range': 0.95,
            'yearly_seasonality': False,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'interval_width': 0.95,
            'n_changepoints': 20,
        }
    else:  # Prophet_Optimized or Prophet_Regressor
        return {
            'changepoint_prior_scale': PROPHET_PARAMS['changepoint_prior_scale'],
            'seasonality_prior_scale': PROPHET_PARAMS['seasonality_prior_scale'],
            'seasonality_mode': PROPHET_PARAMS['seasonality_mode'],
            'changepoint_range': PROPHET_PARAMS['changepoint_range'],
            'yearly_seasonality': False,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'interval_width': 0.95,
            'n_changepoints': PROPHET_PARAMS['n_changepoints'],
        }


def create_backtest_visualization(entity_code, results):
    """
    Create visualization ONLY for the backtest of the latest month (4 weeks).
    This is the key visualization requested by the user.
    """
    fig = plt.figure(figsize=(16, 10))
    
    colors = {
        'actual': '#2C3E50',
        'Prophet_Basic': '#E74C3C',
        'Prophet_Optimized': '#3498DB',
        'Prophet_Regressor': '#27AE60',
        'Prophet_Flexible': '#9B59B6',
        'future': '#F39C12'
    }
    
    test_data = results['test_data']
    best_model = results['best_model']
    
    # Plot 1: Backtest Comparison - Line Chart
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Actual values
    ax1.plot(test_data['ds'], test_data['y'], 
             'o-', color=colors['actual'], linewidth=3, markersize=10,
             label='Actual', alpha=0.9)
    
    # All model forecasts
    for model_name in ['Prophet_Basic', 'Prophet_Optimized', 'Prophet_Regressor', 'Prophet_Flexible']:
        if model_name in results:
            linestyle = '-' if model_name == best_model else '--'
            linewidth = 3 if model_name == best_model else 1.5
            ax1.plot(test_data['ds'], results[model_name]['backtest'],
                    linestyle=linestyle, linewidth=linewidth,
                    color=colors.get(model_name, 'gray'),
                    marker='s', markersize=8 if model_name == best_model else 5,
                    label=f'{model_name}{"*" if model_name == best_model else ""}', 
                    alpha=0.8)
    
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_title(f'{entity_code}: Backtest - Latest 4 Weeks (Prophet Models)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Net Cash Flow (USD)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Backtest Comparison - Bar Chart
    ax2 = fig.add_subplot(2, 2, 2)
    
    x = np.arange(BACKTEST_WEEKS)
    width = 0.15
    
    # Actual bars
    ax2.bar(x - 2*width, test_data['y'].values, width,
           label='Actual', color=colors['actual'], alpha=0.9, edgecolor='black')
    
    # Model forecast bars
    offset = -width
    for model_name in ['Prophet_Basic', 'Prophet_Optimized', 'Prophet_Regressor', 'Prophet_Flexible']:
        if model_name in results:
            alpha = 0.9 if model_name == best_model else 0.6
            edge = 'black' if model_name == best_model else 'none'
            ax2.bar(x + offset, results[model_name]['backtest'], width,
                   label=model_name, color=colors.get(model_name, 'gray'), 
                   alpha=alpha, edgecolor=edge)
            offset += width
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%m/%d') for d in test_data['ds']])
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title('Backtest Comparison (Bar Chart)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Net Cash Flow')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Error Metrics Comparison
    ax3 = fig.add_subplot(2, 2, 3)
    
    model_names = []
    rmse_values = []
    mae_values = []
    
    for model_name in ['Prophet_Basic', 'Prophet_Optimized', 'Prophet_Regressor', 'Prophet_Flexible']:
        if model_name in results:
            model_names.append(model_name.replace('Prophet_', ''))
            rmse_values.append(results[model_name]['metrics']['RMSE'])
            mae_values.append(results[model_name]['metrics']['MAE'])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, rmse_values, width, label='RMSE', color='#E74C3C', alpha=0.8)
    bars2 = ax3.bar(x + width/2, mae_values, width, label='MAE', color='#3498DB', alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names)
    ax3.set_title('Backtest Error Metrics by Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Error Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight best model
    best_short_name = best_model.replace('Prophet_', '')
    if best_short_name in model_names:
        best_idx = model_names.index(best_short_name)
        ax3.get_xticklabels()[best_idx].set_weight('bold')
        ax3.get_xticklabels()[best_idx].set_color('green')
    
    # Plot 4: Backtest Error Values
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Error for best model
    best_backtest = results[best_model]['backtest']
    actual_values = test_data['y'].values
    errors = best_backtest - actual_values
    
    colors_error = ['#27AE60' if e >= 0 else '#E74C3C' for e in errors]
    ax4.bar(x, errors, color=colors_error, alpha=0.8, edgecolor='black')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.strftime('%m/%d') for d in test_data['ds']])
    ax4.set_title(f'Forecast Errors ({best_model})', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Week')
    ax4.set_ylabel('Error (Forecast - Actual)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add error values on bars
    for i, (e, actual, forecast) in enumerate(zip(errors, actual_values, best_backtest)):
        y_offset = e + abs(e) * 0.1 if e >= 0 else e - abs(e) * 0.1
        ax4.text(i, y_offset, f'{e:,.0f}', ha='center', 
                va='bottom' if e >= 0 else 'top', fontsize=8)
    
    plt.suptitle(f'Prophet Backtest Analysis - {entity_code}\n(Latest 4 Weeks: {test_data["ds"].iloc[0].strftime("%Y-%m-%d")} to {test_data["ds"].iloc[-1].strftime("%Y-%m-%d")})', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f'{entity_code}_prophet_backtest.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  Backtest visualization saved to: {output_path}")
    return output_path


def create_forecast_summary_visualization(entity_code, results):
    """Create additional summary visualization with future forecast"""
    fig = plt.figure(figsize=(14, 8))
    
    full_data = results['full_data']
    test_data = results['test_data']
    best_model = results['best_model']
    
    # Main plot: Full time series with backtest and future forecast
    ax = fig.add_subplot(1, 1, 1)
    
    # Historical data
    train_data = results['train_data']
    ax.plot(train_data['ds'], train_data['y'], 
            'o-', color='#2C3E50', linewidth=2, markersize=4,
            label='Historical', alpha=0.7)
    
    # Backtest period - actual
    ax.plot(test_data['ds'], test_data['y'], 
            'o-', color='#1ABC9C', linewidth=3, markersize=8,
            label='Actual (Backtest Period)', alpha=0.9)
    
    # Backtest prediction
    ax.plot(test_data['ds'], results[best_model]['backtest'],
            's--', color='#E74C3C', linewidth=2, markersize=8,
            label=f'Backtest Forecast ({best_model})', alpha=0.9)
    
    # Future forecast
    if 'future' in results:
        fc = results['future']
        ax.plot(fc['dates'], fc['values'], 
                's-', color='#F39C12', linewidth=3, markersize=10,
                label='1-Month Forecast')
        ax.fill_between(fc['dates'], fc['conf_lower'], fc['conf_upper'],
                       color='#F39C12', alpha=0.2, label='95% Confidence Interval')
    
    # Shade backtest period
    ax.axvspan(test_data['ds'].iloc[0], test_data['ds'].iloc[-1],
              alpha=0.1, color='green', label='Backtest Period (Latest 4 Weeks)')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_title(f'{entity_code}: Prophet Forecast - Full View', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Net Cash Flow (USD)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f'{entity_code}_prophet_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Summary visualization saved to: {output_path}")
    return output_path


def print_summary(all_results):
    """Print comprehensive summary of all entity forecasts"""
    print("\n" + "="*90)
    print("PROPHET FORECAST SUMMARY - All Entities")
    print("="*90)
    
    print(f"\n{'Entity':<10} {'Best Model':<20} {'Backtest MAE':>15} {'Backtest RMSE':>15} {'Direction Acc':>15}")
    print("-"*80)
    
    for entity, results in all_results.items():
        if results:
            best = results['best_model']
            metrics = results[best]['metrics']
            print(f"{entity:<10} {best:<20} {metrics['MAE']:>15,.2f} {metrics['RMSE']:>15,.2f} {metrics['Direction_Accuracy']:>14.1f}%")
    
    print("\n" + "="*90)
    print("1-MONTH FORECAST VALUES")
    print("="*90)
    
    for entity, results in all_results.items():
        if results and 'future' in results:
            fc = results['future']
            print(f"\n{entity}:")
            total = 0
            for i, (date, val, low, high) in enumerate(zip(
                fc['dates'], fc['values'], fc['conf_lower'], fc['conf_upper'])):
                print(f"  Week {i+1} ({date.strftime('%Y-%m-%d')}): {val:>15,.2f}  [95% CI: {low:>15,.2f} to {high:>15,.2f}]")
                total += val
            print(f"  {'Monthly Total':<30}: {total:>15,.2f}")
    
    print("\n" + "="*90)
    print("HYPERPARAMETER SUMMARY")
    print("="*90)
    print(f"\nOptimized Prophet Parameters:")
    for key, value in PROPHET_PARAMS.items():
        print(f"  {key}: {value}")


def main():
    """Main execution"""
    print("="*70)
    print("PROPHET NET CASH FLOW FORECASTING")
    print("Backtest: Latest 4 Weeks (1 Month)")
    print("Forecast Horizon: 4 Weeks (1 Month) Ahead")
    print("="*70)
    
    # List of entities (from repository analysis)
    entities = ['KR10', 'ID10', 'MY10', 'PH10', 'SS10', 'TH10', 'TW10', 'VN20']
    
    all_results = {}
    
    for entity in entities:
        try:
            results = run_prophet_forecast(entity)
            if results:
                all_results[entity] = results
                # Create backtest-only visualization (main focus)
                create_backtest_visualization(entity, results)
                # Create summary visualization
                create_forecast_summary_visualization(entity, results)
        except Exception as e:
            print(f"Error processing {entity}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print_summary(all_results)
    
    print(f"\n✅ Prophet forecasting complete!")
    print(f"   Backtest visualizations saved to: {OUTPUT_DIR}")
    
    return all_results


if __name__ == "__main__":
    results = main()
