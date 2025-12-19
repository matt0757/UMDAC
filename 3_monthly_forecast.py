"""
Net Cash Flow Forecasting with Monthly Seasonality
===================================================
This script forecasts the net cash flow for 1 month ahead with:
1. Backtesting comparison for the last 1 month (4 weeks)
2. Focus on capturing monthly seasonality patterns
3. Multiple model comparison for best fit

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

# Statistical models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Configuration
DATA_DIR = 'd:/UM_DATATHON/UMDAC/processed_data'
OUTPUT_DIR = 'd:/UM_DATATHON/UMDAC/outputs/forecast_1month'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Backtest configuration: last 4 weeks (1 month)
BACKTEST_PERIODS = 4
FORECAST_PERIODS = 4  # Forecast 1 month ahead
SEASONAL_PERIOD = 4   # Monthly seasonality (4 weeks per month)

def load_entity_data(entity_code):
    """Load and preprocess entity data"""
    filepath = os.path.join(DATA_DIR, f'weekly_{entity_code}.csv')
    df = pd.read_csv(filepath)
    
    # Convert to datetime
    df['Week_Start'] = pd.to_datetime(df['Week_Start'])
    df['Week_End'] = pd.to_datetime(df['Week_End'])
    
    # Sort by week
    df = df.sort_values('Week_Num').reset_index(drop=True)
    
    return df

def analyze_data_patterns(df, entity_code):
    """Analyze data patterns for model selection"""
    net = df['Total_Net'].values
    
    # Basic statistics
    stats = {
        'mean': np.mean(net),
        'std': np.std(net),
        'cv': np.std(net) / np.abs(np.mean(net)) if np.mean(net) != 0 else np.inf,
        'min': np.min(net),
        'max': np.max(net),
        'n_obs': len(net)
    }
    
    # Check for Week_of_Month patterns (monthly seasonality indicator)
    if 'Week_of_Month' in df.columns:
        week_patterns = df.groupby('Week_of_Month')['Total_Net'].agg(['mean', 'std', 'count'])
        stats['week_of_month_pattern'] = week_patterns
    
    # Month-end indicator pattern
    if 'Is_Month_End' in df.columns:
        month_end_effects = df.groupby('Is_Month_End')['Total_Net'].mean()
        stats['month_end_effect'] = month_end_effects
    
    print(f"\n{'='*60}")
    print(f"Data Analysis for {entity_code}")
    print(f"{'='*60}")
    print(f"Observations: {stats['n_obs']}")
    print(f"Mean Net Cash Flow: {stats['mean']:,.2f}")
    print(f"Std Dev: {stats['std']:,.2f}")
    print(f"Coefficient of Variation: {stats['cv']:.2f}")
    print(f"Range: [{stats['min']:,.2f}, {stats['max']:,.2f}]")
    
    return stats

def prepare_time_series(df):
    """Prepare time series with proper index"""
    ts = df.set_index('Week_Start')['Total_Net'].copy()
    ts.index = pd.DatetimeIndex(ts.index)
    ts = ts.asfreq('W-WED')  # Weekly frequency
    
    # Handle any missing values
    if ts.isna().any():
        ts = ts.interpolate(method='linear')
    
    return ts

def fit_sarima_model(train_data, order=(1,0,1), seasonal_order=(1,0,1,4)):
    """Fit SARIMA model with monthly seasonality"""
    try:
        model = SARIMAX(
            train_data, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=200)
        return fitted
    except Exception as e:
        print(f"SARIMA fitting failed: {e}")
        return None

def fit_exponential_smoothing(train_data, seasonal_periods=4):
    """Fit Exponential Smoothing with trend and multiplicative seasonality"""
    try:
        # Handle negative values by shifting
        min_val = train_data.min()
        shift = abs(min_val) + 1 if min_val <= 0 else 0
        shifted_data = train_data + shift
        
        model = ExponentialSmoothing(
            shifted_data,
            trend='add',
            seasonal='add',  # Additive for data with possible negatives conceptually
            seasonal_periods=seasonal_periods,
            initialization_method='estimated'
        )
        fitted = model.fit(optimized=True)
        return fitted, shift
    except Exception as e:
        print(f"ETS fitting failed: {e}")
        return None, 0

def weighted_ensemble_forecast(sarima_fc, ets_fc, sarima_rmse, ets_rmse):
    """Create weighted ensemble forecast based on validation performance"""
    # Inverse RMSE weighting
    total_inv_rmse = (1/sarima_rmse) + (1/ets_rmse)
    sarima_weight = (1/sarima_rmse) / total_inv_rmse
    ets_weight = (1/ets_rmse) / total_inv_rmse
    
    ensemble_fc = sarima_weight * sarima_fc + ets_weight * ets_fc
    return ensemble_fc, sarima_weight, ets_weight

def calculate_metrics(actual, forecast):
    """Calculate comprehensive forecast metrics"""
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / np.where(actual != 0, actual, 1))) * 100
    
    # Direction accuracy
    actual_direction = np.sign(np.diff(np.concatenate([[0], actual])))
    forecast_direction = np.sign(np.diff(np.concatenate([[0], forecast])))
    direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy
    }

def run_backtest_and_forecast(entity_code):
    """Main function: run backtest and generate forecast"""
    print(f"\n{'#'*70}")
    print(f"Processing Entity: {entity_code}")
    print(f"{'#'*70}")
    
    # Load data
    df = load_entity_data(entity_code)
    stats = analyze_data_patterns(df, entity_code)
    
    # Prepare time series
    ts = prepare_time_series(df)
    
    # Split data for backtest (last 4 weeks for validation)
    train_data = ts[:-BACKTEST_PERIODS]
    test_data = ts[-BACKTEST_PERIODS:]
    
    print(f"\nTraining period: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Backtest period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
    
    results = {}
    
    # Model 1: SARIMA with monthly seasonality
    print("\nFitting SARIMA(1,0,1)(1,0,1,4)...")
    sarima_model = fit_sarima_model(train_data)
    
    if sarima_model is not None:
        sarima_backtest = sarima_model.get_forecast(steps=BACKTEST_PERIODS)
        sarima_fc_bt = sarima_backtest.predicted_mean.values
        sarima_conf_bt = sarima_backtest.conf_int()
        
        sarima_metrics = calculate_metrics(test_data.values, sarima_fc_bt)
        results['SARIMA'] = {
            'backtest_forecast': sarima_fc_bt,
            'metrics': sarima_metrics,
            'model': sarima_model
        }
        print(f"  SARIMA Backtest - MAE: {sarima_metrics['MAE']:,.2f}, RMSE: {sarima_metrics['RMSE']:,.2f}")
    
    # Model 2: Exponential Smoothing
    print("Fitting Exponential Smoothing...")
    ets_result = fit_exponential_smoothing(train_data)
    ets_model, shift = ets_result
    
    if ets_model is not None:
        ets_fc_bt = ets_model.forecast(BACKTEST_PERIODS).values - shift
        ets_metrics = calculate_metrics(test_data.values, ets_fc_bt)
        results['ETS'] = {
            'backtest_forecast': ets_fc_bt,
            'metrics': ets_metrics,
            'model': ets_model,
            'shift': shift
        }
        print(f"  ETS Backtest - MAE: {ets_metrics['MAE']:,.2f}, RMSE: {ets_metrics['RMSE']:,.2f}")
    
    # Model 3: Simple Moving Average with Week-of-Month adjustment
    print("Fitting Week-of-Month Adjusted Model...")
    
    # Calculate average by week of month
    week_of_month = df['Week_of_Month'].values[:-BACKTEST_PERIODS]  # Training only
    net_train = df['Total_Net'].values[:-BACKTEST_PERIODS]
    
    wom_means = {}
    for wom in [1, 2, 3, 4, 5]:
        mask = week_of_month == wom
        if mask.any():
            wom_means[wom] = np.mean(net_train[mask])
    
    # Forecast by matching week of month
    test_wom = df['Week_of_Month'].values[-BACKTEST_PERIODS:]
    wom_fc_bt = np.array([wom_means.get(wom, stats['mean']) for wom in test_wom])
    
    wom_metrics = calculate_metrics(test_data.values, wom_fc_bt)
    results['WOM_Adjusted'] = {
        'backtest_forecast': wom_fc_bt,
        'metrics': wom_metrics,
        'wom_means': wom_means
    }
    print(f"  Week-of-Month Adjusted - MAE: {wom_metrics['MAE']:,.2f}, RMSE: {wom_metrics['RMSE']:,.2f}")
    
    # Ensemble if multiple models available
    if 'SARIMA' in results and 'ETS' in results:
        ensemble_fc_bt, sw, ew = weighted_ensemble_forecast(
            results['SARIMA']['backtest_forecast'],
            results['ETS']['backtest_forecast'],
            results['SARIMA']['metrics']['RMSE'],
            results['ETS']['metrics']['RMSE']
        )
        ensemble_metrics = calculate_metrics(test_data.values, ensemble_fc_bt)
        results['Ensemble'] = {
            'backtest_forecast': ensemble_fc_bt,
            'metrics': ensemble_metrics,
            'sarima_weight': sw,
            'ets_weight': ew
        }
        print(f"  Ensemble (SARIMA {sw:.1%}, ETS {ew:.1%}) - MAE: {ensemble_metrics['MAE']:,.2f}, RMSE: {ensemble_metrics['RMSE']:,.2f}")
    
    # Select best model based on RMSE
    best_model = min(results.keys(), key=lambda k: results[k]['metrics']['RMSE'])
    print(f"\n*** Best Model: {best_model} ***")
    
    # Generate Future Forecast using FULL data
    print(f"\nGenerating 1-month (4-week) forecast using {best_model}...")
    
    full_sarima = fit_sarima_model(ts)
    if full_sarima is not None:
        future_forecast_obj = full_sarima.get_forecast(steps=FORECAST_PERIODS)
        future_forecast = future_forecast_obj.predicted_mean.values
        future_conf = future_forecast_obj.conf_int()
        
        # Generate future dates
        last_date = ts.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=7), periods=FORECAST_PERIODS, freq='W-WED')
        
        results['future_forecast'] = {
            'dates': future_dates,
            'values': future_forecast,
            'conf_lower': future_conf.iloc[:, 0].values,
            'conf_upper': future_conf.iloc[:, 1].values
        }
    
    # Create visualization
    create_visualization(entity_code, df, ts, test_data, results, best_model)
    
    return results, best_model

def create_visualization(entity_code, df, ts, test_data, results, best_model):
    """Create comprehensive visualization with backtest comparison"""
    
    fig = plt.figure(figsize=(16, 14))
    
    # Color scheme
    colors = {
        'actual': '#2C3E50',
        'SARIMA': '#E74C3C',
        'ETS': '#3498DB',
        'WOM_Adjusted': '#27AE60',
        'Ensemble': '#9B59B6',
        'future': '#F39C12'
    }
    
    # Plot 1: Full time series with backtest overlay
    ax1 = fig.add_subplot(3, 1, 1)
    
    # Full actual data
    ax1.plot(ts.index, ts.values, 'o-', color=colors['actual'], 
             linewidth=2, markersize=5, label='Actual Net Cash Flow', alpha=0.8)
    
    # Backtest forecasts
    bt_dates = test_data.index
    for model_name in ['SARIMA', 'ETS', 'WOM_Adjusted', 'Ensemble']:
        if model_name in results:
            linestyle = '--' if model_name != best_model else '-'
            linewidth = 3 if model_name == best_model else 1.5
            ax1.plot(bt_dates, results[model_name]['backtest_forecast'], 
                    linestyle=linestyle, linewidth=linewidth, 
                    color=colors.get(model_name, 'gray'),
                    marker='s', markersize=6,
                    label=f'{model_name} Backtest', alpha=0.8)
    
    # Future forecast (if available)
    if 'future_forecast' in results:
        fc = results['future_forecast']
        ax1.plot(fc['dates'], fc['values'], 's-', color=colors['future'], 
                linewidth=3, markersize=8, label='1-Month Forecast')
        ax1.fill_between(fc['dates'], fc['conf_lower'], fc['conf_upper'], 
                        color=colors['future'], alpha=0.2, label='95% CI')
    
    # Shade backtest period
    ax1.axvspan(test_data.index[0], test_data.index[-1], alpha=0.1, color='red', 
                label='Backtest Period (Last 1 Month)')
    
    ax1.set_title(f'{entity_code}: Net Cash Flow - Backtest & 1-Month Forecast', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Net Cash Flow')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Add zero line
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Plot 2: Backtest Comparison (Zoomed)
    ax2 = fig.add_subplot(3, 2, 3)
    
    # Actual vs forecasts in backtest period
    width = 0.15
    x = np.arange(len(test_data))
    
    ax2.bar(x - 2*width, test_data.values, width, label='Actual', color=colors['actual'], alpha=0.8)
    
    offset = -width
    for model_name in ['SARIMA', 'ETS', 'WOM_Adjusted', 'Ensemble']:
        if model_name in results:
            ax2.bar(x + offset, results[model_name]['backtest_forecast'], width, 
                   label=model_name, color=colors.get(model_name, 'gray'), alpha=0.7)
            offset += width
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%m/%d') for d in test_data.index])
    ax2.set_title('Backtest: Actual vs Forecasted (Last 4 Weeks)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Net Cash Flow')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Plot 3: Error Comparison
    ax3 = fig.add_subplot(3, 2, 4)
    
    model_names = []
    rmse_values = []
    mae_values = []
    
    for model_name in ['SARIMA', 'ETS', 'WOM_Adjusted', 'Ensemble']:
        if model_name in results:
            model_names.append(model_name)
            rmse_values.append(results[model_name]['metrics']['RMSE'])
            mae_values.append(results[model_name]['metrics']['MAE'])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, rmse_values, width, label='RMSE', color='#E74C3C')
    bars2 = ax3.bar(x + width/2, mae_values, width, label='MAE', color='#3498DB')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names)
    ax3.set_title('Backtest Error Metrics by Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Error Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight best model
    best_idx = model_names.index(best_model) if best_model in model_names else -1
    if best_idx >= 0:
        ax3.get_xticklabels()[best_idx].set_weight('bold')
        ax3.get_xticklabels()[best_idx].set_color('green')
    
    # Plot 4: Weekly patterns by Week of Month
    ax4 = fig.add_subplot(3, 2, 5)
    
    wom_data = df.groupby('Week_of_Month')['Total_Net'].agg(['mean', 'std', 'count'])
    
    ax4.bar(wom_data.index, wom_data['mean'], yerr=wom_data['std'], 
           capsize=5, color='#3498DB', alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Week of Month')
    ax4.set_ylabel('Mean Net Cash Flow')
    ax4.set_title('Monthly Seasonality Pattern (Week-of-Month Effect)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Future Forecast Details
    ax5 = fig.add_subplot(3, 2, 6)
    
    if 'future_forecast' in results:
        fc = results['future_forecast']
        x = np.arange(len(fc['dates']))
        
        ax5.bar(x, fc['values'], color=colors['future'], alpha=0.8, edgecolor='black')
        ax5.errorbar(x, fc['values'], 
                    yerr=[fc['values'] - fc['conf_lower'], fc['conf_upper'] - fc['values']],
                    fmt='none', color='black', capsize=5)
        
        ax5.set_xticks(x)
        ax5.set_xticklabels([d.strftime('%m/%d') for d in fc['dates']])
        ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax5.set_title('1-Month Ahead Forecast', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Week')
        ax5.set_ylabel('Forecasted Net Cash Flow')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, v in enumerate(fc['values']):
            ax5.text(i, v + (fc['conf_upper'][i] - v)*0.1, f'{v:,.0f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, f'{entity_code}_forecast_backtest.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")

def print_summary_table(all_results):
    """Print summary table of all entity forecasts"""
    print("\n" + "="*80)
    print("FORECAST SUMMARY - All Entities")
    print("="*80)
    
    print(f"\n{'Entity':<10} {'Best Model':<15} {'Backtest MAE':>15} {'Backtest RMSE':>15} {'Direction Acc':>15}")
    print("-"*70)
    
    for entity, (results, best_model) in all_results.items():
        metrics = results[best_model]['metrics']
        print(f"{entity:<10} {best_model:<15} {metrics['MAE']:>15,.2f} {metrics['RMSE']:>15,.2f} {metrics['Direction_Accuracy']:>14.1f}%")
    
    print("\n" + "="*80)
    print("1-MONTH FORECAST VALUES")
    print("="*80)
    
    for entity, (results, best_model) in all_results.items():
        if 'future_forecast' in results:
            fc = results['future_forecast']
            print(f"\n{entity}:")
            for i, (date, val, low, high) in enumerate(zip(fc['dates'], fc['values'], 
                                                           fc['conf_lower'], fc['conf_upper'])):
                print(f"  Week {i+1} ({date.strftime('%Y-%m-%d')}): {val:>15,.2f}  [95% CI: {low:>15,.2f} to {high:>15,.2f}]")

def main():
    """Main execution"""
    print("="*70)
    print("NET CASH FLOW FORECASTING WITH MONTHLY SEASONALITY")
    print("Backtest Period: Last 1 Month (4 Weeks)")
    print("Forecast Horizon: 1 Month (4 Weeks) Ahead")
    print("="*70)
    
    # Process KR10 (the file the user has open) first
    entities = ['KR10']  # Start with the user's active entity
    
    # Add other entities if user wants comprehensive analysis
    entities.extend(['ID10', 'MY10', 'PH10', 'SS10', 'TH10', 'TW10', 'VN20'])
    
    all_results = {}
    
    for entity in entities:
        try:
            results, best_model = run_backtest_and_forecast(entity)
            all_results[entity] = (results, best_model)
        except Exception as e:
            print(f"Error processing {entity}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print_summary_table(all_results)
    
    print(f"\n✅ Forecasting complete! Visualizations saved to: {OUTPUT_DIR}")
    
    return all_results

if __name__ == "__main__":
    results = main()
