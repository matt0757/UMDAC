"""
Net Cash Flow Forecasting from Raw Transaction Data
====================================================
This script processes clean_transactions.csv directly to:
1. Aggregate daily net cash flow by entity
2. Identify and leverage monthly seasonality patterns
3. Backtest forecast accuracy for last 1 month
4. Generate 1-month ahead forecast with visualization

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
DATA_PATH = 'd:/UM_DATATHON/UMDAC/processed_data/clean_transactions.csv'
OUTPUT_DIR = 'd:/UM_DATATHON/UMDAC/outputs/forecast_raw'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Forecast settings
BACKTEST_WEEKS = 4  # Last 1 month for backtest
FORECAST_WEEKS = 4  # Forecast 1 month ahead

def load_and_process_raw_data():
    """Load raw transactions and aggregate to weekly net cash flow"""
    print("Loading raw transaction data...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Raw data shape: {df.shape}")
    print(f"Date range: {df['Pstng Date'].min()} to {df['Pstng Date'].max()}")
    print(f"Entities: {df['Name'].unique()}")
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Pstng Date'], format='%m/%d/%Y')
    
    # Create week number (ISO week)
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df['Week_Start'] = df['Date'] - pd.to_timedelta(df['Date'].dt.weekday, unit='D')
    
    # Extract time features
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Day_of_Month'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Is_Month_End'] = df['Day_of_Month'] >= 25
    df['Week_of_Month'] = ((df['Day_of_Month'] - 1) // 7) + 1
    
    return df

def create_weekly_aggregation(df, entity=None):
    """Aggregate transactions to weekly net cash flow"""
    if entity:
        df = df[df['Name'] == entity].copy()
    
    # Aggregate by week
    weekly = df.groupby(['Name', 'Week_Start']).agg({
        'Amount in USD': ['sum', 'count', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()],
        'Is_Month_End': 'max',
        'Week_of_Month': 'first',
        'Month': 'first'
    }).reset_index()
    
    # Flatten column names
    weekly.columns = ['Entity', 'Week_Start', 'Net_Cash_Flow', 'Transaction_Count', 
                      'Inflow_Count', 'Outflow_Count', 'Is_Month_End', 'Week_of_Month', 'Month']
    
    weekly = weekly.sort_values('Week_Start').reset_index(drop=True)
    
    return weekly

def analyze_seasonality(weekly_df, entity):
    """Analyze weekly and monthly seasonality patterns"""
    print(f"\n{'='*60}")
    print(f"Seasonality Analysis for {entity}")
    print(f"{'='*60}")
    
    data = weekly_df[weekly_df['Entity'] == entity].copy()
    
    # Week-of-Month pattern
    wom_pattern = data.groupby('Week_of_Month')['Net_Cash_Flow'].agg(['mean', 'std', 'count'])
    print("\nWeek-of-Month Pattern:")
    print(wom_pattern)
    
    # Month-end effect
    month_end_effect = data.groupby('Is_Month_End')['Net_Cash_Flow'].mean()
    print(f"\nMonth-end Effect (>=25th day):")
    print(f"  Non-Month-End weeks mean: {month_end_effect.get(False, 0):,.2f}")
    print(f"  Month-End weeks mean: {month_end_effect.get(True, 0):,.2f}")
    
    # Monthly pattern
    month_pattern = data.groupby('Month')['Net_Cash_Flow'].mean()
    print("\nMonthly Pattern:")
    print(month_pattern)
    
    return wom_pattern, month_end_effect

def create_features_for_forecast(data):
    """Create features for forecasting"""
    df = data.copy()
    
    # Lag features
    df['Net_Lag1'] = df['Net_Cash_Flow'].shift(1)
    df['Net_Lag2'] = df['Net_Cash_Flow'].shift(2)
    df['Net_Lag4'] = df['Net_Cash_Flow'].shift(4)  # Same week last month
    
    # Rolling statistics
    df['Net_Rolling4_Mean'] = df['Net_Cash_Flow'].rolling(4).mean()
    df['Net_Rolling4_Std'] = df['Net_Cash_Flow'].rolling(4).std()
    
    # Week-of-Month encoding
    df['WoM_1'] = (df['Week_of_Month'] == 1).astype(int)
    df['WoM_2'] = (df['Week_of_Month'] == 2).astype(int)
    df['WoM_3'] = (df['Week_of_Month'] == 3).astype(int)
    df['WoM_4'] = (df['Week_of_Month'] == 4).astype(int)
    df['WoM_5'] = (df['Week_of_Month'] == 5).astype(int)
    
    return df

def fit_sarima_model(train_series, order=(1,0,1), seasonal_order=(1,0,1,4)):
    """Fit SARIMA model with monthly seasonality"""
    try:
        model = SARIMAX(
            train_series, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=200)
        return fitted
    except Exception as e:
        print(f"  SARIMA fitting failed: {e}")
        return None

def fit_ets_model(train_series, seasonal_periods=4):
    """Fit Holt-Winters Exponential Smoothing"""
    try:
        # Handle negative values
        min_val = train_series.min()
        shift = abs(min_val) + 1 if min_val <= 0 else 0
        shifted_data = train_series + shift
        
        model = ExponentialSmoothing(
            shifted_data,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_periods,
            initialization_method='estimated'
        )
        fitted = model.fit(optimized=True)
        return fitted, shift
    except Exception as e:
        print(f"  ETS fitting failed: {e}")
        return None, 0

def week_of_month_model(train_df, test_wom):
    """Simple model using Week-of-Month averages"""
    wom_means = train_df.groupby('Week_of_Month')['Net_Cash_Flow'].mean().to_dict()
    overall_mean = train_df['Net_Cash_Flow'].mean()
    
    forecasts = [wom_means.get(wom, overall_mean) for wom in test_wom]
    return np.array(forecasts), wom_means

def hybrid_model(train_df, train_series, forecast_periods, future_wom):
    """
    Hybrid model combining:
    1. SARIMA for trend/autocorrelation
    2. Week-of-Month adjustments for seasonality
    """
    # Fit SARIMA
    sarima = fit_sarima_model(train_series)
    if sarima is None:
        return None
    
    # Get base forecast from SARIMA
    sarima_fc = sarima.get_forecast(steps=forecast_periods)
    base_forecast = sarima_fc.predicted_mean.values
    
    # Calculate Week-of-Month adjustments
    wom_effects = train_df.groupby('Week_of_Month')['Net_Cash_Flow'].mean()
    overall_mean = train_df['Net_Cash_Flow'].mean()
    wom_adjustments = (wom_effects - overall_mean).to_dict()
    
    # Apply adjustments
    adjusted_forecast = []
    for i, wom in enumerate(future_wom):
        adj = wom_adjustments.get(wom, 0)
        adjusted_forecast.append(base_forecast[i] + adj * 0.3)  # Blend factor
    
    return np.array(adjusted_forecast), sarima_fc.conf_int()

def calculate_metrics(actual, forecast):
    """Calculate forecast accuracy metrics"""
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    
    # Avoid division by zero in MAPE
    mape_mask = actual != 0
    if mape_mask.any():
        mape = np.mean(np.abs((actual[mape_mask] - forecast[mape_mask]) / actual[mape_mask])) * 100
    else:
        mape = np.nan
    
    # Direction accuracy
    actual_diff = np.diff(actual)
    forecast_diff = np.diff(forecast)
    if len(actual_diff) > 0:
        direction_acc = np.mean(np.sign(actual_diff) == np.sign(forecast_diff)) * 100
    else:
        direction_acc = np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Direction_Accuracy': direction_acc
    }

def run_forecast_for_entity(weekly_df, entity):
    """Run complete forecast pipeline for one entity"""
    print(f"\n{'#'*70}")
    print(f"Forecasting for Entity: {entity}")
    print(f"{'#'*70}")
    
    # Filter data for entity
    data = weekly_df[weekly_df['Entity'] == entity].copy().reset_index(drop=True)
    data = data.sort_values('Week_Start').reset_index(drop=True)
    
    if len(data) < 10:
        print(f"  Insufficient data ({len(data)} weeks). Skipping.")
        return None
    
    print(f"  Data: {len(data)} weeks from {data['Week_Start'].min()} to {data['Week_Start'].max()}")
    
    # Analyze seasonality
    wom_pattern, month_end_effect = analyze_seasonality(weekly_df, entity)
    
    # Create features
    data = create_features_for_forecast(data)
    
    # Split data for backtest
    train_data = data.iloc[:-BACKTEST_WEEKS].copy()
    test_data = data.iloc[-BACKTEST_WEEKS:].copy()
    
    train_series = train_data['Net_Cash_Flow']
    test_series = test_data['Net_Cash_Flow'].values
    test_wom = test_data['Week_of_Month'].values
    
    print(f"\n  Training period: {train_data['Week_Start'].iloc[0].strftime('%Y-%m-%d')} to {train_data['Week_Start'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Backtest period: {test_data['Week_Start'].iloc[0].strftime('%Y-%m-%d')} to {test_data['Week_Start'].iloc[-1].strftime('%Y-%m-%d')}")
    
    results = {}
    
    # Model 1: SARIMA
    print("\n  Fitting SARIMA(1,0,1)(1,0,1,4)...")
    sarima_model = fit_sarima_model(train_series)
    if sarima_model:
        sarima_fc = sarima_model.get_forecast(steps=BACKTEST_WEEKS)
        sarima_backtest = sarima_fc.predicted_mean.values
        sarima_metrics = calculate_metrics(test_series, sarima_backtest)
        results['SARIMA'] = {
            'backtest': sarima_backtest,
            'metrics': sarima_metrics,
            'model': sarima_model
        }
        print(f"    MAE: {sarima_metrics['MAE']:,.2f}, RMSE: {sarima_metrics['RMSE']:,.2f}")
    
    # Model 2: ETS
    print("  Fitting Exponential Smoothing...")
    ets_result = fit_ets_model(train_series)
    if ets_result[0] is not None:
        ets_model, shift = ets_result
        ets_backtest = ets_model.forecast(BACKTEST_WEEKS).values - shift
        ets_metrics = calculate_metrics(test_series, ets_backtest)
        results['ETS'] = {
            'backtest': ets_backtest,
            'metrics': ets_metrics,
            'model': ets_model,
            'shift': shift
        }
        print(f"    MAE: {ets_metrics['MAE']:,.2f}, RMSE: {ets_metrics['RMSE']:,.2f}")
    
    # Model 3: Week-of-Month
    print("  Fitting Week-of-Month Model...")
    wom_backtest, wom_means = week_of_month_model(train_data, test_wom)
    wom_metrics = calculate_metrics(test_series, wom_backtest)
    results['WoM'] = {
        'backtest': wom_backtest,
        'metrics': wom_metrics,
        'wom_means': wom_means
    }
    print(f"    MAE: {wom_metrics['MAE']:,.2f}, RMSE: {wom_metrics['RMSE']:,.2f}")
    
    # Model 4: Hybrid (SARIMA + WoM adjustment)
    print("  Fitting Hybrid Model...")
    hybrid_result = hybrid_model(train_data, train_series, BACKTEST_WEEKS, test_wom)
    if hybrid_result is not None:
        hybrid_backtest, _ = hybrid_result
        hybrid_metrics = calculate_metrics(test_series, hybrid_backtest)
        results['Hybrid'] = {
            'backtest': hybrid_backtest,
            'metrics': hybrid_metrics
        }
        print(f"    MAE: {hybrid_metrics['MAE']:,.2f}, RMSE: {hybrid_metrics['RMSE']:,.2f}")
    
    # Select best model
    if results:
        best_model = min(results.keys(), key=lambda k: results[k]['metrics']['RMSE'])
        print(f"\n  *** Best Model: {best_model} (RMSE: {results[best_model]['metrics']['RMSE']:,.2f}) ***")
    else:
        print("  No models fitted successfully.")
        return None
    
    # Generate future forecast using full data
    print(f"\n  Generating {FORECAST_WEEKS}-week ahead forecast...")
    full_series = data['Net_Cash_Flow']
    
    # Determine future Week-of-Month values
    last_date = data['Week_Start'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=7), periods=FORECAST_WEEKS, freq='W-MON')
    future_wom = [((d.day - 1) // 7) + 1 for d in future_dates]
    
    # Use best performing model approach for final forecast
    full_sarima = fit_sarima_model(full_series)
    if full_sarima:
        forecast_obj = full_sarima.get_forecast(steps=FORECAST_WEEKS)
        future_forecast = forecast_obj.predicted_mean.values
        future_conf = forecast_obj.conf_int()
        
        results['future'] = {
            'dates': future_dates,
            'values': future_forecast,
            'conf_lower': future_conf.iloc[:, 0].values,
            'conf_upper': future_conf.iloc[:, 1].values
        }
    
    # Store test data for visualization
    results['test_data'] = test_data
    results['train_data'] = train_data
    results['full_data'] = data
    results['best_model'] = best_model
    
    return results

def create_visualization(entity, results):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(16, 14))
    
    colors = {
        'actual': '#2C3E50',
        'SARIMA': '#E74C3C',
        'ETS': '#3498DB',
        'WoM': '#27AE60',
        'Hybrid': '#9B59B6',
        'future': '#F39C12'
    }
    
    full_data = results['full_data']
    test_data = results['test_data']
    best_model = results['best_model']
    
    # Plot 1: Full time series with backtest
    ax1 = fig.add_subplot(3, 1, 1)
    
    # Actual values
    ax1.plot(full_data['Week_Start'], full_data['Net_Cash_Flow'], 
             'o-', color=colors['actual'], linewidth=2, markersize=5,
             label='Actual Net Cash Flow', alpha=0.8)
    
    # Backtest forecasts
    bt_dates = test_data['Week_Start']
    for model_name in ['SARIMA', 'ETS', 'WoM', 'Hybrid']:
        if model_name in results:
            linestyle = '-' if model_name == best_model else '--'
            linewidth = 3 if model_name == best_model else 1.5
            ax1.plot(bt_dates, results[model_name]['backtest'],
                    linestyle=linestyle, linewidth=linewidth,
                    color=colors.get(model_name, 'gray'),
                    marker='s', markersize=6,
                    label=f'{model_name} Backtest', alpha=0.8)
    
    # Future forecast
    if 'future' in results:
        fc = results['future']
        ax1.plot(fc['dates'], fc['values'], 's-', color=colors['future'],
                linewidth=3, markersize=8, label='1-Month Forecast')
        ax1.fill_between(fc['dates'], fc['conf_lower'], fc['conf_upper'],
                        color=colors['future'], alpha=0.2)
    
    # Shade backtest period
    ax1.axvspan(test_data['Week_Start'].iloc[0], test_data['Week_Start'].iloc[-1],
               alpha=0.1, color='red', label='Backtest Period')
    
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_title(f'{entity}: Net Cash Flow - Backtest & 1-Month Forecast', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Net Cash Flow (USD)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Backtest comparison bars
    ax2 = fig.add_subplot(3, 2, 3)
    
    x = np.arange(BACKTEST_WEEKS)
    width = 0.15
    
    ax2.bar(x - 2*width, test_data['Net_Cash_Flow'].values, width,
           label='Actual', color=colors['actual'], alpha=0.8)
    
    offset = -width
    for model_name in ['SARIMA', 'ETS', 'WoM', 'Hybrid']:
        if model_name in results:
            ax2.bar(x + offset, results[model_name]['backtest'], width,
                   label=model_name, color=colors.get(model_name, 'gray'), alpha=0.7)
            offset += width
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%m/%d') for d in test_data['Week_Start']])
    ax2.set_title('Backtest: Actual vs Forecasted (Last 4 Weeks)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Net Cash Flow')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Plot 3: Error metrics
    ax3 = fig.add_subplot(3, 2, 4)
    
    model_names = []
    rmse_values = []
    mae_values = []
    
    for model_name in ['SARIMA', 'ETS', 'WoM', 'Hybrid']:
        if model_name in results:
            model_names.append(model_name)
            rmse_values.append(results[model_name]['metrics']['RMSE'])
            mae_values.append(results[model_name]['metrics']['MAE'])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax3.bar(x - width/2, rmse_values, width, label='RMSE', color='#E74C3C')
    ax3.bar(x + width/2, mae_values, width, label='MAE', color='#3498DB')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names)
    ax3.set_title('Backtest Error Metrics by Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Error Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight best
    if best_model in model_names:
        best_idx = model_names.index(best_model)
        ax3.get_xticklabels()[best_idx].set_weight('bold')
        ax3.get_xticklabels()[best_idx].set_color('green')
    
    # Plot 4: Week-of-Month pattern
    ax4 = fig.add_subplot(3, 2, 5)
    
    wom_data = full_data.groupby('Week_of_Month')['Net_Cash_Flow'].agg(['mean', 'std', 'count'])
    
    ax4.bar(wom_data.index, wom_data['mean'], yerr=wom_data['std'],
           capsize=5, color='#3498DB', alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Week of Month')
    ax4.set_ylabel('Mean Net Cash Flow')
    ax4.set_title('Monthly Seasonality Pattern', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Future forecast details
    ax5 = fig.add_subplot(3, 2, 6)
    
    if 'future' in results:
        fc = results['future']
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
            y_offset = (fc['conf_upper'][i] - v) * 0.1 if v >= 0 else -abs(fc['conf_lower'][i] - v) * 0.1
            ax5.text(i, v + y_offset, f'{v:,.0f}', ha='center', 
                    va='bottom' if v >= 0 else 'top', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, f'{entity}_forecast_from_raw.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  Visualization saved to: {output_path}")
    return output_path

def print_summary(all_results):
    """Print comprehensive summary"""
    print("\n" + "="*80)
    print("FORECAST SUMMARY")
    print("="*80)
    
    print(f"\n{'Entity':<10} {'Best Model':<12} {'Backtest MAE':>15} {'Backtest RMSE':>15} {'Direction':>12}")
    print("-"*70)
    
    for entity, results in all_results.items():
        if results:
            best = results['best_model']
            metrics = results[best]['metrics']
            print(f"{entity:<10} {best:<12} {metrics['MAE']:>15,.2f} {metrics['RMSE']:>15,.2f} {metrics['Direction_Accuracy']:>11.1f}%")
    
    print("\n" + "="*80)
    print("1-MONTH FORECAST VALUES")
    print("="*80)
    
    for entity, results in all_results.items():
        if results and 'future' in results:
            fc = results['future']
            print(f"\n{entity}:")
            total = 0
            for i, (date, val, low, high) in enumerate(zip(
                fc['dates'], fc['values'], fc['conf_lower'], fc['conf_upper'])):
                print(f"  Week {i+1} ({date.strftime('%Y-%m-%d')}): {val:>15,.2f}  [95% CI: {low:>15,.2f} to {high:>15,.2f}]")
                total += val
            print(f"  {'Monthly Total':>30}: {total:>15,.2f}")

def main():
    """Main execution"""
    print("="*70)
    print("NET CASH FLOW FORECASTING FROM RAW TRANSACTIONS")
    print("="*70)
    
    # Load and process data
    raw_df = load_and_process_raw_data()
    
    # Create weekly aggregation
    print("\nCreating weekly aggregations...")
    weekly_df = create_weekly_aggregation(raw_df)
    print(f"Weekly data shape: {weekly_df.shape}")
    
    # Get list of entities
    entities = weekly_df['Entity'].unique()
    print(f"Entities to process: {entities}")
    
    # Focus on KR10 first (user's active file was weekly_KR10.csv)
    priority_entities = ['KR10', 'ID10', 'MY10', 'PH10', 'SS10', 'TH10', 'TW10', 'VN20']
    
    all_results = {}
    
    for entity in priority_entities:
        results = run_forecast_for_entity(weekly_df, entity)
        if results:
            all_results[entity] = results
            create_visualization(entity, results)
    
    # Print summary
    print_summary(all_results)
    
    print(f"\n✅ Forecasting complete! Visualizations saved to: {OUTPUT_DIR}")
    
    return all_results

if __name__ == "__main__":
    results = main()
