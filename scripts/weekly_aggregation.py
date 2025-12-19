"""
📊 Weekly Aggregation by Entity

Create feature-rich weekly dataset where each row = 1 week for 1 entity.
Week numbering: Jan 1-7 = Week 1, Jan 8-14 = Week 2, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Load clean data
print("Loading data...")
df = pd.read_csv(os.path.join(base_dir, 'processed_data/clean_transactions.csv'))
df['Pstng Date'] = pd.to_datetime(df['Pstng Date'], format='mixed')
print(f"Total transactions: {len(df):,}")
print(f"Date range: {df['Pstng Date'].min()} to {df['Pstng Date'].max()}")
print(f"Entities: {df['Name'].unique()}")

# =============================================================================
# 1. Create Fixed Week Numbers (Jan 1-7 = Week 1)
# =============================================================================
year_start = datetime(2025, 1, 1)

# Calculate week number from Jan 1 (Week 1 = Jan 1-7, Week 2 = Jan 8-14, etc.)
df['Days_From_Start'] = (df['Pstng Date'] - year_start).dt.days
df['Week_Num'] = (df['Days_From_Start'] // 7) + 1

# Calculate week start and end dates
df['Week_Start'] = year_start + pd.to_timedelta((df['Week_Num'] - 1) * 7, unit='D')
df['Week_End'] = df['Week_Start'] + timedelta(days=6)

print(f"\nWeek range: Week {df['Week_Num'].min()} to Week {df['Week_Num'].max()}")

# =============================================================================
# 2. Create Feature-Rich Weekly Dataset by Entity
# =============================================================================
entities = df['Name'].unique()
weeks = sorted(df['Week_Num'].unique())
categories = df['Category'].unique()

print(f"\nEntities: {entities}")
print(f"Weeks: {len(weeks)}")
print(f"Categories: {len(categories)}")

print("\nCreating weekly aggregations...")
weekly_data = []

for entity in entities:
    entity_df = df[df['Name'] == entity]
    
    for week in weeks:
        week = int(week)  # Convert numpy.int64 to Python int
        week_df = entity_df[entity_df['Week_Num'] == week]
        
        # Basic info
        row = {
            'Entity': entity,
            'Week_Num': week,
            'Week_Start': year_start + timedelta(days=(week-1)*7),
            'Week_End': year_start + timedelta(days=week*7-1),
        }
        
        # Overall metrics
        row['Total_Net'] = week_df['Amount in USD'].sum()
        row['Total_Inflow'] = week_df[week_df['Amount in USD'] > 0]['Amount in USD'].sum()
        row['Total_Outflow'] = week_df[week_df['Amount in USD'] < 0]['Amount in USD'].sum()
        row['Outflow_Abs'] = abs(row['Total_Outflow'])
        row['Transaction_Count'] = len(week_df)
        row['Inflow_Count'] = len(week_df[week_df['Amount in USD'] > 0])
        row['Outflow_Count'] = len(week_df[week_df['Amount in USD'] < 0])
        
        # Average transaction size
        row['Avg_Transaction'] = week_df['Amount in USD'].mean() if len(week_df) > 0 else 0
        row['Avg_Inflow'] = week_df[week_df['Amount in USD'] > 0]['Amount in USD'].mean() if row['Inflow_Count'] > 0 else 0
        row['Avg_Outflow'] = week_df[week_df['Amount in USD'] < 0]['Amount in USD'].mean() if row['Outflow_Count'] > 0 else 0
        
        # Max/Min transactions
        row['Max_Transaction'] = week_df['Amount in USD'].max() if len(week_df) > 0 else 0
        row['Min_Transaction'] = week_df['Amount in USD'].min() if len(week_df) > 0 else 0
        
        # Category breakdowns (net amount per category)
        for cat in categories:
            cat_df = week_df[week_df['Category'] == cat]
            # Clean category name for column
            cat_clean = cat.replace(' ', '_').replace('/', '_')
            row[f'Cat_{cat_clean}_Net'] = cat_df['Amount in USD'].sum()
            row[f'Cat_{cat_clean}_Count'] = len(cat_df)
        
        # PK breakdown (40 = Debit/Inflow, 50 = Credit/Outflow)
        row['PK40_Amount'] = week_df[week_df['PK'] == 40]['Amount in USD'].sum()
        row['PK50_Amount'] = week_df[week_df['PK'] == 50]['Amount in USD'].sum()
        row['PK40_Count'] = len(week_df[week_df['PK'] == 40])
        row['PK50_Count'] = len(week_df[week_df['PK'] == 50])
        
        weekly_data.append(row)

# Create DataFrame
weekly_df = pd.DataFrame(weekly_data)
print(f"Created: {len(weekly_df)} rows (entities x weeks)")

# Fill NaN with 0 (for weeks with no transactions)
weekly_df = weekly_df.fillna(0)

# Add time features
weekly_df['Month'] = weekly_df['Week_Start'].dt.month
weekly_df['Week_of_Month'] = ((weekly_df['Week_Start'].dt.day - 1) // 7) + 1
weekly_df['Is_Month_End'] = weekly_df['Week_Start'].dt.day > 21  # Last week of month
weekly_df['Quarter'] = weekly_df['Week_Start'].dt.quarter

print(f"\nDataset shape: {weekly_df.shape}")

# =============================================================================
# 3. Add Lag Features (Previous Weeks)
# =============================================================================
print("\nAdding lag features...")

# Sort by entity and week
weekly_df = weekly_df.sort_values(['Entity', 'Week_Num']).reset_index(drop=True)

# Add lag features per entity
for lag in [1, 2, 4]:  # Previous week, 2 weeks ago, 4 weeks ago
    weekly_df[f'Net_Lag{lag}'] = weekly_df.groupby('Entity')['Total_Net'].shift(lag)
    weekly_df[f'Inflow_Lag{lag}'] = weekly_df.groupby('Entity')['Total_Inflow'].shift(lag)
    weekly_df[f'Outflow_Lag{lag}'] = weekly_df.groupby('Entity')['Total_Outflow'].shift(lag)

# Rolling averages (4-week moving average)
weekly_df['Net_Rolling4_Mean'] = weekly_df.groupby('Entity')['Total_Net'].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)
weekly_df['Net_Rolling4_Std'] = weekly_df.groupby('Entity')['Total_Net'].transform(
    lambda x: x.rolling(window=4, min_periods=1).std()
)

# Cumulative sum per entity
weekly_df['Cumulative_Net'] = weekly_df.groupby('Entity')['Total_Net'].cumsum()

# =============================================================================
# 4. Save the Dataset
# =============================================================================
output_dir = os.path.join(base_dir, 'processed_data')

# Save full dataset
weekly_df.to_csv(os.path.join(output_dir, 'weekly_entity_features.csv'), index=False)
print(f"\n✅ Saved: weekly_entity_features.csv")
print(f"   Rows: {len(weekly_df)} (44 weeks x 8 entities = 352 rows)")
print(f"   Columns: {len(weekly_df.columns)}")

# Also save separate files per entity for per-entity modeling
for entity in entities:
    entity_weekly = weekly_df[weekly_df['Entity'] == entity].copy()
    entity_weekly.to_csv(os.path.join(output_dir, f'weekly_{entity}.csv'), index=False)
    print(f"✅ Saved: weekly_{entity}.csv ({len(entity_weekly)} rows)")

# =============================================================================
# 5. Data Summary
# =============================================================================
print("\n" + "="*60)
print("📊 Entity Summary:")
print("="*60)

entity_summary = weekly_df.groupby('Entity').agg(
    Total_Net=('Total_Net', 'sum'),
    Total_Inflow=('Total_Inflow', 'sum'),
    Total_Outflow=('Total_Outflow', 'sum'),
    Weeks_Active=('Transaction_Count', lambda x: (x > 0).sum()),
    Total_Transactions=('Transaction_Count', 'sum'),
    Avg_Weekly_Net=('Total_Net', 'mean'),
    Std_Weekly_Net=('Total_Net', 'std')
).round(2)

print(entity_summary)

print("\n✅ Weekly aggregation complete!")
print(f"   Main file: {os.path.join(output_dir, 'weekly_entity_features.csv')}")
