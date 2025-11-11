"""
Compute Normalization Statistics for Conditioning Variables

This script fetches historical market data and computes the mean and std
that should be used to normalize new data to match the training distribution.
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import torch

# Fetch historical period from 2019 onwards
start_date = datetime(2019, 1, 1).date()
end_date = datetime.now().date()

print("="*80)
print("COMPUTING NORMALIZATION STATISTICS FOR CONDITIONING VARIABLES")
print("="*80)
print(f"\nFetching historical data from {start_date} to {end_date}...")
print("(Using data from 2019 to present)")

# Fetch market data
print("  → Fetching market data...")

# Helper function to safely extract Close column
def get_close_series(data):
    """Extract Close column from yfinance data, handling any format"""
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-level columns: get first Close value
        close_col = [c for c in data.columns if c[0] == 'Close']
        return data[close_col[0]] if close_col else pd.Series(dtype=float)
    else:
        return data['Close'] if 'Close' in data.columns else pd.Series(dtype=float)

# Fetch India VIX via yfinance
try:
    vix_raw = yf.download('^INDIAVIX', start=start_date, end=end_date, progress=False)
    vix_data = pd.DataFrame({'india_vix': get_close_series(vix_raw)})
    vix_data['india_vix'] = pd.to_numeric(vix_data['india_vix'], errors='coerce')
    vix_data = vix_data.dropna()
    print(f"     India VIX (^INDIAVIX): {len(vix_data)} days")
except Exception as e:
    print(f"     India VIX failed: {e}")
    exit(1)

usdinr_raw = yf.download('USDINR=X', start=start_date, end=end_date, progress=False)
usdinr_data = pd.DataFrame({'usdinr': get_close_series(usdinr_raw)})

crude_raw = yf.download('BZ=F', start=start_date, end=end_date, progress=False)
crude_data = pd.DataFrame({'crude_oil': get_close_series(crude_raw)})

yield_raw = yf.download('^TNX', start=start_date, end=end_date, progress=False)
yield_data = pd.DataFrame({'us_10y_yield': get_close_series(yield_raw)})

print(f"     USD/INR: {len(usdinr_data)} days")
print(f"     Crude Oil: {len(crude_data)} days")
print(f"     US 10Y Yield: {len(yield_data)} days")

# Combine
combined_data = pd.concat([vix_data, usdinr_data, crude_data, yield_data], axis=1).dropna()
print(f"\n Combined data: {len(combined_data)} days")

# Compute rolling features
print("\nComputing rolling features...")
combined_data['india_vix_7d_mean'] = combined_data['india_vix'].rolling(7, min_periods=1).mean()
combined_data['india_vix_30d_mean'] = combined_data['india_vix'].rolling(30, min_periods=1).mean()
combined_data['crude_oil_7d_mean'] = combined_data['crude_oil'].rolling(7, min_periods=1).mean()
combined_data['crude_oil_30d_mean'] = combined_data['crude_oil'].rolling(30, min_periods=1).mean()
combined_data['usdinr_quarterly_mean'] = combined_data['usdinr'].rolling(90, min_periods=1).mean()

print(" Rolling features computed")

# Extract features
print("\nExtracting features...")
features_list = []
for idx in combined_data.index:
    row = combined_data.loc[idx]
    features = np.array([
        float(row['crude_oil_30d_mean']),
        float(row['crude_oil_7d_mean']),
        0.0,  # unrest_index_yearly (placeholder)
        float(row['crude_oil']),
        float(row['usdinr_quarterly_mean']),
        float(row['india_vix_30d_mean']),
        float(row['india_vix_7d_mean']),
        float(row['us_10y_yield'])
    ])
    features_list.append(features)

features_array = np.array(features_list)
print(f" Extracted {len(features_array)} feature vectors")

# Compute statistics
print("\nComputing statistics...")
raw_mean = features_array.mean(axis=0)
raw_std = features_array.std(axis=0)

# Load actual unrest index statistics from GDELT data
gdelt_file = os.path.join(os.path.dirname(__file__), 'gdelt_india_unrest_index.csv')
if os.path.exists(gdelt_file):
    print(f"\n Loading unrest index from: {gdelt_file}")
    gdelt_df = pd.read_csv(gdelt_file)
    if 'unrest_index_yearly' in gdelt_df.columns:
        unrest_mean = gdelt_df['unrest_index_yearly'].mean()
        unrest_std = gdelt_df['unrest_index_yearly'].std()
        raw_mean[2] = unrest_mean
        raw_std[2] = unrest_std
        print(f"  Unrest index mean: {unrest_mean:.6f}")
        print(f"  Unrest index std: {unrest_std:.6f}")
        print(f"  Data range: {gdelt_df['Event_Date'].min()} to {gdelt_df['Event_Date'].max()}")
    else:
        print("   Warning: unrest_index_yearly column not found, using placeholder")
        raw_mean[2] = 100.0
        raw_std[2] = 10.0
else:
    print(f"\n Warning: GDELT file not found at {gdelt_file}")
    print("  Run fetch_and_compute_unrest_index.py first to get actual statistics")
    print("  Using placeholder values for now")
    raw_mean[2] = 100.0
    raw_std[2] = 10.0

var_names = [
    'crude_oil_30d_mean',
    'crude_oil_7d_mean',
    'unrest_index_yearly',
    'crude_oil',
    'usdinr_quarterly_mean',
    'india_vix_30d_mean',
    'india_vix_7d_mean',
    'us_10y_yield'
]

print("\nNormalization Statistics:")
print("-" * 60)
for i, name in enumerate(var_names):
    print(f"{name:30s}: mean={raw_mean[i]:10.4f}, std={raw_std[i]:10.4f}")

# Save to file
stats_dict = {
    'raw_mean': torch.tensor(raw_mean, dtype=torch.float32),
    'raw_std': torch.tensor(raw_std, dtype=torch.float32),
    'var_names': var_names,
    'date_range': f"{start_date} to {end_date}",
    'n_samples': len(features_array)
}

script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, 'conditioning_normalization_stats.pt')
torch.save(stats_dict, output_file)

print(f"\n Saved normalization statistics to: {output_file}")
print("\nThese statistics should be used to normalize new raw data:")
print("  normalized = (raw - raw_mean) / raw_std")
print("="*80)
