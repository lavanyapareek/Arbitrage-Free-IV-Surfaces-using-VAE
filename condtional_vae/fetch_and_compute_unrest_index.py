"""
Fetch GDELT data and compute unrest index statistics
Based on the methodology from conditional_vars_extraction.ipynb
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
except ImportError:
    print("Error: Please install required packages:")
    print("  pip install google-cloud-bigquery google-auth")
    exit(1)

print("="*80)
print("FETCHING GDELT DATA AND COMPUTING UNREST INDEX STATISTICS")
print("="*80)

# Set up BigQuery client
script_dir = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(os.path.dirname(script_dir), 'gdeltplaypal-be8da892c655.json')
if not os.path.exists(credentials_path):
    print(f"Error: Credentials file not found: {credentials_path}")
    exit(1)

credentials = service_account.Credentials.from_service_account_file(
    credentials_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

bigquery_client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# SQL query for India
country = 'IND'
query = f"""
SELECT
    FORMAT_TIMESTAMP('%Y-%m-%d', PARSE_TIMESTAMP('%Y%m%d', CAST(SQLDATE AS STRING))) AS Event_Date,
    '{country}' AS Country,
    SUM(CASE WHEN EventCode = '141' THEN NumMentions ELSE 0 END) AS Demonstrations_Mentions,
    SUM(CASE WHEN EventCode = '142' THEN NumMentions ELSE 0 END) AS Hunger_Strikes_Mentions,
    SUM(CASE WHEN EventCode = '145' THEN NumMentions ELSE 0 END) AS Strikes_Boycotts_Mentions,
    SUM(CASE WHEN EventCode = '181' THEN NumMentions ELSE 0 END) AS Attacks_Mentions,
    SUM(CASE WHEN EventCode = '183' THEN NumMentions ELSE 0 END) AS Mass_Violence_Mentions,
    SUM(CASE WHEN EventCode = '190' THEN NumMentions ELSE 0 END) AS General_Coerce_Mentions,
    SUM(CASE WHEN EventCode = '191' THEN NumMentions ELSE 0 END) AS Seizure_Mentions,
    AVG(AvgTone) AS Average_Tone,
    AVG(GoldsteinScale) AS Average_GoldsteinScale,
    SUM(NumSources) AS Total_Sources
FROM
    `gdelt-bq.full.events` 
WHERE
    EventCode IN ('141', '142', '145', '181', '183', '190', '191')
    AND Actor1CountryCode = '{country}'
    AND CAST(SQLDATE AS INT64) >= 20150101
    AND NumSources >= 0
    AND NumMentions >= 0
GROUP BY
    Event_Date
ORDER BY
    Event_Date ASC;
"""

print(f"\nFetching GDELT data for {country} (2015 onwards)...")
print("This may take several minutes...")
print("Note: BigQuery may have data delays. Check gdelt-bq.full.events publication date.")

df_ind = None  # Initialize for fallback logic
use_fresh_data = False

try:
    # Configure query with appropriate timeout and settings
    job_config = bigquery.QueryJobConfig()
    job_config.use_query_cache = False  # Force fresh query
    
    query_job = bigquery_client.query(query, job_config=job_config)
    query_job.result()  # Wait for completion
    df_ind = query_job.to_dataframe()
    
    if len(df_ind) == 0:
        print(" WARNING: BigQuery returned empty results!")
        print("This may indicate BigQuery data is not available beyond 2020.")
        print("GDELT has known delays in BigQuery ingestion (data may lag 6+ months behind real-time).")
        df_ind = None
    else:
        # Convert date column to datetime
        df_ind['Event_Date'] = pd.to_datetime(df_ind['Event_Date'])
        max_date = df_ind['Event_Date'].max()
        print(f" Fetched {len(df_ind)} days of data")
        print(f"  Date range: {df_ind['Event_Date'].min().date()} to {max_date.date()}")
        use_fresh_data = True
            
except Exception as e:
    print(f" Error fetching data: {e}")
    print("\nTroubleshooting:")
    print("  1. Verify Google Cloud credentials are set up: gdeltplaypal-be8da892c655.json")
    print("  2. Check BigQuery quotas and billing")
    print("  3. GDELT in BigQuery may have data ingestion delays (up to 6 months)")
    print("  4. Falling back to existing data if available...")
    df_ind = None

# If fetch failed, use existing file
if not use_fresh_data:
    print("\nUsing existing gdelt_india_unrest_index.csv")
    if not os.path.exists('gdelt_india_unrest_index.csv'):
        print(" Error: No existing GDELT data file found!")
        exit(1)
    
    # Read existing CSV
    df_ind = pd.read_csv('gdelt_india_unrest_index.csv', index_col=0, parse_dates=True)
    print(f" Loaded {len(df_ind)} rows from existing file (data through {df_ind.index.max().date()})")
else:
    # Set index for newly fetched data
    df_ind.set_index('Event_Date', inplace=True)

# Process the data
if df_ind is None or len(df_ind) == 0:
    print(" Error: Could not load GDELT data!")
    exit(1)

print("\nProcessing data...")

# Only process newly fetched data (index not already set)
if 'Event_Date' in df_ind.columns:
    df_ind['Event_Date'] = pd.to_datetime(df_ind['Event_Date'])
    df_ind.set_index('Event_Date', inplace=True)

# Fill missing values
print("Missing values per column:")
print(df_ind.isnull().sum())
df_ind = df_ind.fillna(0)

# 1) Base weights
base_weights = {
    'Demonstrations_Mentions': 1.0,
    'Hunger_Strikes_Mentions': 1.5,
    'Strikes_Boycotts_Mentions': 2.0,
    'Attacks_Mentions': 6.0,
    'Mass_Violence_Mentions': 8.0,
    'General_Coerce_Mentions': 3.0,
    'Seizure_Mentions': 5.0
}

# 2) Compute base weighted sum
df_ind['Weighted_Base'] = 0.0
for col, w in base_weights.items():
    if col in df_ind.columns:
        df_ind['Weighted_Base'] += df_ind[col].fillna(0) * w

# 3) Goldstein multiplier
gold_col = 'Average_GoldsteinScale'
df_ind[gold_col] = df_ind[gold_col].fillna(0).astype(float)
df_ind['Goldstein_Factor'] = 1.0 + (-df_ind[gold_col] / 10.0)
df_ind['Goldstein_Factor'] = df_ind['Goldstein_Factor'].clip(lower=0.1, upper=2.0)

# 4) Source breadth multiplier
source_col = 'Total_Sources'
df_ind[source_col] = df_ind[source_col].fillna(0).astype(float)
avg_sources = df_ind[source_col].replace(0, np.nan).mean()
if np.isnan(avg_sources) or avg_sources <= 0:
    avg_sources = df_ind[source_col].mean() + 1.0

df_ind['Source_Factor'] = np.log1p(df_ind[source_col]) / np.log1p(avg_sources)
df_ind['Source_Factor'] = df_ind['Source_Factor'].clip(lower=0.25, upper=4.0)

# 5) Combined weighted unrest score
df_ind['Weighted_Unrest_Score'] = df_ind['Weighted_Base'] * df_ind['Goldstein_Factor'] * df_ind['Source_Factor']

# 6) RSUI-A style daily index
df_ind['Rolling_Global_Media'] = df_ind[source_col].rolling(window=30, min_periods=1).mean()
mean_x = df_ind['Weighted_Unrest_Score'].mean()
mean_z = df_ind[source_col].mean()
mean_z = mean_z if mean_z != 0 else 1.0

df_ind['RSUI_weighted_daily'] = (df_ind['Weighted_Unrest_Score'] / df_ind['Rolling_Global_Media']) * 100 * (mean_x / mean_z)

# Rebase to mean=100
if df_ind['RSUI_weighted_daily'].abs().sum() != 0:
    df_ind['RSUI_weighted_daily'] = df_ind['RSUI_weighted_daily'] / df_ind['RSUI_weighted_daily'].mean() * 100

# 7) Compute yearly rolling unrest index (365-day window)
print("\nComputing yearly rolling unrest index...")
df_ind['unrest_index_yearly'] = df_ind['RSUI_weighted_daily'].rolling(window=365, min_periods=1).mean()

# Compute statistics
print("\n" + "="*80)
print("UNREST INDEX STATISTICS (RAW, BEFORE Z-SCORE NORMALIZATION)")
print("="*80)

unrest_mean = df_ind['unrest_index_yearly'].mean()
unrest_std = df_ind['unrest_index_yearly'].std()
unrest_min = df_ind['unrest_index_yearly'].min()
unrest_max = df_ind['unrest_index_yearly'].max()

print(f"\nunrest_index_yearly (raw):")
print(f"  Mean:  {unrest_mean:.6f}")
print(f"  Std:   {unrest_std:.6f}")
print(f"  Min:   {unrest_min:.6f}")
print(f"  Max:   {unrest_max:.6f}")
print(f"  Count: {len(df_ind)}")

# Save the data
output_file = 'gdelt_india_unrest_index.csv'
df_ind[['RSUI_weighted_daily', 'unrest_index_yearly']].to_csv(output_file)
print(f"\n Saved unrest index data to: {output_file}")

print("\n" + "="*80)
print("USE THESE VALUES IN compute_normalization_stats.py:")
print("="*80)
print(f"raw_mean[2] = {unrest_mean:.6f}  # unrest_index_yearly mean")
print(f"raw_std[2] = {unrest_std:.6f}   # unrest_index_yearly std")
print("="*80)
