"""
Fetch GDELT Data from Google BigQuery

This script fetches GDELT event data from Google BigQuery for India-related events
and processes it to compute unrest index components.

Usage: 
    python fetch_gdelt_data.py --start_date 2022-01-01 --end_date 2023-12-31 --output gdelt_india.csv
"""

import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
except ImportError:
    print("Error: Please install required packages:")
    print("  pip install google-cloud-bigquery google-auth")
    exit(1)

def normalize_series(series):
    """Min-max normalization"""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)

def fetch_gdelt_events(credentials_path, start_date, end_date, country_code='IN'):
    """
    Fetch GDELT events from BigQuery for a specific country and date range.
    
    Args:
        credentials_path: Path to Google Cloud service account JSON key
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        country_code: Country code (default: 'IN' for India)
    
    Returns:
        DataFrame with GDELT events
    """
    print(f"Fetching GDELT data from {start_date} to {end_date} for country: {country_code}")
    
    # Load credentials
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # Create BigQuery client
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    
    # Convert dates to YYYYMMDD format for GDELT
    start_date_int = int(start_date.replace('-', ''))
    end_date_int = int(end_date.replace('-', ''))
    
    # GDELT BigQuery query
    # Focus on events related to India (Actor1CountryCode or Actor2CountryCode = IN)
    query = f"""
    SELECT 
        SQLDATE,
        Actor1CountryCode,
        Actor2CountryCode,
        EventCode,
        EventRootCode,
        QuadClass,
        GoldsteinScale,
        NumMentions,
        NumSources,
        NumArticles,
        AvgTone,
        Actor1Geo_CountryCode,
        Actor2Geo_CountryCode,
        ActionGeo_CountryCode
    FROM 
        `gdelt-bq.gdeltv2.events`
    WHERE 
        SQLDATE >= {start_date_int}
        AND SQLDATE <= {end_date_int}
        AND (
            Actor1CountryCode = '{country_code}'
            OR Actor2CountryCode = '{country_code}'
            OR ActionGeo_CountryCode = '{country_code}'
        )
    ORDER BY 
        SQLDATE
    """
    
    print("Executing BigQuery query...")
    print(f"  → Query: Fetching events for {country_code} from {start_date_int} to {end_date_int}")
    
    # Execute query
    query_job = client.query(query)
    
    # Get results
    print("  → Downloading results...")
    df = query_job.to_dataframe()
    
    print(f" Fetched {len(df)} events")
    
    return df

def process_gdelt_data(df):
    """
    Process GDELT data to compute unrest index components.
    
    Args:
        df: Raw GDELT DataFrame
    
    Returns:
        Processed DataFrame with daily aggregated unrest components
    """
    print("\nProcessing GDELT data...")
    
    # Convert SQLDATE to datetime
    df['Event_Date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
    
    # Calculate unrest components
    print("  → Calculating unrest components...")
    
    # 1. Protest intensity (based on event codes)
    # Event codes 14* are protest-related
    df['is_protest'] = df['EventRootCode'].astype(str).str.startswith('14').astype(int)
    df['protest_intensity'] = df['is_protest'] * df['NumMentions']
    
    # 2. Violence intensity (negative Goldstein scale indicates conflict)
    df['violence_intensity'] = -df['GoldsteinScale'].clip(upper=0)
    
    # 3. Conflict score (QuadClass 3 and 4 are verbal/material conflict)
    df['is_conflict'] = df['QuadClass'].isin([3, 4]).astype(int)
    df['conflict_score'] = df['is_conflict'] * df['NumMentions'] * abs(df['GoldsteinScale'])
    
    # 4. Scaled tone (negative tone indicates negative sentiment)
    df['scaled_tone'] = -df['AvgTone'] * df['NumArticles']
    
    # 5. Total sources (media coverage)
    df['Total_Sources'] = df['NumSources']
    
    # Aggregate by date
    print("  → Aggregating by date...")
    df_daily = df.groupby('Event_Date').agg({
        'protest_intensity': 'sum',
        'violence_intensity': 'sum',
        'conflict_score': 'sum',
        'scaled_tone': 'sum',
        'Total_Sources': 'sum',
        'NumMentions': 'sum',
        'NumArticles': 'sum'
    }).reset_index()
    
    print(f" Aggregated to {len(df_daily)} daily records")
    
    return df_daily

def compute_unrest_index(df, window, label):
    """
    Compute unrest index for given window.
    
    Args:
        df: DataFrame with unrest components
        window: Rolling window size in days
        label: Label for the index (e.g., '7d', '30d', 'yearly')
    
    Returns:
        Series with unrest index
    """
    # Set Event_Date as index for rolling calculations
    df_indexed = df.set_index('Event_Date')
    
    # Rolling sum of components
    rolled = df_indexed[['protest_intensity', 'violence_intensity',
                         'conflict_score', 'scaled_tone', 'Total_Sources']].rolling(
        window=window, min_periods=1
    ).sum()
    
    # Normalize each component
    rolled['protests_norm'] = normalize_series(rolled['protest_intensity'])
    rolled['violence_norm'] = normalize_series(rolled['violence_intensity'])
    rolled['conflict_norm'] = normalize_series(rolled['conflict_score'])
    rolled['tone_norm'] = normalize_series(rolled['scaled_tone'])
    
    # Equal-weight composite unrest index
    unrest_index = (
        0.25 * rolled['protests_norm'] +
        0.25 * rolled['violence_norm'] +
        0.25 * rolled['conflict_norm'] +
        0.25 * rolled['tone_norm']
    )
    
    return unrest_index

def main():
    parser = argparse.ArgumentParser(description='Fetch and process GDELT data from BigQuery')
    parser.add_argument('--credentials', type=str, 
                       default='../gdeltplaypal-be8da892c655.json',
                       help='Path to Google Cloud service account JSON key')
    parser.add_argument('--start_date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='gdelt_india_processed.csv',
                       help='Output CSV file')
    parser.add_argument('--country', type=str, default='IN',
                       help='Country code (default: IN for India)')
    parser.add_argument('--compute_indices', action='store_true',
                       help='Compute rolling unrest indices (7d, 30d, quarterly, yearly)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GDELT DATA FETCHER - Google BigQuery")
    print("="*80)
    
    # Check credentials file exists
    if not os.path.exists(args.credentials):
        print(f" Credentials file not found: {args.credentials}")
        print("  Please provide a valid Google Cloud service account JSON key")
        return
    
    # Fetch data
    try:
        df_raw = fetch_gdelt_events(
            args.credentials,
            args.start_date,
            args.end_date,
            args.country
        )
    except Exception as e:
        print(f" Error fetching data: {e}")
        return
    
    # Process data
    df_processed = process_gdelt_data(df_raw)
    
    # Compute rolling indices if requested
    if args.compute_indices:
        print("\nComputing rolling unrest indices...")
        
        df_processed = df_processed.set_index('Event_Date')
        
        # Compute indices for different time windows
        df_processed['unrest_index_7d'] = compute_unrest_index(
            df_processed.reset_index(), window=7, label='7d'
        )
        df_processed['unrest_index_30d'] = compute_unrest_index(
            df_processed.reset_index(), window=30, label='30d'
        )
        df_processed['unrest_index_quarterly'] = compute_unrest_index(
            df_processed.reset_index(), window=90, label='quarterly'
        )
        df_processed['unrest_index_yearly'] = compute_unrest_index(
            df_processed.reset_index(), window=365, label='yearly'
        )
        
        df_processed = df_processed.reset_index()
        
        print(" Computed rolling indices: 7d, 30d, quarterly, yearly")
    
    # Save
    df_processed.to_csv(args.output, index=False)
    print(f"\n Saved processed data to: {args.output}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Date range:        {df_processed['Event_Date'].min()} to {df_processed['Event_Date'].max()}")
    print(f"Total days:        {len(df_processed)}")
    print(f"Total events:      {df_raw['NumMentions'].sum():,.0f} mentions")
    print(f"Total articles:    {df_raw['NumArticles'].sum():,.0f}")
    print(f"\nColumns in output:")
    for col in df_processed.columns:
        print(f"  - {col}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
