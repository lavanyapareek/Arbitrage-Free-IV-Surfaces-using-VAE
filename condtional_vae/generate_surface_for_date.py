"""
Generate IV Surface for a Specific Date using Conditional VAE

This script:
1. Fetches raw market data (India VIX, crude oil, USD/INR, US 10Y yield) for target date
2. Computes rolling features (7d, 30d, quarterly means)
3. Calculates unrest index from GDELT data (1 year buffer)
4. Normalizes features using training statistics
5. Generates IV surface using the trained CVAE model

Usage: python generate_surface_for_date.py --date 2023-10-15 --n_samples 100
"""

import os, sys, json, argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm import tqdm

try:
    import yfinance as yf
    from nsepy import get_history
except ImportError:
    print("Error: pip install yfinance nsepy")
    sys.exit(1)

from cvae_model import ConditionalVAE_SingleHeston
from heston_model_ql import HestonModelQL

# ============================================================================
# Argument Parsing
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, required=True, help='Target date (YYYY-MM-DD)')
parser.add_argument('--n_samples', type=int, default=500, help='Number of parameter samples to generate')
parser.add_argument('--output_dir', type=str, default='results_date', help='Output directory')
parser.add_argument('--spot', type=float, default=None, help='Spot price (if None, fetches from NSE)')
parser.add_argument('--r', type=float, default=0.067, help='Risk-free rate')
parser.add_argument('--q', type=float, default=0.0, help='Dividend yield')
parser.add_argument('--gdelt_file', type=str, default=None, help='Path to GDELT data CSV (optional)')
parser.add_argument('--fetch_gdelt', action='store_true', default=True, help='Automatically fetch GDELT data from BigQuery')
parser.add_argument('--gdelt_credentials', type=str, default='../gdeltplaypal-be8da892c655.json',
                   help='Path to Google Cloud service account JSON key')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*80)
print(f"CONDITIONAL VAE IV SURFACE GENERATOR - {args.date}")
print("="*80)

target_date = datetime.strptime(args.date, '%Y-%m-%d').date()

# ============================================================================
# Fetch Spot Price (if not provided)
# ============================================================================

if args.spot is None:
    print("\nFetching NIFTY spot price...")
    try:
        # Try yfinance first (more reliable)
        fetch_start_spot = target_date - timedelta(days=7)
        nifty_data = yf.download('^NSEI', start=fetch_start_spot, end=target_date + timedelta(days=1), progress=False)
        if len(nifty_data) > 0 and 'Close' in nifty_data.columns:
            args.spot = float(nifty_data['Close'].iloc[-1])
            actual_date = nifty_data.index[-1].date()
            if actual_date == target_date:
                print(f"  ✓ NIFTY 50 spot price: {args.spot:.2f}")
            else:
                print(f"  ⚠ Using nearest available date: {actual_date}")
                print(f"  ✓ NIFTY 50 spot price: {args.spot:.2f}")
        else:
            raise ValueError("No NIFTY data available from yfinance")
    except Exception as e:
        print(f"  ✗ Error fetching spot price: {e}")
        print(f"  → Using default spot price: 21000.0")
        args.spot = 21000.0
else:
    print(f"\nUsing provided spot price: {args.spot:.2f}")

# ============================================================================
# Load Model and Configuration
# ============================================================================

config_path = os.path.join(script_dir, 'config.json')
with open(config_path) as f:
    config = json.load(f)

model_path = os.path.join(script_dir, 'results', 'best_cvae_model.pt')
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

# Extract normalization statistics
param_mean = checkpoint['param_mean'].to(device)
param_std = checkpoint['param_std'].to(device)

# Load conditioning variable normalization stats (computed from historical raw data)
norm_stats_path = os.path.join(script_dir, 'conditioning_normalization_stats.pt')
if os.path.exists(norm_stats_path):
    norm_stats = torch.load(norm_stats_path, map_location=device, weights_only=False)
    cond_raw_mean = norm_stats['raw_mean'].to(device)
    cond_raw_std = norm_stats['raw_std'].to(device)
    print(f"✓ Loaded conditioning normalization stats from: conditioning_normalization_stats.pt")
else:
    print(f"⚠ Normalization stats file not found. Run compute_normalization_stats.py first!")
    sys.exit(1)

# Load model
model = ConditionalVAE_SingleHeston(
    param_dim=5, conditioning_dim=8, latent_dim=4,
    hidden_dims=[128, 64, 32],
    encoder_activation='tanh', decoder_activation='relu', dropout=0.15
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded")

# ============================================================================
# Fetch Market Data (1 year buffer for rolling features)
# ============================================================================

fetch_start = target_date - timedelta(days=365)
fetch_end = target_date

print(f"\nFetching market data from {fetch_start} to {fetch_end}...")

# Fetch India VIX
vix_success = False
try:
    print("  → Fetching India VIX from NSEPy...")
    vix_data = get_history(symbol="INDIAVIX", start=fetch_start, end=fetch_end, index=True)
    if len(vix_data) > 0 and 'Close' in vix_data.columns:
        vix_data = vix_data[['Close']].rename(columns={'Close': 'india_vix'})
        print(f"    ✓ India VIX fetched: {len(vix_data)} days")
        vix_success = True
    else:
        raise ValueError("No VIX data returned")
except Exception as e:
    print(f"    ✗ India VIX failed: {e}")
    print(f"    → Trying alternative: Using ^VIX (US VIX) as proxy...")
    try:
        vix_data_us = yf.download('^VIX', start=fetch_start, end=fetch_end, progress=False)
        if len(vix_data_us) > 0:
            # Scale US VIX to approximate India VIX (India VIX typically ~1.2-1.5x US VIX)
            vix_data = vix_data_us[['Close']].rename(columns={'Close': 'india_vix'})
            vix_data['india_vix'] = vix_data['india_vix'] * 1.3  # Approximate scaling
            print(f"    ✓ Using US VIX as proxy (scaled by 1.3x): {len(vix_data)} days")
            vix_success = True
        else:
            raise ValueError("No US VIX data available")
    except Exception as e2:
        print(f"    ✗ Alternative also failed: {e2}")
        print(f"    → Cannot proceed without volatility data")
        sys.exit(1)

# Fetch other market data
print("  → Fetching USD/INR, Crude Oil, US 10Y Yield...")
usdinr_data = yf.download('USDINR=X', start=fetch_start, end=fetch_end, progress=False)[['Close']].rename(columns={'Close': 'usdinr'})
crude_data = yf.download('BZ=F', start=fetch_start, end=fetch_end, progress=False)[['Close']].rename(columns={'Close': 'crude_oil'})
yield_data = yf.download('^TNX', start=fetch_start, end=fetch_end, progress=False)[['Close']].rename(columns={'Close': 'us_10y_yield'})

print(f"    ✓ USD/INR: {len(usdinr_data)} days")
print(f"    ✓ Crude Oil: {len(crude_data)} days")
print(f"    ✓ US 10Y Yield: {len(yield_data)} days")

# Combine all market data
combined_data = pd.concat([vix_data, usdinr_data, crude_data, yield_data], axis=1).dropna()
print(f"✓ Combined market data: {len(combined_data)} days")

# ============================================================================
# Compute Rolling Features
# ============================================================================

print("\nComputing rolling features...")

# India VIX rolling means
combined_data['india_vix_7d_mean'] = combined_data['india_vix'].rolling(7, min_periods=1).mean()
combined_data['india_vix_30d_mean'] = combined_data['india_vix'].rolling(30, min_periods=1).mean()

# Crude oil rolling means
combined_data['crude_oil_7d_mean'] = combined_data['crude_oil'].rolling(7, min_periods=1).mean()
combined_data['crude_oil_30d_mean'] = combined_data['crude_oil'].rolling(30, min_periods=1).mean()

# USD/INR quarterly mean
combined_data['usdinr_quarterly_mean'] = combined_data['usdinr'].rolling(90, min_periods=1).mean()

print("✓ Rolling features computed")

# ============================================================================
# Load GDELT Unrest Index Data
# ============================================================================

print("\nLoading GDELT unrest index data...")

# First, try to load the pre-computed unrest index
gdelt_unrest_file = os.path.join(script_dir, 'gdelt_india_unrest_index.csv')
TRAINING_DATA_START = datetime(2015, 1, 1).date()
TRAINING_DATA_END = datetime(2020, 12, 31).date()
DEFAULT_UNREST_INDEX = 100.457836  # Mean from training data (2015-2020)

if os.path.exists(gdelt_unrest_file):
    print(f"  → Loading pre-computed unrest index from: {gdelt_unrest_file}")
    gdelt_unrest_df = pd.read_csv(gdelt_unrest_file, parse_dates=['Event_Date'], index_col='Event_Date')
    
    # Check if target date is within training range
    target_ts = pd.Timestamp(target_date)
    
    if target_date < TRAINING_DATA_START or target_date > TRAINING_DATA_END:
        print(f"    ⚠ Target date {target_date} is outside training range ({TRAINING_DATA_START} to {TRAINING_DATA_END})")
        print(f"    → Using default unrest index (training mean): {DEFAULT_UNREST_INDEX:.6f}")
        print(f"    ℹ Model was trained on 2015-2020 data; extrapolation to other periods may be unreliable")
        unrest_index_yearly = DEFAULT_UNREST_INDEX
        args.fetch_gdelt = False
    elif target_ts in gdelt_unrest_df.index:
        unrest_index_yearly = float(gdelt_unrest_df.loc[target_ts, 'unrest_index_yearly'])
        print(f"    ✓ Found unrest index for {target_date}: {unrest_index_yearly:.6f}")
        args.fetch_gdelt = False
    else:
        # Find nearest date
        nearest_idx = gdelt_unrest_df.index.get_indexer([target_ts], method='nearest')[0]
        nearest_date = gdelt_unrest_df.index[nearest_idx]
        date_diff = abs((nearest_date - target_ts).days)
        
        if date_diff <= 7:  # Within a week, use it
            unrest_index_yearly = float(gdelt_unrest_df.iloc[nearest_idx]['unrest_index_yearly'])
            print(f"    ⚠ Using nearest date: {nearest_date.date()} (diff: {date_diff} days)")
            print(f"    ✓ Unrest index: {unrest_index_yearly:.6f}")
            args.fetch_gdelt = False
        else:
            print(f"    ⚠ Target date not in pre-computed data (nearest: {nearest_date.date()}, diff: {date_diff} days)")
            print(f"    → Using default value (mean from training data): {DEFAULT_UNREST_INDEX:.6f}")
            unrest_index_yearly = DEFAULT_UNREST_INDEX
            args.fetch_gdelt = False
else:
    print(f"  ⚠ Pre-computed unrest index file not found: {gdelt_unrest_file}")
    print(f"    → Using default value (mean from training data): {DEFAULT_UNREST_INDEX:.6f}")
    unrest_index_yearly = DEFAULT_UNREST_INDEX
    args.fetch_gdelt = False

# Only fetch from GDELT if needed and requested
if args.fetch_gdelt:
    print("\n  Fetching GDELT data from BigQuery for missing date...")
    
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        
        # Check credentials
        gdelt_creds_path = os.path.join(script_dir, args.gdelt_credentials)
        if not os.path.exists(gdelt_creds_path):
            print(f"    ✗ Credentials not found: {gdelt_creds_path}")
            print(f"    → Using default unrest index: {100.457836}")
            unrest_index_yearly = 100.457836
            args.fetch_gdelt = False
        else:
            print(f"    → Using credentials: {gdelt_creds_path}")
            print(f"    → This will take a few minutes...")
            
            # Load credentials
            credentials = service_account.Credentials.from_service_account_file(
                gdelt_creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            client = bigquery.Client(credentials=credentials, project=credentials.project_id)
            
            # Query for a wider range to compute yearly rolling average
            query_start = target_date - timedelta(days=400)
            query_end = target_date
            
            # Use the same query structure as fetch_and_compute_unrest_index.py
            query = f"""
            SELECT
                FORMAT_TIMESTAMP('%Y-%m-%d', PARSE_TIMESTAMP('%Y%m%d', CAST(SQLDATE AS STRING))) AS Event_Date,
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
                AND Actor1CountryCode = 'IND'
                AND CAST(SUBSTR(CAST(SQLDATE AS STRING), 0, 4) AS INT) >= {query_start.year}
                AND CAST(SUBSTR(CAST(SQLDATE AS STRING), 0, 4) AS INT) <= {query_end.year}
                AND NumSources >= 0
                AND NumMentions >= 0
            GROUP BY
                Event_Date
            ORDER BY
                Event_Date ASC;
            """
            
            print(f"    → Querying GDELT from {query_start} to {query_end}...")
            query_job = client.query(query)
            gdelt_df = query_job.to_dataframe()
            
            print(f"      ✓ Fetched {len(gdelt_df)} days")
            
            # Compute unrest index using the same methodology
            # (simplified version - full implementation in fetch_and_compute_unrest_index.py)
            gdelt_df['Event_Date'] = pd.to_datetime(gdelt_df['Event_Date'])
            gdelt_df.set_index('Event_Date', inplace=True)
            
            # Compute weighted unrest score (simplified)
            base_weights = {
                'Demonstrations_Mentions': 1.0,
                'Hunger_Strikes_Mentions': 1.5,
                'Strikes_Boycotts_Mentions': 2.0,
                'Attacks_Mentions': 6.0,
                'Mass_Violence_Mentions': 8.0,
                'General_Coerce_Mentions': 3.0,
                'Seizure_Mentions': 5.0
            }
            
            gdelt_df['Weighted_Base'] = sum(gdelt_df[col].fillna(0) * w for col, w in base_weights.items())
            gdelt_df['RSUI_weighted_daily'] = gdelt_df['Weighted_Base']  # Simplified
            gdelt_df['unrest_index_yearly'] = gdelt_df['RSUI_weighted_daily'].rolling(window=365, min_periods=1).mean()
            
            # Get value for target date
            target_ts_gdelt = pd.Timestamp(target_date)
            if target_ts_gdelt in gdelt_df.index:
                unrest_index_yearly = float(gdelt_df.loc[target_ts_gdelt, 'unrest_index_yearly'])
            else:
                nearest_idx = gdelt_df.index.get_indexer([target_ts_gdelt], method='nearest')[0]
                unrest_index_yearly = float(gdelt_df.iloc[nearest_idx]['unrest_index_yearly'])
            
            print(f"      ✓ Computed unrest index: {unrest_index_yearly:.6f}")
            
    except ImportError:
        print("    ✗ Google Cloud libraries not installed")
        print("      Install with: pip install google-cloud-bigquery google-auth db-dtypes")
        print(f"    → Using default unrest index: {100.457836}")
        unrest_index_yearly = 100.457836
    except Exception as e:
        print(f"    ✗ Error fetching GDELT data: {e}")
        print(f"    → Using default unrest index: {100.457836}")
        unrest_index_yearly = 100.457836

# Unrest index already loaded/computed above
print(f"\n✓ Using unrest index: {unrest_index_yearly:.6f}")

# ============================================================================
# Extract Features for Target Date
# ============================================================================

print(f"\nExtracting features for {target_date}...")

# Get data for target date (or nearest available)
target_ts = pd.Timestamp(target_date)
if target_ts in combined_data.index:
    row = combined_data.loc[target_ts]
    print(f"  ✓ Exact match found for {target_date}")
else:
    nearest_idx = combined_data.index.get_indexer([target_ts], method='nearest')[0]
    nearest_date = combined_data.index[nearest_idx]
    row = combined_data.loc[nearest_date]
    print(f"  ⚠ Using nearest date: {nearest_date.date()} (requested: {target_date})")

# Build raw conditioning vector (in the order specified in config)
# Order: crude_oil_30d_mean, crude_oil_7d_mean, unrest_index_yearly, crude_oil, 
#        usdinr_quarterly_mean, india_vix_30d_mean, india_vix_7d_mean, us_10y_yield
conditioning_raw = np.array([
    float(row['crude_oil_30d_mean']),
    float(row['crude_oil_7d_mean']),
    float(unrest_index_yearly),
    float(row['crude_oil']),
    float(row['usdinr_quarterly_mean']),
    float(row['india_vix_30d_mean']),
    float(row['india_vix_7d_mean']),
    float(row['us_10y_yield'])
], dtype=np.float32)

print("\nRaw conditioning variables:")
for i, var_name in enumerate(config['data']['conditioning_vars']):
    print(f"  {var_name:25s}: {conditioning_raw[i]:10.4f}")

# ============================================================================
# Normalize Conditioning Variables
# ============================================================================

print("\nNormalizing conditioning variables using historical statistics...")

# Normalize using the pre-computed statistics from historical data
cond_raw_tensor = torch.tensor(conditioning_raw, dtype=torch.float32)
conditioning_normalized = (cond_raw_tensor - cond_raw_mean.cpu()) / (cond_raw_std.cpu() + 1e-8)

print("\nNormalized conditioning variables:")
for i, var_name in enumerate(config['data']['conditioning_vars']):
    print(f"  {var_name:25s}: raw={conditioning_raw[i]:8.4f}, norm={conditioning_normalized[i].item():8.4f}")

# Prepare batch for model
conditioning_tensor = conditioning_normalized.to(device)
conditioning_batch = conditioning_tensor.unsqueeze(0).repeat(args.n_samples, 1)

print(f"\n✓ Conditioning batch prepared: shape {conditioning_batch.shape}")

# ============================================================================
# Generate Heston Parameters
# ============================================================================

print(f"\nGenerating {args.n_samples} Heston parameter sets...")

with torch.no_grad():
    # Sample from latent space
    z = torch.randn(args.n_samples, 4).to(device)
    
    # Decode to get normalized parameters
    normalized_params = model.decode(z, conditioning_batch)
    
    # Denormalize: x = x_norm * std + mean
    params_transformed = normalized_params * param_std + param_mean
    
    # Inverse transforms to get original scale
    params_original = params_transformed.clone()
    
    # kappa, theta, sigma_v, v0: exp transform
    for idx in [0, 1, 2, 4]:
        params_original[:, idx] = torch.exp(params_transformed[:, idx])
    
    # rho: tanh transform (map from [-inf, inf] to [-1, 1])
    params_original[:, 3] = torch.tanh(params_transformed[:, 3])
    
    params_np = params_original.cpu().numpy()

print(f"✓ Generated {len(params_np)} parameter sets")

# Print parameter statistics
param_names = ['kappa', 'theta', 'sigma_v', 'rho', 'v0']
print("\nParameter statistics:")
for i, name in enumerate(param_names):
    print(f"  {name:10s}: mean={params_np[:, i].mean():8.4f}, "
          f"std={params_np[:, i].std():8.4f}, "
          f"min={params_np[:, i].min():8.4f}, "
          f"max={params_np[:, i].max():8.4f}")

# ============================================================================
# Generate IV Surfaces
# ============================================================================

print("\nGenerating IV surfaces...")

# Define grid - MUST match the training data grid
T_grid = np.array([0.08333333, 0.16666667, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])  # 1M, 2M, 3M, 6M, 9M, 12M, 18M, 24M
logm_grid = np.linspace(-0.2, 0.2, 21)  # 21 strikes from -20% to +20%

def generate_iv_surface(params, T_grid, logm_grid, spot, r, q):
    """Generate IV surface for given Heston parameters"""
    kappa, theta, sigma_v, rho, v0 = params
    iv_surface = np.zeros((len(T_grid), len(logm_grid)))
    
    try:
        hes_model = HestonModelQL(float(kappa), float(theta), float(sigma_v), float(rho), float(v0), r, q)
    except:
        return None, False
    
    nan_count = 0
    total_points = len(T_grid) * len(logm_grid)
    
    for mat_idx, tau in enumerate(T_grid):
        for strike_idx, logm in enumerate(logm_grid):
            try:
                price_ratio = hes_model.price_ratio(logm, tau)
                heston_price = price_ratio * spot * np.exp(-r * tau)
                K = spot * np.exp(logm)
                F = spot * np.exp((r - q) * tau)
                
                # Check if price is reasonable
                intrinsic = max(F - K, 0) * np.exp(-r * tau)
                if heston_price < intrinsic * 0.99:  # Price should be >= intrinsic
                    iv_surface[mat_idx, strike_idx] = np.nan
                    nan_count += 1
                    continue
                
                def bs_objective(iv):
                    d1 = (np.log(F / K) + 0.5 * iv**2 * tau) / (iv * np.sqrt(tau))
                    d2 = d1 - iv * np.sqrt(tau)
                    bs_price = np.exp(-r * tau) * (F * norm.cdf(d1) - K * norm.cdf(d2))
                    return bs_price - heston_price
                
                iv = brentq(bs_objective, 0.001, 3.0, xtol=1e-6)
                iv_surface[mat_idx, strike_idx] = iv
            except:
                iv_surface[mat_idx, strike_idx] = np.nan
                nan_count += 1
    
    # Accept surface if at least 90% of points are valid
    success = (nan_count / total_points) < 0.1
    return iv_surface, success

# Generate surfaces for all parameter sets
surfaces = []
failed_reasons = {'model_creation': 0, 'pricing': 0, 'iv_extraction': 0}
for idx in tqdm(range(args.n_samples), desc="Generating surfaces"):
    iv_surface, success = generate_iv_surface(params_np[idx], T_grid, logm_grid, args.spot, args.r, args.q)
    if success:
        surfaces.append(iv_surface)
    else:
        # Debug: check why it failed
        kappa, theta, sigma_v, rho, v0 = params_np[idx]
        try:
            hes_model = HestonModelQL(float(kappa), float(theta), float(sigma_v), float(rho), float(v0), args.r, args.q)
            failed_reasons['pricing'] += 1
        except:
            failed_reasons['model_creation'] += 1

if len(surfaces) == 0:
    print(f"✗ No valid surfaces generated (0/{args.n_samples})")
    print(f"   Failure reasons:")
    print(f"     - Model creation failed: {failed_reasons['model_creation']}")
    print(f"     - Pricing/IV extraction failed: {failed_reasons['pricing']}")
    print("   Try increasing --n_samples or check parameter distributions")
    sys.exit(1)

surfaces_array = np.array(surfaces)

# Use nanmean/nanmedian to handle any NaN values
mean_surface = np.nanmean(surfaces_array, axis=0)
median_surface = np.nanmedian(surfaces_array, axis=0)
p5_surface = np.nanpercentile(surfaces_array, 5, axis=0)
p95_surface = np.nanpercentile(surfaces_array, 95, axis=0)

print(f"✓ Valid surfaces: {len(surfaces)}/{args.n_samples} ({len(surfaces)/args.n_samples*100:.1f}%)")

# Check for NaN values
nan_count = np.isnan(mean_surface).sum()
total_points = mean_surface.size
if nan_count > 0:
    print(f"\n⚠ Warning: {nan_count}/{total_points} points have NaN values ({nan_count/total_points*100:.1f}%)")
    print("  These are typically extreme OTM strikes that failed to price")

# Print summary statistics
print("\nIV Surface Summary Statistics:")
print(f"  Mean IV (overall):   {np.nanmean(mean_surface) * 100:.2f}%")
print(f"  Median IV (overall): {np.nanmean(median_surface) * 100:.2f}%")
print(f"  Min IV:              {np.nanmin(mean_surface) * 100:.2f}%")
print(f"  Max IV:              {np.nanmax(mean_surface) * 100:.2f}%")

# ============================================================================
# Save Results
# ============================================================================

output_dir = os.path.join(script_dir, args.output_dir, target_date.strftime('%Y-%m-%d'))
os.makedirs(output_dir, exist_ok=True)

# Save surfaces and metadata
torch.save({
    'date': target_date.strftime('%Y-%m-%d'),
    'conditioning_raw': conditioning_raw,
    'conditioning_normalized': conditioning_normalized.cpu().numpy(),
    'params': params_np,
    'surfaces': surfaces_array,
    'mean_surface': mean_surface,
    'median_surface': median_surface,
    'p5_surface': p5_surface,
    'p95_surface': p95_surface,
    'T_grid': T_grid,
    'logm_grid': logm_grid,
    'spot': args.spot,
    'r': args.r,
    'q': args.q
}, os.path.join(output_dir, 'iv_surfaces.pt'))

print(f"\n✓ Saved surfaces to: {os.path.join(output_dir, 'iv_surfaces.pt')}")

# Save mean and median IV matrices as CSV files
# Create DataFrames with proper labels
maturity_labels = [f'{int(round(T*12))}M' for T in T_grid]  # e.g., '1M', '2M', etc.
strike_labels = [f'{logm*100:.1f}%' for logm in logm_grid]  # e.g., '-20.0%', '-18.0%', etc.

mean_iv_df = pd.DataFrame(
    mean_surface * 100,  # Convert to percentage
    index=maturity_labels,
    columns=strike_labels
)
mean_iv_df.index.name = 'Maturity'

median_iv_df = pd.DataFrame(
    median_surface * 100,  # Convert to percentage
    index=maturity_labels,
    columns=strike_labels
)
median_iv_df.index.name = 'Maturity'

# Save to CSV
mean_iv_df.to_csv(os.path.join(output_dir, 'mean_iv_surface.csv'))
median_iv_df.to_csv(os.path.join(output_dir, 'median_iv_surface.csv'))

print(f"✓ Saved mean IV matrix to: {os.path.join(output_dir, 'mean_iv_surface.csv')}")
print(f"✓ Saved median IV matrix to: {os.path.join(output_dir, 'median_iv_surface.csv')}")

# Print the matrices
print("\n" + "="*80)
print("MEAN IV SURFACE (%) - 8 Maturities x 21 Strikes")
print("="*80)
print(mean_iv_df.to_string())

print("\n" + "="*80)
print("MEDIAN IV SURFACE (%) - 8 Maturities x 21 Strikes")
print("="*80)
print(median_iv_df.to_string())

# ============================================================================
# Plot Results
# ============================================================================

print("\nGenerating plots...")

# Plot 1: ATM Term Structure
fig, ax = plt.subplots(figsize=(10, 6))
maturities_months = T_grid * 12
atm_idx = len(logm_grid) // 2

ax.plot(maturities_months, mean_surface[:, atm_idx] * 100, 'b-', linewidth=2, label='Mean', marker='o')
ax.fill_between(maturities_months, p5_surface[:, atm_idx] * 100, p95_surface[:, atm_idx] * 100,
                alpha=0.3, color='blue', label='5th-95th %ile')
ax.set_xlabel('Maturity (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
ax.set_title(f'ATM Term Structure - {target_date}', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'atm_term_structure.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ ATM term structure plot saved")

# Plot 2: Mean IV Surface Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.contourf(logm_grid * 100, T_grid * 12, mean_surface * 100, levels=20, cmap='viridis')
ax.set_xlabel('Log-Moneyness (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Maturity (months)', fontsize=12, fontweight='bold')
ax.set_title(f'Mean IV Surface - {target_date}', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, label='Implied Volatility (%)')
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mean_surface_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Mean surface heatmap saved")

# Plot 2b: Median IV Surface Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.contourf(logm_grid * 100, T_grid * 12, median_surface * 100, levels=20, cmap='viridis')
ax.set_xlabel('Log-Moneyness (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Maturity (months)', fontsize=12, fontweight='bold')
ax.set_title(f'Median IV Surface - {target_date}', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, label='Implied Volatility (%)')
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'median_surface_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Median surface heatmap saved")

# Plot 2c: Comparison of Mean vs Median
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Mean surface
im1 = axes[0].contourf(logm_grid * 100, T_grid * 12, mean_surface * 100, levels=20, cmap='viridis')
axes[0].set_xlabel('Log-Moneyness (%)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Maturity (months)', fontsize=11, fontweight='bold')
axes[0].set_title('Mean IV Surface', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0], label='IV (%)')

# Median surface
im2 = axes[1].contourf(logm_grid * 100, T_grid * 12, median_surface * 100, levels=20, cmap='viridis')
axes[1].set_xlabel('Log-Moneyness (%)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Maturity (months)', fontsize=11, fontweight='bold')
axes[1].set_title('Median IV Surface', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[1], label='IV (%)')

plt.suptitle(f'Mean vs Median IV Surfaces - {target_date}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mean_vs_median_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Mean vs Median comparison saved")

# Plot 3: Smile at different maturities
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

maturity_indices = [0, 3, 5, 7]  # 1M, 6M, 12M, 24M
for idx, mat_idx in enumerate(maturity_indices):
    ax = axes[idx]
    
    # Get the data for this maturity
    mean_iv = mean_surface[mat_idx, :] * 100
    p5_iv = p5_surface[mat_idx, :] * 100
    p95_iv = p95_surface[mat_idx, :] * 100
    
    ax.plot(logm_grid * 100, mean_iv, 'b-', linewidth=2, label='Mean', marker='o', markersize=4)
    ax.fill_between(logm_grid * 100, p5_iv, p95_iv,
                    alpha=0.3, color='blue', label='5th-95th %ile')
    ax.set_xlabel('Log-Moneyness (%)', fontsize=10)
    ax.set_ylabel('Implied Volatility (%)', fontsize=10)
    ax.set_title(f'Maturity: {int(round(T_grid[mat_idx]*12))} months', fontsize=11, fontweight='bold')
    
    # Set y-axis limits to show the smile structure better
    iv_range = np.nanmax(mean_iv) - np.nanmin(mean_iv)
    y_min = np.nanmin(mean_iv) - 0.5 * iv_range
    y_max = np.nanmax(mean_iv) + 0.5 * iv_range
    ax.set_ylim([y_min, y_max])
    
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='ATM')

plt.suptitle(f'IV Smiles at Different Maturities - {target_date}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'iv_smiles.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ IV smiles plot saved")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print(f"✓ GENERATION COMPLETE!")
print(f"{'='*80}")
print(f"Date:              {target_date}")
print(f"Spot price:        {args.spot:.2f}")
print(f"Risk-free rate:    {args.r:.4f}")
print(f"Dividend yield:    {args.q:.4f}")
print(f"Samples generated: {args.n_samples}")
print(f"Valid surfaces:    {len(surfaces)}/{args.n_samples} ({len(surfaces)/args.n_samples*100:.1f}%)")
print(f"Output directory:  {output_dir}")
print(f"\nFiles generated:")
print(f"  Data files:")
print(f"    - iv_surfaces.pt (PyTorch data with all surfaces)")
print(f"    - mean_iv_surface.csv (8x21 mean IV matrix)")
print(f"    - median_iv_surface.csv (8x21 median IV matrix)")
print(f"  Plots:")
print(f"    - atm_term_structure.png")
print(f"    - mean_surface_heatmap.png")
print(f"    - median_surface_heatmap.png")
print(f"    - mean_vs_median_comparison.png")
print(f"    - iv_smiles.png")
print(f"{'='*80}\n")
