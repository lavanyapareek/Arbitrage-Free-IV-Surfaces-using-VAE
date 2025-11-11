"""
Generate IV Surface for a Specific Date using Conditional VAE

This script generates an IV surface for a given date by:
1. Fetching market data (VIX, crude oil, USD/INR, US 10Y yield)
2. Computing rolling features (7d, 30d, quarterly means)
3. Loading GDELT unrest index for the date
4. Normalizing features using training statistics
5. Sampling Heston parameters from the trained CVAE model
6. Generating and visualizing the IV surface

Usage:
    python generate_iv_surface_by_date.py --date 2020-06-01 --n_samples 500
    python generate_iv_surface_by_date.py --date 2021-03-15 --n_samples 1000 --spot 16000
"""

import os
import sys
import json
import argparse
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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvae_model import ConditionalVAE_SingleHeston
from heston_model_ql import HestonModelQL

# ============================================================================
# Configuration
# ============================================================================

# Grid parameters (must match training data)
T_GRID = np.array([0.08333333, 0.16666667, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])  # 1M, 2M, 3M, 6M, 9M, 12M, 18M, 24M
LOGM_GRID = np.linspace(-0.2, 0.2, 21)  # 21 strikes from -20% to +20%

# Market training data range
TRAINING_DATA_START = datetime(2015, 1, 1).date()
TRAINING_DATA_END = datetime(2026, 12, 31).date()
DEFAULT_UNREST_INDEX = 100.457836  # Mean from training data (2015-2020)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Helper Functions
# ============================================================================

def fetch_market_data(target_date, buffer_days=365):
    """
    Fetch market data for the target date and compute rolling features.
    
    Args:
        target_date: datetime.date object
        buffer_days: days of history to fetch for rolling features
        
    Returns:
        dict with market data for target date and rolling features
    """
    print(f"\nFetching market data ({buffer_days}-day buffer)...")
    
    fetch_start = target_date - timedelta(days=buffer_days)
    fetch_end = target_date
    
    # Fetch India VIX via yfinance
    print("  → Fetching India VIX (^INDIAVIX)...")
    try:
        vix_data = yf.download('^INDIAVIX', start=fetch_start, end=fetch_end, progress=False)
        
        if len(vix_data) > 0:
            # Handle yfinance's multi-level column indexing for single ticker
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data = vix_data[('Close', '^INDIAVIX')].to_frame(name='india_vix')
            else:
                vix_data = vix_data[['Close']].rename(columns={'Close': 'india_vix'})
            
            vix_data['india_vix'] = pd.to_numeric(vix_data['india_vix'], errors='coerce')
            vix_data = vix_data.dropna()
            
            if len(vix_data) > 0:
                print(f"    ✓ India VIX: {len(vix_data)} days")
            else:
                raise ValueError("No valid India VIX data after cleaning")
        else:
            raise ValueError("No India VIX data available")
    except Exception as e:
        print(f"    ✗ India VIX fetch failed: {e}")
        sys.exit(1)
    
    # Fetch other market data
    print("  → Fetching USD/INR, Crude Oil, US 10Y Yield...")
    
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
    
    usdinr_raw = yf.download('USDINR=X', start=fetch_start, end=fetch_end, progress=False)
    usdinr_data = pd.DataFrame({'usdinr': get_close_series(usdinr_raw)})
    
    crude_raw = yf.download('BZ=F', start=fetch_start, end=fetch_end, progress=False)
    crude_data = pd.DataFrame({'crude_oil': get_close_series(crude_raw)})
    
    yield_raw = yf.download('^TNX', start=fetch_start, end=fetch_end, progress=False)
    yield_data = pd.DataFrame({'us_10y_yield': get_close_series(yield_raw)})
    
    print(f"    ✓ USD/INR: {len(usdinr_data)} days")
    print(f"    ✓ Crude Oil: {len(crude_data)} days")
    print(f"    ✓ US 10Y Yield: {len(yield_data)} days")
    
    # Combine all market data
    combined_data = pd.concat([vix_data, usdinr_data, crude_data, yield_data], axis=1).dropna()
    print(f"✓ Combined market data: {len(combined_data)} days")
    
    if len(combined_data) == 0:
        print(f"  ✗ Warning: No overlapping dates found in market data")
        print(f"    Date range requested: {fetch_start} to {fetch_end}")
        print(f"    VIX range: {vix_data.index.min() if len(vix_data) > 0 else 'N/A'} to {vix_data.index.max() if len(vix_data) > 0 else 'N/A'}")
        print(f"    USD/INR range: {usdinr_data.index.min() if len(usdinr_data) > 0 else 'N/A'} to {usdinr_data.index.max() if len(usdinr_data) > 0 else 'N/A'}")
        return combined_data
    
    # Compute rolling features
    print("\nComputing rolling features...")
    combined_data['india_vix_7d_mean'] = combined_data['india_vix'].rolling(7, min_periods=1).mean()
    combined_data['india_vix_30d_mean'] = combined_data['india_vix'].rolling(30, min_periods=1).mean()
    combined_data['crude_oil_7d_mean'] = combined_data['crude_oil'].rolling(7, min_periods=1).mean()
    combined_data['crude_oil_30d_mean'] = combined_data['crude_oil'].rolling(30, min_periods=1).mean()
    combined_data['usdinr_quarterly_mean'] = combined_data['usdinr'].rolling(90, min_periods=1).mean()
    print("✓ Rolling features computed")
    
    return combined_data


def get_conditioning_vector(combined_data, target_date, unrest_index, config):
    """
    Extract conditioning vector for target date.
    
    Args:
        combined_data: DataFrame with market data
        target_date: datetime.date object
        unrest_index: float, GDELT unrest index
        config: dict with conditioning variable names
        
    Returns:
        np.array of shape (8,) with normalized conditioning variables
    """
    print(f"\nExtracting features for {target_date}...")
    
    # Check if combined_data is empty
    if len(combined_data) == 0:
        print(f"  ✗ Error: No market data available for {target_date}")
        print(f"  → Possible causes: Market closed, data source issue, or date range mismatch")
        sys.exit(1)
    
    target_ts = pd.Timestamp(target_date)
    if target_ts in combined_data.index:
        row = combined_data.loc[target_ts]
        print(f"  ✓ Exact match found for {target_date}")
    else:
        # Get nearest available date
        nearest_indices = combined_data.index.get_indexer([target_ts], method='nearest')
        if nearest_indices[0] == -1 or len(combined_data) == 0:
            print(f"  ✗ Error: No valid data to extract features from")
            sys.exit(1)
        
        nearest_idx = nearest_indices[0]
        nearest_date = combined_data.index[nearest_idx]
        row = combined_data.loc[nearest_date]
        date_diff = abs((nearest_date - target_ts).days)
        print(f"  ⚠ Using nearest date: {nearest_date.date()} (diff: {date_diff} days)")
    
    # Build raw conditioning vector in config order
    # Order: crude_oil_30d_mean, crude_oil_7d_mean, unrest_index_yearly, crude_oil,
    #        usdinr_quarterly_mean, india_vix_30d_mean, india_vix_7d_mean, us_10y_yield
    conditioning_raw = np.array([
        float(row['crude_oil_30d_mean']),
        float(row['crude_oil_7d_mean']),
        float(unrest_index),
        float(row['crude_oil']),
        float(row['usdinr_quarterly_mean']),
        float(row['india_vix_30d_mean']),
        float(row['india_vix_7d_mean']),
        float(row['us_10y_yield'])
    ], dtype=np.float32)
    
    print("\nRaw conditioning variables:")
    for i, var_name in enumerate(config['data']['conditioning_vars']):
        print(f"  {var_name:25s}: {conditioning_raw[i]:10.4f}")
    
    return conditioning_raw


def load_unrest_index(target_date, gdelt_file):
    """
    Load GDELT unrest index for target date.
    
    Args:
        target_date: datetime.date object
        gdelt_file: str, path to GDELT CSV file
        
    Returns:
        float, unrest index value
    """
    print("\nLoading GDELT unrest index data...")
    
    if not os.path.exists(gdelt_file):
        print(f"  ⚠ File not found: {gdelt_file}")
        print(f"    → Using default value (training mean): {DEFAULT_UNREST_INDEX:.6f}")
        return DEFAULT_UNREST_INDEX
    
    print(f"  → Loading from: {gdelt_file}")
    gdelt_df = pd.read_csv(gdelt_file, parse_dates=['Event_Date'], index_col='Event_Date')
    
    target_ts = pd.Timestamp(target_date)
    
    # Check if target date is within training range
    if target_date < TRAINING_DATA_START or target_date > TRAINING_DATA_END:
        print(f"    ⚠ Target date {target_date} is outside training range ({TRAINING_DATA_START} to {TRAINING_DATA_END})")
        print(f"    → Using default value: {DEFAULT_UNREST_INDEX:.6f}")
        return DEFAULT_UNREST_INDEX
    
    # Try to find exact match
    if target_ts in gdelt_df.index:
        unrest = float(gdelt_df.loc[target_ts, 'unrest_index_yearly'])
        print(f"    ✓ Found for {target_date}: {unrest:.6f}")
        return unrest
    else:
        # Find nearest date
        nearest_idx = gdelt_df.index.get_indexer([target_ts], method='nearest')[0]
        nearest_date = gdelt_df.index[nearest_idx]
        date_diff = abs((nearest_date - target_ts).days)
        
        if date_diff <= 7:
            unrest = float(gdelt_df.iloc[nearest_idx]['unrest_index_yearly'])
            print(f"    ⚠ Using nearest date: {nearest_date.date()} (diff: {date_diff} days)")
            print(f"    ✓ Unrest index: {unrest:.6f}")
            return unrest
        else:
            print(f"    ⚠ Nearest date {nearest_date.date()} is {date_diff} days away")
            print(f"    → Using default value: {DEFAULT_UNREST_INDEX:.6f}")
            return DEFAULT_UNREST_INDEX


def generate_iv_surface(params, T_grid, logm_grid, spot, r, q):
    """
    Generate IV surface for a single Heston parameter set.
    
    Args:
        params: (5,) array [kappa, theta, sigma_v, rho, v0]
        T_grid: array of maturities
        logm_grid: array of log-moneyness
        spot: float, spot price
        r: float, risk-free rate
        q: float, dividend yield
        
    Returns:
        (iv_surface, success): IV surface array and bool indicating success
    """
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
                if heston_price < intrinsic * 0.99:
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


# ============================================================================
# Main Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate IV surface for a specific date using Conditional VAE')
    parser.add_argument('--date', type=str, required=True, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of parameter samples (default: 500)')
    parser.add_argument('--output_dir', type=str, default='results_date', help='Output directory')
    parser.add_argument('--spot', type=float, default=None, help='Spot price (if None, fetches from yfinance)')
    parser.add_argument('--r', type=float, default=0.067, help='Risk-free rate')
    parser.add_argument('--q', type=float, default=0.0, help='Dividend yield')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*80)
    print(f"CONDITIONAL VAE IV SURFACE GENERATOR - {args.date}")
    print("="*80)
    
    # Parse target date
    target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    
    # ========================================================================
    # Load Configuration and Model
    # ========================================================================
    
    print("\nLoading configuration and model...")
    
    config_path = os.path.join(script_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    
    model_path = os.path.join(script_dir, '..', 'llm_options_assistant', 'best_model_2025', 'cvae_model.pt')
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    param_mean = checkpoint['param_mean'].to(DEVICE)
    param_std = checkpoint['param_std'].to(DEVICE)
    
    norm_stats_path = os.path.join(script_dir, 'conditioning_normalization_stats.pt')
    if os.path.exists(norm_stats_path):
        norm_stats = torch.load(norm_stats_path, map_location=DEVICE, weights_only=False)
        cond_raw_mean = norm_stats['raw_mean'].to(DEVICE)
        cond_raw_std = norm_stats['raw_std'].to(DEVICE)
        print("✓ Loaded conditioning normalization stats")
    else:
        print("✗ Normalization stats file not found!")
        sys.exit(1)
    
    # Load model
    model = ConditionalVAE_SingleHeston(
        param_dim=5, conditioning_dim=8, latent_dim=4,
        hidden_dims=[128, 64, 32],
        encoder_activation='tanh', decoder_activation='relu', dropout=0.15
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    
    # ========================================================================
    # Fetch Data and Prepare Conditioning Variables
    # ========================================================================
    
    # Fetch spot price if not provided
    if args.spot is None:
        print("\nFetching NIFTY spot price...")
        try:
            fetch_start_spot = target_date - timedelta(days=7)
            nifty_data = yf.download('^NSEI', start=fetch_start_spot, end=target_date + timedelta(days=1), progress=False)
            if len(nifty_data) > 0 and 'Close' in nifty_data.columns:
                args.spot = float(nifty_data['Close'].iloc[-1])
                actual_date = nifty_data.index[-1].date()
                if actual_date == target_date:
                    print(f"  ✓ NIFTY 50 spot: {args.spot:.2f}")
                else:
                    print(f"  ✓ NIFTY 50 spot (nearest: {actual_date}): {args.spot:.2f}")
            else:
                raise ValueError("No NIFTY data available")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  → Using default: 21000.0")
            args.spot = 21000.0
    else:
        print(f"\nUsing provided spot price: {args.spot:.2f}")
    
    # Fetch market data
    combined_data = fetch_market_data(target_date, buffer_days=365)
    
    # Load unrest index
    gdelt_file = os.path.join(script_dir, 'gdelt_india_unrest_index.csv')
    unrest_index = load_unrest_index(target_date, gdelt_file)
    
    # Extract and normalize conditioning variables
    conditioning_raw = get_conditioning_vector(combined_data, target_date, unrest_index, config)
    
    cond_raw_tensor = torch.tensor(conditioning_raw, dtype=torch.float32)
    conditioning_normalized = (cond_raw_tensor - cond_raw_mean.cpu()) / (cond_raw_std.cpu() + 1e-8)
    
    print("\nNormalized conditioning variables:")
    for i, var_name in enumerate(config['data']['conditioning_vars']):
        print(f"  {var_name:25s}: norm={conditioning_normalized[i].item():8.4f}")
    
    conditioning_tensor = conditioning_normalized.to(DEVICE)
    conditioning_batch = conditioning_tensor.unsqueeze(0).repeat(args.n_samples, 1)
    
    print(f"\n✓ Conditioning batch prepared: shape {conditioning_batch.shape}")
    
    # ========================================================================
    # Generate Heston Parameters
    # ========================================================================
    
    print(f"\nGenerating {args.n_samples} Heston parameter sets...")
    
    with torch.no_grad():
        z = torch.randn(args.n_samples, 4).to(DEVICE)
        normalized_params = model.decode(z, conditioning_batch)
        params_transformed = normalized_params * param_std + param_mean
        
        params_original = params_transformed.clone()
        for idx in [0, 1, 2, 4]:
            params_original[:, idx] = torch.exp(params_transformed[:, idx])
        params_original[:, 3] = torch.tanh(params_transformed[:, 3])
        
        params_np = params_original.cpu().numpy()
    
    print(f"✓ Generated {len(params_np)} parameter sets")
    
    # Print parameter statistics
    param_names = ['kappa', 'theta', 'sigma_v', 'rho', 'v0']
    print("\nParameter statistics:")
    for i, name in enumerate(param_names):
        print(f"  {name:10s}: mean={params_np[:, i].mean():8.4f}, std={params_np[:, i].std():8.4f}")
    
    # ========================================================================
    # Generate IV Surfaces
    # ========================================================================
    
    print(f"\nGenerating IV surfaces...")
    
    surfaces = []
    for idx in tqdm(range(args.n_samples), desc="Generating"):
        iv_surface, success = generate_iv_surface(params_np[idx], T_GRID, LOGM_GRID, args.spot, args.r, args.q)
        if success:
            surfaces.append(iv_surface)
    
    if len(surfaces) == 0:
        print("✗ No valid surfaces generated!")
        sys.exit(1)
    
    surfaces_array = np.array(surfaces)
    mean_surface = np.nanmean(surfaces_array, axis=0)
    median_surface = np.nanmedian(surfaces_array, axis=0)
    p5_surface = np.nanpercentile(surfaces_array, 5, axis=0)
    p95_surface = np.nanpercentile(surfaces_array, 95, axis=0)
    
    print(f"✓ Valid surfaces: {len(surfaces)}/{args.n_samples} ({len(surfaces)/args.n_samples*100:.1f}%)")
    print(f"\nIV Surface Summary Statistics:")
    print(f"  Mean IV (overall):   {np.nanmean(mean_surface) * 100:.2f}%")
    print(f"  Median IV (overall): {np.nanmean(median_surface) * 100:.2f}%")
    print(f"  Min IV:              {np.nanmin(mean_surface) * 100:.2f}%")
    print(f"  Max IV:              {np.nanmax(mean_surface) * 100:.2f}%")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    output_dir = os.path.join(script_dir, args.output_dir, target_date.strftime('%Y-%m-%d'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PyTorch data
    torch.save({
        'date': target_date.strftime('%Y-%m-%d'),
        'params': params_np,
        'surfaces': surfaces_array,
        'mean_surface': mean_surface,
        'median_surface': median_surface,
        'p5_surface': p5_surface,
        'p95_surface': p95_surface,
        'T_grid': T_GRID,
        'logm_grid': LOGM_GRID,
        'spot': args.spot,
        'r': args.r,
        'q': args.q
    }, os.path.join(output_dir, 'iv_surfaces.pt'))
    
    print(f"\n✓ Saved to: {os.path.join(output_dir, 'iv_surfaces.pt')}")
    
    # Save CSV matrices
    maturity_labels = [f'{int(round(T*12))}M' for T in T_GRID]
    strike_labels = [f'{logm*100:.1f}%' for logm in LOGM_GRID]
    
    mean_iv_df = pd.DataFrame(mean_surface * 100, index=maturity_labels, columns=strike_labels)
    median_iv_df = pd.DataFrame(median_surface * 100, index=maturity_labels, columns=strike_labels)
    
    mean_iv_df.to_csv(os.path.join(output_dir, 'mean_iv_surface.csv'))
    median_iv_df.to_csv(os.path.join(output_dir, 'median_iv_surface.csv'))
    
    print(f"✓ Saved CSV matrices to output directory")
    
    # ========================================================================
    # Generate Visualizations
    # ========================================================================
    
    print(f"\nGenerating plots...")
    
    # ATM Term Structure
    fig, ax = plt.subplots(figsize=(10, 6))
    maturities_months = T_GRID * 12
    atm_idx = len(LOGM_GRID) // 2
    
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
    
    # Mean IV Surface Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.contourf(LOGM_GRID * 100, T_GRID * 12, mean_surface * 100, levels=20, cmap='viridis')
    ax.set_xlabel('Log-Moneyness (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Maturity (months)', fontsize=12, fontweight='bold')
    ax.set_title(f'Mean IV Surface - {target_date}', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Implied Volatility (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_surface_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # IV Smiles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    maturity_indices = [0, 3, 5, 7]  # 1M, 6M, 12M, 24M
    for idx, mat_idx in enumerate(maturity_indices):
        ax = axes[idx]
        mean_iv = mean_surface[mat_idx, :] * 100
        p5_iv = p5_surface[mat_idx, :] * 100
        p95_iv = p95_surface[mat_idx, :] * 100
        
        ax.plot(LOGM_GRID * 100, mean_iv, 'b-', linewidth=2, label='Mean', marker='o', markersize=4)
        ax.fill_between(LOGM_GRID * 100, p5_iv, p95_iv, alpha=0.3, color='blue', label='5th-95th %ile')
        ax.set_xlabel('Log-Moneyness (%)', fontsize=10)
        ax.set_ylabel('Implied Volatility (%)', fontsize=10)
        ax.set_title(f'Maturity: {int(round(T_GRID[mat_idx]*12))} months', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle(f'IV Smiles at Different Maturities - {target_date}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iv_smiles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Generated 3 plots")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"✓ GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Date:                    {target_date}")
    print(f"Spot price:              {args.spot:.2f}")
    print(f"Risk-free rate:          {args.r:.4f}")
    print(f"Dividend yield:          {args.q:.4f}")
    print(f"Samples generated:       {args.n_samples}")
    print(f"Valid surfaces:          {len(surfaces)}/{args.n_samples} ({len(surfaces)/args.n_samples*100:.1f}%)")
    print(f"Output directory:        {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - iv_surfaces.pt (PyTorch data)")
    print(f"  - mean_iv_surface.csv (8x21 matrix)")
    print(f"  - median_iv_surface.csv (8x21 matrix)")
    print(f"  - atm_term_structure.png")
    print(f"  - mean_surface_heatmap.png")
    print(f"  - iv_smiles.png")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
