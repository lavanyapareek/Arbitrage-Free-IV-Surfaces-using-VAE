"""
Test Single Heston Calibration Setup

Tests the calibration on a single surface to verify everything works.
"""

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import json
import pickle
import numpy as np
from heston_model_ql import HestonModelQL

print("=" * 80)
print("TESTING SINGLE HESTON CALIBRATION SETUP")
print("=" * 80)

# Load config
print("\n1. Loading configuration...")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)
print("    Config loaded")

# Load data
print("\n2. Loading NIFTY surfaces...")
data_file = os.path.join(script_dir, config['input']['surfaces_file'])

with open(data_file, 'rb') as f:
    nifty_data = pickle.load(f)

surfaces = nifty_data['surfaces']
spot_prices = nifty_data['spot_prices']
dates = sorted(surfaces.keys())
test_date = dates[0]
test_surface = surfaces[test_date]
S = spot_prices[test_date]

print(f"    Data loaded")
print(f"   Test date: {test_date}")
print(f"   Spot price: {S:.2f}")
print(f"   Surface shape: {test_surface['IV_surface'].shape}")

# Test data extraction
print("\n3. Testing data extraction...")

IV_data = test_surface['IV_surface']
logm_grid = test_surface['logm_grid']
T_grid = test_surface['T_grid']

# Collect all valid data
all_logm = []
all_tau = []
all_IV = []

for mat_idx in range(len(T_grid)):
    IV_current = IV_data[mat_idx, :]
    valid_mask = ~np.isnan(IV_current) & (IV_current > 0) & (IV_current < 2.0)
    
    if np.sum(valid_mask) < 3:
        continue
    
    valid_logm = logm_grid[valid_mask]
    valid_IV = IV_current[valid_mask]
    tau_current = T_grid[mat_idx]
    
    all_logm.extend(valid_logm)
    all_tau.extend([tau_current] * len(valid_logm))
    all_IV.extend(valid_IV)

print(f"    Total data points: {len(all_IV)}")
print(f"    Maturities: {len(T_grid)}")
print(f"    IV range: [{min(all_IV):.4f}, {max(all_IV):.4f}]")

# Test Heston model creation
print("\n4. Testing Heston model...")

test_params = {
    'kappa': 2.0,
    'theta': 0.09,
    'sigma_v': 0.3,
    'rho': -0.7,
    'v0': 0.09
}

model = HestonModelQL(
    kappa=test_params['kappa'],
    theta=test_params['theta'],
    sigma_v=test_params['sigma_v'],
    rho=test_params['rho'],
    v0=test_params['v0'],
    r=config['calibration']['r'],
    q=config['calibration']['q']
)

print(f"    Model created")

# Test pricing
K_test = S * np.exp(0.0)  # ATM
tau_test = T_grid[0]

try:
    price = model.price_call(S, K_test, tau_test)
    print(f"    Test price (ATM, τ={tau_test:.3f}): {price:.4f}")
except Exception as e:
    print(f"    Pricing failed: {e}")

# Test objective function components
print("\n5. Testing objective function components...")

from scipy.stats import norm

def compute_option_prices(IV, log_moneyness, tau, S, r, q):
    K = S * np.exp(log_moneyness)
    F = S * np.exp((r - q) * tau)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(F/K) + 0.5 * IV**2 * tau) / (IV * sqrt_tau)
    d2 = d1 - IV * sqrt_tau
    call_price = np.exp(-r * tau) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return call_price

# Convert IVs to prices
all_logm = np.array(all_logm)
all_tau = np.array(all_tau)
all_IV = np.array(all_IV)
all_K = S * np.exp(all_logm)

market_prices = np.array([
    compute_option_prices(iv, lm, tau, S, config['calibration']['r'], config['calibration']['q'])
    for iv, lm, tau in zip(all_IV, all_logm, all_tau)
])

print(f"    Market prices computed")
print(f"    Price range: [{market_prices.min():.4f}, {market_prices.max():.4f}]")

# Compute model prices
model_prices = np.array([
    model.price_call(S, K, tau)
    for K, tau in zip(all_K, all_tau)
])

print(f"    Model prices computed")
print(f"    Price range: [{model_prices.min():.4f}, {model_prices.max():.4f}]")

# Compute RMSE
price_rmse = np.sqrt(np.mean((model_prices - market_prices)**2))
print(f"    Price RMSE: {price_rmse:.6f}")

# Test Feller condition
feller_satisfied = 2 * test_params['kappa'] * test_params['theta'] > test_params['sigma_v']**2
print(f"    Feller condition: {feller_satisfied}")

print("\n" + "=" * 80)
print(" SETUP TEST COMPLETE!")
print("=" * 80)
print("\nAll systems operational. Ready to run full calibration:")
print("  python run_single_heston_calibration.py")
print("=" * 80)
