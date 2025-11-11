"""
Check term structure of calibrated Heston parameters
"""

import torch
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import sys
from tqdm import tqdm

sys.path.append('/Users/lavanyapareek/Documents/GenAI/HestonVAE')
from heston_model_ql import HestonModelQL

print("=" * 80)
print("CHECKING TERM STRUCTURE OF CALIBRATED HESTON PARAMETERS")
print("=" * 80)

# Load calibrated parameters and results
print("\n1. Loading calibrated parameters...")
params = torch.load('NIFTY_heston_single_params_tensor.pt')
print(f"    Loaded {len(params)} calibrated parameter sets")
print(f"    Shape: {params.shape}")

# Load calibration results to get spot prices and maturities
import pickle
with open('NIFTY_heston_single_params.pickle', 'rb') as f:
    calib_results = pickle.load(f)

# The pickle has: dates, params, fit_errors, etc.
# We need to load the original surface data to get spot prices and maturities
print(f"    Pickle keys: {calib_results.keys()}")

# Load the original NIFTY surface data
with open('../nifty_advanced_surfaces.pickle', 'rb') as f:
    nifty_data = pickle.load(f)

surfaces = nifty_data['surfaces']
spot_prices_dict = nifty_data['spot_prices']
dates = calib_results['dates']

# Get spot prices in the same order as calibrated params
spot_prices = [spot_prices_dict[date] for date in dates]
print(f"    Loaded {len(spot_prices)} spot prices")
print(f"    Spot range: [{min(spot_prices):.2f}, {max(spot_prices):.2f}]")

# Get maturities from canonical grid
maturities = nifty_data['canonical_grid']['T_grid']
print(f"    Maturities from data: {maturities}")

# Settings
r = 0.067
q = 0.0

print(f"\n2. Checking ATM term structure...")
print(f"   Maturities: {maturities}")
print("=" * 80)

inverted_count = 0
total_checked = 0
error_count = 0

# Check all parameter sets with progress bar
n_check = len(params)  # Check all 500

for idx in tqdm(range(n_check), desc="Checking surfaces"):
    kappa, theta, sigma_v, rho, v0 = params[idx].numpy()
    spot = spot_prices[idx]  # Use the actual spot price from calibration
    
    # Debug first parameter set
    if idx == 0:
        print(f"\n  First parameter set:")
        print(f"    kappa={kappa:.6f}, theta={theta:.6f}, sigma_v={sigma_v:.6f}, rho={rho:.6f}, v0={v0:.6f}")
        print(f"    spot={spot:.2f}")
        # Check Feller condition
        feller_lhs = 2 * kappa * theta
        feller_rhs = sigma_v ** 2
        print(f"    Feller: 2*kappa*theta={feller_lhs:.6f} vs sigma_v^2={feller_rhs:.6f} -> {' OK' if feller_lhs >= feller_rhs else ' VIOLATED'}")
    
    try:
        model = HestonModelQL(
            kappa=float(kappa), 
            theta=float(theta), 
            sigma_v=float(sigma_v), 
            rho=float(rho), 
            v0=float(v0),
            r=r, 
            q=q
        )
        
        # Use price_ratio (normalized) like the calibration does
        logm_atm = 0.0  # ATM
        ivs = []
        
        for tau in maturities:
            try:
                # Get price ratio (normalized price)
                price_ratio = model.price_ratio(logm_atm, tau)
                
                if idx == 0:
                    print(f"      tau={tau:.2f}: price_ratio={price_ratio:.4f}")
                
                # Convert price ratio back to IV using BS formula
                # price_ratio = c / (S * exp(-q*tau))
                # For ATM (logm=0, K=S): price_ratio = N(d1) - exp(-r*tau)*N(d2)
                # where d1 = (r-q+0.5*sigma^2)*tau / (sigma*sqrt(tau))
                
                def bs_price_ratio(sigma):
                    if sigma <= 0:
                        return 1e10
                    sqrt_tau = np.sqrt(tau)
                    d1 = ((r - q + 0.5*sigma**2)*tau) / (sigma*sqrt_tau)
                    d2 = d1 - sigma*sqrt_tau
                    pr = norm.cdf(d1) - np.exp(-r*tau)*norm.cdf(d2)
                    return (pr - price_ratio)**2
                
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(bs_price_ratio, bounds=(0.01, 3.0), method='bounded')
                iv = result.x
                ivs.append(iv)
            except Exception as e:
                if idx == 0:
                    print(f"      tau={tau:.2f}: ERROR - {e}")
                ivs.append(np.nan)
        
        valid_count = len([iv for iv in ivs if not np.isnan(iv)])
        
        if valid_count == len(maturities):
            total_vars = [iv**2 * tau for iv, tau in zip(ivs, maturities)]
            is_inverted = any(total_vars[i+1] < total_vars[i] for i in range(len(total_vars)-1))
            
            if is_inverted:
                inverted_count += 1
                status = ' INVERTED'
            else:
                status = ' OK'
            
            total_checked += 1
            
            # Print details for first one only
            if idx == 0:
                print(f"\n  Set {idx}: kappa={kappa:.3f}, theta={theta:.4f}, sigma_v={sigma_v:.3f}, rho={rho:.3f}, v0={v0:.4f}")
                print(f"    ATM IVs: {' -> '.join([f'{iv:.3f}' for iv in ivs])} {status}")
                print(f"    Total Var: {' -> '.join([f'{tv:.4f}' for tv in total_vars])}")
        else:
            error_count += 1
            if idx < 3:
                print(f"\n  Set {idx}: ERROR - Only {valid_count}/{len(maturities)} IVs computed")
            
    except Exception as e:
        error_count += 1
        if idx < 3:
            print(f"\n  Set {idx}: EXCEPTION - {e}")

print("\n" + "=" * 80)
print("RESULTS:")
print("=" * 80)

if total_checked > 0:
    print(f"\n Successfully checked: {total_checked}/{n_check} surfaces")
    print(f" Inverted term structure: {inverted_count}/{total_checked} ({inverted_count/total_checked*100:.1f}%)")
    print(f" Proper term structure: {total_checked - inverted_count}/{total_checked} ({(total_checked - inverted_count)/total_checked*100:.1f}%)")
    
    if error_count > 0:
        print(f"\n Errors: {error_count}/{n_check}")
    
    if inverted_count > 0:
        print(f"\n WARNING: {inverted_count} calibrated parameter sets produce INVERTED term structures!")
        print("   This means the calibration data itself may have issues.")
    else:
        print("\n All calibrated parameters produce proper term structures!")
else:
    print("\n No valid surfaces could be checked")
    print(f"   Errors: {error_count}/{n_check}")

print("\n" + "=" * 80)
