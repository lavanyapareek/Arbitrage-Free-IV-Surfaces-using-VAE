"""
Single Heston Calibration with Wasserstein Penalty

Fits ONE Heston model per day across all strikes and maturities.
Uses two-stage approach: Fast calibration → Wasserstein refinement.

Output: (n_days, 5) parameter tensor ready for VAE training.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import time
import warnings
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

from heston_model_ql import HestonModelQL
import torch

print("=" * 80)
print("SINGLE HESTON CALIBRATION WITH WASSERSTEIN PENALTY")
print("One Heston model per day across all maturities")
print("=" * 80)

# ============================================================================
# 1. Load Configuration
# ============================================================================

print("\n1. Loading configuration...")

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)

print(f"   ✓ Config loaded")
print(f"   Input: {config['input']['surfaces_file']}")
print(f"   Loss: Price RMSE + {config['loss_function']['wasserstein_weight']}*Wasserstein")
print(f"   Two-stage: Fast → Wasserstein refinement")

# ============================================================================
# 2. Load NIFTY Data
# ============================================================================

print("\n2. Loading filtered NIFTY surfaces...")

data_file = config['input']['surfaces_file']
if not os.path.isabs(data_file):
    data_file = os.path.join(script_dir, data_file)

with open(data_file, 'rb') as f:
    nifty_data = pickle.load(f)

surfaces = nifty_data['surfaces']
spot_prices = nifty_data['spot_prices']
canonical_grid = nifty_data['canonical_grid']

all_dates = sorted(surfaces.keys())
dates = all_dates

log_moneyness_grid = canonical_grid['logm_grid']
tau_grid = canonical_grid['T_grid']
n_maturities = len(tau_grid)

print(f"   Total surfaces: {len(surfaces)}")
print(f"   Date range: {dates[0]} to {dates[-1]}")
print(f"   Grid: {len(log_moneyness_grid)} strikes × {n_maturities} maturities")

# ============================================================================
# 3. Single Heston Calibration Function
# ============================================================================

def compute_price_ratio(IV, log_moneyness, tau, r, q):
    """Convert IV to price ratio (normalized price)"""
    from scipy.stats import norm
    
    # Price ratio: c / (S * exp(-q*tau))
    KF = np.exp(log_moneyness - (r - q) * tau)
    sqrt_tau = np.sqrt(tau)
    d1 = (-np.log(KF) + 0.5 * IV**2 * tau) / (IV * sqrt_tau)
    d2 = d1 - IV * sqrt_tau
    
    price_ratio = norm.cdf(d1) - KF * norm.cdf(d2)
    
    return price_ratio


def compute_density_wasserstein(model, log_moneyness, tau, market_pr, r, q, n_points=50):
    """
    Compute Wasserstein distance between market and model densities.
    Uses finite differences on price ratios.
    """
    # Create finer grid for density computation
    lm_min, lm_max = log_moneyness.min(), log_moneyness.max()
    lm_fine = np.linspace(lm_min, lm_max, n_points)
    
    # Interpolate market price ratios to fine grid
    market_pr_fine = np.interp(lm_fine, log_moneyness, market_pr)
    
    # Compute model price ratios on fine grid
    try:
        model_pr_fine = model.price_ratio(lm_fine, tau)
    except:
        return np.inf
    
    if np.any(np.isnan(model_pr_fine)) or np.any(model_pr_fine < 0):
        return np.inf
    
    # Compute densities via finite differences: f(x) = exp(rT) * d²PR/dx²
    dlm = lm_fine[1] - lm_fine[0]
    
    # Central differences for second derivative
    market_density = np.zeros(n_points)
    model_density = np.zeros(n_points)
    
    for i in range(1, n_points - 1):
        d2_market = (market_pr_fine[i+1] - 2*market_pr_fine[i] + market_pr_fine[i-1]) / (dlm**2)
        d2_model = (model_pr_fine[i+1] - 2*model_pr_fine[i] + model_pr_fine[i-1]) / (dlm**2)
        
        market_density[i] = np.exp(r * tau) * d2_market
        model_density[i] = np.exp(r * tau) * d2_model
    
    # Ensure non-negative and normalize
    market_density = np.maximum(market_density, 0)
    model_density = np.maximum(model_density, 0)
    
    if np.sum(market_density) < 1e-10 or np.sum(model_density) < 1e-10:
        return np.inf
    
    market_density = market_density / np.sum(market_density)
    model_density = model_density / np.sum(model_density)
    
    # Compute Wasserstein distance
    try:
        was_dist = wasserstein_distance(lm_fine, lm_fine, market_density, model_density)
    except:
        was_dist = np.sqrt(np.sum((market_density - model_density)**2))
    
    return was_dist


def fit_single_heston(date, iv_surface, spot_price, prev_params=None):
    """
    Fit single Heston model to all strikes and maturities for one day.
    
    Two-stage approach:
    - Stage 1: Fast (no Wasserstein)
    - Stage 2: Wasserstein refinement
    """
    IV_data = iv_surface['IV_surface']
    T_grid = iv_surface['T_grid']
    logm_grid = iv_surface['logm_grid']
    S = spot_price
    
    # Extract config
    r = config['calibration']['r']
    q = config['calibration']['q']
    tol = config['calibration']['tolerance']
    max_allowed_error = config['calibration']['max_fit_error']
    
    stage1_config = config['stage1_fast']
    stage2_config = config['stage2_wasserstein']
    wass_config = config['wasserstein']
    
    # Collect all valid data points across all maturities
    all_logm = []
    all_tau = []
    all_IV = []
    all_pr = []
    
    for mat_idx in range(n_maturities):
        IV_current = IV_data[mat_idx, :]
        valid_mask = ~np.isnan(IV_current) & (IV_current > 0) & (IV_current < 2.0)
        
        if np.sum(valid_mask) < 3:
            continue
        
        valid_logm = logm_grid[valid_mask]
        valid_IV = IV_current[valid_mask]
        tau_current = T_grid[mat_idx]
        
        # Convert IV to price ratios
        pr = compute_price_ratio(valid_IV, valid_logm, tau_current, r, q)
        
        all_logm.extend(valid_logm)
        all_tau.extend([tau_current] * len(valid_logm))
        all_IV.extend(valid_IV)
        all_pr.extend(pr)
    
    if len(all_pr) < 10:
        return None
    
    all_logm = np.array(all_logm)
    all_tau = np.array(all_tau)
    all_IV = np.array(all_IV)
    all_pr = np.array(all_pr)
    
    # Initial guess
    if prev_params is not None:
        x0 = np.array([
            prev_params['kappa'],
            prev_params['theta'],
            prev_params['sigma_v'],
            prev_params['rho'],
            prev_params['v0']
        ])
    else:
        # Use ATM vol from mid-maturity for better initial guess
        mid_mat_idx = n_maturities // 2
        atm_idx = np.argmin(np.abs(logm_grid))
        atm_vol = IV_data[mid_mat_idx, atm_idx]
        if np.isnan(atm_vol) or atm_vol <= 0:
            atm_vol = 0.3
        
        # More conservative initial guess to satisfy Feller
        kappa_init = 3.0
        theta_init = atm_vol**2
        sigma_v_init = 0.5  # Lower to help satisfy Feller
        rho_init = -0.7
        v0_init = atm_vol**2
        
        x0 = np.array([kappa_init, theta_init, sigma_v_init, rho_init, v0_init])
    
    # Bounds
    bounds = [
        (1e-3, 20.0),   # kappa
        (1e-3, 2.0),    # theta
        (1e-3, 5.0),    # sigma_v
        (-0.999, 0.999), # rho
        (1e-3, 2.0)     # v0
    ]
    
    # ========================================================================
    # STAGE 1: Fast Calibration (No Wasserstein)
    # ========================================================================
    
    print(f"      [Stage 1] Fast calibration...")
    t0_stage1 = time.time()
    
    def objective_stage1(params):
        kappa, theta, sigma_v, rho, v0 = params
        
        # Create model
        model = HestonModelQL(
            kappa=kappa, theta=theta, sigma_v=sigma_v,
            rho=rho, v0=v0, r=r, q=q
        )
        
        # Compute model price ratios
        try:
            model_pr = np.array([
                model.price_ratio(lm, tau)
                for lm, tau in zip(all_logm, all_tau)
            ])
        except:
            return 1e10
        
        if np.any(np.isnan(model_pr)) or np.any(model_pr < 0):
            return 1e10
        
        # Price ratio RMSE
        pr_rmse = np.sqrt(np.mean((model_pr - all_pr)**2))
        
        # Drift penalty (ATM matching)
        atm_mask = np.abs(all_logm) < 0.05
        if np.sum(atm_mask) > 0:
            drift_penalty = np.abs(np.mean(model_pr[atm_mask]) - np.mean(all_pr[atm_mask]))
        else:
            drift_penalty = 0.0
        
        # Feller penalty (encourage 2κθ > σ_v²)
        feller_violation = max(0, sigma_v**2 - 2*kappa*theta)
        feller_penalty = 5.0 * feller_violation  # Weight: 5.0
        
        return pr_rmse + config['loss_function']['drift_penalty_weight'] * drift_penalty + feller_penalty
    
    # Try multiple random starts
    best_result_stage1 = None
    best_loss_stage1 = np.inf
    
    for attempt in range(stage1_config['num_random_starts']):
        if attempt == 0:
            x0_try = x0
        else:
            # Random perturbation
            x0_try = x0 * (1 + 0.3 * np.random.randn(5))
            x0_try = np.clip(x0_try, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            result = minimize(
                objective_stage1,
                x0_try,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': tol, 'maxiter': 500}
            )
            
            if result.fun < best_loss_stage1:
                best_loss_stage1 = result.fun
                best_result_stage1 = result
        except:
            continue
    
    if best_result_stage1 is None:
        print(f"      [Stage 1] ✗ Failed (all attempts returned None)")
        return None
    
    if not best_result_stage1.success:
        print(f"      [Stage 1] ⚠ Did not converge but using best attempt (loss: {best_loss_stage1:.6f})")
        # Continue anyway if loss is reasonable
        if best_loss_stage1 > 1.0:  # Very bad fit
            print(f"      [Stage 1] ✗ Failed (loss too high)")
            return None
    
    stage1_time = time.time() - t0_stage1
    stage1_params = best_result_stage1.x
    
    print(f"      [Stage 1] ✓ Complete in {stage1_time:.2f}s | Loss: {best_loss_stage1:.6f}")
    
    # ========================================================================
    # STAGE 2: Wasserstein Refinement
    # ========================================================================
    
    print(f"      [Stage 2] Wasserstein refinement...")
    t0_stage2 = time.time()
    
    # Counter for debugging
    stage2_calls = [0]
    
    def objective_stage2(params):
        stage2_calls[0] += 1
        kappa, theta, sigma_v, rho, v0 = params
        
        # Create model
        model = HestonModelQL(
            kappa=kappa, theta=theta, sigma_v=sigma_v,
            rho=rho, v0=v0, r=r, q=q
        )
        
        # Compute model price ratios
        try:
            model_pr = np.array([
                model.price_ratio(lm, tau)
                for lm, tau in zip(all_logm, all_tau)
            ])
        except:
            return 1e10
        
        if np.any(np.isnan(model_pr)) or np.any(model_pr < 0):
            return 1e10
        
        # Price ratio RMSE
        pr_rmse = np.sqrt(np.mean((model_pr - all_pr)**2))
        
        # Drift penalty
        atm_mask = np.abs(all_logm) < 0.05
        if np.sum(atm_mask) > 0:
            drift_penalty = np.abs(np.mean(model_pr[atm_mask]) - np.mean(all_pr[atm_mask]))
        else:
            drift_penalty = 0.0
        
        # Feller penalty (encourage 2κθ > σ_v²)
        feller_violation = max(0, sigma_v**2 - 2*kappa*theta)
        feller_penalty = 5.0 * feller_violation  # Weight: 5.0
        
        # Wasserstein penalty (average across maturities)
        wasserstein_distances = []
        for mat_idx in range(n_maturities):
            tau_current = T_grid[mat_idx]
            
            # Get market data for this maturity
            mat_mask = np.abs(all_tau - tau_current) < 1e-6
            if np.sum(mat_mask) < 3:
                continue
            
            mat_logm = all_logm[mat_mask]
            mat_pr = all_pr[mat_mask]
            
            was_dist = compute_density_wasserstein(
                model, mat_logm, tau_current, mat_pr, r, q,
                n_points=wass_config['density_grid_points']
            )
            
            if np.isfinite(was_dist):
                wasserstein_distances.append(was_dist)
        
        avg_wasserstein = np.mean(wasserstein_distances) if wasserstein_distances else 0.0
        
        total_loss = (pr_rmse + 
                     config['loss_function']['drift_penalty_weight'] * drift_penalty +
                     feller_penalty +
                     config['loss_function']['wasserstein_weight'] * avg_wasserstein)
        
        return total_loss
    
    # Use Stage 1 result as initial guess (with perturbation to encourage exploration)
    # Perturb by 5% to avoid immediate convergence
    x0_stage2 = stage1_params * (1 + 0.05 * np.random.randn(5))
    x0_stage2 = np.clip(x0_stage2, [b[0] for b in bounds], [b[1] for b in bounds])
    
    try:
        result_stage2 = minimize(
            objective_stage2,
            x0_stage2,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-5, 'maxiter': 500, 'maxfun': 1000}  # More relaxed, more iterations
        )
        
        if result_stage2.success:
            final_params = result_stage2.x
            final_loss = result_stage2.fun
            n_iterations = result_stage2.nit if hasattr(result_stage2, 'nit') else 0
            print(f"      [Stage 2] ✓ Converged in {n_iterations} iterations")
        else:
            # Fall back to Stage 1
            print(f"      [Stage 2] ⚠ Failed ({result_stage2.message}), using Stage 1 result")
            final_params = stage1_params
            final_loss = best_loss_stage1
    except Exception as e:
        print(f"      [Stage 2] ⚠ Exception: {str(e)[:50]}, using Stage 1 result")
        final_params = stage1_params
        final_loss = best_loss_stage1
    
    stage2_time = time.time() - t0_stage2
    
    print(f"      [Stage 2] ✓ Complete in {stage2_time:.2f}s | Loss: {final_loss:.6f} | Calls: {stage2_calls[0]}")
    
    # Extract final parameters
    kappa, theta, sigma_v, rho, v0 = final_params
    
    # Check Feller condition
    feller_satisfied = 2 * kappa * theta > sigma_v**2
    
    # Compute final Wasserstein distance for reporting
    final_model = HestonModelQL(
        kappa=kappa, theta=theta, sigma_v=sigma_v,
        rho=rho, v0=v0, r=r, q=q
    )
    
    final_wass_distances = []
    for mat_idx in range(n_maturities):
        tau_current = T_grid[mat_idx]
        mat_mask = np.abs(all_tau - tau_current) < 1e-6
        if np.sum(mat_mask) >= 3:
            mat_logm = all_logm[mat_mask]
            mat_pr = all_pr[mat_mask]
            was_dist = compute_density_wasserstein(
                final_model, mat_logm, tau_current, mat_pr, r, q,
                n_points=wass_config['density_grid_points']
            )
            if np.isfinite(was_dist):
                final_wass_distances.append(was_dist)
    
    avg_wass = np.mean(final_wass_distances) if final_wass_distances else 0.0
    
    # Check if fit is acceptable
    success = final_loss < max_allowed_error
    
    return {
        'date': date,
        'params': {
            'kappa': float(kappa),
            'theta': float(theta),
            'sigma_v': float(sigma_v),
            'rho': float(rho),
            'v0': float(v0)
        },
        'fit_error': float(final_loss),
        'wasserstein_distance': float(avg_wass),
        'feller_satisfied': bool(feller_satisfied),
        'success': success,
        'stage1_time': stage1_time,
        'stage2_time': stage2_time,
        'n_data_points': len(all_pr)
    }

# ============================================================================
# 4. Run Calibration
# ============================================================================

print("\n3. Running single Heston calibration...")
print("=" * 80)

all_results = []
failed_days = []
prev_params = None

start_time = time.time()
total_stage1_time = 0
total_stage2_time = 0

for day_idx, date in enumerate(tqdm(dates, desc="Calibrating")):
    surface = surfaces[date]
    spot = spot_prices[date]
    
    if config['performance']['show_timing']:
        print(f"\n[DEBUG] Processing day {day_idx+1}/{len(dates)}: {date}")
    
    result = fit_single_heston(date, surface, spot, prev_params=prev_params)
    
    if result is not None and result['success']:
        all_results.append(result)
        prev_params = result['params']
        total_stage1_time += result['stage1_time']
        total_stage2_time += result['stage2_time']
        
        print(f"\n  ✓ Day {day_idx+1}/{len(dates)} ({date}): SUCCESS")
        print(f"    Fit error: {result['fit_error']:.6f} | Wasserstein: {result['wasserstein_distance']:.6f} | Feller: {result['feller_satisfied']}")
        print(f"    Stage 1: {result['stage1_time']:.1f}s | Stage 2: {result['stage2_time']:.1f}s")
        
        if (day_idx + 1) % config['performance']['show_progress_every'] == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (day_idx + 1)
            remaining = avg_time * (len(dates) - day_idx - 1)
            success_rate = len(all_results) / (day_idx + 1) * 100
            
            print(f"\n  📊 CHECKPOINT at Day {day_idx+1}/{len(dates)}")
            print(f"    Success rate: {success_rate:.1f}% ({len(all_results)}/{day_idx+1})")
            print(f"    Avg time: {avg_time:.1f}s/day | Remaining: {remaining/60:.1f} min")
    
    elif result is not None:
        # Failed but use previous params
        failed_days.append(date)
        if config['calibration']['use_previous_day_on_failure'] and prev_params is not None:
            print(f"\n  ⚠ Day {day_idx+1}/{len(dates)} ({date}): Using previous day's params")
            all_results.append({
                'date': date,
                'params': prev_params,
                'fit_error': result['fit_error'],
                'feller_satisfied': result['feller_satisfied'],
                'success': False,
                'stage1_time': result['stage1_time'],
                'stage2_time': result['stage2_time'],
                'n_data_points': result['n_data_points'],
                'fallback': True
            })
        else:
            print(f"\n  ✗ Day {day_idx+1}/{len(dates)} ({date}): FAILED")
    else:
        failed_days.append(date)
        print(f"\n  ✗ Day {day_idx+1}/{len(dates)} ({date}): FAILED (returned None)")

total_time = time.time() - start_time

print("\n" + "=" * 80)
print(f"\n⚡ Single Heston calibration complete!")
print(f"   Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
print(f"   Successful: {len(all_results)}/{len(dates)}")
print(f"   Failed: {len(failed_days)}")
print(f"   Avg time per day: {total_time/len(dates):.1f}s")

# ============================================================================
# 5. Save Results
# ============================================================================

print("\n4. Saving results...")

# Prepare output data
output_data = {
    'dates': [r['date'] for r in all_results],
    'params': [r['params'] for r in all_results],
    'fit_errors': [r['fit_error'] for r in all_results],
    'feller_satisfied': [r['feller_satisfied'] for r in all_results],
    'calibration_settings': config,
    'timing': {
        'total_time': total_time,
        'total_stage1_time': total_stage1_time,
        'total_stage2_time': total_stage2_time,
        'avg_per_day': total_time / len(dates)
    },
    'n_successful': len(all_results),
    'n_failed': len(failed_days),
    'failed_dates': failed_days
}

output_file = os.path.join(script_dir, config['output']['pickle_file'])
with open(output_file, 'wb') as f:
    pickle.dump(output_data, f)

print(f"   ✓ Saved: {config['output']['pickle_file']}")

# ============================================================================
# 6. Prepare VAE Training Tensor
# ============================================================================

print("\n5. Preparing VAE training tensor...")

# Extract parameters in order: [kappa, theta, sigma_v, rho, v0]
param_list = []
for result in all_results:
    p = result['params']
    param_list.append([p['kappa'], p['theta'], p['sigma_v'], p['rho'], p['v0']])

param_array = np.array(param_list)
param_tensor = torch.tensor(param_array, dtype=torch.float32)

print(f"   Tensor shape: {param_tensor.shape}")
print(f"   Format: [kappa, theta, sigma_v, rho, v0]")
print(f"   Ready for VAE training!")

tensor_file = os.path.join(script_dir, config['output']['tensor_file'])
torch.save(param_tensor, tensor_file)
print(f"   ✓ Saved: {config['output']['tensor_file']}")

# ============================================================================
# 7. Visualization
# ============================================================================

print("\n6. Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fit errors
fit_errors = [r['fit_error'] for r in all_results]
axes[0].plot(fit_errors, linewidth=1.5, alpha=0.7, color='blue')
axes[0].set_ylabel('Fit Error (RMSE)', fontsize=12)
axes[0].set_xlabel('Day Index', fontsize=12)
axes[0].set_title('Single Heston Calibration: Fit Errors', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(config['calibration']['max_fit_error'], color='red', linestyle='--', label='Threshold')
axes[0].legend()

# Parameter distributions
param_names = ['kappa', 'theta', 'sigma_v', 'rho', 'v0']
for i, name in enumerate(param_names):
    values = [r['params'][name] for r in all_results]
    axes[1].hist(values, bins=30, alpha=0.5, label=name)

axes[1].set_xlabel('Parameter Value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Parameter Distributions', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(script_dir, config['output']['plot_file'])
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {config['output']['plot_file']}")

print("\n" + "=" * 80)
print("✅ SINGLE HESTON CALIBRATION COMPLETE!")
print("=" * 80)
print(f"\nOutput files:")
print(f"  - {config['output']['pickle_file']}")
print(f"  - {config['output']['tensor_file']}")
print(f"  - {config['output']['plot_file']}")
print(f"\nNext: Train VAE on 5-parameter dataset")
print("=" * 80)
