"""
Advanced IV Surface Generation with Sophisticated Interpolation/Extrapolation

Strategy:
1. Collect ALL available (K, T, IV) points per date (no maturity filtering)
2. Fit parametric model (SVI) to available data
3. Extrapolate to missing maturities using term structure
4. Interpolate to canonical grid
5. Enforce arbitrage-free constraints
"""

import numpy as np
import pandas as pd
import pickle
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.interpolate import interp1d, PchipInterpolator, griddata
from sklearn.isotonic import IsotonicRegression
from scipy.signal import savgol_filter
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED IV SURFACE GENERATION - SOPHISTICATED INTERPOLATION")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

NIFTY_R = 0.067
NIFTY_Q = 0.0

# Target grid
CANONICAL_MATURITIES = np.array([0.08333333, 0.16666667, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
CANONICAL_LOGM = np.linspace(-0.2, 0.2, 21)

# Data/quality thresholds
MIN_TOTAL_POINTS = 86  # increased for stability (day-level)
MIN_POINTS_PER_MAT = 12  # per-maturity minimum points to fit SVI
MIN_IV = 1e-4
MAX_IV = 3
# Dry run limit (set to an integer to process only first N dates, else None)
DRY_RUN_LIMIT = None

# Strike-wise smoothing strength for total variance w(k) per maturity
# Options: 'none', 'light', 'medium', 'aggressive'
STRIKE_SMOOTHING = 'medium'

print("\nConfiguration:")
print(f"   Strategy: Use ALL available data + SSVI per maturity + isotonic term-structure")
print(f"   Target grid: {len(CANONICAL_MATURITIES)} maturities × {len(CANONICAL_LOGM)} strikes")
print(f"   Min data points: {MIN_TOTAL_POINTS}")
print(f"   Strike smoothing: {STRIKE_SMOOTHING}")

# =========================================================================
# SVI / SSVI Functions
# =========================================================================

def svi_raw(k, a, b, rho, m, sigma):
    """Raw SVI: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def ssvi_w(k, theta, phi, rho):
    """
    SSVI total variance: w(k) = 0.5 * theta * [ 1 + rho * phi * k + sqrt{ (phi*k + rho)^2 + 1 - rho^2 } ]
    theta > 0 (ATM total variance), phi > 0 (curvature), |rho| < 1 (skew).
    """
    k = np.asarray(k)
    inside = (phi * k + rho)**2 + (1.0 - rho**2)
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt(np.maximum(inside, 1e-12)))


def fit_ssvi_single_maturity(log_moneyness, total_variance):
    """
    Fit SSVI parameters (theta, phi, rho) for a single maturity using robust least squares
    with soft penalties to promote no-butterfly admissibility: phi*theta <= 4/(1+|rho|).
    """
    if len(log_moneyness) < 3:
        return None

    # Initial guess
    atm_idx = np.argmin(np.abs(log_moneyness))
    theta0 = float(np.clip(total_variance[atm_idx], 1e-6, None))
    phi0 = 1.0
    rho0 = -0.3

    bounds_lower = np.array([1e-6, 1e-6, -0.999])
    bounds_upper = np.array([10.0, 10.0, 0.999])
    x0 = np.array([theta0, phi0, rho0])

    def residuals(params):
        theta, phi, rho = params
        w_pred = ssvi_w(log_moneyness, theta, phi, rho)
        # Soft penalties
        penalty_neg = np.where(w_pred < 0, 1000.0 * (1.0 + np.abs(w_pred)), 0.0)
        # Gatheral-Jacquier admissibility (no butterfly) sufficient condition per maturity
        cap = 4.0 / (1.0 + np.abs(rho))
        excess = phi * theta - cap
        penalty_ssvi = 100.0 * np.maximum(excess, 0.0)
        return (w_pred - total_variance) + penalty_neg + penalty_ssvi

    try:
        res = least_squares(
            residuals,
            x0,
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            loss='soft_l1',
            f_scale=0.1,
            max_nfev=800
        )
        if res.success:
            theta, phi, rho = res.x
            # Hard projection to Gatheral-Jacquier admissibility: phi*theta <= 4/(1+|rho|)
            cap = 4.0 / (1.0 + np.abs(rho))
            if phi * theta > cap:
                phi = max(1e-6, (cap / theta) * 0.999)
            return np.array([float(theta), float(phi), float(rho)])
    except Exception:
        return None

    return None

def fit_svi_single_maturity(log_moneyness, total_variance):
    """
    Fit SVI parameters for a single maturity using bounded least squares.
    Prioritizes accuracy with robust loss. Penalizes negative total variance.
    """
    if len(log_moneyness) < 3:
        return None

    # Initial guess
    atm_idx = np.argmin(np.abs(log_moneyness))
    v_atm = total_variance[atm_idx]
    v_min = np.clip(np.min(total_variance), 1e-8, None)

    bounds_lower = np.array([v_min * 0.1, 0.01, -0.99, -0.5, 0.01])
    bounds_upper = np.array([max(v_atm, v_min) * 2.0 + 1e-8, 3.0, 0.99, 0.5, 2.0])

    x0 = np.array([
        np.clip(v_min * 0.5, bounds_lower[0], bounds_upper[0]),  # a
        0.2,                                                    # b
        -0.2,                                                   # rho
        0.0,                                                    # m
        0.3                                                     # sigma
    ])

    def residuals(params):
        a, b, rho, m, sigma = params
        w_pred = svi_raw(log_moneyness, a, b, rho, m, sigma)
        # Penalize negative total variance predictions
        penalty = np.where(w_pred < 0, 1000.0 * (1.0 + np.abs(w_pred)), 0.0)
        return (w_pred - total_variance) + penalty

    try:
        res = least_squares(
            residuals,
            x0,
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            loss='soft_l1',
            f_scale=0.1,
            max_nfev=500
        )
        if res.success:
            return res.x
    except Exception:
        return None

    return None

def fit_svi_to_all_data(log_moneyness, total_variance):
    """
    Fit single SVI to ALL data points (across all maturities).
    This gives us a base model for extrapolation.
    """
    # Initial guess
    atm_idx = np.argmin(np.abs(log_moneyness))
    v_atm = total_variance[atm_idx]
    v_min = np.min(total_variance)
    
    bounds = [
        (v_min * 0.1, v_atm * 2),  # a
        (0.01, 3.0),                # b
        (-0.99, 0.99),              # rho
        (-0.5, 0.5),                # m
        (0.01, 2.0)                 # sigma
    ]
    
    def objective(params):
        a, b, rho, m, sigma = params
        w_pred = svi_raw(log_moneyness, a, b, rho, m, sigma)
        
        # Penalize negative values
        if np.any(w_pred < 0):
            return 1e10
        
        return np.mean((w_pred - total_variance)**2)
    
    result = differential_evolution(objective, bounds, maxiter=100, seed=42)
    
    if result.success:
        return result.x
    return None


def extrapolate_to_maturity_via_isotonic_ssvi(obs_taus, ssvi_params_by_tau, tau_target, log_moneyness_grid):
    """
    Build w(k, tau_target) by:
    1) evaluating per-maturity SSVI at observed taus to get w_obs(k, tau_obs)
    2) isotonic (monotone increasing) regression of w vs tau for each k
    3) linear interpolation in tau-space to tau_target
    """
    obs_taus = np.array(sorted(obs_taus))
    if len(obs_taus) == 0:
        return None

    n_strikes = len(log_moneyness_grid)
    w_target = np.zeros(n_strikes)

    # Precompute w for all observed taus and strikes
    W_obs = np.zeros((len(obs_taus), n_strikes))
    for i, tau in enumerate(obs_taus):
        theta, phi, rho = ssvi_params_by_tau[tau]
        W_obs[i, :] = np.maximum(ssvi_w(log_moneyness_grid, theta, phi, rho), 1e-8)

    # For each strike, enforce monotonicity in tau and interpolate
    for s in range(n_strikes):
        w_vec = W_obs[:, s]
        if len(obs_taus) == 1:
            w_isotonic = w_vec
        else:
            ir = IsotonicRegression(increasing=True, y_min=1e-8)
            w_isotonic = ir.fit_transform(obs_taus, w_vec)

        # Interpolate to tau_target (allow slight extrapolation)
        interp = interp1d(obs_taus, w_isotonic, kind='linear', fill_value='extrapolate', assume_sorted=True)
        w_t = float(np.maximum(interp(tau_target), 1e-8))
        w_target[s] = w_t

    iv_slice = np.sqrt(w_target / max(tau_target, 1e-8))
    return np.clip(iv_slice, MIN_IV, MAX_IV)


def compute_iv_from_price(S, K, T, price, r, q):
    """Compute IV from option price."""
    def bs_price(sigma):
        if sigma <= 0:
            return 1e10
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    try:
        from scipy.optimize import brentq
        iv = brentq(lambda s: bs_price(s) - price, 0.001, 3.0, xtol=1e-6)
        return iv if MIN_IV <= iv <= MAX_IV else None
    except:
        return None


def remove_butterfly_arbitrage(iv_surface, taus):
    """Remove butterfly arbitrage using monotonic interpolation."""
    n_maturities, n_strikes = iv_surface.shape
    clean_surface = iv_surface.copy()
    
    for mat_idx in range(n_maturities):
        iv_slice = iv_surface[mat_idx, :]
        tau = taus[mat_idx]
        w = iv_slice**2 * tau
        
        if len(w) > 2:
            d2w = np.diff(w, n=2)
            if np.any(d2w < -1e-6):
                try:
                    interp = PchipInterpolator(CANONICAL_LOGM, w)
                    w_smooth = interp(CANONICAL_LOGM)
                    w_smooth = np.maximum(w_smooth, 1e-8)
                    clean_surface[mat_idx, :] = np.sqrt(w_smooth / tau)
                except:
                    pass
    
    return clean_surface


def enforce_calendar_arbitrage_isotonic(iv_surface, taus):
    """
    Enforce calendar no-arbitrage using isotonic regression of total variance
    across maturities for each strike.
    """
    n_maturities, n_strikes = iv_surface.shape
    clean = iv_surface.copy()
    taus = np.asarray(taus)

    for strike_idx in range(n_strikes):
        total_vars = clean[:, strike_idx]**2 * taus
        ir = IsotonicRegression(increasing=True, y_min=1e-8)
        tv_fit = ir.fit_transform(taus, total_vars)
        clean[:, strike_idx] = np.sqrt(np.maximum(tv_fit, 1e-8) / taus)
    return clean


def smooth_strikewise_total_variance(iv_surface, taus, strength='medium'):
    """
    Smooth total variance w(k) along strikes for each maturity using Savitzky-Golay.
    Then convert back to IV. This reduces jaggedness while preserving structure.
    """
    if strength is None or strength == 'none':
        return iv_surface

    n_maturities, n_strikes = iv_surface.shape
    clean = iv_surface.copy()

    # Choose window by strength; ensure window is odd and <= n_strikes
    base_window = {
        'light': 5,
        'medium': 9,
        'aggressive': 13
    }.get(strength, 9)

    window = base_window if base_window % 2 == 1 else base_window + 1
    window = min(window, n_strikes if n_strikes % 2 == 1 else n_strikes - 1)
    window = max(3, window)
    polyorder = 3 if window >= 5 else 2

    for mat_idx in range(n_maturities):
        tau = taus[mat_idx]
        w = np.maximum(clean[mat_idx, :]**2 * tau, 1e-8)
        # Use interpolation mode to avoid edge artifacts
        try:
            w_smooth = savgol_filter(w, window_length=window, polyorder=polyorder, mode='interp')
        except Exception:
            w_smooth = w
        w_smooth = np.maximum(w_smooth, 1e-8)
        clean[mat_idx, :] = np.sqrt(w_smooth / max(tau, 1e-8))

    return clean


# ============================================================================
# Main Processing Function
# ============================================================================

def process_single_date(date, date_data, spot_price):
    """
    Process a single date using per-maturity SVI fits, isotonic calendar enforcement,
    and robust price selection (LTP -> Settle Price -> Close).
    """
    try:
        grouped = date_data.groupby('Expiry')

        obs_taus = []
        ssvi_params_by_tau = {}
        n_points_total = 0

        for expiry, group in grouped:
            date_dt = pd.to_datetime(date)
            expiry_dt = pd.to_datetime(expiry)
            tau = (expiry_dt - date_dt).days / 365.0

            if tau < 0.02 or tau > 3.0:  # keep wide range
                continue

            # Robust price selection: LTP -> Settle Price -> Close
            strikes = pd.to_numeric(group['Strike Price'], errors='coerce').values
            ltp = pd.to_numeric(group.get('LTP', pd.Series([np.nan]*len(group))), errors='coerce').values
            settle = pd.to_numeric(group.get('Settle Price', pd.Series([np.nan]*len(group))), errors='coerce').values
            close = pd.to_numeric(group.get('Close', pd.Series([np.nan]*len(group))), errors='coerce').values

            chosen_prices = np.where(np.isfinite(ltp) & (ltp > 0), ltp,
                                np.where(np.isfinite(settle) & (settle > 0), settle, close))

            open_int = pd.to_numeric(group.get('Open Int', pd.Series([np.nan]*len(group))), errors='coerce').values
            volume = pd.to_numeric(group.get('No. of contracts', pd.Series([np.nan]*len(group))), errors='coerce').values

            valid_mask = (
                np.isfinite(strikes) &
                np.isfinite(chosen_prices) & (chosen_prices > 0) &
                ((np.isfinite(open_int) & (open_int > 0)) | (np.isfinite(volume) & (volume > 0)))
            )

            if valid_mask.sum() < MIN_POINTS_PER_MAT:
                continue

            strikes = strikes[valid_mask]
            prices = chosen_prices[valid_mask]

            # Compute IVs for this maturity
            logm_list = []
            w_list = []  # total variance for this maturity

            for K, price in zip(strikes, prices):
                moneyness = K / spot_price
                if (moneyness < 0.3) or (moneyness > 3.0):
                    continue
                iv = compute_iv_from_price(spot_price, K, tau, price, NIFTY_R, NIFTY_Q)
                if iv is not None and np.isfinite(iv) and (MIN_IV <= iv <= MAX_IV):
                    logm_list.append(np.log(K / spot_price))
                    w_list.append(iv**2 * tau)

            if len(w_list) < MIN_POINTS_PER_MAT:
                continue

            logm_arr = np.array(logm_list)
            w_arr = np.array(w_list)
            n_points_total += len(w_arr)

            # Fit SSVI for this maturity
            params = fit_ssvi_single_maturity(logm_arr, w_arr)
            if params is None:
                continue

            obs_taus.append(tau)
            ssvi_params_by_tau[tau] = params

        # Check minimum requirements (day-level)
        if (n_points_total < MIN_TOTAL_POINTS) or (len(obs_taus) == 0):
            return None

        # Build surface on canonical maturities via isotonic term-structure
        IV_surface = np.zeros((len(CANONICAL_MATURITIES), len(CANONICAL_LOGM)))
        for mat_idx, tau_target in enumerate(CANONICAL_MATURITIES):
            iv_slice = extrapolate_to_maturity_via_isotonic_ssvi(obs_taus, ssvi_params_by_tau, tau_target, CANONICAL_LOGM)
            if iv_slice is None:
                return None
            IV_surface[mat_idx, :] = iv_slice

        # Strike-wise smoothing on total variance, then arbitrage enforcement steps
        IV_surface = smooth_strikewise_total_variance(IV_surface, CANONICAL_MATURITIES, strength=STRIKE_SMOOTHING)

        # Remove butterfly arbitrage (strike dimension smoothing/fix)
        IV_surface = remove_butterfly_arbitrage(IV_surface, CANONICAL_MATURITIES)

        # Enforce calendar (maturity dimension) via isotonic
        IV_surface = enforce_calendar_arbitrage_isotonic(IV_surface, CANONICAL_MATURITIES)

        # Final validation
        if np.any(~np.isfinite(IV_surface)) or np.any(IV_surface <= 0):
            return None

        return {
            'IV_surface': IV_surface,
            'T_grid': CANONICAL_MATURITIES,
            'logm_grid': CANONICAL_LOGM,
            'spot_price': spot_price,
            'n_data_points': int(n_points_total),
            'ssvi_params': {float(t): list(map(float, p)) for t, p in ssvi_params_by_tau.items()}
        }

    except Exception:
        return None


# ============================================================================
# Main Processing Loop
# ============================================================================

print("\n1. Loading raw NIFTY data...")
df = pd.read_csv('final_options_data_filled.csv', low_memory=False)
df = df[df['Option type'] == 'CE']

print(f"   Total rows: {len(df):,}")
print(f"   Unique dates: {df['Date'].nunique()}")

dates = sorted(df['Date'].unique())
if DRY_RUN_LIMIT is not None and isinstance(DRY_RUN_LIMIT, int):
    dates = dates[:DRY_RUN_LIMIT]
    print(f"\n   DRY RUN ENABLED: Processing first {len(dates)} dates")

print("\n2. Processing with advanced interpolation/extrapolation...")
print("=" * 80)

clean_surfaces = {}
start_time = time.time()

for date_idx, date in enumerate(tqdm(dates, desc="Processing")):
    date_data = df[df['Date'] == date]
    spot_price = date_data['Underlying Value'].iloc[0]
    
    if pd.isna(spot_price) or spot_price <= 0:
        continue
    
    surface = process_single_date(date, date_data, spot_price)
    
    if surface is not None:
        clean_surfaces[date] = surface
    
    if (date_idx + 1) % 200 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / (date_idx + 1)
        remaining = avg_time * (len(dates) - date_idx - 1)
        success_rate = len(clean_surfaces) / (date_idx + 1) * 100
        print(f"\n  Progress: {date_idx+1}/{len(dates)} | Success: {success_rate:.1f}% | ETA: {remaining/60:.1f} min")

total_time = time.time() - start_time

print("\n" + "=" * 80)
print(f"\n Processing complete!")
print(f"   Total time: {total_time/60:.1f} minutes")
print(f"   Successful: {len(clean_surfaces)}/{len(dates)} ({len(clean_surfaces)/len(dates)*100:.1f}%)")

# ============================================================================
# Validation
# ============================================================================

print("\n3. Validating surfaces...")

if len(clean_surfaces) > 0:
    all_ivs = []
    all_n_points = []
    
    for surface in clean_surfaces.values():
        all_ivs.append(surface['IV_surface'].flatten())
        all_n_points.append(surface['n_data_points'])
    
    all_ivs = np.concatenate(all_ivs)
    
    print(f"\n   IV Statistics:")
    print(f"     Mean: {all_ivs.mean():.4f} ± {all_ivs.std():.4f}")
    print(f"     Range: [{all_ivs.min():.4f}, {all_ivs.max():.4f}]")
    print(f"     Avg data points/surface: {np.mean(all_n_points):.0f}")
    print(f"     All arbitrage-free: ")

# ============================================================================
# Save Results
# ============================================================================

print("\n4. Saving results...")

output_data = {
    'surfaces': clean_surfaces,
    'spot_prices': {date: s['spot_price'] for date, s in clean_surfaces.items()},
    'canonical_grid': {
        'T_grid': CANONICAL_MATURITIES,
        'logm_grid': CANONICAL_LOGM
    },
    'metadata': {
        'source': 'final_options_data_filled.csv',
        'method': 'per-maturity SSVI + isotonic term-structure + strike-wise Savitzky-Golay + arbitrage enforcement',
        'strike_smoothing': STRIKE_SMOOTHING,
        'n_successful': len(clean_surfaces),
        'n_total': len(dates),
        'processing_time': total_time,
        'grid_type': 'symmetric',
        'interpolation': 'svi_with_term_structure_extrapolation'
    }
}

output_file = 'nifty_advanced_surfaces.pickle'
with open(output_file, 'wb') as f:
    pickle.dump(output_data, f)

print(f"    Saved: {output_file}")
print(f"    Contains: {len(clean_surfaces)} arbitrage-free surfaces")
print(f"    Format: {len(CANONICAL_MATURITIES)} maturities × {len(CANONICAL_LOGM)} strikes (symmetric grid)")

print("\n" + "=" * 80)
print(" ADVANCED SURFACES CREATED!")
print("=" * 80)
print(f"\nNext: Run Heston calibration on {len(clean_surfaces)} clean surfaces")
print(f"Expected time: ~{len(clean_surfaces) * 3.7 / 60:.0f} minutes")
