"""
Generate IV Surfaces from Regime-Sampled Heston Parameters
Categorizes surfaces by market regime (low/moderate/high volatility)
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
from tqdm import tqdm
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heston_model_ql import HestonModelQL

# Set style
sns.set_style("whitegrid")

print("\n" + "="*80)
print("GENERATE REGIME-SPECIFIC IV SURFACES")
print("="*80)

# ============================================================================
# 1. Parse Arguments
# ============================================================================

parser = argparse.ArgumentParser(description='Generate IV surfaces from regime-sampled parameters')
parser.add_argument('--input', type=str, 
                   default='results/regime_samples/all_regimes_heston_params.pt',
                   help='Input file with regime parameters')
parser.add_argument('--regimes', type=str, nargs='+', default=['low_volatility', 'crisis', 'high_volatility'],
                   help='Regimes to process')
parser.add_argument('--output_dir', type=str, default='regime_surfaces',
                   help='Output directory')
parser.add_argument('--spot', type=float, default=21000.0,
                   help='Spot price')
parser.add_argument('--r', type=float, default=0.067,
                   help='Risk-free rate')
parser.add_argument('--q', type=float, default=0.0,
                   help='Dividend yield')
parser.add_argument('--max_samples', type=int, default=None,
                   help='Maximum samples per regime (None = all)')

args = parser.parse_args()

# ============================================================================
# 2. Define Grid
# ============================================================================

print("\n1. Setting up grid...")

# Maturity grid (8 maturities in years) - MUST match training data grid
T_grid = np.array([0.08333333, 0.16666667, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])  # 1M, 2M, 3M, 6M, 9M, 12M, 18M, 24M
n_maturities = len(T_grid)

# Log-moneyness grid (21 strikes) - MUST match training data grid
logm_grid = np.linspace(-0.2, 0.2, 21)  # 21 strikes from -20% to +20%
n_strikes = len(logm_grid)

print(f"   ✓ Maturities: {n_maturities} ({T_grid[0]:.3f} to {T_grid[-1]:.3f} years)")
print(f"   ✓ Log-moneyness: {n_strikes} ({logm_grid[0]:.2f} to {logm_grid[-1]:.2f})")
print(f"   ✓ Output shape per surface: ({n_maturities}, {n_strikes})")

# ============================================================================
# 3. Load Regime Parameters
# ============================================================================

print("\n2. Loading regime parameters...")

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, args.input)

if not os.path.exists(input_path):
    print(f"   ✗ File not found: {input_path}")
    print(f"   Run sample_regimes.py first!")
    exit(1)

data = torch.load(input_path, weights_only=False)
available_regimes = list(data['regimes'].keys())

print(f"   ✓ Loaded: {input_path}")
print(f"   Available regimes: {', '.join(available_regimes)}")

# Filter regimes
regimes_to_process = [r for r in args.regimes if r in available_regimes]
if not regimes_to_process:
    print(f"   ✗ No valid regimes found!")
    exit(1)

print(f"   Processing: {', '.join(regimes_to_process)}")

# Create output directory
output_dir = os.path.join(script_dir, 'results', args.output_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"   Output: {output_dir}")

# ============================================================================
# 4. Helper Function: Generate IV Surface
# ============================================================================

def generate_iv_surface(params, T_grid, logm_grid, spot, r, q):
    """
    Generate IV surface for single parameter set
    
    Args:
        params: (5,) array [kappa, theta, sigma_v, rho, v0]
        
    Returns:
        iv_surface: (n_maturities, n_strikes) array
        success: bool
    """
    kappa, theta, sigma_v, rho, v0 = params
    
    iv_surface = np.zeros((len(T_grid), len(logm_grid)))
    
    try:
        # Create Heston model
        model = HestonModelQL(
            kappa=float(kappa),
            theta=float(theta),
            sigma_v=float(sigma_v),
            rho=float(rho),
            v0=float(v0),
            r=r,
            q=q
        )
    except Exception as e:
        return None, False
    
    # Generate IV for each maturity and strike
    for mat_idx, tau in enumerate(T_grid):
        for strike_idx, logm in enumerate(logm_grid):
            try:
                # Get price ratio from Heston model
                price_ratio = model.price_ratio(logm, tau)
                
                # Convert to absolute price
                heston_price = price_ratio * spot * np.exp(-r * tau)
                
                # Skip if price too small
                if heston_price < 1e-10 or np.isnan(price_ratio):
                    iv_surface[mat_idx, strike_idx] = np.nan
                    continue
                
                # Convert log-moneyness to strike
                K = spot * np.exp(logm)
                
                # Black-Scholes call price function
                def bs_call_price(sigma):
                    if sigma <= 0 or tau <= 0:
                        return 1e10
                    d1 = (logm + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
                    d2 = d1 - sigma * np.sqrt(tau)
                    return spot * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
                
                def objective(sigma):
                    return bs_call_price(sigma) - heston_price
                
                # Find IV
                try:
                    iv = brentq(objective, 0.001, 3.0, xtol=1e-6, maxiter=100)
                    
                    # Sanity check
                    if np.isnan(iv) or iv <= 0 or iv > 3.0:
                        iv = np.nan
                    
                    iv_surface[mat_idx, strike_idx] = iv
                except:
                    iv_surface[mat_idx, strike_idx] = np.nan
                
            except Exception as e:
                iv_surface[mat_idx, strike_idx] = np.nan
    
    # Check if surface is valid
    success = not np.isnan(iv_surface).any()
    
    return iv_surface, success

# ============================================================================
# 4.5. Arbitrage Checking Functions
# ============================================================================

def check_static_arbitrage(iv_surface, T_grid, logm_grid, spot, r, q):
    """
    Check static arbitrage: Call prices must be decreasing in strike
    
    Returns:
        violations: list of (mat_idx, strike_idx) tuples with violations
    """
    violations = []
    n_mat, n_strike = iv_surface.shape
    strikes = spot * np.exp(logm_grid)
    
    for mat_idx in range(n_mat):
        tau = T_grid[mat_idx]
        
        # Compute call prices for this maturity
        prices = []
        for strike_idx in range(n_strike):
            iv = iv_surface[mat_idx, strike_idx]
            if np.isnan(iv) or iv <= 0:
                return None  # Invalid surface
            
            K = strikes[strike_idx]
            logm = logm_grid[strike_idx]
            
            # Black-Scholes call price
            d1 = (logm + (r - q + 0.5 * iv**2) * tau) / (iv * np.sqrt(tau))
            d2 = d1 - iv * np.sqrt(tau)
            price = spot * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
            prices.append(price)
        
        # Check monotonicity: prices[i] >= prices[i+1]
        for i in range(n_strike - 1):
            if prices[i] < prices[i+1] - 1e-6:  # Allow small numerical error
                violations.append((mat_idx, i))
    
    return violations


def check_calendar_arbitrage(iv_surface, T_grid, logm_grid, spot, r, q):
    """
    Check calendar arbitrage: Call prices must be increasing in maturity (for same strike)
    
    Returns:
        violations: list of (mat_idx, strike_idx) tuples with violations
    """
    violations = []
    n_mat, n_strike = iv_surface.shape
    strikes = spot * np.exp(logm_grid)
    
    for strike_idx in range(n_strike):
        K = strikes[strike_idx]
        logm = logm_grid[strike_idx]
        
        # Compute call prices for this strike across maturities
        prices = []
        for mat_idx in range(n_mat):
            tau = T_grid[mat_idx]
            iv = iv_surface[mat_idx, strike_idx]
            
            if np.isnan(iv) or iv <= 0:
                return None  # Invalid surface
            
            # Black-Scholes call price
            d1 = (logm + (r - q + 0.5 * iv**2) * tau) / (iv * np.sqrt(tau))
            d2 = d1 - iv * np.sqrt(tau)
            price = spot * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
            prices.append(price)
        
        # Check monotonicity: prices[i] <= prices[i+1]
        for i in range(n_mat - 1):
            if prices[i] > prices[i+1] + 1e-6:  # Allow small numerical error
                violations.append((i, strike_idx))
    
    return violations


def check_all_arbitrage(iv_surface, T_grid, logm_grid, spot, r, q):
    """
    Check static and calendar arbitrage
    
    Returns:
        dict with violation counts and details
    """
    static_viols = check_static_arbitrage(iv_surface, T_grid, logm_grid, spot, r, q)
    calendar_viols = check_calendar_arbitrage(iv_surface, T_grid, logm_grid, spot, r, q)
    
    # Check if any check returned None (invalid surface)
    if static_viols is None or calendar_viols is None:
        return None
    
    return {
        'static_violations': len(static_viols),
        'calendar_violations': len(calendar_viols),
        'total_violations': len(static_viols) + len(calendar_viols),
        'is_arbitrage_free': (len(static_viols) == 0 and len(calendar_viols) == 0)
    }


# ============================================================================
# 5. Generate Surfaces for Each Regime
# ============================================================================

print(f"\n3. Generating IV surfaces...")
print("="*80)

regime_surfaces = {}
regime_stats = {}

for regime_name in regimes_to_process:
    print(f"\n{regime_name.upper().replace('_', ' ')}")
    
    # Get parameters for this regime
    params = data['regimes'][regime_name].numpy()
    n_samples = len(params)
    
    # Limit samples if requested
    if args.max_samples and n_samples > args.max_samples:
        params = params[:args.max_samples]
        n_samples = args.max_samples
    
    print(f"  Generating {n_samples} surfaces...")
    
    # Storage
    surfaces = []
    valid_count = 0
    arbitrage_stats = {
        'static_violations': [],
        'calendar_violations': [],
        'arbitrage_free_count': 0
    }
    
    # Generate with progress bar
    for idx in tqdm(range(n_samples), desc=f"  {regime_name}", leave=False):
        iv_surface, success = generate_iv_surface(
            params[idx], T_grid, logm_grid, args.spot, args.r, args.q
        )
        
        if success:
            surfaces.append(iv_surface)
            valid_count += 1
            
            # Check for arbitrage violations
            arb_check = check_all_arbitrage(iv_surface, T_grid, logm_grid, args.spot, args.r, args.q)
            if arb_check is not None:
                arbitrage_stats['static_violations'].append(arb_check['static_violations'])
                arbitrage_stats['calendar_violations'].append(arb_check['calendar_violations'])
                if arb_check['is_arbitrage_free']:
                    arbitrage_stats['arbitrage_free_count'] += 1
    
    if valid_count > 0:
        surfaces_array = np.array(surfaces)
        regime_surfaces[regime_name] = surfaces_array
        
        # Compute arbitrage statistics
        arb_free_pct = (arbitrage_stats['arbitrage_free_count'] / valid_count * 100) if valid_count > 0 else 0
        static_mean = np.mean(arbitrage_stats['static_violations']) if arbitrage_stats['static_violations'] else 0
        calendar_mean = np.mean(arbitrage_stats['calendar_violations']) if arbitrage_stats['calendar_violations'] else 0
        
        # Compute statistics
        regime_stats[regime_name] = {
            'n_total': n_samples,
            'n_valid': valid_count,
            'success_rate': valid_count / n_samples * 100,
            'atm_iv_mean': surfaces_array[:, :, 10].mean(axis=0),  # ATM is middle strike
            'atm_iv_std': surfaces_array[:, :, 10].std(axis=0),
            'mean_surface': surfaces_array.mean(axis=0),
            'std_surface': surfaces_array.std(axis=0),
            'median_surface': np.median(surfaces_array, axis=0),
            'p5_surface': np.percentile(surfaces_array, 5, axis=0),
            'p95_surface': np.percentile(surfaces_array, 95, axis=0),
            'arbitrage_free_count': arbitrage_stats['arbitrage_free_count'],
            'arbitrage_free_pct': arb_free_pct,
            'avg_static_violations': static_mean,
            'avg_calendar_violations': calendar_mean
        }
        
        print(f"  ✓ Valid surfaces: {valid_count}/{n_samples} ({regime_stats[regime_name]['success_rate']:.1f}%)")
        print(f"  ATM IV (1M): {regime_stats[regime_name]['atm_iv_mean'][0]:.2%}")
        print(f"  ATM IV (24M): {regime_stats[regime_name]['atm_iv_mean'][-1]:.2%}")
        print(f"  Arbitrage-free: {arbitrage_stats['arbitrage_free_count']}/{valid_count} ({arb_free_pct:.1f}%)")
        print(f"    Static violations (avg): {static_mean:.2f}")
        print(f"    Calendar violations (avg): {calendar_mean:.2f}")
    else:
        print(f"  ✗ No valid surfaces generated!")

# ============================================================================
# 6. Save Results
# ============================================================================

print(f"\n4. Saving results...")

# Save individual regime surfaces
for regime_name, surfaces in regime_surfaces.items():
    regime_file = os.path.join(output_dir, f'{regime_name}_iv_surfaces.pt')
    torch.save({
        'surfaces': torch.tensor(surfaces, dtype=torch.float32),
        'T_grid': T_grid,
        'logm_grid': logm_grid,
        'statistics': regime_stats[regime_name],
        'spot': args.spot,
        'r': args.r,
        'q': args.q
    }, regime_file)
    print(f"  ✓ {regime_name}: {regime_file}")

# Save combined file
combined_file = os.path.join(output_dir, 'all_regime_surfaces.pt')
torch.save({
    'regimes': {name: torch.tensor(surf, dtype=torch.float32) 
                for name, surf in regime_surfaces.items()},
    'T_grid': T_grid,
    'logm_grid': logm_grid,
    'statistics': regime_stats,
    'spot': args.spot,
    'r': args.r,
    'q': args.q
}, combined_file)
print(f"  ✓ Combined: {combined_file}")

# ============================================================================
# 7. Create Visualizations
# ============================================================================

print(f"\n5. Creating visualizations...")

# Plot 1: ATM Term Structure Comparison
fig, ax = plt.subplots(figsize=(12, 7))

for regime_name in regime_surfaces.keys():
    atm_mean = regime_stats[regime_name]['atm_iv_mean']
    atm_std = regime_stats[regime_name]['atm_iv_std']
    
    ax.plot(T_grid * 12, atm_mean * 100, 'o-', linewidth=2, markersize=8,
            label=regime_name.replace('_', ' ').title())
    ax.fill_between(T_grid * 12, 
                     (atm_mean - atm_std) * 100,
                     (atm_mean + atm_std) * 100,
                     alpha=0.2)

ax.set_xlabel('Maturity (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('ATM Implied Volatility (%)', fontsize=12, fontweight='bold')
ax.set_title('ATM IV Term Structure by Regime', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

term_structure_path = os.path.join(output_dir, 'atm_term_structure_comparison.png')
plt.savefig(term_structure_path, dpi=300, bbox_inches='tight')
print(f"  ✓ {term_structure_path}")
plt.close()

# Plot 2: Volatility Smile Comparison (6M maturity)
fig, ax = plt.subplots(figsize=(12, 7))

mat_idx = 3  # 6M maturity
strikes_pct = logm_grid * 100

for regime_name in regime_surfaces.keys():
    mean_surface = regime_stats[regime_name]['mean_surface']
    std_surface = regime_stats[regime_name]['std_surface']
    
    smile = mean_surface[mat_idx, :] * 100
    smile_std = std_surface[mat_idx, :] * 100
    
    ax.plot(strikes_pct, smile, 'o-', linewidth=2, markersize=6,
            label=regime_name.replace('_', ' ').title())
    ax.fill_between(strikes_pct, smile - smile_std, smile + smile_std, alpha=0.2)

ax.set_xlabel('Log-Moneyness (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
ax.set_title(f'Volatility Smile at {T_grid[mat_idx]*12:.0f}M Maturity', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)

smile_path = os.path.join(output_dir, 'volatility_smile_comparison.png')
plt.savefig(smile_path, dpi=300, bbox_inches='tight')
print(f"  ✓ {smile_path}")
plt.close()

# Plot 3: Full Surface Heatmaps
n_regimes = len(regime_surfaces)
fig, axes = plt.subplots(1, n_regimes, figsize=(6*n_regimes, 5))

if n_regimes == 1:
    axes = [axes]

for idx, regime_name in enumerate(regime_surfaces.keys()):
    ax = axes[idx]
    mean_surface = regime_stats[regime_name]['mean_surface'] * 100
    
    im = ax.contourf(logm_grid * 100, T_grid * 12, mean_surface, 
                     levels=20, cmap='RdYlBu_r')
    ax.contour(logm_grid * 100, T_grid * 12, mean_surface, 
               levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    ax.set_xlabel('Log-Moneyness (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Maturity (months)', fontsize=11, fontweight='bold')
    ax.set_title(regime_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='IV (%)')

plt.suptitle('Mean IV Surfaces by Regime', fontsize=14, fontweight='bold')
plt.tight_layout()

heatmap_path = os.path.join(output_dir, 'surface_heatmaps.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"  ✓ {heatmap_path}")
plt.close()

# Plot 4: 3D Surface Plot (first regime as example)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(18, 5))

for idx, regime_name in enumerate(regime_surfaces.keys()):
    ax = fig.add_subplot(1, n_regimes, idx+1, projection='3d')
    
    X, Y = np.meshgrid(logm_grid * 100, T_grid * 12)
    Z = regime_stats[regime_name]['mean_surface'] * 100
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    ax.set_xlabel('Log-Moneyness (%)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Maturity (months)', fontsize=9, fontweight='bold')
    ax.set_zlabel('IV (%)', fontsize=9, fontweight='bold')
    ax.set_title(regime_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    
    ax.view_init(elev=20, azim=45)

plt.suptitle('3D IV Surfaces by Regime', fontsize=14, fontweight='bold')
plt.tight_layout()

surface_3d_path = os.path.join(output_dir, 'surfaces_3d.png')
plt.savefig(surface_3d_path, dpi=300, bbox_inches='tight')
print(f"  ✓ {surface_3d_path}")
plt.close()

# Plot 5: Arbitrage Violations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Arbitrage-free percentage
ax1 = axes[0]
regime_names_list = list(regime_surfaces.keys())
arb_free_pcts = [regime_stats[r]['arbitrage_free_pct'] for r in regime_names_list]

colors = ['green' if pct > 95 else 'orange' if pct > 80 else 'red' for pct in arb_free_pcts]
bars = ax1.bar(range(len(regime_names_list)), arb_free_pcts, color=colors, alpha=0.7)

ax1.axhline(95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='95% threshold')
ax1.axhline(80, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='80% threshold')

ax1.set_xticks(range(len(regime_names_list)))
ax1.set_xticklabels([r.replace('_', '\n') for r in regime_names_list], fontsize=10)
ax1.set_ylabel('Arbitrage-Free Surfaces (%)', fontsize=12, fontweight='bold')
ax1.set_title('Arbitrage-Free Surface Percentage', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 105])
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (bar, pct) in enumerate(zip(bars, arb_free_pcts)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Right: Average violations by type
ax2 = axes[1]
violation_types = ['Static', 'Calendar']
x_pos = np.arange(len(regime_names_list))
width = 0.35

for i, viol_type in enumerate(violation_types):
    if viol_type == 'Static':
        vals = [regime_stats[r]['avg_static_violations'] for r in regime_names_list]
    elif viol_type == 'Calendar':
        vals = [regime_stats[r]['avg_calendar_violations'] for r in regime_names_list]
    
    ax2.bar(x_pos + i*width, vals, width, label=viol_type, alpha=0.7)

ax2.set_xticks(x_pos + width)
ax2.set_xticklabels([r.replace('_', '\n') for r in regime_names_list], fontsize=10)
ax2.set_ylabel('Average Violations per Surface', fontsize=12, fontweight='bold')
ax2.set_title('Average Arbitrage Violations by Type', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Arbitrage Analysis Across Regimes', fontsize=15, fontweight='bold')
plt.tight_layout()

arbitrage_path = os.path.join(output_dir, 'arbitrage_analysis.png')
plt.savefig(arbitrage_path, dpi=300, bbox_inches='tight')
print(f"  ✓ {arbitrage_path}")
plt.close()

# ============================================================================
# 8. Summary Statistics
# ============================================================================

print(f"\n6. Summary statistics:")
print("="*80)

for regime_name in regime_surfaces.keys():
    stats = regime_stats[regime_name]
    print(f"\n{regime_name.upper().replace('_', ' ')}")
    print(f"  Valid surfaces: {stats['n_valid']}/{stats['n_total']} ({stats['success_rate']:.1f}%)")
    print(f"  ATM IV range (across maturities):")
    print(f"    Mean: {stats['atm_iv_mean'].min():.2%} - {stats['atm_iv_mean'].max():.2%}")
    print(f"    Std:  {stats['atm_iv_std'].min():.2%} - {stats['atm_iv_std'].max():.2%}")
    print(f"  Arbitrage checks:")
    print(f"    Arbitrage-free: {stats['arbitrage_free_count']}/{stats['n_valid']} ({stats['arbitrage_free_pct']:.1f}%)")
    print(f"    Avg violations - Static: {stats['avg_static_violations']:.2f}, "
          f"Calendar: {stats['avg_calendar_violations']:.2f}")

print("\n" + "="*80)
print("SURFACE GENERATION COMPLETE!")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print(f"\nGenerated files:")
print(f"  - all_regime_surfaces.pt           (Combined surfaces)")
print(f"  - <regime>_iv_surfaces.pt          (Individual regime surfaces)")
print(f"  - atm_term_structure_comparison.png")
print(f"  - volatility_smile_comparison.png")
print(f"  - surface_heatmaps.png")
print(f"  - surfaces_3d.png")
print(f"  - arbitrage_analysis.png           (Arbitrage check results)")
print("\n" + "="*80)
