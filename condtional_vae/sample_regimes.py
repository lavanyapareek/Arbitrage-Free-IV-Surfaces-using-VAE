"""
Sample Heston Parameters from Conditional VAE for Different Market Regimes
Generate parameters for pre-defined regimes or custom conditioning scenarios
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from cvae_model import ConditionalVAE_SingleHeston

# Set style
sns.set_style("whitegrid")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*80)
print("CONDITIONAL HESTON PARAMETER SAMPLING")
print("="*80)

# ============================================================================
# 1. Define Market Regimes
# ============================================================================

# Pre-defined market regimes with conditioning values
MARKET_REGIMES = {
    'low_volatility': {
        'description': 'Calm market, low VIX, stable oil prices',
        'conditioning': {
            'crude_oil_30d_mean': -0.5,
            'crude_oil_7d_mean': -0.5,
            'unrest_index_yearly': -0.5,
            'crude_oil': -0.5,
            'usdinr_quarterly_mean': 0.0,
            'india_vix_30d_mean': -1.5,  # Low VIX
            'india_vix_7d_mean': -1.5,
            'us_10y_yield': 0.0
        }
    },
    'moderate_volatility': {
        'description': 'Normal market conditions',
        'conditioning': {
            'crude_oil_30d_mean': 0.0,
            'crude_oil_7d_mean': 0.0,
            'unrest_index_yearly': 0.0,
            'crude_oil': 0.0,
            'usdinr_quarterly_mean': 0.0,
            'india_vix_30d_mean': 0.0,
            'india_vix_7d_mean': 0.0,
            'us_10y_yield': 0.0
        }
    },
    'high_volatility': {
        'description': 'Elevated volatility, higher uncertainty',
        'conditioning': {
            'crude_oil_30d_mean': 0.5,
            'crude_oil_7d_mean': 0.5,
            'unrest_index_yearly': 1.5,
            'crude_oil': 0.5,
            'usdinr_quarterly_mean': 0.5,
            'india_vix_30d_mean': 2.0,  # High VIX
            'india_vix_7d_mean': 2.0,
            'us_10y_yield': 0.5
        }
    },
    'crisis': {
        'description': 'Market crisis: VIX spike, oil crash, high unrest',
        'conditioning': {
            'crude_oil_30d_mean': -1.5,
            'crude_oil_7d_mean': -1.5,
            'unrest_index_yearly': 2.0,
            'crude_oil': -2.0,
            'usdinr_quarterly_mean': 1.0,
            'india_vix_30d_mean': 3.0,  # Extreme VIX
            'india_vix_7d_mean': 3.0,
            'us_10y_yield': -1.0
        }
    },
    'oil_shock': {
        'description': 'Oil price shock (crash or spike)',
        'conditioning': {
            'crude_oil_30d_mean': -2.5,  # Major oil crash
            'crude_oil_7d_mean': -2.5,
            'unrest_index_yearly': 1.0,
            'crude_oil': -2.5,
            'usdinr_quarterly_mean': 0.5,
            'india_vix_30d_mean': 1.5,
            'india_vix_7d_mean': 1.5,
            'us_10y_yield': 0.0
        }
    },
    'geopolitical_stress': {
        'description': 'High geopolitical uncertainty',
        'conditioning': {
            'crude_oil_30d_mean': 0.5,
            'crude_oil_7d_mean': 0.5,
            'unrest_index_yearly': 2.5,  # Very high unrest
            'crude_oil': 0.5,
            'usdinr_quarterly_mean': 0.5,
            'india_vix_30d_mean': 1.5,
            'india_vix_7d_mean': 1.5,
            'us_10y_yield': 0.5
        }
    },
    'rate_hike_cycle': {
        'description': 'Rising interest rates environment',
        'conditioning': {
            'crude_oil_30d_mean': 0.0,
            'crude_oil_7d_mean': 0.0,
            'unrest_index_yearly': 0.0,
            'crude_oil': 0.0,
            'usdinr_quarterly_mean': 0.0,
            'india_vix_30d_mean': 0.5,
            'india_vix_7d_mean': 0.5,
            'us_10y_yield': 1.5  # High yields
        }
    },
    'fx_stress': {
        'description': 'Currency market stress (USD/INR)',
        'conditioning': {
            'crude_oil_30d_mean': 0.0,
            'crude_oil_7d_mean': 0.0,
            'unrest_index_yearly': 0.5,
            'crude_oil': 0.0,
            'usdinr_quarterly_mean': 1.5,  # High USD/INR
            'india_vix_30d_mean': 1.0,
            'india_vix_7d_mean': 1.0,
            'us_10y_yield': 0.5
        }
    }
}

# ============================================================================
# 2. Load Model
# ============================================================================

print("\n1. Loading trained model...")

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)

results_dir = os.path.join(script_dir, config['output']['results_dir'])
model_path = os.path.join(results_dir, config['output']['model_file'])

# Load checkpoint
checkpoint = torch.load(model_path, map_location=device)

# Initialize model
model = ConditionalVAE_SingleHeston(
    param_dim=config['architecture']['param_dim'],
    conditioning_dim=config['architecture']['conditioning_dim'],
    latent_dim=config['architecture']['latent_dim'],
    hidden_dims=config['architecture']['hidden_dims'],
    encoder_activation=config['architecture']['encoder_activation'],
    decoder_activation=config['architecture']['decoder_activation'],
    dropout=config['architecture']['dropout'],
    feller_penalty_weight=config['loss_weights']['feller_penalty'],
    beta=config['loss_weights']['kl_divergence'],
    arbitrage_penalty_weight=config['loss_weights']['arbitrage_penalty']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

param_mean = checkpoint['param_mean'].to(device)
param_std = checkpoint['param_std'].to(device)

print(f"   ✓ Model loaded from epoch {checkpoint['epoch']}")
print(f"   ✓ Best validation loss: {checkpoint['val_loss']:.4f}")

# ============================================================================
# 3. Parse Arguments
# ============================================================================

parser = argparse.ArgumentParser(description='Sample Heston parameters for different regimes')
parser.add_argument('--regimes', type=str, nargs='+', default=['all'],
                   help='Regimes to sample from. Use "all" for all regimes, or specify: ' + 
                        ', '.join(MARKET_REGIMES.keys()))
parser.add_argument('--num_samples', type=int, default=1000,
                   help='Number of samples per regime')
parser.add_argument('--output_dir', type=str, default='regime_samples',
                   help='Output directory for results')
parser.add_argument('--save_individual', action='store_true',
                   help='Save individual regime files')

args = parser.parse_args()

# Determine which regimes to sample
if 'all' in args.regimes:
    regimes_to_sample = list(MARKET_REGIMES.keys())
else:
    regimes_to_sample = [r for r in args.regimes if r in MARKET_REGIMES]
    if not regimes_to_sample:
        print(f"\nError: No valid regimes specified!")
        print(f"Available regimes: {', '.join(MARKET_REGIMES.keys())}")
        exit(1)

print(f"\n2. Sampling configuration:")
print(f"   Regimes: {', '.join(regimes_to_sample)}")
print(f"   Samples per regime: {args.num_samples}")

# Create output directory
output_dir = os.path.join(results_dir, args.output_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"   Output directory: {output_dir}")

# ============================================================================
# 4. Generate Samples for Each Regime
# ============================================================================

print(f"\n3. Generating samples...")
print("="*80)

cond_var_names = config['data']['conditioning_vars']
param_names = config['data']['param_order']

all_results = {}

for regime_name in regimes_to_sample:
    regime_info = MARKET_REGIMES[regime_name]
    
    print(f"\n{regime_name.upper().replace('_', ' ')}")
    print(f"  Description: {regime_info['description']}")
    
    # Create conditioning tensor
    cond_values = np.array([regime_info['conditioning'][var] for var in cond_var_names])
    cond_tensor = torch.tensor(cond_values, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Generate samples
    with torch.no_grad():
        samples_norm = model.sample(args.num_samples, cond_tensor, device)
        
        # Denormalize and inverse transform
        samples_denorm = samples_norm * param_std + param_mean
        
        params = torch.zeros_like(samples_denorm)
        params[:, 0] = torch.exp(samples_denorm[:, 0])  # kappa
        params[:, 1] = torch.exp(samples_denorm[:, 1])  # theta
        params[:, 2] = torch.exp(samples_denorm[:, 2])  # sigma_v
        params[:, 3] = torch.tanh(samples_denorm[:, 3])  # rho
        params[:, 4] = torch.exp(samples_denorm[:, 4])  # v0
    
    params_np = params.cpu().numpy()
    
    # Compute statistics
    stats = {
        'kappa': {'mean': params_np[:, 0].mean(), 'std': params_np[:, 0].std(),
                 'min': params_np[:, 0].min(), 'max': params_np[:, 0].max()},
        'theta': {'mean': params_np[:, 1].mean(), 'std': params_np[:, 1].std(),
                 'min': params_np[:, 1].min(), 'max': params_np[:, 1].max()},
        'sigma_v': {'mean': params_np[:, 2].mean(), 'std': params_np[:, 2].std(),
                   'min': params_np[:, 2].min(), 'max': params_np[:, 2].max()},
        'rho': {'mean': params_np[:, 3].mean(), 'std': params_np[:, 3].std(),
               'min': params_np[:, 3].min(), 'max': params_np[:, 3].max()},
        'v0': {'mean': params_np[:, 4].mean(), 'std': params_np[:, 4].std(),
              'min': params_np[:, 4].min(), 'max': params_np[:, 4].max()}
    }
    
    # Check Feller condition
    kappa = params_np[:, 0]
    theta = params_np[:, 1]
    sigma_v = params_np[:, 2]
    feller_satisfied = np.mean(2 * kappa * theta > sigma_v ** 2) * 100
    
    print(f"  Statistics:")
    print(f"    κ (mean rev):    {stats['kappa']['mean']:.4f} ± {stats['kappa']['std']:.4f}  [{stats['kappa']['min']:.4f}, {stats['kappa']['max']:.4f}]")
    print(f"    θ (long-term):   {stats['theta']['mean']:.4f} ± {stats['theta']['std']:.4f}  [{stats['theta']['min']:.4f}, {stats['theta']['max']:.4f}]")
    print(f"    σᵥ (vol-of-vol): {stats['sigma_v']['mean']:.4f} ± {stats['sigma_v']['std']:.4f}  [{stats['sigma_v']['min']:.4f}, {stats['sigma_v']['max']:.4f}]")
    print(f"    ρ (correlation): {stats['rho']['mean']:+.4f} ± {stats['rho']['std']:.4f}  [{stats['rho']['min']:+.4f}, {stats['rho']['max']:+.4f}]")
    print(f"    v₀ (init var):   {stats['v0']['mean']:.4f} ± {stats['v0']['std']:.4f}  [{stats['v0']['min']:.4f}, {stats['v0']['max']:.4f}]")
    print(f"  Feller condition: {feller_satisfied:.1f}% satisfied")
    
    # Store results
    all_results[regime_name] = {
        'params': params_np,
        'conditioning': cond_values,
        'conditioning_dict': regime_info['conditioning'],
        'description': regime_info['description'],
        'statistics': stats,
        'feller_satisfaction': feller_satisfied
    }
    
    # Save individual regime file if requested
    if args.save_individual:
        regime_file = os.path.join(output_dir, f'{regime_name}_heston_params.pt')
        torch.save({
            'params': torch.tensor(params_np),
            'conditioning': cond_values,
            'conditioning_dict': regime_info['conditioning'],
            'description': regime_info['description'],
            'statistics': stats,
            'feller_satisfaction': feller_satisfied,
            'param_names': param_names,
            'num_samples': args.num_samples
        }, regime_file)
        print(f"  ✓ Saved: {regime_file}")

# ============================================================================
# 5. Create Comparison Visualizations
# ============================================================================

print(f"\n4. Creating visualizations...")

# Plot 1: Parameter comparison across regimes
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, param_name in enumerate(param_names):
    ax = axes[i]
    
    regime_names = list(all_results.keys())
    means = [all_results[r]['statistics'][param_name]['mean'] for r in regime_names]
    stds = [all_results[r]['statistics'][param_name]['std'] for r in regime_names]
    
    x_pos = np.arange(len(regime_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(regime_names))))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.replace('_', '\n') for r in regime_names], 
                        rotation=0, ha='center', fontsize=8)
    ax.set_ylabel(param_name, fontsize=11, fontweight='bold')
    ax.set_title(f'{param_name.upper()} by Regime', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

# Hide last subplot
axes[-1].axis('off')

plt.suptitle('Heston Parameters Across Market Regimes', fontsize=16, fontweight='bold')
plt.tight_layout()
comparison_path = os.path.join(output_dir, 'regime_comparison.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {comparison_path}")
plt.close()

# Plot 2: Distributions for key parameter (v0)
fig, ax = plt.subplots(figsize=(14, 6))

for regime_name in all_results.keys():
    v0_samples = all_results[regime_name]['params'][:, 4]
    ax.hist(v0_samples, bins=50, alpha=0.5, label=regime_name.replace('_', ' ').title(), 
            density=True)

ax.set_xlabel('v₀ (Initial Variance)', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('Initial Variance (v₀) Distribution Across Regimes', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

v0_dist_path = os.path.join(output_dir, 'v0_distributions.png')
plt.savefig(v0_dist_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {v0_dist_path}")
plt.close()

# Plot 3: Feller satisfaction
fig, ax = plt.subplots(figsize=(12, 6))

regime_names = list(all_results.keys())
feller_pcts = [all_results[r]['feller_satisfaction'] for r in regime_names]

bars = ax.bar(range(len(regime_names)), feller_pcts, 
              color=plt.cm.RdYlGn(np.array(feller_pcts) / 100), alpha=0.8)
ax.axhline(90, color='red', linestyle='--', linewidth=2, label='90% threshold')

ax.set_xticks(range(len(regime_names)))
ax.set_xticklabels([r.replace('_', '\n') for r in regime_names], 
                    rotation=0, ha='center', fontsize=9)
ax.set_ylabel('Feller Satisfaction (%)', fontsize=12, fontweight='bold')
ax.set_title('Feller Condition Satisfaction by Regime', fontsize=14, fontweight='bold')
ax.set_ylim([0, 105])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (bar, pct) in enumerate(zip(bars, feller_pcts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

feller_path = os.path.join(output_dir, 'feller_satisfaction.png')
plt.savefig(feller_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {feller_path}")
plt.close()

# ============================================================================
# 6. Save Combined Results
# ============================================================================

print(f"\n5. Saving combined results...")

# Save all parameters as one file
all_params_dict = {}
for regime_name in all_results.keys():
    all_params_dict[regime_name] = torch.tensor(all_results[regime_name]['params'])

combined_path = os.path.join(output_dir, 'all_regimes_heston_params.pt')
torch.save({
    'regimes': all_params_dict,
    'param_names': param_names,
    'conditioning_vars': cond_var_names,
    'num_samples_per_regime': args.num_samples,
    'metadata': {regime: {
        'description': all_results[regime]['description'],
        'conditioning': all_results[regime]['conditioning_dict'],
        'statistics': all_results[regime]['statistics'],
        'feller_satisfaction': all_results[regime]['feller_satisfaction']
    } for regime in all_results.keys()},
    'timestamp': datetime.now().isoformat()
}, combined_path)
print(f"  ✓ Combined file: {combined_path}")

# Save summary CSV
summary_data = []
for regime_name in all_results.keys():
    stats = all_results[regime_name]['statistics']
    summary_data.append({
        'regime': regime_name,
        'description': all_results[regime_name]['description'],
        'kappa_mean': stats['kappa']['mean'],
        'theta_mean': stats['theta']['mean'],
        'sigma_v_mean': stats['sigma_v']['mean'],
        'rho_mean': stats['rho']['mean'],
        'v0_mean': stats['v0']['mean'],
        'feller_satisfaction': all_results[regime_name]['feller_satisfaction']
    })

summary_df = pd.DataFrame(summary_data)
summary_csv_path = os.path.join(output_dir, 'regime_summary.csv')
summary_df.to_csv(summary_csv_path, index=False)
print(f"  ✓ Summary CSV: {summary_csv_path}")

# Save detailed JSON
detailed_results = {}
for regime_name in all_results.keys():
    detailed_results[regime_name] = {
        'description': all_results[regime_name]['description'],
        'conditioning': all_results[regime_name]['conditioning_dict'],
        'statistics': {
            param: {k: float(v) for k, v in stats.items()}
            for param, stats in all_results[regime_name]['statistics'].items()
        },
        'feller_satisfaction': float(all_results[regime_name]['feller_satisfaction']),
        'num_samples': args.num_samples
    }

json_path = os.path.join(output_dir, 'regime_details.json')
with open(json_path, 'w') as f:
    json.dump(detailed_results, f, indent=2)
print(f"  ✓ Detailed JSON: {json_path}")

# ============================================================================
# 7. Summary
# ============================================================================

print("\n" + "="*80)
print("SAMPLING COMPLETE!")
print("="*80)

print(f"\nGenerated {args.num_samples} Heston parameter sets for {len(regimes_to_sample)} regimes")
print(f"\nOutput location: {output_dir}")

print(f"\nFiles created:")
print(f"  - all_regimes_heston_params.pt  (Combined PyTorch file)")
print(f"  - regime_summary.csv            (Summary statistics)")
print(f"  - regime_details.json           (Detailed results)")
print(f"  - regime_comparison.png         (Parameter comparison)")
print(f"  - v0_distributions.png          (v₀ distributions)")
print(f"  - feller_satisfaction.png       (Feller validation)")

if args.save_individual:
    print(f"  - <regime>_heston_params.pt    (Individual regime files)")

print(f"\nKey insights:")
v0_values = [(r, all_results[r]['statistics']['v0']['mean']) for r in all_results.keys()]
v0_values_sorted = sorted(v0_values, key=lambda x: x[1])

print(f"  v₀ (Initial Variance) ranking:")
for regime, v0 in v0_values_sorted:
    print(f"    {regime:25s}: {v0:.4f}")

print(f"\n  Feller satisfaction:")
for regime in all_results.keys():
    feller = all_results[regime]['feller_satisfaction']
    status = "✓" if feller >= 90 else "⚠"
    print(f"    {status} {regime:25s}: {feller:5.1f}%")

print("\n" + "="*80)
