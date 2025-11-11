"""
Generate Conditional Samples from Trained CVAE
Supports multiple sampling strategies: direct, regime-based, historical
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from cvae_model import ConditionalVAE_SingleHeston

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
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
    
    return model, checkpoint, config


def denormalize_params(params_norm, param_mean, param_std):
    """Denormalize and inverse transform parameters"""
    params_denorm = params_norm * param_std + param_mean
    
    params = torch.zeros_like(params_denorm)
    params[:, 0] = torch.exp(params_denorm[:, 0])  # kappa
    params[:, 1] = torch.exp(params_denorm[:, 1])  # theta
    params[:, 2] = torch.exp(params_denorm[:, 2])  # sigma_v
    params[:, 3] = torch.tanh(params_denorm[:, 3])  # rho
    params[:, 4] = torch.exp(params_denorm[:, 4])  # v0
    
    return params


def sample_direct(model, conditioning, num_samples, param_mean, param_std, device):
    """
    Strategy 1: Direct conditioning
    User provides exact conditioning values
    """
    cond_tensor = torch.tensor(conditioning, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        samples_norm = model.sample(num_samples, cond_tensor, device)
        samples = denormalize_params(samples_norm, param_mean, param_std)
    
    return samples.cpu().numpy()


def sample_regime(model, regime_name, num_samples, param_mean, param_std, device, cond_var_names):
    """
    Strategy 2: Regime-based sampling
    User specifies regime, lookup typical conditioning values
    """
    regimes = {
        'low_volatility': {
            'crude_oil_30d_mean': -0.5,
            'crude_oil_7d_mean': -0.5,
            'unrest_index_yearly': -0.5,
            'crude_oil': -0.5,
            'usdinr_quarterly_mean': 0.0,
            'india_vix_30d_mean': -1.5,
            'india_vix_7d_mean': -1.5,
            'us_10y_yield': 0.0
        },
        'moderate_volatility': {
            'crude_oil_30d_mean': 0.0,
            'crude_oil_7d_mean': 0.0,
            'unrest_index_yearly': 0.0,
            'crude_oil': 0.0,
            'usdinr_quarterly_mean': 0.0,
            'india_vix_30d_mean': 0.0,
            'india_vix_7d_mean': 0.0,
            'us_10y_yield': 0.0
        },
        'high_volatility': {
            'crude_oil_30d_mean': 0.5,
            'crude_oil_7d_mean': 0.5,
            'unrest_index_yearly': 1.5,
            'crude_oil': 0.5,
            'usdinr_quarterly_mean': 0.5,
            'india_vix_30d_mean': 2.0,
            'india_vix_7d_mean': 2.0,
            'us_10y_yield': 0.5
        },
        'crisis': {
            'crude_oil_30d_mean': -1.5,
            'crude_oil_7d_mean': -1.5,
            'unrest_index_yearly': 2.0,
            'crude_oil': -2.0,
            'usdinr_quarterly_mean': 1.0,
            'india_vix_30d_mean': 3.0,
            'india_vix_7d_mean': 3.0,
            'us_10y_yield': -1.0
        }
    }
    
    if regime_name not in regimes:
        raise ValueError(f"Unknown regime: {regime_name}. Choose from: {list(regimes.keys())}")
    
    regime_cond = regimes[regime_name]
    conditioning = np.array([regime_cond[var] for var in cond_var_names])
    
    return sample_direct(model, conditioning, num_samples, param_mean, param_std, device)


def sample_historical(model, cond_data, num_samples, param_mean, param_std, device, strategy='random'):
    """
    Strategy 3: Historical conditioning
    Sample conditioning from historical distribution
    
    strategy:
        - 'random': Random sample from full history
        - 'recent': Sample from most recent period
        - 'crisis': Sample from crisis periods (high VIX)
    """
    if strategy == 'random':
        idx = np.random.randint(0, len(cond_data))
        conditioning = cond_data[idx]
    
    elif strategy == 'recent':
        # Use last 20% of data
        recent_data = cond_data[int(0.8 * len(cond_data)):]
        idx = np.random.randint(0, len(recent_data))
        conditioning = recent_data[idx]
    
    elif strategy == 'crisis':
        # High VIX periods (india_vix_30d_mean > 1.5)
        crisis_mask = cond_data[:, 5] > 1.5  # india_vix_30d_mean is index 5
        if np.any(crisis_mask):
            crisis_data = cond_data[crisis_mask]
            idx = np.random.randint(0, len(crisis_data))
            conditioning = crisis_data[idx]
        else:
            print("Warning: No crisis periods found, using random sample")
            conditioning = cond_data[np.random.randint(0, len(cond_data))]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return sample_direct(model, conditioning, num_samples, param_mean, param_std, device), conditioning


def main():
    parser = argparse.ArgumentParser(description='Generate conditional samples from trained CVAE')
    parser.add_argument('--mode', type=str, default='regime', 
                       choices=['direct', 'regime', 'historical'],
                       help='Sampling mode')
    parser.add_argument('--regime', type=str, default='moderate_volatility',
                       choices=['low_volatility', 'moderate_volatility', 'high_volatility', 'crisis'],
                       help='Regime for regime-based sampling')
    parser.add_argument('--historical_strategy', type=str, default='random',
                       choices=['random', 'recent', 'crisis'],
                       help='Strategy for historical sampling')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: results/generated_samples.pt)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CONDITIONAL SAMPLE GENERATION")
    print("="*80)
    
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    results_dir = os.path.join(script_dir, config['output']['results_dir'])
    model_path = os.path.join(results_dir, config['output']['model_file'])
    
    print(f"\n1. Loading model...")
    model, checkpoint, config = load_model(model_path, device)
    print(f"    Model loaded from epoch {checkpoint['epoch']}")
    
    param_mean = checkpoint['param_mean'].to(device)
    param_std = checkpoint['param_std'].to(device)
    
    # Load conditioning data
    cond_file = os.path.join(script_dir, config['data']['conditioning_file'])
    cond_df = pd.read_csv(cond_file)
    cond_var_names = config['data']['conditioning_vars']
    cond_data = cond_df[cond_var_names].values
    
    print(f"\n2. Generating {args.num_samples} samples using '{args.mode}' mode...")
    
    # Generate samples based on mode
    if args.mode == 'direct':
        print("   Note: Using moderate volatility regime as default")
        print("   For custom conditioning, modify the code to pass your values")
        samples = sample_regime(model, 'moderate_volatility', args.num_samples, 
                               param_mean, param_std, device, cond_var_names)
        conditioning_used = None
    
    elif args.mode == 'regime':
        print(f"   Regime: {args.regime}")
        samples = sample_regime(model, args.regime, args.num_samples, 
                               param_mean, param_std, device, cond_var_names)
        conditioning_used = None
    
    elif args.mode == 'historical':
        print(f"   Strategy: {args.historical_strategy}")
        samples, conditioning_used = sample_historical(
            model, cond_data, args.num_samples, param_mean, param_std, device, 
            strategy=args.historical_strategy
        )
        print(f"   Conditioning used:")
        for i, var_name in enumerate(cond_var_names):
            print(f"     {var_name:25s}: {conditioning_used[i]:+.3f}")
    
    print(f"\n    Generated samples: {samples.shape}")
    
    # Compute statistics
    print(f"\n3. Sample statistics:")
    param_names = config['data']['param_order']
    for i, param_name in enumerate(param_names):
        print(f"   {param_name:10s}: mean={samples[:, i].mean():.4f}, "
              f"std={samples[:, i].std():.4f}, "
              f"min={samples[:, i].min():.4f}, "
              f"max={samples[:, i].max():.4f}")
    
    # Check Feller condition
    kappa = samples[:, 0]
    theta = samples[:, 1]
    sigma_v = samples[:, 2]
    feller_satisfied = np.mean(2 * kappa * theta > sigma_v ** 2) * 100
    print(f"\n   Feller condition satisfied: {feller_satisfied:.1f}%")
    
    # Save samples
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(results_dir, 'generated_samples.pt')
    
    torch.save({
        'samples': torch.tensor(samples),
        'mode': args.mode,
        'regime': args.regime if args.mode == 'regime' else None,
        'conditioning': conditioning_used,
        'num_samples': args.num_samples,
        'param_names': param_names,
        'feller_satisfaction': feller_satisfied
    }, output_path)
    
    print(f"\n    Samples saved: {output_path}")
    
    # Create visualization
    print(f"\n4. Creating visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        ax.hist(samples[:, i], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{param_name} Distribution', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        mean_val = samples[:, i].mean()
        std_val = samples[:, i].std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'μ={mean_val:.3f}')
        ax.legend()
    
    # Hide last subplot
    axes[-1].axis('off')
    
    title = f'Generated Samples: {args.mode.capitalize()}'
    if args.mode == 'regime':
        title += f' ({args.regime})'
    elif args.mode == 'historical':
        title += f' ({args.historical_strategy})'
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_path.replace('.pt', '.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"    Visualization saved: {plot_path}")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
