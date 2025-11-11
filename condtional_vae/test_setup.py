"""
Test Conditional VAE Setup Before Training
Validates data loading, model creation, and forward/backward passes
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cvae_model import ConditionalVAE_SingleHeston

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*80)
print("CONDITIONAL VAE SETUP TEST")
print("="*80)

# ============================================================================
# 1. Load Configuration
# ============================================================================

print("\n1. Loading configuration...")

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)

print(f"    Configuration loaded")
print(f"   Device: {device}")

# ============================================================================
# 2. Load and Merge Data
# ============================================================================

print("\n2. Loading and merging data...")

# Load Heston parameters
heston_file = os.path.join(script_dir, config['data']['heston_params_file'])
heston_params = torch.load(heston_file)
print(f"    Heston parameters: {heston_params.shape}")

# Load conditioning variables
cond_file = os.path.join(script_dir, config['data']['conditioning_file'])
cond_df = pd.read_csv(cond_file)
print(f"    Conditioning variables: {cond_df.shape}")

# Extract dates
date_col = config['data']['date_column']
dates = pd.to_datetime(cond_df[date_col])

# Verify alignment
assert len(dates) == len(heston_params), \
    f"Date mismatch: {len(dates)} dates vs {len(heston_params)} params"
print(f"    Data alignment verified: {len(dates)} samples")
print(f"   Date range: {dates.min()} to {dates.max()}")

# Extract conditioning variables
cond_var_names = config['data']['conditioning_vars']
conditioning = cond_df[cond_var_names].values.astype(np.float32)
print(f"    Extracted {len(cond_var_names)} conditioning variables")

# ============================================================================
# 3. Apply Transformations
# ============================================================================

print("\n3. Applying transformations...")

# Transform parameters
transformed_params = heston_params.clone()
for idx in [0, 1, 2, 4]:  # kappa, theta, sigma_v, v0
    transformed_params[:, idx] = torch.log(heston_params[:, idx])
transformed_params[:, 3] = torch.atanh(heston_params[:, 3] * 0.999)  # rho

# Normalize parameters
param_mean = transformed_params.mean(dim=0)
param_std = transformed_params.std(dim=0)
normalized_params = (transformed_params - param_mean) / param_std

print(f"    Parameters transformed and normalized")
print(f"   Mean: {param_mean.numpy()}")
print(f"   Std:  {param_std.numpy()}")

# Conditioning (already normalized)
conditioning_tensor = torch.tensor(conditioning, dtype=torch.float32)
print(f"    Conditioning variables: {conditioning_tensor.shape}")

# ============================================================================
# 4. Create Small Test Set
# ============================================================================

print("\n4. Creating test batch...")

# Use first 32 samples
test_params = normalized_params[:32]
test_cond = conditioning_tensor[:32]

print(f"    Test parameters: {test_params.shape}")
print(f"    Test conditioning: {test_cond.shape}")

# Move to device
test_params = test_params.to(device)
test_cond = test_cond.to(device)
param_mean_device = param_mean.to(device)
param_std_device = param_std.to(device)

# ============================================================================
# 5. Initialize Model
# ============================================================================

print("\n5. Initializing model...")

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

total_params = sum(p.numel() for p in model.parameters())
print(f"    Model created")
print(f"   Total parameters: {total_params:,}")
print(f"   Architecture: {config['architecture']['hidden_dims']}")
print(f"   Latent dim: {config['architecture']['latent_dim']}")

# ============================================================================
# 6. Test Forward Pass
# ============================================================================

print("\n6. Testing forward pass...")

model.eval()
with torch.no_grad():
    recon, mu, logvar = model(test_params, test_cond)

print(f"    Input parameters: {test_params.shape}")
print(f"    Input conditioning: {test_cond.shape}")
print(f"    Reconstructed: {recon.shape}")
print(f"    Latent mu: {mu.shape}")
print(f"    Latent logvar: {logvar.shape}")

# ============================================================================
# 7. Test Loss Computation
# ============================================================================

print("\n7. Testing loss computation...")

loss, recon_loss, kl_loss, feller_loss, arbitrage_loss = model.loss_function(
    recon, test_params, mu, logvar, param_mean_device, param_std_device
)

print(f"    Total loss:      {loss.item():.6f}")
print(f"    Recon loss:      {recon_loss.item():.6f}")
print(f"    KL loss:         {kl_loss.item():.6f}")
print(f"    Feller loss:     {feller_loss.item():.6f}")
print(f"    Arbitrage loss:  {arbitrage_loss.item():.6f}")

# Verify loss components
expected_total = (recon_loss.item() + 
                 config['loss_weights']['kl_divergence'] * kl_loss.item() +
                 config['loss_weights']['feller_penalty'] * feller_loss.item() +
                 config['loss_weights']['arbitrage_penalty'] * arbitrage_loss.item())

assert abs(loss.item() - expected_total) < 1e-4, \
    f"Loss mismatch: {loss.item()} vs {expected_total}"
print(f"    Loss computation verified")

# ============================================================================
# 8. Test Backward Pass
# ============================================================================

print("\n8. Testing backward pass...")

model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
recon, mu, logvar = model(test_params, test_cond)
loss, _, _, _, _ = model.loss_function(
    recon, test_params, mu, logvar, param_mean_device, param_std_device
)
loss.backward()
optimizer.step()

print(f"    Backward pass successful")
print(f"    Optimizer step successful")

# ============================================================================
# 9. Test Conditional Sampling
# ============================================================================

print("\n9. Testing conditional sampling...")

model.eval()

# Test with single conditioning
single_cond = test_cond[0:1]  # (1, 8)
with torch.no_grad():
    samples_single = model.sample(
        num_samples=10,
        conditioning=single_cond,
        device=device
    )

print(f"    Single condition broadcast: {samples_single.shape}")
assert samples_single.shape == (10, 5), f"Shape mismatch: {samples_single.shape}"

# Test with multiple conditioning
multi_cond = test_cond[:5]  # (5, 8)
with torch.no_grad():
    samples_multi = model.sample(
        num_samples=5,
        conditioning=multi_cond,
        device=device
    )

print(f"    Multiple conditions: {samples_multi.shape}")
assert samples_multi.shape == (5, 5), f"Shape mismatch: {samples_multi.shape}"

# ============================================================================
# 10. Test Parameter Recovery
# ============================================================================

print("\n10. Testing parameter recovery...")

# Denormalize and inverse transform
samples_denorm = samples_single * param_std_device + param_mean_device

params_generated = torch.zeros_like(samples_denorm)
params_generated[:, 0] = torch.exp(samples_denorm[:, 0])  # kappa
params_generated[:, 1] = torch.exp(samples_denorm[:, 1])  # theta
params_generated[:, 2] = torch.exp(samples_denorm[:, 2])  # sigma_v
params_generated[:, 3] = torch.tanh(samples_denorm[:, 3])  # rho
params_generated[:, 4] = torch.exp(samples_denorm[:, 4])  # v0

print(f"    Generated parameters: {params_generated.shape}")
print(f"    Parameter ranges:")
print(f"     kappa:   [{params_generated[:, 0].min():.4f}, {params_generated[:, 0].max():.4f}]")
print(f"     theta:   [{params_generated[:, 1].min():.4f}, {params_generated[:, 1].max():.4f}]")
print(f"     sigma_v: [{params_generated[:, 2].min():.4f}, {params_generated[:, 2].max():.4f}]")
print(f"     rho:     [{params_generated[:, 3].min():.4f}, {params_generated[:, 3].max():.4f}]")
print(f"     v0:      [{params_generated[:, 4].min():.4f}, {params_generated[:, 4].max():.4f}]")

# Check Feller condition
kappa = params_generated[:, 0]
theta = params_generated[:, 1]
sigma_v = params_generated[:, 2]
feller_satisfied = (2 * kappa * theta > sigma_v ** 2).float().mean().item() * 100

print(f"    Feller satisfaction: {feller_satisfied:.1f}%")

# ============================================================================
# 11. Test Conditioning Effect
# ============================================================================

print("\n11. Testing conditioning effect...")

# Create two different conditioning scenarios
# High volatility regime (high VIX indices)
high_vol_cond = test_cond[0:1].clone()
high_vol_cond[0, 5] = 2.0  # india_vix_30d_mean
high_vol_cond[0, 6] = 2.0  # india_vix_7d_mean

# Low volatility regime (low VIX indices)
low_vol_cond = test_cond[0:1].clone()
low_vol_cond[0, 5] = -1.0  # india_vix_30d_mean
low_vol_cond[0, 6] = -1.0  # india_vix_7d_mean

# Generate samples
num_test_samples = 50
with torch.no_grad():
    samples_high_vol = model.sample(num_test_samples, high_vol_cond, device)
    samples_low_vol = model.sample(num_test_samples, low_vol_cond, device)

# Denormalize
high_vol_denorm = samples_high_vol * param_std_device + param_mean_device
low_vol_denorm = samples_low_vol * param_std_device + param_mean_device

# Get v0 (initial variance)
v0_high_vol = torch.exp(high_vol_denorm[:, 4]).mean().item()
v0_low_vol = torch.exp(low_vol_denorm[:, 4]).mean().item()

print(f"    High VIX regime: Mean v0 = {v0_high_vol:.4f}")
print(f"    Low VIX regime:  Mean v0 = {v0_low_vol:.4f}")
print(f"    Difference: {v0_high_vol - v0_low_vol:.4f}")

if v0_high_vol > v0_low_vol:
    print(f"    PASS: High VIX → Higher v0 (as expected from EDA)")
else:
    print(f"    WARNING: High VIX → Lower v0 (unexpected, model may need training)")

# ============================================================================
# 12. Test Data Loader
# ============================================================================

print("\n12. Testing data loader...")

# Create small dataset
small_dataset = TensorDataset(
    normalized_params[:64],
    conditioning_tensor[:64]
)

small_loader = DataLoader(small_dataset, batch_size=16, shuffle=True)

batch_count = 0
for batch_params, batch_cond in small_loader:
    batch_params = batch_params.to(device)
    batch_cond = batch_cond.to(device)
    
    recon, mu, logvar = model(batch_params, batch_cond)
    loss, _, _, _, _ = model.loss_function(
        recon, batch_params, mu, logvar, param_mean_device, param_std_device
    )
    
    batch_count += 1

print(f"    Processed {batch_count} batches")
print(f"    Batch size: 16")
print(f"    Last batch loss: {loss.item():.6f}")

# ============================================================================
# 13. Summary
# ============================================================================

print("\n" + "="*80)
print("SETUP TEST COMPLETE - ALL CHECKS PASSED!")
print("="*80)
print("\nReady to start training!")
print("\nRun: python train_cvae.py")
print("\n" + "="*80)
