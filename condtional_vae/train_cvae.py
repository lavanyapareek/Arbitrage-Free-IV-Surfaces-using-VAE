"""
Train Conditional VAE on Heston Parameters with Market Conditioning
Learns p(θ|c) where θ = Heston params, c = market/macro variables
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from cvae_model import ConditionalVAE_SingleHeston

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. Load Configuration
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.json')

with open(config_path, 'r') as f:
    config = json.load(f)

print("\n" + "="*80)
print("CONDITIONAL VAE TRAINING: HESTON PARAMETERS")
print("="*80)
print(f"Configuration loaded from: {config_path}")

# Create results directory
results_dir = os.path.join(script_dir, config['output']['results_dir'])
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# ============================================================================
# 2. Load and Merge Data
# ============================================================================

print("\n1. Loading data...")

# Load Heston parameters
heston_file = os.path.join(script_dir, config['data']['heston_params_file'])
heston_params = torch.load(heston_file)
print(f"    Heston parameters: {heston_params.shape}")
print(f"     Format: {config['data']['param_order']}")

# Load conditioning variables
cond_file = os.path.join(script_dir, config['data']['conditioning_file'])
cond_df = pd.read_csv(cond_file)
print(f"    Conditioning variables: {cond_df.shape}")

# Extract dates
date_col = config['data']['date_column']
dates = pd.to_datetime(cond_df[date_col])
print(f"    Date range: {dates.min()} to {dates.max()}")

# Verify alignment
assert len(dates) == len(heston_params), \
    f"Date mismatch: {len(dates)} dates vs {len(heston_params)} params"
print(f"    Data alignment verified: {len(dates)} samples")

# Extract conditioning variables
cond_var_names = config['data']['conditioning_vars']
conditioning = cond_df[cond_var_names].values.astype(np.float32)
print(f"\n   Conditioning variables ({len(cond_var_names)}):")
for i, var in enumerate(cond_var_names, 1):
    print(f"     {i}. {var}")

# Create merged dataframe for temporal split
df_merged = pd.DataFrame({
    'date': dates.values
})
for i, param_name in enumerate(config['data']['param_order']):
    df_merged[param_name] = heston_params[:, i].numpy()
for i, var_name in enumerate(cond_var_names):
    df_merged[var_name] = conditioning[:, i]

# Sort by date for temporal split
df_merged = df_merged.sort_values('date').reset_index(drop=True)
print(f"    Data sorted by date for temporal split")

# ============================================================================
# 3. Apply Transformations to Parameters
# ============================================================================

print("\n2. Applying transformations to parameters...")

# Extract parameters from sorted dataframe
params_sorted = torch.tensor(
    df_merged[config['data']['param_order']].values,
    dtype=torch.float32
)

# Apply transformations
transformed_params = params_sorted.clone()

# kappa, theta, sigma_v, v0: log transform
for idx in [0, 1, 2, 4]:
    transformed_params[:, idx] = torch.log(params_sorted[:, idx])

# rho: atanh transform (map from [-1, 1] to [-inf, inf])
transformed_params[:, 3] = torch.atanh(params_sorted[:, 3] * 0.999)  # Clip to avoid ±1

print("    Transforms applied:")
print("     - kappa, theta, sigma_v, v0: log")
print("     - rho: atanh")

# Normalize parameters to zero mean, unit variance
param_mean = transformed_params.mean(dim=0)
param_std = transformed_params.std(dim=0)
normalized_params = (transformed_params - param_mean) / param_std

print(f"    Parameters normalized to N(0,1)")

# Extract conditioning from sorted dataframe
conditioning_sorted = torch.tensor(
    df_merged[cond_var_names].values,
    dtype=torch.float32
)

print(f"    Conditioning variables: already z-normalized (as-is)")

# ============================================================================
# 4. Temporal Train/Val Split
# ============================================================================

print("\n3. Creating temporal train/val split...")

n_total = len(normalized_params)
n_train = int(n_total * config['data']['train_split'])

# Split data (already sorted by date)
train_params = normalized_params[:n_train]
train_cond = conditioning_sorted[:n_train]
train_dates = df_merged['date'].iloc[:n_train]

val_params = normalized_params[n_train:]
val_cond = conditioning_sorted[n_train:]
val_dates = df_merged['date'].iloc[n_train:]

print(f"    Train: {len(train_params)} samples ({train_dates.min()} to {train_dates.max()})")
if len(val_params) > 0:
    print(f"    Val:   {len(val_params)} samples ({val_dates.min()} to {val_dates.max()})")
else:
    print(f"    Val:   {len(val_params)} samples (no validation — train_split == 1)")

# Create data loaders
train_dataset = TensorDataset(train_params, train_cond)
batch_size = config['training']['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Only create a validation loader if we actually have validation samples
has_val = len(val_params) > 0
if has_val:
    val_dataset = TensorDataset(val_params, val_cond)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
else:
    val_dataset = None
    val_loader = None
    print("    Validation disabled (no validation samples)")

# ============================================================================
# 5. Initialize Model
# ============================================================================

print("\n4. Initializing Conditional VAE model...")

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

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"    Model initialized")
print(f"     Total parameters: {total_params:,}")
print(f"     Trainable parameters: {trainable_params:,}")
print(f"     Architecture: {config['architecture']['hidden_dims']}")
print(f"     Latent dim: {config['architecture']['latent_dim']}")
print(f"     Dropout: {config['architecture']['dropout']}")

# ============================================================================
# 6. Setup Optimizer and Scheduler
# ============================================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=50
)

print(f"\n5. Optimizer configured:")
print(f"   Learning rate: {config['training']['learning_rate']}")
print(f"   Weight decay: {config['training']['weight_decay']}")
print(f"   Gradient clipping: {config['training']['gradient_clip']}")

# ============================================================================
# 7. Training Loop
# ============================================================================

print("\n6. Starting training...")
print("="*80)

# Move normalization tensors to device
param_mean_device = param_mean.to(device)
param_std_device = param_std.to(device)

# Training history
history = {
    'epoch': [],
    'train_loss': [],
    'train_recon': [],
    'train_kl': [],
    'train_feller': [],
    'train_arbitrage': [],
    'val_loss': [],
    'val_recon': [],
    'val_kl': [],
    'val_feller': [],
    'val_arbitrage': [],
    'lr': []
}

best_val_loss = float('inf')
epochs = config['training']['epochs']
gradient_clip = config['training']['gradient_clip']

start_time = time.time()

for epoch in range(1, epochs + 1):
    # ========================================================================
    # Training
    # ========================================================================
    model.train()
    train_metrics = {
        'loss': 0, 'recon': 0, 'kl': 0, 'feller': 0, 'arbitrage': 0
    }
    
    for batch_params, batch_cond in train_loader:
        batch_params = batch_params.to(device)
        batch_cond = batch_cond.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu, logvar = model(batch_params, batch_cond)
        
        # Compute loss
        loss, recon_loss, kl_loss, feller_loss, arbitrage_loss = model.loss_function(
            recon, batch_params, mu, logvar, param_mean_device, param_std_device
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Accumulate metrics
        batch_size_actual = len(batch_params)
        train_metrics['loss'] += loss.item() * batch_size_actual
        train_metrics['recon'] += recon_loss.item() * batch_size_actual
        train_metrics['kl'] += kl_loss.item() * batch_size_actual
        train_metrics['feller'] += feller_loss.item() * batch_size_actual
        train_metrics['arbitrage'] += arbitrage_loss.item() * batch_size_actual
    
    # Average training metrics
    for key in train_metrics:
        train_metrics[key] /= len(train_params)
    
    # ========================================================================
    # Validation (if any)
    # ========================================================================
    if has_val:
        model.eval()
        val_metrics = {
            'loss': 0, 'recon': 0, 'kl': 0, 'feller': 0, 'arbitrage': 0
        }
        
        with torch.no_grad():
            for batch_params, batch_cond in val_loader:
                batch_params = batch_params.to(device)
                batch_cond = batch_cond.to(device)
                
                # Forward pass
                recon, mu, logvar = model(batch_params, batch_cond)
                
                # Compute loss
                loss, recon_loss, kl_loss, feller_loss, arbitrage_loss = model.loss_function(
                    recon, batch_params, mu, logvar, param_mean_device, param_std_device
                )
                
                # Accumulate metrics
                batch_size_actual = len(batch_params)
                val_metrics['loss'] += loss.item() * batch_size_actual
                val_metrics['recon'] += recon_loss.item() * batch_size_actual
                val_metrics['kl'] += kl_loss.item() * batch_size_actual
                val_metrics['feller'] += feller_loss.item() * batch_size_actual
                val_metrics['arbitrage'] += arbitrage_loss.item() * batch_size_actual
        
        # Average validation metrics
        for key in val_metrics:
            val_metrics[key] /= len(val_params)

        # Update learning rate using validation loss
        scheduler.step(val_metrics['loss'])
    else:
        # No validation — populate val_metrics with NaN so history keeps shape
        val_metrics = {k: float('nan') for k in ['loss', 'recon', 'kl', 'feller', 'arbitrage']}
        # Update LR based on training loss instead
        scheduler.step(train_metrics['loss'])
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['epoch'].append(epoch)
    history['train_loss'].append(train_metrics['loss'])
    history['train_recon'].append(train_metrics['recon'])
    history['train_kl'].append(train_metrics['kl'])
    history['train_feller'].append(train_metrics['feller'])
    history['train_arbitrage'].append(train_metrics['arbitrage'])
    history['val_loss'].append(val_metrics['loss'])
    history['val_recon'].append(val_metrics['recon'])
    history['val_kl'].append(val_metrics['kl'])
    history['val_feller'].append(val_metrics['feller'])
    history['val_arbitrage'].append(val_metrics['arbitrage'])
    history['lr'].append(current_lr)
    
    # Logging
    if epoch % config['monitoring']['log_every'] == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Recon: {train_metrics['recon']:.4f} | "
              f"KL: {train_metrics['kl']:.4f} | "
              f"Feller: {train_metrics['feller']:.6f} | "
              f"Arb: {train_metrics['arbitrage']:.6f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Recon: {val_metrics['recon']:.4f} | "
              f"KL: {val_metrics['kl']:.4f} | "
              f"Feller: {val_metrics['feller']:.6f} | "
              f"Arb: {val_metrics['arbitrage']:.6f}")
        print(f"  LR: {current_lr:.6f}")
    
    # Save best model (only if validation is enabled)
    if has_val and val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        best_model_path = os.path.join(results_dir, 'best_' + config['output']['model_file'])
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'param_mean': param_mean,
            'param_std': param_std,
            'config': config
        }, best_model_path)
    
    # Save checkpoint
    if epoch % config['monitoring']['save_checkpoint_every'] == 0:
        checkpoint_path = os.path.join(results_dir, f'checkpoint_epoch{epoch}.pt')
        checkpoint_val_loss = val_metrics['loss'] if has_val else train_metrics['loss']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': checkpoint_val_loss,
            'param_mean': param_mean,
            'param_std': param_std,
            'history': history,
            'config': config
        }, checkpoint_path)

# ============================================================================
# 8. Save Final Results
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

elapsed_time = time.time() - start_time
print(f"Total training time: {elapsed_time/60:.2f} minutes")
if best_val_loss == float('inf'):
    print("Best validation loss: N/A (no validation performed)")
else:
    print(f"Best validation loss: {best_val_loss:.4f}")

# Save final model
final_model_path = os.path.join(results_dir, config['output']['model_file'])
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_metrics['loss'] if has_val else train_metrics['loss'],
    'param_mean': param_mean,
    'param_std': param_std,
    'history': history,
    'config': config
}, final_model_path)
print(f" Final model saved: {final_model_path}")

# Save training history
history_df = pd.DataFrame(history)
history_path = os.path.join(results_dir, config['output']['training_history'])
history_df.to_csv(history_path, index=False)
print(f" Training history saved: {history_path}")

# Plot training curves
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Total loss
axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['epoch'], history['val_loss'], label='Val', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].set_title('Total Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Reconstruction loss
axes[0, 1].plot(history['epoch'], history['train_recon'], label='Train', linewidth=2)
axes[0, 1].plot(history['epoch'], history['val_recon'], label='Val', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Reconstruction Loss')
axes[0, 1].set_title('Reconstruction Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# KL divergence
axes[0, 2].plot(history['epoch'], history['train_kl'], label='Train', linewidth=2)
axes[0, 2].plot(history['epoch'], history['val_kl'], label='Val', linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('KL Divergence')
axes[0, 2].set_title('KL Divergence')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Feller penalty
axes[1, 0].plot(history['epoch'], history['train_feller'], label='Train', linewidth=2)
axes[1, 0].plot(history['epoch'], history['val_feller'], label='Val', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Feller Penalty')
axes[1, 0].set_title('Feller Penalty')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Arbitrage penalty
axes[1, 1].plot(history['epoch'], history['train_arbitrage'], label='Train', linewidth=2)
axes[1, 1].plot(history['epoch'], history['val_arbitrage'], label='Val', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Arbitrage Penalty')
axes[1, 1].set_title('Arbitrage Penalty')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Learning rate
axes[1, 2].plot(history['epoch'], history['lr'], linewidth=2, color='green')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Learning Rate')
axes[1, 2].set_title('Learning Rate')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_yscale('log')

plt.suptitle('Conditional VAE Training History', fontsize=16, fontweight='bold')
plt.tight_layout()

plot_path = os.path.join(results_dir, config['output']['loss_plots'])
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f" Training plots saved: {plot_path}")

print(f"\nAll results saved to: {results_dir}")
