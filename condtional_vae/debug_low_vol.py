import torch
import numpy as np

# Load the regime samples
data = torch.load('results/regime_samples/all_regimes_heston_params.pt', weights_only=False)

low_vol_params = data['regimes']['low_volatility'].numpy()
high_vol_params = data['regimes']['high_volatility'].numpy()

print("="*80)
print("PARAMETER ANALYSIS: LOW vs HIGH VOLATILITY")
print("="*80)

print("\nLOW VOLATILITY Parameters:")
print(f"  kappa:   mean={low_vol_params[:, 0].mean():.4f}, min={low_vol_params[:, 0].min():.4f}, max={low_vol_params[:, 0].max():.4f}")
print(f"  theta:   mean={low_vol_params[:, 1].mean():.4f}, min={low_vol_params[:, 1].min():.4f}, max={low_vol_params[:, 1].max():.4f}")
print(f"  sigma_v: mean={low_vol_params[:, 2].mean():.4f}, min={low_vol_params[:, 2].min():.4f}, max={low_vol_params[:, 2].max():.4f}")
print(f"  rho:     mean={low_vol_params[:, 3].mean():.4f}, min={low_vol_params[:, 3].min():.4f}, max={low_vol_params[:, 3].max():.4f}")
print(f"  v0:      mean={low_vol_params[:, 4].mean():.4f}, min={low_vol_params[:, 4].min():.4f}, max={low_vol_params[:, 4].max():.4f}")

print("\nHIGH VOLATILITY Parameters:")
print(f"  kappa:   mean={high_vol_params[:, 0].mean():.4f}, min={high_vol_params[:, 0].min():.4f}, max={high_vol_params[:, 0].max():.4f}")
print(f"  theta:   mean={high_vol_params[:, 1].mean():.4f}, min={high_vol_params[:, 1].min():.4f}, max={high_vol_params[:, 1].max():.4f}")
print(f"  sigma_v: mean={high_vol_params[:, 2].mean():.4f}, min={high_vol_params[:, 2].min():.4f}, max={high_vol_params[:, 2].max():.4f}")
print(f"  rho:     mean={high_vol_params[:, 3].mean():.4f}, min={high_vol_params[:, 3].min():.4f}, max={high_vol_params[:, 3].max():.4f}")
print(f"  v0:      mean={high_vol_params[:, 4].mean():.4f}, min={high_vol_params[:, 4].min():.4f}, max={high_vol_params[:, 4].max():.4f}")

print("\n" + "="*80)
print("POTENTIAL ISSUES:")
print("="*80)

# Check for extreme values
print("\nLow volatility concerns:")

# Very low v0
very_low_v0 = (low_vol_params[:, 4] < 0.005).sum()
print(f"  1. v0 < 0.005 (very low): {very_low_v0}/{len(low_vol_params)} ({100*very_low_v0/len(low_vol_params):.1f}%)")

# Very high rho (close to 1)
very_high_rho = (low_vol_params[:, 3] > 0.95).sum()
print(f"  2. rho > 0.95 (extreme positive): {very_high_rho}/{len(low_vol_params)} ({100*very_high_rho/len(low_vol_params):.1f}%)")

# Very low kappa
very_low_kappa = (low_vol_params[:, 0] < 0.1).sum()
print(f"  3. kappa < 0.1 (very slow mean reversion): {very_low_kappa}/{len(low_vol_params)} ({100*very_low_kappa/len(low_vol_params):.1f}%)")

# Very low sigma_v
very_low_sigmav = (low_vol_params[:, 2] < 0.01).sum()
print(f"  4. sigma_v < 0.01 (very stable vol): {very_low_sigmav}/{len(low_vol_params)} ({100*very_low_sigmav/len(low_vol_params):.1f}%)")

print("\nLikely causes of failure:")
print("  1. Very low v0 → Options nearly worthless → Hard to invert to IV")
print("  2. Extreme rho (>0.95) → Numerical instability in Heston pricing")
print("  3. Low kappa + low sigma_v → Very flat term structure → Numerical issues")
print("  4. Combination: Low volatility = low option prices = IV solver fails")

print("\n" + "="*80)
