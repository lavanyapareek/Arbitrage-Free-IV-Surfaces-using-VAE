# Single Heston Calibration with Wasserstein Penalty

This folder contains the implementation for calibrating **one Heston model per day** across all strikes and maturities, producing a clean 5-parameter dataset ready for VAE training.

## Overview

### Key Difference from Two-Stage Calibration

| Aspect | Two-Stage (Previous) | Single Heston (This) |
|--------|---------------------|---------------------|
| **Parameters per day** | 40 (5 params × 8 maturities) | **5 (one set)** |
| **Model complexity** | 8 separate Heston models | **1 unified model** |
| **Fit quality** | Better (more degrees of freedom) | Slightly worse (more constrained) |
| **Interpretability** | Lower (40D space) | **Higher (5D space)** |
| **VAE training** | More complex | **Simpler, cleaner** |

## Methodology

### Loss Function

```
L(θ) = RMSE_prices + λ_W * W_1 + λ_drift * drift_penalty
```

Where:
- **RMSE_prices**: Root mean squared error of option prices across all strikes and maturities
- **W_1**: Wasserstein distance between market and model risk-neutral densities (averaged across maturities)
- **drift_penalty**: ATM price matching term

### Two-Stage Optimization

**Stage 1: Fast Calibration**
- Minimize: `RMSE_prices + drift_penalty`
- No Wasserstein computation (fast)
- 10 random starts for robustness
- Gets parameters ~90% of the way there

**Stage 2: Wasserstein Refinement**
- Minimize: `RMSE_prices + 0.1*W_1 + drift_penalty`
- Uses Stage 1 result as initial guess
- Computes densities on 50-point grid
- Polishes with density matching

### Wasserstein Distance Computation

1. **Density Extraction**:
   ```
   f(K) = exp(rT) * ∂²C/∂K²
   ```
   Computed via central finite differences on a 50-point uniform grid

2. **Wasserstein Distance**:
   ```
   W_1(f_model, f_market) = ∫ |F_model(x) - F_market(x)| dx
   ```
   L¹ distance between CDFs

3. **Aggregation**:
   Simple average across all 8 maturities

## Files

```
calibration_single_heston/
├── config.json                          # Configuration
├── run_single_heston_calibration.py    # Main calibration script
├── test_setup.py                       # Test on one surface
└── README.md                           # This file
```

## Configuration (`config.json`)

### Key Parameters

```json
{
  "calibration": {
    "max_fit_error": 0.5,  // More lenient than per-maturity (0.3)
    "use_previous_day_on_failure": true  // Temporal continuity
  },
  
  "loss_function": {
    "wasserstein_weight": 0.1,
    "drift_penalty_weight": 1.0
  },
  
  "wasserstein": {
    "density_grid_points": 50,  // Balance accuracy vs speed
    "maturity_aggregation": "simple_average"
  },
  
  "stage1_fast": {
    "num_random_starts": 10  // Robustness
  }
}
```

## Usage

### 1. Test Setup

```bash
cd calibration_single_heston
python test_setup.py
```

Expected output:
```
 SETUP TEST COMPLETE!
All systems operational.
```

### 2. Run Full Calibration

```bash
python run_single_heston_calibration.py
```

**Expected time**: ~4-5 hours for 500 surfaces

**Progress output**:
```
[DEBUG] Processing day 1/500: 2016-02-09
      [Stage 1] Fast calibration...
      [Stage 1]  Complete in 15.2s | Loss: 0.234567
      [Stage 2] Wasserstein refinement...
      [Stage 2]  Complete in 18.5s | Loss: 0.198765

   Day 1/500 (2016-02-09): SUCCESS
    Fit error: 0.198765 | Feller: True
    Stage 1: 15.2s | Stage 2: 18.5s
```

### 3. Output Files

After completion:

1. **`NIFTY_heston_single_params.pickle`**
   - Full calibration results
   - Timing statistics
   - Fit errors and Feller checks

2. **`NIFTY_heston_single_params_tensor.pt`**
   - PyTorch tensor: `(n_days, 5)`
   - Format: `[kappa, theta, sigma_v, rho, v0]`
   - **Ready for VAE training!**

3. **`heston_single_calibration_errors.png`**
   - Fit errors over time
   - Parameter distributions

## Expected Results

### Performance

- **Success rate**: ~85-90% (similar to two-stage)
- **Time per day**: ~30-40 seconds
  - Stage 1: ~15 seconds
  - Stage 2: ~20 seconds
- **Total time**: ~4-5 hours for 500 surfaces

### Fit Quality

- **Fit error**: 0.2-0.5 (vs 0.001-0.01 for per-maturity)
- **Trade-off**: Worse fit but more parsimonious model
- **Benefit**: Much simpler latent space for VAE

### Parameter Statistics

Expected ranges (from historical NIFTY data):
- **κ (kappa)**: 0.5 - 10.0 (mean reversion speed)
- **θ (theta)**: 0.02 - 0.3 (long-term variance)
- **σ_v (sigma_v)**: 0.1 - 2.0 (vol of vol)
- **ρ (rho)**: -0.9 - -0.3 (correlation)
- **v₀ (v0)**: 0.02 - 0.3 (initial variance)

## Failure Handling

If calibration fails for a day:
1. **First attempt**: Use previous day's parameters
2. **Maintains**: Temporal continuity
3. **Preserves**: Sample size for VAE training

## Comparison with Two-Stage

| Metric | Two-Stage | Single Heston |
|--------|-----------|---------------|
| **Output dimension** | 40 | **5** |
| **Fit error** | 0.001-0.01 | 0.2-0.5 |
| **Time per day** | ~40s | ~35s |
| **VAE latent dim** | 5-10 | **2-3** |
| **Interpretability** | Low | **High** |
| **Overfitting risk** | Higher | **Lower** |

## Next Steps: VAE Training

After calibration, train a VAE on the 5-parameter dataset:

```python
import torch

# Load calibrated parameters
params = torch.load('calibration_single_heston/NIFTY_heston_single_params_tensor.pt')

# Shape: (n_days, 5)
# Format: [kappa, theta, sigma_v, rho, v0]

# Apply transforms (log, atanh) before VAE training
# See train_vae_single_heston.py for full implementation
```

### Recommended VAE Architecture

```json
{
  "input_dim": 5,
  "latent_dim": 3,
  "encoder_hidden": [128, 64],
  "decoder_hidden": [64, 128],
  "epochs": 400,
  "beta": 1.0,
  "kl_annealing": true
}
```

## Advantages of Single Heston Approach

1. **Simpler latent space**: 2-3D vs 5-10D
2. **Better interpretability**: Each latent dimension has clearer meaning
3. **Lower overfitting**: Fewer parameters to fit
4. **Faster VAE training**: Smaller input dimension
5. **Easier visualization**: Can plot 2D/3D latent space
6. **More stable**: Less prone to mode collapse

## Troubleshooting

### High Failure Rate
- Increase `num_random_starts` in Stage 1 (e.g., 15-20)
- Relax `max_fit_error` threshold (e.g., 0.7)
- Check if data quality is poor for those dates

### Slow Performance
- Reduce `density_grid_points` to 30-40
- Reduce `num_random_starts` to 5-7
- Skip Wasserstein entirely (set `wasserstein_weight: 0.0`)

### Poor Fit Quality
- This is expected! Single Heston cannot fit as well as per-maturity
- Acceptable range: 0.2-0.5
- Focus on parameter stability rather than perfect fit

### Feller Violations
- Common with aggressive fitting
- Can add soft penalty: `+ λ_feller * max(0, σ_v² - 2κθ)`
- Or just clip parameters post-calibration

## Technical Details

### Why Single Heston?

The time-homogeneous Heston model assumes constant parameters across maturities:
```
dS_t = rS_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ_v√v_t dW_t^v
dW_t^S dW_t^v = ρ dt
```

**Advantages**:
- Parsimonious (5 parameters vs 40)
- Theoretically grounded
- Better for regime identification

**Disadvantages**:
- Cannot capture term structure perfectly
- Higher calibration error
- May miss maturity-specific effects

### Wasserstein vs IV RMSE

**Wasserstein penalty**:
- Enforces distributional consistency
- Prevents overfitting sparse data
- Captures tail behavior
- More robust to outliers

**IV RMSE only**:
- Faster to compute
- May overfit market noise
- Ignores density properties
- Can produce unrealistic tails

## References

- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Breeden, D. & Litzenberger, R. (1978). "Prices of State-Contingent Claims"
- Wasserstein metric for probability distributions
- Time-homogeneous vs time-varying Heston models

---

**Ready to calibrate!** 

Run `python test_setup.py` to verify, then `python run_single_heston_calibration.py` for full calibration.
