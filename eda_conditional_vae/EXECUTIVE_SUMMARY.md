# Executive Summary: Conditional VAE for Heston Parameters

## TL;DR

**We have strong statistical evidence to implement a Conditional Variational Autoencoder (CVAE) that learns Heston parameter distributions conditioned on 8 market/macro variables.**

### **Key Numbers:**
-  **Strongest correlation**: r = 0.681 (v₀ ↔ India VIX)
-  **85% of relationships** are statistically significant (p < 0.05)
-  **Mutual information** up to 0.97 (non-linear dependencies)
-  **All 5 Heston parameters** show significant conditioning dependencies

---

## What We Analyzed

### **Target Variables** (Heston Parameters)
```
5 parameters × 500 daily observations (2016-2025)
├── κ (kappa): Mean reversion speed
├── θ (theta): Long-term variance
├── σᵥ (sigma_v): Volatility of volatility
├── ρ (rho): Asset-volatility correlation
└── v₀ (v0): Initial variance
```

### **Conditioning Variables** (Market/Macro)
```
8 variables × 500 daily observations
├── crude_oil_30d_mean: Oil price (30d avg)
├── crude_oil_7d_mean: Oil price (7d avg)
├── unrest_index_yearly: Geopolitical uncertainty
├── crude_oil: Current oil price
├── usdinr_quarterly_mean: USD/INR FX rate (quarterly avg)
├── india_vix_30d_mean: Volatility index (30d avg)
├── india_vix_7d_mean: Volatility index (7d avg)
└── us_10y_yield: US Treasury yield
```

---

## Key Findings

### 1. **Strong Correlations Exist** 

| Parameter | Best Predictor | Correlation | p-value |
|-----------|---------------|-------------|---------|
| **v₀** | India VIX (7d) | **+0.681** | < 0.001 |
| **σᵥ** | Crude Oil | **-0.552** | < 0.001 |
| **κ** | India VIX (30d) | **+0.522** | < 0.001 |
| **ρ** | Crude Oil | **-0.425** | < 0.001 |
| **θ** | Unrest Index | **+0.388** | < 0.001 |

### 2. **Statistical Significance is High** 

Out of 40 possible parameter-variable pairs:
- **34 pairs (85%)** are significant at p < 0.05
- **29 pairs (72.5%)** are significant at p < 0.01
- **24 pairs (60%)** are significant at p < 0.001

### 3. **Non-Linear Dependencies Detected** 

Mutual information analysis reveals:
- **USD/INR**: MI = 0.97 (extremely high)
- **Crude Oil (30d)**: MI = 0.67 (high)
- **Unrest Index**: MI = 0.61 (high)
- **India VIX (30d)**: MI = 0.58 (moderate-high)

**Implication**: Relationships go beyond linear - perfect for neural networks!

---

## Why Conditional VAE?

### **Problem with Standard VAE**
```python
# Standard VAE ignores market conditions
vae = VAE()
params = vae.sample()  # Random, not market-aware 
```

### **Solution: Conditional VAE**
```python
# CVAE conditions on market state
cvae = ConditionalVAE()
params = cvae.sample(conditions={
    'india_vix': 2.5,      # High volatility regime
    'crude_oil': -1.5,     # Low oil prices
    'unrest_index': 2.0    # High uncertainty
})  # Generates regime-specific parameters 
```

---

## Business Value

### **Use Cases Enabled:**

1. **Stress Testing** 
   - Generate IV surfaces under extreme market conditions
   - Example: "What if VIX spikes to 40 AND oil crashes?"

2. **Scenario Analysis** 
   - Target specific market regimes
   - Example: "Post-election scenarios with high uncertainty"

3. **Risk Management** 
   - Conditional tail risk assessment
   - Example: "Risk profile during geopolitical crisis"

4. **Forecasting** 
   - Use predicted macro variables to forecast IV surfaces
   - Example: "Fed rate decision impact on IV surface"

5. **Market Interpolation** 
   - Smooth transitions between market regimes
   - Example: "Gradual shift from low to high vol regime"

---

## Architecture Overview

```
Input: Heston Parameters θ (5D) + Conditions c (8D)
         ↓
    Encoder: q(z|θ,c)
         ↓
    Latent Space z (3D)
         ↓
    Decoder: p(θ|z,c) ← Also conditioned on c
         ↓
    Output: Reconstructed θ̂ (5D)
```

**Key Innovation**: Conditioning variables are fed to BOTH encoder and decoder, allowing the model to learn `p(θ|c)` instead of just `p(θ)`.

---

## Expected Benefits

### **Quantitative Improvements:**
-  **-15% to -25%** reduction in reconstruction loss
-  **Better out-of-sample** performance (leverages market context)
-  **More realistic** samples (regime-aligned)

### **Qualitative Improvements:**
-  **Interpretable**: Explain generations via conditions
-  **Controllable**: Fine-grained scenario generation
-  **Practical**: Directly addresses business needs
-  **Testable**: Easy to validate with historical regimes

---

## Comparison with Alternatives

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Standard VAE** | Simple | Ignores market context |  Not recommended |
| **Direct Regression** | Very simple | No uncertainty, single output |  Too limited |
| **Gaussian Process** | Uncertainty quantification | O(n³) complexity, limited dims |  Doesn't scale |
| **Conditional VAE** | Full distribution p(θ\|c), scalable, flexible | Slightly more complex |  **RECOMMENDED** |

---

## Implementation Roadmap

### **Phase 1: Baseline** (Week 1-2)
- [ ] Implement basic CVAE with concatenation
- [ ] Train on historical data
- [ ] Validate reconstruction quality
- [ ] Compare with standard VAE

### **Phase 2: Optimization** (Week 3-4)
- [ ] Hyperparameter tuning
- [ ] Test conditioning strategies
- [ ] Add Feller & arbitrage penalties
- [ ] Performance benchmarking

### **Phase 3: Advanced** (Week 5-6)
- [ ] Conditional prior p(z|c)
- [ ] Attention mechanisms
- [ ] Ensemble models
- [ ] Web interface

### **Phase 4: Deployment** (Week 7-8)
- [ ] Production integration
- [ ] Stress test validation
- [ ] API development
- [ ] Documentation

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **Conditioning may not generalize** | Use rolling features (30d, 7d); diverse variables |
| **Training complexity** | Start simple (concatenation); proven architectures |
| **Constraint violations** | Strong Feller penalty; arbitrage penalty; projection |
| **Overfitting to conditions** | Regularization; validation on unseen regimes |

---

## Recommendation

###  **IMPLEMENT CONDITIONAL VAE**

**Confidence Level**: **HIGH**

**Reasoning**:
1.  Strong statistical evidence (85% significant relationships)
2.  High information content (MI up to 0.97)
3.  Clear business value (stress testing, scenarios, forecasting)
4.  Proven architecture (literature support)
5.  Low risk (can fall back to standard VAE)

**Expected Outcome**:
- Significantly improved IV surface generation
- Regime-aware scenario capabilities
- Better risk management tools
- Competitive advantage in stress testing

---

## Files Generated

### ** Analysis & Documentation**
- `CONDITIONAL_VAE_CASE.md` - Full detailed case (14KB)
- `EXECUTIVE_SUMMARY.md` - This document
- `README.md` - Quick reference guide

### ** Visualizations**
- `SUMMARY_INFOGRAPHIC.png` - One-page visual summary
- `correlation_heatmap.png` - Pearson correlations
- `mutual_information_heatmap.png` - Non-linear dependencies
- `heston_timeseries.png` - Parameter evolution
- `conditioning_timeseries.png` - Variable evolution
- `distributions.png` - Parameter distributions
- `scatter_*.png` - Top correlations (5 files)

### ** Data Files**
- `merged_data.csv` - Combined dataset (500 × 14)
- `correlations.csv` - Correlation matrix
- `mutual_information.csv` - MI scores
- `p_values.csv` - Significance tests

### ** Code**
- `explore_data.py` - Main EDA script
- `create_summary_infographic.py` - Visualization script

---

## Next Action

**Start implementation of Conditional VAE architecture immediately.**

Estimated timeline: **2-3 weeks** for baseline, **6-8 weeks** for production-ready system.

Expected ROI: **High** - Better risk management + scenario capabilities + competitive advantage.

---

## Contact & Questions

For technical questions about the analysis:
- Review `CONDITIONAL_VAE_CASE.md` for detailed methodology
- Check `explore_data.py` for implementation details
- Examine visualizations for intuition

For implementation planning:
- Start with architecture proposal in case document
- Follow roadmap in Phase 1
- Iterate based on validation results

---

**Status**:  Analysis Complete - Ready for Implementation  
**Date**: November 2025  
**Next Milestone**: CVAE Baseline Implementation
