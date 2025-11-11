# EDA: Conditional VAE for Heston Parameters

## Overview

This folder contains a comprehensive **Exploratory Data Analysis (EDA)** that makes a strong case for using a **Conditional Variational Autoencoder (CVAE)** architecture to learn the distribution of Heston parameters conditioned on market and macro-economic variables.

---

## 📊 Key Findings

### **Strongest Correlations Discovered:**

| Heston Parameter | Conditioning Variable | Correlation | p-value |
|-----------------|----------------------|-------------|---------|
| **v₀** (initial variance) | india_vix_7d_mean | **r = +0.681** | p < 0.001 ⭐⭐⭐ |
| **σ_v** (vol of vol) | crude_oil | **r = -0.552** | p < 0.001 ⭐⭐⭐ |
| **κ** (mean reversion) | india_vix_30d_mean | **r = +0.522** | p < 0.001 ⭐⭐ |
| **ρ** (correlation) | crude_oil | **r = -0.425** | p < 0.001 ⭐⭐ |
| **θ** (long-term var) | unrest_index_yearly | **r = +0.388** | p < 0.001 ⭐⭐ |

### **Statistical Significance:**
- **All 5 Heston parameters** show significant relationships with conditioning variables (p < 0.05)
- **34 out of 40** parameter-variable pairs are statistically significant
- High mutual information scores (up to **0.97**) indicate strong non-linear dependencies

---

## 📁 Files Generated

### **Analysis Scripts**
- **`explore_data.py`** - Main EDA script that performs all analysis

### **Documentation**
- **`CONDITIONAL_VAE_CASE.md`** - ⭐ **Comprehensive case document** with:
  - Statistical evidence for conditional VAE
  - Architecture proposals
  - Implementation roadmap
  - Expected benefits
  - Use cases and examples

### **Data Files**
- **`merged_data.csv`** (500 × 14) - Combined Heston params + conditioning variables
- **`correlations.csv`** - Pearson correlation matrix
- **`mutual_information.csv`** - Non-linear dependency scores
- **`p_values.csv`** - Statistical significance tests

### **Visualizations**

#### 1. **Correlation Analysis**
- **`correlation_heatmap.png`** - Pearson correlations between all variables
- **`mutual_information_heatmap.png`** - Non-linear dependency heatmap

#### 2. **Time Series**
- **`heston_timeseries.png`** - Evolution of Heston parameters over time
- **`conditioning_timeseries.png`** - Evolution of conditioning variables over time

#### 3. **Distributions**
- **`distributions.png`** - Distribution plots with KDE for all Heston parameters

#### 4. **Scatter Plots** (Top 3 correlations per parameter)
- **`scatter_kappa.png`** - κ vs top conditioning variables
- **`scatter_theta.png`** - θ vs top conditioning variables
- **`scatter_sigma_v.png`** - σ_v vs top conditioning variables
- **`scatter_rho.png`** - ρ vs top conditioning variables
- **`scatter_v0.png`** - v₀ vs top conditioning variables

---

## 🚀 Quick Start

### Run the EDA:

```bash
cd eda_conditional_vae
../venv/bin/python explore_data.py
```

### View the Case Document:

```bash
# Open in any markdown viewer or editor
open CONDITIONAL_VAE_CASE.md
```

---

## 📈 Data Summary

### **Heston Parameters** (Target Variables)
```
Dataset: 500 samples (2016-02-09 to 2025-06-27)
Dimensions: 5
Variables: [kappa, theta, sigma_v, rho, v0]
```

### **Conditioning Variables** (Input Features)
```
Dimensions: 8
Variables:
  1. crude_oil_30d_mean      - 30-day oil price rolling mean
  2. crude_oil_7d_mean       - 7-day oil price rolling mean
  3. unrest_index_yearly     - Geopolitical unrest index
  4. crude_oil               - Current crude oil price
  5. usdinr_quarterly_mean   - USD/INR quarterly mean
  6. india_vix_30d_mean      - India VIX 30-day mean
  7. india_vix_7d_mean       - India VIX 7-day mean
  8. us_10y_yield            - US 10-year Treasury yield
```

---

## 💡 Why Conditional VAE?

### **Problem with Standard VAE:**
```python
# Standard VAE learns: p(θ)
# Ignores market conditions
vae = VAE(latent_dim=3)
z = torch.randn(1, 3)
params = vae.decode(z)  # Random, not regime-aware
```

### **Solution with Conditional VAE:**
```python
# Conditional VAE learns: p(θ|c)
# Conditions on market state
cvae = ConditionalVAE(latent_dim=3, cond_dim=8)

# Generate for high volatility regime
c_high_vol = torch.tensor([[
    crude_oil=-1.5,
    india_vix=2.5,
    us_10y_yield=1.0,
    ...
]])
params = cvae.decode(z, c_high_vol)  # Regime-specific!
```

---

## 🎯 Use Cases Enabled

1. **Stress Testing**
   - Generate Heston params under extreme market conditions
   - Example: VIX spike + oil crash scenario

2. **Scenario Analysis**
   - Target specific market regimes
   - Example: Post-election uncertainty (high unrest index)

3. **Forecasting**
   - Use predicted macro variables to forecast IV surfaces
   - Example: Fed rate decision scenarios

4. **Risk Management**
   - Conditional tail risk assessment
   - Example: Crisis scenarios with high correlations

5. **Market Interpolation**
   - Smooth transitions between market regimes
   - Example: Gradual vol regime shifts

---

## 📊 Key Statistics

### **Correlation Strength Distribution:**
```
Very Strong (|r| > 0.6): 1 pair   (v₀ ↔ india_vix_7d)
Strong      (|r| > 0.5): 3 pairs  (σ_v ↔ crude_oil, etc.)
Moderate    (|r| > 0.4): 7 pairs
Weak        (|r| > 0.3): 8 pairs
```

### **Mutual Information:**
```
Highest:  usdinr_quarterly_mean (avg MI = 0.97)
Lowest:   india_vix_7d_mean     (avg MI = 0.38)
```

### **Statistical Significance:**
```
Total parameter-variable pairs:  40
Significant at p < 0.05:         34  (85%)
Significant at p < 0.01:         29  (72.5%)
Significant at p < 0.001:        24  (60%)
```

---

## 🏗️ Proposed Architecture

```python
class ConditionalVAE(nn.Module):
    def __init__(self):
        # Input: θ (5D) + c (8D)
        self.encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2*latent_dim)  # μ and log σ²
        )
        
        # Input: z (latent) + c (8D)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # θ reconstruction
        )
```

---

## 📖 Recommended Reading Order

1. **Start here**: `CONDITIONAL_VAE_CASE.md` (Executive summary + case)
2. **Visual evidence**: View all `.png` files
3. **Raw data**: Check `.csv` files for detailed numbers
4. **Code**: Review `explore_data.py` for methodology

---

## 🔍 Interpretation Guide

### **Reading Correlation Values:**
- **r > +0.5**: Strong positive relationship
- **r < -0.5**: Strong negative relationship
- **0.3 < |r| < 0.5**: Moderate relationship
- **|r| < 0.3**: Weak relationship

### **Reading Mutual Information:**
- **MI > 0.8**: Very high information content
- **0.5 < MI < 0.8**: High information content
- **0.3 < MI < 0.5**: Moderate information content
- **MI < 0.3**: Low information content

### **Reading p-values:**
- **p < 0.001**: Very strong evidence (⭐⭐⭐)
- **p < 0.01**: Strong evidence (⭐⭐)
- **p < 0.05**: Significant evidence (⭐)
- **p ≥ 0.05**: Not significant

---

## 🎓 Key Insights

1. **v₀ (Initial Variance) is highly responsive to India VIX**
   - When VIX spikes, initial variance increases proportionally
   - Correlation: +0.681 (strongest relationship found)

2. **σ_v (Vol of Vol) inversely tracks Crude Oil**
   - Higher oil prices → Lower volatility of volatility
   - Correlation: -0.552 (strong negative)

3. **κ (Mean Reversion) increases with VIX levels**
   - Higher volatility → Faster mean reversion
   - Correlation: +0.522 (moderate positive)

4. **Multiple variables matter**
   - No single variable dominates all parameters
   - Justifies using all 8 conditioning variables

5. **Non-linear relationships exist**
   - Mutual information > Correlation suggests complexity
   - Neural network approach (CVAE) appropriate

---

## 🔬 Methodology

### **Analyses Performed:**
1. ✅ Pearson correlation (linear relationships)
2. ✅ Spearman correlation (monotonic relationships)
3. ✅ Mutual information (non-linear dependencies)
4. ✅ Statistical significance testing
5. ✅ Time series co-movement analysis
6. ✅ Distribution analysis
7. ✅ Scatter plot visualization

### **Quality Checks:**
- ✅ No missing values
- ✅ Perfect date alignment (500/500 samples matched)
- ✅ All variables normalized (standard scaling)
- ✅ Statistical assumptions verified

---

## 📞 Next Steps

1. **Immediate**: Review `CONDITIONAL_VAE_CASE.md` for full case
2. **Short-term**: Implement basic conditional VAE
3. **Medium-term**: Benchmark vs standard VAE
4. **Long-term**: Deploy for production use

---

## 📚 References

- Kingma & Welling (2013): "Auto-Encoding Variational Bayes"
- Sohn et al. (2015): "Learning Structured Output Representation using Deep Conditional Generative Models"
- Pearson correlation, Mutual Information theory
- Heston model calibration methodology

---

**Status**: ✅ **Analysis Complete - CVAE Recommended**  
**Confidence**: **High** (Strong statistical evidence + practical value)  
**Next Action**: Implement Conditional VAE architecture
