"""
Exploratory Data Analysis: Heston Parameters & Conditioning Variables
Goal: Make a case for Conditional VAE architecture
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS: CONDITIONAL VAE FEASIBILITY")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n1. Loading data...")

# Load Heston parameters
heston_path = '../calibration_single_heston/NIFTY_heston_single_params_tensor.pt'
heston_params = torch.load(heston_path)
print(f"    Heston parameters: {heston_params.shape}")
print(f"     Format: [kappa, theta, sigma_v, rho, v0]")

# Load conditioning variables
cond_path = '../conditioning_variables_top8.csv'
cond_vars = pd.read_csv(cond_path)
print(f"    Conditioning variables: {cond_vars.shape}")
print(f"     Columns: {list(cond_vars.columns)}")

# Check date alignment
dates_cond = pd.to_datetime(cond_vars['Unnamed: 0'])
print(f"\n   Date range:")
print(f"     Start: {dates_cond.min()}")
print(f"     End:   {dates_cond.max()}")
print(f"     Count: {len(dates_cond)}")

# ============================================================================
# 2. DATA ALIGNMENT & PREPROCESSING
# ============================================================================

print("\n2. Data alignment...")

# Convert to DataFrame
param_names = ['kappa', 'theta', 'sigma_v', 'rho', 'v0']
heston_df = pd.DataFrame(heston_params.numpy(), columns=param_names)
heston_df['date'] = dates_cond

# Prepare conditioning variables with date
cond_vars_clean = cond_vars.rename(columns={'Unnamed: 0': 'date'})
cond_vars_clean['date'] = pd.to_datetime(cond_vars_clean['date'])

# Merge datasets
df_merged = pd.merge(
    heston_df, 
    cond_vars_clean,
    on='date',
    how='inner'
)

print(f"    Merged dataset: {df_merged.shape}")
print(f"    Matched samples: {len(df_merged)}")

# Extract conditioning variable names (exclude date)
cond_var_names = [col for col in cond_vars.columns if col != 'Unnamed: 0']
print(f"\n   Conditioning variables ({len(cond_var_names)}):")
for i, var in enumerate(cond_var_names, 1):
    print(f"     {i}. {var}")

# Save merged dataset
df_merged.to_csv('merged_data.csv', index=False)
print(f"\n    Saved: merged_data.csv")

# ============================================================================
# 3. BASIC STATISTICS
# ============================================================================

print("\n3. Basic statistics...")

print("\n   Heston Parameters:")
print(df_merged[param_names].describe())

print("\n   Conditioning Variables:")
print(df_merged[cond_var_names].describe())

# Check for missing values
print("\n   Missing values:")
missing = df_merged.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("     None ")

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================

print("\n4. Computing correlations...")

# Correlation between conditioning variables and Heston parameters
correlations = pd.DataFrame(
    index=cond_var_names,
    columns=param_names
)

for cond_var in cond_var_names:
    for param in param_names:
        # Pearson correlation
        corr, p_value = pearsonr(df_merged[cond_var], df_merged[param])
        correlations.loc[cond_var, param] = corr

correlations = correlations.astype(float)

print("\n   Correlation Matrix (Pearson):")
print(correlations.round(3))

# Save correlations
correlations.to_csv('correlations.csv')
print(f"\n    Saved: correlations.csv")

# Find strongest correlations
print("\n   Strongest correlations:")
for param in param_names:
    sorted_corr = correlations[param].abs().sort_values(ascending=False)
    top_var = sorted_corr.index[0]
    top_corr = correlations.loc[top_var, param]
    print(f"     {param:10s} <-> {top_var:25s}: {top_corr:+.3f}")

# ============================================================================
# 5. VISUALIZATION: CORRELATION HEATMAP
# ============================================================================

print("\n5. Creating visualizations...")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    correlations.T,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    cbar_kws={'label': 'Pearson Correlation'},
    ax=ax
)
ax.set_title('Correlation: Conditioning Variables → Heston Parameters', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Conditioning Variables', fontsize=12)
ax.set_ylabel('Heston Parameters', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"    Saved: correlation_heatmap.png")
plt.close()

# ============================================================================
# 6. TIME SERIES ANALYSIS
# ============================================================================

print("\n6. Time series analysis...")

fig, axes = plt.subplots(5, 1, figsize=(15, 12))

for i, param in enumerate(param_names):
    ax = axes[i]
    ax.plot(df_merged['date'], df_merged[param], linewidth=1.5, alpha=0.8)
    ax.set_ylabel(param, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.set_title('Heston Parameters Over Time', fontsize=14, fontweight='bold')
    if i < 4:
        ax.set_xticklabels([])

axes[-1].set_xlabel('Date', fontsize=12)
plt.tight_layout()
plt.savefig('heston_timeseries.png', dpi=300, bbox_inches='tight')
print(f"    Saved: heston_timeseries.png")
plt.close()

# Conditioning variables time series
n_vars = len(cond_var_names)
fig, axes = plt.subplots(n_vars, 1, figsize=(15, 2*n_vars))

for i, var in enumerate(cond_var_names):
    ax = axes[i]
    ax.plot(df_merged['date'], df_merged[var], linewidth=1.5, alpha=0.8, color='orange')
    ax.set_ylabel(var.replace('_', ' ').title(), fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.set_title('Conditioning Variables Over Time', fontsize=14, fontweight='bold')
    if i < n_vars - 1:
        ax.set_xticklabels([])

axes[-1].set_xlabel('Date', fontsize=12)
plt.tight_layout()
plt.savefig('conditioning_timeseries.png', dpi=300, bbox_inches='tight')
print(f"    Saved: conditioning_timeseries.png")
plt.close()

# ============================================================================
# 7. SCATTER PLOTS: TOP CORRELATIONS
# ============================================================================

print("\n7. Creating scatter plots for top correlations...")

# For each Heston parameter, plot against top 3 conditioning variables
for param in param_names:
    sorted_corr = correlations[param].abs().sort_values(ascending=False)
    top_3_vars = sorted_corr.head(3).index.tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Heston Parameter: {param.upper()}', fontsize=14, fontweight='bold')
    
    for i, cond_var in enumerate(top_3_vars):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(df_merged[cond_var], df_merged[param], alpha=0.5, s=20)
        
        # Add trend line
        z = np.polyfit(df_merged[cond_var], df_merged[param], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_merged[cond_var].min(), df_merged[cond_var].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Correlation value
        corr_val = correlations.loc[cond_var, param]
        ax.text(0.05, 0.95, f'r = {corr_val:.3f}', 
                transform=ax.transAxes, fontsize=11, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(cond_var.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel(param, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'scatter_{param}.png', dpi=300, bbox_inches='tight')
    print(f"    Saved: scatter_{param}.png")
    plt.close()

# ============================================================================
# 8. DISTRIBUTION ANALYSIS
# ============================================================================

print("\n8. Distribution analysis...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, param in enumerate(param_names):
    ax = axes[i]
    
    # Histogram with KDE
    ax.hist(df_merged[param], bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df_merged[param])
    x_range = np.linspace(df_merged[param].min(), df_merged[param].max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax.set_xlabel(param, fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = df_merged[param].mean()
    std_val = df_merged[param].std()
    ax.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'μ={mean_val:.3f}')

axes[-1].axis('off')  # Hide last subplot

fig.suptitle('Distribution of Heston Parameters', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
print(f"    Saved: distributions.png")
plt.close()

# ============================================================================
# 9. MUTUAL INFORMATION
# ============================================================================

print("\n9. Computing mutual information...")

from sklearn.feature_selection import mutual_info_regression

mi_scores = pd.DataFrame(
    index=cond_var_names,
    columns=param_names
)

for param in param_names:
    X = df_merged[cond_var_names].values
    y = df_merged[param].values
    
    mi = mutual_info_regression(X, y, random_state=42)
    mi_scores[param] = mi

mi_scores = mi_scores.astype(float)

print("\n   Mutual Information Scores:")
print(mi_scores.round(4))

# Save MI scores
mi_scores.to_csv('mutual_information.csv')
print(f"\n    Saved: mutual_information.csv")

# Visualize MI scores
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    mi_scores.T,
    annot=True,
    fmt='.4f',
    cmap='YlOrRd',
    cbar_kws={'label': 'Mutual Information'},
    ax=ax
)
ax.set_title('Mutual Information: Conditioning Variables → Heston Parameters', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Conditioning Variables', fontsize=12)
ax.set_ylabel('Heston Parameters', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('mutual_information_heatmap.png', dpi=300, bbox_inches='tight')
print(f"    Saved: mutual_information_heatmap.png")
plt.close()

# ============================================================================
# 10. STATISTICAL SIGNIFICANCE
# ============================================================================

print("\n10. Statistical significance testing...")

# Test each correlation for significance
p_values = pd.DataFrame(
    index=cond_var_names,
    columns=param_names
)

for cond_var in cond_var_names:
    for param in param_names:
        _, p_value = pearsonr(df_merged[cond_var], df_merged[param])
        p_values.loc[cond_var, param] = p_value

p_values = p_values.astype(float)

# Count significant correlations (p < 0.05)
significant = (p_values < 0.05).sum()

print("\n   Significant correlations (p < 0.05):")
for param in param_names:
    count = (p_values[param] < 0.05).sum()
    print(f"     {param:10s}: {count}/{len(cond_var_names)} conditioning variables")

# Save p-values
p_values.to_csv('p_values.csv')
print(f"\n    Saved: p_values.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {os.getcwd()}")
