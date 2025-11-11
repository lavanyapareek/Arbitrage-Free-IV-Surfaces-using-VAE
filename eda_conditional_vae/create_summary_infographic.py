"""
Create a summary infographic for the Conditional VAE case
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Conditional VAE for Heston Parameters: Evidence Summary', 
             fontsize=20, fontweight='bold', y=0.98)

# ============================================================================
# 1. Top Correlations
# ============================================================================

ax1 = plt.subplot(2, 3, 1)
params = ['v₀', 'σᵥ', 'κ', 'ρ', 'θ']
corrs = [0.681, -0.552, 0.522, -0.425, 0.388]
colors = ['green' if c > 0 else 'red' for c in corrs]

bars = ax1.barh(params, np.abs(corrs), color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('|Pearson Correlation|', fontsize=11, fontweight='bold')
ax1.set_title('Top Correlations\n(Parameter ↔ Best Conditioning Variable)', 
              fontsize=12, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.grid(axis='x', alpha=0.3)

# Add correlation values
for i, (bar, corr) in enumerate(zip(bars, corrs)):
    ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{corr:+.3f}', va='center', fontsize=10, fontweight='bold')

# ============================================================================
# 2. Statistical Significance
# ============================================================================

ax2 = plt.subplot(2, 3, 2)
params_sig = ['κ', 'σᵥ', 'ρ', 'v₀', 'θ']
sig_counts = [8, 8, 7, 6, 5]
total = 8

bars = ax2.bar(params_sig, sig_counts, color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_ylabel('# Significant Relationships', fontsize=11, fontweight='bold')
ax2.set_title('Statistical Significance\n(p < 0.05, out of 8 variables)', 
              fontsize=12, fontweight='bold')
ax2.set_ylim(0, 9)
ax2.axhline(total, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Total Variables')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

# Add percentages
for bar, count in zip(bars, sig_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.2,
             f'{count}/{total}\n({100*count/total:.0f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# 3. Mutual Information
# ============================================================================

ax3 = plt.subplot(2, 3, 3)
cond_vars = ['USD/INR\nQtrly', 'Crude Oil\n30d', 'Unrest\nIndex', 'India VIX\n30d', 'US 10Y\nYield']
mi_scores = [0.97, 0.67, 0.61, 0.58, 0.47]

bars = ax3.bar(range(len(cond_vars)), mi_scores, color='coral', alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(cond_vars)))
ax3.set_xticklabels(cond_vars, fontsize=9)
ax3.set_ylabel('Avg Mutual Information', fontsize=11, fontweight='bold')
ax3.set_title('Non-Linear Dependencies\n(Top 5 Conditioning Variables)', 
              fontsize=12, fontweight='bold')
ax3.set_ylim(0, 1.1)
ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax3.grid(axis='y', alpha=0.3)

# Add MI values
for i, (bar, mi) in enumerate(zip(bars, mi_scores)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{mi:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# 4. Architecture Diagram
# ============================================================================

ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)

# Title
ax4.text(5, 9.5, 'Conditional VAE Architecture', 
         ha='center', fontsize=13, fontweight='bold')

# Input
box1 = FancyBboxPatch((0.5, 6.5), 2, 2, boxstyle="round,pad=0.1", 
                       facecolor='lightblue', edgecolor='black', linewidth=2)
ax4.add_patch(box1)
ax4.text(1.5, 7.5, 'Heston\nParams\nθ (5D)', ha='center', va='center', 
         fontsize=10, fontweight='bold')

# Conditioning
box2 = FancyBboxPatch((0.5, 3.5), 2, 2, boxstyle="round,pad=0.1", 
                       facecolor='lightcoral', edgecolor='black', linewidth=2)
ax4.add_patch(box2)
ax4.text(1.5, 4.5, 'Conditioning\nVariables\nc (8D)', ha='center', va='center', 
         fontsize=10, fontweight='bold')

# Encoder
box3 = FancyBboxPatch((3.5, 5), 2, 2, boxstyle="round,pad=0.1", 
                       facecolor='lightgreen', edgecolor='black', linewidth=2)
ax4.add_patch(box3)
ax4.text(4.5, 6, 'Encoder\nq(z|θ,c)', ha='center', va='center', 
         fontsize=10, fontweight='bold')

# Latent
box4 = FancyBboxPatch((6.5, 5), 1.5, 2, boxstyle="round,pad=0.1", 
                       facecolor='lightyellow', edgecolor='black', linewidth=2)
ax4.add_patch(box4)
ax4.text(7.25, 6, 'Latent\nz (3D)', ha='center', va='center', 
         fontsize=10, fontweight='bold')

# Decoder
box5 = FancyBboxPatch((3.5, 1), 2, 2, boxstyle="round,pad=0.1", 
                       facecolor='lightgreen', edgecolor='black', linewidth=2)
ax4.add_patch(box5)
ax4.text(4.5, 2, 'Decoder\np(θ|z,c)', ha='center', va='center', 
         fontsize=10, fontweight='bold')

# Output
box6 = FancyBboxPatch((0.5, 1), 2, 2, boxstyle="round,pad=0.1", 
                       facecolor='lightblue', edgecolor='black', linewidth=2)
ax4.add_patch(box6)
ax4.text(1.5, 2, 'Reconstructed\nθ̂ (5D)', ha='center', va='center', 
         fontsize=10, fontweight='bold')

# Arrows
arrow1 = FancyArrowPatch((2.5, 7.5), (3.5, 6.5), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax4.add_patch(arrow1)

arrow2 = FancyArrowPatch((2.5, 4.5), (3.5, 5.5), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax4.add_patch(arrow2)

arrow3 = FancyArrowPatch((5.5, 6), (6.5, 6), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax4.add_patch(arrow3)

arrow4 = FancyArrowPatch((7.25, 5), (4.5, 3), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax4.add_patch(arrow4)

arrow5 = FancyArrowPatch((3.5, 2), (2.5, 2), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax4.add_patch(arrow5)

arrow6 = FancyArrowPatch((2.5, 4.5), (3.5, 2.5), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black', linestyle='--')
ax4.add_patch(arrow6)

# ============================================================================
# 5. Key Insights
# ============================================================================

ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)

insights = [
    "✅ Strong evidence: 34/40 relationships significant",
    "✅ v₀ highly responsive to India VIX (r=0.681)",
    "✅ σᵥ inversely tracks crude oil (r=-0.552)",
    "✅ High mutual information (up to 0.97)",
    "✅ Non-linear relationships detected",
    "✅ All parameters show dependencies",
]

ax5.text(5, 9.5, 'Key Evidence', ha='center', fontsize=13, fontweight='bold')

for i, insight in enumerate(insights):
    y_pos = 8.5 - i * 1.3
    ax5.text(0.5, y_pos, insight, fontsize=10, va='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

# ============================================================================
# 6. Benefits
# ============================================================================

ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)

benefits = [
    "🎯 Regime-aware generation",
    "📊 Stress testing capability",
    "🔮 Market forecasting integration",
    "🎛️ Fine-grained control",
    "📈 Better reconstruction",
    "💼 Business value: Risk mgmt",
]

ax6.text(5, 9.5, 'CVAE Benefits', ha='center', fontsize=13, fontweight='bold')

for i, benefit in enumerate(benefits):
    y_pos = 8.5 - i * 1.3
    ax6.text(0.5, y_pos, benefit, fontsize=10, va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# ============================================================================
# Add overall status
# ============================================================================

fig.text(0.5, 0.02, 
         '⭐ RECOMMENDATION: Implement Conditional VAE - Strong Statistical Evidence + High Business Value ⭐',
         ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round,pad=1', facecolor='gold', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('SUMMARY_INFOGRAPHIC.png', dpi=300, bbox_inches='tight')
print("\n✅ Summary infographic saved: SUMMARY_INFOGRAPHIC.png")
plt.close()

print("\n" + "="*80)
print("INFOGRAPHIC COMPLETE")
print("="*80)
