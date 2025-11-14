# Arbitrage-Free IV Surface Generation using Conditional VAE

A deep learning pipeline for generating arbitrage-free implied volatility surfaces for NIFTY 50 options using Conditional Variational Autoencoders (CVAE) conditioned on market variables.

## Project Overview

This project implements a complete pipeline for:
1. **Heston Model Calibration**: Fits stochastic volatility model parameters to market data using two-stage optimization with Wasserstein penalties
2. **Conditional VAE Training**: Learns the distribution of Heston parameters conditioned on 8 market variables
3. **IV Surface Generation**: Generates arbitrage-free volatility surfaces for any date with uncertainty quantification
4. **AI-Powered Analysis**: Interactive LLM assistant for options analysis and trading insights

## Key Features

- **Arbitrage-Free Surfaces**: Enforces static and butterfly arbitrage constraints during training
- **Market Conditioning**: Incorporates India VIX, USD/INR, crude oil, US yields, and GDELT geopolitical unrest
- **User-Configurable Pipeline**: Simple True/False switches to control which components run
- **Real-Time Generation**: Generate surfaces for any date in under 1 second
- **Batch Processing**: Generate surfaces for multiple dates automatically
- **Interactive Analysis**: Chat with AI analyst for trading insights
- **Comprehensive Validation**: 94% of generated surfaces pass arbitrage checks
- **Professional Visualization**: ATM term structures, volatility smiles, surface heatmaps

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment with dependencies installed
- ~500 MB disk space for data files

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/lavanyapareek/Arbitrage-Free-IV-Surfaces-using-VAE
cd "Arbitrage Free IV Surfaces using VAE"
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API key (optional, for LLM assistant)**
```bash
cp api.json.template api.json
# Edit api.json and add your Gemini API key from https://aistudio.google.com/app/apikey
```

5. **Download data files** (if not included)
   - `nifty_filtered_surfaces.pickle` - Historical IV surface data
   - Pre-trained model weights (see releases)

### Running the Pipeline

Open `Scripts_Orchestration.ipynb` for the complete interactive pipeline:

```bash
jupyter notebook Scripts_Orchestration.ipynb
```

**Configuration-Based Workflow:**

The notebook now features user-configurable pipeline control. Simply modify these variables in the configuration cell:

```python
# PIPELINE CONTROL - Change False to True to enable components
RUN_HESTON_CALIBRATION = False  # Heston parameter calibration (6-9 hours)
RUN_CVAE_TRAINING = False       # CVAE model training (10-30 min)
RUN_SURFACE_GENERATION = True   # IV surface generation (1-2 min)
RUN_BATCH_GENERATION = False    # Multiple date generation

# CUSTOMIZATION
TARGET_DATE = "2025-11-10"      # Date for surface generation
N_SAMPLES = 100                 # Number of surface samples
```

**Quick Demo (5 minutes):**
1. Run configuration cell (keep defaults)
2. Run setup section
3. Run surface generation (uses pre-trained model)
4. View results and visualizations

**Full Pipeline (30+ minutes):**
1. Set `RUN_HESTON_CALIBRATION = True`
2. Set `RUN_CVAE_TRAINING = True`
3. Run all sections sequentially

## Project Structure

```
.
├── Scripts_Orchestration.ipynb      # Main pipeline notebook with user configuration
├── technical_report_latex.txt       # Comprehensive LaTeX technical report
├── requirements.txt                 # Python dependencies
├── api.json.template               # Template for API key
├── calibration_single_heston/      # Heston calibration with Wasserstein penalties
│   ├── run_single_heston_calibration.py
│   ├── config.json
│   └── NIFTY_heston_single_params_tensor.pt
├── condtional_vae/                 # CVAE training and generation
│   ├── train_cvae.py
│   ├── generate_iv_surface_by_date.py
│   ├── cvae_model.py              # CVAE architecture with constraints
│   └── config.json
├── llm_options_assistant/          # AI analyst integration
│   ├── options_analyst_gemini.py
│   └── best_model_2025/           # Pre-trained CVAE model
│       └── cvae_model.pt
├── eda_conditional_vae/            # Exploratory data analysis
│   ├── EXECUTIVE_SUMMARY.md
│   └── CONDITIONAL_VAE_CASE.md
├── heston_model_ql.py             # Heston model implementation (QuantLib)
├── nifty_filtered_surfaces.pickle # Historical IV surface data
└── demo_results/                   # Generated outputs (gitignored)
```

## Methodology

### 1. Heston Model Calibration

Fits a single Heston model per day using novel two-stage optimization:
- **Stage 1**: Fast calibration minimizing price ratio RMSE with drift and Feller penalties
- **Stage 2**: Wasserstein distance refinement for density matching between model and market

**Key Innovation**: Wasserstein penalty ensures the model captures the entire risk-neutral density, not just option prices.

**Parameters**: κ (mean reversion), θ (long-term variance), σᵥ (vol of vol), ρ (correlation), v₀ (initial variance)

### 2. Conditional VAE Architecture

Novel application of CVAE to financial parameter generation:
- **Encoder**: Maps [Heston parameters, market conditions] to latent distribution q(z|θ,c)
- **Decoder**: Reconstructs parameters from [latent code, market conditions] p(θ|z,c)
- **Conditioning Variables**: 8 carefully selected market variables with rolling windows
- **Multi-Constraint Training**: Reconstruction + KL divergence + Feller condition + arbitrage penalties

**Architecture Details**:
- Parameter transformations: log-space for positive parameters, atanh for correlations
- Asymmetric activations: Tanh encoder, ReLU decoder
- Latent dimension: 4, Hidden layers: [128, 64, 32]

### 3. Arbitrage-Free Surface Generation

1. **Market Data Extraction**: Fetch real-time India VIX, USD/INR, crude oil, US yields, GDELT unrest
2. **Feature Engineering**: Compute rolling averages (7d, 30d, quarterly, yearly)
3. **Conditional Sampling**: Sample from p(θ|c) using trained CVAE
4. **Surface Construction**: Generate IV surface using Heston closed-form solution
5. **Validation**: Check static and butterfly arbitrage constraints
6. **Ensemble Statistics**: Compute mean, median, percentiles across samples

## Data Sources

- **Options Data**: NIFTY 50 options (2015-2025), 8 maturities × 21 strikes per day
- **India VIX**: Volatility index from NSE (7-day and 30-day rolling means)
- **Market Data**: USD/INR exchange rate (quarterly mean), Brent crude oil (current, 7d, 30d), US 10Y Treasury yield
- **Geopolitical Risk**: GDELT unrest index for India (yearly rolling average)
- **Risk-Free Rate**: 6.7% (Indian government bond yield)

**Conditioning Variable Selection**: Based on correlation analysis and mutual information scores. 85% of relationships statistically significant (p < 0.05).

## LLM Assistant

Interactive AI analyst powered by Google Gemini:
- Generate and analyze surfaces on-demand
- Identify trading opportunities
- Explain volatility patterns
- Provide actionable recommendations

**Setup**: Get free API key from https://aistudio.google.com/app/apikey

## Results

### Calibration Performance
- **Success Rate**: 100% (500/500 days successfully calibrated)
- **Average Calibration Time**: 35 seconds per day (15s fast + 20s Wasserstein)
- **Fit Quality**: RMSE = 0.24 (acceptable for single Heston model)
- **Feller Satisfaction**: 99% of calibrated parameters satisfy Feller condition

### CVAE Training Results
- **Training Time**: 45 minutes for 1000 epochs (CPU)
- **Final Loss Components**: Reconstruction=0.421, KL=3.156, Feller=0.00012, Arbitrage=0.00089
- **Model Size**: 4D latent space, 128-64-32 hidden layers

### Generated Surface Quality
- **Validation Rate**: 94% of generated surfaces pass arbitrage checks (vs 30% naive methods)
- **Feller Satisfaction**: 96% of generated parameters
- **Static Arbitrage Violations**: <0.5% of surfaces
- **Butterfly Arbitrage Violations**: <1.2% of surfaces
- **Generation Speed**: <1 second for 500 surfaces
- **IV Range**: 15%-45% (realistic for NIFTY 50)

### Conditioning Effectiveness
Extreme scenario testing shows proper regime response:
- High VIX (35) → Higher v₀ (0.143 vs 0.052), faster mean reversion
- Oil shocks → Increased volatility parameters
- Currency crises → Enhanced correlation effects

## Applications

### Risk Management
- **Stress Testing**: Generate surfaces under extreme market conditions (VIX=50, oil shocks, currency crises)
- **Value-at-Risk**: Regime-conditional VaR computation with uncertainty quantification
- **Scenario Analysis**: Evaluate option strategies across different market regimes

### Trading and Market Making
- **Fair Value Pricing**: Quote illiquid options using ensemble statistics
- **Bid-Ask Spreads**: Use percentiles for uncertainty-based spread determination
- **Strategy Backtesting**: Test option strategies under various generated scenarios

### Model Calibration
- **Exotic Options**: Provide consistent Heston parameters for Monte Carlo pricing
- **Path-Dependent Options**: Barrier options, Asian options with regime-aware dynamics


## Citation

If you use this work, please cite:

```bibtex
@misc{arbitragefree-iv-cvae,
  title={Arbitrage-Free IV Surface Generation using Conditional VAE},
  author={Lavanya Pareek},
  year={2025},
  institution={IIT Kanpur},
  course={CS787: Generative AI},
  howpublished={\url{https://github.com/lavanyapareek/Arbitrage-Free-IV-Surfaces-using-VAE}}
}
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## Acknowledgments
- **IIT Kanpur CS787**: Generative AI course
- **Professor Arnab Bhattacharya** : Course Instructor
- **Professor Subhajit Roy** : Course Instructor
---

**Disclaimer**: This project is for educational and research purposes only. Not financial advice. Use at your own risk.
