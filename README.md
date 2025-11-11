# Arbitrage-Free IV Surface Generation using Conditional VAE

A deep learning pipeline for generating arbitrage-free implied volatility surfaces for NIFTY 50 options using Conditional Variational Autoencoders (CVAE) conditioned on market variables.

## Project Overview

This project implements a complete pipeline for:
1. **Heston Model Calibration**: Fits stochastic volatility model parameters to market data
2. **Conditional VAE Training**: Learns the distribution of Heston parameters conditioned on market state
3. **IV Surface Generation**: Generates forward-looking volatility surfaces for any date
4. **AI-Powered Analysis**: Interactive LLM assistant for options analysis

## Key Features

- **Arbitrage-Free Surfaces**: Ensures no-arbitrage conditions using Heston model
- **Market Conditioning**: Incorporates VIX, USD/INR, crude oil, interest rates, and geopolitical unrest
- **Real-Time Generation**: Generate surfaces for any date in seconds
- **Interactive Analysis**: Chat with AI analyst for trading insights
- **Comprehensive Visualization**: ATM term structures, volatility smiles, heatmaps

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

### Running the Demo

Open `Scripts_Orchestration.ipynb` and follow the Quick Start Guide:

```bash
jupyter notebook Scripts_Orchestration.ipynb
```

**Recommended demo workflow (5-10 minutes):**
1. Run Section 1: Setup
2. Skip Sections 2-3 (uses pre-trained models)
3. Run Section 4: Generate IV surfaces
4. Run Section 5: View visualizations

## Project Structure

```
.
├── Scripts_Orchestration.ipynb      # Main demo notebook
├── requirements.txt                 # Python dependencies
├── api.json.template               # Template for API key
├── calibration_single_heston/      # Heston calibration scripts
│   ├── run_single_heston_calibration.py
│   └── config.json
├── condtional_vae/                 # CVAE training and generation
│   ├── train_cvae.py
│   ├── generate_iv_surface_by_date.py
│   └── model architecture files
├── llm_options_assistant/          # AI analyst integration
│   ├── options_analyst_gemini.py
│   └── best_model_2025/           # Pre-trained CVAE model
├── heston_model_ql.py             # Heston model implementation
└── demo_results/                   # Generated outputs (gitignored)
```

## Methodology

### 1. Heston Model Calibration

Fits a single Heston model per day using two-stage optimization:
- **Stage 1**: Fast calibration minimizing price RMSE
- **Stage 2**: Refinement with Wasserstein distance penalty

**Parameters**: κ (mean reversion), θ (long-term variance), σᵥ (vol of vol), ρ (correlation), v₀ (initial variance)

### 2. Conditional VAE Architecture

- **Encoder**: Maps Heston parameters to latent distribution
- **Decoder**: Reconstructs parameters from latent code
- **Conditioning**: 8 market variables (VIX, FX, commodities, rates, unrest)
- **Training**: KL divergence + reconstruction loss

### 3. Surface Generation

1. Extract current market conditions
2. Sample from conditional latent distribution
3. Decode to Heston parameters
4. Generate IV surface using Heston model
5. Validate arbitrage-free conditions

## Data Sources

- **Options Data**: NIFTY 50 options (2015-2025)
- **India VIX**: Volatility index from NSE
- **Market Data**: USD/INR, Crude Oil, US 10Y Yield (via yfinance)
- **Geopolitical**: GDELT unrest index for India

## LLM Assistant

Interactive AI analyst powered by Google Gemini:
- Generate and analyze surfaces on-demand
- Identify trading opportunities
- Explain volatility patterns
- Provide actionable recommendations

**Setup**: Get free API key from https://aistudio.google.com/app/apikey

## Results

- **Date Range**: 2020-01-01 to 2025-11-10
- **Grid**: 8 maturities × 21 strikes
- **Generation Time**: ~30-60 seconds per surface
- **Validation Rate**: ~30% of samples pass arbitrage checks

## Security Notes

**IMPORTANT**: Never commit sensitive files:
- `api.json` - Contains API keys
- `*credentials*.json` - Google Cloud credentials
- Large data files (>100 MB)

These are excluded via `.gitignore`. Use `api.json.template` for sharing.

## Citation

If you use this work, please cite:

```bibtex
@misc{arbitragefree-iv-cvae,
  title={Arbitrage-Free IV Surface Generation using Conditional VAE},
  author={Lavanya Pareek},
  year={2025},
  howpublished={\url{https://github.com/yourusername/your-repo}}
}
```



## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request



## Acknowledgments

- QuantLib for Heston model implementation
- Google Gemini for LLM capabilities
- GDELT Project for geopolitical data
- NSE for NIFTY options data

---

**Disclaimer**: This project is for educational and research purposes only. Not financial advice. Use at your own risk.
