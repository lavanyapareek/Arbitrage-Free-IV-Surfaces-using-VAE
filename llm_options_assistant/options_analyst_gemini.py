#!/usr/bin/env python3
"""
Interactive Options Analysis Assistant - Google Gemini Version (FREE!)
Uses Google Gemini API (no credit card required)
"""

import os
import sys
import subprocess
import json
from datetime import datetime
import pandas as pd
import google.generativeai as genai

# ============================================================================
# Configuration
# ============================================================================

SYSTEM_PROMPT = """You are an expert options analyst and quantitative researcher specializing in NIFTY 50 options in India.

Your capabilities:
1. Generate IV (Implied Volatility) surfaces for any date using a trained Conditional VAE model
2. Analyze volatility smiles, term structures, and skew patterns
3. Identify trading opportunities (spreads, straddles, butterflies, etc.)
4. Assess market sentiment from volatility patterns
5. Calculate option Greeks and risk metrics
6. Compare surfaces across different dates
7. Provide actionable trading recommendations

When a user asks for analysis:
1. First, offer to generate the IV surface for their requested date
2. Once generated, analyze the surface data (8 maturities × 21 strikes)
3. Provide insights on volatility structure, market sentiment, and trading opportunities
4. Use specific numbers and percentages from the data
5. Be concise but thorough

The IV surface grid:
- Maturities: 1M, 2M, 3M, 6M, 9M, 12M, 18M, 24M
- Strikes: Log-moneyness from -20% to +20% (21 strikes)
- Values: Implied volatility in percentage

Always start by introducing yourself and explaining what you can do.
"""

# ============================================================================
# Helper Functions
# ============================================================================

def generate_iv_surface(date_str):
    """Generate IV surface for a given date using the CVAE model (best_model_2025)"""
    print(f"\n Generating IV surface for {date_str}...")
    print("This may take 30-60 seconds...\n")
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(script_dir, '.venv', 'bin', 'python')
    gen_script = os.path.join(script_dir, 'condtional_vae', 'generate_iv_surface_by_date.py')
    
    try:
        result = subprocess.run(
            [venv_python, gen_script, '--date', date_str, '--n_samples', '100'],
            cwd=os.path.join(script_dir, 'condtional_vae'),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return {
                'status': 'error',
                'message': f"Failed to generate surface. Error: {result.stderr[-500:]}"
            }
        
        output_lines = result.stdout.split('\n')
        spot_price = None
        valid_surfaces = None
        
        for line in output_lines:
            if 'Spot price:' in line:
                spot_price = float(line.split(':')[1].strip())
            if 'Valid surfaces:' in line:
                parts = line.split()
                valid_surfaces = parts[2]
        
        output_dir = os.path.join(script_dir, 'condtional_vae', 'results_date', date_str)
        mean_csv = os.path.join(output_dir, 'mean_iv_surface.csv')
        median_csv = os.path.join(output_dir, 'median_iv_surface.csv')
        
        if not os.path.exists(mean_csv):
            return {
                'status': 'error',
                'message': 'Surface files not found. Generation may have failed.'
            }
        
        mean_df = pd.read_csv(mean_csv, index_col='Maturity')
        median_df = pd.read_csv(median_csv, index_col='Maturity')
        
        print(f" Surface generated successfully!")
        print(f"   Spot: {spot_price:.2f}, Valid surfaces: {valid_surfaces}\n")
        
        return {
            'status': 'success',
            'date': date_str,
            'spot_price': spot_price,
            'valid_surfaces': valid_surfaces,
            'mean_surface': mean_df,
            'median_surface': median_df,
            'output_dir': output_dir
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error: {str(e)}'
        }


def format_surface_for_llm(surface_data):
    """Format surface data for LLM analysis"""
    if surface_data['status'] != 'success':
        return surface_data['message']
    
    mean_df = surface_data['mean_surface']
    
    atm_col = '0.0%'
    atm_term_structure = mean_df[atm_col].to_dict()
    smile_1m = mean_df.loc['1M'].to_dict()
    smile_6m = mean_df.loc['6M'].to_dict()
    smile_12m = mean_df.loc['12M'].to_dict()
    
    summary = f"""
IV SURFACE DATA FOR {surface_data['date']}
{'='*60}

METADATA:
- Date: {surface_data['date']}
- NIFTY Spot: {surface_data['spot_price']:.2f}
- Valid Surfaces: {surface_data['valid_surfaces']}

ATM TERM STRUCTURE (Mean IV %):
{json.dumps(atm_term_structure, indent=2)}

VOLATILITY SMILE AT 1M: {json.dumps(smile_1m, indent=2)}
VOLATILITY SMILE AT 6M: {json.dumps(smile_6m, indent=2)}
VOLATILITY SMILE AT 12M: {json.dumps(smile_12m, indent=2)}

FULL MEAN IV SURFACE:
{mean_df.to_string()}

STATS: Mean={mean_df.values.mean():.2f}%, Min={mean_df.values.min():.2f}%, Max={mean_df.values.max():.2f}%
"""
    return summary


# ============================================================================
# Main Chat Loop
# ============================================================================

def main():
    """Main interactive chat loop using Google Gemini"""
    
    # Check for Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print(" Error: GEMINI_API_KEY environment variable not set")
        print("\nGet your FREE API key:")
        print("  1. Go to: https://aistudio.google.com/app/apikey")
        print("  2. Click 'Create API Key'")
        print("  3. Copy the key")
        print("\nThen set it:")
        print("  export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Use the latest fast model (gemini-2.5-flash is free and fast)
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    # Start chat
    chat = model.start_chat(history=[])
    
    # Print welcome message
    print("\n" + "="*70)
    print(" NIFTY 50 OPTIONS ANALYSIS ASSISTANT (Powered by Google Gemini)")
    print("="*70)
    print("\nInitializing AI analyst...")
    
    # Get initial greeting
    try:
        response = chat.send_message(SYSTEM_PROMPT + "\n\nPlease introduce yourself and explain what you can do.")
        greeting = response.text
        print(f"\n{greeting}\n")
    except Exception as e:
        print(f" Error connecting to Gemini: {e}")
        print("\nMake sure:")
        print("  1. Your API key is valid")
        print("  2. You have internet connection")
        sys.exit(1)
    
    print("="*70)
    print("Type 'quit' or 'exit' to end the conversation")
    print("="*70 + "\n")
    
    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n Goodbye! Happy trading!\n")
                break
            
            # Get LLM response
            response = chat.send_message(user_input)
            assistant_message = response.text
            
            # Check if LLM is asking to generate a surface
            if 'generate' in assistant_message.lower() and ('surface' in assistant_message.lower() or 'date' in user_input.lower()):
                print(f"\nAssistant: {assistant_message}\n")
                
                date_input = input("Please enter the date (YYYY-MM-DD) or 'skip': ").strip()
                
                if date_input.lower() != 'skip':
                    try:
                        datetime.strptime(date_input, '%Y-%m-%d')
                        
                        # Generate surface
                        surface_data = generate_iv_surface(date_input)
                        
                        if surface_data['status'] == 'success':
                            surface_text = format_surface_for_llm(surface_data)
                            
                            # Send to LLM for analysis
                            analysis_prompt = f"Here is the IV surface data:\n\n{surface_text}\n\nPlease provide a comprehensive analysis."
                            analysis_response = chat.send_message(analysis_prompt)
                            
                            analysis = analysis_response.text
                            print(f"\nAssistant: {analysis}\n")
                        else:
                            error_msg = f"Failed to generate surface: {surface_data['message']}"
                            print(f"\n {error_msg}\n")
                    except ValueError:
                        print("\n Invalid date format. Please use YYYY-MM-DD\n")
            else:
                print(f"\nAssistant: {assistant_message}\n")
        
        except KeyboardInterrupt:
            print("\n\n Goodbye! Happy trading!\n")
            break
        except Exception as e:
            print(f"\n Error: {e}\n")
            continue


if __name__ == "__main__":
    main()
