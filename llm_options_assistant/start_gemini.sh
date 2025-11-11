#!/bin/bash

# Startup script for Options Analysis Assistant (Google Gemini - FREE!)

echo "🤖 Starting NIFTY 50 Options Analysis Assistant (Gemini)..."
echo ""

# Check if API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY not set"
    echo ""
    echo "Get your FREE API key (no credit card needed!):"
    echo "  1. Go to: https://aistudio.google.com/app/apikey"
    echo "  2. Click 'Create API Key'"
    echo "  3. Copy the key"
    echo ""
    echo "Then set it:"
    echo "  export GEMINI_API_KEY='your-key-here'"
    echo ""
    echo "Or add to ~/.zshrc for permanent use:"
    echo "  echo 'export GEMINI_API_KEY=\"your-key-here\"' >> ~/.zshrc"
    echo "  source ~/.zshrc"
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Run the assistant
cd "$PROJECT_DIR"
./venv/bin/python llm_options_assistant/options_analyst_gemini.py
