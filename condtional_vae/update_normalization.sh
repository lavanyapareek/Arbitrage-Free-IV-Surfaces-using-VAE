#!/bin/bash

echo "========================================================================"
echo "UPDATING NORMALIZATION STATISTICS (2015 - Present)"
echo "========================================================================"
echo ""
echo "This will:"
echo "  1. Fetch GDELT unrest index data (2015 onwards, no 2020 cap)"
echo "  2. Recompute normalization statistics using all available data"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Step 1: Fetch GDELT data
echo "Step 1: Fetching GDELT unrest index data..."
echo "------------------------------------------------------------------------"
./venv/bin/python condtional_vae/fetch_and_compute_unrest_index.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error: GDELT fetch failed"
    exit 1
fi

echo ""
echo "Step 2: Computing normalization statistics..."
echo "------------------------------------------------------------------------"
./venv/bin/python condtional_vae/compute_normalization_stats.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error: Normalization computation failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ UPDATE COMPLETE!"
echo "========================================================================"
echo ""
echo "Updated files:"
echo "  - condtional_vae/gdelt_india_unrest_index.csv"
echo "  - condtional_vae/conditioning_normalization_stats.pt"
echo ""
echo "You can now:"
echo "  1. Retrain the model with updated data"
echo "  2. Generate surfaces with updated normalization"
echo ""
