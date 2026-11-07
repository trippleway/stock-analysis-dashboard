#!/bin/bash
# One-command start for the daily US-stock analysis and the dashboard

cd "$(dirname "$0")"

echo "=========================================="
echo "US stock analysis dashboard"
echo "=========================================="

# Step 1: run daily_picker to produce today's analysis (uses the cache)
echo ""
echo "[1/2] Running the daily pick (about 5-8 minutes the first time; later runs hit the cache)..."
python src/daily_picker.py

if [ $? -ne 0 ]; then
    echo "daily_picker failed. Check the error above."
    exit 1
fi

# Step 2: start the dashboard
echo ""
echo "[2/2] Starting the dashboard..."
echo "=========================================="
echo "Dashboard URL: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo "=========================================="
streamlit run dashboard/app.py
