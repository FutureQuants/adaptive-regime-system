#!/bin/bash

# Launch script for Streamlit Dashboard
# Usage: ./run_dashboard.sh

echo "🚀 Launching Regime-Aware Trading System Dashboard..."
echo ""
echo "Dashboard will open in your browser at: http://localhost:8501"
echo ""
echo "To stop the dashboard, press Ctrl+C"
echo ""

cd "$(dirname "$0")"

# Run Streamlit
streamlit run app/streamlit_app.py --server.port 8501 --server.headless false