@echo off
REM Launch script for Streamlit Dashboard (Windows)
REM Usage: run_dashboard.bat

echo.
echo 🚀 Launching Regime-Aware Trading System Dashboard...
echo.
echo Dashboard will open in your browser at: http://localhost:8501
echo.
echo To stop the dashboard, press Ctrl+C
echo.

cd /d "%~dp0"

REM Run Streamlit
streamlit run app\streamlit_app.py --server.port 8501 --server.headless false

pause
