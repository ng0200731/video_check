@echo off
echo Installing dependencies...
py -m pip install flask opencv-python numpy --quiet

echo.
echo Starting Label Inspection at http://localhost:5000
echo Press Ctrl+C to stop.
echo.

start http://localhost:5000
cd /d "%~dp0tools"
py web_app.py
