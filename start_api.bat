@echo off
pushd "%~dp0phishing_detector"
echo [API] Dang khoi dong... Cho 20-30 giay de load models
"%~dp0venv312\Scripts\python.exe" -m uvicorn api:app --host 0.0.0.0 --port 8000
pause
