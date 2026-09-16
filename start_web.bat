@echo off
pushd "%~dp0phishing_detector"
echo [WEB] Dang khoi dong Streamlit...
echo Sau khi chay xong, mo trinh duyet: http://localhost:8501
"%~dp0venv312\Scripts\streamlit.exe" run app.py --server.port 8501
pause
