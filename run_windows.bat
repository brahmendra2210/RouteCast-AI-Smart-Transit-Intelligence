@echo off
title BusBoard AI Project
python --version >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or 3.11 and enable "Add Python to PATH".
  pause
  exit /b 1
)
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
pause
