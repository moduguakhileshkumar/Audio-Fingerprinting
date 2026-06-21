@echo off
title Running EE200 Streamlit App
echo ===================================================
echo Launching the EE200 Audio Fingerprinting Website...
echo ===================================================
echo.
streamlit run app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [Error] Failed to start the Streamlit web application.
    echo Please ensure that you have run install_dependencies.bat first.
    pause
)
