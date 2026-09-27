@echo off
cd /d %~dp0
python scripts\bootstrap_train_forecast.py
pause
