@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 启动期货量化交易系统...
python run.py web
pause
