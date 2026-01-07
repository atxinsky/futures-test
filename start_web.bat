@echo off
chcp 65001 >nul
echo ====================================
echo   期货量化交易系统 v2.0 - Web界面
echo ====================================
echo.

cd /d "%~dp0"

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

:: 启动Web界面
echo 正在启动Web界面 (端口: 8504)...
echo 访问地址: http://localhost:8504
echo.
python run.py web -p 8504
pause
