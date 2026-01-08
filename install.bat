@echo off
chcp 65001 > nul
echo.
echo ===================================
echo   期货量化交易系统 v2.0 安装脚本
echo ===================================
echo.

:: 检查Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [1/3] 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
    echo      虚拟环境已创建
) else (
    echo      虚拟环境已存在，跳过
)

echo.
echo [2/3] 激活虚拟环境并安装依赖...
call venv\Scripts\activate.bat
pip install -r requirements.txt -q

echo.
echo [3/3] 安装完成！
echo.
echo ===================================
echo   使用方法：
echo   1. 启动Web界面: python run.py web
echo   2. 或双击 start_web.bat
echo ===================================
echo.
pause
