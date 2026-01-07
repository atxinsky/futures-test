@echo off
chcp 65001 >nul
echo ====================================
echo   期货量化交易系统 v2.0 - TqSdk交易
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

:: 检查tqsdk
python -c "import tqsdk" >nul 2>&1
if errorlevel 1 (
    echo 错误: TqSdk未安装，正在安装...
    pip install tqsdk
)

:: 显示选项
echo 请选择运行模式:
echo   1. TqSdk模拟盘
echo   2. TqSdk实盘
echo   3. 自定义启动
echo.
set /p choice="请输入选项 (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo 启动TqSdk模拟盘...
    python run.py live --sim
) else if "%choice%"=="2" (
    echo.
    echo 启动TqSdk实盘...
    echo 警告: 实盘交易有风险，请确认配置正确！
    set /p confirm="确认启动实盘? (y/n): "
    if /i "%confirm%"=="y" (
        python run.py live --real
    ) else (
        echo 已取消
    )
) else if "%choice%"=="3" (
    echo.
    set /p symbols="输入交易品种 (如 RB,AU,IF): "
    set /p capital="输入初始资金 (默认 100000): "
    if "%capital%"=="" set capital=100000
    python run.py live -s %symbols% -c %capital% --sim
) else (
    echo 无效选项
)

pause
