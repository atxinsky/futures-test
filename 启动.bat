@echo off
chcp 65001 >nul
echo ========================================
echo    期货策略回测系统
echo ========================================
echo.

:: 检查Docker是否运行
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] Docker未运行，使用本地模式启动...
    echo.
    goto :local
)

echo 选择启动方式:
echo   1. Docker启动 (推荐)
echo   2. 本地启动 (需要Python环境)
echo.
set /p choice="请输入选择 (1/2): "

if "%choice%"=="1" goto :docker
if "%choice%"=="2" goto :local
goto :docker

:docker
echo.
echo [Docker模式] 正在启动...
docker-compose up -d --build
echo.
echo ✅ 启动成功!
echo 📊 访问地址: http://localhost:8502
echo.
echo 查看日志: docker-compose logs -f
echo 停止服务: docker-compose down
goto :end

:local
echo.
echo [本地模式] 正在启动...
echo.

:: 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

:: 检查依赖
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 正在安装依赖...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

:: 启动Streamlit
echo.
echo ✅ 正在启动 Streamlit...
echo 📊 访问地址: http://localhost:8502
echo.
start "" http://localhost:8502
streamlit run app.py --server.port 8502

:end
pause
