@echo off
chcp 65001 >nul
echo ====================================
echo   期货量化交易系统 - Docker部署
echo ====================================
echo.

cd /d "%~dp0"

:: 检查Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Docker未安装或未启动
    pause
    exit /b 1
)

echo 请选择启动模式:
echo   1. Web界面 (只有Web，无自动交易)
echo   2. 模拟盘交易 (后台运行，无Web)
echo   3. 完整版 (Web + 模拟盘交易) [推荐]
echo   4. 重新构建镜像
echo   5. 查看日志
echo   6. 停止所有容器
echo.
set /p choice="请输入选项 (1-6): "

if "%choice%"=="1" (
    echo.
    echo 启动Web界面...
    docker compose up -d web
    echo.
    echo Web界面已启动: http://localhost:8504
) else if "%choice%"=="2" (
    echo.
    echo 启动模拟盘交易服务...
    docker compose up -d trading
    echo.
    echo 模拟盘交易服务已启动 (后台运行)
    echo 查看日志: docker logs -f futures-trading
) else if "%choice%"=="3" (
    echo.
    echo 启动完整版 (Web + 模拟盘)...
    docker compose up -d full
    echo.
    echo 完整版已启动:
    echo   - Web界面: http://localhost:8504
    echo   - 模拟盘: 后台运行中
    echo 查看日志: docker logs -f futures-full
) else if "%choice%"=="4" (
    echo.
    echo 重新构建镜像...
    docker compose build --no-cache
    echo 构建完成!
) else if "%choice%"=="5" (
    echo.
    echo 选择查看哪个容器的日志:
    echo   1. futures-web
    echo   2. futures-trading
    echo   3. futures-full
    set /p log_choice="请选择 (1-3): "
    if "%log_choice%"=="1" docker logs -f futures-web --tail 100
    if "%log_choice%"=="2" docker logs -f futures-trading --tail 100
    if "%log_choice%"=="3" docker logs -f futures-full --tail 100
) else if "%choice%"=="6" (
    echo.
    echo 停止所有容器...
    docker compose down
    echo 已停止
) else (
    echo 无效选项
)

pause
