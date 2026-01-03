@echo off
chcp 65001 >nul
echo ========================================
echo    停止期货策略回测系统
echo ========================================
echo.

docker-compose down
echo.
echo ✅ 已停止
pause
