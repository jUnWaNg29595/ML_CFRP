@echo off
chcp 65001 >nul
echo ========================================
echo 预测平台启动器
echo ========================================
echo.

set "PORTAL_PORT=8555"
echo 正在检查端口 %PORTAL_PORT% 占用情况...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $pids=Get-NetTCPConnection -LocalPort 8555 -State Listen | Select-Object -ExpandProperty OwningProcess -Unique; foreach($targetPid in $pids){ $proc=Get-CimInstance Win32_Process -Filter ('ProcessId='+$targetPid); if($proc -and $proc.CommandLine -match 'streamlit' -and $proc.CommandLine -match 'UserPrediction\.py'){ Stop-Process -Id $targetPid -Force; Write-Host ('已关闭预测平台旧进程 PID '+$targetPid) } elseif($proc){ Write-Host ('端口 8555 被其他程序占用（PID '+$targetPid+'，非本项目进程，不会关闭）'); Write-Host '请更换端口或手动释放该端口后重试。'; exit 1 } }"
if %errorlevel%==1 (
    echo [错误] 端口 %PORTAL_PORT% 无法释放，启动中止。
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul

echo 正在启动预测平台: http://localhost:%PORTAL_PORT%
echo 按 Ctrl+C 停止服务
echo.
rem [关键] 禁用 Intel Fortran 运行库(libifcoremd.dll, 由 conda sklearn 间接加载)的
rem Ctrl+C 控制台处理器：否则 Ctrl+C 只打印 forrtl: error (200) 并在退出阶段卡死，
rem 进程关不掉、端口不释放（Intel 官方开关 FOR_DISABLE_CONSOLE_CTRL_HANDLER）
set "FOR_DISABLE_CONSOLE_CTRL_HANDLER=1"
streamlit run UserPrediction.py --server.port %PORTAL_PORT%

pause
