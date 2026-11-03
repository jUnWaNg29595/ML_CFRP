@echo off
chcp 65001 >nul
echo ========================================
echo CFRP预测平台 WSL启动器
echo ========================================
echo.

set "PORTAL_PORT=8501"
echo 正在检查端口 %PORTAL_PORT% 占用情况...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $pids=Get-NetTCPConnection -LocalPort 8501 -State Listen | Select-Object -ExpandProperty OwningProcess -Unique; foreach($targetPid in $pids){ $proc=Get-CimInstance Win32_Process -Filter ('ProcessId='+$targetPid); if($proc -and $proc.CommandLine -match 'streamlit' -and $proc.CommandLine -match 'UserPrediction\.py'){ Stop-Process -Id $targetPid -Force; Write-Host ('已关闭预测平台旧进程 PID '+$targetPid) } elseif($proc){ Write-Host ('端口 8501 被其他程序占用（PID '+$targetPid+'，非本项目进程，不会关闭）'); Write-Host '请更换端口或手动释放该端口后重试。'; exit 1 } }"
if %errorlevel%==1 (
    echo [错误] 端口 %PORTAL_PORT% 无法释放，启动中止。
    pause
    exit /b 1
)

echo 正在通过 WSL (Ubuntu-24.04) 启动 CFRP 系统...
echo 访问地址: http://localhost:%PORTAL_PORT%
echo 按 Ctrl+C 停止服务
echo.

wsl.exe -d Ubuntu-24.04 -u wangj -- bash -lc "cd /home/wangj/project/CFRP && /home/wangj/project/CFRP/run_wsl.sh"

pause
