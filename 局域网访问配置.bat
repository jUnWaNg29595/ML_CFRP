@echo off
chcp 65001 >nul
echo ========================================
echo 局域网访问配置工具
echo ========================================
echo.

echo [1] 查看本机IP地址
echo ----------------------------------------
ipconfig | findstr "IPv4"
echo.

echo [2] 添加防火墙规则（允许8501端口）
echo ----------------------------------------
netsh advfirewall firewall delete rule name="Streamlit CFRP" >nul 2>&1
netsh advfirewall firewall add rule name="Streamlit CFRP" dir=in action=allow protocol=TCP localport=8501
if %errorlevel%==0 (
    echo ✓ 防火墙规则添加成功
) else (
    echo ✗ 防火墙规则添加失败（需要管理员权限）
    echo   请右键点击此文件，选择"以管理员身份运行"
)
echo.

echo [3] 启动Streamlit服务
echo ----------------------------------------
rem 支持自定义端口：主平台默认 8501，可用第一个参数覆盖（如：局域网访问配置.bat 8502）
set "MAIN_PORT=%~1"
if "%MAIN_PORT%"=="" set "MAIN_PORT=8501"
echo 正在检查端口 %MAIN_PORT% 占用情况...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $pids=Get-NetTCPConnection -LocalPort %MAIN_PORT% -State Listen | Select-Object -ExpandProperty OwningProcess -Unique; foreach($targetPid in $pids){ $proc=Get-CimInstance Win32_Process -Filter ('ProcessId='+$targetPid); if($proc -and $proc.CommandLine -match 'streamlit' -and $proc.CommandLine -match '\bapp\.py\b'){ Stop-Process -Id $targetPid -Force; Write-Host ('已关闭本系统旧进程 PID '+$targetPid) } elseif($proc){ Write-Host ('端口 %MAIN_PORT% 被其他程序占用（PID '+$targetPid+'，非本项目进程，不会关闭）'); Write-Host '请更换端口或手动释放该端口后重试。'; exit 1 } }"
if %errorlevel%==1 (
    echo [错误] 端口 %MAIN_PORT% 无法释放，启动中止。
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
echo 正在启动服务...
echo 局域网访问地址：http://你的IP:%MAIN_PORT%
echo 按 Ctrl+C 停止服务
echo.
streamlit run app.py --server.port %MAIN_PORT%

pause
