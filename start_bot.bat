@echo off
setlocal
set "PROJECT_ROOT=%~dp0"

set "QICHI_NAPCAT_ROOT=E:\NapCatQQ"
set "QICHI_QQ_EXE=E:\QQ\QQ.exe"

start "Qichi Production" /D "%PROJECT_ROOT%" powershell.exe ^
  -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden ^
  -File "%PROJECT_ROOT%scripts\start_production.ps1"

if errorlevel 1 (
  echo qichi start: failed to create the production process
  endlocal
  exit /b 1
)
echo qichi start: launched in the current user session; check runtime\start-production.log and scripts\check_ready.py
endlocal
exit /b 0
