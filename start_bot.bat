@echo off
setlocal
set "PROJECT_ROOT=%~dp0"

rem NapCat and QQ live wherever you installed them, so this repository never
rem guesses.  Set the NapCat root once (and optionally the QQ executable):
rem     setx QICHI_NAPCAT_ROOT "D:\NapCatQQ"
rem     setx QICHI_QQ_EXE "C:\Program Files\Tencent\QQNT\QQ.exe"
rem Both variables are inherited by the process this script starts.
if not defined QICHI_NAPCAT_ROOT (
  echo qichi start: QICHI_NAPCAT_ROOT is not set.
  echo qichi start: point it at your NapCat root first, for example:
  echo     setx QICHI_NAPCAT_ROOT "D:\NapCatQQ"
  endlocal
  exit /b 2
)

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
