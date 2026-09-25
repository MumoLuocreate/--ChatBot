@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT_NO_SLASH=%PROJECT_ROOT:~0,-1%"

powershell.exe -NoProfile -ExecutionPolicy Bypass ^
  -File "%PROJECT_ROOT%scripts\stop_production.ps1" ^
  -ProjectRoot "%PROJECT_ROOT_NO_SLASH%"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
