@echo off
rem Builds dist\Plotline-AI-<version>-Setup.exe (+ .bin parts) with Inno Setup 6.
setlocal
set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
"%ISCC%" "%~dp0plotline_ai.iss"
endlocal
