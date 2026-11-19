@echo off
rem Plotline-AI launcher: double-click to start the desktop application.
setlocal
set "ROOT=%~dp0"
set "PY="
if defined PLOTLINE_PYTHON set "PY=%PLOTLINE_PYTHON%"
if not defined PY if exist "%USERPROFILE%\micromamba\envs\roadmarking-ai\python.exe" set "PY=%USERPROFILE%\micromamba\envs\roadmarking-ai\python.exe"
if not defined PY if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY (
  echo Could not find the roadmarking-ai Python environment.
  echo Set PLOTLINE_PYTHON to its python.exe and try again.
  pause
  exit /b 1
)
for %%I in ("%PY%") do set "ENVDIR=%%~dpI"
set "GDAL_DATA=%ENVDIR%Library\share\gdal"
set "GDAL_DRIVER_PATH=%ENVDIR%Library\lib\gdalplugins"
set "PROJ_DATA=%ENVDIR%Library\share\proj"
set "PATH=%ENVDIR%;%ENVDIR%Library\bin;%PATH%"
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"
cd /d "%ROOT%"
start "Plotline-AI" "%ENVDIR%pythonw.exe" -m plotline_ai
endlocal
