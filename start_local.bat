@echo off
setlocal EnableExtensions
title ACG local

set "APP=%~dp0"
set "APP=%APP:~0,-1%"
set "VENV=%APP%\.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "CONFIG=%APP%\configs\app.yaml"
set "REQ=%APP%\requirements.txt"
set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%"

cd /d "%APP%" || (
  echo ERROR: no se pudo entrar a "%APP%"
  pause
  exit /b 1
)

echo.
echo === ACG local ===
echo App: %APP%
echo.

where py >nul 2>&1
if errorlevel 1 (
  where python >nul 2>&1
  if errorlevel 1 (
    echo ERROR: no se encontro Python en PATH.
    pause
    exit /b 1
  )
  set "PY=python"
) else (
  set "PY=py -3"
)

if not exist "%PYTHON%" (
  echo [1/3] Creando entorno virtual .venv ...
  %PY% -m venv "%VENV%"
  if errorlevel 1 (
    echo ERROR: fallo al crear el venv.
    pause
    exit /b 1
  )
) else (
  echo [1/3] Entorno virtual encontrado.
)

if not exist "%CONFIG%" (
  echo ERROR: no se encontro la config:
  echo   "%CONFIG%"
  pause
  exit /b 1
)

if not exist "%REQ%" (
  echo ERROR: no se encontro requirements.txt
  pause
  exit /b 1
)

echo [2/3] Instalando / actualizando dependencias ...
"%PYTHON%" -m pip install -r "%REQ%"
if errorlevel 1 (
  echo ERROR: fallo pip install.
  pause
  exit /b 1
)

if not exist "%APP%\models\opponent_predictor.pt" (
  echo.
  echo AVISO: falta models\opponent_predictor.pt — la IA usara modo heuristico.
)

echo.
echo [3/3] Arrancando app local ...
echo URL: %URL%
echo Config: configs\app.yaml
echo.
echo Deja esta ventana abierta. Ctrl+C para detener.
echo.

REM Open browser from a separate Python process (avoids broken nested start quoting).
start "ACG open browser" /min "%PYTHON%" scripts\open_when_ready.py --host 127.0.0.1 --port %PORT% --url %URL%

"%PYTHON%" scripts\run_app.py --config configs\app.yaml --host 127.0.0.1 --port %PORT%
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
  echo ACG local termino con error %ERR%.
) else (
  echo ACG local se detuvo.
)
pause
exit /b %ERR%
