@echo off
REM Start the TW3K Portrait Generator web UI.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Expected .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo Starting TW3K Portrait Generator...
echo Opening http://127.0.0.1:7860 in your browser shortly.

REM Give the server a head start, then open the browser.
start "" /b cmd /c "timeout /t 20 /nobreak >nul && start http://127.0.0.1:7860"

".venv\Scripts\python.exe" "scripts\app.py"

echo.
echo Server stopped.
pause
