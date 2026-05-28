@echo off
setlocal
set VENV_DIR=.venv
set APP=app.py
set PORT=8501

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [1/3] Creation de l'environnement virtuel...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERREUR : python introuvable. Installez Python 3.10+ et reessayez.
        pause
        exit /b 1
    )
)

call %VENV_DIR%\Scripts\activate.bat

echo [2/3] Verification des dependances...
pip install -r requirements.txt --quiet

echo [3/3] Lancement de l'application...
start "" "http://localhost:%PORT%"
streamlit run %APP% --server.port %PORT% --browser.gatherUsageStats false

endlocal
pause
