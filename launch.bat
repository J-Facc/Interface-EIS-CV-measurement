@echo off
setlocal

:: ── Configuration ──────────────────────────────────────────────
set GITHUB_USER=J-Facc
set GITHUB_REPO=Interface-EIS-CV-measurement
set BRANCH=main
set APP_DIR=%~dp0eis_app
set PORT=8501
:: ───────────────────────────────────────────────────────────────

echo ============================================
echo  EIS Analyzer — Lancement
echo ============================================

:: 1. Télécharger le ZIP depuis GitHub
echo [1/5] Telechargement depuis GitHub...
set ZIP_URL=https://github.com/%GITHUB_USER%/%GITHUB_REPO%/archive/refs/heads/%BRANCH%.zip
set ZIP_FILE=%TEMP%\eis_update.zip

powershell -Command "Invoke-WebRequest -Uri '%ZIP_URL%' -OutFile '%ZIP_FILE%' -UseBasicParsing" 2>nul
if %errorlevel% neq 0 (
    echo Telechargement echoue - utilisation version locale si disponible.
    goto INSTALL
)
echo Telechargement OK.

:: 2. Extraire le ZIP
echo [2/5] Extraction...
if exist "%APP_DIR%" rmdir /s /q "%APP_DIR%"
if exist "%TEMP%\eis_tmp" rmdir /s /q "%TEMP%\eis_tmp"
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%TEMP%\eis_tmp' -Force"
for /d %%i in (%TEMP%\eis_tmp\*) do (
    move "%%i" "%APP_DIR%" >nul
    goto EXTRACTED
)
:EXTRACTED
del "%ZIP_FILE%" 2>nul
if exist "%TEMP%\eis_tmp" rmdir /s /q "%TEMP%\eis_tmp"
echo Extraction OK.

:: 3. Installer Streamlit et les dépendances
:INSTALL
echo [3/5] Installation de Streamlit et des dependances...
pip install streamlit --quiet --disable-pip-version-check
if exist "%APP_DIR%\requirements.txt" (
    pip install -r "%APP_DIR%\requirements.txt" --quiet --disable-pip-version-check
)
echo Dependances OK.

:: 4. Vérifier que app.py existe
echo [4/5] Verification...
if not exist "%APP_DIR%\app.py" (
    echo.
    echo ERREUR : app.py introuvable dans %APP_DIR%
    echo Verifiez votre connexion internet et relancez.
    pause
    exit /b 1
)
echo app.py trouve.

:: 5. Lancer Streamlit
echo [5/5] Lancement de l'interface...
echo.
echo L'interface va s'ouvrir dans votre navigateur.
echo Pour fermer l'app, fermez cette fenetre.
echo.
start "" "http://localhost:%PORT%"
streamlit run "%APP_DIR%\app.py" --server.port %PORT% --browser.gatherUsageStats false

endlocal
