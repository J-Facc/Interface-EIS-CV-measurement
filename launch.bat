@echo off
setlocal

:: ── Configuration ──────────────────────────────────────────────
set GITHUB_USER=J-Facc
set GITHUB_REPO=Interface-EIS-CV-measurement
set BRANCH=main
set APP_DIR=%~dp0eis_app
set VENV_DIR=%~dp0venv
set VERSION_FILE=%~dp0.version
set PORT=8501
:: ───────────────────────────────────────────────────────────────

echo ============================================
echo  EIS Analyzer - Lancement
echo ============================================

:: Activer les chemins longs Windows
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f >nul 2>&1

:: 1. Verifier si une mise a jour est disponible
echo [1/5] Verification des mises a jour...

set REMOTE_SHA=
for /f "delims=" %%i in ('powershell -Command "try { (Invoke-RestMethod -Uri 'https://api.github.com/repos/%GITHUB_USER%/%GITHUB_REPO%/commits/%BRANCH%' -UseBasicParsing).sha } catch { '' }" 2^>nul') do set REMOTE_SHA=%%i

:: Lire le SHA local
set LOCAL_SHA=
if exist "%VERSION_FILE%" (
    set /p LOCAL_SHA=<"%VERSION_FILE%"
)

:: Comparer
if "%REMOTE_SHA%"=="" (
    echo Hors ligne ou GitHub inaccessible - utilisation version locale.
    goto INSTALL
)

if "%REMOTE_SHA%"=="%LOCAL_SHA%" (
    if exist "%APP_DIR%\app.py" (
        echo Deja a jour - demarrage direct.
        goto INSTALL
    )
)

echo Mise a jour disponible - telechargement...

:: 2. Telecharger le ZIP
echo [2/5] Telechargement...
set ZIP_URL=https://github.com/%GITHUB_USER%/%GITHUB_REPO%/archive/refs/heads/%BRANCH%.zip
set ZIP_FILE=%TEMP%\eis_update.zip

powershell -Command "Invoke-WebRequest -Uri '%ZIP_URL%' -OutFile '%ZIP_FILE%' -UseBasicParsing" 2>nul
if %errorlevel% neq 0 (
    echo Telechargement echoue - utilisation version locale.
    goto INSTALL
)

:: 3. Extraire
echo [3/5] Extraction...
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

:: Sauvegarder le SHA local
echo %REMOTE_SHA%>"%VERSION_FILE%"
echo Mise a jour appliquee.

:: 4. Installer les dependances si necessaire
:INSTALL
echo [4/5] Verification des dependances...
if not exist "%VENV_DIR%\Scripts\streamlit.exe" (
    echo Creation du venv - premiere fois uniquement...
    python -m venv "%VENV_DIR%"
    "%VENV_DIR%\Scripts\pip.exe" install streamlit --quiet --disable-pip-version-check
)
if exist "%APP_DIR%\requirements.txt" (
    "%VENV_DIR%\Scripts\pip.exe" install -r "%APP_DIR%\requirements.txt" --quiet --disable-pip-version-check
)

:: 5. Verifier app.py et lancer
echo [5/5] Lancement...
if not exist "%APP_DIR%\app.py" (
    echo ERREUR : app.py introuvable. Verifiez votre connexion et relancez.
    pause
    exit /b 1
)

echo.
echo Interface disponible sur http://localhost:%PORT%
echo Fermez cette fenetre pour arreter l'application.
echo.
start "" "http://localhost:%PORT%"
"%VENV_DIR%\Scripts\streamlit.exe" run "%APP_DIR%\app.py" --server.port %PORT% --browser.gatherUsageStats false

endlocal
