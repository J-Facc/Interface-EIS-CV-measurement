@echo off
setlocal enabledelayedexpansion

:: ============================================================================
::  setup_drt_bayesien.bat
::  Installe l'extra "DRT bayesien" : cvxopt + cmdstanpy, puis cmdstan AVEC sa
::  toolchain C++ (mingw-w64), et precompile le modele Stan vendore pour que la
::  premiere analyse ne paie pas la compilation.
::
::  IMPORTANT : script SEPARE de launch.bat. Il n'est JAMAIS appele au demarrage
::  de l'application. A lancer MANUELLEMENT, une seule fois, apres un premier
::  lancement de launch.bat / launch_app.bat (qui cree le venv du projet).
:: ============================================================================

set SCRIPT_DIR=%~dp0
:: Dossier COURT pour cmdstan -> evite les chemins trop longs (Python MS Store,
:: limite MAX_PATH). cmdstanpy retrouve ce chemin via la variable CMDSTAN.
if "%CMDSTAN_INSTALL_DIR%"=="" set CMDSTAN_INSTALL_DIR=%SystemDrive%\cmdstan
set PATH_OUT=%TEMP%\drt_cmdstan_path.txt

echo ============================================
echo  DRT bayesien - Installation de l'extra
echo ============================================
echo.

:: Activer les chemins longs Windows (utile pour cmdstan / mingw).
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f >nul 2>&1

:: ── 1. Localiser et activer le venv du projet ──────────────────────────
echo [1/5] Activation du venv du projet...
set VENV_DIR=
if exist "%SCRIPT_DIR%venv\Scripts\activate.bat" set VENV_DIR=%SCRIPT_DIR%venv
if "!VENV_DIR!"=="" if exist "%SCRIPT_DIR%.venv\Scripts\activate.bat" set VENV_DIR=%SCRIPT_DIR%.venv
if "!VENV_DIR!"=="" if exist "%SCRIPT_DIR%..\venv\Scripts\activate.bat" set VENV_DIR=%SCRIPT_DIR%..\venv

if "!VENV_DIR!"=="" (
    echo ERREUR : venv du projet introuvable.
    echo         Emplacements testes : venv\ , .venv\ , ..\venv\
    echo         Lancez d'abord launch.bat ou launch_app.bat pour creer le venv.
    pause
    exit /b 1
)
echo         venv : !VENV_DIR!
call "!VENV_DIR!\Scripts\activate.bat"
if !errorlevel! neq 0 (
    echo ERREUR : activation du venv echouee.
    pause
    exit /b 1
)

:: ── 2. Installer l'extra DRT (cvxopt + cmdstanpy) ─────────────────────
echo.
echo [2/5] Installation de l'extra pip : cvxopt cmdstanpy...
python -m pip install cvxopt cmdstanpy --disable-pip-version-check
if !errorlevel! neq 0 (
    echo ERREUR : echec de "pip install cvxopt cmdstanpy".
    echo         Verifiez votre connexion / proxy et reessayez.
    call :SSL_HINT
    pause
    exit /b 1
)

:: ── 3. Installer cmdstan (+ toolchain C++), enregistrer et precompiler ─
echo.
echo [3/5] Installation de cmdstan + toolchain C++ (mingw-w64) et precompilation...
echo        Dossier cmdstan : !CMDSTAN_INSTALL_DIR!
echo        (le premier passage peut durer plusieurs minutes)
if exist "%PATH_OUT%" del "%PATH_OUT%" >nul 2>&1
python "%SCRIPT_DIR%setup_drt_bayesien.py" --cmdstan-dir "!CMDSTAN_INSTALL_DIR!" --path-out "%PATH_OUT%"
set PY_RC=!errorlevel!
if !PY_RC! neq 0 (
    echo.
    echo ECHEC : installation / precompilation cmdstan (code !PY_RC!).
    call :SSL_HINT
    pause
    exit /b !PY_RC!
)

:: ── 4. Rendre le chemin cmdstan persistant (variable CMDSTAN) ──────────
set CMDSTAN_PATH=
if exist "%PATH_OUT%" set /p CMDSTAN_PATH=<"%PATH_OUT%"
if not "!CMDSTAN_PATH!"=="" (
    set CMDSTAN=!CMDSTAN_PATH!
    setx CMDSTAN "!CMDSTAN_PATH!" >nul 2>&1
    echo        CMDSTAN enregistre (persistant) : !CMDSTAN_PATH!
) else (
    echo        [avert] chemin cmdstan non capture : CMDSTAN non enregistre.
)

:: ── 5. Verification finale explicite ──────────────────────────────────
echo.
echo [4/5] Verification : cmdstanpy.cmdstan_path()...
python -c "import cmdstanpy; print(cmdstanpy.cmdstan_path())"
if !errorlevel! neq 0 (
    echo ECHEC : cmdstan_path() n'a pas renvoye de chemin valide.
    pause
    exit /b 1
)

echo.
echo [5/5] Termine.
echo ============================================
echo  SUCCES : DRT bayesien pret a l'emploi.
echo   - cvxopt + cmdstanpy installes
echo   - cmdstan + toolchain C++ installes
echo   - modele Stan vendore precompile (cache)
echo ============================================
echo.
pause
endlocal
exit /b 0

:: ── Sous-routine : aide proxy / SSL ───────────────────────────────────
:SSL_HINT
echo.
echo --------------------------------------------------------------------
echo  En cas d'echec SSL derriere un proxy d'entreprise, definir avant de
echo  relancer ce script :
echo    set HTTPS_PROXY=http://utilisateur:motdepasse@proxy:port
echo    set HTTP_PROXY=http://utilisateur:motdepasse@proxy:port
echo  Interception TLS (certificat interne) -^> pointer vers le CA bundle :
echo    set REQUESTS_CA_BUNDLE=C:\chemin\vers\ca-bundle.pem
echo    set SSL_CERT_FILE=C:\chemin\vers\ca-bundle.pem
echo --------------------------------------------------------------------
echo.
goto :eof
