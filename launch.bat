@echo off
rem ============================================================================
rem  launch.bat - lanceur UNIQUE de EIS Analyzer (Windows)
rem
rem  Depot PUBLIC : aucun jeton. Validation TLS NORMALE (jamais desactivee).
rem  Seul fichier a distribuer : il cree a cote de lui
rem     eis_app\      code de l'application (extrait du ZIP GitHub)
rem     venv\         environnement Python local (chemin court)
rem     stan_cache\   modeles Stan compiles, conserves d'une mise a jour a l'autre
rem     .version      SHA du commit installe
rem     .deps_hash    empreinte des requirements.txt installes
rem     launcher.log  journal du lanceur (reecrit a chaque lancement)
rem
rem  Regles de ce script (pieges du batch Windows, AUDIT.md section 7) :
rem   - setlocal EnableDelayedExpansion ; a l'interieur de TOUT bloc entre parentheses
rem     on n'utilise que !VAR! (jamais %VAR% : il serait expanse a la lecture du bloc,
rem     donc vide ou perime). Pas de parenthese ni de caractere special dans un echo de bloc.
rem   - ROOT est capture AVANT d'activer l'expansion retardee (un "!" dans le chemin
rem     serait sinon consomme).
rem   - Chaque commande significative est suivie d'un test de !errorlevel!, avec un message
rem     qui distingue HORS LIGNE (reseau) de ECHEC REEL.
rem   - Fichier ecrit avec ">fichier echo texte" et non "echo texte>fichier" : un SHA se
rem     terminant par 1 ou 2 serait lu comme "1>" / "2>" (redirection) et perdrait son
rem     dernier caractere.
rem   - Fichier en CRLF (.gitattributes) : avec des LF seuls, goto et call :etiquette
rem     peuvent se tromper de ligne.
rem ============================================================================
setlocal DisableDelayedExpansion
set "ROOT=%~dp0"
setlocal EnableDelayedExpansion

rem -- Configuration ------------------------------------------------------------
set "GITHUB_USER=J-Facc"
set "GITHUB_REPO=Interface-EIS-CV-measurement"
set "BRANCH=main"
set "API_URL=https://api.github.com/repos/!GITHUB_USER!/!GITHUB_REPO!"
set "APP_DIR=!ROOT!eis_app"
set "OLD_DIR=!ROOT!eis_app.old"
set "STAGE_DIR=!ROOT!.eis_update"
set "VENV_DIR=!ROOT!venv"
set "VENV_PY=!VENV_DIR!\Scripts\python.exe"
set "STAN_CACHE=!ROOT!stan_cache"
set "STAN_REL=drt\bayes_drt2\stan_model_files"
set "VERSION_FILE=!ROOT!.version"
set "DEPS_FILE=!ROOT!.deps_hash"
set "LOG_FILE=!ROOT!launcher.log"
set "WORK_DIR=!TEMP!\eis_launcher"
set "RC=0"
rem -----------------------------------------------------------------------------

>"!LOG_FILE!" echo [!date! !time!] demarrage du lanceur
echo ============================================
echo  EIS Analyzer - Lancement
echo ============================================

if not "!APP_DIR:~110,1!"=="" (
    echo.
    echo   AVERTISSEMENT : le chemin de ce dossier est tres long. Windows limite les chemins
    echo   a 260 caracteres et la compilation du moteur DRT peut echouer. Deplacez
    echo   launch.bat dans un dossier court, par exemple C:\EIS, si une erreur survient.
    echo.
    call :log "avertissement : chemin long"
)

rem Chemin non ASCII (accents, etc.) : le shell MSYS lance par mingw32-make recoit le chemin
rem corrompu et la compilation du moteur DRT echoue. Simple information, jamais bloquant : la
rem DRT compile depuis un cache ASCII (drt\stan_compile.py). Detection par PowerShell, qui lit
rem l'environnement en Unicode ; la regle ne contient ni ! ni ^ (expansion de cmd).
set "EIS_ROOT=!ROOT!"
powershell -NoProfile -ExecutionPolicy Bypass -Command "if ($env:EIS_ROOT -cmatch '[\x80-\uFFFF]') { exit 77 } else { exit 0 }" >nul 2>&1
if !errorlevel! equ 77 (
    echo.
    echo   Attention : votre dossier d'installation contient des caracteres accentues, ce qui
    echo   peut poser probleme pour la compilation du moteur DRT. Si l'installation du moteur
    echo   DRT echoue, essayez de deplacer l'application vers un chemin sans accents, par
    echo   exemple C:\EIS_Analyzer.
    echo.
    call :log "avertissement : chemin non ASCII"
)

if not exist "!WORK_DIR!" mkdir "!WORK_DIR!"
if !errorlevel! neq 0 (
    echo   ERREUR : impossible de creer le dossier temporaire !WORK_DIR!
    goto FAIL
)

rem ============================================================================
rem  [1/5] Version distante
rem ============================================================================
echo [1/5] Verification des mises a jour...
set "REMOTE_SHA="
set "LOCAL_SHA="
if exist "!VERSION_FILE!" set /p LOCAL_SHA=<"!VERSION_FILE!"

set "SHA_FILE=!WORK_DIR!\remote_sha.txt"
set "ERR_FILE=!WORK_DIR!\remote_err.txt"
if exist "!SHA_FILE!" del /q "!SHA_FILE!"
if exist "!ERR_FILE!" del /q "!ERR_FILE!"
set "EIS_URL_API=!API_URL!/commits/!BRANCH!"
set "EIS_OUT=!SHA_FILE!"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; $h=@{'User-Agent'='EIS-Launcher';'Accept'='application/vnd.github+json'}; $s=([string](Invoke-RestMethod -Uri $env:EIS_URL_API -Headers $h -TimeoutSec 15 -UseBasicParsing).sha).Trim(); if (($s.Length -ne 40) -or ($s.Trim('0123456789abcdef').Length -ne 0)) { [Console]::Error.WriteLine('Reponse inattendue de l API : ' + $s); exit 12 }; [IO.File]::WriteAllText($env:EIS_OUT, $s); exit 0 } catch { [Console]::Error.WriteLine($_.Exception.Message); if ($_.Exception.Response) { exit 11 } else { exit 10 } }" 2>"!ERR_FILE!"
set "RC=!errorlevel!"
if !RC! neq 0 goto REMOTE_FAILED
if not exist "!SHA_FILE!" (
    set "RC=13"
    goto REMOTE_FAILED
)
set /p REMOTE_SHA=<"!SHA_FILE!"
if "!REMOTE_SHA!"=="" (
    set "RC=13"
    goto REMOTE_FAILED
)
call :log "sha distant !REMOTE_SHA! sha local !LOCAL_SHA!"

if "!REMOTE_SHA!"=="!LOCAL_SHA!" if exist "!APP_DIR!\app.py" (
    echo   Version a jour.
    goto AFTER_UPDATE
)
goto DO_UPDATE

:REMOTE_FAILED
rem La verification a echoue, QUELLE QU'EN SOIT LA CAUSE : on n'essaie ni de mettre a jour
rem ni d'installer quoi que ce soit, on lance la version locale telle quelle.
call :explain_remote !RC!
call :log "verification distante echouee code !RC! - lancement local sans mise a jour"
if exist "!APP_DIR!\app.py" if exist "!VENV_PY!" (
    echo   Lancement de la version locale deja installee, sans mise a jour.
    goto LAUNCH
)
echo   ERREUR : aucune installation locale utilisable. La premiere installation exige
echo   un acces reseau a GitHub : reconnectez-vous puis relancez launch.bat.
goto FAIL

rem ============================================================================
rem  [2/5] Telechargement, extraction, verification, remplacement
rem ============================================================================
:DO_UPDATE
echo   Mise a jour disponible : !REMOTE_SHA:~0,7!
echo [2/5] Telechargement...
set "ZIP_FILE=!WORK_DIR!\eis_update.zip"
if exist "!ZIP_FILE!" del /q "!ZIP_FILE!"
set "EIS_URL_ZIP=!API_URL!/zipball/!REMOTE_SHA!"
set "EIS_OUT=!ZIP_FILE!"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri $env:EIS_URL_ZIP -OutFile $env:EIS_OUT -Headers @{'User-Agent'='EIS-Launcher'} -UseBasicParsing -TimeoutSec 300; exit 0 } catch { [Console]::Error.WriteLine($_.Exception.Message); if ($_.Exception.Response) { exit 11 } else { exit 10 } }" 2>"!ERR_FILE!"
set "RC=!errorlevel!"
if !RC! neq 0 (
    call :explain_remote !RC!
    goto UPDATE_ABORTED
)
if not exist "!ZIP_FILE!" (
    echo   ECHEC REEL : le telechargement n'a produit aucun fichier.
    goto UPDATE_ABORTED
)
call :log "zip telecharge"

echo [3/5] Extraction et verification...
if exist "!STAGE_DIR!" rmdir /s /q "!STAGE_DIR!"
if exist "!STAGE_DIR!" (
    echo   ECHEC REEL : impossible de nettoyer !STAGE_DIR!
    goto UPDATE_ABORTED
)
set "EIS_ZIP=!ZIP_FILE!"
set "EIS_DEST=!STAGE_DIR!"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { Expand-Archive -LiteralPath $env:EIS_ZIP -DestinationPath $env:EIS_DEST -Force; exit 0 } catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }" 2>"!ERR_FILE!"
set "RC=!errorlevel!"
if !RC! neq 0 (
    echo   ECHEC REEL : extraction impossible - archive corrompue ou disque plein.
    type "!ERR_FILE!"
    goto UPDATE_ABORTED
)

set "TOP_DIR="
for /d %%D in ("!STAGE_DIR!\*") do set "TOP_DIR=%%~fD"
if "!TOP_DIR!"=="" (
    echo   ECHEC REEL : l'archive ne contient pas de dossier racine.
    goto UPDATE_ABORTED
)
set "MISSING="
for %%F in (app.py requirements.txt requirements-drt.txt setup_drt_bayesien.py drt\cmdstan_version.py core\app_state.py !STAN_REL!\Series.stan !STAN_REL!\Series_pos.stan) do if not exist "!TOP_DIR!\%%F" set "MISSING=!MISSING! %%F"
if not "!MISSING!"=="" (
    echo   ECHEC REEL : archive incomplete, fichiers absents :!MISSING!
    goto UPDATE_ABORTED
)

call :restore_stan
call :swap_in
set "RC=!errorlevel!"
if !RC! neq 0 (
    echo   ECHEC REEL : remplacement de eis_app impossible, code !RC!. 21 = ancien dossier
    echo   non supprimable, 22 = eis_app verrouille - fermez l'application et les
    echo   explorateurs ouverts dessus, 23 = deplacement refuse. L'ancienne version est conservee.
    goto UPDATE_ABORTED
)
call :keep_state
>"!VERSION_FILE!" echo !REMOTE_SHA!
if !errorlevel! neq 0 echo   AVERTISSEMENT : .version non ecrit, la mise a jour sera retelechargee.
if exist "!OLD_DIR!" rmdir /s /q "!OLD_DIR!"
if exist "!OLD_DIR!" echo   AVERTISSEMENT : ancienne version non supprimee : !OLD_DIR!
if exist "!STAGE_DIR!" rmdir /s /q "!STAGE_DIR!"
del /q "!ZIP_FILE!" 2>nul
echo   Mise a jour appliquee.
call :log "mise a jour appliquee !REMOTE_SHA!"
goto AFTER_UPDATE

:UPDATE_ABORTED
rem Rien n'a ete remplace (ou l'ancien dossier a ete remis en place) : la version locale reste intacte.
call :log "mise a jour abandonnee"
if exist "!STAGE_DIR!" rmdir /s /q "!STAGE_DIR!" 2>nul
if exist "!APP_DIR!\app.py" if exist "!VENV_PY!" (
    echo   Mise a jour abandonnee : la version locale deja installee est conservee et lancee.
    goto LAUNCH
)
echo   ERREUR : pas de version locale utilisable pour prendre le relais.
goto FAIL

rem ============================================================================
rem  [4/5] Python, dependances, moteur DRT
rem ============================================================================
:AFTER_UPDATE
echo [4/5] Environnement Python et dependances...
call :ensure_venv
set "RC=!errorlevel!"
if !RC! neq 0 goto FAIL

call :ensure_deps
set "RC=!errorlevel!"
if !RC! equ 1 goto FAIL
set "DRT_PIP_OK=1"
if !RC! equ 2 set "DRT_PIP_OK=0"

if "!DRT_PIP_OK!"=="1" (
    call :ensure_drt
) else (
    echo   Moteur DRT ignore : l'extra pip n'a pas pu etre installe.
)
goto LAUNCH

rem ============================================================================
rem  [5/5] Demarrage : le navigateur n'est ouvert QU'APRES la reponse du serveur
rem ============================================================================
:LAUNCH
echo [5/5] Demarrage...
if not exist "!APP_DIR!\app.py" (
    echo   ERREUR : app.py introuvable dans !APP_DIR!
    goto FAIL
)
"!VENV_PY!" -c "import streamlit" >nul 2>&1
if !errorlevel! neq 0 (
    echo   ERREUR : streamlit n'est pas installe dans !VENV_DIR!
    echo   Le dossier venv est incomplet. Supprimez-le puis relancez launch.bat avec une
    echo   connexion Internet pour le recreer.
    goto FAIL
)
call :pick_port
if !errorlevel! neq 0 (
    echo   ERREUR : aucun port libre entre 8501 et 8510. Fermez une autre instance.
    goto FAIL
)
set "EIS_URL=http://localhost:!PORT!"
set "EIS_HEALTH=!EIS_URL!/_stcore/health"
echo.
echo   Interface : !EIS_URL!  - le navigateur s'ouvrira des que le serveur repond.
echo   Fermez cette fenetre pour arreter l'application.
echo.
call :log "demarrage de streamlit sur le port !PORT!"
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "for ($i = 0; $i -lt 180; $i++) { try { $r = Invoke-WebRequest -Uri $env:EIS_HEALTH -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { Start-Process $env:EIS_URL; exit 0 } } catch { }; Start-Sleep -Seconds 1 }; Write-Host ('Le serveur n a pas repondu en 3 minutes. Ouvrez ' + $env:EIS_URL + ' a la main.'); exit 1"
cd /d "!APP_DIR!"
"!VENV_PY!" -m streamlit run app.py --server.port !PORT! --server.headless true --browser.gatherUsageStats false
set "RC=!errorlevel!"
call :log "streamlit termine code !RC!"
if !RC! neq 0 (
    echo.
    echo   ERREUR : l'application s'est arretee avec le code !RC!. Voir aussi !LOG_FILE!
    goto FAIL
)
exit /b 0

:FAIL
echo.
echo   Le lancement a echoue. Journal : !LOG_FILE!
pause
exit /b 1

rem ============================================================================
rem  Sous-routines
rem ============================================================================

:log
>>"!LOG_FILE!" echo [!date! !time!] %~1
exit /b 0

:explain_remote
rem %1 = code de sortie du PowerShell : 10 reseau, 11 reponse HTTP d'erreur, 12 reponse
rem inattendue, autre = PowerShell lui-meme.
if "%~1"=="10" (
    echo   HORS LIGNE : GitHub est injoignable - pas de reseau, DNS, proxy ou delai depasse.
) else if "%~1"=="11" (
    echo   ECHEC REEL : GitHub est joignable mais a repondu par une erreur - limite de debit
    echo   de l'API publique, depot renomme ou acces filtre.
) else if "%~1"=="12" (
    echo   ECHEC REEL : reponse inattendue de l'API GitHub.
) else (
    echo   ECHEC REEL : PowerShell a echoue avec le code %~1 - indisponible ou bloque.
)
if exist "!ERR_FILE!" (
    echo   Detail :
    type "!ERR_FILE!"
    type "!ERR_FILE!" >>"!LOG_FILE!"
)
exit /b 0

:swap_in
rem Remplace eis_app par le dossier extrait ET verifie. 21/22/23 : voir le message appelant.
if exist "!OLD_DIR!" rmdir /s /q "!OLD_DIR!"
if exist "!OLD_DIR!" exit /b 21
if not exist "!APP_DIR!" goto swap_move
ren "!APP_DIR!" "eis_app.old"
if errorlevel 1 exit /b 22
:swap_move
move "!TOP_DIR!" "!APP_DIR!" >nul
if errorlevel 1 goto swap_rollback
exit /b 0
:swap_rollback
if exist "!OLD_DIR!" ren "!OLD_DIR!" "eis_app"
exit /b 23

:keep_state
rem L'etat local ecrit par l'application dans son arborescence survit a la mise a jour.
for %%K in (logs sessions) do call :keep_one %%K
exit /b 0
:keep_one
if not exist "!OLD_DIR!\%~1" exit /b 0
move "!OLD_DIR!\%~1" "!APP_DIR!\%~1" >nul
if errorlevel 1 echo   AVERTISSEMENT : le dossier %~1 n'a pas pu etre conserve.
exit /b 0

:restore_stan
rem Les modeles Stan compiles (plusieurs minutes de compilation) sont gardes dans
rem stan_cache, HORS de eis_app. On ne les remet dans le nouvel arbre QUE si le .stan
rem correspondant est identique octet pour octet : un executable perime ne doit jamais
rem servir a un modele modifie.
if not exist "!STAN_CACHE!" exit /b 0
for %%S in ("!TOP_DIR!\!STAN_REL!\*.stan") do call :restore_one "%%~nS"
exit /b 0
:restore_one
set "MODEL=%~1"
if not exist "!STAN_CACHE!\!MODEL!.stan" exit /b 0
fc /b "!STAN_CACHE!\!MODEL!.stan" "!TOP_DIR!\!STAN_REL!\!MODEL!.stan" >nul 2>&1
if errorlevel 1 exit /b 0
for %%X in ("!STAN_CACHE!\!MODEL!.*") do call :restore_file "%%~fX" "%%~nxX"
exit /b 0
:restore_file
if /i "%~x1"==".stan" exit /b 0
copy /y "%~1" "!TOP_DIR!\!STAN_REL!\" >nul
if errorlevel 1 exit /b 0
rem Remet la date de modification a maintenant : le ZIP date les .stan du commit, et
rem cmdstanpy recompile si la source est plus recente que l'executable.
copy /b "!TOP_DIR!\!STAN_REL!\%~2"+,, >nul 2>&1
exit /b 0

:sync_stan_cache
if not exist "!STAN_CACHE!" mkdir "!STAN_CACHE!"
xcopy "!APP_DIR!\!STAN_REL!\*" "!STAN_CACHE!\" /y /d /q >nul
if errorlevel 1 echo   AVERTISSEMENT : stan_cache non mis a jour, les modeles seront recompiles apres la prochaine mise a jour.
exit /b 0

:ensure_venv
if exist "!VENV_PY!" exit /b 0
echo   Creation de l'environnement Python local - premiere fois uniquement...
call :find_python
if !errorlevel! neq 0 (
    echo   ERREUR : Python 3.12 ou plus recent est introuvable. Installez-le depuis
    echo   https://www.python.org/downloads/ en cochant Add python.exe to PATH, puis relancez.
    echo   Le stub du Microsoft Store ne convient pas.
    exit /b 1
)
echo   Python utilise : !PY_CMD!
!PY_CMD! -m venv "!VENV_DIR!"
if !errorlevel! neq 0 (
    echo   ECHEC REEL : la creation du venv a echoue. Cette etape ne demande pas de reseau.
    if exist "!VENV_DIR!" rmdir /s /q "!VENV_DIR!"
    exit /b 1
)
if not exist "!VENV_PY!" (
    echo   ECHEC REEL : le venv a ete cree sans python.exe.
    if exist "!VENV_DIR!" rmdir /s /q "!VENV_DIR!"
    exit /b 1
)
rem Un venv neuf n'a AUCUNE dependance : l'empreinte memorisee de l'ancien est caduque.
if exist "!DEPS_FILE!" del /q "!DEPS_FILE!"
call :log "venv cree"
exit /b 0

:find_python
set "PY_CMD="
for %%C in ("py -3" "python" "python3") do if not defined PY_CMD call :try_python %%C
if not defined PY_CMD exit /b 1
exit /b 0
:try_python
set "CAND=%~1"
!CAND! -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
set "PY_CMD=!CAND!"
exit /b 0

:ensure_deps
rem Retour : 0 tout est installe, 1 echec bloquant (requirements.txt), 2 seul l'extra DRT manque.
set "EIS_REQ1=!APP_DIR!\requirements.txt"
set "EIS_REQ2=!APP_DIR!\requirements-drt.txt"
set "HASH_FILE=!WORK_DIR!\deps_hash_now.txt"
if exist "!HASH_FILE!" del /q "!HASH_FILE!"
set "EIS_OUT=!HASH_FILE!"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { $t = ''; foreach ($f in @($env:EIS_REQ1, $env:EIS_REQ2)) { $t += (Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash }; [IO.File]::WriteAllText($env:EIS_OUT, $t); exit 0 } catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }"
if !errorlevel! neq 0 (
    echo   ECHEC REEL : calcul de l'empreinte des requirements impossible.
    exit /b 1
)
set "DEPS_NOW="
set "DEPS_OLD="
set /p DEPS_NOW=<"!HASH_FILE!"
if exist "!DEPS_FILE!" set /p DEPS_OLD=<"!DEPS_FILE!"
if "!DEPS_NOW!"=="!DEPS_OLD!" (
    echo   Dependances deja a jour.
    exit /b 0
)
echo   Installation des dependances - premiere fois ou apres mise a jour...
"!VENV_PY!" -m pip install --disable-pip-version-check --timeout 30 --retries 2 -r "!EIS_REQ1!"
if !errorlevel! neq 0 (
    call :probe_net pypi.org
    if !errorlevel! neq 0 (
        echo   HORS LIGNE : pip n'a pas pu joindre pypi.org. Reconnectez-vous puis relancez.
    ) else (
        echo   ECHEC REEL : pip a echoue alors que le reseau repond - voir les messages ci-dessus.
    )
    call :log "pip requirements.txt echoue"
    exit /b 1
)
"!VENV_PY!" -m pip install --disable-pip-version-check --timeout 30 --retries 2 -r "!EIS_REQ2!"
if !errorlevel! neq 0 (
    call :probe_net pypi.org
    if !errorlevel! neq 0 (
        echo   HORS LIGNE : l'extra DRT n'a pas pu etre telecharge. L'application demarre sans DRT.
    ) else (
        echo   ECHEC REEL : l'extra DRT - cvxopt, cmdstanpy - n'a pas pu etre installe.
        echo   L'application demarre sans DRT ; voir les messages pip ci-dessus.
    )
    call :log "pip requirements-drt.txt echoue"
    exit /b 2
)
>"!DEPS_FILE!" echo !DEPS_NOW!
if !errorlevel! neq 0 echo   AVERTISSEMENT : .deps_hash non ecrit, pip sera relance au prochain lancement.
call :log "dependances installees"
exit /b 0

:ensure_drt
rem Idempotent : setup_drt_bayesien.py --check ne touche a rien ; --ensure n'installe et ne
rem compile que ce qui manque. On l'appelle depuis le venv, on ne reecrit pas sa logique.
rem --check n'est PAS redirige : sa raison, et surtout l'avertissement quand une AUTRE version
rem de CmdStan que la version validee est installee, doit etre lue. Elle est aussi journalisee.
set "SETUP_PY=!APP_DIR!\setup_drt_bayesien.py"
"!VENV_PY!" "!SETUP_PY!" --check
if !errorlevel! equ 0 (
    echo   Moteur DRT deja pret.
    call :sync_stan_cache
    exit /b 0
)
call :log "drt --check : moteur non pret, preparation du moteur"
echo   Installation du moteur DRT : CmdStan en version validee et modeles Stan. Plusieurs
echo   minutes la premiere fois - telechargement puis compilation C++. Une autre version de
echo   CmdStan deja installee est conservee, jamais supprimee. Ne fermez pas la fenetre.
"!VENV_PY!" -u "!SETUP_PY!" --ensure
set "RC_DRT=!errorlevel!"
if !RC_DRT! equ 0 (
    echo   Moteur DRT pret.
    call :sync_stan_cache
    call :log "moteur drt installe"
    exit /b 0
)
if !RC_DRT! equ 7 (
    echo   HORS LIGNE : CmdStan n'a pas pu etre telecharge. L'application demarre sans DRT ;
    echo   elle sera preparee au prochain lancement avec reseau.
) else (
    echo   ECHEC REEL de l'installation du moteur DRT, code !RC_DRT! - voir les messages
    echo   ci-dessus. L'application demarre sans DRT. 3 = cmdstanpy absent, 4 = installation
    echo   CmdStan, 5 = chemin CmdStan, 6 = erreur de compilation Stan ou chemin avec accents,
    echo   8 = toolchain C++ absente ou non fonctionnelle.
)
call :log "moteur drt non installe code !RC_DRT!"
exit /b 0

:probe_net
rem %1 = hote. Retourne 0 si une connexion TCP 443 aboutit en 4 s, 1 sinon. Sert uniquement
rem a ETIQUETER un echec deja constate.
set "EIS_HOST=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object Net.Sockets.TcpClient; $a = $c.BeginConnect($env:EIS_HOST, 443, $null, $null); if ($a.AsyncWaitHandle.WaitOne(4000) -and $c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
exit /b !errorlevel!

:pick_port
rem Premier port libre de 8501 a 8510. Test de liaison reel : netstat est localise et ne
rem se laisse pas lire de facon fiable sur un Windows non anglophone.
set "PORT="
for /l %%P in (8501,1,8510) do if not defined PORT call :port_free %%P
if not defined PORT exit /b 1
exit /b 0
:port_free
set "EIS_PORT=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $l = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Any, [int]$env:EIS_PORT); $l.Start(); $l.Stop(); exit 0 } catch { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 set "PORT=%~1"
exit /b 0
