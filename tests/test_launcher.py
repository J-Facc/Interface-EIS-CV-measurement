"""Garde-fous STATIQUES de launch.bat (le script ne peut pas s'exécuter sous Linux).

Ils codent les pièges du batch Windows relevés par AUDIT.md §7 : %VAR% dans un bloc
entre parenthèses (expansé à la lecture, donc périmé), TLS désactivé, jeton, fins de
ligne, caractères non ASCII, erreurs masquées.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = (ROOT / "launch.bat").read_bytes()
TEXT = RAW.decode("ascii")          # ASCII strict : un accent casserait selon la page de code
LINES = TEXT.split("\r\n")


def _code_lines():
    """(numéro, ligne) hors commentaires ``rem``."""
    return [(i, l) for i, l in enumerate(LINES, 1) if not l.lstrip().lower().startswith("rem ")]


def test_crlf_only():
    assert b"\r\n" in RAW and b"\n" not in RAW.replace(b"\r\n", b"")
    assert (ROOT / ".gitattributes").read_text().count("*.bat text eol=crlf") == 1


def test_single_launcher_remains():
    for gone in ("launch_app.bat", "launch_app.sh", "setup_drt_bayesien.bat", "token.txt"):
        assert not (ROOT / gone).exists(), gone


def test_no_token_and_tls_validation_untouched():
    low = TEXT.lower()
    assert "servercertificatevalidationcallback" not in low
    assert "token" not in low and "authorization" not in low and "bearer" not in low
    assert "hklm" not in low and "reg add" not in low        # L-10 : plus d'ecriture registre
    assert "setx" not in low


def test_delayed_expansion_enabled_and_root_captured_before():
    code = [l for _, l in _code_lines()]
    first_setlocal = next(l for l in code if l.lower().startswith("setlocal"))
    assert "disabledelayedexpansion" in first_setlocal.lower()
    i_root = code.index('set "ROOT=%~dp0"')
    i_enable = next(i for i, l in enumerate(code) if "enabledelayedexpansion" in l.lower())
    assert i_root < i_enable


def test_no_percent_variable_inside_parenthesised_blocks():
    """À l'intérieur de tout bloc ( ... ) : uniquement !VAR! ; %%X (variable de for) et
    %~1 (paramètre, constant) restent permis, jamais %VAR%."""
    depth = 0
    offenders = []
    for n, line in _code_lines():
        stripped = line.strip()
        opens = stripped.endswith("(")
        closes = stripped.startswith(")")
        if closes:
            depth -= 1
        if depth > 0 or opens:
            body = re.sub(r"%%[A-Za-z]|%~[a-z]*\d", "", stripped)
            if re.search(r"%[A-Za-z_]", body):
                offenders.append((n, stripped))
        if opens and not closes:
            depth += 1
        elif closes and opens:
            depth += 1
    assert not offenders, offenders
    assert depth == 0, "parentheses de bloc desequilibrees"


def test_no_percent_variable_anywhere_but_the_root_capture():
    """Hors le %~dp0 initial, aucun %VAR% : tout est !VAR! (expansion retardee)."""
    bad = []
    for n, line in _code_lines():
        scrubbed = re.sub(r"%%~?[a-z]*[A-Za-z]|%~[a-z0-9]+", "", line)
        if re.search(r"%[A-Za-z_]", scrubbed):
            bad.append((n, line.strip()))
    assert not bad, bad


def test_blocks_contain_no_unescaped_parenthesis_in_echo():
    """Une ')' dans un echo de bloc le referme prematurement."""
    depth = 0
    bad = []
    for n, line in _code_lines():
        s = line.strip()
        if s.startswith(")"):
            depth -= 1
        if depth > 0 and s.lower().startswith("echo") and re.search(r"[()]", s[4:]):
            bad.append((n, s))
        if s.endswith("("):
            depth += 1
    assert not bad, bad


def test_sha_file_written_without_trailing_digit_redirect_trap():
    """`echo sha>file` perd le dernier caractere d'un SHA finissant par 1 ou 2."""
    for n, line in _code_lines():
        if "REMOTE_SHA!" in line and "echo" in line:
            assert re.match(r'^>"!VERSION_FILE!" echo ', line.strip()) or ">" not in line, (n, line)
    assert '>"!VERSION_FILE!" echo !REMOTE_SHA!' in TEXT


def test_errorlevel_is_checked_after_each_significant_command():
    for needle in ("Invoke-RestMethod", "Invoke-WebRequest", "Expand-Archive", " -m venv ",
                   "pip install", "--ensure", "move \"!TOP_DIR!\""):
        idx = [i for i, l in enumerate(LINES) if needle in l and not l.lstrip().lower().startswith("rem")
               and not l.startswith("start ")]   # sondeur de sante en tache de fond
        assert idx, needle
        for i in idx:
            window = "\n".join(LINES[i + 1:i + 3])
            assert "errorlevel" in window or "!RC" in window, (needle, LINES[i][:60])


def test_no_silent_error_masking_on_significant_commands():
    for n, line in _code_lines():
        if any(k in line for k in ("Expand-Archive", "pip install", "--ensure", "Invoke-WebRequest")):
            assert ">nul 2>&1" not in line, (n, line[:80])


def test_swap_happens_after_extraction_and_validation():
    i_extract = TEXT.index("Expand-Archive")
    i_validate = TEXT.index("archive incomplete")
    i_swap = TEXT.index("call :swap_in")
    assert i_extract < i_validate < i_swap
    # eis_app n'est jamais supprime : seulement renomme en eis_app.old
    assert not re.search(r'rmdir[^\r\n]*"!APP_DIR!"', TEXT)


def test_offline_path_skips_pip_and_cmdstan():
    """Echec de la verification distante -> LAUNCH direct, sans passer par [4/5]."""
    block = TEXT[TEXT.index(":REMOTE_FAILED"):TEXT.index(":DO_UPDATE")]
    assert "goto LAUNCH" in block and "AFTER_UPDATE" not in block and "pip" not in block


def test_browser_opened_only_after_health_check():
    i_start = TEXT.index("streamlit run")
    i_poll = TEXT.index("_stcore/health")
    assert "Start-Process" in TEXT and TEXT.index("Start-Process") > i_poll - 1000
    assert TEXT.count("start \"\" \"http") == 0          # plus d'ouverture aveugle
    assert i_poll < i_start


def test_stan_models_survive_updates():
    assert "stan_cache" in TEXT and "fc /b" in TEXT and ":sync_stan_cache" in TEXT


def test_cmdstan_step_reuses_python_script_and_is_idempotent():
    assert "setup_drt_bayesien.py\" --check" in TEXT.replace("!SETUP_PY!", "setup_drt_bayesien.py")
    assert "--ensure" in TEXT
    assert TEXT.index("--check") < TEXT.index("--ensure")


def test_python_version_floor_matches_pinned_numpy():
    reqs = (ROOT / "requirements.txt").read_text()
    assert "numpy==2.5" in reqs                       # numpy 2.5.x exige Python >= 3.12
    assert "(3, 12)" in TEXT


def test_archive_check_requires_both_stan_models_and_the_version_module():
    """La précompilation couvre Series ET Series_pos (modèle du réglage par défaut) : une archive
    qui en perd un doit être refusée AVANT le remplacement, comme celle qui perd Series.stan."""
    line = next(l for l in LINES if "for %%F in (app.py" in l)
    for needed in ("Series.stan", "Series_pos.stan", "drt\\cmdstan_version.py", "setup_drt_bayesien.py"):
        assert needed in line, needed


def test_drt_check_output_is_visible_not_hidden():
    """La raison de --check — surtout l'avertissement « autre version de CmdStan que la version
    validée » — doit être lue par l'utilisateur, pas envoyée vers nul."""
    line = next(l for l in LINES if "!SETUP_PY!\" --check" in l)
    assert ">nul" not in line and "2>&1" not in line
    assert 'call :log "drt --check' in TEXT                    # et elle laisse une trace au journal


def test_launcher_explains_every_failure_code_of_the_setup_script():
    import setup_drt_bayesien as S

    block = TEXT[TEXT.index("!RC_DRT! equ 7"):TEXT.index('call :log "moteur drt non installe')]
    for code in (S.EXIT_NO_CMDSTANPY, S.EXIT_INSTALL, S.EXIT_BAD_PATH, S.EXIT_COMPILE, S.EXIT_TOOLCHAIN):
        assert f"{code} = " in block, f"code {code} non expliqué par launch.bat"
    assert "toolchain C++" in block


def test_launcher_never_hardcodes_a_cmdstan_version():
    """La version vit dans drt/cmdstan_version.py ; launch.bat délègue à setup_drt_bayesien.py."""
    assert not re.search(r"\b2\.\d{2}\.\d\b", TEXT)


def test_non_ascii_install_path_is_announced_early_and_never_blocks():
    """Un chemin accentue fait echouer make sous Windows : avertir AVANT [1/5], sans jamais bloquer
    (la DRT contourne par un cache ASCII, et l'application, elle, n'a pas ce probleme)."""
    start = TEXT.index('set "EIS_ROOT=!ROOT!"')
    block = TEXT[start:TEXT.index("if not exist \"!WORK_DIR!\" mkdir")]
    assert TEXT.index("Verification des mises a jour") > start       # avant la premiere etape
    assert "-cmatch" in block and "exit 77" in block                   # code dedie, pas un crash
    assert "!errorlevel! equ 77" in block
    assert "caracteres accentues" in block and "sans accents" in block
    assert "goto" not in block and "exit /b" not in block              # simple information
    assert "!" not in block.split("-Command", 1)[1].split("\r\n", 1)[0]   # ni ! ni ^ dans la regle PowerShell
    assert "^" not in block.split("-Command", 1)[1].split("\r\n", 1)[0]
    assert 'call :log "avertissement : chemin non ASCII"' in block

