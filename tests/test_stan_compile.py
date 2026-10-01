"""Tests de drt/stan_compile.py — compilation Stan robuste aux chemins non-ASCII.

Le défaut (Windows) : mingw32-make passe le chemin au shell MSYS, qui le reçoit corrompu dès
qu'il contient un « è ». Ces tests rejouent la LOGIQUE qui l'évite sur n'importe quel système :
un répertoire au nom accentué est créé, un faux ``cmdstanpy`` remplace CmdStan (aucune
toolchain, aucun réseau) et enregistre le chemin EXACT qu'il aurait donné à make. La propriété
vérifiée est celle qui compte : **aucun chemin non-ASCII n'atteint jamais la « compilation »**.
Ce que ces tests ne peuvent pas prouver : le comportement réel de mingw32-make sous Windows.
"""

import hashlib
import os
import re
import sys
import types
from pathlib import Path

import pytest

import setup_drt_bayesien as S
from drt import stan_compile as sc

ROOT = Path(__file__).resolve().parent.parent
ACCENTED = "Thèse é ü"                     # espaces ET accents : le pire cas du nom de dossier
STAN = "// modele Stan factice\n"


class FakeCmdstanpy(types.ModuleType):
    """Faux cmdstanpy : ``CmdStanModel`` note le chemin qu'il « compile » et pose un exécutable."""

    def __init__(self, cmdstan: Path):
        super().__init__("cmdstanpy")
        self._cmdstan = cmdstan
        self.compiled = []                 # chemins reçus par la « compilation » (make)
        self.constructed = []              # (stan_file, exe_file, force_compile) de TOUS les appels
        self.fail = None                   # exception à lever à la compilation
        outer = self

        class CmdStanModel:
            def __init__(self, stan_file=None, exe_file=None, force_compile=False):
                outer.constructed.append((stan_file, exe_file, force_compile))
                self.stan_file = stan_file
                exe = Path(stan_file).with_suffix(sc.EXE_SUFFIX)
                if exe_file is None:
                    if force_compile or not exe.exists():
                        outer.compiled.append(stan_file)
                        if outer.fail is not None:
                            raise outer.fail
                        exe.write_text("exe")
                    exe_file = str(exe)
                self.exe_file = str(exe_file)

        self.CmdStanModel = CmdStanModel

    def cmdstan_path(self):
        return str(self._cmdstan)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """CmdStan 2.36.0 factice ASCII, un ``.stan`` sous un dossier ACCENTUÉ, un cache à côté de CmdStan."""
    if not str(tmp_path).isascii():
        pytest.skip("tmp_path non-ASCII : ces tests construisent eux-mêmes le chemin accentué")
    cmdstan = tmp_path / "cmdstan" / "cmdstan-2.36.0"
    cmdstan.mkdir(parents=True)
    accented = tmp_path / ACCENTED / "stan_model_files"
    try:
        accented.mkdir(parents=True)
    except (OSError, UnicodeError):
        pytest.skip("le système de fichiers refuse un nom accentué")
    stan = accented / "Series.stan"
    stan.write_text(STAN)
    fake = FakeCmdstanpy(cmdstan)
    monkeypatch.setitem(sys.modules, "cmdstanpy", fake)
    monkeypatch.delenv(sc.CACHE_ENV, raising=False)
    # Les replis ne doivent jamais écrire hors de tmp_path.
    monkeypatch.setattr(sc, "_default_cmdstan_parent", lambda: str(tmp_path / "default_parent"))
    monkeypatch.setattr(sc.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    return types.SimpleNamespace(fake=fake, stan=stan, accented=accented, tmp=tmp_path, cmdstan=cmdstan,
                                 cache=tmp_path / "cmdstan" / "model_cache" / "Series")


# ── détection ────────────────────────────────────────────────────────────────
def test_detection_of_non_ascii_paths():
    assert sc.has_non_ascii(r"C:\Users\jeanf\Desktop\.Thèse\x.stan")
    assert sc.has_non_ascii("/home/é/x.stan")
    assert not sc.has_non_ascii("/home/jean/these/x.stan")


def test_detection_uses_the_real_path_cmdstanpy_hands_to_make(world):
    """cmdstanpy passe ``os.path.realpath`` : un lien ASCII vers un dossier accentué est piégeux."""
    link = world.tmp / "lien_ascii"
    try:
        os.symlink(world.accented, link)
    except (OSError, NotImplementedError):
        pytest.skip("liens symboliques indisponibles")
    assert str(link).isascii()
    assert sc.has_non_ascii(link / "Series.stan")


# ── chemin ASCII : comportement inchangé ─────────────────────────────────────
def test_ascii_path_is_passed_straight_to_cmdstanpy_without_cache(world):
    ascii_stan = world.tmp / "plain" / "Series.stan"
    ascii_stan.parent.mkdir()
    ascii_stan.write_text(STAN)
    sc.compile_stan_model(ascii_stan)
    assert world.fake.constructed == [(str(ascii_stan), None, False)]
    sc.compile_stan_model(ascii_stan, force=True)
    assert world.fake.constructed[-1] == (str(ascii_stan), None, True)
    assert not (world.tmp / "cmdstan" / "model_cache").exists()


# ── chemin non-ASCII : copie, compilation ASCII, cache par hash ──────────────
def test_non_ascii_stan_is_compiled_from_an_ascii_copy_never_from_its_own_location(world):
    model = sc.compile_stan_model(world.stan)

    assert world.fake.compiled == [str(world.cache / "Series.stan")]
    assert all(str(p).isascii() for p in world.fake.compiled), "make a reçu un chemin non-ASCII"
    assert (world.cache / "Series.stan").read_text() == STAN       # copie fidèle
    assert model.exe_file == str(world.cache / ("Series" + sc.EXE_SUFFIX))
    assert str(model.exe_file).isascii()
    # Le dossier d'origine n'est ni compilé ni pollué.
    assert sorted(p.name for p in world.accented.iterdir()) == ["Series.stan"]


def test_unchanged_stan_is_not_recompiled_and_the_cached_exe_is_used(world):
    sc.compile_stan_model(world.stan)
    world.fake.constructed.clear()
    world.fake.compiled.clear()

    again = sc.compile_stan_model(world.stan)

    assert world.fake.compiled == []
    # exe_file= donné explicitement : cmdstanpy n'a même pas l'occasion de comparer des dates.
    assert world.fake.constructed == [(str(world.cache / "Series.stan"),
                                       str(world.cache / ("Series" + sc.EXE_SUFFIX)), False)]
    assert again.exe_file == str(world.cache / ("Series" + sc.EXE_SUFFIX))


def test_cache_validity_is_the_content_hash_not_the_name_or_date(world):
    sc.compile_stan_model(world.stan)
    world.fake.compiled.clear()
    stan_before = (world.cache / "Series.stan").stat().st_mtime

    world.stan.write_text(STAN + "// modifié\n")            # même nom, autre contenu
    sc.compile_stan_model(world.stan)

    assert world.fake.compiled == [str(world.cache / "Series.stan")]
    assert (world.cache / "Series.stan").read_text() == STAN + "// modifié\n"
    stamp = sc._read_stamp(world.cache, "Series")
    assert stamp["stan_sha256"] == hashlib.sha256((STAN + "// modifié\n").encode()).hexdigest()
    assert (world.cache / "Series.stan").stat().st_mtime >= stan_before


def test_identical_content_with_a_new_timestamp_is_not_recompiled(world):
    """Un ZIP de mise à jour redate tous les fichiers : le contenu seul décide."""
    sc.compile_stan_model(world.stan)
    world.fake.compiled.clear()
    os.utime(world.stan, (4102444800, 4102444800))          # an 2100
    sc.compile_stan_model(world.stan)
    assert world.fake.compiled == []


def test_another_cmdstan_version_recompiles(world):
    sc.compile_stan_model(world.stan)
    world.fake.compiled.clear()
    other = world.tmp / "cmdstan" / "cmdstan-2.39.0"
    other.mkdir()
    world.fake._cmdstan = other
    sc.compile_stan_model(world.stan)
    assert len(world.fake.compiled) == 1
    assert sc._read_stamp(world.cache, "Series")["cmdstan_version"] == "2.39.0"


def test_force_and_a_missing_executable_both_recompile(world):
    sc.compile_stan_model(world.stan)
    world.fake.compiled.clear()
    sc.compile_stan_model(world.stan, force=True)
    assert len(world.fake.compiled) == 1
    (world.cache / ("Series" + sc.EXE_SUFFIX)).unlink()
    sc.compile_stan_model(world.stan)
    assert len(world.fake.compiled) == 2


def test_a_failed_compilation_leaves_no_valid_cache(world):
    sc.compile_stan_model(world.stan)
    world.stan.write_text(STAN + "// casse\n")
    world.fake.fail = ValueError("Failed to compile Stan model: erreur de syntaxe Stan")

    with pytest.raises(ValueError, match="erreur de syntaxe Stan"):    # la vraie erreur, inchangée
        sc.compile_stan_model(world.stan)

    assert sc.cached_exe(str(world.stan), "2.36.0") is None            # ni ancien exe, ni tampon
    world.fake.fail = None
    world.fake.compiled.clear()
    sc.compile_stan_model(world.stan)
    assert len(world.fake.compiled) == 1                              # le prochain appel recompile


def test_cached_exe_reports_only_a_matching_entry(world):
    assert sc.cached_exe(str(world.stan), "2.36.0") is None
    sc.compile_stan_model(world.stan)
    assert sc.cached_exe(str(world.stan), "2.36.0") == str(world.cache / ("Series" + sc.EXE_SUFFIX))
    assert sc.cached_exe(str(world.stan), "2.39.0") is None
    assert sc.cached_exe(str(world.accented / "absent.stan"), "2.36.0") is None


# ── impossibilité de copier : message explicite, jamais l'erreur brute ───────
def test_unwritable_cache_gives_the_explicit_message(world, monkeypatch):
    def refuse(*_a, **_k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(sc.shutil, "copyfile", refuse)
    with pytest.raises(sc.NonAsciiPathError) as info:
        sc.compile_stan_model(world.stan)
    text = str(info.value)
    assert "le chemin d'installation contient des caractères accentués ou spéciaux" in text
    assert "ce qui empêche la compilation du moteur DRT sur certains systèmes Windows" in text
    assert "déplacez le dossier vers un chemin sans accents (ex: C:\\EIS_Analyzer) et relancez" in text
    assert "mingw32-make" not in text and "No such file" not in text
    assert world.fake.compiled == []                      # rien n'a été tenté


def test_a_cache_root_blocked_by_a_file_is_skipped_for_the_next_candidate(world):
    blocker = world.tmp / "cmdstan" / "model_cache"
    blocker.write_text("un fichier, pas un dossier")
    sc.compile_stan_model(world.stan)
    fallback = world.tmp / "default_parent" / "model_cache" / "Series" / "Series.stan"
    assert world.fake.compiled == [str(fallback)]


def test_a_non_ascii_cmdstan_parent_falls_back_to_another_ascii_root(world):
    accented_cmdstan = world.tmp / ACCENTED / "cmdstan" / "cmdstan-2.36.0"
    accented_cmdstan.mkdir(parents=True)
    world.fake._cmdstan = accented_cmdstan
    roots = sc._candidate_roots(str(accented_cmdstan))
    assert any(r.isascii() for r in roots)
    # CmdStan lui-même sous un chemin accentué ne se contourne pas : message dédié, avant de copier.
    with pytest.raises(sc.NonAsciiPathError, match="CmdStan est installé sous un chemin"):
        sc.compile_stan_model(world.stan)
    assert world.fake.compiled == []


def test_no_usable_ascii_root_at_all_gives_the_explicit_message(world, monkeypatch):
    monkeypatch.setattr(sc, "_candidate_roots", lambda _d: [str(world.accented / "cache")])
    with pytest.raises(sc.NonAsciiPathError) as info:
        sc.compile_stan_model(world.stan)
    assert "déplacez le dossier vers un chemin sans accents" in str(info.value)
    assert world.fake.compiled == []


def test_a_failure_with_an_accented_cmdstan_dir_is_explained_even_for_an_ascii_stan(world):
    """Le .stan est ASCII mais make tourne dans un CmdStan accentué : on nomme la cause."""
    accented_cmdstan = world.tmp / ACCENTED / "cmdstan-2.36.0"
    accented_cmdstan.mkdir(parents=True)
    world.fake._cmdstan = accented_cmdstan
    world.fake.fail = ValueError("make: sh: No such file or directory")
    ascii_stan = world.tmp / "plain.stan"
    ascii_stan.write_text(STAN)
    with pytest.raises(sc.NonAsciiPathError, match="CmdStan est installé sous un chemin"):
        sc.compile_stan_model(ascii_stan)


def test_the_cache_root_can_be_forced_by_environment(world, monkeypatch):
    forced = world.tmp / "forced"
    monkeypatch.setenv(sc.CACHE_ENV, str(forced))
    sc.compile_stan_model(world.stan)
    assert world.fake.compiled == [str(forced / "Series" / "Series.stan")]


# ── setup_drt_bayesien : la logique de prêt/compilation passe par le cache ───
@pytest.fixture
def setup_world(world, monkeypatch):
    monkeypatch.setattr(S, "_stan_dir", lambda: str(world.accented))
    monkeypatch.setitem(sys.modules, "drt.bayes_drt2.inversion",
                        types.SimpleNamespace(Inverter=lambda: None))
    world.fake.set_cmdstan_path = lambda p: None
    for name in S.STAN_TARGETS:
        (world.accented / name).write_text(STAN + name)
    return world


def test_models_under_an_accented_dir_are_to_compile_until_the_ascii_cache_is_built(setup_world):
    w = setup_world
    assert S._models_to_compile("2.36.0") == S.STAN_TARGETS
    assert S.precompile() == S.STAN_TARGETS
    assert S._models_to_compile("2.36.0") == []
    assert S._models_to_compile("2.39.0") == S.STAN_TARGETS           # autre version : à refaire
    # Aucun fichier compilé, ni marqueur, dans le dossier accentué.
    assert sorted(p.name for p in w.accented.iterdir()) == sorted(S.STAN_TARGETS)
    assert all(str(p).isascii() for p in w.fake.compiled) and len(w.fake.compiled) == 2


def test_precompile_does_not_rebuild_what_the_cache_already_holds(setup_world):
    w = setup_world
    S.precompile()
    w.fake.compiled.clear()
    S.precompile()
    assert w.fake.compiled == []
    S.precompile(force=True)
    assert len(w.fake.compiled) == 2


def test_precompile_surfaces_the_explicit_message_not_a_wrapped_compile_error(setup_world, monkeypatch):
    def refuse(*_a, **_k):
        raise OSError("disque plein")

    monkeypatch.setattr(sc.shutil, "copyfile", refuse)
    with pytest.raises(sc.NonAsciiPathError):
        S.precompile()


def test_ensure_returns_the_explicit_message_when_the_copy_is_impossible(setup_world, monkeypatch):
    w = setup_world
    parent = w.tmp / "cmdstan"
    monkeypatch.setattr(S, "toolchain_status", lambda: (True, [], ["make : faux"]))
    monkeypatch.setattr(sc.shutil, "copyfile", lambda *_a, **_k: (_ for _ in ()).throw(OSError("plein")))
    w.fake.install_cmdstan = lambda **_k: True
    w.fake.set_cmdstan_path = lambda _p: setattr(w.fake, "_cmdstan", w.cmdstan)

    code, message = S._ensure(parent=str(parent), allow_install=True)

    assert code == S.EXIT_COMPILE
    assert "déplacez le dossier vers un chemin sans accents (ex: C:\\EIS_Analyzer) et relancez" in message
    assert "Erreur de compilation Stan/C++" not in message


def test_ensure_refuses_an_accented_cmdstan_parent_before_any_download(world, monkeypatch):
    installs = []
    world.fake.install_cmdstan = lambda **k: installs.append(k) or True
    world.fake.set_cmdstan_path = lambda _p: None
    code, message = S._ensure(parent=str(world.tmp / ACCENTED / "cmdstan"), allow_install=True)
    assert code == S.EXIT_BAD_PATH and installs == []
    assert "caractères accentués ou spéciaux" in message and "C:\\cmdstan" in message


def test_the_app_path_never_refuses_on_an_accented_parent(world, monkeypatch):
    """allow_install=False n'installe ni ne compile : rien à refuser, l'état est seulement constaté."""
    world.fake.set_cmdstan_path = lambda _p: None
    code, message = S._ensure(parent=str(world.tmp / ACCENTED / "cmdstan"), allow_install=False)
    assert code != S.EXIT_BAD_PATH or "caractères accentués" not in message


def test_main_warns_early_when_the_install_folder_is_accented(world, monkeypatch, capsys):
    monkeypatch.setattr(S, "REPO_DIR", str(world.accented))
    monkeypatch.setattr(S, "_ensure", lambda **_k: (S.EXIT_OK, "ok"))
    assert S.main(["--ensure"]) == S.EXIT_OK
    out = capsys.readouterr().out
    assert "Attention : votre dossier d'installation contient des caractères accentués" in out
    assert out.index("Attention") < out.index("SUCCES")


def test_main_stays_silent_when_the_install_folder_is_ascii(world, monkeypatch, capsys):
    monkeypatch.setattr(S, "REPO_DIR", str(world.tmp / "ascii"))
    monkeypatch.setattr(S, "_ensure", lambda **_k: (S.EXIT_OK, "ok"))
    S.main(["--ensure"])
    assert "Attention" not in capsys.readouterr().out


# ── garde-fous de couverture : aucun autre point d'appel à make ──────────────
_SOURCES = [p for p in ROOT.rglob("*.py")
            if not {"tests", "venv", ".venv", "node_modules"} & set(p.relative_to(ROOT).parts)]


def test_no_source_calls_cmdstanmodel_directly_except_the_safe_wrapper():
    """Tout appel direct à cmdstanpy.CmdStanModel(…) contournerait la protection ASCII."""
    allowed = {ROOT / "drt" / "stan_compile.py", ROOT / "drt" / "bayes_drt2" / "inversion.py"}
    offenders = [str(p.relative_to(ROOT)) for p in _SOURCES
                 if p not in allowed and re.search(r"\bCmdStanModel\(", p.read_text(encoding="utf-8"))]
    assert offenders == []
    inversion = (ROOT / "drt" / "bayes_drt2" / "inversion.py").read_text(encoding="utf-8")
    assert "from cmdstanpy import CmdStanModel" not in inversion
    assert "from ..stan_compile import compile_stan_model as CmdStanModel" in inversion


def test_install_cmdstan_is_only_called_from_the_guarded_setup_script():
    callers = [str(p.relative_to(ROOT)) for p in _SOURCES
               if re.search(r"\binstall_cmdstan\(", p.read_text(encoding="utf-8"))]
    assert callers == ["setup_drt_bayesien.py"]


def test_vendored_inverter_really_compiles_through_the_safe_wrapper(world, monkeypatch):
    pytest.importorskip("cvxopt")
    from drt.bayes_drt2 import inversion

    assert inversion.CmdStanModel is sc.compile_stan_model
    monkeypatch.setattr(inversion, "script_dir", str(world.accented.parent))
    inv = inversion.Inverter()
    model, name = inv._get_stan_model(False, False, False, None, False, False)
    assert name == "Series"
    assert world.fake.compiled and all(str(p).isascii() for p in world.fake.compiled)
    assert model.exe_file.isascii()
