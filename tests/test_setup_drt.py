"""Tests de setup_drt_bayesien.py — la logique d'installation DRT appelée par launch.bat.

Aucun vrai CmdStan ni réseau : un faux module ``cmdstanpy`` est injecté (la CI de base
n'installe pas l'extra DRT). Ce qui est vérifié, c'est le CONTRAT dont dépend le lanceur :
codes de sortie, idempotence (rien n'est réinstallé ni recompilé quand tout est prêt),
distinction « hors ligne » / « échec réel », absence d'installation côté application — et
les trois défauts mesurés lors du premier test réel sous Windows :

* une version de CmdStan plus récente (2.39.0) que la version validée (2.36.0), déjà présente
  dans le dossier d'installation, était enregistrée à la place de la version épinglée ;
* la toolchain C++ n'était jamais vérifiée avant de compiler (``mingw32-make`` introuvable) ;
* les exécutables Stan compilés sous une autre version de CmdStan étaient réutilisés.
"""

import inspect
import sys
import types
from pathlib import Path

import pytest

import setup_drt_bayesien as S
from drt.cmdstan_version import PINNED_CMDSTAN_VERSION

OTHER = "2.39.0"          # la version relevée dans le log du test Windows


class FakeCmdstanpy(types.ModuleType):
    """Faux cmdstanpy : enregistre les appels, ne touche à rien."""

    def __init__(self, install_ok: bool = True, parent: Path = None):
        super().__init__("cmdstanpy")
        self.install_ok = install_ok
        self.parent = parent
        self.install_calls = []
        self.registered = None

    def install_cmdstan(self, **kwargs):
        self.install_calls.append(kwargs)
        if self.install_ok and self.parent is not None:
            (self.parent / f"cmdstan-{kwargs['version']}").mkdir(parents=True, exist_ok=True)
        return self.install_ok

    def set_cmdstan_path(self, path):
        self.registered = path

    def cmdstan_path(self):
        return self.registered


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Dossiers temporaires + faux cmdstanpy + modèles Stan factices ; compteurs d'appels.

    La toolchain est déclarée fonctionnelle par défaut (les tests qui l'étudient la
    remplacent) : aucun test ne dépend de ce qui est installé sur la machine.
    """
    parent = tmp_path / "cmdstan"
    stan_dir = tmp_path / "stan_model_files"
    stan_dir.mkdir()
    fake = FakeCmdstanpy(parent=parent)
    monkeypatch.setitem(sys.modules, "cmdstanpy", fake)
    monkeypatch.setattr(S, "_stan_dir", lambda: str(stan_dir))
    monkeypatch.setattr(S, "_network_down", lambda timeout=5.0: False)
    monkeypatch.setattr(S, "toolchain_status", lambda: (True, [], ["make : GNU Make (faux)"]))
    compiled = []

    def fake_precompile(force=False):
        version = S.installed_version(fake.registered) or "inconnue"
        for name in S._models_to_compile(version, force=force):
            exe = S._exe_path(str(stan_dir), name)
            Path(exe).write_text("exe")
            Path(S._marker_path(str(stan_dir), name)).write_text(version + "\n")
        compiled.append(True)
        return list(S.STAN_TARGETS)

    monkeypatch.setattr(S, "precompile", fake_precompile)
    return types.SimpleNamespace(parent=parent, stan_dir=stan_dir, fake=fake, compiled=compiled)


def _install_cmdstan_dir(env, version=PINNED_CMDSTAN_VERSION):
    (env.parent / f"cmdstan-{version}").mkdir(parents=True)


def _compile_models(env, version=PINNED_CMDSTAN_VERSION):
    """Modèles compilés « sous » ``version`` (exécutable + marqueur), sans les compter."""
    for name in S.STAN_TARGETS:
        Path(S._exe_path(str(env.stan_dir), name)).write_text("exe")
        Path(S._marker_path(str(env.stan_dir), name)).write_text(version + "\n")


def test_pinned_version_is_a_real_version_triplet():
    assert S.PINNED_CMDSTAN_VERSION.count(".") == 2


def test_check_without_cmdstanpy_is_not_ready(monkeypatch):
    monkeypatch.setitem(sys.modules, "cmdstanpy", None)       # `import cmdstanpy` -> ImportError
    ready, why = S.check_drt_ready("nowhere")
    assert not ready and "cmdstanpy" in why
    assert S.main(["--check", "--cmdstan-dir", "nowhere"]) == S.EXIT_NOT_READY


def test_check_never_installs_nor_compiles(env):
    ready, why = S.check_drt_ready(str(env.parent))
    assert not ready and "CmdStan" in why
    assert env.fake.install_calls == [] and env.compiled == []


def test_check_requires_compiled_models(env):
    _install_cmdstan_dir(env)
    ready, why = S.check_drt_ready(str(env.parent))
    assert not ready and "Stan" in why
    _compile_models(env)
    ready, _ = S.check_drt_ready(str(env.parent))
    assert ready
    assert S.main(["--check", "--cmdstan-dir", str(env.parent)]) == S.EXIT_OK


def test_ensure_is_idempotent_when_everything_is_ready(env):
    """Cas du lancement ordinaire : ni installation ni compilation."""
    _install_cmdstan_dir(env)
    _compile_models(env)
    assert S._ensure(str(env.parent)) == (
        S.EXIT_OK, f"DRT prete : CmdStan {PINNED_CMDSTAN_VERSION} enregistre et modeles Stan compiles.")
    assert env.fake.install_calls == [] and env.compiled == []


def test_ensure_installs_pinned_version_then_compiles_once(env):
    code, _ = S._ensure(str(env.parent))
    assert code == S.EXIT_OK
    assert len(env.fake.install_calls) == 1
    call = env.fake.install_calls[0]
    assert call["version"] == S.PINNED_CMDSTAN_VERSION and call["compiler"] is True
    assert call["overwrite"] is False and call["dir"] == str(env.parent)
    assert env.compiled == [True]
    # deuxième passage : plus rien à faire
    assert S._ensure(str(env.parent))[0] == S.EXIT_OK
    assert len(env.fake.install_calls) == 1 and env.compiled == [True]


def test_ensure_recompiles_only_the_missing_models(env):
    _install_cmdstan_dir(env)                               # CmdStan présent, modèles absents
    assert S._ensure(str(env.parent))[0] == S.EXIT_OK
    assert env.fake.install_calls == [] and env.compiled == [True]


def test_force_recompiles_even_when_ready(env):
    _install_cmdstan_dir(env)
    _compile_models(env)
    assert S._ensure(str(env.parent), force=True)[0] == S.EXIT_OK
    assert env.compiled == [True]


def test_failed_install_is_reported_not_swallowed(env):
    """install_cmdstan renvoie False au lieu de lever : l'échec ne doit pas devenir un
    message sans rapport sur un dossier introuvable."""
    env.fake.install_ok = False
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_INSTALL and "install_cmdstan a echoue" in message
    assert env.compiled == []


def test_offline_failure_is_distinguished_from_a_real_failure(env, monkeypatch):
    env.fake.install_ok = False
    monkeypatch.setattr(S, "_network_down", lambda timeout=5.0: True)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_OFFLINE and "Hors ligne" in message
    assert S.EXIT_OFFLINE != S.EXIT_INSTALL


def test_without_install_permission_the_app_path_only_observes(env):
    """app.py : allow_install=False ne télécharge ni ne compile jamais."""
    ready, message = S.ensure_drt_ready(str(env.parent), allow_install=False)
    assert not ready and "launch.bat" in message
    _install_cmdstan_dir(env)
    ready, message = S.ensure_drt_ready(str(env.parent), allow_install=False)
    assert not ready and "launch.bat" in message             # modèles non compilés
    assert env.fake.install_calls == [] and env.compiled == []
    _compile_models(env)
    assert S.ensure_drt_ready(str(env.parent), allow_install=False)[0] is True


def test_ensure_never_raises_without_cmdstanpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cmdstanpy", None)
    ready, message = S.ensure_drt_ready("nowhere")
    assert ready is False and "cmdstanpy" in message
    assert S.main(["--cmdstan-dir", "nowhere"]) == S.EXIT_NO_CMDSTANPY


def test_network_probe_is_never_conclusive_behind_a_proxy(monkeypatch):
    """Derrière un proxy, une connexion directe échoue alors que pip passe : on n'étiquette
    donc jamais « hors ligne » quand une variable de proxy est déclarée."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")

    def refuse(*a, **k):
        raise OSError("direct connection refused")

    monkeypatch.setattr(S.socket, "create_connection", refuse)
    assert S._network_down() is False
    monkeypatch.delenv("HTTPS_PROXY")
    for v in ("https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(v, raising=False)
    assert S._network_down() is True


# ═════════════════════════════════════════════════════════════════════════════
# Version épinglée — le défaut du test Windows : 2.39.0 enregistrée au lieu de 2.36.0
# ═════════════════════════════════════════════════════════════════════════════
def test_register_prefers_the_pinned_version_over_a_newer_one(env):
    """Reproduit le log (cmdstan_path() = C:\\cmdstan\\cmdstan-2.39.0) : avec les deux
    dossiers présents, « le plus récent » donnait 2.39.0 ; c'est la version épinglée qui doit
    être enregistrée."""
    _install_cmdstan_dir(env, OTHER)
    _install_cmdstan_dir(env, PINNED_CMDSTAN_VERSION)
    assert S._latest_cmdstan(str(env.parent)).endswith(f"cmdstan-{OTHER}")   # l'ancien choix
    path = S.register(str(env.parent))
    assert path.endswith(f"cmdstan-{PINNED_CMDSTAN_VERSION}")
    assert env.fake.registered == path


def test_register_falls_back_to_another_version_only_when_the_pinned_one_is_absent(env):
    _install_cmdstan_dir(env, OTHER)
    assert S.register(str(env.parent)).endswith(f"cmdstan-{OTHER}")


def test_check_reports_another_installed_version_with_an_explicit_warning(env, capsys):
    _install_cmdstan_dir(env, OTHER)
    _compile_models(env, OTHER)                       # tout est compilé : seule la version diffère
    ready, why = S.check_drt_ready(str(env.parent))
    assert not ready
    assert OTHER in why and PINNED_CMDSTAN_VERSION in why and "ATTENTION" in why
    assert "pas été vérifié" in why                   # « n'a pas été vérifié sur cette version »
    assert S.main(["--check", "--cmdstan-dir", str(env.parent)]) == S.EXIT_NOT_READY
    assert "ATTENTION" in capsys.readouterr().out     # signalé dans les logs, pas seulement renvoyé
    assert env.fake.install_calls == [] and env.compiled == []                # --check n'installe rien


def test_ensure_installs_the_pinned_version_next_to_the_other_and_never_removes_it(env):
    _install_cmdstan_dir(env, OTHER)
    _compile_models(env, OTHER)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_OK and "ATTENTION" not in message
    assert [c["version"] for c in env.fake.install_calls] == [PINNED_CMDSTAN_VERSION]
    assert (env.parent / f"cmdstan-{OTHER}").is_dir()                         # jamais désinstallée
    assert env.fake.registered.endswith(f"cmdstan-{PINNED_CMDSTAN_VERSION}")
    assert env.compiled == [True]                     # modèles bâtis sous 2.39.0 : recompilés
    assert S.check_drt_ready(str(env.parent))[0] is True


def test_ensure_falls_back_to_the_other_version_with_a_warning_when_offline(env, monkeypatch):
    """Version épinglée non installable (hors ligne) : la DRT reste utilisable sur l'autre
    version, mais l'avertissement est dit — au log ET dans le message renvoyé à l'UI."""
    env.fake.install_ok = False
    monkeypatch.setattr(S, "_network_down", lambda timeout=5.0: True)
    _install_cmdstan_dir(env, OTHER)
    _compile_models(env, OTHER)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_OK
    assert "ATTENTION" in message and OTHER in message and PINNED_CMDSTAN_VERSION in message
    assert len(env.fake.install_calls) == 1           # la version validée a bien été tentée
    assert env.compiled == []                         # rien à recompiler : marqueurs = 2.39.0


def test_app_path_uses_another_version_but_says_so(env):
    _install_cmdstan_dir(env, OTHER)
    _compile_models(env, OTHER)
    ready, message = S.ensure_drt_ready(str(env.parent), allow_install=False)
    assert ready is True and "ATTENTION" in message and OTHER in message
    assert env.fake.install_calls == []


def test_makefile_version_wins_over_a_renamed_directory(tmp_path):
    from drt.cmdstan_version import installed_version, version_warning

    d = tmp_path / "cmdstan-9.9.9"
    d.mkdir()
    (d / "makefile").write_text("CMDSTAN_VERSION := 2.36.0\n")
    assert installed_version(str(d)) == "2.36.0"
    assert version_warning(installed_version(str(d))) is None
    assert version_warning(None) is not None          # indéterminable ≠ confirmé


# ═════════════════════════════════════════════════════════════════════════════
# Exécutables Stan : la version de CmdStan qui les a compilés compte
# ═════════════════════════════════════════════════════════════════════════════
def test_models_compiled_under_another_version_are_recompiled(env):
    _install_cmdstan_dir(env)
    _compile_models(env, OTHER)
    assert S._models_to_compile(PINNED_CMDSTAN_VERSION) == S.STAN_TARGETS
    ready, why = S.check_drt_ready(str(env.parent))
    assert not ready and "non compiles" in why and "Series.stan" in why and "Series_pos.stan" in why
    assert S._ensure(str(env.parent))[0] == S.EXIT_OK
    assert env.compiled == [True]
    assert S._models_to_compile(PINNED_CMDSTAN_VERSION) == []


def test_an_executable_of_unknown_origin_is_never_presumed_good(env):
    """Exécutable sans marqueur (installeur antérieur : celui du test Windows) : recompilé."""
    _install_cmdstan_dir(env)
    _compile_models(env)
    Path(S._marker_path(str(env.stan_dir), "Series_pos.stan")).unlink()
    assert S._models_to_compile(PINNED_CMDSTAN_VERSION) == ["Series_pos.stan"]


def test_only_the_stale_model_is_listed(env):
    _install_cmdstan_dir(env)
    _compile_models(env)
    Path(S._exe_path(str(env.stan_dir), "Series_pos.stan")).unlink()
    assert S._models_to_compile(PINNED_CMDSTAN_VERSION) == ["Series_pos.stan"]


class _FakeInverterModule(types.ModuleType):
    def __init__(self):
        super().__init__("drt.bayes_drt2.inversion")
        self.Inverter = lambda: None


@pytest.fixture
def real_precompile(env, monkeypatch):
    """La VRAIE ``precompile`` contre un faux ``CmdStanModel`` : on observe ses appels."""
    monkeypatch.undo()                                # retire le faux precompile de `env`…
    monkeypatch.setitem(sys.modules, "cmdstanpy", env.fake)
    monkeypatch.setattr(S, "_stan_dir", lambda: str(env.stan_dir))
    monkeypatch.setitem(sys.modules, "drt.bayes_drt2.inversion", _FakeInverterModule())
    calls = []

    class FakeModel:
        def __init__(self, stan_file=None, **kwargs):
            calls.append((Path(stan_file).name, kwargs))
            Path(S._exe_path(str(env.stan_dir), Path(stan_file).name)).write_text("exe")
            self.exe_file = stan_file

    env.fake.CmdStanModel = FakeModel
    for name in S.STAN_TARGETS:
        (env.stan_dir / name).write_text("// stan")
    _install_cmdstan_dir(env)
    S.register(str(env.parent))
    env.calls = calls
    return env


def test_precompile_compiles_both_models_and_stamps_the_cmdstan_version(real_precompile):
    env = real_precompile
    assert S.precompile() == S.STAN_TARGETS
    assert [c[0] for c in env.calls] == S.STAN_TARGETS
    assert all(c[1].get("force_compile") is True for c in env.calls)          # exécutable périmé ignoré
    for name in S.STAN_TARGETS:
        assert S._compiled_with(str(env.stan_dir), name) == PINNED_CMDSTAN_VERSION


def test_precompile_skips_up_to_date_models_and_force_really_forces(real_precompile):
    env = real_precompile
    S.precompile()
    env.calls.clear()
    S.precompile()                                    # tout à jour : aucun appel
    assert env.calls == []
    S.precompile(force=True)                          # --force ne doit pas être un no-op
    assert [c[0] for c in env.calls] == S.STAN_TARGETS


def test_a_missing_stan_source_is_an_error_not_a_skip(real_precompile):
    env = real_precompile
    (env.stan_dir / "Series_pos.stan").unlink()
    with pytest.raises(RuntimeError, match="Series_pos.stan"):
        S.precompile()


def test_a_failed_compilation_leaves_no_version_stamp(real_precompile):
    env = real_precompile

    def boom(stan_file=None, **kwargs):
        raise ValueError("Failed to compile Stan model")

    env.fake.CmdStanModel = boom
    with pytest.raises(S.StanCompileError) as info:
        S.precompile()
    assert info.value.model == "Series.stan"
    assert S._compiled_with(str(env.stan_dir), "Series.stan") is None


# ═════════════════════════════════════════════════════════════════════════════
# Toolchain C++ — le défaut du test Windows : « mingw32-make [WinError 2] »
# ═════════════════════════════════════════════════════════════════════════════
MISSING_MAKE = ["mingw32-make : introuvable dans le PATH"]


def test_a_missing_toolchain_is_reported_before_any_compilation(env, monkeypatch):
    """Avant : mingw32-make absent → erreur brute de compilation. Maintenant : la cause, dite
    comme telle, et AUCUNE compilation n'est tentée."""
    _install_cmdstan_dir(env)
    monkeypatch.setattr(S, "toolchain_status", lambda: (False, MISSING_MAKE, []))
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_TOOLCHAIN
    assert "mingw32-make" in message and "Toolchain C++ absente" in message
    assert "PAS une erreur dans les modeles" in message
    assert env.compiled == []


def test_toolchain_is_installed_then_verified_on_the_path_before_compiling(env, monkeypatch):
    """Windows simulé : toolchain absente → install_cmdstan(compiler=True) → re-vérification
    explicite que les binaires sont maintenant utilisables → seulement alors, compilation."""
    monkeypatch.setattr(S, "_is_windows", lambda: True)
    state = {"ok": False}
    monkeypatch.setattr(S, "toolchain_status", lambda: (
        (True, [], ["mingw32-make : GNU Make 4.4"]) if state["ok"] else (False, MISSING_MAKE, [])))
    real_install = env.fake.install_cmdstan

    def install_and_put_on_path(**kwargs):
        state["ok"] = True                            # cmdstanpy a ajouté RTools au PATH
        return real_install(**kwargs)

    env.fake.install_cmdstan = install_and_put_on_path
    _install_cmdstan_dir(env)                         # CmdStan là, seule la toolchain manque
    code, _ = S._ensure(str(env.parent))
    assert code == S.EXIT_OK
    assert len(env.fake.install_calls) == 1 and env.fake.install_calls[0]["compiler"] is True
    assert env.compiled == [True]


def test_toolchain_installed_but_not_on_the_path_is_a_failure_not_a_success(env, monkeypatch):
    monkeypatch.setattr(S, "_is_windows", lambda: True)
    monkeypatch.setattr(S, "toolchain_status", lambda: (False, MISSING_MAKE, []))
    _install_cmdstan_dir(env)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_TOOLCHAIN
    assert len(env.fake.install_calls) == 1           # l'installation a bien été tentée
    assert "ne sont toujours pas utilisables" in message
    assert env.compiled == []


def test_toolchain_install_failure_is_labelled_offline_or_real(env, monkeypatch):
    monkeypatch.setattr(S, "_is_windows", lambda: True)
    monkeypatch.setattr(S, "toolchain_status", lambda: (False, MISSING_MAKE, []))
    env.fake.install_ok = False
    _install_cmdstan_dir(env)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_TOOLCHAIN and "echec de l'installation" in message
    monkeypatch.setattr(S, "_network_down", lambda timeout=5.0: True)
    _, message = S._ensure(str(env.parent))
    assert "reseau injoignable" in message


def test_no_toolchain_install_is_attempted_off_windows(env, monkeypatch):
    """cmdstanpy n'installe un compilateur que sous Windows : relancer install_cmdstan ne
    réparerait rien et coûterait plusieurs minutes."""
    monkeypatch.setattr(S, "_is_windows", lambda: False)
    monkeypatch.setattr(S, "toolchain_status", lambda: (False, ["make : introuvable dans le PATH"], []))
    _install_cmdstan_dir(env)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_TOOLCHAIN
    assert "build-essential" in message or "xcode-select" in message
    assert env.fake.install_calls == []


def test_the_toolchain_is_not_probed_when_nothing_needs_compiling(env, monkeypatch):
    """Tout est compilé pour la bonne version : le lancement ordinaire ne sonde rien."""
    _install_cmdstan_dir(env)
    _compile_models(env)

    def forbidden():
        raise AssertionError("toolchain_status ne doit pas être appelée")

    monkeypatch.setattr(S, "toolchain_status", forbidden)
    assert S._ensure(str(env.parent))[0] == S.EXIT_OK


def test_compile_error_with_a_working_toolchain_is_labelled_as_such(env, monkeypatch):
    _install_cmdstan_dir(env)

    def boom(force=False):
        raise S.StanCompileError("Series_pos.stan", ValueError("Failed to compile\nerror: syntax error line 12"))

    monkeypatch.setattr(S, "precompile", boom)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_COMPILE
    assert "Erreur de compilation Stan/C++ de Series_pos.stan" in message
    assert "toolchain, elle, fonctionne" in message and "syntax error line 12" in message


def test_compile_failure_caused_by_a_vanished_toolchain_is_a_toolchain_error(env, monkeypatch):
    """[WinError 2] pendant la compilation : le diagnostic doit nommer la toolchain."""
    _install_cmdstan_dir(env)
    states = iter([(True, [], ["mingw32-make : ok"]), (False, MISSING_MAKE, [])])
    monkeypatch.setattr(S, "toolchain_status", lambda: next(states))

    def boom(force=False):
        raise S.StanCompileError("Series.stan", FileNotFoundError("[WinError 2] mingw32-make"))

    monkeypatch.setattr(S, "precompile", boom)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_TOOLCHAIN and "toolchain C++ ne fonctionne plus" in message


def test_a_failure_before_the_cpp_compilation_is_not_blamed_on_the_compiler(env, monkeypatch):
    _install_cmdstan_dir(env)

    def boom(force=False):
        raise ImportError("No module named 'cvxopt'")

    monkeypatch.setattr(S, "precompile", boom)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_COMPILE and "avant la compilation C++" in message and "cvxopt" in message


def test_toolchain_probe_distinguishes_missing_from_working(monkeypatch):
    ok, detail = S._probe_tool("this-tool-does-not-exist-xyz")
    assert not ok and "introuvable dans le PATH" in detail
    ok, detail = S._probe_tool(sys.executable)        # répond à --version, code 0
    assert ok and "Python" in detail
    monkeypatch.setattr(S, "_toolchain_needs", lambda: [("this-tool-does-not-exist-xyz",), (sys.executable,)])
    ok, problems, found = S.toolchain_status()
    assert not ok and len(problems) == 1 and "this-tool-does-not-exist-xyz" in problems[0]
    assert len(found) == 1


def test_activate_toolchain_puts_an_existing_rtools_on_the_path_without_installing(env, monkeypatch):
    """cmdstanpy n'ajoute RTools au PATH que dans le process qui INSTALLE : tout autre process
    (relance, application) doit l'y remettre lui-même, sans rien télécharger."""
    monkeypatch.setattr(S, "_is_windows", lambda: True)
    monkeypatch.setattr(S, "toolchain_status", lambda: (False, MISSING_MAKE, []))
    seen = []
    utils = types.ModuleType("cmdstanpy.utils")
    utils.cxx_toolchain_path = lambda version=None, install_dir=None: seen.append((version, install_dir))
    monkeypatch.setitem(sys.modules, "cmdstanpy.utils", utils)
    assert S.activate_toolchain(str(env.parent)) is True
    assert seen == [(None, str(env.parent))] and env.fake.install_calls == []

    def no_rtools(version=None, install_dir=None):
        raise ValueError("no RTools toolchain installation found")

    utils.cxx_toolchain_path = no_rtools
    assert S.activate_toolchain(str(env.parent)) is False        # absence : pas une erreur ici


def test_activate_toolchain_is_a_no_op_off_windows_or_when_tools_already_work(env, monkeypatch):
    monkeypatch.setattr(S, "_is_windows", lambda: False)
    assert S.activate_toolchain(str(env.parent)) is False
    monkeypatch.setattr(S, "_is_windows", lambda: True)          # toolchain_status : fonctionnelle
    assert S.activate_toolchain(str(env.parent)) is False


def test_log_survives_a_console_that_cannot_encode_a_character(monkeypatch, capsys):
    class Narrow:
        encoding = "ascii"
        written = []

        def write(self, text):
            text.encode("ascii")                      # lève UnicodeEncodeError comme cp850 sur « ≠ »
            self.written.append(text)

        def flush(self):
            pass

    narrow = Narrow()
    monkeypatch.setattr(sys, "stdout", narrow)
    S.log("version \u2260 attendue")                  # ne doit pas lever
    assert any("version" in w for w in narrow.written)


# ═════════════════════════════════════════════════════════════════════════════
# Couverture : tout modèle Stan que le moteur peut sélectionner est précompilé
# ═════════════════════════════════════════════════════════════════════════════
def _selected_models(monkeypatch):
    """Modèle choisi par le VRAI ``Inverter`` pour nonneg=True / False (CmdStanModel factice)."""
    pytest.importorskip("cvxopt")                     # inversion.py l'importe au niveau module
    from drt.bayes_drt2 import inversion

    class NoCompile:
        def __init__(self, stan_file=None, **kwargs):
            self.stan_file = stan_file

    monkeypatch.setattr(inversion, "CmdStanModel", NoCompile)
    inv = inversion.Inverter()
    return {nonneg: inv._get_stan_model(nonneg, False, False, None, False, False)[1] + ".stan"
            for nonneg in (True, False)}, inversion


def test_precompile_covers_every_model_the_engine_can_select(monkeypatch):
    """Le test Windows ne précompilait que Series.stan alors que le réglage retenu
    (nonneg=True, drt/engine.py) utilise Series_pos.stan. Ici : la sélection du vrai Inverter,
    pour les deux valeurs de nonneg, doit être EXACTEMENT ``STAN_TARGETS``."""
    selected, inversion = _selected_models(monkeypatch)
    assert set(selected.values()) == set(S.STAN_TARGETS)
    stan_dir = Path(inversion.__file__).parent / "stan_model_files"
    for name in S.STAN_TARGETS:
        assert (stan_dir / name).is_file(), name


def test_the_default_engine_setting_selects_a_precompiled_model(monkeypatch):
    from drt import engine

    selected, _ = _selected_models(monkeypatch)
    assert selected[engine.DRTSettings().nonneg] in S.STAN_TARGETS
    assert engine.DEFAULT_NONNEG is True and selected[True] == "Series_pos.stan"


def test_engine_does_not_pass_options_that_would_select_another_stan_model():
    """La sélection ci-dessus suppose outliers=False, fitY=False, SA=False et pas de drift :
    ce sont les défauts d'``Inverter.fit`` que le moteur ne surcharge pas. Si le moteur se mettait à
    passer l'un d'eux, d'autres modèles seraient utilisés SANS être précompilés : ce test
    échoue alors, avant qu'un utilisateur ne paie la compilation à son premier spectre."""
    pytest.importorskip("cvxopt")
    from drt import engine
    from drt.bayes_drt2.inversion import Inverter

    defaults = inspect.signature(Inverter.fit).parameters
    for option in ("outliers", "fitY", "SA"):
        if option in defaults:
            assert defaults[option].default is False, option
    source = inspect.getsource(engine._run_inversion)
    for option in ("outliers", "fitY", "SA", "drift"):
        assert option not in source, f"drt.engine passe maintenant « {option} » : étendre STAN_TARGETS"
