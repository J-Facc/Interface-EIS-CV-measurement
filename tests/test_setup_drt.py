"""Tests de setup_drt_bayesien.py — la logique d'installation DRT appelée par launch.bat.

Aucun vrai CmdStan ni réseau : un faux module ``cmdstanpy`` est injecté (la CI de base
n'installe pas l'extra DRT). Ce qui est vérifié, c'est le CONTRAT dont dépend le lanceur :
codes de sortie, idempotence (rien n'est réinstallé ni recompilé quand tout est prêt),
distinction « hors ligne » / « échec réel », et l'absence d'installation côté application.
"""

import sys
import types
from pathlib import Path

import pytest

import setup_drt_bayesien as S


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
    """Dossiers temporaires + faux cmdstanpy + modèles Stan factices ; compteurs d'appels."""
    parent = tmp_path / "cmdstan"
    stan_dir = tmp_path / "stan_model_files"
    stan_dir.mkdir()
    fake = FakeCmdstanpy(parent=parent)
    monkeypatch.setitem(sys.modules, "cmdstanpy", fake)
    monkeypatch.setattr(S, "_stan_dir", lambda: str(stan_dir))
    monkeypatch.setattr(S, "_network_down", lambda timeout=5.0: False)
    compiled = []

    def fake_precompile():
        for name in S.STAN_TARGETS:
            (stan_dir / (name[:-len(".stan")] + (".exe" if S.os.name == "nt" else ""))).write_text("exe")
        compiled.append(True)
        return list(S.STAN_TARGETS)

    monkeypatch.setattr(S, "precompile", fake_precompile)
    return types.SimpleNamespace(parent=parent, stan_dir=stan_dir, fake=fake, compiled=compiled)


def _install_cmdstan_dir(env, version="2.36.0"):
    (env.parent / f"cmdstan-{version}").mkdir(parents=True)


def _compile_models(env):
    env.compiled.clear()
    S.precompile()
    env.compiled.clear()


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
    assert S._ensure(str(env.parent)) == (S.EXIT_OK, "DRT prete : CmdStan enregistre et modeles Stan compiles.")
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


def test_compile_failure_has_its_own_code(env, monkeypatch):
    _install_cmdstan_dir(env)

    def boom():
        raise RuntimeError("g++ introuvable")

    monkeypatch.setattr(S, "precompile", boom)
    code, message = S._ensure(str(env.parent))
    assert code == S.EXIT_COMPILE and "g++ introuvable" in message


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
