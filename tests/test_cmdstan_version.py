"""Version de CmdStan : celle de la validation des réglages DRT, et ce qui la garde.

Tous les réglages de ``drt/engine.py`` ont été mesurés sur CmdStan 2.36.0
(``drt/VALIDATION_REGLAGES.md``). Premier test réel sous Windows : ``cmdstan_path()`` valait
``C:\\cmdstan\\cmdstan-2.39.0``. Ces tests :

* gardent la valeur unique de la version et sa cohérence avec la CI et la documentation ;
* vérifient, sur l'installation RÉELLE quand il y en a une, que CmdStan est bien en 2.36.0 — et
  ALERTENT sinon (un avertissement pytest ; un échec en CI, où la version est épinglée) ;
* vérifient que le moteur signale une autre version dans chaque résultat de DRT.
"""

import os
import re
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from drt import engine
from drt.cmdstan_version import (
    PINNED_CMDSTAN_VERSION,
    installed_version,
    version_from_dirname,
    version_from_makefile,
    version_warning,
)

ROOT = Path(__file__).resolve().parent.parent
VALIDATED = "2.36.0"


def test_the_pinned_version_is_the_validated_one():
    """Valeur littérale ici : la changer exige de revalider les réglages, pas seulement le
    code — ce test force à le faire consciemment."""
    assert PINNED_CMDSTAN_VERSION == VALIDATED


def test_ci_documentation_and_launcher_agree_with_the_pinned_version():
    ci = (ROOT / ".github" / "workflows" / "validate.yml").read_text(encoding="utf-8")
    assert re.search(r'CMDSTAN_VERSION:\s*"%s"' % re.escape(PINNED_CMDSTAN_VERSION), ci)
    for doc in ("README.md", "drt/VALIDATION_REGLAGES.md", "docs/ARCHITECTURE.md", "docs/DRT_BAYESIENNE.md"):
        assert PINNED_CMDSTAN_VERSION in (ROOT / doc).read_text(encoding="utf-8"), doc
    # launch.bat ne recopie PAS la version : il délègue à setup_drt_bayesien.py. Une copie ici
    # divergerait en silence de la source unique.
    launcher = (ROOT / "launch.bat").read_text(encoding="ascii")
    assert not re.search(r"2\.\d{2}\.\d", launcher), "launch.bat ne doit contenir aucune version de CmdStan"


def test_every_installation_path_in_setup_pins_the_version():
    """Aucun endroit de setup_drt_bayesien.py ne suppose « la dernière version » : l'appel
    d'installation nomme la version, et il n'y a qu'un appel."""
    source = (ROOT / "setup_drt_bayesien.py").read_text(encoding="utf-8")
    calls = re.findall(r"install_cmdstan\(([^)]*)\)", source)
    calls = [c for c in calls if "version=" in c or "dir=" in c]
    assert len(calls) == 1 and "version=PINNED_CMDSTAN_VERSION" in calls[0]
    assert "latest_version" not in source
    # _latest_cmdstan existe mais ne sert qu'au repli de register()
    assert source.count("_latest_cmdstan(") == 2        # définition + repli de register()


# ── Fonctions pures ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("name, expected", [
    ("cmdstan-2.36.0", "2.36.0"),
    ("/home/u/.cmdstan/cmdstan-2.39.0/", "2.39.0"),
    ("cmdstan-2.37.0-rc1", None),                         # candidat à la publication : pas de triplet nu
    ("cmdstan-git-develop", None),
    ("autre-dossier", None),
])
def test_version_from_dirname(name, expected):
    assert version_from_dirname(name) == expected


def test_version_from_makefile(tmp_path):
    (tmp_path / "makefile").write_text("# en-tête\nCMDSTAN_VERSION := 2.36.0\nautre := 1\n")
    assert version_from_makefile(str(tmp_path)) == "2.36.0"
    (tmp_path / "makefile").write_text("CMDSTAN_VERSION := 2.36\n")       # pas un triplet
    assert version_from_makefile(str(tmp_path)) is None
    assert version_from_makefile(str(tmp_path / "absent")) is None


def test_installed_version_reads_the_makefile_first_then_the_directory_name(tmp_path):
    d = tmp_path / "cmdstan-2.39.0"
    d.mkdir()
    assert installed_version(str(d)) == "2.39.0"          # repli : nom du dossier
    (d / "makefile").write_text("CMDSTAN_VERSION := 2.36.0\n")
    assert installed_version(str(d)) == "2.36.0"          # le makefile fait foi


def test_version_warning_is_silent_only_for_the_validated_version():
    assert version_warning(VALIDATED) is None
    for other in ("2.39.0", "2.36.1", "2.35.0", None):
        msg = version_warning(other)
        assert msg and VALIDATED in msg and "pas été vérifié" in msg
        if other:
            assert other in msg
    # Latin-1 seulement : le message part vers des consoles Windows (cp850/cp1252).
    version_warning("2.39.0").encode("cp1252")


# ── Le moteur signale une autre version ────────────────────────────────────
def _fake_cmdstanpy(monkeypatch, path):
    fake = types.ModuleType("cmdstanpy")
    fake.cmdstan_path = lambda: str(path)
    monkeypatch.setitem(sys.modules, "cmdstanpy", fake)


def test_cmdstan_version_info_warns_on_another_version(monkeypatch, tmp_path):
    d = tmp_path / "cmdstan-2.39.0"
    d.mkdir()
    _fake_cmdstanpy(monkeypatch, d)
    version, warning = engine.cmdstan_version_info()
    assert version == "2.39.0" and warning and "2.39.0" in warning and VALIDATED in warning


def test_cmdstan_version_info_is_silent_on_the_validated_version(monkeypatch, tmp_path):
    d = tmp_path / f"cmdstan-{VALIDATED}"
    d.mkdir()
    _fake_cmdstanpy(monkeypatch, d)
    assert engine.cmdstan_version_info() == (VALIDATED, None)


def test_cmdstan_version_info_without_cmdstan_is_not_a_warning(monkeypatch):
    """CmdStan absent relève de engine_available(), pas d'un avertissement de version."""
    fake = types.ModuleType("cmdstanpy")

    def absent():
        raise ValueError("No CmdStan installation found")

    fake.cmdstan_path = absent
    monkeypatch.setitem(sys.modules, "cmdstanpy", fake)
    assert engine.cmdstan_version_info() == (None, None)
    monkeypatch.setitem(sys.modules, "cmdstanpy", None)
    assert engine.cmdstan_version_info() == (None, None)


class _SeriesInverter:
    """Faux Inverter minimal et déterministe : γ(τ) gaussien, Z reconstruit exactement."""

    def __init__(self, distributions):           # dict neuf exigé (engine._fresh_distributions)
        self.tau = np.logspace(-6, 2, 80)
        self.distributions = {engine.DIST_NAME: {"tau": self.tau}}
        self.stan_model_name = "Series_pos"
        self.stan_mle = types.SimpleNamespace(converged=True)

    def fit(self, f, Z, **kwargs):
        self.Z = Z

    def predict_distribution(self, name, tau=None, percentile=None):
        ln = np.log(tau)
        return 100.0 * np.exp(-0.5 * ((ln - np.log(1e-2)) / 1.0) ** 2) + 30.0 * np.exp(
            -0.5 * ((ln - np.log(1e-4)) / 1.0) ** 2)

    def predict_Rp(self, percentile=None):
        return 130.0

    def predict_Z(self, f):
        return self.Z                                  # même ordre que celui passé à fit()


def _fit_with_fake_inverter(monkeypatch):
    monkeypatch.setattr(engine, "engine_available", lambda: (True, None))
    monkeypatch.setattr(engine, "_import_inverter", lambda: _SeriesInverter)
    f = np.logspace(5, -2, 40)
    Z = 10.0 + 100.0 / (1 + 1j * 2 * np.pi * f * 1e-2)
    spectrum = types.SimpleNamespace(f=f, Zre=Z.real, Zim=-Z.imag)
    return engine.fit_drt(spectrum, mode="optimize")


def test_every_drt_result_carries_the_cmdstan_version_and_warns_when_it_is_not_the_validated_one(
        monkeypatch, tmp_path):
    other = tmp_path / "cmdstan-2.39.0"
    other.mkdir()
    _fake_cmdstanpy(monkeypatch, other)
    fr = _fit_with_fake_inverter(monkeypatch)
    assert fr.drt_diagnostics["cmdstan_version"] == "2.39.0"
    assert fr.drt_diagnostics["cmdstan_validated_version"] == VALIDATED
    assert any("2.39.0" in w and VALIDATED in w for w in fr.warnings)       # remonte à l'UI
    assert any("2.39.0" in n for n in fr.drt_diagnostics["notes"])
    assert fr.converged is not None and fr.drt_diagnostics["quality_ok"] in (True, False)


def test_a_result_on_the_validated_version_carries_no_version_warning(monkeypatch, tmp_path):
    ok = tmp_path / f"cmdstan-{VALIDATED}"
    ok.mkdir()
    _fake_cmdstanpy(monkeypatch, ok)
    fr = _fit_with_fake_inverter(monkeypatch)
    assert fr.drt_diagnostics["cmdstan_version"] == VALIDATED
    assert not any("validés" in w for w in fr.warnings)


def test_the_version_warning_does_not_make_the_result_unconverged_nor_unqualified(monkeypatch, tmp_path):
    """Un avertissement de provenance n'est ni une alerte de convergence ni de qualité : il ne
    doit pas changer `converged` ni `quality_ok` (réglages inchangés, seule l'information l'est)."""
    other = tmp_path / "cmdstan-2.39.0"
    other.mkdir()
    _fake_cmdstanpy(monkeypatch, other)
    with_warning = _fit_with_fake_inverter(monkeypatch)
    ok = tmp_path / f"cmdstan-{VALIDATED}"
    ok.mkdir()
    _fake_cmdstanpy(monkeypatch, ok)
    without = _fit_with_fake_inverter(monkeypatch)
    assert with_warning.converged == without.converged
    assert with_warning.drt_diagnostics["quality_ok"] == without.drt_diagnostics["quality_ok"]
    assert with_warning.drt_diagnostics["alerts"] == without.drt_diagnostics["alerts"]


# ── Installation RÉELLE ─────────────────────────────────────────────────────
_HAVE_CMDSTAN = engine.engine_available()[0]


@pytest.mark.skipif(not _HAVE_CMDSTAN, reason="exige cvxopt + cmdstanpy + CmdStan (job CI « drt »)")
def test_the_installed_cmdstan_is_the_validated_version():
    """Alerte si le CmdStan réellement utilisé n'est pas la version validée. Sur un poste de
    développement : un avertissement visible dans le résumé pytest. En CI (``CI`` défini), où la
    version est épinglée par le workflow : un échec — une divergence y est un vrai défaut."""
    import cmdstanpy

    version = installed_version(cmdstanpy.cmdstan_path())
    assert version is not None, "version de CmdStan indéterminable"
    if version != VALIDATED:
        message = (f"CmdStan {version} installé ({cmdstanpy.cmdstan_path()}), mais les réglages "
                   f"DRT n'ont été validés que sur {VALIDATED} (drt/VALIDATION_REGLAGES.md).")
        if os.environ.get("CI"):
            pytest.fail(message)
        warnings.warn(message, UserWarning, stacklevel=1)
    assert engine.cmdstan_version_info() == (version, version_warning(version))
