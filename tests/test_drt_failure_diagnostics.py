"""Diagnostic des échecs de calcul DRT — sans CmdStan, avec un faux ``Inverter``.

Premier test réel sous Windows : « DRT de « e1_c3_r1.csv » non calculée : domain error ». Un
message sans étape, sans origine, sans traceback : impossible de dire d'où venait l'erreur
sans reproduire le cas. Ces tests exigent maintenant que TOUT échec de calcul porte :

* l'étape, la classe ET le message de la cause, et la ligne d'où elle a été levée ;
* le traceback complet de la cause (``DRTComputationError.detail``), écrit au journal ;
* les avertissements émis AVANT l'échec (numpy « invalid value »…, souvent la vraie cause) ;
* le contexte du spectre (points, plage de fréquences, |Z| min/max, points à Z = 0).

Et que le contrat de ``fit_drt`` reste « seules ValueError/RuntimeError sortent » : le pipeline
n'y perd qu'UN spectre.
"""

import logging
import types
import warnings

import numpy as np
import pytest

from core import pipeline
from core.models import GroupAnalysis
from drt import engine
from tests.synthetic_data import make_spectrum


def _cvxopt_like_solver():
    """Frame d'origine de l'échec : le message doit la NOMMER."""
    raise ValueError("domain error")


class FailingInverter:
    stan_model_name = "Series_pos"
    distributions = {}

    def fit(self, f, Z, **kwargs):
        warnings.warn("invalid value encountered in divide", RuntimeWarning)
        _cvxopt_like_solver()


class NoDistributionInverter:
    """L'inversion « réussit » mais n'expose pas la distribution attendue."""
    stan_model_name = "Series_pos"
    distributions = {}

    def fit(self, f, Z, **kwargs):
        pass


@pytest.fixture
def spectrum():
    f = np.logspace(5, -2, 40)
    Z = 10.0 + 100.0 / (1 + 1j * 2 * np.pi * f * 1e-2)
    return types.SimpleNamespace(f=f, Zre=Z.real, Zim=-Z.imag)


def _use(monkeypatch, inverter_cls):
    monkeypatch.setattr(engine, "engine_available", lambda: (True, None))
    monkeypatch.setattr(engine, "_import_inverter", lambda: inverter_cls)


def test_a_calculation_failure_names_the_stage_the_cause_and_where_it_was_raised(monkeypatch, spectrum):
    _use(monkeypatch, FailingInverter)
    with pytest.raises(engine.DRTComputationError) as info:
        engine.fit_drt(spectrum, mode="optimize")
    message = str(info.value)
    assert "inversion bayes_drt2" in message and "mode optimize" in message
    assert "ValueError: domain error" in message                # classe ET message, pas « domain error » seul
    assert "_cvxopt_like_solver()" in message and "test_drt_failure_diagnostics.py:" in message
    assert "40 points" in message
    assert "logs/eis_analyzer.log" in message                    # dit où est le traceback complet
    assert isinstance(info.value.__cause__, ValueError)         # cause d'origine chaînée


def test_the_detail_carries_the_full_traceback_context_and_prior_warnings(monkeypatch, spectrum):
    _use(monkeypatch, FailingInverter)
    with pytest.raises(engine.DRTComputationError) as info:
        engine.fit_drt(spectrum, mode="optimize", nonneg=True)
    detail = info.value.detail
    assert "Traceback (most recent call last)" in detail
    assert "_cvxopt_like_solver" in detail and 'raise ValueError("domain error")' in detail
    assert "RuntimeWarning: invalid value encountered in divide" in detail   # émis AVANT l'échec
    assert "40 points" in detail and "0 point(s) à Z = 0" in detail
    assert "'nonneg': True" in detail and "'mode': 'optimize'" in detail
    assert "Series_pos" in detail and "CmdStan :" in detail


def test_zero_impedance_points_are_counted_in_the_detail(monkeypatch, spectrum):
    """Hypothèse à pouvoir vérifier d'un coup d'œil dans le journal : un point à Z = 0."""
    spectrum.Zre = spectrum.Zre.copy()
    spectrum.Zim = spectrum.Zim.copy()
    spectrum.Zre[7] = spectrum.Zim[7] = 0.0
    _use(monkeypatch, FailingInverter)
    with pytest.raises(engine.DRTComputationError) as info:
        engine.fit_drt(spectrum, mode="optimize")
    assert "1 point(s) à Z = 0" in info.value.detail


def test_the_failure_is_logged_with_its_traceback_by_the_engine(monkeypatch, spectrum, caplog):
    _use(monkeypatch, FailingInverter)
    with caplog.at_level(logging.ERROR, logger="drt.engine"):
        with pytest.raises(engine.DRTComputationError):
            engine.fit_drt(spectrum, mode="optimize")
    assert "Traceback (most recent call last)" in caplog.text and "domain error" in caplog.text


def test_a_failure_after_the_inversion_is_reported_with_its_own_stage(monkeypatch, spectrum):
    """Ni cvxopt ni Stan ici : un KeyError à la lecture de γ(τ). Avant, il sortait tel quel
    (ni ValueError ni RuntimeError) et le pipeline laissait tomber TOUTE l'analyse."""
    _use(monkeypatch, NoDistributionInverter)
    with pytest.raises(engine.DRTComputationError) as info:
        engine.fit_drt(spectrum, mode="optimize")
    assert "lecture de γ(τ)" in str(info.value) and "KeyError" in str(info.value)
    assert isinstance(info.value, RuntimeError)


def test_an_exception_without_a_message_is_still_identified(monkeypatch, spectrum):
    class Silent(FailingInverter):
        def fit(self, f, Z, **kwargs):
            raise ZeroDivisionError()

    _use(monkeypatch, Silent)
    with pytest.raises(engine.DRTComputationError) as info:
        engine.fit_drt(spectrum, mode="optimize")
    assert "ZeroDivisionError" in str(info.value)


def test_input_validation_errors_are_still_plain_value_errors_and_not_wrapped(monkeypatch, spectrum):
    """Une entrée invalide est un ValueError net, levé AVANT le calcul : pas de faux « échec »."""
    _use(monkeypatch, FailingInverter)
    spectrum.f = spectrum.f[:5]
    spectrum.Zre, spectrum.Zim = spectrum.Zre[:5], spectrum.Zim[:5]
    with pytest.raises(ValueError, match="trop court") as info:
        engine.fit_drt(spectrum, mode="optimize")
    assert not isinstance(info.value, engine.DRTComputationError)


def test_keyboard_interrupt_is_never_swallowed(monkeypatch, spectrum):
    class Interrupted(FailingInverter):
        def fit(self, f, Z, **kwargs):
            raise KeyboardInterrupt

    _use(monkeypatch, Interrupted)
    with pytest.raises(KeyboardInterrupt):
        engine.fit_drt(spectrum, mode="optimize")


# ── Pipeline : message court à l'utilisateur, diagnostic complet au journal ───
def _failing_fit_drt(*args, **kwargs):
    raise engine.DRTComputationError(
        "échec à l'étape « inversion » (ValueError: domain error)", "DIAGNOSTIC COMPLET\nTraceback (most recent call last):\n...")


def test_pipeline_keeps_the_short_message_for_the_user_and_logs_the_full_detail(monkeypatch, caplog):
    monkeypatch.setattr(pipeline.drt_engine, "fit_drt", _failing_fit_drt)
    sp, analysis = make_spectrum("e1_c3_r1.csv"), GroupAnalysis(label="g")
    with caplog.at_level(logging.ERROR, logger="pipeline"):
        pipeline._run_drt(sp, {"mode": "optimize"}, analysis)
    assert analysis.drt_failures["e1_c3_r1.csv"] == "échec à l'étape « inversion » (ValueError: domain error)"
    assert any("e1_c3_r1.csv" in w and "domain error" in w for w in analysis.warnings)
    assert "DIAGNOSTIC COMPLET" in caplog.text and "Traceback" in caplog.text        # au journal
    assert not any("DIAGNOSTIC COMPLET" in w for w in analysis.warnings)             # pas à l'écran
    assert engine.DIST_NAME not in sp.fit_results and not sp.fit_results             # rien de partiel


def test_pipeline_does_not_log_a_detail_for_a_plain_error(monkeypatch, caplog):
    def plain(*a, **k):
        raise RuntimeError("Moteur DRT indisponible — CmdStan introuvable")

    monkeypatch.setattr(pipeline.drt_engine, "fit_drt", plain)
    analysis = GroupAnalysis(label="g")
    with caplog.at_level(logging.ERROR, logger="pipeline"):
        pipeline._run_drt(make_spectrum("a"), {"mode": "optimize"}, analysis)
    assert "diagnostic complet" not in caplog.text
    assert "indisponible" in analysis.drt_failures["a"]


def test_recompute_drt_logs_the_detail_then_raises_to_the_ui(monkeypatch, caplog):
    """L'UI affiche le message court (st.error) ; seul le journal recevait RIEN auparavant."""
    monkeypatch.setattr(pipeline.drt_engine, "fit_drt", _failing_fit_drt)
    sp = make_spectrum("e1_c3_r1.csv")
    with caplog.at_level(logging.ERROR, logger="pipeline"):
        with pytest.raises(engine.DRTComputationError, match="domain error"):
            pipeline.recompute_drt(pipeline.EISSession(), sp, {}, mode="optimize")
    assert "DIAGNOSTIC COMPLET" in caplog.text
