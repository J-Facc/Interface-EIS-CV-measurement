"""Tests de CARACTÉRISATION de core/pipeline.py (filet de sécurité avant refonte).

Ces tests figent le comportement RÉEL du code à l'instant T (HEAD audité par
AUDIT.md), y compris ses défauts connus. Ils ne disent pas que ce comportement est
correct : ils détectent qu'il a CHANGÉ. Quand un défaut documenté est corrigé, le
test qui le fige doit échouer — c'est le signal voulu, on met alors le test à jour
en même temps que le correctif.

Défauts figés ici (chacun est signalé par un commentaire « COMPORTEMENT ACTUEL
BOGUÉ ») : ERR-1, ERR-2, ERR-3, DRT-1/DRT-2, plus deux observations non listées
dans l'audit (asymétrie ``group.fit_results`` / ``group.spectrum.fit_results``,
et ``recompute_drt`` sur un groupe).

Organisation
    A. run_pipeline, cas normal (réplicats + structure d'erreur caractérisable)
    B. ERR-1 : ni réplicats ni structure persistée → fits vides sans exception
    C. ERR-3 : toute erreur est convertie en « spectre sans fit »
    D. recompute_drt (modèle DRT factice, sans CmdStan)
    E. DRT réelle de bout en bout (sautée sans CmdStan ; exécutée par le job CI « drt »)

Aucun test n'écrit dans config/error_structure.json : chaque run reçoit son propre
``persistence_path`` (tests/synthetic_data.make_config).
"""

import json
import logging
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import fits.drt_fit as drt
import fits.registry as registry
from core.config import config_to_dict, load_config
from core.loader import load_spectrum
from core.models import EISSession, EISSpectrum, FitResult
from core.pipeline import (
    DRT_MODEL_NAME,
    _iter_session_spectra,
    _resolve_spectrum,
    _run_kk,
    recompute_drt,
    run_pipeline,
    validate_session,
)
from core.validator import ValidationResult
from fits.base import BaseFitModel
from fits.randles_full import _PARAM_NAMES
from tests.synthetic_data import (
    N_POINTS,
    N_POINTS_AUDIT,
    eclab_bytes,
    make_config,
    noisy_arrays,
    randles_file,
    replicate_assignments,
)

_HAVE_CMDSTAN = drt.bayes_available() and drt.cmdstan_available()
_NEEDS_CMDSTAN = pytest.mark.skipif(
    not _HAVE_CMDSTAN,
    reason="DRT réelle : exige cvxopt + cmdstanpy + une installation CmdStan (job CI « drt »)",
)

# Rct vrais des jeux synthétiques (fixture `nominal`).
_RCT_BARE, _RCT_PROBE, _RCT_C1, _RCT_C2 = 2500.0, 3000.0, 3500.0, 4200.0
_C1, _C2 = 1e-9, 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Outils
# ─────────────────────────────────────────────────────────────────────────────

class _StubModel(BaseFitModel):
    """Modèle de fit factice : remplace un plugin du registre sans rien calculer.

    Enregistre chaque appel (label du spectre, config reçue) et renvoie un
    FitResult minimal, ou lève ``raises``. Sert à tester l'orchestration du pipeline
    indépendamment de la numérique de Randles et de CmdStan.
    """

    label = "stub"

    def __init__(self, name, rct=1234.0, raises=None):
        self.name = name
        self.rct = rct
        self.raises = raises
        self.calls = []

    def fit(self, spectrum, config, weights=None):
        self.calls.append((spectrum.label, config))
        if self.raises is not None:
            raise self.raises
        n = len(spectrum.f)
        z = np.ones(n)
        return FitResult(
            model_name=self.name, params={"Rct": self.rct}, params_std={},
            Zfit_re=z, Zfit_im=z, chi2_reduced=1.0, residuals_re=z, residuals_im=z,
            Rct=self.rct, Rct_std=0.0, converged=True,
            drt_mode=((config or {}).get("fit", {}).get("drt", {}) or {}).get("mode"),
        )

    def initial_guess(self, spectrum, config):
        return {}

    def bounds(self, config):
        return ({}, {})


@pytest.fixture
def stub_registry(monkeypatch):
    """Permet d'injecter des modèles factices : ``stubs["drt_bayes"] = _StubModel(...)``.

    Les noms absents de ``stubs`` retombent sur le vrai registre. run_pipeline et
    recompute_drt importent ``get_model`` à l'appel, donc le patch est vu.
    """
    stubs = {}
    real_get = registry.get_model

    def fake_get(name):
        return stubs[name] if name in stubs else real_get(name)

    monkeypatch.setattr(registry, "get_model", fake_get)
    return stubs


def _full_assignments():
    """bare + probe + 2 concentrations, 3 réplicats chacun.

    Les concentrations sont fournies dans le DÉSORDRE (1e-8 avant 1e-9) : le
    pipeline doit les trier.
    """
    fa = []
    fa += replicate_assignments("bare", 0.0, _RCT_BARE, 3, 100, "bare")
    fa += replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
    fa += replicate_assignments("hybridization", _C2, _RCT_C2, 3, 16, "c2")
    fa += replicate_assignments("hybridization", _C1, _RCT_C1, 3, 10, "c1")
    return fa


@pytest.fixture
def nominal(tmp_path):
    """Run complet réplicats + structure d'erreur caractérisable, Randles seul."""
    path = tmp_path / "es.json"
    cfg = make_config(path)
    session, validation = run_pipeline(_full_assignments(), cfg, active_models=["randles_full"])
    return SimpleNamespace(session=session, validation=validation, cfg=cfg, path=path)


def _persisted(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _coeffs(fr):
    """(α, β, γ, δ) de la structure d'erreur qui a pondéré ce fit."""
    c = fr.error_structure_coeffs
    return (c["alpha"], c["beta"], c["gamma"], c["delta"])


def _error_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


# ═════════════════════════════════════════════════════════════════════════════
# A. Cas normal
# ═════════════════════════════════════════════════════════════════════════════

def test_session_structure(nominal):
    s = nominal.session
    assert isinstance(s, EISSession)
    assert s.config is nominal.cfg           # la config passée est conservée telle quelle
    assert isinstance(s.created_at, datetime)
    assert s.bare_reference is None          # jamais posée par le pipeline (affichage seul)

    # Le spectre moyenné porte le label du premier réplicat + " (avg)".
    assert s.bare.label == "bare_r0.txt (avg)"
    assert s.probe.label == "probe_r0.txt (avg)"
    for sp in (s.bare, s.probe):
        assert sp.n_replicates == 3
        assert sp.n_points == N_POINTS
        assert sp.sigma_re is not None and sp.sigma_im is not None

    # Groupes triés par concentration croissante, quel que soit l'ordre d'entrée.
    assert [g.concentration for g in s.groups] == [_C1, _C2]
    assert [g.spectrum.label for g in s.groups] == ["c1_r0.txt (avg)", "c2_r0.txt (avg)"]


def test_averaged_fits_recover_rct_and_carry_provenance(nominal):
    s = nominal.session
    cases = [
        (s.bare.fit_results, _RCT_BARE),
        (s.probe.fit_results, _RCT_PROBE),
        (s.groups[0].fit_results, _RCT_C1),
        (s.groups[1].fit_results, _RCT_C2),
    ]
    for fit_results, rct_true in cases:
        assert list(fit_results) == ["randles_full"]
        fr = fit_results["randles_full"]
        assert fr.Rct == pytest.approx(rct_true, rel=0.03)
        assert fr.Rct == fr.params["Rct"]
        assert set(fr.params) == set(_PARAM_NAMES)
        assert fr.converged is True
        assert isinstance(fr.warnings, list)
        # Le spectre moyenné porte des réplicats : sa structure d'erreur est
        # caractérisée sur lui-même.
        assert fr.error_structure_source == "characterized_now"
        assert _coeffs(fr) != (0.0, 0.0, 0.0, 0.0)


def test_group_fits_live_on_the_group_not_on_its_spectrum(nominal):
    """Asymétrie (observation non listée dans l'audit).

    bare/probe rangent leurs fits dans ``spectrum.fit_results`` ; les groupes de
    concentration dans ``group.fit_results`` — et ``group.spectrum.fit_results``
    reste VIDE. Conséquence vérifiée dans test_exporter_full : les exports qui lisent
    ``grp.spectrum.fit_results`` (DRT, reconstructions) omettent les groupes.
    """
    s = nominal.session
    assert "randles_full" in s.probe.fit_results
    for g in s.groups:
        assert "randles_full" in g.fit_results
        assert g.spectrum.fit_results == {}


def test_kk_verdict_is_attached_to_averaged_fits_only(nominal):
    s = nominal.session
    averaged = [s.bare, s.probe]
    for fr in [sp.fit_results["randles_full"] for sp in averaged] + [
        g.fit_results["randles_full"] for g in s.groups
    ]:
        assert isinstance(fr.kk_passed, bool)
        assert set(fr.kk_residuals) == {
            "kk_passed", "residuals_re", "residuals_im",
            "Z_kk_re", "Z_kk_im", "max_residual", "mu",
        }
    # Les fits de réplicats n'ont pas de verdict KK.
    for rep in s.probe_replicate_spectra:
        assert rep.fit_results["randles_full"].kk_passed is None
        assert rep.fit_results["randles_full"].kk_residuals is None


def test_replicates_are_kept_and_fitted_individually(nominal):
    s = nominal.session
    families = [
        (s.bare_replicate_spectra, s.bare, _RCT_BARE),
        (s.probe_replicate_spectra, s.probe, _RCT_PROBE),
        (s.groups[0].replicate_spectra, s.groups[0].spectrum, _RCT_C1),
        (s.groups[1].replicate_spectra, s.groups[1].spectrum, _RCT_C2),
    ]
    for reps, averaged, rct_true in families:
        assert [r.label[-6:] for r in reps] == ["r0.txt", "r1.txt", "r2.txt"]
        # Ce sont les MÊMES objets que ceux conservés dans le spectre moyenné.
        assert len(averaged.replicates) == 3
        assert all(a is b for a, b in zip(reps, averaged.replicates))
        for r in reps:
            assert list(r.fit_results) == ["randles_full"]
            assert r.fit_results["randles_full"].Rct == pytest.approx(rct_true, rel=0.05)


def test_replicate_fits_reuse_a_structure_they_did_not_characterize(nominal):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-2, à revoir à l'étape 5.

    Un réplicat individuel n'a pas de σ propre : il est pondéré par la « dernière
    structure persistée » au moment de son fit (source « reused_persisted »).
      * réplicats de bare/probe → structure du PREMIER spectre caractérisé (ici bare,
        via _characterize_error_structure_upfront), pas la leur ;
      * réplicats d'une concentration → structure de leur groupe, persistée juste
        avant par le fit du spectre moyenné.
    Le résultat d'un fit dépend donc de l'ordre et de l'historique des analyses.
    """
    s = nominal.session
    bare_avg = _coeffs(s.bare.fit_results["randles_full"])
    probe_avg = _coeffs(s.probe.fit_results["randles_full"])

    for rep in s.bare_replicate_spectra + s.probe_replicate_spectra:
        assert rep.fit_results["randles_full"].error_structure_source == "reused_persisted"
    for rep in s.probe_replicate_spectra:
        used = _coeffs(rep.fit_results["randles_full"])
        assert used == pytest.approx(bare_avg)      # structure de bare…
        assert used != pytest.approx(probe_avg)     # …et non celle de probe
    for g in s.groups:
        group_avg = _coeffs(g.fit_results["randles_full"])
        for rep in g.replicate_spectra:
            assert _coeffs(rep.fit_results["randles_full"]) == pytest.approx(group_avg)


def test_error_structure_history_grows_on_every_run(nominal):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-2, à revoir à l'étape 5.

    La persistance est un JSON en append, jamais purgé : 1 entrée pour la
    caractérisation préalable + 1 par spectre moyenné porteur de réplicats fitté
    (bare, probe, 2 groupes) = 5 par run, et 10 après un second run identique.
    """
    entries = _persisted(nominal.path)
    assert len(entries) == 5
    assert all(e["source"] == "characterized_now" for e in entries)
    assert entries[0]["campaign_id"] == "bare_r0.txt (avg)"   # caractérisation préalable

    run_pipeline(_full_assignments(), nominal.cfg, active_models=["randles_full"])
    assert len(_persisted(nominal.path)) == 10


def test_validation_results_cover_every_replicate_group(nominal):
    val = nominal.validation
    # Clés dans l'ordre d'ARRIVÉE des fichiers (ici 1e-8 avant 1e-9, cf.
    # _full_assignments), contrairement à ``session.groups`` qui est trié.
    assert list(val) == ["bare", "probe", "hyb_1.00e-08", "hyb_1.00e-09"]
    for label, vr in val.items():
        assert isinstance(vr, ValidationResult)
        assert vr.label == label
        assert len(vr.replicates) == 3
        assert [r.label for r in vr.replicates] == [f"{label}_rep{i}" for i in (1, 2, 3)]
        assert vr.sigma_re.shape == (N_POINTS,)
        assert vr.f_min_common < vr.f_max_common


def test_drt_is_never_run_on_replicates(nominal, stub_registry):
    """La DRT n'est calculée que sur les spectres MOYENNÉS (pipeline.py:_fit_replicates)."""
    stub = stub_registry["drt_bayes"] = _StubModel("drt_bayes")
    session, _ = run_pipeline(
        _full_assignments(), nominal.cfg, active_models=["randles_full", "drt_bayes"]
    )

    assert [label for label, _cfg in stub.calls] == [
        "bare_r0.txt (avg)", "probe_r0.txt (avg)", "c1_r0.txt (avg)", "c2_r0.txt (avg)",
    ]
    # L'ordre des fit_results suit celui de active_models.
    assert list(session.probe.fit_results) == ["randles_full", "drt_bayes"]
    assert [list(g.fit_results) for g in session.groups] == [["randles_full", "drt_bayes"]] * 2
    for rep in session.probe_replicate_spectra + session.groups[0].replicate_spectra:
        assert list(rep.fit_results) == ["randles_full"]


def test_all_models_run_when_active_models_is_none_and_drt_stack_is_absent(tmp_path, caplog):
    """active_models=None lance tous les plugins du registre, DRT comprise.

    Sans pile DRT complète (cvxopt/cmdstanpy/CmdStan), le fit DRT échoue, l'exception
    est avalée (cf. ERR-3) et le modèle disparaît en silence des résultats.
    """
    if _HAVE_CMDSTAN:
        pytest.skip("la pile DRT est présente : le fit DRT réussirait (voir section E)")
    assert "drt_bayes" in [m.name for m in registry.all_models()]   # découvert quand même

    cfg = make_config(tmp_path / "es.json")
    fa = replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
    with caplog.at_level(logging.ERROR):
        session, _ = run_pipeline(fa, cfg, active_models=None)

    assert list(session.probe.fit_results) == ["randles_full"]
    assert any("Fit 'drt_bayes' [probe] failed" in m for m in _error_messages(caplog))


def test_upfront_characterization_uses_first_eligible_spectrum(tmp_path):
    """Ordre de choix : bare, puis probe, puis la plus petite concentration."""
    # Sans bare → probe.
    path = tmp_path / "a.json"
    fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
          + replicate_assignments("hybridization", _C1, _RCT_C1, 3, 10, "c1"))
    run_pipeline(fa, make_config(path), active_models=["randles_full"])
    assert _persisted(path)[0]["campaign_id"] == "probe_r0.txt (avg)"

    # Probe sans réplicats (donc sans σ) → la première concentration porteuse de réplicats.
    path = tmp_path / "b.json"
    fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 1, 0, "probe")
          + replicate_assignments("hybridization", _C2, _RCT_C2, 3, 16, "c2")
          + replicate_assignments("hybridization", _C1, _RCT_C1, 3, 10, "c1"))
    run_pipeline(fa, make_config(path), active_models=["randles_full"])
    assert _persisted(path)[0]["campaign_id"] == "c1_r0.txt (avg)"


def test_groups_are_keyed_by_step_and_float_concentration(tmp_path):
    """Un même couple (step, concentration) fusionne ses fichiers ; concentration
    absente = 0.0, ce qui donne un groupe « hyb_0.00e+00 »."""
    cfg = make_config(tmp_path / "es.json")
    fa = replicate_assignments("hybridization", 1e-9, _RCT_C1, 2, 10, "a")
    fa += replicate_assignments("hybridization", 1e-9, _RCT_C1, 1, 20, "b")   # même clé
    fa.append(dict(content=randles_file(_RCT_C2, seed=30), filename="noconc.txt",
                   step="hybridization"))                                     # pas de "concentration"

    session, val = run_pipeline(fa, cfg, active_models=["randles_full"])

    assert [g.concentration for g in session.groups] == [0.0, 1e-9]
    assert session.groups[1].spectrum.n_replicates == 3
    assert list(val) == ["hyb_1.00e-09", "hyb_0.00e+00"]   # ordre d'apparition des fichiers


def test_unreadable_file_is_skipped_and_the_rest_is_used(tmp_path, caplog):
    cfg = make_config(tmp_path / "es.json")
    fa = replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
    fa.append(dict(content=b"ceci n'est pas un spectre", filename="bad.txt",
                   step="probe", concentration=0.0))
    with caplog.at_level(logging.ERROR):
        session, val = run_pipeline(fa, cfg, active_models=["randles_full"])

    assert session.probe.n_replicates == 3                       # bad.txt ignoré
    assert len(val["probe"].replicates) == 3
    assert any("Skipped 'bad.txt'" in m for m in _error_messages(caplog))


def test_nothing_loadable_gives_an_empty_session_without_error(tmp_path):
    cfg = make_config(tmp_path / "es.json")
    for fa in ([], [dict(content=b"junk", filename="bad.txt", step="probe", concentration=0.0)]):
        session, val = run_pipeline(fa, cfg, active_models=["randles_full"])
        assert session.bare is None and session.probe is None
        assert session.groups == []
        assert val == {}


def test_unknown_model_name_raises_before_any_work(tmp_path):
    with pytest.raises(KeyError, match="not found"):
        run_pipeline([], make_config(tmp_path / "es.json"), active_models=["nope"])


def test_bare_only_run_has_no_probe(tmp_path):
    cfg = make_config(tmp_path / "es.json")
    fa = replicate_assignments("bare", 0.0, _RCT_BARE, 3, 100, "bare")
    session, val = run_pipeline(fa, cfg, active_models=["randles_full"])
    assert session.probe is None and session.groups == []
    assert list(val) == ["bare"]
    assert session.bare.fit_results["randles_full"].Rct == pytest.approx(_RCT_BARE, rel=0.03)


def test_validate_session_maps_each_group_to_a_validation_result():
    groups = {}
    for label, seed0 in (("g1", 0), ("g2", 5)):
        arrays = [noisy_arrays(_RCT_PROBE, 0.0, seed0 + k) for k in range(2)]
        groups[label] = {
            "f": [a[0] for a in arrays],
            "zre": [a[1] for a in arrays],
            "zim": [a[2] for a in arrays],
        }
    out = validate_session(groups, config={})
    assert list(out) == ["g1", "g2"]
    assert all(isinstance(v, ValidationResult) and len(v.replicates) == 2 for v in out.values())
    assert out["g1"].label == "g1"
    assert validate_session({}, config={}) == {}


def test_run_kk_returns_a_verdict_dict_for_a_spectrum():
    f, zre, zim = noisy_arrays(_RCT_PROBE, 0.0, 0)
    sp = load_spectrum(eclab_bytes(f, zre, zim), "kk.txt")
    kk = _run_kk(sp, {}, "kk")
    assert kk["kk_passed"] is True
    assert kk["max_residual"] < 0.05


# ═════════════════════════════════════════════════════════════════════════════
# B. ERR-1 — ni réplicats ni structure persistée
# ═════════════════════════════════════════════════════════════════════════════

def _single_file_assignments():
    """Le cas du premier usage : un fichier par condition, aucun réplicat."""
    return (
        replicate_assignments("probe", 0.0, _RCT_PROBE, 1, 0, "probe")
        + replicate_assignments("hybridization", _C1, _RCT_C1, 1, 10, "c1")
        + replicate_assignments("hybridization", _C2, _RCT_C2, 1, 16, "c2")
    )


def test_err1_no_replicates_and_no_persisted_structure_gives_empty_fits(tmp_path, caplog):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-1, à corriger à l'étape 5.

    Reproduit l'Annexe A.7. Sans réplicats ET sans structure d'erreur persistée
    (premier usage, ou après une mise à jour de launch.bat qui efface
    config/error_structure.json), ``resolve_error_structure`` lève
    ErrorStructureUnavailable. run_pipeline l'attrape et se contente de
    journaliser : AUCUNE exception, les groupes sont créés, mais TOUS les
    ``fit_results`` sont vides. L'UI affiche pourtant « ✅ Analyse terminée ».

    Le jour où ce cas lèvera une erreur ou remontera un diagnostic explicite,
    ce test devra changer — c'est le signal que ERR-1 est corrigé.
    """
    path = tmp_path / "never_written.json"      # n'existe pas → premier usage
    cfg = make_config(path)

    with caplog.at_level(logging.ERROR):
        session, val = run_pipeline(_single_file_assignments(), cfg, active_models=["randles_full"])

    # Les 2 groupes existent bel et bien…
    assert [g.concentration for g in session.groups] == [_C1, _C2]
    assert session.probe is not None
    # …mais aucun fit n'a abouti, nulle part : ni sur les moyennes, ni sur les réplicats.
    assert session.probe.fit_results == {}
    assert [g.fit_results for g in session.groups] == [{}, {}]
    assert [g.spectrum.fit_results for g in session.groups] == [{}, {}]
    assert [r.fit_results for r in session.probe_replicate_spectra] == [{}]
    assert [[r.fit_results for r in g.replicate_spectra] for g in session.groups] == [[{}], [{}]]
    # Rien n'a jamais été caractérisé ni persisté.
    assert not path.exists()
    # La seule trace est dans les logs, jamais dans la valeur de retour.
    errors = _error_messages(caplog)
    assert errors
    assert all("structure d'erreur non caractérisée" in m for m in errors)
    assert any("Fit 'randles_full' [probe] failed" in m for m in errors)
    # La validation KK, elle, aboutit (sans σ : pas de réplicats).
    assert list(val) == ["probe", "hyb_1.00e-09", "hyb_1.00e-08"]
    assert all(v.sigma_re is None for v in val.values())


def test_err1_two_replicates_are_not_enough_either(tmp_path):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-1, à corriger à l'étape 5.

    Variante non listée dans l'audit : la caractérisation exige ≥ 3 réplicats
    (_MIN_REPLICATES_DEFAULT). Avec 2 réplicats par condition, on retombe dans
    exactement le même silence : fits vides, aucune exception.
    """
    path = tmp_path / "never_written.json"
    fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 2, 0, "probe")
          + replicate_assignments("hybridization", _C1, _RCT_C1, 2, 10, "c1"))
    session, _ = run_pipeline(fa, make_config(path), active_models=["randles_full"])

    assert session.probe.n_replicates == 2       # les réplicats existent bien…
    assert session.probe.fit_results == {}       # …mais ne suffisent pas à caractériser
    assert session.groups[0].fit_results == {}
    assert not path.exists()


def test_err1_disappears_as_soon_as_a_structure_is_persisted(caplog):
    """Contre-épreuve : les MÊMES fichiers, avec une structure persistée, sont fittés.

    Le ``conftest.py`` (autouse) pré-remplit la persistance PAR DÉFAUT : c'est la
    raison pour laquelle aucun test hérité ne voit ERR-1 (AUDIT.md §8.3). Ici la
    config ne fixe pas de persistence_path, donc cette structure de référence sert.
    """
    cfg = config_to_dict(load_config())          # pas de persistence_path → défaut (conftest)
    session, _ = run_pipeline(_single_file_assignments(), cfg, active_models=["randles_full"])

    for fr in [session.probe.fit_results["randles_full"]] + [
        g.fit_results["randles_full"] for g in session.groups
    ]:
        assert fr.error_structure_source == "reused_persisted"
        assert fr.converged is True
    assert session.groups[1].fit_results["randles_full"].Rct == pytest.approx(_RCT_C2, rel=0.05)


# ═════════════════════════════════════════════════════════════════════════════
# C. ERR-3 — toute erreur devient « spectre sans fit »
# ═════════════════════════════════════════════════════════════════════════════

def test_err3_a_crashing_model_is_swallowed_everywhere(tmp_path, stub_registry, caplog):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-3, à corriger à l'étape 5.

    Un modèle qui lève (ici un RuntimeError, donc un vrai bug de programmation)
    est indiscernable d'un fichier invalide : chaque ``except Exception`` du
    pipeline le convertit en « spectre sans fit ». Aucune exception ne remonte.
    """
    stub = stub_registry["randles_full"] = _StubModel("randles_full", raises=RuntimeError("boom"))
    cfg = make_config(tmp_path / "es.json")

    with caplog.at_level(logging.ERROR):
        session, val = run_pipeline(_full_assignments(), cfg, active_models=["randles_full"])

    assert len(session.groups) == 2
    assert session.bare.fit_results == {} and session.probe.fit_results == {}
    assert [g.fit_results for g in session.groups] == [{}, {}]
    assert all(r.fit_results == {} for r in session.probe_replicate_spectra)
    assert len(val) == 4                               # la validation n'est pas affectée
    errors = _error_messages(caplog)
    assert any("Fit 'randles_full' [probe] failed: boom" in m for m in errors)
    assert any("Fit réplicat 'randles_full' [probe_r0.txt] failed: boom" in m for m in errors)
    # bare, probe, 2 groupes (moyennes) + 12 réplicats
    assert len(stub.calls) == 4 + 12


def test_err3_a_crashing_kk_check_only_removes_the_kk_verdict(tmp_path, stub_registry, monkeypatch, caplog):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-3, à corriger à l'étape 5."""
    stub_registry["randles_full"] = _StubModel("randles_full")

    def boom(spectrum, config):
        raise RuntimeError("kk exploded")

    monkeypatch.setattr("fits.kk_validation.kramers_kronig_check", boom)
    cfg = make_config(tmp_path / "es.json")
    fa = replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")

    with caplog.at_level(logging.ERROR):
        session, _ = run_pipeline(fa, cfg, active_models=["randles_full"])

    fr = session.probe.fit_results["randles_full"]
    assert fr.kk_passed is None and fr.kk_residuals is None
    assert any("kramers_kronig_check [probe] failed: kk exploded" in m for m in _error_messages(caplog))
    assert _run_kk(session.probe, cfg, "x") is None


def test_err3_a_crashing_validation_gives_an_empty_dict(tmp_path, monkeypatch, caplog):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-3, à corriger à l'étape 5."""
    import core.pipeline as pipeline_mod

    def boom(groups, config):
        raise RuntimeError("validation exploded")

    monkeypatch.setattr(pipeline_mod, "validate_session", boom)
    cfg = make_config(tmp_path / "es.json")
    fa = replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")

    with caplog.at_level(logging.ERROR):
        session, val = run_pipeline(fa, cfg, active_models=["randles_full"])

    assert val == {}
    assert "randles_full" in session.probe.fit_results        # le reste de la session est intact
    assert any("validate_session failed: validation exploded" in m for m in _error_messages(caplog))


# ═════════════════════════════════════════════════════════════════════════════
# D. recompute_drt — avec un modèle DRT factice (aucune dépendance à CmdStan)
# ═════════════════════════════════════════════════════════════════════════════

def test_drt_model_name_is_the_registry_name():
    assert DRT_MODEL_NAME == "drt_bayes" == drt.DRTBayesModel.name


def test_iter_session_spectra_order_and_labels(nominal):
    labels = [label for label, _sp in _iter_session_spectra(nominal.session)]
    assert labels == (
        ["bare"] + [f"bare#{i}" for i in range(3)]
        + ["probe"] + [f"probe#{i}" for i in range(3)]
        + ["1.00e-09"] + [f"1.00e-09#{i}" for i in range(3)]
        + ["1.00e-08"] + [f"1.00e-08#{i}" for i in range(3)]
    )
    assert list(_iter_session_spectra(EISSession())) == []


@pytest.mark.parametrize("label", [
    "bare", "bare#0", "probe", "probe#2", "1.00e-09", "1.00e-09#1", "1.00e-08", "1.00e-08#2",
])
def test_recompute_drt_resolves_a_spectrum_by_session_label(nominal, stub_registry, label):
    stub = stub_registry["drt_bayes"] = _StubModel("drt_bayes")
    expected = dict(_iter_session_spectra(nominal.session))[label]

    fr = recompute_drt(nominal.session, label, nominal.cfg)

    assert _resolve_spectrum(nominal.session, label) is expected
    assert len(stub.calls) == 1
    assert expected.fit_results["drt_bayes"] is fr          # le résultat REMPLACE/PLACE l'entrée
    assert stub.calls[0][0] == expected.label


def test_recompute_drt_accepts_a_spectrum_object_even_outside_the_session(nominal, stub_registry):
    stub_registry["drt_bayes"] = _StubModel("drt_bayes")
    foreign = EISSpectrum(
        label="hors session", f=np.logspace(5, -1, 10), Zre=np.ones(10), Zim=np.ones(10),
        concentration=0.0, step="probe", n_points=10,
    )
    assert _resolve_spectrum(nominal.session, foreign) is foreign   # aucune vérification d'appartenance
    fr = recompute_drt(nominal.session, foreign, nominal.cfg)
    assert foreign.fit_results["drt_bayes"] is fr


def test_recompute_drt_unknown_spectrum_raises_value_error(nominal, stub_registry):
    stub = stub_registry["drt_bayes"] = _StubModel("drt_bayes")
    with pytest.raises(ValueError, match="Spectre introuvable dans la session : 'probe#9'"):
        recompute_drt(nominal.session, "probe#9", nominal.cfg)
    assert stub.calls == []


def test_recompute_drt_forces_the_mode_without_mutating_the_shared_config(nominal, stub_registry):
    stub = stub_registry["drt_bayes"] = _StubModel("drt_bayes")
    cfg = nominal.cfg
    assert cfg["fit"]["drt"]["mode"] == "optimize"        # défaut du YAML

    fr = recompute_drt(nominal.session, "probe", cfg)                  # mode par défaut = 'sample'
    received = stub.calls[-1][1]
    assert received["fit"]["drt"]["mode"] == "sample"
    assert fr.drt_mode == "sample"
    assert received["fit"]["error_structure"] == cfg["fit"]["error_structure"]   # reste préservé

    recompute_drt(nominal.session, "probe", cfg, mode="optimize")
    assert stub.calls[-1][1]["fit"]["drt"]["mode"] == "optimize"

    # La config partagée n'a jamais été modifiée (copie des dicts imbriqués).
    assert cfg["fit"]["drt"]["mode"] == "optimize"
    assert received is not cfg and received["fit"] is not cfg["fit"]
    assert received["fit"]["drt"] is not cfg["fit"]["drt"]


def test_recompute_drt_tolerates_a_missing_config(nominal, stub_registry):
    stub = stub_registry["drt_bayes"] = _StubModel("drt_bayes")
    recompute_drt(nominal.session, "probe", None, mode="optimize")
    assert stub.calls[0][1] == {"fit": {"drt": {"mode": "optimize"}}}


def test_recompute_drt_on_a_group_label_does_not_touch_group_fit_results(nominal, stub_registry):
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit).

    Pour un label de groupe (« 1.00e-09 »), le résultat est rangé dans
    ``group.spectrum.fit_results`` alors que le pipeline avait rangé les fits du
    groupe dans ``group.fit_results`` : le recalcul n'atteint jamais ce dernier.
    Voir aussi la version DRT réelle (section E), où les deux copies divergent.
    """
    stub_registry["drt_bayes"] = _StubModel("drt_bayes")
    group = nominal.session.groups[0]

    fr = recompute_drt(nominal.session, "1.00e-09", nominal.cfg, mode="optimize")

    assert group.spectrum.fit_results["drt_bayes"] is fr
    assert "drt_bayes" not in group.fit_results


# ═════════════════════════════════════════════════════════════════════════════
# E. DRT réelle de bout en bout — sautée sans CmdStan, exécutée par le job CI « drt »
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def drt_run(tmp_path_factory):
    """Scénario de l'Annexe A.2 (60 points, Randles bruités à 0,5 %), DRT seule."""
    cfg = make_config(tmp_path_factory.mktemp("drt") / "es.json")
    n = N_POINTS_AUDIT
    fa = (replicate_assignments("probe", 0.0, 3000.0, 3, 0, "probe", n_points=n)
          + replicate_assignments("hybridization", 1e-9, 3500.0, 3, 10, "c1", n_points=n)
          + replicate_assignments("hybridization", 1e-8, 4200.0, 3, 13, "c2", n_points=n))
    session, val = run_pipeline(fa, cfg, active_models=["drt_bayes"])
    return SimpleNamespace(session=session, validation=val, cfg=cfg)


@_NEEDS_CMDSTAN
def test_drt_default_map_is_silently_wrong_on_randles_like_spectra(drt_run):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md DRT-1 / DRT-2, à corriger lors de la refonte.

    Reproduit l'Annexe A.2/§4.4 : avec les réglages par défaut du vendor
    (``init_from_ridge=False``, ``nonneg=False``, graine 1234), le MAP tombe dans un
    optimum dégénéré sur des spectres de type Randles. Mesuré : Rct_drt = −6080 /
    −7130 / −8410 Ω et Rp < 0, mais ``converged=True`` (codé en dur) et
    ``warnings=[]`` : aucun garde-fou ne le voit.

    Seuls les SIGNES sont figés (pas les valeurs) : l'optimum dégénéré dépend de
    l'initialisation Stan. Un correctif de la DRT doit faire échouer ce test.
    """
    s = drt_run.session
    fits = [s.probe.fit_results["drt_bayes"]] + [g.fit_results["drt_bayes"] for g in s.groups]
    assert len(fits) == 3
    for fr in fits:
        assert fr.Rct < 0
        assert fr.params["Rp"] < 0
        assert fr.params["rct_source"] == "peak_penultimate"
        assert fr.converged is True            # codé en dur (DRT-2)
        assert fr.warnings == []               # aucune garde qualité (DRT-2)
        assert fr.drt_mode == "optimize"
        assert fr.model_name == "drt_bayes"


@_NEEDS_CMDSTAN
def test_drt_runs_on_averaged_spectra_only(drt_run):
    s = drt_run.session
    assert all(r.fit_results == {} for r in s.probe_replicate_spectra)
    assert all(r.fit_results == {} for g in s.groups for r in g.replicate_spectra)
    assert [g.spectrum.fit_results for g in s.groups] == [{}, {}]      # asymétrie (cf. section A)


@_NEEDS_CMDSTAN
def test_recompute_drt_real_optimize_is_deterministic_and_replaces_the_result(drt_run):
    s, cfg = drt_run.session, drt_run.cfg
    previous = s.probe.fit_results["drt_bayes"]

    fr = recompute_drt(s, "probe", cfg, mode="optimize")

    assert fr is not previous
    assert s.probe.fit_results["drt_bayes"] is fr
    assert fr.drt_mode == "optimize"
    assert fr.Rct == pytest.approx(previous.Rct, rel=1e-6)   # même graine, même optimum


@_NEEDS_CMDSTAN
def test_recompute_drt_real_on_a_replicate_and_on_a_group(drt_run):
    s, cfg = drt_run.session, drt_run.cfg
    rep = s.probe_replicate_spectra[0]
    assert "drt_bayes" not in rep.fit_results
    fr_rep = recompute_drt(s, "probe#0", cfg, mode="optimize")
    assert rep.fit_results["drt_bayes"] is fr_rep

    # COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit) : sur un groupe,
    # le recalcul crée une SECONDE copie dans group.spectrum.fit_results ; celle du
    # pipeline, dans group.fit_results, n'est pas remplacée — les deux coexistent.
    group = s.groups[0]
    original = group.fit_results["drt_bayes"]
    fr_group = recompute_drt(s, "1.00e-09", cfg, mode="optimize")
    assert group.spectrum.fit_results["drt_bayes"] is fr_group
    assert group.fit_results["drt_bayes"] is original
    assert fr_group is not original


@_NEEDS_CMDSTAN
@pytest.mark.slow
def test_recompute_drt_real_sample_mode_returns_credible_bounds(drt_run):
    """HMC (~30 s) : le mode 'sample' est celui du bouton « recalculer » de l'UI."""
    fr = recompute_drt(drt_run.session, "probe", drt_run.cfg, mode="sample")
    assert fr.drt_mode == "sample"
    assert fr.drt_gamma_lo is not None and fr.drt_gamma_hi is not None
    assert len(fr.drt_gamma_lo) == len(fr.drt_gamma) == len(fr.drt_tau)
    assert drt_run.session.probe.fit_results["drt_bayes"] is fr
