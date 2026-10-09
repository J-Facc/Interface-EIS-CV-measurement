"""Tests de core/pipeline.py — pipeline de l'étape 5 (measurement model AVANT le fit,
fit Orazem et DRT sur CHAQUE réplicat brut ET sur la moyenne).

Ce fichier remplace les tests de CARACTÉRISATION de l'étape 1, qui figeaient le
comportement de l'ancien pipeline, défauts compris. Chaque test qui figeait un défaut
(« COMPORTEMENT ACTUEL BOGUÉ ») vérifie désormais le comportement CORRIGÉ ; son nom ou
sa docstring dit « corrigé » et rappelle l'ancien comportement :

  * ERR-1 — sans structure d'erreur, l'ancien pipeline rendait des ``fit_results`` vides
    sans rien signaler : le groupe est maintenant ARRÊTÉ avec un statut et un message
    explicites, portés par la session (section B) et affichés par l'UI (section H) ;
  * ERR-2 — les réplicats étaient pondérés par la structure « persistée » d'un AUTRE
    groupe, et un JSON grossissait à chaque run : chaque groupe est pondéré par SA
    structure, rien n'est écrit sur disque (section A) ;
  * ERR-3 — tout ``except Exception`` convertissait un bug en « spectre sans fit » :
    donnée invalide → résultat dégradé avec message ; bug → l'exception REMONTE (C) ;
  * DRT absente des réplicats, KK calculé APRÈS les fits (AUDIT.md §5.5) : sections A, D ;
  * ``recompute_drt`` sur un groupe n'atteignait pas ``group.fit_results`` (E) ;
  * DRT MAP fausse en silence avec les réglages amont (DRT-1/DRT-2) : section G.

Organisation
    A. Cas nominal (réplicats, structure d'erreur caractérisable), DRT désactivée
    B. ERR-1 corrigé : groupe arrêté, statut explicite
    C. ERR-3 corrigé : donnée invalide vs bug logiciel
    D. DRT par réplicat et sur la moyenne (moteur DRT factice, sans CmdStan)
    E. recompute_drt (moteur factice)
    F. Tables de résultats (intra-fit vs inter-réplicats) et export
    G. DRT réelle de bout en bout (sautée sans CmdStan ; job CI « drt »)
    H. Propagation jusqu'à l'UI (Streamlit AppTest)
"""

import copy
import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import core.pipeline as pipeline
import drt.engine as drt_engine
from core.models import (
    GROUP_ERROR_STRUCTURE_UNAVAILABLE,
    GROUP_INVALID_INPUT,
    GROUP_OK,
    EISSession,
    EISSpectrum,
    FitResult,
)
from core.pipeline import (
    DRT_MODEL_NAME,
    InvalidAnalysisInput,
    _iter_session_spectra,
    _resolve_spectrum,
    aggregate_drt_target,
    build_circuit_fit,
    recompute_drt,
    run_pipeline,
)
from core.results_table import drt_hmc_summary, group_rows, replicate_rows
from core.validator import ValidationResult
from fits.orazem_fit import ORAZEM_MODEL_NAME
from tests.synthetic_data import (
    N_POINTS,
    N_POINTS_AUDIT,
    RANDLES_EXPRESSION,
    make_config,
    randles_file,
    replicate_assignments,
)

REPO = Path(__file__).resolve().parents[1]

_HAVE_CMDSTAN, _WHY_NOT = drt_engine.engine_available()
_NEEDS_CMDSTAN = pytest.mark.skipif(
    not _HAVE_CMDSTAN, reason=f"DRT réelle : {_WHY_NOT} (job CI « drt »)")

# Rct vrais des jeux synthétiques.
_RCT_BARE, _RCT_PROBE, _RCT_C1, _RCT_C2 = 2500.0, 3000.0, 3500.0, 4200.0
_C1, _C2 = 1e-9, 1e-8
_NO_DRT = dict(enabled=False)


# ─────────────────────────────────────────────────────────────────────────────
# Outils
# ─────────────────────────────────────────────────────────────────────────────

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


def _single_file_assignments():
    """Le cas du premier usage (ERR-1) : un fichier par condition, aucun réplicat."""
    return (
        replicate_assignments("probe", 0.0, _RCT_PROBE, 1, 0, "probe")
        + replicate_assignments("hybridization", _C1, _RCT_C1, 1, 10, "c1")
        + replicate_assignments("hybridization", _C2, _RCT_C2, 1, 16, "c2")
    )


def _groups(session):
    """[(label d'affichage, spectre moyen, réplicats, GroupAnalysis)]."""
    return list(session.iter_groups())


class _FakeDRT:
    """Remplace ``drt.engine.fit_drt`` : enregistre chaque appel, rend un FitResult DRT
    plausible (Rct = étendue de Zre, std a posteriori 5 Ω en 'sample', NaN en MAP), ou
    lève l'exception prévue pour un label donné."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = dict(fail or {})

    def __call__(self, spectrum, *, mode, **kwargs):
        self.calls.append((spectrum.label, mode))
        if spectrum.label in self.fail:
            raise self.fail[spectrum.label]
        n = len(spectrum.f)
        hmc = mode == "sample"
        rct = float(np.max(spectrum.Zre) - np.min(spectrum.Zre))
        std = 5.0 if hmc else float("nan")
        sampler = ({"rhat_max": 1.002, "divergences": 0, "ess_bulk_min": 900.0, "ess_tail_min": 800.0}
                   if hmc else {"optimizer_converged": True})
        return FitResult(
            model_name=DRT_MODEL_NAME, params={"Rct": rct, "drt_mode": mode},
            params_std={"Rct": std}, Zfit_re=np.asarray(spectrum.Zre, float),
            Zfit_im=np.asarray(spectrum.Zim, float), chi2_reduced=float("nan"),
            residuals_re=np.zeros(n), residuals_im=np.zeros(n), target_param="Rct",
            target_value=rct, target_std=std, converged=True, drt_mode=mode,
            drt_tau=np.geomspace(1e-6, 10.0, 20), drt_gamma=np.ones(20),
            drt_diagnostics={"sampler": sampler, "alerts": []},
        )


@pytest.fixture
def fake_drt(monkeypatch):
    fake = _FakeDRT()
    monkeypatch.setattr(drt_engine, "fit_drt", fake)
    monkeypatch.setattr(drt_engine, "engine_available", lambda: (True, None))
    return fake


@pytest.fixture(scope="module")
def nominal():
    """Run complet réplicats + structure d'erreur caractérisable, DRT désactivée."""
    cfg = make_config(**_NO_DRT)
    session, validation = run_pipeline(_full_assignments(), cfg)
    return SimpleNamespace(session=session, validation=validation, cfg=cfg)


@pytest.fixture(scope="module")
def _fake_drt_run():
    """probe + 1 concentration, 3 réplicats, DRT FACTICE en mode MAP (config)."""
    fake = _FakeDRT()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(drt_engine, "fit_drt", fake)
        mp.setattr(drt_engine, "engine_available", lambda: (True, None))
        cfg = make_config(enabled=True, mode="optimize")
        fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
              + replicate_assignments("hybridization", _C1, _RCT_C1, 3, 10, "c1"))
        session, validation = run_pipeline(fa, cfg)
    return SimpleNamespace(session=session, validation=validation, cfg=cfg, fake=fake)


@pytest.fixture
def with_fake_drt(_fake_drt_run):
    """Copie PROFONDE du run à DRT factice : un test peut recalculer sans toucher aux
    autres (deepcopy conserve les alias internes : group.fit_results, réplicats)."""
    return copy.deepcopy(_fake_drt_run)


# ═════════════════════════════════════════════════════════════════════════════
# A. Cas nominal
# ═════════════════════════════════════════════════════════════════════════════

def test_session_structure(nominal):
    s = nominal.session
    assert isinstance(s, EISSession)
    assert s.config is nominal.cfg
    assert s.bare_reference is None                       # jamais posée par le pipeline
    assert s.circuit["expression"] == RANDLES_EXPRESSION and s.circuit["target_param"] == "Rct"
    assert s.drt_mode is None and s.messages == [] and s.load_errors == []

    assert s.bare.label == "bare_r0.txt (avg)"
    assert s.probe.label == "probe_r0.txt (avg)"
    # Groupes triés par concentration croissante, quel que soit l'ordre d'entrée.
    assert [g.concentration for g in s.groups] == [_C1, _C2]
    assert [g.spectrum.label for g in s.groups] == ["c1_r0.txt (avg)", "c2_r0.txt (avg)"]
    assert [an.status for *_x, an in _groups(s)] == [GROUP_OK] * 4
    assert s.failed_groups() == []


def test_the_mean_spectrum_carries_the_raw_replicates_at_every_step(nominal):
    """Les réplicats BRUTS sont conservés tels que chargés, et le spectre moyen les porte
    (avant : seule la moyenne était propagée après average_replicates)."""
    for _lbl, mean_sp, reps, _an in _groups(nominal.session):
        assert [r.label[-6:] for r in reps] == ["r0.txt", "r1.txt", "r2.txt"]
        assert all(a is b for a, b in zip(reps, mean_sp.replicates))       # mêmes objets
        assert all(r.replicates == [] for r in reps)                        # un réplicat est brut
        assert mean_sp not in reps
        assert mean_sp.n_replicates == 3 and mean_sp.n_points == N_POINTS
        assert np.all(np.diff(mean_sp.f) < 0)                               # HF → BF
        stack = np.array([r.Zre for r in reps])
        np.testing.assert_allclose(mean_sp.Zre, stack.mean(axis=0))
        np.testing.assert_allclose(mean_sp.sigma_re, stack.std(axis=0, ddof=1))


def test_the_measurement_model_runs_on_raw_replicates_before_any_fit(monkeypatch):
    """Ordre imposé : measurement model + KK sur les réplicats BRUTS, PUIS fit Orazem, PUIS
    DRT — et le fit reçoit EXACTEMENT l'analyse et les réplicats bruts (avant : la
    validation KK était calculée après tous les fits, AUDIT.md §5.5)."""
    events = []
    real_mm, real_fit = pipeline.analyze_replicates, pipeline.fit_replicate_group

    def spy_mm(reps, **kw):
        out = real_mm(reps, **kw)
        events.append(("mm", kw["label"], out, list(reps)))
        return out

    def spy_fit(Z, names, reps, analysis, specs, target, **kw):
        events.append(("fit", analysis.label, analysis, list(reps)))
        return real_fit(Z, names, reps, analysis, specs, target, **kw)

    fake = _FakeDRT()

    def spy_drt(sp, **kw):
        events.append(("drt", sp.label, None, None))
        return fake(sp, **kw)

    monkeypatch.setattr(pipeline, "analyze_replicates", spy_mm)
    monkeypatch.setattr(pipeline, "fit_replicate_group", spy_fit)
    monkeypatch.setattr(drt_engine, "engine_available", lambda: (True, None))
    monkeypatch.setattr(drt_engine, "fit_drt", spy_drt)
    fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
          + replicate_assignments("hybridization", _C1, _RCT_C1, 3, 10, "c1"))
    session, validation = run_pipeline(fa, make_config(mode="optimize"))

    kinds = [e[0] for e in events]
    assert kinds == ["mm", "fit"] + ["drt"] * 4 + ["mm", "fit"] + ["drt"] * 4
    for (k1, lbl1, mm, mm_reps), (k2, lbl2, fit_analysis, fit_reps), sp_reps in (
            (events[0], events[1], session.probe_replicate_spectra),
            (events[6], events[7], session.groups[0].replicate_spectra)):
        assert lbl1 == lbl2
        assert fit_analysis is mm                                         # MÊME analyse
        assert all(a is b for a, b in zip(mm_reps, sp_reps))              # réplicats BRUTS
        assert all(a is b for a, b in zip(fit_reps, sp_reps))
    # Le verdict KK rendu est celui de cette même analyse.
    assert validation["probe"].measurement_model is events[0][2]
    assert session.probe_analysis.orazem.analysis is events[0][2]


def test_every_raw_replicate_and_the_mean_are_fitted(nominal):
    """Fit Orazem sur CHAQUE réplicat (σ d'une mesure) ET sur la moyenne (σ/√n)."""
    truths = [_RCT_BARE, _RCT_PROBE, _RCT_C1, _RCT_C2]
    for (_lbl, mean_sp, reps, an), rct in zip(_groups(nominal.session), truths):
        for r in reps:
            fr = r.fit_results[ORAZEM_MODEL_NAME]
            assert list(r.fit_results) == [ORAZEM_MODEL_NAME]               # DRT désactivée
            assert fr.target_param == "Rct" and fr.converged
            assert fr.target_value == pytest.approx(rct, rel=0.05)
            assert fr is an.orazem.replicate_fits[reps.index(r)]
        fm = mean_sp.fit_results[ORAZEM_MODEL_NAME]
        assert fm.target_value == pytest.approx(rct, rel=0.03)
        assert fm.target_value == an.orazem.mean_fit.target_value
        assert set(fm.params) == {"Re", "Re_prime", "Rct", "R_D", "tau_d", "Qdl", "alpha", "Cb"}
        # Le fit de la moyenne est pondéré par σ/√n : incertitude intra-fit plus petite.
        assert fm.target_std < np.mean([r.fit_results[ORAZEM_MODEL_NAME].target_std for r in reps])


def test_mean_fit_arrays_follow_the_mean_spectrum_frequency_order(nominal):
    """Le measurement model travaille sur une grille CROISSANTE ; le FitResult rangé sur
    le spectre moyen est remis en ordre HF → BF, aligné point à point sur ``f``."""
    for _lbl, mean_sp, _reps, an in _groups(nominal.session):
        fm = mean_sp.fit_results[ORAZEM_MODEL_NAME]
        np.testing.assert_allclose(fm.Zfit_re, an.orazem.mean_fit.Zfit_re[::-1])
        np.testing.assert_allclose(mean_sp.Zre - fm.Zfit_re, fm.residuals_re, atol=1e-9)
        np.testing.assert_allclose(mean_sp.Zim - fm.Zfit_im, fm.residuals_im, atol=1e-9)


def test_aggregates_separate_intra_fit_and_inter_replicate_uncertainty(nominal):
    for _lbl, _mean, reps, an in _groups(nominal.session):
        t = an.orazem.target
        values = [r.fit_results[ORAZEM_MODEL_NAME].target_value for r in reps]
        stds = [r.fit_results[ORAZEM_MODEL_NAME].target_std for r in reps]
        assert t.n == 3 and t.n_excluded == 0
        assert t.mean == pytest.approx(np.mean(values))
        assert t.std_between == pytest.approx(np.std(values, ddof=1))        # inter-réplicats
        assert t.std_within == pytest.approx(np.sqrt(np.mean(np.square(stds))))  # intra-fit
        assert t.sem == pytest.approx(math.sqrt(max(t.std_between ** 2, t.std_within ** 2) / 3))
        assert set(an.orazem.aggregate) == set(an.orazem.param_names)


def test_err2_fixed_each_group_is_weighted_by_its_own_error_structure(nominal):
    """CORRIGÉ (était « COMPORTEMENT ACTUEL BOGUÉ — ERR-2 ») : les réplicats de bare/probe
    étaient pondérés par la structure du PREMIER spectre caractérisé (« reused_persisted »).
    Chaque fit d'un groupe — réplicats ET moyenne — l'est désormais par la structure
    caractérisée sur CE groupe, dans cette analyse."""
    seen = []
    for _lbl, mean_sp, reps, an in _groups(nominal.session):
        own = an.validation.measurement_model.error_structure.to_dict()
        for sp in reps + [mean_sp]:
            fr = sp.fit_results[ORAZEM_MODEL_NAME]
            assert fr.error_structure_source == "characterized_now"
            assert fr.error_structure_coeffs == own
        seen.append((own["alpha"], own["beta"], own["delta"]))
    assert len(set(seen)) == 4                                              # 4 structures distinctes


def test_err2_fixed_nothing_is_written_to_disk(tmp_path, monkeypatch):
    """CORRIGÉ : l'ancien pipeline ajoutait 5 entrées par run à config/error_structure.json
    (jamais purgé). Plus aucune persistance : deux runs identiques donnent le même
    résultat et n'écrivent rien."""
    monkeypatch.chdir(tmp_path)
    fa = replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
    s1, _ = run_pipeline(fa, make_config(**_NO_DRT))
    s2, _ = run_pipeline(fa, make_config(**_NO_DRT))
    assert not (REPO / "config" / "error_structure.json").exists()
    assert [p.name for p in tmp_path.iterdir()] == []
    assert (s1.probe.fit_results[ORAZEM_MODEL_NAME].target_value
            == s2.probe.fit_results[ORAZEM_MODEL_NAME].target_value)


def test_group_fit_results_is_the_mean_spectrum_dict(nominal):
    """CORRIGÉ (asymétrie relevée à l'étape 1) : ``group.fit_results`` et
    ``group.spectrum.fit_results`` étaient deux dictionnaires (le second vide). C'est
    désormais LE MÊME objet."""
    for g in nominal.session.groups:
        assert g.fit_results is g.spectrum.fit_results
        assert ORAZEM_MODEL_NAME in g.fit_results
        assert g.analysis is not None and g.replicate_spectra == g.spectrum.replicates


def test_validation_results_cover_every_group_and_come_from_the_same_analysis(nominal):
    val = nominal.validation
    # Ordre d'ARRIVÉE des fichiers (1e-8 avant 1e-9), contrairement à session.groups (trié).
    assert list(val) == ["bare", "probe", "hyb_1.00e-08", "hyb_1.00e-09"]
    analyses = {an.label: an for *_x, an in _groups(nominal.session)}
    for label, vr in val.items():
        assert isinstance(vr, ValidationResult)
        assert vr is analyses[label].validation
        assert [r.label for r in vr.replicates] == [f"{label}_rep{i}" for i in (1, 2, 3)]
        assert vr.sigma_re.shape == (N_POINTS,)
        # Le bruit synthétique (relatif, par composante) viole σ_r = σ_j : le test de la
        # structure d'erreur le rejette et estime deux structures.
        assert vr.all_valid is True and vr.error_structure_message is None
        assert vr.measurement_model.error_structure.equal_re_im is False


def test_groups_are_keyed_by_step_and_float_concentration():
    """Un même couple (step, concentration) fusionne ses fichiers ; concentration
    absente = 0.0 (groupe « hyb_0.00e+00 »)."""
    fa = replicate_assignments("hybridization", 1e-9, _RCT_C1, 2, 10, "a")
    fa += replicate_assignments("hybridization", 1e-9, _RCT_C1, 1, 20, "b")   # même clé
    fa.append(dict(content=randles_file(_RCT_C2, seed=30), filename="noconc.txt",
                   step="hybridization"))                                     # pas de "concentration"

    session, val = run_pipeline(fa, make_config(**_NO_DRT))

    assert [g.concentration for g in session.groups] == [0.0, 1e-9]
    assert len(session.groups[1].replicate_spectra) == 3
    assert list(val) == ["hyb_1.00e-09", "hyb_0.00e+00"]
    assert session.groups[1].analysis.ok and not session.groups[0].analysis.ok


def test_bare_only_run_has_no_probe():
    fa = replicate_assignments("bare", 0.0, _RCT_BARE, 3, 100, "bare")
    session, val = run_pipeline(fa, make_config(**_NO_DRT))
    assert session.probe is None and session.groups == []
    assert list(val) == ["bare"]
    assert session.bare.fit_results[ORAZEM_MODEL_NAME].target_value == pytest.approx(_RCT_BARE, rel=0.03)


def test_nothing_loadable_gives_an_empty_session():
    for fa in ([], [dict(content=b"junk", filename="bad.txt", step="probe", concentration=0.0)]):
        session, val = run_pipeline(fa, make_config(**_NO_DRT))
        assert session.bare is None and session.probe is None
        assert session.groups == [] and val == {}


def test_an_explicit_circuit_overrides_the_config_and_its_target_is_used():
    """Le circuit passé par l'UI prime sur ``fit.circuit`` ; le paramètre cible désigné
    (ici un AUTRE que Rct) est celui de target_value et de l'agrégat."""
    cf = build_circuit_fit(
        "Re + parallel(R(Rct) + ZD_bounded(R_D, tau_d), Q(Qdl, alpha))",
        {"Re": (100.0, 0.0, 1e5), "Rct": (1500.0, 0.0, 1e9), "R_D": (300.0, 0.0, 1e7),
         "tau_d": (0.2, 1e-6, 1e4), "Qdl": (5e-6, 0.0, 1e-2), "alpha": (0.8, 0.3, 1.0)},
        "R_D")
    fa = replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
    session, _ = run_pipeline(fa, make_config(**_NO_DRT), cf)
    fr = session.probe.fit_results[ORAZEM_MODEL_NAME]
    assert fr.target_param == "R_D" and fr.target_value == fr.params["R_D"]
    assert session.probe_analysis.orazem.target.name == "R_D"
    assert session.circuit["target_param"] == "R_D"


# ═════════════════════════════════════════════════════════════════════════════
# B. ERR-1 corrigé — groupe arrêté, statut explicite
# ═════════════════════════════════════════════════════════════════════════════

def test_err1_fixed_no_replicates_stops_every_group_with_an_explicit_status(fake_drt):
    """CORRIGÉ (était « COMPORTEMENT ACTUEL BOGUÉ — ERR-1 », Annexe A.7).

    Avant : ``ErrorStructureUnavailable`` était attrapée et journalisée seulement ; les
    groupes existaient avec des ``fit_results`` VIDES, sans aucune trace dans la valeur
    de retour, et l'UI affichait « ✅ Analyse terminée ». Maintenant chaque groupe porte
    ``status = GROUP_ERROR_STRUCTURE_UNAVAILABLE`` et le message destiné à l'utilisateur ;
    aucun fit — ni Orazem, ni DRT — n'est produit pour lui.
    """
    session, val = run_pipeline(_single_file_assignments(), make_config(mode="optimize"))

    groups = _groups(session)
    assert [lbl for lbl, *_x in groups] == ["Probe", "1.00e-09 M", "1.00e-08 M"]
    for _lbl, mean_sp, reps, an in groups:
        assert an.status == GROUP_ERROR_STRUCTURE_UNAVAILABLE and not an.ok
        assert "structure d'erreur non caractérisable" in an.message
        assert "Aucun fit n'a été réalisé pour ce groupe" in an.message
        assert an.orazem is None and an.drt_target is None
        assert mean_sp.fit_results == {} and [r.fit_results for r in reps] == [{}]
        assert mean_sp.replicates == reps and mean_sp is not reps[0]
    assert fake_drt.calls == []                                            # DRT arrêtée aussi
    assert [lbl for lbl, _an in session.failed_groups()] == ["Probe", "1.00e-09 M", "1.00e-08 M"]
    # Le verdict KK rendu porte le MÊME message (onglet Validation KK).
    assert list(val) == ["probe", "hyb_1.00e-09", "hyb_1.00e-08"]
    for (_l, _m, _r, an), vr in zip(groups, val.values()):
        assert vr is an.validation
        assert vr.all_valid is None and vr.error_structure_message == an.message


def test_err1_fixed_two_replicates_are_not_enough_either():
    """CORRIGÉ : la caractérisation exige ≥ 3 réplicats ; avec 2, même arrêt EXPLICITE."""
    fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 2, 0, "probe")
          + replicate_assignments("hybridization", _C1, _RCT_C1, 2, 10, "c1"))
    session, _ = run_pipeline(fa, make_config(**_NO_DRT))
    assert len(session.probe_replicate_spectra) == 2
    for _lbl, mean_sp, _reps, an in _groups(session):
        assert an.status == GROUP_ERROR_STRUCTURE_UNAVAILABLE
        assert "2 réplicat(s) fourni(s), 3 requis" in an.message
        assert mean_sp.fit_results == {}


def test_err1_only_the_failing_group_is_stopped():
    """Un groupe sans réplicats n'empêche pas l'analyse des autres."""
    fa = (replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
          + replicate_assignments("hybridization", _C1, _RCT_C1, 1, 10, "c1"))
    session, _ = run_pipeline(fa, make_config(**_NO_DRT))
    assert session.probe_analysis.ok and ORAZEM_MODEL_NAME in session.probe.fit_results
    g = session.groups[0]
    assert g.analysis.status == GROUP_ERROR_STRUCTURE_UNAVAILABLE and g.fit_results == {}
    assert [lbl for lbl, _an in session.failed_groups()] == ["1.00e-09 M"]


def test_a_circuit_incompatible_with_a_groups_data_stops_that_group_as_invalid_input():
    """``FitSpecificationError`` levée PENDANT le fit d'un groupe (ici Z(ω) non fini au
    guess : capacité initiale nulle) → statut ``GROUP_INVALID_INPUT`` et message ; pas
    de FitResult partiel."""
    cf = build_circuit_fit("Re + parallel(R(Rct), C(Cdl))",
                           {"Re": (100.0, 0.0, 1e5), "Rct": (1e3, 0.0, 1e9), "Cdl": (0.0, 0.0, 1.0)},
                           "Rct")
    fa = replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
    session, val = run_pipeline(fa, make_config(**_NO_DRT), cf)
    an = session.probe_analysis
    assert an.status == GROUP_INVALID_INPUT
    assert "fit du circuit impossible" in an.message and "n'est pas fini au guess" in an.message
    assert session.probe.fit_results == {}
    assert all(r.fit_results == {} for r in session.probe_replicate_spectra)
    assert val["probe"].all_valid is True                     # le verdict KK, lui, a été rendu


# ═════════════════════════════════════════════════════════════════════════════
# C. ERR-3 corrigé — donnée invalide (résultat dégradé + message) vs bug (remonte)
# ═════════════════════════════════════════════════════════════════════════════

def test_err3_an_unreadable_file_is_reported_in_the_session_not_only_in_the_logs():
    fa = replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
    fa.append(dict(content=b"ceci n'est pas un spectre", filename="bad.txt",
                   step="probe", concentration=0.0))
    session, val = run_pipeline(fa, make_config(**_NO_DRT))
    assert len(session.probe_replicate_spectra) == 3                    # bad.txt écarté
    assert [e["filename"] for e in session.load_errors] == ["bad.txt"]
    assert session.load_errors[0]["message"]
    assert session.probe_analysis.ok and len(val["probe"].replicates) == 3


def test_err3_fixed_a_crashing_fit_is_a_bug_and_propagates(monkeypatch):
    """CORRIGÉ (était « COMPORTEMENT ACTUEL BOGUÉ — ERR-3 ») : un fit qui lève une erreur
    de programmation était avalé partout (« spectre sans fit »). Il REMONTE."""
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "fit_replicate_group", boom)
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(_full_assignments(), make_config(**_NO_DRT))


def test_err3_fixed_a_crashing_measurement_model_is_a_bug_and_propagates(monkeypatch):
    """Seule ``ErrorStructureUnavailable`` (donnée insuffisante) arrête un groupe en
    douceur ; toute autre exception du measurement model remonte."""
    def boom(*args, **kwargs):
        raise ZeroDivisionError("bug numérique")

    monkeypatch.setattr(pipeline, "analyze_replicates", boom)
    with pytest.raises(ZeroDivisionError):
        run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                     make_config(**_NO_DRT))


def test_err3_a_crashing_loader_is_a_bug_and_propagates(monkeypatch):
    """Le contrat du loader est ``ValueError`` pour un fichier invalide ; toute autre
    exception (ici TypeError) est un bug et remonte."""
    def boom(**kwargs):
        raise TypeError("bug du loader")

    monkeypatch.setattr(pipeline, "load_spectrum", boom)
    with pytest.raises(TypeError, match="bug du loader"):
        run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                     make_config(**_NO_DRT))


@pytest.mark.parametrize("exc", [ValueError("spectre invalide"), RuntimeError("CmdStan a échoué")])
def test_err3_a_drt_data_or_engine_failure_degrades_only_that_spectrum(fake_drt, exc):
    """DRT impossible pour UN spectre (contrat de ``drt.engine.fit_drt`` : ValueError =
    spectre invalide, RuntimeError = échec de CmdStan) → motif dans ``drt_failures`` et
    les alertes du groupe ; le reste du groupe est conservé."""
    fake_drt.fail = {"probe_r1.txt": exc}
    session, _ = run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                              make_config(mode="optimize"))
    an = session.probe_analysis
    assert an.ok
    assert an.drt_failures == {"probe_r1.txt": str(exc)}
    assert any(w.startswith("DRT de « probe_r1.txt » non calculée") for w in an.warnings)
    reps = session.probe_replicate_spectra
    assert [DRT_MODEL_NAME in r.fit_results for r in reps] == [True, False, True]
    assert DRT_MODEL_NAME in session.probe.fit_results
    assert an.drt_target.n == 2 and an.drt_target.n_excluded == 1
    assert all(ORAZEM_MODEL_NAME in r.fit_results for r in reps)


def test_err3_a_drt_bug_propagates(fake_drt):
    fake_drt.fail = {"probe_r0.txt": KeyError("distribution absente")}
    with pytest.raises(KeyError):
        run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                     make_config(mode="optimize"))


def test_a_missing_drt_engine_is_reported_once_at_session_level(monkeypatch):
    calls = []
    monkeypatch.setattr(drt_engine, "engine_available", lambda: (False, "CmdStan introuvable"))
    monkeypatch.setattr(drt_engine, "fit_drt", lambda *a, **k: calls.append(a))
    session, _ = run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                              make_config(mode="optimize"))
    assert calls == [] and session.drt_mode is None
    assert len(session.messages) == 1 and "CmdStan introuvable" in session.messages[0]
    assert session.probe_analysis.ok and ORAZEM_MODEL_NAME in session.probe.fit_results


@pytest.mark.parametrize("kwargs, match", [
    (dict(circuit_expr="Re + __import__('os')"), "Circuit ou paramètres invalides"),
    (dict(target="Nope"), "paramètre cible « Nope » absent"),
    (dict(params={"Re": (100.0, 0.0, 1e5)}), "sans guess/bornes"),
])
def test_invalid_circuit_input_is_refused_before_any_work(monkeypatch, kwargs, match):
    loaded = []
    monkeypatch.setattr(pipeline, "load_spectrum", lambda **kw: loaded.append(kw))
    params = kwargs.get("params", {"Re": (100.0, 0.0, 1e5), "Rct": (1e3, 0.0, 1e9),
                                   "Cdl": (1e-6, 0.0, 1.0)})
    with pytest.raises(InvalidAnalysisInput, match=match):
        cf = build_circuit_fit(kwargs.get("circuit_expr", "Re + parallel(R(Rct), C(Cdl))"),
                               params, kwargs.get("target", "Rct"))
        run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                     make_config(**_NO_DRT), cf)
    assert loaded == []


def test_invalid_config_circuit_drt_mode_or_step_are_refused_before_any_work(monkeypatch):
    loaded = []
    monkeypatch.setattr(pipeline, "load_spectrum", lambda **kw: loaded.append(kw))
    fa = replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe")
    cfg = make_config(**_NO_DRT)
    cfg["fit"]["circuit"]["parameters"]["Rct"]["initial"] = None            # case vide dans l'UI
    with pytest.raises(InvalidAnalysisInput, match="Rct"):
        run_pipeline(fa, cfg)
    with pytest.raises(InvalidAnalysisInput, match="Réglages DRT invalides"):
        run_pipeline(fa, make_config(), run_drt=True, drt_mode="nuts")
    with pytest.raises(InvalidAnalysisInput, match="Étape inconnue"):
        run_pipeline([dict(fa[0], step="sonde")], make_config(**_NO_DRT))
    assert loaded == []


def test_unknown_model_names_no_longer_exist_as_an_api():
    """Le registre (``active_models``) a disparu : un seul moteur de fit, une DRT."""
    with pytest.raises(TypeError):
        run_pipeline([], make_config(), active_models=["randles_full"])


# ═════════════════════════════════════════════════════════════════════════════
# D. DRT sur chaque réplicat ET sur la moyenne (moteur factice)
# ═════════════════════════════════════════════════════════════════════════════

def test_drt_runs_on_every_raw_replicate_and_on_the_mean(with_fake_drt):
    """CORRIGÉ : la DRT était « volontairement exclue » des réplicats."""
    s = with_fake_drt.session
    assert with_fake_drt.fake.calls == [
        ("probe_r0.txt", "optimize"), ("probe_r1.txt", "optimize"), ("probe_r2.txt", "optimize"),
        ("probe_r0.txt (avg)", "optimize"),
        ("c1_r0.txt", "optimize"), ("c1_r1.txt", "optimize"), ("c1_r2.txt", "optimize"),
        ("c1_r0.txt (avg)", "optimize"),
    ]
    assert s.drt_mode == "optimize"
    for _lbl, mean_sp, reps, _an in _groups(s):
        for sp in reps + [mean_sp]:
            assert list(sp.fit_results) == [ORAZEM_MODEL_NAME, DRT_MODEL_NAME]


def test_drt_aggregate_in_map_mode_reports_inter_replicate_spread_only(with_fake_drt):
    """En MAP, pas d'incertitude a posteriori : l'intra-fit est NaN (« non calculé »,
    jamais 0) et l'incertitude de la moyenne vient de la seule dispersion."""
    for _lbl, _m, reps, an in _groups(with_fake_drt.session):
        d = an.drt_target
        values = [r.fit_results[DRT_MODEL_NAME].target_value for r in reps]
        assert d.name == "Rct" and d.n == 3
        assert d.mean == pytest.approx(np.mean(values))
        assert d.std_between == pytest.approx(np.std(values, ddof=1))
        assert math.isnan(d.std_within) and math.isnan(d.q_pvalue)
        assert d.sem == pytest.approx(d.std_between / math.sqrt(3))


def test_drt_aggregate_with_posterior_uncertainty_uses_the_orazem_aggregation():
    reps = []
    for k, v in enumerate((100.0, 104.0, 98.0)):
        sp = EISSpectrum(label=f"r{k}", f=np.ones(3), Zre=np.ones(3), Zim=np.ones(3),
                         concentration=0.0, step="probe", n_points=3)
        sp.fit_results[DRT_MODEL_NAME] = dataclasses.replace(
            _FakeDRT()(sp, mode="sample"), target_value=v)
        reps.append(sp)
    d = aggregate_drt_target(reps)
    assert d.std_within == pytest.approx(5.0) and not math.isnan(d.q_pvalue)
    assert aggregate_drt_target([EISSpectrum("x", np.ones(2), np.ones(2), np.ones(2), 0, "p", 2)]) is None


def test_drt_mode_comes_from_the_argument_before_the_config(fake_drt):
    session, _ = run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                              make_config(mode="optimize"), drt_mode="sample")
    assert {m for _l, m in fake_drt.calls} == {"sample"} and session.drt_mode == "sample"
    session, _ = run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                              make_config(mode="sample"), run_drt=False)
    assert session.drt_mode is None and DRT_MODEL_NAME not in session.probe.fit_results


# ═════════════════════════════════════════════════════════════════════════════
# E. recompute_drt (moteur factice)
# ═════════════════════════════════════════════════════════════════════════════

def test_iter_session_spectra_order_and_labels(nominal):
    labels = [label for label, _sp in _iter_session_spectra(nominal.session)]
    assert labels == (
        ["bare"] + [f"bare#{i}" for i in range(3)]
        + ["probe"] + [f"probe#{i}" for i in range(3)]
        + ["1.00e-09"] + [f"1.00e-09#{i}" for i in range(3)]
        + ["1.00e-08"] + [f"1.00e-08#{i}" for i in range(3)]
    )
    assert list(_iter_session_spectra(EISSession())) == []


@pytest.mark.parametrize("label", ["probe", "probe#2", "1.00e-09", "1.00e-09#1"])
def test_recompute_drt_resolves_a_spectrum_by_session_label(with_fake_drt, fake_drt, label):
    s = with_fake_drt.session
    expected = dict(_iter_session_spectra(s))[label]

    fr = recompute_drt(s, label, with_fake_drt.cfg)

    assert _resolve_spectrum(s, label) is expected
    assert fake_drt.calls == [(expected.label, "sample")]
    assert expected.fit_results[DRT_MODEL_NAME] is fr


def test_recompute_drt_on_a_group_label_now_reaches_group_fit_results(with_fake_drt, fake_drt):
    """CORRIGÉ (était « COMPORTEMENT ACTUEL BOGUÉ ») : le recalcul sur un groupe écrivait
    dans ``group.spectrum.fit_results`` sans atteindre ``group.fit_results`` ; les deux
    copies divergeaient. Ce n'est plus qu'un seul dictionnaire."""
    group = with_fake_drt.session.groups[0]
    fr = recompute_drt(with_fake_drt.session, "1.00e-09", with_fake_drt.cfg, mode="optimize")
    assert group.fit_results[DRT_MODEL_NAME] is fr is group.spectrum.fit_results[DRT_MODEL_NAME]


def test_recompute_drt_on_a_replicate_refreshes_the_group_aggregate(fake_drt):
    session, _ = run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                              make_config(mode="optimize"))
    an = session.probe_analysis
    before = an.drt_target
    rep = session.probe_replicate_spectra[1]

    fr = recompute_drt(session, rep, session.config, mode="sample")

    assert rep.fit_results[DRT_MODEL_NAME] is fr and fr.drt_mode == "sample"
    assert an.drt_target is not before and an.drt_target.n == 3
    assert any("modes DRT différents" in w for w in an.warnings)
    for r in session.probe_replicate_spectra:                       # tous en 'sample'
        recompute_drt(session, r, session.config, mode="sample")
    assert not any("modes DRT différents" in w for w in an.warnings)
    assert an.drt_target.std_within == pytest.approx(5.0)          # a posteriori disponible


def test_recompute_drt_clears_a_recorded_drt_failure(fake_drt):
    fake_drt.fail = {"probe_r0.txt": RuntimeError("CmdStan a échoué")}
    session, _ = run_pipeline(replicate_assignments("probe", 0.0, _RCT_PROBE, 3, 0, "probe"),
                              make_config(mode="optimize"))
    an = session.probe_analysis
    assert "probe_r0.txt" in an.drt_failures and an.drt_target.n == 2
    fake_drt.fail = {}
    recompute_drt(session, "probe#0", session.config, mode="optimize")
    assert an.drt_failures == {} and an.drt_target.n == 3
    assert not any(w.startswith("DRT de « probe_r0.txt »") for w in an.warnings)


def test_recompute_drt_accepts_a_spectrum_object_even_outside_the_session(with_fake_drt, fake_drt):
    foreign = EISSpectrum(label="hors session", f=np.logspace(5, -1, 10), Zre=np.ones(10),
                          Zim=np.ones(10), concentration=0.0, step="probe", n_points=10)
    assert _resolve_spectrum(with_fake_drt.session, foreign) is foreign
    fr = recompute_drt(with_fake_drt.session, foreign, with_fake_drt.cfg)
    assert foreign.fit_results[DRT_MODEL_NAME] is fr


def test_recompute_drt_unknown_spectrum_or_mode_raises(with_fake_drt, fake_drt):
    with pytest.raises(ValueError, match="Spectre introuvable dans la session : 'probe#9'"):
        recompute_drt(with_fake_drt.session, "probe#9", with_fake_drt.cfg)
    with pytest.raises(InvalidAnalysisInput):
        recompute_drt(with_fake_drt.session, "probe", with_fake_drt.cfg, mode="nuts")
    assert fake_drt.calls == []


def test_recompute_drt_forces_the_mode_without_mutating_the_config(with_fake_drt, fake_drt):
    cfg = with_fake_drt.cfg
    assert cfg["fit"]["drt"]["mode"] == "optimize"
    fr = recompute_drt(with_fake_drt.session, "probe", cfg)          # défaut = 'sample'
    assert fake_drt.calls[-1][1] == "sample" and fr.drt_mode == "sample"
    recompute_drt(with_fake_drt.session, "probe", None, mode="optimize")   # config absente tolérée
    assert fake_drt.calls[-1][1] == "optimize"
    assert cfg["fit"]["drt"]["mode"] == "optimize"


def test_recompute_drt_propagates_engine_failures_to_the_caller(with_fake_drt, fake_drt):
    fake_drt.fail = {"probe_r0.txt (avg)": RuntimeError("CmdStan a échoué")}
    with pytest.raises(RuntimeError, match="CmdStan"):
        recompute_drt(with_fake_drt.session, "probe", with_fake_drt.cfg)


# ═════════════════════════════════════════════════════════════════════════════
# F. Tables de résultats (UI + export) : intra-fit et inter-réplicats côte à côte
# ═════════════════════════════════════════════════════════════════════════════

def test_group_rows_put_intra_fit_and_inter_replicate_side_by_side(with_fake_drt):
    rows = group_rows({1: with_fake_drt.session})
    assert [r["group"] for r in rows] == ["Probe", "1.00e-09 M"]
    for r, (_l, _m, _reps, an) in zip(rows, _groups(with_fake_drt.session)):
        t = an.orazem.target
        assert r["status"] == GROUP_OK and r["kk_conform"] is True
        assert (r["fit_mean"], r["fit_std_between"], r["fit_std_within"], r["fit_sem"]) == (
            t.mean, t.std_between, t.std_within, t.sem)
        assert r["fit_mean_spectrum"] == an.orazem.mean_fit.target_value
        assert r["drt_mode"] == "optimize" and r["drt_n_used"] == 3
        assert math.isnan(r["drt_std_within"]) and r["drt_std_between"] > 0


def test_replicate_rows_carry_hmc_diagnostics_only_in_sample_mode(with_fake_drt, fake_drt):
    s = with_fake_drt.session
    for r in s.probe_replicate_spectra:
        recompute_drt(s, r, with_fake_drt.cfg, mode="sample")
    rows = [r for r in replicate_rows({1: s}) if r["group"] == "Probe"]
    assert [r["kind"] for r in rows] == ["réplicat"] * 3 + ["moyenne"]
    for r in rows[:3]:
        assert (r["drt_mode"], r["drt_rhat_max"], r["drt_divergences"]) == ("sample", 1.002, 0)
        assert r["drt_ess_bulk_min"] == 900.0 and r["drt_Rct_std_intra"] == 5.0
        assert r["fit_value"] > 0 and r["fit_std_intra"] > 0
        assert r["fit_chi2_ci_low"] < r["fit_chi2_ci_high"]
    mean_row = rows[3]
    assert mean_row["drt_mode"] == "optimize" and mean_row["drt_rhat_max"] is None   # MAP : n/a
    assert drt_hmc_summary(s.probe.fit_results[DRT_MODEL_NAME])["divergences"] is None


def test_group_rows_of_a_stopped_group_carry_the_status_and_message():
    session, _ = run_pipeline(_single_file_assignments(), make_config(**_NO_DRT))
    rows = group_rows({1: session})
    assert {r["status"] for r in rows} == {GROUP_ERROR_STRUCTURE_UNAVAILABLE}
    assert all("structure d'erreur non caractérisable" in r["message"] for r in rows)
    assert all("fit_mean" not in r for r in rows)


# ═════════════════════════════════════════════════════════════════════════════
# G. DRT réelle de bout en bout — sautée sans CmdStan, exécutée par le job CI « drt »
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def drt_run():
    """Scénario de l'Annexe A.2 (60 points, Randles bruités à 0,5 %), DRT MAP réelle."""
    if not _HAVE_CMDSTAN:
        pytest.skip(f"DRT réelle : {_WHY_NOT}")
    n = N_POINTS_AUDIT
    fa = (replicate_assignments("probe", 0.0, 3000.0, 3, 0, "probe", n_points=n)
          + replicate_assignments("hybridization", 1e-9, 3500.0, 3, 10, "c1", n_points=n))
    cfg = make_config(enabled=True, mode="optimize")
    session, val = run_pipeline(fa, cfg)
    return SimpleNamespace(session=session, validation=val, cfg=cfg)


@_NEEDS_CMDSTAN
def test_real_drt_map_is_correct_on_every_replicate_and_mean(drt_run):
    """CORRIGÉ (était « COMPORTEMENT ACTUEL BOGUÉ — DRT-1/DRT-2 ») : avec les réglages
    amont, le MAP rendait Rct < 0 et Rp < 0 sur ces spectres, ``converged=True`` codé en
    dur et aucune alerte. Le moteur drt/engine.py (nonneg, init ridge) rend des valeurs
    correctes, sur chaque réplicat ET sur la moyenne."""
    truths = [3000.0, 3500.0]
    for (_lbl, mean_sp, reps, an), rct in zip(_groups(drt_run.session), truths):
        for sp in reps + [mean_sp]:
            fr = sp.fit_results[DRT_MODEL_NAME]
            assert fr.drt_mode == "optimize" and fr.converged is True
            assert fr.params["Rp"] > 0 and fr.target_value > 0
            assert fr.target_value == pytest.approx(rct, rel=0.08)
            assert np.all(fr.drt_gamma >= 0)
        assert an.drt_failures == {}
        assert an.drt_target.n == 3 and an.drt_target.mean == pytest.approx(rct, rel=0.08)
        assert math.isnan(an.drt_target.std_within)                    # MAP : non calculé


@_NEEDS_CMDSTAN
def test_real_recompute_drt_optimize_is_deterministic(drt_run):
    s, cfg = drt_run.session, drt_run.cfg
    previous = s.probe.fit_results[DRT_MODEL_NAME]
    fr = recompute_drt(s, "probe", cfg, mode="optimize")
    assert fr is not previous and s.probe.fit_results[DRT_MODEL_NAME] is fr
    assert fr.target_value == pytest.approx(previous.target_value, rel=1e-6)   # même graine


@_NEEDS_CMDSTAN
@pytest.mark.slow
def test_real_recompute_drt_sample_on_a_replicate_exposes_hmc_diagnostics(drt_run):
    """HMC (2 à 5 min) sur UN réplicat : intervalles, R-hat/divergences lus et visibles
    dans les tables, agrégat du groupe recalculé (et signalé : modes mélangés)."""
    s, cfg = drt_run.session, drt_run.cfg
    rep = s.probe_replicate_spectra[0]
    fr = recompute_drt(s, rep, cfg, mode="sample")
    assert fr.drt_mode == "sample" and fr.drt_gamma_lo is not None
    h = drt_hmc_summary(fr)
    assert h["rhat_max"] is not None and h["divergences"] is not None
    assert np.isfinite(fr.target_std) and fr.target_std > 0
    row = next(r for r in replicate_rows({1: s}) if r["spectrum"] == rep.label)
    assert row["drt_rhat_max"] == h["rhat_max"] and row["drt_divergences"] == h["divergences"]
    assert any("modes DRT différents" in w for w in s.probe_analysis.warnings)


# ═════════════════════════════════════════════════════════════════════════════
# H. Propagation jusqu'à l'UI (Streamlit AppTest, page EIS réelle)
# ═════════════════════════════════════════════════════════════════════════════

def _experiment_clean(n_rep):
    import io
    probe = [io.BytesIO(randles_file(_RCT_PROBE, seed=k)) for k in range(n_rep)]
    c1 = [io.BytesIO(randles_file(_RCT_C1, seed=10 + k)) for k in range(n_rep)]
    return {"mode": "eis_only", "n_electrodes": 1, "concentrations": [_C1],
            "probe": {"eis": {"electrode_1": probe}}, "calibration": {"eis": {"electrode_1": [c1]}}}


def _run_eis_page(monkeypatch, experiment):
    import streamlit
    from streamlit.testing.v1 import AppTest

    # Une page lancée seule (sans st.navigation) ne peut pas résoudre st.page_link.
    monkeypatch.setattr(streamlit, "page_link", lambda *a, **k: None)
    at = AppTest.from_file(str(REPO / "pages" / "A_eis.py"), default_timeout=300)
    at.session_state["preprocessing_done"] = True
    at.session_state["experiment_clean"] = experiment
    at.session_state["eis_drt_enabled"] = False                # DRT décochée : test rapide
    at.run()
    return at


def test_ui_err1_fixed_the_eis_page_shows_the_stopped_groups_and_no_success(monkeypatch):
    """CORRIGÉ (ERR-1, côté UI) : l'ancienne page affichait « ✅ Analyse terminée — 2
    groupe(s) » sur des fits vides. Elle affiche maintenant le message de CHAQUE groupe
    arrêté et un bilan en erreur, jamais le succès."""
    at = _run_eis_page(monkeypatch, _experiment_clean(n_rep=1))
    assert not at.exception, [e.value for e in at.exception]
    errors = [e.value for e in at.error]
    assert sum("structure d'erreur non caractérisable" in e for e in errors) >= 2
    assert any("2 groupe(s) ARRÊTÉ(S)" in e for e in errors)
    assert not any("Analyse terminée —" in s.value for s in at.success)


def test_ui_a_complete_analysis_reports_success(monkeypatch):
    at = _run_eis_page(monkeypatch, _experiment_clean(n_rep=3))
    assert not at.exception, [e.value for e in at.exception]
    assert any("Analyse terminée — 2 groupe(s) analysé(s)" in s.value for s in at.success)
    sessions = at.session_state["eis_sessions"]
    assert sessions[1].probe_analysis.ok and ORAZEM_MODEL_NAME in sessions[1].probe.fit_results


def test_ui_a_software_bug_is_shown_as_such_not_as_a_data_problem(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bug simulé")

    monkeypatch.setattr(pipeline, "fit_replicate_group", boom)
    at = _run_eis_page(monkeypatch, _experiment_clean(n_rep=3))
    assert any("Électrode 1 non analysée" in w.value and "erreur LOGICIELLE" in w.value
               for w in at.warning)
    assert any("bug simulé" in str(e.value) for e in at.exception)     # trace affichée (st.exception)
    assert "eis_sessions" not in at.session_state                       # rien de faux n'est stocké


def test_ui_a_failing_electrode_does_not_lose_the_others(monkeypatch):
    """Isolation par électrode : un bug sur l'électrode 2 laisse l'électrode 1 intacte,
    et l'avertissement porte le numéro de l'électrode fautive."""
    experiment = _experiment_clean(n_rep=3)
    experiment["n_electrodes"] = 2
    experiment["probe"]["eis"]["electrode_2"] = experiment["probe"]["eis"]["electrode_1"]
    experiment["calibration"]["eis"]["electrode_2"] = experiment["calibration"]["eis"]["electrode_1"]
    real = pipeline.run_pipeline
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:                       # 2ᵉ électrode analysée
            raise RuntimeError("bug simulé électrode 2")
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "run_pipeline", flaky)
    at = _run_eis_page(monkeypatch, experiment)
    assert any("Électrode 2 non analysée" in w.value for w in at.warning)
    assert not any("Électrode 1 non analysée" in w.value for w in at.warning)
    sessions = at.session_state["eis_sessions"]
    assert set(sessions) == {1}
