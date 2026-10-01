"""Tests de CARACTÉRISATION de core/validator.py (filet de sécurité avant refonte).

Figent le comportement RÉEL du validateur KK et de la détection de dérive, défauts
compris (voir l'en-tête de tests/test_pipeline.py pour la philosophie). Défauts
encore figés, chacun signalé par « COMPORTEMENT ACTUEL BOGUÉ » :

  * ERR-4 (AUDIT.md) — ``_detect_drift`` : faux positifs sur des réplicats
    stationnaires ;
  * deux observations SUPPLÉMENTAIRES, non listées dans l'audit : avertissement de
    non-chevauchement écrasé, listes de longueurs différentes tronquées sans erreur.

CORRIGÉS à l'étape 4 (tests mis à jour en même temps que le correctif) :
  * ERR-5 — ``linKK`` ignorait ``c`` : wrapper supprimé, M choisi par le critère µ
    de Schönleber (voir tests/test_measurement_model.py) ;
  * ERR-6 — deux critères KK sans source (résidu < 2 % + paliers 10/25 %, et
    ``drt_kk_tol``) : remplacés par le critère UNIQUE ``kk_verdict`` appliqué au test
    par measurement model pondéré par la structure d'erreur du groupe ;
  * AUDIT.md §2.4 — plancher de σ à 0,1 % de |Z| (``_compute_sigma``) : supprimé, σ
    vient de la structure d'erreur ;
  * spectre vide : ne fait plus planter le gestionnaire d'erreur ; spectres de 1-2
    points : refusés au lieu d'être déclarés valides ; « Lin-KK échoué » n'avale
    plus n'importe quelle exception.

Les fabriques de spectres viennent de tests/synthetic_data.py (style Annexe A).
"""

import logging

import numpy as np
import pytest

from core import validator as V
from core.measurement_model import ErrorStructureUnavailable, characterize_error_structure
from core.validator import (
    DRIFT_CV_THRESHOLD,
    KKResult,
    ValidationResult,
    _detect_drift,
    _find_valid_range,
    validate_replicate_group,
    validate_spectrum,
)
from tests.synthetic_data import N_POINTS, noisy_arrays, orazem_noisy_arrays

_RCT = 3000.0


# ─────────────────────────────────────────────────────────────────────────────
# Outils
# ─────────────────────────────────────────────────────────────────────────────

def _clean():
    """(f, Zre, Zim) sans bruit, ordre HF → BF."""
    return noisy_arrays(_RCT, noise=0.0)


def _orazem(seed, drift=0.0):
    """(f, Zre, Zim) bruités selon la structure d'Orazem (σ_r = σ_j)."""
    return orazem_noisy_arrays(_RCT, seed, drift=drift)[:3]


def _kk(lo, hi, valid=True, res_im=None, freqs=None):
    """KKResult fabriqué à la main (résidus nuls par défaut)."""
    fr = np.logspace(-1, 5, 30) if freqs is None else freqs
    return KKResult(
        label="k", frequencies=fr, residuals_re=np.zeros(len(fr)),
        residuals_im=np.zeros(len(fr)) if res_im is None else res_im,
        method="measurement_model", n_elements=3, chi2_reduced=1.0,
        f_min_valid=lo, f_max_valid=hi, is_valid=valid,
    )


def _group(reps):
    """reps : liste de (f, zre, zim) → arguments de validate_replicate_group."""
    return [r[0] for r in reps], [r[1] for r in reps], [r[2] for r in reps]


@pytest.fixture(scope="module")
def conform_group():
    """3 réplicats stationnaires, bruit d'Orazem : le cas nominal (calculé une fois)."""
    reps = [_orazem(seed) for seed in (0, 1, 2)]
    return reps, validate_replicate_group(*_group(reps), label="nominal")


# ═════════════════════════════════════════════════════════════════════════════
# validate_replicate_group — verdict par measurement model
# ═════════════════════════════════════════════════════════════════════════════

def test_module_keeps_only_the_drift_threshold():
    """Les seuils KK sans source ont disparu (ERR-6) : seul reste celui de la dérive."""
    assert DRIFT_CV_THRESHOLD == 0.5
    for gone in ("MU_THRESHOLD", "RESIDUAL_THRESHOLD_PCT", "INVALID_FRACTION_WARN",
                 "INVALID_FRACTION_REJECT", "_compute_sigma"):
        assert not hasattr(V, gone), gone


def test_stationary_group_is_conform_by_the_measurement_model(conform_group):
    _reps, vr = conform_group
    assert isinstance(vr, ValidationResult)
    assert vr.all_valid is True
    assert vr.error_structure_message is None
    assert "conforme Kramers-Kronig" in vr.kk_message and "peut être proposé" in vr.kk_message
    assert vr.measurement_model is not None and vr.measurement_model.kk_conform is True
    assert [r.label for r in vr.replicates] == ["nominal_rep1", "nominal_rep2", "nominal_rep3"]
    for kk in vr.replicates:
        assert kk.method == "measurement_model"
        assert kk.is_valid is True and kk.warning is None
        assert kk.n_elements >= 3 and np.isfinite(kk.chi2_reduced)
        assert np.all(np.diff(kk.frequencies) > 0)                 # fréquences CROISSANTES
        assert len(kk.residuals_re) == len(kk.band_re) == N_POINTS
        assert np.all(kk.band_re > 0) and np.all(kk.band_im > 0)
        assert kk.n_outside <= kk.n_allowed
    assert vr.f_min_common < vr.f_max_common


def test_sigma_comes_from_the_error_structure_without_any_floor(conform_group):
    """AUDIT.md §2.4 corrigé : σ = structure d'erreur évaluée sur le spectre moyen,
    en fréquences croissantes — plus de plancher à 0,1 % de |Z|."""
    _reps, vr = conform_group
    mm = vr.measurement_model
    s_re, s_im = mm.error_structure.sigmas(mm.mean_Zre, mm.mean_Zim)
    np.testing.assert_allclose(vr.sigma_re, s_re)
    np.testing.assert_allclose(vr.sigma_im, s_im)
    assert np.all(np.diff(mm.frequencies) > 0)
    # Le vrai bruit (générateur) est retrouvé à ~15 % près en moyenne.
    f, zre, zim, true_sigma = orazem_noisy_arrays(_RCT, 0)
    true_sigma = true_sigma[::-1]                      # ordre croissant, comme le validateur
    assert np.mean(vr.sigma_re / true_sigma) == pytest.approx(1.0, abs=0.15)


def test_a_replicate_drifting_during_its_sweep_makes_the_group_non_conform():
    """Rct +20 % pendant le balayage d'UN réplicat : spectre non stationnaire, donc
    non conforme KK — c'est ce que le test doit voir, sur ce réplicat."""
    reps = [_orazem(10), _orazem(11, drift=0.20), _orazem(12)]
    vr = validate_replicate_group(*_group(reps), label="derive")
    assert vr.all_valid is False
    assert "NON conforme" in vr.kk_message
    assert vr.replicates[1].is_valid is False
    assert "NON conforme Kramers-Kronig" in vr.replicates[1].warning
    assert vr.replicates[1].n_outside > vr.replicates[1].n_allowed


@pytest.mark.parametrize("n_rep", [1, 2])
def test_fewer_than_three_replicates_give_an_undetermined_verdict(n_rep):
    """Pas de structure d'erreur → PAS de verdict (ni vert, ni rouge) : les résidus
    Lin-KK sont montrés à titre indicatif et le motif est donné à l'utilisateur."""
    reps = [_orazem(seed) for seed in range(n_rep)]
    vr = validate_replicate_group(*_group(reps), label="peu")
    assert vr.all_valid is None
    assert vr.measurement_model is None
    assert f"{n_rep} réplicat(s) fourni(s), 3 requis" in vr.error_structure_message
    assert "Aucun fit n'a été réalisé" in vr.error_structure_message
    assert vr.sigma_re is None and vr.sigma_im is None
    for kk in vr.replicates:
        assert kk.method == "lin_kk" and kk.is_valid is None and kk.band_re is None
        assert "indéterminé" in kk.warning


def test_identical_replicates_are_refused_instead_of_floored():
    """Trois copies du même fichier : aucune dispersion, donc aucun bruit mesurable.
    L'ancien validateur rendait σ = plancher arbitraire ; désormais : refus motivé."""
    reps = [_orazem(0)] * 3
    vr = validate_replicate_group(*_group(reps), label="copies")
    assert vr.all_valid is None
    assert "dispersion inter-réplicats nulle" in vr.error_structure_message
    assert vr.sigma_re is None


def test_empty_group_is_invalid_with_a_message():
    vr = validate_replicate_group([], [], [], label="x")
    assert isinstance(vr, ValidationResult)
    assert vr.replicates == []
    assert vr.all_valid is False
    assert vr.drift_warning == "Aucun réplicat fourni."
    assert vr.drift_detected is False
    assert vr.f_min_common == 0.0 and vr.f_max_common == np.inf
    assert vr.sigma_re is None and vr.sigma_im is None


def test_the_group_verdict_uses_the_configured_minimum_of_replicates():
    from core.measurement_model import MeasurementModelOptions

    reps = [_orazem(seed) for seed in range(3)]
    vr = validate_replicate_group(*_group(reps), label="g",
                                  options=MeasurementModelOptions(min_replicates=4))
    assert vr.all_valid is None and "4 requis" in vr.error_structure_message


# ═════════════════════════════════════════════════════════════════════════════
# validate_spectrum — un spectre seul
# ═════════════════════════════════════════════════════════════════════════════

def test_single_spectrum_without_error_structure_shows_lin_kk_residuals_only():
    f, zre, zim = _clean()
    kk = validate_spectrum(f, zre, zim, label="seul")
    assert kk.label == "seul"
    assert kk.method == "lin_kk" and kk.is_valid is None
    assert "indéterminé" in kk.warning
    assert kk.n_elements >= 20                           # M de Schönleber (dernier franchissement)
    assert np.abs(kk.residuals_re).max() < 0.05          # % : spectre exact
    assert np.all(np.diff(kk.frequencies) > 0)
    assert (kk.f_min_valid, kk.f_max_valid) == (pytest.approx(0.1), pytest.approx(1e5))


def test_single_spectrum_with_an_error_structure_gets_a_verdict():
    reps = [orazem_noisy_arrays(_RCT, s) for s in (20, 21, 22)]
    from types import SimpleNamespace
    est = characterize_error_structure(
        [SimpleNamespace(f=a[0], Zre=a[1], Zim=a[2], label=str(i)) for i, a in enumerate(reps)])
    f, zre, zim, _ = orazem_noisy_arrays(_RCT, 23)
    ok = validate_spectrum(f, zre, zim, label="conforme", error_structure=est.error_structure)
    assert ok.method == "measurement_model" and ok.is_valid is True

    f, zre, zim, _ = orazem_noisy_arrays(_RCT, 24, drift=0.3)
    bad = validate_spectrum(f, zre, zim, label="derive", error_structure=est.error_structure)
    assert bad.is_valid is False and "NON conforme" in bad.warning


def test_input_order_does_not_matter():
    f, zre, zim = _clean()
    ref = validate_spectrum(f, zre, zim)
    idx = np.random.default_rng(5).permutation(N_POINTS)
    shuffled = validate_spectrum(f[idx], zre[idx], zim[idx])
    assert shuffled.n_elements == ref.n_elements
    np.testing.assert_allclose(shuffled.residuals_re, ref.residuals_re)
    np.testing.assert_allclose(shuffled.residuals_im, ref.residuals_im)


def test_nan_in_the_data_is_reported_as_invalid():
    f, zre, zim = _clean()
    zre = zre.copy()
    zre[3] = np.nan
    kk = validate_spectrum(f, zre, zim)
    assert kk.is_valid is False
    assert kk.warning.startswith("données inexploitables pour le test KK")


def test_empty_input_is_invalid_instead_of_crashing():
    """Correctif (observation non listée dans l'audit) : le gestionnaire d'erreur
    lisait ``f_sorted[0]`` sur un tableau vide et levait IndexError."""
    empty = np.array([])
    kk = validate_spectrum(empty, empty, empty)
    assert kk.is_valid is False and len(kk.frequencies) == 0


@pytest.mark.parametrize("n", [1, 2])
def test_one_or_two_points_are_refused(n):
    """Correctif : 1 ou 2 points étaient déclarés « valides » (un Voigt les reproduit
    forcément). Lin-KK exige désormais au moins 3 points."""
    f, zre, zim = _clean()
    kk = validate_spectrum(f[:n], zre[:n], zim[:n])
    assert kk.is_valid is False and "au moins 3 points" in kk.warning


def test_a_programming_error_in_lin_kk_is_no_longer_swallowed(monkeypatch):
    """L'ancien ``except Exception`` transformait tout bug en « lin-KK échoué »."""
    def boom(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr("core.validator.lin_kk", boom)
    f, zre, zim = _clean()
    with pytest.raises(RuntimeError, match="bug"):
        validate_spectrum(f, zre, zim)


# ═════════════════════════════════════════════════════════════════════════════
# _find_valid_range
# ═════════════════════════════════════════════════════════════════════════════

def test_find_valid_range_returns_the_longest_contiguous_run():
    f = np.arange(1.0, 9.0)                                 # 1..8, ordre croissant (comme validate_spectrum)
    mask = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)   # runs de 2, 3 et 1 points
    assert _find_valid_range(f, mask) == (4.0, 6.0)


def test_find_valid_range_first_run_wins_a_tie():
    f = np.arange(1.0, 7.0)
    mask = np.array([1, 1, 0, 1, 1, 0], dtype=bool)
    assert _find_valid_range(f, mask) == (1.0, 2.0)


def test_find_valid_range_single_valid_point_and_degenerate_masks():
    f = np.arange(1.0, 5.0)
    assert _find_valid_range(f, np.array([0, 0, 1, 0], dtype=bool)) == (3.0, 3.0)
    assert _find_valid_range(f, np.ones(4, dtype=bool)) == (1.0, 4.0)
    assert _find_valid_range(f, np.zeros(4, dtype=bool)) == (1.0, 4.0)   # aucun valide → plage complète


# ═════════════════════════════════════════════════════════════════════════════
# validate_replicate_group — logique d'agrégation (chemin sans structure d'erreur)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def no_error_structure(monkeypatch):
    """Force le chemin « structure d'erreur non caractérisable » : les KKResult sont
    alors produits par ``validate_spectrum`` (remplaçable par un faux)."""
    def refuse(reps, **kwargs):
        raise ErrorStructureUnavailable("forcé par le test", group_label=kwargs.get("label", ""))

    monkeypatch.setattr("core.validator.analyze_replicates", refuse)


def test_single_replicate_group_has_no_verdict_no_drift_check_and_no_sigma():
    f, zre, zim = _clean()
    vr = validate_replicate_group([f], [zre], [zim], label="one")

    assert [r.label for r in vr.replicates] == ["one_rep1"]
    assert vr.all_valid is None                                   # indéterminé, pas « valide »
    assert vr.drift_detected is False and vr.drift_warning is None
    assert vr.sigma_re is None and vr.sigma_im is None
    assert (vr.f_min_common, vr.f_max_common) == (pytest.approx(0.1), pytest.approx(1e5))


def test_common_range_is_the_intersection_of_replicate_ranges(monkeypatch, no_error_structure):
    results = iter([_kk(1e-1, 1e4), _kk(1e0, 1e5), _kk(1e-2, 1e3)])
    monkeypatch.setattr(V, "validate_spectrum", lambda *a, **k: next(results))
    z = [np.ones(30)] * 3

    vr = validate_replicate_group([np.logspace(-1, 5, 30)] * 3, z, z, label="g")

    assert vr.f_min_common == 1e0          # max des bornes basses
    assert vr.f_max_common == 1e3          # min des bornes hautes
    assert vr.all_valid is None            # sans structure d'erreur : pas de verdict
    assert vr.error_structure_message.startswith("Groupe « g » : analyse Orazem interrompue")


def test_one_unusable_replicate_makes_the_group_invalid(monkeypatch, no_error_structure):
    results = iter([_kk(1e-1, 1e5), _kk(1e-1, 1e5, valid=False)])
    monkeypatch.setattr(V, "validate_spectrum", lambda *a, **k: next(results))
    z = [np.ones(30)] * 2

    vr = validate_replicate_group([np.logspace(-1, 5, 30)] * 2, z, z, label="g")

    assert vr.all_valid is False
    assert [r.is_valid for r in vr.replicates] == [True, False]


def test_disjoint_ranges_invalidate_the_group_but_the_warning_is_overwritten(monkeypatch,
                                                                             no_error_structure):
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit).

    Quand les plages KK-valides des réplicats ne se chevauchent pas, le code pose
    ``drift_warning = "…plages KK-valides…ne se chevauchent pas…"`` puis, dès qu'il y a
    ≥ 2 réplicats, ré-affecte INCONDITIONNELLEMENT ``drift_warning`` avec le résultat de
    ``_detect_drift``. Le message de non-chevauchement est donc perdu : le groupe est
    invalide (``all_valid=False``) sans aucun message pour l'expliquer.
    """
    results = iter([_kk(1e-1, 1e1), _kk(1e3, 1e5)])
    monkeypatch.setattr(V, "validate_spectrum", lambda *a, **k: next(results))
    z = [np.ones(30)] * 2

    vr = validate_replicate_group([np.logspace(-1, 5, 30)] * 2, z, z, label="disjoint")

    assert vr.f_min_common == 1e3 and vr.f_max_common == 1e1     # bornes croisées
    assert vr.all_valid is False
    assert vr.drift_detected is False
    assert vr.drift_warning is None                              # message écrasé


def test_drift_threshold_is_forwarded_to_the_drift_check():
    reps = [noisy_arrays(_RCT, 0.005, seed) for seed in (0, 1, 2)]
    assert validate_replicate_group(*_group(reps), label="g").drift_detected is True
    assert validate_replicate_group(*_group(reps), label="g", drift_cv_threshold=100.0).drift_detected is False


def test_mismatched_list_lengths_are_silently_truncated():
    """COMPORTEMENT ACTUEL BOGUÉ (observation non listée dans l'audit) : ``zip`` tronque
    sans erreur — 3 fréquences mais 2 jeux de Z → 2 réplicats validés, le 3ᵉ ignoré."""
    reps = [_clean()] * 3
    f_list, zre_list, zim_list = _group(reps)
    vr = validate_replicate_group(f_list, zre_list[:2], zim_list[:2], label="g")
    assert len(vr.replicates) == 2


# ═════════════════════════════════════════════════════════════════════════════
# _detect_drift
# ═════════════════════════════════════════════════════════════════════════════

def test_stationary_replicates_are_flagged_as_drifting(caplog):
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-4, à corriger à l'étape 5.

    Sur des réplicats STATIONNAIRES (même spectre, bruit gaussien indépendant de
    0,5 %), ``_detect_drift`` annonce une dérive sur > 60 % des fréquences, à chaque
    fois : le critère CV = σ/moyenne des résidus KK inter-réplicats vaut ~1 pour du
    bruit blanc, donc dépasse toujours le seuil 0,5. L'UI l'affiche en rouge.
    """
    for base_seed in (0, 100, 200, 300, 400):
        reps = [noisy_arrays(_RCT, 0.005, base_seed + k) for k in range(3)]
        with caplog.at_level(logging.WARNING, logger="core.validator"):
            vr = validate_replicate_group(*_group(reps), label="stationnaire")
        # Stationnaires par construction ; le verdict KK (measurement model) le confirme…
        assert vr.all_valid is True
        assert vr.drift_detected is True                  # …et pourtant « dérive »
        assert vr.drift_warning.startswith("Drift détecté sur ")
        assert "(CV résidus KK > 0.5)" in vr.drift_warning
        assert "entre les 3 réplicats" in vr.drift_warning
        pct = int(vr.drift_warning.split("sur ")[1].split("%")[0])
        assert pct > 20                                    # au-delà du seuil de 20 % des fréquences
    assert any("Drift détecté" in r.getMessage() for r in caplog.records)


def test_real_differences_between_replicates_are_not_detected():
    """COMPORTEMENT ACTUEL BOGUÉ — cf AUDIT.md ERR-4 (faux négatif), à corriger à l'étape 5.

    Le critère mesure la dispersion des RÉSIDUS KK, pas celle des spectres : deux
    réplicats sans bruit dont le Rct diffère d'un facteur 2 ont chacun un résidu KK
    minuscule (sous le garde ``mean_res > 0.1``), donc aucune dérive n'est signalée.

    Le verdict KK n'est pas l'objet de ce test : sans bruit, la « structure d'erreur »
    caractérisée n'est que l'erreur de troncature du measurement model (hors de la
    prémisse bruit ≫ troncature), et le verdict qui en découle n'a pas de sens.
    """
    a = noisy_arrays(_RCT, 0.0)
    b = noisy_arrays(2 * _RCT, 0.0)
    vr = validate_replicate_group(*_group([a, a, b]), label="deux_rct")

    assert vr.drift_detected is False and vr.drift_warning is None


def test_low_noise_replicates_are_not_flagged():
    """Le garde ``mean_res > 0.1`` (en %) neutralise le CV quand les résidus sont
    minuscules : à 0,1 % de bruit, aucune dérive — le faux positif est propre au
    bruit d'au moins ~0,3 %."""
    reps = [noisy_arrays(_RCT, 0.001, k) for k in range(3)]
    vr = validate_replicate_group(*_group(reps), label="bas_bruit")
    assert vr.drift_detected is False and vr.drift_warning is None


def test_white_noise_residuals_always_exceed_the_cv_threshold():
    rng = np.random.default_rng(0)
    kks = [_kk(0, 0, res_im=rng.normal(0, 1, 30)) for _ in range(3)]
    drifted, msg = _detect_drift(kks, 0.5)
    assert drifted is True
    assert msg.startswith("Drift détecté sur ")


def test_drift_needs_strictly_more_than_twenty_percent_of_frequencies():
    res = [np.zeros(30) for _ in range(3)]
    for i in range(6):                         # 6 fréquences sur 30 = 20 % exactement
        res[0][i], res[1][i] = 2.0, -2.0
    assert _detect_drift([_kk(0, 0, res_im=r) for r in res], 0.5) == (False, None)

    res[0][6], res[1][6] = 2.0, -2.0           # 7 sur 30 = 23 % → dérive
    drifted, msg = _detect_drift([_kk(0, 0, res_im=r) for r in res], 0.5)
    assert drifted is True
    assert "sur 23% des fréquences" in msg


def test_residuals_below_a_tenth_of_a_percent_never_count_as_drift():
    kks = [_kk(0, 0, res_im=np.full(30, 0.05) * s) for s in (1.0, -1.0, 1.0)]
    assert _detect_drift(kks, 0.5) == (False, None)


def test_identical_constant_residuals_are_not_drift():
    kks = [_kk(0, 0, res_im=np.full(30, 2.0)) for _ in range(3)]
    assert _detect_drift(kks, 0.5) == (False, None)


def test_a_huge_cv_threshold_disables_the_detection():
    rng = np.random.default_rng(0)
    kks = [_kk(0, 0, res_im=rng.normal(0, 1, 30)) for _ in range(3)]
    assert _detect_drift(kks, 100.0) == (False, None)


def test_replicates_on_different_frequency_grids_are_interpolated():
    a = _kk(0, 0, freqs=np.logspace(-1, 5, 30))
    b = _kk(0, 0, freqs=np.logspace(-1, 5, 50))
    assert _detect_drift([a, b], 0.5) == (False, None)
