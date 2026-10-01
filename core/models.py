"""Data models for EIS Analyzer.

``FitResult`` est défini dans ``fits/result.py`` et ré-exporté ici : ``core`` dépend de
``fits``, jamais l'inverse (AUDIT.md CPL-1, voir ``core/pipeline.py``).
"""

from dataclasses import InitVar, dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from fits.result import FitResult

__all__ = [
    "EISSpectrum",
    "FitResult",
    "GroupAnalysis",
    "ConcentrationGroup",
    "EISSession",
    "DisplayGroup",
    "GROUP_OK",
    "GROUP_ERROR_STRUCTURE_UNAVAILABLE",
    "GROUP_INVALID_INPUT",
]


@dataclass
class EISSpectrum:
    """Single EIS spectrum with metadata.

    Un spectre est soit un RÉPLICAT BRUT (un fichier, ``replicates`` vide), soit la
    MOYENNE d'un groupe de réplicats : il porte alors la liste de ses réplicats bruts
    (``replicates``), conservés tels que chargés — chacun avec ses propres
    ``fit_results`` (fit Orazem et DRT par réplicat, remplis par core/pipeline.py).

    Attributes:
        label: Display name.
        f: Frequency array (Hz), sorted HF→BF.
        Zre: Real impedance (Ω).
        Zim: Imaginary impedance (Ω), positive convention (semicircle above real axis).
        concentration: Analyte concentration (mol/L); 0.0 for bare/probe.
        step: Measurement step: 'bare', 'probe', or 'hybridization'.
        n_points: Number of frequency points.
        source_files: Original filenames contributing to this spectrum.
        fit_results: Dict mapping model name to FitResult (populated by pipeline).
        replicates: réplicats BRUTS (EISSpectrum) dont ce spectre est la moyenne ;
            vide pour un réplicat brut.
    """

    label: str
    f: np.ndarray
    Zre: np.ndarray
    Zim: np.ndarray
    concentration: float
    step: str
    n_points: int
    source_files: list = field(default_factory=list)
    fit_results: dict = field(default_factory=dict)
    # Validation KK — renseigné par core/validator.py
    validation: Optional[object] = None       # ValidationResult (évite import circulaire)
    sigma_re: Optional[object] = None         # np.ndarray σ_re(f) inter-réplicats (BRUT, sans plancher)
    sigma_im: Optional[object] = None         # np.ndarray σ_im(f) inter-réplicats (BRUT, sans plancher)
    n_replicates: Optional[int] = None        # nb de réplicats moyennés
    replicates: list = field(default_factory=list)   # réplicats BRUTS (vide pour un réplicat)
    f_min_valid: Optional[float] = None       # Hz — borne basse KK-valide
    f_max_valid: Optional[float] = None       # Hz — borne haute KK-valide

    def __post_init__(self):
        self.n_points = len(self.f)
        if self.replicates is None:
            self.replicates = []


# ─────────────────────────────────────────────────────────────────────────────
# Analyse d'un groupe de réplicats (bare, probe ou une concentration)
# ─────────────────────────────────────────────────────────────────────────────

#: Groupe analysé jusqu'au bout (measurement model, verdict KK, fits).
GROUP_OK = "ok"
#: Structure d'erreur non caractérisable (AUDIT.md ERR-1) : l'analyse du groupe est
#: ARRÊTÉE avant tout fit ; ``GroupAnalysis.message`` dit pourquoi et comment y remédier.
GROUP_ERROR_STRUCTURE_UNAVAILABLE = "error_structure_unavailable"
#: Saisie incompatible avec les données de ce groupe (ex. plus de paramètres que
#: d'observations) : aucun fit n'est rendu pour le groupe.
GROUP_INVALID_INPUT = "invalid_input"


@dataclass
class GroupAnalysis:
    """Tout ce que le pipeline a produit pour UN groupe de réplicats, dans l'ordre.

    1. ``validation`` — measurement model + verdict Kramers-Kronig sur les réplicats
       BRUTS, calculés AVANT tout fit (``core.validator.ValidationResult``) ;
    2. ``orazem`` — fit Orazem de chaque réplicat et de la moyenne, et agrégation
       (``fits.orazem_fit.OrazemGroupResult`` : valeurs par réplicat, incertitude
       intra-fit, variabilité inter-réplicats) ;
    3. ``drt_target`` — Rct DRT agrégé sur les réplicats (``AggregatedParameter``) ; les
       DRT elles-mêmes sont dans ``fit_results['drt_bayes']`` de chaque spectre.

    Les ``FitResult`` individuels restent rangés dans ``EISSpectrum.fit_results`` (de
    chaque réplicat brut et du spectre moyen) ; ce conteneur porte le STATUT du groupe
    et les résultats AGRÉGÉS.

    Attributes:
        label: identifiant ('bare', 'probe', 'hyb_1.00e-09').
        status: ``GROUP_OK`` ou un statut d'erreur (aucun fit produit pour le groupe).
        message: phrase destinée à l'utilisateur quand ``status`` n'est pas OK.
        validation: ValidationResult (verdict KK, structure d'erreur).
        orazem: OrazemGroupResult, ou None.
        drt_target: AggregatedParameter du Rct DRT sur les réplicats, ou None.
        drt_failures: {label du spectre: motif} des DRT non calculées (donnée invalide,
            échec de CmdStan) — le reste du groupe est conservé.
        warnings: alertes non bloquantes du groupe.
    """

    label: str
    status: str = GROUP_OK
    message: Optional[str] = None
    validation: Optional[object] = None
    orazem: Optional[object] = None
    drt_target: Optional[object] = None
    drt_failures: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == GROUP_OK


@dataclass
class ConcentrationGroup:
    """A spectrum and its fit results, grouped by analyte concentration.

    ``fit_results`` est un ALIAS de ``spectrum.fit_results`` (même dictionnaire) : les
    fits du spectre moyen ne vivent qu'à un seul endroit, quel que soit le chemin
    d'accès (corrige l'asymétrie où un recalcul DRT écrivait dans
    ``group.spectrum.fit_results`` sans atteindre ``group.fit_results``). Le
    constructeur accepte toujours ``fit_results=`` : son contenu est versé dans
    ``spectrum.fit_results``.

    Attributes:
        concentration: mol/L.
        spectrum: spectre MOYEN du groupe (porte aussi ses réplicats bruts).
        replicate_spectra: réplicats BRUTS (les mêmes objets que ``spectrum.replicates``),
            chacun avec ses propres ``fit_results``.
        analysis: GroupAnalysis (statut, verdict KK, résultats agrégés).
    """

    concentration: float
    spectrum: EISSpectrum
    fit_results: InitVar[Optional[dict]] = None
    replicate_spectra: list = field(default_factory=list)
    analysis: Optional[GroupAnalysis] = None

    def __post_init__(self, fit_results):
        if fit_results:
            self.spectrum.fit_results.update(fit_results)


def _group_fit_results_get(self) -> dict:
    return self.spectrum.fit_results


def _group_fit_results_set(self, value) -> None:
    self.spectrum.fit_results = value


# Propriété posée après la création de la dataclass : le paramètre ``fit_results`` du
# constructeur (InitVar) reste disponible, et l'attribut devient un alias.
ConcentrationGroup.fit_results = property(_group_fit_results_get, _group_fit_results_set)


@dataclass
class DisplayGroup:
    """Un groupe de réplicats tel qu'AFFICHÉ par l'onglet « Visualisation ».

    Ni fit ni résultat : seulement des spectres. Les groupes BRUTS (avant les
    exclusions du prétraitement) sont construits par la page EIS après l'analyse et
    attachés à ``EISSession.raw_groups`` ; les groupes prétraités viennent de
    ``EISSession.iter_groups()`` (``EISSession.display_groups``).

    Attributes:
        label: libellé d'affichage ('Bare', 'Probe', '1.00e-09 M').
        concentration: mol/L (0.0 pour bare/probe).
        step: 'bare', 'probe' ou 'hybridization'.
        mean: spectre moyen, ou None si les réplicats ne se moyennent pas (grilles de
            fréquences différentes — cas des données brutes seulement).
        replicates: réplicats (EISSpectrum).
    """

    label: str
    concentration: float
    step: str
    mean: Optional[EISSpectrum]
    replicates: list = field(default_factory=list)


@dataclass
class EISSession:
    """Full analysis session state (one electrode), stored in st.session_state['eis_sessions'].

    Attributes:
        bare, probe: spectres MOYENS (chacun porte ses réplicats bruts).
        groups: ConcentrationGroup triés par concentration croissante.
        bare_replicate_spectra, probe_replicate_spectra: réplicats BRUTS de bare/probe.
        bare_analysis, probe_analysis: GroupAnalysis de bare/probe (statut, agrégats).
        circuit: description du circuit ajusté (expression, cible, guess/bornes).
        drt_mode: mode DRT demandé ('sample'/'optimize'), None si DRT non lancée.
        load_errors: fichiers écartés au chargement ({"filename", "message"}).
        messages: messages de niveau session (ex. DRT indisponible).
    """

    created_at: datetime = field(default_factory=datetime.now)
    bare: Optional[EISSpectrum] = None
    probe: Optional[EISSpectrum] = None
    groups: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    bare_replicate_spectra: list = field(default_factory=list)
    probe_replicate_spectra: list = field(default_factory=list)
    bare_analysis: Optional[GroupAnalysis] = None
    probe_analysis: Optional[GroupAnalysis] = None
    circuit: Optional[dict] = None
    drt_mode: Optional[str] = None
    load_errors: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    # Référence « électrode nue » — AFFICHAGE SEUL, JAMAIS utilisée dans les
    # calculs (ni fit, ni normalisation, ni calibration, ni export de
    # valeurs calculées). Champ dédié et séparé de `bare`/`probe`/`groups` afin
    # que le pipeline soit structurellement incapable de la lire : elle est
    # attachée à la session APRÈS l'analyse et seulement superposée au Nyquist.
    bare_reference: Optional[EISSpectrum] = None
    # Groupes BRUTS (``DisplayGroup``, avant exclusions du prétraitement) — AFFICHAGE
    # SEUL, même statut que ``bare_reference`` : attachés APRÈS l'analyse par la page EIS,
    # jamais lus par le pipeline. Vide si non renseignés.
    raw_groups: list = field(default_factory=list)

    def display_groups(self) -> list:
        """``DisplayGroup`` des spectres PRÉTRAITÉS analysés : bare, probe, concentrations."""
        out = []
        for label, mean_sp, reps, _an in self.iter_groups():
            out.append(DisplayGroup(label=label, concentration=float(mean_sp.concentration),
                                    step=mean_sp.step, mean=mean_sp, replicates=list(reps)))
        return out

    def iter_groups(self):
        """(libellé d'affichage, spectre moyen, réplicats bruts, GroupAnalysis) de chaque
        groupe présent : bare, probe, puis concentrations croissantes."""
        if self.bare is not None:
            yield "Bare", self.bare, self.bare_replicate_spectra, self.bare_analysis
        if self.probe is not None:
            yield "Probe", self.probe, self.probe_replicate_spectra, self.probe_analysis
        for grp in self.groups:
            yield f"{grp.concentration:.2e} M", grp.spectrum, grp.replicate_spectra, grp.analysis

    def failed_groups(self) -> list:
        """[(libellé, GroupAnalysis)] des groupes ARRÊTÉS (statut non OK)."""
        return [(lbl, an) for lbl, _sp, _reps, an in self.iter_groups()
                if an is not None and not an.ok]
