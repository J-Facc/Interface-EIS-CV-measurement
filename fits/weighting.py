"""Pondération UNIQUE du fit EIS (CNLS complexe) — structure d'erreur d'Orazem.

Il n'existe plus qu'UNE seule méthode de pondération. Les anciens modes
"modulus" (1/(alpha_noise·|Z|)²) et "sigma" (1/σ² inter-réplicats bruts) ont été
supprimés, ainsi que tout sélecteur `weight_mode` et le paramètre `alpha_noise`.

Les poids sont TOUJOURS de vraies inverses de variance construites à partir de la
structure d'erreur stochastique de l'instrument (fits/error_structure.py) :

    σ_i   = α·|Z_re,i| + β·|Z_im,i| + γ·(|Z_i|²/R_m) + δ
    w_re,i = w_im,i = 1/σ_i²        (absolute_sigma = True, TOUJOURS)

Par conséquent la covariance des paramètres n'est JAMAIS rééchelonnée par le χ²
réduit (cf. fits/randles_full.py), et χ²_red est un vrai test d'adéquation.

Les coefficients (α, β, γ, δ) proviennent de réplicats (caractérisation stricte
d'Orazem) ou, à défaut, des derniers coefficients persistés ; sinon le fit est
refusé (ErrorStructureUnavailable). Voir fits/error_structure.py.

Références : Orazem & Tribollet, "Electrochemical Impedance Spectroscopy", Wiley
(chap. Measurement Model / Error Structure) ; Agarwal, Orazem & García-Rubio,
J. Electrochem. Soc. (1992-1995).

Aucune dépendance Streamlit (importable depuis core/ et fits/).
"""

import numpy as np

from fits.error_structure import ErrorStructure, resolve_error_structure

# Ré-export pour les appelants historiques qui attrapent l'exception.
from fits.error_structure import ErrorStructureUnavailable  # noqa: F401

# Garde-fou : σ ne doit jamais être nul/négatif (division par zéro des poids).
# La structure d'erreur garantit σ ≥ δ ≥ 0 ; ce plancher ABSOLU (et non relatif)
# ne s'active qu'en cas de δ = 0 combiné à |Z| = 0, situation pathologique.
_SIGMA_ABS_FLOOR = 1e-12


def resolve_weights(spectrum, config):
    """Retourne (w_re, w_im, error_structure) pour un spectre et une config.

    Args:
        spectrum: EISSpectrum (Zre, Zim ; éventuellement sigma_re/sigma_im,
            n_replicates, replicates pour la caractérisation).
        config: dict de config app (clé "fit").

    Returns:
        w_re: np.ndarray — poids par point sur la partie réelle (= 1/σ_i²).
        w_im: np.ndarray — poids par point sur la partie imaginaire (= 1/σ_i²).
        error_structure: ErrorStructure — coefficients + provenance
            (source ∈ {"characterized_now", "reused_persisted"}, horodatage…).

    Raises:
        ErrorStructureUnavailable: si aucune structure d'erreur n'est disponible
            (ni réplicats, ni coefficients persistés) — le fit doit être refusé.
    """
    es: ErrorStructure = resolve_error_structure(spectrum, config)

    # σ_struct décrit le bruit d'UNE mesure unique (l'écart-type inter-réplicats,
    # ddof=1, estime le bruit d'un réplicat individuel). L'écart-type du DATUM
    # réellement ajusté dépend de sa nature :
    #   - spectre = moyenne de N réplicats  → Var = σ_struct²/N  → σ_i = σ_struct/√N ;
    #   - spectre = mesure unique (N = 1)    → σ_i = σ_struct.
    # Pondérer par 1/σ_i² (et non 1/σ_struct²) est indispensable pour que χ²_red
    # soit un VRAI test d'adéquation sur le datum ajusté (moyenne ou réplicat).
    # C'est bien la forme w = 1/σ_i² : seul σ_i est correctement identifié par datum.
    n_eff = getattr(spectrum, "n_replicates", None) or 1
    n_eff = max(int(n_eff), 1)

    sigma_struct = es.sigma(spectrum.Zre, spectrum.Zim)
    sigma_i = np.maximum(np.asarray(sigma_struct, dtype=float), _SIGMA_ABS_FLOOR) / np.sqrt(n_eff)
    w = 1.0 / sigma_i ** 2
    # equal_re_im : σ (donc w) identique sur réel et imaginaire.
    return w, w, es
