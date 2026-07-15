"""Résolution de la pondération du fit EIS (CNLS complexe).

Deux modes, sélectionnés par `config.fit.weight_mode` :

- "modulus" (défaut) : pondération modulus, w_re = w_im = 1/(alpha_noise·|Z|)².
  alpha_noise est un niveau de bruit relatif ARBITRAIRE : les poids ne sont pas
  de vraies inverses de variance, donc la covariance doit être rééchelonnée par
  le χ² réduit (absolute_sigma=False) et chi2_reduced≈1 n'a pas de sens statistique.

- "sigma" (Measurement Model, Orazem) : si le spectre porte des écarts-types
  inter-réplicats mesurés (sigma_re(f), sigma_im(f)), w_re = 1/σ_re², w_im = 1/σ_im².
  Les poids SONT les vraies inverses de variance → covariance directe
  (absolute_sigma=True) et chi2_reduced≈1 devient le test d'adéquation
  modèle+erreur. Si σ indisponible (un seul réplicat), on retombe sur "modulus".

Module volontairement sans dépendance lourde (numpy seul) pour être importable
aussi bien depuis core/pipeline.py que depuis les modèles de fits/ sans risque
d'import circulaire.
"""

import numpy as np


def resolve_weights(spectrum, config):
    """Retourne (w_re, w_im, absolute_sigma) pour un spectre et une config donnés.

    Args:
        spectrum: EISSpectrum (attributs Zre, Zim et éventuellement sigma_re/sigma_im).
        config: dict de config app (clé "fit").

    Returns:
        w_re: np.ndarray — poids par point sur la partie réelle.
        w_im: np.ndarray — poids par point sur la partie imaginaire.
        absolute_sigma: bool — True si les poids sont de vraies 1/variance
            (mode "sigma" effectif), False en pondération modulus rééchelonnée.
    """
    fit_cfg = config.get("fit", {}) if isinstance(config, dict) else {}
    mode = fit_cfg.get("weight_mode", "modulus")

    sigma_re = getattr(spectrum, "sigma_re", None)
    sigma_im = getattr(spectrum, "sigma_im", None)
    has_sigma = sigma_re is not None and sigma_im is not None

    if mode == "sigma" and has_sigma:
        sre = np.asarray(sigma_re, dtype=float)
        sim = np.asarray(sigma_im, dtype=float)
        w_re = 1.0 / sre ** 2
        w_im = 1.0 / sim ** 2
        return w_re, w_im, True

    # Pondération modulus (par défaut, ou fallback si σ indisponible en mode sigma).
    alpha = float(fit_cfg.get("alpha_noise", 0.001))
    Zmod = np.sqrt(np.asarray(spectrum.Zre, dtype=float) ** 2
                   + np.asarray(spectrum.Zim, dtype=float) ** 2)
    w = 1.0 / (alpha * Zmod) ** 2
    return w, w, False
