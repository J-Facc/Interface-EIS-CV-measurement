"""Détection des pics redox (anodique/cathodique) sur un voltammogramme CV.

Ajout pour la réorganisation des onglets CV (onglet "Pic redox").
Stratégie : léger lissage Savitzky-Golay puis recherche du maximum global
(pic anodique, Ipa/Epa) et du minimum global (pic cathodique, Ipc/Epc) du
courant lissé — robuste pour un système à un seul couple redox (Fe(CN)6 etc.),
cohérent avec l'approche déjà utilisée dans pages/B_cv.py (_extract_peaks).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from core.cv_models import CVScan


def detect_redox_peaks(scan: CVScan, window_length: int = 9, polyorder: int = 3) -> dict:
    """Détecte les pics anodique et cathodique d'un voltammogramme.

    Args:
        scan: CVScan (E, I).
        window_length: fenêtre Savitzky-Golay (impaire, réduite si le scan est court).
        polyorder: ordre du polynôme local Savitzky-Golay.

    Returns:
        dict avec Ipa, Epa (pic anodique : courant max, potentiel associé),
        Ipc, Epc (pic cathodique : courant min, potentiel associé),
        delta_Ep = Epa - Epc.
    """
    E = np.asarray(scan.E, dtype=float)
    I = np.asarray(scan.I, dtype=float)

    n = len(I)
    if n == 0:
        return {"Ipa": float("nan"), "Epa": float("nan"),
                "Ipc": float("nan"), "Epc": float("nan"),
                "delta_Ep": float("nan")}

    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl < polyorder + 2:
        I_smooth = I
    else:
        if wl % 2 == 0:
            wl -= 1
        try:
            I_smooth = savgol_filter(I, window_length=wl, polyorder=min(polyorder, wl - 1))
        except Exception:
            I_smooth = I

    idx_a = int(np.argmax(I_smooth))
    idx_c = int(np.argmin(I_smooth))

    Ipa, Epa = float(I[idx_a]), float(E[idx_a])
    Ipc, Epc = float(I[idx_c]), float(E[idx_c])

    return {
        "Ipa": Ipa, "Epa": Epa,
        "Ipc": Ipc, "Epc": Epc,
        "delta_Ep": Epa - Epc,
    }
