from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CVScan:
    label: str
    E: np.ndarray
    I: np.ndarray
    concentration: float
    step: str  # 'probe' ou 'hybridization'
    source_files: list = field(default_factory=list)


@dataclass
class CVConcentrationGroup:
    concentration: float
    scan: CVScan
    delta_signal: np.ndarray  # |I_probe - I_conc| / |I_probe| interpolé


@dataclass
class CVSession:
    probe: Optional[CVScan] = None
    groups: list = field(default_factory=list)  # list[CVConcentrationGroup]
    # Référence « électrode nue » — AFFICHAGE SEUL, JAMAIS utilisée dans les
    # calculs (ni delta_signal, ni calibration, ni export de valeurs calculées).
    # Champ dédié et séparé de `probe`/`groups` : attachée à la session APRÈS
    # l'analyse et seulement superposée à la courbe I/E.
    bare_reference: Optional[CVScan] = None
