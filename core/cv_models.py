"""Dataclasses for CV (cyclic voltammetry) data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CVScan:
    label: str
    E: np.ndarray
    I: np.ndarray
    concentration: float
    step: str
    source_files: list = field(default_factory=list)


@dataclass
class CVConcentrationGroup:
    concentration: float
    scan: CVScan
    delta_signal: np.ndarray


@dataclass
class CVSession:
    probe: Optional[CVScan] = None
    groups: list = field(default_factory=list)
