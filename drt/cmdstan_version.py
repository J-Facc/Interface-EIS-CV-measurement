# -*- coding: utf-8 -*-
"""drt/cmdstan_version.py — la version de CmdStan sur laquelle la DRT a été validée.

Module volontairement minimal (bibliothèque standard seule) : il est lu à la fois par
``setup_drt_bayesien.py`` (qui épingle l'installation), par ``drt/engine.py`` (qui signale
une autre version dans chaque résultat) et par ``app.py`` (qui l'affiche), sans qu'aucun de
ces trois ne dépende des autres.

Pourquoi une version épinglée : tous les réglages de ``drt/engine.py`` (``nonneg``,
``init_from_ridge``, ``adapt_delta``…) et leurs seuils de diagnostic ont été mesurés avec
CmdStan 2.36.0 (``drt/VALIDATION_REGLAGES.md``, job CI ``drt``). Une autre version peut très
bien donner le même résultat, mais cela n'a **pas été vérifié** : c'est ce que dit
:func:`version_warning`, jamais davantage.
"""

from __future__ import annotations

import os
import re
from typing import Optional

#: Version de CmdStan de VALIDATION_REGLAGES.md. Doit rester égale à ``CMDSTAN_VERSION`` de
#: ``.github/workflows/validate.yml`` (gardé par ``tests/test_cmdstan_version.py``).
PINNED_CMDSTAN_VERSION = "2.36.0"

_DIRNAME_RE = re.compile(r"^cmdstan-(\d+\.\d+\.\d+)$")
_MAKEFILE_RE = re.compile(r"^\s*CMDSTAN_VERSION\s*:?=\s*(\d+\.\d+\.\d+)\s*$", re.MULTILINE)


def version_from_dirname(path: str) -> Optional[str]:
    """``'…/cmdstan-2.36.0'`` → ``'2.36.0'`` ; ``None`` pour tout autre nom (rc, git:…)."""
    match = _DIRNAME_RE.match(os.path.basename(os.path.normpath(path)))
    return match.group(1) if match else None


def version_from_makefile(cmdstan_dir: str) -> Optional[str]:
    """Version lue dans le ``makefile`` de CmdStan (``CMDSTAN_VERSION := x.y.z``), ou ``None``."""
    try:
        with open(os.path.join(cmdstan_dir, "makefile"), encoding="utf-8", errors="replace") as fh:
            match = _MAKEFILE_RE.search(fh.read())
    except OSError:
        return None
    return match.group(1) if match else None


def installed_version(cmdstan_dir: str) -> Optional[str]:
    """Version « x.y.z » du CmdStan installé dans ``cmdstan_dir``, ``None`` si indéterminable.

    Le ``makefile`` fait foi (un dossier peut avoir été renommé) ; le nom du dossier ne sert
    qu'en repli. ``cmdstanpy.cmdstan_version()`` n'est pas utilisé : il ne rend que
    ``(majeur, mineur)``, donc ne distingue pas 2.36.0 de 2.36.1.
    """
    return version_from_makefile(cmdstan_dir) or version_from_dirname(cmdstan_dir)


def version_warning(version: Optional[str]) -> Optional[str]:
    """Avertissement lisible si ``version`` n'est pas la version validée, sinon ``None``.

    Une version indéterminable est traitée comme une version différente : l'absence
    d'information n'est pas une confirmation.
    """
    if version == PINNED_CMDSTAN_VERSION:
        return None
    seen = f"CmdStan {version}" if version else "Une version de CmdStan indéterminée"
    return (
        f"{seen} est utilisé, alors que les réglages de la DRT (drt/VALIDATION_REGLAGES.md) "
        f"n'ont été validés que sur CmdStan {PINNED_CMDSTAN_VERSION} : leur comportement sur "
        f"cette version n'a pas été vérifié. Relancez launch.bat avec accès réseau pour "
        f"installer {PINNED_CMDSTAN_VERSION}."
    )
