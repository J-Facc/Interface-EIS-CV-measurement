"""Sens des dépendances entre paquets (AUDIT.md CPL-1) — vérifié sur l'arbre d'imports RÉEL.

Règle (documentée dans core/pipeline.py) :

    pages / ui / plotting / exports ──► core ──► {fits, drt, circuit}
    drt ──► fits.result ;  fits ──► circuit ;  circuit ──► (rien)

* ``fits/``, ``drt/`` et ``circuit/`` n'importent JAMAIS ``core`` (ni ui, pages,
  plotting, exports, streamlit) : l'ancienne dépendance circulaire core ↔ fits, masquée
  par des imports locaux, n'existe plus.
* ``core/`` n'importe ``fits``/``drt``/``circuit`` qu'au niveau MODULE : plus aucun
  import local qui « évite un cycle au chargement ».
* Les modules supprimés à l'étape 5 ne sont plus importés nulle part.
"""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_LIB_PACKAGES = ("fits", "drt", "circuit")
_UPPER = ("core", "ui", "pages", "plotting", "exports", "streamlit", "app")
_REMOVED = ("fits.randles_full", "fits.drt_fit", "fits.registry", "fits.base", "fits.error_structure",
            "fits.weighting", "vendor", "core.regression_stats")


def _py_files(*dirs):
    for d in dirs:
        for p in sorted((REPO / d).rglob("*.py")):
            if "bayes_drt2" not in p.parts:          # code tiers (drt/PROVENANCE.md)
                yield p


def _imports(path):
    """[(module importé, ligne, au niveau module ?)]."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = set(id(n) for n in tree.body)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno, id(node) in top) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module, node.lineno, id(node) in top))
    return out


def _root(mod):
    return mod.split(".")[0]


@pytest.mark.parametrize("path", list(_py_files(*_LIB_PACKAGES)), ids=lambda p: str(p.relative_to(REPO)))
def test_numeric_libraries_never_import_core_or_ui(path):
    bad = [(m, line) for m, line, _top in _imports(path) if _root(m) in _UPPER]
    assert bad == [], f"{path.relative_to(REPO)} importe une couche supérieure : {bad}"


def test_circuit_is_a_leaf_and_fits_does_not_import_drt():
    for path in _py_files("circuit"):
        assert all(_root(m) != "fits" and _root(m) != "drt" for m, _l, _t in _imports(path))
    for path in _py_files("fits"):
        assert all(_root(m) != "drt" for m, _l, _t in _imports(path)), path
    for path in _py_files("drt"):
        fits_imports = {m for m, _l, _t in _imports(path) if _root(m) == "fits"}
        assert fits_imports <= {"fits.result"}, (path, fits_imports)


@pytest.mark.parametrize("path", list(_py_files("core")), ids=lambda p: str(p.relative_to(REPO)))
def test_core_imports_the_numeric_libraries_at_module_level_only(path):
    local = [(m, line) for m, line, top in _imports(path)
             if _root(m) in _LIB_PACKAGES and not top]
    assert local == [], f"import local masquant une dépendance dans {path.relative_to(REPO)} : {local}"


def test_removed_modules_are_imported_nowhere():
    offenders = []
    for path in _py_files("core", "fits", "drt", "circuit", "plotting", "exports", "ui", "pages", "tests"):
        for m, line, _top in _imports(path):
            if any(m == r or m.startswith(r + ".") for r in _REMOVED):
                offenders.append((str(path.relative_to(REPO)), line, m))
    for extra in ("app.py", "setup_drt_bayesien.py"):
        for m, line, _top in _imports(REPO / extra):
            if any(m == r or m.startswith(r + ".") for r in _REMOVED):
                offenders.append((extra, line, m))
    assert offenders == []
    for gone in ("fits/randles_full.py", "fits/drt_fit.py", "fits/registry.py", "fits/base.py",
                 "fits/error_structure.py", "fits/weighting.py", "vendor"):
        assert not (REPO / gone).exists(), gone


def test_fit_result_has_a_single_definition_re_exported_by_core():
    from core.models import FitResult as from_core
    from fits.result import FitResult as from_fits

    assert from_core is from_fits
