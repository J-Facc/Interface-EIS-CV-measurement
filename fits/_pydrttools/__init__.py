"""Vendored subset of pyDRTtools (https://github.com/ciuccislab/pyDRTtools).

MIT License, Copyright (c) 2023 ciuccislab — see THIRD_PARTY_LICENSES.md.

Only `basics.py`, `parameter_selection.py`, and `nearest_PD.py` are
vendored: the pure NumPy/SciPy/cvxopt/sklearn computational core, with no
PyQt5/Qt-matplotlib dependency. The upstream package's own `__init__.py`
unconditionally imports its GUI module (which hard-imports PyQt5), so a
plain `pip install pyDRTtools` cannot be used headlessly — hence vendoring
instead of depending on the PyPI package.
"""
