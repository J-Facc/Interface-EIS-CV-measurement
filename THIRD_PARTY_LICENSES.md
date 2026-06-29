# Third-party licenses

## pyDRTtools

`fits/_pydrttools/` vendors a subset of [pyDRTtools](https://github.com/ciuccislab/pyDRTtools)
(`basics.py`, `parameter_selection.py`, `nearest_PD.py` — the pure
NumPy/SciPy/cvxopt/scikit-learn computational core, with no PyQt5/Qt
dependency). It is vendored rather than installed from PyPI because
pyDRTtools' own `__init__.py` unconditionally imports its GUI module,
which hard-imports PyQt5 — making the package impossible to import
headlessly even just for `basics`.

`fits/drt_tikhonov.py` calls these vendored functions directly to compute
the model-free DRT (radial-basis-function discretization + Tikhonov
regularization + non-negativity-constrained quadratic programming via
cvxopt), replacing an earlier from-scratch reimplementation, for fidelity
to the reference tool used in the literature (see citations in
`fits/drt_tikhonov.py` and `ARCHITECTURE.md`).

License: MIT

```
MIT License

Copyright (c) 2023 ciuccislab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
