# Third-party licenses

## bayes-drt2

`drt/bayes_drt2/` is a copy of [bayes-drt2](https://github.com/jdhuang-csm/bayes-drt2)
(Jake Huang, Colorado School of Mines) — hierarchical Bayesian inversion of
electrochemical impedance data (DRT) — taken from an **identified** git clone: commit
`99d5b603d98469a6382ddde5210a1bfdc0d76b3d` (branch `main`, cloned 2026-09-30), with two
documented patches (numpy ≥ 2.4 compatibility; `adapt_delta` exposed in `Inverter.fit`).
It is the DRT engine of the app: `drt/engine.py` wraps `drt.bayes_drt2.inversion.Inverter`
(HMC `mode='sample'` and MAP `mode='optimize'`). Provenance, diffs and full license text:
`drt/PROVENANCE.md`.

The former unidentified vendoring (`vendor/bayes_drt2/`) and its wrapper
`fits/drt_fit.py` were removed at step 5 of the refactoring (see `drt/PROVENANCE.md`
for the comparison between both copies).

License: BSD 3-clause — see `drt/bayes_drt2/LICENSE`.
