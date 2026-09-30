# Third-party licenses

## bayes-drt2

`vendor/bayes_drt2/` vendors [bayes-drt2](https://github.com/jdhuang-csm/bayes-drt2)
(Jake Huang, Colorado School of Mines) — hierarchical Bayesian inversion of
electrochemical impedance data (DRT). It is the DRT engine of the app:
`fits/drt_fit.py` wraps `vendor.bayes_drt2.inversion.Inverter` for the MAP fit
(`fit(..., mode='optimize')`, default) and the HMC reference fit with credible
intervals (`fit(..., mode='sample')`, on demand).

See `vendor/README.md` for provenance (per-file sha256), the single
`np.trapz → np.trapezoid` compatibility patch (numpy ≥ 2), and usage notes.

License: BSD 3-clause — see `vendor/bayes_drt2/LICENSE`.

### `drt/bayes_drt2/` — identified clone (replacement for `vendor/bayes_drt2/`)

Same library, copied from an **identified** git clone: commit
`99d5b603d98469a6382ddde5210a1bfdc0d76b3d` (branch `main`, cloned 2026-09-30), with two
documented patches (numpy ≥ 2.4 compatibility; `adapt_delta` exposed in `Inverter.fit`).
Hardened engine: `drt/engine.py`. Provenance, diffs and full license text: `drt/PROVENANCE.md`.
`vendor/bayes_drt2/` and `fits/drt_fit.py` stay in place until the pipeline is switched over.

License: BSD 3-clause — see `drt/bayes_drt2/LICENSE`.
