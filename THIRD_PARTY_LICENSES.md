# Third-party licenses

## bayes-drt2

`vendor/bayes_drt2/` vendors [bayes-drt2](https://github.com/jdhuang-csm/bayes-drt2)
(Jake Huang, Colorado School of Mines) — hierarchical Bayesian inversion of
electrochemical impedance data (DRT). It is the DRT engine of the app:
`fits/drt_fit.py` wraps `vendor.bayes_drt2.inversion.Inverter` for the
hyperparametric ridge preview (`ridge_fit`) and the HMC reference fit with
credible intervals (`fit(..., mode='sample')`).

See `vendor/README.md` for provenance (per-file sha256), the single
`np.trapz → np.trapezoid` compatibility patch (numpy ≥ 2), and usage notes.

License: BSD 3-clause — see `vendor/bayes_drt2/LICENSE`.
