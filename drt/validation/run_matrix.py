# -*- coding: utf-8 -*-
"""Matrice de validation numérique des réglages de bayes_drt2 (drt/bayes_drt2).

Rejoue, sur le clone identifié de ``drt/bayes_drt2`` (voir ``drt/PROVENANCE.md``),
les scénarios de ``AUDIT.md`` Annexe A.3 et A.4, pour les DEUX modes de résolution
(``optimize`` = MAP L-BFGS, ``sample`` = HMC/NUTS ; en HMC sur le périmètre restreint du §3.0
de ``drt/VALIDATION_REGLAGES.md``) et quatre réglages :

* ``default``          : ``Inverter.fit`` sans option (réglages amont) ;
* ``init_from_ridge``  : initialisation par la solution ridge hyperparamétrique ;
* ``nonneg``           : DRT contrainte ≥ 0 (modèle Stan ``Series_pos``) ;
* ``nonneg+ridge``     : les deux.

Aucun code applicatif n'est utilisé (ni l'ancien ``fits/drt_fit.py`` ni ``drt/engine.py``) :
on appelle ``Inverter`` directement, comme les scripts de l'audit, pour mesurer la
bibliothèque et non un wrapper.

Métriques (identiques à l'audit, plus les diagnostics HMC) :

* ``Rp`` (``Inverter.predict_Rp()``) et son écart relatif à la valeur vraie ;
* ``gamma_min`` : min de γ(τ) sur la grille de base ;
* ``recon_max`` : max_i |Z_fit − Z| / |Z| aux fréquences mesurées ;
* HMC : divergences post-warmup, itérations à profondeur d'arbre maximale,
  R-hat (rang-normalisé, split — ``stansummary`` de CmdStan) et ESS bulk/tail
  minimaux par groupe de paramètres, E-BFMI minimal par chaîne, IC 95 % de Rp.

Usage (depuis la racine du dépôt, CMDSTAN défini) ::

    python drt/validation/run_matrix.py --mode optimize --out drt/validation/results_optimize.jsonl
    # HMC : périmètre restreint (VALIDATION_REGLAGES.md §3.0) — candidats partout, amont sur 3 cas,
    # combinaison sur les scénarios divergents en MAP (listés par report.py --hmc-scope) :
    python drt/validation/run_matrix.py --mode sample --out drt/validation/results_sample.jsonl --jobs 2 \
        --settings init_from_ridge nonneg
    python drt/validation/run_matrix.py --mode sample --out drt/validation/results_sample.jsonl --jobs 2 \
        --settings default --cases A3-60pts-1e5-1e-1 A3-60pts-bruit0.5% A4-Randles-Rct3000
    python drt/validation/run_matrix.py --mode sample --out drt/validation/results_sample.jsonl \
        --settings nonneg+ridge --cases A3-80pts-1e6-1e-1
    # étude adapt_delta / warmup (§3 de VALIDATION_REGLAGES.md) :
    python drt/validation/run_matrix.py --mode sample --out drt/validation/results_adapt.jsonl \
        --settings nonneg+ridge --cases A3-repo-71pts-1e5-1e-2 A3-60pts-bruit0.5% A3-80pts-1e6-1e-1 \
        A4-Randles-Rct3500 --chains 4 --warmup 500 --samples 500 --adapt-delta 0.99

Chaque ligne du JSONL est un essai ; ``drt/validation/report.py`` en fait les
tableaux de ``drt/VALIDATION_REGLAGES.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETTINGS = {
    "default": {},
    "init_from_ridge": {"init_from_ridge": True},
    "nonneg": {"nonneg": True},
    "nonneg+ridge": {"nonneg": True, "init_from_ridge": True},
}

# Graine Stan par défaut de bayes_drt2 (Inverter.fit(random_seed=1234)) — celle de l'audit.
DEFAULT_STAN_SEED = 1234


# ─────────────────────────────────────────────────────────────────────────────
# Cas de l'Annexe A.3 / A.4 (données reconstruites À L'IDENTIQUE des scripts de l'audit)
# ─────────────────────────────────────────────────────────────────────────────
def _two_rc(n, f0, f1, noise=0.0, seed=0):
    """A.3 : R0=10 + 50/(1+jωτ1) + 80/(1+jωτ2), τ1=1 ms, τ2=0,1 s. Rp vrai = 130 Ω."""
    f = np.logspace(f0, f1, n)
    w = 2 * np.pi * f
    Z = 10 + 50 / (1 + 1j * w * 1e-3) + 80 / (1 + 1j * w * 1e-1)
    r = np.random.default_rng(seed)
    Z = Z * (1 + noise * (r.standard_normal(n) + 1j * r.standard_normal(n)))
    return f, Z, 130.0


def _randles(Rct, noise=0.005, seed=None):
    """A.4 : Randles complet, 60 pts 1e5→1e-1 Hz, bruit 0,5 % (graine = Rct).

    Circuit construit par ``circuit.parse_circuit`` (même expression que ``fit.circuit``
    par défaut) : identique à l'ancien ``fits/physics.py:Z_randles_full`` à 1e-12
    près (``tests/test_circuit_parser.py``), donc mêmes spectres que les mesures publiées.
    Rp vrai = Z(0) − Z(∞) = R'e + Rct + R_D = 20 + 1,3·Rct.
    """
    from circuit import parse_circuit

    f = np.logspace(5, -1, 60)
    w = 2 * np.pi * f
    r = np.random.default_rng(Rct if seed is None else seed)
    Z_randles, _names = parse_circuit(
        "Re + parallel(Re_prime + parallel(R(Rct) + ZD_bounded(R_D, tau_d), Q(Qdl, alpha)), C(Cb))")
    Z = Z_randles(w, Re=200., Re_prime=20., Cb=1e-9, Rct=Rct, Qdl=2e-6, alpha=0.9,
                  R_D=Rct * 0.3, tau_d=0.5)
    Z = Z.real * (1 + noise * r.standard_normal(60)) + 1j * Z.imag * (1 + noise * r.standard_normal(60))
    return f, Z, 20.0 + 1.3 * Rct


def build_cases():
    """(case_id, description, stan_seed, builder) — l'ordre suit l'Annexe A."""
    cases = [
        ("A3-repo-71pts-1e5-1e-2", "2 RC, 71 pts, 1e5→1e-2 Hz, sans bruit (scénario de tests/test_drt.py)",
         DEFAULT_STAN_SEED, lambda: _two_rc(71, 5, -2)),
        ("A3-60pts-1e5-1e-1", "2 RC, 60 pts, 1e5→1e-1 Hz, sans bruit", DEFAULT_STAN_SEED, lambda: _two_rc(60, 5, -1)),
        ("A3-40pts-1e5-1e-1", "2 RC, 40 pts, 1e5→1e-1 Hz, sans bruit", DEFAULT_STAN_SEED, lambda: _two_rc(40, 5, -1)),
        ("A3-80pts-1e6-1e-1", "2 RC, 80 pts, 1e6→1e-1 Hz, sans bruit", DEFAULT_STAN_SEED, lambda: _two_rc(80, 6, -1)),
        ("A3-60pts-bruit0.5%", "2 RC, 60 pts, 1e5→1e-1 Hz, bruit 0,5 % (graine bruit 0)", DEFAULT_STAN_SEED,
         lambda: _two_rc(60, 5, -1, noise=0.005)),
    ]
    for s in (1, 2, 3):
        cases.append((f"A3-60pts-bruit0.5%-seed{s}", f"idem, graine Stan {s}", s,
                      lambda: _two_rc(60, 5, -1, noise=0.005)))
    for R in (3000, 3500, 4200, 5200):
        cases.append((f"A4-Randles-Rct{R}", f"Randles Rct={R} Ω, 60 pts, bruit 0,5 % (graine bruit {R})",
                      DEFAULT_STAN_SEED, (lambda R=R: _randles(R))))
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics HMC (API cmdstanpy 1.3 : CmdStanMCMC.summary / divergences /
# max_treedepths / method_variables) — groupes de paramètres du modèle Series
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GROUPS = {
    "x": ("x",),
    "offsets": ("Rinf_raw", "induc_raw"),
    "error": ("sigma_res_raw", "alpha_prop_raw", "alpha_re_raw", "alpha_im_raw"),
    "hyper": ("ups_raw", "d0_strength", "d1_strength", "d2_strength"),
}


def _hmc_diagnostics(inv):
    fit = inv.stan_mcmc
    summ = fit.summary()
    base = np.array([i.split("[")[0] for i in summ.index])
    out = {
        "chains": int(fit.chains),
        "draws_per_chain": int(fit.num_draws_sampling),
        "divergences": int(np.sum(fit.divergences)),
        "max_treedepth_hits": int(np.sum(fit.max_treedepths)),
    }
    groups = dict(PARAM_GROUPS)
    groups["params"] = tuple(v for g in PARAM_GROUPS.values() for v in g)
    groups["lp__"] = ("lp__",)
    for g, names in groups.items():
        m = np.isin(base, names)
        sub = summ[m]
        out[f"rhat_max_{g}"] = float(np.nanmax(sub["R_hat"])) if len(sub) else None
        out[f"ess_bulk_min_{g}"] = float(np.nanmin(sub["ESS_bulk"])) if len(sub) else None
        out[f"ess_tail_min_{g}"] = float(np.nanmin(sub["ESS_tail"])) if len(sub) else None
        out[f"rhat_nan_{g}"] = int(np.isnan(sub["R_hat"].to_numpy(dtype=float)).sum())
    out["rhat_max_all"] = float(np.nanmax(summ["R_hat"]))
    # E-BFMI (Betancourt 2016) par chaîne à partir de energy__ (draws × chains)
    energy = np.asarray(fit.method_variables()["energy__"], dtype=float)
    energy = energy.reshape(energy.shape[0], -1)
    ebfmi = [float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2)) for e in energy.T]
    out["ebfmi_min"] = float(min(ebfmi))
    out["rp_ci95"] = [float(inv.predict_Rp(percentile=2.5)), float(inv.predict_Rp(percentile=97.5))]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Un essai
# ─────────────────────────────────────────────────────────────────────────────
def run_one(case_id, desc, stan_seed, builder_idx, setting, mode, sample_kw):
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    from drt.bayes_drt2.inversion import Inverter

    builder = build_cases()[builder_idx][3]
    f, Z, rp_true = builder()
    rec = {"case": case_id, "desc": desc, "setting": setting, "mode": mode, "stan_seed": stan_seed,
           "n_points": int(len(f)), "rp_true": rp_true}
    kw = dict(SETTINGS[setting])
    if mode == "sample":
        kw.update(sample_kw)
    rec["fit_kwargs"] = kw
    t0 = time.time()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            inv = Inverter()
            inv.fit(f, Z, mode=mode, random_seed=stan_seed, **kw)
            tau = np.asarray(inv.distributions["DRT"]["tau"], dtype=float)
            g = np.asarray(inv.predict_distribution("DRT", tau=tau), dtype=float)
            Zf = np.asarray(inv.predict_Z(f))
            rel = np.abs(Zf - Z) / np.abs(Z)
            rp = float(inv.predict_Rp())
            rec.update(status="ok", rp=rp, rp_rel_err=(rp - rp_true) / rp_true,
                       gamma_min=float(g.min()), gamma_max=float(g.max()),
                       recon_max=float(rel.max()), recon_mean=float(rel.mean()),
                       n_tau=int(len(tau)), stan_model=inv.stan_model_name,
                       R_inf=float(np.ravel(inv.R_inf)[0]))
            if mode == "sample":
                rec.update(_hmc_diagnostics(inv))
        except Exception as exc:  # noqa: BLE001 — on enregistre l'échec tel quel
            rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
    rec["time_s"] = round(time.time() - t0, 2)
    rec["py_warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in ws
                                 if not issubclass(w.category, (SyntaxWarning, DeprecationWarning))})
    return rec


def environment():
    import cmdstanpy
    import cvxopt
    import scipy
    import subprocess

    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = None
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "cvxopt": cvxopt.__version__, "cmdstanpy": cmdstanpy.__version__,
            "cmdstan": os.path.basename(cmdstanpy.cmdstan_path()), "repo_head": sha,
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def precompile():
    """Compile Series/Series_pos une fois (évite des compilations concurrentes)."""
    from drt.stan_compile import compile_stan_model

    d = ROOT / "drt" / "bayes_drt2" / "stan_model_files"
    for name in ("Series", "Series_pos"):
        compile_stan_model(str(d / f"{name}.stan"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("optimize", "sample"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS))
    ap.add_argument("--cases", nargs="*", default=None, help="sous-ensemble de case_id")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="remplace la graine Stan de chaque cas (balayage de robustesse)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--adapt-delta", type=float, default=None,
                    help="adapt_delta NUTS (patch 2 de drt/PROVENANCE.md) ; défaut amont 0,9")
    args = ap.parse_args(argv)

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    precompile()
    sample_kw = dict(warmup=args.warmup, samples=args.samples, chains=args.chains)
    if args.adapt_delta is not None:
        sample_kw["adapt_delta"] = args.adapt_delta
    cases = build_cases()
    jobs = []
    for idx, (cid, desc, seed, _b) in enumerate(cases):
        if args.cases and cid not in args.cases:
            continue
        for s in (args.seeds or [seed]):
            for st in args.settings:
                jobs.append((cid, desc, s, idx, st, args.mode, sample_kw))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = environment()
    with out.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, *j): j for j in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            rec["env"] = env
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            msg = (f"{rec['case']:<28} {rec['setting']:<16} seed={rec['stan_seed']:<5} {rec['status']:<5} "
                   f"Rp={rec.get('rp', float('nan')):9.1f} (vrai {rec['rp_true']:.0f}) "
                   f"gmin={rec.get('gamma_min', float('nan')):9.2f} recon={rec.get('recon_max', float('nan')):.3g}")
            if args.mode == "sample" and rec["status"] == "ok":
                msg += (f" div={rec['divergences']} mtd={rec['max_treedepth_hits']} "
                        f"rhat={rec['rhat_max_params']:.3f} essb={rec['ess_bulk_min_params']:.0f}")
            print(msg + f" t={rec['time_s']}s", flush=True)


if __name__ == "__main__":
    main()
