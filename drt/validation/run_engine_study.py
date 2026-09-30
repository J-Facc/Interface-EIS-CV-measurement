# -*- coding: utf-8 -*-
"""Étude complémentaire via ``drt/engine.py`` : budget HMC, localisation de Rp, Rct.

Complète ``run_matrix.py`` (bibliothèque nue) en passant par le moteur durci, pour
mesurer ce que l'interface verra : Rct (pic pénultième) et son IC a posteriori, alertes,
et OÙ se trouve l'aire de γ(τ) par rapport à la fenêtre de mesure
[1/(2π f_max), 1/(2π f_min)] — pour expliquer un éventuel biais de Rp.

Sortie : une ligne JSON par essai (``--out``), + γ(τ) moyen dans un ``.npz`` voisin.

Usage (racine du dépôt, CMDSTAN défini) ::

    # réglage retenu (défauts du moteur) :
    python drt/validation/run_engine_study.py --out drt/validation/results_engine_final_sample.jsonl \
        --settings nonneg+ridge
    python drt/validation/run_engine_study.py --out drt/validation/results_engine_final_optimize.jsonl \
        --settings nonneg+ridge default --mode optimize
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drt.validation.run_matrix import SETTINGS, build_cases, environment  # noqa: E402

# Grandeurs « vraies » de l'arc visé par la convention du pic pénultième :
# 2 RC → arc à τ1 = 1 ms (R1 = 50 Ω) ; Randles → Rct du circuit.
def _rct_target(case_id):
    if case_id.startswith("A3"):
        return 50.0, 1e-3
    return float(case_id.split("Rct")[1]), None


def window_masses(tau, gamma, f):
    """Aires de γ (∫ γ dlnτ) sous, dans et au-dessus de la fenêtre mesurée."""
    lo, hi = 1 / (2 * np.pi * f.max()), 1 / (2 * np.pi * f.min())
    order = np.argsort(tau)
    t, g = tau[order], gamma[order]
    x = np.log(t)

    def area(m):
        return float(np.trapezoid(np.where(m, g, 0.0), x))

    return {"tau_window": [lo, hi], "area_below": area(t < lo), "area_in": area((t >= lo) & (t <= hi)),
            "area_above": area(t > hi)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="sample", choices=("sample", "optimize"))
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    from drt import engine as eng

    ap.add_argument("--chains", type=int, default=eng.DEFAULT_CHAINS)
    ap.add_argument("--warmup", type=int, default=eng.DEFAULT_WARMUP)
    ap.add_argument("--samples", type=int, default=eng.DEFAULT_SAMPLES)
    ap.add_argument("--adapt-delta", type=float, default=eng.DEFAULT_ADAPT_DELTA)
    args = ap.parse_args(argv)

    from core.models import EISSpectrum
    from drt.engine import fit_drt

    out = Path(args.out)
    npz_dir = out.with_suffix("")
    npz_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    for cid, desc, seed0, builder in build_cases():
        if args.cases and cid not in args.cases:
            continue
        f, Z, rp_true = builder()
        sp = EISSpectrum(label=cid, f=f, Zre=Z.real, Zim=-Z.imag, concentration=0.0, step="probe",
                         n_points=len(f))
        rct_true, tau_true = _rct_target(cid)
        for seed in (args.seeds or [seed0]):
            for st in args.settings:
                kw = dict(SETTINGS[st])
                kw.setdefault("nonneg", False)
                kw.setdefault("init_from_ridge", False)
                t0 = time.time()
                rec = {"case": cid, "setting": st, "mode": args.mode, "stan_seed": seed,
                       "chains": args.chains, "warmup": args.warmup, "samples": args.samples,
                       "adapt_delta": args.adapt_delta, "rct_peaks_in_window": eng.RCT_PEAKS_IN_MEASURED_WINDOW,
                       "rp_true": rp_true, "rct_target": rct_true, "tau_target": tau_true}
                try:
                    fr = fit_drt(sp, mode=args.mode, random_seed=seed, chains=args.chains,
                                 warmup=args.warmup, samples=args.samples, adapt_delta=args.adapt_delta, **kw)
                    d = fr.drt_diagnostics
                    s = dict(d["sampler"] or {})
                    s.pop("per_variable", None)
                    rec.update(status="ok", rp=fr.params["Rp"], rp_ci95=d["Rp_ci95"], rct=fr.Rct,
                               rct_std=fr.Rct_std, rct_ci95=d["Rct_ci95"], tau_rct=fr.params["tau_Rct"],
                               rct_source=fr.params["rct_source"], converged=fr.converged,
                               alerts=[a["code"] for a in d["alerts"]], notes=d["notes"],
                               recon_max=fr.reconstruction_error_relative, recon_rms=fr.reconstruction_error,
                               gamma_min=float(np.min(fr.drt_gamma)),
                               negative_area_fraction=d["negative_area_fraction"], sampler=s,
                               library_warnings=d["library_warnings"],
                               **window_masses(fr.drt_tau, fr.drt_gamma, f))
                    np.savez(npz_dir / f"{cid}__{st}__s{seed}.npz", tau=fr.drt_tau, gamma=fr.drt_gamma,
                             lo=fr.drt_gamma_lo if fr.drt_gamma_lo is not None else np.array([]),
                             hi=fr.drt_gamma_hi if fr.drt_gamma_hi is not None else np.array([]))
                except Exception as exc:  # noqa: BLE001
                    rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
                rec["time_s"] = round(time.time() - t0, 1)
                rec["env"] = env
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
                if rec["status"] == "ok":
                    print(f"{cid:<26} {st:<15} s={seed:<5} Rp={rec['rp']:8.1f} (vrai {rp_true:.0f}) "
                          f"in={rec['area_in']:8.1f} above={rec['area_above']:6.1f} below={rec['area_below']:6.1f} "
                          f"Rct={rec['rct']:8.1f}±{rec['rct_std']:.1f} τ={rec['tau_rct']:.2g} ({rec['rct_source']}) "
                          f"recon={rec['recon_max'] * 100:.2f}% alerts={rec['alerts']} "
                          f"div={s.get('divergences')} mtd={s.get('max_treedepth_hits')} "
                          f"rhat={s.get('rhat_max')} essb={s.get('ess_bulk_min')} t={rec['time_s']}s", flush=True)
                else:
                    print(cid, st, seed, rec["error"], flush=True)


if __name__ == "__main__":
    main()
