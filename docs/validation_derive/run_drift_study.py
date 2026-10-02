# -*- coding: utf-8 -*-
"""Calibration empirique des critères de dérive entre réplicats (docs/VALIDATION_DETECTION_DERIVE.md).

Pour chaque groupe synthétique de réplicats (Randles de l'Annexe A, 40 points 1e5 → 1e-1 Hz),
exécute EXACTEMENT les chemins de production — ``core.measurement_model.analyze_replicates``,
``core.validator.validation_from_analysis`` (donc ``_detect_drift``),
``fits.orazem_fit.fit_replicate_group`` (options par défaut du pipeline : 8 départs) — puis
enregistre, une ligne JSON par groupe, le verdict de chaque critère candidat :

* ``detect_drift``   : ``ValidationResult.drift_detected`` (CV des résidus KK > 0,5 sur > 20 %
  des fréquences) et la fraction de fréquences signalées ;
* ``eq_pvalue``      : p du test F de σ_r = σ_j (``ErrorStructure.equality_pvalue``) ;
* ``kk_conform``     : verdict KK du groupe (réplicats + moyenne, Bonferroni) ;
* ``voigt_chi2_p``   : p de χ²(dof) du measurement model final de chaque réplicat ;
* ``q_pvalue``       : Q de Cochran de chaque paramètre du fit Orazem (le pipeline n'alerte que
  sur le paramètre cible, ici Rct) ;
* ``raw_T``          : statistique EXPLORATOIRE (non implémentée dans l'application) —
  Σ_ω Σ_k (Z_k − Z̄)²/σ² sur les deux composantes, σ de la structure d'erreur, ν = 2N(n − 1).

Usage (depuis la racine du dépôt) ::

    OMP_NUM_THREADS=1 python docs/validation_derive/run_drift_study.py run  --out docs/validation_derive/results.jsonl
    python docs/validation_derive/run_drift_study.py report docs/validation_derive/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import stats  # noqa: E402

from circuit.parser import parse_circuit  # noqa: E402
from core.measurement_model import ErrorStructureUnavailable, analyze_replicates  # noqa: E402
from core.models import EISSpectrum  # noqa: E402
from core.validator import DRIFT_CV_THRESHOLD, validation_from_analysis  # noqa: E402
from fits.orazem_fit import FitOptions, ParameterSpec, fit_replicate_group  # noqa: E402
from tests.synthetic_data import RANDLES_EXPRESSION, Z_randles_reference, frequencies  # noqa: E402

# ── Données : Randles de l'Annexe A (tests/synthetic_data.py) ────────────────────────────────
RCT = 3000.0
BASE = dict(Re=200.0, Re_prime=20.0, Cb=1e-9, Qdl=2e-6, alpha=0.9, tau_d=0.5, Rct=RCT, R_D=0.3 * RCT)
N_POINTS = 40

# Spécifications du fit : celles de tests/test_orazem_fit.py (guess à 2× de la vérité pour Rct).
Z_FUNC, NAMES = parse_circuit(RANDLES_EXPRESSION)
SPECS = {
    "Re": ParameterSpec(100.0, 0.0, 1e5),
    "Re_prime": ParameterSpec(50.0, 0.0, 1e5),
    "Rct": ParameterSpec(1500.0, 0.0, 1e9),
    "R_D": ParameterSpec(300.0, 0.0, 1e7),
    "tau_d": ParameterSpec(0.2, 1e-6, 1e4),
    "Qdl": ParameterSpec(5e-6, 0.0, 1e-2),
    "alpha": ParameterSpec(0.8, 0.3, 1.0),
    "Cb": ParameterSpec(3e-9, 0.0, 1e-3),
}

#: Bruits : « orazem » = structure d'Orazem σ = α|Zj| + β|Zr − Re| + δ, même σ sur Re et Im
#: (ORAZEM_NOISE des tests × facteur) ; « relatif » = bruit relatif par composante de
#: ``noisy_arrays`` (σ_re = p|Zre|, σ_im = p|Zim| — viole σ_r = σ_j ; données de l'ERR-4).
NOISES = {
    "orazem_x0.5": ("orazem", 0.5),
    "orazem_x1": ("orazem", 1.0),
    "orazem_x2": ("orazem", 2.0),
    "relatif_0.5%": ("relatif", 0.005),
    "relatif_1%": ("relatif", 0.01),
}


def _sigma(noise, zre, zim):
    kind, level = NOISES[noise]
    if kind == "orazem":
        s = level * (0.004 * np.abs(zim) + 0.004 * np.abs(zre - BASE["Re"]) + 0.5)
        return s, s
    return level * np.abs(zre), level * np.abs(zim)


def make_group(noise, scenario, param, amplitude, n_rep, seed):
    """Réplicats d'un groupe. Temps : point i du réplicat k au rang k·N + i (balayage HF → BF).

    * ``stationnaire``    : tous les réplicats au même Z (bruit seul) ;
    * ``entre_balayages`` : ``param`` constant pendant chaque balayage, P_k = P·(1 + a·k/(n − 1)) ;
    * ``continue``        : ``param`` varie linéairement dans le temps sur toute la série,
      P(t) = P·(1 + a·t), t ∈ [0, 1] du premier point du 1er réplicat au dernier du n-ième
      (dérive PENDANT ET ENTRE les balayages).
    """
    f = frequencies(N_POINTS)
    omega = 2 * np.pi * f
    rng = np.random.default_rng(seed)
    reps = []
    for k in range(n_rep):
        if scenario == "stationnaire":
            Z = Z_randles_reference(omega, **BASE)
        elif scenario == "entre_balayages":
            Z = Z_randles_reference(omega, **dict(BASE, **{param: BASE[param] * (1 + amplitude * k / (n_rep - 1))}))
        elif scenario == "continue":
            t = (k * N_POINTS + np.arange(N_POINTS)) / (n_rep * N_POINTS - 1)
            Z = np.array([Z_randles_reference(omega[i:i + 1],
                                              **dict(BASE, **{param: BASE[param] * (1 + amplitude * t[i])}))[0]
                          for i in range(N_POINTS)])
        else:
            raise ValueError(scenario)
        zre, zim = Z.real, -Z.imag
        s_re, s_im = _sigma(noise, zre, zim)
        reps.append(EISSpectrum(label=f"r{k}", f=f, Zre=zre + rng.normal(0.0, s_re),
                                Zim=zim + rng.normal(0.0, s_im), concentration=0.0, step="probe",
                                n_points=N_POINTS))
    return reps


def _drift_fraction(kk_results, cv_threshold=DRIFT_CV_THRESHOLD):
    """Fraction de fréquences signalées par ``core.validator._detect_drift`` (même calcul)."""
    f_ref = kk_results[0].frequencies
    stack = np.array([np.interp(np.log10(f_ref), np.log10(kk.frequencies), kk.residuals_im)
                      for kk in kk_results])
    mean_res = np.mean(np.abs(stack), axis=0)
    std_res = np.std(stack, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(mean_res > 0.1, std_res / mean_res, 0.0)
    return float((cv > cv_threshold).mean())


def run_one(job):
    noise, scenario, param, amplitude, n_rep, seed = job
    t0 = time.time()
    reps = make_group(noise, scenario, param, amplitude, n_rep, seed)
    row = dict(noise=noise, scenario=scenario, param=param, amplitude=amplitude, n_rep=n_rep,
               seed=seed, n_points=N_POINTS)
    try:
        an = analyze_replicates(reps, label="g")
    except ErrorStructureUnavailable as exc:
        row.update(status="error_structure_unavailable", reason=exc.reason, seconds=time.time() - t0)
        return row
    vr = validation_from_analysis(an, "g")
    es = an.error_structure
    frac = _drift_fraction(vr.replicates)
    # Statistique exploratoire : dispersion BRUTE inter-réplicats rapportée à σ(ω) du MM.
    s_re, s_im = es.sigmas(an.mean_Zre, an.mean_Zim)
    zr = np.array([kk.Zre for kk in an.kk_replicates])
    zj = np.array([kk.Zim for kk in an.kk_replicates])
    T = float(np.sum((zr - zr.mean(0)) ** 2 / s_re ** 2) + np.sum((zj - zj.mean(0)) ** 2 / s_im ** 2))
    nu_T = 2 * zr.shape[1] * (n_rep - 1)
    row.update(
        status="ok",
        detect_drift=bool(vr.drift_detected), drift_fraction=frac,
        drift_warning=vr.drift_warning,
        eq_pvalue=float(es.equality_pvalue), equal_re_im=bool(es.equal_re_im),
        kk_conform=bool(an.kk_conform), kk_mean_conform=bool(an.kk_mean.conform),
        kk_rep_conform=[bool(k.conform) for k in an.kk_replicates],
        voigt_chi2_p=[float(stats.chi2.sf(vm.chi2, vm.dof)) for vm in an.estimate.voigt_models],
        voigt_K=[int(vm.n_elements) for vm in an.estimate.voigt_models],
        sigma_dof=int(es.dof), raw_T=T, raw_nu=nu_T,
    )
    og = fit_replicate_group(Z_FUNC, NAMES, reps, an, SPECS, "Rct", options=FitOptions())
    row.update(
        n_converged=sum(fr.converged for fr in og.replicate_fits),
        q_pvalue={p: (None if not np.isfinite(a.q_pvalue) else float(a.q_pvalue)) for p, a in og.aggregate.items()},
        rct=[float(fr.params["Rct"]) for fr in og.replicate_fits],
        rct_std=[float(fr.params_std["Rct"]) for fr in og.replicate_fits],
        cochran_warning=any("Cochran" in w for w in og.warnings),
        seconds=time.time() - t0,
    )
    return row


def build_jobs():
    jobs = []
    # H0 : réplicats stationnaires, 100 groupes par bruit (n = 3), + n = 5 au bruit nominal.
    for i, noise in enumerate(NOISES):
        jobs += [(noise, "stationnaire", None, 0.0, 3, 10_000 * (i + 1) + s) for s in range(100)]
    jobs += [("orazem_x1", "stationnaire", None, 0.0, 5, 90_000 + s) for s in range(100)]
    # H1 : dérive de Rct entre balayages, 40 groupes par amplitude.
    for i, noise in enumerate(("orazem_x1", "orazem_x2", "relatif_0.5%")):
        for j, a in enumerate((0.005, 0.01, 0.02, 0.05, 0.10, 0.20)):
            jobs += [(noise, "entre_balayages", "Rct", a, 3, 200_000 + 10_000 * i + 1000 * j + s)
                     for s in range(40)]
    # H1 : dérive continue de Rct (pendant ET entre les balayages).
    for i, noise in enumerate(("orazem_x1", "relatif_0.5%")):
        for j, a in enumerate((0.01, 0.02, 0.05, 0.10, 0.20)):
            jobs += [(noise, "continue", "Rct", a, 3, 400_000 + 10_000 * i + 1000 * j + s)
                     for s in range(40)]
    # H1 : dérive d'un paramètre NON cible (Qdl) entre balayages.
    for j, a in enumerate((0.02, 0.05, 0.10, 0.20)):
        jobs += [("orazem_x1", "entre_balayages", "Qdl", a, 3, 600_000 + 1000 * j + s) for s in range(40)]
    return jobs


def cmd_run(args):
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["noise"], r["scenario"], r["param"], r["amplitude"], r["n_rep"], r["seed"]))
    jobs = [j for j in build_jobs() if j not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    print(f"{len(jobs)} groupes à calculer ({len(done)} déjà faits)", flush=True)
    with Pool(args.workers) as pool, out.open("a", encoding="utf-8") as fh:
        for n, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), 1):
            row["env"] = env
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(f"{n}/{len(jobs)}", flush=True)


# ── Rapport ───────────────────────────────────────────────────────────────────────────────────

ALPHA = 0.05


def verdicts(r):
    """{critère: True si « dérive / incohérence » signalée}."""
    n = r["n_rep"]
    q = r["q_pvalue"].get("Rct")
    q_any = [p for p in r["q_pvalue"].values() if p is not None]
    return {
        "detect_drift": r["detect_drift"],
        "eq_rejet": r["eq_pvalue"] < ALPHA,
        "kk_non_conforme": not r["kk_conform"],
        "voigt_chi2": min(r["voigt_chi2_p"]) < ALPHA / n,
        "cochran_Rct": q is not None and q < ALPHA,
        "cochran_tout_param": bool(q_any) and min(q_any) < ALPHA / len(q_any),
        "explo_T_chi2": stats.chi2.sf(r["raw_T"], r["raw_nu"]) < ALPHA,
        "explo_T_F": stats.f.sf(r["raw_T"] / r["raw_nu"], r["raw_nu"], r["sigma_dof"]) < ALPHA,
    }


CRITERIA = ["detect_drift", "eq_rejet", "kk_non_conforme", "voigt_chi2", "cochran_Rct",
            "cochran_tout_param", "explo_T_chi2", "explo_T_F"]


def _ci(k, n):
    """Intervalle de Wilson à 95 % d'une proportion k/n (en %)."""
    if n == 0:
        return float("nan"), float("nan")
    z = 1.959964
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def cmd_report(args):
    rows, seen = [], set()
    for line in Path(args.path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            key = (r["noise"], r["scenario"], r["param"], r["amplitude"], r["n_rep"], r["seed"])
            if key not in seen:                     # reprise après interruption : un groupe = une ligne
                seen.add(key)
                rows.append(r)
    ok =[r for r in rows if r["status"] == "ok"]
    ko = [r for r in rows if r["status"] != "ok"]
    print(f"{len(rows)} groupes, {len(ko)} sans structure d'erreur\n")
    keys = []
    for r in ok:
        k = (r["scenario"], r["param"], r["noise"], r["n_rep"], r["amplitude"])
        if k not in keys:
            keys.append(k)
    head = "| scénario | bruit | n | ampl. | groupes | " + " | ".join(CRITERIA) + " |"
    print(head)
    print("|" + "---|" * (5 + len(CRITERIA)))
    for k in keys:
        sub = [r for r in ok if (r["scenario"], r["param"], r["noise"], r["n_rep"], r["amplitude"]) == k]
        vs = [verdicts(r) for r in sub]
        cells = []
        for c in CRITERIA:
            m = sum(v[c] for v in vs)
            lo, hi = _ci(m, len(vs))
            cells.append(f"{100 * m / len(vs):.0f} % [{lo:.0f}-{hi:.0f}]" if args.ci else f"{m}/{len(vs)}")
        name = k[0] + (f" {k[1]}" if k[1] else "")
        print(f"| {name} | {k[2]} | {k[3]} | {100 * k[4]:g} % | {len(sub)} | " + " | ".join(cells) + " |")
    print()
    # Diagnostics complémentaires.
    for k in keys:
        if k[0] != "stationnaire":
            continue
        sub = [r for r in ok if (r["scenario"], r["param"], r["noise"], r["n_rep"], r["amplitude"]) == k]
        fr = np.array([r["drift_fraction"] for r in sub])
        T = np.array([r["raw_T"] / r["raw_nu"] for r in sub])
        conv = np.mean([r["n_converged"] == r["n_rep"] for r in sub])
        print(f"H0 {k[2]} n={k[3]} : fraction _detect_drift médiane {100 * np.median(fr):.0f} % "
              f"(min {100 * fr.min():.0f}, max {100 * fr.max():.0f}) ; T/ν moyenne {T.mean():.3f} "
              f"(é.-t. {T.std(ddof=1):.3f}, attendu sous χ² 1 ± {np.sqrt(2 / sub[0]['raw_nu']):.3f}) ; "
              f"tous réplicats convergés {100 * conv:.0f} %")
    secs = [r["seconds"] for r in ok]
    print(f"\ndurée par groupe : médiane {np.median(secs):.1f} s, max {max(secs):.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--out", default=str(Path(__file__).with_name("results.jsonl")))
    a.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("report")
    b.add_argument("path", nargs="?", default=str(Path(__file__).with_name("results.jsonl")))
    b.add_argument("--ci", action="store_true", help="pourcentages avec IC de Wilson 95 %")
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    main()
