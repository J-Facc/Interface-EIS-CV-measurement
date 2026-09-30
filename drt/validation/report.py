# -*- coding: utf-8 -*-
"""Tableaux Markdown de ``drt/VALIDATION_REGLAGES.md`` à partir des JSONL de validation.

Usage ::

    python drt/validation/report.py drt/validation/results_optimize.jsonl
    python drt/validation/report.py drt/validation/results_sample.jsonl
    python drt/validation/report.py --engine drt/validation/results_engine.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict

SETTING_ORDER = ["default", "init_from_ridge", "nonneg", "nonneg+ridge"]
REL_OK = 0.01  # |Rp − vrai|/vrai ≤ 1 % : « correct » pour le décompte

# ── Périmètre de la matrice HMC (décision utilisateur du 2026-09-30, VALIDATION_REGLAGES.md §3.0)
#: (a) réglage amont `default` : 2-3 scénarios seulement, pour confirmer son échec en HMC — un
#:     par famille où le MAP amont échoue (grille nue, bruit, Randles) ;
HMC_SCOPE_DEFAULT_CASES = ("A3-60pts-1e5-1e-1", "A3-60pts-bruit0.5%", "A4-Randles-Rct3000")
#: (b) les deux candidats réels : tous les scénarios ;
HMC_SCOPE_FULL_SETTINGS = ("init_from_ridge", "nonneg")
#: (c) `nonneg+ridge` : seulement là où init_from_ridge et nonneg DIVERGENT en MAP
#:     (l'un correct, l'autre non) — calculé par :func:`map_divergent_cases`.


def _correct(r):
    return r.get("status") == "ok" and r["rp"] > 0 and abs(r["rp_rel_err"]) <= REL_OK


def map_divergent_cases(opt_rows):
    """Scénarios où `init_from_ridge` et `nonneg` ne sont pas tous deux corrects, ou tous deux faux, en MAP."""
    by = {(r["case"], r["setting"]): r for r in opt_rows}
    cases = list(OrderedDict.fromkeys(r["case"] for r in opt_rows))
    return [c for c in cases if (c, "init_from_ridge") in by and (c, "nonneg") in by
            and _correct(by[(c, "init_from_ridge")]) != _correct(by[(c, "nonneg")])]


def hmc_scope(sample_rows, opt_rows):
    """Essais HMC retenus dans la matrice comparative (règle (a)-(c) ci-dessus)."""
    combo = set(map_divergent_cases(opt_rows))
    keep = []
    for r in sample_rows:
        s, c = r["setting"], r["case"]
        if (s == "default" and c in HMC_SCOPE_DEFAULT_CASES) or s in HMC_SCOPE_FULL_SETTINGS \
                or (s == "nonneg+ridge" and c in combo):
            keep.append(r)
    return keep


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _fmt_rp(r):
    if r.get("status") != "ok":
        return "ERREUR"
    ok = abs(r["rp_rel_err"]) <= REL_OK and r["rp"] > 0
    return f"{r['rp']:.1f} ({r['rp_rel_err'] * 100:+.1f} %){'' if ok else ' ❌'}"


def matrix_tables(rows):
    cases = list(OrderedDict.fromkeys(r["case"] for r in rows))
    by = {(r["case"], r["setting"]): r for r in rows}
    mode = rows[0]["mode"]
    out = []
    out.append(f"#### Rp (écart à la vraie valeur) — mode `{mode}`\n")
    out.append("| Cas | Rp vrai | " + " | ".join(f"`{s}`" for s in SETTING_ORDER) + " |")
    out.append("|---|---|" + "---|" * len(SETTING_ORDER))
    for c in cases:
        rp_true = next(r["rp_true"] for r in rows if r["case"] == c)
        cells = [_fmt_rp(by[(c, s)]) if (c, s) in by else "—" for s in SETTING_ORDER]
        out.append(f"| {c} | {rp_true:.0f} | " + " | ".join(cells) + " |")
    out.append("")
    out.append(f"#### γ_min (Ω) / erreur de reconstruction max (%) — mode `{mode}`\n")
    out.append("| Cas | " + " | ".join(f"`{s}`" for s in SETTING_ORDER) + " |")
    out.append("|---|" + "---|" * len(SETTING_ORDER))
    for c in cases:
        cells = []
        for s in SETTING_ORDER:
            r = by.get((c, s))
            cells.append("—" if r is None or r["status"] != "ok"
                         else f"{r['gamma_min']:.2f} / {r['recon_max'] * 100:.2f}")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("")
    if mode == "sample":
        out.append("#### Diagnostics HMC : R-hat max · divergences · itérations à profondeur max · "
                   "ESS bulk min · Rp vrai ∈ IC 95 % ?\n")
        out.append("| Cas | " + " | ".join(f"`{s}`" for s in SETTING_ORDER) + " |")
        out.append("|---|" + "---|" * len(SETTING_ORDER))
        for c in cases:
            cells = []
            for s in SETTING_ORDER:
                r = by.get((c, s))
                if r is None or r["status"] != "ok":
                    cells.append("—")
                    continue
                lo, hi = r["rp_ci95"]
                inside = "oui" if lo <= r["rp_true"] <= hi else "**non**"
                cells.append(f"{r['rhat_max_all']:.3f} · {r['divergences']} · {r['max_treedepth_hits']} · "
                             f"{r['ess_bulk_min_params']:.0f} · {inside}")
            out.append(f"| {c} | " + " | ".join(cells) + " |")
        out.append("")
    # Décompte
    out.append(f"#### Décompte — mode `{mode}` (correct = Rp > 0 et |écart| ≤ {REL_OK * 100:.0f} %)\n")
    out.append("| Réglage | Rp correct | Rp < 0 | erreur recon. max > 10 % |" +
               (" R-hat > 1,05 | divergences > 0 | prof. max > 0 | Rp vrai hors IC 95 % |" if mode == "sample" else ""))
    out.append("|---|---|---|---|" + ("---|---|---|---|" if mode == "sample" else ""))
    for s in SETTING_ORDER:
        rs = [r for r in rows if r["setting"] == s and r["status"] == "ok"]
        n = len(rs)
        if n == 0:
            continue
        ok = sum(1 for r in rs if r["rp"] > 0 and abs(r["rp_rel_err"]) <= REL_OK)
        neg = sum(1 for r in rs if r["rp"] <= 0)
        rec = sum(1 for r in rs if r["recon_max"] > 0.10)
        line = f"| `{s}` | {ok}/{n} | {neg}/{n} | {rec}/{n} |"
        if mode == "sample":
            rh = sum(1 for r in rs if r["rhat_max_all"] > 1.05)
            dv = sum(1 for r in rs if r["divergences"] > 0)
            td = sum(1 for r in rs if r["max_treedepth_hits"] > 0)
            ci = sum(1 for r in rs if not (r["rp_ci95"][0] <= r["rp_true"] <= r["rp_ci95"][1]))
            line += f" {rh}/{n} | {dv}/{n} | {td}/{n} | {ci}/{n} |"
        out.append(line)
    return "\n".join(out)


def engine_table(rows):
    out = ["| Cas | Réglage | graine | Rp (écart) | aire hors fenêtre BF / HF | Rct (IC 95 %) | τ_Rct | "
           "cible | erreur recon. max | R-hat · div · prof. max · ESS bulk/tail | alertes | durée |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") != "ok":
            out.append(f"| {r['case']} | `{r['setting']}` | {r['stan_seed']} | ERREUR {r.get('error', '')} |"
                       + " |" * 8)
            continue
        s = r.get("sampler") or {}
        err = (r["rp"] - r["rp_true"]) / r["rp_true"] * 100
        ci = r.get("rct_ci95") or [float("nan")] * 2
        tgt = f"{r['rct_target']:.0f}" + (f" @ {r['tau_target']:.0e} s" if r.get("tau_target") else "")
        diag = "—"
        if "rhat_max" in s:
            diag = (f"{s['rhat_max']:.3f} · {s['divergences']} · {s['max_treedepth_hits']} · "
                    f"{s['ess_bulk_min']:.0f}/{s['ess_tail_min']:.0f}")
        out.append(
            f"| {r['case']} | `{r['setting']}` | {r['stan_seed']} | {r['rp']:.1f} ({err:+.2f} %) | "
            f"{r['area_above']:.1f} / {r['area_below']:.1f} | {r['rct']:.1f} ({ci[0]:.1f}–{ci[1]:.1f}) | "
            f"{r['tau_rct']:.2g} s | {tgt} | {r['recon_max'] * 100:.2f} % | {diag} | "
            f"{', '.join(r['alerts']) or 'aucune'} | {r['time_s']:.0f} s |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--engine", action="store_true")
    ap.add_argument("--hmc-scope", metavar="RESULTS_OPTIMIZE_JSONL", default=None,
                    help="restreint une matrice HMC au périmètre (a)-(c) ; le fichier MAP sert à (c)")
    a = ap.parse_args(argv)
    rows = _load(a.path)
    if a.hmc_scope:
        opt = _load(a.hmc_scope)
        print(f"<!-- périmètre HMC : default sur {list(HMC_SCOPE_DEFAULT_CASES)} ; "
              f"{list(HMC_SCOPE_FULL_SETTINGS)} partout ; nonneg+ridge sur {map_divergent_cases(opt)} ; "
              f"{len(hmc_scope(rows, opt))}/{len(rows)} essais retenus -->\n")
        rows = hmc_scope(rows, opt)
    print(engine_table(rows) if a.engine else matrix_tables(rows))


if __name__ == "__main__":
    main()
