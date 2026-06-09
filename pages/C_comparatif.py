"""Page C — Comparaison des 6 méthodes d'extraction de concentration.

Orchestre l'entrée des données (EIS + CV), le calcul du rapport comparatif
via comparison/report.py et l'affichage structuré en sections dépliables.
Aucun calcul scientifique inline — tout passe par comparison/report.py.
"""

import io
import numpy as np
import plotly.io as pio
import streamlit as st

from core.config import load_config, config_to_dict
from core.loader import load_spectrum, average_replicates
from core.cv_loader import load_cv_file, average_cv_replicates
from fits.registry import get_model
from comparison.report import compute_full_report
from comparison.plots import plot_pls_loadings
from ui.data_input import render_data_input

_CONFIG = None


def _get_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = config_to_dict(load_config())
    return _CONFIG


# ---------------------------------------------------------------------------
# Textes d'interprétation (section 9 de PROJECT.md) — dict indexé par clé
# ---------------------------------------------------------------------------

_INTERPRET = {
    "rmsecv_rmsep": """\
**Erreur quadratique moyenne** entre concentration prédite et vraie, en décades de log([c]).

| Valeur | Signification |
|--------|---------------|
| 0.3 décade | erreur typique d'un facteur 2 sur la concentration |
| 0.5 décade | erreur typique d'un facteur 3 |
| 1.0 décade | erreur d'un ordre de grandeur — modèle peu utile |

**Comparer RMSECV et RMSEP :**
- RMSEP ≈ RMSECV → le modèle généralise bien
- RMSEP >> RMSECV → surapprentissage : le modèle a mémorisé les données de calibration
- Écart acceptable : RMSEP < 1.5 × RMSECV""",

    "bias": """\
**Erreur moyenne signée :** mean(log(c_pred) − log(c_true))
- Positif → surestimation systématique. Négatif → sous-estimation.

| Valeur | Signification |
|--------|---------------|
| \|biais\| < 0.2 décade | acceptable |
| \|biais\| > 0.5 décade | erreur systématique significative à investiguer |

**Causes fréquentes :**
- Biais positif aux hautes concentrations → saturation du signal non modélisée
- Biais négatif aux basses concentrations → signal proche du bruit (LOD atteint)
- Biais constant sur toute la gamme → erreur d'étalonnage de la solution mère""",

    "rpd": """\
**RPD = std(log([c])) / RMSEP**
Normalise la performance par la difficulté de la tâche (étendue de la gamme).

| RPD | Interprétation |
|-----|----------------|
| < 2 | modèle non informatif |
| 2–3 | screening grossier uniquement |
| 3–5 | quantification acceptable |
| > 5 | modèle excellent |

Utilité : permet de comparer des méthodes sur des gammes légèrement différentes
sans que la gamme plus large avantage artificiellement une méthode.""",

    "sigma_intra": """\
**Écart-type des prédictions** sur les réplicats d'une même électrode, à concentration fixée.
C'est le plancher incompressible : bruit instrumental + dérive de l'interface.

- **σ_intra faible** → mesure répétable, signal stable
- **σ_intra élevé** → instabilité de l'interface (désorption, dérive thermique),
  ou paramètre d'extraction sensible au bruit (ex : Rct par fit paramétrique sur spectre bruité)

Comparer σ_intra entre méthodes : une méthode avec σ_intra plus faible
est intrinsèquement plus robuste au bruit, indépendamment de sa calibration.""",

    "sigma_inter": """\
**Différence entre les prédictions moyennes des deux électrodes** à concentration identique.
Dans le dispositif multiplexé, les deux électrodes voient exactement la même solution :
toute différence est donc de la variabilité de fonctionnalisation pure.

- **σ_inter avant normalisation** → état brut, dépend de la qualité du SAM
- **σ_inter après normalisation** → ce qui reste après correction par le probe

Si σ_inter_norm ≈ σ_inter_brut : la normalisation par le probe ne corrige pas cette variabilité.
Si σ_inter_norm << σ_inter_brut : la normalisation est efficace.""",

    "reduction_factor": """\
**Facteur = σ_inter_brut / σ_inter_norm**

| Facteur | Signification |
|---------|---------------|
| 1 | normalisation sans effet |
| 3 | la normalisation divise par 3 la variabilité |
| 10 | normalisation très efficace |

Un facteur élevé valide que le spectre probe encode l'état de fonctionnalisation.
Un facteur faible suggère une autre source de variabilité que le probe ne capture pas.""",

    "dm_test": """\
**Compare statistiquement les erreurs de deux méthodes** sur les mêmes points de validation.
H₀ : les deux méthodes ont la même performance prédictive.
p < 0.05 → différence statistiquement significative.

| Résultat | Lecture |
|----------|---------|
| p > 0.05, RMSEP similaires | méthodes équivalentes — préférer la plus simple |
| p > 0.05, RMSEP différents | différence peut être due au hasard (puissance faible avec peu de points) |
| p < 0.05 | différence réelle — examiner biais et loadings |

⚠️ Avec peu de points de validation, une différence de RMSEP de 0.3 décade
peut ne pas être détectable statistiquement.""",

    "ci95": """\
**Intervalle de confiance à 95 %** — exprimé en facteur multiplicatif k : [c/k, c×k]

| k | Signification |
|---|---------------|
| 2 | facteur 2 de part et d'autre — bon |
| 5 | facteur 5 — acceptable selon l'application |
| 10 | un ordre de grandeur — limite d'utilité |

C'est la métrique la plus directement utile en pratique : elle dit ce que
tu peux affirmer sur la concentration réelle d'une mesure inconnue.""",

    "loadings": """\
**Les loadings indiquent quelles fréquences (EIS) ou quels potentiels (CV)
contribuent le plus à la prédiction de la concentration.**

**Pour EIS :**
- Loading concentré autour d'une fréquence f* → la méthode exploite principalement
  le demi-cercle à cette fréquence, cohérent avec Rct
  (fréquence typique biosenseurs ADN/Pt : 1–100 Hz)
- Loading étalé sur toute la gamme → exploitation de la forme globale, moins interprétable
- Loading bruité sans structure → risque de surapprentissage

**Pour CV :**
- Loading concentré autour des potentiels de pic redox → valide la méthode scalaire A1
- Loading sur d'autres zones → la CV contient de l'information complémentaire au pic redox""",

    "scree": """\
**Part de la variance des spectres expliquée par chaque composante latente PLS.**

| Variance C1 | Interprétation |
|-------------|----------------|
| > 80 % | signal dominé par un seul processus (variation globale d'amplitude) |
| C2 > 5 % | second processus indépendant (changement de forme spectrale) |
| C3 < 2 % | k=2 suffit, k=3 risque de surapprendre |

Variance X élevée mais RMSECV grand : les composantes capturent la structure
des spectres mais pas leur corrélation avec la concentration — revoir la normalisation.""",
}


# ---------------------------------------------------------------------------
# Chargement des fichiers — construit session_data pour compute_full_report()
# ---------------------------------------------------------------------------

def _read_file(uploaded_file):
    """Lit un UploadedFile, retourne (bytes, name) ou (None, None)."""
    if uploaded_file is None:
        return None, None
    content = uploaded_file.read()
    uploaded_file.seek(0)
    return content, uploaded_file.name


def _fit_eis_spectrum(spectrum, config, model_names=("randles_full", "drt_fit")):
    """Ajuste un spectre EIS avec les modèles demandés. Retourne dict {name: FitResult}."""
    results = {}
    for name in model_names:
        try:
            model = get_model(name)
            fr = model.fit(spectrum, config)
            results[name] = fr
        except Exception as exc:
            st.warning(f"Fit {name} échoué sur {spectrum.label} : {exc}")
    return results


def _load_session_data(data: dict, progress_cb) -> dict | None:
    """
    Charge tous les fichiers uploadés et construit le dict session_data
    attendu par compute_full_report().

    Parameters
    ----------
    data : dict retourné par render_data_input(mode='both')
    progress_cb : callable(float, str) pour mettre à jour la barre Streamlit

    Returns
    -------
    dict session_data ou None en cas d'erreur fatale
    """
    config    = _get_config()
    concs     = np.asarray(data["concentrations"], dtype=float)
    n_conc    = len(concs)
    n_elec    = data["n_electrodes"]
    calib     = data["calibration"]
    eis_block = calib["eis"]
    cv_block  = calib["cv"]

    # Grilles communes
    freq_grid = np.logspace(0, 5, 50)   # 1 Hz → 100 kHz, 50 points
    pot_grid  = np.linspace(-0.3, 0.7, 60)

    # Résultats par électrode
    out = {
        "eis_rct_fit": {f"e{e}": [] for e in range(1, n_elec + 1)},
        "eis_rct_drt": {f"e{e}": [] for e in range(1, n_elec + 1)},
        "cv_delta_I":  {f"e{e}": [] for e in range(1, n_elec + 1)},
        "eis_spectra": {f"e{e}": [] for e in range(1, n_elec + 1)},
        "cv_spectra":  {f"e{e}": [] for e in range(1, n_elec + 1)},
        "probe_rct_fit": None,
        "probe_rct_drt": None,
        "probe_delta_I": None,
    }

    # ---- Probe EIS (moyenne de toutes les électrodes, première concentration) ----
    progress_cb(0.02, "Chargement des probes EIS…")
    probe_eis_spectra = []
    for e in range(1, n_elec + 1):
        probe_list = eis_block.get(f"probe_{e}", [])
        for ci, pf in enumerate(probe_list):
            content, name = _read_file(pf)
            if content is None:
                continue
            try:
                sp = load_spectrum(content, name, concentration=0.0, step="probe", config=config)
                probe_eis_spectra.append(sp)
            except Exception as exc:
                st.warning(f"Probe EIS e{e} c{ci+1} : {exc}")

    if not probe_eis_spectra:
        st.error("Aucun fichier probe EIS valide — impossible de normaliser.")
        return None

    probe_eis = average_replicates(probe_eis_spectra)
    probe_fits = _fit_eis_spectrum(probe_eis, config)
    out["probe_rct_fit"] = probe_fits.get("randles_full") and probe_fits["randles_full"].Rct
    out["probe_rct_drt"] = probe_fits.get("drt_fit") and probe_fits["drt_fit"].Rct
    if out["probe_rct_fit"] is None:
        out["probe_rct_fit"] = 0.0
    if out["probe_rct_drt"] is None:
        out["probe_rct_drt"] = 0.0

    # Spectre probe pour normalisation PLS
    Zre_probe = np.interp(freq_grid, probe_eis.f[::-1], probe_eis.Zre[::-1])
    Zim_probe = np.interp(freq_grid, probe_eis.f[::-1], probe_eis.Zim[::-1])

    # ---- Probe CV ----
    progress_cb(0.06, "Chargement des probes CV…")
    probe_cv_scans = []
    for e in range(1, n_elec + 1):
        probe_list = cv_block.get(f"probe_{e}", [])
        for ci, pf in enumerate(probe_list):
            content, name = _read_file(pf)
            if content is None:
                continue
            try:
                sc = load_cv_file(content, name, concentration=0.0, step="probe")
                probe_cv_scans.append(sc)
            except Exception as exc:
                st.warning(f"Probe CV e{e} c{ci+1} : {exc}")

    if not probe_cv_scans:
        st.error("Aucun fichier probe CV valide.")
        return None

    probe_cv = average_cv_replicates(probe_cv_scans)
    probe_I_pic = float(np.max(probe_cv.I))   # pic d'oxydation
    out["probe_delta_I"] = probe_I_pic if abs(probe_I_pic) > 1e-15 else 1.0
    I_probe_grid = np.interp(pot_grid, probe_cv.E, probe_cv.I)

    # ---- Boucle concentrations × électrodes ----
    n_total_steps = n_conc * n_elec
    step_idx = 0

    for ci, conc in enumerate(concs):
        for e in range(1, n_elec + 1):
            step_idx += 1
            frac = 0.08 + 0.80 * step_idx / n_total_steps
            progress_cb(frac, f"Analyse {conc:.1e} M — électrode {e}…")
            ek = f"e{e}"

            # ---- EIS ----
            rep_list = (eis_block.get(f"electrode_{e}") or [])
            rep_files = rep_list[ci] if ci < len(rep_list) else []
            eis_spectra_rep = []
            for rf in (rep_files or []):
                content, name = _read_file(rf)
                if content is None:
                    continue
                try:
                    sp = load_spectrum(content, name, concentration=conc, step="hybridization", config=config)
                    eis_spectra_rep.append(sp)
                except Exception as exc:
                    st.warning(f"EIS e{e} c{ci+1} r : {exc}")

            if eis_spectra_rep:
                sp_avg = average_replicates(eis_spectra_rep) if len(eis_spectra_rep) > 1 else eis_spectra_rep[0]
                fits_h = _fit_eis_spectrum(sp_avg, config)

                rct_fit = fits_h.get("randles_full")
                out["eis_rct_fit"][ek].append(float(rct_fit.Rct) if rct_fit else np.nan)

                rct_drt = fits_h.get("drt_fit")
                out["eis_rct_drt"][ek].append(float(rct_drt.Rct) if rct_drt else np.nan)

                # Features spectrales PLS : [Zre_norm, Zim_norm] interpolées sur freq_grid
                Zre_h = np.interp(freq_grid, sp_avg.f[::-1], sp_avg.Zre[::-1])
                Zim_h = np.interp(freq_grid, sp_avg.f[::-1], sp_avg.Zim[::-1])
                with np.errstate(invalid="ignore", divide="ignore"):
                    Zre_norm = np.where(np.abs(Zre_probe) > 1e-10,
                                        (Zre_h - Zre_probe) / np.abs(Zre_probe), 0.0)
                    Zim_norm = np.where(np.abs(Zim_probe) > 1e-10,
                                        (Zim_h - Zim_probe) / np.abs(Zim_probe), 0.0)
                out["eis_spectra"][ek].append(np.concatenate([Zre_norm, Zim_norm]))
            else:
                out["eis_rct_fit"][ek].append(np.nan)
                out["eis_rct_drt"][ek].append(np.nan)
                out["eis_spectra"][ek].append(np.zeros(len(freq_grid) * 2))

            # ---- CV ----
            cv_rep_list = (cv_block.get(f"electrode_{e}") or [])
            cv_rep_files = cv_rep_list[ci] if ci < len(cv_rep_list) else []
            cv_scans_rep = []
            for rf in (cv_rep_files or []):
                content, name = _read_file(rf)
                if content is None:
                    continue
                try:
                    sc = load_cv_file(content, name, concentration=conc, step="hybridization")
                    cv_scans_rep.append(sc)
                except Exception as exc:
                    st.warning(f"CV e{e} c{ci+1} r : {exc}")

            if cv_scans_rep:
                cv_avg = average_cv_replicates(cv_scans_rep) if len(cv_scans_rep) > 1 else cv_scans_rep[0]
                I_pic_h = float(np.max(cv_avg.I))
                out["cv_delta_I"][ek].append(I_pic_h)

                I_h_grid = np.interp(pot_grid, cv_avg.E, cv_avg.I)
                with np.errstate(invalid="ignore", divide="ignore"):
                    I_norm = np.where(np.abs(I_probe_grid) > 1e-15,
                                      (I_h_grid - I_probe_grid) / np.abs(I_probe_grid), 0.0)
                out["cv_spectra"][ek].append(I_norm)
            else:
                out["cv_delta_I"][ek].append(np.nan)
                out["cv_spectra"][ek].append(np.zeros(len(pot_grid)))

    # ---- Données de validation ----
    val_out = None
    val_data = data.get("validation")
    if val_data:
        progress_cb(0.88, "Chargement des données de validation…")
        val_out = _load_validation_data(val_data, config, freq_grid, pot_grid,
                                        Zre_probe, Zim_probe, I_probe_grid,
                                        out["probe_rct_fit"], out["probe_rct_drt"])

    progress_cb(0.95, "Assemblage du rapport…")

    session_data = {
        "concentrations":  concs.tolist(),
        "n_electrodes":    n_elec,
        "n_replicats":     data["n_replicats"],
        "eis_rct_fit":     out["eis_rct_fit"],
        "eis_rct_drt":     out["eis_rct_drt"],
        "cv_delta_I":      out["cv_delta_I"],
        "eis_spectra":     out["eis_spectra"],
        "cv_spectra":      out["cv_spectra"],
        "freq_grid":       freq_grid,
        "pot_grid":        pot_grid,
        "probe_rct_fit":   out["probe_rct_fit"],
        "probe_rct_drt":   out["probe_rct_drt"],
        "probe_delta_I":   out["probe_delta_I"],
        "validation":      val_out,
        "theme_mode":      st.session_state.get("theme_mode", "light"),
        # Stocker les scans bruts pour les figures Nyquist et voltammogrammes
        "_probe_eis":      probe_eis,
        "_probe_cv":       probe_cv,
    }
    return session_data


def _load_validation_data(val_data, config, freq_grid, pot_grid,
                           Zre_probe, Zim_probe, I_probe_grid,
                           probe_rct_fit, probe_rct_drt):
    """Charge les données de validation dans le même format que les données de calibration."""
    val_concs = np.asarray(val_data["concentrations"], dtype=float)
    n_val = len(val_concs)
    n_elec = len([k for k in val_data.get("eis", {}) if k.startswith("electrode_")])
    n_elec = max(n_elec, 1)

    eis_block = val_data.get("eis", {})
    cv_block  = val_data.get("cv", {})

    out = {
        "eis_rct_fit": {f"e{e}": [] for e in range(1, n_elec + 1)},
        "eis_rct_drt": {f"e{e}": [] for e in range(1, n_elec + 1)},
        "cv_delta_I":  {f"e{e}": [] for e in range(1, n_elec + 1)},
        "eis_spectra": {f"e{e}": [] for e in range(1, n_elec + 1)},
        "cv_spectra":  {f"e{e}": [] for e in range(1, n_elec + 1)},
    }

    for ci, conc in enumerate(val_concs):
        for e in range(1, n_elec + 1):
            ek = f"e{e}"

            # EIS
            rep_files = (eis_block.get(f"electrode_{e}") or [[]])[ci] if ci < len(eis_block.get(f"electrode_{e}", [])) else []
            eis_reps = []
            for rf in (rep_files or []):
                content, name = _read_file(rf)
                if content:
                    try:
                        eis_reps.append(load_spectrum(content, name, conc, "hybridization", config))
                    except Exception:
                        pass
            if eis_reps:
                sp = average_replicates(eis_reps) if len(eis_reps) > 1 else eis_reps[0]
                fits_h = _fit_eis_spectrum(sp, config)
                rct_fit = fits_h.get("randles_full")
                rct_drt = fits_h.get("drt_fit")
                out["eis_rct_fit"][ek].append(float(rct_fit.Rct) if rct_fit else np.nan)
                out["eis_rct_drt"][ek].append(float(rct_drt.Rct) if rct_drt else np.nan)
                Zre_h = np.interp(freq_grid, sp.f[::-1], sp.Zre[::-1])
                Zim_h = np.interp(freq_grid, sp.f[::-1], sp.Zim[::-1])
                with np.errstate(invalid="ignore", divide="ignore"):
                    Zre_n = np.where(np.abs(Zre_probe) > 1e-10, (Zre_h - Zre_probe) / np.abs(Zre_probe), 0.0)
                    Zim_n = np.where(np.abs(Zim_probe) > 1e-10, (Zim_h - Zim_probe) / np.abs(Zim_probe), 0.0)
                out["eis_spectra"][ek].append(np.concatenate([Zre_n, Zim_n]))
            else:
                out["eis_rct_fit"][ek].append(np.nan)
                out["eis_rct_drt"][ek].append(np.nan)
                out["eis_spectra"][ek].append(np.zeros(len(freq_grid) * 2))

            # CV
            cv_files = (cv_block.get(f"electrode_{e}") or [[]])[ci] if ci < len(cv_block.get(f"electrode_{e}", [])) else []
            cv_reps = []
            for rf in (cv_files or []):
                content, name = _read_file(rf)
                if content:
                    try:
                        cv_reps.append(load_cv_file(content, name, conc, "hybridization"))
                    except Exception:
                        pass
            if cv_reps:
                sc = average_cv_replicates(cv_reps) if len(cv_reps) > 1 else cv_reps[0]
                out["cv_delta_I"][ek].append(float(np.max(sc.I)))
                I_h = np.interp(pot_grid, sc.E, sc.I)
                with np.errstate(invalid="ignore", divide="ignore"):
                    I_n = np.where(np.abs(I_probe_grid) > 1e-15, (I_h - I_probe_grid) / np.abs(I_probe_grid), 0.0)
                out["cv_spectra"][ek].append(I_n)
            else:
                out["cv_delta_I"][ek].append(np.nan)
                out["cv_spectra"][ek].append(np.zeros(len(pot_grid)))

    return {
        "concentrations": val_concs.tolist(),
        **out,
    }


# ---------------------------------------------------------------------------
# Rendu du rapport — sections
# ---------------------------------------------------------------------------

def _section1_data_quality(report: dict, session_data: dict) -> None:
    """Section 1 — Qualité des données brutes."""
    with st.expander("Section 1 — Qualité des données brutes", expanded=True):
        probe_eis = session_data.get("_probe_eis")
        probe_cv  = session_data.get("_probe_cv")
        concs = session_data["concentrations"]
        n_elec = session_data["n_electrodes"]

        col_eis, col_cv = st.columns(2)

        with col_eis:
            st.markdown("**Spectres EIS — Nyquist**")
            import plotly.graph_objects as go
            fig = go.Figure()
            if probe_eis:
                fig.add_trace(go.Scatter(
                    x=probe_eis.Zre, y=probe_eis.Zim,
                    mode="markers", name="Probe",
                    marker=dict(color="black", size=5, symbol="circle-open"),
                ))
            colors = _colorscale(len(concs))
            for ci, (conc, color) in enumerate(zip(concs, colors)):
                for e in range(1, n_elec + 1):
                    sp_list = session_data["eis_spectra"].get(f"e{e}", [])
                    if ci < len(sp_list):
                        feat = sp_list[ci]
                        n_f = len(session_data["freq_grid"])
                        Zre_n = feat[:n_f]
                        Zim_n = feat[n_f:]
                        fig.add_trace(go.Scatter(
                            x=Zre_n, y=Zim_n,
                            mode="markers",
                            name=f"{conc:.1e} M e{e}",
                            marker=dict(color=color, size=4),
                            showlegend=(e == 1),
                        ))
            fig.update_layout(
                xaxis_title="ΔZre_norm", yaxis_title="ΔZim_norm",
                height=350, margin=dict(t=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_cv:
            st.markdown("**Courbes CV — Courant normalisé**")
            fig2 = go.Figure()
            pot_grid = session_data["pot_grid"]
            for ci, (conc, color) in enumerate(zip(concs, colors)):
                for e in range(1, n_elec + 1):
                    cv_list = session_data["cv_spectra"].get(f"e{e}", [])
                    if ci < len(cv_list):
                        fig2.add_trace(go.Scatter(
                            x=pot_grid, y=cv_list[ci],
                            mode="lines",
                            name=f"{conc:.1e} M e{e}",
                            line=dict(color=color, width=1),
                            showlegend=(e == 1),
                        ))
            fig2.update_layout(
                xaxis_title="Potentiel E (V)", yaxis_title="ΔI_norm",
                height=350, margin=dict(t=30),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Tableau σ_inter avant normalisation
        st.markdown("**Cohérence inter-électrode (σ_inter brut, par méthode)**")
        if n_elec > 1:
            import pandas as pd
            rows = []
            for m, res in report["methods"].items():
                rows.append({
                    "Méthode": m,
                    "σ_inter brut": f"{res.get('sigma_inter_raw', 0):.4f}" if res.get("sigma_inter_raw") is not None else "—",
                    "σ_inter norm": f"{res.get('sigma_inter_norm', 0):.4f}" if res.get("sigma_inter_norm") is not None else "—",
                    "Facteur réduction": f"{res.get('reduction_factor', 1):.2f}" if res.get("reduction_factor") is not None else "—",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Méthode"), use_container_width=True)
        else:
            st.info("Une seule électrode — σ_inter non calculable.")


def _section2_calibration(report: dict) -> None:
    """Section 2 — Calibration de chaque méthode."""
    with st.expander("Section 2 — Calibration de chaque méthode", expanded=False):
        methods_a = ["A1", "A2", "A3"]
        methods_b = ["B1", "B2", "B3"]

        # Méthodes A — 3 colonnes
        st.markdown("#### Méthodes A — Scalaires (OLS)")
        cols = st.columns(3)
        for col, m in zip(cols, methods_a):
            res = report["methods"].get(m)
            if res is None:
                continue
            with col:
                st.markdown(f"**{m}**")
                fig = res.get("calibration_fig")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                _metric_with_help("RMSECV", res.get("rmsecv"), ".3f", "décades", "rmsecv_rmsep")

        # Méthodes B — 3 colonnes
        st.markdown("#### Méthodes B — Spectrales (PLS)")
        cols_b = st.columns(3)
        for col, m in zip(cols_b, methods_b):
            res = report["methods"].get(m)
            if res is None:
                col.info(f"{m} non calculée (données spectrales absentes)")
                continue
            with col:
                st.markdown(f"**{m}** (k={res.get('n_components', '?')} composantes)")
                fig = res.get("calibration_fig")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                _metric_with_help("RMSECV", res.get("rmsecv"), ".3f", "décades", "rmsecv_rmsep")
                with st.expander("ℹ️  Interpréter le scree plot"):
                    st.markdown(_INTERPRET["scree"])


def _section3_variance(report: dict) -> None:
    """Section 3 — Décomposition de la variance."""
    with st.expander("Section 3 — Décomposition de la variance", expanded=False):
        fig = report["figures"].get("variance_decomposition")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        import pandas as pd
        rows = []
        for m, res in report["methods"].items():
            rows.append({
                "Méthode": m,
                "σ_intra": res.get("sigma_intra"),
                "σ_inter brut": res.get("sigma_inter_raw"),
                "σ_inter norm": res.get("sigma_inter_norm"),
                "Facteur réduction": res.get("reduction_factor"),
            })

        df = pd.DataFrame(rows).set_index("Méthode")
        for col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:.4f}" if v is not None and not (isinstance(v, float) and np.isinf(v)) else ("∞" if isinstance(v, float) and np.isinf(v) else "—"))
        st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            with st.expander("ℹ️  σ_intra"):
                st.markdown(_INTERPRET["sigma_intra"])
        with col2:
            with st.expander("ℹ️  σ_inter"):
                st.markdown(_INTERPRET["sigma_inter"])
        with col3:
            with st.expander("ℹ️  Facteur de réduction"):
                st.markdown(_INTERPRET["reduction_factor"])


def _section4_validation(report: dict) -> None:
    """Section 4 — Performance prédictive (si données de validation)."""
    has_validation = any(
        res.get("rmsep") is not None for res in report["methods"].values()
    )
    if not has_validation:
        return

    with st.expander("Section 4 — Performance prédictive (validation)", expanded=False):
        fig = report["figures"].get("predicted_vs_true")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Tableau RMSEP / Biais / CI_95 / RPD
        import pandas as pd
        rows = []
        for m, res in report["methods"].items():
            ci = res.get("ci95")
            if ci and not any(np.isnan(c) for c in ci):
                ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
            else:
                ci_str = "—"
            rows.append({
                "Méthode": m,
                "RMSEP (déc.)": _fmt(res.get("rmsep"), ".3f"),
                "Biais (déc.)": _fmt(res.get("bias"),  "+.3f"),
                "RPD":          _fmt(res.get("rpd"),   ".2f"),
                "CI 95 %":     ci_str,
            })

        st.markdown("#### Métriques de prédiction")
        st.dataframe(pd.DataFrame(rows).set_index("Méthode"), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        for col, key, label in [
            (col1, "rmsecv_rmsep", "RMSEP"),
            (col2, "bias",         "Biais"),
            (col3, "rpd",          "RPD"),
            (col4, "ci95",         "CI 95 %"),
        ]:
            with col:
                with st.expander(f"ℹ️  {label}"):
                    st.markdown(_INTERPRET[key])

        # Tableau Diebold-Mariano
        dm_data = {}
        for m, res in report["methods"].items():
            dm_tests = res.get("dm_tests") or {}
            if dm_tests:
                dm_data[m] = dm_tests

        if dm_data:
            st.markdown("#### Tests de Diebold-Mariano (p-values)")
            all_m = sorted(report["methods"].keys())
            dm_rows = []
            for ma in all_m:
                row = {"Méthode": ma}
                for mb in all_m:
                    if ma == mb:
                        row[mb] = "—"
                    elif mb in (dm_data.get(ma) or {}):
                        _, pval = dm_data[ma][mb]
                        sig = " ✱" if pval < 0.05 else ""
                        row[mb] = f"{pval:.3f}{sig}"
                    else:
                        row[mb] = ""
                dm_rows.append(row)
            st.dataframe(pd.DataFrame(dm_rows).set_index("Méthode"), use_container_width=True)
            st.caption("✱ p < 0.05 — différence statistiquement significative")
            with st.expander("ℹ️  Interpréter le test de Diebold-Mariano"):
                st.markdown(_INTERPRET["dm_test"])


def _section5_loadings(report: dict, session_data: dict) -> None:
    """Section 5 — Loadings PLS."""
    methods_b = [m for m in ("B1", "B2", "B3") if m in report["methods"]]
    if not methods_b:
        return

    freq_grid = np.asarray(session_data["freq_grid"])
    pot_grid  = np.asarray(session_data["pot_grid"])

    with st.expander("Section 5 — Loadings PLS (méthodes B)", expanded=False):
        for m in methods_b:
            res = report["methods"][m]
            loadings_figs = res.get("loadings_figs", {})
            if not loadings_figs:
                st.info(f"{m} : figures de loadings non disponibles.")
                continue

            st.markdown(f"**{m}**")
            tabs = st.tabs([k for k in loadings_figs.keys()])
            for tab, (key, fig) in zip(tabs, loadings_figs.items()):
                with tab:
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("ℹ️  Interpréter les loadings"):
                        st.markdown(_INTERPRET["loadings"])


# ---------------------------------------------------------------------------
# Export HTML autonome
# ---------------------------------------------------------------------------

def _build_html_report(report: dict, session_data: dict) -> str:
    """Génère un rapport HTML autonome avec toutes les figures et métriques."""
    parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="utf-8">',
        "<title>EIS Analyzer v3 — Rapport comparatif</title>",
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        "<style>body{font-family:sans-serif;max-width:1200px;margin:auto;padding:20px}",
        "table{border-collapse:collapse;width:100%}",
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:center}",
        "th{background:#4a90d9;color:white}",
        "h2{border-bottom:2px solid #4a90d9;padding-bottom:4px}",
        "</style></head><body>",
        "<h1>Rapport comparatif — EIS Analyzer v3</h1>",
        f"<p>Concentrations : {len(session_data['concentrations'])} points — "
        f"Électrodes : {session_data['n_electrodes']}</p>",
    ]

    # Section calibrations
    parts.append("<h2>Calibration par méthode</h2>")
    for m, res in report["methods"].items():
        fig = res.get("calibration_fig")
        if fig:
            parts.append(f"<h3>{m}</h3>")
            parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Figures globales
    for title, key in [
        ("Performance prédictive", "predicted_vs_true"),
        ("Décomposition de la variance", "variance_decomposition"),
    ]:
        fig = report["figures"].get(key)
        if fig:
            parts.append(f"<h2>{title}</h2>")
            parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Loadings
    parts.append("<h2>Loadings PLS</h2>")
    for m, res in report["methods"].items():
        for key, fig in (res.get("loadings_figs") or {}).items():
            parts.append(f"<h3>{m} — {key}</h3>")
            parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Tableau métriques
    parts.append("<h2>Métriques de performance</h2>")
    parts.append("<table><tr><th>Méthode</th><th>RMSECV</th><th>RMSEP</th>"
                 "<th>Biais</th><th>RPD</th><th>σ_intra</th>"
                 "<th>σ_inter brut</th><th>σ_inter norm</th><th>Facteur réduction</th></tr>")
    for m, res in report["methods"].items():
        parts.append(
            f"<tr><td><b>{m}</b></td>"
            f"<td>{_fmt(res.get('rmsecv'),'.3f')}</td>"
            f"<td>{_fmt(res.get('rmsep'),'.3f')}</td>"
            f"<td>{_fmt(res.get('bias'),'+.3f')}</td>"
            f"<td>{_fmt(res.get('rpd'),'.2f')}</td>"
            f"<td>{_fmt(res.get('sigma_intra'),'.4f')}</td>"
            f"<td>{_fmt(res.get('sigma_inter_raw'),'.4f')}</td>"
            f"<td>{_fmt(res.get('sigma_inter_norm'),'.4f')}</td>"
            f"<td>{_fmt(res.get('reduction_factor'),'.2f')}</td></tr>"
        )
    parts.append("</table></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Utilitaires d'affichage
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".3f") -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and np.isinf(v):
        return "∞"
    if isinstance(v, float) and np.isnan(v):
        return "—"
    try:
        return format(v, fmt)
    except (TypeError, ValueError):
        return str(v)


def _metric_with_help(label: str, value, fmt: str, unit: str, help_key: str) -> None:
    """Affiche une métrique + son expander d'interprétation."""
    st.metric(label, f"{_fmt(value, fmt)} {unit}" if value is not None else "—")
    with st.expander(f"ℹ️  Interpréter {label}"):
        st.markdown(_INTERPRET[help_key])


def _colorscale(n: int) -> list[str]:
    """n couleurs interpolées bleu → rouge."""
    import plotly.colors as pc
    scale = pc.sample_colorscale("RdYlBu_r", [i / max(n - 1, 1) for i in range(n)])
    return scale


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📊 Comparaison des 6 méthodes d'extraction de concentration")
    st.caption(
        "A1 (ΔI_CV) · A2 (Rct fit) · A3 (Rct DRT) · "
        "B1 (PLS CV) · B2 (PLS EIS) · B3 (PLS EIS+CV)"
    )

    # ---- Entrée des données ----
    data = render_data_input(mode="both", prefix="page_C")

    if not data["run_clicked"]:
        st.info("Complétez l'upload des fichiers EIS et CV, puis cliquez sur **▶ Lancer l'analyse**.")
        return

    # ---- Calcul du rapport ----
    progress_bar = st.progress(0, text="Initialisation…")

    def _progress(frac: float, msg: str):
        progress_bar.progress(min(frac, 1.0), text=msg)

    with st.spinner("Analyse en cours…"):
        session_data = _load_all_data_safe(data, _progress)

    if session_data is None:
        progress_bar.empty()
        return

    _progress(0.97, "Calcul des métriques comparatives…")
    report = compute_full_report(session_data)
    st.session_state["comparison_report"]      = report
    st.session_state["comparison_session_data"] = session_data
    progress_bar.progress(1.0, text="Analyse terminée ✓")

    # ---- Affichage du rapport ----
    _render_report(report, session_data)


def _load_all_data_safe(data: dict, progress_cb) -> dict | None:
    """Wrapper avec gestion d'exception globale."""
    try:
        return _load_session_data(data, progress_cb)
    except Exception as exc:
        st.error(f"Erreur lors du chargement des données : {exc}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None


def _render_report(report: dict, session_data: dict) -> None:
    """Affiche le rapport complet en sections dépliables."""
    st.divider()
    st.subheader("Rapport comparatif")

    _section1_data_quality(report, session_data)
    _section2_calibration(report)
    _section3_variance(report)
    _section4_validation(report)
    _section5_loadings(report, session_data)

    # ---- Export ----
    st.divider()
    st.subheader("💾 Export du rapport")
    col1, col2 = st.columns(2)
    with col1:
        html = _build_html_report(report, session_data)
        st.download_button(
            label="🌐 Rapport HTML complet",
            data=html.encode("utf-8"),
            file_name="rapport_comparatif.html",
            mime="text/html",
            use_container_width=True,
        )
    with col2:
        import pandas as pd
        rows = []
        for m, res in report["methods"].items():
            ci = res.get("ci95")
            rows.append({
                "methode": m,
                "rmsecv":            res.get("rmsecv"),
                "rmsep":             res.get("rmsep"),
                "bias":              res.get("bias"),
                "rpd":               res.get("rpd"),
                "ci95_low":          ci[0] if ci else None,
                "ci95_high":         ci[1] if ci else None,
                "sigma_intra":       res.get("sigma_intra"),
                "sigma_inter_raw":   res.get("sigma_inter_raw"),
                "sigma_inter_norm":  res.get("sigma_inter_norm"),
                "reduction_factor":  res.get("reduction_factor"),
            })
        csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Métriques CSV",
            data=csv_bytes,
            file_name="metriques_comparatives.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
