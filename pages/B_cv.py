"""Page B — Analyse CV seule.

Charge les courbes de voltammétrie cyclique, extrait les pics redox,
normalise par le probe, et produit une calibration OLS log([c]) ~ ΔI_norm.
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.signal import find_peaks

from core.cv_loader import load_cv_file, average_cv_replicates
from core.cv_models import CVScan
from ui.data_input import render_data_input


# ---------------------------------------------------------------------------
# Extraction des pics redox
# ---------------------------------------------------------------------------

def _extract_peaks(scan: CVScan) -> dict:
    """
    Extrait les pics d'oxydation et de réduction d'un voltammogramme.

    Stratégie : pic d'oxydation = maximum global de I (courant le plus positif),
    pic de réduction = minimum global de I (courant le plus négatif).
    Robuste sur un seul couple redox, ce qui est le cas des biosenseurs Fe(CN)₆.

    Returns
    -------
    dict avec les clés :
        I_ox, E_ox   — courant (A) et potentiel (V) du pic d'oxydation
        I_red, E_red — courant (A) et potentiel (V) du pic de réduction
        delta_I      — I_ox - abs(I_red), en A
        E_mid        — (E_ox + E_red) / 2, potentiel de demi-vague (V)
    """
    idx_ox  = int(np.argmax(scan.I))
    idx_red = int(np.argmin(scan.I))

    I_ox  = float(scan.I[idx_ox])
    E_ox  = float(scan.E[idx_ox])
    I_red = float(scan.I[idx_red])
    E_red = float(scan.E[idx_red])

    return {
        "I_ox":    I_ox,
        "E_ox":    E_ox,
        "I_red":   I_red,
        "E_red":   E_red,
        "delta_I": I_ox - abs(I_red),
        "E_mid":   (E_ox + E_red) / 2.0,
    }


# ---------------------------------------------------------------------------
# Normalisation par le probe
# ---------------------------------------------------------------------------

def _normalize_peak(I_pic_hyb: float, I_pic_probe: float) -> float:
    """
    ΔI_norm = (I_pic_hyb − I_pic_probe) / |I_pic_probe|

    Retourne 0.0 si |I_pic_probe| < 1e-15 A pour éviter les divisions par zéro.
    """
    denom = abs(I_pic_probe)
    if denom < 1e-15:
        return 0.0
    return (I_pic_hyb - I_pic_probe) / denom


# ---------------------------------------------------------------------------
# Chargement des fichiers depuis le dict data_input
# ---------------------------------------------------------------------------

def _load_scan_from_file(uploaded_file, label: str, concentration: float, step: str) -> CVScan | None:
    """Lit un UploadedFile Streamlit et retourne un CVScan. Retourne None si le fichier est None."""
    if uploaded_file is None:
        return None
    content = uploaded_file.read()
    uploaded_file.seek(0)
    try:
        return load_cv_file(content, label, concentration, step)
    except Exception as exc:
        st.warning(f"⚠️  Impossible de charger {label} : {exc}")
        return None


def _run_cv_analysis(data: dict) -> dict | None:
    """
    Construit les scans moyennés, extrait les pics et calcule ΔI_norm.

    Parameters
    ----------
    data : dict retourné par render_data_input(mode='cv_only')

    Returns
    -------
    dict avec :
        probe_peaks  : dict — pics du probe (valeurs de référence)
        probe_scan   : CVScan (moyenné sur toutes les électrodes)
        groups       : list[dict] — une entrée par concentration, avec :
            concentration : float (M)
            label         : str
            scans_e1      : list[CVScan] — réplicats électrode 1
            scans_e2      : list[CVScan] — réplicats électrode 2 (peut être vide)
            avg_e1        : CVScan moyenné électrode 1
            avg_e2        : CVScan | None
            peaks_e1      : dict
            peaks_e2      : dict | None
            delta_I_norm_e1 : float
            delta_I_norm_e2 : float | None
            delta_I_norm_mean : float  — moyenne des deux électrodes
    Retourne None en cas d'erreur fatale.
    """
    cv   = data["calibration"]["cv"]
    concs = data["concentrations"]
    n_conc = len(concs)
    n_elec = data["n_electrodes"]
    n_rep  = data["n_replicats"]

    errors = []

    # --- Probe : moyennage de toutes les électrodes disponibles ---
    probe_scans_all = []
    for e in range(1, n_elec + 1):
        probe_list = cv.get(f"probe_{e}", [])
        for ci, pf in enumerate(probe_list):
            sc = _load_scan_from_file(pf, f"probe_e{e}_c{ci+1}", concs[ci], "probe")
            if sc is not None:
                probe_scans_all.append(sc)

    if not probe_scans_all:
        st.error("Aucun fichier probe chargé — impossible de normaliser.")
        return None

    probe_scan = average_cv_replicates(probe_scans_all)
    probe_peaks = _extract_peaks(probe_scan)

    # --- Groupes par concentration ---
    groups = []
    for ci, conc in enumerate(concs):
        c_label = f"{conc:.2e} M"
        group: dict = {
            "concentration": conc,
            "label": c_label,
        }

        for e in range(1, 3):   # on gère toujours électrode 1 et 2
            key_reps = f"electrode_{e}"
            reps_for_conc = cv.get(key_reps, [])
            rep_files = reps_for_conc[ci] if ci < len(reps_for_conc) else []

            scans = []
            if rep_files:
                for r, rf in enumerate(rep_files):
                    sc = _load_scan_from_file(rf, f"e{e}_c{ci+1}_r{r+1}", conc, "hybridization")
                    if sc is not None:
                        scans.append(sc)

            group[f"scans_e{e}"] = scans

            if scans:
                avg = average_cv_replicates(scans)
                group[f"avg_e{e}"] = avg
                group[f"peaks_e{e}"] = _extract_peaks(avg)
                group[f"delta_I_norm_e{e}"] = _normalize_peak(
                    group[f"peaks_e{e}"]["I_ox"],
                    probe_peaks["I_ox"],
                )
            else:
                group[f"avg_e{e}"] = None
                group[f"peaks_e{e}"] = None
                group[f"delta_I_norm_e{e}"] = None

        # Moyenne des électrodes disponibles
        norms = [
            group[f"delta_I_norm_e{e}"]
            for e in range(1, 3)
            if group[f"delta_I_norm_e{e}"] is not None
        ]
        group["delta_I_norm_mean"] = float(np.mean(norms)) if norms else None

        groups.append(group)

    return {
        "probe_scan":  probe_scan,
        "probe_peaks": probe_peaks,
        "groups":      groups,
    }


# ---------------------------------------------------------------------------
# Régression OLS avec IC 95 %
# ---------------------------------------------------------------------------

def _ols_calibration(groups: list) -> dict | None:
    """
    Régression OLS : log10([c]) = a·ΔI_norm + b

    Retourne None si moins de 3 points valides.
    """
    concs, norms = [], []
    for g in groups:
        if g["concentration"] > 0 and g["delta_I_norm_mean"] is not None:
            concs.append(g["concentration"])
            norms.append(g["delta_I_norm_mean"])

    if len(concs) < 3:
        return None

    x = np.array(norms)
    y = np.log10(concs)

    res = stats.linregress(x, y)
    n = len(x)
    t95 = stats.t.ppf(0.975, df=n - 2)

    # IC 95 % sur la droite de régression
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = res.slope * x_fit + res.intercept

    x_mean = x.mean()
    se_fit = res.stderr * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
    ci_low = y_fit - t95 * se_fit
    ci_high = y_fit + t95 * se_fit

    return {
        "slope":     res.slope,
        "intercept": res.intercept,
        "r2":        res.rvalue ** 2,
        "stderr":    res.stderr,
        "p_value":   res.pvalue,
        "n":         n,
        "x_data":    x,
        "y_data":    y,
        "x_fit":     x_fit,
        "y_fit":     y_fit,
        "ci_low":    ci_low,
        "ci_high":   ci_high,
    }


# ---------------------------------------------------------------------------
# Figures Plotly
# ---------------------------------------------------------------------------

def _fig_voltammograms(result: dict) -> go.Figure:
    """Superposition de toutes les courbes CV par concentration + probe."""
    fig = go.Figure()
    probe = result["probe_scan"]
    fig.add_trace(go.Scatter(
        x=probe.E, y=probe.I * 1e6,
        mode="lines", name="Probe",
        line=dict(color="black", dash="dash", width=2),
    ))

    colors = _concentration_colorscale(len(result["groups"]))
    for grp, color in zip(result["groups"], colors):
        for e in (1, 2):
            avg = grp.get(f"avg_e{e}")
            if avg is not None:
                fig.add_trace(go.Scatter(
                    x=avg.E, y=avg.I * 1e6,
                    mode="lines",
                    name=f"{grp['label']} – élec.{e}",
                    line=dict(color=color),
                    legendgroup=grp["label"],
                ))

    fig.update_layout(
        title="Voltammogrammes CV — Courant vs Potentiel",
        xaxis_title="Potentiel E (V)",
        yaxis_title="Courant I (µA)",
        legend_title="Scan",
    )
    return fig


def _fig_pic_redox(result: dict) -> go.Figure:
    """ΔI_pic vs concentration — axes linéaire et log."""
    concs, delta_I = [], []
    for g in result["groups"]:
        if g["concentration"] > 0 and g["delta_I_norm_mean"] is not None:
            concs.append(g["concentration"])
            delta_I.append(g["delta_I_norm_mean"])

    if not concs:
        return go.Figure()

    log_c = np.log10(concs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_c, y=delta_I,
        mode="markers+lines",
        name="ΔI_norm",
        marker=dict(size=9),
    ))
    fig.update_layout(
        title="Signal normalisé ΔI_norm vs log([c])",
        xaxis_title="log₁₀([c] / M)",
        yaxis_title="ΔI_norm = (I_hyb − I_probe) / |I_probe|",
    )
    return fig


def _fig_calibration(ols: dict) -> go.Figure:
    """Droite OLS + IC 95 % + nuage de points."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ols["x_data"], y=ols["y_data"],
        mode="markers",
        name="Données",
        marker=dict(size=10, color="steelblue"),
    ))
    fig.add_trace(go.Scatter(
        x=ols["x_fit"], y=ols["y_fit"],
        mode="lines",
        name=f"OLS (R²={ols['r2']:.3f})",
        line=dict(color="firebrick", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([ols["x_fit"], ols["x_fit"][::-1]]),
        y=np.concatenate([ols["ci_high"], ols["ci_low"][::-1]]),
        fill="toself",
        fillcolor="rgba(178,34,34,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC 95 %",
        hoverinfo="skip",
    ))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.04, y=0.96,
        text=(
            f"<b>OLS</b> : log([c]) = {ols['slope']:.3f}·ΔI_norm "
            f"+ {ols['intercept']:.3f}<br>"
            f"R² = {ols['r2']:.4f} — n = {ols['n']} points"
        ),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="gray", borderwidth=1,
    )
    fig.update_layout(
        title="Calibration CV — Régression OLS",
        xaxis_title="ΔI_norm",
        yaxis_title="log₁₀([c] / M)",
    )
    return fig


def _fig_params_table(result: dict) -> go.Figure:
    """Tableau récapitulatif des paramètres extraits par concentration."""
    rows = []
    for g in result["groups"]:
        for e in (1, 2):
            pk = g.get(f"peaks_e{e}")
            if pk is None:
                continue
            dn = g.get(f"delta_I_norm_e{e}")
            rows.append({
                "Concentration": g["label"],
                "Électrode": f"E{e}",
                "I_pic_ox (µA)": f"{pk['I_ox']*1e6:.3f}",
                "E_pic_ox (V)":  f"{pk['E_ox']:.4f}",
                "I_pic_red (µA)": f"{pk['I_red']*1e6:.3f}",
                "E_pic_red (V)":  f"{pk['E_red']:.4f}",
                "ΔI_norm":        f"{dn:.4f}" if dn is not None else "—",
            })

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows)
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in df.columns],
            fill_color="steelblue",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[df[c] for c in df.columns],
            fill_color=[["white", "#f5f5f5"] * (len(df) // 2 + 1)],
            align="center",
        ),
    )])
    fig.update_layout(title="Paramètres extraits par concentration et électrode")
    return fig


def _concentration_colorscale(n: int) -> list[str]:
    """Génère n couleurs interpolées entre bleu clair et rouge."""
    import plotly.colors as pc
    scale = pc.sample_colorscale("RdYlBu_r", [i / max(n - 1, 1) for i in range(n)])
    return [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            for r, g, b in [pc.unlabel_rgb(s) for s in scale]]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export_params_csv(result: dict) -> bytes:
    rows = []
    for g in result["groups"]:
        for e in (1, 2):
            pk = g.get(f"peaks_e{e}")
            if pk is None:
                continue
            rows.append({
                "concentration_M": g["concentration"],
                "electrode": e,
                "I_pic_ox_A":  pk["I_ox"],
                "E_pic_ox_V":  pk["E_ox"],
                "I_pic_red_A": pk["I_red"],
                "E_pic_red_V": pk["E_red"],
                "delta_I_norm": g.get(f"delta_I_norm_e{e}"),
            })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _export_scans_csv(result: dict) -> bytes:
    rows = []
    probe = result["probe_scan"]
    for E, I in zip(probe.E, probe.I):
        rows.append({"label": "probe", "concentration_M": 0.0, "E_V": E, "I_A": I})
    for g in result["groups"]:
        for e in (1, 2):
            avg = g.get(f"avg_e{e}")
            if avg is None:
                continue
            for E, I in zip(avg.E, avg.I):
                rows.append({"label": g["label"], "electrode": e,
                             "concentration_M": g["concentration"], "E_V": E, "I_A": I})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Analyse CV", page_icon="📈", layout="wide")
    st.title("📈 Analyse CV — Voltammétrie cyclique")
    st.caption("Extraction des pics redox · Normalisation probe · Calibration OLS")

    # --- Saisie des données ---
    data = render_data_input(mode="cv_only", prefix="page_B")

    if not data["run_clicked"]:
        st.info("Complétez l'upload des fichiers puis cliquez sur **▶ Lancer l'analyse**.")
        return

    # --- Analyse ---
    with st.spinner("Chargement et analyse des courbes CV…"):
        result = _run_cv_analysis(data)

    if result is None:
        return

    ols = _ols_calibration(result["groups"])

    # --- Onglets ---
    tab_volt, tab_pic, tab_calib, tab_params, tab_export = st.tabs([
        "📉 Voltammogrammes",
        "🔴 Pic redox",
        "📊 Calibration",
        "🔢 Paramètres",
        "💾 Export",
    ])

    with tab_volt:
        st.subheader("Voltammogrammes — Superposition par concentration")
        st.plotly_chart(_fig_voltammograms(result), use_container_width=True)

    with tab_pic:
        st.subheader("Signal normalisé ΔI_norm vs concentration")
        st.plotly_chart(_fig_pic_redox(result), use_container_width=True)

        # Résumé rapide des pics probe
        pp = result["probe_peaks"]
        col1, col2, col3 = st.columns(3)
        col1.metric("I_pic_ox probe", f"{pp['I_ox']*1e6:.3f} µA")
        col2.metric("I_pic_red probe", f"{pp['I_red']*1e6:.3f} µA")
        col3.metric("E_mid probe", f"{pp['E_mid']*1e3:.1f} mV")

    with tab_calib:
        st.subheader("Calibration OLS — log₁₀([c]) ~ ΔI_norm")
        if ols is None:
            st.warning("Moins de 3 points de calibration valides — régression impossible.")
        else:
            st.plotly_chart(_fig_calibration(ols), use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",      f"{ols['r2']:.4f}")
            c2.metric("Pente a", f"{ols['slope']:.4f}")
            c3.metric("Ordonnée b", f"{ols['intercept']:.4f}")
            c4.metric("p-value",    f"{ols['p_value']:.2e}")

            with st.expander("ℹ️  Interpréter la calibration OLS"):
                st.markdown("""
**Modèle :** log₁₀([c]) = a·ΔI_norm + b

- **R² > 0.99** — calibration excellente, signal proportionnel à log([c])
- **R² 0.95–0.99** — acceptable selon la gamme de concentration
- **R² < 0.95** — vérifier la normalisation probe et la qualité des voltammogrammes

**IC 95 % (bande grisée)** — intervalle dans lequel se trouve la vraie droite avec 95 % de probabilité.
Une bande large aux extrémités indique que les points extrêmes ont plus d'influence (biais de levier).
                """)

    with tab_params:
        st.subheader("Paramètres extraits par concentration et électrode")
        st.plotly_chart(_fig_params_table(result), use_container_width=True)

    with tab_export:
        st.subheader("Télécharger les résultats")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Paramètres CSV",
                data=_export_params_csv(result),
                file_name="cv_params.csv",
                mime="text/csv",
            )
            st.download_button(
                label="📊 Courbes CSV",
                data=_export_scans_csv(result),
                file_name="cv_scans.csv",
                mime="text/csv",
            )
        with col2:
            if ols:
                import plotly.io as pio
                fig_calib = _fig_calibration(ols)
                st.download_button(
                    label="🌐 Calibration HTML",
                    data=pio.to_html(fig_calib, full_html=True),
                    file_name="cv_calibration.html",
                    mime="text/html",
                )
                try:
                    st.download_button(
                        label="🖼 Calibration PNG",
                        data=fig_calib.to_image(format="png", scale=2),
                        file_name="cv_calibration.png",
                        mime="image/png",
                    )
                except Exception:
                    st.caption("PNG non disponible (kaleido requis).")


if __name__ == "__main__":
    main()
