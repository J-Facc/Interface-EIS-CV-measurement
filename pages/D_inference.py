"""Page D — Inférence de concentration sur une mesure inconnue.

Charge un modèle de calibration (session active ou YAML),
reçoit les fichiers d'une mesure inconnue, normalise par le probe
uploadé, prédit la concentration avec les 6 méthodes disponibles
et affiche le résultat avec indicateur de cohérence inter-électrode.
"""

import io
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from core.config import load_config, config_to_dict
from core.loader import load_spectrum, average_replicates
from core.cv_loader import load_cv_file
from fits.registry import get_model
from comparison.report import predict_from_session


_CONFIG = None
_METHOD_LABELS = {
    "A1": "A1 — ΔI_CV (OLS)",
    "A2": "A2 — Rct fit Randles (OLS)",
    "A3": "A3 — Rct DRT (OLS)",
    "B1": "B1 — PLS CV spectral",
    "B2": "B2 — PLS EIS spectral",
    "B3": "B3 — PLS EIS+CV",
}
_METHOD_COLORS = {
    "A1": "#1f77b4", "A2": "#ff7f0e", "A3": "#2ca02c",
    "B1": "#d62728", "B2": "#9467bd", "B3": "#8c564b",
}


def _get_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = config_to_dict(load_config())
    return _CONFIG


def _read_file(uploaded_file):
    if uploaded_file is None:
        return None, None
    content = uploaded_file.read()
    uploaded_file.seek(0)
    return content, uploaded_file.name


def _fmt_conc(log10_c: float) -> str:
    """Formate une concentration log10([c]) en notation scientifique lisible."""
    if log10_c is None or not np.isfinite(log10_c):
        return "—"
    exp = int(np.floor(log10_c))
    mant = 10 ** (log10_c - exp)
    sup_map = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return f"{mant:.2f} × 10{str(exp).translate(sup_map)} M"


def _fmt_ci(ci_factor: float | None) -> str:
    if ci_factor is None:
        return "—"
    return f"×{ci_factor:.1f}"


# ---------------------------------------------------------------------------
# Chargement de la session de calibration
# ---------------------------------------------------------------------------

def _load_session_from_yaml(yaml_bytes: bytes) -> dict | None:
    """
    Recharge une session de calibration depuis un fichier YAML exporté.

    Le YAML doit contenir les champs produits par export_inference_yaml().
    Retourne None si le format est invalide.
    """
    try:
        raw = yaml.safe_load(yaml_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        st.error(f"Impossible de lire le YAML : {exc}")
        return None

    required = {"concentrations", "eis_rct_fit", "eis_rct_drt", "cv_delta_I",
                "probe_rct_fit", "probe_rct_drt", "probe_delta_I"}
    missing = required - set(raw.keys())
    if missing:
        st.error(f"YAML incomplet — champs manquants : {missing}")
        return None

    # Convertit les listes YAML en numpy pour les grilles
    session = dict(raw)
    for grid_key in ("freq_grid", "pot_grid"):
        if grid_key in session and isinstance(session[grid_key], list):
            session[grid_key] = np.asarray(session[grid_key], dtype=float)

    for sig_key in ("eis_rct_fit", "eis_rct_drt", "cv_delta_I"):
        for ek in ("e1", "e2"):
            if ek in session.get(sig_key, {}):
                session[sig_key][ek] = [
                    float(v) if v is not None else np.nan
                    for v in session[sig_key][ek]
                ]

    # Les spectres PLS ne sont pas stockés en YAML — méthodes B indisponibles
    session.setdefault("eis_spectra", {"e1": [], "e2": []})
    session.setdefault("cv_spectra",  {"e1": [], "e2": []})

    return session


def export_inference_yaml(session_data: dict) -> str:
    """
    Exporte les données scalaires de calibration en YAML.

    Les données spectrales PLS (eis_spectra, cv_spectra) sont omises
    car leur taille serait trop importante. Les méthodes B1/B2/B3
    ne seront donc pas disponibles lors d'un rechargement YAML.
    """
    exportable = {
        "concentrations":  session_data.get("concentrations", []),
        "n_electrodes":    session_data.get("n_electrodes", 1),
        "eis_rct_fit":     {
            k: [float(v) if np.isfinite(v) else None for v in arr]
            for k, arr in session_data.get("eis_rct_fit", {}).items()
        },
        "eis_rct_drt":     {
            k: [float(v) if np.isfinite(v) else None for v in arr]
            for k, arr in session_data.get("eis_rct_drt", {}).items()
        },
        "cv_delta_I":      {
            k: [float(v) if np.isfinite(v) else None for v in arr]
            for k, arr in session_data.get("cv_delta_I", {}).items()
        },
        "probe_rct_fit":   float(session_data.get("probe_rct_fit") or 0),
        "probe_rct_drt":   float(session_data.get("probe_rct_drt") or 0),
        "probe_delta_I":   float(session_data.get("probe_delta_I") or 0),
        "freq_grid":       (
            session_data["freq_grid"].tolist()
            if hasattr(session_data.get("freq_grid"), "tolist")
            else list(session_data.get("freq_grid") or [])
        ),
        "pot_grid":        (
            session_data["pot_grid"].tolist()
            if hasattr(session_data.get("pot_grid"), "tolist")
            else list(session_data.get("pot_grid") or [])
        ),
    }
    return yaml.dump(exportable, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Pipeline de chargement de la mesure inconnue
# ---------------------------------------------------------------------------

def _process_new_measurement(
    probe_eis_e1, probe_eis_e2,
    probe_cv_e1,  probe_cv_e2,
    eis_e1,       eis_e2,
    cv_e1,        cv_e2,
    session_data: dict,
) -> dict | None:
    """
    Charge, moyenne et normalise les fichiers de la mesure inconnue.

    Retourne le dict new_signals attendu par predict_from_session(),
    ou None si les fichiers probe sont absents.
    """
    config     = _get_config()
    freq_grid  = np.asarray(session_data.get("freq_grid") or np.logspace(0, 5, 50))
    pot_grid   = np.asarray(session_data.get("pot_grid")  or np.linspace(-0.3, 0.7, 60))

    errors = []

    # ---- Probe EIS ----
    probe_eis_scans = []
    for uf, label in [(probe_eis_e1, "probe EIS e1"), (probe_eis_e2, "probe EIS e2")]:
        content, name = _read_file(uf)
        if content is None:
            continue
        try:
            sp = load_spectrum(content, name, 0.0, "probe", config)
            probe_eis_scans.append(sp)
        except Exception as exc:
            errors.append(f"{label} : {exc}")

    if not probe_eis_scans:
        st.error("Au moins un fichier probe EIS est requis pour la normalisation.")
        for e in errors:
            st.caption(e)
        return None

    probe_eis_avg = average_replicates(probe_eis_scans)

    # Fit probe EIS
    probe_fits_eis = {}
    for mname in ("randles_full", "drt_fit"):
        try:
            probe_fits_eis[mname] = get_model(mname).fit(probe_eis_avg, config)
        except Exception:
            pass

    probe_rct_fit_new = float(probe_fits_eis["randles_full"].Rct) if "randles_full" in probe_fits_eis else 0.0
    probe_rct_drt_new = float(probe_fits_eis["drt_fit"].Rct) if "drt_fit" in probe_fits_eis else 0.0

    Zre_probe = np.interp(freq_grid, probe_eis_avg.f[::-1], probe_eis_avg.Zre[::-1])
    Zim_probe = np.interp(freq_grid, probe_eis_avg.f[::-1], probe_eis_avg.Zim[::-1])

    # ---- Probe CV ----
    probe_cv_scans = []
    for uf, label in [(probe_cv_e1, "probe CV e1"), (probe_cv_e2, "probe CV e2")]:
        content, name = _read_file(uf)
        if content is None:
            continue
        try:
            from core.cv_loader import average_cv_replicates
            sc = load_cv_file(content, name, 0.0, "probe")
            probe_cv_scans.append(sc)
        except Exception as exc:
            errors.append(f"{label} : {exc}")

    if not probe_cv_scans:
        st.warning("Aucun probe CV — méthodes A1 et B1/B3 indisponibles.")
        probe_cv_avg = None
        probe_dI_new = 0.0
        I_probe_grid = np.zeros(len(pot_grid))
    else:
        from core.cv_loader import average_cv_replicates
        probe_cv_avg = average_cv_replicates(probe_cv_scans)
        probe_dI_new = float(np.max(probe_cv_avg.I))
        I_probe_grid = np.interp(pot_grid, probe_cv_avg.E, probe_cv_avg.I)

    # ---- EIS mesure inconnue ----
    def _load_eis(uf, elec_label):
        content, name = _read_file(uf)
        if content is None:
            return None
        try:
            return load_spectrum(content, name, 0.0, "hybridization", config)
        except Exception as exc:
            errors.append(f"EIS {elec_label} : {exc}")
            return None

    eis_sp_e1 = _load_eis(eis_e1, "e1")
    eis_sp_e2 = _load_eis(eis_e2, "e2")

    def _extract_eis_signals(sp):
        """Retourne (rct_fit, rct_drt, feat_array) pour un EISSpectrum."""
        if sp is None:
            return None, None, None
        fits_h = {}
        for mname in ("randles_full", "drt_fit"):
            try:
                fits_h[mname] = get_model(mname).fit(sp, config)
            except Exception:
                pass
        rct_fit = float(fits_h["randles_full"].Rct) if "randles_full" in fits_h else None
        rct_drt = float(fits_h["drt_fit"].Rct) if "drt_fit" in fits_h else None

        Zre_h = np.interp(freq_grid, sp.f[::-1], sp.Zre[::-1])
        Zim_h = np.interp(freq_grid, sp.f[::-1], sp.Zim[::-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            Zre_n = np.where(np.abs(Zre_probe) > 1e-10, (Zre_h - Zre_probe) / np.abs(Zre_probe), 0.0)
            Zim_n = np.where(np.abs(Zim_probe) > 1e-10, (Zim_h - Zim_probe) / np.abs(Zim_probe), 0.0)
        return rct_fit, rct_drt, np.concatenate([Zre_n, Zim_n])

    rct_fit_e1, rct_drt_e1, eis_feat_e1 = _extract_eis_signals(eis_sp_e1)
    rct_fit_e2, rct_drt_e2, eis_feat_e2 = _extract_eis_signals(eis_sp_e2)

    # ---- CV mesure inconnue ----
    def _load_cv(uf, elec_label):
        content, name = _read_file(uf)
        if content is None:
            return None
        try:
            return load_cv_file(content, name, 0.0, "hybridization")
        except Exception as exc:
            errors.append(f"CV {elec_label} : {exc}")
            return None

    cv_sc_e1 = _load_cv(cv_e1, "e1")
    cv_sc_e2 = _load_cv(cv_e2, "e2")

    def _extract_cv_signals(sc):
        if sc is None:
            return None, None
        delta_I = float(np.max(sc.I))
        I_h = np.interp(pot_grid, sc.E, sc.I)
        with np.errstate(invalid="ignore", divide="ignore"):
            I_n = np.where(np.abs(I_probe_grid) > 1e-15, (I_h - I_probe_grid) / np.abs(I_probe_grid), 0.0)
        return delta_I, I_n

    delta_I_e1, cv_feat_e1 = _extract_cv_signals(cv_sc_e1)
    delta_I_e2, cv_feat_e2 = _extract_cv_signals(cv_sc_e2)

    for e in errors:
        st.warning(e)

    return {
        "rct_fit_e1":    rct_fit_e1,
        "rct_fit_e2":    rct_fit_e2,
        "rct_drt_e1":    rct_drt_e1,
        "rct_drt_e2":    rct_drt_e2,
        "delta_I_e1":    delta_I_e1,
        "delta_I_e2":    delta_I_e2,
        "eis_feat_e1":   eis_feat_e1,
        "eis_feat_e2":   eis_feat_e2,
        "cv_feat_e1":    cv_feat_e1,
        "cv_feat_e2":    cv_feat_e2,
        "probe_rct_fit": probe_rct_fit_new,
        "probe_rct_drt": probe_rct_drt_new,
        "probe_delta_I": probe_dI_new,
    }


# ---------------------------------------------------------------------------
# Visualisation — placer le point prédit sur la courbe de calibration
# ---------------------------------------------------------------------------

def _plot_prediction_on_calibration(
    method: str,
    result: dict,
    session_data: dict,
) -> go.Figure | None:
    """
    Trace la courbe de calibration OLS + le point de la mesure inconnue
    mis en évidence. Pour les méthodes B, retourne None (pas de courbe OLS).
    """
    if not method.startswith("A"):
        return None

    concs = np.asarray(session_data["concentrations"], dtype=float)
    log_c = np.log10(np.where(concs > 0, concs, np.nan))

    # Signal normalisé de calibration
    probe_val_map = {
        "A1": session_data.get("probe_delta_I", 0),
        "A2": session_data.get("probe_rct_fit", 0),
        "A3": session_data.get("probe_rct_drt", 0),
    }
    raw_map = {
        "A1": "cv_delta_I",
        "A2": "eis_rct_fit",
        "A3": "eis_rct_drt",
    }
    probe_val = probe_val_map.get(method, 0)
    raw_key   = raw_map.get(method, "cv_delta_I")

    raw_e1 = np.asarray(session_data[raw_key].get("e1", []), dtype=float)
    raw_e2_list = session_data[raw_key].get("e2")
    raw_e2 = np.asarray(raw_e2_list, dtype=float) if raw_e2_list else None

    def _norm(arr, p):
        return (arr - p) / abs(p) if p != 0 else arr

    norm1 = _norm(raw_e1, probe_val) if len(raw_e1) > 0 else None
    norm2 = _norm(raw_e2, probe_val) if raw_e2 is not None and len(raw_e2) > 0 else None
    norm_mean = (norm1 + norm2) / 2.0 if norm2 is not None and norm1 is not None else (norm1 or norm2)

    if norm_mean is None:
        return None

    valid = np.isfinite(norm_mean) & np.isfinite(log_c)
    if valid.sum() < 2:
        return None

    from scipy import stats as sp_stats
    r = sp_stats.linregress(norm_mean[valid], log_c[valid])
    x_fit = np.linspace(norm_mean[valid].min(), norm_mean[valid].max(), 200)
    y_fit = r.slope * x_fit + r.intercept

    color = _METHOD_COLORS.get(method, "#333333")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=norm_mean[valid], y=log_c[valid],
        mode="markers",
        name="Calibration",
        marker=dict(color=color, size=8, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode="lines",
        name=f"OLS (R²={r.rvalue**2:.3f})",
        line=dict(color=color, dash="dash", width=2),
    ))

    # Point mesuré inconnu — moyenne des deux électrodes si dispo
    log10_c_pred = result.get("log10_c_mean")
    if log10_c_pred is not None and np.isfinite(log10_c_pred):
        # Signal normalisé de la mesure inconnue (rétro-calculé depuis la prédiction)
        x_pred = (log10_c_pred - r.intercept) / r.slope if r.slope != 0 else None
        if x_pred is not None:
            sigma = result.get("sigma_pred")
            ci_y  = 1.96 * sigma if sigma else 0

            fig.add_trace(go.Scatter(
                x=[x_pred], y=[log10_c_pred],
                mode="markers",
                name="Mesure inconnue",
                marker=dict(color="gold", size=16, symbol="star",
                            line=dict(color="black", width=1.5)),
                error_y=dict(type="data", array=[ci_y], visible=ci_y > 0,
                             color="gold", thickness=2),
            ))

    fig.update_layout(
        title=f"Calibration {method} — point de mesure inconnue",
        xaxis_title="Signal normalisé",
        yaxis_title="log₁₀([c] / M)",
        height=380,
        margin=dict(t=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Affichage des résultats
# ---------------------------------------------------------------------------

def _render_results(predictions: dict, session_data: dict) -> None:
    """Affiche le tableau de résultats et les indicateurs de cohérence."""
    import pandas as pd

    # ---- Tableau principal ----
    st.subheader("Concentration estimée")

    rows = []
    for m, res in predictions.items():
        c_mean    = res.get("c_mean")
        log_mean  = res.get("log10_c_mean")
        ci_factor = res.get("ci_factor")
        sigma     = res.get("sigma_pred")
        coherent  = res.get("coherent", True)

        badge = "✓" if coherent else "⚠️"

        rows.append({
            "Méthode":             _METHOD_LABELS.get(m, m),
            "Concentration":       _fmt_conc(log_mean),
            "IC 95 % (facteur)":   _fmt_ci(ci_factor),
            "Incertitude (déc.)":  f"±{sigma:.3f}" if sigma else "—",
            "Inter-électrode":     badge,
        })

    st.dataframe(
        pd.DataFrame(rows).set_index("Méthode"),
        use_container_width=True,
    )

    # ---- Métriques visuelles (méthode par méthode) ----
    st.divider()
    st.subheader("Détail par méthode")

    n_methods = len(predictions)
    if n_methods == 0:
        st.warning("Aucune prédiction disponible.")
        return

    cols = st.columns(min(n_methods, 3))
    for (m, res), col in zip(predictions.items(), cols * ((n_methods // 3) + 1)):
        log_mean  = res.get("log10_c_mean")
        p1        = res.get("log10_c_e1")
        p2        = res.get("log10_c_e2")
        coherent  = res.get("coherent", True)
        delta     = res.get("delta_elec")
        sigma     = res.get("sigma_pred")
        ci_factor = res.get("ci_factor")

        with col:
            color = _METHOD_COLORS.get(m, "#333")
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:8px 12px;"
                f"border-radius:4px;background:#f8f9fa'>"
                f"<b>{_METHOD_LABELS.get(m, m)}</b></div>",
                unsafe_allow_html=True,
            )

            if log_mean is not None:
                st.metric("Estimation", _fmt_conc(log_mean))
                if ci_factor is not None:
                    st.caption(f"IC 95 % : ×{ci_factor:.1f}  (incertitude ±{sigma:.2f} décades)")
            else:
                st.metric("Estimation", "—")
                st.caption("Signal absent ou modèle non disponible")

            if p1 is not None and p2 is not None:
                st.caption(f"Électrode 1 : {_fmt_conc(p1)}")
                st.caption(f"Électrode 2 : {_fmt_conc(p2)}")
                if not coherent:
                    st.warning(
                        f"⚠️  MESURE SUSPECTE — |E1 − E2| = {delta:.3f} déc. "
                        f"> 2 × σ_inter calibration"
                    )
                else:
                    st.success("✓ Cohérence inter-électrode")
            elif p1 is not None:
                st.caption(f"Électrode 1 : {_fmt_conc(p1)}")
            elif p2 is not None:
                st.caption(f"Électrode 2 : {_fmt_conc(p2)}")

    # ---- Figures de calibration avec point ----
    st.divider()
    st.subheader("Position sur la courbe de calibration")

    methods_a = [m for m in predictions if m.startswith("A")]
    if methods_a:
        fig_cols = st.columns(len(methods_a))
        for col, m in zip(fig_cols, methods_a):
            fig = _plot_prediction_on_calibration(m, predictions[m], session_data)
            if fig:
                with col:
                    st.plotly_chart(fig, use_container_width=True)

    methods_b = [m for m in predictions if m.startswith("B")]
    if methods_b:
        st.markdown("**Méthodes B (PLS) — prédiction hors courbe OLS**")
        for m in methods_b:
            res = predictions[m]
            log_mean = res.get("log10_c_mean")
            st.caption(
                f"**{_METHOD_LABELS[m]}** : {_fmt_conc(log_mean)}"
                + (f" (±{res['sigma_pred']:.2f} déc.)" if res.get("sigma_pred") else "")
            )


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Inférence — Concentration inconnue",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🔬 Inférence de concentration")
    st.caption("Prédit la concentration d'une mesure inconnue à partir d'une session de calibration.")

    # -----------------------------------------------------------------------
    # Étape 1 — Source du modèle de calibration
    # -----------------------------------------------------------------------
    st.subheader("Étape 1 — Source du modèle de calibration")

    session_data = None

    has_active = "comparison_session_data" in st.session_state
    source = st.radio(
        "Source",
        options=["session_active", "yaml"],
        format_func=lambda x: {
            "session_active": "📊 Utiliser la session de calibration active (page C)",
            "yaml":           "📂 Importer un fichier YAML de session",
        }[x],
        horizontal=True,
        key="di_source",
    )

    if source == "session_active":
        if has_active:
            session_data = st.session_state["comparison_session_data"]
            # Attacher le rapport calculé pour les incertitudes
            if "comparison_report" in st.session_state:
                session_data = dict(session_data)
                session_data["_report"] = st.session_state["comparison_report"]
            concs = session_data.get("concentrations", [])
            st.success(
                f"✓ Session active chargée — {len(concs)} concentrations, "
                f"{session_data.get('n_electrodes', '?')} électrode(s)."
            )
            # Bouton export YAML pour archivage
            with st.expander("💾 Exporter la session en YAML"):
                yaml_str = export_inference_yaml(session_data)
                st.download_button(
                    "Télécharger session.yaml",
                    data=yaml_str.encode("utf-8"),
                    file_name="session_calibration.yaml",
                    mime="text/yaml",
                )
        else:
            st.warning(
                "⚠️  Aucune session de calibration active. "
                "Lancez d'abord une analyse sur la **page C — Comparaison des méthodes**."
            )
            st.stop()

    else:  # yaml
        yaml_file = st.file_uploader(
            "Fichier YAML de session (produit par l'export de la page C ou D)",
            type=["yaml", "yml"],
            key="di_yaml_upload",
        )
        if yaml_file is None:
            st.info("Importez un fichier YAML pour continuer.")
            st.stop()

        content, _ = _read_file(yaml_file)
        session_data = _load_session_from_yaml(content)
        if session_data is None:
            st.stop()

        st.success("✓ Session YAML chargée.")
        st.info(
            "ℹ️  Les données spectrales PLS ne sont pas stockées dans le YAML — "
            "les méthodes B1/B2/B3 ne seront pas disponibles."
        )

    # -----------------------------------------------------------------------
    # Étape 2 — Upload de la mesure inconnue
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Étape 2 — Mesure inconnue")
    st.caption("Le probe est obligatoire pour la normalisation. EIS et CV sont optionnels selon les méthodes souhaitées.")

    col_eis, col_cv = st.columns(2)

    with col_eis:
        st.markdown("**⚡ EIS**")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("*Électrode 1*")
            probe_eis_e1 = st.file_uploader("Probe EIS e1", type=["csv", "txt"], key="di_probe_eis_e1")
            eis_e1       = st.file_uploader("EIS e1",       type=["csv", "txt"], key="di_eis_e1")
        with col_e2:
            st.markdown("*Électrode 2*")
            probe_eis_e2 = st.file_uploader("Probe EIS e2", type=["csv", "txt"], key="di_probe_eis_e2")
            eis_e2       = st.file_uploader("EIS e2",       type=["csv", "txt"], key="di_eis_e2")

    with col_cv:
        st.markdown("**📈 CV**")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("*Électrode 1*")
            probe_cv_e1 = st.file_uploader("Probe CV e1", type=["csv", "txt"], key="di_probe_cv_e1")
            cv_e1       = st.file_uploader("CV e1",       type=["csv", "txt"], key="di_cv_e1")
        with col_e2:
            st.markdown("*Électrode 2*")
            probe_cv_e2 = st.file_uploader("Probe CV e2", type=["csv", "txt"], key="di_probe_cv_e2")
            cv_e2       = st.file_uploader("CV e2",       type=["csv", "txt"], key="di_cv_e2")

    probe_eis_present = probe_eis_e1 is not None or probe_eis_e2 is not None
    data_present      = any(f is not None for f in (eis_e1, eis_e2, cv_e1, cv_e2))

    if not probe_eis_present:
        st.info("Uploadez au minimum un probe EIS pour activer le bouton de prédiction.")

    predict_btn = st.button(
        "🔬 Prédire la concentration",
        type="primary",
        disabled=not probe_eis_present,
        use_container_width=True,
        key="di_predict_btn",
    )

    # -----------------------------------------------------------------------
    # Étape 3 — Prédiction et affichage
    # -----------------------------------------------------------------------
    if not predict_btn:
        return

    with st.spinner("Chargement des fichiers et calcul des fits…"):
        new_signals = _process_new_measurement(
            probe_eis_e1, probe_eis_e2,
            probe_cv_e1,  probe_cv_e2,
            eis_e1,       eis_e2,
            cv_e1,        cv_e2,
            session_data,
        )

    if new_signals is None:
        return

    with st.spinner("Prédiction en cours…"):
        predictions = predict_from_session(session_data, new_signals)

    if not predictions:
        st.error("Aucune prédiction disponible — vérifiez que les fichiers EIS et CV sont corrects.")
        return

    st.divider()
    _render_results(predictions, session_data)


if __name__ == "__main__":
    main()
