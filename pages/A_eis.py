"""Page A — Analyse EIS seule.

Charge les spectres d'impédance depuis experiment_clean (session_state), puis lance
core.pipeline.run_pipeline par électrode : pour chaque groupe de réplicats, measurement
model + verdict Kramers-Kronig AVANT le fit, fit Orazem du circuit défini ici (chaque
réplicat + la moyenne) et DRT (chaque réplicat + la moyenne). Les groupes arrêtés
(structure d'erreur non caractérisable, AUDIT.md ERR-1) sont signalés explicitement.

Résultats présentés en QUATRE onglets (ui/tabs.py::render_eis_tabs) : Visualisation (spectres
mesurés), Measurement model & fit Orazem, DRT, Calibration.
"""

import copy
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from circuit import CircuitError, parse_circuit
from core.analysis_io import AnalysisFileError, ANALYSIS_KEYS, count_drt, load_analysis, save_analysis
from core.app_state import preprocessing_ready, reset_analysis_results
from core.drt_recompute import STORE_KEY, compute_pending_drt, pending_targets
from core.config import load_config, config_to_dict
from core.loader import load_spectrum, average_replicates
from core.models import DisplayGroup
from core.pipeline import InvalidAnalysisInput, build_circuit_fit, run_pipeline
from drt import engine as drt_engine
from plotting.eis_plots import _spectrum_label
from ui.tabs import render_eis_tabs

_DEFAULT_CONFIG = config_to_dict(load_config())

#: Clés de st.session_state des résultats EIS (sous-ensemble de core.app_state).
_EIS_RESULT_KEYS = ("eis_sessions", "eis_validations", "eis_normalized", "eis_drt_store",
                    "eis_drt_pending")

#: Clé de st.session_state : manifeste de l'analyse rechargée depuis un fichier (None sinon).
_LOADED_KEY = "eis_loaded_manifest"


def _render_analysis_file_io(has_results: bool) -> None:
    """Chargement d'une analyse sauvegardée (toujours proposé) et bouton de sauvegarde.

    Une analyse rechargée remplace l'analyse courante SANS aucun calcul : spectres, fits et DRT
    déjà calculées sont relus tels quels.
    """
    with st.expander("💾 Sauvegarder / charger une analyse (DRT comprises)", expanded=not has_results):
        up = st.file_uploader("Charger une analyse sauvegardée (.zip)", type=["zip"],
                              key="eis_analysis_upload")
        if up is not None and st.button("📂 Charger cette analyse", key="eis_analysis_load_btn"):
            try:
                loaded = load_analysis(up.getvalue())
            except AnalysisFileError as exc:
                st.error(f"Chargement impossible : {exc}")
            else:
                reset_analysis_results(st.session_state)
                for k in ANALYSIS_KEYS:
                    st.session_state[k] = loaded[k]
                if st.session_state.get(STORE_KEY) is None:
                    st.session_state[STORE_KEY] = {}
                st.session_state["eis_drt_pending"] = None
                st.session_state[_LOADED_KEY] = loaded["manifest"]
                st.rerun()
        if has_results:
            sessions = st.session_state["eis_sessions"]
            done, total = count_drt(sessions)
            pending = bool(st.session_state.get("eis_drt_pending"))
            st.caption(f"DRT calculées : {done}/{total} spectre(s)."
                       + (" ⏳ Calcul en cours : attendez la fin pour une sauvegarde complète."
                          if pending else ""))
            try:
                data = save_analysis(st.session_state)
            except AnalysisFileError as exc:
                st.error(str(exc))
            else:
                st.download_button(
                    "💾 Sauvegarder l'analyse (spectres, fits et DRT)", data=data,
                    file_name=f"analyse_eis_{datetime.now():%Y%m%d_%H%M}.zip", mime="application/zip",
                    key="eis_analysis_save_btn",
                    help="Enregistre tous les résultats, DRT calculées comprises : rechargez-les plus "
                         "tard ici, sans rien recalculer. Ne contient pas les fichiers de mesure "
                         "(pour ceux-ci, voir la page Import).")


def _progress_bar(total: int):
    """Callback ``on_progress`` affichant « n/N — spectre en cours » dans une ``st.progress``."""
    bar = st.progress(0.0, text=f"0/{total}")

    def on_progress(done: int, tot: int, label: str) -> None:
        bar.progress(done / tot if tot else 1.0,
                     text=f"{done}/{tot}" + (f" — DRT de {label}…" if label else " — terminé"))

    return on_progress


def _run_deferred_drt(sessions: dict) -> None:
    """Calcule la DRT en attente APRÈS l'affichage des onglets, puis rafraîchit la page.

    Reprenable : une interaction pendant le calcul relance le script, qui ne recalcule que les
    spectres encore sans DRT.
    """
    mode = st.session_state.get("eis_drt_pending")
    if not mode:
        return
    targets = pending_targets(sessions)
    if targets:
        ok, why = drt_engine.engine_available()
        if not ok:
            for s in sessions.values():
                s.messages.append(f"DRT non calculée : moteur indisponible — {why}.")
            st.session_state["eis_drt_pending"] = None
            st.rerun()
        st.info(f"⏳ Calcul de la DRT ({len(targets)} spectre(s), mode {mode}) — les onglets "
                "ci-dessus sont déjà utilisables ; l'onglet DRT se met à jour à la fin.")
        try:
            compute_pending_drt(sessions, st.session_state.get("eis_config"), mode,
                                st.session_state.get(STORE_KEY), _progress_bar(len(targets)))
        except InvalidAnalysisInput as exc:
            st.session_state["eis_drt_pending"] = None
            st.error(f"DRT non calculée : {exc}")
            return
    st.session_state["eis_drt_pending"] = None
    st.rerun()


def _render_analysis_status(sessions: dict) -> None:
    """Statut EXPLICITE de l'analyse : groupes arrêtés, fichiers écartés, moteur DRT.

    Jamais « ✅ Analyse terminée » quand un groupe n'a produit aucun fit (ERR-1).
    """
    n_total = n_failed = 0
    for e, session in sorted(sessions.items()):
        for msg in session.messages:
            st.warning(f"Électrode {e} : {msg}")
        for le in session.load_errors:
            st.warning(f"Électrode {e} : fichier « {le['filename']} » écarté — {le['message']}")
        for _lbl, _sp, _reps, an in session.iter_groups():
            n_total += 1
            if an is not None and not an.ok:
                n_failed += 1
                st.error(f"Électrode {e} — {an.message}")
    if n_total == 0:
        st.warning("⚠️ Aucun spectre EIS exploitable. Vérifiez le prétraitement.")
    elif n_failed:
        st.error(
            f"❌ Analyse terminée avec **{n_failed} groupe(s) ARRÊTÉ(S)** sur {n_total} : "
            "aucun fit (ni circuit ni DRT) pour ces groupes — voir les messages ci-dessus."
        )
    else:
        st.success(f"✅ Analyse terminée — {n_total} groupe(s) analysé(s) sur "
                   f"{len(sessions)} électrode(s).")


def _num_or_none(v):
    """Cellule du tableau → float, ou None si vide / NaN."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _circuit_editor(defaults: dict):
    """Saisie du circuit, des guess/bornes et du paramètre cible.

    Returns:
        (CircuitFit ou None si la saisie est invalide — l'erreur est affichée, description
        sérialisable de la saisie pour détecter une modification après l'analyse).
    """
    c = defaults["fit"]["circuit"]
    expr = st.text_area(
        "Circuit équivalent (syntaxe : docs/CIRCUIT_UTILISATEUR.md)",
        value=c["expression"], key="eis_circuit_expr", height=80,
        help="Ex. Re + parallel(R(Rct), Q(Qdl, alpha)). `+` = série, parallel(a, b, …) = "
             "parallèle ; tout nom autre que w et les éléments est un paramètre ajusté.",
    )
    try:
        _z, names = parse_circuit(expr)
    except CircuitError as exc:
        st.error(f"Circuit invalide : {exc}")
        return None, None
    params_cfg = c.get("parameters") or {}
    table = pd.DataFrame([
        {"paramètre": n,
         "initial": (params_cfg.get(n) or {}).get("initial"),
         "borne basse": (params_cfg.get(n) or {}).get("lower"),
         "borne haute": (params_cfg.get(n) or {}).get("upper")}
        for n in names
    ])
    st.caption(
        "Guess de départ et bornes de chaque paramètre (case vide = non borné). Valeurs "
        "proposées : config/default.yaml (`fit.circuit`) ; un paramètre nouveau doit être "
        "renseigné. Le fit part de ce guess puis de 7 perturbations (±0,5 décade)."
    )
    edited = st.data_editor(table, key=f"eis_circuit_params_{'_'.join(names)}",
                            disabled=["paramètre"], hide_index=True, width='stretch')
    default_target = c.get("target_param")
    target = st.selectbox("Paramètre cible (signal de calibration)", names,
                          index=names.index(default_target) if default_target in names else 0,
                          key="eis_circuit_target")
    params = {}
    for _i, row in edited.iterrows():
        lo, hi = _num_or_none(row["borne basse"]), _num_or_none(row["borne haute"])
        params[row["paramètre"]] = {
            "initial": _num_or_none(row["initial"]),
            "lower": -math.inf if lo is None else lo,
            "upper": math.inf if hi is None else hi,
        }
    fingerprint = {"expression": expr, "parameters": params, "target": target}
    try:
        return build_circuit_fit(expr, params, target), fingerprint
    except InvalidAnalysisInput as exc:
        st.error(str(exc))
        return None, fingerprint


def _load_bare_eis(experiment: dict, elec_idx: int):
    """Charge et moyenne les fichiers « électrode nue » EIS d'une électrode.

    Réutilise les loaders EXISTANTS (load_spectrum + average_replicates) — aucun
    parsing maison. Retourne un EISSpectrum (moyenne des réplicats, UNE seule
    trace) ou None. Référence d'AFFICHAGE SEULE : jamais passée au pipeline.
    """
    key = f"electrode_{elec_idx}"
    bare_dict = (experiment.get("bare") or {}).get("eis") or {}
    bare_files = bare_dict.get(key) or []

    specs = []
    for ri, bio in enumerate(bare_files):
        if bio is None:
            continue
        try:
            bio.seek(0)
            sp = load_spectrum(
                bio.read(),
                label="Électrode nue (réf.)",
                concentration=0.0,
                step="bare",
            )
            bio.seek(0)
            specs.append(sp)
        except Exception:
            pass

    if not specs:
        return None
    return average_replicates(specs) if len(specs) > 1 else specs[0]


def _load_raw_groups(experiment: dict, elec_idx: int, config: dict) -> list:
    """Groupes BRUTS d'une électrode (``DisplayGroup``) : spectres de l'expérience AVANT les
    exclusions du prétraitement, pour l'onglet « Visualisation ».

    Mêmes loaders et même regroupement que le pipeline ; AFFICHAGE SEUL (rattaché à
    ``session.raw_groups`` après l'analyse, jamais lu par ``run_pipeline``). Un fichier
    illisible est ignoré : le pipeline l'a déjà signalé (``session.load_errors``).
    """
    by_key: dict = {}
    for fa in _build_file_assignments_electrode(experiment, elec_idx):
        try:
            sp = load_spectrum(content=fa["content"], label=fa["filename"],
                               concentration=fa["concentration"], step=fa["step"], config=config)
        except ValueError:
            continue
        by_key.setdefault((fa["step"], fa["concentration"]), []).append(sp)
    groups = []
    for (step, conc), reps in sorted(by_key.items(), key=lambda kv: (kv[0][0] != "probe", kv[0][1])):
        label = "Probe" if step == "probe" else f"{conc:.2e} M"
        groups.append(DisplayGroup(label=label, concentration=float(conc), step=step,
                                   mean=average_replicates(reps), replicates=reps))
    return groups


def _build_normalized_session(sessions: dict) -> dict:
    """Combine les sessions par électrode en spectres normalisés par concentration.

    Retourne {label_concentration: {"Zre_norm": array, "Zim_norm": array, "concentration": float}}
    en moyennant les versions normalisées de chaque électrode disponible.
    """
    def _normalize_electrode(session) -> dict:
        """Zre_norm/Zim_norm = |(Z_probe - Z_Ci) / Z_probe|, point à point, par concentration."""
        probe = session.probe if session is not None else None
        if probe is None:
            return {}
        out = {}
        for grp in session.groups:
            sp = grp.spectrum
            if len(sp.f) == len(probe.f):
                Zre_c, Zim_c = np.asarray(sp.Zre), np.asarray(sp.Zim)
            else:
                log_f_probe = np.log10(np.asarray(probe.f, dtype=float))
                log_f_c     = np.log10(np.asarray(sp.f, dtype=float))
                order = np.argsort(log_f_c)
                Zre_c = np.interp(log_f_probe, log_f_c[order], np.asarray(sp.Zre)[order])
                Zim_c = np.interp(log_f_probe, log_f_c[order], np.asarray(sp.Zim)[order])

            Zre_probe = np.asarray(probe.Zre)
            Zim_probe = np.asarray(probe.Zim)
            with np.errstate(invalid="ignore", divide="ignore"):
                Zre_norm = np.abs((Zre_probe - Zre_c) / Zre_probe)
                Zim_norm = np.abs((Zim_probe - Zim_c) / Zim_probe)

            out[grp.concentration] = {
                "label":         _spectrum_label(grp.spectrum),
                "f":             probe.f,
                "Zre_norm":      Zre_norm,
                "Zim_norm":      Zim_norm,
                "concentration": grp.concentration,
            }
        return out

    norm_e1 = _normalize_electrode(sessions.get(1))
    norm_e2 = _normalize_electrode(sessions.get(2))

    if not norm_e1 and not norm_e2:
        return {}
    if not norm_e2:
        return norm_e1
    if not norm_e1:
        return norm_e2

    result = {}
    all_concs = set(norm_e1) | set(norm_e2)
    for conc in all_concs:
        d1 = norm_e1.get(conc)
        d2 = norm_e2.get(conc)
        if d1 is not None and d2 is not None and len(d1["Zre_norm"]) == len(d2["Zre_norm"]):
            result[conc] = {
                "label":         d1["label"],
                "f":             d1["f"],
                "Zre_norm":      (d1["Zre_norm"] + d2["Zre_norm"]) / 2,
                "Zim_norm":      (d1["Zim_norm"] + d2["Zim_norm"]) / 2,
                "concentration": conc,
            }
        else:
            result[conc] = d1 if d1 is not None else d2
    return result


def _build_file_assignments_electrode(experiment: dict, elec_idx: int) -> list:
    """Convertit experiment_clean en liste de file_assignments pour une seule électrode."""
    mode = experiment.get("mode", "both")
    concentrations = experiment.get("concentrations") or []
    assignments = []

    if mode not in ("eis_only", "both"):
        return assignments

    probe_eis = (experiment.get("probe") or {}).get("eis") or {}
    cal_eis   = (experiment.get("calibration") or {}).get("eis") or {}

    key = f"electrode_{elec_idx}"
    for ri, bio in enumerate(probe_eis.get(key) or []):
        if bio is None:
            continue
        bio.seek(0)
        content = bio.read()
        bio.seek(0)
        assignments.append({
            "content":       content,
            "filename":      f"probe_e{elec_idx}_r{ri + 1}.csv",
            "step":          "probe",
            "concentration": 0.0,
        })
    for ci, rep_list in enumerate(cal_eis.get(key) or []):
        conc = concentrations[ci] if ci < len(concentrations) else 0.0
        for ri, bio in enumerate(rep_list or []):
            if bio is None:
                continue
            bio.seek(0)
            content = bio.read()
            bio.seek(0)
            assignments.append({
                "content":       content,
                "filename":      f"e{elec_idx}_c{ci + 1}_r{ri + 1}.csv",
                "step":          "hybridization",
                "concentration": conc,
            })

    return assignments


def main() -> None:
    st.title("📡 Analyse EIS — Spectroscopie d'impédance")
    st.caption("Visualisation · Measurement model & Kramers-Kronig, fit Orazem du circuit · DRT · Calibration")

    has_results = bool(st.session_state.get("eis_sessions"))
    _render_analysis_file_io(has_results)

    # Vérification que les données sont disponibles (B-STATE-b : jamais de KeyError). Une analyse
    # rechargée d'un fichier se consulte sans données de mesure.
    can_analyze = preprocessing_ready(st.session_state)
    if not has_results and not can_analyze:
        st.warning(
            "⚠️ Aucune donnée disponible. "
            "Importez et prétraitez vos données d'abord, ou chargez une analyse sauvegardée."
        )
        st.page_link("pages/0_import.py", label="→ Aller à l'import", icon="📂")
        st.stop()

    experiment = st.session_state.get("experiment_clean")

    with st.expander("⚙️ Circuit, DRT et pondération", expanded=False):
        circuit_fit, circuit_inputs = _circuit_editor(_DEFAULT_CONFIG)

        st.markdown("**DRT** (drt/engine.py — chaque réplicat ET la moyenne de chaque groupe)")
        drt_cfg = _DEFAULT_CONFIG["fit"]["drt"]
        run_drt = st.checkbox("Calculer la DRT", value=bool(drt_cfg.get("enabled", True)),
                              key="eis_drt_enabled")
        drt_mode = st.radio(
            "Mode DRT", ["optimize", "sample"],
            index=0 if drt_cfg.get("mode", "optimize") == "optimize" else 1,
            format_func=lambda m: ("MAP (optimize) — ~1 s/spectre, sans diagnostic de convergence"
                                   if m == "optimize" else
                                   "HMC (sample) — 2 à 5 min/spectre, R̂/divergences/ESS + intervalles"),
            key="eis_drt_mode", disabled=not run_drt,
        )

        st.markdown("**Pondération du fit**")
        st.caption(
            "⚖️ Méthode UNIQUE : structure d'erreur d'Orazem σ = α|Zj| + β|Zr − R_sol| + "
            "γ|Z|² + δ, caractérisée par le measurement model sur les réplicats BRUTS de "
            "CHAQUE groupe (≥ 3), à chaque analyse ; poids = 1/σ² (σ/√n pour la moyenne). "
            "χ²ᵣ ≈ 1 est un vrai test d'adéquation. Sans réplicats en nombre suffisant, le "
            "groupe est **arrêté** avec un message (aucun repli arbitraire)."
        )

    cfg = copy.deepcopy(_DEFAULT_CONFIG)   # copie isolée : le pipeline ne touche jamais au défaut
    inputs = {"circuit": circuit_inputs, "run_drt": run_drt, "drt_mode": drt_mode}

    if st.button("↺ Relancer l'analyse", key="eis_rerun_btn", disabled=not can_analyze,
                 help=None if can_analyze else "Les données de mesure ne sont pas chargées "
                                               "(analyse rechargée d'un fichier)."):
        for key in _EIS_RESULT_KEYS:
            st.session_state[key] = None
        st.session_state["eis_failures"] = None
        st.session_state[_LOADED_KEY] = None
        st.rerun()

    if not st.session_state.get("eis_sessions"):
        if circuit_fit is None:
            st.warning("⚠️ Corrigez le circuit ou ses paramètres (section ⚙️ ci-dessus) "
                       "pour lancer l'analyse.")
            return
        n_elec = experiment.get("n_electrodes", 2)
        sessions = {}
        validations = {}
        failures: dict = {}      # {n° électrode: message} — conservé pour les reruns suivants
        with st.spinner("Analyse EIS en cours… (la DRT est calculée ensuite, spectres déjà visibles)"):
            # Isolation PAR ÉLECTRODE : une exception sur l'une ne fait pas perdre les autres.
            for e in range(1, n_elec + 1):
                try:
                    file_assignments = _build_file_assignments_electrode(experiment, e)
                    if not file_assignments:
                        continue
                    # DRT différée (run_drt=False) : les spectres et les fits s'affichent sans
                    # l'attendre ; elle est calculée ensuite par _run_deferred_drt.
                    session, vr_pipeline = run_pipeline(
                        file_assignments, cfg, circuit_fit, run_drt=False,
                    )
                    # Référence « électrode nue » — attachée APRÈS l'analyse,
                    # jamais lue par run_pipeline (affichage seul).
                    session.bare_reference = _load_bare_eis(experiment, e)
                    # Spectres BRUTS (avant exclusions) — même statut : affichage seul.
                    raw_experiment = st.session_state.get("experiment")
                    if raw_experiment:
                        session.raw_groups = _load_raw_groups(raw_experiment, e, cfg)
                    sessions[e] = session
                    validations[e] = vr_pipeline or None
                except InvalidAnalysisInput as exc:      # saisie invalide : message clair
                    failures[e] = f"⚠️ Électrode {e} non analysée : {exc}"
                except Exception as exc:                 # bug logiciel : jamais avalé
                    failures[e] = (
                        f"⚠️ Électrode {e} non analysée — erreur LOGICIELLE (elle ne vient "
                        "pas de vos données ; merci de la signaler). "
                        f"Les autres électrodes sont conservées. Détail : {exc!r}"
                    )
                    st.exception(exc)

        if failures and not sessions:       # rien à conserver : le prochain rerun réessaie
            for msg in failures.values():
                st.warning(msg)
            return
        st.session_state["eis_failures"] = failures

        st.session_state["eis_sessions"]    = sessions
        st.session_state["eis_drt_store"]   = {}     # nouvelle analyse : registre DRT vierge
        st.session_state["eis_config"]      = cfg
        st.session_state["eis_validations"] = validations
        st.session_state["eis_inputs"]      = inputs
        st.session_state[_LOADED_KEY]       = None
        st.session_state["eis_drt_pending"] = drt_mode if run_drt else None

    sessions = st.session_state.get("eis_sessions")
    if not sessions:
        st.warning("⚠️ Aucun spectre EIS trouvé dans l'expérience. Vérifiez le prétraitement.")
        return

    for msg in (st.session_state.get("eis_failures") or {}).values():
        st.warning(msg)

    loaded = st.session_state.get(_LOADED_KEY)
    if loaded:
        st.success(f"📂 Analyse rechargée depuis un fichier (sauvegardée le "
                   f"{str(loaded.get('saved_at', '?')).replace('T', ' à ')}) — aucun calcul refait.")
    elif st.session_state.get("eis_inputs") != inputs:
        st.info("ℹ️ Le circuit ou les réglages DRT ont changé depuis cette analyse : "
                "cliquez sur **↺ Relancer l'analyse** pour les appliquer.")

    _render_analysis_status(sessions)

    # Verdict KK, alertes et diagnostics de fit : onglet 2 (visibles d'emblée) ; diagnostics DRT :
    # onglet 3. Pas de récapitulatif séparé ici, qui dupliquerait les onglets.
    normalized = _build_normalized_session(sessions)
    st.session_state["eis_normalized"] = normalized
    render_eis_tabs(sessions, normalized=normalized)
    _run_deferred_drt(sessions)     # EN DERNIER : les onglets ci-dessus sont déjà affichés


if __name__ == "__main__":
    main()
