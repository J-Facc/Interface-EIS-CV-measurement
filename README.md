# EIS Analyzer

Application web locale (Python/Streamlit) pour l'analyse de spectres
d'impédance électrochimique (EIS) appliquée à des biosenseurs microfluidiques
ADN/ARN.

## Démarrage rapide

### Windows (utilisateur) — un seul fichier

1. Installez **Python 3.12 ou plus récent** depuis https://www.python.org/downloads/
   (cochez *Add python.exe to PATH* ; le stub du Microsoft Store ne convient pas).
2. Téléchargez [`launch.bat`](launch.bat), placez-le dans un **dossier court** (par exemple
   `C:\EIS`) et double-cliquez dessus.

Un dossier **sans accents ni caractères spéciaux** reste le choix le plus sûr : sous Windows,
`mingw32-make` reçoit un chemin accentué (`C:\Users\x\Desktop\.Thèse\…`) corrompu et échoue. L'application
le contourne — les modèles Stan s'y compilent depuis un cache ASCII (`C:\cmdstan\model_cache`,
`drt/stan_compile.py`) — et `launch.bat` signale un tel chemin au démarrage ; si malgré tout la
compilation du moteur DRT échoue, déplacez le dossier (par exemple `C:\EIS_Analyzer`) et relancez.
CmdStan lui-même doit rester dans un dossier sans accents (`C:\cmdstan` par défaut).

C'est tout : le dépôt est public, aucun jeton n'est nécessaire. À chaque lancement, `launch.bat`

* vérifie la dernière version sur GitHub (validation TLS normale) et ne la télécharge que si
  elle a changé ; l'ancienne version n'est remplacée qu'**après** extraction et contrôle de la
  nouvelle ;
* **hors ligne, ou si cette vérification échoue pour quelque raison que ce soit**, il lance
  directement la version déjà installée, sans rien tenter d'autre (le message distingue
  « HORS LIGNE » d'un « ECHEC REEL ») ;
* crée l'environnement Python (`venv\`) et n'installe les dépendances (versions épinglées) que
  si elles ont changé ;
* installe au **premier lancement** le moteur DRT (CmdStan 2.36.0 — la version sur laquelle les
  réglages DRT ont été validés — dans `C:\cmdstan`, puis compilation des deux modèles Stan
  `Series` et `Series_pos` : plusieurs minutes, une seule fois ; les modèles compilés survivent
  aux mises à jour). Une **autre version** de CmdStan déjà présente est conservée, jamais
  supprimée, mais n'est pas utilisée tant que la 2.36.0 est installable : si elle ne l'est pas
  (hors ligne), l'application s'en sert **en le signalant** dans l'onglet DRT. Si cette étape
  échoue, l'application démarre sans DRT ;
* ouvre le navigateur sur http://localhost:8501 (ou le premier port libre jusqu'à 8510)
  **seulement quand le serveur répond**.

Fermez la fenêtre noire pour arrêter l'application. Le journal du lanceur est `launcher.log`.

### Développement (Linux, macOS, ou Windows sans le lanceur)

```bash
python3.12 -m venv .venv
source .venv/bin/activate              # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt    # exécution + pytest
pip install -r requirements-drt.txt    # facultatif : DRT (cvxopt, cmdstanpy)
python setup_drt_bayesien.py --ensure  # facultatif : CmdStan + modèles Stan (idempotent)
streamlit run app.py
```

Sans l'extra DRT, l'application fonctionne : l'analyse par circuit reste complète et
l'onglet DRT l'indique. L'application elle-même n'installe jamais rien ; l'installation est
l'affaire de `launch.bat` ou de `setup_drt_bayesien.py`.

## Utilisation (v3)

Cinq pages, dans l'ordre :

1. **Configuration** — biomarqueurs (noms mémorisés dans `user_data/biomarkers.json`, à côté de
   `launch.bat` : ils survivent aux mises à jour), canaux (1 canal = 2 électrodes), biomarqueur et
   concentrations de chaque canal. Le mode est **déduit** : 1 concentration → « une concentration
   par canal », plusieurs → « plusieurs concentrations sur un canal ». On peut aussi y **charger une
   archive** `.zip` (v3 : tout est restauré sans recalcul ; v2 : configuration et fichiers bruts).
2. **Import & nettoyage** — **un dossier par canal**, tel qu'il sort du poste de mesure. Fichiers
   EC-Lab `<libre>_<NN>_<CV|PEIS>_C<nn>.mpr` (fin fixe : réplicat, technique, voie de l'instrument) ;
   la partie libre contient l'électrode (`w3`) et `bare`, `probe` ou la concentration (`10-18` =
   1e-18 M), à n'importe quelle position — ex. `10-18_w3_01_CV_C01.mpr`. `.mpr` converti ;
   `.txt/.csv/.mpt` aussi lus. Les autres fichiers (autres techniques, réglages, journaux…) sont
   ignorés. Les concentrations trouvées peuvent être adoptées pour le canal ; le tableau
   d'assignation reste corrigeable ; contrôles de cohérence, revue des spectres EIS graphe par graphe
   (sélection au lasso → suppression et spectre suivant, boutons Précédent/Suivant), puis
   nettoyage. Les réplicats sont conservés individuellement.
3. **Visualisation** — menu « Canal » : **vue synthétique** par défaut (toutes les courbes moyennes,
   avec zone d'erreur ± écart-type), ou un canal : électrode 1, électrode 2 et les moyennes
   (E1, E2, E1+E2), en CV, Nyquist ou Bode. Aucun fit ni DRT.
4. **Analyse** — « Lancer l'analyse » : fits Randles (Orazem) et DRT de tous les spectres **en
   arrière-plan** (processus parallèles), puis onglets **DRT · Rct (fit Randles) · Étalonnage EIS ·
   CV · Étalonnage CV**.
5. **Export** — archive `.zip` rechargeable, rapport **PDF** (A4) et rapport **HTML** interactif.

## Format des fichiers d'entrée

- EIS : export EC-Lab texte (`freq/Hz`, `Re(Z)/Ohm`, `-Im(Z)/Ohm`) ou `.mpr` ; séparateur et
  décimale auto-détectés ; signe de Zim corrigé ; fréquences parasites 50/100 Hz retirées ; tri HF → BF.
- CV : `Ewe/V`, `<I>/mA` (ou `I/…`, converti en A), `cycle number` ; le **dernier cycle** est gardé
  par défaut (`cv.cycle`), dans l'ordre de mesure.

## Analyse EIS (`core/analysis.py` → `core/pipeline.py`)

Pour CHAQUE groupe de réplicats (canal, électrode, étape), dans cet ordre :

| Étape | Module | Ce qui est produit |
|-------|--------|--------------------|
| 1. Measurement model + Kramers-Kronig | `core/measurement_model.py` | structure d'erreur σ(ω) et verdict KK, sur les réplicats **bruts**, avant tout fit ([docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md)) |
| 2. Fit Orazem du circuit utilisateur | `fits/orazem_fit.py`, `circuit/` | CNLS pondéré 1/σ² de **chaque réplicat** et de la **moyenne** (σ/√n) ; paramètre cible agrégé : incertitude intra-fit vs variabilité inter-réplicats ([docs/CIRCUIT_UTILISATEUR.md](docs/CIRCUIT_UTILISATEUR.md)) |
| 3. DRT bayésienne | `drt/engine.py` (clone identifié de [bayes-drt2](https://github.com/jdhuang-csm/bayes-drt2), `drt/PROVENANCE.md`) | γ(τ) de **chaque réplicat** et de la **moyenne** ; Rct = arc de transfert (pic pénultième, convention Bissessur) ; mode `optimize` (MAP, ~1 s) ou `sample` (HMC, R̂/divergences/ESS + intervalles, 2 à 5 min par spectre) |

Un groupe dont la structure d'erreur n'est pas caractérisable (moins de 3 réplicats…) est
quand même ajusté, en **repli** signalé : pondération proportionnelle σ = 1 %·|Z|, verdict KK
« indéterminé ». Un verdict KK « non conforme » est **signalé**, jamais excluant.

**Rct par DRT** = aire du pic **principal** de γ(τ) (le plus haut dans la fenêtre
`fit.drt.tau_window`, défaut [1e-5 ; 1e-1] s), intégrée entre les minima qui l'encadrent
(`core/drt_peak.py`) ; le Rct « pic pénultième » du moteur reste affiché en contrôle.

**Étalonnages** (`core/calibration.py`) : chaque réplicat est normalisé par la moyenne « probe »
de **son** électrode — EIS : log10(Rct/Rct_probe) (Rct par DRT ou par le fit), CV :
ΔI/I = (I_probe − I)/I_probe au pic anodique (`cv.peak_window_V`) — puis moyenne ± écart-type
(ddof = 1) par concentration, régression sur log10([c]) ; vues « par électrode » et « regroupé
par biomarqueur ». Les anciennes courbes (spectre moyen, |Rct_probe − Rct_c|/Rct_probe et
log-log) restent disponibles.

Le calcul DRT compile des modèles Stan via CmdStan : la toolchain C++ (RTools 4.0, mingw-w64)
est installée par `launch.bat` au premier lancement (`install_cmdstan(compiler=True)`, sans
exiger d'installer RTools soi-même) et **vérifiée** — `mingw32-make` et `g++` répondent — avant
toute compilation, puis les modèles sont mis en cache — voir [docs/DRT_BAYESIENNE.md](docs/DRT_BAYESIENNE.md) pour lire les intervalles de
crédibilité et les diagnostics. La figure DRT trace γ(τ) (Ω) en fonction de τ (s, échelle log), le pic principal (★) et la
zone intégrée, avec le mode (`optimize`/`sample`) dans la légende. Référence DRT : Bissessur, Man, Gamby, *Use of an
approach with a distribution of relaxation times for impedance analysis of a
channel electrode in microfluidics*, Phys. Rev. E **113**, 025502 (2026), DOI:
10.1103/fn2s-z364.

## Tests

```bash
python -m pytest tests/ -v                # ajouter -m "not slow" pour exclure le HMC lent
```

## Documentation

| Document | Contenu |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | architecture v3 : arborescence, modèle de données, flux, archive, lanceur |
| [docs/REFERENCE_MAINTENANCE.html](docs/REFERENCE_MAINTENANCE.html) | aide-mémoire de maintenance : où modifier quoi, réglages, tests, pannes fréquentes |
| [docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md) | measurement model, structure d'erreur d'Orazem, Kramers-Kronig, fit |
| [docs/CIRCUIT_UTILISATEUR.md](docs/CIRCUIT_UTILISATEUR.md) | écrire son propre circuit équivalent |
| [docs/DRT_BAYESIENNE.md](docs/DRT_BAYESIENNE.md) | DRT, HMC/NUTS, lecture des intervalles de crédibilité |
| [drt/PROVENANCE.md](drt/PROVENANCE.md), [drt/VALIDATION_REGLAGES.md](drt/VALIDATION_REGLAGES.md) | origine du code DRT, validation des réglages |

Les anciens rapports d'audit sont dans l'historique git : `git show 60371c7:AUDIT.md`.
