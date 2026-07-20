"""
robust_loader.py — Lecture robuste des fichiers EC-Lab (EIS et CV) en ASCII.

Gère :
  - separateur decimal virgule (export FR) OU point
  - separateur de colonnes : tabulation, point-virgule, ou espaces multiples
  - encodages Windows (cp1252 / latin-1) avec repli automatique
  - preambule EC-Lab .mpt ("Nb header lines : N")
  - mapping des colonnes PAR NOM (robuste a l'ordre et aux colonnes en trop)
  - discrimination automatique EIS vs CV
  - convention de signe de Im(Z) : "Im(Z)" (negatif) vs "-Im(Z)" (positif)

Renvoie un ParsedFile neutre, a convertir en EISSpectrum par le loader appelant.
Aucune dependance au reste du projet -> testable isolement.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

# unites de courant -> facteur vers ampere (pour la CV)
_I_UNIT_TO_A = {"a": 1.0, "ma": 1e-3, "ua": 1e-6, "µa": 1e-6, "na": 1e-9}


@dataclass
class ParsedFile:
    kind: str                                     # "EIS" | "CV" | "UNKNOWN"
    columns: Dict[str, np.ndarray]                # cles normalisees
    n_rows: int
    warnings: List[str] = field(default_factory=list)
    header_tokens: List[str] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)

    # raccourcis EIS
    @property
    def f(self):   return self.columns.get("freq")
    @property
    def Zre(self): return self.columns.get("Zre")
    @property
    def Zim(self): return self.columns.get("Zim")   # convention -Im(Z) > 0


# --------------------------------------------------------------------------
# Lecture texte avec repli d'encodage
# --------------------------------------------------------------------------
def _read_text(path: str) -> str:
    for enc in _ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as fh:
                return fh.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, "rb") as fh:            # dernier repli : ne peut pas echouer
        return fh.read().decode("latin-1")


# --------------------------------------------------------------------------
# Detection du delimiteur de colonnes a partir de la ligne d'en-tete
# (l'en-tete ne contient pas de nombre, donc pas d'ambiguite avec la virgule)
# --------------------------------------------------------------------------
def _detect_delimiter(header_line: str) -> Optional[str]:
    if "\t" in header_line:
        return "\t"
    if ";" in header_line:
        return ";"
    if "," in header_line:                  # en-tete separee par des virgules => CSV US
        return ","
    return None                             # None => decoupage sur espaces (str.split())


def _split(line: str, delim: Optional[str]) -> List[str]:
    parts = line.split(delim) if delim is not None else line.split()
    return [p.strip() for p in parts]


# --------------------------------------------------------------------------
# Identification de la ligne d'en-tete
# --------------------------------------------------------------------------
_HEADER_TOKENS = ("freq", "re(z)", "im(z)", "ewe", "<i>", "i/", "ecell")


def _find_header(lines: List[str]) -> int:
    # indice EC-Lab explicite ?
    for i, ln in enumerate(lines[:5]):
        low = ln.lower()
        if "nb header lines" in low:
            for tok in ln.replace(":", " ").split():
                if tok.isdigit():
                    return int(tok) - 1     # la Nieme ligne est l'en-tete de colonnes
    # sinon : premiere ligne contenant un token connu
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(tok in low for tok in _HEADER_TOKENS):
            return i
    return 0


# --------------------------------------------------------------------------
# Mapping d'un token d'en-tete vers une cle normalisee (+ signe / unite)
# --------------------------------------------------------------------------
def _classify(token: str):
    """Retourne (cle, meta) ou (None, None) si colonne ignoree.
    meta['sign'] pour Zim ; meta['unit'] pour le courant."""
    t = token.lower().replace(" ", "")
    if "freq" in t:
        return "freq", {}
    if "re(z)" in t:
        return "Zre", {}
    if "im(z)" in t:
        # signe : "-im(z)" deja positif ; "im(z)" a inverser
        sign = +1.0 if t.lstrip().startswith("-im(z)") or "-im(z)" in t else -1.0
        return "Zim", {"sign": sign}
    if "ewe" in t or t.startswith("e/v") or t == "ecell/v":
        return "Ewe", {}
    # courant : "<i>/ma", "i/ma", "i/µa"... (apres im(z) pour ne pas capter le "i")
    if t.startswith("<i>") or t.startswith("i/") or "<i>/" in t:
        unit = "a"
        for u in ("ma", "µa", "ua", "na", "a"):
            if t.endswith("/" + u) or t.endswith(u):
                unit = u
                break
        return "I", {"unit": unit}
    return None, None


# --------------------------------------------------------------------------
# Parseur principal
# --------------------------------------------------------------------------
def parse_eclab_file(path: str) -> ParsedFile:
    raw = _read_text(path)
    lines = [ln for ln in raw.splitlines()]
    if not lines:
        return ParsedFile("UNKNOWN", {}, 0, ["fichier vide"])

    h_idx = _find_header(lines)
    header_line = lines[h_idx]
    delim = _detect_delimiter(header_line)
    do_decimal_fix = delim != ","           # si delim=',' les decimales sont des points
    header_tokens = _split(header_line, delim)

    # colonne d'index -> (cle, meta)
    colmap: Dict[int, tuple] = {}
    for j, tok in enumerate(header_tokens):
        if not tok:
            continue
        key, meta = _classify(tok)
        if key is not None and key not in (k for k, _ in colmap.values()):
            colmap[j] = (key, meta)

    warnings: List[str] = []
    data: Dict[str, List[float]] = {key: [] for key, _ in colmap.values()}
    n_bad = 0

    for ln in lines[h_idx + 1:]:
        if not ln.strip():
            continue
        fields = _split(ln, delim)
        if len(fields) < len(header_tokens):
            # ligne incomplete (commentaire, ligne de queue...) -> on tente quand meme
            if len(fields) <= max(colmap):
                n_bad += 1
                continue
        try:
            row = {}
            for j, (key, meta) in colmap.items():
                s = fields[j]
                if do_decimal_fix:
                    s = s.replace(",", ".")
                val = float(s)
                if key == "Zim":
                    val *= meta["sign"]     # -> convention -Im(Z) > 0
                if key == "I":
                    val *= _I_UNIT_TO_A.get(meta["unit"], 1.0)  # -> ampere
                row[key] = val
            for key, v in row.items():
                data[key].append(v)
        except (ValueError, IndexError):
            n_bad += 1
            continue

    if n_bad:
        warnings.append(f"{n_bad} ligne(s) non numerique(s) ignoree(s)")

    columns = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    n_rows = len(next(iter(columns.values()))) if columns else 0

    # discrimination EIS vs CV
    # EIS reconnu dès que Re(Z) ET Im(Z) sont présents ; la colonne fréquence est
    # facultative (certains exports EC-Lab ne fournissent que Re(Z)/-Im(Z), sans
    # fréquence : l'appelant reconstruira l'axe fréquence à partir de la plage de
    # balayage). Si la fréquence est absente, on le signale.
    if {"Zre", "Zim"} <= set(columns):
        kind = "EIS"
        if "freq" not in columns:
            warnings.append(
                "colonne frequence absente ; axe frequence a reconstruire "
                "depuis la plage de balayage"
            )
    elif {"Ewe", "I"} <= set(columns):
        kind = "CV"
    else:
        kind = "UNKNOWN"
        warnings.append(f"type indetermine ; colonnes trouvees: {sorted(columns)}")

    meta = {"delimiter": repr(delim), "decimal_comma": str(do_decimal_fix),
            "header_line_index": str(h_idx)}
    return ParsedFile(kind, columns, n_rows, warnings, header_tokens, meta)
