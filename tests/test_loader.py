"""Tests for core/loader.py."""

import numpy as np
import pytest

from core.loader import load_spectrum, average_replicates
from core.cv_loader import load_cv_file, load_cv_curve
from core.models import CVCurve, EISSpectrum
from core.robust_loader import parse_eclab_file


def _make_csv(n: int = 20, neg_zim: bool = False) -> bytes:
    """Generate CSV with frequencies spaced to avoid the 50/100 Hz parasitic filter."""
    lines = ["frequency_hz,zreal_ohm,zimag_ohm"]
    # Use 200 Hz – 10000 Hz range to stay clear of 50 Hz / 100 Hz ± 3 Hz filters
    freqs = np.logspace(np.log10(200), 4, n)
    for i, f in enumerate(freqs):
        zre = 500 + i * 100
        zim = 200 - i * 5
        if neg_zim:
            zim = -zim
        lines.append(f"{f:.4f},{zre:.2f},{zim:.2f}")
    return "\n".join(lines).encode()


def test_load_valid_csv_points_and_order():
    data = _make_csv(20)
    sp = load_spectrum(data, "test.csv", concentration=1e-9, step="hybridization")
    assert sp.n_points == 20, f"Expected 20 points, got {sp.n_points}"
    assert sp.f[0] > sp.f[-1], "Frequencies should be sorted HF→BF"


def test_load_valid_csv_zim_positive():
    data = _make_csv(20)
    sp = load_spectrum(data, "test.csv", concentration=1e-9, step="hybridization")
    assert np.all(sp.Zim >= 0), "Zim should be positive after loading"


def test_load_negative_zim_corrected():
    data = _make_csv(20, neg_zim=True)
    sp = load_spectrum(data, "neg.csv", concentration=1e-9, step="hybridization")
    assert np.all(sp.Zim >= 0), "Negative Zim (EC-Lab convention) should be auto-corrected"


def test_load_too_few_points_raises():
    lines = ["frequency_hz,zreal_ohm,zimag_ohm"]
    for i in range(3):
        lines.append(f"{1000 - i * 100},{500 + i},{100 + i}")
    data = "\n".join(lines).encode()
    with pytest.raises(ValueError, match="moins de 5 points"):
        load_spectrum(data, "tiny.csv")


def test_load_tab_separator():
    lines = ["frequency_hz\tzreal_ohm\tzimag_ohm"]
    # Stay above 200 Hz to avoid 50/100 Hz ± 3 Hz parasitic filter
    for i in range(10):
        f = 10 ** (4 - i * 0.2)  # 10000 Hz down to ~630 Hz
        lines.append(f"{f:.2f}\t{500 + i * 50:.2f}\t{100 + i * 10:.2f}")
    data = "\n".join(lines).encode()
    sp = load_spectrum(data, "tab.txt")
    assert sp.n_points == 10


def test_average_replicates_mean():
    f = np.array([1000.0, 100.0, 10.0])
    sp1 = EISSpectrum(
        label="a", f=f, Zre=np.array([100.0, 200.0, 300.0]),
        Zim=np.array([50.0, 100.0, 150.0]),
        concentration=1e-9, step="hybridization", n_points=3,
    )
    sp2 = EISSpectrum(
        label="b", f=f, Zre=np.array([110.0, 210.0, 310.0]),
        Zim=np.array([60.0, 110.0, 160.0]),
        concentration=1e-9, step="hybridization", n_points=3,
    )
    avg = average_replicates([sp1, sp2])
    assert np.allclose(avg.Zre, [105.0, 205.0, 305.0])
    assert np.allclose(avg.Zim, [55.0, 105.0, 155.0])


def test_average_single_returns_same():
    sp = EISSpectrum(
        label="x", f=np.array([100.0, 10.0]),
        Zre=np.array([500.0, 600.0]),
        Zim=np.array([50.0, 60.0]),
        concentration=0.0, step="bare", n_points=2,
    )
    assert average_replicates([sp]) is sp


def test_average_empty_raises():
    with pytest.raises(ValueError):
        average_replicates([])


# ─────────────────────────────────────────────────────────────────────────────
# Intégration du parseur EC-Lab robuste (core/robust_loader.py) dans le loader
# ─────────────────────────────────────────────────────────────────────────────

def _write(tmp_path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_robust_eis_fr_comma_tab_negative_im(tmp_path):
    """EIS FR : tabulation + virgule décimale, Im(Z) négatif → Zim positif."""
    text = (
        "freq/Hz\tRe(Z)/Ohm\tIm(Z)/Ohm\n"
        "1,0000186E+006\t2,1766477E+003\t-2,4268961E+002\n"
    )
    pf = parse_eclab_file(_write(tmp_path, "eis_fr.txt", text))
    assert pf.kind == "EIS"
    assert pf.f[0] == pytest.approx(1.0000186e6, rel=1e-6)
    assert pf.Zre[0] == pytest.approx(2176.6477, rel=1e-6)
    # Im(Z) exporté négatif → convention -Im(Z) > 0 → positif
    assert pf.Zim[0] == pytest.approx(242.68961, rel=1e-6)
    assert pf.Zim[0] > 0


def test_robust_cv_fr_comma_tab_ma_to_ampere(tmp_path):
    """CV FR : tabulation + virgule décimale, courant en mA converti en A."""
    text = (
        "Ewe/V\t<I>/mA\n"
        "-9,2510954E-003\t-6,674337782897055E-004\n"
    )
    pf = parse_eclab_file(_write(tmp_path, "cv_fr.txt", text))
    assert pf.kind == "CV"
    assert pf.columns["Ewe"][0] == pytest.approx(-9.2510954e-3, rel=1e-6)
    # -6,674...E-004 mA → × 1e-3 → -6.674e-7 A
    assert pf.columns["I"][0] == pytest.approx(-6.674337782897055e-7, rel=1e-9)


def test_robust_eis_neg_im_header_stays_positive_no_double_flip(tmp_path):
    """En-tête '-Im(Z)/Ohm' (valeur positive), séparé par espaces : Zim reste
    positif et n'est PAS ré-inversé (anti-double-signe)."""
    text = (
        "freq/Hz Re(Z)/Ohm -Im(Z)/Ohm Phase/deg\n"
        "1000000 2176.65 242.69 -6.3\n"
        "100000 2100.00 300.00 -5.0\n"
        "10000 2000.00 350.00 -4.0\n"
        "1000 1900.00 400.00 -3.0\n"
        "500 1850.00 420.00 -2.5\n"
        "200 1800.00 440.00 -2.0\n"
    )
    pf = parse_eclab_file(_write(tmp_path, "eis_negimz.txt", text))
    assert pf.kind == "EIS"
    assert np.all(pf.Zim > 0), "'-Im(Z)' positif doit rester positif (pas de flip)"
    assert pf.Zim[0] == pytest.approx(242.69, rel=1e-4)

    # Bout en bout via load_spectrum : toujours positif, aucun double signe.
    sp = load_spectrum(text.encode(), "eis_negimz.txt")
    assert np.all(sp.Zim > 0)
    assert sp.f[0] > sp.f[-1], "tri HF→BF"
    # La valeur HF (1 MHz) conserve son -Im(Z) = 242.69, non ré-inversée.
    assert sp.Zim[0] == pytest.approx(242.69, rel=1e-4)


def test_robust_unknown_type_warns_without_exception(tmp_path):
    """Un fichier de type inconnu produit un warning, pas une exception."""
    text = "time/s\ttemperature/C\n0.0\t25.0\n1.0\t25.1\n"
    pf = parse_eclab_file(_write(tmp_path, "weird.txt", text))
    assert pf.kind == "UNKNOWN"
    assert pf.warnings, "un type indéterminé doit remonter un warning"


def _eclab_eis_fr_multirow(n: int = 12) -> bytes:
    """Fichier EC-Lab FR (tab + virgule décimale, Im(Z) négatif) à n points,
    dans la plage 200 Hz–1 MHz pour éviter le filtre parasite 50/100 Hz."""
    lines = ["freq/Hz\tRe(Z)/Ohm\tIm(Z)/Ohm"]
    freqs = np.logspace(np.log10(200), 6, n)
    for i, f in enumerate(freqs):
        zre = 500 + i * 100
        zim = -(200 + i * 5)  # Im(Z) négatif (convention EC-Lab)
        lines.append(
            f"{f:.6E}\t{zre:.6E}\t{zim:.6E}".replace(".", ",")
        )
    return "\n".join(lines).encode()


def test_load_spectrum_uses_robust_parser_fr(tmp_path):
    """load_spectrum lit un export EC-Lab FR complet via le parseur robuste :
    Zim positif, tri HF→BF, ≥5 points."""
    sp = load_spectrum(_eclab_eis_fr_multirow(12), "eis_fr_multi.txt",
                       concentration=1e-9, step="hybridization")
    assert isinstance(sp, EISSpectrum)
    assert sp.n_points >= 5
    assert np.all(sp.Zim >= 0), "Zim doit être positif (-Im(Z) > 0)"
    assert sp.f[0] > sp.f[-1], "tri HF→BF"


def test_load_spectrum_rejects_cv_file_clearly(tmp_path):
    """Un fichier CV envoyé au loader EIS remonte une erreur claire (pas muette)."""
    text = "Ewe/V\t<I>/mA\n" + "\n".join(
        f"{-0.5 + i * 0.01:.4f}\t{0.1 * i:.4f}".replace(".", ",") for i in range(10)
    )
    with pytest.raises(ValueError, match="CV"):
        load_spectrum(text.encode(), "cv_misrouted.txt")


def test_load_cv_file_fr_returns_amperes(tmp_path):
    """load_cv_file lit un export CV FR : E en volts, I converti en ampères."""
    text = (
        "Ewe/V\t<I>/mA\n"
        + "\n".join(
            f"{-0.5 + i * 0.05:.4f}\t{1.0}".replace(".", ",") for i in range(6)
        )
    )
    scan = load_cv_file(text.encode(), "cv_fr.txt", concentration=1e-9, step="probe")
    assert scan.E.size == 6
    # 1 mA → 1e-3 A
    assert np.allclose(scan.I, 1e-3)
    assert np.all(np.diff(scan.E) >= 0), "tri par potentiel croissant"


def test_load_cv_curve_returns_cvcurve(tmp_path):
    """load_cv_curve renvoie la dataclass légère CVCurve(Ewe, I, label)."""
    text = (
        "Ewe/V\t<I>/mA\n"
        "-9,2510954E-003\t-6,674337782897055E-004\n"
        "1,0000000E-002\t-5,000000000000000E-004\n"
    )
    curve = load_cv_curve(text.encode(), "cv_curve.txt")
    assert isinstance(curve, CVCurve)
    assert curve.label == "cv_curve.txt"
    assert curve.Ewe.size == 2 and curve.I.size == 2
    # courant déjà en ampères
    assert np.all(np.abs(curve.I) < 1e-3)
