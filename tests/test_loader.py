"""Tests for core/loader.py."""

import numpy as np
import pytest

from core.loader import load_spectrum, average_replicates
from core.models import EISSpectrum


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
