"""Tests pour core/mpr_converter.py."""

import pytest

from core.mpr_converter import is_mpr, mpr_to_csv_bytes


def test_is_mpr():
    assert is_mpr("fichier.mpr")
    assert is_mpr("fichier.MPR")
    assert not is_mpr("fichier.csv")


def test_mpr_to_csv_bytes_raises_on_invalid_content():
    with pytest.raises(RuntimeError):
        mpr_to_csv_bytes(b"ceci n'est pas un fichier .mpr valide" * 10)
