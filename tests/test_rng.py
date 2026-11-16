"""Zufallsgenerator: SplitMix64 gegen den veröffentlichten Referenzwert, vektorisierte Folge gleich der Klasse, Zusatzfunktionen der Demo."""

import math

import numpy as np
import pytest

import wpi_rng as R


def test_splitmix64_reference_value_for_seed_zero():
    """Der bekannte erste Ausgabewert des SplitMix64 mit Zustand 0 ist 0xE220A8397B1DCDAF (Referenzimplementierung von Vigna)."""
    assert R.SplitMix64(0).next_u64() == 0xE220A8397B1DCDAF


def test_uniform_array_equals_the_class_sequence():
    g = R.SplitMix64(15)
    assert np.array_equal(R.uniform_array(15, 500), np.array([g.random() for _ in range(500)]))
    x = R.uniform_array(1506, 1000)
    assert x.min() >= 0.0 and x.max() < 1.0 and abs(float(x.mean()) - 0.5) < 0.05


def test_exact_exp_matches_math_exp_and_inverts_exact_log():
    for x in (-3.0, -0.5, 0.0, 0.25, 1.0, 4.0):
        assert R.exact_exp(x) == pytest.approx(math.exp(x), rel=1e-15)
    for x in (0.1, 0.5, 1.0, 2.5, 100.0):
        assert R.exact_exp(R.exact_log(x)) == pytest.approx(x, rel=1e-15)
    assert R.exact_exp(0.0) == 1.0


def test_exact_log_and_cos_match_the_math_module_closely():
    assert R.exact_log(math.e) == pytest.approx(1.0, rel=1e-15) and R.exact_cos(0.0) == 1.0 and R.exact_cos(math.pi) == pytest.approx(-1.0, rel=1e-15)
    for x in (0.1, 0.5, 1.0, 1.5):
        assert R.exact_cos(x) == pytest.approx(math.cos(x), rel=1e-14)


def test_same_seed_same_sequence_other_seed_other_sequence():
    assert R.uniform_array(5, 20).tolist() == R.uniform_array(5, 20).tolist() and R.uniform_array(5, 20).tolist() != R.uniform_array(6, 20).tolist()
