"""Gemeinsame Fixtures: das Handbeispiel in Brüchen, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wpi_dynamics import Plant  # noqa: E402


@pytest.fixture
def hand():
    """Strecke y⁺ = (y + u)/2 (a = 1/2, G = 1), PI mit kp = 1 und ki = 1/2, Sollwert 3 in den Perioden 0 und 1, danach 1/4; alles in Brüchen.
    Spektralradius der linearen Schleife: Doppelwurzel 1/2 (z² − z + 1/4)."""
    return dict(plant=Plant("lag1", (F(1, 2), 0), F(1)), kp=F(1), ki=F(1, 2), t1=2, r1=F(3), r2=F(1, 4))
