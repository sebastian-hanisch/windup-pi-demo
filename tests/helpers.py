"""Gemeinsame Hilfen der Tests: zufällige Strecken mit stabiler Schleife (feste Folge aus SplitMix64)."""

import wpi_dynamics as D
from wpi_dynamics import Plant
from wpi_rng import SplitMix64, exact_cos


def random_plant(rng: SplitMix64):
    """Zufällige Strecke (vier Arten) mit PI-Satz, lineare Schleife stabil (ρ < 0,95); feste Folge aus SplitMix64."""
    while True:
        art = rng.below(4)
        gain = 0.5 + 2.5 * rng.random()
        if art == 0:
            plant = Plant("lag1", (0.3 + 0.67 * rng.random(), 0), gain)
        elif art == 1:
            plant = Plant("lag1", (0.3 + 0.65 * rng.random(), 1 + rng.below(8)), gain)
        elif art == 2:
            plant = Plant("lag2", (0.3 + 0.65 * rng.random(), 0.3 + 0.65 * rng.random()), gain)
        else:
            r, th = 0.6 + 0.33 * rng.random(), 0.15 + 0.85 * rng.random()
            plant = Plant("osc", (2 * r * exact_cos(th), r * r), gain)
        kp, ki = 0.3 + 3.7 * rng.random(), 0.03 + 0.37 * rng.random()
        if D.rho_pi(plant, kp, ki) < 0.95:
            return plant, kp, ki, gain
