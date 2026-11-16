"""Orakel-Tests (ORACLE-PLAYBOOK §2, §5; DEMO-PLAYBOOK §12): der Kern gegen einen ANDEREN Rechenweg, schnell (< 10 s).

(a) Ausstieg aus der Sättigung: Vorhersage aus der offenen Stufenantwort und Summen gegen die Regler-Simulation (kleine Zufallsmenge mit festem Zufalls-Seed; exakter Vergleich nur
    dort, wo der Abstand von v zur Grenze über 1e-6 liegt, sonst wäre er ein Gleichstand, den die Rundung entscheidet).
(b) Brüche (`fractions`) gegen Gleitkomma: dieselbe Bahn, Abweichung unter 1e-12.
(c) Rekursion I⁺ = (1 − kaw)·I + (ki − kaw·kp)·e + kaw in der Sättigung gegen die Simulation (Fall a1: 4,925 gegen 4,925).
(d) Identität Σe = (kaw/ki)·Σ(v − u) + ΔI/ki aus demselben Lauf.
(e) Spektralradius der linearen Schleife aus den Nullstellen des charakteristischen Polynoms gegen die Eigenwerte der Zustandsmatrix.
(f) Stufenantwort der Strecke gegen `scipy.signal.dlsim` (Zustandsraummodell aus `Plant.matrices`).
(g) Die ganze Regelschleife gegen eine zweite Implementierung mit Matrizen (numpy) statt der Zustandsobjekte des Kerns.
Referenzwerte: Handrechnung in tests/test_dynamics.py; die Zahlen der README stehen in tests/test_claims.py."""

from fractions import Fraction as F

import numpy as np
import pytest

import wpi_constants as C
import wpi_dynamics as D
from wpi_dynamics import Plant, simulate
from helpers import random_plant as _random_plant
from wpi_rng import SplitMix64


# --- (a) Vorhersage aus der Stufenantwort ------------------------------------------------------------------------------------------------------

def test_exit_prediction_equals_the_simulation_on_random_cases():
    rng = SplitMix64(2026)
    n_checked = 0
    for _ in range(50):
        plant, kp, ki, gain = _random_plant(rng)
        t1 = 5 + rng.below(145)
        r1 = gain * (1.5 + 4.5 * rng.random())
        r2 = gain * (0.2 + 0.7 * rng.random())
        vor, abstand, status = D.predict_exit(plant, kp, ki, t1, r1, r2)
        if status != "ok" or abstand < 1e-6:
            continue                                              # Annahme (Sättigung bis zum Sprung) verletzt oder Gleichstand: kein exakter Vergleich
        sim = simulate(plant, kp, ki, "keine", 0.0, t1, r1, r2)
        assert vor == sim.ausstieg, (plant, kp, ki, t1, r1, r2)
        n_checked += 1
    assert n_checked >= 30                                        # die Prüfung greift in den meisten Fällen (gemessen 41 von 50)


def test_exit_prediction_on_the_24_measured_cases():
    for cid, _label, kind, par, gain, kp, ki in C.FAELLE:
        plant = Plant(kind, par, gain)
        vor, abstand, status = D.predict_exit(plant, kp, ki, C.SZ_T1, C.SZ_R1, C.SZ_R2)
        assert status == "ok" and abstand > 1e-4, cid
        assert vor == simulate(plant, kp, ki, "keine", 0.0, C.SZ_T1, C.SZ_R1, C.SZ_R2).ausstieg, cid


# --- (b) Brüche gegen Gleitkomma ---------------------------------------------------------------------------------------------------------------

def test_fraction_run_equals_floating_point_run_within_1e_12():
    for var, kaw in (("keine", 0), ("bc", F(1, 2)), ("bed", 0), ("ilim", 0)):
        _, bx = simulate(Plant("lag1", (F(1, 2), 0), F(1)), F(1), F(1, 2), var, F(kaw), 2, F(3), F(1, 4), horizon=60, still=None, record=True, tol=F(1, 100))
        _, bf = simulate(Plant("lag1", (0.5, 0), 1.0), 1.0, 0.5, var, float(kaw), 2, 3.0, 0.25, horizon=60, still=None, record=True, tol=0.01)
        for name in ("y", "v", "u", "I"):
            assert max(abs(float(a) - b) for a, b in zip(getattr(bx, name), getattr(bf, name))) < 1e-12, (var, name)


def test_v6_case_in_fractions_exits_at_1215_and_in_floating_point_at_1216():
    """Gleichstand v = 1: In Brüchen kommt v − 1 (fällt je Periode um genau 0,04) nach 1 215 Perioden auf Null, in Gleitkomma bleibt ein Rest von rund 7e-13 und der Ausstieg liegt bei 1 216."""
    px = Plant("lag1", (F(9, 10), 0), F(1))
    kz = simulate(px, F(2), F(1, 5), "keine", F(0), 60, F(5), F(4, 5), horizon=1300, still=None, tol=F(2, 100))
    assert kz.ausstieg == 1215
    kzf = simulate(Plant("lag1", (0.9, 0), 1.0), 2.0, 0.2, "keine", 0.0, 60, 5.0, 0.8, horizon=1300, still=None)
    assert kzf.ausstieg in (1215, 1216)                           # die Rundung entscheidet: Band statt exakt


# --- (c) Rekursion in der Sättigung ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kaw", [0.5, 1.0, 1.5])
def test_saturation_recursion_equals_the_simulation(kaw):
    """Fall a1, 57 gesättigte Perioden: |I| = 4,925 (kaw = 0,5), 5,665 (1,0), 5,913 (1,5) in Simulation und Rekursion (mit den Fehlern des Laufs)."""
    plant = Plant("lag1", (0.9, 0), 1.0)
    kz, b = simulate(plant, 2.0, 0.2, "bc", kaw, 57, 4.7, 0.77, horizon=2, still=None, record=True)
    # Die Rekursion gilt nur, solange v gesättigt bleibt. Bei kaw = 1,5 dippt v in den ersten Perioden unter die Grenze (−2,46 in k = 1, 0,87 in k = 2); die Rekursion ist
    # kontrahierend (Faktor 0,5), der Unterschied klingt in 12 Perioden auf unter 1e-3 und bis Periode 57 auf unter 1e-14 ab: das Ergebnis stimmt trotzdem.
    assert all(v > 1 for v in b.v[{0.5: 0, 1.0: 0, 1.5: 12}[kaw]:57])
    rec = D.integrator_recursion(2.0, 0.2, kaw, b.e[:57])
    assert rec == pytest.approx(b.I[57], rel=1e-9)
    assert abs(b.I[57]) == pytest.approx({0.5: 4.925, 1.0: 5.665, 1.5: 5.913}[kaw], abs=1e-3)


# --- (d) Identität der Summen --------------------------------------------------------------------------------------------------------------------

def test_sum_identity_holds_in_every_back_calculation_run():
    rng = SplitMix64(77)
    for _ in range(20):
        plant, kp, ki, gain = _random_plant(rng)
        kaw = 0.05 + 1.5 * rng.random()
        noise = D.uniform_noise(rng.below(1000), 3000, 0.3 * rng.random())
        res = D.noise_run(plant, kp, ki, "bc", kaw, 0.9 * gain, noise, n_warm=500)
        if res["divergiert"]:
            continue
        assert res["identity_left"] == pytest.approx(res["identity_right"], rel=1e-9, abs=1e-9)


def test_without_back_calculation_the_mean_error_vanishes_when_the_integrator_stays_bounded():
    """kaw = 0: Σe = ΔI/ki; bei beschränktem I ist das Mittel von e über viele Perioden klein (hier unter 1e-3 bei 20 000 Perioden)."""
    plant = Plant("lag1", (0.9, 0), 1.0)
    res = D.noise_run(plant, 2.0, 0.2, "keine", 0.0, 0.9, D.uniform_noise(15, 22000, 0.4), n_warm=2000)
    assert abs(res["mean_e"]) < 1e-3
    assert res["sum_e"] == pytest.approx((res["I_end"] - res["I_start"]) / 0.2, rel=1e-9, abs=1e-9)


# --- (e) Spektralradius -------------------------------------------------------------------------------------------------------------------------

def test_spectral_radius_from_the_characteristic_polynomial():
    rng = SplitMix64(5)
    for _ in range(60):
        a = 0.99 * rng.random()
        gain = 0.5 + 2.5 * rng.random()
        kp, ki = 0.1 + 4.9 * rng.random(), 0.01 + 0.49 * rng.random()
        roots = np.roots(D.char_poly_lag1(a, gain, kp, ki))
        assert D.rho_pi(Plant("lag1", (a, 0), gain), kp, ki) == pytest.approx(float(max(abs(roots))), rel=1e-6, abs=1e-6)


# --- (f) Stufenantwort gegen scipy.signal.dlsim --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("plant", [Plant("lag1", (0.9, 0), 1.0), Plant("lag1", (0.8, 5), 2.0), Plant("lag2", (0.8, 0.9), 1.0), Plant("osc", (1.6, 0.8), 1.0), Plant("osc", (1.2, 0.5), 2.0)])
def test_step_response_equals_scipy_dlsim(plant):
    signal = pytest.importorskip("scipy.signal")
    A, B, C_, = plant.matrices()
    _t, y, _x = signal.dlsim((A, B, C_, np.zeros((1, 1)), 1.0), np.ones((60, 1)), t=np.arange(60))
    assert np.allclose(y[:, 0], plant.step_response(60), atol=1e-12)


# --- (g) Regelschleife mit Matrizen --------------------------------------------------------------------------------------------------------------

def _matrix_loop(plant, kp, ki, variant, kaw, t1, r1, r2, n):
    A, B, C_ = plant.matrices()
    x = np.zeros((A.shape[0], 1))
    I = 0.0
    out = []
    for k in range(n):
        r = r1 if k < t1 else r2
        y = float((C_ @ x)[0, 0])
        e = r - y
        v = kp * e + I
        u = float(np.clip(v, -1.0, 1.0))
        if variant == "keine":
            I_next = I + ki * e
        elif variant == "bc":
            I_next = I + ki * e + kaw * (u - v)
        elif variant == "bed":
            I_next = I if (abs(v) > 1 and e * (v - u) > 0) else I + ki * e
        else:
            I_next = float(np.clip(I + ki * e, -1.0, 1.0))
        out.append((y, v, u, I))
        x = A @ x + B * u
        I = I_next
    return np.array(out)


def test_kernel_loop_equals_the_matrix_loop_on_the_24_cases():
    for cid, _label, kind, par, gain, kp, ki in C.FAELLE:
        plant = Plant(kind, par, gain)
        for var, kaw in (("keine", 0.0), ("bc", ki / kp), ("bed", 0.0), ("ilim", 0.0)):
            _kz, b = simulate(plant, kp, ki, var, kaw, C.SZ_T1, C.SZ_R1, C.SZ_R2, horizon=343, still=None, record=True)
            ref = _matrix_loop(plant, kp, ki, var, kaw, C.SZ_T1, C.SZ_R1, C.SZ_R2, 400)
            got = np.array([b.y, b.v, b.u, b.I]).T
            assert np.allclose(got, ref, atol=1e-8, rtol=1e-9), (cid, var)


def test_matrix_loop_agrees_on_random_plants_with_dead_time_and_oscillation():
    rng = SplitMix64(31)
    for _ in range(25):
        plant, kp, ki, gain = _random_plant(rng)
        _kz, b = simulate(plant, kp, ki, "bc", ki / kp, 40, 3 * gain, 0.6 * gain, horizon=260, still=None, record=True)
        ref = _matrix_loop(plant, kp, ki, "bc", ki / kp, 40, 3 * gain, 0.6 * gain, 300)
        assert np.allclose(np.array([b.y, b.v, b.u, b.I]).T, ref, atol=1e-7, rtol=1e-8)
