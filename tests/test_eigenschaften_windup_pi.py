"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Beziehungen, die für Regler mit Stellgrenze gelten müssen, ohne dass man das richtige Ergebnis kennt.
Je Eigenschaft 25 bis 300 Zufallsinstanzen aus SplitMix64 mit festem Zufalls-Seed, zusammen unter 10 s.

Gilt für diesen Regler (PI, symmetrische Grenze), die vier Streckenarten der Demo mit stabiler linearer Schleife (ρ < 0,95) und die Sollwertfolge r1 für T1 Perioden, danach r2."""

import numpy as np
import pytest

import wpi_constants as C
import wpi_dynamics as D
from wpi_dynamics import Plant, simulate
from wpi_rng import SplitMix64

from helpers import random_plant as _random_plant

VARIANTS = (("keine", 0.0), ("bc", 0.5), ("bed", 0.0), ("ilim", 0.0))


def _sequence_run(plant, kp, ki, variant, kaw, setpoints):
    """Eigene Schleife über eine beliebige Sollwertfolge (für die Zeitinvarianz): Listen y, v, u, I je Periode."""
    state = plant.state()
    I = 0.0
    ys, vs, us, Is = [], [], [], []
    for r in setpoints:
        y = state.y
        e = r - y
        v = kp * e + I
        u = D.sat(v)
        state.step(u)
        ys.append(y); vs.append(v); us.append(u); Is.append(I)
        I = D.integrator_update(variant, I, e, v, u, ki, kaw)
    return ys, vs, us, Is


def test_without_saturation_all_variants_give_the_same_path():
    """Bleibt |v| ≤ 1, schneidet die Grenze nie ab: die Rückrechnung addiert u − v = 0, die bedingte Integration hält nie an, die Integratorgrenze greift nicht (bit-gleich).
    Hier 300 Zufallsinstanzen mit kleinem Sollwert; die Instanzen, in denen |v| doch über 1 kommt, zählen nicht (gemessen unter 1 % davon)."""
    rng = SplitMix64(4242)
    checked = 0
    for _ in range(300):
        plant, kp, ki, gain = _random_plant(rng)
        r1, r2 = 0.1 * gain * rng.random(), 0.1 * gain * rng.random()
        runs = [simulate(plant, kp, ki, var, kaw, 60, r1, r2, horizon=120, still=None, record=True)[1] for var, kaw in VARIANTS]
        if max(abs(v) for v in runs[0].v) > 1:
            continue
        checked += 1
        for b in runs[1:]:
            assert b.y == runs[0].y and b.v == runs[0].v and b.I == runs[0].I
    assert checked >= 270


def test_back_calculation_with_kaw_zero_is_no_anti_windup():
    rng = SplitMix64(11)
    for _ in range(25):
        plant, kp, ki, gain = _random_plant(rng)
        t1, r1, r2 = 5 + rng.below(100), gain * (1.5 + 3 * rng.random()), gain * (0.2 + 0.7 * rng.random())
        assert simulate(plant, kp, ki, "bc", 0.0, t1, r1, r2) == simulate(plant, kp, ki, "keine", 0.0, t1, r1, r2)


def test_time_invariance_prefix_of_zero_setpoint_changes_nothing():
    """Bei r = 0 bleibt alles im Nullzustand (e = 0, v = 0, u = 0, I = 0): eine Vorlaufzeit mit r = 0 verschiebt die Bahn nur."""
    rng = SplitMix64(23)
    for _ in range(25):
        plant, kp, ki, gain = _random_plant(rng)
        t1, r1, r2 = 5 + rng.below(60), gain * (1.5 + 3 * rng.random()), gain * (0.2 + 0.7 * rng.random())
        m = 1 + rng.below(40)
        seq = [r1] * t1 + [r2] * 300
        for var, kaw in VARIANTS:
            a = _sequence_run(plant, kp, ki, var, kaw, seq)
            b = _sequence_run(plant, kp, ki, var, kaw, [0.0] * m + seq)
            for xa, xb in zip(a, b):
                assert xb[:m] == [0.0] * m and xb[m:] == xa, (var, m)


def test_scaling_the_whole_problem_by_powers_of_two_keeps_all_times():
    """Grenze, r1 und r2 mit c multipliziert (Strecke und Regler linear): y, u, I wachsen mit c, alle Zeiten bleiben gleich. Potenzen von 2 rechnen in Gleitkomma exakt."""
    rng = SplitMix64(37)
    for _ in range(25):
        plant, kp, ki, gain = _random_plant(rng)
        t1, r1, r2 = 5 + rng.below(100), gain * (1.5 + 3 * rng.random()), gain * (0.2 + 0.7 * rng.random())
        for var, kaw in VARIANTS:
            base = simulate(plant, kp, ki, var, kaw, t1, r1, r2)
            for c in (0.5, 2.0, 8.0):
                scaled = simulate(plant, kp, ki, var, kaw, t1, c * r1, c * r2, umax=c)
                assert (scaled.einschwing, scaled.ausstieg, scaled.n_sat, scaled.erste_einfahrt) == (base.einschwing, base.ausstieg, base.n_sat, base.erste_einfahrt), (var, c)
                assert scaled.unter == pytest.approx(c * base.unter, rel=1e-12, abs=1e-12)


def test_exit_grows_monotonically_and_linearly_with_the_length_of_the_saturation():
    """Ausstieg(T1) ist nicht fallend in T1, und die Steigung zwischen T1 = 120 und 240 liegt bei (r1 − G)/(G − r2) (hier Band ±2 %; gemessen größte Abweichung 0,8 % in 191 Zufallsfällen)."""
    rng = SplitMix64(99)
    n = 0
    for _ in range(40):
        plant, kp, ki, gain = _random_plant(rng)
        r1, r2 = gain * (1.5 + 4.5 * rng.random()), gain * (0.2 + 0.7 * rng.random())
        a60, a120, a240 = (simulate(plant, kp, ki, "keine", 0.0, t, r1, r2).ausstieg for t in (60, 120, 240))
        if None in (a60, a120, a240):
            continue
        n += 1
        assert a60 <= a120 <= a240
        assert (a240 - a120) / 120 == pytest.approx((r1 - gain) / (gain - r2), rel=0.02)
    assert n >= 30


def test_exit_slope_of_the_scenario_is_16_087():
    plant, kp, ki = C.case_plant("a1")
    a120, a240 = (simulate(plant, kp, ki, "keine", 0.0, t, C.SZ_R1, C.SZ_R2).ausstieg for t in (120, 240))
    assert (a240 - a120) / 120 == pytest.approx(3.7 / 0.23, rel=0.01)          # (4,7 − 1)/(1 − 0,77) = 16,087


def test_conditional_integration_and_integrator_limit_invariants():
    rng = SplitMix64(53)
    for _ in range(25):
        plant, kp, ki, gain = _random_plant(rng)
        t1, r1, r2 = 5 + rng.below(100), gain * (1.5 + 3 * rng.random()), gain * (0.2 + 0.7 * rng.random())
        _kz, b = simulate(plant, kp, ki, "ilim", 0.0, t1, r1, r2, horizon=400, still=None, record=True)
        assert max(abs(x) for x in b.I) <= 1.0                                      # die Integratorgrenze hält I in [−1, 1]
        _kz, b = simulate(plant, kp, ki, "bed", 0.0, t1, r1, r2, horizon=400, still=None, record=True)
        for k in range(len(b.I) - 1):
            if b.v[k] != b.u[k] and b.e[k] * (b.v[k] - b.u[k]) > 0:
                assert b.I[k + 1] == b.I[k]                                         # angehalten, solange gesättigt und der Fehler in Richtung der Sättigung zeigt


def test_back_calculation_never_lets_the_integrator_grow_faster_than_without():
    """In der Sättigung addiert die Rückrechnung kaw·(u − v) < 0 (v > u): der Integrator wächst langsamer als ohne Rückrechnung, Periode für Periode bei gleichem Fehler."""
    rng = SplitMix64(61)
    for _ in range(25):
        kaw = 0.05 + rng.random()
        I, e, ki = 5 * rng.random(), rng.random() * 3, 0.01 + 0.3 * rng.random()
        v = 1.0 + 4 * rng.random()
        assert D.integrator_update("bc", I, e, v, 1.0, ki, kaw) < D.integrator_update("keine", I, e, v, 1.0, ki, kaw)


def test_steady_state_is_r2_and_integrator_r2_over_gain_for_every_variant():
    """Im Beharrungszustand: y = r2 und I = r2/G (der Regler gibt u = r2/G aus, e = 0), für alle Varianten (Band 1e-6)."""
    rng = SplitMix64(71)
    for _ in range(20):
        plant, kp, ki, gain = _random_plant(rng)
        r2 = gain * (0.2 + 0.7 * rng.random())
        for var, kaw in VARIANTS:
            _kz, b = simulate(plant, kp, ki, var, kaw if var != "bc" else ki / kp, 20, 2 * gain, r2, horizon=4000, still=None, record=True)
            assert b.y[-1] == pytest.approx(r2, abs=1e-6), var
            assert b.I[-1] == pytest.approx(r2 / gain, abs=1e-6), var


def test_same_zufalls_seed_gives_the_same_noise_run_and_another_seed_another():
    plant = Plant("lag1", (0.9, 0), 1.0)
    a = D.noise_run(plant, 2.0, 0.2, "bc", 0.5, 0.9, D.uniform_noise(15, 4000, 0.4), n_warm=500)
    b = D.noise_run(plant, 2.0, 0.2, "bc", 0.5, 0.9, D.uniform_noise(15, 4000, 0.4), n_warm=500)
    c = D.noise_run(plant, 2.0, 0.2, "bc", 0.5, 0.9, D.uniform_noise(16, 4000, 0.4), n_warm=500)
    assert a == b and a["mean_y"] != c["mean_y"]


def test_without_anti_windup_the_mean_of_y_equals_the_setpoint_for_twenty_zufalls_seeds():
    """Mit Messrauschen und ohne Anti-Windup ist das Mittel von y gleich dem Sollwert (Band 0,01; exakte Identität: das Mittel des Fehlers verschwindet): 20 Zufalls-Seeds, Soll 0,9 nahe der Grenze."""
    plant = Plant("lag1", (0.9, 0), 1.0)
    means = [D.noise_run(plant, 2.0, 0.2, "keine", 0.0, 0.9, D.uniform_noise(seed, 8000, 0.4), n_warm=1000)["mean_y"] for seed in range(100, 120)]
    assert all(abs(m - 0.9) < 0.01 for m in means), (min(means), max(means))
    assert abs(float(np.mean(means)) - 0.9) < 0.003
