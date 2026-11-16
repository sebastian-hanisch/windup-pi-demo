"""Einheiten des Kerns an Mini-Instanzen von Hand (DEMO-PLAYBOOK §1b): jede Regel einzeln, jeder Erwartungswert als Bruch oder Rechnung.

Handbeispiel: Strecke y⁺ = (y + u)/2, PI mit kp = 1 und ki = 1/2, Grenze |u| ≤ 1, Sollwert 3 in den Perioden 0 und 1, danach 1/4. Die Zeilen (k, r, y, e, v, u, I) sind die
Werte am Anfang der Periode k (I vor dem Update). Von Hand: k = 0: e = 3, v = 3 + 0 > 1, also u = 1, y₁ = (0 + 1)/2 = 1/2, I₁ = 0 + 3/2. k = 1: e = 5/2, v = 5/2 + 3/2 = 4, u = 1,
y₂ = 3/4, I₂ = 3/2 + 5/4 = 11/4. k = 2: Sprung auf 1/4: e = 1/4 − 3/4 = −1/2, v = −1/2 + 11/4 = 9/4 > 1: noch gesättigt, obwohl der Fehler negativ ist; I₃ = 11/4 − 1/4 = 5/2."""

from fractions import Fraction as F

import numpy as np
import pytest

import wpi_dynamics as D
from wpi_dynamics import Plant, simulate


def _rows(bahn, n):
    return [(bahn.r[k], bahn.y[k], bahn.e[k], bahn.v[k], bahn.u[k], bahn.I[k]) for k in range(n)]


def _run(hand, variant, kaw=F(0), horizon=120, tol=F(1, 100)):
    return simulate(hand["plant"], hand["kp"], hand["ki"], variant, kaw, hand["t1"], hand["r1"], hand["r2"], horizon=horizon, still=None, record=True, tol=tol)


# --- Strecken: Stufenantwort von Hand ----------------------------------------------------------------------------------------------------------

def test_step_response_first_order_by_hand():
    # a = 1/2, G = 1, u = 1: y_k = 1 − (1/2)^k
    y = Plant("lag1", (F(1, 2), 0), F(1)).step_response(5)
    assert y == [F(0), F(1, 2), F(3, 4), F(7, 8), F(15, 16)]


def test_step_response_with_dead_time_by_hand():
    # Totzeit 2: u wirkt zwei Perioden später: y_0 = y_1 = y_2 = 0, y_3 = (1/2)·0 + (1/2)·1 = 1/2, y_4 = 1/4 + 1/2 = 3/4, y_5 = 3/8 + 1/2 = 7/8
    assert Plant("lag1", (F(1, 2), 2), F(1)).step_response(6) == [0, 0, 0, F(1, 2), F(3, 4), F(7, 8)]


def test_step_response_two_lags_by_hand():
    # zwei Verzögerungen 1/2, 1/2: x1: 0, 1/2, 3/4, 7/8; Ausgang x2: 0, 0, (1/2)·0 + (1/2)·(1/2) = 1/4, (1/2)(1/4) + (1/2)(3/4) = 1/2
    assert Plant("lag2", (F(1, 2), F(1, 2)), F(1)).step_response(4) == [0, 0, F(1, 4), F(1, 2)]


def test_step_response_oscillating_by_hand():
    # y⁺ = c1·y − c0·y₋ + G(1 − c1 + c0)·u mit c1 = 1, c0 = 1/2, G = 1: y⁺ = y − y₋/2 + u/2: 0, 1/2, 1/2 − 0 + 1/2 = 1, 1 − 1/4 + 1/2 = 5/4
    assert Plant("osc", (F(1), F(1, 2)), F(1)).step_response(4) == [0, F(1, 2), F(1), F(5, 4)]


def test_dc_gain_of_every_plant_kind_is_G():
    for plant in (Plant("lag1", (0.9, 0), 2.0), Plant("lag1", (0.8, 3), 2.0), Plant("lag2", (0.7, 0.6), 2.0), Plant("osc", (1.2, 0.5), 2.0)):
        assert plant.step_response(3000)[-1] == pytest.approx(2.0, rel=1e-9)


def test_unknown_plant_kind_is_rejected():
    with pytest.raises(ValueError):
        Plant("xyz", (1, 2)).state()


# --- Stellgrenze und Varianten einzeln ---------------------------------------------------------------------------------------------------------

def test_sat_clips_symmetrically():
    assert D.sat(F(9, 4)) == 1 and D.sat(F(-9, 4)) == -1 and D.sat(F(1, 2)) == F(1, 2) and D.sat(F(1)) == 1 and D.sat(F(-1)) == -1
    assert D.sat(5, umax=3) == 3 and D.sat(-5, umax=3) == -3


def test_integrator_update_each_variant_by_hand():
    # Zahlen: I = 2, e = 1/2, v = 3 (gesättigt, u = 1), ki = 1/4, kaw = 1/2
    I, e, v, u, ki, kaw = F(2), F(1, 2), F(3), F(1), F(1, 4), F(1, 2)
    assert D.integrator_update("keine", I, e, v, u, ki, kaw) == F(2) + F(1, 8)                 # I + ki·e
    assert D.integrator_update("bc", I, e, v, u, ki, kaw) == F(2) + F(1, 8) + F(1, 2) * (1 - 3)  # + kaw·(u − v) = −1
    assert D.integrator_update("bed", I, e, v, u, ki, kaw) == F(2)                                # v ≠ u und e·(v − u) = 1 > 0: angehalten
    assert D.integrator_update("ilim", I, e, v, u, ki, kaw) == F(1)                               # 2 + 1/8 auf 1 abgeschnitten


def test_conditional_integration_integrates_when_the_error_points_out_of_saturation():
    # gesättigt (v = 3, u = 1), aber e negativ: e·(v − u) = −1 < 0, also wird integriert (der Integrator fällt)
    assert D.integrator_update("bed", F(2), F(-1, 2), F(3), F(1), F(1, 4), F(0)) == F(2) - F(1, 8)
    # ungesättigt: immer integrieren
    assert D.integrator_update("bed", F(2), F(1, 2), F(1, 2), F(1, 2), F(1, 4), F(0)) == F(2) + F(1, 8)
    # untere Grenze: v = −3, u = −1, e negativ: e·(v − u) = (−1/2)(−2) = 1 > 0: angehalten
    assert D.integrator_update("bed", F(-2), F(-1, 2), F(-3), F(-1), F(1, 4), F(0)) == F(-2)


def test_integrator_limit_is_symmetric():
    assert D.integrator_update("ilim", F(-1), F(-4), F(0), F(0), F(1, 2), F(0)) == -1 and D.integrator_update("ilim", F(1, 2), F(1, 4), F(0), F(0), F(1, 2), F(0)) == F(5, 8)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        D.integrator_update("xyz", 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        simulate(Plant("lag1", (0.5, 0), 1.0), 1.0, 0.5, "xyz", 0.0, 2, 3.0, 0.25)


# --- Handbeispiel: ohne Anti-Windup -------------------------------------------------------------------------------------------------------------

def test_hand_example_without_anti_windup_rows_k0_to_k6(hand):
    kz, b = _run(hand, "keine")
    assert _rows(b, 7) == [
        (3, 0, 3, 3, 1, 0),
        (3, F(1, 2), F(5, 2), 4, 1, F(3, 2)),
        (F(1, 4), F(3, 4), F(-1, 2), F(9, 4), 1, F(11, 4)),
        (F(1, 4), F(7, 8), F(-5, 8), F(15, 8), 1, F(5, 2)),
        (F(1, 4), F(15, 16), F(-11, 16), F(3, 2), 1, F(35, 16)),
        (F(1, 4), F(31, 32), F(-23, 32), F(9, 8), 1, F(59, 32)),
        (F(1, 4), F(63, 64), F(-47, 64), F(3, 4), F(3, 4), F(95, 64)),
    ]


def test_hand_example_exit_is_period_6_four_periods_after_the_jump(hand):
    kz, b = _run(hand, "keine")
    assert kz.ausstieg == 4                                   # k = 6 minus T1 = 2; in k = 5 ist v = 9/8 noch über der Grenze (kein Gleichstand)
    assert b.v[5] == F(9, 8) and b.v[6] == F(3, 4)
    assert kz.n_sat == 4                                      # gesättigt in k = 2, 3, 4, 5
    assert kz.einschwing == 14                                # Band ±1/100, nach der Handrechnung der Planung
    assert kz.unter == 0 and kz.erste_einfahrt == 14          # y fällt monoton von 3/4 auf 1/4: kein Unterschwingen, die erste Einfahrt ist zugleich die endgültige


def test_hand_example_y_at_the_jump_is_the_row_of_period_T1(hand):
    kz, b = _run(hand, "keine")
    assert b.y[hand["t1"]] == F(3, 4)
    assert kz.y_sprung == pytest.approx(0.75)


# --- Handbeispiel: Rückrechnung, bedingte Integration, Integratorgrenze -------------------------------------------------------------------------

def test_hand_example_back_calculation_with_kaw_equal_to_ki_over_kp(hand):
    kz, b = _run(hand, "bc", F(1, 2))
    # I₁ = 1/2, I₂ = 3/4; ab k = 2 ist v = u = 1/4 = r2, y_k = 1/4 + (1/2)^(k−1), kein Unterschwingen
    assert [b.I[k] for k in range(3)] == [0, F(1, 2), F(3, 4)]
    for k in range(2, 60):
        assert b.v[k] == F(1, 4) and b.u[k] == F(1, 4)
        assert b.y[k] == F(1, 4) + F(1, 2) ** (k - 1)
    assert kz.unter == 0 and kz.einschwing == 6 and kz.ausstieg == 0


def test_hand_example_back_calculation_with_kaw_one_undershoots(hand):
    kz, b = _run(hand, "bc", F(1))
    assert [(b.y[k], b.v[k], b.u[k]) for k in range(6)] == [
        (0, 3, 1), (F(1, 2), 2, 1), (F(3, 4), F(-3, 4), F(-3, 4)), (0, F(-1, 4), F(-1, 4)), (F(-1, 8), 0, 0), (F(-1, 16), F(1, 8), F(1, 8))]
    assert min(b.y[hand["t1"]:]) == F(-1, 8)
    assert kz.unter == pytest.approx(0.25 + 0.125) and kz.einschwing == 10


def test_hand_example_conditional_integration(hand):
    kz, b = _run(hand, "bed")
    # I bleibt 0 in k = 0 und 1 (gesättigt, e > 0: angehalten); k = 2: v = −1/2 ungesättigt, I₃ = −1/4
    assert [(b.v[k], b.u[k], b.I[k]) for k in range(5)] == [
        (3, 1, 0), (F(5, 2), 1, 0), (F(-1, 2), F(-1, 2), 0), (F(-1, 8), F(-1, 8), F(-1, 4)), (F(1, 16), F(1, 16), F(-3, 16))]
    assert min(b.y[hand["t1"]:]) == 0 and kz.einschwing == 10


def test_hand_example_integrator_limit(hand):
    kz, b = _run(hand, "ilim")
    # I₂ = 1 statt 11/4; y fällt monoton von 3/4 auf 1/4
    assert [(b.v[k], b.u[k], b.I[k]) for k in range(5)] == [
        (3, 1, 0), (F(7, 2), 1, 1), (F(1, 2), F(1, 2), 1), (F(3, 8), F(3, 8), F(3, 4)), (F(5, 16), F(5, 16), F(9, 16))]
    assert min(b.y[hand["t1"]:]) >= hand["r2"] and kz.einschwing == 8


def test_hand_example_settling_times_in_both_bands(hand):
    """Band ±1/100 (Plan, Handrechnung): 14, 6, 10, 10, 8. Band 2,5 % von r2 = 1/160 (App): 14, 7, 11, 9 (kaw = 1 hier nicht gemessen)."""
    plan = [simulate(hand["plant"], hand["kp"], hand["ki"], v, k, 2, hand["r1"], hand["r2"], horizon=200, still=None, tol=F(1, 100)).einschwing
            for v, k in (("keine", 0), ("bc", F(1, 2)), ("bc", F(1)), ("bed", 0), ("ilim", 0))]
    assert plan == [14, 6, 10, 10, 8]
    app = [simulate(hand["plant"], hand["kp"], hand["ki"], v, k, 2, hand["r1"], hand["r2"], horizon=200, still=None, tol=F(1, 160)).einschwing
           for v, k in (("keine", 0), ("bc", F(1, 2)), ("bed", 0), ("ilim", 0))]
    assert app == [14, 7, 11, 9]


def test_hand_example_in_floating_point_gives_the_same_exit_and_settling_times():
    p = Plant("lag1", (0.5, 0), 1.0)
    assert simulate(p, 1.0, 0.5, "keine", 0.0, 2, 3.0, 0.25, tol=0.01).ausstieg == 4
    assert [simulate(p, 1.0, 0.5, v, k, 2, 3.0, 0.25, tol=0.01).einschwing for v, k in (("keine", 0), ("bc", 0.5), ("bc", 1.0), ("bed", 0), ("ilim", 0))] == [14, 6, 10, 10, 8]


def test_linear_loop_of_the_hand_example_has_the_double_root_one_half():
    # Zustand (y, I): y⁺ = (r + I)/2 bei kp = 1, a = 1/2; I⁺ = I + ki (r − y): z² − z + 1/4 = (z − 1/2)²
    assert D.rho_pi(Plant("lag1", (0.5, 0), 1.0), 1.0, 0.5) == pytest.approx(0.5, abs=1e-6)       # Doppelwurzel: Eigenwerte nur auf ~1e-8 genau
    assert D.char_poly_lag1(0.5, 1.0, 1.0, 0.5) == [1.0, -1.0, 0.25]


# --- Kennzahlen: Definitionen an einem Lauf, dessen Werte man abzählt -------------------------------------------------------------------------------

def test_kz_definitions_on_the_kaw_one_run(hand):
    """Der kaw = 1-Lauf (y: 3/4, 0, −1/8, −1/16, 1/32, …) liefert alle Felder von Hand: kleinstes y = −1/8, Unterschwingen r2 − (−1/8) = 3/8, y_sprung = 3/4."""
    kz = simulate(hand["plant"], hand["kp"], hand["ki"], "bc", F(1), 2, hand["r1"], hand["r2"], horizon=100, still=None, tol=F(1, 100))
    assert kz.unter == pytest.approx(3 / 8)
    assert kz.ueber == pytest.approx(0.5)                     # größtes y − r2 = 3/4 − 1/4 (die Periode des Sprungs zählt mit)
    assert kz.y_sprung == pytest.approx(0.75)
    assert kz.erste_einfahrt == 10 and kz.einschwing == 10    # y: 3/4, 0, −1/8, −1/16, 1/32 …: schwingt gedämpft aus und verlässt das Band ±1/100 ab Periode 10 nicht mehr
    assert kz.n == 100 and not kz.divergiert


def test_first_entry_versus_last_exit_for_back_calculation_with_kaw_half():
    """V6-Fall (a = 0,9; PI 2 / 0,2; 5 für 60 Perioden, dann 0,8), kaw = 0,5: y ist nach dem ersten Schritt schon im Band (erste Einfahrt 1: V6 meldete „0“), fällt danach bis 0,1321
    und ist erst nach 48 Perioden endgültig im Band (letzte Ausfahrt)."""
    kz = simulate(Plant("lag1", (0.9, 0), 1.0), 2.0, 0.2, "bc", 0.5, 60, 5.0, 0.8)
    assert kz.erste_einfahrt == 1 and kz.einschwing == 48
    assert kz.unter == pytest.approx(0.668, abs=5e-4)


def test_no_jump_when_the_run_ends_before_it():
    kz = simulate(Plant("lag1", (0.5, 0), 1.0), 1.0, 0.5, "keine", 0.0, 10, 3.0, 0.25, horizon=-5, still=None)
    assert kz.n == 0 and kz.einschwing == 0 and kz.ausstieg is None


def test_divergence_is_reported_not_nan():
    kz = simulate(Plant("lag1", (0.9, 0), 1.0), 2.0, 0.2, "bc", 2.5, 57, 4.7, 0.77)
    assert kz.divergiert and kz.einschwing is None and kz.ausstieg is None and kz.n == 0


def test_settling_time_never_reached_stays_at_the_horizon():
    # r2 ≥ G: y bleibt unter r2 (Sollwert unerreichbar), das Band wird nie erreicht; Einschwingzeit = Zahl der Perioden nach dem Sprung
    kz = simulate(Plant("lag1", (0.5, 0), 1.0), 1.0, 0.5, "keine", 0.0, 2, 3.0, 1.5, horizon=50, still=None)
    assert kz.einschwing == 50 and kz.ausstieg is None and kz.erste_einfahrt is None


def test_still_stops_the_run_after_500_quiet_periods(hand):
    kz = simulate(Plant("lag1", (0.5, 0), 1.0), 1.0, 0.5, "keine", 0.0, 2, 3.0, 0.25)
    assert kz.n == kz.einschwing + D.STILL                    # Abbruch genau nach STILL Perioden im Band mit |v| <= 1 (die erste dieser Perioden steht nach der Ausfahrt)


# --- Vorhersage des Ausstiegs --------------------------------------------------------------------------------------------------------------------

def test_predict_exit_hand_example(hand):
    """s_k = 1 − (1/2)^k, I_T1 = (1/2)·[(3 − 0) + (3 − 1/2)] = 11/4, dann fällt I um (1/2)(s_j − 1/4): erste Periode mit (1/4 − s_j) + I_j <= 1 ist j = 6."""
    n, abstand, status = D.predict_exit(hand["plant"], hand["kp"], hand["ki"], 2, hand["r1"], hand["r2"], max_periods=100)
    assert (n, status) == (4, "ok") and abstand == F(1, 8)         # v_6 = 3/4 (Abstand 1/4), v_5 = 9/8 (Abstand 1/8): der kleinere zählt


def test_predict_exit_assumption_is_checked(hand):
    """Ist der Sollwert nie unerreichbar (r1 = 1/8), gibt es keine Sättigung bis zum Sprung: Status „annahme“."""
    assert D.predict_exit(hand["plant"], hand["kp"], hand["ki"], 2, F(1, 8), hand["r2"], max_periods=100) == (None, None, "annahme")


def test_predict_exit_without_exit_in_the_horizon():
    # Sollwert r2 = 1,5 ≥ G: v bleibt über der Grenze
    n, _a, status = D.predict_exit(Plant("lag1", (0.5, 0), 1.0), 1.0, 0.5, 2, 3.0, 1.5, max_periods=40)
    assert n is None and status == "kein_ausstieg"


def test_exit_approximation_and_slope_by_hand(hand):
    # S = (3 − 0) + (3 − 1/2) = 11/2, Δ = G − r2 = 3/4: N ≈ S/Δ − (1 + kp·Δ)/(ki·Δ) = (11/2)/(3/4) − (7/4)/(3/8) = 22/3 − 14/3 = 8/3
    assert D.exit_approximation(hand["plant"], hand["kp"], hand["ki"], 2, hand["r1"], hand["r2"]) == pytest.approx(8 / 3)
    assert D.exit_slope(hand["plant"], hand["r1"], hand["r2"]) == pytest.approx((3 - 1) / (1 - 0.25))
    assert D.exit_approximation(hand["plant"], hand["kp"], hand["ki"], 2, hand["r1"], F(2)) is None          # r2 ≥ G: unerreichbar
    assert D.exit_slope(hand["plant"], hand["r1"], F(2)) is None


# --- Rückrechnung in der Sättigung ----------------------------------------------------------------------------------------------------------------

def test_saturation_recursion_by_hand():
    # kp = 1, ki = 1/2, kaw = 1/2: Faktor (1 − 1/2), Fehlerterm (1/2 − 1/2)·e = 0, + 1/2: I⁺ = (I + 1)/2: 0 → 1/2 → 3/4
    assert D.integrator_recursion(F(1), F(1, 2), F(1, 2), [F(3), F(5, 2)]) == F(3, 4)
    # kaw = 1, ki = 1/2, kp = 1: I⁺ = (1/2 − 1)e + 1 = 1 − e/2: e = 3 → −1/2 (Handbeispiel kaw = 1: I₁ = −1/2)
    assert D.integrator_recursion(F(1), F(1, 2), F(1), [F(3)]) == F(-1, 2)


def test_v_after_jump_formula_by_hand():
    # v = umax − kp·Δ + (ki/kaw − kp)(r1 − y_sat) mit kp = 2, ki = 1/5, kaw = 1/10, y_sat = 1, r1 = 5, r2 = 4/5: 1 − 2·(1/5) + (2 − 2)·4 = 3/5
    assert D.v_after_jump(2, F(1, 5), F(1, 10), 1, 5, F(4, 5)) == F(3, 5)
    # kaw = 1/2: ki/kaw = 2/5, (2/5 − 2) = −8/5, mal (r1 − y_sat) = 4 gibt −32/5; dazu 1 − kp·Δ = 3/5: v = −29/5
    assert D.v_after_jump(2, F(1, 5), F(1, 2), 1, 5, F(4, 5)) == F(-29, 5)


# --- Messrauschen -----------------------------------------------------------------------------------------------------------------------------------

def test_noise_run_identity_on_a_tiny_run():
    """Σe = (kaw/ki)·Σ(v − u) + (I_Ende − I_Start)/ki gilt für die Rückrechnung in jedem Lauf (Summe der Updates); hier ein kurzes Feld, das Rauschen ist von Hand gesetzt."""
    noise = [0.5, -0.5, 0.25, 0.0, 0.5, 0.5, -0.25, 0.25]
    res = D.noise_run(Plant("lag1", (0.5, 0), 1.0), 1.0, 0.5, "bc", 0.5, 0.9, noise, n_warm=2)
    assert res["identity_left"] == pytest.approx(res["identity_right"], rel=1e-12, abs=1e-12)
    assert res["n"] == 6


def test_uniform_noise_stays_in_the_amplitude_band_and_is_reproducible():
    x = D.uniform_noise(15, 1000, 0.3)
    assert x.min() >= -0.3 and x.max() < 0.3 and abs(x.mean()) < 0.03
    assert np.array_equal(x, D.uniform_noise(15, 1000, 0.3)) and not np.array_equal(x, D.uniform_noise(16, 1000, 0.3))
