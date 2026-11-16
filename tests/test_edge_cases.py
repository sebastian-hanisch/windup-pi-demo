"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): T₁ = 0 und 1, r₂ ≥ G, kaw an der Stabilitätsgrenze, kp = 0, Totzeit 0 und 8, G = 2, Divergenz ohne NaN."""

import math

import pytest

import wpi_constants as C
import wpi_dynamics as D
import wpi_evaluation as ev
from wpi_dynamics import Plant, simulate


def test_t1_zero_has_no_jump_the_start_up_is_the_scenario():
    """T₁ = 0: es gilt von Anfang an r₂; y = 0 beim „Sprung“ (y_sprung = 0 ≤ r₂), also kein Unterschwingen im Sinne der Messung."""
    plant = Plant("lag1", (0.9, 0), 1.0)
    kz = simulate(plant, 2.0, 0.2, "keine", 0.0, 0, 0.0, 0.9)
    assert kz.y_sprung == 0.0 and not ev.undershoot_applies(kz, 0.9) and kz.n > 0
    assert kz.ausstieg == 41                                              # Anlauf: kp·e = 1,8 > 1 sättigt, der Integrator zieht auf
    kz_small = simulate(plant, 2.0, 0.2, "keine", 0.0, 0, 0.0, 0.2)       # kp·e = 0,4 < 1: nie gesättigt
    assert kz_small.ausstieg == 0 and kz_small.n_sat == 0


def test_t1_one_is_a_single_saturated_period():
    plant = Plant("lag1", (0.9, 0), 1.0)
    kz = simulate(plant, 2.0, 0.2, "keine", 0.0, 1, 4.7, 0.77)
    assert kz.ausstieg == 28 and D.predict_exit(plant, 2.0, 0.2, 1, 4.7, 0.77)[0] == 28      # Fall a1 bei T₁ = 1


def test_second_setpoint_at_or_above_g_never_exits():
    plant = Plant("lag1", (0.9, 0), 1.0)
    for r2 in (1.0, 1.2):
        kz = simulate(plant, 2.0, 0.2, "keine", 0.0, 57, 4.7, r2, horizon=300, still=None)
        assert kz.ausstieg is None
        assert D.predict_exit(plant, 2.0, 0.2, 57, 4.7, r2, max_periods=300)[2] == "kein_ausstieg"
    # r2 = 1,2: y bleibt unter 1 und erreicht das Band nie; bei r2 = G = 1 liegt y = 0,9975 schon im Band (±2,5 %), v aber nie unter der Grenze: kein Ausstieg
    assert simulate(plant, 2.0, 0.2, "keine", 0.0, 57, 4.7, 1.2, horizon=300, still=None).erste_einfahrt is None
    assert simulate(plant, 2.0, 0.2, "keine", 0.0, 57, 4.7, 1.0, horizon=300, still=None).erste_einfahrt == 0
    sc = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=1.0)
    assert not sc.reachable and ev.exit_sweep(sc, (1, 2))[0]["ausstieg"] is None
    assert ev.Scenario(r2=0.99).reachable and not ev.Scenario(r2=1.0).reachable
    assert D.exit_slope(plant, 4.7, 1.0) is None and D.exit_approximation(plant, 2.0, 0.2, 57, 4.7, 1.0) is None


def test_kaw_at_the_edge_of_stability():
    """In der Sättigung ist |1 − kaw| < 1 nur für 0 < kaw < 2: 1,9 und 2,0 laufen im Szenario a1 beschränkt, 2,5 und 5 divergieren (Anzeige „divergiert“, nie NaN)."""
    plant = Plant("lag1", (0.9, 0), 1.0)
    for kaw in (1.9, 2.0, 2.1):
        kz = simulate(plant, 2.0, 0.2, "bc", kaw, 57, 4.7, 0.77)
        assert not kz.divergiert and kz.einschwing is not None and math.isfinite(kz.unter)
    for kaw in (2.5, 5.0):
        kz = simulate(plant, 2.0, 0.2, "bc", kaw, 57, 4.7, 0.77)
        assert kz.divergiert and kz.einschwing is None and math.isnan(kz.unter) is True      # Kz.unter ist nan; die App zeigt bei „divergiert“ nur den Text, nie die Zahl
    rows = ev.kaw_curve(ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=0.77), 2.5)
    assert [r["kaw"] for r in rows if r["divergiert"]] == [2.5]


def test_kaw_zero_in_the_curve_and_ratio_row_is_marked():
    sc = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=0.77)
    rows = ev.kaw_curve(sc, 0.0)
    assert rows[0]["kaw"] == 0.0 and rows[0]["einschwing"] == 959                          # kaw = 0 ist „Keine“
    assert [r["kaw"] for r in rows if r["is_ratio"]] == [0.1]
    assert [r["kaw"] for r in rows] == sorted(r["kaw"] for r in rows)


def test_pure_integral_controller_kp_zero_runs_and_stays_finite():
    """kp = 0 ist kein Regler der Demo (Regler ab 0,1), die Kernfunktion rechnet ihn trotzdem: nicht gemessen, als Randfall ohne Fehler."""
    plant = Plant("lag1", (0.5, 0), 1.0)
    for variant in D.VARIANTEN:
        kz = simulate(plant, 0.0, 0.2, variant, 0.2, 5, 3.0, 0.5)
        assert kz.einschwing is not None or kz.divergiert
        assert kz.divergiert or math.isfinite(kz.unter)


def test_dead_time_zero_equals_first_order_and_eight_runs():
    assert ev.Scenario(kind="tot", a1=0.9, dead=0).plant() == Plant("lag1", (0.9, 0), 1.0)
    base = simulate(Plant("lag1", (0.9, 0), 1.0), 2.0, 0.2, "keine", 0.0, 57, 4.7, 0.77)
    assert simulate(ev.Scenario(kind="tot", a1=0.9, dead=0).plant(), 2.0, 0.2, "keine", 0.0, 57, 4.7, 0.77) == base
    kz8 = simulate(Plant("lag1", (0.95, 8), 1.0), 1.0, 0.05, "keine", 0.0, 57, 4.7, 0.77)
    assert kz8.einschwing == 1002 and kz8.ausstieg is not None                 # Fall c7


def test_gain_two_case_has_a_short_windup_and_a_small_undershoot_without_anti_windup():
    plant, kp, ki = C.case_plant("a8")
    kz = simulate(plant, kp, ki, "keine", 0.0, 57, 4.7, 0.77)
    assert kz.einschwing == 162 and kz.unter == pytest.approx(0.06, abs=5e-3)


def test_unknown_scenario_kind_is_rejected_and_unreachable_checks_use_the_gain():
    with pytest.raises(ValueError):
        ev.Scenario(kind="xyz").plant()
    assert ev.Scenario(gain=2.0, r2=1.5).reachable and not ev.Scenario(gain=0.5, r2=0.6).reachable


def test_exit_sweep_adds_the_current_t1_and_skips_zero():
    sc = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=0.77)
    assert [r["t1"] for r in ev.exit_sweep(sc)] == sorted(set(C.T1_SWEEP) | {57})
    assert 0 not in [r["t1"] for r in ev.exit_sweep(ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=0, r1=4.7, r2=0.77))]


def test_noise_with_zero_amplitude_is_the_deterministic_steady_state():
    rows = {r["name"]: r for r in ev.noise_table(ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2), 0.9, 0.0, 15)}
    for name, r in rows.items():
        assert r["mean_y"] == pytest.approx(0.9, abs=1e-6), name
        assert r["sat_share"] == 0.0 or name != "Keine"


def test_noise_run_reports_divergence_instead_of_nan_for_an_unstable_loop():
    plant = Plant("lag1", (0.9, 0), 1.0)
    noise = D.uniform_noise(1, 5000, 0.1)
    ok = D.noise_run(plant, 2.0, 0.2, "bc", 0.5, 0.9, noise, 500)
    # unerreichbarer Sollwert 4,7 (immer gesättigt) und kaw = 2,5: |1 − kaw| = 1,5 > 1, der Integrator wächst geometrisch bis zur Divergenzgrenze
    bad = D.noise_run(plant, 2.0, 0.2, "bc", 2.5, 4.7, noise, 500)
    assert not ok["divergiert"] and bad["divergiert"] and math.isnan(bad["mean_y"]) and bad["n"] == 0


def test_table_rows_and_plot_length_bounds():
    a = ev.analyse(ev.Settings(ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=0.77), "keine"))
    assert len(ev.table_rows(a.bahn_selected, 9)) == 9 and ev.table_rows(a.bahn_selected, 9)[0]["k"] == 0
    assert 57 + 30 <= a.n_plot <= 57 + 4500
    tiny = ev.analyse(ev.Settings())                                          # Handbeispiel: kurz
    assert tiny.n_plot == 2 + 30                                              # Einschwingzeit 14: 1,2 mal 14 plus 10 = 26 < 30, also mindestens 30 Perioden nach dem Sprung
    assert len(ev.table_rows(tiny.bahn_selected, 9)) == 9
