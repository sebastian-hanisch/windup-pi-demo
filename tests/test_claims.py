"""Claims-Tests: jede Zahl der Befunde B1 bis B10 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung überall: dieses Modell (PI, symmetrische Grenze |u| ≤ 1, Zeitlage wie in wpi_dynamics.py), Szenario r₁ = 4,7 für 57 Perioden, dann r₂ = 0,77, außer wo anders genannt.
Die Regelschleife rechnet nur mit +, −, ×, ÷ und Vergleichen (IEEE 754) und gibt auf jeder Plattform dieselben Bits; trotzdem stehen Zählungen aus Neu-Rechnung als Band
(Zählungen ± 2, Floats relativ 1e-9 oder auf 1e-5), und ganze Zahlen, die an keiner Schwelle knapp liegen, exakt. Die vorgerechneten Reihen (`data/wpi_frozen.json`) werden hier
neu erzeugt (24 Fälle ganz, die ersten 300 Zufallsfälle; mit WPI_FULL=1 alle 3 000, rund 20 s) und auf Konsistenz mit der Datei und der README geprüft."""

import os
import re
import statistics
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import wpi_constants as C
import wpi_dynamics as D
import wpi_evaluation as ev
from helpers import random_plant
from wpi_dynamics import Plant, simulate
from wpi_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
NB = " "
FROZEN = ev.frozen()
SZ = (C.SZ_T1, C.SZ_R1, C.SZ_R2)


def _readme_has(*snippets):
    flat = README.replace(NB, " ").replace("\\|", "|")
    for s in snippets:
        assert s in flat, s


def _run(cid, variant, kaw=None, **kw):
    plant, kp, ki = C.case_plant(cid)
    k = ki / kp if kaw == "ratio" else (0.0 if kaw is None else kaw)
    return simulate(plant, kp, ki, variant, k, *SZ, **kw)


def _a1():
    return C.case_plant("a1")


# --- B1: V6 nachgebaut ----------------------------------------------------------------------------------------------------------------------------

def test_b1_v6_tie_exit_1215_in_fractions_and_1216_in_floating_point():
    kz = simulate(Plant("lag1", (F(9, 10), 0), F(1)), F(2), F(1, 5), "keine", F(0), 60, F(5), F(4, 5), horizon=1300, still=None, tol=F(2, 100))
    assert kz.ausstieg == 1215
    plant = Plant("lag1", (0.9, 0), 1.0)
    kzf = simulate(plant, 2.0, 0.2, "keine", 0.0, 60, 5.0, 0.8)
    assert kzf.ausstieg in (1215, 1216)                     # Band: die Rundung entscheidet
    assert 1242 <= kzf.einschwing <= 1244
    # Rest des Gleichstands in Brüchen: v − 1 fällt um genau 0,04 je Periode und trifft in Periode 1215 die Null (bis auf eine Potenz von 9/10 unter 1e-50)
    s = Plant("lag1", (F(9, 10), 0), F(1)).step_response(1300)
    I = F(1, 5) * sum(F(5) - s[k] for k in range(60))
    j = 60
    while F(2) * (F(4, 5) - s[j]) + I > 1:
        I -= F(1, 5) * (s[j] - F(4, 5))
        j += 1
    v = F(2) * (F(4, 5) - s[j]) + I
    assert j - 60 == 1215 and abs(v - 1) < F(1, 10 ** 50)
    _readme_has("1 215 Perioden", "1 216", "1 244", "1 243", "unter 10⁻⁵⁰", "Ausstieg 1 215 bis 1 216, Einschwingzeit 1 242 bis 1 244")


def test_b1_v6_old_measure_first_entry_after_the_step_was_1243():
    """Die Vormessung speicherte y nach dem Schritt und zählte die erste Einfahrt (|y − 0,8| < 0,02): ohne Anti-Windup 1 243, mit Rückrechnung kaw = 0,5 sofort (0)."""
    plant = Plant("lag1", (0.9, 0), 1.0)
    for variant, kaw, expected in (("keine", 0.0, 1243), ("bc", 0.5, 0)):
        _kz, b = simulate(plant, 2.0, 0.2, variant, kaw, 60, 5.0, 0.8, horizon=1400, still=None, record=True)
        after = b.y[61:]                                    # y nach dem Schritt der Periode 60, 61, …
        first = next(j for j, y in enumerate(after) if abs(y - 0.8) < 0.02)
        assert first == expected, variant


def test_b1_v6_back_calculation_passes_through_the_band_and_undershoots():
    plant = Plant("lag1", (0.9, 0), 1.0)
    kz, b = simulate(plant, 2.0, 0.2, "bc", 0.5, 60, 5.0, 0.8, horizon=300, still=None, record=True)
    assert b.y[61] == pytest.approx(0.798, abs=5e-4)        # nach einem Schritt im Band
    assert min(b.y[60:]) == pytest.approx(0.132, abs=5e-4) and kz.unter == pytest.approx(0.668, abs=5e-4)
    assert kz.einschwing == 48 and kz.erste_einfahrt == 1
    bed = simulate(plant, 2.0, 0.2, "bed", 0.0, 60, 5.0, 0.8)
    ilim = simulate(plant, 2.0, 0.2, "ilim", 0.0, 60, 5.0, 0.8)
    assert (bed.einschwing, round(bed.unter, 3)) == (37, 0.229) and ilim.einschwing == 11 and ilim.unter == 0
    _readme_has("(0,798;", "bis 0,132 (Unterschwingen 0,668)", "nach 48 Perioden", "bedingte Integration 37 (Unterschwingen 0,229), Integratorgrenze 11")


# --- B2: 24 Fälle ---------------------------------------------------------------------------------------------------------------------------------

def _same(a, b):
    """Vorgerechnet gegen neu gerechnet: ganze Zahlen ± 2 (Band), Floats auf 1e-5."""
    for k, v in a.items():
        w = b[k]
        if isinstance(v, bool) or v is None or isinstance(v, str):
            assert v == w, (k, v, w)
        elif isinstance(v, int):
            assert abs(v - w) <= 2, (k, v, w)
        elif isinstance(v, float):
            assert v == pytest.approx(w, abs=1e-5), (k, v, w)
        elif isinstance(v, dict):
            _same(v, w)


def test_b2_frozen_24_cases_are_reproduced_by_the_script():
    _same(FROZEN["faelle"], ev.compute_cases())


def test_b2_counts_over_the_24_cases():
    cnt = ev.case_counts(FROZEN["faelle"])
    k, kp, hi, bed, il = cnt["keine"], cnt["bckp"], cnt["bc05"], cnt["bed"], cnt["ilim"]
    assert (k["median"], k["min"], k["max"]) == (951, 162, 1065) and k["unter_gross"] == 1 and round(k["unter_max"], 2) == 0.06
    assert (kp["median"], kp["min"], kp["max"], kp["kuerzer"], kp["unter_gross"], round(kp["unter_max"], 2)) == (22.5, 4, 51, 24, 6, 0.13)
    assert (hi["median"], hi["min"], hi["max"], hi["kuerzer"], hi["unter_gross"], round(hi["unter_max"], 2)) == (47, 4, 109, 24, 23, 1.71)
    assert (bed["median"], bed["min"], bed["max"], bed["kuerzer"], bed["unter_gross"], round(bed["unter_max"], 2)) == (40, 9, 98, 24, 24, 0.90)
    assert (il["median"], il["min"], il["max"], il["kuerzer"], il["unter_gross"], round(il["unter_max"], 2)) == (23, 4, 51, 24, 5, 0.13)
    _readme_has("im Median 951 (Spanne 162 bis 1 065", "Median 22,5 (4 bis 51), kürzer als „Keine“ in 24 von 24, Unterschwingen über 0,05 in 6 von 24 (größtes 0,13;",
                "Median 47 (4 bis 109)", "in 23 von 24 (größtes 1,71)", "Median 40 (9 bis 98)", "in 24 von 24 (größtes 0,90)", "Median 23 (4 bis 51)", "in 5 von 24 (größtes 0,13)")


def test_b2_in_18_of_24_the_ratio_variant_undershoots_at_most_0_05_and_keine_in_the_g2_case_only():
    over = [cid for cid in C.CASE_IDS if FROZEN["faelle"][cid]["bckp"]["unter"] > 0.05]
    assert len(over) == 6 and 24 - len(over) == 18
    assert [cid for cid in C.CASE_IDS if FROZEN["faelle"][cid]["keine"]["unter"] > 0.05] == ["a8"]
    _readme_has("in 18 von 24 höchstens 0,05", "„Keine“ unterschwingt in 1 von 24 über 0,05 (0,06, im Fall mit G = 2)")


def test_b2_keine_per_plant_kind_and_the_g2_exception():
    z = {c: FROZEN["faelle"][c]["keine"]["einschwing"] for c in C.CASE_IDS}
    a = [z[f"a{i}"] for i in range(1, 9)]
    b = [z[f"b{i}"] for i in range(1, 9)]
    c = [z[f"c{i}"] for i in range(1, 9)]
    assert (min(a[:7]), max(a[:7]), a[7]) == (917, 1065, 162)
    assert (min(b), max(b)) == (916, 1015) and (min(c), max(c)) == (921, 1038)
    g1 = [v for cid, v in z.items() if cid != "a8"]
    assert (min(g1), max(g1)) == (916, 1065)
    rho = [FROZEN["faelle"][cid]["rho"] for cid in C.CASE_IDS]
    assert min(rho) == pytest.approx(0.50, abs=0.005) and max(rho) == pytest.approx(0.96, abs=0.005)
    _readme_has("916 bis 1 065 Perioden zum Einschwingen (in 23 der 24 Fälle mit Verstärkung 1", "erste Ordnung 917 bis 1 065 (G = 2: 162), zwei Verzögerungen und schwach gedämpft 916 bis 1 015, Totzeit 921 bis 1 038",
                "ρ 0,50 bis 0,96")


# --- B3: Vorhersage ----------------------------------------------------------------------------------------------------------------------------------

def test_b3_prediction_on_24_and_30_cases():
    for cid in C.CASE_IDS:
        plant, kp, ki = C.case_plant(cid)
        vor, abstand, status = D.predict_exit(plant, kp, ki, *SZ)
        assert status == "ok" and vor == simulate(plant, kp, ki, "keine", 0.0, *SZ).ausstieg and abstand >= 0.99e-3
    assert min(FROZEN["faelle"][c]["vorhersage"]["abstand"] for c in C.CASE_IDS) == pytest.approx(1.0e-3, abs=2e-5)
    n = 0
    for cid in ("a1", "c1", "b1"):
        plant, kp, ki = C.case_plant(cid)
        for t1 in C.T1_SWEEP:
            n += D.predict_exit(plant, kp, ki, t1, C.SZ_R1, C.SZ_R2)[0] == simulate(plant, kp, ki, "keine", 0.0, t1, C.SZ_R1, C.SZ_R2).ausstieg
    assert n == 30
    _readme_has("in 24 von 24 Fällen (kleinster Abstand von v zur Grenze 1,0·10⁻³), in 30 von 30 weiteren")


def test_b3_prediction_on_the_random_cases_is_in_the_frozen_data():
    v = FROZEN["zufall"]["vorhersage"]
    assert v["annahme_erfuellt"] == 2727 and v["gleich"] == 2727 and FROZEN["zufall"]["n"] - v["annahme_erfuellt"] == 273
    assert v["kleinster_abstand"] == pytest.approx(1.2e-5, abs=2e-7)
    _readme_has("in 2 727 von 2 727 Zufallsfällen", "(273 Fälle erfüllten sie nicht; kleinster Abstand 1,2·10⁻⁵)")


def test_b3_approximation_deviation_in_the_24_cases():
    dev = {}
    for cid in C.CASE_IDS:
        plant, kp, ki = C.case_plant(cid)
        meas = simulate(plant, kp, ki, "keine", 0.0, *SZ).ausstieg
        dev[cid] = abs(meas - D.exit_approximation(plant, kp, ki, *SZ)) / meas
    assert statistics.median(dev.values()) == pytest.approx(0.001, abs=0.0006) and max(dev.values()) == pytest.approx(0.026, abs=0.0006)
    assert {c for c, d in dev.items() if d > 0.02} == {"a6", "a7"}                   # a = 0,97: y erreicht bei T1 = 57 die Grenze noch nicht
    _readme_has("im Median 0,1 %, höchstens 2,6 % (a = 0,97,")


# --- B4: Dauer gegen T1 ---------------------------------------------------------------------------------------------------------------------------

def test_b4_slope_and_values_for_three_cases():
    expected = {"a1": (28, 977, 3873, 12), "c1": (41, 991, 3886, 16), "b1": (50, 999, 3895, 25)}
    for cid, (t1_1, t1_60, t1_240, bc60) in expected.items():
        plant, kp, ki = C.case_plant(cid)
        aus = {t: simulate(plant, kp, ki, "keine", 0.0, t, C.SZ_R1, C.SZ_R2).ausstieg for t in (1, 20, 60, 120, 240)}
        assert (aus[1], aus[60], aus[240]) == (t1_1, t1_60, t1_240)
        assert 16.07 <= (aus[240] - aus[120]) / 120 <= 16.10 and 16.07 <= (aus[60] - aus[20]) / 40 <= 16.10
        assert simulate(plant, kp, ki, "bc", ki / kp, 60, C.SZ_R1, C.SZ_R2).einschwing == bc60
        assert simulate(plant, kp, ki, "bc", ki / kp, 240, C.SZ_R1, C.SZ_R2).einschwing == bc60          # ab T1 = 60 konstant
    bed = {}
    for cid in expected:
        plant, kp, ki = C.case_plant(cid)
        bed[cid] = simulate(plant, kp, ki, "bed", 0.0, 60, C.SZ_R1, C.SZ_R2).einschwing
    assert bed == {"a1": 38, "c1": 38, "b1": 31}
    _readme_has("Steigung 16,08 bis 16,10", "(r₁ − G)/(G − r₂) = 3,7/0,23 = 16,087", "28 / 41 / 50 Perioden", "977 / 991 / 999", "3 873 / 3 886 / 3 895", "(12 / 16 / 25)", "(38 / 38 / 31)")


def test_b4_200_random_cases_monotone_and_slope_within_one_percent():
    rng = SplitMix64(99)
    valid, mono, worst = 0, 0, 0.0
    for _ in range(200):
        plant, kp, ki, gain = random_plant(rng)
        r1, r2 = gain * (1.5 + 4.5 * rng.random()), gain * (0.2 + 0.7 * rng.random())
        a60, a120, a240 = (simulate(plant, kp, ki, "keine", 0.0, t, r1, r2).ausstieg for t in (60, 120, 240))
        if None in (a60, a120, a240):
            continue
        valid += 1
        mono += a60 <= a120 <= a240
        worst = max(worst, abs((a240 - a120) / 120 / ((r1 - gain) / (gain - r2)) - 1))
    assert 189 <= valid <= 193 and mono == valid and worst < 0.01
    _readme_has("Zufalls-Seed 99", "191 davon mit Ausstieg bei T₁ = 60, 120 und 240", "größte Abweichung 0,8 %")


# --- B5: kaw -------------------------------------------------------------------------------------------------------------------------------------

KAWS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 1.9)


def test_b5_kaw_grid_and_the_ratio_rule():
    grid = {cid: [_run(cid, "bc", k).einschwing for k in KAWS] for cid in C.CASE_IDS}
    ratio = {cid: _run(cid, "bc", "ratio").einschwing for cid in C.CASE_IDS}
    k001 = [grid[c][1] for c in C.CASE_IDS if c != "a8"]
    assert (min(k001), max(k001)) == (212, 258) and grid["a8"][1] == 95
    k005 = [grid[c][3] for c in C.CASE_IDS if c not in ("a8", "b8", "c7", "c8")]
    assert (min(k005), max(k005)) == (59, 103) and (grid["a8"][3], grid["b8"][3], grid["c7"][3], grid["c8"][3]) == (47, 31, 32, 51)
    within = [c for c in C.CASE_IDS if ratio[c] <= min(grid[c]) * 1.5 + 2]
    not_larger = [c for c in C.CASE_IDS if ratio[c] <= min(grid[c])]
    assert len(within) == 23 and set(C.CASE_IDS) - set(within) == {"c2"} and len(not_larger) == 20
    better = {c: (KAWS[int(np.argmin(grid[c]))], min(grid[c]), ratio[c]) for c in C.CASE_IDS if c not in not_larger}
    assert better == {"a8": (1.0, 23, 25), "b4": (0.2, 18, 21), "c2": (1.9, 14, 26), "c8": (1.5, 48, 51)}
    assert (min(ratio.values()), max(ratio.values())) == (4, 51)
    assert (grid["a1"][1], ratio["a1"], grid["a1"][6]) == (229, 12, 48) and round(_run("a1", "bc", 0.5).unter, 2) == 0.63
    _readme_has("kaw = 0,01 gibt eine Einschwingzeit von 212 bis 258 (Fall a8: 95)", "kaw = 0,05 von 59 bis 103 (a8: 47; die drei Fälle mit ki/kp = 0,05: 31, 32, 51)",
                "in 23 von 24 Fällen höchstens beim 1,5-Fachen des besten Gitterwerts plus 2 (Ausnahme c2: bester Wert 14 bei kaw = 1,9, bei ki/kp = 0,08: 26)",
                "in 20 von 24 nicht größer als der beste Gitterwert", "a8 (kaw = 1: 23 gegen 25), b4 (0,2: 18 gegen 21), c2 (1,9: 14 gegen 26), c8 (1,5: 48 gegen 51)",
                "kaw = 0,01: 229, kaw = ki/kp = 0,1: 12, kaw = 0,5: 48 mit Unterschwingen 0,63")


# --- B6: Gleichgewicht und obere Grenze ----------------------------------------------------------------------------------------------------------

def test_b6_jump_formula_against_the_simulation_for_case_a1():
    plant, kp, ki = _a1()
    sc = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=kp, ki=ki, t1=57, r1=4.7, r2=0.77)
    expected = {0.05: (7.596, 7.950), 0.1: (0.5425, 0.5449), 0.2: (-3.160, -3.158), 0.5: (-5.380, -5.379), 1.0: (-6.120, -6.1195)}
    for kaw, (gem, formel) in expected.items():
        jc = ev.jump_check(sc, kaw)
        assert jc["y_sat"] == pytest.approx(0.9975, abs=5e-5)
        assert jc["v_gemessen"] == pytest.approx(gem, abs=6e-4) and jc["v_formel"] == pytest.approx(formel, abs=6e-4), kaw
        if kaw > ki / kp:
            assert jc["v_gemessen"] < -1                    # bei kaw > ki/kp weit unter −1
    assert ev.jump_check(sc, 0.0) is None
    _readme_has("kaw = 0,1 (= ki/kp): 0,5425 gegen 0,5449; 0,2: −3,160 gegen −3,158; 0,5: −5,380 gegen −5,379; 1,0: −6,120 gegen −6,1195; bei kaw = 0,05: 7,596 gegen 7,950")


def test_b6_recursion_and_upper_limit_of_kaw():
    plant, kp, ki = _a1()
    _kz, b = simulate(plant, kp, ki, "bc", 0.5, 57, C.SZ_R1, C.SZ_R2, horizon=2, still=None, record=True)
    assert abs(b.I[57]) == pytest.approx(4.925, abs=1e-3)
    for kaw, div in ((1.9, False), (2.0, False), (2.1, False), (2.5, True), (5.0, True)):
        assert simulate(plant, kp, ki, "bc", kaw, *SZ).divergiert is div, kaw
    # Faktor |1 − kaw| in der Sättigung: unter 1 genau für 0 < kaw < 2
    assert [abs(1 - k) < 1 for k in (0.01, 1.0, 1.99, 2.0, 2.5)] == [True, True, True, False, False]
    _readme_has("(|I| nach 57 Perioden 4,925 bei kaw = 0,5)", "kaw = 2,5 und 5 divergieren", "bei 2,0 und 2,1 bleibt der Lauf in diesem Szenario beschränkt")


def test_b6_kaw_2_5_diverges_only_with_the_long_saturation():
    """Korrektur gegenüber dem Plan: im Handbeispiel (T1 = 2) ist die Einschwingzeit bei kaw = 2,5 elf Perioden (Experiment 4, Preset „Von Hand“)."""
    sc = ev.settings_from_values(C.PRESETS[C.HAND]).scenario
    rows = ev.kaw_curve(sc, 2.5)
    r25 = next(r for r in rows if abs(r["kaw"] - 2.5) < 1e-9)
    assert not r25["divergiert"] and r25["einschwing"] == 11
    _readme_has("bei kaw = 2,5 elf Perioden")


# --- B7: Rauschen --------------------------------------------------------------------------------------------------------------------------------

NOISE_EXPECTED = {   # Mittel von y bei Rauschen ±0,2 / ±0,4 / ±0,8
    "Keine": (0.8998, 0.8997, 0.8993),
    "Rückrechnung, kaw = 0,05": (0.8792, 0.8385, 0.7518),
    "Rückrechnung, kaw = 0,50": (0.8022, 0.6488, 0.3491),
    "Rückrechnung, kaw = 1,00": (0.7637, 0.5604, 0.2163),
    "Bedingte Integration": (0.8286, 0.7289, 0.5281),
    "Integratorgrenze": (0.8884, 0.8372, 0.7264),
}


def test_b7_noise_means_identity_and_saturation_share():
    sc = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=0.77)
    sat_share = []
    for i, amp in enumerate(C.NOISE_AMPLITUDES):
        rows = {r["name"]: r for r in ev.noise_table(sc, 0.9, amp, 15)}
        for name, exp in NOISE_EXPECTED.items():
            assert rows[name]["mean_y"] == pytest.approx(exp[i], abs=0.002), (name, amp)
        sat_share.append(rows["Keine"]["sat_share"])
        assert abs(rows["Keine"]["mean_e"]) < 2e-4
        for r in rows.values():
            if r["variant"] == "bc":
                assert r["identity_left"] == pytest.approx(r["identity_right"], rel=1e-9)
                # mit Rückrechnung gleich kaw/ki mal dem mittleren Abschneiden (Summen desselben Laufs)
                assert r["mean_e"] == pytest.approx(r["kaw"] / 0.2 * r["sum_vu"] / r["n"] + (r["I_end"] - r["I_start"]) / 0.2 / r["n"], rel=1e-9)
        assert rows["Rückrechnung, kaw = 0,50"]["mean_y"] < rows["Rückrechnung, kaw = 0,05"]["mean_y"] < rows["Keine"]["mean_y"]
    assert [round(100 * s) for s in sat_share] == [50, 65, 76]
    _readme_has("Keine 0,8998 / 0,8997 / 0,8993", "kaw = 0,05: 0,8792 / 0,8385 / 0,7518", "kaw = 0,5: 0,8022 / 0,6488 / 0,3491", "kaw = 1: 0,7637 / 0,5604 / 0,2163",
                "bedingte Integration 0,8286 / 0,7289 / 0,5281", "Integratorgrenze 0,8884 / 0,8372 / 0,7264", "Band im Test ±0,002", "bei 50 / 65 / 76 %",
                "Σe = (kaw/ki)·Σ(v − u) + (I_Ende − I_Start)/ki", "0,0000 bis 0,0001", "Zufalls-Seed 15")


def test_b7_other_zufalls_seeds_move_the_numbers_only_in_the_band_of_the_seed_scatter():
    """Gegenprobe zur Seed-Abhängigkeit: Mittel von y ohne Anti-Windup bleibt bei jedem von 10 Zufalls-Seeds im Band 0,9 ± 0,01, mit kaw = 0,5 liegt es bei jedem unter 0,85."""
    plant = Plant("lag1", (0.9, 0), 1.0)
    for seed in range(200, 210):
        noise = D.uniform_noise(seed, 12000, 0.2)
        keine = D.noise_run(plant, 2.0, 0.2, "keine", 0.0, 0.9, noise, 2000)["mean_y"]
        bc = D.noise_run(plant, 2.0, 0.2, "bc", 0.5, 0.9, noise, 2000)["mean_y"]
        assert abs(keine - 0.9) < 0.01 and bc < 0.85, seed


# --- B8: 3 000 Zufallsfälle -----------------------------------------------------------------------------------------------------------------------

def test_b8_frozen_random_counts_and_medians():
    z = FROZEN["zufall"]
    assert (z["seed"], z["n"], z["versuche"]) == (1506, 3000, 5347)
    v = z["vergleich"]
    assert (v["bckp"]["kuerzer"], v["bckp"]["gleich"], v["bckp"]["laenger"], v["bckp"]["groesste_verlaengerung"]) == (2983, 4, 13, 5934)
    assert (v["bed"]["kuerzer"], v["bed"]["gleich"], v["bed"]["laenger"]) == (2978, 3, 19)
    assert (v["ilim"]["kuerzer"], v["ilim"]["gleich"], v["ilim"]["laenger"]) == (2980, 12, 8)
    m = z["median"]
    assert (m["lag1"]["n"], m["lag1"]["keine"], m["lag1"]["bckp"], m["lag1"]["bed"], m["lag1"]["ilim"]) == (856, 461, 30, 40, 31)
    assert (m["tot"]["n"], m["tot"]["keine"], m["tot"]["bckp"], m["tot"]["bed"], m["tot"]["ilim"]) == (445, 447, 48, 63, 50)
    assert (m["lag2"]["n"], m["lag2"]["keine"], m["lag2"]["bckp"], m["lag2"]["bed"], m["lag2"]["ilim"]) == (846, 465.5, 35, 48.5, 37)
    assert (m["osc"]["n"], m["osc"]["keine"], m["osc"]["bckp"], m["osc"]["bed"], m["osc"]["ilim"]) == (853, 408, 34, 45, 37)
    assert z["windup"] == {"mit": 2973, "ohne": 26} and round(100 * v["bckp"]["kuerzer"] / 3000, 1) == 99.4
    _readme_has("kürzer als „Keine“ in 2 983 Fällen, gleich in 4, länger in 13 (größte Verlängerung 5 934 Perioden)", "bedingte Integration 2 978 / 3 / 19; Integratorgrenze 2 980 / 12 / 8",
                "erste Ordnung 461 / 30 / 40 / 31, Totzeit 447 / 48 / 63 / 50, zwei Verzögerungen 465,5 / 35 / 48,5 / 37, schwach gedämpft 408 / 34 / 45 / 37",
                "in 2 973 von 3 000 Fällen gesättigt (26 Fälle ohne Windup)", "99,4 %", "5 347 Versuche")


def test_b8_exceptions_by_plant_kind_and_short_saturation():
    ex = [e for e in FROZEN["zufall"]["ausnahmen"] if "bckp" in e["laenger"]]
    assert len(ex) == 13
    kinds = [e["art"] for e in ex]
    assert kinds.count("osc") == 10 and sorted(k for k in kinds if k != "osc") == ["lag1", "lag2", "tot"]
    assert sum(1 for e in ex if e["t1"] <= 10) == 6
    assert max(e["einschwing"]["bckp"] - e["einschwing"]["keine"] for e in ex) == 5934
    _readme_has("betreffen 10 die schwach gedämpfte Strecke (je einmal erste Ordnung, zwei Verzögerungen, Totzeit), und 6 haben T₁ ≤ 10")


def test_b8_seven_runs_do_not_settle_and_repeat_exactly_every_3_to_6_periods():
    z = FROZEN["zufall"]
    cases = {e["index"]: e for e in z["ausnahmen"]}
    assert len(z["nicht_eingeschwungen"]) == 7 and z["nicht_eingeschwungen_arten"] == {"osc": 7}
    periods, shares = [], []
    for i in z["nicht_eingeschwungen"]:
        e = cases[i]
        assert e["einschwing"]["keine"] < D.HORIZON                       # „Keine“ schwingt in allen sieben ein
        info = ev.limit_cycle_info(e, "bckp")
        assert info["period"] is not None
        periods.append(info["period"])
        shares.append(info["sat_share"])
    assert 3 <= min(periods) and max(periods) <= 6 and sorted(set(periods)) == [3, 4, 5, 6]
    assert round(100 * min(shares)) == 67 and max(shares) == 1.0
    _readme_has("**7 Fälle** (alle schwach gedämpft)", "alle 3 bis 6 Perioden (Grenzzyklus), 67 % bis 100 % der Perioden sind gesättigt", "„Keine“ schwingt in allen sieben ein")


def test_b8_recomputing_the_first_300_random_cases_equals_the_frozen_data():
    n = 3000 if os.environ.get("WPI_FULL") else 300
    cases, attempts = ev.random_cases(n=n)
    kompakt = FROZEN["zufall"]["kompakt"][:n]
    diff = 0
    for c, row in zip(cases, kompakt):
        mine = [c["kz"][k].einschwing for k in ("keine", "bckp", "bed", "ilim")] + [c["kz"]["keine"].ausstieg if c["kz"]["keine"].ausstieg is not None else -1]
        diff += mine != row
    assert diff <= 2                                                       # Band: höchstens 2 abweichende Fälle (Eigenwert-Schwelle ρ < 0,97 an der Plattformgrenze)
    if n == 3000:
        s = ev.summarize_random(cases, attempts)
        assert s["vergleich"] == FROZEN["zufall"]["vergleich"] and s["nicht_eingeschwungen"] == FROZEN["zufall"]["nicht_eingeschwungen"]


# --- B9: Eigenschaften --------------------------------------------------------------------------------------------------------------------------

def test_b9_without_saturation_all_variants_equal_in_the_24_cases_and_3000_random_ones():
    equal = 0
    for cid in C.CASE_IDS:
        plant, kp, ki = C.case_plant(cid)
        runs = [simulate(plant, kp, ki, v, k, 60, 0.1, 0.05, horizon=240, still=None, record=True)[1] for v, k in (("keine", 0.0), ("bc", 0.5), ("bed", 0.0), ("ilim", 0.0))]
        assert max(abs(x) for x in runs[0].v) <= 1
        equal += all(b.y == runs[0].y and b.I == runs[0].I for b in runs[1:])
    assert equal == 24
    rng = SplitMix64(4242)
    unsat = ok = 0
    for _ in range(3000):
        plant, kp, ki, gain = random_plant(rng)
        r1, r2 = 0.1 * gain * rng.random(), 0.1 * gain * rng.random()
        runs = [simulate(plant, kp, ki, v, k, 60, r1, r2, horizon=120, still=None, record=True)[1] for v, k in (("keine", 0.0), ("bc", 0.5), ("bed", 0.0), ("ilim", 0.0))]
        if max(abs(x) for x in runs[0].v) <= 1:
            unsat += 1
            ok += all(b.y == runs[0].y for b in runs[1:])
    assert ok == unsat and 2986 <= unsat <= 2992 and 3000 - unsat in range(8, 15)
    _readme_has("in 24 von 24 Fällen und in allen ungesättigten Fällen von 3 000 Zufallsinstanzen mit kleinem Sollwert (2 989 von 3 000 blieben ungesättigt; in den übrigen 11 überstieg kp·e die Grenze)")


def test_b9_kaw_zero_is_keine_and_all_96_runs_settle():
    settle = 0
    for cid in C.CASE_IDS:
        assert _run(cid, "bc", 0.0) == _run(cid, "keine")
        for variant, kaw in (("keine", None), ("bc", "ratio"), ("bed", None), ("ilim", None)):
            settle += _run(cid, variant, kaw).einschwing < D.HORIZON - 10
    assert settle == 96
    _readme_has("Rückrechnung mit kaw = 0 ist „Keine“ in 24 von 24 (alle Kennzahlen); alle 96 Läufe (24 Fälle × 4 Varianten) schwingen im Horizont ein")


# --- B10: Handbeispiel --------------------------------------------------------------------------------------------------------------------------

def test_b10_hand_example_numbers_in_the_readme():
    plant = Plant("lag1", (F(1, 2), 0), F(1))
    _kz, b = simulate(plant, F(1), F(1, 2), "keine", F(0), 2, F(3), F(1, 4), horizon=20, still=None, record=True, tol=F(1, 100))
    assert b.v[2] == F(9, 4) and b.v[5] == F(9, 8) and b.v[6] == F(3, 4) and all(b.u[k] == 1 for k in range(6))
    assert all(b.e[k] < 0 for k in range(2, 7))
    _kz, bc = simulate(plant, F(1), F(1, 2), "bc", F(1, 2), 2, F(3), F(1, 4), horizon=20, still=None, record=True)
    assert all(bc.u[k] == F(1, 4) == bc.v[k] and bc.y[k] == F(1, 4) + F(1, 2) ** (k - 1) for k in range(2, 20))
    _kz, b1 = simulate(plant, F(1), F(1, 2), "bc", F(1), 2, F(3), F(1, 4), horizon=60, still=None, record=True)
    assert min(b1.y[2:]) == F(-1, 8) and F(1, 4) - min(b1.y[2:]) == F(3, 8)
    plan = [simulate(plant, F(1), F(1, 2), v, k, 2, F(3), F(1, 4), horizon=200, still=None, tol=F(1, 100)).einschwing for v, k in (("keine", 0), ("bc", F(1, 2)), ("bc", F(1)), ("bed", 0), ("ilim", 0))]
    app = [simulate(plant, F(1), F(1, 2), v, k, 2, F(3), F(1, 4), horizon=200, still=None).einschwing for v, k in (("keine", 0), ("bc", F(1, 2)), ("bed", 0), ("ilim", 0))]
    assert plan == [14, 6, 10, 10, 8] and app == [14, 7, 11, 9]
    assert D.rho_pi(Plant("lag1", (0.5, 0), 1.0), 1.0, 0.5) == pytest.approx(0.5, abs=1e-6)
    _readme_has("(v₂ = 9/4, v₅ = 9/8); erstmals |v| ≤ 1 in k = 6 (v = ¾), vier Perioden nach dem Sprung", "y_k = ¼ + (½)^(k−1)", "bis −1/8 (Unterschwingen 3/8)",
                "14 / 6 / 10 / 10 / 8", "14 / 7 / 11 / 9", "Doppelwurzel ½")


def test_b10_app_hand_example_in_fractions_matches_the_kernel():
    rows = ev.hand_example("keine", 9)
    assert [(r["k"], r["v"], r["u"], r["I"]) for r in rows[:7]] == [(0, 3, 1, 0), (1, 4, 1, F(3, 2)), (2, F(9, 4), 1, F(11, 4)), (3, F(15, 8), 1, F(5, 2)), (4, F(3, 2), 1, F(35, 16)),
                                                                         (5, F(9, 8), 1, F(59, 32)), (6, F(3, 4), F(3, 4), F(95, 64))]
    assert next(r["k"] for r in rows if r["k"] >= 2 and abs(r["v"]) <= 1) == 6


# --- Korrekturen gegenüber dem Plan --------------------------------------------------------------------------------------------------------------

def test_correction_rausch_preset_with_t1_zero_would_have_a_tie_at_the_exit():
    """Der Plan sah Sollwert 0,9 bei T1 = 0 vor; dabei liegt v beim Ausstieg bis auf 9·10⁻¹⁶ auf der Grenze (Gleichstand): deshalb ein eigener Sollwert-Regler im Rauschexperiment."""
    plant, kp, ki = _a1()
    _vor, abstand, status = D.predict_exit(plant, kp, ki, 0, 0.9, 0.9)
    assert status == "ok" and abstand < 1e-12 and abstand == pytest.approx(9e-16, abs=5e-16)
    _readme_has("Abstand von v zur Grenze 9·10⁻¹⁶")
    assert all(p["t1"] > 0 for p in C.PRESETS.values())


def test_readme_quotes_the_sources_without_numbers_from_the_chapter_and_without_literal_text():
    flat = README.replace(NB, " ")
    assert "Keine Zahl aus dem Kapitel ist übernommen" in flat and "in eigenen Worten" in flat
    assert re.search(r"Abschnitt 11\.4 der 2\. Auflage bzw\. 10\.4 der 1\. Auflage", flat)
