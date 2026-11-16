"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, je Preset die erwartete Verdikt-Zeile, kein Gleichstand beim Ausstieg.

Zahlen der Verdikte: Messung dieses Baus (Python-Schleife ohne Bibliotheksfunktionen, auf jeder Plattform dieselben Bits); Bänder nur bei den Mittelwerten des Rauschens."""

import pytest

import wpi_constants as C
import wpi_dynamics as D
import wpi_evaluation as ev
import wpi_presets as P

NB = " "


def _analysis(name):
    return ev.analyse(ev.settings_from_values(C.PRESETS[name]))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_names_are_the_six_of_the_plan_in_order():
    assert list(C.PRESETS) == ["Von Hand (a = ½, T₁ = 2)", "Lange Sättigung (Strecke 0,9, T₁ = 57)", "Totzeit 3", "Schwingende Strecke", "Rauschen nahe der Grenze", "Zu großes kaw"]


def test_preset_values_are_inside_the_regler_bounds_and_the_choices_are_valid():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.settings_from_values(C.PRESETS[C.DEFAULT_PRESET]) == ev.Settings()
    assert ev.Settings().scenario == ev.Scenario()
    for state_key, spec in P.SETTING_SPECS.items():
        key = next(k for k, v in P.PRESET_KEYS.items() if v == state_key)
        assert spec.default == C.PRESETS[C.DEFAULT_PRESET][key]


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_preset_has_a_round_v6_scenario_or_v_exactly_on_the_limit():
    """Kein Preset nutzt die runden Zahlen (5, 60, 0,8) und keines hat v beim Ausstieg im Abstand unter 1e-3 von der Grenze (Gleichstand, den die Rundung entscheidet)."""
    for name, p in C.PRESETS.items():
        assert (p["r1"], p["t1"], p["r2"]) != (C.B1_R1, C.B1_T1, C.B1_R2), name
        sc = ev.settings_from_values(p).scenario
        vor, abstand, status = D.predict_exit(sc.plant(), sc.kp, sc.ki, sc.t1, sc.r1, sc.r2)
        assert status == "ok" and abstand > 1e-3, (name, abstand)


def test_the_measured_scenario_presets_use_4_7_57_and_0_77():
    for name in (C.LANG, C.TOT, C.OSC, C.RAUSCH, C.GROSS):
        assert (C.PRESETS[name]["t1"], C.PRESETS[name]["r1"], C.PRESETS[name]["r2"]) == (57, 4.7, 0.77)


# --- je Preset die erwartete Verdikt-Zeile ---------------------------------------------------------------------------------------------------------------

def test_preset_hand_verdict_four_periods_at_the_limit_then_seven_with_back_calculation():
    a = _analysis(C.HAND)
    kind, text = ev.verdict_windup(a)
    assert kind == "warning" and "noch **4 Perioden** an der Grenze (Einschwingzeit 14 Perioden)" in text and "kaw = ki/kp = 0,50: 7 Perioden." in text
    assert a.selected.key == "keine" and a.rho == pytest.approx(0.5, abs=1e-6)


def test_preset_lang_verdict_929_periods():
    a = _analysis(C.LANG)
    _kind, text = ev.verdict_windup(a)
    assert "noch **929 Perioden** an der Grenze (Einschwingzeit 959 Perioden)" in text and "kaw = ki/kp = 0,10: 12 Perioden." in text
    assert C.PRESET_HELP[C.LANG].endswith("noch 929 Perioden an der Grenze.")


def test_preset_tot_and_osc_verdicts():
    _k, text = ev.verdict_windup(_analysis(C.TOT))
    assert "**942 Perioden**" in text and "Einschwingzeit 971 Perioden" in text and "kaw = ki/kp = 0,10: 16 Perioden." in text
    a = _analysis(C.OSC)
    _k, text = ev.verdict_windup(a)
    assert "**889 Perioden**" in text and "Einschwingzeit 924 Perioden" in text and "kaw = ki/kp = 0,10: 30 Perioden." in text
    assert a.row("bc_kp").kz.unter == pytest.approx(0.073, abs=5e-4)
    assert 0.8 < a.rho < 1.0


def test_preset_gross_back_calculation_with_kaw_half_undershoots_by_0_63():
    a = _analysis(C.GROSS)
    assert a.selected.key == "bc" and a.selected.kaw == 0.5
    assert a.selected.kz.einschwing == 48 and a.selected.kz.unter == pytest.approx(0.632, abs=5e-4)
    _k, text = ev.verdict_cost(a, None)
    assert "Rückrechnung, kaw = 0,50: 0,63" in text and "Rückrechnung, kaw = ki/kp: 0,00" in text
    assert "fünfmal ki/kp" in C.PRESET_HELP[C.GROSS] and C.PRESETS[C.GROSS]["kaw"] == 5 * C.PRESETS[C.GROSS]["ki"] / C.PRESETS[C.GROSS]["kp"]


def test_preset_rausch_noise_experiment_shows_the_shifted_mean():
    s = ev.settings_from_values(C.PRESETS[C.RAUSCH])
    assert (s.noise_amp, s.noise_r, s.noise_seed) == (0.4, 0.9, 15)
    rows = ev.noise_table(s.scenario, s.noise_r, s.noise_amp, s.noise_seed)
    kind, text = ev.verdict_noise(rows, s.noise_r, s.noise_amp)
    assert kind == "warning" and "Mittel von y 0,900 (Soll 0,90)" in text and "kaw = 0,5 0,649" in text
    assert rows[0]["mean_y"] == pytest.approx(0.8997, abs=0.002) and next(r for r in rows if r["kaw"] == 0.5)["mean_y"] == pytest.approx(0.6488, abs=0.002)


def test_preset_without_effect_reports_no_effect():
    """Ein Szenario ohne Sättigung (Sollwert 0,1 und 0,05 im Handbeispiel) darf kein Windup melden."""
    sc = ev.Scenario(kind="lag1", a1=0.5, gain=1.0, kp=1.0, ki=0.5, t1=2, r1=0.1, r2=0.05)
    a = ev.analyse(ev.Settings(sc))
    kind, text = ev.verdict_windup(a)
    assert kind == "info" and "kein Windup" in text
    assert all(r.kz.n_sat == 0 for r in a.rows)
    assert len({(r.kz.einschwing, r.kz.ausstieg) for r in a.rows}) == 1


def test_the_noise_experiment_at_a_reachable_setpoint_far_from_the_limit_reports_no_shift():
    rows = ev.noise_table(ev.Scenario(), 0.25, 0.2, 15)
    kind, text = ev.verdict_noise(rows, 0.25, 0.2)
    assert kind == "info" and "sättigt der Regler kaum" in text


# --- Aktives Preset ---------------------------------------------------------------------------------------------------------------------------------

def test_equal_helper_compares_floats_with_tolerance_and_other_types_exactly():
    assert P._equal(0.9, 0.9 + 1e-12) and not P._equal(0.9, 0.91) and P._equal("lag1", "lag1") and not P._equal("lag1", "osc") and P._equal(3, 3.0)
