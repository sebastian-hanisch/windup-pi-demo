"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt fertige Kennzahlen, keine Zufalls-Seeds: der Test setzt sie von Hand und prüft Grenzen und jede genannte Zahl."""

import wpi_dynamics as D
import wpi_evaluation as ev

NB = " "


def _kz(einschwing, ausstieg, unter=0.0, y_sprung=1.0, divergiert=False):
    return D.Kz(einschwing, ausstieg, 0.0, unter, None, 0, 100, divergiert, y_sprung)


def _analysis(sc, keine, bckp, bc=None, bed=None, ilim=None, selected="keine"):
    """Analyse aus fünf von Hand gesetzten Kz (Reihenfolge keine, bc_kp, bc, bed, ilim); die Bahn wird von den Verdikten nicht gelesen."""
    ratio = sc.ki / sc.kp
    kzs = [keine, bckp, bc or bckp, bed or bckp, ilim or bckp]
    rows = tuple(ev.Row(k, ev.VARIANT_CHOICE_NAME[k] if k != "bc" else "Rückrechnung, kaw = 0,50", "bc" if k in ("bc_kp", "bc") else k,
                        ratio if k == "bc_kp" else (0.5 if k == "bc" else 0.0), kz) for k, kz in zip(ev.VARIANT_CHOICES, kzs))
    empty = D.Bahn([], [], [], [], [], [])
    return ev.Analysis(ev.Settings(sc, selected), 0.9, rows, next(r for r in rows if r.key == selected), empty, empty, 100, (None, None, "ok"), None, None)


SC = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=0.77)


# --- Kernfrage 1 ---------------------------------------------------------------------------------------------------------------------------------

def test_windup_verdict_names_exit_settling_time_and_the_back_calculation_time():
    kind, text = ev.verdict_windup(_analysis(SC, _kz(959, 929), _kz(12, 0)))
    assert kind == "warning"
    assert "noch **929 Perioden** an der Grenze (Einschwingzeit 959 Perioden)" in text and "Mit Rückrechnung und kaw = ki/kp = 0,10: 12 Perioden." in text


def test_windup_verdict_thousands_separator_and_singular():
    _kind, text = ev.verdict_windup(_analysis(SC, _kz(1244, 1216), _kz(1, 0)))
    assert f"**1{NB}216 Perioden**" in text and f"Einschwingzeit 1{NB}244 Perioden" in text and ": 1 Periode." in text


def test_windup_verdict_without_windup_is_info_and_says_so():
    kind, text = ev.verdict_windup(_analysis(SC, _kz(30, 0), _kz(30, 0)))
    assert kind == "info" and "kein Windup" in text and "Einschwingzeit 30 Perioden" in text and "in 57 Perioden Sättigung" in text


def test_windup_verdict_when_the_second_setpoint_is_unreachable():
    sc = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=1.2)
    kind, text = ev.verdict_windup(_analysis(sc, _kz(None, None), _kz(None, None)))
    assert kind == "warning" and "r₂ = 1,20" in text and "G·umax = 1,00" in text and "keinen Ausstieg" in text
    # genau r2 = G ist unerreichbar (nicht unter G)
    sc2 = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=1.0)
    assert "keinen Ausstieg" in ev.verdict_windup(_analysis(sc2, _kz(None, None), _kz(None, None)))[1]


def test_windup_verdict_divergence_and_no_exit_in_the_horizon():
    kind, text = ev.verdict_windup(_analysis(SC, _kz(None, None, divergiert=True), _kz(5, 0)))
    assert kind == "warning" and "divergiert" in text
    kind, text = ev.verdict_windup(_analysis(SC, _kz(6000, None), _kz(5, 0)))
    assert kind == "warning" and f"6{NB}000 Perioden" in text and "nicht" in text
    kind, text = ev.verdict_windup(_analysis(SC, _kz(959, 929), _kz(None, None, divergiert=True)))
    assert "Mit Rückrechnung (kaw = ki/kp) divergiert der Lauf." in text


def test_windup_verdict_at_t1_zero_speaks_of_the_start_up():
    sc = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=0, r1=0.9, r2=0.9)
    _kind, text = ev.verdict_windup(_analysis(sc, _kz(60, 41), _kz(22, 16)))
    assert "im Anlauf (T₁ = 0" in text and "**41 Perioden**" in text and "nach dem Sprung" not in text


# --- Kernfrage 2 ---------------------------------------------------------------------------------------------------------------------------------

COUNTS = {"bc05": dict(unter_gross=23, unter_max=1.707872), "bed": dict(unter_gross=24, unter_max=0.90375), "bckp": dict(unter_gross=6, unter_max=0.131934)}


def test_cost_verdict_lists_the_undershoot_per_variant_and_the_frozen_counts():
    a = _analysis(SC, _kz(959, 929), _kz(12, 0, 0.0), bc=_kz(48, 3, 0.632), bed=_kz(38, 0, 0.224), ilim=_kz(12, 0, 0.0))
    kind, text = ev.verdict_cost(a, COUNTS)
    assert kind == "info"
    assert "Rückrechnung, kaw = ki/kp: 0,00" in text and "Rückrechnung, kaw = 0,50: 0,63" in text and "Bedingte Integration: 0,22" in text and "Integratorgrenze: 0,00" in text
    assert "in 23 von 24 um mehr als 0,05 (größtes 1,71)" in text and "bedingte Integration in 24 von 24" in text and "in 6 von 24 (größtes 0,13)" in text


def test_cost_verdict_says_when_y_does_not_come_from_above():
    a = _analysis(SC, _kz(60, 41, 0.9, y_sprung=0.0), _kz(22, 16, 0.9, y_sprung=0.0))
    _kind, text = ev.verdict_cost(a, None)
    assert "kommt y beim Sprung nicht von oben" in text and "0,90" not in text


def test_cost_verdict_divergence_is_named():
    a = _analysis(SC, _kz(959, 929), _kz(12, 0), bc=_kz(None, None, divergiert=True))
    assert "Rückrechnung, kaw = 0,50: divergiert" in ev.verdict_cost(a, None)[1]


# --- Kernfrage 3 ---------------------------------------------------------------------------------------------------------------------------------

ZUFALL = {"n": 3000, "vergleich": {"bckp": {"kuerzer": 2983, "gleich": 4, "laenger": 13}}, "nicht_eingeschwungen": [1, 2, 3, 4, 5, 6, 7]}


def test_notwhen_verdict_without_exception_is_info_and_cites_the_random_counts():
    kind, text = ev.verdict_notwhen(_analysis(SC, _kz(959, 929), _kz(12, 0)), ZUFALL)
    assert kind == "info" and "höchstens so lang wie „Keine“" in text
    assert f"In 3{NB}000 Zufallsfällen" in text and f"in 2{NB}983 kürzer" in text and "in 4 gleich und in 13 länger" in text and f"7 davon schwingen in 6{NB}000 Perioden nicht ein" in text


def test_notwhen_verdict_names_a_longer_counter_measure_with_both_times():
    a = _analysis(SC, _kz(47, 1), _kz(55, 0), bc=_kz(40, 0), bed=_kz(40, 0), ilim=_kz(47, 0))
    kind, text = ev.verdict_notwhen(a, None)
    assert kind == "warning" and "länger als gar keins: Rückrechnung, kaw = ki/kp (55 Perioden gegen 47 Perioden). " in text         # nur diese Variante: 40 und 47 sind nicht länger als 47


def test_notwhen_verdict_divergence_counts_as_worse():
    a = _analysis(SC, _kz(47, 1), _kz(40, 0), bc=_kz(None, None, divergiert=True))
    kind, text = ev.verdict_notwhen(a, None)
    assert kind == "warning" and "Rückrechnung, kaw = 0,50 (divergiert)" in text


# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------------

def _row(t1, aus, bc=12, bed=38):
    return dict(t1=t1, ausstieg=aus, einschwing=None if aus is None else aus + 30, vorhergesagt=aus, status="ok", abstand=0.01, naeherung=None, bc=bc, bed=bed)


def test_exit_verdict_names_slope_expectation_and_the_last_row():
    rows = [_row(60, 977, bc=12), _row(120, 1943), _row(240, 3873, bc=12)]
    kind, text = ev.verdict_exit(rows, SC, 16.087)
    assert kind == "info"
    assert "um **16,1 Perioden** je gesättigter Periode" in text and "erwartet (r₁ − G)/(G − r₂) = 16,1" in text and f"Bei T₁ = 240 sind es 3{NB}873 Perioden." in text
    assert "bei T₁ = 60 12, bei T₁ = 240 12 Perioden" in text


def test_exit_verdict_branches_unreachable_reachable_first_and_missing_slope():
    sc_unreach = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=1.5)
    assert ev.verdict_exit([], sc_unreach, None)[0] == "warning" and "keinen Ausstieg" in ev.verdict_exit([], sc_unreach, None)[1]
    sc_easy = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=0.9, r2=0.5)
    kind, text = ev.verdict_exit([_row(60, 5)], sc_easy, None)
    assert kind == "info" and "r₁ = 0,90" in text and "nicht über G·umax = 1,00" in text and "−0,2" in text           # (0,9 − 1)/(1 − 0,5) = −0,2
    assert ev.verdict_exit([_row(60, 977)], SC, 16.087)[1].startswith("Für die Steigung fehlt")


def test_measured_slope_is_none_without_both_measurements():
    assert ev.measured_slope([_row(60, 977), _row(120, 1943)]) is None
    assert ev.measured_slope([_row(120, 1943), _row(240, 3873)]) == (3873 - 1943) / 120


# --- Experiment 3 bis 6 ----------------------------------------------------------------------------------------------------------------------------

def test_cases_verdict_uses_the_counts_over_the_24_cases():
    counts = ev.case_counts(ev.frozen()["faelle"])
    kind, text = ev.verdict_cases(counts)
    assert kind == "info"
    assert f"„Keine“ braucht 162 bis 1{NB}065 Perioden (Median 951)" in text and "Rückrechnung mit kaw = ki/kp 4 bis 51 (Median 22,5)" in text
    assert "kürzer als „Keine“ in 24 von 24 und mit Unterschwingen über 0,05 in 6 von 24" in text
    assert "bedingte Integration unterschwingt in 24 von 24 um mehr als 0,05 (größtes 0,90)" in text and "kaw = 0,5 in 23 von 24 (größtes 1,71)" in text
    assert "Integratorgrenze braucht im Median 23 Perioden bei 5 von 24" in text


def _kaw_row(kaw, einschwing, unter=0.0, div=False, y_sprung=1.0):
    return dict(kaw=kaw, einschwing=einschwing, unter=unter, divergiert=div, ausstieg=0, is_ratio=False, y_sprung=y_sprung)


def test_kaw_verdict_names_ratio_chosen_value_and_both_ends():
    rows = [_kaw_row(0.01, 229), _kaw_row(0.1, 12), _kaw_row(0.5, 48, 0.632), _kaw_row(2.5, None, div=True)]
    kind, text = ev.verdict_kaw(rows, 0.5, 0.1)
    assert kind == "info"
    assert text == "In diesem Szenario: kaw = ki/kp = 0,10: 12 Perioden; kaw = 0,50: 48 Perioden (Unterschwingen 0,63); kaw = 0,01: 229 Perioden; kaw = 2,50: divergiert."


def test_kaw_verdict_does_not_repeat_the_value_when_it_equals_the_ratio_or_a_grid_end():
    rows = [_kaw_row(0.01, 229), _kaw_row(0.1, 12), _kaw_row(2.5, 11)]
    text = ev.verdict_kaw(rows, 0.1, 0.1)[1]
    assert text.count("kaw = ki/kp = 0,10") == 1 and text.count("kaw = 0,10") == 0
    text = ev.verdict_kaw(rows, 0.01, 0.1)[1]
    assert text.count("kaw = 0,01") == 1


def _noise_rows(mean_keine, mean_bc, sat=0.5, div=False):
    base = dict(sat_share=sat, mean_e=0.0, divergiert=False)
    return [dict(base, name="Keine", variant="keine", kaw=0.0, mean_y=mean_keine, **{}),
            dict(base, name="Rückrechnung, kaw = 0,50", variant="bc", kaw=0.5, mean_y=mean_bc, divergiert=div)]


def test_noise_verdict_shifted_mean_is_a_warning_with_both_means_and_the_saturation_share():
    kind, text = ev.verdict_noise(_noise_rows(0.8998, 0.8022, 0.501), 0.9, 0.2)
    assert kind == "warning"
    assert "Mittel von y 0,900 (Soll 0,90)" in text and "kaw = 0,5 0,802" in text and "Anteil gesättigter Perioden ohne Anti-Windup: 50 %" in text and "Identität" in text


def test_noise_verdict_hardly_saturated_is_info_and_claims_no_effect():
    kind, text = ev.verdict_noise(_noise_rows(0.25, 0.25, 0.004), 0.25, 0.2)
    assert kind == "info" and "kaum (0,4 % der Perioden)" in text and "verschiebt auch das Gegenmittel nichts" in text
    # genau 1 % Sättigung ist schon „greift“
    assert ev.verdict_noise(_noise_rows(0.25, 0.24, 0.01), 0.25, 0.2)[0] == "warning"


def test_noise_verdict_divergence():
    kind, text = ev.verdict_noise(_noise_rows(0.9, 0.5, div=True), 0.9, 0.2)
    assert kind == "warning" and "divergiert" in text


def test_random_verdict_counts_and_the_plant_kind_of_the_unsettled_cases():
    z = dict(ZUFALL, nicht_eingeschwungen_arten={"osc": 7})
    kind, text = ev.verdict_random(z)
    assert kind == "info"
    assert f"kürzer als „Keine“ in 2{NB}983 von 3{NB}000 Fällen, gleich in 4, länger in 13." in text and f"7 Fälle schwingen in 6{NB}000 Perioden nicht ein, davon 7 mit schwingender Strecke." in text


def test_format_helpers():
    assert ev.de(0.5) == "0,50" and ev.de(-1.5, 1) == "−1,5" and ev.de(0.25, 3, sign=True) == "+0,250"
    assert ev.fmt_int(1215) == f"1{NB}215" and ev.fmt_int(12) == "12"
    assert ev.periods(None) == "nie" and ev.periods(1) == "1 Periode" and ev.periods(14) == "14 Perioden" and ev.periods(1244) == f"1{NB}244 Perioden"
    assert ev.fraction_text(ev.hand_example("keine", 3)[1]["e"]) == "5/2" and ev.fraction_text(-ev.hand_example("keine", 3)[2]["e"] / 1) == "1/2"
