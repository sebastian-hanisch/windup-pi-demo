"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Preset-Regler stehen in der Seitenleiste (Strecke, Regler,
Sollwertfolge) und im Hauptbereich (Variante und kaw in Experiment 1, Rauschen in Experiment 5); der Test unten stellt sicher, dass JEDER gezeichnete Regler entweder in
SETTING_SPECS oder in der Liste der reinen Experimentregler steht, und dass jeder Regler aus SETTING_SPECS gezeichnet wird."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import wpi_constants as C
import wpi_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for bereich in (at.sidebar, at.main):
        for art in REGLER_ARTEN:
            try:
                return getattr(bereich, art)(key=key)
            except KeyError:
                continue
    raise AssertionError(f"Regler {key} wird nicht gezeichnet")


def _alle_regler(at):
    return [w for bereich in (at.sidebar, at.main) for art in REGLER_ARTEN for w in getattr(bereich, art) if w.key]


def _anderer_wert(spec, aktuell):
    if spec.choices is not None:
        return next(c for c in spec.choices if c != aktuell)
    return spec.hi if aktuell != spec.hi else spec.lo


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [C.DEFAULT_PRESET]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_gezeichnete_regler_steht_in_setting_specs_oder_ist_ein_reiner_experimentregler():
    at = _app()
    schluessel = {w.key for w in _alle_regler(at)}
    fehl = {k for k in schluessel if k not in P.SETTING_SPECS and k not in P.EXPERIMENT_KEYS and not k.startswith(P.EXPERIMENT_KEY_PREFIXES)}
    assert not fehl, f"Regler {sorted(fehl)} stehen weder in SETTING_SPECS noch in EXPERIMENT_KEYS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel, f"nicht gezeichnet: {sorted(set(P.SETTING_SPECS) - schluessel)}"


def test_preset_regler_der_seitenleiste_und_des_hauptbereichs():
    at = _app()
    seitenleiste = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    assert seitenleiste == {"kind_select", "a1_slider", "a2_slider", "dead_slider", "theta_slider", "gain_slider", "kp_slider", "ki_slider", "t1_slider", "r1_slider", "r2_slider"}
    hauptbereich = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key}
    assert {"variant_radio", "kaw_slider", "noise_amp_slider", "noise_r_slider", "noise_seed_input"} <= hauptbereich


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    if regler.disabled:                                   # gesperrte Regler (nicht benutzt bei dieser Streckenart bzw. Variante) lassen sich nicht per Widget setzen
        at.session_state[state_key] = _anderer_wert(spec, regler.value)
        at.run()
    else:
        regler.set_value(_anderer_wert(spec, regler.value)).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    # erneutes Anklicken markiert wieder
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_nicht_benutzte_streckenregler_sind_gesperrt():
    at = _app()                                                       # Voreinstellung: Streckenart „Erste Ordnung“
    assert at.sidebar.slider(key="a2_slider").disabled and at.sidebar.slider(key="dead_slider").disabled and at.sidebar.slider(key="theta_slider").disabled
    assert not at.sidebar.slider(key="a1_slider").disabled and at.main.slider(key="kaw_slider").disabled     # kaw wirkt nur bei „Rückrechnung, kaw frei“
    at.sidebar.selectbox(key="kind_select").set_value("tot").run()
    assert not at.sidebar.slider(key="dead_slider").disabled and at.sidebar.slider(key="a2_slider").disabled
    at.sidebar.selectbox(key="kind_select").set_value("osc").run()
    assert not at.sidebar.slider(key="theta_slider").disabled
    at.sidebar.selectbox(key="kind_select").set_value("lag2").run()
    assert not at.sidebar.slider(key="a2_slider").disabled
    at.main.radio(key="variant_radio").set_value("bc").run()
    assert not at.main.slider(key="kaw_slider").disabled
