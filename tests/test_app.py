"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen,
wählt jeden Eintrag der Auswahlfelder, lädt Permalinks und prüft, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key=) und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max). Dazu Worst-Case-Einstellungen mit Zeitgrenze."""

import os
import re
import time

import pytest
from streamlit.testing.v1 import AppTest

import wpi_constants as C
import wpi_evaluation as ev

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [m.value for m in at.caption] + [w.value for w in at.warning] + [i.value for i in at.info] + [s.value for s in at.success]
    parts += [str(d.value) for d in at.dataframe]
    return "\n".join(parts)


def test_app_loads_without_exception():
    _fresh_app()


def test_app_shows_the_six_sections_and_the_math_expander():
    at = _fresh_app()
    headings = " ".join(m.value for m in at.markdown) + " " + " ".join(s.value for s in at.subheader)
    for part in ("🎯 Experiment 1: Windup von Hand", "❓ Kernfragen", "Experiment 2: Wie lange dauert es?", "Vier Streckenarten, 24 Parametersätze", "kaw wählen", "Messrauschen nahe der Grenze",
                 "Wann hilft es nicht?", "Wo die Annahmen enden"):
        assert part in headings, part
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)          # per Titel, nicht per Index (DEMO-PLAYBOOK §2)
    assert not re.search(r"\bnan\b", _all_text(at), re.I)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            if slider.disabled:                          # gesperrte Regler (nicht benutzt bei dieser Streckenart bzw. Variante) lassen sich nicht per Widget setzen
                at.session_state[slider.key] = value
                at.run(timeout=TIMEOUT)
            else:
                slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_choice_of_the_selectboxes_and_the_variant_radio_does_not_raise():
    at = _fresh_app()
    for kind in C.KINDS:
        at.sidebar.selectbox(key="kind_select").set_value(kind).run(timeout=TIMEOUT)
        assert not at.exception, (kind, [str(e) for e in at.exception])
    for variant in ev.VARIANT_CHOICES:
        at.main.radio(key="variant_radio").set_value(variant).run(timeout=TIMEOUT)
        assert not at.exception, (variant, [str(e) for e in at.exception])
    for cid in C.CASE_IDS:
        at.main.selectbox(key="case_select").set_value(cid).run(timeout=TIMEOUT)
        assert not at.exception, (cid, [str(e) for e in at.exception])


def test_every_exception_of_experiment_6_loads_with_each_of_its_counter_measures():
    at = _fresh_app()
    exc = ev.frozen()["zufall"]["ausnahmen"]
    assert len(exc) == 21
    for e in exc:
        at.main.selectbox(key="exc_select").set_value(e["index"]).run(timeout=TIMEOUT)
        assert not at.exception, (e["index"], [str(x) for x in at.exception])
        radio = at.main.radio(key=f"exc_variant_select_{e['index']}")
        for variant in e["laenger"]:
            radio.set_value(variant).run(timeout=TIMEOUT)
            assert not at.exception, (e["index"], variant, [str(x) for x in at.exception])
            radio = at.main.radio(key=f"exc_variant_select_{e['index']}")


def test_unsettled_case_shows_the_limit_cycle_warning():
    at = _fresh_app()
    idx = ev.frozen()["zufall"]["nicht_eingeschwungen"][0]
    at.main.selectbox(key="exc_select").set_value(idx).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("schwingt in" in w.value and "Grenzzyklus" in w.value for w in at.warning)


def test_kaw_2_5_in_experiment_4_shows_diverges_not_nan():
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{C.LANG}").click().run(timeout=TIMEOUT)
    at.main.slider(key="kaw4_slider").set_value(2.5).run(timeout=TIMEOUT)
    assert not at.exception
    text = _all_text(at)
    assert "divergiert" in text and not re.search(r"\bnan\b", text, re.I)


def test_permalink_values_are_taken_and_clamped_and_invalid_values_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["plant"] = "osc"
    at.query_params["a1"] = "0.7"
    at.query_params["kp"] = "99"          # über der Grenze: auf 5,0 begrenzt
    at.query_params["t1"] = "abc"         # ungültig: ignoriert
    at.query_params["var"] = "gibtsnicht"  # nicht erlaubt: ignoriert
    at.query_params["seed"] = "-5"        # unter 0: auf 0 begrenzt
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.selectbox(key="kind_select").value == "osc" and at.sidebar.slider(key="a1_slider").value == pytest.approx(0.7)
    assert at.sidebar.slider(key="kp_slider").value == pytest.approx(5.0) and at.sidebar.slider(key="t1_slider").value == 2
    assert at.main.radio(key="variant_radio").value == "keine" and at.main.number_input(key="noise_seed_input").value == 0


@pytest.mark.parametrize("scenario", [
    ev.Scenario(kind="lag2", a1=0.99, a2=0.99, gain=3.0, kp=0.1, ki=0.01, t1=240, r1=10.0, r2=2.9),     # träge Strecke, kleine Verstärkungen, längste Sättigung
    ev.Scenario(kind="osc", a1=0.99, theta=0.05, gain=3.0, kp=5.0, ki=0.5, t1=240, r1=10.0, r2=0.1),    # schwach gedämpft, größte Verstärkungen
    ev.Scenario(kind="tot", a1=0.99, dead=8, gain=0.5, kp=5.0, ki=0.5, t1=240, r1=10.0, r2=0.4),         # Totzeit 8 mit instabiler Schleife
])
def test_worst_case_settings_finish_within_the_time_limit(scenario):
    """Worst-Case-Test am Regler-Maximum (DEMO-PLAYBOOK §1c): Zeitgrenze statt Zeit-Assertion auf die Sekunde; der Lauf mit allen sechs Experimenten bleibt unter 60 s (gemessen unter 2 s)."""
    at = _fresh_app()
    values = {"kind_select": scenario.kind, "a1_slider": scenario.a1, "a2_slider": scenario.a2, "dead_slider": scenario.dead, "theta_slider": scenario.theta, "gain_slider": scenario.gain,
              "kp_slider": scenario.kp, "ki_slider": scenario.ki, "t1_slider": scenario.t1, "r1_slider": scenario.r1, "r2_slider": scenario.r2}
    for key, value in values.items():
        at.session_state[key] = value                    # direkt in den Zustand: gesperrte Regler lassen sich nicht per Widget setzen
    at.session_state["variant_radio"] = "bc_kp"
    t0 = time.time()
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]
    assert time.time() - t0 < 60
