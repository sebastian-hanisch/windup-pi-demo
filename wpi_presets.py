"""Permalink (Begrenzen/Einrasten), Presets (Schnellstart) und die Zuordnung der Regler zu den Einstellungen (Standardmuster aus dem Demo-Portfolio).

Alle Regler, die ein Preset setzen kann, stehen in `SETTING_SPECS` (Seitenleiste: Strecke, Regler, Sollwertfolge; Hauptbereich: Variante und kaw in Experiment 1, Rauschen in
Experiment 5). Reine Experimentregler ohne Preset-Wert (kaw in Experiment 4, Fall- und Ausnahmeauswahl) stehen in `EXPERIMENT_KEYS`; tests/test_layout.py prüft, dass jeder
gezeichnete Regler in einer der beiden Listen steht und dass jede Änderung eines Preset-Reglers die Hervorhebung des aktiven Presets aufhebt."""

from __future__ import annotations

import math
from dataclasses import dataclass

import streamlit as st

import wpi_constants as C
from wpi_evaluation import VARIANT_CHOICES, Settings, settings_from_values


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


_D = C.PRESETS[C.DEFAULT_PRESET]
SETTING_SPECS = {
    "kind_select": SettingSpec("plant", str, _D["kind"], choices=C.KINDS),
    "a1_slider": SettingSpec("a1", float, _D["a1"], C.A_MIN, C.A_MAX),
    "a2_slider": SettingSpec("a2", float, _D["a2"], C.A_MIN, C.A_MAX),
    "dead_slider": SettingSpec("dead", int, _D["dead"], C.DEAD_MIN, C.DEAD_MAX),
    "theta_slider": SettingSpec("theta", float, _D["theta"], C.THETA_MIN, C.THETA_MAX),
    "gain_slider": SettingSpec("g", float, _D["gain"], C.GAIN_MIN, C.GAIN_MAX),
    "kp_slider": SettingSpec("kp", float, _D["kp"], C.KP_MIN, C.KP_MAX),
    "ki_slider": SettingSpec("ki", float, _D["ki"], C.KI_MIN, C.KI_MAX),
    "t1_slider": SettingSpec("t1", int, _D["t1"], C.T1_MIN, C.T1_MAX),
    "r1_slider": SettingSpec("r1", float, _D["r1"], C.R1_MIN, C.R1_MAX),
    "r2_slider": SettingSpec("r2", float, _D["r2"], C.R2_MIN, C.R2_MAX),
    "variant_radio": SettingSpec("var", str, _D["variant"], choices=VARIANT_CHOICES),
    "kaw_slider": SettingSpec("kaw", float, _D["kaw"], C.KAW_MIN, C.KAW_MAX),
    "noise_amp_slider": SettingSpec("noise", float, _D["noise_amp"], C.NOISE_MIN, C.NOISE_MAX),
    "noise_r_slider": SettingSpec("nr", float, _D["noise_r"], C.NOISE_R_MIN, C.NOISE_R_MAX),
    "noise_seed_input": SettingSpec("seed", int, _D["noise_seed"], 0, C.SEED_MAX),
}
PRESET_KEYS = {
    "kind": "kind_select", "a1": "a1_slider", "a2": "a2_slider", "dead": "dead_slider", "theta": "theta_slider", "gain": "gain_slider", "kp": "kp_slider",
    "ki": "ki_slider", "t1": "t1_slider", "r1": "r1_slider", "r2": "r2_slider", "variant": "variant_radio", "kaw": "kaw_slider",
    "noise_amp": "noise_amp_slider", "noise_r": "noise_r_slider", "noise_seed": "noise_seed_input",
}
EXPERIMENT_KEYS = ("kaw4_slider", "case_select", "exc_select")
EXPERIMENT_KEY_PREFIXES = ("exc_variant_select_",)      # je Ausnahme ein eigener Schlüssel (die Auswahl hängt vom Fall ab)


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (Auswahlwerte nur, wenn sie erlaubt sind)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            raw = qp[spec.url_param]
            value = spec.caster(float(raw)) if spec.caster is int else spec.caster(raw)
        except (ValueError, TypeError, OverflowError):
            continue
        if spec.choices is not None:
            if value not in spec.choices:
                continue
        else:
            if isinstance(value, float) and not math.isfinite(value):
                continue
            if spec.lo is not None:
                value = max(spec.lo, value)
            if spec.hi is not None:
                value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def _equal(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State, Gleitkommazahlen
    auf 1e-9. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _equal(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def current_settings() -> Settings:
    """Die Einstellungen aus dem Session State (nach dem Zeichnen der Regler)."""
    values = {}
    for key, state_key in PRESET_KEYS.items():
        values[key] = st.session_state.get(state_key, SETTING_SPECS[state_key].default)
    return settings_from_values(values)
