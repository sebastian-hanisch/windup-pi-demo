"""Konstanten: Reglergrenzen, feste Annahmen, Farben, die 24 Fälle und die Presets (Schnellstart)."""

from wpi_dynamics import Plant

# --- Szenario der Messungen: unerreichbarer Sollwert r1 für T1 Perioden, danach r2 ---------------------------------------
SZ_T1, SZ_R1, SZ_R2 = 57, 4.7, 0.77   # ungerundet gewählt: bei den runden Zahlen (5, 60, 0,8) fällt der Ausstieg auf einen Gleichstand v = 1 (README, B1)
B1_T1, B1_R1, B1_R2 = 60, 5.0, 0.8     # der Gleichstandsfall, nur als Nachbau in den Tests und in der README (kein Preset)

# --- Regler der Seitenleiste: (Minimum, Maximum) -------------------------------------------------------------------------------------------------
KINDS = ("lag1", "tot", "lag2", "osc")
KIND_NAME = {
    "lag1": "Erste Ordnung",
    "tot": "Erste Ordnung mit Totzeit",
    "lag2": "Zwei Verzögerungen in Reihe",
    "osc": "Schwach gedämpft (zweite Ordnung)",
}
A_MIN, A_MAX = 0.0, 0.99            # Pol a (bzw. a₁, bzw. Polradius r bei der schwingenden Strecke)
THETA_MIN, THETA_MAX = 0.05, 1.5    # Polwinkel θ der schwingenden Strecke (Bogenmaß)
DEAD_MIN, DEAD_MAX = 0, 8
GAIN_MIN, GAIN_MAX = 0.5, 3.0
KP_MIN, KP_MAX = 0.1, 5.0
KI_MIN, KI_MAX = 0.01, 0.5
T1_MIN, T1_MAX = 0, 240
R1_MIN, R1_MAX = 0.0, 10.0
R2_MIN, R2_MAX = 0.0, 3.0
KAW_MIN, KAW_MAX = 0.0, 2.5         # 2,5 und mehr divergiert in der Sättigung (B7)
NOISE_MIN, NOISE_MAX = 0.0, 0.8
NOISE_R_MIN, NOISE_R_MAX, NOISE_R_DEFAULT = 0.1, 1.5, 0.9     # Sollwert im Rauschexperiment (nahe der Stellgrenze 1)
SEED_MAX = 999999
KAW4_MIN, KAW4_MAX = 0.01, 2.5      # Experiment 4: eigener Regler, nicht Teil der Presets

# --- Experimente ----------------------------------------------------------------------------------------------------------------------------------
T1_SWEEP = (1, 2, 3, 5, 10, 20, 40, 60, 120, 240)      # Experiment 2
KAW_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 1.9, 2.0, 2.1, 2.5)   # Experiment 4: Kurve (dazu ki/kp und der Regler)
NOISE_AMPLITUDES = (0.2, 0.4, 0.8)
NOISE_PERIODS, NOISE_WARM, NOISE_DEFAULT_SEED = 22000, 2000, 15     # 20 000 Perioden nach 2 000 Einschwingperioden
NOISE_KAWS = (0.05, 0.5, 1.0)
RANDOM_SEED, RANDOM_N, RANDOM_RHO_MAX = 1506, 3000, 0.97            # Experiment 6 und B8

# --- Farben ----------------------------------------------------------------------------------------------------------------------------------------
COLOR_Y = "#2F6B65"
COLOR_U = "#c0504d"
COLOR_V = "#9b59b6"
COLOR_I = "#e69f00"
COLOR_R = "#555555"
COLOR_BAND = "rgba(47,107,101,0.15)"
VARIANT_COLOR = {"keine": "#c0504d", "bc": "#2F6B65", "bed": "#4c78a8", "ilim": "#e69f00"}

# --- Die 24 Fälle: (Kennung, Streckenart, Parameter, Verstärkung G, kp, ki) -----------------------------------------------------
FAELLE = (
    ("a1", "Erste Ordnung, a = 0,9", "lag1", (0.9, 0), 1.0, 2.0, 0.2),
    ("a2", "Erste Ordnung, a = 0,9", "lag1", (0.9, 0), 1.0, 0.5, 0.05),
    ("a3", "Erste Ordnung, a = 0,9", "lag1", (0.9, 0), 1.0, 5.0, 0.5),
    ("a4", "Erste Ordnung, a = 0,5", "lag1", (0.5, 0), 1.0, 1.0, 0.5),
    ("a5", "Erste Ordnung, a = 0,5", "lag1", (0.5, 0), 1.0, 0.5, 0.2),
    ("a6", "Erste Ordnung, a = 0,97", "lag1", (0.97, 0), 1.0, 3.0, 0.2),
    ("a7", "Erste Ordnung, a = 0,97", "lag1", (0.97, 0), 1.0, 5.0, 0.5),
    ("a8", "Erste Ordnung, a = 0,9, G = 2", "lag1", (0.9, 0), 2.0, 1.0, 0.2),
    ("b1", "Zwei Verzögerungen 0,8 / 0,9", "lag2", (0.8, 0.9), 1.0, 2.0, 0.2),
    ("b2", "Zwei Verzögerungen 0,8 / 0,9", "lag2", (0.8, 0.9), 1.0, 1.0, 0.1),
    ("b3", "Zwei Verzögerungen 0,5 / 0,5", "lag2", (0.5, 0.5), 1.0, 0.5, 0.2),
    ("b4", "Zwei Verzögerungen 0,5 / 0,5", "lag2", (0.5, 0.5), 1.0, 2.0, 0.5),
    ("b5", "Schwingend c₁ = 1,6 / c₀ = 0,8", "osc", (1.6, 0.8), 1.0, 2.0, 0.2),
    ("b6", "Schwingend c₁ = 1,6 / c₀ = 0,8", "osc", (1.6, 0.8), 1.0, 1.0, 0.1),
    ("b7", "Zwei Verzögerungen 0,9 / 0,9", "lag2", (0.9, 0.9), 1.0, 3.0, 0.2),
    ("b8", "Zwei Verzögerungen 0,9 / 0,9", "lag2", (0.9, 0.9), 1.0, 2.0, 0.1),
    ("c1", "Totzeit 3, a = 0,9", "lag1", (0.9, 3), 1.0, 2.0, 0.2),
    ("c2", "Totzeit 3, a = 0,9", "lag1", (0.9, 3), 1.0, 1.0, 0.08),
    ("c3", "Totzeit 5, a = 0,8", "lag1", (0.8, 5), 1.0, 0.5, 0.1),
    ("c4", "Totzeit 5, a = 0,8", "lag1", (0.8, 5), 1.0, 1.0, 0.1),
    ("c5", "Totzeit 2, a = 0,5", "lag1", (0.5, 2), 1.0, 0.5, 0.2),
    ("c6", "Totzeit 2, a = 0,5", "lag1", (0.5, 2), 1.0, 1.0, 0.2),
    ("c7", "Totzeit 8, a = 0,95", "lag1", (0.95, 8), 1.0, 1.0, 0.05),
    ("c8", "Totzeit 8, a = 0,95", "lag1", (0.95, 8), 1.0, 2.0, 0.1),
)
CASE_IDS = tuple(f[0] for f in FAELLE)


def case_plant(case_id: str) -> tuple:
    """(Plant, kp, ki) eines der 24 Fälle."""
    for cid, _label, kind, par, gain, kp, ki in FAELLE:
        if cid == case_id:
            return Plant(kind, par, gain), kp, ki
    raise KeyError(case_id)


def case_label(case_id: str) -> str:
    for cid, label, _kind, _par, gain, kp, ki in FAELLE:
        if cid == case_id:
            return f"{cid}: {label}, kp = {kp:g}, ki = {ki:g}".replace(".", ",")
    raise KeyError(case_id)


# --- Presets (Schnellstart in der Seitenleiste); Werte, die ein Preset nicht nennt, gelten als Standard (SETTING_SPECS) ---------------------------------
# variant: keine | bc_kp (Rückrechnung mit kaw = ki/kp) | bc (Rückrechnung mit freiem kaw) | bed | ilim
HAND = "Von Hand (a = ½, T₁ = 2)"
LANG = "Lange Sättigung (Strecke 0,9, T₁ = 57)"
TOT = "Totzeit 3"
OSC = "Schwingende Strecke"
RAUSCH = "Rauschen nahe der Grenze"
GROSS = "Zu großes kaw"

PRESETS = {
    HAND: {"kind": "lag1", "a1": 0.5, "a2": 0.5, "dead": 3, "theta": 0.5, "gain": 1.0, "kp": 1.0, "ki": 0.5, "t1": 2, "r1": 3.0, "r2": 0.25,
           "variant": "keine", "kaw": 0.5, "noise_amp": 0.2, "noise_r": NOISE_R_DEFAULT, "noise_seed": NOISE_DEFAULT_SEED},
    LANG: {"kind": "lag1", "a1": 0.9, "a2": 0.5, "dead": 3, "theta": 0.5, "gain": 1.0, "kp": 2.0, "ki": 0.2, "t1": SZ_T1, "r1": SZ_R1, "r2": SZ_R2,
           "variant": "keine", "kaw": 0.5, "noise_amp": 0.2, "noise_r": NOISE_R_DEFAULT, "noise_seed": NOISE_DEFAULT_SEED},
    TOT: {"kind": "tot", "a1": 0.9, "a2": 0.5, "dead": 3, "theta": 0.5, "gain": 1.0, "kp": 2.0, "ki": 0.2, "t1": SZ_T1, "r1": SZ_R1, "r2": SZ_R2,
          "variant": "bc_kp", "kaw": 0.5, "noise_amp": 0.2, "noise_r": NOISE_R_DEFAULT, "noise_seed": NOISE_DEFAULT_SEED},
    OSC: {"kind": "osc", "a1": 0.9, "a2": 0.5, "dead": 3, "theta": 0.5, "gain": 1.0, "kp": 2.0, "ki": 0.2, "t1": SZ_T1, "r1": SZ_R1, "r2": SZ_R2,
          "variant": "bc_kp", "kaw": 0.5, "noise_amp": 0.2, "noise_r": NOISE_R_DEFAULT, "noise_seed": NOISE_DEFAULT_SEED},
    RAUSCH: {"kind": "lag1", "a1": 0.9, "a2": 0.5, "dead": 3, "theta": 0.5, "gain": 1.0, "kp": 2.0, "ki": 0.2, "t1": SZ_T1, "r1": SZ_R1, "r2": SZ_R2,
             "variant": "bc", "kaw": 0.5, "noise_amp": 0.4, "noise_r": NOISE_R_DEFAULT, "noise_seed": NOISE_DEFAULT_SEED},
    GROSS: {"kind": "lag1", "a1": 0.9, "a2": 0.5, "dead": 3, "theta": 0.5, "gain": 1.0, "kp": 2.0, "ki": 0.2, "t1": SZ_T1, "r1": SZ_R1, "r2": SZ_R2,
            "variant": "bc", "kaw": 0.5, "noise_amp": 0.2, "noise_r": NOISE_R_DEFAULT, "noise_seed": NOISE_DEFAULT_SEED},
}
PRESET_HELP = {
    HAND: "Das Handbeispiel: Strecke y⁺ = (y + u)/2, PI mit kp = 1 und ki = ½, Sollwert 3 in den Perioden 0 und 1, danach ¼. Alle Zahlen sind Brüche; ohne Anti-Windup bleibt u bis k = 5 an der Grenze.",
    LANG: "Strecke erster Ordnung mit a = 0,9, PI 2 / 0,2, Sollwert 4,7 für 57 Perioden (unerreichbar), danach 0,77: ohne Anti-Windup bleibt die Stellgröße nach dem Sprung noch 929 Perioden an der Grenze.",
    TOT: "Dieselbe Strecke mit drei Perioden Totzeit; die Variante ist Rückrechnung mit kaw = ki/kp.",
    OSC: "Schwach gedämpfte Strecke (Polradius 0,9, Winkel 0,5); die Variante ist Rückrechnung mit kaw = ki/kp. Bei dieser Streckenart liegen 10 der 13 Zufallsfälle, in denen die Rückrechnung länger braucht als „Keine“ (Experiment 6).",
    RAUSCH: "Dieselbe Strecke und derselbe Regler; in Experiment 5 (weiter unten) liegt der Sollwert 0,9 nahe an der Stellgrenze 1 und das Messrauschen ist ±0,4: Der Regler sättigt dann auch im ruhigen Betrieb.",
    GROSS: "Wie „Lange Sättigung“, aber mit Rückrechnung und kaw = 0,5, fünfmal ki/kp: der Regler schießt nach dem Sprung weit unter den Sollwert.",
}
DEFAULT_PRESET = HAND
