"""Kern der Demo: Strecke, PI-Regler mit Stellgrenze und vier Anti-Windup-Varianten, Kennzahlen, Vorhersage des Ausstiegs aus der Stufenantwort.

Eigene Umsetzung in diesem Modell (Namen der Verfahren wie in der Literatur, Regeln hier festgelegt; keine Neuheitsaussage).

Zeitlage: Abtastzeit 1, y_k ist, was der Regler in Periode k sieht. Regler: e = r − y, v = kp·e + I, u = sat(v) mit |u| ≤ UMAX,
I⁺ = I + ki·e + (Anti-Windup-Term). Die Strecke bekommt u_k und liefert y_(k+1) (zwei Verzögerungen in Reihe erst y_(k+2), Totzeit d erst y_(k+1+d)).

Der Kern rechnet mit Python-Zahlen ohne Bibliotheksfunktionen (+, −, ×, ÷, Vergleiche): dieselben Funktionen laufen mit `float` und mit `fractions.Fraction`
(Handbeispiel in Brüchen, tests/). Damit sind die Bahnen auf jeder Plattform dieselben Bits; nur `rho_pi` (Eigenwerte, numpy) hängt an der Linearen Algebra
der Plattform und dient nie als Schwelle einer Messreihe außer „kleiner als 0,97“ der Zufallsfälle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

UMAX = 1                 # Stellgrenze |u| <= 1
TOL_REL = 0.025          # Einschwingband: |y − r2| <= 2,5 % von r2 (bei r2 = 0,8 genau 0,02, dem Band des Nachbaus B1)
HORIZON = 6000           # Perioden nach dem Sprung, höchstens
STILL = 500              # so viele Perioden im Band (und |v| <= UMAX) hintereinander gelten als eingeschwungen: Abbruch des Laufs
DIVERGENZ = 1e12         # |I| darüber: der Lauf gilt als divergiert

VARIANTEN = ("keine", "bc", "bed", "ilim")
VARIANTEN_NAME = {
    "keine": "Keine",
    "bc": "Rückrechnung",
    "bed": "Bedingte Integration",
    "ilim": "Integratorgrenze",
}


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Strecken
# ---------------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Plant:
    """kind 'lag1' (a, d): x⁺ = a·x + G(1 − a)·u_(k−d), mit Totzeit d Perioden;
    'lag2' (a1, a2): zwei Verzögerungen in Reihe: x1⁺ = a1·x1 + G(1 − a1)·u, x2⁺ = a2·x2 + (1 − a2)·x1, y = x2; das erste Glied hat die Verstärkung G, das zweite 1,
    insgesamt G; u_k wirkt erst in y_(k+2), eine Periode später als bei den anderen Arten;
    'osc' (c1, c0): y⁺ = c1·y − c0·y₋ + G(1 − c1 + c0)·u (schwach gedämpft, wenn die Pole nahe am Einheitskreis liegen). Jede Strecke hat die Gleichstromverstärkung G."""

    kind: str
    par: tuple
    gain: float = 1.0

    def state(self) -> "_State":
        return _State(self)

    def step_response(self, n: int, u=UMAX) -> list:
        """Offene Antwort y_0 … y_(n−1) auf konstantes u ab Nullzustand (y_k vor dem Schritt k)."""
        s = self.state()
        out = []
        for _ in range(n):
            out.append(s.y)
            s.step(u)
        return out

    def matrices(self):
        """(A, B, C) der Strecke als Zustandsraummodell (Totzeit als Zustandskette), für Stabilitätsprüfung und unabhängige Gegenproben."""
        g = float(self.gain)
        if self.kind == "lag1":
            a, d = self.par
            a = float(a)
            n = 1 + d
            A = np.zeros((n, n))
            B = np.zeros((n, 1))
            C = np.zeros((1, n))
            A[0, 0] = a
            C[0, 0] = 1.0
            if d == 0:
                B[0, 0] = g * (1 - a)
            else:
                A[0, d] = g * (1 - a)
                for j in range(1, d):
                    A[j + 1, j] = 1.0
                B[1, 0] = 1.0
            return A, B, C
        if self.kind == "lag2":
            a1, a2 = (float(p) for p in self.par)
            return (np.array([[a1, 0.0], [1 - a2, a2]]), np.array([[g * (1 - a1)], [0.0]]), np.array([[0.0, 1.0]]))
        c1, c0 = (float(p) for p in self.par)
        return (np.array([[c1, -c0], [1.0, 0.0]]), np.array([[g * (1 - c1 + c0)], [0.0]]), np.array([[1.0, 0.0]]))


class _State:
    __slots__ = ("kind", "a", "b", "x1", "x2", "buf", "g")

    def __init__(self, plant: Plant):
        self.kind = plant.kind
        self.g = plant.gain
        self.x1 = 0
        self.x2 = 0
        self.buf = None
        if plant.kind == "lag1":
            a, d = plant.par
            self.a = a
            self.b = d
            self.buf = deque([0] * d, maxlen=d) if d > 0 else None
        elif plant.kind == "lag2":
            self.a, self.b = plant.par
        elif plant.kind == "osc":
            self.a, self.b = plant.par
        else:
            raise ValueError(f"unbekannte Streckenart {plant.kind!r}")

    @property
    def y(self):
        return self.x2 if self.kind == "lag2" else self.x1

    def step(self, u):
        kind = self.kind
        if kind == "lag1":
            if self.buf is not None:
                uin = self.buf[0]
                self.buf.append(u)               # maxlen: das älteste fällt heraus
            else:
                uin = u
            self.x1 = self.a * self.x1 + self.g * (1 - self.a) * uin
        elif kind == "lag2":
            n1 = self.a * self.x1 + self.g * (1 - self.a) * u
            n2 = self.b * self.x2 + (1 - self.b) * self.x1
            self.x1, self.x2 = n1, n2
        else:
            n1 = self.a * self.x1 - self.b * self.x2 + self.g * (1 - self.a + self.b) * u
            self.x2 = self.x1
            self.x1 = n1


def rho_pi(plant: Plant, kp: float, ki: float) -> float:
    """Spektralradius der linearen Schleife (ohne Stellgrenze): Zustand (Streckenzustand, I); Eigenwerte mit numpy."""
    A, B, C = plant.matrices()
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A - kp * B @ C
    M[:n, n] = B[:, 0]
    M[n, :n] = -ki * C[0]
    M[n, n] = 1.0
    return float(max(abs(np.linalg.eigvals(M))))


def char_poly_lag1(a: float, gain: float, kp: float, ki: float) -> list:
    """Charakteristisches Polynom z² − (1 + a − b·kp)·z + (a − b·kp + b·ki) der Schleife aus erster Ordnung (ohne Totzeit) und PI, b = G(1 − a)."""
    b = gain * (1 - a)
    return [1.0, -(1 + a - b * kp), (a - b * kp) + b * ki]


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Regler
# ---------------------------------------------------------------------------------------------------------------------------------------------

def sat(v, umax=UMAX):
    return umax if v > umax else (-umax if v < -umax else v)


def integrator_update(variant: str, I, e, v, u, ki, kaw, umax=UMAX):
    """Neuer Integratorwert nach einer Periode (die vier Varianten, eigene Festlegungen):
    keine: I + ki·e;  bc (Rückrechnung): I + ki·e + kaw·(u − v);  bed (bedingte Integration): I bleibt stehen, solange v ≠ u und e in Richtung der Sättigung zeigt
    (e·(v − u) > 0);  ilim (Integratorgrenze): I + ki·e, abgeschnitten auf [−UMAX, UMAX]."""
    if variant == "keine":
        return I + ki * e
    if variant == "bc":
        return I + ki * e + kaw * (u - v)
    if variant == "bed":
        if v != u and e * (v - u) > 0:
            return I
        return I + ki * e
    if variant == "ilim":
        return max(-umax, min(umax, I + ki * e))
    raise ValueError(f"unbekannte Variante {variant!r}")


@dataclass(frozen=True)
class Kz:
    """Kennzahlen nach dem Sprung auf r2 (Zähleinheit Periode).
    einschwing: Perioden ab dem Sprung bis zur LETZTEN Ausfahrt aus dem Band (nicht der ersten Einfahrt: sie täuscht bei Unterschwingen);
    ausstieg: Perioden ab dem Sprung bis zur ersten Periode mit |v| <= UMAX (None: nie);
    ueber / unter: größtes y − r2 bzw. r2 − y nach dem Sprung (nicht negativ); erste_einfahrt: erste Periode im Band; n_sat: Perioden nach dem Sprung mit |v| > UMAX;
    n: Zahl der gerechneten Perioden nach dem Sprung; y_sprung: y am Anfang der Periode des Sprungs (liegt es nicht über r2, gibt es kein „Unterschwingen“ im Sinne
    der Messung: y nähert sich dann von unten); divergiert: |I| ist über DIVERGENZ gewachsen (dann sind die übrigen Felder leer)."""

    einschwing: int | None
    ausstieg: int | None
    ueber: float
    unter: float
    erste_einfahrt: int | None
    n_sat: int
    n: int
    divergiert: bool = False
    y_sprung: float | None = None


@dataclass(frozen=True)
class Bahn:
    """Aufgezeichneter Lauf: Werte am Anfang der Periode k (I vor dem Update)."""

    r: list
    y: list
    e: list
    v: list
    u: list
    I: list


def simulate(plant: Plant, kp, ki, variant: str, kaw, t1: int, r1, r2, horizon: int = HORIZON, still: int | None = STILL, noise=None, tol=None,
             record: bool = False, umax=UMAX):
    """Der Regelkreis: Sollwert r1 in den Perioden 0 … t1 − 1, danach r2; Anti-Windup-Variante `variant` (kaw nur für 'bc').
    Läuft höchstens t1 + horizon Perioden und bricht nach `still` Perioden im Band (und |v| <= umax) ab; `still=None`: ohne Abbruch.
    noise: None oder Feld mit Messrauschen (wird zum gemessenen y addiert, Länge mindestens t1 + horizon). tol: Band (Standard 2,5 % von |r2|).
    Gibt Kz zurück, mit record=True (Kz, Bahn)."""
    if variant not in VARIANTEN:
        raise ValueError(f"unbekannte Variante {variant!r}")
    state = plant.state()
    if noise is not None and hasattr(noise, "tolist"):
        noise = noise.tolist()
    if tol is None:
        tol = TOL_REL * abs(r2)
    I = 0
    ruhig = 0
    last_out = 0
    y_sprung = None
    ausstieg = None
    erste = None
    y_min = None
    y_max = None
    n_sat = 0
    rec = ([], [], [], [], [], []) if record else None
    n_total = t1 + horizon
    k = -1
    for k in range(n_total):
        r = r1 if k < t1 else r2
        y = state.y
        ym = y if noise is None else y + noise[k]
        e = r - ym
        v = kp * e + I
        u = umax if v > umax else (-umax if v < -umax else v)
        state.step(u)
        I_vor = I
        I = integrator_update(variant, I, e, v, u, ki, kaw, umax)
        if abs(I) > DIVERGENZ or I != I:
            kz = Kz(None, None, float("nan"), float("nan"), None, 0, 0, True)
            return (kz, None) if record else kz
        if record:
            rec[0].append(r)
            rec[1].append(y)
            rec[2].append(e)
            rec[3].append(v)
            rec[4].append(u)
            rec[5].append(I_vor)
        if k >= t1:
            j = k - t1
            if j == 0:
                y_sprung = float(y)
            d = y - r2
            if y_min is None or y < y_min:
                y_min = y
            if y_max is None or y > y_max:
                y_max = y
            im_band = abs(d) <= tol
            if not im_band:
                last_out = j + 1
            elif erste is None:
                erste = j
            if abs(v) <= umax:
                if ausstieg is None:
                    ausstieg = j
            else:
                n_sat += 1
            if still is not None:
                if im_band and abs(v) <= umax:
                    ruhig += 1
                    if ruhig >= still:
                        break
                else:
                    ruhig = 0
    n_nach = max(k + 1 - t1, 0)
    if y_min is None:                       # Lauf endete vor dem Sprung (t1 >= Länge)
        kz = Kz(0, None, 0.0, 0.0, None, 0, 0)
    else:
        kz = Kz(last_out, ausstieg, float(max(y_max - r2, 0)), float(max(r2 - y_min, 0)), erste, n_sat, n_nach, False, y_sprung)
    if record:
        return kz, Bahn(*rec)
    return kz


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Vorhersage des Ausstiegs aus der Sättigung (anderer Rechenweg: offene Stufenantwort und Summen, keine Regler-Simulation)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def predict_exit(plant: Plant, kp, ki, t1: int, r1, r2, max_periods: int = HORIZON, umax=UMAX):
    """Ausstieg aus der Sättigung ohne Regler vorhergesagt. Gilt nur, wenn |u| = umax in allen Perioden 0 … t1 − 1 galt (Annahme, wird geprüft).
    Solange u = umax gilt, ist y die offene Stufenantwort s_k der Strecke, I_(T1) = ki·Σ_(k<T1)(r1 − s_k), danach fällt I je Periode um ki·(s_k − r2),
    und der Ausstieg ist die erste Periode j mit kp·(r2 − s_j) + I_j <= umax.
    Rückgabe (perioden_nach_dem_sprung, abstand, status): abstand = Abstand von v zur Grenze in der Ausstiegsperiode und in der Periode davor (kleinster Wert;
    ein Abstand nahe 1e−12 wäre ein Gleichstand, den die Rundung entscheidet); status 'ok', 'annahme' (Sättigung schon vor dem Sprung beendet) oder 'kein_ausstieg'."""
    s = plant.step_response(t1 + max_periods, umax)
    I = 0
    for k in range(t1):                       # Annahme prüfen: v > umax bis zum Sprung
        v = kp * (r1 - s[k]) + I
        if v <= umax:
            return None, None, "annahme"
        I = I + ki * (r1 - s[k])
    j = t1
    while j < t1 + max_periods:
        v = kp * (r2 - s[j]) + I
        if v <= umax:
            abstand = abs(v - umax)
            if j > t1:
                v_davor = kp * (r2 - s[j - 1]) + I + ki * (s[j - 1] - r2)
                abstand = min(abstand, abs(v_davor - umax))
            return j - t1, float(abstand), "ok"
        I = I - ki * (s[j] - r2)
        j += 1
    return None, None, "kein_ausstieg"


def exit_approximation(plant: Plant, kp, ki, t1: int, r1, r2, umax=UMAX):
    """Näherung N ≈ S/Δ − (umax + kp·Δ)/(ki·Δ) mit S = Σ_(k<t1)(r1 − s_k) und Δ = G·umax − r2 (Verstärkung G der Strecke); None, wenn Δ <= 0 (Sollwert bleibt unerreichbar)."""
    delta = plant.gain * umax - r2
    if delta <= 0 or ki == 0:
        return None
    s = plant.step_response(t1, umax)
    S = sum(r1 - sk for sk in s)
    return float(S / delta - (umax + kp * delta) / (ki * delta))


def exit_slope(plant: Plant, r1, r2, umax=UMAX):
    """Zuwachs des Ausstiegs je zusätzlicher gesättigter Periode: (r1 − G·umax)/(G·umax − r2) (eigene Herleitung, Näherung für lange Sättigung); None, wenn Δ <= 0."""
    delta = plant.gain * umax - r2
    if delta <= 0:
        return None
    return float((r1 - plant.gain * umax) / delta)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Rückrechnung: Gleichgewicht des Integrators in der Sättigung, v nach dem Sprung
# ---------------------------------------------------------------------------------------------------------------------------------------------

def integrator_recursion(kp, ki, kaw, errors, umax=UMAX, I0=0.0) -> float:
    """I nach len(errors) Perioden Sättigung (Rückrechnung): I⁺ = (1 − kaw)·I + (ki − kaw·kp)·e + kaw·umax mit den Fehlern e_k der Periodenfolge `errors`;
    gilt, solange v gesättigt bleibt (u = umax). Andere Form der Rechnung als `integrator_update` (u − v ist hier schon eingesetzt)."""
    I = I0
    for e in errors:
        I = (1 - kaw) * I + (ki - kaw * kp) * e + kaw * umax
    return I


def v_after_jump(kp, ki, kaw, y_sat, r1, r2, umax=UMAX):
    """v unmittelbar nach dem Sprung bei Rückrechnung im Gleichgewicht der Sättigung: v = umax − kp·Δ + (ki/kaw − kp)·(r1 − y_sat), Δ = y_sat − r2.
    Näherung für y ≈ y_sat und e ≈ konstant. Bei kaw > ki/kp liegt v tiefer als bei kaw = ki/kp, in Fall a1 ab kaw = 0,2 weit unter −umax (passt zum Unterschwingen), bei kaw = ki/kp fällt der Term mit e weg."""
    delta = y_sat - r2
    return umax - kp * delta + (ki / kaw - kp) * (r1 - y_sat)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Messrauschen: Mittel von y, Anteil gesättigter Perioden, Identität der Summen
# ---------------------------------------------------------------------------------------------------------------------------------------------

def noise_run(plant: Plant, kp, ki, variant: str, kaw, r, noise, n_warm: int = 2000, umax=UMAX) -> dict:
    """Regelkreis mit konstantem Sollwert r und Messrauschen `noise` (Feld, zum gemessenen y addiert). Die ersten n_warm Perioden zählen nicht.
    Rückgabe: divergiert (der Integrator ist über DIVERGENZ gewachsen: dann sind alle Kennzahlen nan), mean_y, std_y (über das wahre y), sat_share (Anteil Perioden mit v ≠ u), mean_e (Mittel des Fehlers e = r − (y + Rauschen)) und die
    Summen der Identität  Σe = (kaw/ki)·Σ(v − u) + (I_Ende − I_Start)/ki  (nur für 'bc' gültig: links, rechts)."""
    state = plant.state()
    noise = noise.tolist() if hasattr(noise, "tolist") else list(noise)
    I = 0.0
    I_start = 0.0
    ys = []
    n_sat = 0
    sum_e = 0.0
    sum_vu = 0.0
    n = len(noise)
    for k in range(n):
        if k == n_warm:
            I_start = I
        y = state.y
        e = r - (y + noise[k])
        v = kp * e + I
        u = umax if v > umax else (-umax if v < -umax else v)
        state.step(u)
        I = integrator_update(variant, I, e, v, u, ki, kaw, umax)
        if abs(I) > DIVERGENZ or I != I:
            nan = float("nan")
            return dict(divergiert=True, mean_y=nan, std_y=nan, sat_share=nan, mean_e=nan, sum_e=nan, sum_vu=nan, I_start=nan, I_end=nan, n=0)
        if k >= n_warm:
            ys.append(y)
            n_sat += v != u
            sum_e += e
            sum_vu += v - u
    arr = np.asarray(ys, dtype=float)
    m = len(ys)
    out = dict(divergiert=False, mean_y=float(arr.mean()), std_y=float(arr.std()), sat_share=n_sat / m, mean_e=sum_e / m, sum_e=sum_e, sum_vu=sum_vu, I_start=I_start, I_end=I, n=m)
    if variant == "bc" and ki != 0:
        out["identity_left"] = sum_e
        out["identity_right"] = kaw / ki * sum_vu + (I - I_start) / ki
    return out


def uniform_noise(seed: int, n: int, amplitude: float):
    """Gleichverteiltes Messrauschen in [−amplitude, amplitude] aus SplitMix64 (nur Ganzzahl- und Multiplikationsarithmetik: auf jeder Plattform dieselben Bits)."""
    from wpi_rng import uniform_array

    return (uniform_array(seed, n) - 0.5) * 2 * amplitude
