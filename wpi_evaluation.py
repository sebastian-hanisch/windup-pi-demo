"""Kennzahlen, Messreihen und Verdikte der Demo (alles eigene Messung in diesem Modell, bei diesen Parametern).

Die Verdikte sind reine Funktionen `verdict_*(…) -> (Art, Text)`: die App ruft `getattr(st, Art)(Text)`, tests/test_verdicts.py prüft Schwellen und jede Zahl im Text
an handgesetzten Werten. Vorgerechnete Messreihen (24 Fälle, 3 000 Zufallsfälle) liegen in `data/wpi_frozen.json` und werden von `tools/make_frozen.py` erzeugt."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import wpi_constants as C
import wpi_dynamics as D
from wpi_dynamics import Plant, simulate
from wpi_rng import SplitMix64, exact_cos, exact_exp, exact_log

ROOT = Path(__file__).resolve().parent
FROZEN_PATH = ROOT / "data" / "wpi_frozen.json"
NB = " "                                                  # geschütztes Leerzeichen als Tausendertrenner


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Anzeige
# ---------------------------------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma (Anzeige)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def fmt_int(n: int) -> str:
    """Ganze Zahl mit geschütztem Leerzeichen als Tausendertrenner (1479 -> 1 479)."""
    return f"{n:,}".replace(",", NB)


def periods(n: int | None) -> str:
    """Zeitangabe in Perioden; None heißt „nie“ (im Horizont nicht erreicht)."""
    return "nie" if n is None else f"{fmt_int(n)} Perioden" if n != 1 else "1 Periode"


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """Strecke, PI-Regler und Sollwertfolge (Voreinstellung: das Handbeispiel)."""

    kind: str = "lag1"
    a1: float = 0.5
    a2: float = 0.5
    dead: int = 3
    theta: float = 0.5
    gain: float = 1.0
    kp: float = 1.0
    ki: float = 0.5
    t1: int = 2
    r1: float = 3.0
    r2: float = 0.25

    def plant(self) -> Plant:
        """Die Strecke: 'lag1' mit a1; 'tot' mit a1 und Totzeit; 'lag2' mit a1, a2; 'osc' mit Polradius a1 und Winkel θ (c₁ = 2·r·cos θ, c₀ = r²)."""
        if self.kind == "lag1":
            return Plant("lag1", (self.a1, 0), self.gain)
        if self.kind == "tot":
            return Plant("lag1", (self.a1, int(self.dead)), self.gain)
        if self.kind == "lag2":
            return Plant("lag2", (self.a1, self.a2), self.gain)
        if self.kind == "osc":
            return Plant("osc", (2 * self.a1 * exact_cos(self.theta), self.a1 * self.a1), self.gain)
        raise ValueError(f"unbekannte Streckenart {self.kind!r}")

    @property
    def reachable(self) -> bool:
        """Ist der zweite Sollwert unter der größten erreichbaren Ausgabe G·umax (sonst bleibt er auch nach dem Sprung unerreichbar)?"""
        return self.r2 < self.gain * D.UMAX


@dataclass(frozen=True)
class Settings:
    """Alle Einstellungen der App: Szenario, Variante und kaw (Experiment 1), Rauschen (Experiment 5)."""

    scenario: Scenario = Scenario()
    variant: str = "keine"          # keine | bc_kp | bc | bed | ilim
    kaw: float = 0.5
    noise_amp: float = 0.2
    noise_r: float = C.NOISE_R_DEFAULT
    noise_seed: int = C.NOISE_DEFAULT_SEED


VARIANT_CHOICES = ("keine", "bc_kp", "bc", "bed", "ilim")
VARIANT_CHOICE_NAME = {
    "keine": "Keine",
    "bc_kp": "Rückrechnung, kaw = ki/kp",
    "bc": "Rückrechnung, kaw frei",
    "bed": "Bedingte Integration",
    "ilim": "Integratorgrenze",
}


def settings_from_values(v: dict) -> Settings:
    """Aus einem Wörterbuch mit den Schlüsseln der Presets (`wpi_constants.PRESETS`)."""
    sc = Scenario(kind=v["kind"], a1=v["a1"], a2=v["a2"], dead=int(v["dead"]), theta=v["theta"], gain=v["gain"], kp=v["kp"], ki=v["ki"],
                  t1=int(v["t1"]), r1=v["r1"], r2=v["r2"])
    return Settings(sc, v["variant"], v["kaw"], v["noise_amp"], v["noise_r"], int(v["noise_seed"]))


def resolve_variant(choice: str, kp: float, ki: float, kaw: float) -> tuple[str, float]:
    """(Variante des Kerns, kaw): 'bc_kp' ist Rückrechnung mit kaw = ki/kp, 'bc' mit dem freien kaw."""
    if choice == "keine":
        return "keine", 0.0
    if choice == "bc_kp":
        return "bc", ki / kp
    if choice == "bc":
        return "bc", float(kaw)
    if choice in ("bed", "ilim"):
        return choice, 0.0
    raise ValueError(f"unbekannte Wahl {choice!r}")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: Windup von Hand
# ---------------------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    key: str            # keine | bc_kp | bc | bed | ilim
    name: str
    variant: str        # Variante des Kerns
    kaw: float
    kz: D.Kz


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    rho: float
    rows: tuple[Row, ...]
    selected: Row
    bahn_selected: D.Bahn
    bahn_keine: D.Bahn
    n_plot: int
    prediction: tuple          # (Perioden, Abstand, Status) aus der offenen Stufenantwort
    approximation: float | None
    slope: float | None

    @property
    def keine(self) -> Row:
        return self.rows[0]

    def row(self, key: str) -> Row:
        return next(r for r in self.rows if r.key == key)


def _rows(sc: Scenario, kaw: float, plant: Plant) -> tuple[Row, ...]:
    rows = []
    for key in VARIANT_CHOICES:
        variant, k = resolve_variant(key, sc.kp, sc.ki, kaw)
        name = VARIANT_CHOICE_NAME[key] if key != "bc" else f"Rückrechnung, kaw = {de(kaw, 2)}"
        rows.append(Row(key, name, variant, k, simulate(plant, sc.kp, sc.ki, variant, k, sc.t1, sc.r1, sc.r2)))
    return tuple(rows)


def plot_length(sc: Scenario, rows: tuple[Row, ...], selected_key: str) -> int:
    """Zahl der gezeichneten Perioden: der Sprung plus gut das 1,2-Fache der Einschwingzeit der gewählten Variante (mindestens 30, höchstens 4 500 nach dem Sprung)."""
    sel = next(r for r in rows if r.key == selected_key)
    e = sel.kz.einschwing if sel.kz.einschwing is not None else D.HORIZON
    return sc.t1 + min(max(int(1.2 * e) + 10, 30), 4500)


def analyse(settings: Settings) -> Analysis:
    sc = settings.scenario
    plant = sc.plant()
    rows = _rows(sc, settings.kaw, plant)
    selected = next(r for r in rows if r.key == settings.variant)
    n_plot = plot_length(sc, rows, settings.variant)
    horizon = n_plot - sc.t1
    _, bahn_sel = simulate(plant, sc.kp, sc.ki, selected.variant, selected.kaw, sc.t1, sc.r1, sc.r2, horizon=horizon, still=None, record=True)
    _, bahn_keine = simulate(plant, sc.kp, sc.ki, "keine", 0.0, sc.t1, sc.r1, sc.r2, horizon=horizon, still=None, record=True)
    return Analysis(settings, D.rho_pi(plant, sc.kp, sc.ki), rows, selected, bahn_sel, bahn_keine, n_plot,
                    D.predict_exit(plant, sc.kp, sc.ki, sc.t1, sc.r1, sc.r2),
                    D.exit_approximation(plant, sc.kp, sc.ki, sc.t1, sc.r1, sc.r2), D.exit_slope(plant, sc.r1, sc.r2))


def table_rows(bahn: D.Bahn, n: int = 9) -> list[dict]:
    """Die ersten n Perioden (k = 0 … n − 1) mit r, y, e, v, u und I (vor dem Update)."""
    m = min(n, len(bahn.y))
    return [dict(k=k, r=bahn.r[k], y=bahn.y[k], e=bahn.e[k], v=bahn.v[k], u=bahn.u[k], I=bahn.I[k]) for k in range(m)]


def undershoot_applies(kz: D.Kz, r2: float) -> bool:
    """„Unterschwingen“ (r2 − kleinstes y) ist nur eine Messgröße, wenn y beim Sprung über r2 liegt; kommt y von unten (Anlauf), ist es der Anlauf selbst."""
    return kz.y_sprung is not None and kz.y_sprung > r2


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Handbeispiel in Brüchen (feste Aufgabe, unabhängig von den Reglern)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def hand_example(variant: str = "keine", n: int = 9) -> list[dict]:
    """Strecke y⁺ = (y + u)/2, PI mit kp = 1 und ki = 1/2, Grenze |u| ≤ 1, Sollwert 3 in den Perioden 0 und 1, danach 1/4: die Zeilen k = 0 … n − 1 als Brüche."""
    from fractions import Fraction as F

    plant = Plant("lag1", (F(1, 2), 0), F(1))
    kaw = F(1, 2) if variant == "bc" else F(0)
    _, bahn = simulate(plant, F(1), F(1, 2), variant, kaw, 2, F(3), F(1, 4), horizon=n, still=None, record=True, tol=F(1, 100))
    return table_rows(bahn, n)


def fraction_text(x) -> str:
    """Bruch als Text (Zähler/Nenner, ganze Zahlen ohne Nenner), Minus als „−“."""
    s = str(x)
    return s.replace("-", "−")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Wie lange dauert es?
# ---------------------------------------------------------------------------------------------------------------------------------------------

def exit_sweep(sc: Scenario, t1_values=C.T1_SWEEP) -> list[dict]:
    """Ausstieg aus der Sättigung gegen die Dauer T1 der Sättigung: gemessen (Simulation 'keine'), vorhergesagt aus der Stufenantwort, Näherung, dazu die Einschwingzeiten
    der Rückrechnung mit kaw = ki/kp und der bedingten Integration. Der Sollwert-Sprung r1 → r2 und der Regler sind die des Szenarios."""
    plant = sc.plant()
    values = sorted({int(t) for t in t1_values} | ({sc.t1} if sc.t1 > 0 else set()))
    rows = []
    for t1 in values:
        k0 = simulate(plant, sc.kp, sc.ki, "keine", 0.0, t1, sc.r1, sc.r2)
        pred = D.predict_exit(plant, sc.kp, sc.ki, t1, sc.r1, sc.r2)
        bc = simulate(plant, sc.kp, sc.ki, "bc", sc.ki / sc.kp, t1, sc.r1, sc.r2)
        bed = simulate(plant, sc.kp, sc.ki, "bed", 0.0, t1, sc.r1, sc.r2)
        rows.append(dict(t1=t1, ausstieg=k0.ausstieg, einschwing=k0.einschwing, vorhergesagt=pred[0], status=pred[2], abstand=pred[1],
                         naeherung=D.exit_approximation(plant, sc.kp, sc.ki, t1, sc.r1, sc.r2), bc=bc.einschwing, bed=bed.einschwing))
    return rows


def measured_slope(rows: list[dict], lo: int = 120, hi: int = 240):
    """Zuwachs des gemessenen Ausstiegs je zusätzlicher gesättigter Periode zwischen T1 = lo und hi; None, wenn eine der beiden Messungen fehlt."""
    a = next((r for r in rows if r["t1"] == lo), None)
    b = next((r for r in rows if r["t1"] == hi), None)
    if a is None or b is None or a["ausstieg"] is None or b["ausstieg"] is None:
        return None
    return (b["ausstieg"] - a["ausstieg"]) / (hi - lo)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: kaw wählen
# ---------------------------------------------------------------------------------------------------------------------------------------------

def kaw_curve(sc: Scenario, extra: float | None = None, grid=C.KAW_GRID) -> list[dict]:
    """Rückrechnung: Einschwingzeit und Unterschwingen über kaw (Gitter, kaw = ki/kp und der gewählte Wert)."""
    plant = sc.plant()
    ratio = sc.ki / sc.kp
    values = sorted({round(float(k), 12) for k in grid} | {round(ratio, 12)} | ({round(float(extra), 12)} if extra is not None else set()))
    rows = []
    for k in values:
        kz = simulate(plant, sc.kp, sc.ki, "bc", k, sc.t1, sc.r1, sc.r2)
        rows.append(dict(kaw=k, einschwing=kz.einschwing, unter=kz.unter, divergiert=kz.divergiert, ausstieg=kz.ausstieg,
                         is_ratio=abs(k - ratio) < 1e-9, y_sprung=kz.y_sprung))
    return rows


def jump_check(sc: Scenario, kaw: float) -> dict | None:
    """v unmittelbar nach dem Sprung bei Rückrechnung: gemessen (Simulation) gegen die Formel v = umax − kp·Δ + (ki/kaw − kp)(r1 − y_sat), Δ = y_sat − r2.
    None bei kaw = 0 oder t1 = 0 (kein Sprung nach Sättigung)."""
    if kaw <= 0 or sc.t1 <= 0:
        return None
    plant = sc.plant()
    kz, bahn = simulate(plant, sc.kp, sc.ki, "bc", kaw, sc.t1, sc.r1, sc.r2, horizon=2, still=None, record=True)
    if kz.divergiert or bahn is None or len(bahn.v) <= sc.t1:
        return None
    y_sat = bahn.y[sc.t1]
    return dict(v_gemessen=bahn.v[sc.t1], v_formel=D.v_after_jump(sc.kp, sc.ki, kaw, y_sat, sc.r1, sc.r2), y_sat=y_sat, I_sprung=bahn.I[sc.t1], faktor=abs(1 - kaw))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: Messrauschen
# ---------------------------------------------------------------------------------------------------------------------------------------------

def noise_table(sc: Scenario, r: float, amp: float, seed: int, n: int = C.NOISE_PERIODS, n_warm: int = C.NOISE_WARM) -> list[dict]:
    """Mittel von y bei Messrauschen (gleichverteilt in ±amp, SplitMix64) und konstantem Sollwert r (Strecke und Regler aus dem Szenario): ohne Anti-Windup, Rückrechnung mit kaw = 0,05 / 0,5 / 1 / ki/kp,
    bedingte Integration, Integratorgrenze."""
    plant = sc.plant()
    noise = D.uniform_noise(seed, n, amp)
    specs = [("Keine", "keine", 0.0)]
    specs += [(f"Rückrechnung, kaw = {de(k, 2)}", "bc", k) for k in C.NOISE_KAWS]
    specs += [(f"Rückrechnung, kaw = ki/kp = {de(sc.ki / sc.kp, 2)}", "bc", sc.ki / sc.kp), ("Bedingte Integration", "bed", 0.0), ("Integratorgrenze", "ilim", 0.0)]
    rows = []
    for name, variant, kaw in specs:
        res = D.noise_run(plant, sc.kp, sc.ki, variant, kaw, r, noise, n_warm)
        res.update(name=name, variant=variant, kaw=kaw)
        rows.append(res)
    return rows


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Vorgerechnete Messreihen (data/wpi_frozen.json)
# ---------------------------------------------------------------------------------------------------------------------------------------------

CASE_VARIANTS = (("keine", "keine", None), ("bc05", "bc", 0.5), ("bckp", "bc", "ratio"), ("bed", "bed", None), ("ilim", "ilim", None))
CASE_VARIANT_NAME = {"keine": "Keine", "bc05": "Rückrechnung, kaw = 0,5", "bckp": "Rückrechnung, kaw = ki/kp", "bed": "Bedingte Integration", "ilim": "Integratorgrenze"}


def _kz_dict(kz: D.Kz) -> dict:
    return dict(einschwing=kz.einschwing, ausstieg=kz.ausstieg, unter=round(kz.unter, 6), ueber=round(kz.ueber, 6), divergiert=kz.divergiert)


def compute_cases() -> dict:
    """Die 24 Fälle der Vormessung V15b: Spektralradius und Kennzahlen der fünf Varianten (Szenario 4,7 für 57 Perioden, dann 0,77)."""
    out = {}
    for cid, _label, kind, par, gain, kp, ki in C.FAELLE:
        plant = Plant(kind, par, gain)
        entry = {"rho": round(D.rho_pi(plant, kp, ki), 6)}
        for key, variant, kaw in CASE_VARIANTS:
            k = ki / kp if kaw == "ratio" else (0.0 if kaw is None else kaw)
            entry[key] = _kz_dict(simulate(plant, kp, ki, variant, k, C.SZ_T1, C.SZ_R1, C.SZ_R2))
        pred = D.predict_exit(plant, kp, ki, C.SZ_T1, C.SZ_R1, C.SZ_R2)
        entry["vorhersage"] = dict(ausstieg=pred[0], abstand=pred[1], status=pred[2])
        out[cid] = entry
    return out


def random_cases(n: int = C.RANDOM_N, seed: int = C.RANDOM_SEED, rho_max: float = C.RANDOM_RHO_MAX) -> tuple[list[dict], int]:
    """Zufallsfälle der Vormessung V15g mit SplitMix64 (Zufalls-Seed 1506): Strecke zufällig aus vier Arten, PI-Satz mit ρ < 0,97, T1 5 bis 149, r1 = 1,5 bis 6·G, r2 = 0,2 bis 0,9·G.
    Logarithmen und Kosinus über `exact_log`/`exact_cos` (bitgleich auf jeder Plattform). Rückgabe (Fälle, Versuche)."""
    rng = SplitMix64(seed)

    def unif(lo, hi):
        return lo + (hi - lo) * rng.random()

    def logunif(lo, hi):
        return exact_exp(unif(exact_log(lo), exact_log(hi)))

    cases = []
    attempts = 0
    while len(cases) < n:
        attempts += 1
        art = ("lag1", "lag2", "osc", "tot")[int(rng.random() * 4) % 4]
        gain = logunif(0.5, 3.0)
        if art == "lag1":
            plant = Plant("lag1", (unif(0.3, 0.97), 0), gain)
        elif art == "tot":
            a = unif(0.3, 0.95)
            plant = Plant("lag1", (a, 1 + int(rng.random() * 8)), gain)
        elif art == "lag2":
            a1 = unif(0.3, 0.95)
            plant = Plant("lag2", (a1, unif(0.3, 0.95)), gain)
        else:
            rr = unif(0.6, 0.93)
            th = unif(0.15, 1.0)
            plant = Plant("osc", (2 * rr * exact_cos(th), rr * rr), gain)
        kp = logunif(0.3, 4.0)
        ki = logunif(0.03, 0.4)
        rho = D.rho_pi(plant, kp, ki)
        if rho >= rho_max:
            continue
        t1 = int(unif(5, 150))
        r1 = gain * unif(1.5, 6.0)
        r2 = gain * unif(0.2, 0.9)
        kz = {}
        for key, variant, kaw in (("keine", "keine", 0.0), ("bckp", "bc", ki / kp), ("bed", "bed", 0.0), ("ilim", "ilim", 0.0)):
            kz[key] = simulate(plant, kp, ki, variant, kaw, t1, r1, r2)
        pred = D.predict_exit(plant, kp, ki, t1, r1, r2)
        cases.append(dict(art=art, kind=plant.kind, par=list(plant.par), gain=gain, kp=kp, ki=ki, t1=t1, r1=r1, r2=r2, rho=rho, kz=kz, pred=pred))
    return cases, attempts


def _eff(kz: D.Kz) -> float:
    """Einschwingzeit für Vergleiche; ein divergierter Lauf zählt als unendlich lang."""
    return math.inf if kz.divergiert or kz.einschwing is None else kz.einschwing


def summarize_random(cases: list[dict], attempts: int) -> dict:
    """Zählungen der Zufallsfälle: je Gegenmittel kürzer / gleich / länger als „keine“, Mediane je Streckenart, Ausnahmen mit allen Parametern."""
    out: dict = {"seed": C.RANDOM_SEED, "n": len(cases), "versuche": attempts, "vergleich": {}, "median": {}, "ausnahmen": [], "nicht_eingeschwungen": [], "nicht_eingeschwungen_arten": {}}
    for key in ("bckp", "bed", "ilim"):
        kuerzer = sum(1 for c in cases if _eff(c["kz"][key]) < _eff(c["kz"]["keine"]))
        gleich = sum(1 for c in cases if _eff(c["kz"][key]) == _eff(c["kz"]["keine"]))
        laenger = sum(1 for c in cases if _eff(c["kz"][key]) > _eff(c["kz"]["keine"]))
        schlimm = max((_eff(c["kz"][key]) - _eff(c["kz"]["keine"])) for c in cases)
        out["vergleich"][key] = dict(kuerzer=kuerzer, gleich=gleich, laenger=laenger, groesste_verlaengerung=None if math.isinf(schlimm) else int(schlimm))
    import statistics

    for art in ("lag1", "tot", "lag2", "osc"):
        sub = [c for c in cases if c["art"] == art]
        out["median"][art] = dict(n=len(sub), **{k: statistics.median(_eff(c["kz"][k]) for c in sub) for k in ("keine", "bckp", "bed", "ilim")})
    out["windup"] = dict(mit=sum(1 for c in cases if (c["kz"]["keine"].ausstieg or 0) > 0), ohne=sum(1 for c in cases if c["kz"]["keine"].ausstieg == 0))
    ok = [c for c in cases if c["pred"][2] == "ok"]
    out["vorhersage"] = dict(annahme_erfuellt=len(ok), gleich=sum(1 for c in ok if c["pred"][0] == c["kz"]["keine"].ausstieg),
                             kleinster_abstand=min(c["pred"][1] for c in ok))
    for i, c in enumerate(cases):
        laenger = [k for k in ("bckp", "bed", "ilim") if _eff(c["kz"][k]) > _eff(c["kz"]["keine"])]
        if laenger:
            out["ausnahmen"].append(dict(index=i, art=c["art"], kind=c["kind"], par=c["par"], gain=c["gain"], kp=c["kp"], ki=c["ki"], t1=c["t1"], r1=c["r1"], r2=c["r2"],
                                         rho=c["rho"], laenger=laenger, einschwing={k: c["kz"][k].einschwing for k in ("keine", "bckp", "bed", "ilim")},
                                         unter={k: round(c["kz"][k].unter, 4) for k in ("keine", "bckp", "bed", "ilim")}))
        if c["kz"]["bckp"].einschwing is not None and c["kz"]["bckp"].einschwing >= D.HORIZON:
            out["nicht_eingeschwungen"].append(i)
            out["nicht_eingeschwungen_arten"][c["art"]] = out["nicht_eingeschwungen_arten"].get(c["art"], 0) + 1
    out["kompakt"] = [[c["kz"][k].einschwing if c["kz"][k].einschwing is not None else -1 for k in ("keine", "bckp", "bed", "ilim")]
                      + [c["kz"]["keine"].ausstieg if c["kz"]["keine"].ausstieg is not None else -1] for c in cases]
    return out


@lru_cache(maxsize=1)
def frozen() -> dict:
    """Die vorgerechneten Messreihen aus data/wpi_frozen.json."""
    return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))


def case_counts(faelle: dict) -> dict:
    """Zählungen über die 24 Fälle je Variante: Median, Spanne, kürzer als „keine“, Unterschwingen über 0,05, größtes Unterschwingen."""
    import statistics

    out = {}
    for key, _v, _k in CASE_VARIANTS:
        z = [faelle[c][key]["einschwing"] for c in C.CASE_IDS]
        ref = [faelle[c]["keine"]["einschwing"] for c in C.CASE_IDS]
        u = [faelle[c][key]["unter"] for c in C.CASE_IDS]
        out[key] = dict(median=statistics.median(z), min=min(z), max=max(z), kuerzer=sum(1 for a, b in zip(z, ref) if a < b),
                        unter_gross=sum(1 for x in u if x > 0.05), unter_max=max(u))
    return out


def limit_cycle_info(case: dict, variant: str = "bckp", periods_total: int = 20000, last: int = 2000) -> dict:
    """Ein nicht eingeschwungener Zufallsfall (Eintrag aus `ausnahmen`) mit sehr langem Lauf: kleinstes und größtes y in den letzten `last` Perioden, Anteil gesättigter Perioden dort,
    kleinste Periodenlänge p ≤ 600, mit der sich y in den letzten Perioden auf 1e-9 wiederholt (None: keine gefunden)."""
    plant = Plant(case["kind"], tuple(case["par"]), case["gain"])
    var, kaw = {"bckp": ("bc", case["ki"] / case["kp"]), "bed": ("bed", 0.0), "ilim": ("ilim", 0.0), "keine": ("keine", 0.0)}[variant]
    kz, bahn = simulate(plant, case["kp"], case["ki"], var, kaw, case["t1"], case["r1"], case["r2"], horizon=periods_total, still=None, record=True)
    ys = bahn.y[-last:]
    vs = bahn.v[-last:]
    period = None
    for p in range(1, 601):
        if all(abs(ys[i] - ys[i + p]) < 1e-9 for i in range(len(ys) - p)):
            period = p
            break
    return dict(y_min=min(ys), y_max=max(ys), sat_share=sum(1 for v in vs if abs(v) > D.UMAX) / len(vs), period=period, periods_total=case["t1"] + periods_total)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Verdikte (reine Funktionen: (Art, Text))
# ---------------------------------------------------------------------------------------------------------------------------------------------

def verdict_windup(a: Analysis) -> tuple[str, str]:
    """Kernfrage 1: Wie lange hält die Grenze den Regler fest? Ohne Anti-Windup gegen Rückrechnung mit kaw = ki/kp, beide am gewählten Szenario."""
    sc = a.settings.scenario
    k0 = a.keine.kz
    kb = a.row("bc_kp").kz
    if not sc.reachable:
        return "warning", (f"Der zweite Sollwert r₂ = {de(sc.r2)} liegt nicht unter der größten erreichbaren Ausgabe G·umax = {de(sc.gain * D.UMAX)}: "
                           "Er bleibt auch nach dem Sprung unerreichbar, die Stellgröße bleibt an der Grenze, es gibt keinen Ausstieg und keine Einschwingzeit.")
    if k0.divergiert:
        return "warning", "Ohne Anti-Windup divergiert der Integrator in diesem Szenario; die Kennzahlen entfallen."
    if k0.ausstieg is None:
        return "warning", f"Ohne Anti-Windup verlässt die Stellgröße die Grenze innerhalb von {fmt_int(D.HORIZON)} Perioden nach dem Sprung nicht."
    if k0.ausstieg == 0:
        return "info", (f"Hier gibt es kein Windup: Schon in der ersten Periode nach dem Sprung gilt |v| ≤ {D.UMAX}, die Stellgröße verlässt die Grenze sofort "
                        f"(Einschwingzeit {periods(k0.einschwing)}). Der Integrator hat in {sc.t1} Perioden Sättigung nicht genug aufgezogen; mehr Perioden bei r₁ oder ein größerer Sollwert r₁ zeigen den Effekt.")
    if sc.t1 == 0:
        text = (f"Ohne Anti-Windup bleibt die Stellgröße im Anlauf (T₁ = 0, y beginnt bei 0) **{periods(k0.ausstieg)}** an der Grenze (Einschwingzeit {periods(k0.einschwing)}), "
                f"obwohl der Sollwert erreichbar ist.")
    else:
        text = (f"Ohne Anti-Windup bleibt die Stellgröße nach dem Sprung noch **{periods(k0.ausstieg)}** an der Grenze (Einschwingzeit {periods(k0.einschwing)}), "
                f"obwohl der Sollwert ab dem Sprung erreichbar ist.")
    if kb.divergiert:
        text += " Mit Rückrechnung (kaw = ki/kp) divergiert der Lauf."
    elif kb.einschwing is not None:
        text += f" Mit Rückrechnung und kaw = ki/kp = {de(a.row('bc_kp').kaw)}: {periods(kb.einschwing)}."
    return "warning", text


def verdict_cost(a: Analysis, counts: dict | None = None) -> tuple[str, str]:
    """Kernfrage 2: Was kostet das Gegenmittel? Aktuelles Szenario: Unterschwingen je Variante; dazu die Zählung über die 24 Fälle der Vormessung (`counts` aus `case_counts`)."""
    sc = a.settings.scenario
    parts = []
    for r in a.rows:
        if r.key == "keine":
            continue
        if r.kz.divergiert:
            parts.append(f"{r.name}: divergiert")
        elif undershoot_applies(r.kz, sc.r2):
            parts.append(f"{r.name}: {de(r.kz.unter)}")
    if parts:
        text = "Unterschwingen unter den Sollwert r₂ im gewählten Szenario: " + "; ".join(parts) + ". "
    else:
        text = "Im gewählten Szenario kommt y beim Sprung nicht von oben auf r₂ zu; ein Unterschwingen im Sinne der Messung gibt es nicht. "
    if counts:
        text += (f"In den 24 Fällen der Vormessung unterschwingt die Rückrechnung mit kaw = 0,5 in {counts['bc05']['unter_gross']} von 24 um mehr als 0,05 "
                 f"(größtes {de(counts['bc05']['unter_max'])}), die bedingte Integration in {counts['bed']['unter_gross']} von 24, die Rückrechnung mit kaw = ki/kp in "
                 f"{counts['bckp']['unter_gross']} von 24 (größtes {de(counts['bckp']['unter_max'])}).")
    return "info", text


def verdict_notwhen(a: Analysis, random_summary: dict | None = None) -> tuple[str, str]:
    """Kernfrage 3: Wann hilft das Gegenmittel nicht? Aktuelles Szenario (längere Einschwingzeit als „keine“, Divergenz) und die Zählung über die Zufallsfälle."""
    k0 = a.keine.kz
    worse = []
    for r in a.rows[1:]:
        if r.kz.divergiert:
            worse.append(f"{r.name} (divergiert)")
        elif k0.einschwing is not None and not k0.divergiert and r.kz.einschwing is not None and r.kz.einschwing > k0.einschwing:
            worse.append(f"{r.name} ({periods(r.kz.einschwing)} gegen {periods(k0.einschwing)})")
    kind = "warning" if worse else "info"
    text = ("Hier ist mindestens ein Gegenmittel länger als gar keins: " + "; ".join(worse) + ". ") if worse else "Im gewählten Szenario ist jedes Gegenmittel höchstens so lang wie „Keine“ (oder die Läufe sind nicht vergleichbar). "
    if random_summary:
        v = random_summary["vergleich"]["bckp"]
        n = random_summary["n"]
        text += (f"In {fmt_int(n)} Zufallsfällen der Vormessung ist die Rückrechnung mit kaw = ki/kp in {fmt_int(v['kuerzer'])} kürzer als „Keine“, in {v['gleich']} gleich und in {v['laenger']} länger; "
                 f"{len(random_summary['nicht_eingeschwungen'])} davon schwingen in {fmt_int(D.HORIZON)} Perioden nicht ein.")
    return kind, text


def verdict_exit(rows: list[dict], sc: Scenario, slope: float | None) -> tuple[str, str]:
    """Experiment 2: Ausstieg wächst linear mit T1; Steigung gemessen (T1 = 120 bis 240) gegen die Erwartung (r1 − G·umax)/(G·umax − r2)."""
    if not sc.reachable:
        return "warning", f"Der Sollwert r₂ = {de(sc.r2)} bleibt unerreichbar (G·umax = {de(sc.gain * D.UMAX)}): Es gibt keinen Ausstieg, die Messung entfällt."
    if sc.r1 <= sc.gain * D.UMAX:
        return "info", (f"Der erste Sollwert r₁ = {de(sc.r1)} liegt nicht über G·umax = {de(sc.gain * D.UMAX)}: Er ist erreichbar, der Zuwachs (r₁ − G)/(G − r₂) = {de((sc.r1 - sc.gain * D.UMAX) / (sc.gain * D.UMAX - sc.r2), 1)} "
                        "ist nicht positiv, die Formel gilt hier nicht. Die Tabelle zeigt die gemessenen Werte; für die Linie braucht r₁ einen unerreichbaren Wert (über G).")
    last = rows[-1]
    gemessen = measured_slope(rows)
    if gemessen is None or slope is None:
        return "info", "Für die Steigung fehlt eine Messung bei T₁ = 120 oder 240 (kein Ausstieg im Horizont oder kein Windup)."
    first60 = next((r for r in rows if r["t1"] == 60), None)
    text = (f"Der Ausstieg wächst um **{de(gemessen, 1)} Perioden** je gesättigter Periode (gemessen zwischen T₁ = 120 und 240; erwartet (r₁ − G)/(G − r₂) = {de(slope, 1)}). "
            f"Bei T₁ = {last['t1']} sind es {fmt_int(last['ausstieg'])} Perioden.")
    if first60 is not None and first60["bc"] is not None and last["bc"] is not None:
        text += f" Die Rückrechnung mit kaw = ki/kp braucht bei T₁ = 60 {first60['bc']}, bei T₁ = {last['t1']} {last['bc']} Perioden."
    return "info", text


def verdict_cases(counts: dict) -> tuple[str, str]:
    """Experiment 3: die 24 Fälle in Zählungen."""
    k, b, c, d, i = counts["keine"], counts["bckp"], counts["bed"], counts["bc05"], counts["ilim"]
    return "info", (f"„Keine“ braucht {fmt_int(k['min'])} bis {fmt_int(k['max'])} Perioden (Median {fmt_int(round(k['median']))}); Rückrechnung mit kaw = ki/kp {b['min']} bis {b['max']} "
                    f"(Median {de(b['median'], 1)}), kürzer als „Keine“ in {b['kuerzer']} von 24 und mit Unterschwingen über 0,05 in {b['unter_gross']} von 24. "
                    f"Die bedingte Integration unterschwingt in {c['unter_gross']} von 24 um mehr als 0,05 (größtes {de(c['unter_max'])}), die Rückrechnung mit kaw = 0,5 in {d['unter_gross']} von 24 (größtes {de(d['unter_max'])}); "
                    f"die Integratorgrenze braucht im Median {de(i['median'], 0)} Perioden bei {i['unter_gross']} von 24 mit Unterschwingen über 0,05.")


def verdict_kaw(rows: list[dict], kaw: float, ratio: float) -> tuple[str, str]:
    """Experiment 4: Einschwingzeit und Unterschwingen bei kaw = ki/kp, beim gewählten Wert und an den Rändern des Gitters."""
    def pick(k):
        return min(rows, key=lambda r: abs(r["kaw"] - k))

    def desc(r):
        if r["divergiert"]:
            return f"kaw = {de(r['kaw'])}: divergiert"
        s = f"kaw = {de(r['kaw'])}: {periods(r['einschwing'])}"
        if r["unter"] > 0.05 and (r["y_sprung"] or 0) > 0:
            s += f" (Unterschwingen {de(r['unter'])})"
        return s

    parts = [desc(pick(ratio)).replace("kaw = ", "kaw = ki/kp = ", 1)]
    if abs(kaw - ratio) > 1e-9:
        parts.append(desc(pick(kaw)))
    lo, hi = min(rows, key=lambda r: r["kaw"]), max(rows, key=lambda r: r["kaw"])
    parts += [desc(lo), desc(hi)]
    seen, uniq = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return "info", "In diesem Szenario: " + "; ".join(uniq) + "."


def verdict_noise(rows: list[dict], r: float, amp: float) -> tuple[str, str]:
    """Experiment 5: Mittel von y gegen den Sollwert; ohne Rückrechnung gleich Soll, mit Rückrechnung verschoben."""
    keine = rows[0]
    bc = next(x for x in rows if x["variant"] == "bc" and abs(x["kaw"] - 0.5) < 1e-9)
    if keine["divergiert"] or bc["divergiert"]:
        return "warning", "Mindestens eine der verglichenen Varianten divergiert mit diesen Reglerwerten (instabile Schleife): Das Mittel von y ist dort nicht definiert."
    if keine["sat_share"] < 0.01:
        return "info", (f"Bei Soll {de(r)} und Rauschen ±{de(amp)} sättigt der Regler kaum ({de(100 * keine['sat_share'], 1)} % der Perioden): Die Mittel aller Varianten liegen beieinander "
                        f"(Keine {de(keine['mean_y'], 3)}, kaw = 0,5 {de(bc['mean_y'], 3)}); die Grenze greift nicht, also verschiebt auch das Gegenmittel nichts.")
    return "warning", (f"Ohne Anti-Windup ist das Mittel von y {de(keine['mean_y'], 3)} (Soll {de(r)}); mit Rückrechnung und kaw = 0,5 {de(bc['mean_y'], 3)}. "
                       f"Das Gegenmittel verschiebt das Mittel: Nach der exakten Identität ist das Mittel des Fehlers mit Rückrechnung kaw/ki mal dem mittleren Abschneiden v − u, und das ist nicht null, "
                       f"sobald die Grenze greift (Anteil gesättigter Perioden ohne Anti-Windup: {de(100 * keine['sat_share'], 0)} %).")


def verdict_random(summary: dict) -> tuple[str, str]:
    """Experiment 6: Zählung über die Zufallsfälle, Ausnahmen und nicht eingeschwungene Fälle."""
    v = summary["vergleich"]["bckp"]
    n = summary["n"]
    osc = summary["nicht_eingeschwungen_arten"].get("osc", 0)
    return "info", (f"Rückrechnung mit kaw = ki/kp: kürzer als „Keine“ in {fmt_int(v['kuerzer'])} von {fmt_int(n)} Fällen, gleich in {v['gleich']}, länger in {v['laenger']}. "
                    f"{len(summary['nicht_eingeschwungen'])} Fälle schwingen in {fmt_int(D.HORIZON)} Perioden nicht ein, davon {osc} mit schwingender Strecke.")
