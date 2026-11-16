"""Windup – wenn die Stellgrenze den Integrator überlistet - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein PI-Regler mit Stellgrenze: Der Integrator erfährt von der Grenze nichts und wächst weiter (Windup). Gemessen werden die Dauer, drei Gegenmittel (Rückrechnung, bedingte
Integration, Integratorgrenze), ihre Kosten und ihre Ausnahmen; alles eigene Messung in diesem Modell, bei diesen Parametern (Kern: wpi_dynamics.py).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import wpi_constants as C
import wpi_dynamics as D
import wpi_evaluation as ev
from wpi_evaluation import Scenario, Settings, de, fmt_int
from wpi_layout import dunkles_design, kopf_css
from wpi_presets import (
    bounds,
    current_settings,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)
from wpi_visualization import build_bahn, build_cases, build_exit, build_kaw, build_noise

st.set_page_config(page_title="Windup – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne die Regelkreise …")
def _analysis(settings: Settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Rechne Ausstieg und Vorhersage für zehn Längen der Sättigung …")
def _exit_sweep(sc: Scenario):
    return ev.exit_sweep(sc)


@st.cache_data(show_spinner=False)
def _kaw_curve(sc: Scenario, kaw: float):
    return ev.kaw_curve(sc, kaw)


@st.cache_data(show_spinner=False)
def _jump_check(sc: Scenario, kaw: float):
    return ev.jump_check(sc, kaw)


@st.cache_data(show_spinner="Rechne 22 000 Perioden Messrauschen für sieben Varianten …")
def _noise(sc: Scenario, r: float, amp: float, seed: int):
    return ev.noise_table(sc, r, amp, seed)


@st.cache_data(show_spinner=False)
def _hand(variant: str):
    return ev.hand_example(variant)


@st.cache_data(show_spinner=False)
def _frozen():
    return ev.frozen()


@st.cache_data(show_spinner="Rechne den Fall …")
def _exception_run(case: dict, variant_key: str, t1_plot: int):
    plant = D.Plant(case["kind"], tuple(case["par"]), case["gain"])
    var, kaw = {"bckp": ("bc", case["ki"] / case["kp"]), "bed": ("bed", 0.0), "ilim": ("ilim", 0.0)}[variant_key]
    horizon = t1_plot
    _, b_var = D.simulate(plant, case["kp"], case["ki"], var, kaw, case["t1"], case["r1"], case["r2"], horizon=horizon, still=None, record=True)
    _, b_keine = D.simulate(plant, case["kp"], case["ki"], "keine", 0.0, case["t1"], case["r1"], case["r2"], horizon=horizon, still=None, record=True)
    return b_var, b_keine


@st.cache_data(show_spinner="Rechne 20 000 Perioden …")
def _cycle(case: dict, variant_key: str):
    return ev.limit_cycle_info(case, variant_key)


def _table_cell(x, nd=3):
    return de(float(x), nd)


def _kz_text(n):
    return "nie" if n is None else str(n)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in wpi_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎚️ Windup – wenn die Stellgrenze den Integrator überlistet")
st.markdown(
    """
Ein **PI-Regler** rechnet linear; die **Stellgrenze** schneidet nur hinter dem Regler ab. Der **Integrator** erfährt davon nichts: Ist der Sollwert eine Zeit lang unerreichbar, wächst er weiter
und hält die Stellgröße noch lange an der Grenze, nachdem der Sollwert längst erreichbar ist (**Windup**). Die Demo misst, wie lange das dauert, was die drei gängigen Gegenmittel
(**Rückrechnung**, **bedingte Integration**, **Integratorgrenze**) kosten und wann sie selbst Schaden anrichten. Alles ist eigene Messung in diesem Modell, bei diesen Parametern.

Weiter unten: die **❓ Kernfragen**, die Experimente 2 bis 6, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Windup", expanded=True):
    st.markdown(
        """
1. **Der Regler** bildet den Fehler e = r − y und den Ausgang v = kp·e + I. Die Stellgröße ist u = v, auf [−1, 1] abgeschnitten.
2. **Der Integrator** addiert je Periode ki·e. Ist v über der Grenze, wirkt u nicht mehr wie v, aber I wächst weiter, solange der Fehler positiv ist.
3. **Der Sprung:** Wird der Sollwert danach erreichbar, ist e negativ, aber I ist groß. v bleibt über 1, die Stellgröße bleibt an der Grenze, bis I abgebaut ist; je länger die Sättigung dauerte, desto länger dauert das.
4. **Die Gegenmittel** geben dem Integrator Kenntnis von der Grenze: **Rückrechnung** addiert kaw·(u − v), **bedingte Integration** hält I an, solange v gesättigt und der Fehler in Richtung der Sättigung gerichtet ist, die **Integratorgrenze** schneidet I auf [−1, 1] ab.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Strecke** (Abtastzeit 1)")
    kind = st.selectbox("Streckenart", C.KINDS, key="kind_select", format_func=C.KIND_NAME.get,
                        help="Erste Ordnung: y⁺ = a·y + G(1 − a)·u. Mit Totzeit: u wirkt d Perioden später. Zwei Verzögerungen: zwei Glieder erster Ordnung in Reihe. Schwach gedämpft: Pole mit Radius r und Winkel θ.")
    st.slider("Pol a (a₁; bei „schwach gedämpft“ der Polradius r)", *bounds("a1_slider"), step=0.01, key="a1_slider",
              help="Je näher an 1, desto träger die Strecke.")
    st.slider("Zweiter Pol a₂ (nur bei zwei Verzögerungen)", *bounds("a2_slider"), step=0.01, key="a2_slider", disabled=kind != "lag2",
              help="Wirkt nur bei der Streckenart „Zwei Verzögerungen in Reihe“.")
    st.slider("Totzeit d (Perioden; nur bei Totzeit)", *bounds("dead_slider"), step=1, key="dead_slider", disabled=kind != "tot",
              help="Wirkt nur bei der Streckenart „Erste Ordnung mit Totzeit“.")
    st.slider("Polwinkel θ (nur bei schwach gedämpft)", *bounds("theta_slider"), step=0.01, key="theta_slider", disabled=kind != "osc",
              help="Wirkt nur bei der schwach gedämpften Strecke: c₁ = 2·r·cos θ, c₀ = r².")
    st.slider("Verstärkung G der Strecke", *bounds("gain_slider"), step=0.1, key="gain_slider",
              help="Mit Stellgrenze 1 erreicht y höchstens G. Ein Sollwert über G ist unerreichbar.")
    st.markdown("**PI-Regler**")
    st.slider("kp", *bounds("kp_slider"), step=0.1, key="kp_slider", help="Proportionalverstärkung.")
    st.slider("ki", *bounds("ki_slider"), step=0.01, key="ki_slider", help="Integralverstärkung (je Periode).")
    st.markdown("**Sollwertfolge**")
    st.slider("Dauer T₁ des ersten Sollwerts (Perioden)", *bounds("t1_slider"), step=1, key="t1_slider",
              help="So viele Perioden gilt r₁. Bei 0 gibt es nur r₂ (Anlauf aus y = 0).")
    st.slider("Erster Sollwert r₁", *bounds("r1_slider"), step=0.1, key="r1_slider", help="Über G unerreichbar: der Regler sättigt.")
    st.slider("Zweiter Sollwert r₂", *bounds("r2_slider"), step=0.01, key="r2_slider", help="Danach gilt r₂; unter G erreichbar. Einschwingband: ±2,5 % von r₂.")

settings = current_settings()
sc = settings.scenario
sync_query_params({k: st.session_state[k] for k in (
    "kind_select", "a1_slider", "a2_slider", "dead_slider", "theta_slider", "gain_slider", "kp_slider", "ki_slider", "t1_slider", "r1_slider", "r2_slider",
    "variant_radio", "kaw_slider", "noise_amp_slider", "noise_r_slider", "noise_seed_input")})

# --- Experiment 1: Windup von Hand --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Experiment 1: Windup von Hand")
st.caption("Der Regelkreis aus der Seitenleiste: y, u, v und I über der Zeit. Voreinstellung ist das Handbeispiel (alle Zahlen als Brüche weiter unten).")
cv1, cv2 = st.columns([3, 2])
with cv1:
    variant = st.radio("Anti-Windup-Variante", ev.VARIANT_CHOICES, key="variant_radio", format_func=ev.VARIANT_CHOICE_NAME.get, horizontal=True,
                       help="Keine: I + ki·e. Rückrechnung: dazu kaw·(u − v). Bedingte Integration: I bleibt stehen, solange v gesättigt und e in Richtung der Sättigung gerichtet ist. Integratorgrenze: I auf [−1, 1] abgeschnitten.")
with cv2:
    st.slider("kaw (Rückrechnung, frei)", *bounds("kaw_slider"), step=0.01, key="kaw_slider", disabled=variant != "bc",
              help="Wirkt nur bei „Rückrechnung, kaw frei“. Die Wahl „kaw = ki/kp“ setzt den Wert selbst.")
settings = current_settings()
sc = settings.scenario
a = _analysis(settings)
sel, ref = a.selected.kz, a.keine.kz

if a.rho >= 1:
    st.warning(f"Die lineare Schleife (ohne Stellgrenze) ist mit diesen Reglerwerten nicht stabil (Spektralradius ρ = {de(a.rho, 3)}); die Kennzahlen unten gelten trotzdem für die Schleife mit Stellgrenze, sind aber nicht die Messreihe der Vormessung.")

st.caption("Einschwingzeit und Ausstieg zählen in Perioden ab dem Sprung auf r₂.")
m1, m2, m3, m4 = st.columns(4)


def _delta(x, r):
    if a.selected.key == "keine":
        return "Referenz", "off"
    if x is None or r is None or sel.divergiert or ref.divergiert:
        return None, "off"
    # Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
    return f"{x - r:+d} gegen „Keine“", "inverse"


d1, c1 = _delta(sel.einschwing, ref.einschwing)
d2, c2 = _delta(sel.ausstieg, ref.ausstieg)
m1.metric("Einschwingzeit", "divergiert" if sel.divergiert else _kz_text(sel.einschwing), delta=d1, delta_color=c1,
          help="Perioden ab dem Sprung bis zur letzten Ausfahrt aus dem Band ±2,5 % von r₂.")
m2.metric("Ausstieg", "divergiert" if sel.divergiert else _kz_text(sel.ausstieg), delta=d2, delta_color=c2,
          help="Perioden ab dem Sprung bis zur ersten Periode mit |v| ≤ 1, also bis die Stellgröße die Grenze verlässt.")
m3.metric("Unterschwingen", de(sel.unter) if (ev.undershoot_applies(sel, sc.r2) and not sel.divergiert) else "–", delta="unter r₂", delta_color="off",
          help="r₂ minus kleinstes y nach dem Sprung; nur dort gezeigt, wo y beim Sprung über r₂ liegt.")
m4.metric("Spektralradius ρ", de(a.rho, 3), delta="ohne Stellgrenze", delta_color="off", help="Größter Eigenwertbetrag der linearen Schleife aus Strecke und PI-Regler (ohne Stellgrenze); unter 1 stabil.")

bahnen = [(a.selected.name, a.bahn_selected, C.VARIANT_COLOR[a.selected.variant], False)]
if a.selected.key != "keine":
    bahnen.append(("Keine", a.bahn_keine, C.VARIANT_COLOR["keine"], True))
st.plotly_chart(build_bahn(bahnen, sc.t1, sc.r2), width="stretch", key="main_chart")
st.caption("Gestrichelte senkrechte Linie: der Sprung von r₁ auf r₂ (nur bei T₁ > 0); Band um r₂: ±2,5 % (das Einschwingband). Gezeichnet ist gut das 1,2-Fache der Einschwingzeit der gewählten Variante, der Gesamtlauf wird bis zum Einschwingen gerechnet.")

with st.expander("📋 Die ersten neun Perioden als Tabelle", expanded=sc.t1 <= 5):
    tab = ev.table_rows(a.bahn_selected, 9)
    st.dataframe(
        [{"k": t["k"], "r": _table_cell(t["r"]), "y": _table_cell(t["y"]), "e": _table_cell(t["e"]), "v": _table_cell(t["v"]), "u": _table_cell(t["u"]), "I (vor dem Update)": _table_cell(t["I"]),
          "gesättigt": "ja" if t["v"] != t["u"] else "–"} for t in tab],
        hide_index=True, width="stretch",
    )

with st.expander("🧮 Das Handbeispiel in Brüchen (feste Aufgabe)"):
    st.markdown(
        """
Strecke y⁺ = (y + u)/2, PI mit kp = 1 und ki = ½, Grenze |u| ≤ 1, Sollwert 3 in den Perioden 0 und 1, danach ¼, Anfangszustand y = 0 und I = 0. Die Zeilen zeigen die Werte am Anfang der Periode k
(I vor dem Update); der Kern rechnet hier mit Brüchen (`fractions`), nicht mit Gleitkommazahlen.
        """
    )
    for var, title in (("keine", "Ohne Anti-Windup"), ("bc", "Mit Rückrechnung, kaw = ki/kp = ½")):
        st.markdown(f"**{title}**")
        st.dataframe(
            [{"k": t["k"], "r": ev.fraction_text(t["r"]), "y": ev.fraction_text(t["y"]), "e": ev.fraction_text(t["e"]), "v": ev.fraction_text(t["v"]), "u": ev.fraction_text(t["u"]),
              "I": ev.fraction_text(t["I"])} for t in _hand(var)],
            hide_index=True, width="stretch",
        )
    exit_k = next(t["k"] for t in _hand("keine") if t["k"] >= 2 and abs(t["v"]) <= 1)
    st.markdown(
        f"Ohne Anti-Windup ist v in k = 2 noch 9/4 über der Grenze, obwohl der Fehler schon negativ ist: der Integrator fällt je Periode nur um ki·(y − ¼), rund ¼. "
        f"Erstmals gilt |v| ≤ 1 in k = {exit_k}, das sind {exit_k - 2} Perioden nach dem Sprung. Mit Rückrechnung und kaw = ki/kp = ½ ist ab k = 2 genau v = u = ¼ (die Rückrechnung setzt I so, dass v den neuen Sollwert trifft), "
        "und y fällt ohne Unterschwingen auf ¼."
    )

st.markdown("**Alle Varianten im gewählten Szenario**")
st.dataframe(
    [{"Variante": r.name, "kaw": de(r.kaw, 3) if r.variant == "bc" else "–", "Ausstieg": "divergiert" if r.kz.divergiert else _kz_text(r.kz.ausstieg),
      "Einschwingzeit": "divergiert" if r.kz.divergiert else _kz_text(r.kz.einschwing),
      "Unterschwingen": de(r.kz.unter) if (ev.undershoot_applies(r.kz, sc.r2) and not r.kz.divergiert) else "–"} for r in a.rows],
    hide_index=True, width="stretch",
)

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.markdown("## ❓ Kernfragen")
frozen = _frozen()
counts = ev.case_counts(frozen["faelle"])

st.markdown("**1. Wie lange hält die Grenze den Regler fest?**")
kind_, text = ev.verdict_windup(a)
getattr(st, kind_)(text)
st.markdown("**2. Was kostet das Gegenmittel?**")
kind_, text = ev.verdict_cost(a, counts)
getattr(st, kind_)(text)
st.markdown("**3. Wann hilft es nicht?**")
kind_, text = ev.verdict_notwhen(a, frozen["zufall"])
getattr(st, kind_)(text)

st.markdown("---")

# --- Experiment 2: Wie lange dauert es? --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie lange dauert es?")
st.caption("Strecke, Regler und Sollwerte aus der Seitenleiste; nur die Dauer T₁ der Sättigung wird variiert. Der Ausstieg ist gemessen und zugleich aus der offenen Stufenantwort der Strecke vorhergesagt (anderer Rechenweg, ohne Regler-Simulation).")
if sc.reachable:
    rows2 = _exit_sweep(sc)
    st.plotly_chart(build_exit(rows2), width="stretch", key="exit_chart")
    st.dataframe(
        [{"T₁": r["t1"], "Ausstieg gemessen": _kz_text(r["ausstieg"]), "vorhergesagt": _kz_text(r["vorhergesagt"]),
          "Näherung": "–" if r["naeherung"] is None else de(r["naeherung"], 1), "Einschwingzeit Keine": _kz_text(r["einschwing"]),
          "Rückrechnung kaw = ki/kp": _kz_text(r["bc"]), "Bedingte Integration": _kz_text(r["bed"])} for r in rows2],
        hide_index=True, width="stretch",
    )
    kind_, text = ev.verdict_exit(rows2, sc, D.exit_slope(sc.plant(), sc.r1, sc.r2))
else:
    kind_, text = ev.verdict_exit([], sc, None)
getattr(st, kind_)(text)

st.markdown("---")

# --- Experiment 3: 24 Fälle (vorgerechnet) --------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Drei Streckenarten, 24 Parametersätze (vorgerechnet)")
st.caption("Vorgerechnet mit `tools/make_frozen.py` (data/wpi_frozen.json): Szenario r₁ = 4,7 für 57 Perioden, dann r₂ = 0,77. Acht Parametersätze je Streckenart (erste Ordnung, zwei Verzögerungen und eine schwach gedämpfte Strecke, Totzeit), jeder mit linearer Schleife stabil.")
case_id = st.selectbox("Fall", C.CASE_IDS, key="case_select", format_func=C.case_label)
st.plotly_chart(build_cases(frozen["faelle"][case_id]), width="stretch", key="cases_chart")
st.dataframe(
    [{"Fall": C.case_label(cid), "ρ": de(frozen["faelle"][cid]["rho"], 3), **{ev.CASE_VARIANT_NAME[k]: f"{frozen['faelle'][cid][k]['einschwing']} ({de(frozen['faelle'][cid][k]['unter'])})" for k, _v, _kk in ev.CASE_VARIANTS}}
     for cid in C.CASE_IDS],
    hide_index=True, width="stretch",
)
st.caption("Tabelle: Einschwingzeit in Perioden, in Klammern das Unterschwingen unter r₂.")
kind_, text = ev.verdict_cases(counts)
getattr(st, kind_)(text)

st.markdown("---")

# --- Experiment 4: kaw wählen -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: kaw wählen")
st.caption("Rückrechnung im Szenario der Seitenleiste: Einschwingzeit und Unterschwingen über kaw. In der Sättigung ist die Rekursion von I nur für 0 < kaw < 2 stabil (Faktor |1 − kaw|).")
kaw4 = st.slider("kaw (Experiment)", C.KAW4_MIN, C.KAW4_MAX, 0.5, step=0.01, key="kaw4_slider", help="Dieser Regler wirkt nur in diesem Experiment (nicht auf die Auswahl oben).")
ratio = sc.ki / sc.kp
rows4 = _kaw_curve(sc, float(kaw4))
st.plotly_chart(build_kaw(rows4, ratio, float(kaw4)), width="stretch", key="kaw_chart")
jc = _jump_check(sc, float(kaw4))
if jc is not None:
    st.markdown(
        f"**v unmittelbar nach dem Sprung** bei kaw = {de(kaw4)}: gemessen {de(jc['v_gemessen'], 3)}, nach der Formel v = 1 − kp·Δ + (ki/kaw − kp)·(r₁ − y) = {de(jc['v_formel'], 3)} "
        f"(y = {de(jc['y_sat'], 4)} beim Sprung; Näherung für konstanten Fehler in der Sättigung). Faktor der Rekursion in der Sättigung |1 − kaw| = {de(jc['faktor'], 2)}"
        + (" – **nicht kleiner als 1: die Rekursion divergiert**." if jc["faktor"] >= 1 else ".")
    )
    st.caption("Liegt v nach dem Sprung weit unter −1 (bei kaw > ki/kp), passt das zum Unterschwingen in diesem Fall (nur Formel und Beobachtung, kein Gegenversuch); bei kaw = ki/kp fällt der Term mit dem Fehler weg. Die Formel gilt nur, solange v in der Sättigung bleibt.")
kind_, text = ev.verdict_kaw(rows4, float(kaw4), ratio)
getattr(st, kind_)(text)

st.markdown("---")

# --- Experiment 5: Rauschen -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Messrauschen nahe der Grenze")
st.caption("Konstanter Sollwert r, gleichverteiltes Messrauschen ±Amplitude (SplitMix64) wird zum gemessenen y addiert; 22 000 Perioden, die ersten 2 000 zählen nicht. Strecke und Regler aus der Seitenleiste; der Sollwert r nahe der Stellgrenze 1 ist die Voreinstellung.")
cn0, cn1, cn2 = st.columns(3)
with cn0:
    noise_r = st.slider("Sollwert r im Rauschexperiment", *bounds("noise_r_slider"), step=0.05, key="noise_r_slider", help="Je näher an der Stellgrenze 1 (mal G), desto öfter sättigt der Regler im ruhigen Betrieb.")
with cn1:
    amp = st.slider("Rauschamplitude ±", *bounds("noise_amp_slider"), step=0.05, key="noise_amp_slider", help="Gleichverteilt in [−Amplitude, +Amplitude], auf das gemessene y addiert.")
with cn2:
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("noise_seed_input"), key="noise_seed_input", step=1)
rows5 = _noise(sc, float(noise_r), float(amp), int(seed))
if any(x["divergiert"] for x in rows5):
    st.warning("Mindestens eine Variante divergiert mit diesen Reglerwerten (instabile Schleife); die Tabelle zeigt „divergiert“.")
else:
    st.plotly_chart(build_noise(rows5, float(noise_r)), width="stretch", key="noise_chart")
st.dataframe(
    [{"Variante": x["name"], "Mittel von y": "divergiert" if x["divergiert"] else de(x["mean_y"], 3), "Soll": de(float(noise_r)),
      "Anteil gesättigter Perioden": "divergiert" if x["divergiert"] else de(100 * x["sat_share"], 1) + " %",
      "Mittel von e": "divergiert" if x["divergiert"] else de(x["mean_e"], 4, sign=True)} for x in rows5],
    hide_index=True, width="stretch",
)
bc_row = next((x for x in rows5 if x["variant"] == "bc" and not x["divergiert"] and abs(x["kaw"] - 0.5) < 1e-9), None)
if bc_row is not None and "identity_left" in bc_row:
    st.markdown(
        f"**Exakte Identität** (Summen desselben Laufs, Rückrechnung kaw = 0,5): Σe = (kaw/ki)·Σ(v − u) + (I_Ende − I_Start)/ki, links {de(bc_row['identity_left'], 4)}, rechts {de(bc_row['identity_right'], 4)}. "
        "Ohne Rückrechnung ist das Mittel des Fehlers damit Null; mit Rückrechnung gleich kaw/ki mal dem mittleren Abschneiden."
    )
kind_, text = ev.verdict_noise(rows5, float(noise_r), float(amp))
getattr(st, kind_)(text)

st.markdown("---")

# --- Experiment 6: Wann hilft es nicht? -------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 6: Wann hilft es nicht? (3 000 Zufallsfälle, vorgerechnet)")
zu = frozen["zufall"]
st.caption(f"Vorgerechnet (Zufalls-Seed {zu['seed']}): zufällige Strecke aus den vier Arten, PI-Satz mit ρ < 0,97, T₁ = 5 bis 149, r₁ = 1,5 bis 6·G, r₂ = 0,2 bis 0,9·G. Hier die Fälle, in denen mindestens ein Gegenmittel länger braucht als „Keine“.")
vz = zu["vergleich"]
st.dataframe(
    [{"Gegenmittel": name, "kürzer als „Keine“": fmt_int(vz[k]["kuerzer"]), "gleich": vz[k]["gleich"], "länger": vz[k]["laenger"]}
     for k, name in (("bckp", "Rückrechnung, kaw = ki/kp"), ("bed", "Bedingte Integration"), ("ilim", "Integratorgrenze"))],
    hide_index=True, width="stretch",
)
art_name = {"lag1": "erste Ordnung", "tot": "Totzeit", "lag2": "zwei Verzögerungen", "osc": "schwingend"}
exc = zu["ausnahmen"]
names_k = {"bckp": "Rückrechnung", "bed": "Bedingte Integration", "ilim": "Integratorgrenze"}
labels = {e["index"]: f"Fall {e['index']}: {art_name[e['art']]}, T₁ = {e['t1']}, länger: {', '.join(names_k[k] for k in e['laenger'])}" for e in exc}
idx = st.selectbox("Ausnahme laden", list(labels), key="exc_select", format_func=labels.get)
case = next(e for e in exc if e["index"] == idx)
choices = [k for k in ("bckp", "bed", "ilim") if k in case["laenger"]]
vkey = st.radio("Gegenmittel", choices, key=f"exc_variant_select_{idx}", format_func=names_k.get, horizontal=True)
nc = case["einschwing"]
t_plot = 400 if (nc[vkey] or 0) >= D.HORIZON else min(max(int(1.2 * max(nc[vkey], nc["keine"])) + 20, 60), 4500)
b_var, b_keine = _exception_run(case, vkey, t_plot)
st.plotly_chart(build_bahn([(names_k[vkey], b_var, C.VARIANT_COLOR["bc" if vkey == "bckp" else vkey], False), ("Keine", b_keine, C.VARIANT_COLOR["keine"], True)],
                           case["t1"], case["r2"], with_integrator=False, height=520), width="stretch", key="exc_chart")
st.markdown(
    f"Einschwingzeit: **{names_k[vkey]} {nc[vkey]} Perioden**, „Keine“ {nc['keine']} Perioden (Unterschwingen {de(case['unter'][vkey])} bzw. {de(case['unter']['keine'])}). "
    f"Fall: {art_name[case['art']]}, Verstärkung G = {de(case['gain'])}, kp = {de(case['kp'])}, ki = {de(case['ki'], 3)}, ρ = {de(case['rho'], 3)}, T₁ = {case['t1']}, r₁ = {de(case['r1'])}, r₂ = {de(case['r2'])}."
)
if nc[vkey] >= D.HORIZON:
    cyc = _cycle(case, vkey)
    st.warning(
        f"Dieser Lauf schwingt in {fmt_int(D.HORIZON)} Perioden nicht ein. In den letzten 2 000 von {fmt_int(cyc['periods_total'])} Perioden liegt y zwischen {de(cyc['y_min'], 3)} und {de(cyc['y_max'], 3)} (Soll {de(case['r2'], 3)}), "
        f"{de(100 * cyc['sat_share'], 0)} % der Perioden sind gesättigt"
        + (f", und die Folge wiederholt sich exakt alle {cyc['period']} Perioden (Grenzzyklus)." if cyc["period"] else ", eine Periode bis 600 wurde nicht gefunden.")
    )
kind_, text = ev.verdict_random(zu)
getattr(st, kind_)(text)

st.markdown("---")

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Sollwert nach dem Sprung erreichbar** (r₂ < G) | Sonst bleibt die Stellgröße an der Grenze: kein Ausstieg, keine Einschwingzeit (die App meldet „nie“). | – |
| **PI-Regler, symmetrische Grenze, stabile Strecke** (drei Streckenarten) | Instabile Strecken, Strecken mit Nullstelle rechts, PID, Rate-Limit, asymmetrische Grenzen, Mehrgrößenregelung: nicht gemessen. | – |
| **Sollwertsprung statt Störung** | Ein Lastsprung am Eingang mit anschließendem Wegfall ist nicht gemessen. | – |
| **kaw = ki/kp** | Eigene Messung in einem Szenario (lange Sättigung, dann erreichbarer Sollwert); die Quelle für diese Wahl ist nicht gelesen. Bei kurzer Sättigung oder Sollwertfolge nicht abgestimmt. | – |
| **Rückrechnung aus der Sättigungsfunktion** | Rückrechnung aus dem gemessenen Stellglied statt aus der Sättigungsfunktion ist nicht gemessen. | – |
| **Rauschen:** eine Strecke, ein Sollwert, ein Zufalls-Seed, gleichverteilt | Andere Verteilungen, andere Strecken und Sollwerte nicht gemessen. | – |
| **Grenzzyklen der schwingenden Strecke** | Gemessen, nicht erklärt; ob „Keine“ bei anderem Anfangszustand denselben Zyklus hätte, ist nicht geprüft. | – |
| **Einschwingband ±2,5 % von r₂** | Andere Bänder nicht gemessen; die Zeit hängt am Band. | – |
| **Stellgrenze nur hinter dem Regler** | Steht die Grenze im Entwurf (MPC als QP), braucht es keinen Anti-Windup; ein Kostenvergleich PI mit Anti-Windup gegen MPC ist nicht gemessen. | [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo) |
| **Integrator als Mittel gegen Dauerfehler** | Hier seine Kehrseite an der Grenze; die Voraussetzung „Grenzen im Beharrungszustand inaktiv“ verbindet beide. | [`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind eine stabile Strecke mit Gleichstromverstärkung $G$, ein PI-Regler mit Stellgrenze $|u| \le 1$ und eine Sollwertfolge ($r_1$ für $T_1$ Perioden, danach $r_2 < G$).
Gesucht sind die Perioden bis zum Ausstieg aus der Sättigung und bis zum Einschwingen, je Anti-Windup-Variante.

| Symbol | Bedeutung |
|---|---|
| $y_k, r_k$ | Ausgang und Sollwert am Anfang der Periode $k$ |
| $e_k = r_k - y_k$ | Fehler |
| $I_k$ | Integratorzustand vor dem Update |
| $v_k$, $u_k$ | Reglerausgang vor und nach der Grenze, $u_k = \operatorname{sat}(v_k)$ |
| $k_p$, $k_i$, $k_{aw}$ | Proportional-, Integral- und Rückrechnungsverstärkung |
| $a$, $b = G(1-a)$ | Pol und Eingangsverstärkung der Strecke erster Ordnung |
| $s_k$ | offene Stufenantwort der Strecke auf $u = 1$ ab Nullzustand |
| $T_1$ | Dauer des ersten Sollwerts $r_1$ in Perioden |

**Strecke (erster Ordnung) und Regler.**
$$y_{k+1} = a\,y_k + G(1-a)\,u_k$$
$$e_k = r_k - y_k, \qquad v_k = k_p e_k + I_k, \qquad u_k = \operatorname{sat}(v_k)$$
$$I_{k+1} = I_k + k_i e_k + (\text{Anti-Windup-Term})$$

**Die vier Varianten** (eigene Festlegungen in diesem Modell; die Namen folgen der Literatur):
$$\text{keine: } I_{k+1} = I_k + k_i e_k$$
$$\text{Rückrechnung: } I_{k+1} = I_k + k_i e_k + k_{aw}(u_k - v_k)$$
$$\text{bedingte Integration: } I_{k+1} = \begin{cases} I_k & \text{falls } v_k \ne u_k \text{ und } e_k (v_k - u_k) > 0 \\ I_k + k_i e_k & \text{sonst} \end{cases}$$
$$\text{Integratorgrenze: } I_{k+1} = \operatorname{clip}(I_k + k_i e_k,\, -1,\, 1)$$

**Ausstieg aus der Sättigung ohne Regler-Simulation.** Solange $u = 1$ gilt, ist $y_k = s_k$, und der Integrator folgt aus Summen:
$$I_{T_1} = k_i \sum_{k<T_1} (r_1 - s_k), \qquad I_{j+1} = I_j - k_i (s_j - r_2) \quad (j \ge T_1)$$
$$N_{\text{Ausstieg}} = \min\{\, j \ge T_1 : k_p (r_2 - s_j) + I_j \le 1 \,\} - T_1$$
Näherung für lange Sättigung mit $S = \sum_{k<T_1}(r_1 - s_k)$ und $\Delta = G - r_2$, und der Zuwachs je gesättigter Periode:
$$N_{\text{Ausstieg}} \approx \frac{S}{\Delta} - \frac{1 + k_p \Delta}{k_i \Delta}, \qquad \frac{dN}{dT_1} \approx \frac{r_1 - G}{G - r_2}$$

**Rückrechnung in der Sättigung** (solange $v$ gesättigt bleibt, also $u = 1$):
$$I_{k+1} = (1 - k_{aw})\, I_k + (k_i - k_{aw} k_p)\, e_k + k_{aw}, \qquad I^\ast = 1 + \left(\frac{k_i}{k_{aw}} - k_p\right) e$$
Die Rekursion ist für $0 < k_{aw} < 2$ stabil (Faktor $|1 - k_{aw}|$). Bei $k_{aw} > k_i/k_p$ liegt $v$ nach dem Sprung tiefer als bei $k_{aw} = k_i/k_p$, in Fall a1 ab $k_{aw} = 0{,}2$ weit unter $-1$; bei $k_{aw} = k_i/k_p$ fällt der Term mit $e$ weg.

**Messrauschen.** Die Summe der Updates ergibt die exakte Identität
$$\sum_k e_k = \frac{k_{aw}}{k_i} \sum_k (v_k - u_k) + \frac{I_{\text{Ende}} - I_{\text{Start}}}{k_i}$$
Ohne Rückrechnung ($k_{aw} = 0$) ist das Mittel des Fehlers damit Null, sobald $I$ beschränkt bleibt; mit Rückrechnung ist es gleich $k_{aw}/k_i$ mal dem mittleren Abschneiden.

**Lineare Schleife** (ohne Stellgrenze, erste Ordnung ohne Totzeit): charakteristisches Polynom
$$z^2 - (1 + a - b k_p)\, z + (a - b k_p + b k_i) = 0$$
Der Spektralradius ist der größte Betrag der Nullstellen (allgemein: der größte Eigenwert der Zustandsmatrix aus Strecke und Integrator).

**Literatur und Quellen.**
- Åström/Murray, *Feedback Systems*: Integrator-Windup und Rückrechnung (Abschnitt 11.4 der 2. Auflage, Kapitel-PDF von 2019; die 1. Auflage, Fassung Februar 2009, führt den Abschnitt als 10.4). Die Quellenberichte nennen beide Nummern, nicht gegen die gedruckte Ausgabe geprüft. Gelesen ist der Fließtext: Bei gesättigtem Stellglied ist der Regelkreis aufgetrennt, der Integralterm läuft weiter hoch; Gegenmittel ist die Rückführung des Sättigungsfehlers über $k_{aw}$ in den Integrator; ein Vielfaches von $k_i$ sei eine vernünftige Wahl; ein zu großes $k_{aw}$ lasse Messrauschen den Integrator zurücksetzen. Zahlen und Abbildungsparameter des Kapitels sind nicht übernommen (die Angaben zu $k_i$ widersprechen sich im Kapitel). Die Bedingung $k_{aw} > k_i/k_p$ aus Übung 11.13 gilt in stetiger Zeit (Kreiskriterium) und ist hier nicht nachgerechnet; die gemessene Grenze $k_{aw} = k_i/k_p$ für das Unterschwingen in diesem diskreten Modell ist ein anderer Befund. „Bedingte Integration“ steht dort als Name in Übung 11.12; die Regel oben ist eigene Festlegung.
- Rawlings/Mayne/Diehl, *Model Predictive Control* (Abschnitt 1.5.2, S. 49 bis 53): Offsetfreiheit mit integrierendem Störungsmodell gilt nur, wenn die Beschränkungen im Beharrungszustand inaktiv sind (in eigenen Worten wiedergegeben).
- Alle Formeln und Befunde dieser Seite sind eigene Herleitung und Messung in diesem Modell, bei diesen Parametern; Windup, Rückrechnung und bedingte Integration sind Lehrbuchstoff, eine Neuheit wird nicht behauptet.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
