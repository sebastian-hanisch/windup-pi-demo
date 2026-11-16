# Windup – wenn die Stellgrenze den Integrator überlistet – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-windup-pi-demo.streamlit.app/)** -->

Stück 6 der Linie **Regelung (von LQR bis MPC)** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Kante zu [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo)
ist ein **Kontrast**: Dort steht die Stellgrenze im Entwurf des Reglers (MPC als quadratisches Programm); hier steht sie **nicht** darin. Ein PI-Regler rechnet linear, die Grenze schneidet nur hinter dem Regler ab, und der Integrator weiß davon nichts.
Das Stück braucht von den anderen der Linie nichts außer der Idee „Stellgrenze“. Nachbarn: [`offsetfrei-demo`](https://github.com/sebastian-hanisch/offsetfrei-demo) (dort ist der Integrator das Mittel gegen den Dauerfehler, hier seine Kehrseite an der Grenze),
[`lqr-riccati-demo`](https://github.com/sebastian-hanisch/lqr-riccati-demo) (ein LQR hat keinen Integrator), [`regelung-verzug-demo`](https://github.com/sebastian-hanisch/regelung-verzug-demo) (dort ist Verzug eine Eigenschaft des Reglers; hier ist die Totzeit nur eine Streckeneigenschaft, kein Prädiktor).
Aus den Schwesterlinien: [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo) und [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) (Rückkopplung mit Totzeit am Lager; hier eine Strecke mit Verstärkung, kein Lager) und
[`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo) (dort eine Klemmung ohne Integrator: Chaos statt Windup, eine andere Ursache).

## Warum dieses Problem

Ein PI-Regler gibt u = sat(kp·e + I) aus, der Integrator addiert je Periode ki·e. Ist der Sollwert eine Zeit lang unerreichbar, sättigt u, aber I wächst weiter. **Wird der Sollwert danach erreichbar, wie lange hält der aufgezogene Integrator die Stellgröße noch an der Grenze, was kosten die gängigen Gegenmittel, und wann helfen sie nicht?**
Gemessen in diesem Modell (Strecke erster Ordnung, zwei Verzögerungen oder schwach gedämpft, mit und ohne Totzeit; PI, symmetrische Grenze; erster Sollwert 4,7 für 57 Perioden, dann 0,77): Ohne Anti-Windup braucht der Regler 916 bis 1 065 Perioden zum Einschwingen (in 23 der 24 Fälle mit Verstärkung 1; der 24. Fall mit Verstärkung 2: 162),
mit Rückrechnung und kaw = ki/kp 4 bis 51. Die Dauer wächst linear mit der Länge der Sättigung (16,1 Perioden je gesättigter Periode beim Verhältnis (r₁ − G)/(G − r₂)). Die Gegenmittel sind nicht gratis: Die Rückrechnung mit kaw = 0,5 unterschwingt in 23 von 24 Fällen um mehr als 0,05 (bis 1,71),
mit Messrauschen verschiebt die Rückrechnung den Mittelwert von y, sobald das Rauschen die Grenze erreichen lässt (bei Soll 0,9 und Rauschen ±0,2: 0,879 bei kaw = 0,05, 0,764 bei kaw = 1, ohne Anti-Windup 0,900), und in 13 von 3 000 Zufallsfällen ist die Rückrechnung länger als gar nichts (bei einem Zufalls-Seed; 7 davon, alle mit schwingender Strecke, schwingen in 6 000 Perioden nicht ein).
Alles ist **eigene Herleitung und Messung in diesem Modell, bei diesen Parametern**. Windup, Rückrechnung und bedingte Integration sind Lehrbuchstoff; eine Neuheitsaussage gibt es nicht. Nicht gemessen: Störung statt Sollwert, instabile Strecken, PID, Vergleich mit MPC.

## Modell

**Zeitlage (gilt für jede Zahl):** Abtastzeit 1; y_k ist der Ausgang am Anfang der Periode k, also das, was der Regler in Periode k sieht. e_k = r_k − y_k, v_k = kp·e_k + I_k, u_k = sat(v_k) mit |u| ≤ 1, I_(k+1) = I_k + ki·e_k + (Anti-Windup-Term). Die Strecke bekommt u_k und liefert y_(k+1). Anfangszustand y = 0, I = 0.

- **Strecken** (alle mit Gleichstromverstärkung G; mit Stellgrenze 1 erreicht y höchstens G): erste Ordnung y⁺ = a·y + G(1 − a)·u, wahlweise mit Totzeit d Perioden (u_(k−d)); zwei Verzögerungen in Reihe; schwach gedämpft y⁺ = c₁·y − c₀·y₋ + G(1 − c₁ + c₀)·u (in der App über Polradius r und Winkel θ: c₁ = 2r·cos θ, c₀ = r²).
- **Anti-Windup-Varianten** (eigene Umsetzung; die Namen folgen der Literatur): **keine**; **Rückrechnung** (Back-Calculation) I += kaw·(u − v); **bedingte Integration**: I bleibt stehen, solange v ≠ u und e·(v − u) > 0 (der Fehler zeigt in Richtung der Sättigung); **Integratorgrenze**: I auf [−1, 1] abgeschnitten. Die Regel der bedingten Integration und die Integratorgrenze sind eigene Festlegungen.
- **Szenario aller Messungen** außer dem Nachbau B1: Sollwert r₁ = 4,7 für T₁ = 57 Perioden (unerreichbar), dann r₂ = 0,77. Die runden Zahlen (5, 60, 0,8) sind **kein** Preset (kein Schnellstart-Szenario der Seitenleiste): dort fällt der Ausstieg auf einen Gleichstand v = 1 (B1), und die Rundung entscheidet über eine Periode.
- **Kennzahlen** (Zähleinheit Periode, ab dem Sprung auf r₂): **Einschwingzeit** = Perioden bis zur **letzten** Ausfahrt aus dem Band |y − r₂| ≤ 0,025·r₂ (die erste Einfahrt täuscht bei Unterschwingen, B1); **Ausstieg** = Perioden bis zur ersten Periode mit |v| ≤ 1; **Unterschwingen** = größtes r₂ − y nach dem Sprung (nur dort als Kennzahl gezeigt, wo y beim Sprung über r₂ liegt).
- Die Formeln stehen im Expander 📐 der App. **Zufall:** nur das Messrauschen (gleichverteilt, ganzzahliger SplitMix64, `wpi_rng.py`, Zufalls-Seed einstellbar) und die Zufallsfälle von Experiment 6 (nur in `tools/make_frozen.py` und den Tests); Logarithmen, Exponentialfunktion und Kosinus laufen dort über `decimal` (auf jeder Plattform dieselben Bits), nie über `math.log` oder `math.cos`.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Regelschleife** (`wpi_dynamics.simulate`) | Periode für Periode, nur +, −, ×, ÷ und Vergleiche: dieselben Funktionen laufen mit `float` und `fractions.Fraction`; Abbruch nach 500 Perioden im Band, Horizont 6 000 Perioden, Divergenz bei \|I\| > 10¹² | „Nicht eingeschwungen“ heißt: bis 6 000 Perioden nicht im Band; Gleichstände an Schwellen (v = 1) hängen an der Rundung, Bänder statt exakter Zahlen |
| **Vorhersage des Ausstiegs** (`predict_exit`) | Anderer Rechenweg: offene Stufenantwort der Strecke und Summen, ohne Regler-Simulation (B3) | gilt nur, wenn \|u\| bis zum Sprung durchgehend an der Grenze lag (wird geprüft, sonst „Annahme verletzt“) |
| **Rückrechnung in der Sättigung** (`integrator_recursion`, `v_after_jump`) | Rekursion I⁺ = (1 − kaw)·I + (ki − kaw·kp)·e + kaw und das Gleichgewicht I\* = 1 + (ki/kaw − kp)·e | gilt nur, solange v gesättigt bleibt; stabil für 0 < kaw < 2 (B6) |
| **Messrauschen** (`noise_run`) | 22 000 Perioden, die ersten 2 000 zählen nicht; Mittel von y, Anteil gesättigter Perioden, exakte Summenidentität | eine Strecke, ein Sollwert, gleichverteilt (B7) |
| **Vorgerechnet** (`data/wpi_frozen.json`, erzeugt von `tools/make_frozen.py`) | 24 Fälle und 3 000 Zufallsfälle (Zufalls-Seed 1506) | die App kennzeichnet sie als „vorgerechnet“; die Rechnung dauert rund eine halbe Minute |

Die Live-Rechnung der App (jedes Experiment) braucht auf dem Entwicklungsrechner höchstens etwa 0,2 Sekunden je Eingabe (Messung mit Python 3.14, nicht festgeschrieben; das Zeichnen kommt hinzu). Alle Zahlen der Messreihen sind mit `tools/make_frozen.py` und den Tests reproduzierbar; Zählungen aus Neu-Rechnung stehen in den Tests als Band.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: in diesem Modell, bei diesen Parametern; Szenario 4,7 für 57 Perioden, dann 0,77, außer wo anders genannt. Die Spalte „Test“ nennt die Funktion in `tests/test_claims.py`, die genau diese Form prüft (Band dort, wo Plattformen abweichen können; „Messung“ heißt: Zahl dieser README, im Test nur auf Konsistenz).

| Nr | Frage | Befund | Test |
|---|---|---|---|
| B1 | V6 nachgebaut (Strecke 0,9; PI 2 / 0,2; Sollwert 5 für 60 Perioden, dann 0,8) | Ohne Anti-Windup liegt der Ausstieg in Brüchen bei 1 215 Perioden (v − 1 fällt je Periode um genau 0,04 und trifft 0 mit einem Rest unter 10⁻⁵⁰), in Gleitkomma bei 1 216: ein **Gleichstand** v = 1, den die Rundung entscheidet. Einschwingzeit in dieser Zählung (y vor dem Schritt) 1 244; die Vormessung zählte y nach dem Schritt und meldete 1 243. Die Zahlen dieses Falls gelten nur als Band (Ausstieg 1 215 bis 1 216, Einschwingzeit 1 242 bis 1 244). Mit Rückrechnung und kaw = 0,5 ist y schon nach einem Schritt im Band (0,798; die Vormessung meldete „erste Einfahrt: 0“), fällt danach aber bis 0,132 (Unterschwingen 0,668) und ist erst nach 48 Perioden endgültig im Band; bedingte Integration 37 (Unterschwingen 0,229), Integratorgrenze 11 (kein Unterschwingen). | `test_b1_v6_*` |
| B2 | Dauer und Wirkung in 24 Fällen (8 Sätze je Streckenart: erste Ordnung mit a = 0,5 / 0,9 / 0,97 und einmal G = 2, zwei Verzögerungen und schwach gedämpft, Totzeit 2, 3, 5, 8; PI-Sätze mit kp 0,5 bis 5 und ki 0,05 bis 0,5; lineare Schleife stabil, ρ 0,50 bis 0,96) | Einschwingzeit **ohne Anti-Windup** im Median 951 (Spanne 162 bis 1 065; 916 bis 1 065 in den 23 Fällen mit G = 1, der Fall mit G = 2: 162), **Rückrechnung kaw = ki/kp** Median 22,5 (4 bis 51), kürzer als „Keine“ in 24 von 24, Unterschwingen über 0,05 in 6 von 24 (größtes 0,13; in 18 von 24 höchstens 0,05); **Rückrechnung kaw = 0,5** Median 47 (4 bis 109), kürzer in 24 von 24, Unterschwingen über 0,05 in 23 von 24 (größtes 1,71); **bedingte Integration** Median 40 (9 bis 98), kürzer in 24 von 24, Unterschwingen über 0,05 in 24 von 24 (größtes 0,90); **Integratorgrenze** Median 23 (4 bis 51), kürzer in 24 von 24, Unterschwingen über 0,05 in 5 von 24 (größtes 0,13). „Keine“ unterschwingt in 1 von 24 über 0,05 (0,06, im Fall mit G = 2). Je Streckenart ohne Anti-Windup: erste Ordnung 917 bis 1 065 (G = 2: 162), zwei Verzögerungen und schwach gedämpft 916 bis 1 015, Totzeit 921 bis 1 038. Die Windup-Dauer hängt bei G = 1 kaum an Streckenart und Parametersatz; bei G = 2 sind es 162, sie hängt also stark an der Gleichstromverstärkung. | `test_b2_*` (Neu-Rechnung und vorgerechnet) |
| B3 | Ausstieg ohne Regler vorhergesagt | Aus der offenen Stufenantwort und Summen (anderer Rechenweg) ist die Vorhersage gleich der Regler-Simulation in 24 von 24 Fällen (kleinster Abstand von v zur Grenze 1,0·10⁻³), in 30 von 30 weiteren (T₁ = 1 bis 240, drei Strecken) und in 2 727 von 2 727 Zufallsfällen, in denen die Annahme „bis zum Sprung gesättigt“ galt (273 Fälle erfüllten sie nicht: in 272 war die Sättigung schon vor dem Sprung beendet, in einem weiteren gab es keinen Ausstieg im Horizont von 6 000 Perioden; kleinster Abstand 1,2·10⁻⁵). Näherung S/Δ − (1 + kp·Δ)/(ki·Δ): Abweichung in den 24 Fällen im Median 0,1 %, höchstens 2,6 % (a = 0,97, wo y bei T₁ = 57 die Grenze noch nicht erreicht hat). | `test_b3_*` |
| B4 | Dauer gegen die Länge T₁ der Sättigung | Der Ausstieg wächst linear in T₁ mit der Steigung 16,08 bis 16,10 (Erwartung (r₁ − G)/(G − r₂) = 3,7/0,23 = 16,087; Fälle a1, c1, b1); bei T₁ = 1: 28 / 41 / 50 Perioden, bei T₁ = 60: 977 / 991 / 999, bei T₁ = 240: 3 873 / 3 886 / 3 895. Die Einschwingzeit mit Rückrechnung (kaw = ki/kp) ist ab T₁ = 60 konstant (12 / 16 / 25), die der bedingten Integration auch (38 / 38 / 31): beide hängen nicht an der Länge der Sättigung. In 200 Zufallsfällen (Zufalls-Seed 99, zufällige Strecke, kp, ki, r₁, r₂; 191 davon mit Ausstieg bei T₁ = 60, 120 und 240) wächst der Ausstieg monoton in T₁, und die Steigung zwischen T₁ = 120 und 240 weicht höchstens 1 % von der Erwartung ab (größte Abweichung 0,8 %). | `test_b4_*` |
| B5 | Wahl von kaw (Gitter 0,01 bis 1,9) | kaw = 0,01 gibt eine Einschwingzeit von 212 bis 258 (Fall a8: 95), kaw = 0,05 von 59 bis 103 (a8: 47; die drei Fälle mit ki/kp = 0,05: 31, 32, 51): zu kleines kaw lässt den Integrator mit der Zeitkonstante 1/kaw abklingen. **kaw = ki/kp** liegt in 23 von 24 Fällen höchstens beim 1,5-Fachen des besten Gitterwerts plus 2 (Ausnahme c2: bester Wert 14 bei kaw = 1,9, bei ki/kp = 0,08: 26) und ist in 20 von 24 nicht größer als der beste Gitterwert (ein Gitterwert muss ki/kp nicht treffen); in vier Fällen schlägt ein anderes kaw: a8 (kaw = 1: 23 gegen 25), b4 (0,2: 18 gegen 21), c2 (1,9: 14 gegen 26), c8 (1,5: 48 gegen 51). Fall a1: kaw = 0,01: 229, kaw = ki/kp = 0,1: 12, kaw = 0,5: 48 mit Unterschwingen 0,63. | `test_b5_*` |
| B6 | Gleichgewicht des Integrators in der Sättigung, obere Grenze von kaw | v nach dem Sprung gemessen gegen die Formel v = 1 − kp·Δ + (ki/kaw − kp)·(r₁ − y) (Fall a1, y = 0,9975): kaw = 0,1 (= ki/kp): 0,5425 gegen 0,5449; 0,2: −3,160 gegen −3,158; 0,5: −5,380 gegen −5,379; 1,0: −6,120 gegen −6,1195; bei kaw = 0,05: 7,596 gegen 7,950 (der Integrator hat nach 57 Perioden sein Gleichgewicht nicht ganz erreicht). **Bei kaw > ki/kp liegt v nach dem Sprung tiefer als bei kaw = ki/kp, für kaw ab 0,2 weit unter −1** (hier −3 bis −6): das passt zum Unterschwingen in diesem Szenario (nur Formel und Beobachtung an Fall a1; ein Gegenversuch, der v als Ursache prüft, fehlt, und knapp über ki/kp tritt das Unterschwingen schon auf, wo v noch nicht weit unter −1 liegt); bei kaw = ki/kp fällt der Term mit e weg. Die Rekursion stimmt mit der Simulation überein (\|I\| nach 57 Perioden 4,925 bei kaw = 0,5), in der Sättigung stabil für 0 < kaw < 2; kaw = 2,5 und 5 divergieren (Abbruch bei \|I\| > 10¹²), bei 2,0 und 2,1 bleibt der Lauf in diesem Szenario beschränkt (die Rekursion gilt nur, solange v gesättigt bleibt). | `test_b6_*` |
| B7 | Messrauschen nahe der Grenze (Fall a1, Soll 0,9, gleichverteiltes Rauschen, Zufalls-Seed 15, 20 000 Perioden nach 2 000 Einschwingperioden) | Mittel von y bei Rauschen ±0,2 / ±0,4 / ±0,8: Keine 0,8998 / 0,8997 / 0,8993; Rückrechnung kaw = 0,05: 0,8792 / 0,8385 / 0,7518; kaw = 0,5: 0,8022 / 0,6488 / 0,3491; kaw = 1: 0,7637 / 0,5604 / 0,2163; bedingte Integration 0,8286 / 0,7289 / 0,5281; Integratorgrenze 0,8884 / 0,8372 / 0,7264 (Band im Test ±0,002). Der Anteil gesättigter Perioden ohne Anti-Windup liegt bei 50 / 65 / 76 %. **Exakte Identität** (Summen desselben Laufs): Σe = (kaw/ki)·Σ(v − u) + (I_Ende − I_Start)/ki; ohne Rückrechnung ist das Mittel von e damit 0,0000 bis 0,0001, mit Rückrechnung gleich kaw/ki mal dem mittleren Abschneiden. Das passt in diesem Modell zur Aussage der Quelle, ein zu großes kaw lasse Messrauschen den Integrator zurücksetzen (nur Fließtext; gemessen ist die Verschiebung des Mittelwerts, dass sie dasselbe ist wie das „Zurücksetzen“ der Quelle, ist eine Deutung); eine Strecke, ein Sollwert, ein Zufalls-Seed. | `test_b7_*` |
| B8 | Gegenprobe mit 3 000 Zufallsfällen (Zufalls-Seed 1506; zufällige Strecke aus den vier Arten, PI-Satz mit ρ < 0,97, T₁ = 5 bis 149, r₁ = 1,5 bis 6·G, r₂ = 0,2 bis 0,9·G; 5 347 Versuche) | **Rückrechnung kaw = ki/kp** ist kürzer als „Keine“ in 2 983 Fällen, gleich in 4, länger in 13 (in 7 Fällen nie eingeschwungen, siehe unten; in den übrigen 6 höchstens 23 Perioden länger; die „größte Verlängerung“ von 5 934 Perioden ist ein Artefakt des Horizonts von 6 000 Perioden, die tatsächliche Verlängerung ist dort unbegrenzt); bedingte Integration 2 978 / 3 / 19; Integratorgrenze 2 980 / 12 / 8. Mediane (Keine / Rückrechnung / bedingt / Grenze): erste Ordnung 461 / 30 / 40 / 31, Totzeit 447 / 48 / 63 / 50, zwei Verzögerungen 465,5 / 35 / 48,5 / 37, schwach gedämpft 408 / 34 / 45 / 37. „Keine“ bleibt nach dem Sprung in 2 973 von 3 000 Fällen gesättigt (26 Fälle ohne Windup, ein weiterer ohne Ausstieg im Horizont von 6 000 Perioden). Von den 13 Ausnahmen der Rückrechnung betreffen 10 die schwach gedämpfte Strecke (je einmal erste Ordnung, zwei Verzögerungen, Totzeit), und 6 haben T₁ ≤ 10 (der Integrator ist kaum aufgezogen, das Gegenmittel unterschwingt selbst). **7 Fälle** (alle schwach gedämpft) schwingen mit Rückrechnung in 6 000 Perioden nicht ein; in den letzten 2 000 von rund 20 000 Perioden wiederholt sich y dort exakt alle 3 bis 6 Perioden (Grenzzyklus), 67 % bis 100 % der Perioden sind gesättigt; „Keine“ schwingt in allen sieben ein. Ob „Keine“ bei anderem Anfangszustand denselben Zyklus hätte, ist nicht geprüft. „Anti-Windup verkürzt immer“ ist damit falsch; die Aussage gilt in 99,4 % der Zufallsfälle für die Rückrechnung. | `test_b8_*` |
| B9 | Eigenschaften | Ohne Sättigung (Sollwert 0,1 und 0,05) liefern alle vier Varianten dieselbe Bahn in 24 von 24 Fällen und in allen ungesättigten Fällen von 3 000 Zufallsinstanzen mit kleinem Sollwert (2 989 von 3 000 blieben ungesättigt; in den übrigen 11 überstieg kp·e die Grenze); Rückrechnung mit kaw = 0 ist „Keine“ in 24 von 24 (alle Kennzahlen); alle 96 Läufe (24 Fälle × 4 Varianten) schwingen im Horizont ein. | `test_b9_*` |
| B10 | Handbeispiel in Brüchen (Strecke y⁺ = (y + u)/2, PI mit kp = 1 und ki = ½, Sollwert 3 in den Perioden 0 und 1, danach ¼) | Ohne Anti-Windup bleibt u bis k = 5 an der Grenze, obwohl der Fehler ab k = 2 negativ ist (v₂ = 9/4, v₅ = 9/8); erstmals \|v\| ≤ 1 in k = 6 (v = ¾), vier Perioden nach dem Sprung, ohne Gleichstand. Mit Rückrechnung und kaw = ki/kp = ½ ist ab k = 2 genau u = v = ¼ und y_k = ¼ + (½)^(k−1) ohne Unterschwingen; mit kaw = 1 fällt y bis −1/8 (Unterschwingen 3/8). Einschwingzeiten im Band ±1/100 (Handrechnung der Planung): 14 / 6 / 10 / 10 / 8 (keine / kaw = ½ / kaw = 1 / bedingt / Grenze), im Band 2,5 % von r₂ der App: 14 / 7 / 11 / 9 (ohne kaw = 1). Spektralradius der linearen Schleife: Doppelwurzel ½. | `test_b10_*`, `tests/test_dynamics.py` |

### Befunde und Korrekturen gegenüber dem Plan

Vormessungszahlen sind Erwartungen; im Bau neu gemessen. Die Zählungen blieben dieselben (5 347 Versuche, 2 983 / 4 / 13 bei den Zufallsfällen), obwohl die Logarithmen, die Exponentialfunktion und der Kosinus der Zufallsfälle hier über `decimal` statt über `numpy` laufen. Abweichungen und Präzisierungen:

- **Einschwingband des Handbeispiels.** Der Plan nannte 14 / 6 / 10 / 10 / 8 im Band ±1/100; die App benutzt das Band 2,5 % von r₂ (hier 1/160) für alle Szenarien und zeigt 14 / 7 / 11 / 9. Beide Zählungen stehen in B10.
- **Median der Rückrechnung.** Der Plan nannte 22; der Median über 24 Fälle ist 22,5 (der Mittelwert der beiden mittleren Werte). Ebenso 465,5 und 48,5 statt 466 und 49 bei den zwei Verzögerungen in B8.
- **Grenzzyklus.** Der Plan nannte ihn „nicht erklärt, nicht auf Eindeutigkeit geprüft“. Gemessen ist jetzt, dass die sieben nicht eingeschwungenen Läufe sich in den letzten 2 000 Perioden exakt wiederholen (Periodenlänge 3 bis 6). Erklärt und auf Eindeutigkeit geprüft ist er weiterhin nicht.
- **kaw = 2,5 divergiert** nur im Szenario mit langer Sättigung (Fall a1, T₁ = 57), nicht allgemein: im Handbeispiel (T₁ = 2) ist die Einschwingzeit bei kaw = 2,5 elf Perioden (Experiment 4 im Preset „Von Hand“).
- **Preset „Rauschen nahe der Grenze“.** Der Plan sah Sollwert 0,9 mit T₁ = 0 im Szenario vor; dabei fällt der Ausstieg aus dem Anlauf auf einen Gleichstand (Abstand von v zur Grenze 9·10⁻¹⁶). Der Sollwert des Rauschexperiments ist deshalb ein eigener Regler (Voreinstellung 0,9), das Preset nutzt das Szenario „Lange Sättigung“ mit Rauschen ±0,4.
- **Unterschwingen nur mit Sinn.** Kommt y beim Sprung von unten auf r₂ zu (Anlauf aus y = 0), ist r₂ − kleinstes y der Anlauf selbst; die App zeigt die Kennzahl dann nicht.
- **Quellenwiderspruch.** Die Quellenberichte nennen für Windup bei Åström/Murray einmal Abschnitt 11.4 (2. Auflage, Kapitel-PDF von 2019) und einmal 10.4 (1. Auflage, Fassung Februar 2009). Nach den Berichten sind es zwei Ausgaben und keine Verwechslung; gegen die gedruckte Ausgabe nicht geprüft. <!-- lint-ok: dezimalpunkt -->
- **Quelle für kaw = ki/kp** ist nicht gelesen: die Wahl ist eigene Messung in diesem Szenario. Übung 11.13 der Quelle nennt für stetige Zeit eine Stabilitätsbedingung mit kaw > ki/kp (Kreiskriterium); sie ist hier nicht nachgerechnet und wird nicht mit der gemessenen Grenze des Unterschwingens gleichgesetzt.

## Quellen

- **Åström/Murray, *Feedback Systems*** (Windup, Rückrechnung): nach dem Quellenbericht im Fließtext gelesen (Abschnitt 11.4 der 2. Auflage bzw. 10.4 der 1. Auflage): Bei gesättigtem Stellglied ist der Regelkreis aufgetrennt, der Integralterm läuft weiter hoch und hält das Stellglied gesättigt, auch wenn der Fehler das Vorzeichen wechselt; Gegenmittel ist die Rückführung des Sättigungsfehlers über kaw in den Integrator, mit kaw als Vielfachem von ki; ein zu großes kaw lasse Messrauschen den Integrator zurücksetzen. **Keine Zahl aus dem Kapitel ist übernommen** (die Angaben zu ki widersprechen sich im Kapitel). „Bedingte Integration“ steht dort als Name in Übung 11.12; die hier benutzte Regel ist eigene Festlegung, die Formel der Rückrechnung ist die in Abschnitt Modell genannte mit h = 1. <!-- lint-ok: dezimalpunkt -->
- **Rawlings/Mayne/Diehl, *Model Predictive Control*** (Abschnitt 1.5.2, S. 49 bis 53): Offsetfreiheit mit integrierendem Störungsmodell gilt nur, wenn die Beschränkungen im Beharrungszustand inaktiv sind (in eigenen Worten; nichts wörtlich übernommen: das PDF enthält den Vermerk „Weitergabe nur mit Erlaubnis des Verlags“).
- Alles Übrige (Messungen, Formeln, Vorhersage aus der Stufenantwort, Identität der Summen, Gleichgewicht in der Sättigung) ist eigene Herleitung und Messung.

## Ehrliche Grenzen

- **Gilt für** PI mit symmetrischer Grenze, die vier Streckenarten, stabile lineare Schleife (ρ < 1) und das Szenario „unerreichbarer Sollwert für T₁ Perioden, dann erreichbar“. **Nicht gemessen:** Störung statt Sollwert (Lastsprung am Eingang, dann Wegfall), instabile Strecken und Strecken mit Nullstelle rechts (dort bringt die Grenze ein anderes Problem: Stabilität in der Sättigung), PID, Rate-Limit, asymmetrische Grenzen, Mehrgrößenregelung.
- **kaw nur in einem Szenario abgestimmt** (lange Sättigung, dann erreichbarer Sollwert), nicht bei kurzer Sättigung (B8: bei T₁ ≤ 10 liegen 6 der 13 Ausnahmen) oder bei Sollwertfolge; die Wahl kaw = ki/kp ist eigene Messung, die Quelle dafür nicht gelesen.
- **Rauschen:** eine Strecke, ein Sollwert, ein Zufalls-Seed, gleichverteilt (B7); die Rückrechnung ist aus der Sättigungsfunktion gerechnet, nicht aus dem gemessenen Stellglied (nicht gemessen).
- **Das Einschwingband** (2,5 % von r₂) ist eine Wahl; andere Bänder sind nicht gemessen, und die Einschwingzeit hängt am Band (B10).
- **Gleichstände an Schwellen:** v = 1 beim Ausstieg (B1) und Perioden genau auf dem Rand des Bandes entscheiden die Rundung; Presets sind so gewählt, dass |v − 1| beim Ausstieg über 10⁻³ liegt (getestet), der V6-Fall ist nur als Band im Test.
- **Plattform:** Die Regelschleife rechnet nur mit +, −, ×, ÷ und Vergleichen (Gleitkomma nach IEEE 754) und sollte auf Windows und Linux dieselben Bits geben (nur auf Windows geprüft); trotzdem stehen Zählungen aus Neu-Rechnung als Band (Zählungen ± 2). Die Eigenwerte (ρ) kommen aus `numpy` und können in der letzten Stelle abweichen; sie dienen nur als Schwelle „ρ < 0,97“ bei der Auswahl der Zufallsfälle.
- **Kein Vergleich mit MPC:** Qin/Badgwell (2003, Abschnitt 5) nennen nach dem Quellenbericht einen Vergleich von Goodwin u. a. (2001) zwischen unbeschränktem LQR mit Anti-Windup und MPC; nicht nachgerechnet. Kosten von PI mit Anti-Windup gegen MPC sind nicht gemessen.
- **Keine Empfehlung:** Welches Gegenmittel in der Praxis passt, hängt an Strecke, Stellglied und Sollwertfolge; jede Messung gilt für das gemessene Paar aus Strecke und Regler.

## Verifikation

- **Kern:** jede Regel einzeln an Mini-Instanzen von Hand (Handbeispiel in Brüchen, alle vier Varianten, Strecken von Hand), Randfälle (T₁ = 0 und 1, r₂ ≥ G, kaw = 0, 1,9, 2, 2,5, Totzeit 0 und 8, G = 2).
- **Orakel** (`tests/test_oracle_windup_pi.py`): Vorhersage des Ausstiegs aus der Stufenantwort gegen die Regler-Simulation; Brüche gegen Gleitkomma; Rekursion in der Sättigung gegen die Simulation; Summenidentität; Spektralradius aus den Nullstellen des charakteristischen Polynoms gegen die Eigenwerte; Stufenantwort gegen `scipy.signal.dlsim`; die ganze Schleife gegen eine zweite Implementierung mit Matrizen.
- **Eigenschaften** (`tests/test_eigenschaften_windup_pi.py`): ohne Sättigung alle Varianten gleich; kaw = 0 gleich „Keine“; Zeitinvarianz (Vorlauf mit Sollwert 0); Skalierung des ganzen Problems mit Zweierpotenzen; Monotonie und Linearität des Ausstiegs in T₁; Invarianten der bedingten Integration und der Integratorgrenze; Beharrungszustand.
- **Claims** (`tests/test_claims.py`): jede Zahl der Befunde B1 bis B10 mit ihren Voraussetzungen; die vorgerechneten Reihen werden von `tools/make_frozen.py` neu erzeugt (24 Fälle ganz, die ersten 300 Zufallsfälle; mit der Umgebungsvariable `WPI_FULL=1` alle 3 000) und mit der Datei verglichen. Zahl → Test: siehe die Spalte „Test“ der Befunde; Zahlen in den Presets und Verdikten stehen in `tests/test_presets.py` und `tests/test_verdicts.py`.
- **Presets:** vollständig, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung, je Preset die erwartete Verdikt-Zeile, Abstand von v zur Grenze beim Ausstieg über 10⁻³.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, Worst-Case-Einstellungen mit Zeitgrenze) und Footer-Test.
- **Layout:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Preset-Reglers hebt die Markierung auf (`tests/test_layout.py`).
- **Testzahl:** 196 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Windup von Hand, Kernfragen, fünf Experimente, Grenzen, Formeln |
| `data/wpi_frozen.json` | vorgerechnet: 24 Fälle und 3 000 Zufallsfälle (Zufalls-Seed 1506) |
| `tools/make_frozen.py` | erzeugt (und prüft mit `--check`) die vorgerechneten Reihen |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `wpi_constants.py` | Reglergrenzen, feste Annahmen, die 24 Fälle, Farben, Presets |
| `wpi_dynamics.py` | Kern: Strecken, PI-Regler mit vier Varianten, Kennzahlen, Vorhersage, Rückrechnung, Rauschen |
| `wpi_evaluation.py` | Einstellungen, Analysen der Experimente, vorgerechnete Reihen, Verdikte als reine Funktionen |
| `wpi_layout.py` | Fester Kopf (Stylesheet) |
| `wpi_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) |
| `wpi_rng.py` | SplitMix64 (Kopie der geprüften Vorlage) plus `exact_exp` und `uniform_array` |
| `wpi_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Mini-Instanzen von Hand, Orakel, Eigenschaften, Claims, Verdikte, Randfälle, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **PID mit Ableitungsanteil, Rate-Limit, asymmetrische Grenzen, Mehrgrößenregelung:** höchstens eine spätere Erweiterung.
- **Störung statt Sollwert** (Lastsprung am Eingang): als weiteres Experiment erst nach `offsetfrei-demo`.
- **Instabile Strecken und Strecken mit Nullstelle rechts:** anderes Problem (Stabilität in der Sättigung).
- **Kostenvergleich mit MPC:** die Grenze im Entwurf behandelt [`mpc-stellgrenzen-demo`](https://github.com/sebastian-hanisch/mpc-stellgrenzen-demo); der Vergleich ist nicht gemessen.
- **PDF-Export:** die Experimente zeigen ihre Zahlen auf der Seite; ein Export fehlt.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
