"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import wpi_constants as C
import wpi_dynamics as D


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.18), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_bahn(bahnen, t1: int, r2: float, with_integrator: bool = True, height: int = 640):
    """Bahnen über der Zeit: oben y und Sollwert (Band um r2 nach dem Sprung), Mitte Stellgröße u und Reglerausgang v (gestrichelt, Grenzen ±1), unten der Integrator I.
    `bahnen`: Liste (Name, Bahn, Farbe, gestrichelt?); die erste ist die Hauptbahn (mit Sollwert und Band)."""
    rows = 3 if with_integrator else 2
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Ausgang y und Sollwert r", "Stellgröße u (Linie) und Reglerausgang v (gestrichelt)") + (("Integrator I (vor dem Update)",) if with_integrator else ()))
    main = bahnen[0][1]
    ks = list(range(len(main.y)))
    fig.add_trace(go.Scatter(x=ks, y=main.r, mode="lines", line=dict(color=C.COLOR_R, width=1.5, dash="dot"), name="Sollwert r"), row=1, col=1)
    tol = D.TOL_REL * abs(r2)
    fig.add_hrect(y0=r2 - tol, y1=r2 + tol, fillcolor=C.COLOR_BAND, line_width=0, row=1, col=1)
    for name, bahn, color, dash in bahnen:
        kk = list(range(len(bahn.y)))
        d = "dash" if dash else "solid"
        fig.add_trace(go.Scatter(x=kk, y=[float(x) for x in bahn.y], mode="lines", line=dict(color=color, width=2.5 if not dash else 1.8, dash=d), name=name,
                                 legendgroup=name, hovertemplate="k = %{x}: %{y:.3f}<extra>" + name + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=kk, y=[float(x) for x in bahn.u], mode="lines", line=dict(color=color, width=2 if not dash else 1.4, dash=d), name=name + ": u",
                                 legendgroup=name, showlegend=False, hovertemplate="k = %{x}: u = %{y:.3f}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=kk, y=[float(x) for x in bahn.v], mode="lines", line=dict(color=color, width=1.2, dash="dot"), name=name + ": v",
                                 legendgroup=name, showlegend=False, hovertemplate="k = %{x}: v = %{y:.3f}<extra></extra>", opacity=0.7), row=2, col=1)
        if with_integrator:
            fig.add_trace(go.Scatter(x=kk, y=[float(x) for x in bahn.I], mode="lines", line=dict(color=color, width=2 if not dash else 1.4, dash=d), name=name + ": I",
                                     legendgroup=name, showlegend=False, hovertemplate="k = %{x}: I = %{y:.3f}<extra></extra>"), row=3, col=1)
    for yv in (-D.UMAX, D.UMAX):
        fig.add_hline(y=yv, line=dict(color="rgba(120,120,120,0.7)", width=1, dash="dash"), row=2, col=1)
    if t1 > 0:
        for r in range(1, rows + 1):
            fig.add_vline(x=t1, line=dict(color="rgba(120,120,120,0.6)", width=1, dash="dash"), row=r, col=1)
    fig.update_xaxes(title_text="Periode k", row=rows, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.12), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_exit(rows: list[dict]):
    """Ausstieg aus der Sättigung gegen die Dauer T₁ (gemessen, vorhergesagt) und Einschwingzeiten der Gegenmittel; Zeiten logarithmisch."""
    xs = [r["t1"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[max(r["ausstieg"] or 0, 0.5) for r in rows], mode="markers", marker=dict(size=11, color=C.VARIANT_COLOR["keine"], symbol="circle"),
                             name="Ausstieg ohne Anti-Windup (gemessen)", hovertemplate="T₁ = %{x}: %{y:.0f} Perioden<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[max(r["vorhergesagt"] or 0, 0.5) for r in rows], mode="lines", line=dict(color=C.VARIANT_COLOR["keine"], width=2, dash="dot"),
                             name="vorhergesagt aus der Stufenantwort", hovertemplate="T₁ = %{x}: %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[max(r["bc"] or 0, 0.5) for r in rows], mode="lines+markers", line=dict(color=C.VARIANT_COLOR["bc"], width=2.5),
                             name="Einschwingzeit, Rückrechnung kaw = ki/kp", hovertemplate="T₁ = %{x}: %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[max(r["bed"] or 0, 0.5) for r in rows], mode="lines+markers", line=dict(color=C.VARIANT_COLOR["bed"], width=2.5),
                             name="Einschwingzeit, bedingte Integration", hovertemplate="T₁ = %{x}: %{y:.0f}<extra></extra>"))
    fig.update_xaxes(title_text="Dauer T₁ der Sättigung (Perioden)", type="log")
    fig.update_yaxes(title_text="Perioden ab dem Sprung (logarithmisch)", type="log")
    return _base(fig, 400)


def build_cases(entry: dict):
    """Ein Fall der 24: Einschwingzeit (Balken, logarithmisch) und Unterschwingen der fünf Varianten, nebeneinander."""
    names = ["keine", "bc05", "bckp", "bed", "ilim"]
    labels = ["Keine", "Rückr. kaw = 0,5", "Rückr. kaw = ki/kp", "Bedingte Integr.", "Integratorgrenze"]
    colors = [C.VARIANT_COLOR["keine"], "#7fb8b2", C.VARIANT_COLOR["bc"], C.VARIANT_COLOR["bed"], C.VARIANT_COLOR["ilim"]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Einschwingzeit (Perioden, logarithmisch)", "Unterschwingen unter r₂"), horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=labels, y=[max(entry[n]["einschwing"], 0.5) for n in names], marker_color=colors, text=[str(entry[n]["einschwing"]) for n in names], textposition="outside",
                         showlegend=False, hovertemplate="%{x}: %{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=[entry[n]["unter"] for n in names], marker_color=colors, text=[f"{entry[n]['unter']:.2f}".replace(".", ",") for n in names], textposition="outside",
                         showlegend=False, hovertemplate="%{x}: %{text}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_kaw(rows: list[dict], ratio: float, chosen: float):
    """Rückrechnung: Einschwingzeit (links, logarithmisch) und Unterschwingen (rechts) über kaw (logarithmische Achse); divergierte Werte als Kreuze am oberen Rand der linken Abbildung,
    ki/kp und der gewählte Wert markiert."""
    ok = [r for r in rows if not r["divergiert"]]
    bad = [r for r in rows if r["divergiert"]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Einschwingzeit (Perioden, logarithmisch)", "Unterschwingen unter r₂"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=[r["kaw"] for r in ok], y=[max(r["einschwing"], 0.5) for r in ok], mode="lines+markers", line=dict(color=C.VARIANT_COLOR["bc"], width=2.5),
                             name="Einschwingzeit", hovertemplate="kaw = %{x}: %{y:.0f} Perioden<extra></extra>"), row=1, col=1)
    if bad:
        top = max([max(r["einschwing"], 1) for r in ok] or [10])
        fig.add_trace(go.Scatter(x=[r["kaw"] for r in bad], y=[top] * len(bad), mode="markers", marker=dict(symbol="x", size=13, color="#000000"), name="divergiert",
                                 hovertemplate="kaw = %{x}: divergiert<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r["kaw"] for r in ok], y=[r["unter"] if (r["y_sprung"] or 0) > 0 else 0 for r in ok], mode="lines+markers",
                             line=dict(color=C.COLOR_U, width=2.5), name="Unterschwingen", hovertemplate="kaw = %{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=ratio, line=dict(color=C.COLOR_I, width=2, dash="dash"), row=1, col=col)
        if abs(chosen - ratio) > 1e-9:
            fig.add_vline(x=chosen, line=dict(color="rgba(80,80,80,0.8)", width=1.5, dash="dot"), row=1, col=col)
    # Legendeneinträge für die senkrechten Linien (Shapes haben keinen eigenen Eintrag)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C.COLOR_I, width=2, dash="dash"), name="kaw = ki/kp"), row=1, col=1)
    if abs(chosen - ratio) > 1e-9:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="rgba(80,80,80,0.8)", width=1.5, dash="dot"), name="gewählter Wert"), row=1, col=1)
    fig.update_xaxes(title_text="kaw (logarithmisch)", type="log")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="linear", rangemode="tozero", row=1, col=2)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_noise(rows: list[dict], r: float):
    """Mittel von y je Variante bei Messrauschen, gegen den Sollwert."""
    colors = [C.VARIANT_COLOR["keine"]] + [C.VARIANT_COLOR["bc"]] * 4 + [C.VARIANT_COLOR["bed"], C.VARIANT_COLOR["ilim"]]
    fig = go.Figure(go.Bar(x=[x["name"] for x in rows], y=[x["mean_y"] for x in rows], marker_color=colors, text=[f"{x['mean_y']:.3f}".replace(".", ",") for x in rows],
                           textposition="outside", hovertemplate="%{x}: %{text}<extra></extra>"))
    fig.add_hline(y=r, line=dict(color=C.COLOR_R, width=2, dash="dash"), annotation_text="Sollwert", annotation_position="top left")
    fig.update_yaxes(title_text="Mittel von y")
    fig.update_xaxes(tickangle=-25)
    return _base(fig, 420)
