"""Abbildungen: alle Achsen gesperrt (fixedrange), richtige Zahl der Spuren, keine Ausnahme bei Randfällen (divergierte Werte, leere Spuren)."""

import wpi_constants as C
import wpi_evaluation as ev
import wpi_visualization as V

SC = ev.Scenario(kind="lag1", a1=0.9, gain=1.0, kp=2.0, ki=0.2, t1=57, r1=4.7, r2=0.77)


def _locked(fig) -> bool:
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(a.get("fixedrange") is True for a in axes)


def test_trajectory_figure_has_three_panels_and_locked_axes():
    a = ev.analyse(ev.Settings(SC, "bc_kp"))
    fig = V.build_bahn([(a.selected.name, a.bahn_selected, C.VARIANT_COLOR["bc"], False), ("Keine", a.bahn_keine, C.VARIANT_COLOR["keine"], True)], SC.t1, SC.r2)
    assert _locked(fig)
    assert len(fig.data) == 1 + 2 * 4                         # Sollwert, dann je Bahn y, u, v, I
    vlines = [s for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1]
    assert len(vlines) == 3 and all(s.x0 == SC.t1 for s in vlines)      # der Sprung in jedem der drei Felder


def test_trajectory_figure_without_integrator_panel_and_without_jump_line():
    a = ev.analyse(ev.Settings(ev.Scenario(t1=0, r1=0.9, r2=0.9)))
    fig = V.build_bahn([("Keine", a.bahn_selected, C.VARIANT_COLOR["keine"], False)], 0, 0.9, with_integrator=False, height=400)
    assert _locked(fig) and len(fig.data) == 1 + 3
    assert not [s for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1]       # bei T₁ = 0 gibt es keine Sprunglinie


def test_exit_figure_log_axes_and_four_traces():
    fig = V.build_exit(ev.exit_sweep(SC))
    assert _locked(fig) and len(fig.data) == 4 and fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert all(y > 0 for t in fig.data for y in t.y)          # logarithmische Achse: keine Nullen


def test_cases_figure_for_every_case():
    for cid in C.CASE_IDS:
        fig = V.build_cases(ev.frozen()["faelle"][cid])
        assert _locked(fig) and len(fig.data) == 2 and len(fig.data[0].y) == 5 and all(y > 0 for y in fig.data[0].y)


def test_kaw_figure_marks_divergence_and_legend_entries():
    rows = ev.kaw_curve(SC, 0.5)
    fig = V.build_kaw(rows, SC.ki / SC.kp, 0.5)
    names = [t.name for t in fig.data]
    assert _locked(fig) and "divergiert" in names and "kaw = ki/kp" in names and "gewählter Wert" in names
    fig2 = V.build_kaw(ev.kaw_curve(SC, 0.1), SC.ki / SC.kp, 0.1)
    assert "gewählter Wert" not in [t.name for t in fig2.data]          # gewählter Wert gleich ki/kp: keine zweite Linie


def test_noise_figure_has_one_bar_per_variant():
    rows = ev.noise_table(SC, 0.9, 0.2, 15)
    fig = V.build_noise(rows, 0.9)
    assert _locked(fig) and len(fig.data[0].y) == 7 == len(rows)
