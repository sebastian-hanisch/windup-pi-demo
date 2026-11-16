"""Erzeugt `data/wpi_frozen.json`: die vorgerechneten Messreihen der Demo (24 Fälle, 3 000 Zufallsfälle mit Zufalls-Seed 1506).

Aufruf (aus dem Repo-Ordner): `python tools/make_frozen.py` schreibt die Datei, `python tools/make_frozen.py --check` rechnet neu und vergleicht mit der Datei
(Zählungen exakt, wo die Rechnung nur aus +, −, ×, ÷ und dem ganzzahligen SplitMix64 besteht; auf einer anderen Plattform dürfen höchstens wenige Zufallsfälle wegen der
Eigenwerte-Schwelle ρ < 0,97 abweichen, dann ist das ein Befund, kein Fehler der Zählung). Rechenzeit rund eine halbe Minute.
Die Datei ist in README und App als „vorgerechnet“ gekennzeichnet."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wpi_evaluation as ev  # noqa: E402


def build() -> dict:
    cases, attempts = ev.random_cases()
    return {"faelle": ev.compute_cases(), "zufall": ev.summarize_random(cases, attempts)}


def main() -> int:
    t0 = time.time()
    data = build()
    text = json.dumps(data, ensure_ascii=False, indent=None, separators=(",", ":"))
    if "--check" in sys.argv:
        old = json.loads(ev.FROZEN_PATH.read_text(encoding="utf-8"))
        new = json.loads(text)
        print("identisch" if old == new else "ABWEICHUNG")
        return 0 if old == new else 1
    ev.FROZEN_PATH.parent.mkdir(exist_ok=True)
    ev.FROZEN_PATH.write_text(text + "\n", encoding="utf-8", newline="\n")
    z = data["zufall"]
    print(f"geschrieben: {ev.FROZEN_PATH} ({len(text) // 1024} KB, {time.time() - t0:.0f} s); {z['n']} Zufallsfälle aus {z['versuche']} Versuchen; Vergleich:", z["vergleich"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
