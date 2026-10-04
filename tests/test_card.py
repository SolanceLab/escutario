# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
import json
from pathlib import Path

from escutario.card import build_card

ROOT = Path(__file__).resolve().parents[1]


def test_card_from_a_minimal_entry():
    entry = {
        "duration": 30.0, "key": {"key": "C minor", "level": "high"}, "tempo": {"bpm": 116.9, "ambiguous": False},
        "sections": [12.0], "changes": [{"t": 12.0, "strength": 0.9, "shifts": [{"stem": "bass", "share_change_pct": 40}], "arrivals": ["bass enters"]},
                                          {"t": 3.0, "strength": 0.8, "relabel": {"from": "other", "to": "piano"}, "shifts": []}],
        "voices": [0, 1, 1, 2, 2, 3], "riffs": [{"start_s": 5.0, "end_s": 8.0, "notes": 9, "low": "E3", "high": "A3", "confidence": "high"}],
        "held": [{"t": 14.0, "d": 2.4, "note": "G4", "src": "pitch"}, {"t": 20.0, "d": 0.9, "note": "C4"}],
        "vibrato": {"judged": 4, "with_vibrato": [], "straight": 4},
        "breaths": [{"t": 11.5, "d": 0.3, "c": "high"}], "longest_phrase_s": 6.2,
        "ring": {"overall": -9.0, "label": "strong"}, "tilt": {"alpha": -1.5, "alpha_label": "full/pressed"}, "dynamic_range": 31.0,
    }
    card = build_card(entry, title="Clip", artist="Someone", source="https://youtu.be/x")
    assert card.startswith("ESCUTÁRIO · Clip — Someone")
    assert "tempo 117 bpm" in card and "C minor" in card
    assert "0:12.0 bass enters" in card and "piano" not in card.split("ARRIVALS")[1].split("\n")[0]   # relabels are not arrivals
    assert "two or more at once 60%" in card and "three or more 20%" in card
    assert "0:05.0–0:08.0 9 notes E3–A3" in card
    assert "G4 2.4s" in card and "C4" not in card.split("HELD NOTES")[1].split("\n")[0]
    assert "0 of 4 held notes waver — sung straight" in card
    assert "LIMITS:" in card
    for mood in ("sad", "angry", "emotional", "happy"):
        assert mood not in card.lower()


def test_card_on_an_empty_entry_does_not_crash():
    assert build_card({}).startswith("ESCUTÁRIO · Untitled clip")


def test_card_on_the_real_mashup_is_compact():
    viz = ROOT / "web" / "viz_data.json"
    if not viz.exists():
        return
    entry = json.loads(viz.read_text()).get("apparition-x-unethical")
    if not entry:
        return
    card = build_card(entry, title="The Apparition × UNETHICAL", artist="Sleep Token × Faouzia (Goobsie)")
    assert len(card) < 2500 and "RIFFS" in card and "VIBRATO: 1 of 58" in card
