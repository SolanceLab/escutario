# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
import json

import numpy as np

from escutario import breakdown as B


def lane(pattern, n=200):
    """dB lane: -80 (silent) where pattern is 0, loudness following pattern elsewhere."""
    return [float(v) for v in np.where(pattern[:n] > 0, -20 + 5 * pattern[:n], -80)]


rng = np.random.default_rng(3)
piano = (np.arange(200) > 0).astype(float) * (1 + rng.random(200))
strings = (np.arange(200) > 60).astype(float) * (1 + rng.random(200))
violin = (np.arange(200) > 80).astype(float) * (1 + rng.random(200))
bass = (np.arange(200) > 40).astype(float) * (1 + rng.random(200))


def stem(p):
    return {"lane_db": lane(p)}


def test_duplicates_merge_rare_echoes_fold_away_and_a_section_stays():
    stems = {
        "piano": stem(piano), "keys": stem(piano),                 # the same sound under two names -> one row
        "strings": stem(strings), "violin": stem(violin),
        "viola": stem(violin * 0.98 + 0.01),                     # moves with violin, but a section plays together
        "ukulele": stem(strings * 1.01),                          # an unlikely instrument moving with strings -> folded away
        "bass": stem(bass),
        "flute": stem(np.zeros(200)),                             # silent -> no row
        "harp": stem((np.arange(200) == 5).astype(float)),        # a blip -> no row
    }
    out = B.summarise(stems, 0.5)
    names = [i["name"] for i in out["instruments"]]
    assert names == ["Piano", "Bass", "Strings", "Violin", "Viola"]
    assert out["instruments"][0]["stems"] == ["piano", "keys"]
    assert out["instruments"][2]["first_s"] == 30.5
    hidden = {h["name"]: h["reason"] for h in out["instruments_hidden"]}
    assert set(hidden) == {"Ukulele"} and "moves with Strings" in hidden["Ukulele"]


def test_a_likely_instrument_needs_a_near_perfect_match_before_it_is_called_an_echo():
    guitar = (np.arange(200) > 100).astype(float) * (1 + rng.random(200))
    stems = {"strings": stem(strings), "electric-guitar": stem(strings * 0.7 + guitar * 0.6)}
    names = [i["name"] for i in B.summarise(stems, 0.5)["instruments"]]
    assert "Electric guitar" in names


def test_apply_to_score_writes_rows_into_the_score(tmp_path):
    (tmp_path / "viz.json").write_text(json.dumps({"duration": 100, "pitch": []}))
    grouped = B.apply_to_score(tmp_path, {"step_s": 0.5, "stems": {"piano": stem(piano)}})
    viz = json.loads((tmp_path / "viz.json").read_text())
    assert viz["duration"] == 100 and viz["instruments"][0]["name"] == "Piano"
    assert viz["instruments"] == grouped["instruments"] and viz["instruments_hidden"] == []
