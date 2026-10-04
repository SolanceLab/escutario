# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
import numpy as np

from escutario.entrances import arrangement_changes, detect_entrances, format_entrances_section, level_envelope

SR = 8000
RNG = np.random.default_rng(7)


def tone(dur, freq, amp, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(dur, sr=SR):
    return np.zeros(int(dur * sr), dtype=np.float32)


def floor_noise(dur, amp=1e-4, sr=SR):
    return (amp * RNG.standard_normal(int(dur * sr))).astype(np.float32)


def near(events, t, stem, kind=None, tol=0.8):
    return [e for e in events if e["stem"] == stem and abs(e["t"] - t) <= tol and (kind is None or e["kind"] == kind)]


def test_bass_enters_from_silence_at_known_time():
    bass = np.concatenate([floor_noise(20), tone(20, 80, 0.3)])
    voice = tone(40, 440, 0.2)
    r = detect_entrances({"bass": bass, "vocals": voice}, SR)
    hits = near(r["events"], 20.0, "bass", "enters")
    assert len(hits) == 1, r["events"]
    assert hits[0]["confidence"] == "high"
    assert not [e for e in r["events"] if e["stem"] == "vocals"], "a steady voice must not produce events"


def test_swell_is_rises_not_enters():
    other = np.concatenate([tone(15, 300, 0.03), tone(15, 300, 0.3)])  # +20 dB, never silent
    r = detect_entrances({"other": other, "vocals": tone(30, 440, 0.2)}, SR)
    hits = near(r["events"], 15.0, "other")
    assert hits and hits[0]["kind"] == "rises" and 15 <= hits[0]["change_db"] <= 25


def test_track_leaving_is_reported():
    guitar = np.concatenate([tone(15, 200, 0.2), floor_noise(15)])
    r = detect_entrances({"guitar": guitar, "vocals": tone(30, 440, 0.2)}, SR)
    hits = near(r["events"], 15.0, "guitar", "leaves")
    assert len(hits) == 1


def test_simultaneous_arrivals_form_an_opening():
    drums = np.concatenate([floor_noise(12), (0.2 * RNG.standard_normal(int(12 * SR))).astype(np.float32)])
    piano = np.concatenate([floor_noise(12.5), tone(11.5, 262, 0.2)])
    voice = tone(24, 440, 0.2)
    r = detect_entrances({"drums": drums, "piano": piano, "vocals": voice}, SR)
    assert any(set(o["stems"]) >= {"drums", "piano"} and abs(o["t"] - 12.0) < 1.5 for o in r["openings"]), r["openings"]


def test_slow_crescendo_is_not_an_arrival():
    n = int(60 * SR)
    amp = np.geomspace(0.01, 0.2, n)  # +26 dB over a minute, about 1.3 dB per 3 s
    other = (amp * np.sin(2 * np.pi * 300 * np.arange(n) / SR)).astype(np.float32)
    r = detect_entrances({"other": other, "vocals": tone(60, 440, 0.2)}, SR)
    assert not [e for e in r["events"] if e["stem"] == "other"], r["events"]


def test_single_hit_is_not_a_sustained_arrival():
    drums = floor_noise(30)
    drums[int(15 * SR):int(15.05 * SR)] = 0.8
    r = detect_entrances({"drums": drums, "vocals": tone(30, 440, 0.2)}, SR)
    assert not [e for e in r["events"] if e["stem"] == "drums" and e["kind"] in ("enters", "rises")]


def test_inaudible_arrival_under_a_loud_mix_is_ignored():
    faint = np.concatenate([silence(15), tone(15, 500, 0.001)])  # -60 dBFS, far below the voice
    r = detect_entrances({"guitar": faint, "vocals": tone(30, 440, 0.5)}, SR)
    assert not [e for e in r["events"] if e["stem"] == "guitar"]


def test_empty_and_short_inputs():
    assert detect_entrances({}, SR)["events"] == []
    assert detect_entrances({"a": silence(0.1)}, SR)["events"] == []
    assert level_envelope(silence(0.1), SR).size == 0


def test_card_format():
    bass = np.concatenate([floor_noise(20), tone(20, 80, 0.3)])
    txt = format_entrances_section(detect_entrances({"bass": bass, "vocals": tone(40, 440, 0.2)}, SR))
    assert txt.startswith("ARRIVALS:") and "bass enters" in txt
    assert format_entrances_section({"events": []}) == "ARRIVALS: none detected"


def test_arrangement_change_found_even_when_overall_level_stays_flat():
    # piano alone, then guitar takes over at the same loudness: overall level flat, balance flips
    piano = np.concatenate([tone(20, 262, 0.2), floor_noise(20)])
    guitar = np.concatenate([floor_noise(20), tone(20, 196, 0.2)])
    voice = tone(40, 440, 0.2)
    ch = arrangement_changes({"piano": piano, "guitar": guitar, "vocals": voice}, SR)
    assert ch and abs(ch[0]["t"] - 20.0) <= 0.8, ch[:2]
    stems_moved = {s["stem"] for s in ch[0]["shifts"]}
    assert {"piano", "guitar"} <= stems_moved


def test_no_arrangement_change_in_a_steady_mix():
    ch = arrangement_changes({"piano": tone(30, 262, 0.2), "vocals": tone(30, 440, 0.2)}, SR)
    assert ch == []


def test_same_sound_moving_between_tracks_is_a_relabel_not_an_arrival():
    # the splitter files the opening piano under 'other' for 20 s, then under 'piano'
    sound = tone(40, 262, 0.2)
    other = np.concatenate([sound[: 20 * SR], floor_noise(20)])
    piano = np.concatenate([floor_noise(20), sound[20 * SR:]])
    ch = arrangement_changes({"other": other, "piano": piano, "vocals": tone(40, 440, 0.2)}, SR)
    assert ch and abs(ch[0]["t"] - 20.0) <= 0.8
    assert ch[0]["relabel"] == {"from": "other", "to": "piano", "pair_change_db": ch[0]["relabel"]["pair_change_db"]}
    assert ch[0]["relabel"]["pair_change_db"] <= 1.5


def test_real_takeover_with_a_level_change_is_not_a_relabel():
    piano = np.concatenate([tone(20, 262, 0.05), floor_noise(20)])     # quiet piano leaves
    guitar = np.concatenate([floor_noise(20), tone(20, 196, 0.3)])     # loud guitar arrives: +15 dB
    ch = arrangement_changes({"piano": piano, "guitar": guitar, "vocals": tone(40, 440, 0.1)}, SR)
    assert ch and abs(ch[0]["t"] - 20.0) <= 0.8 and ch[0]["relabel"] is None


def test_balanced_swap_between_different_sounds_is_not_a_relabel():
    # drums (noise) fade out while a voice (tone) comes in at the same combined level
    drums = np.concatenate([(0.2 * RNG.standard_normal(20 * SR)).astype(np.float32), floor_noise(20)])
    voice = np.concatenate([floor_noise(20), tone(20, 440, 0.2 * np.sqrt(2))])
    ch = arrangement_changes({"drums": drums, "vocals": voice, "bass": tone(40, 80, 0.2)}, SR)
    assert ch and abs(ch[0]["t"] - 20.0) <= 0.8 and ch[0]["relabel"] is None
