# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
from escutario.voicelines import count_voices, detect_riffs, remove_octave_ghosts

NOTE = {"C3": 48, "D3": 50, "E3": 52, "F#3": 54, "G#3": 56, "A3": 57, "G#4": 68, "C4": 60, "D4": 62, "E4": 64, "F4": 65, "G4": 67, "C5": 72}


def n(name, start, dur, sal=0.7):
    return {"start_s": start, "dur_s": dur, "midi": NOTE[name], "salience": sal}


# Anne's cengkok in the mashup at 0:30.8-0:36.5, exactly as basic-pitch reported it
CENGKOK = [
    n("A3", 30.77, 0.31, .72), n("G#3", 30.98, 0.34, .69), n("F#3", 31.29, 0.52, .62), n("E3", 32.04, 0.22, .52),
    n("F#3", 32.46, 0.13, .43), n("G#3", 32.60, 0.72, .74), n("G#4", 32.60, 0.35, .40), n("A3", 33.08, 0.14, .62),
    n("F#3", 33.30, 0.87, .70), n("A3", 34.34, 0.14, .67), n("G#3", 34.67, 0.23, .70), n("F#3", 34.87, 0.50, .62),
    n("E3", 35.36, 1.33, .66), n("D3", 36.30, 0.16, .44),
]


def at(counts, t, step=0.1):
    return counts[int(round(t / step))]


def test_octave_overtone_is_removed():
    kept = remove_octave_ghosts(CENGKOK)
    assert not any(k["midi"] == NOTE["G#4"] for k in kept)
    assert any(k["midi"] == NOTE["G#3"] and abs(k["start_s"] - 32.60) < 1e-6 for k in kept)


def test_annes_cengkok_is_one_voice():
    counts = count_voices(CENGKOK, 40)
    assert max(counts[int(30 / 0.1):int(37 / 0.1)]) == 1, counts[300:370]
    assert at(counts, 32.8) == 1


def test_annes_cengkok_is_a_riff():
    riffs = detect_riffs(CENGKOK)
    assert riffs, "the cengkok should be found"
    assert any(r["start_s"] <= 33.1 and r["end_s"] >= 33.2 for r in riffs), riffs


def test_two_real_voices_holding_a_third_apart_count_as_two():
    notes = [n("C4", 10, 3.0), n("E4", 10.2, 2.6), n("C3", 20, 2.0)]
    counts = count_voices(notes, 25)
    assert at(counts, 11.5) == 2 and at(counts, 21) == 1


def test_a_real_second_singer_an_octave_below_at_full_strength_is_kept():
    notes = [n("C4", 10, 3.0, .7), n("C3", 10, 3.0, .65)]  # doubled an octave apart, nearly as strong
    counts = count_voices(notes, 15)
    assert at(counts, 11.5) == 2


def test_long_held_notes_are_not_a_riff():
    notes = [n("C4", 0, 1.2), n("E4", 1.25, 1.2), n("G4", 2.5, 1.2), n("C5", 3.75, 1.2)]
    assert detect_riffs(notes) == []


def test_fast_wide_leaps_are_not_a_riff():
    notes = [n("C3", 0, .15), n("C4", .2, .15), n("C3", .4, .15), n("C4", .6, .15), n("C3", .8, .15)]
    assert detect_riffs(notes) == []


def test_fast_stepwise_run_is_a_riff_with_its_contour():
    notes = [n("E4", 0, .12), n("D3", .5, .12)]  # unrelated
    run = [n(x, 5 + i * 0.15, 0.14) for i, x in enumerate(["G4", "F4", "E4", "F4", "G4", "F4", "E4", "D4"])]
    riffs = detect_riffs(notes + run)
    assert len(riffs) == 1 and riffs[0]["notes"] == 8 and riffs[0]["turns"] >= 2 and riffs[0]["low"] == "D4", riffs


def test_a_slow_stepwise_melody_is_not_a_riff():
    notes = [n(x, i * 1.0, 0.9) for i, x in enumerate(["C4", "D4", "E4", "D4", "C4", "D4"])]
    assert detect_riffs(notes) == []


def test_a_straight_scale_without_turns_is_not_a_riff():
    notes = [n(x, i * 0.2, 0.18) for i, x in enumerate(["C4", "D4", "E4", "F4", "G4"])]
    assert detect_riffs(notes) == []


def test_three_held_voices_count_as_three_not_four():
    notes = [n("C4", 10, 3.0), n("E4", 10, 3.0), n("G4", 10, 3.0)]
    counts = count_voices(notes, 15)
    assert at(counts, 11.5) == 3


# Faouzia's riff on "baby", 3:26.9-3:31, as basic-pitch reported it: the B is reported an octave high
BABY = [
    {"start_s": 207.38, "dur_s": 0.23, "midi": 64, "salience": .6}, {"start_s": 207.62, "dur_s": 0.26, "midi": 63, "salience": .6},
    {"start_s": 208.16, "dur_s": 0.33, "midi": 61, "salience": .6}, {"start_s": 208.60, "dur_s": 0.69, "midi": 71, "salience": .55},
    {"start_s": 209.22, "dur_s": 0.16, "midi": 64, "salience": .6}, {"start_s": 209.42, "dur_s": 0.29, "midi": 63, "salience": .6},
    {"start_s": 209.72, "dur_s": 0.46, "midi": 61, "salience": .6}, {"start_s": 210.41, "dur_s": 0.19, "midi": 71, "salience": .5},
]


def test_riff_survives_a_tracker_octave_slip():
    riffs = detect_riffs(BABY)
    assert riffs and riffs[0]["start_s"] <= 207.4 and riffs[0]["end_s"] >= 210.1, riffs


def test_a_weaker_lower_voice_is_not_deleted_as_an_overtone_of_a_stronger_upper_one():
    notes = [n("C4", 10, 2.0, .8), n("C3", 10, 2.0, .4)]
    kept = remove_octave_ghosts(notes)
    assert any(k["midi"] == NOTE["C3"] for k in kept)


def test_a_cluster_spanning_a_third_counts_as_two_voices():
    notes = [n("C4", 10, 3.0), n("D4", 10, 3.0), n("E4", 10, 3.0)]
    assert at(count_voices(notes, 15), 11.5) == 2
