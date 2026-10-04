# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The listening card — what another room of the House receives when Escutário hears a clip.

A compact, timestamped text summary built from the score entry that `listen.py` produces
(the same data the Listening Score page draws). Numbers, not moods: the card reports what the
sound did and how sure the ear is; whoever reads it does the feeling.
"""

from __future__ import annotations

CARD_MAX_ITEMS = 6        # longest list shown per line (arrivals, riffs, held notes, breaths)
HELD_MIN_S = 1.5          # held notes worth naming on the card
CARD_MAX_CHARS = 7900     # the House API refuses cards over 8000 characters
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _t(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    m = int(seconds // 60)
    return f"{m}:{seconds - 60 * m:04.1f}"


def _share(counts: list[int], at_least: int) -> int:
    sung = [c for c in counts if c > 0]
    return round(100 * sum(1 for c in sung if c >= at_least) / len(sung)) if sung else 0


def words_lines(words: dict | None, budget: int, skipped: str | None = None) -> list[str]:
    """The words section: what a model heard sung, checked against the ear, within `budget` characters."""
    if not words:
        return [f"WORDS: not heard — {skipped}"] if skipped else []
    chk = words.get("check") or {}
    head = (f"WORDS & DELIVERY (heard by {words.get('model') or 'a model'} from the recording, not measured; lyrics can be misheard) — "
            f"checked against the ear: a voice was measured at {chk.get('with_measured_voice', 0)} of {chk.get('lines', 0)} lines")
    if chk.get("sustain_claims"):
        head += f"; {chk.get('sustain_with_held_note', 0)} of {chk['sustain_claims']} lines called sustained sit on a measured held note"
    if not chk.get("verified"):
        head += "; UNVERIFIED: the model's timing disagrees with the ear, so treat these words as unconfirmed"
    out = [head]
    interp = (words.get("interpretation") or "").strip()
    tail = [f"INTERPRETATION (the model's reading of the performance, not a measurement): {interp}"] if interp else []
    room = budget - sum(len(x) + 1 for x in out + tail)
    lines = words.get("lines") or []
    for i, ln in enumerate(lines):
        row = f"{_t(ln['t'])} {ln.get('voice', 'unknown')} — {ln['text']}"
        if ln.get("delivery"):
            row += f" · {ln['delivery']}"
        if ln.get("ear_heard_voice") is False:
            row += " [no voice measured here]"
        more = f"… {len(lines) - i} more lines in facts.words"
        if len(row) + 1 > room - (len(more) + 1):
            out.append(more)
            break
        out.append(row)
        room -= len(row) + 1
    return out + tail


def build_card(entry: dict, *, title: str = "", artist: str = "", source: str = "", notes: list[str] | None = None,
               words: dict | None = None, words_skipped: str | None = None) -> str:
    e = entry or {}
    lines: list[str] = []
    head = " — ".join(x for x in (title, artist) if x) or "Untitled clip"
    lines.append(f"ESCUTÁRIO · {head}")
    if source:
        lines.append(f"source: {source}")

    key, tempo = e.get("key") or {}, e.get("tempo") or {}
    facts = [f"length {_t(e.get('duration'))}"]
    if key.get("key"):
        facts.append(f"key {key['key']} ({key.get('level') or key.get('confidence')})")
    if tempo.get("bpm"):
        facts.append(f"tempo {round(tempo['bpm'])} bpm" + (" (could be half/double)" if tempo.get("ambiguous") else ""))
    if e.get("sections"):
        facts.append("sections change at " + ", ".join(_t(s) for s in e["sections"][:CARD_MAX_ITEMS]))
    lines.append("SONG: " + " · ".join(facts))

    changes = [c for c in (e.get("changes") or []) if not c.get("relabel")]
    if changes:
        items = []
        for c in sorted(changes, key=lambda c: -c.get("strength", 0))[:CARD_MAX_ITEMS]:
            ups = [f"{s['stem']} +{s['share_change_pct']}%" for s in c.get("shifts", []) if s.get("share_change_pct", 0) > 0]
            what = ", ".join(c.get("arrivals") or []) or ", ".join(ups[:2]) or "balance shifts"
            items.append(f"{_t(c['t'])} {what}")
        lines.append("ARRIVALS (biggest arrangement changes): " + " | ".join(sorted(items)))

    voices = e.get("voices") or []
    if voices:
        lines.append(f"VOICES: two or more at once {_share(voices, 2)}% of sung time, three or more {_share(voices, 3)}%")

    riffs = e.get("riffs") or []
    if riffs:
        lines.append("RIFFS (cengkok): " + " | ".join(
            f"{_t(r['start_s'])}–{_t(r['end_s'])} {r['notes']} notes {r['low']}–{r['high']}"
            + (" (octave uncertain)" if r.get("octave_uncertain") else "") + f" [{r['confidence']}]"
            for r in sorted(sorted(riffs, key=lambda r: (r['confidence'] != 'high', -r['notes']))[:CARD_MAX_ITEMS], key=lambda r: r['start_s'])))

    held = sorted((h for h in (e.get("held") or []) if h.get("d", 0) >= HELD_MIN_S), key=lambda h: -h["d"])
    if held:
        lines.append("HELD NOTES: " + " | ".join(
            f"{_t(h['t'])} {h['note']} {h['d']:.1f}s" + (" (in the layers)" if h.get("src") == "poly" else "")
            for h in sorted(held[:CARD_MAX_ITEMS], key=lambda h: h["t"])))

    vib = e.get("vibrato") or {}
    if vib.get("judged"):
        found = vib.get("with_vibrato") or []
        lines.append(f"VIBRATO: {len(found)} of {vib['judged']} held notes waver" + (
            " — " + " | ".join(f"{_t(v['start_s'])} {NOTE_NAMES[v['midi'] % 12]}{v['midi'] // 12 - 1} {v['rate_hz']} Hz ±{v['extent_cents']}¢" for v in found[:CARD_MAX_ITEMS])
            if found else " — sung straight"))

    breaths = e.get("breaths") or []
    if breaths:
        lines.append(f"BREATH: {len(breaths)} heard ({sum(1 for b in breaths if b.get('c') == 'high')} confident) at "
                     + ", ".join(_t(b["t"]) for b in breaths[:CARD_MAX_ITEMS + 2])
                     + (f" · longest continuous phrase {e['longest_phrase_s']:.1f}s" if e.get("longest_phrase_s") else ""))

    ring, tilt = e.get("ring") or {}, e.get("tilt") or {}
    texture = []
    if ring.get("overall") is not None:
        texture.append(f"2–4 kHz ring {ring['overall']} dB ({ring.get('label')})")
    if tilt.get("alpha") is not None:
        texture.append(f"alpha ratio {tilt['alpha']} dB ({tilt.get('alpha_label')})")
    if e.get("dynamic_range") is not None:
        texture.append(f"voice dynamic range {round(e['dynamic_range'])} dB")
    if texture:
        lines.append("VOICE TEXTURE: " + " · ".join(texture))

    falls = e.get("falls") or []
    if falls:
        lines.append("CRYING FALLS: " + " | ".join(f"{_t(f['t'])} {f['note']} ↓{round(f['cents'])}¢" for f in falls[:CARD_MAX_ITEMS]))

    closing = ["LIMITS: measurements, not feelings. Grit is uncalibrated; quiet background voices are undercounted; "
               "breaths are unchecked by ear; thresholds were tuned on two songs."]
    closing += [f"note: {n}"[:400] for n in notes or []]
    base = len("\n".join(lines + closing)) + 1
    lines += words_lines(words, CARD_MAX_CHARS - base, words_skipped) if (words or words_skipped) else [WORDS_HINT]
    return "\n".join(lines + closing)[:CARD_MAX_CHARS]


WORDS_HINT = "WORDS: not asked for — escutario_words sends this recording to Google once for the sung words, their delivery and a reading"


def add_words_to_card(card: str, words: dict) -> str:
    """Put the words section into an existing card, replacing any earlier words, keeping everything else."""
    body, sep, closing = card.partition("\nLIMITS:")
    if not sep:
        body, closing = card, ""
    kept = []
    for ln in body.split("\n"):
        if ln.startswith(("WORDS", "INTERPRETATION")):
            break   # the words section always runs from its header to LIMITS
        kept.append(ln)
    tail = ("LIMITS:" + closing) if sep else ""
    base = len("\n".join(kept + ([tail] if tail else []))) + 1
    out = kept + words_lines(words, CARD_MAX_CHARS - base) + ([tail] if tail else [])
    return "\n".join(out)[:CARD_MAX_CHARS]
