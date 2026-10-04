// Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
// Escutário — what is happening at one second of the song: the line being sung (with the one before
// and after), and the marks that hold that second. Drives the "now" window under the score.

export const MOMENT_HOLD_S = 5      // a moment mark stays in the window this long after its second
export const LINE_HOLD_S = 12       // a sung line stops counting as "now" this long after it starts

/** Heard words as {t, voice, text, delivery, published}, in song order, from facts.words rows and the lyrics check. */
export function wordLines(words, lyricsCheck) {
  const diffs = new Map((Array.isArray(lyricsCheck?.differences) ? lyricsCheck.differences : [])
    .map((d) => [Number(d.t), d]))
  return (Array.isArray(words?.lines) ? words.lines : [])
    .map((row) => {
      const [t, voice, text, delivery] = Array.isArray(row) ? row : [row?.t, row?.voice, row?.text, row?.delivery]
      const d = diffs.get(Number(t))
      return {
        t: Number(t), voice: voice || '', text: String(text ?? ''), delivery: delivery || '',
        published: d ? (d.published || null) : undefined,   // undefined: matched or unchecked; null: not in the published lyrics
      }
    })
    .filter((l) => Number.isFinite(l.t) && l.text)
    .sort((a, b) => a.t - b.t)
}

export function nowAt(lines, marks, t) {
  const time = Number(t) || 0
  let i = -1
  for (let k = 0; k < lines.length; k++) {
    if (lines[k].t <= time) i = k
    else break
  }
  // a line is 'now' until the next one starts, but never longer than LINE_HOLD_S: a long gap is music, not the old line
  const current = i >= 0 && time - lines[i].t < LINE_HOLD_S ? lines[i] : null
  const next = lines[i + 1] || null
  const prev = i >= 1 ? lines[i - 1] : null
  const active = (Array.isArray(marks) ? marks : [])
    .filter((m) => m && m.kind !== 'preview')
    .filter((m) => {
      const a = Number(m.t)
      const b = m.t_end == null ? null : Number(m.t_end)
      return b != null ? a <= time && time <= b : a <= time && time < a + MOMENT_HOLD_S
    })
    .sort((x, y) => Number(x.t) - Number(y.t))
  return { prev: current ? prev : lines[i] || null, current, next, marks: active }
}
