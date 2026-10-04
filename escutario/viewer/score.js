// Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
// Escutário — the Listening Score drawing.
//
// Every function takes a song's score (`d`, the viz.json Escutário wrote for it) and a `view`
// ({ W, zoom, pitchRange, layout }), and reads no page globals, so the page can redraw freely.

export const LANES = (() => {
  const lanes = [
    { key: 'ruler', h: 30, label: '' },
    { key: 'arrive', h: 40, label: 'Arrivals' },
    { key: 'breath', h: 38, label: 'Breath' },
    { key: 'voices', h: 34, label: 'Voices' },
    { key: 'voice', h: 230, label: 'Voice' },
    { key: 'loud', h: 86, label: 'Loudness' },
    { key: 'texture', h: 64, label: 'Texture' },
    { key: 'inst', h: 118, label: 'Instruments' },
  ]
  let top = 0
  for (const l of lanes) { l.y = top; top += l.h }
  return lanes
})()
export const TOTAL_H = LANES.reduce((s, l) => s + l.h, 0)
export const WHAT_LABEL = { words: 'The words', voice: 'The voice', melody: 'The melody', unnamed: "Something I can't name" }
export const PAD = 10
export const STEMS = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']

const C = { page: '#e8e3d4', dim: '#b8b3a4', muted: '#9a9483', faint: '#5a5549', gold: '#BA7517', goldText: '#e09b41', blue: '#378ADD' }
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

export const noteName = (m) => NOTE_NAMES[((Math.round(m) % 12) + 12) % 12] + (Math.floor(Math.round(m) / 12) - 1)
export const fmt = (t) => { const m = Math.floor(t / 60), s = t - m * 60; return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1) }
export const fmtShort = (t) => { const m = Math.floor(t / 60), s = Math.floor(t - m * 60); return m + ':' + (s < 10 ? '0' : '') + s }
export const lane = (k) => LANES.find((l) => l.key === k)
export const HIDEABLE = LANES.filter((l) => l.key !== 'ruler').map((l) => ({ key: l.key, label: l.label }))

/** The lanes to draw, without the hidden ones, stacked from the top; the time ruler always shows. */
export const INST_ROW_H = 15
export const instLaneHeight = (rows) => 24 + Math.max(1, rows) * INST_ROW_H + 4   // 6 rows -> 118, the original lane

export function layoutLanes(hidden = [], instRows = STEMS.length) {
  const off = new Set(hidden)
  let top = 0
  const lanes = LANES.filter((l) => l.key === 'ruler' || !off.has(l.key)).map((l) => {
    const placed = { ...l, y: top, h: l.key === 'inst' ? instLaneHeight(instRows) : l.h }
    top += placed.h
    return placed
  })
  return { lanes, totalH: top }
}
const FULL_LAYOUT = layoutLanes()
const layoutOf = (view) => view.layout || FULL_LAYOUT
const laneIn = (layout, k) => layout.lanes.find((l) => l.key === k)

/** A score with every list the drawing reads present, so an older or partial score never throws. */
export function normaliseScore(raw) {
  const d = raw && typeof raw === 'object' ? raw : {}
  const arr = (v) => (Array.isArray(v) ? v : [])
  const objs = (v) => arr(v).filter((x) => x && typeof x === 'object' && !Array.isArray(x))
  const step = (v, fallback) => (Number.isFinite(v) && v >= 0.001 && v <= 10 ? v : fallback)
  const duration = Number(d.duration)
  const stems = {}
  for (const s of STEMS) stems[s] = arr(d.stems?.[s])
  return {
    ...d,
    duration: Number.isFinite(duration) && duration > 0 && duration <= 3600 ? duration : 0,
    pitch_step: step(d.pitch_step, 0.01),
    pitch: arr(d.pitch),
    level_step: step(d.level_step, 0.1),
    level_vocals: arr(d.level_vocals),
    level_mix: arr(d.level_mix),
    stem_step: step(d.stem_step, 0.25),
    stems,
    breaths: objs(d.breaths),
    silent_gaps: arr(d.silent_gaps).filter((g) => Array.isArray(g) && g.length >= 2),
    falls: objs(d.falls),
    slips: arr(d.slips).filter(Number.isFinite),
    grit: objs(d.grit),
    ring: { ...(d.ring || {}), per_phrase: arr(d.ring?.per_phrase).filter((r) => Array.isArray(r) && r.length >= 3) },
    held: objs(d.held),
    sections: arr(d.sections).filter(Number.isFinite),
    changes: objs(d.changes),
    arrivals: objs(d.arrivals),
    voices: Array.isArray(d.voices) ? d.voices : null,
    voices_step: step(d.voices_step, 0.1),
    riffs: objs(d.riffs),
    vibrato: d.vibrato && typeof d.vibrato === 'object' ? d.vibrato : { with_vibrato: [] },
    loudest_t: Number.isFinite(d.loudest_t) ? d.loudest_t : null,
    // the named instruments found by the 53-stem breakdown, when the song has one; else the six stems are drawn
    instruments: objs(d.instruments)
      .filter((i) => typeof i.name === 'string' && i.name && Array.isArray(i.lane_db))
      .slice(0, 40)
      .map((i) => ({ name: i.name.slice(0, 40), step: step(i.step, 0.5), lane_db: i.lane_db.map((v) => (Number.isFinite(v) ? v : -90)) })),
  }
}

/** Rows of the instruments lane: named instruments when known, else the six separated stems. */
export function instrumentRows(d) {
  if (d.instruments && d.instruments.length) return d.instruments.map((i) => ({ name: i.name, lane: i.lane_db, step: i.step }))
  return STEMS.map((s) => ({ name: s, lane: d.stems[s], step: d.stem_step }))
}

export function pitchRangeOf(d) {
  const v = d.pitch.filter((x) => x != null).sort((a, b) => a - b)
  if (!v.length) return [48, 80]
  const q = (p) => v[Math.floor(p * (v.length - 1))]
  return [Math.floor(q(0.01)) - 2, Math.ceil(q(0.995)) + 2]
}

export function geometry(d, view) {
  const W = view.W
  const dur = d.duration || 1
  const xOf = (t) => PAD + (t / dur) * (W - 2 * PAD)
  const tOf = (x) => Math.max(0, Math.min(dur, ((x - PAD) / (W - 2 * PAD)) * dur))
  const [lo, hi] = view.pitchRange
  const layout = layoutOf(view)
  const V = laneIn(layout, 'voice')
  const yPitch = (m) => (V ? V.y + 14 + (1 - (m - lo) / (hi - lo)) * (V.h - 28) : NaN)
  return { W, xOf, tOf, yPitch, layout, H: layout.totalH, laneOf: (k) => laneIn(layout, k) }
}

export function pitchAt(d, t) {
  const i = Math.round(t / d.pitch_step)
  for (let k = 0; k < 6; k++) for (const j of [i - k, i + k]) { const v = d.pitch[j]; if (v != null) return v }
  return null
}

export function medianPitch(d, a, b) {
  const v = []
  for (let i = Math.floor(a / d.pitch_step); i <= Math.ceil(b / d.pitch_step); i++) { const p = d.pitch[i]; if (p != null) v.push(p) }
  if (!v.length) return null
  v.sort((x, y) => x - y)
  return v[Math.floor(v.length / 2)]
}

/** The lane labels (lane names, octave ticks, stem names), as positioned items for the label column. */
export function labelItems(d, view) {
  const { yPitch, layout, laneOf } = geometry(d, view)
  const items = layout.lanes.filter((l) => l.label).map((l) => ({ kind: 'lane', top: l.y + 6, text: l.label }))
  const [lo, hi] = view.pitchRange
  if (laneOf('voice')) for (let m = Math.ceil(lo / 12) * 12; m <= hi; m += 12) items.push({ kind: 'tick', top: yPitch(m), text: noteName(m) })
  const I = laneOf('inst')
  if (I) instrumentRows(d).forEach((row, i) => items.push({ kind: 'stem', top: I.y + 24 + i * INST_ROW_H + 5, text: row.name }))
  return items
}

export function drawBase(g, d, view) {
  const { W, xOf, yPitch, layout, H, laneOf } = geometry(d, view)
  g.clearRect(0, 0, W, H)

  g.strokeStyle = 'rgba(232,227,212,.10)'; g.lineWidth = 1
  for (const l of layout.lanes.slice(1)) { g.beginPath(); g.moveTo(0, l.y + 0.5); g.lineTo(W, l.y + 0.5); g.stroke() }

  // ruler, section changes, loudest point
  const R = laneOf('ruler')
  const tickEvery = view.zoom >= 8 ? 5 : view.zoom >= 3 ? 10 : 30
  g.font = '300 10px "JetBrains Mono", monospace'; g.fillStyle = C.muted; g.textBaseline = 'middle'
  for (let t = 0; t <= d.duration; t += tickEvery) {
    const x = xOf(t)
    g.fillRect(Math.round(x), R.y + R.h - 7, 1, 7)
    if (x + 30 < W) g.fillText(fmtShort(t), x + 4, R.y + 13)
  }
  g.setLineDash([2, 4]); g.strokeStyle = 'rgba(232,227,212,.16)'
  for (const t of d.sections) { const x = Math.round(xOf(t)) + 0.5; g.beginPath(); g.moveTo(x, R.y + R.h); g.lineTo(x, H); g.stroke() }
  g.setLineDash([])
  if (d.loudest_t != null) {
    const x = xOf(d.loudest_t)
    g.fillStyle = C.goldText; g.beginPath(); g.moveTo(x, R.y + 17); g.lineTo(x + 5, R.y + 23); g.lineTo(x, R.y + 29); g.lineTo(x - 5, R.y + 23); g.closePath(); g.fill()
    g.font = '300 9.5px "JetBrains Mono", monospace'; g.textAlign = x > W - 70 ? 'right' : 'left'
    g.fillText('PEAK', x + (x > W - 70 ? -9 : 9), R.y + 23); g.textAlign = 'left'
  }

  // arrivals: where the balance of the arrangement shifts most
  const A = laneOf('arrive')
  if (A) {
    const maxS = Math.max(0.5, ...d.changes.map((c) => c.strength || 0))
    for (const c of d.changes) {
      const x = xOf(c.t), k = Math.min(1, (c.strength || 0) / maxS), h = 8 + 20 * k
      g.beginPath(); g.moveTo(x, A.y + A.h - 6 - h); g.lineTo(x + 5, A.y + A.h - 6); g.lineTo(x - 5, A.y + A.h - 6); g.closePath()
      if (c.relabel) { g.strokeStyle = 'rgba(232,227,212,.45)'; g.lineWidth = 1; g.stroke() } else { g.fillStyle = `rgba(232,227,212,${(0.25 + 0.6 * k).toFixed(2)})`; g.fill() }
      const lead = (c.shifts || []).filter((sh) => sh.share_change_pct > 0)[0]
      if (lead && !c.relabel && (view.zoom >= 3 || k > 0.8)) {
        g.font = '300 9px "JetBrains Mono", monospace'; g.fillStyle = C.muted; g.textAlign = 'left'; g.textBaseline = 'middle'
        if (x + 70 < W) g.fillText(lead.stem, x + 8, A.y + 12)
      }
    }
    g.textBaseline = 'alphabetic'
  }

  // breath: silent gaps and breath marks
  const B = laneOf('breath')
  if (B) {
    g.strokeStyle = 'rgba(232,227,212,.22)'
    for (const [a, b] of d.silent_gaps) { const x0 = xOf(a), x1 = xOf(b); g.beginPath(); g.moveTo(x0, B.y + B.h - 6); g.lineTo(x0, B.y + B.h - 3); g.lineTo(x1, B.y + B.h - 3); g.lineTo(x1, B.y + B.h - 6); g.stroke() }
    g.font = '500 34px "Cormorant Garamond", serif'; g.textAlign = 'center'
    for (const br of d.breaths) {
      const x = xOf(br.t)
      if (br.c === 'high') { g.fillStyle = C.goldText; g.fillText(',', x, B.y + 26) } else { g.strokeStyle = C.goldText; g.lineWidth = 1; g.globalAlpha = 0.75; g.strokeText(',', x, B.y + 26); g.globalAlpha = 1 }
    }
    g.textAlign = 'left'
  }

  // voices: how many vocal lines sound at once
  const Vs = laneOf('voices')
  if (Vs) {
    if (d.voices) {
      const rowH = 7, base = Vs.y + Vs.h - 5
      for (let lvl = 1; lvl <= 3; lvl++) {
        g.fillStyle = `rgba(232,227,212,${lvl === 1 ? 0.28 : lvl === 2 ? 0.52 : 0.8})`
        let start = null
        for (let i = 0; i <= d.voices.length; i++) {
          const on = i < d.voices.length && d.voices[i] >= lvl
          if (on && start == null) start = i
          if (!on && start != null) { const x0 = xOf(start * d.voices_step), x1 = xOf(i * d.voices_step); g.fillRect(x0, base - lvl * (rowH + 1), Math.max(1, x1 - x0), rowH); start = null }
        }
      }
    }
  }

  // voice: octave lines, held notes, riffs, voice changes, pitch, crying falls
  const V = laneOf('voice')
  if (V) {
    const [lo, hi] = view.pitchRange
    g.strokeStyle = 'rgba(232,227,212,.07)'; g.lineWidth = 1
    for (let m = Math.ceil(lo / 12) * 12; m <= hi; m += 12) { const y = Math.round(yPitch(m)) + 0.5; g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke() }
    for (const h of d.held) {
      const poly = h.src === 'poly'
      const mid = poly ? h.m : medianPitch(d, h.t, h.t + h.d)
      if (mid == null) continue
      const x0 = xOf(h.t), x1 = xOf(h.t + h.d), y = yPitch(mid)
      g.fillStyle = poly ? 'rgba(186,117,23,.09)' : 'rgba(186,117,23,.16)'; g.fillRect(x0, y - 6, Math.max(2, x1 - x0), 12)
      if (poly) g.setLineDash([3, 2])
      g.strokeStyle = poly ? 'rgba(186,117,23,.55)' : 'rgba(186,117,23,.7)'; g.strokeRect(x0 + 0.5, y - 5.5, Math.max(1, x1 - x0 - 1), 11)
      g.setLineDash([])
      if (x1 - x0 > 26) { g.font = '300 9px "JetBrains Mono", monospace'; g.fillStyle = C.goldText; g.fillText(h.note, x0 + 2, y - 9) }
    }
    for (const r of d.riffs) {
      const x0 = xOf(r.start_s), x1 = Math.max(xOf(r.end_s), x0 + 10), yb = V.y + 10
      g.strokeStyle = r.confidence === 'high' ? 'rgba(232,227,212,.85)' : 'rgba(232,227,212,.45)'; g.lineWidth = 1.3
      g.beginPath()
      const waves = Math.max(2, Math.round((x1 - x0) / 7))
      for (let j = 0; j <= waves * 8; j++) { const x = x0 + (x1 - x0) * j / (waves * 8), y = yb + Math.sin(j / 8 * Math.PI * 2) * 3; if (j) g.lineTo(x, y); else g.moveTo(x, y) }
      g.stroke(); g.lineWidth = 1
      if (x1 - x0 > 34) { g.font = '300 9px "JetBrains Mono", monospace'; g.fillStyle = C.muted; g.textAlign = 'left'; g.fillText('riff', x0, yb - 6) }
    }
    g.fillStyle = 'rgba(55,138,221,.55)'
    for (const t of d.slips) g.fillRect(Math.round(xOf(t)), V.y + V.h - 12, 1.5, 9)
    g.fillStyle = 'rgba(232,227,212,.62)'
    const rad = view.zoom >= 3 ? 1.35 : 1.05
    d.pitch.forEach((m, i) => { if (m == null) return; g.beginPath(); g.arc(xOf(i * d.pitch_step), yPitch(m), rad, 0, Math.PI * 2); g.fill() })
    g.strokeStyle = C.goldText; g.lineWidth = 1.6
    for (const f of d.falls) {
      const m0 = medianPitch(d, f.t - 0.25, f.t) ?? pitchAt(d, f.t)
      if (m0 == null) continue
      const x0 = xOf(f.t), x1 = Math.max(xOf(f.t + f.d), x0 + 14), y0 = yPitch(m0), y1 = yPitch(m0 - f.cents / 100) + 10
      g.beginPath(); g.moveTo(x0 - 8, y0 - 3); g.bezierCurveTo(x0 + 2, y0 - 3, x1 - 4, y0, x1, y1); g.stroke()
      g.beginPath(); g.moveTo(x1 - 4, y1 - 4); g.lineTo(x1, y1 + 1); g.lineTo(x1 + 3, y1 - 5); g.stroke()
    }
    g.lineWidth = 1
  }

  // loudness ribbon
  const Lz = laneOf('loud')
  if (Lz) {
    const cy = Lz.y + Lz.h / 2 + 4, half = Lz.h / 2 - 12
    const amp = (db) => Math.pow(Math.max(0, Math.min(1, (db + 42) / 36)), 1.6) * half
    const smooth = (arr, w) => arr.map((_, i) => { let sum = 0, n = 0; for (let k = Math.max(0, i - w); k <= Math.min(arr.length - 1, i + w); k++) { sum += Math.pow(10, arr[k] / 10); n++ } return 10 * Math.log10(sum / n + 1e-12) })
    const lv = smooth(d.level_vocals, 3), lm = smooth(d.level_mix, 3)
    if (lv.length) {
      g.beginPath()
      lv.forEach((db, i) => { const x = xOf(i * d.level_step), a = amp(db); if (i) g.lineTo(x, cy - a); else g.moveTo(x, cy - a) })
      for (let i = lv.length - 1; i >= 0; i--) g.lineTo(xOf(i * d.level_step), cy + amp(lv[i]))
      g.closePath(); g.fillStyle = 'rgba(232,227,212,.30)'; g.fill()
    }
    if (lm.length) {
      g.beginPath(); g.strokeStyle = 'rgba(232,227,212,.30)'
      lm.forEach((db, i) => { const x = xOf(i * d.level_step), a = amp(db); if (i) g.lineTo(x, cy - a); else g.moveTo(x, cy - a) })
      g.stroke()
    }
  }

  // texture: ring bars and strong single-voice grit
  const T = laneOf('texture')
  if (T) {
    const base = T.y + T.h - 8, th = T.h - 22
    for (const [a, b, r] of d.ring.per_phrase) {
      const k = Math.max(0, Math.min(1, (r + 30) / 25)); const x0 = xOf(a), x1 = xOf(b)
      g.fillStyle = `rgba(232,227,212,${0.12 + 0.45 * k})`; g.fillRect(x0, base - th * k, Math.max(1.5, x1 - x0 - 1), th * k)
    }
    const strongGrit = (g2) => g2.kind === 'grit' && g2.c === 'high' && (!d.voices || Math.max(0, ...d.voices.slice(Math.floor(g2.t0 / d.voices_step), Math.floor(g2.t1 / d.voices_step) + 1)) <= 1)
    for (const gr of d.grit.filter(strongGrit)) {
      const x0 = xOf(gr.t0), x1 = Math.max(xOf(gr.t1), x0 + 4)
      g.save(); g.beginPath(); g.rect(x0, T.y + 8, x1 - x0, T.h - 16); g.clip()
      g.strokeStyle = 'rgba(224,155,65,.9)'; g.lineWidth = 1.2
      for (let x = x0 - T.h; x < x1 + T.h; x += 4) { g.beginPath(); g.moveTo(x, T.y + T.h); g.lineTo(x + T.h, T.y); g.stroke() }
      g.restore()
    }
  }

  // instruments: one strip per separated track, brighter is louder
  const I = laneOf('inst')
  if (I) {
    instrumentRows(d).forEach((row, si) => {
      const y = I.y + 24 + si * INST_ROW_H
      row.lane.forEach((db, i) => {
        const k = Math.max(0, Math.min(1, (db + 55) / 45)); if (k <= 0.02) return
        g.fillStyle = `rgba(232,227,212,${(0.06 + 0.7 * k * k).toFixed(3)})`
        const x0 = xOf(i * row.step), x1 = xOf((i + 1) * row.step)
        g.fillRect(x0, y, Math.max(1, x1 - x0 + 0.3), 10)
      })
    })
  }
}

/**
 * The overlay: marks (a fermata over each; a band for a section), the preview (a dashed frame),
 * the stretch being chosen, the cursor and the playhead.
 * overlay = { marks, range, pinned, cursor, playhead }
 */
export function drawOver(g, d, view, overlay) {
  const { W, xOf, yPitch, H: TOTAL, laneOf } = geometry(d, view)
  const hasVoice = Boolean(laneOf('voice'))
  g.clearRect(0, 0, W, TOTAL)
  const line = (t, color, alpha) => { const x = Math.round(xOf(t)) + 0.5; g.strokeStyle = color; g.globalAlpha = alpha; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, TOTAL); g.stroke(); g.globalAlpha = 1 }
  if (overlay.pinned != null) line(overlay.pinned, C.goldText, 0.75)
  if (overlay.cursor != null && overlay.cursor !== overlay.pinned) line(overlay.cursor, C.page, 0.5)
  const live = overlay.range
  if (live && live.b > live.a) {
    const x0 = xOf(live.a), x1 = xOf(live.b)
    g.fillStyle = 'rgba(224,155,65,.10)'; g.fillRect(x0, 30, x1 - x0, TOTAL - 30)
    g.strokeStyle = 'rgba(224,155,65,.55)'; g.strokeRect(x0 + 0.5, 30.5, x1 - x0 - 1, TOTAL - 31)
  }
  for (const mo of overlay.marks || []) {
    const t = Number(mo.t), tEnd = mo.t_end == null ? null : Number(mo.t_end)
    const x = xOf(t)
    if (mo.kind === 'preview') {
      if (tEnd == null) continue
      const x1 = xOf(tEnd)
      g.setLineDash([4, 3]); g.strokeStyle = 'rgba(224,155,65,.65)'; g.lineWidth = 1
      g.strokeRect(x + 0.5, 3.5, x1 - x - 1, 24)
      g.setLineDash([])
      g.font = '300 9px "JetBrains Mono", monospace'; g.fillStyle = C.goldText; g.textBaseline = 'middle'
      if (x1 - x > 60) g.fillText('PREVIEW', x + 6, 15.5)
      g.textBaseline = 'alphabetic'
      continue
    }
    if (tEnd != null && tEnd > t) {
      const x1 = xOf(tEnd)
      g.fillStyle = 'rgba(224,155,65,.13)'; g.fillRect(x, 30, x1 - x, TOTAL - 30)
      g.strokeStyle = 'rgba(224,155,65,.5)'; g.lineWidth = 1; g.beginPath(); g.moveTo(x1 + 0.5, 30); g.lineTo(x1 + 0.5, TOTAL); g.stroke()
      g.beginPath(); g.moveTo(x1 - 6, 17); g.lineTo(x1, 17); g.lineTo(x1, 29); g.strokeStyle = C.goldText; g.lineWidth = 1.6; g.stroke()
    }
    g.strokeStyle = 'rgba(224,155,65,.22)'; g.lineWidth = 6; g.beginPath(); g.moveTo(x, 30); g.lineTo(x, TOTAL); g.stroke()
    g.lineWidth = 1.6; g.strokeStyle = C.goldText; g.beginPath(); g.arc(x, 24, 8, Math.PI, 0); g.stroke()
    g.fillStyle = C.goldText; g.beginPath(); g.arc(x, 21, 1.8, 0, Math.PI * 2); g.fill()
  }
  g.lineWidth = 1
  if (overlay.playhead != null) {
    const pt = overlay.playhead, x = Math.round(xOf(pt)) + 0.5
    g.strokeStyle = 'rgba(55,138,221,.25)'; g.lineWidth = 5; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, TOTAL); g.stroke()
    g.strokeStyle = C.blue; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, TOTAL); g.stroke(); g.lineWidth = 1
    const pm = hasVoice ? pitchAt(d, pt) : null; if (pm != null) { g.fillStyle = C.blue; g.beginPath(); g.arc(x, yPitch(pm), 4, 0, Math.PI * 2); g.fill() }
  }
  if (overlay.cursor != null) {
    const m = hasVoice ? pitchAt(d, overlay.cursor) : null
    if (m != null) { g.fillStyle = C.goldText; g.beginPath(); g.arc(xOf(overlay.cursor), yPitch(m), 3.5, 0, Math.PI * 2); g.fill() }
  }
}

/** What Escutário measured at one second, in words — kept beside a mark so the mark can be read later. */
export function measureAt(d, t) {
  const parts = []
  const m = pitchAt(d, t)
  if (m != null) { const r = Math.round(m), cents = Math.round((m - r) * 100); parts.push(`Voice ${noteName(r)} ${cents >= 0 ? '+' : '−'}${Math.abs(cents)}¢`) } else parts.push('Voice silent')
  const rf = d.riffs.find((r) => t >= r.start_s - 0.1 && t <= r.end_s + 0.1)
  if (rf) parts.push(`Riff · ${rf.notes} notes circling ${rf.low}–${rf.high}${rf.octave_uncertain ? ' (octave uncertain)' : ''}${rf.turns ? `, ${rf.turns} turns` : ''}`)
  const nv = d.voices ? d.voices[Math.floor(t / d.voices_step)] : null
  if (nv != null && nv >= 2) parts.push(`${nv === 2 ? 'Two' : nv === 3 ? 'Three' : 'Four'} voices at once`)
  const lv = d.level_vocals[Math.floor(t / d.level_step)]
  if (lv != null) parts.push(`Voice level ${lv > -89 ? Math.round(lv) + ' dB' : 'silent'}`)
  const playing = d.instruments && d.instruments.length
    ? d.instruments.filter((i) => (i.lane_db[Math.floor(t / i.step)] ?? -90) > -42).map((i) => i.name.toLowerCase())
    : STEMS.filter((s) => (d.stems[s][Math.floor(t / d.stem_step)] ?? -90) > -40)
  parts.push(`Playing ${playing.length ? playing.join(', ') : 'nothing loud'}`)
  const near = (arr, get, win) => arr.find((x) => Math.abs(get(x) - t) <= win)
  const br = near(d.breaths, (x) => x.t, 1.2); if (br) parts.push(`Breath ${fmt(br.t)} · ${br.c === 'high' ? 'confident' : 'likely'}`)
  const hs = d.held.filter((x) => t >= x.t - 0.1 && t <= x.t + x.d + 0.1).sort((a, b) => b.d - a.d)
  if (hs.length) parts.push(`Held ${hs.slice(0, 2).map((h) => `${h.note} ${h.d.toFixed(1)} s${h.src === 'poly' ? ' (through the layers)' : ''}`).join(', ')}`)
  const f = near(d.falls, (x) => x.t, 0.8); if (f) parts.push(`Crying fall ${f.note} ↓ ${Math.round(f.cents)}¢`)
  const gr = d.grit.find((x) => x.kind === 'grit' && x.c === 'high' && t >= x.t0 - 0.4 && t <= x.t1 + 0.4); if (gr) parts.push(`Grit while loud · HNR ${gr.hnr} dB (uncalibrated)`)
  if (near(d.slips, (x) => x, 0.3) != null) parts.push('Voice change')
  if (d.loudest_t != null && Math.abs(t - d.loudest_t) <= 1.5) parts.push('Near peak loudness')
  const vb = (d.vibrato.with_vibrato || []).find((v) => t >= v.start_s && t <= v.start_s + v.dur_s)
  if (vb) parts.push(`Vibrato · ${vb.rate_hz} Hz, ±${vb.extent_cents}¢`)
  const ch = d.changes.find((c) => Math.abs(c.t - t) <= 1.5)
  if (ch && ch.relabel) {
    parts.push(`Same sound, new label · the splitter moved it from ${ch.relabel.from} to ${ch.relabel.to}; nothing new arrived`)
  } else if (ch) {
    const sh = (ch.shifts || []).map((x) => `${x.stem} ${x.share_change_pct > 0 ? '+' : ''}${x.share_change_pct}%`).join(', ')
    parts.push(`Arrangement shifts · ${sh}${ch.arrivals && ch.arrivals.length ? ' · ' + ch.arrivals.join(', ') : ''}`)
  } else {
    const ar = d.arrivals.filter((x) => Math.abs(x.t - t) <= 1.2)
    if (ar.length) parts.push(`Arriving · ${ar.map((x) => (x.kind === 'enters' ? `${x.stem} enters` : `${x.stem} +${Math.round(x.db)} dB`)).join(', ')}`)
  }
  return parts
}

/** What Escutário measured across a stretch, in words. */
export function measureRange(d, a, b) {
  const parts = []
  const avg = (arr, step, t0, t1) => { const v = arr.slice(Math.max(0, Math.floor(t0 / step)), Math.max(0, Math.ceil(t1 / step))); if (!v.length) return null; return 10 * Math.log10(v.reduce((s2, x) => s2 + Math.pow(10, x / 10), 0) / v.length + 1e-12) }
  parts.push(`Section ${fmt(a)}–${fmt(b)} · ${(b - a).toFixed(1)} s`)
  const enters = []
  for (const st of STEMS) {
    const before = avg(d.stems[st], d.stem_step, a - 4, a), after = avg(d.stems[st], d.stem_step, a, a + 4)
    if (before != null && after != null && after - before >= 6 && after > -45) enters.push(before < -55 ? `${st} enters at the start` : `${st} +${Math.round(after - before)} dB at the start`)
    let best = null
    for (let t = a + 2; t <= b - 2; t += 0.5) { const pre = avg(d.stems[st], d.stem_step, t - 3, t), post = avg(d.stems[st], d.stem_step, t, t + 3); if (pre != null && post != null && post - pre >= 8 && post > -45 && (!best || post - pre > best.rise)) best = { t, rise: post - pre } }
    if (best) { const pre = avg(d.stems[st], d.stem_step, best.t - 3, best.t); enters.push(pre < -55 ? `${st} enters at ${fmt(best.t)}` : `${st} rises +${Math.round(best.rise)} dB at ${fmt(best.t)}`) }
  }
  if (enters.length) parts.push(`Arriving · ${enters.join(', ')}`)
  const m0 = avg(d.level_mix, d.level_step, a, Math.min(b, a + 3)), m1 = avg(d.level_mix, d.level_step, Math.max(a, b - 3), b)
  if (m0 != null && m1 != null) parts.push(`Whole song ${Math.round(m0)} → ${Math.round(m1)} dB`)
  if (d.loudest_t != null && d.loudest_t >= a && d.loudest_t <= b) parts.push(`Holds the peak loudness ${fmt(d.loudest_t)}`)
  const vbs = (d.vibrato.with_vibrato || []).filter((v) => v.start_s >= a && v.start_s <= b); if (vbs.length) parts.push(`${vbs.length} vibrato${vbs.length > 1 ? 's' : ''}`)
  const p = d.pitch.slice(Math.floor(a / d.pitch_step), Math.ceil(b / d.pitch_step)).filter((x) => x != null).sort((x, y) => x - y)
  if (p.length > 10) { const q = (fr) => p[Math.floor(fr * (p.length - 1))]; parts.push(`Voice ${noteName(q(0.05))}–${noteName(q(0.95))}, ${Math.round(100 * p.filter((x) => x >= 60).length / p.length)}% above C4`) }
  const within = (t) => t >= a && t <= b
  const br = d.breaths.filter((x) => within(x.t)).length; if (br) parts.push(`${br} breath${br > 1 ? 's' : ''}`)
  const held = d.held.filter((x) => within(x.t)).sort((x, y) => y.d - x.d).slice(0, 3); if (held.length) parts.push(`Held ${held.map((h) => `${h.note} ${h.d.toFixed(1)} s`).join(', ')}`)
  const falls = d.falls.filter((x) => within(x.t)).length; if (falls) parts.push(`${falls} crying fall${falls > 1 ? 's' : ''}`)
  const grit = d.grit.filter((x) => within(x.t0) && x.kind === 'grit' && x.c === 'high').length; if (grit) parts.push(`${grit} strong grit spot${grit > 1 ? 's' : ''} (uncalibrated)`)
  if (d.voices) { const vs = d.voices.slice(Math.floor(a / d.voices_step), Math.ceil(b / d.voices_step)); const sung = vs.filter((v) => v > 0).length; if (sung) parts.push(`Voices · two or more ${Math.round(100 * vs.filter((v) => v >= 2).length / sung)}% of sung time, three or more ${Math.round(100 * vs.filter((v) => v >= 3).length / sung)}%`) }
  const rfs = d.riffs.filter((r) => r.end_s >= a && r.start_s <= b); if (rfs.length) parts.push(`${rfs.length} riff${rfs.length > 1 ? 's' : ''} · ${rfs.map((r) => `${fmt(r.start_s)} ${r.notes} notes ${r.low}–${r.high}`).join(', ')}`)
  const vc = d.slips.filter(within).length; if (vc) parts.push(`${vc} voice change${vc > 1 ? 's' : ''}`)
  const secs = d.sections.filter(within); if (secs.length) parts.push(`Song changes section at ${secs.map(fmt).join(', ')}`)
  return parts.slice(0, 20).map((x) => x.slice(0, 300))
}
