// Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
// The local viewer: pick a song Escutário has heard, play it with the score moving, mark where it reaches you.
// Talks only to the viewer's own server (escutario/view.py). Every piece of text from a song or a mark is
// written with textContent, never as HTML.

import {
  fmt, drawBase, drawOver, WHAT_LABEL, measureRange, layoutLanes, geometry, measureAt, HIDEABLE, fmtShort, normaliseScore,
  pitchRangeOf, labelItems,
} from './score.js'
import { nowAt, wordLines } from './now.js'

const $ = (id) => document.getElementById(id)
const r10 = (t) => Math.round(t * 10) / 10
const HIDDEN_KEY = 'escutario-hidden-lanes'
const ZOOMS = [[1, 'Whole song'], [3, 'Closer'], [8, 'Phrase by phrase']]

/** A DOM element with text and attributes; children may be elements or strings (always text). */
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag)
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue
    if (k === 'class') el.className = v
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v)
    else el.setAttribute(k, v === true ? '' : String(v))
  }
  for (const k of kids.flat()) if (k != null && k !== false) el.append(k instanceof Node ? k : document.createTextNode(String(k)))
  return el
}

async function api(path, init) {
  const r = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) } })
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`)
  return body
}

const hashSlug = () => { try { return decodeURIComponent(location.hash.slice(1)) } catch { return '' } }

const readHidden = () => {
  try {
    const v = JSON.parse(localStorage.getItem(HIDDEN_KEY) || '[]')
    return Array.isArray(v) ? v.filter((k) => HIDEABLE.some((l) => l.key === k)) : []
  } catch { return [] }
}

const S = {
  slug: null, song: null, score: null, lines: [], words: null, marks: [],
  zoom: 1, hidden: readHidden(), follow: true, view: null,
  cursor: null, pinned: null, range: null, panel: null, drag: null, status: '',
}
const audio = $('audio')

// ---------------------------------------------------------------- songs

async function start() {
  let songs = []
  try { songs = (await api('/api/songs')).songs || [] } catch (e) { $('title').textContent = `Could not reach the viewer: ${e.message}` ; return }
  if (!songs.length) { $('title').textContent = 'No songs yet'; $('empty').hidden = false; return }
  const sel = $('song')
  sel.replaceChildren(...songs.map((s) => h('option', { value: s.slug }, `${s.title}${s.artist ? ' · ' + s.artist : ''}`)))
  const wanted = hashSlug()
  sel.value = songs.some((s) => s.slug === wanted) ? wanted : songs[0].slug
  sel.addEventListener('change', () => { location.hash = sel.value })
  window.addEventListener('hashchange', () => {
    const slug = hashSlug()
    if (slug !== S.slug && songs.some((s) => s.slug === slug)) { sel.value = slug; loadSong(slug) }
  })
  buildControls()
  await loadSong(sel.value)
}

async function loadSong(slug) {
  audio.pause()
  Object.assign(S, { slug, cursor: null, pinned: null, range: null, panel: null, status: '' })
  const data = await api(`/api/songs/${encodeURIComponent(slug)}`)
  if (S.slug !== slug) return   // another song was chosen while this one loaded
  S.song = data.song
  S.score = normaliseScore(data.score)
  S.words = data.words && Array.isArray(data.words.lines) ? data.words : null
  S.lines = wordLines(S.words, null)
  S.marks = Array.isArray(data.marks) ? data.marks : []
  audio.src = `/api/songs/${encodeURIComponent(slug)}/audio`
  audio.preload = 'metadata'
  $('audio-error').hidden = true
  $('audio-slug').textContent = slug
  $('play').disabled = !S.song.has_audio
  for (const id of ['score-section', 'marks-section']) $(id).hidden = false
  $('clock').textContent = fmt(0)
  renderHeader(); layout(); renderMarks(); renderWords(); renderPanel(); renderRangeBar(); renderNow(0); updateMarkButton()
  setReadout(['Press play, or click the score to jump there.'])
  $('scroller').scrollLeft = 0
}

function renderHeader() {
  const { song, score } = S
  $('title').textContent = song.title
  $('artist').textContent = song.artist || ''
  const userMarks = S.marks.filter((m) => m.kind !== 'preview').length
  const bpm = Number(score.tempo?.bpm ?? score.tempo)
  const facts = [fmtShort(score.duration), typeof score.key === 'object' ? score.key?.key : score.key, Number.isFinite(bpm) && bpm > 0 ? `${Math.round(bpm)} bpm` : null,
    `${userMarks} mark${userMarks === 1 ? '' : 's'}`, song.has_audio ? null : 'audio not on this machine']
  $('facts').replaceChildren(...facts.filter((x) => x && typeof x === 'string').map((x) => h('span', {}, x)))
  $('dur').textContent = fmtShort(score.duration)
}

// ---------------------------------------------------------------- controls

function buildControls() {
  $('zooms').replaceChildren(...ZOOMS.map(([z, label]) => h('button', {
    class: 'pill', 'aria-pressed': String(S.zoom === z), 'data-z': z,
    onclick: () => {
      const sc = $('scroller')
      const centre = S.view ? geometry(S.score, S.view).tOf(sc.scrollLeft + sc.clientWidth / 2) : 0
      S.zoom = z
      for (const b of $('zooms').children) b.setAttribute('aria-pressed', String(Number(b.dataset.z) === z))
      layout()
      scrollToTime(S.pinned ?? centre)
    },
  }, label)))
  $('show').replaceChildren(h('span', { class: 'meta' }, 'Show'), ...HIDEABLE.map((l) => h('button', {
    class: 'lane', 'aria-pressed': String(!S.hidden.includes(l.key)),
    onclick: (e) => {
      S.hidden = S.hidden.includes(l.key) ? S.hidden.filter((k) => k !== l.key) : [...S.hidden, l.key]
      try { localStorage.setItem(HIDDEN_KEY, JSON.stringify(S.hidden)) } catch { /* private window: lasts this visit */ }
      e.currentTarget.setAttribute('aria-pressed', String(!S.hidden.includes(l.key)))
      layout()
    },
  }, l.label)))
  $('play').addEventListener('click', togglePlay)
  $('follow').addEventListener('change', (e) => { S.follow = e.target.checked })
  $('mark').addEventListener('click', openMark)
  const over = $('over')
  over.addEventListener('pointermove', onPointerMove)
  over.addEventListener('pointerdown', onPointerDown)
  over.addEventListener('pointerup', onPointerUp)
  over.addEventListener('pointerleave', () => { S.cursor = S.pinned; paintOver() })
  over.addEventListener('pointercancel', () => { S.drag = null; S.cursor = S.pinned; paintOver() })
  audio.addEventListener('play', () => { S.pinned = null; setPlayIcon(true); updateMarkButton(); if (!ticking) { ticking = true; requestAnimationFrame(tick) } })
  audio.addEventListener('pause', () => { setPlayIcon(false); paintOver(); updateMarkButton() })
  audio.addEventListener('ended', () => { setPlayIcon(false); paintOver() })
  audio.addEventListener('error', () => { if (audio.src) $('audio-error').hidden = false })
  document.addEventListener('keydown', onKey)
  new ResizeObserver(() => { if (S.score) layout() }).observe($('scroller'))
}

function setPlayIcon(on) {
  $('play-icon').setAttribute('d', on ? 'M6.5 4.5h4v15h-4zM13.5 4.5h4v15h-4z' : 'M7 4.5v15l13-7.5z')
  $('play').setAttribute('aria-label', on ? 'Pause the song' : 'Play the song')
}

// ---------------------------------------------------------------- drawing

function layout() {
  const { score } = S
  if (!score) return
  const width = $('scroller').clientWidth || 760
  S.view = { W: Math.round(Math.max(width, 760) * S.zoom), zoom: S.zoom, pitchRange: pitchRangeOf(score), layout: layoutLanes(S.hidden, score.instruments.length || 6) }
  const { W } = S.view, H = S.view.layout.totalH
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  $('canvases').style.width = W + 'px'; $('canvases').style.height = H + 'px'
  for (const c of [$('base'), $('over')]) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); c.style.width = W + 'px'; c.style.height = H + 'px' }
  const lab = $('labels'); lab.style.height = H + 'px'
  lab.replaceChildren(...labelItems(score, S.view).map((it) => {
    const el = h('span', { class: `l-${it.kind}`, title: it.kind === 'stem' ? it.text : null }, it.text)
    el.style.top = it.top + 'px'
    return el
  }))
  const g = $('base').getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0)
  drawBase(g, score, S.view)
  paintOver()
}

function overlay() {
  const p = S.panel
  const playing = !audio.paused
  return {
    marks: S.marks,
    range: S.range || (p && p.t_end == null && playing ? { a: p.t, b: audio.currentTime } : p?.t_end != null ? { a: p.t, b: p.t_end } : null),
    pinned: S.pinned, cursor: S.cursor,
    playhead: playing || audio.currentTime > 0 ? audio.currentTime : null,
  }
}

function paintOver() {
  if (!S.score || !S.view) return
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const g = $('over').getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0)
  drawOver(g, S.score, S.view, overlay())
}

function scrollToTime(t) {
  const sc = $('scroller')
  if (S.view) sc.scrollLeft = Math.max(0, geometry(S.score, S.view).xOf(t) - sc.clientWidth / 2)
}

function setReadout(parts) {
  $('readout').replaceChildren(...parts.map((r, i) => h('span', { class: i === 0 && /^\d+:/.test(r) ? 't' : null }, r)))
}

// ---------------------------------------------------------------- playback

let lastReadout = 0
let ticking = false
function tick(now) {
  if (audio.paused) { ticking = false; return }
  const t = audio.currentTime
  $('clock').textContent = fmt(t)
  renderNow(t)
  paintOver()
  if (now - lastReadout > 150) { lastReadout = now; S.cursor = null; setReadout([fmt(t), ...measureAt(S.score, t)]) }
  if (S.follow && S.view) {
    const sc = $('scroller'), x = geometry(S.score, S.view).xOf(t)
    if (x < sc.scrollLeft + sc.clientWidth * 0.15 || x > sc.scrollLeft + sc.clientWidth * 0.7) sc.scrollLeft = Math.max(0, x - sc.clientWidth * 0.3)
  }
  requestAnimationFrame(tick)
}

async function togglePlay() {
  if (!S.song?.has_audio) return
  if (!audio.paused) { audio.pause(); return }
  if (S.pinned != null && Math.abs(audio.currentTime - S.pinned) > 0.2) seek(S.pinned)
  try { await audio.play() } catch { $('audio-error').hidden = false }
}

function seek(t) {
  try { audio.currentTime = t } catch { /* not loaded yet: play() starts from the pinned second */ }
  $('clock').textContent = fmt(t)
  renderNow(t)
}

function jumpTo(t) {
  S.pinned = t; S.cursor = t
  seek(t); scrollToTime(t); paintOver(); updateMarkButton()
  setReadout([fmt(t), ...measureAt(S.score, t)])
  $('score-section').scrollIntoView({ block: 'start', behavior: 'smooth' })
}

// ---------------------------------------------------------------- pointing

const timeAt = (e) => geometry(S.score, S.view).tOf(e.clientX - $('over').getBoundingClientRect().left)
const xIn = (e) => e.clientX - $('over').getBoundingClientRect().left

function onPointerMove(e) {
  if (!S.score) return
  const t = timeAt(e)
  S.cursor = t
  if (S.drag && Math.abs(xIn(e) - S.drag.x) > 8) {
    S.drag.moved = true
    if (!S.drag.touch) S.range = { a: Math.min(S.drag.t, t), b: Math.max(S.drag.t, t) }
  }
  if (audio.paused) setReadout([fmt(t), ...measureAt(S.score, t)])
  paintOver()
}

function onPointerDown(e) {
  if (!S.score || (e.pointerType === 'mouse' && e.button !== 0)) return
  // a finger swipes the score sideways; a mouse drag chooses a stretch
  S.drag = { x: xIn(e), t: timeAt(e), moved: false, touch: e.pointerType === 'touch' }
  if (!S.drag.touch) $('over').setPointerCapture?.(e.pointerId)
}

function onPointerUp() {
  const drag = S.drag
  S.drag = null
  if (!drag || (drag.touch && drag.moved)) return
  if (drag.moved && S.range && S.range.b - S.range.a >= 0.5) {
    S.pinned = S.range.a; seek(S.range.a)
    setReadout([`${fmt(S.range.a)}–${fmt(S.range.b)}`, 'Stretch chosen · Mark this section'])
  } else {
    S.range = null; S.pinned = drag.t; seek(drag.t)
    setReadout([fmt(drag.t), ...measureAt(S.score, drag.t)])
  }
  renderRangeBar(); updateMarkButton(); paintOver()
}

function onKey(e) {
  const tag = e.target?.tagName || ''
  if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return
  if (e.key === ' ' && tag !== 'BUTTON') { e.preventDefault(); togglePlay() } else if (e.key === 'm' || e.key === 'M') {
    if (!S.panel) { if (!$('mark').disabled) { e.preventDefault(); openMark() } } else if (S.panel.t_end == null) { e.preventDefault(); endHere() }
  }
}

// ---------------------------------------------------------------- now window

function renderNow(t) {
  const hasWords = S.lines.length > 0
  const userMarks = S.marks.filter((m) => m.kind !== 'preview')
  $('now').hidden = !hasWords && !userMarks.length
  if ($('now').hidden) return
  const now = nowAt(S.lines, S.marks, t)
  $('now-clock').textContent = fmt(t)
  const lines = []
  if (!hasWords) lines.push(h('p', { class: 'faint' }, 'No words heard for this song. The words step (escutario.words) adds them.'))
  else {
    if (now.prev) lines.push(h('button', { class: 'n-prev', onclick: () => jumpTo(now.prev.t) }, now.prev.text))
    if (now.current) {
      lines.push(h('p', { class: 'n-cur' }, now.current.text))
      if (now.current.voice || now.current.delivery) lines.push(h('p', { class: 'n-how' }, [now.current.voice, now.current.delivery].filter(Boolean).join(' · ')))
    } else lines.push(h('p', { class: 'n-wait' }, now.next ? 'Music before the next line…' : 'No more lines.'))
    if (now.next) lines.push(h('button', { class: 'n-next', onclick: () => jumpTo(now.next.t) }, `${fmt(now.next.t)} · ${now.next.text}`))
  }
  $('now-lines').replaceChildren(...lines)
  $('now-marks').replaceChildren(...(now.marks.length ? now.marks.map((m) => h('button', { class: 'n-mark', onclick: () => jumpTo(Number(m.t)) },
    h('span', { class: 't' }, `${fmt(Number(m.t))}${m.t_end != null ? '–' + fmt(Number(m.t_end)) : ''}${(m.what || []).length ? ' · ' + m.what.map((w) => WHAT_LABEL[w] || w).join(', ') : ''}`),
    m.words ? h('span', { class: 'w' }, m.words) : null)) : [h('p', { class: 'faint' }, 'None at this moment.')]))
}

// ---------------------------------------------------------------- marking

function updateMarkButton() {
  const can = Boolean(S.score && (S.range || !audio.paused || audio.currentTime > 0 || S.pinned != null))
  $('mark').disabled = !can || Boolean(S.panel)
  $('mark').textContent = S.range ? 'Mark this section' : 'Mark'
}

function markSource() {
  if (!audio.paused || (audio.currentTime > 0 && S.pinned == null)) return audio.currentTime
  return S.pinned
}

function openMark() {
  if (!S.score) return
  if (S.range) { audio.pause(); S.panel = { t: r10(S.range.a), t_end: r10(S.range.b), what: new Set(), words: '' } } else {
    const t = markSource()
    if (t == null) return
    S.panel = { t: r10(t), t_end: null, what: new Set(), words: '' }
    S.pinned = r10(t)
  }
  S.status = ''
  renderPanel(); renderRangeBar(); updateMarkButton(); paintOver()
  $('mark-words')?.focus()
}

function endHere() {
  const p = S.panel
  if (!p) return
  const t = !audio.paused ? audio.currentTime : (S.cursor ?? audio.currentTime ?? p.t)
  if (t <= p.t + 0.3) { S.status = 'The end needs to come after the start'; renderPanel(); return }
  p.t_end = r10(t)
  audio.pause()
  renderPanel(); paintOver()
}

function nudge(which, dt) {
  const p = S.panel
  if (!p) return
  if (which === 'start') {
    p.t = Math.max(0, Math.min(S.score.duration, r10(p.t + dt)))
    if (p.t_end != null && p.t_end <= p.t) p.t_end = r10(p.t + 1)
    S.pinned = p.t; seek(p.t); scrollToTime(p.t)
  } else if (p.t_end != null) p.t_end = Math.max(r10(p.t + 0.5), Math.min(S.score.duration, r10(p.t_end + dt)))
  renderPanel(); paintOver()
}

async function saveMark() {
  const p = S.panel
  if (!p || p.saving) return
  const what = [...p.what]
  if (!what.length) { S.status = 'Choose what moved you first'; renderPanel(); return }
  const isSection = p.t_end != null && p.t_end > p.t
  const body = {
    t: p.t, t_end: isSection ? p.t_end : null, what, words: p.words.trim() || null,
    measured: isSection ? measureRange(S.score, p.t, p.t_end) : measureAt(S.score, p.t),
  }
  S.status = 'Keeping…'; p.saving = true; renderPanel()
  try {
    const { mark } = await api(`/api/songs/${encodeURIComponent(S.slug)}/marks`, { method: 'POST', body: JSON.stringify(body) })
    S.marks = [...S.marks, mark].sort((a, b) => Number(a.t) - Number(b.t))
    S.panel = null; S.range = null; S.status = ''
    renderPanel(); renderRangeBar(); renderMarks(); renderHeader(); updateMarkButton(); paintOver(); renderNow(audio.currentTime)
  } catch (e) { p.saving = false; S.status = `Could not keep it (${e.message})`; renderPanel() }
}

function renderPanel() {
  const box = $('panel'), p = S.panel
  box.hidden = !p
  if (!p) { box.replaceChildren(); return }
  const time = (v) => h('span', { class: 'big' }, v)
  const pill = (label, onclick, extra = {}) => h('button', { class: 'pill', onclick, ...extra }, label)
  box.replaceChildren(
    h('div', { class: 'row' }, h('span', { class: 'meta', style: null }, 'From'),
      pill('−1s', () => nudge('start', -1), { 'aria-label': 'Start one second earlier' }), time(fmt(p.t)),
      pill('+1s', () => nudge('start', 1), { 'aria-label': 'Start one second later' })),
    h('div', { class: 'row' }, h('span', { class: 'meta' }, 'To'),
      pill('−1s', () => nudge('end', -1), { disabled: p.t_end == null, 'aria-label': 'End one second earlier' }),
      time(p.t_end == null ? (!audio.paused ? 'playing…' : '—') : fmt(p.t_end)),
      pill('+1s', () => nudge('end', 1), { disabled: p.t_end == null, 'aria-label': 'End one second later' }),
      h('button', { class: 'cta', onclick: endHere }, p.t_end == null ? 'End here' : 'Move end to here'),
      p.t_end != null ? pill('Just the moment', () => { p.t_end = null; renderPanel(); paintOver() }) : h('span', { class: 'hint' }, 'Leave it open for a single moment, or press End here where the section ends.')),
    h('fieldset', { class: 'row', style: null },
      h('legend', { class: 'meta' }, 'What moved you · choose any'),
      ...Object.entries(WHAT_LABEL).map(([k, label]) => h('button', {
        class: 'pill', 'aria-pressed': String(p.what.has(k)),
        onclick: (e) => { if (p.what.has(k)) p.what.delete(k); else p.what.add(k); e.currentTarget.setAttribute('aria-pressed', String(p.what.has(k))) },
      }, label))),
    h('label', { for: 'mark-words', class: 'meta' }, 'In your own words · optional'),
    (() => { const ta = h('textarea', { id: 'mark-words', maxlength: 2000, placeholder: 'What happened in you here?' }); ta.value = p.words; ta.addEventListener('input', () => { p.words = ta.value }); return ta })(),
    h('div', { class: 'row' },
      h('button', { class: 'cta', onclick: saveMark, disabled: p.saving }, `Keep this ${p.t_end != null ? 'section' : 'moment'}`),
      pill('Cancel', () => { S.panel = null; S.range = null; renderPanel(); renderRangeBar(); updateMarkButton(); paintOver() }),
      h('span', { class: 'meta', role: 'status' }, S.status)),
  )
  const fs = box.querySelector('fieldset'); fs.style.border = '0'; fs.style.padding = '0'; fs.style.margin = '0'
}

function renderRangeBar() {
  const bar = $('range-bar')
  bar.hidden = !(S.range && !S.panel)
  if (bar.hidden) { bar.replaceChildren(); return }
  bar.replaceChildren(h('span', { class: 'meta' }, `${fmt(S.range.a)}–${fmt(S.range.b)} chosen`),
    h('button', { class: 'pill', onclick: () => { S.range = null; renderRangeBar(); updateMarkButton(); paintOver() } }, 'Clear'))
}

// ---------------------------------------------------------------- lists

function renderMarks() {
  const marks = S.marks.filter((m) => m.kind !== 'preview')
  $('marks-count').textContent = `${marks.length} in this song`
  $('marks').replaceChildren(...(marks.length ? marks.map(markRow) : [h('li', { style: null }, h('p', { class: 'faint' }, 'No marks yet. Play the song and press Mark where it reaches you, or drag across the score to mark a section.'))]))
}

function markRow(m) {
  const t = Number(m.t), tEnd = m.t_end == null ? null : Number(m.t_end)
  const body = h('div', { class: 'grid' },
    h('div', { class: 'chips' }, ...(m.what || []).map((w) => h('span', { class: 'chip' }, WHAT_LABEL[w] || w))),
    m.words ? h('p', { class: 'm-words' }, m.words) : null,
    Array.isArray(m.measured) && m.measured.length ? h('p', { class: 'm-heard' }, h('b', {}, 'Escutário heard'), ` · ${m.measured.join(' · ')}`) : null)
  body.style.display = 'grid'; body.style.gap = '8px'; body.style.minWidth = '0'
  let armed = false
  const removeBtn = h('button', { class: 'link' }, 'Remove')
  removeBtn.addEventListener('click', async () => {
    if (!armed) { armed = true; removeBtn.textContent = 'Remove for good?'; removeBtn.classList.add('armed'); setTimeout(() => { armed = false; removeBtn.textContent = 'Remove'; removeBtn.classList.remove('armed') }, 4000); return }
    try {
      await api(`/api/songs/${encodeURIComponent(S.slug)}/marks/${encodeURIComponent(m.id)}`, { method: 'DELETE' })
      S.marks = S.marks.filter((x) => x.id !== m.id); renderMarks(); renderHeader(); paintOver(); renderNow(audio.currentTime)
    } catch (e) { removeBtn.textContent = `Not removed (${e.message})` }
  })
  const editBtn = h('button', { class: 'link', onclick: () => {
    const ta = h('textarea', { 'aria-label': 'Your words for this mark', maxlength: 2000 }); ta.value = m.words || ''
    const save = h('button', { class: 'cta', onclick: async () => {
      try {
        const { mark } = await api(`/api/songs/${encodeURIComponent(S.slug)}/marks/${encodeURIComponent(m.id)}`, { method: 'PATCH', body: JSON.stringify({ words: ta.value.trim() || null }) })
        S.marks = S.marks.map((x) => (x.id === mark.id ? mark : x)); renderMarks(); renderNow(audio.currentTime)
      } catch (e) { save.textContent = `Not saved (${e.message})` }
    } }, 'Save')
    body.replaceChildren(ta, h('div', { class: 'row' }, save, h('button', { class: 'pill', onclick: renderMarks }, 'Cancel')))
    ta.focus()
  } }, 'Edit words')
  return h('li', {},
    h('button', { class: 'jump', onclick: () => jumpTo(t) }, fmt(t), tEnd != null ? h('br') : null, tEnd != null ? `– ${fmt(tEnd)}` : null),
    body,
    h('div', { class: 'acts' }, editBtn, removeBtn))
}

function renderWords() {
  const has = S.lines.length > 0
  $('words-section').hidden = !has
  if (!has) return
  $('words-by').textContent = `heard by ${S.words.model || 'a model'} · not measured`
  $('words').replaceChildren(...S.lines.map((line) => h('li', {},
    h('button', { class: 'jump', onclick: () => jumpTo(line.t) }, fmt(line.t)),
    (() => {
      const d = h('div', {}, h('p', { class: 'w-text' }, line.text), h('p', { class: 'n-how' }, [line.voice, line.delivery].filter(Boolean).join(' · ')))
      d.style.display = 'grid'; d.style.gap = '4px'; d.style.gridColumn = 'span 2'; d.style.minWidth = '0'
      return d
    })())))
  const reading = typeof S.words.interpretation === 'string' && S.words.interpretation.trim()
  $('reading').hidden = !reading
  if (reading) $('reading-text').textContent = S.words.interpretation
}

start()
