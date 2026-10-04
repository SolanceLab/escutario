# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
import json, numpy as np, sys
sys.path.insert(0,'.')
from escutario.audio import load_mono
TMP='/tmp'
def db(x):
    r=np.sqrt(np.mean(np.square(x,dtype=np.float64))) if x.size else 0
    return round(float(20*np.log10(r)),1) if r>1e-9 else -90.0
out={}
for song in ['forbidden-fruit','apparition-x-unethical']:
    d=f'out/{song}'
    p=np.load(f'{d}/pitch.npz'); t=p['times']; f0=p['f0']; cf=p['conf']
    dur=float(json.load(open(f'{d}/attune.json'))['music']['duration_s'])
    step=0.05; pitch=[]
    for i in range(int(dur/step)):
        m=(t>=i*step)&(t<(i+1)*step)&(f0>0)&(cf>=0.6)
        pitch.append(round(float(np.median(69+12*np.log2(f0[m]/440))),2) if m.sum()>=2 else None)
    sr=4000
    voc,_=load_mono(f'{d}/stems/vocals.wav',sr); mix,_=load_mono(f'{d}/source.wav',sr)
    hop=int(0.1*sr)
    lvl_v=[db(voc[i:i+hop]) for i in range(0,len(voc)-hop+1,hop)]
    lvl_m=[db(mix[i:i+hop]) for i in range(0,len(mix)-hop+1,hop)]
    stems={}
    for s in ['vocals','drums','bass','guitar','piano','other']:
        x,_=load_mono(f'{d}/stems/{s}.wav',sr); h=int(0.5*sr)
        stems[s]=[db(x[i:i+h]) for i in range(0,len(x)-h+1,h)]
    b=json.load(open(f'{d}/breath.json')); v=json.load(open(f'{TMP}/voice-{song}.json')); a=json.load(open(f'{d}/attune.json'))
    mu=a['music']; sg=a['singing']
    out[song]=dict(
      duration=dur, pitch_step=step, pitch=pitch, level_step=0.1, level_vocals=lvl_v, level_mix=lvl_m, stem_step=0.5, stems=stems,
      breaths=[dict(t=round(x['start_s'],2),d=x['duration_s'],c=x['confidence'],rel=x['level_db_rel_phrase']) for x in b['breaths']],
      silent_gaps=[[round(g['start_s'],2),round(g['end_s'],2)] for g in b['silent_gaps']],
      longest_phrase_s=b['longest_phrase_s'],
      falls=[dict(t=round(f['start_s'],2),d=f['duration_s'],note=f['start_note'],cents=f['drop_cents'],c=f['confidence']) for f in v['falls']],
      flips=[dict(t=round(x.get('time_s',x.get('start_s',0)),2),frm=x.get('from_note'),to=x.get('to_note'),cents=x.get('jump_cents')) for x in v['breaks']],
      slips=[round(e['time_s'],2) for e in v['voice_changes']['events']],
      grit=[dict(t0=round(g['start_s'],2),t1=round(g['end_s'],2),hnr=g['hnr_db'],kind=g['kind'],c=g['confidence']) for g in v['grit']['regions']],
      ring=dict(overall=v['ring']['overall_db'],label=v['ring']['label'],per_phrase=[[round(x['start_s'],2),round(x['end_s'],2),x['ring_db']] for x in v['ring']['per_phrase'] if x.get('ring_db') is not None]),
      tilt=dict(alpha=v['tilt']['alpha_ratio_db'],alpha_label=v['tilt']['alpha_label'],h1h2=v['tilt'].get('h1_h2_db'),h1h2_label=v['tilt'].get('h1_h2_label')),
      register=dict(low=v['register']['low_note'],high=v['register']['high_note'],top_share=v['register']['top_third_share'],top_from=v['register']['top_third_from_note'],longest=v['register']['longest_top_note']),
      hnr_median=v['grit']['hnr_median_db'], multi_voice=v.get('multi_voice_suspected'),
      held=[dict(t=n['start_s'],d=n['dur_s'],note=n['note_name']) for n in sg['notes'] if n['dur_s']>=0.9],
      key=mu['key'], tempo=dict(bpm=mu['tempo']['bpm'],c=mu['tempo']['confidence']), sections=mu['section_changes'], loudest_t=mu['energy']['loudest_t'],
      dynamic_range=sg['dynamics']['dynamic_range_db'],
    )
    print(song, len(pitch), len(lvl_v), {k:len(x) for k,x in stems.items()}, 'breaths',len(out[song]['breaths']),'held',len(out[song]['held']),'loudest',mu['energy']['loudest_t'])
json.dump(out, open(f'{TMP}/viz_data.json','w'), separators=(',',':'))
import os; print(os.path.getsize(f'{TMP}/viz_data.json')//1024,'KB')
