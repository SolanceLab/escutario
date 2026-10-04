# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""Build web/listening-score.html from score_template.html + viz data + embedded 96k AAC (the published page)."""
import base64, pathlib, sys
here = pathlib.Path(__file__).parent
data = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else str(here / 'viz_data.json')).read_text()
s = (here / 'score_template.html').read_text().replace('__DATA__', data)
for key, song in (('__EMB_FF__', 'forbidden-fruit'), ('__EMB_AP__', 'apparition-x-unethical')):
    s = s.replace(key, base64.b64encode((here / 'audio' / f'{song}-96k.m4a').read_bytes()).decode())
(here / 'listening-score.html').write_text(s)
print(f'{len(s)/1e6:.1f} MB')
