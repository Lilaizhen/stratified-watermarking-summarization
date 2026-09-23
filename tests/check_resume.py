"""GPU integration test: simulate a partial batch without modifying a real run."""
import json,subprocess,sys,tempfile
from pathlib import Path
root=Path('runs/smoke')
with tempfile.TemporaryDirectory(prefix='watermark_resume_') as name:
    out=Path(name);d=out/'nw';d.mkdir()
    meta=json.loads((root/'nw/metadata.json').read_text())
    (d/'metadata.json').write_text(json.dumps(meta))
    lines=(root/'nw/generations.jsonl').read_text().splitlines()
    (d/'generations.jsonl').write_text('\n'.join(lines[:3])+'\n{"interrupted":')
    (out/'variants.json').write_text(json.dumps([{'name':'nw','method':'none'}]))
    (out/'splits.json').write_text(json.dumps({'test':meta['selected']}))
    subprocess.run([sys.executable,'src/generate.py','--base','configs/base.json','--variants',str(out/'variants.json'),'--splits',str(out/'splits.json'),'--split','test','--output-root',str(out)],check=True)
    actual=[json.loads(x)['token_ids'] for x in (d/'generations.jsonl').open()]
    expected=[json.loads(x)['token_ids'] for x in lines]
    assert actual==expected
    print('PASS: incomplete batch/torn JSONL resumed with identical tokens, no duplicates.')
