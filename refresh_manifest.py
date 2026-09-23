"""Refresh portable input hashes; exclude notebook/bootstrap to avoid a commit cycle."""
from pathlib import Path
import json, hashlib
root=Path(__file__).resolve().parent
files=[]
for pattern in ['src/*.py','tests/*.py','configs/**/*.json','archived/*']:
    files.extend(root.glob(pattern))
files += [root/n for n in ['README.md','USAGE.md','requirements.txt','report.pdf','VERIFICATION.json']]
manifest={}
for p in sorted(set(files)):
    name=str(p.relative_to(root));content=p.read_bytes()
    if name=='configs/focused/alignscore.json':
        value=json.loads(content)
        for key in ['checkpoint','backbone_snapshot']: value.pop(key,None)
        content=(json.dumps(value,indent=2)+'\n').encode()
    manifest[name]=hashlib.sha256(content).hexdigest()
(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
