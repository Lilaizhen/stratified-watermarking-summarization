"""Notebook orchestration; fresh subprocess per model, resumable generation/evaluation."""
import json,subprocess,sys
from pathlib import Path


def run(mode='demo',n=16,n_calibration=200,quality=True,output='runs/demo'):
    if mode not in ['demo','main','full']:raise ValueError(mode)
    split=json.loads(Path('configs/splits.json').read_text())
    if mode=='demo':
        if not 1<=n<=500 or not 1<=n_calibration<=200:raise ValueError('Invalid sample counts')
        split={'test':split['test'][:n],'calibration':split['calibration'][:n_calibration]}
    if set(r['id'] for r in split['test'])&set(r['id'] for r in split['calibration']):raise ValueError('Split overlap')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    protocol={'mode':mode,'splits':split,'base':json.loads(Path('configs/base.json').read_text()),'variants':json.loads(Path(f'configs/{mode}.json').read_text())}
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=protocol:
        raise ValueError('Run protocol differs; use a new output directory')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'splits.json').write_text(json.dumps(split,indent=2))
    def call(script,*args):subprocess.run([sys.executable,'src/'+script,*args],check=True)
    for which,variants in [('calibration','calibration'),('test',mode)]:
        call('generate.py','--base','configs/base.json','--variants',f'configs/{variants}.json','--splits',str(root/'splits.json'),'--split',which,'--output-root',str(root))
    for stage in ['basic','detection']+(['quality','alignscore'] if quality else []):
        call('evaluate.py','--root',str(root),'--stage',stage)
    from analysis import analyze
    return analyze(root)
