"""Calibrate on independent negatives; evaluate all methods on shared articles."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def threshold(scores,alpha=.01):
    """Ascending ceil((n+1)(1-alpha)) order statistic, with a strict > rule."""
    n=len(scores)
    if not n:raise ValueError('Empty calibration scores')
    k=math.ceil((n+1)*(1-alpha))
    return float(np.sort(scores)[k-1]) if k<=n else float('inf')


def auc(pos,neg):
    return float(np.mean([(np.mean(p>neg)+.5*np.mean(p==neg)) for p in pos]))


def scores(record,gamma):
    return np.array([r['z'] if r['z'] is not None else -np.inf for r in record['scores_by_gamma'][str(gamma)]])


def analyze(root):
    root=Path(root)
    def read(name,file):return json.loads((root/name/file).read_text())
    cal=read('unwatermarked_calibration','detection.json');neg=read('nw','detection.json')
    calids=[r['id'] for r in cal['scores_by_gamma']['0.5']]
    ids=[r['id'] for r in neg['scores_by_gamma']['0.5']]
    if set(calids)&set(ids):raise ValueError('Calibration and evaluation overlap')
    result=[]
    for d in sorted(root.iterdir()):
        if d.name=='unwatermarked_calibration' or not (d/'detection.json').exists():continue
        cfg=read(d.name,'metadata.json')['config'];g=cfg['gamma'];det=read(d.name,'detection.json')
        if [r['id'] for r in det['scores_by_gamma'][str(g)]]!=ids:raise ValueError('Evaluation article mismatch')
        cut=threshold(scores(cal,g));positive=scores(det,g);negative=scores(neg,g)
        row=dict(name=d.name,method=cfg['method'],delta=cfg.get('delta'),rho=cfg.get('rho'),gamma=g,n=len(ids),calibration_n=len(calids),threshold=cut,fpr=float(np.mean(negative>cut)),tpr=None if cfg['method']=='none' else float(np.mean(positive>cut)),auc=None if cfg['method']=='none' else auc(positive,negative))
        for stage in ['basic','quality','alignscore']:
            if (d/f'{stage}.json').exists():
                value=read(d.name,f'{stage}.json');row.update(value['means'])
                if stage=='quality':row['ppl']=value['ppl']
        result.append(row)
    frame=pd.DataFrame(result);frame.to_csv(root/'metrics.csv',index=False)
    if len(calids)<200:print('DEMO calibration only: these rates are not a report-level 1% FPR estimate.')
    return frame


def select_targets(frame):
    rows=[]
    for target in [.90,.95,.99]:
        for method in ['kgw','headmass']:
            mask=(frame.method==method)&(frame.gamma==.5)&(frame.tpr>=target)
            if method=='headmass':mask &= frame.rho.eq(.98)
            sub=frame[mask].dropna(subset=['ppl']).sort_values(['ppl','delta'])
            if len(sub):rows.append(dict(target=target,**sub.iloc[0].to_dict()))
    return pd.DataFrame(rows)
