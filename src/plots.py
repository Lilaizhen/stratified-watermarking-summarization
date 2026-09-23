"""Reconstruct report metrics and plots from compact per-article evidence."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from analysis import threshold,auc,select_targets


def rebuild_metrics(output='outputs/archived'):
    """Validate archived records and recompute metrics, without drawing figures."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=json.loads(Path('archived/records.json').read_text())
    calibration=json.loads(Path('archived/calibration.json').read_text())
    negatives=json.loads(Path('archived/negatives.json').read_text())
    def z(rows):return np.array([r['z'] if r['z'] is not None else -np.inf for r in rows])
    rows=[]
    for name,item in records.items():
        cfg=item['config'];stages=item['stages'];g=cfg['gamma'];key=str(g)
        ids=[r['id'] for r in stages['basic']['per_sample']]
        assert ids==[r['id'] for r in negatives[key]['scores']]
        assert not set(ids)&{r['id'] for r in calibration[key]['scores']}
        assert len(ids)==500 and len(calibration[key]['scores'])==200
        assert len({v['source_sha256'] for v in stages.values()})==1
        for stage in ['quality','alignscore']:assert ids==[r['id'] for r in stages[stage]['per_sample']]
        scores=stages['detection']['scores_by_gamma'][key]
        assert ids==[r['id'] for r in scores]
        cut=threshold(z(calibration[key]['scores']));neg=z(negatives[key]['scores']);pos=z(scores)
        quality=stages['quality']['per_sample'];ppl=math.exp(sum(r['nll'] for r in quality)/sum(r['ppl_tokens'] for r in quality))
        assert abs(ppl-stages['quality']['ppl'])<1e-8
        row=dict(name=name,method=cfg['method'],delta=cfg.get('delta'),rho=cfg.get('rho'),gamma=g,ppl=ppl,threshold=cut,fpr=float(np.mean(neg>cut)),tpr=None if name=='nw' else float(np.mean(pos>cut)),auc=None if name=='nw' else auc(pos,neg))
        for metric in ['rouge1_fmeasure','rouge2_fmeasure','rougeL_fmeasure']:
            row[metric]=float(np.mean([r[metric] for r in stages['basic']['per_sample']]))
        vals=[r['alignscore_nli_sp'] for r in stages['alignscore']['per_sample'] if r['alignscore_nli_sp'] is not None]
        row['alignscore_nli_sp']=float(np.mean(vals));rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    selected=select_targets(frame);selected.to_csv(out/'target_comparison.csv',index=False)
    gamma=frame[(frame.method=='kgw')&frame.delta.eq(2.5)].sort_values('gamma')
    gamma.to_csv(out/'gamma_comparison.csv',index=False)
    print(f'Reconstructed {len(frame)} configurations; independent thresholds, PPL and ROUGE verified.')
    return frame,selected,gamma


def rebuild(output='outputs/archived'):
    """Command-line convenience: reconstruct metrics and export all figures."""
    import matplotlib.pyplot as plt
    frame,selected,gamma=rebuild_metrics(output)
    out=Path(output)
    main=frame[(frame.gamma==.5)&((frame.method!='headmass')|frame.rho.eq(.98))]
    soft=main[main.method=='kgw'].sort_values('delta');strat=main[main.method=='headmass'].sort_values('delta')
    nw=frame[frame.name=='nw'].iloc[0];hard=frame[frame.name=='hard'].iloc[0]
    def finish(fig,name):
        for axis in fig.axes:
            label=axis.get_ylabel()
            if label:
                axis.set_ylabel('');axis.text(0,1.04,label,transform=axis.transAxes,ha='left',fontsize=9)
        fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=160,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3));ax.plot(soft.delta,100*soft.tpr,'o-',label='Soft');ax.axhline(100*hard.tpr,ls=':',label='Hard',color='red');ax.set(xlabel=r'Bias $\delta$',ylabel='TPR (%) ↑');ax.legend();finish(fig,'bias_detection')
    metrics=[('ppl','PPL ↓'),('rougeL_fmeasure','ROUGE-L ↑'),('alignscore_nli_sp','AlignScore ↑')]
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,(m,label) in zip(axes,metrics):
        ax.plot(soft.delta,soft[m],'o-',label='Soft');ax.axhline(nw[m],ls='--',color='gray',label='Unwatermarked');ax.axhline(hard[m],ls=':',color='red',label='Hard');ax.set(xlabel=r'Bias $\delta$',ylabel=label)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.12));finish(fig,'bias_quality')
    fig,ax=plt.subplots(figsize=(7,3))
    for rho in [.9,.95,.98]:
        v=frame[(frame.method=='headmass')&frame.rho.eq(rho)].sort_values('delta');ax.plot(v.delta,v.tpr*100,'o-',label=f'ρ={rho}')
    ax.set(xlabel=r'Bias $\delta$',ylabel='TPR (%) ↑');ax.legend();finish(fig,'rho_detection')
    allmetrics=[('rouge1_fmeasure','ROUGE-1 ↑'),('rouge2_fmeasure','ROUGE-2 ↑'),('rougeL_fmeasure','ROUGE-L ↑'),('ppl','PPL ↓'),('alignscore_nli_sp','AlignScore ↑')]
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for ax,(m,label) in zip(axes.flat,allmetrics):
        for v,name in [(soft,'Soft'),(strat,'Stratified (ρ=0.98)')]:ax.plot(100*v.tpr,v[m],'o-',label=name,ms=3)
        ax.axhline(nw[m],ls='--',color='gray',label='Unwatermarked');ax.scatter([100*hard.tpr],[hard[m]],color='red',marker='D',label='Hard');ax.set(xlabel='TPR (%) →',ylabel=label)
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center');finish(fig,'quality_detection')
    head=pd.read_csv('archived/head_mass.csv');head=head[(head.group=='all')&head.method.isin(['NW','W-soft'])].sort_values('delta')
    fig,ax=plt.subplots(figsize=(7,3))
    for r in [90,95,98]:ax.plot(head.delta,100*head[f'head{r}_after_mean'],'o-',label=f'{r}% head')
    ax.set(xlabel=r'Bias $\delta$',ylabel='Original-head probability mass (%)');ax.legend();finish(fig,'head_mass')
    return frame,selected,gamma

if __name__=='__main__':rebuild()
