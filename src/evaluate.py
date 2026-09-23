"""Separate-process evaluation: ROUGE, keyed detection, corpus PPL, AlignScore."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def load(directory):
    rows=[json.loads(line) for line in (directory/'generations.jsonl').open()]
    done=json.loads((directory/'complete.json').read_text())
    if len(rows)!=done['records']: raise ValueError('Incomplete generation')
    if done['source_sha256']!=hashlib.sha256((directory/'generations.jsonl').read_bytes()).hexdigest():
        raise ValueError('Generation hash changed')
    return rows


def save(directory,stage,rows,extra=None):
    keys=set().union(*(r.keys() for r in rows))-{'id'}
    means={k:statistics.mean(r[k] for r in rows if r.get(k) is not None) for k in keys
           if all(r.get(k) is None or isinstance(r[k],(int,float)) for r in rows)
           and any(r.get(k) is not None for r in rows)}
    result={'source_sha256':hashlib.sha256((directory/'generations.jsonl').read_bytes()).hexdigest(),
            'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'per_sample':rows,'means':means,**(extra or {})}
    target=directory/f'{stage}.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(target)
    print(stage,directory.name,len(rows),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['basic','detection','quality','alignscore'],required=True)
    a=p.parse_args();root=Path(a.root)
    directories=[]
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d/'complete.json').exists(): continue
        if d.name=='unwatermarked_calibration' and a.stage!='detection': continue
        if (d/f'{a.stage}.json').exists():
            old=json.loads((d/f'{a.stage}.json').read_text())
            if old['source_sha256']!=hashlib.sha256((d/'generations.jsonl').read_bytes()).hexdigest():
                raise ValueError('Cached evaluator source changed')
            continue
        directories.append(d)
    if not directories:return
    import torch
    torch.set_num_threads(4)
    import nltk
    nltk.data.path.insert(0,str(Path('data/nltk').resolve()))
    base=json.loads(Path('configs/base.json').read_text())
    if a.stage=='basic':
        from rouge_score import rouge_scorer
        rouge=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
        for d in directories:
            out=[]
            for r in load(d):
                scores=rouge.score('\n'.join(nltk.sent_tokenize(r['reference'])), '\n'.join(nltk.sent_tokenize(r['summary'])))
                out.append({'id':r['id'],**{f'{k}_fmeasure':v.fmeasure*100 for k,v in scores.items()}})
            save(d,a.stage,out)
    elif a.stage=='detection':
        from transformers import AutoTokenizer,AutoConfig,WatermarkLogitsProcessor
        from token_detector import score_tokens
        if not torch.cuda.is_available():raise RuntimeError('CUDA RNG required for report-compatible keyed colors')
        tok=AutoTokenizer.from_pretrained(base['model'],revision=base['revision'])
        vocab=AutoConfig.from_pretrained(base['model'],revision=base['revision']).vocab_size
        for d in directories:
            rows=load(d);cfg=json.loads((d/'metadata.json').read_text())['config']
            gammas=[.1,.25,.5,.75,.9] if cfg['method']=='none' else [cfg['gamma']]
            encoded={r['id']:tok.encode(r['summary'],add_special_tokens=False) for r in rows}
            scores={}
            for g in gammas:
                proc=WatermarkLogitsProcessor(vocab,'cuda',greenlist_ratio=g,bias=0,hashing_key=base['hash_key'],seeding_scheme='lefthash',context_width=1)
                scores[str(g)]=[{'id':r['id'],**score_tokens(encoded[r['id']],proc,set(tok.all_special_ids))} for r in rows]
            save(d,a.stage,[{'id':r['id']} for r in rows],{'scores_by_gamma':scores})
    elif a.stage=='quality':
        from transformers import AutoTokenizer,AutoModelForCausalLM
        cfg=json.loads(Path('configs/m1_assets.json').read_text())['models'][0]
        tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
        model=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision'],torch_dtype=torch.float32).to('cuda').eval()
        for d in directories:
            rows=load(d);out=[]
            with torch.inference_mode():
                for start in range(0,len(rows),8):
                    batch=rows[start:start+8]
                    seq=[[tok.bos_token_id]+tok.encode(r['summary'],add_special_tokens=False) for r in batch]
                    maximum=max(map(len,seq))
                    if maximum>model.config.max_position_embeddings:raise ValueError('Summary exceeds PPL context window')
                    if maximum==1:
                        out.extend({'id':r['id'],'nll':0.,'ppl_tokens':0} for r in batch);continue
                    ids=torch.tensor([v+[tok.eos_token_id]*(maximum-len(v)) for v in seq],device='cuda')
                    mask=torch.tensor([[1]*len(v)+[0]*(maximum-len(v)) for v in seq],device='cuda')
                    labels=ids.clone();labels[mask==0]=-100
                    logits=model(ids,attention_mask=mask).logits
                    nll=torch.nn.functional.cross_entropy(logits[:,:-1].transpose(1,2),labels[:,1:],ignore_index=-100,reduction='none').sum(-1)
                    out.extend({'id':r['id'],'nll':n,'ppl_tokens':len(v)-1} for r,n,v in zip(batch,nll.tolist(),seq))
            total=sum(r['ppl_tokens'] for r in out)
            save(d,a.stage,out,{'ppl':math.exp(sum(r['nll'] for r in out)/total) if total else None,'model':cfg})
    else:
        from alignscore_local import AlignScoreLocal
        model=AlignScoreLocal()
        for d in directories:
            rows=[{'id':r['id'],'alignscore_nli_sp':model.score(r['article'],r['summary'])} for r in load(d)]
            save(d,a.stage,rows,{'config':model.cfg,'missing':sum(r['alignscore_nli_sp'] is None for r in rows)})

if __name__=='__main__':main()
