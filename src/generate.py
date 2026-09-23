"""Pinned generation for the final report; resumable at batch boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch
import transformers
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, LogitsProcessorList,
                          WatermarkLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, set_seed)
from headmass_processor import HeadMassProcessor
from hard_ow_processor import HardOWProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='configs/base.json')
    parser.add_argument('--variants', required=True)
    parser.add_argument('--output-root', default='results/focused_v1')
    parser.add_argument('--split', choices=['calibration','test'], default='test')
    parser.add_argument('--splits', default='configs/splits.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = json.loads(Path(args.base).read_text())
    variants = json.loads(Path(args.variants).read_text())
    selected = json.loads(Path(args.splits).read_text())[args.split]
    rows = pq.read_table(base['data']).take([r['row'] for r in selected]).to_pylist()
    assert [r['id'] for r in rows] == [r['id'] for r in selected]
    if not torch.cuda.is_available():
        raise RuntimeError('Select a CUDA GPU runtime')
    tokenizer = AutoTokenizer.from_pretrained(base['model'], revision=base['revision'])
    model = AutoModelForSeq2SeqLM.from_pretrained(base['model'], revision=base['revision'],
                                                torch_dtype=getattr(torch,base['dtype']), attn_implementation='eager').to('cuda').eval()
    for variant in variants:
        cfg = base | variant
        if cfg.get('order','post') != 'post': raise ValueError('Final protocol requires temperature then bias')
        if cfg['method']=='headmass' and cfg.get('protect_special_tokens',False): raise ValueError('Final protocol uses pure stratification')
        if cfg['method'] not in ('none', 'kgw', 'headmass', 'hard_ow'):
            raise ValueError('Unknown method')
        directory = Path(args.output_root) / variant['name']
        directory.parent.mkdir(parents=True, exist_ok=True)
        identity = {'config': cfg, 'split': args.split, 'selected': selected}
        if (directory/'metadata.json').exists():
            old = json.loads((directory/'metadata.json').read_text())
            if any(old[k] != v for k,v in identity.items()):
                raise ValueError(f'Existing run has different settings: {directory}')
            if (directory/'complete.json').exists():
                saved = [json.loads(x) for x in (directory/'generations.jsonl').open()]
                done = json.loads((directory/'complete.json').read_text())
                if [r['id'] for r in saved] != [r['id'] for r in selected] or done['source_sha256'] != hashlib.sha256((directory/'generations.jsonl').read_bytes()).hexdigest():
                    raise ValueError('Completed generation changed')
                print('Already complete:', cfg['name'], flush=True)
                continue
        directory.mkdir(exist_ok=True)
        saved = []
        if (directory/'generations.jsonl').exists():
            for line in (directory/'generations.jsonl').open():
                try: saved.append(json.loads(line))
                except json.JSONDecodeError: break
            if [r['id'] for r in saved] != [r['id'] for r in selected[:len(saved)]]:
                raise ValueError('Partial generation IDs differ')
        resume = len(saved) if len(saved)==len(rows) else len(saved)//cfg['batch_size']*cfg['batch_size']
        with (directory/'generations.jsonl').open('w') as stream:
            for record in saved[:resume]: stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        ordinary = [TemperatureLogitsWarper(cfg['temperature'])]
        if cfg['top_k'] != 0 or cfg['top_p'] != 1: raise ValueError('Final protocol uses no candidate truncation')
        watermark = None
        if cfg['method'] == 'hard_ow':
            watermark = HardOWProcessor(model.config.vocab_size, 'cuda', cfg['gamma'], cfg['hash_key'], model.generation_config.forced_bos_token_id)
        elif cfg['method'] == 'kgw':
            watermark = WatermarkLogitsProcessor(model.config.vocab_size, 'cuda', greenlist_ratio=cfg['gamma'],
                                                  bias=cfg['delta'],hashing_key=cfg['hash_key'],seeding_scheme='lefthash',context_width=1)
        elif cfg['method'] == 'headmass':
            if cfg.get('order', 'post') != 'post':
                raise ValueError('Stratified uses bias after temperature scaling')
            watermark = HeadMassProcessor(model.config.vocab_size, 'cuda', cfg['gamma'], cfg['hash_key'],
                cfg['delta'], [], cfg['rho'])
        if watermark is None:
            processors = ordinary
        else:
            processors = ordinary + [watermark]
        generation = GenerationConfig(do_sample=True, num_beams=1, top_k=0, top_p=1., temperature=1.,
                                      max_new_tokens=cfg['max_new_tokens'],min_new_tokens=cfg['min_new_tokens'],
                                      decoder_start_token_id=model.config.decoder_start_token_id,
                                      bos_token_id=model.config.bos_token_id,eos_token_id=model.config.eos_token_id,
                                      pad_token_id=model.config.pad_token_id,forced_bos_token_id=model.generation_config.forced_bos_token_id,
                                      forced_eos_token_id=None)
        meta = {'config':cfg,'split':args.split,'selected':selected,'generation':generation.to_dict(),
                'splits_file':args.splits,'splits_sha256':hashlib.sha256(Path(args.splits).read_bytes()).hexdigest(),
                'processors':[type(p).__name__ for p in processors],
                'hard_ow_script_sha256':hashlib.sha256(Path('src/hard_ow_processor.py').read_bytes()).hexdigest(),
                'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'headmass_script_sha256':hashlib.sha256(Path('src/headmass_processor.py').read_bytes()).hexdigest(),
                'data_sha256':hashlib.sha256(Path(cfg['data']).read_bytes()).hexdigest(),
                'seed_convention':'seed + batch starting offset; same article ordering and batch size for every variant; not identical token sampling paths'}
        (directory/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        began=time.monotonic()
        with (directory/'generations.jsonl').open('a') as stream:
            for offset in range(resume,len(rows),cfg['batch_size']):
                batch=rows[offset:offset+cfg['batch_size']]
                inputs=tokenizer([r['article'] for r in batch],padding=True,truncation=True,max_length=cfg['max_input_tokens'],return_tensors='pt').to('cuda')
                set_seed(cfg['seed']+offset)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                with torch.inference_mode():
                    generated=model.generate(**inputs,generation_config=generation,logits_processor=LogitsProcessorList(processors))
                torch.cuda.synchronize();elapsed=time.monotonic()-start
                for j,row in enumerate(batch):
                    ids=generated[j].tolist()
                    try:
                        end=ids.index(model.config.eos_token_id,1)+1
                        ids=ids[:end]
                    except ValueError:
                        pass
                    record={'id':row['id'],'row':selected[offset+j]['row'],'variant':cfg['name'],'method':cfg['method'],
                            'article':row['article'],'reference':row['highlights'],'summary':tokenizer.decode(ids,skip_special_tokens=True),
                            'token_ids':ids,'generated_tokens':len(ids)-1,'hit_length_cap':len(ids)-1>=cfg['max_new_tokens'],
                            'batch_seed':cfg['seed']+offset,'batch_seconds':elapsed,'amortized_generation_seconds':elapsed/len(batch),
                            'input_tokens':int(inputs['attention_mask'][j].sum()),'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                stream.flush()
                print(cfg['name'],min(offset+len(batch),len(rows)),len(rows),flush=True)
        summary={'wall_seconds_this_session':time.monotonic()-began,'records':len(rows), 'source_sha256':hashlib.sha256((directory/'generations.jsonl').read_bytes()).hexdigest()}
        (directory/'complete.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
