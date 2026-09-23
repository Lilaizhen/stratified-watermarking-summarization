"""Same-prefix head/tail diagnostics on saved NW trajectories (not generation)."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, WatermarkLogitsProcessor

from headmass_processor import mass_preserving_tilt


RHOS = (.90, .95, .98)
CONDITIONS = [('NW', 0.)] + [('W-soft', i/2) for i in range(1, 13)]



def diagnostics(raw, green, temperature, soft_order='pre'):
    scores = raw.float() / temperature
    p = scores.softmax(-1)
    sorted_p, order = p.sort(descending=True, stable=True)
    cumulative = sorted_p.cumsum(-1)
    original_k = {r: (cumulative < r).sum(-1) + 1 for r in RHOS}
    original_mass = {r: cumulative.gather(1, (k-1)[:, None]).squeeze(1)
                     for r, k in original_k.items()}
    top = order[:, :1]
    top_green = green.gather(1, top).squeeze(1)
    logp = scores.log_softmax(-1)
    nll = torch.where(torch.isfinite(logp), -logp, 0.)
    base = dict(top1_green=top_green, top1_p=p.gather(1, top).squeeze(1),
                green_before=(p * green).sum(-1))
    outputs = []
    for method, delta in CONDITIONS:
        if method == 'NW':
            q = p
        elif method == 'W-soft':
            effective = delta / temperature if soft_order == 'pre' else delta
            q = (scores + effective * green).softmax(-1)
            # Verify the analytic color-mass formula against logit softmax.
            factor = np.exp(effective)
            expected = p * torch.where(green, factor, 1.)
            expected /= expected.sum(-1, keepdim=True)
            torch.testing.assert_close(q, expected, atol=2e-6, rtol=2e-5)
        else:
            q = mass_preserving_tilt(scores, green, [], .98, delta).softmax(-1)
        torch.testing.assert_close(q.sum(-1), torch.ones(len(q), device=q.device),
                                   atol=2e-6, rtol=0.)
        q_sorted = q.sort(descending=True).values
        q_cumulative = q_sorted.cumsum(-1)
        original_order_mass = q.gather(1, order).cumsum(-1)
        metrics = dict(base, top1_q=q.gather(1, top).squeeze(1),
                       top1_rank_after=(q > q.gather(1, top)).sum(-1)+1,
                       top1_flip=q.argmax(-1) != top.squeeze(1),
                       green_after=(q*green).sum(-1),
                       entropy_before=(p*nll).sum(-1),
                       entropy_after=-(q*q.clamp_min(1e-38).log()).sum(-1),
                       expected_bart_nll_shift=((q-p)*nll).sum(-1))
        for rho in RHOS:
            tag = str(round(rho*100))
            k = original_k[rho]
            mass = original_order_mass.gather(1, (k-1)[:, None]).squeeze(1)
            metrics.update({f'k{tag}_before': k,
                            f'k{tag}_after': (q_cumulative < rho).sum(-1)+1,
                            f'head{tag}_before': original_mass[rho],
                            f'head{tag}_after': mass,
                            f'head{tag}_shift': mass-original_mass[rho]})
            if method.startswith('Strat') and rho == .98:
                torch.testing.assert_close(mass, original_mass[rho], atol=3e-6, rtol=0.)
        arrays = {k: v.cpu().tolist() for k, v in metrics.items()}
        outputs.append((method, delta, arrays))
    return outputs


def summarize(frame):
    summaries = []
    numeric = [k for k in frame.columns if k not in ('id', 'position', 'method', 'delta')]
    for (method, delta), block in frame.groupby(['method', 'delta'], sort=False):
        for label, group in [('all', block), ('top1_red', block[~block.top1_green]),
                             ('top1_green', block[block.top1_green])]:
            if group.empty:
                continue
            item = dict(method=method, delta=delta, group=label, positions=len(group),
                        articles=group.id.nunique())
            for key in numeric:
                values = group[key].astype(float)
                item[key+'_mean'] = values.mean()
                item[key+'_p10'] = values.quantile(.1)
                item[key+'_median'] = values.median()
                item[key+'_p90'] = values.quantile(.9)
            for rho in RHOS:
                tag = str(round(rho*100))
                change = group[f'head{tag}_shift']
                item[f'head{tag}_decrease_fraction'] = (change < -1e-5).mean()
                item[f'head{tag}_drop_over_5pp_fraction'] = (change < -.05).mean()
                item[f'head{tag}_shift_article_mean'] = group.groupby('id')[f'head{tag}_shift'].mean().mean()
                item[f'k{tag}_increase_fraction'] = (group[f'k{tag}_after'] > group[f'k{tag}_before']).mean()
            summaries.append(item)
    return pd.DataFrame(summaries)


def plot(summary, output, soft_order='post'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=summary[(summary.group=='all') & summary.method.isin(['NW','W-soft'])].sort_values('delta')
    fig,ax=plt.subplots(figsize=(7,3.5))
    for rho in (90,95,98):
        ax.plot(frame.delta,100*frame[f'head{rho}_after_mean'],'o-',label=f'{rho}% head')
    ax.set(xlabel=r'Bias $\delta$',ylabel='Original-head probability mass (%)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/f'head_mass.{suffix}',dpi=170)
    plt.close(fig)


def main():
    global CONDITIONS
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='runs/demo/nw')
    parser.add_argument('--output', default='outputs/new_head_mass')
    parser.add_argument('--articles', type=int, default=16)
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--soft-order', choices=['pre','post'], default='post')
    parser.add_argument('--soft-only', action='store_true')
    args = parser.parse_args()
    if args.soft_only:
        CONDITIONS = [c for c in CONDITIONS if c[0] in ('NW', 'W-soft')]
    source, output = Path(args.source), Path(args.output)
    if args.summarize_only:
        frame = pd.read_json(output/'positions.jsonl', lines=True)
        summary = summarize(frame)
        summary.to_csv(output/'summary.csv', index=False)
        summary.to_json(output/'summary.json', orient='records', indent=2)
        protocol = json.loads((output/'complete.json').read_text())
        plot(summary, output, protocol.get('soft_order','pre'))
        protocol['aggregation_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        protocol['head_mass_decrease_tolerance'] = 1e-5
        (output/'complete.json').write_text(json.dumps(protocol, indent=2)+'\n')
        return
    output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((source/'metadata.json').read_text())
    cfg = meta['config']
    assert cfg['method'] == 'none' and cfg['top_k'] == 0 and cfg['top_p'] == 1
    rows = [json.loads(line) for line in (source/'generations.jsonl').read_text().splitlines()][:args.articles]
    assert len(rows) == args.articles
    began = time.monotonic()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model'], revision=cfg['revision'],
        torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    colors = WatermarkLogitsProcessor(model.config.vocab_size, 'cuda',
        greenlist_ratio=cfg['gamma'], bias=0., hashing_key=cfg['hash_key'],
        seeding_scheme='lefthash', context_width=1)
    records = []
    with torch.inference_mode(), (output/'positions.jsonl').open('w') as stream:
        for index, row in enumerate(rows):
            ids = torch.tensor([row['token_ids']], device='cuda')
            inputs = tokenizer(row['article'], truncation=True, max_length=cfg['max_input_tokens'],
                               return_tensors='pt').to('cuda')
            raw = model(**inputs, decoder_input_ids=ids[:, :-1], use_cache=False).logits[0].float()
            # Position j predicts ids[j+1]; j sampled tokens precede it.
            raw[:cfg['min_new_tokens'], model.config.eos_token_id] = -torch.inf
            positions = [j for j in range(len(row['token_ids'])-1)
                         if row['token_ids'][j] not in tokenizer.all_special_ids]
            for start in range(0, len(positions), 32):
                selected = positions[start:start+32]
                logits = raw[selected]
                green = torch.zeros_like(logits, dtype=torch.bool)
                for b, j in enumerate(selected):
                    green[b, colors._get_greenlist_ids(ids[0, :j+1])] = True
                for method, delta, arrays in diagnostics(logits, green, cfg['temperature'], args.soft_order):
                    for b, j in enumerate(selected):
                        item = dict(id=row['id'], position=j, method=method, delta=delta,
                                    **{k: v[b] for k, v in arrays.items()})
                        stream.write(json.dumps(item)+'\n')
                        records.append(item)
            if (index+1) % 10 == 0:
                print(f'{index+1}/{len(rows)} articles; {time.monotonic()-began:.1f}s', flush=True)
    frame = pd.DataFrame(records)
    summary = summarize(frame)
    summary.to_csv(output/'summary.csv', index=False)
    summary.to_json(output/'summary.json', orient='records', indent=2)
    plot(summary, output, args.soft_order)
    protocol = dict(config=cfg, soft_order=args.soft_order, conditions=CONDITIONS, rhos=RHOS, articles=len(rows),
        positions=int((frame.method=='NW').sum()), article_ids=[r['id'] for r in rows],
        source=str(source), source_sha256=hashlib.sha256((source/'generations.jsonl').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        headmass_script_sha256=hashlib.sha256(Path('src/headmass_processor.py').read_bytes()).hexdigest(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), elapsed_seconds=time.monotonic()-began,
        notes=[f'First {len(rows)} stored evaluation articles, chosen by existing order, no new generations.',
               'Teacher-forced saved NW prefixes; FP16 eager BART forward, FP32 diagnostic transforms.',
               'Not bit-identical batched cached-generation replay; no claims about sampled sequence changes.',
               'Min-new-token EOS constraint retained; exclude special previous tokens including forced BOS step.',
               'Full vocabulary including special candidates; final EOS decision included when present.',
               'Minimal coverage set includes crossing token: actual head mass can exceed nominal rho.',
               f'W-soft order={args.soft_order}; effective bias is delta/T for pre, delta for post; stratification post-temperature delta, rho=.98.',
               'Pooled positions are correlated within articles; these are descriptive statistics.',
               'Expected BART NLL shift is local constrained-distribution surprisal, not GPT2 PPL.'])
    (output/'complete.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(summary[summary.group=='all'][['method','delta','positions','head90_after_mean',
          'head95_after_mean','head98_after_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
