"""Inference-only port of AlignScore-base nli_sp (yuh-zha/AlignScore).

Same pooled RoBERTa NLI head, sentence grouping, max-over-source then
mean-over-claim aggregation. No training/MLM heads or Lightning dependency.
Public source formulas archived with SHA256 in configs/focused/alignscore.json.
"""
import json
from pathlib import Path
import nltk
import torch
from transformers import AutoConfig, AutoTokenizer, RobertaModel

nltk.data.path.insert(0,str(Path('data/nltk').absolute()))


class AlignScoreLocal:
    def __init__(self, config='configs/focused/alignscore.json', device='cuda', batch_size=32):
        self.cfg=json.loads(Path(config).read_text())
        state=torch.load(self.cfg['checkpoint'],map_location='cpu',weights_only=True)['state_dict']
        backbone_cfg=AutoConfig.from_pretrained(self.cfg['backbone_snapshot'])
        self.model=RobertaModel(backbone_cfg)
        weights={k.removeprefix('base_model.'):v for k,v in state.items() if k.startswith('base_model.')}
        # Legacy deterministic position buffer, absent from current state_dict.
        weights.pop('embeddings.position_ids',None)
        self.model.load_state_dict(weights,strict=True)
        self.head=torch.nn.Linear(backbone_cfg.hidden_size,3)
        self.head.load_state_dict({k.removeprefix('tri_layer.'):v for k,v in state.items() if k.startswith('tri_layer.')},strict=True)
        self.model.to(device).eval();self.head.to(device).eval()
        self.tokenizer=AutoTokenizer.from_pretrained(self.cfg['backbone_snapshot'])
        self.device=device;self.batch_size=batch_size
        self.truncated_pairs=0

    def score(self,source,claim):
        sentences=nltk.sent_tokenize(source) or ['']
        chunk_count=len(source.strip().split())//350+1
        per_chunk=max(len(sentences)//chunk_count,1)
        chunks=[' '.join(sentences[i:i+per_chunk]) for i in range(0,len(sentences),per_chunk)]
        claims=nltk.sent_tokenize(claim)
        if not claims:return None
        pairs=[(s,c) for s in chunks for c in claims]
        scores=[]
        with torch.inference_mode():
            for i in range(0,len(pairs),self.batch_size):
                batch=pairs[i:i+self.batch_size]
                first=[p[0] for p in batch];second=[p[1] for p in batch]
                self.truncated_pairs+=sum(len(self.tokenizer.encode(a,b,add_special_tokens=True))>512 for a,b in batch)
                inputs=self.tokenizer(first,second,truncation='only_first',padding=True,max_length=512,return_tensors='pt').to(self.device)
                values=self.head(self.model(**inputs).pooler_output).softmax(-1)[:,0]
                scores.extend(values.tolist())
        return torch.tensor(scores).reshape(len(chunks),len(claims)).max(0).values.mean().item()


if __name__=='__main__':
    model=AlignScoreLocal()
    source='Alice won the election in 2020. Bob lost the election. The vote took place in Paris.'
    claims=['Alice won the election in 2020.','Bob won the election in 2020.','The election took place in London.']
    values=[model.score(source,c) for c in claims]
    if not values[0]>max(values[1:]):raise AssertionError(values)
    Path('results/focused_v1/alignscore_smoke.json').write_text(json.dumps({'source':source,'claims':claims,'scores':values,'notes':'Simple sanity check, not human validation or upstream numerical equivalence proof.'},indent=2)+'\n')
    print(values)
