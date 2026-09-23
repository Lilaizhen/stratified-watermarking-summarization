"""Download content-pinned data and optional evaluators; rerunnable with cache."""
import argparse,hashlib,json,shutil
from pathlib import Path
from huggingface_hub import hf_hub_download,snapshot_download
import nltk


def main():
    p=argparse.ArgumentParser();p.add_argument('--quality',action='store_true');a=p.parse_args()
    source=Path(hf_hub_download('abisee/cnn_dailymail','3.0.0/validation-00000-of-00001.parquet',repo_type='dataset'))
    expected='65a5ccce932b08f050114ce6e3c39355d563fa24f194052bb2f27d1c6c499c91'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Dataset content changed')
    target=Path('data/source/3.0.0/validation-00000-of-00001.parquet');target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:shutil.copyfile(source,target)
    for resource in ['punkt','punkt_tab']:
        try:nltk.data.find(f'tokenizers/{resource}',paths=[str(Path('data/nltk').resolve())])
        except LookupError:nltk.download(resource,download_dir='data/nltk',quiet=True,raise_on_error=True)
    if a.quality:
        p=Path('configs/focused/alignscore.json');v=json.loads(p.read_text())
        ckpt=Path(hf_hub_download('yzha/AlignScore','AlignScore-base.ckpt',revision=v['revision']))
        if hashlib.sha256(ckpt.read_bytes()).hexdigest()!=v['sha256']:raise ValueError('AlignScore checkpoint hash mismatch')
        v['checkpoint']=str(ckpt);v['backbone_snapshot']=snapshot_download(v['backbone'],revision=v['backbone_revision'])
        p.write_text(json.dumps(v,indent=2)+'\n')
    print('Assets ready. Data SHA256 verified; weights download on demand.')
if __name__=='__main__':main()
