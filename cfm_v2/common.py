from pathlib import Path
import sys,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN=HERE/'run'
DATA=ROOT/'cat_binary_dataset_curated_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def sa(exy):
    e=np.asarray(exy);return np.stack([.5*np.log(e).sum(-1),np.log(e[...,0]/e[...,1])],axis=-1)
