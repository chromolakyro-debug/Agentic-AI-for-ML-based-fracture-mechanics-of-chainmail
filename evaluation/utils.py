from pathlib import Path
import hashlib,json,csv,importlib,sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def pixel_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def resolve(p,root=ROOT):return Path(p) if Path(p).is_absolute() else Path(root)/p
def adapter(name):
    module,cls=name.split(':');return getattr(importlib.import_module(module),cls)
def to_sa(ex,ey,soft):
    if np.any(np.asarray(ex)<=0) or np.any(np.asarray(ey)<=0) or soft<=0:raise ValueError('Moduli must be positive')
    return np.stack([.5*(np.log(np.asarray(ex)/soft)+np.log(np.asarray(ey)/soft)),np.log(np.asarray(ex)/np.asarray(ey))],axis=-1)
def from_sa(s,a,soft):return soft*np.stack([np.exp(np.asarray(s)+np.asarray(a)/2),np.exp(np.asarray(s)-np.asarray(a)/2)],axis=-1)
def code_identity():
    paths=list(HERE.rglob('*.py'))+[ROOT/'mechanics.py',ROOT/'cfm_v0/model.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths if '__pycache__' not in p.parts and 'runs' not in p.parts}
def distance(q,p,cfg):return np.sqrt(np.sum(((np.asarray(q)-np.asarray(p))/np.asarray(cfg['scales']))**2,axis=-1))
def distance_text(cfg):
    x,y=cfg['scales'];return f'd(q,p)=sqrt(((Ex_q-Ex_p)/{x:g})^2+((Ey_q-Ey_p)/{y:g})^2)'

def package_versions(names):
    from importlib.metadata import version
    return {name:version(name) for name in names}

