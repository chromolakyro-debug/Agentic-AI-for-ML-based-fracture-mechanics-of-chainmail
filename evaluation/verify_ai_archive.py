"""Run inside the extracted AI archive: Python + NumPy + Pillow; original project unnecessary."""
import csv,json,hashlib,re
from pathlib import Path
import numpy as np
from PIL import Image

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main(root):
    inventory=read(root/'ARCHIVE_INVENTORY.json')
    for item in inventory['files']:
        p=root/item['path'];assert p.stat().st_size==item['bytes'];assert digest_file(p)==item['sha256']
    with (root/'comparison/training_catalog.csv').open(encoding='utf-8-sig',newline='') as f:catalog=list(csv.DictReader(f))
    split=read(root/'comparison/training_splits.json');indices=np.array([int(r['dataset_index']) for r in catalog]);assert set(indices)==set(split['train'])
    assert not set(indices)&(set(split['validation'])|set(split['test']))
    props=np.array([[float(r['Ex_original']),float(r['Ey_original'])] for r in catalog]);source=[r['source_key'] for r in catalog]
    counts=[]
    for tag in ['old_20k','old_125k']:
        out=root/'runs'/tag;cfg=read(out/'config.json');rows=read(out/'results.json');summary=read(out/'summary.json');targets=read(out/'targets.json')['targets'];seeds=read(out/'seeds.json')
        assert len(rows)==240 and len(targets)==24 and len(seeds)==10
        scale=np.array(cfg['retrieval']['distance']['scales']);soft=cfg['fem']['parameters']['soft_modulus'];tol=cfg['tolerances']
        packed=np.load(out/'fem_inputs.npz',allow_pickle=False);assert packed['binary'].shape==(240,224,224)
        for i,r in enumerate(rows):
            b=packed['binary'][i];assert set(np.unique(b))<={0,1};assert packed['sample_keys'][i]==r['sample_key'];assert np.array_equal(b,np.asarray(Image.open(out/r['image']))//255)
            h=hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest();assert h==r['binary_pixel_sha256']==r['input_pixel_sha256']
            if r['status']!='ok':continue
            e=np.array([r['Ex_FEM'],r['Ey_FEM']]);q=np.array([r['Ex_target'],r['Ey_target']]);d=float(np.linalg.norm((e-q)/scale));assert np.isclose(d,r['d_mech'],rtol=1e-13,atol=1e-15)
            s=.5*np.log(e/soft).sum();a=np.log(e[0]/e[1]);assert np.allclose([s,a],[r['s_FEM'],r['a_FEM']],atol=1e-14,rtol=1e-13)
            assert r['hit_s_a']==bool(abs(s-r['s_target'])<=tol['s'] and abs(a-r['a_target'])<=tol['a'])
        for t,st in zip(targets,summary):
            selected=[r for r in rows if r['target_id']==t['target_id']];valid=[r for r in selected if r['status']=='ok'];d=np.array([r['d_mech'] for r in valid])
            assert len(selected)==10 and sorted(r['seed'] for r in selected)==sorted(seeds);assert st['n_valid']==len(valid)
            if len(d):assert np.isclose(d.mean(),st['mean_d_mech'],rtol=1e-13,atol=1e-15)
            if len(d)>1:assert np.isclose(d.var(ddof=1),st['var_d_mech'],rtol=1e-13,atol=1e-15)
            else:assert st['var_d_mech'] is None
        for kind in ['target_retrieval','fem_retrieval']:
            for q in read(out/f'{kind}.json'):
                if not q['neighbors']:continue
                query=np.array([q['query_Ex'],q['query_Ey']]);d0=np.linalg.norm((props-query)/scale,axis=1);d1=np.linalg.norm((props[:,::-1]-query)/scale,axis=1);dd=np.minimum(d0,d1)
                order=np.lexsort((indices,dd));seen=set();oracle=[]
                for pos in order:
                    if source[pos] in seen:continue
                    seen.add(source[pos]);oracle.append((int(indices[pos]),bool(d1[pos]<d0[pos]),float(dd[pos])))
                    if len(oracle)==10:break
                for actual,want in zip(q['neighbors'],oracle):
                    assert actual['dataset_index']==want[0] and actual['axis_swapped']==want[1];assert np.isclose(actual['distance'],want[2],rtol=1e-13,atol=1e-15)
                if kind=='fem_retrieval':
                    best=min([r for r in rows if r['target_id']==q['target_id'] and r['status']=='ok'],key=lambda r:(r['d_mech'],seeds.index(r['seed'])))
                    assert best['sample_key']==q['query_sample_key'] and best['Ex_FEM']==q['query_Ex'] and best['Ey_FEM']==q['query_Ey']
        html=(out/'index.html').read_text(encoding='utf-8');embedded=json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',html,re.S).group(1))
        assert embedded['rows']==rows and embedded['summary']==summary
        counts.append(dict(run=tag,images=len(rows),valid=sum(r['status']=='ok' for r in rows)))
    print(json.dumps(dict(status='passed',files_verified=len(inventory['files']),runs=counts),indent=2))

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent)
