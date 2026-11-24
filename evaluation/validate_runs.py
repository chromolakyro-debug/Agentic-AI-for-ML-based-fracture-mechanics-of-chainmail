"""Cross-check actual runs, source-only retrieval, matching noise, and live re-sampling."""
import numpy as np
from utils import HERE,ROOT,read,save,adapter,pixel_hash,sha
from pipeline import Run

def main(formal=False):
    runs=[Run(HERE/f'configs/{name}.json',diagnostic=not formal) for name in ['old_20k','old_125k']]
    first,second=runs
    assert read(first.out/'initial_noise.json')==read(second.out/'initial_noise.json')
    assert read(first.out/'target_retrieval.json')==read(second.out/'target_retrieval.json')
    checks=[]
    for run in runs:
        ds=run.dataset;trainset=set(ds.train.tolist());rows=read(run.out/'results.json');assert len(rows)==len(run.targets)*len(run.seeds)
        for kind in ['target_retrieval','fem_retrieval']:
            for q in read(run.out/f'{kind}.json'):
                if not q['neighbors']:continue
                # Independent brute-force oracle, explicitly grouping each original source.
                best={};query=np.array([q['query_Ex'],q['query_Ey']]);scale=np.array(run.distance['scales'])
                for idx in ds.train:
                    e=ds.moduli[idx];d0=float(np.linalg.norm((query-e)/scale));d1=float(np.linalg.norm((query-e[::-1])/scale));d=min(d0,d1)
                    v=(d,int(idx),d1<d0);key=ds.source_keys[idx]
                    if key not in best or v[:2]<best[key][:2]:best[key]=v
                oracle=sorted(best.values())[:run.k]
                for r,want in zip(q['neighbors'],oracle):
                    assert r['dataset_index'] in trainset;assert r['dataset_index']==want[1];assert r['axis_swapped']==want[2];assert np.isclose(r['distance'],want[0],rtol=1e-14,atol=0)
                if kind=='fem_retrieval':
                    actual=min([r for r in rows if r['target_id']==q['target_id'] and r['status']=='ok'],key=lambda r:(r['d_mech'],run.seeds.index(r['seed'])))
                    assert actual['sample_key']==q['query_sample_key'] and actual['Ex_FEM']==q['query_Ex'] and actual['Ey_FEM']==q['query_Ey']
        for r,b in run.generated():
            assert r['binary_pixel_sha256']==pixel_hash(b)
            actual=next(x for x in rows if x['sample_key']==r['sample_key']);assert actual['input_pixel_sha256']==pixel_hash(b)
        checks.append(dict(run=run.cfg['run_id'],retrieval_oracle='passed',FEM_image_linkage='passed',best_seed_selection='passed'))
    model=adapter(first.cfg['model']['adapter'])(first.cfg['model'],first.cfg['sampling'],first.dataset,ROOT)
    raw,c=model.generate(first.targets[0],first.seeds)
    saved=np.load(first.out/'arrays'/f'{first.targets[0]["target_id"]}.npz')['raw']
    exact=bool(np.array_equal(raw,saved));delta=float(abs(raw-saved).max())
    save(HERE/('validation/formal_run_checks.json' if formal else 'validation/run_checks.json'),dict(checks=checks,initial_noise_identical=True,target_retrieval_identical=True,
        live_resample_exact=exact,live_resample_max_absolute_difference=delta,checkpoint=first.cfg['model']['checkpoint'],seeds=first.seeds))
    assert exact,'Repeat generation changed; investigate reproducibility before formal evaluation'
    print('Both runs: train-only retrieval oracle, actual FEM linkage, shared noise and exact live re-sampling passed.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--formal',action='store_true');a=p.parse_args();main(a.formal)
