"""Frozen, resumable FEM verification of exact D4 image augmentation; no training."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=ROOT/'cat_binary_dataset_curated_v1'
OUT=HERE/'results'
sys.path.insert(0,str(ROOT))
from mechanics import FiniteSpecimenFEM
from evaluate_mechanics import mechanics_fingerprint
from augmentation import transform, map_ex_ey


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def pixels_sha(arr):return hashlib.sha256(arr.tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def csvsave(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metadata():
    with (DATA/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def check_operators():
    # Independent pixel coordinate mapping, not another call to rot90/flip.
    a=np.arange(25).reshape(5,5)%2; a[0,1]=1; a[1,0]=0
    for k in range(4):
        for flip in (False,True):
            actual=transform(a,k,flip);expected=np.empty_like(a)
            for r in range(5):
                for c in range(5):
                    rr,cc=[(r,c),(4-c,r),(4-r,4-c),(c,4-r)][k]
                    if flip:cc=4-cc
                    expected[rr,cc]=a[r,c]
            assert np.array_equal(actual,expected)
            assert actual.flags.c_contiguous and not np.shares_memory(actual,a)
            assert actual.dtype==a.dtype and np.bincount(actual.ravel(),minlength=2).tolist()==np.bincount(a.ravel(),minlength=2).tolist()
            assert map_ex_ey(2,7,k)==((7,2) if k%2 else (2,7))
    # An upper-right physical marker moves to upper-left under +90deg CCW.
    marker=np.zeros((5,5),np.uint8);marker[0,4]=1
    assert transform(marker,1,False)[0,0]==1
    assert transform(marker,0,True)[0,0]==1
    return dict(all_eight_index_mappings=True,physical_corner_orientation=True,contiguous_copy=True)


def prepare():
    if (OUT/'config.json').exists():
        raise RuntimeError('Frozen run already exists. Use solve/report to resume, not prepare.')
    OUT.mkdir(parents=True,exist_ok=True)
    rows=metadata();images=np.load(DATA/'dataset.npy',mmap_mode='r',allow_pickle=False)
    labels=np.load(DATA/'data.npz',allow_pickle=False);classes=read(DATA/'classes.json')['classes']
    assert images.shape==(7857,224,224) and images.dtype==np.uint8
    seed=2026111601
    streams=np.random.SeedSequence(seed).spawn(2)
    selection_rng=np.random.default_rng(streams[0]);augmentation_rng=np.random.default_rng(streams[1])
    samples=[]
    for c in classes:
        eligible=[i for i,r in enumerate(rows) if r['group_id']==c['group_id']]
        selected=selection_rng.choice(eligible,size=min(20,len(eligible)),replace=False)
        for i in selected:
            i=int(i);row=rows[i];arr=images[i]
            k=int(augmentation_rng.integers(0,4));flip=bool(augmentation_rng.integers(0,2))
            aug=transform(arr,k,flip)
            assert pixels_sha(arr)==row['sha256_pixels']
            with Image.open(DATA/row['new_relative_path']) as im:
                assert np.array_equal(np.asarray(im,dtype=np.uint8),arr)
            assert row['new_id']==labels['sample_ids'][i]
            assert np.array_equal(labels['moduli'][i],[float(row['Ex_app']),float(row['Ey_app'])])
            for field,value in {'E_soft':1.,'E_hard':10.,'nu_soft':.3,'nu_hard':.3,
                'specimen_width_mm':40.,'specimen_height_mm':40.,'specimen_thickness_mm':2.,'nominal_strain':.001}.items():
                assert float(row[field])==value,(field,row[field])
            assert row['solver_protocol_sha256']==mechanics_fingerprint()
            assert arr.shape==aug.shape and arr.dtype==aug.dtype and int(arr.sum())==int(aug.sum())
            samples.append(dict(sample_order=len(samples),dataset_index=i,new_id=row['new_id'],original_id=row['original_id'],
                group_id=c['group_id'],group_name_zh=c['group_name_zh'],relative_path=row['new_relative_path'],
                split=row['curation.split'],angle_ccw=k*90,quarter_turns=k,horizontal_flip=flip,swap_ex_ey=bool(k%2),
                Ex_metadata=float(row['Ex_app']),Ey_metadata=float(row['Ey_app']),
                original_pixel_sha256=pixels_sha(arr),augmented_pixel_sha256=pixels_sha(aug),
                png_sha256=sha(DATA/row['new_relative_path']),hard_pixels=int((arr==0).sum()),soft_pixels=int(arr.sum())))
    assert len(samples)==len({r['new_id'] for r in samples})==160
    # One unselected-for-outcome sample per group. Sparse direct cross-check on groups01,05.
    precision=[next(r['sample_order'] for r in samples if r['group_id']==c['group_id']) for c in classes]
    files=[DATA/p for p in ('dataset.npy','metadata.csv','metadata.jsonl','index.csv','data.npz','splits.json','STATUS.json','integrity_manifest.json')]
    files += [ROOT/p for p in ('mechanics.py','evaluate_mechanics.py','cfm_v0/train.py','cat_binary_dataset/dataset.npy','cat_binary_dataset/metadata.csv')]
    files += [DATA/r['relative_path'] for r in samples]
    protected={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    import scipy,pyamg
    config=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=seed,rng='numpy PCG64; SeedSequence.spawn(2): selection, augmentation',
        selection='20 per class without replacement from all current splits; no rebalancing of transformations',
        n=160,order='np.rot90(image,k,axes=(0,1)) then np.flip(axis=1) if flag; contiguous independent copy',
        probabilities=dict(angle_each=.25,horizontal_flip=.5),physics=dict(shape=[224,224],soft_modulus=1.,hard_modulus=10.,poisson=.3,
            width=40.,height=40.,thickness=2.,strain=.001,max_iterations=1200),
        main_solver='amg',main_tolerance=1e-9,atol=1e-8,rtol=1e-7,relative_denominator_floor=1e-12,
        tolerance_basis='Prespecified regression tolerance supported by existing AMG/direct tests at 1e-7 and cfm_v0/run/smoke_fem.json; residual is not a modulus error bound. Confirm with fixed 8-pair 1e-11 solves and 2-pair direct solves.',
        floor_basis='1e-12*Esoft; current fully solid common-nu materials have Ex,Ey>=Esoft=1 by energy ordering, so the floor is inactive.',
        precision_sample_orders=precision,direct_sample_orders=[precision[0],precision[4]],precision_tolerance=1e-11,
        failure_policy='Keep original failed results; diagnose/refine separately, never widen frozen tolerance.',
        figure_sample_orders=precision,workers=8,protocol_fingerprint=mechanics_fingerprint(),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pyamg=pyamg.__version__),
        historical_direct_relative_discrepancy=read(ROOT/'cfm_v0/run/smoke_fem.json')['direct_relative_discrepancy'],
        operators=check_operators(),protected_files_sha256=protected,
        script_sha256={p.name:sha(p) for p in (HERE/'run.py',HERE/'augmentation.py')})
    save(OUT/'config.json',config);save(OUT/'samples.json',samples);csvsave(OUT/'samples.csv',samples)
    save(OUT/'frozen_inputs.json',{p:sha(OUT/p) for p in ('config.json','samples.json','samples.csv')})
    print(json.dumps(dict(n=len(samples),transforms=dict(Counter(f'{r["angle_ccw"]}_{r["horizontal_flip"]}' for r in samples)),config_sha256=sha(OUT/'config.json')),indent=2),flush=True)


_SOLVER=None
def worker_init(solver,tolerance):
    global _SOLVER,_LIMIT
    from threadpoolctl import threadpool_limits
    _LIMIT=threadpool_limits(1)
    np.random.seed(10731)
    _SOLVER=FiniteSpecimenFEM(**read(OUT/'config.json')['physics'],solver=solver,tolerance=tolerance)


def solve_one(item):
    key,pixels,config_hash=item
    start=time.perf_counter()
    base=dict(key=key,pixel_sha256=pixels_sha(pixels),config_sha256=config_hash)
    try:return dict(**base,status='ok',**_SOLVER.solve(pixels))
    except Exception as exc:
        return dict(**base,status='failed',fem_valid=False,error=repr(exc),traceback=traceback.format_exc(),solve_seconds=time.perf_counter()-start)


def solve(mode):
    cfg=read(OUT/'config.json');samples=read(OUT/'samples.json')
    for p,digest in read(OUT/'frozen_inputs.json').items():assert sha(OUT/p)==digest
    assert mechanics_fingerprint()==cfg['protocol_fingerprint']
    images=np.load(DATA/'dataset.npy',allow_pickle=False,mmap_mode='r')
    orders=range(len(samples)) if mode=='main' else cfg['precision_sample_orders' if mode=='precision' else 'direct_sample_orders']
    items=[]
    for n in orders:
        row=samples[n];arr=images[row['dataset_index']];aug=transform(arr,row['quarter_turns'],row['horizontal_flip'])
        for kind,pix in [('original',arr),('augmented',aug)]:
            assert pixels_sha(pix)==row[kind+'_pixel_sha256']
            items.append((f'{n:03d}_{kind}',pix.copy(),sha(OUT/'config.json')))
    path=OUT/(mode+'_fem.jsonl');cached={}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            assert r['config_sha256']==sha(OUT/'config.json');cached[(r['key'],r['pixel_sha256'])]=r
    pending=[item for item in items if (item[0],pixels_sha(item[1])) not in cached]
    workers=cfg['workers'] if mode=='main' else (4 if mode=='precision' else 1)
    solver='direct' if mode=='direct' else 'amg';tol=cfg['main_tolerance'] if mode=='main' else cfg['precision_tolerance']
    start=time.perf_counter();print(f'{mode}: {len(pending)} actual image FEM solves; {workers} CPU workers',flush=True)
    with path.open('a',encoding='utf-8') as f:
        if pending:
            f.write('\n');f.flush()
            with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(solver,tol)) as pool:
                futures=[pool.submit(solve_one,item) for item in pending]
                for i,job in enumerate(as_completed(futures),1):
                    r=job.result();f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
                    if i%16==0 or i==len(pending):print(f'{mode}: {i}/{len(pending)}; {time.perf_counter()-start:.1f}s; last={r["status"]}',flush=True)
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    save(OUT/(mode+'_timing.json'),dict(solver=solver,tolerance=tol,workers=workers,computed_this_invocation=len(pending),
        invocation_seconds=time.perf_counter()-start,valid=sum(r['fem_valid'] for r in rows),failed=sum(not r['fem_valid'] for r in rows),
        sum_solve_seconds=sum(r['solve_seconds'] for r in rows),median_solve_seconds=float(np.median([r['solve_seconds'] for r in rows]))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','main','precision','direct'])
    a=parser.parse_args()
    if a.mode=='prepare':prepare()
    else:solve(a.mode)
