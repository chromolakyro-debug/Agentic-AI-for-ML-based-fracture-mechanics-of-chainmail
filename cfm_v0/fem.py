"""Original full-resolution FEM, hash/protocol cache, explicit failures, resumable batching."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import sys
import time
import traceback
import numpy as np
from common import ROOT, DATA, OUT, read_json, save_json, to_sa
sys.path.insert(0,str(ROOT))
from evaluate_mechanics import initialize_worker, solve_item, mechanics_fingerprint


def safe_solve(item):
    name, pixels = item
    try:
        result = solve_item(item)
        fraction=result['hard_fraction']
        lower,upper=1/(1-.9*fraction),1+9*fraction
        result.update(fixed_fraction_reuss_lower=lower,fixed_fraction_voigt_upper=upper)
        if not all(lower-1e-6<=result[k]<=upper+1e-6 for k in ('Ex_app','Ey_app')):
            raise RuntimeError(f'Fixed-fraction necessary bound violation: {result}')
        sa = to_sa(result['Ex_app'],result['Ey_app'])
        result.update(s_hat=float(sa[0]),a_hat=float(sa[1]))
        return result
    except Exception as error:
        return dict(key=name,sha256_pixels=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    protocol_fingerprint=mechanics_fingerprint(),fem_valid=False,
                    error=repr(error),traceback=traceback.format_exc())


def smoke():
    from threadpoolctl import threadpool_limits
    from mechanics import FiniteSpecimenFEM
    with threadpool_limits(1):
        d=np.load(OUT/'data.npz'); images=np.load(DATA/'dataset.npy',mmap_mode='r')
        start=time.perf_counter(); initialize_worker(); setup=time.perf_counter()-start
        indices=d['train'][[0,1700,3500,6000]]
        rows=[]
        for idx in indices:
            result=safe_solve((f'data_{idx}',images[idx]))
            assert result['fem_valid'], result
            error=float(np.max(np.abs(np.array([result['Ex_app'],result['Ey_app']])/d['moduli'][idx]-1)))
            assert error < 1e-6, error
            rows.append(dict(source_index=int(idx),label_max_relative_error=error,**result))
            print('FEM smoke',idx,result['solve_seconds'],error,flush=True)
        direct=FiniteSpecimenFEM(shape=(224,224),solver='direct').solve(images[indices[0]])
        discrepancy=max(abs(direct[k]/rows[0][k]-1) for k in ('Ex_app','Ey_app'))
        assert discrepancy < 1e-7
        generated=[]
        for i,im in enumerate(np.load(OUT/'smoke_binary.npy')):
            result=safe_solve((f'smoke_generated_{i}',im)); assert result['fem_valid'],result
            generated.append(result)
        save_json(OUT/'smoke_fem.json',dict(setup_seconds=setup,real_samples=rows,generated_samples=generated,
                  median_solve_seconds=float(np.median([r['solve_seconds'] for r in rows])),
                  direct_relative_discrepancy=discrepancy,protocol_fingerprint=mechanics_fingerprint(),
                  elapsed_seconds=time.perf_counter()-start))
        print('FEM smoke passed',flush=True)


def evaluate(folder,workers):
    folder=OUT/folder
    manifest=read_json(folder/'manifest.json')
    binary=np.load(folder/'binary.npy',mmap_mode='r')
    assert binary.dtype==np.uint8 and binary.shape==(len(manifest['samples']),224,224)
    protocol=mechanics_fingerprint()
    destination=folder/'fem.jsonl'
    cached={}
    if destination.exists():
        # A partial last line after process interruption is ignored, all complete failures are retained.
        for line in destination.read_text(encoding='utf-8').splitlines():
            try: r=json.loads(line)
            except json.JSONDecodeError: continue
            if r['protocol_fingerprint']==protocol: cached[(r['key'],r['sha256_pixels'])]=r
    pending=[]
    for row,image in zip(manifest['samples'],binary):
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        assert digest==row['sha256_pixels'] and np.isin(image,(0,1)).all()
        if (row['key'],digest) not in cached: pending.append((row['key'],image.copy()))
    start=time.perf_counter()
    print(f'FEM {folder.name}: {len(binary)-len(pending)} cached, {len(pending)} pending, {workers} workers',flush=True)
    with destination.open('a',encoding='utf-8') as stream:
        if pending:
            # Start with newline so a truncated last record cannot consume a resumed valid record.
            stream.write('\n'); stream.flush()
            with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as executor:
                futures=[executor.submit(safe_solve,item) for item in pending]
                for i,future in enumerate(as_completed(futures),1):
                    result=future.result()
                    stream.write(json.dumps(result)+'\n');stream.flush()
                    cached[(result['key'],result['sha256_pixels'])]=result
                    if i%32==0 or i==len(pending): print(f'FEM {i}/{len(pending)}, wall {time.perf_counter()-start:.1f}s',flush=True)
    results=[cached[(r['key'],r['sha256_pixels'])] for r in manifest['samples']]
    save_json(folder/'fem_summary.json',dict(total=len(results),valid=sum(r['fem_valid'] for r in results),
              failed=sum(not r['fem_valid'] for r in results),workers=workers,computed_this_invocation=len(pending),
              invocation_wall_seconds=time.perf_counter()-start,protocol_fingerprint=protocol,
              sum_solver_seconds=sum(r.get('solve_seconds',0) for r in results)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','evaluate'])
    p.add_argument('--folder',default='final');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.mode=='smoke':smoke()
    else:evaluate(a.folder,a.workers)
