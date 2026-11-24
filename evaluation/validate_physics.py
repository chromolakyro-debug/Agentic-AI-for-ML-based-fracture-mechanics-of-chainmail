"""Small independent real-FEM integration check; does not substitute for the 24-cell benchmark."""
import time
import numpy as np
from utils import HERE,ROOT,read,save,adapter,sha,pixel_hash

def main():
    cfg=read(HERE/'configs/old_20k.json');ds=adapter(cfg['dataset']['adapter'])(cfg['dataset'],ROOT);audit=ds.audit(HERE/'cache')
    f=adapter(cfg['fem']['adapter']);f.initializer(cfg['fem'])
    # Largest train anisotropy makes an axis assignment error observable.
    i=int(ds.train[np.argmax(abs(np.log(ds.moduli[ds.train,0]/ds.moduli[ds.train,1])))]);original=np.array(ds.images[i])
    # Threshold fixed before FEM: original D4 study established rtol=1e-7 at solver rtol=1e-9.
    tolerances=dict(atol=1e-8,rtol=1e-7,source='augmentation_validation; independent original/rot90/left-right mirror check')
    save(HERE/'validation/physics_plan.json',dict(sample=ds.describe(i),tolerances=tolerances,fem=cfg['fem'],
        order='rot90 CCW then optional np.flip(axis=1)',image_x='columns right',physical_y='rows reversed by original FEM'))
    results=[]
    for name,b,swap in [('original',original,False),('ccw90',np.ascontiguousarray(np.rot90(original)),True),('horizontal_flip',np.ascontiguousarray(np.flip(original,axis=1)),False)]:
        r=f.solve((name,b));assert r['status']=='ok';expected=ds.moduli[i,::-1] if swap else ds.moduli[i]
        actual=np.array([r['Ex_app'],r['Ey_app']]);err=abs(actual-expected)
        results.append(dict(transform=name,expected=expected.tolist(),absolute_errors=err.tolist(),passed=bool(np.all(err<=tolerances['atol']+tolerances['rtol']*abs(expected))),
            phase_count_preserved=int(b.sum())==int(original.sum()),**r))
    invalid=f.solve(('intentional_bad_phase',np.full_like(original,2)))
    assert invalid['status']=='failed' and invalid['error']
    save(HERE/'validation/physics_results.json',dict(dataset=audit,sample=ds.describe(i),tolerances=tolerances,results=results,
        deliberate_invalid_input=invalid,passed=all(r['passed'] and r['phase_count_preserved'] for r in results)))
    assert all(r['passed'] for r in results)
    # Future dataset adapter is checked without attaching an old checkpoint to the new split.
    newcfg=read(HERE/'configs/curated_dataset_adapter.json');new=adapter(newcfg['adapter'])(newcfg,ROOT)
    save(HERE/'validation/curated_adapter_audit.json',new.audit(HERE/'cache'))
    print('Real FEM axis/phase/metadata checks and both dataset adapters passed.')

if __name__=='__main__':main()
