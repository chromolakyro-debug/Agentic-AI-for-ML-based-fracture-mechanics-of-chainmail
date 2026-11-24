"""Independent numeric audit; --base may point at an extracted review ZIP."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def verify(base,complete=False):
    run=base/'run';cfg=read(run/'config.json');d=np.load(run/'data.npz',allow_pickle=False);splits=read(run/'splits.json');tr=d['train']
    assert len(d['sample_ids'])==7857 and len(tr)==6285
    seen=set()
    for k in ['train','validation','test']:
        assert np.array_equal(d[k],splits[k]);assert not seen&set(d[k]);seen.update(d[k])
    assert seen==set(range(7857));assert np.array_equal(d['train_weight_sample_ids'],d['sample_ids'][tr])
    exy=d['moduli'];sa=np.column_stack([.5*np.log(exy).sum(1),np.log(exy[:,0]/exy[:,1])])
    np.testing.assert_allclose(sa,d['sa'],rtol=1e-13,atol=1e-14)
    np.testing.assert_array_equal(d['mean'],sa[tr].mean(0));np.testing.assert_array_equal(d['std'],sa[tr].std(0))
    edges=[np.linspace(sa[tr,j].min()-1e-9,sa[tr,j].max()+1e-9,n+1) for j,n in enumerate((8,12))]
    bins=[np.clip(np.searchsorted(edges[j],sa[tr,j],side='right')-1,0,n-1) for j,n in enumerate((8,12))]
    code=bins[0]*12+bins[1];counts=np.bincount(code,minlength=96);inverse=counts[code]**(-.5);weight=np.minimum(.5+.5*inverse/inverse.mean(),4.);weight/=weight.mean()
    np.testing.assert_allclose(weight,d['train_weights'],rtol=1e-14,atol=1e-14)
    groups=np.array([str(v).split('_')[0] for v in d['sample_ids']]);audit=read(run/'data_audit.json')
    assert {g:int(np.sum(groups==g)) for g in np.unique(groups)}==audit['class_counts']
    # Full image and original fine-style association checks when the original workspace is present.
    root=base.parent;metadata=root/'cat_binary_dataset_curated_v1/metadata.csv';mapping_checked=False
    if metadata.exists():
        rows=list(csv.DictReader(metadata.open(encoding='utf-8-sig',newline='')))
        rules=read(root/'cat_binary_dataset_curated_v1/classification_rules.json')
        mappings=next(v for v in rules.values() if isinstance(v,dict) and 'halftone__00' in v)
        for r in rows:
            m=mappings[r['requested_style']];assert m['action']=='keep';assert m['group_id']==r['group_id']
        mapping_checked=True
    result=dict(status='passed',samples=7857,train=6285,train_only_normalization=True,train_only_weights=True,split_and_weight_ids=True,whole_style_mapping_checked=mapping_checked)
    if not complete:return result
    entries=[json.loads(line) for line in (run/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    train={r['step']:r for r in entries if r['kind']=='train'};vals={r['step']:r for r in entries if r['kind']=='validation'}
    summary=read(run/'training_summary.json');last=summary['step'];assert set(train)==set(range(1,last+1))
    schedule=cfg['scheduler'];period=schedule['cycle_steps']
    for step,r in train.items():
        cycle,position=divmod(step-1,period);peak=max(schedule['peak_floor'],schedule['max_lr']*schedule['peak_decay']**cycle)
        expected=schedule['min_lr']+(peak-schedule['min_lr'])*(1+math.cos(math.pi*position/(period-1)))/2
        if cycle==0:expected*=min(step/schedule['initial_warmup_updates'],1)
        assert abs(expected-r['lr'])<1e-16;assert r['samples_seen']==step*cfg['batch']
    best_step=min(vals,key=lambda step:vals[step]['validation_velocity_mse']);assert best_step==summary['best_step']
    assert vals[best_step]['validation_velocity_mse']==summary['best_val_mse'];assert summary['restart_count']==(last-1)//period
    # Rebuild cycle decisions from raw validation records, independent of the controller implementation.
    history=read(run/'plateau_history.json');prior_best=None;a_count=b_count=0;active=False;armed=None;p=cfg['plateau']
    for i,r in enumerate(history,1):
        end=i*period;v=min(x['validation_velocity_mse'] for step,x in vals.items() if end-period<step<=end)
        improvement=None if prior_best is None else (prior_best-v)/max(abs(prior_best),p['epsilon']);eligible=end>=p['initial_training_updates'] and improvement is not None
        stop=False;start=False
        if eligible:
            a_count=a_count+1 if improvement<p['evaluation_relative_threshold'] else 0;b_count=b_count+1 if improvement<p['stop_relative_threshold'] else 0
            if a_count>=p['evaluation_patience'] and not active:active=True;start=True
            if improvement>=p['stop_relative_threshold']:armed=None
            if b_count>=p['stop_patience']:
                if armed is None:armed=i
                elif i>armed:stop=True
        assert r['cycle_best']==v and r['best_before']==prior_best and r['relative_improvement']==improvement
        assert (r['evaluation_patience_count'],r['stop_patience_count'],r['eval_active'],r['started_evaluation'],r['stop_armed_cycle'],r['stop'])==(a_count,b_count,active,start,armed,stop)
        prior_best=v if prior_best is None else min(prior_best,v)
    if summary['reason']=='convergence_criterion':assert history[-1]['stop']
    elif summary['reason']=='update_budget':assert last==cfg['limits']['max_updates']
    elif summary['reason']=='training_time_budget':assert summary['training_seconds']>=cfg['limits']['max_training_seconds']
    else:raise ValueError('Unexpected terminal reason')
    evaluation=base/'final_evaluation'
    if not evaluation.exists():evaluation=root/read(run/'final_evaluation.json')['output']
    rows=read(evaluation/'results.json');stats=read(evaluation/'summary.json');assert len(rows)==240
    binary=np.load(run/'final_fem_inputs.npz',allow_pickle=False)
    for i,r in enumerate(rows):
        assert binary['sample_keys'][i]==r['sample_key'];h=hashlib.sha256(np.ascontiguousarray(binary['binary'][i]).tobytes()).hexdigest();assert h==r['binary_pixel_sha256']==r['input_pixel_sha256']
        if r['status']!='ok':continue
        delta=np.array([r['Ex_FEM']-r['Ex_target'],r['Ey_FEM']-r['Ey_target']]);assert np.isclose(np.linalg.norm(delta),r['d_mech'],rtol=1e-13,atol=1e-14)
    for st in stats:
        valid=[r for r in rows if r['target_id']==st['target_id'] and r['status']=='ok'];assert st['n_valid']==len(valid)
        if valid:assert np.isclose(np.mean([r['d_mech'] for r in valid]),st['mean_d_mech'],rtol=1e-13)
        if len(valid)>1:assert np.isclose(np.var([r['d_mech'] for r in valid],ddof=1),st['var_d_mech'],rtol=1e-13)
        else:assert st['var_d_mech'] is None
    result.update(steps=last,all_step_LR_verified=True,validation_selection=True,plateau_decisions=True,FEM_binary_linkage=True,mechanical_statistics=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--complete',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.base,a.complete),ensure_ascii=False,indent=2))
