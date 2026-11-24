"""Frozen final-panel mechanics metrics and one-time held-out test loss."""
import argparse
import csv
import json
import numpy as np
from common import OUT, DATA, S_TOL, A_TOL, read_json, save_json


def scalar_summary(values):
    v=np.asarray(values,dtype=float)
    v=v[np.isfinite(v)]
    if not len(v): return dict(valid=0,median=None,p90=None,mean=None)
    return dict(valid=len(v),median=float(np.median(v)),p90=float(np.quantile(v,.9)),mean=float(np.mean(v)))


def mechanics_metrics():
    folder=OUT/'final';manifest=read_json(folder/'manifest.json')
    fem={}
    for line in (folder/'fem.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            fem[(r['key'],r['sha256_pixels'])]=r
    targets=manifest['targets'];retrieval=read_json(OUT/'retrieval.json')['results']
    retrieval={r['target_id']:r for r in retrieval}
    rows=[]
    for source in manifest['samples']:
        r=source | fem[(source['key'],source['sha256_pixels'])]
        valid=r['fem_valid']
        ds=r['s_hat']-r['s'] if valid else None
        da=r['a_hat']-r['a'] if valid else None
        r.update(delta_s=ds,delta_a=da,abs_delta_s=abs(ds) if valid else None,
                 abs_delta_a=abs(da) if valid else None,
                 hit_s=bool(valid and abs(ds)<=S_TOL),hit_a=bool(valid and abs(da)<=A_TOL),
                 hit_joint=bool(valid and abs(ds)<=S_TOL and abs(da)<=A_TOL))
        rows.append(r)
    lookup={(r['target_id'],r['seed']):r for r in rows}
    per_target=[]
    for target in targets:
        group=[r for r in rows if r['target_id']==target['target_id']]
        valid=[r for r in group if r['fem_valid']]
        held='s' if target['varying']=='a' else 'a'
        first=next(t for t in targets if t['group']==target['group'])
        paired=[]
        for r in group:
            baseline=lookup[(first['target_id'],r['seed'])]
            r['paired_cross_drift_from_scan_start']=r[held+'_hat']-baseline[held+'_hat'] if r['fem_valid'] and baseline['fem_valid'] else None
            if r['paired_cross_drift_from_scan_start'] is not None:paired.append(r['paired_cross_drift_from_scan_start'])
        summary=target | dict(n=len(group),valid=len(valid),failure_rate=1-len(valid)/len(group),
            abs_delta_s=scalar_summary([abs(r['delta_s']) for r in valid]),
            abs_delta_a=scalar_summary([abs(r['delta_a']) for r in valid]),
            bias_s=float(np.mean([r['delta_s'] for r in valid])) if valid else None,
            bias_a=float(np.mean([r['delta_a'] for r in valid])) if valid else None,
            std_s=float(np.std([r['s_hat'] for r in valid],ddof=1)) if len(valid)>1 else None,
            std_a=float(np.std([r['a_hat'] for r in valid],ddof=1)) if len(valid)>1 else None,
            hit_s=sum(r['hit_s'] for r in group)/len(group),hit_a=sum(r['hit_a'] for r in group)/len(group),
            hit_joint=sum(r['hit_joint'] for r in group)/len(group),
            paired_cross_drift=scalar_summary(np.abs(paired)),retrieval=retrieval[target['target_id']])
        if valid:
            best=min(valid,key=lambda r:(r['delta_s']/S_TOL)**2+(r['delta_a']/A_TOL)**2)
            summary['best_of_32_fem_selected']=dict(key=best['key'],seed=best['seed'],delta_s=best['delta_s'],delta_a=best['delta_a'],hit=best['hit_joint'])
        per_target.append(summary)
    scans=[];rng=np.random.default_rng(2026111501)
    for name in dict.fromkeys(t['group'] for t in targets):
        group=[t for t in targets if t['group']==name]
        varying=group[0]['varying'];held='s' if varying=='a' else 'a'
        x=np.array([t[varying] for t in group]);xx=x-x.mean()
        values=[];used=[]
        for seed in manifest['seeds']:
            seed_rows=[lookup[(t['target_id'],seed)] for t in group]
            if not all(r['fem_valid'] for r in seed_rows):continue
            main=np.array([r[varying+'_hat'] for r in seed_rows])
            cross=np.array([r[held+'_hat'] for r in seed_rows])
            values.append([float(xx@main/(xx@xx)),float(xx@cross/(xx@xx)),float(np.ptp(cross)),float(np.ptp(main))]);used.append(seed)
        values=np.asarray(values).reshape(-1,4)
        summary=dict(group=name,varying=varying,held=held,fixed_value=group[0][held],n_complete_seeds=len(values),
                     n_failed_or_incomplete_seeds=len(manifest['seeds'])-len(values),seeds=used,
                     target_range=float(np.ptp(x)),per_seed_slopes_and_ranges=values.tolist())
        if len(values):
            bootstrap=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
            ratio=(A_TOL/S_TOL) if varying=='a' else (S_TOL/A_TOL)
            summary.update(main_slope_mean=float(values[:,0].mean()),cross_slope_mean=float(values[:,1].mean()),
                    main_slope_95ci=np.quantile(bootstrap[:,0],[.025,.975]).tolist(),
                    cross_slope_95ci=np.quantile(bootstrap[:,1],[.025,.975]).tolist(),
                    tolerance_normalized_cross_slope=float(values[:,1].mean()*ratio),
                    held_range_median=float(np.median(values[:,2])))
        ry=np.array([retrieval[t['target_id']][varying] for t in group])
        rc=np.array([retrieval[t['target_id']][held] for t in group])
        summary.update(retrieval_main_slope=float(xx@ry/(xx@xx)),retrieval_cross_slope=float(xx@rc/(xx@xx)))
        scans.append(summary)
    density=[]
    for category in ('dense','medium','sparse'):
        ids={t['target_id'] for t in targets if t['density']==category};sub=[r for r in rows if r['target_id'] in ids]
        valid=[r for r in sub if r['fem_valid']]
        density.append(dict(density=category,targets=len(ids),n=len(sub),valid=len(valid),
                    hit_joint=sum(r['hit_joint'] for r in sub)/max(len(sub),1),
                    abs_delta_s=scalar_summary([abs(r['delta_s']) for r in valid]),
                    abs_delta_a=scalar_summary([abs(r['delta_a']) for r in valid]),
                    retrieval_hit_targets=sum(retrieval[t]['hit'] for t in ids)))
    valid=[r for r in rows if r['fem_valid']]
    overall=dict(n=len(rows),valid=len(valid),failure_rate=1-len(valid)/len(rows),
                 hit_joint=sum(r['hit_joint'] for r in rows)/len(rows),hit_s=sum(r['hit_s'] for r in rows)/len(rows),
                 hit_a=sum(r['hit_a'] for r in rows)/len(rows),
                 abs_delta_s=scalar_summary([abs(r['delta_s']) for r in valid]),
                 abs_delta_a=scalar_summary([abs(r['delta_a']) for r in valid]),
                 bias_s=float(np.mean([r['delta_s'] for r in valid])) if valid else None,
                 bias_a=float(np.mean([r['delta_a'] for r in valid])) if valid else None,
                 retrieval_hit_targets=sum(r['hit'] for r in retrieval.values()),
                 retrieval_abs_delta_s=scalar_summary([abs(r['delta_s']) for r in retrieval.values()]),
                 retrieval_abs_delta_a=scalar_summary([abs(r['delta_a']) for r in retrieval.values()]),
                 targets_with_fem_selected_best_hit=sum(t.get('best_of_32_fem_selected',{}).get('hit',False) for t in per_target))
    save_json(folder/'metrics.json',dict(overall=overall,per_target=per_target,scans=scans,density=density,
                failure_policy='Invalid FEM counts as hit failure. Error quantiles and slopes conditional on valid outputs; counts always shown.',
                slope_definition='OLS per seed across each fixed scan; mean and 2000 paired-seed bootstrap CI. Cross slope is d(held actual)/d(varying target).'))
    save_json(folder/'samples.json',rows)
    fields=['key','target_id','seed','s','a','s_hat','a_hat','Ex_app','Ey_app','delta_s','delta_a','hit_s','hit_a','hit_joint',
            'paired_cross_drift_from_scan_start','fem_valid','solve_seconds','hard_fraction','residual_x','residual_y',
            'iterations_x','iterations_y','energy_relative_error_x','energy_relative_error_y','force_balance_relative_error_x',
            'force_balance_relative_error_y','fraction_near_threshold_0p1','fraction_near_threshold_0p25','sha256_pixels','error']
    with (folder/'samples.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    flat=[]
    for target in per_target:
        record={key:target[key] for key in ('target_id','group','varying','s','a','density','support_r20','n','valid',
                    'failure_rate','bias_s','bias_a','std_s','std_a','hit_s','hit_a','hit_joint')}
        for name in ('abs_delta_s','abs_delta_a','paired_cross_drift'):
            for statistic in ('median','p90','mean'):record[name+'_'+statistic]=target[name][statistic]
        record.update(retrieval_training_index=target['retrieval']['training_index'],
                      retrieval_delta_s=target['retrieval']['delta_s'],retrieval_delta_a=target['retrieval']['delta_a'],
                      retrieval_hit=target['retrieval']['hit'])
        flat.append(record)
    with (folder/'target_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(overall,indent=2),flush=True)


def heldout_test():
    # Only invoke after the main run and final generation; never selects weights or settings.
    assert (OUT/'final'/'manifest.json').exists()
    import torch
    from train import validation_loss
    from generate import load_model
    model,checkpoint=load_model(OUT/'checkpoints'/'best.pt')
    d=np.load(OUT/'data.npz');images=np.load(DATA/'dataset.npy',mmap_mode='r')
    te=torch.as_tensor(images[d['test']],device='cuda')
    tc=torch.as_tensor(((d['sa'][d['test']]-d['mean'])/d['std']).astype(np.float32),device='cuda')
    loss=validation_loss(model,te,tc)
    save_json(OUT/'test_once.json',dict(n=800,checkpoint_step=checkpoint['step'],velocity_mse=loss,
              role='Final heldout forward loss only; not used for selection, targets, weights, support or tuning'))
    print('Test velocity MSE',loss,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mechanics','test']);a=p.parse_args()
    if a.mode=='mechanics':mechanics_metrics()
    else:heldout_test()
