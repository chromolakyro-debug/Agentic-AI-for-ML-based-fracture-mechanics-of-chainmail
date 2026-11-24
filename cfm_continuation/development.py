"""Fixed unfiltered development images, real FEM, auxiliary geometry and exact train NN."""
import paths
import argparse
import csv
import json
import numpy as np
from scipy import ndimage
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT, DATA, read_json, save_json, S_TOL, A_TOL
from plots import save_figure


def analyse(step):
    folder=OUT/f'development_{step:06d}';m=read_json(folder/'manifest.json');images=np.load(folder/'binary.npy')
    fem={}
    for line in (folder/'fem.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            r=json.loads(line);fem[r['key'],r['sha256_pixels']]=r
    d=np.load(OUT/'data.npz');tr=d['train'];sa=d['sa'][tr];training=np.load(DATA/'dataset.npy',mmap_mode='r')[tr]
    with threadpool_limits(8,user_api='blas'):
        xx=training.reshape(len(tr),-1).astype(np.float32);qq=images.reshape(len(images),-1).astype(np.float32)
        distance=(qq.sum(1)[:,None]+xx.sum(1)[None]-2*qq@xx.T)/(224*224)
    nearest=distance.argmin(1);minimum=distance.min(1);rows=[]
    for i,source in enumerate(m['samples']):
        r=source|fem[source['key'],source['sha256_pixels']];im=images[i]
        if r['fem_valid']:
            r.update(delta_s=r['s_hat']-r['s'],delta_a=r['a_hat']-r['a'])
            r['hit_joint']=abs(r['delta_s'])<=S_TOL and abs(r['delta_a'])<=A_TOL
        else:r.update(delta_s=None,delta_a=None,hit_joint=False)
        edge=np.r_[im[0],im[-1],im[1:-1,0],im[1:-1,-1]];bg=int(edge.mean()>=.5);ink=im!=bg
        lab,n=ndimage.label(ink,structure=np.ones((3,3)));areas=np.bincount(lab.ravel())[1:]
        r.update(inferred_background=bg,ink_fraction=float(ink.mean()),components_ge_16px=int(np.sum(areas>=16)),
                 largest_component_ink_fraction=float(areas.max()/ink.sum()) if len(areas) else 0.,
                 foreground_boundary_fraction=float(np.mean(edge!=bg)),pixel_nearest_training_index=int(tr[nearest[i]]),
                 pixel_hamming_fraction=float(minimum[i]),exact_pixel_duplicate=bool(minimum[i]==0))
        rows.append(r)
    valid=[r for r in rows if r['fem_valid']];lookup={(r['target_id'],r['seed']):r for r in rows}
    overall=dict(n=len(rows),valid=len(valid),failure_rate=1-len(valid)/len(rows),hit_joint=sum(r['hit_joint'] for r in rows)/len(rows))
    for axis in ('s','a'):
        v=np.array([r['delta_'+axis] for r in valid]);overall['abs_delta_'+axis]=dict(median=float(np.median(abs(v))),p90=float(np.quantile(abs(v),.9)),bias=float(v.mean())) if len(v) else None
    scans=[];retrieval=[]
    for target in m['targets']:
        j=int(np.argmin(np.linalg.norm((sa-[target['s'],target['a']])/[S_TOL,A_TOL],axis=1)))
        ds,da=sa[j]-[target['s'],target['a']]
        retrieval.append(dict(target_id=target['target_id'],training_index=int(tr[j]),s=float(sa[j,0]),a=float(sa[j,1]),delta_s=float(ds),delta_a=float(da),hit_joint=bool(abs(ds)<=S_TOL and abs(da)<=A_TOL)))
    for group in dict.fromkeys(t['group'] for t in m['targets']):
        targets=[t for t in m['targets'] if t['group']==group];axis=targets[0]['varying'];held='s' if axis=='a' else 'a'
        x=np.array([t[axis] for t in targets]);xc=x-x.mean();per_seed=[]
        for seed in m['seeds']:
            rr=[lookup[t['target_id'],seed] for t in targets]
            if all(r['fem_valid'] for r in rr):
                main=np.array([r[axis+'_hat'] for r in rr]);cross=np.array([r[held+'_hat'] for r in rr])
                per_seed.append(dict(seed=seed,main_slope=float(xc@main/(xc@xc)),cross_slope=float(xc@cross/(xc@xc)),held_range=float(np.ptp(cross))))
        scans.append(dict(group=group,varying=axis,held=held,fixed=targets[0][held],per_seed=per_seed,
                          caveat='Only two paired seeds; development diagnostic, not precise inference'))
    diversity=[]
    for target in m['targets']:
        ii=[i for i,r in enumerate(rows) if r['target_id']==target['target_id']]
        diversity.append(dict(target_id=target['target_id'],pairwise_hamming=float(np.mean(images[ii[0]]!=images[ii[1]])),unique_hashes=len(set(rows[i]['sha256_pixels'] for i in ii))))
    aux=dict(pixel_nearest_hamming_median=float(np.median(minimum)),pixel_nearest_hamming_min=float(minimum.min()),
             exact_duplicates=int(np.sum(minimum==0)),median_components_ge_16px=float(np.median([r['components_ge_16px'] for r in rows])),
             median_largest_component_ink_fraction=float(np.median([r['largest_component_ink_fraction'] for r in rows])),
             median_foreground_boundary_fraction=float(np.median([r['foreground_boundary_fraction'] for r in rows])),
             caveat='Background-inferred connected components, boundary contact and raw pixel distances are auxiliary style-sensitive diagnostics, NOT cat recognition or human visual confirmation.')
    save_json(folder/'metrics.json',dict(step=step,overall=overall,scans=scans,retrieval=retrieval,diversity=diversity,auxiliary=aux))
    save_json(folder/'samples.json',rows)
    with (folder/'samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['key','target_id','seed','s','a','s_hat','a_hat','delta_s','delta_a','hit_joint','fem_valid','pixel_nearest_training_index','pixel_hamming_fraction','components_ge_16px','largest_component_ink_fraction','foreground_boundary_fraction','sha256_pixels']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(5,6,figsize=(13.8,12.8))
    for ax,im,r in zip(axes.ravel(),images,rows):
        target=next(t for t in m['targets'] if t['target_id']==r['target_id'])
        ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
        actual=f"FEM ({r['s_hat']:.2f},{r['a_hat']:+.2f}) {'HIT' if r['hit_joint'] else 'MISS'}" if r['fem_valid'] else 'FEM FAILED'
        ax.set_title(f"{r['target_id']} seed {r['seed']} | {target['density']}\ntarget ({r['s']:.2f},{r['a']:+.2f})\n{actual}",fontsize=8)
    fig.suptitle(f'CFM continuation | {step:,} steps | ALL 30 fixed development samples\nSame targets, seeds, midpoint64 and zero threshold; no visual/FEM filtering',fontsize=12)
    fig.subplots_adjust(left=.012,right=.988,bottom=.012,top=.90,hspace=.33,wspace=.04)
    save_figure(fig,folder/'fixed_samples.png',130);plt.close(fig)
    print(json.dumps({'step':step,'overall':overall,'auxiliary':aux}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);analyse(p.parse_args().step)
