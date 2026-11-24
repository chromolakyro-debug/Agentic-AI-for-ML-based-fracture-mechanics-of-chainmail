"""Compact scientific figures using stored evidence only; no model/FEM surrogate."""
import paths
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT,DATA,read_json,save_json,S_TOL,A_TOL
from plots import save_figure


def history():
    old=[json.loads(line) for line in (paths.ORIGINAL/'training.jsonl').read_text().splitlines() if line.strip()]
    new=[json.loads(line) for line in (OUT/'training.jsonl').read_text().splitlines() if line.strip()]
    result=[];batch=read_json(OUT/'continuation_config.json')['batch']
    for phase,rows in [('v0',old),('continuation',new)]:
        for row in rows:
            r=dict(row,phase=phase)
            r['images_seen']=row.get('images_seen',row['step']*16)
            r['equivalent_epochs']=r['images_seen']/6400
            r['batch']=16 if phase=='v0' else batch
            result.append(r)
    return result


def curves():
    rows=history();fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'complete_training_history.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    save_json(OUT/'complete_training_history.json',rows)
    training=[r for r in rows if 'loss' in r];validation=[r for r in rows if 'validation_velocity_mse' in r]
    x=np.array([r['step'] for r in training])/1000;y=np.array([r['loss'] for r in training])
    fig,axes=plt.subplots(3,1,figsize=(11,9.2),sharex=True,gridspec_kw={'height_ratios':[2,1,1.25]})
    axes[0].plot(x,y,color='#c6dbe7',lw=.7,label='Training: each 100-step average')
    if len(y)>=10:axes[0].plot(x[9:],np.convolve(y,np.ones(10)/10,mode='valid'),color='#287a9e',lw=1.2,label='Training: rolling 1,000-step average')
    axes[0].plot([r['step']/1000 for r in validation],[r['validation_velocity_mse'] for r in validation],'-o',color='#c46d36',ms=3,lw=1.3,label='Fixed-noise validation EMA velocity MSE')
    axes[0].set(ylabel='Velocity MSE',yscale='log');axes[0].legend(fontsize=8,ncol=2)
    continued=[r for r in validation if r['phase']=='continuation']
    if len(continued)>=2:
        zoom=axes[0].inset_axes([.43,.35,.54,.43])
        mask=x>=20
        if np.sum(mask)>=10:zoom.plot(x[mask][9:],np.convolve(y[mask],np.ones(10)/10,mode='valid'),color='#287a9e',lw=1)
        zoom.plot([r['step']/1000 for r in continued],[r['validation_velocity_mse'] for r in continued],'-o',color='#c46d36',ms=2,lw=1)
        zoom.set_title('Continuation zoom (linear MSE)',fontsize=8);zoom.tick_params(labelsize=7);zoom.grid(alpha=.15)
    axes[1].plot(x,[r['lr'] for r in training],color='#74629c',lw=1.5);axes[1].set(ylabel='Learning rate')
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    milestones=[]
    for p in sorted(OUT.glob('development_*/metrics.json')):
        m=read_json(p);vpath=p.parent/'visual_review.json'
        if not vpath.exists():continue
        v=read_json(vpath);milestones.append((m['step']/1000,m['overall']['hit_joint'],v['counts'].get('2',0)/30))
    if milestones:
        xx,mech,visual=np.asarray(milestones).T
        axes[2].plot(xx,mech,'o-',color='#4c8d71',label='Development FEM joint hit (30 samples)')
        axes[2].plot(xx,visual,'s-',color='#ad6278',label='AI-rated coherent cat (30 fixed images)')
    axes[2].set(ylabel='Fraction (development)',xlabel='Cumulative optimizer steps (thousands)',ylim=(0,1.05));axes[2].legend(fontsize=8,ncol=2)
    for ax in axes:
        ax.axvline(20,color='#555555',ls=':',lw=1);ax.grid(alpha=.15)
    fig.suptitle('CFM continuation: training, validation, LR and fixed development checks\nWeighted training MSE and unweighted validation MSE have different sampling; AI visual ratings are subjective.',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.95]);save_figure(fig,paths.HERE/'training_curves.png',150);plt.close(fig)


def support():
    p=read_json(OUT/'protocol.json');d=np.load(OUT/'data.npz');sa=d['sa'][d['train']]
    from scipy.spatial import cKDTree
    s=np.linspace(.01,np.log(10)-.01,240);a=np.linspace(-1.1,1.1,240);ss,aa=np.meshgrid(s,a)
    points=np.c_[ss.ravel(),aa.ravel()];tree=cKDTree((sa-d['mean'])/d['std'])
    r=tree.query((points-d['mean'])/d['std'],k=20)[0][:,-1].reshape(ss.shape)/p['support']['threshold']
    exy=np.exp(np.stack([ss+aa/2,ss-aa/2],-1));physical=(exy.min(-1)>=1)&(exy.max(-1)<=10)&(exy.max(-1)<=11-10/exy.min(-1))
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    axes[0].hexbin(sa[:,0],sa[:,1],gridsize=60,mincnt=1,bins='log',cmap='Greys',linewidths=0)
    plot=axes[1].pcolormesh(ss,aa,np.where(physical,r,np.nan),cmap='YlGnBu_r',vmin=0,vmax=1.3,shading='auto');axes[1].contour(ss,aa,((r<=1)&physical),levels=[.5],colors='black',linewidths=.7)
    fig.colorbar(plot,ax=axes[1],label='20-NN distance / frozen support threshold',shrink=.8)
    for ax in axes:
        for name,color,marker in [('development','#da7538','s'),('final','#287e9d','o')]:
            tt=p[name+'_targets'];ax.scatter([t['s'] for t in tt],[t['a'] for t in tt],s=28,marker=marker,facecolors='none' if name=='final' else color,edgecolors=color,label=name)
        ax.set(xlabel='s',ylabel='a',xlim=(0,np.log(10)),ylim=(-1.1,1.1));ax.legend(fontsize=8)
    axes[0].set_title('Training-only label distribution and frozen panels')
    axes[1].set_title('Empirical support (not physical attainability)')
    fig.suptitle('15 development targets x 2 seeds | 20 historical final targets x 32 seeds',fontsize=12)
    save_figure(fig,paths.HERE/'support_and_targets.png',150);plt.close(fig)


def comparison():
    old=read_json(paths.ORIGINAL/'final'/'metrics.json');new=read_json(OUT/'final'/'metrics.json')
    samples=read_json(OUT/'final'/'samples.json');by_old={t['target_id']:t for t in old['per_target']}
    fig,axes=plt.subplots(2,5,figsize=(16,7))
    for col,scan in enumerate(new['scans']):
        varying=scan['varying'];held=scan['held'];group=[t for t in new['per_target'] if t['group']==scan['group']]
        x=np.array([t[varying] for t in group])
        for row,axis in enumerate((varying,held)):
            ax=axes[row,col]
            for t in group:
                rr=[r for r in samples if r['target_id']==t['target_id'] and r['fem_valid']]
                ax.scatter(np.full(len(rr),t[varying]),[r[axis+'_hat'] for r in rr],color='#327e9b',alpha=.16,s=9)
            for values,color,label in [(group,'#327e9b','Continuation'),([by_old[t['target_id']] for t in group],'#a5a5a5','20k baseline')]:
                ax.errorbar(x,[t[axis]+t['bias_'+axis] for t in values],yerr=[t['std_'+axis] for t in values],color=color,fmt='o-',capsize=2,ms=3,lw=1,label=label)
            ax.plot(x,[t['retrieval'][axis] for t in group],color='#479470',marker='x',ls='--',ms=4,lw=.9,label='Train retrieval')
            ideal=x if row==0 else np.full(len(x),scan['fixed_value']);tol=S_TOL if axis=='s' else A_TOL
            ax.plot(x,ideal,'k:',lw=.8,label='Target');ax.fill_between(x,ideal-tol,ideal+tol,color='black',alpha=.04)
            ax.set(xlabel=f'Target {varying}',ylabel=f'FEM {axis}');ax.grid(alpha=.12)
            if row==0:ax.set_title(f"{held} = {scan['fixed_value']:.3f}\nmain gain {scan.get('main_slope_mean',float('nan')):.2f}",fontsize=10)
            else:ax.set_title(f"cross slope {scan.get('cross_slope_mean',float('nan')):+.3f}\nheld range median {scan.get('held_range_median',float('nan')):.3f}",fontsize=9)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9)
    fig.suptitle('Historical final benchmark: same targets and 32 shared seeds\nDots: all valid continuation samples; error bars: mean +/- sample SD. Invalid FEM counts as failed hit.',fontsize=12)
    fig.tight_layout(rect=[0,.045,1,.92]);save_figure(fig,paths.HERE/'controllability_comparison.png',155);plt.close(fig)


def examples():
    rules=read_json(paths.HERE/'evaluation_rules.json');new=read_json(OUT/'final'/'samples.json');old=read_json(paths.ORIGINAL/'final'/'samples.json')
    lookup_new={r['key']:i for i,r in enumerate(new)};lookup_old={r['key']:i for i,r in enumerate(old)}
    chosen=rules['fixed_example_keys'][:]
    extra=[r for r in new if r['seed'] in (730000,730001) and r['key'] not in chosen]
    extra.sort(key=lambda r:(r['delta_s']/S_TOL)**2+(r['delta_a']/A_TOL)**2 if r['fem_valid'] else float('inf'),reverse=True)
    chosen += [r['key'] for r in extra[:2]]
    oldim=np.load(paths.ORIGINAL/'final'/'binary.npy');newim=np.load(OUT/'final'/'binary.npy')
    fig,axes=plt.subplots(4,4,figsize=(12,13))
    for j,key in enumerate(chosen):
        row=j//2;col=(j%2)*2
        for offset,lookup,records,images,label in [(0,lookup_old,old,oldim,'20k'),(1,lookup_new,new,newim,'Continuation')]:
            i=lookup[key];r=records[i];ax=axes[row,col+offset];ax.imshow(images[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
            actual=f"FEM ({r['s_hat']:.3f},{r['a_hat']:+.3f})" if r['fem_valid'] else 'FEM failed'
            status='HIT' if r['hit_joint'] else 'MISS';tag='fixed' if j<6 else 'FEM-selected high error'
            ax.set_title(f"{label} | {key} | {status}\ntarget ({r['s']:.3f},{r['a']:+.3f})\n{actual} | {tag}",fontsize=8)
    fig.suptitle('Binary outputs: paired visual examples (black hard, white soft)\nFirst 6 pairs fixed before continuation final generation; last 2 explicitly FEM-selected, not a random-sample quality estimate.',fontsize=11)
    fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.90,hspace=.36,wspace=.04)
    save_figure(fig,paths.HERE/'generated_examples.png',145);plt.close(fig)
    save_json(OUT/'final'/'figure_examples.json',dict(keys=chosen,fixed_count=6,selection=rules['additional_example_rule']))


def neighbor_examples():
    samples=read_json(OUT/'final'/'samples.json');nn=read_json(OUT/'final'/'image_neighbors.json')['results']
    lookup={r['key']:i for i,r in enumerate(samples)};neighbors={r['key']:r for r in nn}
    retrieval={r['target_id']:r for r in read_json(OUT/'retrieval.json')['results']}
    chosen=['T00_seed730000','T05_seed730000','T16_seed730000']
    closest=min([r for r in nn if r['descriptor_valid']],key=lambda r:r['shape_pooled_rms'])['key']
    if closest not in chosen:chosen.append(closest)
    training=np.load(DATA/'dataset.npy',mmap_mode='r');generated=np.load(OUT/'final'/'binary.npy');d=np.load(OUT/'data.npz')
    fig,axes=plt.subplots(len(chosen),3,figsize=(9.5,len(chosen)*3.2));records=[]
    for j,key in enumerate(chosen):
        r=samples[lookup[key]];n=neighbors[key];attribute=retrieval[r['target_id']]['training_index'];shape=n['shape_nearest_training_index']
        entry=dict(key=key,attribute_training_index=attribute,shape_training_index=shape,shape_rms=n['shape_pooled_rms'],
                   selection='Fixed illustrative key' if j<3 else 'Post-hoc closest normalized shape among all 640; diagnostic extreme, not prevalence estimate')
        records.append(entry)
        for ax,im,label,s,a in zip(axes[j],[generated[lookup[key]],training[shape],training[attribute]],
                [f'Generated {key}',f'Train shape-NN #{shape}',f'Train attribute-NN #{attribute}'],
                [r.get('s_hat',float('nan')),d['sa'][shape,0],d['sa'][attribute,0]],
                [r.get('a_hat',float('nan')),d['sa'][shape,1],d['sa'][attribute,1]]):
            ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off');ax.set_title(f'{label}\nFEM (s,a)=({s:.3f},{a:+.3f})',fontsize=8)
        axes[j,0].set_xlabel(f"shape RMS={n['shape_pooled_rms']:.3f}"+(' | post-hoc extreme' if j==3 else ''),fontsize=8)
    fig.suptitle('Original binary pixels: generated vs training-only neighbors\nFirst 3 keys fixed in code before final generation; last row is an explicitly selected similarity extreme.',fontsize=11)
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.89,hspace=.32,wspace=.06)
    save_figure(fig,paths.HERE/'training_neighbor_examples.png',140);plt.close(fig)
    save_json(OUT/'final'/'neighbor_examples.json',dict(records=records,caveat='Different pixels or different nearest attributes do not prove meaningful structural novelty. Shape normalization is used only to find neighbors; displayed material maps are original pixels.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['curves','support','final']);a=p.parse_args()
    if a.mode=='curves':curves()
    elif a.mode=='support':support()
    else:curves();comparison();examples();neighbor_examples()
