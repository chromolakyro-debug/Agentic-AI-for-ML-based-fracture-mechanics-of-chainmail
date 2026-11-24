"""Descriptive paired comparison on the frozen grid, with train-only coverage counts."""
import numpy as np
from utils import HERE,ROOT,read,save,csvwrite,adapter,to_sa,distance_text

def main():
    configs=[read(HERE/f'configs/{tag}.json') for tag in ['old_20k','old_125k']]
    ds=adapter(configs[0]['dataset']['adapter'])(configs[0]['dataset'],ROOT);grid=read(HERE/'configs/grid.json');targets=read(HERE/'configs/targets.json')['targets']
    sa=to_sa(ds.moduli[:,0],ds.moduli[:,1],grid['esoft']);support=[]
    for t in targets:
        row,col=t['grid_row']-1,t['grid_col']-1;slo,shi=grid['s_edges'][col:col+2];alo,ahi=grid['a_edges'][row:row+2]
        ids=ds.train;smask=(sa[ids,0]>=slo)&(sa[ids,0]<shi);a0=(sa[ids,1]>=alo)&(sa[ids,1]<ahi);a1=(-sa[ids,1]>=alo)&(-sa[ids,1]<ahi)
        support.append(dict(target_id=t['target_id'],train_sources_original_orientation=len({ds.source_keys[i] for i in ids[smask&a0]}),
            train_sources_either_orientation=len({ds.source_keys[i] for i in ids[smask&(a0|a1)]})))
    save(HERE/'comparison/training_bin_coverage.json',support)
    overall=[];detail=[];scans=[]
    for cfg in configs:
        out=HERE/'runs'/cfg['run_id'];rr=read(out/'results.json');ss=read(out/'summary.json');valid=[r for r in rr if r['status']=='ok'];a=read(out/'target_retrieval.json')
        distances=np.array([r['d_mech'] for r in valid]);tr1=[q['neighbors'][0]['distance'] for q in a];tr10=[r['distance'] for q in a for r in q['neighbors']]
        data=dict(run_id=cfg['run_id'],samples=len(rr),valid=len(valid),mean_d_mech=float(distances.mean()),median_d_mech=float(np.median(distances)),p90_d_mech=float(np.quantile(distances,.9)),
            mean_within_target_variance=float(np.mean([s['var_d_mech'] for s in ss if s['var_d_mech'] is not None])),
            hits_Ex_Ey=sum(r['hit_Ex_Ey'] for r in valid),hits_s_a=sum(r['hit_s_a'] for r in valid),
            target_retrieval_top1_mean_distance=float(np.mean(tr1)),target_retrieval_top10_mean_distance=float(np.mean(tr10)),
            FEM_selected_best_of_10_mean_distance=float(np.mean([min(r['d_mech'] for r in valid if r['target_id']==t['target_id']) for t in targets])))
        for axis in ['s','a','Ex','Ey']:
            e=np.array([r[f'delta_{axis}'] for r in valid]);data.update({f'{axis}_absolute_error_median':float(np.median(abs(e))),f'{axis}_absolute_error_p90':float(np.quantile(abs(e),.9)),f'{axis}_bias':float(e.mean())})
        for label,test in [('positive_a',lambda t:t['a_target']>0),('negative_a',lambda t:t['a_target']<0),('extreme_a',lambda t:abs(t['a_target'])>.8),('middle_a',lambda t:abs(t['a_target'])<.3)]:
            selected=[r for r in valid if test(r)];data[label]=dict(n=len(selected),mean_d_mech=float(np.mean([r['d_mech'] for r in selected])),hits_s_a=sum(r['hit_s_a'] for r in selected),hits_Ex_Ey=sum(r['hit_Ex_Ey'] for r in selected))
        overall.append(data)
        for t,s,cov,q in zip(targets,ss,support,a):detail.append({'run_id':cfg['run_id'],**t,**s,**cov,'target_retrieval_nearest_distance':q['neighbors'][0]['distance']})
        # Descriptive slopes over each selected scan; no independence test or causal claim.
        for fixed_key,varying in [('grid_row','s'),('grid_col','a')]:
            cross='a' if varying=='s' else 's'
            for idx in sorted({t[fixed_key] for t in targets}):
                selected=[t for t in targets if t[fixed_key]==idx]
                if len(selected)<2:continue
                slopes=[];leaks=[]
                for seed in read(out/'seeds.json'):
                    values=[r for r in valid if r['seed']==seed and r[fixed_key]==idx]
                    if len(values)<2:continue
                    x=[r[varying+'_target'] for r in values]
                    slopes.append(float(np.polyfit(x,[r[varying+'_FEM'] for r in values],1)[0]));leaks.append(float(np.polyfit(x,[r[cross+'_FEM'] for r in values],1)[0]))
                scans.append(dict(run_id=cfg['run_id'],fixed_axis=cross,fixed_bin_index=idx,varying_axis=varying,n_targets=len(selected),n_seeds=len(slopes),
                    response_slope_mean=float(np.mean(slopes)),cross_slope_mean=float(np.mean(leaks)),cross_slope_abs_mean=float(np.mean(abs(np.array(leaks))))))
    save(HERE/'comparison/overall.json',overall);save(HERE/'comparison/per_target.json',detail);csvwrite(HERE/'comparison/per_target.csv',detail);csvwrite(HERE/'comparison/scan_slopes.csv',scans);save(HERE/'comparison/scan_slopes.json',scans)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True);x=np.arange(len(targets))
    for cfg,color in zip(configs,['#697c93','#c25b18']):
        ss=read(HERE/'runs'/cfg['run_id']/'summary.json')
        axes[0].plot(x,[s['mean_d_mech'] for s in ss],'.-',label=cfg['run_id'],color=color);axes[1].plot(x,[s['var_d_mech'] for s in ss],'.-',label=cfg['run_id'],color=color)
    q=read(HERE/'runs/old_20k/target_retrieval.json');axes[0].plot(x,[r['neighbors'][0]['distance'] for r in q],'--',color='black',label='train retrieval top 1 (axis-aware)')
    axes[0].set_ylabel('mean(d_mech)');axes[1].set_ylabel('var(d_mech), ddof=1');axes[0].legend();axes[1].set_xticks(x,[t['target_id'] for t in targets],rotation=45)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Frozen 24-target / 10-seed benchmark');fig.text(.5,.015,distance_text(configs[0]['retrieval']['distance']),ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.96));fig.savefig(HERE/'comparison/paired_comparison.png',dpi=170);plt.close(fig)
    print(overall)

if __name__=='__main__':main()
