"""Three core figures plus bounded internal visual-review sheets."""
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import OUT, HERE, S_TOL, A_TOL, read_json, from_sa, save_json

COLORS=['#287a9e','#60a38b','#9a70b3','#cd843e','#b74f67']
FIGURES=HERE if OUT==(HERE/'run').resolve() else OUT
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.facecolor':'white'})


def save_figure(fig,path,dpi):
    # Write a new file before replacement; avoids truncating an image open in a viewer.
    temporary=path.with_name(path.stem+'.rendering.png')
    fig.savefig(temporary,dpi=dpi)
    temporary.replace(path)


def support():
    p=read_json(OUT/'protocol.json');d=np.load(OUT/'data.npz');sa=d['sa'][d['train']]
    tree=cKDTree((sa-d['mean'])/d['std'])
    s=np.linspace(0,np.log(10),241);a=np.linspace(-1.1,1.1,231)
    ss,aa=np.meshgrid(s,a);points=np.stack([ss.ravel(),aa.ravel()],1)
    r=tree.query((points-d['mean'])/d['std'],k=20)[0][:,-1].reshape(ss.shape)
    exy=from_sa(points).reshape(*ss.shape,2)
    physical=(exy.min(-1)>=1)&(exy.max(-1)<=10)&(exy.max(-1)<=11-10/exy.min(-1))
    threshold=p['support']['threshold']
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
    h=axes[0].hexbin(sa[:,0],sa[:,1],gridsize=(55,42),mincnt=1,cmap='Greys',bins='log',linewidths=0)
    fig.colorbar(h,ax=axes[0],label='Training count per hexagon',shrink=.77)
    axes[0].set_title('A  Training data and frozen scan intervals',loc='left')
    image=axes[1].pcolormesh(ss,aa,np.where(physical,r/threshold,np.nan),vmin=0,vmax=1.3,cmap='YlGnBu_r',shading='auto',rasterized=True)
    axes[1].contour(ss,aa,(r<=threshold)&physical,levels=[.5],colors=['#222222'],linewidths=.8)
    axes[1].set_title('B  Empirical support and final targets',loc='left')
    fig.colorbar(image,ax=axes[1],label='20-NN distance / validation threshold',shrink=.77)
    groups=list(dict.fromkeys(t['group'] for t in p['final_targets']))
    handles=[]
    for gi,name in enumerate(groups):
        group=[t for t in p['final_targets'] if t['group']==name]
        x=[t['s'] for t in group];y=[t['a'] for t in group]
        varying=group[0]['varying'];held='s' if varying=='a' else 'a'
        label=f"vary {varying}, {held}={group[0][held]:.2f}"
        handles.append(Line2D([0],[0],color=COLORS[gi],marker='o',label=label))
        for ax in axes:
            ax.plot(x,y,color=COLORS[gi],lw=1.4,zorder=3)
            for target in group:
                ax.scatter(target['s'],target['a'],marker='o' if target['density']=='dense' else ('s' if target['density']=='medium' else '^'),
                           s=47,color=COLORS[gi],edgecolor='white',linewidth=.6,zorder=5)
        for j,target in enumerate(group):
            axes[1].annotate(target['target_id'][1:],(target['s'],target['a']),xytext=((6 if gi%2 else -15),3 if j%2 else -11),
                            textcoords='offset points',fontsize=7,color=COLORS[gi],bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=.5))
    for ax in axes:
        ax.set(xlim=(0,np.log(10)),ylim=(-1.1,1.1),xlabel='s = log geometric-mean modulus',ylabel='a = log(Ex/Ey)')
        ax.axhline(0,color='grey',lw=.6,alpha=.4)
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('CFM v0 | 6,400 training samples; 20 targets fixed before model evaluation\n'
                 'Circle: dense; square: medium; triangle: sparse. Support is empirical, not proof of attainability.',fontsize=12)
    save_figure(fig,FIGURES/'support_and_targets.png',180);plt.close(fig)


def review(folder):
    path=OUT/folder;m=read_json(path/'manifest.json');images=np.load(path/'binary.npy')
    seeds=m['seeds'][:2]
    selected=[i for i,r in enumerate(m['samples']) if r['seed'] in seeds]
    columns=4 if len(selected)<=8 else 5
    nrows=int(np.ceil(len(selected)/columns))
    fig,axes=plt.subplots(nrows,columns,figsize=(columns*2.6,nrows*3.0),squeeze=False)
    for ax in axes.ravel():ax.axis('off')
    for ax,i in zip(axes.ravel(),selected):
        row=m['samples'][i];ax.imshow(images[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(f"{row['target_id']} seed {row['seed']}\ns={row['s']:.3f}, a={row['a']:.3f}",fontsize=9)
    fig.subplots_adjust(hspace=.23,wspace=.10,top=.95,bottom=.02,left=.02,right=.98)
    save_figure(fig,path/'visual_review_fixed_seeds.png',110);plt.close(fig)
    save_json(path/'visual_review_manifest.json',dict(indices=selected,seeds=seeds,
              selection='First two frozen seeds for every target, no property/appearance filtering',
              review_status='pending actual visual inspection'))


def final_figures():
    folder=OUT/'final';metrics=read_json(folder/'metrics.json');rows=read_json(folder/'samples.json')
    p=read_json(OUT/'protocol.json');images=np.load(folder/'binary.npy')
    fig,axes=plt.subplots(2,5,figsize=(16,7.2))
    for col,scan in enumerate(metrics['scans']):
        varying=scan['varying'];held=scan['held'];group=[t for t in metrics['per_target'] if t['group']==scan['group']]
        x=np.array([t[varying] for t in group]);color=COLORS[col]
        for row_index,property in enumerate((varying,held)):
            ax=axes[row_index,col]
            for target in group:
                vals=np.array([r[property+'_hat'] for r in rows if r['target_id']==target['target_id'] and r['fem_valid']])
                if len(vals):
                    ax.scatter(np.full(len(vals),target[varying]),vals,s=7,color=color,alpha=.2,lw=0)
                    low,median,high=np.quantile(vals,[.1,.5,.9])
                    ax.errorbar(target[varying],median,yerr=np.array([[median-low],[high-median]]),fmt='o',color=color,capsize=3,ms=5)
                ax.scatter(target[varying],target['retrieval'][property],s=34,marker='D',facecolors='none',edgecolors='#d16c24',lw=1.1,zorder=5)
            ideal=x if property==varying else np.full_like(x,scan['fixed_value'])
            ax.plot(x,ideal,'k--',lw=1)
            tol=S_TOL if property=='s' else A_TOL
            ax.fill_between(x,ideal-tol,ideal+tol,color='grey',alpha=.1)
            ax.set(xlabel=f'target {varying}',ylabel=f'FEM {property}_hat')
            ax.grid(alpha=.15)
            if row_index==0:
                ax.set_title(f"{held} fixed at {scan['fixed_value']:.3f}\nresponse slope {scan.get('main_slope_mean',float('nan')):.2f}",fontsize=10)
            else:
                ax.set_title(f"cross slope {scan.get('cross_slope_mean',float('nan')):.3f}",fontsize=10)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='#287a9e',label='CFM median + P10-P90; dots = all valid seeds'),
                        Line2D([0],[0],marker='D',color='#d16c24',mfc='none',ls='',label='Training nearest retrieval'),
                        Line2D([0],[0],color='black',ls='--',label='Ideal; shaded = fixed hit tolerance')],
               loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('CFM v0 | Final binary images, original FEM, same 32 noise seeds per target',fontsize=14)
    fig.tight_layout(rect=(0,.07,1,.95));save_figure(fig,FIGURES/'controllability.png',180);plt.close(fig)
    # Eight predetermined examples + four FEM-selected illustrations, explicitly labelled.
    fixed_ids=['T00','T05','T12','T19'];seeds=p['final_noise_seeds'][:2]
    chosen=[i for target in fixed_ids for i,r in enumerate(rows) if r['target_id']==target and r['seed'] in seeds]
    candidates=[i for i,r in enumerate(rows) if r['seed'] in seeds and r['fem_valid'] and i not in chosen]
    error=lambda i:(rows[i]['delta_s']/S_TOL)**2+(rows[i]['delta_a']/A_TOL)**2
    order=sorted(candidates,key=error)
    chosen+=order[:2]+order[-2:]
    fig,axes=plt.subplots(3,4,figsize=(12,13))
    for j,(ax,i) in enumerate(zip(axes.ravel(),chosen)):
        r=rows[i];ax.imshow(images[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        status='HIT' if r['hit_joint'] else ('FEM FAIL' if not r['fem_valid'] else 'MISS')
        tag='fixed example' if j<8 else 'FEM-selected illustration'
        ax.set_title(f"{r['target_id']} / seed {r['seed']} / {status}\n{tag}",fontsize=9,color='#218263' if r['hit_joint'] else '#ad4d44')
        text=f"target (s,a)=({r['s']:.3f}, {r['a']:.3f})\nFEM ({r.get('s_hat',float('nan')):.3f}, {r.get('a_hat',float('nan')):.3f})"
        ax.set_xlabel(text,fontsize=9)
    fig.suptitle('Generated binary material maps | Black = hard, white = soft\n'
                 'Top 8: predetermined targets/seeds. Bottom 4: low/high error among the fixed visual-review seeds.',fontsize=12)
    fig.subplots_adjust(left=.035,right=.985,bottom=.065,top=.90,hspace=.45,wspace=.13)
    save_figure(fig,FIGURES/'generated_examples.png',180);plt.close(fig)
    save_json(folder/'figure_examples.json',dict(indices=chosen,rule='8 predeclared target/seed pairs plus 2 smallest and 2 largest FEM scaled errors among first two seeds',
            note='Last 4 are explicitly FEM-selected illustrations, never aggregate generation performance'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['support','review','final']);p.add_argument('--folder',default='final');a=p.parse_args()
    if a.mode=='support':support()
    elif a.mode=='review':review(a.folder)
    else:final_figures()
