import argparse
import numpy as np
from utils import HERE,read,save,sha,from_sa,csvwrite

def make_targets(cells,grid):
    if len(cells)!=grid['expected_targets'] or len({tuple(c) for c in cells})!=len(cells):
        raise ValueError('Exactly 24 distinct user-selected cells required')
    targets=[]
    for i,(r,c) in enumerate(cells,1):
        if not isinstance(r,int) or not isinstance(c,int) or not (1<=r<len(grid['a_edges']) and 1<=c<len(grid['s_edges'])):raise ValueError('Invalid one-based cell')
        s=float(np.mean(grid['s_edges'][c-1:c+1]));a=float(np.mean(grid['a_edges'][r-1:r+1]));ex,ey=from_sa(s,a,grid['esoft'])
        targets.append(dict(target_id=f'T{i:02d}',grid_row=r,grid_col=c,s_target=s,a_target=a,Ex_target=float(ex),Ey_target=float(ey)))
    return targets

def reference(grid,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,8));x=grid['s_edges'];y=grid['a_edges']
    for a in x:ax.axvline(a,color='.7',lw=.8)
    for a in y:ax.axhline(a,color='.7',lw=.8)
    for r in range(1,len(y)):
        for c in range(1,len(x)):
            ax.text(np.mean(x[c-1:c+1]),np.mean(y[r-1:r+1]),f'R{r} C{c}',ha='center',va='center')
    ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),xlabel='s = 0.5 [ln(Ex/Esoft) + ln(Ey/Esoft)]',ylabel='a = ln(Ex/Ey)',title='Original 6 x 6 grid: selection of 24 red-circle cells REQUIRED')
    ax.set_xticks(x);ax.set_yticks(y);fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)

def frozen_reference(grid,targets,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(8,7));x=grid['s_edges'];y=grid['a_edges']
    for v in x:ax.axvline(v,color='.7',lw=.7)
    for v in y:ax.axhline(v,color='.7',lw=.7)
    for t in targets:
        r,c=t['grid_row']-1,t['grid_col']-1
        ax.add_patch(Rectangle((x[c],y[r]),x[c+1]-x[c],y[r+1]-y[r],color='#dcebf8',zorder=0))
        ax.scatter(t['s_target'],t['a_target'],marker='*',s=80,c='#bb231e',zorder=3)
        ax.annotate(f'{t["target_id"]} (R{r+1} C{c+1})',(t['s_target'],t['a_target']),xytext=(0,12),textcoords='offset points',ha='center',fontsize=7)
    ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),xlabel='s',ylabel='a',title='24 user-selected cells: exact frozen bin midpoints')
    ax.set_xticks(x);ax.set_yticks(y);fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cells',help='JSON list of [row,column], in desired target order');args=ap.parse_args()
    p=HERE/'configs/grid.json';grid=read(p);reference(grid,HERE/'configs/grid_reference.png')
    if args.cells:
        out=HERE/'configs/targets.json'
        if out.exists() and read(out)['status']=='frozen':raise ValueError('Targets already frozen; do not silently redefine the benchmark')
        cells=read(args.cells);targets=make_targets(cells,grid)
        save(out,dict(status='frozen',grid_sha256=sha(p),selection_source_sha256=sha(args.cells),selected_cells=cells,targets=targets))
        csvwrite(out.with_suffix('.csv'),targets)
    frozen=read(HERE/'configs/targets.json')
    if frozen['status']=='frozen':frozen_reference(grid,frozen['targets'],HERE/'configs/selected_targets.png')
