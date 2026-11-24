"""Single training figure, concise measured report and a compact external-AI audit ZIP."""
import argparse,csv,json,zipfile,datetime
from pathlib import Path
import numpy as np
from common import HERE,ROOT,RUN,read,save,sha

def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def curves():
    rows=[json.loads(x) for x in (RUN/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    train=[r for r in rows if r['kind']=='train'];val=[r for r in rows if r['kind']=='validation']
    # On an abnormal resume, the trailing pre-checkpoint log can contain replayed updates.
    # Preserve all raw records; plotted/CSV view takes the last record for each step.
    train=list({r['step']:r for r in train}.values());train.sort(key=lambda r:r['step'])
    val=list({r['step']:r for r in val}.values());val.sort(key=lambda r:r['step'])
    write_csv(RUN/'training_history.csv',train);write_csv(RUN/'validation_history.csv',val)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True)
    x=np.array([r['step'] for r in train]);y=np.array([r['loss'] for r in train])
    ax[0].plot([r['step'] for r in val],[r['validation_velocity_mse'] for r in val],'.-',label='fixed validation EMA velocity MSE');ax[0].set_ylabel('Validation MSE')
    ax[1].plot(x,y,color='#6699c0',lw=.25,alpha=.25,label='raw per-update train MSE')
    if len(y)>=100:ax[1].plot(x[99:],np.convolve(y,np.ones(100)/100,mode='valid'),lw=.7,label='100-update mean (display only)')
    ax[1].set_ylabel('Train MSE');ax[1].set_yscale('log')
    ax[2].plot(x,[r['lr'] for r in train],color='#b15b25');ax[2].set_ylabel('Learning rate');ax[2].set_xlabel('Optimizer updates')
    for a in ax:
        a.grid(alpha=.2)
        for step in range(20000,int(x.max())+1,20000):a.axvline(step,ls=':',lw=.6,color='.4')
    ax[0].legend();ax[1].legend();fig.suptitle('CFM v1 continuation — velocity MSE and cosine warm restarts (not mechanical error)');fig.tight_layout();fig.savefig(HERE/'training_curves.png',dpi=170);plt.close(fig)
    return train,val

if __name__=='__main__':curves()
