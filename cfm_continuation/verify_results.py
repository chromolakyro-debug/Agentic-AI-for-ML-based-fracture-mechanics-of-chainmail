"""Independent numerical/data audit; no torch, FEM solve or model weights required."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()


def verify(run):
    run=Path(run);protocol=read(run/'protocol.json');data=np.load(run/'data.npz',allow_pickle=False)
    train=data['train'];assert len(train)==6400 and len(data['validation'])==len(data['test'])==800
    assert len(np.unique(np.r_[train,data['validation'],data['test']]))==8000
    assert np.allclose(data['sa'][train].mean(0),data['mean']) and np.allclose(data['sa'][train].std(0),data['std'])
    assert digest(run/'protocol.json')==read(run/'frozen_protocol_sha256.json')['sha256']
    original_data=read(run/'original_identity.json')
    assert protocol['data_sha256']==next(r['sha256'] for r in original_data['files'] if r['path'].replace('\\','/').endswith('/cat_binary_dataset/dataset.npy'))
    from scipy.spatial import cKDTree
    tree=cKDTree((data['sa'][train]-data['mean'])/data['std'])
    vr=tree.query((data['sa'][data['validation']]-data['mean'])/data['std'],k=20)[0][:,-1]
    assert abs(np.quantile(vr,.95)-protocol['support']['threshold'])<1e-12
    assert np.allclose(vr,data['validation_r20'])
    for name in ('development','final'):
        tt=protocol[name+'_targets'];points=np.array([[r['s'],r['a']] for r in tt]);r=tree.query((points-data['mean'])/data['std'],k=20)[0][:,-1]
        assert np.all(r<=protocol['support']['threshold']) and np.allclose(r,[v['support_r20'] for v in tt])
        exy=np.exp(np.c_[points[:,0]+points[:,1]/2,points[:,0]-points[:,1]/2])
        assert np.all(exy>=1) and np.all(exy<=10)
        assert np.all(exy.max(1)<=11-10/exy.min(1)+1e-12)
    folders=sorted(run.glob('development_*'))+[run/'final'];summaries=[]
    for folder in folders:
        if not (folder/'manifest.json').exists():continue
        manifest=read(folder/'manifest.json');binary=np.load(folder/'binary.npy',allow_pickle=False)
        panel='final' if folder.name=='final' else 'development'
        assert manifest['targets']==protocol[panel+'_targets']
        assert manifest['seeds']==protocol['final_noise_seeds' if panel=='final' else 'dev_noise_seeds']
        assert manifest['binary_threshold']==0 and manifest['ode_steps']==64
        rows=read(folder/'samples.json');assert len(rows)==len(manifest['samples'])==len(binary)
        assert binary.shape[1:]==(224,224) and binary.dtype==np.uint8 and np.isin(binary,[0,1]).all()
        valid=0;hit=0;ds=[];da=[]
        for source,row,im in zip(manifest['samples'],rows,binary):
            assert source['key']==row['key'] and source['sha256_pixels']==row['sha256_pixels']==hashlib.sha256(im.tobytes()).hexdigest()
            if row['fem_valid']:
                valid+=1;s=.5*(np.log(row['Ex_app']/protocol['esoft'])+np.log(row['Ey_app']/protocol['esoft']));a=np.log(row['Ex_app']/row['Ey_app'])
                assert abs(s-row['s_hat'])<1e-12 and abs(a-row['a_hat'])<1e-12
                delta_s=s-row['s'];delta_a=a-row['a'];ds.append(abs(delta_s));da.append(abs(delta_a))
                expected=abs(delta_s)<=np.log(1.05) and abs(delta_a)<=np.log(1.10)
                assert expected==row['hit_joint'];hit+=int(expected)
            else:assert row['hit_joint'] is False
        metrics=read(folder/'metrics.json')['overall']
        assert metrics['n']==len(rows) and metrics['valid']==valid and abs(metrics['hit_joint']-hit/len(rows))<1e-12
        if valid:
            for key,values in [('abs_delta_s',ds),('abs_delta_a',da)]:
                assert abs(metrics[key]['median']-np.median(values))<1e-12 and abs(metrics[key]['p90']-np.quantile(values,.9))<1e-12
        summaries.append(dict(folder=folder.name,n=len(rows),valid=valid,joint_hits=hit))
    # Independently recompute train-only nearest retrieval with fixed evaluation scales.
    retrieval=read(run/'retrieval.json')['results']
    for target,r in zip(protocol['final_targets'],retrieval):
        delta=data['sa'][train]-[target['s'],target['a']];norm=np.linalg.norm(delta/[np.log(1.05),np.log(1.10)],axis=1)
        assert int(train[np.argmin(norm)])==r['training_index'] and r['training_index'] in train
    selection=read(run/'checkpoint_selection.json');summary=read(run/'training_summary.json');config=read(run/'continuation_config.json')
    assert selection['step']==read(run/'final'/'manifest.json')['checkpoint_step']==summary['final_checkpoint_step']
    assert selection['sha256']==read(run/'final'/'manifest.json')['checkpoint_sha256']
    assert summary['steps']<=200000 and (summary['steps']>=80000 or summary['segment_exit_reason']=='safety_cap')
    assert summary['images_seen']==320000+(summary['steps']-20000)*config['batch']
    return dict(status='passed',folders=summaries,training_only_retrieval=True,split_and_support=True,strict_binary_pixel_hashes=True,
                recomputed_moduli_sa_errors_hits_quantiles=True,historical_final_panel=True,
                limits='Verifies stored FEM results and image identity; independently re-solving FEM requires full solver environment and is not done by this numerical checker.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path(__file__).resolve().parent/'run');a=p.parse_args()
    print(json.dumps(verify(a.run),indent=2))
