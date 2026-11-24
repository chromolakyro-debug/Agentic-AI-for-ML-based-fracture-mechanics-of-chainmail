"""Verify curated IDs/phases/labels and freeze train-only preprocessing, no new split."""
import csv,collections,shutil
import numpy as np
from PIL import Image
from common import HERE,ROOT,RUN,DATA,read,save,sha,sa

def main():
    if (RUN/'config.json').exists():raise RuntimeError('Configuration already frozen')
    RUN.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((DATA/'metadata.csv').open(encoding='utf-8-sig',newline='')))
    original=list(csv.DictReader((ROOT/'cat_binary_dataset/metadata.csv').open(encoding='utf-8-sig',newline='')))
    origin={r['filename']:r for r in original}
    images=np.load(DATA/'dataset.npy',mmap_mode='r');labels=np.load(DATA/'data.npz');splits=read(DATA/'splits.json')
    assert images.shape==(7857,224,224) and images.dtype==np.uint8 and len(rows)==7857
    assert len(set(r['new_id'] for r in rows))==7857
    seen=set();counts={}
    for split in ['train','validation','test']:
        ids=np.array(splits[split]);assert not seen&set(ids);seen.update(ids);assert np.array_equal(ids,labels[split]);counts[split]=len(ids)
    assert seen==set(range(7857)) and counts==dict(train=6285,validation=786,test=786)
    mapping={};class_counts=collections.Counter();png_hashes=[]
    for i,r in enumerate(rows):
        assert r['new_id']==labels['sample_ids'][i];assert int(r['curation.dataset_index'])==i
        assert r['original_id']==labels['original_ids'][i]
        path=DATA/'images'/r['filename'];pixels=np.asarray(Image.open(path).convert('L'))
        assert np.isin(pixels,[0,255]).all() and np.array_equal(pixels//255,images[i])
        import hashlib
        assert hashlib.sha256(images[i].tobytes()).hexdigest()==r['sha256_pixels']
        old=origin[r['original_filename']];assert r['requested_style']==old['requested_style']
        assert float(r['Ex_app'])==float(old['Ex_app'])==labels['moduli'][i,0]
        assert float(r['Ey_app'])==float(old['Ey_app'])==labels['moduli'][i,1]
        style=r['requested_style'];group=r['group_id'];mapping.setdefault(style,group);assert mapping[style]==group
        class_counts[group]+=1;png_hashes.append(sha(path))
    expected=read(DATA/'validation.json')['class_counts'];assert dict(class_counts)==expected and len(mapping)==51
    retained={r['original_filename'] for r in rows};deleted=[r for r in original if r['filename'] not in retained]
    assert len(deleted)==143 and len({r['requested_style'] for r in deleted})==13
    assert not set(mapping)&{r['requested_style'] for r in deleted}
    tr=np.array(splits['train']);physical=labels['moduli'];values=sa(physical);assert np.allclose(values,labels['sa'],atol=1e-14)
    mean=values[tr].mean(0);std=values[tr].std(0)
    edges=[np.linspace(values[tr,j].min()-1e-9,values[tr,j].max()+1e-9,n+1) for j,n in enumerate((8,12))]
    bins=np.stack([np.clip(np.digitize(values[tr,j],edges[j])-1,0,len(edges[j])-2) for j in range(2)],axis=1)
    codes=bins[:,0]*12+bins[:,1];cell_counts=np.bincount(codes,minlength=96);raw=1/np.sqrt(cell_counts[codes]);weights=np.minimum(.5+.5*raw/raw.mean(),4.);weights/=weights.mean()
    np.savez(RUN/'data.npz',moduli=physical,sa=values,mean=mean,std=std,train=tr,validation=np.array(splits['validation']),test=np.array(splits['test']),
        sample_ids=labels['sample_ids'],original_ids=labels['original_ids'],train_weights=weights,train_weight_sample_ids=labels['sample_ids'][tr],cell_counts=cell_counts,bin_edges_s=edges[0],bin_edges_a=edges[1])
    shutil.copyfile(DATA/'splits.json',RUN/'splits.json')
    config=dict(version='cfm_v1',seed=20261117,dataset='cat_binary_dataset_curated_v1',initialization='random',batch=32,
        architecture=dict(channels=[32,64,128,192,256],emb=256),model_parameters=11529633,
        optimizer=dict(name='AdamW',weight_decay=.01,foreach=True),gradient_clip_norm=1.,ema_decay=.999,precision='bfloat16 autocast, float32 loss/optimizer',
        augmentation='uniform CCW rot90 in {0,1,2,3}; independent horizontal flip p=.5; rotate THEN flip; training only',
        normalization=dict(mean=mean.tolist(),std=std.tolist(),fit='unaugmented new training set only; same z-score definition; user approved refit'),
        sampling=dict(bins=[8,12],rule='min(0.5+0.5*(1/sqrt(count))/mean(1/sqrt(count)),4), then normalize mean to1',
            order='draw original train row with replacement, then augment; no class equalization, no augmented-density fitting',min_weight=float(weights.min()),max_weight=float(weights.max()),effective_sample_size=float(weights.sum()**2/(weights**2).sum())),
        scheduler=dict(type='update-based cosine warm restart',max_lr=2e-4,min_lr=3e-5,cycle_steps=20000,cycle_multiplier=1,peak_decay=.5,peak_floor=1e-4,initial_warmup_updates=200,reset_optimizer=False),
        plateau=dict(initial_training_updates=80000,evaluation_relative_threshold=.01,evaluation_patience=2,stop_relative_threshold=.005,stop_patience=3,epsilon=1e-12,
            rule='compare best EMA validation in each completed 20k cycle to best before that cycle; strict plateau must persist through one additional full cycle after arming; >=0.5% improvement clears stop patience'),
        validation=dict(interval=5000,batch=16,augmentation=False,noise_seed=20261114+567,repeats=1,selection='EMA velocity MSE, all786 validation rows in frozen order',
            definition='mean over samples and all224x224 pixels of (v_theta(x_t,t,c)-(x-epsilon))^2; x=2*binary-1; no other loss components; not FEM error'),
        evaluation=dict(interval_cycles_after_activation=1,targets='evaluation/configs/targets.json',seeds='evaluation/configs/seeds.json',selection='best validation EMA so far; checkpoint/config hash reuse',full_final=True),
        limits=dict(max_updates=200000,max_training_seconds=43200,training_time_definition='active train/validation/checkpoint wall time; excludes evaluation and setup'),
        checkpoint_interval=1000,log_interval=100,cudnn_benchmark=True,allow_tf32=True,
        input_hashes=dict(metadata=sha(DATA/'metadata.csv'),array=sha(DATA/'dataset.npy'),data=sha(RUN/'data.npz'),splits=sha(RUN/'splits.json'),model_code=sha(ROOT/'cfm_v0/model.py')))
    save(RUN/'config.json',config)
    save(RUN/'data_audit.json',dict(status='passed',samples=7857,splits=counts,class_counts=dict(class_counts),mapping=mapping,deleted_count=143,png_sha256=png_hashes,
        source_normalization=read(DATA/'normalization.json'),new_train_mean=mean.tolist(),new_train_std=std.tolist(),sampling_class_probability={g:float(weights[[rows[i]['group_id']==g for i in tr]].sum()/weights.sum()) for g in sorted(class_counts)}))
    save(RUN/'frozen_config.json',dict(sha256=sha(RUN/'config.json')))
    print('Verified all 7857 PNG/array/metadata/ID/label links; frozen config and train-only weights.')
if __name__=='__main__':main()
