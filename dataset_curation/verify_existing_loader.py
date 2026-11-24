"""Call the actual existing CFM reader, without models, optimizers, FEM or training."""
import sys
from pathlib import Path
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'cfm_v0'))
from train import setup
from common import save_json


def main():
    root=ROOT/'cat_binary_dataset_curated_v1'
    data,images=setup(data_dir=root,run_dir=root)
    assert images.shape==(7857,224,224) and images.dtype==np.uint8
    assert set(data['class_labels'])==set(range(8))
    counts={str(k):int(np.sum(data['class_labels']==k)) for k in range(8)}
    assert list(counts.values())==[750,1500,750,750,759,766,1500,1082]
    split_counts={key:len(data[key]) for key in ('train','validation','test')}
    assert split_counts==dict(train=6285,validation=786,test=786)
    selected=[]
    for label in range(8):
        members=np.flatnonzero(data['class_labels']==label)
        selected.extend([int(members[0]),int(members[-1])])
    import torch
    batch=torch.from_numpy(images[selected].copy()).float()[:,None]*2-1
    conditions=(data['sa'][selected]-data['mean'])/data['std']
    assert tuple(batch.shape)==(16,1,224,224) and set(batch.unique().tolist())=={-1.,1.}
    assert np.isfinite(conditions).all() and all((root/p).is_file() for p in data['relative_paths'])
    hashes=[hashlib.sha256(images[i].tobytes()).hexdigest() for i in selected]
    records=[dict(index=i,new_id=str(data['sample_ids'][i]),class_label=int(data['class_labels'][i]),
                  path=str(data['relative_paths'][i]),sha256_pixels=h) for i,h in zip(selected,hashes)]
    data.close();del images,batch
    # Default paths still read the untouched original dataset.
    old_data,old_images=setup()
    assert old_images.shape==(8000,224,224)
    assert [len(old_data[k]) for k in ('train','validation','test')]==[6400,800,800]
    old_data.close();del old_images
    result=dict(status='passed',actual_reader='cfm_v0.train.setup(data_dir=curated_root, run_dir=curated_root)',
        all_array_rows_loaded=7857,all_image_paths_exist=True,class_counts=counts,split_counts=split_counts,
        cpu_batch_shape=[16,1,224,224],condition_shape=list(conditions.shape),class_examples=records,
        unchanged_default_original_reader=True,no_model_or_optimizer_created=True,no_training_or_FEM=True)
    save_json(root/'loader_validation.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
