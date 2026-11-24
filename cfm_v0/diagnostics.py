"""Training-only retrieval and image-neighbor diagnostics; never used to select a checkpoint."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='2'
import argparse
import csv
import sys
import time
from types import SimpleNamespace
import numpy as np
from common import ROOT, DATA, OUT, S_TOL, A_TOL, read_json, save_json
sys.path.insert(0,str(ROOT))
import mechanics_selection as ms
import reference_style_selection as rs


def prepare_baseline():
    started=time.perf_counter()
    d=np.load(OUT/'data.npz');protocol=read_json(OUT/'protocol.json')
    rows=list(csv.DictReader((DATA/'metadata.csv').open(encoding='utf-8-sig',newline='')))
    train=d['train'];sa=d['sa'][train]
    records=[]
    for target in protocol['final_targets']:
        delta=(sa-[target['s'],target['a']])
        # Evaluation's axis tolerances define the metric, no validation/test retrieval candidates.
        metric=np.linalg.norm(delta/[S_TOL,A_TOL],axis=1)
        order=np.argsort(metric,kind='stable')
        j=order[0];idx=int(train[j])
        records.append(dict(target_id=target['target_id'],training_index=idx,filename=rows[idx]['filename'],
                    s=float(sa[j,0]),a=float(sa[j,1]),delta_s=float(delta[j,0]),delta_a=float(delta[j,1]),
                    error_scaled_l2=float(metric[j]),hit=bool(np.all(np.abs(delta[j])<=[S_TOL,A_TOL])),
                    top32_training_indices=train[order[:32]].tolist(),
                    training_count_in_hit_box=int(np.sum(np.all(np.abs(delta)<=[S_TOL,A_TOL],axis=1)))))
    save_json(OUT/'retrieval.json',dict(metric='sqrt((delta_s/log(1.05))^2+(delta_a/log(1.10))^2)',
              candidate_split='train only (6400)',results=records))
    path=OUT/'training_descriptors.npz'
    if not path.exists():
        images=np.load(DATA/'dataset.npy',mmap_mode='r')
        samples=[dict(pixels=images[i],spec=dict(background=int(rows[i]['background_value']))) for i in train]
        desc=rs._Descriptors(samples)
        np.savez_compressed(path,**vars(desc),indices=train)
    print('Baseline/descriptor precomputation',time.perf_counter()-started,'seconds',flush=True)


def image_neighbors():
    # Keep this CPU analysis process free of Torch's separate Windows OpenMP runtime.
    # NumPy FP32 integer dot products compute the same exact binary Hamming distances.
    from threadpoolctl import threadpool_limits
    blas_limit=threadpool_limits(8,user_api='blas')
    folder=OUT/'final';manifest=read_json(folder/'manifest.json')
    images=np.load(folder/'binary.npy');d=np.load(OUT/'data.npz');train=d['train']
    training_images=np.load(DATA/'dataset.npy',mmap_mode='r')[train]
    archive=np.load(OUT/'training_descriptors.npz')
    assert np.array_equal(archive['indices'],train)
    desc=SimpleNamespace(**{k:archive[k] for k in archive.files if k!='indices'})
    # Infer background ONLY for the visual descriptor; material images are unchanged.
    def polarity(image):
        edges=np.r_[image[0],image[-1],image[1:-1,0],image[1:-1,-1]]
        return int(edges.mean()>=.5)
    valid=np.array([im.min()!=im.max() for im in images])
    samples=[dict(pixels=im,spec=dict(background=polarity(im))) for im in images[valid]]
    generated=rs._Descriptors(samples)
    ntrain=len(train)
    combined=SimpleNamespace(**{k:np.concatenate([getattr(desc,k),getattr(generated,k)]) for k in vars(desc)})
    visual_index=rs._VisualIndex(combined.hashes)
    for i in range(ntrain):visual_index.add(i)
    # Full-resolution, aligned binary Hamming distance over all 6400 training images.
    # FP32 inputs 0/1 and integer sums are exact at these sizes (<2**24).
    x=training_images.reshape(ntrain,-1).astype(np.float32)
    sums=x.sum(1)
    pixel_indices=[];pixel_dist=[]
    for start in range(0,len(images),16):
        q=images[start:start+16].reshape(-1,224*224).astype(np.float32)
        distance=(q.sum(1)[:,None]+sums[None]-2*(q@x.T))/(224*224)
        indices=distance.argmin(1);values=distance[np.arange(len(q)),indices]
        pixel_indices.extend(indices.tolist());pixel_dist.extend(values.tolist())
        if (start+16)%128==0:print('Exact full-resolution Hamming',min(start+16,len(images)),'/',len(images),flush=True)
    del x
    results=[];j=0
    for i,row in enumerate(manifest['samples']):
        result=dict(key=row['key'],target_id=row['target_id'],seed=row['seed'],
                    pixel_nearest_training_index=int(train[pixel_indices[i]]),
                    pixel_hamming_fraction=float(pixel_dist[i]),exact_pixel_duplicate=bool(pixel_dist[i]==0),
                    descriptor_valid=bool(valid[i]))
        # Validate selected nearest Hamming value independently on CPU.
        assert abs(np.mean(images[i]!=training_images[pixel_indices[i]])-pixel_dist[i])<1e-7
        if valid[i]:
            index=ntrain+j
            best=np.full(ntrain,np.inf)
            for lref,rref in rs.PAIR_COMPARISONS:
                dist=np.sqrt(np.mean((desc.features[:,rref]-generated.features[j,lref])**2,axis=1))
                best=np.minimum(best,dist)
            nearest=int(np.argmin(best))
            candidates=visual_index.query(index)
            matches,_=rs._visual_matches(index,candidates,combined)
            result.update(shape_nearest_training_index=int(train[nearest]),shape_pooled_rms=float(best[nearest]),
                          strict_visual_matches_training_indices=train[candidates[matches]].tolist(),
                          inferred_background=polarity(images[i]))
            j+=1
        results.append(result)
    diversity=[]
    valid_map=np.full(len(images),-1,dtype=int)
    valid_map[np.flatnonzero(valid)]=np.arange(np.sum(valid))
    for target in manifest['targets']:
        indices=[i for i,r in enumerate(manifest['samples']) if r['target_id']==target['target_id']]
        group=images[indices].reshape(len(indices),-1).astype(np.float32)
        mismatch=(group.sum(1)[:,None]+group.sum(1)[None]-2*group@group.T)/(224*224)
        upper=mismatch[np.triu_indices(len(indices),1)]
        local=valid_map[indices];local=local[local>=0]
        shape_distance=np.full((len(local),len(local)),np.inf)
        for lref,rref in rs.PAIR_COMPARISONS:
            left=generated.features[local,lref];right=generated.features[local,rref]
            distance=np.sqrt(np.maximum(0,(left**2).mean(1)[:,None]+(right**2).mean(1)[None]-2*(left@right.T)/left.shape[1]))
            shape_distance=np.minimum(shape_distance,distance)
        shape_upper=shape_distance[np.triu_indices(len(local),1)]
        diversity.append(dict(target_id=target['target_id'],pairwise_hamming_median=float(np.median(upper)),
                    pairwise_hamming_p10=float(np.quantile(upper,.1)),pairwise_hamming_p90=float(np.quantile(upper,.9)),
                    pairwise_normalized_shape_rms_median=float(np.median(shape_upper)) if len(shape_upper) else None,
                    descriptor_valid_count=len(local),
                    unique_binary_hashes=len(set(manifest['samples'][i]['sha256_pixels'] for i in indices))))
    save_json(folder/'image_neighbors.json',dict(results=results,diversity=diversity,
            method='All-train full 224px Hamming; existing reflection-aware bbox-normalized shape/ink descriptors and strict visual predicate',
            caveat='Similarity/uniqueness diagnostics do not establish meaningful structural novelty; descriptor normalization is for comparison only'))
    print('Image-neighbor diagnostics completed',len(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','neighbors']);a=p.parse_args()
    if a.mode=='prepare':prepare_baseline()
    else:image_neighbors()
