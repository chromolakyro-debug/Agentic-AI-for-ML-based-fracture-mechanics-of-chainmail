import numpy as np
from utils import distance,digest

def nearest(query,properties,source_keys,indices,distance_cfg,k):
    """One slot per source; compare original/swapped in the same fixed units."""
    props=np.asarray(properties)[indices]
    direct=distance(query,props,distance_cfg)
    swapped=distance(query,props[:,::-1],distance_cfg) if distance_cfg['allow_axis_swap'] else np.full(len(indices),np.inf)
    use_swap=swapped<direct # deterministic ties prefer original axes
    best=np.minimum(direct,swapped)
    order=np.lexsort((np.asarray(indices),best));seen=set();result=[]
    for pos in order:
        idx=int(indices[pos]);key=source_keys[idx]
        if key in seen:continue
        seen.add(key);matched=props[pos,::-1] if use_swap[pos] else props[pos]
        result.append(dict(rank=len(result)+1,dataset_index=idx,axis_swapped=bool(use_swap[pos]),distance=float(best[pos]),
            Ex_matched=float(matched[0]),Ey_matched=float(matched[1])))
        if len(result)==k:break
    if len(result)<k:raise ValueError('Not enough distinct training sources for retrieval')
    return result
def target_cache_key(dataset_identity,targets,distance_cfg,k,implementation):
    return digest(dict(training_identity=dataset_identity,targets=targets,distance=distance_cfg,top_k=k,implementation=implementation))
