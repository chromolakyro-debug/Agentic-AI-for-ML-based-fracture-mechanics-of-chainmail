"""Record already-performed AI visual review of exact selected pixels.

This tool does no semantic evaluation. Cell numbers must come from actual review.
Anatomical exclusions propagate to every framing variant of the same source.
"""
from pathlib import Path
import argparse
import json
import hashlib
import candidate_generation as gen

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--stage',choices=['pilot','full'],required=True)
    p.add_argument('--start',type=int,default=1)
    p.add_argument('--end',type=int,required=True)
    p.add_argument('--exclude',action='append',default=[],help='one-based-cell:reason')
    a=p.parse_args()
    selected=json.loads((a.work_dir/f'{a.stage}_selection.json').read_text(encoding='utf-8'))
    exclusions={int(s.split(':',1)[0]):s.split(':',1)[1] for s in a.exclude}
    if not 1<=a.start<=a.end<=selected['selected_count']:raise ValueError('Invalid reviewed range')
    if any(i<a.start or i>a.end for i in exclusions):raise ValueError('Exclusion outside reviewed range')
    path=a.work_dir/'augmentation_visual_review.json'
    result=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    passed=set(result.get('passed_hashes',[]));excluded=result.get('excluded_hashes',{})
    excluded_seeds={}
    for i in range(a.start,a.end+1):
        h=selected['selected_hashes'][i-1]
        if i in exclusions:excluded_seeds[selected['selected_seeds'][i-1]]=exclusions[i]
        else:passed.add(h)
    pools=gen.read_batch(a.work_dir/f'{a.stage}_variants.npz')
    if a.stage=='full' and (a.work_dir/'styles_variants.npz').exists():
        pools+=gen.read_batch(a.work_dir/'styles_variants.npz')
    for sample in pools:
        if sample.spec.seed in excluded_seeds:
            h=hashlib.sha256(sample.pixels.tobytes()).hexdigest()
            excluded[h]=excluded_seeds[sample.spec.seed]
            passed.discard(h)
    result.update(passed_hashes=sorted(passed),excluded_hashes=excluded,
                  method='AI visual inspection of selected actual binary pixel maps; no human inspection claimed')
    result.setdefault('review_batches',[]).append(dict(stage=a.stage,start=a.start,end=a.end,
          selection_hashes=selected['selected_hashes'][a.start-1:a.end],excluded_cells=exclusions))
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed_hashes=len(passed),excluded_hashes=len(excluded))))

if __name__=='__main__':main()
