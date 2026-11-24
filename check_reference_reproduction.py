"""Regenerate complete original batches covering every retained new style.

Run after full selection with the GPU generation process finished. Read-only
candidate/release access; writes only one detailed and one compact scratch result.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import candidate_generation as base
import reference_style_candidates as engine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    manifest=json.loads((args.work_dir/'full_selection.json').read_text(encoding='utf-8'))
    selected={r['key']:r for r in manifest['selected']}
    batches=defaultdict(set)
    for path in sorted((args.work_dir/'source_review_pages').glob('page_*.json')):
        page=json.loads(path.read_text(encoding='utf-8'))
        for item in page['entries']:
            if item['source_key'] in selected:
                batches[item['cache_file']].add(selected[item['source_key']]['style_family'])
    missing=set(engine.STYLE_FAMILIES);chosen=[]
    while missing:
        best=min(batches,key=lambda b:(-len(batches[b]&missing),b))
        if not batches[best]&missing:raise ValueError('No selected batch covers remaining styles')
        chosen.append(best);missing-=batches[best]
    originals=[]
    for name in chosen:
        path=Path(name);old=base.read_batch(path)
        if len(old)!=4:raise ValueError('Original batch size must be four')
        start=int(path.name.split('_')[1])
        expected=engine.specs_for('full',start+4,engine.MASTER_SEED)[start:start+4]
        if [s.spec for s in old]!=expected:raise ValueError('Original specifications differ')
        for s in old:
            key=f'reference:{s.spec.seed}'
            if key in selected and base.fingerprint(s.pixels)!=selected[key]['hash']:
                raise ValueError('Selection hash differs from source cache')
        originals.append((path,old))
    summary=dict(batches=len(originals),checked_sources=4*len(originals),
                 retained_reference_families=sorted(set().union(*(batches[p] for p in chosen))),
                 batch_size=4,master_seed=engine.MASTER_SEED)
    if args.dry_run:
        print(json.dumps(dict(summary,input_checks='pass')),flush=True);return
    models=engine.LocalModels();records=[]
    for path,old in originals:
        new=models.generate([s.spec for s in old])
        for a,b in zip(old,new):
            key=f'reference:{a.spec.seed}'
            records.append(dict(key=key,retained=key in selected,style_family=a.diagnostics['style_family'],
                batch=path.name,pixels_exact=bool(np.array_equal(a.pixels,b.pixels)),
                source_mask_exact=bool(np.array_equal(a.source_mask,b.source_mask)),
                original_sha256=base.fingerprint(a.pixels),regenerated_sha256=base.fingerprint(b.pixels)))
    exact=all(r['pixels_exact'] and r['source_mask_exact'] for r in records)
    summary.update(all_exact=exact,checked_retained_sources=sum(r['retained'] for r in records),
        limitation='Sampled original batches on this workstation; not cross-platform determinism.')
    detail=dict(summary,records=records,model_source=base.SOURCE,
        torch_version=str(models.torch.__version__),gpu=models.torch.cuda.get_device_name(0),
        cache_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p,_ in originals})
    (args.work_dir/'reproduction_details.json').write_text(json.dumps(detail,indent=2),encoding='utf-8')
    (args.work_dir/'reproduction.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    if not exact:raise RuntimeError('Reproduced pixels differ; inspect detailed report')


if __name__=='__main__':main()
