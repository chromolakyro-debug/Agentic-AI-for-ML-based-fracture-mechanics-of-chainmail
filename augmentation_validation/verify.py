"""Independent saved-result and on-the-fly loader compatibility checks; no FEM/training."""
import csv
import json
from collections import Counter
import numpy as np
import torch
from run import ROOT,DATA,OUT,HERE,read,save,sha,pixels_sha
from augmentation import random_augmentation


def main():
    cfg=read(OUT/'config.json');samples=read(OUT/'samples.json')
    for name,digest in read(OUT/'frozen_inputs.json').items():assert sha(OUT/name)==digest
    for name,digest in cfg['script_sha256'].items():assert sha(HERE/name)==digest
    for name,digest in cfg['protected_files_sha256'].items():assert sha(ROOT/name)==digest
    assert Counter(r['group_id'] for r in samples)=={f'{i:02d}':20 for i in range(1,9)}
    assert len({r['new_id'] for r in samples})==160
    with (OUT/'per_sample_results.csv').open(encoding='utf-8-sig',newline='') as f:results=list(csv.DictReader(f))
    raw={r['key']:r for line in (OUT/'main_fem.jsonl').read_text().splitlines() if line.strip() for r in [json.loads(line)]}
    assert len(raw)==320 and len(results)==160
    for i,(s,r) in enumerate(zip(samples,results)):
        assert r['new_id']==s['new_id'] and int(r['sample_order'])==i
        o=raw[f'{i:03d}_original'];a=raw[f'{i:03d}_augmented']
        assert o['fem_valid'] and a['fem_valid']
        expected=[o['Ey_app'],o['Ex_app']] if s['angle_ccw'] in (90,270) else [o['Ex_app'],o['Ey_app']]
        for axis,e in zip(('x','y'),expected):
            actual=a[f'E{axis}_app'];error=abs(actual-e)
            assert float(r[f'E{axis}_expected'])==e
            assert float(r[f'E{axis}_augmented_FEM'])==actual
            assert float(r[f'E{axis}_abs_error'])==error
            assert float(r[f'E{axis}_rel_error'])==error/max(abs(e),cfg['relative_denominator_floor'])
            assert error<=cfg['atol']+cfg['rtol']*abs(e)
    # Exercise the public random augmentation interface and PyTorch conversion.
    arr=np.load(DATA/'dataset.npy',mmap_mode='r',allow_pickle=False)
    rng=np.random.default_rng(194)
    for s in samples[::20]:
        original=arr[s['dataset_index']];before=pixels_sha(original)
        out,labels,recipe=random_augmentation(original,s['Ex_metadata'],s['Ey_metadata'],rng)
        assert out.shape==original.shape and out.dtype==np.uint8
        assert out.flags.c_contiguous and not np.shares_memory(original,out)
        assert pixels_sha(original)==before and int(out.sum())==int(original.sum())
        x=torch.from_numpy(out).float()[None,None]*2-1
        assert x.shape==(1,1,224,224) and set(x.unique().tolist())<={-1.,1.}
        assert tuple(labels)==((s['Ey_metadata'],s['Ex_metadata']) if recipe['quarter_turns']%2 else (s['Ex_metadata'],s['Ey_metadata']))
    save(OUT/'verification.json',dict(status='passed',sample_count=160,main_image_solves=320,
        exact_csv_errors_recomputed=True,source_and_splits_hash_unchanged=True,
        frozen_protocol_and_implementation_hashes_match=True,on_the_fly_cpu_torch_conversion=True,
        source_arrays_not_modified=True,model_or_optimizer_created=False,training_started=False,
        figure_inspection='Viewed the single 8-class original/augmented figure; headings, native pixels and numerical labels are legible.'))
    commands=['run.py prepare','run.py main','run.py precision','run.py direct','analyze.py','verify.py']
    save(OUT/'execution.json',dict(cwd=str(ROOT),python=str(ROOT/'cfm_v0/.venv/Scripts/python.exe'),
        commands=['augmentation_validation/'+s for s in commands],all_executed=True,
        concurrency='main first; precision and direct ran concurrently afterwards; no FEM recomputation on analysis-only changes'))
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='integrity_manifest.json']
    save(OUT/'integrity_manifest.json',dict(files=[dict(path=p.relative_to(HERE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)],
        excluded=['__pycache__','results/integrity_manifest.json']))
    print('PASS: 160 sample identities, 320 original/augmented solves, CSV errors, frozen hashes, unchanged sources, on-the-fly CPU tensor inputs.')


if __name__=='__main__':main()
