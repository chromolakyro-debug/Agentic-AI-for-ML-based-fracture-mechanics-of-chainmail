"""Record checkpoint identity and state schema; weights stay outside AI ZIP."""
import paths
import gc
import json
import torch
from common import OUT,read_json,save_json,sha_file


def main():
    records=[]
    history=[json.loads(line) for line in (OUT/'training.jsonl').read_text().splitlines() if line.strip()]
    validations={r['step']:r['validation_velocity_mse'] for r in history if 'validation_velocity_mse' in r}
    for f in sorted((OUT/'checkpoints').glob('*.pt')):
        c=torch.load(f,map_location='cpu');r=dict(path=str(f.resolve()),filename=f.name,bytes=f.stat().st_size,sha256=sha_file(f),step=c['step'],keys=list(c),
            model_parameters=11529633,contains_optimizer='optimizer' in c,contains_rng=all(k in c for k in ('rng_cpu','rng_cuda','rng_numpy','rng_python')),
            contains_ema='ema' in c,validation_loss=c.get('validation_loss',validations.get(c['step'])))
        if 'optimizer' in c:
            r['tracked_best_validation_loss']=c['best_loss']
            r['tracked_best_step']=c['best_step']
            r['optimizer_steps']=sorted(set(float(v['step']) for v in c['optimizer']['state'].values()))
            r['optimizer_lr']=c['optimizer']['param_groups'][0]['lr']
        records.append(r);del c;gc.collect()
    originals=read_json(OUT/'original_identity.json')['files']
    for r in originals:assert sha_file(r['path'])==r['sha256'],f'Original modified: {r["path"]}'
    save_json(OUT/'checkpoint_inventory.json',dict(checkpoints=records,original_files_hash_verified_unchanged=len(originals),
        excluded_from_ai_zip=True,restore='Resume from continuation last.pt; model, optimizer, EMA, RNG, sample count, elapsed budget, scheduler/restart metadata and immutable config included.'))
    print({'checkpoints':len(records),'originals_unchanged':len(originals)})

if __name__=='__main__':main()
