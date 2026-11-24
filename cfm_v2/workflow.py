"""Bounded unattended train -> cycle decisions -> existing real-FEM evaluation -> audit."""
import sys,subprocess,json,shutil,time,datetime,traceback
import numpy as np
from common import HERE,ROOT,RUN,DATA,read,save,sha

def command(args,log):
    record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=args,cwd=str(ROOT),log=str(log))
    with (RUN/'commands.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    with log.open('a',encoding='utf-8') as out:r=subprocess.run(args,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'Command failed ({r.returncode}); see {log}')

def evaluate(reason):
    import torch
    source=RUN/'checkpoints/best_val_mse.pt';state=torch.load(source,map_location='cpu',weights_only=False);step=state['step'];value=state['validation_loss'];del state
    folder=RUN/'evaluation_checkpoints';folder.mkdir(exist_ok=True);snapshot=folder/f'best_val_{step:06d}.pt'
    if not snapshot.exists():shutil.copyfile(source,snapshot)
    assert sha(snapshot)==sha(source)
    for rel,digest in read(RUN/'reference_dependencies.json').items():assert sha(ROOT/rel)==digest, f'Reference dependency changed: {rel}'
    config=read(RUN/'evaluation_template.json');dataset=config['dataset']
    dataset.update(labels='cfm_v2/run/data.npz',splits='cfm_v2/run/splits.json');config['dataset']=dataset;config['run_id']=f'cfm_v2_step_{step:06d}'
    d=np.load(RUN/'data.npz');model=config['model'];model.update(checkpoint=snapshot.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(snapshot),normalization=dict(mean=d['mean'].tolist(),std=d['std'].tolist()))
    model['dataset_binding']=dict(metadata=sha(DATA/'metadata.csv'),array=sha(DATA/'dataset.npy'),labels=sha(RUN/'data.npz'),splits=sha(RUN/'splits.json'))
    path=ROOT/'evaluation/configs'/f'{config["run_id"]}.json'
    if path.exists():assert read(path)==config
    else:save(path,config)
    started=time.perf_counter();command([sys.executable,'evaluation/run.py','--config',str(path)],RUN/f'evaluation_{step:06d}.log')
    row=dict(reason=reason,checkpoint_step=step,checkpoint_sha256=sha(snapshot),validation_mse=value,config=str(path.relative_to(ROOT)),
        output=f'evaluation/runs/{config["run_id"]}',seconds=time.perf_counter()-started,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    history=read(RUN/'evaluations.json') if (RUN/'evaluations.json').exists() else [];history.append(row);save(RUN/'evaluations.json',history)
    if reason=='final_best_val_mse':save(RUN/'final_evaluation.json',row)
    return row

def main():
    cfg=read(RUN/'config.json');assert read(RUN/'checks.json')['status']=='passed'
    snapshot=RUN/'code_snapshot';snapshot.mkdir(exist_ok=True)
    sources=list(HERE.glob('*.py'))+[ROOT/'cfm_v0/model.py',ROOT/'mechanics.py']
    for p in sources:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(p),'Training source changed during run; record and review before resume'
        else:shutil.copyfile(p,target)
    save(RUN/'code_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    while not (RUN/'training_summary.json').exists():
        previous=read(RUN/'segment_result.json')['step'] if (RUN/'segment_result.json').exists() else 0
        # After an interruption, recover actual progress from last, never from a stale segment summary.
        if (RUN/'checkpoints/last.pt').exists():
            import torch
            state=torch.load(RUN/'checkpoints/last.pt',map_location='cpu',weights_only=False);previous=state['step'];del state
        cycle=cfg['scheduler']['cycle_steps'];until=min((previous//cycle+1)*cycle,cfg['limits']['max_updates'])
        args=[sys.executable,'cfm_v2/train.py','--until',str(until)]
        if previous:args.append('--resume')
        command(args,RUN/'training_console.log');segment=read(RUN/'segment_result.json')
        print(json.dumps(segment),flush=True)
        command([sys.executable,'cfm_v2/finalize.py','--curves-only'],RUN/'curves_console.log')
        if segment['eval_active'] and not (RUN/'training_summary.json').exists():evaluate('periodic_after_stage_A')
    evaluate('final_best_val_mse');command([sys.executable,'cfm_v2/finalize.py'],RUN/'finalization_console.log')
    save(RUN/'workflow_complete.json',dict(status='complete',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary=read(RUN/'training_summary.json'),final_evaluation=read(RUN/'final_evaluation.json')))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(RUN/'workflow_error.json',dict(reason='abnormal_interruption',error=repr(e),traceback=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        raise
