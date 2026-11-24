"""Resume -> full-cycle evaluation -> final best-validation report and audit."""
import sys,shutil,subprocess,time,json,datetime,traceback
import torch
from common import ROOT,HERE,RUN,read,save,sha

def command(args,log):
    with (RUN/'commands.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(argv=args,cwd=str(ROOT),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
    with log.open('a',encoding='utf-8') as out:
        result=subprocess.run(args,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Command failed: {log}')

def evaluate(reason):
    source=RUN/'checkpoints/best_val_mse.pt';state=torch.load(source,map_location='cpu',weights_only=False)
    step=state['step'];value=state['validation_loss'];del state
    folder=RUN/'evaluation_checkpoints';folder.mkdir(exist_ok=True);snapshot=folder/f'best_val_{step:06d}.pt'
    if not snapshot.exists():shutil.copyfile(source,snapshot)
    assert sha(source)==sha(snapshot)
    cfg=read(ROOT/'evaluation/configs/cfm_v1_step_280000.json')
    cfg['dataset'].update(labels='cfm_v1_cont1/run/data.npz',splits='cfm_v1_cont1/run/splits.json')
    cfg['model'].update(checkpoint=snapshot.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(snapshot))
    cfg['run_id']=f'cfm_v1_cont1_step_{step:06d}'
    cfg['output_directory']=f'evaluation/runs/cfm_v1_cont1_step_{step:06d}'
    cfg['cache']='cfm_v1_cont1/cache'
    path=HERE/'configs'/f'evaluation_{step:06d}.json'
    if path.exists():assert read(path)==cfg
    else:save(path,cfg)
    start=time.perf_counter();command([sys.executable,'evaluation/run.py','--config',str(path)],RUN/f'evaluation_{step:06d}.log')
    row=dict(reason=reason,checkpoint_step=step,validation_mse=value,checkpoint_sha256=sha(snapshot),
        output=cfg['output_directory'],config=path.relative_to(ROOT).as_posix(),seconds=time.perf_counter()-start)
    history=read(RUN/'evaluations.json') if (RUN/'evaluations.json').exists() else []
    history.append(row);save(RUN/'evaluations.json',history)
    if reason=='final_best_val_mse':save(RUN/'final_evaluation.json',row)

def main():
    assert read(RUN/'checks.json')['status']=='passed'
    cfg=read(RUN/'config.json');snapshot=RUN/'code_snapshot';snapshot.mkdir(exist_ok=True)
    files=list(HERE.glob('*.py'))+[ROOT/'cfm_v0/model.py',ROOT/'mechanics.py',ROOT/'evaluation/pipeline.py']
    for p in files:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=sha(p):
                migration=read(RUN/'evaluation_relocation.json')
                allowed=migration['post_training_code_changes'][p.relative_to(ROOT).as_posix()]
                assert (RUN/'training_summary.json').exists() and allowed==dict(before=sha(target),after=sha(p)), 'Unrecorded code change'
        else:shutil.copyfile(p,target)
    save(RUN/'code_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    while not (RUN/'training_summary.json').exists():
        state=torch.load(RUN/'checkpoints/last.pt',map_location='cpu',weights_only=False);step=state['step'];del state
        until=min((step//20000+1)*20000,cfg['limits']['max_updates'])
        command([sys.executable,'cfm_v1_cont1/train.py','--until',str(until),'--resume'],RUN/'training_console.log')
        command([sys.executable,'cfm_v1_cont1/curves.py'],RUN/'curves_console.log')
        if not (RUN/'training_summary.json').exists():evaluate('periodic_20k')
    evaluate('final_best_val_mse')
    command([sys.executable,'cfm_v1_cont1/finalize.py'],RUN/'finalization_console.log')
    save(RUN/'workflow_complete.json',dict(status='complete',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(RUN/'workflow_error.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
