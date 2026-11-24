"""Audited full-state continuation from v1; source files remain untouched."""
import copy, shutil, datetime
import numpy as np
import torch
from common import ROOT,HERE,RUN,read,save,sha
from control import Plateau

def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    else:assert a==b

def main():
    assert not (RUN/'config.json').exists(),'Do not bootstrap an existing continuation'
    source=ROOT/'cfm_v1/run';RUN.mkdir(parents=True,exist_ok=True);(RUN/'checkpoints').mkdir()
    checkpoint=source/'checkpoints/last.pt';source_hash=sha(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);original=copy.deepcopy(state)
    assert state['step']==280000 and state['scheduler']['updates']==280000
    cfg=copy.deepcopy(state['config']);cfg['version']='cfm_v1_cont1';cfg['initialization']='resume full v1 step280000 state'
    cfg['plateau'].update(initial_training_updates=300000,stop_relative_threshold=.001,stop_patience=4,
        rule='From 280k: compare complete20k cycle best to all prior best; four cycles <0.1%, then one additional full restart cycle; >=0.1% resets strict patience')
    cfg['limits'].update(max_updates=480000,max_training_seconds=state['training_seconds']+12*3600)
    cfg['continuation']=dict(source_checkpoint=str(checkpoint.relative_to(ROOT)),sha256=source_hash,start_step=state['step'],
        prior_training_seconds=state['training_seconds'],max_additional_steps=200000,max_additional_training_seconds=43200,
        new_stop_counter_starts_at=280000,old_decisions_preserved=True)
    for key in ['architecture','batch','optimizer','scheduler','normalization','sampling','validation','augmentation','input_hashes']:
        assert cfg[key]==original['config'][key]
    state['config']=cfg
    state['plateau'].update(cfg=cfg['plateau'],a_count=0,b_count=0,eval_active=True,stop_armed_cycle=None)
    for name in ['data.npz','splits.json','data_audit.json','training.jsonl']:
        shutil.copyfile(source/name,RUN/name)
    save(RUN/'parent_config.json',original['config']);save(RUN/'parent_training_summary.json',read(source/'training_summary.json'))
    save(RUN/'parent_plateau_history.json',original['plateau']['history']);save(RUN/'plateau_history.json',state['plateau']['history'])
    shutil.copyfile(source/'checkpoints/best_val_mse.pt',RUN/'checkpoints/best_val_mse.pt')
    save(RUN/'config.json',cfg);save(RUN/'frozen_config.json',dict(sha256=sha(RUN/'config.json')))
    torch.save(state,RUN/'checkpoints/last.pt')
    restored=torch.load(RUN/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    for key in original:
        if key not in ['config','plateau']:equal(original[key],restored[key])
    equal(original['plateau']['history'],restored['plateau']['history'])
    controller=Plateau(cfg['plateau'],20000);controller.load_state_dict(copy.deepcopy(state['plateau']))
    best=state['best_val_mse']
    decisions=[]
    for step in range(300000,380001,20000):
        best*=.9995;decisions.append(controller.completed_cycle(step,best))
    assert [d['stop'] for d in decisions]==[False,False,False,False,True]
    controller=Plateau(cfg['plateau'],20000);controller.load_state_dict(copy.deepcopy(state['plateau']))
    best=state['best_val_mse']
    for step in range(300000,360001,20000):best*=.9995;controller.completed_cycle(step,best)
    best*=.998;check=controller.completed_cycle(380000,best)
    assert not check['stop'] and controller.b_count==0 and controller.stop_armed_cycle is None
    assert sha(checkpoint)==source_hash
    save(RUN/'bootstrap_checks.json',dict(status='passed',all_model_optimizer_ema_scheduler_rng_state_equal=True,
        source_unchanged=True,strict_plateau_extra_cycle_test=True,improvement_resets_patience=True,
        old_stop_threshold=.005,new_stop_threshold=.001,old_patience=3,new_patience=4,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    del restored,original,state
    from train import load_data,build,validate,restore_rng
    data,train_data,val=load_data();model,ema,opt,schedule,plateau=build(cfg,data)
    state=torch.load(RUN/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer'])
    schedule.load_state_dict(state['scheduler']);plateau.load_state_dict(state['plateau']);restore_rng(state['rng'])
    value=validate(ema,*val,cfg['validation']);repeat=validate(ema,*val,cfg['validation'])
    assert abs(value-state['best_val_mse'])<1e-7 and value==repeat
    assert abs(schedule.value(280001)-.0001)<1e-15
    save(RUN/'checks.json',dict(status='passed',restored_validation_mse=value,repeat_validation_mse=repeat,
        original_validation_mse=state['best_val_mse'],next_update=280001,next_lr=schedule.value(280001),
        gpu=torch.cuda.get_device_name(),source_sha256=source_hash))
    print(read(RUN/'checks.json'))

if __name__=='__main__':main()
