"""CFM v1 ablation: identical training, with D4 image/label transforms disabled."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
import sys,copy,random,time,json,argparse
import numpy as np
import torch
from common import ROOT,RUN,DATA,read,save,sha
sys.path.insert(0,str(ROOT))
from cfm_v0.model import ConditionalUNet
from augmentation import apply,conditions
from control import RestartSchedule,Plateau
from retention import Retention,POLICY

def atomic(value,path):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)
def seed_all(seed):
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
def rng_state():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(r):
    torch.set_rng_state(r['cpu'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda']]);np.random.set_state(r['numpy']);random.setstate(r['python'])

def build(cfg,data):
    seed_all(cfg['seed']);model=ConditionalUNet(**cfg['architecture']).cuda().to(memory_format=torch.channels_last)
    assert sum(p.numel() for p in model.parameters())==cfg['model_parameters']
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['scheduler']['max_lr'],weight_decay=cfg['optimizer']['weight_decay'],foreach=True)
    return model,ema,optimizer,RestartSchedule(optimizer,cfg['scheduler']),Plateau(cfg['plateau'],cfg['scheduler']['cycle_steps'])

def update(model,ema,opt,schedule,data,cfg,step):
    tr,moduli,mean,std,weights=data;batch=cfg['batch'];lr=schedule.step(step)
    idx=torch.multinomial(weights,batch,replacement=True)
    turns=torch.randint(0,4,(batch,),device='cuda');flips=torch.rand(batch,device='cuda')<.5
    # Consume the same D4 random draws as v1 to preserve sample/noise/time RNG streams.
    # Neither draw changes images or labels in this no-augmentation ablation.
    binary,mapped=tr[idx],moduli[idx]
    x=(binary.float()[:,None]*2-1).contiguous(memory_format=torch.channels_last);c=conditions(mapped,mean,std)
    epsilon=torch.randn_like(x);t=torch.rand(batch,device='cuda');xt=(1-t[:,None,None,None])*epsilon+t[:,None,None,None]*x
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(xt,t,c)
    loss=(pred.float()-(x-epsilon)).square().mean()
    if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite velocity loss at {step}')
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
    if not torch.isfinite(grad):raise RuntimeError(f'Nonfinite gradient at {step}')
    opt.step()
    with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-min(cfg['ema_decay'],(1+step)/(10+step)))
    return dict(loss=float(loss),lr=lr,grad_norm=float(grad),sampled_local_indices=idx,rotations=turns,flips=flips)

@torch.inference_mode()
def validate(model,images,moduli,mean,std,cfg):
    model.eval();generator=torch.Generator(device='cuda').manual_seed(cfg['noise_seed']);total=0.;batch=cfg['batch']
    for start in range(0,len(images),batch):
        x=(images[start:start+batch].float()[:,None]*2-1).contiguous(memory_format=torch.channels_last);c=conditions(moduli[start:start+batch],mean,std)
        noise=torch.randn(x.shape,device='cuda',generator=generator);t=torch.rand(len(x),device='cuda',generator=generator)
        xt=(1-t[:,None,None,None])*noise+t[:,None,None,None]*x
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(xt,t,c)
        total+=(pred.float()-(x-noise)).square().flatten(1).mean(1).sum().item()
    return total/len(images)

def load_data():
    d=np.load(RUN/'data.npz',allow_pickle=False);images=np.load(DATA/'dataset.npy',mmap_mode='r')
    tr=torch.as_tensor(np.array(images[d['train']]),device='cuda');va=torch.as_tensor(np.array(images[d['validation']]),device='cuda')
    mean=torch.as_tensor(d['mean'],device='cuda');std=torch.as_tensor(d['std'],device='cuda')
    tm=torch.as_tensor(d['moduli'][d['train']],device='cuda');vm=torch.as_tensor(d['moduli'][d['validation']],device='cuda')
    weights=torch.as_tensor(d['train_weights'],device='cuda')
    assert np.array_equal(d['train_weight_sample_ids'],d['sample_ids'][d['train']])
    return d,(tr,tm,mean,std,weights),(va,vm,mean,std)

def train(until,resume):
    cfg=read(RUN/'config.json');assert sha(RUN/'config.json')==read(RUN/'frozen_config.json')['sha256']
    assert sha(DATA/'metadata.csv')==cfg['input_hashes']['metadata'];assert sha(DATA/'dataset.npy')==cfg['input_hashes']['array']
    assert sha(RUN/'data.npz')==cfg['input_hashes']['data'];assert sha(RUN/'splits.json')==cfg['input_hashes']['splits'];assert sha(ROOT/'cfm_v0/model.py')==cfg['input_hashes']['model_code']
    cp=RUN/'checkpoints';cp.mkdir(exist_ok=True);last=cp/'last.pt'
    d,data,val=load_data();model,ema,opt,schedule,plateau=build(cfg,d)
    step=0;prior=0.;best=float('inf');best_step=0;cycle_best=float('inf');cycle_started=0
    retention=Retention();save(RUN/'checkpoint_retention_policy.json',POLICY)
    if last.exists():
        if not resume:raise ValueError('Existing run requires --resume')
        state=torch.load(last,map_location='cpu',weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer']);schedule.load_state_dict(state['scheduler']);plateau.load_state_dict(state['plateau'])
        step=state['step'];prior=state['training_seconds'];best=state['best_val_mse'];best_step=state['best_step'];cycle_best=state['cycle_best'];cycle_started=state['cycle_started']
        retention.load_state_dict(state['retention']);restore_rng(state['rng']);del state
    assert step==schedule.updates and step<until<=cfg['limits']['max_updates']
    started=time.perf_counter();last_log_time=started;last_log_step=step;model.train();reason='segment_complete';torch.cuda.reset_peak_memory_stats()
    def checkpoint():
        atomic(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),plateau=plateau.state_dict(),
            step=step,training_seconds=prior+time.perf_counter()-started,best_val_mse=best,best_step=best_step,cycle_best=cycle_best,cycle_started=cycle_started,
            samples_seen=step*cfg['batch'],rng=rng_state(),config=cfg,retention=retention.state_dict()),last)
        if retention.due(step):
            import shutil
            target=cp/f'step_{step:06d}.pt'
            digest=sha(last)
            if target.exists():assert sha(target)==digest, 'Refusing to replace retained checkpoint'
            else:shutil.copyfile(last,target)
            inventory=read(RUN/'retained_checkpoints.json') if (RUN/'retained_checkpoints.json').exists() else []
            if not any(r['step']==step for r in inventory):
                inventory.append(dict(step=step,trigger_step=retention.trigger_step,path=target.relative_to(ROOT).as_posix(),sha256=digest,bytes=target.stat().st_size))
                save(RUN/'retained_checkpoints.json',inventory)
    with (RUN/'training.jsonl').open('a',encoding='utf-8') as log:
        def record(row):log.write(json.dumps(row)+'\n')
        for step in range(step+1,until+1):
            r=update(model,ema,opt,schedule,data,cfg,step);elapsed=prior+time.perf_counter()-started
            record(dict(kind='train',step=step,loss=r['loss'],lr=r['lr'],grad_norm=r['grad_norm'],samples_seen=step*cfg['batch'],equivalent_epochs=step*cfg['batch']/len(data[0]),training_seconds=elapsed))
            stop_budget=elapsed>=cfg['limits']['max_training_seconds'] or step==cfg['limits']['max_updates']
            cycle_end=step%cfg['scheduler']['cycle_steps']==0
            if step%cfg['validation']['interval']==0 or stop_budget:
                value=validate(ema,*val,cfg['validation']);cycle_best=min(cycle_best,value)
                if value<best:
                    best=value;best_step=step
                    atomic(dict(ema=ema.state_dict(),step=step,validation_loss=value,config=cfg),cp/'best_val_mse.pt')
                row=dict(kind='validation',step=step,validation_velocity_mse=value,best_val_mse=best,best_step=best_step,lr=r['lr'],training_seconds=prior+time.perf_counter()-started)
                record(row);print(json.dumps(row),flush=True)
                retention_row=retention.observe(step,value)
                record(dict(kind='checkpoint_retention',**retention_row));save(RUN/'checkpoint_retention_state.json',retention.state_dict())
                if retention_row['triggered']:print(json.dumps(dict(kind='checkpoint_retention_triggered',**retention_row)),flush=True)
            if cycle_end:
                record_cycle=plateau.completed_cycle(step,cycle_best);record(dict(kind='plateau',**record_cycle));save(RUN/'plateau_history.json',plateau.history)
                cycle_best=float('inf');cycle_started=step
                if record_cycle['stop']:reason='convergence_criterion'
            if stop_budget:reason='update_budget' if step==cfg['limits']['max_updates'] else 'training_time_budget'
            stop=stop_budget or reason=='convergence_criterion' or step==until
            if step%cfg['log_interval']==0 or stop:
                tick=time.perf_counter();row=dict(step=step,loss=r['loss'],lr=r['lr'],best_step=best_step,best_val_mse=best if np.isfinite(best) else None,
                    training_seconds=prior+tick-started,images_per_second=(step-last_log_step)*cfg['batch']/(tick-last_log_time),gpu_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    gpu_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,samples_seen=step*cfg['batch'],equivalent_epochs=step*cfg['batch']/len(data[0]),eval_active=plateau.eval_active)
                save(RUN/'progress.json',row);record(dict(kind='progress',**row));log.flush();print(json.dumps(row),flush=True);last_log_time=tick;last_log_step=step
            if step%cfg['checkpoint_interval']==0 or stop or retention.due(step):checkpoint()
            if stop:break
    summary=dict(step=step,best_step=best_step,best_val_mse=best,training_seconds=prior+time.perf_counter()-started,reason=reason,eval_active=plateau.eval_active,
        restart_count=(step-1)//cfg['scheduler']['cycle_steps'],samples_seen=step*cfg['batch'],equivalent_epochs=step*cfg['batch']/len(data[0]))
    save(RUN/'segment_result.json',summary)
    if reason!='segment_complete':save(RUN/'training_summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--until',type=int,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    train(args.until,args.resume)
