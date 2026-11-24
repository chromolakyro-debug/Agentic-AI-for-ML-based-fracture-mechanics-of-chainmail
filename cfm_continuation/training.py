"""Full-state continuation with separate immutable schedule, validation and budgets."""
import paths
import argparse
import copy
import json
import math
import random
import time
import numpy as np
import torch
from common import OUT, read_json, save_json, sha_file
from model import ConditionalUNet
from train import setup, validation_loss, atomic_torch_save


def load_state(source):
    d, images = setup()
    model = ConditionalUNet().cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01, foreach=True)
    ckpt = torch.load(source, map_location='cuda')
    assert all(k in ckpt for k in ('model','ema','optimizer','rng_cpu','rng_cuda','rng_numpy','rng_python'))
    model.load_state_dict(ckpt['model']); ema.load_state_dict(ckpt['ema']); optimizer.load_state_dict(ckpt['optimizer'])
    tr = torch.as_tensor(images[d['train']], device='cuda')
    va = torch.as_tensor(images[d['validation']], device='cuda')
    tc = torch.as_tensor(((d['sa'][d['train']]-d['mean'])/d['std']).astype(np.float32), device='cuda')
    vc = torch.as_tensor(((d['sa'][d['validation']]-d['mean'])/d['std']).astype(np.float32), device='cuda')
    weights = torch.as_tensor(d['train_weights'],device='cuda')
    torch.set_rng_state(ckpt['rng_cpu'].cpu()); torch.cuda.set_rng_state_all([r.cpu() for r in ckpt['rng_cuda']])
    np.random.set_state(ckpt['rng_numpy']); random.setstate(ckpt['rng_python'])
    # Retain scalar/RNG metadata, release redundant GPU tensor copies.
    meta = {k:v for k,v in ckpt.items() if k not in ('model','ema','optimizer')}
    return model, ema, optimizer, (tr,va,tc,vc,weights), meta


def update(model, ema, optimizer, data, batch, step, lr):
    tr,va,tc,vc,weights = data
    optimizer.param_groups[0]['lr'] = lr
    indices = torch.multinomial(weights,batch,replacement=True)
    x = (tr[indices].float()[:,None]*2-1).contiguous(memory_format=torch.channels_last)
    epsilon = torch.randn_like(x); t = torch.rand(batch,device='cuda')
    xt = (1-t[:,None,None,None])*epsilon+t[:,None,None,None]*x
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16): prediction=model(xt,t,tc[indices])
    loss=(prediction.float()-(x-epsilon)).square().mean()
    if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss {step}')
    loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(grad): raise RuntimeError(f'Nonfinite gradient {step}')
    optimizer.step()
    with torch.no_grad(): torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-min(.999,(1+step)/(10+step)))
    return loss.item(), float(grad)


def learning_rate(step, restart=None):
    # Global 20k is the source checkpoint. Warm up continuously, then very slow cosine to a 3e-5 floor.
    origin=20000; start_lr=2e-5; peak=1e-4
    if restart:
        origin=restart['step'];start_lr=restart['from_lr'];peak=8e-5
    local=step-origin
    if local<=2000: return start_lr+(peak-start_lr)*max(local,0)/2000
    fraction=min(1.,(local-2000)/max(1,200000-origin-2000))
    return 3e-5+(peak-3e-5)*.5*(1+math.cos(math.pi*fraction))


def benchmark(batch):
    started=time.perf_counter(); record=dict(batch=batch,source_checkpoint=str(paths.ORIGINAL/'checkpoints'/'last.pt'),warmup_steps=10,measured_steps=40)
    try:
        model,ema,opt,data,meta=load_state(paths.ORIGINAL/'checkpoints'/'last.pt'); model.train()
        assert meta['step']==20000 and opt.param_groups[0]['lr']==2e-5
        torch.cuda.reset_peak_memory_stats();times=[];losses=[];grads=[]
        for i in range(50):
            torch.cuda.synchronize();tick=time.perf_counter()
            loss,grad=update(model,ema,opt,data,batch,20001+i,learning_rate(20001+i))
            torch.cuda.synchronize();duration=time.perf_counter()-tick
            if i>=10: times.append(duration);losses.append(loss);grads.append(grad)
        free,total=torch.cuda.mem_get_info()
        record.update(status='stable',images_per_second=batch*len(times)/sum(times),median_step_seconds=float(np.median(times)),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            free_device_gib=free/2**30,total_device_gib=total/2**30,loss_min=min(losses),loss_max=max(losses),grad_max=max(grads),
            gpu=torch.cuda.get_device_name(),model_parameters=sum(p.numel() for p in model.parameters()))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        record.update(status='failed',error=repr(error))
    record['elapsed_seconds']=time.perf_counter()-started
    save_json(OUT/f'benchmark_batch_{batch}.json',record);print(json.dumps(record),flush=True)


def configure():
    assert not (OUT/'continuation_config.json').exists()
    results=[read_json(OUT/f'benchmark_batch_{b}.json') for b in (16,32,64)]
    base=results[0];assert base['status']=='stable'
    b32=results[1]
    use32=b32['status']=='stable' and b32['images_per_second']>base['images_per_second']*1.03 and b32['peak_reserved_gib']<20.5 and b32['free_device_gib']>2
    batch=32 if use32 else 16
    config=dict(source_checkpoint=str(paths.ORIGINAL/'checkpoints'/'last.pt'),source_checkpoint_sha256=sha_file(paths.ORIGINAL/'checkpoints'/'last.pt'),
       source_config=read_json(paths.ORIGINAL/'train_config.json'),protocol_sha256=sha_file(OUT/'protocol.json'),
       batch=batch,batch_selection_rule='Prefer 32 if measured throughput >16 by 3%, peak reserved <20.5 GiB and device free >2 GiB; otherwise16. 64 benchmarked for comparison, not preferred.',
       max_global_steps=200000,max_additional_training_seconds=28800,minimum_plateau_stop_global_step=80000,
       validation_interval=5000,validation_batch=16,validation_scheme='Original train.validation_loss: 800 frozen validation rows in original order, CUDA seed 20261114+567, fixed batch16, fresh identical noise/time every call, unweighted EMA velocity MSE.',
       milestone_interval=20000,lr_schedule='Global20k->22k linear 2e-5->1e-4; 22k->200k cosine to floor3e-5. Original optimizer moments/EMA/RNG preserved; original cosine NOT resumed.',
       optional_restart_rule='At most one further restart only after two plateau windows and current LR<=5e-5: warm 2000steps to8e-5, decay to3e-5 at200k. Requires recorded development review.',
       plateau_rule='At milestones >=80k: best validation relative improvement <0.5% in BOTH immediately preceding nonoverlapping 20k windows. Stop only after fixed-sample visual/FEM review finds no meaningful continued visual improvement; otherwise continue to safety cap.',
       checkpoint_selection='At stop, use minimum fixed validation EMA MSE unless fixed development visual review documents a material regression; any override must precede final generation.',
       parameter_count=11529633,architecture=read_json(paths.ORIGINAL/'train_config.json')['architecture'],augmentation='none',ema_decay=.999,
       gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,weight_decay=.01,gradient_clip_norm=1.,benchmarks=results)
    save_json(OUT/'continuation_config.json',config);save_json(OUT/'train_config.json',config)
    print(json.dumps({'selected_batch':batch,'config':str(OUT/'continuation_config.json')}))


def train_until(until):
    assert not (OUT/'training_summary.json').exists(), 'Run finalized: preserve this audit and migrate last.pt into a new run for future training'
    config=read_json(OUT/'continuation_config.json');cp=OUT/'checkpoints';last=cp/'last.pt'
    assert sha_file(OUT/'protocol.json')==config['protocol_sha256']
    source=last if last.exists() else paths.ORIGINAL/'checkpoints'/'last.pt'
    if not last.exists(): assert sha_file(source)==config['source_checkpoint_sha256']
    model,ema,opt,data,meta=load_state(source)
    if last.exists(): assert meta['config']==config, 'Continuation resume config must stay identical'
    else: assert meta['config']==config['source_config'] and meta['step']==20000
    step=meta['step'];assert step<until<=config['max_global_steps']
    elapsed_prior=meta.get('continuation_elapsed_seconds',0.);images_seen=meta.get('images_seen',320000)
    if (OUT/'progress.json').exists():
        previous=read_json(OUT/'progress.json')
        if previous['step']==step:
            # The sidecar also includes the time spent serializing the final checkpoint itself.
            elapsed_prior=max(elapsed_prior,previous['continuation_elapsed_seconds'])
    best_loss=meta['best_loss'];best_step=meta['best_step'];restart=meta.get('lr_restart')
    started=time.perf_counter();block=time.perf_counter();sumloss=0.;nloss=0;batch=config['batch']
    torch.cuda.reset_peak_memory_stats();model.train()
    records=[]
    def write(row,stream):
        stream.write(json.dumps(row)+'\n');stream.flush()
        if 'validation_velocity_mse' in row or row['step']%1000==0:print(json.dumps(row),flush=True)
    def save_last():
        atomic_torch_save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=step,
            continuation_elapsed_seconds=elapsed_prior+time.perf_counter()-started,elapsed_seconds=2005.493062700145+elapsed_prior+time.perf_counter()-started,
            images_seen=images_seen,best_loss=best_loss,best_step=best_step,lr_restart=restart,
            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),config=config),last)
    with (OUT/'training.jsonl').open('a',encoding='utf-8') as stream:
        if step==20000 and not meta.get('continuation_elapsed_seconds'):
            verified=validation_loss(ema,data[1],data[3],batch=16)
            # cuDNN benchmark kernel selection can differ after a new process/load;
            # identical samples/noise/time need not be bitwise equal under BF16.
            assert abs(verified-best_loss)<1e-5,(verified,best_loss)
            save_json(OUT/'restored_20k_validation.json',dict(step=step,value=verified,original=best_loss,absolute_difference=abs(verified-best_loss),
                comparison_absolute_tolerance=1e-5,note='Fixed validation draws unchanged; BF16/cuDNN autotuning is not cross-process bitwise deterministic.',
                optimizer_steps=sorted(set(float(v['step']) for v in opt.state.values())),optimizer_lr=opt.param_groups[0]['lr'],restored_full_state=True))
            best_loss=verified
            atomic_torch_save(dict(ema=ema.state_dict(),step=step,validation_loss=verified,config=config),cp/'best.pt')
            write(dict(step=step,validation_velocity_mse=verified,best_loss=best_loss,best_step=best_step,images_seen=images_seen,continuation_elapsed_seconds=0.),stream)
        for step in range(step+1,until+1):
            model.train();lr=learning_rate(step,restart)
            loss,grad=update(model,ema,opt,data,batch,step,lr);sumloss+=loss;nloss+=1;images_seen+=batch
            elapsed=elapsed_prior+time.perf_counter()-started
            # Reserve ten seconds for final validation/checkpoint serialization under the 8h cap.
            safety=elapsed>=config['max_additional_training_seconds']-10 or step==config['max_global_steps']
            boundary=step==until or safety
            if step%100==0 or boundary:
                write(dict(step=step,loss=sumloss/nloss,loss_averaging_steps=nloss,lr=lr,grad_norm=grad,images_seen=images_seen,equivalent_epochs=images_seen/6400,
                    images_per_second=batch*nloss/(time.perf_counter()-block),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,continuation_elapsed_seconds=elapsed),stream)
                sumloss=0.;nloss=0;block=time.perf_counter()
            if step%5000==0 or boundary:
                value=validation_loss(ema,data[1],data[3],batch=16)
                if value<best_loss:
                    best_loss=value;best_step=step
                    atomic_torch_save(dict(ema=ema.state_dict(),step=step,validation_loss=value,config=config),cp/'best.pt')
                write(dict(step=step,validation_velocity_mse=value,best_step=best_step,best_loss=best_loss,lr=lr,images_seen=images_seen,
                           continuation_elapsed_seconds=elapsed_prior+time.perf_counter()-started),stream)
                save_last();block=time.perf_counter()
            if boundary:
                atomic_torch_save(dict(ema=ema.state_dict(),step=step,validation_loss=value,config=config),cp/f'milestone_{step:06d}.pt')
                break
    save_json(OUT/'progress.json',dict(step=step,images_seen=images_seen,equivalent_epochs=images_seen/6400,
       continuation_elapsed_seconds=elapsed_prior+time.perf_counter()-started,best_step=best_step,best_validation_loss=best_loss,
       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
       segment_exit_reason='safety_cap' if safety else 'milestone_review_required',last_lr=lr,lr_restart=restart))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['benchmark','configure','train'])
    parser.add_argument('--batch',type=int);parser.add_argument('--until',type=int)
    a=parser.parse_args()
    if a.mode=='benchmark':benchmark(a.batch)
    elif a.mode=='configure':configure()
    else:train_until(a.until)
