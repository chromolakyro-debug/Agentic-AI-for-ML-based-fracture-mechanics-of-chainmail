"""Bounded CFM training, EMA validation selection, and exact-state resumable checkpoints."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[name] = '4'
import argparse
import copy
import json
import math
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from common import DATA, OUT, SEED, save_json, read_json, sha_file
from model import ConditionalUNet, sample


def setup(data_dir=None, run_dir=None):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Optional explicit paths allow reading a separately curated dataset without
    # changing the original experiment's default data/run directories.
    d = np.load((OUT if run_dir is None else Path(run_dir)) / 'data.npz', allow_pickle=False)
    images = np.load((DATA if data_dir is None else Path(data_dir)) / 'dataset.npy', allow_pickle=False)
    return d, images


def atomic_torch_save(value, path):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


@torch.inference_mode()
def validation_loss(model, images, conditions, batch=16):
    model.eval()
    generator = torch.Generator(device='cuda').manual_seed(SEED + 567)
    total = 0.
    for start in range(0, len(images), batch):
        x = images[start:start+batch].float()[:, None] * 2 - 1
        x = x.contiguous(memory_format=torch.channels_last)
        c = conditions[start:start+batch]
        epsilon = torch.randn(x.shape, device='cuda', generator=generator)
        t = torch.rand((len(x),), device='cuda', generator=generator)
        xt = (1 - t[:,None,None,None]) * epsilon + t[:,None,None,None] * x
        with torch.autocast('cuda', dtype=torch.bfloat16): prediction = model(xt, t, c)
        loss = (prediction.float() - (x - epsilon)).square().flatten(1).mean(1)
        total += loss.sum().item()
    return total / len(images)


def smoke(batch):
    d, images = setup()
    model = ConditionalUNet().cuda().to(memory_format=torch.channels_last)
    indices = d['train'][[0, 1000, 3000, 5000]]
    x = torch.as_tensor(images[indices], device='cuda', dtype=torch.float32)[:,None] * 2 - 1
    c = torch.as_tensor(((d['sa'][indices] - d['mean']) / d['std']).astype(np.float32), device='cuda')
    eps = torch.randn_like(x)
    t = torch.tensor([.2, .4, .6, .8], device='cuda')
    xt = (1 - t[:,None,None,None]) * eps + t[:,None,None,None] * x
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0., foreach=True)
    losses = []
    started = time.perf_counter()
    for step in range(160):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16): pred = model(xt, t, c)
        loss = (pred.float() - (x-eps)).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(loss.item())
        if step % 40 == 0: print('overfit', step, losses[-1], flush=True)
    assert np.isfinite(losses).all() and losses[-1] < .15 * losses[0], losses[-1]
    with torch.inference_mode():
        model.eval()
        generated = sample(model, c[:1].expand(2,-1), [123,124], steps=16)
        assert torch.isfinite(generated).all()
        binary = (generated >= 0).cpu().numpy().astype(np.uint8)
        np.save(OUT / 'smoke_binary.npy', binary)
    # Actual full-resolution batch training benchmark, fresh noise/time each iteration.
    xx = x.repeat(math.ceil(batch/4),1,1,1)[:batch].contiguous(memory_format=torch.channels_last)
    cc = c.repeat(math.ceil(batch/4),1)[:batch]
    timings = []
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for i in range(12):
        torch.cuda.synchronize(); tick = time.perf_counter()
        e = torch.randn_like(xx); tt = torch.rand(batch, device='cuda')
        zz = (1-tt[:,None,None,None])*e + tt[:,None,None,None]*xx
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16): p = model(zz,tt,cc)
        ll = (p.float()-(xx-e)).square().mean()
        ll.backward(); optimizer.step()
        torch.cuda.synchronize(); timings.append(time.perf_counter()-tick)
    save_json(OUT / 'smoke_training.json', dict(overfit_indices=indices.tolist(), steps=160,
              overfit_loss_initial=losses[0], overfit_loss_final=losses[-1], losses=losses,
              caveat='Fixed (4 images, noise, time) wiring/optimization smoke, not a generative quality test',
              batch=batch, benchmark_step_seconds=float(np.median(timings[3:])),
              gpu=torch.cuda.get_device_name(), model_parameters=sum(p.numel() for p in model.parameters()),
              peak_memory_gb=torch.cuda.max_memory_allocated()/2**30, elapsed_seconds=time.perf_counter()-started))
    print(read_json(OUT/'smoke_training.json') | {'losses':'saved'}, flush=True)


def train(args):
    d, images = setup()
    protocol = read_json(OUT/'protocol.json')
    assert sha_file(OUT/'protocol.json') == read_json(OUT/'frozen_protocol_sha256.json')['sha256']
    checkpoint_dir = OUT / 'checkpoints'; checkpoint_dir.mkdir(exist_ok=True)
    config = dict(seed=SEED, steps=args.steps, wall_budget_minutes=args.minutes, batch=args.batch,
                  architecture='FiLM conditional residual U-Net [32,64,128,192,256], 14px attention, single channel 224px',
                  model_parameters=0, lr=2e-4, weight_decay=.01, warmup_steps=200, ema_decay=.999,
                  validation_interval=1000, checkpoint_selection='minimum fixed-noise validation EMA velocity MSE',
                  stop='first of fixed max steps or wall budget; no hyperparameter search or result-based extension',
                  augmentation='none', precision='bfloat16 autocast, FP32 loss and optimizer state',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                  cudnn_benchmark=True, cross_hardware_bitwise_reproducibility=False,
                  protocol_sha256=sha_file(OUT/'protocol.json'))
    model = ConditionalUNet().cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    config['model_parameters'] = sum(p.numel() for p in model.parameters())
    if (OUT/'train_config.json').exists():
        assert config == read_json(OUT/'train_config.json'), 'Resume must use the original fixed configuration'
    else: save_json(OUT/'train_config.json', config)
    tr = torch.as_tensor(images[d['train']],device='cuda')
    va = torch.as_tensor(images[d['validation']],device='cuda')
    tc = torch.as_tensor(((d['sa'][d['train']]-d['mean'])/d['std']).astype(np.float32), device='cuda')
    vc = torch.as_tensor(((d['sa'][d['validation']]-d['mean'])/d['std']).astype(np.float32), device='cuda')
    weights = torch.as_tensor(d['train_weights'],device='cuda')
    del images
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], foreach=True)
    start_step, prior_elapsed, best_loss, best_step = 0, 0., float('inf'), 0
    last = checkpoint_dir/'last.pt'
    if args.resume and last.exists():
        ckpt = torch.load(last,map_location='cuda')
        assert ckpt['config'] == config
        model.load_state_dict(ckpt['model']); ema.load_state_dict(ckpt['ema']); optimizer.load_state_dict(ckpt['optimizer'])
        start_step, prior_elapsed, best_loss, best_step = ckpt['step'], ckpt['elapsed_seconds'], ckpt['best_loss'], ckpt['best_step']
        torch.set_rng_state(ckpt['rng_cpu'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in ckpt['rng_cuda']])
        np.random.set_state(ckpt['rng_numpy']); random.setstate(ckpt['rng_python'])
        del ckpt
    elif last.exists():
        raise RuntimeError('Existing run requires --resume')
    torch.cuda.reset_peak_memory_stats()
    model.train()
    started = time.perf_counter()
    running, n_running = 0., 0
    step = start_step
    with (OUT/'training.jsonl').open('a',encoding='utf-8') as log:
        for step in range(start_step+1,args.steps+1):
            lr = config['lr'] * min(step/config['warmup_steps'],1.) * (.1 + .9*.5*(1+math.cos(math.pi*step/args.steps)))
            optimizer.param_groups[0]['lr'] = lr
            idx = torch.multinomial(weights,args.batch,replacement=True)
            x = tr[idx].float()[:,None] * 2 - 1
            x = x.contiguous(memory_format=torch.channels_last)
            c = tc[idx]
            epsilon = torch.randn_like(x)
            t = torch.rand(args.batch,device='cuda')
            xt = (1-t[:,None,None,None])*epsilon + t[:,None,None,None]*x
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): prediction = model(xt,t,c)
            loss = (prediction.float()-(x-epsilon)).square().mean()
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at step {step}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            with torch.no_grad():
                decay = min(config['ema_decay'], (1+step)/(10+step))
                torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-decay)
            running += loss.item(); n_running += 1
            elapsed = prior_elapsed + time.perf_counter()-started
            stop = elapsed >= args.minutes*60 or step == args.steps
            if step % 100 == 0 or stop:
                row = dict(step=step,loss=running/n_running,lr=lr,grad_norm=float(grad_norm),elapsed_seconds=elapsed)
                log.write(json.dumps(row)+'\n'); log.flush()
                print(json.dumps(row),flush=True)
                running,n_running = 0.,0
            if step % 1000 == 0 or stop:
                value = validation_loss(ema,va,vc)
                if value < best_loss:
                    best_loss,best_step = value,step
                    atomic_torch_save(dict(ema=ema.state_dict(),step=step,validation_loss=value,config=config),checkpoint_dir/'best.pt')
                row = dict(step=step,validation_velocity_mse=value,best_step=best_step,best_loss=best_loss,elapsed_seconds=prior_elapsed+time.perf_counter()-started)
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row),flush=True)
                atomic_torch_save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                        step=step,elapsed_seconds=prior_elapsed+time.perf_counter()-started,best_loss=best_loss,best_step=best_step,
                        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),
                        rng_python=random.getstate(),config=config),last)
            if stop: break
    save_json(OUT/'training_summary.json',dict(steps=step,images_seen=step*args.batch,nominal_epochs=step*args.batch/len(tr),
              elapsed_seconds=prior_elapsed+time.perf_counter()-started,best_step=best_step,best_validation_loss=best_loss,
              peak_memory_gb=torch.cuda.max_memory_allocated()/2**30,stop_reason='step_budget' if step==args.steps else 'wall_budget'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['smoke','train'])
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--minutes',type=float,default=75.)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.mode=='smoke': smoke(args.batch)
    else: train(args)
