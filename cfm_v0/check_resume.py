"""Read-only checkpoint recovery check: restore twice and compare the next training update."""
import copy
import math
import random
import numpy as np
import torch
from common import OUT, DATA, save_json
from model import ConditionalUNet


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    checkpoint=torch.load(OUT/'checkpoints'/'last.pt',map_location='cpu')
    d=np.load(OUT/'data.npz');im=np.load(DATA/'dataset.npy',mmap_mode='r')
    images=torch.as_tensor(im[d['train']],device='cuda')
    conditions=torch.as_tensor(((d['sa'][d['train']]-d['mean'])/d['std']).astype(np.float32),device='cuda')
    weights=torch.as_tensor(d['train_weights'],device='cuda')
    results=[];parameter_reference=None;max_difference=0.
    for trial in range(2):
        model=ConditionalUNet().cuda().to(memory_format=torch.channels_last).train()
        model.load_state_dict(checkpoint['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=checkpoint['config']['lr'],weight_decay=.01,foreach=True)
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        torch.set_rng_state(checkpoint['rng_cpu']);torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        np.random.set_state(checkpoint['rng_numpy']);random.setstate(checkpoint['rng_python'])
        step=checkpoint['step']+1;cfg=checkpoint['config']
        lr=cfg['lr']*min(step/cfg['warmup_steps'],1.)*(.1+.9*.5*(1+math.cos(math.pi*step/cfg['steps'])))
        optimizer.param_groups[0]['lr']=lr
        idx=torch.multinomial(weights,cfg['batch'],replacement=True)
        x=(images[idx].float()[:,None]*2-1).contiguous(memory_format=torch.channels_last)
        epsilon=torch.randn_like(x);t=torch.rand(cfg['batch'],device='cuda')
        xt=(1-t[:,None,None,None])*epsilon+t[:,None,None,None]*x
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(xt,t,conditions[idx])
        loss=(prediction.float()-(x-epsilon)).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        results.append(dict(loss=float(loss),sample_indices=idx.cpu().tolist(),times=t.cpu().tolist(),
                            epsilon_prefix=epsilon.flatten()[:8].cpu().tolist()))
        if trial==0: parameter_reference=[p.detach().cpu().clone() for p in model.parameters()]
        else:
            max_difference=max(float((reference-p.detach().cpu()).abs().max()) for reference,p in zip(parameter_reference,model.parameters()))
        del model,optimizer
        torch.cuda.empty_cache()
    assert results[0]['sample_indices']==results[1]['sample_indices']
    assert results[0]['times']==results[1]['times'] and results[0]['epsilon_prefix']==results[1]['epsilon_prefix']
    assert abs(results[0]['loss']-results[1]['loss'])<1e-6 and max_difference<1e-6
    save_json(OUT/'resume_check.json',dict(restored_checkpoint_step=checkpoint['step'],trials=results,
              max_parameter_difference_after_update=max_difference,passed=True,
              note='Read-only restoration into disposable models; the main run and stored checkpoint were not modified'))
    print('Checkpoint restoration passed',checkpoint['step'],max_difference,flush=True)


if __name__=='__main__':main()
