"""Bounded wiring/resume tests. All diagnostic weights are discarded."""
import copy,time,tempfile
import numpy as np
import torch
from common import RUN,HERE,read,save
from augmentation import apply,conditions
from control import RestartSchedule,Plateau
from train import build,update,load_data,validate,rng_state,restore_rng,atomic

def main():
    cfg=read(RUN/'config.json');d,data,val=load_data();model,ema,opt,scheduler,plateau=build(cfg,d)
    # All eight precise transformations, with non-symmetric images and physical moduli.
    a=data[0][:8];m=data[1][:8];k=torch.arange(8,device='cuda')//2;flip=torch.arange(8,device='cuda')%2==1
    b,e=apply(a,m,k,flip)
    for i in range(8):
        expected=torch.rot90(a[i],int(k[i]),(-2,-1));expected=expected.flip(-1) if flip[i] else expected
        assert torch.equal(expected,b[i]);assert int(a[i].sum())==int(b[i].sum());assert b.dtype==a.dtype
        assert torch.equal(e[i],m[i].flip(-1) if int(k[i])%2 else m[i])
    c=conditions(e,data[2],data[3]).cpu().numpy();physical=e.cpu().numpy();values=np.stack([.5*np.log(physical).sum(-1),np.log(physical[:,0]/physical[:,1])],-1)
    np.testing.assert_allclose(c,((values-d['mean'])/d['std']).astype(np.float32),rtol=1e-6,atol=1e-6)
    # A checkpoint exactly at a synthetic cycle boundary tests that restart position survives restore.
    scheduler.updates=19998
    update(model,ema,opt,scheduler,data,cfg,19999);update(model,ema,opt,scheduler,data,cfg,20000)
    path=RUN/'resume_check_temporary.pt'
    atomic(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state()),path)
    first=update(model,ema,opt,scheduler,data,cfg,20001);expected_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    del model,ema,opt,scheduler;torch.cuda.empty_cache()
    model,ema,opt,scheduler,plateau=build(cfg,d);state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'])
    second=update(model,ema,opt,scheduler,data,cfg,20001)
    for key in ['sampled_local_indices','rotations','flips']:assert torch.equal(first[key],second[key])
    assert first['loss']==second['loss'] and first['lr']==second['lr']==1e-4
    maximum=max(float((expected_weights[k]-v.detach().cpu()).abs().max()) for k,v in model.state_dict().items());assert maximum==0
    # Full fixed validation twice: verify stable EMA comparison, not an estimate of FEM error.
    v1=validate(ema,*val,cfg['validation']);v2=validate(ema,*val,cfg['validation']);assert v1==v2
    model.train();timings=[];losses=[];torch.cuda.reset_peak_memory_stats()
    for j in range(30):
        torch.cuda.synchronize();start=time.perf_counter();r=update(model,ema,opt,scheduler,data,cfg,20002+j);torch.cuda.synchronize()
        if j>=10:timings.append(time.perf_counter()-start);losses.append(r['loss'])
    # Plateau cannot stop at arming, and a new effective improvement clears patience.
    controller=Plateau(cfg['plateau'],20000)
    results=[controller.completed_cycle(i*20000,x) for i,x in enumerate([1.,.8,.7,.699,.698,.697,.696],1)]
    assert results[4]['started_evaluation'] and results[5]['stop_armed_cycle']==6 and not results[5]['stop'] and results[6]['stop']
    other=Plateau(cfg['plateau'],20000)
    for i,x in enumerate([1.,.8,.7,.699,.698,.697,.68],1):last=other.completed_cycle(i*20000,x)
    assert not last['stop'] and other.stop_armed_cycle is None and other.b_count==0
    record=dict(status='passed',augmentation_cases=8,checkpoint_resume=dict(test='synthetic boundary wiring only; not trained20k',exact_weights=True,max_absolute_weight_difference=maximum,exact_indices_rotations_flips=True,next_lr=second['lr']),
        fixed_validation_repeat_values=[v1,v2],validation_repeat_difference=abs(v1-v2),plateau_test='arming needs extra complete cycle; effective improvement resets',
        batch=cfg['batch'],images_per_second=cfg['batch']*len(timings)/sum(timings),median_step_seconds=float(np.median(timings)),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        gpu=torch.cuda.get_device_name(),loss_min=min(losses),loss_max=max(losses),diagnostic_weights='discarded; main run starts random with fresh optimizer and RNG')
    save(RUN/'checks.json',record);print(record)
if __name__=='__main__':main()
