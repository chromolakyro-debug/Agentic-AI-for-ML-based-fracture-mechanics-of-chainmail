"""Bounded wiring/resume tests. All diagnostic weights are discarded."""
import copy,time,tempfile
import numpy as np
import torch
from common import ROOT,RUN,HERE,read,save,sha
from augmentation import apply,conditions
from control import RestartSchedule,Plateau
from train import build,update,load_data,validate,rng_state,restore_rng,atomic
from retention import check as check_retention

def main():
    cfg=read(RUN/'config.json');d,data,val=load_data();model,ema,opt,scheduler,plateau=build(cfg,d)
    # Compare actual initialization against the unmodified v1 builder.
    import importlib.util, hashlib
    spec=importlib.util.spec_from_file_location('ablation_reference_train',ROOT/'cfm_v1/train.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    ref_model,ref_ema,ref_opt,ref_scheduler,ref_plateau=reference.build(cfg,d)
    assert all(torch.equal(v,ref_model.state_dict()[k]) for k,v in model.state_dict().items())
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    initialization_sha256=h.hexdigest()
    del ref_model,ref_ema,ref_opt,ref_scheduler,ref_plateau
    # Reconstruct the exact original (untransformed) forward inputs using the same RNG draws.
    state_before=rng_state();batch=cfg['batch']
    idx=torch.multinomial(data[4],batch,replacement=True)
    turns=torch.randint(0,4,(batch,),device='cuda');flips=torch.rand(batch,device='cuda')<.5
    assert (turns!=0).any() and flips.any()
    x=(data[0][idx].float()[:,None]*2-1).contiguous(memory_format=torch.channels_last)
    expected_c=conditions(data[1][idx],data[2],data[3])
    noise=torch.randn_like(x);expected_t=torch.rand(batch,device='cuda')
    expected_xt=(1-expected_t[:,None,None,None])*noise+expected_t[:,None,None,None]*x
    expected_rng=rng_state();restore_rng(state_before)
    captured=[]
    handle=model.register_forward_pre_hook(lambda module,args:captured.append(tuple(v.detach().clone() for v in args)))
    identity=update(model,ema,opt,scheduler,data,cfg,1);handle.remove()
    assert len(captured)==1
    assert all(torch.equal(v,w) for v,w in zip(captured[0],(expected_xt,expected_t,expected_c)))
    assert torch.equal(identity['sampled_local_indices'],idx)
    assert torch.equal(identity['rotations'],turns) and torch.equal(identity['flips'],flips)
    actual_rng=rng_state()
    assert torch.equal(expected_rng['cpu'],actual_rng['cpu'])
    assert all(torch.equal(v,w) for v,w in zip(expected_rng['cuda'],actual_rng['cuda']))
    # Exercise the original augmented update from exactly the same RNG state.
    restore_rng(state_before);reference_result=reference.update(model,ema,opt,scheduler,data,cfg,2)
    for key in ['sampled_local_indices','rotations','flips']:assert torch.equal(identity[key],reference_result[key])
    assert all(torch.equal(v,w) for v,w in zip(expected_rng['cuda'],rng_state()['cuda']))
    del captured,expected_xt,expected_c,expected_t,x,noise
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
    record=dict(status='passed',no_augmentation_forward_inputs_exact=True,physical_labels_unchanged=True,v1_initialization_exact=True,initialization_sha256=initialization_sha256,v1_rng_stream_exact=True,checkpoint_resume=dict(test='synthetic boundary wiring only; not trained20k',exact_weights=True,max_absolute_weight_difference=maximum,exact_indices_rotations_flips=True,next_lr=second['lr']),
        fixed_validation_repeat_values=[v1,v2],validation_repeat_difference=abs(v1-v2),plateau_test='arming needs extra complete cycle; effective improvement resets',
        batch=cfg['batch'],images_per_second=cfg['batch']*len(timings)/sum(timings),median_step_seconds=float(np.median(timings)),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        gpu=torch.cuda.get_device_name(),loss_min=min(losses),loss_max=max(losses),diagnostic_weights='discarded; main run starts random with fresh optimizer and RNG')
    record['checkpoint_retention']=check_retention()
    save(RUN/'checks.json',record);print(record)
if __name__=='__main__':main()
