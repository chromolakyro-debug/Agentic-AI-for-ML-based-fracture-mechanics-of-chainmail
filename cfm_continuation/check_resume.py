"""Compare uninterrupted next update with full-state save/reload next update."""
import paths
import gc
import hashlib
import random
import numpy as np
import torch
from common import OUT, save_json, read_json
from training import load_state, update, learning_rate
from train import atomic_torch_save


def digest(model, ema, opt):
    h=hashlib.sha256()
    for net in (model,ema):
        for k,v in net.state_dict().items(): h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    for state in opt.state.values():
        for k,v in state.items(): h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    h.update(torch.get_rng_state().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all():h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    batch=read_json(OUT/'continuation_config.json')['batch']
    assert learning_rate(20000)==2e-5
    assert abs(learning_rate(22000)-1e-4)<1e-14 and abs(learning_rate(200000)-3e-5)<1e-14
    model,ema,opt,data,meta=load_state(paths.ORIGINAL/'checkpoints'/'last.pt')
    assert meta['step']==20000
    assert all(float(v['step'])==20000 for v in opt.state.values())
    update(model,ema,opt,data,batch,20001,learning_rate(20001))
    temporary=OUT/'checkpoints'/'resume_smoke_temporary.pt'
    atomic_torch_save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate(),step=20001),temporary)
    loss1,grad1=update(model,ema,opt,data,batch,20002,learning_rate(20002));hash1=digest(model,ema,opt)
    del model,ema,opt,data;gc.collect();torch.cuda.empty_cache()
    model,ema,opt,data,meta=load_state(temporary)
    loss2,grad2=update(model,ema,opt,data,batch,20002,learning_rate(20002));hash2=digest(model,ema,opt)
    assert hash1==hash2 and loss1==loss2 and grad1==grad2,(hash1,hash2,loss1,loss2)
    save_json(OUT/'resume_check.json',dict(full_model_ema_optimizer_rng_next_update_exact=True,sha256_uninterrupted=hash1,
        sha256_resumed=hash2,loss_uninterrupted=loss1,loss_resumed=loss2,optimizer_source_step=20000,batch=batch,lr_endpoint_checks=True))
    temporary.unlink()
    print('Continuation full-state restore and schedule endpoint checks passed',flush=True)

if __name__=='__main__':main()
