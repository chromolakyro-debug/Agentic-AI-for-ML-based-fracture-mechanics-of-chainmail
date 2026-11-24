"""Generate exactly the frozen panels, or sample user-supplied (s,a)/(Ex,Ey)."""
import argparse
import hashlib
import time
import numpy as np
import torch
from common import OUT, read_json, save_json, sha_file, to_sa
from model import ConditionalUNet, sample


def load_model(checkpoint):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    ckpt=torch.load(checkpoint,map_location='cpu')
    model=ConditionalUNet().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt['ema']);model.eval()
    return model,ckpt


def generate(args):
    protocol=read_json(OUT/'protocol.json')
    assert sha_file(OUT/'protocol.json')==read_json(OUT/'frozen_protocol_sha256.json')['sha256']
    checkpoint=OUT/'checkpoints'/args.checkpoint
    model,ckpt=load_model(checkpoint)
    mean=np.array(protocol['training_statistics']['mean']);std=np.array(protocol['training_statistics']['std'])
    if args.panel:
        if args.panel=='final':
            assert (OUT/'training_summary.json').exists(), 'Final panel requires completed checkpoint selection'
            assert args.checkpoint=='best.pt'
        targets=protocol[args.panel+'_targets']
        seeds=protocol['final_noise_seeds' if args.panel=='final' else 'dev_noise_seeds']
        folder=OUT/(args.panel if args.panel=='final' else f'development_{ckpt["step"]:06d}')
    else:
        if args.ex is not None and args.ey is not None: cond=to_sa(args.ex,args.ey)
        elif args.s is not None and args.a is not None: cond=[args.s,args.a]
        else: raise ValueError('Provide --panel, or both --ex --ey, or both --s --a')
        targets=[dict(target_id='custom',s=float(cond[0]),a=float(cond[1]))]
        seeds=list(range(args.seed,args.seed+args.count));folder=OUT/args.output
    folder.mkdir(parents=True,exist_ok=True)
    ckpt_hash=sha_file(checkpoint)
    size=len(targets)*len(seeds)
    # Each target chunk is committed atomically; restart never resamples a finished chunk.
    samples=[];arrays=[];raw_diagnostics=[];started=time.perf_counter()
    for target in targets:
        chunk=folder/(target['target_id']+'.npz')
        condition=(np.array([target['s'],target['a']])-mean)/std
        if chunk.exists():
            archive=np.load(chunk,allow_pickle=False)
            assert archive['checkpoint_sha256'].item()==ckpt_hash
            assert np.array_equal(archive['seeds'],seeds)
            assert np.allclose(archive['condition'],condition,rtol=0,atol=0)
            binary=archive['binary'];raw_stats=archive['raw_stats']
        else:
            binary_batches=[];stats=[]
            for offset in range(0,len(seeds),args.batch):
                continuous=sample(model,condition,seeds[offset:offset+args.batch],steps=protocol['ode_steps'])
                assert torch.isfinite(continuous).all()
                binary_batches.append((continuous>=protocol['threshold']).cpu().numpy().astype(np.uint8))
                # No gray image is used for mechanics. These diagnostics quantify threshold sensitivity only.
                stats.append(torch.stack([(continuous.abs()<.1).float().mean((1,2)),
                           (continuous.abs()<.25).float().mean((1,2)),continuous.mean((1,2)),
                           (continuous.abs()-1).abs().mean((1,2))],dim=1).cpu().numpy())
            binary=np.concatenate(binary_batches);raw_stats=np.concatenate(stats)
            with chunk.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,binary=binary,raw_stats=raw_stats,seeds=seeds,condition=condition,checkpoint_sha256=ckpt_hash)
            chunk.with_suffix('.tmp').replace(chunk)
        for j,(seed,image,stats) in enumerate(zip(seeds,binary,raw_stats)):
            samples.append(dict(index=len(samples),key=f'{target["target_id"]}_seed{seed}',target_id=target['target_id'],
                    seed=int(seed),s=target['s'],a=target['a'],sha256_pixels=hashlib.sha256(image.tobytes()).hexdigest(),
                    hard_fraction=float(np.mean(image==0)),fraction_near_threshold_0p1=float(stats[0]),
                    fraction_near_threshold_0p25=float(stats[1]),continuous_mean=float(stats[2]),
                    distance_to_binary_mean=float(stats[3])))
        arrays.append(binary)
        print(f'Generated {len(samples)}/{size} ({target["target_id"]}); wall {time.perf_counter()-started:.1f}s',flush=True)
    np.save(folder/'binary.npy',np.concatenate(arrays))
    save_json(folder/'manifest.json',dict(checkpoint_sha256=ckpt_hash,checkpoint_step=ckpt['step'],
              protocol_sha256=sha_file(OUT/'protocol.json'),targets=targets,seeds=seeds,
              ode_steps=protocol['ode_steps'],ode_solver=protocol['ode_solver'],binary_threshold=protocol['threshold'],
              samples=samples,invocation_wall_seconds=time.perf_counter()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--panel',choices=['development','final'])
    p.add_argument('--checkpoint',default='best.pt')
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--s',type=float);p.add_argument('--a',type=float)
    p.add_argument('--ex',type=float);p.add_argument('--ey',type=float)
    p.add_argument('--seed',type=int,default=830000);p.add_argument('--count',type=int,default=4)
    p.add_argument('--output',default='custom')
    generate(p.parse_args())
