"""Focused checks of material convention, condition conversion, ODE integration and split isolation."""
import numpy as np
import torch
from common import OUT, DATA, read_json, save_json, sha_file, to_sa, from_sa
from model import sample


def main():
    d=np.load(OUT/'data.npz');p=read_json(OUT/'protocol.json')
    sets=[set(d[k].tolist()) for k in ('train','validation','test')]
    assert [len(s) for s in sets]==[6400,800,800]
    assert not (sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
    assert set.union(*sets)==set(range(8000))
    assert np.allclose(d['mean'],d['sa'][d['train']].mean(0),rtol=0,atol=1e-14)
    assert np.allclose(d['std'],d['sa'][d['train']].std(0),rtol=0,atol=1e-14)
    roundtrip=float(np.max(np.abs(from_sa(to_sa(d['moduli'][:,0],d['moduli'][:,1]))-d['moduli'])))
    assert roundtrip<1e-12
    assert sha_file(DATA/'dataset.npy')==p['data_sha256']
    assert sha_file(DATA/'metadata.csv')==p['metadata_sha256']
    assert sha_file(OUT/'protocol.json')==read_json(OUT/'frozen_protocol_sha256.json')['sha256']
    # Test the actual midpoint sampling implementation against an exactly integrable constant flow.
    seeds=[112,113]
    noise=torch.stack([torch.randn((1,224,224),generator=torch.Generator(device='cuda').manual_seed(s),device='cuda') for s in seeds])
    images=np.load(DATA/'dataset.npy',mmap_mode='r')[d['train'][:2]]
    goal=torch.as_tensor(images,device='cuda',dtype=torch.float32)[:,None]*2-1
    class ConstantFlow(torch.nn.Module):
        def forward(self,x,t,c): return goal-noise
    generated=sample(ConstantFlow(),[0.,0.],seeds,steps=64)
    error=float((generated-goal[:,0]).abs().max())
    assert error<2e-5
    assert np.array_equal((generated>=0).cpu().numpy().astype(np.uint8),images)
    # Full binary meaning is preserved: soft=1 -> +1, hard=0 -> -1.
    assert (goal[torch.as_tensor(images[:,None]==0,device='cuda')]==-1).all()
    assert (goal[torch.as_tensor(images[:,None]==1,device='cuda')]==1).all()
    save_json(OUT/'core_checks.json',dict(split_disjoint=True,training_statistics_only=True,
              source_hashes_unchanged=True,protocol_unchanged=True,modulus_roundtrip_max_error=roundtrip,
              midpoint_constant_flow_max_error=error,binary_convention_pass=True))
    print('Core checks passed',flush=True)


if __name__=='__main__':main()
