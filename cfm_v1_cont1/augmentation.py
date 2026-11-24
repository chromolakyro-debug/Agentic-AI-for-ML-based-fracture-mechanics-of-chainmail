"""Exact CCW rot90 then horizontal flip; labels transform in physical units first."""
import torch
def apply(binary,moduli,rotations,flips):
    output=torch.empty_like(binary)
    for k in range(4):
        for mirror in [False,True]:
            mask=(rotations==k)&(flips==mirror)
            if not mask.any():continue
            pixels=torch.rot90(binary[mask],k,dims=(-2,-1))
            if mirror:pixels=torch.flip(pixels,dims=(-1,))
            output[mask]=pixels
    mapped=moduli.clone();odd=rotations%2==1;mapped[odd]=mapped[odd].flip(-1)
    return output.contiguous(),mapped
def conditions(moduli,mean,std):
    values=torch.stack([.5*torch.log(moduli).sum(-1),torch.log(moduli[:,0]/moduli[:,1])],dim=-1)
    return ((values-mean)/std).float()
