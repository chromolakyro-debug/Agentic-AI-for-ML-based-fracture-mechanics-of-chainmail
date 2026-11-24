"""Single-channel 224px conditional U-Net and Gaussian-to-image linear CFM."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.affine = nn.Linear(emb, cout * 2)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.affine(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(h))


class Attention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h*w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1))
        return x + self.proj(out.transpose(-2, -1).reshape(b, c, h, w))


class ConditionalUNet(nn.Module):
    def __init__(self, channels=(32, 64, 128, 192, 256), emb=256):
        super().__init__()
        self.register_buffer('time_freq', torch.exp(torch.linspace(0, -math.log(10000), 64)))
        self.register_buffer('cond_freq', torch.tensor([.5, 1., 2., 4.]) * math.pi)
        self.embedding = nn.Sequential(nn.Linear(128 + 18, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.input = nn.Conv2d(1, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList([nn.ModuleList([ResBlock(c, c, emb), ResBlock(c, c, emb)]) for c in channels])
        self.down = nn.ModuleList([nn.Conv2d(c, d, 3, stride=2, padding=1) for c, d in zip(channels[:-1], channels[1:])])
        self.middle = ResBlock(channels[-1], channels[-1], emb)
        self.attention = Attention(channels[-1])
        self.up = nn.ModuleList([nn.Conv2d(channels[i+1], channels[i], 3, padding=1) for i in range(len(channels)-1)])
        self.decoder = nn.ModuleList([nn.ModuleList([ResBlock(c*2, c, emb), ResBlock(c, c, emb)]) for c in channels[:-1]])
        self.output = nn.Sequential(nn.GroupNorm(8, channels[0]), nn.SiLU(), nn.Conv2d(channels[0], 1, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, t, condition):
        phase = (t[:, None].float() * 1000) * self.time_freq[None]
        cp = condition.float()[:, :, None] * self.cond_freq[None, None]
        embedding = self.embedding(torch.cat([phase.sin(), phase.cos(), condition.float(),
                                             cp.sin().flatten(1), cp.cos().flatten(1)], dim=1))
        h = self.input(x)
        skips = []
        for i, blocks in enumerate(self.encoder):
            for block in blocks: h = block(h, embedding)
            skips.append(h)
            if i < len(self.down): h = self.down[i](h)
        h = self.middle(self.attention(h), embedding)
        for i in reversed(range(len(self.up))):
            h = self.up[i](F.interpolate(h, scale_factor=2, mode='nearest'))
            h = torch.cat([h, skips[i]], dim=1)
            for block in self.decoder[i]: h = block(h, embedding)
        return self.output(h)


@torch.inference_mode()
def sample(model, standardized_condition, seeds, steps=64, device='cuda'):
    """Fixed midpoint ODE; returns continuous image, caller uses x>=0 for soft=1."""
    noise = [torch.randn((1, 224, 224), generator=torch.Generator(device=device).manual_seed(int(seed)), device=device) for seed in seeds]
    x = torch.stack(noise).contiguous(memory_format=torch.channels_last)
    c = torch.as_tensor(standardized_condition, device=device, dtype=torch.float32)
    if c.ndim == 1: c = c[None].expand(len(seeds), -1)
    dt = 1. / steps
    for i in range(steps):
        t = torch.full((len(seeds),), i * dt, device=device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            v = model(x, t, c)
            midpoint = x + .5 * dt * v.float()
            v2 = model(midpoint, t + .5 * dt, c)
        x = x + dt * v2.float()
    return x[:, 0]
