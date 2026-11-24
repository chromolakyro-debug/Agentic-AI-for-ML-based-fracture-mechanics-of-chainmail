"""Optional, logged, at-most-once mild LR restart; never resets weights or moments."""
import paths
import argparse
import shutil
from datetime import datetime,timezone
import torch
from common import OUT,read_json,save_json,sha_file
from train import atomic_torch_save
from status import plateau
from training import learning_rate


def main(reason):
    last=OUT/'checkpoints'/'last.pt';ckpt=torch.load(last,map_location='cpu');step=ckpt['step']
    review=read_json(OUT/f'development_{step:06d}'/'visual_review.json')
    assert ckpt.get('lr_restart') is None, 'At most one additional restart'
    assert plateau(step)['numerical_plateau'] and learning_rate(step)<=5e-5
    assert step<200000 and ckpt['continuation_elapsed_seconds']<28800
    before=sha_file(last);shutil.copy2(last,last.with_name(f'pre_restart_{step:06d}.pt'))
    event=dict(step=step,from_lr=learning_rate(step),peak_lr=8e-5,warmup_steps=2000,floor_lr=3e-5,
        reason=reason,recorded_utc=datetime.now(timezone.utc).isoformat(),checkpoint_before_sha256=before,
        visual_review_sha256=sha_file(OUT/f'development_{step:06d}'/'visual_review.json'),
        policy='Keep all model, EMA, AdamW, RNG and counters. Change schedule metadata only.')
    ckpt['lr_restart']=event;atomic_torch_save(ckpt,last)
    save_json(OUT/'additional_lr_restart.json',event|{'checkpoint_after_sha256':sha_file(last)})
    print(event)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reason',required=True);main(p.parse_args().reason)
