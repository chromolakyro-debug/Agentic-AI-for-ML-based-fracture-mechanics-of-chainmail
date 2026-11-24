"""Retain full-state checkpoints without changing the active training process."""
import argparse
import datetime
import json
import shutil
import time

import torch

from common import RUN, read, save, sha


def retain_once(policy):
    folder = RUN / 'checkpoints'
    source = folder / 'last.pt'
    if not source.exists():
        return None
    # last is atomically replaced every 1,000 steps. Read metadata only when
    # a 10,000-step boundary can have been reached, without touching CUDA/RNG.
    progress = read(RUN / 'progress.json')
    boundary = progress['step'] // policy['interval_steps'] * policy['interval_steps']
    destination = folder / f'step_{boundary:06d}.pt'
    if boundary < policy['first_step'] or destination.exists():
        return None
    temporary = folder / 'retention_copy.tmp'
    before = sha(source)
    shutil.copyfile(source, temporary)
    digest = sha(temporary)
    if digest != before or sha(source) != before:
        return None  # Atomic last replacement raced the read; retry next poll.
    state = torch.load(temporary, map_location='cpu', weights_only=False)
    step = state['step']
    if step < policy['first_step'] or step % policy['interval_steps']:
        return None
    required = ['model', 'ema', 'optimizer', 'scheduler', 'plateau', 'rng', 'config',
                'training_seconds', 'samples_seen', 'best_step', 'best_val_mse']
    assert all(key in state for key in required)
    assert state['scheduler']['updates'] == step
    destination = folder / f'step_{step:06d}.pt'
    if destination.exists():
        assert sha(destination) == digest
    else:
        temporary.replace(destination)
    row = dict(step=step, path=destination.relative_to(RUN).as_posix(),
               sha256=digest, bytes=destination.stat().st_size,
               state='full_training_state', source='atomic last.pt byte copy',
               utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    manifest = RUN / 'retained_checkpoints.json'
    records = read(manifest) if manifest.exists() else []
    if not any(r['step'] == step for r in records):
        records.append(row)
        save(manifest, records)
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    policy = read(RUN / 'checkpoint_retention_policy.json')
    while True:
        retain_once(policy)
        if args.once or (RUN / 'training_summary.json').exists():
            # Terminal summary is written after the final atomic checkpoint.
            retain_once(policy)
            break
        time.sleep(policy['poll_seconds'])


if __name__ == '__main__':
    main()
