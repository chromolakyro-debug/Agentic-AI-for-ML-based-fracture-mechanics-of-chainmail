"""Logged actual subprocess commands. Milestones require subsequent visual review."""
import paths
import argparse
from datetime import datetime,timezone
import json
import os
import shutil
import subprocess
import sys
import time
from common import OUT, sha_file, read_json


def run(script,*args):
    command=[sys.executable,str(paths.ROOT/script),*map(str,args)]
    tick=time.perf_counter();record=dict(command=command,cwd=str(paths.ROOT),env={'CFM_V0_RUN_DIR':str(OUT)},started_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha_file(paths.ROOT/script))
    snapshot=OUT/'code_snapshots'/record['script_sha256']/(paths.ROOT/script).name
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():shutil.copy2(paths.ROOT/script,snapshot)
    with (OUT/'commands.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record|{'event':'start'})+'\n')
    p=subprocess.run(command,cwd=paths.ROOT,env=os.environ.copy())
    record.update(event='finish',returncode=p.returncode,elapsed_seconds=time.perf_counter()-tick,finished_utc=datetime.now(timezone.utc).isoformat())
    with (OUT/'commands.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    if p.returncode:raise RuntimeError(f'Command failed: {command}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['milestone','train','final']);p.add_argument('--step',type=int)
    a=p.parse_args()
    if a.mode=='train':run('cfm_continuation/training.py','train','--until',a.step)
    elif a.mode=='milestone':
        folder=OUT/f'development_{a.step:06d}'
        if not (folder/'manifest.json').exists():run('cfm_v0/generate.py','--panel','development','--checkpoint',f'milestone_{a.step:06d}.pt','--batch','2')
        if not (folder/'fem_summary.json').exists():run('cfm_v0/fem.py','evaluate','--folder',folder.name,'--workers','8')
        run('cfm_continuation/development.py','--step',a.step)
    else:
        selected=read_json(OUT/'checkpoint_selection.json')
        assert sha_file(OUT/'checkpoints'/'best.pt')==selected['sha256']
        assert (OUT/f'development_{selected["step"]:06d}'/'visual_review.json').exists(), 'Inspect selected checkpoint on development panel first'
        if not (OUT/'final'/'manifest.json').exists():run('cfm_v0/generate.py','--panel','final','--checkpoint','best.pt','--batch','16')
        if not (OUT/'final'/'fem_summary.json').exists():run('cfm_v0/fem.py','evaluate','--folder','final','--workers','8')
        run('cfm_v0/metrics.py','mechanics')
        if not (OUT/'final'/'image_neighbors.json').exists():run('cfm_v0/diagnostics.py','neighbors')
        run('cfm_v0/plots.py','review','--folder','final')
