"""Honor an explicit user stop at the next atomic full-state save, without losing that state."""
import paths
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
import torch
from common import OUT, read_json, save_json, sha_file


def main(step, pid):
    request=read_json(OUT/'user_stop_request.json')
    assert request['stop_at_next_checkpoint']==step
    cp=OUT/'checkpoints';last=cp/'last.pt';previous=last.stat().st_mtime_ns
    started=time.perf_counter()
    while True:
        if last.stat().st_mtime_ns!=previous:
            state=torch.load(last,map_location='cpu')
            if state['step']>=step:break
            previous=last.stat().st_mtime_ns
        assert time.perf_counter()-started<1800, 'Checkpoint stop wait exceeded 30 minutes'
        time.sleep(2)
    assert state['step']==step
    assert sorted(set(float(v['step']) for v in state['optimizer']['state'].values()))==[float(step)]
    # Validate this task-owned trainer before stopping it; do not touch any other Python job.
    command=(f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}'; "
             "if (-not $p -or $p.CommandLine -notlike '*cfm_continuation*training.py*train*--until*140000*') "
             "{ throw 'Trainer process identity mismatch' }; "
             f"Stop-Process -Id {pid} -ErrorAction Stop")
    subprocess.run(['powershell','-NoProfile','-Command',command],check=True)
    rows=[json.loads(line) for line in (OUT/'training.jsonl').read_text().splitlines() if line.strip()]
    validation=next(r for r in reversed(rows) if r['step']==step and 'validation_velocity_mse' in r)
    assert max(r['step'] for r in rows)==step, 'Unexpected logged updates after selected checkpoint'
    assert state['images_seen']==320000+(step-20000)*state['config']['batch']
    # The authoritative retained state is exactly the atomic full checkpoint. Any few
    # in-flight, uncheckpointed updates between save and process termination are discarded.
    milestone=dict(ema=state['ema'],step=step,validation_loss=validation['validation_velocity_mse'],config=state['config'])
    torch.save(milestone,cp/f'milestone_{step:06d}.pt')
    progress=dict(step=step,images_seen=state['images_seen'],equivalent_epochs=state['images_seen']/6400,
        continuation_elapsed_seconds=state['continuation_elapsed_seconds'],best_step=state['best_step'],
        best_validation_loss=state['best_loss'],peak_allocated_gib=max(r.get('peak_allocated_gib',0) for r in rows),
        peak_reserved_gib=max(r.get('peak_reserved_gib',0) for r in rows),segment_exit_reason='user_requested_stop_at_complete_checkpoint',
        last_lr=state['optimizer']['param_groups'][0]['lr'],lr_restart=state.get('lr_restart'))
    save_json(OUT/'progress.json',progress)
    save_json(OUT/'controlled_stop.json',dict(request=request,step=step,stopped_pid=pid,
        stopped_utc=datetime.now(timezone.utc).isoformat(),last_checkpoint_sha256=sha_file(last),
        waiting_seconds=time.perf_counter()-started,full_optimizer_ema_rng_state_preserved=True,
        note='Task-owned trainer terminated immediately after its atomic 125k full-state save. Parent workflow nonzero exit is expected and retained. A few in-flight updates after that save may have been discarded; retained optimizer step and all reported training samples are 125000. Checkpoint elapsed omits at most a few seconds of final serialization/termination overhead. Original frozen plateau rule is not rewritten.'))
    print(json.dumps(progress,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',required=True,type=int);p.add_argument('--pid',required=True,type=int)
    a=p.parse_args();main(a.step,a.pid)
