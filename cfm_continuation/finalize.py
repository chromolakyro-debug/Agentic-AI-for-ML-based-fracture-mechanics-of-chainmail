"""Commit a reviewed stop and checkpoint choice before opening historical final panel."""
import paths
import argparse
import shutil
import json
from datetime import datetime,timezone
import torch
from common import OUT,read_json,save_json,sha_file
from status import plateau


def main(reason, checkpoint='best-validation', selection_reason=None):
    assert not (OUT/'training_summary.json').exists(), 'Final selection already committed'
    progress=read_json(OUT/'progress.json');step=progress['step'];cp=OUT/'checkpoints'
    review=read_json(OUT/f'development_{step:06d}'/'visual_review.json')
    result=plateau(step,True)
    if progress['segment_exit_reason']=='user_requested_stop_at_complete_checkpoint':
        request=read_json(OUT/'user_stop_request.json')
        controlled=read_json(OUT/'controlled_stop.json')
        assert controlled['step']==step==request['stop_at_next_checkpoint']
        assert controlled['last_checkpoint_sha256']==sha_file(cp/'last.pt')
        stop='user_accepted_diminishing_validation_returns'
    elif progress['segment_exit_reason']!='safety_cap':
        assert result['numerical_plateau'], 'No numerical plateau and no safety cap'
        assert not review['meaningful_visual_improvement'], 'Visual improvement still calls for continuation'
        stop='validation_plateau_two_windows_and_development_visual_review'
    else: stop='safety_cap_not_a_convergence_claim'
    best=torch.load(cp/'best.pt',map_location='cpu');last=torch.load(cp/'last.pt',map_location='cpu')
    logged=[json.loads(line) for line in (OUT/'training.jsonl').read_text().splitlines() if line.strip()]
    for key in ('peak_allocated_gib','peak_reserved_gib'):
        progress[key]=max(r.get(key,0) for r in logged)
    assert last['step']==step and best['step']==progress['best_step']
    shutil.copy2(cp/'best.pt',cp/'best-validation.pt')
    selected=best['step']
    if checkpoint=='last':
        assert selection_reason and (OUT/f'development_{best["step"]:06d}'/'visual_review.json').exists()
        # Reused v0 final sampler requires the literal name best.pt. Preserve the
        # actual validation optimum in best-validation.pt, then make this alias
        # the explicitly chosen EMA. Full last.pt remains unchanged/resumable.
        selected=step
        shutil.copy2(cp/f'milestone_{step:06d}.pt',cp/'best.pt')
    save_json(OUT/'checkpoint_selection.json',dict(selected=checkpoint+' EMA',step=selected,sha256=sha_file(cp/'best.pt'),
        last_step=step,best_validation_loss=best['validation_loss'],reason=selection_reason or 'Minimum fixed-noise validation EMA velocity MSE; selected checkpoint inspected on frozen development panel before final generation.',
        sampler_alias='best.pt is the selected EMA for the reused final sampler; best-validation.pt always retains the actual validation optimum, and last.pt retains full optimizer/RNG state.',
        committed_utc=datetime.now(timezone.utc).isoformat(),final_panel_accessed_at_commit=False))
    save_json(OUT/'training_summary.json',progress|dict(steps=step,stop_reason=stop,stop_explanation=reason,
        numerical_plateau=result['numerical_plateau'],plateau_windows=result['windows'],final_checkpoint_step=selected,
        source_training_seconds=read_json(paths.ORIGINAL/'training_summary.json')['elapsed_seconds'],
        additional_lr_restart=read_json(OUT/'additional_lr_restart.json') if (OUT/'additional_lr_restart.json').exists() else None,
        note='Continuation elapsed is active trainer wall time including validation/checkpoint work, excluding benchmark, generation, FEM and human/AI review waits. Source20k images_seen=320000; continuation uses chosen benchmark batch.'))
    if not (cp/f'milestone_{selected:06d}.pt').exists():shutil.copy2(cp/'best.pt',cp/f'milestone_{selected:06d}.pt')
    print({'stop':stop,'last_step':step,'selected_step':selected})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reason',required=True)
    p.add_argument('--checkpoint',choices=['best-validation','last'],default='best-validation')
    p.add_argument('--selection-reason')
    a=p.parse_args();main(a.reason,a.checkpoint,a.selection_reason)
