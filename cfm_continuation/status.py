"""Read-only concise progress and frozen nonoverlapping-window plateau check."""
import paths
import argparse
import json
from common import OUT, read_json, save_json


def plateau(step,write=False):
    rows=[json.loads(line) for line in (OUT/'training.jsonl').read_text().splitlines() if line.strip()]
    validation=[r for r in rows if 'validation_velocity_mse' in r]
    def best_at(s): return min(r['validation_velocity_mse'] for r in validation if r['step']<=s)
    windows=[]
    if step>=60000:
        for start,end in ((step-40000,step-20000),(step-20000,step)):
            earlier,later=best_at(start),best_at(end)
            windows.append(dict(start=start,end=end,best_before=earlier,best_after=later,relative_improvement=(earlier-later)/earlier,
                                plateau=(earlier-later)/earlier<.005))
    result=dict(step=step,windows=windows,numerical_plateau=step>=80000 and len(windows)==2 and all(w['plateau'] for w in windows),
                requires_actual_visual_and_FEM_review=True)
    if write:save_json(OUT/f'development_{step:06d}'/'plateau_check.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int);a=p.parse_args()
    if a.step:print(json.dumps(plateau(a.step,True),indent=2))
    else:
        rows=[json.loads(line) for line in (OUT/'training.jsonl').read_text().splitlines() if line.strip()]
        print(json.dumps({'latest':rows[-1],'latest_validation':next(r for r in reversed(rows) if 'validation_velocity_mse' in r)},indent=2))
