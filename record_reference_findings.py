"""Record already-completed, pixel-bound AI review findings, never infer a verdict."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import reference_style_review as helper


def record_all(work_dir):
    work_dir=Path(work_dir)
    review_path=work_dir/'reference_visual_review.json'
    recorded=json.loads(review_path.read_text(encoding='utf-8')).get('source_review_pages',{})
    summaries=[]
    for path in sorted((work_dir/'review_findings').glob('page_*.json')):
        finding=json.loads(path.read_text(encoding='utf-8'))
        number=int(finding['page'])
        cfg=SimpleNamespace(work_dir=work_dir,stage='full',seed=20261114,batch_size=4,page=number,exclude=[])
        manifest=helper.load_manifest(cfg,number)
        if manifest['page_identity'] in recorded:
            continue
        if finding.get('page_identity')!=manifest['page_identity'] or finding.get('png_sha256')!=manifest['png_sha256']:
            raise ValueError(f'Unbound finding {path}')
        if not finding.get('reviewer') or 'AI' not in finding.get('method',''):
            raise ValueError(f'Missing explicit AI-review provenance {path}')
        if sorted(finding['inspected_cells'])!=list(range(1,manifest['count']+1)):
            raise ValueError(f'Incomplete review {path}')
        verdicts=finding['passed']+finding['excluded']
        if sorted(v['cell'] for v in verdicts)!=list(range(1,manifest['count']+1)):
            raise ValueError(f'Incomplete or duplicate verdicts {path}')
        for verdict in verdicts:
            if verdict['pixels_sha256']!=manifest['entries'][verdict['cell']-1]['pixels_sha256']:
                raise ValueError(f'Pixel hash mismatch {path}')
        cfg.exclude=[f"{v['cell']}:{v['reason']}" for v in finding['excluded']]
        helper.record(cfg)
        summaries.append(dict(page=number,passed=len(finding['passed']),excluded=len(finding['excluded'])))
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(dict(recorded_findings=record_all(args.work_dir))),flush=True)


if __name__=='__main__':main()
