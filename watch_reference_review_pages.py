"""Publish immutable candidate review pages while the GPU writes its cache."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import time
import reference_style_review as review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--stage',default='full',choices=('pilot','full'))
    p.add_argument('--seed',type=int,default=20261114)
    p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args();args.work_dir=args.work_dir.resolve()
    while True:
        review.emit(args)
        path=args.work_dir/f'{args.stage}_generation_state.json'
        state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if state.get('complete'):
            print('All currently generated candidate review pages published',flush=True)
            break
        time.sleep(30)


if __name__=='__main__':main()
