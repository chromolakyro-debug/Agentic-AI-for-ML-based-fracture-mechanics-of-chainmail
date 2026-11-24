"""Label completed candidate batches while the independent GPU producer runs."""
from pathlib import Path
import argparse
import time
import candidate_generation as gen
from evaluate_mechanics import label_items


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--count",type=int,default=1600)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    directory=args.work_dir/"v6-full-1142114-b4"
    seen=set();samples=[];labelled=0
    while len(samples)<args.count:
        paths=[p for p in sorted(directory.glob("batch_*.npz")) if p.name not in seen]
        for path in paths:
            samples.extend(gen.read_batch(path));seen.add(path.name)
        valid=[s for s in samples if not gen.rejection(s)]
        if len(valid)-labelled>=32 or len(samples)>=args.count:
            label_items([(str(s.spec.candidate),s.pixels) for s in valid],
                        args.work_dir/"full_labels.jsonl",args.workers)
            labelled=len(valid)
            print(f"Producer {len(samples)}/{args.count}; FEM-labelled valid maps {labelled}",flush=True)
        if len(samples)<args.count:
            time.sleep(15)
    print("All produced valid candidates are FEM labelled",flush=True)


if __name__=="__main__":
    main()
