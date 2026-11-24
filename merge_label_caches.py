"""Finish the current-protocol cache after both labelling processes have exited."""
from pathlib import Path
import argparse
import json
import os
import candidate_generation as gen
from evaluate_mechanics import mechanics_fingerprint
from release_workflow import load_pool


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,required=True)
    args=parser.parse_args()
    protocol=mechanics_fingerprint()
    cache={}
    for name in ("full_labels.jsonl","protocol_refreshed_labels.jsonl"):
        for line in (args.work_dir/name).read_text().splitlines():
            row=json.loads(line)
            if row.get("protocol_fingerprint")==protocol:
                cache[(row["key"],row["sha256_pixels"])]=row
    pool=load_pool(args.work_dir/"v6-full-1142114-b4")
    required=[s for s in pool if not gen.rejection(s)]
    missing=[s.spec.candidate for s in required if (str(s.spec.candidate),gen.fingerprint(s.pixels)) not in cache]
    if missing:
        raise ValueError(f"Current-protocol FEM labels missing: {missing}")
    target=args.work_dir/"full_labels.jsonl"
    temporary=target.with_suffix(".merged.jsonl")
    with temporary.open("w",encoding="utf-8") as stream:
        for row in sorted(cache.values(),key=lambda r:int(r["key"])):
            stream.write(json.dumps(row)+"\n")
    os.replace(temporary,target)
    print(json.dumps(dict(current_protocol=protocol,required=len(required),cached=len(cache),missing=0)))


if __name__=="__main__":
    main()
