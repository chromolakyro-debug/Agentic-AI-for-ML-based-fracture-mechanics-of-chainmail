"""Verify each reviewed image hash before binding AI reviews to a full pool."""
from pathlib import Path
import argparse
import json
import candidate_generation as gen
from release_workflow import load_pool,pool_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--count",type=int,default=1600)
    args=parser.parse_args()
    pool=load_pool(args.work_dir/"v6-full-1142114-b4")
    if len(pool)!=args.count:
        raise ValueError(f"Expected complete pool of{args.count}, found{len(pool)}")
    by_id={s.spec.candidate:s for s in pool}
    expected={s.spec.candidate for s in pool if not gen.rejection(s)}
    reviewed=set();excluded={};notes=[]
    for name in ("early_visual_review.json","middle_visual_review.json","late_visual_review.json"):
        manifest=json.loads((args.work_dir/name).read_text(encoding="utf-8"))
        if not manifest.get("complete",False):
            raise ValueError(f"Review not completed: {name}")
        for candidate in manifest["reviewed_candidates"]:
            if candidate not in expected:
                raise ValueError(f"Unexpected reviewed candidate {candidate}")
            digest=gen.fingerprint(by_id[candidate].pixels)
            if manifest["reviewed_pixels_sha256"].get(str(candidate))!=digest:
                raise ValueError(f"Review pixel hash mismatch: {candidate}")
            reviewed.add(candidate)
        for candidate,reason in manifest["visual_exclusion_reasons"].items():
            if int(candidate) not in manifest["reviewed_candidates"]:
                raise ValueError("Exclusion without review")
            excluded[str(candidate)]=reason
        recorded_notes=manifest.get("notes",[])
        notes.extend([recorded_notes] if isinstance(recorded_notes,str) else recorded_notes)
    if reviewed!=expected:
        raise ValueError(f"Visual-review coverage mismatch, missing={sorted(expected-reviewed)}")
    review=dict(pool_sha256=pool_hash(pool),reviewed_candidates=sorted(reviewed),excluded=excluded,
                method="AI visual inspection of actual224x224 binary candidates on contact sheets, with enlarged ambiguous cases",
                human_inspection=False,notes=notes,complete=True)
    path=args.work_dir/"full_visual_review.json"
    path.write_text(json.dumps(review,indent=2),encoding="utf-8")
    print(json.dumps(dict(candidates=len(pool),reviewed=len(reviewed),excluded=len(excluded),retained=len(reviewed)-len(excluded),manifest=str(path))))


if __name__=="__main__":
    main()
