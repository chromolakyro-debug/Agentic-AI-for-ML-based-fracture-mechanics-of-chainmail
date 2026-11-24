"""Regenerate original full batches and compare binary pixels exactly."""
from pathlib import Path
import argparse
import json
import numpy as np
import candidate_generation as gen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--batches",type=int,default=3)
    args=parser.parse_args()
    release_path=args.work_dir/"release_manifest.json"
    release=json.loads(release_path.read_text()) if release_path.exists() else None
    selected=set(release["selected_candidates"]) if release else None
    files=sorted((args.work_dir/"v6-full-1142114-b4").glob("batch_*.npz"))
    eligible=[path for path in files if selected is None or any(s.spec.candidate in selected for s in gen.read_batch(path))]
    rng=np.random.default_rng(1142114+2026)
    chosen=[eligible[i] for i in sorted(rng.choice(len(eligible),args.batches,replace=False))]
    model=gen.LocalModels()
    results=[]
    for path in chosen:
        original=gen.read_batch(path)
        regenerated=model.generate([s.spec for s in original])
        for old,new in zip(original,regenerated):
            record=dict(candidate_id=old.spec.candidate,retained=None if selected is None else old.spec.candidate in selected,
                        seed=old.spec.seed,pixel_exact=bool(np.array_equal(old.pixels,new.pixels)),
                        source_mask_exact=bool(np.array_equal(old.source_mask,new.source_mask)),
                        reference_sha256=gen.fingerprint(old.pixels),regenerated_sha256=gen.fingerprint(new.pixels))
            results.append(record)
        print(f"Reproduced {path.name}: pixel exact={all(r['pixel_exact'] for r in results)}",flush=True)
    report=dict(all_pixels_exact=all(r["pixel_exact"] for r in results),checked_candidates=len(results),
                retained_images=None if selected is None else sum(r["retained"] for r in results),original_batch_size=4,
                batches=[p.name for p in chosen],records=results,
                caveat="Sampled same-workstation regeneration, not a cross-platform determinism guarantee")
    (args.work_dir/"reproduction_result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({key:report[key] for key in ("all_pixels_exact","checked_candidates","retained_images")}),flush=True)
    if not report["all_pixels_exact"]:
        raise RuntimeError("Reference image regeneration did not match")


if __name__=="__main__":
    main()
