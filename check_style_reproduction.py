"""Same-workstation style-batch regeneration and exact pixel-block validation.

Recreates four independently seeded source images as one original batch and all
their framing candidates. Never changes source caches or dataset manifests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import candidate_generation as base
import style_candidates as styles


def sample_key(sample):
    return f"styles:{sample.spec.seed}:{sample.diagnostics['variant_id']}"


def selected_keys(work_dir):
    path=work_dir/"full_selection.json"
    if not path.exists():return set()
    record=json.loads(path.read_text(encoding="utf-8"))
    return {k for k in record.get("selected_keys",[]) if k.startswith("styles:")}


def block_check(sample):
    grid=int(sample.diagnostics.get("pixel_art_grid",0))
    if not grid:return None
    block=224//grid
    pixels=sample.pixels
    expected=np.repeat(np.repeat(pixels[::block,::block],block,axis=0),block,axis=1)
    return bool(grid in (56,112) and pixels.shape==(224,224) and np.array_equal(pixels,expected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=styles.MASTER_SEED)
    parser.add_argument("--batch-start",type=int,default=None,
                        help="Default selects a complete batch covering all three style families when possible")
    args=parser.parse_args()
    cache=args.work_dir/f"styles-v{styles.VERSION}-styles-{args.seed}-b4"
    variants=base.read_batch(args.work_dir/"styles_variants.npz")
    retained=selected_keys(args.work_dir)
    retained_seeds={int(k.split(":")[1]) for k in retained}
    files=sorted(cache.glob("batch_*_4.npz"))
    if args.batch_start is not None:
        files=[cache/f"batch_{args.batch_start:05d}_4.npz"]
    eligible=[]
    for path in files:
        sources=base.read_batch(path)
        family_count=len({s.diagnostics['style_family'] for s in sources})
        retained_count=sum(s.spec.seed in retained_seeds for s in sources)
        eligible.append((family_count,retained_count,path,sources))
    if not eligible:raise ValueError("No complete four-source style batch is available")
    # Prefer diversity in the check, then coverage of currently selected sources.
    _,_,path,original=max(eligible,key=lambda r:(r[0],r[1],-int(r[2].stem.split('_')[1])))
    start=int(path.stem.split('_')[1])
    if [s.spec for s in original]!=styles.specs_for("styles",start+4,args.seed)[start:start+4]:
        raise ValueError("Cached style specs do not match current source generator")
    checked_grid=[s for s in variants if s.diagnostics.get("pixel_art_grid",0)]
    if not checked_grid or not all(block_check(s) for s in checked_grid):
        raise ValueError("Actual cached pixel-art variants violate their constant-block grid contract")
    source_seeds={s.spec.seed for s in original}
    reference={sample_key(s):s for s in variants if s.spec.seed in source_seeds}
    print(f"Regenerating {path.name}: families={sorted({s.diagnostics['style_family'] for s in original})}, "
          f"selected sources={len(source_seeds & retained_seeds)}, all cached grid checks={len(checked_grid)}",flush=True)
    model=styles.LocalModels()
    regenerated=model.generate([s.spec for s in original])
    source_records=[];variant_records=[]
    for old,new in zip(original,regenerated):
        source_records.append(dict(seed=int(old.spec.seed),source_id=int(old.spec.candidate),
                                   style_family=old.diagnostics['style_family'],
                                   source_pixels_exact=bool(np.array_equal(old.pixels,new.pixels)),
                                   source_mask_exact=bool(np.array_equal(old.source_mask,new.source_mask)),
                                   source_sha256=base.fingerprint(old.pixels),
                                   regenerated_source_sha256=base.fingerprint(new.pixels)))
        for variant in styles.variants_for_sample(new,count=3):
            key=sample_key(variant)
            if key not in reference:raise ValueError(f"Regeneration produced an unexpected framing candidate: {key}")
            original_variant=reference[key]
            variant_records.append(dict(key=key,seed=int(variant.spec.seed),
                                        pixels_exact=bool(np.array_equal(original_variant.pixels,variant.pixels)),
                                        original_sha256=base.fingerprint(original_variant.pixels),
                                        regenerated_sha256=base.fingerprint(variant.pixels),
                                        pixel_grid=variant.diagnostics.get("pixel_art_grid",0),
                                        pixel_grid_pass=block_check(variant)))
    retained=selected_keys(args.work_dir)
    for record in variant_records:record['selected_at_check']=record['key'] in retained
    all_exact=(all(r['source_pixels_exact'] and r['source_mask_exact'] for r in source_records)
               and len(variant_records)==len(reference) and all(r['pixels_exact'] for r in variant_records))
    report=dict(batch=path.name,checked_sources=4,source_style_families=sorted({s.diagnostics['style_family'] for s in original}),
                checked_variants=len(variant_records),selected_variants_at_check=sum(r['selected_at_check'] for r in variant_records),
                all_exact=all_exact,all_cached_pixel_grid_checks_passed=True,
                cached_pixel_grid_variants_checked=len(checked_grid),
                cached_pixel_grid_counts=dict(Counter(int(s.diagnostics['pixel_art_grid']) for s in checked_grid)),
                source_records=source_records,variant_records=variant_records,
                limitation="One original four-source batch on the same workstation. Selection may be revised after visual review; all variants in this batch were checked.")
    (args.work_dir/"style_reproduction_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ('batch','checked_sources','checked_variants','selected_variants_at_check',
                                          'all_exact','cached_pixel_grid_variants_checked')}),flush=True)
    if not all_exact:raise RuntimeError("Style source or candidate pixel regeneration failed")


if __name__=="__main__":main()
