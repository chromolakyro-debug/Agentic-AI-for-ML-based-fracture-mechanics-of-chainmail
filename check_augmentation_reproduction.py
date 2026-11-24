"""Regenerate one full 4-source batch and check retained targeted variants.

Run only after stage selection, with no other GPU generation process active.
The batch must contain at least one selected source. Nothing in the candidate
cache or release is changed; the report is written to scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import candidate_generation as base
import targeted_candidates as targeted


def key(sample, stage):
    return f"{stage}:{sample.spec.seed}:{sample.diagnostics.get('variant_id', 0)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=targeted.MASTER_SEED)
    parser.add_argument("--dry-run", action="store_true", help="Check selection/cache inputs without loading GPU models")
    args = parser.parse_args()
    if args.batch_start < 0 or args.batch_start % 4:
        parser.error("batch-start must be a nonnegative multiple of 4")
    cache = args.work_dir / f"targeted-v{targeted.VERSION}-{args.stage}-{args.seed}-b4"
    path = cache / f"batch_{args.batch_start:05d}_4.npz"
    if not path.is_file():
        raise FileNotFoundError(f"A complete original 4-source batch is required: {path}")
    original = base.read_batch(path)
    expected = targeted.specs_for(args.stage, args.batch_start+4, args.seed)[args.batch_start:args.batch_start+4]
    if len(original) != 4 or [s.spec for s in original] != expected:
        raise ValueError("Source cache no longer agrees with the exact original specifications.")
    manifest = json.loads((args.work_dir/f"{args.stage}_selection.json").read_text(encoding="utf-8"))
    expected_hash = dict(zip(manifest["selected_keys"], manifest["selected_hashes"]))
    selected_seeds = {int(k.split(":")[1]) for k in expected_hash}
    retained_seeds = {s.spec.seed for s in original} & selected_seeds
    if not retained_seeds:
        raise ValueError("This batch has no selected source. Choose another --batch-start; no GPU work was started.")
    variant_cache = base.read_batch(args.work_dir/f"{args.stage}_variants.npz")
    selected_reference = {key(s,args.stage):s for s in variant_cache
                          if s.spec.seed in retained_seeds and key(s,args.stage) in expected_hash}
    if len(selected_reference) != len(retained_seeds):
        raise ValueError("Selected variant cache is missing, duplicated, or disagrees with the selection manifest.")
    for name,sample in selected_reference.items():
        if base.fingerprint(sample.pixels) != expected_hash[name]:
            raise ValueError(f"Selected reference hash mismatch: {name}")
    if args.dry_run:
        print(json.dumps(dict(stage=args.stage,batch=path.name,sources=4,
                              selected_variants=len(selected_reference),input_checks="pass")),flush=True)
        return
    models = targeted.LocalModels()
    regenerated = models.generate([s.spec for s in original])
    records = []
    selected_checks = []
    for old,new in zip(original,regenerated):
        records.append(dict(source_id=int(old.spec.candidate), seed=int(old.spec.seed),
                            retained=old.spec.seed in retained_seeds,
                            source_pixels_exact=bool(np.array_equal(old.pixels,new.pixels)),
                            source_mask_exact=bool(np.array_equal(old.source_mask,new.source_mask)),
                            original_pixels_sha256=base.fingerprint(old.pixels),
                            regenerated_pixels_sha256=base.fingerprint(new.pixels)))
        for variant in targeted.variants_for_sample(new,count=3):
            name = key(variant,args.stage)
            if name in selected_reference:
                reference = selected_reference[name]
                selected_checks.append(dict(key=name,source_id=int(variant.diagnostics["source_id"]),
                                            variant_id=int(variant.diagnostics["variant_id"]),
                                            pixels_exact=bool(np.array_equal(reference.pixels,variant.pixels)),
                                            expected_sha256=expected_hash[name],
                                            regenerated_sha256=base.fingerprint(variant.pixels)))
    source_exact = all(r["source_pixels_exact"] and r["source_mask_exact"] for r in records)
    selected_exact = len(selected_checks) == len(selected_reference) and all(r["pixels_exact"] for r in selected_checks)
    report = dict(stage=args.stage,batch=path.name,original_batch_size=4,
                  master_seed=args.seed,checked_sources=4,checked_selected_variants=len(selected_checks),
                  all_sources_exact=source_exact,all_selected_variants_exact=selected_exact,
                  all_exact=source_exact and selected_exact,source_records=records,selected_records=selected_checks,
                  source_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  model_source=base.SOURCE,gpu=models.torch.cuda.get_device_name(0),
                  torch_version=str(models.torch.__version__),
                  limitation="One sampled original batch on the same workstation; not a cross-platform determinism guarantee.")
    (args.work_dir/"reproduction_check.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("stage","batch","checked_sources","checked_selected_variants","all_exact")}),flush=True)
    if not report["all_exact"]:
        raise RuntimeError("Regenerated source pixels/masks or retained variant pixels differed from the reference.")


if __name__ == "__main__":
    main()
