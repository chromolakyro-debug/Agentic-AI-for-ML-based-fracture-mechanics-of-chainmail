"""Prepare one reframed material map per independently generated v5 source.

This is a contingency candidate pool, not a final dataset. CLI preparation is
explicitly gated by --prepare so the 50-image pilot can be reviewed first.
The original source masks are unavailable: the cached v5 224x224 render is the
reframing source, and inherited semantic/visual evidence is marked accordingly.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import asdict
import tempfile
from pathlib import Path

import numpy as np

import candidate_generation as current

DEFAULT_WORK = (Path(tempfile.gettempdir()) / 'binary-cat-mechanics')
DEFAULT_CACHE = (Path(tempfile.gettempdir()) / 'binary-cat-v4/v5-full-1142031-b4')
REFRAME_SEED = 1142114
REFRAME_VERSION = 1


def import_reference(path: Path):
    name = "cat_generation_v5_reference_for_reframing"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reference generation code: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def approved_sources(reference_script: Path, cache: Path):
    reference = import_reference(reference_script)
    pool = [sample for path in sorted(cache.glob("batch_*.npz")) for sample in reference.read_batch(path)]
    review = reference.RELEASE_REVIEW
    digest = hashlib.sha256("\n".join(
        f"{sample.spec.candidate}:{reference.fingerprint(sample.pixels)}" for sample in pool
    ).encode()).hexdigest()
    if (len(pool) != review["candidate_count"] or digest != review["pool_sha256"]
            or review["pipeline_version"] != 5 or review["seed"] != 1142031):
        raise RuntimeError("The source pool does not match the exact v5 reference visual review.")
    reviewed = set(review["reviewed_candidates"])
    excluded = set(review["excluded_candidates"])
    accepted = [sample for sample in pool
                if not reference.rejection(sample)
                and sample.spec.candidate in reviewed
                and sample.spec.candidate not in excluded]
    if len({sample.spec.seed for sample in accepted}) != len(accepted):
        raise RuntimeError("Approved source seeds are not unique.")
    return accepted, digest


def framing_rank(sources):
    """Stable seed-shuffled ranks of the entire approved source pool.

    Rank is formed before an optional count limit. Extending a cached prefix
    cannot change existing candidates' framing.
    """
    order = sorted(sources, key=lambda sample: hashlib.sha256(
        f"{REFRAME_SEED}:{sample.spec.seed}:reframe-rank-v{REFRAME_VERSION}".encode()).digest())
    return {sample.spec.seed: rank for rank, sample in enumerate(order)}


def reframe_source(source, rank: int, pool_digest: str):
    seed = source.spec.seed
    rng = np.random.default_rng([REFRAME_SEED, seed & 0xffffffff, seed >> 32, REFRAME_VERSION])
    fraction_bin = (rank // 2) % 48
    target = float(np.linspace(.055, .475, 48)[fraction_bin] + rng.uniform(-.003, .003))
    background = rank % 2
    # Exactly two of each eight ranks use maximal extent, with both polarities.
    extent_mode = "max_extent" if rank % 8 in (0, 3) else "fraction"
    composition = tuple(current.ANCHORS)[int(rng.integers(len(current.ANCHORS)))]
    previous = source.spec
    spec = current.Spec(
        candidate=100000+previous.candidate, seed=seed, pose=previous.pose, view=previous.view,
        morphology=previous.morphology, tail=previous.tail, style=previous.style,
        orientation=previous.orientation, composition=composition, action_variant=previous.action_variant,
        background=background, target_fraction=target, prompt=previous.prompt,
        geometry_family="legacy_"+previous.pose, extent_mode=extent_mode,
    )
    # Polarity comes from the recorded v5 spec, never from an edge heuristic.
    original_ink = source.pixels != previous.background
    x0,y0,x1,y1 = current.bbox(original_ink)
    native_crop = original_ink[y0:y1,x0:x1]
    source_mask = np.zeros((512,512), dtype=np.uint8)
    source_mask[:native_crop.shape[0], :native_crop.shape[1]] = native_crop
    pixels, details, geometry = current.frame_mask(source_mask.astype(bool), spec)
    fraction = float((pixels == 0).mean())
    issue = "" if .05 <= fraction <= .95 else "black_fraction_out_of_range"
    details.update({
        "method": "v5_independent_diffusion_source_reframed_once",
        "reframe_version": REFRAME_VERSION, "reframe_seed": REFRAME_SEED,
        "source_pipeline_version": 5, "source_master_seed": 1142031,
        "source_candidate_id": previous.candidate, "source_seed": seed,
        "source_background": previous.background,
        "source_pixels_sha256": current.fingerprint(source.pixels),
        "source_reference_pool_sha256": pool_digest,
        "source_native_resolution": [224,224], "source_native_bbox": [x0,y0,x1,y1],
        "source_mask_encoding": "native_224_foreground_crop_at_top_left_of_zero_512_canvas; not_raw_512_generation",
        "source_spec_v5": asdict(previous),
        "source_visual_review": "passed_exact_v5_reference_AI_review",
        "current_visual_review": "not_reviewed",
        "semantic_score_scope": "original_v5_render; not_rescored_after_reframing",
        "semantic_embedding_scope": "original_v5_render; not_recomputed_after_reframing",
        "phase_inversion_from_v5": bool(background != previous.background),
        "one_output_per_source_seed": True,
    })
    return current.Sample(
        spec, pixels, source.embedding.copy(), source.cat_logit, source.cat_margin,
        source.alternative, source.view_scores.copy(), fraction, issue, geometry, details, source_mask,
    )


def prepare_legacy(work_dir: Path = DEFAULT_WORK, old_cache: Path = DEFAULT_CACHE,
                   reference_script: Path | None = None, batch_size: int = 4,
                   limit: int | None = None):
    """Create cached candidates only; call after the improved pilot passes."""
    work_dir = Path(work_dir)
    reference_script = Path(reference_script) if reference_script else work_dir/"generation_v5_reference.py"
    sources, digest = approved_sources(reference_script, Path(old_cache))
    ranks = framing_rank(sources)
    if limit is not None:
        sources = sources[:limit]
    destination = work_dir/"legacy-reframed"
    destination.mkdir(parents=True,exist_ok=True)
    samples = []
    for start in range(0,len(sources),batch_size):
        batch = [reframe_source(source,ranks[source.spec.seed],digest) for source in sources[start:start+batch_size]]
        path = destination/f"batch_{start:05d}_{len(batch)}.npz"
        if path.exists():
            stored = current.read_batch(path)
            if ([sample.spec for sample in stored] != [sample.spec for sample in batch]
                    or [current.fingerprint(sample.pixels) for sample in stored]
                       != [current.fingerprint(sample.pixels) for sample in batch]
                    or [sample.diagnostics for sample in stored] != [sample.diagnostics for sample in batch]):
                raise RuntimeError(f"Cached legacy candidates do not match reproduction: {path}")
            batch = stored
        else:
            current.cache_batch(path,batch)
        samples.extend(batch)
    valid = [sample for sample in samples if not current.rejection(sample)]
    summary = {
        "candidate_count": len(samples), "valid_count": len(valid),
        "source_pool_sha256": digest, "unique_source_seeds": len({sample.spec.seed for sample in samples}),
        "exact_duplicate_count": len(samples)-len({current.fingerprint(sample.pixels) for sample in samples}),
        "black_fraction_range": [min(sample.black_fraction for sample in valid),max(sample.black_fraction for sample in valid)] if valid else [],
        "background_counts": dict(Counter(sample.spec.background for sample in samples)),
        "extent_counts": dict(Counter(sample.spec.extent_mode for sample in samples)),
        "rejections": dict(Counter(current.rejection(sample) for sample in samples if current.rejection(sample))),
        "source_visual_review": "passed_exact_v5_reference_AI_review",
        "current_visual_review": "not_reviewed", "current_semantic_rescoring": "not_performed",
        "mechanical_validation": "not_performed", "output_is_final_dataset": False,
    }
    (destination/"candidate_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true",help="Prepare the contingency pool after the pilot passes.")
    parser.add_argument("--work-dir",type=Path,default=DEFAULT_WORK)
    parser.add_argument("--old-cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--reference-script",type=Path)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    if not args.prepare:
        parser.error("No candidates prepared. Pass --prepare only after the improved pilot passes.")
    if args.batch_size<=0 or (args.limit is not None and args.limit<=0):
        parser.error("batch-size and limit must be positive")
    prepare_legacy(args.work_dir,args.old_cache,args.reference_script,args.batch_size,args.limit)


if __name__=="__main__":
    main()
