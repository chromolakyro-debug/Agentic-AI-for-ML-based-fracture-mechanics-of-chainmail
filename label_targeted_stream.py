"""CPU-label early targeted source batches while independent GPU work continues.

Only a prefix (default 280/350 sources) is handled. A conservative cache-progress
guard stops new submissions before the GPU producer reaches its own FEM stage.
Run only one stream writer; the main producer must not yet be writing labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

# Import first so spawned FEM workers inherit single-thread BLAS settings.
from evaluate_mechanics import label_items, mechanics_fingerprint
import candidate_generation as base
import targeted_candidates as targeted


def batch_path(cache, start, total, batch_size):
    return cache/f"batch_{start:05d}_{min(batch_size,total-start)}.npz"


def completed_prefix(cache, total, batch_size):
    completed = 0
    for start in range(0,total,batch_size):
        if not batch_path(cache,start,total,batch_size).is_file():
            break
        completed += min(batch_size,total-start)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",type=Path,required=True)
    parser.add_argument("--destination",type=Path)
    parser.add_argument("--stage",choices=("pilot","full"),default="full")
    parser.add_argument("--seed",type=int,default=targeted.MASTER_SEED)
    parser.add_argument("--total-sources",type=int,default=350)
    parser.add_argument("--limit",type=int,default=280)
    parser.add_argument("--chunk-sources",type=int,default=20)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--stop-at-cache-sources",type=int,default=300,
                        help="Conservative progress guard; no new chunk starts once this many source caches exist")
    parser.add_argument("--poll-seconds",type=float,default=2.)
    parser.add_argument("--max-wait-seconds",type=float,default=3600.)
    args = parser.parse_args()
    if not 0 < args.limit < args.stop_at_cache_sources < args.total_sources:
        parser.error("Require 0 < limit < stop-at-cache-sources < total-sources")
    if args.chunk_sources % args.batch_size or args.limit % args.batch_size:
        parser.error("chunk-sources and limit must be multiples of source batch size")
    if args.workers < 1 or args.poll_seconds <= 0:
        parser.error("workers and poll-seconds must be positive")
    args.destination = args.destination or args.cache_dir.parent/f"{args.stage}_labels.jsonl"
    expected = targeted.specs_for(args.stage,args.total_sources,args.seed)
    report_path = args.destination.parent/f"{args.stage}_stream_label_status.json"
    report = dict(stage=args.stage,source_limit=args.limit,total_source_count=args.total_sources,
                  source_count_processed=0,valid_variants_labelled=0,submissions_stopped=False,
                  mechanics_protocol=mechanics_fingerprint(),status="waiting")
    began = time.monotonic()
    last_wait_message = 0.

    def save(status):
        report.update(status=status,elapsed_seconds=time.monotonic()-began,
                      source_caches_observed=completed_prefix(args.cache_dir,args.total_sources,args.batch_size))
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")

    for start in range(0,args.limit,args.chunk_sources):
        end = min(args.limit,start+args.chunk_sources)
        while True:
            cached_count = completed_prefix(args.cache_dir,args.total_sources,args.batch_size)
            if cached_count >= args.stop_at_cache_sources:
                report["submissions_stopped"] = True
                save("stopped_before_main_fem")
                print(f"Stream stopping: {cached_count}/{args.total_sources} source caches exist; no further FEM submission.",flush=True)
                return
            if cached_count >= end:
                break
            if time.monotonic()-began > args.max_wait_seconds:
                save("generation_wait_timeout")
                raise TimeoutError("Source generation did not provide the next chunk in time")
            if time.monotonic()-last_wait_message >= 30:
                print(f"Stream waiting: need sources {start}–{end-1}; {cached_count} cached.",flush=True)
                last_wait_message = time.monotonic()
                save("waiting")
            time.sleep(args.poll_seconds)
        sources = []
        for batch_start in range(start,end,args.batch_size):
            batch = base.read_batch(batch_path(args.cache_dir,batch_start,args.total_sources,args.batch_size))
            if [s.spec for s in batch] != expected[batch_start:batch_start+len(batch)]:
                raise ValueError(f"Source specification/cache mismatch at batch {batch_start}")
            sources.extend(batch)
        variants = [variant for source in sources for variant in targeted.variants_for_sample(source,count=3)]
        valid = [variant for variant in variants if not targeted.rejection(variant)]
        items = [(f"{args.stage}:{s.spec.seed}:{s.diagnostics['variant_id']}",s.pixels) for s in valid]
        # Recheck immediately before entering the synchronous write operation.
        cached_count = completed_prefix(args.cache_dir,args.total_sources,args.batch_size)
        if cached_count >= args.stop_at_cache_sources:
            report["submissions_stopped"] = True
            save("stopped_before_main_fem")
            print("Stream stopping before chunk submission to avoid the main FEM writer.",flush=True)
            return
        print(f"Stream FEM chunk: sources {start}–{end-1}; {len(items)} valid variants.",flush=True)
        save("labelling_current_chunk")
        label_items(items,args.destination,workers=args.workers)
        # label_items has closed its output stream before this point. Even if
        # source generation finished during that call, no new work is submitted.
        report["source_count_processed"] = end
        report["valid_variants_labelled"] += len(items)
        cached_count = completed_prefix(args.cache_dir,args.total_sources,args.batch_size)
        if cached_count >= args.stop_at_cache_sources:
            report["submissions_stopped"] = True
            save("stopped_after_current_write")
            print(f"Stream closed after current write; source caches now {cached_count}/{args.total_sources}.",flush=True)
            return
        save("chunk_complete")
    save("complete_prefix")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
