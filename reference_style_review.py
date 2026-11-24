"""Stream immutable source contact sheets; record only already-performed AI review.

This CPU-only development utility never generates images, estimates semantics, or
changes the source cache. `record` is an explicit attestation that the caller has
inspected every populated cell of the named page. Positions are ONE-based within
a page; page numbers are ZERO-based. Only one writer should run per subcommand.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import candidate_generation as base
import reference_style_candidates as engine


SCHEMA = 1
PAGE_SIZE = 100
GRID = dict(columns=10, rows=10, tile=224, gap=6, label_height=34)
BATCH_RE = re.compile(r"batch_(\d+)_(\d+)\.npz$")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha(pixels):
    return sha(np.asarray(pixels, dtype=np.uint8).tobytes())


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    pending.replace(path)


def immutable_write(path, content):
    # An interrupted run may have published the PNG before its manifest. Accept
    # identical bytes on recovery, but never overwrite an existing sealed file.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + "-", suffix=".pending", delete=False) as stream:
        pending = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            # Atomic, no-clobber publication on the local NTFS workspace. Both
            # paths are on the same volume; a reader can never see half a file.
            os.link(pending, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite immutable page file: {path}")
    finally:
        pending.unlink(missing_ok=True)


def config_for(args):
    return dict(schema=SCHEMA, stage=args.stage, master_seed=args.seed,
                batch_size=args.batch_size, engine_version=engine.VERSION,
                cache_dir=str(engine.cache_dir(args.stage, args.work_dir, args.seed,
                                               args.batch_size).resolve()))


def page_paths(args, number):
    folder = args.work_dir / "source_review_pages"
    return folder / f"page_{number:03d}.json", folder / f"page_{number:03d}.png"


def identity(manifest):
    return sha(canonical({k: v for k, v in manifest.items() if k != "page_identity"}))


def load_manifest(args, number):
    path, png = page_paths(args, number)
    manifest = read_json(path)
    if not manifest or manifest.get("page_identity") != identity(manifest):
        raise ValueError(f"Missing or changed page manifest: {path}")
    if manifest["config"] != config_for(args) or manifest["page"] != number:
        raise ValueError(f"Page belongs to a different generation run: {path}")
    if manifest["grid"] != GRID or not png.exists():
        raise ValueError(f"Invalid or incomplete sealed page: {path}")
    entries = manifest["entries"]
    if not 1 <= len(entries) <= PAGE_SIZE:
        raise ValueError(f"Invalid page size: {path}")
    if [e["position"] for e in entries] != list(range(1, len(entries) + 1)):
        raise ValueError(f"Invalid page positions: {path}")
    if manifest["ordered_pixel_hashes"] != [e["pixels_sha256"] for e in entries]:
        raise ValueError(f"Pixel order mismatch: {path}")
    return manifest


def page_numbers(args):
    folder = args.work_dir / "source_review_pages"
    numbers = sorted(int(p.stem.split("_")[-1]) for p in folder.glob("page_*.json"))
    if numbers != list(range(len(numbers))):
        raise ValueError("Sealed page sequence has a gap")
    return numbers


def make_entry(sample, cache_file, batch_index):
    d = sample.diagnostics
    return dict(source_key=f"reference:{sample.spec.seed}", seed=int(sample.spec.seed),
                source_id=int(sample.spec.candidate), style_family=d["style_family"],
                reference_style_id=d["reference_style_id"],
                reference_filename=d["reference_filename"],
                pixels_sha256=pixel_sha(sample.pixels),
                black_fraction=float(np.mean(sample.pixels == 0)),
                cache_file=str(cache_file.resolve()), batch_index=batch_index)


def samples_for(args, entries):
    batches, samples = {}, []
    expected_cache = Path(config_for(args)["cache_dir"])
    for entry in entries:
        path = Path(entry["cache_file"])
        if not path.is_absolute() or path.resolve().parent != expected_cache:
            raise ValueError("Cache reference is outside the configured source cache")
        if str(path) not in batches:
            batches[str(path)] = base.read_batch(path)
        batch = batches[str(path)]
        index = entry["batch_index"]
        if not isinstance(index, int) or index < 0 or index >= len(batch):
            raise ValueError("Invalid source batch index")
        sample = batch[index]
        actual = make_entry(sample, path, index)
        if any(entry.get(field) != value for field, value in actual.items()):
            raise ValueError(f"Source pixels or metadata changed: {entry['source_key']}")
        if engine.rejection(sample):
            raise ValueError(f"Source no longer passes automatic screening: {entry['source_key']}")
        samples.append(sample)
    return samples


def render(samples, entries, page, tile=224, columns=10, rows=10):
    gap, label_height = GRID["gap"], GRID["label_height"]
    sheet = Image.new("1", (gap + columns * (tile + gap),
                            gap + rows * (tile + label_height + gap)), 1)
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default()
    for index, (sample, entry) in enumerate(zip(samples, entries)):
        x, y = gap + index % columns * (tile + gap), gap + index // columns * (tile + label_height + gap)
        image = Image.fromarray(sample.pixels.astype(bool)).resize((tile, tile), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y))
        draw.rectangle((x - 1, y - 1, x + tile, y + tile), outline=0)
        label = (f"p{page:03d} #{entry['position']:03d} ref{entry['reference_style_id']} "
                 f"B={entry['black_fraction']:.1%}\nseed {entry['seed']}")
        draw.multiline_text((x + 3, y + tile + 3), label, fill=0, font=font, spacing=2)
    output = io.BytesIO()
    sheet.save(output, format="PNG")
    return output.getvalue()


def seal_page(args, number, pending, final_partial=False):
    entries = [dict(e, position=i + 1) for i, e in enumerate(pending)]
    samples = samples_for(args, entries)
    png_bytes = render(samples, entries, number)
    manifest = dict(config=config_for(args), page=number, count=len(entries), grid=GRID,
                    final_partial=bool(final_partial), entries=entries,
                    ordered_pixel_hashes=[e["pixels_sha256"] for e in entries],
                    png_sha256=sha(png_bytes))
    manifest["page_identity"] = identity(manifest)
    path, png = page_paths(args, number)
    path.parent.mkdir(parents=True, exist_ok=True)
    immutable_write(png, png_bytes)
    immutable_write(path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    return manifest


def verify_page(args, number):
    manifest = load_manifest(args, number)
    _, png = page_paths(args, number)
    if file_sha(png) != manifest["png_sha256"]:
        raise ValueError(f"Page PNG changed: {png}")
    samples = samples_for(args, manifest["entries"])
    # Verify displayed pixels, independently of the PNG/container hash. Text and
    # borders are outside these crops. This binds review to the actual maps.
    with Image.open(png) as sheet:
        expected_size = (6 + 10 * 230, 6 + 10 * 264)
        if sheet.size != expected_size:
            raise ValueError("Contact sheet has unexpected dimensions")
        for index, sample in enumerate(samples):
            x, y = 6 + index % 10 * 230, 6 + index // 10 * 264
            crop = np.asarray(sheet.crop((x, y, x + 224, y + 224)).convert("L"))
            if not np.array_equal(crop, sample.pixels * 255):
                raise ValueError(f"Displayed pixels differ at page {number}, cell {index + 1}")
    return manifest, samples


def generation_state(args):
    path = args.work_dir / f"{args.stage}_generation_state.json"
    state = read_json(path)
    if state is None:
        return dict(complete=False)
    for field, expected in (("stage", args.stage), ("seed", args.seed), ("batch_size", args.batch_size)):
        if state.get(field) != expected:
            raise ValueError(f"Generation state {field} differs from review configuration")
    return state


def emit(args):
    config = config_for(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.work_dir / "source_review_index.json"
    index = read_json(index_path, dict(config=config, seen_batches={}, pending_cache_refs=[],
                                     next_source_index=0, rejected_counts={}))
    if index["config"] != config:
        raise ValueError("Review index belongs to another stage, seed, batch size or engine version")
    numbers = page_numbers(args)
    sealed_keys = set()
    for number in numbers:
        manifest = load_manifest(args, number)
        for entry in manifest["entries"]:
            if entry["source_key"] in sealed_keys:
                raise ValueError("One source appears on multiple sealed pages")
            sealed_keys.add(entry["source_key"])
    # Recover safely if a page was sealed immediately before an interrupted
    # index update. Already sealed sources can never enter a later page.
    pending = [e for e in index["pending_cache_refs"] if e["source_key"] not in sealed_keys]
    known_keys = {k for b in index["seen_batches"].values() for k in b["source_keys"]}
    rejected = Counter(index["rejected_counts"])
    for name, old in index["seen_batches"].items():
        path = Path(name)
        stat = path.stat()
        if stat.st_size != old["size"] or stat.st_mtime_ns != old["mtime_ns"]:
            if file_sha(path) != old["sha256"]:
                raise ValueError(f"Previously read immutable source batch changed: {path}")
    new_batches = new_sources = 0
    paths = sorted(Path(config["cache_dir"]).glob("batch_*.npz"))
    for path in paths:
        match = BATCH_RE.fullmatch(path.name)
        if not match:
            continue
        name = str(path.resolve())
        if name in index["seen_batches"]:
            continue
        start, count = map(int, match.groups())
        expected = index["next_source_index"]
        if start > expected:
            break  # An earlier atomic publication is not visible yet.
        if start < expected:
            raise ValueError(f"Overlapping source cache batch: {path}")
        batch = base.read_batch(path)
        if len(batch) != count:
            raise ValueError(f"Source count does not match cache filename: {path}")
        keys = []
        for batch_index, sample in enumerate(batch):
            d = sample.diagnostics
            if (d.get("source_mode") != args.stage or
                    d.get("generation_master_seed") != args.seed or
                    d.get("reference_pipeline_version") != engine.VERSION):
                raise ValueError(f"Source belongs to another generation run: {path}")
            key = f"reference:{sample.spec.seed}"
            if key in known_keys or key in keys:
                raise ValueError(f"Duplicate source seed in cache: {key}")
            keys.append(key)
            reason = engine.rejection(sample)
            if reason:
                rejected[reason] += 1
            elif key not in sealed_keys:
                pending.append(make_entry(sample, path, batch_index))
                new_sources += 1
        known_keys.update(keys)
        stat = path.stat()
        index["seen_batches"][name] = dict(sha256=file_sha(path), size=stat.st_size,
                                           mtime_ns=stat.st_mtime_ns, source_keys=keys)
        index["next_source_index"] += count
        new_batches += 1
    state = generation_state(args)
    complete = bool(state.get("complete"))
    if complete and index["next_source_index"] != state["source_count"]:
        raise ValueError("Generation is complete but the full contiguous source cache is unavailable")
    if numbers and load_manifest(args, numbers[-1])["final_partial"] and pending:
        raise ValueError("Cannot extend a completed run after its partial final page was sealed")
    created, next_page = [], len(numbers)
    while len(pending) >= PAGE_SIZE or (complete and pending):
        count = min(PAGE_SIZE, len(pending))
        seal_page(args, next_page, pending[:count], final_partial=count < PAGE_SIZE)
        del pending[:count]
        created.append(next_page)
        next_page += 1
    index.update(pending_cache_refs=pending, rejected_counts=dict(rejected),
                 sealed_page_count=next_page, generation_complete=complete)
    write_json(index_path, index)
    print(json.dumps(dict(new_batches=new_batches, new_valid_sources=new_sources,
                          emitted_pages=created, sealed_pages=next_page,
                          pending_sources=len(pending), processed_sources=index["next_source_index"],
                          generation_complete=complete)), flush=True)


def exclusions_for(values, count):
    exclusions = {}
    for item in values:
        position, sep, reason = item.partition(":")
        if not sep or not reason.strip():
            raise ValueError("Each exclusion must be POSITION:nonempty reason")
        position = int(position)
        if position < 1 or position > count or position in exclusions:
            raise ValueError("Exclusion position is outside the page or repeated")
        exclusions[position] = reason.strip()
    return exclusions


def record(args):
    manifest, _ = verify_page(args, args.page)
    exclusions = exclusions_for(args.exclude, manifest["count"])
    path = args.work_dir / "reference_visual_review.json"
    review = read_json(path, {})
    binding = dict(page=args.page, stage=args.stage, page_identity=manifest["page_identity"],
                   png_sha256=manifest["png_sha256"],
                   ordered_pixel_hashes=manifest["ordered_pixel_hashes"],
                   source_keys=[e["source_key"] for e in manifest["entries"]],
                   excluded_cells={str(k): v for k, v in exclusions.items()})
    records = review.setdefault("source_review_pages", {})
    old = records.get(manifest["page_identity"])
    if old and any(old.get(k) != v for k, v in binding.items()):
        raise ValueError("This page already has a different recorded verdict; edit only through an explicit correction workflow")
    passed = set(review.get("passed_hashes", []))
    excluded = dict(review.get("excluded_hashes", {}))
    for entry in manifest["entries"]:
        digest = entry["pixels_sha256"]
        if entry["position"] in exclusions:
            excluded[digest] = exclusions[entry["position"]]
            passed.discard(digest)
        elif digest not in excluded:
            passed.add(digest)
    if not old:
        records[manifest["page_identity"]] = dict(binding, recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            method="Caller attests completed AI inspection of every populated source-sheet cell; no automatic or human review claimed")
    review.update(passed_hashes=sorted(passed), excluded_hashes=excluded,
                  method="AI contact-sheet inspection of actual binary maps; ambiguous specimens enlarged; no human review claimed")
    write_json(path, review)
    print(json.dumps(dict(page=args.page, reviewed_cells=manifest["count"],
                          excluded_cells=exclusions, total_passed_hashes=len(passed),
                          total_excluded_hashes=len(excluded), idempotent=bool(old))), flush=True)


def pending_pages(args):
    review = read_json(args.work_dir / "reference_visual_review.json", {})
    records = review.get("source_review_pages", {})
    pending = []
    for number in page_numbers(args):
        manifest = load_manifest(args, number)
        if manifest["page_identity"] not in records:
            _, png = page_paths(args, number)
            pending.append(dict(page=number, count=manifest["count"], png=str(png.resolve())))
    index = read_json(args.work_dir / "source_review_index.json", {})
    print(json.dumps(dict(unreviewed_pages=pending,
                          waiting_sources=len(index.get("pending_cache_refs", []))), indent=2), flush=True)


def inspect(args):
    manifest, samples = verify_page(args, args.page)
    cells = [int(value) for raw in args.cells for value in raw.split(",") if value]
    if not cells or len(set(cells)) != len(cells) or any(c < 1 or c > len(samples) for c in cells):
        raise ValueError("Inspection cells must be distinct valid one-based positions")
    columns = min(4, len(cells))
    rows = (len(cells) + columns - 1) // columns
    png = render([samples[i - 1] for i in cells], [manifest["entries"][i - 1] for i in cells],
                 args.page, tile=args.tile, columns=columns, rows=rows)
    path = args.work_dir / "source_review_inspection.png"
    path.write_bytes(png)  # Only this optional zoom sheet is intentionally replaceable.
    print(json.dumps(dict(page=args.page, cells=cells, tile=args.tile, png=str(path.resolve()))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("emit", "pending", "record", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--work-dir", type=Path, required=True)
        command.add_argument("--stage", choices=("pilot", "full"), default="full")
        command.add_argument("--seed", type=int, default=engine.MASTER_SEED)
        command.add_argument("--batch-size", type=int, default=4)
        if name in ("record", "inspect"):
            command.add_argument("--page", type=int, required=True)
        if name == "record":
            command.add_argument("--exclude", action="append", default=[], metavar="POSITION:reason",
                                 help="Record already-inspected rejected cells; every other populated cell was inspected and accepted")
        if name == "inspect":
            command.add_argument("--cells", nargs="+", required=True, help="One-based cells, separated by spaces or commas")
            command.add_argument("--tile", type=int, choices=(224, 448), default=448)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    if args.batch_size <= 0 or getattr(args, "page", 0) < 0:
        parser.error("batch-size must be positive; page must be nonnegative")
    dict(emit=emit, pending=pending_pages, record=record, inspect=inspect)[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"Review integrity error: {error}", file=sys.stderr)
        raise SystemExit(2)
