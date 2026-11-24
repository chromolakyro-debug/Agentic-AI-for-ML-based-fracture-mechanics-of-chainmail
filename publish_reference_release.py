"""Guarded publication of the fully checked reference-style release.

Only the six dataset deliverables and newly added PNGs are written. Existing
user notes/cache and all original PNG bytes are retained. No recursive removal.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
FILES = ('dataset.npy', 'metadata.csv', 'preview.png', 'quality_report.md',
         'generate_dataset.py', 'README.md')
CHANGED_GLOBAL_FIELDS = {'property_bin_count', 'balanced_sampling_weight', 'preview_cell'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def atomic_copy(source, destination):
    temporary = destination.with_name(destination.name + '.publishing')
    shutil.copy2(source, temporary)
    if sha(source) != sha(temporary):
        raise ValueError(f'Copy integrity failed: {destination.name}')
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--total', type=int, default=8000)
    parser.add_argument('--publish', action='store_true', help='Default is read-only preflight')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    release = work / 'release'
    live = (ROOT / 'cat_binary_dataset').resolve()
    if live.parent != ROOT.resolve() or not live.is_dir() or release == live:
        raise ValueError('Unexpected publication destination')
    finalized = read_json(work / 'finalized_staging.json')
    package = read_json(work / 'package_export_check.json')
    if (finalized.get('count') != args.total or
        finalized.get('standalone_export_audit') != 'pass' or
        not package.get('all_six_deliverable_files_byte_identical') or
        not package.get('all_pngs_byte_identical')):
        raise ValueError('Staging/package checks are incomplete')
    if finalized.get('package_audit_sha256') != sha(work / 'package_export_check.json'):
        raise ValueError('Finalization is not bound to this packaging audit')
    if package.get('staged_deliverable_sha256') != {name: sha(release / name) for name in FILES}:
        raise ValueError('Staging deliverables changed after the packaging audit')
    identity = read_json(work / 'base_identity.json')
    if sha(live / 'dataset.npy') != identity['dataset_file_sha256'] or \
       sha(live / 'metadata.csv') != identity['metadata_sha256']:
        raise ValueError('Live original dataset changed since this expansion began')
    original_rows = read_rows(live / 'metadata.csv')
    rows = read_rows(release / 'metadata.csv')
    if len(original_rows) != 1250 or len(rows) != args.total:
        raise ValueError('Unexpected original/release count')
    for old, new in zip(original_rows, rows):
        for field, value in old.items():
            if field not in CHANGED_GLOBAL_FIELDS and new.get(field) != value:
                raise ValueError(f'Original metadata changed: {old["filename"]}/{field}')
    original = np.load(live / 'dataset.npy', mmap_mode='r', allow_pickle=False)
    expanded = np.load(release / 'dataset.npy', mmap_mode='r', allow_pickle=False)
    if expanded.shape != (args.total, 224, 224) or not np.array_equal(original, expanded[:1250]):
        raise ValueError('Original array pixels changed')
    del original, expanded
    existing = {p.name for p in (live / 'images').glob('*.png')}
    expected = {f'cat_{i:04d}.png' for i in range(args.total)}
    if existing != {f'cat_{i:04d}.png' for i in range(1250)}:
        raise ValueError('Live PNG inventory changed')
    if {p.name for p in (release / 'images').glob('*.png')} != expected:
        raise ValueError('Staging PNG inventory differs')
    png_manifest = hashlib.sha256()
    for name in sorted(expected):
        png_manifest.update(f'{name}:{sha(release / "images" / name)}\n'.encode('ascii'))
    if png_manifest.hexdigest() != package.get('staged_ordered_png_manifest_sha256'):
        raise ValueError('Staging PNGs changed after the packaging audit')
    for name in existing:
        if sha(live / 'images' / name) != sha(release / 'images' / name):
            raise ValueError(f'Original PNG bytes changed: {name}')
    notes = {p.name: sha(p) for p in live.iterdir() if p.is_file() and p.name not in FILES}
    subprocess.run([sys.executable, str(release / 'generate_dataset.py'), 'validate',
                    '--output', str(release), '--total', str(args.total)], check=True, cwd=ROOT)
    print('Publication preflight passed.', flush=True)
    if not args.publish:
        return
    backup = work / 'prepublication_original_deliverables'
    backup.mkdir(exist_ok=True)
    for name in FILES:
        if (backup / name).exists() and sha(backup / name) != sha(live / name):
            raise ValueError('A different prepublication backup already exists')
        shutil.copy2(live / name, backup / name)
    journal = work / 'publication_state.json'
    journal.write_text(json.dumps(dict(stage='copying_new_images', output=str(live),
                                      preserved_user_files=notes), indent=2), encoding='utf-8')
    for i in range(1250, args.total):
        name = f'cat_{i:04d}.png'
        atomic_copy(release / 'images' / name, live / 'images' / name)
    for name in FILES:
        atomic_copy(release / name, live / name)
    subprocess.run([sys.executable, str(live / 'generate_dataset.py'), 'validate',
                    '--output', str(live), '--total', str(args.total)], check=True, cwd=ROOT)
    if any(not (live / name).exists() or sha(live / name) != digest for name, digest in notes.items()):
        raise ValueError('User note changed during publication')
    if any(sha(live / name) != sha(release / name) for name in FILES):
        raise ValueError('Published deliverable differs')
    if any(sha(live / 'images' / name) != sha(release / 'images' / name) for name in expected):
        raise ValueError('Published PNG differs')
    result = dict(stage='published_and_validated', count=args.total, output=str(live),
                  original1250_pixels_and_png_bytes_preserved=True,
                  original1250_sample_metadata_preserved=True,
                  recomputed_global_metadata_fields=sorted(CHANGED_GLOBAL_FIELDS),
                  preserved_user_files=notes,
                  deliverable_sha256={name: sha(live / name) for name in FILES})
    journal.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
