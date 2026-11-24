"""Freeze continuation inputs and new development scans without looking at outputs."""
import paths
import csv
import hashlib
import shutil
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from common import DATA, OUT, read_json, save_json, sha_file, from_sa


def main():
    assert not (OUT / 'protocol.json').exists(), 'Already frozen; do not overwrite'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'checkpoints').mkdir(exist_ok=True)
    old = paths.ORIGINAL
    p = read_json(old / 'protocol.json')
    assert sha_file(DATA / 'dataset.npy') == p['data_sha256']
    assert sha_file(DATA / 'metadata.csv') == p['metadata_sha256']
    images = np.load(DATA / 'dataset.npy', mmap_mode='r')
    rows = list(csv.DictReader((DATA / 'metadata.csv').open(encoding='utf-8-sig', newline='')))
    assert images.shape == (8000, 224, 224) and images.dtype == np.uint8 and len(rows) == 8000
    for i, row in enumerate(rows):
        assert row['filename'] == f'cat_{i:04d}.png'
        with Image.open(DATA / 'images' / row['filename']) as im:
            assert np.array_equal(np.asarray(im).astype(np.uint8), images[i])
        assert np.isin(images[i], [0, 1]).all()
    d = np.load(old / 'data.npz'); train = d['train']; sa = d['sa']
    exey = np.array([[float(r['Ex_app']), float(r['Ey_app'])] for r in rows])
    assert np.array_equal(exey, d['moduli'])
    assert len(train) == 6400 and len(d['validation']) == len(d['test']) == 800
    mean, std = d['mean'], d['std']
    nn = NearestNeighbors(n_neighbors=20).fit((sa[train] - mean) / std)
    def scores(x): return nn.kneighbors((np.asarray(x) - mean) / std)[0][:, -1]
    threshold = p['support']['threshold']; cuts = p['support']['density_cutoffs_train_r20']
    dev = []; ranges = []
    def scan(varying, fixed, grid):
        pts = np.stack([np.full_like(grid, fixed), grid], -1) if varying == 'a' else np.stack([grid, np.full_like(grid, fixed)], -1)
        ex = from_sa(pts)
        ok = (scores(pts) <= threshold) & (ex.min(1) >= 1) & (ex.max(1) <= 10)
        edges = np.flatnonzero(np.diff(np.r_[False, ok, False])); runs = edges.reshape(-1, 2)
        lo, end = max(runs, key=lambda r: r[1] - r[0]); hi = end - 1
        group = f'dev_vary_{varying}_{len(ranges)}'
        chosen = np.linspace(grid[lo] + .12 * (grid[hi]-grid[lo]), grid[hi] - .12 * (grid[hi]-grid[lo]), 3)
        ranges.append(dict(group=group, varying=varying, fixed=float(fixed), supported_interval=[float(grid[lo]),float(grid[hi])]))
        for v in chosen:
            point = [fixed, v] if varying == 'a' else [v, fixed]
            r = float(scores([point])[0]); assert r <= threshold
            dev.append(dict(target_id=f'D{len(dev):02d}', group=group, varying=varying,
                            s=float(point[0]), a=float(point[1]), Ex=float(from_sa(point)[0]), Ey=float(from_sa(point)[1]),
                            support_r20=r, support_ratio=r/threshold,
                            density='dense' if r<=cuts[0] else 'sparse' if r>=cuts[1] else 'medium'))
    for q in (.35, .65, .90): scan('a', float(np.quantile(sa[train,0], q)), np.linspace(-1.4,1.4,2801))
    magnitude = float(np.quantile(np.abs(sa[train,1]), .65))
    for a in (-magnitude, magnitude): scan('s', a, np.linspace(.01,np.log(10)-.01,2401))
    assert len(dev) == 15 and {t['density'] for t in dev} >= {'dense','sparse'}
    p.update(development_targets=dev, dev_noise_seeds=[630000,630001], development_scan_ranges=ranges,
             development_panel_rule='Train s Q35/Q65/Q90 and a=+/-train abs(a) Q65; 3 points at 12/50/88 percent of longest empirically supported interval. 15 targets x 2 seeds, all unfiltered, all FEM.',
             final_panel_status='Historical v0 20-target/32-seed benchmark reused for paired comparison; NOT a new untouched holdout. Not generated during continuation selection.',
             frozen_utc=datetime.now(timezone.utc).isoformat())
    for name in ('data.npz','splits.json','retrieval.json','training_descriptors.npz','data_audit.json'):
        shutil.copy2(old/name,OUT/name)
    save_json(OUT/'protocol.json',p)
    save_json(OUT/'frozen_protocol_sha256.json',dict(sha256=sha_file(OUT/'protocol.json')))
    originals = [old/'checkpoints'/'last.pt',old/'checkpoints'/'best.pt',old/'training.jsonl',old/'train_config.json',old/'protocol.json',old/'data.npz',DATA/'dataset.npy',DATA/'metadata.csv',paths.ROOT/'mechanics.py']
    originals += list((paths.ROOT/'cfm_v0').glob('*.py'))
    save_json(OUT/'original_identity.json',dict(files=[dict(path=str(f),sha256=sha_file(f),bytes=f.stat().st_size) for f in originals],
        data_audit='All 8000 PNG/array/CSV filenames and binary pixels rechecked; array+CSV hashes match original frozen protocol; all Ex/Ey match frozen data.npz exactly.'))
    shutil.copy2(old/'checkpoints'/'best.pt',OUT/'checkpoints'/'milestone_020000.pt')
    shutil.copy2(old/'checkpoints'/'best.pt',OUT/'checkpoints'/'best.pt')
    print({'development':len(dev),'seeds':p['dev_noise_seeds'],'densities':[t['density'] for t in dev],'protocol_sha256':sha_file(OUT/'protocol.json')})

if __name__ == '__main__': main()
