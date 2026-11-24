"""Audit all 8000 images; freeze splits, train-only statistics and supported panels."""
import csv
import hashlib
import time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from common import DATA, OUT, SEED, S_TOL, A_TOL, save_json, to_sa, from_sa, sha_file


def longest_run(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    if not len(starts):
        raise RuntimeError('No supported interval in requested scan')
    best = np.argmax(ends - starts)
    return int(starts[best]), int(ends[best])


def main():
    started = time.perf_counter()
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Frozen protocol already exists; do not silently overwrite an experiment.')
    arr = np.load(DATA / 'dataset.npy', mmap_mode='r', allow_pickle=False)
    rows = list(csv.DictReader((DATA / 'metadata.csv').open(encoding='utf-8-sig', newline='')))
    assert arr.shape == (8000, 224, 224) and arr.dtype == np.uint8 and len(rows) == 8000
    assert len(list((DATA / 'images').glob('*.png'))) == 8000
    hashes = []
    for i, row in enumerate(rows):
        assert row['filename'] == f'cat_{i:04d}.png'
        image = np.asarray(Image.open(DATA / 'images' / row['filename']).convert('L'))
        assert np.isin(image, (0, 255)).all()
        assert np.array_equal(image // 255, arr[i]) and np.isin(arr[i], (0, 1)).all()
        digest = hashlib.sha256(arr[i].tobytes()).hexdigest()
        assert digest == row['sha256_pixels']
        hashes.append(digest)
        assert float(row['E_soft']) == 1. and float(row['E_hard']) == 10.
        assert float(row['nu_soft']) == float(row['nu_hard']) == .3
        assert float(row['specimen_width_mm']) == float(row['specimen_height_mm']) == 40.
        assert float(row['specimen_thickness_mm']) == 2. and float(row['nominal_strain']) == .001
        assert row['fem_valid'].lower() == 'true'
        assert abs(float(row['black_fraction']) - np.mean(arr[i] == 0)) < 1e-12
        for axis in ('x', 'y'):
            assert float(row[f'residual_{axis}']) < 1e-7
            assert float(row[f'energy_relative_error_{axis}']) < 1e-6
            assert float(row[f'force_balance_relative_error_{axis}']) < 1e-7
            assert abs(float(row[f'E{axis}_app']) - float(row[f'E{axis}_over_Es'])) < 1e-12
        if (i + 1) % 2000 == 0:
            print(f'Audited {i+1}/8000 PNG/array/hash/metadata records', flush=True)
    assert len(set(hashes)) == 8000
    moduli = np.array([[float(r['Ex_app']), float(r['Ey_app'])] for r in rows])
    sa = to_sa(moduli[:, 0], moduli[:, 1])
    fraction = np.mean(arr == 0, axis=(1, 2))
    assert np.all(moduli >= (1 / (1 - .9 * fraction))[:, None] - 1e-6)
    assert np.all(moduli <= (1 + 9 * fraction)[:, None] + 1e-6)
    # Fixed physical coarse bins are used ONLY to assign splits, never to estimate training statistics.
    bin_s = np.clip(np.floor(sa[:, 0] / np.log(10) * 12).astype(int), 0, 11)
    bin_a = np.clip(np.floor((sa[:, 1] + 1.5) / 3 * 12).astype(int), 0, 11)
    strata = bin_s * 12 + bin_a
    counts = np.bincount(strata, minlength=144)
    strata[counts[strata] < 10] = 144  # merge rare cells, preserving exact split sizes
    train, rest = train_test_split(np.arange(8000), test_size=.2, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=.5, stratify=strata[rest], random_state=SEED + 1)
    train, val, test = map(np.sort, (train, val, test))
    mean, std = sa[train].mean(0), sa[train].std(0)
    ztrain = (sa[train] - mean) / std
    tree = cKDTree(ztrain)
    train_r = tree.query(ztrain, k=21)[0][:, -1]  # excludes the query point itself
    val_r = tree.query((sa[val] - mean) / std, k=20)[0][:, -1]
    threshold = float(np.quantile(val_r, .95))
    density_cutoffs = np.quantile(train_r, [.5, .8]).tolist()
    # Moderate inverse-square-root cell balancing with a uniform mixture and a hard weight cap.
    edges = [np.linspace(sa[train, j].min() - 1e-9, sa[train, j].max() + 1e-9, n + 1)
             for j, n in enumerate((8, 12))]
    bins = np.stack([np.clip(np.digitize(sa[train, j], edges[j]) - 1, 0, len(edges[j])-2)
                     for j in range(2)], axis=1)
    codes = bins[:, 0] * 12 + bins[:, 1]
    cell_counts = np.bincount(codes, minlength=96)
    raw = 1 / np.sqrt(cell_counts[codes])
    weights = np.minimum(.5 + .5 * raw / raw.mean(), 4.)
    weights /= weights.mean()

    def score(points):
        return tree.query((np.asarray(points) - mean) / std, k=20)[0][..., -1]

    def physical(points):
        exy = from_sa(points)
        return ((exy.min(axis=-1) >= 1) & (exy.max(axis=-1) <= 10) &
                (exy.max(axis=-1) <= 11 - 10 / exy.min(axis=-1)))

    targets = []
    scan_ranges = []
    def append_scan(group, varying, fixed, grid):
        points = np.stack([np.full_like(grid, fixed), grid], -1) if varying == 'a' else np.stack([grid, np.full_like(grid, fixed)], -1)
        supported = (score(points) <= threshold) & physical(points)
        lo, hi = longest_run(supported)
        # Select within the longest supported connected interval; do not jump over unsupported holes.
        chosen = np.linspace(grid[lo] + .05 * (grid[hi]-grid[lo]), grid[hi] - .05 * (grid[hi]-grid[lo]), 4)
        scan_ranges.append(dict(group=group, varying=varying, fixed=float(fixed), supported_interval=[float(grid[lo]),float(grid[hi])]))
        for v in chosen:
            p = [fixed, v] if varying == 'a' else [v, fixed]
            r = float(score([p])[0])
            assert r <= threshold and physical(np.array([p]))[0]
            targets.append(dict(target_id=f'T{len(targets):02d}', group=group, varying=varying,
                                s=float(p[0]), a=float(p[1]), Ex=float(from_sa(p)[0]), Ey=float(from_sa(p)[1]),
                                support_r20=r, support_ratio=r/threshold,
                                density='dense' if r <= density_cutoffs[0] else ('sparse' if r >= density_cutoffs[1] else 'medium')))
    for j, quantile in enumerate((.3, .7, .94)):
        append_scan(f'vary_a_{j}', 'a', float(np.quantile(sa[train, 0], quantile)), np.linspace(-1.4, 1.4, 2801))
    magnitude = float(np.quantile(np.abs(sa[train, 1]), .75))
    for j, fixed in enumerate((-magnitude, magnitude)):
        append_scan(f'vary_s_{j}', 's', fixed, np.linspace(.01, np.log(10)-.01, 2401))
    assert len(targets) == 20 and any(t['density'] == 'sparse' for t in targets)
    # Development panel is separate; selected without model outputs, using only training labels.
    dev = []
    for quantile in (.15, .45, .75, .9):
        candidate = int(np.argmin(np.linalg.norm(ztrain - [(np.quantile(sa[train,0],quantile)-mean[0])/std[0], 0],axis=1)))
        p = sa[train[candidate]]
        dev.append(dict(target_id=f'D{len(dev):02d}', s=float(p[0]), a=float(p[1]), source_index=int(train[candidate]), support_r20=float(score([p])[0])))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / 'data.npz', sa=sa, moduli=moduli, train=train, validation=val, test=test,
             train_weights=weights, mean=mean, std=std, train_r20=train_r, validation_r20=val_r,
             cell_counts=cell_counts, bin_edges_s=edges[0], bin_edges_a=edges[1])
    save_json(OUT / 'splits.json', dict(seed=SEED, train=train.tolist(), validation=val.tolist(), test=test.tolist(),
                                     method='12x12 fixed physical s/a bins; cells with <10 pooled; two stratified random splits'))
    protocol = dict(seed=SEED, esoft=1., ehard=10., representation='x=2*binary-1; -1 black hard, +1 white soft',
                    augmentation='none', threshold=0., ode_solver='midpoint', ode_steps=64,
                    final_noise_seeds=list(range(730000, 730032)), dev_noise_seeds=[630000,630001],
                    hit_tolerances=dict(s=S_TOL, a=A_TOL), support=dict(k=20, threshold=threshold,
                    rule='95th percentile validation-to-training kNN distance in training-standardized s/a',
                    density_cutoffs_train_r20=density_cutoffs, warning='Empirical support only, not physical attainability'),
                    training_statistics=dict(mean=mean.tolist(),std=std.tolist(),
                    sampling='0.5 uniform + 0.5 normalized inverse-sqrt training 8x12 cell count; cap 4 then mean-normalize',
                    weight_min=float(weights.min()), weight_max=float(weights.max()),
                    effective_sample_size=float(weights.sum()**2/(weights**2).sum())),
                    panel_rule='3 fixed-s training quantiles (0.30,0.70,0.94), and a=+/- train abs(a) Q75; 4 interior equispaced points per longest supported interval',
                    scan_ranges=scan_ranges, final_targets=targets, development_targets=dev,
                    data_sha256=sha_file(DATA/'dataset.npy'), metadata_sha256=sha_file(DATA/'metadata.csv'))
    save_json(OUT / 'protocol.json', protocol)
    save_json(OUT / 'frozen_protocol_sha256.json', dict(sha256=sha_file(OUT/'protocol.json')))
    save_json(OUT / 'data_audit.json', dict(n=8000, all_png_array_metadata_hashes_match=True, unique_pixel_hashes=8000,
              material_and_fem_diagnostics_pass=True, split_sizes=[len(train),len(val),len(test)],
              label_protocol_sha256=sorted(set(r['solver_protocol_sha256'] for r in rows)),
              row_alignment='cat_0000..cat_7999, array index = CSV row = filename number',
              elapsed_seconds=time.perf_counter()-started))
    print('Support threshold', threshold, 'train density cutoffs', density_cutoffs, flush=True)
    for t in targets: print(t, flush=True)


if __name__ == '__main__':
    main()
