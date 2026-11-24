"""Append-only, FEM-aware selection for independently generated cat sources.

This module never changes pixels, mechanical labels, or existing dataset files.
The fixed base may contain the original 1,000 images and an accepted pilot.
Each selected addition has a new source seed and is checked against every base
image and every other selected addition by the existing joint near-duplicate
rule. Multiple candidate variants of a new source are allowed at input, but at
most one is retained. A short eligible pool is reported, never padded.

API:
    load_base(dataset_dir) -> (samples, labels, original_csv_rows)
    select_additions(base_samples, base_labels, samples, labels, target=250)
        -> (candidate_input_indices, statistics, pair_audit)
    compare_coverage(base_labels, added_labels, base_fractions, added_fractions)

Sample-like dictionaries/objects follow candidate_generation.Sample. Visual
review is an upstream responsibility: root must pass reviewed candidates only.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import mechanics_selection as ms

SELECTION_VERSION = 1
GROUPS = ("high_modulus", "x_dominant", "y_dominant", "exploration")
GROUP_WEIGHTS = np.array([.40, .30, .20, .10])
NEIGHBOR_RADIUS = .5
PROBE_TARGETS = np.array([
    [8.8, 8.8], [9.25, 9.25], [8.8, 7.8], [7.8, 8.8],
    [7., 4.45], [4.5, 2.45], [5.8, 3.], [6.5, 3.5],
    [3.5, 7.5], [4.5, 8.], [6.8, 8.7], [8.3, 5.], [5., 8.3],
], dtype=float)


def _truth(value):
    return str(value).strip().lower() not in ("false", "0", "no", "none", "")


def _values(labels):
    if isinstance(labels, np.ndarray):
        return np.asarray(labels, dtype=float).reshape(-1, 2)
    values = []
    for label in labels:
        values.append([float(ms._get(label, "Ex_app", ms._get(label, "Ex_over_Es"))),
                       float(ms._get(label, "Ey_app", ms._get(label, "Ey_over_Es")))])
    return np.asarray(values, dtype=float).reshape(-1, 2)


def _seed(sample):
    return int(ms._get(ms._get(sample, "spec"), "seed"))


def _digest(sample):
    return hashlib.sha256(np.asarray(ms._get(sample, "pixels"), dtype=np.uint8).tobytes()).hexdigest()


def _diagnostics(sample):
    result = ms._get(sample, "diagnostics", {})
    return result if isinstance(result, dict) else {}


def load_base(dataset_dir):
    """Load immutable base views; validate filename, pixel hash and CSV fraction.

    Returned CSV rows retain their original strings. Samples contain only the
    data needed by selection; labels contain the stored FEM values, unchanged.
    """
    path = Path(dataset_dir)
    pixels = np.load(path / "dataset.npy", allow_pickle=False)
    with (path / "metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if pixels.dtype != np.uint8 or pixels.shape != (len(rows), 224, 224):
        raise ValueError("Base dataset/CSV dimensions or dtype do not agree.")
    if not np.isin(pixels, [0, 1]).all():
        raise ValueError("Base images are not binary uint8 maps.")
    samples, labels = [], []
    for index, (image, row) in enumerate(zip(pixels, rows)):
        if row["filename"] != f"cat_{index:04d}.png":
            raise ValueError(f"Base ordering mismatch at {index}.")
        if hashlib.sha256(image.tobytes()).hexdigest() != row["sha256_pixels"]:
            raise ValueError(f"Base pixel hash mismatch: {row['filename']}.")
        if abs(float((image == 0).mean()) - float(row["black_fraction"])) > 1e-12:
            raise ValueError(f"Base phase fraction mismatch: {row['filename']}.")
        sample = dict(pixels=image, spec=dict(seed=int(row["seed"]),
                      background=int(row["background_value"])), filename=row["filename"],
                      cat_margin=float(row.get("siglip_cat_margin", 0)),
                      cat_logit=float(row.get("siglip_cat_logit", 0)), issue="",
                      diagnostics=dict(source_id=row.get("source_id", row.get("candidate_id", ""))))
        label = dict(row)
        label.update(Ex_app=float(row["Ex_app"]), Ey_app=float(row["Ey_app"]),
                     fem_valid=_truth(row.get("fem_valid", True)))
        samples.append(sample)
        labels.append(label)
    if len({_digest(s) for s in samples}) != len(samples):
        raise ValueError("Base contains exact pixel duplicates; append-only selection cannot fix these.")
    if len({_seed(s) for s in samples}) != len(samples):
        raise ValueError("Base contains repeated source seeds.")
    return samples, labels, rows


def _fine_bins(values, width=.5):
    size = int(round(9 / width))
    xy = np.clip(np.floor((np.asarray(values)-1)/width).astype(int), 0, size-1)
    return xy[:, 1]*size + xy[:, 0], size


def _distribution(values, fractions):
    bins = ms.property_bins(values)
    support = np.flatnonzero(np.bincount(bins, minlength=36))
    out = ms._distribution(values, bins, fractions, support)
    fine, size = _fine_bins(values)
    counts = np.bincount(fine, minlength=size*size)
    out.update(fine_bin_width=.5, fine_bin_counts_y_rows_x_columns=counts.reshape(size, size).tolist(),
               observed_fine_bins=int(np.count_nonzero(counts)),
               high_tail_counts=dict(Ex_ge_8_5=int(np.sum(values[:, 0] >= 8.5)),
                                     Ey_ge_8_5=int(np.sum(values[:, 1] >= 8.5)),
                                     both_ge_8_5=int(np.sum(np.all(values >= 8.5, axis=1))),
                                     Ex_ge_9=int(np.sum(values[:, 0] >= 9)),
                                     Ey_ge_9=int(np.sum(values[:, 1] >= 9)),
                                     both_ge_9=int(np.sum(np.all(values >= 9, axis=1)))),
               anisotropy_counts=dict(Ex_over_Ey_ge_1_5=int(np.sum(values[:, 0]/values[:, 1] >= 1.5)),
                                      Ey_over_Ex_ge_1_5=int(np.sum(values[:, 1]/values[:, 0] >= 1.5))))
    return out


def compare_coverage(base_labels, added_labels, base_fractions, added_fractions):
    """Describe actual point coverage without claiming a complete feasible set.

    Probe target distances are selection diagnostics, not an independent test
    set or a measure of coverage over all attainable cat geometries.
    """
    base, new = _values(base_labels), _values(added_labels)
    base_fractions = np.asarray(base_fractions, dtype=float)
    added_fractions = np.asarray(added_fractions, dtype=float)
    if len(base) != len(base_fractions) or len(new) != len(added_fractions):
        raise ValueError("Fractions and labels must align.")
    if not len(base):
        raise ValueError("An immutable nonempty base is required.")
    combined = np.vstack([base, new])
    fractions = np.concatenate([base_fractions, added_fractions])
    before = _distribution(base, base_fractions)
    after = _distribution(combined, fractions)
    fine_before, fine_size = _fine_bins(base)
    fine_new, _ = _fine_bins(new)
    new_fine = sorted(set(fine_new.tolist()) - set(fine_before.tolist()))
    new_coarse = sorted(set(ms.property_bins(new).tolist()) - set(ms.property_bins(base).tolist()))
    old_tree, full_tree = cKDTree(base), cKDTree(combined)
    old_distance = old_tree.query(PROBE_TARGETS, p=np.inf)[0]
    full_distance = full_tree.query(PROBE_TARGETS, p=np.inf)[0]
    probes = []
    for i, target in enumerate(PROBE_TARGETS):
        probes.append(dict(target_Ex=float(target[0]), target_Ey=float(target[1]),
                           nearest_linf_before=float(old_distance[i]), nearest_linf_after=float(full_distance[i]),
                           count_within_0_25_before=len(old_tree.query_ball_point(target, .25, p=np.inf)),
                           count_within_0_25_after=len(full_tree.query_ball_point(target, .25, p=np.inf)),
                           count_within_0_5_before=len(old_tree.query_ball_point(target, .5, p=np.inf)),
                           count_within_0_5_after=len(full_tree.query_ball_point(target, .5, p=np.inf))))
    distances = old_tree.query(new, p=np.inf)[0] if len(new) else np.array([])
    return dict(before=before, after=after, added_count=len(new),
                new_observed_coarse_bins=new_coarse,
                new_observed_fine_bins=[dict(flat_id=int(i), x_bin=int(i % fine_size), y_bin=int(i//fine_size),
                                            Ex_range=[float(1+.5*(i % fine_size)), float(1+.5*(i % fine_size+1))],
                                            Ey_range=[float(1+.5*(i//fine_size)), float(1+.5*(i//fine_size+1))]) for i in new_fine],
                added_distance_to_fixed_base_linf=ms._summary(distances),
                added_points_at_least_0_25_from_base=int(np.sum(distances >= .25)),
                probe_target_neighborhoods=probes,
                caveat="Grid occupancy depends on bin width/origin. Nonempty bins are observations, not certified feasible interiors. Probe targets are predeclared diagnostics, not an independent test set or complete feasible-domain coverage.")


def _actual_group(values):
    result = np.full(len(values), 3, dtype=int)
    result[(values[:, 1]/values[:, 0] >= 1.5) & (values[:, 1] >= 4.5)] = 2
    result[(values[:, 0]/values[:, 1] >= 1.35) & (values[:, 0] >= 3.5)] = 1
    result[np.max(values, axis=1) >= 8.5] = 0
    return result


def select_additions(base_samples, base_labels, samples, labels, target=250, seed=1143186):
    """Return reviewed candidate indices while preserving all fixed base images.

    Quotas prefer 40% high-modulus, 30% x-dominant, 20% y-dominant and 10%
    exploratory observations, classified by actual FEM labels. Within groups,
    greedy choice favors unrepresented mechanics, distinct local silhouettes,
    low existing local density, and closeness to the requested target. Quotas
    are soft: empty groups cannot be fabricated. Every source is retained once.
    """
    if len(base_samples) != len(base_labels) or len(samples) != len(labels):
        raise ValueError("Samples and FEM labels must be aligned.")
    if not len(base_samples) or not isinstance(target, (int, np.integer)) or target < 1:
        raise ValueError("A nonempty fixed base and positive integer target are required.")
    rng = np.random.default_rng(seed)
    ties = rng.random(len(samples))
    base_values = _values(base_labels)
    ms.property_bins(base_values)
    base_digests = {_digest(s) for s in base_samples}
    base_seeds = {_seed(s) for s in base_samples}
    if len(base_digests) != len(base_samples) or len(base_seeds) != len(base_samples):
        raise ValueError("Immutable base has exact/source duplicates.")
    valid, exclusions, values_list, quality_list = [], Counter(), [], []
    for i, (sample, label) in enumerate(zip(samples, labels)):
        pixels = np.asarray(ms._get(sample, "pixels"))
        if pixels.dtype != np.uint8 or pixels.shape != (224, 224) or not np.isin(pixels, [0, 1]).all():
            exclusions["invalid_binary_image"] += 1; continue
        if pixels.min() == pixels.max():
            exclusions["blank_image"] += 1; continue
        spec = ms._get(sample, "spec")
        if ms._get(spec, "background") not in (0, 1) or ms._get(spec, "seed") is None:
            exclusions["missing_source_or_polarity"] += 1; continue
        if ms._get(sample, "issue", ""):
            exclusions["generation_issue"] += 1; continue
        if label is None or not _truth(ms._get(label, "fem_valid", True)):
            exclusions["missing_or_invalid_fem"] += 1; continue
        try:
            value = _values([label])[0]
            ms.property_bins(value)
        except (ValueError, TypeError, KeyError):
            exclusions["invalid_moduli"] += 1; continue
        fraction = float((pixels == 0).mean())
        if np.any(value < 1/(1-.9*fraction)-1e-5) or np.any(value > 1+9*fraction+1e-5):
            exclusions["violates_fixed_fraction_necessary_bound"] += 1; continue
        if _digest(sample) in base_digests:
            exclusions["exact_duplicate_of_base"] += 1; continue
        if _seed(sample) in base_seeds:
            exclusions["source_seed_already_in_base"] += 1; continue
        margin, logit = float(ms._get(sample, "cat_margin", 0)), float(ms._get(sample, "cat_logit", 0))
        if not np.isfinite([margin, logit]).all():
            exclusions["nonfinite_semantic_score"] += 1; continue
        valid.append(i); values_list.append(value)
        quality_list.append(np.tanh(margin/10)+.05*np.tanh(logit/10))
    if not valid:
        return [], dict(version=SELECTION_VERSION, target=int(target), target_met=False, selected_count=0,
                        shortfall=int(target), input_count=len(samples), exclusions=dict(exclusions),
                        base_count=len(base_samples), base_preserved=True), {}

    nbase, n = len(base_samples), len(valid)
    all_samples = list(base_samples) + [samples[i] for i in valid]
    values = np.vstack([base_values, np.asarray(values_list)])
    descriptors = [ms._descriptor(s) for s in all_samples]
    masks = np.stack([d[0] for d in descriptors])
    hashes = np.stack([d[1] for d in descriptors])
    areas = np.array([d[2] for d in descriptors])
    features = [d[3] for d in descriptors]
    fractions = np.array([d["black_fraction"] for d in features])
    polarity = np.array([d["background"] for d in features])
    graph, graph_stats = ms._conflicts(values, fractions, polarity, masks, hashes, areas)
    old_conflicts = int(graph[:nbase, :nbase].nnz//2)
    if old_conflicts:
        raise ValueError(f"The fixed base has {old_conflicts} declared near-duplicate pairs; cannot remove them in append-only selection.")
    available = np.asarray(graph[nbase:, :nbase].getnnz(axis=1) == 0).ravel()
    exclusions["near_duplicate_of_base"] += int((~available).sum())
    candidate_values = values[nbase:]
    groups = _actual_group(candidate_values)
    seeds = np.array([_seed(s) for s in all_samples[nbase:]], dtype=np.int64)
    digests = np.array([_digest(s) for s in all_samples[nbase:]])
    quality = np.asarray(quality_list)
    base_tree = cKDTree(base_values)
    distance = base_tree.query(candidate_values, p=np.inf)[0]
    neighbors = base_tree.query_ball_point(candidate_values, NEIGHBOR_RADIUS, p=np.inf)
    local_count = np.array([len(a) for a in neighbors], dtype=int)
    geometry_novelty = np.ones(n)
    for i, near in enumerate(neighbors):
        if len(near):
            similarity, _, _ = ms._compare(np.full(len(near), nbase+i, dtype=int), np.asarray(near), masks, hashes, areas)
            geometry_novelty[i] = 1 - float(similarity.max())
    target_distances = np.full(n, np.nan)
    for j, input_index in enumerate(valid):
        diagnostics = _diagnostics(samples[input_index])
        tx, ty = diagnostics.get("target_Ex"), diagnostics.get("target_Ey")
        if tx is not None and ty is not None and np.isfinite([float(tx), float(ty)]).all():
            target_distances[j] = np.max(np.abs(candidate_values[j]-[float(tx), float(ty)]))
    target_bonus = np.zeros(n)
    known_targets = np.isfinite(target_distances)
    target_bonus[known_targets] = .10*np.exp(-target_distances[known_targets]/.25)
    chosen, selection_trace = [], []
    group_counts = np.zeros(4, dtype=int)
    blocked_near = blocked_source = blocked_pixel = 0

    def add(index, reason):
        nonlocal blocked_near, blocked_source, blocked_pixel
        index = int(index)
        global_index = nbase+index
        selection_trace.append(dict(input_index=valid[index], reason=reason, actual_group=GROUPS[groups[index]],
                                    nearest_existing_linf=float(distance[index]),
                                    existing_neighbors_within_0_5=int(local_count[index]),
                                    local_silhouette_novelty=float(geometry_novelty[index]),
                                    requested_target_linf_error=float(target_distances[index]) if np.isfinite(target_distances[index]) else None))
        chosen.append(index); group_counts[groups[index]] += 1
        available[index] = False
        same_source = seeds == seeds[index]
        blocked_source += int((available & same_source).sum())
        available[same_source] = False
        same_pixels = digests == digests[index]
        blocked_pixel += int((available & same_pixels).sum())
        available[same_pixels] = False
        conflict = graph.indices[graph.indptr[global_index]:graph.indptr[global_index+1]]
        conflict = conflict[conflict >= nbase]-nbase
        blocked_near += int(available[conflict].sum())
        available[conflict] = False
        deltas = np.max(np.abs(candidate_values-candidate_values[index]), axis=1)
        distance[:] = np.minimum(distance, deltas)
        local_count[:] += deltas <= NEIGHBOR_RADIUS
        close = np.flatnonzero(available & (deltas <= NEIGHBOR_RADIUS))
        if len(close):
            similarity, _, _ = ms._compare(np.full(len(close), global_index), nbase+close, masks, hashes, areas)
            geometry_novelty[close] = np.minimum(geometry_novelty[close], 1-similarity)

    # Preserve true extensions, rather than re-reserving old low-fraction extremes.
    extrema = [("new_Ex_max", candidate_values[:, 0], base_values[:, 0].max(), .05),
               ("new_Ey_max", candidate_values[:, 1], base_values[:, 1].max(), .05),
               ("new_Ex_over_Ey_max", candidate_values[:, 0]/candidate_values[:, 1],
                np.max(base_values[:, 0]/base_values[:, 1]), .03),
               ("new_Ey_over_Ex_max", candidate_values[:, 1]/candidate_values[:, 0],
                np.max(base_values[:, 1]/base_values[:, 0]), .03)]
    for name, metric, old_max, minimum_improvement in extrema:
        eligible = np.flatnonzero(available & (metric >= old_max+minimum_improvement))
        if len(eligible) and len(chosen) < target:
            best = eligible[np.lexsort((ties[np.asarray(valid)[eligible]], -quality[eligible], -metric[eligible]))[0]]
            add(best, name)
    while len(chosen) < target and available.any():
        capacities = np.bincount(groups[available], minlength=4)
        active = np.flatnonzero(capacities)
        group = min(active, key=lambda g: (group_counts[g]/GROUP_WEIGHTS[g], g))
        options = np.flatnonzero(available & (groups == group))
        scores = (.45*np.minimum(distance[options]/.5, 1) + .35*geometry_novelty[options]
                  + .20/(1+local_count[options]) + target_bonus[options]
                  + 1e-4*quality[options] + 1e-8*ties[np.asarray(valid)[options]])
        add(options[int(np.argmax(scores))], "targeted_local_coverage_and_geometry")
    chosen = np.asarray(chosen, dtype=int)
    selected_inputs = [valid[i] for i in chosen]
    selected_global = nbase+chosen

    # Every new-base and new-new pair is audited, including geometrically similar
    # pairs deliberately retained because their properties/material polarity differ.
    shape_near = same_shape = joint_near = 0
    examples = []
    for position, left in enumerate(selected_global):
        right_all = np.concatenate([np.arange(nbase), selected_global[:position]])
        for start in range(0, len(right_all), ms.PAIR_BATCH):
            right = right_all[start:start+ms.PAIR_BATCH]
            first = np.full(len(right), left)
            iou, phash, similar = ms._compare(first, right, masks, hashes, areas)
            same = polarity[left] == polarity[right]
            mechanical_delta = np.max(np.abs(values[left]-values[right]), axis=1)
            local = similar & same & (mechanical_delta <= ms.MECHANICAL_TOLERANCE)
            local &= np.abs(fractions[left]-fractions[right]) <= ms.FRACTION_TOLERANCE
            shape_near += int(similar.sum()); same_shape += int((similar & same).sum()); joint_near += int(local.sum())
            for j in np.flatnonzero(similar)[:3]:
                examples.append(dict(new_input_index=valid[left-nbase],
                                     other_base_index=int(right[j]) if right[j] < nbase else None,
                                     other_new_input_index=valid[right[j]-nbase] if right[j] >= nbase else None,
                                     silhouette_iou=float(iou[j]), phash_distance=int(phash[j]),
                                     mechanical_linf=float(mechanical_delta[j]), same_polarity=bool(same[j])))
            examples.sort(key=lambda e: -e["silhouette_iou"])
            del examples[20:]
    if joint_near:
        raise AssertionError(f"Final append set contains {joint_near} prohibited near pairs.")
    selected_digests = [digests[i] for i in chosen]
    selected_seeds = [int(seeds[i]) for i in chosen]
    if len(set(selected_digests)) != len(chosen) or len(set(selected_seeds)) != len(chosen):
        raise AssertionError("Final append set repeated pixels or source seeds.")

    chosen_labels = [labels[i] for i in selected_inputs]
    coverage = compare_coverage(base_labels, chosen_labels, fractions[:nbase], fractions[selected_global])
    stats = dict(version=SELECTION_VERSION, seed=int(seed), target=int(target), target_met=len(chosen) == target,
                 base_count=nbase, base_preserved=True, input_count=len(samples), eligible_variants=n,
                 eligible_independent_sources=int(len(set(seeds.tolist()))), selected_count=len(chosen),
                 shortfall=max(0, int(target-len(chosen))), exclusions=dict(exclusions),
                 near_conflict_candidates_blocked=blocked_near, same_source_variants_blocked=blocked_source,
                 exact_duplicate_variants_blocked=blocked_pixel,
                 selected_input_indices=selected_inputs, selected_source_seeds=selected_seeds,
                 actual_group_counts=dict(zip(GROUPS, group_counts.tolist())),
                 requested_group_counts=dict(Counter(str(_diagnostics(samples[i]).get("target_group", "unavailable")) for i in selected_inputs)),
                 target_error=ms._summary(target_distances[chosen][np.isfinite(target_distances[chosen])]),
                 target_hits_linf_le_0_25=int(np.sum(target_distances[chosen] <= .25)),
                 selection_trace=selection_trace, coverage=coverage,
                 method="append-only actual-property group allocation, mechanics-distance and local silhouette novelty, one variant per new independent source",
                 group_allocation_caveat="40/30/20/10 are soft priorities, not claims of uniform feasibility; sparse-group shortages are reported through actual counts.")
    pair_audit = dict(base_count=nbase, added_count=len(chosen),
                      audited_new_base_pairs=nbase*len(chosen), audited_new_new_pairs=len(chosen)*(len(chosen)-1)//2,
                      exact_duplicate_pairs_involving_new=0, repeated_source_seeds_involving_new=0,
                      local_near_duplicate_pairs_involving_new=joint_near,
                      shape_near_pairs_involving_new_all_polarities=shape_near,
                      shape_near_pairs_involving_new_same_polarity=same_shape,
                      most_shape_similar_examples=examples,
                      selected_features=[dict(input_index=valid[i], **features[nbase+i]) for i in chosen],
                      initial_conflict_graph=graph_stats,
                      rule="same background AND max(|dEx|,|dEy|)<=0.12 AND |dblack_fraction|<=0.025 AND (mirror-aware silhouette IoU>=0.91 OR (IoU>=0.85 AND pHash<=6 in same comparison)); symmetric pair order")
    return selected_inputs, stats, pair_audit
