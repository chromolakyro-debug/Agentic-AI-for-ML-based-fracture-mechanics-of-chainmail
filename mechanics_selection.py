"""Select diverse cat specimens with balanced joint apparent-modulus coverage.

``select(samples, labels, target)`` accepts Sample-like objects and equally long,
aligned FEM dictionaries containing Ex_app and Ey_app. Returned indices address
the original inputs. It never changes pixels, labels, or files. A short candidate
pool returns fewer than ``target`` with ``stats['target_met'] == False``; it is
never padded with duplicates. Property bins describe observed candidate support,
not a proof that their interiors or the entire [1, 10]^2 square are feasible.

NumPy, SciPy and Pillow are the only dependencies. Pair comparisons use bounded
batches and a sparse conflict graph, rather than an N x N x image-size tensor.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from collections.abc import Mapping

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.fft import dctn
from scipy.spatial import cKDTree

SEED = 1142114
BIN_COUNT = 6
MODULUS_MIN, MODULUS_MAX = 1., 10.
MECHANICAL_TOLERANCE = .12
FRACTION_TOLERANCE = .025
POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
PAIR_BATCH = 4096


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def property_bins(values):
    """Fixed 6x6 bins; flat IDs are y-bin * 6 + x-bin, including the value 10."""
    values = np.asarray(values, dtype=float).reshape(-1, 2)
    if not np.isfinite(values).all():
        raise ValueError("Properties must be finite.")
    if np.any(values < MODULUS_MIN - 1e-6) or np.any(values > MODULUS_MAX + 1e-6):
        raise ValueError("This selector is configured for Es=1, Eh=10 and common nu.")
    xy = np.floor((values - MODULUS_MIN) / (MODULUS_MAX - MODULUS_MIN) * BIN_COUNT)
    xy = np.clip(xy.astype(int), 0, BIN_COUNT - 1)
    return xy[:, 1] * BIN_COUNT + xy[:, 0]


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0}
    return dict(count=int(values.size), minimum=float(values.min()),
                p10=float(np.quantile(values, .1)), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()),
                mean=float(values.mean()))


def _descriptor(sample):
    pixels = np.asarray(_get(sample, "pixels"), dtype=np.uint8)
    background = int(_get(_get(sample, "spec"), "background"))
    foreground = pixels != background
    ys, xs = np.nonzero(foreground)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    width, height = int(x1 - x0), int(y1 - y0)
    cropped = foreground[y0:y1, x0:x1]
    # Keep aspect ratio when removing scale/translation. Filling holes affects
    # the silhouette metric only; pHash retains broad internal drawing details.
    scale = 32. / max(width, height)
    dimensions = (max(1, round(width * scale)), max(1, round(height * scale)))
    offset = ((32 - dimensions[0]) // 2, (32 - dimensions[1]) // 2)
    shape_canvas = Image.new("L", (32, 32), 0)
    shape_canvas.paste(Image.fromarray(ndimage.binary_fill_holes(cropped).astype(np.uint8) * 255)
                       .resize(dimensions, Image.Resampling.NEAREST), offset)
    ink_canvas = Image.new("L", (32, 32), 255)
    ink_canvas.paste(Image.fromarray((~cropped).astype(np.uint8) * 255)
                     .resize(dimensions, Image.Resampling.LANCZOS), offset)
    shape = np.asarray(shape_canvas) > 0
    ink = np.asarray(ink_canvas, dtype=float)
    ox, oy = offset
    rw, rh = dimensions
    shape_content = shape[oy:oy + rh, ox:ox + rw]
    ink_content = ink[oy:oy + rh, ox:ox + rw]
    shapes, hashes = [], []
    # Reflections preserve directional apparent moduli under this particular
    # loading protocol. Transposition is excluded because it exchanges Ex/Ey.
    for row_flip, col_flip in ((False, False), (False, True), (True, False), (True, True)):
        # Reflect before centering: odd amounts of padding otherwise introduce
        # an artificial one-pixel shift between exact reflected drawings.
        s = np.zeros((32, 32), dtype=bool)
        image = np.full((32, 32), 255., dtype=float)
        s[oy:oy + rh, ox:ox + rw] = shape_content[::-1 if row_flip else 1, ::-1 if col_flip else 1]
        image[oy:oy + rh, ox:ox + rw] = ink_content[::-1 if row_flip else 1, ::-1 if col_flip else 1]
        coefficients = dctn(image, norm="ortho")[:8, :8].ravel()
        bits = coefficients > np.median(coefficients[1:])
        bits[0] = False
        shapes.append(np.packbits(s.ravel()))
        hashes.append(np.packbits(bits))
    hard = pixels == 0
    components, count = ndimage.label(hard, structure=ndimage.generate_binary_structure(2, 1))
    areas = np.bincount(components.ravel())[1:]
    span_x = np.intersect1d(np.unique(components[:, 0]), np.unique(components[:, -1]))
    span_y = np.intersect1d(np.unique(components[0]), np.unique(components[-1]))
    features = {
        "background": background, "black_fraction": float(hard.mean()),
        "foreground_bbox_width_fraction": width / pixels.shape[1],
        "foreground_bbox_height_fraction": height / pixels.shape[0],
        "foreground_bbox_aspect_ratio": width / height,
        "foreground_bbox_fill_fraction": float(cropped.mean()),
        "foreground_centroid_x": float((xs.mean() + .5) / pixels.shape[1]),
        "foreground_centroid_y": float((ys.mean() + .5) / pixels.shape[0]),
        "hard_edge_left": float(hard[:, 0].mean()),
        "hard_edge_right": float(hard[:, -1].mean()),
        "hard_edge_top": float(hard[0].mean()),
        "hard_edge_bottom": float(hard[-1].mean()),
        "hard_spans_x_4connected": bool(np.any(span_x > 0)),
        "hard_spans_y_4connected": bool(np.any(span_y > 0)),
        "hard_component_count_4connected": int(count),
        "hard_largest_component_share": float(areas.max() / hard.sum()) if areas.size else 0.,
        "minimum_hard_column_fraction": float(hard.mean(axis=0).min()),
        "minimum_hard_row_fraction": float(hard.mean(axis=1).min()),
        "foreground_touches_boundary": bool(foreground[0].any() or foreground[-1].any()
                                             or foreground[:, 0].any() or foreground[:, -1].any()),
    }
    return np.stack(shapes), np.stack(hashes), int(shape.sum()), features


def _compare(left, right, masks, hashes, areas):
    """Pair arrays -> symmetric reflection-aware IoU, pHash distance, shape-near.

    The combined IoU/pHash criterion must hold for the SAME comparison. Reflect
    either member while holding the other fixed. A reflected pHash is recomputed
    and median-thresholded, so reflecting its DCT is not a fixed permutation of
    the original hash bits. Comparing only A versus reflected B can therefore
    differ from comparing B versus reflected A. Both orders are necessary to
    make this an unordered-pair criterion. The reported extrema need not occur
    in the same comparison.
    """
    left, right = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    max_iou = np.zeros(len(left), dtype=float)
    min_hash = np.full(len(left), 64, dtype=np.int16)
    near = np.zeros(len(left), dtype=bool)
    comparisons = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0))
    for left_reflection, right_reflection in comparisons:
        intersection = POPCOUNT[np.bitwise_and(masks[left, left_reflection], masks[right, right_reflection])].sum(axis=1)
        iou = intersection / np.maximum(areas[left] + areas[right] - intersection, 1)
        distance = POPCOUNT[np.bitwise_xor(hashes[left, left_reflection], hashes[right, right_reflection])].sum(axis=1)
        max_iou = np.maximum(max_iou, iou)
        min_hash = np.minimum(min_hash, distance)
        near |= (iou >= .91) | ((iou >= .85) & (distance <= 6))
    return max_iou, min_hash, near


def _pair_batches(indices):
    indices = np.asarray(indices, dtype=int)
    for position, left in enumerate(indices[:-1]):
        rest = indices[position + 1:]
        for start in range(0, len(rest), PAIR_BATCH):
            right = rest[start:start + PAIR_BATCH]
            yield np.full(len(right), left, dtype=int), right


def _conflicts(values, fractions, polarities, masks, hashes, areas):
    tree = cKDTree(values)
    edge_batches = []
    close_count = fraction_count = eligible_count = 0
    examples = []
    for left in range(len(values)):
        neighbors = np.asarray(tree.query_ball_point(values[left], MECHANICAL_TOLERANCE, p=np.inf), dtype=int)
        neighbors = np.sort(neighbors[neighbors > left])
        close_count += len(neighbors)
        neighbors = neighbors[np.abs(fractions[left] - fractions[neighbors]) <= FRACTION_TOLERANCE]
        fraction_count += len(neighbors)
        # Morphological normalization does not make inverse material maps duplicates.
        neighbors = neighbors[polarities[left] == polarities[neighbors]]
        eligible_count += len(neighbors)
        for start in range(0, len(neighbors), PAIR_BATCH):
            right = neighbors[start:start + PAIR_BATCH]
            first = np.full(len(right), left, dtype=int)
            iou, phash, near = _compare(first, right, masks, hashes, areas)
            if near.any():
                edge_batches.append(np.column_stack((first[near], right[near])))
                for offset in np.flatnonzero(near)[:max(0, 20 - len(examples))]:
                    examples.append({"left": int(left), "right": int(right[offset]),
                                     "max_reflection_iou": float(iou[offset]),
                                     "min_reflection_phash_distance": int(phash[offset]),
                                     "max_modulus_difference": float(np.max(np.abs(values[left] - values[right[offset]])))})
    edges = np.concatenate(edge_batches) if edge_batches else np.empty((0, 2), dtype=int)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    graph = sparse.coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                              shape=(len(values), len(values))).tocsr()
    return graph, {
        "total_unordered_pairs": len(values) * (len(values) - 1) // 2,
        "mechanically_close_pairs_linf_le_0_12": int(close_count),
        "also_fraction_close_pairs": int(fraction_count),
        "also_same_phase_polarity_pairs": int(eligible_count),
        "local_near_duplicate_pairs": int(len(edges)), "examples": examples,
    }


def _distribution(values, bins, fractions, reference_support):
    counts = np.bincount(bins, minlength=BIN_COUNT ** 2)
    supported = counts[reference_support]
    probabilities = supported / max(int(supported.sum()), 1)
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    support_count = len(reference_support)
    out = {
        "count": len(values), "bin_counts_y_rows_x_columns": counts.reshape(BIN_COUNT, BIN_COUNT).tolist(),
        "observed_nonempty_bins": int(np.count_nonzero(counts)),
        "reference_candidate_support_bins": int(support_count),
        "count_cv_over_reference_support": float(supported.std() / supported.mean()) if supported.sum() else None,
        "normalized_entropy_over_reference_support": entropy / np.log(support_count) if support_count > 1 else (1. if len(values) else None),
        "effective_occupied_bin_count": float(np.exp(entropy)) if len(values) else 0.,
        "black_fraction_histogram_edges": np.linspace(0, 1, 11).tolist(),
        "black_fraction_histogram_counts": np.histogram(fractions, bins=np.linspace(0, 1, 11))[0].tolist(),
        "black_fraction": _summary(fractions),
        "Ex_app": _summary(values[:, 0]), "Ey_app": _summary(values[:, 1]),
        "Ex_over_Ey": _summary(values[:, 0] / values[:, 1]),
    }
    if len(values) >= 2:
        tree = cKDTree(values)
        out["nearest_mechanical_distance"] = {
            "normalized_euclidean": _summary(tree.query(values, k=2)[0][:, 1] / 9.),
            "modulus_units_linf": _summary(tree.query(values, k=2, p=np.inf)[0][:, 1]),
        }
    else:
        out["nearest_mechanical_distance"] = None
    return out


def _feature_summary(features):
    if not features:
        return {}
    out = {}
    for key in features[0]:
        values = [item[key] for item in features]
        if isinstance(values[0], bool):
            out[key] = {"true_count": int(sum(values)), "fraction": float(np.mean(values))}
        elif key == "background":
            out[key] = {str(k): int(v) for k, v in Counter(values).items()}
        else:
            out[key] = _summary(values)
    return out


def balanced_minibatch_weights(bins):
    """Inverse occupied-bin frequency; sum equals the selected sample count."""
    bins = np.asarray(bins, dtype=int)
    if not len(bins):
        return []
    counts = np.bincount(bins, minlength=BIN_COUNT ** 2)
    weights = 1. / counts[bins]
    return (weights * len(weights) / weights.sum()).tolist()


def select(samples, labels, target=1000, seed=SEED, reserve_extrema=True):
    """Return (original_input_indices, stats, geometry_pair_metrics).

    Up to eight distinct specimens first preserve observed hard-fraction and
    property extrema. Fair filling then allocates one specimen at a time to the
    least-filled available property bin, breaking ties in favor of scarce bins.
    Inside a bin, farthest-first selection favors unlike morphologies.
    Sparse conflict edges suppress only jointly similar geometry AND mechanics.
    Opposite background polarities are excluded from the near-duplicate rule.
    """
    if len(samples) != len(labels):
        raise ValueError("samples and labels must have equal length and matching order.")
    if not isinstance(target, (int, np.integer)) or target < 1:
        raise ValueError("target must be a positive integer.")
    rng = np.random.default_rng(seed)
    ties = rng.random(len(samples))
    valid, reasons = [], Counter()
    qualities = np.zeros(len(samples))
    properties = {}
    for index, (sample, label) in enumerate(zip(samples, labels)):
        pixels = np.asarray(_get(sample, "pixels"))
        spec = _get(sample, "spec")
        if pixels.shape != (224, 224) or not np.isin(pixels, [0, 1]).all():
            reasons["invalid_binary_image"] += 1; continue
        if _get(spec, "background") not in (0, 1) or _get(spec, "seed") is None:
            reasons["missing_polarity_or_source_seed"] += 1; continue
        if not pixels.min() < pixels.max():
            reasons["blank_image"] += 1; continue
        if _get(sample, "issue", ""):
            reasons["generation_issue"] += 1; continue
        if label is None or not _get(label, "fem_valid", True):
            reasons["missing_or_invalid_fem"] += 1; continue
        try:
            value = np.array([_get(label, "Ex_app"), _get(label, "Ey_app")], dtype=float)
            property_bins(value)
        except (ValueError, TypeError):
            reasons["invalid_moduli"] += 1; continue
        margin, logit = float(_get(sample, "cat_margin", 0.)), float(_get(sample, "cat_logit", 0.))
        if not np.isfinite([margin, logit]).all():
            reasons["invalid_semantic_score"] += 1; continue
        qualities[index] = np.tanh(margin / 10.) + .05 * np.tanh(logit / 10.)
        properties[index] = value
        valid.append(index)
    # Choose the highest semantic-score occurrence before balancing; neither phase
    # inversion nor reflection is used to construct the exact-pixel fingerprint.
    ranked = sorted(valid, key=lambda i: (-qualities[i], ties[i], i))
    seen_pixels, seen_seeds, kept = set(), set(), []
    for index in ranked:
        sample = samples[index]
        fingerprint = hashlib.sha256(np.asarray(_get(sample, "pixels"), dtype=np.uint8).tobytes()).digest()
        source_seed = int(_get(_get(sample, "spec"), "seed"))
        if fingerprint in seen_pixels:
            reasons["exact_pixel_duplicate"] += 1; continue
        if source_seed in seen_seeds:
            reasons["repeated_source_seed"] += 1; continue
        seen_pixels.add(fingerprint); seen_seeds.add(source_seed); kept.append(index)
    kept.sort()
    if not kept:
        return [], {"target": int(target), "target_met": False, "selected_count": 0,
                    "shortfall": int(target), "input_count": len(samples), "exclusions": dict(reasons)}, {}
    values = np.stack([properties[i] for i in kept])
    descriptors = [_descriptor(samples[i]) for i in kept]
    masks = np.stack([d[0] for d in descriptors])
    hashes = np.stack([d[1] for d in descriptors])
    areas = np.array([d[2] for d in descriptors])
    features = [d[3] for d in descriptors]
    fractions = np.array([f["black_fraction"] for f in features])
    polarities = np.array([f["background"] for f in features])
    bins = property_bins(values)
    support = np.flatnonzero(np.bincount(bins, minlength=BIN_COUNT ** 2))
    graph, before_pairs = _conflicts(values, fractions, polarities, masks, hashes, areas)
    for pair in before_pairs["examples"]:
        pair["left_input_index"] = kept[pair.pop("left")]
        pair["right_input_index"] = kept[pair.pop("right")]
    embeddings = [_get(samples[i], "embedding") for i in kept]
    use_embedding = all(e is not None and np.asarray(e).ndim == 1 for e in embeddings)
    use_embedding = use_embedding and len({np.asarray(e).size for e in embeddings}) == 1
    embedding_matrix = None
    if use_embedding:
        embedding_matrix = np.asarray(embeddings, dtype=float)
        norms = np.linalg.norm(embedding_matrix, axis=1)
        if np.isfinite(embedding_matrix).all() and np.all(norms > 0):
            embedding_matrix /= norms[:, None]
        else:
            embedding_matrix = None
    available = np.ones(len(kept), dtype=bool)
    novelty = np.ones(len(kept))
    selected, selected_counts = [], np.zeros(BIN_COUNT ** 2, dtype=int)
    bin_ties = rng.random(BIN_COUNT ** 2)
    quality = qualities[kept]
    blocked_count = 0

    def add_selection(chosen):
        nonlocal blocked_count
        chosen = int(chosen)
        chosen_bin = bins[chosen]
        selected.append(chosen); selected_counts[chosen_bin] += 1
        available[chosen] = False
        neighbors = graph.indices[graph.indptr[chosen]:graph.indptr[chosen + 1]]
        blocked_count += int(available[neighbors].sum())
        available[neighbors] = False
        remaining = np.flatnonzero(available & (bins == chosen_bin))
        for start in range(0, len(remaining), PAIR_BATCH):
            right = remaining[start:start + PAIR_BATCH]
            iou, _, _ = _compare(np.full(len(right), chosen), right, masks, hashes, areas)
            distance = 1. - iou
            if embedding_matrix is not None:
                cosine = np.clip(embedding_matrix[right] @ embedding_matrix[chosen], -1, 1)
                distance = .9 * distance + .1 * (1. - cosine) / 2.
            novelty[right] = np.minimum(novelty[right], distance)

    extrema_decisions = []
    if reserve_extrema:
        # Fraction endpoints are considered first because a 1.5-wide modulus
        # bin can otherwise discard the entire rare low/high-fraction tail.
        criteria = (("black_fraction", fractions), ("Ex_app", values[:, 0]),
                    ("Ey_app", values[:, 1]), ("Ex_over_Ey", values[:, 0] / values[:, 1]))
        local_ties = ties[np.asarray(kept)]
        for name, metric in criteria:
            for direction, sign in (("min", 1.), ("max", -1.)):
                order = np.lexsort((local_ties, -quality, sign * metric))
                desired = int(order[0])
                record = {"criterion": name + "_" + direction,
                          "candidate_input_index": kept[desired],
                          "original_extreme_value": float(metric[desired])}
                represented = np.zeros(len(kept), dtype=bool)
                represented[selected] = True
                # An existing selected neighbor can be closer to the missing
                # extreme than every unselected candidate. Do not waste another
                # quota slot on a worse substitute in that case.
                eligible = order[(available | represented)[order]]
                if not len(eligible):
                    chosen, status = None, "no_remaining_candidate"
                else:
                    chosen = int(eligible[0])
                    if represented[chosen]:
                        status = ("already_represented" if abs(metric[chosen] - metric[desired]) <= 1e-12
                                  else "represented_by_near_conflict")
                    elif len(selected) >= target:
                        chosen, status = None, "target_capacity_exhausted"
                    else:
                        status = "reserved" if chosen == desired else "near_conflict_substitution"
                        add_selection(chosen)
                record.update(status=status,
                              selected_input_index=None if chosen is None else kept[chosen],
                              selected_value=None if chosen is None else float(metric[chosen]))
                extrema_decisions.append(record)
    reserved_count = len(selected)
    while len(selected) < target and available.any():
        capacities = np.bincount(bins[available], minlength=BIN_COUNT ** 2)
        available_bins = np.flatnonzero(capacities)
        chosen_bin = min(available_bins, key=lambda b: (selected_counts[b], capacities[b], bin_ties[b]))
        options = np.flatnonzero(available & (bins == chosen_bin))
        scores = novelty[options] + quality[options] * 1e-4 + ties[np.asarray(kept)[options]] * 1e-8
        add_selection(options[np.argmax(scores)])
    selected = np.asarray(selected, dtype=int)
    chosen_input = [kept[i] for i in selected]
    selected_bins = bins[selected]
    # Final audit includes shape-similar pairs with different mechanics; these
    # are informative samples, not automatically redundant training examples.
    shape_near_count = same_polarity_shape_count = local_near_count = 0
    top_pairs = []
    audit_conflicts = []
    for left, right in _pair_batches(selected):
        iou, phash, shape_near = _compare(left, right, masks, hashes, areas)
        mechanical_delta = np.max(np.abs(values[left] - values[right]), axis=1)
        same_polarity = polarities[left] == polarities[right]
        local_near = shape_near & same_polarity & (mechanical_delta <= MECHANICAL_TOLERANCE)
        local_near &= np.abs(fractions[left] - fractions[right]) <= FRACTION_TOLERANCE
        shape_near_count += int(shape_near.sum())
        same_polarity_shape_count += int((shape_near & same_polarity).sum())
        local_near_count += int(local_near.sum())
        for j in np.flatnonzero(local_near)[:max(0, 10 - len(audit_conflicts))]:
            reverse_iou, reverse_hash, reverse_near = _compare(right[j:j+1], left[j:j+1], masks, hashes, areas)
            audit_conflicts.append({"left_input_index": kept[left[j]], "right_input_index": kept[right[j]],
                                   "max_reflection_iou": float(iou[j]),
                                   "min_reflection_phash_distance": int(phash[j]),
                                   "reverse_iou": float(reverse_iou[0]),
                                   "reverse_phash": int(reverse_hash[0]),
                                   "reverse_shape_near": bool(reverse_near[0]),
                                   "graph_conflict": bool(graph[left[j], right[j]])})
        best = np.argsort(-iou, kind="stable")[:5]
        for j in best:
            top_pairs.append({"left_input_index": kept[left[j]], "right_input_index": kept[right[j]],
                              "max_reflection_iou": float(iou[j]),
                              "min_reflection_phash_distance": int(phash[j]),
                              "max_modulus_difference": float(mechanical_delta[j]),
                              "black_fraction_difference": float(abs(fractions[left[j]] - fractions[right[j]])),
                              "same_phase_polarity": bool(same_polarity[j])})
        top_pairs.sort(key=lambda p: (-p["max_reflection_iou"], p["left_input_index"], p["right_input_index"]))
        del top_pairs[20:]
    if local_near_count:
        raise AssertionError(f"Selected set contains {local_near_count} conflicts from the declared near-duplicate rule: {audit_conflicts}")
    weights = balanced_minibatch_weights(selected_bins)
    stats = {
        "seed": int(seed), "target": int(target), "target_met": len(selected) == target,
        "selected_count": len(selected), "shortfall": max(0, int(target - len(selected))),
        "input_count": len(samples), "eligible_unique_candidates": len(kept),
        "exclusions": dict(reasons), "near_conflict_candidates_blocked": blocked_count,
        "available_unselected_candidates": int(available.sum()),
        "selection_method": "optional observed-extrema reservation, then scarcity-aware fair fill of 6x6 joint bins and farthest-first morphology inside bins",
        "extrema_reservation_enabled": bool(reserve_extrema),
        "extrema_reserved_distinct_samples": reserved_count,
        "extrema_reservation_decisions": extrema_decisions,
        "bin_edges": np.linspace(MODULUS_MIN, MODULUS_MAX, BIN_COUNT + 1).tolist(),
        "bin_layout": "rows Ey_app, columns Ex_app; flat_id=y*6+x",
        "before": _distribution(values, bins, fractions, support),
        "after": _distribution(values[selected], selected_bins, fractions[selected], support),
        "feature_summary_before": _feature_summary(features),
        "feature_summary_after": _feature_summary([features[i] for i in selected]),
        "selected_input_indices": chosen_input,
        "selected_property_bin_ids": selected_bins.tolist(),
        "balanced_minibatch_weights": weights,
        "balanced_weight_sum": float(sum(weights)),
        "embedding_used_as_secondary_novelty": embedding_matrix is not None,
        "support_caveat": "Only observed candidate bins; neither bin interiors nor empty bins are certified feasible.",
    }
    pair_metrics = {
        "rule": "same background polarity AND max(|dEx|,|dEy|)<=0.12 AND |dblack_fraction|<=0.025 AND (reflection-aware silhouette IoU>=0.91 OR (IoU>=0.85 AND pHash distance<=6 in the same comparison)); reflect either pair member, making the rule symmetric",
        "silhouette_normalization": "bbox crop, holes filled, aspect-preserving 32x32; four axis reflections evaluated in both pair orders",
        "candidate_pool": before_pairs,
        "selected": {
            "total_unordered_pairs": len(selected) * (len(selected) - 1) // 2,
            "exact_pixel_duplicate_pairs": 0, "repeated_source_seeds": 0,
            "local_near_duplicate_pairs": local_near_count,
            "shape_near_pairs_all_polarities_without_mechanical_filter": shape_near_count,
            "shape_near_pairs_same_polarity_without_mechanical_filter": same_polarity_shape_count,
            "most_shape_similar_pairs": top_pairs,
        },
        "selected_features": [{"input_index": kept[i], **features[i]} for i in selected],
        "boundary_feature_caveat": "4-connected hard-phase spanning is a geometric descriptor, not a substitute for FEM or a claim that the soft solid carries no load.",
    }
    return chosen_input, stats, pair_metrics
