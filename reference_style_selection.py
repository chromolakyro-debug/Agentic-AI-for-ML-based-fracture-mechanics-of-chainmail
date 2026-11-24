"""Scalable append-only selection for the nine reference-style cat families.

The existing base is immutable. Each new source is retained at most once.
Mechanical near-duplicates use the prior exact declared predicate. A stricter
visual predicate additionally compares normalized *actual ink*, so similar
silhouettes with different lattice/halftone/line structure are not conflated.

No dense all-pairs matrix or candidate-to-candidate conflict graph is built.
Mechanical comparisons use an exact spatial grid; visual comparisons use a
five-band pigeonhole index that cannot miss a 64-bit pHash distance <= 4.
The visual predicate is descriptor-based, not a guarantee of human dissimilarity.

Public API:
    load_base(directory) -> samples, labels, original_rows
    select_additions(base_samples, base_labels, samples, labels, family_quotas,
                     seed=1145186, target=None, progress=None)
        -> candidate_input_indices, statistics, audit

Visual approval is separate and must be performed by the calling workflow.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import mechanics_selection as ms
from augmentation_selection import load_base, compare_coverage

VERSION = 1
INK_SIDE = 128
VISUAL_IOU = .97
VISUAL_PHASH = 4
LOCAL_RADIUS = .5
PAIR_COMPARISONS = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0))
BAND_BITS = (13, 13, 13, 13, 12)
BAND_START = (0, 13, 26, 39, 52)


def _get(obj, key, default=None):
    return ms._get(obj, key, default)


def _values(labels):
    return np.asarray([[float(_get(l, 'Ex_app')), float(_get(l, 'Ey_app'))]
                       for l in labels], dtype=float).reshape(-1, 2)


def _family(sample):
    details = _get(sample, 'diagnostics', {}) or {}
    return str(details.get('style_family', details.get('family', details.get('reference_style', ''))))


def _source(sample):
    return int(_get(_get(sample, 'spec'), 'seed'))


def _pixels(sample):
    return np.asarray(_get(sample, 'pixels'))


def _digest(sample):
    return hashlib.sha256(_pixels(sample).tobytes()).hexdigest()


def _valid_flag(value):
    return str(value).strip().lower() not in ('false', '0', 'no', 'none', '')


def _actual_ink(sample):
    """Four packed 128px actual-ink masks and a pooled ink-density descriptor.

    Canonical foreground is relative to the declared background. No hole fill
    is applied here. Reflections are performed before centering, consistently
    with the existing silhouette/pHash descriptor.
    """
    image = _pixels(sample)
    foreground = image != int(_get(_get(sample, 'spec'), 'background'))
    yy, xx = np.nonzero(foreground)
    crop = foreground[yy.min():yy.max()+1, xx.min():xx.max()+1]
    height, width = crop.shape
    scale = INK_SIDE/max(height, width)
    rw, rh = max(1, round(width*scale)), max(1, round(height*scale))
    content = np.asarray(Image.fromarray(crop).resize((rw, rh), Image.Resampling.NEAREST), dtype=bool)
    ox, oy = (INK_SIDE-rw)//2, (INK_SIDE-rh)//2
    packed, features = [], []
    for flip_y, flip_x in ((False, False), (False, True), (True, False), (True, True)):
        canvas = np.zeros((INK_SIDE, INK_SIDE), dtype=bool)
        canvas[oy:oy+rh, ox:ox+rw] = content[::-1 if flip_y else 1, ::-1 if flip_x else 1]
        packed.append(np.packbits(canvas.ravel()))
        # These pooled features are only a soft novelty score. Hard visual
        # rejection uses every 128x128 ink bit, not this coarse representation.
        features.append(canvas.reshape(16, 8, 16, 8).mean(axis=(1, 3)).ravel())
    return np.stack(packed), int(content.sum()), np.asarray(features, dtype=np.float32)


class _Descriptors:
    def __init__(self, samples):
        n = len(samples)
        self.masks = np.empty((n, 4, 128), dtype=np.uint8)
        self.hashes = np.empty((n, 4, 8), dtype=np.uint8)
        self.areas = np.empty(n, dtype=np.int32)
        self.ink = np.empty((n, 4, INK_SIDE*INK_SIDE//8), dtype=np.uint8)
        self.ink_areas = np.empty(n, dtype=np.int32)
        self.features = np.empty((n, 4, 256), dtype=np.float32)
        self.fractions = np.empty(n)
        self.polarities = np.empty(n, dtype=np.uint8)
        for i, sample in enumerate(samples):
            masks, hashes, area, feature = ms._descriptor(sample)
            ink, ink_area, pooled = _actual_ink(sample)
            self.masks[i], self.hashes[i], self.areas[i] = masks, hashes, area
            self.ink[i], self.ink_areas[i], self.features[i] = ink, ink_area, pooled
            self.fractions[i] = feature['black_fraction']
            self.polarities[i] = feature['background']


class _SpatialIndex:
    """Exact range candidate retrieval; final distances are filtered explicitly."""
    def __init__(self, values, width=.25):
        self.values, self.width, self.cells = values, float(width), defaultdict(list)

    def add(self, index):
        key = tuple(np.floor(self.values[index]/self.width).astype(int))
        self.cells[key].append(int(index))

    def query(self, value, radius):
        lo = np.floor((np.asarray(value)-radius)/self.width).astype(int)-1
        hi = np.floor((np.asarray(value)+radius)/self.width).astype(int)+1
        hits = []
        for a in range(lo[0], hi[0]+1):
            for b in range(lo[1], hi[1]+1):
                hits.extend(self.cells.get((a, b), ()))
        ids = np.asarray(hits, dtype=int)
        if len(ids):
            ids = ids[np.max(np.abs(self.values[ids]-value), axis=1) <= radius]
        return ids


def _bands(hashes):
    keys = set()
    for hash_bytes in hashes:
        value = int.from_bytes(hash_bytes.tobytes(), 'little')
        for band, (start, width) in enumerate(zip(BAND_START, BAND_BITS)):
            keys.add((band, (value >> start) & ((1 << width)-1)))
    return keys


class _VisualIndex:
    """Exact candidate superset for any reflected pHash pair of distance <=4.

    Five disjoint bit bands cover all 64 bits. At most four differing bits imply
    at least one unchanged band. All four reflected hashes are indexed/queried;
    the final symmetric seven-comparison predicate filters this superset.
    """
    def __init__(self, hashes):
        self.hashes, self.buckets = hashes, defaultdict(list)

    def add(self, index):
        for key in _bands(self.hashes[index]):
            self.buckets[key].append(int(index))

    def query(self, index):
        hits = set()
        for key in _bands(self.hashes[index]):
            hits.update(self.buckets.get(key, ()))
        return np.asarray(sorted(hits), dtype=int)


def _visual_matches(left, right, descriptor):
    """Return exact declared visual conflicts for one proposed image.

    Phase polarity and mechanical properties are intentionally not restrictions.
    Both silhouette and actual-ink overlap must meet .97 in the SAME reflection
    comparison, and that comparison's pHash distance must be <=4.
    """
    right = np.asarray(right, dtype=int)
    conflict = np.zeros(len(right), dtype=bool)
    best_ink = np.zeros(len(right))
    for lref, rref in PAIR_COMPARISONS:
        active = np.flatnonzero(~conflict)
        if not len(active):
            break
        distance = ms.POPCOUNT[np.bitwise_xor(descriptor.hashes[left, lref],
                                             descriptor.hashes[right[active], rref])].sum(axis=1)
        active = active[distance <= VISUAL_PHASH]
        if not len(active):
            continue
        intersection = ms.POPCOUNT[np.bitwise_and(descriptor.masks[left, lref],
                                                 descriptor.masks[right[active], rref])].sum(axis=1)
        iou = intersection/np.maximum(descriptor.areas[left]+descriptor.areas[right[active]]-intersection, 1)
        active = active[iou >= VISUAL_IOU]
        if not len(active):
            continue
        intersection = ms.POPCOUNT[np.bitwise_and(descriptor.ink[left, lref],
                                                 descriptor.ink[right[active], rref])].sum(axis=1)
        iou = intersection/np.maximum(descriptor.ink_areas[left]+descriptor.ink_areas[right[active]]-intersection, 1)
        best_ink[active] = np.maximum(best_ink[active], iou)
        conflict[active[iou >= VISUAL_IOU]] = True
    return conflict, best_ink


def _novelty(index, spatial, descriptor, values):
    """Soft morphology/ink diversity among the closest 24 property neighbors."""
    neighbors = spatial.query(values[index], LOCAL_RADIUS)
    if not len(neighbors):
        return 1.
    if len(neighbors) > 24:
        distances = np.max(np.abs(values[neighbors]-values[index]), axis=1)
        neighbors = neighbors[np.argpartition(distances, 23)[:24]]
    nearest = 1.
    for lref, rref in PAIR_COMPARISONS:
        difference = descriptor.features[neighbors, rref]-descriptor.features[index, lref]
        rms = np.sqrt(np.mean(difference*difference, axis=1))
        nearest = min(nearest, float(rms.min()))
    return min(1., 3*nearest)


def _coverage(base_values, added_values, base_fractions, added_fractions):
    result = compare_coverage(base_values, added_values, base_fractions, added_fractions)
    combined = np.vstack([base_values, added_values])
    before_counts = np.bincount(ms.property_bins(base_values), minlength=36)
    after_counts = np.bincount(ms.property_bins(combined), minlength=36)
    support = np.flatnonzero(after_counts)
    result['common_observed_support'] = dict(
        flat_bin_ids=support.tolist(), before_counts=before_counts[support].tolist(),
        after_counts=after_counts[support].tolist(),
        before_count_CV=float(before_counts[support].std()/before_counts[support].mean()),
        after_count_CV=float(after_counts[support].std()/after_counts[support].mean()),
        definition='CV on the same observed final-bin support; uniform counts would have CV=0. Not a feasible-area uniformity test.')
    return result


def select_additions(base_samples, base_labels, samples, labels, family_quotas,
                     seed=1145186, target=None, progress=None):
    """Select new independent sources with hard style quotas and soft coverage.

    family_quotas describes THIS call's additions. If a pilot is already part
    of the immutable base, subtract its per-family counts before calling.
    Infeasible quotas return explicit shortfalls, never duplicate padding.

    progress, if supplied, is called with small dictionaries every 100 accepted
    images. It must not mutate the candidate/base arrays.
    """
    started = time.perf_counter()
    if len(base_samples) != len(base_labels) or len(samples) != len(labels) or not base_samples:
        raise ValueError('A nonempty aligned fixed base and aligned candidate labels are required.')
    quota_map = {str(k): int(v) for k, v in family_quotas.items()}
    if not quota_map or any(v < 0 for v in quota_map.values()):
        raise ValueError('Provide nonnegative per-family quotas.')
    requested = sum(quota_map.values())
    if target is not None and int(target) != requested:
        raise ValueError('target must equal the sum of this stage family_quotas.')
    families = list(quota_map)
    family_id = {name:i for i,name in enumerate(families)}
    quotas = np.array([quota_map[f] for f in families], dtype=int)
    base_values = _values(base_labels)
    ms.property_bins(base_values)
    nbase = len(base_samples)
    base_digests = {_digest(s) for s in base_samples}
    base_sources = {_source(s) for s in base_samples}
    if len(base_digests) != nbase or len(base_sources) != nbase:
        raise ValueError('Immutable base has exact/source duplicates; this selector cannot remove base data.')
    valid, candidate_values, candidate_families = [], [], []
    reasons, reasons_by_family = Counter(), {f:Counter() for f in families}
    for index, (sample, label) in enumerate(zip(samples, labels)):
        family = _family(sample)
        reason = ''
        pixels = _pixels(sample)
        if family not in family_id or quota_map.get(family, 0) == 0:
            reason = 'family_not_requested'
        elif pixels.dtype != np.uint8 or pixels.shape != (224, 224) or pixels.min() != 0 or pixels.max() != 1:
            reason = 'invalid_binary_or_blank_image'
        elif _get(_get(sample, 'spec'), 'background') not in (0, 1) or _get(_get(sample, 'spec'), 'seed') is None:
            reason = 'missing_source_or_background'
        elif _get(sample, 'issue', ''):
            reason = 'generation_issue'
        elif label is None or not _valid_flag(_get(label, 'fem_valid', True)):
            reason = 'invalid_fem'
        else:
            try:
                value = _values([label])[0]
                ms.property_bins(value)
                fraction = float((pixels == 0).mean())
                if np.any(value < 1/(1-.9*fraction)-1e-5) or np.any(value > 1+9*fraction+1e-5):
                    reason = 'fixed_fraction_necessary_bound_failed'
                elif _digest(sample) in base_digests:
                    reason = 'exact_duplicate_of_base'
                elif _source(sample) in base_sources:
                    reason = 'source_already_in_base'
                elif not np.isfinite([float(_get(sample, 'cat_margin', 0)), float(_get(sample, 'cat_logit', 0))]).all():
                    reason = 'nonfinite_semantic_score'
            except (TypeError, ValueError, KeyError):
                reason = 'invalid_label_or_source'
        if reason:
            reasons[reason] += 1
            if family in reasons_by_family:
                reasons_by_family[family][reason] += 1
            continue
        valid.append(index); candidate_values.append(value); candidate_families.append(family_id[family])
    if not valid or requested == 0:
        stats = dict(version=VERSION, base_preserved=True, base_count=nbase, target=requested,
                     selected_count=0, target_met=requested == 0, shortfall=requested,
                     family_quotas=quota_map, family_selected={f:0 for f in families},
                     family_shortfalls=quota_map, exclusions=dict(reasons),
                     exclusions_by_family={f:dict(c) for f,c in reasons_by_family.items()},
                     eligible_variants=len(valid), eligible_independent_sources=0,
                     family_eligible_independent_sources={f:0 for f in families},
                     selected_input_indices=[], selected_source_seeds=[], selection_trace=[],
                     coverage=_coverage(base_values, np.empty((0, 2)),
                                        [float((_pixels(x) == 0).mean()) for x in base_samples], []),
                     elapsed_seconds=time.perf_counter()-started)
        return [], stats, dict(audited_new_base_pairs=0, audited_new_new_pairs=0,
                              mechanical_near_conflicts_remaining=0, strict_visual_conflicts_remaining=0)

    candidate_values = np.asarray(candidate_values)
    candidate_families = np.asarray(candidate_families, dtype=int)
    selected_inputs = np.asarray(valid, dtype=int)
    all_samples = list(base_samples)+[samples[i] for i in valid]
    values = np.vstack([base_values, candidate_values])
    if progress:
        progress(dict(stage='descriptors', total=len(all_samples)))
    descriptor = _Descriptors(all_samples)
    n = len(valid)
    digests = np.array([_digest(s) for s in all_samples[nbase:]])
    sources = np.array([_source(s) for s in all_samples[nbase:]], dtype=np.int64)
    source_members, pixel_members = defaultdict(list), defaultdict(list)
    for i, (source, digest) in enumerate(zip(sources, digests)):
        source_members[int(source)].append(i); pixel_members[str(digest)].append(i)
    available = np.ones(n, dtype=bool)
    chosen, traces = [], []
    family_counts = np.zeros(len(families), dtype=int)
    family_independent_sources = {f:len(set(sources[candidate_families == family_id[f]].tolist())) for f in families}
    spatial, visual = _SpatialIndex(values), _VisualIndex(descriptor.hashes)
    for i in range(nbase):
        spatial.add(i); visual.add(i)
    base_tree = cKDTree(base_values)
    nearest = base_tree.query(candidate_values, p=np.inf)[0]
    local_counts = base_tree.query_ball_point(candidate_values, LOCAL_RADIUS, p=np.inf, return_length=True).astype(int)
    coarse_ids = ms.property_bins(values)
    fine_xy = np.clip(np.floor((values-1)/.5).astype(int), 0, 17)
    fine_ids = fine_xy[:, 1]*18+fine_xy[:, 0]
    coarse_counts = np.bincount(coarse_ids[:nbase], minlength=36)
    fine_counts = np.bincount(fine_ids[:nbase], minlength=324)
    active_coarse_count = max(1, len(set(coarse_ids.tolist())))
    active_fine_count = max(1, len(set(fine_ids.tolist())))
    rng = np.random.default_rng(seed)
    ties = rng.random(n)
    semantic = np.array([np.tanh(float(_get(s, 'cat_margin', 0))/10) for s in all_samples[nbase:]])
    target_bonus = np.zeros(n)
    for j, sample in enumerate(all_samples[nbase:]):
        details = _get(sample, 'diagnostics', {}) or {}
        if details.get('target_Ex') is not None and details.get('target_Ey') is not None:
            error = np.max(np.abs(candidate_values[j]-[float(details['target_Ex']), float(details['target_Ey'])]))
            if np.isfinite(error):
                target_bonus[j] = .05*np.exp(-error/.25)
    mechanical_pairs_evaluated = visual_hash_candidates = visual_pairs_evaluated = 0
    rejected_examples = []

    def reject(index, reason, other=None, detail=None):
        available[index] = False
        family = families[candidate_families[index]]
        reasons[reason] += 1; reasons_by_family[family][reason] += 1
        if len(rejected_examples) < 30:
            rejected_examples.append(dict(input_index=valid[index], family=family, reason=reason,
                                          other_base_index=int(other) if other is not None and other < nbase else None,
                                          other_new_input_index=valid[other-nbase] if other is not None and other >= nbase else None,
                                          detail=detail))

    def admissible(index):
        nonlocal mechanical_pairs_evaluated, visual_hash_candidates, visual_pairs_evaluated
        global_index = nbase+index
        neighbors = spatial.query(values[global_index], ms.MECHANICAL_TOLERANCE)
        neighbors = neighbors[descriptor.polarities[neighbors] == descriptor.polarities[global_index]]
        neighbors = neighbors[np.abs(descriptor.fractions[neighbors]-descriptor.fractions[global_index]) <= ms.FRACTION_TOLERANCE]
        mechanical_pairs_evaluated += len(neighbors)
        if len(neighbors):
            _, _, near = ms._compare(np.full(len(neighbors), global_index), neighbors,
                                     descriptor.masks, descriptor.hashes, descriptor.areas)
            if near.any():
                reject(index, 'mechanical_geometry_near_duplicate', int(neighbors[np.flatnonzero(near)[0]]))
                return False
        neighbors = visual.query(global_index)
        visual_hash_candidates += len(neighbors)
        for start in range(0, len(neighbors), 1024):
            batch = neighbors[start:start+1024]
            visual_pairs_evaluated += len(batch)
            near, ink = _visual_matches(global_index, batch, descriptor)
            if near.any():
                where = np.flatnonzero(near)[0]
                reject(index, 'strict_visual_near_duplicate', int(batch[where]),
                       dict(actual_ink_iou=float(ink[where])))
                return False
        return True

    while len(chosen) < requested:
        capacity = np.bincount(candidate_families[available], minlength=len(families))
        active_families = np.flatnonzero((family_counts < quotas) & (capacity > 0))
        if not len(active_families):
            break
        family = min(active_families, key=lambda f:(family_counts[f]/quotas[f], f))
        options = np.flatnonzero(available & (candidate_families == family))
        total = nbase+len(chosen)
        coarse_pressure = coarse_counts[coarse_ids[nbase+options]]/(total/active_coarse_count)
        fine_pressure = fine_counts[fine_ids[nbase+options]]/(total/active_fine_count)
        # This is soft coverage balancing on observed candidate support. It is
        # not an assertion of equal attainable area for every rectangular bin.
        scores = (.40*np.minimum(nearest[options]/.5, 1) + .20/(1+local_counts[options])
                  + .15/(1+coarse_pressure) + .10/(1+fine_pressure)
                  + target_bonus[options] + 1e-4*semantic[options] + 1e-8*ties[options])
        shortlist_size = min(4, len(options))
        shortlist = np.argpartition(-scores, shortlist_size-1)[:shortlist_size]
        ranked = []
        for offset in shortlist:
            index = int(options[offset])
            novelty = _novelty(nbase+index, spatial, descriptor, values)
            ranked.append((float(scores[offset]+.25*novelty), -int(valid[index]), index, novelty))
        accepted = False
        for _, _, index, novelty in sorted(ranked, reverse=True):
            if not admissible(index):
                continue
            global_index = nbase+index
            traces.append(dict(input_index=valid[index], family=families[family],
                               nearest_preexisting_linf=float(nearest[index]),
                               neighbors_within_0_5=int(local_counts[index]), local_ink_novelty=float(novelty)))
            chosen.append(index); family_counts[family] += 1
            available[index] = False
            for reason, members in (('same_source_variant_blocked', source_members[int(sources[index])]),
                                    ('exact_candidate_duplicate_blocked', pixel_members[str(digests[index])])):
                for member in members:
                    if available[member]:
                        reject(member, reason, global_index)
            spatial.add(global_index); visual.add(global_index)
            delta = np.max(np.abs(candidate_values-candidate_values[index]), axis=1)
            nearest = np.minimum(nearest, delta)
            local_counts += delta <= LOCAL_RADIUS
            coarse_counts[coarse_ids[global_index]] += 1
            fine_counts[fine_ids[global_index]] += 1
            accepted = True
            if progress and (len(chosen) % 100 == 0 or len(chosen) == requested):
                progress(dict(stage='selection', accepted=len(chosen), requested=requested,
                              family_counts=dict(zip(families, family_counts.tolist())),
                              elapsed_seconds=round(time.perf_counter()-started, 2)))
            break
        # A rejected shortlist is permanently removed, so this always makes
        # progress even if a style has only repeated shapes left.
        if not accepted and not len(shortlist):
            raise AssertionError('Empty shortlist with nonempty options.')

    chosen = np.asarray(chosen, dtype=int)
    indices = [valid[i] for i in chosen]
    if len(set(sources[chosen].tolist())) != len(chosen) or len(set(digests[chosen].tolist())) != len(chosen):
        raise AssertionError('Selected source/pixel duplicates escaped incremental checks.')
    coverage = _coverage(base_values, candidate_values[chosen], descriptor.fractions[:nbase], descriptor.fractions[nbase+chosen])
    statistics = dict(version=VERSION, seed=int(seed), base_count=nbase, base_preserved=True,
                      input_count=len(samples), eligible_variants=n,
                      eligible_independent_sources=int(len(set(sources.tolist()))),
                      family_eligible_independent_sources=family_independent_sources,
                      eligibility_interpretation='Input validation and base exact/source checks only; eligible counts are not a proof of joint duplicate-free capacity.',
                      family_quotas=quota_map, family_selected=dict(zip(families, family_counts.tolist())),
                      family_shortfalls=dict(zip(families, (quotas-family_counts).tolist())),
                      target=requested, selected_count=len(chosen), target_met=len(chosen) == requested,
                      shortfall=requested-len(chosen), exclusions=dict(reasons),
                      exclusions_by_family={f:dict(c) for f,c in reasons_by_family.items()},
                      selected_input_indices=indices, selected_source_seeds=[int(sources[i]) for i in chosen],
                      selection_trace=traces, coverage=coverage, elapsed_seconds=time.perf_counter()-started,
                      visual_review='upstream/downstream workflow responsibility; not inferred by this selector',
                      shortfall_interpretation='Shortfalls are for this deterministic greedy selection, not a proof that no alternative global conflict-free selection exists. No padding or quota transfer is performed.',
                      selection_method='hard per-family quotas; soft nearest-property, local-density and observed-bin coverage; pooled actual-ink geometry novelty; exact incremental duplicate predicates')
    audit = dict(base_count=nbase, added_count=len(chosen),
                 audited_new_base_pairs=nbase*len(chosen), audited_new_new_pairs=len(chosen)*(len(chosen)-1)//2,
                 exact_pixel_duplicate_pairs_involving_new=0, repeated_source_seeds_involving_new=0,
                 mechanical_near_conflicts_remaining=0, strict_visual_conflicts_remaining=0,
                 actual_mechanical_descriptor_pair_comparisons=mechanical_pairs_evaluated,
                 visual_hash_index_candidate_pairs=visual_hash_candidates,
                 actual_visual_predicate_pair_comparisons=visual_pairs_evaluated,
                 rejected_examples=rejected_examples,
                 mechanical_rule='same polarity AND abs(f1-f2)<=.025 AND modulus Linf<=.12 AND (mirror-aware silhouette IoU>=.91 OR silhouette IoU>=.85 with pHash<=6 in the same comparison)',
                 strict_visual_rule='independent of modulus/polarity: normalized actual-ink 128x128 IoU>=.97 AND normalized hole-filled silhouette 32x32 IoU>=.97 AND pHash<=4, all in the same one of seven symmetric reflection comparisons',
                 index_guarantee='Mechanical spatial buckets are exact range supersets; five disjoint pHash bands guarantee retrieval of every declared <=4-bit visual candidate; final predicates are explicitly evaluated.',
                 audit_interpretation='Every accepted addition was checked against the whole fixed base and all previously accepted additions, with logically impossible pairs pruned by exact necessary predicates. Existing base-base pairs are not changed or reclassified. Rejection/evaluation counts include unselected candidates.',
                 limitation='These are declared numerical near-duplicate rules, not a claim that all pictures look different to every human.')
    return indices, statistics, audit
