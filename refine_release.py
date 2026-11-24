"""Spot-check final release labels and pixel-preserving mesh refinement.

Selects up to eight actual release specimens: directional-modulus and
anisotropy extrema, plus thin four-connected hard spanning paths.  The latter
use the smallest column/row cross-section as a screening proxy, not a minimum
cut calculation and not a claim that the hard phase carries all of the load.

Example:
  python refine_release.py --dataset path/to/release --output scratch/final_refinement.json
  python refine_release.py --dataset path/to/release --output scratch/style_refinement.json --first-index 1250 --indices 1250 1382 1640

Explicit --indices refer to global array positions / cat_NNNN.png filenames,
preserve the supplied order, and are not limited to eight specimens. The
choose_style_indices helper provides a deterministic style-stratified plan
without running FEM; pass its global indices to this command for later solving.

The 224 mesh is recomputed with a sparse direct solver, independently of the
cached CSV labels. The 448 mesh splits each original material pixel into four
equal Q4 elements. This preserves geometry and defines nested displacement
spaces, so the apparent energy stiffness must not increase on refinement
(up to numerical tolerance). Sampled changes are not continuum-error bounds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
from threadpoolctl import threadpool_limits

from mechanics import FiniteSpecimenFEM


def _csv_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def hard_path_descriptors(image):
    """Actual four-connected spanning components and simple width proxies."""
    hard = image == 0
    labels, _ = ndimage.label(hard, structure=ndimage.generate_binary_structure(2, 1))
    x_ids = np.intersect1d(np.unique(labels[:, 0]), np.unique(labels[:, -1]))
    y_ids = np.intersect1d(np.unique(labels[0, :]), np.unique(labels[-1, :]))
    x_ids, y_ids = x_ids[x_ids != 0], y_ids[y_ids != 0]
    span_x, span_y = len(x_ids) > 0, len(y_ids) > 0
    x_width = int(np.isin(labels, x_ids).sum(axis=0).min()) if span_x else None
    y_width = int(np.isin(labels, y_ids).sum(axis=1).min()) if span_y else None
    return {
        "hard_spans_x": span_x, "hard_spans_y": span_y,
        "minimum_hard_column_fraction": float(hard.mean(axis=0).min()),
        "minimum_hard_row_fraction": float(hard.mean(axis=1).min()),
        "minimum_spanning_hard_column_pixels": x_width,
        "minimum_spanning_hard_row_pixels": y_width,
    }


def choose_indices(images, rows):
    """Deterministic extrema plus up to two distinct narrow spanning examples."""
    properties = np.array([[float(r["Ex_app"]), float(r["Ey_app"])] for r in rows])
    if not np.isfinite(properties).all() or np.any(properties <= 0):
        raise ValueError("CSV contains nonfinite or nonpositive moduli.")
    ratio = properties[:, 0] / properties[:, 1]
    reasons = {}

    def add(index, reason):
        reasons.setdefault(int(index), []).append(reason)

    for values, label in ((properties[:, 0], "Ex"), (properties[:, 1], "Ey"),
                          (ratio, "Ex_over_Ey")):
        add(np.argmin(values), f"minimum_{label}")
        add(np.argmax(values), f"maximum_{label}")

    descriptors = {}
    # CSV connectivity is checked against pixels for every potential narrow-path
    # specimen. It is never used to manufacture an absent spanning path.
    for i, (image, row) in enumerate(zip(images, rows)):
        row_x = _csv_bool(row["hard_spans_x"]) if "hard_spans_x" in row else True
        row_y = _csv_bool(row["hard_spans_y"]) if "hard_spans_y" in row else True
        if row_x or row_y or i in reasons:
            descriptor = hard_path_descriptors(image)
            if "hard_spans_x" in row and descriptor["hard_spans_x"] != row_x:
                raise ValueError(f"CSV x connectivity disagrees with pixels at {i}.")
            if "hard_spans_y" in row and descriptor["hard_spans_y"] != row_y:
                raise ValueError(f"CSV y connectivity disagrees with pixels at {i}.")
            descriptors[i] = descriptor

    for axis, key in (("x", "minimum_spanning_hard_column_pixels"),
                      ("y", "minimum_spanning_hard_row_pixels")):
        ranked = sorted((d[key], i) for i, d in descriptors.items() if d[key] is not None)
        if ranked:
            # Record the absolute thinnest even if it was already an extremum;
            # then select a distinct spanning example if capacity permits.
            add(ranked[0][1], f"thinnest_spanning_hard_path_{axis}")
            if len(reasons) < 8:
                for _, i in ranked:
                    if i not in reasons:
                        add(i, f"additional_narrow_spanning_hard_path_{axis}")
                        break
    # Adding the x example can fill slot 8 before y, so preserve all extrema and
    # prioritize the absolute thinnest paths, never arbitrary repeated samples.
    indices = list(reasons)
    if len(indices) > 8:
        priority = lambda i: (not any(r.startswith(("minimum_", "maximum_", "thinnest_"))
                                    for r in reasons[i]), indices.index(i))
        indices = sorted(indices, key=priority)[:8]
    return indices, {i: reasons[i] for i in indices}, {i: descriptors[i] for i in indices}


def resolve_selection(images, rows, first_index=0, indices=None):
    """Resolve automatic suffix selection or explicitly requested global IDs."""
    if len(images) != len(rows) or not len(images):
        raise ValueError('Images and CSV rows must be nonempty and aligned.')
    if not 0 <= first_index < len(images):
        raise ValueError('first_index must identify an existing specimen.')
    if indices is None:
        selected, reasons, descriptors = choose_indices(images[first_index:], rows[first_index:])
        return ([i+first_index for i in selected],
                {i+first_index:v for i,v in reasons.items()},
                {i+first_index:v for i,v in descriptors.items()})
    selected = list(indices)
    if not selected or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in selected):
        raise ValueError('indices must be a nonempty sequence of integer global indices.')
    selected = [int(i) for i in selected]
    if len(set(selected)) != len(selected):
        raise ValueError('Explicit indices must not contain duplicates.')
    if any(i < first_index or i >= len(images) for i in selected):
        raise ValueError('Explicit indices must lie between first_index and the final specimen index.')
    descriptors = {}
    for i in selected:
        if rows[i].get('filename') != f'cat_{i:04d}.png':
            raise ValueError(f'Filename/array ordering mismatch at {i}.')
        descriptor = hard_path_descriptors(images[i])
        for axis in ('x', 'y'):
            name = f'hard_spans_{axis}'
            if name in rows[i] and _csv_bool(rows[i][name]) != descriptor[name]:
                raise ValueError(f'CSV {axis} connectivity disagrees with pixels at {i}.')
        descriptors[i] = descriptor
    return selected, {i:['explicit_global_index'] for i in selected}, descriptors


def choose_style_indices(images, rows, first_index=1250, families=None, per_family=1):
    """Plan one probe/style plus global Ex/Ey/anisotropy extrema, without FEM.

    The risk probe is the narrowest four-connected hard spanning component by
    a cross-section proxy, or the highest material-interface density if no
    specimen in that style spans. This fallback includes internal fine texture
    but does not identify a true load-bearing bottleneck. Optional per_family=2
    adds maximal normalized distance from each style's median (Ex,Ey). Finally
    add directional and anisotropy extrema of the whole requested suffix.
    The default nine-style plan contains 9 to 15 distinct specimens.
    """
    if len(images) != len(rows) or not 0 <= first_index < len(rows):
        raise ValueError('Provide aligned data and a valid first_index.')
    if per_family not in (1, 2):
        raise ValueError('per_family must be one or two.')
    family_order = list(families) if families is not None else list(dict.fromkeys(
        r.get('style_family', '') for r in rows[first_index:]))
    if not family_order or any(not name for name in family_order) or len(set(family_order)) != len(family_order):
        raise ValueError('Style families must be named and unique.')
    properties = np.array([[float(r['Ex_app']), float(r['Ey_app'])] for r in rows])
    if not np.isfinite(properties).all() or np.any(properties <= 0):
        raise ValueError('CSV contains nonfinite or nonpositive moduli.')
    reasons, descriptors = {}, {}

    def add(index, reason):
        reasons.setdefault(int(index), []).append(reason)

    for family in family_order:
        members = [i for i in range(first_index, len(rows)) if rows[i].get('style_family') == family]
        if not members:
            raise ValueError(f'No specimens for requested style {family!r}.')
        risk = []
        for i in members:
            descriptor = hard_path_descriptors(images[i])
            hard = np.asarray(images[i]) == 0
            interface = float((np.count_nonzero(hard[:, 1:] != hard[:, :-1])
                               + np.count_nonzero(hard[1:] != hard[:-1]))
                              / (hard.shape[0]*(hard.shape[1]-1)+hard.shape[1]*(hard.shape[0]-1)))
            descriptor['material_interface_density'] = interface
            descriptors[i] = descriptor
            widths = [descriptor[k] for k in ('minimum_spanning_hard_column_pixels',
                                              'minimum_spanning_hard_row_pixels') if descriptor[k] is not None]
            risk.append((min(widths) if widths else np.inf, -interface, i))
        width, _, narrow = min(risk)
        add(narrow, f'style:{family}:'+('thinnest_spanning_hard_path_proxy' if np.isfinite(width)
                                      else 'highest_interface_density_no_spanning_path'))
        if per_family == 2:
            style_values = properties[members]
            scale = np.maximum(np.ptp(style_values, axis=0), 1e-12)
            distance = np.max(np.abs(style_values-np.median(style_values, axis=0))/scale, axis=1)
            ranked = sorted(zip(-distance, members))
            extreme = next((i for _,i in ranked if i != narrow), narrow)
            add(extreme, f'style:{family}:extreme_relative_to_style_median')
    suffix = properties[first_index:]
    for values, name in ((suffix[:, 0], 'Ex'), (suffix[:, 1], 'Ey'),
                         (suffix[:, 0]/suffix[:, 1], 'Ex_over_Ey')):
        add(int(np.argmin(values))+first_index, f'new_subset_minimum_{name}')
        add(int(np.argmax(values))+first_index, f'new_subset_maximum_{name}')
    selected = list(reasons)
    # Reuse explicit validation for all final probes, including CSV connectivity.
    _, _, checked = resolve_selection(images, rows, first_index, selected)
    for i in selected:
        if i not in descriptors:
            descriptors[i] = checked[i]
    return selected, reasons, {i:descriptors[i] for i in selected}


def run(dataset, output, fail_on_large_change=False, review_threshold=.05, first_index=0, indices=None):
    folder = Path(dataset)
    if folder.is_file():
        if folder.name != "dataset.npy":
            raise ValueError("Provide the release folder or its dataset.npy.")
        folder = folder.parent
    images = np.load(folder / "dataset.npy", mmap_mode="r", allow_pickle=False)
    with (folder / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if images.ndim != 3 or images.shape[1:] != (224, 224) or images.dtype != np.uint8:
        raise ValueError("Expected uint8 dataset with shape (N,224,224).")
    if len(rows) != len(images) or len(images) == 0:
        raise ValueError("CSV and dataset lengths differ or are empty.")
    if not np.isin(images, (0, 1)).all():
        raise ValueError("Nonbinary dataset.")
    explicit_indices = None if indices is None else list(indices)
    indices, reasons, descriptors = resolve_selection(images, rows, first_index, explicit_indices)
    started = time.perf_counter()
    threadpool_limits(1)
    coarse_fem = FiniteSpecimenFEM(solver="direct")
    refined_fem = FiniteSpecimenFEM((448, 448))
    records, problems, warnings = [], [], []
    parameters = {"E_soft": 1., "E_hard": 10., "nu_soft": .3, "nu_hard": .3,
                  "specimen_width_mm": 40., "specimen_height_mm": 40.,
                  "specimen_thickness_mm": 2., "nominal_strain": .001}
    for index in indices:
        row, pixels = rows[index], np.asarray(images[index])
        filename = f"cat_{index:04d}.png"
        if row["filename"] != filename:
            raise ValueError(f"Filename/array ordering mismatch at {index}.")
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if row.get("sha256_pixels") != digest:
            raise ValueError(f"CSV image hash disagrees with {filename}.")
        for key, expected in parameters.items():
            if key not in row or not np.isclose(float(row[key]), expected, rtol=0., atol=1e-12):
                raise ValueError(f"Unexpected material/test parameter {key} in {filename}.")
        if row.get("mechanical_protocol") != "finite_plane_stress_lateral_free_tension_Q4_224":
            raise ValueError(f"Unexpected mechanical protocol in {filename}.")
        coarse = coarse_fem.solve(pixels)
        fine_pixels = np.repeat(np.repeat(pixels, 2, axis=0), 2, axis=1)
        fine = refined_fem.solve(fine_pixels)
        csv_labels = {key: float(row[key]) for key in ("Ex_app", "Ey_app")}
        label_errors = {key: abs(coarse[key] - csv_labels[key]) / abs(coarse[key]) for key in csv_labels}
        changes = {key: abs(coarse[key] - fine[key]) / abs(fine[key]) for key in csv_labels}
        increases = {key: (fine[key] - coarse[key]) / abs(coarse[key]) for key in csv_labels}
        if max(label_errors.values()) > 1e-7:
            problems.append(f"{filename}: cached CSV differs from direct 224 solve by >1e-7 relative.")
        if max(increases.values()) > 1e-5:
            problems.append(f"{filename}: nested refinement increases energy stiffness by >1e-5 relative.")
        if max(changes.values()) > review_threshold:
            warnings.append(f"{filename}: refinement change exceeds review threshold {review_threshold:.1%}.")
        record = {
            "index": index, "filename": filename, "sha256_pixels": digest,
            "style_family": row.get('style_family', ''),
            "selection_reasons": reasons[index], "path_descriptors": descriptors[index],
            "hard_fraction": float((pixels == 0).mean()),
            "csv_224": csv_labels,
            "direct_224": {key: coarse[key] for key in csv_labels},
            "amg_448": {key: fine[key] for key in csv_labels},
            "csv_recomputation_relative_error": label_errors,
            "relative_change": changes,
            "signed_refinement_increase_relative_to_224": increases,
            "maximum_residual": max(coarse["residual_x"], coarse["residual_y"],
                                    fine["residual_x"], fine["residual_y"]),
            "maximum_energy_relative_error": max(coarse["energy_relative_error_x"],
                                                 coarse["energy_relative_error_y"],
                                                 fine["energy_relative_error_x"], fine["energy_relative_error_y"]),
            "seconds_224": coarse["solve_seconds"], "seconds_448": fine["solve_seconds"],
        }
        records.append(record)
        print(f"Refinement {len(records)}/{len(indices)} {filename}: "
              f"max change {max(changes.values()):.4%}, label error {max(label_errors.values()):.2e}", flush=True)
    report = {
        "dataset_directory": str(folder.resolve()),
        "dataset_sha256_pixels": hashlib.sha256(images.tobytes()).hexdigest(),
        "count": len(records), "indices": indices, "parameters": parameters,
        "candidate_index_range": [first_index,len(images)],
        "selection_mode": 'explicit_global_indices' if explicit_indices is not None else 'automatic_extrema_and_paths',
        "requested_indices": explicit_indices,
        "coarse_mesh": [224, 224], "refined_mesh": [448, 448],
        "coarse_solver": "SciPy sparse direct", "refined_solver": "CG, homogeneous AMG preconditioner",
        "maximum_relative_change": max(max(r["relative_change"].values()) for r in records),
        "maximum_csv_recomputation_relative_error": max(max(r["csv_recomputation_relative_error"].values()) for r in records),
        "maximum_refinement_increase": max(max(r["signed_refinement_increase_relative_to_224"].values()) for r in records),
        "label_recomputation_tolerance": 1e-7, "nested_space_monotonicity_tolerance": 1e-5,
        "large_change_review_threshold": review_threshold,
        "checks_passed": not problems, "large_change_review_required": bool(warnings),
        "problems": problems, "warnings": warnings, "specimens": records,
        "elapsed_seconds": time.perf_counter() - started,
        "limitation": "Selected examples only. Observed 224-to-448 changes are not certified continuum errors or a guarantee for untested specimens. Hard-path width is a column/row screening proxy; the soft solid also carries load.",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("count", "maximum_relative_change",
                     "maximum_csv_recomputation_relative_error", "checks_passed", "large_change_review_required")}), flush=True)
    if problems or (fail_on_large_change and warnings):
        raise RuntimeError(f"Refinement checks require review; see {output}.")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-large-change", action="store_true")
    parser.add_argument("--review-threshold", type=float, default=.05)
    parser.add_argument("--first-index", type=int, default=0,
                        help="Select refinement probes only from this array index onward.")
    parser.add_argument("--indices", type=int, nargs='+',
                        help="Explicit global cat_NNNN indices, space separated; must be >= first-index. No eight-probe cap.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args.dataset, args.output, args.fail_on_large_change, args.review_threshold, args.first_index, args.indices)
