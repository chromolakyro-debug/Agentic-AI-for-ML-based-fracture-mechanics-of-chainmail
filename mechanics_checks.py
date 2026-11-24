"""Analytical, independent-solver, symmetry and refinement tests for mechanics.py."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from threadpoolctl import threadpool_limits
from mechanics import FiniteSpecimenFEM


def relative(a, b):
    return float(abs(a - b) / max(abs(b), 1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "cat_binary_dataset" / "dataset.npy")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-refinement", action="store_true")
    args = parser.parse_args()
    threadpool_limits(1)
    started = time.perf_counter()
    out = {"model": "fully integrated Q4 plane stress, finite square, lateral-slip tensile loading",
           "parameters": {"Es": 1., "Eh": 10., "nu": .3, "width": 40., "height": 40.,
                          "thickness": 2., "nominal_strain": .001},
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__}, "checks": {}}
    import pyamg
    out["versions"]["pyamg"] = pyamg.__version__
    n = 32
    fem = FiniteSpecimenFEM((n, n))
    for phase, value, expected in [("soft", 1, 1.), ("hard", 0, 10.)]:
        result = fem.solve(np.full((n, n), value, np.uint8))
        error = max(relative(result[k], expected) for k in ("Ex_app", "Ey_app"))
        assert error < 1e-8
        out["checks"]["homogeneous_" + phase] = {"Ex_app": result["Ex_app"],
                                                 "Ey_app": result["Ey_app"], "max_relative_error": error}
    layers = np.ones((n, n), np.uint8)
    layers[:, :n//2] = 0
    zero_nu = FiniteSpecimenFEM((n, n), poisson=0.)
    result = zero_nu.solve(layers)
    expected_x, expected_y = 1. / (.5 / 10. + .5 / 1.), 5.5
    error = max(relative(result["Ex_app"], expected_x), relative(result["Ey_app"], expected_y))
    assert error < 1e-7
    out["checks"]["layers_nu_zero_analytic"] = {"Ex_app": result["Ex_app"], "Ey_app": result["Ey_app"],
                                               "expected_x": expected_x, "expected_y": expected_y,
                                               "max_relative_error": error}
    rng = np.random.default_rng(1142031)
    pattern = np.repeat(np.repeat(rng.integers(0, 2, (8, 8), dtype=np.uint8), 4, 0), 4, 1)
    iterative = fem.solve(pattern)
    direct = FiniteSpecimenFEM((n, n), solver="direct").solve(pattern)
    error = max(relative(iterative[k], direct[k]) for k in ("Ex_app", "Ey_app"))
    assert error < 1e-7
    out["checks"]["amg_vs_direct"] = {"amg": [iterative["Ex_app"], iterative["Ey_app"]],
                                        "direct": [direct["Ex_app"], direct["Ey_app"]], "max_relative_error": error}
    transpose = fem.solve(pattern.T)
    error = max(relative(iterative["Ex_app"], transpose["Ey_app"]),
                relative(iterative["Ey_app"], transpose["Ex_app"]))
    assert error < 1e-7
    out["checks"]["axis_swap"] = {"max_relative_error": error}
    alternate_strain = FiniteSpecimenFEM((n, n), strain=.0001).solve(pattern)
    error = max(relative(iterative[k], alternate_strain[k]) for k in ("Ex_app", "Ey_app"))
    assert error < 1e-7
    out["checks"]["strain_linearity"] = {"strain_ratio": 10., "max_relative_error": error}
    scaled = FiniteSpecimenFEM((n, n), soft_modulus=7., hard_modulus=70.).solve(pattern)
    error = max(relative(scaled[k], 7. * iterative[k]) for k in ("Ex_app", "Ey_app"))
    assert error < 1e-7
    out["checks"]["modulus_scale"] = {"scale_factor": 7., "max_relative_error": error}
    dataset = np.load(args.dataset, allow_pickle=False, mmap_mode="r")
    setup = time.perf_counter()
    full = FiniteSpecimenFEM(dataset.shape[1:])
    out["setup_seconds_224"] = time.perf_counter() - setup
    records = []
    for index in (72, 83, 75, 120):
        result = full.solve(dataset[index])
        records.append({"source_index": index,
                        "source_image_sha256": hashlib.sha256(dataset[index].tobytes()).hexdigest(),
                        **result})
    out["reference_images"] = records
    out["median_seconds_per_224_image_two_tests"] = float(np.median([r["solve_seconds"] for r in records]))
    # One complete image checked using a separate sparse direct solver.
    direct_full = FiniteSpecimenFEM(dataset.shape[1:], solver="direct").solve(dataset[72])
    error = max(relative(records[0][k], direct_full[k]) for k in ("Ex_app", "Ey_app"))
    assert error < 1e-7
    out["checks"]["full_resolution_amg_vs_direct"] = {"max_relative_error": error,
                                                      "direct_solve_seconds": direct_full["solve_seconds"]}
    if not args.skip_refinement:
        refined = FiniteSpecimenFEM(tuple(2 * a for a in dataset.shape[1:]))
        values = []
        for record in records:
            index = record["source_index"]
            # Refinement preserves every original square material pixel exactly.
            fine_image = np.repeat(np.repeat(dataset[index], 2, 0), 2, 1)
            fine = refined.solve(fine_image)
            errors = {k: relative(record[k], fine[k]) for k in ("Ex_app", "Ey_app")}
            values.append({"source_index": index, "coarse": [record["Ex_app"], record["Ey_app"]],
                           "refined": [fine["Ex_app"], fine["Ey_app"]],
                           "relative_change": errors, "solve_seconds": fine["solve_seconds"]})
        out["checks"]["pixel_preserving_refinement_224_to_448"] = values
        out["max_refinement_change"] = max(max(x["relative_change"].values()) for x in values)
        # The threshold is a regression guard, not a universal discretization guarantee.
        assert out["max_refinement_change"] < .03
    out["all_checks_passed"] = True
    out["elapsed_seconds"] = time.perf_counter() - started
    encoded = json.dumps(out, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, flush=True)


if __name__ == "__main__":
    main()
