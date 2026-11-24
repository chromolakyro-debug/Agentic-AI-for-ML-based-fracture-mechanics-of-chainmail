"""Finite two-phase specimen tests, fully integrated bilinear Q4 plane stress.

Black (0) is the hard solid and white (1) is the soft solid.  The specimen is
not periodic.  Each directional tensile test prescribes only the displacement
normal to its two loaded edges.  Transverse displacement is free except for
one pin that removes a rigid translation.  Image rows run downwards; they are
reversed when assigning the material to the physical y-up mesh.

NumPy/SciPy are required.  Optional pyamg provides a reusable homogeneous
elasticity preconditioner.  Labels are apparent specimen moduli under this
specific protocol, not periodic homogenized elastic constants.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve


def q4_stiffness(hx: float, hy: float, poisson: float, thickness: float) -> np.ndarray:
    """Unit-Young-modulus element; 2x2 Gauss integration, engineering shear."""
    nu = float(poisson)
    if not -1.0 < nu < 0.5:
        raise ValueError("Poisson ratio must be between -1 and 0.5.")
    constitutive = np.array([[1., nu, 0.], [nu, 1., 0.],
                            [0., 0., (1. - nu) / 2.]]) / (1. - nu * nu)
    stiffness = np.zeros((8, 8), dtype=np.float64)
    for xi in (-1. / np.sqrt(3.), 1. / np.sqrt(3.)):
        for eta in (-1. / np.sqrt(3.), 1. / np.sqrt(3.)):
            dx = np.array([-(1.-eta), 1.-eta, 1.+eta, -(1.+eta)]) / (2. * hx)
            dy = np.array([-(1.-xi), -(1.+xi), 1.+xi, 1.-xi]) / (2. * hy)
            B = np.zeros((3, 8))
            B[0, 0::2], B[1, 1::2] = dx, dy
            B[2, 0::2], B[2, 1::2] = dy, dx
            stiffness += (B.T @ constitutive @ B) * (hx * hy * thickness / 4.)
    return stiffness


@dataclass
class _LoadCase:
    free: np.ndarray
    fixed: np.ndarray
    loaded: np.ndarray
    prescribed: np.ndarray
    initial: np.ndarray
    preconditioner: object = None


class FiniteSpecimenFEM:
    """Cache mesh and load cases; call solve(image) repeatedly for a dataset.

    Recommended production settings: solver='amg', tolerance=1e-9.  The AMG
    hierarchy is built on a homogeneous mesh once and reused for both phases.
    'direct' does not require pyamg and is useful for independent cross-checks.
    """

    def __init__(self, shape=(224, 224), soft_modulus=1., hard_modulus=10.,
                 poisson=.3, width=40., height=40., thickness=2., strain=.001,
                 solver="amg", tolerance=1e-9, max_iterations=1200):
        self.shape = tuple(map(int, shape))
        self.ny, self.nx = self.shape
        if min(self.shape) < 1:
            raise ValueError("Mesh dimensions must be positive.")
        self.soft_modulus, self.hard_modulus = float(soft_modulus), float(hard_modulus)
        self.poisson, self.width, self.height = float(poisson), float(width), float(height)
        self.thickness, self.strain = float(thickness), float(strain)
        self.solver, self.tolerance = solver, float(tolerance)
        self.max_iterations = int(max_iterations)
        if min(self.soft_modulus, self.hard_modulus, self.width, self.height,
               self.thickness, self.strain) <= 0:
            raise ValueError("Material moduli, dimensions, and test strain must be positive.")
        if solver not in ("amg", "direct"):
            raise ValueError("solver must be 'amg' or 'direct'.")
        self.nnode = (self.nx + 1) * (self.ny + 1)
        self.ndof = 2 * self.nnode
        yy, xx = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        first = (yy * (self.nx + 1) + xx).ravel()
        nodes = np.stack([first, first + 1, first + self.nx + 2,
                          first + self.nx + 1], axis=1)
        self.edof = np.stack([2 * nodes, 2 * nodes + 1], axis=-1).reshape(-1, 8)
        self.rows = np.repeat(self.edof, 8, axis=1).ravel().astype(np.int32)
        self.cols = np.tile(self.edof, (1, 8)).ravel().astype(np.int32)
        self.ke = q4_stiffness(self.width / self.nx, self.height / self.ny,
                               self.poisson, self.thickness)
        yn, xn = np.meshgrid(np.linspace(0, self.height, self.ny + 1),
                             np.linspace(0, self.width, self.nx + 1), indexing="ij")
        self.x, self.y = xn.ravel(), yn.ravel()
        self.cases = [self._case(axis) for axis in (0, 1)]
        if solver == "amg":
            try:
                import pyamg
            except ImportError as exc:
                raise ImportError("Install pyamg==5.0.1 or use solver='direct'.") from exc
            baseline = self._assemble(np.ones(self.nx * self.ny))
            near_null = np.zeros((self.ndof, 3))
            near_null[0::2, 0] = 1.
            near_null[1::2, 1] = 1.
            near_null[0::2, 2] = -self.y / max(self.width, self.height)
            near_null[1::2, 2] = self.x / max(self.width, self.height)
            for case in self.cases:
                matrix = baseline[case.free][:, case.free].tocsr()
                hierarchy = pyamg.smoothed_aggregation_solver(
                    matrix, B=near_null[case.free], symmetry="symmetric",
                    max_coarse=80,
                    presmoother=("block_gauss_seidel", {"sweep": "symmetric"}),
                    postsmoother=("block_gauss_seidel", {"sweep": "symmetric"}))
                case.preconditioner = hierarchy.aspreconditioner(cycle="V")

    def _case(self, axis: int) -> _LoadCase:
        nodes = np.arange(self.nnode).reshape(self.ny + 1, self.nx + 1)
        if axis == 0:
            fixed_edge, loaded_nodes = nodes[:, 0], nodes[:, -1]
            length = self.width
        else:
            fixed_edge, loaded_nodes = nodes[0, :], nodes[-1, :]
            length = self.height
        fixed = np.unique(np.concatenate((2 * fixed_edge + axis,
                                         2 * loaded_nodes + axis, [1 - axis])))
        free = np.setdiff1d(np.arange(self.ndof), fixed, assume_unique=True)
        loaded = 2 * loaded_nodes + axis
        prescribed = np.zeros(self.ndof)
        prescribed[loaded] = self.strain * length
        initial = np.empty(self.ndof)
        initial[axis::2] = self.strain * (self.x if axis == 0 else self.y)
        initial[1-axis::2] = -self.poisson * self.strain * (self.y if axis == 0 else self.x)
        return _LoadCase(free, fixed, loaded, prescribed, initial)

    def _assemble(self, moduli: np.ndarray):
        values = (moduli[:, None] * self.ke.ravel()[None, :]).ravel()
        return sparse.coo_matrix((values, (self.rows, self.cols)),
                                 shape=(self.ndof, self.ndof)).tocsr()

    def solve(self, image, return_fields=False) -> dict:
        start = time.perf_counter()
        arr = np.asarray(image)
        if arr.shape != self.shape or not np.isin(arr, (0, 1)).all():
            raise ValueError(f"Expected binary image with shape {self.shape}.")
        moduli = np.where(np.flipud(arr).ravel() == 0, self.hard_modulus, self.soft_modulus)
        matrix = self._assemble(moduli)
        result = {"Ex_app": None, "Ey_app": None,
                  "hard_fraction": float(np.mean(arr == 0))}
        fields = []
        for axis, case in enumerate(self.cases):
            A = matrix[case.free][:, case.free].tocsr()
            rhs = -(matrix @ case.prescribed)[case.free]
            iterations = [0]
            def count_iteration(_):
                iterations[0] += 1
            if self.solver == "direct":
                solution = spsolve(A, rhs)
                info = 0
            else:
                solution, info = cg(A, rhs, x0=case.initial[case.free],
                                    M=case.preconditioner, tol=self.tolerance,
                                    atol=0., maxiter=self.max_iterations,
                                    callback=count_iteration)
            if info != 0:
                raise RuntimeError(f"CG failed in axis {axis}; info={info}.")
            u = case.prescribed.copy()
            u[case.free] = solution
            reactions = matrix @ u
            force = reactions[case.loaded].sum()
            cross_section = self.thickness * (self.height if axis == 0 else self.width)
            modulus = float(force / (cross_section * self.strain))
            relative_residual = float(np.linalg.norm(reactions[case.free]) /
                                      max(np.linalg.norm(rhs), 1e-30))
            energy_modulus = float(u @ reactions / (self.strain ** 2 *
                                                    self.width * self.height * self.thickness))
            label = "x" if axis == 0 else "y"
            result[f"E{label}_app"] = modulus
            result[f"residual_{label}"] = relative_residual
            result[f"iterations_{label}"] = iterations[0]
            result[f"energy_relative_error_{label}"] = float(abs(energy_modulus - modulus) /
                                                              max(abs(modulus), 1e-30))
            result[f"force_balance_relative_error_{label}"] = float(
                np.linalg.norm([reactions[0::2].sum(), reactions[1::2].sum()]) /
                max(abs(force), 1e-30))
            if not np.isfinite(modulus) or modulus <= 0 or relative_residual > max(1e-7, self.tolerance * 5):
                raise RuntimeError(f"Invalid FEM solution for axis {axis}: {result}")
            if (result[f"energy_relative_error_{label}"] > 1e-6 or
                    result[f"force_balance_relative_error_{label}"] > 1e-7):
                raise RuntimeError(f"Energy/reaction audit failed for axis {axis}: {result}")
            # For identical Poisson ratios, ordering E orders the complete
            # constitutive tensors and the apparent energy stiffness.
            lo, hi = sorted((self.soft_modulus, self.hard_modulus))
            if not lo * (1. - 1e-6) <= modulus <= hi * (1. + 1e-6):
                raise RuntimeError(f"Apparent modulus violates homogeneous bounds: {modulus}")
            if return_fields:
                fields.append(u.reshape(self.ny + 1, self.nx + 1, 2))
        result["solve_seconds"] = float(time.perf_counter() - start)
        result["fem_valid"] = True
        if return_fields:
            result["displacement_fields"] = np.stack(fields)
        return result


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Binary PNG or 2D NPY image")
    parser.add_argument("--solver", choices=("amg", "direct"), default="amg")
    arguments = parser.parse_args()
    # Small sparse iterative operations are faster without BLAS oversubscription.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    if Path(arguments.image).suffix.lower() == ".npy":
        input_image = np.load(arguments.image, allow_pickle=False)
    else:
        from PIL import Image
        pixels = np.asarray(Image.open(arguments.image).convert("L"))
        if not np.isin(pixels, (0, 255)).all():
            raise ValueError("Input PNG must contain strictly black and white pixels.")
        input_image = pixels // 255
    print(json.dumps(FiniteSpecimenFEM(input_image.shape, solver=arguments.solver).solve(input_image), indent=2))
