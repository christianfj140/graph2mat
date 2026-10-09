"""Optional reference-based physical objectives; NumPy preparation is Torch-free."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import sisl


def normalize_physical_labels(labels: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """Select charge/energy preparation explicitly, preserving the default off path."""
    if labels is None:
        return ()
    if isinstance(labels, str) or any(x not in ("charge", "energy") for x in labels):
        raise ValueError("physical_labels must contain charge and/or energy")
    return tuple(x for x in ("charge", "energy") if x in labels)


def validate_physical_weights(
    electron_weight: float = 0., mulliken_weight: float = 0., energy_weight: float = 0.,
) -> Tuple[float, float, float]:
    """Reject nonfinite or negative physical weights instead of disabling a term."""
    try:
        values = tuple(float(x) for x in (electron_weight, mulliken_weight, energy_weight))
    except (TypeError, ValueError) as error:
        raise ValueError("physical weights must be finite and nonnegative") from error
    if any(not np.isfinite(x) or x < 0 for x in values):
        raise ValueError("physical weights must be finite and nonnegative")
    return values[0], values[1], values[2]


def _validate_mesh(kpoints, kweights):
    points, weights = np.asarray(kpoints, dtype=float), np.asarray(kweights, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("reference kpoint mesh must be finite with shape (n, 3)")
    if (weights.shape != (len(points),) or not np.isfinite(weights).all()
            or (weights < 0).any() or not (weights > 0).any()
            or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("reference kpoint weights must be nonnegative and sum to one")
    return points, weights


def _validate_geometry(reference, matrix, name):
    geometry = matrix.geometry
    if (reference.na != geometry.na or reference.no != geometry.no
            or not np.allclose(reference.xyz, geometry.xyz, rtol=0, atol=1e-8)
            or not np.allclose(reference.cell, geometry.cell, rtol=0, atol=1e-8)):
        raise ValueError(f"{name} geometry/cell does not match overlap")
    if (not np.array_equal(reference.nsc, geometry.nsc)
            or not np.array_equal(reference.sc_off, geometry.sc_off)):
        raise ValueError(f"{name} nsc/image ordering does not match overlap")
    if any(not a.equal(b, R=True) for a, b in zip(reference.atoms, geometry.atoms)):
        raise ValueError(f"{name} basis does not match overlap")


def _physical_channels(matrix, name):
    if np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError(f"{name} physical objectives require real, not complex coefficients")
    if isinstance(matrix, sisl.Overlap):
        return 1
    if not (matrix.spin.is_unpolarized or matrix.spin.is_polarized):
        raise ValueError(f"{name} physical objectives require unpolarized or collinear spin")
    return 1 if matrix.spin.is_unpolarized else 2


def _validate_hermitian(matrix, channels, name, *, allow_unknown=False):
    transposed = matrix.transpose()
    for component in range(channels):
        original = matrix.tocsr(component).astype(np.float64)
        opposite = transposed.tocsr(component).astype(np.float64)
        if np.isinf(original.data).any() or (not allow_unknown and np.isnan(original.data).any()):
            raise ValueError(f"{name} reference must be finite")
        # Unknown coefficients are not zero; compare masks before finite values.
        unknown, other_unknown = original.copy(), opposite.copy()
        unknown.data = np.isnan(unknown.data).astype(np.int8)
        other_unknown.data = np.isnan(other_unknown.data).astype(np.int8)
        if (unknown != other_unknown).nnz:
            raise ValueError(f"{name} reference unknown mask is not Hermitian")
        original.data[np.isnan(original.data)] = 0.
        opposite.data[np.isnan(opposite.data)] = 0.
        difference = original - opposite
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-10:
            raise ValueError(f"{name} reference is not Hermitian across opposite images")


def _observed_sparse(matrix, component):
    """Stored finite coefficients, including explicit zeros, are observations."""
    observed = matrix.tocsr(component).copy()
    observed.data = np.isfinite(observed.data).astype(np.int8)
    observed.eliminate_zeros()
    return observed


def _weighted_reference_rows(matrix, component, weights, name):
    """Read unknown-capable DM/EDM only on a nonzero fixed partner's support."""
    weights = weights.copy()
    weights.eliminate_zeros()
    coordinates = weights.tocoo()
    observed = _observed_sparse(matrix, component)
    known = np.asarray(observed[coordinates.row, coordinates.col]).ravel()
    if not np.all(known == 1):
        raise ValueError(f"{name} has unknown/unobserved coefficients with a nonzero physical partner")
    values = np.asarray(matrix.tocsr(component)[coordinates.row, coordinates.col]).ravel()
    result = np.zeros(matrix.geometry.no, dtype=np.float64)
    np.add.at(result, coordinates.row, values * coordinates.data)
    return result


def physical_reference_targets(
    overlap: sisl.Overlap, density: sisl.DensityMatrix, *,
    hamiltonian: Optional[sisl.Hamiltonian] = None,
    energy_density: Optional[sisl.EnergyDensityMatrix] = None,
    kpoints: np.ndarray, kweights: np.ndarray, requested: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Validate real references and contract all periodic images, grouped by atom.

    S is checked in float64 on the explicitly declared reference mesh, without
    clipping. Occupations may be fractional; no idempotency or extra spin factor
    is imposed. Energy is HD minus ES in the supplied common energy gauge.
    """
    requested = normalize_physical_labels(requested)
    if not requested:
        raise ValueError("physical reference preparation needs charge and/or energy")
    points, _ = _validate_mesh(kpoints, kweights)
    matrices = {"overlap": (overlap, sisl.Overlap), "density": (density, sisl.DensityMatrix)}
    if "energy" in requested:
        matrices.update(hamiltonian=(hamiltonian, sisl.Hamiltonian),
                        energy_density=(energy_density, sisl.EnergyDensityMatrix))
    channels = {}
    for name, (matrix, cls) in matrices.items():
        if matrix is None or not isinstance(matrix, cls):
            raise ValueError(f"physical reference {name} is missing or has the wrong matrix type")
        channels[name] = _physical_channels(matrix, name)
    geometry = overlap.geometry
    if geometry.na == 0 or geometry.no == 0:
        raise ValueError("physical references require nonempty geometry and basis")
    for name, (matrix, _) in matrices.items():
        _validate_geometry(geometry, matrix, name)
        _validate_hermitian(matrix, channels[name], name, allow_unknown=name in ("density", "energy_density"))
    spin = channels["density"]
    if any(channels[name] != spin for name in channels if name != "overlap"):
        raise ValueError("physical reference spin components do not agree")
    for kpoint in points:
        sk = np.asarray(overlap.Sk(kpoint, format="array"), dtype=np.complex128)
        if not np.isfinite(sk).all() or not np.allclose(sk, sk.conj().T, rtol=0, atol=1e-12):
            raise ValueError("reference overlap must be finite and Hermitian")
        eigenvalues = np.linalg.eigvalsh(sk)
        if eigenvalues[0] <= 0 or eigenvalues[0] / eigenvalues[-1] <= 1e-12:
            raise ValueError("reference overlap is singular or ill-conditioned")
    scalar_overlap = overlap.tocsr(0)
    for name, (matrix, _) in matrices.items():
        if name == "overlap" or matrix.orthogonal:
            continue
        difference = matrix.tocsr(matrix.S_idx) - scalar_overlap
        if difference.nnz and (not np.isfinite(difference.data).all() or np.max(np.abs(difference.data)) > 1e-10):
            raise ValueError(f"{name} embedded overlap does not match the reference overlap")

    owners = geometry.o2a(np.arange(geometry.no))

    def atomic_sum(rows):
        population = np.zeros(geometry.na, dtype=np.float64)
        np.add.at(population, owners, rows)
        return population

    targets = {"spin_components": np.array([spin], dtype=np.int64),
               "capabilities": np.array([int("charge" in requested) + 2*int("energy" in requested)], dtype=np.int64)}
    if "charge" in requested:
        rows = sum(_weighted_reference_rows(density, c, scalar_overlap, "density") for c in range(spin))
        targets["population"] = atomic_sum(rows)
    if "energy" in requested:
        assert hamiltonian is not None and energy_density is not None
        rows = sum(_weighted_reference_rows(density, c, hamiltonian.tocsr(c), "density")
                   - _weighted_reference_rows(energy_density, c, scalar_overlap, "energy_density") for c in range(spin))
        targets["energy_residual"] = atomic_sum(rows)
        if np.max(np.abs(targets["energy_residual"])) > 1e-6:
            raise ValueError("physical energy reference residual exceeds 1e-6 eV/atom; check its energy gauge")
    return targets
