"""Optional reference-based physical objectives; NumPy preparation is Torch-free."""

from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import sisl

from .configuration import BasisConfiguration
from .matrices import OrbitalMatrix, get_matrix_cls
from .processing import BasisMatrixData
from .sparse import csr_to_block_dict

_HERMITICITY_RTOL = 1e-8


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
        scale = max(1., float(np.max(np.abs(original.data))) if original.nnz else 0.)
        if difference.nnz and np.max(np.abs(difference.data)) > _HERMITICITY_RTOL * scale:
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
        antisymmetric = sk - sk.conj().T
        scale = max(1., float(np.max(np.abs(sk))))
        if not np.isfinite(sk).all() or np.max(np.abs(antisymmetric)) > _HERMITICITY_RTOL * scale:
            raise ValueError("reference overlap must be finite and Hermitian")
        # Certify numerical positivity without changing the S used by observables.
        eigenvalues = np.linalg.eigvalsh((sk + sk.conj().T) / 2)
        uncertainty = np.linalg.norm(antisymmetric, ord="fro") / 2
        if eigenvalues[0] <= uncertainty or eigenvalues[0] / eigenvalues[-1] <= 1e-12:
            raise ValueError("reference overlap is singular or ill-conditioned")
    scalar_overlap = overlap.tocsr(0)
    reverse_overlap = overlap.transpose().tocsr(0)
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
        rows = sum(_weighted_reference_rows(density, c, reverse_overlap, "density") for c in range(spin))
        targets["population"] = atomic_sum(rows)
    if "energy" in requested:
        assert hamiltonian is not None and energy_density is not None
        reverse_density = density.transpose()
        rows = sum(_weighted_reference_rows(reverse_density, c, hamiltonian.tocsr(c), "density")
                   - _weighted_reference_rows(energy_density, c, reverse_overlap, "energy_density") for c in range(spin))
        targets["energy_residual"] = atomic_sum(rows)
        if np.max(np.abs(targets["energy_residual"])) > 1e-6:
            raise ValueError("physical energy reference residual exceeds 1e-6 eV/atom; check its energy gauge")
    return targets


def prepare_physical_data(
    processor, *, overlap, density, hamiltonian=None, energy_density=None,
    kpoints, kweights, requested, data_cls=BasisMatrixData,
):
    """Prepare validated physical references on exactly the prediction graph.

    Auxiliary matrices never enter model features. Missing DM/EDM coefficients
    remain unknown, including where S or H happens to vanish.
    """
    requested = normalize_physical_labels(requested)
    targets = physical_reference_targets(overlap, density, hamiltonian=hamiltonian,
        energy_density=energy_density, kpoints=kpoints, kweights=kweights, requested=requested)
    kinds = {"hamiltonian": (0, hamiltonian), "density_matrix": (1, density),
             "energy_density_matrix": (2, energy_density)}
    if processor.out_matrix not in kinds:
        raise ValueError("physical objectives require an H, DM or EDM prediction target")
    kind, target = kinds[processor.out_matrix]
    if target is None:
        raise ValueError("requested physical prediction target is absent")
    spin = int(targets["spin_components"][0])
    if kind == 1 and spin == 2 and processor.sub_point_matrix:
        raise ValueError("collinear density baseline is not certified")
    _validate_geometry(overlap.geometry, target, "prediction target")
    if _physical_channels(target, "prediction target") != spin:
        raise ValueError("physical prediction target spin does not match references")

    def blocks(matrix, channels, fill_value):
        return csr_to_block_dict(matrix._csr, matrix.atoms, nsc=matrix.nsc,
            matrix_cls=OrbitalMatrix if isinstance(matrix, sisl.Overlap) else get_matrix_cls(type(matrix)),
            fill_value=fill_value,
            component_indices=range(channels))

    config = BasisConfiguration.new(target)
    physical_blocks = blocks(target, spin, np.nan if kind in (1, 2) else 0.)
    # Only Hamiltonian gauge labels retain S; the physical reference never does.
    if kind == 0:
        config = BasisConfiguration.new(target, matrix_component_policy=processor.matrix_component_policy)
    else:
        config = replace(config, matrix=physical_blocks)
    data = data_cls.from_config(config, processor)
    arrays = data.numpy_arrays()
    edge_index = arrays.edge_index[:, ::2] if processor.symmetric_matrix else arrays.edge_index
    images = arrays.neigh_isc[::2] if processor.symmetric_matrix else arrays.neigh_isc

    def flatten(matrix, channels, fill_value):
        values = blocks(matrix, channels, fill_value).to_flat_nodes_and_edges(
            edge_index=edge_index, edge_sc_shifts=images)
        return tuple(value.reshape(-1, channels) for value in values)

    ref_n, ref_e = flatten(target, spin, np.nan if kind in (1, 2) else 0.)
    overlap_n, overlap_e = flatten(overlap.transpose(), 1, 0.)
    _, overlap_reverse_e = flatten(overlap, 1, 0.)
    auxiliary = {
        "physical_version": np.array([1], dtype=np.int64),
        "physical_kind": np.array([kind], dtype=np.int64),
        "physical_spin": targets["spin_components"],
        "physical_capabilities": targets["capabilities"],
        "physical_reference_nodes": ref_n, "physical_reference_edges": ref_e,
        "physical_reference_known_nodes": np.isfinite(ref_n),
        "physical_reference_known_edges": np.isfinite(ref_e),
        "physical_overlap_nodes": overlap_n, "physical_overlap_edges": overlap_e,
        "physical_overlap_reverse_edges": overlap_reverse_e,
    }
    if kind == 1:
        baseline = np.zeros_like(ref_n)
        if processor.sub_point_matrix:
            baseline[:, 0] = np.concatenate([
                processor.basis_table.atomic_DM[point_type].reshape(-1)
                for point_type in arrays.point_types
                if processor.basis_table.basis_size[point_type] > 0])
        auxiliary["physical_baseline_nodes"] = baseline
    if "charge" in requested:
        auxiliary["physical_population_reference"] = targets["population"]
        if kind == 1 and (np.any(~np.isfinite(ref_n) & (overlap_n != 0))
                or np.any(~np.isfinite(ref_e) & ((overlap_e != 0) | (overlap_reverse_e != 0)))):
            raise ValueError("unknown DM reference with nonzero charge partner on the graph")
    if "energy" in requested:
        partner = (density, hamiltonian, overlap)[kind]
        channels = 1 if kind == 2 else spin
        unknown_partner = kind == 0
        pn, pe = flatten(partner.transpose(), channels, np.nan if unknown_partner else 0.)
        _, reverse_pe = flatten(partner, channels, np.nan if unknown_partner else 0.)
        if unknown_partner:
            if not (np.isfinite(pn).all() and np.isfinite(pe).all() and np.isfinite(reverse_pe).all()):
                raise ValueError("unknown/unobserved density partner where predicted H can vary")
        elif (np.any(~np.isfinite(ref_n) & (pn != 0))
                or np.any(~np.isfinite(ref_e) & ((pe != 0) | (reverse_pe != 0)))):
            raise ValueError("unknown target reference with nonzero energy partner on the graph")
        auxiliary.update(physical_partner_nodes=pn, physical_partner_edges=pe,
            physical_partner_reverse_edges=reverse_pe,
            physical_partner_known_nodes=np.isfinite(pn), physical_partner_known_edges=np.isfinite(pe),
            physical_partner_known_reverse_edges=np.isfinite(reverse_pe),
            physical_energy_reference_residual=targets["energy_residual"])
    for name, array in auxiliary.items():
        value = data.process_input_array(name, np.ascontiguousarray(array))
        setattr(data, name, value)
        data._data[name] = value
    physical_layout(data, processor.basis_table)
    return data


def load_physical_data(path, processor, *, requested, data_cls=BasisMatrixData):
    """Load local FDF/TSHS/TSDE with an explicit diagonal, zero-offset MP grid.

    TSDE supplies no independent geometry or overlap. Its physical channels are
    read without overlap injection, preserving the original observation support.
    """
    requested = normalize_physical_labels(requested)
    try:
        path = Path(path)
    except TypeError as error:
        raise ValueError("physical labels require local FDF or explicitly prepared data") from error
    if path.suffix.lower() != ".fdf":
        raise ValueError("physical labels require a local FDF or explicitly prepared data")
    fdf = sisl.get_sile(path)
    try:
        grid = np.asarray([line.split() for line in fdf.get("kgrid_Monkhorst_Pack", default=[])], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: missing or unsupported explicit kpoint grid") from error
    if (grid.shape != (3, 4) or not np.isfinite(grid).all()
            or np.any(grid[:, 3] != 0) or np.any(grid[:, :3] != np.diag(np.diag(grid[:, :3])))
            or np.any(np.diag(grid[:, :3]) <= 0) or np.any(grid[:, :3] != np.round(grid[:, :3]))):
        raise ValueError(f"{path}: physical labels require a positive diagonal zero-offset kpoint grid")
    if not requested or ("energy" not in requested and processor.out_matrix != "density_matrix"):
        raise ValueError(f"{path}: charge-only preparation requires a DM prediction target")
    label = fdf.get("SystemLabel", default="siesta")
    sources = {name: Path(fdf.dir_file(label + suffix)) for name, suffix in (
        ("overlap", ".TSHS"), ("density", ".TSDE"))}
    for name, source in sources.items():
        if not source.is_file():
            raise ValueError(f"{path}: physical reference {name} is absent: {source}")
    try:
        overlap = fdf.read_overlap(order=["TSHS"])
        reader = sisl.get_sile(sources["density"])

        def physical_channels(matrix, name):
            channels = _physical_channels(matrix, name)
            return type(matrix).fromsp(matrix.geometry, [matrix.tocsr(c) for c in range(channels)])

        density = physical_channels(reader.read_density_matrix(geometry=overlap.geometry.copy()), "density")
        hamiltonian = energy_density = None
        if "energy" in requested:
            tshs_fermi = sisl.get_sile(sources["overlap"]).read_fermi_level()
            tsde_fermi = reader.read_fermi_level()
            if not (np.isfinite(tshs_fermi) and np.isfinite(tsde_fermi)
                    and np.isclose(tshs_fermi, tsde_fermi, rtol=0, atol=1e-8)):
                raise ValueError("TSHS/TSDE Fermi levels do not certify a common energy gauge")
            hamiltonian = fdf.read_hamiltonian(order=["TSHS"], geometry=overlap.geometry.copy())
            energy_density = physical_channels(reader.read_energy_density_matrix(
                geometry=overlap.geometry.copy()), "energy_density")
        mesh = sisl.MonkhorstPack(overlap.geometry, np.diag(grid[:, :3]).astype(int), centered=True, trs=False)
        data = prepare_physical_data(processor, overlap=overlap, density=density,
            hamiltonian=hamiltonian, energy_density=energy_density, kpoints=mesh.k,
            kweights=mesh.weight, requested=requested, data_cls=data_cls)
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError(f"{path}: physical reference loading failed: {error}") from error
    data.metadata["physical_reference"] = {"fdf": str(path),
        **{name: str(source) for name, source in sources.items()},
        "kpoints": mesh.k.copy(), "kweights": mesh.weight.copy()}
    return data


def physical_layout(batch, basis_table):
    """CPU ownership of flattened label rows, retaining orbital-less atoms."""
    from .metrics import _as_numpy_int, _batch_field, _label_edge_mask, _node_graph, _num_structures

    point_types = _as_numpy_int(_batch_field(batch, "point_types"))
    edge_types = _as_numpy_int(_batch_field(batch, "edge_types"))
    edge_index = _as_numpy_int(_batch_field(batch, "edge_index"))
    sizes = basis_table.basis_size[point_types]
    atom_structure = _node_graph(batch)
    count = _num_structures(batch, atom_structure)
    atoms_per_structure = np.bincount(atom_structure, minlength=count)
    orbital_counts = np.bincount(atom_structure, weights=sizes, minlength=count)
    if np.any(atoms_per_structure == 0) or np.any(orbital_counts == 0):
        raise ValueError("physical objectives require nonempty structures and bases")
    labelled = _label_edge_mask(_batch_field(batch, "edge_labels"), edge_types, edge_index, point_types, basis_table)
    sources, targets = edge_index[:, labelled]
    rows = sizes[sources] * sizes[targets]
    return dict(physical_node_owner=np.repeat(np.arange(len(point_types)), sizes * sizes),
        physical_edge_source=np.repeat(sources, rows), physical_edge_target=np.repeat(targets, rows),
        physical_atom_structure=atom_structure, physical_atoms_per_structure=atoms_per_structure,
        physical_symmetric=bool(not labelled.all()))
