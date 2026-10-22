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
    kpoints, kweights, requested, data_cls=BasisMatrixData, _data=None,
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
    # Joint targets reuse their canonical graph; single-target callers build it as before.
    data = data_cls.from_config(config, processor) if _data is None else _data
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


def _read_reference_bundle(path, *, need_density, need_hamiltonian,
                           need_energy_density, require_mesh, processor=None):
    """Read coincident references; never use TSDE's placeholder overlap."""
    try:
        path = Path(path)
    except TypeError as error:
        raise ValueError("physical labels require local FDF or explicitly prepared data") from error
    if path.suffix.lower() != ".fdf":
        raise ValueError("physical labels require a local FDF or explicitly prepared data")
    fdf = sisl.get_sile(path)
    if require_mesh:
        try:
            grid = np.asarray([line.split() for line in fdf.get("kgrid_Monkhorst_Pack", default=[])], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: missing or unsupported explicit kpoint grid") from error
        if (grid.shape != (3, 4) or not np.isfinite(grid).all()
                or np.any(grid[:, 3] != 0) or np.any(grid[:, :3] != np.diag(np.diag(grid[:, :3])))
                or np.any(np.diag(grid[:, :3]) <= 0) or np.any(grid[:, :3] != np.round(grid[:, :3]))):
            raise ValueError(f"{path}: physical labels require a positive diagonal zero-offset kpoint grid")
    label = fdf.get("SystemLabel", default="siesta")
    sources = {"overlap": Path(fdf.dir_file(label + ".TSHS"))}
    if need_density or need_energy_density:
        sources["density"] = Path(fdf.dir_file(label + ".TSDE"))
    for name, source in sources.items():
        if not source.is_file():
            raise ValueError(f"{path}: physical reference {name} is absent: {source}")
    try:
        geometry_kwargs = {}
        if processor is not None:
            geometry = processor.add_basis_to_geometry(fdf.read_geometry(output=False))
            geometry_kwargs["geometry"] = geometry
        overlap = fdf.read_overlap(order=["TSHS"], **geometry_kwargs)
        reader = sisl.get_sile(sources["density"]) if "density" in sources else None

        def physical_channels(matrix, name):
            channels = _physical_channels(matrix, name)
            return type(matrix).fromsp(matrix.geometry, [matrix.tocsr(c) for c in range(channels)])

        density = None
        if need_density:
            assert reader is not None  # "density" is in sources whenever a density is needed
            density = physical_channels(reader.read_density_matrix(geometry=overlap.geometry.copy()), "density")
        hamiltonian = (fdf.read_hamiltonian(order=["TSHS"], geometry=overlap.geometry.copy())
                       if need_hamiltonian else None)
        energy_density = None
        if need_energy_density:
            assert reader is not None
            tshs_fermi = sisl.get_sile(sources["overlap"]).read_fermi_level()
            tsde_fermi = reader.read_fermi_level()
            if not (np.isfinite(tshs_fermi) and np.isfinite(tsde_fermi)
                    and np.isclose(tshs_fermi, tsde_fermi, rtol=0, atol=1e-8)):
                raise ValueError("TSHS/TSDE Fermi levels do not certify a common energy gauge")
            energy_density = physical_channels(reader.read_energy_density_matrix(
                geometry=overlap.geometry.copy()), "energy_density")
        mesh = (sisl.MonkhorstPack(overlap.geometry, np.diag(grid[:, :3]).astype(int), centered=True, trs=False)
                if require_mesh else None)
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError(f"{path}: physical reference loading failed: {error}") from error
    metadata = {"fdf": str(path),
        **{name: str(source) for name, source in sources.items()},
        **({"kpoints": mesh.k.copy(), "kweights": mesh.weight.copy()} if mesh is not None else {})}
    return dict(overlap=overlap, density=density, hamiltonian=hamiltonian,
                energy_density=energy_density, kpoints=None if mesh is None else mesh.k,
                kweights=None if mesh is None else mesh.weight), metadata


def load_physical_data(path, processor, *, requested, data_cls=BasisMatrixData):
    """Load local FDF/TSHS/TSDE and validated physical reference objectives."""
    requested = normalize_physical_labels(requested)
    if not requested or ("energy" not in requested and processor.out_matrix != "density_matrix"):
        raise ValueError(f"{path}: charge-only preparation requires a DM prediction target")
    bundle, metadata = _read_reference_bundle(path, need_density=True,
        need_hamiltonian="energy" in requested, need_energy_density="energy" in requested,
        require_mesh=True)
    try:
        data = prepare_physical_data(processor, **bundle, requested=requested, data_cls=data_cls)
    except (OSError, ValueError, AttributeError) as error:
        raise ValueError(f"{path}: physical reference loading failed: {error}") from error
    data.metadata["physical_reference"] = metadata
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


def _physical_array(value, like, dtype=None):
    from .metrics import _is_torch_tensor

    if _is_torch_tensor(value):
        if value.is_complex():
            raise ValueError("physical arrays must be real")
    elif np.iscomplexobj(value):
        raise ValueError("physical arrays must be real")
    if _is_torch_tensor(like):
        import torch
        return torch.as_tensor(value, device=like.device,
            dtype=getattr(torch, dtype) if dtype else like.dtype)
    if _is_torch_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype or like.dtype)


def _sum_by_structure(values, owners, count):
    from .metrics import _is_torch_tensor

    if _is_torch_tensor(values):
        return values.new_zeros(count).index_add(0, owners, values)
    result = np.zeros(count, dtype=values.dtype)
    np.add.at(result, owners, values)
    return result


def physical_contractions(nodes, edges, partner_nodes, partner_edges, *, layout,
                          reverse_partner_edges=None):
    """Atomic real-space traces; half edges contribute to both endpoints.

    The reverse partner retains the opposite archived orientation. NaN values
    are excluded only at exactly zero fixed partners, before multiplication.
    """
    from .metrics import _is_torch_tensor

    is_torch = _is_torch_tensor(nodes)
    if is_torch != _is_torch_tensor(edges) or nodes.dtype != edges.dtype:
        raise ValueError("physical node/edge predictions need the same backend and dtype")
    if nodes.ndim != 2 or edges.ndim != 2 or nodes.shape[1] != edges.shape[1]:
        raise ValueError("physical contractions require (rows, spin) shapes")
    if is_torch:
        import torch
        if not nodes.is_floating_point() or nodes.device != edges.device:
            raise ValueError("physical predictions must be floating point on the same device")
        finite, where = torch.isfinite, torch.where
    else:
        if nodes.dtype.kind != "f":
            raise ValueError("physical predictions must be real floating point")
        finite, where = np.isfinite, np.where
    atoms = len(layout["physical_atom_structure"])

    def owner(name, rows):
        result = _physical_array(layout[name], nodes, "int64")
        if result.shape != (rows,) or bool(((result < 0) | (result >= atoms)).any()):
            raise ValueError(f"physical layout {name} has incompatible shape/owners")
        return result

    def contract(values, fixed):
        fixed = _physical_array(fixed, values)
        if (fixed.ndim != 2 or fixed.shape[0] != values.shape[0]
                or fixed.shape[1] not in (1, values.shape[1])):
            raise ValueError("physical partner shape must be (rows, 1 or spin)")
        if not bool(finite(fixed).all()):
            raise ValueError("physical fixed partners must be finite and observed")
        selected = where(fixed != 0, values, 0.)
        if not bool(finite(selected).all()):
            raise ValueError("unknown/nonfinite coefficient with nonzero physical partner")
        return (selected * fixed).sum(axis=-1)

    result = _sum_by_structure(contract(nodes, partner_nodes), owner("physical_node_owner", len(nodes)), atoms)
    result = result + _sum_by_structure(contract(edges, partner_edges), owner("physical_edge_source", len(edges)), atoms)
    if layout["physical_symmetric"]:
        reverse = partner_edges if reverse_partner_edges is None else reverse_partner_edges
        result = result + _sum_by_structure(contract(edges, reverse), owner("physical_edge_target", len(edges)), atoms)
    return result


def physical_loss_terms(nodes_pred, edges_pred, *, batch, basis_table,
                        electron_weight=0., mulliken_weight=0., energy_weight=0.,
                        aligned_nodes=None, aligned_edges=None):
    """Optional structure-mean charge/Mulliken and energy perturbation objectives.

    Charge compares the represented matrix to full reference populations. Energy
    continues the reference outside the graph, not the writer's truncated matrix.
    Fixed arrays are validated; the prediction path stays sparse and differentiable.
    """
    from .metrics import _batch_field, _is_torch_tensor
    from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

    electron_weight, mulliken_weight, energy_weight = validate_physical_weights(
        electron_weight, mulliken_weight, energy_weight)
    is_torch = _is_torch_tensor(nodes_pred)
    if is_torch:
        import torch
        finite, where = torch.isfinite, torch.where
        zero = nodes_pred.new_zeros(())
    else:
        finite, where = np.isfinite, np.where
        zero = np.asarray(0., dtype=nodes_pred.dtype)
    if not any((electron_weight, mulliken_weight, energy_weight)):
        return zero, {}
    if batch is None or basis_table is None or _batch_field(batch, "physical_version") is None:
        raise ValueError("physical objectives require prepared references and a basis table")
    layout = _batch_field(batch, LAYOUT_ATTR) or {}
    if "physical_atom_structure" not in layout:
        layout = physical_layout(batch, basis_table)
    layout = {name: _physical_array(value, nodes_pred, "int64")
        if name != "physical_symmetric" else value
        for name, value in layout.items() if name.startswith("physical_")}
    structure = layout["physical_atom_structure"]
    atoms_per_structure = layout["physical_atoms_per_structure"]
    structures, atoms = len(atoms_per_structure), len(structure)
    if structures == 0 or bool((atoms_per_structure <= 0).any()):
        raise ValueError("physical objectives require nonempty structures")

    def meta(name, allowed):
        raw = _batch_field(batch, name)
        if raw is None:
            raise ValueError(f"missing physical reference field {name}")
        raw = raw.detach().cpu().numpy() if _is_torch_tensor(raw) else np.asarray(raw)
        if raw.shape != (structures,) or raw.dtype.kind not in "iu" or not np.isin(raw, allowed).all():
            raise ValueError(f"invalid physical reference {name} shape/value")
        return raw

    meta("physical_version", (1,))
    kinds, spins = meta("physical_kind", (0, 1, 2)), meta("physical_spin", (1, 2))
    if not (np.all(kinds == kinds[0]) and np.all(spins == spins[0])):
        raise ValueError("physical prediction kind/spin must agree within a batch")
    kind, spin = int(kinds[0]), int(spins[0])
    required = int(bool(electron_weight or mulliken_weight)) + 2*int(bool(energy_weight))
    capabilities = meta("physical_capabilities", (1, 2, 3))
    if np.any(capabilities & required != required):
        raise ValueError("physical charge/energy reference capability is absent")
    if (electron_weight or mulliken_weight) and kind != 1:
        raise ValueError("charge objectives require a density/DM prediction")

    nodes = nodes_pred[:, None] if nodes_pred.ndim == 1 else nodes_pred
    edges = edges_pred[:, None] if edges_pred.ndim == 1 else edges_pred
    expected_components = (spin, spin+1) if kind == 0 else (spin,)
    if (nodes.ndim != 2 or edges.ndim != 2 or nodes.shape[1] not in expected_components
            or nodes.shape[1] != edges.shape[1]):
        raise ValueError("physical prediction shape does not match its spin contract")
    nodes, edges = nodes[:, :spin], edges[:, :spin]

    def field(name, shape, *, mask=False, allow_unknown=False):
        raw = _batch_field(batch, name)
        if raw is None:
            raise ValueError(f"missing physical reference field {name}")
        if mask:
            bool_dtype = str(raw.dtype) == "torch.bool" if _is_torch_tensor(raw) else np.asarray(raw).dtype == bool
            if not bool_dtype:
                raise ValueError(f"physical reference mask {name} must be boolean")
        array = _physical_array(raw, nodes, "bool" if mask else None)
        if array.shape != shape:
            raise ValueError(f"physical reference {name} shape {tuple(array.shape)} != {shape}")
        if not mask and not allow_unknown and not bool(finite(array).all()):
            raise ValueError(f"physical reference {name} must be finite")
        return array

    rn = field("physical_reference_nodes", tuple(nodes.shape), allow_unknown=True)
    re = field("physical_reference_edges", tuple(edges.shape), allow_unknown=True)
    kn = field("physical_reference_known_nodes", tuple(nodes.shape), mask=True)
    ke = field("physical_reference_known_edges", tuple(edges.shape), mask=True)
    if bool((kn & ~finite(rn)).any()) or bool((ke & ~finite(re)).any()):
        raise ValueError("physical observed reference coefficients must be finite")
    baseline = field("physical_baseline_nodes", tuple(nodes.shape)) if kind == 1 else 0.
    nodes = nodes + baseline
    stats = {"physical_structures": structures}
    loss = zero

    def detached(value):
        return value.detach() if _is_torch_tensor(value) else value

    def clean(values, known, partner, reverse=None):
        nonzero = partner != 0
        if reverse is not None and layout["physical_symmetric"]:
            nonzero = nonzero | (reverse != 0)
        if bool((~known & nonzero).any()):
            raise ValueError("unknown/unobserved reference with nonzero physical partner")
        if bool((known & ~finite(values)).any()):
            raise ValueError("physical predictions must be finite on observed reference coefficients")
        return where(known, values, 0.)

    if electron_weight or mulliken_weight:
        sn = field("physical_overlap_nodes", (len(nodes), 1))
        se = field("physical_overlap_edges", (len(edges), 1))
        sr = field("physical_overlap_reverse_edges", (len(edges), 1))
        population = physical_contractions(clean(nodes, kn, sn), clean(edges, ke, se, sr),
            sn, se, layout=layout, reverse_partner_edges=sr)
        delta = population - field("physical_population_reference", (atoms,))
        delta_n = _sum_by_structure(delta, structure, structures)
        electron = ((delta_n/atoms_per_structure)**2).mean()
        mulliken = (_sum_by_structure(delta**2, structure, structures)/atoms_per_structure).mean()
        loss = loss + electron_weight*electron + mulliken_weight*mulliken
        if electron_weight:
            stats["physical_electron_mse"] = detached(electron)
        if mulliken_weight:
            stats["physical_mulliken_mse"] = detached(mulliken)
        stats.update(physical_delta_n_mean=detached(delta_n.mean()),
            physical_delta_n_rmse=detached((delta_n**2).mean()**.5))
    if energy_weight:
        channels = 1 if kind == 2 else spin
        pn = field("physical_partner_nodes", (len(nodes), channels))
        pe = field("physical_partner_edges", (len(edges), channels))
        pr = field("physical_partner_reverse_edges", (len(edges), channels))
        for name, shape in (("physical_partner_known_nodes", tuple(pn.shape)),
                            ("physical_partner_known_edges", tuple(pe.shape)),
                            ("physical_partner_known_reverse_edges", tuple(pr.shape))):
            if not bool(field(name, shape, mask=True).all()):
                raise ValueError("physical fixed partner must be observed, never an invented zero")
        if (aligned_nodes is None) != (aligned_edges is None):
            raise ValueError("physical aligned node/edge deltas must be supplied together")
        if aligned_nodes is not None:
            if kind != 0 or aligned_nodes.shape != nodes.shape or aligned_edges.shape != edges.shape:
                raise ValueError("physical gauge-aligned deltas require the H prediction shape")
            masks = []
            for name, known in (("point_labels", kn), ("edge_labels", ke)):
                labels = _physical_array(_batch_field(batch, name), nodes)
                valid = finite(labels).all(axis=-1) if labels.ndim == 2 else finite(labels)
                masks.append(known & valid[:, None])
            kn, ke = masks
            dn, de = aligned_nodes, aligned_edges
        else:
            dn, de = nodes-rn, edges-re
        perturbation = physical_contractions(clean(dn, kn, pn), clean(de, ke, pe, pr),
            pn, pe, layout=layout, reverse_partner_edges=pr)
        reference = field("physical_energy_reference_residual", (atoms,))
        if bool((abs(reference) > 1e-6).any()):
            raise ValueError("physical energy reference residual exceeds its certified bound")
        residual = reference + (-1. if kind == 2 else 1.)*perturbation
        energy = (_sum_by_structure(residual**2, structure, structures)/atoms_per_structure).mean()
        loss = loss + energy_weight*energy
        stats["physical_energy_mse"] = detached(energy)
    return loss, stats
