"""Conversion between different sparse representations.

Different sparse representations of a matrix are required during the different
steps of a typical workflow using `graph2mat`.
"""
from typing import Dict, Tuple, Type, Optional, Callable, Any, Sequence, Union, TypeVar, List

from functools import partial

import numpy as np
from numpy.typing import ArrayLike
import sisl
from sisl.physics.sparse import SparseOrbital
from scipy.sparse import coo_array, csr_array

from .matrices import BasisMatrix, OrbitalMatrix
from .formats import Formats, conversions
from ._sparse import _csr_to_block_dict
from .spin_orbit import SOC_TRANSPOSE_SIGN


def register_all_sisl(source, target, converter, manager):
    if target != Formats.SISL:
        return

    specific_targets = [
        (Formats.SISL_H, sisl.Hamiltonian),
        (Formats.SISL_DM, sisl.DensityMatrix),
        (Formats.SISL_EDM, sisl.EnergyDensityMatrix),
    ]

    for specific_target, sp_class in specific_targets:
        if manager.has_converter(source, specific_target):
            continue

        specific_converter = partial(converter, sp_class=sp_class)
        manager.register_converter(source, specific_target, specific_converter)


conversions.add_callback(register_all_sisl)

# -----------------------------------------------
#            Conversion functions
# -----------------------------------------------

converter = conversions.converter
_ArrayType = TypeVar("_ArrayType", bound=Any)


@converter
def _csr_to_numpy(csr: csr_array) -> np.ndarray:
    return csr.toarray()


@converter
def _coo_to_csr(coo: coo_array) -> csr_array:
    return coo.tocsr()


@converter
def _csr_to_coo(csr: csr_array) -> coo_array:
    return csr.tocoo()


@converter
def _coo_to_numpy(coo: coo_array) -> np.ndarray:
    return coo.toarray()


@converter(Formats.SCIPY_CSR, Formats.BASISMATRIX)
def csr_to_block_dict(
    spmat: sisl.SparseCSR,
    atoms: sisl.Atoms,
    nsc: np.ndarray,
    geometry_atoms: Optional[sisl.Atoms] = None,
    matrix_cls: Type[BasisMatrix] = OrbitalMatrix,
    fill_value: float = 0,
    component_indices: Optional[Sequence[int]] = None,
    component_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> BasisMatrix:
    """Creates a BasisMatrix object from a SparseCSR matrix

    Parameters
    ----------
    spmat
        The sparse matrix to convert to a block dictionary.
    atoms
        The atoms object for the matrix, containing orbital information.
    nsc
        The auxiliary supercell size.
    matrix_cls
        Matrix class to initialize.
    geometry_atoms
        The atoms object for the full geometry. This allows the matrix to contain
        atoms without any orbital. Geometry atoms should contain the matrix atoms
        first and then the orbital-less atoms.
    fill_value
        The value to use for the empty elements in the matrix. Use `np.nan` if the
        matrix is not really sparse (empty elements are just elements that you don't
        want to fit) like the density matrix. Models will not attempt to fit the
        `np.nan` values.
    component_indices
        Optional component indices to preserve from ``spmat.data`` before building
        matrix blocks. If one component is selected, blocks use the regular 2D
        representation and labels flatten to 1D.
    component_transform
        Optional function mapping the compacted, component-selected data
        ``(nnz, n_components)`` to ``(nnz, n_new_components)``. It is applied after
        the ``ncol`` compaction and the component selection, so it only sees stored
        values.
    """
    orbitals = atoms.orbitals
    data = spmat.data
    ptr, cols = spmat.ptr, spmat.col
    ncol = getattr(spmat, "ncol", None)
    if ncol is not None:
        # Only the first ncol entries of a row are values (DATA_PIPE-28); keep them, compacted.
        ncol = np.asarray(ncol)
        starts = np.concatenate([[0], np.cumsum(ncol)])
        keep = np.repeat(ptr[:-1] - starts[:-1], ncol) + np.arange(starts[-1])
        ptr = starts.astype(ptr.dtype)
        cols, data = cols[keep], data[keep]

    if component_indices is not None:
        selected_components = np.asarray(component_indices, dtype=np.int64)
        if selected_components.ndim != 1:
            raise ValueError("component_indices must be a 1D sequence of integers.")
        if selected_components.size == 0:
            raise ValueError("component_indices must select at least one component.")
        if selected_components.min() < 0 or selected_components.max() >= data.shape[1]:
            raise ValueError(
                "component_indices contains an out-of-range component index: "
                f"indices={selected_components.tolist()}, n_components={data.shape[1]}."
            )
        data = data[:, selected_components]

    if component_transform is not None:
        data = component_transform(data)

    if data.shape[1] == 1:
        block_dict = _csr_to_block_dict(
            data=data[:, 0],
            ptr=ptr,
            cols=cols,
            atom_first_orb=atoms.firsto,
            orbitals=orbitals,
            n_atoms=len(atoms.species),
            fill_value=fill_value,
        )
    else:
        block_dict = _csr_to_block_dict_components(
            data=data,
            ptr=ptr,
            cols=cols,
            atom_first_orb=atoms.firsto,
            orbitals=orbitals,
            n_atoms=len(atoms.species),
            fill_value=fill_value,
        )

    orbitals = geometry_atoms.orbitals if geometry_atoms is not None else atoms.orbitals

    matrix: BasisMatrix
    if issubclass(matrix_cls, OrbitalMatrix):
        matrix = matrix_cls(block_dict=block_dict, nsc=nsc, orbital_count=orbitals)
    else:
        matrix = matrix_cls(block_dict=block_dict, nsc=nsc, basis_count=orbitals)
    matrix._fill_value = float(fill_value)
    matrix._n_components = data.shape[1]
    return matrix


@converter(Formats.BLOCK_DICT, Formats.SCIPY_COO)
def block_dict_to_coo(
    block_dict: Dict[Tuple[int, int, int], np.ndarray],
    first_orb: np.ndarray,
    n_supercells: int = 1,
    threshold: float = 1e-8,
) -> coo_array:
    """Converts a block dictionary into a coo array.

    Conversions to any other sparse structure can be done once we've got the coo array.
    """
    data: List[Any] = []
    rows = []
    cols = []

    no = first_orb[-1]
    for (i_at, j_at, i_sc), block in block_dict.items():
        flat_block = block.ravel()
        mask = abs(flat_block) > threshold

        data.extend(flat_block[mask])

        i_start = first_orb[i_at]
        j_start = first_orb[j_at]
        i_end = i_start + block.shape[0]
        j_end = j_start + block.shape[1]

        block_rows, block_cols = np.mgrid[i_start:i_end, j_start:j_end].reshape(2, -1)

        block_cols += no * i_sc

        rows.extend(block_rows[mask])
        cols.extend(block_cols[mask])

    return coo_array((data, (rows, cols)), (no, no * n_supercells))


def _blockmatrix_coo_coords(
    orbitals: Union[Sequence[int], np.ndarray],
    edge_index: _ArrayType,
    n_supercells: int = 1,
    edge_neigh_isc: Optional[Union[Sequence[int], np.ndarray]] = None,
    symmetrize_edges: bool = False,
):
    """Returns the coo cordinates of a block matrix.

    This function assumes that:

        - All node blocks contain nonzero entries.
        - Edge blocks for edges included in `edge_index` contain
        nonzero entries.
        - All individual blocks are dense. I.e. if a block "exists",
        there are non-zero entries for all of its elements.

    The order of the coordinates returned by this function is:

        1. Node blocks.
        2. Edge blocks.
        3. Edge blocks in reverse direction (if `symmetrize_edges == True`).

    Blocks (1) and (2) are assumed to be in row-major order. Blocks (3), if
    any, are assumed to contain the data in the exact same order as (2).

    Parameters
    ----------
    orbitals:
        for each atom, the amount of orbitals it has.
    edge_index:
        shape (2, n_edges), for each edge the indices of the atoms
        that participate. If `symmetrize_edges` is `True`, this must
        ONLY contain the edges in one of the directions.
    n_supercells:
        number of supercells in the matrix.
    edge_neigh_isc:
        shape (n_edges, ), for each edge the index of the supercell
        of the interaction. If not provided, all interactions are assumed
        to be in the unit cell.
    symmetrize_edges:
        whether we should assume that the matrix contains also the edges
        that are in the opposite direction as the ones provided in
        `edge_index`.
    """
    orbitals = np.asarray(orbitals, dtype=np.int64)
    edges = np.asarray(edge_index, dtype=np.int64)
    first_orb = np.r_[0, np.cumsum(orbitals)]
    no = first_orb[-1]
    nodes = np.arange(len(orbitals))
    sources = np.r_[nodes, edges[0]]
    targets = np.r_[nodes, edges[1]]
    sizes = orbitals[sources] * orbitals[targets]
    starts = np.cumsum(sizes) - sizes
    local = np.arange(sizes.sum()) - np.repeat(starts, sizes)
    widths = np.repeat(orbitals[targets], sizes)
    rows = np.repeat(first_orb[sources], sizes) + local // widths
    cols = np.repeat(first_orb[targets], sizes) + local % widths

    images = np.zeros(edges.shape[1], dtype=np.int64) if edge_neigh_isc is None else np.asarray(edge_neigh_isc)
    edge_sizes = sizes[len(orbitals):]
    edge_start = int(np.square(orbitals).sum())
    image_offsets = no * np.repeat(images, edge_sizes)
    edge_cols = cols[edge_start:].copy()
    cols[edge_start:] += image_offsets
    if symmetrize_edges:
        # Swap original coordinates, not block traversal: reverse values keep their order.
        opposite = np.where(images == 0, 0, n_supercells - images)
        cols = np.r_[cols, rows[edge_start:] + no * np.repeat(opposite, edge_sizes)]
        rows = np.r_[rows, edge_cols]

    return rows, cols, (no, no * n_supercells)


def _nodes_and_edges_to_coo(
    concatenate: Callable[[Sequence[_ArrayType]], _ArrayType],
    init_coo: Callable[[_ArrayType, np.ndarray, np.ndarray, tuple[int, int]], Any],
    node_vals: _ArrayType,
    edge_vals: _ArrayType,
    edge_index: _ArrayType,
    orbitals: Union[Sequence[int], np.ndarray],
    n_supercells: int = 1,
    edge_neigh_isc: Optional[Union[Sequence[int], np.ndarray]] = None,
    threshold: Optional[float] = None,
    symmetrize_edges: bool = False,
) -> Any:
    """Converts an orbital matrix from node and edges array to coo.

    This is a generic function that can be used to convert to any coo format
    (e.g. scipy coo or torch coo).

    Parameters
    ----------
    concatenate
        function to concatenate the node and edge values to generate the full array
        of values. It receives a list of arrays like:

        - [node_vals, edge_vals, edge_vals] if ``symmetrize_edges`` is ``True``.
        - [node_vals, edge_vals] otherwise.
    init_coo
        function to initialize the coo array. It receives four arguments:

        - The matrix values (as returned by ``concatenate``).
        - The rows corresponding to the matrix values.
        - The columns corresponding to the matrix values.
        - The shape of the matrix.
    node_vals
        Flat array containing the values of the node blocks.
        The order of the values is first by node index, then row then column.
    edge_vals
        Flat array containing the values of the edge blocks.
        The order of the values is first by edge index, then row then column.
    edge_index
        Array of shape (2, n_edges) containing the indices of the atoms
        that participate in each edge.
    orbitals
        Array of shape (n_nodes, ) containing the number of orbitals for each atom.
    n_supercells
        Number of auxiliary supercells.
    edge_neigh_isc
        Array of shape (n_edges, ) containing the supercell index of the second atom
        in each edge with respect to the first atom.
        If not provided, all interactions are assumed to be in the unit cell.
    threshold
        Matrix elements with a value below this number are set to 0.
    symmetrize_edges
        whether for each edge only one direction is provided. The edge block for the
        opposite direction is then created as the transpose.
    """

    rows, cols, shape = _blockmatrix_coo_coords(
        orbitals=orbitals,
        edge_index=edge_index,
        n_supercells=n_supercells,
        edge_neigh_isc=edge_neigh_isc,
        symmetrize_edges=symmetrize_edges,
    )

    if symmetrize_edges:
        sparse_data = concatenate([node_vals, edge_vals, edge_vals])
    else:
        sparse_data = concatenate([node_vals, edge_vals])

    if len(rows) != len(sparse_data):
        orbitals = np.asarray(orbitals)
        edge_block_sizes = orbitals[edge_index[0]] * orbitals[edge_index[1]]
        raise ValueError(
            "Mismatch between COO coordinates and matrix values in "
            "_nodes_and_edges_to_coo: "
            f"len(rows)={len(rows)}, len(sparse_data)={len(sparse_data)}, "
            f"len(node_vals)={len(node_vals)}, len(edge_vals)={len(edge_vals)}, "
            f"n_edges={edge_index.shape[1]}, symmetrize_edges={symmetrize_edges}, "
            f"sum(edge_block_sizes)={int(np.sum(edge_block_sizes))}, "
            f"sum(node_block_sizes)={int(np.sum(np.square(orbitals)))}."
        )

    if threshold is not None:
        mask = abs(sparse_data) > threshold
    else:
        # Remove NaNs
        mask = sparse_data == sparse_data

    sparse_data = sparse_data[mask]
    rows = rows[mask]
    cols = cols[mask]

    return init_coo(sparse_data, rows, cols, shape)


@converter(Formats.NODESEDGES, Formats.SCIPY_COO)
def nodes_and_edges_to_coo(
    node_vals: np.ndarray,
    edge_vals: np.ndarray,
    edge_index: np.ndarray,
    orbitals: np.ndarray,
    n_supercells: int = 1,
    edge_neigh_isc: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    symmetrize_edges: bool = False,
) -> coo_array:
    """Converts an orbital matrix from node and edges array to scipy coo.

    Conversions to any other sparse structure can be done once we've got the coo array.

    Parameters
    ----------
    node_vals
        Flat array containing the values of the node blocks.
        The order of the values is first by node index, then row then column.
    edge_vals
        Flat array containing the values of the edge blocks.
        The order of the values is first by edge index, then row then column.
    edge_index
        Array of shape (2, n_edges) containing the indices of the atoms
        that participate in each edge.
    orbitals
        Array of shape (n_nodes, ) containing the number of orbitals for each atom.
    n_supercells
        Number of auxiliary supercells.
    edge_neigh_isc
        Array of shape (n_edges, ) containing the supercell index of the second atom
        in each edge with respect to the first atom.
        If not provided, all interactions are assumed to be in the unit cell.
    threshold
        Matrix elements with a value below this number are set to 0.
    symmetrize_edges
        whether for each edge only one direction is provided. The edge block for the
        opposite direction is then created as the transpose.
    """

    if np.asarray(node_vals).ndim == 2 or np.asarray(edge_vals).ndim == 2:
        raise ValueError(
            "Multi-component node/edge values cannot be represented in scipy COO/CSR. "
            "Convert directly to a sisl matrix format instead."
        )

    def _init_coo(data, rows, cols, shape):
        return coo_array((data, (rows, cols)), shape)

    return _nodes_and_edges_to_coo(
        concatenate=np.concatenate,
        init_coo=_init_coo,
        node_vals=node_vals,
        edge_vals=edge_vals,
        edge_index=edge_index,
        orbitals=orbitals,
        n_supercells=n_supercells,
        edge_neigh_isc=edge_neigh_isc,
        threshold=threshold,
        symmetrize_edges=symmetrize_edges,
    )


def _spin_orbit_csr_to_sisl(csr, geometry, sp_class) -> SparseOrbital:
    """Orthogonal ``spin-orbit`` Hamiltonian from the 4 TRS channels ``[h0, gx, gy, gz]``."""
    n = len(csr) if isinstance(csr, (list, tuple)) else 1
    if n != 4:
        raise ValueError(f"matrix_component_policy='spin_orbit' requires 4 components [h0, gx, gy, gz], got {n}.")
    if not issubclass(sp_class, sisl.Hamiltonian):
        raise ValueError(f"matrix_component_policy='spin_orbit' requires a sisl.Hamiltonian, got {sp_class.__name__}.")
    # Same channel -> SIESTA component map as ``soc_channels_to_siesta``.
    h0, gx, gy, gz = csr
    return sp_class.fromsp(geometry, [h0, h0, gy, gx, gz, -gz, -gy, gx], spin=sisl.Spin("spin-orbit"))


@converter(Formats.SCIPY_CSR, Formats.SISL)
def csr_to_sisl_sparse_orbital(
    csr: Union[csr_array, Sequence[csr_array]],
    geometry: sisl.Geometry,
    sp_class: Type[SparseOrbital] = SparseOrbital,
    matrix_component_policy: Optional[str] = None,
) -> SparseOrbital:
    """Converts a scipy CSR array to a sisl sparse orbital matrix."""
    if matrix_component_policy == "spin_orbit":
        return _spin_orbit_csr_to_sisl(csr, geometry, sp_class)

    if isinstance(csr, (list, tuple)):
        if len(csr) == 1:
            return sp_class.fromsp(geometry, csr[0])

        if issubclass(sp_class, sisl.Hamiltonian):
            if len(csr) < 2:
                raise ValueError(
                    "Hamiltonian multi-component conversion requires at least two components."
                )

            if matrix_component_policy in ("h_only", "spin_h_only"):
                if len(csr) != 2:
                    raise ValueError(
                        "Hamiltonian spin serialization with "
                        f"matrix_component_policy={matrix_component_policy!r} "
                        "requires exactly two Hamiltonian components."
                    )
                return sp_class.fromsp(
                    geometry,
                    list(csr),
                    spin=sisl.Spin("polarized"),
                )

            if matrix_component_policy == "h_and_overlap":
                if len(csr) == 2:
                    return sp_class.fromsp(geometry, csr[0], S=csr[1])
                if len(csr) == 3:
                    return sp_class.fromsp(
                        geometry,
                        list(csr[:-1]),
                        S=csr[-1],
                        spin=sisl.Spin("polarized"),
                    )
                raise ValueError(
                    "Hamiltonian overlap serialization with "
                    "matrix_component_policy='h_and_overlap' requires either "
                    "two components (H, S) or three components "
                    "(H_up, H_down, S)."
                )

            if matrix_component_policy == "raw_components":
                raise ValueError(
                    "Cannot serialize raw_components multi-component Hamiltonian "
                    "labels as a physical sisl.Hamiltonian. Use 'h_only' or "
                    "'spin_h_only' for spin Hamiltonian channels, or "
                    "'h_and_overlap' when the final component is an explicit "
                    "overlap matrix."
                )

            raise ValueError(
                "Hamiltonian multi-component serialization requires an explicit "
                "matrix_component_policy. Use 'spin_h_only' for two spin "
                "Hamiltonian channels, 'h_and_overlap' for explicit overlap, "
                "or 'raw_components' to preserve labels without writing a "
                "physical Hamiltonian."
            )

        if issubclass(sp_class, (sisl.DensityMatrix, sisl.EnergyDensityMatrix)) and len(csr) == 2:
            return sp_class.fromsp(geometry, list(csr), spin=sisl.Spin("polarized"))

        raise ValueError(
            "Multi-component CSR conversion requires an explicit Hamiltonian policy "
            f"or two polarized DM/EDM channels. Got {sp_class.__name__} with {len(csr)} components."
        )

    return sp_class.fromsp(geometry, csr)


@converter(Formats.NODESEDGES, Formats.SISL)
def nodes_and_edges_to_sparse_orbital(
    node_vals: np.ndarray,
    edge_vals: np.ndarray,
    edge_index: np.ndarray,
    geometry: sisl.Geometry,
    sp_class: Type[SparseOrbital] = SparseOrbital,
    edge_neigh_isc: Optional[np.ndarray] = None,
    threshold: float = 1e-8,
    symmetrize_edges: bool = False,
    matrix_component_policy: Optional[str] = None,
) -> SparseOrbital:
    node_vals = np.asarray(node_vals)
    edge_vals = np.asarray(edge_vals)

    node_components = node_vals.shape[1] if node_vals.ndim == 2 else 1
    edge_components = edge_vals.shape[1] if edge_vals.ndim == 2 else 1
    n_components = max(node_components, edge_components)
    transpose_sign = SOC_TRANSPOSE_SIGN if matrix_component_policy == "spin_orbit" else None

    if n_components == 1 and transpose_sign is None:
        new_csr = conversions.get_converter(Formats.NODESEDGES, Formats.SCIPY_CSR)(
            node_vals=node_vals,
            edge_vals=edge_vals,
            edge_index=edge_index,
            edge_neigh_isc=edge_neigh_isc,
            orbitals=geometry.orbitals,
            n_supercells=geometry.n_s,
            threshold=threshold,
            symmetrize_edges=symmetrize_edges,
        )

        new_csr.indices = new_csr.indices.astype(np.int32)
        new_csr.indptr = new_csr.indptr.astype(np.int32)
        return csr_to_sisl_sparse_orbital(
            new_csr,
            geometry=geometry,
            sp_class=sp_class,
            matrix_component_policy=matrix_component_policy,
        )

    values = _concatenate_nodes_and_edges_multicomponent(
        node_vals=node_vals,
        edge_vals=edge_vals,
        symmetrize_edges=symmetrize_edges,
        transpose_sign=transpose_sign,
    )
    rows, cols, shape = _blockmatrix_coo_coords(
        orbitals=geometry.orbitals,
        edge_index=edge_index,
        n_supercells=geometry.n_s,
        edge_neigh_isc=edge_neigh_isc,
        symmetrize_edges=symmetrize_edges,
    )

    if threshold is not None:
        mask = np.any(np.abs(values) > threshold, axis=1)
    else:
        mask = np.any(values == values, axis=1)

    rows = rows[mask]
    cols = cols[mask]
    values = values[mask]

    csr_components: list[csr_array] = [
        coo_array((values[:, i], (rows, cols)), shape).tocsr()
        for i in range(n_components)
    ]
    for csr in csr_components:
        csr.indices = csr.indices.astype(np.int32)
        csr.indptr = csr.indptr.astype(np.int32)

    return csr_to_sisl_sparse_orbital(
        csr_components,
        geometry=geometry,
        sp_class=sp_class,
        matrix_component_policy=matrix_component_policy,
    )


def _concatenate_nodes_and_edges_multicomponent(
    node_vals: np.ndarray,
    edge_vals: np.ndarray,
    symmetrize_edges: bool,
    transpose_sign: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stacks node and edge values, plus the transposed edge copy if ``symmetrize_edges``.

    ``transpose_sign`` (one factor per component) multiplies the transposed copy, for
    components that change sign under matrix transposition (e.g. spin-orbit ``g``).
    """
    if node_vals.ndim == 1:
        node_vals = node_vals[:, None]
    if edge_vals.ndim == 1:
        edge_vals = edge_vals[:, None]

    if symmetrize_edges:
        transposed = edge_vals if transpose_sign is None else edge_vals * np.asarray(transpose_sign)
        return np.concatenate([node_vals, edge_vals, transposed], axis=0)
    return np.concatenate([node_vals, edge_vals], axis=0)


def _csr_to_block_dict_components(
    data: np.ndarray,
    ptr: np.ndarray,
    cols: np.ndarray,
    atom_first_orb: np.ndarray,
    orbitals: np.ndarray,
    n_atoms: int,
    fill_value: float = 0.0,
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """Converts multi-component SparseCSR data into block_dict.

    Each block has shape ``(n_orb_i, n_orb_j, n_components)``.
    """
    no = atom_first_orb[n_atoms]
    n_components = data.shape[1]
    block_dict: Dict[Tuple[int, int, int], np.ndarray] = {}
    if no == 0:
        return block_dict
    entries = np.arange(ptr[0], ptr[no])
    rows = np.repeat(np.arange(no), np.diff(ptr[:no + 1]))
    columns = cols[entries]
    # Unused sisl capacity terminates a row; later positive columns are not entries.
    stops = ptr[1:no + 1].copy()
    unused = columns < 0
    np.minimum.at(stops, rows[unused], entries[unused])
    valid = entries < stops[rows]
    entries, rows, columns = entries[valid], rows[valid], columns[valid]
    if entries.size == 0:
        return block_dict

    atoms = np.repeat(np.arange(len(orbitals)), orbitals)
    keys = np.column_stack((atoms[rows], atoms[columns % no], columns // no))
    unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    # Stable sorting keeps the last original entry of every repeated coordinate.
    order = np.lexsort((columns, rows, inverse))
    keep = np.r_[(rows[order[1:]] != rows[order[:-1]]) |
                 (columns[order[1:]] != columns[order[:-1]]), True]
    order = order[keep]
    counts = np.bincount(inverse[order], minlength=len(unique))
    groups = np.split(order, np.cumsum(counts)[:-1])
    dtype = np.result_type(data.dtype, np.float64)
    for group in np.argsort(first):
        atom_i, atom_j, image = unique[group]
        indices = groups[group]
        local_rows = rows[indices] - atom_first_orb[atom_i]
        local_cols = columns[indices] % no - atom_first_orb[atom_j]
        positions = local_rows * orbitals[atom_j] + local_cols
        block = np.full((orbitals[atom_i], orbitals[atom_j], n_components), float(fill_value), dtype=dtype)
        block.reshape(-1, n_components)[positions] = data[entries[indices]]
        block_dict[atom_i, atom_j, image] = block
    return block_dict
