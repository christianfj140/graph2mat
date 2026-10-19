"""Independent entry-wise oracles for vectorized sparse conversion boundaries."""

import numpy as np
import pytest

from graph2mat.core.data.sparse import _blockmatrix_coo_coords, _csr_to_block_dict_components


def literal_coords(orbitals, edge_index, n_supercells=1, edge_neigh_isc=None, symmetrize_edges=False):
    first = np.r_[0, np.cumsum(orbitals)]
    no = int(first[-1])
    rows, cols, reverse_rows, reverse_cols = [], [], [], []
    for atom, count in enumerate(orbitals):
        for i in range(count):
            for j in range(count):
                rows.append(first[atom] + i)
                cols.append(first[atom] + j)
    images = [0] * edge_index.shape[1] if edge_neigh_isc is None else edge_neigh_isc
    for (source, target), image in zip(edge_index.T, images):
        opposite = 0 if image == 0 else n_supercells - image
        for i in range(orbitals[source]):
            for j in range(orbitals[target]):
                rows.append(first[source] + i)
                cols.append(first[target] + j + no * image)
                if symmetrize_edges:
                    reverse_rows.append(first[target] + j)
                    reverse_cols.append(first[source] + i + no * opposite)
    return np.asarray(rows + reverse_rows, dtype=np.int64), np.asarray(cols + reverse_cols, dtype=np.int64), (no, no * n_supercells)


@pytest.mark.parametrize("orbitals", [[2, 0, 3, 1], [1, 3, 5, 7], [0, 0, 0, 0]])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("images", [None, [0, 1, 2, 0, 2, 1]])
def test_coords_match_literal_rectangular_blocks_ghosts_and_images(orbitals, symmetric, images):
    # Non-contiguous input view; ghost edges retain their place but contribute no entries.
    edges = np.array([[0, 3, 1, 2, 0, 2, 3, 0, 2, 1, 1, 3], [2, 0, 0, 3, 3, 1, 2, 1, 3, 2, 0, 2]])[:, ::2]
    kwargs = dict(orbitals=orbitals, edge_index=edges, n_supercells=3,
                  edge_neigh_isc=images, symmetrize_edges=symmetric)
    actual = _blockmatrix_coo_coords(**kwargs)
    expected = literal_coords(**kwargs)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[0].dtype.kind == actual[1].dtype.kind == "i"
    assert actual[2] == expected[2]


@pytest.mark.parametrize("orbitals", [[], [0], [2, 0]])
@pytest.mark.parametrize("symmetric", [False, True])
def test_coords_without_edges_or_orbitals(orbitals, symmetric):
    kwargs = dict(orbitals=orbitals, edge_index=np.empty((2, 0), dtype=int), symmetrize_edges=symmetric)
    actual, expected = _blockmatrix_coo_coords(**kwargs), literal_coords(**kwargs)
    for a, e in zip(actual[:2], expected[:2]):
        np.testing.assert_array_equal(a, e)
        assert a.dtype.kind == "i"
    assert actual[2] == expected[2]


def test_coords_do_not_construct_a_grid_per_block(monkeypatch):
    class ForbiddenGrid:
        def __getitem__(self, _):
            raise AssertionError("per-block mgrid is the measured bottleneck")

    monkeypatch.setattr(np, "mgrid", ForbiddenGrid())
    rows, cols, shape = _blockmatrix_coo_coords([2, 3], np.array([[0], [1]]), symmetrize_edges=True)
    assert len(rows) == len(cols) == 25
    assert shape == (5, 5)


def literal_component_blocks(data, ptr, cols, atom_first_orb, orbitals, n_atoms, fill_value):
    blocks = {}
    no = int(atom_first_orb[n_atoms])
    for source in range(n_atoms):
        for local_row, row in enumerate(range(atom_first_orb[source], atom_first_orb[source + 1])):
            for index in range(ptr[row], ptr[row + 1]):
                if cols[index] < 0:
                    break
                image, column = divmod(int(cols[index]), no)
                target = next(atom for atom in range(len(orbitals)) if atom_first_orb[atom] <= column < atom_first_orb[atom + 1])
                key = (source, target, image)
                if key not in blocks:
                    blocks[key] = np.full((orbitals[source], orbitals[target], data.shape[1]), float(fill_value), dtype=np.result_type(data.dtype, np.float64))
                blocks[key][local_row, column - atom_first_orb[target]] = data[index]
    return blocks


@pytest.mark.parametrize("dtype", [np.int32, np.float64, np.complex128])
@pytest.mark.parametrize("components", [2, 8])
def test_component_duplicates_are_resolved_globally_not_unique_per_block(monkeypatch, dtype, components):
    orbitals = np.array([2, 1], dtype=np.int32)
    first = np.array([0, 2, 3], dtype=np.int32)
    ptr = np.array([0, 5, 8, 11], dtype=np.int32)
    cols = np.array([4, 0, 4, 2, -1, 1, 2, 1, 0, 5, 5], dtype=np.int32)
    data = np.arange(len(cols)*components).reshape(-1, components).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        data += 1j * data[::-1]
    kwargs = dict(data=data, ptr=ptr, cols=cols, atom_first_orb=first,
                  orbitals=orbitals, n_atoms=2, fill_value=np.nan)
    expected = literal_component_blocks(**kwargs)
    unique = np.unique
    calls = []
    def counted(*args, **options):
        calls.append(None)
        return unique(*args, **options)
    monkeypatch.setattr(np, "unique", counted)
    actual = _csr_to_block_dict_components(**kwargs)
    assert len(calls) == 1, "per-block unique repeats sorting and allocation"
    assert list(actual) == list(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert actual[key].dtype == expected[key].dtype


@pytest.mark.parametrize("fill", [0., np.nan])
@pytest.mark.parametrize("components", [2, 3, 8])
def test_component_grouping_preserves_capacity_break_duplicates_and_dict_order(fill, components):
    orbitals = np.array([2, 0, 3], dtype=np.int32)
    first = np.array([0, 2, 2, 5], dtype=np.int32)
    ptr = np.array([0, 5, 8, 10, 13, 13], dtype=np.int32)
    cols = np.array([12, 1, 12, -1, 3, 7, -1, 0, -1, 4, 0, 4, 4], dtype=np.int32)
    data = np.arange(len(cols) * components, dtype=float).reshape(-1, components) / 8
    data[2, 0] = np.nan
    before = [array.copy() for array in (data, ptr, cols, first, orbitals)]
    kwargs = dict(data=data, ptr=ptr, cols=cols, atom_first_orb=first, orbitals=orbitals,
                  n_atoms=3, fill_value=fill)
    actual, expected = _csr_to_block_dict_components(**kwargs), literal_component_blocks(**kwargs)
    assert list(actual) == list(expected) == [(0, 2, 2), (0, 0, 0), (0, 2, 1), (2, 0, 0), (2, 2, 0)]
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert actual[key].dtype == expected[key].dtype
    for array, original in zip((data, ptr, cols, first, orbitals), before):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("orbitals", [[], [0, 0], [2, 0, 3]])
def test_component_empty_sparse_data_does_not_create_blocks(orbitals):
    orbitals = np.asarray(orbitals, dtype=np.int32)
    first = np.r_[0, np.cumsum(orbitals)].astype(np.int32)
    result = _csr_to_block_dict_components(np.empty((0, 2)), np.zeros(int(first[-1]) + 1, dtype=np.int32),
        np.empty(0, dtype=np.int32), first, orbitals, len(orbitals), np.nan)
    assert result == {}


@pytest.mark.parametrize("seed", range(12))
def test_component_random_sparse_rows_match_entrywise_oracle(seed):
    rng = np.random.default_rng(seed)
    orbitals = np.array([2, 0, 3, 1], dtype=np.int32)
    first = np.r_[0, np.cumsum(orbitals)].astype(np.int32)
    counts = rng.integers(0, 12, int(first[-1]))
    ptr = np.r_[0, np.cumsum(counts)].astype(np.int32)
    cols = rng.integers(-1, 3 * int(first[-1]), int(ptr[-1]), dtype=np.int32)
    data = rng.normal(size=(len(cols), 3))
    data[rng.random(data.shape) < .2] = np.nan
    kwargs = dict(data=data, ptr=ptr, cols=cols, atom_first_orb=first,
                  orbitals=orbitals, n_atoms=4, fill_value=np.nan)
    actual, expected = _csr_to_block_dict_components(**kwargs), literal_component_blocks(**kwargs)
    assert list(actual) == list(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
