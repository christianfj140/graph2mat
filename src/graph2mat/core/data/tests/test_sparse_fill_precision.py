"""Integer zero fill must not truncate floating CSR labels in the Python path."""

import numpy as np
import pytest

from graph2mat.core.data._sparse import _csr_to_block_dict
from graph2mat.core.data.sparse import _csr_to_block_dict_components, csr_to_block_dict


@pytest.mark.parametrize("fill", [0, 1, 0., np.nan])
def test_csr_blocks_preserve_fractional_values_with_integer_fill(fill):
    result = _csr_to_block_dict(
        data=np.array([.125, -.375]), ptr=np.array([0, 1, 2], dtype=np.int32),
        cols=np.array([0, 1], dtype=np.int32), atom_first_orb=np.array([0, 2], dtype=np.int32),
        orbitals=np.array([2], dtype=np.int32), n_atoms=1, fill_value=fill,
    )
    expected = np.array([[.125, fill], [fill, -.375]], dtype=float)
    np.testing.assert_array_equal(result[0, 0, 0], expected)
    assert result[0, 0, 0].dtype.kind == "f"


@pytest.mark.parametrize("fill", [0, 1, 0., np.nan])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_component_blocks_preserve_values_and_missing_fill(fill, dtype):
    values = np.array([[.125, -.375], [.75, .25]], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        values += np.array([[.5j, -.25j], [1j, -2j]], dtype=dtype)
    result = _csr_to_block_dict_components(
        data=values, ptr=np.array([0, 1, 2], dtype=np.int32),
        cols=np.array([0, 1], dtype=np.int32), atom_first_orb=np.array([0, 2], dtype=np.int32),
        orbitals=np.array([2], dtype=np.int32), n_atoms=1, fill_value=fill,
    )
    expected = np.full((2, 2, 2), float(fill), dtype=np.result_type(dtype, np.float64))
    expected[0, 0] = values[0]
    expected[1, 1] = values[1]
    np.testing.assert_array_equal(result[0, 0, 0], expected)
    assert result[0, 0, 0].dtype == expected.dtype


@pytest.mark.parametrize("selected", [None, [1, 0], [1]])
def test_public_component_converter_default_integer_fill_preserves_decimals(selected):
    import sisl
    from scipy.sparse import csr_array

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1), sisl.AtomicOrbital("2s", R=1)])
    geometry = sisl.Geometry([[0., 0., 0.]], atoms=[atom])
    matrix = sisl.DensityMatrix(geometry, spin="polarized")
    matrix._csr = matrix._csr.fromsp([
        csr_array([[.125, 0.], [0., -.375]]), csr_array([[.75, 0.], [0., .25]])])
    before = matrix._csr.data.copy()
    converted = csr_to_block_dict(matrix._csr, geometry.atoms, nsc=[1, 1, 1], component_indices=selected)
    expected = np.array([[[.125, .75], [0., 0.]], [[0., 0.], [-.375, .25]]])
    if selected is not None:
        expected = expected[:, :, selected]
        if len(selected) == 1:
            expected = expected[:, :, 0]
    np.testing.assert_array_equal(converted.block_dict[0, 0, 0], expected)
    np.testing.assert_array_equal(matrix._csr.data, before)
