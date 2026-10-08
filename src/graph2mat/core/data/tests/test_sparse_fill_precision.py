"""Integer zero fill must not truncate floating CSR labels in the Python path."""

import numpy as np
import pytest

from graph2mat.core.data._sparse import _csr_to_block_dict


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
