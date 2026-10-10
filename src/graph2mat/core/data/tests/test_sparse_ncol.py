"""Roadmap 3.11 (DATA_PIPE-28): only the first ncol entries of each sisl CSR row are
values; the conversion must not rely on the -1 sentinel of the unused capacity."""

import numpy as np
import pytest
import sisl

from graph2mat.core.data.sparse import csr_to_block_dict


@pytest.mark.parametrize("components", [1, 2])
def test_entries_beyond_ncol_are_ignored(components):
    geometry = sisl.geom.graphene()
    H = sisl.Hamiltonian(geometry, dim=components, nnzpr=6)
    H[0, 0] = [1.0] * components
    H[0, 1] = [2.0] * components
    H[1, 1] = [3.0] * components
    csr = H._csr
    # A stale (non-negative) column in row 0's unused capacity, past ncol.
    csr.col[csr.ptr[0] + csr.ncol[0]] = 0
    csr._D[csr.ptr[0] + csr.ncol[0]] = 99.0

    blocks = csr_to_block_dict(csr, geometry.atoms, nsc=geometry.nsc).block_dict
    assert np.all(np.asarray(blocks[0, 0, 0]) == 1.0)
    assert np.all(np.asarray(blocks[0, 1, 0]) == 2.0)
